/*
 * cnn1d_inference.c -- deployment check for the TinyML 1D-CNN artifact.
 *
 * Loads the ternary-packed model (examples/gesture_cnn/modeldata/cnn1d_packed.bin) and the held-out
 * val split (examples/gesture_cnn/modeldata/cnn1d_valset.bin) written by `example-cnn`, rebuilds
 * the network, and verifies that the packed artifact reaches the same val
 * accuracy as the Zig trainer reported before export.
 *
 * This is a genuinely different deployment path than the trainer's eval: the
 * kernels here consume the packed trits directly (decode 5 trits/byte, then
 * add / skip / subtract per trit -- no per-weight float table), which is how
 * BitNetMCU-style ternary inference runs on MCUs.
 *
 * Usage: cnn1d-c <model.bin> <valset.bin>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <time.h>
#endif

#define C0 3
#define T0 128
#define N_CLASS 4
#define C1 8
#define K1 8
#define T1 (T0 - K1 + 1)
#define P1 (T1 / 2)
#define C2 16
#define K2 4
#define T2 (P1 - K2 + 1)
#define P2 (T2 / 2)
#define FC_IN (C2 * P2)

static void die(const char *msg) {
    fprintf(stderr, "error: %s\n", msg);
    exit(1);
}

static uint8_t *read_file(const char *path, size_t *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "error: cannot open %s\n", path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = malloc((size_t)len);
    if (!buf) die("out of memory");
    if (fread(buf, 1, (size_t)len, f) != (size_t)len) die("short read");
    fclose(f);
    *len_out = (size_t)len;
    return buf;
}

/* Same division-free decode as src/framework/ternary.zig: multiply by 3, take the top
 * byte as the next trit, mask, repeat. */
static void unpack5(int8_t *dst, const uint8_t *src, size_t n) {
    size_t i = 0, g = 0;
    while (i < n) {
        unsigned b = src[g++];
        for (int k = 0; k < 5 && i < n; k++) {
            b *= 3;
            unsigned t = b >> 8;
            b &= 0xFF;
            dst[i++] = (int8_t)t - 1;
        }
    }
}

/* Ternary dot product straight off the trits: +1 adds the input, -1
 * subtracts it, 0 skips -- one fused multiply-add becomes add/skip/sub. */
static inline float ternary_dot(const int8_t *trits, const float *x, int n, float scale) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) {
        int8_t t = trits[i];
        if (t > 0) s += x[i];
        else if (t < 0) s -= x[i];
    }
    return s * scale;
}

/* Valid 1D conv with ternary kernels. x: [cin][tin], trits: [cout][cin*k]. */
static void conv1d_tern(const float *x, const int8_t *trits, float scale, const float *b,
                        int cin, int tin, int cout, int k, float *out) {
    int tout = tin - k + 1;
    for (int o = 0; o < cout; o++) {
        const int8_t *wr = trits + o * cin * k;
        for (int t = 0; t < tout; t++) {
            float s = b[o];
            for (int c = 0; c < cin; c++)
                s += ternary_dot(wr + c * k, x + c * tin + t, k, scale);
            out[o * tout + t] = s > 0 ? s : 0; /* fused ReLU */
        }
    }
}

static void maxpool2(const float *x, int cin, int tin, float *out) {
    int tout = tin / 2;
    for (int c = 0; c < cin; c++)
        for (int t = 0; t < tout; t++) {
            float a = x[c * tin + 2 * t];
            float b = x[c * tin + 2 * t + 1];
            out[c * tout + t] = b > a ? b : a;
        }
}

static double now_s(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] : "examples/gesture_cnn/modeldata/cnn1d_packed.bin";
    const char *valset_path = argc > 2 ? argv[2] : "examples/gesture_cnn/modeldata/cnn1d_valset.bin";

    /* ---- model ---- */
    size_t mlen;
    uint8_t *m = read_file(model_path, &mlen);
    if (mlen < 48 || memcmp(m, "CNN1Q002", 8) != 0) die("bad model magic");
    uint32_t dims[7];
    memcpy(dims, m + 8, sizeof(dims));
    if (dims[0] != C0 || dims[1] != T0 || dims[2] != C1 || dims[3] != K1 ||
        dims[4] != C2 || dims[5] != K2 || dims[6] != N_CLASS)
        die("model shape mismatch");
    size_t cur = 8 + sizeof(dims);

    int8_t *t1 = malloc(C1 * C0 * K1);
    int8_t *t2 = malloc(C2 * C1 * K2);
    int8_t *t3 = malloc(N_CLASS * FC_IN);
    float s1, s2, s3;
    uint32_t nb;
    memcpy(&s1, m + cur, 4); cur += 4;
    memcpy(&nb, m + cur, 4); cur += 4;
    if (nb != (C1 * C0 * K1 + 4) / 5) die("bad w1 size");
    unpack5(t1, m + cur, C1 * C0 * K1); cur += nb;
    memcpy(&s2, m + cur, 4); cur += 4;
    memcpy(&nb, m + cur, 4); cur += 4;
    if (nb != (C2 * C1 * K2 + 4) / 5) die("bad w2 size");
    unpack5(t2, m + cur, C2 * C1 * K2); cur += nb;
    memcpy(&s3, m + cur, 4); cur += 4;
    memcpy(&nb, m + cur, 4); cur += 4;
    if (nb != (N_CLASS * FC_IN + 4) / 5) die("bad w3 size");
    unpack5(t3, m + cur, N_CLASS * FC_IN); cur += nb;

    float b1[C1], b2[C2], b3[N_CLASS];
    memcpy(b1, m + cur, sizeof(b1)); cur += sizeof(b1);
    memcpy(b2, m + cur, sizeof(b2)); cur += sizeof(b2);
    memcpy(b3, m + cur, sizeof(b3)); cur += sizeof(b3);
    /* footer: the val accuracy the Zig trainer measured before export */
    double expected_acc;
    memcpy(&expected_acc, m + cur, sizeof(expected_acc)); cur += sizeof(expected_acc);
    if (cur != mlen) die("trailing bytes in model file");
    free(m);
    printf("model %s: %d trits packed, scales %.5f / %.5f / %.5f\n",
           model_path, C1 * C0 * K1 + C2 * C1 * K2 + N_CLASS * FC_IN, s1, s2, s3);

    /* ---- val split ---- */
    size_t vlen;
    uint8_t *v = read_file(valset_path, &vlen);
    if (vlen < 24 || memcmp(v, "CNN1V001", 8) != 0) die("bad valset magic");
    uint32_t vdims[4];
    memcpy(vdims, v + 8, sizeof(vdims));
    uint32_t n_val = vdims[0];
    if (vdims[1] != C0 || vdims[2] != T0 || vdims[3] != N_CLASS) die("valset shape mismatch");
    if (24 + n_val * (C0 * T0 * 4 + 1) != vlen) die("valset size mismatch");
    const float *xs = (const float *)(v + 24);
    const uint8_t *ys = v + 24 + (size_t)n_val * C0 * T0 * 4;

    /* ---- evaluate ---- */
    static float c1[C1 * T1], p1[C1 * P1], c2[C2 * T2], p2[C2 * P2], logits[N_CLASS];
    int confusion[N_CLASS][N_CLASS] = {{0}};
    int correct = 0;
    double loss_sum = 0;

    double t0 = now_s();
    const int reps = 100;
    for (int rep = 0; rep < reps; rep++) {
        for (uint32_t i = 0; i < n_val; i++) {
            const float *x = xs + (size_t)i * C0 * T0;
            conv1d_tern(x, t1, s1, b1, C0, T0, C1, K1, c1);
            maxpool2(c1, C1, T1, p1);
            conv1d_tern(p1, t2, s2, b2, C1, P1, C2, K2, c2);
            maxpool2(c2, C2, T2, p2);
            for (int o = 0; o < N_CLASS; o++)
                logits[o] = ternary_dot(t3 + o * FC_IN, p2, FC_IN, s3) + b3[o];

            int pred = 0;
            for (int c = 1; c < N_CLASS; c++)
                if (logits[c] > logits[pred]) pred = c;
            if (rep == 0) {
                if (pred == ys[i]) correct++;
                confusion[ys[i]][pred]++;
                double mx = logits[0];
                for (int c = 1; c < N_CLASS; c++) if (logits[c] > mx) mx = logits[c];
                double se = 0;
                for (int c = 0; c < N_CLASS; c++) se += exp((double)logits[c] - mx);
                loss_sum += -(logits[ys[i]] - mx) + log(se);
            }
        }
    }
    double elapsed = now_s() - t0;
    free(v);

    static const char *names[N_CLASS] = { "rest", "shake", "circle", "tap" };
    printf("\nCNN1Q002 deployment check (%d val samples)\n", (int)n_val);
    double acc = (double)correct / (double)n_val;
    printf("  val accuracy : %.2f%%  (loss %.4f)\n", 100.0 * acc, loss_sum / n_val);
    printf("  footer       : %.2f%%  (the trainer's number, stored in the artifact)\n", 100.0 * expected_acc);
    printf("  status       : %s\n", fabs(acc - expected_acc) <= 1e-12 ? "OK (matches the Zig trainer)" : "MISMATCH");
    printf("  confusion matrix (rows = true, cols = pred)\n");
    printf("            rest  shake circle   tap | recall\n");
    for (int tc = 0; tc < N_CLASS; tc++) {
        int row_total = 0;
        for (int pc = 0; pc < N_CLASS; pc++) row_total += confusion[tc][pc];
        printf("  %6s", names[tc]);
        for (int pc = 0; pc < N_CLASS; pc++) printf(" %6d", confusion[tc][pc]);
        printf(" | %5.1f%%\n", 100.0 * confusion[tc][tc] / row_total);
    }
    double us = elapsed / ((double)reps * n_val) * 1e6;
    printf("  latency      : %.2f us/inference (%.1fM MACs/s, ternary trit kernels)\n",
           us, 54208.0 / us);

    free(t1); free(t2); free(t3);
    return fabs(acc - expected_acc) <= 1e-12 ? 0 : 1;
}
