/*
 * ncps_inference.c -- deployment check for the Neural Circuit Policies
 * (CfC / NCP) artifacts written by `example-ncps`.
 *
 * Loads a ternary-packed model (examples/liquid_sine/modeldata/ncps_{cfc,fc,ncp}_packed.bin) and
 * the held-out val split (examples/liquid_sine/modeldata/ncps_valset.bin), rebuilds the network
 * (backbone -> CfC layer stack -> projection head, exactly the ncps CfC
 * math), and verifies that the packed artifact reaches the val MSE the Zig
 * trainer reported before export. The model file carries the expected MSE
 * in its footer, so the check is self-contained: OK / MISMATCH.
 *
 * Like cnn1d_inference.c, this is a genuinely different deployment path
 * than the trainer's eval: the kernels here consume the packed trits
 * directly (decode 5 trits/byte, then add / skip / subtract per trit -- no
 * per-weight float table). The NCP sparsity pattern rides in the trits
 * themselves: masked synapses are zero trits, so no side-car mask is needed.
 *
 * Usage: ncps-c <model.bin> <valset.bin>
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
 * subtracts it, 0 skips -- one fused multiply-add becomes add/skip/sub. The
 * per-element (trit * scale) matches the Zig trainer's deployed table
 * exactly, so the two engines agree to the last bit of the dot products. */
static inline void ternary_matvec(const int8_t *trits, float scale, const float *x,
                                  int n_in, int n_out, const float *bias, float *out) {
    for (int o = 0; o < n_out; o++) {
        const int8_t *wr = trits + (size_t)o * n_in;
        float s = bias[o];
        for (int j = 0; j < n_in; j++) {
            int8_t t = wr[j];
            if (t > 0) s += scale * x[j];
            else if (t < 0) s -= scale * x[j];
        }
        out[o] = s;
    }
}

#define MAXN 512


static float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

/* One wired/unwired CfC cell step (ncps.torch.CfCCell, mode default). */
static void cfc_step(int mode, const int8_t *t1, float s1, const float *b1,
                     const int8_t *t2, float s2, const float *b2,
                     const int8_t *ta, float sa, const float *ba,
                     const int8_t *tb, float sb, const float *bb,
                     const float *tau, const float *apc,
                     const float *x, const float *h_prev, float ts,
                     int n_sens, int n_out, float *xc, float *h_out) {
    memcpy(xc, x, (size_t)n_sens * sizeof(float));
    memcpy(xc + n_sens, h_prev, (size_t)n_out * sizeof(float));
    if (mode == 1) { /* pure */
        float ff1[256];
        (void)t2; (void)s2; (void)b2; (void)ta; (void)sa; (void)ba; (void)tb; (void)sb; (void)bb;
        ternary_matvec(t1, s1, xc, n_sens + n_out, n_out, b1, ff1);
        for (int i = 0; i < n_out; i++) {
            float e = expf(-ts * (fabsf(tau[i]) + fabsf(ff1[i])));
            h_out[i] = apc[i] - apc[i] * e * ff1[i];
        }
        return;
    }
    float a1[256], a2[256], ta_[256], tb_[256];
    ternary_matvec(t1, s1, xc, n_sens + n_out, n_out, b1, a1);
    for (int i = 0; i < n_out; i++) a1[i] = tanhf(a1[i]);
    ternary_matvec(t2, s2, xc, n_sens + n_out, n_out, b2, a2);
    for (int i = 0; i < n_out; i++) a2[i] = tanhf(a2[i]);
    ternary_matvec(ta, sa, xc, n_sens + n_out, n_out, ba, ta_);
    ternary_matvec(tb, sb, xc, n_sens + n_out, n_out, bb, tb_);
    for (int i = 0; i < n_out; i++) {
        float g = sigmoidf(ta_[i] * ts + tb_[i]);
        h_out[i] = (mode == 0) ? a1[i] * (1.0f - g) + g * a2[i] /* default */
                               : a1[i] + g * a2[i];            /* no_gate */
    }
}

/* One liquid time-constant cell step (ncps.torch.LTCCell): a per-neuron
 * conductance ODE solved with `U` Euler fixed-point iterations. The synapse
 * strengths w/sensory_w arrive as packed trits -- the NCP sparsity pattern
 * rides in the trits (masked synapse == trit 0), so add/skip/sub per trit
 * reproduces the masked activation exactly. The fp32 physical parameters
 * follow the export order: sigma, mu, erev, sensory_sigma, sensory_mu,
 * sensory_erev, gleak, vleak, cm, input_w, input_b, output_w, output_b. */

static void ltc_step(const float *x, const float *state, float *new_state, float *pred,
                     const int8_t *wt, float ws, const int8_t *swt, float sws,
                     const float *sigma, const float *mu, const float *erev,
                     const float *ssig, const float *smu, const float *serev,
                     const float *gleak, const float *vleak, const float *cm,
                     const float *inw, const float *inb, const float *outw, const float *outb,
                     int st, int ns, int motor, int U,
                     float *vhist, float *s_u, float *sens_s, float *xin) {
    /* affine input mapping (the trainer quantizes this under QAT; inference
     * runs float activations, matching the trainer's eval path) */
    for (int j = 0; j < ns; j++) xin[j] = x[j] * inw[j] + inb[j];

    /* sensory synapse terms are loop-invariant across the unfolds */
    float snum[MAXN] = {0}, sden[MAXN] = {0};
    for (int j = 0; j < ns; j++) {
        for (int i = 0; i < st; i++) {
            float s = sigmoidf(ssig[j * st + i] * (xin[j] - smu[j * st + i]));
            sens_s[(size_t)j * st + i] = s;
            int8_t t = swt[(size_t)j * st + i];
            if (t > 0) {
                snum[i] += sws * s * serev[j * st + i];
                sden[i] += sws * s;
            } else if (t < 0) {
                snum[i] -= sws * s * serev[j * st + i];
                sden[i] -= sws * s;
            }
        }
    }

    memcpy(vhist, state, (size_t)st * sizeof(float));
    float cm_t[MAXN];
    for (int i = 0; i < st; i++) cm_t[i] = cm[i] / (1.0f / (float)U);
    for (int u = 0; u < U; u++) {
        const float *v = vhist + (size_t)u * st;
        float *srow = s_u + (size_t)u * st * st;
        float wnum[MAXN] = {0}, wden[MAXN] = {0};
        for (int j = 0; j < st; j++) {
            for (int i = 0; i < st; i++) {
                /* gate[j][i] is driven by the SOURCE voltage v[j] (ncps
                 * broadcasts v_pre over the target axis); act feeds target i */
                float s = sigmoidf(sigma[j * st + i] * (v[j] - mu[j * st + i]));
                srow[(size_t)j * st + i] = s;
                int8_t t = wt[(size_t)j * st + i];
                if (t > 0) {
                    wnum[i] += ws * s * erev[j * st + i];
                    wden[i] += ws * s;
                } else if (t < 0) {
                    wnum[i] -= ws * s * erev[j * st + i];
                    wden[i] -= ws * s;
                }
            }
        }
        for (int i = 0; i < st; i++) {
            float num = cm_t[i] * v[i] + gleak[i] * vleak[i] + wnum[i] + snum[i];
            float den = cm_t[i] + gleak[i] + wden[i] + sden[i];
            vhist[(size_t)(u + 1) * st + i] = num / (den + 1e-8f);
        }
    }
    const float *vout = vhist + (size_t)U * st;
    memcpy(new_state, vout, (size_t)st * sizeof(float));
    for (int i = 0; i < motor; i++) pred[i] = vout[i] * outw[i] + outb[i];
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

/* Packed matrix: (f32 scale, u32 nbytes, bytes). Returns read offset. */
static const uint8_t *load_matrix(const uint8_t *m, size_t len, size_t cur,
                                  int n, int8_t **trits, float *scale) {
    if (cur + 8 > len) die("truncated model");
    memcpy(scale, m + cur, 4); cur += 4;
    uint32_t nb;
    memcpy(&nb, m + cur, 4); cur += 4;
    if (nb != (uint32_t)((n + 4) / 5)) die("bad matrix size");
    if (cur + nb > len) die("truncated model");
    *trits = malloc(n);
    unpack5(*trits, m + cur, n);
    return m + cur + nb;
}


int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] : "examples/liquid_sine/modeldata/ncps_ncp_packed.bin";
    const char *valset_path = argc > 2 ? argv[2] : "examples/liquid_sine/modeldata/ncps_valset.bin";

    /* ---- model ---- */
    size_t mlen;
    uint8_t *m = read_file(model_path, &mlen);
    if (mlen < 36 || memcmp(m, "NCPSQ001", 8) != 0) die("bad model magic");
    uint32_t in_size, units_total, out_dim, n_layers, mode, bb_units, proj_size, cell_kind;
    memcpy(&in_size, m + 8, 4);
    memcpy(&units_total, m + 12, 4);
    memcpy(&out_dim, m + 16, 4);
    memcpy(&n_layers, m + 20, 4);
    memcpy(&mode, m + 24, 4);
    memcpy(&bb_units, m + 28, 4);
    memcpy(&proj_size, m + 32, 4);
    memcpy(&cell_kind, m + 36, 4);
    if (in_size > MAXN || units_total > MAXN || out_dim > MAXN || n_layers > 3 || bb_units > MAXN || proj_size > MAXN)
        die("model dims out of range");
    size_t cur = 40;

    /* liquid cell (cell_kind 1): ternary synapse strengths + 13 fp32 params */
    int8_t *ltc_wt = NULL, *ltc_swt = NULL;
    float ltc_ws = 0, ltc_sws = 0;
    float *ltc_p[13] = {0};
    int ltc_U = 6;
    if (cell_kind == 1) {
        cur = (size_t)(load_matrix(m, mlen, cur, (int)(units_total * units_total), &ltc_wt, &ltc_ws) - m);
        cur = (size_t)(load_matrix(m, mlen, cur, (int)(in_size * units_total), &ltc_swt, &ltc_sws) - m);
        static const size_t kinds[13] = { 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5 };
        /* kinds: 1 = units*units, 2 = in*units, 3 = units, 4 = in, 5 = out */
        for (int k = 0; k < 13; k++) {
            size_t n = kinds[k] == 1 ? (size_t)units_total * units_total
                     : kinds[k] == 2 ? (size_t)in_size * units_total
                     : kinds[k] == 3 ? (size_t)units_total
                     : kinds[k] == 4 ? (size_t)in_size
                                     : (size_t)out_dim;
            if (cur + n * 4 > mlen) die("truncated model");
            ltc_p[k] = malloc(n * sizeof(float));
            memcpy(ltc_p[k], m + cur, n * 4);
            cur += n * 4;
        }
    }

    /* backbone (unwired models): [bb x (in + units)] over concat([x, h]) */
    int8_t *bb_t = NULL;
    float bb_scale = 0, bb_bias[MAXN] = {0};
    if (bb_units > 0) {
        cur = (size_t)(load_matrix(m, mlen, cur, (int)(bb_units * (in_size + units_total)), &bb_t, &bb_scale) - m);
        if (cur + bb_units * 4 > mlen) die("truncated model");
        memcpy(bb_bias, m + cur, bb_units * 4); cur += bb_units * 4;
    }

    /* per-layer CfC projections */
    int8_t *t1[3] = {0}, *t2[3] = {0}, *ta[3] = {0}, *tb[3] = {0};
    float s1[3], s2[3], sa[3], sb[3];
    float b1[3][MAXN] = {{0}}, b2[3][MAXN] = {{0}}, ba_[3][MAXN] = {{0}}, bbb[3][MAXN] = {{0}};
    float tau[3][MAXN] = {{0}}, apc[3][MAXN] = {{0}};
    uint32_t n_sens[3], n_out[3];
    for (uint32_t l = 0; l < n_layers; l++) {
        if (cur + 8 > mlen) die("truncated model");
        memcpy(&n_sens[l], m + cur, 4); cur += 4;
        memcpy(&n_out[l], m + cur, 4); cur += 4;
        if (n_sens[l] > MAXN || n_out[l] > MAXN) die("layer dims out of range");
        int cat = (int)(n_sens[l] + n_out[l]);
        cur = (size_t)(load_matrix(m, mlen, cur, (int)(n_out[l] * cat), &t1[l], &s1[l]) - m);
        if (cur + n_out[l] * 4 > mlen) die("truncated model");
        memcpy(b1[l], m + cur, n_out[l] * 4); cur += n_out[l] * 4;
        if (mode != 1) {
            cur = (size_t)(load_matrix(m, mlen, cur, (int)(n_out[l] * cat), &t2[l], &s2[l]) - m);
            memcpy(b2[l], m + cur, n_out[l] * 4); cur += n_out[l] * 4;
            cur = (size_t)(load_matrix(m, mlen, cur, (int)(n_out[l] * cat), &ta[l], &sa[l]) - m);
            memcpy(ba_[l], m + cur, n_out[l] * 4); cur += n_out[l] * 4;
            cur = (size_t)(load_matrix(m, mlen, cur, (int)(n_out[l] * cat), &tb[l], &sb[l]) - m);
            memcpy(bbb[l], m + cur, n_out[l] * 4); cur += n_out[l] * 4;
        } else {
            if (cur + n_out[l] * 8 > mlen) die("truncated model");
            memcpy(tau[l], m + cur, n_out[l] * 4); cur += n_out[l] * 4;
            memcpy(apc[l], m + cur, n_out[l] * 4); cur += n_out[l] * 4;
        }
    }

    /* projection head (unwired models): [proj x units] */
    int8_t *pj_t = NULL;
    float pj_scale = 0, pj_bias[MAXN] = {0};
    if (proj_size > 0) {
        cur = (size_t)(load_matrix(m, mlen, cur, (int)(proj_size * units_total), &pj_t, &pj_scale) - m);
        if (cur + proj_size * 4 > mlen) die("truncated model");
        memcpy(pj_bias, m + cur, proj_size * 4); cur += proj_size * 4;
    }
    double expected_mse;
    if (cur + 8 > mlen) die("truncated model");
    memcpy(&expected_mse, m + cur, 8);
    free(m);

    /* ---- val split ---- */
    size_t vlen;
    uint8_t *v = read_file(valset_path, &vlen);
    if (vlen < 24 || memcmp(v, "NCPSV001", 8) != 0) die("bad valset magic");
    uint32_t n_val, seq_len, v_in, v_out;
    memcpy(&n_val, v + 8, 4);
    memcpy(&seq_len, v + 12, 4);
    memcpy(&v_in, v + 16, 4);
    memcpy(&v_out, v + 20, 4);
    if (v_in != in_size || v_out != out_dim) die("valset shape mismatch");
    if (24 + (size_t)n_val * seq_len * (v_in + v_out) * 4 != vlen) die("valset size mismatch");
    const float *xs = (const float *)(v + 24);
    const float *ys = (const float *)(v + 24 + (size_t)n_val * seq_len * v_in * 4);

    printf("NCPSQ001 deployment check: in %u, units %u, out %u, layers %u, cell %s, mode %s%s%s%s\n",
           in_size, units_total, out_dim, n_layers,
           cell_kind == 1 ? "LTC (liquid ODE)" : "CfC",
           mode == 0 ? "default" : (mode == 1 ? "pure" : "no_gate"),
           bb_units ? ", backbone" : "", proj_size ? ", proj head" : "",
           n_layers > 1 ? ", NCP wiring (sparsity in trits)" : "");

    /* ---- evaluate ---- */
    float state[MAXN] = {0}, h0[MAXN] = {0}, bb_out[MAXN] = {0}, xc[MAXN] = {0};
    float pred[MAXN] = {0}, xbb[MAXN] = {0};
    /* state layout = concat of layer outputs in processing order (inter,
     * command, motor for an NCP), so each layer owns a slice of the state */
    uint32_t loff[3] = {0, 0, 0};
    for (uint32_t l = 1; l < n_layers; l++) loff[l] = loff[l - 1] + n_out[l - 1];
    /* LTC scratch: state through the unfolds, per-unfold sigmoids, sensory
     * sigmoids, mapped input */
    static float vh[MAXN * (6 + 1)], su[MAXN * 6 * MAXN], ssc[MAXN * MAXN], xin[MAXN];
    double loss_sum = 0;
    const int reps = 3;
    double t0 = now_s();
    for (int rep = 0; rep < reps; rep++) {
        for (uint32_t si = 0; si < n_val; si++) {
            const float *x = xs + (size_t)si * seq_len * v_in;
            const float *y = ys + (size_t)si * seq_len * v_out;
            memset(state, 0, sizeof(state));
            double seq_loss = 0;
            for (uint32_t t = 0; t < seq_len; t++) {
                const float *xt = x + (size_t)t * v_in;
                if (cell_kind == 1) {
                    ltc_step(xt, state, state, pred,
                             ltc_wt, ltc_ws, ltc_swt, ltc_sws,
                             ltc_p[0], ltc_p[1], ltc_p[2], ltc_p[3], ltc_p[4], ltc_p[5],
                             ltc_p[6], ltc_p[7], ltc_p[8], ltc_p[9], ltc_p[10], ltc_p[11], ltc_p[12],
                             (int)units_total, (int)in_size, (int)out_dim, ltc_U,
                             vh, su, ssc, xin);
                } else {
                const float *layer_in = xt;
                if (bb_units > 0) {
                    memcpy(xbb, xt, in_size * sizeof(float));
                    memcpy(xbb + in_size, state, units_total * sizeof(float));
                    ternary_matvec(bb_t, bb_scale, xbb, (int)(in_size + units_total), (int)bb_units, bb_bias, bb_out);
                    for (uint32_t i = 0; i < bb_units; i++)
                        bb_out[i] = 1.7159f * tanhf(0.666f * bb_out[i]);
                    layer_in = bb_out;
                }
                for (uint32_t l = 0; l < n_layers; l++) {
                    cfc_step((int)mode, t1[l], s1[l], b1[l], t2[l], s2[l], b2[l],
                             ta[l], sa[l], ba_[l], tb[l], sb[l], bbb[l],
                             tau[l], apc[l], layer_in, state + loff[l], 1.0f,
                             (int)n_sens[l], (int)n_out[l], xc, h0);
                    /* layer l's output becomes layer l+1's input and the new
                     * state slice; the readout is the motor slice */
                    memcpy(state + loff[l], h0, n_out[l] * sizeof(float));
                    layer_in = h0;
                }
                if (proj_size > 0) {
                    ternary_matvec(pj_t, pj_scale, h0, (int)units_total, (int)proj_size, pj_bias, pred);
                } else {
                    memcpy(pred, state + loff[n_layers - 1], out_dim * sizeof(float));
                }
                }
                if (rep == 0) {
                    for (uint32_t i = 0; i < out_dim; i++) {
                        double d = (double)pred[i] - y[t * v_out + i];
                        seq_loss += d * d;
                    }
                }
            }
            if (rep == 0) loss_sum += seq_loss / (seq_len * v_out);
        }
    }
    double elapsed = now_s() - t0;
    free(v);

    double mse = loss_sum / n_val;
    double us = elapsed / ((double)reps * n_val) * 1e6;
    printf("  val MSE  : %.6f  (trainer reported %.6f)\n", mse, expected_mse);
    printf("  status   : %s\n", fabs(mse - expected_mse) <= 1e-4 ? "OK (matches the Zig trainer)" : "MISMATCH");
    printf("  latency  : %.2f us/sequence (ternary trit kernels, %d-step sequences)\n", us, (int)seq_len);

    for (int l = 0; l < 3; l++) { free(t1[l]); free(t2[l]); free(ta[l]); free(tb[l]); }
    free(bb_t); free(pj_t);
    if (cell_kind == 1) {
        free(ltc_wt); free(ltc_swt);
        for (int k = 0; k < 13; k++) free(ltc_p[k]);
    }
    return fabs(mse - expected_mse) <= 1e-4 ? 0 : 1;
}
