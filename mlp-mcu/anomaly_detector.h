/*
 * anomaly_detector.h — Anomaly detection application layer
 *
 * Binary MLP classifier for anomaly detection:
 *   Input:  feature vector (sensor statistics, signal features)
 *   Output: single sigmoid value ∈ [0, 1]
 *           0.0 = normal, 1.0 = anomaly
 *
 * Threshold-based decision with configurable sensitivity.
 * Optional running statistics for adaptive thresholding.
 */

#ifndef ANOMALY_DETECTOR_H
#define ANOMALY_DETECTOR_H

#include "mlp_core.h"

/* ================================================================== */
/* Anomaly detection result                                            */
/* ================================================================== */

typedef struct {
  float anomaly_score;   /* raw sigmoid output [0, 1] */
  int   is_anomaly;      /* 1 if score > threshold, 0 otherwise */
  float threshold;       /* decision threshold used */
  float confidence;      /* |score - 0.5| * 2: how certain the decision is [0, 1] */
} ad_result_t;

/* ================================================================== */
/* Adaptive threshold (running mean + std of anomaly scores)           */
/* ================================================================== */

typedef struct {
  float running_mean;
  float running_std;
  float alpha;           /* exponential moving average decay (0.001-0.01 typical) */
  float n_sigma;         /* threshold = mean + n_sigma * std */
  float min_threshold;   /* minimum threshold floor */
  float max_threshold;   /* maximum threshold ceiling */
  int   initialized;
  float warmup_sum;
  float warmup_sum_sq;
  int   warmup_count;
  int   warmup_n;        /* number of samples before adaptive threshold activates */
} ad_threshold_t;

static inline void ad_threshold_init(ad_threshold_t *t,
                                      float alpha, float n_sigma,
                                      float min_thresh, float max_thresh,
                                      int warmup_n) {
  t->running_mean    = 0.5f;
  t->running_std     = 0.1f;
  t->alpha           = alpha;
  t->n_sigma         = n_sigma;
  t->min_threshold   = min_thresh;
  t->max_threshold   = max_thresh;
  t->initialized     = 0;
  t->warmup_sum      = 0.0f;
  t->warmup_sum_sq   = 0.0f;
  t->warmup_count    = 0;
  t->warmup_n        = warmup_n;
}

static inline float ad_threshold_get(ad_threshold_t *t) {
  float thresh = t->running_mean + t->n_sigma * t->running_std;
  if (thresh < t->min_threshold) thresh = t->min_threshold;
  if (thresh > t->max_threshold) thresh = t->max_threshold;
  return thresh;
}

static inline void ad_threshold_update(ad_threshold_t *t, float score) {
  if (!t->initialized) {
    t->warmup_sum    += score;
    t->warmup_sum_sq += score * score;
    t->warmup_count++;
    if (t->warmup_count >= t->warmup_n) {
      t->running_mean = t->warmup_sum / (float)t->warmup_count;
      float var = t->warmup_sum_sq / (float)t->warmup_count - t->running_mean * t->running_mean;
      t->running_std = sqrtf(var > 1e-10f ? var : 1e-4f);
      t->initialized = 1;
    }
    return;
  }

  /* Exponential moving average update */
  float diff = score - t->running_mean;
  t->running_mean += t->alpha * diff;
  t->running_std  += t->alpha * (fabsf(diff) - t->running_std);
  if (t->running_std < 1e-4f) t->running_std = 1e-4f;
}

/* ================================================================== */
/* Core anomaly detection                                              */
/* ================================================================== */

/* Fixed-threshold anomaly detection */
static inline ad_result_t ad_detect(const float *input_features, float threshold) {
  ad_result_t r;
  float logit;
  mlp_forward_f32(input_features, &logit);

  r.anomaly_score = MLP_SIGMOIDF(logit);
  r.threshold     = threshold;
  r.is_anomaly    = (r.anomaly_score > threshold) ? 1 : 0;
  r.confidence    = fabsf(r.anomaly_score - 0.5f) * 2.0f;
  return r;
}

/* Adaptive-threshold anomaly detection */
static inline ad_result_t ad_detect_adaptive(const float *input_features,
                                              ad_threshold_t *t) {
  ad_result_t r;
  float logit;
  mlp_forward_f32(input_features, &logit);

  r.anomaly_score = MLP_SIGMOIDF(logit);
  r.threshold     = ad_threshold_get(t);
  r.is_anomaly    = (r.anomaly_score > r.threshold) ? 1 : 0;
  r.confidence    = fabsf(r.anomaly_score - 0.5f) * 2.0f;

  /* Update running statistics */
  ad_threshold_update(t, r.anomaly_score);

  return r;
}

/* ================================================================== */
/* Sliding window anomaly detection                                    */
/* ================================================================== */

#ifndef AD_WINDOW_SIZE
  #define AD_WINDOW_SIZE 16
#endif

typedef struct {
  float scores[AD_WINDOW_SIZE];
  int   head;
  int   count;
  float window_mean;
  float window_max;
} ad_window_t;

static inline void ad_window_init(ad_window_t *w) {
  w->head = 0;
  w->count = 0;
  w->window_mean = 0.0f;
  w->window_max = 0.0f;
  for (int i = 0; i < AD_WINDOW_SIZE; i++) w->scores[i] = 0.0f;
}

static inline void ad_window_push(ad_window_t *w, float score) {
  w->scores[w->head] = score;
  w->head = (w->head + 1) % AD_WINDOW_SIZE;
  if (w->count < AD_WINDOW_SIZE) w->count++;

  /* Recompute statistics */
  float sum = 0.0f;
  float mx  = 0.0f;
  for (int i = 0; i < w->count; i++) {
    sum += w->scores[i];
    if (w->scores[i] > mx) mx = w->scores[i];
  }
  w->window_mean = sum / (float)w->count;
  w->window_max  = mx;
}

/* Windowed anomaly: flag if window mean OR max exceeds threshold */
static inline ad_result_t ad_detect_windowed(const float *input_features,
                                              ad_window_t *w,
                                              float mean_threshold,
                                              float max_threshold) {
  ad_result_t r;
  float logit;
  mlp_forward_f32(input_features, &logit);

  r.anomaly_score = MLP_SIGMOIDF(logit);
  ad_window_push(w, r.anomaly_score);

  r.is_anomaly = (w->window_mean > mean_threshold || w->window_max > max_threshold) ? 1 : 0;
  r.threshold  = mean_threshold;
  r.confidence = fabsf(r.anomaly_score - 0.5f) * 2.0f;
  return r;
}

/* ================================================================== */
/* Online learning: score threshold from positive examples             */
/* ================================================================== */

/*
 * If labeled normal data is available, compute threshold as:
 *   threshold = mean(normal_scores) + k * std(normal_scores)
 *
 * ad_compute_threshold_from_labels:
 *   scores[]: anomaly scores for labeled normal data
 *   n: number of samples
 *   k: number of standard deviations (typically 2-3)
 */
static inline float ad_compute_threshold(const float *scores, int n, float k) {
  float sum = 0.0f, sum_sq = 0.0f;
  for (int i = 0; i < n; i++) {
    sum    += scores[i];
    sum_sq += scores[i] * scores[i];
  }
  float mean = sum / (float)n;
  float var  = sum_sq / (float)n - mean * mean;
  float std  = sqrtf(var > 0.0f ? var : 1e-8f);
  return mean + k * std;
}

#endif /* ANOMALY_DETECTOR_H */
