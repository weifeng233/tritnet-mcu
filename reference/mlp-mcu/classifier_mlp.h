/*
 * classifier_mlp.h — Classification application layer
 *
 * Feature extraction → MLP → Softmax/Argmax → class label + confidence
 *
 * Supports:
 *   - Multi-class classification (softmax output)
 *   - Binary classification (sigmoid output)
 *   - Multi-label classification (independent sigmoid per class)
 *
 * Typical pipeline:
 *   1. Collect sensor data (raw ADC, IMU, etc.)
 *   2. Extract features (statistical, frequency domain, etc.)
 *   3. Normalize features
 *   4. MLP forward pass
 *   5. Softmax / Argmax → predicted class + confidence
 */

#ifndef CLASSIFIER_MLP_H
#define CLASSIFIER_MLP_H

#include "mlp_core.h"

#ifndef CLF_MAX_FEATURES
  #define CLF_MAX_FEATURES 64
#endif

#ifndef CLF_MAX_CLASSES
  #define CLF_MAX_CLASSES  32
#endif

/* ================================================================== */
/* Classifier result                                                   */
/* ================================================================== */

typedef struct {
  int   class_id;                  /* predicted class index (0-based) */
  float confidence;                /* probability of predicted class */
  float probabilities[CLF_MAX_CLASSES]; /* full probability vector */
  int   n_classes;                 /* number of output classes */
} clf_result_t;

/* ================================================================== */
/* Feature normalization (z-score)                                     */
/* ================================================================== */

/*
 * Feature normalization: x_norm = (x - mean) / std
 * mean and std are model-specific, embedded in model.h
 *
 * model.h should define:
 *   const float clf_input_mean[CLF_INPUT_DIM];
 *   const float clf_input_std[CLF_INPUT_DIM];
 */

/* Normalize features in-place (z-score) */
static inline void clf_normalize_zscore(float *features, int n,
                                         const float *mean, const float *std) {
  for (int i = 0; i < n; i++)
    features[i] = (features[i] - mean[i]) / (std[i] + 1e-8f);
}

/* Normalize features to [0, 1] range (min-max) */
static inline void clf_normalize_minmax(float *features, int n,
                                         const float *min_val, const float *max_val) {
  for (int i = 0; i < n; i++) {
    float range = max_val[i] - min_val[i];
    features[i] = (range > 1e-8f) ? ((features[i] - min_val[i]) / range) : 0.0f;
  }
}

/* ================================================================== */
/* Softmax (standalone, using platform primitives)                     */
/* ================================================================== */

static inline void clf_softmax(const float *logits, int n, float *probs) {
  MLP_SOFTMAX_FWD(logits, n, probs);
}

/* ================================================================== */
/* Prediction functions                                                */
/* ================================================================== */

/* Multi-class classification: softmax + argmax */
static inline void clf_predict(const float *input_features, int n_classes,
                                clf_result_t *result) {
  float logits[CLF_MAX_CLASSES];
  mlp_forward_f32(input_features, logits);

  clf_softmax(logits, n_classes, result->probabilities);
  result->n_classes = n_classes;

  /* Argmax */
  result->class_id   = 0;
  result->confidence = result->probabilities[0];
  for (int i = 1; i < n_classes; i++) {
    if (result->probabilities[i] > result->confidence) {
      result->confidence = result->probabilities[i];
      result->class_id   = i;
    }
  }
}

/* Binary classification: single sigmoid output */
static inline float clf_predict_binary(const float *input_features) {
  float logit;
  mlp_forward_f32(input_features, &logit);
  return MLP_SIGMOIDF(logit);
}

/* Multi-label classification: independent sigmoid per class */
static inline void clf_predict_multilabel(const float *input_features, int n_classes,
                                           float *probabilities) {
  float logits[CLF_MAX_CLASSES];
  mlp_forward_f32(input_features, logits);

  for (int i = 0; i < n_classes; i++)
    probabilities[i] = MLP_SIGMOIDF(logits[i]);
}

/* ================================================================== */
/* Common feature extraction utilities                                 */
/* ================================================================== */

/* Statistical features from a signal buffer */
typedef struct {
  float mean;
  float variance;
  float rms;
  float min_val;
  float max_val;
  float peak_to_peak;
  float skewness;       /* third moment / variance^1.5 */
  float kurtosis;       /* fourth moment / variance^2 - 3 */
  float crest_factor;   /* peak / rms */
  float zero_crossings; /* number of sign changes */
} clf_signal_features_t;

static inline void clf_extract_signal_features(const float *signal, int n,
                                                clf_signal_features_t *f) {
  /* Mean */
  float sum = 0.0f;
  for (int i = 0; i < n; i++) sum += signal[i];
  f->mean = sum / (float)n;

  /* Variance, min, max, sum of squares */
  float var_sum = 0.0f, sum_sq = 0.0f;
  f->min_val = signal[0];
  f->max_val = signal[0];
  for (int i = 0; i < n; i++) {
    float d = signal[i] - f->mean;
    var_sum += d * d;
    sum_sq += signal[i] * signal[i];
    if (signal[i] < f->min_val) f->min_val = signal[i];
    if (signal[i] > f->max_val) f->max_val = signal[i];
  }
  f->variance     = var_sum / (float)n;
  f->rms          = sqrtf(sum_sq / (float)n);
  f->peak_to_peak = f->max_val - f->min_val;

  /* Crest factor */
  float abs_max = fabsf(f->min_val) > fabsf(f->max_val) ? fabsf(f->min_val) : fabsf(f->max_val);
  f->crest_factor = (f->rms > 1e-10f) ? (abs_max / f->rms) : 0.0f;

  /* Skewness and kurtosis */
  float m3_sum = 0.0f, m4_sum = 0.0f;
  for (int i = 0; i < n; i++) {
    float d = signal[i] - f->mean;
    float d2 = d * d;
    m3_sum += d2 * d;
    m4_sum += d2 * d2;
  }
  float var_pow15 = powf(f->variance, 1.5f);
  float var_pow2  = f->variance * f->variance;
  f->skewness = (var_pow15 > 1e-10f) ? (m3_sum / (float)n / var_pow15) : 0.0f;
  f->kurtosis = (var_pow2  > 1e-10f) ? (m4_sum / (float)n / var_pow2 - 3.0f) : 0.0f;

  /* Zero crossings */
  f->zero_crossings = 0.0f;
  for (int i = 1; i < n; i++) {
    if ((signal[i] >= 0.0f && signal[i-1] < 0.0f) ||
        (signal[i] <  0.0f && signal[i-1] >= 0.0f))
      f->zero_crossings += 1.0f;
  }
}

/* Pack signal features into a flat array for MLP input */
static inline void clf_pack_signal_features(const clf_signal_features_t *f,
                                             float *out, int *offset) {
  out[(*offset)++] = f->mean;
  out[(*offset)++] = f->variance;
  out[(*offset)++] = f->rms;
  out[(*offset)++] = f->min_val;
  out[(*offset)++] = f->max_val;
  out[(*offset)++] = f->peak_to_peak;
  out[(*offset)++] = f->skewness;
  out[(*offset)++] = f->kurtosis;
  out[(*offset)++] = f->crest_factor;
  out[(*offset)++] = f->zero_crossings;
}

/* ================================================================== */
/* Label mapping (user-defined)                                        */
/* ================================================================== */

/*
 * model.h should define a label map for human-readable output:
 *
 *   const char *clf_labels[] = {
 *     "normal", "unbalance", "misalignment", "bearing_fault"
 *   };
 *
 * Usage:
 *   result.class_id → clf_labels[result.class_id]
 */

#endif /* CLASSIFIER_MLP_H */
