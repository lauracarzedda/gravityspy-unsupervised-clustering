#ifndef VAE_C_OBJECTIVE_H
#define VAE_C_OBJECTIVE_H

#include "vae_c_ops.h"


#define VAE_C_VAE_LOSS_BETA_KL_V1 1.0f


typedef struct {
    size_t batch_size;
    size_t values_per_sample;
    size_t total_values;

    double total_sum;
    double mean_per_sample;
    double mean_per_value;
} VAECBCEWithLogitsStats;


typedef struct {
    size_t batch_size;
    size_t latent_dim;
    size_t total_values;

    double total_sum;
    double mean_per_sample;
    double mean_per_dimension;
} VAECKLGaussianStats;


typedef struct {
    size_t batch_size;

    float beta_kl;

    VAECBCEWithLogitsStats reconstruction;
    VAECKLGaussianStats kl;

    double total_sum;
    double mean_per_sample;
} VAECVAELossStats;


/*
 * PAPER: the VAE lower bound contains both a Bernoulli reconstruction term
 * and a KL divergence term.
 *
 * PROJECT: the decoder emits linear logits.  This API evaluates a stable
 * BCE-with-logits expression for float32 targets in [0, 1], reduces by sum
 * per sample and mean over the batch, exposes logging statistics, and
 * accumulates float32 logit gradients using an explicit chain-rule factor.
 */
void vae_c_bce_with_logits_stats_init(
    VAECBCEWithLogitsStats *stats
);

VAECOpStatus vae_c_bce_with_logits_forward(
    const VAECTensor *logits,
    const VAECTensor *targets,
    VAECBCEWithLogitsStats *stats
);

VAECOpStatus vae_c_bce_with_logits_backward(
    VAECTensor *logits,
    const VAECTensor *targets,
    float upstream_gradient
);

/*
 * PAPER: the VAE lower bound contains a KL divergence from a Gaussian
 * posterior to the standard Gaussian prior, summed over latent dimensions.
 *
 * PROJECT: the diagonal posterior is parameterized by mu and logvar.  This
 * API uses expm1 for stability, reduces by sum per sample and mean over the
 * batch, exposes per-dimension logging, and accumulates float32 gradients
 * using an explicit scalar chain-rule factor.
 */
void vae_c_kl_gaussian_stats_init(
    VAECKLGaussianStats *stats
);

VAECOpStatus vae_c_kl_gaussian_forward(
    const VAECTensor *mu,
    const VAECTensor *logvar,
    VAECKLGaussianStats *stats
);

VAECOpStatus vae_c_kl_gaussian_backward(
    VAECTensor *mu,
    VAECTensor *logvar,
    float upstream_gradient
);

/*
 * PAPER: VAE training minimizes the negative lower bound composed of a
 * Bernoulli reconstruction term and a KL divergence to a standard Gaussian
 * prior.
 *
 * PROJECT: the v1 C objective composes the existing BCE-with-logits and
 * Gaussian mu/logvar objectives, preserves their sum-per-sample then
 * mean-over-batch reductions, fixes beta_KL to one, exposes aggregate
 * statistics, and applies one upstream chain-rule factor atomically.
 */
void vae_c_vae_loss_stats_init(
    VAECVAELossStats *stats
);

VAECOpStatus vae_c_vae_loss_forward(
    const VAECTensor *logits,
    const VAECTensor *targets,
    const VAECTensor *mu,
    const VAECTensor *logvar,
    VAECVAELossStats *stats
);

VAECOpStatus vae_c_vae_loss_backward(
    VAECTensor *logits,
    const VAECTensor *targets,
    VAECTensor *mu,
    VAECTensor *logvar,
    float upstream_gradient
);

#endif
