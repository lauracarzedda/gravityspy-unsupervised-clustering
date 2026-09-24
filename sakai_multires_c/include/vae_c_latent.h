#ifndef VAE_C_LATENT_H
#define VAE_C_LATENT_H

#include "vae_c_ops.h"


typedef enum {
    VAE_C_LATENT_MODE_STOCHASTIC = 1,
    VAE_C_LATENT_MODE_MEAN = 2
} VAECLatentMode;


/*
 * PAPER: the encoder produces a mean and a dispersion parameter, epsilon is
 * standard normal, reparameterization is z = mu + epsilon * sigma, and mu is
 * the deterministic feature used by the later IIC phase.
 *
 * PROJECT: the dispersion parameter is represented as logvar.  The packed
 * encoder output is laid out per event as [mu, logvar], and this API performs
 * an explicit split copy.  Epsilon is supplied by the caller; stochastic and
 * mean-only execution, finite-result checks, backward accumulation, and the
 * exact error semantics are project decisions.
 *
 * Inputs and outputs must be distinct objects.  Exact pointer aliases across
 * their data and gradient buffers are rejected; partial overlap between
 * distinct C objects is deliberately not detected.  Forward overwrites output
 * data, while backward accumulates upstream gradients with +=.
 */
VAECOpStatus vae_c_latent_split_forward(
    const VAECTensor *packed,
    VAECTensor *mu,
    VAECTensor *logvar
);

VAECOpStatus vae_c_latent_split_backward(
    const VAECTensor *packed,
    const VAECTensor *mu,
    const VAECTensor *logvar
);

VAECOpStatus vae_c_reparameterize_forward(
    VAECLatentMode mode,
    const VAECTensor *mu,
    const VAECTensor *logvar,
    const VAECTensor *epsilon,
    VAECTensor *z
);

VAECOpStatus vae_c_reparameterize_backward(
    VAECLatentMode mode,
    const VAECTensor *mu,
    const VAECTensor *logvar,
    const VAECTensor *epsilon,
    const VAECTensor *z
);

#endif
