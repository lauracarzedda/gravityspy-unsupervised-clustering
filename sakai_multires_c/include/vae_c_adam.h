#ifndef VAE_C_ADAM_H
#define VAE_C_ADAM_H

#include "vae_c_ops.h"
#include "vae_c_parameter.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


#define VAE_C_ADAM_LEARNING_RATE_V1 5.0e-4
#define VAE_C_ADAM_BETA1_V1         0.9
#define VAE_C_ADAM_BETA2_V1         0.999
#define VAE_C_ADAM_EPSILON_V1       1.0e-8


typedef struct {
    double learning_rate;
    double beta1;
    double beta2;
    double adam_epsilon;

    uint64_t step;

    double beta1_power;
    double beta2_power;

    bool configured;
} VAECAdam;


void vae_c_adam_init(VAECAdam *optimizer);

void vae_c_adam_free(VAECAdam *optimizer);

bool vae_c_adam_is_empty(const VAECAdam *optimizer);

bool vae_c_adam_is_configured(const VAECAdam *optimizer);

/*
 * PAPER: Case 3 uses Adam with learning rate 5e-4, beta1 0.9,
 * beta2 0.999, and Adam epsilon 1e-8.
 *
 * PROJECT: v1 exposes that fixed configuration explicitly.  Recursive
 * beta powers are retained in double and no arbitrary hyperparameter
 * configuration is exposed by this API.
 */
VAECOpStatus vae_c_adam_configure_v1(VAECAdam *optimizer);

/*
 * The caller supplies the complete parameter registry exactly once per
 * model update.  The operation reads but never clears gradients.  It
 * validates every parameter and every numeric candidate before committing
 * values, first moments, second moments, and finally the single model step.
 */
VAECOpStatus vae_c_adam_step(
    VAECAdam *optimizer,
    VAECParameter *const *parameters,
    size_t parameter_count
);

#endif
