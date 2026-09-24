#ifndef VAE_C_INIT_H
#define VAE_C_INIT_H

#include <stddef.h>
#include <stdint.h>


typedef enum {
    VAE_C_INIT_OK = 0,
    VAE_C_INIT_INVALID_ARGUMENT,
    VAE_C_INIT_INVALID_FAN,
    VAE_C_INIT_FAN_SUM_OVERFLOW,
    VAE_C_INIT_NONFINITE_BOUND,
    VAE_C_INIT_RNG_ERROR
} VAECInitStatus;


VAECInitStatus vae_c_init_zero_f32(float *values, size_t count);

VAECInitStatus vae_c_init_one_f32(float *values, size_t count);

/*
 * gamma, beta, running_mean and running_variance must be distinct,
 * non-overlapping arrays whenever count is greater than zero.
 */
VAECInitStatus vae_c_init_batchnorm_f32(
    float *gamma,
    float *beta,
    float *running_mean,
    float *running_variance,
    size_t count
);

/*
 * PROJECT initialization policies:
 * He/Kaiming uniform uses sqrt(6 / fan_in).
 * Xavier uniform uses sqrt(6 / (fan_in + fan_out)).
 * Parameter keys derive independent initialization streams.
 */
VAECInitStatus vae_c_init_he_uniform_f32(
    float *values,
    size_t count,
    uint64_t base_seed,
    uint64_t parameter_key,
    size_t fan_in
);

VAECInitStatus vae_c_init_xavier_uniform_f32(
    float *values,
    size_t count,
    uint64_t base_seed,
    uint64_t parameter_key,
    size_t fan_in,
    size_t fan_out
);

const char *vae_c_init_status_name(VAECInitStatus status);

#endif
