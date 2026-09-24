#ifndef VAE_C_RNG_H
#define VAE_C_RNG_H

#include <stddef.h>
#include <stdint.h>


typedef enum {
    VAE_C_RNG_DOMAIN_INITIALIZATION = 1,
    VAE_C_RNG_DOMAIN_SHUFFLE = 2,
    VAE_C_RNG_DOMAIN_AUGMENTATION = 3,
    VAE_C_RNG_DOMAIN_LATENT_TRAIN = 4,
    VAE_C_RNG_DOMAIN_LATENT_VALIDATION = 5
} VAECRngDomain;


typedef enum {
    VAE_C_RNG_OK = 0,
    VAE_C_RNG_INVALID_ARGUMENT,
    VAE_C_RNG_INVALID_DOMAIN,
    VAE_C_RNG_NONFINITE_RESULT
} VAECRngStatus;


typedef struct {
    uint64_t state;
} VAECRng;


VAECRngStatus vae_c_rng_init(VAECRng *rng, uint64_t seed);

VAECRngStatus vae_c_rng_next_u64(VAECRng *rng, uint64_t *value);

VAECRngStatus vae_c_rng_derive_seed(
    uint64_t base_seed,
    VAECRngDomain domain,
    uint64_t epoch,
    uint64_t manifest_index,
    uint64_t stream_index,
    uint64_t *derived_seed
);

VAECRngStatus vae_c_rng_uniform_f32(VAECRng *rng, float *value);

VAECRngStatus vae_c_rng_uniform_open_f64(VAECRng *rng, double *value);

/*
 * PAPER: VAE reparameterization uses standard-normal epsilon.
 * PROJECT: SplitMix64 and Box-Muller define how epsilon is generated here.
 * Each call consumes exactly two uint64_t values and keeps no spare sample.
 * Integer sequences and seed derivation are bitwise-defined. Box-Muller
 * reproducibility is required within the same toolchain/libm environment;
 * bitwise identity across different libm implementations is not promised.
 */
VAECRngStatus vae_c_rng_standard_normal_f32(
    VAECRng *rng,
    float *value
);

VAECRngStatus vae_c_rng_fill_standard_normal_f32(
    VAECRng *rng,
    float *values,
    size_t count
);

VAECRngStatus vae_c_rng_fill_latent_train_epsilon_f32(
    uint64_t base_seed,
    uint64_t epoch,
    uint64_t manifest_index,
    float *values,
    size_t count
);

VAECRngStatus vae_c_rng_fill_latent_validation_epsilon_f32(
    uint64_t base_seed,
    uint64_t manifest_index,
    float *values,
    size_t count
);

const char *vae_c_rng_status_name(VAECRngStatus status);

#endif
