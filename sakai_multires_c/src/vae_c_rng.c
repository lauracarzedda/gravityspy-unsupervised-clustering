#include "vae_c_rng.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>


static int vae_c_rng_domain_is_valid(VAECRngDomain domain)
{
    return domain >= VAE_C_RNG_DOMAIN_INITIALIZATION &&
           domain <= VAE_C_RNG_DOMAIN_LATENT_VALIDATION;
}


static uint64_t vae_c_rng_mix64(uint64_t value)
{
    value += UINT64_C(0x9E3779B97F4A7C15);
    value =
        (value ^ (value >> 30)) *
        UINT64_C(0xBF58476D1CE4E5B9);
    value =
        (value ^ (value >> 27)) *
        UINT64_C(0x94D049BB133111EB);
    return value ^ (value >> 31);
}


VAECRngStatus vae_c_rng_init(VAECRng *rng, uint64_t seed)
{
    if (rng == NULL) {
        return VAE_C_RNG_INVALID_ARGUMENT;
    }

    rng->state = seed;
    return VAE_C_RNG_OK;
}


VAECRngStatus vae_c_rng_next_u64(VAECRng *rng, uint64_t *value)
{
    uint64_t state;

    if (rng == NULL || value == NULL) {
        return VAE_C_RNG_INVALID_ARGUMENT;
    }

    state = rng->state + UINT64_C(0x9E3779B97F4A7C15);
    rng->state = state;

    state =
        (state ^ (state >> 30)) *
        UINT64_C(0xBF58476D1CE4E5B9);
    state =
        (state ^ (state >> 27)) *
        UINT64_C(0x94D049BB133111EB);
    *value = state ^ (state >> 31);

    return VAE_C_RNG_OK;
}


VAECRngStatus vae_c_rng_derive_seed(
    uint64_t base_seed,
    VAECRngDomain domain,
    uint64_t epoch,
    uint64_t manifest_index,
    uint64_t stream_index,
    uint64_t *derived_seed
)
{
    uint64_t hash;

    if (derived_seed == NULL) {
        return VAE_C_RNG_INVALID_ARGUMENT;
    }

    if (!vae_c_rng_domain_is_valid(domain)) {
        return VAE_C_RNG_INVALID_DOMAIN;
    }

    hash = vae_c_rng_mix64(
        base_seed ^ UINT64_C(0xD1B54A32D192ED03)
    );
    hash = vae_c_rng_mix64(hash ^ (uint64_t)domain);
    hash = vae_c_rng_mix64(hash ^ epoch);
    hash = vae_c_rng_mix64(hash ^ manifest_index);
    hash = vae_c_rng_mix64(hash ^ stream_index);

    *derived_seed = hash;
    return VAE_C_RNG_OK;
}


VAECRngStatus vae_c_rng_uniform_f32(VAECRng *rng, float *value)
{
    uint64_t random_bits;
    uint64_t bits24;
    VAECRngStatus status;

    if (rng == NULL || value == NULL) {
        return VAE_C_RNG_INVALID_ARGUMENT;
    }

    status = vae_c_rng_next_u64(rng, &random_bits);
    if (status != VAE_C_RNG_OK) {
        return status;
    }

    bits24 = random_bits >> 40;
    *value = (float)bits24 * (1.0f / 16777216.0f);
    return VAE_C_RNG_OK;
}


VAECRngStatus vae_c_rng_uniform_open_f64(VAECRng *rng, double *value)
{
    uint64_t random_bits;
    uint64_t bits52;
    double candidate;
    VAECRngStatus status;

    if (rng == NULL || value == NULL) {
        return VAE_C_RNG_INVALID_ARGUMENT;
    }

    status = vae_c_rng_next_u64(rng, &random_bits);
    if (status != VAE_C_RNG_OK) {
        return status;
    }

    bits52 = random_bits >> 12;
    candidate =
        ((double)bits52 + 0.5) *
        (1.0 / 4503599627370496.0);

    if (!isfinite(candidate) ||
        candidate <= 0.0 ||
        candidate >= 1.0) {

        return VAE_C_RNG_NONFINITE_RESULT;
    }

    *value = candidate;
    return VAE_C_RNG_OK;
}


VAECRngStatus vae_c_rng_standard_normal_f32(
    VAECRng *rng,
    float *value
)
{
    const double two_pi = 6.283185307179586476925286766559;
    double uniform1;
    double uniform2;
    double radius;
    double angle;
    double normal;
    float candidate;
    VAECRngStatus status;

    if (rng == NULL || value == NULL) {
        return VAE_C_RNG_INVALID_ARGUMENT;
    }

    status = vae_c_rng_uniform_open_f64(rng, &uniform1);
    if (status != VAE_C_RNG_OK) {
        return status;
    }

    status = vae_c_rng_uniform_open_f64(rng, &uniform2);
    if (status != VAE_C_RNG_OK) {
        return status;
    }

    radius = sqrt(-2.0 * log(uniform1));
    angle = two_pi * uniform2;
    normal = radius * cos(angle);

    if (!isfinite(normal)) {
        return VAE_C_RNG_NONFINITE_RESULT;
    }

    candidate = (float)normal;
    if (!isfinite((double)candidate)) {
        return VAE_C_RNG_NONFINITE_RESULT;
    }

    *value = candidate;
    return VAE_C_RNG_OK;
}


VAECRngStatus vae_c_rng_fill_standard_normal_f32(
    VAECRng *rng,
    float *values,
    size_t count
)
{
    size_t index;
    VAECRngStatus status;

    if (count == 0) {
        return VAE_C_RNG_OK;
    }

    if (rng == NULL || values == NULL) {
        return VAE_C_RNG_INVALID_ARGUMENT;
    }

    for (index = 0; index < count; index++) {
        status = vae_c_rng_standard_normal_f32(rng, &values[index]);
        if (status != VAE_C_RNG_OK) {
            return status;
        }
    }

    return VAE_C_RNG_OK;
}


static VAECRngStatus vae_c_rng_fill_derived_epsilon_f32(
    uint64_t base_seed,
    VAECRngDomain domain,
    uint64_t epoch,
    uint64_t manifest_index,
    float *values,
    size_t count
)
{
    uint64_t derived_seed;
    VAECRng rng;
    VAECRngStatus status;

    if (count == 0) {
        return VAE_C_RNG_OK;
    }

    if (values == NULL) {
        return VAE_C_RNG_INVALID_ARGUMENT;
    }

    status = vae_c_rng_derive_seed(
        base_seed,
        domain,
        epoch,
        manifest_index,
        UINT64_C(0),
        &derived_seed
    );
    if (status != VAE_C_RNG_OK) {
        return status;
    }

    status = vae_c_rng_init(&rng, derived_seed);
    if (status != VAE_C_RNG_OK) {
        return status;
    }

    return vae_c_rng_fill_standard_normal_f32(&rng, values, count);
}


VAECRngStatus vae_c_rng_fill_latent_train_epsilon_f32(
    uint64_t base_seed,
    uint64_t epoch,
    uint64_t manifest_index,
    float *values,
    size_t count
)
{
    return vae_c_rng_fill_derived_epsilon_f32(
        base_seed,
        VAE_C_RNG_DOMAIN_LATENT_TRAIN,
        epoch,
        manifest_index,
        values,
        count
    );
}


VAECRngStatus vae_c_rng_fill_latent_validation_epsilon_f32(
    uint64_t base_seed,
    uint64_t manifest_index,
    float *values,
    size_t count
)
{
    return vae_c_rng_fill_derived_epsilon_f32(
        base_seed,
        VAE_C_RNG_DOMAIN_LATENT_VALIDATION,
        UINT64_C(0),
        manifest_index,
        values,
        count
    );
}


const char *vae_c_rng_status_name(VAECRngStatus status)
{
    switch (status) {
        case VAE_C_RNG_OK:
            return "success";
        case VAE_C_RNG_INVALID_ARGUMENT:
            return "invalid_argument";
        case VAE_C_RNG_INVALID_DOMAIN:
            return "invalid_domain";
        case VAE_C_RNG_NONFINITE_RESULT:
            return "nonfinite_result";
        default:
            return "unknown";
    }
}
