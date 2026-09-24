#include "vae_c_init.h"

#include "vae_c_rng.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>


static VAECInitStatus vae_c_init_fill_constant_f32(
    float *values,
    size_t count,
    float constant
)
{
    size_t index;

    if (count == 0) {
        return VAE_C_INIT_OK;
    }

    if (values == NULL) {
        return VAE_C_INIT_INVALID_ARGUMENT;
    }

    for (index = 0; index < count; index++) {
        values[index] = constant;
    }

    return VAE_C_INIT_OK;
}


static VAECInitStatus vae_c_init_uniform_with_bound_f32(
    float *values,
    size_t count,
    uint64_t base_seed,
    uint64_t parameter_key,
    double bound
)
{
    uint64_t derived_seed;
    VAECRng rng;
    VAECRngStatus rng_status;
    size_t index;

    if (count == 0) {
        return VAE_C_INIT_OK;
    }

    if (values == NULL) {
        return VAE_C_INIT_INVALID_ARGUMENT;
    }

    if (!isfinite(bound) || bound <= 0.0) {
        return VAE_C_INIT_NONFINITE_BOUND;
    }

    rng_status = vae_c_rng_derive_seed(
        base_seed,
        VAE_C_RNG_DOMAIN_INITIALIZATION,
        UINT64_C(0),
        UINT64_C(0),
        parameter_key,
        &derived_seed
    );
    if (rng_status != VAE_C_RNG_OK) {
        return VAE_C_INIT_RNG_ERROR;
    }

    rng_status = vae_c_rng_init(&rng, derived_seed);
    if (rng_status != VAE_C_RNG_OK) {
        return VAE_C_INIT_RNG_ERROR;
    }

    for (index = 0; index < count; index++) {
        float uniform;
        double candidate;

        rng_status = vae_c_rng_uniform_f32(&rng, &uniform);
        if (rng_status != VAE_C_RNG_OK) {
            return VAE_C_INIT_RNG_ERROR;
        }

        candidate =
            -bound +
            2.0 * bound * (double)uniform;

        if (!isfinite(candidate) ||
            !isfinite((double)(float)candidate)) {

            return VAE_C_INIT_NONFINITE_BOUND;
        }

        values[index] = (float)candidate;
    }

    return VAE_C_INIT_OK;
}


VAECInitStatus vae_c_init_zero_f32(float *values, size_t count)
{
    return vae_c_init_fill_constant_f32(values, count, 0.0f);
}


VAECInitStatus vae_c_init_one_f32(float *values, size_t count)
{
    return vae_c_init_fill_constant_f32(values, count, 1.0f);
}


VAECInitStatus vae_c_init_batchnorm_f32(
    float *gamma,
    float *beta,
    float *running_mean,
    float *running_variance,
    size_t count
)
{
    size_t index;

    if (count == 0) {
        return VAE_C_INIT_OK;
    }

    if (gamma == NULL ||
        beta == NULL ||
        running_mean == NULL ||
        running_variance == NULL) {

        return VAE_C_INIT_INVALID_ARGUMENT;
    }

    for (index = 0; index < count; index++) {
        gamma[index] = 1.0f;
        beta[index] = 0.0f;
        running_mean[index] = 0.0f;
        running_variance[index] = 1.0f;
    }

    return VAE_C_INIT_OK;
}


VAECInitStatus vae_c_init_he_uniform_f32(
    float *values,
    size_t count,
    uint64_t base_seed,
    uint64_t parameter_key,
    size_t fan_in
)
{
    double bound;

    if (fan_in == 0) {
        return VAE_C_INIT_INVALID_FAN;
    }

    if (count > 0 && values == NULL) {
        return VAE_C_INIT_INVALID_ARGUMENT;
    }

    bound = sqrt(6.0 / (double)fan_in);
    if (!isfinite(bound) || bound <= 0.0) {
        return VAE_C_INIT_NONFINITE_BOUND;
    }

    return vae_c_init_uniform_with_bound_f32(
        values,
        count,
        base_seed,
        parameter_key,
        bound
    );
}


VAECInitStatus vae_c_init_xavier_uniform_f32(
    float *values,
    size_t count,
    uint64_t base_seed,
    uint64_t parameter_key,
    size_t fan_in,
    size_t fan_out
)
{
    size_t fan_sum;
    double bound;

    if (fan_in == 0 || fan_out == 0) {
        return VAE_C_INIT_INVALID_FAN;
    }

    if (count > 0 && values == NULL) {
        return VAE_C_INIT_INVALID_ARGUMENT;
    }

    if (fan_in > SIZE_MAX - fan_out) {
        return VAE_C_INIT_FAN_SUM_OVERFLOW;
    }

    fan_sum = fan_in + fan_out;
    bound = sqrt(6.0 / (double)fan_sum);

    if (!isfinite(bound) || bound <= 0.0) {
        return VAE_C_INIT_NONFINITE_BOUND;
    }

    return vae_c_init_uniform_with_bound_f32(
        values,
        count,
        base_seed,
        parameter_key,
        bound
    );
}


const char *vae_c_init_status_name(VAECInitStatus status)
{
    switch (status) {
        case VAE_C_INIT_OK:
            return "success";
        case VAE_C_INIT_INVALID_ARGUMENT:
            return "invalid_argument";
        case VAE_C_INIT_INVALID_FAN:
            return "invalid_fan";
        case VAE_C_INIT_FAN_SUM_OVERFLOW:
            return "fan_sum_overflow";
        case VAE_C_INIT_NONFINITE_BOUND:
            return "nonfinite_bound";
        case VAE_C_INIT_RNG_ERROR:
            return "rng_error";
        default:
            return "unknown";
    }
}
