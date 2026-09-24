#include "vae_c_shuffle.h"

#include <stddef.h>
#include <stdint.h>


VAECRngStatus vae_c_shuffle_bounded_u64(
    VAECRng *rng,
    uint64_t upper_bound_exclusive,
    uint64_t *value
)
{
    uint64_t random_value;
    uint64_t threshold;
    VAECRngStatus status;

    if (rng == NULL || value == NULL || upper_bound_exclusive == 0) {
        return VAE_C_RNG_INVALID_ARGUMENT;
    }

    threshold = (UINT64_C(0) - upper_bound_exclusive) %
                upper_bound_exclusive;

    do {
        status = vae_c_rng_next_u64(rng, &random_value);
        if (status != VAE_C_RNG_OK) {
            return status;
        }
    } while (random_value < threshold);

    *value = random_value % upper_bound_exclusive;
    return VAE_C_RNG_OK;
}


VAECRngStatus vae_c_shuffle_make_permutation(
    size_t count,
    uint64_t base_seed,
    uint64_t epoch,
    size_t *permutation
)
{
    uint64_t derived_seed;
    VAECRng rng;
    VAECRngStatus status;
    size_t index;

    if (count == 0) {
        return VAE_C_RNG_OK;
    }

    if (permutation == NULL) {
        return VAE_C_RNG_INVALID_ARGUMENT;
    }

#if SIZE_MAX > UINT64_MAX
    if (count > (size_t)UINT64_MAX) {
        return VAE_C_RNG_INVALID_ARGUMENT;
    }
#endif

    status = vae_c_rng_derive_seed(
        base_seed,
        VAE_C_RNG_DOMAIN_SHUFFLE,
        epoch,
        UINT64_C(0),
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

    for (index = 0; index < count; index++) {
        permutation[index] = index;
    }

    for (index = count - 1; index > 0; index--) {
        uint64_t selected;
        size_t selected_index;
        size_t temporary;

        status = vae_c_shuffle_bounded_u64(
            &rng,
            (uint64_t)index + UINT64_C(1),
            &selected
        );
        if (status != VAE_C_RNG_OK) {
            return status;
        }

        selected_index = (size_t)selected;
        temporary = permutation[index];
        permutation[index] = permutation[selected_index];
        permutation[selected_index] = temporary;
    }

    return VAE_C_RNG_OK;
}
