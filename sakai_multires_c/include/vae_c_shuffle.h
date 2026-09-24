#ifndef VAE_C_SHUFFLE_H
#define VAE_C_SHUFFLE_H

#include "vae_c_rng.h"

#include <stddef.h>
#include <stdint.h>


/*
 * PROJECT: draw uniformly from [0, upper_bound_exclusive) using the
 * supplied VAE RNG stream. Rejection sampling removes modulo bias.
 * On failure, value is left unchanged.
 */
VAECRngStatus vae_c_shuffle_bounded_u64(
    VAECRng *rng,
    uint64_t upper_bound_exclusive,
    uint64_t *value
);

/*
 * PROJECT: construct a deterministic, unbiased Fisher-Yates permutation
 * of 0..count-1. The stream is derived solely from base_seed and epoch in
 * VAE_C_RNG_DOMAIN_SHUFFLE; no global or persistent state is used.
 *
 * For count == 0, permutation may be NULL. For count > 0 it must point to
 * storage for at least count size_t values. Invalid arguments are rejected
 * before any output element is written.
 */
VAECRngStatus vae_c_shuffle_make_permutation(
    size_t count,
    uint64_t base_seed,
    uint64_t epoch,
    size_t *permutation
);

#endif
