#ifndef VAE_C_IIC_ENSEMBLE_H
#define VAE_C_IIC_ENSEMBLE_H

#include "vae_c_iic_pair_train.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


#define VAE_C_IIC_ENSEMBLE_K 5
#define VAE_C_IIC_ENSEMBLE_MAIN_DIM 36
#define VAE_C_IIC_ENSEMBLE_OVER_DIM 250
#define VAE_C_IIC_ENSEMBLE_FIRST_SEED UINT64_C(123)
#define VAE_C_IIC_ENSEMBLE_SHUFFLE_SEED UINT64_C(101112)


/* PROJECT C5B5B initialization, not bitwise PyTorch RNG parity:
 * one caller-owned SplitMix64 stream per pair, initialized with its raw seed.
 * Main then overcluster each consume one uniform_f32 draw per weight/bias
 * value, in Linear1 weight -> Linear2 weight -> Linear2 bias order. The
 * PyTorch 2.11 Linear bound is 1/sqrt(fan_in); BatchNorm is deterministic.
 * No VAE initialization-domain seed derivation is applied.
 *
 * Pairs and workspaces own memory and must never be copied by assignment
 * after allocation. The caller must initialize this object before use.
 */
typedef struct {
    size_t input_dim;
    size_t capacity;
    uint64_t seeds[VAE_C_IIC_ENSEMBLE_K];
    VAECIICPair pairs[VAE_C_IIC_ENSEMBLE_K];
    VAECIICPairWorkspace workspaces[VAE_C_IIC_ENSEMBLE_K];
    VAECAdam optimizers[VAE_C_IIC_ENSEMBLE_K];
    bool allocated;
} VAECIICEnsemble;


void vae_c_iic_ensemble_init(VAECIICEnsemble *ensemble);
void vae_c_iic_ensemble_free(VAECIICEnsemble *ensemble);
bool vae_c_iic_ensemble_is_empty(const VAECIICEnsemble *ensemble);
VAECOpStatus vae_c_iic_ensemble_validate(const VAECIICEnsemble *ensemble);

/* Official PROJECT seeds 123,124,125,126,127. */
VAECOpStatus vae_c_iic_ensemble_allocate_v1(
    VAECIICEnsemble *ensemble,
    size_t input_dim,
    size_t capacity
);

/* Controlled seed-perturbation fixture entry point. The same initialization
 * formula and stream order are used; seeds must be pairwise distinct. */
VAECOpStatus vae_c_iic_ensemble_allocate_seeded(
    VAECIICEnsemble *ensemble,
    size_t input_dim,
    size_t capacity,
    const uint64_t seeds[VAE_C_IIC_ENSEMBLE_K]
);

/* Sends the same input tensor objects to all five independent pair/Adam
 * instances, in index order 0..4. Each per-pair step clears its own parameter
 * gradients. Input gradients follow the frozen pair API and accumulate.
 * Inputs must satisfy the frozen pair-train preflight (including grad buffers).
 * Failure after a pair's update begins cannot roll back prior pair updates. */
VAECOpStatus vae_c_iic_ensemble_train_step(
    VAECIICEnsemble *ensemble,
    const VAECTensor *input_a,
    const VAECTensor *input_b
);

#endif
