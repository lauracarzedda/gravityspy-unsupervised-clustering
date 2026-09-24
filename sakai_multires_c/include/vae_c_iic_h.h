#ifndef VAE_C_IIC_H_H
#define VAE_C_IIC_H_H

#include "vae_c_iic_ensemble.h"

#include <stdbool.h>
#include <stddef.h>

#define VAE_C_IIC_H_COLUMNS \
    (VAE_C_IIC_ENSEMBLE_K * VAE_C_IIC_ENSEMBLE_MAIN_DIM)

/* Scratch belongs to the caller. No overcluster workspace is allocated. */
typedef struct {
    size_t capacity;
    size_t input_dim;
    float *row_major_scratch;
    VAECIICHeadWorkspace main[VAE_C_IIC_ENSEMBLE_K];
    bool allocated;
} VAECIICHWorkspace;

void vae_c_iic_h_workspace_init(VAECIICHWorkspace *workspace);
void vae_c_iic_h_workspace_free(VAECIICHWorkspace *workspace);
VAECOpStatus vae_c_iic_h_workspace_allocate(
    const VAECIICEnsemble *ensemble,
    size_t capacity,
    VAECIICHWorkspace *workspace
);

/* Input is a contiguous rank-2 float32 posterior-mu tensor [B,input_dim].
 * Output is row-major [B,5*36], with block k taken only from main head k's
 * Softmax in EVAL. The caller supplies at least B*180 floats. Invalid input
 * leaves output untouched. No gradient, Adam step, or overcluster forward is
 * performed. EVAL may change transient head/workspace generation and scratch,
 * but not parameters, Adam, or persistent BatchNorm running state. */
VAECOpStatus vae_c_iic_h_forward(
    VAECIICEnsemble *ensemble,
    const VAECTensor *central_mu,
    VAECIICHWorkspace *workspace,
    float *output,
    size_t output_float_capacity
);

#endif
