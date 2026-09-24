#ifndef VAE_C_DENSE_H
#define VAE_C_DENSE_H

#include "vae_c_ops.h"
#include "vae_c_parameter.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


typedef struct {
    size_t in_features;
    size_t out_features;
    bool with_bias;

    VAECParameter weights;
    VAECParameter bias;
} VAECDense;


void vae_c_dense_init(VAECDense *layer);

void vae_c_dense_free(VAECDense *layer);

bool vae_c_dense_is_empty(const VAECDense *layer);

VAECOpStatus vae_c_dense_allocate(
    VAECDense *layer,
    size_t in_features,
    size_t out_features,
    bool with_bias,
    uint64_t weight_parameter_key,
    uint64_t bias_parameter_key
);

VAECOpStatus vae_c_dense_zero_grad(VAECDense *layer);

/*
 * Input and output must be distinct, non-overlapping tensors. Exact pointer
 * equality is rejected; partial overlap between different C objects is not
 * detected portably. Forward overwrites output data. Backward accumulates
 * input and parameter gradients with += and never clears them implicitly.
 */
VAECOpStatus vae_c_dense_forward(
    const VAECDense *layer,
    const VAECTensor *input,
    VAECTensor *output
);

VAECOpStatus vae_c_dense_backward(
    VAECDense *layer,
    const VAECTensor *input,
    const VAECTensor *output
);

#endif
