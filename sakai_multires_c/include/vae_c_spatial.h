#ifndef VAE_C_SPATIAL_H
#define VAE_C_SPATIAL_H

#include "vae_c_ops.h"
#include "vae_c_tensor.h"

#include <stdbool.h>
#include <stddef.h>


typedef struct {
    size_t *argmax_indices;

    size_t capacity;
    size_t owned_bytes;

    size_t last_count;
    VAECShape last_input_shape;
    VAECShape last_output_shape;

    bool cache_valid;
} VAECMaxPool2D;


void vae_c_max_pool2d_init(VAECMaxPool2D *layer);

void vae_c_max_pool2d_free(VAECMaxPool2D *layer);

bool vae_c_max_pool2d_is_empty(const VAECMaxPool2D *layer);

VAECOpStatus vae_c_max_pool2d_allocate(
    VAECMaxPool2D *layer,
    size_t max_output_elements
);

/*
 * PROJECT contract: fixed 2x2 kernel, stride 2, no padding. Ties select
 * the first maximum visited in row-major order. The cache stores absolute
 * input-array indices and never stores tensor pointers.
 *
 * Input and output must be distinct, non-overlapping tensors. Exact pointer
 * equality is rejected; partial overlap between different C objects is not
 * detected portably. Forward overwrites output data and the cache. Backward
 * accumulates input gradients with += and never clears them implicitly.
 */
VAECOpStatus vae_c_max_pool2d_forward(
    VAECMaxPool2D *layer,
    const VAECTensor *input,
    VAECTensor *output
);

VAECOpStatus vae_c_max_pool2d_backward(
    const VAECMaxPool2D *layer,
    const VAECTensor *input,
    const VAECTensor *output
);

/*
 * PROJECT nearest-neighbour mapping:
 * src_y = (out_y * H_in) / H_out
 * src_x = (out_x * W_in) / W_out
 * using integer arithmetic. No framework-equivalence claim is implied.
 */
VAECOpStatus vae_c_nearest_upsample2d_forward(
    const VAECTensor *input,
    VAECTensor *output
);

VAECOpStatus vae_c_nearest_upsample2d_backward(
    const VAECTensor *input,
    const VAECTensor *output
);

#endif
