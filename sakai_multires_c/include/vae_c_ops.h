#ifndef VAE_C_OPS_H
#define VAE_C_OPS_H

#include "vae_c_tensor.h"


typedef enum {
    VAE_C_OP_OK = 0,
    VAE_C_OP_INVALID_ARGUMENT,
    VAE_C_OP_INVALID_STATE,
    VAE_C_OP_SHAPE_MISMATCH,
    VAE_C_OP_GRADIENT_NOT_AVAILABLE,
    VAE_C_OP_OBJECT_NOT_EMPTY,
    VAE_C_OP_OVERFLOW,
    VAE_C_OP_ALLOCATION_FAILED
} VAECOpStatus;


/*
 * Input and output must be distinct objects. Their data and gradient
 * buffers must not overlap. Exact pointer equality is rejected, but this
 * API does not attempt the non-portable task of detecting partial overlap
 * between different C objects.
 *
 * Forward operations overwrite output data. Backward operations accumulate
 * gradients with += and never clear them implicitly.
 */
VAECOpStatus vae_c_relu_forward(
    const VAECTensor *input,
    VAECTensor *output
);

VAECOpStatus vae_c_relu_backward(
    const VAECTensor *input,
    const VAECTensor *output
);

VAECOpStatus vae_c_global_avg_pool2d_forward(
    const VAECTensor *input,
    VAECTensor *output
);

VAECOpStatus vae_c_global_avg_pool2d_backward(
    const VAECTensor *input,
    const VAECTensor *output
);

const char *vae_c_op_status_name(VAECOpStatus status);

#endif
