#ifndef VAE_C_BATCHNORM_H
#define VAE_C_BATCHNORM_H

#include "vae_c_ops.h"
#include "vae_c_parameter.h"
#include "vae_c_tensor.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


#define VAE_C_BATCH_NORM_EPSILON 1.0e-5f
#define VAE_C_BATCH_NORM_MOMENTUM 0.1f


typedef enum {
    VAE_C_BATCH_NORM_KIND_1D = 1,
    VAE_C_BATCH_NORM_KIND_2D = 2
} VAECBatchNormKind;


typedef enum {
    VAE_C_BATCH_NORM_MODE_TRAIN = 1,
    VAE_C_BATCH_NORM_MODE_EVAL = 2
} VAECBatchNormMode;


typedef struct {
    VAECBatchNormKind kind;
    size_t num_features;

    float epsilon;
    float momentum;

    VAECParameter gamma;
    VAECParameter beta;

    float *running_mean;
    float *running_variance;

    float *batch_mean;
    float *batch_inv_std;

    float *work_mean;
    float *work_inv_std;
    float *work_unbiased_variance;

    float *owned_block;
    size_t owned_bytes;

    size_t last_sample_count;
    VAECShape last_input_shape;
    VAECShape last_output_shape;
    bool cache_valid;
} VAECBatchNorm;


void vae_c_batch_norm_init(VAECBatchNorm *layer);

void vae_c_batch_norm_free(VAECBatchNorm *layer);

bool vae_c_batch_norm_is_empty(const VAECBatchNorm *layer);

VAECOpStatus vae_c_batch_norm1d_allocate(
    VAECBatchNorm *layer,
    size_t num_features,
    uint64_t gamma_parameter_key,
    uint64_t beta_parameter_key
);

VAECOpStatus vae_c_batch_norm2d_allocate(
    VAECBatchNorm *layer,
    size_t num_channels,
    uint64_t gamma_parameter_key,
    uint64_t beta_parameter_key
);

VAECOpStatus vae_c_batch_norm_zero_grad(VAECBatchNorm *layer);

/*
 * PAPER: Figure 8 marks hidden convolution and decoder fully-connected
 * outputs with Batch Normalization before ReLU.
 *
 * PROJECT: rank-2 ND tensors use BatchNorm1d over N, while rank-4 NCHW
 * tensors use BatchNorm2d over N*H*W per channel. Training uses biased
 * variance for normalization and unbiased variance for the running update.
 */
VAECOpStatus vae_c_batch_norm1d_forward(
    VAECBatchNorm *layer,
    VAECBatchNormMode mode,
    const VAECTensor *input,
    VAECTensor *output
);

VAECOpStatus vae_c_batch_norm1d_backward(
    VAECBatchNorm *layer,
    const VAECTensor *input,
    const VAECTensor *output
);

VAECOpStatus vae_c_batch_norm2d_forward(
    VAECBatchNorm *layer,
    VAECBatchNormMode mode,
    const VAECTensor *input,
    VAECTensor *output
);

VAECOpStatus vae_c_batch_norm2d_backward(
    VAECBatchNorm *layer,
    const VAECTensor *input,
    const VAECTensor *output
);

#endif
