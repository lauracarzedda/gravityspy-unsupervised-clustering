#ifndef VAE_C_CONV2D_H
#define VAE_C_CONV2D_H

#include "vae_c_ops.h"
#include "vae_c_parameter.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


#define VAE_C_CONV2D_KERNEL_SIZE 3
#define VAE_C_CONV2D_PADDING 1


typedef struct {
    size_t in_channels;
    size_t out_channels;

    size_t kernel_size;
    size_t stride;
    size_t padding;

    bool with_bias;

    VAECParameter weights;
    VAECParameter bias;
} VAECConv2D;


void vae_c_conv2d_init(VAECConv2D *layer);

void vae_c_conv2d_free(VAECConv2D *layer);

bool vae_c_conv2d_is_empty(const VAECConv2D *layer);

VAECOpStatus vae_c_conv2d_allocate(
    VAECConv2D *layer,
    size_t in_channels,
    size_t out_channels,
    size_t stride,
    bool with_bias,
    uint64_t weight_parameter_key,
    uint64_t bias_parameter_key
);

VAECOpStatus vae_c_conv2d_zero_grad(VAECConv2D *layer);

/*
 * PAPER: Figure 8 contains encoder and decoder convolutions and places
 * Batch Normalization plus ReLU after hidden convolutions.
 *
 * PROJECT: this reference backend uses direct NCHW/OIHW cross-correlation,
 * a fixed 3x3 kernel, zero padding 1, and stride 1 or 2. These operational
 * choices are project decisions, not parameters reported by Sakai et al.
 */
VAECOpStatus vae_c_conv2d_forward(
    const VAECConv2D *layer,
    const VAECTensor *input,
    VAECTensor *output
);

VAECOpStatus vae_c_conv2d_backward(
    VAECConv2D *layer,
    const VAECTensor *input,
    const VAECTensor *output
);

#endif
