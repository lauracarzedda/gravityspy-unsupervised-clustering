#ifndef VAE_C_MODEL_WORKSPACE_H
#define VAE_C_MODEL_WORKSPACE_H

#include "vae_c_model.h"

#include <stdbool.h>
#include <stddef.h>


#define VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS \
    (7 * VAE_C_MAX_CONV_DEPTH + 10)


/*
 * PAPER: the encoder/decoder order follows the convolution, BatchNorm,
 * ReLU, pooling, fully-connected, and nearest-upsample structure of
 * Figure 8.
 *
 * PROJECT: this separate C workspace owns forward/backward activation
 * storage, except for decoder_reshape_view.  That field is the sole
 * non-owning view and aliases decoder_fc_relu_output.  This owning type
 * is not copy-safe through plain assignment by callers; use its lifecycle
 * API instead.
 */
typedef struct {
    char profile_id[VAE_C_PROFILE_TEXT_CAP];
    VAECConfig config;

    size_t batch_size;

    VAECTensor input;

    VAECTensor encoder_conv_outputs[
        VAE_C_MODEL_MAX_ENCODER_CONVS
    ];
    VAECTensor encoder_batch_norm_outputs[
        VAE_C_MODEL_MAX_ENCODER_CONVS
    ];
    VAECTensor encoder_relu_outputs[
        VAE_C_MODEL_MAX_ENCODER_CONVS
    ];

    VAECTensor encoder_pool_output;
    VAECTensor encoder_global_avg_output;
    VAECTensor encoder_packed_output;

    VAECTensor mu;
    VAECTensor logvar;
    VAECTensor epsilon;
    VAECTensor z;

    VAECTensor decoder_fc_output;
    VAECTensor decoder_fc_batch_norm_output;
    VAECTensor decoder_fc_relu_output;

    VAECTensor decoder_reshape_view;

    VAECTensor decoder_upsample_outputs[
        VAE_C_MAX_DECODER_UPSAMPLES
    ];
    VAECTensor decoder_hidden_conv_outputs[
        VAE_C_MODEL_MAX_DECODER_HIDDEN_CONVS
    ];
    VAECTensor decoder_hidden_batch_norm_outputs[
        VAE_C_MODEL_MAX_DECODER_HIDDEN_CONVS
    ];
    VAECTensor decoder_hidden_relu_outputs[
        VAE_C_MODEL_MAX_DECODER_HIDDEN_CONVS
    ];

    VAECTensor logits;

    size_t owned_tensor_count;
    size_t logical_value_count;
    size_t storage_float_count;
    size_t storage_bytes;

    /*
     * PROJECT: a nonzero, model-bound ticket is issued only by a completed
     * STOCHASTIC+TRAIN composite VAE forward.  Stale tickets remain
     * structurally valid but are not backward-ready.
     */
    const VAECModel *vae_backward_model;
    uint64_t vae_backward_generation;

    bool allocated;
} VAECModelWorkspace;


void vae_c_model_workspace_init(VAECModelWorkspace *workspace);

void vae_c_model_workspace_free(VAECModelWorkspace *workspace);

bool vae_c_model_workspace_is_empty(
    const VAECModelWorkspace *workspace
);

bool vae_c_model_workspace_is_allocated(
    const VAECModel *model,
    const VAECModelWorkspace *workspace
);

VAECOpStatus vae_c_model_workspace_allocate(
    const VAECModel *model,
    size_t batch_size,
    VAECModelWorkspace *workspace
);

VAECOpStatus vae_c_model_workspace_validate(
    const VAECModel *model,
    const VAECModelWorkspace *workspace
);

VAECOpStatus vae_c_model_workspace_zero_grad(
    const VAECModel *model,
    VAECModelWorkspace *workspace
);

#endif
