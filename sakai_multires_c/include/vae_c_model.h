#ifndef VAE_C_MODEL_H
#define VAE_C_MODEL_H

#include "vae_c_batchnorm.h"
#include "vae_c_config.h"
#include "vae_c_conv2d.h"
#include "vae_c_dense.h"
#include "vae_c_ops.h"
#include "vae_c_parameter.h"
#include "vae_c_spatial.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


#define VAE_C_MODEL_MAX_ENCODER_CONVS \
    VAE_C_MAX_CONV_DEPTH

#define VAE_C_MODEL_MAX_DECODER_HIDDEN_CONVS \
    VAE_C_MAX_DECODER_HIDDEN_CONVS

#define VAE_C_MODEL_MAX_PARAMETERS 28


typedef enum {
    VAE_C_MODEL_PARAM_KEY_ENCODER_CONV1_WEIGHT = 1,
    VAE_C_MODEL_PARAM_KEY_ENCODER_BN1_GAMMA = 2,
    VAE_C_MODEL_PARAM_KEY_ENCODER_BN1_BETA = 3,
    VAE_C_MODEL_PARAM_KEY_ENCODER_CONV2_WEIGHT = 4,
    VAE_C_MODEL_PARAM_KEY_ENCODER_BN2_GAMMA = 5,
    VAE_C_MODEL_PARAM_KEY_ENCODER_BN2_BETA = 6,
    VAE_C_MODEL_PARAM_KEY_ENCODER_CONV3_WEIGHT = 7,
    VAE_C_MODEL_PARAM_KEY_ENCODER_BN3_GAMMA = 8,
    VAE_C_MODEL_PARAM_KEY_ENCODER_BN3_BETA = 9,
    VAE_C_MODEL_PARAM_KEY_ENCODER_CONV4_WEIGHT = 10,
    VAE_C_MODEL_PARAM_KEY_ENCODER_BN4_GAMMA = 11,
    VAE_C_MODEL_PARAM_KEY_ENCODER_BN4_BETA = 12,
    VAE_C_MODEL_PARAM_KEY_ENCODER_FC_WEIGHT = 13,
    VAE_C_MODEL_PARAM_KEY_ENCODER_FC_BIAS = 14,
    VAE_C_MODEL_PARAM_KEY_DECODER_FC_WEIGHT = 15,
    VAE_C_MODEL_PARAM_KEY_DECODER_FC_BN_GAMMA = 16,
    VAE_C_MODEL_PARAM_KEY_DECODER_FC_BN_BETA = 17,
    VAE_C_MODEL_PARAM_KEY_DECODER_CONV1_WEIGHT = 18,
    VAE_C_MODEL_PARAM_KEY_DECODER_BN1_GAMMA = 19,
    VAE_C_MODEL_PARAM_KEY_DECODER_BN1_BETA = 20,
    VAE_C_MODEL_PARAM_KEY_DECODER_CONV2_WEIGHT = 21,
    VAE_C_MODEL_PARAM_KEY_DECODER_BN2_GAMMA = 22,
    VAE_C_MODEL_PARAM_KEY_DECODER_BN2_BETA = 23,
    VAE_C_MODEL_PARAM_KEY_DECODER_CONV3_WEIGHT = 24,
    VAE_C_MODEL_PARAM_KEY_DECODER_BN3_GAMMA = 25,
    VAE_C_MODEL_PARAM_KEY_DECODER_BN3_BETA = 26,
    VAE_C_MODEL_PARAM_KEY_DECODER_OUTPUT_WEIGHT = 27,
    VAE_C_MODEL_PARAM_KEY_DECODER_OUTPUT_BIAS = 28
} VAECModelParameterKey;


/*
 * PROJECT: this type owns all memory held by its layers and is not
 * copy-safe through plain structure assignment.  Use its lifecycle API.
 * The parameter registry contains non-owning pointers into this object.
 */
typedef struct {
    char profile_id[VAE_C_PROFILE_TEXT_CAP];

    VAECConfig config;
    size_t max_batch_size;

    VAECConv2D encoder_convs[VAE_C_MODEL_MAX_ENCODER_CONVS];
    VAECBatchNorm encoder_batch_norms[VAE_C_MODEL_MAX_ENCODER_CONVS];

    VAECMaxPool2D encoder_max_pool;

    VAECDense encoder_fc;

    VAECDense decoder_fc;
    VAECBatchNorm decoder_fc_batch_norm;

    VAECConv2D
        decoder_hidden_convs[VAE_C_MODEL_MAX_DECODER_HIDDEN_CONVS];

    VAECBatchNorm
        decoder_hidden_batch_norms[
            VAE_C_MODEL_MAX_DECODER_HIDDEN_CONVS
        ];

    VAECConv2D decoder_output_conv;

    VAECParameter *parameters[VAE_C_MODEL_MAX_PARAMETERS];
    size_t parameter_count;

    uint64_t trainable_value_count;
    uint64_t initialization_seed;
    /* PROJECT: monotonic version of model-level forward cache writes. */
    uint64_t vae_cache_generation;

    bool parameters_initialized;
    bool allocated;
} VAECModel;


void vae_c_model_init(VAECModel *model);

void vae_c_model_free(VAECModel *model);

bool vae_c_model_is_empty(const VAECModel *model);

bool vae_c_model_is_allocated(const VAECModel *model);

bool vae_c_model_is_initialized(const VAECModel *model);

/*
 * PAPER: Figure 8 supplies the general convolution, BatchNorm, pooling,
 * fully-connected, and nearest-upsample encoder/decoder structure.
 *
 * PROJECT: allocation follows the validated mini profile, supports depth
 * three and four, assigns stable parameter keys, and builds a complete
 * non-owning parameter registry for the resulting C object.
 */
VAECOpStatus vae_c_model_allocate(
    VAECModel *model,
    const VAECProfile *profile,
    size_t max_batch_size
);

VAECOpStatus vae_c_model_validate(const VAECModel *model);

/*
 * PROJECT: hidden ReLU-path weights use He/Kaiming uniform, while the
 * encoder distribution head and final logits convolution use Xavier
 * uniform.  Stable parameter keys select independent RNG streams.
 */
VAECOpStatus vae_c_model_initialize_parameters_v1(
    VAECModel *model,
    uint64_t base_seed
);

#endif
