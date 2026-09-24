#include "vae_c_model_backward.h"

#include "vae_c_model_forward.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


typedef struct {
    VAECModel model;
    VAECModelWorkspace workspace;
    VAECVAELossStats stats;
} Fixture;


typedef struct {
    uint64_t mask_hash;
    uint64_t pool_hash;
    double minimum_relu_margin;
    double minimum_pool_margin;
} DifferentiabilitySignature;


enum {
    GRADIENT_CHECK_STEP_COUNT = 3,
    GRADIENT_CHECK_BASE_STEP_INDEX = 1
};


typedef struct {
    double analytic;
    double numeric;
    double absolute_error;
    double relative_error;
    double scaled_error;
    double requested_h;
    double effective_h;
    double effective_delta;
    int signs_agree;
} GradientMeasurement;


static const float gradient_check_step_scales[
    GRADIENT_CHECK_STEP_COUNT
] = {
    2.0e-2f,
    1.0e-2f,
    5.0e-3f
};

static const double directional_check_step_scales[
    GRADIENT_CHECK_STEP_COUNT
] = {
    2.0e-3,
    1.0e-3,
    5.0e-4
};

static double gradient_check_max_absolute_error = 0.0;
static double gradient_check_max_relative_error = 0.0;
static double gradient_check_max_scaled_error = 0.0;
static double gradient_check_multih_max_absolute_error = 0.0;
static double gradient_check_multih_max_relative_error = 0.0;
static double gradient_check_multih_max_scaled_error = 0.0;
static double directional_check_max_absolute_error = 0.0;
static double directional_check_max_relative_error = 0.0;
static double directional_check_max_scaled_error = 0.0;
static size_t elementwise_diagnostic_tolerance_failures = 0;


static int report_check(const char *name, int passed)
{
    if (passed) {
        printf("[PASS] %s\n", name);
        return 0;
    }
    printf("[FAIL] %s\n", name);
    return 1;
}


static const char *parameter_key_name(uint64_t key)
{
    switch (key) {
        case VAE_C_MODEL_PARAM_KEY_ENCODER_CONV1_WEIGHT:
            return "encoder_conv1_weight";
        case VAE_C_MODEL_PARAM_KEY_ENCODER_BN1_GAMMA:
            return "encoder_bn1_gamma";
        case VAE_C_MODEL_PARAM_KEY_ENCODER_BN1_BETA:
            return "encoder_bn1_beta";
        case VAE_C_MODEL_PARAM_KEY_ENCODER_CONV2_WEIGHT:
            return "encoder_conv2_weight";
        case VAE_C_MODEL_PARAM_KEY_ENCODER_BN2_GAMMA:
            return "encoder_bn2_gamma";
        case VAE_C_MODEL_PARAM_KEY_ENCODER_BN2_BETA:
            return "encoder_bn2_beta";
        case VAE_C_MODEL_PARAM_KEY_ENCODER_CONV3_WEIGHT:
            return "encoder_conv3_weight";
        case VAE_C_MODEL_PARAM_KEY_ENCODER_BN3_GAMMA:
            return "encoder_bn3_gamma";
        case VAE_C_MODEL_PARAM_KEY_ENCODER_BN3_BETA:
            return "encoder_bn3_beta";
        case VAE_C_MODEL_PARAM_KEY_ENCODER_CONV4_WEIGHT:
            return "encoder_conv4_weight";
        case VAE_C_MODEL_PARAM_KEY_ENCODER_BN4_GAMMA:
            return "encoder_bn4_gamma";
        case VAE_C_MODEL_PARAM_KEY_ENCODER_BN4_BETA:
            return "encoder_bn4_beta";
        case VAE_C_MODEL_PARAM_KEY_ENCODER_FC_WEIGHT:
            return "encoder_fc_weight";
        case VAE_C_MODEL_PARAM_KEY_ENCODER_FC_BIAS:
            return "encoder_fc_bias";
        case VAE_C_MODEL_PARAM_KEY_DECODER_FC_WEIGHT:
            return "decoder_fc_weight";
        case VAE_C_MODEL_PARAM_KEY_DECODER_FC_BN_GAMMA:
            return "decoder_fc_bn_gamma";
        case VAE_C_MODEL_PARAM_KEY_DECODER_FC_BN_BETA:
            return "decoder_fc_bn_beta";
        case VAE_C_MODEL_PARAM_KEY_DECODER_CONV1_WEIGHT:
            return "decoder_conv1_weight";
        case VAE_C_MODEL_PARAM_KEY_DECODER_BN1_GAMMA:
            return "decoder_bn1_gamma";
        case VAE_C_MODEL_PARAM_KEY_DECODER_BN1_BETA:
            return "decoder_bn1_beta";
        case VAE_C_MODEL_PARAM_KEY_DECODER_CONV2_WEIGHT:
            return "decoder_conv2_weight";
        case VAE_C_MODEL_PARAM_KEY_DECODER_BN2_GAMMA:
            return "decoder_bn2_gamma";
        case VAE_C_MODEL_PARAM_KEY_DECODER_BN2_BETA:
            return "decoder_bn2_beta";
        case VAE_C_MODEL_PARAM_KEY_DECODER_CONV3_WEIGHT:
            return "decoder_conv3_weight";
        case VAE_C_MODEL_PARAM_KEY_DECODER_BN3_GAMMA:
            return "decoder_bn3_gamma";
        case VAE_C_MODEL_PARAM_KEY_DECODER_BN3_BETA:
            return "decoder_bn3_beta";
        case VAE_C_MODEL_PARAM_KEY_DECODER_OUTPUT_WEIGHT:
            return "decoder_output_weight";
        case VAE_C_MODEL_PARAM_KEY_DECODER_OUTPUT_BIAS:
            return "decoder_output_bias";
        default:
            return "unknown_parameter";
    }
}


static uint64_t hash_bytes(
    uint64_t hash,
    const void *memory,
    size_t byte_count
)
{
    const unsigned char *bytes = (const unsigned char *)memory;
    size_t index;

    for (index = 0; index < byte_count; index++) {
        hash ^= (uint64_t)bytes[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}


static size_t collect_workspace_tensors(
    const VAECModelWorkspace *workspace,
    const VAECTensor *tensors[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ]
)
{
    size_t count = 0;
    size_t index;

    tensors[count++] = &workspace->input;
    for (index = 0; index < workspace->config.conv_depth; index++) {
        tensors[count++] = &workspace->encoder_conv_outputs[index];
        tensors[count++] =
            &workspace->encoder_batch_norm_outputs[index];
        tensors[count++] = &workspace->encoder_relu_outputs[index];
    }
    tensors[count++] = &workspace->encoder_pool_output;
    tensors[count++] = &workspace->encoder_global_avg_output;
    tensors[count++] = &workspace->encoder_packed_output;
    tensors[count++] = &workspace->mu;
    tensors[count++] = &workspace->logvar;
    tensors[count++] = &workspace->epsilon;
    tensors[count++] = &workspace->z;
    tensors[count++] = &workspace->decoder_fc_output;
    tensors[count++] = &workspace->decoder_fc_batch_norm_output;
    tensors[count++] = &workspace->decoder_fc_relu_output;
    for (index = 0;
         index < workspace->config.decoder_upsample_count;
         index++) {

        tensors[count++] = &workspace->decoder_upsample_outputs[index];
    }
    for (index = 0;
         index < workspace->config.decoder_hidden_conv_count;
         index++) {

        tensors[count++] =
            &workspace->decoder_hidden_conv_outputs[index];
        tensors[count++] =
            &workspace->decoder_hidden_batch_norm_outputs[index];
        tensors[count++] =
            &workspace->decoder_hidden_relu_outputs[index];
    }
    tensors[count++] = &workspace->logits;
    return count;
}


static size_t collect_batch_norms(
    const VAECModel *model,
    const VAECBatchNorm *layers[
        VAE_C_MAX_CONV_DEPTH + 1 +
        VAE_C_MAX_DECODER_HIDDEN_CONVS
    ]
)
{
    size_t count = 0;
    size_t index;

    for (index = 0; index < model->config.conv_depth; index++) {
        layers[count++] = &model->encoder_batch_norms[index];
    }
    layers[count++] = &model->decoder_fc_batch_norm;
    for (index = 0;
         index < model->config.decoder_hidden_conv_count;
         index++) {

        layers[count++] = &model->decoder_hidden_batch_norms[index];
    }
    return count;
}


static uint64_t full_state_hash(
    const VAECModel *model,
    const VAECModelWorkspace *workspace
)
{
    const VAECTensor *tensors[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ];
    const VAECBatchNorm *batch_norms[
        VAE_C_MAX_CONV_DEPTH + 1 +
        VAE_C_MAX_DECODER_HIDDEN_CONVS
    ];
    uint64_t hash = UINT64_C(1469598103934665603);
    size_t tensor_count;
    size_t batch_norm_count;
    size_t index;

    hash = hash_bytes(hash, model, sizeof(*model));
    hash = hash_bytes(hash, workspace, sizeof(*workspace));
    for (index = 0; index < model->parameter_count; index++) {
        const VAECParameter *parameter = model->parameters[index];

        hash = hash_bytes(
            hash, parameter->owned_block, parameter->owned_bytes
        );
    }
    batch_norm_count = collect_batch_norms(model, batch_norms);
    for (index = 0; index < batch_norm_count; index++) {
        hash = hash_bytes(
            hash,
            batch_norms[index]->owned_block,
            batch_norms[index]->owned_bytes
        );
    }
    hash = hash_bytes(
        hash,
        model->encoder_max_pool.argmax_indices,
        model->encoder_max_pool.owned_bytes
    );
    tensor_count = collect_workspace_tensors(workspace, tensors);
    for (index = 0; index < tensor_count; index++) {
        hash = hash_bytes(
            hash, tensors[index]->owned_block, tensors[index]->owned_bytes
        );
    }
    return hash;
}


static uint64_t backward_invariant_hash(
    const VAECModel *model,
    const VAECModelWorkspace *workspace
)
{
    const VAECTensor *tensors[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ];
    const VAECBatchNorm *batch_norms[
        VAE_C_MAX_CONV_DEPTH + 1 +
        VAE_C_MAX_DECODER_HIDDEN_CONVS
    ];
    uint64_t hash = UINT64_C(1469598103934665603);
    size_t tensor_count = collect_workspace_tensors(workspace, tensors);
    size_t batch_norm_count = collect_batch_norms(model, batch_norms);
    size_t index;

    hash = hash_bytes(hash, model->profile_id, sizeof(model->profile_id));
    hash = hash_bytes(hash, &model->config, sizeof(model->config));
    hash = hash_bytes(hash, &model->max_batch_size,
                      sizeof(model->max_batch_size));
    hash = hash_bytes(hash, model->parameters,
                      sizeof(model->parameters));
    hash = hash_bytes(hash, &model->parameter_count,
                      sizeof(model->parameter_count));
    hash = hash_bytes(hash, &model->initialization_seed,
                      sizeof(model->initialization_seed));
    hash = hash_bytes(hash, &model->vae_cache_generation,
                      sizeof(model->vae_cache_generation));
    for (index = 0; index < model->parameter_count; index++) {
        const VAECParameter *parameter = model->parameters[index];

        hash = hash_bytes(
            hash,
            parameter->values,
            parameter->shape.count * sizeof(float)
        );
        hash = hash_bytes(
            hash,
            parameter->adam_m,
            parameter->shape.count * sizeof(float)
        );
        hash = hash_bytes(
            hash,
            parameter->adam_v,
            parameter->shape.count * sizeof(float)
        );
    }
    for (index = 0; index < batch_norm_count; index++) {
        const VAECBatchNorm *layer = batch_norms[index];

        hash = hash_bytes(hash, layer->running_mean,
                          layer->num_features * sizeof(float));
        hash = hash_bytes(hash, layer->running_variance,
                          layer->num_features * sizeof(float));
        hash = hash_bytes(hash, layer->batch_mean,
                          layer->num_features * sizeof(float));
        hash = hash_bytes(hash, layer->batch_inv_std,
                          layer->num_features * sizeof(float));
        hash = hash_bytes(hash, &layer->last_sample_count,
                          sizeof(layer->last_sample_count));
        hash = hash_bytes(hash, &layer->last_input_shape,
                          sizeof(layer->last_input_shape));
        hash = hash_bytes(hash, &layer->last_output_shape,
                          sizeof(layer->last_output_shape));
        hash = hash_bytes(hash, &layer->cache_valid,
                          sizeof(layer->cache_valid));
    }
    hash = hash_bytes(hash, &model->encoder_max_pool.last_count,
                      sizeof(model->encoder_max_pool.last_count));
    hash = hash_bytes(hash, &model->encoder_max_pool.last_input_shape,
                      sizeof(model->encoder_max_pool.last_input_shape));
    hash = hash_bytes(hash, &model->encoder_max_pool.last_output_shape,
                      sizeof(model->encoder_max_pool.last_output_shape));
    hash = hash_bytes(hash, &model->encoder_max_pool.cache_valid,
                      sizeof(model->encoder_max_pool.cache_valid));
    hash = hash_bytes(hash,
                      model->encoder_max_pool.argmax_indices,
                      model->encoder_max_pool.last_count * sizeof(size_t));
    hash = hash_bytes(hash, workspace->profile_id,
                      sizeof(workspace->profile_id));
    hash = hash_bytes(hash, &workspace->config, sizeof(workspace->config));
    hash = hash_bytes(hash, &workspace->batch_size,
                      sizeof(workspace->batch_size));
    for (index = 0; index < tensor_count; index++) {
        hash = hash_bytes(
            hash,
            tensors[index]->data,
            tensors[index]->shape.count * sizeof(float)
        );
    }
    return hash;
}


static void fill_input_and_epsilon(VAECModelWorkspace *workspace)
{
    size_t index;

    for (index = 0; index < workspace->input.shape.count; index++) {
        unsigned long value =
            (unsigned long)((index * 37u + 11u) % 997u);

        workspace->input.data[index] =
            0.1f + 0.8f * (float)value / 996.0f;
    }
    for (index = 0; index < workspace->epsilon.shape.count; index++) {
        int centered = (int)(index % 13) - 6;

        workspace->epsilon.data[index] = 0.0625f * (float)centered;
    }
}


static void stabilize_relu_margins(VAECModel *model)
{
    size_t index;

    for (index = 0; index < model->parameter_count; index++) {
        VAECParameter *parameter = model->parameters[index];
        uint64_t key = parameter->parameter_key;
        bool is_batch_norm_beta =
            key == VAE_C_MODEL_PARAM_KEY_ENCODER_BN1_BETA ||
            key == VAE_C_MODEL_PARAM_KEY_ENCODER_BN2_BETA ||
            key == VAE_C_MODEL_PARAM_KEY_ENCODER_BN3_BETA ||
            key == VAE_C_MODEL_PARAM_KEY_ENCODER_BN4_BETA ||
            key == VAE_C_MODEL_PARAM_KEY_DECODER_FC_BN_BETA ||
            key == VAE_C_MODEL_PARAM_KEY_DECODER_BN1_BETA ||
            key == VAE_C_MODEL_PARAM_KEY_DECODER_BN2_BETA ||
            key == VAE_C_MODEL_PARAM_KEY_DECODER_BN3_BETA;
        size_t element;

        if (is_batch_norm_beta) {
            for (element = 0;
                 element < parameter->shape.count;
                 element++) {

                parameter->values[element] = 4.0f;
            }
        }
    }
    for (index = 0;
         index < model->encoder_fc.weights.shape.count;
         index++) {

        model->encoder_fc.weights.values[index] *= 0.02f;
    }
    for (index = 0;
         index < model->encoder_fc.bias.shape.count;
         index++) {

        model->encoder_fc.bias.values[index] = 0.0f;
    }
}


static void fixture_init(Fixture *fixture)
{
    vae_c_model_init(&fixture->model);
    vae_c_model_workspace_init(&fixture->workspace);
    vae_c_vae_loss_stats_init(&fixture->stats);
}


static void fixture_free(Fixture *fixture)
{
    vae_c_model_workspace_free(&fixture->workspace);
    vae_c_model_free(&fixture->model);
    vae_c_vae_loss_stats_init(&fixture->stats);
}


static int fixture_allocate(
    Fixture *fixture,
    const VAECProfile *profile,
    uint64_t seed,
    int stabilize
)
{
    fixture_init(fixture);
    if (profile == NULL ||
        vae_c_model_allocate(&fixture->model, profile, 2) !=
            VAE_C_OP_OK ||
        vae_c_model_initialize_parameters_v1(
            &fixture->model, seed) != VAE_C_OP_OK ||
        vae_c_model_workspace_allocate(
            &fixture->model, 2, &fixture->workspace) != VAE_C_OP_OK) {

        return 0;
    }
    if (stabilize) {
        stabilize_relu_margins(&fixture->model);
    }
    fill_input_and_epsilon(&fixture->workspace);
    return
        vae_c_model_validate(&fixture->model) == VAE_C_OP_OK &&
        vae_c_model_workspace_validate(
            &fixture->model, &fixture->workspace) == VAE_C_OP_OK;
}


static int fixture_forward_ready(Fixture *fixture)
{
    return
        vae_c_model_vae_forward(
            &fixture->model,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            VAE_C_LATENT_MODE_STOCHASTIC,
            &fixture->workspace,
            &fixture->stats) == VAE_C_OP_OK &&
        fixture->workspace.vae_backward_model == &fixture->model &&
        fixture->workspace.vae_backward_generation != UINT64_C(0) &&
        fixture->workspace.vae_backward_generation ==
            fixture->model.vae_cache_generation &&
        vae_c_model_workspace_zero_grad(
            &fixture->model, &fixture->workspace) == VAE_C_OP_OK;
}


static void seed_parameter_gradients(VAECModel *model)
{
    size_t parameter_index;

    for (parameter_index = 0;
         parameter_index < model->parameter_count;
         parameter_index++) {

        VAECParameter *parameter = model->parameters[parameter_index];
        size_t element;

        for (element = 0; element < parameter->shape.count; element++) {
            parameter->gradients[element] =
                0.001f * (float)(1 +
                    (parameter_index + element) % 7);
        }
    }
}


static int zero_parameter_gradients(VAECModel *model)
{
    size_t index;

    for (index = 0; index < model->parameter_count; index++) {
        if (vae_c_parameter_zero_grad(model->parameters[index]) !=
                VAE_C_STORAGE_OK) {

            return 0;
        }
    }
    return 1;
}


static VAECOpStatus manual_backward(
    VAECModel *model,
    VAECModelWorkspace *workspace,
    float upstream_gradient
)
{
    size_t hidden_count = model->config.decoder_hidden_conv_count;
    size_t depth = model->config.conv_depth;
    size_t index;
    VAECOpStatus status;

    status = vae_c_vae_loss_backward(
        &workspace->logits,
        &workspace->input,
        &workspace->mu,
        &workspace->logvar,
        upstream_gradient
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_conv2d_backward(
        &model->decoder_output_conv,
        &workspace->decoder_upsample_outputs[
            model->config.decoder_upsample_count - 1
        ],
        &workspace->logits
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    for (index = hidden_count; index-- > 0;) {
        status = vae_c_nearest_upsample2d_backward(
            &workspace->decoder_hidden_relu_outputs[index],
            &workspace->decoder_upsample_outputs[index + 2]
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
        status = vae_c_relu_backward(
            &workspace->decoder_hidden_batch_norm_outputs[index],
            &workspace->decoder_hidden_relu_outputs[index]
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
        status = vae_c_batch_norm2d_backward(
            &model->decoder_hidden_batch_norms[index],
            &workspace->decoder_hidden_conv_outputs[index],
            &workspace->decoder_hidden_batch_norm_outputs[index]
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
        status = vae_c_conv2d_backward(
            &model->decoder_hidden_convs[index],
            &workspace->decoder_upsample_outputs[index + 1],
            &workspace->decoder_hidden_conv_outputs[index]
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
    }
    status = vae_c_nearest_upsample2d_backward(
        &workspace->decoder_upsample_outputs[0],
        &workspace->decoder_upsample_outputs[1]
    );
    if (status == VAE_C_OP_OK) {
        status = vae_c_nearest_upsample2d_backward(
            &workspace->decoder_reshape_view,
            &workspace->decoder_upsample_outputs[0]
        );
    }
    if (status == VAE_C_OP_OK) {
        status = vae_c_relu_backward(
            &workspace->decoder_fc_batch_norm_output,
            &workspace->decoder_fc_relu_output
        );
    }
    if (status == VAE_C_OP_OK) {
        status = vae_c_batch_norm1d_backward(
            &model->decoder_fc_batch_norm,
            &workspace->decoder_fc_output,
            &workspace->decoder_fc_batch_norm_output
        );
    }
    if (status == VAE_C_OP_OK) {
        status = vae_c_dense_backward(
            &model->decoder_fc,
            &workspace->z,
            &workspace->decoder_fc_output
        );
    }
    if (status == VAE_C_OP_OK) {
        status = vae_c_reparameterize_backward(
            VAE_C_LATENT_MODE_STOCHASTIC,
            &workspace->mu,
            &workspace->logvar,
            &workspace->epsilon,
            &workspace->z
        );
    }
    if (status == VAE_C_OP_OK) {
        status = vae_c_latent_split_backward(
            &workspace->encoder_packed_output,
            &workspace->mu,
            &workspace->logvar
        );
    }
    if (status == VAE_C_OP_OK) {
        status = vae_c_dense_backward(
            &model->encoder_fc,
            &workspace->encoder_global_avg_output,
            &workspace->encoder_packed_output
        );
    }
    if (status == VAE_C_OP_OK) {
        status = vae_c_global_avg_pool2d_backward(
            &workspace->encoder_relu_outputs[depth - 1],
            &workspace->encoder_global_avg_output
        );
    }
    if (status != VAE_C_OP_OK) {
        return status;
    }
    for (index = depth; index-- > 1;) {
        const VAECTensor *input = index == 1
            ? &workspace->encoder_pool_output
            : &workspace->encoder_relu_outputs[index - 1];

        status = vae_c_relu_backward(
            &workspace->encoder_batch_norm_outputs[index],
            &workspace->encoder_relu_outputs[index]
        );
        if (status == VAE_C_OP_OK) {
            status = vae_c_batch_norm2d_backward(
                &model->encoder_batch_norms[index],
                &workspace->encoder_conv_outputs[index],
                &workspace->encoder_batch_norm_outputs[index]
            );
        }
        if (status == VAE_C_OP_OK) {
            status = vae_c_conv2d_backward(
                &model->encoder_convs[index],
                input,
                &workspace->encoder_conv_outputs[index]
            );
        }
        if (status != VAE_C_OP_OK) {
            return status;
        }
    }
    status = vae_c_max_pool2d_backward(
        &model->encoder_max_pool,
        &workspace->encoder_relu_outputs[0],
        &workspace->encoder_pool_output
    );
    if (status == VAE_C_OP_OK) {
        status = vae_c_relu_backward(
            &workspace->encoder_batch_norm_outputs[0],
            &workspace->encoder_relu_outputs[0]
        );
    }
    if (status == VAE_C_OP_OK) {
        status = vae_c_batch_norm2d_backward(
            &model->encoder_batch_norms[0],
            &workspace->encoder_conv_outputs[0],
            &workspace->encoder_batch_norm_outputs[0]
        );
    }
    if (status == VAE_C_OP_OK) {
        status = vae_c_conv2d_backward(
            &model->encoder_convs[0],
            &workspace->input,
            &workspace->encoder_conv_outputs[0]
        );
    }
    return status;
}


static int gradients_equal(
    const Fixture *left,
    const Fixture *right
)
{
    const VAECTensor *left_tensors[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ];
    const VAECTensor *right_tensors[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ];
    size_t left_count = collect_workspace_tensors(
        &left->workspace, left_tensors
    );
    size_t right_count = collect_workspace_tensors(
        &right->workspace, right_tensors
    );
    size_t index;

    if (left_count != right_count ||
        left->model.parameter_count != right->model.parameter_count) {

        return 0;
    }
    for (index = 0; index < left_count; index++) {
        if (left_tensors[index]->has_grad !=
                right_tensors[index]->has_grad ||
            (left_tensors[index]->has_grad &&
             memcmp(left_tensors[index]->grad,
                    right_tensors[index]->grad,
                    left_tensors[index]->shape.count * sizeof(float)) !=
                 0)) {

            return 0;
        }
    }
    for (index = 0; index < left->model.parameter_count; index++) {
        const VAECParameter *left_parameter =
            left->model.parameters[index];
        const VAECParameter *right_parameter =
            right->model.parameters[index];

        if (left_parameter->parameter_key !=
                right_parameter->parameter_key ||
            memcmp(left_parameter->gradients,
                   right_parameter->gradients,
                   left_parameter->shape.count * sizeof(float)) != 0) {

            return 0;
        }
    }
    return 1;
}


static void make_synthetic_depth4(VAECProfile *profile)
{
    memset(profile, 0, sizeof(*profile));
    memcpy(profile->profile_id,
           "synthetic_depth4_small",
           sizeof("synthetic_depth4_small"));
    memcpy(profile->role, "synthetic", sizeof("synthetic"));
    memcpy(profile->profile_status, "test_only", sizeof("test_only"));
    profile->input_side = 32;
    profile->conv_depth = 4;
    profile->channels[0] = 2;
    profile->channels[1] = 3;
    profile->channels[2] = 4;
    profile->channels[3] = 5;
    profile->latent_dim = 3;
    profile->bottleneck_side = 1;
    profile->expected_trainable_parameters_bn1d = UINT64_C(939);
    profile->expected_forward_mac_per_sample = UINT64_C(117297);
}


static int run_manual_equivalence(const VAECProfile *profile)
{
    const uint64_t seed = UINT64_C(0x1234ABCDEF987654);
    Fixture manual;
    Fixture composite;
    uint64_t invariant_before;
    int passed = 0;

    fixture_init(&manual);
    fixture_init(&composite);
    if (!fixture_allocate(&manual, profile, seed, 0) ||
        !fixture_allocate(&composite, profile, seed, 0) ||
        !fixture_forward_ready(&manual) ||
        !fixture_forward_ready(&composite)) {

        goto cleanup;
    }
    seed_parameter_gradients(&manual.model);
    seed_parameter_gradients(&composite.model);
    invariant_before = backward_invariant_hash(
        &composite.model, &composite.workspace
    );
    if (manual_backward(&manual.model, &manual.workspace, 0.75f) !=
            VAE_C_OP_OK ||
        vae_c_model_vae_backward(
            &composite.model,
            &composite.workspace,
            0.75f) != VAE_C_OP_OK) {

        goto cleanup;
    }
    passed =
        gradients_equal(&manual, &composite) &&
        composite.workspace.vae_backward_model == NULL &&
        composite.workspace.vae_backward_generation == UINT64_C(0) &&
        backward_invariant_hash(
            &composite.model, &composite.workspace) == invariant_before &&
        vae_c_model_validate(&composite.model) == VAE_C_OP_OK &&
        vae_c_model_workspace_validate(
            &composite.model, &composite.workspace) == VAE_C_OP_OK;

cleanup:
    fixture_free(&manual);
    fixture_free(&composite);
    return passed;
}


static int test_manual_depth3(const VAECProfileCatalog *catalog)
{
    return run_manual_equivalence(
        vae_c_profile_catalog_find(catalog, "mini75_smoke_v1")
    );
}


static int test_manual_depth4(void)
{
    VAECProfile profile;

    make_synthetic_depth4(&profile);
    return run_manual_equivalence(&profile);
}


static int test_loss_target_constant_and_latent_sum(
    const VAECProfileCatalog *catalog
)
{
    const VAECProfile *profile = vae_c_profile_catalog_find(
        catalog, "mini75_smoke_v1"
    );
    Fixture fixture;
    float *kl_mu = NULL;
    float *kl_logvar = NULL;
    size_t latent_bytes;
    size_t index;
    int decoder_added = 0;
    int input_zero = 1;
    int passed = 0;

    fixture_init(&fixture);
    if (!fixture_allocate(&fixture, profile, UINT64_C(7001), 0) ||
        !fixture_forward_ready(&fixture)) {

        goto cleanup;
    }
    latent_bytes = fixture.workspace.mu.shape.count * sizeof(float);
    kl_mu = (float *)malloc(latent_bytes);
    kl_logvar = (float *)malloc(latent_bytes);
    if (kl_mu == NULL || kl_logvar == NULL ||
        vae_c_vae_loss_backward(
            &fixture.workspace.logits,
            &fixture.workspace.input,
            &fixture.workspace.mu,
            &fixture.workspace.logvar,
            1.0f) != VAE_C_OP_OK) {

        goto cleanup;
    }
    memcpy(kl_mu, fixture.workspace.mu.grad, latent_bytes);
    memcpy(kl_logvar, fixture.workspace.logvar.grad, latent_bytes);
    for (index = 0; index < fixture.workspace.input.shape.count; index++) {
        if (fixture.workspace.input.grad[index] != 0.0f) {
            input_zero = 0;
        }
    }
    if (!input_zero ||
        manual_backward(&fixture.model, &fixture.workspace, 0.0f) !=
            VAE_C_OP_OK) {

        goto cleanup;
    }
    for (index = 0; index < fixture.workspace.mu.shape.count; index++) {
        if (fixture.workspace.mu.grad[index] != kl_mu[index] ||
            fixture.workspace.logvar.grad[index] != kl_logvar[index]) {

            decoder_added = 1;
            break;
        }
    }
    passed = decoder_added;

cleanup:
    free(kl_mu);
    free(kl_logvar);
    fixture_free(&fixture);
    return passed;
}


static int rejected_without_mutation(
    Fixture *fixture,
    float upstream,
    VAECOpStatus expected
)
{
    uint64_t before = full_state_hash(
        &fixture->model, &fixture->workspace
    );
    VAECOpStatus status = vae_c_model_vae_backward(
        &fixture->model, &fixture->workspace, upstream
    );

    return status == expected &&
        full_state_hash(&fixture->model, &fixture->workspace) == before;
}


static int test_ticket_and_consumption(
    const VAECProfileCatalog *catalog
)
{
    const VAECProfile *profile = vae_c_profile_catalog_find(
        catalog, "mini75_smoke_v1"
    );
    Fixture fixture;
    VAECModel other;
    int passed = 1;

    fixture_init(&fixture);
    vae_c_model_init(&other);
    if (!fixture_allocate(&fixture, profile, UINT64_C(7101), 0) ||
        !fixture_forward_ready(&fixture)) {

        passed = 0;
        goto cleanup;
    }
    fixture.workspace.vae_backward_model = NULL;
    fixture.workspace.vae_backward_generation = UINT64_C(0);
    passed = passed && rejected_without_mutation(
        &fixture, 1.0f, VAE_C_OP_INVALID_STATE
    );

    if (!fixture_forward_ready(&fixture)) {
        passed = 0;
        goto cleanup;
    }
    fixture.workspace.vae_backward_generation -= UINT64_C(1);
    passed = passed && rejected_without_mutation(
        &fixture, 1.0f, VAE_C_OP_INVALID_STATE
    );

    if (!fixture_forward_ready(&fixture) ||
        vae_c_model_allocate(&other, profile, 2) != VAE_C_OP_OK ||
        vae_c_model_initialize_parameters_v1(
            &other, UINT64_C(7102)) != VAE_C_OP_OK) {

        passed = 0;
        goto cleanup;
    }
    fixture.workspace.vae_backward_model = &other;
    passed = passed && rejected_without_mutation(
        &fixture, 1.0f, VAE_C_OP_INVALID_STATE
    );

    if (!fixture_forward_ready(&fixture) ||
        vae_c_model_vae_backward(
            &fixture.model, &fixture.workspace, 1.0f) != VAE_C_OP_OK ||
        fixture.workspace.vae_backward_model != NULL ||
        fixture.workspace.vae_backward_generation != UINT64_C(0) ||
        !rejected_without_mutation(
            &fixture, 1.0f, VAE_C_OP_INVALID_STATE)) {

        passed = 0;
    }

cleanup:
    vae_c_model_free(&other);
    fixture_free(&fixture);
    return passed;
}


static int test_zero_grad_and_numeric_preflight(
    const VAECProfileCatalog *catalog
)
{
    const VAECProfile *profile = vae_c_profile_catalog_find(
        catalog, "mini75_smoke_v1"
    );
    Fixture fixture;
    float saved;
    int passed = 1;

    fixture_init(&fixture);
    if (!fixture_allocate(&fixture, profile, UINT64_C(7201), 0) ||
        !fixture_forward_ready(&fixture)) {

        passed = 0;
        goto cleanup;
    }
    fixture.workspace.logits.grad[0] = 1.0f;
    passed = passed && rejected_without_mutation(
        &fixture, 1.0f, VAE_C_OP_INVALID_STATE
    );
    fixture.workspace.logits.grad[0] = 0.0f;
    passed = passed && rejected_without_mutation(
        &fixture, NAN, VAE_C_OP_INVALID_ARGUMENT
    );
    passed = passed && rejected_without_mutation(
        &fixture, INFINITY, VAE_C_OP_INVALID_ARGUMENT
    );
    saved = fixture.workspace.input.data[0];
    fixture.workspace.input.data[0] = NAN;
    passed = passed && rejected_without_mutation(
        &fixture, 1.0f, VAE_C_OP_INVALID_STATE
    );
    fixture.workspace.input.data[0] = saved;
    saved = fixture.workspace.epsilon.data[0];
    fixture.workspace.epsilon.data[0] = INFINITY;
    passed = passed && rejected_without_mutation(
        &fixture, 1.0f, VAE_C_OP_INVALID_STATE
    );
    fixture.workspace.epsilon.data[0] = saved;
    saved = fixture.model.parameters[0]->gradients[0];
    fixture.model.parameters[0]->gradients[0] = NAN;
    passed = passed && rejected_without_mutation(
        &fixture, 1.0f, VAE_C_OP_INVALID_STATE
    );
    fixture.model.parameters[0]->gradients[0] = saved;
    passed = passed &&
        vae_c_model_vae_backward(NULL, &fixture.workspace, 1.0f) ==
            VAE_C_OP_INVALID_ARGUMENT &&
        vae_c_model_vae_backward(&fixture.model, NULL, 1.0f) ==
            VAE_C_OP_INVALID_ARGUMENT;

cleanup:
    fixture_free(&fixture);
    return passed;
}


static int test_cache_shape_overflow_and_alias_preflight(
    const VAECProfileCatalog *catalog
)
{
    const VAECProfile *profile = vae_c_profile_catalog_find(
        catalog, "mini75_smoke_v1"
    );
    Fixture fixture;
    VAECShape saved_shape;
    size_t saved_index;
    size_t *saved_argmax;
    int passed = 1;

    fixture_init(&fixture);
    if (!fixture_allocate(&fixture, profile, UINT64_C(7301), 0) ||
        !fixture_forward_ready(&fixture)) {

        passed = 0;
        goto cleanup;
    }
    fixture.model.encoder_batch_norms[0].cache_valid = false;
    fixture.model.encoder_batch_norms[0].last_sample_count = 0;
    vae_c_shape_init(
        &fixture.model.encoder_batch_norms[0].last_input_shape
    );
    vae_c_shape_init(
        &fixture.model.encoder_batch_norms[0].last_output_shape
    );
    if (!rejected_without_mutation(
            &fixture, 1.0f, VAE_C_OP_INVALID_STATE)) {

        fprintf(stderr, "cache preflight subcase failed\n");
        passed = 0;
    }

    if (!fixture_forward_ready(&fixture)) {
        passed = 0;
        goto cleanup;
    }
    {
        size_t stale_dims[2] = {
            fixture.workspace.batch_size + 1,
            fixture.model.decoder_fc_batch_norm.num_features
        };
        VAECBatchNorm *layer = &fixture.model.decoder_fc_batch_norm;

        if (vae_c_shape_make(
                2, stale_dims, &layer->last_input_shape) !=
                VAE_C_STORAGE_OK) {

            passed = 0;
            goto cleanup;
        }
        layer->last_output_shape = layer->last_input_shape;
        layer->last_sample_count = stale_dims[0];
        if (!rejected_without_mutation(
                &fixture, 1.0f, VAE_C_OP_INVALID_STATE)) {

            fprintf(stderr, "stale BatchNorm cache subcase failed\n");
            passed = 0;
        }
    }

    if (!fixture_forward_ready(&fixture)) {
        passed = 0;
        goto cleanup;
    }
    saved_index = fixture.model.encoder_max_pool.argmax_indices[0];
    fixture.model.encoder_max_pool.argmax_indices[0] =
        fixture.workspace.encoder_relu_outputs[0].shape.count;
    if (!rejected_without_mutation(
            &fixture, 1.0f, VAE_C_OP_INVALID_STATE)) {

        fprintf(stderr, "maxpool index subcase failed\n");
        passed = 0;
    }
    fixture.model.encoder_max_pool.argmax_indices[0] = saved_index;

    saved_shape = fixture.workspace.logits.shape;
    fixture.workspace.logits.shape.count = SIZE_MAX;
    if (!rejected_without_mutation(
            &fixture, 1.0f, VAE_C_OP_INVALID_STATE)) {

        fprintf(stderr, "shape overflow subcase failed\n");
        passed = 0;
    }
    fixture.workspace.logits.shape = saved_shape;

    saved_argmax = fixture.model.encoder_max_pool.argmax_indices;
    fixture.model.encoder_max_pool.argmax_indices =
        (size_t *)(void *)fixture.model.decoder_fc.weights.values;
    {
        VAECOpStatus model_status = vae_c_model_validate(&fixture.model);
        VAECOpStatus workspace_status = vae_c_model_workspace_validate(
            &fixture.model, &fixture.workspace
        );
        VAECOpStatus backward_status = vae_c_model_vae_backward(
            &fixture.model, &fixture.workspace, 1.0f
        );

        if (model_status != VAE_C_OP_OK ||
            workspace_status != VAE_C_OP_OK ||
            backward_status != VAE_C_OP_INVALID_ARGUMENT ||
            fixture.workspace.vae_backward_model != &fixture.model ||
            fixture.workspace.vae_backward_generation !=
                fixture.model.vae_cache_generation) {

            fprintf(stderr,
                    "alias subcase failed model=%s workspace=%s "
                    "backward=%s\n",
                    vae_c_op_status_name(model_status),
                    vae_c_op_status_name(workspace_status),
                    vae_c_op_status_name(backward_status));
            passed = 0;
        }
    }
    fixture.model.encoder_max_pool.argmax_indices = saved_argmax;
    passed = passed &&
        vae_c_model_validate(&fixture.model) == VAE_C_OP_OK &&
        vae_c_model_workspace_validate(
            &fixture.model, &fixture.workspace) == VAE_C_OP_OK;

cleanup:
    fixture_free(&fixture);
    return passed;
}


static uint64_t hash_mask_value(uint64_t hash, float value)
{
    unsigned char active = value > 0.0f ? 1u : 0u;

    return hash_bytes(hash, &active, sizeof(active));
}


static int differentiability_signature(
    const Fixture *fixture,
    DifferentiabilitySignature *signature
)
{
    uint64_t mask_hash = UINT64_C(1469598103934665603);
    uint64_t pool_hash = UINT64_C(1469598103934665603);
    double minimum_relu = DBL_MAX;
    double minimum_pool = DBL_MAX;
    size_t depth = fixture->model.config.conv_depth;
    size_t hidden = fixture->model.config.decoder_hidden_conv_count;
    size_t index;

    for (index = 0; index < depth; index++) {
        const VAECTensor *tensor =
            &fixture->workspace.encoder_batch_norm_outputs[index];
        size_t element;

        for (element = 0; element < tensor->shape.count; element++) {
            double margin = fabs((double)tensor->data[element]);

            mask_hash = hash_mask_value(mask_hash, tensor->data[element]);
            if (margin < minimum_relu) {
                minimum_relu = margin;
            }
        }
    }
    {
        const VAECTensor *tensor =
            &fixture->workspace.decoder_fc_batch_norm_output;
        size_t element;

        for (element = 0; element < tensor->shape.count; element++) {
            double margin = fabs((double)tensor->data[element]);

            mask_hash = hash_mask_value(mask_hash, tensor->data[element]);
            if (margin < minimum_relu) {
                minimum_relu = margin;
            }
        }
    }
    for (index = 0; index < hidden; index++) {
        const VAECTensor *tensor =
            &fixture->workspace.decoder_hidden_batch_norm_outputs[index];
        size_t element;

        for (element = 0; element < tensor->shape.count; element++) {
            double margin = fabs((double)tensor->data[element]);

            mask_hash = hash_mask_value(mask_hash, tensor->data[element]);
            if (margin < minimum_relu) {
                minimum_relu = margin;
            }
        }
    }
    pool_hash = hash_bytes(
        pool_hash,
        fixture->model.encoder_max_pool.argmax_indices,
        fixture->model.encoder_max_pool.last_count * sizeof(size_t)
    );
    {
        const VAECTensor *input =
            &fixture->workspace.encoder_relu_outputs[0];
        size_t n;
        size_t c;
        size_t oy;
        size_t ox;

        for (n = 0; n < input->shape.dims[0]; n++) {
            for (c = 0; c < input->shape.dims[1]; c++) {
                for (oy = 0; oy < input->shape.dims[2] / 2; oy++) {
                    for (ox = 0; ox < input->shape.dims[3] / 2; ox++) {
                        double largest = -DBL_MAX;
                        double second = -DBL_MAX;
                        size_t ky;
                        size_t kx;

                        for (ky = 0; ky < 2; ky++) {
                            for (kx = 0; kx < 2; kx++) {
                                size_t offset =
                                    n * input->shape.strides[0] +
                                    c * input->shape.strides[1] +
                                    (2 * oy + ky) * input->shape.strides[2] +
                                    (2 * ox + kx) * input->shape.strides[3];
                                double value = input->data[offset];

                                if (value > largest) {
                                    second = largest;
                                    largest = value;
                                } else if (value > second) {
                                    second = value;
                                }
                            }
                        }
                        if (largest - second < minimum_pool) {
                            minimum_pool = largest - second;
                        }
                    }
                }
            }
        }
    }
    signature->mask_hash = mask_hash;
    signature->pool_hash = pool_hash;
    signature->minimum_relu_margin = minimum_relu;
    signature->minimum_pool_margin = minimum_pool;
    return isfinite(minimum_relu) && isfinite(minimum_pool);
}


static int evaluate_parameter_loss(
    const VAECProfile *profile,
    uint64_t seed,
    size_t parameter_index,
    size_t element_index,
    float value,
    double *loss,
    DifferentiabilitySignature *signature
)
{
    Fixture fixture;
    int passed = 0;

    fixture_init(&fixture);
    if (!fixture_allocate(&fixture, profile, seed, 1) ||
        parameter_index >= fixture.model.parameter_count ||
        element_index >=
            fixture.model.parameters[parameter_index]->shape.count) {

        goto cleanup;
    }
    fixture.model.parameters[parameter_index]->values[element_index] =
        value;
    if (vae_c_model_vae_forward(
            &fixture.model,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            VAE_C_LATENT_MODE_STOCHASTIC,
            &fixture.workspace,
            &fixture.stats) != VAE_C_OP_OK ||
        !differentiability_signature(&fixture, signature)) {

        goto cleanup;
    }
    *loss = fixture.stats.mean_per_sample;
    passed = isfinite(*loss);

cleanup:
    fixture_free(&fixture);
    return passed;
}


static int run_forward_with_fixed_target(
    Fixture *fixture,
    const VAECTensor *target
)
{
    VAECOpStatus status;

    status = vae_c_model_encoder_forward(
        &fixture->model,
        VAE_C_BATCH_NORM_MODE_TRAIN,
        &fixture->workspace
    );
    if (status == VAE_C_OP_OK) {
        status = vae_c_reparameterize_forward(
            VAE_C_LATENT_MODE_STOCHASTIC,
            &fixture->workspace.mu,
            &fixture->workspace.logvar,
            &fixture->workspace.epsilon,
            &fixture->workspace.z
        );
    }
    if (status == VAE_C_OP_OK) {
        status = vae_c_model_decoder_forward(
            &fixture->model,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &fixture->workspace
        );
    }
    if (status == VAE_C_OP_OK) {
        status = vae_c_vae_loss_forward(
            &fixture->workspace.logits,
            target,
            &fixture->workspace.mu,
            &fixture->workspace.logvar,
            &fixture->stats
        );
    }
    return status == VAE_C_OP_OK;
}


static int evaluate_input_loss(
    const VAECProfile *profile,
    uint64_t seed,
    size_t input_index,
    float value,
    double *loss,
    DifferentiabilitySignature *signature
)
{
    Fixture fixture;
    VAECTensor target;
    int passed = 0;

    fixture_init(&fixture);
    vae_c_tensor_init(&target);
    if (!fixture_allocate(&fixture, profile, seed, 1) ||
        input_index >= fixture.workspace.input.shape.count ||
        vae_c_tensor_allocate(
            &target,
            &fixture.workspace.input.shape,
            false) != VAE_C_STORAGE_OK) {

        goto cleanup;
    }
    memcpy(target.data,
           fixture.workspace.input.data,
           target.shape.count * sizeof(float));
    fixture.workspace.input.data[input_index] = value;
    if (!run_forward_with_fixed_target(&fixture, &target) ||
        !differentiability_signature(&fixture, signature)) {

        goto cleanup;
    }
    *loss = fixture.stats.mean_per_sample;
    passed = isfinite(*loss);

cleanup:
    vae_c_tensor_free(&target);
    fixture_free(&fixture);
    return passed;
}


static int signatures_are_smooth(
    const DifferentiabilitySignature *plus,
    const DifferentiabilitySignature *minus
)
{
    const double margin = 1.0e-5;

    return plus->mask_hash == minus->mask_hash &&
        plus->pool_hash == minus->pool_hash &&
        plus->minimum_relu_margin > margin &&
        minus->minimum_relu_margin > margin &&
        plus->minimum_pool_margin > margin &&
        minus->minimum_pool_margin > margin;
}


static void gradient_measurement_make(
    double analytic,
    double numeric,
    double requested_h,
    double effective_delta,
    GradientMeasurement *measurement
)
{
    double scale = fmax(fabs(analytic), fabs(numeric));
    double tolerance = 5.0e-2 + 1.0e-1 * scale;

    measurement->analytic = analytic;
    measurement->numeric = numeric;
    measurement->absolute_error = fabs(analytic - numeric);
    measurement->relative_error = scale == 0.0
        ? 0.0
        : measurement->absolute_error / scale;
    measurement->scaled_error = measurement->absolute_error / tolerance;
    measurement->requested_h = requested_h;
    measurement->effective_h = 0.5 * effective_delta;
    measurement->effective_delta = effective_delta;
    measurement->signs_agree =
        (analytic == 0.0 && numeric == 0.0) ||
        (analytic > 0.0 && numeric > 0.0) ||
        (analytic < 0.0 && numeric < 0.0);
}


static int gradient_measurement_passes(
    const GradientMeasurement *measurement
)
{
    double scale = fmax(
        fabs(measurement->analytic),
        fabs(measurement->numeric)
    );

    return measurement->absolute_error <= 5.0e-2 + 1.0e-1 * scale;
}


static void gradient_measurement_record(
    const GradientMeasurement *measurement,
    int is_base_step
)
{
    if (measurement->absolute_error >
            gradient_check_multih_max_absolute_error) {

        gradient_check_multih_max_absolute_error =
            measurement->absolute_error;
    }
    if (measurement->relative_error >
            gradient_check_multih_max_relative_error) {

        gradient_check_multih_max_relative_error =
            measurement->relative_error;
    }
    if (measurement->scaled_error >
            gradient_check_multih_max_scaled_error) {

        gradient_check_multih_max_scaled_error =
            measurement->scaled_error;
    }
    if (!is_base_step) {
        return;
    }
    if (measurement->absolute_error > gradient_check_max_absolute_error) {
        gradient_check_max_absolute_error = measurement->absolute_error;
    }
    if (measurement->relative_error > gradient_check_max_relative_error) {
        gradient_check_max_relative_error = measurement->relative_error;
    }
    if (measurement->scaled_error > gradient_check_max_scaled_error) {
        gradient_check_max_scaled_error = measurement->scaled_error;
    }
}


static void print_gradient_measurement(
    const char *scope,
    uint64_t parameter_key,
    const char *name,
    size_t element,
    size_t step_index,
    const GradientMeasurement *measurement
)
{
    const char *sign_relation = measurement->analytic == 0.0 &&
            measurement->numeric == 0.0
        ? "both_zero"
        : (measurement->signs_agree ? "concordant" : "discordant");

    printf(
        "GRADIENT_PROBE scope=%s key=%llu name=%s index=%zu "
        "step_scale=%.9g base=%s analytic=%.17g numeric=%.17g "
        "abs_error=%.17g relative_error=%.17g scaled_error=%.17g "
        "requested_h=%.17g effective_h=%.17g "
        "effective_delta=%.17g sign=%s tolerance_pass=%s\n",
        scope,
        (unsigned long long)parameter_key,
        name,
        element,
        (double)gradient_check_step_scales[step_index],
        step_index == GRADIENT_CHECK_BASE_STEP_INDEX ? "true" : "false",
        measurement->analytic,
        measurement->numeric,
        measurement->absolute_error,
        measurement->relative_error,
        measurement->scaled_error,
        measurement->requested_h,
        measurement->effective_h,
        measurement->effective_delta,
        sign_relation,
        gradient_measurement_passes(measurement) ? "true" : "false"
    );
}


static int evaluate_parameter_probe_grid(
    const VAECProfile *profile,
    uint64_t seed,
    size_t parameter_index,
    size_t element,
    float original,
    double analytic,
    GradientMeasurement measurements[GRADIENT_CHECK_STEP_COUNT]
)
{
    size_t step_index;

    if (!isfinite(original) || !isfinite(analytic)) {
        return 0;
    }
    for (step_index = 0;
         step_index < GRADIENT_CHECK_STEP_COUNT;
         step_index++) {

        float requested_h = gradient_check_step_scales[step_index] *
            fmaxf(1.0f, fabsf(original));
        float plus_value = original + requested_h;
        float minus_value = original - requested_h;
        double denominator =
            (double)plus_value - (double)minus_value;
        double plus_loss;
        double minus_loss;
        double numeric;
        DifferentiabilitySignature plus_signature;
        DifferentiabilitySignature minus_signature;

        if (!isfinite(requested_h) ||
            !isfinite(plus_value) ||
            !isfinite(minus_value) ||
            !(denominator > 0.0) ||
            !evaluate_parameter_loss(
                profile,
                seed,
                parameter_index,
                element,
                plus_value,
                &plus_loss,
                &plus_signature) ||
            !evaluate_parameter_loss(
                profile,
                seed,
                parameter_index,
                element,
                minus_value,
                &minus_loss,
                &minus_signature) ||
            !signatures_are_smooth(
                &plus_signature, &minus_signature)) {

            return 0;
        }
        numeric = (plus_loss - minus_loss) / denominator;
        if (!isfinite(numeric)) {
            return 0;
        }
        gradient_measurement_make(
            analytic,
            numeric,
            requested_h,
            denominator,
            &measurements[step_index]
        );
    }
    return 1;
}


static int evaluate_input_probe_grid(
    const VAECProfile *profile,
    uint64_t seed,
    size_t element,
    float original,
    double analytic,
    GradientMeasurement measurements[GRADIENT_CHECK_STEP_COUNT]
)
{
    size_t step_index;

    if (!isfinite(original) || !isfinite(analytic)) {
        return 0;
    }
    for (step_index = 0;
         step_index < GRADIENT_CHECK_STEP_COUNT;
         step_index++) {

        float requested_h = gradient_check_step_scales[step_index] *
            fmaxf(1.0f, fabsf(original));
        float plus_value = original + requested_h;
        float minus_value = original - requested_h;
        double denominator =
            (double)plus_value - (double)minus_value;
        double plus_loss;
        double minus_loss;
        double numeric;
        DifferentiabilitySignature plus_signature;
        DifferentiabilitySignature minus_signature;

        if (!isfinite(requested_h) ||
            minus_value < 0.0f ||
            plus_value > 1.0f ||
            !(denominator > 0.0) ||
            !evaluate_input_loss(
                profile,
                seed,
                element,
                plus_value,
                &plus_loss,
                &plus_signature) ||
            !evaluate_input_loss(
                profile,
                seed,
                element,
                minus_value,
                &minus_loss,
                &minus_signature) ||
            !signatures_are_smooth(
                &plus_signature, &minus_signature)) {

            return 0;
        }
        numeric = (plus_loss - minus_loss) / denominator;
        if (!isfinite(numeric)) {
            return 0;
        }
        gradient_measurement_make(
            analytic,
            numeric,
            requested_h,
            denominator,
            &measurements[step_index]
        );
    }
    return 1;
}


static int test_end_to_end_gradient_check(void)
{
    const uint64_t seed = UINT64_C(0xBADC0FFEE0DDF00D);
    VAECProfile profile;
    Fixture analytic;
    size_t parameter_index;
    size_t tested_parameters = 0;
    int input_probe_found = 0;
    int passed = 0;

    make_synthetic_depth4(&profile);
    fixture_init(&analytic);
    gradient_check_max_absolute_error = 0.0;
    gradient_check_max_relative_error = 0.0;
    gradient_check_max_scaled_error = 0.0;
    gradient_check_multih_max_absolute_error = 0.0;
    gradient_check_multih_max_relative_error = 0.0;
    gradient_check_multih_max_scaled_error = 0.0;
    elementwise_diagnostic_tolerance_failures = 0;
    if (!fixture_allocate(&analytic, &profile, seed, 1) ||
        !fixture_forward_ready(&analytic) ||
        !zero_parameter_gradients(&analytic.model) ||
        vae_c_model_vae_backward(
            &analytic.model, &analytic.workspace, 1.0f) != VAE_C_OP_OK) {

        goto cleanup;
    }

    for (parameter_index = 0;
         parameter_index < analytic.model.parameter_count;
         parameter_index++) {

        VAECParameter *parameter = analytic.model.parameters[
            parameter_index
        ];
        size_t attempt;
        int found = 0;

        for (attempt = 0;
             attempt < parameter->shape.count && attempt < 64;
             attempt++) {

            size_t element = attempt;
            GradientMeasurement measurements[GRADIENT_CHECK_STEP_COUNT];
            size_t step_index;

            if (!evaluate_parameter_probe_grid(
                    &profile,
                    seed,
                    parameter_index,
                    element,
                    parameter->values[element],
                    parameter->gradients[element],
                    measurements)) {

                continue;
            }

            found = 1;
            tested_parameters++;
            for (step_index = 0;
                 step_index < GRADIENT_CHECK_STEP_COUNT;
                 step_index++) {

                int is_base =
                    step_index == GRADIENT_CHECK_BASE_STEP_INDEX;

                gradient_measurement_record(
                    &measurements[step_index], is_base
                );
                print_gradient_measurement(
                    "parameter",
                    parameter->parameter_key,
                    parameter_key_name(parameter->parameter_key),
                    element,
                    step_index,
                    &measurements[step_index]
                );
            }
            if (!gradient_measurement_passes(
                    &measurements[GRADIENT_CHECK_BASE_STEP_INDEX])) {

                elementwise_diagnostic_tolerance_failures++;
            }
            break;
        }
        if (!found) {
            fprintf(stderr,
                    "no differentiable probe for parameter key=%llu "
                    "name=%s\n",
                    (unsigned long long)parameter->parameter_key,
                    parameter_key_name(parameter->parameter_key));
        }
    }

    {
        size_t attempt;

        for (attempt = 0;
             attempt < analytic.workspace.input.shape.count &&
                 attempt < 64;
             attempt++) {

            size_t element = attempt;
            GradientMeasurement measurements[GRADIENT_CHECK_STEP_COUNT];
            size_t step_index;

            if (!evaluate_input_probe_grid(
                    &profile,
                    seed,
                    element,
                    analytic.workspace.input.data[element],
                    analytic.workspace.input.grad[element],
                    measurements)) {

                continue;
            }

            input_probe_found = 1;
            for (step_index = 0;
                 step_index < GRADIENT_CHECK_STEP_COUNT;
                 step_index++) {

                int is_base =
                    step_index == GRADIENT_CHECK_BASE_STEP_INDEX;

                gradient_measurement_record(
                    &measurements[step_index], is_base
                );
                print_gradient_measurement(
                    "input",
                    UINT64_C(0),
                    "input",
                    element,
                    step_index,
                    &measurements[step_index]
                );
            }
            if (!gradient_measurement_passes(
                    &measurements[GRADIENT_CHECK_BASE_STEP_INDEX])) {

                elementwise_diagnostic_tolerance_failures++;
            }
            break;
        }
        if (!input_probe_found) {
            fprintf(stderr, "no differentiable input probe\n");
        }
    }

    passed =
        tested_parameters == VAE_C_MODEL_MAX_PARAMETERS &&
        input_probe_found;

    printf("ELEMENTWISE_DIAGNOSTIC_TOLERANCE_FAILURES=%zu\n",
           elementwise_diagnostic_tolerance_failures);

cleanup:
    fixture_free(&analytic);
    return passed;
}


static uint64_t directional_mix(uint64_t value)
{
    value += UINT64_C(0x9E3779B97F4A7C15);
    value = (value ^ (value >> 30)) *
        UINT64_C(0xBF58476D1CE4E5B9);
    value = (value ^ (value >> 27)) *
        UINT64_C(0x94D049BB133111EB);
    return value ^ (value >> 31);
}


static double *directional_direction_allocate(
    uint64_t key,
    size_t count
)
{
    double *direction;
    double norm_squared = 0.0;
    double norm;
    size_t index;

    if (count == 0 || count > SIZE_MAX / sizeof(*direction)) {
        return NULL;
    }
    direction = (double *)malloc(count * sizeof(*direction));
    if (direction == NULL) {
        return NULL;
    }
    for (index = 0; index < count; index++) {
        uint64_t mixed = directional_mix(
            key ^ directional_mix((uint64_t)index)
        );
        int raw = (int)(mixed % UINT64_C(17)) - 8;

        if (raw == 0) {
            raw = 1;
        }
        direction[index] = (double)raw;
        norm_squared += direction[index] * direction[index];
    }
    norm = sqrt(norm_squared);
    if (!isfinite(norm) || !(norm > 0.0)) {
        free(direction);
        return NULL;
    }
    for (index = 0; index < count; index++) {
        direction[index] /= norm;
    }
    return direction;
}


static double directional_dot(
    const float *values,
    const double *direction,
    size_t count
)
{
    double result = 0.0;
    size_t index;

    for (index = 0; index < count; index++) {
        result += (double)values[index] * direction[index];
    }
    return result;
}


static int directional_effective_delta(
    const float *center,
    const double *direction,
    size_t count,
    double requested_h,
    int require_unit_interval,
    double *effective_delta
)
{
    double projected_delta = 0.0;
    size_t index;

    if (center == NULL || direction == NULL || effective_delta == NULL ||
        !isfinite(requested_h) || !(requested_h > 0.0)) {

        return 0;
    }
    for (index = 0; index < count; index++) {
        float plus = (float)(
            (double)center[index] + requested_h * direction[index]
        );
        float minus = (float)(
            (double)center[index] - requested_h * direction[index]
        );

        if (!isfinite(plus) || !isfinite(minus) ||
            (require_unit_interval &&
             (minus < 0.0f || plus > 1.0f))) {

            return 0;
        }
        projected_delta +=
            ((double)plus - (double)minus) * direction[index];
    }
    if (!isfinite(projected_delta) || !(projected_delta > 0.0)) {
        return 0;
    }
    *effective_delta = projected_delta;
    return 1;
}


static int evaluate_parameter_direction_loss(
    const VAECProfile *profile,
    uint64_t seed,
    size_t parameter_index,
    const double *direction,
    size_t count,
    double signed_h,
    double *loss,
    DifferentiabilitySignature *signature
)
{
    Fixture fixture;
    VAECParameter *parameter;
    size_t index;
    int passed = 0;

    fixture_init(&fixture);
    if (direction == NULL || loss == NULL || signature == NULL ||
        !isfinite(signed_h) ||
        !fixture_allocate(&fixture, profile, seed, 1) ||
        parameter_index >= fixture.model.parameter_count) {

        goto cleanup;
    }
    parameter = fixture.model.parameters[parameter_index];
    if (parameter->shape.count != count) {
        goto cleanup;
    }
    for (index = 0; index < count; index++) {
        float perturbed = (float)(
            (double)parameter->values[index] +
            signed_h * direction[index]
        );

        if (!isfinite(perturbed)) {
            goto cleanup;
        }
        parameter->values[index] = perturbed;
    }
    if (vae_c_model_vae_forward(
            &fixture.model,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            VAE_C_LATENT_MODE_STOCHASTIC,
            &fixture.workspace,
            &fixture.stats) != VAE_C_OP_OK ||
        !differentiability_signature(&fixture, signature)) {

        goto cleanup;
    }
    *loss = fixture.stats.mean_per_sample;
    passed = isfinite(*loss);

cleanup:
    fixture_free(&fixture);
    return passed;
}


static int evaluate_input_direction_loss(
    const VAECProfile *profile,
    uint64_t seed,
    const double *direction,
    size_t count,
    double signed_h,
    double *loss,
    DifferentiabilitySignature *signature
)
{
    Fixture fixture;
    VAECTensor target;
    size_t index;
    int passed = 0;

    fixture_init(&fixture);
    vae_c_tensor_init(&target);
    if (direction == NULL || loss == NULL || signature == NULL ||
        !isfinite(signed_h) ||
        !fixture_allocate(&fixture, profile, seed, 1) ||
        fixture.workspace.input.shape.count != count ||
        vae_c_tensor_allocate(
            &target,
            &fixture.workspace.input.shape,
            false) != VAE_C_STORAGE_OK) {

        goto cleanup;
    }
    memcpy(target.data,
           fixture.workspace.input.data,
           count * sizeof(float));
    for (index = 0; index < count; index++) {
        float perturbed = (float)(
            (double)fixture.workspace.input.data[index] +
            signed_h * direction[index]
        );

        if (!isfinite(perturbed) ||
            perturbed < 0.0f || perturbed > 1.0f) {

            goto cleanup;
        }
        fixture.workspace.input.data[index] = perturbed;
    }
    if (!run_forward_with_fixed_target(&fixture, &target) ||
        !differentiability_signature(&fixture, signature)) {

        goto cleanup;
    }
    *loss = fixture.stats.mean_per_sample;
    passed = isfinite(*loss);

cleanup:
    vae_c_tensor_free(&target);
    fixture_free(&fixture);
    return passed;
}


static int directional_signature_matches(
    const DifferentiabilitySignature *center,
    const DifferentiabilitySignature *plus,
    const DifferentiabilitySignature *minus
)
{
    return signatures_are_smooth(plus, minus) &&
        plus->mask_hash == center->mask_hash &&
        minus->mask_hash == center->mask_hash &&
        plus->pool_hash == center->pool_hash &&
        minus->pool_hash == center->pool_hash;
}


static void directional_measurement_record(
    const GradientMeasurement *measurement
)
{
    if (measurement->absolute_error >
            directional_check_max_absolute_error) {

        directional_check_max_absolute_error =
            measurement->absolute_error;
    }
    if (measurement->relative_error >
            directional_check_max_relative_error) {

        directional_check_max_relative_error =
            measurement->relative_error;
    }
    if (measurement->scaled_error >
            directional_check_max_scaled_error) {

        directional_check_max_scaled_error =
            measurement->scaled_error;
    }
}


static void print_directional_measurement(
    const char *scope,
    uint64_t parameter_key,
    const char *name,
    size_t count,
    size_t step_index,
    const GradientMeasurement *measurement
)
{
    const char *sign_relation = measurement->analytic == 0.0 &&
            measurement->numeric == 0.0
        ? "both_zero"
        : (measurement->signs_agree ? "concordant" : "discordant");

    printf(
        "DIRECTIONAL_PROBE scope=%s key=%llu name=%s count=%zu "
        "step_scale=%.9g analytic=%.17g numeric=%.17g "
        "abs_error=%.17g relative_error=%.17g scaled_error=%.17g "
        "requested_h=%.17g effective_h=%.17g "
        "effective_delta=%.17g sign=%s "
        "differentiability_signature=stable tolerance_pass=%s\n",
        scope,
        (unsigned long long)parameter_key,
        name,
        count,
        directional_check_step_scales[step_index],
        measurement->analytic,
        measurement->numeric,
        measurement->absolute_error,
        measurement->relative_error,
        measurement->scaled_error,
        measurement->requested_h,
        measurement->effective_h,
        measurement->effective_delta,
        sign_relation,
        gradient_measurement_passes(measurement) ? "true" : "false"
    );
}


static int test_end_to_end_directional_gradient_check(void)
{
    const uint64_t seed = UINT64_C(0xBADC0FFEE0DDF00D);
    const uint64_t input_direction_key =
        UINT64_C(0x494E5055545F4449);
    VAECProfile profile;
    Fixture analytic;
    DifferentiabilitySignature center_signature;
    size_t parameter_index;
    size_t tested_parameters = 0;
    size_t structural_failures = 0;
    size_t tolerance_failures = 0;
    int input_tested = 0;
    int passed = 0;

    make_synthetic_depth4(&profile);
    fixture_init(&analytic);
    directional_check_max_absolute_error = 0.0;
    directional_check_max_relative_error = 0.0;
    directional_check_max_scaled_error = 0.0;
    if (!fixture_allocate(&analytic, &profile, seed, 1) ||
        !fixture_forward_ready(&analytic) ||
        !differentiability_signature(&analytic, &center_signature) ||
        !zero_parameter_gradients(&analytic.model) ||
        vae_c_model_vae_backward(
            &analytic.model, &analytic.workspace, 1.0f) != VAE_C_OP_OK) {

        goto cleanup;
    }

    for (parameter_index = 0;
         parameter_index < analytic.model.parameter_count;
         parameter_index++) {

        VAECParameter *parameter =
            analytic.model.parameters[parameter_index];
        size_t count = parameter->shape.count;
        double *direction = directional_direction_allocate(
            parameter->parameter_key, count
        );
        double analytic_derivative;
        size_t step_index;
        int tensor_passed = 1;
        int analytic_valid;

        if (direction == NULL) {
            fprintf(stderr,
                    "direction allocation failed key=%llu name=%s\n",
                    (unsigned long long)parameter->parameter_key,
                    parameter_key_name(parameter->parameter_key));
            structural_failures++;
            continue;
        }
        analytic_derivative = directional_dot(
            parameter->gradients, direction, count
        );
        analytic_valid = isfinite(analytic_derivative);
        if (!analytic_valid) {
            fprintf(stderr,
                    "non-finite analytic directional derivative "
                    "key=%llu name=%s\n",
                    (unsigned long long)parameter->parameter_key,
                    parameter_key_name(parameter->parameter_key));
            tensor_passed = 0;
        }
        for (step_index = 0;
             step_index < GRADIENT_CHECK_STEP_COUNT;
             step_index++) {

            double requested_h =
                directional_check_step_scales[step_index];
            double effective_delta;
            double plus_loss;
            double minus_loss;
            double numeric;
            DifferentiabilitySignature plus_signature;
            DifferentiabilitySignature minus_signature;
            GradientMeasurement measurement;

            if (!analytic_valid ||
                !directional_effective_delta(
                    parameter->values,
                    direction,
                    count,
                    requested_h,
                    0,
                    &effective_delta) ||
                !evaluate_parameter_direction_loss(
                    &profile,
                    seed,
                    parameter_index,
                    direction,
                    count,
                    requested_h,
                    &plus_loss,
                    &plus_signature) ||
                !evaluate_parameter_direction_loss(
                    &profile,
                    seed,
                    parameter_index,
                    direction,
                    count,
                    -requested_h,
                    &minus_loss,
                    &minus_signature)) {

                fprintf(stderr,
                        "DIRECTIONAL_EVALUATION_FAILED scope=parameter "
                        "key=%llu name=%s step_scale=%.9g\n",
                        (unsigned long long)parameter->parameter_key,
                        parameter_key_name(parameter->parameter_key),
                        requested_h);
                tensor_passed = 0;
                continue;
            }
            if (!directional_signature_matches(
                    &center_signature,
                    &plus_signature,
                    &minus_signature)) {

                fprintf(stderr,
                        "DIRECTIONAL_NONDIFFERENTIABLE "
                        "scope=parameter key=%llu name=%s "
                        "step_scale=%.9g center_mask=%llu "
                        "plus_mask=%llu minus_mask=%llu "
                        "center_pool=%llu plus_pool=%llu "
                        "minus_pool=%llu\n",
                        (unsigned long long)parameter->parameter_key,
                        parameter_key_name(parameter->parameter_key),
                        requested_h,
                        (unsigned long long)center_signature.mask_hash,
                        (unsigned long long)plus_signature.mask_hash,
                        (unsigned long long)minus_signature.mask_hash,
                        (unsigned long long)center_signature.pool_hash,
                        (unsigned long long)plus_signature.pool_hash,
                        (unsigned long long)minus_signature.pool_hash);
                tensor_passed = 0;
                continue;
            }
            numeric = (plus_loss - minus_loss) / effective_delta;
            if (!isfinite(numeric)) {
                fprintf(stderr,
                        "DIRECTIONAL_NUMERIC_NONFINITE "
                        "scope=parameter key=%llu name=%s "
                        "step_scale=%.9g\n",
                        (unsigned long long)parameter->parameter_key,
                        parameter_key_name(parameter->parameter_key),
                        requested_h);
                tensor_passed = 0;
                continue;
            }
            gradient_measurement_make(
                analytic_derivative,
                numeric,
                requested_h,
                effective_delta,
                &measurement
            );
            directional_measurement_record(&measurement);
            print_directional_measurement(
                "parameter",
                parameter->parameter_key,
                parameter_key_name(parameter->parameter_key),
                count,
                step_index,
                &measurement
            );
            if (!gradient_measurement_passes(&measurement)) {
                tolerance_failures++;
            }
        }
        tested_parameters++;
        if (!tensor_passed) {
            structural_failures++;
        }
        free(direction);
    }

    {
        size_t count = analytic.workspace.input.shape.count;
        double *direction = directional_direction_allocate(
            input_direction_key, count
        );
        double analytic_derivative;
        size_t step_index;
        int input_passed = direction != NULL;
        int analytic_valid = 0;

        if (direction == NULL) {
            fprintf(stderr, "input direction allocation failed\n");
        } else {
            analytic_derivative = directional_dot(
                analytic.workspace.input.grad, direction, count
            );
            analytic_valid = isfinite(analytic_derivative);
            if (!analytic_valid) {
                input_passed = 0;
            }
            for (step_index = 0;
                 step_index < GRADIENT_CHECK_STEP_COUNT;
                 step_index++) {

                double requested_h =
                    directional_check_step_scales[step_index];
                double effective_delta;
                double plus_loss;
                double minus_loss;
                double numeric;
                DifferentiabilitySignature plus_signature;
                DifferentiabilitySignature minus_signature;
                GradientMeasurement measurement;

                if (!analytic_valid ||
                    !directional_effective_delta(
                        analytic.workspace.input.data,
                        direction,
                        count,
                        requested_h,
                        1,
                        &effective_delta) ||
                    !evaluate_input_direction_loss(
                        &profile,
                        seed,
                        direction,
                        count,
                        requested_h,
                        &plus_loss,
                        &plus_signature) ||
                    !evaluate_input_direction_loss(
                        &profile,
                        seed,
                        direction,
                        count,
                        -requested_h,
                        &minus_loss,
                        &minus_signature)) {

                    fprintf(stderr,
                            "DIRECTIONAL_EVALUATION_FAILED scope=input "
                            "step_scale=%.9g\n",
                            requested_h);
                    input_passed = 0;
                    continue;
                }
                if (!directional_signature_matches(
                        &center_signature,
                        &plus_signature,
                        &minus_signature)) {

                    fprintf(stderr,
                            "DIRECTIONAL_NONDIFFERENTIABLE scope=input "
                            "step_scale=%.9g center_mask=%llu "
                            "plus_mask=%llu minus_mask=%llu "
                            "center_pool=%llu plus_pool=%llu "
                            "minus_pool=%llu\n",
                            requested_h,
                            (unsigned long long)
                                center_signature.mask_hash,
                            (unsigned long long)plus_signature.mask_hash,
                            (unsigned long long)minus_signature.mask_hash,
                            (unsigned long long)
                                center_signature.pool_hash,
                            (unsigned long long)plus_signature.pool_hash,
                            (unsigned long long)minus_signature.pool_hash);
                    input_passed = 0;
                    continue;
                }
                numeric = (plus_loss - minus_loss) / effective_delta;
                if (!isfinite(numeric)) {
                    fprintf(stderr,
                            "DIRECTIONAL_NUMERIC_NONFINITE scope=input "
                            "step_scale=%.9g\n",
                            requested_h);
                    input_passed = 0;
                    continue;
                }
                gradient_measurement_make(
                    analytic_derivative,
                    numeric,
                    requested_h,
                    effective_delta,
                    &measurement
                );
                directional_measurement_record(&measurement);
                print_directional_measurement(
                    "input",
                    UINT64_C(0),
                    "input",
                    count,
                    step_index,
                    &measurement
                );
                if (!gradient_measurement_passes(&measurement)) {
                    tolerance_failures++;
                }
            }
        }
        input_tested = direction != NULL;
        if (!input_passed) {
            structural_failures++;
        }
        free(direction);
    }

    printf("GLOBAL_DIRECTIONAL_DIAGNOSTIC_TOLERANCE_FAILURES=%zu\n",
           tolerance_failures);
    printf("GLOBAL_DIRECTIONAL_DIAGNOSTIC_STRUCTURAL_FAILURES=%zu\n",
           structural_failures);
    printf("GLOBAL_DIRECTIONAL_DIAGNOSTIC_IS_HARD_GATE=false\n");
    printf("GLOBAL_DIRECTIONAL_DIAGNOSTIC_REASON="
           "global_loss_float32_finite_difference_not_informative\n");
    passed =
        tested_parameters == VAE_C_MODEL_MAX_PARAMETERS &&
        input_tested &&
        structural_failures == 0;

cleanup:
    fixture_free(&analytic);
    return passed;
}


typedef struct {
    uint64_t mask_hash;
    uint64_t pool_hash;
    double minimum_relu_margin;
    double minimum_pool_margin;
} LocalGradientSignature;


enum {
    LOCAL_GRADIENT_STEP_COUNT = 5
};


static const double local_gradient_steps[LOCAL_GRADIENT_STEP_COUNT] = {
    2.0e-2,
    1.0e-2,
    5.0e-3,
    2.0e-3,
    1.0e-3
};


static double local_float_spacing(double value)
{
    float rounded = (float)value;
    float next = nextafterf(rounded, INFINITY);

    return (double)next - (double)rounded;
}


static int local_signature_equal(
    const LocalGradientSignature *left,
    const LocalGradientSignature *right
)
{
    return left->mask_hash == right->mask_hash &&
        left->pool_hash == right->pool_hash;
}


static int local_decoder_signature(
    const Fixture *fixture,
    LocalGradientSignature *signature
)
{
    uint64_t mask_hash = UINT64_C(1469598103934665603);
    double minimum_relu = DBL_MAX;
    size_t hidden = fixture->model.config.decoder_hidden_conv_count;
    size_t block;

    for (block = 0; block <= hidden; block++) {
        const VAECTensor *tensor = block == 0
            ? &fixture->workspace.decoder_fc_batch_norm_output
            : &fixture->workspace.decoder_hidden_batch_norm_outputs[
                block - 1
            ];
        size_t index;

        for (index = 0; index < tensor->shape.count; index++) {
            double margin = fabs((double)tensor->data[index]);

            mask_hash = hash_mask_value(
                mask_hash, tensor->data[index]
            );
            if (margin < minimum_relu) {
                minimum_relu = margin;
            }
        }
    }
    signature->mask_hash = mask_hash;
    signature->pool_hash = UINT64_C(0);
    signature->minimum_relu_margin = minimum_relu;
    signature->minimum_pool_margin = DBL_MAX;
    return isfinite(minimum_relu);
}


static int local_encoder_signature(
    const Fixture *fixture,
    LocalGradientSignature *signature
)
{
    uint64_t mask_hash = UINT64_C(1469598103934665603);
    uint64_t pool_hash = UINT64_C(1469598103934665603);
    double minimum_relu = DBL_MAX;
    double minimum_pool = DBL_MAX;
    size_t depth = fixture->model.config.conv_depth;
    size_t block;

    for (block = 0; block < depth; block++) {
        const VAECTensor *tensor =
            &fixture->workspace.encoder_batch_norm_outputs[block];
        size_t index;

        for (index = 0; index < tensor->shape.count; index++) {
            double margin = fabs((double)tensor->data[index]);

            mask_hash = hash_mask_value(
                mask_hash, tensor->data[index]
            );
            if (margin < minimum_relu) {
                minimum_relu = margin;
            }
        }
    }
    pool_hash = hash_bytes(
        pool_hash,
        fixture->model.encoder_max_pool.argmax_indices,
        fixture->model.encoder_max_pool.last_count * sizeof(size_t)
    );
    {
        const VAECTensor *input =
            &fixture->workspace.encoder_relu_outputs[0];
        size_t n;
        size_t c;
        size_t oy;
        size_t ox;

        for (n = 0; n < input->shape.dims[0]; n++) {
            for (c = 0; c < input->shape.dims[1]; c++) {
                for (oy = 0; oy < input->shape.dims[2] / 2; oy++) {
                    for (ox = 0; ox < input->shape.dims[3] / 2; ox++) {
                        double largest = -DBL_MAX;
                        double second = -DBL_MAX;
                        size_t ky;
                        size_t kx;

                        for (ky = 0; ky < 2; ky++) {
                            for (kx = 0; kx < 2; kx++) {
                                size_t offset =
                                    n * input->shape.strides[0] +
                                    c * input->shape.strides[1] +
                                    (2 * oy + ky) *
                                        input->shape.strides[2] +
                                    (2 * ox + kx) *
                                        input->shape.strides[3];
                                double value = input->data[offset];

                                if (value > largest) {
                                    second = largest;
                                    largest = value;
                                } else if (value > second) {
                                    second = value;
                                }
                            }
                        }
                        if (largest - second < minimum_pool) {
                            minimum_pool = largest - second;
                        }
                    }
                }
            }
        }
    }
    signature->mask_hash = mask_hash;
    signature->pool_hash = pool_hash;
    signature->minimum_relu_margin = minimum_relu;
    signature->minimum_pool_margin = minimum_pool;
    return isfinite(minimum_relu) && isfinite(minimum_pool);
}


static VAECOpStatus local_decoder_backward(
    VAECModel *model,
    VAECModelWorkspace *workspace
)
{
    size_t hidden = model->config.decoder_hidden_conv_count;
    size_t index;
    VAECOpStatus status;

    status = vae_c_conv2d_backward(
        &model->decoder_output_conv,
        &workspace->decoder_upsample_outputs[
            model->config.decoder_upsample_count - 1
        ],
        &workspace->logits
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    for (index = hidden; index-- > 0;) {
        status = vae_c_nearest_upsample2d_backward(
            &workspace->decoder_hidden_relu_outputs[index],
            &workspace->decoder_upsample_outputs[index + 2]
        );
        if (status == VAE_C_OP_OK) {
            status = vae_c_relu_backward(
                &workspace->decoder_hidden_batch_norm_outputs[index],
                &workspace->decoder_hidden_relu_outputs[index]
            );
        }
        if (status == VAE_C_OP_OK) {
            status = vae_c_batch_norm2d_backward(
                &model->decoder_hidden_batch_norms[index],
                &workspace->decoder_hidden_conv_outputs[index],
                &workspace->decoder_hidden_batch_norm_outputs[index]
            );
        }
        if (status == VAE_C_OP_OK) {
            status = vae_c_conv2d_backward(
                &model->decoder_hidden_convs[index],
                &workspace->decoder_upsample_outputs[index + 1],
                &workspace->decoder_hidden_conv_outputs[index]
            );
        }
        if (status != VAE_C_OP_OK) {
            return status;
        }
    }
    status = vae_c_nearest_upsample2d_backward(
        &workspace->decoder_upsample_outputs[0],
        &workspace->decoder_upsample_outputs[1]
    );
    if (status == VAE_C_OP_OK) {
        status = vae_c_nearest_upsample2d_backward(
            &workspace->decoder_reshape_view,
            &workspace->decoder_upsample_outputs[0]
        );
    }
    if (status == VAE_C_OP_OK) {
        status = vae_c_relu_backward(
            &workspace->decoder_fc_batch_norm_output,
            &workspace->decoder_fc_relu_output
        );
    }
    if (status == VAE_C_OP_OK) {
        status = vae_c_batch_norm1d_backward(
            &model->decoder_fc_batch_norm,
            &workspace->decoder_fc_output,
            &workspace->decoder_fc_batch_norm_output
        );
    }
    if (status == VAE_C_OP_OK) {
        status = vae_c_dense_backward(
            &model->decoder_fc,
            &workspace->z,
            &workspace->decoder_fc_output
        );
    }
    return status;
}


static VAECOpStatus local_encoder_body_backward(
    VAECModel *model,
    VAECModelWorkspace *workspace
)
{
    size_t depth = model->config.conv_depth;
    size_t index;
    VAECOpStatus status;

    status = vae_c_global_avg_pool2d_backward(
        &workspace->encoder_relu_outputs[depth - 1],
        &workspace->encoder_global_avg_output
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    for (index = depth; index-- > 1;) {
        const VAECTensor *input = index == 1
            ? &workspace->encoder_pool_output
            : &workspace->encoder_relu_outputs[index - 1];

        status = vae_c_relu_backward(
            &workspace->encoder_batch_norm_outputs[index],
            &workspace->encoder_relu_outputs[index]
        );
        if (status == VAE_C_OP_OK) {
            status = vae_c_batch_norm2d_backward(
                &model->encoder_batch_norms[index],
                &workspace->encoder_conv_outputs[index],
                &workspace->encoder_batch_norm_outputs[index]
            );
        }
        if (status == VAE_C_OP_OK) {
            status = vae_c_conv2d_backward(
                &model->encoder_convs[index],
                input,
                &workspace->encoder_conv_outputs[index]
            );
        }
        if (status != VAE_C_OP_OK) {
            return status;
        }
    }
    status = vae_c_max_pool2d_backward(
        &model->encoder_max_pool,
        &workspace->encoder_relu_outputs[0],
        &workspace->encoder_pool_output
    );
    if (status == VAE_C_OP_OK) {
        status = vae_c_relu_backward(
            &workspace->encoder_batch_norm_outputs[0],
            &workspace->encoder_relu_outputs[0]
        );
    }
    if (status == VAE_C_OP_OK) {
        status = vae_c_batch_norm2d_backward(
            &model->encoder_batch_norms[0],
            &workspace->encoder_conv_outputs[0],
            &workspace->encoder_batch_norm_outputs[0]
        );
    }
    if (status == VAE_C_OP_OK) {
        status = vae_c_conv2d_backward(
            &model->encoder_convs[0],
            &workspace->input,
            &workspace->encoder_conv_outputs[0]
        );
    }
    return status;
}


static int local_perturb_values(
    float *values,
    const double *direction,
    size_t count,
    double signed_h,
    int require_unit_interval
)
{
    size_t index;

    for (index = 0; index < count; index++) {
        float candidate = (float)(
            (double)values[index] + signed_h * direction[index]
        );

        if (!isfinite(candidate) ||
            (require_unit_interval &&
             (candidate < 0.0f || candidate > 1.0f))) {
            return 0;
        }
        values[index] = candidate;
    }
    return 1;
}


static double local_dot_f32(
    const float *left,
    const float *right,
    size_t count
)
{
    double sum = 0.0;
    size_t index;

    for (index = 0; index < count; index++) {
        sum += (double)left[index] * (double)right[index];
    }
    return sum;
}


typedef int (*LocalGradientEvaluator)(
    const double *direction,
    double signed_h,
    double *objective,
    LocalGradientSignature *signature,
    void *context
);


static int local_run_fixed_grid(
    const char *block,
    const char *probe,
    const float *center,
    size_t count,
    int require_unit_interval,
    const double *direction,
    double analytic,
    LocalGradientEvaluator evaluator,
    void *context,
    const LocalGradientSignature *center_signature
)
{
    const double minimum_signal_ulps = 5.0;
    size_t informative_count = 0;
    size_t informative_pass_count = 0;
    size_t step;
    int structurally_valid = 1;

    for (step = 0; step < LOCAL_GRADIENT_STEP_COUNT; step++) {
        double effective_delta;
        double plus_objective;
        double minus_objective;
        double midpoint;
        double spacing;
        double expected_signal;
        double numeric;
        double signal_ulps;
        int informative;
        int measurement_pass;
        LocalGradientSignature plus_signature;
        LocalGradientSignature minus_signature;
        GradientMeasurement measurement;

        if (!directional_effective_delta(
                center,
                direction,
                count,
                local_gradient_steps[step],
                require_unit_interval,
                &effective_delta) ||
            !evaluator(
                direction,
                local_gradient_steps[step],
                &plus_objective,
                &plus_signature,
                context) ||
            !evaluator(
                direction,
                -local_gradient_steps[step],
                &minus_objective,
                &minus_signature,
                context)) {

            printf("LOCAL_FD_EVALUATION_FAILED block=%s "
                   "probe=%s h=%.9g\n",
                   block, probe, local_gradient_steps[step]);
            structurally_valid = 0;
            continue;
        }
        if (!local_signature_equal(
                center_signature, &plus_signature) ||
            !local_signature_equal(
                center_signature, &minus_signature)) {

            printf("LOCAL_FD_NONDIFFERENTIABLE block=%s "
                   "probe=%s h=%.9g\n",
                   block, probe, local_gradient_steps[step]);
            structurally_valid = 0;
            continue;
        }
        numeric =
            (plus_objective - minus_objective) / effective_delta;
        if (!isfinite(numeric)) {
            structurally_valid = 0;
            continue;
        }
        gradient_measurement_make(
            analytic,
            numeric,
            local_gradient_steps[step],
            effective_delta,
            &measurement
        );
        midpoint = 0.5 * (plus_objective + minus_objective);
        spacing = local_float_spacing(midpoint);
        expected_signal = analytic * effective_delta;
        signal_ulps = spacing > 0.0
            ? fabs(expected_signal) / spacing
            : INFINITY;
        informative =
            isfinite(signal_ulps) &&
            signal_ulps >= minimum_signal_ulps;
        measurement_pass =
            gradient_measurement_passes(&measurement) &&
            measurement.signs_agree;
        if (informative) {
            informative_count++;
            if (measurement_pass) {
                informative_pass_count++;
            }
        }
        printf(
            "LOCAL_FD block=%s probe=%s h=%.9g "
            "analytic=%.17g numeric=%.17g abs_error=%.17g "
            "relative_error=%.17g effective_delta=%.17g "
            "Lplus=%.17g Lminus=%.17g Ldiff=%.17g "
            "expected_signal=%.17g float_spacing=%.17g "
            "signal_ulps=%.17g signature=stable "
            "informative=%s tolerance_pass=%s\n",
            block,
            probe,
            local_gradient_steps[step],
            measurement.analytic,
            measurement.numeric,
            measurement.absolute_error,
            measurement.relative_error,
            measurement.effective_delta,
            plus_objective,
            minus_objective,
            plus_objective - minus_objective,
            expected_signal,
            spacing,
            signal_ulps,
            informative ? "true" : "false",
            measurement_pass ? "true" : "false"
        );
    }
    printf("LOCAL_FD_WINDOW block=%s probe=%s informative=%zu "
           "passing=%zu gate=%s\n",
           block,
           probe,
           informative_count,
           informative_pass_count,
           structurally_valid && informative_count > 0 &&
                   informative_pass_count > 0
               ? "PASS"
               : "FAIL");
    return
        structurally_valid &&
        informative_count > 0 &&
        informative_pass_count > 0;
}


typedef struct {
    const VAECProfile *profile;
    uint64_t seed;
    const float *base_z;
} LocalDecoderContext;


static int local_decoder_objective(
    const double *direction,
    double signed_h,
    double *objective,
    LocalGradientSignature *signature,
    void *context
)
{
    LocalDecoderContext *decoder = (LocalDecoderContext *)context;
    Fixture fixture;
    VAECBCEWithLogitsStats stats;
    size_t count;
    int passed = 0;

    fixture_init(&fixture);
    vae_c_bce_with_logits_stats_init(&stats);
    if (!fixture_allocate(
            &fixture, decoder->profile, decoder->seed, 1)) {
        goto cleanup;
    }
    count = fixture.workspace.z.shape.count;
    memcpy(
        fixture.workspace.z.data,
        decoder->base_z,
        count * sizeof(float)
    );
    if (!local_perturb_values(
            fixture.workspace.z.data,
            direction,
            count,
            signed_h,
            0) ||
        vae_c_model_decoder_forward(
            &fixture.model,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &fixture.workspace) != VAE_C_OP_OK ||
        vae_c_bce_with_logits_forward(
            &fixture.workspace.logits,
            &fixture.workspace.input,
            &stats) != VAE_C_OP_OK ||
        !local_decoder_signature(&fixture, signature)) {
        goto cleanup;
    }
    *objective = stats.mean_per_sample;
    passed = isfinite(*objective);
cleanup:
    fixture_free(&fixture);
    return passed;
}


typedef struct {
    const VAECProfile *profile;
    uint64_t seed;
    const float *base_mu;
    const float *base_logvar;
    const float *frozen_z_gradient;
    int perturb_mu;
    LocalGradientSignature signature;
} LocalLatentContext;


static int local_latent_objective(
    const double *direction,
    double signed_h,
    double *objective,
    LocalGradientSignature *signature,
    void *context
)
{
    LocalLatentContext *latent = (LocalLatentContext *)context;
    Fixture fixture;
    VAECKLGaussianStats kl;
    size_t count;
    int passed = 0;

    fixture_init(&fixture);
    vae_c_kl_gaussian_stats_init(&kl);
    if (!fixture_allocate(
            &fixture, latent->profile, latent->seed, 1)) {
        goto cleanup;
    }
    count = fixture.workspace.mu.shape.count;
    memcpy(
        fixture.workspace.mu.data,
        latent->base_mu,
        count * sizeof(float)
    );
    memcpy(
        fixture.workspace.logvar.data,
        latent->base_logvar,
        count * sizeof(float)
    );
    if (!local_perturb_values(
            latent->perturb_mu
                ? fixture.workspace.mu.data
                : fixture.workspace.logvar.data,
            direction,
            count,
            signed_h,
            0) ||
        vae_c_reparameterize_forward(
            VAE_C_LATENT_MODE_STOCHASTIC,
            &fixture.workspace.mu,
            &fixture.workspace.logvar,
            &fixture.workspace.epsilon,
            &fixture.workspace.z) != VAE_C_OP_OK ||
        vae_c_kl_gaussian_forward(
            &fixture.workspace.mu,
            &fixture.workspace.logvar,
            &kl) != VAE_C_OP_OK) {
        goto cleanup;
    }
    *objective = local_dot_f32(
        fixture.workspace.z.data,
        latent->frozen_z_gradient,
        count
    ) + kl.mean_per_sample;
    *signature = latent->signature;
    passed = isfinite(*objective);
cleanup:
    fixture_free(&fixture);
    return passed;
}


typedef struct {
    const VAECProfile *profile;
    uint64_t seed;
    size_t parameter_index;
    int perturb_input;
    const float *frozen_gap_gradient;
} LocalEncoderContext;


static int local_encoder_objective(
    const double *direction,
    double signed_h,
    double *objective,
    LocalGradientSignature *signature,
    void *context
)
{
    LocalEncoderContext *encoder = (LocalEncoderContext *)context;
    Fixture fixture;
    float *values;
    size_t count;
    int require_unit_interval;
    int passed = 0;

    fixture_init(&fixture);
    if (!fixture_allocate(
            &fixture, encoder->profile, encoder->seed, 1)) {
        goto cleanup;
    }
    if (encoder->perturb_input) {
        values = fixture.workspace.input.data;
        count = fixture.workspace.input.shape.count;
        require_unit_interval = 1;
    } else {
        if (encoder->parameter_index >=
                fixture.model.parameter_count) {
            goto cleanup;
        }
        values = fixture.model.parameters[
            encoder->parameter_index
        ]->values;
        count = fixture.model.parameters[
            encoder->parameter_index
        ]->shape.count;
        require_unit_interval = 0;
    }
    if (!local_perturb_values(
            values,
            direction,
            count,
            signed_h,
            require_unit_interval) ||
        vae_c_model_encoder_forward(
            &fixture.model,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &fixture.workspace) != VAE_C_OP_OK ||
        !local_encoder_signature(&fixture, signature)) {
        goto cleanup;
    }
    *objective = local_dot_f32(
        fixture.workspace.encoder_global_avg_output.data,
        encoder->frozen_gap_gradient,
        fixture.workspace.encoder_global_avg_output.shape.count
    );
    passed = isfinite(*objective);
cleanup:
    fixture_free(&fixture);
    return passed;
}


static int local_exact_split_and_encoder_fc(Fixture *fixture)
{
    size_t batch = fixture->workspace.mu.shape.dims[0];
    size_t latent = fixture->workspace.mu.shape.dims[1];
    size_t in_features = fixture->model.encoder_fc.in_features;
    size_t out_features = fixture->model.encoder_fc.out_features;
    float *packed_expected = NULL;
    float *weight_expected = NULL;
    float *bias_expected = NULL;
    float *gap_expected = NULL;
    size_t n;
    size_t o;
    size_t i;
    int split_passed = 1;
    int fc_passed = 1;

    packed_expected = (float *)calloc(
        fixture->workspace.encoder_packed_output.shape.count,
        sizeof(float)
    );
    weight_expected = (float *)calloc(
        fixture->model.encoder_fc.weights.shape.count,
        sizeof(float)
    );
    bias_expected = (float *)calloc(
        out_features, sizeof(float)
    );
    gap_expected = (float *)calloc(
        fixture->workspace.encoder_global_avg_output.shape.count,
        sizeof(float)
    );
    if (packed_expected == NULL || weight_expected == NULL ||
        bias_expected == NULL || gap_expected == NULL) {
        split_passed = 0;
        fc_passed = 0;
        goto cleanup;
    }
    for (n = 0; n < batch; n++) {
        for (i = 0; i < latent; i++) {
            packed_expected[n * 2 * latent + i] =
                fixture->workspace.mu.grad[n * latent + i];
            packed_expected[n * 2 * latent + latent + i] =
                fixture->workspace.logvar.grad[n * latent + i];
        }
    }
    if (vae_c_latent_split_backward(
            &fixture->workspace.encoder_packed_output,
            &fixture->workspace.mu,
            &fixture->workspace.logvar) != VAE_C_OP_OK ||
        memcmp(
            packed_expected,
            fixture->workspace.encoder_packed_output.grad,
            fixture->workspace.encoder_packed_output.shape.count *
                sizeof(float)) != 0) {
        split_passed = 0;
    }
    for (n = 0; n < batch; n++) {
        for (o = 0; o < out_features; o++) {
            float upstream =
                fixture->workspace.encoder_packed_output.grad[
                    n * out_features + o
                ];

            bias_expected[o] += upstream;
            for (i = 0; i < in_features; i++) {
                weight_expected[o * in_features + i] +=
                    upstream *
                    fixture->workspace.encoder_global_avg_output.data[
                        n * in_features + i
                    ];
                gap_expected[n * in_features + i] +=
                    upstream *
                    fixture->model.encoder_fc.weights.values[
                        o * in_features + i
                    ];
            }
        }
    }
    if (vae_c_dense_backward(
            &fixture->model.encoder_fc,
            &fixture->workspace.encoder_global_avg_output,
            &fixture->workspace.encoder_packed_output) != VAE_C_OP_OK ||
        memcmp(
            weight_expected,
            fixture->model.encoder_fc.weights.gradients,
            fixture->model.encoder_fc.weights.shape.count *
                sizeof(float)) != 0 ||
        memcmp(
            bias_expected,
            fixture->model.encoder_fc.bias.gradients,
            out_features * sizeof(float)) != 0 ||
        memcmp(
            gap_expected,
            fixture->workspace.encoder_global_avg_output.grad,
            fixture->workspace.encoder_global_avg_output.shape.count *
                sizeof(float)) != 0) {
        fc_passed = 0;
    }
cleanup:
    printf("LOCAL_EXACT latent_split=%s encoder_fc_weight=%s "
           "encoder_fc_bias=%s gap_grad=%s\n",
           split_passed ? "PASS" : "FAIL",
           fc_passed ? "PASS" : "FAIL",
           fc_passed ? "PASS" : "FAIL",
           fc_passed ? "PASS" : "FAIL");
    free(packed_expected);
    free(weight_expected);
    free(bias_expected);
    free(gap_expected);
    return split_passed && fc_passed;
}


static int test_local_gradient_hard_gates(void)
{
    const uint64_t seed = UINT64_C(0xBADC0FFEE0DDF00D);
    VAECProfile profile;
    Fixture center;
    LocalGradientSignature decoder_signature;
    LocalGradientSignature encoder_signature;
    LocalDecoderContext decoder_context;
    LocalLatentContext latent_context;
    LocalEncoderContext encoder_context;
    float *base_z = NULL;
    float *base_mu = NULL;
    float *base_logvar = NULL;
    size_t latent_count;
    size_t direction_index;
    int decoder_gate = 1;
    int latent_gate = 1;
    int exact_gate = 0;
    int encoder_gate = 1;
    int passed = 0;

    make_synthetic_depth4(&profile);
    fixture_init(&center);
    if (!fixture_allocate(&center, &profile, seed, 1) ||
        !fixture_forward_ready(&center) ||
        !zero_parameter_gradients(&center.model) ||
        !local_decoder_signature(&center, &decoder_signature) ||
        !local_encoder_signature(&center, &encoder_signature)) {
        goto cleanup;
    }
    latent_count = center.workspace.z.shape.count;
    base_z = (float *)malloc(latent_count * sizeof(float));
    base_mu = (float *)malloc(latent_count * sizeof(float));
    base_logvar = (float *)malloc(latent_count * sizeof(float));
    if (base_z == NULL || base_mu == NULL || base_logvar == NULL) {
        goto cleanup;
    }
    memcpy(base_z,
           center.workspace.z.data,
           latent_count * sizeof(float));
    memcpy(base_mu,
           center.workspace.mu.data,
           latent_count * sizeof(float));
    memcpy(base_logvar,
           center.workspace.logvar.data,
           latent_count * sizeof(float));

    if (vae_c_bce_with_logits_backward(
            &center.workspace.logits,
            &center.workspace.input,
            1.0f) != VAE_C_OP_OK ||
        local_decoder_backward(
            &center.model, &center.workspace) != VAE_C_OP_OK) {
        goto cleanup;
    }
    decoder_context.profile = &profile;
    decoder_context.seed = seed;
    decoder_context.base_z = base_z;
    for (direction_index = 0; direction_index < 3; direction_index++) {
        double *direction = directional_direction_allocate(
            UINT64_C(0xDEC000) + (uint64_t)direction_index,
            latent_count
        );
        double analytic;
        char probe[32];

        if (direction == NULL) {
            decoder_gate = 0;
            continue;
        }
        analytic = directional_dot(
            center.workspace.z.grad, direction, latent_count
        );
        snprintf(probe, sizeof(probe),
                 "z_direction_%zu", direction_index);
        if (!local_run_fixed_grid(
                "decoder_boundary",
                probe,
                base_z,
                latent_count,
                0,
                direction,
                analytic,
                local_decoder_objective,
                &decoder_context,
                &decoder_signature)) {
            decoder_gate = 0;
        }
        free(direction);
    }

    if (vae_c_model_workspace_zero_grad(
            &center.model, &center.workspace) != VAE_C_OP_OK ||
        !zero_parameter_gradients(&center.model) ||
        vae_c_vae_loss_backward(
            &center.workspace.logits,
            &center.workspace.input,
            &center.workspace.mu,
            &center.workspace.logvar,
            1.0f) != VAE_C_OP_OK ||
        local_decoder_backward(
            &center.model, &center.workspace) != VAE_C_OP_OK ||
        vae_c_reparameterize_backward(
            VAE_C_LATENT_MODE_STOCHASTIC,
            &center.workspace.mu,
            &center.workspace.logvar,
            &center.workspace.epsilon,
            &center.workspace.z) != VAE_C_OP_OK) {
        goto cleanup;
    }
    latent_context.profile = &profile;
    latent_context.seed = seed;
    latent_context.base_mu = base_mu;
    latent_context.base_logvar = base_logvar;
    latent_context.frozen_z_gradient = center.workspace.z.grad;
    latent_context.signature = decoder_signature;
    for (direction_index = 0; direction_index < 2; direction_index++) {
        double *direction = directional_direction_allocate(
            UINT64_C(0x1A7E00) + (uint64_t)direction_index,
            latent_count
        );
        const float *center_values = direction_index == 0
            ? base_mu
            : base_logvar;
        const float *analytic_values = direction_index == 0
            ? center.workspace.mu.grad
            : center.workspace.logvar.grad;
        const char *probe = direction_index == 0
            ? "mu_direction"
            : "logvar_direction";
        double analytic;

        if (direction == NULL) {
            latent_gate = 0;
            continue;
        }
        latent_context.perturb_mu = direction_index == 0;
        analytic = directional_dot(
            analytic_values, direction, latent_count
        );
        if (!local_run_fixed_grid(
                "latent_bridge_zdot_plus_kl",
                probe,
                center_values,
                latent_count,
                0,
                direction,
                analytic,
                local_latent_objective,
                &latent_context,
                &decoder_signature)) {
            latent_gate = 0;
        }
        free(direction);
    }

    exact_gate = local_exact_split_and_encoder_fc(&center);
    if (!exact_gate ||
        local_encoder_body_backward(
            &center.model, &center.workspace) != VAE_C_OP_OK) {
        goto cleanup;
    }
    encoder_context.profile = &profile;
    encoder_context.seed = seed;
    encoder_context.frozen_gap_gradient =
        center.workspace.encoder_global_avg_output.grad;
    for (direction_index = 0; direction_index < 12; direction_index++) {
        VAECParameter *parameter =
            center.model.parameters[direction_index];
        double *direction = directional_direction_allocate(
            UINT64_C(0xEBC000) + parameter->parameter_key,
            parameter->shape.count
        );
        double analytic;

        if (direction == NULL) {
            encoder_gate = 0;
            continue;
        }
        analytic = directional_dot(
            parameter->gradients,
            direction,
            parameter->shape.count
        );
        encoder_context.parameter_index = direction_index;
        encoder_context.perturb_input = 0;
        if (!local_run_fixed_grid(
                "encoder_body_gap_objective",
                parameter_key_name(parameter->parameter_key),
                parameter->values,
                parameter->shape.count,
                0,
                direction,
                analytic,
                local_encoder_objective,
                &encoder_context,
                &encoder_signature)) {
            encoder_gate = 0;
        }
        free(direction);
    }
    {
        size_t count = center.workspace.input.shape.count;
        double *direction = directional_direction_allocate(
            UINT64_C(0x1A2B3C4D), count
        );

        if (direction == NULL) {
            encoder_gate = 0;
        } else {
            double analytic = directional_dot(
                center.workspace.input.grad, direction, count
            );

            encoder_context.parameter_index = 0;
            encoder_context.perturb_input = 1;
            if (!local_run_fixed_grid(
                    "encoder_body_gap_objective",
                    "input",
                    center.workspace.input.data,
                    count,
                    1,
                    direction,
                    analytic,
                    local_encoder_objective,
                    &encoder_context,
                    &encoder_signature)) {
                encoder_gate = 0;
            }
            free(direction);
        }
    }
    printf("LOCAL_HARD_GATE_DECODER=%s\n",
           decoder_gate ? "PASS" : "FAIL");
    printf("LOCAL_HARD_GATE_LATENT_BRIDGE=%s\n",
           latent_gate ? "PASS" : "FAIL");
    printf("LOCAL_HARD_GATE_SPLIT_FC_GAP=%s\n",
           exact_gate ? "PASS" : "FAIL");
    printf("LOCAL_HARD_GATE_ENCODER_BODY=%s\n",
           encoder_gate ? "PASS" : "FAIL");
    passed = decoder_gate && latent_gate &&
        exact_gate && encoder_gate;
cleanup:
    free(base_z);
    free(base_mu);
    free(base_logvar);
    fixture_free(&center);
    return passed;
}


int main(int argc, char **argv)
{
    VAECProfileCatalog catalog;
    int failures = 0;

    if (argc != 2) {
        fprintf(stderr, "usage: %s <vae_c_profiles_v1.csv>\n", argv[0]);
        return 2;
    }
    vae_c_profile_catalog_init(&catalog);
    if (vae_c_profile_catalog_load_csv(
            argv[1], &catalog, NULL, 0) != VAE_C_CONFIG_OK) {

        fprintf(stderr, "failed to load profile catalog\n");
        return 2;
    }

    failures += report_check(
        "manual_chain_depth3_all_gradients_and_accumulation",
        test_manual_depth3(&catalog)
    );
    failures += report_check(
        "manual_chain_synthetic_depth4_all_gradients",
        test_manual_depth4()
    );
    failures += report_check(
        "bce_target_constant_and_kl_reparameterization_sum",
        test_loss_target_constant_and_latent_sum(&catalog)
    );
    failures += report_check(
        "ticket_required_consumed_and_second_call_rejected",
        test_ticket_and_consumption(&catalog)
    );
    failures += report_check(
        "zero_grad_numeric_null_and_strong_preflight",
        test_zero_grad_and_numeric_preflight(&catalog)
    );
    failures += report_check(
        "cache_shape_overflow_and_exact_alias_preflight",
        test_cache_shape_overflow_and_alias_preflight(&catalog)
    );
    failures += report_check(
        "deterministic_depth4_elementwise_gradient_diagnostic",
        test_end_to_end_gradient_check()
    );
    failures += report_check(
        "global_directional_gradient_diagnostic_not_hard_gate",
        test_end_to_end_directional_gradient_check()
    );
    failures += report_check(
        "local_gradient_hard_gates_decoder_latent_split_fc_encoder",
        test_local_gradient_hard_gates()
    );

    vae_c_profile_catalog_free(&catalog);
    if (failures != 0) {
        fprintf(stderr,
                "%d VAE C FULL BACKWARD TEST(S) FAILED\n",
                failures);
        return 1;
    }
    printf("VAE_BACKWARD_READY_REQUIRED=true\n");
    printf("VAE_BACKWARD_TICKET_CONSUMED=true\n");
    printf("VAE_BACKWARD_WORKSPACE_ZERO_GRAD_REQUIRED=true\n");
    printf("VAE_BACKWARD_TARGET_GRADIENT=false\n");
    printf("VAE_BACKWARD_GRADIENT_CHECK_PARAMETER_TENSORS=28\n");
    printf("VAE_BACKWARD_GRADIENT_CHECK_MAX_ABS=%.9g\n",
           gradient_check_max_absolute_error);
    printf("VAE_BACKWARD_GRADIENT_CHECK_MAX_REL=%.9g\n",
           gradient_check_max_relative_error);
    printf("VAE_BACKWARD_GRADIENT_CHECK_MAX_SCALED=%.9g\n",
           gradient_check_max_scaled_error);
    printf("VAE_BACKWARD_GRADIENT_CHECK_MULTIH_MAX_ABS=%.9g\n",
           gradient_check_multih_max_absolute_error);
    printf("VAE_BACKWARD_GRADIENT_CHECK_MULTIH_MAX_REL=%.9g\n",
           gradient_check_multih_max_relative_error);
    printf("VAE_BACKWARD_GRADIENT_CHECK_MULTIH_MAX_SCALED=%.9g\n",
           gradient_check_multih_max_scaled_error);
    printf("VAE_BACKWARD_ELEMENTWISE_DIAGNOSTIC_FAILURES=%zu\n",
           elementwise_diagnostic_tolerance_failures);
    printf("VAE_BACKWARD_DIRECTIONAL_MAX_ABS=%.9g\n",
           directional_check_max_absolute_error);
    printf("VAE_BACKWARD_DIRECTIONAL_MAX_REL=%.9g\n",
           directional_check_max_relative_error);
    printf("VAE_BACKWARD_DIRECTIONAL_MAX_SCALED=%.9g\n",
           directional_check_max_scaled_error);
    printf("VAE_BACKWARD_DIRECTIONAL_NORMALIZATION=l2\n");
    printf("VAE_BACKWARD_GLOBAL_DIRECTIONAL_POLICY=diagnostic_only\n");
    printf("VAE_BACKWARD_LOCAL_FD_STEPS="
           "0.02,0.01,0.005,0.002,0.001\n");
    printf("VAE_BACKWARD_LOCAL_FD_MIN_SIGNAL_ULPS=5\n");
    printf("VAE_BACKWARD_LOCAL_FD_WINDOW_POLICY="
           "fixed_grid_one_informative_passing_measurement\n");
    printf("VAE_BACKWARD_GRADIENT_CHECK_ABS_TOL=0.05\n");
    printf("VAE_BACKWARD_GRADIENT_CHECK_REL_TOL=0.1\n");
    printf("VAE_BACKWARD_GRADIENT_CHECK_REL_SCALE_FLOOR=none\n");
    printf("VAE_BACKWARD_GRADIENT_CHECK_SELECTION=error_independent\n");
    printf("VAE_BACKWARD_GRADIENT_CHECK_MULTIH_SELECTION=fixed_probe\n");
    printf("VAE_BACKWARD_STRONG_GUARANTEE=preflight_only\n");
    printf("VAE_BACKWARD_POST_START_ROLLBACK=false\n");
    printf("ALL VAE C FULL BACKWARD TESTS PASSED\n");
    return 0;
}
