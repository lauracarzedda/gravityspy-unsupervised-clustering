#include "vae_c_model_forward.h"

#include "vae_c_latent.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define TEST_BATCH_NORM_COUNT \
    (VAE_C_MAX_CONV_DEPTH + 1 + VAE_C_MAX_DECODER_HIDDEN_CONVS)

#define TEST_MODEL_ALIAS_REGION_CAPACITY \
    (4 * VAE_C_MODEL_MAX_PARAMETERS + 7 * TEST_BATCH_NORM_COUNT + 1)

#define TEST_WORKSPACE_ALIAS_REGION_CAPACITY \
    (2 * VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS)


typedef struct {
    const void *pointer;
    size_t byte_count;
    unsigned char *copy;
} ByteSnapshot;


typedef struct {
    VAECModel metadata;
    ByteSnapshot parameters[VAE_C_MODEL_MAX_PARAMETERS];
    size_t parameter_count;
    ByteSnapshot batch_norms[TEST_BATCH_NORM_COUNT];
    ByteSnapshot max_pool;
    bool captured;
} ModelSnapshot;


typedef struct {
    VAECModelWorkspace metadata;
    ByteSnapshot tensors[VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS];
    size_t tensor_count;
    bool captured;
} WorkspaceSnapshot;


typedef enum {
    TEST_MODEL_REGION_PARAMETER_VALUES = 0,
    TEST_MODEL_REGION_PARAMETER_GRADIENTS,
    TEST_MODEL_REGION_PARAMETER_ADAM_M,
    TEST_MODEL_REGION_PARAMETER_ADAM_V,
    TEST_MODEL_REGION_BATCH_NORM_RUNNING_MEAN,
    TEST_MODEL_REGION_BATCH_NORM_RUNNING_VARIANCE,
    TEST_MODEL_REGION_BATCH_NORM_BATCH_MEAN,
    TEST_MODEL_REGION_BATCH_NORM_BATCH_INV_STD,
    TEST_MODEL_REGION_BATCH_NORM_WORK_MEAN,
    TEST_MODEL_REGION_BATCH_NORM_WORK_INV_STD,
    TEST_MODEL_REGION_BATCH_NORM_WORK_UNBIASED_VARIANCE,
    TEST_MODEL_REGION_MAX_POOL_ARGMAX
} TestModelRegionKind;


typedef enum {
    TEST_WORKSPACE_REGION_DATA = 0,
    TEST_WORKSPACE_REGION_GRAD
} TestWorkspaceRegionKind;


typedef struct {
    const void *pointer;
    TestModelRegionKind kind;
    const void *owner;
} TestModelAliasRegion;


typedef struct {
    const void *pointer;
    TestWorkspaceRegionKind kind;
    const VAECTensor *tensor;
} TestWorkspaceAliasRegion;


typedef struct {
    size_t count;
    TestModelRegionKind model_kind;
    TestWorkspaceRegionKind workspace_kind;
    const void *model_owner;
    const VAECTensor *workspace_tensor;
} TestExactAliasSummary;


typedef struct {
    float *running_mean;
    float *running_variance;
    float *batch_mean;
    float *batch_inv_std;
    float *work_mean;
    float *work_inv_std;
    float *work_unbiased_variance;
    float *owned_block;
} TestBatchNormBufferPointers;


static int report_check(const char *name, int passed)
{
    if (passed) {
        printf("[PASS] %s\n", name);
        return 0;
    }
    fprintf(stderr, "[FAIL] %s\n", name);
    return 1;
}


static void byte_snapshot_init(ByteSnapshot *snapshot)
{
    if (snapshot != NULL) {
        memset(snapshot, 0, sizeof(*snapshot));
    }
}


static void byte_snapshot_free(ByteSnapshot *snapshot)
{
    if (snapshot != NULL) {
        free(snapshot->copy);
        byte_snapshot_init(snapshot);
    }
}


static int byte_snapshot_capture(
    const void *pointer,
    size_t byte_count,
    ByteSnapshot *snapshot
)
{
    unsigned char *copy;

    if (pointer == NULL || byte_count == 0 || snapshot == NULL ||
        snapshot->copy != NULL) {

        return 0;
    }
    copy = (unsigned char *)malloc(byte_count);
    if (copy == NULL) {
        return 0;
    }
    memcpy(copy, pointer, byte_count);
    snapshot->pointer = pointer;
    snapshot->byte_count = byte_count;
    snapshot->copy = copy;
    return 1;
}


static int byte_snapshot_matches(
    const void *pointer,
    size_t byte_count,
    const ByteSnapshot *snapshot
)
{
    return snapshot != NULL && snapshot->copy != NULL &&
        pointer == snapshot->pointer &&
        byte_count == snapshot->byte_count &&
        memcmp(pointer, snapshot->copy, byte_count) == 0;
}


static int shape_fields_equal(
    const VAECShape *left,
    const VAECShape *right
)
{
    size_t index;

    if (left == NULL || right == NULL || left->rank != right->rank ||
        left->count != right->count) {

        return 0;
    }
    for (index = 0; index < VAE_C_TENSOR_MAX_RANK; index++) {
        if (left->dims[index] != right->dims[index] ||
            left->strides[index] != right->strides[index]) {

            return 0;
        }
    }
    return 1;
}


static int config_fields_equal(
    const VAECConfig *left,
    const VAECConfig *right
)
{
    size_t index;

    if (left == NULL || right == NULL ||
        left->input_channels != right->input_channels ||
        left->input_side != right->input_side ||
        left->conv_depth != right->conv_depth ||
        left->pool_side != right->pool_side ||
        left->latent_dim != right->latent_dim ||
        left->bottleneck_side != right->bottleneck_side ||
        left->decoder_fc_features != right->decoder_fc_features ||
        left->decoder_upsample_count != right->decoder_upsample_count ||
        left->decoder_hidden_conv_count !=
            right->decoder_hidden_conv_count ||
        left->values_per_sample != right->values_per_sample ||
        left->trainable_parameters_bn1d !=
            right->trainable_parameters_bn1d ||
        left->forward_mac_per_sample != right->forward_mac_per_sample) {

        return 0;
    }
    for (index = 0; index < VAE_C_MAX_CONV_DEPTH; index++) {
        if (left->channels[index] != right->channels[index] ||
            left->encoder_conv_sides[index] !=
                right->encoder_conv_sides[index]) {

            return 0;
        }
    }
    for (index = 0; index < VAE_C_MAX_DECODER_UPSAMPLES; index++) {
        if (left->decoder_upsample_targets[index] !=
            right->decoder_upsample_targets[index]) {

            return 0;
        }
    }
    for (index = 0; index < VAE_C_MAX_DECODER_HIDDEN_CONVS; index++) {
        if (left->decoder_hidden_conv_sides[index] !=
            right->decoder_hidden_conv_sides[index]) {

            return 0;
        }
    }
    return 1;
}


static int parameter_metadata_exact(
    const VAECParameter *left,
    const VAECParameter *right
)
{
    return left != NULL && right != NULL &&
        shape_fields_equal(&left->shape, &right->shape) &&
        left->parameter_key == right->parameter_key &&
        left->values == right->values &&
        left->gradients == right->gradients &&
        left->adam_m == right->adam_m &&
        left->adam_v == right->adam_v &&
        left->owned_block == right->owned_block &&
        left->owned_bytes == right->owned_bytes;
}


static int parameter_layout_valid(const VAECParameter *parameter)
{
    size_t count;

    if (parameter == NULL) {
        return 0;
    }
    count = parameter->shape.count;
    if (count == 0) {
        return parameter->parameter_key == UINT64_C(0) &&
            parameter->values == NULL && parameter->gradients == NULL &&
            parameter->adam_m == NULL && parameter->adam_v == NULL &&
            parameter->owned_block == NULL && parameter->owned_bytes == 0;
    }
    return parameter->parameter_key != UINT64_C(0) &&
        parameter->owned_block != NULL &&
        parameter->values == parameter->owned_block &&
        parameter->gradients == parameter->values + count &&
        parameter->adam_m == parameter->gradients + count &&
        parameter->adam_v == parameter->adam_m + count &&
        parameter->owned_bytes == 4 * count * sizeof(float);
}


static int parameter_state_equal(
    const VAECParameter *left,
    const VAECParameter *right
)
{
    if (left == NULL || right == NULL ||
        !shape_fields_equal(&left->shape, &right->shape) ||
        left->parameter_key != right->parameter_key ||
        left->owned_bytes != right->owned_bytes ||
        !parameter_layout_valid(left) ||
        !parameter_layout_valid(right)) {

        return 0;
    }
    return left->owned_bytes == 0 ||
        memcmp(left->owned_block,
               right->owned_block,
               left->owned_bytes) == 0;
}


static int conv_metadata_exact(
    const VAECConv2D *left,
    const VAECConv2D *right
)
{
    return left != NULL && right != NULL &&
        left->in_channels == right->in_channels &&
        left->out_channels == right->out_channels &&
        left->kernel_size == right->kernel_size &&
        left->stride == right->stride &&
        left->padding == right->padding &&
        left->with_bias == right->with_bias &&
        parameter_metadata_exact(&left->weights, &right->weights) &&
        parameter_metadata_exact(&left->bias, &right->bias);
}


static int conv_state_equal(
    const VAECConv2D *left,
    const VAECConv2D *right
)
{
    return left != NULL && right != NULL &&
        left->in_channels == right->in_channels &&
        left->out_channels == right->out_channels &&
        left->kernel_size == right->kernel_size &&
        left->stride == right->stride &&
        left->padding == right->padding &&
        left->with_bias == right->with_bias &&
        parameter_state_equal(&left->weights, &right->weights) &&
        parameter_state_equal(&left->bias, &right->bias);
}


static int dense_metadata_exact(
    const VAECDense *left,
    const VAECDense *right
)
{
    return left != NULL && right != NULL &&
        left->in_features == right->in_features &&
        left->out_features == right->out_features &&
        left->with_bias == right->with_bias &&
        parameter_metadata_exact(&left->weights, &right->weights) &&
        parameter_metadata_exact(&left->bias, &right->bias);
}


static int dense_state_equal(
    const VAECDense *left,
    const VAECDense *right
)
{
    return left != NULL && right != NULL &&
        left->in_features == right->in_features &&
        left->out_features == right->out_features &&
        left->with_bias == right->with_bias &&
        parameter_state_equal(&left->weights, &right->weights) &&
        parameter_state_equal(&left->bias, &right->bias);
}


static int batch_norm_metadata_exact(
    const VAECBatchNorm *left,
    const VAECBatchNorm *right
)
{
    return left != NULL && right != NULL &&
        left->kind == right->kind &&
        left->num_features == right->num_features &&
        left->epsilon == right->epsilon &&
        left->momentum == right->momentum &&
        parameter_metadata_exact(&left->gamma, &right->gamma) &&
        parameter_metadata_exact(&left->beta, &right->beta) &&
        left->running_mean == right->running_mean &&
        left->running_variance == right->running_variance &&
        left->batch_mean == right->batch_mean &&
        left->batch_inv_std == right->batch_inv_std &&
        left->work_mean == right->work_mean &&
        left->work_inv_std == right->work_inv_std &&
        left->work_unbiased_variance == right->work_unbiased_variance &&
        left->owned_block == right->owned_block &&
        left->owned_bytes == right->owned_bytes &&
        left->last_sample_count == right->last_sample_count &&
        shape_fields_equal(&left->last_input_shape,
                           &right->last_input_shape) &&
        shape_fields_equal(&left->last_output_shape,
                           &right->last_output_shape) &&
        left->cache_valid == right->cache_valid;
}


static int batch_norm_layout_valid(const VAECBatchNorm *layer)
{
    size_t count;

    if (layer == NULL) {
        return 0;
    }
    count = layer->num_features;
    if (count == 0) {
        return layer->running_mean == NULL &&
            layer->running_variance == NULL && layer->batch_mean == NULL &&
            layer->batch_inv_std == NULL && layer->work_mean == NULL &&
            layer->work_inv_std == NULL &&
            layer->work_unbiased_variance == NULL &&
            layer->owned_block == NULL && layer->owned_bytes == 0;
    }
    return layer->owned_block != NULL &&
        layer->running_mean == layer->owned_block &&
        layer->running_variance == layer->running_mean + count &&
        layer->batch_mean == layer->running_variance + count &&
        layer->batch_inv_std == layer->batch_mean + count &&
        layer->work_mean == layer->batch_inv_std + count &&
        layer->work_inv_std == layer->work_mean + count &&
        layer->work_unbiased_variance == layer->work_inv_std + count &&
        layer->owned_bytes == 7 * count * sizeof(float);
}


static int batch_norm_state_equal(
    const VAECBatchNorm *left,
    const VAECBatchNorm *right
)
{
    if (left == NULL || right == NULL ||
        left->kind != right->kind ||
        left->num_features != right->num_features ||
        left->epsilon != right->epsilon ||
        left->momentum != right->momentum ||
        !parameter_state_equal(&left->gamma, &right->gamma) ||
        !parameter_state_equal(&left->beta, &right->beta) ||
        left->owned_bytes != right->owned_bytes ||
        !batch_norm_layout_valid(left) ||
        !batch_norm_layout_valid(right) ||
        left->last_sample_count != right->last_sample_count ||
        !shape_fields_equal(&left->last_input_shape,
                            &right->last_input_shape) ||
        !shape_fields_equal(&left->last_output_shape,
                            &right->last_output_shape) ||
        left->cache_valid != right->cache_valid) {

        return 0;
    }
    return left->owned_bytes == 0 ||
        memcmp(left->owned_block,
               right->owned_block,
               left->owned_bytes) == 0;
}


static int max_pool_metadata_exact(
    const VAECMaxPool2D *left,
    const VAECMaxPool2D *right
)
{
    return left != NULL && right != NULL &&
        left->argmax_indices == right->argmax_indices &&
        left->capacity == right->capacity &&
        left->owned_bytes == right->owned_bytes &&
        left->last_count == right->last_count &&
        shape_fields_equal(&left->last_input_shape,
                           &right->last_input_shape) &&
        shape_fields_equal(&left->last_output_shape,
                           &right->last_output_shape) &&
        left->cache_valid == right->cache_valid;
}


static int max_pool_state_equal(
    const VAECMaxPool2D *left,
    const VAECMaxPool2D *right
)
{
    if (left == NULL || right == NULL ||
        (left->argmax_indices == NULL) !=
            (right->argmax_indices == NULL) ||
        left->capacity != right->capacity ||
        left->owned_bytes != right->owned_bytes ||
        left->last_count != right->last_count ||
        !shape_fields_equal(&left->last_input_shape,
                            &right->last_input_shape) ||
        !shape_fields_equal(&left->last_output_shape,
                            &right->last_output_shape) ||
        left->cache_valid != right->cache_valid) {

        return 0;
    }
    return left->last_count == 0 ||
        memcmp(left->argmax_indices,
               right->argmax_indices,
               left->last_count * sizeof(size_t)) == 0;
}


static int tensor_metadata_exact(
    const VAECTensor *left,
    const VAECTensor *right
)
{
    return left != NULL && right != NULL &&
        shape_fields_equal(&left->shape, &right->shape) &&
        left->data == right->data && left->grad == right->grad &&
        left->owned_block == right->owned_block &&
        left->owned_bytes == right->owned_bytes &&
        left->owns_memory == right->owns_memory &&
        left->has_grad == right->has_grad;
}


static int tensor_layout_valid(const VAECTensor *tensor)
{
    size_t count;
    size_t multiplier;

    if (tensor == NULL) {
        return 0;
    }
    count = tensor->shape.count;
    if (count == 0) {
        return tensor->data == NULL && tensor->grad == NULL &&
            tensor->owned_block == NULL && tensor->owned_bytes == 0 &&
            !tensor->owns_memory && !tensor->has_grad;
    }
    if (!tensor->owns_memory) {
        return tensor->data != NULL && tensor->owned_block == NULL &&
            tensor->owned_bytes == 0 &&
            (tensor->has_grad
                ? tensor->grad != NULL
                : tensor->grad == NULL);
    }
    multiplier = tensor->has_grad ? 2 : 1;
    return tensor->data == tensor->owned_block &&
        (tensor->has_grad
            ? tensor->grad == tensor->data + count
            : tensor->grad == NULL) &&
        tensor->owned_bytes == multiplier * count * sizeof(float);
}


static int tensor_state_equal(
    const VAECTensor *left,
    const VAECTensor *right
)
{
    size_t bytes;

    if (left == NULL || right == NULL ||
        !shape_fields_equal(&left->shape, &right->shape) ||
        left->owned_bytes != right->owned_bytes ||
        left->owns_memory != right->owns_memory ||
        left->has_grad != right->has_grad ||
        !tensor_layout_valid(left) || !tensor_layout_valid(right)) {

        return 0;
    }
    bytes = left->shape.count * sizeof(float);
    if (bytes != 0 && memcmp(left->data, right->data, bytes) != 0) {
        return 0;
    }
    return !left->has_grad ||
        memcmp(left->grad, right->grad, bytes) == 0;
}


static int model_metadata_exact(
    const VAECModel *left,
    const VAECModel *right
)
{
    size_t index;

    if (left == NULL || right == NULL ||
        memcmp(left->profile_id,
               right->profile_id,
               VAE_C_PROFILE_TEXT_CAP) != 0 ||
        !config_fields_equal(&left->config, &right->config) ||
        left->max_batch_size != right->max_batch_size ||
        !max_pool_metadata_exact(&left->encoder_max_pool,
                                 &right->encoder_max_pool) ||
        !dense_metadata_exact(&left->encoder_fc, &right->encoder_fc) ||
        !dense_metadata_exact(&left->decoder_fc, &right->decoder_fc) ||
        !batch_norm_metadata_exact(&left->decoder_fc_batch_norm,
                                   &right->decoder_fc_batch_norm) ||
        !conv_metadata_exact(&left->decoder_output_conv,
                             &right->decoder_output_conv) ||
        left->parameter_count != right->parameter_count ||
        left->trainable_value_count != right->trainable_value_count ||
        left->initialization_seed != right->initialization_seed ||
        left->vae_cache_generation != right->vae_cache_generation ||
        left->parameters_initialized != right->parameters_initialized ||
        left->allocated != right->allocated) {

        return 0;
    }
    for (index = 0; index < VAE_C_MODEL_MAX_ENCODER_CONVS; index++) {
        if (!conv_metadata_exact(&left->encoder_convs[index],
                                 &right->encoder_convs[index]) ||
            !batch_norm_metadata_exact(
                &left->encoder_batch_norms[index],
                &right->encoder_batch_norms[index])) {

            return 0;
        }
    }
    for (index = 0; index < VAE_C_MODEL_MAX_DECODER_HIDDEN_CONVS;
         index++) {

        if (!conv_metadata_exact(&left->decoder_hidden_convs[index],
                                 &right->decoder_hidden_convs[index]) ||
            !batch_norm_metadata_exact(
                &left->decoder_hidden_batch_norms[index],
                &right->decoder_hidden_batch_norms[index])) {

            return 0;
        }
    }
    for (index = 0; index < VAE_C_MODEL_MAX_PARAMETERS; index++) {
        if (left->parameters[index] != right->parameters[index]) {
            return 0;
        }
    }
    return 1;
}


static int model_state_equal(
    const VAECModel *left,
    const VAECModel *right
)
{
    size_t index;

    if (left == NULL || right == NULL ||
        memcmp(left->profile_id,
               right->profile_id,
               VAE_C_PROFILE_TEXT_CAP) != 0 ||
        !config_fields_equal(&left->config, &right->config) ||
        left->max_batch_size != right->max_batch_size ||
        !max_pool_state_equal(&left->encoder_max_pool,
                              &right->encoder_max_pool) ||
        !dense_state_equal(&left->encoder_fc, &right->encoder_fc) ||
        !dense_state_equal(&left->decoder_fc, &right->decoder_fc) ||
        !batch_norm_state_equal(&left->decoder_fc_batch_norm,
                                &right->decoder_fc_batch_norm) ||
        !conv_state_equal(&left->decoder_output_conv,
                          &right->decoder_output_conv) ||
        left->parameter_count != right->parameter_count ||
        left->trainable_value_count != right->trainable_value_count ||
        left->initialization_seed != right->initialization_seed ||
        left->vae_cache_generation != right->vae_cache_generation ||
        left->parameters_initialized != right->parameters_initialized ||
        left->allocated != right->allocated) {

        return 0;
    }
    for (index = 0; index < VAE_C_MODEL_MAX_ENCODER_CONVS; index++) {
        if (!conv_state_equal(&left->encoder_convs[index],
                              &right->encoder_convs[index]) ||
            !batch_norm_state_equal(&left->encoder_batch_norms[index],
                                    &right->encoder_batch_norms[index])) {

            return 0;
        }
    }
    for (index = 0; index < VAE_C_MODEL_MAX_DECODER_HIDDEN_CONVS;
         index++) {

        if (!conv_state_equal(&left->decoder_hidden_convs[index],
                              &right->decoder_hidden_convs[index]) ||
            !batch_norm_state_equal(
                &left->decoder_hidden_batch_norms[index],
                &right->decoder_hidden_batch_norms[index])) {

            return 0;
        }
    }
    for (index = 0; index < VAE_C_MODEL_MAX_PARAMETERS; index++) {
        const VAECParameter *a = left->parameters[index];
        const VAECParameter *b = right->parameters[index];

        if ((a == NULL) != (b == NULL) ||
            (a != NULL && !parameter_state_equal(a, b))) {

            return 0;
        }
    }
    return 1;
}


static size_t collect_all_workspace_tensors(
    const VAECModelWorkspace *workspace,
    const VAECTensor *tensors[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ]
);


static int workspace_metadata_exact(
    const VAECModelWorkspace *left,
    const VAECModelWorkspace *right
)
{
    size_t index;

    if (left == NULL || right == NULL ||
        memcmp(left->profile_id,
               right->profile_id,
               VAE_C_PROFILE_TEXT_CAP) != 0 ||
        !config_fields_equal(&left->config, &right->config) ||
        left->batch_size != right->batch_size ||
        !tensor_metadata_exact(&left->input, &right->input) ||
        !tensor_metadata_exact(&left->encoder_pool_output,
                               &right->encoder_pool_output) ||
        !tensor_metadata_exact(&left->encoder_global_avg_output,
                               &right->encoder_global_avg_output) ||
        !tensor_metadata_exact(&left->encoder_packed_output,
                               &right->encoder_packed_output) ||
        !tensor_metadata_exact(&left->mu, &right->mu) ||
        !tensor_metadata_exact(&left->logvar, &right->logvar) ||
        !tensor_metadata_exact(&left->epsilon, &right->epsilon) ||
        !tensor_metadata_exact(&left->z, &right->z) ||
        !tensor_metadata_exact(&left->decoder_fc_output,
                               &right->decoder_fc_output) ||
        !tensor_metadata_exact(&left->decoder_fc_batch_norm_output,
                               &right->decoder_fc_batch_norm_output) ||
        !tensor_metadata_exact(&left->decoder_fc_relu_output,
                               &right->decoder_fc_relu_output) ||
        !tensor_metadata_exact(&left->decoder_reshape_view,
                               &right->decoder_reshape_view) ||
        !tensor_metadata_exact(&left->logits, &right->logits) ||
        left->owned_tensor_count != right->owned_tensor_count ||
        left->logical_value_count != right->logical_value_count ||
        left->storage_float_count != right->storage_float_count ||
        left->storage_bytes != right->storage_bytes ||
        left->vae_backward_model != right->vae_backward_model ||
        left->vae_backward_generation != right->vae_backward_generation ||
        left->allocated != right->allocated) {

        return 0;
    }
    for (index = 0; index < VAE_C_MODEL_MAX_ENCODER_CONVS; index++) {
        if (!tensor_metadata_exact(&left->encoder_conv_outputs[index],
                                   &right->encoder_conv_outputs[index]) ||
            !tensor_metadata_exact(
                &left->encoder_batch_norm_outputs[index],
                &right->encoder_batch_norm_outputs[index]) ||
            !tensor_metadata_exact(&left->encoder_relu_outputs[index],
                                   &right->encoder_relu_outputs[index])) {

            return 0;
        }
    }
    for (index = 0; index < VAE_C_MAX_DECODER_UPSAMPLES; index++) {
        if (!tensor_metadata_exact(&left->decoder_upsample_outputs[index],
                                   &right->decoder_upsample_outputs[index])) {

            return 0;
        }
    }
    for (index = 0; index < VAE_C_MODEL_MAX_DECODER_HIDDEN_CONVS;
         index++) {

        if (!tensor_metadata_exact(
                &left->decoder_hidden_conv_outputs[index],
                &right->decoder_hidden_conv_outputs[index]) ||
            !tensor_metadata_exact(
                &left->decoder_hidden_batch_norm_outputs[index],
                &right->decoder_hidden_batch_norm_outputs[index]) ||
            !tensor_metadata_exact(
                &left->decoder_hidden_relu_outputs[index],
                &right->decoder_hidden_relu_outputs[index])) {

            return 0;
        }
    }
    return 1;
}


static int workspace_state_equal(
    const VAECModelWorkspace *left,
    const VAECModelWorkspace *right
)
{
    const VAECTensor *left_tensors[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ];
    const VAECTensor *right_tensors[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ];
    size_t left_count;
    size_t right_count;
    size_t index;

    if (left == NULL || right == NULL ||
        memcmp(left->profile_id,
               right->profile_id,
               VAE_C_PROFILE_TEXT_CAP) != 0 ||
        !config_fields_equal(&left->config, &right->config) ||
        left->batch_size != right->batch_size ||
        left->owned_tensor_count != right->owned_tensor_count ||
        left->logical_value_count != right->logical_value_count ||
        left->storage_float_count != right->storage_float_count ||
        left->storage_bytes != right->storage_bytes ||
        left->vae_backward_model != right->vae_backward_model ||
        left->vae_backward_generation != right->vae_backward_generation ||
        left->allocated != right->allocated ||
        !tensor_state_equal(&left->decoder_reshape_view,
                            &right->decoder_reshape_view) ||
        left->decoder_reshape_view.data !=
            left->decoder_fc_relu_output.data ||
        right->decoder_reshape_view.data !=
            right->decoder_fc_relu_output.data ||
        left->decoder_reshape_view.grad !=
            left->decoder_fc_relu_output.grad ||
        right->decoder_reshape_view.grad !=
            right->decoder_fc_relu_output.grad) {

        return 0;
    }
    left_count = collect_all_workspace_tensors(left, left_tensors);
    right_count = collect_all_workspace_tensors(right, right_tensors);
    if (left_count != right_count ||
        left_count != VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS) {

        return 0;
    }
    for (index = 0; index < left_count; index++) {
        if (!tensor_state_equal(left_tensors[index],
                                right_tensors[index])) {

            return 0;
        }
    }
    return 1;
}


static size_t collect_all_batch_norms(
    const VAECModel *model,
    const VAECBatchNorm *layers[TEST_BATCH_NORM_COUNT]
)
{
    size_t count = 0;
    size_t index;

    for (index = 0; index < VAE_C_MAX_CONV_DEPTH; index++) {
        layers[count++] = &model->encoder_batch_norms[index];
    }
    layers[count++] = &model->decoder_fc_batch_norm;
    for (index = 0;
         index < VAE_C_MAX_DECODER_HIDDEN_CONVS;
         index++) {

        layers[count++] = &model->decoder_hidden_batch_norms[index];
    }
    return count;
}


static size_t collect_all_workspace_tensors(
    const VAECModelWorkspace *workspace,
    const VAECTensor *tensors[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ]
)
{
    size_t count = 0;
    size_t index;

    tensors[count++] = &workspace->input;
    for (index = 0; index < VAE_C_MAX_CONV_DEPTH; index++) {
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
    for (index = 0; index < VAE_C_MAX_DECODER_UPSAMPLES; index++) {
        tensors[count++] = &workspace->decoder_upsample_outputs[index];
    }
    for (index = 0;
         index < VAE_C_MAX_DECODER_HIDDEN_CONVS;
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


static int test_size_multiply(
    size_t left,
    size_t right,
    size_t *result
)
{
    if (result == NULL || (left != 0 && right > SIZE_MAX / left)) {
        return 0;
    }
    *result = left * right;
    return 1;
}


static int test_float_values_are_finite(
    const float *values,
    size_t count
)
{
    size_t index;

    if (values == NULL) {
        return 0;
    }
    for (index = 0; index < count; index++) {
        if (!isfinite(values[index])) {
            return 0;
        }
    }
    return 1;
}


static int test_append_model_alias_region(
    TestModelAliasRegion regions[TEST_MODEL_ALIAS_REGION_CAPACITY],
    size_t *count,
    const void *pointer,
    TestModelRegionKind kind,
    const void *owner
)
{
    if (regions == NULL || count == NULL || pointer == NULL ||
        owner == NULL || *count >= TEST_MODEL_ALIAS_REGION_CAPACITY) {

        return 0;
    }
    regions[*count].pointer = pointer;
    regions[*count].kind = kind;
    regions[*count].owner = owner;
    (*count)++;
    return 1;
}


static int test_append_workspace_alias_region(
    TestWorkspaceAliasRegion regions[
        TEST_WORKSPACE_ALIAS_REGION_CAPACITY
    ],
    size_t *count,
    const void *pointer,
    TestWorkspaceRegionKind kind,
    const VAECTensor *tensor
)
{
    if (regions == NULL || count == NULL || pointer == NULL ||
        tensor == NULL ||
        *count >= TEST_WORKSPACE_ALIAS_REGION_CAPACITY) {

        return 0;
    }
    regions[*count].pointer = pointer;
    regions[*count].kind = kind;
    regions[*count].tensor = tensor;
    (*count)++;
    return 1;
}


static int test_collect_model_alias_regions(
    const VAECModel *model,
    TestModelAliasRegion regions[TEST_MODEL_ALIAS_REGION_CAPACITY],
    size_t *region_count
)
{
    const VAECBatchNorm *batch_norms[TEST_BATCH_NORM_COUNT];
    size_t batch_norm_count;
    size_t count = 0;
    size_t index;

    if (model == NULL || regions == NULL || region_count == NULL ||
        model->parameter_count > VAE_C_MODEL_MAX_PARAMETERS) {

        return 0;
    }
    for (index = 0; index < model->parameter_count; index++) {
        const VAECParameter *parameter = model->parameters[index];

        if (parameter == NULL ||
            !test_append_model_alias_region(
                regions, &count, parameter->values,
                TEST_MODEL_REGION_PARAMETER_VALUES, parameter) ||
            !test_append_model_alias_region(
                regions, &count, parameter->gradients,
                TEST_MODEL_REGION_PARAMETER_GRADIENTS, parameter) ||
            !test_append_model_alias_region(
                regions, &count, parameter->adam_m,
                TEST_MODEL_REGION_PARAMETER_ADAM_M, parameter) ||
            !test_append_model_alias_region(
                regions, &count, parameter->adam_v,
                TEST_MODEL_REGION_PARAMETER_ADAM_V, parameter)) {

            return 0;
        }
    }

    batch_norm_count = collect_all_batch_norms(model, batch_norms);
    if (batch_norm_count != TEST_BATCH_NORM_COUNT) {
        return 0;
    }
    for (index = 0; index < batch_norm_count; index++) {
        const VAECBatchNorm *layer = batch_norms[index];

        if (layer == NULL) {
            return 0;
        }
        if (layer->owned_block == NULL) {
            if (layer->num_features != 0 || layer->running_mean != NULL ||
                layer->running_variance != NULL ||
                layer->batch_mean != NULL ||
                layer->batch_inv_std != NULL || layer->work_mean != NULL ||
                layer->work_inv_std != NULL ||
                layer->work_unbiased_variance != NULL) {

                return 0;
            }
            continue;
        }
        if (!test_append_model_alias_region(
                regions, &count, layer->running_mean,
                TEST_MODEL_REGION_BATCH_NORM_RUNNING_MEAN, layer) ||
            !test_append_model_alias_region(
                regions, &count, layer->running_variance,
                TEST_MODEL_REGION_BATCH_NORM_RUNNING_VARIANCE, layer) ||
            !test_append_model_alias_region(
                regions, &count, layer->batch_mean,
                TEST_MODEL_REGION_BATCH_NORM_BATCH_MEAN, layer) ||
            !test_append_model_alias_region(
                regions, &count, layer->batch_inv_std,
                TEST_MODEL_REGION_BATCH_NORM_BATCH_INV_STD, layer) ||
            !test_append_model_alias_region(
                regions, &count, layer->work_mean,
                TEST_MODEL_REGION_BATCH_NORM_WORK_MEAN, layer) ||
            !test_append_model_alias_region(
                regions, &count, layer->work_inv_std,
                TEST_MODEL_REGION_BATCH_NORM_WORK_INV_STD, layer) ||
            !test_append_model_alias_region(
                regions, &count, layer->work_unbiased_variance,
                TEST_MODEL_REGION_BATCH_NORM_WORK_UNBIASED_VARIANCE,
                layer)) {

            return 0;
        }
    }
    if (!test_append_model_alias_region(
            regions, &count, model->encoder_max_pool.argmax_indices,
            TEST_MODEL_REGION_MAX_POOL_ARGMAX,
            &model->encoder_max_pool)) {

        return 0;
    }
    *region_count = count;
    return 1;
}


static int test_collect_workspace_alias_regions(
    const VAECModelWorkspace *workspace,
    TestWorkspaceAliasRegion regions[
        TEST_WORKSPACE_ALIAS_REGION_CAPACITY
    ],
    size_t *region_count
)
{
    const VAECTensor *tensors[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ];
    size_t tensor_count;
    size_t count = 0;
    size_t index;

    if (workspace == NULL || regions == NULL || region_count == NULL) {
        return 0;
    }
    tensor_count = collect_all_workspace_tensors(workspace, tensors);
    if (tensor_count != VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS) {
        return 0;
    }
    for (index = 0; index < tensor_count; index++) {
        const VAECTensor *tensor = tensors[index];

        if (tensor == NULL) {
            return 0;
        }
        if (tensor->owned_block == NULL) {
            if (tensor->data != NULL || tensor->grad != NULL) {
                return 0;
            }
            continue;
        }
        if (!test_append_workspace_alias_region(
                regions, &count, tensor->data,
                TEST_WORKSPACE_REGION_DATA, tensor) ||
            (tensor->grad != NULL &&
             !test_append_workspace_alias_region(
                 regions, &count, tensor->grad,
                 TEST_WORKSPACE_REGION_GRAD, tensor))) {

            return 0;
        }
    }
    *region_count = count;
    return 1;
}


static int test_summarize_exact_cross_domain_aliases(
    const VAECModel *model,
    const VAECModelWorkspace *workspace,
    TestExactAliasSummary *summary
)
{
    TestModelAliasRegion model_regions[
        TEST_MODEL_ALIAS_REGION_CAPACITY
    ];
    TestWorkspaceAliasRegion workspace_regions[
        TEST_WORKSPACE_ALIAS_REGION_CAPACITY
    ];
    size_t model_count;
    size_t workspace_count;
    size_t model_index;
    size_t workspace_index;

    if (summary == NULL ||
        !test_collect_model_alias_regions(
            model, model_regions, &model_count) ||
        !test_collect_workspace_alias_regions(
            workspace, workspace_regions, &workspace_count)) {

        return 0;
    }
    memset(summary, 0, sizeof(*summary));
    for (workspace_index = 0;
         workspace_index < workspace_count;
         workspace_index++) {

        for (model_index = 0; model_index < model_count; model_index++) {
            if (workspace_regions[workspace_index].pointer ==
                model_regions[model_index].pointer) {

                if (summary->count == SIZE_MAX) {
                    return 0;
                }
                if (summary->count == 0) {
                    summary->model_kind = model_regions[model_index].kind;
                    summary->workspace_kind =
                        workspace_regions[workspace_index].kind;
                    summary->model_owner =
                        model_regions[model_index].owner;
                    summary->workspace_tensor =
                        workspace_regions[workspace_index].tensor;
                }
                summary->count++;
            }
        }
    }
    return 1;
}


static int test_exact_alias_is(
    const TestExactAliasSummary *summary,
    TestModelRegionKind model_kind,
    const void *model_owner,
    TestWorkspaceRegionKind workspace_kind,
    const VAECTensor *workspace_tensor
)
{
    return summary != NULL && summary->count == 1 &&
        summary->model_kind == model_kind &&
        summary->model_owner == model_owner &&
        summary->workspace_kind == workspace_kind &&
        summary->workspace_tensor == workspace_tensor;
}


static void test_batch_norm_buffer_pointers_capture(
    const VAECBatchNorm *layer,
    TestBatchNormBufferPointers *pointers
)
{
    if (layer == NULL || pointers == NULL) {
        return;
    }
    pointers->running_mean = layer->running_mean;
    pointers->running_variance = layer->running_variance;
    pointers->batch_mean = layer->batch_mean;
    pointers->batch_inv_std = layer->batch_inv_std;
    pointers->work_mean = layer->work_mean;
    pointers->work_inv_std = layer->work_inv_std;
    pointers->work_unbiased_variance = layer->work_unbiased_variance;
    pointers->owned_block = layer->owned_block;
}


static void test_batch_norm_buffer_pointers_restore(
    VAECBatchNorm *layer,
    const TestBatchNormBufferPointers *pointers
)
{
    if (layer == NULL || pointers == NULL) {
        return;
    }
    layer->running_mean = pointers->running_mean;
    layer->running_variance = pointers->running_variance;
    layer->batch_mean = pointers->batch_mean;
    layer->batch_inv_std = pointers->batch_inv_std;
    layer->work_mean = pointers->work_mean;
    layer->work_inv_std = pointers->work_inv_std;
    layer->work_unbiased_variance = pointers->work_unbiased_variance;
    layer->owned_block = pointers->owned_block;
}


static int test_batch_norm_buffers_rebind(
    VAECBatchNorm *layer,
    float *block,
    size_t available_float_count
)
{
    size_t required_float_count;

    if (layer == NULL || block == NULL || layer->num_features == 0 ||
        !test_size_multiply(
            layer->num_features, 7, &required_float_count) ||
        available_float_count < required_float_count ||
        !test_float_values_are_finite(block, required_float_count)) {

        return 0;
    }
    layer->owned_block = block;
    layer->running_mean = block;
    layer->running_variance =
        layer->running_mean + layer->num_features;
    layer->batch_mean =
        layer->running_variance + layer->num_features;
    layer->batch_inv_std = layer->batch_mean + layer->num_features;
    layer->work_mean = layer->batch_inv_std + layer->num_features;
    layer->work_inv_std = layer->work_mean + layer->num_features;
    layer->work_unbiased_variance =
        layer->work_inv_std + layer->num_features;
    return 1;
}


static int test_expand_tensor_backing_capacity(
    VAECTensor *tensor,
    size_t multiplier,
    size_t *expanded_float_count
)
{
    float *expanded;
    size_t float_count;
    size_t byte_count;
    size_t expected_owned_float_count;
    size_t expected_owned_bytes;

    if (tensor == NULL || expanded_float_count == NULL ||
        !tensor->owns_memory || !tensor->has_grad ||
        tensor->owned_block == NULL || tensor->data != tensor->owned_block ||
        tensor->grad != tensor->data + tensor->shape.count ||
        !test_size_multiply(tensor->shape.count, 2,
                            &expected_owned_float_count) ||
        !test_size_multiply(expected_owned_float_count, sizeof(float),
                            &expected_owned_bytes) ||
        tensor->owned_bytes != expected_owned_bytes ||
        multiplier < 2 ||
        !test_size_multiply(tensor->shape.count, multiplier, &float_count) ||
        !test_size_multiply(float_count, sizeof(float), &byte_count)) {

        return 0;
    }
    expanded = (float *)calloc(float_count, sizeof(float));
    if (expanded == NULL) {
        return 0;
    }
    memcpy(expanded, tensor->owned_block, tensor->owned_bytes);
    free(tensor->owned_block);
    tensor->owned_block = expanded;
    tensor->data = expanded;
    tensor->grad = expanded + tensor->shape.count;
    *expanded_float_count = float_count;
    return byte_count >= tensor->owned_bytes;
}


static void model_snapshot_init(ModelSnapshot *snapshot)
{
    if (snapshot != NULL) {
        memset(snapshot, 0, sizeof(*snapshot));
    }
}


static void model_snapshot_free(ModelSnapshot *snapshot)
{
    size_t index;

    if (snapshot == NULL) {
        return;
    }
    for (index = 0; index < VAE_C_MODEL_MAX_PARAMETERS; index++) {
        byte_snapshot_free(&snapshot->parameters[index]);
    }
    for (index = 0; index < TEST_BATCH_NORM_COUNT; index++) {
        byte_snapshot_free(&snapshot->batch_norms[index]);
    }
    byte_snapshot_free(&snapshot->max_pool);
    model_snapshot_init(snapshot);
}


static int model_snapshot_capture(
    const VAECModel *model,
    ModelSnapshot *snapshot
)
{
    const VAECBatchNorm *batch_norms[TEST_BATCH_NORM_COUNT];
    ModelSnapshot candidate;
    size_t batch_norm_count;
    size_t index;

    if (model == NULL || snapshot == NULL || snapshot->captured ||
        model->parameter_count > VAE_C_MODEL_MAX_PARAMETERS) {

        return 0;
    }
    model_snapshot_init(&candidate);
    candidate.metadata = *model;
    candidate.parameter_count = model->parameter_count;
    for (index = 0; index < model->parameter_count; index++) {
        const VAECParameter *parameter = model->parameters[index];

        if (parameter == NULL || parameter->owned_block == NULL ||
            !byte_snapshot_capture(
                parameter->owned_block,
                parameter->owned_bytes,
                &candidate.parameters[index])) {

            model_snapshot_free(&candidate);
            return 0;
        }
    }
    batch_norm_count = collect_all_batch_norms(model, batch_norms);
    for (index = 0; index < batch_norm_count; index++) {
        if (batch_norms[index]->owned_block != NULL &&
            !byte_snapshot_capture(
                batch_norms[index]->owned_block,
                batch_norms[index]->owned_bytes,
                &candidate.batch_norms[index])) {

            model_snapshot_free(&candidate);
            return 0;
        }
    }
    if (model->encoder_max_pool.argmax_indices != NULL &&
        !byte_snapshot_capture(
            model->encoder_max_pool.argmax_indices,
            model->encoder_max_pool.owned_bytes,
            &candidate.max_pool)) {

        model_snapshot_free(&candidate);
        return 0;
    }
    candidate.captured = true;
    *snapshot = candidate;
    model_snapshot_init(&candidate);
    return 1;
}


static int model_snapshot_matches(
    const VAECModel *model,
    const ModelSnapshot *snapshot
)
{
    size_t index;

    if (model == NULL || snapshot == NULL || !snapshot->captured ||
        !model_metadata_exact(model, &snapshot->metadata)) {

        return 0;
    }
    for (index = 0; index < snapshot->parameter_count; index++) {
        const VAECParameter *parameter = model->parameters[index];

        if (parameter == NULL || !byte_snapshot_matches(
                parameter->owned_block,
                parameter->owned_bytes,
                &snapshot->parameters[index])) {

            return 0;
        }
    }
    for (index = 0; index < TEST_BATCH_NORM_COUNT; index++) {
        if (snapshot->batch_norms[index].copy != NULL &&
            !byte_snapshot_matches(
                snapshot->batch_norms[index].pointer,
                snapshot->batch_norms[index].byte_count,
                &snapshot->batch_norms[index])) {

            return 0;
        }
    }
    if (snapshot->max_pool.copy != NULL &&
        !byte_snapshot_matches(
            model->encoder_max_pool.argmax_indices,
            model->encoder_max_pool.owned_bytes,
            &snapshot->max_pool)) {

        return 0;
    }
    return 1;
}


static void workspace_snapshot_init(WorkspaceSnapshot *snapshot)
{
    if (snapshot != NULL) {
        memset(snapshot, 0, sizeof(*snapshot));
    }
}


static void workspace_snapshot_free(WorkspaceSnapshot *snapshot)
{
    size_t index;

    if (snapshot == NULL) {
        return;
    }
    for (index = 0;
         index < VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS;
         index++) {

        byte_snapshot_free(&snapshot->tensors[index]);
    }
    workspace_snapshot_init(snapshot);
}


static int workspace_snapshot_capture(
    const VAECModelWorkspace *workspace,
    WorkspaceSnapshot *snapshot
)
{
    const VAECTensor *tensors[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ];
    WorkspaceSnapshot candidate;
    size_t fixed_count;
    size_t index;

    if (workspace == NULL || snapshot == NULL || snapshot->captured) {
        return 0;
    }
    workspace_snapshot_init(&candidate);
    candidate.metadata = *workspace;
    fixed_count = collect_all_workspace_tensors(workspace, tensors);
    if (fixed_count != VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS) {
        return 0;
    }
    for (index = 0; index < fixed_count; index++) {
        if (tensors[index]->owned_block != NULL) {
            if (!byte_snapshot_capture(
                    tensors[index]->owned_block,
                    tensors[index]->owned_bytes,
                    &candidate.tensors[candidate.tensor_count])) {

                workspace_snapshot_free(&candidate);
                return 0;
            }
            candidate.tensor_count++;
        }
    }
    candidate.captured = true;
    *snapshot = candidate;
    workspace_snapshot_init(&candidate);
    return 1;
}


static int workspace_snapshot_matches(
    const VAECModelWorkspace *workspace,
    const WorkspaceSnapshot *snapshot
)
{
    size_t index;

    if (workspace == NULL || snapshot == NULL || !snapshot->captured ||
        !workspace_metadata_exact(workspace, &snapshot->metadata)) {

        return 0;
    }
    for (index = 0; index < snapshot->tensor_count; index++) {
        if (!byte_snapshot_matches(
                snapshot->tensors[index].pointer,
                snapshot->tensors[index].byte_count,
                &snapshot->tensors[index])) {

            return 0;
        }
    }
    return 1;
}


static int rejected_without_mutation(
    VAECModel *model,
    VAECBatchNormMode mode,
    VAECModelWorkspace *workspace
)
{
    ModelSnapshot model_before;
    WorkspaceSnapshot workspace_before;
    VAECOpStatus status;
    int passed;

    model_snapshot_init(&model_before);
    workspace_snapshot_init(&workspace_before);
    if (!model_snapshot_capture(model, &model_before) ||
        !workspace_snapshot_capture(workspace, &workspace_before)) {

        model_snapshot_free(&model_before);
        workspace_snapshot_free(&workspace_before);
        return 0;
    }
    status = vae_c_model_encoder_forward(model, mode, workspace);
    passed = status != VAE_C_OP_OK &&
        model_snapshot_matches(model, &model_before) &&
        workspace_snapshot_matches(workspace, &workspace_before);
    model_snapshot_free(&model_before);
    workspace_snapshot_free(&workspace_before);
    return passed;
}


static void fill_input(VAECModelWorkspace *workspace)
{
    size_t index;

    for (index = 0; index < workspace->input.shape.count; index++) {
        workspace->input.data[index] =
            (float)((index * 17u) % 251u) / 250.0f;
    }
}


static int tensor_data_equal(
    const VAECTensor *left,
    const VAECTensor *right
)
{
    return tensor_state_equal(left, right);
}


static int batch_norm_equal(
    const VAECBatchNorm *left,
    const VAECBatchNorm *right
)
{
    return batch_norm_state_equal(left, right);
}


static int parameter_blocks_equal(
    const VAECModel *left,
    const VAECModel *right
)
{
    size_t index;

    if (left->parameter_count != right->parameter_count) {
        return 0;
    }
    for (index = 0; index < left->parameter_count; index++) {
        const VAECParameter *a = left->parameters[index];
        const VAECParameter *b = right->parameters[index];

        if (!parameter_state_equal(a, b)) {

            return 0;
        }
    }
    return 1;
}


static int encoder_results_equal(
    const VAECModel *left_model,
    const VAECModelWorkspace *left_workspace,
    const VAECModel *right_model,
    const VAECModelWorkspace *right_workspace
)
{
    size_t depth = left_model->config.conv_depth;
    size_t index;

    if (depth != right_model->config.conv_depth ||
        !tensor_data_equal(&left_workspace->input,
                           &right_workspace->input) ||
        !tensor_data_equal(&left_workspace->encoder_pool_output,
                           &right_workspace->encoder_pool_output) ||
        !tensor_data_equal(
            &left_workspace->encoder_global_avg_output,
            &right_workspace->encoder_global_avg_output) ||
        !tensor_data_equal(&left_workspace->encoder_packed_output,
                           &right_workspace->encoder_packed_output) ||
        !tensor_data_equal(&left_workspace->mu,
                           &right_workspace->mu) ||
        !tensor_data_equal(&left_workspace->logvar,
                           &right_workspace->logvar) ||
        !max_pool_state_equal(&left_model->encoder_max_pool,
                              &right_model->encoder_max_pool)) {

        return 0;
    }
    for (index = 0; index < depth; index++) {
        if (!tensor_data_equal(
                &left_workspace->encoder_conv_outputs[index],
                &right_workspace->encoder_conv_outputs[index]) ||
            !tensor_data_equal(
                &left_workspace->encoder_batch_norm_outputs[index],
                &right_workspace->encoder_batch_norm_outputs[index]) ||
            !tensor_data_equal(
                &left_workspace->encoder_relu_outputs[index],
                &right_workspace->encoder_relu_outputs[index]) ||
            !batch_norm_equal(
                &left_model->encoder_batch_norms[index],
                &right_model->encoder_batch_norms[index])) {

            return 0;
        }
    }
    return 1;
}


static VAECOpStatus manual_encoder_forward(
    VAECModel *model,
    VAECBatchNormMode mode,
    VAECModelWorkspace *workspace
)
{
    VAECOpStatus status;
    const VAECTensor *input;
    size_t index;

    status = vae_c_conv2d_forward(
        &model->encoder_convs[0],
        &workspace->input,
        &workspace->encoder_conv_outputs[0]
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_batch_norm2d_forward(
        &model->encoder_batch_norms[0], mode,
        &workspace->encoder_conv_outputs[0],
        &workspace->encoder_batch_norm_outputs[0]
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_relu_forward(
        &workspace->encoder_batch_norm_outputs[0],
        &workspace->encoder_relu_outputs[0]
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_max_pool2d_forward(
        &model->encoder_max_pool,
        &workspace->encoder_relu_outputs[0],
        &workspace->encoder_pool_output
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    input = &workspace->encoder_pool_output;
    for (index = 1; index < model->config.conv_depth; index++) {
        status = vae_c_conv2d_forward(
            &model->encoder_convs[index], input,
            &workspace->encoder_conv_outputs[index]
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
        status = vae_c_batch_norm2d_forward(
            &model->encoder_batch_norms[index], mode,
            &workspace->encoder_conv_outputs[index],
            &workspace->encoder_batch_norm_outputs[index]
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
        status = vae_c_relu_forward(
            &workspace->encoder_batch_norm_outputs[index],
            &workspace->encoder_relu_outputs[index]
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
        input = &workspace->encoder_relu_outputs[index];
    }
    status = vae_c_global_avg_pool2d_forward(
        input, &workspace->encoder_global_avg_output
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_dense_forward(
        &model->encoder_fc,
        &workspace->encoder_global_avg_output,
        &workspace->encoder_packed_output
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    return vae_c_latent_split_forward(
        &workspace->encoder_packed_output,
        &workspace->mu,
        &workspace->logvar
    );
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


static int allocate_initialized_pair(
    const VAECProfile *profile,
    size_t batch_size,
    VAECModel *first_model,
    VAECModelWorkspace *first_workspace,
    VAECModel *second_model,
    VAECModelWorkspace *second_workspace
)
{
    const uint64_t seed = UINT64_C(123456789);

    vae_c_model_init(first_model);
    vae_c_model_init(second_model);
    vae_c_model_workspace_init(first_workspace);
    vae_c_model_workspace_init(second_workspace);
    if (profile == NULL ||
        vae_c_model_allocate(first_model, profile, batch_size) !=
            VAE_C_OP_OK ||
        vae_c_model_allocate(second_model, profile, batch_size) !=
            VAE_C_OP_OK ||
        vae_c_model_initialize_parameters_v1(first_model, seed) !=
            VAE_C_OP_OK ||
        vae_c_model_initialize_parameters_v1(second_model, seed) !=
            VAE_C_OP_OK ||
        vae_c_model_workspace_allocate(
            first_model, batch_size, first_workspace) != VAE_C_OP_OK ||
        vae_c_model_workspace_allocate(
            second_model, batch_size, second_workspace) != VAE_C_OP_OK) {

        return 0;
    }
    fill_input(first_workspace);
    fill_input(second_workspace);
    return model_state_equal(first_model, second_model) &&
        workspace_state_equal(first_workspace, second_workspace);
}


static void free_pair(
    VAECModel *first_model,
    VAECModelWorkspace *first_workspace,
    VAECModel *second_model,
    VAECModelWorkspace *second_workspace
)
{
    vae_c_model_workspace_free(first_workspace);
    vae_c_model_workspace_free(second_workspace);
    vae_c_model_free(first_model);
    vae_c_model_free(second_model);
}


static int test_mini75_train_manual_equivalence(
    const VAECProfileCatalog *catalog
)
{
    const VAECProfile *profile = vae_c_profile_catalog_find(
        catalog, "mini75_smoke_v1"
    );
    VAECModel manual_model;
    VAECModel composite_model;
    VAECModelWorkspace manual_workspace;
    VAECModelWorkspace composite_workspace;
    int passed;

    if (!allocate_initialized_pair(
            profile, 2,
            &manual_model, &manual_workspace,
            &composite_model, &composite_workspace)) {

        free_pair(&manual_model, &manual_workspace,
                  &composite_model, &composite_workspace);
        return 0;
    }
    passed =
        manual_encoder_forward(
            &manual_model,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &manual_workspace) == VAE_C_OP_OK &&
        vae_c_model_encoder_forward(
            &composite_model,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &composite_workspace) == VAE_C_OP_OK &&
        encoder_results_equal(
            &manual_model, &manual_workspace,
            &composite_model, &composite_workspace) &&
        parameter_blocks_equal(&manual_model, &composite_model);
    free_pair(&manual_model, &manual_workspace,
              &composite_model, &composite_workspace);
    return passed;
}


static int test_mini75_eval_manual_equivalence(
    const VAECProfileCatalog *catalog
)
{
    const VAECProfile *profile = vae_c_profile_catalog_find(
        catalog, "mini75_smoke_v1"
    );
    VAECModel manual_model;
    VAECModel composite_model;
    VAECModelWorkspace manual_workspace;
    VAECModelWorkspace composite_workspace;
    ByteSnapshot manual_running[VAE_C_MAX_CONV_DEPTH];
    ByteSnapshot composite_running[VAE_C_MAX_CONV_DEPTH];
    size_t index;
    int passed = 1;

    memset(manual_running, 0, sizeof(manual_running));
    memset(composite_running, 0, sizeof(composite_running));
    if (!allocate_initialized_pair(
            profile, 2,
            &manual_model, &manual_workspace,
            &composite_model, &composite_workspace)) {

        free_pair(&manual_model, &manual_workspace,
                  &composite_model, &composite_workspace);
        return 0;
    }
    passed = passed &&
        manual_encoder_forward(
            &manual_model,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &manual_workspace) == VAE_C_OP_OK &&
        manual_encoder_forward(
            &composite_model,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &composite_workspace) == VAE_C_OP_OK;
    for (index = 0; passed && index < profile->conv_depth; index++) {
        size_t bytes = 2 * profile->channels[index] * sizeof(float);

        passed = byte_snapshot_capture(
            manual_model.encoder_batch_norms[index].running_mean,
            bytes,
            &manual_running[index]
        ) && byte_snapshot_capture(
            composite_model.encoder_batch_norms[index].running_mean,
            bytes,
            &composite_running[index]
        );
    }
    fill_input(&manual_workspace);
    fill_input(&composite_workspace);
    passed = passed &&
        manual_encoder_forward(
            &manual_model,
            VAE_C_BATCH_NORM_MODE_EVAL,
            &manual_workspace) == VAE_C_OP_OK &&
        vae_c_model_encoder_forward(
            &composite_model,
            VAE_C_BATCH_NORM_MODE_EVAL,
            &composite_workspace) == VAE_C_OP_OK &&
        encoder_results_equal(
            &manual_model, &manual_workspace,
            &composite_model, &composite_workspace);
    for (index = 0; index < profile->conv_depth; index++) {
        size_t bytes = 2 * profile->channels[index] * sizeof(float);

        passed = passed && byte_snapshot_matches(
            manual_model.encoder_batch_norms[index].running_mean,
            bytes,
            &manual_running[index]
        ) && byte_snapshot_matches(
            composite_model.encoder_batch_norms[index].running_mean,
            bytes,
            &composite_running[index]
        ) && !manual_model.encoder_batch_norms[index].cache_valid &&
            !composite_model.encoder_batch_norms[index].cache_valid;
        byte_snapshot_free(&manual_running[index]);
        byte_snapshot_free(&composite_running[index]);
    }
    free_pair(&manual_model, &manual_workspace,
              &composite_model, &composite_workspace);
    return passed;
}


static int test_synthetic_depth4_manual_equivalence(void)
{
    VAECProfile profile;
    VAECModel manual_model;
    VAECModel composite_model;
    VAECModelWorkspace manual_workspace;
    VAECModelWorkspace composite_workspace;
    int passed;

    make_synthetic_depth4(&profile);
    if (!allocate_initialized_pair(
            &profile, 2,
            &manual_model, &manual_workspace,
            &composite_model, &composite_workspace)) {

        free_pair(&manual_model, &manual_workspace,
                  &composite_model, &composite_workspace);
        return 0;
    }
    passed = manual_workspace.encoder_packed_output.shape.dims[0] == 2 &&
        manual_workspace.encoder_packed_output.shape.dims[1] == 6 &&
        manual_workspace.mu.shape.dims[0] == 2 &&
        manual_workspace.mu.shape.dims[1] == 3 &&
        manual_workspace.logvar.shape.dims[0] == 2 &&
        manual_workspace.logvar.shape.dims[1] == 3 &&
        manual_encoder_forward(
            &manual_model,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &manual_workspace) == VAE_C_OP_OK &&
        vae_c_model_encoder_forward(
            &composite_model,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &composite_workspace) == VAE_C_OP_OK &&
        encoder_results_equal(
            &manual_model, &manual_workspace,
            &composite_model, &composite_workspace);
    free_pair(&manual_model, &manual_workspace,
              &composite_model, &composite_workspace);
    return passed;
}


static int test_synthetic_depth4_batch_modes(void)
{
    VAECProfile profile;
    VAECModel model;
    VAECModelWorkspace workspace;
    size_t index;
    int passed;

    make_synthetic_depth4(&profile);
    vae_c_model_init(&model);
    vae_c_model_workspace_init(&workspace);
    if (vae_c_model_allocate(&model, &profile, 2) != VAE_C_OP_OK ||
        vae_c_model_initialize_parameters_v1(
            &model, UINT64_C(321)) != VAE_C_OP_OK ||
        vae_c_model_workspace_allocate(&model, 1, &workspace) !=
            VAE_C_OP_OK) {

        vae_c_model_workspace_free(&workspace);
        vae_c_model_free(&model);
        return 0;
    }
    fill_input(&workspace);
    passed = rejected_without_mutation(
        &model, VAE_C_BATCH_NORM_MODE_TRAIN, &workspace
    );
    passed = passed && vae_c_model_encoder_forward(
        &model, VAE_C_BATCH_NORM_MODE_EVAL, &workspace
    ) == VAE_C_OP_OK;
    for (index = 0; passed && index < workspace.mu.shape.count; index++) {
        passed = isfinite(workspace.mu.data[index]) &&
            isfinite(workspace.logvar.data[index]);
    }
    vae_c_model_workspace_free(&workspace);
    vae_c_model_free(&model);
    return passed;
}


static size_t collect_non_encoder_tensors(
    const VAECModelWorkspace *workspace,
    const VAECTensor *tensors[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ]
);


static int test_official_profiles_eval(
    const VAECProfileCatalog *catalog
)
{
    const char *ids[4] = {
        "mini75_smoke_v1",
        "mini75_experiment_v1",
        "mini112_primary_v1",
        "mini112_scaleup_v1"
    };
    size_t profile_index;

    for (profile_index = 0; profile_index < 4; profile_index++) {
        const VAECProfile *profile = vae_c_profile_catalog_find(
            catalog, ids[profile_index]
        );
        size_t batch_size;

        for (batch_size = 1; batch_size <= 2; batch_size++) {
            VAECModel model;
            VAECModelWorkspace workspace;
            const VAECTensor *non_encoder[
                VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
            ];
            ByteSnapshot non_encoder_before[
                VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
            ];
            VAECTensor reshape_before;
            size_t non_encoder_count = 0;
            size_t value;
            size_t index;
            int passed;

            memset(non_encoder_before, 0, sizeof(non_encoder_before));
            vae_c_model_init(&model);
            vae_c_model_workspace_init(&workspace);
            passed = profile != NULL &&
                vae_c_model_allocate(&model, profile, 2) == VAE_C_OP_OK &&
                vae_c_model_initialize_parameters_v1(
                    &model,
                    UINT64_C(9000) + (uint64_t)profile_index
                ) == VAE_C_OP_OK &&
                vae_c_model_workspace_allocate(
                    &model, batch_size, &workspace
                ) == VAE_C_OP_OK;
            if (passed) {
                fill_input(&workspace);
                non_encoder_count = collect_non_encoder_tensors(
                    &workspace, non_encoder
                );
                for (index = 0; index < non_encoder_count; index++) {
                    for (value = 0;
                         value < non_encoder[index]->shape.count;
                         value++) {

                        non_encoder[index]->data[value] =
                            0.1f + 0.01f * (float)index +
                            0.0001f * (float)(value % 37);
                    }
                    passed = passed && byte_snapshot_capture(
                        non_encoder[index]->owned_block,
                        non_encoder[index]->owned_bytes,
                        &non_encoder_before[index]
                    );
                }
                reshape_before = workspace.decoder_reshape_view;
                passed = vae_c_model_encoder_forward(
                    &model,
                    VAE_C_BATCH_NORM_MODE_EVAL,
                    &workspace
                ) == VAE_C_OP_OK && passed;
            }
            for (value = 0;
                 passed && value < workspace.mu.shape.count;
                 value++) {

                passed = isfinite(workspace.mu.data[value]) &&
                    isfinite(workspace.logvar.data[value]);
            }
            for (index = 0; index < non_encoder_count; index++) {
                passed = passed && byte_snapshot_matches(
                    non_encoder[index]->owned_block,
                    non_encoder[index]->owned_bytes,
                    &non_encoder_before[index]
                );
                byte_snapshot_free(&non_encoder_before[index]);
            }
            passed = passed && tensor_metadata_exact(
                &reshape_before,
                &workspace.decoder_reshape_view
            );
            vae_c_model_workspace_free(&workspace);
            vae_c_model_free(&model);
            if (!passed) {
                return 0;
            }
        }
    }
    return 1;
}


static int test_eval_determinism(
    const VAECProfileCatalog *catalog
)
{
    const VAECProfile *profile = vae_c_profile_catalog_find(
        catalog, "mini75_smoke_v1"
    );
    VAECModel first_model;
    VAECModel second_model;
    VAECModelWorkspace first_workspace;
    VAECModelWorkspace second_workspace;
    int passed;

    if (!allocate_initialized_pair(
            profile, 2,
            &first_model, &first_workspace,
            &second_model, &second_workspace)) {

        free_pair(&first_model, &first_workspace,
                  &second_model, &second_workspace);
        return 0;
    }
    passed = vae_c_model_encoder_forward(
        &first_model,
        VAE_C_BATCH_NORM_MODE_EVAL,
        &first_workspace
    ) == VAE_C_OP_OK && vae_c_model_encoder_forward(
        &second_model,
        VAE_C_BATCH_NORM_MODE_EVAL,
        &second_workspace
    ) == VAE_C_OP_OK && encoder_results_equal(
        &first_model, &first_workspace,
        &second_model, &second_workspace
    ) && parameter_blocks_equal(&first_model, &second_model);
    free_pair(&first_model, &first_workspace,
              &second_model, &second_workspace);
    return passed;
}


static int make_initialized_fixture_from_profile(
    const VAECProfile *profile,
    VAECModel *model,
    VAECModelWorkspace *workspace
)
{
    vae_c_model_init(model);
    vae_c_model_workspace_init(workspace);
    if (profile == NULL ||
        vae_c_model_allocate(model, profile, 2) != VAE_C_OP_OK ||
        vae_c_model_initialize_parameters_v1(
            model, UINT64_C(777)) != VAE_C_OP_OK ||
        vae_c_model_workspace_allocate(model, 2, workspace) !=
            VAE_C_OP_OK) {

        return 0;
    }
    fill_input(workspace);
    return 1;
}


static int make_initialized_fixture(
    const VAECProfileCatalog *catalog,
    VAECModel *model,
    VAECModelWorkspace *workspace
)
{
    return make_initialized_fixture_from_profile(
        vae_c_profile_catalog_find(catalog, "mini75_smoke_v1"),
        model,
        workspace
    );
}


static int test_invalid_arguments_and_empty_state(
    const VAECProfileCatalog *catalog
)
{
    VAECModel model;
    VAECModel empty_model;
    VAECModelWorkspace workspace;
    VAECModelWorkspace empty_workspace;
    ModelSnapshot model_before;
    WorkspaceSnapshot workspace_before;
    int passed = 1;

    vae_c_model_init(&empty_model);
    vae_c_model_workspace_init(&empty_workspace);
    model_snapshot_init(&model_before);
    workspace_snapshot_init(&workspace_before);
    if (!make_initialized_fixture(catalog, &model, &workspace)) {
        vae_c_model_workspace_free(&workspace);
        vae_c_model_free(&model);
        return 0;
    }

    passed = workspace_snapshot_capture(&workspace, &workspace_before) &&
        vae_c_model_encoder_forward(
            NULL, VAE_C_BATCH_NORM_MODE_EVAL, &workspace
        ) == VAE_C_OP_INVALID_ARGUMENT &&
        workspace_snapshot_matches(&workspace, &workspace_before);
    workspace_snapshot_free(&workspace_before);

    passed = passed && model_snapshot_capture(&model, &model_before) &&
        vae_c_model_encoder_forward(
            &model, VAE_C_BATCH_NORM_MODE_EVAL, NULL
        ) == VAE_C_OP_INVALID_ARGUMENT &&
        model_snapshot_matches(&model, &model_before);
    model_snapshot_free(&model_before);

    passed = passed && rejected_without_mutation(
        &model, (VAECBatchNormMode)999, &workspace
    ) && rejected_without_mutation(
        &empty_model, VAE_C_BATCH_NORM_MODE_EVAL, &empty_workspace
    ) && rejected_without_mutation(
        &empty_model, VAE_C_BATCH_NORM_MODE_EVAL, &workspace
    ) && rejected_without_mutation(
        &model, VAE_C_BATCH_NORM_MODE_EVAL, &empty_workspace
    );

    vae_c_model_workspace_free(&workspace);
    vae_c_model_free(&model);
    return passed;
}


static int test_uninitialized_model_rejected(
    const VAECProfileCatalog *catalog
)
{
    const VAECProfile *profile = vae_c_profile_catalog_find(
        catalog, "mini75_smoke_v1"
    );
    VAECModel model;
    VAECModelWorkspace workspace;
    int passed;

    vae_c_model_init(&model);
    vae_c_model_workspace_init(&workspace);
    if (profile == NULL ||
        vae_c_model_allocate(&model, profile, 2) != VAE_C_OP_OK ||
        vae_c_model_workspace_allocate(&model, 2, &workspace) !=
            VAE_C_OP_OK) {

        vae_c_model_workspace_free(&workspace);
        vae_c_model_free(&model);
        return 0;
    }
    fill_input(&workspace);
    passed = rejected_without_mutation(
        &model, VAE_C_BATCH_NORM_MODE_EVAL, &workspace
    );
    vae_c_model_workspace_free(&workspace);
    vae_c_model_free(&model);
    return passed;
}


static int test_preflight_input_domain(
    const VAECProfileCatalog *catalog
)
{
    const float invalid_values[5] = {
        NAN, INFINITY, -INFINITY, -0.001f, 1.001f
    };
    VAECModel model;
    VAECModelWorkspace workspace;
    size_t index;
    int passed = 1;

    if (!make_initialized_fixture(catalog, &model, &workspace)) {
        vae_c_model_workspace_free(&workspace);
        vae_c_model_free(&model);
        return 0;
    }
    for (index = 0; index < 5; index++) {
        float saved = workspace.input.data[index];

        workspace.input.data[index] = invalid_values[index];
        passed = passed && rejected_without_mutation(
            &model, VAE_C_BATCH_NORM_MODE_EVAL, &workspace
        );
        workspace.input.data[index] = saved;
    }
    vae_c_model_workspace_free(&workspace);
    vae_c_model_free(&model);
    return passed;
}


static int test_preflight_parameter_values(
    const VAECProfileCatalog *catalog
)
{
    const float invalid_values[3] = {NAN, INFINITY, -INFINITY};
    VAECModel model;
    VAECModelWorkspace workspace;
    VAECParameter *parameter;
    size_t index;
    int passed = 1;

    if (!make_initialized_fixture(catalog, &model, &workspace)) {
        vae_c_model_workspace_free(&workspace);
        vae_c_model_free(&model);
        return 0;
    }
    parameter = model.parameters[0];
    for (index = 0; index < 3; index++) {
        float saved = parameter->values[index];

        parameter->values[index] = invalid_values[index];
        passed = passed && rejected_without_mutation(
            &model, VAE_C_BATCH_NORM_MODE_EVAL, &workspace
        );
        parameter->values[index] = saved;
    }
    vae_c_model_workspace_free(&workspace);
    vae_c_model_free(&model);
    return passed;
}


static int test_preflight_structural_corruptions(
    const VAECProfileCatalog *catalog
)
{
    VAECModel model;
    VAECModelWorkspace workspace;
    char saved_char;
    size_t saved_size;
    float saved_float;
    bool saved_bool;
    int passed = 1;

    if (!make_initialized_fixture(catalog, &model, &workspace)) {
        vae_c_model_workspace_free(&workspace);
        vae_c_model_free(&model);
        return 0;
    }

    saved_bool = model.allocated;
    model.allocated = false;
    passed = passed && rejected_without_mutation(
        &model, VAE_C_BATCH_NORM_MODE_EVAL, &workspace
    );
    model.allocated = saved_bool;

    saved_bool = workspace.allocated;
    workspace.allocated = false;
    passed = passed && rejected_without_mutation(
        &model, VAE_C_BATCH_NORM_MODE_EVAL, &workspace
    );
    workspace.allocated = saved_bool;

    saved_char = workspace.profile_id[0];
    workspace.profile_id[0] = 'X';
    passed = passed && rejected_without_mutation(
        &model, VAE_C_BATCH_NORM_MODE_EVAL, &workspace
    );
    workspace.profile_id[0] = saved_char;

    saved_size = workspace.config.input_side;
    workspace.config.input_side++;
    passed = passed && rejected_without_mutation(
        &model, VAE_C_BATCH_NORM_MODE_EVAL, &workspace
    );
    workspace.config.input_side = saved_size;

    saved_size = model.config.conv_depth;
    model.config.conv_depth = 2;
    passed = passed && rejected_without_mutation(
        &model, VAE_C_BATCH_NORM_MODE_EVAL, &workspace
    );
    model.config.conv_depth = saved_size;

    saved_size = workspace.batch_size;
    workspace.batch_size = 0;
    passed = passed && rejected_without_mutation(
        &model, VAE_C_BATCH_NORM_MODE_EVAL, &workspace
    );
    workspace.batch_size = saved_size;

    saved_float = model.encoder_batch_norms[0].running_mean[0];
    model.encoder_batch_norms[0].running_mean[0] = NAN;
    passed = passed && rejected_without_mutation(
        &model, VAE_C_BATCH_NORM_MODE_EVAL, &workspace
    );
    model.encoder_batch_norms[0].running_mean[0] = saved_float;

    saved_size = model.encoder_max_pool.capacity;
    model.encoder_max_pool.capacity = 0;
    passed = passed && rejected_without_mutation(
        &model, VAE_C_BATCH_NORM_MODE_EVAL, &workspace
    );
    model.encoder_max_pool.capacity = saved_size;

    passed = passed && vae_c_model_validate(&model) == VAE_C_OP_OK &&
        vae_c_model_workspace_validate(&model, &workspace) == VAE_C_OP_OK;
    vae_c_model_workspace_free(&workspace);
    vae_c_model_free(&model);
    return passed;
}


static int test_preflight_exact_alias(
    const VAECProfileCatalog *catalog
)
{
    typedef enum {
        ALIAS_WORKSPACE_DATA_PARAMETER_VALUES = 0,
        ALIAS_WORKSPACE_DATA_PARAMETER_GRADIENTS_ONLY,
        ALIAS_WORKSPACE_GRAD_MODEL_DECODER_BN_REGION_ONLY,
        ALIAS_WORKSPACE_DATA_ADAM_M,
        ALIAS_WORKSPACE_DATA_ADAM_V,
        ALIAS_WORKSPACE_DATA_BATCH_NORM_STATE,
        ALIAS_ENCODER_DATA_NON_ENCODER_DATA,
        ALIAS_MAX_POOL_CACHE_WORKSPACE_DATA,
        ALIAS_CASE_COUNT
    } AliasCase;
    const VAECProfile *profile;
    VAECModel model;
    VAECModelWorkspace workspace;
    VAECParameter *parameter;
    VAECBatchNorm *decoder_batch_norm;
    VAECTensor *decoder_workspace_tensor;
    size_t expanded_float_count = 0;
    size_t expanded_byte_count;
    size_t decoder_batch_norm_float_count;
    size_t alias_case;
    int passed = 1;

    vae_c_model_init(&model);
    vae_c_model_workspace_init(&workspace);
    profile = vae_c_profile_catalog_find(catalog, "mini75_smoke_v1");
    if (profile == NULL ||
        vae_c_model_allocate(&model, profile, 1) != VAE_C_OP_OK ||
        vae_c_model_initialize_parameters_v1(
            &model, UINT64_C(777)) != VAE_C_OP_OK ||
        vae_c_model_workspace_allocate(&model, 1, &workspace) !=
            VAE_C_OP_OK) {

        vae_c_model_workspace_free(&workspace);
        vae_c_model_free(&model);
        return 0;
    }
    fill_input(&workspace);
    parameter = &model.encoder_batch_norms[2].gamma;
    decoder_batch_norm = &model.decoder_fc_batch_norm;
    decoder_workspace_tensor = &workspace.decoder_fc_output;
    if (parameter->shape.count != workspace.epsilon.shape.count ||
        workspace.epsilon.grad != NULL ||
        model.encoder_batch_norms[2].owned_bytes <
            workspace.epsilon.owned_bytes ||
        workspace.input.owned_bytes < model.encoder_max_pool.owned_bytes ||
        parameter->owned_bytes / 4 != workspace.epsilon.owned_bytes ||
        !test_float_values_are_finite(
            parameter->gradients, parameter->shape.count) ||
        decoder_batch_norm->num_features !=
            decoder_workspace_tensor->shape.count ||
        !test_size_multiply(
            decoder_batch_norm->num_features, 7,
            &decoder_batch_norm_float_count) ||
        !test_expand_tensor_backing_capacity(
            decoder_workspace_tensor, 8, &expanded_float_count) ||
        !test_size_multiply(
            expanded_float_count, sizeof(float), &expanded_byte_count) ||
        expanded_float_count < decoder_workspace_tensor->shape.count ||
        expanded_float_count - decoder_workspace_tensor->shape.count !=
            decoder_batch_norm_float_count ||
        !test_float_values_are_finite(
            decoder_workspace_tensor->grad,
            decoder_batch_norm_float_count) ||
        vae_c_model_validate(&model) != VAE_C_OP_OK ||
        vae_c_model_workspace_validate(&model, &workspace) != VAE_C_OP_OK) {

        vae_c_model_workspace_free(&workspace);
        vae_c_model_free(&model);
        return 0;
    }

    for (alias_case = 0; alias_case < (size_t)ALIAS_CASE_COUNT;
         alias_case++) {

        ModelSnapshot model_before;
        WorkspaceSnapshot workspace_before;
        ByteSnapshot expanded_backing_before;
        float *saved_data = NULL;
        float *saved_owned_block = NULL;
        size_t *saved_argmax = NULL;
        VAECBatchNorm *rebound_batch_norm = NULL;
        TestBatchNormBufferPointers saved_batch_norm_pointers;
        TestExactAliasSummary alias_summary;
        TestModelRegionKind expected_model_kind =
            TEST_MODEL_REGION_PARAMETER_VALUES;
        TestWorkspaceRegionKind expected_workspace_kind =
            TEST_WORKSPACE_REGION_DATA;
        const void *expected_model_owner = NULL;
        const VAECTensor *expected_workspace_tensor = NULL;
        int expect_cross_domain_alias = 1;
        int alias_isolated = 0;
        VAECOpStatus expected_status = VAE_C_OP_INVALID_ARGUMENT;
        VAECOpStatus status;

        memset(&saved_batch_norm_pointers, 0,
               sizeof(saved_batch_norm_pointers));
        memset(&alias_summary, 0, sizeof(alias_summary));
        model_snapshot_init(&model_before);
        workspace_snapshot_init(&workspace_before);
        byte_snapshot_init(&expanded_backing_before);
        if (!model_snapshot_capture(&model, &model_before) ||
            !workspace_snapshot_capture(&workspace, &workspace_before) ||
            !byte_snapshot_capture(
                decoder_workspace_tensor->owned_block,
                expanded_byte_count,
                &expanded_backing_before)) {

            model_snapshot_free(&model_before);
            workspace_snapshot_free(&workspace_before);
            byte_snapshot_free(&expanded_backing_before);
            passed = 0;
            break;
        }

        switch ((AliasCase)alias_case) {
        case ALIAS_WORKSPACE_DATA_PARAMETER_VALUES:
            saved_data = workspace.epsilon.data;
            saved_owned_block = workspace.epsilon.owned_block;
            workspace.epsilon.data = parameter->values;
            workspace.epsilon.owned_block = parameter->values;
            expected_model_kind = TEST_MODEL_REGION_PARAMETER_VALUES;
            expected_model_owner = parameter;
            expected_workspace_tensor = &workspace.epsilon;
            break;
        case ALIAS_WORKSPACE_DATA_PARAMETER_GRADIENTS_ONLY:
            saved_data = workspace.epsilon.data;
            saved_owned_block = workspace.epsilon.owned_block;
            workspace.epsilon.data = parameter->gradients;
            workspace.epsilon.owned_block = parameter->gradients;
            expected_model_kind = TEST_MODEL_REGION_PARAMETER_GRADIENTS;
            expected_model_owner = parameter;
            expected_workspace_tensor = &workspace.epsilon;
            break;
        case ALIAS_WORKSPACE_GRAD_MODEL_DECODER_BN_REGION_ONLY:
            /*
             * Batch size one makes decoder_fc_output.grad and each
             * decoder FC BatchNorm region equally long.  The test-only
             * backing has six additional, zero-filled regions after grad,
             * so the seven-region BN layout remains valid while only
             * running_mean exactly coincides with the logical grad region.
             */
            rebound_batch_norm = decoder_batch_norm;
            test_batch_norm_buffer_pointers_capture(
                rebound_batch_norm, &saved_batch_norm_pointers
            );
            passed = passed && test_batch_norm_buffers_rebind(
                rebound_batch_norm,
                decoder_workspace_tensor->grad,
                expanded_float_count -
                    decoder_workspace_tensor->shape.count
            );
            expected_model_kind =
                TEST_MODEL_REGION_BATCH_NORM_RUNNING_MEAN;
            expected_model_owner = rebound_batch_norm;
            expected_workspace_kind = TEST_WORKSPACE_REGION_GRAD;
            expected_workspace_tensor = decoder_workspace_tensor;
            break;
        case ALIAS_WORKSPACE_DATA_ADAM_M:
            saved_data = workspace.epsilon.data;
            saved_owned_block = workspace.epsilon.owned_block;
            workspace.epsilon.data = parameter->adam_m;
            workspace.epsilon.owned_block = parameter->adam_m;
            expected_model_kind = TEST_MODEL_REGION_PARAMETER_ADAM_M;
            expected_model_owner = parameter;
            expected_workspace_tensor = &workspace.epsilon;
            break;
        case ALIAS_WORKSPACE_DATA_ADAM_V:
            saved_data = workspace.epsilon.data;
            saved_owned_block = workspace.epsilon.owned_block;
            workspace.epsilon.data = parameter->adam_v;
            workspace.epsilon.owned_block = parameter->adam_v;
            expected_model_kind = TEST_MODEL_REGION_PARAMETER_ADAM_V;
            expected_model_owner = parameter;
            expected_workspace_tensor = &workspace.epsilon;
            break;
        case ALIAS_WORKSPACE_DATA_BATCH_NORM_STATE:
            saved_data = workspace.epsilon.data;
            saved_owned_block = workspace.epsilon.owned_block;
            workspace.epsilon.data =
                model.encoder_batch_norms[2].running_mean;
            workspace.epsilon.owned_block = workspace.epsilon.data;
            expected_model_kind =
                TEST_MODEL_REGION_BATCH_NORM_RUNNING_MEAN;
            expected_model_owner = &model.encoder_batch_norms[2];
            expected_workspace_tensor = &workspace.epsilon;
            break;
        case ALIAS_ENCODER_DATA_NON_ENCODER_DATA:
            saved_data = workspace.epsilon.data;
            saved_owned_block = workspace.epsilon.owned_block;
            workspace.epsilon.data = workspace.mu.data;
            workspace.epsilon.owned_block = workspace.mu.data;
            expected_status = VAE_C_OP_INVALID_STATE;
            expect_cross_domain_alias = 0;
            break;
        case ALIAS_MAX_POOL_CACHE_WORKSPACE_DATA:
            saved_argmax = model.encoder_max_pool.argmax_indices;
            model.encoder_max_pool.argmax_indices =
                (size_t *)(void *)workspace.input.data;
            expected_model_kind = TEST_MODEL_REGION_MAX_POOL_ARGMAX;
            expected_model_owner = &model.encoder_max_pool;
            expected_workspace_tensor = &workspace.input;
            break;
        default:
            passed = 0;
            break;
        }

        if (passed && test_summarize_exact_cross_domain_aliases(
                &model, &workspace, &alias_summary)) {

            alias_isolated = expect_cross_domain_alias
                ? test_exact_alias_is(
                    &alias_summary,
                    expected_model_kind,
                    expected_model_owner,
                    expected_workspace_kind,
                    expected_workspace_tensor
                )
                : alias_summary.count == 0 &&
                    workspace.epsilon.data == workspace.mu.data;
        }
        status = passed && alias_isolated
            ? vae_c_model_encoder_forward(
                &model, VAE_C_BATCH_NORM_MODE_EVAL, &workspace)
            : VAE_C_OP_OK;

        if ((AliasCase)alias_case ==
            ALIAS_MAX_POOL_CACHE_WORKSPACE_DATA) {

            model.encoder_max_pool.argmax_indices = saved_argmax;
        } else if ((AliasCase)alias_case ==
                   ALIAS_WORKSPACE_GRAD_MODEL_DECODER_BN_REGION_ONLY) {

            test_batch_norm_buffer_pointers_restore(
                rebound_batch_norm, &saved_batch_norm_pointers
            );
        } else {
            workspace.epsilon.data = saved_data;
            workspace.epsilon.owned_block = saved_owned_block;
        }

        passed = passed && alias_isolated && status == expected_status &&
            model_snapshot_matches(&model, &model_before) &&
            workspace_snapshot_matches(&workspace, &workspace_before) &&
            byte_snapshot_matches(
                decoder_workspace_tensor->owned_block,
                expanded_byte_count,
                &expanded_backing_before) &&
            vae_c_model_validate(&model) == VAE_C_OP_OK &&
            vae_c_model_workspace_validate(&model, &workspace) ==
                VAE_C_OP_OK;
        if (passed && (AliasCase)alias_case ==
                ALIAS_WORKSPACE_DATA_PARAMETER_GRADIENTS_ONLY) {

            printf(
                "[PASS] alias_workspace_data_parameter_gradients_only "
                "alias_count=1 model=PARAMETER_GRADIENTS "
                "workspace=WORKSPACE_DATA "
                "status=VAE_C_OP_INVALID_ARGUMENT\n"
            );
        } else if (passed && (AliasCase)alias_case ==
                   ALIAS_WORKSPACE_GRAD_MODEL_DECODER_BN_REGION_ONLY) {

            printf(
                "[PASS] alias_workspace_grad_model_bn_region_only "
                "alias_count=1 model=BATCH_NORM_RUNNING_MEAN "
                "workspace=WORKSPACE_GRAD "
                "status=VAE_C_OP_INVALID_ARGUMENT\n"
            );
        }
        model_snapshot_free(&model_before);
        workspace_snapshot_free(&workspace_before);
        byte_snapshot_free(&expanded_backing_before);
        if (!passed) {
            break;
        }
    }
    vae_c_model_workspace_free(&workspace);
    vae_c_model_free(&model);
    return passed;
}


static size_t collect_non_encoder_tensors(
    const VAECModelWorkspace *workspace,
    const VAECTensor *tensors[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ]
)
{
    size_t count = 0;
    size_t index;

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


static int snapshot_segment_matches(
    const unsigned char *current,
    size_t offset,
    size_t byte_count,
    const ByteSnapshot *snapshot
)
{
    return current != NULL && snapshot != NULL && snapshot->copy != NULL &&
        offset <= snapshot->byte_count &&
        byte_count <= snapshot->byte_count - offset &&
        memcmp(current + offset,
               snapshot->copy + offset,
               byte_count) == 0;
}


static int batch_norm_forward_metadata_allowed(
    const VAECBatchNorm *current,
    const VAECBatchNorm *before
)
{
    return current != NULL && before != NULL &&
        current->kind == before->kind &&
        current->num_features == before->num_features &&
        current->epsilon == before->epsilon &&
        current->momentum == before->momentum &&
        parameter_metadata_exact(&current->gamma, &before->gamma) &&
        parameter_metadata_exact(&current->beta, &before->beta) &&
        current->running_mean == before->running_mean &&
        current->running_variance == before->running_variance &&
        current->batch_mean == before->batch_mean &&
        current->batch_inv_std == before->batch_inv_std &&
        current->work_mean == before->work_mean &&
        current->work_inv_std == before->work_inv_std &&
        current->work_unbiased_variance ==
            before->work_unbiased_variance &&
        current->owned_block == before->owned_block &&
        current->owned_bytes == before->owned_bytes;
}


static int encoder_batch_norm_mutation_allowed(
    const VAECBatchNorm *current,
    const VAECBatchNorm *before,
    VAECBatchNormMode mode,
    const ByteSnapshot *state_before
)
{
    size_t segment_bytes;
    size_t segment;

    if (!batch_norm_forward_metadata_allowed(current, before) ||
        state_before == NULL || state_before->copy == NULL ||
        current->num_features == 0) {

        return 0;
    }
    if (mode == VAE_C_BATCH_NORM_MODE_TRAIN) {
        return current->cache_valid && current->last_sample_count >= 2 &&
            shape_fields_equal(&current->last_input_shape,
                               &current->last_output_shape);
    }
    if (mode != VAE_C_BATCH_NORM_MODE_EVAL || current->cache_valid ||
        current->last_sample_count != 0 ||
        current->last_input_shape.rank != 0 ||
        current->last_input_shape.count != 0 ||
        current->last_output_shape.rank != 0 ||
        current->last_output_shape.count != 0) {

        return 0;
    }

    /* EVAL may refresh only work_inv_std and invalidate cache metadata. */
    segment_bytes = current->num_features * sizeof(float);
    for (segment = 0; segment < 7; segment++) {
        if (segment != 5 && !snapshot_segment_matches(
                (const unsigned char *)current->owned_block,
                segment * segment_bytes,
                segment_bytes,
                state_before)) {

            return 0;
        }
    }
    return 1;
}


static int model_forward_mutation_allowed(
    const VAECModel *model,
    const ModelSnapshot *snapshot,
    VAECBatchNormMode mode
)
{
    const VAECModel *before;
    size_t index;

    if (model == NULL || snapshot == NULL || !snapshot->captured) {
        return 0;
    }
    before = &snapshot->metadata;
    if (memcmp(model->profile_id,
               before->profile_id,
               VAE_C_PROFILE_TEXT_CAP) != 0 ||
        !config_fields_equal(&model->config, &before->config) ||
        model->max_batch_size != before->max_batch_size ||
        !dense_metadata_exact(&model->encoder_fc,
                              &before->encoder_fc) ||
        !dense_metadata_exact(&model->decoder_fc,
                              &before->decoder_fc) ||
        !batch_norm_metadata_exact(&model->decoder_fc_batch_norm,
                                   &before->decoder_fc_batch_norm) ||
        !conv_metadata_exact(&model->decoder_output_conv,
                             &before->decoder_output_conv) ||
        model->encoder_max_pool.argmax_indices !=
            before->encoder_max_pool.argmax_indices ||
        model->encoder_max_pool.capacity !=
            before->encoder_max_pool.capacity ||
        model->encoder_max_pool.owned_bytes !=
            before->encoder_max_pool.owned_bytes ||
        model->parameter_count != before->parameter_count ||
        model->trainable_value_count != before->trainable_value_count ||
        model->initialization_seed != before->initialization_seed ||
        before->vae_cache_generation == UINT64_MAX ||
        model->vae_cache_generation !=
            before->vae_cache_generation + UINT64_C(1) ||
        model->parameters_initialized != before->parameters_initialized ||
        model->allocated != before->allocated ||
        !model->encoder_max_pool.cache_valid ||
        model->encoder_max_pool.last_count == 0) {

        return 0;
    }
    for (index = 0; index < VAE_C_MODEL_MAX_ENCODER_CONVS; index++) {
        if (!conv_metadata_exact(&model->encoder_convs[index],
                                 &before->encoder_convs[index])) {

            return 0;
        }
        if (index < model->config.conv_depth) {
            if (!encoder_batch_norm_mutation_allowed(
                    &model->encoder_batch_norms[index],
                    &before->encoder_batch_norms[index],
                    mode,
                    &snapshot->batch_norms[index])) {

                return 0;
            }
        } else if (!batch_norm_metadata_exact(
                       &model->encoder_batch_norms[index],
                       &before->encoder_batch_norms[index])) {

            return 0;
        }
    }
    for (index = 0; index < VAE_C_MODEL_MAX_DECODER_HIDDEN_CONVS;
         index++) {

        if (!conv_metadata_exact(&model->decoder_hidden_convs[index],
                                 &before->decoder_hidden_convs[index]) ||
            !batch_norm_metadata_exact(
                &model->decoder_hidden_batch_norms[index],
                &before->decoder_hidden_batch_norms[index])) {

            return 0;
        }
    }
    for (index = 0; index < VAE_C_MODEL_MAX_PARAMETERS; index++) {
        if (model->parameters[index] != before->parameters[index]) {
            return 0;
        }
    }
    for (index = 0; index < snapshot->parameter_count; index++) {
        const VAECParameter *parameter = model->parameters[index];

        if (parameter == NULL || !byte_snapshot_matches(
                parameter->owned_block,
                parameter->owned_bytes,
                &snapshot->parameters[index])) {

            return 0;
        }
    }
    for (index = VAE_C_MAX_CONV_DEPTH;
         index < TEST_BATCH_NORM_COUNT;
         index++) {

        const VAECBatchNorm *layers[TEST_BATCH_NORM_COUNT];

        if (collect_all_batch_norms(model, layers) !=
                TEST_BATCH_NORM_COUNT ||
            (snapshot->batch_norms[index].copy != NULL &&
             !byte_snapshot_matches(
                 layers[index]->owned_block,
                 layers[index]->owned_bytes,
                 &snapshot->batch_norms[index]))) {

            return 0;
        }
    }
    return 1;
}


static const ByteSnapshot *workspace_tensor_snapshot(
    const WorkspaceSnapshot *snapshot,
    const VAECTensor *tensor
)
{
    size_t index;

    if (snapshot == NULL || tensor == NULL || tensor->owned_block == NULL) {
        return NULL;
    }
    for (index = 0; index < snapshot->tensor_count; index++) {
        if (snapshot->tensors[index].pointer == tensor->owned_block) {
            return &snapshot->tensors[index];
        }
    }
    return NULL;
}


static int workspace_tensor_block_unchanged(
    const WorkspaceSnapshot *snapshot,
    const VAECTensor *tensor
)
{
    const ByteSnapshot *before = workspace_tensor_snapshot(snapshot, tensor);

    return before != NULL && byte_snapshot_matches(
        tensor->owned_block, tensor->owned_bytes, before
    );
}


static int workspace_tensor_grad_unchanged(
    const WorkspaceSnapshot *snapshot,
    const VAECTensor *tensor
)
{
    const ByteSnapshot *before = workspace_tensor_snapshot(snapshot, tensor);
    size_t data_bytes;

    if (tensor == NULL || !tensor->has_grad) {
        return tensor != NULL && tensor->grad == NULL;
    }
    data_bytes = tensor->shape.count * sizeof(float);
    return before != NULL && snapshot_segment_matches(
        (const unsigned char *)tensor->owned_block,
        data_bytes,
        data_bytes,
        before
    );
}


static int workspace_forward_mutation_allowed(
    const VAECModelWorkspace *workspace,
    const WorkspaceSnapshot *snapshot
)
{
    const VAECModelWorkspace *before;
    VAECModelWorkspace expected;
    const VAECTensor *all_tensors[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ];
    size_t tensor_count;
    size_t index;

    if (workspace == NULL || snapshot == NULL || !snapshot->captured) {
        return 0;
    }
    before = &snapshot->metadata;
    expected = *before;
    expected.vae_backward_model = NULL;
    expected.vae_backward_generation = UINT64_C(0);
    if (!workspace_metadata_exact(workspace, &expected) ||
        !workspace_tensor_block_unchanged(snapshot, &workspace->input)) {

        return 0;
    }

    for (index = 0; index < workspace->config.conv_depth; index++) {
        if (!workspace_tensor_grad_unchanged(
                snapshot, &workspace->encoder_conv_outputs[index]) ||
            !workspace_tensor_grad_unchanged(
                snapshot,
                &workspace->encoder_batch_norm_outputs[index]) ||
            !workspace_tensor_grad_unchanged(
                snapshot, &workspace->encoder_relu_outputs[index])) {

            return 0;
        }
    }
    if (!workspace_tensor_grad_unchanged(
            snapshot, &workspace->encoder_pool_output) ||
        !workspace_tensor_grad_unchanged(
            snapshot, &workspace->encoder_global_avg_output) ||
        !workspace_tensor_grad_unchanged(
            snapshot, &workspace->encoder_packed_output) ||
        !workspace_tensor_grad_unchanged(snapshot, &workspace->mu) ||
        !workspace_tensor_grad_unchanged(snapshot, &workspace->logvar)) {

        return 0;
    }

    tensor_count = collect_all_workspace_tensors(workspace, all_tensors);
    for (index = 0; index < tensor_count; index++) {
        const VAECTensor *tensor = all_tensors[index];
        size_t encoder_index;
        int data_may_change =
            tensor == &workspace->encoder_pool_output ||
            tensor == &workspace->encoder_global_avg_output ||
            tensor == &workspace->encoder_packed_output ||
            tensor == &workspace->mu || tensor == &workspace->logvar;

        for (encoder_index = 0;
             encoder_index < workspace->config.conv_depth;
             encoder_index++) {

            data_may_change = data_may_change ||
                tensor == &workspace->encoder_conv_outputs[encoder_index] ||
                tensor ==
                    &workspace->encoder_batch_norm_outputs[encoder_index] ||
                tensor == &workspace->encoder_relu_outputs[encoder_index];
        }
        if (tensor->owned_block != NULL && !data_may_change &&
            !workspace_tensor_block_unchanged(snapshot, tensor)) {

            return 0;
        }
    }
    return tensor_count == VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS;
}


static int run_valid_forward_mutation_boundary(
    const VAECProfile *profile,
    VAECBatchNormMode mode
)
{
    const VAECTensor *all_tensors[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ];
    const VAECTensor *non_encoder[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ];
    ModelSnapshot model_before;
    WorkspaceSnapshot workspace_before;
    VAECModel model;
    VAECModelWorkspace workspace;
    size_t tensor_count;
    size_t non_encoder_count;
    size_t index;
    int passed = 1;

    model_snapshot_init(&model_before);
    workspace_snapshot_init(&workspace_before);
    if (!make_initialized_fixture_from_profile(
            profile, &model, &workspace)) {
        vae_c_model_workspace_free(&workspace);
        vae_c_model_free(&model);
        return 0;
    }
    for (index = 0; index < model.parameter_count; index++) {
        VAECParameter *parameter = model.parameters[index];
        size_t value;

        for (value = 0; value < parameter->shape.count; value++) {
            parameter->gradients[value] =
                0.01f * (float)((value % 7) + 1);
            parameter->adam_m[value] =
                -0.02f * (float)((value % 5) + 1);
            parameter->adam_v[value] =
                0.03f * (float)((value % 3) + 1);
        }
    }
    model.vae_cache_generation = UINT64_C(17);
    workspace.vae_backward_model = &model;
    workspace.vae_backward_generation = UINT64_C(17);
    tensor_count = collect_all_workspace_tensors(&workspace, all_tensors);
    for (index = 0; index < tensor_count; index++) {
        size_t value;

        if (all_tensors[index]->grad != NULL) {
            for (value = 0;
                 value < all_tensors[index]->shape.count;
                 value++) {

                all_tensors[index]->grad[value] =
                    0.125f + 0.001f * (float)(value % 19);
            }
        }
    }
    non_encoder_count = collect_non_encoder_tensors(
        &workspace, non_encoder
    );
    for (index = 0; index < non_encoder_count; index++) {
        size_t value;

        for (value = 0; value < non_encoder[index]->shape.count; value++) {
            non_encoder[index]->data[value] =
                0.2f + 0.01f * (float)index +
                0.0001f * (float)(value % 31);
        }
    }
    passed = model_snapshot_capture(&model, &model_before) &&
        workspace_snapshot_capture(&workspace, &workspace_before) &&
        vae_c_model_validate(&model) == VAE_C_OP_OK &&
        vae_c_model_workspace_validate(&model, &workspace) == VAE_C_OP_OK &&
        vae_c_model_encoder_forward(&model, mode, &workspace) ==
            VAE_C_OP_OK &&
        model_forward_mutation_allowed(&model, &model_before, mode) &&
        workspace_forward_mutation_allowed(&workspace, &workspace_before) &&
        vae_c_model_validate(&model) == VAE_C_OP_OK &&
        vae_c_model_workspace_validate(&model, &workspace) == VAE_C_OP_OK;

    model_snapshot_free(&model_before);
    workspace_snapshot_free(&workspace_before);
    vae_c_model_workspace_free(&workspace);
    vae_c_model_free(&model);
    return passed;
}


static int test_generation_ticket_and_overflow(
    const VAECProfileCatalog *catalog
)
{
    const VAECProfile *profile = vae_c_profile_catalog_find(
        catalog, "mini75_smoke_v1"
    );
    VAECModel model;
    VAECModelWorkspace workspace;
    ModelSnapshot model_before;
    WorkspaceSnapshot workspace_before;
    int passed;

    model_snapshot_init(&model_before);
    workspace_snapshot_init(&workspace_before);
    if (!make_initialized_fixture_from_profile(
            profile, &model, &workspace)) {

        vae_c_model_workspace_free(&workspace);
        vae_c_model_free(&model);
        return 0;
    }

    model.vae_cache_generation = UINT64_C(23);
    workspace.vae_backward_model = &model;
    workspace.vae_backward_generation = UINT64_C(23);
    passed = vae_c_model_encoder_forward(
        &model, VAE_C_BATCH_NORM_MODE_EVAL, &workspace
    ) == VAE_C_OP_OK &&
        model.vae_cache_generation == UINT64_C(24) &&
        workspace.vae_backward_model == NULL &&
        workspace.vae_backward_generation == UINT64_C(0);

    model.vae_cache_generation = UINT64_MAX;
    workspace.vae_backward_model = &model;
    workspace.vae_backward_generation = UINT64_MAX;
    passed = passed &&
        model_snapshot_capture(&model, &model_before) &&
        workspace_snapshot_capture(&workspace, &workspace_before) &&
        vae_c_model_encoder_forward(
            &model, VAE_C_BATCH_NORM_MODE_EVAL, &workspace
        ) == VAE_C_OP_OVERFLOW &&
        model_snapshot_matches(&model, &model_before) &&
        workspace_snapshot_matches(&workspace, &workspace_before);

    model_snapshot_free(&model_before);
    workspace_snapshot_free(&workspace_before);
    vae_c_model_workspace_free(&workspace);
    vae_c_model_free(&model);
    return passed;
}


static int test_valid_forward_mutation_boundaries(
    const VAECProfileCatalog *catalog
)
{
    const VAECProfile *depth3 = vae_c_profile_catalog_find(
        catalog, "mini75_smoke_v1"
    );
    VAECProfile depth4;

    make_synthetic_depth4(&depth4);
    return depth3 != NULL && run_valid_forward_mutation_boundary(
        depth3, VAE_C_BATCH_NORM_MODE_TRAIN
    ) && run_valid_forward_mutation_boundary(
        depth3, VAE_C_BATCH_NORM_MODE_EVAL
    ) && run_valid_forward_mutation_boundary(
        &depth4, VAE_C_BATCH_NORM_MODE_TRAIN
    ) && run_valid_forward_mutation_boundary(
        &depth4, VAE_C_BATCH_NORM_MODE_EVAL
    );
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
        "mini75_train_manual_equivalence",
        test_mini75_train_manual_equivalence(&catalog)
    );
    failures += report_check(
        "mini75_eval_manual_equivalence",
        test_mini75_eval_manual_equivalence(&catalog)
    );
    failures += report_check(
        "synthetic_depth4_train_manual_equivalence",
        test_synthetic_depth4_manual_equivalence()
    );
    failures += report_check(
        "synthetic_depth4_train_batch1_rejected_eval_batch1_valid",
        test_synthetic_depth4_batch_modes()
    );
    failures += report_check(
        "all_official_profiles_eval_batch1_and_batch2",
        test_official_profiles_eval(&catalog)
    );
    failures += report_check(
        "eval_determinism_independent_models",
        test_eval_determinism(&catalog)
    );
    failures += report_check(
        "invalid_arguments_and_empty_state_atomic",
        test_invalid_arguments_and_empty_state(&catalog)
    );
    failures += report_check(
        "uninitialized_model_rejected_atomic",
        test_uninitialized_model_rejected(&catalog)
    );
    failures += report_check(
        "input_domain_preflight_atomic",
        test_preflight_input_domain(&catalog)
    );
    failures += report_check(
        "parameter_value_preflight_atomic",
        test_preflight_parameter_values(&catalog)
    );
    failures += report_check(
        "structural_preflight_atomic",
        test_preflight_structural_corruptions(&catalog)
    );
    failures += report_check(
        "exact_alias_preflight_atomic",
        test_preflight_exact_alias(&catalog)
    );
    failures += report_check(
        "valid_forward_mutation_boundaries",
        test_valid_forward_mutation_boundaries(&catalog)
    );
    failures += report_check(
        "generation_ticket_and_overflow",
        test_generation_ticket_and_overflow(&catalog)
    );

    vae_c_profile_catalog_free(&catalog);
    if (failures != 0) {
        fprintf(stderr, "%d VAE C ENCODER FORWARD TEST(S) FAILED\n",
                failures);
        return 1;
    }
    printf("ENCODER_FORWARD_STRONG_GUARANTEE=preflight_only\n");
    printf("ENCODER_FORWARD_POST_START_ROLLBACK=false\n");
    printf("POST_START_FAILURE_INJECTION=not_forced_portably\n");
    printf("ALL VAE C MODEL ENCODER FORWARD TESTS PASSED\n");
    return 0;
}
