#include "vae_c_model.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>


static int model_size_multiply(
    size_t left,
    size_t right,
    size_t *result
)
{
    if (result == NULL) {
        return 1;
    }
    if (left != 0 && right > SIZE_MAX / left) {
        return 1;
    }
    *result = left * right;
    return 0;
}


static bool model_bytes_are_zero(const void *object, size_t size)
{
    const unsigned char *bytes;
    size_t index;

    if (object == NULL) {
        return false;
    }
    bytes = (const unsigned char *)object;
    for (index = 0; index < size; index++) {
        if (bytes[index] != 0) {
            return false;
        }
    }
    return true;
}


static bool model_shape_is_zero(const VAECShape *shape)
{
    size_t index;

    if (shape == NULL || shape->rank != 0 || shape->count != 0) {
        return false;
    }
    for (index = 0; index < VAE_C_TENSOR_MAX_RANK; index++) {
        if (shape->dims[index] != 0 || shape->strides[index] != 0) {
            return false;
        }
    }
    return true;
}


static VAECOpStatus model_validate_shape(const VAECShape *shape)
{
    VAECShape canonical;
    VAECStorageStatus storage_status;

    if (shape == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    storage_status = vae_c_shape_make(
        shape->rank,
        shape->dims,
        &canonical
    );
    if (storage_status != VAE_C_STORAGE_OK) {
        return
            storage_status == VAE_C_STORAGE_OVERFLOW
                ? VAE_C_OP_OVERFLOW
                : VAE_C_OP_INVALID_STATE;
    }
    return
        memcmp(shape, &canonical, sizeof(canonical)) == 0
            ? VAE_C_OP_OK
            : VAE_C_OP_INVALID_STATE;
}


static bool model_all_finite(const float *values, size_t count)
{
    size_t index;

    if (values == NULL) {
        return false;
    }
    for (index = 0; index < count; index++) {
        if (!isfinite(values[index])) {
            return false;
        }
    }
    return true;
}


static VAECOpStatus model_validate_parameter(
    const VAECParameter *parameter
)
{
    VAECOpStatus status;
    size_t total_elements;
    size_t expected_bytes;
    size_t index;

    if (parameter == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    status = model_validate_shape(&parameter->shape);
    if (status != VAE_C_OP_OK || parameter->shape.count == 0) {
        return VAE_C_OP_INVALID_STATE;
    }
    if (parameter->parameter_key == UINT64_C(0) ||
        parameter->owned_block == NULL ||
        parameter->values != parameter->owned_block ||
        parameter->gradients !=
            parameter->values + parameter->shape.count ||
        parameter->adam_m !=
            parameter->gradients + parameter->shape.count ||
        parameter->adam_v !=
            parameter->adam_m + parameter->shape.count) {

        return VAE_C_OP_INVALID_STATE;
    }
    if (model_size_multiply(
            parameter->shape.count,
            4,
            &total_elements) != 0 ||
        model_size_multiply(
            total_elements,
            sizeof(float),
            &expected_bytes) != 0 ||
        parameter->owned_bytes != expected_bytes ||
        !model_all_finite(
            parameter->values,
            parameter->shape.count) ||
        !model_all_finite(
            parameter->gradients,
            parameter->shape.count) ||
        !model_all_finite(
            parameter->adam_m,
            parameter->shape.count) ||
        !model_all_finite(
            parameter->adam_v,
            parameter->shape.count)) {

        return VAE_C_OP_INVALID_STATE;
    }
    for (index = 0; index < parameter->shape.count; index++) {
        if (parameter->adam_v[index] < 0.0f) {
            return VAE_C_OP_INVALID_STATE;
        }
    }
    return VAE_C_OP_OK;
}


static bool model_parameters_have_exact_alias(
    const VAECParameter *left,
    const VAECParameter *right
)
{
    const void *left_regions[5];
    const void *right_regions[5];
    size_t left_index;
    size_t right_index;

    if (left == NULL || right == NULL || left == right) {
        return true;
    }
    left_regions[0] = left->values;
    left_regions[1] = left->gradients;
    left_regions[2] = left->adam_m;
    left_regions[3] = left->adam_v;
    left_regions[4] = left->owned_block;
    right_regions[0] = right->values;
    right_regions[1] = right->gradients;
    right_regions[2] = right->adam_m;
    right_regions[3] = right->adam_v;
    right_regions[4] = right->owned_block;

    for (left_index = 0; left_index < 5; left_index++) {
        for (right_index = 0; right_index < 5; right_index++) {
            if (left_regions[left_index] == right_regions[right_index]) {
                return true;
            }
        }
    }
    return false;
}


static VAECOpStatus model_validate_conv2d(
    const VAECConv2D *layer,
    size_t in_channels,
    size_t out_channels,
    size_t stride,
    bool with_bias,
    uint64_t weight_key,
    uint64_t bias_key
)
{
    if (layer == NULL ||
        layer->in_channels != in_channels ||
        layer->out_channels != out_channels ||
        layer->kernel_size != VAE_C_CONV2D_KERNEL_SIZE ||
        layer->stride != stride ||
        layer->padding != VAE_C_CONV2D_PADDING ||
        layer->with_bias != with_bias ||
        model_validate_parameter(&layer->weights) != VAE_C_OP_OK ||
        layer->weights.parameter_key != weight_key ||
        layer->weights.shape.rank != 4 ||
        layer->weights.shape.dims[0] != out_channels ||
        layer->weights.shape.dims[1] != in_channels ||
        layer->weights.shape.dims[2] != VAE_C_CONV2D_KERNEL_SIZE ||
        layer->weights.shape.dims[3] != VAE_C_CONV2D_KERNEL_SIZE) {

        return VAE_C_OP_INVALID_STATE;
    }
    if (!with_bias) {
        return
            vae_c_parameter_is_empty(&layer->bias)
                ? VAE_C_OP_OK
                : VAE_C_OP_INVALID_STATE;
    }
    if (model_validate_parameter(&layer->bias) != VAE_C_OP_OK ||
        layer->bias.parameter_key != bias_key ||
        layer->bias.shape.rank != 1 ||
        layer->bias.shape.dims[0] != out_channels ||
        model_parameters_have_exact_alias(
            &layer->weights,
            &layer->bias)) {

        return VAE_C_OP_INVALID_STATE;
    }
    return VAE_C_OP_OK;
}


static VAECOpStatus model_validate_dense(
    const VAECDense *layer,
    size_t in_features,
    size_t out_features,
    bool with_bias,
    uint64_t weight_key,
    uint64_t bias_key
)
{
    if (layer == NULL ||
        layer->in_features != in_features ||
        layer->out_features != out_features ||
        layer->with_bias != with_bias ||
        model_validate_parameter(&layer->weights) != VAE_C_OP_OK ||
        layer->weights.parameter_key != weight_key ||
        layer->weights.shape.rank != 2 ||
        layer->weights.shape.dims[0] != out_features ||
        layer->weights.shape.dims[1] != in_features) {

        return VAE_C_OP_INVALID_STATE;
    }
    if (!with_bias) {
        return
            vae_c_parameter_is_empty(&layer->bias)
                ? VAE_C_OP_OK
                : VAE_C_OP_INVALID_STATE;
    }
    if (model_validate_parameter(&layer->bias) != VAE_C_OP_OK ||
        layer->bias.parameter_key != bias_key ||
        layer->bias.shape.rank != 1 ||
        layer->bias.shape.dims[0] != out_features ||
        model_parameters_have_exact_alias(
            &layer->weights,
            &layer->bias)) {

        return VAE_C_OP_INVALID_STATE;
    }
    return VAE_C_OP_OK;
}


static bool model_batch_norm_buffers_are_unique(
    const VAECBatchNorm *layer
)
{
    const void *buffers[15];
    size_t left;
    size_t right;

    buffers[0] = layer->gamma.values;
    buffers[1] = layer->gamma.gradients;
    buffers[2] = layer->gamma.adam_m;
    buffers[3] = layer->gamma.adam_v;
    buffers[4] = layer->beta.values;
    buffers[5] = layer->beta.gradients;
    buffers[6] = layer->beta.adam_m;
    buffers[7] = layer->beta.adam_v;
    buffers[8] = layer->running_mean;
    buffers[9] = layer->running_variance;
    buffers[10] = layer->batch_mean;
    buffers[11] = layer->batch_inv_std;
    buffers[12] = layer->work_mean;
    buffers[13] = layer->work_inv_std;
    buffers[14] = layer->work_unbiased_variance;

    for (left = 0; left < 15; left++) {
        if (buffers[left] == NULL) {
            return false;
        }
        for (right = left + 1; right < 15; right++) {
            if (buffers[left] == buffers[right]) {
                return false;
            }
        }
    }
    return true;
}


static VAECOpStatus model_validate_batch_norm_cache(
    const VAECBatchNorm *layer
)
{
    size_t expected_samples;
    size_t spatial;
    size_t feature;

    if (!layer->cache_valid) {
        return
            layer->last_sample_count == 0 &&
            model_shape_is_zero(&layer->last_input_shape) &&
            model_shape_is_zero(&layer->last_output_shape)
                ? VAE_C_OP_OK
                : VAE_C_OP_INVALID_STATE;
    }
    if (layer->last_sample_count < 2 ||
        model_validate_shape(&layer->last_input_shape) != VAE_C_OP_OK ||
        model_validate_shape(&layer->last_output_shape) != VAE_C_OP_OK ||
        !vae_c_shape_equal(
            &layer->last_input_shape,
            &layer->last_output_shape)) {

        return VAE_C_OP_INVALID_STATE;
    }
    if (layer->kind == VAE_C_BATCH_NORM_KIND_1D) {
        if (layer->last_input_shape.rank != 2 ||
            layer->last_input_shape.dims[1] != layer->num_features) {

            return VAE_C_OP_INVALID_STATE;
        }
        expected_samples = layer->last_input_shape.dims[0];
    } else {
        if (layer->last_input_shape.rank != 4 ||
            layer->last_input_shape.dims[1] != layer->num_features ||
            model_size_multiply(
                layer->last_input_shape.dims[2],
                layer->last_input_shape.dims[3],
                &spatial) != 0 ||
            model_size_multiply(
                layer->last_input_shape.dims[0],
                spatial,
                &expected_samples) != 0) {

            return VAE_C_OP_INVALID_STATE;
        }
    }
    if (expected_samples != layer->last_sample_count) {
        return VAE_C_OP_INVALID_STATE;
    }
    for (feature = 0; feature < layer->num_features; feature++) {
        if (!isfinite(layer->batch_mean[feature]) ||
            !isfinite(layer->batch_inv_std[feature]) ||
            layer->batch_inv_std[feature] <= 0.0f) {

            return VAE_C_OP_INVALID_STATE;
        }
    }
    return VAE_C_OP_OK;
}


static VAECOpStatus model_validate_batch_norm(
    const VAECBatchNorm *layer,
    VAECBatchNormKind kind,
    size_t num_features,
    uint64_t gamma_key,
    uint64_t beta_key
)
{
    size_t total_elements;
    size_t expected_bytes;
    size_t feature;

    if (layer == NULL ||
        layer->kind != kind ||
        layer->num_features != num_features ||
        layer->epsilon != VAE_C_BATCH_NORM_EPSILON ||
        layer->momentum != VAE_C_BATCH_NORM_MOMENTUM ||
        model_validate_parameter(&layer->gamma) != VAE_C_OP_OK ||
        model_validate_parameter(&layer->beta) != VAE_C_OP_OK ||
        layer->gamma.parameter_key != gamma_key ||
        layer->beta.parameter_key != beta_key ||
        layer->gamma.shape.rank != 1 ||
        layer->gamma.shape.dims[0] != num_features ||
        layer->beta.shape.rank != 1 ||
        layer->beta.shape.dims[0] != num_features ||
        layer->owned_block == NULL ||
        model_size_multiply(num_features, 7, &total_elements) != 0 ||
        model_size_multiply(
            total_elements,
            sizeof(float),
            &expected_bytes) != 0 ||
        layer->owned_bytes != expected_bytes ||
        layer->running_mean != layer->owned_block ||
        layer->running_variance !=
            layer->running_mean + num_features ||
        layer->batch_mean !=
            layer->running_variance + num_features ||
        layer->batch_inv_std !=
            layer->batch_mean + num_features ||
        layer->work_mean !=
            layer->batch_inv_std + num_features ||
        layer->work_inv_std !=
            layer->work_mean + num_features ||
        layer->work_unbiased_variance !=
            layer->work_inv_std + num_features ||
        !model_batch_norm_buffers_are_unique(layer) ||
        !model_all_finite(layer->owned_block, total_elements)) {

        return VAE_C_OP_INVALID_STATE;
    }
    for (feature = 0; feature < num_features; feature++) {
        if (layer->running_variance[feature] < 0.0f) {
            return VAE_C_OP_INVALID_STATE;
        }
    }
    return model_validate_batch_norm_cache(layer);
}


static VAECOpStatus model_validate_max_pool(
    const VAECMaxPool2D *layer,
    size_t expected_capacity
)
{
    size_t expected_bytes;
    size_t expected_height;
    size_t expected_width;

    if (layer == NULL ||
        layer->argmax_indices == NULL ||
        layer->capacity != expected_capacity ||
        model_size_multiply(
            expected_capacity,
            sizeof(size_t),
            &expected_bytes) != 0 ||
        layer->owned_bytes != expected_bytes) {

        return VAE_C_OP_INVALID_STATE;
    }
    if (!layer->cache_valid) {
        return
            layer->last_count == 0 &&
            model_shape_is_zero(&layer->last_input_shape) &&
            model_shape_is_zero(&layer->last_output_shape)
                ? VAE_C_OP_OK
                : VAE_C_OP_INVALID_STATE;
    }
    if (layer->last_count == 0 ||
        layer->last_count > layer->capacity ||
        model_validate_shape(&layer->last_input_shape) != VAE_C_OP_OK ||
        model_validate_shape(&layer->last_output_shape) != VAE_C_OP_OK ||
        layer->last_input_shape.rank != 4 ||
        layer->last_output_shape.rank != 4 ||
        layer->last_input_shape.dims[2] < 2 ||
        layer->last_input_shape.dims[3] < 2) {

        return VAE_C_OP_INVALID_STATE;
    }
    expected_height =
        (layer->last_input_shape.dims[2] - 2) / 2 + 1;
    expected_width =
        (layer->last_input_shape.dims[3] - 2) / 2 + 1;
    if (layer->last_input_shape.dims[0] !=
            layer->last_output_shape.dims[0] ||
        layer->last_input_shape.dims[1] !=
            layer->last_output_shape.dims[1] ||
        layer->last_output_shape.dims[2] != expected_height ||
        layer->last_output_shape.dims[3] != expected_width ||
        layer->last_output_shape.count != layer->last_count) {

        return VAE_C_OP_INVALID_STATE;
    }
    return VAE_C_OP_OK;
}


static bool model_profile_id_is_canonical(
    const char profile_id[VAE_C_PROFILE_TEXT_CAP],
    size_t *length
)
{
    const char *terminator;
    size_t index;
    size_t text_length;

    if (profile_id == NULL || profile_id[0] == '\0') {
        return false;
    }
    terminator = (const char *)memchr(
        profile_id,
        '\0',
        VAE_C_PROFILE_TEXT_CAP
    );
    if (terminator == NULL) {
        return false;
    }
    text_length = (size_t)(terminator - profile_id);
    for (index = text_length + 1;
         index < VAE_C_PROFILE_TEXT_CAP;
         index++) {

        if (profile_id[index] != '\0') {
            return false;
        }
    }
    if (length != NULL) {
        *length = text_length;
    }
    return true;
}


static VAECOpStatus model_map_config_status(VAECConfigStatus status)
{
    if (status == VAE_C_CONFIG_OK) {
        return VAE_C_OP_OK;
    }
    if (status == VAE_C_CONFIG_INVALID_ARGUMENT) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (status == VAE_C_CONFIG_ALLOCATION_FAILED) {
        return VAE_C_OP_ALLOCATION_FAILED;
    }
    if (status == VAE_C_CONFIG_SHAPE_MISMATCH ||
        status == VAE_C_CONFIG_COUNT_MISMATCH) {

        return VAE_C_OP_SHAPE_MISMATCH;
    }
    return VAE_C_OP_INVALID_STATE;
}


static VAECOpStatus model_validate_config(const VAECModel *model)
{
    static const char validation_role[] = "model_validation";
    static const char validation_status[] = "allocated";
    VAECProfile profile;
    VAECConfig rebuilt;
    VAECConfigStatus config_status;
    size_t profile_id_length;
    size_t index;

    if (!model_profile_id_is_canonical(
            model->profile_id,
            &profile_id_length)) {

        return VAE_C_OP_INVALID_STATE;
    }
    memset(&profile, 0, sizeof(profile));
    memcpy(
        profile.profile_id,
        model->profile_id,
        profile_id_length + 1
    );
    memcpy(profile.role, validation_role, sizeof(validation_role));
    memcpy(
        profile.profile_status,
        validation_status,
        sizeof(validation_status)
    );
    profile.input_side = model->config.input_side;
    profile.conv_depth = model->config.conv_depth;
    for (index = 0; index < VAE_C_MAX_CONV_DEPTH; index++) {
        profile.channels[index] = model->config.channels[index];
    }
    profile.latent_dim = model->config.latent_dim;
    profile.bottleneck_side = model->config.bottleneck_side;
    profile.expected_trainable_parameters_bn1d =
        model->config.trainable_parameters_bn1d;
    profile.expected_forward_mac_per_sample =
        model->config.forward_mac_per_sample;

    config_status = vae_c_config_build(
        &profile,
        &rebuilt,
        NULL,
        0
    );
    if (config_status != VAE_C_CONFIG_OK ||
        memcmp(&model->config, &rebuilt, sizeof(rebuilt)) != 0) {

        return VAE_C_OP_INVALID_STATE;
    }
    return VAE_C_OP_OK;
}


static VAECOpStatus model_expected_parameter_count(
    size_t conv_depth,
    size_t *parameter_count
)
{
    size_t product;

    if (parameter_count == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (conv_depth != 3 && conv_depth != 4) {
        return VAE_C_OP_INVALID_STATE;
    }
    if (model_size_multiply(conv_depth, 6, &product) != 0 ||
        product > SIZE_MAX - 4) {

        return VAE_C_OP_OVERFLOW;
    }
    *parameter_count = product + 4;
    return VAE_C_OP_OK;
}


static size_t model_expected_keys(
    size_t conv_depth,
    uint64_t keys[VAE_C_MODEL_MAX_PARAMETERS]
)
{
    size_t index;
    size_t count;

    count = 0;
    for (index = 0; index < conv_depth; index++) {
        keys[count++] = UINT64_C(1) + UINT64_C(3) * (uint64_t)index;
        keys[count++] = UINT64_C(2) + UINT64_C(3) * (uint64_t)index;
        keys[count++] = UINT64_C(3) + UINT64_C(3) * (uint64_t)index;
    }
    keys[count++] = VAE_C_MODEL_PARAM_KEY_ENCODER_FC_WEIGHT;
    keys[count++] = VAE_C_MODEL_PARAM_KEY_ENCODER_FC_BIAS;
    keys[count++] = VAE_C_MODEL_PARAM_KEY_DECODER_FC_WEIGHT;
    keys[count++] = VAE_C_MODEL_PARAM_KEY_DECODER_FC_BN_GAMMA;
    keys[count++] = VAE_C_MODEL_PARAM_KEY_DECODER_FC_BN_BETA;
    for (index = 0; index < conv_depth - 1; index++) {
        keys[count++] = UINT64_C(18) + UINT64_C(3) * (uint64_t)index;
        keys[count++] = UINT64_C(19) + UINT64_C(3) * (uint64_t)index;
        keys[count++] = UINT64_C(20) + UINT64_C(3) * (uint64_t)index;
    }
    keys[count++] = VAE_C_MODEL_PARAM_KEY_DECODER_OUTPUT_WEIGHT;
    keys[count++] = VAE_C_MODEL_PARAM_KEY_DECODER_OUTPUT_BIAS;
    return count;
}


static VAECParameter *model_parameter_by_key(
    VAECModel *model,
    uint64_t key
)
{
    switch (key) {
        case VAE_C_MODEL_PARAM_KEY_ENCODER_CONV1_WEIGHT:
            return &model->encoder_convs[0].weights;
        case VAE_C_MODEL_PARAM_KEY_ENCODER_BN1_GAMMA:
            return &model->encoder_batch_norms[0].gamma;
        case VAE_C_MODEL_PARAM_KEY_ENCODER_BN1_BETA:
            return &model->encoder_batch_norms[0].beta;
        case VAE_C_MODEL_PARAM_KEY_ENCODER_CONV2_WEIGHT:
            return &model->encoder_convs[1].weights;
        case VAE_C_MODEL_PARAM_KEY_ENCODER_BN2_GAMMA:
            return &model->encoder_batch_norms[1].gamma;
        case VAE_C_MODEL_PARAM_KEY_ENCODER_BN2_BETA:
            return &model->encoder_batch_norms[1].beta;
        case VAE_C_MODEL_PARAM_KEY_ENCODER_CONV3_WEIGHT:
            return &model->encoder_convs[2].weights;
        case VAE_C_MODEL_PARAM_KEY_ENCODER_BN3_GAMMA:
            return &model->encoder_batch_norms[2].gamma;
        case VAE_C_MODEL_PARAM_KEY_ENCODER_BN3_BETA:
            return &model->encoder_batch_norms[2].beta;
        case VAE_C_MODEL_PARAM_KEY_ENCODER_CONV4_WEIGHT:
            return &model->encoder_convs[3].weights;
        case VAE_C_MODEL_PARAM_KEY_ENCODER_BN4_GAMMA:
            return &model->encoder_batch_norms[3].gamma;
        case VAE_C_MODEL_PARAM_KEY_ENCODER_BN4_BETA:
            return &model->encoder_batch_norms[3].beta;
        case VAE_C_MODEL_PARAM_KEY_ENCODER_FC_WEIGHT:
            return &model->encoder_fc.weights;
        case VAE_C_MODEL_PARAM_KEY_ENCODER_FC_BIAS:
            return &model->encoder_fc.bias;
        case VAE_C_MODEL_PARAM_KEY_DECODER_FC_WEIGHT:
            return &model->decoder_fc.weights;
        case VAE_C_MODEL_PARAM_KEY_DECODER_FC_BN_GAMMA:
            return &model->decoder_fc_batch_norm.gamma;
        case VAE_C_MODEL_PARAM_KEY_DECODER_FC_BN_BETA:
            return &model->decoder_fc_batch_norm.beta;
        case VAE_C_MODEL_PARAM_KEY_DECODER_CONV1_WEIGHT:
            return &model->decoder_hidden_convs[0].weights;
        case VAE_C_MODEL_PARAM_KEY_DECODER_BN1_GAMMA:
            return &model->decoder_hidden_batch_norms[0].gamma;
        case VAE_C_MODEL_PARAM_KEY_DECODER_BN1_BETA:
            return &model->decoder_hidden_batch_norms[0].beta;
        case VAE_C_MODEL_PARAM_KEY_DECODER_CONV2_WEIGHT:
            return &model->decoder_hidden_convs[1].weights;
        case VAE_C_MODEL_PARAM_KEY_DECODER_BN2_GAMMA:
            return &model->decoder_hidden_batch_norms[1].gamma;
        case VAE_C_MODEL_PARAM_KEY_DECODER_BN2_BETA:
            return &model->decoder_hidden_batch_norms[1].beta;
        case VAE_C_MODEL_PARAM_KEY_DECODER_CONV3_WEIGHT:
            return &model->decoder_hidden_convs[2].weights;
        case VAE_C_MODEL_PARAM_KEY_DECODER_BN3_GAMMA:
            return &model->decoder_hidden_batch_norms[2].gamma;
        case VAE_C_MODEL_PARAM_KEY_DECODER_BN3_BETA:
            return &model->decoder_hidden_batch_norms[2].beta;
        case VAE_C_MODEL_PARAM_KEY_DECODER_OUTPUT_WEIGHT:
            return &model->decoder_output_conv.weights;
        case VAE_C_MODEL_PARAM_KEY_DECODER_OUTPUT_BIAS:
            return &model->decoder_output_conv.bias;
        default:
            return NULL;
    }
}


static const VAECParameter *model_const_parameter_by_key(
    const VAECModel *model,
    uint64_t key
)
{
    return model_parameter_by_key((VAECModel *)(void *)model, key);
}


static void model_rebind_registry(VAECModel *model)
{
    uint64_t keys[VAE_C_MODEL_MAX_PARAMETERS];
    size_t count;
    size_t index;

    for (index = 0; index < VAE_C_MODEL_MAX_PARAMETERS; index++) {
        model->parameters[index] = NULL;
    }
    count = model_expected_keys(model->config.conv_depth, keys);
    for (index = 0; index < count; index++) {
        model->parameters[index] = model_parameter_by_key(
            model,
            keys[index]
        );
    }
    model->parameter_count = count;
}


static VAECOpStatus model_sum_registry(
    const VAECModel *model,
    uint64_t *total
)
{
    uint64_t sum;
    size_t index;

    if (model == NULL || total == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    sum = UINT64_C(0);
    for (index = 0; index < model->parameter_count; index++) {
        uint64_t count;

        if (model->parameters[index] == NULL) {
            return VAE_C_OP_INVALID_STATE;
        }
        count = (uint64_t)model->parameters[index]->shape.count;
        if ((size_t)count != model->parameters[index]->shape.count ||
            count > UINT64_MAX - sum) {

            return VAE_C_OP_OVERFLOW;
        }
        sum += count;
    }
    *total = sum;
    return VAE_C_OP_OK;
}


static VAECOpStatus model_validate_registry(const VAECModel *model)
{
    uint64_t keys[VAE_C_MODEL_MAX_PARAMETERS];
    uint64_t total;
    size_t expected_count;
    size_t key_count;
    size_t index;
    size_t other;

    if (model_expected_parameter_count(
            model->config.conv_depth,
            &expected_count) != VAE_C_OP_OK ||
        expected_count > VAE_C_MODEL_MAX_PARAMETERS ||
        model->parameter_count != expected_count) {

        return VAE_C_OP_INVALID_STATE;
    }
    key_count = model_expected_keys(model->config.conv_depth, keys);
    if (key_count != expected_count) {
        return VAE_C_OP_INVALID_STATE;
    }
    for (index = 0; index < expected_count; index++) {
        const VAECParameter *parameter = model->parameters[index];
        const VAECParameter *expected = model_const_parameter_by_key(
            model,
            keys[index]
        );

        if (parameter == NULL ||
            expected == NULL ||
            parameter != expected ||
            model_validate_parameter(parameter) != VAE_C_OP_OK ||
            parameter->parameter_key != keys[index] ||
            (index > 0 &&
             model->parameters[index - 1]->parameter_key >=
                 parameter->parameter_key)) {

            return VAE_C_OP_INVALID_STATE;
        }
        for (other = 0; other < index; other++) {
            if (model_parameters_have_exact_alias(
                    parameter,
                    model->parameters[other])) {

                return VAE_C_OP_INVALID_STATE;
            }
        }
    }
    for (index = expected_count;
         index < VAE_C_MODEL_MAX_PARAMETERS;
         index++) {

        if (model->parameters[index] != NULL) {
            return VAE_C_OP_INVALID_STATE;
        }
    }
    if (model_sum_registry(model, &total) != VAE_C_OP_OK ||
        total != model->trainable_value_count ||
        total != model->config.trainable_parameters_bn1d) {

        return VAE_C_OP_INVALID_STATE;
    }
    return VAE_C_OP_OK;
}


static VAECOpStatus model_max_pool_capacity(
    size_t max_batch_size,
    const VAECConfig *config,
    size_t *capacity
)
{
    size_t product;

    if (config == NULL || capacity == NULL || max_batch_size == 0) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (model_size_multiply(
            max_batch_size,
            config->channels[0],
            &product) != 0 ||
        model_size_multiply(
            product,
            config->pool_side,
            &product) != 0 ||
        model_size_multiply(
            product,
            config->pool_side,
            &product) != 0) {

        return VAE_C_OP_OVERFLOW;
    }
    *capacity = product;
    return VAE_C_OP_OK;
}


static VAECOpStatus model_allocate_layers(
    VAECModel *model,
    size_t encoder_fc_out,
    size_t pool_capacity
)
{
    VAECOpStatus status;
    size_t index;

    for (index = 0; index < model->config.conv_depth; index++) {
        size_t in_channels =
            index == 0
                ? VAE_C_INPUT_CHANNELS
                : model->config.channels[index - 1];
        uint64_t key_base = UINT64_C(1) +
            UINT64_C(3) * (uint64_t)index;

        status = vae_c_conv2d_allocate(
            &model->encoder_convs[index],
            in_channels,
            model->config.channels[index],
            2,
            false,
            key_base,
            UINT64_C(0)
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
        status = vae_c_batch_norm2d_allocate(
            &model->encoder_batch_norms[index],
            model->config.channels[index],
            key_base + UINT64_C(1),
            key_base + UINT64_C(2)
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
    }
    status = vae_c_max_pool2d_allocate(
        &model->encoder_max_pool,
        pool_capacity
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_dense_allocate(
        &model->encoder_fc,
        model->config.channels[model->config.conv_depth - 1],
        encoder_fc_out,
        true,
        VAE_C_MODEL_PARAM_KEY_ENCODER_FC_WEIGHT,
        VAE_C_MODEL_PARAM_KEY_ENCODER_FC_BIAS
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_dense_allocate(
        &model->decoder_fc,
        model->config.latent_dim,
        model->config.decoder_fc_features,
        false,
        VAE_C_MODEL_PARAM_KEY_DECODER_FC_WEIGHT,
        UINT64_C(0)
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_batch_norm1d_allocate(
        &model->decoder_fc_batch_norm,
        model->config.decoder_fc_features,
        VAE_C_MODEL_PARAM_KEY_DECODER_FC_BN_GAMMA,
        VAE_C_MODEL_PARAM_KEY_DECODER_FC_BN_BETA
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    for (index = 0;
         index < model->config.decoder_hidden_conv_count;
         index++) {

        size_t input_index = model->config.conv_depth - 1 - index;
        size_t output_index = input_index - 1;
        uint64_t key_base = UINT64_C(18) +
            UINT64_C(3) * (uint64_t)index;

        status = vae_c_conv2d_allocate(
            &model->decoder_hidden_convs[index],
            model->config.channels[input_index],
            model->config.channels[output_index],
            1,
            false,
            key_base,
            UINT64_C(0)
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
        status = vae_c_batch_norm2d_allocate(
            &model->decoder_hidden_batch_norms[index],
            model->config.channels[output_index],
            key_base + UINT64_C(1),
            key_base + UINT64_C(2)
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
    }
    return vae_c_conv2d_allocate(
        &model->decoder_output_conv,
        model->config.channels[0],
        VAE_C_INPUT_CHANNELS,
        1,
        true,
        VAE_C_MODEL_PARAM_KEY_DECODER_OUTPUT_WEIGHT,
        VAE_C_MODEL_PARAM_KEY_DECODER_OUTPUT_BIAS
    );
}


void vae_c_model_init(VAECModel *model)
{
    if (model != NULL) {
        memset(model, 0, sizeof(*model));
    }
}


void vae_c_model_free(VAECModel *model)
{
    size_t index;

    if (model == NULL) {
        return;
    }
    for (index = 0; index < VAE_C_MODEL_MAX_ENCODER_CONVS; index++) {
        vae_c_conv2d_free(&model->encoder_convs[index]);
        vae_c_batch_norm_free(&model->encoder_batch_norms[index]);
    }
    vae_c_max_pool2d_free(&model->encoder_max_pool);
    vae_c_dense_free(&model->encoder_fc);
    vae_c_dense_free(&model->decoder_fc);
    vae_c_batch_norm_free(&model->decoder_fc_batch_norm);
    for (index = 0;
         index < VAE_C_MODEL_MAX_DECODER_HIDDEN_CONVS;
         index++) {

        vae_c_conv2d_free(&model->decoder_hidden_convs[index]);
        vae_c_batch_norm_free(
            &model->decoder_hidden_batch_norms[index]
        );
    }
    vae_c_conv2d_free(&model->decoder_output_conv);
    vae_c_model_init(model);
}


bool vae_c_model_is_empty(const VAECModel *model)
{
    size_t index;

    if (model == NULL ||
        !model_bytes_are_zero(model->profile_id, sizeof(model->profile_id)) ||
        !model_bytes_are_zero(&model->config, sizeof(model->config)) ||
        model->max_batch_size != 0 ||
        !vae_c_max_pool2d_is_empty(&model->encoder_max_pool) ||
        !vae_c_dense_is_empty(&model->encoder_fc) ||
        !vae_c_dense_is_empty(&model->decoder_fc) ||
        !vae_c_batch_norm_is_empty(&model->decoder_fc_batch_norm) ||
        !vae_c_conv2d_is_empty(&model->decoder_output_conv) ||
        model->parameter_count != 0 ||
        model->trainable_value_count != UINT64_C(0) ||
        model->initialization_seed != UINT64_C(0) ||
        model->vae_cache_generation != UINT64_C(0) ||
        model->parameters_initialized ||
        model->allocated) {

        return false;
    }
    for (index = 0; index < VAE_C_MODEL_MAX_ENCODER_CONVS; index++) {
        if (!vae_c_conv2d_is_empty(&model->encoder_convs[index]) ||
            !vae_c_batch_norm_is_empty(
                &model->encoder_batch_norms[index])) {

            return false;
        }
    }
    for (index = 0;
         index < VAE_C_MODEL_MAX_DECODER_HIDDEN_CONVS;
         index++) {

        if (!vae_c_conv2d_is_empty(
                &model->decoder_hidden_convs[index]) ||
            !vae_c_batch_norm_is_empty(
                &model->decoder_hidden_batch_norms[index])) {

            return false;
        }
    }
    for (index = 0; index < VAE_C_MODEL_MAX_PARAMETERS; index++) {
        if (model->parameters[index] != NULL) {
            return false;
        }
    }
    return true;
}


bool vae_c_model_is_allocated(const VAECModel *model)
{
    return vae_c_model_validate(model) == VAE_C_OP_OK;
}


VAECOpStatus vae_c_model_validate(const VAECModel *model)
{
    VAECOpStatus status;
    size_t encoder_fc_out;
    size_t pool_capacity;
    size_t index;

    if (model == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (!model->allocated || model->max_batch_size == 0 ||
        (!model->parameters_initialized &&
         model->initialization_seed != UINT64_C(0)) ||
        model_validate_config(model) != VAE_C_OP_OK ||
        model_size_multiply(
            model->config.latent_dim,
            2,
            &encoder_fc_out) != 0 ||
        model_max_pool_capacity(
            model->max_batch_size,
            &model->config,
            &pool_capacity) != VAE_C_OP_OK) {

        return VAE_C_OP_INVALID_STATE;
    }
    for (index = 0; index < model->config.conv_depth; index++) {
        size_t in_channels =
            index == 0
                ? VAE_C_INPUT_CHANNELS
                : model->config.channels[index - 1];
        uint64_t key_base = UINT64_C(1) +
            UINT64_C(3) * (uint64_t)index;

        status = model_validate_conv2d(
            &model->encoder_convs[index],
            in_channels,
            model->config.channels[index],
            2,
            false,
            key_base,
            UINT64_C(0)
        );
        if (status != VAE_C_OP_OK ||
            model_validate_batch_norm(
                &model->encoder_batch_norms[index],
                VAE_C_BATCH_NORM_KIND_2D,
                model->config.channels[index],
                key_base + UINT64_C(1),
                key_base + UINT64_C(2)) != VAE_C_OP_OK) {

            return VAE_C_OP_INVALID_STATE;
        }
    }
    for (index = model->config.conv_depth;
         index < VAE_C_MODEL_MAX_ENCODER_CONVS;
         index++) {

        if (!vae_c_conv2d_is_empty(&model->encoder_convs[index]) ||
            !vae_c_batch_norm_is_empty(
                &model->encoder_batch_norms[index])) {

            return VAE_C_OP_INVALID_STATE;
        }
    }
    if (model_validate_max_pool(
            &model->encoder_max_pool,
            pool_capacity) != VAE_C_OP_OK ||
        model_validate_dense(
            &model->encoder_fc,
            model->config.channels[model->config.conv_depth - 1],
            encoder_fc_out,
            true,
            VAE_C_MODEL_PARAM_KEY_ENCODER_FC_WEIGHT,
            VAE_C_MODEL_PARAM_KEY_ENCODER_FC_BIAS) != VAE_C_OP_OK ||
        model_validate_dense(
            &model->decoder_fc,
            model->config.latent_dim,
            model->config.decoder_fc_features,
            false,
            VAE_C_MODEL_PARAM_KEY_DECODER_FC_WEIGHT,
            UINT64_C(0)) != VAE_C_OP_OK ||
        model_validate_batch_norm(
            &model->decoder_fc_batch_norm,
            VAE_C_BATCH_NORM_KIND_1D,
            model->config.decoder_fc_features,
            VAE_C_MODEL_PARAM_KEY_DECODER_FC_BN_GAMMA,
            VAE_C_MODEL_PARAM_KEY_DECODER_FC_BN_BETA) != VAE_C_OP_OK) {

        return VAE_C_OP_INVALID_STATE;
    }
    for (index = 0;
         index < model->config.decoder_hidden_conv_count;
         index++) {

        size_t input_index = model->config.conv_depth - 1 - index;
        size_t output_index = input_index - 1;
        uint64_t key_base = UINT64_C(18) +
            UINT64_C(3) * (uint64_t)index;

        if (model_validate_conv2d(
                &model->decoder_hidden_convs[index],
                model->config.channels[input_index],
                model->config.channels[output_index],
                1,
                false,
                key_base,
                UINT64_C(0)) != VAE_C_OP_OK ||
            model_validate_batch_norm(
                &model->decoder_hidden_batch_norms[index],
                VAE_C_BATCH_NORM_KIND_2D,
                model->config.channels[output_index],
                key_base + UINT64_C(1),
                key_base + UINT64_C(2)) != VAE_C_OP_OK) {

            return VAE_C_OP_INVALID_STATE;
        }
    }
    for (index = model->config.decoder_hidden_conv_count;
         index < VAE_C_MODEL_MAX_DECODER_HIDDEN_CONVS;
         index++) {

        if (!vae_c_conv2d_is_empty(
                &model->decoder_hidden_convs[index]) ||
            !vae_c_batch_norm_is_empty(
                &model->decoder_hidden_batch_norms[index])) {

            return VAE_C_OP_INVALID_STATE;
        }
    }
    if (model_validate_conv2d(
            &model->decoder_output_conv,
            model->config.channels[0],
            VAE_C_INPUT_CHANNELS,
            1,
            true,
            VAE_C_MODEL_PARAM_KEY_DECODER_OUTPUT_WEIGHT,
            VAE_C_MODEL_PARAM_KEY_DECODER_OUTPUT_BIAS) != VAE_C_OP_OK ||
        model_validate_registry(model) != VAE_C_OP_OK) {

        return VAE_C_OP_INVALID_STATE;
    }
    return VAE_C_OP_OK;
}


VAECOpStatus vae_c_model_allocate(
    VAECModel *model,
    const VAECProfile *profile,
    size_t max_batch_size
)
{
    VAECModel candidate;
    VAECConfig config;
    VAECConfigStatus config_status;
    VAECOpStatus status;
    uint64_t trainable_total;
    const char *profile_terminator;
    size_t profile_id_length;
    size_t encoder_fc_out;
    size_t pool_capacity;
    size_t expected_parameter_count;

    if (model == NULL || profile == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (!vae_c_model_is_empty(model)) {
        return VAE_C_OP_OBJECT_NOT_EMPTY;
    }
    if (max_batch_size == 0) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (profile->latent_dim > SIZE_MAX / 2) {
        return VAE_C_OP_OVERFLOW;
    }
    config_status = vae_c_config_build(profile, &config, NULL, 0);
    if (config_status != VAE_C_CONFIG_OK) {
        return model_map_config_status(config_status);
    }
    profile_terminator = (const char *)memchr(
        profile->profile_id,
        '\0',
        VAE_C_PROFILE_TEXT_CAP
    );
    if (profile_terminator == NULL || profile->profile_id[0] == '\0') {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    profile_id_length =
        (size_t)(profile_terminator - profile->profile_id);
    if (model_size_multiply(config.latent_dim, 2, &encoder_fc_out) != 0 ||
        model_max_pool_capacity(
            max_batch_size,
            &config,
            &pool_capacity) != VAE_C_OP_OK ||
        model_expected_parameter_count(
            config.conv_depth,
            &expected_parameter_count) != VAE_C_OP_OK ||
        expected_parameter_count > VAE_C_MODEL_MAX_PARAMETERS) {

        return VAE_C_OP_OVERFLOW;
    }

    vae_c_model_init(&candidate);
    memcpy(
        candidate.profile_id,
        profile->profile_id,
        profile_id_length + 1
    );
    candidate.config = config;
    candidate.max_batch_size = max_batch_size;

    status = model_allocate_layers(
        &candidate,
        encoder_fc_out,
        pool_capacity
    );
    if (status != VAE_C_OP_OK) {
        vae_c_model_free(&candidate);
        return status;
    }
    model_rebind_registry(&candidate);
    status = model_sum_registry(&candidate, &trainable_total);
    if (status != VAE_C_OP_OK ||
        candidate.parameter_count != expected_parameter_count ||
        trainable_total != config.trainable_parameters_bn1d) {

        vae_c_model_free(&candidate);
        return
            status != VAE_C_OP_OK
                ? status
                : VAE_C_OP_INVALID_STATE;
    }
    candidate.trainable_value_count = trainable_total;
    candidate.allocated = true;
    status = vae_c_model_validate(&candidate);
    if (status != VAE_C_OP_OK) {
        vae_c_model_free(&candidate);
        return status;
    }

    *model = candidate;
    model_rebind_registry(model);
    vae_c_model_init(&candidate);
    return VAE_C_OP_OK;
}
