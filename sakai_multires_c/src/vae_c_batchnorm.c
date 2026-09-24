#include "vae_c_batchnorm.h"

#include "vae_c_init.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>


static int vae_c_batch_norm_size_multiply(
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


static VAECOpStatus vae_c_batch_norm_map_storage_status(
    VAECStorageStatus status
)
{
    switch (status) {
        case VAE_C_STORAGE_OK:
            return VAE_C_OP_OK;
        case VAE_C_STORAGE_INVALID_ARGUMENT:
            return VAE_C_OP_INVALID_ARGUMENT;
        case VAE_C_STORAGE_OBJECT_NOT_EMPTY:
            return VAE_C_OP_OBJECT_NOT_EMPTY;
        case VAE_C_STORAGE_OVERFLOW:
            return VAE_C_OP_OVERFLOW;
        case VAE_C_STORAGE_ALLOCATION_FAILED:
            return VAE_C_OP_ALLOCATION_FAILED;
        default:
            return VAE_C_OP_INVALID_STATE;
    }
}


static bool vae_c_batch_norm_shape_is_zero_initialized(
    const VAECShape *shape
)
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


static VAECOpStatus vae_c_batch_norm_validate_shape(
    const VAECShape *shape
)
{
    VAECShape canonical;
    VAECStorageStatus storage_status;
    size_t index;

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
    if (shape->count != canonical.count) {
        return VAE_C_OP_INVALID_STATE;
    }
    for (index = 0; index < shape->rank; index++) {
        if (shape->dims[index] != canonical.dims[index] ||
            shape->strides[index] != canonical.strides[index]) {

            return VAE_C_OP_INVALID_STATE;
        }
    }
    for (index = shape->rank;
         index < VAE_C_TENSOR_MAX_RANK;
         index++) {

        if (shape->dims[index] != 0 || shape->strides[index] != 0) {
            return VAE_C_OP_INVALID_STATE;
        }
    }
    return VAE_C_OP_OK;
}


static bool vae_c_batch_norm_all_finite(
    const float *values,
    size_t count
)
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


static VAECOpStatus vae_c_batch_norm_validate_tensor(
    const VAECTensor *tensor
)
{
    VAECOpStatus status;
    size_t array_count;
    size_t total_elements;
    size_t expected_bytes;

    if (tensor == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    status = vae_c_batch_norm_validate_shape(&tensor->shape);
    if (status != VAE_C_OP_OK) {
        return status;
    }
    if (tensor->data == NULL) {
        return VAE_C_OP_INVALID_STATE;
    }
    if (tensor->has_grad) {
        if (tensor->grad == NULL || tensor->grad == tensor->data) {
            return VAE_C_OP_INVALID_STATE;
        }
    } else if (tensor->grad != NULL) {
        return VAE_C_OP_INVALID_STATE;
    }
    if (!tensor->owns_memory) {
        return
            tensor->owned_block == NULL && tensor->owned_bytes == 0
                ? VAE_C_OP_OK
                : VAE_C_OP_INVALID_STATE;
    }
    if (tensor->owned_block == NULL ||
        tensor->data != tensor->owned_block) {

        return VAE_C_OP_INVALID_STATE;
    }
    array_count = tensor->has_grad ? 2 : 1;
    if (vae_c_batch_norm_size_multiply(
            tensor->shape.count,
            array_count,
            &total_elements) != 0 ||
        vae_c_batch_norm_size_multiply(
            total_elements,
            sizeof(float),
            &expected_bytes) != 0 ||
        tensor->owned_bytes != expected_bytes) {

        return VAE_C_OP_INVALID_STATE;
    }
    if (tensor->has_grad &&
        tensor->grad != tensor->data + tensor->shape.count) {

        return VAE_C_OP_INVALID_STATE;
    }
    return VAE_C_OP_OK;
}


static VAECOpStatus vae_c_batch_norm_validate_parameter(
    const VAECParameter *parameter,
    size_t num_features
)
{
    VAECOpStatus status;
    size_t total_elements;
    size_t expected_bytes;

    if (parameter == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    status = vae_c_batch_norm_validate_shape(&parameter->shape);
    if (status != VAE_C_OP_OK ||
        parameter->shape.rank != 1 ||
        parameter->shape.dims[0] != num_features ||
        parameter->shape.count != num_features) {

        return VAE_C_OP_INVALID_STATE;
    }
    if (parameter->owned_block == NULL ||
        parameter->values != parameter->owned_block ||
        parameter->gradients != parameter->values + num_features ||
        parameter->adam_m != parameter->gradients + num_features ||
        parameter->adam_v != parameter->adam_m + num_features) {

        return VAE_C_OP_INVALID_STATE;
    }
    if (vae_c_batch_norm_size_multiply(
            num_features,
            4,
            &total_elements) != 0 ||
        vae_c_batch_norm_size_multiply(
            total_elements,
            sizeof(float),
            &expected_bytes) != 0 ||
        parameter->owned_bytes != expected_bytes) {

        return VAE_C_OP_INVALID_STATE;
    }
    if (!vae_c_batch_norm_all_finite(parameter->values, num_features) ||
        !vae_c_batch_norm_all_finite(
            parameter->gradients,
            num_features) ||
        !vae_c_batch_norm_all_finite(parameter->adam_m, num_features) ||
        !vae_c_batch_norm_all_finite(parameter->adam_v, num_features)) {

        return VAE_C_OP_INVALID_STATE;
    }
    return VAE_C_OP_OK;
}


static bool vae_c_batch_norm_internal_buffers_are_unique(
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


static VAECOpStatus vae_c_batch_norm_sample_count_from_shape(
    VAECBatchNormKind kind,
    size_t num_features,
    const VAECShape *shape,
    size_t *sample_count,
    size_t *spatial_count
)
{
    size_t spatial;
    size_t samples;

    if (shape == NULL || sample_count == NULL || spatial_count == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (kind == VAE_C_BATCH_NORM_KIND_1D) {
        if (shape->rank != 2 || shape->dims[1] != num_features) {
            return VAE_C_OP_SHAPE_MISMATCH;
        }
        *sample_count = shape->dims[0];
        *spatial_count = 1;
        return VAE_C_OP_OK;
    }
    if (kind != VAE_C_BATCH_NORM_KIND_2D ||
        shape->rank != 4 ||
        shape->dims[1] != num_features) {

        return VAE_C_OP_SHAPE_MISMATCH;
    }
    if (vae_c_batch_norm_size_multiply(
            shape->dims[2],
            shape->dims[3],
            &spatial) != 0 ||
        vae_c_batch_norm_size_multiply(
            shape->dims[0],
            spatial,
            &samples) != 0) {

        return VAE_C_OP_OVERFLOW;
    }
    *sample_count = samples;
    *spatial_count = spatial;
    return VAE_C_OP_OK;
}


static VAECOpStatus vae_c_batch_norm_validate_layer(
    const VAECBatchNorm *layer
)
{
    VAECOpStatus status;
    size_t total_elements;
    size_t expected_bytes;
    size_t expected_samples;
    size_t spatial_count;
    size_t feature;

    if (layer == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if ((layer->kind != VAE_C_BATCH_NORM_KIND_1D &&
         layer->kind != VAE_C_BATCH_NORM_KIND_2D) ||
        layer->num_features == 0 ||
        !isfinite(layer->epsilon) ||
        layer->epsilon <= 0.0f ||
        !isfinite(layer->momentum) ||
        layer->momentum != VAE_C_BATCH_NORM_MOMENTUM) {

        return VAE_C_OP_INVALID_STATE;
    }
    status = vae_c_batch_norm_validate_parameter(
        &layer->gamma,
        layer->num_features
    );
    if (status != VAE_C_OP_OK) {
        return VAE_C_OP_INVALID_STATE;
    }
    status = vae_c_batch_norm_validate_parameter(
        &layer->beta,
        layer->num_features
    );
    if (status != VAE_C_OP_OK) {
        return VAE_C_OP_INVALID_STATE;
    }
    if (layer->owned_block == NULL ||
        vae_c_batch_norm_size_multiply(
            layer->num_features,
            7,
            &total_elements) != 0 ||
        vae_c_batch_norm_size_multiply(
            total_elements,
            sizeof(float),
            &expected_bytes) != 0 ||
        layer->owned_bytes != expected_bytes ||
        layer->running_mean != layer->owned_block ||
        layer->running_variance !=
            layer->running_mean + layer->num_features ||
        layer->batch_mean !=
            layer->running_variance + layer->num_features ||
        layer->batch_inv_std !=
            layer->batch_mean + layer->num_features ||
        layer->work_mean !=
            layer->batch_inv_std + layer->num_features ||
        layer->work_inv_std !=
            layer->work_mean + layer->num_features ||
        layer->work_unbiased_variance !=
            layer->work_inv_std + layer->num_features ||
        !vae_c_batch_norm_internal_buffers_are_unique(layer)) {

        return VAE_C_OP_INVALID_STATE;
    }
    for (feature = 0; feature < layer->num_features; feature++) {
        if (!isfinite(layer->running_mean[feature]) ||
            !isfinite(layer->running_variance[feature]) ||
            layer->running_variance[feature] < 0.0f) {

            return VAE_C_OP_INVALID_STATE;
        }
    }
    if (!layer->cache_valid) {
        return
            layer->last_sample_count == 0 &&
            vae_c_batch_norm_shape_is_zero_initialized(
                &layer->last_input_shape) &&
            vae_c_batch_norm_shape_is_zero_initialized(
                &layer->last_output_shape)
                ? VAE_C_OP_OK
                : VAE_C_OP_INVALID_STATE;
    }
    if (layer->last_sample_count < 2 ||
        vae_c_batch_norm_validate_shape(&layer->last_input_shape) !=
            VAE_C_OP_OK ||
        vae_c_batch_norm_validate_shape(&layer->last_output_shape) !=
            VAE_C_OP_OK ||
        !vae_c_shape_equal(
            &layer->last_input_shape,
            &layer->last_output_shape)) {

        return VAE_C_OP_INVALID_STATE;
    }
    status = vae_c_batch_norm_sample_count_from_shape(
        layer->kind,
        layer->num_features,
        &layer->last_input_shape,
        &expected_samples,
        &spatial_count
    );
    if (status != VAE_C_OP_OK ||
        expected_samples != layer->last_sample_count) {

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


void vae_c_batch_norm_init(VAECBatchNorm *layer)
{
    if (layer != NULL) {
        memset(layer, 0, sizeof(*layer));
    }
}


void vae_c_batch_norm_free(VAECBatchNorm *layer)
{
    if (layer == NULL) {
        return;
    }
    vae_c_parameter_free(&layer->gamma);
    vae_c_parameter_free(&layer->beta);
    free(layer->owned_block);
    vae_c_batch_norm_init(layer);
}


bool vae_c_batch_norm_is_empty(const VAECBatchNorm *layer)
{
    return
        layer != NULL &&
        layer->kind == 0 &&
        layer->num_features == 0 &&
        layer->epsilon == 0.0f &&
        layer->momentum == 0.0f &&
        vae_c_parameter_is_empty(&layer->gamma) &&
        vae_c_parameter_is_empty(&layer->beta) &&
        layer->running_mean == NULL &&
        layer->running_variance == NULL &&
        layer->batch_mean == NULL &&
        layer->batch_inv_std == NULL &&
        layer->work_mean == NULL &&
        layer->work_inv_std == NULL &&
        layer->work_unbiased_variance == NULL &&
        layer->owned_block == NULL &&
        layer->owned_bytes == 0 &&
        layer->last_sample_count == 0 &&
        vae_c_batch_norm_shape_is_zero_initialized(
            &layer->last_input_shape) &&
        vae_c_batch_norm_shape_is_zero_initialized(
            &layer->last_output_shape) &&
        !layer->cache_valid;
}


static VAECOpStatus vae_c_batch_norm_allocate(
    VAECBatchNorm *layer,
    VAECBatchNormKind kind,
    size_t num_features,
    uint64_t gamma_parameter_key,
    uint64_t beta_parameter_key
)
{
    VAECBatchNorm candidate;
    VAECShape parameter_shape;
    VAECStorageStatus storage_status;
    VAECInitStatus init_status;
    size_t total_elements;
    size_t total_bytes;
    float *block;

    if (layer == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (!vae_c_batch_norm_is_empty(layer)) {
        return VAE_C_OP_OBJECT_NOT_EMPTY;
    }
    if (num_features == 0) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (vae_c_batch_norm_size_multiply(
            num_features,
            7,
            &total_elements) != 0 ||
        vae_c_batch_norm_size_multiply(
            total_elements,
            sizeof(float),
            &total_bytes) != 0) {

        return VAE_C_OP_OVERFLOW;
    }
    storage_status = vae_c_shape_make(
        1,
        &num_features,
        &parameter_shape
    );
    if (storage_status != VAE_C_STORAGE_OK) {
        return vae_c_batch_norm_map_storage_status(storage_status);
    }
    vae_c_batch_norm_init(&candidate);
    storage_status = vae_c_parameter_allocate(
        &candidate.gamma,
        &parameter_shape,
        gamma_parameter_key
    );
    if (storage_status != VAE_C_STORAGE_OK) {
        vae_c_batch_norm_free(&candidate);
        return vae_c_batch_norm_map_storage_status(storage_status);
    }
    storage_status = vae_c_parameter_allocate(
        &candidate.beta,
        &parameter_shape,
        beta_parameter_key
    );
    if (storage_status != VAE_C_STORAGE_OK) {
        vae_c_batch_norm_free(&candidate);
        return vae_c_batch_norm_map_storage_status(storage_status);
    }
    block = (float *)calloc(total_elements, sizeof(float));
    if (block == NULL) {
        vae_c_batch_norm_free(&candidate);
        return VAE_C_OP_ALLOCATION_FAILED;
    }
    candidate.kind = kind;
    candidate.num_features = num_features;
    candidate.epsilon = VAE_C_BATCH_NORM_EPSILON;
    candidate.momentum = VAE_C_BATCH_NORM_MOMENTUM;
    candidate.owned_block = block;
    candidate.owned_bytes = total_bytes;
    candidate.running_mean = block;
    candidate.running_variance = block + num_features;
    candidate.batch_mean = block + 2 * num_features;
    candidate.batch_inv_std = block + 3 * num_features;
    candidate.work_mean = block + 4 * num_features;
    candidate.work_inv_std = block + 5 * num_features;
    candidate.work_unbiased_variance = block + 6 * num_features;

    init_status = vae_c_init_batchnorm_f32(
        candidate.gamma.values,
        candidate.beta.values,
        candidate.running_mean,
        candidate.running_variance,
        num_features
    );
    if (init_status != VAE_C_INIT_OK) {
        vae_c_batch_norm_free(&candidate);
        return VAE_C_OP_INVALID_STATE;
    }
    *layer = candidate;
    return VAE_C_OP_OK;
}


VAECOpStatus vae_c_batch_norm1d_allocate(
    VAECBatchNorm *layer,
    size_t num_features,
    uint64_t gamma_parameter_key,
    uint64_t beta_parameter_key
)
{
    return vae_c_batch_norm_allocate(
        layer,
        VAE_C_BATCH_NORM_KIND_1D,
        num_features,
        gamma_parameter_key,
        beta_parameter_key
    );
}


VAECOpStatus vae_c_batch_norm2d_allocate(
    VAECBatchNorm *layer,
    size_t num_channels,
    uint64_t gamma_parameter_key,
    uint64_t beta_parameter_key
)
{
    return vae_c_batch_norm_allocate(
        layer,
        VAE_C_BATCH_NORM_KIND_2D,
        num_channels,
        gamma_parameter_key,
        beta_parameter_key
    );
}


VAECOpStatus vae_c_batch_norm_zero_grad(VAECBatchNorm *layer)
{
    VAECStorageStatus status;

    if (layer == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (vae_c_batch_norm_is_empty(layer)) {
        return VAE_C_OP_OK;
    }
    if (vae_c_batch_norm_validate_layer(layer) != VAE_C_OP_OK) {
        return VAE_C_OP_INVALID_STATE;
    }
    status = vae_c_parameter_zero_grad(&layer->gamma);
    if (status != VAE_C_STORAGE_OK) {
        return VAE_C_OP_INVALID_STATE;
    }
    status = vae_c_parameter_zero_grad(&layer->beta);
    return
        status == VAE_C_STORAGE_OK
            ? VAE_C_OP_OK
            : VAE_C_OP_INVALID_STATE;
}


static bool vae_c_batch_norm_tensors_have_exact_alias(
    const VAECTensor *input,
    const VAECTensor *output
)
{
    if (input == output || input->data == output->data) {
        return true;
    }
    if (input->grad != NULL &&
        (input->grad == output->data ||
         input->grad == output->grad)) {

        return true;
    }
    return
        output->grad != NULL && input->data == output->grad;
}


static bool vae_c_batch_norm_tensors_alias_layer(
    const VAECBatchNorm *layer,
    const VAECTensor *input,
    const VAECTensor *output
)
{
    const void *tensor_buffers[4];
    const void *layer_buffers[15];
    size_t tensor_index;
    size_t layer_index;

    tensor_buffers[0] = input->data;
    tensor_buffers[1] = input->grad;
    tensor_buffers[2] = output->data;
    tensor_buffers[3] = output->grad;
    layer_buffers[0] = layer->gamma.values;
    layer_buffers[1] = layer->gamma.gradients;
    layer_buffers[2] = layer->gamma.adam_m;
    layer_buffers[3] = layer->gamma.adam_v;
    layer_buffers[4] = layer->beta.values;
    layer_buffers[5] = layer->beta.gradients;
    layer_buffers[6] = layer->beta.adam_m;
    layer_buffers[7] = layer->beta.adam_v;
    layer_buffers[8] = layer->running_mean;
    layer_buffers[9] = layer->running_variance;
    layer_buffers[10] = layer->batch_mean;
    layer_buffers[11] = layer->batch_inv_std;
    layer_buffers[12] = layer->work_mean;
    layer_buffers[13] = layer->work_inv_std;
    layer_buffers[14] = layer->work_unbiased_variance;

    for (tensor_index = 0; tensor_index < 4; tensor_index++) {
        if (tensor_buffers[tensor_index] == NULL) {
            continue;
        }
        for (layer_index = 0; layer_index < 15; layer_index++) {
            if (tensor_buffers[tensor_index] == layer_buffers[layer_index]) {
                return true;
            }
        }
    }
    return false;
}


static VAECOpStatus vae_c_batch_norm_validate_pair(
    const VAECTensor *input,
    const VAECTensor *output,
    bool require_gradients
)
{
    VAECOpStatus status;

    if (input == NULL || output == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    status = vae_c_batch_norm_validate_tensor(input);
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_batch_norm_validate_tensor(output);
    if (status != VAE_C_OP_OK) {
        return status;
    }
    if (vae_c_batch_norm_tensors_have_exact_alias(input, output)) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (require_gradients &&
        (!input->has_grad || !output->has_grad)) {

        return VAE_C_OP_GRADIENT_NOT_AVAILABLE;
    }
    return VAE_C_OP_OK;
}


static size_t vae_c_batch_norm_element_index(
    VAECBatchNormKind kind,
    const VAECShape *shape,
    size_t spatial_count,
    size_t feature,
    size_t sample
)
{
    if (kind == VAE_C_BATCH_NORM_KIND_1D) {
        return sample * shape->strides[0] + feature;
    }
    return
        (sample / spatial_count) * shape->strides[0] +
        feature * shape->strides[1] +
        sample % spatial_count;
}


static VAECOpStatus vae_c_batch_norm_validate_forward(
    const VAECBatchNorm *layer,
    VAECBatchNormKind expected_kind,
    VAECBatchNormMode mode,
    const VAECTensor *input,
    const VAECTensor *output,
    size_t *sample_count,
    size_t *spatial_count
)
{
    VAECOpStatus status;

    status = vae_c_batch_norm_validate_layer(layer);
    if (status != VAE_C_OP_OK) {
        return status;
    }
    if (layer->kind != expected_kind) {
        return VAE_C_OP_INVALID_STATE;
    }
    if (mode != VAE_C_BATCH_NORM_MODE_TRAIN &&
        mode != VAE_C_BATCH_NORM_MODE_EVAL) {

        return VAE_C_OP_INVALID_ARGUMENT;
    }
    status = vae_c_batch_norm_validate_pair(input, output, false);
    if (status != VAE_C_OP_OK) {
        return status;
    }
    if (vae_c_batch_norm_tensors_alias_layer(layer, input, output)) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (!vae_c_shape_equal(&input->shape, &output->shape)) {
        return VAE_C_OP_SHAPE_MISMATCH;
    }
    status = vae_c_batch_norm_sample_count_from_shape(
        expected_kind,
        layer->num_features,
        &input->shape,
        sample_count,
        spatial_count
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    if (mode == VAE_C_BATCH_NORM_MODE_TRAIN && *sample_count < 2) {
        return VAE_C_OP_SHAPE_MISMATCH;
    }
    if (!vae_c_batch_norm_all_finite(
            input->data,
            input->shape.count)) {

        return VAE_C_OP_INVALID_STATE;
    }
    return VAE_C_OP_OK;
}


static VAECOpStatus vae_c_batch_norm_forward(
    VAECBatchNorm *layer,
    VAECBatchNormKind expected_kind,
    VAECBatchNormMode mode,
    const VAECTensor *input,
    VAECTensor *output
)
{
    VAECOpStatus status;
    size_t sample_count;
    size_t spatial_count;
    size_t feature;
    size_t sample;
    float sample_count_f;

    status = vae_c_batch_norm_validate_forward(
        layer,
        expected_kind,
        mode,
        input,
        output,
        &sample_count,
        &spatial_count
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    sample_count_f = (float)sample_count;

    if (mode == VAE_C_BATCH_NORM_MODE_TRAIN) {
        for (feature = 0; feature < layer->num_features; feature++) {
            float sum = 0.0f;
            float mean;
            float sum_squared_deviation = 0.0f;
            float variance_biased;
            float variance_unbiased;
            float variance_with_epsilon;
            float inverse_std;
            float running_mean_new;
            float running_variance_new;

            for (sample = 0; sample < sample_count; sample++) {
                size_t index = vae_c_batch_norm_element_index(
                    expected_kind,
                    &input->shape,
                    spatial_count,
                    feature,
                    sample
                );
                sum += input->data[index];
            }
            mean = sum / sample_count_f;
            if (!isfinite(mean)) {
                return VAE_C_OP_INVALID_STATE;
            }
            for (sample = 0; sample < sample_count; sample++) {
                size_t index = vae_c_batch_norm_element_index(
                    expected_kind,
                    &input->shape,
                    spatial_count,
                    feature,
                    sample
                );
                float deviation = input->data[index] - mean;
                float squared_deviation = deviation * deviation;

                if (!isfinite(squared_deviation)) {
                    return VAE_C_OP_INVALID_STATE;
                }
                sum_squared_deviation += squared_deviation;
            }
            if (!isfinite(sum_squared_deviation) ||
                sum_squared_deviation < 0.0f) {

                return VAE_C_OP_INVALID_STATE;
            }
            variance_biased = sum_squared_deviation / sample_count_f;
            variance_unbiased =
                sum_squared_deviation / (float)(sample_count - 1);
            variance_with_epsilon = variance_biased + layer->epsilon;
            if (!isfinite(variance_biased) || variance_biased < 0.0f ||
                !isfinite(variance_unbiased) || variance_unbiased < 0.0f ||
                !isfinite(variance_with_epsilon) ||
                variance_with_epsilon <= 0.0f) {

                return VAE_C_OP_INVALID_STATE;
            }
            inverse_std = 1.0f / sqrtf(variance_with_epsilon);
            running_mean_new =
                (1.0f - layer->momentum) * layer->running_mean[feature] +
                layer->momentum * mean;
            running_variance_new =
                (1.0f - layer->momentum) *
                    layer->running_variance[feature] +
                layer->momentum * variance_unbiased;
            if (!isfinite(inverse_std) || inverse_std <= 0.0f ||
                !isfinite(running_mean_new) ||
                !isfinite(running_variance_new) ||
                running_variance_new < 0.0f) {

                return VAE_C_OP_INVALID_STATE;
            }
            layer->work_mean[feature] = mean;
            layer->work_inv_std[feature] = inverse_std;
            layer->work_unbiased_variance[feature] = variance_unbiased;
        }
        for (feature = 0; feature < layer->num_features; feature++) {
            for (sample = 0; sample < sample_count; sample++) {
                size_t index = vae_c_batch_norm_element_index(
                    expected_kind,
                    &input->shape,
                    spatial_count,
                    feature,
                    sample
                );
                float normalized =
                    (input->data[index] - layer->work_mean[feature]) *
                    layer->work_inv_std[feature];
                float result =
                    layer->gamma.values[feature] * normalized +
                    layer->beta.values[feature];

                if (!isfinite(normalized) || !isfinite(result)) {
                    return VAE_C_OP_INVALID_STATE;
                }
            }
        }
        for (feature = 0; feature < layer->num_features; feature++) {
            for (sample = 0; sample < sample_count; sample++) {
                size_t index = vae_c_batch_norm_element_index(
                    expected_kind,
                    &input->shape,
                    spatial_count,
                    feature,
                    sample
                );
                float normalized =
                    (input->data[index] - layer->work_mean[feature]) *
                    layer->work_inv_std[feature];

                output->data[index] =
                    layer->gamma.values[feature] * normalized +
                    layer->beta.values[feature];
            }
            layer->running_mean[feature] =
                (1.0f - layer->momentum) * layer->running_mean[feature] +
                layer->momentum * layer->work_mean[feature];
            layer->running_variance[feature] =
                (1.0f - layer->momentum) *
                    layer->running_variance[feature] +
                layer->momentum *
                    layer->work_unbiased_variance[feature];
            layer->batch_mean[feature] = layer->work_mean[feature];
            layer->batch_inv_std[feature] = layer->work_inv_std[feature];
        }
        layer->last_sample_count = sample_count;
        layer->last_input_shape = input->shape;
        layer->last_output_shape = output->shape;
        layer->cache_valid = true;
        return VAE_C_OP_OK;
    }

    for (feature = 0; feature < layer->num_features; feature++) {
        float variance_with_epsilon =
            layer->running_variance[feature] + layer->epsilon;
        float inverse_std;

        if (!isfinite(variance_with_epsilon) ||
            variance_with_epsilon <= 0.0f) {

            return VAE_C_OP_INVALID_STATE;
        }
        inverse_std = 1.0f / sqrtf(variance_with_epsilon);
        if (!isfinite(inverse_std) || inverse_std <= 0.0f) {
            return VAE_C_OP_INVALID_STATE;
        }
        layer->work_inv_std[feature] = inverse_std;
    }
    for (feature = 0; feature < layer->num_features; feature++) {
        for (sample = 0; sample < sample_count; sample++) {
            size_t index = vae_c_batch_norm_element_index(
                expected_kind,
                &input->shape,
                spatial_count,
                feature,
                sample
            );
            float normalized =
                (input->data[index] - layer->running_mean[feature]) *
                layer->work_inv_std[feature];
            float result =
                layer->gamma.values[feature] * normalized +
                layer->beta.values[feature];

            if (!isfinite(normalized) || !isfinite(result)) {
                return VAE_C_OP_INVALID_STATE;
            }
        }
    }
    for (feature = 0; feature < layer->num_features; feature++) {
        for (sample = 0; sample < sample_count; sample++) {
            size_t index = vae_c_batch_norm_element_index(
                expected_kind,
                &input->shape,
                spatial_count,
                feature,
                sample
            );
            float normalized =
                (input->data[index] - layer->running_mean[feature]) *
                layer->work_inv_std[feature];

            output->data[index] =
                layer->gamma.values[feature] * normalized +
                layer->beta.values[feature];
        }
    }
    layer->last_sample_count = 0;
    vae_c_shape_init(&layer->last_input_shape);
    vae_c_shape_init(&layer->last_output_shape);
    layer->cache_valid = false;
    return VAE_C_OP_OK;
}


static VAECOpStatus vae_c_batch_norm_validate_backward(
    const VAECBatchNorm *layer,
    VAECBatchNormKind expected_kind,
    const VAECTensor *input,
    const VAECTensor *output,
    size_t *sample_count,
    size_t *spatial_count
)
{
    VAECOpStatus status;

    status = vae_c_batch_norm_validate_layer(layer);
    if (status != VAE_C_OP_OK) {
        return status;
    }
    if (layer->kind != expected_kind || !layer->cache_valid) {
        return VAE_C_OP_INVALID_STATE;
    }
    status = vae_c_batch_norm_validate_pair(input, output, true);
    if (status != VAE_C_OP_OK) {
        return status;
    }
    if (vae_c_batch_norm_tensors_alias_layer(layer, input, output)) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (!vae_c_shape_equal(&input->shape, &layer->last_input_shape) ||
        !vae_c_shape_equal(&output->shape, &layer->last_output_shape)) {

        return VAE_C_OP_INVALID_STATE;
    }
    status = vae_c_batch_norm_sample_count_from_shape(
        expected_kind,
        layer->num_features,
        &input->shape,
        sample_count,
        spatial_count
    );
    if (status != VAE_C_OP_OK ||
        *sample_count != layer->last_sample_count) {

        return VAE_C_OP_INVALID_STATE;
    }
    if (!vae_c_batch_norm_all_finite(input->data, input->shape.count) ||
        !vae_c_batch_norm_all_finite(input->grad, input->shape.count) ||
        !vae_c_batch_norm_all_finite(output->grad, output->shape.count)) {

        return VAE_C_OP_INVALID_STATE;
    }
    return VAE_C_OP_OK;
}


static VAECOpStatus vae_c_batch_norm_backward(
    VAECBatchNorm *layer,
    VAECBatchNormKind expected_kind,
    const VAECTensor *input,
    const VAECTensor *output
)
{
    VAECOpStatus status;
    size_t sample_count;
    size_t spatial_count;
    size_t feature;
    size_t sample;
    float sample_count_f;

    status = vae_c_batch_norm_validate_backward(
        layer,
        expected_kind,
        input,
        output,
        &sample_count,
        &spatial_count
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    sample_count_f = (float)sample_count;

    for (feature = 0; feature < layer->num_features; feature++) {
        float sum_dy = 0.0f;
        float sum_dy_xhat = 0.0f;

        for (sample = 0; sample < sample_count; sample++) {
            size_t index = vae_c_batch_norm_element_index(
                expected_kind,
                &input->shape,
                spatial_count,
                feature,
                sample
            );
            float xhat =
                (input->data[index] - layer->batch_mean[feature]) *
                layer->batch_inv_std[feature];
            float product = output->grad[index] * xhat;

            if (!isfinite(xhat) || !isfinite(product)) {
                return VAE_C_OP_INVALID_STATE;
            }
            sum_dy += output->grad[index];
            sum_dy_xhat += product;
        }
        if (!isfinite(sum_dy) || !isfinite(sum_dy_xhat)) {
            return VAE_C_OP_INVALID_STATE;
        }
        layer->work_mean[feature] = sum_dy;
        layer->work_inv_std[feature] = sum_dy_xhat;
    }

    for (feature = 0; feature < layer->num_features; feature++) {
        float gamma_gradient_new =
            layer->gamma.gradients[feature] +
            layer->work_inv_std[feature];
        float beta_gradient_new =
            layer->beta.gradients[feature] +
            layer->work_mean[feature];
        float coefficient =
            layer->gamma.values[feature] *
            layer->batch_inv_std[feature] / sample_count_f;

        if (!isfinite(gamma_gradient_new) ||
            !isfinite(beta_gradient_new) ||
            !isfinite(coefficient)) {

            return VAE_C_OP_INVALID_STATE;
        }
        for (sample = 0; sample < sample_count; sample++) {
            size_t index = vae_c_batch_norm_element_index(
                expected_kind,
                &input->shape,
                spatial_count,
                feature,
                sample
            );
            float xhat =
                (input->data[index] - layer->batch_mean[feature]) *
                layer->batch_inv_std[feature];
            float centered_gradient =
                sample_count_f * output->grad[index] -
                layer->work_mean[feature] -
                xhat * layer->work_inv_std[feature];
            float contribution = coefficient * centered_gradient;
            float accumulated = input->grad[index] + contribution;

            if (!isfinite(xhat) || !isfinite(centered_gradient) ||
                !isfinite(contribution) || !isfinite(accumulated)) {

                return VAE_C_OP_INVALID_STATE;
            }
        }
    }

    for (feature = 0; feature < layer->num_features; feature++) {
        float coefficient =
            layer->gamma.values[feature] *
            layer->batch_inv_std[feature] / sample_count_f;

        layer->gamma.gradients[feature] += layer->work_inv_std[feature];
        layer->beta.gradients[feature] += layer->work_mean[feature];
        for (sample = 0; sample < sample_count; sample++) {
            size_t index = vae_c_batch_norm_element_index(
                expected_kind,
                &input->shape,
                spatial_count,
                feature,
                sample
            );
            float xhat =
                (input->data[index] - layer->batch_mean[feature]) *
                layer->batch_inv_std[feature];
            float centered_gradient =
                sample_count_f * output->grad[index] -
                layer->work_mean[feature] -
                xhat * layer->work_inv_std[feature];

            input->grad[index] += coefficient * centered_gradient;
        }
    }
    return VAE_C_OP_OK;
}


VAECOpStatus vae_c_batch_norm1d_forward(
    VAECBatchNorm *layer,
    VAECBatchNormMode mode,
    const VAECTensor *input,
    VAECTensor *output
)
{
    return vae_c_batch_norm_forward(
        layer,
        VAE_C_BATCH_NORM_KIND_1D,
        mode,
        input,
        output
    );
}


VAECOpStatus vae_c_batch_norm1d_backward(
    VAECBatchNorm *layer,
    const VAECTensor *input,
    const VAECTensor *output
)
{
    return vae_c_batch_norm_backward(
        layer,
        VAE_C_BATCH_NORM_KIND_1D,
        input,
        output
    );
}


VAECOpStatus vae_c_batch_norm2d_forward(
    VAECBatchNorm *layer,
    VAECBatchNormMode mode,
    const VAECTensor *input,
    VAECTensor *output
)
{
    return vae_c_batch_norm_forward(
        layer,
        VAE_C_BATCH_NORM_KIND_2D,
        mode,
        input,
        output
    );
}


VAECOpStatus vae_c_batch_norm2d_backward(
    VAECBatchNorm *layer,
    const VAECTensor *input,
    const VAECTensor *output
)
{
    return vae_c_batch_norm_backward(
        layer,
        VAE_C_BATCH_NORM_KIND_2D,
        input,
        output
    );
}
