#include "vae_c_dense.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>


static int vae_c_dense_size_multiply(
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


static VAECOpStatus vae_c_dense_map_storage_status(
    VAECStorageStatus status
)
{
    switch (status) {
        case VAE_C_STORAGE_OK:
            return VAE_C_OP_OK;
        case VAE_C_STORAGE_INVALID_ARGUMENT:
            return VAE_C_OP_INVALID_ARGUMENT;
        case VAE_C_STORAGE_OVERFLOW:
            return VAE_C_OP_OVERFLOW;
        case VAE_C_STORAGE_ALLOCATION_FAILED:
            return VAE_C_OP_ALLOCATION_FAILED;
        case VAE_C_STORAGE_OBJECT_NOT_EMPTY:
            return VAE_C_OP_OBJECT_NOT_EMPTY;
        case VAE_C_STORAGE_SHAPE_MISMATCH:
        case VAE_C_STORAGE_INVALID_RANK:
        case VAE_C_STORAGE_INVALID_DIMENSION:
            return VAE_C_OP_SHAPE_MISMATCH;
        case VAE_C_STORAGE_GRADIENT_NOT_AVAILABLE:
            return VAE_C_OP_GRADIENT_NOT_AVAILABLE;
        case VAE_C_STORAGE_INVALID_STATE:
        case VAE_C_STORAGE_OUT_OF_RANGE:
        default:
            return VAE_C_OP_INVALID_STATE;
    }
}


static VAECOpStatus vae_c_dense_validate_shape(const VAECShape *shape)
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
        return vae_c_dense_map_storage_status(storage_status);
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


static VAECOpStatus vae_c_dense_validate_tensor(
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

    status = vae_c_dense_validate_shape(&tensor->shape);
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
        if (tensor->owned_block != NULL || tensor->owned_bytes != 0) {
            return VAE_C_OP_INVALID_STATE;
        }
        return VAE_C_OP_OK;
    }

    if (tensor->owned_block == NULL ||
        tensor->data != tensor->owned_block) {

        return VAE_C_OP_INVALID_STATE;
    }

    array_count = tensor->has_grad ? 2 : 1;
    if (vae_c_dense_size_multiply(
            tensor->shape.count,
            array_count,
            &total_elements) != 0 ||
        vae_c_dense_size_multiply(
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


static VAECOpStatus vae_c_dense_validate_parameter(
    const VAECParameter *parameter
)
{
    VAECOpStatus status;
    size_t total_elements;
    size_t expected_bytes;

    if (parameter == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    status = vae_c_dense_validate_shape(&parameter->shape);
    if (status != VAE_C_OP_OK) {
        return status;
    }

    if (parameter->owned_block == NULL ||
        parameter->values != parameter->owned_block ||
        parameter->gradients !=
            parameter->values + parameter->shape.count ||
        parameter->adam_m !=
            parameter->gradients + parameter->shape.count ||
        parameter->adam_v !=
            parameter->adam_m + parameter->shape.count) {

        return VAE_C_OP_INVALID_STATE;
    }

    if (vae_c_dense_size_multiply(
            parameter->shape.count,
            4,
            &total_elements) != 0 ||
        vae_c_dense_size_multiply(
            total_elements,
            sizeof(float),
            &expected_bytes) != 0 ||
        parameter->owned_bytes != expected_bytes) {

        return VAE_C_OP_INVALID_STATE;
    }

    return VAE_C_OP_OK;
}


static VAECOpStatus vae_c_dense_validate_layer(
    const VAECDense *layer
)
{
    VAECOpStatus status;

    if (layer == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    if (layer->in_features == 0 || layer->out_features == 0) {
        return VAE_C_OP_INVALID_STATE;
    }

    status = vae_c_dense_validate_parameter(&layer->weights);
    if (status != VAE_C_OP_OK) {
        return status;
    }

    if (layer->weights.shape.rank != 2 ||
        layer->weights.shape.dims[0] != layer->out_features ||
        layer->weights.shape.dims[1] != layer->in_features) {

        return VAE_C_OP_INVALID_STATE;
    }

    if (!layer->with_bias) {
        return
            vae_c_parameter_is_empty(&layer->bias)
                ? VAE_C_OP_OK
                : VAE_C_OP_INVALID_STATE;
    }

    status = vae_c_dense_validate_parameter(&layer->bias);
    if (status != VAE_C_OP_OK) {
        return status;
    }

    if (layer->bias.shape.rank != 1 ||
        layer->bias.shape.dims[0] != layer->out_features) {

        return VAE_C_OP_INVALID_STATE;
    }

    return VAE_C_OP_OK;
}


static bool vae_c_dense_tensors_have_exact_alias(
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
        output->grad != NULL &&
        input->data == output->grad;
}


static bool vae_c_dense_pointer_matches_parameter(
    const float *pointer,
    const VAECParameter *parameter
)
{
    if (pointer == NULL || parameter == NULL) {
        return false;
    }

    return
        pointer == parameter->values ||
        pointer == parameter->gradients ||
        pointer == parameter->adam_m ||
        pointer == parameter->adam_v;
}


static bool vae_c_dense_tensor_aliases_layer(
    const VAECTensor *tensor,
    const VAECDense *layer
)
{
    if (vae_c_dense_pointer_matches_parameter(
            tensor->data,
            &layer->weights) ||
        vae_c_dense_pointer_matches_parameter(
            tensor->grad,
            &layer->weights)) {

        return true;
    }

    return
        layer->with_bias &&
        (vae_c_dense_pointer_matches_parameter(
             tensor->data,
             &layer->bias) ||
         vae_c_dense_pointer_matches_parameter(
             tensor->grad,
             &layer->bias));
}


static VAECOpStatus vae_c_dense_validate_execution(
    const VAECDense *layer,
    const VAECTensor *input,
    const VAECTensor *output,
    bool require_gradients
)
{
    VAECOpStatus status;

    status = vae_c_dense_validate_layer(layer);
    if (status != VAE_C_OP_OK) {
        return status;
    }

    status = vae_c_dense_validate_tensor(input);
    if (status != VAE_C_OP_OK) {
        return status;
    }

    status = vae_c_dense_validate_tensor(output);
    if (status != VAE_C_OP_OK) {
        return status;
    }

    if (vae_c_dense_tensors_have_exact_alias(input, output) ||
        vae_c_dense_tensor_aliases_layer(input, layer) ||
        vae_c_dense_tensor_aliases_layer(output, layer)) {

        return VAE_C_OP_INVALID_ARGUMENT;
    }

    if (input->shape.rank != 2 ||
        output->shape.rank != 2 ||
        input->shape.dims[0] != output->shape.dims[0] ||
        input->shape.dims[1] != layer->in_features ||
        output->shape.dims[1] != layer->out_features) {

        return VAE_C_OP_SHAPE_MISMATCH;
    }

    if (require_gradients &&
        (!input->has_grad || !output->has_grad)) {

        return VAE_C_OP_GRADIENT_NOT_AVAILABLE;
    }

    return VAE_C_OP_OK;
}


void vae_c_dense_init(VAECDense *layer)
{
    if (layer != NULL) {
        memset(layer, 0, sizeof(*layer));
    }
}


void vae_c_dense_free(VAECDense *layer)
{
    if (layer == NULL) {
        return;
    }

    vae_c_parameter_free(&layer->weights);
    vae_c_parameter_free(&layer->bias);
    vae_c_dense_init(layer);
}


bool vae_c_dense_is_empty(const VAECDense *layer)
{
    return
        layer != NULL &&
        layer->in_features == 0 &&
        layer->out_features == 0 &&
        !layer->with_bias &&
        vae_c_parameter_is_empty(&layer->weights) &&
        vae_c_parameter_is_empty(&layer->bias);
}


VAECOpStatus vae_c_dense_allocate(
    VAECDense *layer,
    size_t in_features,
    size_t out_features,
    bool with_bias,
    uint64_t weight_parameter_key,
    uint64_t bias_parameter_key
)
{
    VAECDense candidate;
    VAECShape weight_shape;
    VAECShape bias_shape;
    VAECStorageStatus storage_status;
    const size_t weight_dims[2] = {out_features, in_features};
    const size_t bias_dims[1] = {out_features};

    if (layer == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    if (!vae_c_dense_is_empty(layer)) {
        return VAE_C_OP_OBJECT_NOT_EMPTY;
    }

    if (in_features == 0 || out_features == 0) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    storage_status = vae_c_shape_make(
        2,
        weight_dims,
        &weight_shape
    );
    if (storage_status != VAE_C_STORAGE_OK) {
        return vae_c_dense_map_storage_status(storage_status);
    }

    vae_c_dense_init(&candidate);
    storage_status = vae_c_parameter_allocate(
        &candidate.weights,
        &weight_shape,
        weight_parameter_key
    );
    if (storage_status != VAE_C_STORAGE_OK) {
        return vae_c_dense_map_storage_status(storage_status);
    }

    if (with_bias) {
        storage_status = vae_c_shape_make(1, bias_dims, &bias_shape);
        if (storage_status != VAE_C_STORAGE_OK) {
            vae_c_dense_free(&candidate);
            return vae_c_dense_map_storage_status(storage_status);
        }

        storage_status = vae_c_parameter_allocate(
            &candidate.bias,
            &bias_shape,
            bias_parameter_key
        );
        if (storage_status != VAE_C_STORAGE_OK) {
            vae_c_dense_free(&candidate);
            return vae_c_dense_map_storage_status(storage_status);
        }
    }

    candidate.in_features = in_features;
    candidate.out_features = out_features;
    candidate.with_bias = with_bias;
    *layer = candidate;
    return VAE_C_OP_OK;
}


VAECOpStatus vae_c_dense_zero_grad(VAECDense *layer)
{
    VAECOpStatus status;
    VAECStorageStatus storage_status;

    if (layer == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    if (vae_c_dense_is_empty(layer)) {
        return VAE_C_OP_OK;
    }

    status = vae_c_dense_validate_layer(layer);
    if (status != VAE_C_OP_OK) {
        return status;
    }

    storage_status = vae_c_parameter_zero_grad(&layer->weights);
    if (storage_status != VAE_C_STORAGE_OK) {
        return vae_c_dense_map_storage_status(storage_status);
    }

    if (layer->with_bias) {
        storage_status = vae_c_parameter_zero_grad(&layer->bias);
        if (storage_status != VAE_C_STORAGE_OK) {
            return vae_c_dense_map_storage_status(storage_status);
        }
    }

    return VAE_C_OP_OK;
}


VAECOpStatus vae_c_dense_forward(
    const VAECDense *layer,
    const VAECTensor *input,
    VAECTensor *output
)
{
    VAECOpStatus status;
    size_t input_row_offset;
    size_t output_index;
    size_t n;
    size_t o;

    status = vae_c_dense_validate_execution(
        layer,
        input,
        output,
        false
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }

    input_row_offset = 0;
    output_index = 0;

    for (n = 0; n < input->shape.dims[0]; n++) {
        size_t weight_row_offset = 0;

        for (o = 0; o < layer->out_features; o++) {
            size_t i;
            float sum =
                layer->with_bias
                    ? layer->bias.values[o]
                    : 0.0f;

            for (i = 0; i < layer->in_features; i++) {
                sum +=
                    input->data[input_row_offset + i] *
                    layer->weights.values[weight_row_offset + i];
            }

            output->data[output_index] = sum;
            output_index++;
            weight_row_offset += layer->in_features;
        }

        input_row_offset += layer->in_features;
    }

    return VAE_C_OP_OK;
}


VAECOpStatus vae_c_dense_backward(
    VAECDense *layer,
    const VAECTensor *input,
    const VAECTensor *output
)
{
    VAECOpStatus status;
    size_t input_row_offset;
    size_t output_index;
    size_t n;
    size_t o;

    status = vae_c_dense_validate_execution(
        layer,
        input,
        output,
        true
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }

    input_row_offset = 0;
    output_index = 0;

    for (n = 0; n < input->shape.dims[0]; n++) {
        size_t weight_row_offset = 0;

        for (o = 0; o < layer->out_features; o++) {
            size_t i;
            float gradient = output->grad[output_index];

            if (layer->with_bias) {
                layer->bias.gradients[o] += gradient;
            }

            for (i = 0; i < layer->in_features; i++) {
                layer->weights.gradients[weight_row_offset + i] +=
                    gradient * input->data[input_row_offset + i];
                input->grad[input_row_offset + i] +=
                    gradient *
                    layer->weights.values[weight_row_offset + i];
            }

            output_index++;
            weight_row_offset += layer->in_features;
        }

        input_row_offset += layer->in_features;
    }

    return VAE_C_OP_OK;
}
