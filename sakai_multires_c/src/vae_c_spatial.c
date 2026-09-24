#include "vae_c_spatial.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>


static int vae_c_spatial_size_multiply(
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


static VAECOpStatus vae_c_spatial_validate_shape(
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


static bool vae_c_spatial_shape_is_zero_initialized(
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


static VAECOpStatus vae_c_spatial_validate_tensor(
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

    status = vae_c_spatial_validate_shape(&tensor->shape);
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
    if (vae_c_spatial_size_multiply(
            tensor->shape.count,
            array_count,
            &total_elements) != 0 ||
        vae_c_spatial_size_multiply(
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


static bool vae_c_spatial_tensors_have_exact_alias(
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


static VAECOpStatus vae_c_spatial_validate_pair(
    const VAECTensor *input,
    const VAECTensor *output,
    bool require_gradients
)
{
    VAECOpStatus status;

    if (input == NULL || output == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    status = vae_c_spatial_validate_tensor(input);
    if (status != VAE_C_OP_OK) {
        return status;
    }

    status = vae_c_spatial_validate_tensor(output);
    if (status != VAE_C_OP_OK) {
        return status;
    }

    if (vae_c_spatial_tensors_have_exact_alias(input, output)) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    if (require_gradients &&
        (!input->has_grad || !output->has_grad)) {

        return VAE_C_OP_GRADIENT_NOT_AVAILABLE;
    }

    return VAE_C_OP_OK;
}


static bool vae_c_max_pool2d_cache_aliases_tensors(
    const VAECMaxPool2D *layer,
    const VAECTensor *input,
    const VAECTensor *output
)
{
    const void *cache_pointer;

    cache_pointer = (const void *)layer->argmax_indices;
    return
        cache_pointer == (const void *)input->data ||
        cache_pointer == (const void *)input->grad ||
        cache_pointer == (const void *)output->data ||
        cache_pointer == (const void *)output->grad;
}


static VAECOpStatus vae_c_max_pool2d_validate_layer(
    const VAECMaxPool2D *layer
)
{
    size_t expected_height;
    size_t expected_width;
    size_t expected_bytes;

    if (layer == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    if (layer->argmax_indices == NULL || layer->capacity == 0) {
        return VAE_C_OP_INVALID_STATE;
    }

    if (vae_c_spatial_size_multiply(
            layer->capacity,
            sizeof(size_t),
            &expected_bytes) != 0 ||
        layer->owned_bytes != expected_bytes) {

        return VAE_C_OP_INVALID_STATE;
    }

    if (!layer->cache_valid) {
        return
            layer->last_count == 0 &&
            vae_c_spatial_shape_is_zero_initialized(
                &layer->last_input_shape) &&
            vae_c_spatial_shape_is_zero_initialized(
                &layer->last_output_shape)
                ? VAE_C_OP_OK
                : VAE_C_OP_INVALID_STATE;
    }

    if (layer->last_count == 0 ||
        layer->last_count > layer->capacity ||
        vae_c_spatial_validate_shape(&layer->last_input_shape) !=
            VAE_C_OP_OK ||
        vae_c_spatial_validate_shape(&layer->last_output_shape) !=
            VAE_C_OP_OK ||
        layer->last_output_shape.count != layer->last_count) {

        return VAE_C_OP_INVALID_STATE;
    }

    if (layer->last_input_shape.rank != 4 ||
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
        layer->last_output_shape.dims[3] != expected_width) {

        return VAE_C_OP_INVALID_STATE;
    }

    return VAE_C_OP_OK;
}


static VAECOpStatus vae_c_max_pool2d_validate_shapes(
    const VAECTensor *input,
    const VAECTensor *output,
    size_t *input_plane_size,
    size_t *input_row_step
)
{
    size_t expected_height;
    size_t expected_width;
    size_t candidate_plane_size;
    size_t candidate_row_step;

    if (input_plane_size == NULL || input_row_step == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    if (input->shape.rank != 4 || output->shape.rank != 4 ||
        input->shape.dims[2] < 2 || input->shape.dims[3] < 2) {

        return VAE_C_OP_SHAPE_MISMATCH;
    }

    expected_height = (input->shape.dims[2] - 2) / 2 + 1;
    expected_width = (input->shape.dims[3] - 2) / 2 + 1;

    if (input->shape.dims[0] != output->shape.dims[0] ||
        input->shape.dims[1] != output->shape.dims[1] ||
        output->shape.dims[2] != expected_height ||
        output->shape.dims[3] != expected_width) {

        return VAE_C_OP_SHAPE_MISMATCH;
    }

    if (vae_c_spatial_size_multiply(
            input->shape.dims[2],
            input->shape.dims[3],
            &candidate_plane_size) != 0 ||
        vae_c_spatial_size_multiply(
            2,
            input->shape.dims[3],
            &candidate_row_step) != 0) {

        return VAE_C_OP_OVERFLOW;
    }

    *input_plane_size = candidate_plane_size;
    *input_row_step = candidate_row_step;
    return VAE_C_OP_OK;
}


void vae_c_max_pool2d_init(VAECMaxPool2D *layer)
{
    if (layer != NULL) {
        memset(layer, 0, sizeof(*layer));
    }
}


void vae_c_max_pool2d_free(VAECMaxPool2D *layer)
{
    if (layer == NULL) {
        return;
    }

    free(layer->argmax_indices);
    vae_c_max_pool2d_init(layer);
}


bool vae_c_max_pool2d_is_empty(const VAECMaxPool2D *layer)
{
    return
        layer != NULL &&
        layer->argmax_indices == NULL &&
        layer->capacity == 0 &&
        layer->owned_bytes == 0 &&
        layer->last_count == 0 &&
        vae_c_spatial_shape_is_zero_initialized(
            &layer->last_input_shape) &&
        vae_c_spatial_shape_is_zero_initialized(
            &layer->last_output_shape) &&
        !layer->cache_valid;
}


VAECOpStatus vae_c_max_pool2d_allocate(
    VAECMaxPool2D *layer,
    size_t max_output_elements
)
{
    VAECMaxPool2D candidate;
    size_t total_bytes;
    size_t *indices;

    if (layer == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    if (!vae_c_max_pool2d_is_empty(layer)) {
        return VAE_C_OP_OBJECT_NOT_EMPTY;
    }

    if (max_output_elements == 0) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    if (vae_c_spatial_size_multiply(
            max_output_elements,
            sizeof(size_t),
            &total_bytes) != 0) {

        return VAE_C_OP_OVERFLOW;
    }

    indices = (size_t *)calloc(max_output_elements, sizeof(size_t));
    if (indices == NULL) {
        return VAE_C_OP_ALLOCATION_FAILED;
    }

    vae_c_max_pool2d_init(&candidate);
    candidate.argmax_indices = indices;
    candidate.capacity = max_output_elements;
    candidate.owned_bytes = total_bytes;
    *layer = candidate;
    return VAE_C_OP_OK;
}


VAECOpStatus vae_c_max_pool2d_forward(
    VAECMaxPool2D *layer,
    const VAECTensor *input,
    VAECTensor *output
)
{
    VAECOpStatus status;
    size_t input_plane_size;
    size_t input_row_step;
    size_t input_plane_base;
    size_t output_index;
    size_t n;
    size_t c;
    size_t out_y;

    status = vae_c_max_pool2d_validate_layer(layer);
    if (status != VAE_C_OP_OK) {
        return status;
    }

    status = vae_c_spatial_validate_pair(input, output, false);
    if (status != VAE_C_OP_OK) {
        return status;
    }

    if (vae_c_max_pool2d_cache_aliases_tensors(layer, input, output)) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    status = vae_c_max_pool2d_validate_shapes(
        input,
        output,
        &input_plane_size,
        &input_row_step
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }

    if (output->shape.count > layer->capacity) {
        return VAE_C_OP_SHAPE_MISMATCH;
    }

    input_plane_base = 0;
    output_index = 0;

    for (n = 0; n < input->shape.dims[0]; n++) {
        for (c = 0; c < input->shape.dims[1]; c++) {
            size_t row_base = input_plane_base;

            for (out_y = 0;
                 out_y < output->shape.dims[2];
                 out_y++) {

                size_t out_x;
                size_t column = 0;
                size_t next_row_base = row_base + input->shape.dims[3];

                for (out_x = 0;
                     out_x < output->shape.dims[3];
                     out_x++) {

                    size_t best_index = row_base + column;
                    float best_value = input->data[best_index];
                    size_t candidate_index = best_index + 1;
                    float candidate_value = input->data[candidate_index];

                    if (candidate_value > best_value) {
                        best_value = candidate_value;
                        best_index = candidate_index;
                    }

                    candidate_index = next_row_base + column;
                    candidate_value = input->data[candidate_index];
                    if (candidate_value > best_value) {
                        best_value = candidate_value;
                        best_index = candidate_index;
                    }

                    candidate_index++;
                    candidate_value = input->data[candidate_index];
                    if (candidate_value > best_value) {
                        best_value = candidate_value;
                        best_index = candidate_index;
                    }

                    output->data[output_index] = best_value;
                    layer->argmax_indices[output_index] = best_index;
                    output_index++;
                    column += 2;
                }

                row_base += input_row_step;
            }

            input_plane_base += input_plane_size;
        }
    }

    layer->last_count = output->shape.count;
    layer->last_input_shape = input->shape;
    layer->last_output_shape = output->shape;
    layer->cache_valid = true;
    return VAE_C_OP_OK;
}


VAECOpStatus vae_c_max_pool2d_backward(
    const VAECMaxPool2D *layer,
    const VAECTensor *input,
    const VAECTensor *output
)
{
    VAECOpStatus status;
    size_t index;

    status = vae_c_max_pool2d_validate_layer(layer);
    if (status != VAE_C_OP_OK) {
        return status;
    }

    status = vae_c_spatial_validate_pair(input, output, true);
    if (status != VAE_C_OP_OK) {
        return status;
    }

    if (vae_c_max_pool2d_cache_aliases_tensors(layer, input, output)) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    if (!layer->cache_valid ||
        layer->last_count != output->shape.count ||
        !vae_c_shape_equal(
            &layer->last_input_shape,
            &input->shape) ||
        !vae_c_shape_equal(
            &layer->last_output_shape,
            &output->shape)) {

        return VAE_C_OP_INVALID_STATE;
    }

    for (index = 0; index < layer->last_count; index++) {
        if (layer->argmax_indices[index] >= input->shape.count) {
            return VAE_C_OP_INVALID_STATE;
        }
    }

    for (index = 0; index < layer->last_count; index++) {
        input->grad[layer->argmax_indices[index]] += output->grad[index];
    }

    return VAE_C_OP_OK;
}


static VAECOpStatus vae_c_nearest_upsample2d_validate(
    const VAECTensor *input,
    const VAECTensor *output,
    bool require_gradients,
    size_t *input_plane_size
)
{
    VAECOpStatus status;
    size_t candidate_plane_size;

    if (input_plane_size == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    status = vae_c_spatial_validate_pair(
        input,
        output,
        require_gradients
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }

    if (input->shape.rank != 4 || output->shape.rank != 4 ||
        input->shape.dims[0] != output->shape.dims[0] ||
        input->shape.dims[1] != output->shape.dims[1] ||
        output->shape.dims[2] < input->shape.dims[2] ||
        output->shape.dims[3] < input->shape.dims[3]) {

        return VAE_C_OP_SHAPE_MISMATCH;
    }

    if (output->shape.dims[2] - 1 >
            SIZE_MAX / input->shape.dims[2] ||
        output->shape.dims[3] - 1 >
            SIZE_MAX / input->shape.dims[3]) {

        return VAE_C_OP_OVERFLOW;
    }

    if (vae_c_spatial_size_multiply(
            input->shape.dims[2],
            input->shape.dims[3],
            &candidate_plane_size) != 0) {

        return VAE_C_OP_OVERFLOW;
    }

    *input_plane_size = candidate_plane_size;
    return VAE_C_OP_OK;
}


VAECOpStatus vae_c_nearest_upsample2d_forward(
    const VAECTensor *input,
    VAECTensor *output
)
{
    VAECOpStatus status;
    size_t input_plane_size;
    size_t input_plane_base;
    size_t output_index;
    size_t n;
    size_t c;
    size_t out_y;

    status = vae_c_nearest_upsample2d_validate(
        input,
        output,
        false,
        &input_plane_size
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }

    input_plane_base = 0;
    output_index = 0;

    for (n = 0; n < input->shape.dims[0]; n++) {
        for (c = 0; c < input->shape.dims[1]; c++) {
            for (out_y = 0;
                 out_y < output->shape.dims[2];
                 out_y++) {

                size_t source_y =
                    (out_y * input->shape.dims[2]) /
                    output->shape.dims[2];
                size_t source_row =
                    input_plane_base +
                    source_y * input->shape.dims[3];
                size_t out_x;

                for (out_x = 0;
                     out_x < output->shape.dims[3];
                     out_x++) {

                    size_t source_x =
                        (out_x * input->shape.dims[3]) /
                        output->shape.dims[3];

                    output->data[output_index] =
                        input->data[source_row + source_x];
                    output_index++;
                }
            }

            input_plane_base += input_plane_size;
        }
    }

    return VAE_C_OP_OK;
}


VAECOpStatus vae_c_nearest_upsample2d_backward(
    const VAECTensor *input,
    const VAECTensor *output
)
{
    VAECOpStatus status;
    size_t input_plane_size;
    size_t input_plane_base;
    size_t output_index;
    size_t n;
    size_t c;
    size_t out_y;

    status = vae_c_nearest_upsample2d_validate(
        input,
        output,
        true,
        &input_plane_size
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }

    input_plane_base = 0;
    output_index = 0;

    for (n = 0; n < input->shape.dims[0]; n++) {
        for (c = 0; c < input->shape.dims[1]; c++) {
            for (out_y = 0;
                 out_y < output->shape.dims[2];
                 out_y++) {

                size_t source_y =
                    (out_y * input->shape.dims[2]) /
                    output->shape.dims[2];
                size_t source_row =
                    input_plane_base +
                    source_y * input->shape.dims[3];
                size_t out_x;

                for (out_x = 0;
                     out_x < output->shape.dims[3];
                     out_x++) {

                    size_t source_x =
                        (out_x * input->shape.dims[3]) /
                        output->shape.dims[3];

                    input->grad[source_row + source_x] +=
                        output->grad[output_index];
                    output_index++;
                }
            }

            input_plane_base += input_plane_size;
        }
    }

    return VAE_C_OP_OK;
}
