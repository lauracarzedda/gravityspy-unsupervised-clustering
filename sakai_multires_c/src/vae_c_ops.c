#include "vae_c_ops.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


static int vae_c_op_size_multiply(
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


static VAECOpStatus vae_c_op_validate_shape(const VAECShape *shape)
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


static VAECOpStatus vae_c_op_validate_tensor(const VAECTensor *tensor)
{
    VAECOpStatus status;
    size_t array_count;
    size_t total_elements;
    size_t expected_bytes;

    if (tensor == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    status = vae_c_op_validate_shape(&tensor->shape);
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
    if (vae_c_op_size_multiply(
            tensor->shape.count,
            array_count,
            &total_elements) != 0 ||
        vae_c_op_size_multiply(
            total_elements,
            sizeof(float),
            &expected_bytes) != 0) {

        return VAE_C_OP_INVALID_STATE;
    }

    if (tensor->owned_bytes != expected_bytes) {
        return VAE_C_OP_INVALID_STATE;
    }

    if (tensor->has_grad &&
        tensor->grad != tensor->data + tensor->shape.count) {

        return VAE_C_OP_INVALID_STATE;
    }

    return VAE_C_OP_OK;
}


static bool vae_c_op_tensors_have_exact_alias(
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
        (input->data == output->grad);
}


static VAECOpStatus vae_c_op_validate_pair(
    const VAECTensor *input,
    const VAECTensor *output,
    bool require_gradients
)
{
    VAECOpStatus status;

    if (input == NULL || output == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    status = vae_c_op_validate_tensor(input);
    if (status != VAE_C_OP_OK) {
        return status;
    }

    status = vae_c_op_validate_tensor(output);
    if (status != VAE_C_OP_OK) {
        return status;
    }

    if (vae_c_op_tensors_have_exact_alias(input, output)) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    if (require_gradients &&
        (!input->has_grad || !output->has_grad)) {

        return VAE_C_OP_GRADIENT_NOT_AVAILABLE;
    }

    return VAE_C_OP_OK;
}


VAECOpStatus vae_c_relu_forward(
    const VAECTensor *input,
    VAECTensor *output
)
{
    VAECOpStatus status;
    size_t index;

    status = vae_c_op_validate_pair(input, output, false);
    if (status != VAE_C_OP_OK) {
        return status;
    }

    if (!vae_c_shape_equal(&input->shape, &output->shape)) {
        return VAE_C_OP_SHAPE_MISMATCH;
    }

    for (index = 0; index < input->shape.count; index++) {
        output->data[index] =
            input->data[index] > 0.0f
                ? input->data[index]
                : 0.0f;
    }

    return VAE_C_OP_OK;
}


VAECOpStatus vae_c_relu_backward(
    const VAECTensor *input,
    const VAECTensor *output
)
{
    VAECOpStatus status;
    size_t index;

    status = vae_c_op_validate_pair(input, output, true);
    if (status != VAE_C_OP_OK) {
        return status;
    }

    if (!vae_c_shape_equal(&input->shape, &output->shape)) {
        return VAE_C_OP_SHAPE_MISMATCH;
    }

    for (index = 0; index < input->shape.count; index++) {
        if (input->data[index] > 0.0f) {
            input->grad[index] += output->grad[index];
        }
    }

    return VAE_C_OP_OK;
}


static VAECOpStatus vae_c_global_avg_pool2d_validate(
    const VAECTensor *input,
    const VAECTensor *output,
    bool require_gradients,
    size_t *spatial_count
)
{
    VAECOpStatus status;
    size_t candidate_spatial_count;

    if (spatial_count == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    status = vae_c_op_validate_pair(
        input,
        output,
        require_gradients
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }

    if (input->shape.rank != 4 ||
        output->shape.rank != 2 ||
        input->shape.dims[0] != output->shape.dims[0] ||
        input->shape.dims[1] != output->shape.dims[1]) {

        return VAE_C_OP_SHAPE_MISMATCH;
    }

    if (vae_c_op_size_multiply(
            input->shape.dims[2],
            input->shape.dims[3],
            &candidate_spatial_count) != 0) {

        return VAE_C_OP_OVERFLOW;
    }

    *spatial_count = candidate_spatial_count;
    return VAE_C_OP_OK;
}


VAECOpStatus vae_c_global_avg_pool2d_forward(
    const VAECTensor *input,
    VAECTensor *output
)
{
    VAECOpStatus status;
    size_t spatial_count;
    size_t input_index;
    size_t output_index;
    size_t n;
    size_t c;
    size_t spatial_index;
    float inverse_spatial_count;

    status = vae_c_global_avg_pool2d_validate(
        input,
        output,
        false,
        &spatial_count
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }

    inverse_spatial_count = 1.0f / (float)spatial_count;
    input_index = 0;
    output_index = 0;

    for (n = 0; n < input->shape.dims[0]; n++) {
        for (c = 0; c < input->shape.dims[1]; c++) {
            float sum = 0.0f;

            for (spatial_index = 0;
                 spatial_index < spatial_count;
                 spatial_index++) {

                sum += input->data[input_index];
                input_index++;
            }

            output->data[output_index] =
                sum * inverse_spatial_count;
            output_index++;
        }
    }

    return VAE_C_OP_OK;
}


VAECOpStatus vae_c_global_avg_pool2d_backward(
    const VAECTensor *input,
    const VAECTensor *output
)
{
    VAECOpStatus status;
    size_t spatial_count;
    size_t input_index;
    size_t output_index;
    size_t n;
    size_t c;
    size_t spatial_index;
    float inverse_spatial_count;

    status = vae_c_global_avg_pool2d_validate(
        input,
        output,
        true,
        &spatial_count
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }

    inverse_spatial_count = 1.0f / (float)spatial_count;
    input_index = 0;
    output_index = 0;

    for (n = 0; n < input->shape.dims[0]; n++) {
        for (c = 0; c < input->shape.dims[1]; c++) {
            float contribution =
                output->grad[output_index] *
                inverse_spatial_count;

            for (spatial_index = 0;
                 spatial_index < spatial_count;
                 spatial_index++) {

                input->grad[input_index] += contribution;
                input_index++;
            }

            output_index++;
        }
    }

    return VAE_C_OP_OK;
}


const char *vae_c_op_status_name(VAECOpStatus status)
{
    switch (status) {
        case VAE_C_OP_OK:
            return "success";
        case VAE_C_OP_INVALID_ARGUMENT:
            return "invalid_argument";
        case VAE_C_OP_INVALID_STATE:
            return "invalid_state";
        case VAE_C_OP_SHAPE_MISMATCH:
            return "shape_mismatch";
        case VAE_C_OP_GRADIENT_NOT_AVAILABLE:
            return "gradient_not_available";
        case VAE_C_OP_OBJECT_NOT_EMPTY:
            return "object_not_empty";
        case VAE_C_OP_OVERFLOW:
            return "overflow";
        case VAE_C_OP_ALLOCATION_FAILED:
            return "allocation_failed";
        default:
            return "unknown";
    }
}
