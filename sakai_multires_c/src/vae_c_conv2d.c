#include "vae_c_conv2d.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>


static int vae_c_conv2d_size_add(
    size_t left,
    size_t right,
    size_t *result
)
{
    if (result == NULL || right > SIZE_MAX - left) {
        return 1;
    }
    *result = left + right;
    return 0;
}


static int vae_c_conv2d_size_multiply(
    size_t left,
    size_t right,
    size_t *result
)
{
    if (result == NULL || (left != 0 && right > SIZE_MAX / left)) {
        return 1;
    }
    *result = left * right;
    return 0;
}


static VAECOpStatus vae_c_conv2d_map_storage_status(
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
        case VAE_C_STORAGE_INVALID_RANK:
        case VAE_C_STORAGE_INVALID_DIMENSION:
        case VAE_C_STORAGE_SHAPE_MISMATCH:
            return VAE_C_OP_SHAPE_MISMATCH;
        case VAE_C_STORAGE_GRADIENT_NOT_AVAILABLE:
            return VAE_C_OP_GRADIENT_NOT_AVAILABLE;
        case VAE_C_STORAGE_INVALID_STATE:
        case VAE_C_STORAGE_OUT_OF_RANGE:
        default:
            return VAE_C_OP_INVALID_STATE;
    }
}


static VAECOpStatus vae_c_conv2d_validate_shape(const VAECShape *shape)
{
    VAECShape canonical;
    VAECStorageStatus storage_status;
    size_t axis;

    if (shape == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    storage_status = vae_c_shape_make(
        shape->rank,
        shape->dims,
        &canonical
    );
    if (storage_status != VAE_C_STORAGE_OK) {
        return vae_c_conv2d_map_storage_status(storage_status);
    }
    if (shape->count != canonical.count) {
        return VAE_C_OP_INVALID_STATE;
    }
    for (axis = 0; axis < shape->rank; axis++) {
        if (shape->dims[axis] != canonical.dims[axis] ||
            shape->strides[axis] != canonical.strides[axis]) {

            return VAE_C_OP_INVALID_STATE;
        }
    }
    for (axis = shape->rank; axis < VAE_C_TENSOR_MAX_RANK; axis++) {
        if (shape->dims[axis] != 0 || shape->strides[axis] != 0) {
            return VAE_C_OP_INVALID_STATE;
        }
    }
    return VAE_C_OP_OK;
}


static VAECOpStatus vae_c_conv2d_validate_tensor(
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
    status = vae_c_conv2d_validate_shape(&tensor->shape);
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
    if (vae_c_conv2d_size_multiply(
            tensor->shape.count,
            array_count,
            &total_elements) != 0 ||
        vae_c_conv2d_size_multiply(
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


static VAECOpStatus vae_c_conv2d_validate_parameter(
    const VAECParameter *parameter
)
{
    VAECOpStatus status;
    size_t total_elements;
    size_t expected_bytes;

    if (parameter == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    status = vae_c_conv2d_validate_shape(&parameter->shape);
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
    if (vae_c_conv2d_size_multiply(
            parameter->shape.count,
            4,
            &total_elements) != 0 ||
        vae_c_conv2d_size_multiply(
            total_elements,
            sizeof(float),
            &expected_bytes) != 0 ||
        parameter->owned_bytes != expected_bytes) {

        return VAE_C_OP_INVALID_STATE;
    }
    return VAE_C_OP_OK;
}


static bool vae_c_conv2d_parameters_have_exact_alias(
    const VAECParameter *left,
    const VAECParameter *right
)
{
    const void *left_buffers[4];
    const void *right_buffers[4];
    size_t left_index;
    size_t right_index;

    left_buffers[0] = left->values;
    left_buffers[1] = left->gradients;
    left_buffers[2] = left->adam_m;
    left_buffers[3] = left->adam_v;
    right_buffers[0] = right->values;
    right_buffers[1] = right->gradients;
    right_buffers[2] = right->adam_m;
    right_buffers[3] = right->adam_v;

    for (left_index = 0; left_index < 4; left_index++) {
        for (right_index = 0; right_index < 4; right_index++) {
            if (left_buffers[left_index] == right_buffers[right_index]) {
                return true;
            }
        }
    }
    return false;
}


static VAECOpStatus vae_c_conv2d_validate_layer(
    const VAECConv2D *layer
)
{
    VAECOpStatus status;

    if (layer == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (layer->in_channels == 0 ||
        layer->out_channels == 0 ||
        layer->kernel_size != VAE_C_CONV2D_KERNEL_SIZE ||
        layer->padding != VAE_C_CONV2D_PADDING ||
        (layer->stride != 1 && layer->stride != 2)) {

        return VAE_C_OP_INVALID_STATE;
    }
    status = vae_c_conv2d_validate_parameter(&layer->weights);
    if (status != VAE_C_OP_OK ||
        layer->weights.shape.rank != 4 ||
        layer->weights.shape.dims[0] != layer->out_channels ||
        layer->weights.shape.dims[1] != layer->in_channels ||
        layer->weights.shape.dims[2] != VAE_C_CONV2D_KERNEL_SIZE ||
        layer->weights.shape.dims[3] != VAE_C_CONV2D_KERNEL_SIZE) {

        return VAE_C_OP_INVALID_STATE;
    }
    if (!layer->with_bias) {
        return
            vae_c_parameter_is_empty(&layer->bias)
                ? VAE_C_OP_OK
                : VAE_C_OP_INVALID_STATE;
    }
    status = vae_c_conv2d_validate_parameter(&layer->bias);
    if (status != VAE_C_OP_OK ||
        layer->bias.shape.rank != 1 ||
        layer->bias.shape.dims[0] != layer->out_channels ||
        vae_c_conv2d_parameters_have_exact_alias(
            &layer->weights,
            &layer->bias)) {

        return VAE_C_OP_INVALID_STATE;
    }
    return VAE_C_OP_OK;
}


static bool vae_c_conv2d_tensors_have_exact_alias(
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
    return output->grad != NULL && input->data == output->grad;
}


static bool vae_c_conv2d_pointer_matches_parameter(
    const float *pointer,
    const VAECParameter *parameter
)
{
    return
        pointer != NULL &&
        (pointer == parameter->values ||
         pointer == parameter->gradients ||
         pointer == parameter->adam_m ||
         pointer == parameter->adam_v);
}


static bool vae_c_conv2d_tensor_aliases_layer(
    const VAECTensor *tensor,
    const VAECConv2D *layer
)
{
    if (vae_c_conv2d_pointer_matches_parameter(
            tensor->data,
            &layer->weights) ||
        vae_c_conv2d_pointer_matches_parameter(
            tensor->grad,
            &layer->weights)) {

        return true;
    }
    return
        layer->with_bias &&
        (vae_c_conv2d_pointer_matches_parameter(
             tensor->data,
             &layer->bias) ||
         vae_c_conv2d_pointer_matches_parameter(
             tensor->grad,
             &layer->bias));
}


static VAECOpStatus vae_c_conv2d_output_dimension(
    size_t input_dimension,
    size_t kernel_size,
    size_t stride,
    size_t padding,
    size_t *output_dimension
)
{
    size_t padding_twice;
    size_t padded;
    size_t numerator;
    size_t quotient;

    if (output_dimension == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (vae_c_conv2d_size_multiply(padding, 2, &padding_twice) != 0 ||
        vae_c_conv2d_size_add(
            input_dimension,
            padding_twice,
            &padded) != 0) {

        return VAE_C_OP_OVERFLOW;
    }
    if (padded < kernel_size) {
        return VAE_C_OP_SHAPE_MISMATCH;
    }
    numerator = padded - kernel_size;
    quotient = numerator / stride;
    if (vae_c_conv2d_size_add(quotient, 1, output_dimension) != 0) {
        return VAE_C_OP_OVERFLOW;
    }
    return VAE_C_OP_OK;
}


static VAECOpStatus vae_c_conv2d_validate_mapping_extent(
    size_t output_dimension,
    size_t stride,
    size_t kernel_size
)
{
    size_t last_base;
    size_t last_padded_coordinate;

    if (output_dimension == 0 || kernel_size == 0) {
        return VAE_C_OP_SHAPE_MISMATCH;
    }
    if (vae_c_conv2d_size_multiply(
            output_dimension - 1,
            stride,
            &last_base) != 0 ||
        vae_c_conv2d_size_add(
            last_base,
            kernel_size - 1,
            &last_padded_coordinate) != 0) {

        return VAE_C_OP_OVERFLOW;
    }
    return VAE_C_OP_OK;
}


static VAECOpStatus vae_c_conv2d_validate_index_extent(
    const VAECShape *shape
)
{
    size_t offset = 0;
    size_t axis;

    for (axis = 0; axis < shape->rank; axis++) {
        size_t term;

        if (vae_c_conv2d_size_multiply(
                shape->dims[axis] - 1,
                shape->strides[axis],
                &term) != 0 ||
            vae_c_conv2d_size_add(offset, term, &offset) != 0) {

            return VAE_C_OP_OVERFLOW;
        }
    }
    return offset < shape->count ? VAE_C_OP_OK : VAE_C_OP_INVALID_STATE;
}


static VAECOpStatus vae_c_conv2d_validate_execution(
    const VAECConv2D *layer,
    const VAECTensor *input,
    const VAECTensor *output,
    bool require_gradients
)
{
    VAECOpStatus status;
    size_t expected_height;
    size_t expected_width;

    status = vae_c_conv2d_validate_layer(layer);
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_conv2d_validate_tensor(input);
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_conv2d_validate_tensor(output);
    if (status != VAE_C_OP_OK) {
        return status;
    }
    if (vae_c_conv2d_tensors_have_exact_alias(input, output) ||
        vae_c_conv2d_tensor_aliases_layer(input, layer) ||
        vae_c_conv2d_tensor_aliases_layer(output, layer)) {

        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (input->shape.rank != 4 || output->shape.rank != 4 ||
        input->shape.dims[0] != output->shape.dims[0] ||
        input->shape.dims[1] != layer->in_channels ||
        output->shape.dims[1] != layer->out_channels) {

        return VAE_C_OP_SHAPE_MISMATCH;
    }
    status = vae_c_conv2d_output_dimension(
        input->shape.dims[2],
        layer->kernel_size,
        layer->stride,
        layer->padding,
        &expected_height
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_conv2d_output_dimension(
        input->shape.dims[3],
        layer->kernel_size,
        layer->stride,
        layer->padding,
        &expected_width
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_conv2d_validate_mapping_extent(
        output->shape.dims[2],
        layer->stride,
        layer->kernel_size
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_conv2d_validate_mapping_extent(
        output->shape.dims[3],
        layer->stride,
        layer->kernel_size
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    if (output->shape.dims[2] != expected_height ||
        output->shape.dims[3] != expected_width) {

        return VAE_C_OP_SHAPE_MISMATCH;
    }
    status = vae_c_conv2d_validate_index_extent(&input->shape);
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_conv2d_validate_index_extent(&output->shape);
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_conv2d_validate_index_extent(&layer->weights.shape);
    if (status != VAE_C_OP_OK) {
        return status;
    }
    if (require_gradients && (!input->has_grad || !output->has_grad)) {
        return VAE_C_OP_GRADIENT_NOT_AVAILABLE;
    }
    return VAE_C_OP_OK;
}


void vae_c_conv2d_init(VAECConv2D *layer)
{
    if (layer != NULL) {
        memset(layer, 0, sizeof(*layer));
    }
}


void vae_c_conv2d_free(VAECConv2D *layer)
{
    if (layer == NULL) {
        return;
    }
    vae_c_parameter_free(&layer->weights);
    vae_c_parameter_free(&layer->bias);
    vae_c_conv2d_init(layer);
}


bool vae_c_conv2d_is_empty(const VAECConv2D *layer)
{
    return
        layer != NULL &&
        layer->in_channels == 0 &&
        layer->out_channels == 0 &&
        layer->kernel_size == 0 &&
        layer->stride == 0 &&
        layer->padding == 0 &&
        !layer->with_bias &&
        vae_c_parameter_is_empty(&layer->weights) &&
        vae_c_parameter_is_empty(&layer->bias);
}


VAECOpStatus vae_c_conv2d_allocate(
    VAECConv2D *layer,
    size_t in_channels,
    size_t out_channels,
    size_t stride,
    bool with_bias,
    uint64_t weight_parameter_key,
    uint64_t bias_parameter_key
)
{
    VAECConv2D candidate;
    VAECShape weight_shape;
    VAECShape bias_shape;
    VAECStorageStatus storage_status;
    const size_t weight_dims[4] = {
        out_channels,
        in_channels,
        VAE_C_CONV2D_KERNEL_SIZE,
        VAE_C_CONV2D_KERNEL_SIZE
    };
    const size_t bias_dims[1] = {out_channels};

    if (layer == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (!vae_c_conv2d_is_empty(layer)) {
        return VAE_C_OP_OBJECT_NOT_EMPTY;
    }
    if (in_channels == 0 || out_channels == 0 ||
        (stride != 1 && stride != 2)) {

        return VAE_C_OP_INVALID_ARGUMENT;
    }
    storage_status = vae_c_shape_make(4, weight_dims, &weight_shape);
    if (storage_status != VAE_C_STORAGE_OK) {
        return vae_c_conv2d_map_storage_status(storage_status);
    }
    vae_c_conv2d_init(&candidate);
    storage_status = vae_c_parameter_allocate(
        &candidate.weights,
        &weight_shape,
        weight_parameter_key
    );
    if (storage_status != VAE_C_STORAGE_OK) {
        return vae_c_conv2d_map_storage_status(storage_status);
    }
    if (with_bias) {
        storage_status = vae_c_shape_make(1, bias_dims, &bias_shape);
        if (storage_status != VAE_C_STORAGE_OK) {
            vae_c_conv2d_free(&candidate);
            return vae_c_conv2d_map_storage_status(storage_status);
        }
        storage_status = vae_c_parameter_allocate(
            &candidate.bias,
            &bias_shape,
            bias_parameter_key
        );
        if (storage_status != VAE_C_STORAGE_OK) {
            vae_c_conv2d_free(&candidate);
            return vae_c_conv2d_map_storage_status(storage_status);
        }
    }
    candidate.in_channels = in_channels;
    candidate.out_channels = out_channels;
    candidate.kernel_size = VAE_C_CONV2D_KERNEL_SIZE;
    candidate.stride = stride;
    candidate.padding = VAE_C_CONV2D_PADDING;
    candidate.with_bias = with_bias;
    *layer = candidate;
    return VAE_C_OP_OK;
}


VAECOpStatus vae_c_conv2d_zero_grad(VAECConv2D *layer)
{
    VAECOpStatus status;
    VAECStorageStatus storage_status;

    if (layer == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (vae_c_conv2d_is_empty(layer)) {
        return VAE_C_OP_OK;
    }
    status = vae_c_conv2d_validate_layer(layer);
    if (status != VAE_C_OP_OK) {
        return status;
    }
    storage_status = vae_c_parameter_zero_grad(&layer->weights);
    if (storage_status != VAE_C_STORAGE_OK) {
        return vae_c_conv2d_map_storage_status(storage_status);
    }
    if (layer->with_bias) {
        storage_status = vae_c_parameter_zero_grad(&layer->bias);
        if (storage_status != VAE_C_STORAGE_OK) {
            return vae_c_conv2d_map_storage_status(storage_status);
        }
    }
    return VAE_C_OP_OK;
}


VAECOpStatus vae_c_conv2d_forward(
    const VAECConv2D *layer,
    const VAECTensor *input,
    VAECTensor *output
)
{
    VAECOpStatus status;
    size_t output_index = 0;
    size_t n;
    size_t output_channel;
    size_t output_y;
    size_t output_x;

    status = vae_c_conv2d_validate_execution(layer, input, output, false);
    if (status != VAE_C_OP_OK) {
        return status;
    }

    for (n = 0; n < input->shape.dims[0]; n++) {
        size_t input_batch_offset = n * input->shape.strides[0];

        for (output_channel = 0;
             output_channel < layer->out_channels;
             output_channel++) {

            size_t weight_output_offset =
                output_channel * layer->weights.shape.strides[0];

            for (output_y = 0;
                 output_y < output->shape.dims[2];
                 output_y++) {

                size_t base_y = output_y * layer->stride;

                for (output_x = 0;
                     output_x < output->shape.dims[3];
                     output_x++) {

                    size_t base_x = output_x * layer->stride;
                    size_t input_channel;
                    float sum =
                        layer->with_bias
                            ? layer->bias.values[output_channel]
                            : 0.0f;

                    for (input_channel = 0;
                         input_channel < layer->in_channels;
                         input_channel++) {

                        size_t input_channel_offset =
                            input_batch_offset +
                            input_channel * input->shape.strides[1];
                        size_t weight_input_offset =
                            weight_output_offset +
                            input_channel * layer->weights.shape.strides[1];
                        size_t kernel_y;

                        for (kernel_y = 0;
                             kernel_y < layer->kernel_size;
                             kernel_y++) {

                            size_t padded_y = base_y + kernel_y;
                            bool y_valid = padded_y >= layer->padding;
                            size_t input_y =
                                y_valid ? padded_y - layer->padding : 0;
                            size_t kernel_x;

                            if (y_valid && input_y >= input->shape.dims[2]) {
                                y_valid = false;
                            }
                            for (kernel_x = 0;
                                 kernel_x < layer->kernel_size;
                                 kernel_x++) {

                                size_t padded_x = base_x + kernel_x;
                                size_t input_x;
                                size_t input_index;
                                size_t weight_index;

                                if (!y_valid || padded_x < layer->padding) {
                                    continue;
                                }
                                input_x = padded_x - layer->padding;
                                if (input_x >= input->shape.dims[3]) {
                                    continue;
                                }
                                input_index =
                                    input_channel_offset +
                                    input_y * input->shape.strides[2] +
                                    input_x;
                                weight_index =
                                    weight_input_offset +
                                    kernel_y *
                                        layer->weights.shape.strides[2] +
                                    kernel_x;
                                sum +=
                                    input->data[input_index] *
                                    layer->weights.values[weight_index];
                            }
                        }
                    }
                    output->data[output_index] = sum;
                    output_index++;
                }
            }
        }
    }
    return VAE_C_OP_OK;
}


VAECOpStatus vae_c_conv2d_backward(
    VAECConv2D *layer,
    const VAECTensor *input,
    const VAECTensor *output
)
{
    VAECOpStatus status;
    size_t output_index = 0;
    size_t n;
    size_t output_channel;
    size_t output_y;
    size_t output_x;

    status = vae_c_conv2d_validate_execution(layer, input, output, true);
    if (status != VAE_C_OP_OK) {
        return status;
    }

    for (n = 0; n < input->shape.dims[0]; n++) {
        size_t input_batch_offset = n * input->shape.strides[0];

        for (output_channel = 0;
             output_channel < layer->out_channels;
             output_channel++) {

            size_t weight_output_offset =
                output_channel * layer->weights.shape.strides[0];

            for (output_y = 0;
                 output_y < output->shape.dims[2];
                 output_y++) {

                size_t base_y = output_y * layer->stride;

                for (output_x = 0;
                     output_x < output->shape.dims[3];
                     output_x++) {

                    size_t base_x = output_x * layer->stride;
                    float gradient = output->grad[output_index];
                    size_t input_channel;

                    if (layer->with_bias) {
                        layer->bias.gradients[output_channel] += gradient;
                    }
                    for (input_channel = 0;
                         input_channel < layer->in_channels;
                         input_channel++) {

                        size_t input_channel_offset =
                            input_batch_offset +
                            input_channel * input->shape.strides[1];
                        size_t weight_input_offset =
                            weight_output_offset +
                            input_channel * layer->weights.shape.strides[1];
                        size_t kernel_y;

                        for (kernel_y = 0;
                             kernel_y < layer->kernel_size;
                             kernel_y++) {

                            size_t padded_y = base_y + kernel_y;
                            bool y_valid = padded_y >= layer->padding;
                            size_t input_y =
                                y_valid ? padded_y - layer->padding : 0;
                            size_t kernel_x;

                            if (y_valid && input_y >= input->shape.dims[2]) {
                                y_valid = false;
                            }
                            for (kernel_x = 0;
                                 kernel_x < layer->kernel_size;
                                 kernel_x++) {

                                size_t padded_x = base_x + kernel_x;
                                size_t input_x;
                                size_t input_index;
                                size_t weight_index;

                                if (!y_valid || padded_x < layer->padding) {
                                    continue;
                                }
                                input_x = padded_x - layer->padding;
                                if (input_x >= input->shape.dims[3]) {
                                    continue;
                                }
                                input_index =
                                    input_channel_offset +
                                    input_y * input->shape.strides[2] +
                                    input_x;
                                weight_index =
                                    weight_input_offset +
                                    kernel_y *
                                        layer->weights.shape.strides[2] +
                                    kernel_x;
                                layer->weights.gradients[weight_index] +=
                                    gradient * input->data[input_index];
                                input->grad[input_index] +=
                                    gradient *
                                    layer->weights.values[weight_index];
                            }
                        }
                    }
                    output_index++;
                }
            }
        }
    }
    return VAE_C_OP_OK;
}
