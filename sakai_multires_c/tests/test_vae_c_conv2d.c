#include "vae_c_conv2d.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>


typedef struct {
    double max_absolute;
    double max_relative;
    size_t worst_index;
} GradientStats;


static int report_check(const char *name, int passed)
{
    if (passed) {
        printf("[PASS] %s\n", name);
        return 0;
    }
    fprintf(stderr, "[FAIL] %s\n", name);
    return 1;
}


static int close_double(double actual, double expected, double tolerance)
{
    return
        isfinite(actual) &&
        isfinite(expected) &&
        fabs(actual - expected) <= tolerance;
}


static int float_array_close_double(
    const float *actual,
    const double *expected,
    size_t count,
    double tolerance
)
{
    size_t index;

    for (index = 0; index < count; index++) {
        if (!close_double((double)actual[index], expected[index], tolerance)) {
            return 0;
        }
    }
    return 1;
}


static int float_array_equal(
    const float *left,
    const float *right,
    size_t count
)
{
    return memcmp(left, right, count * sizeof(float)) == 0;
}


static int float_array_is_zero(const float *values, size_t count)
{
    size_t index;

    for (index = 0; index < count; index++) {
        if (values[index] != 0.0f) {
            return 0;
        }
    }
    return 1;
}


static int make_tensor(
    size_t rank,
    const size_t *dims,
    int with_gradient,
    VAECTensor *tensor
)
{
    VAECShape shape;

    return
        vae_c_shape_make(rank, dims, &shape) == VAE_C_STORAGE_OK &&
        vae_c_tensor_allocate(tensor, &shape, with_gradient != 0) ==
            VAE_C_STORAGE_OK;
}


static int make_view(
    size_t rank,
    const size_t *dims,
    float *data,
    float *gradient,
    VAECTensor *tensor
)
{
    VAECShape shape;

    if (vae_c_shape_make(rank, dims, &shape) != VAE_C_STORAGE_OK) {
        return 0;
    }
    vae_c_tensor_init(tensor);
    tensor->shape = shape;
    tensor->data = data;
    tensor->grad = gradient;
    tensor->has_grad = gradient != NULL;
    return 1;
}


static size_t reference_output_dimension(size_t input, size_t stride)
{
    return ((input + 2 - 3) / stride) + 1;
}


static size_t reference_nchw_index(
    size_t n,
    size_t channel,
    size_t y,
    size_t x,
    size_t channels,
    size_t height,
    size_t width
)
{
    return (((n * channels + channel) * height + y) * width) + x;
}


static size_t reference_oihw_index(
    size_t output_channel,
    size_t input_channel,
    size_t kernel_y,
    size_t kernel_x,
    size_t input_channels
)
{
    return
        (((output_channel * input_channels + input_channel) * 3 +
          kernel_y) * 3) + kernel_x;
}


static void reference_forward(
    size_t batch,
    size_t input_channels,
    size_t output_channels,
    size_t input_height,
    size_t input_width,
    size_t stride,
    int with_bias,
    const float *input,
    const float *weights,
    const float *bias,
    double *output
)
{
    const size_t output_height =
        reference_output_dimension(input_height, stride);
    const size_t output_width =
        reference_output_dimension(input_width, stride);
    size_t n;
    size_t output_channel;
    size_t output_y;
    size_t output_x;

    for (n = 0; n < batch; n++) {
        for (output_channel = 0;
             output_channel < output_channels;
             output_channel++) {

            for (output_y = 0; output_y < output_height; output_y++) {
                for (output_x = 0; output_x < output_width; output_x++) {
                    double sum =
                        with_bias ? (double)bias[output_channel] : 0.0;
                    size_t input_channel;

                    for (input_channel = 0;
                         input_channel < input_channels;
                         input_channel++) {

                        size_t kernel_y;

                        for (kernel_y = 0; kernel_y < 3; kernel_y++) {
                            const size_t padded_y =
                                output_y * stride + kernel_y;
                            size_t kernel_x;

                            for (kernel_x = 0; kernel_x < 3; kernel_x++) {
                                const size_t padded_x =
                                    output_x * stride + kernel_x;
                                size_t input_y;
                                size_t input_x;
                                size_t input_index;
                                size_t weight_index;

                                if (padded_y < 1 || padded_x < 1) {
                                    continue;
                                }
                                input_y = padded_y - 1;
                                input_x = padded_x - 1;
                                if (input_y >= input_height ||
                                    input_x >= input_width) {

                                    continue;
                                }
                                input_index = reference_nchw_index(
                                    n,
                                    input_channel,
                                    input_y,
                                    input_x,
                                    input_channels,
                                    input_height,
                                    input_width
                                );
                                weight_index = reference_oihw_index(
                                    output_channel,
                                    input_channel,
                                    kernel_y,
                                    kernel_x,
                                    input_channels
                                );
                                sum +=
                                    (double)input[input_index] *
                                    (double)weights[weight_index];
                            }
                        }
                    }
                    output[reference_nchw_index(
                        n,
                        output_channel,
                        output_y,
                        output_x,
                        output_channels,
                        output_height,
                        output_width
                    )] = sum;
                }
            }
        }
    }
}


static void reference_backward(
    size_t batch,
    size_t input_channels,
    size_t output_channels,
    size_t input_height,
    size_t input_width,
    size_t stride,
    int with_bias,
    const float *input,
    const float *weights,
    const float *upstream,
    double *input_gradient,
    double *weight_gradient,
    double *bias_gradient
)
{
    const size_t output_height =
        reference_output_dimension(input_height, stride);
    const size_t output_width =
        reference_output_dimension(input_width, stride);
    size_t n;
    size_t output_channel;
    size_t output_y;
    size_t output_x;

    for (n = 0; n < batch; n++) {
        for (output_channel = 0;
             output_channel < output_channels;
             output_channel++) {

            for (output_y = 0; output_y < output_height; output_y++) {
                for (output_x = 0; output_x < output_width; output_x++) {
                    const size_t output_index = reference_nchw_index(
                        n,
                        output_channel,
                        output_y,
                        output_x,
                        output_channels,
                        output_height,
                        output_width
                    );
                    const double gradient = (double)upstream[output_index];
                    size_t input_channel;

                    if (with_bias) {
                        bias_gradient[output_channel] += gradient;
                    }
                    for (input_channel = 0;
                         input_channel < input_channels;
                         input_channel++) {

                        size_t kernel_y;

                        for (kernel_y = 0; kernel_y < 3; kernel_y++) {
                            const size_t padded_y =
                                output_y * stride + kernel_y;
                            size_t kernel_x;

                            for (kernel_x = 0; kernel_x < 3; kernel_x++) {
                                const size_t padded_x =
                                    output_x * stride + kernel_x;
                                size_t input_y;
                                size_t input_x;
                                size_t input_index;
                                size_t weight_index;

                                if (padded_y < 1 || padded_x < 1) {
                                    continue;
                                }
                                input_y = padded_y - 1;
                                input_x = padded_x - 1;
                                if (input_y >= input_height ||
                                    input_x >= input_width) {

                                    continue;
                                }
                                input_index = reference_nchw_index(
                                    n,
                                    input_channel,
                                    input_y,
                                    input_x,
                                    input_channels,
                                    input_height,
                                    input_width
                                );
                                weight_index = reference_oihw_index(
                                    output_channel,
                                    input_channel,
                                    kernel_y,
                                    kernel_x,
                                    input_channels
                                );
                                weight_gradient[weight_index] +=
                                    gradient * (double)input[input_index];
                                input_gradient[input_index] +=
                                    gradient * (double)weights[weight_index];
                            }
                        }
                    }
                }
            }
        }
    }
}


static void update_gradient_stats(
    GradientStats *stats,
    size_t index,
    double analytic,
    double numeric
)
{
    const double absolute = fabs(analytic - numeric);
    double denominator = fabs(analytic) + fabs(numeric);
    double relative;

    if (denominator < 1.0e-6) {
        denominator = 1.0e-6;
    }
    relative = absolute / denominator;
    if (absolute > stats->max_absolute) {
        stats->max_absolute = absolute;
    }
    if (relative > stats->max_relative) {
        stats->max_relative = relative;
        stats->worst_index = index;
    }
}


static int gradient_stats_pass(const GradientStats *stats)
{
    return
        stats->max_absolute <= 1.0e-3 &&
        stats->max_relative <= 1.0e-2;
}


static int test_lifecycle_and_layout(void)
{
    VAECConv2D biased;
    VAECConv2D unbiased;
    VAECConv2D snapshot;
    VAECConv2D corrupted;
    float weight_values_snapshot[54];
    float weight_adam_snapshot[108];
    float bias_values_snapshot[3];
    float bias_adam_snapshot[6];
    size_t index;
    int passed;

    vae_c_conv2d_init(&biased);
    vae_c_conv2d_init(&unbiased);
    snapshot = biased;
    if (!vae_c_conv2d_is_empty(&biased) ||
        vae_c_conv2d_zero_grad(&biased) != VAE_C_OP_OK ||
        vae_c_conv2d_allocate(NULL, 1, 1, 1, false, 0, 0) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        vae_c_conv2d_allocate(&biased, 0, 1, 1, false, 0, 0) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        memcmp(&biased, &snapshot, sizeof(biased)) != 0 ||
        vae_c_conv2d_allocate(&biased, 1, 0, 1, false, 0, 0) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        vae_c_conv2d_allocate(&biased, 1, 1, 0, false, 0, 0) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        vae_c_conv2d_allocate(&biased, 1, 1, 3, false, 0, 0) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        vae_c_conv2d_allocate(
            &biased,
            SIZE_MAX / 9 + 1,
            1,
            1,
            false,
            0,
            0) != VAE_C_OP_OVERFLOW ||
        memcmp(&biased, &snapshot, sizeof(biased)) != 0 ||
        vae_c_conv2d_allocate(&biased, 2, 3, 2, true, 101, 0) !=
            VAE_C_OP_OK ||
        vae_c_conv2d_allocate(&unbiased, 1, 2, 1, false, 0, 0) !=
            VAE_C_OP_OK) {

        vae_c_conv2d_free(&unbiased);
        vae_c_conv2d_free(&biased);
        return 0;
    }
    passed =
        biased.in_channels == 2 &&
        biased.out_channels == 3 &&
        biased.kernel_size == 3 &&
        biased.padding == 1 &&
        biased.stride == 2 &&
        biased.with_bias &&
        biased.weights.shape.rank == 4 &&
        biased.weights.shape.dims[0] == 3 &&
        biased.weights.shape.dims[1] == 2 &&
        biased.weights.shape.dims[2] == 3 &&
        biased.weights.shape.dims[3] == 3 &&
        biased.weights.shape.count == 54 &&
        biased.weights.parameter_key == 101 &&
        biased.bias.shape.rank == 1 &&
        biased.bias.shape.dims[0] == 3 &&
        biased.bias.shape.count == 3 &&
        biased.bias.parameter_key == 0 &&
        unbiased.weights.parameter_key == 0 &&
        !unbiased.with_bias &&
        vae_c_parameter_is_empty(&unbiased.bias) &&
        float_array_is_zero(biased.weights.owned_block, 54 * 4) &&
        float_array_is_zero(biased.bias.owned_block, 3 * 4);

    for (index = 0; index < 54; index++) {
        biased.weights.values[index] = (float)index * 0.01f;
        biased.weights.gradients[index] = (float)index + 1.0f;
        biased.weights.adam_m[index] = (float)index * 0.02f;
        biased.weights.adam_v[index] = (float)index * 0.03f;
    }
    for (index = 0; index < 3; index++) {
        biased.bias.values[index] = (float)index * 0.1f;
        biased.bias.gradients[index] = (float)index + 2.0f;
        biased.bias.adam_m[index] = (float)index * 0.2f;
        biased.bias.adam_v[index] = (float)index * 0.3f;
    }
    memcpy(
        weight_values_snapshot,
        biased.weights.values,
        sizeof(weight_values_snapshot));
    memcpy(
        weight_adam_snapshot,
        biased.weights.adam_m,
        sizeof(weight_adam_snapshot));
    memcpy(
        bias_values_snapshot,
        biased.bias.values,
        sizeof(bias_values_snapshot));
    memcpy(
        bias_adam_snapshot,
        biased.bias.adam_m,
        sizeof(bias_adam_snapshot));
    passed =
        passed &&
        vae_c_conv2d_zero_grad(&biased) == VAE_C_OP_OK &&
        float_array_is_zero(biased.weights.gradients, 54) &&
        float_array_is_zero(biased.bias.gradients, 3) &&
        float_array_equal(
            biased.weights.values,
            weight_values_snapshot,
            54) &&
        float_array_equal(
            biased.weights.adam_m,
            weight_adam_snapshot,
            108) &&
        float_array_equal(biased.bias.values, bias_values_snapshot, 3) &&
        float_array_equal(biased.bias.adam_m, bias_adam_snapshot, 6);

    snapshot = biased;
    passed =
        passed &&
        vae_c_conv2d_allocate(&biased, 1, 1, 1, false, 0, 0) ==
            VAE_C_OP_OBJECT_NOT_EMPTY &&
        memcmp(&biased, &snapshot, sizeof(biased)) == 0;
    corrupted = biased;
    corrupted.kernel_size = 5;
    passed =
        passed &&
        vae_c_conv2d_zero_grad(&corrupted) == VAE_C_OP_INVALID_STATE;

    vae_c_conv2d_free(&unbiased);
    vae_c_conv2d_free(&biased);
    vae_c_conv2d_free(&biased);
    vae_c_conv2d_free(NULL);
    return
        passed &&
        vae_c_conv2d_is_empty(&biased) &&
        vae_c_conv2d_is_empty(&unbiased);
}


static int test_single_channel_forward(void)
{
    const size_t dims[4] = {1, 1, 3, 3};
    const float input_values[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    const float expected_ones[9] = {
        12, 21, 16,
        27, 45, 33,
        24, 39, 28
    };
    VAECConv2D layer;
    VAECTensor input;
    VAECTensor output;
    double expected_asymmetric[9];
    double rotated_center = 0.0;
    size_t index;
    int passed = 0;

    vae_c_conv2d_init(&layer);
    vae_c_tensor_init(&input);
    vae_c_tensor_init(&output);
    if (vae_c_conv2d_allocate(&layer, 1, 1, 1, false, 0, 0) !=
            VAE_C_OP_OK ||
        !make_tensor(4, dims, true, &input) ||
        !make_tensor(4, dims, true, &output)) {

        goto cleanup;
    }
    memcpy(input.data, input_values, sizeof(input_values));
    for (index = 0; index < 9; index++) {
        layer.weights.values[index] = 1.0f;
    }
    if (vae_c_conv2d_forward(&layer, &input, &output) != VAE_C_OP_OK ||
        !float_array_equal(output.data, expected_ones, 9)) {

        goto cleanup;
    }
    for (index = 0; index < 9; index++) {
        layer.weights.values[index] = (float)(index + 1);
        rotated_center +=
            (double)input_values[index] * (double)(9 - index);
    }
    reference_forward(
        1, 1, 1, 3, 3, 1, 0,
        input.data,
        layer.weights.values,
        NULL,
        expected_asymmetric
    );
    passed =
        vae_c_conv2d_forward(&layer, &input, &output) == VAE_C_OP_OK &&
        float_array_close_double(output.data, expected_asymmetric, 9, 1e-6) &&
        close_double((double)output.data[4], expected_asymmetric[4], 1e-6) &&
        !close_double((double)output.data[4], rotated_center, 1e-6);

cleanup:
    vae_c_tensor_free(&output);
    vae_c_tensor_free(&input);
    vae_c_conv2d_free(&layer);
    return passed;
}


static int test_multichannel_forward_and_shapes(void)
{
    const size_t input_dims[4] = {1, 2, 3, 4};
    const size_t output_dims[4] = {1, 2, 3, 4};
    const size_t sides[7] = {5, 75, 112, 19, 10, 28, 14};
    const size_t expected_sides[7] = {3, 38, 56, 10, 5, 14, 7};
    VAECConv2D layer;
    VAECConv2D shape_layer;
    VAECTensor input;
    VAECTensor output;
    double expected[24];
    float input_snapshot[24];
    float input_grad_snapshot[24];
    float output_grad_snapshot[24];
    float weight_snapshot[144];
    float bias_snapshot[8];
    size_t index;
    int passed = 0;

    vae_c_conv2d_init(&layer);
    vae_c_conv2d_init(&shape_layer);
    vae_c_tensor_init(&input);
    vae_c_tensor_init(&output);
    if (vae_c_conv2d_allocate(&layer, 2, 2, 1, true, 0, 0) !=
            VAE_C_OP_OK ||
        vae_c_conv2d_allocate(&shape_layer, 1, 1, 2, false, 0, 0) !=
            VAE_C_OP_OK ||
        !make_tensor(4, input_dims, true, &input) ||
        !make_tensor(4, output_dims, true, &output)) {

        goto cleanup;
    }
    for (index = 0; index < 24; index++) {
        input.data[index] =
            (float)((int)(index % 11) - 5) * 0.25f;
        input.grad[index] = (float)index * 0.01f;
        output.grad[index] = (float)index * -0.02f;
    }
    for (index = 0; index < 36; index++) {
        layer.weights.values[index] =
            (float)((int)(index % 9) - 4) * 0.07f;
        layer.weights.gradients[index] = (float)index * 0.03f;
        layer.weights.adam_m[index] = (float)index * 0.04f;
        layer.weights.adam_v[index] = (float)index * 0.05f;
    }
    layer.bias.values[0] = 0.3f;
    layer.bias.values[1] = -0.2f;
    layer.bias.gradients[0] = 0.7f;
    layer.bias.gradients[1] = -0.8f;
    layer.bias.adam_m[0] = 0.9f;
    layer.bias.adam_m[1] = -1.0f;
    layer.bias.adam_v[0] = 1.1f;
    layer.bias.adam_v[1] = 1.2f;
    reference_forward(
        1, 2, 2, 3, 4, 1, 1,
        input.data,
        layer.weights.values,
        layer.bias.values,
        expected
    );
    memcpy(input_snapshot, input.data, sizeof(input_snapshot));
    memcpy(input_grad_snapshot, input.grad, sizeof(input_grad_snapshot));
    memcpy(output_grad_snapshot, output.grad, sizeof(output_grad_snapshot));
    memcpy(weight_snapshot, layer.weights.owned_block, sizeof(weight_snapshot));
    memcpy(bias_snapshot, layer.bias.owned_block, sizeof(bias_snapshot));
    if (vae_c_conv2d_forward(&layer, &input, &output) != VAE_C_OP_OK ||
        !float_array_close_double(output.data, expected, 24, 2e-6) ||
        !float_array_equal(input.data, input_snapshot, 24) ||
        !float_array_equal(input.grad, input_grad_snapshot, 24) ||
        !float_array_equal(output.grad, output_grad_snapshot, 24) ||
        !float_array_equal(layer.weights.owned_block, weight_snapshot, 144) ||
        !float_array_equal(layer.bias.owned_block, bias_snapshot, 8)) {

        goto cleanup;
    }

    for (index = 0; index < 7; index++) {
        const size_t shape_input_dims[4] = {1, 1, sides[index], sides[index]};
        const size_t shape_output_dims[4] = {
            1, 1, expected_sides[index], expected_sides[index]
        };
        VAECTensor shape_input;
        VAECTensor shape_output;

        vae_c_tensor_init(&shape_input);
        vae_c_tensor_init(&shape_output);
        if (!make_tensor(4, shape_input_dims, false, &shape_input) ||
            !make_tensor(4, shape_output_dims, false, &shape_output) ||
            vae_c_conv2d_forward(
                &shape_layer,
                &shape_input,
                &shape_output) != VAE_C_OP_OK ||
            !float_array_is_zero(
                shape_output.data,
                shape_output.shape.count)) {

            vae_c_tensor_free(&shape_output);
            vae_c_tensor_free(&shape_input);
            goto cleanup;
        }
        vae_c_tensor_free(&shape_output);
        vae_c_tensor_free(&shape_input);
    }
    passed = 1;

cleanup:
    vae_c_tensor_free(&output);
    vae_c_tensor_free(&input);
    vae_c_conv2d_free(&shape_layer);
    vae_c_conv2d_free(&layer);
    return passed;
}


static int run_analytic_backward_case(size_t stride, int with_bias)
{
    const size_t input_dims[4] = {1, 2, 3, 3};
    size_t output_dims[4] = {1, 2, 0, 0};
    VAECConv2D layer;
    VAECTensor input;
    VAECTensor output;
    double expected_input_gradient[18] = {0};
    double expected_weight_gradient[36] = {0};
    double expected_bias_gradient[2] = {0};
    float input_data_snapshot[18];
    float output_data_snapshot[18];
    float output_grad_snapshot[18];
    float weight_values_snapshot[36];
    float weight_adam_snapshot[72];
    float bias_values_snapshot[2];
    float bias_adam_snapshot[4];
    size_t output_count;
    size_t index;
    int passed = 0;

    output_dims[2] = reference_output_dimension(3, stride);
    output_dims[3] = reference_output_dimension(3, stride);
    output_count = 2 * output_dims[2] * output_dims[3];
    vae_c_conv2d_init(&layer);
    vae_c_tensor_init(&input);
    vae_c_tensor_init(&output);
    if (vae_c_conv2d_allocate(
            &layer,
            2,
            2,
            stride,
            with_bias != 0,
            0,
            0) != VAE_C_OP_OK ||
        !make_tensor(4, input_dims, true, &input) ||
        !make_tensor(4, output_dims, true, &output)) {

        goto cleanup;
    }
    for (index = 0; index < 18; index++) {
        input.data[index] =
            (float)((int)(index % 8) - 3) * 0.2f;
    }
    for (index = 0; index < 36; index++) {
        layer.weights.values[index] =
            (float)((int)(index % 10) - 5) * 0.06f;
        layer.weights.adam_m[index] = (float)index * 0.01f;
        layer.weights.adam_v[index] = (float)index * -0.01f;
    }
    if (with_bias) {
        layer.bias.values[0] = 0.15f;
        layer.bias.values[1] = -0.25f;
        layer.bias.adam_m[0] = 0.3f;
        layer.bias.adam_m[1] = -0.4f;
        layer.bias.adam_v[0] = 0.5f;
        layer.bias.adam_v[1] = -0.6f;
    }
    if (vae_c_conv2d_forward(&layer, &input, &output) != VAE_C_OP_OK) {
        goto cleanup;
    }
    for (index = 0; index < output_count; index++) {
        output.grad[index] =
            (float)((int)(index % 7) - 3) * 0.17f;
    }
    reference_backward(
        1, 2, 2, 3, 3, stride, with_bias,
        input.data,
        layer.weights.values,
        output.grad,
        expected_input_gradient,
        expected_weight_gradient,
        expected_bias_gradient
    );
    memcpy(input_data_snapshot, input.data, sizeof(input_data_snapshot));
    memcpy(
        output_data_snapshot,
        output.data,
        output_count * sizeof(float));
    memcpy(
        output_grad_snapshot,
        output.grad,
        output_count * sizeof(float));
    memcpy(
        weight_values_snapshot,
        layer.weights.values,
        sizeof(weight_values_snapshot));
    memcpy(
        weight_adam_snapshot,
        layer.weights.adam_m,
        sizeof(weight_adam_snapshot));
    if (with_bias) {
        memcpy(
            bias_values_snapshot,
            layer.bias.values,
            sizeof(bias_values_snapshot));
        memcpy(
            bias_adam_snapshot,
            layer.bias.adam_m,
            sizeof(bias_adam_snapshot));
    }
    if (vae_c_conv2d_backward(&layer, &input, &output) != VAE_C_OP_OK ||
        !float_array_close_double(
            input.grad,
            expected_input_gradient,
            18,
            3e-6) ||
        !float_array_close_double(
            layer.weights.gradients,
            expected_weight_gradient,
            36,
            3e-6) ||
        (with_bias &&
         !float_array_close_double(
             layer.bias.gradients,
             expected_bias_gradient,
             2,
             2e-6)) ||
        !float_array_equal(input.data, input_data_snapshot, 18) ||
        !float_array_equal(output.data, output_data_snapshot, output_count) ||
        !float_array_equal(output.grad, output_grad_snapshot, output_count) ||
        !float_array_equal(
            layer.weights.values,
            weight_values_snapshot,
            36) ||
        !float_array_equal(layer.weights.adam_m, weight_adam_snapshot, 72) ||
        (with_bias &&
         (!float_array_equal(layer.bias.values, bias_values_snapshot, 2) ||
          !float_array_equal(layer.bias.adam_m, bias_adam_snapshot, 4)))) {

        goto cleanup;
    }
    if (vae_c_conv2d_backward(&layer, &input, &output) != VAE_C_OP_OK) {
        goto cleanup;
    }
    for (index = 0; index < 18; index++) {
        if (!close_double(
                (double)input.grad[index],
                2.0 * expected_input_gradient[index],
                6e-6)) {

            goto cleanup;
        }
    }
    for (index = 0; index < 36; index++) {
        if (!close_double(
                (double)layer.weights.gradients[index],
                2.0 * expected_weight_gradient[index],
                6e-6)) {

            goto cleanup;
        }
    }
    if (with_bias) {
        for (index = 0; index < 2; index++) {
            if (!close_double(
                    (double)layer.bias.gradients[index],
                    2.0 * expected_bias_gradient[index],
                    4e-6)) {

                goto cleanup;
            }
        }
    }
    if (!float_array_equal(input.data, input_data_snapshot, 18) ||
        !float_array_equal(output.data, output_data_snapshot, output_count) ||
        !float_array_equal(output.grad, output_grad_snapshot, output_count) ||
        !float_array_equal(
            layer.weights.values,
            weight_values_snapshot,
            36) ||
        !float_array_equal(layer.weights.adam_m, weight_adam_snapshot, 72) ||
        (with_bias &&
         (!float_array_equal(layer.bias.values, bias_values_snapshot, 2) ||
          !float_array_equal(layer.bias.adam_m, bias_adam_snapshot, 4)))) {

        goto cleanup;
    }
    passed = 1;

cleanup:
    vae_c_tensor_free(&output);
    vae_c_tensor_free(&input);
    vae_c_conv2d_free(&layer);
    return passed;
}


static int test_analytic_backward(void)
{
    return
        run_analytic_backward_case(1, 1) &&
        run_analytic_backward_case(2, 0);
}


static double gradient_objective(
    const VAECConv2D *layer,
    const VAECTensor *input,
    VAECTensor *output,
    const float *upstream
)
{
    double objective = 0.0;
    size_t index;

    if (vae_c_conv2d_forward(layer, input, output) != VAE_C_OP_OK) {
        return NAN;
    }
    for (index = 0; index < output->shape.count; index++) {
        objective +=
            (double)output->data[index] * (double)upstream[index];
    }
    return objective;
}


static int run_gradient_check(size_t stride, int with_bias)
{
    const size_t input_height = stride == 1 ? 3 : 5;
    const size_t input_width = stride == 1 ? 4 : 5;
    const size_t input_dims[4] = {1, 2, input_height, input_width};
    size_t output_dims[4] = {1, 2, 0, 0};
    const float epsilon = 2.0e-3f;
    VAECConv2D layer;
    VAECTensor input;
    VAECTensor output;
    float upstream[32];
    float analytic_input[50];
    float analytic_weights[36];
    float analytic_bias[2];
    GradientStats input_stats = {0.0, 0.0, 0};
    GradientStats weight_stats = {0.0, 0.0, 0};
    GradientStats bias_stats = {0.0, 0.0, 0};
    size_t index;
    int passed = 1;

    output_dims[2] = reference_output_dimension(input_height, stride);
    output_dims[3] = reference_output_dimension(input_width, stride);
    vae_c_conv2d_init(&layer);
    vae_c_tensor_init(&input);
    vae_c_tensor_init(&output);
    if (vae_c_conv2d_allocate(
            &layer,
            2,
            2,
            stride,
            with_bias != 0,
            0,
            0) != VAE_C_OP_OK ||
        !make_tensor(4, input_dims, true, &input) ||
        !make_tensor(4, output_dims, true, &output)) {

        passed = 0;
        goto cleanup;
    }
    for (index = 0; index < input.shape.count; index++) {
        input.data[index] =
            (float)((int)(index % 17) - 8) * 0.08f +
            (float)index * 0.003f;
    }
    for (index = 0; index < layer.weights.shape.count; index++) {
        layer.weights.values[index] =
            (float)((int)(index % 13) - 6) * 0.05f +
            (float)index * 0.002f;
    }
    if (with_bias) {
        layer.bias.values[0] = 0.13f;
        layer.bias.values[1] = -0.21f;
    }
    for (index = 0; index < output.shape.count; index++) {
        upstream[index] =
            (float)((int)(index % 11) - 5) * 0.09f +
            (float)index * 0.001f;
        output.grad[index] = upstream[index];
    }
    if (vae_c_conv2d_forward(&layer, &input, &output) != VAE_C_OP_OK ||
        vae_c_tensor_zero_grad(&input) != VAE_C_STORAGE_OK ||
        vae_c_conv2d_zero_grad(&layer) != VAE_C_OP_OK ||
        vae_c_conv2d_backward(&layer, &input, &output) != VAE_C_OP_OK) {

        passed = 0;
        goto cleanup;
    }
    memcpy(
        analytic_input,
        input.grad,
        input.shape.count * sizeof(float));
    memcpy(
        analytic_weights,
        layer.weights.gradients,
        layer.weights.shape.count * sizeof(float));
    if (with_bias) {
        memcpy(analytic_bias, layer.bias.gradients, sizeof(analytic_bias));
    }

    for (index = 0; index < input.shape.count; index++) {
        const float original = input.data[index];
        double plus;
        double minus;
        double numeric;

        input.data[index] = original + epsilon;
        plus = gradient_objective(&layer, &input, &output, upstream);
        input.data[index] = original - epsilon;
        minus = gradient_objective(&layer, &input, &output, upstream);
        input.data[index] = original;
        numeric = (plus - minus) / (2.0 * (double)epsilon);
        update_gradient_stats(
            &input_stats,
            index,
            (double)analytic_input[index],
            numeric
        );
    }
    for (index = 0; index < layer.weights.shape.count; index++) {
        const float original = layer.weights.values[index];
        double plus;
        double minus;
        double numeric;

        layer.weights.values[index] = original + epsilon;
        plus = gradient_objective(&layer, &input, &output, upstream);
        layer.weights.values[index] = original - epsilon;
        minus = gradient_objective(&layer, &input, &output, upstream);
        layer.weights.values[index] = original;
        numeric = (plus - minus) / (2.0 * (double)epsilon);
        update_gradient_stats(
            &weight_stats,
            index,
            (double)analytic_weights[index],
            numeric
        );
    }
    if (with_bias) {
        for (index = 0; index < 2; index++) {
            const float original = layer.bias.values[index];
            double plus;
            double minus;
            double numeric;

            layer.bias.values[index] = original + epsilon;
            plus = gradient_objective(&layer, &input, &output, upstream);
            layer.bias.values[index] = original - epsilon;
            minus = gradient_objective(&layer, &input, &output, upstream);
            layer.bias.values[index] = original;
            numeric = (plus - minus) / (2.0 * (double)epsilon);
            update_gradient_stats(
                &bias_stats,
                index,
                (double)analytic_bias[index],
                numeric
            );
        }
    }
    printf(
        "CONV_STRIDE%zu_INPUT_GRAD_MAX_ABS=%.9g MAX_REL=%.9g "
        "WORST_INDEX=%zu\n",
        stride,
        input_stats.max_absolute,
        input_stats.max_relative,
        input_stats.worst_index
    );
    printf(
        "CONV_STRIDE%zu_WEIGHT_GRAD_MAX_ABS=%.9g MAX_REL=%.9g "
        "WORST_INDEX=%zu\n",
        stride,
        weight_stats.max_absolute,
        weight_stats.max_relative,
        weight_stats.worst_index
    );
    if (with_bias) {
        printf(
            "CONV_STRIDE%zu_BIAS_GRAD_MAX_ABS=%.9g MAX_REL=%.9g "
            "WORST_INDEX=%zu\n",
            stride,
            bias_stats.max_absolute,
            bias_stats.max_relative,
            bias_stats.worst_index
        );
    } else {
        printf("CONV_STRIDE%zu_BIAS_GRAD=not_applicable\n", stride);
    }
    passed =
        gradient_stats_pass(&input_stats) &&
        gradient_stats_pass(&weight_stats) &&
        (!with_bias || gradient_stats_pass(&bias_stats)) &&
        float_array_equal(output.grad, upstream, output.shape.count);

cleanup:
    vae_c_tensor_free(&output);
    vae_c_tensor_free(&input);
    vae_c_conv2d_free(&layer);
    return passed;
}


static int test_forward_errors_and_strong_guarantee(void)
{
    const size_t dims[4] = {1, 1, 3, 3};
    const size_t rank_two_dims[2] = {3, 3};
    const size_t batch_two_dims[4] = {2, 1, 3, 3};
    const size_t channels_two_dims[4] = {1, 2, 3, 3};
    const size_t wrong_spatial_dims[4] = {1, 1, 2, 3};
    const size_t huge_dims[4] = {1, 1, SIZE_MAX, 1};
    VAECConv2D layer;
    VAECConv2D corrupted;
    VAECTensor input;
    VAECTensor output;
    VAECTensor rank_two;
    VAECTensor batch_two;
    VAECTensor channels_two;
    VAECTensor wrong_spatial;
    VAECTensor alias_output;
    VAECTensor noncanonical;
    VAECTensor huge_input;
    VAECTensor huge_output;
    float output_snapshot[9];
    float input_snapshot[18];
    float output_grad_snapshot[9];
    float weight_snapshot[36];
    float bias_snapshot[4];
    float dummy = 0.0f;
    size_t index;
    int passed = 0;

    vae_c_conv2d_init(&layer);
    vae_c_tensor_init(&input);
    vae_c_tensor_init(&output);
    vae_c_tensor_init(&rank_two);
    vae_c_tensor_init(&batch_two);
    vae_c_tensor_init(&channels_two);
    vae_c_tensor_init(&wrong_spatial);
    vae_c_tensor_init(&huge_input);
    vae_c_tensor_init(&huge_output);
    if (vae_c_conv2d_allocate(&layer, 1, 1, 2, true, 0, 0) !=
            VAE_C_OP_OK ||
        !make_tensor(4, dims, true, &input) ||
        !make_tensor(4, dims, true, &output) ||
        !make_tensor(2, rank_two_dims, true, &rank_two) ||
        !make_tensor(4, batch_two_dims, true, &batch_two) ||
        !make_tensor(4, channels_two_dims, true, &channels_two) ||
        !make_tensor(4, wrong_spatial_dims, true, &wrong_spatial) ||
        !make_view(4, huge_dims, &dummy, NULL, &huge_input) ||
        !make_view(4, huge_dims, &dummy, NULL, &huge_output)) {

        goto cleanup;
    }
    for (index = 0; index < 9; index++) {
        input.data[index] = (float)index * 0.1f;
        input.grad[index] = (float)index * 0.2f;
        output.data[index] = 71.0f;
        output.grad[index] = (float)index * -0.3f;
        layer.weights.values[index] = (float)index * 0.04f;
        layer.weights.gradients[index] = (float)index * 0.05f;
        layer.weights.adam_m[index] = (float)index * 0.06f;
        layer.weights.adam_v[index] = (float)index * 0.07f;
    }
    layer.bias.values[0] = 0.1f;
    layer.bias.gradients[0] = 0.2f;
    layer.bias.adam_m[0] = 0.3f;
    layer.bias.adam_v[0] = 0.4f;
    memcpy(output_snapshot, output.data, sizeof(output_snapshot));
    memcpy(input_snapshot, input.data, sizeof(input_snapshot));
    memcpy(
        input_snapshot + 9,
        input.grad,
        9 * sizeof(float));
    memcpy(output_grad_snapshot, output.grad, sizeof(output_grad_snapshot));
    memcpy(weight_snapshot, layer.weights.owned_block, sizeof(weight_snapshot));
    memcpy(bias_snapshot, layer.bias.owned_block, sizeof(bias_snapshot));

    alias_output = output;
    alias_output.owns_memory = false;
    alias_output.owned_block = NULL;
    alias_output.owned_bytes = 0;
    alias_output.has_grad = false;
    alias_output.grad = NULL;
    alias_output.data = input.data;
    noncanonical = input;
    noncanonical.shape.strides[0]++;
    corrupted = layer;
    corrupted.with_bias = false;

    if (vae_c_conv2d_forward(NULL, &input, &output) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        vae_c_conv2d_forward(&layer, &input, &input) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        vae_c_conv2d_forward(&layer, &rank_two, &output) !=
            VAE_C_OP_SHAPE_MISMATCH ||
        vae_c_conv2d_forward(&layer, &input, &rank_two) !=
            VAE_C_OP_SHAPE_MISMATCH ||
        vae_c_conv2d_forward(&layer, &input, &batch_two) !=
            VAE_C_OP_SHAPE_MISMATCH ||
        vae_c_conv2d_forward(&layer, &channels_two, &output) !=
            VAE_C_OP_SHAPE_MISMATCH ||
        vae_c_conv2d_forward(&layer, &input, &channels_two) !=
            VAE_C_OP_SHAPE_MISMATCH ||
        vae_c_conv2d_forward(&layer, &input, &wrong_spatial) !=
            VAE_C_OP_SHAPE_MISMATCH ||
        vae_c_conv2d_forward(&layer, &noncanonical, &output) !=
            VAE_C_OP_INVALID_STATE ||
        vae_c_conv2d_forward(&layer, &input, &alias_output) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        vae_c_conv2d_forward(&corrupted, &input, &output) !=
            VAE_C_OP_INVALID_STATE ||
        vae_c_conv2d_forward(&layer, &huge_input, &output) !=
            VAE_C_OP_OVERFLOW ||
        vae_c_conv2d_forward(&layer, &input, &huge_output) !=
            VAE_C_OP_OVERFLOW) {

        goto cleanup;
    }
    alias_output.data = layer.weights.values;
    if (vae_c_conv2d_forward(&layer, &input, &alias_output) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        !float_array_equal(output.data, output_snapshot, 9) ||
        !float_array_equal(input.data, input_snapshot, 9) ||
        !float_array_equal(input.grad, input_snapshot + 9, 9) ||
        !float_array_equal(output.grad, output_grad_snapshot, 9) ||
        !float_array_equal(layer.weights.owned_block, weight_snapshot, 36) ||
        !float_array_equal(layer.bias.owned_block, bias_snapshot, 4)) {

        goto cleanup;
    }
    passed = 1;

cleanup:
    vae_c_tensor_free(&wrong_spatial);
    vae_c_tensor_free(&channels_two);
    vae_c_tensor_free(&batch_two);
    vae_c_tensor_free(&rank_two);
    vae_c_tensor_free(&output);
    vae_c_tensor_free(&input);
    vae_c_conv2d_free(&layer);
    return passed;
}


static int test_backward_errors_and_strong_guarantee(void)
{
    const size_t input_dims[4] = {1, 1, 3, 3};
    const size_t output_dims[4] = {1, 1, 2, 2};
    VAECConv2D layer;
    VAECConv2D corrupted;
    VAECTensor input;
    VAECTensor output;
    VAECTensor input_without_grad;
    VAECTensor output_without_grad;
    VAECTensor alias_output;
    VAECTensor wrong_output;
    float input_data_snapshot[9];
    float input_grad_snapshot[9];
    float output_data_snapshot[4];
    float output_grad_snapshot[4];
    float weight_snapshot[36];
    float bias_snapshot[4];
    size_t index;
    int passed = 0;

    vae_c_conv2d_init(&layer);
    vae_c_tensor_init(&input);
    vae_c_tensor_init(&output);
    vae_c_tensor_init(&input_without_grad);
    vae_c_tensor_init(&output_without_grad);
    vae_c_tensor_init(&wrong_output);
    if (vae_c_conv2d_allocate(&layer, 1, 1, 2, true, 0, 0) !=
            VAE_C_OP_OK ||
        !make_tensor(4, input_dims, true, &input) ||
        !make_tensor(4, output_dims, true, &output) ||
        !make_tensor(4, input_dims, false, &input_without_grad) ||
        !make_tensor(4, output_dims, false, &output_without_grad) ||
        !make_tensor(4, input_dims, true, &wrong_output)) {

        goto cleanup;
    }
    for (index = 0; index < 9; index++) {
        input.data[index] = (float)index * 0.11f;
        input.grad[index] = (float)index * -0.12f;
        layer.weights.values[index] = (float)index * 0.13f;
        layer.weights.gradients[index] = (float)index * 0.14f;
        layer.weights.adam_m[index] = (float)index * 0.15f;
        layer.weights.adam_v[index] = (float)index * 0.16f;
    }
    layer.bias.values[0] = 0.2f;
    layer.bias.gradients[0] = 0.3f;
    layer.bias.adam_m[0] = 0.4f;
    layer.bias.adam_v[0] = 0.5f;
    if (vae_c_conv2d_forward(&layer, &input, &output) != VAE_C_OP_OK) {
        goto cleanup;
    }
    for (index = 0; index < 4; index++) {
        output.grad[index] = (float)index * 0.17f - 0.2f;
    }
    memcpy(input_data_snapshot, input.data, sizeof(input_data_snapshot));
    memcpy(input_grad_snapshot, input.grad, sizeof(input_grad_snapshot));
    memcpy(output_data_snapshot, output.data, sizeof(output_data_snapshot));
    memcpy(output_grad_snapshot, output.grad, sizeof(output_grad_snapshot));
    memcpy(weight_snapshot, layer.weights.owned_block, sizeof(weight_snapshot));
    memcpy(bias_snapshot, layer.bias.owned_block, sizeof(bias_snapshot));

    corrupted = layer;
    corrupted.weights.gradients = corrupted.weights.values;
    if (vae_c_conv2d_backward(
            &layer,
            &input_without_grad,
            &output) != VAE_C_OP_GRADIENT_NOT_AVAILABLE ||
        vae_c_conv2d_backward(
            &layer,
            &input,
            &output_without_grad) != VAE_C_OP_GRADIENT_NOT_AVAILABLE ||
        vae_c_conv2d_backward(&layer, &input, &wrong_output) !=
            VAE_C_OP_SHAPE_MISMATCH ||
        vae_c_conv2d_backward(&corrupted, &input, &output) !=
            VAE_C_OP_INVALID_STATE) {

        goto cleanup;
    }
    corrupted = layer;
    corrupted.bias.gradients = NULL;
    if (vae_c_conv2d_backward(&corrupted, &input, &output) !=
            VAE_C_OP_INVALID_STATE) {

        goto cleanup;
    }
    alias_output = output;
    alias_output.owns_memory = false;
    alias_output.owned_block = NULL;
    alias_output.owned_bytes = 0;
    alias_output.grad = input.grad;
    if (vae_c_conv2d_backward(&layer, &input, &alias_output) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        !float_array_equal(input.data, input_data_snapshot, 9) ||
        !float_array_equal(input.grad, input_grad_snapshot, 9) ||
        !float_array_equal(output.data, output_data_snapshot, 4) ||
        !float_array_equal(output.grad, output_grad_snapshot, 4) ||
        !float_array_equal(layer.weights.owned_block, weight_snapshot, 36) ||
        !float_array_equal(layer.bias.owned_block, bias_snapshot, 4)) {

        goto cleanup;
    }
    passed = 1;

cleanup:
    vae_c_tensor_free(&wrong_output);
    vae_c_tensor_free(&output_without_grad);
    vae_c_tensor_free(&input_without_grad);
    vae_c_tensor_free(&output);
    vae_c_tensor_free(&input);
    vae_c_conv2d_free(&layer);
    return passed;
}


int main(void)
{
    int failures = 0;

    failures += report_check(
        "conv2d_lifecycle_and_oihw_layout",
        test_lifecycle_and_layout()
    );
    failures += report_check(
        "conv2d_single_channel_cross_correlation",
        test_single_channel_forward()
    );
    failures += report_check(
        "conv2d_multichannel_bias_and_stride_shapes",
        test_multichannel_forward_and_shapes()
    );
    failures += report_check(
        "conv2d_analytic_backward_and_accumulation",
        test_analytic_backward()
    );
    failures += report_check(
        "conv2d_stride1_gradient_check",
        run_gradient_check(1, 1)
    );
    failures += report_check(
        "conv2d_stride2_gradient_check",
        run_gradient_check(2, 0)
    );
    failures += report_check(
        "conv2d_forward_errors_and_strong_guarantee",
        test_forward_errors_and_strong_guarantee()
    );
    failures += report_check(
        "conv2d_backward_errors_and_strong_guarantee",
        test_backward_errors_and_strong_guarantee()
    );

    if (failures != 0) {
        fprintf(stderr, "%d VAE C CONV2D TEST(S) FAILED\n", failures);
        return 1;
    }
    printf("ALL VAE C CONV2D TESTS PASSED\n");
    return 0;
}
