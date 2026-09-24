#include "vae_c_dense.h"
#include "vae_c_ops.h"
#include "vae_c_parameter.h"
#include "vae_c_tensor.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>


typedef struct {
    double max_absolute;
    double max_relative;
    size_t worst_index;
} GradientCheckStats;


static int report_check(const char *name, int passed)
{
    if (passed) {
        printf("[PASS] %s\n", name);
        return 0;
    }

    fprintf(stderr, "[FAIL] %s\n", name);
    return 1;
}


static int close_enough(float actual, float expected, float tolerance)
{
    return fabs((double)actual - (double)expected) <= (double)tolerance;
}


static int array_close(
    const float *actual,
    const float *expected,
    size_t count,
    float tolerance
)
{
    size_t index;

    if (actual == NULL || expected == NULL) {
        return 0;
    }

    for (index = 0; index < count; index++) {
        if (!close_enough(actual[index], expected[index], tolerance)) {
            return 0;
        }
    }

    return 1;
}


static int array_equal(
    const float *left,
    const float *right,
    size_t count
)
{
    size_t index;

    if (left == NULL || right == NULL) {
        return 0;
    }

    for (index = 0; index < count; index++) {
        if (left[index] != right[index]) {
            return 0;
        }
    }

    return 1;
}


static int array_all_equal(
    const float *values,
    size_t count,
    float expected
)
{
    size_t index;

    if (values == NULL) {
        return 0;
    }

    for (index = 0; index < count; index++) {
        if (values[index] != expected) {
            return 0;
        }
    }

    return 1;
}


static int make_tensor(
    size_t rank,
    const size_t *dims,
    bool with_gradient,
    VAECTensor *tensor
)
{
    VAECShape shape;

    if (tensor == NULL) {
        return 0;
    }

    vae_c_tensor_init(tensor);
    return
        vae_c_shape_make(rank, dims, &shape) == VAE_C_STORAGE_OK &&
        vae_c_tensor_allocate(
            tensor,
            &shape,
            with_gradient) == VAE_C_STORAGE_OK;
}


static int parameter_fields_equal(
    const VAECParameter *left,
    const VAECParameter *right
)
{
    return
        left != NULL &&
        right != NULL &&
        left->shape.rank == right->shape.rank &&
        left->shape.count == right->shape.count &&
        memcmp(
            left->shape.dims,
            right->shape.dims,
            sizeof(left->shape.dims)) == 0 &&
        memcmp(
            left->shape.strides,
            right->shape.strides,
            sizeof(left->shape.strides)) == 0 &&
        left->parameter_key == right->parameter_key &&
        left->values == right->values &&
        left->gradients == right->gradients &&
        left->adam_m == right->adam_m &&
        left->adam_v == right->adam_v &&
        left->owned_block == right->owned_block &&
        left->owned_bytes == right->owned_bytes;
}


static int dense_fields_equal(
    const VAECDense *left,
    const VAECDense *right
)
{
    return
        left != NULL &&
        right != NULL &&
        left->in_features == right->in_features &&
        left->out_features == right->out_features &&
        left->with_bias == right->with_bias &&
        parameter_fields_equal(&left->weights, &right->weights) &&
        parameter_fields_equal(&left->bias, &right->bias);
}


static void update_gradient_stats(
    GradientCheckStats *stats,
    size_t index,
    double analytic,
    double numeric
)
{
    double absolute_error;
    double denominator;
    double relative_error;

    absolute_error = fabs(analytic - numeric);
    denominator = fabs(analytic) + fabs(numeric);
    if (denominator < 1.0e-12) {
        denominator = 1.0e-12;
    }
    relative_error = absolute_error / denominator;

    if (absolute_error > stats->max_absolute) {
        stats->max_absolute = absolute_error;
        stats->worst_index = index;
    }
    if (relative_error > stats->max_relative) {
        stats->max_relative = relative_error;
    }
}


static int gradient_within_tolerance(
    double analytic,
    double numeric,
    double absolute_tolerance,
    double relative_tolerance
)
{
    double absolute_error = fabs(analytic - numeric);
    double scale = fabs(analytic);

    if (fabs(numeric) > scale) {
        scale = fabs(numeric);
    }

    return
        absolute_error <= absolute_tolerance +
            relative_tolerance * scale;
}


static int test_dense_analytic_and_lifecycle(void)
{
    const size_t input_dims[2] = {2, 3};
    const size_t output_dims[2] = {2, 2};
    const float input_values[6] = {
        1.0f, 2.0f, -1.0f,
        0.5f, -2.0f, 3.0f
    };
    const float weight_values[6] = {
        0.2f, -0.5f, 1.0f,
        -1.0f, 0.25f, 0.5f
    };
    const float bias_values[2] = {0.1f, -0.2f};
    const float output_gradients[4] = {1.5f, -0.5f, 2.0f, 1.0f};
    const float expected_output[4] = {-1.7f, -1.2f, 4.2f, 0.3f};
    const float expected_input_grad[6] = {
        0.8f, -0.875f, 1.25f,
        -0.6f, -0.75f, 2.5f
    };
    const float expected_weight_grad[6] = {
        2.5f, -1.0f, 4.5f,
        0.0f, -3.0f, 3.5f
    };
    const float expected_bias_grad[2] = {3.5f, 0.5f};
    VAECDense layer;
    VAECTensor input;
    VAECTensor output;
    float output_grad_snapshot[4];
    float output_data_snapshot[4];
    size_t index;

    vae_c_dense_init(&layer);
    vae_c_tensor_init(&input);
    vae_c_tensor_init(&output);

    if (!vae_c_dense_is_empty(&layer) ||
        vae_c_dense_allocate(
            &layer,
            3,
            2,
            true,
            UINT64_C(0),
            UINT64_C(0)) != VAE_C_OP_OK ||
        vae_c_dense_is_empty(&layer) ||
        layer.weights.parameter_key != UINT64_C(0) ||
        layer.bias.parameter_key != UINT64_C(0) ||
        !make_tensor(2, input_dims, true, &input) ||
        !make_tensor(2, output_dims, true, &output)) {

        vae_c_tensor_free(&output);
        vae_c_tensor_free(&input);
        vae_c_dense_free(&layer);
        return 0;
    }

    memcpy(input.data, input_values, sizeof(input_values));
    memcpy(layer.weights.values, weight_values, sizeof(weight_values));
    memcpy(layer.bias.values, bias_values, sizeof(bias_values));
    memcpy(output.grad, output_gradients, sizeof(output_gradients));
    for (index = 0; index < layer.weights.shape.count; index++) {
        layer.weights.adam_m[index] = 7.0f;
        layer.weights.adam_v[index] = 8.0f;
    }
    for (index = 0; index < layer.bias.shape.count; index++) {
        layer.bias.adam_m[index] = 9.0f;
        layer.bias.adam_v[index] = 10.0f;
    }

    if (vae_c_dense_forward(&layer, &input, &output) != VAE_C_OP_OK ||
        !array_close(output.data, expected_output, 4, 1.0e-6f) ||
        !array_equal(input.data, input_values, 6) ||
        !array_equal(output.grad, output_gradients, 4)) {

        vae_c_tensor_free(&output);
        vae_c_tensor_free(&input);
        vae_c_dense_free(&layer);
        return 0;
    }

    memcpy(output_data_snapshot, output.data, sizeof(output_data_snapshot));
    memcpy(output_grad_snapshot, output.grad, sizeof(output_grad_snapshot));

    if (vae_c_dense_backward(&layer, &input, &output) != VAE_C_OP_OK ||
        !array_close(input.grad, expected_input_grad, 6, 1.0e-6f) ||
        !array_close(
            layer.weights.gradients,
            expected_weight_grad,
            6,
            1.0e-6f) ||
        !array_close(
            layer.bias.gradients,
            expected_bias_grad,
            2,
            1.0e-6f) ||
        !array_equal(output.data, output_data_snapshot, 4) ||
        !array_equal(output.grad, output_grad_snapshot, 4) ||
        !array_equal(input.data, input_values, 6) ||
        !array_equal(layer.weights.values, weight_values, 6) ||
        !array_equal(layer.bias.values, bias_values, 2)) {

        vae_c_tensor_free(&output);
        vae_c_tensor_free(&input);
        vae_c_dense_free(&layer);
        return 0;
    }

    if (vae_c_dense_backward(&layer, &input, &output) != VAE_C_OP_OK) {
        vae_c_tensor_free(&output);
        vae_c_tensor_free(&input);
        vae_c_dense_free(&layer);
        return 0;
    }

    for (index = 0; index < 6; index++) {
        if (!close_enough(
                input.grad[index],
                2.0f * expected_input_grad[index],
                2.0e-6f) ||
            !close_enough(
                layer.weights.gradients[index],
                2.0f * expected_weight_grad[index],
                2.0e-6f)) {

            vae_c_tensor_free(&output);
            vae_c_tensor_free(&input);
            vae_c_dense_free(&layer);
            return 0;
        }
    }
    for (index = 0; index < 2; index++) {
        if (!close_enough(
                layer.bias.gradients[index],
                2.0f * expected_bias_grad[index],
                2.0e-6f)) {

            vae_c_tensor_free(&output);
            vae_c_tensor_free(&input);
            vae_c_dense_free(&layer);
            return 0;
        }
    }

    if (vae_c_dense_zero_grad(&layer) != VAE_C_OP_OK ||
        !array_all_equal(layer.weights.gradients, 6, 0.0f) ||
        !array_all_equal(layer.bias.gradients, 2, 0.0f) ||
        !array_equal(layer.weights.values, weight_values, 6) ||
        !array_equal(layer.bias.values, bias_values, 2) ||
        !array_all_equal(layer.weights.adam_m, 6, 7.0f) ||
        !array_all_equal(layer.weights.adam_v, 6, 8.0f) ||
        !array_all_equal(layer.bias.adam_m, 2, 9.0f) ||
        !array_all_equal(layer.bias.adam_v, 2, 10.0f)) {

        vae_c_tensor_free(&output);
        vae_c_tensor_free(&input);
        vae_c_dense_free(&layer);
        return 0;
    }

    vae_c_tensor_free(&output);
    vae_c_tensor_free(&input);
    vae_c_dense_free(&layer);
    vae_c_dense_free(&layer);

    return
        vae_c_dense_is_empty(&layer) &&
        vae_c_dense_zero_grad(&layer) == VAE_C_OP_OK;
}


static int test_dense_without_bias(void)
{
    const size_t input_dims[2] = {1, 2};
    const size_t output_dims[2] = {1, 1};
    VAECDense layer;
    VAECTensor input;
    VAECTensor output;
    int passed;

    vae_c_dense_init(&layer);
    vae_c_tensor_init(&input);
    vae_c_tensor_init(&output);

    if (vae_c_dense_allocate(
            &layer,
            2,
            1,
            false,
            UINT64_C(4),
            UINT64_C(5)) != VAE_C_OP_OK ||
        !vae_c_parameter_is_empty(&layer.bias) ||
        !make_tensor(2, input_dims, true, &input) ||
        !make_tensor(2, output_dims, true, &output)) {

        vae_c_tensor_free(&output);
        vae_c_tensor_free(&input);
        vae_c_dense_free(&layer);
        return 0;
    }

    input.data[0] = 3.0f;
    input.data[1] = 4.0f;
    layer.weights.values[0] = 2.0f;
    layer.weights.values[1] = -1.0f;
    output.grad[0] = 0.5f;

    passed =
        vae_c_dense_forward(&layer, &input, &output) == VAE_C_OP_OK &&
        close_enough(output.data[0], 2.0f, 1.0e-6f) &&
        vae_c_dense_backward(&layer, &input, &output) == VAE_C_OP_OK &&
        close_enough(input.grad[0], 1.0f, 1.0e-6f) &&
        close_enough(input.grad[1], -0.5f, 1.0e-6f) &&
        close_enough(layer.weights.gradients[0], 1.5f, 1.0e-6f) &&
        close_enough(layer.weights.gradients[1], 2.0f, 1.0e-6f);

    vae_c_tensor_free(&output);
    vae_c_tensor_free(&input);
    vae_c_dense_free(&layer);
    return passed;
}


static int test_relu_analytic(void)
{
    const size_t dims[1] = {5};
    const float input_values[5] = {-2.0f, 0.0f, 3.0f, -0.5f, 1.0f};
    const float expected_output[5] = {0.0f, 0.0f, 3.0f, 0.0f, 1.0f};
    const float output_gradients[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    const float expected_first_grad[5] = {
        0.25f, 0.25f, 3.25f, 0.25f, 5.25f
    };
    const float expected_second_grad[5] = {
        0.25f, 0.25f, 6.25f, 0.25f, 10.25f
    };
    VAECTensor input;
    VAECTensor output;
    size_t index;
    int passed;

    if (!make_tensor(1, dims, true, &input) ||
        !make_tensor(1, dims, true, &output)) {

        vae_c_tensor_free(&output);
        vae_c_tensor_free(&input);
        return 0;
    }

    memcpy(input.data, input_values, sizeof(input_values));
    memcpy(output.grad, output_gradients, sizeof(output_gradients));
    for (index = 0; index < 5; index++) {
        input.grad[index] = 0.25f;
    }

    passed =
        vae_c_relu_forward(&input, &output) == VAE_C_OP_OK &&
        array_equal(output.data, expected_output, 5) &&
        array_equal(input.data, input_values, 5) &&
        vae_c_relu_backward(&input, &output) == VAE_C_OP_OK &&
        array_equal(input.grad, expected_first_grad, 5) &&
        array_equal(output.grad, output_gradients, 5) &&
        array_equal(input.data, input_values, 5) &&
        vae_c_relu_backward(&input, &output) == VAE_C_OP_OK &&
        array_equal(input.grad, expected_second_grad, 5) &&
        array_equal(output.grad, output_gradients, 5);

    vae_c_tensor_free(&output);
    vae_c_tensor_free(&input);
    return passed;
}


static int test_gap_analytic(void)
{
    const size_t input_dims[4] = {1, 2, 2, 2};
    const size_t output_dims[2] = {1, 2};
    const size_t second_input_dims[4] = {2, 1, 1, 3};
    const size_t second_output_dims[2] = {2, 1};
    const float input_values[8] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        2.0f, 4.0f, 6.0f, 8.0f
    };
    const float expected_output[2] = {2.5f, 5.0f};
    const float output_gradients[2] = {2.0f, -4.0f};
    const float expected_first_grad[8] = {
        0.5f, 0.5f, 0.5f, 0.5f,
        -1.0f, -1.0f, -1.0f, -1.0f
    };
    const float expected_second_grad[8] = {
        1.0f, 1.0f, 1.0f, 1.0f,
        -2.0f, -2.0f, -2.0f, -2.0f
    };
    const float second_values[6] = {1.0f, 2.0f, 6.0f, 3.0f, 6.0f, 9.0f};
    const float second_expected[2] = {3.0f, 6.0f};
    VAECTensor input;
    VAECTensor output;
    VAECTensor second_input;
    VAECTensor second_output;
    int passed;

    vae_c_tensor_init(&input);
    vae_c_tensor_init(&output);
    vae_c_tensor_init(&second_input);
    vae_c_tensor_init(&second_output);

    if (!make_tensor(4, input_dims, true, &input) ||
        !make_tensor(2, output_dims, true, &output) ||
        !make_tensor(4, second_input_dims, false, &second_input) ||
        !make_tensor(2, second_output_dims, false, &second_output)) {

        vae_c_tensor_free(&second_output);
        vae_c_tensor_free(&second_input);
        vae_c_tensor_free(&output);
        vae_c_tensor_free(&input);
        return 0;
    }

    memcpy(input.data, input_values, sizeof(input_values));
    memcpy(output.grad, output_gradients, sizeof(output_gradients));
    memcpy(second_input.data, second_values, sizeof(second_values));

    passed =
        vae_c_global_avg_pool2d_forward(
            &input,
            &output) == VAE_C_OP_OK &&
        array_close(output.data, expected_output, 2, 1.0e-6f) &&
        vae_c_global_avg_pool2d_backward(
            &input,
            &output) == VAE_C_OP_OK &&
        array_close(input.grad, expected_first_grad, 8, 1.0e-6f) &&
        array_equal(output.grad, output_gradients, 2) &&
        vae_c_global_avg_pool2d_backward(
            &input,
            &output) == VAE_C_OP_OK &&
        array_close(input.grad, expected_second_grad, 8, 1.0e-6f) &&
        array_equal(output.grad, output_gradients, 2) &&
        vae_c_global_avg_pool2d_forward(
            &second_input,
            &second_output) == VAE_C_OP_OK &&
        array_close(
            second_output.data,
            second_expected,
            2,
            1.0e-6f);

    vae_c_tensor_free(&second_output);
    vae_c_tensor_free(&second_input);
    vae_c_tensor_free(&output);
    vae_c_tensor_free(&input);
    return passed;
}


static double dense_objective(
    const VAECDense *layer,
    const VAECTensor *input,
    VAECTensor *output,
    const float *upstream
)
{
    double objective = 0.0;
    size_t index;

    if (vae_c_dense_forward(layer, input, output) != VAE_C_OP_OK) {
        return NAN;
    }

    for (index = 0; index < output->shape.count; index++) {
        objective +=
            (double)output->data[index] * (double)upstream[index];
    }

    return objective;
}


static int test_dense_gradient_check(void)
{
    const size_t input_dims[2] = {2, 2};
    const size_t output_dims[2] = {2, 2};
    const float input_values[4] = {0.2f, -0.4f, 0.7f, 0.1f};
    const float weight_values[4] = {0.3f, -0.2f, 0.5f, 0.4f};
    const float bias_values[2] = {0.1f, -0.3f};
    const float upstream[4] = {0.6f, -0.8f, 0.2f, 0.9f};
    const float epsilon = 1.0e-3f;
    const double absolute_tolerance = 8.0e-5;
    const double relative_tolerance = 8.0e-4;
    VAECDense layer;
    VAECTensor input;
    VAECTensor output;
    GradientCheckStats stats;
    size_t flat_index;
    size_t index;
    int passed;

    vae_c_dense_init(&layer);
    vae_c_tensor_init(&input);
    vae_c_tensor_init(&output);
    stats.max_absolute = 0.0;
    stats.max_relative = 0.0;
    stats.worst_index = 0;
    flat_index = 0;
    passed = 1;

    if (vae_c_dense_allocate(
            &layer,
            2,
            2,
            true,
            UINT64_C(11),
            UINT64_C(12)) != VAE_C_OP_OK ||
        !make_tensor(2, input_dims, true, &input) ||
        !make_tensor(2, output_dims, true, &output)) {

        vae_c_tensor_free(&output);
        vae_c_tensor_free(&input);
        vae_c_dense_free(&layer);
        return 0;
    }

    memcpy(input.data, input_values, sizeof(input_values));
    memcpy(layer.weights.values, weight_values, sizeof(weight_values));
    memcpy(layer.bias.values, bias_values, sizeof(bias_values));
    memcpy(output.grad, upstream, sizeof(upstream));

    if (vae_c_dense_forward(&layer, &input, &output) != VAE_C_OP_OK ||
        vae_c_dense_backward(&layer, &input, &output) != VAE_C_OP_OK) {

        vae_c_tensor_free(&output);
        vae_c_tensor_free(&input);
        vae_c_dense_free(&layer);
        return 0;
    }

    for (index = 0; index < input.shape.count; index++, flat_index++) {
        float original = input.data[index];
        double plus;
        double minus;
        double numeric;
        double analytic = (double)input.grad[index];

        input.data[index] = original + epsilon;
        plus = dense_objective(&layer, &input, &output, upstream);
        input.data[index] = original - epsilon;
        minus = dense_objective(&layer, &input, &output, upstream);
        input.data[index] = original;
        numeric = (plus - minus) / (2.0 * (double)epsilon);
        update_gradient_stats(
            &stats,
            flat_index,
            analytic,
            numeric
        );
        if (!gradient_within_tolerance(
                analytic,
                numeric,
                absolute_tolerance,
                relative_tolerance)) {

            passed = 0;
        }
    }

    for (index = 0;
         index < layer.weights.shape.count;
         index++, flat_index++) {

        float original = layer.weights.values[index];
        double plus;
        double minus;
        double numeric;
        double analytic = (double)layer.weights.gradients[index];

        layer.weights.values[index] = original + epsilon;
        plus = dense_objective(&layer, &input, &output, upstream);
        layer.weights.values[index] = original - epsilon;
        minus = dense_objective(&layer, &input, &output, upstream);
        layer.weights.values[index] = original;
        numeric = (plus - minus) / (2.0 * (double)epsilon);
        update_gradient_stats(
            &stats,
            flat_index,
            analytic,
            numeric
        );
        if (!gradient_within_tolerance(
                analytic,
                numeric,
                absolute_tolerance,
                relative_tolerance)) {

            passed = 0;
        }
    }

    for (index = 0;
         index < layer.bias.shape.count;
         index++, flat_index++) {

        float original = layer.bias.values[index];
        double plus;
        double minus;
        double numeric;
        double analytic = (double)layer.bias.gradients[index];

        layer.bias.values[index] = original + epsilon;
        plus = dense_objective(&layer, &input, &output, upstream);
        layer.bias.values[index] = original - epsilon;
        minus = dense_objective(&layer, &input, &output, upstream);
        layer.bias.values[index] = original;
        numeric = (plus - minus) / (2.0 * (double)epsilon);
        update_gradient_stats(
            &stats,
            flat_index,
            analytic,
            numeric
        );
        if (!gradient_within_tolerance(
                analytic,
                numeric,
                absolute_tolerance,
                relative_tolerance)) {

            passed = 0;
        }
    }

    printf(
        "DENSE_GRAD_CHECK_MAX_ABS=%.9g MAX_REL=%.9g WORST_INDEX=%zu\n",
        stats.max_absolute,
        stats.max_relative,
        stats.worst_index
    );

    vae_c_tensor_free(&output);
    vae_c_tensor_free(&input);
    vae_c_dense_free(&layer);
    return passed;
}


static double relu_objective(
    const VAECTensor *input,
    VAECTensor *output,
    const float *upstream
)
{
    double objective = 0.0;
    size_t index;

    if (vae_c_relu_forward(input, output) != VAE_C_OP_OK) {
        return NAN;
    }

    for (index = 0; index < output->shape.count; index++) {
        objective +=
            (double)output->data[index] * (double)upstream[index];
    }

    return objective;
}


static int test_relu_gradient_check(void)
{
    const size_t dims[1] = {4};
    const float input_values[4] = {-1.2f, -0.4f, 0.6f, 1.3f};
    const float upstream[4] = {0.7f, -0.2f, 1.1f, -0.9f};
    const float epsilon = 1.0e-3f;
    const double absolute_tolerance = 8.0e-5;
    const double relative_tolerance = 8.0e-4;
    VAECTensor input;
    VAECTensor output;
    GradientCheckStats stats;
    size_t index;
    int passed;

    stats.max_absolute = 0.0;
    stats.max_relative = 0.0;
    stats.worst_index = 0;
    passed = 1;

    if (!make_tensor(1, dims, true, &input) ||
        !make_tensor(1, dims, true, &output)) {

        vae_c_tensor_free(&output);
        vae_c_tensor_free(&input);
        return 0;
    }

    memcpy(input.data, input_values, sizeof(input_values));
    memcpy(output.grad, upstream, sizeof(upstream));

    if (vae_c_relu_forward(&input, &output) != VAE_C_OP_OK ||
        vae_c_relu_backward(&input, &output) != VAE_C_OP_OK) {

        vae_c_tensor_free(&output);
        vae_c_tensor_free(&input);
        return 0;
    }

    for (index = 0; index < input.shape.count; index++) {
        float original = input.data[index];
        double plus;
        double minus;
        double numeric;
        double analytic = (double)input.grad[index];

        input.data[index] = original + epsilon;
        plus = relu_objective(&input, &output, upstream);
        input.data[index] = original - epsilon;
        minus = relu_objective(&input, &output, upstream);
        input.data[index] = original;
        numeric = (plus - minus) / (2.0 * (double)epsilon);
        update_gradient_stats(&stats, index, analytic, numeric);
        if (!gradient_within_tolerance(
                analytic,
                numeric,
                absolute_tolerance,
                relative_tolerance)) {

            passed = 0;
        }
    }

    printf(
        "RELU_GRAD_CHECK_MAX_ABS=%.9g MAX_REL=%.9g WORST_INDEX=%zu\n",
        stats.max_absolute,
        stats.max_relative,
        stats.worst_index
    );

    vae_c_tensor_free(&output);
    vae_c_tensor_free(&input);
    return passed;
}


static double gap_objective(
    const VAECTensor *input,
    VAECTensor *output,
    const float *upstream
)
{
    double objective = 0.0;
    size_t index;

    if (vae_c_global_avg_pool2d_forward(input, output) != VAE_C_OP_OK) {
        return NAN;
    }

    for (index = 0; index < output->shape.count; index++) {
        objective +=
            (double)output->data[index] * (double)upstream[index];
    }

    return objective;
}


static int test_gap_gradient_check(void)
{
    const size_t input_dims[4] = {1, 1, 2, 3};
    const size_t output_dims[2] = {1, 1};
    const float input_values[6] = {
        -0.7f, 0.2f, 1.1f, 0.5f, -0.3f, 0.9f
    };
    const float upstream[1] = {1.7f};
    const float epsilon = 1.0e-3f;
    const double absolute_tolerance = 8.0e-5;
    const double relative_tolerance = 8.0e-4;
    VAECTensor input;
    VAECTensor output;
    GradientCheckStats stats;
    size_t index;
    int passed;

    stats.max_absolute = 0.0;
    stats.max_relative = 0.0;
    stats.worst_index = 0;
    passed = 1;

    if (!make_tensor(4, input_dims, true, &input) ||
        !make_tensor(2, output_dims, true, &output)) {

        vae_c_tensor_free(&output);
        vae_c_tensor_free(&input);
        return 0;
    }

    memcpy(input.data, input_values, sizeof(input_values));
    memcpy(output.grad, upstream, sizeof(upstream));

    if (vae_c_global_avg_pool2d_forward(
            &input,
            &output) != VAE_C_OP_OK ||
        vae_c_global_avg_pool2d_backward(
            &input,
            &output) != VAE_C_OP_OK) {

        vae_c_tensor_free(&output);
        vae_c_tensor_free(&input);
        return 0;
    }

    for (index = 0; index < input.shape.count; index++) {
        float original = input.data[index];
        double plus;
        double minus;
        double numeric;
        double analytic = (double)input.grad[index];

        input.data[index] = original + epsilon;
        plus = gap_objective(&input, &output, upstream);
        input.data[index] = original - epsilon;
        minus = gap_objective(&input, &output, upstream);
        input.data[index] = original;
        numeric = (plus - minus) / (2.0 * (double)epsilon);
        update_gradient_stats(&stats, index, analytic, numeric);
        if (!gradient_within_tolerance(
                analytic,
                numeric,
                absolute_tolerance,
                relative_tolerance)) {

            passed = 0;
        }
    }

    printf(
        "GAP_GRAD_CHECK_MAX_ABS=%.9g MAX_REL=%.9g WORST_INDEX=%zu\n",
        stats.max_absolute,
        stats.max_relative,
        stats.worst_index
    );

    vae_c_tensor_free(&output);
    vae_c_tensor_free(&input);
    return passed;
}


static int test_dense_errors_and_strong_guarantee(void)
{
    const size_t good_dims[2] = {2, 2};
    const size_t bad_input_dims[1] = {4};
    const size_t bad_output_dims[1] = {4};
    const size_t bad_feature_dims[2] = {2, 3};
    const size_t bad_batch_dims[2] = {1, 2};
    VAECDense layer;
    VAECDense empty;
    VAECDense snapshot;
    VAECDense corrupted;
    VAECTensor input;
    VAECTensor output;
    VAECTensor bad_input;
    VAECTensor bad_output;
    VAECTensor bad_feature;
    VAECTensor bad_batch;
    VAECTensor data_only_input;
    VAECTensor data_only_output;
    VAECTensor alias;
    float input_grad_snapshot[4];
    float weight_grad_snapshot[4];
    float bias_grad_snapshot[2];
    size_t index;
    int passed;

    vae_c_dense_init(&layer);
    vae_c_dense_init(&empty);
    vae_c_tensor_init(&input);
    vae_c_tensor_init(&output);
    vae_c_tensor_init(&bad_input);
    vae_c_tensor_init(&bad_output);
    vae_c_tensor_init(&bad_feature);
    vae_c_tensor_init(&bad_batch);
    vae_c_tensor_init(&data_only_input);
    vae_c_tensor_init(&data_only_output);

    snapshot = empty;
    if (vae_c_dense_allocate(
            &empty,
            0,
            2,
            true,
            UINT64_C(1),
            UINT64_C(2)) != VAE_C_OP_INVALID_ARGUMENT ||
        !dense_fields_equal(&empty, &snapshot) ||
        vae_c_dense_allocate(
            NULL,
            2,
            2,
            true,
            UINT64_C(1),
            UINT64_C(2)) != VAE_C_OP_INVALID_ARGUMENT ||
        vae_c_dense_zero_grad(NULL) != VAE_C_OP_INVALID_ARGUMENT ||
        vae_c_dense_forward(NULL, NULL, NULL) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        vae_c_dense_backward(NULL, NULL, NULL) !=
            VAE_C_OP_INVALID_ARGUMENT) {

        return 0;
    }

    if (vae_c_dense_allocate(
            &layer,
            2,
            2,
            true,
            UINT64_C(0),
            UINT64_C(0)) != VAE_C_OP_OK ||
        !make_tensor(2, good_dims, true, &input) ||
        !make_tensor(2, good_dims, true, &output) ||
        !make_tensor(1, bad_input_dims, true, &bad_input) ||
        !make_tensor(1, bad_output_dims, true, &bad_output) ||
        !make_tensor(2, bad_feature_dims, true, &bad_feature) ||
        !make_tensor(2, bad_batch_dims, true, &bad_batch) ||
        !make_tensor(2, good_dims, false, &data_only_input) ||
        !make_tensor(2, good_dims, false, &data_only_output)) {

        vae_c_tensor_free(&data_only_output);
        vae_c_tensor_free(&data_only_input);
        vae_c_tensor_free(&bad_batch);
        vae_c_tensor_free(&bad_feature);
        vae_c_tensor_free(&bad_output);
        vae_c_tensor_free(&bad_input);
        vae_c_tensor_free(&output);
        vae_c_tensor_free(&input);
        vae_c_dense_free(&layer);
        return 0;
    }

    for (index = 0; index < 4; index++) {
        input.data[index] = (float)(index + 1);
        input.grad[index] = 11.0f;
        output.data[index] = 13.0f;
        output.grad[index] = 2.0f;
        layer.weights.values[index] = 0.25f;
        layer.weights.gradients[index] = 17.0f;
    }
    for (index = 0; index < 2; index++) {
        layer.bias.values[index] = 0.5f;
        layer.bias.gradients[index] = 19.0f;
    }

    snapshot = layer;
    if (vae_c_dense_allocate(
            &layer,
            2,
            2,
            false,
            UINT64_C(8),
            UINT64_C(9)) != VAE_C_OP_OBJECT_NOT_EMPTY ||
        !dense_fields_equal(&layer, &snapshot)) {

        passed = 0;
        goto cleanup;
    }

    corrupted = layer;
    corrupted.in_features = 3;
    if (vae_c_dense_forward(&corrupted, &input, &output) !=
            VAE_C_OP_INVALID_STATE ||
        !array_all_equal(output.data, 4, 13.0f)) {

        passed = 0;
        goto cleanup;
    }

    if (vae_c_dense_forward(&layer, &bad_input, &output) !=
            VAE_C_OP_SHAPE_MISMATCH ||
        !array_all_equal(output.data, 4, 13.0f) ||
        vae_c_dense_forward(&layer, &input, &bad_output) !=
            VAE_C_OP_SHAPE_MISMATCH ||
        vae_c_dense_forward(&layer, &bad_feature, &output) !=
            VAE_C_OP_SHAPE_MISMATCH ||
        !array_all_equal(output.data, 4, 13.0f) ||
        vae_c_dense_forward(&layer, &input, &bad_batch) !=
            VAE_C_OP_SHAPE_MISMATCH ||
        vae_c_dense_forward(&layer, &input, &input) !=
            VAE_C_OP_INVALID_ARGUMENT) {

        passed = 0;
        goto cleanup;
    }

    alias = input;
    if (vae_c_dense_forward(&layer, &input, &alias) !=
            VAE_C_OP_INVALID_ARGUMENT) {

        passed = 0;
        goto cleanup;
    }

    memcpy(input_grad_snapshot, input.grad, sizeof(input_grad_snapshot));
    memcpy(
        weight_grad_snapshot,
        layer.weights.gradients,
        sizeof(weight_grad_snapshot)
    );
    memcpy(
        bias_grad_snapshot,
        layer.bias.gradients,
        sizeof(bias_grad_snapshot)
    );

    if (vae_c_dense_backward(
            &layer,
            &data_only_input,
            &output) != VAE_C_OP_GRADIENT_NOT_AVAILABLE ||
        vae_c_dense_backward(
            &layer,
            &input,
            &data_only_output) != VAE_C_OP_GRADIENT_NOT_AVAILABLE ||
        !array_equal(input.grad, input_grad_snapshot, 4) ||
        !array_equal(
            layer.weights.gradients,
            weight_grad_snapshot,
            4) ||
        !array_equal(layer.bias.gradients, bias_grad_snapshot, 2)) {

        passed = 0;
        goto cleanup;
    }

    passed = 1;

cleanup:
    vae_c_tensor_free(&data_only_output);
    vae_c_tensor_free(&data_only_input);
    vae_c_tensor_free(&bad_batch);
    vae_c_tensor_free(&bad_feature);
    vae_c_tensor_free(&bad_output);
    vae_c_tensor_free(&bad_input);
    vae_c_tensor_free(&output);
    vae_c_tensor_free(&input);
    vae_c_dense_free(&layer);
    vae_c_dense_free(&layer);
    return passed && vae_c_dense_is_empty(&layer);
}


static int test_relu_gap_errors_and_strong_guarantee(void)
{
    const size_t dims_1d[1] = {4};
    const size_t dims_1d_other[1] = {3};
    const size_t gap_input_dims[4] = {1, 2, 2, 2};
    const size_t gap_output_dims[2] = {1, 2};
    const size_t gap_bad_nc_dims[2] = {1, 1};
    const size_t gap_bad_batch_dims[2] = {2, 2};
    VAECTensor relu_input;
    VAECTensor relu_output;
    VAECTensor relu_mismatch;
    VAECTensor relu_data_only;
    VAECTensor gap_input;
    VAECTensor gap_output;
    VAECTensor gap_bad_nc;
    VAECTensor gap_bad_batch;
    VAECTensor gap_rank_one;
    VAECTensor gap_output_rank_one;
    VAECTensor gap_data_only;
    VAECTensor corrupted;
    VAECTensor alias;
    float relu_output_snapshot[4];
    float relu_mismatch_snapshot[3];
    float relu_input_grad_snapshot[4];
    float gap_output_snapshot[2];
    float gap_input_grad_snapshot[8];
    size_t index;
    int passed;

    vae_c_tensor_init(&relu_input);
    vae_c_tensor_init(&relu_output);
    vae_c_tensor_init(&relu_mismatch);
    vae_c_tensor_init(&relu_data_only);
    vae_c_tensor_init(&gap_input);
    vae_c_tensor_init(&gap_output);
    vae_c_tensor_init(&gap_bad_nc);
    vae_c_tensor_init(&gap_bad_batch);
    vae_c_tensor_init(&gap_rank_one);
    vae_c_tensor_init(&gap_output_rank_one);
    vae_c_tensor_init(&gap_data_only);

    if (!make_tensor(1, dims_1d, true, &relu_input) ||
        !make_tensor(1, dims_1d, true, &relu_output) ||
        !make_tensor(1, dims_1d_other, true, &relu_mismatch) ||
        !make_tensor(1, dims_1d, false, &relu_data_only) ||
        !make_tensor(4, gap_input_dims, true, &gap_input) ||
        !make_tensor(2, gap_output_dims, true, &gap_output) ||
        !make_tensor(2, gap_bad_nc_dims, true, &gap_bad_nc) ||
        !make_tensor(2, gap_bad_batch_dims, true, &gap_bad_batch) ||
        !make_tensor(1, dims_1d, true, &gap_rank_one) ||
        !make_tensor(1, dims_1d, true, &gap_output_rank_one) ||
        !make_tensor(2, gap_output_dims, false, &gap_data_only)) {

        passed = 0;
        goto cleanup;
    }

    for (index = 0; index < 4; index++) {
        relu_output.data[index] = 21.0f;
        relu_input.grad[index] = 22.0f;
    }
    for (index = 0; index < 3; index++) {
        relu_mismatch.data[index] = 23.0f;
    }
    memcpy(
        relu_output_snapshot,
        relu_output.data,
        sizeof(relu_output_snapshot)
    );
    memcpy(
        relu_input_grad_snapshot,
        relu_input.grad,
        sizeof(relu_input_grad_snapshot)
    );
    memcpy(
        relu_mismatch_snapshot,
        relu_mismatch.data,
        sizeof(relu_mismatch_snapshot)
    );

    if (vae_c_relu_forward(&relu_input, &relu_mismatch) !=
            VAE_C_OP_SHAPE_MISMATCH ||
        !array_equal(
            relu_mismatch.data,
            relu_mismatch_snapshot,
            3) ||
        vae_c_relu_forward(&relu_input, &relu_input) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        vae_c_relu_backward(&relu_data_only, &relu_output) !=
            VAE_C_OP_GRADIENT_NOT_AVAILABLE ||
        vae_c_relu_backward(&relu_input, &relu_data_only) !=
            VAE_C_OP_GRADIENT_NOT_AVAILABLE ||
        !array_equal(
            relu_input.grad,
            relu_input_grad_snapshot,
            4)) {

        passed = 0;
        goto cleanup;
    }

    alias = relu_input;
    if (vae_c_relu_forward(&relu_input, &alias) !=
            VAE_C_OP_INVALID_ARGUMENT) {

        passed = 0;
        goto cleanup;
    }

    for (index = 0; index < 2; index++) {
        gap_output.data[index] = 31.0f;
    }
    for (index = 0; index < 8; index++) {
        gap_input.grad[index] = 32.0f;
    }
    memcpy(
        gap_output_snapshot,
        gap_output.data,
        sizeof(gap_output_snapshot)
    );
    memcpy(
        gap_input_grad_snapshot,
        gap_input.grad,
        sizeof(gap_input_grad_snapshot)
    );

    if (vae_c_global_avg_pool2d_forward(
            &gap_rank_one,
            &gap_output) != VAE_C_OP_SHAPE_MISMATCH ||
        vae_c_global_avg_pool2d_forward(
            &gap_input,
            &gap_output_rank_one) != VAE_C_OP_SHAPE_MISMATCH ||
        vae_c_global_avg_pool2d_forward(
            &gap_input,
            &gap_bad_nc) != VAE_C_OP_SHAPE_MISMATCH ||
        vae_c_global_avg_pool2d_forward(
            &gap_input,
            &gap_bad_batch) != VAE_C_OP_SHAPE_MISMATCH ||
        !array_equal(gap_output.data, gap_output_snapshot, 2) ||
        vae_c_global_avg_pool2d_backward(
            &gap_input,
            &gap_data_only) != VAE_C_OP_GRADIENT_NOT_AVAILABLE ||
        !array_equal(
            gap_input.grad,
            gap_input_grad_snapshot,
            8)) {

        passed = 0;
        goto cleanup;
    }

    corrupted = gap_input;
    corrupted.shape.dims[2] = 0;
    if (vae_c_global_avg_pool2d_forward(
            &corrupted,
            &gap_output) != VAE_C_OP_INVALID_STATE ||
        !array_equal(gap_output.data, gap_output_snapshot, 2)) {

        passed = 0;
        goto cleanup;
    }

    alias = gap_input;
    if (vae_c_global_avg_pool2d_forward(
            &gap_input,
            &alias) != VAE_C_OP_INVALID_ARGUMENT) {

        passed = 0;
        goto cleanup;
    }

    corrupted = relu_input;
    corrupted.owns_memory = true;
    corrupted.owned_block = NULL;
    if (vae_c_relu_forward(&corrupted, &relu_output) !=
            VAE_C_OP_INVALID_STATE ||
        !array_equal(
            relu_output.data,
            relu_output_snapshot,
            4)) {

        passed = 0;
        goto cleanup;
    }

    passed =
        vae_c_relu_forward(NULL, &relu_output) ==
            VAE_C_OP_INVALID_ARGUMENT &&
        vae_c_relu_backward(&relu_input, NULL) ==
            VAE_C_OP_INVALID_ARGUMENT &&
        vae_c_global_avg_pool2d_forward(NULL, &gap_output) ==
            VAE_C_OP_INVALID_ARGUMENT &&
        vae_c_global_avg_pool2d_backward(&gap_input, NULL) ==
            VAE_C_OP_INVALID_ARGUMENT;

cleanup:
    vae_c_tensor_free(&gap_data_only);
    vae_c_tensor_free(&gap_output_rank_one);
    vae_c_tensor_free(&gap_rank_one);
    vae_c_tensor_free(&gap_bad_batch);
    vae_c_tensor_free(&gap_bad_nc);
    vae_c_tensor_free(&gap_output);
    vae_c_tensor_free(&gap_input);
    vae_c_tensor_free(&relu_data_only);
    vae_c_tensor_free(&relu_mismatch);
    vae_c_tensor_free(&relu_output);
    vae_c_tensor_free(&relu_input);
    return passed;
}


static int test_status_names(void)
{
    size_t status;

    for (status = VAE_C_OP_OK;
         status <= VAE_C_OP_ALLOCATION_FAILED;
         status++) {

        if (strcmp(
                vae_c_op_status_name((VAECOpStatus)status),
                "unknown") == 0) {

            return 0;
        }
    }

    vae_c_dense_init(NULL);
    vae_c_dense_free(NULL);

    return
        !vae_c_dense_is_empty(NULL) &&
        strcmp(
            vae_c_op_status_name((VAECOpStatus)999),
            "unknown") == 0;
}


int main(void)
{
    int failures = 0;

    failures += report_check(
        "dense_analytic_and_lifecycle",
        test_dense_analytic_and_lifecycle()
    );
    failures += report_check(
        "dense_without_bias",
        test_dense_without_bias()
    );
    failures += report_check(
        "relu_analytic",
        test_relu_analytic()
    );
    failures += report_check(
        "global_avg_pool2d_analytic",
        test_gap_analytic()
    );
    failures += report_check(
        "dense_gradient_check",
        test_dense_gradient_check()
    );
    failures += report_check(
        "relu_gradient_check",
        test_relu_gradient_check()
    );
    failures += report_check(
        "global_avg_pool2d_gradient_check",
        test_gap_gradient_check()
    );
    failures += report_check(
        "dense_errors_and_strong_guarantee",
        test_dense_errors_and_strong_guarantee()
    );
    failures += report_check(
        "relu_gap_errors_and_strong_guarantee",
        test_relu_gap_errors_and_strong_guarantee()
    );
    failures += report_check(
        "status_names",
        test_status_names()
    );

    if (failures != 0) {
        fprintf(
            stderr,
            "%d VAE C BASIC OPS TEST(S) FAILED\n",
            failures
        );
        return 1;
    }

    printf("ALL VAE C BASIC OPS TESTS PASSED\n");
    return 0;
}
