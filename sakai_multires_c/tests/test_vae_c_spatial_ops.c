#include "vae_c_spatial.h"
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


static int size_array_equal(
    const size_t *left,
    const size_t *right,
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


static int shape_fields_equal(
    const VAECShape *left,
    const VAECShape *right
)
{
    return
        left != NULL &&
        right != NULL &&
        left->rank == right->rank &&
        left->count == right->count &&
        memcmp(
            left->dims,
            right->dims,
            sizeof(left->dims)) == 0 &&
        memcmp(
            left->strides,
            right->strides,
            sizeof(left->strides)) == 0;
}


static int cache_fields_equal(
    const VAECMaxPool2D *left,
    const VAECMaxPool2D *right
)
{
    return
        left != NULL &&
        right != NULL &&
        left->argmax_indices == right->argmax_indices &&
        left->capacity == right->capacity &&
        left->owned_bytes == right->owned_bytes &&
        left->last_count == right->last_count &&
        shape_fields_equal(
            &left->last_input_shape,
            &right->last_input_shape) &&
        shape_fields_equal(
            &left->last_output_shape,
            &right->last_output_shape) &&
        left->cache_valid == right->cache_valid;
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


static void update_gradient_stats(
    GradientCheckStats *stats,
    size_t index,
    double analytic,
    double numeric
)
{
    double absolute_error = fabs(analytic - numeric);
    double denominator = fabs(analytic) + fabs(numeric);
    double relative_error;

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
    double scale = fabs(analytic);

    if (fabs(numeric) > scale) {
        scale = fabs(numeric);
    }

    return
        fabs(analytic - numeric) <=
        absolute_tolerance + relative_tolerance * scale;
}


static int test_max_pool_lifecycle_and_shapes(void)
{
    VAECMaxPool2D layer;
    VAECMaxPool2D snapshot;
    const size_t overflow_capacity = SIZE_MAX / sizeof(size_t) + 1;

    vae_c_max_pool2d_init(&layer);
    if (!vae_c_max_pool2d_is_empty(&layer)) {
        return 0;
    }

    snapshot = layer;
    if (vae_c_max_pool2d_allocate(&layer, 0) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        !cache_fields_equal(&layer, &snapshot) ||
        vae_c_max_pool2d_allocate(&layer, overflow_capacity) !=
            VAE_C_OP_OVERFLOW ||
        !cache_fields_equal(&layer, &snapshot) ||
        vae_c_max_pool2d_allocate(NULL, 4) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        vae_c_max_pool2d_allocate(&layer, 8) != VAE_C_OP_OK ||
        layer.capacity != 8 ||
        layer.owned_bytes != 8 * sizeof(size_t) ||
        layer.cache_valid ||
        layer.last_count != 0) {

        vae_c_max_pool2d_free(&layer);
        return 0;
    }

    snapshot = layer;
    if (vae_c_max_pool2d_allocate(&layer, 8) !=
            VAE_C_OP_OBJECT_NOT_EMPTY ||
        !cache_fields_equal(&layer, &snapshot) ||
        (38 - 2) / 2 + 1 != 19 ||
        (56 - 2) / 2 + 1 != 28) {

        vae_c_max_pool2d_free(&layer);
        return 0;
    }

    vae_c_max_pool2d_free(&layer);
    vae_c_max_pool2d_free(&layer);
    vae_c_max_pool2d_free(NULL);
    return vae_c_max_pool2d_is_empty(&layer);
}


static int test_max_pool_analytic(void)
{
    const size_t input_dims[4] = {1, 1, 4, 4};
    const size_t output_dims[4] = {1, 1, 2, 2};
    const float input_values[16] = {
        -1.0f, 2.0f, 3.0f, 0.0f,
        4.0f, 1.0f, -2.0f, 5.0f,
        7.0f, 6.0f, 8.0f, 9.0f,
        0.0f, -3.0f, 10.0f, 2.0f
    };
    const float expected_output[4] = {4.0f, 5.0f, 7.0f, 10.0f};
    const size_t expected_indices[4] = {4, 7, 8, 14};
    const float upstream[4] = {1.0f, -2.0f, 3.0f, 4.0f};
    const float expected_grad[16] = {
        0.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, -2.0f,
        3.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 4.0f, 0.0f
    };
    VAECMaxPool2D layer;
    VAECTensor input;
    VAECTensor output;
    float output_data_snapshot[4];
    float output_grad_snapshot[4];
    size_t index;
    int passed;

    vae_c_max_pool2d_init(&layer);
    vae_c_tensor_init(&input);
    vae_c_tensor_init(&output);

    if (vae_c_max_pool2d_allocate(&layer, 4) != VAE_C_OP_OK ||
        !make_tensor(4, input_dims, true, &input) ||
        !make_tensor(4, output_dims, true, &output)) {

        vae_c_tensor_free(&output);
        vae_c_tensor_free(&input);
        vae_c_max_pool2d_free(&layer);
        return 0;
    }

    memcpy(input.data, input_values, sizeof(input_values));
    memcpy(output.grad, upstream, sizeof(upstream));

    if (vae_c_max_pool2d_forward(&layer, &input, &output) !=
            VAE_C_OP_OK ||
        !array_equal(output.data, expected_output, 4) ||
        !size_array_equal(layer.argmax_indices, expected_indices, 4) ||
        !layer.cache_valid || layer.last_count != 4 ||
        !shape_fields_equal(&layer.last_input_shape, &input.shape) ||
        !shape_fields_equal(&layer.last_output_shape, &output.shape)) {

        vae_c_tensor_free(&output);
        vae_c_tensor_free(&input);
        vae_c_max_pool2d_free(&layer);
        return 0;
    }

    memcpy(output_data_snapshot, output.data, sizeof(output_data_snapshot));
    memcpy(output_grad_snapshot, output.grad, sizeof(output_grad_snapshot));

    if (vae_c_max_pool2d_backward(&layer, &input, &output) !=
            VAE_C_OP_OK ||
        !array_equal(input.grad, expected_grad, 16) ||
        !array_equal(input.data, input_values, 16) ||
        !array_equal(output.data, output_data_snapshot, 4) ||
        !array_equal(output.grad, output_grad_snapshot, 4) ||
        vae_c_max_pool2d_backward(&layer, &input, &output) !=
            VAE_C_OP_OK) {

        vae_c_tensor_free(&output);
        vae_c_tensor_free(&input);
        vae_c_max_pool2d_free(&layer);
        return 0;
    }

    passed = 1;
    for (index = 0; index < 16; index++) {
        if (input.grad[index] != 2.0f * expected_grad[index]) {
            passed = 0;
        }
    }
    passed =
        passed &&
        array_equal(output.grad, output_grad_snapshot, 4) &&
        size_array_equal(layer.argmax_indices, expected_indices, 4);

    vae_c_tensor_free(&output);
    vae_c_tensor_free(&input);
    vae_c_max_pool2d_free(&layer);
    return passed;
}


static int test_max_pool_tie_and_batch(void)
{
    const size_t tie_input_dims[4] = {1, 1, 2, 2};
    const size_t tie_output_dims[4] = {1, 1, 1, 1};
    const size_t batch_input_dims[4] = {2, 1, 2, 2};
    const size_t batch_output_dims[4] = {2, 1, 1, 1};
    const float tie_values[4] = {5.0f, 5.0f, 1.0f, 0.0f};
    const float batch_values[8] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        9.0f, 1.0f, 2.0f, 3.0f
    };
    const float batch_expected[2] = {4.0f, 9.0f};
    const size_t batch_indices[2] = {3, 4};
    VAECMaxPool2D tie_layer;
    VAECMaxPool2D batch_layer;
    VAECTensor tie_input;
    VAECTensor tie_output;
    VAECTensor batch_input;
    VAECTensor batch_output;
    int passed;

    vae_c_max_pool2d_init(&tie_layer);
    vae_c_max_pool2d_init(&batch_layer);
    vae_c_tensor_init(&tie_input);
    vae_c_tensor_init(&tie_output);
    vae_c_tensor_init(&batch_input);
    vae_c_tensor_init(&batch_output);

    if (vae_c_max_pool2d_allocate(&tie_layer, 1) != VAE_C_OP_OK ||
        vae_c_max_pool2d_allocate(&batch_layer, 2) != VAE_C_OP_OK ||
        !make_tensor(4, tie_input_dims, true, &tie_input) ||
        !make_tensor(4, tie_output_dims, true, &tie_output) ||
        !make_tensor(4, batch_input_dims, false, &batch_input) ||
        !make_tensor(4, batch_output_dims, false, &batch_output)) {

        passed = 0;
        goto cleanup;
    }

    memcpy(tie_input.data, tie_values, sizeof(tie_values));
    tie_output.grad[0] = 3.0f;
    memcpy(batch_input.data, batch_values, sizeof(batch_values));

    passed =
        vae_c_max_pool2d_forward(
            &tie_layer,
            &tie_input,
            &tie_output) == VAE_C_OP_OK &&
        tie_output.data[0] == 5.0f &&
        tie_layer.argmax_indices[0] == 0 &&
        vae_c_max_pool2d_backward(
            &tie_layer,
            &tie_input,
            &tie_output) == VAE_C_OP_OK &&
        tie_input.grad[0] == 3.0f &&
        tie_input.grad[1] == 0.0f &&
        vae_c_max_pool2d_forward(
            &batch_layer,
            &batch_input,
            &batch_output) == VAE_C_OP_OK &&
        array_equal(batch_output.data, batch_expected, 2) &&
        size_array_equal(batch_layer.argmax_indices, batch_indices, 2);

cleanup:
    vae_c_tensor_free(&batch_output);
    vae_c_tensor_free(&batch_input);
    vae_c_tensor_free(&tie_output);
    vae_c_tensor_free(&tie_input);
    vae_c_max_pool2d_free(&batch_layer);
    vae_c_max_pool2d_free(&tie_layer);
    return passed;
}


static int check_height_mapping(size_t input_height, size_t output_height)
{
    const size_t input_dims[4] = {1, 1, input_height, 1};
    const size_t output_dims[4] = {1, 1, output_height, 1};
    VAECTensor input;
    VAECTensor output;
    size_t index;
    int passed;

    if (!make_tensor(4, input_dims, false, &input) ||
        !make_tensor(4, output_dims, false, &output)) {

        vae_c_tensor_free(&output);
        vae_c_tensor_free(&input);
        return 0;
    }

    for (index = 0; index < input_height; index++) {
        input.data[index] = (float)(index + 1);
    }

    passed =
        vae_c_nearest_upsample2d_forward(&input, &output) ==
        VAE_C_OP_OK;
    for (index = 0; passed && index < output_height; index++) {
        size_t source = (index * input_height) / output_height;
        if (output.data[index] != input.data[source]) {
            passed = 0;
        }
    }

    vae_c_tensor_free(&output);
    vae_c_tensor_free(&input);
    return passed;
}


static int test_upsample_analytic_and_profiles(void)
{
    const size_t input_dims[4] = {1, 1, 2, 3};
    const size_t output_dims[4] = {1, 1, 3, 5};
    const float input_values[6] = {1, 2, 3, 4, 5, 6};
    const float expected_output[15] = {
        1, 1, 2, 2, 3,
        1, 1, 2, 2, 3,
        4, 4, 5, 5, 6
    };
    const float expected_grad[6] = {16, 24, 15, 23, 27, 15};
    VAECTensor input;
    VAECTensor output;
    float output_grad_snapshot[15];
    size_t index;
    int passed;

    if (!make_tensor(4, input_dims, true, &input) ||
        !make_tensor(4, output_dims, true, &output)) {

        vae_c_tensor_free(&output);
        vae_c_tensor_free(&input);
        return 0;
    }

    memcpy(input.data, input_values, sizeof(input_values));
    for (index = 0; index < 15; index++) {
        output.grad[index] = (float)(index + 1);
    }
    memcpy(output_grad_snapshot, output.grad, sizeof(output_grad_snapshot));

    if (vae_c_nearest_upsample2d_forward(&input, &output) !=
            VAE_C_OP_OK ||
        !array_equal(output.data, expected_output, 15) ||
        vae_c_nearest_upsample2d_backward(&input, &output) !=
            VAE_C_OP_OK ||
        !array_equal(input.grad, expected_grad, 6) ||
        !array_equal(output.grad, output_grad_snapshot, 15) ||
        vae_c_nearest_upsample2d_backward(&input, &output) !=
            VAE_C_OP_OK) {

        vae_c_tensor_free(&output);
        vae_c_tensor_free(&input);
        return 0;
    }

    passed = 1;
    for (index = 0; index < 6; index++) {
        if (input.grad[index] != 2.0f * expected_grad[index]) {
            passed = 0;
        }
    }

    passed =
        passed &&
        array_equal(output.grad, output_grad_snapshot, 15) &&
        check_height_mapping(5, 10) &&
        check_height_mapping(10, 19) &&
        check_height_mapping(38, 75) &&
        check_height_mapping(7, 14) &&
        check_height_mapping(14, 28) &&
        check_height_mapping(28, 56) &&
        check_height_mapping(56, 112) &&
        check_height_mapping(6, 6);

    vae_c_tensor_free(&output);
    vae_c_tensor_free(&input);
    return passed;
}


static double max_pool_objective(
    VAECMaxPool2D *layer,
    const VAECTensor *input,
    VAECTensor *output,
    const float *upstream
)
{
    double objective = 0.0;
    size_t index;

    if (vae_c_max_pool2d_forward(layer, input, output) != VAE_C_OP_OK) {
        return NAN;
    }

    for (index = 0; index < output->shape.count; index++) {
        objective +=
            (double)output->data[index] * (double)upstream[index];
    }

    return objective;
}


static int test_max_pool_gradient_check(void)
{
    const size_t input_dims[4] = {1, 1, 4, 4};
    const size_t output_dims[4] = {1, 1, 2, 2};
    const float input_values[16] = {
        0.1f, 1.0f, -1.0f, 2.0f,
        3.0f, 0.2f, 0.5f, 4.0f,
        5.0f, 0.3f, 2.0f, 1.0f,
        0.0f, 6.0f, 7.0f, -2.0f
    };
    const float upstream[4] = {0.7f, -0.4f, 1.2f, 0.3f};
    const float epsilon = 1.0e-3f;
    const double absolute_tolerance = 8.0e-5;
    const double relative_tolerance = 8.0e-4;
    VAECMaxPool2D layer;
    VAECTensor input;
    VAECTensor output;
    GradientCheckStats stats;
    size_t index;
    int passed;

    vae_c_max_pool2d_init(&layer);
    stats.max_absolute = 0.0;
    stats.max_relative = 0.0;
    stats.worst_index = 0;
    passed = 1;

    if (vae_c_max_pool2d_allocate(&layer, 4) != VAE_C_OP_OK ||
        !make_tensor(4, input_dims, true, &input) ||
        !make_tensor(4, output_dims, true, &output)) {

        vae_c_tensor_free(&output);
        vae_c_tensor_free(&input);
        vae_c_max_pool2d_free(&layer);
        return 0;
    }

    memcpy(input.data, input_values, sizeof(input_values));
    memcpy(output.grad, upstream, sizeof(upstream));

    if (vae_c_max_pool2d_forward(&layer, &input, &output) !=
            VAE_C_OP_OK ||
        vae_c_tensor_zero_grad(&input) != VAE_C_STORAGE_OK ||
        vae_c_max_pool2d_backward(&layer, &input, &output) !=
            VAE_C_OP_OK) {

        vae_c_tensor_free(&output);
        vae_c_tensor_free(&input);
        vae_c_max_pool2d_free(&layer);
        return 0;
    }

    for (index = 0; index < input.shape.count; index++) {
        float original = input.data[index];
        double plus;
        double minus;
        double numeric;
        double analytic = (double)input.grad[index];

        input.data[index] = original + epsilon;
        plus = max_pool_objective(&layer, &input, &output, upstream);
        input.data[index] = original - epsilon;
        minus = max_pool_objective(&layer, &input, &output, upstream);
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
        "MAXPOOL_GRAD_CHECK_MAX_ABS=%.9g MAX_REL=%.9g WORST_INDEX=%zu\n",
        stats.max_absolute,
        stats.max_relative,
        stats.worst_index
    );

    vae_c_tensor_free(&output);
    vae_c_tensor_free(&input);
    vae_c_max_pool2d_free(&layer);
    return passed;
}


static double upsample_objective(
    const VAECTensor *input,
    VAECTensor *output,
    const float *upstream
)
{
    double objective = 0.0;
    size_t index;

    if (vae_c_nearest_upsample2d_forward(input, output) != VAE_C_OP_OK) {
        return NAN;
    }

    for (index = 0; index < output->shape.count; index++) {
        objective +=
            (double)output->data[index] * (double)upstream[index];
    }

    return objective;
}


static int test_upsample_gradient_check(void)
{
    const size_t input_dims[4] = {1, 1, 2, 3};
    const size_t output_dims[4] = {1, 1, 3, 5};
    const float input_values[6] = {-0.5f, 0.2f, 1.1f, 0.7f, -0.3f, 0.9f};
    const float upstream[15] = {
        0.1f, -0.2f, 0.3f, 0.4f, -0.5f,
        0.6f, 0.2f, -0.1f, 0.7f, 0.8f,
        -0.4f, 0.9f, 0.5f, -0.6f, 0.25f
    };
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
        !make_tensor(4, output_dims, true, &output)) {

        vae_c_tensor_free(&output);
        vae_c_tensor_free(&input);
        return 0;
    }

    memcpy(input.data, input_values, sizeof(input_values));
    memcpy(output.grad, upstream, sizeof(upstream));

    if (vae_c_nearest_upsample2d_forward(&input, &output) !=
            VAE_C_OP_OK ||
        vae_c_tensor_zero_grad(&input) != VAE_C_STORAGE_OK ||
        vae_c_nearest_upsample2d_backward(&input, &output) !=
            VAE_C_OP_OK) {

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
        plus = upsample_objective(&input, &output, upstream);
        input.data[index] = original - epsilon;
        minus = upsample_objective(&input, &output, upstream);
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
        "UPSAMPLE_GRAD_CHECK_MAX_ABS=%.9g MAX_REL=%.9g WORST_INDEX=%zu\n",
        stats.max_absolute,
        stats.max_relative,
        stats.worst_index
    );

    vae_c_tensor_free(&output);
    vae_c_tensor_free(&input);
    return passed;
}


static int test_max_pool_errors_and_strong_guarantee(void)
{
    const size_t input_dims[4] = {1, 1, 4, 4};
    const size_t output_dims[4] = {1, 1, 2, 2};
    const size_t rank_one_dims[1] = {16};
    const size_t wrong_nc_dims[4] = {1, 2, 2, 2};
    const size_t wrong_hw_dims[4] = {1, 1, 2, 1};
    const size_t small_input_dims[4] = {1, 1, 1, 4};
    const size_t incoherent_cache_dims[4] = {1, 1, 1, 4};
    const size_t other_input_dims[4] = {1, 1, 2, 8};
    const float zero_grad[16] = {0};
    VAECMaxPool2D layer;
    VAECMaxPool2D fresh;
    VAECMaxPool2D empty_layer;
    VAECMaxPool2D snapshot;
    VAECMaxPool2D corrupted_layer;
    VAECMaxPool2D alias_layer;
    VAECTensor input;
    VAECTensor output;
    VAECTensor rank_one;
    VAECTensor wrong_nc;
    VAECTensor wrong_hw;
    VAECTensor small_input;
    VAECTensor data_only_input;
    VAECTensor data_only_output;
    VAECTensor other_input;
    VAECTensor alias;
    VAECTensor alias_output;
    VAECTensor corrupted_input;
    VAECShape incoherent_cache_shape;
    float output_snapshot[4];
    float input_grad_snapshot[16];
    float output_grad_snapshot[4];
    size_t cache_snapshot[4];
    size_t saved_index;
    size_t index;
    int passed;

    vae_c_max_pool2d_init(&layer);
    vae_c_max_pool2d_init(&fresh);
    vae_c_max_pool2d_init(&empty_layer);
    vae_c_tensor_init(&input);
    vae_c_tensor_init(&output);
    vae_c_tensor_init(&rank_one);
    vae_c_tensor_init(&wrong_nc);
    vae_c_tensor_init(&wrong_hw);
    vae_c_tensor_init(&small_input);
    vae_c_tensor_init(&data_only_input);
    vae_c_tensor_init(&data_only_output);
    vae_c_tensor_init(&other_input);

    if (vae_c_max_pool2d_forward(NULL, NULL, NULL) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        vae_c_max_pool2d_backward(NULL, NULL, NULL) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        vae_c_max_pool2d_allocate(&layer, 4) != VAE_C_OP_OK ||
        vae_c_max_pool2d_allocate(&fresh, 4) != VAE_C_OP_OK ||
        !make_tensor(4, input_dims, true, &input) ||
        !make_tensor(4, output_dims, true, &output) ||
        !make_tensor(1, rank_one_dims, true, &rank_one) ||
        !make_tensor(4, wrong_nc_dims, true, &wrong_nc) ||
        !make_tensor(4, wrong_hw_dims, true, &wrong_hw) ||
        !make_tensor(4, small_input_dims, true, &small_input) ||
        !make_tensor(4, input_dims, false, &data_only_input) ||
        !make_tensor(4, output_dims, false, &data_only_output) ||
        !make_tensor(4, other_input_dims, true, &other_input)) {

        passed = 0;
        goto cleanup;
    }

    for (index = 0; index < 16; index++) {
        input.data[index] = (float)index;
        input.grad[index] = 41.0f;
    }
    for (index = 0; index < 4; index++) {
        output.data[index] = 42.0f;
        output.grad[index] = 43.0f;
    }

    if (vae_c_max_pool2d_forward(&empty_layer, &input, &output) !=
            VAE_C_OP_INVALID_STATE ||
        vae_c_max_pool2d_backward(&fresh, &input, &output) !=
            VAE_C_OP_INVALID_STATE ||
        vae_c_max_pool2d_forward(&layer, &input, &output) !=
            VAE_C_OP_OK) {

        passed = 0;
        goto cleanup;
    }

    snapshot = layer;
    memcpy(cache_snapshot, layer.argmax_indices, sizeof(cache_snapshot));
    memcpy(output_snapshot, output.data, sizeof(output_snapshot));
    memcpy(input_grad_snapshot, input.grad, sizeof(input_grad_snapshot));
    memcpy(output_grad_snapshot, output.grad, sizeof(output_grad_snapshot));

    if (vae_c_max_pool2d_forward(&layer, &rank_one, &output) !=
            VAE_C_OP_SHAPE_MISMATCH ||
        vae_c_max_pool2d_forward(&layer, &input, &rank_one) !=
            VAE_C_OP_SHAPE_MISMATCH ||
        vae_c_max_pool2d_forward(&layer, &input, &wrong_nc) !=
            VAE_C_OP_SHAPE_MISMATCH ||
        vae_c_max_pool2d_forward(&layer, &input, &wrong_hw) !=
            VAE_C_OP_SHAPE_MISMATCH ||
        vae_c_max_pool2d_forward(&layer, &small_input, &output) !=
            VAE_C_OP_SHAPE_MISMATCH ||
        vae_c_max_pool2d_forward(&layer, &input, &input) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        !array_equal(output.data, output_snapshot, 4) ||
        !size_array_equal(layer.argmax_indices, cache_snapshot, 4) ||
        !cache_fields_equal(&layer, &snapshot)) {

        passed = 0;
        goto cleanup;
    }

    alias = input;
    if (vae_c_max_pool2d_forward(&layer, &input, &alias) !=
            VAE_C_OP_INVALID_ARGUMENT) {

        passed = 0;
        goto cleanup;
    }

    alias_output = output;
    alias_output.owns_memory = false;
    alias_output.owned_block = NULL;
    alias_output.owned_bytes = 0;
    alias_output.data = input.data;
    alias_output.has_grad = false;
    alias_output.grad = NULL;
    if (vae_c_max_pool2d_forward(&layer, &input, &alias_output) !=
            VAE_C_OP_INVALID_ARGUMENT) {

        passed = 0;
        goto cleanup;
    }

    alias_output.data = output.data;
    alias_output.has_grad = true;
    alias_output.grad = input.data;
    if (vae_c_max_pool2d_forward(&layer, &input, &alias_output) !=
            VAE_C_OP_INVALID_ARGUMENT) {

        passed = 0;
        goto cleanup;
    }

    alias_output.data = input.grad;
    alias_output.has_grad = false;
    alias_output.grad = NULL;
    if (vae_c_max_pool2d_forward(&layer, &input, &alias_output) !=
            VAE_C_OP_INVALID_ARGUMENT) {

        passed = 0;
        goto cleanup;
    }

    alias_output.data = output.data;
    alias_output.has_grad = true;
    alias_output.grad = input.grad;
    if (vae_c_max_pool2d_forward(&layer, &input, &alias_output) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        !array_equal(output.data, output_snapshot, 4) ||
        !size_array_equal(layer.argmax_indices, cache_snapshot, 4) ||
        !cache_fields_equal(&layer, &snapshot)) {

        passed = 0;
        goto cleanup;
    }

    alias_layer = layer;
    alias_layer.argmax_indices = (size_t *)(void *)input.data;
    if (vae_c_max_pool2d_forward(&alias_layer, &input, &output) !=
            VAE_C_OP_INVALID_ARGUMENT) {

        passed = 0;
        goto cleanup;
    }

    corrupted_layer = layer;
    corrupted_layer.last_count++;
    if (vae_c_max_pool2d_backward(&corrupted_layer, &input, &output) !=
            VAE_C_OP_INVALID_STATE ||
        !array_equal(input.grad, input_grad_snapshot, 16)) {

        passed = 0;
        goto cleanup;
    }

    if (vae_c_shape_make(
            4,
            incoherent_cache_dims,
            &incoherent_cache_shape) != VAE_C_STORAGE_OK) {

        passed = 0;
        goto cleanup;
    }
    corrupted_layer = layer;
    corrupted_layer.last_output_shape = incoherent_cache_shape;
    if (vae_c_max_pool2d_forward(&corrupted_layer, &input, &output) !=
            VAE_C_OP_INVALID_STATE ||
        !array_equal(output.data, output_snapshot, 4) ||
        !size_array_equal(layer.argmax_indices, cache_snapshot, 4) ||
        !cache_fields_equal(&layer, &snapshot)) {

        passed = 0;
        goto cleanup;
    }

    if (vae_c_max_pool2d_backward(
            &layer,
            &data_only_input,
            &output) != VAE_C_OP_GRADIENT_NOT_AVAILABLE ||
        vae_c_max_pool2d_backward(
            &layer,
            &input,
            &data_only_output) != VAE_C_OP_GRADIENT_NOT_AVAILABLE ||
        !array_equal(input.grad, input_grad_snapshot, 16)) {

        passed = 0;
        goto cleanup;
    }

    saved_index = layer.argmax_indices[0];
    layer.argmax_indices[0] = input.shape.count;
    if (vae_c_max_pool2d_backward(&layer, &input, &output) !=
            VAE_C_OP_INVALID_STATE ||
        !array_equal(input.grad, input_grad_snapshot, 16) ||
        !array_equal(output.grad, output_grad_snapshot, 4)) {

        passed = 0;
        layer.argmax_indices[0] = saved_index;
        goto cleanup;
    }
    layer.argmax_indices[0] = saved_index;

    corrupted_input = input;
    corrupted_input.shape.dims[2] = 3;
    if (vae_c_max_pool2d_backward(&layer, &corrupted_input, &output) !=
            VAE_C_OP_INVALID_STATE ||
        !array_equal(input.grad, input_grad_snapshot, 16)) {

        passed = 0;
        goto cleanup;
    }

    if (vae_c_max_pool2d_backward(&layer, &other_input, &output) !=
            VAE_C_OP_INVALID_STATE ||
        !array_equal(other_input.grad, zero_grad, 16)) {

        passed = 0;
        goto cleanup;
    }

    vae_c_max_pool2d_free(&fresh);
    if (vae_c_max_pool2d_allocate(&fresh, 1) != VAE_C_OP_OK ||
        vae_c_max_pool2d_forward(&fresh, &input, &output) !=
            VAE_C_OP_SHAPE_MISMATCH) {

        passed = 0;
        goto cleanup;
    }

    passed = 1;

cleanup:
    vae_c_tensor_free(&other_input);
    vae_c_tensor_free(&data_only_output);
    vae_c_tensor_free(&data_only_input);
    vae_c_tensor_free(&small_input);
    vae_c_tensor_free(&wrong_hw);
    vae_c_tensor_free(&wrong_nc);
    vae_c_tensor_free(&rank_one);
    vae_c_tensor_free(&output);
    vae_c_tensor_free(&input);
    vae_c_max_pool2d_free(&fresh);
    vae_c_max_pool2d_free(&empty_layer);
    vae_c_max_pool2d_free(&layer);
    return passed;
}


static int test_upsample_errors_and_strong_guarantee(void)
{
    const size_t input_dims[4] = {1, 1, 2, 3};
    const size_t output_dims[4] = {1, 1, 3, 5};
    const size_t rank_one_dims[1] = {6};
    const size_t wrong_nc_dims[4] = {1, 2, 3, 5};
    const size_t smaller_dims[4] = {1, 1, 1, 2};
    const size_t narrow_input_dims[4] = {1, 1, 2, 1};
    const size_t huge_dims[4] = {1, 1, SIZE_MAX, 1};
    VAECTensor input;
    VAECTensor output;
    VAECTensor rank_one;
    VAECTensor wrong_nc;
    VAECTensor smaller;
    VAECTensor data_only_input;
    VAECTensor data_only_output;
    VAECTensor narrow_input;
    VAECTensor alias;
    VAECTensor corrupted;
    VAECTensor huge_output;
    VAECShape huge_shape;
    float output_snapshot[15];
    float input_grad_snapshot[6];
    float dummy_output = 77.0f;
    size_t index;
    int passed;

    vae_c_tensor_init(&input);
    vae_c_tensor_init(&output);
    vae_c_tensor_init(&rank_one);
    vae_c_tensor_init(&wrong_nc);
    vae_c_tensor_init(&smaller);
    vae_c_tensor_init(&data_only_input);
    vae_c_tensor_init(&data_only_output);
    vae_c_tensor_init(&narrow_input);
    vae_c_tensor_init(&huge_output);

    if (!make_tensor(4, input_dims, true, &input) ||
        !make_tensor(4, output_dims, true, &output) ||
        !make_tensor(1, rank_one_dims, true, &rank_one) ||
        !make_tensor(4, wrong_nc_dims, true, &wrong_nc) ||
        !make_tensor(4, smaller_dims, true, &smaller) ||
        !make_tensor(4, input_dims, false, &data_only_input) ||
        !make_tensor(4, output_dims, false, &data_only_output) ||
        !make_tensor(4, narrow_input_dims, false, &narrow_input)) {

        passed = 0;
        goto cleanup;
    }

    for (index = 0; index < 15; index++) {
        output.data[index] = 51.0f;
    }
    for (index = 0; index < 6; index++) {
        input.grad[index] = 52.0f;
    }
    memcpy(output_snapshot, output.data, sizeof(output_snapshot));
    memcpy(input_grad_snapshot, input.grad, sizeof(input_grad_snapshot));

    if (vae_c_nearest_upsample2d_forward(NULL, &output) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        vae_c_nearest_upsample2d_forward(&input, NULL) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        vae_c_nearest_upsample2d_forward(&input, &input) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        vae_c_nearest_upsample2d_forward(&rank_one, &output) !=
            VAE_C_OP_SHAPE_MISMATCH ||
        vae_c_nearest_upsample2d_forward(&input, &rank_one) !=
            VAE_C_OP_SHAPE_MISMATCH ||
        vae_c_nearest_upsample2d_forward(&input, &wrong_nc) !=
            VAE_C_OP_SHAPE_MISMATCH ||
        vae_c_nearest_upsample2d_forward(&input, &smaller) !=
            VAE_C_OP_SHAPE_MISMATCH ||
        !array_equal(output.data, output_snapshot, 15)) {

        passed = 0;
        goto cleanup;
    }

    alias = input;
    if (vae_c_nearest_upsample2d_forward(&input, &alias) !=
            VAE_C_OP_INVALID_ARGUMENT) {

        passed = 0;
        goto cleanup;
    }

    corrupted = input;
    corrupted.shape.strides[2]++;
    if (vae_c_nearest_upsample2d_forward(&corrupted, &output) !=
            VAE_C_OP_INVALID_STATE ||
        !array_equal(output.data, output_snapshot, 15)) {

        passed = 0;
        goto cleanup;
    }

    if (vae_c_nearest_upsample2d_backward(
            &data_only_input,
            &output) != VAE_C_OP_GRADIENT_NOT_AVAILABLE ||
        vae_c_nearest_upsample2d_backward(
            &input,
            &data_only_output) != VAE_C_OP_GRADIENT_NOT_AVAILABLE ||
        !array_equal(input.grad, input_grad_snapshot, 6)) {

        passed = 0;
        goto cleanup;
    }

    if (vae_c_shape_make(4, huge_dims, &huge_shape) !=
            VAE_C_STORAGE_OK) {

        passed = 0;
        goto cleanup;
    }
    huge_output.shape = huge_shape;
    huge_output.data = &dummy_output;
    if (vae_c_nearest_upsample2d_forward(
            &narrow_input,
            &huge_output) != VAE_C_OP_OVERFLOW ||
        dummy_output != 77.0f) {

        passed = 0;
        goto cleanup;
    }

    passed =
        vae_c_nearest_upsample2d_backward(NULL, &output) ==
            VAE_C_OP_INVALID_ARGUMENT &&
        vae_c_nearest_upsample2d_backward(&input, NULL) ==
            VAE_C_OP_INVALID_ARGUMENT;

cleanup:
    vae_c_tensor_free(&narrow_input);
    vae_c_tensor_free(&data_only_output);
    vae_c_tensor_free(&data_only_input);
    vae_c_tensor_free(&smaller);
    vae_c_tensor_free(&wrong_nc);
    vae_c_tensor_free(&rank_one);
    vae_c_tensor_free(&output);
    vae_c_tensor_free(&input);
    return passed;
}


int main(void)
{
    int failures = 0;

    failures += report_check(
        "max_pool_lifecycle_and_shapes",
        test_max_pool_lifecycle_and_shapes()
    );
    failures += report_check(
        "max_pool_analytic",
        test_max_pool_analytic()
    );
    failures += report_check(
        "max_pool_tie_and_batch",
        test_max_pool_tie_and_batch()
    );
    failures += report_check(
        "nearest_upsample_analytic_and_profiles",
        test_upsample_analytic_and_profiles()
    );
    failures += report_check(
        "max_pool_gradient_check",
        test_max_pool_gradient_check()
    );
    failures += report_check(
        "nearest_upsample_gradient_check",
        test_upsample_gradient_check()
    );
    failures += report_check(
        "max_pool_errors_and_strong_guarantee",
        test_max_pool_errors_and_strong_guarantee()
    );
    failures += report_check(
        "nearest_upsample_errors_and_strong_guarantee",
        test_upsample_errors_and_strong_guarantee()
    );

    if (failures != 0) {
        fprintf(
            stderr,
            "%d VAE C SPATIAL OPS TEST(S) FAILED\n",
            failures
        );
        return 1;
    }

    printf("ALL VAE C SPATIAL OPS TESTS PASSED\n");
    return 0;
}
