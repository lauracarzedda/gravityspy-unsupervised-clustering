#include "vae_c_batchnorm.h"
#include "vae_c_tensor.h"

#include <float.h>
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

    if (actual == NULL || expected == NULL) {
        return 0;
    }
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
    return
        left != NULL &&
        right != NULL &&
        memcmp(left, right, count * sizeof(float)) == 0;
}


static int shape_is_zero(const VAECShape *shape)
{
    size_t index;

    if (shape == NULL || shape->rank != 0 || shape->count != 0) {
        return 0;
    }
    for (index = 0; index < VAE_C_TENSOR_MAX_RANK; index++) {
        if (shape->dims[index] != 0 || shape->strides[index] != 0) {
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
        memcmp(left, right, sizeof(*left)) == 0;
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


static size_t reference_index(
    VAECBatchNormKind kind,
    size_t num_features,
    size_t spatial_count,
    size_t feature,
    size_t sample
)
{
    if (kind == VAE_C_BATCH_NORM_KIND_1D) {
        return sample * num_features + feature;
    }
    return
        ((sample / spatial_count) * num_features + feature) *
            spatial_count +
        sample % spatial_count;
}


static void reference_train(
    VAECBatchNormKind kind,
    size_t num_features,
    size_t sample_count,
    size_t spatial_count,
    const float *input,
    const float *gamma,
    const float *beta,
    const float *running_mean_old,
    const float *running_variance_old,
    double *mean,
    double *inverse_std,
    double *unbiased_variance,
    double *output,
    double *running_mean_new,
    double *running_variance_new
)
{
    size_t feature;
    size_t sample;

    for (feature = 0; feature < num_features; feature++) {
        double sum = 0.0;
        double squared = 0.0;

        for (sample = 0; sample < sample_count; sample++) {
            size_t index = reference_index(
                kind,
                num_features,
                spatial_count,
                feature,
                sample
            );
            sum += (double)input[index];
        }
        mean[feature] = sum / (double)sample_count;
        for (sample = 0; sample < sample_count; sample++) {
            size_t index = reference_index(
                kind,
                num_features,
                spatial_count,
                feature,
                sample
            );
            double difference = (double)input[index] - mean[feature];
            squared += difference * difference;
        }
        inverse_std[feature] =
            1.0 /
            sqrt(
                squared / (double)sample_count +
                (double)VAE_C_BATCH_NORM_EPSILON);
        unbiased_variance[feature] =
            squared / (double)(sample_count - 1);
        running_mean_new[feature] =
            0.9 * (double)running_mean_old[feature] +
            0.1 * mean[feature];
        running_variance_new[feature] =
            0.9 * (double)running_variance_old[feature] +
            0.1 * unbiased_variance[feature];

        for (sample = 0; sample < sample_count; sample++) {
            size_t index = reference_index(
                kind,
                num_features,
                spatial_count,
                feature,
                sample
            );
            double normalized =
                ((double)input[index] - mean[feature]) *
                inverse_std[feature];
            output[index] =
                (double)gamma[feature] * normalized +
                (double)beta[feature];
        }
    }
}


static void reference_backward(
    VAECBatchNormKind kind,
    size_t num_features,
    size_t sample_count,
    size_t spatial_count,
    const float *input,
    const float *gamma,
    const float *upstream,
    const double *mean,
    const double *inverse_std,
    double *input_gradient,
    double *gamma_gradient,
    double *beta_gradient
)
{
    size_t feature;
    size_t sample;

    for (feature = 0; feature < num_features; feature++) {
        double sum_dy = 0.0;
        double sum_dy_xhat = 0.0;

        for (sample = 0; sample < sample_count; sample++) {
            size_t index = reference_index(
                kind,
                num_features,
                spatial_count,
                feature,
                sample
            );
            double xhat =
                ((double)input[index] - mean[feature]) *
                inverse_std[feature];
            sum_dy += (double)upstream[index];
            sum_dy_xhat += (double)upstream[index] * xhat;
        }
        beta_gradient[feature] = sum_dy;
        gamma_gradient[feature] = sum_dy_xhat;
        for (sample = 0; sample < sample_count; sample++) {
            size_t index = reference_index(
                kind,
                num_features,
                spatial_count,
                feature,
                sample
            );
            double xhat =
                ((double)input[index] - mean[feature]) *
                inverse_std[feature];
            input_gradient[index] =
                (double)gamma[feature] * inverse_std[feature] /
                (double)sample_count *
                ((double)sample_count * (double)upstream[index] -
                 sum_dy - xhat * sum_dy_xhat);
        }
    }
}


static void reference_eval(
    VAECBatchNormKind kind,
    size_t num_features,
    size_t sample_count,
    size_t spatial_count,
    const float *input,
    const float *gamma,
    const float *beta,
    const float *running_mean,
    const float *running_variance,
    double *output
)
{
    size_t feature;
    size_t sample;

    for (feature = 0; feature < num_features; feature++) {
        double inverse_std =
            1.0 /
            sqrt(
                (double)running_variance[feature] +
                (double)VAE_C_BATCH_NORM_EPSILON);

        for (sample = 0; sample < sample_count; sample++) {
            size_t index = reference_index(
                kind,
                num_features,
                spatial_count,
                feature,
                sample
            );
            output[index] =
                (double)gamma[feature] *
                ((double)input[index] -
                 (double)running_mean[feature]) *
                inverse_std +
                (double)beta[feature];
        }
    }
}


static void update_stats(
    GradientStats *stats,
    size_t index,
    double analytic,
    double numeric
)
{
    double absolute = fabs(analytic - numeric);
    double denominator = fabs(analytic) + fabs(numeric);
    double relative;

    if (denominator < 1.0e-12) {
        denominator = 1.0e-12;
    }
    relative = absolute / denominator;
    if (absolute > stats->max_absolute) {
        stats->max_absolute = absolute;
        stats->worst_index = index;
    }
    if (relative > stats->max_relative) {
        stats->max_relative = relative;
    }
}


static int gradient_passes(double analytic, double numeric)
{
    double scale = fabs(analytic);

    if (fabs(numeric) > scale) {
        scale = fabs(numeric);
    }
    return fabs(analytic - numeric) <= 3.0e-3 + 3.0e-2 * scale;
}


static int test_lifecycle_and_layout(void)
{
    VAECBatchNorm one_d;
    VAECBatchNorm two_d;
    VAECBatchNorm snapshot;
    VAECBatchNorm corrupted;
    const size_t overflow_regions = SIZE_MAX / 7 + 1;
    const size_t overflow_bytes = SIZE_MAX / (7 * sizeof(float)) + 1;
    size_t feature;
    int passed;

    vae_c_batch_norm_init(&one_d);
    vae_c_batch_norm_init(&two_d);
    if (!vae_c_batch_norm_is_empty(&one_d) ||
        vae_c_batch_norm_zero_grad(&one_d) != VAE_C_OP_OK) {

        return 0;
    }
    snapshot = one_d;
    if (vae_c_batch_norm1d_allocate(NULL, 2, 0, 0) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        vae_c_batch_norm1d_allocate(&one_d, 0, 0, 0) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        memcmp(&one_d, &snapshot, sizeof(one_d)) != 0 ||
        vae_c_batch_norm1d_allocate(
            &one_d,
            overflow_regions,
            0,
            0) != VAE_C_OP_OVERFLOW ||
        memcmp(&one_d, &snapshot, sizeof(one_d)) != 0 ||
        vae_c_batch_norm1d_allocate(
            &one_d,
            overflow_bytes,
            0,
            0) != VAE_C_OP_OVERFLOW ||
        memcmp(&one_d, &snapshot, sizeof(one_d)) != 0 ||
        vae_c_batch_norm1d_allocate(&one_d, 3, 0, 0) != VAE_C_OP_OK ||
        vae_c_batch_norm2d_allocate(&two_d, 2, 0, 0) != VAE_C_OP_OK) {

        vae_c_batch_norm_free(&two_d);
        vae_c_batch_norm_free(&one_d);
        return 0;
    }
    passed =
        one_d.kind == VAE_C_BATCH_NORM_KIND_1D &&
        two_d.kind == VAE_C_BATCH_NORM_KIND_2D &&
        one_d.epsilon == VAE_C_BATCH_NORM_EPSILON &&
        one_d.momentum == VAE_C_BATCH_NORM_MOMENTUM &&
        one_d.owned_bytes == 3 * 7 * sizeof(float) &&
        one_d.running_mean == one_d.owned_block &&
        one_d.running_variance == one_d.owned_block + 3 &&
        one_d.batch_mean == one_d.owned_block + 6 &&
        one_d.batch_inv_std == one_d.owned_block + 9 &&
        one_d.work_mean == one_d.owned_block + 12 &&
        one_d.work_inv_std == one_d.owned_block + 15 &&
        one_d.work_unbiased_variance == one_d.owned_block + 18 &&
        !one_d.cache_valid && one_d.last_sample_count == 0 &&
        shape_is_zero(&one_d.last_input_shape) &&
        shape_is_zero(&one_d.last_output_shape);
    for (feature = 0; passed && feature < 3; feature++) {
        passed =
            one_d.gamma.values[feature] == 1.0f &&
            one_d.beta.values[feature] == 0.0f &&
            one_d.running_mean[feature] == 0.0f &&
            one_d.running_variance[feature] == 1.0f &&
            one_d.batch_mean[feature] == 0.0f &&
            one_d.batch_inv_std[feature] == 0.0f;
    }
    snapshot = one_d;
    passed =
        passed &&
        vae_c_batch_norm1d_allocate(&one_d, 3, 1, 2) ==
            VAE_C_OP_OBJECT_NOT_EMPTY &&
        memcmp(&one_d, &snapshot, sizeof(one_d)) == 0;
    corrupted = one_d;
    corrupted.running_mean++;
    passed =
        passed &&
        vae_c_batch_norm_zero_grad(&corrupted) == VAE_C_OP_INVALID_STATE;

    vae_c_batch_norm_free(&two_d);
    vae_c_batch_norm_free(&one_d);
    vae_c_batch_norm_free(&one_d);
    vae_c_batch_norm_free(NULL);
    return
        passed &&
        vae_c_batch_norm_is_empty(&one_d) &&
        vae_c_batch_norm_is_empty(&two_d);
}


static int test_batch_norm1d_analytic(void)
{
    const size_t dims[2] = {3, 2};
    const float input_values[6] = {1, 2, 3, 6, 5, 4};
    const float upstream[6] = {0.5f, -1.0f, 1.5f, 0.25f, -0.75f, 2.0f};
    VAECBatchNorm layer;
    VAECTensor input;
    VAECTensor output;
    double mean[2];
    double inverse_std[2];
    double unbiased[2];
    double expected_output[6];
    double running_mean_new[2];
    double running_variance_new[2];
    double expected_input_grad[6];
    double expected_gamma_grad[2];
    double expected_beta_grad[2];
    float output_grad_snapshot[6];
    float gamma_values_snapshot[2];
    float beta_values_snapshot[2];
    float gamma_adam_snapshot[4];
    float beta_adam_snapshot[4];
    float internal_snapshot[14];
    float running_mean_old[2] = {0.5f, -1.0f};
    float running_variance_old[2] = {2.0f, 3.0f};
    size_t index;
    int passed;

    vae_c_batch_norm_init(&layer);
    vae_c_tensor_init(&input);
    vae_c_tensor_init(&output);
    if (vae_c_batch_norm1d_allocate(&layer, 2, 11, 12) != VAE_C_OP_OK ||
        !make_tensor(2, dims, true, &input) ||
        !make_tensor(2, dims, true, &output)) {

        vae_c_tensor_free(&output);
        vae_c_tensor_free(&input);
        vae_c_batch_norm_free(&layer);
        return 0;
    }
    memcpy(input.data, input_values, sizeof(input_values));
    memcpy(output.grad, upstream, sizeof(upstream));
    layer.gamma.values[0] = 1.5f;
    layer.gamma.values[1] = -0.75f;
    layer.beta.values[0] = 0.25f;
    layer.beta.values[1] = 1.0f;
    memcpy(layer.running_mean, running_mean_old, sizeof(running_mean_old));
    memcpy(
        layer.running_variance,
        running_variance_old,
        sizeof(running_variance_old));
    for (index = 0; index < 2; index++) {
        layer.gamma.adam_m[index] = (float)(index + 1) * 0.1f;
        layer.gamma.adam_v[index] = (float)(index + 1) * 0.2f;
        layer.beta.adam_m[index] = (float)(index + 1) * 0.3f;
        layer.beta.adam_v[index] = (float)(index + 1) * 0.4f;
    }
    memcpy(output_grad_snapshot, output.grad, sizeof(output_grad_snapshot));
    memcpy(gamma_values_snapshot, layer.gamma.values, sizeof(gamma_values_snapshot));
    memcpy(beta_values_snapshot, layer.beta.values, sizeof(beta_values_snapshot));
    memcpy(gamma_adam_snapshot, layer.gamma.adam_m, sizeof(gamma_adam_snapshot));
    memcpy(beta_adam_snapshot, layer.beta.adam_m, sizeof(beta_adam_snapshot));

    reference_train(
        VAE_C_BATCH_NORM_KIND_1D,
        2,
        3,
        1,
        input.data,
        layer.gamma.values,
        layer.beta.values,
        running_mean_old,
        running_variance_old,
        mean,
        inverse_std,
        unbiased,
        expected_output,
        running_mean_new,
        running_variance_new
    );
    reference_backward(
        VAE_C_BATCH_NORM_KIND_1D,
        2,
        3,
        1,
        input.data,
        layer.gamma.values,
        upstream,
        mean,
        inverse_std,
        expected_input_grad,
        expected_gamma_grad,
        expected_beta_grad
    );
    passed =
        vae_c_batch_norm1d_forward(
            &layer,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &input,
            &output) == VAE_C_OP_OK &&
        float_array_close_double(output.data, expected_output, 6, 3.0e-5) &&
        float_array_close_double(layer.batch_mean, mean, 2, 2.0e-6) &&
        float_array_close_double(layer.batch_inv_std, inverse_std, 2, 3.0e-5) &&
        float_array_close_double(layer.running_mean, running_mean_new, 2, 2.0e-6) &&
        float_array_close_double(
            layer.running_variance,
            running_variance_new,
            2,
            3.0e-6) &&
        layer.cache_valid && layer.last_sample_count == 3 &&
        shape_fields_equal(&layer.last_input_shape, &input.shape) &&
        shape_fields_equal(&layer.last_output_shape, &output.shape) &&
        float_array_equal(output.grad, output_grad_snapshot, 6) &&
        vae_c_batch_norm1d_backward(&layer, &input, &output) == VAE_C_OP_OK &&
        float_array_close_double(input.grad, expected_input_grad, 6, 2.0e-4) &&
        float_array_close_double(
            layer.gamma.gradients,
            expected_gamma_grad,
            2,
            2.0e-4) &&
        float_array_close_double(
            layer.beta.gradients,
            expected_beta_grad,
            2,
            2.0e-6) &&
        vae_c_batch_norm1d_backward(&layer, &input, &output) == VAE_C_OP_OK;
    for (index = 0; passed && index < 6; index++) {
        passed = close_double(
            (double)input.grad[index],
            2.0 * expected_input_grad[index],
            4.0e-4);
    }
    for (index = 0; passed && index < 2; index++) {
        passed =
            close_double(
                (double)layer.gamma.gradients[index],
                2.0 * expected_gamma_grad[index],
                4.0e-4) &&
            close_double(
                (double)layer.beta.gradients[index],
                2.0 * expected_beta_grad[index],
                4.0e-6);
    }
    memcpy(internal_snapshot, layer.owned_block, sizeof(internal_snapshot));
    passed =
        passed &&
        vae_c_batch_norm_zero_grad(&layer) == VAE_C_OP_OK &&
        layer.gamma.gradients[0] == 0.0f &&
        layer.gamma.gradients[1] == 0.0f &&
        layer.beta.gradients[0] == 0.0f &&
        layer.beta.gradients[1] == 0.0f &&
        float_array_equal(layer.gamma.values, gamma_values_snapshot, 2) &&
        float_array_equal(layer.beta.values, beta_values_snapshot, 2) &&
        float_array_equal(layer.gamma.adam_m, gamma_adam_snapshot, 4) &&
        float_array_equal(layer.beta.adam_m, beta_adam_snapshot, 4) &&
        float_array_equal(layer.owned_block, internal_snapshot, 14) &&
        float_array_equal(output.grad, output_grad_snapshot, 6) &&
        layer.cache_valid;

    vae_c_tensor_free(&output);
    vae_c_tensor_free(&input);
    vae_c_batch_norm_free(&layer);
    return passed;
}


static int test_batch_norm2d_analytic(void)
{
    const size_t dims[4] = {2, 2, 2, 2};
    const float input_values[16] = {
        1, 2, 3, 4, 10, 8, 6, 4,
        5, 7, 9, 11, 2, 3, 5, 7
    };
    const float upstream[16] = {
        0.2f, -0.4f, 0.6f, 0.8f,
        -1.0f, 0.3f, 0.5f, -0.7f,
        1.1f, -0.9f, 0.4f, 0.2f,
        -0.3f, 0.7f, -0.5f, 1.3f
    };
    VAECBatchNorm layer;
    VAECTensor input;
    VAECTensor output;
    double mean[2];
    double inverse_std[2];
    double unbiased[2];
    double expected_output[16];
    double running_mean_new[2];
    double running_variance_new[2];
    double expected_input_grad[16];
    double expected_gamma_grad[2];
    double expected_beta_grad[2];
    float running_mean_old[2] = {0.0f, 0.0f};
    float running_variance_old[2] = {1.0f, 1.0f};
    float output_grad_snapshot[16];
    float gamma_values_snapshot[2];
    float beta_values_snapshot[2];
    float gamma_adam_snapshot[4];
    float beta_adam_snapshot[4];
    size_t index;
    int passed;

    vae_c_batch_norm_init(&layer);
    vae_c_tensor_init(&input);
    vae_c_tensor_init(&output);
    if (vae_c_batch_norm2d_allocate(&layer, 2, 21, 22) != VAE_C_OP_OK ||
        !make_tensor(4, dims, true, &input) ||
        !make_tensor(4, dims, true, &output)) {

        vae_c_tensor_free(&output);
        vae_c_tensor_free(&input);
        vae_c_batch_norm_free(&layer);
        return 0;
    }
    memcpy(input.data, input_values, sizeof(input_values));
    memcpy(output.grad, upstream, sizeof(upstream));
    layer.gamma.values[0] = 0.75f;
    layer.gamma.values[1] = -1.25f;
    layer.beta.values[0] = -0.5f;
    layer.beta.values[1] = 0.2f;
    memcpy(output_grad_snapshot, output.grad, sizeof(output_grad_snapshot));
    memcpy(gamma_values_snapshot, layer.gamma.values, sizeof(gamma_values_snapshot));
    memcpy(beta_values_snapshot, layer.beta.values, sizeof(beta_values_snapshot));
    memcpy(gamma_adam_snapshot, layer.gamma.adam_m, sizeof(gamma_adam_snapshot));
    memcpy(beta_adam_snapshot, layer.beta.adam_m, sizeof(beta_adam_snapshot));
    reference_train(
        VAE_C_BATCH_NORM_KIND_2D,
        2,
        8,
        4,
        input.data,
        layer.gamma.values,
        layer.beta.values,
        running_mean_old,
        running_variance_old,
        mean,
        inverse_std,
        unbiased,
        expected_output,
        running_mean_new,
        running_variance_new
    );
    reference_backward(
        VAE_C_BATCH_NORM_KIND_2D,
        2,
        8,
        4,
        input.data,
        layer.gamma.values,
        upstream,
        mean,
        inverse_std,
        expected_input_grad,
        expected_gamma_grad,
        expected_beta_grad
    );
    passed =
        vae_c_batch_norm2d_forward(
            &layer,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &input,
            &output) == VAE_C_OP_OK &&
        float_array_close_double(output.data, expected_output, 16, 4.0e-5) &&
        float_array_close_double(layer.batch_mean, mean, 2, 3.0e-6) &&
        float_array_close_double(layer.batch_inv_std, inverse_std, 2, 3.0e-5) &&
        float_array_close_double(layer.running_mean, running_mean_new, 2, 3.0e-6) &&
        float_array_close_double(
            layer.running_variance,
            running_variance_new,
            2,
            5.0e-6) &&
        layer.cache_valid && layer.last_sample_count == 8 &&
        vae_c_batch_norm2d_backward(&layer, &input, &output) == VAE_C_OP_OK &&
        float_array_close_double(input.grad, expected_input_grad, 16, 3.0e-4) &&
        float_array_close_double(
            layer.gamma.gradients,
            expected_gamma_grad,
            2,
            3.0e-4) &&
        float_array_close_double(
            layer.beta.gradients,
            expected_beta_grad,
            2,
            3.0e-5) &&
        float_array_equal(output.grad, output_grad_snapshot, 16) &&
        vae_c_batch_norm2d_backward(&layer, &input, &output) == VAE_C_OP_OK;
    for (index = 0; passed && index < 16; index++) {
        passed = close_double(
            (double)input.grad[index],
            2.0 * expected_input_grad[index],
            6.0e-4);
    }
    for (index = 0; passed && index < 2; index++) {
        passed =
            close_double(
                (double)layer.gamma.gradients[index],
                2.0 * expected_gamma_grad[index],
                6.0e-4) &&
            close_double(
                (double)layer.beta.gradients[index],
                2.0 * expected_beta_grad[index],
                6.0e-5);
    }
    passed =
        passed &&
        float_array_equal(layer.gamma.values, gamma_values_snapshot, 2) &&
        float_array_equal(layer.beta.values, beta_values_snapshot, 2) &&
        float_array_equal(layer.gamma.adam_m, gamma_adam_snapshot, 4) &&
        float_array_equal(layer.beta.adam_m, beta_adam_snapshot, 4) &&
        float_array_equal(output.grad, output_grad_snapshot, 16);
    vae_c_tensor_free(&output);
    vae_c_tensor_free(&input);
    vae_c_batch_norm_free(&layer);
    return passed;
}


static int test_running_statistics_and_evaluation(void)
{
    const size_t train_dims[2] = {3, 2};
    const size_t eval_dims[2] = {1, 2};
    const size_t train2d_dims[4] = {2, 2, 1, 1};
    const size_t eval2d_dims[4] = {1, 2, 1, 1};
    const float first_batch[6] = {1, 2, 3, 4, 5, 8};
    const float second_batch[6] = {-2, 1, 0, 5, 4, 7};
    VAECBatchNorm one_d;
    VAECBatchNorm two_d;
    VAECTensor train_input;
    VAECTensor train_output;
    VAECTensor eval_input;
    VAECTensor eval_output;
    VAECTensor train2d_input;
    VAECTensor train2d_output;
    VAECTensor eval2d_input;
    VAECTensor eval2d_output;
    double mean[2];
    double inv[2];
    double unbiased[2];
    double expected[6];
    double running_mean_new[2];
    double running_variance_new[2];
    double eval_expected[2];
    float running_mean_old[2];
    float running_variance_old[2];
    float running_snapshot[4];
    float input_grad_snapshot[6];
    float train2d_grad_snapshot[4];
    float gamma_parameter_snapshot[8];
    float beta_parameter_snapshot[8];
    int passed = 0;

    vae_c_batch_norm_init(&one_d);
    vae_c_batch_norm_init(&two_d);
    vae_c_tensor_init(&train_input);
    vae_c_tensor_init(&train_output);
    vae_c_tensor_init(&eval_input);
    vae_c_tensor_init(&eval_output);
    vae_c_tensor_init(&train2d_input);
    vae_c_tensor_init(&train2d_output);
    vae_c_tensor_init(&eval2d_input);
    vae_c_tensor_init(&eval2d_output);
    if (vae_c_batch_norm1d_allocate(&one_d, 2, 31, 32) != VAE_C_OP_OK ||
        vae_c_batch_norm2d_allocate(&two_d, 2, 33, 34) != VAE_C_OP_OK ||
        !make_tensor(2, train_dims, true, &train_input) ||
        !make_tensor(2, train_dims, true, &train_output) ||
        !make_tensor(2, eval_dims, true, &eval_input) ||
        !make_tensor(2, eval_dims, true, &eval_output) ||
        !make_tensor(4, train2d_dims, true, &train2d_input) ||
        !make_tensor(4, train2d_dims, true, &train2d_output) ||
        !make_tensor(4, eval2d_dims, true, &eval2d_input) ||
        !make_tensor(4, eval2d_dims, true, &eval2d_output)) {

        goto cleanup;
    }
    memcpy(train_input.data, first_batch, sizeof(first_batch));
    memcpy(running_mean_old, one_d.running_mean, sizeof(running_mean_old));
    memcpy(
        running_variance_old,
        one_d.running_variance,
        sizeof(running_variance_old));
    reference_train(
        VAE_C_BATCH_NORM_KIND_1D, 2, 3, 1,
        train_input.data, one_d.gamma.values, one_d.beta.values,
        running_mean_old, running_variance_old,
        mean, inv, unbiased, expected,
        running_mean_new, running_variance_new);
    if (vae_c_batch_norm1d_forward(
            &one_d,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &train_input,
            &train_output) != VAE_C_OP_OK ||
        !float_array_close_double(one_d.running_mean, running_mean_new, 2, 3e-6) ||
        !float_array_close_double(one_d.running_variance, running_variance_new, 2, 4e-6) ||
        !float_array_close_double(train_output.data, expected, 6, 3e-5)) {

        goto cleanup;
    }
    memcpy(train_input.data, second_batch, sizeof(second_batch));
    memcpy(running_mean_old, one_d.running_mean, sizeof(running_mean_old));
    memcpy(
        running_variance_old,
        one_d.running_variance,
        sizeof(running_variance_old));
    reference_train(
        VAE_C_BATCH_NORM_KIND_1D, 2, 3, 1,
        train_input.data, one_d.gamma.values, one_d.beta.values,
        running_mean_old, running_variance_old,
        mean, inv, unbiased, expected,
        running_mean_new, running_variance_new);
    if (vae_c_batch_norm1d_forward(
            &one_d,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &train_input,
            &train_output) != VAE_C_OP_OK ||
        !float_array_close_double(one_d.running_mean, running_mean_new, 2, 3e-6) ||
        !float_array_close_double(one_d.running_variance, running_variance_new, 2, 5e-6) ||
        !float_array_close_double(train_output.data, expected, 6, 3e-5)) {

        goto cleanup;
    }

    one_d.running_mean[0] = 0.4f;
    one_d.running_mean[1] = -0.2f;
    one_d.running_variance[0] = 1.5f;
    one_d.running_variance[1] = 0.5f;
    one_d.gamma.values[0] = 1.2f;
    one_d.gamma.values[1] = -0.8f;
    one_d.beta.values[0] = 0.1f;
    one_d.beta.values[1] = 0.7f;
    eval_input.data[0] = 2.0f;
    eval_input.data[1] = -1.0f;
    reference_eval(
        VAE_C_BATCH_NORM_KIND_1D, 2, 1, 1,
        eval_input.data, one_d.gamma.values, one_d.beta.values,
        one_d.running_mean, one_d.running_variance, eval_expected);
    memcpy(running_snapshot, one_d.running_mean, sizeof(running_snapshot));
    memcpy(input_grad_snapshot, train_input.grad, sizeof(input_grad_snapshot));
    memcpy(
        gamma_parameter_snapshot,
        one_d.gamma.owned_block,
        sizeof(gamma_parameter_snapshot));
    memcpy(
        beta_parameter_snapshot,
        one_d.beta.owned_block,
        sizeof(beta_parameter_snapshot));
    if (vae_c_batch_norm1d_forward(
            &one_d,
            VAE_C_BATCH_NORM_MODE_EVAL,
            &eval_input,
            &eval_output) != VAE_C_OP_OK ||
        !float_array_close_double(eval_output.data, eval_expected, 2, 3e-6) ||
        !float_array_equal(one_d.running_mean, running_snapshot, 4) ||
        !float_array_equal(
            one_d.gamma.owned_block,
            gamma_parameter_snapshot,
            8) ||
        !float_array_equal(
            one_d.beta.owned_block,
            beta_parameter_snapshot,
            8) ||
        one_d.cache_valid || one_d.last_sample_count != 0 ||
        !shape_is_zero(&one_d.last_input_shape) ||
        !shape_is_zero(&one_d.last_output_shape) ||
        vae_c_batch_norm1d_backward(
            &one_d,
            &train_input,
            &train_output) != VAE_C_OP_INVALID_STATE ||
        !float_array_equal(train_input.grad, input_grad_snapshot, 6)) {

        goto cleanup;
    }

    train2d_input.data[0] = 1.0f;
    train2d_input.data[1] = 4.0f;
    train2d_input.data[2] = 3.0f;
    train2d_input.data[3] = 8.0f;
    if (vae_c_batch_norm2d_forward(
            &two_d,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &train2d_input,
            &train2d_output) != VAE_C_OP_OK) {

        goto cleanup;
    }
    two_d.running_mean[0] = -0.5f;
    two_d.running_mean[1] = 0.25f;
    two_d.running_variance[0] = 2.0f;
    two_d.running_variance[1] = 0.75f;
    eval2d_input.data[0] = 0.5f;
    eval2d_input.data[1] = 2.0f;
    reference_eval(
        VAE_C_BATCH_NORM_KIND_2D, 2, 1, 1,
        eval2d_input.data, two_d.gamma.values, two_d.beta.values,
        two_d.running_mean, two_d.running_variance, eval_expected);
    memcpy(running_snapshot, two_d.running_mean, sizeof(running_snapshot));
    memcpy(
        gamma_parameter_snapshot,
        two_d.gamma.owned_block,
        sizeof(gamma_parameter_snapshot));
    memcpy(
        beta_parameter_snapshot,
        two_d.beta.owned_block,
        sizeof(beta_parameter_snapshot));
    memcpy(
        train2d_grad_snapshot,
        train2d_input.grad,
        sizeof(train2d_grad_snapshot));
    passed =
        vae_c_batch_norm2d_forward(
            &two_d,
            VAE_C_BATCH_NORM_MODE_EVAL,
            &eval2d_input,
            &eval2d_output) == VAE_C_OP_OK &&
        float_array_close_double(eval2d_output.data, eval_expected, 2, 3e-6) &&
        float_array_equal(two_d.running_mean, running_snapshot, 4) &&
        float_array_equal(
            two_d.gamma.owned_block,
            gamma_parameter_snapshot,
            8) &&
        float_array_equal(
            two_d.beta.owned_block,
            beta_parameter_snapshot,
            8) &&
        !two_d.cache_valid &&
        vae_c_batch_norm2d_backward(
            &two_d,
            &train2d_input,
            &train2d_output) == VAE_C_OP_INVALID_STATE &&
        float_array_equal(train2d_input.grad, train2d_grad_snapshot, 4);

cleanup:
    vae_c_tensor_free(&eval2d_output);
    vae_c_tensor_free(&eval2d_input);
    vae_c_tensor_free(&train2d_output);
    vae_c_tensor_free(&train2d_input);
    vae_c_tensor_free(&eval_output);
    vae_c_tensor_free(&eval_input);
    vae_c_tensor_free(&train_output);
    vae_c_tensor_free(&train_input);
    vae_c_batch_norm_free(&two_d);
    vae_c_batch_norm_free(&one_d);
    return passed;
}


static double training_objective(
    VAECBatchNorm *layer,
    VAECBatchNormKind kind,
    VAECTensor *input,
    VAECTensor *output,
    const float *upstream
)
{
    VAECOpStatus status;
    double objective = 0.0;
    size_t index;

    status =
        kind == VAE_C_BATCH_NORM_KIND_1D
            ? vae_c_batch_norm1d_forward(
                layer,
                VAE_C_BATCH_NORM_MODE_TRAIN,
                input,
                output)
            : vae_c_batch_norm2d_forward(
                layer,
                VAE_C_BATCH_NORM_MODE_TRAIN,
                input,
                output);
    if (status != VAE_C_OP_OK) {
        return NAN;
    }
    for (index = 0; index < output->shape.count; index++) {
        objective +=
            (double)output->data[index] * (double)upstream[index];
    }
    return objective;
}


static int run_gradient_check(VAECBatchNormKind kind)
{
    const size_t dims1d[2] = {3, 2};
    const size_t dims2d[4] = {2, 2, 2, 2};
    const float values1d[6] = {-1.2f, 0.3f, 0.7f, 2.1f, 3.4f, -0.8f};
    const float values2d[16] = {
        -1.0f, 0.2f, 1.5f, 2.7f,
        0.4f, -0.6f, 3.1f, 1.2f,
        2.0f, -1.4f, 0.8f, 4.0f,
        -0.2f, 1.7f, 2.4f, -0.9f
    };
    const size_t count =
        kind == VAE_C_BATCH_NORM_KIND_1D ? 6 : 16;
    const float epsilon = 2.0e-3f;
    VAECBatchNorm layer;
    VAECTensor input;
    VAECTensor output;
    float upstream[16];
    float analytic_input[16];
    float analytic_gamma[2];
    float analytic_beta[2];
    float running_mean_initial[2];
    float running_variance_initial[2];
    GradientStats input_stats = {0.0, 0.0, 0};
    GradientStats gamma_stats = {0.0, 0.0, 0};
    GradientStats beta_stats = {0.0, 0.0, 0};
    size_t index;
    int passed = 1;

    vae_c_batch_norm_init(&layer);
    vae_c_tensor_init(&input);
    vae_c_tensor_init(&output);
    if ((kind == VAE_C_BATCH_NORM_KIND_1D
            ? vae_c_batch_norm1d_allocate(&layer, 2, 41, 42)
            : vae_c_batch_norm2d_allocate(&layer, 2, 41, 42)) !=
            VAE_C_OP_OK ||
        !make_tensor(
            kind == VAE_C_BATCH_NORM_KIND_1D ? 2 : 4,
            kind == VAE_C_BATCH_NORM_KIND_1D ? dims1d : dims2d,
            true,
            &input) ||
        !make_tensor(
            kind == VAE_C_BATCH_NORM_KIND_1D ? 2 : 4,
            kind == VAE_C_BATCH_NORM_KIND_1D ? dims1d : dims2d,
            true,
            &output)) {

        passed = 0;
        goto cleanup;
    }
    memcpy(
        input.data,
        kind == VAE_C_BATCH_NORM_KIND_1D ? values1d : values2d,
        count * sizeof(float));
    layer.gamma.values[0] = 1.3f;
    layer.gamma.values[1] = -0.7f;
    layer.beta.values[0] = 0.2f;
    layer.beta.values[1] = -0.4f;
    for (index = 0; index < count; index++) {
        upstream[index] =
            (float)((int)(index % 7) - 3) * 0.17f +
            (float)index * 0.01f;
    }
    memcpy(output.grad, upstream, count * sizeof(float));
    memcpy(running_mean_initial, layer.running_mean, sizeof(running_mean_initial));
    memcpy(
        running_variance_initial,
        layer.running_variance,
        sizeof(running_variance_initial));
    if ((kind == VAE_C_BATCH_NORM_KIND_1D
            ? vae_c_batch_norm1d_forward(
                &layer,
                VAE_C_BATCH_NORM_MODE_TRAIN,
                &input,
                &output)
            : vae_c_batch_norm2d_forward(
                &layer,
                VAE_C_BATCH_NORM_MODE_TRAIN,
                &input,
                &output)) != VAE_C_OP_OK ||
        vae_c_tensor_zero_grad(&input) != VAE_C_STORAGE_OK ||
        vae_c_batch_norm_zero_grad(&layer) != VAE_C_OP_OK ||
        (kind == VAE_C_BATCH_NORM_KIND_1D
            ? vae_c_batch_norm1d_backward(&layer, &input, &output)
            : vae_c_batch_norm2d_backward(&layer, &input, &output)) !=
            VAE_C_OP_OK) {

        passed = 0;
        goto cleanup;
    }
    memcpy(analytic_input, input.grad, count * sizeof(float));
    memcpy(analytic_gamma, layer.gamma.gradients, sizeof(analytic_gamma));
    memcpy(analytic_beta, layer.beta.gradients, sizeof(analytic_beta));

    for (index = 0; index < count; index++) {
        float original = input.data[index];
        double plus;
        double minus;
        double numeric;

        memcpy(layer.running_mean, running_mean_initial, sizeof(running_mean_initial));
        memcpy(layer.running_variance, running_variance_initial, sizeof(running_variance_initial));
        input.data[index] = original + epsilon;
        plus = training_objective(&layer, kind, &input, &output, upstream);
        memcpy(layer.running_mean, running_mean_initial, sizeof(running_mean_initial));
        memcpy(layer.running_variance, running_variance_initial, sizeof(running_variance_initial));
        input.data[index] = original - epsilon;
        minus = training_objective(&layer, kind, &input, &output, upstream);
        input.data[index] = original;
        numeric = (plus - minus) / (2.0 * (double)epsilon);
        update_stats(&input_stats, index, (double)analytic_input[index], numeric);
        if (!gradient_passes((double)analytic_input[index], numeric)) {
            passed = 0;
        }
    }
    for (index = 0; index < 2; index++) {
        float original = layer.gamma.values[index];
        double plus;
        double minus;
        double numeric;

        memcpy(layer.running_mean, running_mean_initial, sizeof(running_mean_initial));
        memcpy(layer.running_variance, running_variance_initial, sizeof(running_variance_initial));
        layer.gamma.values[index] = original + epsilon;
        plus = training_objective(&layer, kind, &input, &output, upstream);
        memcpy(layer.running_mean, running_mean_initial, sizeof(running_mean_initial));
        memcpy(layer.running_variance, running_variance_initial, sizeof(running_variance_initial));
        layer.gamma.values[index] = original - epsilon;
        minus = training_objective(&layer, kind, &input, &output, upstream);
        layer.gamma.values[index] = original;
        numeric = (plus - minus) / (2.0 * (double)epsilon);
        update_stats(&gamma_stats, index, (double)analytic_gamma[index], numeric);
        if (!gradient_passes((double)analytic_gamma[index], numeric)) {
            passed = 0;
        }
    }
    for (index = 0; index < 2; index++) {
        float original = layer.beta.values[index];
        double plus;
        double minus;
        double numeric;

        memcpy(layer.running_mean, running_mean_initial, sizeof(running_mean_initial));
        memcpy(layer.running_variance, running_variance_initial, sizeof(running_variance_initial));
        layer.beta.values[index] = original + epsilon;
        plus = training_objective(&layer, kind, &input, &output, upstream);
        memcpy(layer.running_mean, running_mean_initial, sizeof(running_mean_initial));
        memcpy(layer.running_variance, running_variance_initial, sizeof(running_variance_initial));
        layer.beta.values[index] = original - epsilon;
        minus = training_objective(&layer, kind, &input, &output, upstream);
        layer.beta.values[index] = original;
        numeric = (plus - minus) / (2.0 * (double)epsilon);
        update_stats(&beta_stats, index, (double)analytic_beta[index], numeric);
        if (!gradient_passes((double)analytic_beta[index], numeric)) {
            passed = 0;
        }
    }
    printf(
        "BN%u_INPUT_GRAD_MAX_ABS=%.9g MAX_REL=%.9g WORST_INDEX=%zu\n",
        (unsigned int)kind,
        input_stats.max_absolute,
        input_stats.max_relative,
        input_stats.worst_index
    );
    printf(
        "BN%u_GAMMA_GRAD_MAX_ABS=%.9g MAX_REL=%.9g WORST_INDEX=%zu\n",
        (unsigned int)kind,
        gamma_stats.max_absolute,
        gamma_stats.max_relative,
        gamma_stats.worst_index
    );
    printf(
        "BN%u_BETA_GRAD_MAX_ABS=%.9g MAX_REL=%.9g WORST_INDEX=%zu\n",
        (unsigned int)kind,
        beta_stats.max_absolute,
        beta_stats.max_relative,
        beta_stats.worst_index
    );

cleanup:
    vae_c_tensor_free(&output);
    vae_c_tensor_free(&input);
    vae_c_batch_norm_free(&layer);
    return passed;
}


static int test_batch_norm1d_errors_and_strong_guarantee(void)
{
    const size_t dims[2] = {3, 2};
    const size_t rank_one_dims[1] = {6};
    const size_t wrong_shape_dims[2] = {2, 2};
    const size_t wrong_feature_dims[2] = {3, 3};
    const size_t singleton_dims[2] = {1, 2};
    VAECBatchNorm layer;
    VAECBatchNorm fresh;
    VAECBatchNorm corrupted_layer;
    VAECTensor input;
    VAECTensor output;
    VAECTensor rank_one;
    VAECTensor wrong_shape_input;
    VAECTensor wrong_shape_output;
    VAECTensor wrong_feature;
    VAECTensor singleton_input;
    VAECTensor singleton_output;
    VAECTensor data_only_input;
    VAECTensor data_only_output;
    VAECTensor alias_output;
    float output_snapshot[6];
    float running_snapshot[4];
    float cache_snapshot[4];
    float input_grad_snapshot[6];
    float gamma_grad_snapshot[2];
    float beta_grad_snapshot[2];
    VAECShape input_shape_snapshot;
    VAECShape output_shape_snapshot;
    size_t sample_count_snapshot;
    float saved;
    size_t index;
    int passed = 0;

    vae_c_batch_norm_init(&layer);
    vae_c_batch_norm_init(&fresh);
    vae_c_tensor_init(&input);
    vae_c_tensor_init(&output);
    vae_c_tensor_init(&rank_one);
    vae_c_tensor_init(&wrong_shape_input);
    vae_c_tensor_init(&wrong_shape_output);
    vae_c_tensor_init(&wrong_feature);
    vae_c_tensor_init(&singleton_input);
    vae_c_tensor_init(&singleton_output);
    vae_c_tensor_init(&data_only_input);
    vae_c_tensor_init(&data_only_output);
    if (vae_c_batch_norm1d_allocate(&layer, 2, 51, 52) != VAE_C_OP_OK ||
        vae_c_batch_norm1d_allocate(&fresh, 2, 53, 54) != VAE_C_OP_OK ||
        !make_tensor(2, dims, true, &input) ||
        !make_tensor(2, dims, true, &output) ||
        !make_tensor(1, rank_one_dims, true, &rank_one) ||
        !make_tensor(2, wrong_shape_dims, true, &wrong_shape_input) ||
        !make_tensor(2, wrong_shape_dims, true, &wrong_shape_output) ||
        !make_tensor(2, wrong_feature_dims, true, &wrong_feature) ||
        !make_tensor(2, singleton_dims, true, &singleton_input) ||
        !make_tensor(2, singleton_dims, true, &singleton_output) ||
        !make_tensor(2, dims, false, &data_only_input) ||
        !make_tensor(2, dims, false, &data_only_output)) {

        goto cleanup;
    }
    for (index = 0; index < 6; index++) {
        input.data[index] = (float)index - 2.0f;
        output.data[index] = 71.0f;
        output.grad[index] = (float)index * 0.2f - 0.3f;
        input.grad[index] = 72.0f;
    }
    if (vae_c_batch_norm1d_forward(
            &layer,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &input,
            &output) != VAE_C_OP_OK) {

        goto cleanup;
    }
    memcpy(output_snapshot, output.data, sizeof(output_snapshot));
    memcpy(running_snapshot, layer.running_mean, sizeof(running_snapshot));
    memcpy(cache_snapshot, layer.batch_mean, sizeof(cache_snapshot));
    memcpy(input_grad_snapshot, input.grad, sizeof(input_grad_snapshot));
    memcpy(gamma_grad_snapshot, layer.gamma.gradients, sizeof(gamma_grad_snapshot));
    memcpy(beta_grad_snapshot, layer.beta.gradients, sizeof(beta_grad_snapshot));
    input_shape_snapshot = layer.last_input_shape;
    output_shape_snapshot = layer.last_output_shape;
    sample_count_snapshot = layer.last_sample_count;

    if (vae_c_batch_norm1d_forward(
            NULL,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &input,
            &output) != VAE_C_OP_INVALID_ARGUMENT ||
        vae_c_batch_norm1d_forward(
            &layer,
            (VAECBatchNormMode)99,
            &input,
            &output) != VAE_C_OP_INVALID_ARGUMENT ||
        vae_c_batch_norm2d_forward(
            &layer,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &input,
            &output) != VAE_C_OP_INVALID_STATE ||
        vae_c_batch_norm1d_forward(
            &layer,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &rank_one,
            &output) != VAE_C_OP_SHAPE_MISMATCH ||
        vae_c_batch_norm1d_forward(
            &layer,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &wrong_shape_input,
            &output) != VAE_C_OP_SHAPE_MISMATCH ||
        vae_c_batch_norm1d_forward(
            &layer,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &input,
            &wrong_feature) != VAE_C_OP_SHAPE_MISMATCH ||
        vae_c_batch_norm1d_forward(
            &layer,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &singleton_input,
            &singleton_output) != VAE_C_OP_SHAPE_MISMATCH ||
        vae_c_batch_norm1d_forward(
            &layer,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &input,
            &input) != VAE_C_OP_INVALID_ARGUMENT) {

        goto cleanup;
    }
    alias_output = output;
    alias_output.owns_memory = false;
    alias_output.owned_block = NULL;
    alias_output.owned_bytes = 0;
    alias_output.has_grad = false;
    alias_output.grad = NULL;
    alias_output.data = input.data;
    if (vae_c_batch_norm1d_forward(
            &layer,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &input,
            &alias_output) != VAE_C_OP_INVALID_ARGUMENT) {

        goto cleanup;
    }
    alias_output.data = layer.gamma.values;
    if (vae_c_batch_norm1d_forward(
            &layer,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &input,
            &alias_output) != VAE_C_OP_INVALID_ARGUMENT) {

        goto cleanup;
    }

    saved = input.data[0];
    input.data[0] = NAN;
    if (vae_c_batch_norm1d_forward(
            &layer,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &input,
            &output) != VAE_C_OP_INVALID_STATE ||
        vae_c_batch_norm1d_forward(
            &layer,
            VAE_C_BATCH_NORM_MODE_EVAL,
            &input,
            &output) != VAE_C_OP_INVALID_STATE) {
        goto cleanup;
    }
    input.data[0] = saved;
    saved = layer.gamma.values[0];
    layer.gamma.values[0] = NAN;
    if (vae_c_batch_norm1d_forward(
            &layer,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &input,
            &output) != VAE_C_OP_INVALID_STATE) {
        goto cleanup;
    }
    layer.gamma.values[0] = saved;
    saved = layer.beta.values[0];
    layer.beta.values[0] = INFINITY;
    if (vae_c_batch_norm1d_forward(
            &layer,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &input,
            &output) != VAE_C_OP_INVALID_STATE) {
        goto cleanup;
    }
    layer.beta.values[0] = saved;
    saved = layer.running_mean[0];
    layer.running_mean[0] = NAN;
    if (vae_c_batch_norm1d_forward(
            &layer,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &input,
            &output) != VAE_C_OP_INVALID_STATE) {
        goto cleanup;
    }
    layer.running_mean[0] = saved;
    saved = layer.running_variance[0];
    layer.running_variance[0] = -1.0f;
    if (vae_c_batch_norm1d_forward(
            &layer,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &input,
            &output) != VAE_C_OP_INVALID_STATE) {
        goto cleanup;
    }
    layer.running_variance[0] = saved;
    layer.running_variance[0] = INFINITY;
    if (vae_c_batch_norm1d_forward(
            &layer,
            VAE_C_BATCH_NORM_MODE_EVAL,
            &input,
            &output) != VAE_C_OP_INVALID_STATE) {
        goto cleanup;
    }
    layer.running_variance[0] = saved;
    saved = layer.gamma.values[0];
    layer.gamma.values[0] = FLT_MAX;
    layer.beta.values[0] = FLT_MAX;
    if (vae_c_batch_norm1d_forward(
            &layer,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &input,
            &output) != VAE_C_OP_INVALID_STATE) {
        goto cleanup;
    }
    layer.gamma.values[0] = saved;
    layer.beta.values[0] = 0.0f;

    if (!float_array_equal(output.data, output_snapshot, 6) ||
        !float_array_equal(layer.running_mean, running_snapshot, 4) ||
        !float_array_equal(layer.batch_mean, cache_snapshot, 4) ||
        !layer.cache_valid ||
        layer.last_sample_count != sample_count_snapshot ||
        !shape_fields_equal(&layer.last_input_shape, &input_shape_snapshot) ||
        !shape_fields_equal(&layer.last_output_shape, &output_shape_snapshot) ||
        vae_c_batch_norm1d_backward(&fresh, &input, &output) !=
            VAE_C_OP_INVALID_STATE ||
        vae_c_batch_norm1d_backward(
            &layer,
            &data_only_input,
            &output) != VAE_C_OP_GRADIENT_NOT_AVAILABLE ||
        vae_c_batch_norm1d_backward(
            &layer,
            &input,
            &data_only_output) != VAE_C_OP_GRADIENT_NOT_AVAILABLE ||
        vae_c_batch_norm1d_backward(
            &layer,
            &wrong_shape_input,
            &wrong_shape_output) != VAE_C_OP_INVALID_STATE) {

        goto cleanup;
    }
    corrupted_layer = layer;
    corrupted_layer.last_sample_count++;
    if (vae_c_batch_norm1d_backward(
            &corrupted_layer,
            &input,
            &output) != VAE_C_OP_INVALID_STATE) {

        goto cleanup;
    }
    saved = layer.batch_mean[0];
    layer.batch_mean[0] = NAN;
    if (vae_c_batch_norm1d_backward(&layer, &input, &output) !=
            VAE_C_OP_INVALID_STATE) {
        goto cleanup;
    }
    layer.batch_mean[0] = saved;
    saved = layer.batch_inv_std[0];
    layer.batch_inv_std[0] = 0.0f;
    if (vae_c_batch_norm1d_backward(&layer, &input, &output) !=
            VAE_C_OP_INVALID_STATE) {
        goto cleanup;
    }
    layer.batch_inv_std[0] = saved;
    layer.batch_inv_std[0] = NAN;
    if (vae_c_batch_norm1d_backward(&layer, &input, &output) !=
            VAE_C_OP_INVALID_STATE) {
        goto cleanup;
    }
    layer.batch_inv_std[0] = saved;
    saved = output.grad[0];
    output.grad[0] = NAN;
    if (vae_c_batch_norm1d_backward(&layer, &input, &output) !=
            VAE_C_OP_INVALID_STATE) {
        goto cleanup;
    }
    output.grad[0] = saved;
    saved = layer.gamma.values[0];
    layer.gamma.values[0] = NAN;
    if (vae_c_batch_norm1d_backward(&layer, &input, &output) !=
            VAE_C_OP_INVALID_STATE) {
        goto cleanup;
    }
    layer.gamma.values[0] = saved;

    alias_output = output;
    alias_output.owns_memory = false;
    alias_output.owned_block = NULL;
    alias_output.owned_bytes = 0;
    alias_output.data = output.data;
    alias_output.grad = input.grad;
    if (vae_c_batch_norm1d_backward(&layer, &input, &alias_output) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        !float_array_equal(input.grad, input_grad_snapshot, 6) ||
        !float_array_equal(
            layer.gamma.gradients,
            gamma_grad_snapshot,
            2) ||
        !float_array_equal(
            layer.beta.gradients,
            beta_grad_snapshot,
            2)) {

        goto cleanup;
    }
    passed = 1;

cleanup:
    vae_c_tensor_free(&data_only_output);
    vae_c_tensor_free(&data_only_input);
    vae_c_tensor_free(&singleton_output);
    vae_c_tensor_free(&singleton_input);
    vae_c_tensor_free(&wrong_feature);
    vae_c_tensor_free(&wrong_shape_output);
    vae_c_tensor_free(&wrong_shape_input);
    vae_c_tensor_free(&rank_one);
    vae_c_tensor_free(&output);
    vae_c_tensor_free(&input);
    vae_c_batch_norm_free(&fresh);
    vae_c_batch_norm_free(&layer);
    return passed;
}


static int test_batch_norm2d_errors_and_strong_guarantee(void)
{
    const size_t dims[4] = {2, 2, 1, 1};
    const size_t rank_two_dims[2] = {2, 2};
    const size_t wrong_shape_dims[4] = {2, 2, 1, 2};
    const size_t wrong_channel_dims[4] = {2, 3, 1, 1};
    const size_t singleton_dims[4] = {1, 2, 1, 1};
    VAECBatchNorm layer;
    VAECTensor input;
    VAECTensor output;
    VAECTensor rank_two;
    VAECTensor wrong_shape;
    VAECTensor wrong_channel;
    VAECTensor wrong_channel_output;
    VAECTensor singleton_input;
    VAECTensor singleton_output;
    VAECTensor overflow_input;
    float output_snapshot[4];
    size_t index;
    int passed = 0;

    vae_c_batch_norm_init(&layer);
    vae_c_tensor_init(&input);
    vae_c_tensor_init(&output);
    vae_c_tensor_init(&rank_two);
    vae_c_tensor_init(&wrong_shape);
    vae_c_tensor_init(&wrong_channel);
    vae_c_tensor_init(&wrong_channel_output);
    vae_c_tensor_init(&singleton_input);
    vae_c_tensor_init(&singleton_output);
    if (vae_c_batch_norm2d_allocate(&layer, 2, 61, 62) != VAE_C_OP_OK ||
        !make_tensor(4, dims, true, &input) ||
        !make_tensor(4, dims, true, &output) ||
        !make_tensor(2, rank_two_dims, true, &rank_two) ||
        !make_tensor(4, wrong_shape_dims, true, &wrong_shape) ||
        !make_tensor(4, wrong_channel_dims, true, &wrong_channel) ||
        !make_tensor(4, wrong_channel_dims, true, &wrong_channel_output) ||
        !make_tensor(4, singleton_dims, true, &singleton_input) ||
        !make_tensor(4, singleton_dims, true, &singleton_output)) {

        goto cleanup;
    }
    for (index = 0; index < 4; index++) {
        input.data[index] = (float)index + 1.0f;
        output.data[index] = 81.0f;
    }
    memcpy(output_snapshot, output.data, sizeof(output_snapshot));
    overflow_input = input;
    overflow_input.shape.dims[0] = SIZE_MAX;
    overflow_input.shape.dims[1] = 2;
    overflow_input.shape.dims[2] = 2;
    overflow_input.shape.dims[3] = 1;

    if (vae_c_batch_norm1d_forward(
            &layer,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &input,
            &output) != VAE_C_OP_INVALID_STATE ||
        vae_c_batch_norm2d_forward(
            &layer,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &rank_two,
            &output) != VAE_C_OP_SHAPE_MISMATCH ||
        vae_c_batch_norm2d_forward(
            &layer,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &input,
            &wrong_shape) != VAE_C_OP_SHAPE_MISMATCH ||
        vae_c_batch_norm2d_forward(
            &layer,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &wrong_channel,
            &wrong_channel_output) != VAE_C_OP_SHAPE_MISMATCH ||
        vae_c_batch_norm2d_forward(
            &layer,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &singleton_input,
            &singleton_output) != VAE_C_OP_SHAPE_MISMATCH ||
        vae_c_batch_norm2d_forward(
            &layer,
            VAE_C_BATCH_NORM_MODE_EVAL,
            &singleton_input,
            &singleton_output) != VAE_C_OP_OK ||
        vae_c_batch_norm2d_forward(
            &layer,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &overflow_input,
            &output) != VAE_C_OP_OVERFLOW ||
        !float_array_equal(output.data, output_snapshot, 4)) {

        goto cleanup;
    }
    passed = 1;

cleanup:
    vae_c_tensor_free(&singleton_output);
    vae_c_tensor_free(&singleton_input);
    vae_c_tensor_free(&wrong_channel_output);
    vae_c_tensor_free(&wrong_channel);
    vae_c_tensor_free(&wrong_shape);
    vae_c_tensor_free(&rank_two);
    vae_c_tensor_free(&output);
    vae_c_tensor_free(&input);
    vae_c_batch_norm_free(&layer);
    return passed;
}


int main(void)
{
    int failures = 0;

    failures += report_check(
        "batch_norm_lifecycle_and_layout",
        test_lifecycle_and_layout()
    );
    failures += report_check(
        "batch_norm1d_analytic_train_backward",
        test_batch_norm1d_analytic()
    );
    failures += report_check(
        "batch_norm2d_analytic_train_backward",
        test_batch_norm2d_analytic()
    );
    failures += report_check(
        "batch_norm_running_statistics_and_evaluation",
        test_running_statistics_and_evaluation()
    );
    failures += report_check(
        "batch_norm1d_gradient_check",
        run_gradient_check(VAE_C_BATCH_NORM_KIND_1D)
    );
    failures += report_check(
        "batch_norm2d_gradient_check",
        run_gradient_check(VAE_C_BATCH_NORM_KIND_2D)
    );
    failures += report_check(
        "batch_norm1d_errors_and_strong_guarantee",
        test_batch_norm1d_errors_and_strong_guarantee()
    );
    failures += report_check(
        "batch_norm2d_errors_and_strong_guarantee",
        test_batch_norm2d_errors_and_strong_guarantee()
    );

    if (failures != 0) {
        fprintf(stderr, "%d VAE C BATCHNORM TEST(S) FAILED\n", failures);
        return 1;
    }
    printf("ALL VAE C BATCHNORM TESTS PASSED\n");
    return 0;
}
