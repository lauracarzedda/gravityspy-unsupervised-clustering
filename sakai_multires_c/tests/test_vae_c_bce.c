#include "vae_c_objective.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>


static int report_check(const char *name, int passed)
{
    if (passed) {
        printf("[PASS] %s\n", name);
        return 0;
    }

    fprintf(stderr, "[FAIL] %s\n", name);
    return 1;
}


static int make_view_4d(
    VAECTensor *tensor,
    float *data,
    float *grad,
    size_t batch,
    size_t channels,
    size_t height,
    size_t width
)
{
    size_t dims[4];

    if (tensor == NULL || data == NULL) {
        return 0;
    }

    dims[0] = batch;
    dims[1] = channels;
    dims[2] = height;
    dims[3] = width;
    vae_c_tensor_init(tensor);

    if (vae_c_shape_make(4, dims, &tensor->shape) !=
        VAE_C_STORAGE_OK) {

        return 0;
    }

    tensor->data = data;
    tensor->grad = grad;
    tensor->owned_block = NULL;
    tensor->owned_bytes = 0;
    tensor->owns_memory = false;
    tensor->has_grad = grad != NULL;
    return 1;
}


static int arrays_bitwise_equal(
    const float *left,
    const float *right,
    size_t count
)
{
    return memcmp(left, right, count * sizeof(float)) == 0;
}


static int arrays_close_f32(
    const float *actual,
    const float *expected,
    size_t count,
    float tolerance
)
{
    size_t index;

    for (index = 0; index < count; index++) {
        if (!isfinite(actual[index]) ||
            fabsf(actual[index] - expected[index]) > tolerance) {

            return 0;
        }
    }

    return 1;
}


static int doubles_close(
    double actual,
    double expected,
    double relative_tolerance
)
{
    double scale = fmax(fmax(fabs(actual), fabs(expected)), 1.0);

    return
        isfinite(actual) && isfinite(expected) &&
        fabs(actual - expected) <= relative_tolerance * scale;
}


static double reference_bce_loss(double logit, double target)
{
    return
        fmax(logit, 0.0) -
        logit * target +
        log1p(exp(-fabs(logit)));
}


static double reference_sigmoid(double logit)
{
    if (logit >= 0.0) {
        double exponential = exp(-logit);

        return 1.0 / (1.0 + exponential);
    }

    {
        double exponential = exp(logit);

        return exponential / (1.0 + exponential);
    }
}


static int stats_all_zero(const VAECBCEWithLogitsStats *stats)
{
    const unsigned char *bytes = (const unsigned char *)stats;
    size_t index;

    for (index = 0; index < sizeof(*stats); index++) {
        if (bytes[index] != 0) {
            return 0;
        }
    }

    return 1;
}


static void set_stats_sentinel(VAECBCEWithLogitsStats *stats)
{
    stats->batch_size = 17;
    stats->values_per_sample = 19;
    stats->total_values = 23;
    stats->total_sum = 29.5;
    stats->mean_per_sample = 31.5;
    stats->mean_per_value = 37.5;
}


static int test_stats_init_and_known_values(void)
{
    float logits_data[3] = { 0.0f, 0.0f, 0.0f };
    float logits_grad[3] = { 0.0f, 0.0f, 0.0f };
    float targets_data[3] = { 0.0f, 1.0f, 0.25f };
    float targets_grad[3] = { 4.0f, 5.0f, 6.0f };
    float targets_grad_before[3];
    const float expected_gradient[3] = { 0.5f, -0.5f, 0.25f };
    VAECTensor logits;
    VAECTensor targets;
    VAECBCEWithLogitsStats stats;
    double log_two = log(2.0);

    memset(&stats, 0xA5, sizeof(stats));
    vae_c_bce_with_logits_stats_init(&stats);
    if (!stats_all_zero(&stats) ||
        !make_view_4d(&logits, logits_data, logits_grad, 1, 1, 1, 3) ||
        !make_view_4d(&targets, targets_data, targets_grad, 1, 1, 1, 3)) {

        return 0;
    }

    memcpy(
        targets_grad_before,
        targets_grad,
        sizeof(targets_grad)
    );

    return
        vae_c_bce_with_logits_forward(
            &logits,
            &targets,
            &stats) == VAE_C_OP_OK &&
        stats.batch_size == 1 &&
        stats.values_per_sample == 3 &&
        stats.total_values == 3 &&
        doubles_close(stats.total_sum, 3.0 * log_two, 1.0e-14) &&
        doubles_close(
            stats.mean_per_sample,
            3.0 * log_two,
            1.0e-14) &&
        doubles_close(stats.mean_per_value, log_two, 1.0e-14) &&
        vae_c_bce_with_logits_backward(
            &logits,
            &targets,
            1.0f) == VAE_C_OP_OK &&
        arrays_bitwise_equal(logits_grad, expected_gradient, 3) &&
        arrays_bitwise_equal(
            targets_grad,
            targets_grad_before,
            3);
}


static int test_analytic_forward_backward(void)
{
    enum { count = 12 };
    float logits_data[count] = {
        0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.5f,
        -0.5f, 3.0f, -3.0f, 0.25f, -0.25f, 1.5f
    };
    float targets_data[count] = {
        0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.25f,
        0.75f, 1.0f, 0.0f, 0.5f, 0.2f, 0.8f
    };
    float logits_grad[count];
    float targets_grad[count];
    float expected_gradient[count];
    float logits_before[count];
    float targets_before[count];
    float targets_grad_before[count];
    VAECTensor logits;
    VAECTensor targets;
    VAECBCEWithLogitsStats stats;
    double expected_total;
    size_t index;

    expected_total = 0.0;
    for (index = 0; index < count; index++) {
        logits_grad[index] = 0.05f * (float)(index + 1);
        targets_grad[index] = -0.1f * (float)(index + 1);
        expected_total += reference_bce_loss(
            (double)logits_data[index],
            (double)targets_data[index]
        );
        expected_gradient[index] = logits_grad[index];
    }

    memcpy(logits_before, logits_data, sizeof(logits_data));
    memcpy(targets_before, targets_data, sizeof(targets_data));
    memcpy(
        targets_grad_before,
        targets_grad,
        sizeof(targets_grad)
    );

    if (!make_view_4d(&logits, logits_data, logits_grad, 2, 1, 2, 3) ||
        !make_view_4d(&targets, targets_data, targets_grad, 2, 1, 2, 3) ||
        vae_c_bce_with_logits_forward(
            &logits,
            &targets,
            &stats) != VAE_C_OP_OK ||
        stats.batch_size != 2 ||
        stats.values_per_sample != 6 ||
        stats.total_values != count ||
        !doubles_close(stats.total_sum, expected_total, 1.0e-14) ||
        !doubles_close(
            stats.mean_per_sample,
            expected_total / 2.0,
            1.0e-14) ||
        !doubles_close(
            stats.mean_per_value,
            expected_total / (double)count,
            1.0e-14) ||
        !arrays_bitwise_equal(logits_data, logits_before, count) ||
        !arrays_bitwise_equal(targets_data, targets_before, count) ||
        !arrays_bitwise_equal(
            targets_grad,
            targets_grad_before,
            count)) {

        return 0;
    }

    for (index = 0; index < count; index++) {
        expected_gradient[index] += (float)(
            (reference_sigmoid((double)logits_data[index]) -
             (double)targets_data[index]) /
            2.0
        );
    }

    if (vae_c_bce_with_logits_backward(
            &logits,
            &targets,
            1.0f) != VAE_C_OP_OK ||
        !arrays_close_f32(
            logits_grad,
            expected_gradient,
            count,
            2.0e-6f)) {

        return 0;
    }

    for (index = 0; index < count; index++) {
        expected_gradient[index] += (float)(
            -0.35 *
            (reference_sigmoid((double)logits_data[index]) -
             (double)targets_data[index]) /
            2.0
        );
    }

    if (vae_c_bce_with_logits_backward(
            &logits,
            &targets,
            -0.35f) != VAE_C_OP_OK ||
        !arrays_close_f32(
            logits_grad,
            expected_gradient,
            count,
            3.0e-6f)) {

        return 0;
    }

    memcpy(
        expected_gradient,
        logits_grad,
        sizeof(expected_gradient)
    );

    return
        vae_c_bce_with_logits_backward(
            &logits,
            &targets,
            0.0f) == VAE_C_OP_OK &&
        arrays_bitwise_equal(logits_grad, expected_gradient, count) &&
        arrays_bitwise_equal(logits_data, logits_before, count) &&
        arrays_bitwise_equal(targets_data, targets_before, count) &&
        arrays_bitwise_equal(
            targets_grad,
            targets_grad_before,
            count);
}


static int test_extreme_logits(void)
{
    enum { count = 8 };
    float logits_data[count] = {
        FLT_MAX, -FLT_MAX, 100.0f, -100.0f,
        FLT_MAX, -FLT_MAX, 100.0f, -100.0f
    };
    float targets_data[count] = {
        1.0f, 0.0f, 0.0f, 1.0f,
        0.0f, 1.0f, 1.0f, 0.0f
    };
    float logits_grad[count];
    float expected_gradient[count];
    float logits_before[count];
    float targets_before[count];
    VAECTensor logits;
    VAECTensor targets;
    VAECBCEWithLogitsStats stats;
    double expected_total;
    size_t index;

    expected_total = 0.0;
    for (index = 0; index < count; index++) {
        logits_grad[index] = 0.0f;
        expected_total += reference_bce_loss(
            (double)logits_data[index],
            (double)targets_data[index]
        );
        expected_gradient[index] = (float)(
            (reference_sigmoid((double)logits_data[index]) -
             (double)targets_data[index]) /
            2.0
        );
    }

    memcpy(logits_before, logits_data, sizeof(logits_data));
    memcpy(targets_before, targets_data, sizeof(targets_data));

    return
        make_view_4d(&logits, logits_data, logits_grad, 2, 1, 1, 4) &&
        make_view_4d(&targets, targets_data, NULL, 2, 1, 1, 4) &&
        vae_c_bce_with_logits_forward(
            &logits,
            &targets,
            &stats) == VAE_C_OP_OK &&
        isfinite(stats.total_sum) &&
        isfinite(stats.mean_per_sample) &&
        isfinite(stats.mean_per_value) &&
        doubles_close(stats.total_sum, expected_total, 1.0e-14) &&
        vae_c_bce_with_logits_backward(
            &logits,
            &targets,
            1.0f) == VAE_C_OP_OK &&
        arrays_close_f32(
            logits_grad,
            expected_gradient,
            count,
            1.0e-6f) &&
        logits_grad[0] == 0.0f &&
        logits_grad[1] == 0.0f &&
        logits_grad[4] == 0.5f &&
        logits_grad[5] == -0.5f &&
        arrays_bitwise_equal(logits_data, logits_before, count) &&
        arrays_bitwise_equal(targets_data, targets_before, count);
}


static double gradient_objective(
    VAECTensor *logits,
    VAECTensor *targets,
    float upstream_gradient,
    int *ok
)
{
    VAECBCEWithLogitsStats stats;

    if (vae_c_bce_with_logits_forward(
            logits,
            targets,
            &stats) != VAE_C_OP_OK) {

        *ok = 0;
        return 0.0;
    }

    return (double)upstream_gradient * stats.mean_per_sample;
}


static int test_gradient_check(void)
{
    enum { count = 8 };
    const float numeric_epsilon = 1.0e-3f;
    const float upstream_gradient = 0.7f;
    float logits_data[count] = {
        -1.2f, -0.7f, -0.2f, 0.3f,
        0.6f, 1.1f, 1.7f, -1.8f
    };
    float targets_data[count] = {
        0.0f, 0.25f, 1.0f, 0.4f,
        0.75f, 0.0f, 1.0f, 0.6f
    };
    float logits_grad[count];
    float analytic_gradient[count];
    VAECTensor logits;
    VAECTensor targets;
    double maximum_absolute_error;
    double maximum_relative_error;
    size_t worst_index;
    size_t index;
    int ok;

    memset(logits_grad, 0, sizeof(logits_grad));
    if (!make_view_4d(&logits, logits_data, logits_grad, 2, 1, 2, 2) ||
        !make_view_4d(&targets, targets_data, NULL, 2, 1, 2, 2) ||
        vae_c_bce_with_logits_backward(
            &logits,
            &targets,
            upstream_gradient) != VAE_C_OP_OK) {

        return 0;
    }

    memcpy(
        analytic_gradient,
        logits_grad,
        sizeof(analytic_gradient)
    );
    maximum_absolute_error = 0.0;
    maximum_relative_error = 0.0;
    worst_index = 0;
    ok = 1;

    for (index = 0; index < count; index++) {
        float original = logits_data[index];
        double objective_plus;
        double objective_minus;
        double numeric_gradient;
        double absolute_error;
        double scale;
        double relative_error;

        logits_data[index] = original + numeric_epsilon;
        objective_plus = gradient_objective(
            &logits,
            &targets,
            upstream_gradient,
            &ok
        );
        logits_data[index] = original - numeric_epsilon;
        objective_minus = gradient_objective(
            &logits,
            &targets,
            upstream_gradient,
            &ok
        );
        logits_data[index] = original;

        if (!ok) {
            return 0;
        }

        numeric_gradient =
            (objective_plus - objective_minus) /
            (2.0 * (double)numeric_epsilon);
        absolute_error = fabs(
            numeric_gradient - (double)analytic_gradient[index]
        );
        scale = fmax(
            fmax(
                fabs(numeric_gradient),
                fabs((double)analytic_gradient[index])),
            1.0e-12
        );
        relative_error = absolute_error / scale;

        if (absolute_error > maximum_absolute_error) {
            maximum_absolute_error = absolute_error;
            worst_index = index;
        }
        if (relative_error > maximum_relative_error) {
            maximum_relative_error = relative_error;
        }
    }

    printf(
        "BCE_GRAD_MAX_ABS=%.9g MAX_REL=%.9g WORST_INDEX=%zu\n",
        maximum_absolute_error,
        maximum_relative_error,
        worst_index
    );

    return
        maximum_absolute_error <= 3.0e-4 &&
        maximum_relative_error <= 3.0e-3;
}


static int test_reduction_contract(void)
{
    float small_logits_data[2] = { 0.0f, 0.0f };
    float small_logits_grad[2] = { 0.0f, 0.0f };
    float small_targets_data[2] = { 0.0f, 0.0f };
    float large_logits_data[8] = {
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    };
    float large_logits_grad[8] = {
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    };
    float large_targets_data[8] = {
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    };
    VAECTensor small_logits;
    VAECTensor small_targets;
    VAECTensor large_logits;
    VAECTensor large_targets;
    VAECBCEWithLogitsStats small_stats;
    VAECBCEWithLogitsStats large_stats;
    size_t index;

    if (!make_view_4d(
            &small_logits,
            small_logits_data,
            small_logits_grad,
            2, 1, 1, 1) ||
        !make_view_4d(
            &small_targets,
            small_targets_data,
            NULL,
            2, 1, 1, 1) ||
        !make_view_4d(
            &large_logits,
            large_logits_data,
            large_logits_grad,
            2, 1, 2, 2) ||
        !make_view_4d(
            &large_targets,
            large_targets_data,
            NULL,
            2, 1, 2, 2) ||
        vae_c_bce_with_logits_forward(
            &small_logits,
            &small_targets,
            &small_stats) != VAE_C_OP_OK ||
        vae_c_bce_with_logits_forward(
            &large_logits,
            &large_targets,
            &large_stats) != VAE_C_OP_OK ||
        vae_c_bce_with_logits_backward(
            &small_logits,
            &small_targets,
            1.0f) != VAE_C_OP_OK ||
        vae_c_bce_with_logits_backward(
            &large_logits,
            &large_targets,
            1.0f) != VAE_C_OP_OK) {

        return 0;
    }

    if (!doubles_close(
            small_stats.mean_per_value,
            large_stats.mean_per_value,
            1.0e-14) ||
        !doubles_close(
            large_stats.mean_per_sample,
            4.0 * small_stats.mean_per_sample,
            1.0e-14) ||
        !doubles_close(
            small_stats.mean_per_sample,
            small_stats.total_sum / 2.0,
            1.0e-14) ||
        !doubles_close(
            large_stats.mean_per_value,
            large_stats.total_sum / 8.0,
            1.0e-14)) {

        return 0;
    }

    for (index = 0; index < 2; index++) {
        if (small_logits_grad[index] != 0.25f) {
            return 0;
        }
    }
    for (index = 0; index < 8; index++) {
        if (large_logits_grad[index] != 0.25f) {
            return 0;
        }
    }

    return 1;
}


static int forward_rejected_preserving_stats(
    const VAECTensor *logits,
    const VAECTensor *targets,
    VAECBCEWithLogitsStats *stats,
    const VAECBCEWithLogitsStats *before
)
{
    return
        vae_c_bce_with_logits_forward(logits, targets, stats) !=
            VAE_C_OP_OK &&
        memcmp(stats, before, sizeof(*stats)) == 0;
}


static int test_forward_negative_contracts(void)
{
    float logits_data[2] = { 0.2f, -0.3f };
    float logits_grad[2] = { 1.0f, 2.0f };
    float targets_data[2] = { 0.25f, 0.75f };
    float targets_grad[2] = { 3.0f, 4.0f };
    float other_data[1] = { 0.5f };
    float logits_data_before[2];
    float logits_grad_before[2];
    float targets_data_before[2];
    float targets_grad_before[2];
    VAECTensor logits;
    VAECTensor targets;
    VAECTensor other;
    VAECTensor corrupted;
    VAECBCEWithLogitsStats stats;
    VAECBCEWithLogitsStats before;

    if (!make_view_4d(&logits, logits_data, logits_grad, 1, 1, 1, 2) ||
        !make_view_4d(&targets, targets_data, targets_grad, 1, 1, 1, 2) ||
        !make_view_4d(&other, other_data, NULL, 1, 1, 1, 1)) {

        return 0;
    }

    memcpy(logits_data_before, logits_data, sizeof(logits_data));
    memcpy(logits_grad_before, logits_grad, sizeof(logits_grad));
    memcpy(targets_data_before, targets_data, sizeof(targets_data));
    memcpy(targets_grad_before, targets_grad, sizeof(targets_grad));
    set_stats_sentinel(&stats);
    before = stats;

    if (!forward_rejected_preserving_stats(
            NULL, &targets, &stats, &before) ||
        !forward_rejected_preserving_stats(
            &logits, NULL, &stats, &before) ||
        vae_c_bce_with_logits_forward(
            &logits, &targets, NULL) != VAE_C_OP_INVALID_ARGUMENT ||
        !forward_rejected_preserving_stats(
            &logits, &logits, &stats, &before) ||
        !forward_rejected_preserving_stats(
            &logits, &other, &stats, &before)) {

        return 0;
    }

    corrupted = logits;
    corrupted.shape.rank = 3;
    corrupted.shape.dims[0] = 1;
    corrupted.shape.dims[1] = 1;
    corrupted.shape.dims[2] = 2;
    corrupted.shape.dims[3] = 0;
    corrupted.shape.strides[0] = 2;
    corrupted.shape.strides[1] = 2;
    corrupted.shape.strides[2] = 1;
    corrupted.shape.strides[3] = 0;
    if (!forward_rejected_preserving_stats(
            &corrupted, &targets, &stats, &before)) {
        return 0;
    }

    corrupted = logits;
    corrupted.shape.strides[3] = 2;
    if (!forward_rejected_preserving_stats(
            &corrupted, &targets, &stats, &before)) {
        return 0;
    }

    corrupted = logits;
    corrupted.shape.dims[0] = 1;
    corrupted.shape.dims[1] = SIZE_MAX;
    corrupted.shape.dims[2] = 2;
    corrupted.shape.dims[3] = 1;
    if (!forward_rejected_preserving_stats(
            &corrupted, &targets, &stats, &before)) {
        return 0;
    }

    corrupted = logits;
    corrupted.shape.dims[0] = SIZE_MAX;
    corrupted.shape.dims[1] = 2;
    corrupted.shape.dims[2] = 1;
    corrupted.shape.dims[3] = 1;
    if (!forward_rejected_preserving_stats(
            &corrupted, &targets, &stats, &before)) {
        return 0;
    }

    if (vae_c_bce_with_logits_forward(
            &logits,
            &targets,
            (VAECBCEWithLogitsStats *)(void *)logits.data) !=
        VAE_C_OP_INVALID_ARGUMENT) {

        return 0;
    }

    logits_data[0] = NAN;
    if (!forward_rejected_preserving_stats(
            &logits, &targets, &stats, &before)) {
        return 0;
    }
    logits_data[0] = INFINITY;
    if (!forward_rejected_preserving_stats(
            &logits, &targets, &stats, &before)) {
        return 0;
    }
    logits_data[0] = -INFINITY;
    set_stats_sentinel(&stats);
    before = stats;
    memcpy(logits_data_before, logits_data, sizeof(logits_data));
    memcpy(logits_grad_before, logits_grad, sizeof(logits_grad));
    memcpy(targets_data_before, targets_data, sizeof(targets_data));
    memcpy(
        targets_grad_before,
        targets_grad,
        sizeof(targets_grad)
    );
    if (vae_c_bce_with_logits_forward(
            &logits,
            &targets,
            &stats) != VAE_C_OP_INVALID_STATE ||
        memcmp(&stats, &before, sizeof(stats)) != 0 ||
        !arrays_bitwise_equal(
            logits_data,
            logits_data_before,
            2) ||
        !arrays_bitwise_equal(
            logits_grad,
            logits_grad_before,
            2) ||
        !arrays_bitwise_equal(
            targets_data,
            targets_data_before,
            2) ||
        !arrays_bitwise_equal(
            targets_grad,
            targets_grad_before,
            2)) {

        return 0;
    }
    logits_data[0] = 0.2f;
    targets_data[0] = NAN;
    if (!forward_rejected_preserving_stats(
            &logits, &targets, &stats, &before)) {
        return 0;
    }
    targets_data[0] = INFINITY;
    if (!forward_rejected_preserving_stats(
            &logits, &targets, &stats, &before)) {
        return 0;
    }
    targets_data[0] = -INFINITY;
    set_stats_sentinel(&stats);
    before = stats;
    memcpy(logits_data_before, logits_data, sizeof(logits_data));
    memcpy(logits_grad_before, logits_grad, sizeof(logits_grad));
    memcpy(targets_data_before, targets_data, sizeof(targets_data));
    memcpy(
        targets_grad_before,
        targets_grad,
        sizeof(targets_grad)
    );
    if (vae_c_bce_with_logits_forward(
            &logits,
            &targets,
            &stats) != VAE_C_OP_INVALID_STATE ||
        memcmp(&stats, &before, sizeof(stats)) != 0 ||
        !arrays_bitwise_equal(
            logits_data,
            logits_data_before,
            2) ||
        !arrays_bitwise_equal(
            logits_grad,
            logits_grad_before,
            2) ||
        !arrays_bitwise_equal(
            targets_data,
            targets_data_before,
            2) ||
        !arrays_bitwise_equal(
            targets_grad,
            targets_grad_before,
            2)) {

        return 0;
    }
    targets_data[0] = 0.25f;
    targets_data[0] = -0.01f;
    if (!forward_rejected_preserving_stats(
            &logits, &targets, &stats, &before)) {
        return 0;
    }
    targets_data[0] = 1.01f;
    if (!forward_rejected_preserving_stats(
            &logits, &targets, &stats, &before)) {
        return 0;
    }
    targets_data[0] = 0.25f;

    return
        arrays_bitwise_equal(
            logits_grad,
            logits_grad_before,
            2) &&
        arrays_bitwise_equal(
            targets_grad,
            targets_grad_before,
            2);
}


static int backward_rejected_preserving_gradient(
    VAECTensor *logits,
    const VAECTensor *targets,
    float upstream,
    const float *gradient_before,
    size_t count
)
{
    return
        vae_c_bce_with_logits_backward(
            logits,
            targets,
            upstream) != VAE_C_OP_OK &&
        arrays_bitwise_equal(logits->grad, gradient_before, count);
}


static int test_backward_negative_contracts(void)
{
    float logits_data[2] = { 0.0f, -0.5f };
    float logits_grad[2] = { 0.2f, -0.3f };
    float targets_data[2] = { 0.0f, 1.0f };
    float targets_grad[2] = { 0.4f, -0.6f };
    float gradient_before[2];
    float logits_before[2];
    float targets_before[2];
    float targets_grad_before[2];
    VAECTensor logits;
    VAECTensor targets;
    VAECTensor other;

    if (!make_view_4d(&logits, logits_data, logits_grad, 1, 1, 1, 2) ||
        !make_view_4d(&targets, targets_data, targets_grad, 1, 1, 1, 2)) {

        return 0;
    }

    memcpy(gradient_before, logits_grad, sizeof(logits_grad));
    memcpy(logits_before, logits_data, sizeof(logits_data));
    memcpy(targets_before, targets_data, sizeof(targets_data));
    memcpy(
        targets_grad_before,
        targets_grad,
        sizeof(targets_grad)
    );

    if (vae_c_bce_with_logits_backward(
            NULL, &targets, 1.0f) == VAE_C_OP_OK ||
        vae_c_bce_with_logits_backward(
            &logits, NULL, 1.0f) == VAE_C_OP_OK ||
        !backward_rejected_preserving_gradient(
            &logits, &logits, 1.0f, gradient_before, 2) ||
        !backward_rejected_preserving_gradient(
            &logits, &targets, NAN, gradient_before, 2) ||
        !backward_rejected_preserving_gradient(
            &logits, &targets, INFINITY, gradient_before, 2)) {

        return 0;
    }

    memcpy(gradient_before, logits_grad, sizeof(logits_grad));
    memcpy(logits_before, logits_data, sizeof(logits_data));
    memcpy(targets_before, targets_data, sizeof(targets_data));
    memcpy(
        targets_grad_before,
        targets_grad,
        sizeof(targets_grad)
    );
    if (vae_c_bce_with_logits_backward(
            &logits,
            &targets,
            -INFINITY) != VAE_C_OP_INVALID_ARGUMENT ||
        !arrays_bitwise_equal(logits_grad, gradient_before, 2) ||
        !arrays_bitwise_equal(logits_data, logits_before, 2) ||
        !arrays_bitwise_equal(targets_data, targets_before, 2) ||
        !arrays_bitwise_equal(
            targets_grad,
            targets_grad_before,
            2)) {

        return 0;
    }

    other = logits;
    other.grad = NULL;
    other.has_grad = false;
    if (vae_c_bce_with_logits_backward(
            &other, &targets, 1.0f) !=
        VAE_C_OP_GRADIENT_NOT_AVAILABLE) {

        return 0;
    }

    other = targets;
    other.data = logits.grad;
    if (!backward_rejected_preserving_gradient(
            &logits, &other, 1.0f, gradient_before, 2)) {
        return 0;
    }

    logits_data[0] = NAN;
    if (!backward_rejected_preserving_gradient(
            &logits, &targets, 1.0f, gradient_before, 2)) {
        return 0;
    }
    logits_data[0] = 0.0f;
    targets_data[0] = -0.1f;
    if (!backward_rejected_preserving_gradient(
            &logits, &targets, 1.0f, gradient_before, 2)) {
        return 0;
    }
    targets_data[0] = 0.0f;

    logits_grad[0] = NAN;
    memcpy(gradient_before, logits_grad, sizeof(logits_grad));
    if (!backward_rejected_preserving_gradient(
            &logits, &targets, 1.0f, gradient_before, 2)) {
        return 0;
    }
    logits_grad[0] = INFINITY;
    memcpy(gradient_before, logits_grad, sizeof(logits_grad));
    if (!backward_rejected_preserving_gradient(
            &logits, &targets, 1.0f, gradient_before, 2)) {
        return 0;
    }
    logits_grad[0] = FLT_MAX;
    logits_grad[1] = -0.3f;
    memcpy(gradient_before, logits_grad, sizeof(logits_grad));

    return
        backward_rejected_preserving_gradient(
            &logits,
            &targets,
            FLT_MAX,
            gradient_before,
            2) &&
        arrays_bitwise_equal(targets_data, targets_before, 2) &&
        arrays_bitwise_equal(
            targets_grad,
            targets_grad_before,
            2);
}


int main(void)
{
    int failures = 0;

    failures += report_check(
        "bce_stats_init_and_known_values",
        test_stats_init_and_known_values()
    );
    failures += report_check(
        "bce_analytic_forward_backward",
        test_analytic_forward_backward()
    );
    failures += report_check(
        "bce_extreme_logits",
        test_extreme_logits()
    );
    failures += report_check(
        "bce_gradient_check",
        test_gradient_check()
    );
    failures += report_check(
        "bce_reduction_contract",
        test_reduction_contract()
    );
    failures += report_check(
        "bce_forward_negative_contracts",
        test_forward_negative_contracts()
    );
    failures += report_check(
        "bce_backward_negative_contracts",
        test_backward_negative_contracts()
    );

    if (failures != 0) {
        fprintf(stderr, "%d VAE C BCE TEST(S) FAILED\n", failures);
        return 1;
    }

    printf("ALL VAE C BCE TESTS PASSED\n");
    return 0;
}
