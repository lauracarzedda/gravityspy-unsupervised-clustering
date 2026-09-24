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


static int make_view_2d(
    VAECTensor *tensor,
    float *data,
    float *grad,
    size_t batch_size,
    size_t latent_dim
)
{
    size_t dims[2];

    if (tensor == NULL || data == NULL) {
        return 0;
    }

    dims[0] = batch_size;
    dims[1] = latent_dim;
    vae_c_tensor_init(tensor);

    if (vae_c_shape_make(2, dims, &tensor->shape) !=
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


static int make_view_1d(
    VAECTensor *tensor,
    float *data,
    float *grad,
    size_t count
)
{
    size_t dims[1];

    if (tensor == NULL || data == NULL) {
        return 0;
    }

    dims[0] = count;
    vae_c_tensor_init(tensor);

    if (vae_c_shape_make(1, dims, &tensor->shape) !=
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


static double reference_kl_element(double mu, double logvar)
{
    return 0.5 * (mu * mu + expm1(logvar) - logvar);
}


static int stats_all_zero(const VAECKLGaussianStats *stats)
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


static void set_stats_sentinel(VAECKLGaussianStats *stats)
{
    stats->batch_size = 17;
    stats->latent_dim = 19;
    stats->total_values = 23;
    stats->total_sum = 29.5;
    stats->mean_per_sample = 31.5;
    stats->mean_per_dimension = 37.5;
}


static int test_stats_init_and_known_values(void)
{
    float mu_data[4] = { 0.0f, 1.0f, 0.0f, 0.0f };
    float logvar_data[4] = {
        0.0f,
        0.0f,
        logf(4.0f),
        logf(0.25f)
    };
    float mu_grad[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float logvar_grad[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float expected_mu_grad[4];
    float expected_logvar_grad[4];
    VAECTensor mu;
    VAECTensor logvar;
    VAECKLGaussianStats stats;
    double expected_total = 0.0;
    size_t index;

    memset(&stats, 0xA5, sizeof(stats));
    vae_c_kl_gaussian_stats_init(&stats);
    if (!stats_all_zero(&stats) ||
        !make_view_2d(&mu, mu_data, mu_grad, 1, 4) ||
        !make_view_2d(&logvar, logvar_data, logvar_grad, 1, 4)) {

        return 0;
    }

    for (index = 0; index < 4; index++) {
        expected_total += reference_kl_element(
            (double)mu_data[index],
            (double)logvar_data[index]
        );
        expected_mu_grad[index] = mu_data[index];
        expected_logvar_grad[index] =
            0.5f * expm1f(logvar_data[index]);
    }

    return
        vae_c_kl_gaussian_forward(&mu, &logvar, &stats) ==
            VAE_C_OP_OK &&
        stats.batch_size == 1 &&
        stats.latent_dim == 4 &&
        stats.total_values == 4 &&
        doubles_close(stats.total_sum, expected_total, 1.0e-14) &&
        doubles_close(
            stats.mean_per_sample,
            expected_total,
            1.0e-14) &&
        doubles_close(
            stats.mean_per_dimension,
            expected_total / 4.0,
            1.0e-14) &&
        doubles_close(
            reference_kl_element(0.0, 0.0),
            0.0,
            0.0) &&
        doubles_close(
            reference_kl_element(1.0, 0.0),
            0.5,
            0.0) &&
        vae_c_kl_gaussian_backward(
            &mu,
            &logvar,
            1.0f) == VAE_C_OP_OK &&
        arrays_close_f32(mu_grad, expected_mu_grad, 4, 1.0e-7f) &&
        arrays_close_f32(
            logvar_grad,
            expected_logvar_grad,
            4,
            1.0e-6f);
}


static int test_analytic_forward_backward(void)
{
    enum { count = 6 };
    float mu_data[count] = {
        0.0f, 0.5f, -1.0f,
        1.5f, -0.25f, 0.75f
    };
    float logvar_data[count] = {
        0.0f, -0.6931471805599453f, 0.6931471805599453f,
        -0.5f, 0.25f, -1.0f
    };
    float mu_data_before[count];
    float logvar_data_before[count];
    float mu_grad[count] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    float logvar_grad[count] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };
    float expected_mu_grad[count];
    float expected_logvar_grad[count];
    float gradient_before_zero[count];
    VAECTensor mu;
    VAECTensor logvar;
    VAECKLGaussianStats stats;
    double expected_total = 0.0;
    float accumulated_upstream;
    size_t index;

    if (!make_view_2d(&mu, mu_data, mu_grad, 2, 3) ||
        !make_view_2d(&logvar, logvar_data, logvar_grad, 2, 3)) {

        return 0;
    }

    memcpy(mu_data_before, mu_data, sizeof(mu_data));
    memcpy(logvar_data_before, logvar_data, sizeof(logvar_data));
    for (index = 0; index < count; index++) {
        expected_total += reference_kl_element(
            (double)mu_data[index],
            (double)logvar_data[index]
        );
        expected_mu_grad[index] = mu_data[index] / 2.0f;
        expected_logvar_grad[index] =
            0.5f * expm1f(logvar_data[index]) / 2.0f;
    }

    if (vae_c_kl_gaussian_forward(&mu, &logvar, &stats) !=
            VAE_C_OP_OK ||
        !doubles_close(stats.total_sum, expected_total, 1.0e-14) ||
        !doubles_close(
            stats.mean_per_sample,
            expected_total / 2.0,
            1.0e-14) ||
        !doubles_close(
            stats.mean_per_dimension,
            expected_total / 6.0,
            1.0e-14) ||
        vae_c_kl_gaussian_backward(
            &mu,
            &logvar,
            1.0f) != VAE_C_OP_OK ||
        !arrays_close_f32(mu_grad, expected_mu_grad, count, 1.0e-7f) ||
        !arrays_close_f32(
            logvar_grad,
            expected_logvar_grad,
            count,
            1.0e-7f)) {

        return 0;
    }

    accumulated_upstream = 1.0f;
    if (vae_c_kl_gaussian_backward(
            &mu,
            &logvar,
            -0.25f) != VAE_C_OP_OK) {

        return 0;
    }
    accumulated_upstream -= 0.25f;

    memcpy(gradient_before_zero, mu_grad, sizeof(mu_grad));
    if (vae_c_kl_gaussian_backward(
            &mu,
            &logvar,
            0.0f) != VAE_C_OP_OK ||
        !arrays_bitwise_equal(
            mu_grad,
            gradient_before_zero,
            count)) {

        return 0;
    }

    if (vae_c_kl_gaussian_backward(
            &mu,
            &logvar,
            0.75f) != VAE_C_OP_OK) {

        return 0;
    }
    accumulated_upstream += 0.75f;

    for (index = 0; index < count; index++) {
        expected_mu_grad[index] =
            accumulated_upstream * mu_data[index] / 2.0f;
        expected_logvar_grad[index] =
            accumulated_upstream *
            0.5f * expm1f(logvar_data[index]) / 2.0f;
    }

    return
        arrays_close_f32(mu_grad, expected_mu_grad, count, 2.0e-7f) &&
        arrays_close_f32(
            logvar_grad,
            expected_logvar_grad,
            count,
            2.0e-7f) &&
        arrays_bitwise_equal(mu_data, mu_data_before, count) &&
        arrays_bitwise_equal(
            logvar_data,
            logvar_data_before,
            count);
}


static int test_near_zero_logvar(void)
{
    float values[4] = { 1.0e-7f, -1.0e-7f, 1.0e-5f, -1.0e-5f };
    size_t index;

    for (index = 0; index < 4; index++) {
        float mu_data[1] = { 0.0f };
        float logvar_data[1] = { values[index] };
        float mu_grad[1] = { 0.0f };
        float logvar_grad[1] = { 0.0f };
        VAECTensor mu;
        VAECTensor logvar;
        VAECKLGaussianStats stats;
        double logvar_f64 = (double)values[index];
        double expected =
            0.5 * (expm1(logvar_f64) - logvar_f64);
        float expected_gradient = 0.5f * expm1f(values[index]);

        if (!make_view_2d(&mu, mu_data, mu_grad, 1, 1) ||
            !make_view_2d(
                &logvar,
                logvar_data,
                logvar_grad,
                1,
                1) ||
            vae_c_kl_gaussian_forward(&mu, &logvar, &stats) !=
                VAE_C_OP_OK ||
            !(stats.total_sum > 0.0) ||
            !isfinite(stats.total_sum) ||
            fabs(stats.total_sum - expected) > 1.0e-24 ||
            vae_c_kl_gaussian_backward(
                &mu,
                &logvar,
                1.0f) != VAE_C_OP_OK ||
            !isfinite(logvar_grad[0]) ||
            logvar_grad[0] != expected_gradient) {

            return 0;
        }
    }

    return 1;
}


static int test_extreme_finite_values(void)
{
    float mu_negative_data[1] = { 0.0f };
    float logvar_negative_data[1] = { -FLT_MAX };
    float mu_negative_grad[1] = { 0.0f };
    float logvar_negative_grad[1] = { 0.0f };
    float mu_large_data[2] = { FLT_MAX, 0.0f };
    float logvar_zero_data[2] = { 0.0f, 0.0f };
    float mu_large_grad[2] = { 0.0f, 0.0f };
    float logvar_zero_grad[2] = { 0.0f, 0.0f };
    float mu_bad_data[1] = { 0.0f };
    float logvar_bad_data[1] = { 100.0f };
    float mu_bad_grad[1] = { 1.25f };
    float logvar_bad_grad[1] = { -2.5f };
    float mu_bad_before[1];
    float logvar_bad_before[1];
    VAECKLGaussianStats stats;
    VAECKLGaussianStats stats_before;
    VAECTensor mu_negative;
    VAECTensor logvar_negative;
    VAECTensor mu_large;
    VAECTensor logvar_zero;
    VAECTensor mu_bad;
    VAECTensor logvar_bad;
    double expected_negative;
    double expected_large;

    if (expm1f(-FLT_MAX) != -1.0f ||
        isfinite(expm1f(100.0f)) ||
        !make_view_2d(
            &mu_negative,
            mu_negative_data,
            mu_negative_grad,
            1,
            1) ||
        !make_view_2d(
            &logvar_negative,
            logvar_negative_data,
            logvar_negative_grad,
            1,
            1)) {

        return 0;
    }

    expected_negative = reference_kl_element(0.0, (double)-FLT_MAX);
    if (vae_c_kl_gaussian_forward(
            &mu_negative,
            &logvar_negative,
            &stats) != VAE_C_OP_OK ||
        !doubles_close(stats.total_sum, expected_negative, 1.0e-14) ||
        vae_c_kl_gaussian_backward(
            &mu_negative,
            &logvar_negative,
            1.0f) != VAE_C_OP_OK ||
        logvar_negative_grad[0] != -0.5f) {

        return 0;
    }

    if (!make_view_2d(
            &mu_large,
            mu_large_data,
            mu_large_grad,
            2,
            1) ||
        !make_view_2d(
            &logvar_zero,
            logvar_zero_data,
            logvar_zero_grad,
            2,
            1)) {

        return 0;
    }

    expected_large = 0.5 * (double)FLT_MAX * (double)FLT_MAX;
    if (vae_c_kl_gaussian_forward(
            &mu_large,
            &logvar_zero,
            &stats) != VAE_C_OP_OK ||
        !doubles_close(stats.total_sum, expected_large, 1.0e-14) ||
        vae_c_kl_gaussian_backward(
            &mu_large,
            &logvar_zero,
            1.0f) != VAE_C_OP_OK ||
        !isfinite(mu_large_grad[0]) ||
        mu_large_grad[0] != 0.5f * FLT_MAX) {

        return 0;
    }

    if (!make_view_2d(&mu_bad, mu_bad_data, mu_bad_grad, 1, 1) ||
        !make_view_2d(
            &logvar_bad,
            logvar_bad_data,
            logvar_bad_grad,
            1,
            1)) {

        return 0;
    }

    set_stats_sentinel(&stats);
    stats_before = stats;
    memcpy(mu_bad_before, mu_bad_grad, sizeof(mu_bad_grad));
    memcpy(logvar_bad_before, logvar_bad_grad, sizeof(logvar_bad_grad));

    return
        vae_c_kl_gaussian_forward(&mu_bad, &logvar_bad, &stats) ==
            VAE_C_OP_INVALID_STATE &&
        memcmp(&stats, &stats_before, sizeof(stats)) == 0 &&
        vae_c_kl_gaussian_backward(
            &mu_bad,
            &logvar_bad,
            1.0f) == VAE_C_OP_INVALID_STATE &&
        arrays_bitwise_equal(mu_bad_grad, mu_bad_before, 1) &&
        arrays_bitwise_equal(logvar_bad_grad, logvar_bad_before, 1);
}


static double gradient_objective(
    const VAECTensor *mu,
    const VAECTensor *logvar,
    float upstream_gradient,
    int *ok
)
{
    VAECKLGaussianStats stats;

    if (vae_c_kl_gaussian_forward(mu, logvar, &stats) !=
        VAE_C_OP_OK) {

        *ok = 0;
        return 0.0;
    }

    *ok = 1;
    return (double)upstream_gradient * stats.mean_per_sample;
}


static int test_gradient_check(void)
{
    enum { N = 2, L = 3, count = N * L };
    const float epsilon = 1.0e-3f;
    const float upstream = 0.7f;
    float mu_data[count] = { -1.1f, -0.4f, 0.2f, 0.7f, 1.3f, -0.8f };
    float logvar_data[count] = {
        -0.9f, -0.3f, 0.1f, 0.4f, 0.8f, -0.6f
    };
    float mu_grad[count] = { 1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f };
    float logvar_grad[count] = {
        -1.0f, 2.0f, -3.0f, 4.0f, -5.0f, 6.0f
    };
    float mu_grad_before_forward[count];
    float logvar_grad_before_forward[count];
    float analytic_mu[count];
    float analytic_logvar[count];
    VAECTensor mu;
    VAECTensor logvar;
    VAECKLGaussianStats stats;
    VAECOpStatus status;
    double maximum_mu_absolute = 0.0;
    double maximum_mu_relative = 0.0;
    double maximum_logvar_absolute = 0.0;
    double maximum_logvar_relative = 0.0;
    size_t worst_mu_index = 0;
    size_t worst_logvar_index = 0;
    size_t index;

    if (!make_view_2d(&mu, mu_data, mu_grad, N, L) ||
        !make_view_2d(&logvar, logvar_data, logvar_grad, N, L)) {

        return 0;
    }

    memcpy(mu_grad_before_forward, mu_grad, sizeof(mu_grad));
    memcpy(
        logvar_grad_before_forward,
        logvar_grad,
        sizeof(logvar_grad)
    );

    vae_c_kl_gaussian_stats_init(&stats);
    status = vae_c_kl_gaussian_forward(&mu, &logvar, &stats);
    if (status != VAE_C_OP_OK ||
        stats.batch_size != N ||
        stats.latent_dim != L ||
        stats.total_values != N * L ||
        !isfinite(stats.total_sum) ||
        !isfinite(stats.mean_per_sample) ||
        !isfinite(stats.mean_per_dimension) ||
        !arrays_bitwise_equal(
            mu_grad,
            mu_grad_before_forward,
            count) ||
        !arrays_bitwise_equal(
            logvar_grad,
            logvar_grad_before_forward,
            count)) {

        return 0;
    }

    if (vae_c_tensor_zero_grad(&mu) != VAE_C_STORAGE_OK ||
        vae_c_tensor_zero_grad(&logvar) != VAE_C_STORAGE_OK) {

        return 0;
    }

    for (index = 0; index < count; index++) {
        if (mu_grad[index] != 0.0f || logvar_grad[index] != 0.0f) {
            return 0;
        }
    }

    status = vae_c_kl_gaussian_backward(&mu, &logvar, upstream);
    if (status != VAE_C_OP_OK) {
        return 0;
    }

    memcpy(analytic_mu, mu_grad, sizeof(mu_grad));
    memcpy(analytic_logvar, logvar_grad, sizeof(logvar_grad));

    for (index = 0; index < count; index++) {
        float original = mu_data[index];
        double objective_plus;
        double objective_minus;
        double numeric_gradient;
        double absolute_error;
        double relative_error;
        double scale;
        int ok;

        mu_data[index] = original + epsilon;
        objective_plus = gradient_objective(
            &mu,
            &logvar,
            upstream,
            &ok
        );
        if (!ok) {
            mu_data[index] = original;
            return 0;
        }

        mu_data[index] = original - epsilon;
        objective_minus = gradient_objective(
            &mu,
            &logvar,
            upstream,
            &ok
        );
        mu_data[index] = original;
        if (!ok) {
            return 0;
        }

        numeric_gradient =
            (objective_plus - objective_minus) /
            (2.0 * (double)epsilon);
        absolute_error = fabs(
            numeric_gradient - (double)analytic_mu[index]
        );
        scale = fmax(
            fmax(fabs(numeric_gradient), fabs((double)analytic_mu[index])),
            1.0e-6
        );
        relative_error = absolute_error / scale;

        if (absolute_error > maximum_mu_absolute) {
            maximum_mu_absolute = absolute_error;
            worst_mu_index = index;
        }
        if (relative_error > maximum_mu_relative) {
            maximum_mu_relative = relative_error;
        }
    }

    for (index = 0; index < count; index++) {
        float original = logvar_data[index];
        double objective_plus;
        double objective_minus;
        double numeric_gradient;
        double absolute_error;
        double relative_error;
        double scale;
        int ok;

        logvar_data[index] = original + epsilon;
        objective_plus = gradient_objective(
            &mu,
            &logvar,
            upstream,
            &ok
        );
        if (!ok) {
            logvar_data[index] = original;
            return 0;
        }

        logvar_data[index] = original - epsilon;
        objective_minus = gradient_objective(
            &mu,
            &logvar,
            upstream,
            &ok
        );
        logvar_data[index] = original;
        if (!ok) {
            return 0;
        }

        numeric_gradient =
            (objective_plus - objective_minus) /
            (2.0 * (double)epsilon);
        absolute_error = fabs(
            numeric_gradient - (double)analytic_logvar[index]
        );
        scale = fmax(
            fmax(
                fabs(numeric_gradient),
                fabs((double)analytic_logvar[index])),
            1.0e-6
        );
        relative_error = absolute_error / scale;

        if (absolute_error > maximum_logvar_absolute) {
            maximum_logvar_absolute = absolute_error;
            worst_logvar_index = index;
        }
        if (relative_error > maximum_logvar_relative) {
            maximum_logvar_relative = relative_error;
        }
    }

    printf(
        "KL_MU_GRAD_MAX_ABS=%.9g MAX_REL=%.9g WORST_INDEX=%zu\n",
        maximum_mu_absolute,
        maximum_mu_relative,
        worst_mu_index
    );
    printf(
        "KL_LOGVAR_GRAD_MAX_ABS=%.9g MAX_REL=%.9g WORST_INDEX=%zu\n",
        maximum_logvar_absolute,
        maximum_logvar_relative,
        worst_logvar_index
    );

    return
        maximum_mu_absolute <= 3.0e-4 &&
        maximum_mu_relative <= 3.0e-3 &&
        maximum_logvar_absolute <= 3.0e-4 &&
        maximum_logvar_relative <= 3.0e-3;
}


static int test_reduction_contract(void)
{
    float small_mu_data[2] = { 1.0f, 1.0f };
    float small_logvar_data[2] = { 0.0f, 0.0f };
    float small_mu_grad[2] = { 0.0f, 0.0f };
    float small_logvar_grad[2] = { 0.0f, 0.0f };
    float large_mu_data[8] = {
        1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f
    };
    float large_logvar_data[8] = {
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    };
    float large_mu_grad[8] = {
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    };
    float large_logvar_grad[8] = {
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    };
    VAECTensor small_mu;
    VAECTensor small_logvar;
    VAECTensor large_mu;
    VAECTensor large_logvar;
    VAECKLGaussianStats small_stats;
    VAECKLGaussianStats large_stats;
    size_t index;

    if (!make_view_2d(
            &small_mu,
            small_mu_data,
            small_mu_grad,
            2,
            1) ||
        !make_view_2d(
            &small_logvar,
            small_logvar_data,
            small_logvar_grad,
            2,
            1) ||
        !make_view_2d(
            &large_mu,
            large_mu_data,
            large_mu_grad,
            2,
            4) ||
        !make_view_2d(
            &large_logvar,
            large_logvar_data,
            large_logvar_grad,
            2,
            4) ||
        vae_c_kl_gaussian_forward(
            &small_mu,
            &small_logvar,
            &small_stats) != VAE_C_OP_OK ||
        vae_c_kl_gaussian_forward(
            &large_mu,
            &large_logvar,
            &large_stats) != VAE_C_OP_OK ||
        !doubles_close(small_stats.total_sum, 1.0, 0.0) ||
        !doubles_close(small_stats.mean_per_sample, 0.5, 0.0) ||
        !doubles_close(small_stats.mean_per_dimension, 0.5, 0.0) ||
        !doubles_close(large_stats.total_sum, 4.0, 0.0) ||
        !doubles_close(large_stats.mean_per_sample, 2.0, 0.0) ||
        !doubles_close(large_stats.mean_per_dimension, 0.5, 0.0) ||
        vae_c_kl_gaussian_backward(
            &small_mu,
            &small_logvar,
            1.0f) != VAE_C_OP_OK ||
        vae_c_kl_gaussian_backward(
            &large_mu,
            &large_logvar,
            1.0f) != VAE_C_OP_OK) {

        return 0;
    }

    for (index = 0; index < 2; index++) {
        if (small_mu_grad[index] != 0.5f) {
            return 0;
        }
    }
    for (index = 0; index < 8; index++) {
        if (large_mu_grad[index] != 0.5f) {
            return 0;
        }
    }

    return 1;
}


static int forward_rejected_preserving_pair(
    const VAECTensor *mu,
    const VAECTensor *logvar,
    VAECKLGaussianStats *stats,
    VAECOpStatus expected_status
)
{
    float mu_data_before[2];
    float mu_grad_before[2];
    float logvar_data_before[2];
    float logvar_grad_before[2];
    VAECKLGaussianStats stats_before = *stats;

    memcpy(mu_data_before, mu->data, sizeof(mu_data_before));
    memcpy(mu_grad_before, mu->grad, sizeof(mu_grad_before));
    memcpy(logvar_data_before, logvar->data, sizeof(logvar_data_before));
    memcpy(
        logvar_grad_before,
        logvar->grad,
        sizeof(logvar_grad_before)
    );

    return
        vae_c_kl_gaussian_forward(mu, logvar, stats) ==
            expected_status &&
        memcmp(stats, &stats_before, sizeof(*stats)) == 0 &&
        arrays_bitwise_equal(mu->data, mu_data_before, 2) &&
        arrays_bitwise_equal(mu->grad, mu_grad_before, 2) &&
        arrays_bitwise_equal(logvar->data, logvar_data_before, 2) &&
        arrays_bitwise_equal(logvar->grad, logvar_grad_before, 2);
}


static int test_forward_negative_contracts(void)
{
    float mu_data[2] = { 0.2f, -0.3f };
    float mu_grad[2] = { 1.0f, 2.0f };
    float logvar_data[2] = { 0.25f, -0.75f };
    float logvar_grad[2] = { 3.0f, 4.0f };
    float other_data[2] = { 0.1f, 0.2f };
    float other_grad[2] = { 5.0f, 6.0f };
    float mu_data_before[2];
    VAECKLGaussianStats stats;
    VAECKLGaussianStats stats_before;
    VAECTensor mu;
    VAECTensor logvar;
    VAECTensor rank_one;
    VAECTensor different_shape;
    VAECTensor corrupted;
    VAECTensor aliased;

    if (!make_view_2d(&mu, mu_data, mu_grad, 1, 2) ||
        !make_view_2d(
            &logvar,
            logvar_data,
            logvar_grad,
            1,
            2) ||
        !make_view_1d(&rank_one, other_data, other_grad, 2) ||
        !make_view_2d(
            &different_shape,
            other_data,
            other_grad,
            2,
            1)) {

        return 0;
    }

    set_stats_sentinel(&stats);
    stats_before = stats;
    if (vae_c_kl_gaussian_forward(NULL, &logvar, &stats) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        memcmp(&stats, &stats_before, sizeof(stats)) != 0 ||
        vae_c_kl_gaussian_forward(&mu, NULL, &stats) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        memcmp(&stats, &stats_before, sizeof(stats)) != 0 ||
        vae_c_kl_gaussian_forward(&mu, &logvar, NULL) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        vae_c_kl_gaussian_forward(&mu, &mu, &stats) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        vae_c_kl_gaussian_forward(&rank_one, &logvar, &stats) !=
            VAE_C_OP_SHAPE_MISMATCH ||
        vae_c_kl_gaussian_forward(
            &mu,
            &different_shape,
            &stats) != VAE_C_OP_SHAPE_MISMATCH) {

        return 0;
    }

    corrupted = mu;
    corrupted.shape.strides[1] = 2;
    if (!forward_rejected_preserving_pair(
            &corrupted,
            &logvar,
            &stats,
            VAE_C_OP_INVALID_STATE)) {

        return 0;
    }

    corrupted = mu;
    corrupted.shape.dims[0] = SIZE_MAX;
    corrupted.shape.dims[1] = 2;
    if (!forward_rejected_preserving_pair(
            &corrupted,
            &logvar,
            &stats,
            VAE_C_OP_OVERFLOW)) {

        return 0;
    }

    corrupted = mu;
    corrupted.owns_memory = true;
    corrupted.owned_block = corrupted.data;
    corrupted.owned_bytes = sizeof(float);
    if (!forward_rejected_preserving_pair(
            &corrupted,
            &logvar,
            &stats,
            VAE_C_OP_INVALID_STATE)) {

        return 0;
    }

    aliased = logvar;
    aliased.data = mu.grad;
    if (!forward_rejected_preserving_pair(
            &mu,
            &aliased,
            &stats,
            VAE_C_OP_INVALID_ARGUMENT)) {

        return 0;
    }

    memcpy(mu_data_before, mu_data, sizeof(mu_data));
    if (vae_c_kl_gaussian_forward(
            &mu,
            &logvar,
            (VAECKLGaussianStats *)(void *)mu.data) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        !arrays_bitwise_equal(mu_data, mu_data_before, 2)) {

        return 0;
    }

    mu_data[0] = NAN;
    if (!forward_rejected_preserving_pair(
            &mu, &logvar, &stats, VAE_C_OP_INVALID_STATE)) {
        return 0;
    }
    mu_data[0] = INFINITY;
    if (!forward_rejected_preserving_pair(
            &mu, &logvar, &stats, VAE_C_OP_INVALID_STATE)) {
        return 0;
    }
    mu_data[0] = -INFINITY;
    if (!forward_rejected_preserving_pair(
            &mu, &logvar, &stats, VAE_C_OP_INVALID_STATE)) {
        return 0;
    }
    mu_data[0] = 0.2f;

    logvar_data[0] = NAN;
    if (!forward_rejected_preserving_pair(
            &mu, &logvar, &stats, VAE_C_OP_INVALID_STATE)) {
        return 0;
    }
    logvar_data[0] = INFINITY;
    if (!forward_rejected_preserving_pair(
            &mu, &logvar, &stats, VAE_C_OP_INVALID_STATE)) {
        return 0;
    }
    logvar_data[0] = -INFINITY;
    if (!forward_rejected_preserving_pair(
            &mu, &logvar, &stats, VAE_C_OP_INVALID_STATE)) {
        return 0;
    }
    logvar_data[0] = 100.0f;
    if (!forward_rejected_preserving_pair(
            &mu, &logvar, &stats, VAE_C_OP_INVALID_STATE)) {
        return 0;
    }
    logvar_data[0] = 0.25f;

    return 1;
}


static int backward_rejected_preserving_pair(
    VAECTensor *mu,
    VAECTensor *logvar,
    float upstream_gradient,
    VAECOpStatus expected_status
)
{
    float mu_data_before[2];
    float mu_grad_before[2];
    float logvar_data_before[2];
    float logvar_grad_before[2];

    memcpy(mu_data_before, mu->data, sizeof(mu_data_before));
    memcpy(mu_grad_before, mu->grad, sizeof(mu_grad_before));
    memcpy(logvar_data_before, logvar->data, sizeof(logvar_data_before));
    memcpy(
        logvar_grad_before,
        logvar->grad,
        sizeof(logvar_grad_before)
    );

    return
        vae_c_kl_gaussian_backward(
            mu,
            logvar,
            upstream_gradient) == expected_status &&
        arrays_bitwise_equal(mu->data, mu_data_before, 2) &&
        arrays_bitwise_equal(mu->grad, mu_grad_before, 2) &&
        arrays_bitwise_equal(logvar->data, logvar_data_before, 2) &&
        arrays_bitwise_equal(logvar->grad, logvar_grad_before, 2);
}


static int test_backward_negative_contracts(void)
{
    float mu_data[2] = { 0.2f, -0.3f };
    float mu_grad[2] = { 1.0f, -2.0f };
    float logvar_data[2] = { 0.25f, -0.75f };
    float logvar_grad[2] = { 3.0f, -4.0f };
    float mu_grad_before[2];
    float logvar_grad_before[2];
    VAECTensor mu;
    VAECTensor logvar;
    VAECTensor no_gradient;
    VAECTensor aliased;

    if (!make_view_2d(&mu, mu_data, mu_grad, 1, 2) ||
        !make_view_2d(
            &logvar,
            logvar_data,
            logvar_grad,
            1,
            2)) {

        return 0;
    }

    if (vae_c_kl_gaussian_backward(NULL, &logvar, 1.0f) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        vae_c_kl_gaussian_backward(&mu, NULL, 1.0f) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        vae_c_kl_gaussian_backward(&mu, &mu, 1.0f) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        !backward_rejected_preserving_pair(
            &mu, &logvar, NAN, VAE_C_OP_INVALID_ARGUMENT) ||
        !backward_rejected_preserving_pair(
            &mu, &logvar, INFINITY, VAE_C_OP_INVALID_ARGUMENT) ||
        !backward_rejected_preserving_pair(
            &mu, &logvar, -INFINITY, VAE_C_OP_INVALID_ARGUMENT)) {

        return 0;
    }

    memcpy(mu_grad_before, mu_grad, sizeof(mu_grad));
    memcpy(logvar_grad_before, logvar_grad, sizeof(logvar_grad));
    no_gradient = mu;
    no_gradient.grad = NULL;
    no_gradient.has_grad = false;
    if (vae_c_kl_gaussian_backward(
            &no_gradient,
            &logvar,
            1.0f) != VAE_C_OP_GRADIENT_NOT_AVAILABLE ||
        !arrays_bitwise_equal(mu_grad, mu_grad_before, 2) ||
        !arrays_bitwise_equal(logvar_grad, logvar_grad_before, 2)) {

        return 0;
    }

    no_gradient = logvar;
    no_gradient.grad = NULL;
    no_gradient.has_grad = false;
    if (vae_c_kl_gaussian_backward(
            &mu,
            &no_gradient,
            1.0f) != VAE_C_OP_GRADIENT_NOT_AVAILABLE ||
        !arrays_bitwise_equal(mu_grad, mu_grad_before, 2) ||
        !arrays_bitwise_equal(logvar_grad, logvar_grad_before, 2)) {

        return 0;
    }

    aliased = logvar;
    aliased.data = mu.grad;
    if (!backward_rejected_preserving_pair(
            &mu,
            &aliased,
            1.0f,
            VAE_C_OP_INVALID_ARGUMENT)) {

        return 0;
    }

    mu_data[0] = NAN;
    if (!backward_rejected_preserving_pair(
            &mu, &logvar, 1.0f, VAE_C_OP_INVALID_STATE)) {
        return 0;
    }
    mu_data[0] = INFINITY;
    if (!backward_rejected_preserving_pair(
            &mu, &logvar, 1.0f, VAE_C_OP_INVALID_STATE)) {
        return 0;
    }
    mu_data[0] = -INFINITY;
    if (!backward_rejected_preserving_pair(
            &mu, &logvar, 1.0f, VAE_C_OP_INVALID_STATE)) {
        return 0;
    }
    mu_data[0] = 0.2f;

    logvar_data[0] = NAN;
    if (!backward_rejected_preserving_pair(
            &mu, &logvar, 1.0f, VAE_C_OP_INVALID_STATE)) {
        return 0;
    }
    logvar_data[0] = INFINITY;
    if (!backward_rejected_preserving_pair(
            &mu, &logvar, 1.0f, VAE_C_OP_INVALID_STATE)) {
        return 0;
    }
    logvar_data[0] = -INFINITY;
    if (!backward_rejected_preserving_pair(
            &mu, &logvar, 1.0f, VAE_C_OP_INVALID_STATE)) {
        return 0;
    }
    logvar_data[0] = 100.0f;
    if (!backward_rejected_preserving_pair(
            &mu, &logvar, 1.0f, VAE_C_OP_INVALID_STATE)) {
        return 0;
    }
    logvar_data[0] = 0.25f;

    mu_grad[0] = NAN;
    if (!backward_rejected_preserving_pair(
            &mu, &logvar, 1.0f, VAE_C_OP_INVALID_STATE)) {
        return 0;
    }
    mu_grad[0] = 1.0f;
    logvar_grad[0] = NAN;
    if (!backward_rejected_preserving_pair(
            &mu, &logvar, 1.0f, VAE_C_OP_INVALID_STATE)) {
        return 0;
    }
    logvar_grad[0] = 3.0f;

    mu_data[0] = FLT_MAX;
    mu_grad[0] = FLT_MAX;
    if (!backward_rejected_preserving_pair(
            &mu, &logvar, 1.0f, VAE_C_OP_INVALID_STATE)) {
        return 0;
    }
    mu_data[0] = 0.0f;
    mu_grad[0] = 1.0f;

    logvar_data[0] = logf(2.0f);
    logvar_grad[0] = FLT_MAX;
    if (!backward_rejected_preserving_pair(
            &mu,
            &logvar,
            FLT_MAX,
            VAE_C_OP_INVALID_STATE)) {

        return 0;
    }

    return 1;
}


int main(void)
{
    int failures = 0;

    failures += report_check(
        "kl_stats_init_and_known_values",
        test_stats_init_and_known_values()
    );
    failures += report_check(
        "kl_analytic_forward_backward",
        test_analytic_forward_backward()
    );
    failures += report_check(
        "kl_near_zero_logvar",
        test_near_zero_logvar()
    );
    failures += report_check(
        "kl_extreme_finite_values",
        test_extreme_finite_values()
    );
    failures += report_check(
        "kl_gradient_check",
        test_gradient_check()
    );
    failures += report_check(
        "kl_reduction_contract",
        test_reduction_contract()
    );
    failures += report_check(
        "kl_forward_negative_contracts",
        test_forward_negative_contracts()
    );
    failures += report_check(
        "kl_backward_negative_contracts",
        test_backward_negative_contracts()
    );

    if (failures != 0) {
        fprintf(stderr, "%d VAE C KL TEST(S) FAILED\n", failures);
        return 1;
    }

    printf("ALL VAE C KL TESTS PASSED\n");
    return 0;
}
