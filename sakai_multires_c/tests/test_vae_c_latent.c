#include "vae_c_latent.h"
#include "vae_c_rng.h"

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
    size_t rows,
    size_t columns
)
{
    size_t dims[2];

    if (tensor == NULL || data == NULL) {
        return 0;
    }

    dims[0] = rows;
    dims[1] = columns;
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


static int arrays_bitwise_equal(
    const float *left,
    const float *right,
    size_t count
)
{
    return memcmp(left, right, count * sizeof(float)) == 0;
}


static int arrays_differ(
    const float *left,
    const float *right,
    size_t count
)
{
    return !arrays_bitwise_equal(left, right, count);
}


static void fill_f32(float *values, size_t count, float value)
{
    size_t index;

    for (index = 0; index < count; index++) {
        values[index] = value;
    }
}


static int test_split_analytic_layout_and_backward(void)
{
    enum { packed_count = 12, latent_count = 6 };
    float packed_data[packed_count] = {
        1.0f, 2.0f, 3.0f, 101.0f, 102.0f, 103.0f,
        4.0f, 5.0f, 6.0f, 104.0f, 105.0f, 106.0f
    };
    float packed_grad[packed_count];
    float mu_data[latent_count];
    float mu_grad[latent_count] = {
        0.25f, -0.5f, 0.75f, -1.0f, 1.25f, -1.5f
    };
    float logvar_data[latent_count];
    float logvar_grad[latent_count] = {
        2.0f, -2.5f, 3.0f, -3.5f, 4.0f, -4.5f
    };
    const float expected_mu[latent_count] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f
    };
    const float expected_logvar[latent_count] = {
        101.0f, 102.0f, 103.0f, 104.0f, 105.0f, 106.0f
    };
    float expected_packed_grad[packed_count];
    float packed_data_before[packed_count];
    float mu_grad_before[latent_count];
    float logvar_grad_before[latent_count];
    VAECTensor packed;
    VAECTensor mu;
    VAECTensor logvar;
    size_t row;
    size_t latent;

    fill_f32(packed_grad, packed_count, 0.5f);
    fill_f32(mu_data, latent_count, -91.0f);
    fill_f32(logvar_data, latent_count, -92.0f);
    memcpy(packed_data_before, packed_data, sizeof(packed_data));
    memcpy(mu_grad_before, mu_grad, sizeof(mu_grad));
    memcpy(logvar_grad_before, logvar_grad, sizeof(logvar_grad));

    if (!make_view_2d(&packed, packed_data, packed_grad, 2, 6) ||
        !make_view_2d(&mu, mu_data, mu_grad, 2, 3) ||
        !make_view_2d(&logvar, logvar_data, logvar_grad, 2, 3) ||
        vae_c_latent_split_forward(&packed, &mu, &logvar) !=
            VAE_C_OP_OK ||
        !arrays_bitwise_equal(mu_data, expected_mu, latent_count) ||
        !arrays_bitwise_equal(
            logvar_data,
            expected_logvar,
            latent_count) ||
        !arrays_bitwise_equal(
            packed_data,
            packed_data_before,
            packed_count) ||
        !arrays_bitwise_equal(mu_grad, mu_grad_before, latent_count) ||
        !arrays_bitwise_equal(
            logvar_grad,
            logvar_grad_before,
            latent_count)) {

        return 0;
    }

    if (vae_c_latent_split_backward(&packed, &mu, &logvar) !=
        VAE_C_OP_OK) {

        return 0;
    }

    for (row = 0; row < 2; row++) {
        for (latent = 0; latent < 3; latent++) {
            size_t output_index = row * 3 + latent;
            size_t packed_row = row * 6;

            expected_packed_grad[packed_row + latent] =
                0.5f + mu_grad[output_index];
            expected_packed_grad[packed_row + 3 + latent] =
                0.5f + logvar_grad[output_index];
        }
    }

    if (!arrays_bitwise_equal(
            packed_grad,
            expected_packed_grad,
            packed_count) ||
        vae_c_latent_split_backward(&packed, &mu, &logvar) !=
            VAE_C_OP_OK) {

        return 0;
    }

    for (row = 0; row < 2; row++) {
        for (latent = 0; latent < 3; latent++) {
            size_t output_index = row * 3 + latent;
            size_t packed_row = row * 6;

            expected_packed_grad[packed_row + latent] =
                0.5f + 2.0f * mu_grad[output_index];
            expected_packed_grad[packed_row + 3 + latent] =
                0.5f + 2.0f * logvar_grad[output_index];
        }
    }

    return
        arrays_bitwise_equal(
            packed_grad,
            expected_packed_grad,
            packed_count) &&
        arrays_bitwise_equal(packed_data, packed_data_before, packed_count) &&
        arrays_bitwise_equal(mu_data, expected_mu, latent_count) &&
        arrays_bitwise_equal(logvar_data, expected_logvar, latent_count) &&
        arrays_bitwise_equal(mu_grad, mu_grad_before, latent_count) &&
        arrays_bitwise_equal(
            logvar_grad,
            logvar_grad_before,
            latent_count);
}


static int test_stochastic_analytic(void)
{
    enum { count = 6 };
    float mu_data[count] = {
        -0.4f, 0.2f, 0.7f, -0.1f, 0.5f, -0.8f
    };
    float logvar_data[count] = {
        -0.6f, 0.0f, 0.4f, -0.2f, 0.8f, -1.0f
    };
    float epsilon_data[count] = {
        0.3f, -0.7f, 1.1f, -0.2f, 0.6f, -1.3f
    };
    float epsilon_grad[count] = {
        9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f
    };
    float z_data[count];
    float mu_grad[count];
    float logvar_grad[count];
    float z_grad[count] = {
        0.5f, -0.4f, 0.3f, -0.2f, 0.7f, -0.6f
    };
    float expected_z[count];
    float expected_mu_grad[count];
    float expected_logvar_grad[count];
    float mu_before[count];
    float logvar_before[count];
    float epsilon_before[count];
    float epsilon_grad_before[count];
    float z_grad_before[count];
    VAECTensor mu;
    VAECTensor logvar;
    VAECTensor epsilon;
    VAECTensor z;
    size_t index;

    fill_f32(z_data, count, -50.0f);
    fill_f32(mu_grad, count, 0.125f);
    fill_f32(logvar_grad, count, -0.25f);
    memcpy(mu_before, mu_data, sizeof(mu_data));
    memcpy(logvar_before, logvar_data, sizeof(logvar_data));
    memcpy(epsilon_before, epsilon_data, sizeof(epsilon_data));
    memcpy(
        epsilon_grad_before,
        epsilon_grad,
        sizeof(epsilon_grad)
    );
    memcpy(z_grad_before, z_grad, sizeof(z_grad));

    for (index = 0; index < count; index++) {
        double sigma = exp(0.5 * (double)logvar_data[index]);

        expected_z[index] = (float)(
            (double)mu_data[index] +
            sigma * (double)epsilon_data[index]
        );
        expected_mu_grad[index] = 0.125f + z_grad[index];
        expected_logvar_grad[index] = (float)(
            -0.25 +
            (double)z_grad[index] * 0.5 * sigma *
                (double)epsilon_data[index]
        );
    }

    if (!make_view_2d(&mu, mu_data, mu_grad, 2, 3) ||
        !make_view_2d(&logvar, logvar_data, logvar_grad, 2, 3) ||
        !make_view_2d(&epsilon, epsilon_data, epsilon_grad, 2, 3) ||
        !make_view_2d(&z, z_data, z_grad, 2, 3) ||
        vae_c_reparameterize_forward(
            VAE_C_LATENT_MODE_STOCHASTIC,
            &mu,
            &logvar,
            &epsilon,
            &z) != VAE_C_OP_OK ||
        !arrays_close_f32(z_data, expected_z, count, 2.0e-6f) ||
        vae_c_reparameterize_backward(
            VAE_C_LATENT_MODE_STOCHASTIC,
            &mu,
            &logvar,
            &epsilon,
            &z) != VAE_C_OP_OK ||
        !arrays_close_f32(
            mu_grad,
            expected_mu_grad,
            count,
            2.0e-6f) ||
        !arrays_close_f32(
            logvar_grad,
            expected_logvar_grad,
            count,
            2.0e-6f)) {

        return 0;
    }

    for (index = 0; index < count; index++) {
        expected_mu_grad[index] =
            0.125f + 2.0f * z_grad[index];
        expected_logvar_grad[index] = (float)(
            -0.25 +
            2.0 * (double)z_grad[index] * 0.5 *
                exp(0.5 * (double)logvar_data[index]) *
                (double)epsilon_data[index]
        );
    }

    return
        vae_c_reparameterize_backward(
            VAE_C_LATENT_MODE_STOCHASTIC,
            &mu,
            &logvar,
            &epsilon,
            &z) == VAE_C_OP_OK &&
        arrays_close_f32(mu_grad, expected_mu_grad, count, 3.0e-6f) &&
        arrays_close_f32(
            logvar_grad,
            expected_logvar_grad,
            count,
            3.0e-6f) &&
        arrays_bitwise_equal(mu_data, mu_before, count) &&
        arrays_bitwise_equal(logvar_data, logvar_before, count) &&
        arrays_bitwise_equal(epsilon_data, epsilon_before, count) &&
        arrays_bitwise_equal(
            epsilon_grad,
            epsilon_grad_before,
            count) &&
        arrays_bitwise_equal(z_grad, z_grad_before, count);
}


static int test_mean_analytic(void)
{
    enum { count = 6 };
    float mu_data[count] = {
        -1.0f, -0.5f, 0.0f, 0.25f, 0.75f, 1.25f
    };
    float logvar_data[count] = {
        -0.3f, 0.2f, 0.4f, -0.8f, 0.6f, -0.1f
    };
    float z_data[count];
    float mu_grad[count];
    float logvar_grad[count];
    float z_grad[count] = {
        0.2f, -0.3f, 0.4f, -0.5f, 0.6f, -0.7f
    };
    float logvar_grad_before[count];
    float z_grad_before[count];
    float expected_mu_grad[count];
    VAECTensor mu;
    VAECTensor logvar;
    VAECTensor z;
    size_t index;

    fill_f32(z_data, count, 33.0f);
    fill_f32(mu_grad, count, 0.1f);
    fill_f32(logvar_grad, count, -0.2f);
    memcpy(logvar_grad_before, logvar_grad, sizeof(logvar_grad));
    memcpy(z_grad_before, z_grad, sizeof(z_grad));

    if (!make_view_2d(&mu, mu_data, mu_grad, 2, 3) ||
        !make_view_2d(&logvar, logvar_data, logvar_grad, 2, 3) ||
        !make_view_2d(&z, z_data, z_grad, 2, 3) ||
        vae_c_reparameterize_forward(
            VAE_C_LATENT_MODE_MEAN,
            &mu,
            &logvar,
            NULL,
            &z) != VAE_C_OP_OK ||
        !arrays_bitwise_equal(z_data, mu_data, count) ||
        vae_c_reparameterize_backward(
            VAE_C_LATENT_MODE_MEAN,
            &mu,
            &logvar,
            NULL,
            &z) != VAE_C_OP_OK) {

        return 0;
    }

    for (index = 0; index < count; index++) {
        expected_mu_grad[index] = 0.1f + z_grad[index];
    }

    if (!arrays_bitwise_equal(mu_grad, expected_mu_grad, count) ||
        !arrays_bitwise_equal(
            logvar_grad,
            logvar_grad_before,
            count) ||
        vae_c_reparameterize_backward(
            VAE_C_LATENT_MODE_MEAN,
            &mu,
            &logvar,
            NULL,
            &z) != VAE_C_OP_OK) {

        return 0;
    }

    for (index = 0; index < count; index++) {
        expected_mu_grad[index] += z_grad[index];
    }

    return
        arrays_bitwise_equal(mu_grad, expected_mu_grad, count) &&
        arrays_bitwise_equal(logvar_grad, logvar_grad_before, count) &&
        arrays_bitwise_equal(z_grad, z_grad_before, count) &&
        arrays_bitwise_equal(z_data, mu_data, count);
}


static int make_rng_z(
    uint64_t base_seed,
    uint64_t epoch,
    uint64_t manifest_index,
    const float *mu_data,
    const float *logvar_data,
    float *epsilon_data,
    float *z_data,
    size_t latent_count
)
{
    VAECTensor mu;
    VAECTensor logvar;
    VAECTensor epsilon;
    VAECTensor z;

    if (vae_c_rng_fill_latent_train_epsilon_f32(
            base_seed,
            epoch,
            manifest_index,
            epsilon_data,
            latent_count) != VAE_C_RNG_OK ||
        !make_view_2d(
            &mu,
            (float *)mu_data,
            NULL,
            1,
            latent_count) ||
        !make_view_2d(
            &logvar,
            (float *)logvar_data,
            NULL,
            1,
            latent_count) ||
        !make_view_2d(
            &epsilon,
            epsilon_data,
            NULL,
            1,
            latent_count) ||
        !make_view_2d(&z, z_data, NULL, 1, latent_count)) {

        return 0;
    }

    return
        vae_c_reparameterize_forward(
            VAE_C_LATENT_MODE_STOCHASTIC,
            &mu,
            &logvar,
            &epsilon,
            &z) == VAE_C_OP_OK;
}


static int test_rng_integration(void)
{
    enum { count = 8 };
    const float mu_data[count] = {
        -0.4f, -0.3f, -0.2f, -0.1f,
        0.1f, 0.2f, 0.3f, 0.4f
    };
    const float logvar_data[count] = {
        -0.8f, -0.6f, -0.4f, -0.2f,
        0.0f, 0.2f, 0.4f, 0.6f
    };
    float epsilon_reference[count];
    float epsilon_repeat[count];
    float epsilon_next_epoch[count];
    float z_reference[count];
    float z_repeat[count];
    float z_next_epoch[count];
    float epsilon_ab_a[count];
    float epsilon_ab_b[count];
    float epsilon_ba_a[count];
    float epsilon_ba_b[count];
    float z_ab_a[count];
    float z_ab_b[count];
    float z_ba_a[count];
    float z_ba_b[count];

    return
        make_rng_z(
            UINT64_C(456), UINT64_C(3), UINT64_C(42),
            mu_data, logvar_data, epsilon_reference, z_reference,
            count) &&
        make_rng_z(
            UINT64_C(456), UINT64_C(3), UINT64_C(42),
            mu_data, logvar_data, epsilon_repeat, z_repeat,
            count) &&
        make_rng_z(
            UINT64_C(456), UINT64_C(4), UINT64_C(42),
            mu_data, logvar_data, epsilon_next_epoch, z_next_epoch,
            count) &&
        arrays_bitwise_equal(
            epsilon_reference,
            epsilon_repeat,
            count) &&
        arrays_bitwise_equal(z_reference, z_repeat, count) &&
        arrays_differ(
            epsilon_reference,
            epsilon_next_epoch,
            count) &&
        arrays_differ(z_reference, z_next_epoch, count) &&
        make_rng_z(
            UINT64_C(9001), UINT64_C(8), UINT64_C(100),
            mu_data, logvar_data, epsilon_ab_a, z_ab_a, count) &&
        make_rng_z(
            UINT64_C(9001), UINT64_C(8), UINT64_C(200),
            mu_data, logvar_data, epsilon_ab_b, z_ab_b, count) &&
        make_rng_z(
            UINT64_C(9001), UINT64_C(8), UINT64_C(200),
            mu_data, logvar_data, epsilon_ba_b, z_ba_b, count) &&
        make_rng_z(
            UINT64_C(9001), UINT64_C(8), UINT64_C(100),
            mu_data, logvar_data, epsilon_ba_a, z_ba_a, count) &&
        arrays_bitwise_equal(epsilon_ab_a, epsilon_ba_a, count) &&
        arrays_bitwise_equal(epsilon_ab_b, epsilon_ba_b, count) &&
        arrays_bitwise_equal(z_ab_a, z_ba_a, count) &&
        arrays_bitwise_equal(z_ab_b, z_ba_b, count);
}


static double split_objective(
    VAECTensor *packed,
    VAECTensor *mu,
    VAECTensor *logvar,
    const float *upstream_mu,
    const float *upstream_logvar,
    int *ok
)
{
    double objective;
    size_t index;

    if (vae_c_latent_split_forward(packed, mu, logvar) !=
        VAE_C_OP_OK) {

        *ok = 0;
        return 0.0;
    }

    objective = 0.0;
    for (index = 0; index < mu->shape.count; index++) {
        objective +=
            (double)mu->data[index] * (double)upstream_mu[index];
        objective +=
            (double)logvar->data[index] *
                (double)upstream_logvar[index];
    }

    return objective;
}


static int test_split_gradient_check(void)
{
    enum { packed_count = 12, latent_count = 6 };
    const float numeric_epsilon = 1.0e-3f;
    float packed_data[packed_count] = {
        -0.4f, 0.2f, 0.7f, -0.6f, 0.1f, 0.5f,
        0.3f, -0.8f, 0.9f, 0.4f, -0.2f, 0.6f
    };
    float packed_grad[packed_count];
    float mu_data[latent_count];
    float mu_grad[latent_count] = {
        0.25f, -0.35f, 0.45f, -0.55f, 0.65f, -0.75f
    };
    float logvar_data[latent_count];
    float logvar_grad[latent_count] = {
        -0.8f, 0.7f, -0.6f, 0.5f, -0.4f, 0.3f
    };
    VAECTensor packed;
    VAECTensor mu;
    VAECTensor logvar;
    double max_absolute_error;
    double max_relative_error;
    size_t worst_index;
    size_t index;
    int ok;

    fill_f32(packed_grad, packed_count, 0.0f);
    fill_f32(mu_data, latent_count, 0.0f);
    fill_f32(logvar_data, latent_count, 0.0f);

    if (!make_view_2d(&packed, packed_data, packed_grad, 2, 6) ||
        !make_view_2d(&mu, mu_data, mu_grad, 2, 3) ||
        !make_view_2d(&logvar, logvar_data, logvar_grad, 2, 3) ||
        vae_c_latent_split_forward(&packed, &mu, &logvar) !=
            VAE_C_OP_OK ||
        vae_c_latent_split_backward(&packed, &mu, &logvar) !=
            VAE_C_OP_OK) {

        return 0;
    }

    max_absolute_error = 0.0;
    max_relative_error = 0.0;
    worst_index = 0;
    ok = 1;

    for (index = 0; index < packed_count; index++) {
        float original = packed_data[index];
        double objective_plus;
        double objective_minus;
        double numeric_gradient;
        double analytic_gradient;
        double absolute_error;
        double denominator;
        double relative_error;

        packed_data[index] = original + numeric_epsilon;
        objective_plus = split_objective(
            &packed,
            &mu,
            &logvar,
            mu_grad,
            logvar_grad,
            &ok
        );
        packed_data[index] = original - numeric_epsilon;
        objective_minus = split_objective(
            &packed,
            &mu,
            &logvar,
            mu_grad,
            logvar_grad,
            &ok
        );
        packed_data[index] = original;

        if (!ok) {
            return 0;
        }

        numeric_gradient =
            (objective_plus - objective_minus) /
            (2.0 * (double)numeric_epsilon);
        analytic_gradient = (double)packed_grad[index];
        absolute_error = fabs(numeric_gradient - analytic_gradient);
        denominator = fmax(
            fmax(fabs(numeric_gradient), fabs(analytic_gradient)),
            1.0e-12
        );
        relative_error = absolute_error / denominator;

        if (absolute_error > max_absolute_error) {
            max_absolute_error = absolute_error;
            worst_index = index;
        }
        if (relative_error > max_relative_error) {
            max_relative_error = relative_error;
        }
    }

    printf(
        "LATENT_SPLIT_GRAD_MAX_ABS=%.9g MAX_REL=%.9g WORST_INDEX=%zu\n",
        max_absolute_error,
        max_relative_error,
        worst_index
    );

    return
        max_absolute_error <= 3.0e-4 &&
        max_relative_error <= 3.0e-3;
}


static double reparameterization_objective(
    VAECTensor *mu,
    VAECTensor *logvar,
    VAECTensor *epsilon,
    VAECTensor *z,
    const float *upstream,
    int *ok
)
{
    double objective;
    size_t index;

    if (vae_c_reparameterize_forward(
            VAE_C_LATENT_MODE_STOCHASTIC,
            mu,
            logvar,
            epsilon,
            z) != VAE_C_OP_OK) {

        *ok = 0;
        return 0.0;
    }

    objective = 0.0;
    for (index = 0; index < z->shape.count; index++) {
        objective += (double)z->data[index] * (double)upstream[index];
    }

    return objective;
}


static void update_error_metrics(
    double numeric_gradient,
    double analytic_gradient,
    size_t index,
    double *maximum_absolute_error,
    double *maximum_relative_error,
    size_t *worst_index
)
{
    double absolute_error = fabs(numeric_gradient - analytic_gradient);
    double denominator = fmax(
        fmax(fabs(numeric_gradient), fabs(analytic_gradient)),
        1.0e-12
    );
    double relative_error = absolute_error / denominator;

    if (absolute_error > *maximum_absolute_error) {
        *maximum_absolute_error = absolute_error;
        *worst_index = index;
    }
    if (relative_error > *maximum_relative_error) {
        *maximum_relative_error = relative_error;
    }
}


static int test_reparameterization_gradient_check(void)
{
    enum { count = 6 };
    const float numeric_epsilon = 1.0e-3f;
    float mu_data[count] = {
        -0.6f, -0.2f, 0.1f, 0.4f, 0.7f, 1.0f
    };
    float logvar_data[count] = {
        -0.8f, -0.4f, -0.1f, 0.2f, 0.5f, 0.7f
    };
    float epsilon_data[count] = {
        0.4f, -0.7f, 0.9f, -0.3f, 0.6f, -0.8f
    };
    float z_data[count];
    float mu_grad[count];
    float logvar_grad[count];
    float z_grad[count] = {
        0.3f, -0.45f, 0.55f, -0.65f, 0.75f, -0.85f
    };
    float epsilon_before[count];
    float z_grad_before[count];
    VAECTensor mu;
    VAECTensor logvar;
    VAECTensor epsilon;
    VAECTensor z;
    double mu_max_absolute_error;
    double mu_max_relative_error;
    double logvar_max_absolute_error;
    double logvar_max_relative_error;
    size_t mu_worst_index;
    size_t logvar_worst_index;
    size_t index;
    int ok;

    fill_f32(z_data, count, 0.0f);
    fill_f32(mu_grad, count, 0.0f);
    fill_f32(logvar_grad, count, 0.0f);
    memcpy(epsilon_before, epsilon_data, sizeof(epsilon_data));
    memcpy(z_grad_before, z_grad, sizeof(z_grad));

    if (!make_view_2d(&mu, mu_data, mu_grad, 2, 3) ||
        !make_view_2d(&logvar, logvar_data, logvar_grad, 2, 3) ||
        !make_view_2d(&epsilon, epsilon_data, NULL, 2, 3) ||
        !make_view_2d(&z, z_data, z_grad, 2, 3) ||
        vae_c_reparameterize_forward(
            VAE_C_LATENT_MODE_STOCHASTIC,
            &mu,
            &logvar,
            &epsilon,
            &z) != VAE_C_OP_OK ||
        vae_c_reparameterize_backward(
            VAE_C_LATENT_MODE_STOCHASTIC,
            &mu,
            &logvar,
            &epsilon,
            &z) != VAE_C_OP_OK) {

        return 0;
    }

    mu_max_absolute_error = 0.0;
    mu_max_relative_error = 0.0;
    logvar_max_absolute_error = 0.0;
    logvar_max_relative_error = 0.0;
    mu_worst_index = 0;
    logvar_worst_index = 0;
    ok = 1;

    for (index = 0; index < count; index++) {
        float original = mu_data[index];
        double objective_plus;
        double objective_minus;
        double numeric_gradient;

        mu_data[index] = original + numeric_epsilon;
        objective_plus = reparameterization_objective(
            &mu, &logvar, &epsilon, &z, z_grad, &ok
        );
        mu_data[index] = original - numeric_epsilon;
        objective_minus = reparameterization_objective(
            &mu, &logvar, &epsilon, &z, z_grad, &ok
        );
        mu_data[index] = original;

        if (!ok) {
            return 0;
        }

        numeric_gradient =
            (objective_plus - objective_minus) /
            (2.0 * (double)numeric_epsilon);
        update_error_metrics(
            numeric_gradient,
            (double)mu_grad[index],
            index,
            &mu_max_absolute_error,
            &mu_max_relative_error,
            &mu_worst_index
        );
    }

    for (index = 0; index < count; index++) {
        float original = logvar_data[index];
        double objective_plus;
        double objective_minus;
        double numeric_gradient;

        logvar_data[index] = original + numeric_epsilon;
        objective_plus = reparameterization_objective(
            &mu, &logvar, &epsilon, &z, z_grad, &ok
        );
        logvar_data[index] = original - numeric_epsilon;
        objective_minus = reparameterization_objective(
            &mu, &logvar, &epsilon, &z, z_grad, &ok
        );
        logvar_data[index] = original;

        if (!ok) {
            return 0;
        }

        numeric_gradient =
            (objective_plus - objective_minus) /
            (2.0 * (double)numeric_epsilon);
        update_error_metrics(
            numeric_gradient,
            (double)logvar_grad[index],
            index,
            &logvar_max_absolute_error,
            &logvar_max_relative_error,
            &logvar_worst_index
        );
    }

    printf(
        "REPARAM_MU_GRAD_MAX_ABS=%.9g MAX_REL=%.9g WORST_INDEX=%zu\n",
        mu_max_absolute_error,
        mu_max_relative_error,
        mu_worst_index
    );
    printf(
        "REPARAM_LOGVAR_GRAD_MAX_ABS=%.9g MAX_REL=%.9g WORST_INDEX=%zu\n",
        logvar_max_absolute_error,
        logvar_max_relative_error,
        logvar_worst_index
    );

    return
        mu_max_absolute_error <= 3.0e-4 &&
        mu_max_relative_error <= 3.0e-3 &&
        logvar_max_absolute_error <= 3.0e-4 &&
        logvar_max_relative_error <= 3.0e-3 &&
        arrays_bitwise_equal(epsilon_data, epsilon_before, count) &&
        arrays_bitwise_equal(z_grad, z_grad_before, count);
}


static int split_forward_rejects_preserving_outputs(
    VAECTensor *packed,
    VAECTensor *mu,
    VAECTensor *logvar,
    const float *mu_before,
    const float *logvar_before
)
{
    return
        vae_c_latent_split_forward(packed, mu, logvar) != VAE_C_OP_OK &&
        arrays_bitwise_equal(
            mu->data,
            mu_before,
            mu->shape.count) &&
        arrays_bitwise_equal(
            logvar->data,
            logvar_before,
            logvar->shape.count);
}


static int reparameterize_forward_rejects_preserving_z(
    VAECLatentMode mode,
    VAECTensor *mu,
    VAECTensor *logvar,
    VAECTensor *epsilon,
    VAECTensor *z,
    const float *z_before
)
{
    return
        vae_c_reparameterize_forward(
            mode,
            mu,
            logvar,
            epsilon,
            z) != VAE_C_OP_OK &&
        arrays_bitwise_equal(z->data, z_before, z->shape.count);
}


static int test_nonfinite_and_strong_guarantee(void)
{
    float packed_data[4] = { 0.1f, 0.2f, 0.3f, 0.4f };
    float packed_grad[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float mu_data[2] = { 0.1f, -0.2f };
    float mu_grad[2] = { 0.0f, 0.0f };
    float logvar_data[2] = { 0.2f, -0.3f };
    float logvar_grad[2] = { 0.0f, 0.0f };
    float epsilon_data[2] = { 0.4f, -0.5f };
    float z_data[2] = { 71.0f, 72.0f };
    float z_grad[2] = { 0.6f, -0.7f };
    float mu_before[2];
    float logvar_before[2];
    float z_before[2];
    float packed_grad_before[4];
    float mu_grad_before[2];
    float logvar_grad_before[2];
    VAECTensor packed;
    VAECTensor mu;
    VAECTensor logvar;
    VAECTensor epsilon;
    VAECTensor z;

    if (!make_view_2d(&packed, packed_data, packed_grad, 1, 4) ||
        !make_view_2d(&mu, mu_data, mu_grad, 1, 2) ||
        !make_view_2d(&logvar, logvar_data, logvar_grad, 1, 2) ||
        !make_view_2d(&epsilon, epsilon_data, NULL, 1, 2) ||
        !make_view_2d(&z, z_data, z_grad, 1, 2)) {

        return 0;
    }

    {
        float edge_mu_data[1] = { 0.0f };
        float edge_mu_grad[1] = { 1.25f };
        float edge_logvar_data[1] = { -300.0f };
        float edge_logvar_grad[1] = { -2.5f };
        float edge_epsilon_data[1] = { 1.0f };
        float edge_epsilon_grad[1] = { 3.75f };
        float edge_z_data[1] = { 47.5f };
        float edge_z_grad[1] = { -4.25f };
        float edge_mu_data_before[1];
        float edge_mu_grad_before[1];
        float edge_logvar_data_before[1];
        float edge_logvar_grad_before[1];
        float edge_epsilon_data_before[1];
        float edge_epsilon_grad_before[1];
        float edge_z_data_before[1];
        float edge_z_grad_before[1];
        VAECTensor edge_mu;
        VAECTensor edge_logvar;
        VAECTensor edge_epsilon;
        VAECTensor edge_z;
        volatile float underflow_half_logvar =
            0.5f * edge_logvar_data[0];
        volatile float underflow_sigma =
            expf(underflow_half_logvar);
        volatile float product_half_logvar;
        volatile float product_sigma;
        volatile float product_value;

        if (!isfinite(underflow_sigma) ||
            underflow_sigma != 0.0f ||
            !make_view_2d(
                &edge_mu,
                edge_mu_data,
                edge_mu_grad,
                1,
                1) ||
            !make_view_2d(
                &edge_logvar,
                edge_logvar_data,
                edge_logvar_grad,
                1,
                1) ||
            !make_view_2d(
                &edge_epsilon,
                edge_epsilon_data,
                edge_epsilon_grad,
                1,
                1) ||
            !make_view_2d(
                &edge_z,
                edge_z_data,
                edge_z_grad,
                1,
                1)) {

            return 0;
        }

        memcpy(
            edge_mu_data_before,
            edge_mu_data,
            sizeof(edge_mu_data)
        );
        memcpy(
            edge_mu_grad_before,
            edge_mu_grad,
            sizeof(edge_mu_grad)
        );
        memcpy(
            edge_logvar_data_before,
            edge_logvar_data,
            sizeof(edge_logvar_data)
        );
        memcpy(
            edge_logvar_grad_before,
            edge_logvar_grad,
            sizeof(edge_logvar_grad)
        );
        memcpy(
            edge_epsilon_data_before,
            edge_epsilon_data,
            sizeof(edge_epsilon_data)
        );
        memcpy(
            edge_epsilon_grad_before,
            edge_epsilon_grad,
            sizeof(edge_epsilon_grad)
        );
        memcpy(
            edge_z_data_before,
            edge_z_data,
            sizeof(edge_z_data)
        );
        memcpy(
            edge_z_grad_before,
            edge_z_grad,
            sizeof(edge_z_grad)
        );

        if (vae_c_reparameterize_forward(
                VAE_C_LATENT_MODE_STOCHASTIC,
                &edge_mu,
                &edge_logvar,
                &edge_epsilon,
                &edge_z) != VAE_C_OP_INVALID_STATE ||
            !arrays_bitwise_equal(
                edge_mu_data,
                edge_mu_data_before,
                1) ||
            !arrays_bitwise_equal(
                edge_mu_grad,
                edge_mu_grad_before,
                1) ||
            !arrays_bitwise_equal(
                edge_logvar_data,
                edge_logvar_data_before,
                1) ||
            !arrays_bitwise_equal(
                edge_logvar_grad,
                edge_logvar_grad_before,
                1) ||
            !arrays_bitwise_equal(
                edge_epsilon_data,
                edge_epsilon_data_before,
                1) ||
            !arrays_bitwise_equal(
                edge_epsilon_grad,
                edge_epsilon_grad_before,
                1) ||
            !arrays_bitwise_equal(
                edge_z_data,
                edge_z_data_before,
                1) ||
            !arrays_bitwise_equal(
                edge_z_grad,
                edge_z_grad_before,
                1)) {

            return 0;
        }

        edge_logvar_data[0] = 2.0f;
        edge_epsilon_data[0] = FLT_MAX;
        edge_z_data[0] = -53.5f;
        product_half_logvar = 0.5f * edge_logvar_data[0];
        product_sigma = expf(product_half_logvar);
        product_value = product_sigma * edge_epsilon_data[0];

        if (!isfinite(product_sigma) ||
            !(product_sigma > 0.0f) ||
            !(product_sigma > 1.0f) ||
            isfinite(product_value)) {

            return 0;
        }

        memcpy(
            edge_mu_data_before,
            edge_mu_data,
            sizeof(edge_mu_data)
        );
        memcpy(
            edge_mu_grad_before,
            edge_mu_grad,
            sizeof(edge_mu_grad)
        );
        memcpy(
            edge_logvar_data_before,
            edge_logvar_data,
            sizeof(edge_logvar_data)
        );
        memcpy(
            edge_logvar_grad_before,
            edge_logvar_grad,
            sizeof(edge_logvar_grad)
        );
        memcpy(
            edge_epsilon_data_before,
            edge_epsilon_data,
            sizeof(edge_epsilon_data)
        );
        memcpy(
            edge_epsilon_grad_before,
            edge_epsilon_grad,
            sizeof(edge_epsilon_grad)
        );
        memcpy(
            edge_z_data_before,
            edge_z_data,
            sizeof(edge_z_data)
        );
        memcpy(
            edge_z_grad_before,
            edge_z_grad,
            sizeof(edge_z_grad)
        );

        if (vae_c_reparameterize_forward(
                VAE_C_LATENT_MODE_STOCHASTIC,
                &edge_mu,
                &edge_logvar,
                &edge_epsilon,
                &edge_z) != VAE_C_OP_INVALID_STATE ||
            !arrays_bitwise_equal(
                edge_mu_data,
                edge_mu_data_before,
                1) ||
            !arrays_bitwise_equal(
                edge_mu_grad,
                edge_mu_grad_before,
                1) ||
            !arrays_bitwise_equal(
                edge_logvar_data,
                edge_logvar_data_before,
                1) ||
            !arrays_bitwise_equal(
                edge_logvar_grad,
                edge_logvar_grad_before,
                1) ||
            !arrays_bitwise_equal(
                edge_epsilon_data,
                edge_epsilon_data_before,
                1) ||
            !arrays_bitwise_equal(
                edge_epsilon_grad,
                edge_epsilon_grad_before,
                1) ||
            !arrays_bitwise_equal(
                edge_z_data,
                edge_z_data_before,
                1) ||
            !arrays_bitwise_equal(
                edge_z_grad,
                edge_z_grad_before,
                1)) {

            return 0;
        }
    }

    memcpy(mu_before, mu_data, sizeof(mu_data));
    memcpy(logvar_before, logvar_data, sizeof(logvar_data));
    packed_data[0] = NAN;
    if (!split_forward_rejects_preserving_outputs(
            &packed, &mu, &logvar, mu_before, logvar_before)) {

        return 0;
    }
    packed_data[0] = INFINITY;
    if (!split_forward_rejects_preserving_outputs(
            &packed, &mu, &logvar, mu_before, logvar_before)) {

        return 0;
    }
    packed_data[0] = 0.1f;

    memcpy(packed_grad_before, packed_grad, sizeof(packed_grad));
    mu_grad[0] = NAN;
    if (vae_c_latent_split_backward(&packed, &mu, &logvar) ==
            VAE_C_OP_OK ||
        !arrays_bitwise_equal(
            packed_grad,
            packed_grad_before,
            4)) {

        return 0;
    }
    mu_grad[0] = 0.0f;
    logvar_grad[0] = INFINITY;
    if (vae_c_latent_split_backward(&packed, &mu, &logvar) ==
            VAE_C_OP_OK ||
        !arrays_bitwise_equal(
            packed_grad,
            packed_grad_before,
            4)) {

        return 0;
    }
    logvar_grad[0] = 0.0f;
    packed_grad[0] = FLT_MAX;
    mu_grad[0] = FLT_MAX;
    memcpy(packed_grad_before, packed_grad, sizeof(packed_grad));
    if (vae_c_latent_split_backward(&packed, &mu, &logvar) ==
            VAE_C_OP_OK ||
        !arrays_bitwise_equal(
            packed_grad,
            packed_grad_before,
            4)) {

        return 0;
    }
    packed_grad[0] = 0.0f;
    mu_grad[0] = 0.0f;

    memcpy(z_before, z_data, sizeof(z_data));
    mu_data[0] = NAN;
    if (!reparameterize_forward_rejects_preserving_z(
            VAE_C_LATENT_MODE_STOCHASTIC,
            &mu, &logvar, &epsilon, &z, z_before)) {
        return 0;
    }
    mu_data[0] = INFINITY;
    if (!reparameterize_forward_rejects_preserving_z(
            VAE_C_LATENT_MODE_STOCHASTIC,
            &mu, &logvar, &epsilon, &z, z_before)) {
        return 0;
    }
    mu_data[0] = 0.1f;
    logvar_data[0] = NAN;
    if (!reparameterize_forward_rejects_preserving_z(
            VAE_C_LATENT_MODE_STOCHASTIC,
            &mu, &logvar, &epsilon, &z, z_before)) {
        return 0;
    }
    logvar_data[0] = INFINITY;
    if (!reparameterize_forward_rejects_preserving_z(
            VAE_C_LATENT_MODE_STOCHASTIC,
            &mu, &logvar, &epsilon, &z, z_before)) {
        return 0;
    }
    logvar_data[0] = 200.0f;
    if (!reparameterize_forward_rejects_preserving_z(
            VAE_C_LATENT_MODE_STOCHASTIC,
            &mu, &logvar, &epsilon, &z, z_before)) {
        return 0;
    }
    logvar_data[0] = 0.2f;
    epsilon_data[0] = NAN;
    if (!reparameterize_forward_rejects_preserving_z(
            VAE_C_LATENT_MODE_STOCHASTIC,
            &mu, &logvar, &epsilon, &z, z_before)) {
        return 0;
    }
    epsilon_data[0] = INFINITY;
    if (!reparameterize_forward_rejects_preserving_z(
            VAE_C_LATENT_MODE_STOCHASTIC,
            &mu, &logvar, &epsilon, &z, z_before)) {
        return 0;
    }
    mu_data[0] = FLT_MAX;
    logvar_data[0] = 0.0f;
    epsilon_data[0] = FLT_MAX;
    if (!reparameterize_forward_rejects_preserving_z(
            VAE_C_LATENT_MODE_STOCHASTIC,
            &mu, &logvar, &epsilon, &z, z_before)) {
        return 0;
    }

    mu_data[0] = 0.1f;
    logvar_data[0] = 0.2f;
    epsilon_data[0] = 0.4f;
    logvar_data[0] = NAN;
    if (!reparameterize_forward_rejects_preserving_z(
            VAE_C_LATENT_MODE_MEAN,
            &mu, &logvar, NULL, &z, z_before)) {
        return 0;
    }
    logvar_data[0] = 0.2f;
    memcpy(mu_grad_before, mu_grad, sizeof(mu_grad));
    memcpy(logvar_grad_before, logvar_grad, sizeof(logvar_grad));
    z_grad[0] = NAN;
    if (vae_c_reparameterize_backward(
            VAE_C_LATENT_MODE_STOCHASTIC,
            &mu, &logvar, &epsilon, &z) == VAE_C_OP_OK ||
        !arrays_bitwise_equal(mu_grad, mu_grad_before, 2) ||
        !arrays_bitwise_equal(logvar_grad, logvar_grad_before, 2)) {
        return 0;
    }
    z_grad[0] = INFINITY;
    if (vae_c_reparameterize_backward(
            VAE_C_LATENT_MODE_STOCHASTIC,
            &mu, &logvar, &epsilon, &z) == VAE_C_OP_OK ||
        !arrays_bitwise_equal(mu_grad, mu_grad_before, 2) ||
        !arrays_bitwise_equal(logvar_grad, logvar_grad_before, 2)) {
        return 0;
    }
    z_grad[0] = FLT_MAX;
    mu_grad[0] = FLT_MAX;
    memcpy(mu_grad_before, mu_grad, sizeof(mu_grad));
    if (vae_c_reparameterize_backward(
            VAE_C_LATENT_MODE_STOCHASTIC,
            &mu, &logvar, &epsilon, &z) == VAE_C_OP_OK ||
        !arrays_bitwise_equal(mu_grad, mu_grad_before, 2) ||
        !arrays_bitwise_equal(logvar_grad, logvar_grad_before, 2)) {
        return 0;
    }
    mu_grad[0] = 0.0f;
    z_grad[0] = 1.0f;
    epsilon_data[0] = FLT_MAX;
    logvar_grad[0] = FLT_MAX;
    memcpy(mu_grad_before, mu_grad, sizeof(mu_grad));
    memcpy(logvar_grad_before, logvar_grad, sizeof(logvar_grad));

    return
        vae_c_reparameterize_backward(
            VAE_C_LATENT_MODE_STOCHASTIC,
            &mu, &logvar, &epsilon, &z) != VAE_C_OP_OK &&
        arrays_bitwise_equal(mu_grad, mu_grad_before, 2) &&
        arrays_bitwise_equal(logvar_grad, logvar_grad_before, 2);
}


static int test_split_negative_contracts(void)
{
    float packed_data[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    float packed_grad[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float mu_data[2] = { 8.0f, 9.0f };
    float mu_grad[2] = { 0.1f, 0.2f };
    float logvar_data[2] = { 10.0f, 11.0f };
    float logvar_grad[2] = { 0.3f, 0.4f };
    float extra_data[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float mu_before[2];
    float logvar_before[2];
    float packed_grad_before[4];
    float huge_mu_data;
    float huge_logvar_data;
    float small_packed_data;
    VAECTensor packed;
    VAECTensor mu;
    VAECTensor logvar;
    VAECTensor other;
    VAECTensor huge_mu;
    VAECTensor huge_logvar;
    VAECTensor small_packed;
    size_t huge_latent = SIZE_MAX / 2 + 1;

    if (!make_view_2d(&packed, packed_data, packed_grad, 1, 4) ||
        !make_view_2d(&mu, mu_data, mu_grad, 1, 2) ||
        !make_view_2d(&logvar, logvar_data, logvar_grad, 1, 2) ||
        !make_view_2d(&other, extra_data, NULL, 2, 2)) {
        return 0;
    }

    memcpy(mu_before, mu_data, sizeof(mu_data));
    memcpy(logvar_before, logvar_data, sizeof(logvar_data));
    memcpy(packed_grad_before, packed_grad, sizeof(packed_grad));

    if (vae_c_latent_split_forward(NULL, &mu, &logvar) == VAE_C_OP_OK ||
        vae_c_latent_split_forward(&packed, NULL, &logvar) == VAE_C_OP_OK ||
        vae_c_latent_split_forward(&packed, &mu, NULL) == VAE_C_OP_OK ||
        vae_c_latent_split_forward(&packed, &packed, &logvar) ==
            VAE_C_OP_OK ||
        vae_c_latent_split_forward(&packed, &other, &logvar) ==
            VAE_C_OP_OK ||
        !arrays_bitwise_equal(mu_data, mu_before, 2) ||
        !arrays_bitwise_equal(logvar_data, logvar_before, 2)) {
        return 0;
    }

    other = mu;
    other.shape.dims[0] = 2;
    other.shape.dims[1] = 1;
    other.shape.strides[0] = 1;
    other.shape.count = 2;
    if (vae_c_latent_split_forward(&packed, &other, &logvar) ==
        VAE_C_OP_OK) {
        return 0;
    }

    other = mu;
    other.shape.rank = 1;
    other.shape.dims[0] = 2;
    other.shape.dims[1] = 0;
    other.shape.strides[0] = 1;
    other.shape.strides[1] = 0;
    other.shape.count = 2;
    if (vae_c_latent_split_forward(&packed, &other, &logvar) ==
        VAE_C_OP_OK) {
        return 0;
    }

    other = mu;
    other.shape.strides[1] = 2;
    if (vae_c_latent_split_forward(&packed, &other, &logvar) ==
        VAE_C_OP_OK) {
        return 0;
    }

    other = packed;
    other.shape.dims[1] = 3;
    other.shape.strides[0] = 3;
    other.shape.count = 3;
    if (vae_c_latent_split_forward(&other, &mu, &logvar) ==
        VAE_C_OP_OK) {
        return 0;
    }

    other = logvar;
    other.data = mu.data;
    if (vae_c_latent_split_forward(&packed, &mu, &other) ==
        VAE_C_OP_OK) {
        return 0;
    }

    other = mu;
    other.grad = NULL;
    other.has_grad = false;
    if (vae_c_latent_split_backward(&packed, &other, &logvar) ==
            VAE_C_OP_OK ||
        !arrays_bitwise_equal(packed_grad, packed_grad_before, 4)) {
        return 0;
    }

    other = packed;
    other.grad = NULL;
    other.has_grad = false;
    if (vae_c_latent_split_backward(&other, &mu, &logvar) ==
            VAE_C_OP_OK ||
        !arrays_bitwise_equal(packed_grad, packed_grad_before, 4)) {
        return 0;
    }

    other = logvar;
    other.grad = NULL;
    other.has_grad = false;
    if (vae_c_latent_split_backward(&packed, &mu, &other) ==
            VAE_C_OP_OK ||
        !arrays_bitwise_equal(packed_grad, packed_grad_before, 4)) {
        return 0;
    }

    other = logvar;
    other.shape.dims[1] = 1;
    other.shape.strides[0] = 1;
    other.shape.count = 1;
    if (vae_c_latent_split_forward(&packed, &mu, &other) ==
        VAE_C_OP_OK) {
        return 0;
    }

    if (!make_view_2d(
            &small_packed, &small_packed_data, NULL, 1, 1) ||
        !make_view_2d(
            &huge_mu, &huge_mu_data, NULL, 1, huge_latent) ||
        !make_view_2d(
            &huge_logvar, &huge_logvar_data, NULL, 1, huge_latent)) {
        return 0;
    }

    return
        vae_c_latent_split_forward(
            &small_packed,
            &huge_mu,
            &huge_logvar) == VAE_C_OP_OVERFLOW &&
        arrays_bitwise_equal(mu_data, mu_before, 2) &&
        arrays_bitwise_equal(logvar_data, logvar_before, 2) &&
        arrays_bitwise_equal(packed_grad, packed_grad_before, 4);
}


static int test_reparameterization_negative_contracts(void)
{
    float mu_data[2] = { 0.1f, -0.2f };
    float mu_grad[2] = { 0.0f, 0.0f };
    float logvar_data[2] = { 0.3f, -0.4f };
    float logvar_grad[2] = { 0.0f, 0.0f };
    float epsilon_data[2] = { 0.5f, -0.6f };
    float z_data[2] = { 13.0f, 14.0f };
    float z_grad[2] = { 0.7f, -0.8f };
    float other_data[3] = { 0.0f, 0.0f, 0.0f };
    float z_before[2];
    float mu_grad_before[2];
    float logvar_grad_before[2];
    VAECTensor mu;
    VAECTensor logvar;
    VAECTensor epsilon;
    VAECTensor z;
    VAECTensor other;

    if (!make_view_2d(&mu, mu_data, mu_grad, 1, 2) ||
        !make_view_2d(&logvar, logvar_data, logvar_grad, 1, 2) ||
        !make_view_2d(&epsilon, epsilon_data, NULL, 1, 2) ||
        !make_view_2d(&z, z_data, z_grad, 1, 2) ||
        !make_view_2d(&other, other_data, NULL, 1, 3)) {
        return 0;
    }

    memcpy(z_before, z_data, sizeof(z_data));
    memcpy(mu_grad_before, mu_grad, sizeof(mu_grad));
    memcpy(logvar_grad_before, logvar_grad, sizeof(logvar_grad));

    if (vae_c_reparameterize_forward(
            (VAECLatentMode)0,
            &mu, &logvar, &epsilon, &z) == VAE_C_OP_OK ||
        vae_c_reparameterize_forward(
            VAE_C_LATENT_MODE_STOCHASTIC,
            NULL, &logvar, &epsilon, &z) == VAE_C_OP_OK ||
        vae_c_reparameterize_forward(
            VAE_C_LATENT_MODE_STOCHASTIC,
            &mu, NULL, &epsilon, &z) == VAE_C_OP_OK ||
        vae_c_reparameterize_forward(
            VAE_C_LATENT_MODE_STOCHASTIC,
            &mu, &logvar, &epsilon, NULL) == VAE_C_OP_OK ||
        vae_c_reparameterize_forward(
            VAE_C_LATENT_MODE_STOCHASTIC,
            &mu, &logvar, NULL, &z) == VAE_C_OP_OK ||
        vae_c_reparameterize_forward(
            VAE_C_LATENT_MODE_MEAN,
            &mu, &logvar, &epsilon, &z) == VAE_C_OP_OK ||
        vae_c_reparameterize_forward(
            VAE_C_LATENT_MODE_STOCHASTIC,
            &mu, &logvar, &other, &z) == VAE_C_OP_OK ||
        vae_c_reparameterize_forward(
            VAE_C_LATENT_MODE_STOCHASTIC,
            &mu, &logvar, &epsilon, &mu) == VAE_C_OP_OK ||
        !arrays_bitwise_equal(z_data, z_before, 2)) {
        return 0;
    }

    other = mu;
    other.shape.rank = 1;
    other.shape.dims[0] = 2;
    other.shape.dims[1] = 0;
    other.shape.strides[0] = 1;
    other.shape.strides[1] = 0;
    other.shape.count = 2;
    if (vae_c_reparameterize_forward(
            VAE_C_LATENT_MODE_MEAN,
            &other, &logvar, NULL, &z) == VAE_C_OP_OK) {
        return 0;
    }

    other = logvar;
    other.data = mu.data;
    if (vae_c_reparameterize_forward(
            VAE_C_LATENT_MODE_MEAN,
            &mu, &other, NULL, &z) == VAE_C_OP_OK) {
        return 0;
    }

    other = mu;
    other.grad = NULL;
    other.has_grad = false;
    if (vae_c_reparameterize_backward(
            VAE_C_LATENT_MODE_MEAN,
            &other, &logvar, NULL, &z) == VAE_C_OP_OK ||
        !arrays_bitwise_equal(mu_grad, mu_grad_before, 2) ||
        !arrays_bitwise_equal(logvar_grad, logvar_grad_before, 2)) {
        return 0;
    }

    other = logvar;
    other.grad = NULL;
    other.has_grad = false;
    if (vae_c_reparameterize_backward(
            VAE_C_LATENT_MODE_MEAN,
            &mu, &other, NULL, &z) == VAE_C_OP_OK ||
        !arrays_bitwise_equal(mu_grad, mu_grad_before, 2) ||
        !arrays_bitwise_equal(logvar_grad, logvar_grad_before, 2)) {
        return 0;
    }

    other = z;
    other.grad = NULL;
    other.has_grad = false;
    if (vae_c_reparameterize_backward(
            VAE_C_LATENT_MODE_MEAN,
            &mu, &logvar, NULL, &other) == VAE_C_OP_OK ||
        !arrays_bitwise_equal(mu_grad, mu_grad_before, 2) ||
        !arrays_bitwise_equal(logvar_grad, logvar_grad_before, 2)) {
        return 0;
    }

    other = epsilon;
    other.data = z.grad;
    return
        vae_c_reparameterize_backward(
            VAE_C_LATENT_MODE_STOCHASTIC,
            &mu, &logvar, &other, &z) != VAE_C_OP_OK &&
        arrays_bitwise_equal(mu_grad, mu_grad_before, 2) &&
        arrays_bitwise_equal(logvar_grad, logvar_grad_before, 2) &&
        arrays_bitwise_equal(z_data, z_before, 2);
}


int main(void)
{
    int failures = 0;

    failures += report_check(
        "latent_split_analytic_layout_and_backward",
        test_split_analytic_layout_and_backward()
    );
    failures += report_check(
        "reparameterization_stochastic_analytic",
        test_stochastic_analytic()
    );
    failures += report_check(
        "reparameterization_mean_analytic",
        test_mean_analytic()
    );
    failures += report_check(
        "latent_rng_integration",
        test_rng_integration()
    );
    failures += report_check(
        "latent_split_gradient_check",
        test_split_gradient_check()
    );
    failures += report_check(
        "reparameterization_gradient_check",
        test_reparameterization_gradient_check()
    );
    failures += report_check(
        "latent_nonfinite_and_strong_guarantee",
        test_nonfinite_and_strong_guarantee()
    );
    failures += report_check(
        "latent_split_negative_contracts",
        test_split_negative_contracts()
    );
    failures += report_check(
        "reparameterization_negative_contracts",
        test_reparameterization_negative_contracts()
    );

    if (failures != 0) {
        fprintf(stderr, "%d VAE C LATENT TEST(S) FAILED\n", failures);
        return 1;
    }

    printf("ALL VAE C LATENT TESTS PASSED\n");
    return 0;
}
