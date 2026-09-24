#include "vae_c_objective.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>


enum {
    RECON_VALUES = 8,
    LATENT_VALUES = 6,
    STORAGE_VALUES = 8
};


typedef struct {
    float logits_data[STORAGE_VALUES];
    float logits_grad[STORAGE_VALUES];
    float targets_data[STORAGE_VALUES];
    float targets_grad[STORAGE_VALUES];
    float mu_data[STORAGE_VALUES];
    float mu_grad[STORAGE_VALUES];
    float logvar_data[STORAGE_VALUES];
    float logvar_grad[STORAGE_VALUES];

    VAECTensor logits;
    VAECTensor targets;
    VAECTensor mu;
    VAECTensor logvar;
} VAEFixture;


typedef struct {
    float logits_data[STORAGE_VALUES];
    float logits_grad[STORAGE_VALUES];
    float targets_data[STORAGE_VALUES];
    float targets_grad[STORAGE_VALUES];
    float mu_data[STORAGE_VALUES];
    float mu_grad[STORAGE_VALUES];
    float logvar_data[STORAGE_VALUES];
    float logvar_grad[STORAGE_VALUES];
} VAEFixtureSnapshot;


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
    if (vae_c_shape_make(4, dims, &tensor->shape) != VAE_C_STORAGE_OK) {
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


static int make_view_2d(
    VAECTensor *tensor,
    float *data,
    float *grad,
    size_t batch,
    size_t latent_dim
)
{
    size_t dims[2];

    if (tensor == NULL || data == NULL) {
        return 0;
    }

    dims[0] = batch;
    dims[1] = latent_dim;
    vae_c_tensor_init(tensor);
    if (vae_c_shape_make(2, dims, &tensor->shape) != VAE_C_STORAGE_OK) {
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


static int all_zero(const float *values, size_t count)
{
    size_t index;

    for (index = 0; index < count; index++) {
        if (values[index] != 0.0f) {
            return 0;
        }
    }

    return 1;
}


static int doubles_close(double left, double right, double tolerance)
{
    double scale = fmax(fmax(fabs(left), fabs(right)), 1.0);

    return
        isfinite(left) && isfinite(right) &&
        fabs(left - right) <= tolerance * scale;
}


static int bytes_all_zero(const void *object, size_t size)
{
    const unsigned char *bytes = (const unsigned char *)object;
    size_t index;

    for (index = 0; index < size; index++) {
        if (bytes[index] != 0) {
            return 0;
        }
    }

    return 1;
}


static int fixture_init(VAEFixture *fixture)
{
    const float logits_values[RECON_VALUES] = {
        -1.2f, -0.7f, -0.2f, 0.3f,
        0.6f, 1.1f, 1.7f, -1.8f
    };
    const float targets_values[RECON_VALUES] = {
        0.0f, 0.25f, 1.0f, 0.4f,
        0.75f, 0.0f, 1.0f, 0.6f
    };
    const float mu_values[LATENT_VALUES] = {
        -1.1f, -0.4f, 0.2f, 0.7f, 1.3f, -0.8f
    };
    const float logvar_values[LATENT_VALUES] = {
        -0.9f, -0.3f, 0.1f, 0.4f, 0.8f, -0.6f
    };
    size_t index;

    if (fixture == NULL) {
        return 0;
    }

    memset(fixture, 0, sizeof(*fixture));
    memcpy(fixture->logits_data, logits_values, sizeof(logits_values));
    memcpy(fixture->targets_data, targets_values, sizeof(targets_values));
    memcpy(fixture->mu_data, mu_values, sizeof(mu_values));
    memcpy(fixture->logvar_data, logvar_values, sizeof(logvar_values));

    for (index = 0; index < STORAGE_VALUES; index++) {
        fixture->logits_grad[index] = 0.01f * (float)(index + 1);
        fixture->targets_grad[index] = -0.02f * (float)(index + 1);
        fixture->mu_grad[index] = 0.03f * (float)(index + 1);
        fixture->logvar_grad[index] = -0.04f * (float)(index + 1);
    }

    return
        make_view_4d(
            &fixture->logits,
            fixture->logits_data,
            fixture->logits_grad,
            2, 1, 2, 2) &&
        make_view_4d(
            &fixture->targets,
            fixture->targets_data,
            fixture->targets_grad,
            2, 1, 2, 2) &&
        make_view_2d(
            &fixture->mu,
            fixture->mu_data,
            fixture->mu_grad,
            2, 3) &&
        make_view_2d(
            &fixture->logvar,
            fixture->logvar_data,
            fixture->logvar_grad,
            2, 3);
}


static void fixture_snapshot(
    const VAEFixture *fixture,
    VAEFixtureSnapshot *snapshot
)
{
    memcpy(snapshot->logits_data, fixture->logits_data,
        sizeof(snapshot->logits_data));
    memcpy(snapshot->logits_grad, fixture->logits_grad,
        sizeof(snapshot->logits_grad));
    memcpy(snapshot->targets_data, fixture->targets_data,
        sizeof(snapshot->targets_data));
    memcpy(snapshot->targets_grad, fixture->targets_grad,
        sizeof(snapshot->targets_grad));
    memcpy(snapshot->mu_data, fixture->mu_data,
        sizeof(snapshot->mu_data));
    memcpy(snapshot->mu_grad, fixture->mu_grad,
        sizeof(snapshot->mu_grad));
    memcpy(snapshot->logvar_data, fixture->logvar_data,
        sizeof(snapshot->logvar_data));
    memcpy(snapshot->logvar_grad, fixture->logvar_grad,
        sizeof(snapshot->logvar_grad));
}


static int fixture_matches_snapshot(
    const VAEFixture *fixture,
    const VAEFixtureSnapshot *snapshot
)
{
    return
        arrays_bitwise_equal(
            fixture->logits_data, snapshot->logits_data, STORAGE_VALUES) &&
        arrays_bitwise_equal(
            fixture->logits_grad, snapshot->logits_grad, STORAGE_VALUES) &&
        arrays_bitwise_equal(
            fixture->targets_data, snapshot->targets_data, STORAGE_VALUES) &&
        arrays_bitwise_equal(
            fixture->targets_grad, snapshot->targets_grad, STORAGE_VALUES) &&
        arrays_bitwise_equal(
            fixture->mu_data, snapshot->mu_data, STORAGE_VALUES) &&
        arrays_bitwise_equal(
            fixture->mu_grad, snapshot->mu_grad, STORAGE_VALUES) &&
        arrays_bitwise_equal(
            fixture->logvar_data, snapshot->logvar_data, STORAGE_VALUES) &&
        arrays_bitwise_equal(
            fixture->logvar_grad, snapshot->logvar_grad, STORAGE_VALUES);
}


static int fixture_data_matches_snapshot(
    const VAEFixture *fixture,
    const VAEFixtureSnapshot *snapshot
)
{
    return
        arrays_bitwise_equal(
            fixture->logits_data, snapshot->logits_data, STORAGE_VALUES) &&
        arrays_bitwise_equal(
            fixture->targets_data, snapshot->targets_data, STORAGE_VALUES) &&
        arrays_bitwise_equal(
            fixture->mu_data, snapshot->mu_data, STORAGE_VALUES) &&
        arrays_bitwise_equal(
            fixture->logvar_data, snapshot->logvar_data, STORAGE_VALUES);
}


static void set_stats_sentinel(VAECVAELossStats *stats)
{
    memset(stats, 0xA5, sizeof(*stats));
}


static int forward_fails_preserving_all(
    VAEFixture *fixture,
    const VAECTensor *logits,
    const VAECTensor *targets,
    const VAECTensor *mu,
    const VAECTensor *logvar
)
{
    VAEFixtureSnapshot before;
    VAECVAELossStats stats;
    VAECVAELossStats stats_before;

    fixture_snapshot(fixture, &before);
    set_stats_sentinel(&stats);
    memcpy(&stats_before, &stats, sizeof(stats));

    return
        vae_c_vae_loss_forward(
            logits, targets, mu, logvar, &stats) != VAE_C_OP_OK &&
        memcmp(&stats, &stats_before, sizeof(stats)) == 0 &&
        fixture_matches_snapshot(fixture, &before);
}


static int backward_fails_preserving_all(
    VAEFixture *fixture,
    VAECTensor *logits,
    const VAECTensor *targets,
    VAECTensor *mu,
    VAECTensor *logvar,
    float upstream_gradient
)
{
    VAEFixtureSnapshot before;

    fixture_snapshot(fixture, &before);
    return
        vae_c_vae_loss_backward(
            logits,
            targets,
            mu,
            logvar,
            upstream_gradient) != VAE_C_OP_OK &&
        fixture_matches_snapshot(fixture, &before);
}


static int test_stats_and_forward_equivalence(void)
{
    VAEFixture fixture;
    VAEFixtureSnapshot before;
    VAECBCEWithLogitsStats reconstruction;
    VAECKLGaussianStats kl;
    VAECVAELossStats combined;
    double expected_total;
    double expected_mean;

    if (!fixture_init(&fixture)) {
        return 0;
    }

    memset(&combined, 0xA5, sizeof(combined));
    vae_c_vae_loss_stats_init(&combined);
    if (!bytes_all_zero(&combined, sizeof(combined)) ||
        VAE_C_VAE_LOSS_BETA_KL_V1 != 1.0f) {

        return 0;
    }

    fixture_snapshot(&fixture, &before);
    if (vae_c_bce_with_logits_forward(
            &fixture.logits,
            &fixture.targets,
            &reconstruction) != VAE_C_OP_OK ||
        vae_c_kl_gaussian_forward(
            &fixture.mu,
            &fixture.logvar,
            &kl) != VAE_C_OP_OK ||
        vae_c_vae_loss_forward(
            &fixture.logits,
            &fixture.targets,
            &fixture.mu,
            &fixture.logvar,
            &combined) != VAE_C_OP_OK) {

        return 0;
    }

    expected_total = reconstruction.total_sum + kl.total_sum;
    expected_mean =
        reconstruction.mean_per_sample + kl.mean_per_sample;

    return
        combined.batch_size == 2 &&
        combined.beta_kl == 1.0f &&
        memcmp(
            &combined.reconstruction,
            &reconstruction,
            sizeof(reconstruction)) == 0 &&
        memcmp(&combined.kl, &kl, sizeof(kl)) == 0 &&
        combined.total_sum == expected_total &&
        doubles_close(combined.mean_per_sample, expected_mean, 2.0e-15) &&
        combined.reconstruction.mean_per_value ==
            reconstruction.mean_per_value &&
        combined.kl.mean_per_dimension == kl.mean_per_dimension &&
        !doubles_close(
            combined.mean_per_sample,
            combined.reconstruction.mean_per_value +
                combined.kl.mean_per_dimension,
            1.0e-12) &&
        fixture_matches_snapshot(&fixture, &before);
}


static int test_batch_aggregation(void)
{
    VAEFixture regular;
    float last_logits_data[4] = { 3.0f, -2.0f, 1.5f, -1.0f };
    float last_targets_data[4] = { 0.0f, 1.0f, 0.25f, 0.8f };
    float last_mu_data[3] = { 1.5f, -0.75f, 0.4f };
    float last_logvar_data[3] = { 0.6f, -0.8f, 0.2f };
    float last_logits_grad[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float last_mu_grad[3] = { 0.0f, 0.0f, 0.0f };
    float last_logvar_grad[3] = { 0.0f, 0.0f, 0.0f };
    VAECTensor last_logits;
    VAECTensor last_targets;
    VAECTensor last_mu;
    VAECTensor last_logvar;
    VAECVAELossStats regular_stats;
    VAECVAELossStats last_stats;
    double aggregate_mean;
    double unweighted_mean;

    if (!fixture_init(&regular) ||
        !make_view_4d(
            &last_logits,
            last_logits_data,
            last_logits_grad,
            1, 1, 2, 2) ||
        !make_view_4d(
            &last_targets,
            last_targets_data,
            NULL,
            1, 1, 2, 2) ||
        !make_view_2d(
            &last_mu,
            last_mu_data,
            last_mu_grad,
            1, 3) ||
        !make_view_2d(
            &last_logvar,
            last_logvar_data,
            last_logvar_grad,
            1, 3) ||
        vae_c_vae_loss_forward(
            &regular.logits,
            &regular.targets,
            &regular.mu,
            &regular.logvar,
            &regular_stats) != VAE_C_OP_OK ||
        vae_c_vae_loss_forward(
            &last_logits,
            &last_targets,
            &last_mu,
            &last_logvar,
            &last_stats) != VAE_C_OP_OK) {

        return 0;
    }

    aggregate_mean =
        (regular_stats.total_sum + last_stats.total_sum) /
        (double)(regular_stats.batch_size + last_stats.batch_size);
    unweighted_mean =
        (regular_stats.mean_per_sample + last_stats.mean_per_sample) /
        2.0;

    return
        regular_stats.batch_size == 2 &&
        last_stats.batch_size == 1 &&
        doubles_close(
            regular_stats.mean_per_sample,
            regular_stats.total_sum / 2.0,
            0.0) &&
        doubles_close(
            last_stats.mean_per_sample,
            last_stats.total_sum,
            0.0) &&
        isfinite(aggregate_mean) &&
        fabs(aggregate_mean - unweighted_mean) > 1.0e-6;
}


static int test_backward_equivalence(void)
{
    VAEFixture separate;
    VAEFixture combined;
    VAEFixtureSnapshot combined_before;
    float first_logits_grad[RECON_VALUES];
    float first_mu_grad[LATENT_VALUES];
    float first_logvar_grad[LATENT_VALUES];
    const float upstream = 0.7f;
    size_t index;

    if (!fixture_init(&separate) || !fixture_init(&combined)) {
        return 0;
    }

    fixture_snapshot(&combined, &combined_before);
    if (vae_c_bce_with_logits_backward(
            &separate.logits,
            &separate.targets,
            upstream) != VAE_C_OP_OK ||
        vae_c_kl_gaussian_backward(
            &separate.mu,
            &separate.logvar,
            upstream) != VAE_C_OP_OK ||
        vae_c_vae_loss_backward(
            &combined.logits,
            &combined.targets,
            &combined.mu,
            &combined.logvar,
            upstream) != VAE_C_OP_OK ||
        !arrays_bitwise_equal(
            separate.logits_grad,
            combined.logits_grad,
            RECON_VALUES) ||
        !arrays_bitwise_equal(
            separate.mu_grad,
            combined.mu_grad,
            LATENT_VALUES) ||
        !arrays_bitwise_equal(
            separate.logvar_grad,
            combined.logvar_grad,
            LATENT_VALUES) ||
        !arrays_bitwise_equal(
            combined.targets_grad,
            combined_before.targets_grad,
            STORAGE_VALUES) ||
        !fixture_data_matches_snapshot(&combined, &combined_before)) {

        return 0;
    }

    memcpy(
        first_logits_grad,
        combined.logits_grad,
        sizeof(first_logits_grad)
    );
    memcpy(first_mu_grad, combined.mu_grad, sizeof(first_mu_grad));
    memcpy(
        first_logvar_grad,
        combined.logvar_grad,
        sizeof(first_logvar_grad)
    );

    if (vae_c_bce_with_logits_backward(
            &separate.logits,
            &separate.targets,
            upstream) != VAE_C_OP_OK ||
        vae_c_kl_gaussian_backward(
            &separate.mu,
            &separate.logvar,
            upstream) != VAE_C_OP_OK ||
        vae_c_vae_loss_backward(
            &combined.logits,
            &combined.targets,
            &combined.mu,
            &combined.logvar,
            upstream) != VAE_C_OP_OK ||
        !arrays_bitwise_equal(
            separate.logits_grad,
            combined.logits_grad,
            RECON_VALUES) ||
        !arrays_bitwise_equal(
            separate.mu_grad,
            combined.mu_grad,
            LATENT_VALUES) ||
        !arrays_bitwise_equal(
            separate.logvar_grad,
            combined.logvar_grad,
            LATENT_VALUES) ||
        !arrays_bitwise_equal(
            combined.targets_grad,
            combined_before.targets_grad,
            STORAGE_VALUES) ||
        !fixture_data_matches_snapshot(&combined, &combined_before)) {

        return 0;
    }

    for (index = 0; index < RECON_VALUES; index++) {
        float first_contribution =
            first_logits_grad[index] - combined_before.logits_grad[index];
        float total_contribution =
            combined.logits_grad[index] - combined_before.logits_grad[index];

        if (fabsf(total_contribution - 2.0f * first_contribution) >
            2.0e-7f) {

            return 0;
        }
    }
    for (index = 0; index < LATENT_VALUES; index++) {
        float first_mu_contribution =
            first_mu_grad[index] - combined_before.mu_grad[index];
        float total_mu_contribution =
            combined.mu_grad[index] - combined_before.mu_grad[index];
        float first_logvar_contribution =
            first_logvar_grad[index] - combined_before.logvar_grad[index];
        float total_logvar_contribution =
            combined.logvar_grad[index] - combined_before.logvar_grad[index];

        if (fabsf(
                total_mu_contribution -
                2.0f * first_mu_contribution) > 2.0e-7f ||
            fabsf(
                total_logvar_contribution -
                2.0f * first_logvar_contribution) > 2.0e-7f) {

            return 0;
        }
    }

    if (!fixture_init(&separate) || !fixture_init(&combined)) {
        return 0;
    }
    fixture_snapshot(&combined, &combined_before);

    if (vae_c_bce_with_logits_backward(
            &separate.logits,
            &separate.targets,
            -0.35f) != VAE_C_OP_OK ||
        vae_c_kl_gaussian_backward(
            &separate.mu,
            &separate.logvar,
            -0.35f) != VAE_C_OP_OK ||
        vae_c_vae_loss_backward(
            &combined.logits,
            &combined.targets,
            &combined.mu,
            &combined.logvar,
            -0.35f) != VAE_C_OP_OK ||
        !arrays_bitwise_equal(
            separate.logits_grad,
            combined.logits_grad,
            RECON_VALUES) ||
        !arrays_bitwise_equal(
            separate.mu_grad,
            combined.mu_grad,
            LATENT_VALUES) ||
        !arrays_bitwise_equal(
            separate.logvar_grad,
            combined.logvar_grad,
            LATENT_VALUES) ||
        !arrays_bitwise_equal(
            combined.targets_grad,
            combined_before.targets_grad,
            STORAGE_VALUES) ||
        !fixture_data_matches_snapshot(&combined, &combined_before)) {

        return 0;
    }

    if (!fixture_init(&combined)) {
        return 0;
    }
    fixture_snapshot(&combined, &combined_before);

    return
        vae_c_vae_loss_backward(
            &combined.logits,
            &combined.targets,
            &combined.mu,
            &combined.logvar,
            0.0f) == VAE_C_OP_OK &&
        fixture_matches_snapshot(&combined, &combined_before);
}


static int test_forward_negative_contracts(void)
{
    VAEFixture fixture;
    VAEFixtureSnapshot before;
    VAECVAELossStats stats;
    VAECTensor alias_tensor;
    VAECTensor short_mu;
    VAECTensor short_logvar;

    if (!fixture_init(&fixture) ||
        !forward_fails_preserving_all(
            &fixture,
            NULL,
            &fixture.targets,
            &fixture.mu,
            &fixture.logvar) ||
        !forward_fails_preserving_all(
            &fixture,
            &fixture.logits,
            NULL,
            &fixture.mu,
            &fixture.logvar) ||
        !forward_fails_preserving_all(
            &fixture,
            &fixture.logits,
            &fixture.targets,
            NULL,
            &fixture.logvar) ||
        !forward_fails_preserving_all(
            &fixture,
            &fixture.logits,
            &fixture.targets,
            &fixture.mu,
            NULL) ||
        !forward_fails_preserving_all(
            &fixture,
            &fixture.logits,
            &fixture.logits,
            &fixture.mu,
            &fixture.logvar)) {

        return 0;
    }

    fixture_snapshot(&fixture, &before);
    if (vae_c_vae_loss_forward(
            &fixture.logits,
            &fixture.targets,
            &fixture.mu,
            &fixture.logvar,
            NULL) != VAE_C_OP_INVALID_ARGUMENT ||
        !fixture_matches_snapshot(&fixture, &before)) {

        return 0;
    }

    alias_tensor = fixture.mu;
    alias_tensor.grad = fixture.logits.grad;
    if (!forward_fails_preserving_all(
            &fixture,
            &fixture.logits,
            &fixture.targets,
            &alias_tensor,
            &fixture.logvar)) {

        return 0;
    }

    alias_tensor = fixture.targets;
    alias_tensor.data = fixture.logvar.data;
    if (!forward_fails_preserving_all(
            &fixture,
            &fixture.logits,
            &alias_tensor,
            &fixture.mu,
            &fixture.logvar)) {

        return 0;
    }

    if (!make_view_2d(
            &short_mu,
            fixture.mu_data,
            fixture.mu_grad,
            1,
            3) ||
        !make_view_2d(
            &short_logvar,
            fixture.logvar_data,
            fixture.logvar_grad,
            1,
            3) ||
        !forward_fails_preserving_all(
            &fixture,
            &fixture.logits,
            &fixture.targets,
            &short_mu,
            &short_logvar)) {

        return 0;
    }

    if (!fixture_init(&fixture)) {
        return 0;
    }
    fixture.targets_data[0] = -0.1f;
    if (!forward_fails_preserving_all(
            &fixture,
            &fixture.logits,
            &fixture.targets,
            &fixture.mu,
            &fixture.logvar)) {

        return 0;
    }

    if (!fixture_init(&fixture)) {
        return 0;
    }
    fixture.logvar_data[0] = NAN;
    if (!forward_fails_preserving_all(
            &fixture,
            &fixture.logits,
            &fixture.targets,
            &fixture.mu,
            &fixture.logvar)) {

        return 0;
    }

    if (!fixture_init(&fixture)) {
        return 0;
    }
    fixture_snapshot(&fixture, &before);
    set_stats_sentinel(&stats);

    return
        vae_c_vae_loss_forward(
            &fixture.logits,
            &fixture.targets,
            &fixture.mu,
            &fixture.logvar,
            (VAECVAELossStats *)(void *)fixture.logits.data) ==
                VAE_C_OP_INVALID_ARGUMENT &&
        fixture_matches_snapshot(&fixture, &before);
}


static int test_backward_strong_guarantee(void)
{
    VAEFixture fixture;
    VAEFixtureSnapshot before;
    VAECTensor altered;
    VAECTensor short_mu;
    VAECTensor short_logvar;
    size_t index;

    if (!fixture_init(&fixture) ||
        !backward_fails_preserving_all(
            &fixture,
            &fixture.logits,
            &fixture.targets,
            &fixture.mu,
            &fixture.logvar,
            NAN) ||
        !backward_fails_preserving_all(
            &fixture,
            &fixture.logits,
            &fixture.targets,
            &fixture.mu,
            &fixture.logvar,
            INFINITY) ||
        !backward_fails_preserving_all(
            &fixture,
            &fixture.logits,
            &fixture.targets,
            &fixture.mu,
            &fixture.logvar,
            -INFINITY)) {

        return 0;
    }

    fixture_snapshot(&fixture, &before);
    if (vae_c_vae_loss_backward(
            NULL,
            &fixture.targets,
            &fixture.mu,
            &fixture.logvar,
            1.0f) == VAE_C_OP_OK ||
        vae_c_vae_loss_backward(
            &fixture.logits,
            NULL,
            &fixture.mu,
            &fixture.logvar,
            1.0f) == VAE_C_OP_OK ||
        vae_c_vae_loss_backward(
            &fixture.logits,
            &fixture.targets,
            NULL,
            &fixture.logvar,
            1.0f) == VAE_C_OP_OK ||
        vae_c_vae_loss_backward(
            &fixture.logits,
            &fixture.targets,
            &fixture.mu,
            NULL,
            1.0f) == VAE_C_OP_OK ||
        !fixture_matches_snapshot(&fixture, &before)) {

        return 0;
    }

    altered = fixture.logits;
    altered.grad = NULL;
    altered.has_grad = false;
    if (!backward_fails_preserving_all(
            &fixture,
            &altered,
            &fixture.targets,
            &fixture.mu,
            &fixture.logvar,
            1.0f)) {

        return 0;
    }

    altered = fixture.mu;
    altered.grad = NULL;
    altered.has_grad = false;
    if (!backward_fails_preserving_all(
            &fixture,
            &fixture.logits,
            &fixture.targets,
            &altered,
            &fixture.logvar,
            1.0f)) {

        return 0;
    }

    altered = fixture.logvar;
    altered.grad = NULL;
    altered.has_grad = false;
    if (!backward_fails_preserving_all(
            &fixture,
            &fixture.logits,
            &fixture.targets,
            &fixture.mu,
            &altered,
            1.0f)) {

        return 0;
    }

    altered = fixture.mu;
    altered.grad = fixture.logits.grad;
    if (!backward_fails_preserving_all(
            &fixture,
            &fixture.logits,
            &fixture.targets,
            &altered,
            &fixture.logvar,
            1.0f)) {

        return 0;
    }

    altered = fixture.targets;
    altered.data = fixture.logvar.data;
    if (!backward_fails_preserving_all(
            &fixture,
            &fixture.logits,
            &altered,
            &fixture.mu,
            &fixture.logvar,
            1.0f)) {

        return 0;
    }

    if (!make_view_2d(
            &short_mu,
            fixture.mu_data,
            fixture.mu_grad,
            1,
            3) ||
        !make_view_2d(
            &short_logvar,
            fixture.logvar_data,
            fixture.logvar_grad,
            1,
            3) ||
        !backward_fails_preserving_all(
            &fixture,
            &fixture.logits,
            &fixture.targets,
            &short_mu,
            &short_logvar,
            1.0f)) {

        return 0;
    }

    if (!fixture_init(&fixture)) {
        return 0;
    }
    fixture.logvar_data[0] = NAN;
    if (!backward_fails_preserving_all(
            &fixture,
            &fixture.logits,
            &fixture.targets,
            &fixture.mu,
            &fixture.logvar,
            1.0f)) {

        return 0;
    }

    if (!fixture_init(&fixture)) {
        return 0;
    }
    fixture.targets_data[0] = 1.1f;
    if (!backward_fails_preserving_all(
            &fixture,
            &fixture.logits,
            &fixture.targets,
            &fixture.mu,
            &fixture.logvar,
            1.0f)) {

        return 0;
    }

    if (!fixture_init(&fixture)) {
        return 0;
    }
    for (index = 0; index < RECON_VALUES; index++) {
        fixture.logits_data[index] = 0.0f;
        fixture.targets_data[index] = 0.5f;
    }
    fixture.mu_data[0] = 1.0f;
    fixture.mu_grad[0] = FLT_MAX;
    if (!backward_fails_preserving_all(
            &fixture,
            &fixture.logits,
            &fixture.targets,
            &fixture.mu,
            &fixture.logvar,
            FLT_MAX)) {

        return 0;
    }

    if (!fixture_init(&fixture)) {
        return 0;
    }
    fixture.logits_data[0] = 0.0f;
    fixture.targets_data[0] = 0.0f;
    fixture.logits_grad[0] = FLT_MAX;

    return backward_fails_preserving_all(
        &fixture,
        &fixture.logits,
        &fixture.targets,
        &fixture.mu,
        &fixture.logvar,
        FLT_MAX);
}


static double gradient_objective(
    const VAECTensor *logits,
    const VAECTensor *targets,
    const VAECTensor *mu,
    const VAECTensor *logvar,
    float upstream_gradient,
    int *ok
)
{
    VAECVAELossStats stats;

    if (vae_c_vae_loss_forward(
            logits,
            targets,
            mu,
            logvar,
            &stats) != VAE_C_OP_OK) {

        *ok = 0;
        return 0.0;
    }

    *ok = 1;
    return (double)upstream_gradient * stats.mean_per_sample;
}


static void update_maximum_error(
    double numeric_gradient,
    double analytic_gradient,
    size_t index,
    double *maximum_absolute,
    double *maximum_relative,
    size_t *worst_index
)
{
    double absolute_error = fabs(numeric_gradient - analytic_gradient);
    double scale = fmax(
        fmax(fabs(numeric_gradient), fabs(analytic_gradient)),
        1.0e-6
    );
    double relative_error = absolute_error / scale;

    if (absolute_error > *maximum_absolute) {
        *maximum_absolute = absolute_error;
        *worst_index = index;
    }
    if (relative_error > *maximum_relative) {
        *maximum_relative = relative_error;
    }
}


static int test_gradient_check(void)
{
    const float epsilon = 1.0e-3f;
    const float upstream_gradient = 0.7f;
    VAEFixture fixture;
    VAEFixtureSnapshot before_forward;
    float analytic_logits[RECON_VALUES];
    float analytic_mu[LATENT_VALUES];
    float analytic_logvar[LATENT_VALUES];
    float targets_grad_before[STORAGE_VALUES];
    VAECVAELossStats stats;
    double logits_maximum_absolute = 0.0;
    double logits_maximum_relative = 0.0;
    double mu_maximum_absolute = 0.0;
    double mu_maximum_relative = 0.0;
    double logvar_maximum_absolute = 0.0;
    double logvar_maximum_relative = 0.0;
    size_t logits_worst_index = 0;
    size_t mu_worst_index = 0;
    size_t logvar_worst_index = 0;
    size_t index;

    if (!fixture_init(&fixture)) {
        return 0;
    }

    fixture_snapshot(&fixture, &before_forward);
    memcpy(
        targets_grad_before,
        fixture.targets_grad,
        sizeof(targets_grad_before)
    );
    if (vae_c_vae_loss_forward(
            &fixture.logits,
            &fixture.targets,
            &fixture.mu,
            &fixture.logvar,
            &stats) != VAE_C_OP_OK ||
        !fixture_matches_snapshot(&fixture, &before_forward) ||
        vae_c_tensor_zero_grad(&fixture.logits) != VAE_C_STORAGE_OK ||
        vae_c_tensor_zero_grad(&fixture.mu) != VAE_C_STORAGE_OK ||
        vae_c_tensor_zero_grad(&fixture.logvar) != VAE_C_STORAGE_OK ||
        !all_zero(fixture.logits_grad, RECON_VALUES) ||
        !all_zero(fixture.mu_grad, LATENT_VALUES) ||
        !all_zero(fixture.logvar_grad, LATENT_VALUES) ||
        vae_c_vae_loss_backward(
            &fixture.logits,
            &fixture.targets,
            &fixture.mu,
            &fixture.logvar,
            upstream_gradient) != VAE_C_OP_OK) {

        return 0;
    }

    memcpy(analytic_logits, fixture.logits_grad, sizeof(analytic_logits));
    memcpy(analytic_mu, fixture.mu_grad, sizeof(analytic_mu));
    memcpy(analytic_logvar, fixture.logvar_grad, sizeof(analytic_logvar));

    for (index = 0; index < RECON_VALUES; index++) {
        float original = fixture.logits_data[index];
        double objective_plus;
        double objective_minus;
        double numeric_gradient;
        int ok;

        fixture.logits_data[index] = original + epsilon;
        objective_plus = gradient_objective(
            &fixture.logits,
            &fixture.targets,
            &fixture.mu,
            &fixture.logvar,
            upstream_gradient,
            &ok
        );
        if (!ok) {
            fixture.logits_data[index] = original;
            return 0;
        }

        fixture.logits_data[index] = original - epsilon;
        objective_minus = gradient_objective(
            &fixture.logits,
            &fixture.targets,
            &fixture.mu,
            &fixture.logvar,
            upstream_gradient,
            &ok
        );
        fixture.logits_data[index] = original;
        if (!ok) {
            return 0;
        }

        numeric_gradient =
            (objective_plus - objective_minus) /
            (2.0 * (double)epsilon);
        update_maximum_error(
            numeric_gradient,
            (double)analytic_logits[index],
            index,
            &logits_maximum_absolute,
            &logits_maximum_relative,
            &logits_worst_index
        );
    }

    for (index = 0; index < LATENT_VALUES; index++) {
        float original = fixture.mu_data[index];
        double objective_plus;
        double objective_minus;
        double numeric_gradient;
        int ok;

        fixture.mu_data[index] = original + epsilon;
        objective_plus = gradient_objective(
            &fixture.logits,
            &fixture.targets,
            &fixture.mu,
            &fixture.logvar,
            upstream_gradient,
            &ok
        );
        if (!ok) {
            fixture.mu_data[index] = original;
            return 0;
        }

        fixture.mu_data[index] = original - epsilon;
        objective_minus = gradient_objective(
            &fixture.logits,
            &fixture.targets,
            &fixture.mu,
            &fixture.logvar,
            upstream_gradient,
            &ok
        );
        fixture.mu_data[index] = original;
        if (!ok) {
            return 0;
        }

        numeric_gradient =
            (objective_plus - objective_minus) /
            (2.0 * (double)epsilon);
        update_maximum_error(
            numeric_gradient,
            (double)analytic_mu[index],
            index,
            &mu_maximum_absolute,
            &mu_maximum_relative,
            &mu_worst_index
        );
    }

    for (index = 0; index < LATENT_VALUES; index++) {
        float original = fixture.logvar_data[index];
        double objective_plus;
        double objective_minus;
        double numeric_gradient;
        int ok;

        fixture.logvar_data[index] = original + epsilon;
        objective_plus = gradient_objective(
            &fixture.logits,
            &fixture.targets,
            &fixture.mu,
            &fixture.logvar,
            upstream_gradient,
            &ok
        );
        if (!ok) {
            fixture.logvar_data[index] = original;
            return 0;
        }

        fixture.logvar_data[index] = original - epsilon;
        objective_minus = gradient_objective(
            &fixture.logits,
            &fixture.targets,
            &fixture.mu,
            &fixture.logvar,
            upstream_gradient,
            &ok
        );
        fixture.logvar_data[index] = original;
        if (!ok) {
            return 0;
        }

        numeric_gradient =
            (objective_plus - objective_minus) /
            (2.0 * (double)epsilon);
        update_maximum_error(
            numeric_gradient,
            (double)analytic_logvar[index],
            index,
            &logvar_maximum_absolute,
            &logvar_maximum_relative,
            &logvar_worst_index
        );
    }

    printf(
        "VAE_LOSS_LOGITS_GRAD_MAX_ABS=%.9g MAX_REL=%.9g "
        "WORST_INDEX=%zu\n",
        logits_maximum_absolute,
        logits_maximum_relative,
        logits_worst_index
    );
    printf(
        "VAE_LOSS_MU_GRAD_MAX_ABS=%.9g MAX_REL=%.9g "
        "WORST_INDEX=%zu\n",
        mu_maximum_absolute,
        mu_maximum_relative,
        mu_worst_index
    );
    printf(
        "VAE_LOSS_LOGVAR_GRAD_MAX_ABS=%.9g MAX_REL=%.9g "
        "WORST_INDEX=%zu\n",
        logvar_maximum_absolute,
        logvar_maximum_relative,
        logvar_worst_index
    );

    return
        arrays_bitwise_equal(
            fixture.targets_grad,
            targets_grad_before,
            STORAGE_VALUES) &&
        logits_maximum_absolute <= 3.0e-4 &&
        logits_maximum_relative <= 3.0e-3 &&
        mu_maximum_absolute <= 3.0e-4 &&
        mu_maximum_relative <= 3.0e-3 &&
        logvar_maximum_absolute <= 3.0e-4 &&
        logvar_maximum_relative <= 3.0e-3;
}


int main(void)
{
    int failures = 0;

    failures += report_check(
        "vae_loss_stats_and_forward_equivalence",
        test_stats_and_forward_equivalence()
    );
    failures += report_check(
        "vae_loss_batch_aggregation",
        test_batch_aggregation()
    );
    failures += report_check(
        "vae_loss_backward_equivalence",
        test_backward_equivalence()
    );
    failures += report_check(
        "vae_loss_forward_negative_contracts",
        test_forward_negative_contracts()
    );
    failures += report_check(
        "vae_loss_backward_strong_guarantee",
        test_backward_strong_guarantee()
    );
    failures += report_check(
        "vae_loss_gradient_check",
        test_gradient_check()
    );

    if (failures != 0) {
        fprintf(stderr, "%d VAE C VAE LOSS TEST(S) FAILED\n", failures);
        return 1;
    }

    printf("ALL VAE C VAE LOSS TESTS PASSED\n");
    return 0;
}
