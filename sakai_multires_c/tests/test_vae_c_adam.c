#include "vae_c_adam.h"

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>


#define TEST_MAX_PARAMETER_COUNT 16


typedef struct {
    VAECParameter metadata;
    float *storage_address;
    float storage[4 * TEST_MAX_PARAMETER_COUNT];
    size_t count;
} ParameterSnapshot;


static int report_check(const char *name, int passed)
{
    if (passed) {
        printf("[PASS] %s\n", name);
        return 0;
    }

    fprintf(stderr, "[FAIL] %s\n", name);
    return 1;
}


static int floats_bitwise_equal(
    const float *left,
    const float *right,
    size_t count
)
{
    return memcmp(left, right, count * sizeof(float)) == 0;
}


static int optimizer_bitwise_equal(
    const VAECAdam *left,
    const VAECAdam *right
)
{
    return memcmp(left, right, sizeof(*left)) == 0;
}


static int parameter_snapshot_take(
    const VAECParameter *parameter,
    size_t actual_count,
    ParameterSnapshot *snapshot
)
{
    float *storage_address;

    if (parameter == NULL || snapshot == NULL ||
        actual_count > TEST_MAX_PARAMETER_COUNT) {

        return 0;
    }

    storage_address =
        parameter->owned_block != NULL
            ? parameter->owned_block
            : parameter->values;
    if (storage_address == NULL) {
        return 0;
    }

    snapshot->metadata = *parameter;
    snapshot->storage_address = storage_address;
    snapshot->count = actual_count;
    memcpy(
        snapshot->storage,
        storage_address,
        4 * actual_count * sizeof(float)
    );
    return 1;
}


static int parameter_matches_snapshot(
    const VAECParameter *parameter,
    const ParameterSnapshot *snapshot
)
{
    return
        parameter != NULL &&
        snapshot != NULL &&
        memcmp(
            parameter,
            &snapshot->metadata,
            sizeof(*parameter)) == 0 &&
        memcmp(
            snapshot->storage_address,
            snapshot->storage,
            4 * snapshot->count * sizeof(float)) == 0;
}


static int allocate_parameter(
    VAECParameter *parameter,
    size_t rank,
    const size_t *dims,
    uint64_t parameter_key
)
{
    VAECShape shape;

    vae_c_parameter_init(parameter);
    return
        vae_c_shape_make(rank, dims, &shape) == VAE_C_STORAGE_OK &&
        shape.count <= TEST_MAX_PARAMETER_COUNT &&
        vae_c_parameter_allocate(
            parameter,
            &shape,
            parameter_key) == VAE_C_STORAGE_OK;
}


static void fill_parameter(
    VAECParameter *parameter,
    const float *values,
    const float *gradients,
    const float *adam_m,
    const float *adam_v
)
{
    size_t index;

    for (index = 0; index < parameter->shape.count; index++) {
        parameter->values[index] = values[index];
        parameter->gradients[index] = gradients[index];
        parameter->adam_m[index] = adam_m[index];
        parameter->adam_v[index] = adam_v[index];
    }
}


static void reference_adam_step(
    VAECAdam *optimizer,
    float *values,
    const float *gradients,
    float *adam_m,
    float *adam_v,
    size_t count
)
{
    double next_beta1_power =
        optimizer->beta1_power * optimizer->beta1;
    double next_beta2_power =
        optimizer->beta2_power * optimizer->beta2;
    double bias_correction1 = 1.0 - next_beta1_power;
    double bias_correction2 = 1.0 - next_beta2_power;
    size_t index;

    for (index = 0; index < count; index++) {
        double m_d =
            optimizer->beta1 * (double)adam_m[index] +
            (1.0 - optimizer->beta1) * (double)gradients[index];
        double v_d =
            optimizer->beta2 * (double)adam_v[index] +
            (1.0 - optimizer->beta2) *
                (double)gradients[index] * (double)gradients[index];
        float rounded_m = (float)m_d;
        float rounded_v = (float)v_d;
        double m_hat = (double)rounded_m / bias_correction1;
        double v_hat = (double)rounded_v / bias_correction2;
        double update =
            optimizer->learning_rate * m_hat /
            (sqrt(v_hat) + optimizer->adam_epsilon);

        adam_m[index] = rounded_m;
        adam_v[index] = rounded_v;
        values[index] = (float)((double)values[index] - update);
    }

    optimizer->step += UINT64_C(1);
    optimizer->beta1_power = next_beta1_power;
    optimizer->beta2_power = next_beta2_power;
}


static int expect_failure_preserving_registry(
    VAECAdam *optimizer,
    VAECParameter *const *parameters,
    size_t parameter_count,
    VAECParameter *const *snapshot_parameters,
    const size_t *actual_counts,
    size_t snapshot_count
)
{
    VAECAdam optimizer_before;
    ParameterSnapshot snapshots[2];
    size_t index;

    if (snapshot_count > 2) {
        return 0;
    }

    optimizer_before = *optimizer;
    for (index = 0; index < snapshot_count; index++) {
        if (!parameter_snapshot_take(
                snapshot_parameters[index],
                actual_counts[index],
                &snapshots[index])) {

            return 0;
        }
    }

    if (vae_c_adam_step(
            optimizer,
            parameters,
            parameter_count) == VAE_C_OP_OK ||
        !optimizer_bitwise_equal(optimizer, &optimizer_before)) {

        return 0;
    }

    for (index = 0; index < snapshot_count; index++) {
        if (!parameter_matches_snapshot(
                snapshot_parameters[index],
                &snapshots[index])) {

            return 0;
        }
    }

    return 1;
}


static int test_lifecycle_and_configuration(void)
{
    const size_t dims[1] = {1};
    const float zeros[1] = {0.0f};
    VAECParameter parameter;
    VAECParameter *registry[1];
    VAECAdam optimizer;
    VAECAdam before;

    vae_c_parameter_init(&parameter);
    vae_c_adam_init(&optimizer);
    if (!vae_c_adam_is_empty(&optimizer) ||
        vae_c_adam_is_configured(&optimizer) ||
        vae_c_adam_is_empty(NULL) ||
        vae_c_adam_is_configured(NULL) ||
        vae_c_adam_configure_v1(NULL) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        vae_c_adam_configure_v1(&optimizer) != VAE_C_OP_OK ||
        vae_c_adam_is_empty(&optimizer) ||
        !vae_c_adam_is_configured(&optimizer) ||
        optimizer.learning_rate != VAE_C_ADAM_LEARNING_RATE_V1 ||
        optimizer.beta1 != VAE_C_ADAM_BETA1_V1 ||
        optimizer.beta2 != VAE_C_ADAM_BETA2_V1 ||
        optimizer.adam_epsilon != VAE_C_ADAM_EPSILON_V1 ||
        optimizer.step != UINT64_C(0) ||
        optimizer.beta1_power != 1.0 ||
        optimizer.beta2_power != 1.0) {

        return 0;
    }

    before = optimizer;
    if (vae_c_adam_configure_v1(&optimizer) !=
            VAE_C_OP_OBJECT_NOT_EMPTY ||
        !optimizer_bitwise_equal(&optimizer, &before)) {

        return 0;
    }

    if (!allocate_parameter(
            &parameter,
            1,
            dims,
            UINT64_C(0))) {

        return 0;
    }
    fill_parameter(&parameter, zeros, zeros, zeros, zeros);
    registry[0] = &parameter;
    if (vae_c_adam_step(&optimizer, registry, 1) != VAE_C_OP_OK ||
        vae_c_adam_is_empty(&optimizer) ||
        !vae_c_adam_is_configured(&optimizer)) {

        vae_c_parameter_free(&parameter);
        return 0;
    }
    vae_c_parameter_free(&parameter);

    vae_c_adam_free(&optimizer);
    if (!vae_c_adam_is_empty(&optimizer) ||
        vae_c_adam_is_configured(&optimizer)) {

        return 0;
    }
    vae_c_adam_free(&optimizer);
    if (!vae_c_adam_is_empty(&optimizer) ||
        vae_c_adam_is_configured(&optimizer) ||
        vae_c_adam_configure_v1(&optimizer) != VAE_C_OP_OK ||
        !vae_c_adam_is_configured(&optimizer)) {

        return 0;
    }

    vae_c_adam_free(&optimizer);
    optimizer.beta1 = VAE_C_ADAM_BETA1_V1;
    before = optimizer;
    if (vae_c_adam_configure_v1(&optimizer) !=
            VAE_C_OP_OBJECT_NOT_EMPTY ||
        vae_c_adam_is_configured(&optimizer) ||
        !optimizer_bitwise_equal(&optimizer, &before)) {

        return 0;
    }

    vae_c_adam_init(NULL);
    vae_c_adam_free(NULL);
    return 1;
}


static int configured_predicate_matches_state(
    VAECAdam *optimizer,
    int expected_empty,
    int expected_configured
)
{
    VAECAdam before;

    before = *optimizer;
    if ((vae_c_adam_is_configured(optimizer) ? 1 : 0) !=
            expected_configured ||
        !optimizer_bitwise_equal(optimizer, &before) ||
        (vae_c_adam_is_configured(optimizer) ? 1 : 0) !=
            expected_configured ||
        !optimizer_bitwise_equal(optimizer, &before) ||
        (vae_c_adam_is_empty(optimizer) ? 1 : 0) != expected_empty ||
        !optimizer_bitwise_equal(optimizer, &before) ||
        (vae_c_adam_is_empty(optimizer) ? 1 : 0) != expected_empty ||
        !optimizer_bitwise_equal(optimizer, &before)) {

        return 0;
    }

    return 1;
}


static void corrupt_predicate_hyperparameter(
    VAECAdam *optimizer,
    size_t case_index
)
{
    switch (case_index) {
        case 0:
            optimizer->learning_rate = 0.0;
            break;
        case 1:
            optimizer->learning_rate = NAN;
            break;
        case 2:
            optimizer->learning_rate = 1.0e-3;
            break;
        case 3:
            optimizer->beta1 = 0.0;
            break;
        case 4:
            optimizer->beta1 = 1.0;
            break;
        case 5:
            optimizer->beta1 = 0.8;
            break;
        case 6:
            optimizer->beta2 = 0.0;
            break;
        case 7:
            optimizer->beta2 = 1.0;
            break;
        case 8:
            optimizer->beta2 = 0.99;
            break;
        case 9:
            optimizer->adam_epsilon = 0.0;
            break;
        case 10:
            optimizer->adam_epsilon = NAN;
            break;
        case 11:
            optimizer->adam_epsilon = 1.0e-7;
            break;
        default:
            optimizer->learning_rate = -1.0;
            break;
    }
}


static int test_configured_predicate_hyperparameter_corruption(void)
{
    VAECAdam valid;
    size_t case_index;

    vae_c_adam_init(&valid);
    if (vae_c_adam_configure_v1(&valid) != VAE_C_OP_OK ||
        !configured_predicate_matches_state(&valid, 0, 1)) {

        return 0;
    }

    for (case_index = 0; case_index < 13; case_index++) {
        VAECAdam corrupted = valid;

        corrupt_predicate_hyperparameter(&corrupted, case_index);
        if (case_index == 12 &&
            (!isfinite(corrupted.learning_rate) ||
             corrupted.learning_rate >= 0.0 ||
             !corrupted.configured)) {

            return 0;
        }
        if (!configured_predicate_matches_state(&corrupted, 0, 0)) {
            return 0;
        }
    }

    return 1;
}


static void corrupt_predicate_initial_power(
    VAECAdam *optimizer,
    size_t case_index
)
{
    switch (case_index) {
        case 0:
            optimizer->beta1_power = 0.9;
            break;
        case 1:
            optimizer->beta2_power = 0.999;
            break;
        case 2:
            optimizer->beta1_power = NAN;
            break;
        case 3:
            optimizer->beta2_power = INFINITY;
            break;
        case 4:
            optimizer->beta1_power = -0.1;
            break;
        default:
            optimizer->beta2_power = 1.1;
            break;
    }
}


static int test_configured_predicate_power_step_and_purity(void)
{
    const size_t dims[1] = {1};
    const float values[1] = {1.0f};
    const float gradients[1] = {0.25f};
    const float zeros[1] = {0.0f};
    VAECParameter parameter;
    VAECParameter *registry[1];
    VAECAdam empty;
    VAECAdam valid_initial;
    VAECAdam valid_after_step;
    size_t case_index;
    int passed = 1;

    vae_c_adam_init(&empty);
    vae_c_adam_init(&valid_initial);
    if (!configured_predicate_matches_state(&empty, 1, 0) ||
        vae_c_adam_configure_v1(&valid_initial) != VAE_C_OP_OK ||
        !configured_predicate_matches_state(&valid_initial, 0, 1)) {

        return 0;
    }

    for (case_index = 0; case_index < 6; case_index++) {
        VAECAdam corrupted = valid_initial;

        corrupt_predicate_initial_power(&corrupted, case_index);
        if (!configured_predicate_matches_state(&corrupted, 0, 0)) {
            return 0;
        }
    }

    {
        VAECAdam corrupted = valid_initial;

        corrupted.configured = false;
        if (!configured_predicate_matches_state(&corrupted, 0, 0)) {
            return 0;
        }
    }
    {
        VAECAdam corrupted = empty;

        corrupted.configured = true;
        if (!configured_predicate_matches_state(&corrupted, 0, 0)) {
            return 0;
        }
    }

    if (!allocate_parameter(
            &parameter,
            1,
            dims,
            UINT64_C(3))) {

        return 0;
    }
    fill_parameter(
        &parameter,
        values,
        gradients,
        zeros,
        zeros
    );
    registry[0] = &parameter;
    valid_after_step = valid_initial;
    if (vae_c_adam_step(
            &valid_after_step,
            registry,
            1) != VAE_C_OP_OK ||
        !configured_predicate_matches_state(
            &valid_after_step,
            0,
            1)) {

        passed = 0;
    }

    if (passed) {
        VAECAdam corrupted = valid_after_step;

        corrupted.beta1_power = 1.0;
        passed = configured_predicate_matches_state(&corrupted, 0, 0);
    }
    if (passed) {
        VAECAdam corrupted = valid_after_step;

        corrupted.beta2_power = 1.0;
        passed = configured_predicate_matches_state(&corrupted, 0, 0);
    }
    if (passed) {
        VAECAdam corrupted = valid_after_step;

        corrupted.step = UINT64_MAX;
        passed = configured_predicate_matches_state(&corrupted, 0, 0);
    }

    vae_c_parameter_free(&parameter);
    return passed;
}


static int test_analytic_first_step(void)
{
    const size_t dims[1] = {4};
    const float initial_values[4] = {1.0f, -2.0f, 0.5f, 3.0f};
    const float gradients[4] = {0.25f, -0.5f, 0.0f, 2.0f};
    const float zeros[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    VAECParameter parameter;
    VAECParameter *registry[1];
    VAECAdam optimizer;
    VAECAdam reference_optimizer;
    VAECShape shape_before;
    uint64_t key_before;
    float reference_values[4];
    float reference_m[4];
    float reference_v[4];
    float gradients_before[4];
    int passed;

    if (!allocate_parameter(
            &parameter,
            1,
            dims,
            UINT64_C(0))) {

        return 0;
    }

    fill_parameter(
        &parameter,
        initial_values,
        gradients,
        zeros,
        zeros
    );
    memcpy(reference_values, initial_values, sizeof(reference_values));
    memcpy(reference_m, zeros, sizeof(reference_m));
    memcpy(reference_v, zeros, sizeof(reference_v));
    memcpy(gradients_before, parameter.gradients, sizeof(gradients_before));
    shape_before = parameter.shape;
    key_before = parameter.parameter_key;

    vae_c_adam_init(&optimizer);
    if (vae_c_adam_configure_v1(&optimizer) != VAE_C_OP_OK) {
        vae_c_parameter_free(&parameter);
        return 0;
    }
    reference_optimizer = optimizer;
    reference_adam_step(
        &reference_optimizer,
        reference_values,
        gradients,
        reference_m,
        reference_v,
        4
    );

    registry[0] = &parameter;
    passed =
        vae_c_adam_step(&optimizer, registry, 1) == VAE_C_OP_OK &&
        optimizer_bitwise_equal(&optimizer, &reference_optimizer) &&
        floats_bitwise_equal(parameter.values, reference_values, 4) &&
        floats_bitwise_equal(parameter.adam_m, reference_m, 4) &&
        floats_bitwise_equal(parameter.adam_v, reference_v, 4) &&
        floats_bitwise_equal(parameter.gradients, gradients_before, 4) &&
        memcmp(&parameter.shape, &shape_before, sizeof(shape_before)) == 0 &&
        parameter.parameter_key == key_before &&
        optimizer.step == UINT64_C(1) &&
        optimizer.beta1_power == VAE_C_ADAM_BETA1_V1 &&
        optimizer.beta2_power == VAE_C_ADAM_BETA2_V1;

    vae_c_parameter_free(&parameter);
    return passed;
}


static int test_multiple_steps_and_momentum(void)
{
    const size_t dims[1] = {3};
    const float initial_values[3] = {0.75f, -1.25f, 2.0f};
    const float initial_m[3] = {0.0f, 0.0f, 0.0f};
    const float initial_v[3] = {0.0f, 0.0f, 0.0f};
    const float gradients[3][3] = {
        {0.2f, -0.4f, 1.0f},
        {-0.1f, 0.0f, 0.5f},
        {0.0f, 0.0f, 0.0f}
    };
    VAECParameter parameter;
    VAECParameter *registry[1];
    VAECAdam optimizer;
    VAECAdam reference_optimizer;
    float reference_values[3];
    float reference_m[3];
    float reference_v[3];
    size_t step_index;
    int passed = 1;

    if (!allocate_parameter(
            &parameter,
            1,
            dims,
            UINT64_C(19))) {

        return 0;
    }

    fill_parameter(
        &parameter,
        initial_values,
        gradients[0],
        initial_m,
        initial_v
    );
    memcpy(reference_values, initial_values, sizeof(reference_values));
    memcpy(reference_m, initial_m, sizeof(reference_m));
    memcpy(reference_v, initial_v, sizeof(reference_v));
    vae_c_adam_init(&optimizer);
    if (vae_c_adam_configure_v1(&optimizer) != VAE_C_OP_OK) {
        vae_c_parameter_free(&parameter);
        return 0;
    }
    reference_optimizer = optimizer;
    registry[0] = &parameter;

    for (step_index = 0; step_index < 3; step_index++) {
        float gradients_before[3];
        float value_before_zero_gradient = parameter.values[0];

        memcpy(
            parameter.gradients,
            gradients[step_index],
            sizeof(gradients[step_index])
        );
        memcpy(
            gradients_before,
            parameter.gradients,
            sizeof(gradients_before)
        );
        reference_adam_step(
            &reference_optimizer,
            reference_values,
            gradients[step_index],
            reference_m,
            reference_v,
            3
        );

        if (vae_c_adam_step(&optimizer, registry, 1) != VAE_C_OP_OK ||
            !optimizer_bitwise_equal(
                &optimizer,
                &reference_optimizer) ||
            !floats_bitwise_equal(
                parameter.values,
                reference_values,
                3) ||
            !floats_bitwise_equal(parameter.adam_m, reference_m, 3) ||
            !floats_bitwise_equal(parameter.adam_v, reference_v, 3) ||
            !floats_bitwise_equal(
                parameter.gradients,
                gradients_before,
                3) ||
            optimizer.step != (uint64_t)(step_index + 1)) {

            passed = 0;
            break;
        }

        if (step_index == 2 &&
            parameter.values[0] == value_before_zero_gradient) {

            passed = 0;
            break;
        }
    }

    vae_c_parameter_free(&parameter);
    return passed;
}


static int test_zero_gradient_first_step(void)
{
    const size_t dims[1] = {3};
    const float values[3] = {1.0f, -2.0f, 3.5f};
    const float zeros[3] = {0.0f, 0.0f, 0.0f};
    VAECParameter parameter;
    VAECParameter *registry[1];
    VAECAdam optimizer;
    float values_before[3];
    int passed;

    if (!allocate_parameter(
            &parameter,
            1,
            dims,
            UINT64_C(4))) {

        return 0;
    }

    fill_parameter(&parameter, values, zeros, zeros, zeros);
    memcpy(values_before, parameter.values, sizeof(values_before));
    vae_c_adam_init(&optimizer);
    registry[0] = &parameter;
    passed =
        vae_c_adam_configure_v1(&optimizer) == VAE_C_OP_OK &&
        vae_c_adam_step(&optimizer, registry, 1) == VAE_C_OP_OK &&
        floats_bitwise_equal(parameter.values, values_before, 3) &&
        floats_bitwise_equal(parameter.gradients, zeros, 3) &&
        floats_bitwise_equal(parameter.adam_m, zeros, 3) &&
        floats_bitwise_equal(parameter.adam_v, zeros, 3) &&
        optimizer.step == UINT64_C(1) &&
        optimizer.beta1_power == VAE_C_ADAM_BETA1_V1 &&
        optimizer.beta2_power == VAE_C_ADAM_BETA2_V1;

    vae_c_parameter_free(&parameter);
    return passed;
}


static int test_multiparameter_and_order_independence(void)
{
    const size_t vector_dims[1] = {3};
    const size_t matrix_dims[2] = {2, 2};
    const float vector_values[3] = {1.0f, -2.0f, 0.25f};
    const float vector_gradients[3] = {0.3f, -0.7f, 0.0f};
    const float vector_m[3] = {0.1f, 0.0f, -0.2f};
    const float vector_v[3] = {0.02f, 0.03f, 0.04f};
    const float matrix_values[4] = {0.5f, 1.5f, -1.0f, 2.5f};
    const float matrix_gradients[4] = {-0.2f, 0.4f, 0.8f, -1.2f};
    const float matrix_m[4] = {0.0f, 0.2f, -0.1f, 0.3f};
    const float matrix_v[4] = {0.0f, 0.01f, 0.02f, 0.03f};
    VAECParameter vector_ab;
    VAECParameter matrix_ab;
    VAECParameter vector_ba;
    VAECParameter matrix_ba;
    VAECParameter *registry_ab[2];
    VAECParameter *registry_ba[2];
    VAECAdam optimizer_ab;
    VAECAdam optimizer_ba;
    float vector_gradients_before[3];
    float matrix_gradients_before[4];
    int passed;

    if (!allocate_parameter(
            &vector_ab,
            1,
            vector_dims,
            UINT64_C(0)) ||
        !allocate_parameter(
            &matrix_ab,
            2,
            matrix_dims,
            UINT64_C(73)) ||
        !allocate_parameter(
            &vector_ba,
            1,
            vector_dims,
            UINT64_C(0)) ||
        !allocate_parameter(
            &matrix_ba,
            2,
            matrix_dims,
            UINT64_C(73))) {

        return 0;
    }

    fill_parameter(
        &vector_ab,
        vector_values,
        vector_gradients,
        vector_m,
        vector_v
    );
    fill_parameter(
        &vector_ba,
        vector_values,
        vector_gradients,
        vector_m,
        vector_v
    );
    fill_parameter(
        &matrix_ab,
        matrix_values,
        matrix_gradients,
        matrix_m,
        matrix_v
    );
    fill_parameter(
        &matrix_ba,
        matrix_values,
        matrix_gradients,
        matrix_m,
        matrix_v
    );
    memcpy(
        vector_gradients_before,
        vector_gradients,
        sizeof(vector_gradients_before)
    );
    memcpy(
        matrix_gradients_before,
        matrix_gradients,
        sizeof(matrix_gradients_before)
    );

    vae_c_adam_init(&optimizer_ab);
    vae_c_adam_init(&optimizer_ba);
    registry_ab[0] = &vector_ab;
    registry_ab[1] = &matrix_ab;
    registry_ba[0] = &matrix_ba;
    registry_ba[1] = &vector_ba;

    passed =
        vae_c_adam_configure_v1(&optimizer_ab) == VAE_C_OP_OK &&
        vae_c_adam_configure_v1(&optimizer_ba) == VAE_C_OP_OK &&
        vae_c_adam_step(&optimizer_ab, registry_ab, 2) == VAE_C_OP_OK &&
        vae_c_adam_step(&optimizer_ba, registry_ba, 2) == VAE_C_OP_OK &&
        optimizer_bitwise_equal(&optimizer_ab, &optimizer_ba) &&
        optimizer_ab.step == UINT64_C(1) &&
        floats_bitwise_equal(vector_ab.values, vector_ba.values, 3) &&
        floats_bitwise_equal(vector_ab.adam_m, vector_ba.adam_m, 3) &&
        floats_bitwise_equal(vector_ab.adam_v, vector_ba.adam_v, 3) &&
        floats_bitwise_equal(matrix_ab.values, matrix_ba.values, 4) &&
        floats_bitwise_equal(matrix_ab.adam_m, matrix_ba.adam_m, 4) &&
        floats_bitwise_equal(matrix_ab.adam_v, matrix_ba.adam_v, 4) &&
        floats_bitwise_equal(
            vector_ab.gradients,
            vector_gradients_before,
            3) &&
        floats_bitwise_equal(
            matrix_ab.gradients,
            matrix_gradients_before,
            4) &&
        vector_ab.parameter_key == UINT64_C(0) &&
        matrix_ab.parameter_key == UINT64_C(73) &&
        vector_ab.shape.rank == 1 &&
        matrix_ab.shape.rank == 2;

    vae_c_parameter_free(&vector_ab);
    vae_c_parameter_free(&matrix_ab);
    vae_c_parameter_free(&vector_ba);
    vae_c_parameter_free(&matrix_ba);
    return passed;
}


static void corrupt_optimizer(VAECAdam *optimizer, size_t case_index)
{
    switch (case_index) {
        case 0:
            optimizer->configured = false;
            break;
        case 1:
            optimizer->learning_rate = NAN;
            break;
        case 2:
            optimizer->learning_rate = 0.0;
            break;
        case 3:
            optimizer->learning_rate = INFINITY;
            break;
        case 4:
            optimizer->learning_rate = 1.0e-3;
            break;
        case 5:
            optimizer->beta1 = 0.0;
            break;
        case 6:
            optimizer->beta1 = 1.0;
            break;
        case 7:
            optimizer->beta1 = NAN;
            break;
        case 8:
            optimizer->beta1 = 0.8;
            break;
        case 9:
            optimizer->beta2 = 0.0;
            break;
        case 10:
            optimizer->beta2 = 1.0;
            break;
        case 11:
            optimizer->beta2 = INFINITY;
            break;
        case 12:
            optimizer->beta2 = 0.99;
            break;
        case 13:
            optimizer->adam_epsilon = 0.0;
            break;
        case 14:
            optimizer->adam_epsilon = -1.0;
            break;
        case 15:
            optimizer->adam_epsilon = NAN;
            break;
        case 16:
            optimizer->adam_epsilon = 1.0e-7;
            break;
        case 17:
            optimizer->beta1_power = NAN;
            break;
        case 18:
            optimizer->beta1_power = INFINITY;
            break;
        case 19:
            optimizer->beta1_power = -0.1;
            break;
        case 20:
            optimizer->beta2_power = 1.1;
            break;
        case 21:
            optimizer->beta2_power = NAN;
            break;
        case 22:
            optimizer->beta1_power = 0.9;
            break;
        case 23:
            optimizer->step = UINT64_C(1);
            optimizer->beta1_power = 1.0;
            optimizer->beta2_power = 0.999;
            break;
        case 24:
            optimizer->step = UINT64_C(1);
            optimizer->beta1_power = 0.9;
            optimizer->beta2_power = 1.0;
            break;
        case 25:
            optimizer->step = UINT64_MAX;
            optimizer->beta1_power = 0.0;
            optimizer->beta2_power = 0.0;
            break;
        default:
            optimizer->learning_rate = -1.0;
            break;
    }
}


static int test_optimizer_validation_and_strong_guarantee(void)
{
    const size_t dims[1] = {2};
    const float values[2] = {1.0f, -1.0f};
    const float gradients[2] = {0.25f, -0.5f};
    const float zeros[2] = {0.0f, 0.0f};
    VAECParameter parameter;
    VAECParameter *registry[1];
    VAECParameter *snap_parameters[1];
    size_t actual_counts[1] = {2};
    size_t case_index;
    int passed = 1;

    if (!allocate_parameter(
            &parameter,
            1,
            dims,
            UINT64_C(8))) {

        return 0;
    }
    fill_parameter(&parameter, values, gradients, zeros, zeros);
    registry[0] = &parameter;
    snap_parameters[0] = &parameter;

    for (case_index = 0; case_index < 27; case_index++) {
        VAECAdam optimizer;

        vae_c_adam_init(&optimizer);
        if (vae_c_adam_configure_v1(&optimizer) != VAE_C_OP_OK) {
            passed = 0;
            break;
        }
        corrupt_optimizer(&optimizer, case_index);
        if (case_index == 26 &&
            (!isfinite(optimizer.learning_rate) ||
             optimizer.learning_rate >= 0.0 ||
             !optimizer.configured)) {

            passed = 0;
            break;
        }
        {
            VAECAdam predicate_before = optimizer;

            if (vae_c_adam_is_configured(&optimizer) ||
                !optimizer_bitwise_equal(
                    &optimizer,
                    &predicate_before) ||
                vae_c_adam_is_configured(&optimizer) ||
                !optimizer_bitwise_equal(
                    &optimizer,
                    &predicate_before) ||
                vae_c_adam_is_empty(&optimizer) ||
                !optimizer_bitwise_equal(
                    &optimizer,
                    &predicate_before) ||
                vae_c_adam_is_empty(&optimizer) ||
                !optimizer_bitwise_equal(
                    &optimizer,
                    &predicate_before)) {

                passed = 0;
                break;
            }
        }
        if (case_index == 26) {
            VAECAdam optimizer_before = optimizer;
            ParameterSnapshot parameter_before;

            if (!parameter_snapshot_take(
                    &parameter,
                    2,
                    &parameter_before) ||
                vae_c_adam_step(&optimizer, registry, 1) !=
                    VAE_C_OP_INVALID_STATE ||
                !optimizer_bitwise_equal(
                    &optimizer,
                    &optimizer_before) ||
                !parameter_matches_snapshot(
                    &parameter,
                    &parameter_before)) {

                passed = 0;
                break;
            }
        } else if (!expect_failure_preserving_registry(
                       &optimizer,
                       registry,
                       1,
                       snap_parameters,
                       actual_counts,
                       1)) {

            passed = 0;
            break;
        }
    }

    if (passed) {
        ParameterSnapshot before;

        passed =
            parameter_snapshot_take(&parameter, 2, &before) &&
            vae_c_adam_step(NULL, registry, 1) ==
                VAE_C_OP_INVALID_ARGUMENT &&
            parameter_matches_snapshot(&parameter, &before);
    }

    vae_c_parameter_free(&parameter);
    return passed;
}


static int test_registry_and_storage_validation(void)
{
    const size_t dims[1] = {3};
    const float values[3] = {1.0f, 2.0f, 3.0f};
    const float gradients[3] = {0.1f, -0.2f, 0.3f};
    const float zeros[3] = {0.0f, 0.0f, 0.0f};
    VAECParameter parameter;
    VAECParameter *registry[2];
    VAECParameter *snap_parameters[1];
    size_t actual_counts[1] = {3};
    VAECParameter valid_metadata;
    VAECAdam optimizer;
    VAECAdam optimizer_before;
    ParameterSnapshot parameter_before;
    int passed = 1;

    if (!allocate_parameter(
            &parameter,
            1,
            dims,
            UINT64_C(0))) {

        return 0;
    }
    fill_parameter(&parameter, values, gradients, zeros, zeros);
    vae_c_adam_init(&optimizer);
    if (vae_c_adam_configure_v1(&optimizer) != VAE_C_OP_OK) {
        vae_c_parameter_free(&parameter);
        return 0;
    }

    registry[0] = &parameter;
    snap_parameters[0] = &parameter;
    optimizer_before = optimizer;
    if (!parameter_snapshot_take(&parameter, 3, &parameter_before) ||
        vae_c_adam_step(&optimizer, NULL, 1) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        vae_c_adam_step(&optimizer, registry, 0) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        !optimizer_bitwise_equal(&optimizer, &optimizer_before) ||
        !parameter_matches_snapshot(&parameter, &parameter_before)) {

        passed = 0;
    }

    if (passed) {
        registry[0] = NULL;
        passed = expect_failure_preserving_registry(
            &optimizer,
            registry,
            1,
            snap_parameters,
            actual_counts,
            1
        );
        registry[0] = &parameter;
    }

    if (passed) {
        registry[1] = &parameter;
        passed = expect_failure_preserving_registry(
            &optimizer,
            registry,
            2,
            snap_parameters,
            actual_counts,
            1
        );
    }

    valid_metadata = parameter;
    if (passed) {
        parameter.shape.rank = 0;
        passed = expect_failure_preserving_registry(
            &optimizer,
            registry,
            1,
            snap_parameters,
            actual_counts,
            1
        );
        parameter = valid_metadata;
    }
    if (passed) {
        parameter.shape.count += 1;
        passed = expect_failure_preserving_registry(
            &optimizer,
            registry,
            1,
            snap_parameters,
            actual_counts,
            1
        );
        parameter = valid_metadata;
    }
    if (passed) {
        parameter.shape.dims[1] = 1;
        passed = expect_failure_preserving_registry(
            &optimizer,
            registry,
            1,
            snap_parameters,
            actual_counts,
            1
        );
        parameter = valid_metadata;
    }
    if (passed) {
        parameter.gradients = parameter.values;
        passed = expect_failure_preserving_registry(
            &optimizer,
            registry,
            1,
            snap_parameters,
            actual_counts,
            1
        );
        parameter = valid_metadata;
    }
    if (passed) {
        parameter.owned_block = NULL;
        passed = expect_failure_preserving_registry(
            &optimizer,
            registry,
            1,
            snap_parameters,
            actual_counts,
            1
        );
        parameter = valid_metadata;
    }
    if (passed) {
        parameter.owned_bytes -= sizeof(float);
        passed = expect_failure_preserving_registry(
            &optimizer,
            registry,
            1,
            snap_parameters,
            actual_counts,
            1
        );
        parameter = valid_metadata;
    }

    vae_c_parameter_free(&parameter);
    return passed;
}


static int manual_parameter_make(
    VAECParameter *parameter,
    float *base,
    size_t count,
    uint64_t key
)
{
    size_t dims[1];

    dims[0] = count;
    vae_c_parameter_init(parameter);
    if (vae_c_shape_make(1, dims, &parameter->shape) !=
            VAE_C_STORAGE_OK) {

        return 0;
    }

    parameter->parameter_key = key;
    parameter->values = base;
    parameter->gradients = base + count;
    parameter->adam_m = base + 2 * count;
    parameter->adam_v = base + 3 * count;
    parameter->owned_block = base;
    parameter->owned_bytes = 4 * count * sizeof(float);
    return 1;
}


static int check_manual_alias(size_t second_base_offset)
{
    float arena[96];
    float before[96];
    VAECParameter first;
    VAECParameter second;
    VAECParameter *registry[2];
    VAECAdam optimizer;
    VAECAdam before_optimizer;
    size_t index;

    for (index = 0; index < 96; index++) {
        arena[index] = 0.01f * (float)(index + 1);
    }

    if (!manual_parameter_make(
            &first,
            arena + 32,
            8,
            UINT64_C(1)) ||
        !manual_parameter_make(
            &second,
            arena + second_base_offset,
            3,
            UINT64_C(2))) {

        return 0;
    }

    memcpy(before, arena, sizeof(before));
    vae_c_adam_init(&optimizer);
    if (vae_c_adam_configure_v1(&optimizer) != VAE_C_OP_OK) {
        return 0;
    }
    before_optimizer = optimizer;
    registry[0] = &first;
    registry[1] = &second;

    return
        vae_c_adam_step(&optimizer, registry, 2) != VAE_C_OP_OK &&
        optimizer_bitwise_equal(&optimizer, &before_optimizer) &&
        memcmp(arena, before, sizeof(arena)) == 0;
}


static int test_duplicate_and_alias_rejection(void)
{
    const size_t dims[1] = {2};
    const float values[2] = {1.0f, 2.0f};
    const float gradients[2] = {0.1f, -0.2f};
    const float zeros[2] = {0.0f, 0.0f};
    VAECParameter parameter;
    VAECParameter clone;
    VAECParameter *registry[2];
    VAECParameter *snap_parameters[1];
    size_t actual_counts[1] = {2};
    VAECAdam optimizer;
    int passed;

    if (!allocate_parameter(
            &parameter,
            1,
            dims,
            UINT64_C(5))) {

        return 0;
    }
    fill_parameter(&parameter, values, gradients, zeros, zeros);
    clone = parameter;
    registry[0] = &parameter;
    registry[1] = &clone;
    snap_parameters[0] = &parameter;
    vae_c_adam_init(&optimizer);
    passed =
        vae_c_adam_configure_v1(&optimizer) == VAE_C_OP_OK &&
        expect_failure_preserving_registry(
            &optimizer,
            registry,
            2,
            snap_parameters,
            actual_counts,
            1) &&
        check_manual_alias(32) &&
        check_manual_alias(37) &&
        check_manual_alias(42) &&
        check_manual_alias(47) &&
        check_manual_alias(29);

    vae_c_parameter_free(&parameter);
    return passed;
}


static int test_optimizer_buffer_alias_rejection(void)
{
    union {
        VAECAdam optimizer;
        float storage[32];
    } shared;
    VAECParameter parameter;
    VAECParameter *registry[1];
    unsigned char before[sizeof(shared)];

    vae_c_adam_init(&shared.optimizer);
    if (vae_c_adam_configure_v1(&shared.optimizer) != VAE_C_OP_OK ||
        !manual_parameter_make(
            &parameter,
            shared.storage,
            2,
            UINT64_C(9))) {

        return 0;
    }

    memcpy(before, &shared, sizeof(before));
    registry[0] = &parameter;
    return
        vae_c_adam_step(&shared.optimizer, registry, 1) != VAE_C_OP_OK &&
        memcmp(&shared, before, sizeof(before)) == 0;
}


static void set_valid_numeric_baseline(VAECParameter *parameter)
{
    parameter->values[0] = 1.0f;
    parameter->gradients[0] = 0.25f;
    parameter->adam_m[0] = 0.0f;
    parameter->adam_v[0] = 0.0f;
}


static void corrupt_numeric_value(
    VAECParameter *parameter,
    size_t case_index
)
{
    switch (case_index) {
        case 0:
            parameter->values[0] = NAN;
            break;
        case 1:
            parameter->values[0] = INFINITY;
            break;
        case 2:
            parameter->values[0] = -INFINITY;
            break;
        case 3:
            parameter->gradients[0] = NAN;
            break;
        case 4:
            parameter->gradients[0] = INFINITY;
            break;
        case 5:
            parameter->gradients[0] = -INFINITY;
            break;
        case 6:
            parameter->adam_m[0] = NAN;
            break;
        case 7:
            parameter->adam_m[0] = INFINITY;
            break;
        case 8:
            parameter->adam_m[0] = -INFINITY;
            break;
        case 9:
            parameter->adam_v[0] = NAN;
            break;
        case 10:
            parameter->adam_v[0] = INFINITY;
            break;
        case 11:
            parameter->adam_v[0] = -INFINITY;
            break;
        case 12:
            parameter->adam_v[0] = -1.0f;
            break;
        case 13:
            parameter->gradients[0] = FLT_MAX;
            break;
        default:
            parameter->gradients[0] = 0.0f;
            parameter->adam_m[0] = FLT_MAX;
            parameter->adam_v[0] = 0.0f;
            break;
    }
}


static int test_numeric_rejection_and_strong_guarantee(void)
{
    const size_t dims[1] = {1};
    VAECParameter parameter;
    VAECParameter *registry[1];
    VAECParameter *snap_parameters[1];
    size_t actual_counts[1] = {1};
    size_t case_index;
    int passed = 1;

    if (!allocate_parameter(
            &parameter,
            1,
            dims,
            UINT64_C(10))) {

        return 0;
    }
    registry[0] = &parameter;
    snap_parameters[0] = &parameter;

    for (case_index = 0; case_index < 15; case_index++) {
        VAECAdam optimizer;

        set_valid_numeric_baseline(&parameter);
        corrupt_numeric_value(&parameter, case_index);
        vae_c_adam_init(&optimizer);
        if (vae_c_adam_configure_v1(&optimizer) != VAE_C_OP_OK ||
            !expect_failure_preserving_registry(
                &optimizer,
                registry,
                1,
                snap_parameters,
                actual_counts,
                1)) {

            passed = 0;
            break;
        }
    }

    vae_c_parameter_free(&parameter);
    return passed;
}


static int test_multiparameter_atomicity(void)
{
    const size_t first_dims[1] = {3};
    const size_t second_dims[2] = {2, 2};
    const float first_values[3] = {1.0f, -2.0f, 3.0f};
    const float first_gradients[3] = {0.1f, -0.2f, 0.3f};
    const float first_m[3] = {0.0f, 0.1f, -0.1f};
    const float first_v[3] = {0.0f, 0.01f, 0.02f};
    const float second_values[4] = {0.5f, 1.5f, -1.0f, 2.0f};
    const float second_gradients[4] = {0.2f, -0.4f, 0.6f, -0.8f};
    const float second_m[4] = {0.0f, 0.0f, 0.1f, -0.1f};
    const float second_v[4] = {0.0f, 0.01f, 0.02f, 0.03f};
    VAECParameter first;
    VAECParameter second;
    VAECParameter *registry[2];
    VAECParameter *snap_parameters[2];
    size_t actual_counts[2] = {3, 4};
    size_t case_index;
    int passed = 1;

    if (!allocate_parameter(
            &first,
            1,
            first_dims,
            UINT64_C(1)) ||
        !allocate_parameter(
            &second,
            2,
            second_dims,
            UINT64_C(2))) {

        return 0;
    }
    registry[0] = &first;
    registry[1] = &second;
    snap_parameters[0] = &first;
    snap_parameters[1] = &second;

    for (case_index = 0; case_index < 3; case_index++) {
        VAECAdam optimizer;

        fill_parameter(
            &first,
            first_values,
            first_gradients,
            first_m,
            first_v
        );
        fill_parameter(
            &second,
            second_values,
            second_gradients,
            second_m,
            second_v
        );
        if (case_index == 0) {
            second.gradients[3] = NAN;
        } else if (case_index == 1) {
            second.adam_v[3] = -1.0f;
        } else {
            second.gradients[3] = FLT_MAX;
        }

        vae_c_adam_init(&optimizer);
        if (vae_c_adam_configure_v1(&optimizer) != VAE_C_OP_OK ||
            !expect_failure_preserving_registry(
                &optimizer,
                registry,
                2,
                snap_parameters,
                actual_counts,
                2)) {

            passed = 0;
            break;
        }
    }

    vae_c_parameter_free(&first);
    vae_c_parameter_free(&second);
    return passed;
}


int main(void)
{
    int failures = 0;

    failures += report_check(
        "adam_lifecycle_and_configuration",
        test_lifecycle_and_configuration()
    );
    failures += report_check(
        "adam_configured_predicate_hyperparameter_corruption",
        test_configured_predicate_hyperparameter_corruption()
    );
    failures += report_check(
        "adam_configured_predicate_power_step_and_purity",
        test_configured_predicate_power_step_and_purity()
    );
    failures += report_check(
        "adam_analytic_first_step",
        test_analytic_first_step()
    );
    failures += report_check(
        "adam_multiple_steps_and_momentum",
        test_multiple_steps_and_momentum()
    );
    failures += report_check(
        "adam_zero_gradient_first_step",
        test_zero_gradient_first_step()
    );
    failures += report_check(
        "adam_multiparameter_and_order_independence",
        test_multiparameter_and_order_independence()
    );
    failures += report_check(
        "adam_optimizer_validation_and_strong_guarantee",
        test_optimizer_validation_and_strong_guarantee()
    );
    failures += report_check(
        "adam_registry_and_storage_validation",
        test_registry_and_storage_validation()
    );
    failures += report_check(
        "adam_duplicate_and_alias_rejection",
        test_duplicate_and_alias_rejection()
    );
    failures += report_check(
        "adam_optimizer_buffer_alias_rejection",
        test_optimizer_buffer_alias_rejection()
    );
    failures += report_check(
        "adam_numeric_rejection_and_strong_guarantee",
        test_numeric_rejection_and_strong_guarantee()
    );
    failures += report_check(
        "adam_multiparameter_atomicity",
        test_multiparameter_atomicity()
    );

    if (failures != 0) {
        fprintf(stderr, "%d VAE C ADAM TEST(S) FAILED\n", failures);
        return 1;
    }

    printf("ALL VAE C ADAM TESTS PASSED\n");
    return 0;
}
