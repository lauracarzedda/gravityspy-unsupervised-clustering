#include "baseline_pca.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


static const float gs_test_data[18] = {
     1.0f,  2.0f,  0.0f,
    -1.0f,  0.0f,  1.0f,
     2.0f, -1.0f,  0.0f,
     0.0f,  1.0f, -2.0f,
    -2.0f, -1.0f,  1.0f,
     1.0f, -1.0f,  0.0f
};


static int gs_test_close(double left, double right, double tolerance)
{
    return isfinite(left) &&
           isfinite(right) &&
           fabs(left - right) <= tolerance;
}


static int gs_test_model_is_zero(const GSBaselinePCA *model)
{
    return model != NULL &&
           model->input_dim == 0 &&
           model->component_count == 0 &&
           model->mean == NULL &&
           model->components == NULL &&
           model->eigenvalues == NULL &&
           model->explained_variance_ratio == NULL &&
           model->cumulative_explained_variance_ratio == NULL &&
           model->total_variance == 0.0 &&
           model->components_90 == 0 &&
           model->components_95 == 0 &&
           model->components_99 == 0;
}


static void gs_test_make_axis_dataset(
    const double *variances,
    size_t dimensions,
    float *x
)
{
    size_t rows = 2 * dimensions;
    size_t count = rows * dimensions;
    size_t feature;

    memset(x, 0, count * sizeof(float));
    for (feature = 0; feature < dimensions; feature++) {
        float amplitude = (float)sqrt(
            variances[feature] * (double)rows / 2.0
        );

        x[(2 * feature) * dimensions + feature] = amplitude;
        x[(2 * feature + 1) * dimensions + feature] = -amplitude;
    }
}


static void gs_test_covariance(
    const float *x,
    size_t rows,
    size_t cols,
    double *mean,
    double *covariance
)
{
    size_t row;
    size_t left;
    size_t right;

    memset(mean, 0, cols * sizeof(double));
    memset(covariance, 0, cols * cols * sizeof(double));

    for (row = 0; row < rows; row++) {
        for (left = 0; left < cols; left++) {
            mean[left] += (double)x[row * cols + left];
        }
    }
    for (left = 0; left < cols; left++) {
        mean[left] /= (double)rows;
    }

    for (left = 0; left < cols; left++) {
        for (right = 0; right < cols; right++) {
            double sum = 0.0;

            for (row = 0; row < rows; row++) {
                sum +=
                    ((double)x[row * cols + left] - mean[left]) *
                    ((double)x[row * cols + right] - mean[right]);
            }
            covariance[left * cols + right] = sum / (double)rows;
        }
    }
}


static int gs_test_models_match_exactly(
    const GSBaselinePCA *left,
    const GSBaselinePCA *right
)
{
    size_t vector_bytes;
    size_t matrix_bytes;

    if (left->input_dim != right->input_dim ||
        left->component_count != right->component_count ||
        left->components_90 != right->components_90 ||
        left->components_95 != right->components_95 ||
        left->components_99 != right->components_99 ||
        memcmp(
            &left->total_variance,
            &right->total_variance,
            sizeof(double)
        ) != 0) {

        return 0;
    }

    vector_bytes = left->input_dim * sizeof(double);
    matrix_bytes =
        left->input_dim * left->component_count * sizeof(double);

    return memcmp(left->mean, right->mean, vector_bytes) == 0 &&
           memcmp(
               left->components,
               right->components,
               matrix_bytes
           ) == 0 &&
           memcmp(
               left->eigenvalues,
               right->eigenvalues,
               vector_bytes
           ) == 0 &&
           memcmp(
               left->explained_variance_ratio,
               right->explained_variance_ratio,
               vector_bytes
           ) == 0 &&
           memcmp(
               left->cumulative_explained_variance_ratio,
               right->cumulative_explained_variance_ratio,
               vector_bytes
           ) == 0;
}


static int gs_test_known_diagonal_covariance(void)
{
    const double variances[3] = {9.0, 4.0, 1.0};
    const double expected_ratio[3] = {
        9.0 / 14.0,
        4.0 / 14.0,
        1.0 / 14.0
    };
    float x[18];
    GSBaselinePCA model = {0};
    double cumulative = 0.0;
    size_t component;
    size_t feature;
    int failed = 0;

    gs_test_make_axis_dataset(variances, 3, x);
    if (gs_baseline_pca_fit(x, 6, 3, &model) !=
        GS_BASELINE_PCA_SUCCESS) {

        return 1;
    }

    if (model.input_dim != 3 || model.component_count != 3) {
        failed = 1;
    }

    for (feature = 0; feature < 3; feature++) {
        if (!gs_test_close(model.mean[feature], 0.0, 0.0)) {
            failed = 1;
        }
    }

    for (component = 0; component < 3; component++) {
        if (!gs_test_close(
                model.eigenvalues[component],
                variances[component],
                2.0e-6
            )) {

            failed = 1;
        }

        for (feature = 0; feature < 3; feature++) {
            double expected = component == feature ? 1.0 : 0.0;

            if (!gs_test_close(
                    model.components[component * 3 + feature],
                    expected,
                    1.0e-12
                )) {

                failed = 1;
            }
        }

        cumulative += expected_ratio[component];
        if (!gs_test_close(
                model.explained_variance_ratio[component],
                expected_ratio[component],
                2.0e-7
            ) ||
            !gs_test_close(
                model.cumulative_explained_variance_ratio[component],
                cumulative,
                2.0e-7
            )) {

            failed = 1;
        }
    }

    gs_baseline_pca_free(&model);
    return failed;
}


static int gs_test_covariance_reconstruction(void)
{
    GSBaselinePCA model = {0};
    double mean[3];
    double covariance[9];
    double maximum_error = 0.0;
    size_t left;
    size_t right;
    size_t component;

    if (gs_baseline_pca_fit(gs_test_data, 6, 3, &model) !=
        GS_BASELINE_PCA_SUCCESS) {

        return 1;
    }

    gs_test_covariance(gs_test_data, 6, 3, mean, covariance);
    for (left = 0; left < 3; left++) {
        for (right = 0; right < 3; right++) {
            double reconstructed = 0.0;
            double error;

            for (component = 0; component < 3; component++) {
                reconstructed +=
                    model.components[component * 3 + left] *
                    model.eigenvalues[component] *
                    model.components[component * 3 + right];
            }

            error = fabs(
                reconstructed - covariance[left * 3 + right]
            );
            if (error > maximum_error) {
                maximum_error = error;
            }
        }
    }

    gs_baseline_pca_free(&model);
    return !isfinite(maximum_error) || maximum_error > 1.0e-10;
}


static int gs_test_full_component_reconstruction(void)
{
    GSBaselinePCA model = {0};
    float scores[18];
    double maximum_error = 0.0;
    size_t row;
    size_t feature;
    size_t component;

    if (gs_baseline_pca_fit(gs_test_data, 6, 3, &model) !=
            GS_BASELINE_PCA_SUCCESS ||
        gs_baseline_pca_transform(
            &model,
            gs_test_data,
            6,
            3,
            3,
            scores,
            18
        ) != GS_BASELINE_PCA_SUCCESS) {

        gs_baseline_pca_free(&model);
        return 1;
    }

    for (row = 0; row < 6; row++) {
        for (feature = 0; feature < 3; feature++) {
            double reconstructed = model.mean[feature];
            double error;

            for (component = 0; component < 3; component++) {
                reconstructed +=
                    (double)scores[row * 3 + component] *
                    model.components[component * 3 + feature];
            }

            error = fabs(
                reconstructed -
                (double)gs_test_data[row * 3 + feature]
            );
            if (error > maximum_error) {
                maximum_error = error;
            }
        }
    }

    gs_baseline_pca_free(&model);
    return !isfinite(maximum_error) || maximum_error > 2.0e-6;
}


static int gs_test_threshold_component_counts(void)
{
    const double variances[4] = {70.0, 21.0, 8.5, 0.5};
    float x[32];
    GSBaselinePCA model = {0};
    size_t count_90 = 0;
    size_t count_95 = 0;
    size_t count_99 = 0;
    int failed;

    gs_test_make_axis_dataset(variances, 4, x);
    if (gs_baseline_pca_fit(x, 8, 4, &model) !=
            GS_BASELINE_PCA_SUCCESS ||
        gs_baseline_pca_components_for_threshold(
            &model, 0.90, &count_90
        ) != GS_BASELINE_PCA_SUCCESS ||
        gs_baseline_pca_components_for_threshold(
            &model, 0.95, &count_95
        ) != GS_BASELINE_PCA_SUCCESS ||
        gs_baseline_pca_components_for_threshold(
            &model, 0.99, &count_99
        ) != GS_BASELINE_PCA_SUCCESS) {

        gs_baseline_pca_free(&model);
        return 1;
    }

    failed = count_90 != 2 ||
             count_95 != 3 ||
             count_99 != 3 ||
             model.components_90 != 2 ||
             model.components_95 != 3 ||
             model.components_99 != 3;

    gs_baseline_pca_free(&model);
    return failed;
}


static int gs_test_deterministic_repeat(void)
{
    GSBaselinePCA first = {0};
    GSBaselinePCA second = {0};
    int failed;

    if (gs_baseline_pca_fit(gs_test_data, 6, 3, &first) !=
            GS_BASELINE_PCA_SUCCESS ||
        gs_baseline_pca_fit(gs_test_data, 6, 3, &second) !=
            GS_BASELINE_PCA_SUCCESS) {

        gs_baseline_pca_free(&second);
        gs_baseline_pca_free(&first);
        return 1;
    }

    failed = !gs_test_models_match_exactly(&first, &second);
    gs_baseline_pca_free(&second);
    gs_baseline_pca_free(&first);
    return failed;
}


static int gs_test_sign_convention(void)
{
    GSBaselinePCA model = {0};
    size_t component;
    int failed = 0;

    if (gs_baseline_pca_fit(gs_test_data, 6, 3, &model) !=
        GS_BASELINE_PCA_SUCCESS) {

        return 1;
    }

    for (component = 0; component < model.component_count; component++) {
        size_t maximum_feature = 0;
        size_t feature;
        double maximum_abs =
            fabs(model.components[component * model.input_dim]);

        for (feature = 1; feature < model.input_dim; feature++) {
            double loading_abs = fabs(
                model.components[
                    component * model.input_dim + feature
                ]
            );

            if (loading_abs > maximum_abs) {
                maximum_abs = loading_abs;
                maximum_feature = feature;
            }
        }

        if (!(maximum_abs > 0.0) ||
            model.components[
                component * model.input_dim + maximum_feature
            ] <= 0.0) {

            failed = 1;
        }
    }

    gs_baseline_pca_free(&model);
    return failed;
}


static int gs_test_train_only_transform_contract(void)
{
    const float validation[9] = {
        10.0f, -5.0f, 3.0f,
         8.0f, -4.0f, 2.0f,
        12.0f, -6.0f, 4.0f
    };
    GSBaselinePCA model = {0};
    double mean_snapshot[3];
    double components_snapshot[9];
    double eigenvalues_snapshot[3];
    double ratio_snapshot[3];
    double cumulative_snapshot[3];
    double total_variance_snapshot;
    size_t components_90_snapshot;
    size_t components_95_snapshot;
    size_t components_99_snapshot;
    float train_scores[18];
    float validation_scores[9];
    size_t row;
    size_t component;
    size_t feature;
    int failed = 0;

    if (gs_baseline_pca_fit(gs_test_data, 6, 3, &model) !=
        GS_BASELINE_PCA_SUCCESS) {

        return 1;
    }

    memcpy(mean_snapshot, model.mean, sizeof(mean_snapshot));
    memcpy(
        components_snapshot,
        model.components,
        sizeof(components_snapshot)
    );
    memcpy(
        eigenvalues_snapshot,
        model.eigenvalues,
        sizeof(eigenvalues_snapshot)
    );
    memcpy(
        ratio_snapshot,
        model.explained_variance_ratio,
        sizeof(ratio_snapshot)
    );
    memcpy(
        cumulative_snapshot,
        model.cumulative_explained_variance_ratio,
        sizeof(cumulative_snapshot)
    );
    total_variance_snapshot = model.total_variance;
    components_90_snapshot = model.components_90;
    components_95_snapshot = model.components_95;
    components_99_snapshot = model.components_99;

    if (gs_baseline_pca_transform(
            &model,
            gs_test_data,
            6,
            3,
            3,
            train_scores,
            18
        ) != GS_BASELINE_PCA_SUCCESS ||
        gs_baseline_pca_transform(
            &model,
            validation,
            3,
            3,
            3,
            validation_scores,
            9
        ) != GS_BASELINE_PCA_SUCCESS) {

        gs_baseline_pca_free(&model);
        return 1;
    }

    if (memcmp(mean_snapshot, model.mean, sizeof(mean_snapshot)) != 0 ||
        memcmp(
            components_snapshot,
            model.components,
            sizeof(components_snapshot)
        ) != 0 ||
        memcmp(
            eigenvalues_snapshot,
            model.eigenvalues,
            sizeof(eigenvalues_snapshot)
        ) != 0 ||
        memcmp(
            ratio_snapshot,
            model.explained_variance_ratio,
            sizeof(ratio_snapshot)
        ) != 0 ||
        memcmp(
            cumulative_snapshot,
            model.cumulative_explained_variance_ratio,
            sizeof(cumulative_snapshot)
        ) != 0 ||
        memcmp(
            &total_variance_snapshot,
            &model.total_variance,
            sizeof(double)
        ) != 0 ||
        components_90_snapshot != model.components_90 ||
        components_95_snapshot != model.components_95 ||
        components_99_snapshot != model.components_99) {

        failed = 1;
    }

    for (row = 0; row < 3; row++) {
        for (component = 0; component < 3; component++) {
            double expected = 0.0;

            for (feature = 0; feature < 3; feature++) {
                expected +=
                    ((double)validation[row * 3 + feature] -
                     mean_snapshot[feature]) *
                    components_snapshot[component * 3 + feature];
            }

            if (validation_scores[row * 3 + component] !=
                (float)expected) {

                failed = 1;
            }
        }
    }

    gs_baseline_pca_free(&model);
    return failed;
}


static int gs_test_transform_prefixes(void)
{
    GSBaselinePCA model = {0};
    float full_scores[18];
    float prefix_scores[12];
    size_t row;
    size_t component;
    int failed = 0;

    if (gs_baseline_pca_fit(gs_test_data, 6, 3, &model) !=
            GS_BASELINE_PCA_SUCCESS ||
        gs_baseline_pca_transform(
            &model,
            gs_test_data,
            6,
            3,
            3,
            full_scores,
            18
        ) != GS_BASELINE_PCA_SUCCESS ||
        gs_baseline_pca_transform(
            &model,
            gs_test_data,
            6,
            3,
            2,
            prefix_scores,
            12
        ) != GS_BASELINE_PCA_SUCCESS) {

        gs_baseline_pca_free(&model);
        return 1;
    }

    for (row = 0; row < 6; row++) {
        for (component = 0; component < 2; component++) {
            if (prefix_scores[row * 2 + component] !=
                full_scores[row * 3 + component]) {

                failed = 1;
            }
        }
    }

    gs_baseline_pca_free(&model);
    return failed;
}


static int gs_test_zero_variance_rejected(void)
{
    const float x[12] = {
        2.0f, -1.0f, 3.0f,
        2.0f, -1.0f, 3.0f,
        2.0f, -1.0f, 3.0f,
        2.0f, -1.0f, 3.0f
    };
    GSBaselinePCA model = {0};
    int status = gs_baseline_pca_fit(x, 4, 3, &model);
    int failed =
        status == GS_BASELINE_PCA_SUCCESS ||
        !gs_test_model_is_zero(&model);

    gs_baseline_pca_free(&model);
    return failed;
}


static int gs_test_invalid_arguments_and_nonfinite(void)
{
    float nan_fit[6] = {1.0f, 2.0f, NAN, 4.0f, 5.0f, 6.0f};
    float inf_fit[6] = {1.0f, 2.0f, 3.0f, INFINITY, 5.0f, 6.0f};
    float nan_transform[3] = {1.0f, NAN, 2.0f};
    float inf_transform[3] = {1.0f, INFINITY, 2.0f};
    float scores[18];
    size_t component_count = 0;
    GSBaselinePCA empty = {0};
    GSBaselinePCA valid = {0};
    GSBaselinePCA invalid = {0};

    if (gs_baseline_pca_fit(gs_test_data, 6, 3, &valid) !=
        GS_BASELINE_PCA_SUCCESS) {

        return 1;
    }

    invalid.input_dim = 3;
    invalid.component_count = 3;

    if (gs_baseline_pca_fit(NULL, 6, 3, &empty) ==
            GS_BASELINE_PCA_SUCCESS ||
        gs_baseline_pca_fit(gs_test_data, 6, 3, NULL) ==
            GS_BASELINE_PCA_SUCCESS ||
        gs_baseline_pca_fit(gs_test_data, 1, 3, &empty) ==
            GS_BASELINE_PCA_SUCCESS ||
        gs_baseline_pca_fit(gs_test_data, 6, 0, &empty) ==
            GS_BASELINE_PCA_SUCCESS ||
        gs_baseline_pca_fit(
            gs_test_data, SIZE_MAX, 2, &empty
        ) != GS_BASELINE_PCA_ERROR_OVERFLOW ||
        gs_baseline_pca_fit(
            gs_test_data, 2, (size_t)INT_MAX + 1, &empty
        ) == GS_BASELINE_PCA_SUCCESS ||
        gs_baseline_pca_fit(nan_fit, 2, 3, &empty) !=
            GS_BASELINE_PCA_ERROR_NONFINITE ||
        gs_baseline_pca_fit(inf_fit, 2, 3, &empty) !=
            GS_BASELINE_PCA_ERROR_NONFINITE ||
        !gs_test_model_is_zero(&empty) ||
        gs_baseline_pca_transform(
            NULL, gs_test_data, 6, 3, 3, scores, 18
        ) == GS_BASELINE_PCA_SUCCESS ||
        gs_baseline_pca_transform(
            &valid, NULL, 6, 3, 3, scores, 18
        ) == GS_BASELINE_PCA_SUCCESS ||
        gs_baseline_pca_transform(
            &valid, gs_test_data, 6, 3, 3, NULL, 18
        ) == GS_BASELINE_PCA_SUCCESS ||
        gs_baseline_pca_transform(
            &invalid, gs_test_data, 6, 3, 3, scores, 18
        ) != GS_BASELINE_PCA_ERROR_MODEL ||
        gs_baseline_pca_transform(
            &valid, gs_test_data, 0, 3, 3, scores, 18
        ) == GS_BASELINE_PCA_SUCCESS ||
        gs_baseline_pca_transform(
            &valid, gs_test_data, 6, 0, 3, scores, 18
        ) == GS_BASELINE_PCA_SUCCESS ||
        gs_baseline_pca_transform(
            &valid, gs_test_data, 6, 2, 2, scores, 12
        ) == GS_BASELINE_PCA_SUCCESS ||
        gs_baseline_pca_transform(
            &valid, gs_test_data, 6, 3, 0, scores, 18
        ) == GS_BASELINE_PCA_SUCCESS ||
        gs_baseline_pca_transform(
            &valid, gs_test_data, 6, 3, 4, scores, 24
        ) == GS_BASELINE_PCA_SUCCESS ||
        gs_baseline_pca_transform(
            &valid, gs_test_data, 6, 3, 3, scores, 17
        ) != GS_BASELINE_PCA_ERROR_BUFFER ||
        gs_baseline_pca_transform(
            &valid, gs_test_data, SIZE_MAX, 3, 2, scores, SIZE_MAX
        ) != GS_BASELINE_PCA_ERROR_OVERFLOW ||
        gs_baseline_pca_transform(
            &valid, nan_transform, 1, 3, 3, scores, 3
        ) != GS_BASELINE_PCA_ERROR_NONFINITE ||
        gs_baseline_pca_transform(
            &valid, inf_transform, 1, 3, 3, scores, 3
        ) != GS_BASELINE_PCA_ERROR_NONFINITE ||
        gs_baseline_pca_components_for_threshold(
            NULL, 0.90, &component_count
        ) == GS_BASELINE_PCA_SUCCESS ||
        gs_baseline_pca_components_for_threshold(
            &valid, 0.90, NULL
        ) == GS_BASELINE_PCA_SUCCESS ||
        gs_baseline_pca_components_for_threshold(
            &invalid, 0.90, &component_count
        ) != GS_BASELINE_PCA_ERROR_MODEL ||
        gs_baseline_pca_components_for_threshold(
            &valid, 0.0, &component_count
        ) == GS_BASELINE_PCA_SUCCESS ||
        gs_baseline_pca_components_for_threshold(
            &valid, 1.01, &component_count
        ) == GS_BASELINE_PCA_SUCCESS ||
        gs_baseline_pca_components_for_threshold(
            &valid, NAN, &component_count
        ) == GS_BASELINE_PCA_SUCCESS) {

        gs_baseline_pca_free(&valid);
        return 1;
    }

    gs_baseline_pca_free(&valid);
    return 0;
}


static int gs_test_input_not_modified(void)
{
    float fit_input[18];
    float fit_copy[18];
    float transform_input[18];
    float transform_copy[18];
    float scores[18];
    GSBaselinePCA model = {0};
    int failed;

    memcpy(fit_input, gs_test_data, sizeof(fit_input));
    memcpy(fit_copy, fit_input, sizeof(fit_copy));
    memcpy(transform_input, gs_test_data, sizeof(transform_input));
    memcpy(transform_copy, transform_input, sizeof(transform_copy));

    if (gs_baseline_pca_fit(fit_input, 6, 3, &model) !=
            GS_BASELINE_PCA_SUCCESS ||
        gs_baseline_pca_transform(
            &model,
            transform_input,
            6,
            3,
            3,
            scores,
            18
        ) != GS_BASELINE_PCA_SUCCESS) {

        gs_baseline_pca_free(&model);
        return 1;
    }

    failed = memcmp(fit_input, fit_copy, sizeof(fit_input)) != 0 ||
             memcmp(
                 transform_input,
                 transform_copy,
                 sizeof(transform_input)
             ) != 0;

    gs_baseline_pca_free(&model);
    return failed;
}


static int gs_test_free_idempotent(void)
{
    GSBaselinePCA model = {0};

    if (gs_baseline_pca_fit(gs_test_data, 6, 3, &model) !=
        GS_BASELINE_PCA_SUCCESS) {

        return 1;
    }

    gs_baseline_pca_free(&model);
    gs_baseline_pca_free(&model);
    gs_baseline_pca_free(NULL);
    return !gs_test_model_is_zero(&model);
}


int main(void)
{
#define GS_RUN_TEST(function_name, display_name)                         \
    do {                                                                 \
        if (function_name()) {                                           \
            fprintf(stderr, "[FAIL] %s\n", display_name);               \
            return EXIT_FAILURE;                                         \
        }                                                                \
        printf("[PASS] %s\n", display_name);                            \
    } while (0)

    GS_RUN_TEST(
        gs_test_known_diagonal_covariance,
        "known_diagonal_covariance"
    );
    GS_RUN_TEST(
        gs_test_covariance_reconstruction,
        "covariance_reconstruction"
    );
    GS_RUN_TEST(
        gs_test_full_component_reconstruction,
        "full_component_reconstruction"
    );
    GS_RUN_TEST(
        gs_test_threshold_component_counts,
        "threshold_component_counts"
    );
    GS_RUN_TEST(gs_test_deterministic_repeat, "deterministic_repeat");
    GS_RUN_TEST(gs_test_sign_convention, "sign_convention");
    GS_RUN_TEST(
        gs_test_train_only_transform_contract,
        "train_only_transform_contract"
    );
    GS_RUN_TEST(gs_test_transform_prefixes, "transform_prefixes");
    GS_RUN_TEST(
        gs_test_zero_variance_rejected,
        "zero_variance_rejected"
    );
    GS_RUN_TEST(
        gs_test_invalid_arguments_and_nonfinite,
        "invalid_arguments_and_nonfinite"
    );
    GS_RUN_TEST(gs_test_input_not_modified, "input_not_modified");
    GS_RUN_TEST(gs_test_free_idempotent, "free_idempotent");

#undef GS_RUN_TEST

    printf("PCA_BACKEND=LAPACKE_dsyevd\n");
    printf("PCA_DDOF=0\n");
    printf("PCA_WHITENING=false\n");
    printf("ALL TESTS PASSED\n");
    return EXIT_SUCCESS;
}
