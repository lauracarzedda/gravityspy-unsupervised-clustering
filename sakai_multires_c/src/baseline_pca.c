#include "baseline_pca.h"

#include <lapacke.h>

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>


static void gs_pca_model_zero(GSBaselinePCA *model)
{
    if (model == NULL) {
        return;
    }

    model->input_dim = 0;
    model->component_count = 0;
    model->mean = NULL;
    model->components = NULL;
    model->eigenvalues = NULL;
    model->explained_variance_ratio = NULL;
    model->cumulative_explained_variance_ratio = NULL;
    model->total_variance = 0.0;
    model->components_90 = 0;
    model->components_95 = 0;
    model->components_99 = 0;
}


void gs_baseline_pca_free(GSBaselinePCA *model)
{
    if (model == NULL) {
        return;
    }

    free(model->cumulative_explained_variance_ratio);
    free(model->explained_variance_ratio);
    free(model->eigenvalues);
    free(model->components);
    free(model->mean);
    gs_pca_model_zero(model);
}


static int gs_pca_checked_product(
    size_t left,
    size_t right,
    size_t *product
)
{
    if (product == NULL || (right != 0 && left > SIZE_MAX / right)) {
        return 0;
    }

    *product = left * right;
    return 1;
}


static size_t gs_pca_threshold_count(
    const double *cumulative,
    size_t count,
    double threshold
)
{
    size_t component;

    for (component = 0; component < count; component++) {
        if (cumulative[component] >= threshold) {
            return component + 1;
        }
    }

    return count;
}


static int gs_pca_model_is_empty(const GSBaselinePCA *model)
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


static int gs_pca_model_is_consistent(const GSBaselinePCA *model)
{
    size_t matrix_count;
    size_t vector_bytes;
    size_t matrix_bytes;
    size_t component;
    size_t feature;
    double previous_eigenvalue = INFINITY;
    double previous_cumulative = 0.0;
    double final_tolerance;

    if (model == NULL ||
        model->input_dim == 0 ||
        model->component_count != model->input_dim ||
        model->input_dim > (size_t)INT_MAX ||
        model->mean == NULL ||
        model->components == NULL ||
        model->eigenvalues == NULL ||
        model->explained_variance_ratio == NULL ||
        model->cumulative_explained_variance_ratio == NULL ||
        !isfinite(model->total_variance) ||
        model->total_variance <= 0.0 ||
        model->components_90 == 0 ||
        model->components_90 > model->component_count ||
        model->components_95 == 0 ||
        model->components_95 > model->component_count ||
        model->components_99 == 0 ||
        model->components_99 > model->component_count ||
        !gs_pca_checked_product(
            model->input_dim,
            model->input_dim,
            &matrix_count
        ) ||
        !gs_pca_checked_product(
            model->input_dim,
            sizeof(double),
            &vector_bytes
        ) ||
        !gs_pca_checked_product(
            matrix_count,
            sizeof(double),
            &matrix_bytes
        )) {

        return 0;
    }

    (void)vector_bytes;
    (void)matrix_bytes;
    final_tolerance =
        1024.0 * DBL_EPSILON * (double)model->component_count;

    for (feature = 0; feature < model->input_dim; feature++) {
        if (!isfinite(model->mean[feature])) {
            return 0;
        }
    }

    for (component = 0;
         component < model->component_count;
         component++) {

        double eigenvalue = model->eigenvalues[component];
        double ratio = model->explained_variance_ratio[component];
        double cumulative =
            model->cumulative_explained_variance_ratio[component];

        if (!isfinite(eigenvalue) ||
            eigenvalue < 0.0 ||
            eigenvalue > previous_eigenvalue ||
            !isfinite(ratio) ||
            ratio < 0.0 ||
            ratio > 1.0 + final_tolerance ||
            !isfinite(cumulative) ||
            cumulative < previous_cumulative ||
            cumulative > 1.0 + final_tolerance) {

            return 0;
        }

        previous_eigenvalue = eigenvalue;
        previous_cumulative = cumulative;

        for (feature = 0; feature < model->input_dim; feature++) {
            if (!isfinite(
                    model->components[
                        component * model->input_dim + feature
                    ]
                )) {

                return 0;
            }
        }
    }

    if (fabs(previous_cumulative - 1.0) > final_tolerance ||
        gs_pca_threshold_count(
            model->cumulative_explained_variance_ratio,
            model->component_count,
            0.90
        ) != model->components_90 ||
        gs_pca_threshold_count(
            model->cumulative_explained_variance_ratio,
            model->component_count,
            0.95
        ) != model->components_95 ||
        gs_pca_threshold_count(
            model->cumulative_explained_variance_ratio,
            model->component_count,
            0.99
        ) != model->components_99) {

        return 0;
    }

    return 1;
}


int gs_baseline_pca_fit(
    const float *x,
    size_t rows,
    size_t cols,
    GSBaselinePCA *model
)
{
    GSBaselinePCA fitted;
    double *centered = NULL;
    double *covariance = NULL;
    double *lapack_eigenvalues = NULL;
    size_t input_count;
    size_t matrix_count;
    size_t vector_bytes;
    size_t matrix_bytes;
    size_t centered_bytes;
    size_t row;
    size_t feature;
    size_t other_feature;
    size_t component;
    double max_abs_eigenvalue = 0.0;
    double negative_tolerance;
    double orthonormality_tolerance;
    double total_variance = 0.0;
    double cumulative = 0.0;
    lapack_int lapack_cols;
    lapack_int info;
    int error = GS_BASELINE_PCA_ERROR_NUMERIC;

    if (x == NULL || model == NULL) {
        return GS_BASELINE_PCA_ERROR_NULL_ARGUMENT;
    }
    if (rows < 2 || cols == 0 || cols > (size_t)INT_MAX) {
        return GS_BASELINE_PCA_ERROR_DIMENSIONS;
    }
    if (!gs_pca_model_is_empty(model) &&
        !gs_pca_model_is_consistent(model)) {

        return GS_BASELINE_PCA_ERROR_MODEL;
    }
    if (!gs_pca_checked_product(rows, cols, &input_count) ||
        !gs_pca_checked_product(cols, cols, &matrix_count) ||
        !gs_pca_checked_product(
            input_count,
            sizeof(double),
            &centered_bytes
        ) ||
        !gs_pca_checked_product(
            cols,
            sizeof(double),
            &vector_bytes
        ) ||
        !gs_pca_checked_product(
            matrix_count,
            sizeof(double),
            &matrix_bytes
        )) {

        return GS_BASELINE_PCA_ERROR_OVERFLOW;
    }

    gs_pca_model_zero(&fitted);
    fitted.input_dim = cols;
    fitted.component_count = cols;

    fitted.mean = (double *)calloc(cols, sizeof(double));
    fitted.components = (double *)malloc(matrix_bytes);
    fitted.eigenvalues = (double *)malloc(vector_bytes);
    fitted.explained_variance_ratio = (double *)malloc(vector_bytes);
    fitted.cumulative_explained_variance_ratio =
        (double *)malloc(vector_bytes);
    centered = (double *)malloc(centered_bytes);
    covariance = (double *)calloc(matrix_count, sizeof(double));
    lapack_eigenvalues = (double *)malloc(vector_bytes);

    if (fitted.mean == NULL ||
        fitted.components == NULL ||
        fitted.eigenvalues == NULL ||
        fitted.explained_variance_ratio == NULL ||
        fitted.cumulative_explained_variance_ratio == NULL ||
        centered == NULL ||
        covariance == NULL ||
        lapack_eigenvalues == NULL) {

        error = GS_BASELINE_PCA_ERROR_ALLOCATION;
        goto cleanup;
    }

    for (row = 0; row < rows; row++) {
        for (feature = 0; feature < cols; feature++) {
            double value = (double)x[row * cols + feature];
            double updated_sum;

            if (!isfinite(value)) {
                error = GS_BASELINE_PCA_ERROR_NONFINITE;
                goto cleanup;
            }

            updated_sum = fitted.mean[feature] + value;
            if (!isfinite(updated_sum)) {
                error = GS_BASELINE_PCA_ERROR_NUMERIC;
                goto cleanup;
            }
            fitted.mean[feature] = updated_sum;
        }
    }

    for (feature = 0; feature < cols; feature++) {
        fitted.mean[feature] /= (double)rows;
        if (!isfinite(fitted.mean[feature])) {
            error = GS_BASELINE_PCA_ERROR_NUMERIC;
            goto cleanup;
        }
    }

    for (row = 0; row < rows; row++) {
        for (feature = 0; feature < cols; feature++) {
            double value =
                (double)x[row * cols + feature] - fitted.mean[feature];

            if (!isfinite(value)) {
                error = GS_BASELINE_PCA_ERROR_NUMERIC;
                goto cleanup;
            }
            centered[row * cols + feature] = value;
        }
    }

    for (row = 0; row < rows; row++) {
        const double *centered_row = centered + row * cols;

        for (feature = 0; feature < cols; feature++) {
            double left = centered_row[feature];

            for (other_feature = feature;
                 other_feature < cols;
                 other_feature++) {

                size_t index = feature * cols + other_feature;
                double updated =
                    covariance[index] +
                    left * centered_row[other_feature];

                if (!isfinite(updated)) {
                    error = GS_BASELINE_PCA_ERROR_NUMERIC;
                    goto cleanup;
                }
                covariance[index] = updated;
            }
        }
    }

    for (feature = 0; feature < cols; feature++) {
        for (other_feature = feature;
             other_feature < cols;
             other_feature++) {

            double value =
                covariance[feature * cols + other_feature] / (double)rows;

            if (!isfinite(value)) {
                error = GS_BASELINE_PCA_ERROR_NUMERIC;
                goto cleanup;
            }
            covariance[feature * cols + other_feature] = value;
            covariance[other_feature * cols + feature] = value;
        }
    }

    free(centered);
    centered = NULL;

    lapack_cols = (lapack_int)cols;
    info = LAPACKE_dsyevd(
        LAPACK_COL_MAJOR,
        'V',
        'U',
        lapack_cols,
        covariance,
        lapack_cols,
        lapack_eigenvalues
    );
    if (info != 0) {
        error = GS_BASELINE_PCA_ERROR_LAPACKE;
        goto cleanup;
    }

    for (component = 0; component < cols; component++) {
        double eigenvalue = lapack_eigenvalues[component];

        if (!isfinite(eigenvalue)) {
            error = GS_BASELINE_PCA_ERROR_NONFINITE;
            goto cleanup;
        }
        if (fabs(eigenvalue) > max_abs_eigenvalue) {
            max_abs_eigenvalue = fabs(eigenvalue);
        }

        for (feature = 0; feature < cols; feature++) {
            if (!isfinite(covariance[feature + component * cols])) {
                error = GS_BASELINE_PCA_ERROR_NONFINITE;
                goto cleanup;
            }
        }
    }

    orthonormality_tolerance =
        1024.0 * DBL_EPSILON * (double)cols;
    for (component = 0; component < cols; component++) {
        size_t other_component;

        for (other_component = 0;
             other_component < cols;
             other_component++) {

            double dot_product = 0.0;
            double expected =
                component == other_component ? 1.0 : 0.0;

            for (feature = 0; feature < cols; feature++) {
                dot_product +=
                    covariance[feature + component * cols] *
                    covariance[feature + other_component * cols];
            }

            if (!isfinite(dot_product) ||
                fabs(dot_product - expected) >
                    orthonormality_tolerance) {

                error = GS_BASELINE_PCA_ERROR_NUMERIC;
                goto cleanup;
            }
        }
    }

    negative_tolerance =
        1024.0 * DBL_EPSILON * (double)cols * max_abs_eigenvalue;
    if (!isfinite(negative_tolerance)) {
        error = GS_BASELINE_PCA_ERROR_NUMERIC;
        goto cleanup;
    }

    for (component = 0; component < cols; component++) {
        if (lapack_eigenvalues[component] < -negative_tolerance) {
            error = GS_BASELINE_PCA_ERROR_NUMERIC;
            goto cleanup;
        }
        if (lapack_eigenvalues[component] < 0.0) {
            lapack_eigenvalues[component] = 0.0;
        }
        total_variance += lapack_eigenvalues[component];
        if (!isfinite(total_variance)) {
            error = GS_BASELINE_PCA_ERROR_NUMERIC;
            goto cleanup;
        }
    }
    if (!(total_variance > 0.0)) {
        error = GS_BASELINE_PCA_ERROR_NUMERIC;
        goto cleanup;
    }
    fitted.total_variance = total_variance;

    for (component = 0; component < cols; component++) {
        size_t source_component = cols - 1 - component;
        size_t maximum_feature = 0;
        double maximum_loading_abs;

        fitted.eigenvalues[component] =
            lapack_eigenvalues[source_component];

        for (feature = 0; feature < cols; feature++) {
            fitted.components[component * cols + feature] =
                covariance[feature + source_component * cols];
        }

        maximum_loading_abs =
            fabs(fitted.components[component * cols]);
        for (feature = 1; feature < cols; feature++) {
            double loading_abs =
                fabs(fitted.components[component * cols + feature]);

            if (loading_abs > maximum_loading_abs) {
                maximum_loading_abs = loading_abs;
                maximum_feature = feature;
            }
        }

        if (fitted.components[
                component * cols + maximum_feature
            ] < 0.0) {

            for (feature = 0; feature < cols; feature++) {
                fitted.components[component * cols + feature] =
                    -fitted.components[component * cols + feature];
            }
        }

        fitted.explained_variance_ratio[component] =
            fitted.eigenvalues[component] / fitted.total_variance;
        cumulative += fitted.explained_variance_ratio[component];

        if (!isfinite(fitted.explained_variance_ratio[component]) ||
            !isfinite(cumulative)) {

            error = GS_BASELINE_PCA_ERROR_NUMERIC;
            goto cleanup;
        }
        fitted.cumulative_explained_variance_ratio[component] =
            cumulative;
    }

    fitted.cumulative_explained_variance_ratio[cols - 1] = 1.0;
    fitted.components_90 = gs_pca_threshold_count(
        fitted.cumulative_explained_variance_ratio,
        cols,
        0.90
    );
    fitted.components_95 = gs_pca_threshold_count(
        fitted.cumulative_explained_variance_ratio,
        cols,
        0.95
    );
    fitted.components_99 = gs_pca_threshold_count(
        fitted.cumulative_explained_variance_ratio,
        cols,
        0.99
    );

    if (!gs_pca_model_is_consistent(&fitted)) {
        error = GS_BASELINE_PCA_ERROR_NUMERIC;
        goto cleanup;
    }

    free(lapack_eigenvalues);
    free(covariance);
    gs_baseline_pca_free(model);
    *model = fitted;
    return GS_BASELINE_PCA_SUCCESS;

cleanup:
    free(lapack_eigenvalues);
    free(covariance);
    free(centered);
    gs_baseline_pca_free(&fitted);
    return error;
}


int gs_baseline_pca_transform(
    const GSBaselinePCA *model,
    const float *x,
    size_t rows,
    size_t cols,
    size_t requested_components,
    float *scores,
    size_t scores_count
)
{
    size_t input_count;
    size_t output_count;
    size_t input_bytes;
    size_t output_bytes;
    size_t row;
    size_t component;
    size_t feature;

    if (model == NULL || x == NULL || scores == NULL) {
        return GS_BASELINE_PCA_ERROR_NULL_ARGUMENT;
    }
    if (!gs_pca_model_is_consistent(model)) {
        return GS_BASELINE_PCA_ERROR_MODEL;
    }
    if (rows == 0 ||
        cols == 0 ||
        cols != model->input_dim ||
        requested_components == 0 ||
        requested_components > model->input_dim) {

        return GS_BASELINE_PCA_ERROR_DIMENSIONS;
    }
    if (!gs_pca_checked_product(rows, cols, &input_count) ||
        !gs_pca_checked_product(
            rows,
            requested_components,
            &output_count
        ) ||
        !gs_pca_checked_product(
            input_count,
            sizeof(float),
            &input_bytes
        ) ||
        !gs_pca_checked_product(
            output_count,
            sizeof(float),
            &output_bytes
        )) {

        return GS_BASELINE_PCA_ERROR_OVERFLOW;
    }
    (void)input_bytes;
    (void)output_bytes;

    if (scores_count < output_count) {
        return GS_BASELINE_PCA_ERROR_BUFFER;
    }

    for (row = 0; row < rows; row++) {
        for (feature = 0; feature < cols; feature++) {
            if (!isfinite((double)x[row * cols + feature])) {
                return GS_BASELINE_PCA_ERROR_NONFINITE;
            }
        }
    }

    for (row = 0; row < rows; row++) {
        for (component = 0;
             component < requested_components;
             component++) {

            double score = 0.0;

            for (feature = 0; feature < cols; feature++) {
                score +=
                    ((double)x[row * cols + feature] -
                     model->mean[feature]) *
                    model->components[
                        component * model->input_dim + feature
                    ];
            }

            if (!isfinite(score) ||
                score > (double)FLT_MAX ||
                score < -(double)FLT_MAX) {

                return GS_BASELINE_PCA_ERROR_NUMERIC;
            }
            scores[row * requested_components + component] =
                (float)score;
        }
    }

    return GS_BASELINE_PCA_SUCCESS;
}


int gs_baseline_pca_components_for_threshold(
    const GSBaselinePCA *model,
    double threshold,
    size_t *component_count
)
{
    if (model == NULL || component_count == NULL) {
        return GS_BASELINE_PCA_ERROR_NULL_ARGUMENT;
    }
    if (!gs_pca_model_is_consistent(model)) {
        return GS_BASELINE_PCA_ERROR_MODEL;
    }
    if (!isfinite(threshold) || threshold <= 0.0 || threshold > 1.0) {
        return GS_BASELINE_PCA_ERROR_DIMENSIONS;
    }

    *component_count = gs_pca_threshold_count(
        model->cumulative_explained_variance_ratio,
        model->component_count,
        threshold
    );
    return GS_BASELINE_PCA_SUCCESS;
}
