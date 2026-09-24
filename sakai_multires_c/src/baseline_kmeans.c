#include "baseline_kmeans.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>


static void gs_kmeans_result_zero(GSBaselineKMeansResult *result)
{
    if (result == NULL) {
        return;
    }

    result->rows = 0;
    result->cols = 0;
    result->k = 0;
    result->iterations = 0;
    result->empty_cluster_reseeds = 0;
    result->converged = 0;
    result->seed = UINT64_C(0);
    result->initial_inertia = 0.0;
    result->inertia = 0.0;
    result->centroids = NULL;
    result->assignments = NULL;
    result->cluster_sizes = NULL;
    result->initial_centroid_indices = NULL;
}


void gs_baseline_kmeans_result_free(GSBaselineKMeansResult *result)
{
    if (result == NULL) {
        return;
    }

    free(result->initial_centroid_indices);
    free(result->cluster_sizes);
    free(result->assignments);
    free(result->centroids);

    gs_kmeans_result_zero(result);
}


static int gs_kmeans_checked_product(
    size_t left,
    size_t right,
    size_t *product
)
{
    if (product == NULL) {
        return 0;
    }

    if (right != 0 && left > SIZE_MAX / right) {
        return 0;
    }

    *product = left * right;
    return 1;
}


static uint64_t gs_kmeans_splitmix64_next(uint64_t *state)
{
    uint64_t value;

    *state += UINT64_C(0x9e3779b97f4a7c15);
    value = *state;
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}


static double gs_kmeans_uniform01(uint64_t *state)
{
    const double two_to_53 = 9007199254740992.0;
    uint64_t high_53_bits = gs_kmeans_splitmix64_next(state) >> 11;

    return (double)high_53_bits / two_to_53;
}


static int gs_kmeans_input_is_finite(
    const float *x,
    size_t value_count
)
{
    size_t index;

    for (index = 0; index < value_count; index++) {
        if (!isfinite(x[index])) {
            return 0;
        }
    }

    return 1;
}


static int gs_kmeans_centroids_are_finite(
    const double *centroids,
    size_t value_count
)
{
    size_t index;

    for (index = 0; index < value_count; index++) {
        if (!isfinite(centroids[index])) {
            return 0;
        }
    }

    return 1;
}


static int gs_kmeans_squared_distance(
    const float *row,
    const double *centroid,
    size_t cols,
    double *squared_distance
)
{
    size_t col;
    double total = 0.0;

    for (col = 0; col < cols; col++) {
        double difference = (double)row[col] - centroid[col];
        double contribution = difference * difference;

        if (!isfinite(difference) || !isfinite(contribution)) {
            return 0;
        }

        total += contribution;
        if (!isfinite(total)) {
            return 0;
        }
    }

    *squared_distance = total;
    return 1;
}


static int gs_kmeans_assign_unchecked(
    const float *x,
    size_t rows,
    size_t cols,
    const double *centroids,
    size_t k,
    size_t *assignments,
    double *squared_distances,
    size_t *changed_assignments
)
{
    size_t row;

    if (changed_assignments != NULL) {
        *changed_assignments = 0;
    }

    for (row = 0; row < rows; row++) {
        const float *current_row = x + row * cols;
        size_t best_cluster = 0;
        size_t cluster;
        double best_distance;

        if (!gs_kmeans_squared_distance(
                current_row,
                centroids,
                cols,
                &best_distance
            )) {

            return GS_BASELINE_KMEANS_ERROR_NUMERIC;
        }

        for (cluster = 1; cluster < k; cluster++) {
            double distance;

            if (!gs_kmeans_squared_distance(
                    current_row,
                    centroids + cluster * cols,
                    cols,
                    &distance
                )) {

                return GS_BASELINE_KMEANS_ERROR_NUMERIC;
            }

            if (distance < best_distance) {
                best_distance = distance;
                best_cluster = cluster;
            }
        }

        if (changed_assignments != NULL &&
            assignments[row] != best_cluster) {

            (*changed_assignments)++;
        }

        assignments[row] = best_cluster;
        if (squared_distances != NULL) {
            squared_distances[row] = best_distance;
        }
    }

    return GS_BASELINE_KMEANS_SUCCESS;
}


int gs_baseline_kmeans_assign(
    const float *x,
    size_t rows,
    size_t cols,
    const double *centroids,
    size_t k,
    size_t *assignments,
    double *squared_distances
)
{
    size_t input_count;
    size_t centroid_count;

    if (x == NULL || centroids == NULL || assignments == NULL) {
        return GS_BASELINE_KMEANS_ERROR_NULL_ARGUMENT;
    }

    if (rows == 0 || cols == 0 || k == 0) {
        return GS_BASELINE_KMEANS_ERROR_DIMENSIONS;
    }

    if (!gs_kmeans_checked_product(rows, cols, &input_count) ||
        !gs_kmeans_checked_product(k, cols, &centroid_count)) {

        return GS_BASELINE_KMEANS_ERROR_OVERFLOW;
    }

    if (!gs_kmeans_input_is_finite(x, input_count) ||
        !gs_kmeans_centroids_are_finite(centroids, centroid_count)) {

        return GS_BASELINE_KMEANS_ERROR_NONFINITE_INPUT;
    }

    return gs_kmeans_assign_unchecked(
        x,
        rows,
        cols,
        centroids,
        k,
        assignments,
        squared_distances,
        NULL
    );
}


static void gs_kmeans_copy_row_to_centroid(
    const float *x,
    size_t row_index,
    size_t cols,
    double *centroid
)
{
    size_t col;
    const float *row = x + row_index * cols;

    for (col = 0; col < cols; col++) {
        centroid[col] = (double)row[col];
    }
}


static int gs_kmeans_initialize_plus_plus(
    const float *x,
    size_t rows,
    size_t cols,
    size_t k,
    uint64_t *prng_state,
    double *centroids,
    size_t *initial_indices,
    unsigned char *selected,
    double *minimum_distances
)
{
    size_t first_index;
    size_t center_number;
    size_t row;
    double uniform = gs_kmeans_uniform01(prng_state);
    double scaled = uniform * (double)rows;

    first_index = (size_t)scaled;
    if (first_index >= rows) {
        first_index = rows - 1;
    }

    initial_indices[0] = first_index;
    selected[first_index] = 1u;
    gs_kmeans_copy_row_to_centroid(x, first_index, cols, centroids);

    for (row = 0; row < rows; row++) {
        if (!gs_kmeans_squared_distance(
                x + row * cols,
                centroids,
                cols,
                &minimum_distances[row]
            )) {

            return GS_BASELINE_KMEANS_ERROR_NUMERIC;
        }
    }

    for (center_number = 1; center_number < k; center_number++) {
        size_t selected_index = SIZE_MAX;
        size_t last_positive_index = SIZE_MAX;
        double total_weight = 0.0;

        for (row = 0; row < rows; row++) {
            if (selected[row] == 0u) {
                total_weight += minimum_distances[row];
                if (!isfinite(total_weight)) {
                    return GS_BASELINE_KMEANS_ERROR_NUMERIC;
                }
                if (minimum_distances[row] > 0.0) {
                    last_positive_index = row;
                }
            }
        }

        if (total_weight > 0.0) {
            double threshold = gs_kmeans_uniform01(prng_state) * total_weight;
            double cumulative = 0.0;

            for (row = 0; row < rows; row++) {
                if (selected[row] == 0u && minimum_distances[row] > 0.0) {
                    cumulative += minimum_distances[row];
                    if (!isfinite(cumulative)) {
                        return GS_BASELINE_KMEANS_ERROR_NUMERIC;
                    }

                    if (cumulative > threshold) {
                        selected_index = row;
                        break;
                    }
                }
            }

            if (selected_index == SIZE_MAX) {
                selected_index = last_positive_index;
            }
        } else {
            for (row = 0; row < rows; row++) {
                if (selected[row] == 0u) {
                    selected_index = row;
                    break;
                }
            }
        }

        if (selected_index == SIZE_MAX || selected[selected_index] != 0u) {
            return GS_BASELINE_KMEANS_ERROR_NUMERIC;
        }

        initial_indices[center_number] = selected_index;
        selected[selected_index] = 1u;
        gs_kmeans_copy_row_to_centroid(
            x,
            selected_index,
            cols,
            centroids + center_number * cols
        );

        for (row = 0; row < rows; row++) {
            double distance;

            if (!gs_kmeans_squared_distance(
                    x + row * cols,
                    centroids + center_number * cols,
                    cols,
                    &distance
                )) {

                return GS_BASELINE_KMEANS_ERROR_NUMERIC;
            }

            if (distance < minimum_distances[row]) {
                minimum_distances[row] = distance;
            }
        }
    }

    return GS_BASELINE_KMEANS_SUCCESS;
}


static int gs_kmeans_sum_distances(
    const double *distances,
    size_t rows,
    double *inertia
)
{
    size_t row;
    double total = 0.0;

    for (row = 0; row < rows; row++) {
        if (!isfinite(distances[row]) || distances[row] < 0.0) {
            return GS_BASELINE_KMEANS_ERROR_NUMERIC;
        }

        total += distances[row];
        if (!isfinite(total)) {
            return GS_BASELINE_KMEANS_ERROR_NUMERIC;
        }
    }

    *inertia = total;
    return GS_BASELINE_KMEANS_SUCCESS;
}


static int gs_kmeans_accumulate_and_update(
    const float *x,
    size_t rows,
    size_t cols,
    size_t k,
    GSBaselineKMeansResult *result,
    double *sums
)
{
    size_t centroid_count = k * cols;
    size_t row;
    size_t cluster;

    memset(sums, 0, centroid_count * sizeof(double));
    memset(result->cluster_sizes, 0, k * sizeof(size_t));

    for (row = 0; row < rows; row++) {
        size_t assigned_cluster = result->assignments[row];
        size_t col;

        if (assigned_cluster >= k) {
            return GS_BASELINE_KMEANS_ERROR_NUMERIC;
        }

        result->cluster_sizes[assigned_cluster]++;

        for (col = 0; col < cols; col++) {
            size_t index = assigned_cluster * cols + col;

            sums[index] += (double)x[row * cols + col];
            if (!isfinite(sums[index])) {
                return GS_BASELINE_KMEANS_ERROR_NUMERIC;
            }
        }
    }

    for (cluster = 0; cluster < k; cluster++) {
        size_t col;

        if (result->cluster_sizes[cluster] == 0) {
            continue;
        }

        for (col = 0; col < cols; col++) {
            double value =
                sums[cluster * cols + col] /
                (double)result->cluster_sizes[cluster];

            if (!isfinite(value)) {
                return GS_BASELINE_KMEANS_ERROR_NUMERIC;
            }

            result->centroids[cluster * cols + col] = value;
        }
    }

    return GS_BASELINE_KMEANS_SUCCESS;
}


static int gs_kmeans_reseed_empty_clusters(
    const float *x,
    size_t rows,
    size_t cols,
    size_t k,
    GSBaselineKMeansResult *result,
    unsigned char *row_already_used,
    size_t *remaining_cluster_sizes,
    size_t *reseeded_count
)
{
    size_t cluster;

    memset(row_already_used, 0, rows * sizeof(unsigned char));
    memcpy(
        remaining_cluster_sizes,
        result->cluster_sizes,
        k * sizeof(size_t)
    );
    *reseeded_count = 0;

    for (cluster = 0; cluster < k; cluster++) {
        size_t row;
        size_t candidate = SIZE_MAX;
        double candidate_distance = 0.0;

        if (result->cluster_sizes[cluster] != 0) {
            continue;
        }

        for (row = 0; row < rows; row++) {
            size_t source_cluster = result->assignments[row];
            double distance;

            if (source_cluster >= k || row_already_used[row] != 0u ||
                remaining_cluster_sizes[source_cluster] <= 1) {

                continue;
            }

            if (!gs_kmeans_squared_distance(
                    x + row * cols,
                    result->centroids + source_cluster * cols,
                    cols,
                    &distance
                )) {

                return GS_BASELINE_KMEANS_ERROR_NUMERIC;
            }

            if (distance > 0.0 &&
                (candidate == SIZE_MAX || distance > candidate_distance)) {

                candidate = row;
                candidate_distance = distance;
            }
        }

        if (candidate != SIZE_MAX) {
            size_t source_cluster = result->assignments[candidate];

            gs_kmeans_copy_row_to_centroid(
                x,
                candidate,
                cols,
                result->centroids + cluster * cols
            );
            row_already_used[candidate] = 1u;
            remaining_cluster_sizes[source_cluster]--;
            result->empty_cluster_reseeds++;
            (*reseeded_count)++;
        }
    }

    return GS_BASELINE_KMEANS_SUCCESS;
}


static int gs_kmeans_finalize_result(
    const float *x,
    size_t rows,
    size_t cols,
    size_t k,
    GSBaselineKMeansResult *result,
    double *distances
)
{
    size_t row;
    size_t cluster;
    size_t total_cluster_size = 0;
    int status;

    status = gs_kmeans_assign_unchecked(
        x,
        rows,
        cols,
        result->centroids,
        k,
        result->assignments,
        distances,
        NULL
    );
    if (status != GS_BASELINE_KMEANS_SUCCESS) {
        return status;
    }

    memset(result->cluster_sizes, 0, k * sizeof(size_t));
    for (row = 0; row < rows; row++) {
        if (result->assignments[row] >= k) {
            return GS_BASELINE_KMEANS_ERROR_NUMERIC;
        }
        result->cluster_sizes[result->assignments[row]]++;
    }

    for (cluster = 0; cluster < k; cluster++) {
        if (total_cluster_size > SIZE_MAX - result->cluster_sizes[cluster]) {
            return GS_BASELINE_KMEANS_ERROR_OVERFLOW;
        }
        total_cluster_size += result->cluster_sizes[cluster];
    }

    if (total_cluster_size != rows ||
        !gs_kmeans_centroids_are_finite(result->centroids, k * cols)) {

        return GS_BASELINE_KMEANS_ERROR_NUMERIC;
    }

    return gs_kmeans_sum_distances(distances, rows, &result->inertia);
}


int gs_baseline_kmeans_fit(
    const float *x,
    size_t rows,
    size_t cols,
    size_t k,
    uint64_t seed,
    size_t max_iterations,
    GSBaselineKMeansResult *result
)
{
    size_t input_count;
    size_t centroid_count;
    size_t input_bytes;
    size_t centroid_bytes;
    size_t row_size_bytes;
    size_t cluster_size_bytes;
    size_t row_double_bytes;
    size_t row_flag_bytes;
    size_t iteration;
    uint64_t prng_state = seed;

    double *distances = NULL;
    double *sums = NULL;
    unsigned char *selected_rows = NULL;
    size_t *remaining_cluster_sizes = NULL;
    int status = GS_BASELINE_KMEANS_SUCCESS;

    if (result == NULL) {
        return GS_BASELINE_KMEANS_ERROR_NULL_ARGUMENT;
    }
    gs_kmeans_result_zero(result);

    if (x == NULL) {
        return GS_BASELINE_KMEANS_ERROR_NULL_ARGUMENT;
    }

    if (rows == 0 || cols == 0 || k == 0 || k > rows ||
        max_iterations == 0) {

        return GS_BASELINE_KMEANS_ERROR_DIMENSIONS;
    }

    if (!gs_kmeans_checked_product(rows, cols, &input_count) ||
        !gs_kmeans_checked_product(k, cols, &centroid_count) ||
        !gs_kmeans_checked_product(input_count, sizeof(float), &input_bytes) ||
        !gs_kmeans_checked_product(
            centroid_count,
            sizeof(double),
            &centroid_bytes
        ) ||
        !gs_kmeans_checked_product(rows, sizeof(size_t), &row_size_bytes) ||
        !gs_kmeans_checked_product(k, sizeof(size_t), &cluster_size_bytes) ||
        !gs_kmeans_checked_product(rows, sizeof(double), &row_double_bytes) ||
        !gs_kmeans_checked_product(
            rows,
            sizeof(unsigned char),
            &row_flag_bytes
        )) {

        return GS_BASELINE_KMEANS_ERROR_OVERFLOW;
    }

    (void)input_bytes;

    if (!gs_kmeans_input_is_finite(x, input_count)) {
        return GS_BASELINE_KMEANS_ERROR_NONFINITE_INPUT;
    }

    result->centroids = (double *)malloc(centroid_bytes);
    result->assignments = (size_t *)malloc(row_size_bytes);
    result->cluster_sizes = (size_t *)malloc(cluster_size_bytes);
    result->initial_centroid_indices = (size_t *)malloc(cluster_size_bytes);

    distances = (double *)malloc(row_double_bytes);
    sums = (double *)malloc(centroid_bytes);
    selected_rows = (unsigned char *)malloc(row_flag_bytes);
    remaining_cluster_sizes = (size_t *)malloc(cluster_size_bytes);

    if (result->centroids == NULL || result->assignments == NULL ||
        result->cluster_sizes == NULL ||
        result->initial_centroid_indices == NULL || distances == NULL ||
        sums == NULL || selected_rows == NULL ||
        remaining_cluster_sizes == NULL) {

        status = GS_BASELINE_KMEANS_ERROR_ALLOCATION;
        goto error;
    }

    result->rows = rows;
    result->cols = cols;
    result->k = k;
    result->seed = seed;

    memset(selected_rows, 0, row_flag_bytes);
    status = gs_kmeans_initialize_plus_plus(
        x,
        rows,
        cols,
        k,
        &prng_state,
        result->centroids,
        result->initial_centroid_indices,
        selected_rows,
        distances
    );
    if (status != GS_BASELINE_KMEANS_SUCCESS) {
        goto error;
    }

    status = gs_kmeans_assign_unchecked(
        x,
        rows,
        cols,
        result->centroids,
        k,
        result->assignments,
        distances,
        NULL
    );
    if (status != GS_BASELINE_KMEANS_SUCCESS) {
        goto error;
    }

    status = gs_kmeans_sum_distances(
        distances,
        rows,
        &result->initial_inertia
    );
    if (status != GS_BASELINE_KMEANS_SUCCESS) {
        goto error;
    }

    for (iteration = 0; iteration < rows; iteration++) {
        result->assignments[iteration] = SIZE_MAX;
    }

    for (iteration = 0; iteration < max_iterations; iteration++) {
        size_t changed_assignments;
        size_t reseeded_count;

        status = gs_kmeans_assign_unchecked(
            x,
            rows,
            cols,
            result->centroids,
            k,
            result->assignments,
            distances,
            &changed_assignments
        );
        if (status != GS_BASELINE_KMEANS_SUCCESS) {
            goto error;
        }

        status = gs_kmeans_accumulate_and_update(
            x,
            rows,
            cols,
            k,
            result,
            sums
        );
        if (status != GS_BASELINE_KMEANS_SUCCESS) {
            goto error;
        }

        status = gs_kmeans_reseed_empty_clusters(
            x,
            rows,
            cols,
            k,
            result,
            selected_rows,
            remaining_cluster_sizes,
            &reseeded_count
        );
        if (status != GS_BASELINE_KMEANS_SUCCESS) {
            goto error;
        }

        result->iterations = iteration + 1;

        if (changed_assignments == 0 && reseeded_count == 0) {
            result->converged = 1;
            break;
        }
    }

    status = gs_kmeans_finalize_result(
        x,
        rows,
        cols,
        k,
        result,
        distances
    );
    if (status != GS_BASELINE_KMEANS_SUCCESS) {
        goto error;
    }

    free(remaining_cluster_sizes);
    free(selected_rows);
    free(sums);
    free(distances);
    return GS_BASELINE_KMEANS_SUCCESS;

error:
    free(remaining_cluster_sizes);
    free(selected_rows);
    free(sums);
    free(distances);
    gs_baseline_kmeans_result_free(result);
    return status;
}
