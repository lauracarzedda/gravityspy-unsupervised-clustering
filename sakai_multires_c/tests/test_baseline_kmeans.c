#include "baseline_kmeans.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>


enum {
    GS_TEST_DEFAULT_SEED = 123
};


static void gs_test_result_init(GSBaselineKMeansResult *result)
{
    memset(result, 0, sizeof(*result));
}


static int gs_test_result_is_valid(
    const GSBaselineKMeansResult *result,
    size_t rows,
    size_t cols,
    size_t k
)
{
    size_t centroid_count = k * cols;
    size_t total_rows = 0;
    size_t index;

    if (result == NULL || result->rows != rows || result->cols != cols ||
        result->k != k || result->iterations == 0 ||
        result->centroids == NULL || result->assignments == NULL ||
        result->cluster_sizes == NULL ||
        result->initial_centroid_indices == NULL ||
        !isfinite(result->initial_inertia) ||
        result->initial_inertia < 0.0 ||
        !isfinite(result->inertia) || result->inertia < 0.0 ||
        (result->converged != 0 && result->converged != 1)) {

        return 0;
    }

    for (index = 0; index < centroid_count; index++) {
        if (!isfinite(result->centroids[index])) {
            return 0;
        }
    }

    for (index = 0; index < rows; index++) {
        if (result->assignments[index] >= k) {
            return 0;
        }
    }

    for (index = 0; index < k; index++) {
        if (total_rows > SIZE_MAX - result->cluster_sizes[index]) {
            return 0;
        }
        total_rows += result->cluster_sizes[index];
    }

    return total_rows == rows;
}


static int gs_test_kmeanspp_deterministic_repeat(void)
{
    const float x[] = {
        -2.0f, -1.0f,
        -1.0f, -2.0f,
         0.0f,  0.0f,
         1.0f,  1.0f,
         9.0f, 10.0f,
        10.0f,  9.0f,
        11.0f, 11.0f,
        12.0f, 10.0f,
        19.0f, -1.0f,
        20.0f,  0.0f,
        21.0f,  1.0f,
        22.0f,  0.0f
    };

    GSBaselineKMeansResult first;
    GSBaselineKMeansResult second;
    int failed;

    gs_test_result_init(&first);
    gs_test_result_init(&second);

    if (gs_baseline_kmeans_fit(
            x, 12, 2, 3, GS_TEST_DEFAULT_SEED, 100, &first
        ) != GS_BASELINE_KMEANS_SUCCESS ||
        gs_baseline_kmeans_fit(
            x, 12, 2, 3, GS_TEST_DEFAULT_SEED, 100, &second
        ) != GS_BASELINE_KMEANS_SUCCESS) {

        gs_baseline_kmeans_result_free(&second);
        gs_baseline_kmeans_result_free(&first);
        return 1;
    }

    failed =
        !gs_test_result_is_valid(&first, 12, 2, 3) ||
        !gs_test_result_is_valid(&second, 12, 2, 3) ||
        memcmp(
            first.initial_centroid_indices,
            second.initial_centroid_indices,
            3 * sizeof(size_t)
        ) != 0 ||
        memcmp(
            first.centroids,
            second.centroids,
            6 * sizeof(double)
        ) != 0 ||
        memcmp(
            first.assignments,
            second.assignments,
            12 * sizeof(size_t)
        ) != 0 ||
        memcmp(
            first.cluster_sizes,
            second.cluster_sizes,
            3 * sizeof(size_t)
        ) != 0 ||
        memcmp(
            &first.initial_inertia,
            &second.initial_inertia,
            sizeof(double)
        ) != 0 ||
        memcmp(&first.inertia, &second.inertia, sizeof(double)) != 0 ||
        first.iterations != second.iterations ||
        first.converged != second.converged ||
        first.empty_cluster_reseeds != second.empty_cluster_reseeds;

    gs_baseline_kmeans_result_free(&second);
    gs_baseline_kmeans_result_free(&first);
    return failed;
}


static int gs_test_kmeanspp_unique_initial_indices(void)
{
    const float x[] = {
         0.0f,  0.0f,
         1.0f,  2.0f,
         2.0f,  1.0f,
         8.0f,  8.0f,
         9.0f, 10.0f,
        10.0f,  9.0f,
        18.0f,  0.0f,
        19.0f,  2.0f,
        20.0f,  1.0f
    };

    GSBaselineKMeansResult result;
    size_t left;
    int failed = 0;

    gs_test_result_init(&result);
    if (gs_baseline_kmeans_fit(
            x, 9, 2, 5, GS_TEST_DEFAULT_SEED, 50, &result
        ) != GS_BASELINE_KMEANS_SUCCESS) {

        return 1;
    }

    for (left = 0; left < result.k; left++) {
        size_t right;

        if (result.initial_centroid_indices[left] >= result.rows) {
            failed = 1;
        }

        for (right = left + 1; right < result.k; right++) {
            if (result.initial_centroid_indices[left] ==
                result.initial_centroid_indices[right]) {

                failed = 1;
            }
        }
    }

    gs_baseline_kmeans_result_free(&result);
    return failed;
}


static int gs_test_two_separated_clusters(void)
{
    const float x[] = {
         0.0f,  0.0f,
         0.0f,  2.0f,
         2.0f,  0.0f,
         2.0f,  2.0f,
        10.0f, 10.0f,
        10.0f, 12.0f,
        12.0f, 10.0f,
        12.0f, 12.0f
    };

    GSBaselineKMeansResult result;
    size_t low_cluster;
    size_t high_cluster;
    size_t row;
    int found_low_centroid = 0;
    int found_high_centroid = 0;
    int failed = 0;

    gs_test_result_init(&result);
    if (gs_baseline_kmeans_fit(
            x, 8, 2, 2, GS_TEST_DEFAULT_SEED, 100, &result
        ) != GS_BASELINE_KMEANS_SUCCESS) {

        return 1;
    }

    low_cluster = result.assignments[0];
    high_cluster = result.assignments[4];

    for (row = 0; row < 4; row++) {
        if (result.assignments[row] != low_cluster) {
            failed = 1;
        }
    }
    for (row = 4; row < 8; row++) {
        if (result.assignments[row] != high_cluster) {
            failed = 1;
        }
    }

    if (low_cluster == high_cluster || result.converged != 1 ||
        fabs(result.inertia - 16.0) > 1.0e-12) {

        failed = 1;
    }

    for (row = 0; row < 2; row++) {
        double first = result.centroids[row * 2];
        double second = result.centroids[row * 2 + 1];

        if (fabs(first - 1.0) <= 1.0e-12 &&
            fabs(second - 1.0) <= 1.0e-12) {

            found_low_centroid = 1;
        }
        if (fabs(first - 11.0) <= 1.0e-12 &&
            fabs(second - 11.0) <= 1.0e-12) {

            found_high_centroid = 1;
        }
    }

    if (!found_low_centroid || !found_high_centroid) {
        failed = 1;
    }

    gs_baseline_kmeans_result_free(&result);
    return failed;
}


static int gs_test_fixed_centroid_assignment(void)
{
    const float x[] = {
         0.0f, 0.0f,
         2.0f, 0.0f,
        10.0f, 0.0f,
        12.0f, 0.0f
    };
    const double centroids[] = {
         0.0, 0.0,
        10.0, 0.0
    };
    const size_t expected_assignments[] = {0, 0, 1, 1};
    const double expected_distances[] = {0.0, 4.0, 0.0, 4.0};
    size_t assignments[4];
    size_t assignments_without_distances[4];
    double distances[4];
    size_t row;

    if (gs_baseline_kmeans_assign(
            x, 4, 2, centroids, 2, assignments, distances
        ) != GS_BASELINE_KMEANS_SUCCESS ||
        gs_baseline_kmeans_assign(
            x,
            4,
            2,
            centroids,
            2,
            assignments_without_distances,
            NULL
        ) != GS_BASELINE_KMEANS_SUCCESS) {

        return 1;
    }

    for (row = 0; row < 4; row++) {
        if (assignments[row] != expected_assignments[row] ||
            assignments_without_distances[row] !=
                expected_assignments[row] ||
            fabs(distances[row] - expected_distances[row]) > 1.0e-12) {

            return 1;
        }
    }

    return 0;
}


static int gs_test_tie_break_lowest_cluster(void)
{
    const float x[] = {2.0f, 2.0f};
    const double centroids[] = {
        1.0, 1.0,
        1.0, 1.0
    };
    size_t assignment = SIZE_MAX;
    double distance = -1.0;

    if (gs_baseline_kmeans_assign(
            x, 1, 2, centroids, 2, &assignment, &distance
        ) != GS_BASELINE_KMEANS_SUCCESS) {

        return 1;
    }

    return assignment == 0 && fabs(distance - 2.0) <= 1.0e-12 ? 0 : 1;
}


static int gs_test_identical_points_degenerate(void)
{
    const float x[] = {
        3.0f, -2.0f,
        3.0f, -2.0f,
        3.0f, -2.0f,
        3.0f, -2.0f,
        3.0f, -2.0f
    };

    GSBaselineKMeansResult result;
    int failed;

    gs_test_result_init(&result);
    if (gs_baseline_kmeans_fit(
            x, 5, 2, 3, GS_TEST_DEFAULT_SEED, 20, &result
        ) != GS_BASELINE_KMEANS_SUCCESS) {

        return 1;
    }

    failed =
        !gs_test_result_is_valid(&result, 5, 2, 3) ||
        result.inertia != 0.0 || result.iterations > 20 ||
        result.empty_cluster_reseeds != 0;

    gs_baseline_kmeans_result_free(&result);
    return failed;
}


static int gs_test_max_iterations_contract(void)
{
    const float x[] = {
         0.0f,  0.0f,
         0.0f,  4.0f,
         4.0f,  0.0f,
         4.0f,  4.0f,
        10.0f, 10.0f,
        10.0f, 14.0f,
        14.0f, 10.0f,
        14.0f, 14.0f
    };

    GSBaselineKMeansResult result;
    int failed;

    gs_test_result_init(&result);
    if (gs_baseline_kmeans_fit(
            x, 8, 2, 2, GS_TEST_DEFAULT_SEED, 1, &result
        ) != GS_BASELINE_KMEANS_SUCCESS) {

        return 1;
    }

    failed =
        !gs_test_result_is_valid(&result, 8, 2, 2) ||
        result.iterations != 1 || result.converged != 0;

    gs_baseline_kmeans_result_free(&result);
    return failed;
}


static int gs_test_fit_must_fail(
    const float *x,
    size_t rows,
    size_t cols,
    size_t k,
    size_t max_iterations
)
{
    GSBaselineKMeansResult result;
    int status;
    int failed;

    gs_test_result_init(&result);
    status = gs_baseline_kmeans_fit(
        x,
        rows,
        cols,
        k,
        GS_TEST_DEFAULT_SEED,
        max_iterations,
        &result
    );

    failed = status == GS_BASELINE_KMEANS_SUCCESS ||
             result.centroids != NULL || result.assignments != NULL ||
             result.cluster_sizes != NULL ||
             result.initial_centroid_indices != NULL;

    gs_baseline_kmeans_result_free(&result);
    return failed;
}


static int gs_test_invalid_arguments_and_nonfinite(void)
{
    const float valid_x[] = {0.0f, 1.0f, 2.0f, 3.0f};
    const float nan_x[] = {0.0f, NAN};
    const float infinity_x[] = {0.0f, INFINITY};
    const double valid_centroids[] = {0.0, 1.0};
    const double invalid_centroids[] = {0.0, INFINITY};
    size_t assignments[2];
    GSBaselineKMeansResult result;

    gs_test_result_init(&result);

    if (gs_baseline_kmeans_fit(
            valid_x, 2, 2, 1, GS_TEST_DEFAULT_SEED, 10, NULL
        ) == GS_BASELINE_KMEANS_SUCCESS ||
        gs_test_fit_must_fail(NULL, 2, 2, 1, 10) ||
        gs_test_fit_must_fail(valid_x, 0, 2, 1, 10) ||
        gs_test_fit_must_fail(valid_x, 2, 0, 1, 10) ||
        gs_test_fit_must_fail(valid_x, 2, 2, 0, 10) ||
        gs_test_fit_must_fail(valid_x, 2, 2, 3, 10) ||
        gs_test_fit_must_fail(valid_x, 2, 2, 1, 0) ||
        gs_test_fit_must_fail(valid_x, SIZE_MAX, 2, 1, 10) ||
        gs_test_fit_must_fail(nan_x, 1, 2, 1, 10) ||
        gs_test_fit_must_fail(infinity_x, 1, 2, 1, 10) ||
        gs_baseline_kmeans_assign(
            NULL, 2, 2, valid_centroids, 1, assignments, NULL
        ) == GS_BASELINE_KMEANS_SUCCESS ||
        gs_baseline_kmeans_assign(
            valid_x, 2, 2, NULL, 1, assignments, NULL
        ) == GS_BASELINE_KMEANS_SUCCESS ||
        gs_baseline_kmeans_assign(
            valid_x, 2, 2, valid_centroids, 1, NULL, NULL
        ) == GS_BASELINE_KMEANS_SUCCESS ||
        gs_baseline_kmeans_assign(
            valid_x, 0, 2, valid_centroids, 1, assignments, NULL
        ) == GS_BASELINE_KMEANS_SUCCESS ||
        gs_baseline_kmeans_assign(
            valid_x, 2, 0, valid_centroids, 1, assignments, NULL
        ) == GS_BASELINE_KMEANS_SUCCESS ||
        gs_baseline_kmeans_assign(
            valid_x, 2, 2, valid_centroids, 0, assignments, NULL
        ) == GS_BASELINE_KMEANS_SUCCESS ||
        gs_baseline_kmeans_assign(
            valid_x,
            SIZE_MAX,
            2,
            valid_centroids,
            1,
            assignments,
            NULL
        ) == GS_BASELINE_KMEANS_SUCCESS ||
        gs_baseline_kmeans_assign(
            valid_x,
            1,
            2,
            valid_centroids,
            SIZE_MAX,
            assignments,
            NULL
        ) == GS_BASELINE_KMEANS_SUCCESS ||
        gs_baseline_kmeans_assign(
            nan_x, 1, 2, valid_centroids, 1, assignments, NULL
        ) == GS_BASELINE_KMEANS_SUCCESS ||
        gs_baseline_kmeans_assign(
            infinity_x, 1, 2, valid_centroids, 1, assignments, NULL
        ) == GS_BASELINE_KMEANS_SUCCESS ||
        gs_baseline_kmeans_assign(
            valid_x, 1, 2, invalid_centroids, 1, assignments, NULL
        ) == GS_BASELINE_KMEANS_SUCCESS) {

        gs_baseline_kmeans_result_free(&result);
        return 1;
    }

    gs_baseline_kmeans_result_free(&result);
    return 0;
}


static int gs_test_input_not_modified(void)
{
    float x[] = {
        0.0f, 0.0f,
        1.0f, 1.0f,
        9.0f, 9.0f,
        10.0f, 10.0f
    };
    float x_copy[8];
    double centroids[] = {
        0.0, 0.0,
        10.0, 10.0
    };
    double centroids_copy[4];
    size_t assignments[4];
    GSBaselineKMeansResult result;
    int failed;

    memcpy(x_copy, x, sizeof(x));
    memcpy(centroids_copy, centroids, sizeof(centroids));
    gs_test_result_init(&result);

    if (gs_baseline_kmeans_fit(
            x, 4, 2, 2, GS_TEST_DEFAULT_SEED, 20, &result
        ) != GS_BASELINE_KMEANS_SUCCESS ||
        memcmp(x, x_copy, sizeof(x)) != 0 ||
        gs_baseline_kmeans_assign(
            x, 4, 2, centroids, 2, assignments, NULL
        ) != GS_BASELINE_KMEANS_SUCCESS) {

        gs_baseline_kmeans_result_free(&result);
        return 1;
    }

    failed = memcmp(x, x_copy, sizeof(x)) != 0 ||
             memcmp(centroids, centroids_copy, sizeof(centroids)) != 0;

    gs_baseline_kmeans_result_free(&result);
    return failed;
}


static int gs_test_result_free_idempotent(void)
{
    const float x[] = {
        0.0f, 0.0f,
        1.0f, 1.0f,
        9.0f, 9.0f,
        10.0f, 10.0f
    };
    GSBaselineKMeansResult result;

    gs_test_result_init(&result);
    if (gs_baseline_kmeans_fit(
            x, 4, 2, 2, GS_TEST_DEFAULT_SEED, 20, &result
        ) != GS_BASELINE_KMEANS_SUCCESS) {

        return 1;
    }

    gs_baseline_kmeans_result_free(&result);
    gs_baseline_kmeans_result_free(&result);

    return result.rows == 0 && result.cols == 0 && result.k == 0 &&
           result.iterations == 0 && result.empty_cluster_reseeds == 0 &&
           result.converged == 0 && result.seed == UINT64_C(0) &&
           result.initial_inertia == 0.0 && result.inertia == 0.0 &&
           result.centroids == NULL && result.assignments == NULL &&
           result.cluster_sizes == NULL &&
           result.initial_centroid_indices == NULL ? 0 : 1;
}


int main(void)
{
    if (gs_test_kmeanspp_deterministic_repeat() != 0) {
        fprintf(stderr, "Errore: ripetibilita' k-means++.\n");
        return 1;
    }
    printf("[PASS] kmeanspp_deterministic_repeat\n");

    if (gs_test_kmeanspp_unique_initial_indices() != 0) {
        fprintf(stderr, "Errore: indici iniziali non unici.\n");
        return 1;
    }
    printf("[PASS] kmeanspp_unique_initial_indices\n");

    if (gs_test_two_separated_clusters() != 0) {
        fprintf(stderr, "Errore: due cluster separati.\n");
        return 1;
    }
    printf("[PASS] two_separated_clusters\n");

    if (gs_test_fixed_centroid_assignment() != 0) {
        fprintf(stderr, "Errore: assegnamento con centroidi fissi.\n");
        return 1;
    }
    printf("[PASS] fixed_centroid_assignment\n");

    if (gs_test_tie_break_lowest_cluster() != 0) {
        fprintf(stderr, "Errore: tie-break del cluster.\n");
        return 1;
    }
    printf("[PASS] tie_break_lowest_cluster\n");

    if (gs_test_identical_points_degenerate() != 0) {
        fprintf(stderr, "Errore: dataset degenere identico.\n");
        return 1;
    }
    printf("[PASS] identical_points_degenerate\n");

    if (gs_test_max_iterations_contract() != 0) {
        fprintf(stderr, "Errore: contratto max_iterations.\n");
        return 1;
    }
    printf("[PASS] max_iterations_contract\n");

    if (gs_test_invalid_arguments_and_nonfinite() != 0) {
        fprintf(stderr, "Errore: argomenti invalidi o non finiti.\n");
        return 1;
    }
    printf("[PASS] invalid_arguments_and_nonfinite\n");

    if (gs_test_input_not_modified() != 0) {
        fprintf(stderr, "Errore: input modificato.\n");
        return 1;
    }
    printf("[PASS] input_not_modified\n");

    if (gs_test_result_free_idempotent() != 0) {
        fprintf(stderr, "Errore: free non idempotente.\n");
        return 1;
    }
    printf("[PASS] result_free_idempotent\n");

    printf("DEFAULT_SEED=123\n");
    printf("CONVERGENCE=assignment_stability\n");
    printf("ALL TESTS PASSED\n");
    return 0;
}
