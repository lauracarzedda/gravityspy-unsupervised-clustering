#include "baseline_clustering_metrics.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>


static int gs_test_close(double left, double right)
{
    return isfinite(left) && isfinite(right) &&
           fabs(left - right) <= 1.0e-12;
}


static int gs_test_result_is_zero(
    const GSBaselineClusteringMetrics *result
)
{
    return result != NULL && result->rows == 0 &&
           result->label_count == 0 && result->cluster_count == 0 &&
           result->contingency == NULL && result->label_totals == NULL &&
           result->cluster_totals == NULL && result->purity == 0.0 &&
           result->normalized_mutual_information == 0.0 &&
           result->adjusted_rand_index == 0.0 &&
           result->homogeneity == 0.0 && result->completeness == 0.0;
}


static int gs_test_metrics_match(
    const GSBaselineClusteringMetrics *left,
    const GSBaselineClusteringMetrics *right
)
{
    return gs_test_close(left->purity, right->purity) &&
           gs_test_close(
               left->normalized_mutual_information,
               right->normalized_mutual_information
           ) &&
           gs_test_close(
               left->adjusted_rand_index,
               right->adjusted_rand_index
           ) &&
           gs_test_close(left->homogeneity, right->homogeneity) &&
           gs_test_close(left->completeness, right->completeness);
}


static int gs_test_contingency_layout_and_totals(void)
{
    const uint32_t labels[] = {0, 0, 1, 1, 1, 2};
    const uint32_t clusters[] = {0, 1, 0, 1, 1, 1};
    const uint64_t expected_contingency[] = {1, 1, 1, 2, 0, 1};
    const uint64_t expected_label_totals[] = {2, 3, 1};
    const uint64_t expected_cluster_totals[] = {2, 4};
    GSBaselineClusteringMetrics result;
    uint64_t total = UINT64_C(0);
    size_t index;
    int failed = 0;

    memset(&result, 0, sizeof(result));
    if (gs_baseline_clustering_metrics_compute(
            labels, clusters, 6, 3, 2, &result
        ) != GS_BASELINE_CLUSTERING_METRICS_SUCCESS) {

        return 1;
    }

    if (result.rows != 6 || result.label_count != 3 ||
        result.cluster_count != 2 ||
        memcmp(
            result.contingency,
            expected_contingency,
            sizeof(expected_contingency)
        ) != 0 ||
        memcmp(
            result.label_totals,
            expected_label_totals,
            sizeof(expected_label_totals)
        ) != 0 ||
        memcmp(
            result.cluster_totals,
            expected_cluster_totals,
            sizeof(expected_cluster_totals)
        ) != 0) {

        failed = 1;
    }

    for (index = 0; index < 6; index++) {
        total += result.contingency[index];
    }
    if (total != 6) {
        failed = 1;
    }

    gs_baseline_clustering_metrics_free(&result);
    return failed;
}


static int gs_test_perfect_permuted_clustering(void)
{
    const uint32_t labels[] = {0, 0, 1, 1, 2, 2};
    const uint32_t clusters[] = {2, 2, 0, 0, 1, 1};
    GSBaselineClusteringMetrics result;
    int failed;

    memset(&result, 0, sizeof(result));
    if (gs_baseline_clustering_metrics_compute(
            labels, clusters, 6, 3, 3, &result
        ) != GS_BASELINE_CLUSTERING_METRICS_SUCCESS) {

        return 1;
    }

    failed = !gs_test_close(result.purity, 1.0) ||
        !gs_test_close(result.normalized_mutual_information, 1.0) ||
        !gs_test_close(result.adjusted_rand_index, 1.0) ||
        !gs_test_close(result.homogeneity, 1.0) ||
        !gs_test_close(result.completeness, 1.0);

    gs_baseline_clustering_metrics_free(&result);
    return failed;
}


static int gs_test_one_cluster_for_two_labels(void)
{
    const uint32_t labels[] = {0, 0, 1, 1};
    const uint32_t clusters[] = {0, 0, 0, 0};
    GSBaselineClusteringMetrics result;
    int failed;

    memset(&result, 0, sizeof(result));
    if (gs_baseline_clustering_metrics_compute(
            labels, clusters, 4, 2, 1, &result
        ) != GS_BASELINE_CLUSTERING_METRICS_SUCCESS) {

        return 1;
    }

    failed = !gs_test_close(result.purity, 0.5) ||
        !gs_test_close(result.normalized_mutual_information, 0.0) ||
        !gs_test_close(result.adjusted_rand_index, 0.0) ||
        !gs_test_close(result.homogeneity, 0.0) ||
        !gs_test_close(result.completeness, 1.0);

    gs_baseline_clustering_metrics_free(&result);
    return failed;
}


static int gs_test_singleton_clusters(void)
{
    const uint32_t labels[] = {0, 0, 1, 1};
    const uint32_t clusters[] = {0, 1, 2, 3};
    GSBaselineClusteringMetrics result;
    int failed;

    memset(&result, 0, sizeof(result));
    if (gs_baseline_clustering_metrics_compute(
            labels, clusters, 4, 2, 4, &result
        ) != GS_BASELINE_CLUSTERING_METRICS_SUCCESS) {

        return 1;
    }

    failed = !gs_test_close(result.purity, 1.0) ||
        !gs_test_close(
            result.normalized_mutual_information,
            2.0 / 3.0
        ) ||
        !gs_test_close(result.adjusted_rand_index, 0.0) ||
        !gs_test_close(result.homogeneity, 1.0) ||
        !gs_test_close(result.completeness, 0.5);

    gs_baseline_clustering_metrics_free(&result);
    return failed;
}


static int gs_test_identical_degenerate_partition(void)
{
    const uint32_t labels[] = {0, 0, 0, 0};
    const uint32_t clusters[] = {0, 0, 0, 0};
    GSBaselineClusteringMetrics result;
    int failed;

    memset(&result, 0, sizeof(result));
    if (gs_baseline_clustering_metrics_compute(
            labels, clusters, 4, 1, 1, &result
        ) != GS_BASELINE_CLUSTERING_METRICS_SUCCESS) {

        return 1;
    }

    failed = !gs_test_close(result.purity, 1.0) ||
        !gs_test_close(result.normalized_mutual_information, 1.0) ||
        !gs_test_close(result.adjusted_rand_index, 1.0) ||
        !gs_test_close(result.homogeneity, 1.0) ||
        !gs_test_close(result.completeness, 1.0);

    gs_baseline_clustering_metrics_free(&result);
    return failed;
}


static int gs_test_permutation_invariance(void)
{
    const uint32_t labels[] = {0, 0, 1, 1, 2, 2, 2, 0};
    const uint32_t clusters[] = {0, 1, 1, 1, 2, 0, 2, 0};
    const uint32_t permuted_labels[] = {2, 0, 1, 2, 1, 2, 1, 0};
    const uint32_t permuted_clusters[] = {1, 2, 1, 1, 0, 2, 0, 2};
    GSBaselineClusteringMetrics original;
    GSBaselineClusteringMetrics permuted;
    int failed;

    memset(&original, 0, sizeof(original));
    memset(&permuted, 0, sizeof(permuted));

    if (gs_baseline_clustering_metrics_compute(
            labels, clusters, 8, 3, 3, &original
        ) != GS_BASELINE_CLUSTERING_METRICS_SUCCESS ||
        gs_baseline_clustering_metrics_compute(
            permuted_labels,
            permuted_clusters,
            8,
            3,
            3,
            &permuted
        ) != GS_BASELINE_CLUSTERING_METRICS_SUCCESS) {

        gs_baseline_clustering_metrics_free(&permuted);
        gs_baseline_clustering_metrics_free(&original);
        return 1;
    }

    failed = !gs_test_metrics_match(&original, &permuted);
    gs_baseline_clustering_metrics_free(&permuted);
    gs_baseline_clustering_metrics_free(&original);
    return failed;
}


static int gs_test_deterministic_repeat(void)
{
    const uint32_t labels[] = {0, 0, 1, 1, 2, 2, 2, 0};
    const uint32_t clusters[] = {0, 1, 1, 1, 2, 0, 2, 0};
    GSBaselineClusteringMetrics first;
    GSBaselineClusteringMetrics second;
    int failed;

    memset(&first, 0, sizeof(first));
    memset(&second, 0, sizeof(second));

    if (gs_baseline_clustering_metrics_compute(
            labels, clusters, 8, 3, 3, &first
        ) != GS_BASELINE_CLUSTERING_METRICS_SUCCESS ||
        gs_baseline_clustering_metrics_compute(
            labels, clusters, 8, 3, 3, &second
        ) != GS_BASELINE_CLUSTERING_METRICS_SUCCESS) {

        gs_baseline_clustering_metrics_free(&second);
        gs_baseline_clustering_metrics_free(&first);
        return 1;
    }

    failed = memcmp(first.contingency, second.contingency,
                    9 * sizeof(uint64_t)) != 0 ||
        memcmp(first.label_totals, second.label_totals,
               3 * sizeof(uint64_t)) != 0 ||
        memcmp(first.cluster_totals, second.cluster_totals,
               3 * sizeof(uint64_t)) != 0 ||
        memcmp(&first.purity, &second.purity, sizeof(double)) != 0 ||
        memcmp(
            &first.normalized_mutual_information,
            &second.normalized_mutual_information,
            sizeof(double)
        ) != 0 ||
        memcmp(
            &first.adjusted_rand_index,
            &second.adjusted_rand_index,
            sizeof(double)
        ) != 0 ||
        memcmp(
            &first.homogeneity,
            &second.homogeneity,
            sizeof(double)
        ) != 0 ||
        memcmp(
            &first.completeness,
            &second.completeness,
            sizeof(double)
        ) != 0;

    gs_baseline_clustering_metrics_free(&second);
    gs_baseline_clustering_metrics_free(&first);
    return failed;
}


static int gs_test_invalid_call(
    const uint32_t *labels,
    const uint32_t *clusters,
    size_t rows,
    size_t label_count,
    size_t cluster_count,
    int expected_error
)
{
    GSBaselineClusteringMetrics result;
    int actual_error;

    memset(&result, 0xa5, sizeof(result));
    actual_error = gs_baseline_clustering_metrics_compute(
        labels,
        clusters,
        rows,
        label_count,
        cluster_count,
        &result
    );

    return actual_error != expected_error || !gs_test_result_is_zero(&result);
}


static int gs_test_invalid_arguments_and_ranges(void)
{
    const uint32_t valid_labels[] = {0, 1};
    const uint32_t valid_clusters[] = {0, 1};
    const uint32_t invalid_labels[] = {0, 2};
    const uint32_t invalid_clusters[] = {0, 2};

    if (gs_baseline_clustering_metrics_compute(
            valid_labels, valid_clusters, 2, 2, 2, NULL
        ) != GS_BASELINE_CLUSTERING_METRICS_ERROR_NULL_ARGUMENT ||
        gs_test_invalid_call(
            NULL, valid_clusters, 2, 2, 2,
            GS_BASELINE_CLUSTERING_METRICS_ERROR_NULL_ARGUMENT
        ) ||
        gs_test_invalid_call(
            valid_labels, NULL, 2, 2, 2,
            GS_BASELINE_CLUSTERING_METRICS_ERROR_NULL_ARGUMENT
        ) ||
        gs_test_invalid_call(
            valid_labels, valid_clusters, 0, 2, 2,
            GS_BASELINE_CLUSTERING_METRICS_ERROR_DIMENSIONS
        ) ||
        gs_test_invalid_call(
            valid_labels, valid_clusters, 2, 0, 2,
            GS_BASELINE_CLUSTERING_METRICS_ERROR_DIMENSIONS
        ) ||
        gs_test_invalid_call(
            valid_labels, valid_clusters, 2, 2, 0,
            GS_BASELINE_CLUSTERING_METRICS_ERROR_DIMENSIONS
        ) ||
        gs_test_invalid_call(
            invalid_labels, valid_clusters, 2, 2, 2,
            GS_BASELINE_CLUSTERING_METRICS_ERROR_RANGE
        ) ||
        gs_test_invalid_call(
            valid_labels, invalid_clusters, 2, 2, 2,
            GS_BASELINE_CLUSTERING_METRICS_ERROR_RANGE
        ) ||
        gs_test_invalid_call(
            valid_labels, valid_clusters, 2, SIZE_MAX, 2,
            GS_BASELINE_CLUSTERING_METRICS_ERROR_OVERFLOW
        ) ||
        gs_test_invalid_call(
            valid_labels,
            valid_clusters,
            2,
            SIZE_MAX / sizeof(uint64_t) + 1,
            1,
            GS_BASELINE_CLUSTERING_METRICS_ERROR_OVERFLOW
        )) {

        return 1;
    }

    return 0;
}


static int gs_test_input_not_modified(void)
{
    uint32_t labels[] = {0, 0, 1, 1, 2, 2};
    uint32_t clusters[] = {2, 2, 0, 0, 1, 1};
    uint32_t labels_copy[6];
    uint32_t clusters_copy[6];
    GSBaselineClusteringMetrics result;
    int failed;

    memcpy(labels_copy, labels, sizeof(labels));
    memcpy(clusters_copy, clusters, sizeof(clusters));
    memset(&result, 0, sizeof(result));

    if (gs_baseline_clustering_metrics_compute(
            labels, clusters, 6, 3, 3, &result
        ) != GS_BASELINE_CLUSTERING_METRICS_SUCCESS) {

        return 1;
    }

    failed = memcmp(labels, labels_copy, sizeof(labels)) != 0 ||
        memcmp(clusters, clusters_copy, sizeof(clusters)) != 0;
    gs_baseline_clustering_metrics_free(&result);
    return failed;
}


static int gs_test_result_free_idempotent(void)
{
    const uint32_t labels[] = {0, 0, 1, 1};
    const uint32_t clusters[] = {0, 0, 1, 1};
    GSBaselineClusteringMetrics result;

    memset(&result, 0, sizeof(result));
    if (gs_baseline_clustering_metrics_compute(
            labels, clusters, 4, 2, 2, &result
        ) != GS_BASELINE_CLUSTERING_METRICS_SUCCESS) {

        return 1;
    }

    gs_baseline_clustering_metrics_free(&result);
    gs_baseline_clustering_metrics_free(&result);
    gs_baseline_clustering_metrics_free(NULL);
    return !gs_test_result_is_zero(&result);
}


int main(void)
{
    if (gs_test_contingency_layout_and_totals()) {
        fprintf(stderr, "[FAIL] contingency_layout_and_totals\n");
        return 1;
    }
    printf("[PASS] contingency_layout_and_totals\n");

    if (gs_test_perfect_permuted_clustering()) {
        fprintf(stderr, "[FAIL] perfect_permuted_clustering\n");
        return 1;
    }
    printf("[PASS] perfect_permuted_clustering\n");

    if (gs_test_one_cluster_for_two_labels()) {
        fprintf(stderr, "[FAIL] one_cluster_for_two_labels\n");
        return 1;
    }
    printf("[PASS] one_cluster_for_two_labels\n");

    if (gs_test_singleton_clusters()) {
        fprintf(stderr, "[FAIL] singleton_clusters\n");
        return 1;
    }
    printf("[PASS] singleton_clusters\n");

    if (gs_test_identical_degenerate_partition()) {
        fprintf(stderr, "[FAIL] identical_degenerate_partition\n");
        return 1;
    }
    printf("[PASS] identical_degenerate_partition\n");

    if (gs_test_permutation_invariance()) {
        fprintf(stderr, "[FAIL] permutation_invariance\n");
        return 1;
    }
    printf("[PASS] permutation_invariance\n");

    if (gs_test_deterministic_repeat()) {
        fprintf(stderr, "[FAIL] deterministic_repeat\n");
        return 1;
    }
    printf("[PASS] deterministic_repeat\n");

    if (gs_test_invalid_arguments_and_ranges()) {
        fprintf(stderr, "[FAIL] invalid_arguments_and_ranges\n");
        return 1;
    }
    printf("[PASS] invalid_arguments_and_ranges\n");

    if (gs_test_input_not_modified()) {
        fprintf(stderr, "[FAIL] input_not_modified\n");
        return 1;
    }
    printf("[PASS] input_not_modified\n");

    if (gs_test_result_free_idempotent()) {
        fprintf(stderr, "[FAIL] result_free_idempotent\n");
        return 1;
    }
    printf("[PASS] result_free_idempotent\n");

    printf("NMI_NORMALIZATION=arithmetic\n");
    printf("PURITY=weighted_cluster_majority\n");
    printf("ALL TESTS PASSED\n");
    return 0;
}
