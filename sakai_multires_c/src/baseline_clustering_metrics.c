#include "baseline_clustering_metrics.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>


static void gs_metrics_result_zero(GSBaselineClusteringMetrics *result)
{
    if (result == NULL) {
        return;
    }

    result->rows = 0;
    result->label_count = 0;
    result->cluster_count = 0;
    result->contingency = NULL;
    result->label_totals = NULL;
    result->cluster_totals = NULL;
    result->purity = 0.0;
    result->normalized_mutual_information = 0.0;
    result->adjusted_rand_index = 0.0;
    result->homogeneity = 0.0;
    result->completeness = 0.0;
}


void gs_baseline_clustering_metrics_free(
    GSBaselineClusteringMetrics *result
)
{
    if (result == NULL) {
        return;
    }

    free(result->cluster_totals);
    free(result->label_totals);
    free(result->contingency);
    gs_metrics_result_zero(result);
}


static int gs_metrics_checked_product(
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


static long double gs_metrics_choose_two(uint64_t count)
{
    long double value = (long double)count;

    return value * (value - 1.0L) / 2.0L;
}


static int gs_metrics_to_unit_interval(
    long double value,
    double *converted
)
{
    const long double tolerance = 1.0e-12L;

    if (converted == NULL || !isfinite(value) ||
        value < -tolerance || value > 1.0L + tolerance) {

        return 0;
    }

    if (value < 0.0L) {
        value = 0.0L;
    } else if (value > 1.0L) {
        value = 1.0L;
    }

    *converted = (double)value;
    return isfinite(*converted);
}


static int gs_metrics_to_signed_unit_interval(
    long double value,
    double *converted
)
{
    const long double tolerance = 1.0e-12L;

    if (converted == NULL || !isfinite(value) ||
        value < -1.0L - tolerance || value > 1.0L + tolerance) {

        return 0;
    }

    if (value < -1.0L) {
        value = -1.0L;
    } else if (value > 1.0L) {
        value = 1.0L;
    }

    *converted = (double)value;
    return isfinite(*converted);
}


static int gs_metrics_partitions_are_equivalent(
    const uint64_t *contingency,
    size_t label_count,
    size_t cluster_count
)
{
    size_t label;
    size_t cluster;

    for (label = 0; label < label_count; label++) {
        size_t nonzero = 0;

        for (cluster = 0; cluster < cluster_count; cluster++) {
            if (contingency[label * cluster_count + cluster] != 0) {
                nonzero++;
            }
        }
        if (nonzero > 1) {
            return 0;
        }
    }

    for (cluster = 0; cluster < cluster_count; cluster++) {
        size_t nonzero = 0;

        for (label = 0; label < label_count; label++) {
            if (contingency[label * cluster_count + cluster] != 0) {
                nonzero++;
            }
        }
        if (nonzero > 1) {
            return 0;
        }
    }

    return 1;
}


int gs_baseline_clustering_metrics_compute(
    const uint32_t *labels,
    const uint32_t *clusters,
    size_t rows,
    size_t label_count,
    size_t cluster_count,
    GSBaselineClusteringMetrics *result
)
{
    uint64_t *contingency = NULL;
    uint64_t *label_totals = NULL;
    uint64_t *cluster_totals = NULL;
    size_t contingency_count;
    size_t contingency_bytes;
    size_t label_bytes;
    size_t cluster_bytes;
    size_t row;
    size_t label;
    size_t cluster;
    uint64_t majority_sum = UINT64_C(0);
    long double row_count;
    long double label_entropy = 0.0L;
    long double cluster_entropy = 0.0L;
    long double mutual_information = 0.0L;
    long double label_given_cluster = 0.0L;
    long double cluster_given_label = 0.0L;
    long double sum_cells_choose_two = 0.0L;
    long double sum_labels_choose_two = 0.0L;
    long double sum_clusters_choose_two = 0.0L;
    long double total_choose_two;
    long double expected_index;
    long double maximum_index;
    long double ari_denominator;
    long double ari_numerator;
    long double ari_value;
    long double nmi_value;
    long double homogeneity_value;
    long double completeness_value;
    double purity;
    double normalized_mutual_information;
    double adjusted_rand_index;
    double homogeneity;
    double completeness;
    int error = GS_BASELINE_CLUSTERING_METRICS_ERROR_NUMERIC;

    if (result == NULL) {
        return GS_BASELINE_CLUSTERING_METRICS_ERROR_NULL_ARGUMENT;
    }
    gs_metrics_result_zero(result);

    if (labels == NULL || clusters == NULL) {
        return GS_BASELINE_CLUSTERING_METRICS_ERROR_NULL_ARGUMENT;
    }
    if (rows == 0 || label_count == 0 || cluster_count == 0) {
        return GS_BASELINE_CLUSTERING_METRICS_ERROR_DIMENSIONS;
    }
    if (rows > UINT64_MAX) {
        return GS_BASELINE_CLUSTERING_METRICS_ERROR_OVERFLOW;
    }

    if (!gs_metrics_checked_product(
            label_count,
            cluster_count,
            &contingency_count
        ) ||
        !gs_metrics_checked_product(
            contingency_count,
            sizeof(uint64_t),
            &contingency_bytes
        ) ||
        !gs_metrics_checked_product(
            label_count,
            sizeof(uint64_t),
            &label_bytes
        ) ||
        !gs_metrics_checked_product(
            cluster_count,
            sizeof(uint64_t),
            &cluster_bytes
        )) {

        return GS_BASELINE_CLUSTERING_METRICS_ERROR_OVERFLOW;
    }

    for (row = 0; row < rows; row++) {
        if ((size_t)labels[row] >= label_count ||
            (size_t)clusters[row] >= cluster_count) {

            return GS_BASELINE_CLUSTERING_METRICS_ERROR_RANGE;
        }
    }

    contingency = (uint64_t *)calloc(contingency_count, sizeof(uint64_t));
    label_totals = (uint64_t *)calloc(label_count, sizeof(uint64_t));
    cluster_totals = (uint64_t *)calloc(cluster_count, sizeof(uint64_t));

    if (contingency == NULL || label_totals == NULL ||
        cluster_totals == NULL) {

        error = GS_BASELINE_CLUSTERING_METRICS_ERROR_ALLOCATION;
        goto cleanup;
    }

    for (row = 0; row < rows; row++) {
        size_t current_label = (size_t)labels[row];
        size_t current_cluster = (size_t)clusters[row];
        size_t cell = current_label * cluster_count + current_cluster;

        if (contingency[cell] == UINT64_MAX ||
            label_totals[current_label] == UINT64_MAX ||
            cluster_totals[current_cluster] == UINT64_MAX) {

            error = GS_BASELINE_CLUSTERING_METRICS_ERROR_OVERFLOW;
            goto cleanup;
        }

        contingency[cell]++;
        label_totals[current_label]++;
        cluster_totals[current_cluster]++;
    }

    row_count = (long double)rows;

    for (label = 0; label < label_count; label++) {
        uint64_t count = label_totals[label];

        if (count != 0) {
            long double probability = (long double)count / row_count;

            label_entropy -= probability * logl(probability);
            sum_labels_choose_two += gs_metrics_choose_two(count);
        }
    }

    for (cluster = 0; cluster < cluster_count; cluster++) {
        uint64_t count = cluster_totals[cluster];
        uint64_t maximum = UINT64_C(0);

        if (count != 0) {
            long double probability = (long double)count / row_count;

            cluster_entropy -= probability * logl(probability);
            sum_clusters_choose_two += gs_metrics_choose_two(count);
        }

        for (label = 0; label < label_count; label++) {
            uint64_t cell_count =
                contingency[label * cluster_count + cluster];

            if (cell_count > maximum) {
                maximum = cell_count;
            }
        }

        if (majority_sum > UINT64_MAX - maximum) {
            error = GS_BASELINE_CLUSTERING_METRICS_ERROR_OVERFLOW;
            goto cleanup;
        }
        majority_sum += maximum;
    }

    for (label = 0; label < label_count; label++) {
        for (cluster = 0; cluster < cluster_count; cluster++) {
            uint64_t cell_count =
                contingency[label * cluster_count + cluster];

            if (cell_count != 0) {
                long double cell_value = (long double)cell_count;
                long double joint_probability = cell_value / row_count;
                long double label_total = (long double)label_totals[label];
                long double cluster_total =
                    (long double)cluster_totals[cluster];

                mutual_information += joint_probability * logl(
                    (cell_value * row_count) /
                    (label_total * cluster_total)
                );
                label_given_cluster -= joint_probability * logl(
                    cell_value / cluster_total
                );
                cluster_given_label -= joint_probability * logl(
                    cell_value / label_total
                );
                sum_cells_choose_two += gs_metrics_choose_two(cell_count);
            }
        }
    }

    if (!isfinite(label_entropy) || !isfinite(cluster_entropy) ||
        !isfinite(mutual_information) ||
        !isfinite(label_given_cluster) ||
        !isfinite(cluster_given_label) ||
        !isfinite(sum_cells_choose_two) ||
        !isfinite(sum_labels_choose_two) ||
        !isfinite(sum_clusters_choose_two)) {

        goto cleanup;
    }

    if (label_entropy + cluster_entropy == 0.0L) {
        nmi_value = 1.0L;
    } else {
        nmi_value = 2.0L * mutual_information /
            (label_entropy + cluster_entropy);
    }

    homogeneity_value = label_entropy == 0.0L
        ? 1.0L
        : 1.0L - label_given_cluster / label_entropy;
    completeness_value = cluster_entropy == 0.0L
        ? 1.0L
        : 1.0L - cluster_given_label / cluster_entropy;

    total_choose_two = gs_metrics_choose_two((uint64_t)rows);
    if (total_choose_two == 0.0L) {
        expected_index = 0.0L;
    } else {
        expected_index =
            sum_labels_choose_two * sum_clusters_choose_two /
            total_choose_two;
    }
    maximum_index =
        (sum_labels_choose_two + sum_clusters_choose_two) / 2.0L;
    ari_denominator = maximum_index - expected_index;
    ari_numerator = sum_cells_choose_two - expected_index;

    if (fabsl(ari_denominator) <=
        16.0L * LDBL_EPSILON *
        (1.0L + fabsl(maximum_index) + fabsl(expected_index))) {

        ari_value = gs_metrics_partitions_are_equivalent(
            contingency,
            label_count,
            cluster_count
        ) ? 1.0L : 0.0L;
    } else {
        ari_value = ari_numerator / ari_denominator;
    }

    if (!gs_metrics_to_unit_interval(
            (long double)majority_sum / row_count,
            &purity
        ) ||
        !gs_metrics_to_unit_interval(
            nmi_value,
            &normalized_mutual_information
        ) ||
        !gs_metrics_to_signed_unit_interval(
            ari_value,
            &adjusted_rand_index
        ) ||
        !gs_metrics_to_unit_interval(
            homogeneity_value,
            &homogeneity
        ) ||
        !gs_metrics_to_unit_interval(
            completeness_value,
            &completeness
        )) {

        goto cleanup;
    }

    result->rows = rows;
    result->label_count = label_count;
    result->cluster_count = cluster_count;
    result->contingency = contingency;
    result->label_totals = label_totals;
    result->cluster_totals = cluster_totals;
    result->purity = purity;
    result->normalized_mutual_information =
        normalized_mutual_information;
    result->adjusted_rand_index = adjusted_rand_index;
    result->homogeneity = homogeneity;
    result->completeness = completeness;

    return GS_BASELINE_CLUSTERING_METRICS_SUCCESS;

cleanup:
    free(cluster_totals);
    free(label_totals);
    free(contingency);
    gs_metrics_result_zero(result);
    return error;
}
