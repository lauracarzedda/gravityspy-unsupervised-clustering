#include "baseline_kmeans.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


enum {
    GS_SMOKE_TRAIN_ROWS = 128,
    GS_SMOKE_VALIDATION_ROWS = 32,
    GS_SMOKE_FEATURE_COUNT = 900,
    GS_SMOKE_K = 4,
    GS_SMOKE_SEED = 123,
    GS_SMOKE_MAX_ITERATIONS = 100
};


static int gs_platform_is_valid(void)
{
    uint16_t value = UINT16_C(1);
    const unsigned char *bytes = (const unsigned char *)&value;

    return sizeof(float) == 4 && FLT_RADIX == 2 &&
           FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 &&
           sizeof(double) == 8 && DBL_MANT_DIG == 53 &&
           DBL_MAX_EXP == 1024 && bytes[0] == 1u;
}


static int gs_file_size(const char *path, size_t *size)
{
    FILE *file;
    long position;
    int failed = 0;

    if (path == NULL || size == NULL) {
        return 1;
    }

    file = fopen(path, "rb");
    if (file == NULL) {
        return 1;
    }
    if (fseek(file, 0L, SEEK_END) != 0) {
        failed = 1;
    }
    position = failed ? -1L : ftell(file);
    if (position < 0) {
        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    if (failed) {
        return 1;
    }
    *size = (size_t)position;
    return 0;
}


static int gs_read_prefix(
    const char *path,
    size_t expected_file_size,
    float *values,
    size_t value_count
)
{
    FILE *file;
    size_t actual_size;
    size_t index;
    int failed = 0;

    if (path == NULL || values == NULL ||
        gs_file_size(path, &actual_size) != 0 ||
        actual_size != expected_file_size) {

        return 1;
    }

    file = fopen(path, "rb");
    if (file == NULL) {
        return 1;
    }
    if (fread(values, sizeof(float), value_count, file) != value_count ||
        ferror(file)) {

        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    if (failed) {
        return 1;
    }

    for (index = 0; index < value_count; index++) {
        if (!isfinite(values[index])) {
            return 1;
        }
    }
    return 0;
}


static int gs_results_are_identical(
    const GSBaselineKMeansResult *left,
    const GSBaselineKMeansResult *right
)
{
    return left->rows == right->rows &&
           left->cols == right->cols && left->k == right->k &&
           left->iterations == right->iterations &&
           left->empty_cluster_reseeds == right->empty_cluster_reseeds &&
           left->converged == right->converged && left->seed == right->seed &&
           memcmp(
               &left->initial_inertia,
               &right->initial_inertia,
               sizeof(double)
           ) == 0 &&
           memcmp(&left->inertia, &right->inertia, sizeof(double)) == 0 &&
           memcmp(
               left->centroids,
               right->centroids,
               left->k * left->cols * sizeof(double)
           ) == 0 &&
           memcmp(
               left->assignments,
               right->assignments,
               left->rows * sizeof(size_t)
           ) == 0 &&
           memcmp(
               left->cluster_sizes,
               right->cluster_sizes,
               left->k * sizeof(size_t)
           ) == 0 &&
           memcmp(
               left->initial_centroid_indices,
               right->initial_centroid_indices,
               left->k * sizeof(size_t)
           ) == 0;
}


static int gs_train_result_is_valid(const GSBaselineKMeansResult *result)
{
    size_t row;
    size_t cluster;
    size_t total = 0;

    if (result == NULL || result->rows != GS_SMOKE_TRAIN_ROWS ||
        result->cols != GS_SMOKE_FEATURE_COUNT || result->k != GS_SMOKE_K ||
        result->iterations == 0 || result->iterations > GS_SMOKE_MAX_ITERATIONS ||
        result->centroids == NULL || result->assignments == NULL ||
        result->cluster_sizes == NULL ||
        !isfinite(result->initial_inertia) || result->initial_inertia < 0.0 ||
        !isfinite(result->inertia) || result->inertia < 0.0) {

        return 0;
    }

    for (row = 0; row < result->rows; row++) {
        if (result->assignments[row] >= result->k) {
            return 0;
        }
    }
    for (row = 0; row < result->k * result->cols; row++) {
        if (!isfinite(result->centroids[row])) {
            return 0;
        }
    }
    for (cluster = 0; cluster < result->k; cluster++) {
        total += result->cluster_sizes[cluster];
    }
    return total == result->rows;
}


int main(int argc, char **argv)
{
    size_t train_value_count =
        GS_SMOKE_TRAIN_ROWS * GS_SMOKE_FEATURE_COUNT;
    size_t validation_value_count =
        GS_SMOKE_VALIDATION_ROWS * GS_SMOKE_FEATURE_COUNT;
    size_t train_bytes = train_value_count * sizeof(float);
    size_t validation_bytes = validation_value_count * sizeof(float);
    size_t centroid_bytes =
        GS_SMOKE_K * GS_SMOKE_FEATURE_COUNT * sizeof(double);

    float *train = NULL;
    float *validation = NULL;
    float *train_copy = NULL;
    float *validation_copy = NULL;
    double *centroid_copy = NULL;
    size_t *validation_assignments = NULL;
    double *validation_distances = NULL;
    GSBaselineKMeansResult first;
    GSBaselineKMeansResult second;
    size_t row;
    int rc = 1;

    memset(&first, 0, sizeof(first));
    memset(&second, 0, sizeof(second));

    if (argc != 3 || !gs_platform_is_valid()) {
        fprintf(stderr, "Errore: argomenti o piattaforma smoke non validi.\n");
        return 1;
    }

    train = (float *)malloc(train_bytes);
    validation = (float *)malloc(validation_bytes);
    train_copy = (float *)malloc(train_bytes);
    validation_copy = (float *)malloc(validation_bytes);
    centroid_copy = (double *)malloc(centroid_bytes);
    validation_assignments = (size_t *)malloc(
        GS_SMOKE_VALIDATION_ROWS * sizeof(size_t)
    );
    validation_distances = (double *)malloc(
        GS_SMOKE_VALIDATION_ROWS * sizeof(double)
    );

    if (train == NULL || validation == NULL || train_copy == NULL ||
        validation_copy == NULL || centroid_copy == NULL ||
        validation_assignments == NULL || validation_distances == NULL) {

        fprintf(stderr, "Errore: memoria smoke insufficiente.\n");
        goto cleanup;
    }

    if (gs_read_prefix(argv[1], 24717600u, train, train_value_count) != 0 ||
        gs_read_prefix(
            argv[2],
            6181200u,
            validation,
            validation_value_count
        ) != 0) {

        fprintf(stderr, "Errore: lettura prefissi reali fallita.\n");
        goto cleanup;
    }
    memcpy(train_copy, train, train_bytes);
    memcpy(validation_copy, validation, validation_bytes);
    printf("[PASS] real_standardized_prefix_load\n");

    if (gs_baseline_kmeans_fit(
            train,
            GS_SMOKE_TRAIN_ROWS,
            GS_SMOKE_FEATURE_COUNT,
            GS_SMOKE_K,
            GS_SMOKE_SEED,
            GS_SMOKE_MAX_ITERATIONS,
            &first
        ) != GS_BASELINE_KMEANS_SUCCESS ||
        gs_baseline_kmeans_fit(
            train,
            GS_SMOKE_TRAIN_ROWS,
            GS_SMOKE_FEATURE_COUNT,
            GS_SMOKE_K,
            GS_SMOKE_SEED,
            GS_SMOKE_MAX_ITERATIONS,
            &second
        ) != GS_BASELINE_KMEANS_SUCCESS ||
        !gs_results_are_identical(&first, &second)) {

        fprintf(stderr, "Errore: smoke non deterministico.\n");
        goto cleanup;
    }
    printf("[PASS] deterministic_repeat\n");

    if (!gs_train_result_is_valid(&first) ||
        memcmp(train, train_copy, train_bytes) != 0 ||
        memcmp(validation, validation_copy, validation_bytes) != 0) {

        fprintf(stderr, "Errore: contratto fit train-only non valido.\n");
        goto cleanup;
    }
    printf("[PASS] train_fit_only\n");

    memcpy(centroid_copy, first.centroids, centroid_bytes);
    if (gs_baseline_kmeans_assign(
            validation,
            GS_SMOKE_VALIDATION_ROWS,
            GS_SMOKE_FEATURE_COUNT,
            first.centroids,
            GS_SMOKE_K,
            validation_assignments,
            validation_distances
        ) != GS_BASELINE_KMEANS_SUCCESS) {

        fprintf(stderr, "Errore: assegnazione validation smoke fallita.\n");
        goto cleanup;
    }
    printf("[PASS] validation_assignment\n");

    if (memcmp(centroid_copy, first.centroids, centroid_bytes) != 0) {
        fprintf(stderr, "Errore: centroidi modificati dalla validation.\n");
        goto cleanup;
    }
    printf("[PASS] centroids_unchanged_after_validation\n");

    for (row = 0; row < GS_SMOKE_VALIDATION_ROWS; row++) {
        if (validation_assignments[row] >= GS_SMOKE_K ||
            !isfinite(validation_distances[row]) ||
            validation_distances[row] < 0.0) {

            fprintf(stderr, "Errore: contratto numerico smoke non valido.\n");
            goto cleanup;
        }
    }

    if (memcmp(train, train_copy, train_bytes) != 0 ||
        memcmp(validation, validation_copy, validation_bytes) != 0) {

        fprintf(stderr, "Errore: input smoke modificato.\n");
        goto cleanup;
    }
    printf("[PASS] numeric_contract\n");

    printf("TRAIN_ROWS_SMOKE=128\n");
    printf("VALIDATION_ROWS_SMOKE=32\n");
    printf("K_SMOKE=4\n");
    printf("ALL SMOKE TESTS PASSED\n");
    rc = 0;

cleanup:
    gs_baseline_kmeans_result_free(&second);
    gs_baseline_kmeans_result_free(&first);
    free(validation_distances);
    free(validation_assignments);
    free(centroid_copy);
    free(validation_copy);
    free(train_copy);
    free(validation);
    free(train);
    return rc;
}
