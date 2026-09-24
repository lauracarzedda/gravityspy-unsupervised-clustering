#include "baseline_kmeans.h"

#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


enum {
    GS_SMOKE_PATH_LENGTH = 4096,
    GS_SMOKE_LINE_LENGTH = 4096,
    GS_SMOKE_FULL_TRAIN_ROWS = 6866,
    GS_SMOKE_FULL_VALIDATION_ROWS = 1717,
    GS_SMOKE_TRAIN_ROWS = 128,
    GS_SMOKE_VALIDATION_ROWS = 32,
    GS_SMOKE_COMPONENTS = 62,
    GS_SMOKE_K = 4,
    GS_SMOKE_SEED = 123,
    GS_SMOKE_MAX_ITERATIONS = 100
};


static int gs_checked_product(size_t left, size_t right, size_t *product)
{
    if (product == NULL || (right != 0 && left > SIZE_MAX / right)) {
        return 1;
    }
    *product = left * right;
    return 0;
}


static int gs_platform_is_valid(void)
{
    const uint16_t marker = UINT16_C(1);
    const unsigned char *bytes = (const unsigned char *)&marker;

    return sizeof(float) == 4 &&
           FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 &&
           sizeof(double) == 8 &&
           DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024 &&
           bytes[0] == 1u;
}


static int gs_path_join(
    const char *directory,
    const char *filename,
    char *path,
    size_t path_size
)
{
    size_t directory_length;
    size_t filename_length;
    size_t required;
    size_t position = 0;
    int separator;

    if (directory == NULL || filename == NULL || path == NULL ||
        path_size == 0) {

        return 1;
    }
    directory_length = strlen(directory);
    filename_length = strlen(filename);
    separator = directory_length != 0 &&
        directory[directory_length - 1] != '/' &&
        directory[directory_length - 1] != '\\';

    if (directory_length > SIZE_MAX - filename_length) {
        return 1;
    }
    required = directory_length + filename_length;
    if (separator) {
        if (required == SIZE_MAX) {
            return 1;
        }
        required++;
    }
    if (required >= path_size) {
        return 1;
    }
    if (directory_length != 0) {
        memcpy(path, directory, directory_length);
        position = directory_length;
    }
    if (separator) {
        path[position++] = '/';
    }
    if (filename_length != 0) {
        memcpy(path + position, filename, filename_length);
        position += filename_length;
    }
    path[position] = '\0';
    return 0;
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


static void gs_strip_line_ending(char *line)
{
    size_t length = strlen(line);

    while (length != 0 &&
           (line[length - 1] == '\r' || line[length - 1] == '\n')) {

        line[--length] = '\0';
    }
}


static int gs_parse_threshold_row(
    char *line,
    const char *expected_threshold,
    size_t expected_count
)
{
    char *first_comma;
    char *second_comma;
    char *end;
    unsigned long long count;
    double cumulative;

    first_comma = strchr(line, ',');
    if (first_comma == NULL) {
        return 1;
    }
    *first_comma = '\0';
    second_comma = strchr(first_comma + 1, ',');
    if (second_comma == NULL || strchr(second_comma + 1, ',') != NULL ||
        strcmp(line, expected_threshold) != 0) {

        return 1;
    }
    *second_comma = '\0';
    errno = 0;
    end = NULL;
    count = strtoull(first_comma + 1, &end, 10);
    if (errno == ERANGE || end == first_comma + 1 || *end != '\0' ||
        count != (unsigned long long)expected_count) {

        return 1;
    }
    errno = 0;
    end = NULL;
    cumulative = strtod(second_comma + 1, &end);
    if (errno == ERANGE || end == second_comma + 1 || *end != '\0' ||
        !isfinite(cumulative) || cumulative < 0.0 || cumulative > 1.0) {

        return 1;
    }
    return 0;
}


static int gs_validate_thresholds(const char *directory)
{
    static const char *thresholds[3] = {"0.90", "0.95", "0.99"};
    static const size_t counts[3] = {21, 62, 263};
    char path[GS_SMOKE_PATH_LENGTH];
    char line[GS_SMOKE_LINE_LENGTH];
    FILE *file;
    size_t row;
    int failed = 0;

    if (gs_path_join(
            directory, "thresholds.csv", path, sizeof(path)
        ) != 0) {

        return 1;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        return 1;
    }
    if (fgets(line, sizeof(line), file) == NULL) {
        failed = 1;
    } else {
        gs_strip_line_ending(line);
        if (strcmp(
                line,
                "threshold,component_count,cumulative_variance"
            ) != 0) {

            failed = 1;
        }
    }
    for (row = 0; !failed && row < 3; row++) {
        if (fgets(line, sizeof(line), file) == NULL) {
            failed = 1;
            break;
        }
        gs_strip_line_ending(line);
        if (gs_parse_threshold_row(
                line, thresholds[row], counts[row]
            ) != 0) {

            failed = 1;
        }
    }
    if (!failed && fgets(line, sizeof(line), file) != NULL) {
        failed = 1;
    }
    if (ferror(file) || fclose(file) != 0) {
        failed = 1;
    }
    return failed;
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
    size_t centroid_bytes;
    size_t assignment_bytes;
    size_t cluster_bytes;

    if (left == NULL || right == NULL ||
        gs_checked_product(
            left->k, left->cols, &centroid_bytes
        ) != 0 ||
        gs_checked_product(centroid_bytes, sizeof(double), &centroid_bytes) != 0 ||
        gs_checked_product(
            left->rows, sizeof(size_t), &assignment_bytes
        ) != 0 ||
        gs_checked_product(left->k, sizeof(size_t), &cluster_bytes) != 0) {

        return 0;
    }
    return left->rows == right->rows &&
           left->cols == right->cols &&
           left->k == right->k &&
           left->iterations == right->iterations &&
           left->empty_cluster_reseeds == right->empty_cluster_reseeds &&
           left->converged == right->converged &&
           left->seed == right->seed &&
           memcmp(
               &left->initial_inertia,
               &right->initial_inertia,
               sizeof(double)
           ) == 0 &&
           memcmp(&left->inertia, &right->inertia, sizeof(double)) == 0 &&
           memcmp(left->centroids, right->centroids, centroid_bytes) == 0 &&
           memcmp(
               left->assignments, right->assignments, assignment_bytes
           ) == 0 &&
           memcmp(
               left->cluster_sizes, right->cluster_sizes, cluster_bytes
           ) == 0 &&
           memcmp(
               left->initial_centroid_indices,
               right->initial_centroid_indices,
               cluster_bytes
           ) == 0;
}


static int gs_train_result_is_valid(const GSBaselineKMeansResult *result)
{
    size_t index;
    size_t total = 0;

    if (result == NULL ||
        result->rows != GS_SMOKE_TRAIN_ROWS ||
        result->cols != GS_SMOKE_COMPONENTS ||
        result->k != GS_SMOKE_K ||
        result->iterations == 0 ||
        result->iterations > GS_SMOKE_MAX_ITERATIONS ||
        result->centroids == NULL ||
        result->assignments == NULL ||
        result->cluster_sizes == NULL ||
        result->initial_centroid_indices == NULL ||
        !isfinite(result->initial_inertia) ||
        result->initial_inertia < 0.0 ||
        !isfinite(result->inertia) ||
        result->inertia < 0.0) {

        return 0;
    }
    for (index = 0; index < result->rows; index++) {
        if (result->assignments[index] >= result->k) {
            return 0;
        }
    }
    for (index = 0; index < result->k * result->cols; index++) {
        if (!isfinite(result->centroids[index])) {
            return 0;
        }
    }
    for (index = 0; index < result->k; index++) {
        if (total > SIZE_MAX - result->cluster_sizes[index]) {
            return 0;
        }
        total += result->cluster_sizes[index];
    }
    return total == result->rows;
}


static void gs_print_usage(const char *program)
{
    printf("Usage: %s <pca_artifact_directory>\n", program);
}


int main(int argc, char **argv)
{
    char train_path[GS_SMOKE_PATH_LENGTH];
    char validation_path[GS_SMOKE_PATH_LENGTH];
    size_t full_train_values;
    size_t full_validation_values;
    size_t full_train_bytes;
    size_t full_validation_bytes;
    size_t train_values;
    size_t validation_values;
    size_t train_bytes;
    size_t validation_bytes;
    size_t centroid_values;
    size_t centroid_bytes;

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

    if (argc == 2 && strcmp(argv[1], "--help") == 0) {
        gs_print_usage(argv[0]);
        return 0;
    }
    if (argc != 2 || !gs_platform_is_valid()) {
        gs_print_usage(argv[0]);
        return 1;
    }
    if (gs_validate_thresholds(argv[1]) != 0 ||
        gs_path_join(
            argv[1], "train_scores_pca95_f32.bin",
            train_path, sizeof(train_path)
        ) != 0 ||
        gs_path_join(
            argv[1], "val_scores_pca95_f32.bin",
            validation_path, sizeof(validation_path)
        ) != 0 ||
        gs_checked_product(
            GS_SMOKE_FULL_TRAIN_ROWS,
            GS_SMOKE_COMPONENTS,
            &full_train_values
        ) != 0 ||
        gs_checked_product(
            GS_SMOKE_FULL_VALIDATION_ROWS,
            GS_SMOKE_COMPONENTS,
            &full_validation_values
        ) != 0 ||
        gs_checked_product(
            full_train_values, sizeof(float), &full_train_bytes
        ) != 0 ||
        gs_checked_product(
            full_validation_values, sizeof(float), &full_validation_bytes
        ) != 0) {

        fprintf(stderr, "Errore: contratto input PCA-95 non valido.\n");
        goto cleanup;
    }
    printf("[PASS] pca95_threshold_and_input_contract\n");

    if (gs_checked_product(
            GS_SMOKE_TRAIN_ROWS, GS_SMOKE_COMPONENTS, &train_values
        ) != 0 ||
        gs_checked_product(
            GS_SMOKE_VALIDATION_ROWS,
            GS_SMOKE_COMPONENTS,
            &validation_values
        ) != 0 ||
        gs_checked_product(train_values, sizeof(float), &train_bytes) != 0 ||
        gs_checked_product(
            validation_values, sizeof(float), &validation_bytes
        ) != 0 ||
        gs_checked_product(
            GS_SMOKE_K, GS_SMOKE_COMPONENTS, &centroid_values
        ) != 0 ||
        gs_checked_product(
            centroid_values, sizeof(double), &centroid_bytes
        ) != 0) {

        fprintf(stderr, "Errore: overflow smoke.\n");
        goto cleanup;
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
    if (train == NULL || validation == NULL ||
        train_copy == NULL || validation_copy == NULL ||
        centroid_copy == NULL || validation_assignments == NULL ||
        validation_distances == NULL) {

        fprintf(stderr, "Errore: memoria smoke insufficiente.\n");
        goto cleanup;
    }
    if (gs_read_prefix(
            train_path, full_train_bytes, train, train_values
        ) != 0 ||
        gs_read_prefix(
            validation_path,
            full_validation_bytes,
            validation,
            validation_values
        ) != 0) {

        fprintf(stderr, "Errore: lettura prefissi PCA-95 fallita.\n");
        goto cleanup;
    }
    memcpy(train_copy, train, train_bytes);
    memcpy(validation_copy, validation, validation_bytes);
    printf("[PASS] real_pca95_prefix_load\n");

    if (gs_baseline_kmeans_fit(
            train,
            GS_SMOKE_TRAIN_ROWS,
            GS_SMOKE_COMPONENTS,
            GS_SMOKE_K,
            GS_SMOKE_SEED,
            GS_SMOKE_MAX_ITERATIONS,
            &first
        ) != GS_BASELINE_KMEANS_SUCCESS ||
        gs_baseline_kmeans_fit(
            train,
            GS_SMOKE_TRAIN_ROWS,
            GS_SMOKE_COMPONENTS,
            GS_SMOKE_K,
            GS_SMOKE_SEED,
            GS_SMOKE_MAX_ITERATIONS,
            &second
        ) != GS_BASELINE_KMEANS_SUCCESS ||
        !gs_results_are_identical(&first, &second)) {

        fprintf(stderr, "Errore: risultati non deterministici.\n");
        goto cleanup;
    }
    printf("[PASS] deterministic_repeat\n");

    if (!gs_train_result_is_valid(&first) ||
        memcmp(train, train_copy, train_bytes) != 0 ||
        memcmp(validation, validation_copy, validation_bytes) != 0) {

        fprintf(stderr, "Errore: contratto train-only non valido.\n");
        goto cleanup;
    }
    printf("[PASS] train_fit_only\n");

    memcpy(centroid_copy, first.centroids, centroid_bytes);
    if (gs_baseline_kmeans_assign(
            validation,
            GS_SMOKE_VALIDATION_ROWS,
            GS_SMOKE_COMPONENTS,
            first.centroids,
            GS_SMOKE_K,
            validation_assignments,
            validation_distances
        ) != GS_BASELINE_KMEANS_SUCCESS) {

        fprintf(stderr, "Errore: assegnazione validation fallita.\n");
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

            fprintf(stderr, "Errore: contratto numerico non valido.\n");
            goto cleanup;
        }
    }
    printf("[PASS] numeric_contract\n");

    if (memcmp(train, train_copy, train_bytes) != 0 ||
        memcmp(validation, validation_copy, validation_bytes) != 0) {

        fprintf(stderr, "Errore: score PCA riscalati o modificati.\n");
        goto cleanup;
    }
    printf("[PASS] no_post_pca_scaling\n");

    printf("TRAIN_ROWS_SMOKE=128\n");
    printf("VALIDATION_ROWS_SMOKE=32\n");
    printf("PCA_COMPONENTS=62\n");
    printf("K_SMOKE=4\n");
    printf("ALL PCA KMEANS SMOKE TESTS PASSED\n");
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
