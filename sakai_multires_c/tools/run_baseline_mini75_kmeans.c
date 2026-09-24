#include "baseline_kmeans.h"

#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/types.h>
#endif


enum {
    GS_RUN_PATH_LEN = 4096,
    GS_RUN_TRAIN_ROWS = 6866,
    GS_RUN_VALIDATION_ROWS = 1717,
    GS_RUN_FEATURE_COUNT = 900,
    GS_RUN_SEED = 123,
    GS_RUN_MAX_ITERATIONS = 300
};


#define GS_RUN_PARTIAL_SUFFIX ".partial"


typedef struct {
    double minimum;
    double maximum;
    double mean;
    double sum;
} GSDistanceStatistics;


static int gs_checked_product(
    size_t left,
    size_t right,
    size_t *product
)
{
    if (product == NULL ||
        (right != 0 && left > SIZE_MAX / right)) {

        return 1;
    }

    *product = left * right;
    return 0;
}


static int gs_platform_is_valid(void)
{
    uint16_t value = UINT16_C(1);
    const unsigned char *bytes = (const unsigned char *)&value;

    return sizeof(float) == 4 &&
           FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 &&
           sizeof(double) == 8 &&
           DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024 &&
           sizeof(uint32_t) == 4 && sizeof(uint64_t) == 8 &&
           bytes[0] == 1u;
}


static int gs_parse_size_value(const char *text, size_t *value)
{
    char *endptr;
    unsigned long parsed;

    if (text == NULL || value == NULL || text[0] == '\0' ||
        text[0] == '-') {

        return 1;
    }

    errno = 0;
    endptr = NULL;
    parsed = strtoul(text, &endptr, 10);

    if (errno == ERANGE || endptr == text || *endptr != '\0') {
        return 1;
    }

    *value = (size_t)parsed;
    return 0;
}


static int gs_parse_u64_value(const char *text, uint64_t *value)
{
    char *endptr;
    unsigned long long parsed;

    if (text == NULL || value == NULL || text[0] == '\0' ||
        text[0] == '-') {

        return 1;
    }

    errno = 0;
    endptr = NULL;
    parsed = strtoull(text, &endptr, 10);

    if (errno == ERANGE || endptr == text || *endptr != '\0') {
        return 1;
    }

    *value = (uint64_t)parsed;
    return 0;
}


static int gs_metadata_value_is_valid(const char *value)
{
    return value != NULL && value[0] != '\0' &&
           strchr(value, '\r') == NULL && strchr(value, '\n') == NULL;
}


static int gs_commit_is_valid(const char *commit)
{
    size_t index;

    if (!gs_metadata_value_is_valid(commit) || strlen(commit) != 40) {
        return 0;
    }

    for (index = 0; index < 40; index++) {
        if (!((commit[index] >= '0' && commit[index] <= '9') ||
              (commit[index] >= 'a' && commit[index] <= 'f'))) {

            return 0;
        }
    }

    return 1;
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
    size_t position;
    int needs_separator;

    if (directory == NULL || filename == NULL || path == NULL ||
        path_size == 0) {

        return 1;
    }

    directory_length = strlen(directory);
    filename_length = strlen(filename);
    needs_separator = directory_length != 0 &&
        directory[directory_length - 1] != '/' &&
        directory[directory_length - 1] != '\\';

    if (directory_length > SIZE_MAX - filename_length) {
        return 1;
    }

    required = directory_length + filename_length;
    if (needs_separator) {
        if (required == SIZE_MAX) {
            return 1;
        }
        required++;
    }

    if (required >= path_size) {
        return 1;
    }

    position = 0;
    if (directory_length != 0) {
        memcpy(path, directory, directory_length);
        position = directory_length;
    }
    if (needs_separator) {
        path[position++] = '/';
    }
    if (filename_length != 0) {
        memcpy(path + position, filename, filename_length);
        position += filename_length;
    }
    path[position] = '\0';
    return 0;
}


static int gs_path_state(const char *path)
{
    struct stat information;

    errno = 0;
    if (stat(path, &information) == 0) {
        return 1;
    }
    return errno == ENOENT ? 0 : -1;
}


static int gs_validate_output_paths(
    const char *partial_path,
    char *final_path,
    size_t final_path_size
)
{
    size_t partial_length;
    size_t suffix_length = strlen(GS_RUN_PARTIAL_SUFFIX);
    size_t final_length;

    if (!gs_metadata_value_is_valid(partial_path) || final_path == NULL ||
        final_path_size == 0) {

        return 1;
    }

    partial_length = strlen(partial_path);
    if (partial_length <= suffix_length ||
        strcmp(
            partial_path + partial_length - suffix_length,
            GS_RUN_PARTIAL_SUFFIX
        ) != 0) {

        fprintf(stderr, "Errore: output deve terminare con .partial.\n");
        return 1;
    }

    final_length = partial_length - suffix_length;
    if (final_length == 0 || final_length >= final_path_size) {
        return 1;
    }

    memcpy(final_path, partial_path, final_length);
    final_path[final_length] = '\0';

    if (gs_path_state(partial_path) != 0 || gs_path_state(final_path) != 0) {
        fprintf(stderr, "Errore: directory partial o finale gia' esistente.\n");
        return 1;
    }

    return 0;
}


static int gs_create_partial_directory(
    const char *partial_path,
    const char *final_path
)
{
    if (gs_path_state(partial_path) != 0 || gs_path_state(final_path) != 0) {
        return 1;
    }

#ifdef _WIN32
    if (_mkdir(partial_path) != 0) {
#else
    if (mkdir(partial_path, 0777) != 0) {
#endif
        fprintf(stderr, "Errore: creazione directory partial fallita.\n");
        return 1;
    }

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


static int gs_read_float_matrix(
    const char *path,
    float *values,
    size_t value_count,
    size_t expected_bytes
)
{
    FILE *file;
    size_t actual_bytes;
    size_t index;
    int failed = 0;

    if (path == NULL || values == NULL ||
        gs_file_size(path, &actual_bytes) != 0 ||
        actual_bytes != expected_bytes) {

        return 1;
    }

    file = fopen(path, "rb");
    if (file == NULL) {
        return 1;
    }

    if (fread(values, sizeof(float), value_count, file) != value_count ||
        fgetc(file) != EOF || ferror(file)) {

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


static int gs_finish_write(FILE *file)
{
    int failed = 0;

    if (file == NULL) {
        return 1;
    }
    if (fflush(file) != 0 || ferror(file)) {
        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    return failed;
}


static int gs_open_output(
    const char *directory,
    const char *filename,
    const char *mode,
    FILE **file
)
{
    char path[GS_RUN_PATH_LEN];

    if (file == NULL ||
        gs_path_join(directory, filename, path, sizeof(path)) != 0) {

        return 1;
    }

    *file = fopen(path, mode);
    return *file == NULL ? 1 : 0;
}


static int gs_write_binary(
    const char *directory,
    const char *filename,
    const void *values,
    size_t element_size,
    size_t count
)
{
    FILE *file = NULL;
    int failed = 0;

    if (values == NULL || element_size == 0 ||
        gs_open_output(directory, filename, "wb", &file) != 0) {

        return 1;
    }

    if (fwrite(values, element_size, count, file) != count) {
        failed = 1;
    }
    if (gs_finish_write(file) != 0) {
        failed = 1;
    }
    return failed;
}


static int gs_distance_statistics(
    const double *distances,
    size_t count,
    GSDistanceStatistics *statistics
)
{
    size_t index;
    double total = 0.0;

    if (distances == NULL || statistics == NULL || count == 0) {
        return 1;
    }

    statistics->minimum = distances[0];
    statistics->maximum = distances[0];

    for (index = 0; index < count; index++) {
        double value = distances[index];

        if (!isfinite(value) || value < 0.0) {
            return 1;
        }
        total += value;
        if (!isfinite(total)) {
            return 1;
        }
        if (value < statistics->minimum) {
            statistics->minimum = value;
        }
        if (value > statistics->maximum) {
            statistics->maximum = value;
        }
    }

    statistics->sum = total;
    statistics->mean = total / (double)count;
    return isfinite(statistics->mean) ? 0 : 1;
}


static int gs_validate_and_convert(
    const GSBaselineKMeansResult *result,
    const size_t *validation_assignments,
    size_t validation_rows,
    const double *train_distances,
    const double *validation_distances,
    uint32_t *train_u32,
    uint32_t *validation_u32,
    uint64_t *cluster_sizes_u64,
    uint64_t *initial_indices_u64,
    GSDistanceStatistics *train_statistics,
    GSDistanceStatistics *validation_statistics,
    size_t *empty_clusters,
    size_t *minimum_cluster_size,
    size_t *maximum_cluster_size
)
{
    size_t *recomputed_sizes;
    size_t row;
    size_t cluster;
    size_t total_size = 0;
    double tolerance;

    if (result == NULL || validation_assignments == NULL ||
        train_distances == NULL || validation_distances == NULL ||
        train_u32 == NULL || validation_u32 == NULL ||
        cluster_sizes_u64 == NULL || initial_indices_u64 == NULL ||
        train_statistics == NULL || validation_statistics == NULL ||
        empty_clusters == NULL || minimum_cluster_size == NULL ||
        maximum_cluster_size == NULL ||
        result->k > (size_t)UINT32_MAX) {

        return 1;
    }

    recomputed_sizes = (size_t *)calloc(result->k, sizeof(size_t));
    if (recomputed_sizes == NULL) {
        return 1;
    }

    for (row = 0; row < result->rows; row++) {
        size_t assignment = result->assignments[row];

        if (assignment >= result->k || assignment > (size_t)UINT32_MAX) {
            free(recomputed_sizes);
            return 1;
        }
        train_u32[row] = (uint32_t)assignment;
        recomputed_sizes[assignment]++;
    }

    for (row = 0; row < validation_rows; row++) {
        size_t assignment = validation_assignments[row];

        if (assignment >= result->k || assignment > (size_t)UINT32_MAX) {
            free(recomputed_sizes);
            return 1;
        }
        validation_u32[row] = (uint32_t)assignment;
    }

    *empty_clusters = 0;
    *minimum_cluster_size = result->rows;
    *maximum_cluster_size = 0;

    for (cluster = 0; cluster < result->k; cluster++) {
        size_t size = result->cluster_sizes[cluster];
        size_t left;

        if (size != recomputed_sizes[cluster] ||
            total_size > SIZE_MAX - size) {

            free(recomputed_sizes);
            return 1;
        }
        total_size += size;
        cluster_sizes_u64[cluster] = (uint64_t)size;

        if (size == 0) {
            (*empty_clusters)++;
        }
        if (size < *minimum_cluster_size) {
            *minimum_cluster_size = size;
        }
        if (size > *maximum_cluster_size) {
            *maximum_cluster_size = size;
        }

        if (result->initial_centroid_indices[cluster] >= result->rows) {
            free(recomputed_sizes);
            return 1;
        }
        initial_indices_u64[cluster] =
            (uint64_t)result->initial_centroid_indices[cluster];

        for (left = 0; left < cluster; left++) {
            if (result->initial_centroid_indices[left] ==
                result->initial_centroid_indices[cluster]) {

                free(recomputed_sizes);
                return 1;
            }
        }
    }

    free(recomputed_sizes);

    if (total_size != result->rows ||
        !isfinite(result->initial_inertia) || result->initial_inertia < 0.0 ||
        !isfinite(result->inertia) || result->inertia < 0.0 ||
        gs_distance_statistics(
            train_distances,
            result->rows,
            train_statistics
        ) != 0 ||
        gs_distance_statistics(
            validation_distances,
            validation_rows,
            validation_statistics
        ) != 0) {

        return 1;
    }

    tolerance = 1.0e-10 * (1.0 + fabs(result->inertia));
    if (fabs(train_statistics->sum - result->inertia) > tolerance) {
        return 1;
    }

    return 0;
}


static int gs_write_assignment_csv(
    const char *directory,
    const char *filename,
    const uint32_t *assignments,
    const double *distances,
    size_t rows
)
{
    FILE *file = NULL;
    size_t row;
    int failed = 0;

    if (assignments == NULL || distances == NULL ||
        gs_open_output(directory, filename, "wb", &file) != 0) {

        return 1;
    }

    if (fputs("row_index,cluster,squared_distance\n", file) == EOF) {
        failed = 1;
    }

    for (row = 0; !failed && row < rows; row++) {
        if (fprintf(
                file,
                "%lu,%lu,%.17g\n",
                (unsigned long)row,
                (unsigned long)assignments[row],
                distances[row]
            ) < 0) {

            failed = 1;
        }
    }

    if (gs_finish_write(file) != 0) {
        failed = 1;
    }
    return failed;
}


static int gs_write_cluster_sizes_csv(
    const char *directory,
    const uint64_t *sizes,
    size_t k
)
{
    FILE *file = NULL;
    size_t cluster;
    int failed = 0;

    if (sizes == NULL ||
        gs_open_output(directory, "cluster_sizes.csv", "wb", &file) != 0) {

        return 1;
    }

    if (fputs("cluster,train_size\n", file) == EOF) {
        failed = 1;
    }
    for (cluster = 0; !failed && cluster < k; cluster++) {
        if (fprintf(
                file,
                "%lu,%lu\n",
                (unsigned long)cluster,
                (unsigned long)sizes[cluster]
            ) < 0) {

            failed = 1;
        }
    }
    if (gs_finish_write(file) != 0) {
        failed = 1;
    }
    return failed;
}


static int gs_write_initial_centroids_csv(
    const char *directory,
    const uint64_t *indices,
    size_t k
)
{
    FILE *file = NULL;
    size_t order;
    int failed = 0;

    if (indices == NULL ||
        gs_open_output(directory, "initial_centroids.csv", "wb", &file) != 0) {

        return 1;
    }

    if (fputs("selection_order,train_row_index\n", file) == EOF) {
        failed = 1;
    }
    for (order = 0; !failed && order < k; order++) {
        if (fprintf(
                file,
                "%lu,%lu\n",
                (unsigned long)order,
                (unsigned long)indices[order]
            ) < 0) {

            failed = 1;
        }
    }
    if (gs_finish_write(file) != 0) {
        failed = 1;
    }
    return failed;
}


static int gs_write_summary(
    const char *directory,
    size_t k,
    uint64_t seed,
    size_t max_iterations,
    const GSBaselineKMeansResult *result,
    const GSDistanceStatistics *train_statistics,
    const GSDistanceStatistics *validation_statistics,
    size_t empty_clusters,
    size_t minimum_cluster_size,
    size_t maximum_cluster_size,
    double cpu_seconds,
    const char *code_commit
)
{
    FILE *file = NULL;
    int failed = 0;

    if (result == NULL || train_statistics == NULL ||
        validation_statistics == NULL ||
        gs_open_output(directory, "summary.txt", "wb", &file) != 0) {

        return 1;
    }

    if (fprintf(
            file,
            "format_version=1\n"
            "profile=mini75_pool5_no_pca\n"
            "algorithm=kmeans_plus_plus_lloyd\n"
            "input=train_features_std_f32\n"
            "validation_policy=assign_to_train_centroids\n"
            "train_rows=6866\n"
            "validation_rows=1717\n"
            "feature_count=900\n"
            "k=%lu\n"
            "seed=%lu\n"
            "max_iterations=%lu\n"
            "iterations=%lu\n"
            "converged=%d\n"
            "empty_cluster_reseeds=%lu\n"
            "initial_inertia=%.17g\n"
            "final_train_inertia=%.17g\n"
            "validation_inertia=%.17g\n"
            "empty_final_clusters=%lu\n"
            "minimum_cluster_size=%lu\n"
            "maximum_cluster_size=%lu\n"
            "train_squared_distance_mean=%.17g\n"
            "train_squared_distance_minimum=%.17g\n"
            "train_squared_distance_maximum=%.17g\n"
            "validation_squared_distance_mean=%.17g\n"
            "validation_squared_distance_minimum=%.17g\n"
            "validation_squared_distance_maximum=%.17g\n"
            "fit_cpu_seconds_approx=%.17g\n"
            "centroid_dtype=float64\n"
            "assignment_dtype=uint32\n"
            "distance_dtype=float64\n"
            "cluster_size_dtype=uint64\n"
            "input_standardization=train_only\n"
            "fit_input=train_only\n"
            "validation_assigned_after_fit=true\n"
            "pca=false\n"
            "augmentation=false\n"
            "labels_used=false\n"
            "code_commit=%s\n",
            (unsigned long)k,
            (unsigned long)seed,
            (unsigned long)max_iterations,
            (unsigned long)result->iterations,
            result->converged,
            (unsigned long)result->empty_cluster_reseeds,
            result->initial_inertia,
            result->inertia,
            validation_statistics->sum,
            (unsigned long)empty_clusters,
            (unsigned long)minimum_cluster_size,
            (unsigned long)maximum_cluster_size,
            train_statistics->mean,
            train_statistics->minimum,
            train_statistics->maximum,
            validation_statistics->mean,
            validation_statistics->minimum,
            validation_statistics->maximum,
            cpu_seconds,
            code_commit
        ) < 0) {

        failed = 1;
    }

    if (gs_finish_write(file) != 0) {
        failed = 1;
    }
    return failed;
}


int main(int argc, char **argv)
{
    size_t train_rows;
    size_t validation_rows;
    size_t feature_count;
    size_t k;
    size_t max_iterations;
    uint64_t seed;
    size_t train_values;
    size_t validation_values;
    size_t centroid_values;
    size_t train_bytes;
    size_t validation_bytes;
    size_t centroid_bytes;
    size_t train_u32_bytes;
    size_t validation_u32_bytes;
    size_t validation_size_t_bytes;
    size_t train_distance_bytes;
    size_t validation_distance_bytes;
    size_t cluster_u64_bytes;
    char final_path[GS_RUN_PATH_LEN];

    float *train = NULL;
    float *validation = NULL;
    size_t *validation_assignments = NULL;
    double *train_distances = NULL;
    double *validation_distances = NULL;
    double *centroids_copy = NULL;
    uint32_t *train_u32 = NULL;
    uint32_t *validation_u32 = NULL;
    uint64_t *cluster_sizes_u64 = NULL;
    uint64_t *initial_indices_u64 = NULL;
    GSBaselineKMeansResult result;
    GSDistanceStatistics train_statistics;
    GSDistanceStatistics validation_statistics;
    size_t empty_clusters;
    size_t minimum_cluster_size;
    size_t maximum_cluster_size;
    clock_t start_clock;
    clock_t end_clock;
    double cpu_seconds;
    int rc = 1;

    memset(&result, 0, sizeof(result));

    if (argc != 11) {
        fprintf(stderr, "Errore: il runner richiede esattamente 10 argomenti.\n");
        return 1;
    }

    if (!gs_platform_is_valid() ||
        gs_parse_size_value(argv[3], &train_rows) != 0 ||
        gs_parse_size_value(argv[4], &validation_rows) != 0 ||
        gs_parse_size_value(argv[5], &feature_count) != 0 ||
        gs_parse_size_value(argv[6], &k) != 0 ||
        gs_parse_u64_value(argv[7], &seed) != 0 ||
        gs_parse_size_value(argv[8], &max_iterations) != 0 ||
        !gs_metadata_value_is_valid(argv[1]) ||
        !gs_metadata_value_is_valid(argv[2]) ||
        !gs_commit_is_valid(argv[10]) ||
        train_rows != GS_RUN_TRAIN_ROWS ||
        validation_rows != GS_RUN_VALIDATION_ROWS ||
        feature_count != GS_RUN_FEATURE_COUNT ||
        (k != 22 && k != 36) || seed != GS_RUN_SEED ||
        max_iterations != GS_RUN_MAX_ITERATIONS ||
        gs_validate_output_paths(argv[9], final_path, sizeof(final_path)) != 0) {

        fprintf(stderr, "Errore: argomenti o piattaforma non validi.\n");
        return 1;
    }

    if (gs_checked_product(train_rows, feature_count, &train_values) != 0 ||
        gs_checked_product(
            validation_rows,
            feature_count,
            &validation_values
        ) != 0 ||
        gs_checked_product(k, feature_count, &centroid_values) != 0 ||
        gs_checked_product(train_values, sizeof(float), &train_bytes) != 0 ||
        gs_checked_product(
            validation_values,
            sizeof(float),
            &validation_bytes
        ) != 0 ||
        gs_checked_product(
            centroid_values,
            sizeof(double),
            &centroid_bytes
        ) != 0 ||
        gs_checked_product(train_rows, sizeof(uint32_t), &train_u32_bytes) != 0 ||
        gs_checked_product(
            validation_rows,
            sizeof(uint32_t),
            &validation_u32_bytes
        ) != 0 ||
        gs_checked_product(
            validation_rows,
            sizeof(size_t),
            &validation_size_t_bytes
        ) != 0 ||
        gs_checked_product(
            train_rows,
            sizeof(double),
            &train_distance_bytes
        ) != 0 ||
        gs_checked_product(
            validation_rows,
            sizeof(double),
            &validation_distance_bytes
        ) != 0 ||
        gs_checked_product(k, sizeof(uint64_t), &cluster_u64_bytes) != 0 ||
        train_bytes != 24717600u || validation_bytes != 6181200u) {

        fprintf(stderr, "Errore: overflow o dimensioni input non valide.\n");
        return 1;
    }

    train = (float *)malloc(train_bytes);
    validation = (float *)malloc(validation_bytes);
    validation_assignments = (size_t *)malloc(validation_size_t_bytes);
    train_distances = (double *)malloc(train_distance_bytes);
    validation_distances = (double *)malloc(validation_distance_bytes);
    centroids_copy = (double *)malloc(centroid_bytes);
    train_u32 = (uint32_t *)malloc(train_u32_bytes);
    validation_u32 = (uint32_t *)malloc(validation_u32_bytes);
    cluster_sizes_u64 = (uint64_t *)malloc(cluster_u64_bytes);
    initial_indices_u64 = (uint64_t *)malloc(cluster_u64_bytes);

    if (train == NULL || validation == NULL ||
        validation_assignments == NULL || train_distances == NULL ||
        validation_distances == NULL || centroids_copy == NULL ||
        train_u32 == NULL || validation_u32 == NULL ||
        cluster_sizes_u64 == NULL || initial_indices_u64 == NULL) {

        fprintf(stderr, "Errore: memoria insufficiente.\n");
        goto cleanup;
    }

    if (gs_read_float_matrix(argv[1], train, train_values, train_bytes) != 0 ||
        gs_read_float_matrix(
            argv[2],
            validation,
            validation_values,
            validation_bytes
        ) != 0) {

        fprintf(stderr, "Errore: lettura matrici standardizzate fallita.\n");
        goto cleanup;
    }
    printf("[PASS] platform_and_inputs\n");

    printf(
        "KMEANS_FIT_STARTED k=%lu rows=6866 cols=900\n",
        (unsigned long)k
    );
    fflush(stdout);

    start_clock = clock();
    if (gs_baseline_kmeans_fit(
            train,
            train_rows,
            feature_count,
            k,
            seed,
            max_iterations,
            &result
        ) != GS_BASELINE_KMEANS_SUCCESS) {

        fprintf(stderr, "Errore: fit k-means fallito.\n");
        goto cleanup;
    }
    end_clock = clock();

    if (start_clock == (clock_t)-1 || end_clock == (clock_t)-1 ||
        end_clock < start_clock) {

        cpu_seconds = -1.0;
    } else {
        cpu_seconds =
            (double)(end_clock - start_clock) / (double)CLOCKS_PER_SEC;
    }
    printf("KMEANS_FIT_COMPLETED\n");

    if (gs_baseline_kmeans_assign(
            train,
            train_rows,
            feature_count,
            result.centroids,
            k,
            result.assignments,
            train_distances
        ) != GS_BASELINE_KMEANS_SUCCESS) {

        fprintf(stderr, "Errore: ricalcolo train fallito.\n");
        goto cleanup;
    }
    printf("[PASS] train_fit_only\n");

    memcpy(centroids_copy, result.centroids, centroid_bytes);
    if (gs_baseline_kmeans_assign(
            validation,
            validation_rows,
            feature_count,
            result.centroids,
            k,
            validation_assignments,
            validation_distances
        ) != GS_BASELINE_KMEANS_SUCCESS ||
        memcmp(centroids_copy, result.centroids, centroid_bytes) != 0) {

        fprintf(stderr, "Errore: assegnazione validation fallita.\n");
        goto cleanup;
    }
    printf("[PASS] validation_assignment\n");

    if (gs_validate_and_convert(
            &result,
            validation_assignments,
            validation_rows,
            train_distances,
            validation_distances,
            train_u32,
            validation_u32,
            cluster_sizes_u64,
            initial_indices_u64,
            &train_statistics,
            &validation_statistics,
            &empty_clusters,
            &minimum_cluster_size,
            &maximum_cluster_size
        ) != 0) {

        fprintf(stderr, "Errore: contratto numerico non valido.\n");
        goto cleanup;
    }
    printf("[PASS] numeric_validation\n");

    if (gs_create_partial_directory(argv[9], final_path) != 0) {
        goto cleanup;
    }

    if (gs_write_binary(
            argv[9], "centroids_f64.bin",
            result.centroids, sizeof(double), centroid_values
        ) != 0 ||
        gs_write_binary(
            argv[9], "train_assignments_u32.bin",
            train_u32, sizeof(uint32_t), train_rows
        ) != 0 ||
        gs_write_binary(
            argv[9], "val_assignments_u32.bin",
            validation_u32, sizeof(uint32_t), validation_rows
        ) != 0 ||
        gs_write_binary(
            argv[9], "train_squared_distances_f64.bin",
            train_distances, sizeof(double), train_rows
        ) != 0 ||
        gs_write_binary(
            argv[9], "val_squared_distances_f64.bin",
            validation_distances, sizeof(double), validation_rows
        ) != 0 ||
        gs_write_binary(
            argv[9], "cluster_sizes_train_u64.bin",
            cluster_sizes_u64, sizeof(uint64_t), k
        ) != 0 ||
        gs_write_binary(
            argv[9], "initial_centroid_indices_u64.bin",
            initial_indices_u64, sizeof(uint64_t), k
        ) != 0 ||
        gs_write_assignment_csv(
            argv[9], "train_assignments.csv",
            train_u32, train_distances, train_rows
        ) != 0 ||
        gs_write_assignment_csv(
            argv[9], "val_assignments.csv",
            validation_u32, validation_distances, validation_rows
        ) != 0 ||
        gs_write_cluster_sizes_csv(argv[9], cluster_sizes_u64, k) != 0 ||
        gs_write_initial_centroids_csv(argv[9], initial_indices_u64, k) != 0 ||
        gs_write_summary(
            argv[9], k, seed, max_iterations, &result,
            &train_statistics, &validation_statistics,
            empty_clusters, minimum_cluster_size, maximum_cluster_size,
            cpu_seconds, argv[10]
        ) != 0) {

        fprintf(stderr, "Errore: scrittura artefatti k-means fallita.\n");
        goto cleanup;
    }
    printf("[PASS] artifacts_written\n");

    printf("K=%lu\n", (unsigned long)k);
    printf("INITIAL_INERTIA=%.17g\n", result.initial_inertia);
    printf("FINAL_TRAIN_INERTIA=%.17g\n", result.inertia);
    printf("VALIDATION_INERTIA=%.17g\n", validation_statistics.sum);
    printf("ITERATIONS=%lu\n", (unsigned long)result.iterations);
    printf("CONVERGED=%d\n", result.converged);
    printf(
        "EMPTY_CLUSTER_RESEEDS=%lu\n",
        (unsigned long)result.empty_cluster_reseeds
    );
    printf("EMPTY_FINAL_CLUSTERS=%lu\n", (unsigned long)empty_clusters);
    printf(
        "CLUSTER_SIZE_MIN=%lu\n",
        (unsigned long)minimum_cluster_size
    );
    printf(
        "CLUSTER_SIZE_MAX=%lu\n",
        (unsigned long)maximum_cluster_size
    );
    printf("FIT_CPU_SECONDS_APPROX=%.17g\n", cpu_seconds);
    printf("KMEANS RUN COMPLETED\n");
    rc = 0;

cleanup:
    gs_baseline_kmeans_result_free(&result);
    free(initial_indices_u64);
    free(cluster_sizes_u64);
    free(validation_u32);
    free(train_u32);
    free(centroids_copy);
    free(validation_distances);
    free(train_distances);
    free(validation_assignments);
    free(validation);
    free(train);
    return rc;
}
