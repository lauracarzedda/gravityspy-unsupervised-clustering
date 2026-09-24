#include "baseline_kmeans.h"

#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


enum {
    GS_TEST_PATH_LENGTH = 4096,
    GS_TEST_LINE_LENGTH = 8192,
    GS_TEST_TRAIN_ROWS = 6866,
    GS_TEST_VALIDATION_ROWS = 1717,
    GS_TEST_SEED = 123,
    GS_TEST_MAX_ITERATIONS = 300
};


typedef struct {
    const char *name;
    const char *train_filename;
    const char *validation_filename;
    size_t components;
} GSPCAProfile;


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
           sizeof(uint32_t) == 4 && sizeof(uint64_t) == 8 &&
           bytes[0] == 1u;
}


static int gs_commit_is_valid(const char *commit)
{
    size_t index;

    if (commit == NULL || strlen(commit) != 40) {
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


static int gs_parse_size_value(const char *text, size_t *value)
{
    char *end;
    unsigned long long parsed;

    if (text == NULL || value == NULL || text[0] == '\0' ||
        text[0] == '-') {

        return 1;
    }
    errno = 0;
    end = NULL;
    parsed = strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' ||
        parsed > (unsigned long long)SIZE_MAX) {

        return 1;
    }
    *value = (size_t)parsed;
    return 0;
}


static int gs_parse_u64_value(const char *text, uint64_t *value)
{
    char *end;
    unsigned long long parsed;

    if (text == NULL || value == NULL || text[0] == '\0' ||
        text[0] == '-') {

        return 1;
    }
    errno = 0;
    end = NULL;
    parsed = strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        return 1;
    }
    *value = (uint64_t)parsed;
    return 0;
}


static int gs_parse_double_value(const char *text, double *value)
{
    char *end;
    double parsed;

    if (text == NULL || value == NULL || text[0] == '\0') {
        return 1;
    }
    errno = 0;
    end = NULL;
    parsed = strtod(text, &end);
    if (errno == ERANGE || end == text || *end != '\0' ||
        !isfinite(parsed)) {

        return 1;
    }
    *value = parsed;
    return 0;
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


static int gs_split_csv_line(
    char *line,
    char **fields,
    size_t expected_fields
)
{
    size_t count = 1;
    char *cursor;

    if (line == NULL || fields == NULL || expected_fields == 0) {
        return 1;
    }
    fields[0] = line;
    for (cursor = line; *cursor != '\0'; cursor++) {
        if (*cursor == ',') {
            if (count >= expected_fields) {
                return 1;
            }
            *cursor = '\0';
            fields[count++] = cursor + 1;
        }
    }
    return count == expected_fields ? 0 : 1;
}


static int gs_read_thresholds(
    const char *directory,
    size_t *d90,
    size_t *d95,
    size_t *d99
)
{
    static const double expected_thresholds[3] = {0.90, 0.95, 0.99};
    char path[GS_TEST_PATH_LENGTH];
    char line[GS_TEST_LINE_LENGTH];
    size_t counts[3];
    double previous_cumulative = 0.0;
    FILE *file;
    size_t row;
    int failed = 0;

    if (d90 == NULL || d95 == NULL || d99 == NULL ||
        gs_path_join(directory, "thresholds.csv", path, sizeof(path)) != 0) {

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
        char *fields[3];
        double threshold;
        double cumulative;

        if (fgets(line, sizeof(line), file) == NULL) {
            failed = 1;
            break;
        }
        gs_strip_line_ending(line);
        if (gs_split_csv_line(line, fields, 3) != 0 ||
            gs_parse_double_value(fields[0], &threshold) != 0 ||
            gs_parse_size_value(fields[1], &counts[row]) != 0 ||
            gs_parse_double_value(fields[2], &cumulative) != 0 ||
            threshold != expected_thresholds[row] ||
            counts[row] == 0 ||
            cumulative < threshold || cumulative > 1.0 ||
            cumulative < previous_cumulative) {

            failed = 1;
            break;
        }
        previous_cumulative = cumulative;
    }
    if (!failed && fgets(line, sizeof(line), file) != NULL) {
        failed = 1;
    }
    if (ferror(file) || fclose(file) != 0) {
        failed = 1;
    }
    if (failed || counts[0] != 21 || counts[1] != 62 || counts[2] != 263) {
        return 1;
    }
    *d90 = counts[0];
    *d95 = counts[1];
    *d99 = counts[2];
    return 0;
}


static int gs_select_profile(
    const char *name,
    size_t d90,
    size_t d95,
    size_t d99,
    GSPCAProfile *profile
)
{
    if (name == NULL || profile == NULL ||
        d90 != 21 || d95 != 62 || d99 != 263) {

        return 1;
    }
    if (strcmp(name, "pca90") == 0) {
        profile->name = "pca90";
        profile->train_filename = "train_scores_pca90_f32.bin";
        profile->validation_filename = "val_scores_pca90_f32.bin";
        profile->components = d90;
    } else if (strcmp(name, "pca95") == 0) {
        profile->name = "pca95";
        profile->train_filename = "train_scores_pca95_f32.bin";
        profile->validation_filename = "val_scores_pca95_f32.bin";
        profile->components = d95;
    } else if (strcmp(name, "pca99") == 0) {
        profile->name = "pca99";
        profile->train_filename = "train_scores_pca99_f32.bin";
        profile->validation_filename = "val_scores_pca99_f32.bin";
        profile->components = d99;
    } else {
        return 1;
    }
    return 0;
}


static int gs_expected_file(
    const char *directory,
    const char *filename,
    size_t expected_size
)
{
    char path[GS_TEST_PATH_LENGTH];
    size_t actual_size;

    if (gs_path_join(directory, filename, path, sizeof(path)) != 0 ||
        gs_file_size(path, &actual_size) != 0) {

        return 1;
    }
    return expected_size == 0 ? actual_size == 0 :
        actual_size != expected_size;
}


static int gs_validate_expected_files(
    const char *directory,
    size_t components,
    size_t k
)
{
    size_t centroid_values;
    size_t centroid_bytes;
    size_t train_assignment_bytes;
    size_t validation_assignment_bytes;
    size_t train_distance_bytes;
    size_t validation_distance_bytes;
    size_t k_u64_bytes;

    if (gs_checked_product(k, components, &centroid_values) != 0 ||
        gs_checked_product(
            centroid_values, sizeof(double), &centroid_bytes
        ) != 0 ||
        gs_checked_product(
            GS_TEST_TRAIN_ROWS,
            sizeof(uint32_t),
            &train_assignment_bytes
        ) != 0 ||
        gs_checked_product(
            GS_TEST_VALIDATION_ROWS,
            sizeof(uint32_t),
            &validation_assignment_bytes
        ) != 0 ||
        gs_checked_product(
            GS_TEST_TRAIN_ROWS,
            sizeof(double),
            &train_distance_bytes
        ) != 0 ||
        gs_checked_product(
            GS_TEST_VALIDATION_ROWS,
            sizeof(double),
            &validation_distance_bytes
        ) != 0 ||
        gs_checked_product(k, sizeof(uint64_t), &k_u64_bytes) != 0) {

        return 1;
    }

    return
        gs_expected_file(directory, "centroids_f64.bin", centroid_bytes) ||
        gs_expected_file(
            directory, "train_assignments_u32.bin", train_assignment_bytes
        ) ||
        gs_expected_file(
            directory,
            "val_assignments_u32.bin",
            validation_assignment_bytes
        ) ||
        gs_expected_file(
            directory,
            "train_squared_distances_f64.bin",
            train_distance_bytes
        ) ||
        gs_expected_file(
            directory,
            "val_squared_distances_f64.bin",
            validation_distance_bytes
        ) ||
        gs_expected_file(
            directory, "cluster_sizes_train_u64.bin", k_u64_bytes
        ) ||
        gs_expected_file(
            directory, "initial_centroid_indices_u64.bin", k_u64_bytes
        ) ||
        gs_expected_file(directory, "train_assignments.csv", 0) ||
        gs_expected_file(directory, "val_assignments.csv", 0) ||
        gs_expected_file(directory, "cluster_sizes.csv", 0) ||
        gs_expected_file(directory, "initial_centroids.csv", 0) ||
        gs_expected_file(directory, "summary.txt", 0);
}


static int gs_read_binary(
    const char *directory,
    const char *filename,
    void *values,
    size_t element_size,
    size_t count
)
{
    char path[GS_TEST_PATH_LENGTH];
    FILE *file;
    int failed = 0;

    if (values == NULL || element_size == 0 ||
        gs_path_join(directory, filename, path, sizeof(path)) != 0) {

        return 1;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        return 1;
    }
    if (fread(values, element_size, count, file) != count ||
        fgetc(file) != EOF || ferror(file)) {

        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    return failed;
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


static int gs_validate_assignment_csv(
    const char *directory,
    const char *filename,
    const uint32_t *assignments,
    const double *distances,
    size_t rows,
    size_t k
)
{
    char path[GS_TEST_PATH_LENGTH];
    char line[GS_TEST_LINE_LENGTH];
    FILE *file;
    size_t row;
    int failed = 0;

    if (gs_path_join(directory, filename, path, sizeof(path)) != 0) {
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
        if (strcmp(line, "row_index,cluster,squared_distance") != 0) {
            failed = 1;
        }
    }
    for (row = 0; !failed && row < rows; row++) {
        char *fields[3];
        size_t parsed_row;
        size_t parsed_cluster;
        double parsed_distance;

        if (fgets(line, sizeof(line), file) == NULL) {
            failed = 1;
            break;
        }
        gs_strip_line_ending(line);
        if (gs_split_csv_line(line, fields, 3) != 0 ||
            gs_parse_size_value(fields[0], &parsed_row) != 0 ||
            gs_parse_size_value(fields[1], &parsed_cluster) != 0 ||
            gs_parse_double_value(fields[2], &parsed_distance) != 0 ||
            parsed_row != row ||
            parsed_cluster != (size_t)assignments[row] ||
            parsed_cluster >= k ||
            memcmp(
                &parsed_distance, &distances[row], sizeof(double)
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


static int gs_validate_cluster_sizes_csv(
    const char *directory,
    const uint64_t *sizes,
    size_t k
)
{
    char path[GS_TEST_PATH_LENGTH];
    char line[GS_TEST_LINE_LENGTH];
    FILE *file;
    size_t cluster;
    int failed = 0;

    if (gs_path_join(
            directory, "cluster_sizes.csv", path, sizeof(path)
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
        if (strcmp(line, "cluster,train_size") != 0) {
            failed = 1;
        }
    }
    for (cluster = 0; !failed && cluster < k; cluster++) {
        char *fields[2];
        size_t parsed_cluster;
        uint64_t parsed_size;

        if (fgets(line, sizeof(line), file) == NULL) {
            failed = 1;
            break;
        }
        gs_strip_line_ending(line);
        if (gs_split_csv_line(line, fields, 2) != 0 ||
            gs_parse_size_value(fields[0], &parsed_cluster) != 0 ||
            gs_parse_u64_value(fields[1], &parsed_size) != 0 ||
            parsed_cluster != cluster || parsed_size != sizes[cluster]) {

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


static int gs_validate_initial_centroids_csv(
    const char *directory,
    const uint64_t *indices,
    size_t k
)
{
    char path[GS_TEST_PATH_LENGTH];
    char line[GS_TEST_LINE_LENGTH];
    FILE *file;
    size_t order;
    int failed = 0;

    if (gs_path_join(
            directory, "initial_centroids.csv", path, sizeof(path)
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
        if (strcmp(line, "selection_order,train_row_index") != 0) {
            failed = 1;
        }
    }
    for (order = 0; !failed && order < k; order++) {
        char *fields[2];
        size_t parsed_order;
        uint64_t parsed_index;

        if (fgets(line, sizeof(line), file) == NULL) {
            failed = 1;
            break;
        }
        gs_strip_line_ending(line);
        if (gs_split_csv_line(line, fields, 2) != 0 ||
            gs_parse_size_value(fields[0], &parsed_order) != 0 ||
            gs_parse_u64_value(fields[1], &parsed_index) != 0 ||
            parsed_order != order || parsed_index != indices[order]) {

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


static int gs_read_text(
    const char *directory,
    const char *filename,
    char **text
)
{
    char path[GS_TEST_PATH_LENGTH];
    size_t size;
    char *buffer;
    FILE *file;
    int failed = 0;

    if (text == NULL ||
        gs_path_join(directory, filename, path, sizeof(path)) != 0 ||
        gs_file_size(path, &size) != 0 ||
        size == 0 || size == SIZE_MAX) {

        return 1;
    }
    buffer = (char *)malloc(size + 1);
    if (buffer == NULL) {
        return 1;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        free(buffer);
        return 1;
    }
    if (fread(buffer, 1, size, file) != size ||
        fgetc(file) != EOF || ferror(file) ||
        memchr(buffer, '\0', size) != NULL) {

        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    if (failed) {
        free(buffer);
        return 1;
    }
    buffer[size] = '\0';
    *text = buffer;
    return 0;
}


static int gs_text_has_line(const char *text, const char *line)
{
    const char *cursor = text;
    size_t length = strlen(line);

    while ((cursor = strstr(cursor, line)) != NULL) {
        const char *after = cursor + length;

        if ((cursor == text || cursor[-1] == '\n') &&
            (*after == '\0' || *after == '\r' || *after == '\n')) {

            return 1;
        }
        cursor++;
    }
    return 0;
}


static int gs_text_has_key(const char *text, const char *key)
{
    const char *cursor = text;
    size_t length = strlen(key);

    while ((cursor = strstr(cursor, key)) != NULL) {
        const char *after = cursor + length;

        if ((cursor == text || cursor[-1] == '\n') &&
            *after != '\0' && *after != '\r' && *after != '\n') {

            return 1;
        }
        cursor++;
    }
    return 0;
}


static int gs_validate_summary(
    const char *directory,
    const GSPCAProfile *profile,
    size_t k,
    const char *commit
)
{
    static const char *fixed_lines[] = {
        "format_version=1",
        "profile=mini75_pool5_pca_kmeans",
        "source_profile=mini75_pool5_pca",
        "algorithm=kmeans_plus_plus_lloyd",
        "validation_policy=assign_to_train_centroids",
        "train_rows=6866",
        "validation_rows=1717",
        "pca_whitening=false",
        "post_pca_standardization=false",
        "max_iterations=300",
        "centroid_dtype=float64",
        "assignment_dtype=uint32",
        "distance_dtype=float64",
        "cluster_size_dtype=uint64",
        "fit_data=train_only",
        "fit_input=train_only",
        "validation_assigned_after_fit=true",
        "augmentation=false",
        "labels_used=false"
    };
    static const char *numeric_keys[] = {
        "iterations=",
        "converged=",
        "empty_cluster_reseeds=",
        "initial_inertia=",
        "final_train_inertia=",
        "validation_inertia=",
        "empty_final_clusters=",
        "minimum_cluster_size=",
        "maximum_cluster_size=",
        "fit_cpu_seconds_approx="
    };
    char input_line[128];
    char feature_line[64];
    char profile_line[64];
    char components_line[64];
    char k_line[64];
    char seed_line[64];
    char commit_line[128];
    char *text = NULL;
    size_t index;
    int failed = 0;

    if (profile == NULL ||
        snprintf(
            input_line,
            sizeof(input_line),
            "input=train_scores_%s_f32",
            profile->name
        ) < 0 ||
        snprintf(
            feature_line,
            sizeof(feature_line),
            "feature_count=%zu",
            profile->components
        ) < 0 ||
        snprintf(
            profile_line,
            sizeof(profile_line),
            "pca_profile=%s",
            profile->name
        ) < 0 ||
        snprintf(
            components_line,
            sizeof(components_line),
            "pca_components=%zu",
            profile->components
        ) < 0 ||
        snprintf(k_line, sizeof(k_line), "k=%zu", k) < 0 ||
        snprintf(
            seed_line,
            sizeof(seed_line),
            "seed=%u",
            (unsigned int)GS_TEST_SEED
        ) < 0 ||
        snprintf(
            commit_line, sizeof(commit_line), "code_commit=%s", commit
        ) < 0 ||
        gs_read_text(directory, "summary.txt", &text) != 0) {

        free(text);
        return 1;
    }

    for (index = 0;
         index < sizeof(fixed_lines) / sizeof(fixed_lines[0]);
         index++) {

        if (!gs_text_has_line(text, fixed_lines[index])) {
            failed = 1;
        }
    }
    for (index = 0;
         index < sizeof(numeric_keys) / sizeof(numeric_keys[0]);
         index++) {

        if (!gs_text_has_key(text, numeric_keys[index])) {
            failed = 1;
        }
    }
    if (!gs_text_has_line(text, input_line) ||
        !gs_text_has_line(text, feature_line) ||
        !gs_text_has_line(text, profile_line) ||
        !gs_text_has_line(text, components_line) ||
        !gs_text_has_line(text, k_line) ||
        !gs_text_has_line(text, seed_line) ||
        !gs_text_has_line(text, commit_line)) {

        failed = 1;
    }
    free(text);
    return failed;
}


static int gs_distance_equal(double left, double right)
{
    double tolerance = 1.0e-10 * (1.0 + fabs(left));

    return isfinite(left) && isfinite(right) &&
           left >= 0.0 && right >= 0.0 &&
           fabs(left - right) <= tolerance;
}


static void gs_print_usage(const char *program)
{
    printf(
        "Usage: %s <pca_artifact_directory> <pca_profile> <k> "
        "<kmeans_artifact_directory> <code_commit>\n"
        "pca_profile: pca90 | pca95 | pca99\n",
        program
    );
}


int main(int argc, char **argv)
{
    GSPCAProfile profile;
    size_t d90;
    size_t d95;
    size_t d99;
    size_t k;
    char train_path[GS_TEST_PATH_LENGTH];
    char validation_path[GS_TEST_PATH_LENGTH];
    size_t train_values;
    size_t validation_values;
    size_t centroid_values;
    size_t train_bytes;
    size_t validation_bytes;
    size_t centroid_bytes;
    size_t train_assignment_bytes;
    size_t validation_assignment_bytes;
    size_t train_size_t_bytes;
    size_t validation_size_t_bytes;
    size_t train_distance_bytes;
    size_t validation_distance_bytes;
    size_t k_u64_bytes;

    float *train = NULL;
    float *validation = NULL;
    float *train_copy = NULL;
    float *validation_copy = NULL;
    double *centroids = NULL;
    double *centroid_copy = NULL;
    uint32_t *train_assignments = NULL;
    uint32_t *validation_assignments = NULL;
    double *train_distances = NULL;
    double *validation_distances = NULL;
    uint64_t *cluster_sizes = NULL;
    uint64_t *initial_indices = NULL;
    size_t *recomputed_train = NULL;
    size_t *recomputed_validation = NULL;
    double *recomputed_train_distances = NULL;
    double *recomputed_validation_distances = NULL;
    uint64_t *recomputed_sizes = NULL;
    size_t index;
    int rc = 1;

    memset(&profile, 0, sizeof(profile));

    if (argc == 2 && strcmp(argv[1], "--help") == 0) {
        gs_print_usage(argv[0]);
        return 0;
    }
    if (argc != 6 || !gs_platform_is_valid() ||
        gs_parse_size_value(argv[3], &k) != 0 ||
        (k != 22 && k != 36) ||
        !gs_commit_is_valid(argv[5]) ||
        gs_read_thresholds(argv[1], &d90, &d95, &d99) != 0 ||
        gs_select_profile(argv[2], d90, d95, d99, &profile) != 0 ||
        gs_path_join(
            argv[1], profile.train_filename,
            train_path, sizeof(train_path)
        ) != 0 ||
        gs_path_join(
            argv[1], profile.validation_filename,
            validation_path, sizeof(validation_path)
        ) != 0) {

        fprintf(stderr, "Errore: argomenti o profilo non validi.\n");
        return 1;
    }
    if (gs_checked_product(
            GS_TEST_TRAIN_ROWS, profile.components, &train_values
        ) != 0 ||
        gs_checked_product(
            GS_TEST_VALIDATION_ROWS,
            profile.components,
            &validation_values
        ) != 0 ||
        gs_checked_product(k, profile.components, &centroid_values) != 0 ||
        gs_checked_product(train_values, sizeof(float), &train_bytes) != 0 ||
        gs_checked_product(
            validation_values, sizeof(float), &validation_bytes
        ) != 0 ||
        gs_checked_product(
            centroid_values, sizeof(double), &centroid_bytes
        ) != 0 ||
        gs_checked_product(
            GS_TEST_TRAIN_ROWS,
            sizeof(uint32_t),
            &train_assignment_bytes
        ) != 0 ||
        gs_checked_product(
            GS_TEST_VALIDATION_ROWS,
            sizeof(uint32_t),
            &validation_assignment_bytes
        ) != 0 ||
        gs_checked_product(
            GS_TEST_TRAIN_ROWS, sizeof(size_t), &train_size_t_bytes
        ) != 0 ||
        gs_checked_product(
            GS_TEST_VALIDATION_ROWS,
            sizeof(size_t),
            &validation_size_t_bytes
        ) != 0 ||
        gs_checked_product(
            GS_TEST_TRAIN_ROWS, sizeof(double), &train_distance_bytes
        ) != 0 ||
        gs_checked_product(
            GS_TEST_VALIDATION_ROWS,
            sizeof(double),
            &validation_distance_bytes
        ) != 0 ||
        gs_checked_product(k, sizeof(uint64_t), &k_u64_bytes) != 0) {

        fprintf(stderr, "Errore: overflow dimensioni.\n");
        return 1;
    }
    {
        size_t actual_size;

        if (gs_file_size(train_path, &actual_size) != 0 ||
            actual_size != train_bytes ||
            gs_file_size(validation_path, &actual_size) != 0 ||
            actual_size != validation_bytes) {

            fprintf(stderr, "Errore: score PCA non validi.\n");
            return 1;
        }
    }
    printf("[PASS] pca_profile_contract\n");

    if (gs_validate_expected_files(argv[4], profile.components, k) != 0) {
        fprintf(stderr, "Errore: artefatti o dimensioni non validi.\n");
        return 1;
    }
    printf("[PASS] expected_files_and_sizes\n");

    train = (float *)malloc(train_bytes);
    validation = (float *)malloc(validation_bytes);
    train_copy = (float *)malloc(train_bytes);
    validation_copy = (float *)malloc(validation_bytes);
    centroids = (double *)malloc(centroid_bytes);
    centroid_copy = (double *)malloc(centroid_bytes);
    train_assignments = (uint32_t *)malloc(train_assignment_bytes);
    validation_assignments = (uint32_t *)malloc(validation_assignment_bytes);
    train_distances = (double *)malloc(train_distance_bytes);
    validation_distances = (double *)malloc(validation_distance_bytes);
    cluster_sizes = (uint64_t *)malloc(k_u64_bytes);
    initial_indices = (uint64_t *)malloc(k_u64_bytes);
    recomputed_train = (size_t *)malloc(train_size_t_bytes);
    recomputed_validation = (size_t *)malloc(validation_size_t_bytes);
    recomputed_train_distances = (double *)malloc(train_distance_bytes);
    recomputed_validation_distances =
        (double *)malloc(validation_distance_bytes);
    recomputed_sizes = (uint64_t *)calloc(k, sizeof(uint64_t));

    if (train == NULL || validation == NULL ||
        train_copy == NULL || validation_copy == NULL ||
        centroids == NULL || centroid_copy == NULL ||
        train_assignments == NULL || validation_assignments == NULL ||
        train_distances == NULL || validation_distances == NULL ||
        cluster_sizes == NULL || initial_indices == NULL ||
        recomputed_train == NULL || recomputed_validation == NULL ||
        recomputed_train_distances == NULL ||
        recomputed_validation_distances == NULL ||
        recomputed_sizes == NULL ||
        gs_read_float_matrix(
            train_path, train, train_values, train_bytes
        ) != 0 ||
        gs_read_float_matrix(
            validation_path,
            validation,
            validation_values,
            validation_bytes
        ) != 0 ||
        gs_read_binary(
            argv[4], "centroids_f64.bin",
            centroids, sizeof(double), centroid_values
        ) != 0 ||
        gs_read_binary(
            argv[4], "train_assignments_u32.bin",
            train_assignments, sizeof(uint32_t), GS_TEST_TRAIN_ROWS
        ) != 0 ||
        gs_read_binary(
            argv[4], "val_assignments_u32.bin",
            validation_assignments,
            sizeof(uint32_t),
            GS_TEST_VALIDATION_ROWS
        ) != 0 ||
        gs_read_binary(
            argv[4], "train_squared_distances_f64.bin",
            train_distances, sizeof(double), GS_TEST_TRAIN_ROWS
        ) != 0 ||
        gs_read_binary(
            argv[4], "val_squared_distances_f64.bin",
            validation_distances,
            sizeof(double),
            GS_TEST_VALIDATION_ROWS
        ) != 0 ||
        gs_read_binary(
            argv[4], "cluster_sizes_train_u64.bin",
            cluster_sizes, sizeof(uint64_t), k
        ) != 0 ||
        gs_read_binary(
            argv[4], "initial_centroid_indices_u64.bin",
            initial_indices, sizeof(uint64_t), k
        ) != 0) {

        fprintf(stderr, "Errore: lettura artefatti fallita.\n");
        goto cleanup;
    }
    memcpy(train_copy, train, train_bytes);
    memcpy(validation_copy, validation, validation_bytes);

    for (index = 0; index < centroid_values; index++) {
        if (!isfinite(centroids[index])) {
            fprintf(stderr, "Errore: centroide non finito.\n");
            goto cleanup;
        }
    }
    printf("[PASS] centroid_contract\n");

    for (index = 0; index < GS_TEST_TRAIN_ROWS; index++) {
        if (train_assignments[index] >= k) {
            fprintf(stderr, "Errore: assegnazione train non valida.\n");
            goto cleanup;
        }
    }
    for (index = 0; index < GS_TEST_VALIDATION_ROWS; index++) {
        if (validation_assignments[index] >= k) {
            fprintf(stderr, "Errore: assegnazione validation non valida.\n");
            goto cleanup;
        }
    }
    if (gs_validate_assignment_csv(
            argv[4],
            "train_assignments.csv",
            train_assignments,
            train_distances,
            GS_TEST_TRAIN_ROWS,
            k
        ) != 0 ||
        gs_validate_assignment_csv(
            argv[4],
            "val_assignments.csv",
            validation_assignments,
            validation_distances,
            GS_TEST_VALIDATION_ROWS,
            k
        ) != 0) {

        fprintf(stderr, "Errore: CSV assegnazioni non validi.\n");
        goto cleanup;
    }
    printf("[PASS] assignment_contract\n");

    for (index = 0; index < GS_TEST_TRAIN_ROWS; index++) {
        if (!isfinite(train_distances[index]) ||
            train_distances[index] < 0.0) {

            fprintf(stderr, "Errore: distanza train non valida.\n");
            goto cleanup;
        }
    }
    for (index = 0; index < GS_TEST_VALIDATION_ROWS; index++) {
        if (!isfinite(validation_distances[index]) ||
            validation_distances[index] < 0.0) {

            fprintf(stderr, "Errore: distanza validation non valida.\n");
            goto cleanup;
        }
    }
    printf("[PASS] distance_contract\n");

    if (gs_baseline_kmeans_assign(
            train,
            GS_TEST_TRAIN_ROWS,
            profile.components,
            centroids,
            k,
            recomputed_train,
            recomputed_train_distances
        ) != GS_BASELINE_KMEANS_SUCCESS) {

        fprintf(stderr, "Errore: ricalcolo train fallito.\n");
        goto cleanup;
    }
    for (index = 0; index < GS_TEST_TRAIN_ROWS; index++) {
        if (recomputed_train[index] != (size_t)train_assignments[index] ||
            !gs_distance_equal(
                recomputed_train_distances[index],
                train_distances[index]
            )) {

            fprintf(stderr, "Errore: ricalcolo train non coincide.\n");
            goto cleanup;
        }
    }
    printf("[PASS] recompute_train_assignments\n");

    memcpy(centroid_copy, centroids, centroid_bytes);
    if (gs_baseline_kmeans_assign(
            validation,
            GS_TEST_VALIDATION_ROWS,
            profile.components,
            centroids,
            k,
            recomputed_validation,
            recomputed_validation_distances
        ) != GS_BASELINE_KMEANS_SUCCESS ||
        memcmp(centroid_copy, centroids, centroid_bytes) != 0) {

        fprintf(stderr, "Errore: ricalcolo validation fallito.\n");
        goto cleanup;
    }
    for (index = 0; index < GS_TEST_VALIDATION_ROWS; index++) {
        if (recomputed_validation[index] !=
                (size_t)validation_assignments[index] ||
            !gs_distance_equal(
                recomputed_validation_distances[index],
                validation_distances[index]
            )) {

            fprintf(stderr, "Errore: ricalcolo validation non coincide.\n");
            goto cleanup;
        }
    }
    printf("[PASS] recompute_validation_assignments\n");

    for (index = 0; index < GS_TEST_TRAIN_ROWS; index++) {
        recomputed_sizes[train_assignments[index]]++;
    }
    for (index = 0; index < k; index++) {
        if (recomputed_sizes[index] != cluster_sizes[index]) {
            fprintf(stderr, "Errore: cluster size non coincide.\n");
            goto cleanup;
        }
    }
    if (gs_validate_cluster_sizes_csv(
            argv[4], cluster_sizes, k
        ) != 0) {

        fprintf(stderr, "Errore: CSV cluster size non valido.\n");
        goto cleanup;
    }
    printf("[PASS] cluster_sizes_contract\n");

    for (index = 0; index < k; index++) {
        size_t left;

        if (initial_indices[index] >= GS_TEST_TRAIN_ROWS) {
            fprintf(stderr, "Errore: indice iniziale fuori range.\n");
            goto cleanup;
        }
        for (left = 0; left < index; left++) {
            if (initial_indices[left] == initial_indices[index]) {
                fprintf(stderr, "Errore: indice iniziale duplicato.\n");
                goto cleanup;
            }
        }
    }
    if (gs_validate_initial_centroids_csv(
            argv[4], initial_indices, k
        ) != 0) {

        fprintf(stderr, "Errore: CSV centroidi iniziali non valido.\n");
        goto cleanup;
    }
    printf("[PASS] initial_centroids_contract\n");

    if (gs_validate_summary(
            argv[4], &profile, k, argv[5]
        ) != 0) {

        fprintf(stderr, "Errore: metadata non validi.\n");
        goto cleanup;
    }
    printf("[PASS] metadata_contract\n");

    if (memcmp(train, train_copy, train_bytes) != 0 ||
        memcmp(validation, validation_copy, validation_bytes) != 0) {

        fprintf(stderr, "Errore: score PCA modificati o riscalati.\n");
        goto cleanup;
    }
    printf("[PASS] no_post_pca_scaling\n");
    printf("PCA_PROFILE=%s\n", profile.name);
    printf("PCA_COMPONENTS=%zu\n", profile.components);
    printf("K=%zu\n", k);
    printf("ALL PCA KMEANS ARTIFACT TESTS PASSED\n");
    rc = 0;

cleanup:
    free(recomputed_sizes);
    free(recomputed_validation_distances);
    free(recomputed_train_distances);
    free(recomputed_validation);
    free(recomputed_train);
    free(initial_indices);
    free(cluster_sizes);
    free(validation_distances);
    free(train_distances);
    free(validation_assignments);
    free(train_assignments);
    free(centroid_copy);
    free(centroids);
    free(validation_copy);
    free(train_copy);
    free(validation);
    free(train);
    return rc;
}
