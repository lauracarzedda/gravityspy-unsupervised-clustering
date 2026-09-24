#include "baseline_pca.h"

#include <dirent.h>
#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif


enum {
    GS224_PCA_TRAIN_ROWS = 800,
    GS224_PCA_VALIDATION_ROWS = 200,
    GS224_PCA_FEATURES = 1024,
    GS224_PCA_PATH_CAPACITY = 4096,
    GS224_PCA_LINE_CAPACITY = 16384,
    GS224_PCA_SOURCE_FILE_COUNT = 10,
    GS224_PCA_SOURCE_INPUT_COUNT = 4,
    GS224_PCA_SOURCE_ARTIFACT_COUNT = 9,
    GS224_PCA_INPUT_COUNT = 7,
    GS224_PCA_ARTIFACT_COUNT = 11,
    GS224_PCA_OUTPUT_FILE_COUNT = 12
};

#define GS224_PCA_BASE_CODE_COMMIT \
    "d87ca996e7922d6873a54d952aec27cdb99a5998"
#define GS224_PCA_FAILURE_ENV "GS_MINI224_SUBSET1000_PCA_TEST_FAIL_AFTER_PARTIAL"
#define GS224_PCA_FAILURE_BASENAME "mini224_subset1000_pca_failure_test.partial"
#define GS224_PCA_UNCOMMITTED_CODE "UNCOMMITTED_STEP10I_TEST"

enum {
    GS224_PCA_MAX_THEORETICAL_RANK = 799,
    GS224_PCA_NUMERICAL_RANK = 799,
    GS224_PCA_NULLSPACE_DIMENSION = 225,
    GS224_PCA_D90 = 15,
    GS224_PCA_D95 = 38,
    GS224_PCA_D99 = 168
};


typedef struct {
    const char *token;
    const char *profile;
    const char *final_basename;
    const char *partial_basename;
    const char *train_score_name;
    const char *validation_score_name;
    double threshold;
} GS224PCAProfile;


typedef struct {
    uint32_t state[8];
    uint64_t completed_bits;
    unsigned char block[64];
    size_t block_size;
} GS224Sha256;


typedef struct {
    size_t input_dim;
    size_t component_count;
    double total_variance;
    size_t d90;
    size_t d95;
    size_t d99;
} GS224PCAModelMetadata;


static const char *const gs224_source_names[
    GS224_PCA_SOURCE_FILE_COUNT
] = {
    "feature_spec.txt",
    "train_features_raw_f32.bin",
    "val_features_raw_f32.bin",
    "train_features_std_f32.bin",
    "val_features_std_f32.bin",
    "scaler.csv",
    "train_rows.csv",
    "val_rows.csv",
    "summary.txt",
    "sha256.txt"
};


static const char *const gs224_source_input_names[
    GS224_PCA_SOURCE_INPUT_COUNT
] = {
    "manifest_events_272.csv",
    "manifest_tensors_224_central.csv",
    "subset224_train_800_seed123.csv",
    "subset224_val_200_seed123.csv"
};


static const char *const gs224_source_input_relative_paths[
    GS224_PCA_SOURCE_INPUT_COUNT
] = {
    "manifest_events_272.csv",
    "manifest_tensors_224_central.csv",
    "sakai_multires_c/configs/subset224_train_800_seed123.csv",
    "sakai_multires_c/configs/subset224_val_200_seed123.csv"
};


static const char *const gs224_source_artifact_names[
    GS224_PCA_SOURCE_ARTIFACT_COUNT
] = {
    "feature_spec.txt",
    "train_features_raw_f32.bin",
    "val_features_raw_f32.bin",
    "train_features_std_f32.bin",
    "val_features_std_f32.bin",
    "scaler.csv",
    "train_rows.csv",
    "val_rows.csv",
    "summary.txt"
};


static const char *const gs224_pca_fixed_artifact_names[9] = {
    "pca_spec.txt",
    "mean_f64.bin",
    "components_f64.bin",
    "eigenvalues_f64.bin",
    "explained_variance_ratio_f64.bin",
    "cumulative_explained_variance_ratio_f64.bin",
    "thresholds.csv",
    "spectrum.csv",
    "summary.txt"
};


static int gs224_checked_product(
    size_t left,
    size_t right,
    size_t *product
)
{
    if (product == NULL ||
        (left != 0 && right > SIZE_MAX / left)) {

        return 1;
    }
    *product = left * right;
    return 0;
}


static int gs224_checked_sum(
    size_t left,
    size_t right,
    size_t *sum
)
{
    if (sum == NULL || left > SIZE_MAX - right) {
        return 1;
    }
    *sum = left + right;
    return 0;
}


static int gs224_platform_is_valid(void)
{
    const uint16_t marker = UINT16_C(1);
    const unsigned char *bytes = (const unsigned char *)&marker;

    return CHAR_BIT == 8 &&
           sizeof(float) == 4 &&
           sizeof(double) == 8 &&
           FLT_RADIX == 2 &&
           FLT_MANT_DIG == 24 &&
           FLT_MAX_EXP == 128 &&
           DBL_MANT_DIG == 53 &&
           DBL_MAX_EXP == 1024 &&
           bytes[0] == 1;
}


static int gs224_metadata_value_is_valid(const char *value)
{
    const unsigned char *cursor;

    if (value == NULL || value[0] == '\0') {
        return 0;
    }
    cursor = (const unsigned char *)value;
    while (*cursor != '\0') {
        if (*cursor < 32u || *cursor == 127u) {
            return 0;
        }
        cursor++;
    }
    return 1;
}


static int gs224_code_commit_is_valid(const char *value)
{
    size_t index;

    if (!gs224_metadata_value_is_valid(value)) {
        return 0;
    }
    if (strcmp(value, GS224_PCA_UNCOMMITTED_CODE) == 0) {
        return 1;
    }
    if (strlen(value) != 40u) {
        return 0;
    }
    for (index = 0; index < 40u; index++) {
        if (!((value[index] >= '0' && value[index] <= '9') ||
              (value[index] >= 'a' && value[index] <= 'f'))) {

            return 0;
        }
    }
    return 1;
}


static const char *gs224_path_basename(const char *path)
{
    const char *basename = path;
    const char *cursor;

    if (path == NULL) {
        return NULL;
    }
    for (cursor = path; *cursor != '\0'; cursor++) {
        if (*cursor == '/' || *cursor == '\\') {
            basename = cursor + 1;
        }
    }
    return basename;
}


static int gs224_path_join(
    const char *directory,
    const char *name,
    char *path,
    size_t path_capacity
)
{
    size_t directory_length;
    size_t name_length;
    size_t required;
    size_t offset;
    int separator;

    if (directory == NULL || name == NULL || path == NULL ||
        path_capacity == 0) {

        return 1;
    }
    directory_length = strlen(directory);
    name_length = strlen(name);
    separator =
        directory_length != 0 &&
        directory[directory_length - 1] != '/' &&
        directory[directory_length - 1] != '\\';
    if (gs224_checked_sum(directory_length, name_length, &required) != 0 ||
        (separator && gs224_checked_sum(required, 1u, &required) != 0) ||
        required >= path_capacity) {

        return 1;
    }

    memcpy(path, directory, directory_length);
    offset = directory_length;
    if (separator) {
        path[offset++] = '/';
    }
    memcpy(path + offset, name, name_length);
    path[offset + name_length] = '\0';
    return 0;
}


static int gs224_path_state(const char *path, int *is_directory)
{
    struct stat information;

    if (path == NULL) {
        return -1;
    }
    errno = 0;
    if (stat(path, &information) == 0) {
        if (is_directory != NULL) {
            *is_directory = S_ISDIR(information.st_mode) ? 1 : 0;
        }
        return 1;
    }
    return errno == ENOENT ? 0 : -1;
}


static int gs224_regular_file_size(
    const char *path,
    size_t *size
)
{
    struct stat information;

    if (path == NULL || size == NULL ||
        stat(path, &information) != 0 ||
        !S_ISREG(information.st_mode) ||
        information.st_size < 0 ||
        (uintmax_t)information.st_size > (uintmax_t)SIZE_MAX) {

        return 1;
    }
    *size = (size_t)information.st_size;
    return 0;
}


static int gs224_profile_from_token(
    const char *token,
    GS224PCAProfile *profile
)
{
    static const GS224PCAProfile profiles[3] = {
        {
            "pca90",
            "mini224_subset1000_pool14_pca90",
            "mini224_subset1000_pool14_pca90",
            "mini224_subset1000_pool14_pca90.partial",
            "train_scores_pca90_f32.bin",
            "val_scores_pca90_f32.bin",
            0.90
        },
        {
            "pca95",
            "mini224_subset1000_pool14_pca95",
            "mini224_subset1000_pool14_pca95",
            "mini224_subset1000_pool14_pca95.partial",
            "train_scores_pca95_f32.bin",
            "val_scores_pca95_f32.bin",
            0.95
        },
        {
            "pca99",
            "mini224_subset1000_pool14_pca99",
            "mini224_subset1000_pool14_pca99",
            "mini224_subset1000_pool14_pca99.partial",
            "train_scores_pca99_f32.bin",
            "val_scores_pca99_f32.bin",
            0.99
        }
    };
    size_t index;

    if (token == NULL || profile == NULL) {
        return 1;
    }
    for (index = 0; index < 3u; index++) {
        if (strcmp(token, profiles[index].token) == 0) {
            *profile = profiles[index];
            return 0;
        }
    }
    return 1;
}


static int gs224_failure_hook_requested(const char *partial_path)
{
    const char *value = getenv(GS224_PCA_FAILURE_ENV);
    const char *basename = gs224_path_basename(partial_path);

    return value != NULL &&
           strcmp(value, "1") == 0 &&
           basename != NULL &&
           strcmp(basename, GS224_PCA_FAILURE_BASENAME) == 0;
}


static int gs224_validate_output_path(
    const char *partial_path,
    const GS224PCAProfile *profile,
    char *final_path,
    size_t final_capacity,
    int *failure_hook
)
{
    static const char suffix[] = ".partial";
    const char *basename;
    size_t path_length;
    size_t suffix_length = sizeof(suffix) - 1u;
    size_t final_length;
    int hook;

    if (!gs224_metadata_value_is_valid(partial_path) ||
        profile == NULL || final_path == NULL || final_capacity == 0 ||
        failure_hook == NULL) {

        return 1;
    }
    basename = gs224_path_basename(partial_path);
    hook = gs224_failure_hook_requested(partial_path);
    if (basename == NULL ||
        (!hook && strcmp(basename, profile->partial_basename) != 0) ||
        (hook && strcmp(basename, GS224_PCA_FAILURE_BASENAME) != 0)) {

        return 1;
    }
    path_length = strlen(partial_path);
    if (path_length <= suffix_length ||
        strcmp(partial_path + path_length - suffix_length, suffix) != 0) {

        return 1;
    }
    final_length = path_length - suffix_length;
    if (final_length == 0 || final_length >= final_capacity) {
        return 1;
    }
    memcpy(final_path, partial_path, final_length);
    final_path[final_length] = '\0';

    if (!hook &&
        strcmp(gs224_path_basename(final_path), profile->final_basename) != 0) {

        return 1;
    }
    if (gs224_path_state(partial_path, NULL) != 0 ||
        gs224_path_state(final_path, NULL) != 0) {

        return 1;
    }
    *failure_hook = hook;
    return 0;
}


static int gs224_normalized_parent(
    const char *path,
    char parent[GS224_PCA_PATH_CAPACITY]
)
{
    size_t length;
    size_t index;

    if (!gs224_metadata_value_is_valid(path) || parent == NULL) {
        return 1;
    }
    length = strlen(path);
    if (length >= GS224_PCA_PATH_CAPACITY) {
        return 1;
    }
    memcpy(parent, path, length + 1u);
    for (index = 0; index < length; index++) {
        if (parent[index] == '\\') {
            parent[index] = '/';
        }
    }
    while (length > 1u && parent[length - 1u] == '/') {
        parent[--length] = '\0';
    }
    while (length != 0u && parent[length - 1u] != '/') {
        length--;
    }
    if (length == 0u) {
        memcpy(parent, ".", 2u);
        return 0;
    }
    while (length > 1u && parent[length - 1u] == '/') {
        length--;
    }
    parent[length] = '\0';
    return 0;
}


static int gs224_output_parent_matches_source(
    const char *source_directory,
    const char *partial_directory
)
{
    char source_parent[GS224_PCA_PATH_CAPACITY];
    char output_parent[GS224_PCA_PATH_CAPACITY];

    if (gs224_normalized_parent(source_directory, source_parent) != 0 ||
        gs224_normalized_parent(partial_directory, output_parent) != 0) {

        return 0;
    }
#ifdef _WIN32
    {
        char source_full[GS224_PCA_PATH_CAPACITY];
        char output_full[GS224_PCA_PATH_CAPACITY];

        if (_fullpath(
                source_full,
                source_parent,
                sizeof(source_full)
            ) == NULL ||
            _fullpath(
                output_full,
                output_parent,
                sizeof(output_full)
            ) == NULL) {

            return 0;
        }
        return _stricmp(source_full, output_full) == 0;
    }
#else
    return strcmp(source_parent, output_parent) == 0;
#endif
}


static int gs224_create_directory(const char *path)
{
#ifdef _WIN32
    return _mkdir(path) == 0 ? 0 : 1;
#else
    return mkdir(path, 0777) == 0 ? 0 : 1;
#endif
}


static int gs224_remove_directory(const char *path)
{
#ifdef _WIN32
    return _rmdir(path) == 0 ? 0 : 1;
#else
    return rmdir(path) == 0 ? 0 : 1;
#endif
}


static int gs224_name_index(
    const char *name,
    const char *const *names,
    size_t count,
    size_t *index
)
{
    size_t position;

    if (name == NULL || names == NULL || index == NULL) {
        return 1;
    }
    for (position = 0; position < count; position++) {
        if (strcmp(name, names[position]) == 0) {
            *index = position;
            return 0;
        }
    }
    return 1;
}


static int gs224_validate_source_inventory(const char *directory)
{
    unsigned char seen[GS224_PCA_SOURCE_FILE_COUNT] = {0};
    DIR *stream;
    struct dirent *entry;
    int seen_kmeans = 0;
    int seen_evaluation = 0;
    int failed = 0;
    size_t index;

    stream = opendir(directory);
    if (stream == NULL) {
        return 1;
    }
    while (!failed) {
        char path[GS224_PCA_PATH_CAPACITY];
        struct stat information;

        errno = 0;
        entry = readdir(stream);
        if (entry == NULL) {
            if (errno != 0) {
                failed = 1;
            }
            break;
        }
        if (strcmp(entry->d_name, ".") == 0 ||
            strcmp(entry->d_name, "..") == 0) {

            continue;
        }
        if (gs224_path_join(
                directory,
                entry->d_name,
                path,
                sizeof(path)
            ) != 0 ||
            stat(path, &information) != 0) {

            failed = 1;
            break;
        }
        if (gs224_name_index(
                entry->d_name,
                gs224_source_names,
                GS224_PCA_SOURCE_FILE_COUNT,
                &index
            ) == 0) {

            if (seen[index] || !S_ISREG(information.st_mode)) {
                failed = 1;
            }
            seen[index] = 1u;
        } else if (strcmp(entry->d_name, "kmeans_seed123") == 0) {
            if (seen_kmeans || !S_ISDIR(information.st_mode)) {
                failed = 1;
            }
            seen_kmeans = 1;
        } else if (strcmp(
                       entry->d_name,
                       "label_evaluation_seed123"
                   ) == 0) {

            if (seen_evaluation || !S_ISDIR(information.st_mode)) {
                failed = 1;
            }
            seen_evaluation = 1;
        } else {
            failed = 1;
        }
    }
    if (closedir(stream) != 0) {
        failed = 1;
    }
    for (index = 0;
         !failed && index < GS224_PCA_SOURCE_FILE_COUNT;
         index++) {

        if (!seen[index]) {
            failed = 1;
        }
    }
    return failed;
}


static int gs224_read_text(
    const char *path,
    size_t maximum_size,
    char **text
)
{
    size_t size;
    FILE *file;
    char *buffer;
    int failed = 0;

    if (path == NULL || text == NULL ||
        gs224_regular_file_size(path, &size) != 0 ||
        size == 0 || size > maximum_size || size == SIZE_MAX) {

        return 1;
    }
    buffer = (char *)malloc(size + 1u);
    if (buffer == NULL) {
        return 1;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        free(buffer);
        return 1;
    }
    if (fread(buffer, 1u, size, file) != size ||
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


static int gs224_text_has_line(const char *text, const char *line)
{
    const char *cursor = text;
    size_t length;

    if (text == NULL || line == NULL) {
        return 0;
    }
    length = strlen(line);
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


static int gs224_validate_source_feature_spec(const char *directory)
{
    static const char *const required[] = {
        "profile=mini224_subset1000_pool14_no_pca",
        "train_rows=800",
        "validation_rows=200",
        "feature_count=1024",
        "matrix_layout=row_major",
        "matrix_dtype=float32_ieee754",
        "matrix_endian=little",
        "scaler_fit=train_only",
        "standardization_ddof=0",
        "pca=none",
        "augmentation=false",
        "labels_used=false",
        "code_commit=" GS224_PCA_BASE_CODE_COMMIT
    };
    char path[GS224_PCA_PATH_CAPACITY];
    char *text = NULL;
    size_t index;
    int failed = 0;

    if (gs224_path_join(
            directory,
            "feature_spec.txt",
            path,
            sizeof(path)
        ) != 0 ||
        gs224_read_text(path, 1024u * 1024u, &text) != 0) {

        free(text);
        return 1;
    }
    for (index = 0;
         index < sizeof(required) / sizeof(required[0]);
         index++) {

        if (!gs224_text_has_line(text, required[index])) {
            failed = 1;
        }
    }
    free(text);
    return failed;
}


static int gs224_validate_source_summary(const char *directory)
{
    static const char *const required[] = {
        "profile=mini224_subset1000_pool14_no_pca",
        "train_rows=800",
        "validation_rows=200",
        "feature_count=1024",
        "train_features_std_bytes=3276800",
        "val_features_std_bytes=819200",
        "constant_feature_count=0",
        "standardized_values_finite=true",
        "standardization_fit=train_only",
        "validation_transformed_without_refit=true",
        "scaler_unchanged_after_validation=true",
        "pca_used=false",
        "augmentation_used=false",
        "labels_used=false",
        "code_commit=" GS224_PCA_BASE_CODE_COMMIT
    };
    char path[GS224_PCA_PATH_CAPACITY];
    char *text = NULL;
    size_t index;
    int failed = 0;

    if (gs224_path_join(
            directory, "summary.txt", path, sizeof(path)
        ) != 0 ||
        gs224_read_text(path, 1024u * 1024u, &text) != 0) {

        free(text);
        return 1;
    }
    for (index = 0;
         index < sizeof(required) / sizeof(required[0]);
         index++) {

        if (!gs224_text_has_line(text, required[index])) {
            failed = 1;
        }
    }
    free(text);
    return failed;
}


static void gs224_strip_line_ending(char *line)
{
    size_t length;

    if (line == NULL) {
        return;
    }
    length = strlen(line);
    while (length != 0 &&
           (line[length - 1] == '\n' || line[length - 1] == '\r')) {

        line[--length] = '\0';
    }
}


static int gs224_parse_decimal_size(
    const char *text,
    size_t *value
)
{
    uintmax_t parsed = 0;
    const unsigned char *cursor;

    if (text == NULL || value == NULL || text[0] == '\0') {
        return 1;
    }
    cursor = (const unsigned char *)text;
    while (*cursor != '\0') {
        unsigned int digit;

        if (*cursor < '0' || *cursor > '9') {
            return 1;
        }
        digit = (unsigned int)(*cursor - '0');
        if (parsed > (UINTMAX_MAX - digit) / 10u) {
            return 1;
        }
        parsed = parsed * 10u + digit;
        cursor++;
    }
    if (parsed > (uintmax_t)SIZE_MAX) {
        return 1;
    }
    *value = (size_t)parsed;
    return 0;
}


static int gs224_split_csv_six(char *line, char *field[6])
{
    size_t count = 1;
    char *cursor;

    if (line == NULL || field == NULL || line[0] == '\0') {
        return 1;
    }
    field[0] = line;
    for (cursor = line; *cursor != '\0'; cursor++) {
        if (*cursor == ',' && count < 6u) {
            *cursor = '\0';
            field[count++] = cursor + 1;
        } else if (*cursor == ',') {
            return 1;
        }
    }
    return count == 6u ? 0 : 1;
}


static int gs224_validate_row_map(
    const char *directory,
    const char *filename,
    size_t expected_rows,
    const char *expected_split
)
{
    static const char header[] =
        "row_index,split_pos,manifest_index,event_index,event_id,split";
    char path[GS224_PCA_PATH_CAPACITY];
    char line[GS224_PCA_LINE_CAPACITY];
    FILE *file;
    size_t row = 0;
    int failed = 0;

    if (gs224_path_join(directory, filename, path, sizeof(path)) != 0) {
        return 1;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        return 1;
    }
    if (fgets(line, sizeof(line), file) == NULL) {
        failed = 1;
    } else {
        gs224_strip_line_ending(line);
        if (strcmp(line, header) != 0 ||
            strstr(line, "label") != NULL ||
            strstr(line, "class") != NULL ||
            strstr(line, "target") != NULL) {

            failed = 1;
        }
    }
    while (!failed && fgets(line, sizeof(line), file) != NULL) {
        char *field[6];
        size_t row_index;
        size_t split_pos;
        size_t manifest_index;
        size_t event_index;
        size_t length = strlen(line);

        if (length != 0 && line[length - 1] != '\n' && !feof(file)) {
            failed = 1;
            break;
        }
        gs224_strip_line_ending(line);
        if (row >= expected_rows ||
            gs224_split_csv_six(line, field) != 0 ||
            gs224_parse_decimal_size(field[0], &row_index) != 0 ||
            gs224_parse_decimal_size(field[1], &split_pos) != 0 ||
            gs224_parse_decimal_size(field[2], &manifest_index) != 0 ||
            gs224_parse_decimal_size(field[3], &event_index) != 0 ||
            row_index != row || split_pos != row ||
            manifest_index == SIZE_MAX ||
            event_index != manifest_index + 1u ||
            field[4][0] == '\0' ||
            strcmp(field[5], expected_split) != 0) {

            failed = 1;
            break;
        }
        row++;
    }
    if (ferror(file) || fclose(file) != 0) {
        failed = 1;
    }
    return failed || row != expected_rows;
}


static uint32_t gs224_rotr32(uint32_t value, unsigned int bits)
{
    return (value >> bits) | (value << (32u - bits));
}


static void gs224_sha256_transform(
    GS224Sha256 *context,
    const unsigned char block[64]
)
{
    static const uint32_t constants[64] = {
        UINT32_C(0x428a2f98), UINT32_C(0x71374491),
        UINT32_C(0xb5c0fbcf), UINT32_C(0xe9b5dba5),
        UINT32_C(0x3956c25b), UINT32_C(0x59f111f1),
        UINT32_C(0x923f82a4), UINT32_C(0xab1c5ed5),
        UINT32_C(0xd807aa98), UINT32_C(0x12835b01),
        UINT32_C(0x243185be), UINT32_C(0x550c7dc3),
        UINT32_C(0x72be5d74), UINT32_C(0x80deb1fe),
        UINT32_C(0x9bdc06a7), UINT32_C(0xc19bf174),
        UINT32_C(0xe49b69c1), UINT32_C(0xefbe4786),
        UINT32_C(0x0fc19dc6), UINT32_C(0x240ca1cc),
        UINT32_C(0x2de92c6f), UINT32_C(0x4a7484aa),
        UINT32_C(0x5cb0a9dc), UINT32_C(0x76f988da),
        UINT32_C(0x983e5152), UINT32_C(0xa831c66d),
        UINT32_C(0xb00327c8), UINT32_C(0xbf597fc7),
        UINT32_C(0xc6e00bf3), UINT32_C(0xd5a79147),
        UINT32_C(0x06ca6351), UINT32_C(0x14292967),
        UINT32_C(0x27b70a85), UINT32_C(0x2e1b2138),
        UINT32_C(0x4d2c6dfc), UINT32_C(0x53380d13),
        UINT32_C(0x650a7354), UINT32_C(0x766a0abb),
        UINT32_C(0x81c2c92e), UINT32_C(0x92722c85),
        UINT32_C(0xa2bfe8a1), UINT32_C(0xa81a664b),
        UINT32_C(0xc24b8b70), UINT32_C(0xc76c51a3),
        UINT32_C(0xd192e819), UINT32_C(0xd6990624),
        UINT32_C(0xf40e3585), UINT32_C(0x106aa070),
        UINT32_C(0x19a4c116), UINT32_C(0x1e376c08),
        UINT32_C(0x2748774c), UINT32_C(0x34b0bcb5),
        UINT32_C(0x391c0cb3), UINT32_C(0x4ed8aa4a),
        UINT32_C(0x5b9cca4f), UINT32_C(0x682e6ff3),
        UINT32_C(0x748f82ee), UINT32_C(0x78a5636f),
        UINT32_C(0x84c87814), UINT32_C(0x8cc70208),
        UINT32_C(0x90befffa), UINT32_C(0xa4506ceb),
        UINT32_C(0xbef9a3f7), UINT32_C(0xc67178f2)
    };
    uint32_t words[64];
    uint32_t a;
    uint32_t b;
    uint32_t c;
    uint32_t d;
    uint32_t e;
    uint32_t f;
    uint32_t g;
    uint32_t h;
    size_t index;

    for (index = 0; index < 16u; index++) {
        size_t offset = index * 4u;

        words[index] =
            ((uint32_t)block[offset] << 24) |
            ((uint32_t)block[offset + 1u] << 16) |
            ((uint32_t)block[offset + 2u] << 8) |
            (uint32_t)block[offset + 3u];
    }
    for (index = 16u; index < 64u; index++) {
        uint32_t s0 =
            gs224_rotr32(words[index - 15u], 7u) ^
            gs224_rotr32(words[index - 15u], 18u) ^
            (words[index - 15u] >> 3);
        uint32_t s1 =
            gs224_rotr32(words[index - 2u], 17u) ^
            gs224_rotr32(words[index - 2u], 19u) ^
            (words[index - 2u] >> 10);

        words[index] =
            words[index - 16u] + s0 +
            words[index - 7u] + s1;
    }

    a = context->state[0];
    b = context->state[1];
    c = context->state[2];
    d = context->state[3];
    e = context->state[4];
    f = context->state[5];
    g = context->state[6];
    h = context->state[7];
    for (index = 0; index < 64u; index++) {
        uint32_t sum1 =
            gs224_rotr32(e, 6u) ^
            gs224_rotr32(e, 11u) ^
            gs224_rotr32(e, 25u);
        uint32_t choice = (e & f) ^ ((~e) & g);
        uint32_t temporary1 =
            h + sum1 + choice + constants[index] + words[index];
        uint32_t sum0 =
            gs224_rotr32(a, 2u) ^
            gs224_rotr32(a, 13u) ^
            gs224_rotr32(a, 22u);
        uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temporary2 = sum0 + majority;

        h = g;
        g = f;
        f = e;
        e = d + temporary1;
        d = c;
        c = b;
        b = a;
        a = temporary1 + temporary2;
    }
    context->state[0] += a;
    context->state[1] += b;
    context->state[2] += c;
    context->state[3] += d;
    context->state[4] += e;
    context->state[5] += f;
    context->state[6] += g;
    context->state[7] += h;
}


static void gs224_sha256_init(GS224Sha256 *context)
{
    static const uint32_t initial[8] = {
        UINT32_C(0x6a09e667), UINT32_C(0xbb67ae85),
        UINT32_C(0x3c6ef372), UINT32_C(0xa54ff53a),
        UINT32_C(0x510e527f), UINT32_C(0x9b05688c),
        UINT32_C(0x1f83d9ab), UINT32_C(0x5be0cd19)
    };

    memcpy(context->state, initial, sizeof(initial));
    context->completed_bits = 0;
    context->block_size = 0;
}


static int gs224_sha256_update(
    GS224Sha256 *context,
    const unsigned char *data,
    size_t size
)
{
    size_t offset = 0;

    if (context == NULL || (data == NULL && size != 0) ||
        size > (UINT64_MAX - context->completed_bits) / 8u) {

        return 1;
    }
    context->completed_bits += (uint64_t)size * UINT64_C(8);
    while (offset < size) {
        size_t available = 64u - context->block_size;
        size_t remaining = size - offset;
        size_t count = remaining < available ? remaining : available;

        memcpy(context->block + context->block_size, data + offset, count);
        context->block_size += count;
        offset += count;
        if (context->block_size == 64u) {
            gs224_sha256_transform(context, context->block);
            context->block_size = 0;
        }
    }
    return 0;
}


static void gs224_sha256_final(
    GS224Sha256 *context,
    unsigned char digest[32]
)
{
    size_t index;

    context->block[context->block_size++] = 0x80u;
    if (context->block_size > 56u) {
        while (context->block_size < 64u) {
            context->block[context->block_size++] = 0u;
        }
        gs224_sha256_transform(context, context->block);
        context->block_size = 0;
    }
    while (context->block_size < 56u) {
        context->block[context->block_size++] = 0u;
    }
    for (index = 0; index < 8u; index++) {
        context->block[56u + index] =
            (unsigned char)(
                context->completed_bits >> (56u - 8u * index)
            );
    }
    gs224_sha256_transform(context, context->block);

    for (index = 0; index < 8u; index++) {
        digest[index * 4u] =
            (unsigned char)(context->state[index] >> 24);
        digest[index * 4u + 1u] =
            (unsigned char)(context->state[index] >> 16);
        digest[index * 4u + 2u] =
            (unsigned char)(context->state[index] >> 8);
        digest[index * 4u + 3u] =
            (unsigned char)context->state[index];
    }
}


static void gs224_digest_hex(
    const unsigned char digest[32],
    char hex[65]
)
{
    static const char alphabet[] = "0123456789abcdef";
    size_t index;

    for (index = 0; index < 32u; index++) {
        hex[index * 2u] = alphabet[digest[index] >> 4];
        hex[index * 2u + 1u] = alphabet[digest[index] & 15u];
    }
    hex[64] = '\0';
}


static int gs224_sha256_bytes(
    const void *data,
    size_t size,
    char hex[65]
)
{
    GS224Sha256 context;
    unsigned char digest[32];

    if ((data == NULL && size != 0) || hex == NULL) {
        return 1;
    }
    gs224_sha256_init(&context);
    if (gs224_sha256_update(
            &context,
            (const unsigned char *)data,
            size
        ) != 0) {

        return 1;
    }
    gs224_sha256_final(&context, digest);
    gs224_digest_hex(digest, hex);
    return 0;
}


static int gs224_sha256_file(const char *path, char hex[65])
{
    unsigned char buffer[65536];
    unsigned char digest[32];
    GS224Sha256 context;
    FILE *file;
    int failed = 0;

    if (path == NULL || hex == NULL) {
        return 1;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        return 1;
    }
    gs224_sha256_init(&context);
    while (!failed) {
        size_t count = fread(buffer, 1u, sizeof(buffer), file);

        if (count != 0 &&
            gs224_sha256_update(&context, buffer, count) != 0) {

            failed = 1;
        }
        if (count < sizeof(buffer)) {
            if (ferror(file)) {
                failed = 1;
            }
            break;
        }
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    if (failed) {
        return 1;
    }
    gs224_sha256_final(&context, digest);
    gs224_digest_hex(digest, hex);
    return 0;
}


static int gs224_hex_is_valid(const char *hex)
{
    size_t index;

    if (hex == NULL || strlen(hex) != 64u) {
        return 0;
    }
    for (index = 0; index < 64u; index++) {
        if (!((hex[index] >= '0' && hex[index] <= '9') ||
              (hex[index] >= 'a' && hex[index] <= 'f'))) {

            return 0;
        }
    }
    return 1;
}


static int gs224_parse_hash_line(
    char *line,
    char **role,
    char **hex,
    char **name
)
{
    char *first;
    char *second;

    if (line == NULL || role == NULL || hex == NULL || name == NULL) {
        return 1;
    }
    first = strchr(line, ' ');
    if (first == NULL || first == line) {
        return 1;
    }
    *first = '\0';
    second = strchr(first + 1, ' ');
    if (second == NULL || second == first + 1) {
        return 1;
    }
    *second = '\0';
    if (strchr(second + 1, ' ') != NULL || second[1] == '\0') {
        return 1;
    }
    *role = line;
    *hex = first + 1;
    *name = second + 1;
    return gs224_hex_is_valid(*hex) ? 0 : 1;
}


static void gs224_normalize_path_separators(char *path)
{
    char *cursor;

    if (path == NULL) {
        return;
    }
    for (cursor = path; *cursor != '\0'; cursor++) {
        if (*cursor == '\\') {
            *cursor = '/';
        }
    }
}


static int gs224_paths_equal(const char *left, const char *right)
{
    if (left == NULL || right == NULL) {
        return 0;
    }
#ifdef _WIN32
    return _stricmp(left, right) == 0;
#else
    return strcmp(left, right) == 0;
#endif
}


#ifdef _WIN32
static int gs224_windows_path_has_reparse_component(const char *path)
{
    char copy[GS224_PCA_PATH_CAPACITY];
    size_t length;
    size_t index;

    if (path == NULL) {
        return 1;
    }
    length = strlen(path);
    if (length < 3u || length >= sizeof(copy) ||
        path[1] != ':' || path[2] != '/') {

        return 1;
    }
    memcpy(copy, path, length + 1u);
    for (index = 3u; index <= length; index++) {
        if (copy[index] == '/' || copy[index] == '\0') {
            char saved = copy[index];
            DWORD attributes;

            copy[index] = '\0';
            attributes = GetFileAttributesA(copy);
            copy[index] = saved;
            if (attributes == INVALID_FILE_ATTRIBUTES ||
                (attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0u) {

                return 1;
            }
        }
    }
    return 0;
}
#endif


static int gs224_canonicalize_existing_path(
    const char *path,
    int require_directory,
    char resolved[GS224_PCA_PATH_CAPACITY]
)
{
    struct stat information;
    size_t length;

    if (!gs224_metadata_value_is_valid(path) || resolved == NULL) {
        return 1;
    }
#ifdef _WIN32
    if (_fullpath(resolved, path, GS224_PCA_PATH_CAPACITY) == NULL) {
        return 1;
    }
#else
    if (realpath(path, resolved) == NULL) {
        return 1;
    }
#endif
    gs224_normalize_path_separators(resolved);
    length = strlen(resolved);
    while (length > 1u && resolved[length - 1u] == '/') {
#ifdef _WIN32
        if (length == 3u && resolved[1] == ':') {
            break;
        }
#endif
        resolved[--length] = '\0';
    }
    if (length == 0u || length >= GS224_PCA_PATH_CAPACITY ||
        stat(resolved, &information) != 0 ||
        (require_directory && !S_ISDIR(information.st_mode)) ||
        (!require_directory && !S_ISREG(information.st_mode))) {

        return 1;
    }
#ifdef _WIN32
    if (gs224_windows_path_has_reparse_component(resolved) != 0) {
        return 1;
    }
#endif
    return 0;
}


static int gs224_resolve_repository_layout(
    const char *source_directory,
    char canonical_source[GS224_PCA_PATH_CAPACITY],
    char repository_root[GS224_PCA_PATH_CAPACITY]
)
{
    static const char source_suffix[] =
        "/sakai_multires_c/results/baseline/"
        "mini224_subset1000_pool14_no_pca";
    char canonical_root[GS224_PCA_PATH_CAPACITY];
    char expected_source[GS224_PCA_PATH_CAPACITY];
    char canonical_expected_source[GS224_PCA_PATH_CAPACITY];
    size_t source_length;
    size_t suffix_length = sizeof(source_suffix) - 1u;
    size_t root_length;

    if (source_directory == NULL || canonical_source == NULL ||
        repository_root == NULL ||
        gs224_canonicalize_existing_path(
            source_directory, 1, canonical_source
        ) != 0) {

        return 1;
    }
    source_length = strlen(canonical_source);
    if (source_length <= suffix_length) {
        return 1;
    }
    root_length = source_length - suffix_length;
#ifdef _WIN32
    if (_stricmp(
            canonical_source + root_length,
            source_suffix
        ) != 0) {
#else
    if (strcmp(
            canonical_source + root_length,
            source_suffix
        ) != 0) {
#endif

        return 1;
    }
    if (root_length == 0u || root_length >= GS224_PCA_PATH_CAPACITY) {
        return 1;
    }
    memcpy(repository_root, canonical_source, root_length);
    repository_root[root_length] = '\0';
    if (gs224_canonicalize_existing_path(
            repository_root, 1, canonical_root
        ) != 0 ||
        !gs224_paths_equal(repository_root, canonical_root) ||
        gs224_path_join(
            repository_root,
            "sakai_multires_c/results/baseline/"
            "mini224_subset1000_pool14_no_pca",
            expected_source,
            sizeof(expected_source)
        ) != 0 ||
        gs224_canonicalize_existing_path(
            expected_source, 1, canonical_expected_source
        ) != 0 ||
        !gs224_paths_equal(canonical_source, canonical_expected_source)) {

        return 1;
    }
    return 0;
}


static int gs224_resolve_source_inputs(
    const char *source_directory,
    char resolved_inputs[GS224_PCA_SOURCE_INPUT_COUNT][
        GS224_PCA_PATH_CAPACITY
    ]
)
{
    char canonical_source[GS224_PCA_PATH_CAPACITY];
    char repository_root[GS224_PCA_PATH_CAPACITY];
    size_t index;

    if (resolved_inputs == NULL ||
        gs224_resolve_repository_layout(
            source_directory,
            canonical_source,
            repository_root
        ) != 0) {

        return 1;
    }
    for (index = 0; index < GS224_PCA_SOURCE_INPUT_COUNT; index++) {
        char expected[GS224_PCA_PATH_CAPACITY];
        size_t other;

        if (gs224_path_join(
                repository_root,
                gs224_source_input_relative_paths[index],
                expected,
                sizeof(expected)
            ) != 0 ||
            gs224_canonicalize_existing_path(
                expected, 0, resolved_inputs[index]
            ) != 0 ||
            !gs224_paths_equal(expected, resolved_inputs[index]) ||
            !gs224_paths_equal(
                gs224_path_basename(resolved_inputs[index]),
                gs224_source_input_names[index]
            )) {

            return 1;
        }
        for (other = 0; other < index; other++) {
            if (gs224_paths_equal(
                    resolved_inputs[index],
                    resolved_inputs[other]
                )) {

                return 1;
            }
        }
    }
    return 0;
}


static int gs224_validate_source_hashes(
    const char *source_directory,
    char source_file_hashes[GS224_PCA_INPUT_COUNT][65],
    char resolved_inputs[GS224_PCA_SOURCE_INPUT_COUNT][
        GS224_PCA_PATH_CAPACITY
    ]
)
{
    char manifest_path[GS224_PCA_PATH_CAPACITY];
    char line[GS224_PCA_LINE_CAPACITY];
    FILE *file;
    size_t line_index = 0;
    int failed = 0;

    if (source_directory == NULL || source_file_hashes == NULL ||
        resolved_inputs == NULL ||
        gs224_resolve_source_inputs(
            source_directory,
            resolved_inputs
        ) != 0 ||
        gs224_path_join(
            source_directory,
            "sha256.txt",
            manifest_path,
            sizeof(manifest_path)
        ) != 0) {

        return 1;
    }
    file = fopen(manifest_path, "rb");
    if (file == NULL) {
        return 1;
    }
    while (!failed && fgets(line, sizeof(line), file) != NULL) {
        char *role;
        char *expected_hash;
        char *name;
        char actual_hash[65];
        char artifact_path[GS224_PCA_PATH_CAPACITY];
        const char *expected_role;
        const char *expected_name;

        if (strlen(line) != 0 &&
            line[strlen(line) - 1u] != '\n' &&
            !feof(file)) {

            failed = 1;
            break;
        }
        gs224_strip_line_ending(line);
        if (line_index >=
                GS224_PCA_SOURCE_INPUT_COUNT +
                GS224_PCA_SOURCE_ARTIFACT_COUNT ||
            gs224_parse_hash_line(
                line,
                &role,
                &expected_hash,
                &name
            ) != 0) {

            failed = 1;
            break;
        }
        if (line_index < GS224_PCA_SOURCE_INPUT_COUNT) {
            expected_role = "INPUT";
            expected_name = gs224_source_input_names[line_index];
            if (gs224_sha256_file(
                    resolved_inputs[line_index],
                    actual_hash
                ) != 0) {

                failed = 1;
                break;
            }
        } else {
            size_t artifact =
                line_index - GS224_PCA_SOURCE_INPUT_COUNT;

            expected_role = "ARTIFACT";
            expected_name = gs224_source_artifact_names[artifact];
            if (strcmp(expected_name, "sha256.txt") == 0 ||
                gs224_path_join(
                    source_directory,
                    expected_name,
                    artifact_path,
                    sizeof(artifact_path)
                ) != 0 ||
                gs224_sha256_file(artifact_path, actual_hash) != 0) {

                failed = 1;
                break;
            }
        }
        if (strcmp(role, expected_role) != 0 ||
            strcmp(name, expected_name) != 0 ||
            strcmp(actual_hash, expected_hash) != 0) {

            failed = 1;
            break;
        }
        line_index++;
    }
    if (ferror(file) || fclose(file) != 0) {
        failed = 1;
    }
    if (failed ||
        line_index !=
            GS224_PCA_SOURCE_INPUT_COUNT +
            GS224_PCA_SOURCE_ARTIFACT_COUNT) {

        return 1;
    }

    for (line_index = 0; line_index < 6u; line_index++) {
        char path[GS224_PCA_PATH_CAPACITY];

        if (gs224_path_join(
                source_directory,
                line_index == 0u ? "feature_spec.txt" :
                line_index == 1u ? "scaler.csv" :
                line_index == 2u ? "train_features_std_f32.bin" :
                line_index == 3u ? "val_features_std_f32.bin" :
                line_index == 4u ? "train_rows.csv" :
                                   "val_rows.csv",
                path,
                sizeof(path)
            ) != 0 ||
            gs224_sha256_file(path, source_file_hashes[line_index]) != 0) {

            return 1;
        }
    }
    if (gs224_sha256_file(
            manifest_path,
            source_file_hashes[6]
        ) != 0) {

        return 1;
    }
    return 0;
}


static int gs224_validate_source_contract(
    const char *source_directory,
    size_t train_bytes,
    size_t validation_bytes,
    char source_file_hashes[GS224_PCA_INPUT_COUNT][65],
    char resolved_inputs[GS224_PCA_SOURCE_INPUT_COUNT][
        GS224_PCA_PATH_CAPACITY
    ]
)
{
    static const char *const matrix_names[4] = {
        "train_features_raw_f32.bin",
        "val_features_raw_f32.bin",
        "train_features_std_f32.bin",
        "val_features_std_f32.bin"
    };
    char path[GS224_PCA_PATH_CAPACITY];
    size_t size;
    size_t index;
    int is_directory = 0;

    if (source_directory == NULL ||
        !gs224_paths_equal(
            gs224_path_basename(source_directory),
            "mini224_subset1000_pool14_no_pca"
        ) ||
        gs224_path_state(source_directory, &is_directory) != 1 ||
        !is_directory ||
        gs224_validate_source_inventory(source_directory) != 0 ||
        gs224_validate_source_feature_spec(source_directory) != 0 ||
        gs224_validate_source_summary(source_directory) != 0 ||
        gs224_validate_source_hashes(
            source_directory,
            source_file_hashes,
            resolved_inputs
        ) != 0 ||
        gs224_validate_row_map(
            source_directory,
            "train_rows.csv",
            GS224_PCA_TRAIN_ROWS,
            "train"
        ) != 0 ||
        gs224_validate_row_map(
            source_directory,
            "val_rows.csv",
            GS224_PCA_VALIDATION_ROWS,
            "validation"
        ) != 0) {

        return 1;
    }

    for (index = 0; index < 4u; index++) {
        size_t expected = (index == 0u || index == 2u)
            ? train_bytes
            : validation_bytes;

        if (gs224_path_join(
                source_directory,
                matrix_names[index],
                path,
                sizeof(path)
            ) != 0 ||
            gs224_regular_file_size(path, &size) != 0 ||
            size != expected) {

            return 1;
        }
    }
    for (index = 0; index < GS224_PCA_SOURCE_FILE_COUNT; index++) {
        if (gs224_path_join(
                source_directory,
                gs224_source_names[index],
                path,
                sizeof(path)
            ) != 0 ||
            gs224_regular_file_size(path, &size) != 0 ||
            size == 0) {

            return 1;
        }
    }
    return 0;
}


static int gs224_read_float_matrix(
    const char *path,
    float *values,
    size_t value_count,
    size_t expected_bytes
)
{
    FILE *file;
    size_t size;
    size_t index;
    int failed = 0;

    if (path == NULL || values == NULL ||
        gs224_regular_file_size(path, &size) != 0 ||
        size != expected_bytes) {

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
        if (!isfinite((double)values[index])) {
            return 1;
        }
    }
    return 0;
}


static int gs224_model_is_valid(
    const GSBaselinePCA *model,
    size_t *zero_count
)
{
    size_t component;
    size_t feature;
    size_t zeros = 0;
    double previous_eigenvalue = INFINITY;
    double previous_cumulative = 0.0;
    double ratio_sum = 0.0;

    if (model == NULL || zero_count == NULL ||
        model->input_dim != GS224_PCA_FEATURES ||
        model->component_count != GS224_PCA_FEATURES ||
        model->mean == NULL || model->components == NULL ||
        model->eigenvalues == NULL ||
        model->explained_variance_ratio == NULL ||
        model->cumulative_explained_variance_ratio == NULL ||
        !isfinite(model->total_variance) ||
        !(model->total_variance > 0.0) ||
        model->components_90 == 0 ||
        model->components_90 > model->components_95 ||
        model->components_95 > model->components_99 ||
        model->components_99 > GS224_PCA_FEATURES) {

        return 1;
    }
    for (feature = 0; feature < GS224_PCA_FEATURES; feature++) {
        if (!isfinite(model->mean[feature])) {
            return 1;
        }
    }
    for (component = 0;
         component < GS224_PCA_FEATURES;
         component++) {

        double eigenvalue = model->eigenvalues[component];
        double ratio = model->explained_variance_ratio[component];
        double cumulative =
            model->cumulative_explained_variance_ratio[component];
        size_t maximum_index = 0;
        double maximum_abs =
            fabs(model->components[component * GS224_PCA_FEATURES]);

        if (!isfinite(eigenvalue) || eigenvalue < 0.0 ||
            eigenvalue > previous_eigenvalue ||
            !isfinite(ratio) || ratio < 0.0 || ratio > 1.0 ||
            !isfinite(cumulative) ||
            cumulative < previous_cumulative ||
            cumulative > 1.0) {

            return 1;
        }
        if (eigenvalue == 0.0) {
            zeros++;
        }
        ratio_sum += ratio;
        previous_eigenvalue = eigenvalue;
        previous_cumulative = cumulative;
        for (feature = 0; feature < GS224_PCA_FEATURES; feature++) {
            size_t offset =
                component * (size_t)GS224_PCA_FEATURES + feature;
            double loading = model->components[offset];
            double loading_abs;

            if (!isfinite(loading)) {
                return 1;
            }
            loading_abs = fabs(loading);
            if (loading_abs > maximum_abs) {
                maximum_abs = loading_abs;
                maximum_index = feature;
            }
        }
        if (!(maximum_abs > 0.0) ||
            model->components[
                component * (size_t)GS224_PCA_FEATURES + maximum_index
            ] <= 0.0) {

            return 1;
        }
    }
    if (previous_cumulative != 1.0 ||
        fabs(ratio_sum - 1.0) > 64.0 * DBL_EPSILON ||
        model->cumulative_explained_variance_ratio[
            model->components_90 - 1u
        ] < 0.90 ||
        (model->components_90 > 1u &&
         model->cumulative_explained_variance_ratio[
             model->components_90 - 2u
         ] >= 0.90) ||
        model->cumulative_explained_variance_ratio[
            model->components_95 - 1u
        ] < 0.95 ||
        (model->components_95 > 1u &&
         model->cumulative_explained_variance_ratio[
             model->components_95 - 2u
         ] >= 0.95) ||
        model->cumulative_explained_variance_ratio[
            model->components_99 - 1u
        ] < 0.99 ||
        (model->components_99 > 1u &&
         model->cumulative_explained_variance_ratio[
             model->components_99 - 2u
         ] >= 0.99)) {

        return 1;
    }
    *zero_count = zeros;
    return 0;
}


static int gs224_rank_and_threshold_contract(
    const GSBaselinePCA *model,
    double *rank_tolerance,
    size_t *numerical_rank,
    size_t *nullspace_dimension
)
{
    const double expected_cumulative90 = 0.90128356272540111;
    const double expected_cumulative95 = 0.95005795165869600;
    const double expected_cumulative99 = 0.99002122575056528;
    const double expected_previous90 = 0.89653906206868739;
    const double expected_previous95 = 0.94902632077505300;
    const double expected_previous99 = 0.98992303384580238;
    double maximum_abs = 0.0;
    double tolerance;
    size_t rank = 0u;
    size_t component;

    if (model == NULL || rank_tolerance == NULL ||
        numerical_rank == NULL || nullspace_dimension == NULL ||
        model->eigenvalues == NULL ||
        model->cumulative_explained_variance_ratio == NULL ||
        model->components_90 != GS224_PCA_D90 ||
        model->components_95 != GS224_PCA_D95 ||
        model->components_99 != GS224_PCA_D99 ||
        GS224_PCA_TRAIN_ROWS - 1u != GS224_PCA_MAX_THEORETICAL_RANK ||
        GS224_PCA_FEATURES - GS224_PCA_NUMERICAL_RANK !=
            GS224_PCA_NULLSPACE_DIMENSION) {

        return 1;
    }
    for (component = 0u; component < GS224_PCA_FEATURES; component++) {
        double value = model->eigenvalues[component];

        if (!isfinite(value)) {
            return 1;
        }
        if (fabs(value) > maximum_abs) {
            maximum_abs = fabs(value);
        }
    }
    tolerance =
        1024.0 * DBL_EPSILON *
        (double)GS224_PCA_FEATURES * maximum_abs;
    if (!isfinite(tolerance) || !(tolerance > 0.0)) {
        return 1;
    }
    for (component = 0u; component < GS224_PCA_FEATURES; component++) {
        if (model->eigenvalues[component] > tolerance) {
            rank++;
        }
    }
    if (rank != GS224_PCA_NUMERICAL_RANK ||
        GS224_PCA_FEATURES - rank != GS224_PCA_NULLSPACE_DIMENSION ||
        model->components_99 > rank ||
        memcmp(
            &model->cumulative_explained_variance_ratio[
                GS224_PCA_D90 - 1u
            ],
            &expected_cumulative90,
            sizeof(double)
        ) != 0 ||
        memcmp(
            &model->cumulative_explained_variance_ratio[
                GS224_PCA_D95 - 1u
            ],
            &expected_cumulative95,
            sizeof(double)
        ) != 0 ||
        memcmp(
            &model->cumulative_explained_variance_ratio[
                GS224_PCA_D99 - 1u
            ],
            &expected_cumulative99,
            sizeof(double)
        ) != 0 ||
        memcmp(
            &model->cumulative_explained_variance_ratio[
                GS224_PCA_D90 - 2u
            ],
            &expected_previous90,
            sizeof(double)
        ) != 0 ||
        memcmp(
            &model->cumulative_explained_variance_ratio[
                GS224_PCA_D95 - 2u
            ],
            &expected_previous95,
            sizeof(double)
        ) != 0 ||
        memcmp(
            &model->cumulative_explained_variance_ratio[
                GS224_PCA_D99 - 2u
            ],
            &expected_previous99,
            sizeof(double)
        ) != 0) {

        return 1;
    }
    *rank_tolerance = tolerance;
    *numerical_rank = rank;
    *nullspace_dimension = GS224_PCA_FEATURES - rank;
    return 0;
}


static int gs224_snapshot_model(
    const GSBaselinePCA *model,
    double *snapshot,
    size_t snapshot_count,
    GS224PCAModelMetadata *metadata
)
{
    size_t matrix_count;
    size_t required;
    size_t offset = 0;

    if (model == NULL || snapshot == NULL || metadata == NULL ||
        gs224_checked_product(
            GS224_PCA_FEATURES,
            GS224_PCA_FEATURES,
            &matrix_count
        ) != 0 ||
        gs224_checked_sum(
            matrix_count,
            4u * (size_t)GS224_PCA_FEATURES,
            &required
        ) != 0 ||
        snapshot_count != required) {

        return 1;
    }
    memcpy(
        snapshot + offset,
        model->mean,
        GS224_PCA_FEATURES * sizeof(double)
    );
    offset += GS224_PCA_FEATURES;
    memcpy(
        snapshot + offset,
        model->components,
        matrix_count * sizeof(double)
    );
    offset += matrix_count;
    memcpy(
        snapshot + offset,
        model->eigenvalues,
        GS224_PCA_FEATURES * sizeof(double)
    );
    offset += GS224_PCA_FEATURES;
    memcpy(
        snapshot + offset,
        model->explained_variance_ratio,
        GS224_PCA_FEATURES * sizeof(double)
    );
    offset += GS224_PCA_FEATURES;
    memcpy(
        snapshot + offset,
        model->cumulative_explained_variance_ratio,
        GS224_PCA_FEATURES * sizeof(double)
    );

    metadata->input_dim = model->input_dim;
    metadata->component_count = model->component_count;
    metadata->total_variance = model->total_variance;
    metadata->d90 = model->components_90;
    metadata->d95 = model->components_95;
    metadata->d99 = model->components_99;
    return 0;
}


static int gs224_model_matches_snapshot(
    const GSBaselinePCA *model,
    const double *snapshot,
    size_t snapshot_count,
    const GS224PCAModelMetadata *metadata
)
{
    size_t matrix_count;
    size_t required;
    size_t offset = 0;

    if (model == NULL || snapshot == NULL || metadata == NULL ||
        gs224_checked_product(
            GS224_PCA_FEATURES,
            GS224_PCA_FEATURES,
            &matrix_count
        ) != 0 ||
        gs224_checked_sum(
            matrix_count,
            4u * (size_t)GS224_PCA_FEATURES,
            &required
        ) != 0 ||
        snapshot_count != required ||
        model->input_dim != metadata->input_dim ||
        model->component_count != metadata->component_count ||
        memcmp(
            &model->total_variance,
            &metadata->total_variance,
            sizeof(double)
        ) != 0 ||
        model->components_90 != metadata->d90 ||
        model->components_95 != metadata->d95 ||
        model->components_99 != metadata->d99) {

        return 0;
    }
    if (memcmp(
            snapshot + offset,
            model->mean,
            GS224_PCA_FEATURES * sizeof(double)
        ) != 0) {

        return 0;
    }
    offset += GS224_PCA_FEATURES;
    if (memcmp(
            snapshot + offset,
            model->components,
            matrix_count * sizeof(double)
        ) != 0) {

        return 0;
    }
    offset += matrix_count;
    if (memcmp(
            snapshot + offset,
            model->eigenvalues,
            GS224_PCA_FEATURES * sizeof(double)
        ) != 0) {

        return 0;
    }
    offset += GS224_PCA_FEATURES;
    if (memcmp(
            snapshot + offset,
            model->explained_variance_ratio,
            GS224_PCA_FEATURES * sizeof(double)
        ) != 0) {

        return 0;
    }
    offset += GS224_PCA_FEATURES;
    return memcmp(
        snapshot + offset,
        model->cumulative_explained_variance_ratio,
        GS224_PCA_FEATURES * sizeof(double)
    ) == 0;
}


static int gs224_scores_are_finite(
    const float *scores,
    size_t rows,
    size_t components
)
{
    size_t count;
    size_t index;

    if (scores == NULL || rows == 0 || components == 0 ||
        gs224_checked_product(rows, components, &count) != 0) {

        return 0;
    }
    for (index = 0; index < count; index++) {
        if (!isfinite((double)scores[index])) {
            return 0;
        }
    }
    return 1;
}


static int gs224_train_score_statistics(
    const float *scores,
    size_t rows,
    size_t components,
    const double *eigenvalues,
    double *maximum_abs_mean,
    double *maximum_variance_error
)
{
    size_t component;
    double max_mean = 0.0;
    double max_error = 0.0;

    if (scores == NULL || rows == 0 || components == 0 ||
        eigenvalues == NULL || maximum_abs_mean == NULL ||
        maximum_variance_error == NULL) {

        return 1;
    }
    for (component = 0; component < components; component++) {
        double mean = 0.0;
        double variance = 0.0;
        size_t row;

        for (row = 0; row < rows; row++) {
            size_t offset = row * components + component;

            mean += (double)scores[offset];
        }
        mean /= (double)rows;
        for (row = 0; row < rows; row++) {
            size_t offset = row * components + component;
            double centered = (double)scores[offset] - mean;

            variance += centered * centered;
        }
        variance /= (double)rows;
        if (!isfinite(mean) || !isfinite(variance)) {
            return 1;
        }
        if (fabs(mean) > max_mean) {
            max_mean = fabs(mean);
        }
        if (fabs(variance - eigenvalues[component]) > max_error) {
            max_error = fabs(variance - eigenvalues[component]);
        }
    }
    *maximum_abs_mean = max_mean;
    *maximum_variance_error = max_error;
    return 0;
}


static int gs224_finish_write(FILE *file)
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


static int gs224_open_output(
    const char *directory,
    const char *name,
    FILE **file
)
{
    char path[GS224_PCA_PATH_CAPACITY];

    if (directory == NULL || name == NULL || file == NULL ||
        gs224_path_join(directory, name, path, sizeof(path)) != 0 ||
        gs224_path_state(path, NULL) != 0) {

        return 1;
    }
    *file = fopen(path, "wb");
    return *file == NULL ? 1 : 0;
}


static int gs224_write_binary(
    const char *directory,
    const char *name,
    const void *values,
    size_t element_size,
    size_t count
)
{
    FILE *file = NULL;
    size_t bytes;
    int failed = 0;

    if (values == NULL ||
        gs224_checked_product(element_size, count, &bytes) != 0 ||
        bytes == 0 ||
        gs224_open_output(directory, name, &file) != 0) {

        return 1;
    }
    if (fwrite(values, element_size, count, file) != count) {
        failed = 1;
    }
    if (gs224_finish_write(file) != 0) {
        failed = 1;
    }
    return failed;
}


static void gs224_fill_artifact_names(
    const GS224PCAProfile *profile,
    const char *names[GS224_PCA_ARTIFACT_COUNT]
)
{
    names[0] = "pca_spec.txt";
    names[1] = "mean_f64.bin";
    names[2] = "components_f64.bin";
    names[3] = "eigenvalues_f64.bin";
    names[4] = "explained_variance_ratio_f64.bin";
    names[5] = "cumulative_explained_variance_ratio_f64.bin";
    names[6] = "thresholds.csv";
    names[7] = "spectrum.csv";
    names[8] = profile->train_score_name;
    names[9] = profile->validation_score_name;
    names[10] = "summary.txt";
}


static int gs224_write_pca_spec(
    const char *directory,
    const GS224PCAProfile *profile,
    const GSBaselinePCA *model,
    size_t retained_components
)
{
    FILE *file = NULL;
    double rank_tolerance;
    size_t numerical_rank;
    size_t nullspace_dimension;
    int failed = 0;

    if (gs224_rank_and_threshold_contract(
            model,
            &rank_tolerance,
            &numerical_rank,
            &nullspace_dimension
        ) != 0 ||
        gs224_open_output(directory, "pca_spec.txt", &file) != 0) {

        return 1;
    }
    if (fprintf(
            file,
            "format_version=1\n"
            "profile=%s\n"
            "source_profile=mini224_subset1000_pool14_no_pca\n"
            "selected_threshold=%.2f\n"
            "original_feature_count=1024\n"
            "numerical_regime=rank_deficient\n"
            "max_theoretical_rank=799\n"
            "numerical_rank=%lu\n"
            "nullspace_dimension=%lu\n"
            "rank_tolerance=%.17g\n"
            "nullspace_basis_scientific_interpretation=false\n"
            "nullspace_validation=orthogonal_projector\n"
            "retained_prefix_validation=bitwise\n"
            "full_model_serialized=true\n"
            "retained_component_count=%lu\n"
            "d90=%lu\n"
            "d95=%lu\n"
            "d99=%lu\n"
            "cumulative_variance_at_retained_count=%.17g\n"
            "train_rows=800\n"
            "validation_rows=200\n"
            "input_dtype=float32\n"
            "model_dtype=float64\n"
            "score_dtype=float32\n"
            "input_order=row_major\n"
            "component_order=component_major\n"
            "score_order=row_major\n"
            "covariance_ddof=0\n"
            "eigensolver=lapacke_dsyevd\n"
            "threading_policy=external_single_thread\n"
            "sign_convention=max_abs_loading_positive\n"
            "whitening=false\n"
            "post_standardization=false\n"
            "fit_input=train_only\n"
            "validation_transform_without_refit=true\n"
            "augmentation=false\n"
            "labels_used=false\n"
            "split_seed=123\n",
            profile->profile,
            profile->threshold,
            (unsigned long)numerical_rank,
            (unsigned long)nullspace_dimension,
            rank_tolerance,
            (unsigned long)retained_components,
            (unsigned long)model->components_90,
            (unsigned long)model->components_95,
            (unsigned long)model->components_99,
            model->cumulative_explained_variance_ratio[
                retained_components - 1u
            ]
        ) < 0) {

        failed = 1;
    }
    if (gs224_finish_write(file) != 0) {
        failed = 1;
    }
    return failed;
}


static int gs224_write_thresholds(
    const char *directory,
    const GSBaselinePCA *model
)
{
    FILE *file = NULL;
    int failed = 0;

    if (gs224_open_output(directory, "thresholds.csv", &file) != 0) {
        return 1;
    }
    if (fprintf(
            file,
            "threshold,component_count,cumulative_variance\n"
            "0.90,%lu,%.17g\n"
            "0.95,%lu,%.17g\n"
            "0.99,%lu,%.17g\n",
            (unsigned long)model->components_90,
            model->cumulative_explained_variance_ratio[
                model->components_90 - 1u
            ],
            (unsigned long)model->components_95,
            model->cumulative_explained_variance_ratio[
                model->components_95 - 1u
            ],
            (unsigned long)model->components_99,
            model->cumulative_explained_variance_ratio[
                model->components_99 - 1u
            ]
        ) < 0) {

        failed = 1;
    }
    if (gs224_finish_write(file) != 0) {
        failed = 1;
    }
    return failed;
}


static int gs224_write_spectrum(
    const char *directory,
    const GSBaselinePCA *model
)
{
    FILE *file = NULL;
    size_t component;
    int failed = 0;

    if (gs224_open_output(directory, "spectrum.csv", &file) != 0) {
        return 1;
    }
    if (fprintf(
            file,
            "component_index,eigenvalue,explained_variance_ratio,"
            "cumulative_explained_variance_ratio\n"
        ) < 0) {

        failed = 1;
    }
    for (component = 0;
         !failed && component < GS224_PCA_FEATURES;
         component++) {

        if (fprintf(
                file,
                "%lu,%.17g,%.17g,%.17g\n",
                (unsigned long)component,
                model->eigenvalues[component],
                model->explained_variance_ratio[component],
                model->cumulative_explained_variance_ratio[component]
            ) < 0) {

            failed = 1;
        }
    }
    if (gs224_finish_write(file) != 0) {
        failed = 1;
    }
    return failed;
}


static int gs224_write_summary(
    const char *directory,
    const GS224PCAProfile *profile,
    const char *code_commit,
    const char source_manifest_hash[65],
    const GSBaselinePCA *model,
    size_t retained_components,
    size_t zero_count,
    double train_score_max_abs_mean,
    double train_score_max_variance_error
)
{
    FILE *file = NULL;
    double rank_tolerance;
    size_t numerical_rank;
    size_t nullspace_dimension;
    int failed = 0;

    if (gs224_rank_and_threshold_contract(
            model,
            &rank_tolerance,
            &numerical_rank,
            &nullspace_dimension
        ) != 0 ||
        gs224_open_output(directory, "summary.txt", &file) != 0) {

        return 1;
    }
    if (fprintf(
            file,
            "format_version=1\n"
            "profile=%s\n"
            "source_profile=mini224_subset1000_pool14_no_pca\n"
            "code_commit=%s\n"
            "source_sha256_manifest_hash=%s\n"
            "selected_threshold=%.2f\n"
            "retained_component_count=%lu\n"
            "numerical_regime=rank_deficient\n"
            "max_theoretical_rank=799\n"
            "numerical_rank=%lu\n"
            "nullspace_dimension=%lu\n"
            "rank_tolerance=%.17g\n"
            "nullspace_basis_scientific_interpretation=false\n"
            "nullspace_validation=orthogonal_projector\n"
            "retained_prefix_validation=bitwise\n"
            "full_model_serialized=true\n"
            "d90=%lu\n"
            "d95=%lu\n"
            "d99=%lu\n"
            "cumulative_variance_at_retained_count=%.17g\n"
            "total_variance=%.17g\n"
            "largest_eigenvalue=%.17g\n"
            "smallest_eigenvalue=%.17g\n"
            "eigenvalues_zero_after_tolerance=%lu\n"
            "train_score_max_abs_mean=%.17g\n"
            "train_score_max_population_variance_error=%.17g\n"
            "pca_fit=train_only\n"
            "validation_transformed_without_refit=true\n"
            "model_unchanged_after_validation=true\n"
            "train_input_unchanged=true\n"
            "validation_input_unchanged=true\n"
            "scores_finite=true\n"
            "covariance_ddof=0\n"
            "eigensolver=lapacke_dsyevd\n"
            "threading_policy=external_single_thread\n"
            "sign_convention=max_abs_loading_positive\n"
            "whitening=false\n"
            "post_standardization=false\n"
            "augmentation=false\n"
            "labels_used=false\n",
            profile->profile,
            code_commit,
            source_manifest_hash,
            profile->threshold,
            (unsigned long)retained_components,
            (unsigned long)numerical_rank,
            (unsigned long)nullspace_dimension,
            rank_tolerance,
            (unsigned long)model->components_90,
            (unsigned long)model->components_95,
            (unsigned long)model->components_99,
            model->cumulative_explained_variance_ratio[
                retained_components - 1u
            ],
            model->total_variance,
            model->eigenvalues[0],
            model->eigenvalues[GS224_PCA_FEATURES - 1u],
            (unsigned long)zero_count,
            train_score_max_abs_mean,
            train_score_max_variance_error
        ) < 0) {

        failed = 1;
    }
    if (gs224_finish_write(file) != 0) {
        failed = 1;
    }
    return failed;
}


static int gs224_write_sha256_manifest(
    const char *source_directory,
    const char *output_directory,
    const GS224PCAProfile *profile,
    char source_file_hashes[GS224_PCA_INPUT_COUNT][65]
)
{
    static const char *const input_names[GS224_PCA_INPUT_COUNT] = {
        "feature_spec.txt",
        "scaler.csv",
        "train_features_std_f32.bin",
        "val_features_std_f32.bin",
        "train_rows.csv",
        "val_rows.csv",
        "source_no_pca_sha256.txt"
    };
    const char *artifact_names[GS224_PCA_ARTIFACT_COUNT];
    char path[GS224_PCA_PATH_CAPACITY];
    char hash[65];
    FILE *file = NULL;
    size_t index;
    int failed = 0;

    (void)source_directory;
    gs224_fill_artifact_names(profile, artifact_names);
    if (gs224_open_output(output_directory, "sha256.txt", &file) != 0) {
        return 1;
    }
    for (index = 0; !failed && index < GS224_PCA_INPUT_COUNT; index++) {
        if (fprintf(
                file,
                "INPUT %s %s\n",
                source_file_hashes[index],
                input_names[index]
            ) < 0) {

            failed = 1;
        }
    }
    for (index = 0;
         !failed && index < GS224_PCA_ARTIFACT_COUNT;
         index++) {

        if (strcmp(artifact_names[index], "sha256.txt") == 0 ||
            gs224_path_join(
                output_directory,
                artifact_names[index],
                path,
                sizeof(path)
            ) != 0 ||
            gs224_sha256_file(path, hash) != 0 ||
            fprintf(
                file,
                "ARTIFACT %s %s\n",
                hash,
                artifact_names[index]
            ) < 0) {

            failed = 1;
        }
    }
    if (gs224_finish_write(file) != 0) {
        failed = 1;
    }
    return failed;
}


static int gs224_output_name_index(
    const char *name,
    const GS224PCAProfile *profile,
    size_t *index
)
{
    const char *artifact_names[GS224_PCA_ARTIFACT_COUNT];
    size_t position;

    gs224_fill_artifact_names(profile, artifact_names);
    for (position = 0;
         position < GS224_PCA_ARTIFACT_COUNT;
         position++) {

        if (strcmp(name, artifact_names[position]) == 0) {
            *index = position;
            return 0;
        }
    }
    if (strcmp(name, "sha256.txt") == 0) {
        *index = GS224_PCA_ARTIFACT_COUNT;
        return 0;
    }
    return 1;
}


static int gs224_validate_output_inventory(
    const char *directory,
    const GS224PCAProfile *profile
)
{
    unsigned char seen[GS224_PCA_OUTPUT_FILE_COUNT] = {0};
    DIR *stream;
    struct dirent *entry;
    int failed = 0;
    size_t index;

    stream = opendir(directory);
    if (stream == NULL) {
        return 1;
    }
    while (!failed) {
        char path[GS224_PCA_PATH_CAPACITY];
        struct stat information;

        errno = 0;
        entry = readdir(stream);
        if (entry == NULL) {
            if (errno != 0) {
                failed = 1;
            }
            break;
        }
        if (strcmp(entry->d_name, ".") == 0 ||
            strcmp(entry->d_name, "..") == 0) {

            continue;
        }
        if (gs224_output_name_index(
                entry->d_name,
                profile,
                &index
            ) != 0 ||
            seen[index] ||
            gs224_path_join(
                directory,
                entry->d_name,
                path,
                sizeof(path)
            ) != 0 ||
            stat(path, &information) != 0 ||
            !S_ISREG(information.st_mode)) {

            failed = 1;
            break;
        }
        seen[index] = 1u;
    }
    if (closedir(stream) != 0) {
        failed = 1;
    }
    for (index = 0;
         !failed && index < GS224_PCA_OUTPUT_FILE_COUNT;
         index++) {

        if (!seen[index]) {
            failed = 1;
        }
    }
    return failed;
}


static int gs224_validate_output_sizes(
    const char *directory,
    const GS224PCAProfile *profile,
    size_t retained_components
)
{
    static const char *const fixed_binary_names[5] = {
        "mean_f64.bin",
        "components_f64.bin",
        "eigenvalues_f64.bin",
        "explained_variance_ratio_f64.bin",
        "cumulative_explained_variance_ratio_f64.bin"
    };
    size_t fixed_binary_sizes[5];
    char path[GS224_PCA_PATH_CAPACITY];
    size_t matrix_count;
    size_t train_score_count;
    size_t validation_score_count;
    size_t size;
    size_t index;

    if (gs224_checked_product(
            GS224_PCA_FEATURES,
            GS224_PCA_FEATURES,
            &matrix_count
        ) != 0 ||
        gs224_checked_product(
            matrix_count,
            sizeof(double),
            &fixed_binary_sizes[1]
        ) != 0 ||
        gs224_checked_product(
            GS224_PCA_FEATURES,
            sizeof(double),
            &fixed_binary_sizes[0]
        ) != 0 ||
        gs224_checked_product(
            GS224_PCA_TRAIN_ROWS,
            retained_components,
            &train_score_count
        ) != 0 ||
        gs224_checked_product(
            GS224_PCA_VALIDATION_ROWS,
            retained_components,
            &validation_score_count
        ) != 0 ||
        gs224_checked_product(
            train_score_count,
            sizeof(float),
            &train_score_count
        ) != 0 ||
        gs224_checked_product(
            validation_score_count,
            sizeof(float),
            &validation_score_count
        ) != 0) {

        return 1;
    }
    fixed_binary_sizes[2] = fixed_binary_sizes[0];
    fixed_binary_sizes[3] = fixed_binary_sizes[0];
    fixed_binary_sizes[4] = fixed_binary_sizes[0];

    for (index = 0; index < 5u; index++) {
        if (gs224_path_join(
                directory,
                fixed_binary_names[index],
                path,
                sizeof(path)
            ) != 0 ||
            gs224_regular_file_size(path, &size) != 0 ||
            size != fixed_binary_sizes[index]) {

            return 1;
        }
    }
    if (gs224_path_join(
            directory,
            profile->train_score_name,
            path,
            sizeof(path)
        ) != 0 ||
        gs224_regular_file_size(path, &size) != 0 ||
        size != train_score_count ||
        gs224_path_join(
            directory,
            profile->validation_score_name,
            path,
            sizeof(path)
        ) != 0 ||
        gs224_regular_file_size(path, &size) != 0 ||
        size != validation_score_count) {

        return 1;
    }
    for (index = 0; index < 9u; index++) {
        if (gs224_path_join(
                directory,
                gs224_pca_fixed_artifact_names[index],
                path,
                sizeof(path)
            ) != 0 ||
            gs224_regular_file_size(path, &size) != 0 ||
            size == 0) {

            return 1;
        }
    }
    if (gs224_path_join(
            directory,
            "sha256.txt",
            path,
            sizeof(path)
        ) != 0 ||
        gs224_regular_file_size(path, &size) != 0 ||
        size == 0) {

        return 1;
    }
    return 0;
}


static int gs224_validate_pca_hashes(
    const char *source_directory,
    const char *artifact_directory,
    const GS224PCAProfile *profile,
    char source_file_hashes[GS224_PCA_INPUT_COUNT][65]
)
{
    static const char *const input_names[GS224_PCA_INPUT_COUNT] = {
        "feature_spec.txt",
        "scaler.csv",
        "train_features_std_f32.bin",
        "val_features_std_f32.bin",
        "train_rows.csv",
        "val_rows.csv",
        "source_no_pca_sha256.txt"
    };
    const char *artifact_names[GS224_PCA_ARTIFACT_COUNT];
    char manifest_path[GS224_PCA_PATH_CAPACITY];
    char line[GS224_PCA_LINE_CAPACITY];
    FILE *file;
    size_t line_index = 0;
    int failed = 0;

    gs224_fill_artifact_names(profile, artifact_names);
    if (gs224_path_join(
            artifact_directory,
            "sha256.txt",
            manifest_path,
            sizeof(manifest_path)
        ) != 0) {

        return 1;
    }
    file = fopen(manifest_path, "rb");
    if (file == NULL) {
        return 1;
    }
    while (!failed && fgets(line, sizeof(line), file) != NULL) {
        char *role;
        char *expected_hash;
        char *name;
        char actual_hash[65];
        char path[GS224_PCA_PATH_CAPACITY];
        const char *expected_role;
        const char *expected_name;

        if (strlen(line) != 0 &&
            line[strlen(line) - 1u] != '\n' &&
            !feof(file)) {

            failed = 1;
            break;
        }
        gs224_strip_line_ending(line);
        if (line_index >=
                GS224_PCA_INPUT_COUNT + GS224_PCA_ARTIFACT_COUNT ||
            gs224_parse_hash_line(
                line,
                &role,
                &expected_hash,
                &name
            ) != 0) {

            failed = 1;
            break;
        }

        if (line_index < GS224_PCA_INPUT_COUNT) {
            expected_role = "INPUT";
            expected_name = input_names[line_index];
            if (gs224_path_join(
                    source_directory,
                    line_index == 6u ? "sha256.txt" : expected_name,
                    path,
                    sizeof(path)
                ) != 0 ||
                gs224_sha256_file(path, actual_hash) != 0 ||
                strcmp(
                    actual_hash,
                    source_file_hashes[line_index]
                ) != 0) {

                failed = 1;
                break;
            }
        } else {
            size_t artifact = line_index - GS224_PCA_INPUT_COUNT;

            expected_role = "ARTIFACT";
            expected_name = artifact_names[artifact];
            if (strcmp(expected_name, "sha256.txt") == 0 ||
                gs224_path_join(
                    artifact_directory,
                    expected_name,
                    path,
                    sizeof(path)
                ) != 0 ||
                gs224_sha256_file(path, actual_hash) != 0) {

                failed = 1;
                break;
            }
        }
        if (strcmp(role, expected_role) != 0 ||
            strcmp(name, expected_name) != 0 ||
            strcmp(expected_hash, actual_hash) != 0) {

            failed = 1;
            break;
        }
        line_index++;
    }
    if (ferror(file) || fclose(file) != 0) {
        failed = 1;
    }
    return failed ||
           line_index !=
               GS224_PCA_INPUT_COUNT + GS224_PCA_ARTIFACT_COUNT;
}


static int gs224_validate_written_artifacts(
    const char *source_directory,
    const char *artifact_directory,
    const GS224PCAProfile *profile,
    size_t retained_components,
    char source_file_hashes[GS224_PCA_INPUT_COUNT][65]
)
{
    return
        gs224_validate_output_inventory(
            artifact_directory,
            profile
        ) != 0 ||
        gs224_validate_output_sizes(
            artifact_directory,
            profile,
            retained_components
        ) != 0 ||
        gs224_validate_pca_hashes(
            source_directory,
            artifact_directory,
            profile,
            source_file_hashes
        ) != 0;
}


static int gs224_remove_created_partial(
    const char *partial_directory,
    const GS224PCAProfile *profile
)
{
    const char *artifact_names[GS224_PCA_ARTIFACT_COUNT];
    char path[GS224_PCA_PATH_CAPACITY];
    size_t index;
    int failed = 0;

    gs224_fill_artifact_names(profile, artifact_names);
    for (index = 0;
         index < GS224_PCA_ARTIFACT_COUNT;
         index++) {

        if (gs224_path_join(
                partial_directory,
                artifact_names[index],
                path,
                sizeof(path)
            ) != 0) {

            failed = 1;
            continue;
        }
        errno = 0;
        if (remove(path) != 0 && errno != ENOENT) {
            failed = 1;
        }
    }
    if (gs224_path_join(
            partial_directory,
            "sha256.txt",
            path,
            sizeof(path)
        ) != 0) {

        failed = 1;
    } else {
        errno = 0;
        if (remove(path) != 0 && errno != ENOENT) {
            failed = 1;
        }
    }
    if (gs224_remove_directory(partial_directory) != 0) {
        failed = 1;
    }
    return failed;
}


static void gs224_print_usage(const char *program)
{
    printf(
        "Usage: %s <source_no_pca_directory> "
        "<pca90|pca95|pca99> <output_partial_directory> "
        "<code_commit>\n",
        program
    );
}


int main(int argc, char **argv)
{
    GSBaselinePCA model = {0};
    GS224PCAProfile profile;
    GS224PCAModelMetadata metadata;
    char source_file_hashes[GS224_PCA_INPUT_COUNT][65];
    char resolved_inputs[GS224_PCA_SOURCE_INPUT_COUNT][
        GS224_PCA_PATH_CAPACITY
    ];
    char final_directory[GS224_PCA_PATH_CAPACITY];
    char train_path[GS224_PCA_PATH_CAPACITY];
    char validation_path[GS224_PCA_PATH_CAPACITY];
    char train_hash_before[65];
    char validation_hash_before[65];
    char train_hash_after[65];
    char validation_hash_after[65];
    float *train = NULL;
    float *validation = NULL;
    float *train_scores = NULL;
    float *validation_scores = NULL;
    double *model_snapshot = NULL;
    size_t train_value_count;
    size_t validation_value_count;
    size_t train_bytes;
    size_t validation_bytes;
    size_t matrix_count;
    size_t snapshot_count;
    size_t snapshot_bytes;
    size_t retained_components;
    size_t train_score_count;
    size_t validation_score_count;
    size_t train_score_bytes;
    size_t validation_score_bytes;
    size_t zero_count = 0;
    size_t numerical_rank = 0;
    size_t nullspace_dimension = 0;
    double rank_tolerance = 0.0;
    double train_score_max_abs_mean;
    double train_score_max_variance_error;
    int failure_hook = 0;
    int partial_created = 0;
    int rc = EXIT_FAILURE;

    if (argc == 2 && strcmp(argv[1], "--help") == 0) {
        gs224_print_usage(argv[0]);
        return EXIT_SUCCESS;
    }
    if (argc != 5) {
        gs224_print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    if (!gs224_platform_is_valid() ||
        !gs224_metadata_value_is_valid(argv[1]) ||
        gs224_profile_from_token(argv[2], &profile) != 0 ||
        !gs224_code_commit_is_valid(argv[4]) ||
        gs224_validate_output_path(
            argv[3],
            &profile,
            final_directory,
            sizeof(final_directory),
            &failure_hook
        ) != 0 ||
        !gs224_output_parent_matches_source(argv[1], argv[3]) ||
        gs224_checked_product(
            GS224_PCA_TRAIN_ROWS,
            GS224_PCA_FEATURES,
            &train_value_count
        ) != 0 ||
        gs224_checked_product(
            GS224_PCA_VALIDATION_ROWS,
            GS224_PCA_FEATURES,
            &validation_value_count
        ) != 0 ||
        gs224_checked_product(
            train_value_count,
            sizeof(float),
            &train_bytes
        ) != 0 ||
        gs224_checked_product(
            validation_value_count,
            sizeof(float),
            &validation_bytes
        ) != 0 ||
        train_bytes != 3276800u ||
        validation_bytes != 819200u ||
        gs224_checked_product(
            GS224_PCA_FEATURES,
            GS224_PCA_FEATURES,
            &matrix_count
        ) != 0 ||
        gs224_checked_sum(
            matrix_count,
            4u * (size_t)GS224_PCA_FEATURES,
            &snapshot_count
        ) != 0 ||
        gs224_checked_product(
            snapshot_count,
            sizeof(double),
            &snapshot_bytes
        ) != 0) {

        fprintf(stderr, "Error: invalid arguments or platform contract.\n");
        return EXIT_FAILURE;
    }

    if (gs224_validate_source_contract(
            argv[1],
            train_bytes,
            validation_bytes,
            source_file_hashes,
            resolved_inputs
        ) != 0 ||
        gs224_path_join(
            argv[1],
            "train_features_std_f32.bin",
            train_path,
            sizeof(train_path)
        ) != 0 ||
        gs224_path_join(
            argv[1],
            "val_features_std_f32.bin",
            validation_path,
            sizeof(validation_path)
        ) != 0) {

        fprintf(stderr, "Error: invalid no-PCA source contract.\n");
        goto cleanup;
    }

    train = (float *)malloc(train_bytes);
    validation = (float *)malloc(validation_bytes);
    model_snapshot = (double *)malloc(snapshot_bytes);
    if (train == NULL || validation == NULL || model_snapshot == NULL ||
        gs224_read_float_matrix(
            train_path,
            train,
            train_value_count,
            train_bytes
        ) != 0 ||
        gs224_read_float_matrix(
            validation_path,
            validation,
            validation_value_count,
            validation_bytes
        ) != 0 ||
        gs224_sha256_bytes(
            train,
            train_bytes,
            train_hash_before
        ) != 0 ||
        gs224_sha256_bytes(
            validation,
            validation_bytes,
            validation_hash_before
        ) != 0) {

        fprintf(stderr, "Error: standardized matrix read failed.\n");
        goto cleanup;
    }

    if (gs_baseline_pca_fit(
            train,
            GS224_PCA_TRAIN_ROWS,
            GS224_PCA_FEATURES,
            &model
        ) != GS_BASELINE_PCA_SUCCESS ||
        gs224_model_is_valid(&model, &zero_count) != 0 ||
        gs224_rank_and_threshold_contract(
            &model,
            &rank_tolerance,
            &numerical_rank,
            &nullspace_dimension
        ) != 0) {

        fprintf(stderr, "Error: train-only PCA fit failed.\n");
        goto cleanup;
    }
    retained_components =
        strcmp(profile.token, "pca90") == 0 ? model.components_90 :
        strcmp(profile.token, "pca95") == 0 ? model.components_95 :
                                             model.components_99;
    if (retained_components == 0 ||
        retained_components > GS224_PCA_FEATURES ||
        gs224_checked_product(
            GS224_PCA_TRAIN_ROWS,
            retained_components,
            &train_score_count
        ) != 0 ||
        gs224_checked_product(
            GS224_PCA_VALIDATION_ROWS,
            retained_components,
            &validation_score_count
        ) != 0 ||
        gs224_checked_product(
            train_score_count,
            sizeof(float),
            &train_score_bytes
        ) != 0 ||
        gs224_checked_product(
            validation_score_count,
            sizeof(float),
            &validation_score_bytes
        ) != 0) {

        fprintf(stderr, "Error: retained score size overflow.\n");
        goto cleanup;
    }
    train_scores = (float *)malloc(train_score_bytes);
    validation_scores = (float *)malloc(validation_score_bytes);
    if (train_scores == NULL || validation_scores == NULL ||
        gs_baseline_pca_transform(
            &model,
            train,
            GS224_PCA_TRAIN_ROWS,
            GS224_PCA_FEATURES,
            retained_components,
            train_scores,
            train_score_count
        ) != GS_BASELINE_PCA_SUCCESS ||
        gs224_snapshot_model(
            &model,
            model_snapshot,
            snapshot_count,
            &metadata
        ) != 0 ||
        gs_baseline_pca_transform(
            &model,
            validation,
            GS224_PCA_VALIDATION_ROWS,
            GS224_PCA_FEATURES,
            retained_components,
            validation_scores,
            validation_score_count
        ) != GS_BASELINE_PCA_SUCCESS ||
        !gs224_model_matches_snapshot(
            &model,
            model_snapshot,
            snapshot_count,
            &metadata
        ) ||
        !gs224_scores_are_finite(
            train_scores,
            GS224_PCA_TRAIN_ROWS,
            retained_components
        ) ||
        !gs224_scores_are_finite(
            validation_scores,
            GS224_PCA_VALIDATION_ROWS,
            retained_components
        ) ||
        gs224_train_score_statistics(
            train_scores,
            GS224_PCA_TRAIN_ROWS,
            retained_components,
            model.eigenvalues,
            &train_score_max_abs_mean,
            &train_score_max_variance_error
        ) != 0 ||
        gs224_sha256_bytes(train, train_bytes, train_hash_after) != 0 ||
        gs224_sha256_bytes(
            validation,
            validation_bytes,
            validation_hash_after
        ) != 0 ||
        strcmp(train_hash_before, train_hash_after) != 0 ||
        strcmp(validation_hash_before, validation_hash_after) != 0) {

        fprintf(stderr, "Error: PCA transform or invariance check failed.\n");
        goto cleanup;
    }

    if (gs224_create_directory(argv[3]) != 0) {
        fprintf(stderr, "Error: cannot create current partial directory.\n");
        goto cleanup;
    }
    partial_created = 1;
    if (failure_hook) {
        fprintf(stderr, "Error: confined failure hook requested.\n");
        goto cleanup;
    }

    if (gs224_write_pca_spec(
            argv[3],
            &profile,
            &model,
            retained_components
        ) != 0 ||
        gs224_write_binary(
            argv[3],
            "mean_f64.bin",
            model.mean,
            sizeof(double),
            GS224_PCA_FEATURES
        ) != 0 ||
        gs224_write_binary(
            argv[3],
            "components_f64.bin",
            model.components,
            sizeof(double),
            matrix_count
        ) != 0 ||
        gs224_write_binary(
            argv[3],
            "eigenvalues_f64.bin",
            model.eigenvalues,
            sizeof(double),
            GS224_PCA_FEATURES
        ) != 0 ||
        gs224_write_binary(
            argv[3],
            "explained_variance_ratio_f64.bin",
            model.explained_variance_ratio,
            sizeof(double),
            GS224_PCA_FEATURES
        ) != 0 ||
        gs224_write_binary(
            argv[3],
            "cumulative_explained_variance_ratio_f64.bin",
            model.cumulative_explained_variance_ratio,
            sizeof(double),
            GS224_PCA_FEATURES
        ) != 0 ||
        gs224_write_thresholds(argv[3], &model) != 0 ||
        gs224_write_spectrum(argv[3], &model) != 0 ||
        gs224_write_binary(
            argv[3],
            profile.train_score_name,
            train_scores,
            sizeof(float),
            train_score_count
        ) != 0 ||
        gs224_write_binary(
            argv[3],
            profile.validation_score_name,
            validation_scores,
            sizeof(float),
            validation_score_count
        ) != 0 ||
        gs224_write_summary(
            argv[3],
            &profile,
            argv[4],
            source_file_hashes[6],
            &model,
            retained_components,
            zero_count,
            train_score_max_abs_mean,
            train_score_max_variance_error
        ) != 0 ||
        gs224_write_sha256_manifest(
            argv[1],
            argv[3],
            &profile,
            source_file_hashes
        ) != 0 ||
        gs224_validate_written_artifacts(
            argv[1],
            argv[3],
            &profile,
            retained_components,
            source_file_hashes
        ) != 0) {

        fprintf(stderr, "Error: PCA artifact transaction validation failed.\n");
        goto cleanup;
    }
    if (rename(argv[3], final_directory) != 0) {
        fprintf(stderr, "Error: atomic partial promotion failed.\n");
        goto cleanup;
    }
    partial_created = 0;

    printf("[PASS] source_inventory_and_sha256\n");
    printf("[PASS] train_only_pca_fit\n");
    printf("[PASS] train_validation_transform_and_invariance\n");
    printf("[PASS] exact_artifact_inventory_and_sha256\n");
    printf("[PASS] atomic_partial_promotion\n");
    printf("PROFILE=%s\n", profile.profile);
    printf("D90=%lu\n", (unsigned long)model.components_90);
    printf("D95=%lu\n", (unsigned long)model.components_95);
    printf("D99=%lu\n", (unsigned long)model.components_99);
    printf("PCA224_NUMERICAL_RANK=%lu\n", (unsigned long)numerical_rank);
    printf(
        "PCA224_NULLSPACE_DIMENSION=%lu\n",
        (unsigned long)nullspace_dimension
    );
    printf("PCA224_RANK_TOLERANCE=%.17g\n", rank_tolerance);
    printf(
        "CUMULATIVE90=%.17g\n",
        model.cumulative_explained_variance_ratio[
            model.components_90 - 1u
        ]
    );
    printf(
        "CUMULATIVE95=%.17g\n",
        model.cumulative_explained_variance_ratio[
            model.components_95 - 1u
        ]
    );
    printf(
        "CUMULATIVE99=%.17g\n",
        model.cumulative_explained_variance_ratio[
            model.components_99 - 1u
        ]
    );
    printf("MINI224_SUBSET1000 PCA ARTIFACT BUILD COMPLETED\n");
    rc = EXIT_SUCCESS;

cleanup:
    if (partial_created &&
        gs224_remove_created_partial(argv[3], &profile) != 0) {

        fprintf(stderr, "Error: cleanup of current partial failed.\n");
        rc = EXIT_FAILURE;
    }
    free(model_snapshot);
    free(validation_scores);
    free(train_scores);
    free(validation);
    free(train);
    gs_baseline_pca_free(&model);
    return rc;
}
