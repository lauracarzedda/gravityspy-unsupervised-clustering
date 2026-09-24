#include "baseline_mini112_pipeline.h"
#include "baseline_standardization.h"
#include "vae_event_dataset.h"

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


enum {
    GS_TEST_PATH_LENGTH = 4096,
    GS_TEST_LINE_LENGTH = 4096,
    GS_TEST_SHA256_LENGTH = 64,
    GS_TEST_EXTRACTION_BLOCK_ROWS = 64,
    GS_TEST_CORE_ARTIFACT_COUNT = 9,
    GS_TEST_TOTAL_FILE_COUNT = 10,
    GS_TEST_INPUT_COUNT = 4
};


#define GS_TEST_STANDARDIZED_TOLERANCE 1.0e-4
#define GS_TEST_SCALER_TOLERANCE 1.0e-15


typedef struct {
    uint32_t state[8];
    uint64_t completed_bits;
    unsigned char block[64];
    size_t block_size;
} GSSha256;


typedef struct {
    size_t constant_feature_count;
    double max_abs_mean;
    double max_abs_population_stddev_error;
} GSStandardizedStatistics;


static const char *const gs_test_core_artifact_names[
    GS_TEST_CORE_ARTIFACT_COUNT
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


static const char *const gs_test_all_artifact_names[
    GS_TEST_TOTAL_FILE_COUNT
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


static const char *const gs_test_input_names[GS_TEST_INPUT_COUNT] = {
    "manifest_events_272.csv",
    "manifest_tensors_112_central.csv",
    "split_train_indices_seed123.csv",
    "split_val_indices_seed123.csv"
};


static int gs_checked_product(
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


static int gs_platform_supports_artifact_contract(void)
{
    uint16_t value = UINT16_C(1);
    const unsigned char *bytes = (const unsigned char *)&value;

    return CHAR_BIT == 8 &&
           sizeof(float) == 4 &&
           FLT_RADIX == 2 &&
           FLT_MANT_DIG == 24 &&
           FLT_MAX_EXP == 128 &&
           bytes[0] == 1u;
}


static const char *gs_path_basename(const char *path)
{
    const char *slash;
    const char *backslash;
    const char *last;

    if (path == NULL) {
        return NULL;
    }

    slash = strrchr(path, '/');
    backslash = strrchr(path, '\\');
    last = slash;

    if (last == NULL || (backslash != NULL && backslash > last)) {
        last = backslash;
    }

    return last == NULL ? path : last + 1;
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
    int needs_separator;

    if (directory == NULL || filename == NULL || path == NULL ||
        path_size == 0) {

        return 1;
    }

    directory_length = strlen(directory);
    filename_length = strlen(filename);
    needs_separator =
        directory_length != 0 &&
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

    if (path == NULL) {
        return -1;
    }

    errno = 0;
    if (stat(path, &information) == 0) {
        return 1;
    }

    return errno == ENOENT ? 0 : -1;
}


static int gs_file_size(const char *path, size_t *size)
{
    struct stat information;

    if (path == NULL || size == NULL ||
        stat(path, &information) != 0 ||
        information.st_size < 0 ||
        (uintmax_t)information.st_size > (uintmax_t)SIZE_MAX) {

        return 1;
    }

    *size = (size_t)information.st_size;
    return 0;
}


static int gs_exact_inventory_is_valid(const char *directory)
{
    unsigned char seen[GS_TEST_TOTAL_FILE_COUNT] = {0};
    DIR *stream;
    struct dirent *entry;
    int kmeans_directory_seen = 0;
    int failed = 0;
    size_t index;

    if (directory == NULL) {
        return 1;
    }

    stream = opendir(directory);
    if (stream == NULL) {
        return 1;
    }

    while (!failed) {
        int matched = 0;
        char path[GS_TEST_PATH_LENGTH];
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

        if (gs_path_join(
                directory,
                entry->d_name,
                path,
                sizeof(path)
            ) != 0 ||
            stat(path, &information) != 0) {

            failed = 1;
            break;
        }

        for (index = 0; index < GS_TEST_TOTAL_FILE_COUNT; index++) {
            if (strcmp(
                    entry->d_name,
                    gs_test_all_artifact_names[index]
                ) == 0) {

                if (seen[index] != 0u ||
                    !S_ISREG(information.st_mode)) {

                    failed = 1;
                }

                seen[index] = 1u;
                matched = 1;
                break;
            }
        }

        if (!matched) {
            if (strcmp(entry->d_name, "kmeans_seed123") == 0 &&
                !kmeans_directory_seen &&
                S_ISDIR(information.st_mode)) {

                kmeans_directory_seen = 1;
            } else {
                failed = 1;
            }
        }
    }

    if (closedir(stream) != 0) {
        failed = 1;
    }

    for (index = 0;
         !failed && index < GS_TEST_TOTAL_FILE_COUNT;
         index++) {

        if (seen[index] == 0u) {
            failed = 1;
        }
    }

    return failed;
}


static int gs_validate_expected_files_and_sizes(
    const char *artifact_directory
)
{
    const size_t expected_sizes[GS_TEST_TOTAL_FILE_COUNT] = {
        0u,
        28123136u,
        7032832u,
        28123136u,
        7032832u,
        0u,
        0u,
        0u,
        0u,
        0u
    };

    size_t index;

    if (gs_exact_inventory_is_valid(artifact_directory) != 0) {
        return 1;
    }

    for (index = 0; index < GS_TEST_TOTAL_FILE_COUNT; index++) {
        char path[GS_TEST_PATH_LENGTH];
        size_t actual_size;

        if (gs_path_join(
                artifact_directory,
                gs_test_all_artifact_names[index],
                path,
                sizeof(path)
            ) != 0 ||
            gs_file_size(path, &actual_size) != 0 ||
            (expected_sizes[index] == 0u && actual_size == 0u) ||
            (expected_sizes[index] != 0u &&
             actual_size != expected_sizes[index])) {

            return 1;
        }
    }

    return 0;
}


static void gs_remove_line_ending(char *line)
{
    size_t length;

    if (line == NULL) {
        return;
    }

    length = strlen(line);
    while (length != 0 &&
           (line[length - 1] == '\r' || line[length - 1] == '\n')) {

        line[--length] = '\0';
    }
}


static int gs_csv_split_line(
    char *line,
    char **fields,
    size_t max_fields
)
{
    char *read_cursor;
    char *write_cursor;
    size_t count = 0;
    int done = 0;

    if (line == NULL || fields == NULL || max_fields == 0) {
        return -1;
    }

    read_cursor = line;
    write_cursor = line;

    while (!done) {
        int delimiter;

        if (count >= max_fields) {
            return -1;
        }

        fields[count++] = write_cursor;

        if (*read_cursor == '"') {
            int closed = 0;
            read_cursor++;

            while (*read_cursor != '\0') {
                if (*read_cursor == '"') {
                    if (read_cursor[1] == '"') {
                        *write_cursor++ = '"';
                        read_cursor += 2;
                    } else {
                        read_cursor++;
                        closed = 1;
                        break;
                    }
                } else if (*read_cursor == '\r' ||
                           *read_cursor == '\n') {

                    return -1;
                } else {
                    *write_cursor++ = *read_cursor++;
                }
            }

            if (!closed) {
                return -1;
            }

            delimiter = (unsigned char)*read_cursor;
            if (delimiter == ',') {
                read_cursor++;
            } else if (delimiter == '\r') {
                read_cursor++;
                if (*read_cursor == '\n') {
                    read_cursor++;
                }
                done = 1;
            } else if (delimiter == '\n') {
                read_cursor++;
                done = 1;
            } else if (delimiter == '\0') {
                done = 1;
            } else {
                return -1;
            }
        } else {
            while (*read_cursor != '\0' &&
                   *read_cursor != ',' &&
                   *read_cursor != '\r' &&
                   *read_cursor != '\n') {

                *write_cursor++ = *read_cursor++;
            }

            delimiter = (unsigned char)*read_cursor;
            if (delimiter == ',') {
                read_cursor++;
            } else if (delimiter == '\r') {
                read_cursor++;
                if (*read_cursor == '\n') {
                    read_cursor++;
                }
                done = 1;
            } else if (delimiter == '\n') {
                read_cursor++;
                done = 1;
            } else {
                done = 1;
            }
        }

        *write_cursor++ = '\0';

        if (done && *read_cursor != '\0') {
            return -1;
        }
    }

    return (int)count;
}


static int gs_parse_size_value(const char *text, size_t *value)
{
    char *endptr;
    unsigned long parsed;

    if (text == NULL || value == NULL ||
        text[0] == '\0' || text[0] == '-') {

        return 1;
    }

    errno = 0;
    endptr = NULL;
    parsed = strtoul(text, &endptr, 10);

    if (errno == ERANGE || endptr == text || *endptr != '\0') {

        return 1;
    }

#if ULONG_MAX > SIZE_MAX
    if (parsed > (unsigned long)SIZE_MAX) {
        return 1;
    }
#endif

    *value = (size_t)parsed;
    return 0;
}


static int gs_parse_int_value(const char *text, int *value)
{
    char *endptr;
    long parsed;

    if (text == NULL || value == NULL || text[0] == '\0') {
        return 1;
    }

    errno = 0;
    endptr = NULL;
    parsed = strtol(text, &endptr, 10);

    if (errno == ERANGE || endptr == text || *endptr != '\0' ||
        parsed < INT_MIN || parsed > INT_MAX) {

        return 1;
    }

    *value = (int)parsed;
    return 0;
}


static int gs_parse_double_value(const char *text, double *value)
{
    char *endptr;
    double parsed;

    if (text == NULL || value == NULL || text[0] == '\0') {
        return 1;
    }

    errno = 0;
    endptr = NULL;
    parsed = strtod(text, &endptr);

    if (errno == ERANGE || endptr == text || *endptr != '\0' ||
        !isfinite(parsed)) {

        return 1;
    }

    *value = parsed;
    return 0;
}


static int gs_header_has_label_column(const char *line)
{
    char copy[GS_TEST_LINE_LENGTH];
    char *fields[32];
    int field_count;
    int index;
    size_t length;

    if (line == NULL) {
        return 1;
    }

    length = strlen(line);
    if (length >= sizeof(copy)) {
        return 1;
    }

    memcpy(copy, line, length + 1u);
    field_count = gs_csv_split_line(copy, fields, 32);

    if (field_count <= 0) {
        return 1;
    }

    for (index = 0; index < field_count; index++) {
        if (strcmp(fields[index], "label") == 0) {
            return 1;
        }
    }

    return 0;
}


static int gs_validate_rows_csv(
    const char *artifact_directory,
    const char *filename,
    const char *expected_split_name,
    const GSEventManifest *events,
    const GSBaselineMini112Pipeline *pipeline,
    size_t row_count,
    int validation_split
)
{
    const char *expected_header =
        "row_index,split_pos,manifest_index,event_index,event_id,split";

    char path[GS_TEST_PATH_LENGTH];
    char line[GS_TEST_LINE_LENGTH];
    FILE *file;
    size_t row;
    int failed = 0;

    if (events == NULL || pipeline == NULL ||
        gs_path_join(
            artifact_directory,
            filename,
            path,
            sizeof(path)
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
        gs_remove_line_ending(line);
        if (strcmp(line, expected_header) != 0 ||
            gs_header_has_label_column(line) != 0) {

            failed = 1;
        }
    }

    for (row = 0; !failed && row < row_count; row++) {
        char *fields[7];
        size_t row_index;
        size_t split_pos;
        size_t manifest_index;
        size_t expected_manifest_index;
        int event_index;
        int field_count;
        int getter_result;
        const GSEventRecord *event_record;

        if (fgets(line, sizeof(line), file) == NULL) {
            failed = 1;
            break;
        }

        field_count = gs_csv_split_line(line, fields, 7);
        getter_result = validation_split ?
            gs_baseline_mini112_validation_manifest_index(
                pipeline,
                row,
                &expected_manifest_index
            ) :
            gs_baseline_mini112_train_manifest_index(
                pipeline,
                row,
                &expected_manifest_index
            );

        if (field_count != 6 ||
            getter_result != GS_BASELINE_MINI112_PIPELINE_SUCCESS ||
            gs_parse_size_value(fields[0], &row_index) != 0 ||
            gs_parse_size_value(fields[1], &split_pos) != 0 ||
            gs_parse_size_value(fields[2], &manifest_index) != 0 ||
            gs_parse_int_value(fields[3], &event_index) != 0 ||
            row_index != row ||
            split_pos != row ||
            manifest_index != expected_manifest_index) {

            failed = 1;
            break;
        }

        event_record = gs_event_manifest_get(events, manifest_index);
        if (event_record == NULL ||
            event_index != event_record->event_index ||
            strcmp(fields[4], event_record->event_id) != 0 ||
            strcmp(fields[5], expected_split_name) != 0) {

            failed = 1;
        }
    }

    if (!failed && fgets(line, sizeof(line), file) != NULL) {
        failed = 1;
    }

    if (ferror(file)) {
        failed = 1;
    }

    if (fclose(file) != 0) {
        failed = 1;
    }

    return failed;
}


static int gs_validate_scaler_csv(
    const char *artifact_directory,
    double *mean,
    double *stddev,
    unsigned char *constant_mask
)
{
    const char *expected_header =
        "feature_index,channel,pooled_row,pooled_col,mean,stddev,constant";

    char path[GS_TEST_PATH_LENGTH];
    char line[GS_TEST_LINE_LENGTH];
    FILE *file;
    size_t feature;
    int failed = 0;

    if (mean == NULL || stddev == NULL || constant_mask == NULL ||
        gs_path_join(
            artifact_directory,
            "scaler.csv",
            path,
            sizeof(path)
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
        gs_remove_line_ending(line);
        if (strcmp(line, expected_header) != 0 ||
            gs_header_has_label_column(line) != 0) {

            failed = 1;
        }
    }

    for (feature = 0;
         !failed && feature < GS_BASELINE_MINI112_FEATURE_COUNT;
         feature++) {

        char *fields[8];
        size_t parsed_feature;
        size_t channel;
        size_t pooled_row;
        size_t pooled_col;
        size_t constant;
        size_t values_per_channel =
            (size_t)GS_BASELINE_MINI112_POOLED_SIDE *
            (size_t)GS_BASELINE_MINI112_POOLED_SIDE;

        size_t expected_channel = feature / values_per_channel;
        size_t remainder = feature % values_per_channel;
        int field_count;

        if (fgets(line, sizeof(line), file) == NULL) {
            failed = 1;
            break;
        }

        field_count = gs_csv_split_line(line, fields, 8);
        if (field_count != 7 ||
            gs_parse_size_value(fields[0], &parsed_feature) != 0 ||
            gs_parse_size_value(fields[1], &channel) != 0 ||
            gs_parse_size_value(fields[2], &pooled_row) != 0 ||
            gs_parse_size_value(fields[3], &pooled_col) != 0 ||
            gs_parse_double_value(fields[4], &mean[feature]) != 0 ||
            gs_parse_double_value(fields[5], &stddev[feature]) != 0 ||
            gs_parse_size_value(fields[6], &constant) != 0 ||
            parsed_feature != feature ||
            channel != expected_channel ||
            pooled_row != remainder / GS_BASELINE_MINI112_POOLED_SIDE ||
            pooled_col != remainder % GS_BASELINE_MINI112_POOLED_SIDE ||
            stddev[feature] < 0.0 ||
            constant > 1u ||
            (constant != 0u &&
             stddev[feature] >
                GS_BASELINE_STANDARDIZATION_DEFAULT_CONSTANT_EPSILON) ||
            (constant == 0u && stddev[feature] <= 0.0)) {

            failed = 1;
            break;
        }

        constant_mask[feature] = (unsigned char)constant;
    }

    if (!failed && fgets(line, sizeof(line), file) != NULL) {
        failed = 1;
    }

    if (ferror(file)) {
        failed = 1;
    }

    if (fclose(file) != 0) {
        failed = 1;
    }

    return failed;
}


static int gs_read_binary_matrix(
    const char *artifact_directory,
    const char *filename,
    float *values,
    size_t value_count
)
{
    char path[GS_TEST_PATH_LENGTH];
    FILE *file;
    int failed = 0;

    if (values == NULL ||
        gs_path_join(
            artifact_directory,
            filename,
            path,
            sizeof(path)
        ) != 0) {

        return 1;
    }

    file = fopen(path, "rb");
    if (file == NULL) {
        return 1;
    }

    if (fread(values, sizeof(float), value_count, file) != value_count ||
        fgetc(file) != EOF ||
        ferror(file)) {

        failed = 1;
    }

    if (fclose(file) != 0) {
        failed = 1;
    }

    return failed;
}


static int gs_validate_raw_values(
    const float *values,
    size_t value_count
)
{
    size_t index;

    if (values == NULL || value_count == 0) {
        return 1;
    }

    for (index = 0; index < value_count; index++) {
        if (!isfinite(values[index]) ||
            values[index] < 0.0f ||
            values[index] > 1.0f) {

            return 1;
        }
    }

    return 0;
}


static int gs_read_text_file(
    const char *artifact_directory,
    const char *filename,
    char **text
)
{
    char path[GS_TEST_PATH_LENGTH];
    size_t size;
    FILE *file;
    char *buffer;
    int failed = 0;

    if (text == NULL ||
        gs_path_join(
            artifact_directory,
            filename,
            path,
            sizeof(path)
        ) != 0 ||
        gs_file_size(path, &size) != 0 ||
        size == 0 ||
        size == SIZE_MAX) {

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

    if (fread(buffer, 1, size, file) != size ||
        fgetc(file) != EOF ||
        ferror(file) ||
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
    const char *cursor;
    size_t line_length;

    if (text == NULL || line == NULL) {
        return 0;
    }

    line_length = strlen(line);
    cursor = text;

    while ((cursor = strstr(cursor, line)) != NULL) {
        const char *after = cursor + line_length;
        int starts_line = cursor == text || cursor[-1] == '\n';
        int ends_line =
            *after == '\0' || *after == '\n' || *after == '\r';

        if (starts_line && ends_line) {
            return 1;
        }

        cursor++;
    }

    return 0;
}


static int gs_make_key_value_line(
    const char *key,
    const char *value,
    char *line,
    size_t line_size
)
{
    size_t key_length;
    size_t value_length;

    if (key == NULL || value == NULL ||
        line == NULL || line_size == 0) {

        return 1;
    }

    key_length = strlen(key);
    value_length = strlen(value);

    if (key_length > SIZE_MAX - value_length ||
        key_length + value_length >= line_size) {

        return 1;
    }

    memcpy(line, key, key_length);
    memcpy(line + key_length, value, value_length);
    line[key_length + value_length] = '\0';
    return 0;
}


static int gs_validate_metadata_contract(
    const char *artifact_directory,
    const char *manifest_events_path,
    const char *tensor_manifest_path,
    const char *train_split_path,
    const char *validation_split_path,
    const char *data_root,
    const char *code_commit
)
{
    const char *feature_lines[] = {
        "format_version=1",
        "profile=mini112_pool7_no_pca",
        "source=central_deterministic_tensors",
        "input_shape=4x112x112",
        "input_layout=channel_major",
        "channels=4",
        "image_side=112",
        "channel_0=0.5s",
        "channel_1=1.0s",
        "channel_2=2.0s",
        "channel_3=4.0s",
        "normalization=uint8_div_255",
        "normalization_before_pooling=uint8/255.0",
        "feature_extractor=average_pool_7x7_non_overlapping",
        "pooling=average",
        "pool_side=7",
        "stride=7",
        "padding=none",
        "pooled_shape=4x16x16",
        "pooled_side=16",
        "feature_count=1024",
        "feature_order=channel_pooled_row_pooled_col",
        "matrix_layout=row_major",
        "matrix_dtype=float32_ieee754",
        "matrix_endian=little",
        "standardization_fit=train_only",
        "scaler_fit=train_only",
        "standardization_variance=population",
        "standardization_ddof=0",
        "constant_epsilon=1e-12",
        "standardization_epsilon=1e-12",
        "constant_output=0",
        "pca=none",
        "augmentation=none",
        "labels_used=false",
        "split_seed=123",
        "train_rows=6866",
        "validation_rows=1717",
        "train_count=6866",
        "validation_count=1717",
        "extraction_block_rows=64"
    };

    const char *summary_lines[] = {
        "format_version=1",
        "profile=mini112_pool7_no_pca",
        "event_count=8583",
        "train_rows=6866",
        "validation_rows=1717",
        "feature_count=1024",
        "extraction_block_rows=64",
        "train_features_raw_rows=6866",
        "train_features_raw_cols=1024",
        "train_features_raw_bytes=28123136",
        "val_features_raw_rows=1717",
        "val_features_raw_cols=1024",
        "val_features_raw_bytes=7032832",
        "train_features_std_rows=6866",
        "train_features_std_cols=1024",
        "train_features_std_bytes=28123136",
        "val_features_std_rows=1717",
        "val_features_std_cols=1024",
        "val_features_std_bytes=7032832",
        "raw_values_in_unit_interval=true",
        "standardized_values_finite=true",
        "standardization_variance=population",
        "standardization_ddof=0",
        "constant_epsilon=1e-12",
        "standardization_fit=train_only",
        "validation_transformed_without_refit=true",
        "scaler_unchanged_after_validation=true",
        "labels_used=false",
        "pca_used=false",
        "augmentation_used=false",
        "kmeans_used=false"
    };

    char *feature_spec = NULL;
    char *summary = NULL;
    char dynamic_line[GS_TEST_PATH_LENGTH];
    size_t index;
    int failed = 0;

    if (gs_read_text_file(
            artifact_directory,
            "feature_spec.txt",
            &feature_spec
        ) != 0 ||
        gs_read_text_file(
            artifact_directory,
            "summary.txt",
            &summary
        ) != 0) {

        failed = 1;
        goto cleanup;
    }

    for (index = 0;
         index < sizeof(feature_lines) / sizeof(feature_lines[0]);
         index++) {

        if (!gs_text_has_line(feature_spec, feature_lines[index])) {
            failed = 1;
        }
    }

    for (index = 0;
         index < sizeof(summary_lines) / sizeof(summary_lines[0]);
         index++) {

        if (!gs_text_has_line(summary, summary_lines[index])) {
            failed = 1;
        }
    }

    if (gs_make_key_value_line(
            "manifest_events_path=",
            manifest_events_path,
            dynamic_line,
            sizeof(dynamic_line)
        ) != 0 ||
        !gs_text_has_line(summary, dynamic_line) ||
        gs_make_key_value_line(
            "tensor_manifest_path=",
            tensor_manifest_path,
            dynamic_line,
            sizeof(dynamic_line)
        ) != 0 ||
        !gs_text_has_line(summary, dynamic_line) ||
        gs_make_key_value_line(
            "train_split_path=",
            train_split_path,
            dynamic_line,
            sizeof(dynamic_line)
        ) != 0 ||
        !gs_text_has_line(summary, dynamic_line) ||
        gs_make_key_value_line(
            "validation_split_path=",
            validation_split_path,
            dynamic_line,
            sizeof(dynamic_line)
        ) != 0 ||
        !gs_text_has_line(summary, dynamic_line) ||
        gs_make_key_value_line(
            "data_root=",
            data_root,
            dynamic_line,
            sizeof(dynamic_line)
        ) != 0 ||
        !gs_text_has_line(summary, dynamic_line) ||
        gs_make_key_value_line(
            "code_commit=",
            code_commit,
            dynamic_line,
            sizeof(dynamic_line)
        ) != 0 ||
        !gs_text_has_line(feature_spec, dynamic_line) ||
        !gs_text_has_line(summary, dynamic_line)) {

        failed = 1;
    }

cleanup:
    free(summary);
    free(feature_spec);
    return failed;
}


static uint32_t gs_sha256_rotate_right(uint32_t value, unsigned int count)
{
    return (value >> count) | (value << (32u - count));
}


static void gs_sha256_transform(
    GSSha256 *context,
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
            gs_sha256_rotate_right(words[index - 15u], 7u) ^
            gs_sha256_rotate_right(words[index - 15u], 18u) ^
            (words[index - 15u] >> 3);

        uint32_t s1 =
            gs_sha256_rotate_right(words[index - 2u], 17u) ^
            gs_sha256_rotate_right(words[index - 2u], 19u) ^
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
            gs_sha256_rotate_right(e, 6u) ^
            gs_sha256_rotate_right(e, 11u) ^
            gs_sha256_rotate_right(e, 25u);

        uint32_t choice = (e & f) ^ ((~e) & g);
        uint32_t temporary1 =
            h + sum1 + choice + constants[index] + words[index];

        uint32_t sum0 =
            gs_sha256_rotate_right(a, 2u) ^
            gs_sha256_rotate_right(a, 13u) ^
            gs_sha256_rotate_right(a, 22u);

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


static void gs_sha256_init(GSSha256 *context)
{
    context->state[0] = UINT32_C(0x6a09e667);
    context->state[1] = UINT32_C(0xbb67ae85);
    context->state[2] = UINT32_C(0x3c6ef372);
    context->state[3] = UINT32_C(0xa54ff53a);
    context->state[4] = UINT32_C(0x510e527f);
    context->state[5] = UINT32_C(0x9b05688c);
    context->state[6] = UINT32_C(0x1f83d9ab);
    context->state[7] = UINT32_C(0x5be0cd19);
    context->completed_bits = 0;
    context->block_size = 0;
}


static int gs_sha256_update(
    GSSha256 *context,
    const unsigned char *data,
    size_t size
)
{
    size_t index;

    if (context == NULL || (data == NULL && size != 0u)) {
        return 1;
    }

    for (index = 0; index < size; index++) {
        context->block[context->block_size++] = data[index];

        if (context->block_size == 64u) {
            if (context->completed_bits > UINT64_MAX - UINT64_C(512)) {
                return 1;
            }

            gs_sha256_transform(context, context->block);
            context->completed_bits += UINT64_C(512);
            context->block_size = 0;
        }
    }

    return 0;
}


static int gs_sha256_final(
    GSSha256 *context,
    unsigned char digest[32]
)
{
    uint64_t total_bits;
    size_t index;

    if (context == NULL || digest == NULL ||
        context->block_size > (UINT64_MAX - context->completed_bits) / 8u) {

        return 1;
    }

    total_bits =
        context->completed_bits + (uint64_t)context->block_size * 8u;

    context->block[context->block_size++] = 0x80u;

    if (context->block_size > 56u) {
        while (context->block_size < 64u) {
            context->block[context->block_size++] = 0u;
        }

        gs_sha256_transform(context, context->block);
        context->block_size = 0;
    }

    while (context->block_size < 56u) {
        context->block[context->block_size++] = 0u;
    }

    for (index = 0; index < 8u; index++) {
        context->block[63u - index] =
            (unsigned char)(total_bits >> (index * 8u));
    }

    gs_sha256_transform(context, context->block);

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

    return 0;
}


static int gs_sha256_file(
    const char *path,
    char output[GS_TEST_SHA256_LENGTH + 1]
)
{
    static const char digits[] = "0123456789abcdef";
    unsigned char buffer[65536];
    unsigned char digest[32];
    GSSha256 context;
    FILE *file;
    size_t count;
    size_t index;
    int failed = 0;

    if (path == NULL || output == NULL) {
        return 1;
    }

    file = fopen(path, "rb");
    if (file == NULL) {
        return 1;
    }

    gs_sha256_init(&context);

    while ((count = fread(buffer, 1, sizeof(buffer), file)) != 0u) {
        if (gs_sha256_update(&context, buffer, count) != 0) {
            failed = 1;
            break;
        }
    }

    if (ferror(file)) {
        failed = 1;
    }

    if (fclose(file) != 0) {
        failed = 1;
    }

    if (failed || gs_sha256_final(&context, digest) != 0) {
        return 1;
    }

    for (index = 0; index < sizeof(digest); index++) {
        output[index * 2u] = digits[digest[index] >> 4];
        output[index * 2u + 1u] = digits[digest[index] & 0x0fu];
    }

    output[GS_TEST_SHA256_LENGTH] = '\0';
    return 0;
}


static int gs_verify_sha256_manifest(
    const char *artifact_directory,
    const char *const input_paths[GS_TEST_INPUT_COUNT]
)
{
    char sha_path[GS_TEST_PATH_LENGTH];
    char line[GS_TEST_LINE_LENGTH];
    char expected[GS_TEST_LINE_LENGTH];
    FILE *file;
    size_t index;
    int failed = 0;

    if (gs_path_join(
            artifact_directory,
            "sha256.txt",
            sha_path,
            sizeof(sha_path)
        ) != 0) {

        return 1;
    }

    file = fopen(sha_path, "rb");
    if (file == NULL) {
        return 1;
    }

    for (index = 0; !failed && index < GS_TEST_INPUT_COUNT; index++) {
        char hash[GS_TEST_SHA256_LENGTH + 1];
        int written;

        if (strcmp(
                gs_path_basename(input_paths[index]),
                gs_test_input_names[index]
            ) != 0 ||
            gs_sha256_file(input_paths[index], hash) != 0 ||
            (written = snprintf(
                expected,
                sizeof(expected),
                "INPUT %s %s",
                hash,
                gs_test_input_names[index]
            )) < 0 ||
            (size_t)written >= sizeof(expected) ||
            fgets(line, sizeof(line), file) == NULL) {

            failed = 1;
            break;
        }

        gs_remove_line_ending(line);
        if (strcmp(line, expected) != 0) {
            failed = 1;
        }
    }

    for (index = 0;
         !failed && index < GS_TEST_CORE_ARTIFACT_COUNT;
         index++) {

        char artifact_path[GS_TEST_PATH_LENGTH];
        char hash[GS_TEST_SHA256_LENGTH + 1];
        int written;

        if (gs_path_join(
                artifact_directory,
                gs_test_core_artifact_names[index],
                artifact_path,
                sizeof(artifact_path)
            ) != 0 ||
            gs_sha256_file(artifact_path, hash) != 0 ||
            (written = snprintf(
                expected,
                sizeof(expected),
                "ARTIFACT %s %s",
                hash,
                gs_test_core_artifact_names[index]
            )) < 0 ||
            (size_t)written >= sizeof(expected) ||
            fgets(line, sizeof(line), file) == NULL) {

            failed = 1;
            break;
        }

        gs_remove_line_ending(line);
        if (strcmp(line, expected) != 0) {
            failed = 1;
        }
    }

    if (!failed && fgets(line, sizeof(line), file) != NULL) {
        failed = 1;
    }

    if (ferror(file)) {
        failed = 1;
    }

    if (fclose(file) != 0) {
        failed = 1;
    }

    return failed;
}


static int gs_reconstruct_raw_split(
    GSBaselineMini112Pipeline *pipeline,
    int validation_split,
    const float *artifact_values,
    size_t row_count,
    float *block,
    double *maximum_difference
)
{
    size_t first_row;

    if (pipeline == NULL || artifact_values == NULL ||
        block == NULL || maximum_difference == NULL ||
        row_count > SIZE_MAX / GS_BASELINE_MINI112_FEATURE_COUNT) {

        return 1;
    }

    for (first_row = 0; first_row < row_count;) {
        size_t remaining = row_count - first_row;
        size_t block_rows =
            remaining < GS_TEST_EXTRACTION_BLOCK_ROWS ?
                remaining :
                GS_TEST_EXTRACTION_BLOCK_ROWS;

        size_t block_values;
        size_t block_bytes;
        size_t row_offset;
        size_t index;
        int extract_result;
        const float *expected;

        if (gs_checked_product(
                block_rows,
                GS_BASELINE_MINI112_FEATURE_COUNT,
                &block_values
            ) != 0 ||
            gs_checked_product(
                first_row,
                GS_BASELINE_MINI112_FEATURE_COUNT,
                &row_offset
            ) != 0 ||
            row_offset > SIZE_MAX - block_values ||
            gs_checked_product(
                block_values,
                sizeof(float),
                &block_bytes
            ) != 0) {

            return 1;
        }

        expected = artifact_values + row_offset;

        extract_result = validation_split ?
            gs_baseline_mini112_extract_validation_rows(
                pipeline,
                first_row,
                block_rows,
                block,
                block_values
            ) :
            gs_baseline_mini112_extract_train_rows(
                pipeline,
                first_row,
                block_rows,
                block,
                block_values
            );

        if (extract_result != GS_BASELINE_MINI112_PIPELINE_SUCCESS) {
            return 1;
        }

        for (index = 0; index < block_values; index++) {
            double difference =
                fabs((double)block[index] - (double)expected[index]);

            if (difference > *maximum_difference) {
                *maximum_difference = difference;
            }
        }

        if (memcmp(
                block,
                expected,
                block_bytes
            ) != 0) {

            return 1;
        }

        first_row += block_rows;
    }

    return 0;
}


static int gs_validate_recomputed_scaler(
    const float *train_raw,
    size_t train_rows,
    const double *stored_mean,
    const double *stored_stddev,
    const unsigned char *stored_mask,
    double *recomputed_mean,
    double *recomputed_stddev,
    unsigned char *recomputed_mask,
    double *maximum_difference,
    size_t *constant_feature_count
)
{
    size_t feature;

    if (train_raw == NULL ||
        stored_mean == NULL || stored_stddev == NULL ||
        stored_mask == NULL ||
        recomputed_mean == NULL || recomputed_stddev == NULL ||
        recomputed_mask == NULL ||
        maximum_difference == NULL ||
        constant_feature_count == NULL) {

        return 1;
    }

    if (gs_baseline_standardization_fit(
            train_raw,
            train_rows,
            GS_BASELINE_MINI112_FEATURE_COUNT,
            recomputed_mean,
            recomputed_stddev,
            recomputed_mask,
            GS_BASELINE_STANDARDIZATION_DEFAULT_CONSTANT_EPSILON
        ) != GS_BASELINE_STANDARDIZATION_SUCCESS) {

        return 1;
    }

    *maximum_difference = 0.0;
    *constant_feature_count = 0;

    for (feature = 0;
         feature < GS_BASELINE_MINI112_FEATURE_COUNT;
         feature++) {

        double mean_difference =
            fabs(recomputed_mean[feature] - stored_mean[feature]);

        double stddev_difference =
            fabs(recomputed_stddev[feature] - stored_stddev[feature]);

        if (mean_difference > *maximum_difference) {
            *maximum_difference = mean_difference;
        }

        if (stddev_difference > *maximum_difference) {
            *maximum_difference = stddev_difference;
        }

        if (stored_mask[feature] != recomputed_mask[feature] ||
            mean_difference > GS_TEST_SCALER_TOLERANCE ||
            stddev_difference > GS_TEST_SCALER_TOLERANCE) {

            return 1;
        }

        if (stored_mask[feature] != 0u) {
            (*constant_feature_count)++;
        }
    }

    return 0;
}


static int gs_validate_standardized_split(
    const float *raw,
    const float *stored_standardized,
    size_t row_count,
    const double *mean,
    const double *stddev,
    const unsigned char *constant_mask,
    float *block,
    double *maximum_difference
)
{
    size_t first_row;

    if (raw == NULL || stored_standardized == NULL ||
        mean == NULL || stddev == NULL ||
        constant_mask == NULL || block == NULL ||
        maximum_difference == NULL ||
        row_count > SIZE_MAX / GS_BASELINE_MINI112_FEATURE_COUNT) {

        return 1;
    }

    for (first_row = 0; first_row < row_count;) {
        size_t remaining = row_count - first_row;
        size_t block_rows =
            remaining < GS_TEST_EXTRACTION_BLOCK_ROWS ?
                remaining :
                GS_TEST_EXTRACTION_BLOCK_ROWS;

        size_t block_values;
        size_t block_bytes;
        size_t row_offset;
        size_t index;
        const float *raw_block;
        const float *expected;

        if (gs_checked_product(
                block_rows,
                GS_BASELINE_MINI112_FEATURE_COUNT,
                &block_values
            ) != 0 ||
            gs_checked_product(
                first_row,
                GS_BASELINE_MINI112_FEATURE_COUNT,
                &row_offset
            ) != 0 ||
            row_offset > SIZE_MAX - block_values ||
            gs_checked_product(
                block_values,
                sizeof(float),
                &block_bytes
            ) != 0) {

            return 1;
        }

        raw_block = raw + row_offset;
        expected = stored_standardized + row_offset;

        if (gs_baseline_standardization_transform(
                raw_block,
                block_rows,
                GS_BASELINE_MINI112_FEATURE_COUNT,
                mean,
                stddev,
                constant_mask,
                block
            ) != GS_BASELINE_STANDARDIZATION_SUCCESS) {

            return 1;
        }

        for (index = 0; index < block_values; index++) {
            double difference =
                fabs((double)block[index] - (double)expected[index]);

            if (!isfinite(expected[index])) {
                return 1;
            }

            if (difference > *maximum_difference) {
                *maximum_difference = difference;
            }
        }

        if (memcmp(
                block,
                expected,
                block_bytes
            ) != 0) {

            return 1;
        }

        first_row += block_rows;
    }

    return 0;
}


static int gs_compute_standardized_train_statistics(
    const float *train_standardized,
    size_t train_rows,
    const unsigned char *constant_mask,
    GSStandardizedStatistics *statistics
)
{
    size_t feature;

    if (train_standardized == NULL ||
        constant_mask == NULL ||
        statistics == NULL ||
        train_rows > SIZE_MAX / GS_BASELINE_MINI112_FEATURE_COUNT) {

        return 1;
    }

    statistics->constant_feature_count = 0;
    statistics->max_abs_mean = 0.0;
    statistics->max_abs_population_stddev_error = 0.0;

    for (feature = 0;
         feature < GS_BASELINE_MINI112_FEATURE_COUNT;
         feature++) {

        size_t row;

        if (constant_mask[feature] != 0u) {
            statistics->constant_feature_count++;

            for (row = 0; row < train_rows; row++) {
                float value = train_standardized[
                    row * GS_BASELINE_MINI112_FEATURE_COUNT + feature
                ];

                if (!isfinite(value) || value != 0.0f) {
                    return 1;
                }
            }
        } else {
            double mean = 0.0;
            double variance = 0.0;
            double mean_error;
            double stddev_error;

            for (row = 0; row < train_rows; row++) {
                float value = train_standardized[
                    row * GS_BASELINE_MINI112_FEATURE_COUNT + feature
                ];

                if (!isfinite(value)) {
                    return 1;
                }

                mean += (double)value;
            }

            mean /= (double)train_rows;

            for (row = 0; row < train_rows; row++) {
                double difference =
                    (double)train_standardized[
                        row * GS_BASELINE_MINI112_FEATURE_COUNT + feature
                    ] - mean;

                variance += difference * difference;
            }

            variance /= (double)train_rows;
            mean_error = fabs(mean);
            stddev_error = fabs(sqrt(variance) - 1.0);

            if (mean_error > GS_TEST_STANDARDIZED_TOLERANCE ||
                stddev_error > GS_TEST_STANDARDIZED_TOLERANCE) {

                return 1;
            }

            if (mean_error > statistics->max_abs_mean) {
                statistics->max_abs_mean = mean_error;
            }

            if (stddev_error >
                statistics->max_abs_population_stddev_error) {

                statistics->max_abs_population_stddev_error =
                    stddev_error;
            }
        }
    }

    return 0;
}


int main(int argc, char **argv)
{
    GSBaselineMini112Pipeline pipeline = {0};
    GSEventManifest events = {0};
    GSStandardizedStatistics standardized_statistics = {0u, 0.0, 0.0};

    float *train_raw = NULL;
    float *validation_raw = NULL;
    float *train_standardized = NULL;
    float *validation_standardized = NULL;
    float *block = NULL;
    double *mean = NULL;
    double *stddev = NULL;
    double *recomputed_mean = NULL;
    double *recomputed_stddev = NULL;
    double *mean_copy = NULL;
    double *stddev_copy = NULL;
    unsigned char *constant_mask = NULL;
    unsigned char *recomputed_mask = NULL;
    unsigned char *constant_mask_copy = NULL;

    const char *input_paths[GS_TEST_INPUT_COUNT];
    size_t event_count;
    size_t train_count;
    size_t validation_count;
    size_t train_value_count;
    size_t validation_value_count;
    size_t train_matrix_bytes;
    size_t validation_matrix_bytes;
    size_t scaler_double_bytes;
    size_t scaler_mask_bytes;
    size_t block_value_count;
    size_t block_bytes;
    size_t constant_feature_count = 0;
    double raw_maximum_difference = 0.0;
    double scaler_maximum_difference = 0.0;
    double standardized_maximum_difference = 0.0;
    char partial_path[GS_TEST_PATH_LENGTH];
    int written;
    int rc = 1;

    if (argc != 8) {
        fprintf(
            stderr,
            "Uso: test_baseline_mini112_artifacts.exe manifest_eventi "
            "manifest_mini112 split_train split_validation data_root "
            "directory_artefatti commit_git\n"
        );
        return 1;
    }

    if (!gs_platform_supports_artifact_contract()) {
        fprintf(
            stderr,
            "Errore: piattaforma float32 little-endian non valida.\n"
        );
        return 1;
    }

    input_paths[0] = argv[1];
    input_paths[1] = argv[2];
    input_paths[2] = argv[3];
    input_paths[3] = argv[4];

    if (strcmp(gs_path_basename(argv[1]), gs_test_input_names[0]) != 0 ||
        strcmp(gs_path_basename(argv[2]), gs_test_input_names[1]) != 0 ||
        strcmp(gs_path_basename(argv[3]), gs_test_input_names[2]) != 0 ||
        strcmp(gs_path_basename(argv[4]), gs_test_input_names[3]) != 0) {

        fprintf(stderr, "Errore: nomi degli input non ufficiali.\n");
        return 1;
    }

    written = snprintf(
        partial_path,
        sizeof(partial_path),
        "%s.partial",
        argv[6]
    );

    if (written < 0 ||
        (size_t)written >= sizeof(partial_path) ||
        gs_path_state(argv[6]) != 1 ||
        gs_path_state(partial_path) != 0) {

        fprintf(stderr, "Errore: stato directory finale/partial non valido.\n");
        return 1;
    }

    if (gs_checked_product(
            GS_BASELINE_MINI112_OFFICIAL_TRAIN_COUNT,
            GS_BASELINE_MINI112_FEATURE_COUNT,
            &train_value_count
        ) != 0 ||
        gs_checked_product(
            GS_BASELINE_MINI112_OFFICIAL_VALIDATION_COUNT,
            GS_BASELINE_MINI112_FEATURE_COUNT,
            &validation_value_count
        ) != 0 ||
        gs_checked_product(
            train_value_count,
            sizeof(float),
            &train_matrix_bytes
        ) != 0 ||
        gs_checked_product(
            validation_value_count,
            sizeof(float),
            &validation_matrix_bytes
        ) != 0 ||
        gs_checked_product(
            GS_BASELINE_MINI112_FEATURE_COUNT,
            sizeof(double),
            &scaler_double_bytes
        ) != 0 ||
        gs_checked_product(
            GS_BASELINE_MINI112_FEATURE_COUNT,
            sizeof(unsigned char),
            &scaler_mask_bytes
        ) != 0 ||
        gs_checked_product(
            GS_TEST_EXTRACTION_BLOCK_ROWS,
            GS_BASELINE_MINI112_FEATURE_COUNT,
            &block_value_count
        ) != 0 ||
        gs_checked_product(
            block_value_count,
            sizeof(float),
            &block_bytes
        ) != 0 ||
        train_matrix_bytes != 28123136u ||
        validation_matrix_bytes != 7032832u) {

        fprintf(stderr, "Errore: overflow o dimensioni non valide.\n");
        return 1;
    }

    gs_baseline_mini112_pipeline_init(&pipeline);

    {
        int open_result = gs_baseline_mini112_pipeline_open(
            &pipeline,
            argv[1],
            argv[2],
            argv[3],
            argv[4],
            argv[5]
        );

        if (open_result != GS_BASELINE_MINI112_PIPELINE_SUCCESS) {
            fprintf(
                stderr,
                "Errore pipeline: %s\n",
                gs_baseline_mini112_pipeline_error_string(open_result)
            );
            goto cleanup;
        }
    }

    if (gs_baseline_mini112_pipeline_counts(
            &pipeline,
            &event_count,
            &train_count,
            &validation_count
        ) != GS_BASELINE_MINI112_PIPELINE_SUCCESS ||
        event_count != GS_BASELINE_MINI112_OFFICIAL_EVENT_COUNT ||
        train_count != GS_BASELINE_MINI112_OFFICIAL_TRAIN_COUNT ||
        validation_count !=
            GS_BASELINE_MINI112_OFFICIAL_VALIDATION_COUNT ||
        gs_event_manifest_load(argv[1], &events) != 0 ||
        events.count != event_count) {

        fprintf(stderr, "Errore: input ufficiali non validi.\n");
        goto cleanup;
    }

    if (gs_validate_expected_files_and_sizes(argv[6]) != 0) {
        fprintf(stderr, "Errore: inventario o dimensioni non validi.\n");
        goto cleanup;
    }
    printf("[PASS] exact_inventory_and_sizes\n");

    if (gs_validate_metadata_contract(
            argv[6],
            argv[1],
            argv[2],
            argv[3],
            argv[4],
            argv[5],
            argv[7]
        ) != 0) {

        fprintf(stderr, "Errore: contratto metadata non valido.\n");
        goto cleanup;
    }
    printf("[PASS] metadata_contract\n");

    if (gs_validate_rows_csv(
            argv[6],
            "train_rows.csv",
            "train",
            &events,
            &pipeline,
            train_count,
            0
        ) != 0 ||
        gs_validate_rows_csv(
            argv[6],
            "val_rows.csv",
            "validation",
            &events,
            &pipeline,
            validation_count,
            1
        ) != 0) {

        fprintf(stderr, "Errore: mapping righe non valido.\n");
        goto cleanup;
    }
    printf("[PASS] complete_row_mapping_without_labels\n");

    train_raw = (float *)malloc(train_matrix_bytes);
    validation_raw = (float *)malloc(validation_matrix_bytes);
    train_standardized = (float *)malloc(train_matrix_bytes);
    validation_standardized = (float *)malloc(validation_matrix_bytes);
    block = (float *)malloc(block_bytes);
    mean = (double *)malloc(scaler_double_bytes);
    stddev = (double *)malloc(scaler_double_bytes);
    recomputed_mean = (double *)malloc(scaler_double_bytes);
    recomputed_stddev = (double *)malloc(scaler_double_bytes);
    mean_copy = (double *)malloc(scaler_double_bytes);
    stddev_copy = (double *)malloc(scaler_double_bytes);
    constant_mask = (unsigned char *)malloc(scaler_mask_bytes);
    recomputed_mask = (unsigned char *)malloc(scaler_mask_bytes);
    constant_mask_copy = (unsigned char *)malloc(scaler_mask_bytes);

    if (train_raw == NULL || validation_raw == NULL ||
        train_standardized == NULL || validation_standardized == NULL ||
        block == NULL ||
        mean == NULL || stddev == NULL ||
        recomputed_mean == NULL || recomputed_stddev == NULL ||
        mean_copy == NULL || stddev_copy == NULL ||
        constant_mask == NULL || recomputed_mask == NULL ||
        constant_mask_copy == NULL) {

        fprintf(stderr, "Errore: memoria insufficiente nel validatore.\n");
        goto cleanup;
    }

    if (gs_read_binary_matrix(
            argv[6],
            "train_features_raw_f32.bin",
            train_raw,
            train_value_count
        ) != 0 ||
        gs_read_binary_matrix(
            argv[6],
            "val_features_raw_f32.bin",
            validation_raw,
            validation_value_count
        ) != 0 ||
        gs_validate_raw_values(train_raw, train_value_count) != 0 ||
        gs_validate_raw_values(
            validation_raw,
            validation_value_count
        ) != 0) {

        fprintf(stderr, "Errore: matrici raw non valide.\n");
        goto cleanup;
    }
    printf("[PASS] complete_raw_matrices_finite_and_in_unit_interval\n");

    if (gs_reconstruct_raw_split(
            &pipeline,
            0,
            train_raw,
            train_count,
            block,
            &raw_maximum_difference
        ) != 0) {

        fprintf(stderr, "Errore: ricostruzione raw train non bitwise.\n");
        goto cleanup;
    }
    printf("RAW_TRAIN_BITWISE=true\n");

    if (gs_reconstruct_raw_split(
            &pipeline,
            1,
            validation_raw,
            validation_count,
            block,
            &raw_maximum_difference
        ) != 0) {

        fprintf(stderr, "Errore: ricostruzione raw validation non bitwise.\n");
        goto cleanup;
    }
    printf("RAW_VALIDATION_BITWISE=true\n");
    printf("RAW_MAX_ABS_DIFF=%.17g\n", raw_maximum_difference);
    printf("[PASS] complete_independent_raw_reconstruction\n");

    if (gs_validate_scaler_csv(
            argv[6],
            mean,
            stddev,
            constant_mask
        ) != 0 ||
        gs_validate_recomputed_scaler(
            train_raw,
            train_count,
            mean,
            stddev,
            constant_mask,
            recomputed_mean,
            recomputed_stddev,
            recomputed_mask,
            &scaler_maximum_difference,
            &constant_feature_count
        ) != 0) {

        fprintf(stderr, "Errore: scaler non coerente col solo train.\n");
        goto cleanup;
    }
    printf("CONSTANT_FEATURES=%zu\n", constant_feature_count);
    printf("SCALER_MAX_ABS_DIFF=%.17g\n", scaler_maximum_difference);
    printf("[PASS] independently_recomputed_train_only_scaler\n");

    if (gs_read_binary_matrix(
            argv[6],
            "train_features_std_f32.bin",
            train_standardized,
            train_value_count
        ) != 0 ||
        gs_read_binary_matrix(
            argv[6],
            "val_features_std_f32.bin",
            validation_standardized,
            validation_value_count
        ) != 0 ||
        gs_validate_standardized_split(
            train_raw,
            train_standardized,
            train_count,
            mean,
            stddev,
            constant_mask,
            block,
            &standardized_maximum_difference
        ) != 0) {

        fprintf(stderr, "Errore: standardizzazione train non bitwise.\n");
        goto cleanup;
    }
    printf("STANDARDIZED_TRAIN_BITWISE=true\n");

    memcpy(mean_copy, mean, scaler_double_bytes);
    memcpy(stddev_copy, stddev, scaler_double_bytes);
    memcpy(constant_mask_copy, constant_mask, scaler_mask_bytes);

    if (gs_validate_standardized_split(
            validation_raw,
            validation_standardized,
            validation_count,
            mean,
            stddev,
            constant_mask,
            block,
            &standardized_maximum_difference
        ) != 0 ||
        memcmp(mean, mean_copy, scaler_double_bytes) != 0 ||
        memcmp(stddev, stddev_copy, scaler_double_bytes) != 0 ||
        memcmp(
            constant_mask,
            constant_mask_copy,
            scaler_mask_bytes
        ) != 0) {

        fprintf(stderr, "Errore: transform validation/scaler non validi.\n");
        goto cleanup;
    }
    printf("STANDARDIZED_VALIDATION_BITWISE=true\n");
    printf(
        "STANDARDIZATION_MAX_ABS_DIFF=%.17g\n",
        standardized_maximum_difference
    );
    printf("SCALER_UNCHANGED_AFTER_VALIDATION=true\n");
    printf("[PASS] complete_standardization_and_train_only_state\n");

    if (gs_compute_standardized_train_statistics(
            train_standardized,
            train_count,
            constant_mask,
            &standardized_statistics
        ) != 0 ||
        standardized_statistics.constant_feature_count !=
            constant_feature_count) {

        fprintf(stderr, "Errore: statistiche train standardizzato.\n");
        goto cleanup;
    }
    printf(
        "TRAIN_STD_MAX_ABS_MEAN=%.17g\n",
        standardized_statistics.max_abs_mean
    );
    printf(
        "TRAIN_STD_MAX_ABS_POPULATION_STDDEV_ERROR=%.17g\n",
        standardized_statistics.max_abs_population_stddev_error
    );
    printf("[PASS] standardized_train_population_statistics\n");

    if (gs_verify_sha256_manifest(argv[6], input_paths) != 0) {
        fprintf(stderr, "Errore: checksum artefatti non corrispondenti.\n");
        goto cleanup;
    }
    printf("INPUT_CHECKSUM_COUNT=4\n");
    printf("ARTIFACT_CHECKSUM_COUNT=9\n");
    printf("SELF_HASH_COUNT=0\n");
    printf("ALL_ARTIFACT_CHECKSUMS_MATCH=true\n");
    printf("[PASS] all_input_and_artifact_sha256\n");

    if (gs_path_state(partial_path) != 0 ||
        gs_exact_inventory_is_valid(argv[6]) != 0) {

        fprintf(stderr, "Errore: residui partial o inventario finale.\n");
        goto cleanup;
    }
    printf("[PASS] final_directory_complete_without_partial\n");

    printf("PROFILE=mini112_pool7_no_pca\n");
    printf("EVENT_COUNT=%zu\n", event_count);
    printf("TRAIN_ROWS=%zu\n", train_count);
    printf("VALIDATION_ROWS=%zu\n", validation_count);
    printf("FEATURE_DIM=%d\n", GS_BASELINE_MINI112_FEATURE_COUNT);
    printf("ALL MINI112 ARTIFACT TESTS PASSED\n");
    rc = 0;

cleanup:
    free(constant_mask_copy);
    free(recomputed_mask);
    free(constant_mask);
    free(stddev_copy);
    free(mean_copy);
    free(recomputed_stddev);
    free(recomputed_mean);
    free(stddev);
    free(mean);
    free(block);
    free(validation_standardized);
    free(train_standardized);
    free(validation_raw);
    free(train_raw);

    gs_event_manifest_free(&events);
    gs_baseline_mini112_pipeline_free(&pipeline);
    return rc;
}
