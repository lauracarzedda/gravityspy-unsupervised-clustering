#include "baseline_mini75_features.h"
#include "baseline_mini75_pipeline.h"
#include "vae_tensor_spec.h"

#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


enum {
    GS_ARTIFACT_TEST_PATH_LEN = 4096,
    GS_ARTIFACT_TEST_LINE_LEN = 4096
};


#define GS_ARTIFACT_TEST_TOLERANCE 1.0e-4
#define GS_ARTIFACT_EXPECTED_COMMIT \
    "02f9360b71c7034c56c7c3ec5506543e35323904"


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


static int gs_platform_supports_artifact_contract(void)
{
    uint16_t value = UINT16_C(1);
    const unsigned char *bytes = (const unsigned char *)&value;

    return sizeof(float) == 4 &&
           FLT_RADIX == 2 &&
           FLT_MANT_DIG == 24 &&
           FLT_MAX_EXP == 128 &&
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
    size_t position;
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


static int gs_validate_expected_files_and_sizes(
    const char *artifact_directory
)
{
    const char *filenames[9] = {
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

    const size_t expected_sizes[9] = {
        0u,
        24717600u,
        6181200u,
        24717600u,
        6181200u,
        0u,
        0u,
        0u,
        0u
    };

    size_t index;

    for (index = 0; index < 9; index++) {
        char path[GS_ARTIFACT_TEST_PATH_LEN];
        size_t actual_size;

        if (gs_path_join(
                artifact_directory,
                filenames[index],
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
                } else if (*read_cursor == '\r' || *read_cursor == '\n') {
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


static int gs_validate_rows_csv(
    const char *artifact_directory,
    const char *filename,
    const char *expected_split_name,
    const GSEventManifest *events,
    const GSTensorManifest *tensors,
    const GSSavedSplit *split
)
{
    const char *expected_header =
        "row_index,split_pos,manifest_index,event_index,event_id,split";

    char path[GS_ARTIFACT_TEST_PATH_LEN];
    char line[GS_ARTIFACT_TEST_LINE_LEN];
    FILE *file;
    size_t row;
    int failed = 0;

    if (gs_path_join(
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
        if (strcmp(line, expected_header) != 0) {
            failed = 1;
        }
    }

    for (row = 0; !failed && row < split->count; row++) {
        char *fields[7];
        int field_count;
        size_t row_index;
        size_t split_pos;
        size_t manifest_index;
        int event_index;
        const GSEventRecord *event_record;
        const GSTensorManifestRecord *tensor_record;

        if (fgets(line, sizeof(line), file) == NULL) {
            failed = 1;
            break;
        }

        field_count = gs_csv_split_line(line, fields, 7);
        if (field_count != 6 ||
            gs_parse_size_value(fields[0], &row_index) != 0 ||
            gs_parse_size_value(fields[1], &split_pos) != 0 ||
            gs_parse_size_value(fields[2], &manifest_index) != 0 ||
            gs_parse_int_value(fields[3], &event_index) != 0 ||
            row_index != row || split_pos != row ||
            manifest_index != split->indices[row]) {

            failed = 1;
            break;
        }

        event_record = gs_event_manifest_get(events, manifest_index);
        tensor_record = gs_tensor_manifest_get(tensors, manifest_index);

        if (event_record == NULL || tensor_record == NULL ||
            event_index != event_record->event_index ||
            event_index != tensor_record->event_index ||
            strcmp(fields[4], event_record->event_id) != 0 ||
            strcmp(fields[4], tensor_record->event_id) != 0 ||
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

    char path[GS_ARTIFACT_TEST_PATH_LEN];
    char line[GS_ARTIFACT_TEST_LINE_LEN];
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
        if (strcmp(line, expected_header) != 0) {
            failed = 1;
        }
    }

    for (feature = 0;
         !failed && feature < GS_BASELINE_MINI75_FEATURE_COUNT;
         feature++) {

        char *fields[8];
        int field_count;
        size_t parsed_feature;
        size_t channel;
        size_t pooled_row;
        size_t pooled_col;
        size_t constant;
        size_t expected_channel = feature / 225;
        size_t remainder = feature % 225;

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
            pooled_row != remainder / GS_BASELINE_MINI75_POOLED_SIDE ||
            pooled_col != remainder % GS_BASELINE_MINI75_POOLED_SIDE ||
            stddev[feature] < 0.0 || constant > 1u) {

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
    char path[GS_ARTIFACT_TEST_PATH_LEN];
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
        fgetc(file) != EOF || ferror(file)) {

        failed = 1;
    }

    if (fclose(file) != 0) {
        failed = 1;
    }

    return failed;
}


static int gs_validate_raw_values(
    const float *train_raw,
    size_t train_value_count,
    const float *validation_raw,
    size_t validation_value_count
)
{
    size_t index;

    if (train_raw == NULL || validation_raw == NULL) {
        return 1;
    }

    for (index = 0; index < train_value_count; index++) {
        if (!isfinite(train_raw[index]) ||
            train_raw[index] < 0.0f || train_raw[index] > 1.0f) {

            return 1;
        }
    }

    for (index = 0; index < validation_value_count; index++) {
        if (!isfinite(validation_raw[index]) ||
            validation_raw[index] < 0.0f || validation_raw[index] > 1.0f) {

            return 1;
        }
    }

    return 0;
}


static int gs_validate_standardized_values(
    const float *train_standardized,
    size_t train_rows,
    const float *validation_standardized,
    size_t validation_rows,
    const unsigned char *constant_mask
)
{
    size_t feature;

    if (train_standardized == NULL || validation_standardized == NULL ||
        constant_mask == NULL) {

        return 1;
    }

    for (feature = 0;
         feature < GS_BASELINE_MINI75_FEATURE_COUNT;
         feature++) {

        size_t row;

        if (constant_mask[feature] != 0u) {
            for (row = 0; row < train_rows; row++) {
                float value = train_standardized[
                    row * GS_BASELINE_MINI75_FEATURE_COUNT + feature
                ];

                if (!isfinite(value) || value != 0.0f) {
                    return 1;
                }
            }

            for (row = 0; row < validation_rows; row++) {
                float value = validation_standardized[
                    row * GS_BASELINE_MINI75_FEATURE_COUNT + feature
                ];

                if (!isfinite(value) || value != 0.0f) {
                    return 1;
                }
            }
        } else {
            double mean = 0.0;
            double variance = 0.0;

            for (row = 0; row < train_rows; row++) {
                float value = train_standardized[
                    row * GS_BASELINE_MINI75_FEATURE_COUNT + feature
                ];

                if (!isfinite(value)) {
                    return 1;
                }
                mean += (double)value;
            }

            for (row = 0; row < validation_rows; row++) {
                float value = validation_standardized[
                    row * GS_BASELINE_MINI75_FEATURE_COUNT + feature
                ];

                if (!isfinite(value)) {
                    return 1;
                }
            }

            mean /= (double)train_rows;

            for (row = 0; row < train_rows; row++) {
                double difference =
                    (double)train_standardized[
                        row * GS_BASELINE_MINI75_FEATURE_COUNT + feature
                    ] - mean;

                variance += difference * difference;
            }

            variance /= (double)train_rows;

            if (fabs(mean) > GS_ARTIFACT_TEST_TOLERANCE ||
                fabs(sqrt(variance) - 1.0) >
                    GS_ARTIFACT_TEST_TOLERANCE) {

                return 1;
            }
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
    char path[GS_ARTIFACT_TEST_PATH_LEN];
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


static int gs_text_has_key(const char *text, const char *key)
{
    const char *cursor;
    size_t key_length;

    if (text == NULL || key == NULL) {
        return 0;
    }

    key_length = strlen(key);
    cursor = text;

    while ((cursor = strstr(cursor, key)) != NULL) {
        const char *value = cursor + key_length;
        int starts_line = cursor == text || cursor[-1] == '\n';

        if (starts_line && *value != '\0' &&
            *value != '\r' && *value != '\n') {

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

    if (key == NULL || value == NULL || line == NULL || line_size == 0) {
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
    const char *validation_split_path
)
{
    const char *feature_lines[] = {
        "format_version=1",
        "profile=mini75_pool5_no_pca",
        "input_shape=4x75x75",
        "input_layout=channel_major",
        "channel_0=0.5s",
        "channel_1=1.0s",
        "channel_2=2.0s",
        "channel_3=4.0s",
        "normalization=uint8_div_255",
        "feature_extractor=average_pool_5x5_non_overlapping",
        "pooled_shape=4x15x15",
        "feature_count=900",
        "feature_order=channel_pooled_row_pooled_col",
        "matrix_layout=row_major",
        "matrix_dtype=float32_ieee754",
        "matrix_endian=little",
        "standardization_fit=train_only",
        "standardization_ddof=0",
        "constant_epsilon=1e-12",
        "constant_output=0",
        "pca=false",
        "augmentation=false",
        "split_seed=123",
        "train_rows=6866",
        "validation_rows=1717",
        "code_commit=" GS_ARTIFACT_EXPECTED_COMMIT
    };

    const char *summary_lines[] = {
        "format_version=1",
        "profile=mini75_pool5_no_pca",
        "code_commit=" GS_ARTIFACT_EXPECTED_COMMIT,
        "event_count=8583",
        "train_rows=6866",
        "validation_rows=1717",
        "feature_count=900",
        "train_features_raw_rows=6866",
        "train_features_raw_cols=900",
        "train_features_raw_bytes=24717600",
        "val_features_raw_rows=1717",
        "val_features_raw_cols=900",
        "val_features_raw_bytes=6181200",
        "train_features_std_rows=6866",
        "train_features_std_cols=900",
        "train_features_std_bytes=24717600",
        "val_features_std_rows=1717",
        "val_features_std_cols=900",
        "val_features_std_bytes=6181200",
        "raw_values_in_unit_interval=true",
        "standardized_values_finite=true",
        "standardization_ddof=0",
        "constant_epsilon=1e-12",
        "standardization_fit=train_only",
        "validation_transformed_without_refit=true",
        "labels_used=false",
        "pca_used=false",
        "augmentation_used=false",
        "kmeans_used=false"
    };

    const char *summary_keys[] = {
        "data_root=",
        "constant_feature_count=",
        "train_raw_min=",
        "train_raw_max=",
        "validation_raw_min=",
        "validation_raw_max=",
        "train_std_nonconstant_max_abs_mean=",
        "train_std_nonconstant_max_abs_population_stddev_error="
    };

    char *feature_spec = NULL;
    char *summary = NULL;
    char dynamic_line[GS_ARTIFACT_TEST_PATH_LEN];
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

    for (index = 0;
         index < sizeof(summary_keys) / sizeof(summary_keys[0]);
         index++) {

        if (!gs_text_has_key(summary, summary_keys[index])) {
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
        !gs_text_has_line(summary, dynamic_line)) {

        failed = 1;
    }

cleanup:
    free(summary);
    free(feature_spec);
    return failed;
}


static int gs_csv_header_has_label_column(
    const char *artifact_directory,
    const char *filename
)
{
    char path[GS_ARTIFACT_TEST_PATH_LEN];
    char line[GS_ARTIFACT_TEST_LINE_LEN];
    char *fields[16];
    FILE *file;
    int field_count;
    int index;
    int failed = 0;

    if (gs_path_join(
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
    }

    if (fclose(file) != 0) {
        failed = 1;
    }

    if (failed) {
        return 1;
    }

    field_count = gs_csv_split_line(line, fields, 16);
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


static int gs_validate_label_independence(
    const char *artifact_directory
)
{
    typedef int (*GSValidateSignature)(
        const GSEventManifest *,
        const GSTensorManifest *,
        const GSSavedSplit *,
        const GSSavedSplit *
    );

    typedef int (*GSExtractSignature)(
        const char *,
        const GSEventManifest *,
        const GSTensorManifest *,
        const GSSavedSplit *,
        size_t,
        size_t,
        float *,
        size_t,
        float *,
        size_t
    );

    GSValidateSignature validate_function =
        gs_baseline_mini75_validate_alignment;
    GSExtractSignature extract_function =
        gs_baseline_mini75_extract_split_rows;

    return validate_function == NULL || extract_function == NULL ||
           gs_csv_header_has_label_column(
               artifact_directory,
               "scaler.csv"
           ) != 0 ||
           gs_csv_header_has_label_column(
               artifact_directory,
               "train_rows.csv"
           ) != 0 ||
           gs_csv_header_has_label_column(
               artifact_directory,
               "val_rows.csv"
           ) != 0;
}


int main(int argc, char **argv)
{
    GSEventManifest events = {0};
    GSTensorManifest tensors = {0};
    GSSavedSplit train = {0};
    GSSavedSplit validation = {0};
    GSTensorSpec spec;

    float *train_raw = NULL;
    float *validation_raw = NULL;
    float *train_standardized = NULL;
    float *validation_standardized = NULL;
    double *mean = NULL;
    double *stddev = NULL;
    unsigned char *constant_mask = NULL;

    size_t train_value_count;
    size_t validation_value_count;
    size_t train_matrix_bytes;
    size_t validation_matrix_bytes;
    size_t scaler_double_bytes;
    size_t scaler_mask_bytes;
    int rc = 1;

    if (argc != 6) {
        fprintf(
            stderr,
            "Uso: test_baseline_mini75_artifacts.exe manifest_eventi "
            "manifest_mini75 split_train split_validation directory_artefatti\n"
        );
        return 1;
    }

    if (!gs_platform_supports_artifact_contract()) {
        fprintf(stderr, "Errore: piattaforma float32 little-endian non valida.\n");
        return 1;
    }

    if (gs_checked_product(
            GS_BASELINE_MINI75_OFFICIAL_TRAIN_COUNT,
            GS_BASELINE_MINI75_FEATURE_COUNT,
            &train_value_count
        ) != 0 ||
        gs_checked_product(
            GS_BASELINE_MINI75_OFFICIAL_VALIDATION_COUNT,
            GS_BASELINE_MINI75_FEATURE_COUNT,
            &validation_value_count
        ) != 0 ||
        gs_checked_product(train_value_count, sizeof(float), &train_matrix_bytes) != 0 ||
        gs_checked_product(
            validation_value_count,
            sizeof(float),
            &validation_matrix_bytes
        ) != 0 ||
        gs_checked_product(
            GS_BASELINE_MINI75_FEATURE_COUNT,
            sizeof(double),
            &scaler_double_bytes
        ) != 0 ||
        gs_checked_product(
            GS_BASELINE_MINI75_FEATURE_COUNT,
            sizeof(unsigned char),
            &scaler_mask_bytes
        ) != 0 ||
        train_matrix_bytes != 24717600u ||
        validation_matrix_bytes != 6181200u) {

        fprintf(stderr, "Errore: overflow o dimensioni matrici non valide.\n");
        return 1;
    }

    if (gs_tensor_spec_init(&spec, GS_BASELINE_MINI75_SIDE) != 0 ||
        gs_event_manifest_load(argv[1], &events) != 0 ||
        gs_tensor_manifest_load(argv[2], &spec, &tensors) != 0 ||
        gs_saved_split_load(argv[3], "train", &events, &train) != 0 ||
        gs_saved_split_load(
            argv[4],
            "validation",
            &events,
            &validation
        ) != 0 ||
        gs_baseline_mini75_validate_alignment(
            &events,
            &tensors,
            &train,
            &validation
        ) != GS_BASELINE_MINI75_PIPELINE_SUCCESS) {

        fprintf(stderr, "Errore: input ufficiali non validi.\n");
        goto cleanup;
    }

    if (gs_validate_expected_files_and_sizes(argv[5]) != 0) {
        fprintf(stderr, "Errore: file o dimensioni degli artefatti non validi.\n");
        goto cleanup;
    }
    printf("[PASS] expected_files_and_sizes\n");

    if (gs_validate_rows_csv(
            argv[5],
            "train_rows.csv",
            "train",
            &events,
            &tensors,
            &train
        ) != 0 ||
        gs_validate_rows_csv(
            argv[5],
            "val_rows.csv",
            "validation",
            &events,
            &tensors,
            &validation
        ) != 0) {

        fprintf(stderr, "Errore: CSV righe non allineati.\n");
        goto cleanup;
    }
    printf("[PASS] row_csv_counts_and_order\n");

    train_raw = (float *)malloc(train_matrix_bytes);
    validation_raw = (float *)malloc(validation_matrix_bytes);
    train_standardized = (float *)malloc(train_matrix_bytes);
    validation_standardized = (float *)malloc(validation_matrix_bytes);
    mean = (double *)malloc(scaler_double_bytes);
    stddev = (double *)malloc(scaler_double_bytes);
    constant_mask = (unsigned char *)malloc(scaler_mask_bytes);

    if (train_raw == NULL || validation_raw == NULL ||
        train_standardized == NULL || validation_standardized == NULL ||
        mean == NULL || stddev == NULL || constant_mask == NULL) {

        fprintf(stderr, "Errore: memoria insufficiente nel validatore.\n");
        goto cleanup;
    }

    if (gs_validate_scaler_csv(argv[5], mean, stddev, constant_mask) != 0) {
        fprintf(stderr, "Errore: contratto scaler non valido.\n");
        goto cleanup;
    }
    printf("[PASS] scaler_contract\n");

    if (gs_read_binary_matrix(
            argv[5],
            "train_features_raw_f32.bin",
            train_raw,
            train_value_count
        ) != 0 ||
        gs_read_binary_matrix(
            argv[5],
            "val_features_raw_f32.bin",
            validation_raw,
            validation_value_count
        ) != 0 ||
        gs_validate_raw_values(
            train_raw,
            train_value_count,
            validation_raw,
            validation_value_count
        ) != 0) {

        fprintf(stderr, "Errore: valori delle matrici raw non validi.\n");
        goto cleanup;
    }
    printf("[PASS] raw_matrix_values\n");

    if (gs_read_binary_matrix(
            argv[5],
            "train_features_std_f32.bin",
            train_standardized,
            train_value_count
        ) != 0 ||
        gs_read_binary_matrix(
            argv[5],
            "val_features_std_f32.bin",
            validation_standardized,
            validation_value_count
        ) != 0 ||
        gs_validate_standardized_values(
            train_standardized,
            train.count,
            validation_standardized,
            validation.count,
            constant_mask
        ) != 0) {

        fprintf(stderr, "Errore: valori standardizzati non validi.\n");
        goto cleanup;
    }
    printf("[PASS] standardized_matrix_values\n");

    if (gs_validate_metadata_contract(
            argv[5],
            argv[1],
            argv[2],
            argv[3],
            argv[4]
        ) != 0) {

        fprintf(stderr, "Errore: contratto metadati non valido.\n");
        goto cleanup;
    }
    printf("[PASS] metadata_contract\n");

    if (gs_validate_label_independence(argv[5]) != 0) {
        fprintf(stderr, "Errore: indipendenza delle interfacce non valida.\n");
        goto cleanup;
    }
    printf("[PASS] label_independence\n");

    printf("TRAIN_ROWS=%d\n", GS_BASELINE_MINI75_OFFICIAL_TRAIN_COUNT);
    printf(
        "VALIDATION_ROWS=%d\n",
        GS_BASELINE_MINI75_OFFICIAL_VALIDATION_COUNT
    );
    printf("FEATURE_DIM=%d\n", GS_BASELINE_MINI75_FEATURE_COUNT);
    printf("ALL ARTIFACT TESTS PASSED\n");
    rc = 0;

cleanup:
    free(constant_mask);
    free(stddev);
    free(mean);
    free(validation_standardized);
    free(train_standardized);
    free(validation_raw);
    free(train_raw);

    gs_saved_split_free(&validation);
    gs_saved_split_free(&train);
    gs_tensor_manifest_free(&tensors);
    gs_event_manifest_free(&events);

    return rc;
}
