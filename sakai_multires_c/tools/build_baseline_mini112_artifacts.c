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

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/types.h>
#include <unistd.h>
#endif


enum {
    GS_ARTIFACT_PATH_LENGTH = 4096,
    GS_ARTIFACT_LINE_LENGTH = 4096,
    GS_ARTIFACT_SHA256_LENGTH = 64,
    GS_ARTIFACT_EXTRACTION_BLOCK_ROWS = 64,
    GS_ARTIFACT_CORE_COUNT = 9,
    GS_ARTIFACT_TOTAL_FILE_COUNT = 10,
    GS_ARTIFACT_INPUT_COUNT = 4
};


#define GS_ARTIFACT_PARTIAL_SUFFIX ".partial"
#define GS_ARTIFACT_NUMERIC_TOLERANCE 1.0e-4
#define GS_ARTIFACT_SIMULATED_FAILURE_ENV \
    "GS_MINI112_TEST_FAIL_AFTER_PARTIAL"
#define GS_ARTIFACT_SIMULATED_FAILURE_BASENAME \
    "mini112_failure_test.partial"


typedef struct {
    double train_raw_min;
    double train_raw_max;
    double validation_raw_min;
    double validation_raw_max;
    double train_standardized_max_abs_mean;
    double train_standardized_max_abs_population_stddev_error;
    size_t constant_feature_count;
} GSArtifactStatistics;


typedef struct {
    uint32_t state[8];
    uint64_t completed_bits;
    unsigned char block[64];
    size_t block_size;
} GSSha256;


static const char *const gs_artifact_core_names[GS_ARTIFACT_CORE_COUNT] = {
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


static const char *const gs_artifact_all_names[
    GS_ARTIFACT_TOTAL_FILE_COUNT
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


static const char *const gs_artifact_input_names[GS_ARTIFACT_INPUT_COUNT] = {
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


static int gs_metadata_value_is_valid(const char *value)
{
    return value != NULL &&
           value[0] != '\0' &&
           strchr(value, '\r') == NULL &&
           strchr(value, '\n') == NULL;
}


static int gs_commit_is_valid(const char *commit)
{
    size_t index;

    if (!gs_metadata_value_is_valid(commit) || strlen(commit) != 40u) {
        return 0;
    }

    for (index = 0; index < 40u; index++) {
        if (!((commit[index] >= '0' && commit[index] <= '9') ||
              (commit[index] >= 'a' && commit[index] <= 'f'))) {

            return 0;
        }
    }

    return 1;
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


static int gs_simulated_failure_requested(const char *partial_path)
{
    const char *value;

    if (partial_path == NULL ||
        strcmp(
            gs_path_basename(partial_path),
            GS_ARTIFACT_SIMULATED_FAILURE_BASENAME
        ) != 0) {

        return 0;
    }

    value = getenv(GS_ARTIFACT_SIMULATED_FAILURE_ENV);
    return value != NULL && strcmp(value, "1") == 0;
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


static int gs_validate_output_paths(
    const char *partial_path,
    char *final_path,
    size_t final_path_size
)
{
    size_t partial_length;
    size_t suffix_length = strlen(GS_ARTIFACT_PARTIAL_SUFFIX);
    size_t final_length;
    int partial_state;
    int final_state;

    if (!gs_metadata_value_is_valid(partial_path) || final_path == NULL ||
        final_path_size == 0) {

        return 1;
    }

    partial_length = strlen(partial_path);
    if (partial_length <= suffix_length ||
        strcmp(
            partial_path + partial_length - suffix_length,
            GS_ARTIFACT_PARTIAL_SUFFIX
        ) != 0) {

        fprintf(
            stderr,
            "Errore: la directory di output deve terminare in .partial.\n"
        );
        return 1;
    }

    final_length = partial_length - suffix_length;
    if (final_length == 0 || final_length >= final_path_size) {
        fprintf(stderr, "Errore: percorso finale troppo lungo o vuoto.\n");
        return 1;
    }

    memcpy(final_path, partial_path, final_length);
    final_path[final_length] = '\0';

    partial_state = gs_path_state(partial_path);
    final_state = gs_path_state(final_path);

    if (partial_state < 0 || final_state < 0) {
        fprintf(stderr, "Errore: controllo directory di output fallito.\n");
        return 1;
    }

    if (partial_state != 0) {
        fprintf(
            stderr,
            "Errore: la directory partial esiste gia': %s\n",
            partial_path
        );
        return 1;
    }

    if (final_state != 0) {
        fprintf(
            stderr,
            "Errore: la directory finale esiste gia': %s\n",
            final_path
        );
        return 1;
    }

    return 0;
}


static int gs_create_partial_directory(
    const char *partial_path,
    const char *final_path
)
{
    if (gs_path_state(partial_path) != 0 ||
        gs_path_state(final_path) != 0) {

        fprintf(stderr, "Errore: output creato da un altro processo.\n");
        return 1;
    }

#ifdef _WIN32
    if (_mkdir(partial_path) != 0) {
#else
    if (mkdir(partial_path, 0777) != 0) {
#endif
        fprintf(
            stderr,
            "Errore: creazione directory partial fallita: %s\n",
            partial_path
        );
        return 1;
    }

    return 0;
}


static int gs_remove_created_partial(const char *partial_path)
{
    size_t index;
    int failed = 0;

    if (partial_path == NULL) {
        return 1;
    }

    for (index = 0; index < GS_ARTIFACT_TOTAL_FILE_COUNT; index++) {
        char path[GS_ARTIFACT_PATH_LENGTH];

        if (gs_path_join(
                partial_path,
                gs_artifact_all_names[index],
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

#ifdef _WIN32
    if (_rmdir(partial_path) != 0) {
#else
    if (rmdir(partial_path) != 0) {
#endif
        failed = 1;
    }

    return failed;
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


static int gs_open_output_file(
    const char *output_directory,
    const char *filename,
    const char *mode,
    FILE **file
)
{
    char path[GS_ARTIFACT_PATH_LENGTH];

    if (file == NULL ||
        gs_path_join(
            output_directory,
            filename,
            path,
            sizeof(path)
        ) != 0) {

        return 1;
    }

    *file = fopen(path, mode);
    if (*file == NULL) {
        fprintf(stderr, "Errore: apertura in scrittura fallita: %s\n", path);
        return 1;
    }

    return 0;
}


static int gs_write_binary_matrix(
    const char *output_directory,
    const char *filename,
    const float *values,
    size_t value_count
)
{
    FILE *file = NULL;
    int failed = 0;

    if (values == NULL ||
        gs_open_output_file(
            output_directory,
            filename,
            "wb",
            &file
        ) != 0) {

        return 1;
    }

    if (fwrite(values, sizeof(float), value_count, file) != value_count) {
        failed = 1;
    }

    if (gs_finish_write(file) != 0) {
        failed = 1;
    }

    if (failed) {
        fprintf(stderr, "Errore: scrittura matrice fallita: %s\n", filename);
    }

    return failed;
}


static int gs_write_csv_field(FILE *file, const char *value)
{
    const char *cursor;
    int needs_quotes;

    if (file == NULL || value == NULL) {
        return 1;
    }

    needs_quotes = strpbrk(value, ",\"\r\n") != NULL;
    if (!needs_quotes) {
        return fputs(value, file) == EOF ? 1 : 0;
    }

    if (fputc('"', file) == EOF) {
        return 1;
    }

    for (cursor = value; *cursor != '\0'; cursor++) {
        if (*cursor == '"' && fputc('"', file) == EOF) {
            return 1;
        }

        if (fputc((unsigned char)*cursor, file) == EOF) {
            return 1;
        }
    }

    return fputc('"', file) == EOF ? 1 : 0;
}


static int gs_write_rows_csv(
    const char *output_directory,
    const char *filename,
    const char *split_name,
    const GSEventManifest *events,
    const GSBaselineMini112Pipeline *pipeline,
    size_t row_count,
    int validation_split
)
{
    FILE *file = NULL;
    size_t split_pos;
    int failed = 0;

    if (events == NULL || pipeline == NULL ||
        gs_open_output_file(
            output_directory,
            filename,
            "wb",
            &file
        ) != 0) {

        return 1;
    }

    if (fputs(
            "row_index,split_pos,manifest_index,event_index,event_id,split\n",
            file
        ) == EOF) {

        failed = 1;
    }

    for (split_pos = 0; !failed && split_pos < row_count; split_pos++) {
        size_t manifest_index;
        const GSEventRecord *event_record;
        int index_result;

        index_result = validation_split ?
            gs_baseline_mini112_validation_manifest_index(
                pipeline,
                split_pos,
                &manifest_index
            ) :
            gs_baseline_mini112_train_manifest_index(
                pipeline,
                split_pos,
                &manifest_index
            );

        event_record = index_result ==
            GS_BASELINE_MINI112_PIPELINE_SUCCESS ?
                gs_event_manifest_get(events, manifest_index) :
                NULL;

        if (event_record == NULL ||
            event_record->event_id[0] == '\0' ||
            event_record->event_index <= 0 ||
            fprintf(
                file,
                "%zu,%zu,%zu,%d,",
                split_pos,
                split_pos,
                manifest_index,
                event_record->event_index
            ) < 0 ||
            gs_write_csv_field(file, event_record->event_id) != 0 ||
            fprintf(file, ",%s\n", split_name) < 0) {

            failed = 1;
        }
    }

    if (gs_finish_write(file) != 0) {
        failed = 1;
    }

    if (failed) {
        fprintf(stderr, "Errore: scrittura CSV righe fallita: %s\n", filename);
    }

    return failed;
}


static int gs_write_scaler_csv(
    const char *output_directory,
    const double *mean,
    const double *stddev,
    const unsigned char *constant_mask
)
{
    FILE *file = NULL;
    size_t feature;
    int failed = 0;

    if (mean == NULL || stddev == NULL || constant_mask == NULL ||
        gs_open_output_file(
            output_directory,
            "scaler.csv",
            "wb",
            &file
        ) != 0) {

        return 1;
    }

    if (fputs(
            "feature_index,channel,pooled_row,pooled_col,mean,stddev,constant\n",
            file
        ) == EOF) {

        failed = 1;
    }

    for (feature = 0;
         !failed && feature < GS_BASELINE_MINI112_FEATURE_COUNT;
         feature++) {

        size_t values_per_channel =
            (size_t)GS_BASELINE_MINI112_POOLED_SIDE *
            (size_t)GS_BASELINE_MINI112_POOLED_SIDE;

        size_t channel = feature / values_per_channel;
        size_t remainder = feature % values_per_channel;
        size_t pooled_row =
            remainder / GS_BASELINE_MINI112_POOLED_SIDE;
        size_t pooled_col =
            remainder % GS_BASELINE_MINI112_POOLED_SIDE;

        if (!isfinite(mean[feature]) ||
            !isfinite(stddev[feature]) ||
            stddev[feature] < 0.0 ||
            constant_mask[feature] > 1u ||
            fprintf(
                file,
                "%zu,%zu,%zu,%zu,%.17g,%.17g,%u\n",
                feature,
                channel,
                pooled_row,
                pooled_col,
                mean[feature],
                stddev[feature],
                (unsigned int)constant_mask[feature]
            ) < 0) {

            failed = 1;
        }
    }

    if (gs_finish_write(file) != 0) {
        failed = 1;
    }

    if (failed) {
        fprintf(stderr, "Errore: scrittura scaler.csv fallita.\n");
    }

    return failed;
}


static int gs_write_feature_spec(
    const char *output_directory,
    const char *code_commit
)
{
    FILE *file = NULL;
    int failed = 0;

    if (gs_open_output_file(
            output_directory,
            "feature_spec.txt",
            "wb",
            &file
        ) != 0) {

        return 1;
    }

    if (fprintf(
            file,
            "format_version=1\n"
            "profile=mini112_pool7_no_pca\n"
            "source=central_deterministic_tensors\n"
            "input_shape=4x112x112\n"
            "input_layout=channel_major\n"
            "channels=4\n"
            "image_side=112\n"
            "channel_0=0.5s\n"
            "channel_1=1.0s\n"
            "channel_2=2.0s\n"
            "channel_3=4.0s\n"
            "normalization=uint8_div_255\n"
            "normalization_before_pooling=uint8/255.0\n"
            "feature_extractor=average_pool_7x7_non_overlapping\n"
            "pooling=average\n"
            "pool_side=7\n"
            "stride=7\n"
            "padding=none\n"
            "pooled_shape=4x16x16\n"
            "pooled_side=16\n"
            "feature_count=1024\n"
            "feature_order=channel_pooled_row_pooled_col\n"
            "matrix_layout=row_major\n"
            "matrix_dtype=float32_ieee754\n"
            "matrix_endian=little\n"
            "standardization_fit=train_only\n"
            "scaler_fit=train_only\n"
            "standardization_variance=population\n"
            "standardization_ddof=0\n"
            "constant_epsilon=1e-12\n"
            "standardization_epsilon=1e-12\n"
            "constant_output=0\n"
            "pca=none\n"
            "augmentation=none\n"
            "labels_used=false\n"
            "split_seed=123\n"
            "train_rows=6866\n"
            "validation_rows=1717\n"
            "train_count=6866\n"
            "validation_count=1717\n"
            "extraction_block_rows=64\n"
            "code_commit=%s\n",
            code_commit
        ) < 0) {

        failed = 1;
    }

    if (gs_finish_write(file) != 0) {
        failed = 1;
    }

    return failed;
}


static int gs_validate_raw_matrix(
    const float *values,
    size_t value_count,
    double *minimum,
    double *maximum
)
{
    size_t index;
    double local_minimum;
    double local_maximum;

    if (values == NULL || value_count == 0 ||
        minimum == NULL || maximum == NULL) {

        return 1;
    }

    local_minimum = (double)values[0];
    local_maximum = (double)values[0];

    for (index = 0; index < value_count; index++) {
        double value = (double)values[index];

        if (!isfinite(value) || value < 0.0 || value > 1.0) {
            return 1;
        }

        if (value < local_minimum) {
            local_minimum = value;
        }

        if (value > local_maximum) {
            local_maximum = value;
        }
    }

    *minimum = local_minimum;
    *maximum = local_maximum;
    return 0;
}


static int gs_validate_standardized_matrices(
    const float *train_values,
    size_t train_rows,
    const float *validation_values,
    size_t validation_rows,
    const double *mean,
    const double *stddev,
    const unsigned char *constant_mask,
    GSArtifactStatistics *statistics
)
{
    size_t feature;

    if (train_values == NULL || validation_values == NULL ||
        mean == NULL || stddev == NULL || constant_mask == NULL ||
        statistics == NULL ||
        train_rows > SIZE_MAX / GS_BASELINE_MINI112_FEATURE_COUNT ||
        validation_rows > SIZE_MAX / GS_BASELINE_MINI112_FEATURE_COUNT) {

        return 1;
    }

    statistics->constant_feature_count = 0;
    statistics->train_standardized_max_abs_mean = 0.0;
    statistics->train_standardized_max_abs_population_stddev_error = 0.0;

    for (feature = 0;
         feature < GS_BASELINE_MINI112_FEATURE_COUNT;
         feature++) {

        size_t row;

        if (!isfinite(mean[feature]) ||
            !isfinite(stddev[feature]) ||
            stddev[feature] < 0.0 ||
            constant_mask[feature] > 1u) {

            return 1;
        }

        if (constant_mask[feature] != 0u) {
            statistics->constant_feature_count++;

            for (row = 0; row < train_rows; row++) {
                float value = train_values[
                    row * GS_BASELINE_MINI112_FEATURE_COUNT + feature
                ];

                if (!isfinite(value) || value != 0.0f) {
                    return 1;
                }
            }

            for (row = 0; row < validation_rows; row++) {
                float value = validation_values[
                    row * GS_BASELINE_MINI112_FEATURE_COUNT + feature
                ];

                if (!isfinite(value) || value != 0.0f) {
                    return 1;
                }
            }
        } else {
            double standardized_mean = 0.0;
            double standardized_variance = 0.0;
            double mean_error;
            double stddev_error;

            if (stddev[feature] <= 0.0) {
                return 1;
            }

            for (row = 0; row < train_rows; row++) {
                float value = train_values[
                    row * GS_BASELINE_MINI112_FEATURE_COUNT + feature
                ];

                if (!isfinite(value)) {
                    return 1;
                }

                standardized_mean += (double)value;
            }

            for (row = 0; row < validation_rows; row++) {
                float value = validation_values[
                    row * GS_BASELINE_MINI112_FEATURE_COUNT + feature
                ];

                if (!isfinite(value)) {
                    return 1;
                }
            }

            standardized_mean /= (double)train_rows;

            for (row = 0; row < train_rows; row++) {
                double difference =
                    (double)train_values[
                        row * GS_BASELINE_MINI112_FEATURE_COUNT + feature
                    ] - standardized_mean;

                standardized_variance += difference * difference;
            }

            standardized_variance /= (double)train_rows;
            mean_error = fabs(standardized_mean);
            stddev_error = fabs(sqrt(standardized_variance) - 1.0);

            if (mean_error > GS_ARTIFACT_NUMERIC_TOLERANCE ||
                stddev_error > GS_ARTIFACT_NUMERIC_TOLERANCE) {

                return 1;
            }

            if (mean_error >
                statistics->train_standardized_max_abs_mean) {

                statistics->train_standardized_max_abs_mean = mean_error;
            }

            if (stddev_error >
                statistics->
                    train_standardized_max_abs_population_stddev_error) {

                statistics->
                    train_standardized_max_abs_population_stddev_error =
                        stddev_error;
            }
        }
    }

    return 0;
}


static int gs_write_summary(
    const char *output_directory,
    const char *manifest_events_path,
    const char *tensor_manifest_path,
    const char *train_split_path,
    const char *validation_split_path,
    const char *data_root,
    const char *code_commit,
    size_t train_matrix_bytes,
    size_t validation_matrix_bytes,
    const GSArtifactStatistics *statistics
)
{
    FILE *file = NULL;
    int failed = 0;

    if (statistics == NULL ||
        gs_open_output_file(
            output_directory,
            "summary.txt",
            "wb",
            &file
        ) != 0) {

        return 1;
    }

    if (fprintf(
            file,
            "format_version=1\n"
            "profile=mini112_pool7_no_pca\n"
            "manifest_events_path=%s\n"
            "tensor_manifest_path=%s\n"
            "train_split_path=%s\n"
            "validation_split_path=%s\n"
            "data_root=%s\n"
            "code_commit=%s\n"
            "event_count=8583\n"
            "train_rows=6866\n"
            "validation_rows=1717\n"
            "feature_count=1024\n"
            "extraction_block_rows=64\n"
            "train_features_raw_rows=6866\n"
            "train_features_raw_cols=1024\n"
            "train_features_raw_bytes=%zu\n"
            "val_features_raw_rows=1717\n"
            "val_features_raw_cols=1024\n"
            "val_features_raw_bytes=%zu\n"
            "train_features_std_rows=6866\n"
            "train_features_std_cols=1024\n"
            "train_features_std_bytes=%zu\n"
            "val_features_std_rows=1717\n"
            "val_features_std_cols=1024\n"
            "val_features_std_bytes=%zu\n"
            "constant_feature_count=%zu\n"
            "train_raw_min=%.17g\n"
            "train_raw_max=%.17g\n"
            "validation_raw_min=%.17g\n"
            "validation_raw_max=%.17g\n"
            "raw_values_in_unit_interval=true\n"
            "standardized_values_finite=true\n"
            "train_std_nonconstant_max_abs_mean=%.17g\n"
            "train_std_nonconstant_max_abs_population_stddev_error=%.17g\n"
            "standardization_variance=population\n"
            "standardization_ddof=0\n"
            "constant_epsilon=1e-12\n"
            "standardization_fit=train_only\n"
            "validation_transformed_without_refit=true\n"
            "scaler_unchanged_after_validation=true\n"
            "labels_used=false\n"
            "pca_used=false\n"
            "augmentation_used=false\n"
            "kmeans_used=false\n",
            manifest_events_path,
            tensor_manifest_path,
            train_split_path,
            validation_split_path,
            data_root,
            code_commit,
            train_matrix_bytes,
            validation_matrix_bytes,
            train_matrix_bytes,
            validation_matrix_bytes,
            statistics->constant_feature_count,
            statistics->train_raw_min,
            statistics->train_raw_max,
            statistics->validation_raw_min,
            statistics->validation_raw_max,
            statistics->train_standardized_max_abs_mean,
            statistics->
                train_standardized_max_abs_population_stddev_error
        ) < 0) {

        failed = 1;
    }

    if (gs_finish_write(file) != 0) {
        failed = 1;
    }

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
    char output[GS_ARTIFACT_SHA256_LENGTH + 1]
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

    output[GS_ARTIFACT_SHA256_LENGTH] = '\0';
    return 0;
}


static int gs_write_sha256_manifest(
    const char *output_directory,
    const char *const input_paths[GS_ARTIFACT_INPUT_COUNT]
)
{
    char input_hashes[GS_ARTIFACT_INPUT_COUNT][
        GS_ARTIFACT_SHA256_LENGTH + 1
    ];

    char artifact_hashes[GS_ARTIFACT_CORE_COUNT][
        GS_ARTIFACT_SHA256_LENGTH + 1
    ];

    FILE *file = NULL;
    size_t index;
    int failed = 0;

    for (index = 0; index < GS_ARTIFACT_INPUT_COUNT; index++) {
        if (strcmp(
                gs_path_basename(input_paths[index]),
                gs_artifact_input_names[index]
            ) != 0 ||
            gs_sha256_file(input_paths[index], input_hashes[index]) != 0) {

            return 1;
        }
    }

    for (index = 0; index < GS_ARTIFACT_CORE_COUNT; index++) {
        char path[GS_ARTIFACT_PATH_LENGTH];

        if (gs_path_join(
                output_directory,
                gs_artifact_core_names[index],
                path,
                sizeof(path)
            ) != 0 ||
            gs_sha256_file(path, artifact_hashes[index]) != 0) {

            return 1;
        }
    }

    if (gs_open_output_file(
            output_directory,
            "sha256.txt",
            "wb",
            &file
        ) != 0) {

        return 1;
    }

    for (index = 0;
         !failed && index < GS_ARTIFACT_INPUT_COUNT;
         index++) {

        if (fprintf(
                file,
                "INPUT %s %s\n",
                input_hashes[index],
                gs_artifact_input_names[index]
            ) < 0) {

            failed = 1;
        }
    }

    for (index = 0;
         !failed && index < GS_ARTIFACT_CORE_COUNT;
         index++) {

        if (fprintf(
                file,
                "ARTIFACT %s %s\n",
                artifact_hashes[index],
                gs_artifact_core_names[index]
            ) < 0) {

            failed = 1;
        }
    }

    if (gs_finish_write(file) != 0) {
        failed = 1;
    }

    return failed;
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


static int gs_verify_sha256_manifest(
    const char *output_directory,
    const char *const input_paths[GS_ARTIFACT_INPUT_COUNT]
)
{
    char sha_path[GS_ARTIFACT_PATH_LENGTH];
    char line[GS_ARTIFACT_LINE_LENGTH];
    char expected[GS_ARTIFACT_LINE_LENGTH];
    FILE *file;
    size_t index;
    int failed = 0;

    if (gs_path_join(
            output_directory,
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

    for (index = 0;
         !failed && index < GS_ARTIFACT_INPUT_COUNT;
         index++) {

        char hash[GS_ARTIFACT_SHA256_LENGTH + 1];
        int written;

        if (gs_sha256_file(input_paths[index], hash) != 0 ||
            (written = snprintf(
                expected,
                sizeof(expected),
                "INPUT %s %s",
                hash,
                gs_artifact_input_names[index]
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
         !failed && index < GS_ARTIFACT_CORE_COUNT;
         index++) {

        char artifact_path[GS_ARTIFACT_PATH_LENGTH];
        char hash[GS_ARTIFACT_SHA256_LENGTH + 1];
        int written;

        if (gs_path_join(
                output_directory,
                gs_artifact_core_names[index],
                artifact_path,
                sizeof(artifact_path)
            ) != 0 ||
            gs_sha256_file(artifact_path, hash) != 0 ||
            (written = snprintf(
                expected,
                sizeof(expected),
                "ARTIFACT %s %s",
                hash,
                gs_artifact_core_names[index]
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


static int gs_exact_inventory_is_valid(const char *directory)
{
    unsigned char seen[GS_ARTIFACT_TOTAL_FILE_COUNT] = {0};
    DIR *stream;
    struct dirent *entry;
    int failed = 0;
    size_t index;

    stream = opendir(directory);
    if (stream == NULL) {
        return 1;
    }

    while (!failed) {
        int matched = 0;

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

        for (index = 0;
             index < GS_ARTIFACT_TOTAL_FILE_COUNT;
             index++) {

            if (strcmp(entry->d_name, gs_artifact_all_names[index]) == 0) {
                char path[GS_ARTIFACT_PATH_LENGTH];
                struct stat information;

                if (seen[index] != 0u ||
                    gs_path_join(
                        directory,
                        entry->d_name,
                        path,
                        sizeof(path)
                    ) != 0 ||
                    stat(path, &information) != 0 ||
                    !S_ISREG(information.st_mode)) {

                    failed = 1;
                }

                seen[index] = 1u;
                matched = 1;
                break;
            }
        }

        if (!matched) {
            failed = 1;
        }
    }

    if (closedir(stream) != 0) {
        failed = 1;
    }

    for (index = 0;
         !failed && index < GS_ARTIFACT_TOTAL_FILE_COUNT;
         index++) {

        if (seen[index] == 0u) {
            failed = 1;
        }
    }

    return failed;
}


static int gs_validate_written_artifacts(
    const char *output_directory,
    size_t train_matrix_bytes,
    size_t validation_matrix_bytes,
    const char *const input_paths[GS_ARTIFACT_INPUT_COUNT]
)
{
    const size_t expected_sizes[GS_ARTIFACT_TOTAL_FILE_COUNT] = {
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

    if (train_matrix_bytes != expected_sizes[1] ||
        validation_matrix_bytes != expected_sizes[2] ||
        gs_exact_inventory_is_valid(output_directory) != 0) {

        return 1;
    }

    for (index = 0;
         index < GS_ARTIFACT_TOTAL_FILE_COUNT;
         index++) {

        char path[GS_ARTIFACT_PATH_LENGTH];
        size_t actual_size;

        if (gs_path_join(
                output_directory,
                gs_artifact_all_names[index],
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

    return gs_verify_sha256_manifest(output_directory, input_paths);
}


static int gs_extract_split_in_blocks(
    GSBaselineMini112Pipeline *pipeline,
    int validation_split,
    size_t row_count,
    float *matrix
)
{
    size_t first_row;

    if (pipeline == NULL || matrix == NULL || row_count == 0 ||
        row_count > SIZE_MAX / GS_BASELINE_MINI112_FEATURE_COUNT) {
        return 1;
    }

    for (first_row = 0; first_row < row_count;) {
        size_t remaining = row_count - first_row;
        size_t block_rows =
            remaining < GS_ARTIFACT_EXTRACTION_BLOCK_ROWS ?
                remaining :
                GS_ARTIFACT_EXTRACTION_BLOCK_ROWS;

        size_t block_values;
        size_t row_offset;
        int extract_result;

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
            row_offset > SIZE_MAX - block_values) {

            return 1;
        }

        extract_result = validation_split ?
            gs_baseline_mini112_extract_validation_rows(
                pipeline,
                first_row,
                block_rows,
                matrix + row_offset,
                block_values
            ) :
            gs_baseline_mini112_extract_train_rows(
                pipeline,
                first_row,
                block_rows,
                matrix + row_offset,
                block_values
            );

        if (extract_result != GS_BASELINE_MINI112_PIPELINE_SUCCESS) {
            fprintf(
                stderr,
                "Errore pipeline mini112: %s\n",
                gs_baseline_mini112_pipeline_error_string(extract_result)
            );
            return 1;
        }

        first_row += block_rows;
    }

    return 0;
}


int main(int argc, char **argv)
{
    GSBaselineMini112Pipeline pipeline = {0};
    GSEventManifest events = {0};
    GSArtifactStatistics statistics = {
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0u
    };

    float *train_raw = NULL;
    float *validation_raw = NULL;
    float *train_standardized = NULL;
    float *validation_standardized = NULL;
    double *mean = NULL;
    double *stddev = NULL;
    double *mean_copy = NULL;
    double *stddev_copy = NULL;
    unsigned char *constant_mask = NULL;
    unsigned char *constant_mask_copy = NULL;

    const char *input_paths[GS_ARTIFACT_INPUT_COUNT];
    size_t event_count;
    size_t train_count;
    size_t validation_count;
    size_t train_value_count;
    size_t validation_value_count;
    size_t train_matrix_bytes;
    size_t validation_matrix_bytes;
    size_t scaler_double_bytes;
    size_t scaler_mask_bytes;
    char final_path[GS_ARTIFACT_PATH_LENGTH];
    int partial_created = 0;
    int rc = 1;

    if (argc != 8) {
        fprintf(
            stderr,
            "Uso: build_baseline_mini112_artifacts.exe manifest_eventi "
            "manifest_mini112 split_train split_validation data_root "
            "directory_partial commit_git\n"
        );
        return 1;
    }

    if (!gs_platform_supports_artifact_contract()) {
        fprintf(
            stderr,
            "Errore: piattaforma float32 little-endian non compatibile.\n"
        );
        return 1;
    }
    printf("[PASS] platform_contract\n");

    input_paths[0] = argv[1];
    input_paths[1] = argv[2];
    input_paths[2] = argv[3];
    input_paths[3] = argv[4];

    if (!gs_metadata_value_is_valid(argv[1]) ||
        !gs_metadata_value_is_valid(argv[2]) ||
        !gs_metadata_value_is_valid(argv[3]) ||
        !gs_metadata_value_is_valid(argv[4]) ||
        !gs_metadata_value_is_valid(argv[5]) ||
        !gs_commit_is_valid(argv[7]) ||
        strcmp(gs_path_basename(argv[1]), gs_artifact_input_names[0]) != 0 ||
        strcmp(gs_path_basename(argv[2]), gs_artifact_input_names[1]) != 0 ||
        strcmp(gs_path_basename(argv[3]), gs_artifact_input_names[2]) != 0 ||
        strcmp(gs_path_basename(argv[4]), gs_artifact_input_names[3]) != 0 ||
        gs_validate_output_paths(
            argv[6],
            final_path,
            sizeof(final_path)
        ) != 0) {

        fprintf(stderr, "Errore: argomenti del builder non validi.\n");
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
        train_matrix_bytes != 28123136u ||
        validation_matrix_bytes != 7032832u) {

        fprintf(stderr, "Errore: overflow o dimensioni binarie inattese.\n");
        return 1;
    }

    if (gs_create_partial_directory(argv[6], final_path) != 0) {
        return 1;
    }
    partial_created = 1;

    if (gs_simulated_failure_requested(argv[6])) {
        fprintf(
            stderr,
            "Errore simulato dopo la creazione della partial di test.\n"
        );
        goto cleanup;
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
                "Errore apertura pipeline mini112: %s\n",
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

        fprintf(stderr, "Errore: conteggi o manifest eventi non validi.\n");
        goto cleanup;
    }
    printf("[PASS] official_inputs_and_pipeline\n");

    train_raw = (float *)malloc(train_matrix_bytes);
    validation_raw = (float *)malloc(validation_matrix_bytes);
    train_standardized = (float *)malloc(train_matrix_bytes);
    validation_standardized = (float *)malloc(validation_matrix_bytes);
    mean = (double *)malloc(scaler_double_bytes);
    stddev = (double *)malloc(scaler_double_bytes);
    mean_copy = (double *)malloc(scaler_double_bytes);
    stddev_copy = (double *)malloc(scaler_double_bytes);
    constant_mask = (unsigned char *)malloc(scaler_mask_bytes);
    constant_mask_copy = (unsigned char *)malloc(scaler_mask_bytes);

    if (train_raw == NULL || validation_raw == NULL ||
        train_standardized == NULL || validation_standardized == NULL ||
        mean == NULL || stddev == NULL ||
        mean_copy == NULL || stddev_copy == NULL ||
        constant_mask == NULL || constant_mask_copy == NULL) {

        fprintf(stderr, "Errore: memoria insufficiente.\n");
        goto cleanup;
    }

    if (gs_extract_split_in_blocks(
            &pipeline,
            0,
            train_count,
            train_raw
        ) != 0 ||
        gs_extract_split_in_blocks(
            &pipeline,
            1,
            validation_count,
            validation_raw
        ) != 0) {

        fprintf(stderr, "Errore: estrazione mini112 a blocchi fallita.\n");
        goto cleanup;
    }
    printf("[PASS] blockwise_raw_extraction\n");

    if (gs_validate_raw_matrix(
            train_raw,
            train_value_count,
            &statistics.train_raw_min,
            &statistics.train_raw_max
        ) != 0 ||
        gs_validate_raw_matrix(
            validation_raw,
            validation_value_count,
            &statistics.validation_raw_min,
            &statistics.validation_raw_max
        ) != 0) {

        fprintf(stderr, "Errore: matrice raw non valida.\n");
        goto cleanup;
    }
    printf("[PASS] raw_values_finite_and_in_unit_interval\n");

    if (gs_baseline_standardization_fit(
            train_raw,
            train_count,
            GS_BASELINE_MINI112_FEATURE_COUNT,
            mean,
            stddev,
            constant_mask,
            GS_BASELINE_STANDARDIZATION_DEFAULT_CONSTANT_EPSILON
        ) != GS_BASELINE_STANDARDIZATION_SUCCESS) {

        fprintf(stderr, "Errore: fit scaler train-only fallito.\n");
        goto cleanup;
    }

    memcpy(mean_copy, mean, scaler_double_bytes);
    memcpy(stddev_copy, stddev, scaler_double_bytes);
    memcpy(constant_mask_copy, constant_mask, scaler_mask_bytes);

    if (gs_baseline_standardization_transform(
            train_raw,
            train_count,
            GS_BASELINE_MINI112_FEATURE_COUNT,
            mean,
            stddev,
            constant_mask,
            train_standardized
        ) != GS_BASELINE_STANDARDIZATION_SUCCESS ||
        gs_baseline_standardization_transform(
            validation_raw,
            validation_count,
            GS_BASELINE_MINI112_FEATURE_COUNT,
            mean,
            stddev,
            constant_mask,
            validation_standardized
        ) != GS_BASELINE_STANDARDIZATION_SUCCESS ||
        memcmp(mean, mean_copy, scaler_double_bytes) != 0 ||
        memcmp(stddev, stddev_copy, scaler_double_bytes) != 0 ||
        memcmp(
            constant_mask,
            constant_mask_copy,
            scaler_mask_bytes
        ) != 0 ||
        gs_validate_standardized_matrices(
            train_standardized,
            train_count,
            validation_standardized,
            validation_count,
            mean,
            stddev,
            constant_mask,
            &statistics
        ) != 0) {

        fprintf(stderr, "Errore: standardizzazione train-only fallita.\n");
        goto cleanup;
    }
    printf("[PASS] train_only_standardization_and_scaler_immutability\n");

    if (gs_write_feature_spec(argv[6], argv[7]) != 0 ||
        gs_write_binary_matrix(
            argv[6],
            "train_features_raw_f32.bin",
            train_raw,
            train_value_count
        ) != 0 ||
        gs_write_binary_matrix(
            argv[6],
            "val_features_raw_f32.bin",
            validation_raw,
            validation_value_count
        ) != 0 ||
        gs_write_binary_matrix(
            argv[6],
            "train_features_std_f32.bin",
            train_standardized,
            train_value_count
        ) != 0 ||
        gs_write_binary_matrix(
            argv[6],
            "val_features_std_f32.bin",
            validation_standardized,
            validation_value_count
        ) != 0 ||
        gs_write_scaler_csv(
            argv[6],
            mean,
            stddev,
            constant_mask
        ) != 0 ||
        gs_write_rows_csv(
            argv[6],
            "train_rows.csv",
            "train",
            &events,
            &pipeline,
            train_count,
            0
        ) != 0 ||
        gs_write_rows_csv(
            argv[6],
            "val_rows.csv",
            "validation",
            &events,
            &pipeline,
            validation_count,
            1
        ) != 0 ||
        gs_write_summary(
            argv[6],
            argv[1],
            argv[2],
            argv[3],
            argv[4],
            argv[5],
            argv[7],
            train_matrix_bytes,
            validation_matrix_bytes,
            &statistics
        ) != 0 ||
        gs_write_sha256_manifest(argv[6], input_paths) != 0) {

        fprintf(stderr, "Errore: scrittura artefatti fallita.\n");
        goto cleanup;
    }
    printf("[PASS] artifacts_written_flushed_and_closed\n");

    if (gs_validate_written_artifacts(
            argv[6],
            train_matrix_bytes,
            validation_matrix_bytes,
            input_paths
        ) != 0) {

        fprintf(stderr, "Errore: validazione interna artefatti fallita.\n");
        goto cleanup;
    }
    printf("[PASS] internal_inventory_size_and_checksum_validation\n");

    if (rename(argv[6], final_path) != 0) {
        fprintf(stderr, "Errore: promozione atomica della partial fallita.\n");
        goto cleanup;
    }
    partial_created = 0;

    printf("[PASS] partial_promoted_to_final\n");
    printf("PROFILE=mini112_pool7_no_pca\n");
    printf("OUTPUT_DIRECTORY=%s\n", final_path);
    printf("EXTRACTION_BLOCK_ROWS=%d\n", GS_ARTIFACT_EXTRACTION_BLOCK_ROWS);
    printf("EVENT_COUNT=%zu\n", event_count);
    printf("TRAIN_ROWS=%zu\n", train_count);
    printf("VALIDATION_ROWS=%zu\n", validation_count);
    printf("FEATURE_DIM=%d\n", GS_BASELINE_MINI112_FEATURE_COUNT);
    printf(
        "CONSTANT_FEATURES=%zu\n",
        statistics.constant_feature_count
    );
    printf(
        "TRAIN_STD_MAX_ABS_MEAN=%.17g\n",
        statistics.train_standardized_max_abs_mean
    );
    printf(
        "TRAIN_STD_MAX_ABS_POPULATION_STDDEV_ERROR=%.17g\n",
        statistics.train_standardized_max_abs_population_stddev_error
    );
    printf("SCALER_UNCHANGED_AFTER_VALIDATION=true\n");
    printf("ARTIFACT BUILD COMPLETED\n");
    rc = 0;

cleanup:
    free(constant_mask_copy);
    free(constant_mask);
    free(stddev_copy);
    free(mean_copy);
    free(stddev);
    free(mean);
    free(validation_standardized);
    free(train_standardized);
    free(validation_raw);
    free(train_raw);

    gs_event_manifest_free(&events);
    gs_baseline_mini112_pipeline_free(&pipeline);

    if (partial_created) {
        if (gs_remove_created_partial(argv[6]) != 0) {
            fprintf(
                stderr,
                "Errore: cleanup della partial creata dalla run fallito.\n"
            );
            rc = 1;
        } else {
            printf("[PASS] current_run_partial_cleanup\n");
        }
    }

    return rc;
}
