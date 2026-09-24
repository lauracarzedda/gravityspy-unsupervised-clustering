#include "baseline_mini75_features.h"
#include "baseline_mini75_pipeline.h"
#include "baseline_standardization.h"
#include "vae_tensor_spec.h"

#include <errno.h>
#include <float.h>
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
#endif


enum {
    GS_ARTIFACT_PATH_LEN = 4096
};


#define GS_ARTIFACT_NUMERIC_TOLERANCE 1.0e-4
#define GS_ARTIFACT_PARTIAL_SUFFIX ".partial"


typedef struct {
    double train_raw_min;
    double train_raw_max;
    double validation_raw_min;
    double validation_raw_max;
    double train_standardized_max_abs_mean;
    double train_standardized_max_abs_stddev_error;
    size_t constant_feature_count;
} GSArtifactStatistics;


static int gs_checked_product(
    size_t left,
    size_t right,
    size_t *product
)
{
    if (product == NULL) {
        return 1;
    }

    if (right != 0 && left > SIZE_MAX / right) {
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


static int gs_path_state(const char *path)
{
    struct stat information;

    errno = 0;
    if (stat(path, &information) == 0) {
        return 1;
    }

    if (errno == ENOENT) {
        return 0;
    }

    return -1;
}


static int gs_validate_output_paths(
    const char *partial_path,
    char *final_path,
    size_t final_path_size
)
{
    const char *suffix = GS_ARTIFACT_PARTIAL_SUFFIX;
    size_t partial_length;
    size_t suffix_length;
    size_t final_length;
    int partial_state;
    int final_state;

    if (!gs_metadata_value_is_valid(partial_path) || final_path == NULL ||
        final_path_size == 0) {

        return 1;
    }

    partial_length = strlen(partial_path);
    suffix_length = strlen(suffix);

    if (partial_length <= suffix_length ||
        strcmp(partial_path + partial_length - suffix_length, suffix) != 0) {

        fprintf(stderr, "Errore: la directory di output deve terminare in .partial.\n");
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
        fprintf(stderr, "Errore: impossibile controllare le directory di output.\n");
        return 1;
    }

    if (partial_state != 0) {
        fprintf(stderr, "Errore: la directory partial esiste gia': %s\n", partial_path);
        return 1;
    }

    if (final_state != 0) {
        fprintf(stderr, "Errore: la directory finale esiste gia': %s\n", final_path);
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
        fprintf(stderr, "Errore: creazione directory partial fallita: %s\n",
                partial_path);
        return 1;
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


static int gs_open_output_file(
    const char *output_directory,
    const char *filename,
    const char *mode,
    FILE **file
)
{
    char path[GS_ARTIFACT_PATH_LEN];

    if (file == NULL ||
        gs_path_join(output_directory, filename, path, sizeof(path)) != 0) {

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
        gs_open_output_file(output_directory, filename, "wb", &file) != 0) {

        return 1;
    }

    if (fwrite(values, sizeof(float), value_count, file) != value_count) {
        failed = 1;
    }

    if (gs_finish_write(file) != 0) {
        failed = 1;
    }

    if (failed) {
        fprintf(stderr, "Errore: scrittura matrice incompleta: %s\n", filename);
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
    const GSTensorManifest *tensors,
    const GSSavedSplit *split
)
{
    FILE *file = NULL;
    size_t split_pos;
    int failed = 0;

    if (events == NULL || tensors == NULL || split == NULL ||
        gs_open_output_file(output_directory, filename, "wb", &file) != 0) {

        return 1;
    }

    if (fputs(
            "row_index,split_pos,manifest_index,event_index,event_id,split\n",
            file
        ) == EOF) {

        failed = 1;
    }

    for (split_pos = 0; !failed && split_pos < split->count; split_pos++) {
        size_t manifest_index = split->indices[split_pos];
        const GSEventRecord *event_record =
            gs_event_manifest_get(events, manifest_index);
        const GSTensorManifestRecord *tensor_record =
            gs_tensor_manifest_get(tensors, manifest_index);

        if (event_record == NULL || tensor_record == NULL ||
            event_record->event_index != tensor_record->event_index ||
            strcmp(event_record->event_id, tensor_record->event_id) != 0) {

            failed = 1;
            break;
        }

        if (fprintf(
                file,
                "%lu,%lu,%lu,%d,",
                (unsigned long)split_pos,
                (unsigned long)split_pos,
                (unsigned long)manifest_index,
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
    size_t feature_index;
    int failed = 0;

    if (mean == NULL || stddev == NULL || constant_mask == NULL ||
        gs_open_output_file(output_directory, "scaler.csv", "wb", &file) != 0) {

        return 1;
    }

    if (fputs(
            "feature_index,channel,pooled_row,pooled_col,mean,stddev,constant\n",
            file
        ) == EOF) {

        failed = 1;
    }

    for (feature_index = 0;
         !failed && feature_index < GS_BASELINE_MINI75_FEATURE_COUNT;
         feature_index++) {

        size_t channel = feature_index / 225;
        size_t remainder = feature_index % 225;
        size_t pooled_row = remainder / GS_BASELINE_MINI75_POOLED_SIDE;
        size_t pooled_col = remainder % GS_BASELINE_MINI75_POOLED_SIDE;

        if (!isfinite(mean[feature_index]) ||
            !isfinite(stddev[feature_index]) ||
            stddev[feature_index] < 0.0 ||
            constant_mask[feature_index] > 1u ||
            fprintf(
                file,
                "%lu,%lu,%lu,%lu,%.17g,%.17g,%u\n",
                (unsigned long)feature_index,
                (unsigned long)channel,
                (unsigned long)pooled_row,
                (unsigned long)pooled_col,
                mean[feature_index],
                stddev[feature_index],
                (unsigned int)constant_mask[feature_index]
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
            "profile=mini75_pool5_no_pca\n"
            "input_shape=4x75x75\n"
            "input_layout=channel_major\n"
            "channel_0=0.5s\n"
            "channel_1=1.0s\n"
            "channel_2=2.0s\n"
            "channel_3=4.0s\n"
            "normalization=uint8_div_255\n"
            "feature_extractor=average_pool_5x5_non_overlapping\n"
            "pooled_shape=4x15x15\n"
            "feature_count=900\n"
            "feature_order=channel_pooled_row_pooled_col\n"
            "matrix_layout=row_major\n"
            "matrix_dtype=float32_ieee754\n"
            "matrix_endian=little\n"
            "standardization_fit=train_only\n"
            "standardization_ddof=0\n"
            "constant_epsilon=1e-12\n"
            "constant_output=0\n"
            "pca=false\n"
            "augmentation=false\n"
            "split_seed=123\n"
            "train_rows=6866\n"
            "validation_rows=1717\n"
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
        statistics == NULL) {

        return 1;
    }

    statistics->constant_feature_count = 0;
    statistics->train_standardized_max_abs_mean = 0.0;
    statistics->train_standardized_max_abs_stddev_error = 0.0;

    for (feature = 0;
         feature < GS_BASELINE_MINI75_FEATURE_COUNT;
         feature++) {

        size_t row;

        if (!isfinite(mean[feature]) || !isfinite(stddev[feature]) ||
            stddev[feature] < 0.0 || constant_mask[feature] > 1u) {

            return 1;
        }

        if (constant_mask[feature] != 0u) {
            statistics->constant_feature_count++;

            for (row = 0; row < train_rows; row++) {
                float value = train_values[
                    row * GS_BASELINE_MINI75_FEATURE_COUNT + feature
                ];

                if (!isfinite(value) || value != 0.0f) {
                    return 1;
                }
            }

            for (row = 0; row < validation_rows; row++) {
                float value = validation_values[
                    row * GS_BASELINE_MINI75_FEATURE_COUNT + feature
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
                    row * GS_BASELINE_MINI75_FEATURE_COUNT + feature
                ];

                if (!isfinite(value)) {
                    return 1;
                }
                standardized_mean += (double)value;
            }

            for (row = 0; row < validation_rows; row++) {
                float value = validation_values[
                    row * GS_BASELINE_MINI75_FEATURE_COUNT + feature
                ];

                if (!isfinite(value)) {
                    return 1;
                }
            }

            standardized_mean /= (double)train_rows;

            for (row = 0; row < train_rows; row++) {
                double difference =
                    (double)train_values[
                        row * GS_BASELINE_MINI75_FEATURE_COUNT + feature
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
                statistics->train_standardized_max_abs_stddev_error) {

                statistics->train_standardized_max_abs_stddev_error =
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
        gs_open_output_file(output_directory, "summary.txt", "wb", &file) != 0) {

        return 1;
    }

    if (fprintf(
            file,
            "format_version=1\n"
            "profile=mini75_pool5_no_pca\n"
            "manifest_events_path=%s\n"
            "tensor_manifest_path=%s\n"
            "train_split_path=%s\n"
            "validation_split_path=%s\n"
            "data_root=%s\n"
            "code_commit=%s\n"
            "event_count=8583\n"
            "train_rows=6866\n"
            "validation_rows=1717\n"
            "feature_count=900\n"
            "train_features_raw_rows=6866\n"
            "train_features_raw_cols=900\n"
            "train_features_raw_bytes=%lu\n"
            "val_features_raw_rows=1717\n"
            "val_features_raw_cols=900\n"
            "val_features_raw_bytes=%lu\n"
            "train_features_std_rows=6866\n"
            "train_features_std_cols=900\n"
            "train_features_std_bytes=%lu\n"
            "val_features_std_rows=1717\n"
            "val_features_std_cols=900\n"
            "val_features_std_bytes=%lu\n"
            "constant_feature_count=%lu\n"
            "train_raw_min=%.17g\n"
            "train_raw_max=%.17g\n"
            "validation_raw_min=%.17g\n"
            "validation_raw_max=%.17g\n"
            "raw_values_in_unit_interval=true\n"
            "standardized_values_finite=true\n"
            "train_std_nonconstant_max_abs_mean=%.17g\n"
            "train_std_nonconstant_max_abs_population_stddev_error=%.17g\n"
            "standardization_ddof=0\n"
            "constant_epsilon=1e-12\n"
            "standardization_fit=train_only\n"
            "validation_transformed_without_refit=true\n"
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
            (unsigned long)train_matrix_bytes,
            (unsigned long)validation_matrix_bytes,
            (unsigned long)train_matrix_bytes,
            (unsigned long)validation_matrix_bytes,
            (unsigned long)statistics->constant_feature_count,
            statistics->train_raw_min,
            statistics->train_raw_max,
            statistics->validation_raw_min,
            statistics->validation_raw_max,
            statistics->train_standardized_max_abs_mean,
            statistics->train_standardized_max_abs_stddev_error
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
    GSEventManifest events = {0};
    GSTensorManifest tensors = {0};
    GSSavedSplit train = {0};
    GSSavedSplit validation = {0};
    GSTensorSpec spec;
    GSArtifactStatistics statistics = {
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0u
    };

    float *train_raw = NULL;
    float *validation_raw = NULL;
    float *train_standardized = NULL;
    float *validation_standardized = NULL;
    float *tensor_buffer = NULL;
    double *mean = NULL;
    double *stddev = NULL;
    unsigned char *constant_mask = NULL;

    size_t train_value_count;
    size_t validation_value_count;
    size_t train_matrix_bytes;
    size_t validation_matrix_bytes;
    size_t tensor_bytes;
    size_t scaler_double_bytes;
    size_t scaler_mask_bytes;
    char final_path[GS_ARTIFACT_PATH_LEN];
    int rc = 1;

    if (argc != 8) {
        fprintf(
            stderr,
            "Uso: build_baseline_mini75_artifacts.exe manifest_eventi "
            "manifest_mini75 split_train split_validation data_root "
            "directory_partial commit_git\n"
        );
        return 1;
    }

    if (!gs_platform_supports_artifact_contract()) {
        fprintf(stderr, "Errore: piattaforma non compatibile con float32 little-endian.\n");
        return 1;
    }
    printf("[PASS] platform_contract\n");

    if (!gs_metadata_value_is_valid(argv[1]) ||
        !gs_metadata_value_is_valid(argv[2]) ||
        !gs_metadata_value_is_valid(argv[3]) ||
        !gs_metadata_value_is_valid(argv[4]) ||
        !gs_metadata_value_is_valid(argv[5]) ||
        !gs_commit_is_valid(argv[7]) ||
        gs_validate_output_paths(argv[6], final_path, sizeof(final_path)) != 0) {

        fprintf(stderr, "Errore: argomenti del builder non validi.\n");
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
            GS_BASELINE_MINI75_INPUT_COUNT,
            sizeof(float),
            &tensor_bytes
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
        ) != 0) {

        fprintf(stderr, "Errore: overflow nel calcolo delle allocazioni.\n");
        return 1;
    }

    if (train_matrix_bytes != 24717600u ||
        validation_matrix_bytes != 6181200u) {

        fprintf(stderr, "Errore: dimensioni binarie inattese.\n");
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

        fprintf(stderr, "Errore: caricamento o allineamento degli input fallito.\n");
        goto cleanup;
    }
    printf("[PASS] official_counts_and_alignment\n");

    train_raw = (float *)malloc(train_matrix_bytes);
    validation_raw = (float *)malloc(validation_matrix_bytes);
    train_standardized = (float *)malloc(train_matrix_bytes);
    validation_standardized = (float *)malloc(validation_matrix_bytes);
    tensor_buffer = (float *)malloc(tensor_bytes);
    mean = (double *)malloc(scaler_double_bytes);
    stddev = (double *)malloc(scaler_double_bytes);
    constant_mask = (unsigned char *)malloc(scaler_mask_bytes);

    if (train_raw == NULL || validation_raw == NULL ||
        train_standardized == NULL || validation_standardized == NULL ||
        tensor_buffer == NULL || mean == NULL || stddev == NULL ||
        constant_mask == NULL) {

        fprintf(stderr, "Errore: memoria insufficiente per le matrici complete.\n");
        goto cleanup;
    }

    if (gs_baseline_mini75_extract_split_rows(
            argv[5],
            &events,
            &tensors,
            &train,
            0,
            train.count,
            tensor_buffer,
            GS_BASELINE_MINI75_INPUT_COUNT,
            train_raw,
            train_value_count
        ) != GS_BASELINE_MINI75_PIPELINE_SUCCESS ||
        gs_baseline_mini75_extract_split_rows(
            argv[5],
            &events,
            &tensors,
            &validation,
            0,
            validation.count,
            tensor_buffer,
            GS_BASELINE_MINI75_INPUT_COUNT,
            validation_raw,
            validation_value_count
        ) != GS_BASELINE_MINI75_PIPELINE_SUCCESS) {

        fprintf(stderr, "Errore: estrazione completa delle feature fallita.\n");
        goto cleanup;
    }
    printf("[PASS] full_feature_extraction\n");

    if (gs_baseline_standardization_fit(
            train_raw,
            train.count,
            GS_BASELINE_MINI75_FEATURE_COUNT,
            mean,
            stddev,
            constant_mask,
            GS_BASELINE_STANDARDIZATION_DEFAULT_CONSTANT_EPSILON
        ) != GS_BASELINE_STANDARDIZATION_SUCCESS ||
        gs_baseline_standardization_transform(
            train_raw,
            train.count,
            GS_BASELINE_MINI75_FEATURE_COUNT,
            mean,
            stddev,
            constant_mask,
            train_standardized
        ) != GS_BASELINE_STANDARDIZATION_SUCCESS ||
        gs_baseline_standardization_transform(
            validation_raw,
            validation.count,
            GS_BASELINE_MINI75_FEATURE_COUNT,
            mean,
            stddev,
            constant_mask,
            validation_standardized
        ) != GS_BASELINE_STANDARDIZATION_SUCCESS) {

        fprintf(stderr, "Errore: standardizzazione train-only fallita.\n");
        goto cleanup;
    }
    printf("[PASS] train_only_standardization\n");

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
        ) != 0 ||
        gs_validate_standardized_matrices(
            train_standardized,
            train.count,
            validation_standardized,
            validation.count,
            mean,
            stddev,
            constant_mask,
            &statistics
        ) != 0) {

        fprintf(stderr, "Errore: validazione numerica fallita.\n");
        goto cleanup;
    }
    printf("[PASS] numeric_validation\n");

    if (gs_create_partial_directory(argv[6], final_path) != 0) {
        goto cleanup;
    }

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
        gs_write_scaler_csv(argv[6], mean, stddev, constant_mask) != 0 ||
        gs_write_rows_csv(
            argv[6],
            "train_rows.csv",
            "train",
            &events,
            &tensors,
            &train
        ) != 0 ||
        gs_write_rows_csv(
            argv[6],
            "val_rows.csv",
            "validation",
            &events,
            &tensors,
            &validation
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
        ) != 0) {

        fprintf(stderr, "Errore: scrittura degli artefatti fallita.\n");
        goto cleanup;
    }
    printf("[PASS] artifacts_written\n");

    printf("OUTPUT_DIRECTORY=%s\n", argv[6]);
    printf("TRAIN_ROWS=%d\n", GS_BASELINE_MINI75_OFFICIAL_TRAIN_COUNT);
    printf("VALIDATION_ROWS=%d\n", GS_BASELINE_MINI75_OFFICIAL_VALIDATION_COUNT);
    printf("FEATURE_DIM=%d\n", GS_BASELINE_MINI75_FEATURE_COUNT);
    printf(
        "CONSTANT_FEATURES=%lu\n",
        (unsigned long)statistics.constant_feature_count
    );
    printf("ARTIFACT BUILD COMPLETED\n");
    rc = 0;

cleanup:
    free(constant_mask);
    free(stddev);
    free(mean);
    free(tensor_buffer);
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
