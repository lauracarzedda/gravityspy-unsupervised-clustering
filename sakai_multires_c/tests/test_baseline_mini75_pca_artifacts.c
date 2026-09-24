#include "baseline_pca.h"

#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


enum {
    GS_PCA_TEST_TRAIN_ROWS = 6866,
    GS_PCA_TEST_VALIDATION_ROWS = 1717,
    GS_PCA_TEST_INPUT_DIM = 900,
    GS_PCA_TEST_PATH_LENGTH = 4096,
    GS_PCA_TEST_ARTIFACT_COUNT = 15
};

#define GS_PCA_TEST_SCORE_MEAN_TOLERANCE 5.0e-5
#define GS_PCA_TEST_SCORE_VARIANCE_TOLERANCE 5.0e-4


typedef struct {
    size_t d90;
    size_t d95;
    size_t d99;
    double cumulative90;
    double cumulative95;
    double cumulative99;
} GSPCAThresholds;


typedef struct {
    const char *name;
    size_t size;
} GSPCAExpectedFile;


static int gs_checked_product(
    size_t left,
    size_t right,
    size_t *product
)
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
           sizeof(double) == 8 &&
           FLT_RADIX == 2 &&
           FLT_MANT_DIG == 24 &&
           DBL_MANT_DIG == 53 &&
           FLT_MAX_EXP == 128 &&
           DBL_MAX_EXP == 1024 &&
           bytes[0] == 1;
}


static int gs_metadata_value_is_valid(const char *value)
{
    const unsigned char *cursor = (const unsigned char *)value;

    if (value == NULL || value[0] == '\0') {
        return 0;
    }
    while (*cursor != '\0') {
        if (*cursor < 32 || *cursor == 127) {
            return 0;
        }
        cursor++;
    }
    return 1;
}


static int gs_commit_is_valid(const char *commit)
{
    size_t length;
    size_t index;

    if (!gs_metadata_value_is_valid(commit)) {
        return 0;
    }
    length = strlen(commit);
    if (length < 7 || length > 64) {
        return 0;
    }
    for (index = 0; index < length; index++) {
        char value = commit[index];

        if (!((value >= '0' && value <= '9') ||
              (value >= 'a' && value <= 'f') ||
              (value >= 'A' && value <= 'F'))) {

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
    size_t position = 0;
    int separator;

    if (directory == NULL || filename == NULL || path == NULL ||
        path_size == 0) {

        return 1;
    }
    directory_length = strlen(directory);
    filename_length = strlen(filename);
    separator =
        directory_length != 0 &&
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


static int gs_expected_file(
    const char *directory,
    const char *filename,
    size_t expected_size,
    int require_exact_size
)
{
    char path[GS_PCA_TEST_PATH_LENGTH];
    size_t actual_size;

    if (gs_path_join(directory, filename, path, sizeof(path)) != 0 ||
        gs_file_size(path, &actual_size) != 0) {

        return 1;
    }
    if (require_exact_size) {
        return actual_size != expected_size;
    }
    return actual_size == 0;
}


static void gs_strip_line_ending(char *line)
{
    size_t length = strlen(line);

    while (length != 0 &&
           (line[length - 1] == '\n' || line[length - 1] == '\r')) {

        line[--length] = '\0';
    }
}


static int gs_split_three_csv_fields(
    char *line,
    char **first,
    char **second,
    char **third
)
{
    char *comma1;
    char *comma2;

    if (line == NULL || first == NULL || second == NULL || third == NULL) {
        return 1;
    }
    comma1 = strchr(line, ',');
    if (comma1 == NULL) {
        return 1;
    }
    *comma1 = '\0';
    comma2 = strchr(comma1 + 1, ',');
    if (comma2 == NULL || strchr(comma2 + 1, ',') != NULL) {
        return 1;
    }
    *comma2 = '\0';

    *first = line;
    *second = comma1 + 1;
    *third = comma2 + 1;
    return 0;
}


static int gs_split_four_csv_fields(
    char *line,
    char **first,
    char **second,
    char **third,
    char **fourth
)
{
    char *comma1;
    char *comma2;
    char *comma3;

    if (line == NULL || first == NULL || second == NULL ||
        third == NULL || fourth == NULL) {

        return 1;
    }
    comma1 = strchr(line, ',');
    if (comma1 == NULL) {
        return 1;
    }
    *comma1 = '\0';
    comma2 = strchr(comma1 + 1, ',');
    if (comma2 == NULL) {
        return 1;
    }
    *comma2 = '\0';
    comma3 = strchr(comma2 + 1, ',');
    if (comma3 == NULL || strchr(comma3 + 1, ',') != NULL) {
        return 1;
    }
    *comma3 = '\0';

    *first = line;
    *second = comma1 + 1;
    *third = comma2 + 1;
    *fourth = comma3 + 1;
    return 0;
}


static int gs_parse_size_value(const char *text, size_t *value)
{
    char *endptr;
    unsigned long long parsed;

    if (text == NULL || value == NULL || text[0] == '\0' ||
        text[0] == '-') {

        return 1;
    }
    errno = 0;
    parsed = strtoull(text, &endptr, 10);
    if (errno == ERANGE || endptr == text || *endptr != '\0' ||
        parsed > (unsigned long long)SIZE_MAX) {

        return 1;
    }
    *value = (size_t)parsed;
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
    parsed = strtod(text, &endptr);
    if (errno == ERANGE || endptr == text || *endptr != '\0' ||
        !isfinite(parsed)) {

        return 1;
    }
    *value = parsed;
    return 0;
}


static int gs_read_thresholds(
    const char *directory,
    GSPCAThresholds *thresholds
)
{
    static const double expected_thresholds[3] = {0.90, 0.95, 0.99};
    char path[GS_PCA_TEST_PATH_LENGTH];
    char line[4096];
    FILE *file;
    size_t counts[3];
    double cumulative[3];
    size_t row;
    int failed = 0;

    if (thresholds == NULL ||
        gs_path_join(
            directory,
            "thresholds.csv",
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
        gs_strip_line_ending(line);
        if (strcmp(
                line,
                "threshold,component_count,cumulative_variance"
            ) != 0) {

            failed = 1;
        }
    }

    for (row = 0; !failed && row < 3; row++) {
        char *field1;
        char *field2;
        char *field3;
        double threshold;

        if (fgets(line, sizeof(line), file) == NULL) {
            failed = 1;
            break;
        }
        gs_strip_line_ending(line);
        if (gs_split_three_csv_fields(
                line,
                &field1,
                &field2,
                &field3
            ) != 0 ||
            gs_parse_double_value(field1, &threshold) != 0 ||
            threshold != expected_thresholds[row] ||
            gs_parse_size_value(field2, &counts[row]) != 0 ||
            gs_parse_double_value(field3, &cumulative[row]) != 0) {

            failed = 1;
        }
    }
    if (!failed && (fgetc(file) != EOF || ferror(file))) {
        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    if (failed ||
        counts[0] == 0 ||
        counts[0] > counts[1] ||
        counts[1] > counts[2] ||
        counts[2] > GS_PCA_TEST_INPUT_DIM ||
        cumulative[0] < 0.90 ||
        cumulative[1] < 0.95 ||
        cumulative[2] < 0.99 ||
        cumulative[0] > cumulative[1] ||
        cumulative[1] > cumulative[2] ||
        cumulative[2] > 1.0 + 1.0e-12) {

        return 1;
    }

    thresholds->d90 = counts[0];
    thresholds->d95 = counts[1];
    thresholds->d99 = counts[2];
    thresholds->cumulative90 = cumulative[0];
    thresholds->cumulative95 = cumulative[1];
    thresholds->cumulative99 = cumulative[2];
    return 0;
}


static int gs_validate_expected_files(
    const char *directory,
    const GSPCAThresholds *thresholds,
    GSPCAExpectedFile files[GS_PCA_TEST_ARTIFACT_COUNT]
)
{
    size_t vector_bytes;
    size_t matrix_count;
    size_t matrix_bytes;
    size_t train90_count;
    size_t validation90_count;
    size_t train95_count;
    size_t validation95_count;
    size_t train99_count;
    size_t validation99_count;
    size_t index;

    if (thresholds == NULL || files == NULL ||
        gs_checked_product(
            GS_PCA_TEST_INPUT_DIM,
            sizeof(double),
            &vector_bytes
        ) != 0 ||
        gs_checked_product(
            GS_PCA_TEST_INPUT_DIM,
            GS_PCA_TEST_INPUT_DIM,
            &matrix_count
        ) != 0 ||
        gs_checked_product(
            matrix_count,
            sizeof(double),
            &matrix_bytes
        ) != 0 ||
        gs_checked_product(
            GS_PCA_TEST_TRAIN_ROWS,
            thresholds->d90,
            &train90_count
        ) != 0 ||
        gs_checked_product(
            GS_PCA_TEST_VALIDATION_ROWS,
            thresholds->d90,
            &validation90_count
        ) != 0 ||
        gs_checked_product(
            GS_PCA_TEST_TRAIN_ROWS,
            thresholds->d95,
            &train95_count
        ) != 0 ||
        gs_checked_product(
            GS_PCA_TEST_VALIDATION_ROWS,
            thresholds->d95,
            &validation95_count
        ) != 0 ||
        gs_checked_product(
            GS_PCA_TEST_TRAIN_ROWS,
            thresholds->d99,
            &train99_count
        ) != 0 ||
        gs_checked_product(
            GS_PCA_TEST_VALIDATION_ROWS,
            thresholds->d99,
            &validation99_count
        ) != 0 ||
        gs_checked_product(train90_count, sizeof(float), &files[8].size) != 0 ||
        gs_checked_product(
            validation90_count,
            sizeof(float),
            &files[9].size
        ) != 0 ||
        gs_checked_product(train95_count, sizeof(float), &files[10].size) != 0 ||
        gs_checked_product(
            validation95_count,
            sizeof(float),
            &files[11].size
        ) != 0 ||
        gs_checked_product(train99_count, sizeof(float), &files[12].size) != 0 ||
        gs_checked_product(
            validation99_count,
            sizeof(float),
            &files[13].size
        ) != 0) {

        return 1;
    }

    files[0].size = 0;
    files[1].size = vector_bytes;
    files[2].size = matrix_bytes;
    files[3].size = vector_bytes;
    files[4].size = vector_bytes;
    files[5].size = vector_bytes;
    files[6].size = 0;
    files[7].size = 0;
    files[14].size = 0;

    for (index = 0; index < GS_PCA_TEST_ARTIFACT_COUNT; index++) {
        int exact = index == 1 || index == 2 || index == 3 ||
                    index == 4 || index == 5 ||
                    (index >= 8 && index <= 13);

        if (gs_expected_file(
                directory,
                files[index].name,
                files[index].size,
                exact
            ) != 0) {

            return 1;
        }
        if (!exact) {
            char path[GS_PCA_TEST_PATH_LENGTH];

            if (gs_path_join(
                    directory,
                    files[index].name,
                    path,
                    sizeof(path)
                ) != 0 ||
                gs_file_size(path, &files[index].size) != 0) {

                return 1;
            }
        }
    }
    return 0;
}


static int gs_read_binary(
    const char *directory,
    const char *filename,
    void *values,
    size_t element_size,
    size_t count
)
{
    char path[GS_PCA_TEST_PATH_LENGTH];
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
        fgetc(file) != EOF ||
        ferror(file)) {

        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    return failed;
}


static int gs_load_model(
    const char *directory,
    const GSPCAThresholds *thresholds,
    GSBaselinePCA *model
)
{
    size_t vector_bytes;
    size_t matrix_count;
    size_t matrix_bytes;
    size_t component;
    double total = 0.0;

    if (directory == NULL || thresholds == NULL || model == NULL ||
        gs_checked_product(
            GS_PCA_TEST_INPUT_DIM,
            sizeof(double),
            &vector_bytes
        ) != 0 ||
        gs_checked_product(
            GS_PCA_TEST_INPUT_DIM,
            GS_PCA_TEST_INPUT_DIM,
            &matrix_count
        ) != 0 ||
        gs_checked_product(
            matrix_count,
            sizeof(double),
            &matrix_bytes
        ) != 0) {

        return 1;
    }

    model->input_dim = GS_PCA_TEST_INPUT_DIM;
    model->component_count = GS_PCA_TEST_INPUT_DIM;
    model->mean = (double *)malloc(vector_bytes);
    model->components = (double *)malloc(matrix_bytes);
    model->eigenvalues = (double *)malloc(vector_bytes);
    model->explained_variance_ratio = (double *)malloc(vector_bytes);
    model->cumulative_explained_variance_ratio =
        (double *)malloc(vector_bytes);

    if (model->mean == NULL ||
        model->components == NULL ||
        model->eigenvalues == NULL ||
        model->explained_variance_ratio == NULL ||
        model->cumulative_explained_variance_ratio == NULL ||
        gs_read_binary(
            directory,
            "mean_f64.bin",
            model->mean,
            sizeof(double),
            GS_PCA_TEST_INPUT_DIM
        ) != 0 ||
        gs_read_binary(
            directory,
            "components_f64.bin",
            model->components,
            sizeof(double),
            matrix_count
        ) != 0 ||
        gs_read_binary(
            directory,
            "eigenvalues_f64.bin",
            model->eigenvalues,
            sizeof(double),
            GS_PCA_TEST_INPUT_DIM
        ) != 0 ||
        gs_read_binary(
            directory,
            "explained_variance_ratio_f64.bin",
            model->explained_variance_ratio,
            sizeof(double),
            GS_PCA_TEST_INPUT_DIM
        ) != 0 ||
        gs_read_binary(
            directory,
            "cumulative_explained_variance_ratio_f64.bin",
            model->cumulative_explained_variance_ratio,
            sizeof(double),
            GS_PCA_TEST_INPUT_DIM
        ) != 0) {

        return 1;
    }

    for (component = 0; component < GS_PCA_TEST_INPUT_DIM; component++) {
        total += model->eigenvalues[component];
        if (!isfinite(total)) {
            return 1;
        }
    }
    model->total_variance = total;
    model->components_90 = thresholds->d90;
    model->components_95 = thresholds->d95;
    model->components_99 = thresholds->d99;
    return 0;
}


static size_t gs_threshold_count(
    const double *cumulative,
    double threshold
)
{
    size_t component;

    for (component = 0; component < GS_PCA_TEST_INPUT_DIM; component++) {
        if (cumulative[component] >= threshold) {
            return component + 1;
        }
    }
    return GS_PCA_TEST_INPUT_DIM;
}


static int gs_validate_model(
    const GSBaselinePCA *model,
    const GSPCAThresholds *thresholds
)
{
    double previous_eigenvalue = INFINITY;
    double previous_cumulative = 0.0;
    double cumulative_from_ratio = 0.0;
    double orthonormal_tolerance =
        1024.0 * DBL_EPSILON * (double)GS_PCA_TEST_INPUT_DIM;
    size_t component;
    size_t feature;

    if (model == NULL || thresholds == NULL ||
        model->input_dim != GS_PCA_TEST_INPUT_DIM ||
        model->component_count != GS_PCA_TEST_INPUT_DIM ||
        !isfinite(model->total_variance) ||
        model->total_variance <= 0.0) {

        return 1;
    }

    for (feature = 0; feature < GS_PCA_TEST_INPUT_DIM; feature++) {
        if (!isfinite(model->mean[feature])) {
            return 1;
        }
    }

    for (component = 0;
         component < GS_PCA_TEST_INPUT_DIM;
         component++) {

        double eigenvalue = model->eigenvalues[component];
        double ratio = model->explained_variance_ratio[component];
        double cumulative =
            model->cumulative_explained_variance_ratio[component];
        size_t maximum_feature = 0;
        double maximum_abs =
            fabs(model->components[
                component * GS_PCA_TEST_INPUT_DIM
            ]);

        cumulative_from_ratio += ratio;
        if (!isfinite(eigenvalue) ||
            eigenvalue < 0.0 ||
            eigenvalue > previous_eigenvalue ||
            !isfinite(ratio) ||
            ratio < 0.0 ||
            ratio > 1.0 ||
            !isfinite(cumulative) ||
            cumulative < previous_cumulative ||
            cumulative > 1.0 + 1.0e-12 ||
            fabs(ratio - eigenvalue / model->total_variance) >
                1.0e-12 ||
            (component + 1 < GS_PCA_TEST_INPUT_DIM &&
             fabs(cumulative - cumulative_from_ratio) > 1.0e-12)) {

            return 1;
        }
        previous_eigenvalue = eigenvalue;
        previous_cumulative = cumulative;

        for (feature = 0; feature < GS_PCA_TEST_INPUT_DIM; feature++) {
            double loading =
                model->components[
                    component * GS_PCA_TEST_INPUT_DIM + feature
                ];
            double loading_abs;

            if (!isfinite(loading)) {
                return 1;
            }
            loading_abs = fabs(loading);
            if (loading_abs > maximum_abs) {
                maximum_abs = loading_abs;
                maximum_feature = feature;
            }
        }
        if (!(maximum_abs > 0.0) ||
            model->components[
                component * GS_PCA_TEST_INPUT_DIM + maximum_feature
            ] <= 0.0) {

            return 1;
        }
    }

    if (fabs(previous_cumulative - 1.0) > 1.0e-12 ||
        gs_threshold_count(
            model->cumulative_explained_variance_ratio,
            0.90
        ) != thresholds->d90 ||
        gs_threshold_count(
            model->cumulative_explained_variance_ratio,
            0.95
        ) != thresholds->d95 ||
        gs_threshold_count(
            model->cumulative_explained_variance_ratio,
            0.99
        ) != thresholds->d99 ||
        model->cumulative_explained_variance_ratio[
            thresholds->d90 - 1
        ] != thresholds->cumulative90 ||
        model->cumulative_explained_variance_ratio[
            thresholds->d95 - 1
        ] != thresholds->cumulative95 ||
        model->cumulative_explained_variance_ratio[
            thresholds->d99 - 1
        ] != thresholds->cumulative99) {

        return 1;
    }

    for (component = 0;
         component < GS_PCA_TEST_INPUT_DIM;
         component++) {

        size_t other_component;

        for (other_component = 0;
             other_component < GS_PCA_TEST_INPUT_DIM;
             other_component++) {

            double dot = 0.0;
            double expected =
                component == other_component ? 1.0 : 0.0;

            for (feature = 0; feature < GS_PCA_TEST_INPUT_DIM; feature++) {
                dot +=
                    model->components[
                        component * GS_PCA_TEST_INPUT_DIM + feature
                    ] *
                    model->components[
                        other_component * GS_PCA_TEST_INPUT_DIM + feature
                    ];
            }
            if (!isfinite(dot) ||
                fabs(dot - expected) > orthonormal_tolerance) {

                return 1;
            }
        }
    }
    return 0;
}


static int gs_validate_spectrum(
    const char *directory,
    const GSBaselinePCA *model
)
{
    char path[GS_PCA_TEST_PATH_LENGTH];
    char line[4096];
    FILE *file;
    size_t component;
    int failed = 0;

    if (model == NULL ||
        gs_path_join(
            directory,
            "spectrum.csv",
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
        gs_strip_line_ending(line);
        if (strcmp(
                line,
                "component_index,eigenvalue,explained_variance_ratio,"
                "cumulative_explained_variance_ratio"
            ) != 0) {

            failed = 1;
        }
    }

    for (component = 0;
         !failed && component < GS_PCA_TEST_INPUT_DIM;
         component++) {

        char *field1;
        char *field2;
        char *field3;
        char *field4;
        size_t parsed_component;
        double eigenvalue;
        double ratio;
        double cumulative;

        if (fgets(line, sizeof(line), file) == NULL) {
            failed = 1;
            break;
        }
        gs_strip_line_ending(line);
        if (gs_split_four_csv_fields(
                line,
                &field1,
                &field2,
                &field3,
                &field4
            ) != 0 ||
            gs_parse_size_value(field1, &parsed_component) != 0 ||
            gs_parse_double_value(field2, &eigenvalue) != 0 ||
            gs_parse_double_value(field3, &ratio) != 0 ||
            gs_parse_double_value(field4, &cumulative) != 0 ||
            parsed_component != component ||
            eigenvalue != model->eigenvalues[component] ||
            ratio != model->explained_variance_ratio[component] ||
            cumulative !=
                model->cumulative_explained_variance_ratio[component]) {

            failed = 1;
        }
    }

    if (!failed && (fgetc(file) != EOF || ferror(file))) {
        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    return failed;
}


static int gs_read_scores(
    const char *directory,
    const char *filename,
    size_t rows,
    size_t components,
    float **scores
)
{
    size_t count;
    size_t bytes;
    size_t index;
    float *values;

    if (scores == NULL ||
        gs_checked_product(rows, components, &count) != 0 ||
        gs_checked_product(count, sizeof(float), &bytes) != 0) {

        return 1;
    }
    values = (float *)malloc(bytes);
    if (values == NULL ||
        gs_read_binary(
            directory,
            filename,
            values,
            sizeof(float),
            count
        ) != 0) {

        free(values);
        return 1;
    }
    for (index = 0; index < count; index++) {
        if (!isfinite((double)values[index])) {
            free(values);
            return 1;
        }
    }
    *scores = values;
    return 0;
}


static int gs_validate_score_prefix(
    const float *prefix,
    const float *full,
    size_t rows,
    size_t prefix_components,
    size_t full_components
)
{
    size_t row;

    if (prefix == NULL || full == NULL ||
        prefix_components == 0 ||
        prefix_components > full_components) {

        return 1;
    }

    for (row = 0; row < rows; row++) {
        if (memcmp(
                prefix + row * prefix_components,
                full + row * full_components,
                prefix_components * sizeof(float)
            ) != 0) {

            return 1;
        }
    }
    return 0;
}


static int gs_read_source_row(
    const char *source_directory,
    const char *filename,
    size_t row_index,
    size_t row_count,
    float row[GS_PCA_TEST_INPUT_DIM]
)
{
    char path[GS_PCA_TEST_PATH_LENGTH];
    size_t value_count;
    size_t expected_bytes;
    size_t row_bytes;
    size_t offset;
    size_t size;
    FILE *file;
    size_t feature;
    int failed = 0;

    if (row == NULL || row_index >= row_count ||
        gs_checked_product(
            row_count,
            GS_PCA_TEST_INPUT_DIM,
            &value_count
        ) != 0 ||
        gs_checked_product(
            value_count,
            sizeof(float),
            &expected_bytes
        ) != 0 ||
        gs_checked_product(
            GS_PCA_TEST_INPUT_DIM,
            sizeof(float),
            &row_bytes
        ) != 0 ||
        gs_checked_product(row_index, row_bytes, &offset) != 0 ||
        offset > (size_t)LONG_MAX ||
        gs_path_join(
            source_directory,
            filename,
            path,
            sizeof(path)
        ) != 0 ||
        gs_file_size(path, &size) != 0 ||
        size != expected_bytes) {

        return 1;
    }

    file = fopen(path, "rb");
    if (file == NULL) {
        return 1;
    }
    if (fseek(file, (long)offset, SEEK_SET) != 0 ||
        fread(row, sizeof(float), GS_PCA_TEST_INPUT_DIM, file) !=
            GS_PCA_TEST_INPUT_DIM ||
        ferror(file)) {

        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    if (failed) {
        return 1;
    }
    for (feature = 0; feature < GS_PCA_TEST_INPUT_DIM; feature++) {
        if (!isfinite((double)row[feature])) {
            return 1;
        }
    }
    return 0;
}


static int gs_validate_recomputed_scores(
    const char *source_directory,
    const char *source_filename,
    size_t rows,
    const GSBaselinePCA *model,
    const float *stored_scores
)
{
    size_t indices[5];
    float row_values[GS_PCA_TEST_INPUT_DIM];
    float *calculated;
    size_t calculated_bytes;
    size_t index;
    int failed = 0;

    indices[0] = 0;
    indices[1] = rows - 1;
    indices[2] = rows / 2;
    indices[3] = rows / 3;
    indices[4] = (rows / 3) * 2;

    if (gs_checked_product(
            model->components_99,
            sizeof(float),
            &calculated_bytes
        ) != 0) {

        return 1;
    }
    calculated = (float *)malloc(calculated_bytes);
    if (calculated == NULL) {
        return 1;
    }

    for (index = 0; index < 5; index++) {
        size_t row_index = indices[index];

        if (gs_read_source_row(
                source_directory,
                source_filename,
                row_index,
                rows,
                row_values
            ) != 0 ||
            gs_baseline_pca_transform(
                model,
                row_values,
                1,
                GS_PCA_TEST_INPUT_DIM,
                model->components_99,
                calculated,
                model->components_99
            ) != GS_BASELINE_PCA_SUCCESS ||
            memcmp(
                calculated,
                stored_scores + row_index * model->components_99,
                calculated_bytes
            ) != 0) {

            failed = 1;
            break;
        }
    }

    free(calculated);
    return failed;
}


static int gs_validate_train_score_statistics(
    const float *scores,
    size_t components,
    const double *eigenvalues
)
{
    size_t component;

    if (scores == NULL || components == 0 || eigenvalues == NULL) {
        return 1;
    }

    for (component = 0; component < components; component++) {
        double mean = 0.0;
        double variance = 0.0;
        double mean_tolerance;
        double variance_tolerance;
        size_t row;

        for (row = 0; row < GS_PCA_TEST_TRAIN_ROWS; row++) {
            mean += (double)scores[row * components + component];
        }
        mean /= (double)GS_PCA_TEST_TRAIN_ROWS;

        for (row = 0; row < GS_PCA_TEST_TRAIN_ROWS; row++) {
            double centered =
                (double)scores[row * components + component] - mean;

            variance += centered * centered;
        }
        variance /= (double)GS_PCA_TEST_TRAIN_ROWS;

        mean_tolerance =
            GS_PCA_TEST_SCORE_MEAN_TOLERANCE *
            (1.0 + sqrt(eigenvalues[component]));
        variance_tolerance =
            GS_PCA_TEST_SCORE_VARIANCE_TOLERANCE *
            (1.0 + eigenvalues[component]);

        if (!isfinite(mean) ||
            !isfinite(variance) ||
            fabs(mean) > mean_tolerance ||
            fabs(variance - eigenvalues[component]) >
                variance_tolerance) {

            return 1;
        }
    }
    return 0;
}


static int gs_read_text(
    const char *directory,
    const char *filename,
    char **text
)
{
    char path[GS_PCA_TEST_PATH_LENGTH];
    size_t size;
    FILE *file;
    char *buffer;
    int failed = 0;

    if (text == NULL ||
        gs_path_join(directory, filename, path, sizeof(path)) != 0 ||
        gs_file_size(path, &size) != 0 ||
        size == 0 ||
        size == SIZE_MAX) {

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


static int gs_validate_metadata(
    const char *source_directory,
    const char *artifact_directory,
    const char *code_commit,
    const GSPCAExpectedFile files[GS_PCA_TEST_ARTIFACT_COUNT]
)
{
    static const char *spec_lines[] = {
        "format_version=1",
        "profile=mini75_pool5_pca",
        "source_profile=mini75_pool5_no_pca",
        "input_rows_train=6866",
        "input_rows_validation=1717",
        "input_dim=900",
        "input_standardization=train_only",
        "pca_fit=train_only",
        "validation_transformed_without_refit=true",
        "covariance_ddof=0",
        "backend=LAPACKE_dsyevd",
        "component_order=eigenvalue_descending",
        "component_layout=component_major",
        "sign_convention=max_abs_loading_positive_lowest_index_tie",
        "whitening=false",
        "threshold_90=0.90",
        "threshold_95=0.95",
        "threshold_99=0.99",
        "principal_configuration=pca95",
        "labels_used=false"
    };
    static const char *summary_lines[] = {
        "format_version=1",
        "profile=mini75_pool5_pca",
        "backend=LAPACKE_dsyevd",
        "train_rows=6866",
        "validation_rows=1717",
        "input_dim=900",
        "pca_fit=train_only",
        "validation_transformed_without_refit=true",
        "whitening=false",
        "labels_used=false"
    };
    static const char *summary_keys[] = {
        "toolchain=",
        "d90=",
        "d95=",
        "d99=",
        "cumulative_variance_90=",
        "cumulative_variance_95=",
        "cumulative_variance_99=",
        "total_variance=",
        "maximum_eigenvalue=",
        "minimum_eigenvalue=",
        "eigenvalues_zero_after_tolerance=",
        "read_cpu_seconds_approx=",
        "fit_pca_cpu_seconds_approx=",
        "transform_train_cpu_seconds_approx=",
        "transform_validation_cpu_seconds_approx=",
        "write_cpu_seconds_approx="
    };
    static const char *size_keys[GS_PCA_TEST_ARTIFACT_COUNT] = {
        "pca_spec_txt_bytes=",
        "mean_f64_bin_bytes=",
        "components_f64_bin_bytes=",
        "eigenvalues_f64_bin_bytes=",
        "explained_variance_ratio_f64_bin_bytes=",
        "cumulative_explained_variance_ratio_f64_bin_bytes=",
        "thresholds_csv_bytes=",
        "spectrum_csv_bytes=",
        "train_scores_pca90_f32_bin_bytes=",
        "val_scores_pca90_f32_bin_bytes=",
        "train_scores_pca95_f32_bin_bytes=",
        "val_scores_pca95_f32_bin_bytes=",
        "train_scores_pca99_f32_bin_bytes=",
        "val_scores_pca99_f32_bin_bytes=",
        "summary_txt_bytes="
    };
    char *spec = NULL;
    char *summary = NULL;
    char line[GS_PCA_TEST_PATH_LENGTH + 128];
    size_t index;
    int failed = 0;

    if (gs_read_text(artifact_directory, "pca_spec.txt", &spec) != 0 ||
        gs_read_text(artifact_directory, "summary.txt", &summary) != 0) {

        free(summary);
        free(spec);
        return 1;
    }

    for (index = 0;
         index < sizeof(spec_lines) / sizeof(spec_lines[0]);
         index++) {

        if (!gs_text_has_line(spec, spec_lines[index])) {
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

    if (snprintf(
            line,
            sizeof(line),
            "code_commit=%s",
            code_commit
        ) < 0 ||
        !gs_text_has_line(spec, line) ||
        !gs_text_has_line(summary, line) ||
        snprintf(
            line,
            sizeof(line),
            "source_directory=%s",
            source_directory
        ) < 0 ||
        !gs_text_has_line(summary, line)) {

        failed = 1;
    }

    for (index = 0; index < GS_PCA_TEST_ARTIFACT_COUNT; index++) {
        if (snprintf(
                line,
                sizeof(line),
                "%s%lu",
                size_keys[index],
                (unsigned long)files[index].size
            ) < 0 ||
            !gs_text_has_line(summary, line)) {

            failed = 1;
        }
    }

    free(summary);
    free(spec);
    return failed;
}


static int gs_validate_train_only_contract(
    const char *source_directory,
    const char *artifact_directory
)
{
    char *source_spec = NULL;
    char *pca_spec = NULL;
    int failed;

    if (gs_read_text(
            source_directory,
            "feature_spec.txt",
            &source_spec
        ) != 0 ||
        gs_read_text(
            artifact_directory,
            "pca_spec.txt",
            &pca_spec
        ) != 0) {

        free(pca_spec);
        free(source_spec);
        return 1;
    }

    failed =
        !gs_text_has_line(
            source_spec,
            "standardization_fit=train_only"
        ) ||
        !gs_text_has_line(source_spec, "pca=false") ||
        !gs_text_has_line(pca_spec, "pca_fit=train_only") ||
        !gs_text_has_line(
            pca_spec,
            "validation_transformed_without_refit=true"
        ) ||
        !gs_text_has_line(pca_spec, "labels_used=false") ||
        !gs_text_has_line(pca_spec, "whitening=false");

    free(pca_spec);
    free(source_spec);
    return failed;
}


static void gs_print_usage(const char *program)
{
    printf(
        "Uso: %s <source_no_pca_directory> "
        "<pca_artifact_directory> <code_commit>\n"
        "Valida modello, score e contratto train-only degli artefatti PCA.\n",
        program
    );
}


int main(int argc, char **argv)
{
    GSPCAThresholds thresholds;
    GSBaselinePCA model = {0};
    GSPCAExpectedFile files[GS_PCA_TEST_ARTIFACT_COUNT] = {
        {"pca_spec.txt", 0},
        {"mean_f64.bin", 0},
        {"components_f64.bin", 0},
        {"eigenvalues_f64.bin", 0},
        {"explained_variance_ratio_f64.bin", 0},
        {"cumulative_explained_variance_ratio_f64.bin", 0},
        {"thresholds.csv", 0},
        {"spectrum.csv", 0},
        {"train_scores_pca90_f32.bin", 0},
        {"val_scores_pca90_f32.bin", 0},
        {"train_scores_pca95_f32.bin", 0},
        {"val_scores_pca95_f32.bin", 0},
        {"train_scores_pca99_f32.bin", 0},
        {"val_scores_pca99_f32.bin", 0},
        {"summary.txt", 0}
    };
    float *train90 = NULL;
    float *validation90 = NULL;
    float *train95 = NULL;
    float *validation95 = NULL;
    float *train99 = NULL;
    float *validation99 = NULL;
    int rc = EXIT_FAILURE;

    if (argc == 2 && strcmp(argv[1], "--help") == 0) {
        gs_print_usage(argv[0]);
        return EXIT_SUCCESS;
    }
    if (argc != 4) {
        gs_print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    if (!gs_platform_is_valid() ||
        !gs_metadata_value_is_valid(argv[1]) ||
        !gs_metadata_value_is_valid(argv[2]) ||
        !gs_commit_is_valid(argv[3])) {

        fprintf(stderr, "Errore: argomenti o piattaforma non validi.\n");
        return EXIT_FAILURE;
    }

    if (gs_read_thresholds(argv[2], &thresholds) != 0 ||
        gs_validate_expected_files(argv[2], &thresholds, files) != 0) {

        fprintf(stderr, "[FAIL] expected_files_and_sizes\n");
        goto cleanup;
    }
    printf("[PASS] expected_files_and_sizes\n");

    if (!(thresholds.d90 <= thresholds.d95 &&
          thresholds.d95 <= thresholds.d99 &&
          thresholds.d99 <= GS_PCA_TEST_INPUT_DIM)) {

        fprintf(stderr, "[FAIL] thresholds_contract\n");
        goto cleanup;
    }
    printf("[PASS] thresholds_contract\n");

    if (gs_load_model(argv[2], &thresholds, &model) != 0 ||
        gs_validate_model(&model, &thresholds) != 0) {

        fprintf(stderr, "[FAIL] model_contract\n");
        goto cleanup;
    }
    printf("[PASS] model_contract\n");

    if (gs_validate_spectrum(argv[2], &model) != 0) {
        fprintf(stderr, "[FAIL] spectrum_contract\n");
        goto cleanup;
    }
    printf("[PASS] spectrum_contract\n");

    if (gs_read_scores(
            argv[2],
            "train_scores_pca90_f32.bin",
            GS_PCA_TEST_TRAIN_ROWS,
            thresholds.d90,
            &train90
        ) != 0 ||
        gs_read_scores(
            argv[2],
            "val_scores_pca90_f32.bin",
            GS_PCA_TEST_VALIDATION_ROWS,
            thresholds.d90,
            &validation90
        ) != 0 ||
        gs_read_scores(
            argv[2],
            "train_scores_pca95_f32.bin",
            GS_PCA_TEST_TRAIN_ROWS,
            thresholds.d95,
            &train95
        ) != 0 ||
        gs_read_scores(
            argv[2],
            "val_scores_pca95_f32.bin",
            GS_PCA_TEST_VALIDATION_ROWS,
            thresholds.d95,
            &validation95
        ) != 0 ||
        gs_read_scores(
            argv[2],
            "train_scores_pca99_f32.bin",
            GS_PCA_TEST_TRAIN_ROWS,
            thresholds.d99,
            &train99
        ) != 0 ||
        gs_read_scores(
            argv[2],
            "val_scores_pca99_f32.bin",
            GS_PCA_TEST_VALIDATION_ROWS,
            thresholds.d99,
            &validation99
        ) != 0 ||
        gs_validate_score_prefix(
            train90,
            train99,
            GS_PCA_TEST_TRAIN_ROWS,
            thresholds.d90,
            thresholds.d99
        ) != 0 ||
        gs_validate_score_prefix(
            validation90,
            validation99,
            GS_PCA_TEST_VALIDATION_ROWS,
            thresholds.d90,
            thresholds.d99
        ) != 0 ||
        gs_validate_score_prefix(
            train95,
            train99,
            GS_PCA_TEST_TRAIN_ROWS,
            thresholds.d95,
            thresholds.d99
        ) != 0 ||
        gs_validate_score_prefix(
            validation95,
            validation99,
            GS_PCA_TEST_VALIDATION_ROWS,
            thresholds.d95,
            thresholds.d99
        ) != 0) {

        fprintf(stderr, "[FAIL] score_prefix_contract\n");
        goto cleanup;
    }
    printf("[PASS] score_prefix_contract\n");

    if (gs_validate_recomputed_scores(
            argv[1],
            "train_features_std_f32.bin",
            GS_PCA_TEST_TRAIN_ROWS,
            &model,
            train99
        ) != 0) {

        fprintf(stderr, "[FAIL] recomputed_train_scores\n");
        goto cleanup;
    }
    printf("[PASS] recomputed_train_scores\n");

    if (gs_validate_recomputed_scores(
            argv[1],
            "val_features_std_f32.bin",
            GS_PCA_TEST_VALIDATION_ROWS,
            &model,
            validation99
        ) != 0) {

        fprintf(stderr, "[FAIL] recomputed_validation_scores\n");
        goto cleanup;
    }
    printf("[PASS] recomputed_validation_scores\n");

    if (gs_validate_train_score_statistics(
            train99,
            thresholds.d99,
            model.eigenvalues
        ) != 0) {

        fprintf(stderr, "[FAIL] train_score_statistics\n");
        goto cleanup;
    }
    printf("[PASS] train_score_statistics\n");

    if (gs_validate_metadata(argv[1], argv[2], argv[3], files) != 0) {
        fprintf(stderr, "[FAIL] metadata_contract\n");
        goto cleanup;
    }
    printf("[PASS] metadata_contract\n");

    if (gs_validate_train_only_contract(argv[1], argv[2]) != 0) {
        fprintf(stderr, "[FAIL] train_only_pca_contract\n");
        goto cleanup;
    }
    printf("[PASS] train_only_pca_contract\n");

    printf("ALL PCA ARTIFACT TESTS PASSED\n");
    rc = EXIT_SUCCESS;

cleanup:
    free(validation99);
    free(train99);
    free(validation95);
    free(train95);
    free(validation90);
    free(train90);
    gs_baseline_pca_free(&model);
    return rc;
}
