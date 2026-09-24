#define main gs112_embedded_builder_entry_point
#include "../tools/build_baseline_mini112_pca_artifacts.c"
#undef main


typedef struct {
    size_t d90;
    size_t d95;
    size_t d99;
    double cumulative90;
    double cumulative95;
    double cumulative99;
} GS112PCAThresholds;


static int gs112_test_validate_top_level_inventory(
    const char *artifact_directory,
    const GS112PCAProfile *profile
)
{
    unsigned char seen[GS112_PCA_OUTPUT_FILE_COUNT] = {0};
    int seen_kmeans = 0;
    int seen_evaluation = 0;
    DIR *stream;
    struct dirent *entry;
    int failed = 0;
    size_t index;

    if (artifact_directory == NULL || profile == NULL) {
        return 1;
    }
    stream = opendir(artifact_directory);
    if (stream == NULL) {
        return 1;
    }
    while (!failed) {
        char path[GS112_PCA_PATH_CAPACITY];
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
        if (gs112_path_join(
                artifact_directory,
                entry->d_name,
                path,
                sizeof(path)
            ) != 0 ||
            stat(path, &information) != 0) {

            failed = 1;
            break;
        }
        if (gs112_output_name_index(
                entry->d_name,
                profile,
                &index
            ) == 0) {

            if (seen[index] || !S_ISREG(information.st_mode)) {
                failed = 1;
                break;
            }
            seen[index] = 1u;
        } else if (strcmp(entry->d_name, "kmeans_seed123") == 0) {
            if (seen_kmeans || !S_ISDIR(information.st_mode)) {
                failed = 1;
                break;
            }
            seen_kmeans = 1;
        } else if (strcmp(
                       entry->d_name,
                       "label_evaluation_seed123"
                   ) == 0) {

            if (seen_evaluation || !S_ISDIR(information.st_mode)) {
                failed = 1;
                break;
            }
            seen_evaluation = 1;
        } else {
            failed = 1;
            break;
        }
    }
    if (closedir(stream) != 0) {
        failed = 1;
    }
    for (index = 0;
         !failed && index < GS112_PCA_OUTPUT_FILE_COUNT;
         index++) {

        if (!seen[index]) {
            failed = 1;
        }
    }
    return failed;
}


static int gs112_test_artifact_path_is_valid(
    const char *artifact_directory,
    const GS112PCAProfile *profile
)
{
    char partial_path[GS112_PCA_PATH_CAPACITY];
    const char *basename;
    size_t length;
    int is_directory = 0;

    if (!gs112_metadata_value_is_valid(artifact_directory) ||
        profile == NULL) {

        return 1;
    }
    basename = gs112_path_basename(artifact_directory);
    if (basename == NULL ||
        strcmp(basename, profile->final_basename) != 0 ||
        gs112_path_state(artifact_directory, &is_directory) != 1 ||
        !is_directory) {

        return 1;
    }
    length = strlen(artifact_directory);
    if (length > SIZE_MAX - sizeof(".partial") ||
        length + sizeof(".partial") > sizeof(partial_path)) {

        return 1;
    }
    memcpy(partial_path, artifact_directory, length);
    memcpy(
        partial_path + length,
        ".partial",
        sizeof(".partial")
    );
    return gs112_path_state(partial_path, NULL) == 0 ? 0 : 1;
}


static int gs112_test_read_binary(
    const char *directory,
    const char *name,
    void *values,
    size_t element_size,
    size_t count
)
{
    char path[GS112_PCA_PATH_CAPACITY];
    FILE *file;
    size_t expected_bytes;
    size_t actual_bytes;
    int failed = 0;

    if (values == NULL ||
        gs112_checked_product(element_size, count, &expected_bytes) != 0 ||
        gs112_path_join(directory, name, path, sizeof(path)) != 0 ||
        gs112_regular_file_size(path, &actual_bytes) != 0 ||
        actual_bytes != expected_bytes) {

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


static int gs112_test_parse_size(const char *text, size_t *value)
{
    return gs112_parse_decimal_size(text, value);
}


static int gs112_test_parse_double(const char *text, double *value)
{
    char *end = NULL;
    double parsed;

    if (text == NULL || value == NULL || text[0] == '\0') {
        return 1;
    }
    errno = 0;
    parsed = strtod(text, &end);
    if (errno == ERANGE || end == text || *end != '\0' ||
        !isfinite(parsed)) {

        return 1;
    }
    *value = parsed;
    return 0;
}


static int gs112_test_split_three(char *line, char *field[3])
{
    char *first;
    char *second;

    if (line == NULL || field == NULL) {
        return 1;
    }
    first = strchr(line, ',');
    if (first == NULL) {
        return 1;
    }
    *first = '\0';
    second = strchr(first + 1, ',');
    if (second == NULL || strchr(second + 1, ',') != NULL) {
        return 1;
    }
    *second = '\0';
    field[0] = line;
    field[1] = first + 1;
    field[2] = second + 1;
    return 0;
}


static int gs112_test_split_four(char *line, char *field[4])
{
    size_t count = 1;
    char *cursor;

    if (line == NULL || field == NULL) {
        return 1;
    }
    field[0] = line;
    for (cursor = line; *cursor != '\0'; cursor++) {
        if (*cursor == ',' && count < 4u) {
            *cursor = '\0';
            field[count++] = cursor + 1;
        } else if (*cursor == ',') {
            return 1;
        }
    }
    return count == 4u ? 0 : 1;
}


static int gs112_test_read_thresholds(
    const char *directory,
    GS112PCAThresholds *thresholds
)
{
    static const char *const expected_thresholds[3] = {
        "0.90", "0.95", "0.99"
    };
    char path[GS112_PCA_PATH_CAPACITY];
    char line[GS112_PCA_LINE_CAPACITY];
    FILE *file;
    size_t counts[3];
    double cumulative[3];
    size_t index;
    int failed = 0;

    if (thresholds == NULL ||
        gs112_path_join(
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
        gs112_strip_line_ending(line);
        if (strcmp(
                line,
                "threshold,component_count,cumulative_variance"
            ) != 0) {

            failed = 1;
        }
    }
    for (index = 0; !failed && index < 3u; index++) {
        char *field[3];

        if (fgets(line, sizeof(line), file) == NULL) {
            failed = 1;
            break;
        }
        gs112_strip_line_ending(line);
        if (gs112_test_split_three(line, field) != 0 ||
            strcmp(field[0], expected_thresholds[index]) != 0 ||
            gs112_test_parse_size(field[1], &counts[index]) != 0 ||
            gs112_test_parse_double(field[2], &cumulative[index]) != 0) {

            failed = 1;
        }
    }
    if (!failed && fgets(line, sizeof(line), file) != NULL) {
        failed = 1;
    }
    if (ferror(file) || fclose(file) != 0) {
        failed = 1;
    }
    if (failed) {
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


static int gs112_test_read_key_double(
    const char *text,
    const char *key,
    double *value
)
{
    const char *cursor = text;
    size_t key_length;

    if (text == NULL || key == NULL || value == NULL) {
        return 1;
    }
    key_length = strlen(key);
    while ((cursor = strstr(cursor, key)) != NULL) {
        if ((cursor == text || cursor[-1] == '\n') &&
            cursor[key_length] == '=') {

            char buffer[128];
            const char *start = cursor + key_length + 1u;
            const char *end = strchr(start, '\n');
            size_t length = end == NULL
                ? strlen(start)
                : (size_t)(end - start);

            if (length != 0 && start[length - 1u] == '\r') {
                length--;
            }
            if (length == 0 || length >= sizeof(buffer)) {
                return 1;
            }
            memcpy(buffer, start, length);
            buffer[length] = '\0';
            return gs112_test_parse_double(buffer, value);
        }
        cursor++;
    }
    return 1;
}


static int gs112_test_validate_metadata(
    const char *artifact_directory,
    const GS112PCAProfile *profile,
    const char *expected_commit,
    size_t retained_components,
    const GS112PCAThresholds *thresholds,
    const char source_manifest_hash[65],
    double *total_variance
)
{
    char spec_path[GS112_PCA_PATH_CAPACITY];
    char summary_path[GS112_PCA_PATH_CAPACITY];
    char line[512];
    char *spec = NULL;
    char *summary = NULL;
    const char *const common_required[] = {
        "source_profile=mini112_pool7_no_pca",
        "covariance_ddof=0",
        "eigensolver=lapacke_dsyevd",
        "threading_policy=external_single_thread",
        "sign_convention=max_abs_loading_positive",
        "whitening=false",
        "post_standardization=false",
        "augmentation=false",
        "labels_used=false"
    };
    size_t index;
    int failed = 0;

    if (gs112_path_join(
            artifact_directory,
            "pca_spec.txt",
            spec_path,
            sizeof(spec_path)
        ) != 0 ||
        gs112_path_join(
            artifact_directory,
            "summary.txt",
            summary_path,
            sizeof(summary_path)
        ) != 0 ||
        gs112_read_text(spec_path, 1024u * 1024u, &spec) != 0 ||
        gs112_read_text(summary_path, 1024u * 1024u, &summary) != 0) {

        free(summary);
        free(spec);
        return 1;
    }
    for (index = 0;
         index < sizeof(common_required) / sizeof(common_required[0]);
         index++) {

        if (!gs112_text_has_line(spec, common_required[index]) ||
            !gs112_text_has_line(summary, common_required[index])) {

            failed = 1;
        }
    }

#define GS112_REQUIRE_SPEC_FORMAT(...) \
    do { \
        int gs112_written = snprintf(line, sizeof(line), __VA_ARGS__); \
        if (gs112_written < 0 || \
            (size_t)gs112_written >= sizeof(line) || \
            !gs112_text_has_line(spec, line)) { \
            failed = 1; \
        } \
    } while (0)

#define GS112_REQUIRE_SUMMARY_FORMAT(...) \
    do { \
        int gs112_written = snprintf(line, sizeof(line), __VA_ARGS__); \
        if (gs112_written < 0 || \
            (size_t)gs112_written >= sizeof(line) || \
            !gs112_text_has_line(summary, line)) { \
            failed = 1; \
        } \
    } while (0)

    GS112_REQUIRE_SPEC_FORMAT("profile=%s", profile->profile);
    GS112_REQUIRE_SPEC_FORMAT(
        "selected_threshold=%.2f",
        profile->threshold
    );
    GS112_REQUIRE_SPEC_FORMAT(
        "retained_component_count=%lu",
        (unsigned long)retained_components
    );
    GS112_REQUIRE_SPEC_FORMAT(
        "d90=%lu",
        (unsigned long)thresholds->d90
    );
    GS112_REQUIRE_SPEC_FORMAT(
        "d95=%lu",
        (unsigned long)thresholds->d95
    );
    GS112_REQUIRE_SPEC_FORMAT(
        "d99=%lu",
        (unsigned long)thresholds->d99
    );
    GS112_REQUIRE_SPEC_FORMAT(
        "cumulative_variance_at_retained_count=%.17g",
        retained_components == thresholds->d90
            ? thresholds->cumulative90
            : retained_components == thresholds->d95
                ? thresholds->cumulative95
                : thresholds->cumulative99
    );
    GS112_REQUIRE_SPEC_FORMAT("original_feature_count=1024");
    GS112_REQUIRE_SPEC_FORMAT("train_rows=6866");
    GS112_REQUIRE_SPEC_FORMAT("validation_rows=1717");
    GS112_REQUIRE_SPEC_FORMAT("input_dtype=float32");
    GS112_REQUIRE_SPEC_FORMAT("model_dtype=float64");
    GS112_REQUIRE_SPEC_FORMAT("score_dtype=float32");
    GS112_REQUIRE_SPEC_FORMAT("input_order=row_major");
    GS112_REQUIRE_SPEC_FORMAT("component_order=component_major");
    GS112_REQUIRE_SPEC_FORMAT("score_order=row_major");
    GS112_REQUIRE_SPEC_FORMAT("fit_input=train_only");
    GS112_REQUIRE_SPEC_FORMAT(
        "validation_transform_without_refit=true"
    );
    GS112_REQUIRE_SPEC_FORMAT("split_seed=123");

    GS112_REQUIRE_SUMMARY_FORMAT("profile=%s", profile->profile);
    GS112_REQUIRE_SUMMARY_FORMAT(
        "code_commit=%s",
        expected_commit
    );
    GS112_REQUIRE_SUMMARY_FORMAT(
        "source_sha256_manifest_hash=%s",
        source_manifest_hash
    );
    GS112_REQUIRE_SUMMARY_FORMAT(
        "retained_component_count=%lu",
        (unsigned long)retained_components
    );
    GS112_REQUIRE_SUMMARY_FORMAT(
        "d90=%lu",
        (unsigned long)thresholds->d90
    );
    GS112_REQUIRE_SUMMARY_FORMAT(
        "d95=%lu",
        (unsigned long)thresholds->d95
    );
    GS112_REQUIRE_SUMMARY_FORMAT(
        "d99=%lu",
        (unsigned long)thresholds->d99
    );
    GS112_REQUIRE_SUMMARY_FORMAT(
        "cumulative_variance_at_retained_count=%.17g",
        retained_components == thresholds->d90
            ? thresholds->cumulative90
            : retained_components == thresholds->d95
                ? thresholds->cumulative95
                : thresholds->cumulative99
    );
    GS112_REQUIRE_SUMMARY_FORMAT("pca_fit=train_only");
    GS112_REQUIRE_SUMMARY_FORMAT(
        "validation_transformed_without_refit=true"
    );
    GS112_REQUIRE_SUMMARY_FORMAT(
        "model_unchanged_after_validation=true"
    );
    GS112_REQUIRE_SUMMARY_FORMAT("train_input_unchanged=true");
    GS112_REQUIRE_SUMMARY_FORMAT("validation_input_unchanged=true");
    GS112_REQUIRE_SUMMARY_FORMAT("scores_finite=true");

#undef GS112_REQUIRE_SUMMARY_FORMAT
#undef GS112_REQUIRE_SPEC_FORMAT

    if (strstr(spec, "whitening=true") != NULL ||
        strstr(spec, "post_standardization=true") != NULL ||
        strstr(spec, "augmentation=true") != NULL ||
        strstr(spec, "labels_used=true") != NULL ||
        strstr(summary, "whitening=true") != NULL ||
        strstr(summary, "post_standardization=true") != NULL ||
        strstr(summary, "augmentation=true") != NULL ||
        strstr(summary, "labels_used=true") != NULL ||
        gs112_test_read_key_double(
            summary,
            "total_variance",
            total_variance
        ) != 0) {

        failed = 1;
    }
    free(summary);
    free(spec);
    return failed;
}


static int gs112_test_load_model(
    const char *artifact_directory,
    const GS112PCAThresholds *thresholds,
    double total_variance,
    GSBaselinePCA *model
)
{
    size_t matrix_count;
    size_t vector_bytes;
    size_t matrix_bytes;

    if (artifact_directory == NULL || thresholds == NULL ||
        model == NULL || !isfinite(total_variance) ||
        !(total_variance > 0.0) ||
        gs112_checked_product(
            GS112_PCA_FEATURES,
            GS112_PCA_FEATURES,
            &matrix_count
        ) != 0 ||
        gs112_checked_product(
            GS112_PCA_FEATURES,
            sizeof(double),
            &vector_bytes
        ) != 0 ||
        gs112_checked_product(
            matrix_count,
            sizeof(double),
            &matrix_bytes
        ) != 0) {

        return 1;
    }
    model->input_dim = GS112_PCA_FEATURES;
    model->component_count = GS112_PCA_FEATURES;
    model->total_variance = total_variance;
    model->components_90 = thresholds->d90;
    model->components_95 = thresholds->d95;
    model->components_99 = thresholds->d99;
    model->mean = (double *)malloc(vector_bytes);
    model->components = (double *)malloc(matrix_bytes);
    model->eigenvalues = (double *)malloc(vector_bytes);
    model->explained_variance_ratio =
        (double *)malloc(vector_bytes);
    model->cumulative_explained_variance_ratio =
        (double *)malloc(vector_bytes);
    if (model->mean == NULL || model->components == NULL ||
        model->eigenvalues == NULL ||
        model->explained_variance_ratio == NULL ||
        model->cumulative_explained_variance_ratio == NULL ||
        gs112_test_read_binary(
            artifact_directory,
            "mean_f64.bin",
            model->mean,
            sizeof(double),
            GS112_PCA_FEATURES
        ) != 0 ||
        gs112_test_read_binary(
            artifact_directory,
            "components_f64.bin",
            model->components,
            sizeof(double),
            matrix_count
        ) != 0 ||
        gs112_test_read_binary(
            artifact_directory,
            "eigenvalues_f64.bin",
            model->eigenvalues,
            sizeof(double),
            GS112_PCA_FEATURES
        ) != 0 ||
        gs112_test_read_binary(
            artifact_directory,
            "explained_variance_ratio_f64.bin",
            model->explained_variance_ratio,
            sizeof(double),
            GS112_PCA_FEATURES
        ) != 0 ||
        gs112_test_read_binary(
            artifact_directory,
            "cumulative_explained_variance_ratio_f64.bin",
            model->cumulative_explained_variance_ratio,
            sizeof(double),
            GS112_PCA_FEATURES
        ) != 0) {

        gs_baseline_pca_free(model);
        return 1;
    }
    return 0;
}


static int gs112_test_validate_spectrum(
    const char *artifact_directory,
    const GSBaselinePCA *model
)
{
    char path[GS112_PCA_PATH_CAPACITY];
    char line[GS112_PCA_LINE_CAPACITY];
    FILE *file;
    size_t component;
    int failed = 0;

    if (gs112_path_join(
            artifact_directory,
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
        gs112_strip_line_ending(line);
        if (strcmp(
                line,
                "component_index,eigenvalue,explained_variance_ratio,"
                "cumulative_explained_variance_ratio"
            ) != 0) {

            failed = 1;
        }
    }
    for (component = 0;
         !failed && component < GS112_PCA_FEATURES;
         component++) {

        char *field[4];
        size_t parsed_component;
        double eigenvalue;
        double ratio;
        double cumulative;

        if (fgets(line, sizeof(line), file) == NULL) {
            failed = 1;
            break;
        }
        gs112_strip_line_ending(line);
        if (gs112_test_split_four(line, field) != 0 ||
            gs112_test_parse_size(field[0], &parsed_component) != 0 ||
            gs112_test_parse_double(field[1], &eigenvalue) != 0 ||
            gs112_test_parse_double(field[2], &ratio) != 0 ||
            gs112_test_parse_double(field[3], &cumulative) != 0 ||
            parsed_component != component ||
            memcmp(
                &eigenvalue,
                &model->eigenvalues[component],
                sizeof(double)
            ) != 0 ||
            memcmp(
                &ratio,
                &model->explained_variance_ratio[component],
                sizeof(double)
            ) != 0 ||
            memcmp(
                &cumulative,
                &model->cumulative_explained_variance_ratio[component],
                sizeof(double)
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


static int gs112_test_model_numeric_contract(
    const GSBaselinePCA *model,
    const GS112PCAThresholds *thresholds
)
{
    size_t component;
    size_t other_component;
    size_t feature;
    size_t zero_count;
    const double tolerance =
        1024.0 * DBL_EPSILON * (double)GS112_PCA_FEATURES;

    if (model == NULL || thresholds == NULL ||
        gs112_model_is_valid(model, &zero_count) != 0 ||
        model->components_90 != thresholds->d90 ||
        model->components_95 != thresholds->d95 ||
        model->components_99 != thresholds->d99 ||
        memcmp(
            &model->cumulative_explained_variance_ratio[
                model->components_90 - 1u
            ],
            &thresholds->cumulative90,
            sizeof(double)
        ) != 0 ||
        memcmp(
            &model->cumulative_explained_variance_ratio[
                model->components_95 - 1u
            ],
            &thresholds->cumulative95,
            sizeof(double)
        ) != 0 ||
        memcmp(
            &model->cumulative_explained_variance_ratio[
                model->components_99 - 1u
            ],
            &thresholds->cumulative99,
            sizeof(double)
        ) != 0) {

        return 1;
    }
    for (component = 0;
         component < GS112_PCA_FEATURES;
         component++) {

        for (other_component = component;
             other_component < GS112_PCA_FEATURES;
             other_component++) {

            double dot = 0.0;
            double expected =
                component == other_component ? 1.0 : 0.0;

            for (feature = 0;
                 feature < GS112_PCA_FEATURES;
                 feature++) {

                size_t left =
                    component * (size_t)GS112_PCA_FEATURES + feature;
                size_t right =
                    other_component *
                        (size_t)GS112_PCA_FEATURES +
                    feature;

                dot +=
                    model->components[left] *
                    model->components[right];
            }
            if (!isfinite(dot) ||
                fabs(dot - expected) > tolerance) {

                return 1;
            }
        }
    }
    return 0;
}


static int gs112_test_mean_and_covariance(
    const float *train,
    const GSBaselinePCA *model,
    double *maximum_error,
    double *tolerance_used
)
{
    double *mean = NULL;
    double *centered = NULL;
    double *covariance = NULL;
    size_t train_count;
    size_t matrix_count;
    size_t centered_bytes;
    size_t matrix_bytes;
    size_t row;
    size_t feature;
    size_t other_feature;
    double scale = 0.0;
    double max_error = 0.0;
    double tolerance;
    int failed = 0;

    if (train == NULL || model == NULL ||
        maximum_error == NULL || tolerance_used == NULL ||
        gs112_checked_product(
            GS112_PCA_TRAIN_ROWS,
            GS112_PCA_FEATURES,
            &train_count
        ) != 0 ||
        gs112_checked_product(
            GS112_PCA_FEATURES,
            GS112_PCA_FEATURES,
            &matrix_count
        ) != 0 ||
        gs112_checked_product(
            train_count,
            sizeof(double),
            &centered_bytes
        ) != 0 ||
        gs112_checked_product(
            matrix_count,
            sizeof(double),
            &matrix_bytes
        ) != 0) {

        return 1;
    }
    mean = (double *)calloc(GS112_PCA_FEATURES, sizeof(double));
    centered = (double *)malloc(centered_bytes);
    covariance = (double *)calloc(matrix_count, sizeof(double));
    if (mean == NULL || centered == NULL || covariance == NULL) {
        failed = 1;
        goto cleanup;
    }

    for (row = 0; row < GS112_PCA_TRAIN_ROWS; row++) {
        for (feature = 0; feature < GS112_PCA_FEATURES; feature++) {
            size_t offset =
                row * (size_t)GS112_PCA_FEATURES + feature;

            mean[feature] += (double)train[offset];
        }
    }
    for (feature = 0; feature < GS112_PCA_FEATURES; feature++) {
        mean[feature] /= (double)GS112_PCA_TRAIN_ROWS;
    }
    if (memcmp(
            mean,
            model->mean,
            GS112_PCA_FEATURES * sizeof(double)
        ) != 0) {

        failed = 1;
        goto cleanup;
    }
    for (row = 0; row < GS112_PCA_TRAIN_ROWS; row++) {
        for (feature = 0; feature < GS112_PCA_FEATURES; feature++) {
            size_t offset =
                row * (size_t)GS112_PCA_FEATURES + feature;

            centered[offset] = (double)train[offset] - mean[feature];
        }
    }
    for (row = 0; row < GS112_PCA_TRAIN_ROWS; row++) {
        const double *centered_row =
            centered + row * (size_t)GS112_PCA_FEATURES;

        for (feature = 0; feature < GS112_PCA_FEATURES; feature++) {
            double left = centered_row[feature];

            for (other_feature = feature;
                 other_feature < GS112_PCA_FEATURES;
                 other_feature++) {

                size_t offset =
                    feature * (size_t)GS112_PCA_FEATURES +
                    other_feature;

                covariance[offset] +=
                    left * centered_row[other_feature];
            }
        }
    }
    for (feature = 0; feature < GS112_PCA_FEATURES; feature++) {
        for (other_feature = feature;
             other_feature < GS112_PCA_FEATURES;
             other_feature++) {

            size_t offset =
                feature * (size_t)GS112_PCA_FEATURES + other_feature;
            double value =
                covariance[offset] / (double)GS112_PCA_TRAIN_ROWS;

            covariance[offset] = value;
            covariance[
                other_feature * (size_t)GS112_PCA_FEATURES + feature
            ] = value;
            if (fabs(value) > scale) {
                scale = fabs(value);
            }
        }
    }

    tolerance =
        32768.0 * DBL_EPSILON *
        (double)GS112_PCA_FEATURES * (1.0 + scale);
    for (feature = 0; feature < GS112_PCA_FEATURES; feature++) {
        for (other_feature = feature;
             other_feature < GS112_PCA_FEATURES;
             other_feature++) {

            size_t component;
            double reconstructed = 0.0;
            double difference;

            for (component = 0;
                 component < GS112_PCA_FEATURES;
                 component++) {

                reconstructed +=
                    model->eigenvalues[component] *
                    model->components[
                        component * (size_t)GS112_PCA_FEATURES +
                        feature
                    ] *
                    model->components[
                        component * (size_t)GS112_PCA_FEATURES +
                        other_feature
                    ];
            }
            difference = fabs(
                reconstructed -
                covariance[
                    feature * (size_t)GS112_PCA_FEATURES +
                    other_feature
                ]
            );
            if (!isfinite(reconstructed) || !isfinite(difference)) {
                failed = 1;
                goto cleanup;
            }
            if (difference > max_error) {
                max_error = difference;
            }
        }
    }
    if (max_error > tolerance) {
        failed = 1;
    }
    *maximum_error = max_error;
    *tolerance_used = tolerance;

cleanup:
    free(covariance);
    free(centered);
    free(mean);
    return failed;
}


static int gs112_test_models_bitwise_equal(
    const GSBaselinePCA *left,
    const GSBaselinePCA *right
)
{
    size_t matrix_count;

    if (left == NULL || right == NULL ||
        gs112_checked_product(
            GS112_PCA_FEATURES,
            GS112_PCA_FEATURES,
            &matrix_count
        ) != 0 ||
        left->input_dim != right->input_dim ||
        left->component_count != right->component_count ||
        left->components_90 != right->components_90 ||
        left->components_95 != right->components_95 ||
        left->components_99 != right->components_99 ||
        memcmp(
            &left->total_variance,
            &right->total_variance,
            sizeof(double)
        ) != 0 ||
        memcmp(
            left->mean,
            right->mean,
            GS112_PCA_FEATURES * sizeof(double)
        ) != 0 ||
        memcmp(
            left->components,
            right->components,
            matrix_count * sizeof(double)
        ) != 0 ||
        memcmp(
            left->eigenvalues,
            right->eigenvalues,
            GS112_PCA_FEATURES * sizeof(double)
        ) != 0 ||
        memcmp(
            left->explained_variance_ratio,
            right->explained_variance_ratio,
            GS112_PCA_FEATURES * sizeof(double)
        ) != 0 ||
        memcmp(
            left->cumulative_explained_variance_ratio,
            right->cumulative_explained_variance_ratio,
            GS112_PCA_FEATURES * sizeof(double)
        ) != 0) {

        return 0;
    }
    return 1;
}


static int gs112_test_scores_and_model_invariance(
    const char *artifact_directory,
    const GS112PCAProfile *profile,
    const float *train,
    const float *validation,
    size_t retained_components,
    GSBaselinePCA *refitted_model
)
{
    float *stored_train = NULL;
    float *stored_validation = NULL;
    float *full_train = NULL;
    float *full_validation = NULL;
    double *snapshot = NULL;
    GS112PCAModelMetadata metadata;
    size_t stored_train_count;
    size_t stored_validation_count;
    size_t full_train_count;
    size_t full_validation_count;
    size_t stored_train_bytes;
    size_t stored_validation_bytes;
    size_t full_train_bytes;
    size_t full_validation_bytes;
    size_t matrix_count;
    size_t snapshot_count;
    size_t snapshot_bytes;
    size_t row;
    int failed = 0;

    if (gs112_checked_product(
            GS112_PCA_TRAIN_ROWS,
            retained_components,
            &stored_train_count
        ) != 0 ||
        gs112_checked_product(
            GS112_PCA_VALIDATION_ROWS,
            retained_components,
            &stored_validation_count
        ) != 0 ||
        gs112_checked_product(
            GS112_PCA_TRAIN_ROWS,
            GS112_PCA_FEATURES,
            &full_train_count
        ) != 0 ||
        gs112_checked_product(
            GS112_PCA_VALIDATION_ROWS,
            GS112_PCA_FEATURES,
            &full_validation_count
        ) != 0 ||
        gs112_checked_product(
            stored_train_count,
            sizeof(float),
            &stored_train_bytes
        ) != 0 ||
        gs112_checked_product(
            stored_validation_count,
            sizeof(float),
            &stored_validation_bytes
        ) != 0 ||
        gs112_checked_product(
            full_train_count,
            sizeof(float),
            &full_train_bytes
        ) != 0 ||
        gs112_checked_product(
            full_validation_count,
            sizeof(float),
            &full_validation_bytes
        ) != 0 ||
        gs112_checked_product(
            GS112_PCA_FEATURES,
            GS112_PCA_FEATURES,
            &matrix_count
        ) != 0 ||
        gs112_checked_sum(
            matrix_count,
            4u * (size_t)GS112_PCA_FEATURES,
            &snapshot_count
        ) != 0 ||
        gs112_checked_product(
            snapshot_count,
            sizeof(double),
            &snapshot_bytes
        ) != 0) {

        return 1;
    }

    stored_train = (float *)malloc(stored_train_bytes);
    stored_validation = (float *)malloc(stored_validation_bytes);
    full_train = (float *)malloc(full_train_bytes);
    full_validation = (float *)malloc(full_validation_bytes);
    snapshot = (double *)malloc(snapshot_bytes);
    if (stored_train == NULL || stored_validation == NULL ||
        full_train == NULL || full_validation == NULL ||
        snapshot == NULL ||
        gs112_test_read_binary(
            artifact_directory,
            profile->train_score_name,
            stored_train,
            sizeof(float),
            stored_train_count
        ) != 0 ||
        gs112_test_read_binary(
            artifact_directory,
            profile->validation_score_name,
            stored_validation,
            sizeof(float),
            stored_validation_count
        ) != 0 ||
        !gs112_scores_are_finite(
            stored_train,
            GS112_PCA_TRAIN_ROWS,
            retained_components
        ) ||
        !gs112_scores_are_finite(
            stored_validation,
            GS112_PCA_VALIDATION_ROWS,
            retained_components
        )) {

        failed = 1;
        goto cleanup;
    }

    if (gs_baseline_pca_transform(
            refitted_model,
            train,
            GS112_PCA_TRAIN_ROWS,
            GS112_PCA_FEATURES,
            GS112_PCA_FEATURES,
            full_train,
            full_train_count
        ) != GS_BASELINE_PCA_SUCCESS ||
        gs112_snapshot_model(
            refitted_model,
            snapshot,
            snapshot_count,
            &metadata
        ) != 0 ||
        gs_baseline_pca_transform(
            refitted_model,
            validation,
            GS112_PCA_VALIDATION_ROWS,
            GS112_PCA_FEATURES,
            GS112_PCA_FEATURES,
            full_validation,
            full_validation_count
        ) != GS_BASELINE_PCA_SUCCESS ||
        !gs112_model_matches_snapshot(
            refitted_model,
            snapshot,
            snapshot_count,
            &metadata
        )) {

        failed = 1;
        goto cleanup;
    }

    for (row = 0; row < GS112_PCA_TRAIN_ROWS; row++) {
        size_t full_offset =
            row * (size_t)GS112_PCA_FEATURES;
        size_t stored_offset = row * retained_components;
        size_t prefix_bytes;

        if (gs112_checked_product(
                retained_components,
                sizeof(float),
                &prefix_bytes
            ) != 0 ||
            memcmp(
                full_train + full_offset,
                stored_train + stored_offset,
                prefix_bytes
            ) != 0) {

            failed = 1;
            goto cleanup;
        }
    }
    for (row = 0; row < GS112_PCA_VALIDATION_ROWS; row++) {
        size_t full_offset =
            row * (size_t)GS112_PCA_FEATURES;
        size_t stored_offset = row * retained_components;
        size_t prefix_bytes;

        if (gs112_checked_product(
                retained_components,
                sizeof(float),
                &prefix_bytes
            ) != 0 ||
            memcmp(
                full_validation + full_offset,
                stored_validation + stored_offset,
                prefix_bytes
            ) != 0) {

            failed = 1;
            goto cleanup;
        }
    }

cleanup:
    free(snapshot);
    free(full_validation);
    free(full_train);
    free(stored_validation);
    free(stored_train);
    return failed;
}


static void gs112_test_print_usage(const char *program)
{
    printf(
        "Usage: %s <source_no_pca_directory> "
        "<pca_artifact_directory> <pca90|pca95|pca99> "
        "<expected_code_commit>\n",
        program
    );
}


int main(int argc, char **argv)
{
    GS112PCAProfile profile;
    GS112PCAThresholds thresholds;
    GSBaselinePCA stored_model = {0};
    GSBaselinePCA refitted_model = {0};
    char source_file_hashes[GS112_PCA_INPUT_COUNT][65];
    char resolved_inputs[GS112_PCA_SOURCE_INPUT_COUNT][
        GS112_PCA_PATH_CAPACITY
    ];
    char train_path[GS112_PCA_PATH_CAPACITY];
    char validation_path[GS112_PCA_PATH_CAPACITY];
    float *train = NULL;
    float *validation = NULL;
    size_t train_count;
    size_t validation_count;
    size_t train_bytes;
    size_t validation_bytes;
    size_t retained_components;
    double total_variance;
    double covariance_maximum_error = 0.0;
    double covariance_tolerance = 0.0;
    int rc = EXIT_FAILURE;

    if (argc == 2 && strcmp(argv[1], "--help") == 0) {
        gs112_test_print_usage(argv[0]);
        return EXIT_SUCCESS;
    }
    if (argc != 5) {
        gs112_test_print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    if (!gs112_platform_is_valid() ||
        !gs112_metadata_value_is_valid(argv[1]) ||
        !gs112_metadata_value_is_valid(argv[2]) ||
        gs112_profile_from_token(argv[3], &profile) != 0 ||
        !gs112_metadata_value_is_valid(argv[4]) ||
        gs112_test_artifact_path_is_valid(argv[2], &profile) != 0 ||
        gs112_checked_product(
            GS112_PCA_TRAIN_ROWS,
            GS112_PCA_FEATURES,
            &train_count
        ) != 0 ||
        gs112_checked_product(
            GS112_PCA_VALIDATION_ROWS,
            GS112_PCA_FEATURES,
            &validation_count
        ) != 0 ||
        gs112_checked_product(
            train_count,
            sizeof(float),
            &train_bytes
        ) != 0 ||
        gs112_checked_product(
            validation_count,
            sizeof(float),
            &validation_bytes
        ) != 0 ||
        train_bytes != 28123136u ||
        validation_bytes != 7032832u) {

        fprintf(stderr, "Error: invalid validator arguments or platform.\n");
        return EXIT_FAILURE;
    }

    if (gs112_validate_source_contract(
            argv[1],
            train_bytes,
            validation_bytes,
            source_file_hashes,
            resolved_inputs
        ) != 0 ||
        gs112_test_read_thresholds(argv[2], &thresholds) != 0 ||
        thresholds.d90 == 0 ||
        thresholds.d90 > thresholds.d95 ||
        thresholds.d95 > thresholds.d99 ||
        thresholds.d99 > GS112_PCA_FEATURES) {

        fprintf(stderr, "Error: source or threshold contract failed.\n");
        goto cleanup;
    }
    retained_components =
        strcmp(profile.token, "pca90") == 0 ? thresholds.d90 :
        strcmp(profile.token, "pca95") == 0 ? thresholds.d95 :
                                             thresholds.d99;

    if (gs112_test_validate_top_level_inventory(
            argv[2],
            &profile
        ) != 0 ||
        gs112_validate_output_sizes(
            argv[2],
            &profile,
            retained_components
        ) != 0 ||
        gs112_validate_pca_hashes(
            argv[1],
            argv[2],
            &profile,
            source_file_hashes
        ) != 0 ||
        gs112_test_validate_metadata(
            argv[2],
            &profile,
            argv[4],
            retained_components,
            &thresholds,
            source_file_hashes[6],
            &total_variance
        ) != 0 ||
        gs112_test_load_model(
            argv[2],
            &thresholds,
            total_variance,
            &stored_model
        ) != 0 ||
        gs112_test_model_numeric_contract(
            &stored_model,
            &thresholds
        ) != 0 ||
        gs112_test_validate_spectrum(
            argv[2],
            &stored_model
        ) != 0) {

        fprintf(stderr, "Error: PCA artifact structure failed.\n");
        goto cleanup;
    }

    if (gs112_path_join(
            argv[1],
            "train_features_std_f32.bin",
            train_path,
            sizeof(train_path)
        ) != 0 ||
        gs112_path_join(
            argv[1],
            "val_features_std_f32.bin",
            validation_path,
            sizeof(validation_path)
        ) != 0) {

        fprintf(stderr, "Error: source matrix path failed.\n");
        goto cleanup;
    }
    train = (float *)malloc(train_bytes);
    validation = (float *)malloc(validation_bytes);
    if (train == NULL || validation == NULL ||
        gs112_read_float_matrix(
            train_path,
            train,
            train_count,
            train_bytes
        ) != 0 ||
        gs112_read_float_matrix(
            validation_path,
            validation,
            validation_count,
            validation_bytes
        ) != 0) {

        fprintf(stderr, "Error: complete matrix read failed.\n");
        goto cleanup;
    }

    if (gs112_test_mean_and_covariance(
            train,
            &stored_model,
            &covariance_maximum_error,
            &covariance_tolerance
        ) != 0) {

        fprintf(stderr, "Error: independent covariance check failed.\n");
        goto cleanup;
    }
    if (gs_baseline_pca_fit(
            train,
            GS112_PCA_TRAIN_ROWS,
            GS112_PCA_FEATURES,
            &refitted_model
        ) != GS_BASELINE_PCA_SUCCESS ||
        !gs112_test_models_bitwise_equal(
            &stored_model,
            &refitted_model
        )) {

        fprintf(stderr, "Error: full train-only PCA refit differs.\n");
        goto cleanup;
    }
    if (gs112_test_scores_and_model_invariance(
            argv[2],
            &profile,
            train,
            validation,
            retained_components,
            &refitted_model
        ) != 0) {

        fprintf(stderr, "Error: complete score recomputation differs.\n");
        goto cleanup;
    }

    printf("[PASS] source_contract_and_hashes\n");
    printf("[PASS] exact_inventory_metadata_and_dimensions\n");
    printf("[PASS] complete_model_numeric_contract\n");
    printf("[PASS] independent_train_mean_and_covariance\n");
    printf("[PASS] full_train_only_pca_refit\n");
    printf("[PASS] complete_train_validation_score_recompute\n");
    printf("PCA_FULL_REFIT_BITWISE=true\n");
    printf("PCA_MEAN_RECOMPUTED_MATCH=true\n");
    printf("PCA_COVARIANCE_RECOMPUTED_MATCH=true\n");
    printf(
        "PCA_COVARIANCE_MAX_ABS_ERROR=%.17g\n",
        covariance_maximum_error
    );
    printf(
        "PCA_COVARIANCE_TOLERANCE=%.17g\n",
        covariance_tolerance
    );
    printf("PCA_COMPONENTS_ORTHONORMAL=true\n");
    printf("PCA_EIGENVALUES_ORDERED=true\n");
    printf("PCA_SIGN_CONVENTION_VALID=true\n");
    printf("PCA_THRESHOLDS_MINIMAL=true\n");
    printf("TRAIN_SCORES_RECOMPUTED_BITWISE=true\n");
    printf("VALIDATION_SCORES_RECOMPUTED_BITWISE=true\n");
    printf("PCA_MODEL_UNCHANGED_AFTER_VALIDATION=true\n");
    printf("ALL_PCA_CHECKSUMS_MATCH=true\n");
    printf("D90=%lu\n", (unsigned long)thresholds.d90);
    printf("D95=%lu\n", (unsigned long)thresholds.d95);
    printf("D99=%lu\n", (unsigned long)thresholds.d99);
    printf("CUMULATIVE90=%.17g\n", thresholds.cumulative90);
    printf("CUMULATIVE95=%.17g\n", thresholds.cumulative95);
    printf("CUMULATIVE99=%.17g\n", thresholds.cumulative99);
    printf("ALL MINI112 PCA ARTIFACT TESTS PASSED\n");
    rc = EXIT_SUCCESS;

cleanup:
    free(validation);
    free(train);
    gs_baseline_pca_free(&refitted_model);
    gs_baseline_pca_free(&stored_model);
    return rc;
}
