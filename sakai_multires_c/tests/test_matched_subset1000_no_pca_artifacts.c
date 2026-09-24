#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

#define GS_MATCHED_SUBSET1000_NO_MAIN
#include "../tools/build_matched_subset1000_no_pca_artifacts.c"

static int test_regular_non_reparse(const char *path)
{
    struct stat information;
    if (path == NULL || stat(path, &information) != 0 ||
        !S_ISREG(information.st_mode)) {
        return 0;
    }
#ifdef _WIN32
    {
        DWORD attributes = GetFileAttributesA(path);
        if (attributes == INVALID_FILE_ATTRIBUTES ||
            (attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0u) {
            return 0;
        }
    }
#endif
    return 1;
}

static int test_directory_non_reparse(const char *path)
{
    struct stat information;
    if (path == NULL || stat(path, &information) != 0 ||
        !S_ISDIR(information.st_mode)) {
        return 0;
    }
#ifdef _WIN32
    {
        DWORD attributes = GetFileAttributesA(path);
        if (attributes == INVALID_FILE_ATTRIBUTES ||
            (attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0u) {
            return 0;
        }
    }
#endif
    return 1;
}

static int test_artifact_inventory_downstream_aware(const char *directory)
{
    unsigned char seen[GS_TOTAL_OUTPUT_FILES] = {0};
    DIR *stream = NULL;
    struct dirent *entry;
    size_t index;
    int kmeans_seen = 0;
    int evaluation_seen = 0;
    int failed = 0;

    stream = opendir(directory);
    if (stream == NULL) {
        return 1;
    }
    while (!failed) {
        char path[GS_PATH_CAPACITY];
        size_t matched = GS_TOTAL_OUTPUT_FILES;

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
                directory, entry->d_name, path, sizeof(path)
            ) != 0) {
            failed = 1;
            continue;
        }
        for (index = 0u; index < GS_ARTIFACT_COUNT; index++) {
            if (strcmp(entry->d_name, gs_artifact_names[index]) == 0) {
                matched = index;
                break;
            }
        }
        if (strcmp(entry->d_name, "sha256.txt") == 0) {
            matched = GS_ARTIFACT_COUNT;
        }
        if (matched < GS_TOTAL_OUTPUT_FILES) {
            if (seen[matched] ||
                !test_regular_non_reparse(path)) {
                failed = 1;
            } else {
                seen[matched] = 1u;
            }
        } else if (strcmp(entry->d_name, "kmeans_seed123") == 0 &&
                   !kmeans_seen &&
                   test_directory_non_reparse(path)) {
            kmeans_seen = 1;
        } else if (strcmp(
                       entry->d_name, "label_evaluation_seed123"
                   ) == 0 &&
                   !evaluation_seen &&
                   test_directory_non_reparse(path)) {
            evaluation_seen = 1;
        } else {
            failed = 1;
        }
    }
    if (closedir(stream) != 0) {
        failed = 1;
    }
    for (index = 0u; !failed && index < GS_TOTAL_OUTPUT_FILES; index++) {
        if (!seen[index]) {
            failed = 1;
        }
    }
    if (!failed && evaluation_seen && !kmeans_seen) {
        failed = 1;
    }
    return failed;
}

static int test_validate_output_downstream_aware(
    const char *directory,
    const char *const input_paths[GS_INPUT_COUNT],
    size_t train_bytes,
    size_t validation_bytes
)
{
    char checksum_path[GS_PATH_CAPACITY];
    char line[GS_LINE_CAPACITY];
    FILE *file = NULL;
    size_t index;
    int failed = 0;

    if (test_artifact_inventory_downstream_aware(directory) != 0) {
        return 1;
    }
    for (index = 0u; index < GS_ARTIFACT_COUNT; index++) {
        char path[GS_PATH_CAPACITY];
        size_t size;
        size_t expected = 0u;
        if (index == 2u || index == 4u) {
            expected = train_bytes;
        }
        if (index == 3u || index == 5u) {
            expected = validation_bytes;
        }
        if (gs_path_join(
                directory, gs_artifact_names[index],
                path, sizeof(path)
            ) != 0 ||
            gs_file_size(path, &size) != 0 ||
            (expected != 0u && size != expected) ||
            (expected == 0u && size == 0u)) {
            return 1;
        }
    }
    if (gs_path_join(
            directory, "sha256.txt",
            checksum_path, sizeof(checksum_path)
        ) != 0 ||
        (file = fopen(checksum_path, "rb")) == NULL) {
        return 1;
    }
    for (index = 0u;
         !failed && index < GS_INPUT_COUNT + GS_ARTIFACT_COUNT;
         index++) {
        char path[GS_PATH_CAPACITY];
        char hash[65];
        char expected[GS_LINE_CAPACITY];
        const char *kind;
        const char *name;
        int written;

        if (index < GS_INPUT_COUNT) {
            kind = "INPUT";
            name = gs_input_logical_names[index];
            strcpy(path, input_paths[index]);
        } else {
            kind = "ARTIFACT";
            name = gs_artifact_names[index - GS_INPUT_COUNT];
            if (gs_path_join(
                    directory, name, path, sizeof(path)
                ) != 0) {
                failed = 1;
                break;
            }
        }
        if (gs_sha_file(path, hash) != 0 ||
            (written = snprintf(
                expected, sizeof(expected),
                "%s %s %s", kind, hash, name
            )) < 0 ||
            (size_t)written >= sizeof(expected) ||
            fgets(line, sizeof(line), file) == NULL) {
            failed = 1;
            break;
        }
        gs_chomp(line);
        if (strcmp(line, expected) != 0) {
            failed = 1;
        }
    }
    if (!failed &&
        (fgets(line, sizeof(line), file) != NULL || ferror(file))) {
        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    return failed;
}

static int test_resolve_artifact_layout(
    const char *manifest_argument,
    const char *source_argument,
    const GSProfile *profile,
    const char *subset_train_argument,
    const char *subset_validation_argument,
    const char *artifact_argument,
    GSResolved *resolved,
    char artifact[GS_PATH_CAPACITY]
)
{
    static const char baseline_prefix[] =
        "/sakai_multires_c/results/baseline/";
    char suffix[GS_PATH_CAPACITY];
    char expected[GS_PATH_CAPACITY];
    char canonical[GS_PATH_CAPACITY];
    size_t source_length;
    size_t suffix_length;
    size_t root_length;
    int written;

    if (profile == NULL || resolved == NULL || artifact == NULL ||
        gs_canonical_existing(source_argument, 1, resolved->source) != 0) {
        return 1;
    }
    written = snprintf(suffix, sizeof(suffix), "%s%s",
                       baseline_prefix, profile->source_basename);
    if (written < 0 || (size_t)written >= sizeof(suffix)) {
        return 1;
    }
    source_length = strlen(resolved->source);
    suffix_length = strlen(suffix);
    if (source_length <= suffix_length) {
        return 1;
    }
    root_length = source_length - suffix_length;
#ifdef _WIN32
    if (_stricmp(resolved->source + root_length, suffix) != 0) {
#else
    if (strcmp(resolved->source + root_length, suffix) != 0) {
#endif
        return 1;
    }
    if (root_length == 0u || root_length >= sizeof(resolved->root)) {
        return 1;
    }
    memcpy(resolved->root, resolved->source, root_length);
    resolved->root[root_length] = '\0';
    if (gs_canonical_existing(resolved->root, 1, canonical) != 0 ||
        !gs_paths_equal(resolved->root, canonical) ||
        gs_path_join(
            resolved->root,
            "sakai_multires_c/results/baseline",
            expected,
            sizeof(expected)
        ) != 0 ||
        gs_canonical_existing(expected, 1, canonical) != 0 ||
        gs_path_join(canonical, profile->source_basename,
                     expected, sizeof(expected)) != 0 ||
        gs_canonical_existing(expected, 1, canonical) != 0 ||
        !gs_paths_equal(resolved->source, canonical)) {
        return 1;
    }
    if (gs_path_join(resolved->root, "manifest_events_272.csv",
                     expected, sizeof(expected)) != 0 ||
        gs_canonical_existing(expected, 0, resolved->manifest) != 0 ||
        gs_canonical_existing(manifest_argument, 0, canonical) != 0 ||
        !gs_paths_equal(resolved->manifest, canonical) ||
        gs_path_join(
            resolved->root,
            "sakai_multires_c/configs/subset224_train_800_seed123.csv",
            expected,
            sizeof(expected)
        ) != 0 ||
        gs_canonical_existing(expected, 0, resolved->subset_train) != 0 ||
        gs_canonical_existing(subset_train_argument, 0, canonical) != 0 ||
        !gs_paths_equal(resolved->subset_train, canonical) ||
        gs_path_join(
            resolved->root,
            "sakai_multires_c/configs/subset224_val_200_seed123.csv",
            expected,
            sizeof(expected)
        ) != 0 ||
        gs_canonical_existing(expected, 0, resolved->subset_validation) != 0 ||
        gs_canonical_existing(subset_validation_argument, 0, canonical) != 0 ||
        !gs_paths_equal(resolved->subset_validation, canonical)) {
        return 1;
    }
    if (gs_path_join(
            resolved->root,
            "sakai_multires_c/results/baseline/matched_subset1000",
            expected,
            sizeof(expected)
        ) != 0 ||
        gs_canonical_existing(expected, 1, resolved->output_parent) != 0 ||
        gs_path_join(resolved->output_parent, profile->matched_basename,
                     expected, sizeof(expected)) != 0 ||
        gs_canonical_existing(expected, 1, artifact) != 0 ||
        gs_canonical_existing(artifact_argument, 1, canonical) != 0 ||
        !gs_paths_equal(artifact, canonical)) {
        return 1;
    }
    return 0;
}

static int test_read_binary_exact(
    const char *directory,
    const char *name,
    const float *expected,
    size_t count
)
{
    char path[GS_PATH_CAPACITY];
    size_t expected_bytes;
    size_t actual_bytes;
    float *actual = NULL;
    FILE *file = NULL;
    int failed = 0;
    if (expected == NULL ||
        gs_checked_product(count, sizeof(float), &expected_bytes) != 0 ||
        gs_path_join(directory, name, path, sizeof(path)) != 0 ||
        gs_file_size(path, &actual_bytes) != 0 ||
        actual_bytes != expected_bytes ||
        (actual = (float *)malloc(expected_bytes)) == NULL ||
        (file = fopen(path, "rb")) == NULL) {
        free(actual);
        return 1;
    }
    if (fread(actual, sizeof(float), count, file) != count ||
        fgetc(file) != EOF || ferror(file) ||
        memcmp(actual, expected, expected_bytes) != 0) {
        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    free(actual);
    return failed;
}

static int test_scaler_matches(
    const char *directory,
    const GSProfile *profile,
    const double *mean,
    const double *stddev,
    const unsigned char *constant_mask
)
{
    char path[GS_PATH_CAPACITY];
    char line[GS_LINE_CAPACITY];
    FILE *file = NULL;
    size_t feature;
    if (gs_path_join(directory, "scaler.csv", path, sizeof(path)) != 0 ||
        (file = fopen(path, "rb")) == NULL ||
        fgets(line, sizeof(line), file) == NULL) {
        if (file != NULL) fclose(file);
        return 1;
    }
    gs_chomp(line);
    if (strcmp(
            line,
            "feature_index,channel,pooled_row,pooled_col,mean,stddev,constant"
        ) != 0) {
        fclose(file);
        return 1;
    }
    for (feature = 0u; feature < profile->feature_count; feature++) {
        char *fields[7];
        size_t count;
        size_t parsed_feature;
        size_t channel;
        size_t pooled_row;
        size_t pooled_col;
        size_t constant_value;
        size_t plane = profile->pooled_side * profile->pooled_side;
        char *end;
        double parsed_mean;
        double parsed_stddev;
        if (fgets(line, sizeof(line), file) == NULL) {
            fclose(file);
            return 1;
        }
        gs_chomp(line);
        if (gs_csv_fields(line, fields, 7u, &count) != 0 || count != 7u ||
            gs_parse_size(fields[0], &parsed_feature) != 0 ||
            gs_parse_size(fields[1], &channel) != 0 ||
            gs_parse_size(fields[2], &pooled_row) != 0 ||
            gs_parse_size(fields[3], &pooled_col) != 0 ||
            gs_parse_size(fields[6], &constant_value) != 0) {
            fclose(file);
            return 1;
        }
        errno = 0;
        parsed_mean = strtod(fields[4], &end);
        if (errno != 0 || *end != '\0') {
            fclose(file);
            return 1;
        }
        errno = 0;
        parsed_stddev = strtod(fields[5], &end);
        if (errno != 0 || *end != '\0' ||
            parsed_feature != feature ||
            channel != feature / plane ||
            pooled_row != (feature % plane) / profile->pooled_side ||
            pooled_col != (feature % plane) % profile->pooled_side ||
            parsed_mean != mean[feature] ||
            parsed_stddev != stddev[feature] ||
            constant_value != (size_t)constant_mask[feature]) {
            fclose(file);
            return 1;
        }
    }
    if (fgets(line, sizeof(line), file) != NULL || ferror(file) ||
        fclose(file) != 0) {
        return 1;
    }
    return 0;
}

static int test_output_rows_match(
    const char *directory,
    const char *name,
    const char *split,
    const GSRow *expected,
    size_t count
)
{
    char path[GS_PATH_CAPACITY];
    GSRow *actual;
    size_t index;
    int failed = 0;
    if ((actual = (GSRow *)calloc(count, sizeof(*actual))) == NULL ||
        gs_path_join(directory, name, path, sizeof(path)) != 0 ||
        gs_load_rows(path, split, count, 0, actual) != 0) {
        free(actual);
        return 1;
    }
    for (index = 0u; index < count; index++) {
        if (!gs_rows_equal(actual + index, expected + index)) {
            failed = 1;
            break;
        }
    }
    free(actual);
    return failed;
}

static int test_metadata_matches(
    const char *directory,
    const GSProfile *profile,
    const char *code_commit,
    size_t constant_count
)
{
    char feature_path[GS_PATH_CAPACITY];
    char summary_path[GS_PATH_CAPACITY];
    char value[64];
    int written;
    if (gs_path_join(directory, "feature_spec.txt", feature_path,
                     sizeof(feature_path)) != 0 ||
        gs_path_join(directory, "summary.txt", summary_path,
                     sizeof(summary_path)) != 0) {
        return 1;
    }
    written = snprintf(value, sizeof(value), "%zu", profile->feature_count);
    if (written < 0 || (size_t)written >= sizeof(value) ||
        !gs_file_has_key_value(feature_path, "profile",
                               profile->matched_profile) ||
        !gs_file_has_key_value(feature_path, "source_profile",
                               profile->source_profile) ||
        !gs_file_has_key_value(
            feature_path,
            "matched_subset_profile",
            "subset224_seed123_prefix_800_200"
        ) ||
        !gs_file_has_key_value(
            feature_path,
            "provenance",
            "slice_raw_official_full_baseline"
        ) ||
        !gs_file_has_key_value(feature_path, "feature_count", value) ||
        !gs_file_has_key_value(feature_path, "train_rows", "800") ||
        !gs_file_has_key_value(feature_path, "validation_rows", "200") ||
        !gs_file_has_key_value(feature_path, "source_train_rows", "6866") ||
        !gs_file_has_key_value(
            feature_path, "source_validation_rows", "1717") ||
        !gs_file_has_key_value(
            feature_path, "normalization",
            "already_applied_in_source_raw") ||
        !gs_file_has_key_value(
            feature_path, "standardization",
            "train_only_population_ddof0") ||
        !gs_file_has_key_value(
            feature_path, "source_code_commit", profile->source_commit) ||
        !gs_file_has_key_value(
            feature_path, "matched_builder_code_commit", code_commit) ||
        !gs_file_has_key_value(
            feature_path, "scaler_unchanged_after_validation", "true") ||
        !gs_file_has_key_value(feature_path, "pca", "false") ||
        !gs_file_has_key_value(feature_path, "augmentation", "false") ||
        !gs_file_has_key_value(feature_path, "labels_used", "false") ||
        (written = snprintf(value, sizeof(value), "%zu", constant_count)) < 0 ||
        (size_t)written >= sizeof(value) ||
        !gs_file_has_key_value(
            feature_path, "constant_feature_count", value) ||
        !gs_file_has_key_value(summary_path, "profile",
                               profile->matched_profile) ||
        !gs_file_has_key_value(
            summary_path, "matched_builder_code_commit", code_commit) ||
        !gs_file_has_key_value(
            summary_path, "raw_train_is_bitwise_source_prefix", "true") ||
        !gs_file_has_key_value(
            summary_path, "raw_validation_is_bitwise_source_prefix", "true") ||
        !gs_file_has_key_value(
            summary_path, "validation_transformed_without_refit", "true") ||
        !gs_file_has_key_value(
            summary_path, "scaler_unchanged_after_validation", "true") ||
        !gs_file_has_key_value(
            summary_path, "constant_feature_count", value)) {
        return 1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    const GSProfile *profile;
    GSResolved resolved;
    char artifact[GS_PATH_CAPACITY];
    char path[GS_PATH_CAPACITY];
    char input_storage[GS_INPUT_COUNT][GS_PATH_CAPACITY];
    const char *input_paths[GS_INPUT_COUNT];
    GSRow *source_train = NULL;
    GSRow *source_validation = NULL;
    GSRow subset_train[GS_MATCHED_TRAIN_ROWS];
    GSRow subset_validation[GS_MATCHED_VALIDATION_ROWS];
    float *train_raw = NULL;
    float *validation_raw = NULL;
    float *train_standardized = NULL;
    float *validation_standardized = NULL;
    double *mean = NULL;
    double *stddev = NULL;
    double *mean_copy = NULL;
    double *stddev_copy = NULL;
    unsigned char *constant_mask = NULL;
    unsigned char *constant_copy = NULL;
    GSStatistics statistics = {0};
    size_t train_values;
    size_t validation_values;
    size_t train_bytes;
    size_t validation_bytes;
    size_t double_bytes;
    size_t index;
    int result = 1;

    if (argc != 8) {
        fprintf(stderr,
            "Usage: test_matched_subset1000_no_pca_artifacts.exe "
            "<manifest_events> <source_full_no_pca_directory> "
            "<mini75_pool5|mini112_pool7> <subset_train_csv> "
            "<subset_val_csv> <artifact_directory> <expected_code_commit>\n");
        return 1;
    }
    profile = gs_profile_from_token(argv[3]);
    if (!gs_platform_valid() || profile == NULL ||
        !gs_commit_valid(argv[7]) ||
        test_resolve_artifact_layout(
            argv[1], argv[2], profile, argv[4], argv[5], argv[6],
            &resolved, artifact) != 0) {
        fprintf(stderr, "ERROR: canonical validation layout rejected.\n");
        return 1;
    }
    printf("[PASS] independent_canonical_resolution\n");
    if (gs_validate_source_inventory(resolved.source) != 0 ||
        gs_validate_source_sha(&resolved, profile) != 0 ||
        gs_validate_source_metadata(&resolved, profile) != 0) {
        fprintf(stderr, "ERROR: independent source provenance validation failed.\n");
        return 1;
    }
    printf("[PASS] independent_full_source_provenance\n");
    source_train = (GSRow *)calloc(GS_SOURCE_TRAIN_ROWS, sizeof(*source_train));
    source_validation =
        (GSRow *)calloc(GS_SOURCE_VALIDATION_ROWS, sizeof(*source_validation));
    if (source_train == NULL || source_validation == NULL ||
        gs_path_join(resolved.source, "train_rows.csv", path,
                     sizeof(path)) != 0 ||
        gs_load_rows(path, "train", GS_SOURCE_TRAIN_ROWS, 0,
                     source_train) != 0 ||
        gs_path_join(resolved.source, "val_rows.csv", path,
                     sizeof(path)) != 0 ||
        gs_load_rows(path, "validation", GS_SOURCE_VALIDATION_ROWS, 0,
                     source_validation) != 0 ||
        gs_load_rows(resolved.subset_train, "train", GS_MATCHED_TRAIN_ROWS,
                     1, subset_train) != 0 ||
        gs_load_rows(resolved.subset_validation, "validation",
                     GS_MATCHED_VALIDATION_ROWS, 1,
                     subset_validation) != 0) {
        goto cleanup;
    }
    for (index = 0u; index < GS_MATCHED_TRAIN_ROWS; index++) {
        if (!gs_rows_equal(source_train + index, subset_train + index)) {
            goto cleanup;
        }
    }
    for (index = 0u; index < GS_MATCHED_VALIDATION_ROWS; index++) {
        if (!gs_rows_equal(source_validation + index,
                           subset_validation + index)) {
            goto cleanup;
        }
    }
    if (gs_validate_manifest_identifiers(
            resolved.manifest, subset_train, subset_validation) != 0) {
        goto cleanup;
    }
    printf("[PASS] independent_subset_and_row_map_reconstruction\n");
    if (gs_checked_product(GS_MATCHED_TRAIN_ROWS, profile->feature_count,
                           &train_values) != 0 ||
        gs_checked_product(GS_MATCHED_VALIDATION_ROWS,
                           profile->feature_count, &validation_values) != 0 ||
        gs_checked_product(train_values, sizeof(float), &train_bytes) != 0 ||
        gs_checked_product(validation_values, sizeof(float),
                           &validation_bytes) != 0 ||
        gs_checked_product(profile->feature_count, sizeof(double),
                           &double_bytes) != 0) {
        goto cleanup;
    }
    train_raw = (float *)malloc(train_bytes);
    validation_raw = (float *)malloc(validation_bytes);
    train_standardized = (float *)malloc(train_bytes);
    validation_standardized = (float *)malloc(validation_bytes);
    mean = (double *)malloc(double_bytes);
    stddev = (double *)malloc(double_bytes);
    mean_copy = (double *)malloc(double_bytes);
    stddev_copy = (double *)malloc(double_bytes);
    constant_mask = (unsigned char *)malloc(profile->feature_count);
    constant_copy = (unsigned char *)malloc(profile->feature_count);
    if (train_raw == NULL || validation_raw == NULL ||
        train_standardized == NULL || validation_standardized == NULL ||
        mean == NULL || stddev == NULL || mean_copy == NULL ||
        stddev_copy == NULL || constant_mask == NULL ||
        constant_copy == NULL) {
        goto cleanup;
    }
    if (gs_path_join(resolved.source, "train_features_raw_f32.bin",
                     path, sizeof(path)) != 0 ||
        gs_read_raw_prefix_and_validate(
            path, GS_SOURCE_TRAIN_ROWS, GS_MATCHED_TRAIN_ROWS,
            profile->feature_count, train_raw,
            &statistics.train_raw_min, &statistics.train_raw_max) != 0 ||
        gs_path_join(resolved.source, "val_features_raw_f32.bin",
                     path, sizeof(path)) != 0 ||
        gs_read_raw_prefix_and_validate(
            path, GS_SOURCE_VALIDATION_ROWS, GS_MATCHED_VALIDATION_ROWS,
            profile->feature_count, validation_raw,
            &statistics.validation_raw_min,
            &statistics.validation_raw_max) != 0) {
        goto cleanup;
    }
    if (test_read_binary_exact(
            artifact, "train_features_raw_f32.bin",
            train_raw, train_values) != 0 ||
        test_read_binary_exact(
            artifact, "val_features_raw_f32.bin",
            validation_raw, validation_values) != 0) {
        goto cleanup;
    }
    printf("RAW_TRAIN_SLICE_BITWISE=true\n");
    printf("RAW_VALIDATION_SLICE_BITWISE=true\n");
    if (gs_baseline_standardization_fit(
            train_raw, GS_MATCHED_TRAIN_ROWS, profile->feature_count,
            mean, stddev, constant_mask,
            GS_BASELINE_STANDARDIZATION_DEFAULT_CONSTANT_EPSILON
        ) != GS_BASELINE_STANDARDIZATION_SUCCESS) {
        goto cleanup;
    }
    memcpy(mean_copy, mean, double_bytes);
    memcpy(stddev_copy, stddev, double_bytes);
    memcpy(constant_copy, constant_mask, profile->feature_count);
    if (gs_baseline_standardization_transform(
            train_raw, GS_MATCHED_TRAIN_ROWS, profile->feature_count,
            mean, stddev, constant_mask, train_standardized
        ) != GS_BASELINE_STANDARDIZATION_SUCCESS ||
        gs_baseline_standardization_transform(
            validation_raw, GS_MATCHED_VALIDATION_ROWS,
            profile->feature_count, mean, stddev, constant_mask,
            validation_standardized
        ) != GS_BASELINE_STANDARDIZATION_SUCCESS ||
        memcmp(mean, mean_copy, double_bytes) != 0 ||
        memcmp(stddev, stddev_copy, double_bytes) != 0 ||
        memcmp(constant_mask, constant_copy, profile->feature_count) != 0 ||
        gs_validate_standardized(
            train_standardized, validation_standardized, mean, stddev,
            constant_mask, profile->feature_count, &statistics) != 0) {
        goto cleanup;
    }
    printf("SCALER_UNCHANGED_AFTER_VALIDATION=true\n");
    if (test_scaler_matches(
            artifact, profile, mean, stddev, constant_mask) != 0) {
        goto cleanup;
    }
    printf("SCALER_RECOMPUTED_MATCH=true\n");
    if (test_read_binary_exact(
            artifact, "train_features_std_f32.bin",
            train_standardized, train_values) != 0 ||
        test_read_binary_exact(
            artifact, "val_features_std_f32.bin",
            validation_standardized, validation_values) != 0) {
        goto cleanup;
    }
    printf("STANDARDIZED_TRAIN_BITWISE=true\n");
    printf("STANDARDIZED_VALIDATION_BITWISE=true\n");
    if (test_output_rows_match(
            artifact, "train_rows.csv", "train",
            subset_train, GS_MATCHED_TRAIN_ROWS) != 0 ||
        test_output_rows_match(
            artifact, "val_rows.csv", "validation",
            subset_validation, GS_MATCHED_VALIDATION_ROWS) != 0) {
        goto cleanup;
    }
    printf("ROW_MAPS_RECOMPUTED_MATCH=true\n");
    if (test_metadata_matches(
            artifact, profile, argv[7], statistics.constant_count) != 0 ||
        gs_source_input_paths(&resolved, input_storage, input_paths) != 0 ||
        test_validate_output_downstream_aware(
            artifact, input_paths, train_bytes, validation_bytes
        ) != 0) {
        goto cleanup;
    }
    printf("[PASS] independently_recomputed_metadata_and_inventory\n");
    printf("ALL_MATCHED_SUBSET_CHECKSUMS_MATCH=true\n");
    printf("PROFILE=%s\n", profile->matched_profile);
    printf("TRAIN_ROWS=%d\n", GS_MATCHED_TRAIN_ROWS);
    printf("VALIDATION_ROWS=%d\n", GS_MATCHED_VALIDATION_ROWS);
    printf("FEATURE_DIM=%zu\n", profile->feature_count);
    printf("CONSTANT_FEATURES=%zu\n", statistics.constant_count);
    printf("ALL MATCHED SUBSET1000 NO-PCA ARTIFACT TESTS PASSED\n");
    result = 0;

cleanup:
    free(constant_copy);
    free(constant_mask);
    free(stddev_copy);
    free(mean_copy);
    free(stddev);
    free(mean);
    free(validation_standardized);
    free(train_standardized);
    free(validation_raw);
    free(train_raw);
    free(source_validation);
    free(source_train);
    if (result != 0) {
        fprintf(stderr, "MATCHED SUBSET1000 ARTIFACT VALIDATION FAILED\n");
    }
    return result;
}
