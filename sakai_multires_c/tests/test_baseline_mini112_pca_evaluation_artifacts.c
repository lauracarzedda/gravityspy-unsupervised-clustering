#define main gs_legacy_mini112_evaluation_validator_main
#include "test_baseline_mini112_evaluation_artifacts.c"
#undef main


enum {
    GSPT_PCA_FILE_COUNT = 12,
    GSPT_KMEANS_FILE_COUNT = 13,
    GSPT_EVALUATION_ENTRY_COUNT = 18
};


typedef struct {
    char token[16];
    char profile[64];
    char threshold_text[16];
    size_t retained_components;
    char code_commit[65];
    char source_directory[GS_TEST_PATH_LEN];
    char train_rows_path[GS_TEST_PATH_LEN];
    char validation_rows_path[GS_TEST_PATH_LEN];
    char train_score_name[64];
    char validation_score_name[64];
} GSPTestPCAProfile;


typedef struct {
    char code_commit[65];
    char train_assignments_path[GS_TEST_PATH_LEN];
    char validation_assignments_path[GS_TEST_PATH_LEN];
} GSPTestKMeansProfile;


static const char *const gspt_kmeans_files[GSPT_KMEANS_FILE_COUNT] = {
    "centroids_f64.bin",
    "train_assignments_u32.bin",
    "val_assignments_u32.bin",
    "train_squared_distances_f64.bin",
    "val_squared_distances_f64.bin",
    "cluster_sizes_train_u64.bin",
    "initial_centroid_indices_u64.bin",
    "train_assignments.csv",
    "val_assignments.csv",
    "cluster_sizes.csv",
    "initial_centroids.csv",
    "summary.txt",
    "sha256.txt"
};


static int gspt_commit_valid(const char *commit)
{
    size_t index;
    if (commit == NULL || strlen(commit) != 40u) return 0;
    for (index = 0; index < 40u; index++) {
        if (!((commit[index] >= '0' && commit[index] <= '9') ||
              (commit[index] >= 'a' && commit[index] <= 'f'))) {
            return 0;
        }
    }
    return 1;
}


static int gspt_parent_path(
    const char *path,
    char *parent,
    size_t capacity
)
{
    size_t length;
    if (path == NULL || parent == NULL || capacity == 0u ||
        !gs_test_value_valid(path)) {
        return 1;
    }
    length = strlen(path);
    while (length != 0u &&
           (path[length - 1u] == '/' || path[length - 1u] == '\\')) {
        length--;
    }
    while (length != 0u &&
           path[length - 1u] != '/' && path[length - 1u] != '\\') {
        length--;
    }
    while (length > 1u &&
           (path[length - 1u] == '/' || path[length - 1u] == '\\')) {
        length--;
    }
    if (length == 0u || length >= capacity) return 1;
    memcpy(parent, path, length);
    parent[length] = '\0';
    return 0;
}


static int gspt_token_contract(
    const char *token,
    const char **threshold_text,
    double *threshold
)
{
    if (token == NULL || threshold_text == NULL || threshold == NULL) return 1;
    if (strcmp(token, "pca90") == 0) {
        *threshold_text = "0.90";
        *threshold = 0.90;
    } else if (strcmp(token, "pca95") == 0) {
        *threshold_text = "0.95";
        *threshold = 0.95;
    } else if (strcmp(token, "pca99") == 0) {
        *threshold_text = "0.99";
        *threshold = 0.99;
    } else {
        return 1;
    }
    return 0;
}


static int gspt_read_key(
    const char *directory,
    const char *filename,
    const char *key,
    char *value,
    size_t capacity
)
{
    char path[GS_TEST_PATH_LEN];
    return gs_test_path_join(directory, filename, path, sizeof(path)) != 0 ||
           gs_test_read_summary_value(path, key, value, capacity) != 0;
}


static int gspt_expect_key(
    const char *directory,
    const char *filename,
    const char *key,
    const char *expected
)
{
    char value[256];
    return expected == NULL ||
           gspt_read_key(directory, filename, key, value, sizeof(value)) != 0 ||
           strcmp(value, expected) != 0;
}


static int gspt_read_thresholds(
    const char *directory,
    const char *token,
    size_t *retained_components
)
{
    static const char *const tokens[3] = {"pca90", "pca95", "pca99"};
    static const double thresholds[3] = {0.90, 0.95, 0.99};
    char path[GS_TEST_PATH_LEN];
    char line[GS_TEST_LINE_LEN];
    FILE *file = NULL;
    size_t row;
    int selected = 0;
    int failed = 0;
    if (retained_components == NULL ||
        gs_test_path_join(directory, "thresholds.csv", path, sizeof(path)) != 0 ||
        (file = fopen(path, "rb")) == NULL) {
        return 1;
    }
    if (fgets(line, sizeof(line), file) == NULL) {
        failed = 1;
    } else {
        gs_test_strip_line_ending(line);
        if (strcmp(line,
                   "threshold,component_count,cumulative_variance") != 0) {
            failed = 1;
        }
    }
    for (row = 0; !failed && row < 3u; row++) {
        char *fields[3];
        double threshold;
        double cumulative;
        size_t count;
        if (fgets(line, sizeof(line), file) == NULL ||
            gs_test_split_csv(line, fields, 3u) != 0 ||
            gs_test_parse_double(fields[0], &threshold) != 0 ||
            gs_test_parse_size(fields[1], &count) != 0 ||
            gs_test_parse_double(fields[2], &cumulative) != 0 ||
            threshold != thresholds[row] || count == 0u ||
            cumulative < threshold || cumulative > 1.0) {
            failed = 1;
            break;
        }
        if (strcmp(token, tokens[row]) == 0) {
            *retained_components = count;
            selected = 1;
        }
    }
    if (!failed && fgets(line, sizeof(line), file) != NULL) failed = 1;
    if (ferror(file) || fclose(file) != 0) failed = 1;
    return failed || !selected;
}


static int gspt_pca_inventory(
    const char *directory,
    const GSPTestPCAProfile *profile
)
{
    static const char *const fixed[10] = {
        "components_f64.bin",
        "cumulative_explained_variance_ratio_f64.bin",
        "eigenvalues_f64.bin",
        "explained_variance_ratio_f64.bin",
        "mean_f64.bin",
        "pca_spec.txt",
        "sha256.txt",
        "spectrum.csv",
        "summary.txt",
        "thresholds.csv"
    };
    unsigned char seen[GSPT_PCA_FILE_COUNT] = {0};
    DIR *stream = NULL;
    struct dirent *entry;
    size_t count = 0u;
    int kmeans_seen = 0;
    int failed = profile == NULL ||
                 (stream = opendir(directory)) == NULL;
    while (!failed) {
        char path[GS_TEST_PATH_LEN];
        size_t index;
        int regular, is_directory;
        errno = 0;
        entry = readdir(stream);
        if (entry == NULL) {
            if (errno != 0) failed = 1;
            break;
        }
        if (strcmp(entry->d_name, ".") == 0 ||
            strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        if (gs_test_path_join(directory, entry->d_name, path, sizeof(path)) != 0 ||
            gs_test_path_state(path, &regular, &is_directory) != 1) {
            failed = 1;
            break;
        }
        if (strcmp(entry->d_name, "kmeans_seed123") == 0) {
            if (kmeans_seen || !is_directory) failed = 1;
            kmeans_seen = 1;
            continue;
        }
        if (strcmp(entry->d_name, "label_evaluation_seed123") == 0) {
            if (!is_directory) failed = 1;
            continue;
        }
        if (strcmp(entry->d_name, profile->train_score_name) == 0) {
            index = 10u;
        } else if (strcmp(entry->d_name,
                          profile->validation_score_name) == 0) {
            index = 11u;
        } else if (!gs_test_name_index(entry->d_name, fixed, 10u, &index)) {
            failed = 1;
            break;
        }
        if (!regular || is_directory || seen[index]) {
            failed = 1;
            break;
        }
        seen[index] = 1u;
        count++;
    }
    if (stream != NULL && closedir(stream) != 0) failed = 1;
    if (count != GSPT_PCA_FILE_COUNT || !kmeans_seen) failed = 1;
    while (!failed && count != 0u) {
        count--;
        if (!seen[count]) failed = 1;
    }
    return failed;
}


static int gspt_build_pca_entries(
    const GSPTestPCAProfile *profile,
    const char *directory,
    GSTestChecksumEntry entries[18],
    char paths[18][GS_TEST_PATH_LEN]
)
{
    static const char *const inputs[7] = {
        "feature_spec.txt", "scaler.csv", "train_features_std_f32.bin",
        "val_features_std_f32.bin", "train_rows.csv", "val_rows.csv",
        "sha256.txt"
    };
    static const char *const artifacts[8] = {
        "pca_spec.txt", "mean_f64.bin", "components_f64.bin",
        "eigenvalues_f64.bin", "explained_variance_ratio_f64.bin",
        "cumulative_explained_variance_ratio_f64.bin", "thresholds.csv",
        "spectrum.csv"
    };
    size_t index;
    if (profile == NULL) return 1;
    for (index = 0; index < 7u; index++) {
        if (gs_test_path_join(profile->source_directory, inputs[index],
                              paths[index], GS_TEST_PATH_LEN) != 0) {
            return 1;
        }
        entries[index].kind = "INPUT";
        entries[index].name = inputs[index];
        entries[index].path = paths[index];
    }
    for (index = 0; index < 8u; index++) {
        if (gs_test_path_join(directory, artifacts[index], paths[index + 7u],
                              GS_TEST_PATH_LEN) != 0) {
            return 1;
        }
        entries[index + 7u].kind = "ARTIFACT";
        entries[index + 7u].name = artifacts[index];
        entries[index + 7u].path = paths[index + 7u];
    }
    if (gs_test_path_join(directory, profile->train_score_name, paths[15],
                          GS_TEST_PATH_LEN) != 0 ||
        gs_test_path_join(directory, profile->validation_score_name, paths[16],
                          GS_TEST_PATH_LEN) != 0 ||
        gs_test_path_join(directory, "summary.txt", paths[17],
                          GS_TEST_PATH_LEN) != 0) {
        return 1;
    }
    entries[15].kind = "ARTIFACT";
    entries[15].name = profile->train_score_name;
    entries[15].path = paths[15];
    entries[16].kind = "ARTIFACT";
    entries[16].name = profile->validation_score_name;
    entries[16].path = paths[16];
    entries[17].kind = "ARTIFACT";
    entries[17].name = "summary.txt";
    entries[17].path = paths[17];
    return 0;
}


static int gspt_validate_pca(
    const char *manifest_path,
    const char *directory,
    const char *token,
    GSPTestPCAProfile *profile
)
{
    GSTestChecksumEntry entries[18];
    char paths[18][GS_TEST_PATH_LEN];
    char parent[GS_TEST_PATH_LEN];
    char expected_profile[64];
    char value[256];
    char sha_path[GS_TEST_PATH_LEN];
    const char *threshold_text;
    size_t retained_components = 0u;
    size_t actual, values, bytes;
    double threshold;
    int written;
    if (profile == NULL ||
        gspt_token_contract(token, &threshold_text, &threshold) != 0 ||
        (written = snprintf(expected_profile, sizeof(expected_profile),
                            "mini112_pool7_%s", token)) < 0 ||
        (size_t)written >= sizeof(expected_profile) ||
        strcmp(gs_test_basename(directory), expected_profile) != 0 ||
        gspt_parent_path(directory, parent, sizeof(parent)) != 0 ||
        gs_test_path_join(parent, "mini112_pool7_no_pca",
                          profile->source_directory,
                          sizeof(profile->source_directory)) != 0 ||
        gs_test_validate_base(manifest_path, profile->source_directory) != 0 ||
        (written = snprintf(profile->train_score_name,
                            sizeof(profile->train_score_name),
                            "train_scores_%s_f32.bin", token)) < 0 ||
        (size_t)written >= sizeof(profile->train_score_name) ||
        (written = snprintf(profile->validation_score_name,
                            sizeof(profile->validation_score_name),
                            "val_scores_%s_f32.bin", token)) < 0 ||
        (size_t)written >= sizeof(profile->validation_score_name) ||
        gspt_read_thresholds(directory, token, &retained_components) != 0) {
        return 1;
    }
    (void)threshold;
    strcpy(profile->token, token);
    strcpy(profile->profile, expected_profile);
    strcpy(profile->threshold_text, threshold_text);
    profile->retained_components = retained_components;
    if (gspt_expect_key(directory, "pca_spec.txt", "profile",
                        profile->profile) ||
        gspt_expect_key(directory, "pca_spec.txt", "source_profile",
                        "mini112_pool7_no_pca") ||
        gspt_expect_key(directory, "pca_spec.txt", "selected_threshold",
                        profile->threshold_text) ||
        gspt_expect_key(directory, "pca_spec.txt", "original_feature_count",
                        "1024") ||
        gspt_expect_key(directory, "pca_spec.txt", "train_rows", "6866") ||
        gspt_expect_key(directory, "pca_spec.txt", "validation_rows", "1717") ||
        gspt_expect_key(directory, "pca_spec.txt", "fit_input", "train_only") ||
        gspt_expect_key(directory, "pca_spec.txt",
                        "validation_transform_without_refit", "true") ||
        gspt_expect_key(directory, "pca_spec.txt", "whitening", "false") ||
        gspt_expect_key(directory, "pca_spec.txt",
                        "post_standardization", "false") ||
        gspt_expect_key(directory, "pca_spec.txt", "augmentation", "false") ||
        gspt_expect_key(directory, "pca_spec.txt", "labels_used", "false") ||
        gspt_read_key(directory, "pca_spec.txt", "retained_component_count",
                      value, sizeof(value)) != 0 ||
        gs_test_parse_size(value, &actual) != 0 ||
        actual != retained_components ||
        gspt_expect_key(directory, "summary.txt", "profile",
                        profile->profile) ||
        gspt_expect_key(directory, "summary.txt", "selected_threshold",
                        profile->threshold_text) ||
        gspt_expect_key(directory, "summary.txt", "pca_fit", "train_only") ||
        gspt_expect_key(directory, "summary.txt",
                        "validation_transformed_without_refit", "true") ||
        gspt_expect_key(directory, "summary.txt", "whitening", "false") ||
        gspt_expect_key(directory, "summary.txt",
                        "post_standardization", "false") ||
        gspt_expect_key(directory, "summary.txt", "labels_used", "false") ||
        gspt_read_key(directory, "summary.txt", "code_commit",
                      profile->code_commit,
                      sizeof(profile->code_commit)) != 0 ||
        !gspt_commit_valid(profile->code_commit) ||
        gspt_pca_inventory(directory, profile) != 0 ||
        gspt_build_pca_entries(profile, directory, entries, paths) != 0 ||
        gs_test_path_join(directory, "sha256.txt", sha_path,
                          sizeof(sha_path)) != 0 ||
        gs_test_verify_checksum_manifest(sha_path, entries, 18u) != 0 ||
        gs_test_path_join(profile->source_directory, "train_rows.csv",
                          profile->train_rows_path,
                          sizeof(profile->train_rows_path)) != 0 ||
        gs_test_path_join(profile->source_directory, "val_rows.csv",
                          profile->validation_rows_path,
                          sizeof(profile->validation_rows_path)) != 0 ||
        gs_test_checked_product(GS_TEST_TRAIN_ROWS, retained_components,
                                &values) != 0 ||
        gs_test_checked_product(values, sizeof(float), &bytes) != 0 ||
        gs_test_file_size(paths[15], &actual) != 0 || actual != bytes ||
        gs_test_checked_product(GS_TEST_VALIDATION_ROWS, retained_components,
                                &values) != 0 ||
        gs_test_checked_product(values, sizeof(float), &bytes) != 0 ||
        gs_test_file_size(paths[16], &actual) != 0 || actual != bytes) {
        return 1;
    }
    return 0;
}


static int gspt_build_kmeans_entries(
    const char *pca_directory,
    const char *directory,
    const GSPTestPCAProfile *profile,
    GSTestChecksumEntry entries[17],
    char paths[17][GS_TEST_PATH_LEN]
)
{
    const char *inputs[5];
    static const char *const artifacts[12] = {
        "centroids_f64.bin", "train_assignments_u32.bin",
        "val_assignments_u32.bin", "train_squared_distances_f64.bin",
        "val_squared_distances_f64.bin", "cluster_sizes_train_u64.bin",
        "initial_centroid_indices_u64.bin", "train_assignments.csv",
        "val_assignments.csv", "cluster_sizes.csv", "initial_centroids.csv",
        "summary.txt"
    };
    size_t index;
    inputs[0] = "pca_spec.txt";
    inputs[1] = "thresholds.csv";
    inputs[2] = profile->train_score_name;
    inputs[3] = profile->validation_score_name;
    inputs[4] = "source_pca_sha256.txt";
    for (index = 0; index < 5u; index++) {
        const char *physical = index == 4u ? "sha256.txt" : inputs[index];
        if (gs_test_path_join(pca_directory, physical, paths[index],
                              GS_TEST_PATH_LEN) != 0) {
            return 1;
        }
        entries[index].kind = "INPUT";
        entries[index].name = inputs[index];
        entries[index].path = paths[index];
    }
    for (index = 0; index < 12u; index++) {
        if (gs_test_path_join(directory, artifacts[index], paths[index + 5u],
                              GS_TEST_PATH_LEN) != 0) {
            return 1;
        }
        entries[index + 5u].kind = "ARTIFACT";
        entries[index + 5u].name = artifacts[index];
        entries[index + 5u].path = paths[index + 5u];
    }
    return 0;
}


static int gspt_validate_kmeans(
    const char *pca_directory,
    const char *directory,
    size_t k,
    const GSPTestPCAProfile *profile,
    GSPTestKMeansProfile *metadata
)
{
    GSTestChecksumEntry entries[17];
    char paths[17][GS_TEST_PATH_LEN];
    char sha_path[GS_TEST_PATH_LEN];
    char expected_profile[96];
    char k_text[32];
    char components_text[32];
    char centroid_path[GS_TEST_PATH_LEN];
    size_t values, bytes, actual;
    int written;
    if (metadata == NULL || profile == NULL || (k != 22u && k != 36u) ||
        gs_test_validate_kmeans_location(pca_directory, directory, k) != 0 ||
        gs_test_exact_files(directory, gspt_kmeans_files,
                            GSPT_KMEANS_FILE_COUNT) != 0 ||
        (written = snprintf(expected_profile, sizeof(expected_profile),
                            "%s_kmeans", profile->profile)) < 0 ||
        (size_t)written >= sizeof(expected_profile) ||
        (written = snprintf(k_text, sizeof(k_text), "%lu",
                            (unsigned long)k)) < 0 ||
        (size_t)written >= sizeof(k_text) ||
        (written = snprintf(components_text, sizeof(components_text), "%lu",
                            (unsigned long)profile->retained_components)) < 0 ||
        (size_t)written >= sizeof(components_text) ||
        gspt_expect_key(directory, "summary.txt", "profile",
                        expected_profile) ||
        gspt_expect_key(directory, "summary.txt", "input_profile",
                        profile->profile) ||
        gspt_expect_key(directory, "summary.txt", "pca_profile",
                        profile->token) ||
        gspt_expect_key(directory, "summary.txt", "pca_threshold",
                        profile->threshold_text) ||
        gspt_expect_key(directory, "summary.txt", "retained_component_count",
                        components_text) ||
        gspt_expect_key(directory, "summary.txt", "feature_count",
                        components_text) ||
        gspt_expect_key(directory, "summary.txt", "k", k_text) ||
        gspt_expect_key(directory, "summary.txt", "seed", "123") ||
        gspt_expect_key(directory, "summary.txt", "max_iterations", "300") ||
        gspt_expect_key(directory, "summary.txt", "fit_input", "train_only") ||
        gspt_expect_key(directory, "summary.txt",
                        "validation_assigned_after_fit", "true") ||
        gspt_expect_key(directory, "summary.txt", "pca", "true") ||
        gspt_expect_key(directory, "summary.txt", "whitening", "false") ||
        gspt_expect_key(directory, "summary.txt",
                        "post_pca_standardization", "false") ||
        gspt_expect_key(directory, "summary.txt", "augmentation", "false") ||
        gspt_expect_key(directory, "summary.txt", "labels_used", "false") ||
        gspt_read_key(directory, "summary.txt", "pca_code_commit",
                      metadata->code_commit,
                      sizeof(metadata->code_commit)) != 0 ||
        strcmp(metadata->code_commit, profile->code_commit) != 0 ||
        gspt_read_key(directory, "summary.txt", "code_commit",
                      metadata->code_commit,
                      sizeof(metadata->code_commit)) != 0 ||
        !gspt_commit_valid(metadata->code_commit) ||
        gspt_build_kmeans_entries(
            pca_directory, directory, profile, entries, paths
        ) != 0 ||
        gs_test_path_join(directory, "sha256.txt", sha_path,
                          sizeof(sha_path)) != 0 ||
        gs_test_verify_checksum_manifest(sha_path, entries, 17u) != 0 ||
        gs_test_path_join(directory, "train_assignments_u32.bin",
                          metadata->train_assignments_path,
                          sizeof(metadata->train_assignments_path)) != 0 ||
        gs_test_path_join(directory, "val_assignments_u32.bin",
                          metadata->validation_assignments_path,
                          sizeof(metadata->validation_assignments_path)) != 0 ||
        gs_test_file_size(metadata->train_assignments_path, &actual) != 0 ||
        actual != GS_TEST_TRAIN_ROWS * sizeof(uint32_t) ||
        gs_test_file_size(metadata->validation_assignments_path, &actual) != 0 ||
        actual != GS_TEST_VALIDATION_ROWS * sizeof(uint32_t) ||
        gs_test_checked_product(k, profile->retained_components, &values) != 0 ||
        gs_test_checked_product(values, sizeof(double), &bytes) != 0 ||
        gs_test_path_join(directory, "centroids_f64.bin", centroid_path,
                          sizeof(centroid_path)) != 0 ||
        gs_test_file_size(centroid_path, &actual) != 0 || actual != bytes) {
        return 1;
    }
    return 0;
}


static int gspt_build_evaluation_entries(
    const char *manifest_path,
    const char *pca_directory,
    const char *kmeans_directory,
    const char *evaluation_directory,
    const GSPTestPCAProfile *profile,
    const GSPTestKMeansProfile *kmeans,
    GSTestChecksumEntry entries[GSPT_EVALUATION_ENTRY_COUNT],
    char paths[17][GS_TEST_PATH_LEN]
)
{
    static const char *const inputs[9] = {
        "manifest_events_272.csv", "train_rows.csv", "val_rows.csv",
        "train_assignments_u32.bin", "val_assignments_u32.bin",
        "pca_spec.txt", "source_pca_sha256.txt",
        "source_kmeans_summary.txt", "source_kmeans_sha256.txt"
    };
    size_t index;
    if (profile == NULL || kmeans == NULL ||
        strcmp(gs_test_basename(manifest_path), inputs[0]) != 0) {
        return 1;
    }
    entries[0].kind = "INPUT";
    entries[0].name = inputs[0];
    entries[0].path = manifest_path;
    if (strlen(profile->train_rows_path) >= GS_TEST_PATH_LEN ||
        strlen(profile->validation_rows_path) >= GS_TEST_PATH_LEN ||
        strlen(kmeans->train_assignments_path) >= GS_TEST_PATH_LEN ||
        strlen(kmeans->validation_assignments_path) >= GS_TEST_PATH_LEN) {
        return 1;
    }
    strcpy(paths[0], profile->train_rows_path);
    strcpy(paths[1], profile->validation_rows_path);
    strcpy(paths[2], kmeans->train_assignments_path);
    strcpy(paths[3], kmeans->validation_assignments_path);
    if (gs_test_path_join(pca_directory, "pca_spec.txt", paths[4],
                          GS_TEST_PATH_LEN) != 0 ||
        gs_test_path_join(pca_directory, "sha256.txt", paths[5],
                          GS_TEST_PATH_LEN) != 0 ||
        gs_test_path_join(kmeans_directory, "summary.txt", paths[6],
                          GS_TEST_PATH_LEN) != 0 ||
        gs_test_path_join(kmeans_directory, "sha256.txt", paths[7],
                          GS_TEST_PATH_LEN) != 0) {
        return 1;
    }
    for (index = 1u; index < 9u; index++) {
        entries[index].kind = "INPUT";
        entries[index].name = inputs[index];
        entries[index].path = paths[index - 1u];
    }
    for (index = 0; index < 9u; index++) {
        if (gs_test_path_join(
                evaluation_directory, gs_test_historical_artifacts[index],
                paths[index + 8u], GS_TEST_PATH_LEN
            ) != 0) {
            return 1;
        }
        entries[index + 9u].kind = "ARTIFACT";
        entries[index + 9u].name = gs_test_historical_artifacts[index];
        entries[index + 9u].path = paths[index + 8u];
    }
    return 0;
}


static int gspt_validate_evaluation_inventory(
    const char *manifest_path,
    const char *pca_directory,
    const char *kmeans_directory,
    const char *evaluation_directory,
    size_t k,
    const GSPTestPCAProfile *profile,
    const GSPTestKMeansProfile *kmeans
)
{
    GSTestChecksumEntry entries[GSPT_EVALUATION_ENTRY_COUNT];
    char paths[17][GS_TEST_PATH_LEN];
    char pca_canonical[GS_TEST_PATH_LEN];
    char evaluation_canonical[GS_TEST_PATH_LEN];
    char expected_parent[GS_TEST_PATH_LEN];
    char expected_path[GS_TEST_PATH_LEN];
    char expected_name[32];
    char sha_path[GS_TEST_PATH_LEN];
    char partial_path[GS_TEST_PATH_LEN];
    size_t index, size;
    int regular, is_directory, written;
    if ((written = snprintf(expected_name, sizeof(expected_name), "k%lu",
                            (unsigned long)k)) < 0 ||
        (size_t)written >= sizeof(expected_name) ||
        strcmp(gs_test_basename(evaluation_directory), expected_name) != 0 ||
        strstr(gs_test_basename(evaluation_directory),
               GS_TEST_PARTIAL_SUFFIX) != NULL ||
        gs_test_canonical_existing(pca_directory, pca_canonical,
                                   sizeof(pca_canonical)) != 0 ||
        gs_test_canonical_existing(evaluation_directory,
                                   evaluation_canonical,
                                   sizeof(evaluation_canonical)) != 0 ||
        gs_test_path_join(pca_canonical, "label_evaluation_seed123",
                          expected_parent, sizeof(expected_parent)) != 0 ||
        gs_test_path_join(expected_parent, expected_name, expected_path,
                          sizeof(expected_path)) != 0 ||
        strcmp(evaluation_canonical, expected_path) != 0 ||
        gs_test_exact_files(evaluation_directory, gs_test_output_files, 10u) != 0) {
        return 1;
    }
    for (index = 0; index < 10u; index++) {
        char path[GS_TEST_PATH_LEN];
        if (gs_test_path_join(evaluation_directory, gs_test_output_files[index],
                              path, sizeof(path)) != 0 ||
            gs_test_file_size(path, &size) != 0 || size == 0u) {
            return 1;
        }
    }
    if (gspt_build_evaluation_entries(
            manifest_path, pca_directory, kmeans_directory,
            evaluation_directory, profile, kmeans, entries, paths
        ) != 0 ||
        gs_test_path_join(evaluation_directory, "sha256.txt", sha_path,
                          sizeof(sha_path)) != 0 ||
        gs_test_verify_checksum_manifest(
            sha_path, entries, GSPT_EVALUATION_ENTRY_COUNT
        ) != 0 ||
        strlen(evaluation_directory) >
            SIZE_MAX - strlen(GS_TEST_PARTIAL_SUFFIX) ||
        strlen(evaluation_directory) + strlen(GS_TEST_PARTIAL_SUFFIX) >=
            sizeof(partial_path)) {
        return 1;
    }
    strcpy(partial_path, evaluation_directory);
    strcat(partial_path, GS_TEST_PARTIAL_SUFFIX);
    if (gs_test_path_state(partial_path, &regular, &is_directory) != 0) {
        return 1;
    }
    return 0;
}


static int gspt_validate_summary(
    const char *directory,
    size_t k,
    const char *commit,
    const GSPTestPCAProfile *profile,
    const GSPTestKMeansProfile *kmeans,
    const GSBaselineClusteringMetrics *train,
    const GSBaselineClusteringMetrics *validation
)
{
    static const char *const keys[] = {
        "format_version", "profile", "input_profile", "evaluation_profile",
        "kmeans_profile", "pca_profile", "pca_threshold",
        "original_feature_count", "evaluation", "labels_policy",
        "label_count", "class_count", "event_count", "seed", "train_rows",
        "validation_rows", "labels_used_for_training", "labels_used_for_pca",
        "labels_used_for_kmeans", "labels_used_for_clustering",
        "labels_used_for_evaluation", "pca", "pca_whitening",
        "post_pca_standardization", "augmentation", "metric_inputs",
        "centroids_used_for_metrics", "pca_scores_used_for_metrics",
        "nmi_normalization", "purity_definition"
    };
    static const char *const metric_keys[10] = {
        "train_purity", "train_nmi_arithmetic", "train_ari",
        "train_homogeneity", "train_completeness", "validation_purity",
        "validation_nmi_arithmetic", "validation_ari",
        "validation_homogeneity", "validation_completeness"
    };
    double metric_values[10];
    char evaluation_profile[96];
    char kmeans_profile[96];
    char path[GS_TEST_PATH_LEN];
    char value[256];
    char count_text[32];
    char k_text[32];
    size_t index;
    int written;
    if (profile == NULL || kmeans == NULL || train == NULL ||
        validation == NULL ||
        (written = snprintf(evaluation_profile, sizeof(evaluation_profile),
                            "%s_label_evaluation", profile->profile)) < 0 ||
        (size_t)written >= sizeof(evaluation_profile) ||
        (written = snprintf(kmeans_profile, sizeof(kmeans_profile),
                            "%s_kmeans", profile->profile)) < 0 ||
        (size_t)written >= sizeof(kmeans_profile) ||
        (written = snprintf(count_text, sizeof(count_text), "%lu",
                            (unsigned long)profile->retained_components)) < 0 ||
        (size_t)written >= sizeof(count_text) ||
        (written = snprintf(k_text, sizeof(k_text), "%lu",
                            (unsigned long)k)) < 0 ||
        (size_t)written >= sizeof(k_text) ||
        gs_test_path_join(directory, "summary.txt", path, sizeof(path)) != 0) {
        return 1;
    }
    {
        const char *const expected[] = {
            "1", profile->profile, profile->profile, evaluation_profile,
            kmeans_profile, profile->token, profile->threshold_text, "1024",
            "labels_after_clustering", "posterior_evaluation_only", "22",
            "22", "8583", "123", "6866", "1717", "false", "false",
            "false", "false", "true", "true", "false", "false", "false",
            "labels_and_frozen_assignments_only", "false", "false",
            "arithmetic", "weighted_cluster_majority"
        };
        for (index = 0; index < sizeof(keys) / sizeof(keys[0]); index++) {
            if (gs_test_read_summary_value(path, keys[index], value,
                                           sizeof(value)) != 0 ||
                strcmp(value, expected[index]) != 0) {
                return 1;
            }
        }
    }
    if (gs_test_read_summary_value(path, "retained_component_count",
                                   value, sizeof(value)) != 0 ||
        strcmp(value, count_text) != 0 ||
        gs_test_read_summary_value(path, "k", value, sizeof(value)) != 0 ||
        strcmp(value, k_text) != 0 ||
        gs_test_read_summary_value(path, "cluster_count", value,
                                   sizeof(value)) != 0 ||
        strcmp(value, k_text) != 0 ||
        gs_test_read_summary_value(path, "input_pca_code_commit",
                                   value, sizeof(value)) != 0 ||
        strcmp(value, profile->code_commit) != 0 ||
        gs_test_read_summary_value(path, "input_kmeans_code_commit",
                                   value, sizeof(value)) != 0 ||
        strcmp(value, kmeans->code_commit) != 0 ||
        gs_test_read_summary_value(path, "code_commit", value,
                                   sizeof(value)) != 0 ||
        strcmp(value, commit) != 0) {
        return 1;
    }
    metric_values[0] = train->purity;
    metric_values[1] = train->normalized_mutual_information;
    metric_values[2] = train->adjusted_rand_index;
    metric_values[3] = train->homogeneity;
    metric_values[4] = train->completeness;
    metric_values[5] = validation->purity;
    metric_values[6] = validation->normalized_mutual_information;
    metric_values[7] = validation->adjusted_rand_index;
    metric_values[8] = validation->homogeneity;
    metric_values[9] = validation->completeness;
    for (index = 0; index < 10u; index++) {
        double parsed;
        if (gs_test_read_summary_value(path, metric_keys[index],
                                       value, sizeof(value)) != 0 ||
            gs_test_parse_double(value, &parsed) != 0 ||
            !gs_test_close(parsed, metric_values[index])) {
            return 1;
        }
    }
    for (index = 1u; index <= 5u; index++) {
        char key[128];
        if ((written = snprintf(key, sizeof(key),
                                "largest_train_cluster_top_%lu_label",
                                (unsigned long)index)) < 0 ||
            (size_t)written >= sizeof(key) ||
            gs_test_read_summary_value(path, key, value, sizeof(value)) != 0 ||
            gs_test_label_index(value) < 0) {
            return 1;
        }
    }
    return 0;
}


static void gspt_usage(const char *program)
{
    printf(
        "Usage: %s <manifest> <pca_directory> <pca_profile> "
        "<kmeans_directory> <evaluation_directory> <k> <code_commit>\n"
        "pca_profile: pca90 | pca95 | pca99\n",
        program
    );
}


int main(int argc, char **argv)
{
    GSEventManifest manifest;
    GSBaselineClusteringMetrics train_metrics;
    GSBaselineClusteringMetrics validation_metrics;
    GSPTestPCAProfile profile;
    GSPTestKMeansProfile kmeans;
    unsigned char *coverage = NULL;
    uint32_t *train_labels = NULL;
    uint32_t *validation_labels = NULL;
    uint32_t *train_assignments = NULL;
    uint32_t *validation_assignments = NULL;
    uint32_t *train_copy = NULL;
    uint32_t *validation_copy = NULL;
    char pca_names[GSPT_PCA_FILE_COUNT][64];
    const char *pca_name_ptrs[GSPT_PCA_FILE_COUNT];
    char pca_before[GSPT_PCA_FILE_COUNT][GS_TEST_SHA_TEXT];
    char pca_after[GSPT_PCA_FILE_COUNT][GS_TEST_SHA_TEXT];
    char kmeans_before[GSPT_KMEANS_FILE_COUNT][GS_TEST_SHA_TEXT];
    char kmeans_after[GSPT_KMEANS_FILE_COUNT][GS_TEST_SHA_TEXT];
    char base_before[10][GS_TEST_SHA_TEXT];
    char base_after[10][GS_TEST_SHA_TEXT];
    size_t train_bytes, validation_bytes, k, index;
    int train_metrics_passed = 0;
    int validation_metrics_passed = 0;
    int rc = 1;

    memset(&manifest, 0, sizeof(manifest));
    memset(&train_metrics, 0, sizeof(train_metrics));
    memset(&validation_metrics, 0, sizeof(validation_metrics));
    memset(&profile, 0, sizeof(profile));
    memset(&kmeans, 0, sizeof(kmeans));

    if (argc == 2 && strcmp(argv[1], "--help") == 0) {
        gspt_usage(argv[0]);
        return 0;
    }
    if (argc != 8 || !gs_test_platform_is_valid() ||
        setlocale(LC_NUMERIC, "C") == NULL ||
        gs_test_parse_size(argv[6], &k) != 0 || (k != 22u && k != 36u) ||
        !gspt_commit_valid(argv[7]) ||
        gs_test_checked_product(GS_TEST_TRAIN_ROWS, sizeof(uint32_t),
                                &train_bytes) != 0 ||
        gs_test_checked_product(GS_TEST_VALIDATION_ROWS, sizeof(uint32_t),
                                &validation_bytes) != 0 ||
        gspt_validate_pca(argv[1], argv[2], argv[3], &profile) != 0 ||
        gspt_validate_kmeans(argv[2], argv[4], k, &profile, &kmeans) != 0) {
        fprintf(stderr, "ERROR: invalid validator arguments or inputs.\n");
        return 1;
    }
    printf("[PASS] frozen_pca_and_kmeans_contract\n");

    {
        static const char *const fixed[10] = {
            "components_f64.bin",
            "cumulative_explained_variance_ratio_f64.bin",
            "eigenvalues_f64.bin",
            "explained_variance_ratio_f64.bin",
            "mean_f64.bin", "pca_spec.txt", "sha256.txt", "spectrum.csv",
            "summary.txt", "thresholds.csv"
        };
        for (index = 0; index < 10u; index++) {
            strcpy(pca_names[index], fixed[index]);
        }
        strcpy(pca_names[10], profile.train_score_name);
        strcpy(pca_names[11], profile.validation_score_name);
        for (index = 0; index < GSPT_PCA_FILE_COUNT; index++) {
            pca_name_ptrs[index] = pca_names[index];
        }
    }
    if (gs_test_snapshot_hashes(profile.source_directory, gs_test_base_files,
                                10u, base_before) != 0 ||
        gs_test_snapshot_hashes(argv[2], pca_name_ptrs,
                                GSPT_PCA_FILE_COUNT, pca_before) != 0 ||
        gs_test_snapshot_hashes(argv[4], gspt_kmeans_files,
                                GSPT_KMEANS_FILE_COUNT, kmeans_before) != 0 ||
        gspt_validate_evaluation_inventory(
            argv[1], argv[2], argv[4], argv[5], k, &profile, &kmeans
        ) != 0) {
        fprintf(stderr, "[FAIL] evaluation_inventory_and_checksums\n");
        goto cleanup;
    }
    printf("[PASS] evaluation_inventory_and_checksums\n");

    coverage = (unsigned char *)calloc(GS_TEST_MANIFEST_ROWS, 1u);
    train_labels = (uint32_t *)malloc(train_bytes);
    validation_labels = (uint32_t *)malloc(validation_bytes);
    train_assignments = (uint32_t *)malloc(train_bytes);
    validation_assignments = (uint32_t *)malloc(validation_bytes);
    train_copy = (uint32_t *)malloc(train_bytes);
    validation_copy = (uint32_t *)malloc(validation_bytes);
    if (coverage == NULL || train_labels == NULL ||
        validation_labels == NULL || train_assignments == NULL ||
        validation_assignments == NULL || train_copy == NULL ||
        validation_copy == NULL ||
        gs_event_manifest_load(argv[1], &manifest) != 0 ||
        gs_test_validate_manifest(&manifest) != 0) {
        fprintf(stderr, "[FAIL] manifest_and_label_contract\n");
        goto cleanup;
    }
    printf("[PASS] manifest_and_label_contract\n");

    if (gs_test_load_rows(
            profile.train_rows_path, "train", GS_TEST_TRAIN_ROWS, &manifest,
            coverage, train_labels
        ) != 0 ||
        gs_test_load_rows(
            profile.validation_rows_path, "validation",
            GS_TEST_VALIDATION_ROWS, &manifest, coverage, validation_labels
        ) != 0 ||
        gs_test_read_assignments(
            kmeans.train_assignments_path, train_assignments,
            GS_TEST_TRAIN_ROWS, k
        ) != 0 ||
        gs_test_read_assignments(
            kmeans.validation_assignments_path, validation_assignments,
            GS_TEST_VALIDATION_ROWS, k
        ) != 0) {
        fprintf(stderr, "[FAIL] row_mapping_assignments_and_coverage\n");
        goto cleanup;
    }
    for (index = 0; index < GS_TEST_MANIFEST_ROWS; index++) {
        if (coverage[index] != 1u) {
            fprintf(stderr, "[FAIL] row_mapping_assignments_and_coverage\n");
            goto cleanup;
        }
    }
    memcpy(train_copy, train_assignments, train_bytes);
    memcpy(validation_copy, validation_assignments, validation_bytes);
    printf("[PASS] row_mapping_assignments_and_coverage\n");

    if (gs_baseline_clustering_metrics_compute(
            train_labels, train_assignments, GS_TEST_TRAIN_ROWS,
            GS_TEST_LABEL_COUNT, k, &train_metrics
        ) != GS_BASELINE_CLUSTERING_METRICS_SUCCESS ||
        gs_baseline_clustering_metrics_compute(
            validation_labels, validation_assignments,
            GS_TEST_VALIDATION_ROWS, GS_TEST_LABEL_COUNT, k,
            &validation_metrics
        ) != GS_BASELINE_CLUSTERING_METRICS_SUCCESS ||
        !gs_test_metrics_consistent(&train_metrics, GS_TEST_TRAIN_ROWS, k) ||
        !gs_test_metrics_consistent(
            &validation_metrics, GS_TEST_VALIDATION_ROWS, k
        )) {
        fprintf(stderr, "[FAIL] independent_metric_recomputation\n");
        goto cleanup;
    }
    printf("[PASS] independent_metric_recomputation\n");

    if (gs_test_validate_label_map(argv[5]) != 0 ||
        gs_test_validate_contingency(
            argv[5], "train_contingency_counts.csv", &train_metrics
        ) != 0 ||
        gs_test_validate_contingency(
            argv[5], "val_contingency_counts.csv", &validation_metrics
        ) != 0 ||
        gs_test_validate_metrics_csv(
            argv[5], k, &train_metrics, &validation_metrics,
            &train_metrics_passed, &validation_metrics_passed
        ) != 0 || !train_metrics_passed || !validation_metrics_passed ||
        gs_test_validate_cluster_summary(
            argv[5], "train_cluster_summary.csv", &train_metrics
        ) != 0 ||
        gs_test_validate_cluster_summary(
            argv[5], "val_cluster_summary.csv", &validation_metrics
        ) != 0 ||
        gs_test_validate_label_summary(
            argv[5], "train_label_summary.csv", &train_metrics
        ) != 0 ||
        gs_test_validate_label_summary(
            argv[5], "val_label_summary.csv", &validation_metrics
        ) != 0) {
        fprintf(stderr, "[FAIL] independent_csv_reconstruction\n");
        goto cleanup;
    }
    printf("[PASS] independent_csv_reconstruction\n");

    if (gspt_validate_summary(
            argv[5], k, argv[7], &profile, &kmeans,
            &train_metrics, &validation_metrics
        ) != 0) {
        fprintf(stderr, "[FAIL] deterministic_summary_contract\n");
        goto cleanup;
    }
    printf("[PASS] deterministic_summary_contract\n");

    if (memcmp(train_assignments, train_copy, train_bytes) != 0 ||
        memcmp(validation_assignments, validation_copy,
               validation_bytes) != 0 ||
        gspt_validate_pca(argv[1], argv[2], argv[3], &profile) != 0 ||
        gspt_validate_kmeans(argv[2], argv[4], k, &profile, &kmeans) != 0 ||
        gs_test_snapshot_hashes(profile.source_directory, gs_test_base_files,
                                10u, base_after) != 0 ||
        gs_test_snapshot_hashes(argv[2], pca_name_ptrs,
                                GSPT_PCA_FILE_COUNT, pca_after) != 0 ||
        gs_test_snapshot_hashes(argv[4], gspt_kmeans_files,
                                GSPT_KMEANS_FILE_COUNT, kmeans_after) != 0 ||
        !gs_test_hashes_equal(base_before, base_after, 10u) ||
        !gs_test_hashes_equal(pca_before, pca_after,
                              GSPT_PCA_FILE_COUNT) ||
        !gs_test_hashes_equal(kmeans_before, kmeans_after,
                              GSPT_KMEANS_FILE_COUNT)) {
        fprintf(stderr, "[FAIL] frozen_inputs_bitwise_unchanged\n");
        goto cleanup;
    }
    printf("[PASS] frozen_inputs_bitwise_unchanged\n");
    printf("LABELS_POLICY=posterior_evaluation_only\n");
    printf("METRIC_INPUTS=labels_and_frozen_assignments_only\n");
    printf("CENTROIDS_USED_FOR_METRICS=false\n");
    printf("PCA_SCORES_USED_FOR_METRICS=false\n");
    printf("ASSIGNMENTS_BITWISE_UNCHANGED=true\n");
    printf("PCA_PROFILE=%s\n", profile.token);
    printf("PCA_COMPONENTS=%lu\n",
           (unsigned long)profile.retained_components);
    printf("K=%lu\n", (unsigned long)k);
    printf("TRAIN_PURITY=%.17g\n", train_metrics.purity);
    printf("TRAIN_NMI_ARITHMETIC=%.17g\n",
           train_metrics.normalized_mutual_information);
    printf("TRAIN_ARI=%.17g\n", train_metrics.adjusted_rand_index);
    printf("TRAIN_HOMOGENEITY=%.17g\n", train_metrics.homogeneity);
    printf("TRAIN_COMPLETENESS=%.17g\n", train_metrics.completeness);
    printf("VALIDATION_PURITY=%.17g\n", validation_metrics.purity);
    printf("VALIDATION_NMI_ARITHMETIC=%.17g\n",
           validation_metrics.normalized_mutual_information);
    printf("VALIDATION_ARI=%.17g\n",
           validation_metrics.adjusted_rand_index);
    printf("VALIDATION_HOMOGENEITY=%.17g\n",
           validation_metrics.homogeneity);
    printf("VALIDATION_COMPLETENESS=%.17g\n",
           validation_metrics.completeness);
    printf("ALL MINI112 PCA EVALUATION ARTIFACT TESTS PASSED\n");
    rc = 0;

cleanup:
    gs_baseline_clustering_metrics_free(&validation_metrics);
    gs_baseline_clustering_metrics_free(&train_metrics);
    gs_event_manifest_free(&manifest);
    free(validation_copy);
    free(train_copy);
    free(validation_assignments);
    free(train_assignments);
    free(validation_labels);
    free(train_labels);
    free(coverage);
    return rc;
}
