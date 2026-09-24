#define main gs_legacy_mini112_evaluator_main
#include "evaluate_baseline_mini112_clusters.c"
#undef main

#undef GS_FAILURE_ENV
#undef GS_FAILURE_BASENAME
#define GS_FAILURE_ENV "GS_MINI112_PCA_EVALUATION_TEST_FAIL_AFTER_PARTIAL"

enum {
    GSP_PCA_FILE_COUNT = 12,
    GSP_KMEANS_FILE_COUNT = 13,
    GSP_EVALUATION_INPUT_COUNT = 9,
    GSP_EVALUATION_ARTIFACT_COUNT = 9,
    GSP_EVALUATION_ENTRY_COUNT = 18
};

static const char *const gsp_pca_fixed_files[GSP_PCA_FILE_COUNT] = {
    "components_f64.bin",
    "cumulative_explained_variance_ratio_f64.bin",
    "eigenvalues_f64.bin",
    "explained_variance_ratio_f64.bin",
    "mean_f64.bin",
    "pca_spec.txt",
    "sha256.txt",
    "spectrum.csv",
    "summary.txt",
    "thresholds.csv",
    NULL,
    NULL
};

static const char *const gsp_kmeans_files[GSP_KMEANS_FILE_COUNT] = {
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

typedef struct {
    char token[16];
    char profile[64];
    char threshold_text[16];
    double threshold;
    double cumulative_variance;
    size_t retained_components;
    size_t train_rows;
    size_t validation_rows;
    size_t original_features;
    char code_commit[65];
    char source_directory[GS_PATH_SIZE];
    char train_rows_path[GS_PATH_SIZE];
    char validation_rows_path[GS_PATH_SIZE];
    char train_score_name[64];
    char validation_score_name[64];
} GSPPCAProfile;

typedef struct {
    size_t k;
    char code_commit[65];
    char pca_code_commit[65];
    char train_assignments_path[GS_PATH_SIZE];
    char validation_assignments_path[GS_PATH_SIZE];
} GSPKMeansProfile;

static int gsp_commit_valid(const char *commit)
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

static int gsp_parent_path(
    const char *path,
    char *parent,
    size_t capacity
)
{
    size_t length;
    if (path == NULL || parent == NULL || capacity == 0u ||
        !gs_value_valid(path)) {
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

static int gsp_token_contract(
    const char *token,
    const char **threshold_text,
    double *threshold
)
{
    if (token == NULL || threshold_text == NULL || threshold == NULL) {
        return 1;
    }
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

static int gsp_read_key(
    const char *directory,
    const char *filename,
    const char *key,
    char *value,
    size_t capacity
)
{
    char path[GS_PATH_SIZE];
    return gs_join(directory, filename, path, sizeof(path)) != 0 ||
           gs_read_summary_value(path, key, value, capacity) != 0;
}

static int gsp_expect_key(
    const char *directory,
    const char *filename,
    const char *key,
    const char *expected
)
{
    char value[256];
    return expected == NULL ||
           gsp_read_key(directory, filename, key, value, sizeof(value)) != 0 ||
           strcmp(value, expected) != 0;
}

static int gsp_parse_double(const char *text, double *value)
{
    char *end;
    double parsed;
    if (text == NULL || value == NULL || text[0] == '\0') return 1;
    errno = 0;
    parsed = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !isfinite(parsed)) {
        return 1;
    }
    *value = parsed;
    return 0;
}

static int gsp_read_thresholds(
    const char *directory,
    const char *token,
    size_t *selected_count,
    double *selected_cumulative
)
{
    static const char *const names[3] = {"pca90", "pca95", "pca99"};
    static const double thresholds[3] = {0.90, 0.95, 0.99};
    char path[GS_PATH_SIZE];
    char line[GS_LINE_SIZE];
    FILE *file = NULL;
    size_t row;
    int selected = 0;
    int failed = 0;
    if (selected_count == NULL || selected_cumulative == NULL ||
        gs_join(directory, "thresholds.csv", path, sizeof(path)) != 0 ||
        (file = fopen(path, "rb")) == NULL) {
        return 1;
    }
    if (fgets(line, sizeof(line), file) == NULL) {
        failed = 1;
    } else {
        gs_strip_eol(line);
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
            gs_split_csv(line, fields, 3u) != 0 ||
            gsp_parse_double(fields[0], &threshold) != 0 ||
            gs_parse_size(fields[1], &count) != 0 ||
            gsp_parse_double(fields[2], &cumulative) != 0 ||
            threshold != thresholds[row] || count == 0u ||
            cumulative < threshold || cumulative > 1.0) {
            failed = 1;
            break;
        }
        if (strcmp(token, names[row]) == 0) {
            *selected_count = count;
            *selected_cumulative = cumulative;
            selected = 1;
        }
    }
    if (!failed && fgets(line, sizeof(line), file) != NULL) failed = 1;
    if (ferror(file) || fclose(file) != 0) failed = 1;
    return failed || !selected;
}

static int gsp_pca_inventory(
    const char *directory,
    const GSPPCAProfile *profile
)
{
    DIR *stream;
    struct dirent *entry;
    unsigned char seen[GSP_PCA_FILE_COUNT] = {0};
    size_t file_count = 0u;
    int kmeans_seen = 0;
    int evaluation_seen = 0;
    int failed = 0;
    if (profile == NULL || (stream = opendir(directory)) == NULL) return 1;
    while (!failed && (entry = readdir(stream)) != NULL) {
        char path[GS_PATH_SIZE];
        struct stat information;
        size_t index;
        const char *name = entry->d_name;
        if (strcmp(name, ".") == 0 || strcmp(name, "..") == 0) continue;
        if (gs_join(directory, name, path, sizeof(path)) != 0 ||
            stat(path, &information) != 0) {
            failed = 1;
            break;
        }
        if (strcmp(name, "kmeans_seed123") == 0) {
            if (kmeans_seen || !S_ISDIR(information.st_mode)) failed = 1;
            kmeans_seen = 1;
            continue;
        }
        if (strcmp(name, "label_evaluation_seed123") == 0) {
            if (evaluation_seen || !S_ISDIR(information.st_mode)) failed = 1;
            evaluation_seen = 1;
            continue;
        }
        if (strcmp(name, profile->train_score_name) == 0) {
            index = 10u;
        } else if (strcmp(name, profile->validation_score_name) == 0) {
            index = 11u;
        } else if (!gs_name_index(
                       name, gsp_pca_fixed_files, 10u, &index
                   )) {
            failed = 1;
            break;
        }
        if (!S_ISREG(information.st_mode) || seen[index]) {
            failed = 1;
            break;
        }
        seen[index] = 1u;
        file_count++;
    }
    if (closedir(stream) != 0) failed = 1;
    if (file_count != GSP_PCA_FILE_COUNT || !kmeans_seen) failed = 1;
    while (!failed && file_count != 0u) {
        file_count--;
        if (!seen[file_count]) failed = 1;
    }
    return failed;
}

static int gsp_build_pca_entries(
    const GSPPCAProfile *profile,
    const char *pca_directory,
    GSChecksumEntry entries[18],
    char paths[18][GS_PATH_SIZE]
)
{
    static const char *const input_names[7] = {
        "feature_spec.txt", "scaler.csv", "train_features_std_f32.bin",
        "val_features_std_f32.bin", "train_rows.csv", "val_rows.csv",
        "sha256.txt"
    };
    static const char *const artifact_fixed[8] = {
        "pca_spec.txt", "mean_f64.bin", "components_f64.bin",
        "eigenvalues_f64.bin", "explained_variance_ratio_f64.bin",
        "cumulative_explained_variance_ratio_f64.bin", "thresholds.csv",
        "spectrum.csv"
    };
    size_t index;
    if (profile == NULL) return 1;
    for (index = 0; index < 7u; index++) {
        if (gs_join(profile->source_directory, input_names[index],
                    paths[index], GS_PATH_SIZE) != 0) {
            return 1;
        }
        entries[index].kind = "INPUT";
        entries[index].name = input_names[index];
        entries[index].path = paths[index];
    }
    for (index = 0; index < 8u; index++) {
        if (gs_join(pca_directory, artifact_fixed[index],
                    paths[index + 7u], GS_PATH_SIZE) != 0) {
            return 1;
        }
        entries[index + 7u].kind = "ARTIFACT";
        entries[index + 7u].name = artifact_fixed[index];
        entries[index + 7u].path = paths[index + 7u];
    }
    if (gs_join(pca_directory, profile->train_score_name,
                paths[15], GS_PATH_SIZE) != 0 ||
        gs_join(pca_directory, profile->validation_score_name,
                paths[16], GS_PATH_SIZE) != 0 ||
        gs_join(pca_directory, "summary.txt",
                paths[17], GS_PATH_SIZE) != 0) {
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

static int gsp_validate_pca(
    const char *manifest_path,
    const char *directory,
    const char *token,
    GSPPCAProfile *profile
)
{
    GSChecksumEntry entries[18];
    char paths[18][GS_PATH_SIZE];
    char parent[GS_PATH_SIZE];
    char expected_profile[64];
    char value[256];
    char manifest_path_sha[GS_PATH_SIZE];
    const char *threshold_text;
    size_t selected_count = 0u;
    size_t train_score_values, validation_score_values;
    size_t train_score_bytes, validation_score_bytes;
    size_t actual;
    double threshold;
    double selected_cumulative = 0.0;
    int written;
    if (profile == NULL ||
        gsp_token_contract(token, &threshold_text, &threshold) != 0 ||
        (written = snprintf(expected_profile, sizeof(expected_profile),
                            "mini112_pool7_%s", token)) < 0 ||
        (size_t)written >= sizeof(expected_profile) ||
        strcmp(gs_basename(directory), expected_profile) != 0 ||
        gsp_parent_path(directory, parent, sizeof(parent)) != 0 ||
        gs_join(parent, "mini112_pool7_no_pca", profile->source_directory,
                sizeof(profile->source_directory)) != 0) {
        fprintf(stderr, "Errore PCA: profilo o percorso non coerente.\n");
        return 1;
    }
    if (gs_validate_base(manifest_path, profile->source_directory) != 0) {
        fprintf(stderr, "Errore PCA: baseline sorgente non valida.\n");
        return 1;
    }
    if (
        (written = snprintf(profile->train_score_name,
                            sizeof(profile->train_score_name),
                            "train_scores_%s_f32.bin", token)) < 0 ||
        (size_t)written >= sizeof(profile->train_score_name) ||
        (written = snprintf(profile->validation_score_name,
                            sizeof(profile->validation_score_name),
                            "val_scores_%s_f32.bin", token)) < 0 ||
        (size_t)written >= sizeof(profile->validation_score_name) ||
        gsp_read_thresholds(
            directory, token, &selected_count, &selected_cumulative
        ) != 0) {
        fprintf(stderr, "Errore PCA: soglie o nomi score non validi.\n");
        return 1;
    }
    memset(profile->token, 0, sizeof(profile->token));
    memset(profile->profile, 0, sizeof(profile->profile));
    memset(profile->threshold_text, 0, sizeof(profile->threshold_text));
    strcpy(profile->token, token);
    strcpy(profile->profile, expected_profile);
    strcpy(profile->threshold_text, threshold_text);
    profile->threshold = threshold;
    profile->retained_components = selected_count;
    profile->cumulative_variance = selected_cumulative;
    profile->train_rows = GS_TRAIN_ROWS;
    profile->validation_rows = GS_VALIDATION_ROWS;
    profile->original_features = 1024u;

    if (gsp_expect_key(directory, "pca_spec.txt", "profile",
                       profile->profile) ||
        gsp_expect_key(directory, "pca_spec.txt", "source_profile",
                       "mini112_pool7_no_pca") ||
        gsp_expect_key(directory, "pca_spec.txt", "selected_threshold",
                       profile->threshold_text) ||
        gsp_expect_key(directory, "pca_spec.txt", "original_feature_count",
                       "1024") ||
        gsp_expect_key(directory, "pca_spec.txt", "train_rows", "6866") ||
        gsp_expect_key(directory, "pca_spec.txt", "validation_rows", "1717") ||
        gsp_expect_key(directory, "pca_spec.txt", "score_dtype", "float32") ||
        gsp_expect_key(directory, "pca_spec.txt", "score_order", "row_major") ||
        gsp_expect_key(directory, "pca_spec.txt", "fit_input", "train_only") ||
        gsp_expect_key(directory, "pca_spec.txt",
                       "validation_transform_without_refit", "true") ||
        gsp_expect_key(directory, "pca_spec.txt", "whitening", "false") ||
        gsp_expect_key(directory, "pca_spec.txt",
                       "post_standardization", "false") ||
        gsp_expect_key(directory, "pca_spec.txt", "augmentation", "false") ||
        gsp_expect_key(directory, "pca_spec.txt", "labels_used", "false") ||
        gsp_read_key(directory, "pca_spec.txt", "retained_component_count",
                     value, sizeof(value)) != 0 ||
        gs_parse_size(value, &actual) != 0 || actual != selected_count ||
        gsp_expect_key(directory, "summary.txt", "profile",
                       profile->profile) ||
        gsp_expect_key(directory, "summary.txt", "selected_threshold",
                       profile->threshold_text) ||
        gsp_expect_key(directory, "summary.txt", "pca_fit", "train_only") ||
        gsp_expect_key(directory, "summary.txt",
                       "validation_transformed_without_refit", "true") ||
        gsp_expect_key(directory, "summary.txt", "whitening", "false") ||
        gsp_expect_key(directory, "summary.txt",
                       "post_standardization", "false") ||
        gsp_expect_key(directory, "summary.txt", "augmentation", "false") ||
        gsp_expect_key(directory, "summary.txt", "labels_used", "false") ||
        gsp_read_key(directory, "summary.txt", "retained_component_count",
                     value, sizeof(value)) != 0 ||
        gs_parse_size(value, &actual) != 0 || actual != selected_count ||
        gsp_read_key(directory, "summary.txt", "code_commit",
                     profile->code_commit,
                     sizeof(profile->code_commit)) != 0 ||
        !gsp_commit_valid(profile->code_commit)) {
        fprintf(stderr, "Errore PCA: metadata non coerenti.\n");
        return 1;
    }
    if (gsp_pca_inventory(directory, profile) != 0) {
        fprintf(stderr, "Errore PCA: inventario non valido.\n");
        return 1;
    }
    if (gsp_build_pca_entries(profile, directory, entries, paths) != 0 ||
        gs_join(directory, "sha256.txt", manifest_path_sha,
                sizeof(manifest_path_sha)) != 0 ||
        gs_verify_checksum_manifest(manifest_path_sha, entries, 18u) != 0) {
        fprintf(stderr, "Errore PCA: checksum non validi.\n");
        return 1;
    }
    if (
        gs_join(profile->source_directory, "train_rows.csv",
                profile->train_rows_path,
                sizeof(profile->train_rows_path)) != 0 ||
        gs_join(profile->source_directory, "val_rows.csv",
                profile->validation_rows_path,
                sizeof(profile->validation_rows_path)) != 0 ||
        gs_checked_product(GS_TRAIN_ROWS, selected_count,
                           &train_score_values) != 0 ||
        gs_checked_product(train_score_values, sizeof(float),
                           &train_score_bytes) != 0 ||
        gs_checked_product(GS_VALIDATION_ROWS, selected_count,
                           &validation_score_values) != 0 ||
        gs_checked_product(validation_score_values, sizeof(float),
                           &validation_score_bytes) != 0 ||
        gs_file_size(paths[15], &actual) != 0 || actual != train_score_bytes ||
        gs_file_size(paths[16], &actual) != 0 ||
        actual != validation_score_bytes) {
        fprintf(stderr, "Errore PCA: dimensioni o percorsi dati non validi.\n");
        return 1;
    }
    return 0;
}

static int gsp_kmeans_inventory(const char *directory)
{
    return gs_exact_files(
        directory, gsp_kmeans_files, GSP_KMEANS_FILE_COUNT
    );
}

static int gsp_validate_kmeans_location(
    const char *pca_directory,
    const char *kmeans_directory,
    size_t k
)
{
    char pca[GS_PATH_SIZE];
    char actual[GS_PATH_SIZE];
    char expected[GS_PATH_SIZE];
    char suffix[64];
    int written;
    if (gs_canonical_existing(pca_directory, pca, sizeof(pca)) != 0 ||
        gs_canonical_existing(kmeans_directory, actual, sizeof(actual)) != 0 ||
        (written = snprintf(suffix, sizeof(suffix),
                            "kmeans_seed123/k%lu",
                            (unsigned long)k)) < 0 ||
        (size_t)written >= sizeof(suffix) ||
        gs_join(pca, suffix, expected, sizeof(expected)) != 0) {
        return 1;
    }
    return strcmp(actual, expected) == 0 ? 0 : 1;
}

static int gsp_build_kmeans_entries(
    const char *pca_directory,
    const char *directory,
    const GSPPCAProfile *profile,
    GSChecksumEntry entries[17],
    char paths[17][GS_PATH_SIZE]
)
{
    const char *input_names[5];
    static const char *const artifact_names[12] = {
        "centroids_f64.bin", "train_assignments_u32.bin",
        "val_assignments_u32.bin", "train_squared_distances_f64.bin",
        "val_squared_distances_f64.bin", "cluster_sizes_train_u64.bin",
        "initial_centroid_indices_u64.bin", "train_assignments.csv",
        "val_assignments.csv", "cluster_sizes.csv", "initial_centroids.csv",
        "summary.txt"
    };
    size_t index;
    input_names[0] = "pca_spec.txt";
    input_names[1] = "thresholds.csv";
    input_names[2] = profile->train_score_name;
    input_names[3] = profile->validation_score_name;
    input_names[4] = "source_pca_sha256.txt";
    for (index = 0; index < 5u; index++) {
        const char *physical = index == 4u ? "sha256.txt" : input_names[index];
        if (gs_join(pca_directory, physical, paths[index],
                    GS_PATH_SIZE) != 0) {
            return 1;
        }
        entries[index].kind = "INPUT";
        entries[index].name = input_names[index];
        entries[index].path = paths[index];
    }
    for (index = 0; index < 12u; index++) {
        if (gs_join(directory, artifact_names[index], paths[index + 5u],
                    GS_PATH_SIZE) != 0) {
            return 1;
        }
        entries[index + 5u].kind = "ARTIFACT";
        entries[index + 5u].name = artifact_names[index];
        entries[index + 5u].path = paths[index + 5u];
    }
    return 0;
}

static int gsp_validate_kmeans(
    const char *pca_directory,
    const char *directory,
    size_t k,
    const GSPPCAProfile *profile,
    GSPKMeansProfile *metadata
)
{
    GSChecksumEntry entries[17];
    char paths[17][GS_PATH_SIZE];
    char manifest_path[GS_PATH_SIZE];
    char k_text[32];
    char components_text[32];
    char centroid_path[GS_PATH_SIZE];
    size_t values, bytes, actual;
    int written;
    if (metadata == NULL || profile == NULL ||
        (k != 22u && k != 36u) ||
        gsp_validate_kmeans_location(pca_directory, directory, k) != 0 ||
        gsp_kmeans_inventory(directory) != 0 ||
        (written = snprintf(k_text, sizeof(k_text), "%lu",
                            (unsigned long)k)) < 0 ||
        (size_t)written >= sizeof(k_text) ||
        (written = snprintf(components_text, sizeof(components_text), "%lu",
                            (unsigned long)profile->retained_components)) < 0 ||
        (size_t)written >= sizeof(components_text)) {
        return 1;
    }
    if (gsp_expect_key(directory, "summary.txt", "profile",
                       "mini112_pool7_pca90_kmeans") == 0) {
        if (strcmp(profile->token, "pca90") != 0) return 1;
    } else if (gsp_expect_key(directory, "summary.txt", "profile",
                              "mini112_pool7_pca95_kmeans") == 0) {
        if (strcmp(profile->token, "pca95") != 0) return 1;
    } else if (gsp_expect_key(directory, "summary.txt", "profile",
                              "mini112_pool7_pca99_kmeans") == 0) {
        if (strcmp(profile->token, "pca99") != 0) return 1;
    } else {
        return 1;
    }
    if (gsp_expect_key(directory, "summary.txt", "input_profile",
                       profile->profile) ||
        gsp_expect_key(directory, "summary.txt", "pca_profile",
                       profile->token) ||
        gsp_expect_key(directory, "summary.txt", "pca_threshold",
                       profile->threshold_text) ||
        gsp_expect_key(directory, "summary.txt", "retained_component_count",
                       components_text) ||
        gsp_expect_key(directory, "summary.txt", "original_feature_count",
                       "1024") ||
        gsp_expect_key(directory, "summary.txt", "train_rows", "6866") ||
        gsp_expect_key(directory, "summary.txt", "validation_rows", "1717") ||
        gsp_expect_key(directory, "summary.txt", "feature_count",
                       components_text) ||
        gsp_expect_key(directory, "summary.txt", "k", k_text) ||
        gsp_expect_key(directory, "summary.txt", "seed", "123") ||
        gsp_expect_key(directory, "summary.txt", "max_iterations", "300") ||
        gsp_expect_key(directory, "summary.txt", "converged", "1") ||
        gsp_expect_key(directory, "summary.txt", "fit_input", "train_only") ||
        gsp_expect_key(directory, "summary.txt",
                       "validation_assigned_after_fit", "true") ||
        gsp_expect_key(directory, "summary.txt", "pca", "true") ||
        gsp_expect_key(directory, "summary.txt", "whitening", "false") ||
        gsp_expect_key(directory, "summary.txt",
                       "post_pca_standardization", "false") ||
        gsp_expect_key(directory, "summary.txt", "augmentation", "false") ||
        gsp_expect_key(directory, "summary.txt", "labels_used", "false") ||
        gsp_read_key(directory, "summary.txt", "pca_code_commit",
                     metadata->pca_code_commit,
                     sizeof(metadata->pca_code_commit)) != 0 ||
        strcmp(metadata->pca_code_commit, profile->code_commit) != 0 ||
        gsp_read_key(directory, "summary.txt", "code_commit",
                     metadata->code_commit,
                     sizeof(metadata->code_commit)) != 0 ||
        !gsp_commit_valid(metadata->code_commit) ||
        gsp_build_kmeans_entries(
            pca_directory, directory, profile, entries, paths
        ) != 0 ||
        gs_join(directory, "sha256.txt", manifest_path,
                sizeof(manifest_path)) != 0 ||
        gs_verify_checksum_manifest(manifest_path, entries, 17u) != 0 ||
        gs_join(directory, "train_assignments_u32.bin",
                metadata->train_assignments_path,
                sizeof(metadata->train_assignments_path)) != 0 ||
        gs_join(directory, "val_assignments_u32.bin",
                metadata->validation_assignments_path,
                sizeof(metadata->validation_assignments_path)) != 0 ||
        gs_file_size(metadata->train_assignments_path, &actual) != 0 ||
        actual != GS_TRAIN_ROWS * sizeof(uint32_t) ||
        gs_file_size(metadata->validation_assignments_path, &actual) != 0 ||
        actual != GS_VALIDATION_ROWS * sizeof(uint32_t) ||
        gs_checked_product(k, profile->retained_components, &values) != 0 ||
        gs_checked_product(values, sizeof(double), &bytes) != 0 ||
        gs_join(directory, "centroids_f64.bin", centroid_path,
                sizeof(centroid_path)) != 0 ||
        gs_file_size(centroid_path, &actual) != 0 || actual != bytes) {
        return 1;
    }
    metadata->k = k;
    return 0;
}

static int gsp_output_paths(
    const char *pca_directory,
    const char *partial,
    size_t k,
    char *final,
    size_t capacity
)
{
    char parent[GS_PATH_SIZE];
    char parent_canonical[GS_PATH_SIZE];
    char pca_canonical[GS_PATH_SIZE];
    char expected_parent[GS_PATH_SIZE];
    char partial_name[32];
    char final_name[32];
    size_t length;
    int written;
    int parent_state;
    int parent_is_directory;
    if ((written = snprintf(partial_name, sizeof(partial_name),
                            "k%lu.partial", (unsigned long)k)) < 0 ||
        (size_t)written >= sizeof(partial_name) ||
        (written = snprintf(final_name, sizeof(final_name),
                            "k%lu", (unsigned long)k)) < 0 ||
        (size_t)written >= sizeof(final_name) ||
        strcmp(gs_basename(partial), partial_name) != 0 ||
        gsp_parent_path(partial, parent, sizeof(parent)) != 0 ||
        (parent_state = gs_path_state(
            parent, NULL, &parent_is_directory
        )) != 1 || !parent_is_directory ||
        gs_canonical_existing(parent, parent_canonical,
                              sizeof(parent_canonical)) != 0 ||
        gs_canonical_existing(pca_directory, pca_canonical,
                              sizeof(pca_canonical)) != 0 ||
        gs_join(pca_canonical, "label_evaluation_seed123",
                expected_parent, sizeof(expected_parent)) != 0 ||
        strcmp(parent_canonical, expected_parent) != 0) {
        return 1;
    }
    length = strlen(partial);
    if (length <= strlen(GS_PARTIAL_SUFFIX) ||
        strcmp(partial + length - strlen(GS_PARTIAL_SUFFIX),
               GS_PARTIAL_SUFFIX) != 0 ||
        length - strlen(GS_PARTIAL_SUFFIX) >= capacity) {
        return 1;
    }
    memcpy(final, partial, length - strlen(GS_PARTIAL_SUFFIX));
    final[length - strlen(GS_PARTIAL_SUFFIX)] = '\0';
    return strcmp(gs_basename(final), final_name) == 0 &&
           gs_path_state(partial, NULL, NULL) == 0 &&
           gs_path_state(final, NULL, NULL) == 0 ? 0 : 1;
}

static int gsp_write_summary(
    const char *directory,
    size_t k,
    const char *code_commit,
    const GSPPCAProfile *profile,
    const GSPKMeansProfile *kmeans,
    const GSBaselineClusteringMetrics *train,
    const GSBaselineClusteringMetrics *validation
)
{
    FILE *file = NULL;
    size_t largest, minimum, maximum, rank;
    uint64_t largest_size;
    double minimum_purity, maximum_purity;
    unsigned char selected[GS_LABEL_COUNT] = {0};
    int failed = gs_open_output(directory, "summary.txt", &file);
    if (failed || profile == NULL || kmeans == NULL) return 1;
    largest = gs_largest_cluster(train);
    largest_size = train->cluster_totals[largest];
    gs_purity_extremes(train, &minimum, &minimum_purity,
                       &maximum, &maximum_purity);
    if (fprintf(
            file,
            "format_version=1\nprofile=%s\n"
            "input_profile=%s\n"
            "evaluation_profile=%s_label_evaluation\n"
            "kmeans_profile=%s_kmeans\n"
            "pca_profile=%s\npca_threshold=%s\n"
            "retained_component_count=%lu\noriginal_feature_count=%lu\n"
            "evaluation=labels_after_clustering\n"
            "labels_policy=posterior_evaluation_only\n"
            "label_count=22\nclass_count=22\ncluster_count=%lu\n"
            "event_count=8583\nk=%lu\nseed=123\ntrain_rows=6866\n"
            "validation_rows=1717\nlabels_used_for_training=false\n"
            "labels_used_for_pca=false\nlabels_used_for_kmeans=false\n"
            "labels_used_for_clustering=false\n"
            "labels_used_for_evaluation=true\npca=true\n"
            "pca_whitening=false\npost_pca_standardization=false\n"
            "augmentation=false\n"
            "metric_inputs=labels_and_frozen_assignments_only\n"
            "centroids_used_for_metrics=false\n"
            "pca_scores_used_for_metrics=false\n"
            "nmi_normalization=arithmetic\n"
            "purity_definition=weighted_cluster_majority\n"
            "input_pca_code_commit=%s\n"
            "input_kmeans_code_commit=%s\ncode_commit=%s\n"
            "train_purity=%.17g\ntrain_nmi_arithmetic=%.17g\n"
            "train_ari=%.17g\ntrain_homogeneity=%.17g\n"
            "train_completeness=%.17g\nvalidation_purity=%.17g\n"
            "validation_nmi_arithmetic=%.17g\nvalidation_ari=%.17g\n"
            "validation_homogeneity=%.17g\nvalidation_completeness=%.17g\n"
            "largest_train_cluster=%lu\nlargest_train_cluster_size=%" PRIu64
            "\nminimum_purity_train_cluster=%lu\n"
            "minimum_train_cluster_purity=%.17g\n"
            "maximum_purity_train_cluster=%lu\n"
            "maximum_train_cluster_purity=%.17g\n",
            profile->profile, profile->profile, profile->profile,
            profile->profile, profile->token, profile->threshold_text,
            (unsigned long)profile->retained_components,
            (unsigned long)profile->original_features,
            (unsigned long)k, (unsigned long)k,
            profile->code_commit, kmeans->code_commit, code_commit,
            train->purity, train->normalized_mutual_information,
            train->adjusted_rand_index, train->homogeneity,
            train->completeness, validation->purity,
            validation->normalized_mutual_information,
            validation->adjusted_rand_index, validation->homogeneity,
            validation->completeness, (unsigned long)largest, largest_size,
            (unsigned long)minimum, minimum_purity,
            (unsigned long)maximum, maximum_purity
        ) < 0) {
        failed = 1;
    }
    for (rank = 0; !failed && rank < 5u; rank++) {
        size_t label, best = SIZE_MAX;
        uint64_t count = 0u;
        for (label = 0; label < GS_LABEL_COUNT; label++) {
            uint64_t current =
                train->contingency[label * train->cluster_count + largest];
            if (!selected[label] && (best == SIZE_MAX || current > count)) {
                best = label;
                count = current;
            }
        }
        if (best == SIZE_MAX) {
            failed = 1;
            break;
        }
        selected[best] = 1u;
        if (fprintf(
                file,
                "largest_train_cluster_top_%lu_label_index=%lu\n"
                "largest_train_cluster_top_%lu_label=%s\n"
                "largest_train_cluster_top_%lu_count=%" PRIu64 "\n"
                "largest_train_cluster_top_%lu_fraction=%.17g\n",
                (unsigned long)(rank + 1u), (unsigned long)best,
                (unsigned long)(rank + 1u), gs_labels[best],
                (unsigned long)(rank + 1u), count,
                (unsigned long)(rank + 1u),
                largest_size == 0u ? 0.0 :
                    (double)count / (double)largest_size
            ) < 0) {
            failed = 1;
        }
    }
    if (!failed &&
        (gs_write_requested_label(file, "Blip", 3u, train, validation) ||
         gs_write_requested_label(file, "Koi_Fish", 7u, train, validation) ||
         gs_write_requested_label(file, "Scattered_Light", 16u,
                                  train, validation) ||
         gs_write_requested_label(file, "None_of_the_Above", 12u,
                                  train, validation))) {
        failed = 1;
    }
    return gs_finish_output(file, failed);
}

static int gsp_write_nine(
    const char *directory,
    size_t k,
    const char *code_commit,
    const GSPPCAProfile *profile,
    const GSPKMeansProfile *kmeans,
    const GSBaselineClusteringMetrics *train,
    const GSBaselineClusteringMetrics *validation
)
{
    return gs_write_label_map(directory) ||
        gs_write_contingency(directory, "train_contingency_counts.csv", train) ||
        gs_write_contingency(directory, "val_contingency_counts.csv", validation) ||
        gs_write_cluster_summary(directory, "train_cluster_summary.csv", train) ||
        gs_write_cluster_summary(directory, "val_cluster_summary.csv", validation) ||
        gs_write_label_summary(directory, "train_label_summary.csv", train) ||
        gs_write_label_summary(directory, "val_label_summary.csv", validation) ||
        gs_write_metrics(directory, k, train, validation) ||
        gsp_write_summary(
            directory, k, code_commit, profile, kmeans, train, validation
        );
}

static int gsp_build_evaluation_entries(
    const char *manifest_path,
    const char *pca_directory,
    const char *kmeans_directory,
    const char *output,
    const GSPPCAProfile *profile,
    const GSPKMeansProfile *kmeans,
    GSChecksumEntry entries[GSP_EVALUATION_ENTRY_COUNT],
    char paths[17][GS_PATH_SIZE]
)
{
    static const char *const input_names[GSP_EVALUATION_INPUT_COUNT] = {
        "manifest_events_272.csv", "train_rows.csv", "val_rows.csv",
        "train_assignments_u32.bin", "val_assignments_u32.bin",
        "pca_spec.txt", "source_pca_sha256.txt",
        "source_kmeans_summary.txt", "source_kmeans_sha256.txt"
    };
    size_t index;
    if (profile == NULL || kmeans == NULL ||
        strcmp(gs_basename(manifest_path), input_names[0]) != 0) {
        return 1;
    }
    entries[0].kind = "INPUT";
    entries[0].name = input_names[0];
    entries[0].path = manifest_path;
    if (strlen(profile->train_rows_path) >= GS_PATH_SIZE ||
        strlen(profile->validation_rows_path) >= GS_PATH_SIZE ||
        strlen(kmeans->train_assignments_path) >= GS_PATH_SIZE ||
        strlen(kmeans->validation_assignments_path) >= GS_PATH_SIZE) {
        return 1;
    }
    strcpy(paths[0], profile->train_rows_path);
    strcpy(paths[1], profile->validation_rows_path);
    strcpy(paths[2], kmeans->train_assignments_path);
    strcpy(paths[3], kmeans->validation_assignments_path);
    if (gs_join(pca_directory, "pca_spec.txt", paths[4], GS_PATH_SIZE) != 0 ||
        gs_join(pca_directory, "sha256.txt", paths[5], GS_PATH_SIZE) != 0 ||
        gs_join(kmeans_directory, "summary.txt", paths[6], GS_PATH_SIZE) != 0 ||
        gs_join(kmeans_directory, "sha256.txt", paths[7], GS_PATH_SIZE) != 0) {
        return 1;
    }
    for (index = 1u; index < GSP_EVALUATION_INPUT_COUNT; index++) {
        entries[index].kind = "INPUT";
        entries[index].name = input_names[index];
        entries[index].path = paths[index - 1u];
    }
    for (index = 0; index < GSP_EVALUATION_ARTIFACT_COUNT; index++) {
        if (gs_join(output, gs_historical_artifacts[index],
                    paths[index + 8u], GS_PATH_SIZE) != 0) {
            return 1;
        }
        entries[index + GSP_EVALUATION_INPUT_COUNT].kind = "ARTIFACT";
        entries[index + GSP_EVALUATION_INPUT_COUNT].name =
            gs_historical_artifacts[index];
        entries[index + GSP_EVALUATION_INPUT_COUNT].path = paths[index + 8u];
    }
    return 0;
}

static int gsp_write_sha256(
    const char *manifest_path,
    const char *pca_directory,
    const char *kmeans_directory,
    const char *output,
    const GSPPCAProfile *profile,
    const GSPKMeansProfile *kmeans
)
{
    GSChecksumEntry entries[GSP_EVALUATION_ENTRY_COUNT];
    char paths[17][GS_PATH_SIZE];
    FILE *file = NULL;
    size_t index;
    int failed = 0;
    if (gsp_build_evaluation_entries(
            manifest_path, pca_directory, kmeans_directory, output,
            profile, kmeans, entries, paths
        ) != 0 ||
        gs_open_output(output, "sha256.txt", &file) != 0) {
        return 1;
    }
    for (index = 0; !failed && index < GSP_EVALUATION_ENTRY_COUNT; index++) {
        char hash[GS_SHA_TEXT_SIZE];
        if (gs_sha_file(entries[index].path, hash) != 0 ||
            fprintf(file, "%s %s %s\n", entries[index].kind,
                    hash, entries[index].name) < 0) {
            failed = 1;
        }
    }
    return gs_finish_output(file, failed);
}

static int gsp_validate_output(
    const char *manifest_path,
    const char *pca_directory,
    const char *kmeans_directory,
    const char *output,
    const GSPPCAProfile *profile,
    const GSPKMeansProfile *kmeans
)
{
    GSChecksumEntry entries[GSP_EVALUATION_ENTRY_COUNT];
    char paths[17][GS_PATH_SIZE];
    char sha_path[GS_PATH_SIZE];
    size_t index, size;
    if (gs_exact_files(output, gs_output_files, 10u) != 0) return 1;
    for (index = 0; index < 10u; index++) {
        char path[GS_PATH_SIZE];
        if (gs_join(output, gs_output_files[index], path, sizeof(path)) != 0 ||
            gs_file_size(path, &size) != 0 || size == 0u) {
            return 1;
        }
    }
    return gsp_build_evaluation_entries(
               manifest_path, pca_directory, kmeans_directory, output,
               profile, kmeans, entries, paths
           ) != 0 ||
           gs_join(output, "sha256.txt", sha_path, sizeof(sha_path)) != 0 ||
           gs_verify_checksum_manifest(
               sha_path, entries, GSP_EVALUATION_ENTRY_COUNT
           ) != 0;
}

static int gsp_failure_hook(void)
{
    const char *value = getenv(GS_FAILURE_ENV);
    return value != NULL && strcmp(value, "1") == 0;
}

static void gsp_print_usage(const char *program)
{
    printf(
        "Usage: %s <manifest> <pca_directory> <pca_profile> "
        "<kmeans_directory> <k> <output_partial_directory> <code_commit>\n"
        "pca_profile: pca90 | pca95 | pca99\n",
        program
    );
}

int main(int argc, char **argv)
{
    GSEventManifest manifest;
    GSPPCAProfile profile;
    GSPKMeansProfile kmeans;
    GSBaselineClusteringMetrics train_metrics;
    GSBaselineClusteringMetrics validation_metrics;
    unsigned char *coverage = NULL;
    uint32_t *train_labels = NULL;
    uint32_t *validation_labels = NULL;
    uint32_t *train_assignments = NULL;
    uint32_t *validation_assignments = NULL;
    size_t train_bytes, validation_bytes, k, index;
    char final_path[GS_PATH_SIZE];
    char train_before[GS_SHA_TEXT_SIZE], train_after[GS_SHA_TEXT_SIZE];
    char validation_before[GS_SHA_TEXT_SIZE];
    char validation_after[GS_SHA_TEXT_SIZE];
    int partial_created = 0;
    int rc = 1;

    memset(&manifest, 0, sizeof(manifest));
    memset(&profile, 0, sizeof(profile));
    memset(&kmeans, 0, sizeof(kmeans));
    memset(&train_metrics, 0, sizeof(train_metrics));
    memset(&validation_metrics, 0, sizeof(validation_metrics));

    if (argc == 2 && strcmp(argv[1], "--help") == 0) {
        gsp_print_usage(argv[0]);
        return 0;
    }
    if (argc != 8 || !gs_platform_valid() ||
        setlocale(LC_NUMERIC, "C") == NULL ||
        gs_parse_size(argv[5], &k) != 0 || (k != 22u && k != 36u) ||
        !gsp_commit_valid(argv[7]) ||
        gs_checked_product(GS_TRAIN_ROWS, sizeof(uint32_t), &train_bytes) ||
        gs_checked_product(GS_VALIDATION_ROWS, sizeof(uint32_t),
                           &validation_bytes)) {
        fprintf(stderr, "Errore: argomenti o piattaforma non validi.\n");
        return 1;
    }
    if (gsp_validate_pca(argv[1], argv[2], argv[3], &profile) != 0) {
        fprintf(stderr, "Errore: contratto PCA non valido.\n");
        return 1;
    }
    if (gsp_validate_kmeans(argv[2], argv[4], k, &profile, &kmeans) != 0) {
        fprintf(stderr, "Errore: contratto k-means non valido.\n");
        return 1;
    }
    if (gsp_output_paths(argv[2], argv[6], k, final_path,
                         sizeof(final_path)) != 0) {
        fprintf(stderr, "Errore: percorso output non valido.\n");
        return 1;
    }
    printf("[PASS] pca_and_kmeans_contract\n");

    coverage = (unsigned char *)calloc(GS_EVENT_COUNT, 1u);
    train_labels = (uint32_t *)malloc(train_bytes);
    validation_labels = (uint32_t *)malloc(validation_bytes);
    train_assignments = (uint32_t *)malloc(train_bytes);
    validation_assignments = (uint32_t *)malloc(validation_bytes);
    if (coverage == NULL || train_labels == NULL || validation_labels == NULL ||
        train_assignments == NULL || validation_assignments == NULL ||
        gs_event_manifest_load(argv[1], &manifest) != 0 ||
        gs_validate_manifest(&manifest) != 0 ||
        gs_load_rows(profile.train_rows_path, "train", GS_TRAIN_ROWS,
                     &manifest, coverage, train_labels) != 0 ||
        gs_load_rows(profile.validation_rows_path, "validation",
                     GS_VALIDATION_ROWS, &manifest, coverage,
                     validation_labels) != 0) {
        fprintf(stderr, "Errore: manifest, label o split non validi.\n");
        goto cleanup;
    }
    for (index = 0; index < GS_EVENT_COUNT; index++) {
        if (coverage[index] != 1u) {
            fprintf(stderr, "Errore: copertura eventi non valida.\n");
            goto cleanup;
        }
    }
    if (gs_read_assignments(kmeans.train_assignments_path, train_assignments,
                            GS_TRAIN_ROWS, k) != 0 ||
        gs_read_assignments(kmeans.validation_assignments_path,
                            validation_assignments,
                            GS_VALIDATION_ROWS, k) != 0 ||
        gs_sha_file(kmeans.train_assignments_path, train_before) != 0 ||
        gs_sha_file(kmeans.validation_assignments_path,
                    validation_before) != 0) {
        fprintf(stderr, "Errore: assignment congelati non validi.\n");
        goto cleanup;
    }
    printf("[PASS] posterior_labels_and_frozen_assignments\n");

    if (gs_baseline_clustering_metrics_compute(
            train_labels, train_assignments, GS_TRAIN_ROWS,
            GS_LABEL_COUNT, k, &train_metrics
        ) != GS_BASELINE_CLUSTERING_METRICS_SUCCESS ||
        gs_baseline_clustering_metrics_compute(
            validation_labels, validation_assignments, GS_VALIDATION_ROWS,
            GS_LABEL_COUNT, k, &validation_metrics
        ) != GS_BASELINE_CLUSTERING_METRICS_SUCCESS ||
        !gs_metrics_valid(&train_metrics, GS_TRAIN_ROWS, k) ||
        !gs_metrics_valid(&validation_metrics, GS_VALIDATION_ROWS, k)) {
        fprintf(stderr, "Errore: metriche non valide.\n");
        goto cleanup;
    }
    printf("[PASS] posterior_train_and_validation_metrics\n");

    if (GS_MKDIR(argv[6]) != 0) {
        fprintf(stderr, "Errore: creazione partial fallita.\n");
        goto cleanup;
    }
    partial_created = 1;
    if (gsp_failure_hook()) {
        fprintf(stderr, "Errore: failure hook PCA evaluation attivato.\n");
        goto cleanup;
    }
    if (gsp_write_nine(
            argv[6], k, argv[7], &profile, &kmeans,
            &train_metrics, &validation_metrics
        ) != 0 ||
        gsp_write_sha256(
            argv[1], argv[2], argv[4], argv[6], &profile, &kmeans
        ) != 0 ||
        gsp_validate_output(
            argv[1], argv[2], argv[4], argv[6], &profile, &kmeans
        ) != 0 ||
        gs_sha_file(kmeans.train_assignments_path, train_after) != 0 ||
        gs_sha_file(kmeans.validation_assignments_path,
                    validation_after) != 0 ||
        strcmp(train_before, train_after) != 0 ||
        strcmp(validation_before, validation_after) != 0 ||
        gsp_validate_pca(argv[1], argv[2], argv[3], &profile) != 0 ||
        gsp_validate_kmeans(argv[2], argv[4], k, &profile, &kmeans) != 0) {
        fprintf(stderr, "Errore: scrittura o validazione output fallita.\n");
        goto cleanup;
    }
    printf("[PASS] exact_inventory_checksums_and_frozen_inputs\n");
    if (rename(argv[6], final_path) != 0) {
        fprintf(stderr, "Errore: promozione atomica fallita.\n");
        goto cleanup;
    }
    partial_created = 0;
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
    printf("MINI112 PCA LABEL EVALUATION COMPLETED\n");
    rc = 0;

cleanup:
    if (partial_created && gs_cleanup_partial(argv[6]) != 0) {
        fprintf(stderr, "Errore: cleanup partial incompleto.\n");
        rc = 1;
    }
    gs_baseline_clustering_metrics_free(&validation_metrics);
    gs_baseline_clustering_metrics_free(&train_metrics);
    gs_event_manifest_free(&manifest);
    free(validation_assignments);
    free(train_assignments);
    free(validation_labels);
    free(train_labels);
    free(coverage);
    return rc;
}
