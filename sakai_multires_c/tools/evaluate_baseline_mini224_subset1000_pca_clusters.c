#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#define main gs_legacy_mini224_subset1000_evaluator_main
#include "evaluate_baseline_mini224_subset1000_clusters.c"
#undef main

#undef GS_FAILURE_ENV
#define GS_FAILURE_ENV \
    "GS_MINI224_SUBSET1000_PCA_EVALUATION_TEST_FAIL_AFTER_PARTIAL"

enum {
    GSP224_PCA_INPUTS = 7,
    GSP224_PCA_ARTIFACTS = 11,
    GSP224_PCA_FILES = 12,
    GSP224_KMEANS_INPUTS = 5,
    GSP224_KMEANS_ARTIFACTS = 12,
    GSP224_KMEANS_FILES = 13,
    GSP224_EVALUATION_INPUTS = 9,
    GSP224_EVALUATION_ARTIFACTS = 9,
    GSP224_EVALUATION_ENTRIES = 18
};

#define GSP224_BASE_CODE_COMMIT \
    "d87ca996e7922d6873a54d952aec27cdb99a5998"
#define GSP224_PCA_CODE_COMMIT \
    "2cd2dc7f53d8a97fd1c69f88cfa416416818599c"
#define GSP224_KMEANS_CODE_COMMIT \
    "509bc14a1a0e3c9e71dd053b7f9205081df320ef"

typedef struct {
    char token[8];
    char profile[64];
    char threshold[8];
    size_t expected_retained;
    size_t retained;
    size_t train_rows;
    size_t validation_rows;
    size_t original_features;
    size_t numerical_rank;
    size_t nullspace_dimension;
    char code_commit[65];
    char score_train[64];
    char score_validation[64];
} GSP224PCA;

typedef struct {
    size_t k;
    char code_commit[65];
    char train_assignments[GS_PATH_SIZE];
    char validation_assignments[GS_PATH_SIZE];
} GSP224KMeans;

typedef struct {
    char repository[GS_PATH_SIZE];
    char manifest[GS_PATH_SIZE];
    char base[GS_PATH_SIZE];
    char pca[GS_PATH_SIZE];
    char kmeans[GS_PATH_SIZE];
} GSP224Layout;

static const char *const gsp224_pca_fixed[10] = {
    "components_f64.bin", "cumulative_explained_variance_ratio_f64.bin",
    "eigenvalues_f64.bin", "explained_variance_ratio_f64.bin",
    "mean_f64.bin", "pca_spec.txt", "sha256.txt", "spectrum.csv",
    "summary.txt", "thresholds.csv"
};

static const char *const gsp224_kmeans_files[GSP224_KMEANS_FILES] = {
    "centroids_f64.bin", "train_assignments_u32.bin",
    "val_assignments_u32.bin", "train_squared_distances_f64.bin",
    "val_squared_distances_f64.bin", "cluster_sizes_train_u64.bin",
    "initial_centroid_indices_u64.bin", "train_assignments.csv",
    "val_assignments.csv", "cluster_sizes.csv", "initial_centroids.csv",
    "summary.txt", "sha256.txt"
};

static int gsp224_commit_valid(const char *value)
{
    size_t i;
    if (value != NULL &&
        strcmp(value, "UNCOMMITTED_STEP10M_TEST") == 0) return 1;
    if (value == NULL || strlen(value) != 40u) return 0;
    for (i = 0; i < 40u; i++) {
        if (!((value[i] >= '0' && value[i] <= '9') ||
              (value[i] >= 'a' && value[i] <= 'f'))) return 0;
    }
    return 1;
}

static void gsp224_normalize(char *path)
{
    size_t i;
    for (i = 0; path != NULL && path[i] != '\0'; i++) {
        if (path[i] == '\\') path[i] = '/';
#ifdef _WIN32
        if (path[i] >= 'A' && path[i] <= 'Z') {
            path[i] = (char)(path[i] - 'A' + 'a');
        }
#endif
    }
}

static int gsp224_reparse_free(const char *path)
{
#ifdef _WIN32
    char full[GS_PATH_SIZE];
    char probe[GS_PATH_SIZE];
    size_t i, length;
    DWORD attributes;
    if (path == NULL || _fullpath(full, path, sizeof(full)) == NULL) return 0;
    length = strlen(full);
    if (length >= sizeof(probe)) return 0;
    for (i = 0; i <= length; i++) {
        if (i == length || full[i] == '/' || full[i] == '\\') {
            if (i <= 2u) continue;
            memcpy(probe, full, i);
            probe[i] = '\0';
            attributes = GetFileAttributesA(probe);
            if (attributes == INVALID_FILE_ATTRIBUTES ||
                (attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0u) return 0;
        }
    }
#else
    (void)path;
#endif
    return 1;
}

static int gsp224_same_existing(
    const char *actual,
    const char *expected
)
{
    char left[GS_PATH_SIZE], right[GS_PATH_SIZE];
    if (!gsp224_reparse_free(actual) || !gsp224_reparse_free(expected) ||
        gs_canonical_existing(actual, left, sizeof(left)) != 0 ||
        gs_canonical_existing(expected, right, sizeof(right)) != 0) return 0;
    gsp224_normalize(left);
    gsp224_normalize(right);
    return strcmp(left, right) == 0;
}

static int gsp224_token(
    const char *token,
    GSP224PCA *profile
)
{
    const char *suffix;
    if (token == NULL || profile == NULL) return 1;
    memset(profile, 0, sizeof(*profile));
    if (strcmp(token, "pca90") == 0) {
        suffix = "pca90"; strcpy(profile->threshold, "0.90");
        profile->expected_retained = 15u;
    } else if (strcmp(token, "pca95") == 0) {
        suffix = "pca95"; strcpy(profile->threshold, "0.95");
        profile->expected_retained = 38u;
    } else if (strcmp(token, "pca99") == 0) {
        suffix = "pca99"; strcpy(profile->threshold, "0.99");
        profile->expected_retained = 168u;
    } else {
        return 1;
    }
    if (snprintf(profile->profile, sizeof(profile->profile),
                 "mini224_subset1000_pool14_%s", suffix) < 0 ||
        snprintf(profile->score_train, sizeof(profile->score_train),
                 "train_scores_%s_f32.bin", suffix) < 0 ||
        snprintf(profile->score_validation,
                 sizeof(profile->score_validation),
                 "val_scores_%s_f32.bin", suffix) < 0) return 1;
    strcpy(profile->token, token);
    return 0;
}

static int gsp224_resolve(
    const char *manifest_argument,
    const char *pca_argument,
    const char *kmeans_argument,
    size_t k,
    const GSP224PCA *profile,
    GSP224Layout *layout
)
{
    char baseline[GS_PATH_SIZE], results[GS_PATH_SIZE];
    char sakai[GS_PATH_SIZE], expected[GS_PATH_SIZE], name[32];
    int written;
    if (layout == NULL || profile == NULL ||
        gs_canonical_existing(pca_argument, layout->pca,
                              sizeof(layout->pca)) != 0 ||
        strcmp(gs_basename(layout->pca), profile->profile) != 0 ||
        gs_parent_path(layout->pca, baseline, sizeof(baseline)) != 0 ||
        strcmp(gs_basename(baseline), "baseline") != 0 ||
        gs_parent_path(baseline, results, sizeof(results)) != 0 ||
        strcmp(gs_basename(results), "results") != 0 ||
        gs_parent_path(results, sakai, sizeof(sakai)) != 0 ||
        strcmp(gs_basename(sakai), "sakai_multires_c") != 0 ||
        gs_parent_path(sakai, layout->repository,
                       sizeof(layout->repository)) != 0 ||
        gs_join(layout->repository, "manifest_events_272.csv",
                layout->manifest, sizeof(layout->manifest)) != 0 ||
        gs_join(baseline, "mini224_subset1000_pool14_no_pca",
                layout->base, sizeof(layout->base)) != 0 ||
        (written = snprintf(name, sizeof(name), "kmeans_seed123/k%lu",
                            (unsigned long)k)) < 0 ||
        (size_t)written >= sizeof(name) ||
        gs_join(layout->pca, name, layout->kmeans,
                sizeof(layout->kmeans)) != 0 ||
        !gsp224_same_existing(manifest_argument, layout->manifest) ||
        !gsp224_same_existing(kmeans_argument, layout->kmeans) ||
        !gsp224_reparse_free(layout->repository) ||
        !gsp224_reparse_free(layout->base) ||
        !gsp224_reparse_free(layout->pca)) return 1;
    if (gs_join(layout->repository,
                "sakai_multires_c/results/baseline", expected,
                sizeof(expected)) != 0 ||
        !gsp224_same_existing(baseline, expected)) return 1;
    return 0;
}

static int gsp224_read_value(
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

static int gsp224_exact_regular_files(
    const char *directory,
    const char *const *names,
    size_t count
)
{
    unsigned char *seen = NULL;
    DIR *stream = NULL;
    struct dirent *entry;
    size_t i;
    int failed = 0;
    seen = (unsigned char *)calloc(count, 1u);
    if (seen == NULL || (stream = opendir(directory)) == NULL) {
        free(seen);
        return 1;
    }
    while (!failed) {
        char path[GS_PATH_SIZE];
        int regular, is_directory;
        size_t index;
        errno = 0;
        entry = readdir(stream);
        if (entry == NULL) {
            if (errno != 0) failed = 1;
            break;
        }
        if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, ".."))
            continue;
        if (gs_join(directory, entry->d_name, path, sizeof(path)) ||
            gs_path_state(path, &regular, &is_directory) != 1) {
            failed = 1;
        } else if (is_directory) {
            if (strcmp(entry->d_name, "kmeans_seed123") &&
                strcmp(entry->d_name, "label_evaluation_seed123")) failed = 1;
        } else if (!regular ||
                   !gs_name_index(entry->d_name, names, count, &index) ||
                   seen[index]) {
            failed = 1;
        } else {
            seen[index] = 1u;
        }
    }
    if (closedir(stream) != 0) failed = 1;
    for (i = 0; !failed && i < count; i++) if (!seen[i]) failed = 1;
    free(seen);
    return failed;
}

static int gsp224_expect(
    const char *directory,
    const char *filename,
    const char *key,
    const char *expected
)
{
    char value[256];
    return gsp224_read_value(directory, filename, key, value,
                             sizeof(value)) != 0 ||
        strcmp(value, expected) != 0;
}

static int gsp224_base_entries(
    const GSP224Layout *layout,
    GSChecksumEntry entries[13],
    char paths[12][GS_PATH_SIZE]
)
{
    static const char *const input_names[4] = {
        "manifest_events_272.csv", "manifest_tensors_224_central.csv",
        "subset224_train_800_seed123.csv", "subset224_val_200_seed123.csv"
    };
    static const char *const relatives[4] = {
        "manifest_events_272.csv", "manifest_tensors_224_central.csv",
        "sakai_multires_c/configs/subset224_train_800_seed123.csv",
        "sakai_multires_c/configs/subset224_val_200_seed123.csv"
    };
    static const char *const artifacts[9] = {
        "feature_spec.txt", "train_features_raw_f32.bin",
        "val_features_raw_f32.bin", "train_features_std_f32.bin",
        "val_features_std_f32.bin", "scaler.csv", "train_rows.csv",
        "val_rows.csv", "summary.txt"
    };
    size_t i;
    for (i = 0; i < 4u; i++) {
        if (gs_join(layout->repository, relatives[i], paths[i],
                    GS_PATH_SIZE) != 0) return 1;
        entries[i].kind = "INPUT";
        entries[i].name = input_names[i];
        entries[i].path = paths[i];
    }
    for (i = 0; i < 9u; i++) {
        if (gs_join(layout->base, artifacts[i], paths[i + 4u],
                    GS_PATH_SIZE) != 0) return 1;
        entries[i + 4u].kind = "ARTIFACT";
        entries[i + 4u].name = artifacts[i];
        entries[i + 4u].path = paths[i + 4u];
    }
    return 0;
}

static int gsp224_validate_base(const GSP224Layout *layout)
{
    GSChecksumEntry entries[13];
    char paths[12][GS_PATH_SIZE], sha[GS_PATH_SIZE];
    size_t i, actual;
    if (gs_base_inventory(layout->base) != 0 ||
        gs_validate_base_metadata(layout->base) != 0 ||
        gsp224_base_entries(layout, entries, paths) != 0 ||
        gs_join(layout->base, "sha256.txt", sha, sizeof(sha)) != 0 ||
        gs_verify_checksum_manifest(sha, entries, 13u) != 0) return 1;
    for (i = 0; i < 10u; i++) {
        if (gs_join(layout->base, gs_base_files[i], paths[0],
                    GS_PATH_SIZE) != 0 ||
            gs_file_size(paths[0], &actual) != 0 ||
            actual != gs_base_sizes[i]) return 1;
    }
    return 0;
}

static int gsp224_pca_entries(
    const GSP224Layout *layout,
    const GSP224PCA *profile,
    GSChecksumEntry entries[18],
    char paths[18][GS_PATH_SIZE]
)
{
    static const char *const input_names[7] = {
        "feature_spec.txt", "scaler.csv", "train_features_std_f32.bin",
        "val_features_std_f32.bin", "train_rows.csv", "val_rows.csv",
        "source_no_pca_sha256.txt"
    };
    static const char *const input_sources[7] = {
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
    size_t i;
    for (i = 0; i < 7u; i++) {
        if (gs_join(layout->base, input_sources[i], paths[i],
                    GS_PATH_SIZE) != 0) return 1;
        entries[i].kind = "INPUT"; entries[i].name = input_names[i];
        entries[i].path = paths[i];
    }
    for (i = 0; i < 8u; i++) {
        if (gs_join(layout->pca, artifact_fixed[i], paths[i + 7u],
                    GS_PATH_SIZE) != 0) return 1;
        entries[i + 7u].kind = "ARTIFACT";
        entries[i + 7u].name = artifact_fixed[i];
        entries[i + 7u].path = paths[i + 7u];
    }
    if (gs_join(layout->pca, profile->score_train, paths[15],
                GS_PATH_SIZE) != 0 ||
        gs_join(layout->pca, profile->score_validation, paths[16],
                GS_PATH_SIZE) != 0 ||
        gs_join(layout->pca, "summary.txt", paths[17],
                GS_PATH_SIZE) != 0) return 1;
    entries[15].kind = "ARTIFACT"; entries[15].name = profile->score_train;
    entries[15].path = paths[15];
    entries[16].kind = "ARTIFACT";
    entries[16].name = profile->score_validation;
    entries[16].path = paths[16];
    entries[17].kind = "ARTIFACT"; entries[17].name = "summary.txt";
    entries[17].path = paths[17];
    return 0;
}

static int gsp224_validate_pca(
    const GSP224Layout *layout,
    GSP224PCA *profile
)
{
    GSChecksumEntry entries[18];
    char paths[18][GS_PATH_SIZE], sha[GS_PATH_SIZE], value[256];
    const char *files[12];
    size_t i;
    for (i = 0; i < 10u; i++) files[i] = gsp224_pca_fixed[i];
    files[10] = profile->score_train;
    files[11] = profile->score_validation;
    if (gsp224_exact_regular_files(layout->pca, files, 12u) != 0 ||
        gsp224_pca_entries(layout, profile, entries, paths) != 0 ||
        gs_join(layout->pca, "sha256.txt", sha, sizeof(sha)) != 0 ||
        gs_verify_checksum_manifest(sha, entries, 18u) != 0 ||
        gsp224_expect(layout->pca, "pca_spec.txt", "profile",
                      profile->profile) ||
        gsp224_expect(layout->pca, "pca_spec.txt", "source_profile",
                      "mini224_subset1000_pool14_no_pca") ||
        gsp224_expect(layout->pca, "pca_spec.txt", "selected_threshold",
                      profile->threshold) ||
        gsp224_expect(layout->pca, "pca_spec.txt", "numerical_regime",
                      "rank_deficient") ||
        gsp224_expect(layout->pca, "pca_spec.txt", "nullspace_validation",
                      "orthogonal_projector") ||
        gsp224_expect(layout->pca, "pca_spec.txt",
                      "retained_prefix_validation", "bitwise") ||
        gsp224_expect(layout->pca, "pca_spec.txt", "full_model_serialized",
                      "true") ||
        gsp224_expect(layout->pca, "pca_spec.txt", "fit_input",
                      "train_only") ||
        gsp224_expect(layout->pca, "pca_spec.txt",
                      "validation_transform_without_refit", "true") ||
        gsp224_expect(layout->pca, "pca_spec.txt", "whitening", "false") ||
        gsp224_expect(layout->pca, "pca_spec.txt", "labels_used", "false") ||
        gsp224_expect(layout->pca, "summary.txt", "code_commit",
                      GSP224_PCA_CODE_COMMIT)) return 1;
#define GSP224_PARSE_PCA(key, member) \
    if (gsp224_read_value(layout->pca, "pca_spec.txt", key, value, \
                          sizeof(value)) != 0 || \
        gs_parse_size(value, &profile->member) != 0) return 1
    GSP224_PARSE_PCA("retained_component_count", retained);
    GSP224_PARSE_PCA("train_rows", train_rows);
    GSP224_PARSE_PCA("validation_rows", validation_rows);
    GSP224_PARSE_PCA("original_feature_count", original_features);
    GSP224_PARSE_PCA("numerical_rank", numerical_rank);
    GSP224_PARSE_PCA("nullspace_dimension", nullspace_dimension);
#undef GSP224_PARSE_PCA
    if (profile->retained != profile->expected_retained ||
        profile->train_rows != 800u ||
        profile->validation_rows != 200u ||
        profile->original_features != 1024u ||
        profile->numerical_rank != 799u ||
        profile->nullspace_dimension != 225u) return 1;
    strcpy(profile->code_commit, GSP224_PCA_CODE_COMMIT);
    return 0;
}

static int gsp224_kmeans_entries(
    const GSP224Layout *layout,
    const GSP224PCA *profile,
    GSChecksumEntry entries[17],
    char paths[17][GS_PATH_SIZE]
)
{
    static const char *const input_names[5] = {
        "pca_spec.txt", "thresholds.csv", NULL, NULL,
        "source_pca_sha256.txt"
    };
    static const char *const artifacts[12] = {
        "centroids_f64.bin", "train_assignments_u32.bin",
        "val_assignments_u32.bin", "train_squared_distances_f64.bin",
        "val_squared_distances_f64.bin", "cluster_sizes_train_u64.bin",
        "initial_centroid_indices_u64.bin", "train_assignments.csv",
        "val_assignments.csv", "cluster_sizes.csv", "initial_centroids.csv",
        "summary.txt"
    };
    const char *source_names[5] = {
        "pca_spec.txt", "thresholds.csv", profile->score_train,
        profile->score_validation, "sha256.txt"
    };
    const char *names[5] = {
        input_names[0], input_names[1], profile->score_train,
        profile->score_validation, input_names[4]
    };
    size_t i;
    for (i = 0; i < 5u; i++) {
        if (gs_join(layout->pca, source_names[i], paths[i],
                    GS_PATH_SIZE) != 0) return 1;
        entries[i].kind = "INPUT"; entries[i].name = names[i];
        entries[i].path = paths[i];
    }
    for (i = 0; i < 12u; i++) {
        if (gs_join(layout->kmeans, artifacts[i], paths[i + 5u],
                    GS_PATH_SIZE) != 0) return 1;
        entries[i + 5u].kind = "ARTIFACT"; entries[i + 5u].name = artifacts[i];
        entries[i + 5u].path = paths[i + 5u];
    }
    return 0;
}

static int gsp224_validate_kmeans(
    const GSP224Layout *layout,
    const GSP224PCA *profile,
    size_t k,
    GSP224KMeans *metadata
)
{
    GSChecksumEntry entries[17];
    char paths[17][GS_PATH_SIZE], sha[GS_PATH_SIZE], expected[64], value[256];
    size_t parsed;
    if (metadata == NULL || gs_exact_files(layout->kmeans,
                                           gsp224_kmeans_files, 13u) != 0 ||
        gsp224_kmeans_entries(layout, profile, entries, paths) != 0 ||
        gs_join(layout->kmeans, "sha256.txt", sha, sizeof(sha)) != 0 ||
        gs_verify_checksum_manifest(sha, entries, 17u) != 0 ||
        snprintf(expected, sizeof(expected),
                 "mini224_subset1000_pool14_%s_kmeans", profile->token) < 0 ||
        gsp224_expect(layout->kmeans, "summary.txt", "profile", expected) ||
        gsp224_expect(layout->kmeans, "summary.txt", "input_profile",
                      profile->profile) ||
        gsp224_expect(layout->kmeans, "summary.txt", "pca_profile",
                      profile->token) ||
        gsp224_expect(layout->kmeans, "summary.txt", "pca_threshold",
                      profile->threshold) ||
        gsp224_expect(layout->kmeans, "summary.txt", "policy_c", "true") ||
        gsp224_expect(layout->kmeans, "summary.txt", "fit_input",
                      "train_only") ||
        gsp224_expect(layout->kmeans, "summary.txt",
                      "validation_assigned_after_fit", "true") ||
        gsp224_expect(layout->kmeans, "summary.txt", "labels_used", "false") ||
        gsp224_expect(layout->kmeans, "summary.txt", "pca_code_commit",
                      profile->code_commit) ||
        gsp224_expect(layout->kmeans, "summary.txt", "code_commit",
                      GSP224_KMEANS_CODE_COMMIT)) return 1;
#define GSP224_KM_SIZE(key, expected_value) \
    if (gsp224_read_value(layout->kmeans, "summary.txt", key, value, \
                          sizeof(value)) != 0 || \
        gs_parse_size(value, &parsed) != 0 || parsed != (expected_value)) return 1
    GSP224_KM_SIZE("retained_component_count", profile->retained);
    GSP224_KM_SIZE("original_feature_count", profile->original_features);
    GSP224_KM_SIZE("numerical_rank", profile->numerical_rank);
    GSP224_KM_SIZE("nullspace_dimension", profile->nullspace_dimension);
    GSP224_KM_SIZE("train_rows", profile->train_rows);
    GSP224_KM_SIZE("validation_rows", profile->validation_rows);
    GSP224_KM_SIZE("feature_count", profile->retained);
    GSP224_KM_SIZE("k", k);
    GSP224_KM_SIZE("seed", 123u);
#undef GSP224_KM_SIZE
    metadata->k = k;
    strcpy(metadata->code_commit, GSP224_KMEANS_CODE_COMMIT);
    strcpy(metadata->train_assignments, paths[6]);
    strcpy(metadata->validation_assignments, paths[7]);
    return 0;
}

static int gsp224_output_paths(
    const char *partial,
    const GSP224Layout *layout,
    size_t k,
    char *final,
    size_t capacity
)
{
    char parent[GS_PATH_SIZE], actual[GS_PATH_SIZE], expected[GS_PATH_SIZE];
    char name[32];
    size_t length, suffix = strlen(GS_PARTIAL_SUFFIX);
    int written;
    if (!gs_value_valid(partial) ||
        (written = snprintf(name, sizeof(name), "k%lu.partial",
                            (unsigned long)k)) < 0 ||
        (size_t)written >= sizeof(name) ||
        strcmp(gs_basename(partial), name) != 0 ||
        gs_parent_path(partial, parent, sizeof(parent)) != 0 ||
        strcmp(gs_basename(parent), "label_evaluation_seed123") != 0 ||
        gs_canonical_existing(parent, actual, sizeof(actual)) != 0 ||
        gs_join(layout->pca, "label_evaluation_seed123", expected,
                sizeof(expected)) != 0 ||
        !gsp224_same_existing(actual, expected) ||
        !gsp224_reparse_free(parent)) return 1;
    length = strlen(partial);
    if (length <= suffix || length - suffix >= capacity ||
        strcmp(partial + length - suffix, GS_PARTIAL_SUFFIX) != 0) return 1;
    memcpy(final, partial, length - suffix);
    final[length - suffix] = '\0';
    return gs_path_state(partial, NULL, NULL) == 0 &&
           gs_path_state(final, NULL, NULL) == 0 ? 0 : 1;
}

static int gsp224_write_summary(
    const char *directory,
    const GSP224PCA *profile,
    const GSP224KMeans *kmeans,
    const char *code_commit,
    const GSBaselineClusteringMetrics *train,
    const GSBaselineClusteringMetrics *validation
)
{
    FILE *file = NULL;
    int failed;
    if (gs_open_output(directory, "summary.txt", &file) != 0) return 1;
    failed = fprintf(
        file,
        "format_version=1\n"
        "profile=mini224_subset1000_pool14_%s_label_evaluation\n"
        "input_profile=%s\n"
        "subset_profile=subset224_seed123_prefix_800_200\n"
        "pca_profile=%s\npca_threshold=%s\n"
        "retained_component_count=%lu\noriginal_feature_count=%lu\n"
        "numerical_rank=%lu\nnullspace_dimension=%lu\npolicy_c=true\n"
        "kmeans_profile=mini224_subset1000_pool14_%s_kmeans\n"
        "evaluation=labels_after_clustering\n"
        "labels_policy=posterior_evaluation_only\n"
        "label_count=22\nclass_count=22\nevent_count=8583\n"
        "subset_rows=1000\ncluster_count=%lu\nk=%lu\nseed=123\n"
        "train_rows=%lu\nvalidation_rows=%lu\n"
        "labels_used_for_feature_extraction=false\n"
        "labels_used_for_standardization=false\n"
        "labels_used_for_pca=false\nlabels_used_for_kmeans=false\n"
        "labels_used_for_evaluation=true\npca=true\nwhitening=false\n"
        "post_pca_standardization=false\naugmentation=false\n"
        "nmi_normalization=arithmetic\n"
        "purity_definition=weighted_cluster_majority\n"
        "base_code_commit=%s\npca_code_commit=%s\n"
        "kmeans_code_commit=%s\ncode_commit=%s\n"
        "train_purity=%.17g\ntrain_nmi_arithmetic=%.17g\n"
        "train_ari=%.17g\ntrain_homogeneity=%.17g\n"
        "train_completeness=%.17g\nvalidation_purity=%.17g\n"
        "validation_nmi_arithmetic=%.17g\nvalidation_ari=%.17g\n"
        "validation_homogeneity=%.17g\nvalidation_completeness=%.17g\n",
        profile->token, profile->profile, profile->token, profile->threshold,
        (unsigned long)profile->retained,
        (unsigned long)profile->original_features,
        (unsigned long)profile->numerical_rank,
        (unsigned long)profile->nullspace_dimension, profile->token,
        (unsigned long)kmeans->k, (unsigned long)kmeans->k,
        (unsigned long)profile->train_rows,
        (unsigned long)profile->validation_rows, GSP224_BASE_CODE_COMMIT,
        profile->code_commit, kmeans->code_commit, code_commit,
        train->purity, train->normalized_mutual_information,
        train->adjusted_rand_index, train->homogeneity, train->completeness,
        validation->purity, validation->normalized_mutual_information,
        validation->adjusted_rand_index, validation->homogeneity,
        validation->completeness
    ) < 0;
    return gs_finish_output(file, failed);
}

static int gsp224_write_nine(
    const char *directory,
    const GSP224PCA *profile,
    const GSP224KMeans *kmeans,
    const char *code_commit,
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
        gs_write_metrics(directory, kmeans->k, train, validation) ||
        gsp224_write_summary(directory, profile, kmeans, code_commit,
                            train, validation);
}

static int gsp224_evaluation_entries(
    const GSP224Layout *layout,
    const char *output,
    GSChecksumEntry entries[18],
    char paths[17][GS_PATH_SIZE]
)
{
    static const char *const names[9] = {
        "manifest_events_272.csv", "train_rows.csv", "val_rows.csv",
        "train_assignments_u32.bin", "val_assignments_u32.bin",
        "pca_spec.txt", "source_pca_sha256.txt",
        "source_kmeans_summary.txt", "source_kmeans_sha256.txt"
    };
    size_t i;
    entries[0].kind = "INPUT"; entries[0].name = names[0];
    entries[0].path = layout->manifest;
    if (gs_join(layout->base, "train_rows.csv", paths[0], GS_PATH_SIZE) ||
        gs_join(layout->base, "val_rows.csv", paths[1], GS_PATH_SIZE) ||
        gs_join(layout->kmeans, "train_assignments_u32.bin", paths[2],
                GS_PATH_SIZE) ||
        gs_join(layout->kmeans, "val_assignments_u32.bin", paths[3],
                GS_PATH_SIZE) ||
        gs_join(layout->pca, "pca_spec.txt", paths[4], GS_PATH_SIZE) ||
        gs_join(layout->pca, "sha256.txt", paths[5], GS_PATH_SIZE) ||
        gs_join(layout->kmeans, "summary.txt", paths[6], GS_PATH_SIZE) ||
        gs_join(layout->kmeans, "sha256.txt", paths[7], GS_PATH_SIZE)) return 1;
    for (i = 1; i < 9u; i++) {
        entries[i].kind = "INPUT"; entries[i].name = names[i];
        entries[i].path = paths[i - 1u];
    }
    for (i = 0; i < 9u; i++) {
        if (gs_join(output, gs_historical_artifacts[i], paths[i + 8u],
                    GS_PATH_SIZE)) return 1;
        entries[i + 9u].kind = "ARTIFACT";
        entries[i + 9u].name = gs_historical_artifacts[i];
        entries[i + 9u].path = paths[i + 8u];
    }
    return 0;
}

static int gsp224_write_sha(
    const GSP224Layout *layout,
    const char *output
)
{
    GSChecksumEntry entries[18];
    char paths[17][GS_PATH_SIZE], hash[65];
    FILE *file = NULL;
    size_t i;
    int failed = 0;
    if (gsp224_evaluation_entries(layout, output, entries, paths) ||
        gs_open_output(output, "sha256.txt", &file)) return 1;
    for (i = 0; !failed && i < 18u; i++) {
        if (gs_sha_file(entries[i].path, hash) ||
            fprintf(file, "%s %s %s\n", entries[i].kind, hash,
                    entries[i].name) < 0) failed = 1;
    }
    return gs_finish_output(file, failed);
}

static int gsp224_validate_output(
    const GSP224Layout *layout,
    const char *output
)
{
    GSChecksumEntry entries[18];
    char paths[17][GS_PATH_SIZE], sha[GS_PATH_SIZE];
    if (gs_exact_files(output, gs_output_files, 10u) ||
        gsp224_evaluation_entries(layout, output, entries, paths) ||
        gs_join(output, "sha256.txt", sha, sizeof(sha)) ||
        gs_verify_checksum_manifest(sha, entries, 18u)) return 1;
    return 0;
}

static int gsp224_failure_hook(const char *partial)
{
    const char *value = getenv(GS_FAILURE_ENV);
    return value != NULL && strcmp(value, "1") == 0 &&
        partial != NULL && strstr(gs_basename(partial), ".partial") != NULL;
}

static void gsp224_usage(const char *program)
{
    printf(
        "Usage: %s <manifest_events> <pca_artifact_directory> "
        "<pca90|pca95|pca99> <kmeans_directory> <22|36> "
        "<output_partial_directory> <code_commit>\n", program
    );
}

int main(int argc, char **argv)
{
    GSP224PCA profile;
    GSP224KMeans kmeans;
    GSP224Layout layout;
    GSEventManifest manifest;
    GSBaselineClusteringMetrics train_metrics, validation_metrics;
    unsigned char *coverage = NULL;
    uint32_t *train_labels = NULL, *validation_labels = NULL;
    uint32_t *train_assignments = NULL, *validation_assignments = NULL;
    char train_rows[GS_PATH_SIZE], validation_rows[GS_PATH_SIZE];
    char final[GS_PATH_SIZE], before_train[65], before_validation[65];
    char after_train[65], after_validation[65];
    size_t k, train_bytes, validation_bytes, i;
    int partial_created = 0, rc = 1;
    memset(&manifest, 0, sizeof(manifest));
    memset(&train_metrics, 0, sizeof(train_metrics));
    memset(&validation_metrics, 0, sizeof(validation_metrics));
    memset(&kmeans, 0, sizeof(kmeans));
    memset(&layout, 0, sizeof(layout));
    if (argc == 2 && strcmp(argv[1], "--help") == 0) {
        gsp224_usage(argv[0]); return 0;
    }
    if (argc != 8 || !gs_platform_valid() ||
        setlocale(LC_NUMERIC, "C") == NULL ||
        gsp224_token(argc > 3 ? argv[3] : NULL, &profile) ||
        gs_parse_size(argc > 5 ? argv[5] : NULL, &k) ||
        (k != 22u && k != 36u) ||
        !gsp224_commit_valid(argc > 7 ? argv[7] : NULL)) {
        fprintf(stderr, "ERROR: invalid arguments or platform.\n");
        return 1;
    }
    if (gsp224_resolve(argv[1], argv[2], argv[4], k, &profile, &layout)) {
        fprintf(stderr, "ERROR: canonical source resolution failed.\n");
        goto cleanup;
    }
    if (gsp224_validate_base(&layout)) {
        fprintf(stderr, "ERROR: frozen no-PCA base is invalid.\n");
        goto cleanup;
    }
    if (gsp224_validate_pca(&layout, &profile)) {
        fprintf(stderr, "ERROR: frozen PCA profile is invalid.\n");
        goto cleanup;
    }
    if (gsp224_validate_kmeans(&layout, &profile, k, &kmeans)) {
        fprintf(stderr, "ERROR: frozen PCA k-means is invalid.\n");
        goto cleanup;
    }
    if (gs_checked_product(profile.train_rows, sizeof(uint32_t),
                           &train_bytes) ||
        gs_checked_product(profile.validation_rows, sizeof(uint32_t),
                           &validation_bytes)) {
        fprintf(stderr, "ERROR: derived dimensions overflow.\n");
        goto cleanup;
    }
    if (gsp224_output_paths(argv[6], &layout, k, final, sizeof(final)) ||
        gs_join(layout.base, "train_rows.csv", train_rows,
                sizeof(train_rows)) ||
        gs_join(layout.base, "val_rows.csv", validation_rows,
                sizeof(validation_rows))) {
        fprintf(stderr, "ERROR: transactional output location is invalid.\n");
        goto cleanup;
    }
    printf("[PASS] canonical_source_and_provenance_contract\n");
    coverage = (unsigned char *)calloc(GS_EVENT_COUNT, 1u);
    train_labels = (uint32_t *)malloc(train_bytes);
    validation_labels = (uint32_t *)malloc(validation_bytes);
    train_assignments = (uint32_t *)malloc(train_bytes);
    validation_assignments = (uint32_t *)malloc(validation_bytes);
    if (!coverage || !train_labels || !validation_labels ||
        !train_assignments || !validation_assignments ||
        gs_read_assignments(kmeans.train_assignments, train_assignments,
                            profile.train_rows, k) ||
        gs_read_assignments(kmeans.validation_assignments,
                            validation_assignments,
                            profile.validation_rows, k) ||
        gs_sha_file(kmeans.train_assignments, before_train) ||
        gs_sha_file(kmeans.validation_assignments, before_validation)) {
        fprintf(stderr, "ERROR: frozen assignments are invalid.\n");
        goto cleanup;
    }
    printf("[PASS] frozen_assignments_before_labels\n");
    if (gs_event_manifest_load(layout.manifest, &manifest) ||
        gs_validate_manifest(&manifest) ||
        gs_load_rows(train_rows, "train", profile.train_rows, &manifest,
                     coverage, train_labels) ||
        gs_load_rows(validation_rows, "validation",
                     profile.validation_rows, &manifest,
                     coverage, validation_labels)) {
        fprintf(stderr, "ERROR: manifest/row map contract is invalid.\n");
        goto cleanup;
    }
    for (i = 0, k = 0; i < GS_EVENT_COUNT; i++) if (coverage[i]) k++;
    if (k != profile.train_rows + profile.validation_rows ||
        gs_sha_file(kmeans.train_assignments, after_train) ||
        gs_sha_file(kmeans.validation_assignments, after_validation) ||
        strcmp(before_train, after_train) ||
        strcmp(before_validation, after_validation)) {
        fprintf(stderr, "ERROR: posterior label alignment failed.\n");
        goto cleanup;
    }
    k = kmeans.k;
    printf("[PASS] posterior_label_alignment\n");
    if (gs_baseline_clustering_metrics_compute(
            train_labels, train_assignments, profile.train_rows,
            GS_LABEL_COUNT, k, &train_metrics
        ) != GS_BASELINE_CLUSTERING_METRICS_SUCCESS ||
        gs_baseline_clustering_metrics_compute(
            validation_labels, validation_assignments,
            profile.validation_rows, GS_LABEL_COUNT, k,
            &validation_metrics
        ) != GS_BASELINE_CLUSTERING_METRICS_SUCCESS ||
        !gs_metrics_valid(&train_metrics, profile.train_rows, k) ||
        !gs_metrics_valid(&validation_metrics,
                          profile.validation_rows, k)) {
        fprintf(stderr, "ERROR: metric computation failed.\n");
        goto cleanup;
    }
    if (GS_MKDIR(argv[6]) != 0) {
        fprintf(stderr, "ERROR: cannot create partial output.\n");
        goto cleanup;
    }
    partial_created = 1;
    if (gsp224_failure_hook(argv[6]) ||
        gsp224_write_nine(argv[6], &profile, &kmeans, argv[7],
                          &train_metrics, &validation_metrics) ||
        gsp224_write_sha(&layout, argv[6]) ||
        gsp224_validate_output(&layout, argv[6]) ||
        gs_sha_file(kmeans.train_assignments, after_train) ||
        gs_sha_file(kmeans.validation_assignments, after_validation) ||
        strcmp(before_train, after_train) ||
        strcmp(before_validation, after_validation) ||
        rename(argv[6], final) != 0) {
        fprintf(stderr, "ERROR: transactional output failed.\n");
        goto cleanup;
    }
    partial_created = 0;
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
    printf("MINI224 SUBSET1000 PCA LABEL EVALUATION COMPLETED\n");
    rc = 0;
cleanup:
    if (partial_created && gs_cleanup_partial(argv[6])) rc = 1;
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
