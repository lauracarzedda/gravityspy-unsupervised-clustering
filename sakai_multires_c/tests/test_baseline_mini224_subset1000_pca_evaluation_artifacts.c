#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#define main gs_legacy_mini224_subset1000_evaluation_validator_main
#include "test_baseline_mini224_subset1000_evaluation_artifacts.c"
#undef main

enum {
    GSPT224_PCA_FILES = 12,
    GSPT224_KMEANS_FILES = 13,
    GSPT224_CHECKSUM_ENTRIES = 18
};

#define GSPT224_BASE_COMMIT \
    "d87ca996e7922d6873a54d952aec27cdb99a5998"
#define GSPT224_PCA_COMMIT \
    "2cd2dc7f53d8a97fd1c69f88cfa416416818599c"
#define GSPT224_KMEANS_COMMIT \
    "509bc14a1a0e3c9e71dd053b7f9205081df320ef"

typedef struct {
    char token[8];
    char profile[64];
    char threshold[8];
    char train_score[64];
    char validation_score[64];
    size_t expected_retained;
    size_t retained;
    size_t train_rows;
    size_t validation_rows;
    size_t original_features;
    size_t rank;
    size_t nullspace;
} GSPT224Profile;

typedef struct {
    char repository[GS_TEST_PATH_LEN];
    char manifest[GS_TEST_PATH_LEN];
    char base[GS_TEST_PATH_LEN];
    char pca[GS_TEST_PATH_LEN];
    char kmeans[GS_TEST_PATH_LEN];
    char evaluation[GS_TEST_PATH_LEN];
} GSPT224Layout;

static const char *const gspt224_kmeans_files[GSPT224_KMEANS_FILES] = {
    "centroids_f64.bin", "train_assignments_u32.bin",
    "val_assignments_u32.bin", "train_squared_distances_f64.bin",
    "val_squared_distances_f64.bin", "cluster_sizes_train_u64.bin",
    "initial_centroid_indices_u64.bin", "train_assignments.csv",
    "val_assignments.csv", "cluster_sizes.csv", "initial_centroids.csv",
    "summary.txt", "sha256.txt"
};

static int gspt224_parent(
    const char *path,
    char *parent,
    size_t capacity
)
{
    size_t length, split;
    if (!gs_test_value_valid(path) || parent == NULL || capacity == 0u) return 1;
    length = strlen(path);
    while (length && (path[length - 1u] == '/' ||
                      path[length - 1u] == '\\')) length--;
    split = length;
    while (split && path[split - 1u] != '/' &&
           path[split - 1u] != '\\') split--;
    while (split > 1u && (path[split - 1u] == '/' ||
                          path[split - 1u] == '\\')) split--;
    if (!split || split >= capacity) return 1;
    memcpy(parent, path, split);
    parent[split] = '\0';
    return 0;
}

static void gspt224_normalize(char *path)
{
    size_t i;
    for (i = 0; path && path[i]; i++) {
        if (path[i] == '\\') path[i] = '/';
#ifdef _WIN32
        if (path[i] >= 'A' && path[i] <= 'Z')
            path[i] = (char)(path[i] - 'A' + 'a');
#endif
    }
}

static int gspt224_reparse_free(const char *path)
{
#ifdef _WIN32
    char full[GS_TEST_PATH_LEN], probe[GS_TEST_PATH_LEN];
    size_t i, length;
    DWORD attributes;
    if (!path || !_fullpath(full, path, sizeof(full))) return 0;
    length = strlen(full);
    for (i = 0; i <= length; i++) {
        if (i == length || full[i] == '/' || full[i] == '\\') {
            if (i <= 2u) continue;
            memcpy(probe, full, i);
            probe[i] = '\0';
            attributes = GetFileAttributesA(probe);
            if (attributes == INVALID_FILE_ATTRIBUTES ||
                (attributes & FILE_ATTRIBUTE_REPARSE_POINT)) return 0;
        }
    }
#else
    (void)path;
#endif
    return 1;
}

static int gspt224_same(const char *left_path, const char *right_path)
{
    char left[GS_TEST_PATH_LEN], right[GS_TEST_PATH_LEN];
    if (!gspt224_reparse_free(left_path) ||
        !gspt224_reparse_free(right_path) ||
        gs_test_canonical_existing(left_path, left, sizeof(left)) ||
        gs_test_canonical_existing(right_path, right, sizeof(right))) return 0;
    gspt224_normalize(left);
    gspt224_normalize(right);
    return strcmp(left, right) == 0;
}

static int gspt224_select(
    const char *token,
    GSPT224Profile *profile
)
{
    if (!token || !profile) return 1;
    memset(profile, 0, sizeof(*profile));
    if (!strcmp(token, "pca90")) {
        profile->expected_retained = 15u; strcpy(profile->threshold, "0.90");
    } else if (!strcmp(token, "pca95")) {
        profile->expected_retained = 38u; strcpy(profile->threshold, "0.95");
    } else if (!strcmp(token, "pca99")) {
        profile->expected_retained = 168u; strcpy(profile->threshold, "0.99");
    } else return 1;
    strcpy(profile->token, token);
    if (snprintf(profile->profile, sizeof(profile->profile),
                 "mini224_subset1000_pool14_%s", token) < 0 ||
        snprintf(profile->train_score, sizeof(profile->train_score),
                 "train_scores_%s_f32.bin", token) < 0 ||
        snprintf(profile->validation_score,
                 sizeof(profile->validation_score),
                 "val_scores_%s_f32.bin", token) < 0) return 1;
    return 0;
}

static int gspt224_resolve(
    const char *manifest_argument,
    const char *pca_argument,
    const char *kmeans_argument,
    const char *evaluation_argument,
    size_t k,
    const GSPT224Profile *profile,
    GSPT224Layout *layout
)
{
    char baseline[GS_TEST_PATH_LEN], results[GS_TEST_PATH_LEN];
    char sakai[GS_TEST_PATH_LEN], expected[GS_TEST_PATH_LEN], suffix[64];
    int written;
    if (!layout || !profile ||
        gs_test_canonical_existing(pca_argument, layout->pca,
                                   sizeof(layout->pca)) ||
        strcmp(gs_test_basename(layout->pca), profile->profile) ||
        gspt224_parent(layout->pca, baseline, sizeof(baseline)) ||
        strcmp(gs_test_basename(baseline), "baseline") ||
        gspt224_parent(baseline, results, sizeof(results)) ||
        strcmp(gs_test_basename(results), "results") ||
        gspt224_parent(results, sakai, sizeof(sakai)) ||
        strcmp(gs_test_basename(sakai), "sakai_multires_c") ||
        gspt224_parent(sakai, layout->repository,
                       sizeof(layout->repository)) ||
        gs_test_path_join(layout->repository, "manifest_events_272.csv",
                          layout->manifest, sizeof(layout->manifest)) ||
        gs_test_path_join(baseline,
                          "mini224_subset1000_pool14_no_pca",
                          layout->base, sizeof(layout->base)) ||
        (written = snprintf(suffix, sizeof(suffix), "kmeans_seed123/k%lu",
                            (unsigned long)k)) < 0 ||
        (size_t)written >= sizeof(suffix) ||
        gs_test_path_join(layout->pca, suffix, layout->kmeans,
                          sizeof(layout->kmeans)) ||
        (written = snprintf(suffix, sizeof(suffix),
                            "label_evaluation_seed123/k%lu",
                            (unsigned long)k)) < 0 ||
        (size_t)written >= sizeof(suffix) ||
        gs_test_path_join(layout->pca, suffix, layout->evaluation,
                          sizeof(layout->evaluation)) ||
        !gspt224_same(manifest_argument, layout->manifest) ||
        !gspt224_same(kmeans_argument, layout->kmeans) ||
        !gspt224_same(evaluation_argument, layout->evaluation) ||
        !gspt224_reparse_free(layout->repository) ||
        !gspt224_reparse_free(layout->base) ||
        !gspt224_reparse_free(layout->pca)) return 1;
    if (gs_test_path_join(layout->repository,
                          "sakai_multires_c/results/baseline",
                          expected, sizeof(expected)) ||
        !gspt224_same(baseline, expected)) return 1;
    return 0;
}

static int gspt224_read(
    const char *directory,
    const char *filename,
    const char *key,
    char *value,
    size_t capacity
)
{
    char path[GS_TEST_PATH_LEN];
    return gs_test_path_join(directory, filename, path, sizeof(path)) ||
        gs_test_read_summary_value(path, key, value, capacity);
}

static int gspt224_exact_regular_files(
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
    if (!seen || !(stream = opendir(directory))) {
        free(seen);
        return 1;
    }
    while (!failed) {
        char path[GS_TEST_PATH_LEN];
        int regular, is_directory;
        size_t index;
        errno = 0;
        entry = readdir(stream);
        if (!entry) {
            if (errno) failed = 1;
            break;
        }
        if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, ".."))
            continue;
        if (gs_test_path_join(directory, entry->d_name, path, sizeof(path)) ||
            gs_test_path_state(path, &regular, &is_directory) != 1) {
            failed = 1;
        } else if (is_directory) {
            if (strcmp(entry->d_name, "kmeans_seed123") &&
                strcmp(entry->d_name, "label_evaluation_seed123")) failed = 1;
        } else if (!regular ||
                   !gs_test_name_index(entry->d_name, names, count, &index) ||
                   seen[index]) {
            failed = 1;
        } else {
            seen[index] = 1u;
        }
    }
    if (closedir(stream)) failed = 1;
    for (i = 0; !failed && i < count; i++) if (!seen[i]) failed = 1;
    free(seen);
    return failed;
}

static int gspt224_expect(
    const char *directory,
    const char *filename,
    const char *key,
    const char *expected
)
{
    char value[256];
    return gspt224_read(directory, filename, key, value, sizeof(value)) ||
        strcmp(value, expected);
}

static int gspt224_base_entries(
    const GSPT224Layout *layout,
    GSTestChecksumEntry entries[13],
    char paths[13][GS_TEST_PATH_LEN]
)
{
    static const char *const inputs[4] = {
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
        if (gs_test_path_join(layout->repository, relatives[i], paths[i],
                              GS_TEST_PATH_LEN)) return 1;
        entries[i].kind = "INPUT"; entries[i].name = inputs[i];
        entries[i].path = paths[i];
    }
    for (i = 0; i < 9u; i++) {
        if (gs_test_path_join(layout->base, artifacts[i], paths[i + 4u],
                              GS_TEST_PATH_LEN)) return 1;
        entries[i + 4u].kind = "ARTIFACT";
        entries[i + 4u].name = artifacts[i];
        entries[i + 4u].path = paths[i + 4u];
    }
    return 0;
}

static int gspt224_validate_base(const GSPT224Layout *layout)
{
    GSTestChecksumEntry entries[13];
    char paths[13][GS_TEST_PATH_LEN], sha[GS_TEST_PATH_LEN];
    size_t i, size;
    if (gs_test_base_inventory(layout->base) ||
        gs_test_validate_base_metadata(layout->base) ||
        gspt224_base_entries(layout, entries, paths) ||
        gs_test_path_join(layout->base, "sha256.txt", sha, sizeof(sha)) ||
        gs_test_verify_checksum_manifest(sha, entries, 13u)) return 1;
    for (i = 0; i < 10u; i++) {
        if (gs_test_path_join(layout->base, gs_test_base_files[i],
                              paths[0], GS_TEST_PATH_LEN) ||
            gs_test_file_size(paths[0], &size) ||
            size != gs_test_base_sizes[i]) return 1;
    }
    return 0;
}

static int gspt224_pca_entries(
    const GSPT224Layout *layout,
    const GSPT224Profile *profile,
    GSTestChecksumEntry entries[18],
    char paths[18][GS_TEST_PATH_LEN]
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
    static const char *const artifacts[8] = {
        "pca_spec.txt", "mean_f64.bin", "components_f64.bin",
        "eigenvalues_f64.bin", "explained_variance_ratio_f64.bin",
        "cumulative_explained_variance_ratio_f64.bin", "thresholds.csv",
        "spectrum.csv"
    };
    size_t i;
    for (i = 0; i < 7u; i++) {
        if (gs_test_path_join(layout->base, input_sources[i], paths[i],
                              GS_TEST_PATH_LEN)) return 1;
        entries[i].kind = "INPUT"; entries[i].name = input_names[i];
        entries[i].path = paths[i];
    }
    for (i = 0; i < 8u; i++) {
        if (gs_test_path_join(layout->pca, artifacts[i], paths[i + 7u],
                              GS_TEST_PATH_LEN)) return 1;
        entries[i + 7u].kind = "ARTIFACT";
        entries[i + 7u].name = artifacts[i];
        entries[i + 7u].path = paths[i + 7u];
    }
    if (gs_test_path_join(layout->pca, profile->train_score, paths[15],
                          GS_TEST_PATH_LEN) ||
        gs_test_path_join(layout->pca, profile->validation_score, paths[16],
                          GS_TEST_PATH_LEN) ||
        gs_test_path_join(layout->pca, "summary.txt", paths[17],
                          GS_TEST_PATH_LEN)) return 1;
    entries[15].kind = "ARTIFACT"; entries[15].name = profile->train_score;
    entries[15].path = paths[15];
    entries[16].kind = "ARTIFACT";
    entries[16].name = profile->validation_score;
    entries[16].path = paths[16];
    entries[17].kind = "ARTIFACT"; entries[17].name = "summary.txt";
    entries[17].path = paths[17];
    return 0;
}

static int gspt224_validate_pca(
    const GSPT224Layout *layout,
    GSPT224Profile *profile
)
{
    static const char *const fixed[10] = {
        "components_f64.bin",
        "cumulative_explained_variance_ratio_f64.bin",
        "eigenvalues_f64.bin", "explained_variance_ratio_f64.bin",
        "mean_f64.bin", "pca_spec.txt", "sha256.txt", "spectrum.csv",
        "summary.txt", "thresholds.csv"
    };
    const char *files[12];
    GSTestChecksumEntry entries[18];
    char paths[18][GS_TEST_PATH_LEN], sha[GS_TEST_PATH_LEN], value[256];
    size_t i;
    for (i = 0; i < 10u; i++) files[i] = fixed[i];
    files[10] = profile->train_score;
    files[11] = profile->validation_score;
    if (gspt224_exact_regular_files(layout->pca, files, 12u) ||
        gspt224_pca_entries(layout, profile, entries, paths) ||
        gs_test_path_join(layout->pca, "sha256.txt", sha, sizeof(sha)) ||
        gs_test_verify_checksum_manifest(sha, entries, 18u) ||
        gspt224_expect(layout->pca, "pca_spec.txt", "profile",
                       profile->profile) ||
        gspt224_expect(layout->pca, "pca_spec.txt", "selected_threshold",
                       profile->threshold) ||
        gspt224_expect(layout->pca, "pca_spec.txt", "numerical_regime",
                       "rank_deficient") ||
        gspt224_expect(layout->pca, "pca_spec.txt", "nullspace_validation",
                       "orthogonal_projector") ||
        gspt224_expect(layout->pca, "pca_spec.txt",
                       "retained_prefix_validation", "bitwise") ||
        gspt224_expect(layout->pca, "pca_spec.txt", "labels_used", "false") ||
        gspt224_expect(layout->pca, "summary.txt", "code_commit",
                       GSPT224_PCA_COMMIT)) return 1;
#define GSPT224_PARSE(key, member) \
    if (gspt224_read(layout->pca, "pca_spec.txt", key, value, \
                     sizeof(value)) || \
        gs_test_parse_size(value, &profile->member)) return 1
    GSPT224_PARSE("retained_component_count", retained);
    GSPT224_PARSE("train_rows", train_rows);
    GSPT224_PARSE("validation_rows", validation_rows);
    GSPT224_PARSE("original_feature_count", original_features);
    GSPT224_PARSE("numerical_rank", rank);
    GSPT224_PARSE("nullspace_dimension", nullspace);
#undef GSPT224_PARSE
    if (profile->retained != profile->expected_retained ||
        profile->train_rows != 800u ||
        profile->validation_rows != 200u ||
        profile->original_features != 1024u ||
        profile->rank != 799u || profile->nullspace != 225u) return 1;
    return 0;
}

static int gspt224_kmeans_entries(
    const GSPT224Layout *layout,
    const GSPT224Profile *profile,
    GSTestChecksumEntry entries[17],
    char paths[17][GS_TEST_PATH_LEN]
)
{
    static const char *const artifacts[12] = {
        "centroids_f64.bin", "train_assignments_u32.bin",
        "val_assignments_u32.bin", "train_squared_distances_f64.bin",
        "val_squared_distances_f64.bin", "cluster_sizes_train_u64.bin",
        "initial_centroid_indices_u64.bin", "train_assignments.csv",
        "val_assignments.csv", "cluster_sizes.csv", "initial_centroids.csv",
        "summary.txt"
    };
    const char *sources[5] = {
        "pca_spec.txt", "thresholds.csv", profile->train_score,
        profile->validation_score, "sha256.txt"
    };
    const char *names[5] = {
        "pca_spec.txt", "thresholds.csv", profile->train_score,
        profile->validation_score, "source_pca_sha256.txt"
    };
    size_t i;
    for (i = 0; i < 5u; i++) {
        if (gs_test_path_join(layout->pca, sources[i], paths[i],
                              GS_TEST_PATH_LEN)) return 1;
        entries[i].kind = "INPUT"; entries[i].name = names[i];
        entries[i].path = paths[i];
    }
    for (i = 0; i < 12u; i++) {
        if (gs_test_path_join(layout->kmeans, artifacts[i], paths[i + 5u],
                              GS_TEST_PATH_LEN)) return 1;
        entries[i + 5u].kind = "ARTIFACT";
        entries[i + 5u].name = artifacts[i];
        entries[i + 5u].path = paths[i + 5u];
    }
    return 0;
}

static int gspt224_validate_kmeans(
    const GSPT224Layout *layout,
    const GSPT224Profile *profile,
    size_t k
)
{
    GSTestChecksumEntry entries[17];
    char paths[17][GS_TEST_PATH_LEN], sha[GS_TEST_PATH_LEN];
    char expected[64], value[256];
    size_t parsed;
    if (gs_test_exact_files(layout->kmeans, gspt224_kmeans_files, 13u) ||
        gspt224_kmeans_entries(layout, profile, entries, paths) ||
        gs_test_path_join(layout->kmeans, "sha256.txt", sha, sizeof(sha)) ||
        gs_test_verify_checksum_manifest(sha, entries, 17u) ||
        snprintf(expected, sizeof(expected),
                 "mini224_subset1000_pool14_%s_kmeans", profile->token) < 0 ||
        gspt224_expect(layout->kmeans, "summary.txt", "profile", expected) ||
        gspt224_expect(layout->kmeans, "summary.txt", "input_profile",
                       profile->profile) ||
        gspt224_expect(layout->kmeans, "summary.txt", "pca_profile",
                       profile->token) ||
        gspt224_expect(layout->kmeans, "summary.txt", "policy_c", "true") ||
        gspt224_expect(layout->kmeans, "summary.txt", "fit_input",
                       "train_only") ||
        gspt224_expect(layout->kmeans, "summary.txt",
                       "validation_assigned_after_fit", "true") ||
        gspt224_expect(layout->kmeans, "summary.txt", "labels_used", "false") ||
        gspt224_expect(layout->kmeans, "summary.txt", "pca_code_commit",
                       GSPT224_PCA_COMMIT) ||
        gspt224_expect(layout->kmeans, "summary.txt", "code_commit",
                       GSPT224_KMEANS_COMMIT)) return 1;
#define GSPT224_KM(key, expected_value) \
    if (gspt224_read(layout->kmeans, "summary.txt", key, value, \
                     sizeof(value)) || gs_test_parse_size(value, &parsed) || \
        parsed != (expected_value)) return 1
    GSPT224_KM("retained_component_count", profile->retained);
    GSPT224_KM("original_feature_count", profile->original_features);
    GSPT224_KM("numerical_rank", profile->rank);
    GSPT224_KM("nullspace_dimension", profile->nullspace);
    GSPT224_KM("train_rows", profile->train_rows);
    GSPT224_KM("validation_rows", profile->validation_rows);
    GSPT224_KM("k", k);
    GSPT224_KM("seed", 123u);
#undef GSPT224_KM
    return 0;
}

static int gspt224_evaluation_entries(
    const GSPT224Layout *layout,
    GSTestChecksumEntry entries[18],
    char paths[17][GS_TEST_PATH_LEN]
)
{
    static const char *const input_names[9] = {
        "manifest_events_272.csv", "train_rows.csv", "val_rows.csv",
        "train_assignments_u32.bin", "val_assignments_u32.bin",
        "pca_spec.txt", "source_pca_sha256.txt",
        "source_kmeans_summary.txt", "source_kmeans_sha256.txt"
    };
    size_t i;
    entries[0].kind = "INPUT"; entries[0].name = input_names[0];
    entries[0].path = layout->manifest;
    if (gs_test_path_join(layout->base, "train_rows.csv", paths[0],
                          GS_TEST_PATH_LEN) ||
        gs_test_path_join(layout->base, "val_rows.csv", paths[1],
                          GS_TEST_PATH_LEN) ||
        gs_test_path_join(layout->kmeans, "train_assignments_u32.bin",
                          paths[2], GS_TEST_PATH_LEN) ||
        gs_test_path_join(layout->kmeans, "val_assignments_u32.bin",
                          paths[3], GS_TEST_PATH_LEN) ||
        gs_test_path_join(layout->pca, "pca_spec.txt", paths[4],
                          GS_TEST_PATH_LEN) ||
        gs_test_path_join(layout->pca, "sha256.txt", paths[5],
                          GS_TEST_PATH_LEN) ||
        gs_test_path_join(layout->kmeans, "summary.txt", paths[6],
                          GS_TEST_PATH_LEN) ||
        gs_test_path_join(layout->kmeans, "sha256.txt", paths[7],
                          GS_TEST_PATH_LEN)) return 1;
    for (i = 1; i < 9u; i++) {
        entries[i].kind = "INPUT"; entries[i].name = input_names[i];
        entries[i].path = paths[i - 1u];
    }
    for (i = 0; i < 9u; i++) {
        if (gs_test_path_join(layout->evaluation,
                              gs_test_historical_artifacts[i],
                              paths[i + 8u], GS_TEST_PATH_LEN)) return 1;
        entries[i + 9u].kind = "ARTIFACT";
        entries[i + 9u].name = gs_test_historical_artifacts[i];
        entries[i + 9u].path = paths[i + 8u];
    }
    return 0;
}

static int gspt224_validate_inventory(const GSPT224Layout *layout)
{
    GSTestChecksumEntry entries[18];
    char paths[17][GS_TEST_PATH_LEN], sha[GS_TEST_PATH_LEN], partial[GS_TEST_PATH_LEN];
    int regular, directory;
    if (gs_test_exact_files(layout->evaluation,
                            gs_test_output_files, 10u) ||
        gspt224_evaluation_entries(layout, entries, paths) ||
        gs_test_path_join(layout->evaluation, "sha256.txt",
                          sha, sizeof(sha)) ||
        gs_test_verify_checksum_manifest(sha, entries, 18u) ||
        strlen(layout->evaluation) + strlen(GS_TEST_PARTIAL_SUFFIX) + 1u >
            sizeof(partial)) return 1;
    strcpy(partial, layout->evaluation);
    strcat(partial, GS_TEST_PARTIAL_SUFFIX);
    return gs_test_path_state(partial, &regular, &directory);
}

static int gspt224_validate_summary(
    const GSPT224Layout *layout,
    const GSPT224Profile *profile,
    size_t k,
    const char *commit,
    const GSBaselineClusteringMetrics *train,
    const GSBaselineClusteringMetrics *validation
)
{
    static const char *const metric_keys[10] = {
        "train_purity", "train_nmi_arithmetic", "train_ari",
        "train_homogeneity", "train_completeness", "validation_purity",
        "validation_nmi_arithmetic", "validation_ari",
        "validation_homogeneity", "validation_completeness"
    };
    double expected_metrics[10];
    char expected[128], value[256];
    size_t parsed, i;
    if (snprintf(expected, sizeof(expected),
                 "mini224_subset1000_pool14_%s_label_evaluation",
                 profile->token) < 0 ||
        gspt224_expect(layout->evaluation, "summary.txt", "profile",
                       expected) ||
        gspt224_expect(layout->evaluation, "summary.txt", "input_profile",
                       profile->profile) ||
        gspt224_expect(layout->evaluation, "summary.txt", "pca_profile",
                       profile->token) ||
        gspt224_expect(layout->evaluation, "summary.txt", "pca_threshold",
                       profile->threshold) ||
        gspt224_expect(layout->evaluation, "summary.txt", "policy_c", "true") ||
        gspt224_expect(layout->evaluation, "summary.txt", "labels_policy",
                       "posterior_evaluation_only") ||
        gspt224_expect(layout->evaluation, "summary.txt",
                       "labels_used_for_feature_extraction", "false") ||
        gspt224_expect(layout->evaluation, "summary.txt",
                       "labels_used_for_standardization", "false") ||
        gspt224_expect(layout->evaluation, "summary.txt",
                       "labels_used_for_pca", "false") ||
        gspt224_expect(layout->evaluation, "summary.txt",
                       "labels_used_for_kmeans", "false") ||
        gspt224_expect(layout->evaluation, "summary.txt",
                       "labels_used_for_evaluation", "true") ||
        gspt224_expect(layout->evaluation, "summary.txt",
                       "base_code_commit", GSPT224_BASE_COMMIT) ||
        gspt224_expect(layout->evaluation, "summary.txt",
                       "pca_code_commit", GSPT224_PCA_COMMIT) ||
        gspt224_expect(layout->evaluation, "summary.txt",
                       "kmeans_code_commit", GSPT224_KMEANS_COMMIT) ||
        gspt224_expect(layout->evaluation, "summary.txt",
                       "code_commit", commit)) return 1;
#define GSPT224_SUM(key, expected_value) \
    if (gspt224_read(layout->evaluation, "summary.txt", key, value, \
                     sizeof(value)) || gs_test_parse_size(value, &parsed) || \
        parsed != (expected_value)) return 1
    GSPT224_SUM("retained_component_count", profile->retained);
    GSPT224_SUM("original_feature_count", profile->original_features);
    GSPT224_SUM("numerical_rank", profile->rank);
    GSPT224_SUM("nullspace_dimension", profile->nullspace);
    GSPT224_SUM("train_rows", profile->train_rows);
    GSPT224_SUM("validation_rows", profile->validation_rows);
    GSPT224_SUM("k", k);
    GSPT224_SUM("cluster_count", k);
#undef GSPT224_SUM
    expected_metrics[0] = train->purity;
    expected_metrics[1] = train->normalized_mutual_information;
    expected_metrics[2] = train->adjusted_rand_index;
    expected_metrics[3] = train->homogeneity;
    expected_metrics[4] = train->completeness;
    expected_metrics[5] = validation->purity;
    expected_metrics[6] = validation->normalized_mutual_information;
    expected_metrics[7] = validation->adjusted_rand_index;
    expected_metrics[8] = validation->homogeneity;
    expected_metrics[9] = validation->completeness;
    for (i = 0; i < 10u; i++) {
        double actual;
        if (gspt224_read(layout->evaluation, "summary.txt",
                         metric_keys[i], value, sizeof(value)) ||
            gs_test_parse_double(value, &actual) ||
            !gs_test_close(actual, expected_metrics[i])) return 1;
    }
    return 0;
}

static void gspt224_usage(const char *program)
{
    printf(
        "Usage: %s <manifest_events> <pca_artifact_directory> "
        "<pca90|pca95|pca99> <kmeans_directory> <evaluation_directory> "
        "<22|36> <expected_code_commit>\n", program
    );
}

int main(int argc, char **argv)
{
    GSPT224Profile profile;
    GSPT224Layout layout;
    GSEventManifest manifest;
    GSBaselineClusteringMetrics train_metrics, validation_metrics;
    unsigned char *coverage = NULL;
    uint32_t *train_labels = NULL, *validation_labels = NULL;
    uint32_t *train_assignments = NULL, *validation_assignments = NULL;
    uint32_t *train_copy = NULL, *validation_copy = NULL;
    char train_rows[GS_TEST_PATH_LEN], validation_rows[GS_TEST_PATH_LEN];
    char train_assignment_path[GS_TEST_PATH_LEN];
    char validation_assignment_path[GS_TEST_PATH_LEN];
    char train_hash_before[GS_TEST_SHA_TEXT], train_hash_after[GS_TEST_SHA_TEXT];
    char val_hash_before[GS_TEST_SHA_TEXT], val_hash_after[GS_TEST_SHA_TEXT];
    size_t train_bytes, validation_bytes, k, i, covered = 0u;
    int train_metrics_ok = 0, validation_metrics_ok = 0, rc = 1;
    memset(&manifest, 0, sizeof(manifest));
    memset(&train_metrics, 0, sizeof(train_metrics));
    memset(&validation_metrics, 0, sizeof(validation_metrics));
    memset(&layout, 0, sizeof(layout));
    if (argc == 2 && !strcmp(argv[1], "--help")) {
        gspt224_usage(argv[0]); return 0;
    }
    if (argc != 8 || !gs_test_platform_is_valid() ||
        setlocale(LC_NUMERIC, "C") == NULL ||
        gspt224_select(argc > 3 ? argv[3] : NULL, &profile) ||
        gs_test_parse_size(argc > 6 ? argv[6] : NULL, &k) ||
        (k != 22u && k != 36u) ||
        !gs_test_value_valid(argc > 7 ? argv[7] : NULL) ||
        gspt224_resolve(argv[1], argv[2], argv[4], argv[5], k,
                        &profile, &layout) ||
        gspt224_validate_base(&layout) ||
        gspt224_validate_pca(&layout, &profile) ||
        gspt224_validate_kmeans(&layout, &profile, k) ||
        gspt224_validate_inventory(&layout) ||
        gs_test_path_join(layout.base, "train_rows.csv", train_rows,
                          sizeof(train_rows)) ||
        gs_test_path_join(layout.base, "val_rows.csv", validation_rows,
                          sizeof(validation_rows)) ||
        gs_test_path_join(layout.kmeans, "train_assignments_u32.bin",
                          train_assignment_path,
                          sizeof(train_assignment_path)) ||
        gs_test_path_join(layout.kmeans, "val_assignments_u32.bin",
                          validation_assignment_path,
                          sizeof(validation_assignment_path))) {
        fprintf(stderr, "[FAIL] frozen_source_and_evaluation_contract\n");
        return 1;
    }
    if (gs_test_checked_product(profile.train_rows, sizeof(uint32_t),
                                &train_bytes) ||
        gs_test_checked_product(profile.validation_rows, sizeof(uint32_t),
                                &validation_bytes)) {
        fprintf(stderr, "[FAIL] derived_dimension_overflow\n");
        return 1;
    }
    printf("[PASS] frozen_source_and_evaluation_contract\n");
    coverage = (unsigned char *)calloc(GS_TEST_MANIFEST_ROWS, 1u);
    train_labels = (uint32_t *)malloc(train_bytes);
    validation_labels = (uint32_t *)malloc(validation_bytes);
    train_assignments = (uint32_t *)malloc(train_bytes);
    validation_assignments = (uint32_t *)malloc(validation_bytes);
    train_copy = (uint32_t *)malloc(train_bytes);
    validation_copy = (uint32_t *)malloc(validation_bytes);
    if (!coverage || !train_labels || !validation_labels ||
        !train_assignments || !validation_assignments ||
        !train_copy || !validation_copy ||
        gs_test_read_assignments(train_assignment_path, train_assignments,
                                 profile.train_rows, k) ||
        gs_test_read_assignments(validation_assignment_path,
                                 validation_assignments,
                                 profile.validation_rows, k) ||
        gs_test_sha_file(train_assignment_path, train_hash_before) ||
        gs_test_sha_file(validation_assignment_path, val_hash_before) ||
        gs_event_manifest_load(layout.manifest, &manifest) ||
        gs_test_validate_manifest(&manifest) ||
        gs_test_load_rows(train_rows, "train", profile.train_rows,
                          &manifest, coverage, train_labels) ||
        gs_test_load_rows(validation_rows, "validation",
                          profile.validation_rows, &manifest,
                          coverage, validation_labels)) {
        fprintf(stderr, "[FAIL] independent_row_mapping\n");
        goto cleanup;
    }
    for (i = 0; i < GS_TEST_MANIFEST_ROWS; i++) if (coverage[i]) covered++;
    if (covered != 1000u) {
        fprintf(stderr, "[FAIL] independent_row_mapping\n");
        goto cleanup;
    }
    memcpy(train_copy, train_assignments, train_bytes);
    memcpy(validation_copy, validation_assignments, validation_bytes);
    if (gs_baseline_clustering_metrics_compute(
            train_labels, train_assignments, profile.train_rows,
            GS_TEST_LABEL_COUNT, k, &train_metrics
        ) != GS_BASELINE_CLUSTERING_METRICS_SUCCESS ||
        gs_baseline_clustering_metrics_compute(
            validation_labels, validation_assignments,
            profile.validation_rows, GS_TEST_LABEL_COUNT, k,
            &validation_metrics
        ) != GS_BASELINE_CLUSTERING_METRICS_SUCCESS ||
        !gs_test_metrics_consistent(&train_metrics, profile.train_rows, k) ||
        !gs_test_metrics_consistent(&validation_metrics,
                                    profile.validation_rows, k) ||
        gs_test_validate_label_map(layout.evaluation) ||
        gs_test_validate_contingency(
            layout.evaluation, "train_contingency_counts.csv", &train_metrics
        ) ||
        gs_test_validate_contingency(
            layout.evaluation, "val_contingency_counts.csv",
            &validation_metrics
        )) {
        fprintf(stderr, "[FAIL] independent_recomputation\n");
        goto cleanup;
    }
    printf("TRAIN_CONTINGENCY_RECOMPUTED_MATCH=true\n");
    printf("VALIDATION_CONTINGENCY_RECOMPUTED_MATCH=true\n");
    if (gs_test_validate_metrics_csv(
            layout.evaluation, k, &train_metrics, &validation_metrics,
            &train_metrics_ok, &validation_metrics_ok
        ) || !train_metrics_ok || !validation_metrics_ok) goto cleanup;
    printf("TRAIN_METRICS_RECOMPUTED_MATCH=true\n");
    printf("VALIDATION_METRICS_RECOMPUTED_MATCH=true\n");
    if (gs_test_validate_cluster_summary(
            layout.evaluation, "train_cluster_summary.csv", &train_metrics
        ) ||
        gs_test_validate_cluster_summary(
            layout.evaluation, "val_cluster_summary.csv", &validation_metrics
        )) goto cleanup;
    printf("TRAIN_CLUSTER_SUMMARY_RECOMPUTED_MATCH=true\n");
    printf("VALIDATION_CLUSTER_SUMMARY_RECOMPUTED_MATCH=true\n");
    if (gs_test_validate_label_summary(
            layout.evaluation, "train_label_summary.csv", &train_metrics
        ) ||
        gs_test_validate_label_summary(
            layout.evaluation, "val_label_summary.csv", &validation_metrics
        ) ||
        gspt224_validate_summary(&layout, &profile, k, argv[7],
                                 &train_metrics, &validation_metrics) ||
        memcmp(train_assignments, train_copy, train_bytes) ||
        memcmp(validation_assignments, validation_copy, validation_bytes) ||
        gs_test_sha_file(train_assignment_path, train_hash_after) ||
        gs_test_sha_file(validation_assignment_path, val_hash_after) ||
        strcmp(train_hash_before, train_hash_after) ||
        strcmp(val_hash_before, val_hash_after) ||
        gspt224_validate_base(&layout) ||
        gspt224_validate_pca(&layout, &profile) ||
        gspt224_validate_kmeans(&layout, &profile, k)) goto cleanup;
    printf("TRAIN_LABEL_SUMMARY_RECOMPUTED_MATCH=true\n");
    printf("VALIDATION_LABEL_SUMMARY_RECOMPUTED_MATCH=true\n");
    printf("SOURCE_ASSIGNMENTS_UNCHANGED=true\n");
    printf("LABELS_POSTERIOR_ONLY=true\n");
    printf("ALL_PCA224_EVALUATION_CHECKSUMS_MATCH=true\n");
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
    printf("ALL MINI224 SUBSET1000 PCA EVALUATION ARTIFACT TESTS PASSED\n");
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
