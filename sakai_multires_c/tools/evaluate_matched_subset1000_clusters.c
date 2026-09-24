#define _CRT_SECURE_NO_WARNINGS

#ifdef _WIN32
#include <windows.h>
#endif

#define main gs_historical_mini224_evaluator_main
#include "evaluate_baseline_mini224_subset1000_clusters.c"
#undef main

enum {
    ME_TRAIN_ROWS = 800,
    ME_VALIDATION_ROWS = 200,
    ME_SUBSET_ROWS = 1000,
    ME_BASE_INPUT_COUNT = 9,
    ME_BASE_ARTIFACT_COUNT = 9,
    ME_BASE_FILE_COUNT = 10,
    ME_KMEANS_INPUT_COUNT = 6,
    ME_KMEANS_ARTIFACT_COUNT = 12,
    ME_KMEANS_FILE_COUNT = 13,
    ME_EVALUATION_INPUT_COUNT = 5,
    ME_EVALUATION_ARTIFACT_COUNT = 9,
    ME_EVALUATION_FILE_COUNT = 10,
    ME_EVALUATION_PATH_COUNT =
        ME_EVALUATION_INPUT_COUNT - 1 + ME_EVALUATION_ARTIFACT_COUNT
};

#define ME_MATCHED_BASE_COMMIT \
    "6e76182f40f4008b8fdd8b33a9dacc4a20241384"
#define ME_MATCHED_KMEANS_COMMIT \
    "ed45edb277224ad380ce6b81a75acdf780a1e463"
#define ME_FAILURE_ENV \
    "GS_MATCHED_SUBSET1000_EVALUATION_TEST_FAIL_AFTER_PARTIAL"

typedef struct {
    const char *token;
    const char *base_name;
    const char *matched_profile;
    const char *source_profile;
    const char *source_base_name;
    size_t feature_count;
} MEProfile;

typedef struct {
    const MEProfile *profile;
    char root[GS_PATH_SIZE];
    char manifest[GS_PATH_SIZE];
    char base[GS_PATH_SIZE];
    char source[GS_PATH_SIZE];
    char subset_train[GS_PATH_SIZE];
    char subset_validation[GS_PATH_SIZE];
    char base_commit[64];
} MEBase;

typedef struct {
    char directory[GS_PATH_SIZE];
    char code_commit[64];
    size_t k;
} MEKMeans;

static const MEProfile me_profiles[] = {
    {
        "mini75_pool5",
        "mini75_pool5_no_pca",
        "matched_subset1000_mini75_pool5_no_pca",
        "mini75_pool5_no_pca",
        "mini75_pool5_no_pca",
        900u
    },
    {
        "mini112_pool7",
        "mini112_pool7_no_pca",
        "matched_subset1000_mini112_pool7_no_pca",
        "mini112_pool7_no_pca",
        "mini112_pool7_no_pca",
        1024u
    }
};

static const char *const me_base_files[ME_BASE_FILE_COUNT] = {
    "feature_spec.txt",
    "scaler.csv",
    "train_features_raw_f32.bin",
    "val_features_raw_f32.bin",
    "train_features_std_f32.bin",
    "val_features_std_f32.bin",
    "train_rows.csv",
    "val_rows.csv",
    "summary.txt",
    "sha256.txt"
};

static const char *const me_base_input_names[ME_BASE_INPUT_COUNT] = {
    "manifest_events_272.csv",
    "subset224_train_800_seed123.csv",
    "subset224_val_200_seed123.csv",
    "source_feature_spec.txt",
    "source_train_features_raw_f32.bin",
    "source_val_features_raw_f32.bin",
    "source_train_rows.csv",
    "source_val_rows.csv",
    "source_no_pca_sha256.txt"
};

static const char *const me_base_artifacts[ME_BASE_ARTIFACT_COUNT] = {
    "feature_spec.txt",
    "scaler.csv",
    "train_features_raw_f32.bin",
    "val_features_raw_f32.bin",
    "train_features_std_f32.bin",
    "val_features_std_f32.bin",
    "train_rows.csv",
    "val_rows.csv",
    "summary.txt"
};

static const char *const me_kmeans_files[ME_KMEANS_FILE_COUNT] = {
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

static const char *const me_kmeans_inputs[ME_KMEANS_INPUT_COUNT] = {
    "feature_spec.txt",
    "scaler.csv",
    "train_features_std_f32.bin",
    "val_features_std_f32.bin",
    "train_rows.csv",
    "val_rows.csv"
};

static const char *const me_kmeans_artifacts[ME_KMEANS_ARTIFACT_COUNT] = {
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
    "summary.txt"
};

static const char *const me_output_files[ME_EVALUATION_FILE_COUNT] = {
    "label_map.csv",
    "metrics.csv",
    "summary.txt",
    "train_cluster_summary.csv",
    "train_contingency_counts.csv",
    "train_label_summary.csv",
    "val_cluster_summary.csv",
    "val_contingency_counts.csv",
    "val_label_summary.csv",
    "sha256.txt"
};

static const char *const me_evaluation_inputs[ME_EVALUATION_INPUT_COUNT] = {
    "manifest_events_272.csv",
    "train_rows.csv",
    "val_rows.csv",
    "train_assignments_u32.bin",
    "val_assignments_u32.bin"
};

static const char *const me_evaluation_artifacts[
    ME_EVALUATION_ARTIFACT_COUNT
] = {
    "label_map.csv",
    "train_contingency_counts.csv",
    "val_contingency_counts.csv",
    "train_cluster_summary.csv",
    "val_cluster_summary.csv",
    "train_label_summary.csv",
    "val_label_summary.csv",
    "metrics.csv",
    "summary.txt"
};

static const MEProfile *me_profile_from_token(const char *token)
{
    size_t index;
    if (!gs_value_valid(token)) {
        return NULL;
    }
    for (index = 0u;
         index < sizeof(me_profiles) / sizeof(me_profiles[0]);
         index++) {
        if (strcmp(token, me_profiles[index].token) == 0) {
            return &me_profiles[index];
        }
    }
    return NULL;
}

static int me_commit_valid(const char *value)
{
    size_t index;
    size_t length;
    if (!gs_value_valid(value)) {
        return 0;
    }
    length = strlen(value);
    if (strcmp(value, "UNCOMMITTED_STEP11F_TEST") == 0) {
        return 1;
    }
    if (length != 40u) {
        return 0;
    }
    for (index = 0u; index < length; index++) {
        if (!((value[index] >= '0' && value[index] <= '9') ||
              (value[index] >= 'a' && value[index] <= 'f'))) {
            return 0;
        }
    }
    return 1;
}

static void me_normalize_separators(char *path)
{
    char *cursor;
    if (path == NULL) {
        return;
    }
    for (cursor = path; *cursor != '\0'; cursor++) {
        if (*cursor == '\\') {
            *cursor = '/';
        }
    }
}

static int me_paths_equal(const char *left, const char *right)
{
#ifdef _WIN32
    return left != NULL && right != NULL && _stricmp(left, right) == 0;
#else
    return left != NULL && right != NULL && strcmp(left, right) == 0;
#endif
}

#ifdef _WIN32
static int me_has_reparse_component(const char *path)
{
    char probe[GS_PATH_SIZE];
    size_t length;
    size_t index;
    if (path == NULL || strlen(path) >= sizeof(probe)) {
        return 1;
    }
    strcpy(probe, path);
    me_normalize_separators(probe);
    length = strlen(probe);
    for (index = 0u; index <= length; index++) {
        if (probe[index] == '/' || probe[index] == '\0') {
            char saved = probe[index];
            DWORD attributes;
            if (index == 2u && probe[1] == ':') {
                continue;
            }
            probe[index] = '\0';
            if (probe[0] != '\0') {
                attributes = GetFileAttributesA(probe);
                if (attributes == INVALID_FILE_ATTRIBUTES ||
                    (attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0u) {
                    probe[index] = saved;
                    return 1;
                }
            }
            probe[index] = saved;
        }
    }
    return 0;
}
#endif

static int me_canonical_existing(
    const char *path,
    int require_directory,
    char output[GS_PATH_SIZE]
)
{
    struct stat information;
    size_t length;
    if (!gs_value_valid(path) || output == NULL) {
        return 1;
    }
#ifdef _WIN32
    if (_fullpath(output, path, GS_PATH_SIZE) == NULL) {
        return 1;
    }
#else
    if (realpath(path, output) == NULL) {
        return 1;
    }
#endif
    me_normalize_separators(output);
    length = strlen(output);
    while (length > 3u && output[length - 1u] == '/') {
        output[--length] = '\0';
    }
    if (stat(output, &information) != 0 ||
        (require_directory
             ? !S_ISDIR(information.st_mode)
             : !S_ISREG(information.st_mode))) {
        return 1;
    }
#ifdef _WIN32
    if (me_has_reparse_component(output)) {
        return 1;
    }
#endif
    return 0;
}

static int me_regular_non_reparse(const char *path)
{
    char canonical[GS_PATH_SIZE];
    return me_canonical_existing(path, 0, canonical) == 0;
}

static int me_directory_non_reparse(const char *path)
{
    char canonical[GS_PATH_SIZE];
    return me_canonical_existing(path, 1, canonical) == 0;
}

static int me_base_inventory_valid(const char *directory)
{
    unsigned char seen[ME_BASE_FILE_COUNT] = {0};
    DIR *stream = NULL;
    struct dirent *entry;
    int kmeans_seen = 0;
    int evaluation_seen = 0;
    int failed = 0;
    size_t index;

    stream = opendir(directory);
    if (stream == NULL) {
        return 1;
    }
    while (!failed) {
        char path[GS_PATH_SIZE];
        size_t matched;
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
        if (gs_join(directory, entry->d_name, path, sizeof(path)) != 0) {
            failed = 1;
        } else if (gs_name_index(
                       entry->d_name,
                       me_base_files,
                       ME_BASE_FILE_COUNT,
                       &matched
                   )) {
            if (seen[matched] || !me_regular_non_reparse(path)) {
                failed = 1;
            } else {
                seen[matched] = 1u;
            }
        } else if (strcmp(entry->d_name, "kmeans_seed123") == 0 &&
                   !kmeans_seen && me_directory_non_reparse(path)) {
            kmeans_seen = 1;
        } else if (strcmp(
                       entry->d_name, "label_evaluation_seed123"
                   ) == 0 &&
                   !evaluation_seen && me_directory_non_reparse(path)) {
            evaluation_seen = 1;
        } else {
            failed = 1;
        }
    }
    if (closedir(stream) != 0) {
        failed = 1;
    }
    for (index = 0u; !failed && index < ME_BASE_FILE_COUNT; index++) {
        if (!seen[index]) {
            failed = 1;
        }
    }
    return failed;
}

static int me_exact_regular_files(
    const char *directory,
    const char *const *names,
    size_t count
)
{
    unsigned char *seen = NULL;
    DIR *stream = NULL;
    struct dirent *entry;
    size_t index;
    int failed = 0;

    seen = (unsigned char *)calloc(count, 1u);
    stream = opendir(directory);
    if (seen == NULL || stream == NULL) {
        free(seen);
        if (stream != NULL) {
            closedir(stream);
        }
        return 1;
    }
    while (!failed) {
        char path[GS_PATH_SIZE];
        size_t matched;
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
        if (!gs_name_index(entry->d_name, names, count, &matched) ||
            seen[matched] ||
            gs_join(directory, entry->d_name, path, sizeof(path)) != 0 ||
            !me_regular_non_reparse(path)) {
            failed = 1;
        } else {
            seen[matched] = 1u;
        }
    }
    if (closedir(stream) != 0) {
        failed = 1;
    }
    for (index = 0u; !failed && index < count; index++) {
        if (!seen[index]) {
            failed = 1;
        }
    }
    free(seen);
    return failed;
}

static int me_resolve_base(
    const char *manifest_argument,
    const char *base_argument,
    const char *token,
    MEBase *base
)
{
    static const char prefix[] =
        "/sakai_multires_c/results/baseline/matched_subset1000/";
    char suffix[GS_PATH_SIZE];
    char expected[GS_PATH_SIZE];
    char canonical[GS_PATH_SIZE];
    size_t base_length;
    size_t suffix_length;
    size_t root_length;
    int written;

    if (base == NULL) {
        return 1;
    }
    memset(base, 0, sizeof(*base));
    base->profile = me_profile_from_token(token);
    if (base->profile == NULL ||
        me_canonical_existing(base_argument, 1, base->base) != 0) {
        return 1;
    }
    written = snprintf(
        suffix, sizeof(suffix), "%s%s", prefix, base->profile->base_name
    );
    if (written < 0 || (size_t)written >= sizeof(suffix)) {
        return 1;
    }
    base_length = strlen(base->base);
    suffix_length = strlen(suffix);
    if (base_length <= suffix_length) {
        return 1;
    }
    root_length = base_length - suffix_length;
#ifdef _WIN32
    if (_stricmp(base->base + root_length, suffix) != 0) {
#else
    if (strcmp(base->base + root_length, suffix) != 0) {
#endif
        return 1;
    }
    if (root_length == 0u || root_length >= sizeof(base->root)) {
        return 1;
    }
    memcpy(base->root, base->base, root_length);
    base->root[root_length] = '\0';
    if (me_canonical_existing(base->root, 1, canonical) != 0 ||
        !me_paths_equal(base->root, canonical) ||
        gs_join(
            base->root,
            "sakai_multires_c/results/baseline/matched_subset1000",
            expected,
            sizeof(expected)
        ) != 0 ||
        me_canonical_existing(expected, 1, canonical) != 0 ||
        gs_join(
            canonical, base->profile->base_name,
            expected, sizeof(expected)
        ) != 0 ||
        me_canonical_existing(expected, 1, canonical) != 0 ||
        !me_paths_equal(base->base, canonical) ||
        gs_join(
            base->root, "manifest_events_272.csv",
            expected, sizeof(expected)
        ) != 0 ||
        me_canonical_existing(expected, 0, base->manifest) != 0 ||
        me_canonical_existing(manifest_argument, 0, canonical) != 0 ||
        !me_paths_equal(base->manifest, canonical) ||
        gs_join(
            base->root, "sakai_multires_c/results/baseline",
            expected, sizeof(expected)
        ) != 0 ||
        me_canonical_existing(expected, 1, canonical) != 0 ||
        gs_join(
            canonical, base->profile->source_base_name,
            expected, sizeof(expected)
        ) != 0 ||
        me_canonical_existing(expected, 1, base->source) != 0 ||
        gs_join(
            base->root,
            "sakai_multires_c/configs/subset224_train_800_seed123.csv",
            expected, sizeof(expected)
        ) != 0 ||
        me_canonical_existing(expected, 0, base->subset_train) != 0 ||
        gs_join(
            base->root,
            "sakai_multires_c/configs/subset224_val_200_seed123.csv",
            expected, sizeof(expected)
        ) != 0 ||
        me_canonical_existing(expected, 0, base->subset_validation) != 0) {
        return 1;
    }
    return me_base_inventory_valid(base->base);
}

static int me_read_value(
    const char *directory,
    const char *name,
    const char *key,
    char *value,
    size_t capacity
)
{
    char path[GS_PATH_SIZE];
    if (gs_join(directory, name, path, sizeof(path)) != 0) {
        return 1;
    }
    return gs_read_summary_value(path, key, value, capacity);
}

static int me_expect_value(
    const char *directory,
    const char *name,
    const char *key,
    const char *expected
)
{
    char value[256];
    return me_read_value(
        directory, name, key, value, sizeof(value)
    ) == 0 && strcmp(value, expected) == 0 ? 0 : 1;
}

static int me_build_base_entries(
    const MEBase *base,
    GSChecksumEntry entries[18],
    char paths[18][GS_PATH_SIZE]
)
{
    static const char *const source_names[6] = {
        "feature_spec.txt",
        "train_features_raw_f32.bin",
        "val_features_raw_f32.bin",
        "train_rows.csv",
        "val_rows.csv",
        "sha256.txt"
    };
    size_t index;

    if (base == NULL) {
        return 1;
    }
    strcpy(paths[0], base->manifest);
    strcpy(paths[1], base->subset_train);
    strcpy(paths[2], base->subset_validation);
    for (index = 0u; index < 6u; index++) {
        if (gs_join(
                base->source, source_names[index],
                paths[index + 3u], GS_PATH_SIZE
            ) != 0) {
            return 1;
        }
    }
    for (index = 0u; index < ME_BASE_INPUT_COUNT; index++) {
        entries[index].kind = "INPUT";
        entries[index].name = me_base_input_names[index];
        entries[index].path = paths[index];
    }
    for (index = 0u; index < ME_BASE_ARTIFACT_COUNT; index++) {
        if (gs_join(
                base->base, me_base_artifacts[index],
                paths[index + ME_BASE_INPUT_COUNT], GS_PATH_SIZE
            ) != 0) {
            return 1;
        }
        entries[index + ME_BASE_INPUT_COUNT].kind = "ARTIFACT";
        entries[index + ME_BASE_INPUT_COUNT].name =
            me_base_artifacts[index];
        entries[index + ME_BASE_INPUT_COUNT].path =
            paths[index + ME_BASE_INPUT_COUNT];
    }
    return 0;
}

static int me_validate_base_metadata(MEBase *base)
{
    char feature_count[32];
    int written;
    if (base == NULL) {
        return 1;
    }
    written = snprintf(
        feature_count, sizeof(feature_count), "%lu",
        (unsigned long)base->profile->feature_count
    );
    if (written < 0 || (size_t)written >= sizeof(feature_count) ||
        me_expect_value(
            base->base, "feature_spec.txt",
            "profile", base->profile->matched_profile
        ) ||
        me_expect_value(
            base->base, "feature_spec.txt",
            "source_profile", base->profile->source_profile
        ) ||
        me_expect_value(
            base->base, "feature_spec.txt", "matched_subset_profile",
            "subset224_seed123_prefix_800_200"
        ) ||
        me_expect_value(
            base->base, "feature_spec.txt", "provenance",
            "slice_raw_official_full_baseline"
        ) ||
        me_expect_value(
            base->base, "feature_spec.txt", "train_rows", "800"
        ) ||
        me_expect_value(
            base->base, "feature_spec.txt", "validation_rows", "200"
        ) ||
        me_expect_value(
            base->base, "feature_spec.txt",
            "feature_count", feature_count
        ) ||
        me_expect_value(
            base->base, "feature_spec.txt", "standardization",
            "train_only_population_ddof0"
        ) ||
        me_expect_value(
            base->base, "feature_spec.txt",
            "standardization_fit", "train_only"
        ) ||
        me_expect_value(
            base->base, "feature_spec.txt", "standardization_ddof", "0"
        ) ||
        me_expect_value(
            base->base, "feature_spec.txt",
            "scaler_unchanged_after_validation", "true"
        ) ||
        me_expect_value(base->base, "feature_spec.txt", "pca", "false") ||
        me_expect_value(
            base->base, "feature_spec.txt", "augmentation", "false"
        ) ||
        me_expect_value(
            base->base, "feature_spec.txt", "labels_used", "false"
        ) ||
        me_read_value(
            base->base, "feature_spec.txt", "matched_builder_code_commit",
            base->base_commit, sizeof(base->base_commit)
        ) != 0 ||
        strcmp(base->base_commit, ME_MATCHED_BASE_COMMIT) != 0 ||
        me_expect_value(
            base->base, "summary.txt",
            "profile", base->profile->matched_profile
        ) ||
        me_expect_value(
            base->base, "summary.txt",
            "source_profile", base->profile->source_profile
        ) ||
        me_expect_value(
            base->base, "summary.txt", "matched_subset_profile",
            "subset224_seed123_prefix_800_200"
        ) ||
        me_expect_value(
            base->base, "summary.txt", "train_rows", "800"
        ) ||
        me_expect_value(
            base->base, "summary.txt", "validation_rows", "200"
        ) ||
        me_expect_value(
            base->base, "summary.txt", "feature_count", feature_count
        ) ||
        me_expect_value(
            base->base, "summary.txt", "standardization",
            "train_only_population_ddof0"
        ) ||
        me_expect_value(
            base->base, "summary.txt",
            "validation_transformed_without_refit", "true"
        ) ||
        me_expect_value(
            base->base, "summary.txt",
            "scaler_unchanged_after_validation", "true"
        ) ||
        me_expect_value(base->base, "summary.txt", "pca", "false") ||
        me_expect_value(
            base->base, "summary.txt", "augmentation", "false"
        ) ||
        me_expect_value(
            base->base, "summary.txt", "labels_used", "false"
        ) ||
        me_expect_value(
            base->base, "summary.txt", "kmeans_used", "false"
        )) {
        return 1;
    }
    return 0;
}

static int me_validate_base(MEBase *base)
{
    GSChecksumEntry entries[18];
    char paths[18][GS_PATH_SIZE];
    char sha_path[GS_PATH_SIZE];
    size_t train_values;
    size_t validation_values;
    size_t train_bytes;
    size_t validation_bytes;
    size_t expected_sizes[ME_BASE_FILE_COUNT];
    size_t index;

    if (base == NULL ||
        me_base_inventory_valid(base->base) != 0 ||
        gs_checked_product(
            ME_TRAIN_ROWS, base->profile->feature_count, &train_values
        ) != 0 ||
        gs_checked_product(
            ME_VALIDATION_ROWS,
            base->profile->feature_count,
            &validation_values
        ) != 0 ||
        gs_checked_product(
            train_values, sizeof(float), &train_bytes
        ) != 0 ||
        gs_checked_product(
            validation_values, sizeof(float), &validation_bytes
        ) != 0) {
        return 1;
    }
    expected_sizes[0] = 0u;
    expected_sizes[1] = 0u;
    expected_sizes[2] = train_bytes;
    expected_sizes[3] = validation_bytes;
    expected_sizes[4] = train_bytes;
    expected_sizes[5] = validation_bytes;
    expected_sizes[6] = 0u;
    expected_sizes[7] = 0u;
    expected_sizes[8] = 0u;
    expected_sizes[9] = 0u;
    for (index = 0u; index < ME_BASE_FILE_COUNT; index++) {
        char path[GS_PATH_SIZE];
        size_t actual;
        if (gs_join(
                base->base, me_base_files[index],
                path, sizeof(path)
            ) != 0 ||
            gs_file_size(path, &actual) != 0 ||
            (expected_sizes[index] == 0u
                 ? actual == 0u
                 : actual != expected_sizes[index])) {
            return 1;
        }
    }
    if (me_build_base_entries(base, entries, paths) != 0 ||
        gs_join(
            base->base, "sha256.txt", sha_path, sizeof(sha_path)
        ) != 0 ||
        gs_verify_checksum_manifest(sha_path, entries, 18u) != 0 ||
        me_validate_base_metadata(base) != 0) {
        return 1;
    }
    return 0;
}

static int me_validate_kmeans_location(
    const MEBase *base,
    const char *argument,
    size_t k,
    char output[GS_PATH_SIZE]
)
{
    char expected[GS_PATH_SIZE];
    char suffix[64];
    int written;
    if (base == NULL ||
        me_canonical_existing(argument, 1, output) != 0 ||
        (written = snprintf(
             suffix, sizeof(suffix),
             "kmeans_seed123/k%lu", (unsigned long)k
         )) < 0 ||
        (size_t)written >= sizeof(suffix) ||
        gs_join(base->base, suffix, expected, sizeof(expected)) != 0 ||
        !me_paths_equal(output, expected)) {
        return 1;
    }
    return 0;
}

static int me_build_kmeans_entries(
    const MEBase *base,
    const char *directory,
    GSChecksumEntry entries[18],
    char paths[18][GS_PATH_SIZE]
)
{
    size_t index;
    for (index = 0u; index < ME_KMEANS_INPUT_COUNT; index++) {
        if (gs_join(
                base->base, me_kmeans_inputs[index],
                paths[index], GS_PATH_SIZE
            ) != 0) {
            return 1;
        }
        entries[index].kind = "INPUT";
        entries[index].name = me_kmeans_inputs[index];
        entries[index].path = paths[index];
    }
    for (index = 0u; index < ME_KMEANS_ARTIFACT_COUNT; index++) {
        if (gs_join(
                directory, me_kmeans_artifacts[index],
                paths[index + ME_KMEANS_INPUT_COUNT], GS_PATH_SIZE
            ) != 0) {
            return 1;
        }
        entries[index + ME_KMEANS_INPUT_COUNT].kind = "ARTIFACT";
        entries[index + ME_KMEANS_INPUT_COUNT].name =
            me_kmeans_artifacts[index];
        entries[index + ME_KMEANS_INPUT_COUNT].path =
            paths[index + ME_KMEANS_INPUT_COUNT];
    }
    return 0;
}

static int me_validate_kmeans_summary(
    const MEBase *base,
    const char *directory,
    size_t k,
    MEKMeans *metadata
)
{
    char expected_profile[256];
    char k_text[32];
    char feature_count[32];
    int written;

    written = snprintf(
        expected_profile, sizeof(expected_profile),
        "%s_kmeans", base->profile->matched_profile
    );
    if (written < 0 || (size_t)written >= sizeof(expected_profile)) {
        return 1;
    }
    written = snprintf(k_text, sizeof(k_text), "%lu", (unsigned long)k);
    if (written < 0 || (size_t)written >= sizeof(k_text)) {
        return 1;
    }
    written = snprintf(
        feature_count, sizeof(feature_count), "%lu",
        (unsigned long)base->profile->feature_count
    );
    if (written < 0 || (size_t)written >= sizeof(feature_count) ||
        me_expect_value(
            directory, "summary.txt", "profile", expected_profile
        ) ||
        me_expect_value(
            directory, "summary.txt",
            "input_profile", base->profile->matched_profile
        ) ||
        me_expect_value(
            directory, "summary.txt", "matched_subset_profile",
            "subset224_seed123_prefix_800_200"
        ) ||
        me_expect_value(
            directory, "summary.txt",
            "source_profile", base->profile->source_profile
        ) ||
        me_expect_value(
            directory, "summary.txt", "train_rows", "800"
        ) ||
        me_expect_value(
            directory, "summary.txt", "validation_rows", "200"
        ) ||
        me_expect_value(
            directory, "summary.txt", "feature_count", feature_count
        ) ||
        me_expect_value(directory, "summary.txt", "k", k_text) ||
        me_expect_value(directory, "summary.txt", "seed", "123") ||
        me_expect_value(
            directory, "summary.txt", "max_iterations", "300"
        ) ||
        me_expect_value(
            directory, "summary.txt", "fit_input", "train_only"
        ) ||
        me_expect_value(
            directory, "summary.txt",
            "validation_assigned_after_fit", "true"
        ) ||
        me_expect_value(
            directory, "summary.txt", "standardization",
            "train_only_population_ddof0"
        ) ||
        me_expect_value(directory, "summary.txt", "pca", "false") ||
        me_expect_value(
            directory, "summary.txt", "whitening", "false"
        ) ||
        me_expect_value(
            directory, "summary.txt", "augmentation", "false"
        ) ||
        me_expect_value(
            directory, "summary.txt", "labels_used", "false"
        ) ||
        me_expect_value(
            directory, "summary.txt", "matched_base_code_commit",
            base->base_commit
        ) ||
        me_read_value(
            directory, "summary.txt", "code_commit",
            metadata->code_commit, sizeof(metadata->code_commit)
        ) != 0 ||
        strcmp(metadata->code_commit, ME_MATCHED_KMEANS_COMMIT) != 0) {
        return 1;
    }
    metadata->k = k;
    return 0;
}

static int me_validate_kmeans(
    const MEBase *base,
    const char *argument,
    size_t k,
    MEKMeans *metadata
)
{
    GSChecksumEntry entries[18];
    char paths[18][GS_PATH_SIZE];
    char sha_path[GS_PATH_SIZE];
    size_t centroid_values;
    size_t centroid_bytes;
    size_t k_bytes;
    size_t expected_sizes[7];
    size_t index;

    if (base == NULL || metadata == NULL ||
        (k != 22u && k != 36u) ||
        me_validate_kmeans_location(
            base, argument, k, metadata->directory
        ) != 0 ||
        me_exact_regular_files(
            metadata->directory,
            me_kmeans_files,
            ME_KMEANS_FILE_COUNT
        ) != 0 ||
        gs_checked_product(
            k, base->profile->feature_count, &centroid_values
        ) != 0 ||
        gs_checked_product(
            centroid_values, sizeof(double), &centroid_bytes
        ) != 0 ||
        gs_checked_product(k, sizeof(uint64_t), &k_bytes) != 0) {
        return 1;
    }
    expected_sizes[0] = centroid_bytes;
    expected_sizes[1] = ME_TRAIN_ROWS * sizeof(uint32_t);
    expected_sizes[2] = ME_VALIDATION_ROWS * sizeof(uint32_t);
    expected_sizes[3] = ME_TRAIN_ROWS * sizeof(double);
    expected_sizes[4] = ME_VALIDATION_ROWS * sizeof(double);
    expected_sizes[5] = k_bytes;
    expected_sizes[6] = k_bytes;
    for (index = 0u; index < ME_KMEANS_FILE_COUNT; index++) {
        char path[GS_PATH_SIZE];
        size_t actual;
        if (gs_join(
                metadata->directory, me_kmeans_files[index],
                path, sizeof(path)
            ) != 0 ||
            gs_file_size(path, &actual) != 0 ||
            (index < 7u
                 ? actual != expected_sizes[index]
                 : actual == 0u)) {
            return 1;
        }
    }
    if (me_build_kmeans_entries(
            base, metadata->directory, entries, paths
        ) != 0 ||
        gs_join(
            metadata->directory, "sha256.txt",
            sha_path, sizeof(sha_path)
        ) != 0 ||
        gs_verify_checksum_manifest(sha_path, entries, 18u) != 0 ||
        me_validate_kmeans_summary(
            base, metadata->directory, k, metadata
        ) != 0) {
        return 1;
    }
    return 0;
}

static int me_output_paths(
    const MEBase *base,
    size_t k,
    const char *partial_argument,
    char partial[GS_PATH_SIZE],
    char final[GS_PATH_SIZE]
)
{
    char absolute[GS_PATH_SIZE];
    char parent_argument[GS_PATH_SIZE];
    char parent[GS_PATH_SIZE];
    char expected_parent[GS_PATH_SIZE];
    char expected_partial[GS_PATH_SIZE];
    char expected_final[GS_PATH_SIZE];
    char partial_name[32];
    char final_name[16];
    int written;

    if (base == NULL || !gs_value_valid(partial_argument) ||
        (k != 22u && k != 36u)) {
        return 1;
    }
#ifdef _WIN32
    if (_fullpath(absolute, partial_argument, sizeof(absolute)) == NULL) {
        return 1;
    }
#else
    if (partial_argument[0] != '/' ||
        strlen(partial_argument) >= sizeof(absolute)) {
        return 1;
    }
    strcpy(absolute, partial_argument);
#endif
    me_normalize_separators(absolute);
    written = snprintf(
        partial_name, sizeof(partial_name),
        "k%lu.partial", (unsigned long)k
    );
    if (written < 0 || (size_t)written >= sizeof(partial_name)) {
        return 1;
    }
    written = snprintf(
        final_name, sizeof(final_name), "k%lu", (unsigned long)k
    );
    if (written < 0 || (size_t)written >= sizeof(final_name) ||
        strcmp(gs_basename(absolute), partial_name) != 0 ||
        gs_parent_path(
            absolute, parent_argument, sizeof(parent_argument)
        ) != 0 ||
        me_canonical_existing(parent_argument, 1, parent) != 0 ||
        gs_join(
            base->base, "label_evaluation_seed123",
            expected_parent, sizeof(expected_parent)
        ) != 0 ||
        !me_paths_equal(parent, expected_parent) ||
        gs_join(
            parent, partial_name, expected_partial,
            sizeof(expected_partial)
        ) != 0 ||
        gs_join(
            parent, final_name, expected_final,
            sizeof(expected_final)
        ) != 0 ||
        !me_paths_equal(absolute, expected_partial) ||
        gs_path_state(expected_partial, NULL, NULL) != 0 ||
        gs_path_state(expected_final, NULL, NULL) != 0) {
        return 1;
    }
    strcpy(partial, expected_partial);
    strcpy(final, expected_final);
    return 0;
}

static char *me_unquote_csv_field(char *field)
{
    size_t length;
    if (field == NULL) {
        return NULL;
    }
    length = strlen(field);
    if (length >= 2u && field[0] == '"' && field[length - 1u] == '"') {
        field[length - 1u] = '\0';
        return field + 1;
    }
    return field;
}

static int me_validate_prelabel_row_file(
    const char *row_path,
    const char *subset_path,
    const char *split,
    size_t rows,
    unsigned char coverage[GS_EVENT_COUNT]
)
{
    static const char row_header[] =
        "row_index,split_pos,manifest_index,event_index,event_id,split";
    static const char subset_header[] =
        "split_name,split_pos,manifest_index,event_index,event_id,tensor_path";
    FILE *row_file = NULL;
    FILE *subset_file = NULL;
    char row_line[GS_LINE_SIZE];
    char subset_line[GS_LINE_SIZE];
    size_t row;
    int failed = 0;

    row_file = fopen(row_path, "rb");
    subset_file = fopen(subset_path, "rb");
    if (row_file == NULL || subset_file == NULL ||
        fgets(row_line, sizeof(row_line), row_file) == NULL ||
        fgets(subset_line, sizeof(subset_line), subset_file) == NULL) {
        failed = 1;
    } else {
        gs_strip_eol(row_line);
        gs_strip_eol(subset_line);
        if (strcmp(row_line, row_header) != 0 ||
            strcmp(subset_line, subset_header) != 0) {
            failed = 1;
        }
    }
    for (row = 0u; !failed && row < rows; row++) {
        char *row_fields[6];
        char *subset_fields[6];
        char *subset_event_id;
        size_t row_index;
        size_t row_split_pos;
        size_t row_manifest_index;
        size_t row_event_index;
        size_t subset_split_pos;
        size_t subset_manifest_index;
        size_t subset_event_index;

        if (fgets(row_line, sizeof(row_line), row_file) == NULL ||
            fgets(subset_line, sizeof(subset_line), subset_file) == NULL ||
            gs_split_csv(row_line, row_fields, 6u) != 0 ||
            gs_split_csv(subset_line, subset_fields, 6u) != 0 ||
            gs_parse_size(row_fields[0], &row_index) != 0 ||
            gs_parse_size(row_fields[1], &row_split_pos) != 0 ||
            gs_parse_size(row_fields[2], &row_manifest_index) != 0 ||
            gs_parse_size(row_fields[3], &row_event_index) != 0 ||
            gs_parse_size(subset_fields[1], &subset_split_pos) != 0 ||
            gs_parse_size(subset_fields[2], &subset_manifest_index) != 0 ||
            gs_parse_size(subset_fields[3], &subset_event_index) != 0) {
            failed = 1;
            break;
        }
        subset_event_id = me_unquote_csv_field(subset_fields[4]);
        if (subset_event_id == NULL ||
            row_index != row || row_split_pos != row ||
            subset_split_pos != row ||
            row_manifest_index >= GS_EVENT_COUNT ||
            coverage[row_manifest_index] ||
            row_manifest_index != subset_manifest_index ||
            row_event_index != subset_event_index ||
            strcmp(row_fields[4], subset_event_id) != 0 ||
            strcmp(row_fields[5], split) != 0 ||
            strcmp(subset_fields[0], split) != 0 ||
            !gs_value_valid(row_fields[4]) ||
            !gs_value_valid(subset_fields[5])) {
            failed = 1;
            break;
        }
        coverage[row_manifest_index] = 1u;
    }
    if (!failed &&
        (fgets(row_line, sizeof(row_line), row_file) != NULL ||
         fgets(subset_line, sizeof(subset_line), subset_file) != NULL)) {
        failed = 1;
    }
    if (row_file != NULL) {
        if (ferror(row_file)) {
            failed = 1;
        }
        if (fclose(row_file) != 0) {
            failed = 1;
        }
    }
    if (subset_file != NULL) {
        if (ferror(subset_file)) {
            failed = 1;
        }
        if (fclose(subset_file) != 0) {
            failed = 1;
        }
    }
    return failed;
}

static int me_validate_prelabel_rows(
    const MEBase *base,
    const char *train_rows_path,
    const char *validation_rows_path,
    unsigned char coverage[GS_EVENT_COUNT]
)
{
    size_t index;
    size_t covered = 0u;
    if (base == NULL || train_rows_path == NULL ||
        validation_rows_path == NULL || coverage == NULL ||
        me_validate_prelabel_row_file(
            train_rows_path, base->subset_train, "train",
            ME_TRAIN_ROWS, coverage
        ) != 0 ||
        me_validate_prelabel_row_file(
            validation_rows_path, base->subset_validation, "validation",
            ME_VALIDATION_ROWS, coverage
        ) != 0) {
        return 1;
    }
    for (index = 0u; index < GS_EVENT_COUNT; index++) {
        if (coverage[index] == 1u) {
            covered++;
        } else if (coverage[index] != 0u) {
            return 1;
        }
    }
    return covered == ME_SUBSET_ROWS ? 0 : 1;
}

static int me_write_summary(
    const char *directory,
    const MEBase *base,
    const MEKMeans *kmeans,
    const char *code_commit,
    const GSBaselineClusteringMetrics *train,
    const GSBaselineClusteringMetrics *validation
)
{
    FILE *file = NULL;
    char profile[256];
    char kmeans_profile[256];
    int written;
    int failed;

    written = snprintf(
        profile, sizeof(profile),
        "%s_label_evaluation", base->profile->matched_profile
    );
    if (written < 0 || (size_t)written >= sizeof(profile)) {
        return 1;
    }
    written = snprintf(
        kmeans_profile, sizeof(kmeans_profile),
        "%s_kmeans", base->profile->matched_profile
    );
    if (written < 0 || (size_t)written >= sizeof(kmeans_profile) ||
        gs_open_output(directory, "summary.txt", &file) != 0) {
        return 1;
    }
    failed = fprintf(
        file,
        "format_version=1\n"
        "profile=%s\n"
        "input_profile=%s\n"
        "matched_subset_profile=subset224_seed123_prefix_800_200\n"
        "source_profile=%s\n"
        "feature_count=%lu\n"
        "standardization=train_only_population_ddof0\n"
        "kmeans_profile=%s\n"
        "evaluation=labels_after_clustering\n"
        "labels_policy=posterior_evaluation_only\n"
        "label_count=22\n"
        "class_count=22\n"
        "event_count=8583\n"
        "subset_rows=1000\n"
        "cluster_count=%lu\n"
        "k=%lu\n"
        "seed=123\n"
        "train_rows=800\n"
        "validation_rows=200\n"
        "labels_used_for_feature_extraction=false\n"
        "labels_used_for_standardization=false\n"
        "labels_used_for_pca=false\n"
        "labels_used_for_kmeans=false\n"
        "labels_used_for_evaluation=true\n"
        "pca=false\n"
        "augmentation=false\n"
        "nmi_normalization=arithmetic\n"
        "purity_definition=weighted_cluster_majority\n"
        "matched_base_code_commit=%s\n"
        "kmeans_code_commit=%s\n"
        "evaluator_code_commit=%s\n"
        "code_commit=%s\n"
        "train_purity=%.17g\n"
        "train_nmi_arithmetic=%.17g\n"
        "train_ari=%.17g\n"
        "train_homogeneity=%.17g\n"
        "train_completeness=%.17g\n"
        "validation_purity=%.17g\n"
        "validation_nmi_arithmetic=%.17g\n"
        "validation_ari=%.17g\n"
        "validation_homogeneity=%.17g\n"
        "validation_completeness=%.17g\n",
        profile,
        base->profile->matched_profile,
        base->profile->source_profile,
        (unsigned long)base->profile->feature_count,
        kmeans_profile,
        (unsigned long)kmeans->k,
        (unsigned long)kmeans->k,
        base->base_commit,
        kmeans->code_commit,
        code_commit,
        code_commit,
        train->purity,
        train->normalized_mutual_information,
        train->adjusted_rand_index,
        train->homogeneity,
        train->completeness,
        validation->purity,
        validation->normalized_mutual_information,
        validation->adjusted_rand_index,
        validation->homogeneity,
        validation->completeness
    ) < 0;
    return gs_finish_output(file, failed);
}

static int me_write_nine(
    const char *directory,
    const MEBase *base,
    const MEKMeans *kmeans,
    const char *code_commit,
    const GSBaselineClusteringMetrics *train,
    const GSBaselineClusteringMetrics *validation
)
{
    return gs_write_label_map(directory) ||
        gs_write_contingency(
            directory, "train_contingency_counts.csv", train
        ) ||
        gs_write_contingency(
            directory, "val_contingency_counts.csv", validation
        ) ||
        gs_write_cluster_summary(
            directory, "train_cluster_summary.csv", train
        ) ||
        gs_write_cluster_summary(
            directory, "val_cluster_summary.csv", validation
        ) ||
        gs_write_label_summary(
            directory, "train_label_summary.csv", train
        ) ||
        gs_write_label_summary(
            directory, "val_label_summary.csv", validation
        ) ||
        gs_write_metrics(
            directory, kmeans->k, train, validation
        ) ||
        me_write_summary(
            directory, base, kmeans, code_commit, train, validation
        );
}

static int me_build_evaluation_entries(
    const MEBase *base,
    const MEKMeans *kmeans,
    const char *output,
    GSChecksumEntry entries[14],
    char paths[ME_EVALUATION_PATH_COUNT][GS_PATH_SIZE]
)
{
    size_t index;
    if (gs_join(
            base->base, "train_rows.csv", paths[0], GS_PATH_SIZE
        ) != 0 ||
        gs_join(
            base->base, "val_rows.csv", paths[1], GS_PATH_SIZE
        ) != 0 ||
        gs_join(
            kmeans->directory, "train_assignments_u32.bin",
            paths[2], GS_PATH_SIZE
        ) != 0 ||
        gs_join(
            kmeans->directory, "val_assignments_u32.bin",
            paths[3], GS_PATH_SIZE
        ) != 0) {
        return 1;
    }
    entries[0].kind = "INPUT";
    entries[0].name = me_evaluation_inputs[0];
    entries[0].path = base->manifest;
    for (index = 1u; index < ME_EVALUATION_INPUT_COUNT; index++) {
        entries[index].kind = "INPUT";
        entries[index].name = me_evaluation_inputs[index];
        entries[index].path = paths[index - 1u];
    }
    for (index = 0u; index < ME_EVALUATION_ARTIFACT_COUNT; index++) {
        if (gs_join(
                output, me_evaluation_artifacts[index],
                paths[index + ME_EVALUATION_INPUT_COUNT - 1u],
                GS_PATH_SIZE
            ) != 0) {
            return 1;
        }
        entries[index + ME_EVALUATION_INPUT_COUNT].kind = "ARTIFACT";
        entries[index + ME_EVALUATION_INPUT_COUNT].name =
            me_evaluation_artifacts[index];
        entries[index + ME_EVALUATION_INPUT_COUNT].path =
            paths[index + ME_EVALUATION_INPUT_COUNT - 1u];
    }
    return 0;
}

static int me_write_sha256(
    const MEBase *base,
    const MEKMeans *kmeans,
    const char *output
)
{
    GSChecksumEntry entries[14];
    char paths[ME_EVALUATION_PATH_COUNT][GS_PATH_SIZE];
    FILE *file = NULL;
    size_t index;
    int failed = 0;

    if (me_build_evaluation_entries(
            base, kmeans, output, entries, paths
        ) != 0 ||
        gs_open_output(output, "sha256.txt", &file) != 0) {
        return 1;
    }
    for (index = 0u; !failed && index < 14u; index++) {
        char hash[65];
        if (gs_sha_file(entries[index].path, hash) != 0 ||
            fprintf(
                file, "%s %s %s\n",
                entries[index].kind,
                hash,
                entries[index].name
            ) < 0) {
            failed = 1;
        }
    }
    return gs_finish_output(file, failed);
}

static int me_validate_written_output(
    const MEBase *base,
    const MEKMeans *kmeans,
    const char *output
)
{
    GSChecksumEntry entries[14];
    char paths[ME_EVALUATION_PATH_COUNT][GS_PATH_SIZE];
    char sha_path[GS_PATH_SIZE];
    size_t index;

    if (me_exact_regular_files(
            output, me_output_files, ME_EVALUATION_FILE_COUNT
        ) != 0) {
        return 1;
    }
    for (index = 0u; index < ME_EVALUATION_FILE_COUNT; index++) {
        char path[GS_PATH_SIZE];
        size_t size;
        if (gs_join(
                output, me_output_files[index], path, sizeof(path)
            ) != 0 ||
            gs_file_size(path, &size) != 0 || size == 0u) {
            return 1;
        }
    }
    return me_build_evaluation_entries(
               base, kmeans, output, entries, paths
           ) != 0 ||
           gs_join(
               output, "sha256.txt", sha_path, sizeof(sha_path)
           ) != 0 ||
           gs_verify_checksum_manifest(sha_path, entries, 14u) != 0;
}

static int me_failure_hook_enabled(const char *partial)
{
    const char *value = getenv(ME_FAILURE_ENV);
    return partial != NULL && value != NULL && strcmp(value, "1") == 0 &&
           (strcmp(gs_basename(partial), "k22.partial") == 0 ||
            strcmp(gs_basename(partial), "k36.partial") == 0);
}

static int me_cleanup_partial(const char *partial)
{
    size_t index;
    int failed = 0;
    for (index = 0u; index < ME_EVALUATION_FILE_COUNT; index++) {
        char path[GS_PATH_SIZE];
        int state;
        if (gs_join(
                partial, me_output_files[index], path, sizeof(path)
            ) != 0) {
            failed = 1;
            continue;
        }
        state = gs_path_state(path, NULL, NULL);
        if (state < 0 || (state == 1 && remove(path) != 0)) {
            failed = 1;
        }
    }
    if (GS_RMDIR(partial) != 0) {
        failed = 1;
    }
    return failed;
}

#ifndef GS_MATCHED_EVALUATION_NO_MAIN
int main(int argc, char **argv)
{
    MEBase base;
    MEKMeans kmeans;
    GSEventManifest manifest;
    GSBaselineClusteringMetrics train_metrics;
    GSBaselineClusteringMetrics validation_metrics;
    unsigned char *coverage = NULL;
    uint32_t *train_labels = NULL;
    uint32_t *validation_labels = NULL;
    uint32_t *train_assignments = NULL;
    uint32_t *validation_assignments = NULL;
    char train_rows_path[GS_PATH_SIZE];
    char validation_rows_path[GS_PATH_SIZE];
    char train_assignments_path[GS_PATH_SIZE];
    char validation_assignments_path[GS_PATH_SIZE];
    char partial_path[GS_PATH_SIZE];
    char final_path[GS_PATH_SIZE];
    char train_hash_before[65];
    char validation_hash_before[65];
    char train_hash_after[65];
    char validation_hash_after[65];
    size_t train_bytes = 0u;
    size_t validation_bytes = 0u;
    size_t k = 0u;
    size_t index = 0u;
    int partial_created = 0;
    int rc = 1;

    memset(&base, 0, sizeof(base));
    memset(&kmeans, 0, sizeof(kmeans));
    memset(&manifest, 0, sizeof(manifest));
    memset(&train_metrics, 0, sizeof(train_metrics));
    memset(&validation_metrics, 0, sizeof(validation_metrics));

    if (argc != 8 || !gs_platform_valid() ||
        setlocale(LC_NUMERIC, "C") == NULL ||
        gs_parse_size(argv[5], &k) != 0 ||
        (k != 22u && k != 36u) ||
        !me_commit_valid(argv[7]) ||
        gs_checked_product(
            ME_TRAIN_ROWS, sizeof(uint32_t), &train_bytes
        ) != 0 ||
        gs_checked_product(
            ME_VALIDATION_ROWS, sizeof(uint32_t), &validation_bytes
        ) != 0) {
        fprintf(
            stderr,
            "Usage: evaluate_matched_subset1000_clusters.exe "
            "<manifest_events> <matched_base_directory> "
            "<mini75_pool5|mini112_pool7> <kmeans_directory> "
            "<22|36> <output_partial_directory> <code_commit>\n"
        );
        return 1;
    }

    if (me_resolve_base(argv[1], argv[2], argv[3], &base) != 0 ||
        me_validate_base(&base) != 0 ||
        me_validate_kmeans(&base, argv[4], k, &kmeans) != 0 ||
        me_output_paths(
            &base, k, argv[6], partial_path, final_path
        ) != 0 ||
        gs_join(
            base.base, "train_rows.csv",
            train_rows_path, sizeof(train_rows_path)
        ) != 0 ||
        gs_join(
            base.base, "val_rows.csv",
            validation_rows_path, sizeof(validation_rows_path)
        ) != 0 ||
        gs_join(
            kmeans.directory, "train_assignments_u32.bin",
            train_assignments_path, sizeof(train_assignments_path)
        ) != 0 ||
        gs_join(
            kmeans.directory, "val_assignments_u32.bin",
            validation_assignments_path,
            sizeof(validation_assignments_path)
        ) != 0) {
        fprintf(stderr, "ERROR: invalid canonical input contract.\n");
        goto cleanup;
    }
    printf("[PASS] canonical_base_and_kmeans_contract\n");

    train_assignments = (uint32_t *)malloc(train_bytes);
    validation_assignments = (uint32_t *)malloc(validation_bytes);
    if (train_assignments == NULL || validation_assignments == NULL ||
        gs_read_assignments(
            train_assignments_path, train_assignments,
            ME_TRAIN_ROWS, k
        ) != 0 ||
        gs_read_assignments(
            validation_assignments_path, validation_assignments,
            ME_VALIDATION_ROWS, k
        ) != 0 ||
        gs_sha_file(
            train_assignments_path, train_hash_before
        ) != 0 ||
        gs_sha_file(
            validation_assignments_path, validation_hash_before
        ) != 0) {
        fprintf(stderr, "ERROR: invalid assignment inputs.\n");
        goto cleanup;
    }
    printf("[PASS] frozen_assignment_inputs\n");

    coverage = (unsigned char *)calloc(GS_EVENT_COUNT, 1u);
    if (coverage == NULL ||
        me_validate_prelabel_rows(
            &base, train_rows_path, validation_rows_path, coverage
        ) != 0) {
        fprintf(stderr, "ERROR: invalid pre-label row mapping.\n");
        goto cleanup;
    }
    printf("[PASS] prelabel_row_mapping_and_coverage\n");
    memset(coverage, 0, GS_EVENT_COUNT);
    train_labels = (uint32_t *)malloc(train_bytes);
    validation_labels = (uint32_t *)malloc(validation_bytes);
    if (train_labels == NULL || validation_labels == NULL ||
        gs_event_manifest_load(base.manifest, &manifest) != 0 ||
        gs_validate_manifest(&manifest) != 0 ||
        gs_load_rows(
            train_rows_path, "train", ME_TRAIN_ROWS,
            &manifest, coverage, train_labels
        ) != 0 ||
        gs_load_rows(
            validation_rows_path, "validation", ME_VALIDATION_ROWS,
            &manifest, coverage, validation_labels
        ) != 0) {
        fprintf(stderr, "ERROR: invalid manifest or row mapping.\n");
        goto cleanup;
    }
    {
        size_t covered = 0u;
        for (index = 0u; index < GS_EVENT_COUNT; index++) {
            if (coverage[index] == 1u) {
                covered++;
            } else if (coverage[index] != 0u) {
                goto cleanup;
            }
        }
        if (covered != ME_SUBSET_ROWS) {
            fprintf(stderr, "ERROR: invalid subset coverage.\n");
            goto cleanup;
        }
    }
    if (gs_sha_file(train_assignments_path, train_hash_after) != 0 ||
        gs_sha_file(
            validation_assignments_path, validation_hash_after
        ) != 0 ||
        strcmp(train_hash_before, train_hash_after) != 0 ||
        strcmp(validation_hash_before, validation_hash_after) != 0) {
        fprintf(stderr, "ERROR: assignments changed during label load.\n");
        goto cleanup;
    }
    printf("[PASS] posterior_label_mapping\n");

    if (gs_baseline_clustering_metrics_compute(
            train_labels, train_assignments, ME_TRAIN_ROWS,
            GS_LABEL_COUNT, k, &train_metrics
        ) != GS_BASELINE_CLUSTERING_METRICS_SUCCESS ||
        gs_baseline_clustering_metrics_compute(
            validation_labels, validation_assignments,
            ME_VALIDATION_ROWS, GS_LABEL_COUNT, k,
            &validation_metrics
        ) != GS_BASELINE_CLUSTERING_METRICS_SUCCESS ||
        !gs_metrics_valid(&train_metrics, ME_TRAIN_ROWS, k) ||
        !gs_metrics_valid(
            &validation_metrics, ME_VALIDATION_ROWS, k
        )) {
        fprintf(stderr, "ERROR: invalid posterior metrics.\n");
        goto cleanup;
    }
    printf("[PASS] posterior_train_and_validation_metrics\n");

    if (GS_MKDIR(partial_path) != 0) {
        fprintf(stderr, "ERROR: partial creation failed.\n");
        goto cleanup;
    }
    partial_created = 1;
    if (me_failure_hook_enabled(partial_path)) {
        fprintf(stderr, "ERROR: confined failure hook requested.\n");
        goto cleanup;
    }
    if (me_write_nine(
            partial_path, &base, &kmeans, argv[7],
            &train_metrics, &validation_metrics
        ) != 0 ||
        me_write_sha256(&base, &kmeans, partial_path) != 0 ||
        me_validate_written_output(
            &base, &kmeans, partial_path
        ) != 0 ||
        gs_sha_file(train_assignments_path, train_hash_after) != 0 ||
        gs_sha_file(
            validation_assignments_path, validation_hash_after
        ) != 0 ||
        strcmp(train_hash_before, train_hash_after) != 0 ||
        strcmp(validation_hash_before, validation_hash_after) != 0) {
        fprintf(stderr, "ERROR: output validation failed.\n");
        goto cleanup;
    }
    printf("[PASS] exact_inventory_and_checksums\n");
    if (rename(partial_path, final_path) != 0) {
        fprintf(stderr, "ERROR: atomic promotion failed.\n");
        goto cleanup;
    }
    partial_created = 0;

    printf("TOKEN=%s\n", base.profile->token);
    printf("FEATURE_COUNT=%lu\n",
           (unsigned long)base.profile->feature_count);
    printf("K=%lu\n", (unsigned long)k);
    printf("TRAIN_PURITY=%.17g\n", train_metrics.purity);
    printf(
        "TRAIN_NMI_ARITHMETIC=%.17g\n",
        train_metrics.normalized_mutual_information
    );
    printf("TRAIN_ARI=%.17g\n", train_metrics.adjusted_rand_index);
    printf("TRAIN_HOMOGENEITY=%.17g\n", train_metrics.homogeneity);
    printf("TRAIN_COMPLETENESS=%.17g\n", train_metrics.completeness);
    printf("VALIDATION_PURITY=%.17g\n", validation_metrics.purity);
    printf(
        "VALIDATION_NMI_ARITHMETIC=%.17g\n",
        validation_metrics.normalized_mutual_information
    );
    printf(
        "VALIDATION_ARI=%.17g\n",
        validation_metrics.adjusted_rand_index
    );
    printf(
        "VALIDATION_HOMOGENEITY=%.17g\n",
        validation_metrics.homogeneity
    );
    printf(
        "VALIDATION_COMPLETENESS=%.17g\n",
        validation_metrics.completeness
    );
    printf("MATCHED SUBSET1000 LABEL EVALUATION COMPLETED\n");
    rc = 0;

cleanup:
    if (partial_created && me_cleanup_partial(partial_path) != 0) {
        fprintf(stderr, "ERROR: partial cleanup failed.\n");
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
#endif
