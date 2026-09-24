#define _CRT_SECURE_NO_WARNINGS

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#define GS_MATCHED_SUBSET1000_NO_MAIN
#include "build_matched_subset1000_no_pca_artifacts.c"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include "baseline_kmeans.h"

#include <locale.h>

enum {
    KM_BASE_INPUT_COUNT = 9,
    KM_BASE_ARTIFACT_COUNT = 9,
    KM_BASE_FILE_COUNT = 10,
    KM_INPUT_COUNT = 6,
    KM_ARTIFACT_COUNT = 12,
    KM_OUTPUT_COUNT = 13,
    KM_SEED = 123,
    KM_MAX_ITERATIONS = 300
};

#define KM_MATCHED_BASE_COMMIT \
    "6e76182f40f4008b8fdd8b33a9dacc4a20241384"
#define KM_TEST_COMMIT "UNCOMMITTED_STEP11D_TEST"
#define KM_FAILURE_ENV \
    "GS_MATCHED_SUBSET1000_KMEANS_TEST_FAIL_AFTER_PARTIAL"

typedef struct {
    const GSProfile *profile;
    char root[GS_PATH_CAPACITY];
    char base[GS_PATH_CAPACITY];
    char source[GS_PATH_CAPACITY];
    char subset_train[GS_PATH_CAPACITY];
    char subset_validation[GS_PATH_CAPACITY];
    char input_storage[KM_INPUT_COUNT][GS_PATH_CAPACITY];
    const char *input_paths[KM_INPUT_COUNT];
    char input_hashes[KM_INPUT_COUNT][65];
    char matched_base_commit[64];
    size_t train_rows;
    size_t validation_rows;
    size_t feature_count;
    float *train;
    float *validation;
} KMBase;

typedef struct {
    double initial_inertia;
    double train_inertia;
    double validation_inertia;
    size_t empty_clusters;
    size_t minimum_cluster_size;
    size_t maximum_cluster_size;
} KMNumericSummary;

static const char *const km_base_names[KM_BASE_FILE_COUNT] = {
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

static const char *const km_input_names[KM_INPUT_COUNT] = {
    "feature_spec.txt",
    "scaler.csv",
    "train_features_std_f32.bin",
    "val_features_std_f32.bin",
    "train_rows.csv",
    "val_rows.csv"
};

static const char *const km_artifact_names[KM_ARTIFACT_COUNT] = {
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

static const char *const km_output_names[KM_OUTPUT_COUNT] = {
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

static void km_base_free(KMBase *base)
{
    if (base == NULL) {
        return;
    }
    free(base->validation);
    free(base->train);
    memset(base, 0, sizeof(*base));
}

static int km_commit_valid(const char *value)
{
    size_t index;

    if (!gs_text_value_valid(value)) {
        return 0;
    }
    if (strcmp(value, KM_TEST_COMMIT) == 0) {
        return 1;
    }
    if (strlen(value) != 40u) {
        return 0;
    }
    for (index = 0u; index < 40u; index++) {
        if (!((value[index] >= '0' && value[index] <= '9') ||
              (value[index] >= 'a' && value[index] <= 'f'))) {
            return 0;
        }
    }
    return 1;
}

static int km_parse_size(const char *text, size_t *value)
{
    char *end = NULL;
    uintmax_t parsed;

    if (text == NULL || value == NULL || text[0] == '\0' ||
        text[0] == '-') {
        return 1;
    }
    errno = 0;
    parsed = strtoumax(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' ||
        parsed > (uintmax_t)SIZE_MAX) {
        return 1;
    }
    *value = (size_t)parsed;
    return 0;
}

static int km_parse_u64(const char *text, uint64_t *value)
{
    char *end = NULL;
    uintmax_t parsed;

    if (text == NULL || value == NULL || text[0] == '\0' ||
        text[0] == '-') {
        return 1;
    }
    errno = 0;
    parsed = strtoumax(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' ||
        parsed > UINT64_MAX) {
        return 1;
    }
    *value = (uint64_t)parsed;
    return 0;
}

static int km_platform_valid(void)
{
    uint16_t endian = UINT16_C(1);
    return CHAR_BIT == 8 &&
        sizeof(float) == 4u && FLT_RADIX == 2 && FLT_MANT_DIG == 24 &&
        sizeof(double) == 8u && DBL_MANT_DIG == 53 &&
        sizeof(uint32_t) == 4u && sizeof(uint64_t) == 8u &&
        *(const unsigned char *)&endian == 1u;
}

static int km_file_value(
    const char *path,
    const char *key,
    char *value,
    size_t capacity
)
{
    char line[GS_LINE_CAPACITY];
    size_t key_length;
    FILE *file;
    int found = 0;

    if (path == NULL || key == NULL || value == NULL || capacity == 0u ||
        (file = fopen(path, "rb")) == NULL) {
        return 1;
    }
    key_length = strlen(key);
    while (fgets(line, sizeof(line), file) != NULL) {
        size_t length;
        gs_chomp(line);
        if (strncmp(line, key, key_length) != 0 ||
            line[key_length] != '=') {
            continue;
        }
        if (found) {
            fclose(file);
            return 1;
        }
        length = strlen(line + key_length + 1u);
        if (length == 0u || length >= capacity) {
            fclose(file);
            return 1;
        }
        memcpy(value, line + key_length + 1u, length + 1u);
        found = 1;
    }
    {
        int read_failed = ferror(file);
        int close_failed = fclose(file) != 0;
        if (read_failed || close_failed || !found ||
            !gs_text_value_valid(value)) {
            return 1;
        }
    }
    return 0;
}

static int km_name_index(
    const char *name,
    const char *const *names,
    size_t count,
    size_t *index
)
{
    size_t current;
    for (current = 0u; current < count; current++) {
        if (strcmp(name, names[current]) == 0) {
            if (index != NULL) {
                *index = current;
            }
            return 1;
        }
    }
    return 0;
}

static int km_base_inventory_valid(const char *directory)
{
    unsigned char seen[KM_BASE_FILE_COUNT] = {0};
    DIR *stream;
    struct dirent *entry;
    size_t index;
    int downstream_seen = 0;

    stream = opendir(directory);
    if (stream == NULL) {
        return 1;
    }
    while ((entry = readdir(stream)) != NULL) {
        char path[GS_PATH_CAPACITY];
        struct stat information;
        size_t matched;
        if (strcmp(entry->d_name, ".") == 0 ||
            strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        if (gs_path_join(directory, entry->d_name, path, sizeof(path)) != 0 ||
            stat(path, &information) != 0) {
            closedir(stream);
            return 1;
        }
#ifdef _WIN32
        {
            DWORD attributes = GetFileAttributesA(path);
            if (attributes == INVALID_FILE_ATTRIBUTES ||
                (attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0u) {
                closedir(stream);
                return 1;
            }
        }
#endif
        if (km_name_index(
                entry->d_name, km_base_names, KM_BASE_FILE_COUNT, &matched
            )) {
            if (!S_ISREG(information.st_mode) || seen[matched] != 0u) {
                closedir(stream);
                return 1;
            }
            seen[matched] = 1u;
        } else if (strcmp(entry->d_name, "kmeans_seed123") == 0) {
            if (!S_ISDIR(information.st_mode) || downstream_seen) {
                closedir(stream);
                return 1;
            }
            downstream_seen = 1;
        } else {
            closedir(stream);
            return 1;
        }
    }
    if (closedir(stream) != 0) {
        return 1;
    }
    for (index = 0u; index < KM_BASE_FILE_COUNT; index++) {
        if (seen[index] == 0u) {
            return 1;
        }
    }
    return 0;
}

static int km_resolve_base(
    const char *argument,
    const char *token,
    KMBase *base
)
{
    static const char prefix[] =
        "/sakai_multires_c/results/baseline/matched_subset1000/";
    char suffix[GS_PATH_CAPACITY];
    char expected[GS_PATH_CAPACITY];
    char canonical[GS_PATH_CAPACITY];
    size_t base_length;
    size_t suffix_length;
    size_t root_length;
    int written;

    if (base == NULL || (base->profile = gs_profile_from_token(token)) == NULL ||
        gs_canonical_existing(argument, 1, base->base) != 0) {
        return 1;
    }
    written = snprintf(
        suffix, sizeof(suffix), "%s%s",
        prefix, base->profile->matched_basename
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
    if (gs_canonical_existing(base->root, 1, canonical) != 0 ||
        !gs_paths_equal(base->root, canonical) ||
        gs_path_join(
            base->root,
            "sakai_multires_c/results/baseline/matched_subset1000",
            expected,
            sizeof(expected)
        ) != 0 ||
        gs_canonical_existing(expected, 1, canonical) != 0 ||
        gs_path_join(
            canonical, base->profile->matched_basename,
            expected, sizeof(expected)
        ) != 0 ||
        gs_canonical_existing(expected, 1, canonical) != 0 ||
        !gs_paths_equal(base->base, canonical) ||
        gs_path_join(
            base->root, "sakai_multires_c/results/baseline",
            expected, sizeof(expected)
        ) != 0 ||
        gs_canonical_existing(expected, 1, canonical) != 0 ||
        gs_path_join(
            canonical, base->profile->source_basename,
            expected, sizeof(expected)
        ) != 0 ||
        gs_canonical_existing(expected, 1, base->source) != 0 ||
        gs_path_join(
            base->root,
            "sakai_multires_c/configs/subset224_train_800_seed123.csv",
            expected, sizeof(expected)
        ) != 0 ||
        gs_canonical_existing(expected, 0, base->subset_train) != 0 ||
        gs_path_join(
            base->root,
            "sakai_multires_c/configs/subset224_val_200_seed123.csv",
            expected, sizeof(expected)
        ) != 0 ||
        gs_canonical_existing(expected, 0, base->subset_validation) != 0) {
        return 1;
    }
    return km_base_inventory_valid(base->base);
}

static int km_build_base_checksum_inputs(
    const KMBase *base,
    char storage[KM_BASE_INPUT_COUNT][GS_PATH_CAPACITY],
    const char *paths[KM_BASE_INPUT_COUNT]
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
    char path[GS_PATH_CAPACITY];
    size_t index;

    if (gs_path_join(
            base->root, "manifest_events_272.csv",
            storage[0], GS_PATH_CAPACITY
        ) != 0) {
        return 1;
    }
    strcpy(storage[1], base->subset_train);
    strcpy(storage[2], base->subset_validation);
    for (index = 0u; index < 6u; index++) {
        if (gs_path_join(
                base->source, source_names[index],
                storage[index + 3u], GS_PATH_CAPACITY
            ) != 0) {
            return 1;
        }
    }
    for (index = 0u; index < KM_BASE_INPUT_COUNT; index++) {
        if (gs_canonical_existing(storage[index], 0, path) != 0 ||
            !gs_paths_equal(storage[index], path)) {
            return 1;
        }
        paths[index] = storage[index];
    }
    return 0;
}

static int km_parse_checksum_line(
    char *line,
    const char *kind,
    const char *expected_name,
    char hash[65]
)
{
    char *fields[3];
    size_t index;

    gs_chomp(line);
    fields[0] = strtok(line, " ");
    fields[1] = strtok(NULL, " ");
    fields[2] = strtok(NULL, " ");
    if (fields[0] == NULL || fields[1] == NULL || fields[2] == NULL ||
        strtok(NULL, " ") != NULL ||
        strcmp(fields[0], kind) != 0 ||
        strcmp(fields[2], expected_name) != 0 ||
        strlen(fields[1]) != 64u) {
        return 1;
    }
    for (index = 0u; index < 64u; index++) {
        if (!((fields[1][index] >= '0' && fields[1][index] <= '9') ||
              (fields[1][index] >= 'a' && fields[1][index] <= 'f'))) {
            return 1;
        }
    }
    memcpy(hash, fields[1], 65u);
    return 0;
}

static int km_validate_base_checksum(KMBase *base)
{
    static const char *const expected_manifest_hashes[] = {
        "081703f4b88f221a58bc7285100552a1fe8afc1dae74993e4a6aa5de94f8ee55",
        "99f51cf2f265f38708fef80c2180dec6bb13cbf01d7bd81b5af27d23a4a30261"
    };
    char input_storage[KM_BASE_INPUT_COUNT][GS_PATH_CAPACITY];
    const char *input_paths[KM_BASE_INPUT_COUNT];
    char sha_path[GS_PATH_CAPACITY];
    char actual[65];
    char line[GS_LINE_CAPACITY];
    FILE *file;
    size_t profile_index =
        strcmp(base->profile->token, "mini75_pool5") == 0 ? 0u : 1u;
    size_t index;

    if (km_build_base_checksum_inputs(base, input_storage, input_paths) != 0 ||
        gs_path_join(base->base, "sha256.txt", sha_path, sizeof(sha_path)) != 0 ||
        gs_sha_file(sha_path, actual) != 0 ||
        strcmp(actual, expected_manifest_hashes[profile_index]) != 0 ||
        (file = fopen(sha_path, "rb")) == NULL) {
        return 1;
    }
    for (index = 0u; index < KM_BASE_INPUT_COUNT; index++) {
        char expected_hash[65];
        if (fgets(line, sizeof(line), file) == NULL ||
            km_parse_checksum_line(
                line, "INPUT", gs_input_logical_names[index], expected_hash
            ) != 0 ||
            gs_sha_file(input_paths[index], actual) != 0 ||
            strcmp(actual, expected_hash) != 0) {
            fclose(file);
            return 1;
        }
    }
    for (index = 0u; index < KM_BASE_ARTIFACT_COUNT; index++) {
        char expected_hash[65];
        char artifact_path[GS_PATH_CAPACITY];
        if (fgets(line, sizeof(line), file) == NULL ||
            km_parse_checksum_line(
                line, "ARTIFACT", gs_artifact_names[index], expected_hash
            ) != 0 ||
            gs_path_join(
                base->base, gs_artifact_names[index],
                artifact_path, sizeof(artifact_path)
            ) != 0 ||
            gs_sha_file(artifact_path, actual) != 0 ||
            strcmp(actual, expected_hash) != 0) {
            fclose(file);
            return 1;
        }
    }
    {
        int extra_line = fgets(line, sizeof(line), file) != NULL;
        int read_failed = ferror(file);
        int close_failed = fclose(file) != 0;
        if (extra_line || read_failed || close_failed) {
            return 1;
        }
    }
    return 0;
}

static int km_validate_metadata(KMBase *base)
{
    char feature_path[GS_PATH_CAPACITY];
    char summary_path[GS_PATH_CAPACITY];
    char value[128];
    char expected[128];

    if (gs_path_join(
            base->base, "feature_spec.txt",
            feature_path, sizeof(feature_path)
        ) != 0 ||
        gs_path_join(
            base->base, "summary.txt",
            summary_path, sizeof(summary_path)
        ) != 0 ||
        km_file_value(
            feature_path, "profile", value, sizeof(value)
        ) != 0 ||
        strcmp(value, base->profile->matched_profile) != 0 ||
        km_file_value(
            feature_path, "source_profile", value, sizeof(value)
        ) != 0 ||
        strcmp(value, base->profile->source_profile) != 0 ||
        !gs_file_has_key_value(
            feature_path, "matched_subset_profile",
            "subset224_seed123_prefix_800_200"
        ) ||
        !gs_file_has_key_value(
            feature_path, "provenance",
            "slice_raw_official_full_baseline"
        ) ||
        !gs_file_has_key_value(
            feature_path, "matrix_layout", "row_major"
        ) ||
        !gs_file_has_key_value(
            feature_path, "matrix_dtype", "float32_ieee754"
        ) ||
        !gs_file_has_key_value(
            feature_path, "matrix_endian", "little"
        ) ||
        !gs_file_has_key_value(
            feature_path, "standardization",
            "train_only_population_ddof0"
        ) ||
        !gs_file_has_key_value(
            feature_path, "standardization_fit", "train_only"
        ) ||
        !gs_file_has_key_value(
            feature_path, "standardization_ddof", "0"
        ) ||
        !gs_file_has_key_value(
            feature_path, "constant_feature_count", "0"
        ) ||
        !gs_file_has_key_value(
            feature_path, "scaler_unchanged_after_validation", "true"
        ) ||
        !gs_file_has_key_value(feature_path, "pca", "false") ||
        !gs_file_has_key_value(feature_path, "augmentation", "false") ||
        !gs_file_has_key_value(feature_path, "labels_used", "false")) {
        return 1;
    }
    if (km_file_value(
            feature_path, "train_rows", value, sizeof(value)
        ) != 0 ||
        km_parse_size(value, &base->train_rows) != 0 ||
        km_file_value(
            feature_path, "validation_rows", value, sizeof(value)
        ) != 0 ||
        km_parse_size(value, &base->validation_rows) != 0 ||
        km_file_value(
            feature_path, "feature_count", value, sizeof(value)
        ) != 0 ||
        km_parse_size(value, &base->feature_count) != 0 ||
        km_file_value(
            feature_path, "matched_builder_code_commit",
            base->matched_base_commit,
            sizeof(base->matched_base_commit)
        ) != 0 ||
        base->train_rows != GS_MATCHED_TRAIN_ROWS ||
        base->validation_rows != GS_MATCHED_VALIDATION_ROWS ||
        base->feature_count != base->profile->feature_count ||
        strcmp(base->matched_base_commit, KM_MATCHED_BASE_COMMIT) != 0) {
        return 1;
    }
    if (snprintf(expected, sizeof(expected), "%zu", base->feature_count) < 0 ||
        !gs_file_has_key_value(summary_path, "profile",
                               base->profile->matched_profile) ||
        !gs_file_has_key_value(summary_path, "source_profile",
                               base->profile->source_profile) ||
        !gs_file_has_key_value(
            summary_path, "matched_subset_profile",
            "subset224_seed123_prefix_800_200"
        ) ||
        !gs_file_has_key_value(
            summary_path, "provenance",
            "slice_raw_official_full_baseline"
        ) ||
        !gs_file_has_key_value(
            summary_path, "matched_builder_code_commit",
            base->matched_base_commit
        ) ||
        !gs_file_has_key_value(summary_path, "train_rows", "800") ||
        !gs_file_has_key_value(summary_path, "validation_rows", "200") ||
        !gs_file_has_key_value(summary_path, "feature_count", expected) ||
        !gs_file_has_key_value(
            summary_path, "standardization",
            "train_only_population_ddof0"
        ) ||
        !gs_file_has_key_value(
            summary_path, "validation_transformed_without_refit", "true"
        ) ||
        !gs_file_has_key_value(
            summary_path, "scaler_unchanged_after_validation", "true"
        ) ||
        !gs_file_has_key_value(
            summary_path, "standardized_values_finite", "true"
        ) ||
        !gs_file_has_key_value(summary_path, "pca", "false") ||
        !gs_file_has_key_value(summary_path, "augmentation", "false") ||
        !gs_file_has_key_value(summary_path, "labels_used", "false") ||
        !gs_file_has_key_value(summary_path, "kmeans_used", "false")) {
        return 1;
    }
    return 0;
}

static int km_load_scaler(
    const KMBase *base,
    double *mean,
    double *stddev,
    unsigned char *constant
)
{
    char path[GS_PATH_CAPACITY];
    char line[GS_LINE_CAPACITY];
    FILE *file = NULL;
    size_t feature;
    const char *header =
        "feature_index,channel,pooled_row,pooled_col,mean,stddev,constant";

    if (base == NULL || mean == NULL || stddev == NULL || constant == NULL ||
        gs_path_join(base->base, "scaler.csv", path, sizeof(path)) != 0 ||
        (file = fopen(path, "rb")) == NULL ||
        fgets(line, sizeof(line), file) == NULL) {
        if (file != NULL) {
            fclose(file);
        }
        return 1;
    }
    gs_chomp(line);
    if (strcmp(line, header) != 0) {
        fclose(file);
        return 1;
    }
    for (feature = 0u; feature < base->feature_count; feature++) {
        char *fields[7];
        size_t count;
        size_t parsed_feature;
        size_t channel;
        size_t pooled_row;
        size_t pooled_col;
        size_t expected_plane =
            base->profile->pooled_side * base->profile->pooled_side;
        char *end = NULL;
        unsigned long constant_value;

        if (fgets(line, sizeof(line), file) == NULL) {
            fclose(file);
            return 1;
        }
        gs_chomp(line);
        if (gs_csv_fields(line, fields, 7u, &count) != 0 || count != 7u ||
            km_parse_size(fields[0], &parsed_feature) != 0 ||
            km_parse_size(fields[1], &channel) != 0 ||
            km_parse_size(fields[2], &pooled_row) != 0 ||
            km_parse_size(fields[3], &pooled_col) != 0 ||
            parsed_feature != feature ||
            channel != feature / expected_plane ||
            pooled_row != (feature % expected_plane) /
                          base->profile->pooled_side ||
            pooled_col != feature % base->profile->pooled_side) {
            fclose(file);
            return 1;
        }
        errno = 0;
        mean[feature] = strtod(fields[4], &end);
        if (errno == ERANGE || end == fields[4] || *end != '\0') {
            fclose(file);
            return 1;
        }
        errno = 0;
        stddev[feature] = strtod(fields[5], &end);
        if (errno == ERANGE || end == fields[5] || *end != '\0') {
            fclose(file);
            return 1;
        }
        errno = 0;
        constant_value = strtoul(fields[6], &end, 10);
        if (errno == ERANGE || end == fields[6] || *end != '\0' ||
            constant_value > 1u || !isfinite(mean[feature]) ||
            !isfinite(stddev[feature]) || stddev[feature] <= 0.0) {
            fclose(file);
            return 1;
        }
        constant[feature] = (unsigned char)constant_value;
        if (constant[feature] != 0u) {
            fclose(file);
            return 1;
        }
    }
    {
        int extra_line = fgets(line, sizeof(line), file) != NULL;
        int read_failed = ferror(file);
        int close_failed = fclose(file) != 0;
        if (extra_line || read_failed || close_failed) {
            return 1;
        }
    }
    return 0;
}

static int km_read_matrix(
    const char *path,
    size_t rows,
    size_t columns,
    float **matrix
)
{
    size_t count;
    size_t bytes;
    size_t actual;
    size_t index;
    FILE *file = NULL;
    float *values = NULL;

    if (matrix == NULL ||
        gs_checked_product(rows, columns, &count) != 0 ||
        gs_checked_product(count, sizeof(float), &bytes) != 0 ||
        gs_file_size(path, &actual) != 0 || actual != bytes ||
        (values = (float *)malloc(bytes)) == NULL ||
        (file = fopen(path, "rb")) == NULL) {
        free(values);
        return 1;
    }
    {
        int failed = 0;
        if (fread(values, sizeof(float), count, file) != count ||
            fgetc(file) != EOF || ferror(file)) {
            failed = 1;
        }
        if (fclose(file) != 0) {
            failed = 1;
        }
        if (failed) {
            free(values);
            return 1;
        }
    }
    for (index = 0u; index < count; index++) {
        if (!isfinite(values[index])) {
            free(values);
            return 1;
        }
    }
    *matrix = values;
    return 0;
}

static int km_validate_rows(const KMBase *base)
{
    GSRow *train = NULL;
    GSRow *validation = NULL;
    GSRow *subset_train = NULL;
    GSRow *subset_validation = NULL;
    char path[GS_PATH_CAPACITY];
    size_t index;
    int failed = 0;

    train = (GSRow *)calloc(base->train_rows, sizeof(*train));
    validation = (GSRow *)calloc(base->validation_rows, sizeof(*validation));
    subset_train =
        (GSRow *)calloc(base->train_rows, sizeof(*subset_train));
    subset_validation =
        (GSRow *)calloc(base->validation_rows, sizeof(*subset_validation));
    if (train == NULL || validation == NULL ||
        subset_train == NULL || subset_validation == NULL ||
        gs_path_join(
            base->base, "train_rows.csv", path, sizeof(path)
        ) != 0 ||
        gs_load_rows(
            path, "train", base->train_rows, 0, train
        ) != 0 ||
        gs_path_join(
            base->base, "val_rows.csv", path, sizeof(path)
        ) != 0 ||
        gs_load_rows(
            path, "validation", base->validation_rows, 0, validation
        ) != 0 ||
        gs_load_rows(
            base->subset_train, "train",
            base->train_rows, 1, subset_train
        ) != 0 ||
        gs_load_rows(
            base->subset_validation, "validation",
            base->validation_rows, 1, subset_validation
        ) != 0) {
        failed = 1;
    }
    for (index = 0u; !failed && index < base->train_rows; index++) {
        if (!gs_rows_equal(&train[index], &subset_train[index])) {
            failed = 1;
        }
    }
    for (index = 0u; !failed && index < base->validation_rows; index++) {
        if (!gs_rows_equal(&validation[index], &subset_validation[index])) {
            failed = 1;
        }
    }
    free(subset_validation);
    free(subset_train);
    free(validation);
    free(train);
    return failed;
}

static int km_prepare_base(
    const char *directory,
    const char *token,
    KMBase *base
)
{
    double *mean = NULL;
    double *stddev = NULL;
    unsigned char *constant = NULL;
    GSStatistics statistics;
    char path[GS_PATH_CAPACITY];
    size_t index;
    int failed = 0;

    if (base == NULL) {
        return 1;
    }
    memset(base, 0, sizeof(*base));
    memset(&statistics, 0, sizeof(statistics));
    if (km_resolve_base(directory, token, base) != 0 ||
        km_validate_base_checksum(base) != 0 ||
        km_validate_metadata(base) != 0 ||
        km_validate_rows(base) != 0) {
        return 1;
    }
    for (index = 0u; index < KM_INPUT_COUNT; index++) {
        if (gs_path_join(
                base->base, km_input_names[index],
                base->input_storage[index], GS_PATH_CAPACITY
            ) != 0 ||
            gs_canonical_existing(
                base->input_storage[index], 0, path
            ) != 0 ||
            !gs_paths_equal(base->input_storage[index], path) ||
            gs_sha_file(
                base->input_storage[index], base->input_hashes[index]
            ) != 0) {
            return 1;
        }
        base->input_paths[index] = base->input_storage[index];
    }
    mean = (double *)malloc(base->feature_count * sizeof(*mean));
    stddev = (double *)malloc(base->feature_count * sizeof(*stddev));
    constant =
        (unsigned char *)malloc(base->feature_count * sizeof(*constant));
    if (mean == NULL || stddev == NULL || constant == NULL ||
        km_load_scaler(base, mean, stddev, constant) != 0 ||
        km_read_matrix(
            base->input_paths[2], base->train_rows,
            base->feature_count, &base->train
        ) != 0 ||
        km_read_matrix(
            base->input_paths[3], base->validation_rows,
            base->feature_count, &base->validation
        ) != 0 ||
        gs_validate_standardized(
            base->train, base->validation,
            mean, stddev, constant, base->feature_count, &statistics
        ) != 0 ||
        statistics.constant_count != 0u) {
        failed = 1;
    }
    free(constant);
    free(stddev);
    free(mean);
    if (failed) {
        km_base_free(base);
    }
    return failed;
}

static int km_resolve_output(
    const KMBase *base,
    size_t k,
    const char *partial_argument,
    char partial[GS_PATH_CAPACITY],
    char final[GS_PATH_CAPACITY]
)
{
    char absolute[GS_PATH_CAPACITY];
    char parent_argument[GS_PATH_CAPACITY];
    char canonical_parent[GS_PATH_CAPACITY];
    char canonical_expected_parent[GS_PATH_CAPACITY];
    char expected_parent[GS_PATH_CAPACITY];
    char expected_name[32];
    char final_name[16];
    int written;

    if (base == NULL || partial_argument == NULL ||
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
    gs_normalize_separators(absolute);
    written = snprintf(expected_name, sizeof(expected_name), "k%zu.partial", k);
    if (written < 0 || (size_t)written >= sizeof(expected_name) ||
        strcmp(gs_basename(absolute), expected_name) != 0 ||
        gs_parent_path(absolute, parent_argument, sizeof(parent_argument)) != 0 ||
        gs_canonical_existing(parent_argument, 1, canonical_parent) != 0 ||
        gs_path_join(
            base->base, "kmeans_seed123",
            expected_parent, sizeof(expected_parent)
        ) != 0 ||
        gs_canonical_existing(
            expected_parent, 1, canonical_expected_parent
        ) != 0 ||
        !gs_paths_equal(canonical_parent, canonical_expected_parent) ||
        gs_path_join(
            canonical_parent, expected_name,
            partial, GS_PATH_CAPACITY
        ) != 0 ||
        !gs_paths_equal(absolute, partial)) {
        return 1;
    }
    written = snprintf(final_name, sizeof(final_name), "k%zu", k);
    if (written < 0 || (size_t)written >= sizeof(final_name) ||
        gs_path_join(
            canonical_parent, final_name, final, GS_PATH_CAPACITY
        ) != 0 ||
        gs_path_state(partial) != 0 || gs_path_state(final) != 0) {
        return 1;
    }
    return 0;
}

static int km_output_inventory_valid(const char *directory)
{
    unsigned char seen[KM_OUTPUT_COUNT] = {0};
    DIR *stream;
    struct dirent *entry;
    size_t index;

    stream = opendir(directory);
    if (stream == NULL) {
        return 1;
    }
    while ((entry = readdir(stream)) != NULL) {
        char path[GS_PATH_CAPACITY];
        struct stat information;
        size_t matched;
        if (strcmp(entry->d_name, ".") == 0 ||
            strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        if (!km_name_index(
                entry->d_name, km_output_names, KM_OUTPUT_COUNT, &matched
            ) ||
            seen[matched] != 0u ||
            gs_path_join(
                directory, entry->d_name, path, sizeof(path)
            ) != 0 ||
            stat(path, &information) != 0 ||
            !S_ISREG(information.st_mode)) {
            closedir(stream);
            return 1;
        }
#ifdef _WIN32
        {
            DWORD attributes = GetFileAttributesA(path);
            if (attributes == INVALID_FILE_ATTRIBUTES ||
                (attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0u) {
                closedir(stream);
                return 1;
            }
        }
#endif
        seen[matched] = 1u;
    }
    if (closedir(stream) != 0) {
        return 1;
    }
    for (index = 0u; index < KM_OUTPUT_COUNT; index++) {
        if (seen[index] == 0u) {
            return 1;
        }
    }
    return 0;
}

static int km_finish_file(FILE *file)
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

static int km_open_output(
    const char *directory,
    const char *name,
    const char *mode,
    FILE **file
)
{
    char path[GS_PATH_CAPACITY];
    if (directory == NULL || name == NULL || mode == NULL || file == NULL ||
        !km_name_index(name, km_output_names, KM_OUTPUT_COUNT, NULL) ||
        gs_path_join(directory, name, path, sizeof(path)) != 0 ||
        gs_path_state(path) != 0) {
        return 1;
    }
    *file = fopen(path, mode);
    return *file == NULL;
}

static int km_write_binary(
    const char *directory,
    const char *name,
    const void *values,
    size_t element_size,
    size_t count
)
{
    FILE *file = NULL;
    int failed = 0;
    if (values == NULL || element_size == 0u ||
        km_open_output(directory, name, "wb", &file) != 0) {
        return 1;
    }
    if (fwrite(values, element_size, count, file) != count) {
        failed = 1;
    }
    if (km_finish_file(file) != 0) {
        failed = 1;
    }
    return failed;
}

static int km_write_assignment_csv(
    const char *directory,
    const char *name,
    const uint32_t *assignments,
    const double *distances,
    size_t rows
)
{
    FILE *file = NULL;
    size_t row;
    int failed = 0;
    if (assignments == NULL || distances == NULL ||
        km_open_output(directory, name, "wb", &file) != 0) {
        return 1;
    }
    if (fputs("row_index,cluster_id,squared_distance\n", file) == EOF) {
        failed = 1;
    }
    for (row = 0u; !failed && row < rows; row++) {
        if (!isfinite(distances[row]) || distances[row] < 0.0 ||
            fprintf(
                file, "%zu,%" PRIu32 ",%.17g\n",
                row, assignments[row], distances[row]
            ) < 0) {
            failed = 1;
        }
    }
    if (km_finish_file(file) != 0) {
        failed = 1;
    }
    return failed;
}

static int km_write_cluster_sizes(
    const char *directory,
    const uint64_t *sizes,
    size_t k
)
{
    FILE *file = NULL;
    size_t cluster;
    int failed = 0;
    if (sizes == NULL ||
        km_open_output(
            directory, "cluster_sizes.csv", "wb", &file
        ) != 0) {
        return 1;
    }
    if (fputs("cluster_id,train_count\n", file) == EOF) {
        failed = 1;
    }
    for (cluster = 0u; !failed && cluster < k; cluster++) {
        if (fprintf(
                file, "%zu,%" PRIu64 "\n", cluster, sizes[cluster]
            ) < 0) {
            failed = 1;
        }
    }
    if (km_finish_file(file) != 0) {
        failed = 1;
    }
    return failed;
}

static int km_write_initial_indices(
    const char *directory,
    const uint64_t *indices,
    size_t k
)
{
    FILE *file = NULL;
    size_t order;
    int failed = 0;
    if (indices == NULL ||
        km_open_output(
            directory, "initial_centroids.csv", "wb", &file
        ) != 0) {
        return 1;
    }
    if (fputs("centroid_order,train_row_index\n", file) == EOF) {
        failed = 1;
    }
    for (order = 0u; !failed && order < k; order++) {
        if (fprintf(
                file, "%zu,%" PRIu64 "\n", order, indices[order]
            ) < 0) {
            failed = 1;
        }
    }
    if (km_finish_file(file) != 0) {
        failed = 1;
    }
    return failed;
}

static int km_prepare_outputs(
    const KMBase *base,
    const GSBaselineKMeansResult *result,
    const size_t *train_assignments,
    const size_t *validation_assignments,
    const double *train_distances,
    const double *validation_distances,
    uint32_t *train_u32,
    uint32_t *validation_u32,
    uint64_t *cluster_sizes,
    uint64_t *initial_indices,
    KMNumericSummary *summary
)
{
    size_t row;
    size_t cluster;
    size_t total = 0u;

    if (base == NULL || result == NULL || train_assignments == NULL ||
        validation_assignments == NULL || train_distances == NULL ||
        validation_distances == NULL || train_u32 == NULL ||
        validation_u32 == NULL || cluster_sizes == NULL ||
        initial_indices == NULL || summary == NULL ||
        result->converged != 1 || result->rows != base->train_rows ||
        result->cols != base->feature_count ||
        result->k == 0u || result->k > UINT32_MAX ||
        !isfinite(result->initial_inertia) ||
        !isfinite(result->inertia) ||
        result->initial_inertia < 0.0 || result->inertia < 0.0) {
        return 1;
    }
    memset(cluster_sizes, 0, result->k * sizeof(*cluster_sizes));
    memset(summary, 0, sizeof(*summary));
    summary->initial_inertia = result->initial_inertia;
    summary->minimum_cluster_size = base->train_rows;

    for (row = 0u; row < base->train_rows; row++) {
        size_t assignment = train_assignments[row];
        if (assignment >= result->k || assignment > UINT32_MAX ||
            !isfinite(train_distances[row]) || train_distances[row] < 0.0 ||
            result->assignments[row] != assignment) {
            return 1;
        }
        train_u32[row] = (uint32_t)assignment;
        cluster_sizes[assignment]++;
        summary->train_inertia += train_distances[row];
        if (!isfinite(summary->train_inertia)) {
            return 1;
        }
    }
    for (row = 0u; row < base->validation_rows; row++) {
        size_t assignment = validation_assignments[row];
        if (assignment >= result->k || assignment > UINT32_MAX ||
            !isfinite(validation_distances[row]) ||
            validation_distances[row] < 0.0) {
            return 1;
        }
        validation_u32[row] = (uint32_t)assignment;
        summary->validation_inertia += validation_distances[row];
        if (!isfinite(summary->validation_inertia)) {
            return 1;
        }
    }
    for (cluster = 0u; cluster < result->k; cluster++) {
        size_t inner;
        if (result->initial_centroid_indices[cluster] >= base->train_rows) {
            return 1;
        }
        for (inner = 0u; inner < cluster; inner++) {
            if (result->initial_centroid_indices[inner] ==
                result->initial_centroid_indices[cluster]) {
                return 1;
            }
        }
        initial_indices[cluster] =
            (uint64_t)result->initial_centroid_indices[cluster];
        if (cluster_sizes[cluster] == 0u) {
            summary->empty_clusters++;
        }
        if (cluster_sizes[cluster] < summary->minimum_cluster_size) {
            summary->minimum_cluster_size = (size_t)cluster_sizes[cluster];
        }
        if (cluster_sizes[cluster] > summary->maximum_cluster_size) {
            summary->maximum_cluster_size = (size_t)cluster_sizes[cluster];
        }
        if (cluster_sizes[cluster] > SIZE_MAX - total) {
            return 1;
        }
        total += (size_t)cluster_sizes[cluster];
    }
    if (total != base->train_rows) {
        return 1;
    }
    for (cluster = 0u; cluster < result->k; cluster++) {
        if (cluster_sizes[cluster] !=
            (uint64_t)result->cluster_sizes[cluster]) {
            return 1;
        }
    }
    if (memcmp(
            &summary->train_inertia, &result->inertia, sizeof(double)
        ) != 0) {
        return 1;
    }
    return 0;
}

static int km_write_summary(
    const char *directory,
    const KMBase *base,
    size_t k,
    uint64_t seed,
    size_t max_iterations,
    const GSBaselineKMeansResult *result,
    const KMNumericSummary *summary,
    const char *code_commit
)
{
    FILE *file = NULL;
    int failed = 0;

    if (base == NULL || result == NULL || summary == NULL ||
        km_open_output(directory, "summary.txt", "wb", &file) != 0) {
        return 1;
    }
    if (fprintf(
            file,
            "format_version=1\n"
            "profile=matched_subset1000_%s_no_pca_kmeans\n"
            "input_profile=%s\n"
            "matched_subset_profile=subset224_seed123_prefix_800_200\n"
            "source_profile=%s\n"
            "algorithm=kmeans_plus_plus_lloyd\n"
            "input=train_features_std_f32\n"
            "validation_policy=assign_to_train_centroids\n"
            "train_rows=%zu\n"
            "validation_rows=%zu\n"
            "feature_count=%zu\n"
            "k=%zu\n"
            "seed=%" PRIu64 "\n"
            "max_iterations=%zu\n"
            "iterations=%zu\n"
            "converged=%d\n"
            "empty_cluster_reseeds=%zu\n"
            "initial_inertia=%.17g\n"
            "final_train_inertia=%.17g\n"
            "validation_inertia=%.17g\n"
            "empty_final_clusters=%zu\n"
            "minimum_cluster_size=%zu\n"
            "maximum_cluster_size=%zu\n"
            "centroid_dtype=float64\n"
            "assignment_dtype=uint32\n"
            "distance_dtype=float64\n"
            "cluster_size_dtype=uint64\n"
            "standardization=train_only_population_ddof0\n"
            "fit_input=train_only\n"
            "validation_assigned_after_fit=true\n"
            "pca=false\n"
            "whitening=false\n"
            "augmentation=false\n"
            "labels_used=false\n"
            "matched_base_code_commit=%s\n"
            "code_commit=%s\n",
            base->profile->token,
            base->profile->matched_profile,
            base->profile->source_profile,
            base->train_rows,
            base->validation_rows,
            base->feature_count,
            k,
            seed,
            max_iterations,
            result->iterations,
            result->converged,
            result->empty_cluster_reseeds,
            summary->initial_inertia,
            summary->train_inertia,
            summary->validation_inertia,
            summary->empty_clusters,
            summary->minimum_cluster_size,
            summary->maximum_cluster_size,
            base->matched_base_commit,
            code_commit
        ) < 0) {
        failed = 1;
    }
    if (km_finish_file(file) != 0) {
        failed = 1;
    }
    return failed;
}

static int km_write_checksum(
    const char *directory,
    const KMBase *base
)
{
    FILE *file = NULL;
    size_t index;
    int failed = 0;

    if (km_open_output(directory, "sha256.txt", "wb", &file) != 0) {
        return 1;
    }
    for (index = 0u; !failed && index < KM_INPUT_COUNT; index++) {
        char actual[65];
        if (gs_sha_file(base->input_paths[index], actual) != 0 ||
            strcmp(actual, base->input_hashes[index]) != 0 ||
            fprintf(
                file, "INPUT %s %s\n", actual, km_input_names[index]
            ) < 0) {
            failed = 1;
        }
    }
    for (index = 0u; !failed && index < KM_ARTIFACT_COUNT; index++) {
        char path[GS_PATH_CAPACITY];
        char hash[65];
        if (gs_path_join(
                directory, km_artifact_names[index],
                path, sizeof(path)
            ) != 0 ||
            gs_sha_file(path, hash) != 0 ||
            fprintf(
                file, "ARTIFACT %s %s\n",
                hash, km_artifact_names[index]
            ) < 0) {
            failed = 1;
        }
    }
    if (km_finish_file(file) != 0) {
        failed = 1;
    }
    return failed;
}

static int km_verify_checksum(
    const char *directory,
    const KMBase *base
)
{
    char path[GS_PATH_CAPACITY];
    char line[GS_LINE_CAPACITY];
    char actual[65];
    FILE *file;
    size_t index;

    if (gs_path_join(
            directory, "sha256.txt", path, sizeof(path)
        ) != 0 || (file = fopen(path, "rb")) == NULL) {
        return 1;
    }
    for (index = 0u; index < KM_INPUT_COUNT; index++) {
        char expected[65];
        if (fgets(line, sizeof(line), file) == NULL ||
            km_parse_checksum_line(
                line, "INPUT", km_input_names[index], expected
            ) != 0 ||
            gs_sha_file(base->input_paths[index], actual) != 0 ||
            strcmp(actual, expected) != 0) {
            fclose(file);
            return 1;
        }
    }
    for (index = 0u; index < KM_ARTIFACT_COUNT; index++) {
        char expected[65];
        if (fgets(line, sizeof(line), file) == NULL ||
            km_parse_checksum_line(
                line, "ARTIFACT", km_artifact_names[index], expected
            ) != 0 ||
            gs_path_join(
                directory, km_artifact_names[index],
                path, sizeof(path)
            ) != 0 ||
            gs_sha_file(path, actual) != 0 ||
            strcmp(actual, expected) != 0) {
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

static int km_validate_output_sizes(
    const char *directory,
    const KMBase *base,
    size_t k
)
{
    size_t expected[KM_OUTPUT_COUNT] = {0};
    size_t centroid_count;
    size_t index;

    if (gs_checked_product(k, base->feature_count, &centroid_count) != 0 ||
        gs_checked_product(
            centroid_count, sizeof(double), &expected[0]
        ) != 0 ||
        gs_checked_product(
            base->train_rows, sizeof(uint32_t), &expected[1]
        ) != 0 ||
        gs_checked_product(
            base->validation_rows, sizeof(uint32_t), &expected[2]
        ) != 0 ||
        gs_checked_product(
            base->train_rows, sizeof(double), &expected[3]
        ) != 0 ||
        gs_checked_product(
            base->validation_rows, sizeof(double), &expected[4]
        ) != 0 ||
        gs_checked_product(k, sizeof(uint64_t), &expected[5]) != 0) {
        return 1;
    }
    expected[6] = expected[5];
    for (index = 0u; index < KM_OUTPUT_COUNT; index++) {
        char path[GS_PATH_CAPACITY];
        size_t actual;
        if (gs_path_join(
                directory, km_output_names[index], path, sizeof(path)
            ) != 0 ||
            gs_file_size(path, &actual) != 0 ||
            (index < 7u && actual != expected[index]) ||
            (index >= 7u && actual == 0u)) {
            return 1;
        }
    }
    return km_output_inventory_valid(directory) ||
           km_verify_checksum(directory, base);
}

static int km_remove_partial(const char *directory)
{
    size_t index;
    int failed = 0;
    if (directory == NULL) {
        return 1;
    }
    for (index = 0u; index < KM_OUTPUT_COUNT; index++) {
        char path[GS_PATH_CAPACITY];
        if (gs_path_join(
                directory, km_output_names[index], path, sizeof(path)
            ) != 0) {
            failed = 1;
        } else {
            errno = 0;
            if (remove(path) != 0 && errno != ENOENT) {
                failed = 1;
            }
        }
    }
    if (failed) {
        return 1;
    }
#ifdef _WIN32
    return _rmdir(directory) == 0 ? 0 : 1;
#else
    return rmdir(directory) == 0 ? 0 : 1;
#endif
}

static int km_failure_hook_enabled(const char *partial)
{
    const char *value = getenv(KM_FAILURE_ENV);
    const char *name = gs_basename(partial);
    return value != NULL && strcmp(value, "1") == 0 &&
        name != NULL &&
        (strcmp(name, "k22.partial") == 0 ||
         strcmp(name, "k36.partial") == 0);
}

#ifndef GS_MATCHED_KMEANS_NO_MAIN
int main(int argc, char **argv)
{
    KMBase base;
    size_t k;
    uint64_t seed;
    size_t max_iterations;
    size_t train_values;
    size_t validation_values;
    size_t centroid_values;
    size_t centroid_bytes;
    size_t train_size_t_bytes;
    size_t validation_size_t_bytes;
    size_t train_u32_bytes;
    size_t validation_u32_bytes;
    size_t train_distance_bytes;
    size_t validation_distance_bytes;
    size_t k_u64_bytes;
    char partial[GS_PATH_CAPACITY];
    char final[GS_PATH_CAPACITY];
    size_t *train_assignments = NULL;
    size_t *validation_assignments = NULL;
    double *train_distances = NULL;
    double *validation_distances = NULL;
    double *centroids_copy = NULL;
    uint32_t *train_u32 = NULL;
    uint32_t *validation_u32 = NULL;
    uint64_t *cluster_sizes = NULL;
    uint64_t *initial_indices = NULL;
    GSBaselineKMeansResult result;
    KMNumericSummary summary;
    int partial_created = 0;
    int rc = 1;

    memset(&base, 0, sizeof(base));
    memset(&result, 0, sizeof(result));
    memset(&summary, 0, sizeof(summary));
    if (setlocale(LC_ALL, "C") == NULL) {
        fprintf(stderr, "ERROR: C locale unavailable.\n");
        return 1;
    }
    if (argc != 8 ||
        !km_platform_valid() ||
        km_parse_size(argv[3], &k) != 0 ||
        km_parse_u64(argv[4], &seed) != 0 ||
        km_parse_size(argv[5], &max_iterations) != 0 ||
        (k != 22u && k != 36u) ||
        seed != KM_SEED || max_iterations != KM_MAX_ITERATIONS ||
        !km_commit_valid(argv[7]) ||
        km_prepare_base(argv[1], argv[2], &base) != 0 ||
        km_resolve_output(
            &base, k, argv[6], partial, final
        ) != 0) {
        fprintf(
            stderr,
            "Usage: run_matched_subset1000_kmeans.exe "
            "<matched_base_directory> "
            "<mini75_pool5|mini112_pool7> <22|36> 123 300 "
            "<output_partial_directory> <code_commit>\n"
        );
        km_base_free(&base);
        return 1;
    }
    printf("[PASS] canonical_matched_base_and_output_resolution\n");
    printf("[PASS] matched_base_inventory_metadata_rows_and_sha256\n");
    printf("[PASS] finite_train_only_standardized_inputs\n");

    if (gs_checked_product(
            base.train_rows, base.feature_count, &train_values
        ) != 0 ||
        gs_checked_product(
            base.validation_rows, base.feature_count, &validation_values
        ) != 0 ||
        gs_checked_product(k, base.feature_count, &centroid_values) != 0 ||
        gs_checked_product(
            centroid_values, sizeof(double), &centroid_bytes
        ) != 0 ||
        gs_checked_product(
            base.train_rows, sizeof(size_t), &train_size_t_bytes
        ) != 0 ||
        gs_checked_product(
            base.validation_rows, sizeof(size_t),
            &validation_size_t_bytes
        ) != 0 ||
        gs_checked_product(
            base.train_rows, sizeof(uint32_t), &train_u32_bytes
        ) != 0 ||
        gs_checked_product(
            base.validation_rows, sizeof(uint32_t),
            &validation_u32_bytes
        ) != 0 ||
        gs_checked_product(
            base.train_rows, sizeof(double), &train_distance_bytes
        ) != 0 ||
        gs_checked_product(
            base.validation_rows, sizeof(double),
            &validation_distance_bytes
        ) != 0 ||
        gs_checked_product(k, sizeof(uint64_t), &k_u64_bytes) != 0 ||
        train_values == 0u || validation_values == 0u) {
        fprintf(stderr, "ERROR: allocation-size overflow.\n");
        goto cleanup;
    }
    train_assignments = (size_t *)malloc(train_size_t_bytes);
    validation_assignments = (size_t *)malloc(validation_size_t_bytes);
    train_distances = (double *)malloc(train_distance_bytes);
    validation_distances = (double *)malloc(validation_distance_bytes);
    centroids_copy = (double *)malloc(centroid_bytes);
    train_u32 = (uint32_t *)malloc(train_u32_bytes);
    validation_u32 = (uint32_t *)malloc(validation_u32_bytes);
    cluster_sizes = (uint64_t *)malloc(k_u64_bytes);
    initial_indices = (uint64_t *)malloc(k_u64_bytes);
    if (train_assignments == NULL || validation_assignments == NULL ||
        train_distances == NULL || validation_distances == NULL ||
        centroids_copy == NULL || train_u32 == NULL ||
        validation_u32 == NULL || cluster_sizes == NULL ||
        initial_indices == NULL) {
        fprintf(stderr, "ERROR: allocation failed.\n");
        goto cleanup;
    }

    printf(
        "KMEANS_FIT_STARTED k=%zu rows=%zu cols=%zu\n",
        k, base.train_rows, base.feature_count
    );
    fflush(stdout);
    if (gs_baseline_kmeans_fit(
            base.train,
            base.train_rows,
            base.feature_count,
            k,
            seed,
            max_iterations,
            &result
        ) != GS_BASELINE_KMEANS_SUCCESS ||
        result.converged != 1 ||
        result.rows != base.train_rows ||
        result.cols != base.feature_count ||
        result.k != k) {
        fprintf(stderr, "ERROR: train-only fit failed or did not converge.\n");
        goto cleanup;
    }
    printf("KMEANS_FIT_COMPLETED\n");

    if (gs_baseline_kmeans_assign(
            base.train,
            base.train_rows,
            base.feature_count,
            result.centroids,
            k,
            train_assignments,
            train_distances
        ) != GS_BASELINE_KMEANS_SUCCESS) {
        fprintf(stderr, "ERROR: train recomputation failed.\n");
        goto cleanup;
    }
    memcpy(centroids_copy, result.centroids, centroid_bytes);
    if (gs_baseline_kmeans_assign(
            base.validation,
            base.validation_rows,
            base.feature_count,
            result.centroids,
            k,
            validation_assignments,
            validation_distances
        ) != GS_BASELINE_KMEANS_SUCCESS ||
        memcmp(
            centroids_copy, result.centroids, centroid_bytes
        ) != 0) {
        fprintf(stderr, "ERROR: frozen-centroid validation assignment failed.\n");
        goto cleanup;
    }
    if (km_prepare_outputs(
            &base,
            &result,
            train_assignments,
            validation_assignments,
            train_distances,
            validation_distances,
            train_u32,
            validation_u32,
            cluster_sizes,
            initial_indices,
            &summary
        ) != 0) {
        fprintf(stderr, "ERROR: numeric result contract failed.\n");
        goto cleanup;
    }
    printf("[PASS] train_fit_and_frozen_validation_assignment\n");

    if (gs_path_state(partial) != 0 || gs_path_state(final) != 0) {
        fprintf(stderr, "ERROR: output appeared before partial creation.\n");
        goto cleanup;
    }
#ifdef _WIN32
    if (_mkdir(partial) != 0) {
#else
    if (mkdir(partial, 0777) != 0) {
#endif
        fprintf(stderr, "ERROR: partial creation failed.\n");
        goto cleanup;
    }
    partial_created = 1;
    if (km_failure_hook_enabled(partial)) {
        fprintf(stderr, "ERROR: confined test failure after partial creation.\n");
        goto cleanup;
    }

    if (km_write_binary(
            partial, "centroids_f64.bin",
            result.centroids, sizeof(double), centroid_values
        ) != 0 ||
        km_write_binary(
            partial, "train_assignments_u32.bin",
            train_u32, sizeof(uint32_t), base.train_rows
        ) != 0 ||
        km_write_binary(
            partial, "val_assignments_u32.bin",
            validation_u32, sizeof(uint32_t), base.validation_rows
        ) != 0 ||
        km_write_binary(
            partial, "train_squared_distances_f64.bin",
            train_distances, sizeof(double), base.train_rows
        ) != 0 ||
        km_write_binary(
            partial, "val_squared_distances_f64.bin",
            validation_distances, sizeof(double), base.validation_rows
        ) != 0 ||
        km_write_binary(
            partial, "cluster_sizes_train_u64.bin",
            cluster_sizes, sizeof(uint64_t), k
        ) != 0 ||
        km_write_binary(
            partial, "initial_centroid_indices_u64.bin",
            initial_indices, sizeof(uint64_t), k
        ) != 0 ||
        km_write_assignment_csv(
            partial, "train_assignments.csv",
            train_u32, train_distances, base.train_rows
        ) != 0 ||
        km_write_assignment_csv(
            partial, "val_assignments.csv",
            validation_u32, validation_distances, base.validation_rows
        ) != 0 ||
        km_write_cluster_sizes(partial, cluster_sizes, k) != 0 ||
        km_write_initial_indices(partial, initial_indices, k) != 0 ||
        km_write_summary(
            partial, &base, k, seed, max_iterations,
            &result, &summary, argv[7]
        ) != 0 ||
        km_write_checksum(partial, &base) != 0 ||
        km_validate_output_sizes(partial, &base, k) != 0) {
        fprintf(stderr, "ERROR: artifact write or validation failed.\n");
        goto cleanup;
    }
    printf("[PASS] exact_inventory_sizes_and_sha256\n");
    if (rename(partial, final) != 0) {
        fprintf(stderr, "ERROR: atomic promotion failed.\n");
        goto cleanup;
    }
    partial_created = 0;
    printf("PROFILE=%s\n", base.profile->matched_profile);
    printf("K=%zu\n", k);
    printf("ITERATIONS=%zu\n", result.iterations);
    printf("CONVERGED=%d\n", result.converged);
    printf(
        "EMPTY_CLUSTER_RESEEDS=%zu\n",
        result.empty_cluster_reseeds
    );
    printf("INITIAL_INERTIA=%.17g\n", summary.initial_inertia);
    printf("FINAL_TRAIN_INERTIA=%.17g\n", summary.train_inertia);
    printf("VALIDATION_INERTIA=%.17g\n", summary.validation_inertia);
    printf("EMPTY_FINAL_CLUSTERS=%zu\n", summary.empty_clusters);
    printf("MINIMUM_CLUSTER_SIZE=%zu\n", summary.minimum_cluster_size);
    printf("MAXIMUM_CLUSTER_SIZE=%zu\n", summary.maximum_cluster_size);
    printf("MATCHED SUBSET1000 KMEANS RUN COMPLETED\n");
    rc = 0;

cleanup:
    gs_baseline_kmeans_result_free(&result);
    free(initial_indices);
    free(cluster_sizes);
    free(validation_u32);
    free(train_u32);
    free(centroids_copy);
    free(validation_distances);
    free(train_distances);
    free(validation_assignments);
    free(train_assignments);
    if (partial_created) {
        if (km_remove_partial(partial) != 0) {
            fprintf(stderr, "ERROR: confined partial cleanup failed.\n");
            rc = 1;
        } else {
            printf("[PASS] confined_current_run_partial_cleanup\n");
        }
    }
    km_base_free(&base);
    return rc;
}
#endif
