#include "baseline_kmeans.h"

#include <dirent.h>
#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <locale.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sys/types.h>
#include <unistd.h>
#endif


enum {
    GS_RUN_PATH_LENGTH = 4096,
    GS_RUN_LINE_LENGTH = 4096,
    GS_RUN_SHA256_LENGTH = 64,
    GS_RUN_TRAIN_ROWS = 800,
    GS_RUN_VALIDATION_ROWS = 200,
    GS_RUN_ORIGINAL_FEATURE_COUNT = 1024,
    GS_RUN_SEED = 123,
    GS_RUN_MAX_ITERATIONS = 300,
    GS_RUN_PCA_INPUT_COUNT = 7,
    GS_RUN_PCA_ARTIFACT_COUNT = 11,
    GS_RUN_PCA_FILE_COUNT = 12,
    GS_RUN_KMEANS_INPUT_COUNT = 5,
    GS_RUN_CLUSTER_INPUT_COUNT = 6,
    GS_RUN_BASE_INPUT_COUNT = 4,
    GS_RUN_BASE_ARTIFACT_COUNT = 9,
    GS_RUN_BASE_FILE_COUNT = 10,
    GS_RUN_CORE_ARTIFACT_COUNT = 12,
    GS_RUN_OUTPUT_FILE_COUNT = 13
};


#define GS_RUN_PARTIAL_SUFFIX ".partial"
#define GS_RUN_FAILURE_ENV "GS_MINI224_SUBSET1000_PCA_KMEANS_TEST_FAIL_AFTER_PARTIAL"
#define GS_RUN_FAILURE_BASENAME "mini224_subset1000_pca_kmeans_failure_test.partial"
#define GS_RUN_BASE_CODE_COMMIT \
    "d87ca996e7922d6873a54d952aec27cdb99a5998"
#define GS_RUN_PCA_CODE_COMMIT \
    "2cd2dc7f53d8a97fd1c69f88cfa416416818599c"


typedef struct {
    uint32_t state[8];
    uint64_t completed_bits;
    unsigned char block[64];
    size_t block_size;
} GSSha256;


typedef struct {
    double initial_inertia;
    double train_inertia;
    double validation_inertia;
    size_t empty_clusters;
    size_t minimum_cluster_size;
    size_t maximum_cluster_size;
} GSNumericSummary;

typedef struct {
    const char *token;
    char profile[64];
    char train_filename[64];
    char validation_filename[64];
    char pca_code_commit[65];
    size_t components;
    size_t d90;
    size_t d95;
    size_t d99;
    size_t numerical_rank;
    size_t nullspace_dimension;
    double threshold;
    double cumulative;
    double rank_tolerance;
} GSPCAProfile;

static const char *const gs_pca_input_names[GS_RUN_PCA_INPUT_COUNT] = {
    "feature_spec.txt",
    "scaler.csv",
    "train_features_std_f32.bin",
    "val_features_std_f32.bin",
    "train_rows.csv",
    "val_rows.csv",
    "source_no_pca_sha256.txt"
};


static const char *const gs224_base_input_names[GS_RUN_BASE_INPUT_COUNT] = {
    "manifest_events_272.csv",
    "manifest_tensors_224_central.csv",
    "subset224_train_800_seed123.csv",
    "subset224_val_200_seed123.csv"
};


static const char *const gs224_base_input_relative_paths[
    GS_RUN_BASE_INPUT_COUNT
] = {
    "manifest_events_272.csv",
    "manifest_tensors_224_central.csv",
    "sakai_multires_c/configs/subset224_train_800_seed123.csv",
    "sakai_multires_c/configs/subset224_val_200_seed123.csv"
};


static const char *const gs224_base_artifact_names[
    GS_RUN_BASE_ARTIFACT_COUNT
] = {
    "feature_spec.txt",
    "train_features_raw_f32.bin",
    "val_features_raw_f32.bin",
    "train_features_std_f32.bin",
    "val_features_std_f32.bin",
    "scaler.csv",
    "train_rows.csv",
    "val_rows.csv",
    "summary.txt"
};


static const char *const gs224_base_all_names[GS_RUN_BASE_FILE_COUNT] = {
    "feature_spec.txt",
    "train_features_raw_f32.bin",
    "val_features_raw_f32.bin",
    "train_features_std_f32.bin",
    "val_features_std_f32.bin",
    "scaler.csv",
    "train_rows.csv",
    "val_rows.csv",
    "summary.txt",
    "sha256.txt"
};

#if 0
static const char *const gs_base_input_names[GS_RUN_BASE_INPUT_COUNT] = {
    "manifest_events_272.csv",
    "manifest_tensors_224_central.csv",
    "subset224_train_800_seed123.csv",
    "subset224_val_200_seed123.csv"
};

static const char *const gs_base_artifact_names[
    GS_RUN_BASE_ARTIFACT_COUNT
] = {
    "feature_spec.txt",
    "train_features_raw_f32.bin",
    "val_features_raw_f32.bin",
    "train_features_std_f32.bin",
    "val_features_std_f32.bin",
    "scaler.csv",
    "train_rows.csv",
    "val_rows.csv",
    "summary.txt"
};

static const char *const gs_base_all_names[GS_RUN_BASE_FILE_COUNT] = {
    "feature_spec.txt",
    "train_features_raw_f32.bin",
    "val_features_raw_f32.bin",
    "train_features_std_f32.bin",
    "val_features_std_f32.bin",
    "scaler.csv",
    "train_rows.csv",
    "val_rows.csv",
    "summary.txt",
    "sha256.txt"
};
#endif


static const char *gs_pca_artifact_names[
    GS_RUN_PCA_ARTIFACT_COUNT
] = {
    "pca_spec.txt",
    "mean_f64.bin",
    "components_f64.bin",
    "eigenvalues_f64.bin",
    "explained_variance_ratio_f64.bin",
    "cumulative_explained_variance_ratio_f64.bin",
    "thresholds.csv",
    "spectrum.csv",
    "train_scores_pca90_f32.bin",
    "val_scores_pca90_f32.bin",
    "summary.txt"
};


static const char *const gs_pca_fixed_names[10] = {
    "pca_spec.txt",
    "mean_f64.bin",
    "components_f64.bin",
    "eigenvalues_f64.bin",
    "explained_variance_ratio_f64.bin",
    "cumulative_explained_variance_ratio_f64.bin",
    "thresholds.csv",
    "spectrum.csv",
    "summary.txt",
    "sha256.txt"
};


static const char *gs_cluster_input_names[
    GS_RUN_CLUSTER_INPUT_COUNT
] = {
    "pca_spec.txt",
    "thresholds.csv",
    "train_scores_pca90_f32.bin",
    "val_scores_pca90_f32.bin",
    "source_pca_sha256.txt",
    "unused"
};


static const char *const gs_core_artifact_names[
    GS_RUN_CORE_ARTIFACT_COUNT
] = {
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


static const char *const gs_output_all_names[GS_RUN_OUTPUT_FILE_COUNT] = {
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


static int gs_checked_product(
    size_t left,
    size_t right,
    size_t *product
)
{
    if (product == NULL || (right != 0u && left > SIZE_MAX / right)) {
        return 1;
    }

    *product = left * right;
    return 0;
}


static int gs_platform_is_valid(void)
{
    uint16_t value = UINT16_C(1);
    const unsigned char *bytes = (const unsigned char *)&value;

    return CHAR_BIT == 8 &&
           sizeof(float) == 4u &&
           FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 &&
           sizeof(double) == 8u &&
           DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024 &&
           sizeof(uint32_t) == 4u && sizeof(uint64_t) == 8u &&
           bytes[0] == 1u;
}


static int gs_parse_size_value(const char *text, size_t *value)
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


static int gs_parse_u64_value(const char *text, uint64_t *value)
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
        parsed > (uintmax_t)UINT64_MAX) {

        return 1;
    }

    *value = (uint64_t)parsed;
    return 0;
}


static int gs_metadata_value_is_valid(const char *value)
{
    return value != NULL && value[0] != '\0' &&
           strchr(value, '\r') == NULL && strchr(value, '\n') == NULL;
}


static int gs_commit_is_valid(const char *commit)
{
    size_t index;

    if (!gs_metadata_value_is_valid(commit)) {
        return 0;
    }
    if (strcmp(commit, "UNCOMMITTED_STEP10K_TEST") == 0) {
        return 1;
    }
    if (strlen(commit) != 40u) {
        return 0;
    }

    for (index = 0; index < 40u; index++) {
        if (!((commit[index] >= '0' && commit[index] <= '9') ||
              (commit[index] >= 'a' && commit[index] <= 'f'))) {

            return 0;
        }
    }

    return 1;
}


static const char *gs_path_basename(const char *path)
{
    const char *slash;
    const char *backslash;

    if (path == NULL) {
        return "";
    }

    slash = strrchr(path, '/');
    backslash = strrchr(path, '\\');

    if (slash == NULL) {
        return backslash == NULL ? path : backslash + 1;
    }
    if (backslash == NULL) {
        return slash + 1;
    }
    return slash > backslash ? slash + 1 : backslash + 1;
}


static int gs_parent_directory(
    const char *path,
    char *directory,
    size_t directory_size
)
{
    const char *basename;
    size_t length;

    if (!gs_metadata_value_is_valid(path) || directory == NULL ||
        directory_size == 0u) {

        return 1;
    }

    basename = gs_path_basename(path);
    if (basename[0] == '\0') {
        return 1;
    }

    if (basename == path) {
        if (directory_size < 2u) {
            return 1;
        }
        directory[0] = '.';
        directory[1] = '\0';
        return 0;
    }

    length = (size_t)(basename - path);
    while (length > 1u &&
           (path[length - 1u] == '/' || path[length - 1u] == '\\')) {
        length--;
    }

    if (length == 0u || length >= directory_size) {
        return 1;
    }

    memcpy(directory, path, length);
    directory[length] = '\0';
    return 0;
}


static int gs_directories_are_same(
    const char *left,
    const char *right
)
{
#ifdef _WIN32
    char left_full[GS_RUN_PATH_LENGTH];
    char right_full[GS_RUN_PATH_LENGTH];
    size_t index;

    if (left == NULL || right == NULL ||
        _fullpath(left_full, left, sizeof(left_full)) == NULL ||
        _fullpath(right_full, right, sizeof(right_full)) == NULL) {

        return 0;
    }

    for (index = 0; left_full[index] != '\0'; index++) {
        if (left_full[index] == '\\') {
            left_full[index] = '/';
        } else if (left_full[index] >= 'A' && left_full[index] <= 'Z') {
            left_full[index] =
                (char)(left_full[index] - 'A' + 'a');
        }
    }

    for (index = 0; right_full[index] != '\0'; index++) {
        if (right_full[index] == '\\') {
            right_full[index] = '/';
        } else if (right_full[index] >= 'A' && right_full[index] <= 'Z') {
            right_full[index] =
                (char)(right_full[index] - 'A' + 'a');
        }
    }

    return strcmp(left_full, right_full) == 0;
#else
    return left != NULL && right != NULL && strcmp(left, right) == 0;
#endif
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
    size_t position = 0u;
    int needs_separator;

    if (directory == NULL || filename == NULL || path == NULL ||
        path_size == 0u) {

        return 1;
    }

    directory_length = strlen(directory);
    filename_length = strlen(filename);
    needs_separator =
        directory_length != 0u &&
        directory[directory_length - 1u] != '/' &&
        directory[directory_length - 1u] != '\\';

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

    if (directory_length != 0u) {
        memcpy(path, directory, directory_length);
        position = directory_length;
    }
    if (needs_separator) {
        path[position++] = '/';
    }
    if (filename_length != 0u) {
        memcpy(path + position, filename, filename_length);
        position += filename_length;
    }
    path[position] = '\0';
    return 0;
}


static void gs_normalize_path_separators(char *path)
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


static int gs_paths_equal(const char *left, const char *right)
{
    if (left == NULL || right == NULL) {
        return 0;
    }
#ifdef _WIN32
    return _stricmp(left, right) == 0;
#else
    return strcmp(left, right) == 0;
#endif
}


#ifdef _WIN32
static int gs_windows_path_has_reparse_component(const char *path)
{
    char copy[GS_RUN_PATH_LENGTH];
    size_t length;
    size_t index;

    if (path == NULL) {
        return 1;
    }
    length = strlen(path);
    if (length < 3u || length >= sizeof(copy) ||
        path[1] != ':' || path[2] != '/') {

        return 1;
    }
    memcpy(copy, path, length + 1u);
    for (index = 3u; index <= length; index++) {
        if (copy[index] == '/' || copy[index] == '\0') {
            char saved = copy[index];
            DWORD attributes;

            copy[index] = '\0';
            attributes = GetFileAttributesA(copy);
            copy[index] = saved;
            if (attributes == INVALID_FILE_ATTRIBUTES ||
                (attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0u) {

                return 1;
            }
        }
    }
    return 0;
}
#endif


static int gs_canonicalize_existing_path(
    const char *path,
    int require_directory,
    char resolved[GS_RUN_PATH_LENGTH]
)
{
    struct stat information;
    size_t length;

    if (!gs_metadata_value_is_valid(path) || resolved == NULL) {
        return 1;
    }
#ifdef _WIN32
    if (_fullpath(resolved, path, GS_RUN_PATH_LENGTH) == NULL) {
        return 1;
    }
#else
    if (realpath(path, resolved) == NULL) {
        return 1;
    }
#endif
    gs_normalize_path_separators(resolved);
    length = strlen(resolved);
    while (length > 1u && resolved[length - 1u] == '/') {
#ifdef _WIN32
        if (length == 3u && resolved[1] == ':') {
            break;
        }
#endif
        resolved[--length] = '\0';
    }
    if (length == 0u || length >= GS_RUN_PATH_LENGTH ||
        stat(resolved, &information) != 0 ||
        (require_directory && !S_ISDIR(information.st_mode)) ||
        (!require_directory && !S_ISREG(information.st_mode))) {

        return 1;
    }
#ifdef _WIN32
    if (gs_windows_path_has_reparse_component(resolved) != 0) {
        return 1;
    }
#endif
    return 0;
}


static int gs_resolve_repository_layout(
    const char *pca_directory,
    const GSPCAProfile *profile,
    char canonical_pca[GS_RUN_PATH_LENGTH],
    char repository_root[GS_RUN_PATH_LENGTH],
    char canonical_base[GS_RUN_PATH_LENGTH]
)
{
    static const char prefix[] =
        "/sakai_multires_c/results/baseline/";
    char suffix[GS_RUN_PATH_LENGTH];
    char canonical_root[GS_RUN_PATH_LENGTH];
    char expected_baseline[GS_RUN_PATH_LENGTH];
    char expected_pca[GS_RUN_PATH_LENGTH];
    char canonical_expected_pca[GS_RUN_PATH_LENGTH];
    char expected_base[GS_RUN_PATH_LENGTH];
    size_t pca_length;
    size_t suffix_length;
    size_t root_length;
    int written;

    if (profile == NULL || canonical_pca == NULL ||
        repository_root == NULL || canonical_base == NULL ||
        gs_canonicalize_existing_path(
            pca_directory, 1, canonical_pca
        ) != 0) {

        return 1;
    }
    written = snprintf(
        suffix, sizeof(suffix), "%s%s", prefix, profile->profile
    );
    if (written < 0 || (size_t)written >= sizeof(suffix)) {
        return 1;
    }
    pca_length = strlen(canonical_pca);
    suffix_length = strlen(suffix);
    if (pca_length <= suffix_length) {
        return 1;
    }
    root_length = pca_length - suffix_length;
#ifdef _WIN32
    if (_stricmp(canonical_pca + root_length, suffix) != 0) {
#else
    if (strcmp(canonical_pca + root_length, suffix) != 0) {
#endif
        return 1;
    }
    if (root_length == 0u || root_length >= GS_RUN_PATH_LENGTH) {
        return 1;
    }
    memcpy(repository_root, canonical_pca, root_length);
    repository_root[root_length] = '\0';
    if (gs_canonicalize_existing_path(
            repository_root, 1, canonical_root
        ) != 0 ||
        !gs_paths_equal(repository_root, canonical_root) ||
        gs_path_join(
            repository_root,
            "sakai_multires_c/results/baseline/"
            "mini224_subset1000_pool14_no_pca",
            expected_base,
            sizeof(expected_base)
        ) != 0 ||
        gs_canonicalize_existing_path(
            expected_base, 1, canonical_base
        ) != 0 ||
        !gs_paths_equal(expected_base, canonical_base) ||
        gs_path_join(
            repository_root,
            "sakai_multires_c/results/baseline",
            expected_baseline,
            sizeof(expected_baseline)
        ) != 0 ||
        gs_path_join(
            expected_baseline, profile->profile,
            expected_pca, sizeof(expected_pca)
        ) != 0 ||
        gs_canonicalize_existing_path(
            expected_pca, 1, canonical_expected_pca
        ) != 0 ||
        !gs_paths_equal(canonical_pca, canonical_expected_pca)) {

        return 1;
    }
    return 0;
}


static int gs_path_state(const char *path)
{
    struct stat information;

    if (path == NULL) {
        return -1;
    }

    errno = 0;
    if (stat(path, &information) == 0) {
        return 1;
    }
    return errno == ENOENT ? 0 : -1;
}


static int gs_file_size(const char *path, size_t *size)
{
    struct stat information;

    if (path == NULL || size == NULL ||
        stat(path, &information) != 0 ||
        !S_ISREG(information.st_mode) ||
        information.st_size < 0 ||
        (uintmax_t)information.st_size > (uintmax_t)SIZE_MAX) {

        return 1;
    }

    *size = (size_t)information.st_size;
    return 0;
}


static void gs_remove_line_ending(char *line)
{
    size_t length;

    if (line == NULL) {
        return;
    }

    length = strlen(line);
    while (length != 0u &&
           (line[length - 1u] == '\r' || line[length - 1u] == '\n')) {
        line[--length] = '\0';
    }
}


static int gs_name_in_list(
    const char *name,
    const char *const *names,
    size_t count,
    size_t *matched_index
)
{
    size_t index;

    if (name == NULL || names == NULL) {
        return 0;
    }

    for (index = 0; index < count; index++) {
        if (strcmp(name, names[index]) == 0) {
            if (matched_index != NULL) {
                *matched_index = index;
            }
            return 1;
        }
    }
    return 0;
}


static int gs_exact_inventory_is_valid(
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

    if (directory == NULL || names == NULL || count == 0u) {
        return 1;
    }

    seen = (unsigned char *)calloc(count, sizeof(unsigned char));
    if (seen == NULL) {
        return 1;
    }

    stream = opendir(directory);
    if (stream == NULL) {
        free(seen);
        return 1;
    }

    while (!failed) {
        size_t matched_index;
        char path[GS_RUN_PATH_LENGTH];
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

        if (!gs_name_in_list(entry->d_name, names, count, &matched_index) ||
            seen[matched_index] != 0u ||
            gs_path_join(
                directory, entry->d_name, path, sizeof(path)
            ) != 0 ||
            stat(path, &information) != 0 ||
            !S_ISREG(information.st_mode)) {

            failed = 1;
            break;
        }

        seen[matched_index] = 1u;
    }

    if (closedir(stream) != 0) {
        failed = 1;
    }

    for (index = 0; !failed && index < count; index++) {
        if (seen[index] == 0u) {
            failed = 1;
        }
    }

    free(seen);
    return failed;
}


static int gs_pca_inventory_is_valid(
    const char *directory,
    const GSPCAProfile *profile
)
{
    unsigned char seen[GS_RUN_PCA_FILE_COUNT] = {0};
    const char *names[GS_RUN_PCA_FILE_COUNT];
    DIR *stream = NULL;
    struct dirent *entry;
    size_t index;
    int kmeans_seen = 0;
    int evaluation_seen = 0;
    int failed = 0;

    if (directory == NULL || profile == NULL ||
        strstr(gs_path_basename(directory), GS_RUN_PARTIAL_SUFFIX) != NULL) {
        return 1;
    }
    for (index = 0; index < 8u; index++) {
        names[index] = gs_pca_fixed_names[index];
    }
    names[8] = profile->train_filename;
    names[9] = profile->validation_filename;
    names[10] = gs_pca_fixed_names[8];
    names[11] = gs_pca_fixed_names[9];

    stream = opendir(directory);
    if (stream == NULL) {
        return 1;
    }
    while (!failed) {
        size_t matched_index;
        char path[GS_RUN_PATH_LENGTH];
        struct stat information;

        errno = 0;
        entry = readdir(stream);
        if (entry == NULL) {
            failed = errno != 0;
            break;
        }
        if (strcmp(entry->d_name, ".") == 0 ||
            strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        if (gs_path_join(directory, entry->d_name, path, sizeof(path)) != 0 ||
            stat(path, &information) != 0) {
            failed = 1;
            break;
        }
        if (gs_name_in_list(
                entry->d_name, names, GS_RUN_PCA_FILE_COUNT, &matched_index
            )) {
            if (seen[matched_index] != 0u ||
                !S_ISREG(information.st_mode)) {
                failed = 1;
            } else {
                seen[matched_index] = 1u;
            }
        } else if (strcmp(entry->d_name, "kmeans_seed123") == 0) {
            if (kmeans_seen || !S_ISDIR(information.st_mode)) {
                failed = 1;
            }
            kmeans_seen = 1;
        } else if (strcmp(entry->d_name, "label_evaluation_seed123") == 0) {
            if (evaluation_seen || !S_ISDIR(information.st_mode)) {
                failed = 1;
            }
            evaluation_seen = 1;
        } else {
            failed = 1;
        }
    }
    if (closedir(stream) != 0) {
        failed = 1;
    }
    for (index = 0; !failed && index < GS_RUN_PCA_FILE_COUNT; index++) {
        if (seen[index] == 0u) {
            failed = 1;
        }
    }
    return failed;
}


static int gs224_base_inventory_is_valid(const char *directory)
{
    unsigned char seen[GS_RUN_BASE_FILE_COUNT] = {0};
    DIR *stream;
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
        char path[GS_RUN_PATH_LENGTH];
        struct stat information;
        size_t matched_index;

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
            ) != 0 ||
            stat(path, &information) != 0) {

            failed = 1;
            break;
        }
        if (gs_name_in_list(
                entry->d_name,
                gs224_base_all_names,
                GS_RUN_BASE_FILE_COUNT,
                &matched_index
            )) {

            if (seen[matched_index] != 0u ||
                !S_ISREG(information.st_mode)) {

                failed = 1;
            } else {
                seen[matched_index] = 1u;
            }
        } else if (strcmp(entry->d_name, "kmeans_seed123") == 0) {
            if (kmeans_seen || !S_ISDIR(information.st_mode)) {
                failed = 1;
            }
            kmeans_seen = 1;
        } else if (strcmp(
                       entry->d_name, "label_evaluation_seed123"
                   ) == 0) {

            if (evaluation_seen || !S_ISDIR(information.st_mode)) {
                failed = 1;
            }
            evaluation_seen = 1;
        } else {
            failed = 1;
        }
    }
    if (closedir(stream) != 0) {
        failed = 1;
    }
    for (index = 0u;
         !failed && index < GS_RUN_BASE_FILE_COUNT;
         index++) {

        if (seen[index] == 0u) {
            failed = 1;
        }
    }
    return failed;
}


#if 0
static int gs_base_inventory_is_valid(
    const char *directory,
    const char *const *names,
    size_t count
)
{
    return gs_exact_inventory_is_valid(directory, names, count);
}
#endif


static uint32_t gs_sha256_rotate_right(uint32_t value, unsigned int count)
{
    return (value >> count) | (value << (32u - count));
}


static void gs_sha256_transform(
    GSSha256 *context,
    const unsigned char block[64]
)
{
    static const uint32_t constants[64] = {
        UINT32_C(0x428a2f98), UINT32_C(0x71374491),
        UINT32_C(0xb5c0fbcf), UINT32_C(0xe9b5dba5),
        UINT32_C(0x3956c25b), UINT32_C(0x59f111f1),
        UINT32_C(0x923f82a4), UINT32_C(0xab1c5ed5),
        UINT32_C(0xd807aa98), UINT32_C(0x12835b01),
        UINT32_C(0x243185be), UINT32_C(0x550c7dc3),
        UINT32_C(0x72be5d74), UINT32_C(0x80deb1fe),
        UINT32_C(0x9bdc06a7), UINT32_C(0xc19bf174),
        UINT32_C(0xe49b69c1), UINT32_C(0xefbe4786),
        UINT32_C(0x0fc19dc6), UINT32_C(0x240ca1cc),
        UINT32_C(0x2de92c6f), UINT32_C(0x4a7484aa),
        UINT32_C(0x5cb0a9dc), UINT32_C(0x76f988da),
        UINT32_C(0x983e5152), UINT32_C(0xa831c66d),
        UINT32_C(0xb00327c8), UINT32_C(0xbf597fc7),
        UINT32_C(0xc6e00bf3), UINT32_C(0xd5a79147),
        UINT32_C(0x06ca6351), UINT32_C(0x14292967),
        UINT32_C(0x27b70a85), UINT32_C(0x2e1b2138),
        UINT32_C(0x4d2c6dfc), UINT32_C(0x53380d13),
        UINT32_C(0x650a7354), UINT32_C(0x766a0abb),
        UINT32_C(0x81c2c92e), UINT32_C(0x92722c85),
        UINT32_C(0xa2bfe8a1), UINT32_C(0xa81a664b),
        UINT32_C(0xc24b8b70), UINT32_C(0xc76c51a3),
        UINT32_C(0xd192e819), UINT32_C(0xd6990624),
        UINT32_C(0xf40e3585), UINT32_C(0x106aa070),
        UINT32_C(0x19a4c116), UINT32_C(0x1e376c08),
        UINT32_C(0x2748774c), UINT32_C(0x34b0bcb5),
        UINT32_C(0x391c0cb3), UINT32_C(0x4ed8aa4a),
        UINT32_C(0x5b9cca4f), UINT32_C(0x682e6ff3),
        UINT32_C(0x748f82ee), UINT32_C(0x78a5636f),
        UINT32_C(0x84c87814), UINT32_C(0x8cc70208),
        UINT32_C(0x90befffa), UINT32_C(0xa4506ceb),
        UINT32_C(0xbef9a3f7), UINT32_C(0xc67178f2)
    };
    uint32_t words[64];
    uint32_t a;
    uint32_t b;
    uint32_t c;
    uint32_t d;
    uint32_t e;
    uint32_t f;
    uint32_t g;
    uint32_t h;
    size_t index;

    for (index = 0; index < 16u; index++) {
        size_t offset = index * 4u;
        words[index] =
            ((uint32_t)block[offset] << 24) |
            ((uint32_t)block[offset + 1u] << 16) |
            ((uint32_t)block[offset + 2u] << 8) |
            (uint32_t)block[offset + 3u];
    }

    for (index = 16u; index < 64u; index++) {
        uint32_t s0 =
            gs_sha256_rotate_right(words[index - 15u], 7u) ^
            gs_sha256_rotate_right(words[index - 15u], 18u) ^
            (words[index - 15u] >> 3);
        uint32_t s1 =
            gs_sha256_rotate_right(words[index - 2u], 17u) ^
            gs_sha256_rotate_right(words[index - 2u], 19u) ^
            (words[index - 2u] >> 10);

        words[index] =
            words[index - 16u] + s0 + words[index - 7u] + s1;
    }

    a = context->state[0];
    b = context->state[1];
    c = context->state[2];
    d = context->state[3];
    e = context->state[4];
    f = context->state[5];
    g = context->state[6];
    h = context->state[7];

    for (index = 0; index < 64u; index++) {
        uint32_t sum1 =
            gs_sha256_rotate_right(e, 6u) ^
            gs_sha256_rotate_right(e, 11u) ^
            gs_sha256_rotate_right(e, 25u);
        uint32_t choice = (e & f) ^ ((~e) & g);
        uint32_t temporary1 =
            h + sum1 + choice + constants[index] + words[index];
        uint32_t sum0 =
            gs_sha256_rotate_right(a, 2u) ^
            gs_sha256_rotate_right(a, 13u) ^
            gs_sha256_rotate_right(a, 22u);
        uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temporary2 = sum0 + majority;

        h = g;
        g = f;
        f = e;
        e = d + temporary1;
        d = c;
        c = b;
        b = a;
        a = temporary1 + temporary2;
    }

    context->state[0] += a;
    context->state[1] += b;
    context->state[2] += c;
    context->state[3] += d;
    context->state[4] += e;
    context->state[5] += f;
    context->state[6] += g;
    context->state[7] += h;
}


static void gs_sha256_init(GSSha256 *context)
{
    context->state[0] = UINT32_C(0x6a09e667);
    context->state[1] = UINT32_C(0xbb67ae85);
    context->state[2] = UINT32_C(0x3c6ef372);
    context->state[3] = UINT32_C(0xa54ff53a);
    context->state[4] = UINT32_C(0x510e527f);
    context->state[5] = UINT32_C(0x9b05688c);
    context->state[6] = UINT32_C(0x1f83d9ab);
    context->state[7] = UINT32_C(0x5be0cd19);
    context->completed_bits = 0u;
    context->block_size = 0u;
}


static int gs_sha256_update(
    GSSha256 *context,
    const unsigned char *data,
    size_t size
)
{
    size_t index;

    if (context == NULL || (data == NULL && size != 0u)) {
        return 1;
    }

    for (index = 0; index < size; index++) {
        context->block[context->block_size++] = data[index];
        if (context->block_size == 64u) {
            if (context->completed_bits > UINT64_MAX - UINT64_C(512)) {
                return 1;
            }
            gs_sha256_transform(context, context->block);
            context->completed_bits += UINT64_C(512);
            context->block_size = 0u;
        }
    }
    return 0;
}


static int gs_sha256_final(
    GSSha256 *context,
    unsigned char digest[32]
)
{
    uint64_t total_bits;
    size_t index;

    if (context == NULL || digest == NULL ||
        context->block_size >
            (size_t)((UINT64_MAX - context->completed_bits) / 8u)) {

        return 1;
    }

    total_bits =
        context->completed_bits + (uint64_t)context->block_size * 8u;
    context->block[context->block_size++] = 0x80u;

    if (context->block_size > 56u) {
        while (context->block_size < 64u) {
            context->block[context->block_size++] = 0u;
        }
        gs_sha256_transform(context, context->block);
        context->block_size = 0u;
    }

    while (context->block_size < 56u) {
        context->block[context->block_size++] = 0u;
    }
    for (index = 0; index < 8u; index++) {
        context->block[63u - index] =
            (unsigned char)(total_bits >> (index * 8u));
    }
    gs_sha256_transform(context, context->block);

    for (index = 0; index < 8u; index++) {
        digest[index * 4u] =
            (unsigned char)(context->state[index] >> 24);
        digest[index * 4u + 1u] =
            (unsigned char)(context->state[index] >> 16);
        digest[index * 4u + 2u] =
            (unsigned char)(context->state[index] >> 8);
        digest[index * 4u + 3u] =
            (unsigned char)context->state[index];
    }
    return 0;
}


static int gs_sha256_file(
    const char *path,
    char output[GS_RUN_SHA256_LENGTH + 1]
)
{
    static const char digits[] = "0123456789abcdef";
    unsigned char buffer[65536];
    unsigned char digest[32];
    GSSha256 context;
    FILE *file;
    size_t count;
    size_t index;
    int failed = 0;

    if (path == NULL || output == NULL) {
        return 1;
    }

    file = fopen(path, "rb");
    if (file == NULL) {
        return 1;
    }
    gs_sha256_init(&context);

    while ((count = fread(buffer, 1u, sizeof(buffer), file)) != 0u) {
        if (gs_sha256_update(&context, buffer, count) != 0) {
            failed = 1;
            break;
        }
    }
    if (ferror(file)) {
        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    if (failed || gs_sha256_final(&context, digest) != 0) {
        return 1;
    }

    for (index = 0; index < sizeof(digest); index++) {
        output[index * 2u] = digits[digest[index] >> 4];
        output[index * 2u + 1u] = digits[digest[index] & 0x0fu];
    }
    output[GS_RUN_SHA256_LENGTH] = '\0';
    return 0;
}


static int gs_sha_text_is_valid(const char *hash)
{
    size_t index;

    if (hash == NULL || strlen(hash) != GS_RUN_SHA256_LENGTH) {
        return 0;
    }
    for (index = 0; index < GS_RUN_SHA256_LENGTH; index++) {
        if (!((hash[index] >= '0' && hash[index] <= '9') ||
              (hash[index] >= 'a' && hash[index] <= 'f'))) {
            return 0;
        }
    }
    return 1;
}


static int gs_parse_sha_line(
    char *line,
    const char *expected_kind,
    const char *expected_name,
    char hash[GS_RUN_SHA256_LENGTH + 1]
)
{
    char *first_space;
    char *second_space;
    char *kind;
    char *name;

    if (line == NULL || expected_kind == NULL || expected_name == NULL ||
        hash == NULL) {
        return 1;
    }

    gs_remove_line_ending(line);
    first_space = strchr(line, ' ');
    if (first_space == NULL) {
        return 1;
    }
    *first_space = '\0';
    kind = line;
    second_space = strchr(first_space + 1, ' ');
    if (second_space == NULL) {
        return 1;
    }
    *second_space = '\0';
    name = second_space + 1;

    if (strchr(name, ' ') != NULL || strchr(name, '/') != NULL ||
        strchr(name, '\\') != NULL ||
        strcmp(kind, expected_kind) != 0 ||
        strcmp(name, expected_name) != 0 ||
        !gs_sha_text_is_valid(first_space + 1)) {

        return 1;
    }

    memcpy(hash, first_space + 1, GS_RUN_SHA256_LENGTH + 1u);
    return 0;
}


#if 0
static int gs_verify_base_sha256(
    const char *base_directory,
    char cluster_hashes[GS_RUN_CLUSTER_INPUT_COUNT][
        GS_RUN_SHA256_LENGTH + 1
    ]
)
{
    char sha_path[GS_RUN_PATH_LENGTH];
    char line[GS_RUN_LINE_LENGTH];
    FILE *file;
    size_t index;
    int failed = 0;

    if (base_directory == NULL || cluster_hashes == NULL ||
        gs_path_join(
            base_directory, "sha256.txt", sha_path, sizeof(sha_path)
        ) != 0) {
        return 1;
    }

    memset(
        cluster_hashes,
        0,
        GS_RUN_CLUSTER_INPUT_COUNT * (GS_RUN_SHA256_LENGTH + 1u)
    );

    file = fopen(sha_path, "rb");
    if (file == NULL) {
        return 1;
    }

    for (index = 0;
         !failed && index < GS_RUN_BASE_INPUT_COUNT;
         index++) {

        char recorded[GS_RUN_SHA256_LENGTH + 1];
        char actual[GS_RUN_SHA256_LENGTH + 1];

        if (fgets(line, sizeof(line), file) == NULL ||
            gs_parse_sha_line(
                line, "INPUT", gs_base_input_names[index], recorded
            ) != 0 ||
            gs_sha256_file(gs_base_input_names[index], actual) != 0 ||
            strcmp(recorded, actual) != 0) {

            failed = 1;
        }
    }

    for (index = 0;
         !failed && index < GS_RUN_BASE_ARTIFACT_COUNT;
         index++) {

        char recorded[GS_RUN_SHA256_LENGTH + 1];
        char actual[GS_RUN_SHA256_LENGTH + 1];
        char artifact_path[GS_RUN_PATH_LENGTH];
        size_t cluster_index;

        if (fgets(line, sizeof(line), file) == NULL ||
            gs_parse_sha_line(
                line, "ARTIFACT", gs_base_artifact_names[index], recorded
            ) != 0 ||
            gs_path_join(
                base_directory,
                gs_base_artifact_names[index],
                artifact_path,
                sizeof(artifact_path)
            ) != 0 ||
            gs_sha256_file(artifact_path, actual) != 0 ||
            strcmp(recorded, actual) != 0) {

            failed = 1;
            break;
        }

        if (gs_name_in_list(
                gs_base_artifact_names[index],
                gs_cluster_input_names,
                GS_RUN_CLUSTER_INPUT_COUNT,
                &cluster_index
            )) {

            memcpy(
                cluster_hashes[cluster_index],
                actual,
                GS_RUN_SHA256_LENGTH + 1u
            );
        }
    }

    if (!failed && fgets(line, sizeof(line), file) != NULL) {
        failed = 1;
    }
    if (ferror(file)) {
        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }

    for (index = 0;
         !failed && index < GS_RUN_CLUSTER_INPUT_COUNT;
         index++) {

        if (!gs_sha_text_is_valid(cluster_hashes[index])) {
            failed = 1;
        }
    }

    return failed;
}
#endif


static int gs_split_exact(
    char *line,
    char **fields,
    size_t expected_fields
)
{
    size_t count = 0u;
    char *start;
    char *cursor;

    if (line == NULL || fields == NULL || expected_fields == 0u) {
        return 1;
    }

    start = line;
    cursor = line;
    while (1) {
        if (*cursor == ',' || *cursor == '\0') {
            if (count >= expected_fields) {
                return 1;
            }
            fields[count++] = start;
            if (*cursor == '\0') {
                break;
            }
            *cursor = '\0';
            start = cursor + 1;
        }
        cursor++;
    }

    return count == expected_fields ? 0 : 1;
}


#if 0
static int gs_validate_row_mapping(
    const char *path,
    size_t expected_rows,
    const char *expected_split
)
{
    static const char header[] =
        "row_index,split_pos,manifest_index,event_index,event_id,split";
    char line[GS_RUN_LINE_LENGTH];
    FILE *file;
    size_t row = 0u;
    int failed = 0;

    if (path == NULL || expected_split == NULL) {
        return 1;
    }

    file = fopen(path, "rb");
    if (file == NULL) {
        return 1;
    }

    if (fgets(line, sizeof(line), file) == NULL) {
        failed = 1;
    } else {
        gs_remove_line_ending(line);
        if (strcmp(line, header) != 0) {
            failed = 1;
        }
    }

    while (!failed && fgets(line, sizeof(line), file) != NULL) {
        char *fields[6];
        size_t row_index;
        size_t split_pos;
        size_t manifest_index;
        size_t event_index;

        if (strchr(line, '\n') == NULL && !feof(file)) {
            failed = 1;
            break;
        }
        gs_remove_line_ending(line);

        if (row >= expected_rows ||
            gs_split_exact(line, fields, 6u) != 0 ||
            gs_parse_size_value(fields[0], &row_index) != 0 ||
            gs_parse_size_value(fields[1], &split_pos) != 0 ||
            gs_parse_size_value(fields[2], &manifest_index) != 0 ||
            gs_parse_size_value(fields[3], &event_index) != 0 ||
            row_index != row || split_pos != row ||
            manifest_index >= 8583u ||
            event_index == 0u || event_index > 8583u ||
            fields[4][0] == '\0' ||
            strcmp(fields[5], expected_split) != 0) {

            failed = 1;
        }
        row++;
    }

    if (ferror(file) || row != expected_rows) {
        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    return failed;
}


static int gs_validate_feature_spec(const char *path)
{
    static const char *const keys[] = {
        "profile",
        "channels",
        "image_side",
        "pool_side",
        "pooled_side",
        "feature_count",
        "matrix_dtype",
        "matrix_layout",
        "pca",
        "augmentation",
        "labels_used",
        "scaler_fit"
    };
    static const char *const values[] = {
        "mini224_subset1000_pool14_no_pca",
        "4",
        "224",
        "14",
        "16",
        "1024",
        "float32_ieee754",
        "row_major",
        "none",
        "none",
        "false",
        "train_only"
    };
    unsigned char seen[sizeof(keys) / sizeof(keys[0])] = {0};
    char line[GS_RUN_LINE_LENGTH];
    FILE *file;
    int failed = 0;

    if (path == NULL) {
        return 1;
    }

    file = fopen(path, "rb");
    if (file == NULL) {
        return 1;
    }

    while (!failed && fgets(line, sizeof(line), file) != NULL) {
        char *equals;
        size_t index;

        if (strchr(line, '\n') == NULL && !feof(file)) {
            failed = 1;
            break;
        }
        gs_remove_line_ending(line);
        equals = strchr(line, '=');
        if (equals == NULL || equals == line ||
            strchr(equals + 1, '=') != NULL) {
            failed = 1;
            break;
        }
        *equals = '\0';

        for (index = 0; index < sizeof(keys) / sizeof(keys[0]); index++) {
            if (strcmp(line, keys[index]) == 0) {
                if (seen[index] != 0u ||
                    strcmp(equals + 1, values[index]) != 0) {
                    failed = 1;
                }
                seen[index] = 1u;
                break;
            }
        }
    }

    if (ferror(file)) {
        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }

    {
        size_t index;
        for (index = 0;
             !failed && index < sizeof(keys) / sizeof(keys[0]);
             index++) {
            if (seen[index] == 0u) {
                failed = 1;
            }
        }
    }
    return failed;
}


static int gs_build_cluster_input_paths(
    const char *train_path,
    const char *validation_path,
    char base_directory[GS_RUN_PATH_LENGTH],
    char storage[GS_RUN_CLUSTER_INPUT_COUNT][GS_RUN_PATH_LENGTH],
    const char *paths[GS_RUN_CLUSTER_INPUT_COUNT]
)
{
    char validation_directory[GS_RUN_PATH_LENGTH];
    size_t index;

    if (strcmp(
            gs_path_basename(train_path),
            "train_features_std_f32.bin"
        ) != 0 ||
        strcmp(
            gs_path_basename(validation_path),
            "val_features_std_f32.bin"
        ) != 0 ||
        gs_parent_directory(
            train_path, base_directory, GS_RUN_PATH_LENGTH
        ) != 0 ||
        gs_parent_directory(
            validation_path,
            validation_directory,
            sizeof(validation_directory)
        ) != 0 ||
        !gs_directories_are_same(
            base_directory, validation_directory
        )) {

        return 1;
    }

    for (index = 0; index < GS_RUN_CLUSTER_INPUT_COUNT; index++) {
        if (gs_path_join(
                base_directory,
                gs_cluster_input_names[index],
                storage[index],
                GS_RUN_PATH_LENGTH
            ) != 0) {

            return 1;
        }
        paths[index] = storage[index];
    }

    return 0;
}


static int gs_validate_base_contract(
    const char *base_directory,
    const char *const cluster_paths[GS_RUN_CLUSTER_INPUT_COUNT],
    char cluster_hashes[GS_RUN_CLUSTER_INPUT_COUNT][
        GS_RUN_SHA256_LENGTH + 1
    ]
)
{
    static const size_t expected_sizes[GS_RUN_CLUSTER_INPUT_COUNT] = {
        958u,
        54237u,
        28123136u,
        7032832u,
        270713u,
        74659u
    };
    size_t index;

    if (gs_base_inventory_is_valid(
            base_directory,
            gs_base_all_names,
            GS_RUN_BASE_FILE_COUNT
        ) != 0 ||
        gs_verify_base_sha256(base_directory, cluster_hashes) != 0) {

        return 1;
    }

    for (index = 0; index < GS_RUN_CLUSTER_INPUT_COUNT; index++) {
        size_t actual_size;
        char actual_hash[GS_RUN_SHA256_LENGTH + 1];

        if (strcmp(
                gs_path_basename(cluster_paths[index]),
                gs_cluster_input_names[index]
            ) != 0 ||
            gs_file_size(cluster_paths[index], &actual_size) != 0 ||
            actual_size != expected_sizes[index] ||
            gs_sha256_file(cluster_paths[index], actual_hash) != 0 ||
            strcmp(actual_hash, cluster_hashes[index]) != 0) {

            return 1;
        }
    }

    return gs_validate_feature_spec(cluster_paths[0]) ||
           gs_validate_row_mapping(
               cluster_paths[4], GS_RUN_TRAIN_ROWS, "train"
           ) ||
           gs_validate_row_mapping(
               cluster_paths[5], GS_RUN_VALIDATION_ROWS, "validation"
           );
}
#endif


static int gs_parse_double_value(const char *text, double *value)
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


static int gs_read_key_value(
    const char *path,
    const char *key,
    char *value,
    size_t value_size
)
{
    char line[GS_RUN_LINE_LENGTH];
    FILE *file;
    int found = 0;
    int failed = 0;

    if (path == NULL || key == NULL || value == NULL || value_size == 0u) {
        return 1;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        return 1;
    }
    while (!failed && fgets(line, sizeof(line), file) != NULL) {
        char *equals;
        size_t length;
        if (strchr(line, '\n') == NULL && !feof(file)) {
            failed = 1;
            break;
        }
        gs_remove_line_ending(line);
        equals = strchr(line, '=');
        if (equals == NULL || equals == line ||
            strchr(equals + 1, '=') != NULL) {
            failed = 1;
            break;
        }
        *equals = '\0';
        if (strcmp(line, key) == 0) {
            length = strlen(equals + 1);
            if (found || length == 0u || length >= value_size ||
                !gs_metadata_value_is_valid(equals + 1)) {
                failed = 1;
                break;
            }
            memcpy(value, equals + 1, length + 1u);
            found = 1;
        }
    }
    if (ferror(file) || fclose(file) != 0) {
        failed = 1;
    }
    return failed || !found;
}


static int gs_expect_key_value(
    const char *path,
    const char *key,
    const char *expected
)
{
    char value[GS_RUN_LINE_LENGTH];
    return gs_read_key_value(path, key, value, sizeof(value)) != 0 ||
           strcmp(value, expected) != 0;
}


static int gs224_validate_base_contract(
    const char *repository_root,
    const char *base_directory
)
{
    char manifest_path[GS_RUN_PATH_LENGTH];
    char feature_spec_path[GS_RUN_PATH_LENGTH];
    char summary_path[GS_RUN_PATH_LENGTH];
    char line[GS_RUN_LINE_LENGTH];
    FILE *file;
    size_t index;
    int failed = 0;

    if (repository_root == NULL || base_directory == NULL ||
        gs224_base_inventory_is_valid(base_directory) != 0 ||
        gs_path_join(
            base_directory, "sha256.txt",
            manifest_path, sizeof(manifest_path)
        ) != 0 ||
        gs_path_join(
            base_directory, "feature_spec.txt",
            feature_spec_path, sizeof(feature_spec_path)
        ) != 0 ||
        gs_path_join(
            base_directory, "summary.txt",
            summary_path, sizeof(summary_path)
        ) != 0 ||
        gs_expect_key_value(
            feature_spec_path, "profile",
            "mini224_subset1000_pool14_no_pca"
        ) != 0 ||
        gs_expect_key_value(
            feature_spec_path, "subset_profile",
            "subset224_seed123_prefix_800_200"
        ) != 0 ||
        gs_expect_key_value(
            feature_spec_path, "feature_count", "1024"
        ) != 0 ||
        gs_expect_key_value(
            feature_spec_path, "train_rows", "800"
        ) != 0 ||
        gs_expect_key_value(
            feature_spec_path, "validation_rows", "200"
        ) != 0 ||
        gs_expect_key_value(
            feature_spec_path, "scaler_fit", "train_only"
        ) != 0 ||
        gs_expect_key_value(
            feature_spec_path, "pca", "none"
        ) != 0 ||
        gs_expect_key_value(
            feature_spec_path, "labels_used", "false"
        ) != 0 ||
        gs_expect_key_value(
            feature_spec_path, "code_commit",
            GS_RUN_BASE_CODE_COMMIT
        ) != 0 ||
        gs_expect_key_value(
            summary_path, "profile",
            "mini224_subset1000_pool14_no_pca"
        ) != 0 ||
        gs_expect_key_value(
            summary_path, "code_commit",
            GS_RUN_BASE_CODE_COMMIT
        ) != 0 ||
        gs_expect_key_value(
            summary_path, "standardization_fit", "train_only"
        ) != 0 ||
        gs_expect_key_value(
            summary_path, "validation_transformed_without_refit", "true"
        ) != 0 ||
        gs_expect_key_value(
            summary_path, "labels_used", "false"
        ) != 0 ||
        gs_expect_key_value(
            summary_path, "pca_used", "false"
        ) != 0) {

        return 1;
    }

    file = fopen(manifest_path, "rb");
    if (file == NULL) {
        return 1;
    }
    for (index = 0u;
         !failed && index < GS_RUN_BASE_INPUT_COUNT;
         index++) {

        char recorded[GS_RUN_SHA256_LENGTH + 1u];
        char actual[GS_RUN_SHA256_LENGTH + 1u];
        char expected[GS_RUN_PATH_LENGTH];
        char canonical[GS_RUN_PATH_LENGTH];

        if (fgets(line, sizeof(line), file) == NULL ||
            gs_parse_sha_line(
                line, "INPUT", gs224_base_input_names[index], recorded
            ) != 0 ||
            gs_path_join(
                repository_root,
                gs224_base_input_relative_paths[index],
                expected,
                sizeof(expected)
            ) != 0 ||
            gs_canonicalize_existing_path(
                expected, 0, canonical
            ) != 0 ||
            !gs_paths_equal(expected, canonical) ||
            gs_sha256_file(canonical, actual) != 0 ||
            strcmp(recorded, actual) != 0) {

            failed = 1;
        }
    }
    for (index = 0u;
         !failed && index < GS_RUN_BASE_ARTIFACT_COUNT;
         index++) {

        char recorded[GS_RUN_SHA256_LENGTH + 1u];
        char actual[GS_RUN_SHA256_LENGTH + 1u];
        char path[GS_RUN_PATH_LENGTH];
        char canonical[GS_RUN_PATH_LENGTH];

        if (fgets(line, sizeof(line), file) == NULL ||
            gs_parse_sha_line(
                line, "ARTIFACT",
                gs224_base_artifact_names[index],
                recorded
            ) != 0 ||
            gs_path_join(
                base_directory,
                gs224_base_artifact_names[index],
                path,
                sizeof(path)
            ) != 0 ||
            gs_canonicalize_existing_path(path, 0, canonical) != 0 ||
            !gs_paths_equal(path, canonical) ||
            gs_sha256_file(canonical, actual) != 0 ||
            strcmp(recorded, actual) != 0) {

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


static int gs_select_pca_profile(
    const char *token,
    GSPCAProfile *profile
)
{
    int written;

    if (token == NULL || profile == NULL ||
        (strcmp(token, "pca90") != 0 &&
         strcmp(token, "pca95") != 0 &&
         strcmp(token, "pca99") != 0)) {
        return 1;
    }
    memset(profile, 0, sizeof(*profile));
    profile->token = token;
    profile->threshold =
        strcmp(token, "pca90") == 0 ? 0.90 :
        strcmp(token, "pca95") == 0 ? 0.95 : 0.99;
    written = snprintf(
        profile->profile, sizeof(profile->profile),
        "mini224_subset1000_pool14_%s", token
    );
    if (written < 0 || (size_t)written >= sizeof(profile->profile)) {
        return 1;
    }
    written = snprintf(
        profile->train_filename, sizeof(profile->train_filename),
        "train_scores_%s_f32.bin", token
    );
    if (written < 0 ||
        (size_t)written >= sizeof(profile->train_filename)) {
        return 1;
    }
    written = snprintf(
        profile->validation_filename,
        sizeof(profile->validation_filename),
        "val_scores_%s_f32.bin", token
    );
    return written < 0 ||
           (size_t)written >= sizeof(profile->validation_filename);
}


static int gs_read_pca_thresholds(
    const char *directory,
    GSPCAProfile *profile
)
{
    static const double expected[3] = {0.90, 0.95, 0.99};
    char path[GS_RUN_PATH_LENGTH];
    char line[GS_RUN_LINE_LENGTH];
    size_t counts[3] = {0u, 0u, 0u};
    double cumulative[3] = {0.0, 0.0, 0.0};
    FILE *file;
    size_t row;
    int failed = 0;

    if (profile == NULL ||
        gs_path_join(directory, "thresholds.csv", path, sizeof(path)) != 0) {
        return 1;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        return 1;
    }
    if (fgets(line, sizeof(line), file) == NULL) {
        failed = 1;
    } else {
        gs_remove_line_ending(line);
        failed = strcmp(
            line, "threshold,component_count,cumulative_variance"
        ) != 0;
    }
    for (row = 0u; !failed && row < 3u; row++) {
        char *fields[3];
        double threshold;
        if (fgets(line, sizeof(line), file) == NULL ||
            (strchr(line, '\n') == NULL && !feof(file))) {
            failed = 1;
            break;
        }
        gs_remove_line_ending(line);
        if (gs_split_exact(line, fields, 3u) != 0 ||
            gs_parse_double_value(fields[0], &threshold) != 0 ||
            gs_parse_size_value(fields[1], &counts[row]) != 0 ||
            gs_parse_double_value(fields[2], &cumulative[row]) != 0 ||
            threshold != expected[row] || counts[row] == 0u ||
            counts[row] > GS_RUN_ORIGINAL_FEATURE_COUNT ||
            cumulative[row] < threshold || cumulative[row] > 1.0 ||
            (row != 0u && (counts[row] < counts[row - 1u] ||
                           cumulative[row] < cumulative[row - 1u]))) {
            failed = 1;
        }
    }
    if (!failed && fgets(line, sizeof(line), file) != NULL) {
        failed = 1;
    }
    if (ferror(file) || fclose(file) != 0) {
        failed = 1;
    }
    if (failed || counts[0] != 15u || counts[1] != 38u ||
        counts[2] != 168u) {
        return 1;
    }
    profile->d90 = counts[0];
    profile->d95 = counts[1];
    profile->d99 = counts[2];
    if (strcmp(profile->token, "pca90") == 0) {
        profile->components = counts[0];
        profile->cumulative = cumulative[0];
    } else if (strcmp(profile->token, "pca95") == 0) {
        profile->components = counts[1];
        profile->cumulative = cumulative[1];
    } else {
        profile->components = counts[2];
        profile->cumulative = cumulative[2];
    }
    return 0;
}


static int gs_validate_pca_metadata(
    const char *directory,
    GSPCAProfile *profile
)
{
    char spec_path[GS_RUN_PATH_LENGTH];
    char summary_path[GS_RUN_PATH_LENGTH];
    char value[GS_RUN_LINE_LENGTH];
    char expected[128];
    size_t parsed_size;
    double parsed_double;
    int written;

    if (directory == NULL || profile == NULL ||
        strcmp(gs_path_basename(directory), profile->profile) != 0 ||
        gs_path_join(directory, "pca_spec.txt", spec_path,
                     sizeof(spec_path)) != 0 ||
        gs_path_join(directory, "summary.txt", summary_path,
                     sizeof(summary_path)) != 0 ||
        gs_read_pca_thresholds(directory, profile) != 0) {
        return 1;
    }

#define GS_EXPECT_SPEC(k, v) \
    do { if (gs_expect_key_value(spec_path, (k), (v)) != 0) return 1; } while (0)
    GS_EXPECT_SPEC("profile", profile->profile);
    GS_EXPECT_SPEC("source_profile", "mini224_subset1000_pool14_no_pca");
    GS_EXPECT_SPEC("original_feature_count", "1024");
    GS_EXPECT_SPEC("numerical_regime", "rank_deficient");
    GS_EXPECT_SPEC("max_theoretical_rank", "799");
    GS_EXPECT_SPEC("nullspace_basis_scientific_interpretation", "false");
    GS_EXPECT_SPEC("nullspace_validation", "orthogonal_projector");
    GS_EXPECT_SPEC("retained_prefix_validation", "bitwise");
    GS_EXPECT_SPEC("full_model_serialized", "true");
    GS_EXPECT_SPEC("train_rows", "800");
    GS_EXPECT_SPEC("validation_rows", "200");
    GS_EXPECT_SPEC("score_dtype", "float32");
    GS_EXPECT_SPEC("score_order", "row_major");
    GS_EXPECT_SPEC("fit_input", "train_only");
    GS_EXPECT_SPEC("validation_transform_without_refit", "true");
    GS_EXPECT_SPEC("whitening", "false");
    GS_EXPECT_SPEC("post_standardization", "false");
    GS_EXPECT_SPEC("augmentation", "false");
    GS_EXPECT_SPEC("labels_used", "false");
#undef GS_EXPECT_SPEC

    if (gs_read_key_value(
            spec_path, "retained_component_count", value, sizeof(value)
        ) != 0 ||
        gs_parse_size_value(value, &parsed_size) != 0 ||
        parsed_size != profile->components ||
        gs_read_key_value(spec_path, "selected_threshold",
                          value, sizeof(value)) != 0 ||
        gs_parse_double_value(value, &parsed_double) != 0 ||
        parsed_double != profile->threshold ||
        gs_read_key_value(
            spec_path, "cumulative_variance_at_retained_count",
            value, sizeof(value)
        ) != 0 ||
        gs_parse_double_value(value, &parsed_double) != 0 ||
        parsed_double != profile->cumulative) {
        return 1;
    }
    written = snprintf(expected, sizeof(expected), "%zu", profile->d90);
    if (written < 0 || (size_t)written >= sizeof(expected) ||
        gs_expect_key_value(spec_path, "d90", expected) != 0 ||
        gs_expect_key_value(summary_path, "d90", expected) != 0) return 1;
    written = snprintf(expected, sizeof(expected), "%zu", profile->d95);
    if (written < 0 || (size_t)written >= sizeof(expected) ||
        gs_expect_key_value(spec_path, "d95", expected) != 0 ||
        gs_expect_key_value(summary_path, "d95", expected) != 0) return 1;
    written = snprintf(expected, sizeof(expected), "%zu", profile->d99);
    if (written < 0 || (size_t)written >= sizeof(expected) ||
        gs_expect_key_value(spec_path, "d99", expected) != 0 ||
        gs_expect_key_value(summary_path, "d99", expected) != 0) return 1;

#define GS_EXPECT_SUMMARY(k, v) \
    do { if (gs_expect_key_value(summary_path, (k), (v)) != 0) return 1; } while (0)
    GS_EXPECT_SUMMARY("profile", profile->profile);
    GS_EXPECT_SUMMARY("source_profile", "mini224_subset1000_pool14_no_pca");
    GS_EXPECT_SUMMARY("numerical_regime", "rank_deficient");
    GS_EXPECT_SUMMARY("max_theoretical_rank", "799");
    GS_EXPECT_SUMMARY(
        "nullspace_basis_scientific_interpretation", "false"
    );
    GS_EXPECT_SUMMARY("nullspace_validation", "orthogonal_projector");
    GS_EXPECT_SUMMARY("retained_prefix_validation", "bitwise");
    GS_EXPECT_SUMMARY("full_model_serialized", "true");
    GS_EXPECT_SUMMARY("pca_fit", "train_only");
    GS_EXPECT_SUMMARY("validation_transformed_without_refit", "true");
    GS_EXPECT_SUMMARY("whitening", "false");
    GS_EXPECT_SUMMARY("post_standardization", "false");
    GS_EXPECT_SUMMARY("augmentation", "false");
    GS_EXPECT_SUMMARY("labels_used", "false");
#undef GS_EXPECT_SUMMARY

    if (gs_read_key_value(
            summary_path, "code_commit", profile->pca_code_commit,
            sizeof(profile->pca_code_commit)
        ) != 0 ||
        strcmp(profile->pca_code_commit, GS_RUN_PCA_CODE_COMMIT) != 0 ||
        gs_read_key_value(
            spec_path, "numerical_rank", value, sizeof(value)
        ) != 0 ||
        gs_parse_size_value(value, &profile->numerical_rank) != 0 ||
        profile->numerical_rank != 799u ||
        gs_read_key_value(
            spec_path, "nullspace_dimension", value, sizeof(value)
        ) != 0 ||
        gs_parse_size_value(value, &profile->nullspace_dimension) != 0 ||
        profile->nullspace_dimension != 225u ||
        gs_read_key_value(
            spec_path, "rank_tolerance", value, sizeof(value)
        ) != 0 ||
        gs_parse_double_value(value, &profile->rank_tolerance) != 0 ||
        profile->rank_tolerance != 1.2022240890916001e-07 ||
        gs_read_key_value(
            summary_path, "numerical_rank", value, sizeof(value)
        ) != 0 ||
        gs_parse_size_value(value, &parsed_size) != 0 ||
        parsed_size != profile->numerical_rank ||
        gs_read_key_value(
            summary_path, "nullspace_dimension", value, sizeof(value)
        ) != 0 ||
        gs_parse_size_value(value, &parsed_size) != 0 ||
        parsed_size != profile->nullspace_dimension ||
        gs_read_key_value(
            summary_path, "rank_tolerance", value, sizeof(value)
        ) != 0 ||
        gs_parse_double_value(value, &parsed_double) != 0 ||
        parsed_double != profile->rank_tolerance ||
        gs_read_key_value(
            summary_path, "retained_component_count", value, sizeof(value)
        ) != 0 ||
        gs_parse_size_value(value, &parsed_size) != 0 ||
        parsed_size != profile->components ||
        gs_read_key_value(summary_path, "selected_threshold",
                          value, sizeof(value)) != 0 ||
        gs_parse_double_value(value, &parsed_double) != 0 ||
        parsed_double != profile->threshold ||
        gs_read_key_value(
            summary_path, "cumulative_variance_at_retained_count",
            value, sizeof(value)
        ) != 0 ||
        gs_parse_double_value(value, &parsed_double) != 0 ||
        parsed_double != profile->cumulative) {
        return 1;
    }
    return 0;
}


static int gs_build_pca_input_paths(
    const char *pca_directory,
    GSPCAProfile *profile,
    char storage[GS_RUN_CLUSTER_INPUT_COUNT][GS_RUN_PATH_LENGTH],
    const char *paths[GS_RUN_CLUSTER_INPUT_COUNT]
)
{
    size_t index;

    gs_pca_artifact_names[8] = profile->train_filename;
    gs_pca_artifact_names[9] = profile->validation_filename;
    gs_cluster_input_names[2] = profile->train_filename;
    gs_cluster_input_names[3] = profile->validation_filename;
    for (index = 0u; index < GS_RUN_KMEANS_INPUT_COUNT; index++) {
        const char *name =
            index == 4u ? "sha256.txt" : gs_cluster_input_names[index];
        if (gs_path_join(pca_directory, name, storage[index],
                         GS_RUN_PATH_LENGTH) != 0) {
            return 1;
        }
        paths[index] = storage[index];
    }
    return 0;
}


static int gs_verify_pca_sha256(
    const char *pca_directory,
    const char *source_directory,
    char cluster_hashes[GS_RUN_CLUSTER_INPUT_COUNT][
        GS_RUN_SHA256_LENGTH + 1
    ]
)
{
    char manifest_path[GS_RUN_PATH_LENGTH];
    char line[GS_RUN_LINE_LENGTH];
    FILE *file;
    size_t index;
    int failed = 0;

    if (pca_directory == NULL || source_directory == NULL ||
        gs_path_join(pca_directory, "sha256.txt", manifest_path,
                     sizeof(manifest_path)) != 0) {
        return 1;
    }
    memset(cluster_hashes, 0, sizeof(char) * GS_RUN_CLUSTER_INPUT_COUNT *
                                  (GS_RUN_SHA256_LENGTH + 1u));
    file = fopen(manifest_path, "rb");
    if (file == NULL) {
        return 1;
    }
    for (index = 0u; !failed && index < GS_RUN_PCA_INPUT_COUNT; index++) {
        char recorded[GS_RUN_SHA256_LENGTH + 1];
        char actual[GS_RUN_SHA256_LENGTH + 1];
        char path[GS_RUN_PATH_LENGTH];
        if (fgets(line, sizeof(line), file) == NULL ||
            gs_parse_sha_line(
                line, "INPUT", gs_pca_input_names[index], recorded
            ) != 0 ||
            gs_path_join(
                source_directory,
                index == 6u ? "sha256.txt" : gs_pca_input_names[index],
                          path, sizeof(path)) != 0 ||
            gs_sha256_file(path, actual) != 0 ||
            strcmp(recorded, actual) != 0) {
            failed = 1;
        }
    }
    for (index = 0u; !failed && index < GS_RUN_PCA_ARTIFACT_COUNT; index++) {
        char recorded[GS_RUN_SHA256_LENGTH + 1];
        char actual[GS_RUN_SHA256_LENGTH + 1];
        char path[GS_RUN_PATH_LENGTH];
        size_t cluster_index;
        if (fgets(line, sizeof(line), file) == NULL ||
            gs_parse_sha_line(
                line, "ARTIFACT", gs_pca_artifact_names[index], recorded
            ) != 0 ||
            gs_path_join(pca_directory, gs_pca_artifact_names[index],
                         path, sizeof(path)) != 0 ||
            gs_sha256_file(path, actual) != 0 ||
            strcmp(recorded, actual) != 0) {
            failed = 1;
            break;
        }
        if (gs_name_in_list(
                gs_pca_artifact_names[index], gs_cluster_input_names,
                4u, &cluster_index
            )) {
            memcpy(cluster_hashes[cluster_index], actual,
                   GS_RUN_SHA256_LENGTH + 1u);
        }
    }
    if (!failed && fgets(line, sizeof(line), file) != NULL) {
        failed = 1;
    }
    if (ferror(file) || fclose(file) != 0) {
        failed = 1;
    }
    if (!failed &&
        gs_sha256_file(manifest_path, cluster_hashes[4]) != 0) {
        failed = 1;
    }
    for (index = 0u;
         !failed && index < GS_RUN_KMEANS_INPUT_COUNT;
         index++) {
        if (!gs_sha_text_is_valid(cluster_hashes[index])) {
            failed = 1;
        }
    }
    return failed;
}


static int gs_validate_pca_contract(
    const char *pca_directory,
    const char *source_directory,
    GSPCAProfile *profile,
    const char *const cluster_paths[GS_RUN_CLUSTER_INPUT_COUNT],
    char cluster_hashes[GS_RUN_CLUSTER_INPUT_COUNT][
        GS_RUN_SHA256_LENGTH + 1
    ],
    size_t train_bytes,
    size_t validation_bytes
)
{
    size_t train_size;
    size_t validation_size;
    char actual[GS_RUN_SHA256_LENGTH + 1];
    char source_sha_path[GS_RUN_PATH_LENGTH];
    char pca_summary_path[GS_RUN_PATH_LENGTH];
    char recorded_source_sha[GS_RUN_SHA256_LENGTH + 1u];
    size_t index;

    if (source_directory == NULL ||
        gs_validate_pca_metadata(pca_directory, profile) != 0 ||
        gs_pca_inventory_is_valid(pca_directory, profile) != 0 ||
        gs_verify_pca_sha256(
            pca_directory, source_directory, cluster_hashes
        ) != 0 ||
        gs_path_join(
            source_directory, "sha256.txt",
            source_sha_path, sizeof(source_sha_path)
        ) != 0 ||
        gs_path_join(
            pca_directory, "summary.txt",
            pca_summary_path, sizeof(pca_summary_path)
        ) != 0 ||
        gs_sha256_file(source_sha_path, actual) != 0 ||
        gs_read_key_value(
            pca_summary_path,
            "source_sha256_manifest_hash",
            recorded_source_sha,
            sizeof(recorded_source_sha)
        ) != 0 ||
        strcmp(actual, recorded_source_sha) != 0 ||
        gs_file_size(cluster_paths[2], &train_size) != 0 ||
        gs_file_size(cluster_paths[3], &validation_size) != 0 ||
        train_size != train_bytes || validation_size != validation_bytes) {
        return 1;
    }
    for (index = 0u; index < GS_RUN_KMEANS_INPUT_COUNT; index++) {
        if (gs_sha256_file(cluster_paths[index], actual) != 0 ||
            strcmp(actual, cluster_hashes[index]) != 0) {
            return 1;
        }
    }
    return 0;
}


static int gs_read_float_matrix(
    const char *path,
    float *values,
    size_t value_count,
    size_t expected_bytes
)
{
    FILE *file;
    size_t actual_size;
    size_t index;
    int failed = 0;

    if (path == NULL || values == NULL ||
        gs_file_size(path, &actual_size) != 0 ||
        actual_size != expected_bytes) {

        return 1;
    }

    file = fopen(path, "rb");
    if (file == NULL) {
        return 1;
    }
    if (fread(values, sizeof(float), value_count, file) != value_count ||
        fgetc(file) != EOF || ferror(file)) {

        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    if (failed) {
        return 1;
    }

    for (index = 0; index < value_count; index++) {
        if (!isfinite(values[index])) {
            return 1;
        }
    }
    return 0;
}


static int gs_validate_output_paths(
    const char *partial_path,
    const char *pca_directory,
    size_t k,
    char *final_path,
    size_t final_path_size
)
{
    char parent[GS_RUN_PATH_LENGTH];
    char grandparent[GS_RUN_PATH_LENGTH];
    char canonical_parent[GS_RUN_PATH_LENGTH];
    char expected_parent[GS_RUN_PATH_LENGTH];
    char canonical_expected_parent[GS_RUN_PATH_LENGTH];
    char expected_basename[32];
    const char *basename;
    size_t partial_length;
    size_t suffix_length = strlen(GS_RUN_PARTIAL_SUFFIX);
    size_t final_length;
    int partial_state;
    int final_state;

    if (!gs_metadata_value_is_valid(partial_path) ||
        !gs_metadata_value_is_valid(pca_directory) ||
        final_path == NULL || final_path_size == 0u) {
        return 1;
    }
    if (snprintf(
            expected_basename, sizeof(expected_basename),
            "k%zu.partial", k
        ) < 0) {
        return 1;
    }
    basename = gs_path_basename(partial_path);
    if (strcmp(basename, expected_basename) != 0 &&
        !(strcmp(basename, GS_RUN_FAILURE_BASENAME) == 0 &&
          getenv(GS_RUN_FAILURE_ENV) != NULL &&
          strcmp(getenv(GS_RUN_FAILURE_ENV), "1") == 0)) {
        return 1;
    }
    if (gs_parent_directory(partial_path, parent, sizeof(parent)) != 0 ||
        strcmp(gs_path_basename(parent), "kmeans_seed123") != 0 ||
        gs_parent_directory(parent, grandparent, sizeof(grandparent)) != 0 ||
        !gs_directories_are_same(grandparent, pca_directory) ||
        gs_canonicalize_existing_path(
            parent, 1, canonical_parent
        ) != 0 ||
        gs_path_join(
            pca_directory, "kmeans_seed123",
            expected_parent, sizeof(expected_parent)
        ) != 0 ||
        gs_canonicalize_existing_path(
            expected_parent, 1, canonical_expected_parent
        ) != 0 ||
        !gs_paths_equal(
            canonical_parent, canonical_expected_parent
        )) {
        return 1;
    }

    partial_length = strlen(partial_path);
    if (partial_length <= suffix_length ||
        strcmp(
            partial_path + partial_length - suffix_length,
            GS_RUN_PARTIAL_SUFFIX
        ) != 0) {

        return 1;
    }

    final_length = partial_length - suffix_length;
    if (final_length == 0u || final_length >= final_path_size) {
        return 1;
    }
    memcpy(final_path, partial_path, final_length);
    final_path[final_length] = '\0';

    partial_state = gs_path_state(partial_path);
    final_state = gs_path_state(final_path);
    return partial_state == 0 && final_state == 0 ? 0 : 1;
}


static int gs_create_partial_directory(
    const char *partial_path,
    const char *final_path
)
{
    if (gs_path_state(partial_path) != 0 ||
        gs_path_state(final_path) != 0) {
        return 1;
    }

#ifdef _WIN32
    return _mkdir(partial_path) == 0 ? 0 : 1;
#else
    return mkdir(partial_path, 0777) == 0 ? 0 : 1;
#endif
}


static int gs_partial_has_only_expected_names(const char *partial_path)
{
    DIR *stream;
    struct dirent *entry;
    int failed = 0;

    stream = opendir(partial_path);
    if (stream == NULL) {
        return 1;
    }

    while (!failed) {
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
        if (!gs_name_in_list(
                entry->d_name,
                gs_output_all_names,
                GS_RUN_OUTPUT_FILE_COUNT,
                NULL
            )) {
            failed = 1;
        }
    }

    if (closedir(stream) != 0) {
        failed = 1;
    }
    return failed;
}


static int gs_remove_created_partial(const char *partial_path)
{
    size_t index;
    int failed = 0;

    if (partial_path == NULL ||
        gs_partial_has_only_expected_names(partial_path) != 0) {
        return 1;
    }

    for (index = 0; index < GS_RUN_OUTPUT_FILE_COUNT; index++) {
        char path[GS_RUN_PATH_LENGTH];

        if (gs_path_join(
                partial_path,
                gs_output_all_names[index],
                path,
                sizeof(path)
            ) != 0) {
            failed = 1;
            continue;
        }

        errno = 0;
        if (remove(path) != 0 && errno != ENOENT) {
            failed = 1;
        }
    }

    if (failed) {
        return 1;
    }
#ifdef _WIN32
    return _rmdir(partial_path) == 0 ? 0 : 1;
#else
    return rmdir(partial_path) == 0 ? 0 : 1;
#endif
}


static int gs_failure_hook_enabled(const char *partial_path)
{
    const char *value = getenv(GS_RUN_FAILURE_ENV);

    return value != NULL && strcmp(value, "1") == 0 &&
           strcmp(
               gs_path_basename(partial_path),
               GS_RUN_FAILURE_BASENAME
           ) == 0;
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
    const char *directory,
    const char *filename,
    const char *mode,
    FILE **file
)
{
    char path[GS_RUN_PATH_LENGTH];

    if (file == NULL ||
        gs_path_join(directory, filename, path, sizeof(path)) != 0) {
        return 1;
    }
    *file = fopen(path, mode);
    return *file == NULL ? 1 : 0;
}


static int gs_write_binary(
    const char *directory,
    const char *filename,
    const void *values,
    size_t element_size,
    size_t count
)
{
    char path[GS_RUN_PATH_LENGTH];
    FILE *file = NULL;
    size_t expected_bytes;
    size_t actual_bytes;
    int failed = 0;

    if (values == NULL || element_size == 0u ||
        gs_checked_product(element_size, count, &expected_bytes) != 0 ||
        gs_open_output_file(directory, filename, "wb", &file) != 0) {
        return 1;
    }

    if (fwrite(values, element_size, count, file) != count) {
        failed = 1;
    }
    if (gs_finish_write(file) != 0) {
        failed = 1;
    }
    if (gs_path_join(directory, filename, path, sizeof(path)) != 0 ||
        gs_file_size(path, &actual_bytes) != 0 ||
        actual_bytes != expected_bytes) {
        failed = 1;
    }
    return failed;
}


static int gs_validate_fit_result(
    const GSBaselineKMeansResult *result,
    size_t k,
    size_t feature_count
)
{
    size_t row;
    size_t cluster;
    size_t total = 0u;
    size_t centroid_count;

    if (result == NULL ||
        result->rows != GS_RUN_TRAIN_ROWS ||
        result->cols != feature_count ||
        result->k != k ||
        result->seed != GS_RUN_SEED ||
        result->iterations == 0u ||
        result->iterations > GS_RUN_MAX_ITERATIONS ||
        result->converged != 1 ||
        result->centroids == NULL ||
        result->assignments == NULL ||
        result->cluster_sizes == NULL ||
        result->initial_centroid_indices == NULL ||
        !isfinite(result->initial_inertia) ||
        result->initial_inertia < 0.0 ||
        !isfinite(result->inertia) ||
        result->inertia < 0.0 ||
        gs_checked_product(k, feature_count, &centroid_count) != 0) {

        return 1;
    }

    for (row = 0; row < centroid_count; row++) {
        if (!isfinite(result->centroids[row])) {
            return 1;
        }
    }
    for (row = 0; row < GS_RUN_TRAIN_ROWS; row++) {
        if (result->assignments[row] >= k) {
            return 1;
        }
    }
    for (cluster = 0; cluster < k; cluster++) {
        size_t left;

        if (total > SIZE_MAX - result->cluster_sizes[cluster] ||
            result->initial_centroid_indices[cluster] >=
                GS_RUN_TRAIN_ROWS) {
            return 1;
        }
        total += result->cluster_sizes[cluster];

        for (left = 0; left < cluster; left++) {
            if (result->initial_centroid_indices[left] ==
                result->initial_centroid_indices[cluster]) {
                return 1;
            }
        }
    }
    return total == GS_RUN_TRAIN_ROWS ? 0 : 1;
}


static int gs_prepare_serialized_outputs(
    const GSBaselineKMeansResult *result,
    const size_t *recomputed_train_assignments,
    const size_t *validation_assignments,
    const double *train_distances,
    const double *validation_distances,
    uint32_t *train_u32,
    uint32_t *validation_u32,
    uint64_t *cluster_sizes_u64,
    uint64_t *initial_indices_u64,
    GSNumericSummary *summary
)
{
    size_t row;
    size_t cluster;
    size_t total = 0u;

    if (result == NULL || recomputed_train_assignments == NULL ||
        validation_assignments == NULL || train_distances == NULL ||
        validation_distances == NULL || train_u32 == NULL ||
        validation_u32 == NULL || cluster_sizes_u64 == NULL ||
        initial_indices_u64 == NULL || summary == NULL) {
        return 1;
    }

    summary->initial_inertia = result->initial_inertia;
    summary->train_inertia = 0.0;
    summary->validation_inertia = 0.0;

    for (cluster = 0; cluster < result->k; cluster++) {
        cluster_sizes_u64[cluster] = UINT64_C(0);
    }

    for (row = 0; row < GS_RUN_TRAIN_ROWS; row++) {
        size_t assignment = recomputed_train_assignments[row];
        double distance = train_distances[row];

        if (assignment >= result->k ||
            assignment != result->assignments[row] ||
            assignment > (size_t)UINT32_MAX ||
            !isfinite(distance) || distance < 0.0) {
            return 1;
        }
        train_u32[row] = (uint32_t)assignment;
        if (cluster_sizes_u64[assignment] == UINT64_MAX) {
            return 1;
        }
        cluster_sizes_u64[assignment]++;
        summary->train_inertia += distance;
        if (!isfinite(summary->train_inertia)) {
            return 1;
        }
    }

    for (row = 0; row < GS_RUN_VALIDATION_ROWS; row++) {
        size_t assignment = validation_assignments[row];
        double distance = validation_distances[row];

        if (assignment >= result->k ||
            assignment > (size_t)UINT32_MAX ||
            !isfinite(distance) || distance < 0.0) {
            return 1;
        }
        validation_u32[row] = (uint32_t)assignment;
        summary->validation_inertia += distance;
        if (!isfinite(summary->validation_inertia)) {
            return 1;
        }
    }

    summary->empty_clusters = 0u;
    summary->minimum_cluster_size = GS_RUN_TRAIN_ROWS;
    summary->maximum_cluster_size = 0u;

    for (cluster = 0; cluster < result->k; cluster++) {
        size_t cluster_size;

        if (cluster_sizes_u64[cluster] > (uint64_t)SIZE_MAX ||
            (size_t)cluster_sizes_u64[cluster] !=
                result->cluster_sizes[cluster]) {
            return 1;
        }
        cluster_size = (size_t)cluster_sizes_u64[cluster];
        if (total > SIZE_MAX - cluster_size) {
            return 1;
        }
        total += cluster_size;
        initial_indices_u64[cluster] =
            (uint64_t)result->initial_centroid_indices[cluster];

        if (cluster_size == 0u) {
            summary->empty_clusters++;
        }
        if (cluster_size < summary->minimum_cluster_size) {
            summary->minimum_cluster_size = cluster_size;
        }
        if (cluster_size > summary->maximum_cluster_size) {
            summary->maximum_cluster_size = cluster_size;
        }
    }

    if (total != GS_RUN_TRAIN_ROWS) {
        return 1;
    }

    {
        double tolerance =
            1.0e-10 * (1.0 + fabs(result->inertia));
        if (fabs(summary->train_inertia - result->inertia) > tolerance) {
            return 1;
        }
    }
    return 0;
}


static int gs_write_assignment_csv(
    const char *directory,
    const char *filename,
    const uint32_t *assignments,
    const double *distances,
    size_t rows
)
{
    FILE *file = NULL;
    size_t row;
    int failed = 0;

    if (assignments == NULL || distances == NULL ||
        gs_open_output_file(directory, filename, "wb", &file) != 0) {
        return 1;
    }

    if (fputs("row_index,cluster_id,squared_distance\n", file) == EOF) {
        failed = 1;
    }
    for (row = 0; !failed && row < rows; row++) {
        if (fprintf(
                file,
                "%zu,%" PRIu32 ",%.17g\n",
                row,
                assignments[row],
                distances[row]
            ) < 0) {
            failed = 1;
        }
    }
    if (gs_finish_write(file) != 0) {
        failed = 1;
    }
    return failed;
}


static int gs_write_cluster_sizes_csv(
    const char *directory,
    const uint64_t *cluster_sizes,
    size_t k
)
{
    FILE *file = NULL;
    size_t cluster;
    int failed = 0;

    if (cluster_sizes == NULL ||
        gs_open_output_file(
            directory, "cluster_sizes.csv", "wb", &file
        ) != 0) {
        return 1;
    }

    if (fputs("cluster_id,train_count\n", file) == EOF) {
        failed = 1;
    }
    for (cluster = 0; !failed && cluster < k; cluster++) {
        if (fprintf(
                file,
                "%zu,%" PRIu64 "\n",
                cluster,
                cluster_sizes[cluster]
            ) < 0) {
            failed = 1;
        }
    }
    if (gs_finish_write(file) != 0) {
        failed = 1;
    }
    return failed;
}


static int gs_write_initial_centroids_csv(
    const char *directory,
    const uint64_t *indices,
    size_t k
)
{
    FILE *file = NULL;
    size_t order;
    int failed = 0;

    if (indices == NULL ||
        gs_open_output_file(
            directory, "initial_centroids.csv", "wb", &file
        ) != 0) {
        return 1;
    }

    if (fputs("centroid_order,train_row_index\n", file) == EOF) {
        failed = 1;
    }
    for (order = 0; !failed && order < k; order++) {
        if (fprintf(
                file,
                "%zu,%" PRIu64 "\n",
                order,
                indices[order]
            ) < 0) {
            failed = 1;
        }
    }
    if (gs_finish_write(file) != 0) {
        failed = 1;
    }
    return failed;
}


static int gs_write_summary(
    const char *directory,
    const GSPCAProfile *profile,
    size_t k,
    const GSBaselineKMeansResult *result,
    const GSNumericSummary *summary,
    const char *code_commit
)
{
    FILE *file = NULL;
    int failed = 0;

    if (profile == NULL || result == NULL || summary == NULL ||
        gs_open_output_file(directory, "summary.txt", "wb", &file) != 0) {
        return 1;
    }

    if (fprintf(
            file,
            "profile=mini224_subset1000_pool14_%s_kmeans\n"
            "input_profile=mini224_subset1000_pool14_%s\n"
            "pca_profile=%s\n"
            "pca_threshold=%.2f\n"
             "retained_component_count=%zu\n"
             "original_feature_count=1024\n"
             "numerical_rank=%zu\n"
             "nullspace_dimension=%zu\n"
             "rank_tolerance=%.17g\n"
             "policy_c=true\n"
             "pca_code_commit=%s\n"
            "algorithm=kmeans_plus_plus_lloyd\n"
            "input=train_scores_%s_f32\n"
            "validation_policy=assign_to_train_centroids\n"
            "train_rows=800\n"
            "validation_rows=200\n"
            "feature_count=%zu\n"
            "k=%zu\n"
            "seed=123\n"
            "max_iterations=300\n"
            "iterations=%zu\n"
            "converged=1\n"
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
            "input_standardization=source_train_only_before_pca\n"
            "fit_input=train_only\n"
            "validation_assigned_after_fit=true\n"
            "pca=true\n"
            "whitening=false\n"
            "post_pca_standardization=false\n"
            "augmentation=false\n"
            "labels_used=false\n"
            "code_commit=%s\n",
            profile->token,
            profile->token,
            profile->token,
             profile->threshold,
             profile->components,
             profile->numerical_rank,
             profile->nullspace_dimension,
             profile->rank_tolerance,
             profile->pca_code_commit,
            profile->token,
            profile->components,
            k,
            result->iterations,
            result->empty_cluster_reseeds,
            summary->initial_inertia,
            summary->train_inertia,
            summary->validation_inertia,
            summary->empty_clusters,
            summary->minimum_cluster_size,
            summary->maximum_cluster_size,
            code_commit
        ) < 0) {
        failed = 1;
    }

    if (gs_finish_write(file) != 0) {
        failed = 1;
    }
    return failed;
}


static int gs_write_sha256_manifest(
    const char *output_directory,
    const char *const input_paths[GS_RUN_CLUSTER_INPUT_COUNT],
    char input_hashes[GS_RUN_CLUSTER_INPUT_COUNT][
        GS_RUN_SHA256_LENGTH + 1
    ]
)
{
    char artifact_hashes[GS_RUN_CORE_ARTIFACT_COUNT][
        GS_RUN_SHA256_LENGTH + 1
    ];
    FILE *file = NULL;
    size_t index;
    int failed = 0;

    for (index = 0; index < GS_RUN_KMEANS_INPUT_COUNT; index++) {
        char actual[GS_RUN_SHA256_LENGTH + 1];
        if ((index != 4u && strcmp(
                 gs_path_basename(input_paths[index]),
                 gs_cluster_input_names[index]
             ) != 0) ||
            (index == 4u && strcmp(
                 gs_path_basename(input_paths[index]), "sha256.txt"
             ) != 0) ||
            gs_sha256_file(input_paths[index], actual) != 0 ||
            strcmp(actual, input_hashes[index]) != 0) {
            return 1;
        }
    }

    for (index = 0; index < GS_RUN_CORE_ARTIFACT_COUNT; index++) {
        char path[GS_RUN_PATH_LENGTH];
        if (gs_path_join(
                output_directory,
                gs_core_artifact_names[index],
                path,
                sizeof(path)
            ) != 0 ||
            gs_sha256_file(path, artifact_hashes[index]) != 0) {
            return 1;
        }
    }

    if (gs_open_output_file(
            output_directory, "sha256.txt", "wb", &file
        ) != 0) {
        return 1;
    }

    for (index = 0;
         !failed && index < GS_RUN_KMEANS_INPUT_COUNT;
         index++) {
        if (fprintf(
                file,
                "INPUT %s %s\n",
                input_hashes[index],
                gs_cluster_input_names[index]
            ) < 0) {
            failed = 1;
        }
    }
    for (index = 0;
         !failed && index < GS_RUN_CORE_ARTIFACT_COUNT;
         index++) {
        if (fprintf(
                file,
                "ARTIFACT %s %s\n",
                artifact_hashes[index],
                gs_core_artifact_names[index]
            ) < 0) {
            failed = 1;
        }
    }
    if (gs_finish_write(file) != 0) {
        failed = 1;
    }
    return failed;
}


static int gs_verify_output_sha256(
    const char *output_directory,
    const char *const input_paths[GS_RUN_CLUSTER_INPUT_COUNT]
)
{
    char path[GS_RUN_PATH_LENGTH];
    char line[GS_RUN_LINE_LENGTH];
    FILE *file;
    size_t index;
    int failed = 0;

    if (gs_path_join(
            output_directory, "sha256.txt", path, sizeof(path)
        ) != 0) {
        return 1;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        return 1;
    }

    for (index = 0;
         !failed && index < GS_RUN_KMEANS_INPUT_COUNT;
         index++) {
        char recorded[GS_RUN_SHA256_LENGTH + 1];
        char actual[GS_RUN_SHA256_LENGTH + 1];

        if (fgets(line, sizeof(line), file) == NULL ||
            gs_parse_sha_line(
                line, "INPUT", gs_cluster_input_names[index], recorded
            ) != 0 ||
            gs_sha256_file(input_paths[index], actual) != 0 ||
            strcmp(recorded, actual) != 0) {
            failed = 1;
        }
    }
    for (index = 0;
         !failed && index < GS_RUN_CORE_ARTIFACT_COUNT;
         index++) {
        char recorded[GS_RUN_SHA256_LENGTH + 1];
        char actual[GS_RUN_SHA256_LENGTH + 1];
        char artifact_path[GS_RUN_PATH_LENGTH];

        if (fgets(line, sizeof(line), file) == NULL ||
            gs_parse_sha_line(
                line, "ARTIFACT", gs_core_artifact_names[index], recorded
            ) != 0 ||
            gs_path_join(
                output_directory,
                gs_core_artifact_names[index],
                artifact_path,
                sizeof(artifact_path)
            ) != 0 ||
            gs_sha256_file(artifact_path, actual) != 0 ||
            strcmp(recorded, actual) != 0) {
            failed = 1;
        }
    }
    if (!failed && fgets(line, sizeof(line), file) != NULL) {
        failed = 1;
    }
    if (ferror(file)) {
        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    return failed;
}


static int gs_validate_written_artifacts(
    const char *output_directory,
    size_t k,
    size_t feature_count,
    const char *const input_paths[GS_RUN_CLUSTER_INPUT_COUNT]
)
{
    size_t centroid_values;
    size_t centroid_bytes;
    size_t k_bytes;
    size_t index;
    size_t expected_sizes[GS_RUN_OUTPUT_FILE_COUNT] = {
        0u, 3200u, 800u, 6400u, 1600u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u
    };

    if (gs_checked_product(k, feature_count, &centroid_values) != 0 ||
        gs_checked_product(
            centroid_values, sizeof(double), &centroid_bytes
        ) != 0 ||
        gs_checked_product(k, sizeof(uint64_t), &k_bytes) != 0) {
        return 1;
    }
    expected_sizes[0] = centroid_bytes;
    expected_sizes[5] = k_bytes;
    expected_sizes[6] = k_bytes;

    if (gs_exact_inventory_is_valid(
            output_directory,
            gs_output_all_names,
            GS_RUN_OUTPUT_FILE_COUNT
        ) != 0) {
        return 1;
    }

    for (index = 0; index < GS_RUN_OUTPUT_FILE_COUNT; index++) {
        char path[GS_RUN_PATH_LENGTH];
        size_t actual_size;

        if (gs_path_join(
                output_directory,
                gs_output_all_names[index],
                path,
                sizeof(path)
            ) != 0 ||
            gs_file_size(path, &actual_size) != 0 ||
            (expected_sizes[index] == 0u && actual_size == 0u) ||
            (expected_sizes[index] != 0u &&
             actual_size != expected_sizes[index])) {
            return 1;
        }
    }

    return gs_verify_output_sha256(output_directory, input_paths);
}


static void gs_print_usage(const char *program)
{
    printf(
        "Usage: %s <pca_artifact_directory> "
        "<pca90|pca95|pca99> <22|36> 123 300 "
        "<output_partial_directory> <code_commit>\n",
        program
    );
}


int main(int argc, char **argv)
{
    size_t feature_count;
    size_t k;
    size_t max_iterations;
    uint64_t seed;
    size_t train_values;
    size_t validation_values;
    size_t centroid_values;
    size_t train_bytes;
    size_t validation_bytes;
    size_t centroid_bytes;
    size_t train_size_t_bytes;
    size_t validation_size_t_bytes;
    size_t train_u32_bytes;
    size_t validation_u32_bytes;
    size_t train_distance_bytes;
    size_t validation_distance_bytes;
    size_t k_u64_bytes;
    char input_storage[GS_RUN_CLUSTER_INPUT_COUNT][GS_RUN_PATH_LENGTH];
    const char *input_paths[GS_RUN_CLUSTER_INPUT_COUNT];
    char input_hashes[GS_RUN_CLUSTER_INPUT_COUNT][
        GS_RUN_SHA256_LENGTH + 1
    ];
    char final_path[GS_RUN_PATH_LENGTH];
    char canonical_pca[GS_RUN_PATH_LENGTH];
    char repository_root[GS_RUN_PATH_LENGTH];
    char canonical_base[GS_RUN_PATH_LENGTH];

    float *train = NULL;
    float *validation = NULL;
    size_t *recomputed_train_assignments = NULL;
    size_t *validation_assignments = NULL;
    double *train_distances = NULL;
    double *validation_distances = NULL;
    double *centroids_copy = NULL;
    uint32_t *train_u32 = NULL;
    uint32_t *validation_u32 = NULL;
    uint64_t *cluster_sizes_u64 = NULL;
    uint64_t *initial_indices_u64 = NULL;
    GSBaselineKMeansResult result;
    GSNumericSummary summary;
    GSPCAProfile profile;
    int partial_created = 0;
    int rc = 1;

    memset(&result, 0, sizeof(result));
    memset(&summary, 0, sizeof(summary));
    memset(&profile, 0, sizeof(profile));

    if (setlocale(LC_ALL, "C") == NULL) {
        fprintf(stderr, "Errore: locale C non disponibile.\n");
        return 1;
    }

    if (argc == 2 && strcmp(argv[1], "--help") == 0) {
        gs_print_usage(argv[0]);
        return 0;
    }
    if (argc != 8 ||
        !gs_platform_is_valid() ||
        gs_select_pca_profile(argv[2], &profile) != 0 ||
        gs_parse_size_value(argv[3], &k) != 0 ||
        gs_parse_u64_value(argv[4], &seed) != 0 ||
        gs_parse_size_value(argv[5], &max_iterations) != 0 ||
        (k != 22u && k != 36u) ||
        seed != GS_RUN_SEED ||
        max_iterations != GS_RUN_MAX_ITERATIONS ||
        !gs_commit_is_valid(argv[7]) ||
        gs_resolve_repository_layout(
            argv[1], &profile,
            canonical_pca, repository_root, canonical_base
        ) != 0 ||
        gs224_validate_base_contract(
            repository_root, canonical_base
        ) != 0 ||
        gs_validate_pca_metadata(canonical_pca, &profile) != 0 ||
        gs_build_pca_input_paths(
            canonical_pca, &profile,
            input_storage,
            input_paths
        ) != 0 ||
        gs_validate_output_paths(
            argv[6], canonical_pca, k, final_path, sizeof(final_path)
        ) != 0) {

        fprintf(stderr, "Errore: argomenti o piattaforma non validi.\n");
        gs_print_usage(argv[0]);
        return 1;
    }
    feature_count = profile.components;

    if (gs_checked_product(
            GS_RUN_TRAIN_ROWS, feature_count, &train_values
        ) != 0 ||
        gs_checked_product(
            GS_RUN_VALIDATION_ROWS, feature_count, &validation_values
        ) != 0 ||
        gs_checked_product(k, feature_count, &centroid_values) != 0 ||
        gs_checked_product(
            train_values, sizeof(float), &train_bytes
        ) != 0 ||
        gs_checked_product(
            validation_values, sizeof(float), &validation_bytes
        ) != 0 ||
        gs_checked_product(
            centroid_values, sizeof(double), &centroid_bytes
        ) != 0 ||
        gs_checked_product(
            GS_RUN_TRAIN_ROWS, sizeof(size_t), &train_size_t_bytes
        ) != 0 ||
        gs_checked_product(
            GS_RUN_VALIDATION_ROWS, sizeof(size_t), &validation_size_t_bytes
        ) != 0 ||
        gs_checked_product(
            GS_RUN_TRAIN_ROWS, sizeof(uint32_t), &train_u32_bytes
        ) != 0 ||
        gs_checked_product(
            GS_RUN_VALIDATION_ROWS, sizeof(uint32_t), &validation_u32_bytes
        ) != 0 ||
        gs_checked_product(
            GS_RUN_TRAIN_ROWS, sizeof(double), &train_distance_bytes
        ) != 0 ||
        gs_checked_product(
            GS_RUN_VALIDATION_ROWS, sizeof(double), &validation_distance_bytes
        ) != 0 ||
        gs_checked_product(k, sizeof(uint64_t), &k_u64_bytes) != 0 ||
        train_u32_bytes != 3200u ||
        validation_u32_bytes != 800u ||
        train_distance_bytes != 6400u ||
        validation_distance_bytes != 1600u) {

        fprintf(stderr, "Errore: overflow o dimensioni non valide.\n");
        return 1;
    }

    if (gs_validate_pca_contract(
            canonical_pca, canonical_base,
            &profile, input_paths, input_hashes,
            train_bytes, validation_bytes
        ) != 0) {
        fprintf(stderr, "Errore: contratto PCA mini224 non valido.\n");
        return 1;
    }
    printf("[PASS] pca_inventory_metadata_dimensions_and_sha256\n");

    train = (float *)malloc(train_bytes);
    validation = (float *)malloc(validation_bytes);
    recomputed_train_assignments =
        (size_t *)malloc(train_size_t_bytes);
    validation_assignments =
        (size_t *)malloc(validation_size_t_bytes);
    train_distances = (double *)malloc(train_distance_bytes);
    validation_distances =
        (double *)malloc(validation_distance_bytes);
    centroids_copy = (double *)malloc(centroid_bytes);
    train_u32 = (uint32_t *)malloc(train_u32_bytes);
    validation_u32 = (uint32_t *)malloc(validation_u32_bytes);
    cluster_sizes_u64 = (uint64_t *)malloc(k_u64_bytes);
    initial_indices_u64 = (uint64_t *)malloc(k_u64_bytes);

    if (train == NULL || validation == NULL ||
        recomputed_train_assignments == NULL ||
        validation_assignments == NULL ||
        train_distances == NULL || validation_distances == NULL ||
        centroids_copy == NULL || train_u32 == NULL ||
        validation_u32 == NULL || cluster_sizes_u64 == NULL ||
        initial_indices_u64 == NULL) {

        fprintf(stderr, "Errore: memoria insufficiente.\n");
        goto cleanup;
    }

    if (gs_read_float_matrix(
            input_paths[2], train, train_values, train_bytes
        ) != 0 ||
        gs_read_float_matrix(
            input_paths[3],
            validation,
            validation_values,
            validation_bytes
        ) != 0) {

        fprintf(stderr, "Errore: score PCA non validi.\n");
        goto cleanup;
    }
    printf("[PASS] finite_pca_scores\n");

    printf(
        "KMEANS_FIT_STARTED profile=%s k=%zu rows=800 cols=%zu\n",
        profile.token, k, feature_count
    );
    fflush(stdout);
    if (gs_baseline_kmeans_fit(
            train,
            GS_RUN_TRAIN_ROWS,
            feature_count,
            k,
            seed,
            max_iterations,
            &result
        ) != GS_BASELINE_KMEANS_SUCCESS ||
        gs_validate_fit_result(&result, k, feature_count) != 0) {

        fprintf(stderr, "Errore: fit train-only non valido o non convergente.\n");
        goto cleanup;
    }
    printf("KMEANS_FIT_COMPLETED\n");

    if (gs_baseline_kmeans_assign(
            train,
            GS_RUN_TRAIN_ROWS,
            feature_count,
            result.centroids,
            k,
            recomputed_train_assignments,
            train_distances
        ) != GS_BASELINE_KMEANS_SUCCESS) {
        fprintf(stderr, "Errore: ricalcolo assegnazioni train fallito.\n");
        goto cleanup;
    }

    memcpy(centroids_copy, result.centroids, centroid_bytes);
    if (gs_baseline_kmeans_assign(
            validation,
            GS_RUN_VALIDATION_ROWS,
            feature_count,
            result.centroids,
            k,
            validation_assignments,
            validation_distances
        ) != GS_BASELINE_KMEANS_SUCCESS ||
        memcmp(
            centroids_copy, result.centroids, centroid_bytes
        ) != 0) {

        fprintf(stderr, "Errore: assegnazione validation non valida.\n");
        goto cleanup;
    }

    if (gs_prepare_serialized_outputs(
            &result,
            recomputed_train_assignments,
            validation_assignments,
            train_distances,
            validation_distances,
            train_u32,
            validation_u32,
            cluster_sizes_u64,
            initial_indices_u64,
            &summary
        ) != 0) {
        fprintf(stderr, "Errore: contratto numerico non valido.\n");
        goto cleanup;
    }
    printf("[PASS] train_fit_and_frozen_validation_assignment\n");

    if (gs_create_partial_directory(argv[6], final_path) != 0) {
        fprintf(stderr, "Errore: creazione partial fallita.\n");
        goto cleanup;
    }
    partial_created = 1;

    if (gs_failure_hook_enabled(argv[6])) {
        fprintf(stderr, "Errore simulato dopo la creazione della partial.\n");
        goto cleanup;
    }

    if (gs_write_binary(
            argv[6],
            "centroids_f64.bin",
            result.centroids,
            sizeof(double),
            centroid_values
        ) != 0 ||
        gs_write_binary(
            argv[6],
            "train_assignments_u32.bin",
            train_u32,
            sizeof(uint32_t),
            GS_RUN_TRAIN_ROWS
        ) != 0 ||
        gs_write_binary(
            argv[6],
            "val_assignments_u32.bin",
            validation_u32,
            sizeof(uint32_t),
            GS_RUN_VALIDATION_ROWS
        ) != 0 ||
        gs_write_binary(
            argv[6],
            "train_squared_distances_f64.bin",
            train_distances,
            sizeof(double),
            GS_RUN_TRAIN_ROWS
        ) != 0 ||
        gs_write_binary(
            argv[6],
            "val_squared_distances_f64.bin",
            validation_distances,
            sizeof(double),
            GS_RUN_VALIDATION_ROWS
        ) != 0 ||
        gs_write_binary(
            argv[6],
            "cluster_sizes_train_u64.bin",
            cluster_sizes_u64,
            sizeof(uint64_t),
            k
        ) != 0 ||
        gs_write_binary(
            argv[6],
            "initial_centroid_indices_u64.bin",
            initial_indices_u64,
            sizeof(uint64_t),
            k
        ) != 0 ||
        gs_write_assignment_csv(
            argv[6],
            "train_assignments.csv",
            train_u32,
            train_distances,
            GS_RUN_TRAIN_ROWS
        ) != 0 ||
        gs_write_assignment_csv(
            argv[6],
            "val_assignments.csv",
            validation_u32,
            validation_distances,
            GS_RUN_VALIDATION_ROWS
        ) != 0 ||
        gs_write_cluster_sizes_csv(
            argv[6], cluster_sizes_u64, k
        ) != 0 ||
        gs_write_initial_centroids_csv(
            argv[6], initial_indices_u64, k
        ) != 0 ||
        gs_write_summary(
            argv[6], &profile, k, &result, &summary, argv[7]
        ) != 0 ||
        gs_write_sha256_manifest(
            argv[6], input_paths, input_hashes
        ) != 0 ||
        gs_validate_written_artifacts(
            argv[6], k, feature_count, input_paths
        ) != 0) {

        fprintf(stderr, "Errore: artefatti k-means non validi.\n");
        goto cleanup;
    }
    printf("[PASS] exact_inventory_sizes_and_sha256\n");

    if (rename(argv[6], final_path) != 0) {
        fprintf(stderr, "Errore: promozione atomica fallita.\n");
        goto cleanup;
    }
    partial_created = 0;

    printf("K=%zu\n", k);
    printf("PCA_PROFILE=%s\n", profile.token);
    printf("PCA_COMPONENTS=%zu\n", feature_count);
    printf("ITERATIONS=%zu\n", result.iterations);
    printf("INITIAL_INERTIA=%.17g\n", summary.initial_inertia);
    printf("FINAL_TRAIN_INERTIA=%.17g\n", summary.train_inertia);
    printf("VALIDATION_INERTIA=%.17g\n", summary.validation_inertia);
    printf("KMEANS RUN COMPLETED\n");
    rc = 0;

cleanup:
    gs_baseline_kmeans_result_free(&result);
    free(initial_indices_u64);
    free(cluster_sizes_u64);
    free(validation_u32);
    free(train_u32);
    free(centroids_copy);
    free(validation_distances);
    free(train_distances);
    free(validation_assignments);
    free(recomputed_train_assignments);
    free(validation);
    free(train);

    if (partial_created) {
        if (gs_remove_created_partial(argv[6]) != 0) {
            fprintf(stderr, "Errore: cleanup partial confinato fallito.\n");
            rc = 1;
        } else {
            printf("[PASS] current_run_partial_cleanup\n");
        }
    }
    return rc;
}
