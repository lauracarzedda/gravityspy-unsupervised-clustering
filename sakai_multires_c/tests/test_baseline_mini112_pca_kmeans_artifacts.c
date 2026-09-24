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
#endif


enum {
    GS_TEST_PATH_LENGTH = 4096,
    GS_TEST_LINE_LENGTH = 4096,
    GS_TEST_SHA256_LENGTH = 64,
    GS_TEST_TRAIN_ROWS = 6866,
    GS_TEST_VALIDATION_ROWS = 1717,
    GS_TEST_FEATURE_COUNT = 1024,
    GS_TEST_SEED = 123,
    GS_TEST_MAX_ITERATIONS = 300,
    GS_TEST_BASE_INPUT_COUNT = 4,
    GS_TEST_BASE_ARTIFACT_COUNT = 9,
    GS_TEST_BASE_FILE_COUNT = 10,
    GS_TEST_KMEANS_INPUT_COUNT = 5,
    GS_TEST_CLUSTER_INPUT_COUNT = 6,
    GS_TEST_PCA_INPUT_COUNT = 7,
    GS_TEST_PCA_ARTIFACT_COUNT = 11,
    GS_TEST_PCA_FILE_COUNT = 12,
    GS_TEST_CORE_ARTIFACT_COUNT = 12,
    GS_TEST_OUTPUT_FILE_COUNT = 13
};


typedef struct {
    uint32_t state[8];
    uint64_t completed_bits;
    unsigned char block[64];
    size_t block_size;
} GSSha256;


typedef struct {
    size_t k;
    size_t iterations;
    size_t empty_cluster_reseeds;
    double initial_inertia;
    double final_train_inertia;
    double validation_inertia;
    size_t empty_final_clusters;
    size_t minimum_cluster_size;
    size_t maximum_cluster_size;
} GSSummary;

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
    double threshold;
    double cumulative;
} GSPCAProfile;

static size_t gs_runtime_feature_count = 0u;


#if 0
static const char *const gs_base_input_names[GS_TEST_BASE_INPUT_COUNT] = {
    "manifest_events_272.csv",
    "manifest_tensors_112_central.csv",
    "split_train_indices_seed123.csv",
    "split_val_indices_seed123.csv"
};


static const char *const gs_base_artifact_names[
    GS_TEST_BASE_ARTIFACT_COUNT
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


static const char *const gs_base_all_names[GS_TEST_BASE_FILE_COUNT] = {
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


static const char *gs_cluster_input_names[
    GS_TEST_CLUSTER_INPUT_COUNT
] = {
    "pca_spec.txt",
    "thresholds.csv",
    "train_scores_pca90_f32.bin",
    "val_scores_pca90_f32.bin",
    "source_pca_sha256.txt",
    "unused"
};

static const char *const gs_pca_input_names[GS_TEST_PCA_INPUT_COUNT] = {
    "feature_spec.txt",
    "scaler.csv",
    "train_features_std_f32.bin",
    "val_features_std_f32.bin",
    "train_rows.csv",
    "val_rows.csv",
    "sha256.txt"
};

static const char *gs_pca_artifact_names[GS_TEST_PCA_ARTIFACT_COUNT] = {
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


static const char *const gs_core_artifact_names[
    GS_TEST_CORE_ARTIFACT_COUNT
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


static const char *const gs_output_all_names[
    GS_TEST_OUTPUT_FILE_COUNT
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


static int gs_commit_is_valid(const char *commit)
{
    size_t index;

    if (commit == NULL || strchr(commit, '\r') != NULL ||
        strchr(commit, '\n') != NULL) {
        return 0;
    }
    if (strcmp(commit, "UNCOMMITTED_STEP7A_TEST") == 0) {
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
        char path[GS_TEST_PATH_LENGTH];
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


#if 0
static int gs_base_inventory_is_valid(
    const char *directory,
    const char *const *names,
    size_t count
)
{
    unsigned char *seen = NULL;
    DIR *stream = NULL;
    struct dirent *entry;
    size_t index;
    int kmeans_directory_seen = 0;
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
        char path[GS_TEST_PATH_LENGTH];
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

        if (gs_path_join(
                directory, entry->d_name, path, sizeof(path)
            ) != 0 ||
            stat(path, &information) != 0) {
            failed = 1;
            break;
        }

        if (gs_name_in_list(
                entry->d_name, names, count, &matched_index
            )) {
            if (seen[matched_index] != 0u ||
                !S_ISREG(information.st_mode)) {
                failed = 1;
                break;
            }
            seen[matched_index] = 1u;
        } else if (strcmp(entry->d_name, "kmeans_seed123") == 0) {
            if (kmeans_directory_seen ||
                !S_ISDIR(information.st_mode)) {
                failed = 1;
                break;
            }
            kmeans_directory_seen = 1;
        } else {
            failed = 1;
            break;
        }
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
    char output[GS_TEST_SHA256_LENGTH + 1]
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
    output[GS_TEST_SHA256_LENGTH] = '\0';
    return 0;
}


static int gs_sha_text_is_valid(const char *hash)
{
    size_t index;

    if (hash == NULL || strlen(hash) != GS_TEST_SHA256_LENGTH) {
        return 0;
    }
    for (index = 0; index < GS_TEST_SHA256_LENGTH; index++) {
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
    char hash[GS_TEST_SHA256_LENGTH + 1]
)
{
    char *first_space;
    char *second_space;
    char *name;

    if (line == NULL || expected_kind == NULL ||
        expected_name == NULL || hash == NULL) {
        return 1;
    }
    gs_remove_line_ending(line);
    first_space = strchr(line, ' ');
    if (first_space == NULL) {
        return 1;
    }
    *first_space = '\0';
    second_space = strchr(first_space + 1, ' ');
    if (second_space == NULL) {
        return 1;
    }
    *second_space = '\0';
    name = second_space + 1;

    if (strcmp(line, expected_kind) != 0 ||
        strcmp(name, expected_name) != 0 ||
        strchr(name, ' ') != NULL || strchr(name, '/') != NULL ||
        strchr(name, '\\') != NULL ||
        !gs_sha_text_is_valid(first_space + 1)) {
        return 1;
    }
    memcpy(hash, first_space + 1, GS_TEST_SHA256_LENGTH + 1u);
    return 0;
}


#if 0
static int gs_verify_base_sha256(
    const char *base_directory,
    char cluster_hashes[GS_TEST_CLUSTER_INPUT_COUNT][
        GS_TEST_SHA256_LENGTH + 1
    ]
)
{
    char path[GS_TEST_PATH_LENGTH];
    char line[GS_TEST_LINE_LENGTH];
    FILE *file;
    size_t index;
    int failed = 0;

    if (gs_path_join(
            base_directory, "sha256.txt", path, sizeof(path)
        ) != 0) {
        return 1;
    }
    memset(
        cluster_hashes,
        0,
        GS_TEST_CLUSTER_INPUT_COUNT * (GS_TEST_SHA256_LENGTH + 1u)
    );
    file = fopen(path, "rb");
    if (file == NULL) {
        return 1;
    }

    for (index = 0;
         !failed && index < GS_TEST_BASE_INPUT_COUNT;
         index++) {
        char recorded[GS_TEST_SHA256_LENGTH + 1];
        char actual[GS_TEST_SHA256_LENGTH + 1];

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
         !failed && index < GS_TEST_BASE_ARTIFACT_COUNT;
         index++) {
        char recorded[GS_TEST_SHA256_LENGTH + 1];
        char actual[GS_TEST_SHA256_LENGTH + 1];
        char artifact_path[GS_TEST_PATH_LENGTH];
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
                GS_TEST_CLUSTER_INPUT_COUNT,
                &cluster_index
            )) {
            memcpy(
                cluster_hashes[cluster_index],
                actual,
                GS_TEST_SHA256_LENGTH + 1u
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
         !failed && index < GS_TEST_KMEANS_INPUT_COUNT;
         index++) {
        if (!gs_sha_text_is_valid(cluster_hashes[index])) {
            failed = 1;
        }
    }
    return failed;
}
#endif


static int gs_verify_output_sha256(
    const char *base_directory,
    const char *output_directory,
    char cluster_hashes[GS_TEST_CLUSTER_INPUT_COUNT][
        GS_TEST_SHA256_LENGTH + 1
    ]
)
{
    char sha_path[GS_TEST_PATH_LENGTH];
    char line[GS_TEST_LINE_LENGTH];
    FILE *file;
    size_t index;
    int failed = 0;

    if (gs_path_join(
            output_directory, "sha256.txt", sha_path, sizeof(sha_path)
        ) != 0) {
        return 1;
    }
    file = fopen(sha_path, "rb");
    if (file == NULL) {
        return 1;
    }

    for (index = 0;
         !failed && index < GS_TEST_KMEANS_INPUT_COUNT;
         index++) {
        char recorded[GS_TEST_SHA256_LENGTH + 1];
        char actual[GS_TEST_SHA256_LENGTH + 1];
        char input_path[GS_TEST_PATH_LENGTH];

        if (fgets(line, sizeof(line), file) == NULL ||
            gs_parse_sha_line(
                line, "INPUT", gs_cluster_input_names[index], recorded
            ) != 0 ||
            gs_path_join(
                base_directory,
                index == 4u ? "sha256.txt" :
                              gs_cluster_input_names[index],
                input_path,
                sizeof(input_path)
            ) != 0 ||
            gs_sha256_file(input_path, actual) != 0 ||
            strcmp(recorded, actual) != 0 ||
            strcmp(recorded, cluster_hashes[index]) != 0) {
            failed = 1;
        }
    }
    for (index = 0;
         !failed && index < GS_TEST_CORE_ARTIFACT_COUNT;
         index++) {
        char recorded[GS_TEST_SHA256_LENGTH + 1];
        char actual[GS_TEST_SHA256_LENGTH + 1];
        char artifact_path[GS_TEST_PATH_LENGTH];

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
    char line[GS_TEST_LINE_LENGTH];
    FILE *file;
    size_t row = 0u;
    int failed = 0;

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
        "profile", "channels", "image_side", "pool_side",
        "pooled_side", "feature_count", "matrix_dtype",
        "matrix_layout", "pca", "augmentation", "labels_used",
        "scaler_fit"
    };
    static const char *const values[] = {
        "mini112_pool7_no_pca", "4", "112", "7", "16", "1024",
        "float32_ieee754", "row_major", "none", "none", "false",
        "train_only"
    };
    unsigned char seen[sizeof(keys) / sizeof(keys[0])] = {0};
    char line[GS_TEST_LINE_LENGTH];
    FILE *file;
    int failed = 0;

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
#endif


#if 0
static int gs_build_base_path(
    const char *base_directory,
    const char *name,
    char path[GS_TEST_PATH_LENGTH]
)
{
    return gs_path_join(
        base_directory, name, path, GS_TEST_PATH_LENGTH
    );
}
#endif


static const char *gs_pca_path_basename(const char *path)
{
    const char *slash;
    const char *backslash;
    if (path == NULL) {
        return "";
    }
    slash = strrchr(path, '/');
    backslash = strrchr(path, '\\');
    if (slash == NULL) return backslash == NULL ? path : backslash + 1;
    if (backslash == NULL) return slash + 1;
    return slash > backslash ? slash + 1 : backslash + 1;
}


static int gs_pca_parent_directory(
    const char *path,
    char *parent,
    size_t parent_size
)
{
    const char *basename;
    size_t length;
    if (path == NULL || parent == NULL || parent_size == 0u) return 1;
    basename = gs_pca_path_basename(path);
    if (basename == path || basename[0] == '\0') return 1;
    length = (size_t)(basename - path);
    while (length > 1u &&
           (path[length - 1u] == '/' || path[length - 1u] == '\\')) {
        length--;
    }
    if (length == 0u || length >= parent_size) return 1;
    memcpy(parent, path, length);
    parent[length] = '\0';
    return 0;
}


static int gs_pca_directories_same(
    const char *left,
    const char *right
)
{
#ifdef _WIN32
    char left_full[GS_TEST_PATH_LENGTH];
    char right_full[GS_TEST_PATH_LENGTH];
    size_t index;
    if (_fullpath(left_full, left, sizeof(left_full)) == NULL ||
        _fullpath(right_full, right, sizeof(right_full)) == NULL) return 0;
    for (index = 0u; left_full[index] != '\0'; index++) {
        if (left_full[index] == '\\') left_full[index] = '/';
        else if (left_full[index] >= 'A' && left_full[index] <= 'Z')
            left_full[index] = (char)(left_full[index] - 'A' + 'a');
    }
    for (index = 0u; right_full[index] != '\0'; index++) {
        if (right_full[index] == '\\') right_full[index] = '/';
        else if (right_full[index] >= 'A' && right_full[index] <= 'Z')
            right_full[index] = (char)(right_full[index] - 'A' + 'a');
    }
    return strcmp(left_full, right_full) == 0;
#else
    return left != NULL && right != NULL && strcmp(left, right) == 0;
#endif
}


static int gs_validate_final_path(
    const char *pca_directory,
    const char *final_directory,
    size_t k
)
{
    char expected[32];
    char parent[GS_TEST_PATH_LENGTH];
    char grandparent[GS_TEST_PATH_LENGTH];
    int written = snprintf(expected, sizeof(expected), "k%zu", k);
    return written < 0 || (size_t)written >= sizeof(expected) ||
           strcmp(gs_pca_path_basename(final_directory), expected) != 0 ||
           strstr(final_directory, ".partial") != NULL ||
           gs_pca_parent_directory(
               final_directory, parent, sizeof(parent)
           ) != 0 ||
           strcmp(gs_pca_path_basename(parent), "kmeans_seed123") != 0 ||
           gs_pca_parent_directory(parent, grandparent,
                                   sizeof(grandparent)) != 0 ||
           !gs_pca_directories_same(grandparent, pca_directory);
}


static int gs_read_pca_key(
    const char *path,
    const char *key,
    char *value,
    size_t value_size
)
{
    char line[GS_TEST_LINE_LENGTH];
    FILE *file;
    int found = 0;
    int failed = 0;
    if (path == NULL || key == NULL || value == NULL || value_size == 0u) {
        return 1;
    }
    file = fopen(path, "rb");
    if (file == NULL) return 1;
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
                strchr(equals + 1, '\r') != NULL ||
                strchr(equals + 1, '\n') != NULL) {
                failed = 1;
                break;
            }
            memcpy(value, equals + 1, length + 1u);
            found = 1;
        }
    }
    if (ferror(file) || fclose(file) != 0) failed = 1;
    return failed || !found;
}


static int gs_expect_pca_key(
    const char *path,
    const char *key,
    const char *expected
)
{
    char value[GS_TEST_LINE_LENGTH];
    return gs_read_pca_key(path, key, value, sizeof(value)) != 0 ||
           strcmp(value, expected) != 0;
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
         strcmp(token, "pca99") != 0)) return 1;
    memset(profile, 0, sizeof(*profile));
    profile->token = token;
    profile->threshold =
        strcmp(token, "pca90") == 0 ? 0.90 :
        strcmp(token, "pca95") == 0 ? 0.95 : 0.99;
    written = snprintf(
        profile->profile, sizeof(profile->profile),
        "mini112_pool7_%s", token
    );
    if (written < 0 || (size_t)written >= sizeof(profile->profile)) return 1;
    written = snprintf(
        profile->train_filename, sizeof(profile->train_filename),
        "train_scores_%s_f32.bin", token
    );
    if (written < 0 ||
        (size_t)written >= sizeof(profile->train_filename)) return 1;
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
    char path[GS_TEST_PATH_LENGTH];
    char line[GS_TEST_LINE_LENGTH];
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
    if (file == NULL) return 1;
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
            counts[row] > 1024u ||
            cumulative[row] < threshold || cumulative[row] > 1.0 ||
            (row != 0u && (counts[row] < counts[row - 1u] ||
                           cumulative[row] < cumulative[row - 1u]))) {
            failed = 1;
        }
    }
    if (!failed && fgets(line, sizeof(line), file) != NULL) failed = 1;
    if (ferror(file) || fclose(file) != 0) failed = 1;
    if (failed || counts[0] != 20u || counts[1] != 61u ||
        counts[2] != 263u) return 1;
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
    char spec[GS_TEST_PATH_LENGTH];
    char summary[GS_TEST_PATH_LENGTH];
    char value[GS_TEST_LINE_LENGTH];
    char expected[64];
    size_t parsed_size;
    double parsed_double;
    int written;
    if (strcmp(gs_pca_path_basename(directory), profile->profile) != 0 ||
        gs_path_join(directory, "pca_spec.txt", spec, sizeof(spec)) != 0 ||
        gs_path_join(directory, "summary.txt", summary, sizeof(summary)) != 0 ||
        gs_read_pca_thresholds(directory, profile) != 0) return 1;

#define GS_PCA_EXPECT(path, key, expected_value) \
    do { if (gs_expect_pca_key((path), (key), (expected_value)) != 0) return 1; } while (0)
    GS_PCA_EXPECT(spec, "profile", profile->profile);
    GS_PCA_EXPECT(spec, "source_profile", "mini112_pool7_no_pca");
    GS_PCA_EXPECT(spec, "original_feature_count", "1024");
    GS_PCA_EXPECT(spec, "train_rows", "6866");
    GS_PCA_EXPECT(spec, "validation_rows", "1717");
    GS_PCA_EXPECT(spec, "score_dtype", "float32");
    GS_PCA_EXPECT(spec, "score_order", "row_major");
    GS_PCA_EXPECT(spec, "fit_input", "train_only");
    GS_PCA_EXPECT(spec, "validation_transform_without_refit", "true");
    GS_PCA_EXPECT(spec, "whitening", "false");
    GS_PCA_EXPECT(spec, "post_standardization", "false");
    GS_PCA_EXPECT(spec, "augmentation", "false");
    GS_PCA_EXPECT(spec, "labels_used", "false");
    GS_PCA_EXPECT(summary, "profile", profile->profile);
    GS_PCA_EXPECT(summary, "source_profile", "mini112_pool7_no_pca");
    GS_PCA_EXPECT(summary, "pca_fit", "train_only");
    GS_PCA_EXPECT(summary, "validation_transformed_without_refit", "true");
    GS_PCA_EXPECT(summary, "whitening", "false");
    GS_PCA_EXPECT(summary, "post_standardization", "false");
    GS_PCA_EXPECT(summary, "augmentation", "false");
    GS_PCA_EXPECT(summary, "labels_used", "false");
#undef GS_PCA_EXPECT

    if (gs_read_pca_key(
            spec, "retained_component_count", value, sizeof(value)
        ) != 0 ||
        gs_parse_size_value(value, &parsed_size) != 0 ||
        parsed_size != profile->components ||
        gs_read_pca_key(spec, "selected_threshold", value, sizeof(value)) != 0 ||
        gs_parse_double_value(value, &parsed_double) != 0 ||
        parsed_double != profile->threshold ||
        gs_read_pca_key(
            spec, "cumulative_variance_at_retained_count",
            value, sizeof(value)
        ) != 0 ||
        gs_parse_double_value(value, &parsed_double) != 0 ||
        parsed_double != profile->cumulative ||
        gs_read_pca_key(
            summary, "code_commit", profile->pca_code_commit,
            sizeof(profile->pca_code_commit)
        ) != 0 ||
        strlen(profile->pca_code_commit) != 40u ||
        !gs_commit_is_valid(profile->pca_code_commit) ||
        gs_read_pca_key(
            summary, "retained_component_count", value, sizeof(value)
        ) != 0 ||
        gs_parse_size_value(value, &parsed_size) != 0 ||
        parsed_size != profile->components ||
        gs_read_pca_key(
            summary, "selected_threshold", value, sizeof(value)
        ) != 0 ||
        gs_parse_double_value(value, &parsed_double) != 0 ||
        parsed_double != profile->threshold ||
        gs_read_pca_key(
            summary, "cumulative_variance_at_retained_count",
            value, sizeof(value)
        ) != 0 ||
        gs_parse_double_value(value, &parsed_double) != 0 ||
        parsed_double != profile->cumulative) return 1;

    written = snprintf(expected, sizeof(expected), "%zu", profile->d90);
    if (written < 0 || (size_t)written >= sizeof(expected) ||
        gs_expect_pca_key(spec, "d90", expected) != 0 ||
        gs_expect_pca_key(summary, "d90", expected) != 0) return 1;
    written = snprintf(expected, sizeof(expected), "%zu", profile->d95);
    if (written < 0 || (size_t)written >= sizeof(expected) ||
        gs_expect_pca_key(spec, "d95", expected) != 0 ||
        gs_expect_pca_key(summary, "d95", expected) != 0) return 1;
    written = snprintf(expected, sizeof(expected), "%zu", profile->d99);
    if (written < 0 || (size_t)written >= sizeof(expected) ||
        gs_expect_pca_key(spec, "d99", expected) != 0 ||
        gs_expect_pca_key(summary, "d99", expected) != 0) return 1;
    return 0;
}


static int gs_validate_pca_inventory(
    const char *directory,
    const GSPCAProfile *profile
)
{
    const char *names[GS_TEST_PCA_FILE_COUNT];
    unsigned char seen[GS_TEST_PCA_FILE_COUNT] = {0};
    DIR *stream;
    struct dirent *entry;
    size_t index;
    int kmeans_seen = 0;
    int evaluation_seen = 0;
    int failed = 0;
    static const char *const fixed[10] = {
        "pca_spec.txt", "mean_f64.bin", "components_f64.bin",
        "eigenvalues_f64.bin", "explained_variance_ratio_f64.bin",
        "cumulative_explained_variance_ratio_f64.bin", "thresholds.csv",
        "spectrum.csv", "summary.txt", "sha256.txt"
    };
    for (index = 0u; index < 8u; index++) names[index] = fixed[index];
    names[8] = profile->train_filename;
    names[9] = profile->validation_filename;
    names[10] = fixed[8];
    names[11] = fixed[9];
    stream = opendir(directory);
    if (stream == NULL) return 1;
    while (!failed) {
        size_t matched;
        char path[GS_TEST_PATH_LENGTH];
        struct stat information;
        errno = 0;
        entry = readdir(stream);
        if (entry == NULL) {
            failed = errno != 0;
            break;
        }
        if (strcmp(entry->d_name, ".") == 0 ||
            strcmp(entry->d_name, "..") == 0) continue;
        if (gs_path_join(directory, entry->d_name, path, sizeof(path)) != 0 ||
            stat(path, &information) != 0) {
            failed = 1;
            break;
        }
        if (gs_name_in_list(
                entry->d_name, names, GS_TEST_PCA_FILE_COUNT, &matched
            )) {
            if (seen[matched] || !S_ISREG(information.st_mode)) failed = 1;
            seen[matched] = 1u;
        } else if (strcmp(entry->d_name, "kmeans_seed123") == 0) {
            if (kmeans_seen || !S_ISDIR(information.st_mode)) failed = 1;
            kmeans_seen = 1;
        } else if (strcmp(entry->d_name, "label_evaluation_seed123") == 0) {
            if (evaluation_seen || !S_ISDIR(information.st_mode)) failed = 1;
            evaluation_seen = 1;
        } else {
            failed = 1;
        }
    }
    if (closedir(stream) != 0) failed = 1;
    for (index = 0u; !failed && index < GS_TEST_PCA_FILE_COUNT; index++) {
        if (!seen[index]) failed = 1;
    }
    return failed;
}


static int gs_validate_pca_sha(
    const char *directory,
    GSPCAProfile *profile,
    char hashes[GS_TEST_CLUSTER_INPUT_COUNT][GS_TEST_SHA256_LENGTH + 1]
)
{
    char parent[GS_TEST_PATH_LENGTH];
    char source[GS_TEST_PATH_LENGTH];
    char manifest[GS_TEST_PATH_LENGTH];
    char line[GS_TEST_LINE_LENGTH];
    FILE *file;
    size_t index;
    int failed = 0;
    gs_pca_artifact_names[8] = profile->train_filename;
    gs_pca_artifact_names[9] = profile->validation_filename;
    gs_cluster_input_names[2] = profile->train_filename;
    gs_cluster_input_names[3] = profile->validation_filename;
    if (gs_pca_parent_directory(directory, parent, sizeof(parent)) != 0 ||
        gs_path_join(parent, "mini112_pool7_no_pca", source,
                     sizeof(source)) != 0 ||
        gs_path_join(directory, "sha256.txt", manifest,
                     sizeof(manifest)) != 0) return 1;
    memset(hashes, 0, sizeof(char) * GS_TEST_CLUSTER_INPUT_COUNT *
                          (GS_TEST_SHA256_LENGTH + 1u));
    file = fopen(manifest, "rb");
    if (file == NULL) return 1;
    for (index = 0u; !failed && index < GS_TEST_PCA_INPUT_COUNT; index++) {
        char recorded[65];
        char actual[65];
        char path[GS_TEST_PATH_LENGTH];
        if (fgets(line, sizeof(line), file) == NULL ||
            gs_parse_sha_line(
                line, "INPUT", gs_pca_input_names[index], recorded
            ) != 0 ||
            gs_path_join(source, gs_pca_input_names[index], path,
                         sizeof(path)) != 0 ||
            gs_sha256_file(path, actual) != 0 ||
            strcmp(recorded, actual) != 0) failed = 1;
    }
    for (index = 0u; !failed && index < GS_TEST_PCA_ARTIFACT_COUNT; index++) {
        char recorded[65];
        char actual[65];
        char path[GS_TEST_PATH_LENGTH];
        size_t cluster_index;
        if (fgets(line, sizeof(line), file) == NULL ||
            gs_parse_sha_line(
                line, "ARTIFACT", gs_pca_artifact_names[index], recorded
            ) != 0 ||
            gs_path_join(directory, gs_pca_artifact_names[index], path,
                         sizeof(path)) != 0 ||
            gs_sha256_file(path, actual) != 0 ||
            strcmp(recorded, actual) != 0) {
            failed = 1;
            break;
        }
        if (gs_name_in_list(
                gs_pca_artifact_names[index],
                gs_cluster_input_names, 4u, &cluster_index
            )) {
            memcpy(hashes[cluster_index], actual, 65u);
        }
    }
    if (!failed && fgets(line, sizeof(line), file) != NULL) failed = 1;
    if (ferror(file) || fclose(file) != 0) failed = 1;
    if (!failed && gs_sha256_file(manifest, hashes[4]) != 0) failed = 1;
    for (index = 0u; !failed && index < GS_TEST_KMEANS_INPUT_COUNT; index++) {
        if (!gs_sha_text_is_valid(hashes[index])) failed = 1;
    }
    return failed;
}


static int gs_validate_pca_contract(
    const char *directory,
    GSPCAProfile *profile,
    char hashes[GS_TEST_CLUSTER_INPUT_COUNT][GS_TEST_SHA256_LENGTH + 1],
    char train_path[GS_TEST_PATH_LENGTH],
    char validation_path[GS_TEST_PATH_LENGTH],
    size_t train_bytes,
    size_t validation_bytes
)
{
    size_t actual_size;
    char actual_hash[65];
    size_t index;
    char paths[GS_TEST_KMEANS_INPUT_COUNT][GS_TEST_PATH_LENGTH];
    if (gs_validate_pca_metadata(directory, profile) != 0 ||
        gs_validate_pca_inventory(directory, profile) != 0 ||
        gs_validate_pca_sha(directory, profile, hashes) != 0 ||
        gs_path_join(directory, "pca_spec.txt", paths[0],
                     sizeof(paths[0])) != 0 ||
        gs_path_join(directory, "thresholds.csv", paths[1],
                     sizeof(paths[1])) != 0 ||
        gs_path_join(directory, profile->train_filename, paths[2],
                     sizeof(paths[2])) != 0 ||
        gs_path_join(directory, profile->validation_filename, paths[3],
                     sizeof(paths[3])) != 0 ||
        gs_path_join(directory, "sha256.txt", paths[4],
                     sizeof(paths[4])) != 0) return 1;
    for (index = 0u; index < GS_TEST_KMEANS_INPUT_COUNT; index++) {
        if (gs_sha256_file(paths[index], actual_hash) != 0 ||
            strcmp(actual_hash, hashes[index]) != 0) return 1;
    }
    if (gs_file_size(paths[2], &actual_size) != 0 ||
        actual_size != train_bytes ||
        gs_file_size(paths[3], &actual_size) != 0 ||
        actual_size != validation_bytes) return 1;
    memcpy(train_path, paths[2], strlen(paths[2]) + 1u);
    memcpy(validation_path, paths[3], strlen(paths[3]) + 1u);
    return 0;
}


#if 0
static int gs_validate_base_contract(
    const char *base_directory,
    char cluster_hashes[GS_TEST_CLUSTER_INPUT_COUNT][
        GS_TEST_SHA256_LENGTH + 1
    ]
)
{
    static const size_t expected_sizes[GS_TEST_CLUSTER_INPUT_COUNT] = {
        958u, 54237u, 28123136u, 7032832u, 270713u, 74659u
    };
    char paths[GS_TEST_CLUSTER_INPUT_COUNT][GS_TEST_PATH_LENGTH];
    size_t index;

    if (gs_base_inventory_is_valid(
            base_directory,
            gs_base_all_names,
            GS_TEST_BASE_FILE_COUNT
        ) != 0 ||
        gs_verify_base_sha256(base_directory, cluster_hashes) != 0) {
        return 1;
    }
    for (index = 0; index < GS_TEST_CLUSTER_INPUT_COUNT; index++) {
        size_t actual_size;
        char actual_hash[GS_TEST_SHA256_LENGTH + 1];

        if (gs_build_base_path(
                base_directory,
                gs_cluster_input_names[index],
                paths[index]
            ) != 0 ||
            gs_file_size(paths[index], &actual_size) != 0 ||
            actual_size != expected_sizes[index] ||
            gs_sha256_file(paths[index], actual_hash) != 0 ||
            strcmp(actual_hash, cluster_hashes[index]) != 0) {
            return 1;
        }
    }
    return gs_validate_feature_spec(paths[0]) ||
           gs_validate_row_mapping(
               paths[4], GS_TEST_TRAIN_ROWS, "train"
           ) ||
           gs_validate_row_mapping(
               paths[5], GS_TEST_VALIDATION_ROWS, "validation"
           );
}
#endif


static int gs_validate_output_inventory_and_sizes(
    const char *output_directory,
    size_t k
)
{
    size_t centroid_values;
    size_t centroid_bytes;
    size_t k_bytes;
    size_t index;
    size_t expected_sizes[GS_TEST_OUTPUT_FILE_COUNT] = {
        0u, 27464u, 6868u, 54928u, 13736u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u
    };
    char partial_path[GS_TEST_PATH_LENGTH];
    int written;

    if (gs_checked_product(
            k, gs_runtime_feature_count, &centroid_values
        ) != 0 ||
        gs_checked_product(
            centroid_values, sizeof(double), &centroid_bytes
        ) != 0 ||
        gs_checked_product(k, sizeof(uint64_t), &k_bytes) != 0 ||
        (written = snprintf(
            partial_path,
            sizeof(partial_path),
            "%s.partial",
            output_directory
        )) < 0 ||
        (size_t)written >= sizeof(partial_path) ||
        gs_path_state(partial_path) != 0) {
        return 1;
    }
    expected_sizes[0] = centroid_bytes;
    expected_sizes[5] = k_bytes;
    expected_sizes[6] = k_bytes;

    if (gs_exact_inventory_is_valid(
            output_directory,
            gs_output_all_names,
            GS_TEST_OUTPUT_FILE_COUNT
        ) != 0) {
        return 1;
    }
    for (index = 0; index < GS_TEST_OUTPUT_FILE_COUNT; index++) {
        char path[GS_TEST_PATH_LENGTH];
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
    char path[GS_TEST_PATH_LENGTH];
    FILE *file;
    size_t expected_bytes;
    size_t actual_bytes;
    int failed = 0;

    if (values == NULL || element_size == 0u ||
        gs_checked_product(element_size, count, &expected_bytes) != 0 ||
        gs_path_join(directory, filename, path, sizeof(path)) != 0 ||
        gs_file_size(path, &actual_bytes) != 0 ||
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


static int gs_read_float_matrix(
    const char *path,
    float *values,
    size_t value_count,
    size_t expected_bytes
)
{
    FILE *file;
    size_t actual_bytes;
    size_t index;
    int failed = 0;

    if (path == NULL || values == NULL ||
        gs_file_size(path, &actual_bytes) != 0 ||
        actual_bytes != expected_bytes) {
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


static int gs_read_summary_pair(
    FILE *file,
    const char *expected_key,
    char value[GS_TEST_LINE_LENGTH]
)
{
    char line[GS_TEST_LINE_LENGTH];
    char *equals;

    if (file == NULL || expected_key == NULL || value == NULL ||
        fgets(line, sizeof(line), file) == NULL ||
        (strchr(line, '\n') == NULL && !feof(file))) {
        return 1;
    }
    gs_remove_line_ending(line);
    equals = strchr(line, '=');
    if (equals == NULL || equals == line ||
        strchr(equals + 1, '=') != NULL) {
        return 1;
    }
    *equals = '\0';
    if (strcmp(line, expected_key) != 0 ||
        equals[1] == '\0') {
        return 1;
    }
    if (strlen(equals + 1) >= GS_TEST_LINE_LENGTH) {
        return 1;
    }
    strcpy(value, equals + 1);
    return 0;
}


static int gs_expect_summary_text(
    FILE *file,
    const char *key,
    const char *expected
)
{
    char value[GS_TEST_LINE_LENGTH];
    return gs_read_summary_pair(file, key, value) ||
           strcmp(value, expected) != 0;
}


static int gs_expect_summary_size(
    FILE *file,
    const char *key,
    size_t *value
)
{
    char text[GS_TEST_LINE_LENGTH];
    return gs_read_summary_pair(file, key, text) ||
           gs_parse_size_value(text, value);
}


static int gs_expect_summary_double(
    FILE *file,
    const char *key,
    double *value
)
{
    char text[GS_TEST_LINE_LENGTH];
    return gs_read_summary_pair(file, key, text) ||
           gs_parse_double_value(text, value);
}


static int gs_parse_summary(
    const char *directory,
    const GSPCAProfile *profile,
    size_t expected_k,
    const char *expected_commit,
    GSSummary *summary
)
{
    char path[GS_TEST_PATH_LENGTH];
    char code_commit[GS_TEST_LINE_LENGTH];
    char extra[GS_TEST_LINE_LENGTH];
    FILE *file;
    size_t train_rows;
    size_t validation_rows;
    size_t feature_count;
    size_t seed;
    size_t max_iterations;
    size_t converged;
    size_t retained_components;
    size_t original_feature_count;
    double pca_threshold;
    int failed = 0;

    if (profile == NULL || summary == NULL ||
        gs_path_join(directory, "summary.txt", path, sizeof(path)) != 0) {
        return 1;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        return 1;
    }

    if (gs_expect_summary_text(
            file, "profile",
            strcmp(profile->token, "pca90") == 0 ?
                "mini112_pool7_pca90_kmeans" :
            strcmp(profile->token, "pca95") == 0 ?
                "mini112_pool7_pca95_kmeans" :
                "mini112_pool7_pca99_kmeans"
        ) ||
        gs_expect_summary_text(
            file, "input_profile", profile->profile
        ) ||
        gs_expect_summary_text(file, "pca_profile", profile->token) ||
        gs_expect_summary_double(file, "pca_threshold", &pca_threshold) ||
        gs_expect_summary_size(
            file, "retained_component_count", &retained_components
        ) ||
        gs_expect_summary_size(
            file, "original_feature_count", &original_feature_count
        ) ||
        gs_expect_summary_text(
            file, "pca_code_commit", profile->pca_code_commit
        ) ||
        gs_expect_summary_text(
            file, "algorithm", "kmeans_plus_plus_lloyd"
        ) ||
        gs_expect_summary_text(
            file, "input",
            strcmp(profile->token, "pca90") == 0 ?
                "train_scores_pca90_f32" :
            strcmp(profile->token, "pca95") == 0 ?
                "train_scores_pca95_f32" :
                "train_scores_pca99_f32"
        ) ||
        gs_expect_summary_text(
            file,
            "validation_policy",
            "assign_to_train_centroids"
        ) ||
        gs_expect_summary_size(file, "train_rows", &train_rows) ||
        gs_expect_summary_size(
            file, "validation_rows", &validation_rows
        ) ||
        gs_expect_summary_size(
            file, "feature_count", &feature_count
        ) ||
        gs_expect_summary_size(file, "k", &summary->k) ||
        gs_expect_summary_size(file, "seed", &seed) ||
        gs_expect_summary_size(
            file, "max_iterations", &max_iterations
        ) ||
        gs_expect_summary_size(
            file, "iterations", &summary->iterations
        ) ||
        gs_expect_summary_size(file, "converged", &converged) ||
        gs_expect_summary_size(
            file,
            "empty_cluster_reseeds",
            &summary->empty_cluster_reseeds
        ) ||
        gs_expect_summary_double(
            file, "initial_inertia", &summary->initial_inertia
        ) ||
        gs_expect_summary_double(
            file,
            "final_train_inertia",
            &summary->final_train_inertia
        ) ||
        gs_expect_summary_double(
            file,
            "validation_inertia",
            &summary->validation_inertia
        ) ||
        gs_expect_summary_size(
            file,
            "empty_final_clusters",
            &summary->empty_final_clusters
        ) ||
        gs_expect_summary_size(
            file,
            "minimum_cluster_size",
            &summary->minimum_cluster_size
        ) ||
        gs_expect_summary_size(
            file,
            "maximum_cluster_size",
            &summary->maximum_cluster_size
        ) ||
        gs_expect_summary_text(file, "centroid_dtype", "float64") ||
        gs_expect_summary_text(file, "assignment_dtype", "uint32") ||
        gs_expect_summary_text(file, "distance_dtype", "float64") ||
        gs_expect_summary_text(file, "cluster_size_dtype", "uint64") ||
        gs_expect_summary_text(
            file, "input_standardization",
            "source_train_only_before_pca"
        ) ||
        gs_expect_summary_text(file, "fit_input", "train_only") ||
        gs_expect_summary_text(
            file, "validation_assigned_after_fit", "true"
        ) ||
        gs_expect_summary_text(file, "pca", "true") ||
        gs_expect_summary_text(file, "whitening", "false") ||
        gs_expect_summary_text(
            file, "post_pca_standardization", "false"
        ) ||
        gs_expect_summary_text(file, "augmentation", "false") ||
        gs_expect_summary_text(file, "labels_used", "false") ||
        gs_read_summary_pair(file, "code_commit", code_commit) ||
        strcmp(code_commit, expected_commit) != 0) {
        failed = 1;
    }

    if (!failed && fgets(extra, sizeof(extra), file) != NULL) {
        failed = 1;
    }
    if (ferror(file)) {
        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }

    if (!failed &&
        (train_rows != GS_TEST_TRAIN_ROWS ||
         validation_rows != GS_TEST_VALIDATION_ROWS ||
         retained_components != profile->components ||
         original_feature_count != 1024u ||
         pca_threshold != profile->threshold ||
         feature_count != gs_runtime_feature_count ||
         summary->k != expected_k ||
         seed != GS_TEST_SEED ||
         max_iterations != GS_TEST_MAX_ITERATIONS ||
         summary->iterations == 0u ||
         summary->iterations > GS_TEST_MAX_ITERATIONS ||
         converged != 1u ||
         summary->initial_inertia < 0.0 ||
         summary->final_train_inertia < 0.0 ||
         summary->validation_inertia < 0.0 ||
         summary->empty_final_clusters > expected_k ||
         summary->minimum_cluster_size > GS_TEST_TRAIN_ROWS ||
         summary->maximum_cluster_size > GS_TEST_TRAIN_ROWS)) {
        failed = 1;
    }
    return failed;
}


static int gs_validate_assignment_csv(
    const char *directory,
    const char *filename,
    const uint32_t *assignments,
    const double *distances,
    size_t rows,
    size_t k
)
{
    static const char header[] =
        "row_index,cluster_id,squared_distance";
    char path[GS_TEST_PATH_LENGTH];
    char line[GS_TEST_LINE_LENGTH];
    FILE *file;
    size_t row = 0u;
    int failed = 0;

    if (gs_path_join(directory, filename, path, sizeof(path)) != 0) {
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
        char *fields[3];
        size_t row_index;
        size_t cluster;
        double distance;

        if (strchr(line, '\n') == NULL && !feof(file)) {
            failed = 1;
            break;
        }
        gs_remove_line_ending(line);
        if (row >= rows ||
            gs_split_exact(line, fields, 3u) != 0 ||
            gs_parse_size_value(fields[0], &row_index) != 0 ||
            gs_parse_size_value(fields[1], &cluster) != 0 ||
            gs_parse_double_value(fields[2], &distance) != 0 ||
            row_index != row || cluster >= k ||
            cluster != (size_t)assignments[row] ||
            memcmp(&distance, &distances[row], sizeof(double)) != 0) {
            failed = 1;
        }
        row++;
    }
    if (ferror(file) || row != rows) {
        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    return failed;
}


static int gs_validate_cluster_sizes_csv(
    const char *directory,
    const uint64_t *cluster_sizes,
    size_t k
)
{
    char path[GS_TEST_PATH_LENGTH];
    char line[GS_TEST_LINE_LENGTH];
    FILE *file;
    size_t cluster = 0u;
    int failed = 0;

    if (gs_path_join(
            directory, "cluster_sizes.csv", path, sizeof(path)
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
        gs_remove_line_ending(line);
        if (strcmp(line, "cluster_id,train_count") != 0) {
            failed = 1;
        }
    }
    while (!failed && fgets(line, sizeof(line), file) != NULL) {
        char *fields[2];
        size_t cluster_id;
        size_t count;

        gs_remove_line_ending(line);
        if (cluster >= k ||
            gs_split_exact(line, fields, 2u) != 0 ||
            gs_parse_size_value(fields[0], &cluster_id) != 0 ||
            gs_parse_size_value(fields[1], &count) != 0 ||
            cluster_id != cluster ||
            (uint64_t)count != cluster_sizes[cluster]) {
            failed = 1;
        }
        cluster++;
    }
    if (ferror(file) || cluster != k) {
        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    return failed;
}


static int gs_validate_initial_centroids_csv(
    const char *directory,
    const uint64_t *initial_indices,
    size_t k
)
{
    char path[GS_TEST_PATH_LENGTH];
    char line[GS_TEST_LINE_LENGTH];
    FILE *file;
    size_t order = 0u;
    int failed = 0;

    if (gs_path_join(
            directory, "initial_centroids.csv", path, sizeof(path)
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
        gs_remove_line_ending(line);
        if (strcmp(line, "centroid_order,train_row_index") != 0) {
            failed = 1;
        }
    }
    while (!failed && fgets(line, sizeof(line), file) != NULL) {
        char *fields[2];
        size_t parsed_order;
        size_t train_row_index;

        gs_remove_line_ending(line);
        if (order >= k ||
            gs_split_exact(line, fields, 2u) != 0 ||
            gs_parse_size_value(fields[0], &parsed_order) != 0 ||
            gs_parse_size_value(fields[1], &train_row_index) != 0 ||
            parsed_order != order ||
            (uint64_t)train_row_index != initial_indices[order]) {
            failed = 1;
        }
        order++;
    }
    if (ferror(file) || order != k) {
        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    return failed;
}


static int gs_distance_equal(double expected, double actual)
{
    double tolerance = 1.0e-10 * (1.0 + fabs(expected));
    return isfinite(expected) && isfinite(actual) &&
           expected >= 0.0 && actual >= 0.0 &&
           fabs(expected - actual) <= tolerance;
}


static int gs_inertia_equal(double expected, double actual)
{
    double tolerance = 1.0e-10 * (1.0 + fabs(expected));
    return isfinite(expected) && isfinite(actual) &&
           expected >= 0.0 && actual >= 0.0 &&
           fabs(expected - actual) <= tolerance;
}


static int gs_recompute_centroids_from_train(
    const float *train,
    const uint32_t *assignments,
    const uint64_t *cluster_sizes,
    const double *centroids,
    size_t k
)
{
    size_t centroid_values;
    size_t centroid_bytes;
    double *means = NULL;
    size_t row;
    size_t cluster;
    size_t feature;
    int failed = 0;

    if (gs_checked_product(
            k, gs_runtime_feature_count, &centroid_values
        ) != 0 ||
        gs_checked_product(
            centroid_values, sizeof(double), &centroid_bytes
        ) != 0) {
        return 1;
    }
    means = (double *)calloc(centroid_values, sizeof(double));
    if (means == NULL) {
        return 1;
    }

    for (row = 0; row < GS_TEST_TRAIN_ROWS; row++) {
        size_t assignment = assignments[row];
        size_t row_offset;
        size_t centroid_offset;

        if (assignment >= k ||
            gs_checked_product(
                row, gs_runtime_feature_count, &row_offset
            ) != 0 ||
            gs_checked_product(
                assignment,
                gs_runtime_feature_count,
                &centroid_offset
            ) != 0) {
            failed = 1;
            break;
        }
        for (feature = 0; feature < gs_runtime_feature_count; feature++) {
            means[centroid_offset + feature] +=
                (double)train[row_offset + feature];
        }
    }

    for (cluster = 0; !failed && cluster < k; cluster++) {
        size_t centroid_offset;

        if (cluster_sizes[cluster] == 0u ||
            gs_checked_product(
                cluster,
                gs_runtime_feature_count,
                &centroid_offset
            ) != 0) {
            failed = 1;
            break;
        }
        for (feature = 0; feature < gs_runtime_feature_count; feature++) {
            double expected;
            double actual;
            double tolerance;

            means[centroid_offset + feature] /=
                (double)cluster_sizes[cluster];
            expected = means[centroid_offset + feature];
            actual = centroids[centroid_offset + feature];
            tolerance = 1.0e-12 * (1.0 + fabs(expected));
            if (!isfinite(actual) ||
                fabs(expected - actual) > tolerance) {
                failed = 1;
                break;
            }
        }
    }

    free(means);
    (void)centroid_bytes;
    return failed;
}


static int gs_refit_matches_artifacts(
    const float *train,
    const float *validation,
    size_t k,
    const GSSummary *summary,
    const double *centroids,
    const uint32_t *train_assignments,
    const uint32_t *validation_assignments,
    const double *validation_distances,
    const uint64_t *cluster_sizes,
    const uint64_t *initial_indices
)
{
    GSBaselineKMeansResult refit;
    size_t centroid_values;
    size_t centroid_bytes;
    size_t validation_assignment_bytes;
    size_t validation_distance_bytes;
    size_t *refit_validation_assignments = NULL;
    double *refit_validation_distances = NULL;
    double *centroid_copy = NULL;
    size_t row;
    size_t cluster;
    int failed = 0;

    memset(&refit, 0, sizeof(refit));
    if (gs_checked_product(
            k, gs_runtime_feature_count, &centroid_values
        ) != 0 ||
        gs_checked_product(
            centroid_values, sizeof(double), &centroid_bytes
        ) != 0 ||
        gs_checked_product(
            GS_TEST_VALIDATION_ROWS,
            sizeof(size_t),
            &validation_assignment_bytes
        ) != 0 ||
        gs_checked_product(
            GS_TEST_VALIDATION_ROWS,
            sizeof(double),
            &validation_distance_bytes
        ) != 0) {
        return 1;
    }

    refit_validation_assignments =
        (size_t *)malloc(validation_assignment_bytes);
    refit_validation_distances =
        (double *)malloc(validation_distance_bytes);
    centroid_copy = (double *)malloc(centroid_bytes);
    if (refit_validation_assignments == NULL ||
        refit_validation_distances == NULL ||
        centroid_copy == NULL ||
        gs_baseline_kmeans_fit(
            train,
            GS_TEST_TRAIN_ROWS,
            gs_runtime_feature_count,
            k,
            GS_TEST_SEED,
            GS_TEST_MAX_ITERATIONS,
            &refit
        ) != GS_BASELINE_KMEANS_SUCCESS) {
        failed = 1;
        goto cleanup;
    }

    if (refit.rows != GS_TEST_TRAIN_ROWS ||
        refit.cols != gs_runtime_feature_count ||
        refit.k != k || refit.seed != GS_TEST_SEED ||
        refit.iterations != summary->iterations ||
        refit.converged != 1 ||
        refit.empty_cluster_reseeds !=
            summary->empty_cluster_reseeds ||
        memcmp(refit.centroids, centroids, centroid_bytes) != 0 ||
        memcmp(
            &refit.initial_inertia,
            &summary->initial_inertia,
            sizeof(double)
        ) != 0 ||
        memcmp(
            &refit.inertia,
            &summary->final_train_inertia,
            sizeof(double)
        ) != 0) {
        failed = 1;
        goto cleanup;
    }

    for (row = 0; row < GS_TEST_TRAIN_ROWS; row++) {
        if (refit.assignments[row] !=
            (size_t)train_assignments[row]) {
            failed = 1;
            goto cleanup;
        }
    }
    for (cluster = 0; cluster < k; cluster++) {
        if ((uint64_t)refit.cluster_sizes[cluster] !=
                cluster_sizes[cluster] ||
            (uint64_t)refit.initial_centroid_indices[cluster] !=
                initial_indices[cluster]) {
            failed = 1;
            goto cleanup;
        }
    }

    memcpy(centroid_copy, refit.centroids, centroid_bytes);
    if (gs_baseline_kmeans_assign(
            validation,
            GS_TEST_VALIDATION_ROWS,
            gs_runtime_feature_count,
            refit.centroids,
            k,
            refit_validation_assignments,
            refit_validation_distances
        ) != GS_BASELINE_KMEANS_SUCCESS ||
        memcmp(centroid_copy, refit.centroids, centroid_bytes) != 0) {
        failed = 1;
        goto cleanup;
    }
    for (row = 0; row < GS_TEST_VALIDATION_ROWS; row++) {
        if (refit_validation_assignments[row] !=
                (size_t)validation_assignments[row] ||
            !gs_distance_equal(
                validation_distances[row],
                refit_validation_distances[row]
            )) {
            failed = 1;
            goto cleanup;
        }
    }

cleanup:
    gs_baseline_kmeans_result_free(&refit);
    free(centroid_copy);
    free(refit_validation_distances);
    free(refit_validation_assignments);
    return failed;
}


int main(int argc, char **argv)
{
    size_t k;
    size_t train_values;
    size_t validation_values;
    size_t centroid_values;
    size_t train_bytes;
    size_t validation_bytes;
    size_t centroid_bytes;
    size_t train_assignment_bytes;
    size_t validation_assignment_bytes;
    size_t train_size_t_bytes;
    size_t validation_size_t_bytes;
    size_t train_distance_bytes;
    size_t validation_distance_bytes;
    size_t k_u64_bytes;
    char cluster_hashes[GS_TEST_CLUSTER_INPUT_COUNT][
        GS_TEST_SHA256_LENGTH + 1
    ];
    char train_path[GS_TEST_PATH_LENGTH];
    char validation_path[GS_TEST_PATH_LENGTH];

    float *train = NULL;
    float *validation = NULL;
    double *centroids = NULL;
    double *centroid_copy = NULL;
    uint32_t *train_assignments = NULL;
    uint32_t *validation_assignments = NULL;
    double *train_distances = NULL;
    double *validation_distances = NULL;
    uint64_t *cluster_sizes = NULL;
    uint64_t *initial_indices = NULL;
    size_t *recomputed_train = NULL;
    size_t *recomputed_validation = NULL;
    double *recomputed_train_distances = NULL;
    double *recomputed_validation_distances = NULL;
    uint64_t *recomputed_sizes = NULL;
    GSSummary summary;
    GSPCAProfile profile;
    double train_inertia = 0.0;
    double validation_inertia = 0.0;
    size_t empty_clusters = 0u;
    size_t minimum_cluster_size = GS_TEST_TRAIN_ROWS;
    size_t maximum_cluster_size = 0u;
    size_t index;
    int rc = 1;

    memset(&summary, 0, sizeof(summary));
    memset(&profile, 0, sizeof(profile));
    if (setlocale(LC_ALL, "C") == NULL ||
        argc != 6 ||
        !gs_platform_is_valid() ||
        gs_select_pca_profile(argv[2], &profile) != 0 ||
        gs_validate_pca_metadata(argv[1], &profile) != 0 ||
        gs_parse_size_value(argv[4], &k) != 0 ||
        (k != 22u && k != 36u) ||
        gs_validate_final_path(argv[1], argv[3], k) != 0 ||
        !gs_commit_is_valid(argv[5])) {
        fprintf(stderr, "Errore: argomenti validatore non validi.\n");
        return 1;
    }
    gs_runtime_feature_count = profile.components;

    if (gs_checked_product(
            GS_TEST_TRAIN_ROWS,
            gs_runtime_feature_count,
            &train_values
        ) != 0 ||
        gs_checked_product(
            GS_TEST_VALIDATION_ROWS,
            gs_runtime_feature_count,
            &validation_values
        ) != 0 ||
        gs_checked_product(
            k, gs_runtime_feature_count, &centroid_values
        ) != 0 ||
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
            GS_TEST_TRAIN_ROWS,
            sizeof(uint32_t),
            &train_assignment_bytes
        ) != 0 ||
        gs_checked_product(
            GS_TEST_VALIDATION_ROWS,
            sizeof(uint32_t),
            &validation_assignment_bytes
        ) != 0 ||
        gs_checked_product(
            GS_TEST_TRAIN_ROWS,
            sizeof(size_t),
            &train_size_t_bytes
        ) != 0 ||
        gs_checked_product(
            GS_TEST_VALIDATION_ROWS,
            sizeof(size_t),
            &validation_size_t_bytes
        ) != 0 ||
        gs_checked_product(
            GS_TEST_TRAIN_ROWS,
            sizeof(double),
            &train_distance_bytes
        ) != 0 ||
        gs_checked_product(
            GS_TEST_VALIDATION_ROWS,
            sizeof(double),
            &validation_distance_bytes
        ) != 0 ||
        gs_checked_product(k, sizeof(uint64_t), &k_u64_bytes) != 0 ||
        gs_runtime_feature_count == 0u) {
        fprintf(stderr, "Errore: overflow validatore.\n");
        return 1;
    }

    if (gs_validate_pca_contract(
            argv[1], &profile, cluster_hashes, train_path,
            validation_path, train_bytes, validation_bytes
        ) != 0 ||
        gs_validate_output_inventory_and_sizes(argv[3], k) != 0 ||
        gs_verify_output_sha256(
            argv[1], argv[3], cluster_hashes
        ) != 0 ||
        gs_parse_summary(
            argv[3], &profile, k, argv[5], &summary
        ) != 0) {
        fprintf(stderr, "Errore: contratto directory o metadata fallito.\n");
        goto cleanup;
    }
    printf("[PASS] pca_and_kmeans_inventory_metadata_sha256\n");

    train = (float *)malloc(train_bytes);
    validation = (float *)malloc(validation_bytes);
    centroids = (double *)malloc(centroid_bytes);
    centroid_copy = (double *)malloc(centroid_bytes);
    train_assignments =
        (uint32_t *)malloc(train_assignment_bytes);
    validation_assignments =
        (uint32_t *)malloc(validation_assignment_bytes);
    train_distances = (double *)malloc(train_distance_bytes);
    validation_distances =
        (double *)malloc(validation_distance_bytes);
    cluster_sizes = (uint64_t *)malloc(k_u64_bytes);
    initial_indices = (uint64_t *)malloc(k_u64_bytes);
    recomputed_train = (size_t *)malloc(train_size_t_bytes);
    recomputed_validation =
        (size_t *)malloc(validation_size_t_bytes);
    recomputed_train_distances =
        (double *)malloc(train_distance_bytes);
    recomputed_validation_distances =
        (double *)malloc(validation_distance_bytes);
    recomputed_sizes = (uint64_t *)calloc(k, sizeof(uint64_t));

    if (train == NULL || validation == NULL ||
        centroids == NULL || centroid_copy == NULL ||
        train_assignments == NULL || validation_assignments == NULL ||
        train_distances == NULL || validation_distances == NULL ||
        cluster_sizes == NULL || initial_indices == NULL ||
        recomputed_train == NULL || recomputed_validation == NULL ||
        recomputed_train_distances == NULL ||
        recomputed_validation_distances == NULL ||
        recomputed_sizes == NULL ||
        gs_read_float_matrix(
            train_path, train, train_values, train_bytes
        ) != 0 ||
        gs_read_float_matrix(
            validation_path,
            validation,
            validation_values,
            validation_bytes
        ) != 0 ||
        gs_read_binary(
            argv[3],
            "centroids_f64.bin",
            centroids,
            sizeof(double),
            centroid_values
        ) != 0 ||
        gs_read_binary(
            argv[3],
            "train_assignments_u32.bin",
            train_assignments,
            sizeof(uint32_t),
            GS_TEST_TRAIN_ROWS
        ) != 0 ||
        gs_read_binary(
            argv[3],
            "val_assignments_u32.bin",
            validation_assignments,
            sizeof(uint32_t),
            GS_TEST_VALIDATION_ROWS
        ) != 0 ||
        gs_read_binary(
            argv[3],
            "train_squared_distances_f64.bin",
            train_distances,
            sizeof(double),
            GS_TEST_TRAIN_ROWS
        ) != 0 ||
        gs_read_binary(
            argv[3],
            "val_squared_distances_f64.bin",
            validation_distances,
            sizeof(double),
            GS_TEST_VALIDATION_ROWS
        ) != 0 ||
        gs_read_binary(
            argv[3],
            "cluster_sizes_train_u64.bin",
            cluster_sizes,
            sizeof(uint64_t),
            k
        ) != 0 ||
        gs_read_binary(
            argv[3],
            "initial_centroid_indices_u64.bin",
            initial_indices,
            sizeof(uint64_t),
            k
        ) != 0) {
        fprintf(stderr, "Errore: lettura binaria validatore fallita.\n");
        goto cleanup;
    }
    printf("[PASS] binary_shapes_and_finite_inputs\n");

    for (index = 0; index < centroid_values; index++) {
        if (!isfinite(centroids[index])) {
            goto cleanup;
        }
    }
    for (index = 0; index < GS_TEST_TRAIN_ROWS; index++) {
        if (train_assignments[index] >= k ||
            !isfinite(train_distances[index]) ||
            train_distances[index] < 0.0) {
            goto cleanup;
        }
    }
    for (index = 0; index < GS_TEST_VALIDATION_ROWS; index++) {
        if (validation_assignments[index] >= k ||
            !isfinite(validation_distances[index]) ||
            validation_distances[index] < 0.0) {
            goto cleanup;
        }
    }

    if (gs_validate_assignment_csv(
            argv[3],
            "train_assignments.csv",
            train_assignments,
            train_distances,
            GS_TEST_TRAIN_ROWS,
            k
        ) != 0 ||
        gs_validate_assignment_csv(
            argv[3],
            "val_assignments.csv",
            validation_assignments,
            validation_distances,
            GS_TEST_VALIDATION_ROWS,
            k
        ) != 0 ||
        gs_validate_cluster_sizes_csv(
            argv[3], cluster_sizes, k
        ) != 0 ||
        gs_validate_initial_centroids_csv(
            argv[3], initial_indices, k
        ) != 0) {
        fprintf(stderr, "Errore: contratto CSV fallito.\n");
        goto cleanup;
    }
    printf("[PASS] exact_csv_headers_rows_and_binary_roundtrip\n");

    if (gs_baseline_kmeans_assign(
            train,
            GS_TEST_TRAIN_ROWS,
            gs_runtime_feature_count,
            centroids,
            k,
            recomputed_train,
            recomputed_train_distances
        ) != GS_BASELINE_KMEANS_SUCCESS) {
        goto cleanup;
    }
    for (index = 0; index < GS_TEST_TRAIN_ROWS; index++) {
        if (recomputed_train[index] !=
                (size_t)train_assignments[index] ||
            !gs_distance_equal(
                train_distances[index],
                recomputed_train_distances[index]
            )) {
            goto cleanup;
        }
        recomputed_sizes[train_assignments[index]]++;
        train_inertia += recomputed_train_distances[index];
        if (!isfinite(train_inertia)) {
            goto cleanup;
        }
    }
    printf("TRAIN_ASSIGNMENTS_RECOMPUTED_MATCH=true\n");
    printf("TRAIN_DISTANCES_RECOMPUTED_MATCH=true\n");

    memcpy(centroid_copy, centroids, centroid_bytes);
    if (gs_baseline_kmeans_assign(
            validation,
            GS_TEST_VALIDATION_ROWS,
            gs_runtime_feature_count,
            centroids,
            k,
            recomputed_validation,
            recomputed_validation_distances
        ) != GS_BASELINE_KMEANS_SUCCESS ||
        memcmp(centroid_copy, centroids, centroid_bytes) != 0) {
        goto cleanup;
    }
    for (index = 0; index < GS_TEST_VALIDATION_ROWS; index++) {
        if (recomputed_validation[index] !=
                (size_t)validation_assignments[index] ||
            !gs_distance_equal(
                validation_distances[index],
                recomputed_validation_distances[index]
            )) {
            goto cleanup;
        }
        validation_inertia += recomputed_validation_distances[index];
        if (!isfinite(validation_inertia)) {
            goto cleanup;
        }
    }
    printf("VALIDATION_ASSIGNMENTS_RECOMPUTED_MATCH=true\n");
    printf("VALIDATION_DISTANCES_RECOMPUTED_MATCH=true\n");
    printf("VALIDATION_CENTROIDS_UNCHANGED=true\n");

    for (index = 0; index < k; index++) {
        size_t left;

        if (cluster_sizes[index] != recomputed_sizes[index] ||
            initial_indices[index] >= GS_TEST_TRAIN_ROWS) {
            goto cleanup;
        }
        if (cluster_sizes[index] == 0u) {
            empty_clusters++;
        }
        if (cluster_sizes[index] < minimum_cluster_size) {
            minimum_cluster_size = (size_t)cluster_sizes[index];
        }
        if (cluster_sizes[index] > maximum_cluster_size) {
            maximum_cluster_size = (size_t)cluster_sizes[index];
        }
        for (left = 0; left < index; left++) {
            if (initial_indices[left] == initial_indices[index]) {
                goto cleanup;
            }
        }
    }

    if (empty_clusters != summary.empty_final_clusters ||
        minimum_cluster_size != summary.minimum_cluster_size ||
        maximum_cluster_size != summary.maximum_cluster_size ||
        !gs_inertia_equal(
            summary.final_train_inertia, train_inertia
        ) ||
        !gs_inertia_equal(
            summary.validation_inertia, validation_inertia
        ) ||
        gs_recompute_centroids_from_train(
            train,
            train_assignments,
            cluster_sizes,
            centroids,
            k
        ) != 0) {
        goto cleanup;
    }
    printf("CENTROIDS_RECOMPUTED_FROM_TRAIN_MATCH=true\n");
    printf("TRAIN_INERTIA_RECOMPUTED_MATCH=true\n");
    printf("VALIDATION_INERTIA_RECOMPUTED_MATCH=true\n");
    printf("CLUSTER_SIZES_RECOMPUTED_MATCH=true\n");
    printf("INITIAL_INDICES_VALID=true\n");

    if (gs_refit_matches_artifacts(
            train,
            validation,
            k,
            &summary,
            centroids,
            train_assignments,
            validation_assignments,
            validation_distances,
            cluster_sizes,
            initial_indices
        ) != 0) {
        fprintf(stderr, "Errore: refit train-only non identico.\n");
        goto cleanup;
    }
    printf("FULL_TRAIN_ONLY_REFIT_BITWISE=true\n");
    printf("ALL_PCA_KMEANS_CHECKSUMS_MATCH=true\n");
    printf("ALL MINI112 PCA KMEANS ARTIFACT TESTS PASSED\n");
    rc = 0;

cleanup:
    free(recomputed_sizes);
    free(recomputed_validation_distances);
    free(recomputed_train_distances);
    free(recomputed_validation);
    free(recomputed_train);
    free(initial_indices);
    free(cluster_sizes);
    free(validation_distances);
    free(train_distances);
    free(validation_assignments);
    free(train_assignments);
    free(centroid_copy);
    free(centroids);
    free(validation);
    free(train);
    return rc;
}
