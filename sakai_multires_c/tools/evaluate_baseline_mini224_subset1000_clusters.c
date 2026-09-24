#include "baseline_clustering_metrics.h"
#include "vae_event_dataset.h"

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
#define GS_MKDIR(path) _mkdir(path)
#define GS_RMDIR(path) _rmdir(path)
#else
#include <unistd.h>
#define GS_MKDIR(path) mkdir(path, 0777)
#define GS_RMDIR(path) rmdir(path)
#endif

enum {
    GS_LABEL_COUNT = 22,
    GS_TRAIN_ROWS = 800,
    GS_VALIDATION_ROWS = 200,
    GS_EVENT_COUNT = 8583,
    GS_PATH_SIZE = 4096,
    GS_LINE_SIZE = 8192,
    GS_SHA_BLOCK_SIZE = 64,
    GS_SHA_TEXT_SIZE = 65
};

#define GS_PARTIAL_SUFFIX ".partial"
#define GS_FAILURE_ENV "GS_MINI224_SUBSET1000_EVALUATION_TEST_FAIL_AFTER_PARTIAL"
#define GS_BASE_CODE_COMMIT \
    "d87ca996e7922d6873a54d952aec27cdb99a5998"
#define GS_KMEANS_CODE_COMMIT \
    "78f45052db766968e3058454781bdf82c5833188"

static const char *const gs_labels[GS_LABEL_COUNT] = {
    "1080Lines", "1400Ripples", "Air_Compressor", "Blip", "Chirp",
    "Extremely_Loud", "Helix", "Koi_Fish", "Light_Modulation",
    "Low_Frequency_Burst", "Low_Frequency_Lines", "No_Glitch",
    "None_of_the_Above", "Paired_Doves", "Power_Line",
    "Repeating_Blips", "Scattered_Light", "Scratchy", "Tomte",
    "Violin_Mode", "Wandering_Line", "Whistle"
};

static const char *const gs_base_files[10] = {
    "feature_spec.txt", "train_features_raw_f32.bin",
    "val_features_raw_f32.bin", "train_features_std_f32.bin",
    "val_features_std_f32.bin", "scaler.csv", "train_rows.csv",
    "val_rows.csv", "summary.txt", "sha256.txt"
};

static const size_t gs_base_sizes[10] = {
    1116u, 3276800u, 819200u, 3276800u, 819200u,
    54215u, 30060u, 8392u, 1626u, 1241u
};

static const char *const gs_kmeans_files[13] = {
    "centroids_f64.bin", "train_assignments_u32.bin",
    "val_assignments_u32.bin", "train_squared_distances_f64.bin",
    "val_squared_distances_f64.bin", "cluster_sizes_train_u64.bin",
    "initial_centroid_indices_u64.bin", "train_assignments.csv",
    "val_assignments.csv", "cluster_sizes.csv", "initial_centroids.csv",
    "summary.txt", "sha256.txt"
};

static const char *const gs_output_files[10] = {
    "label_map.csv", "metrics.csv", "summary.txt",
    "train_cluster_summary.csv", "train_contingency_counts.csv",
    "train_label_summary.csv", "val_cluster_summary.csv",
    "val_contingency_counts.csv", "val_label_summary.csv", "sha256.txt"
};

static const char *const gs_historical_artifacts[9] = {
    "label_map.csv", "train_contingency_counts.csv",
    "val_contingency_counts.csv", "train_cluster_summary.csv",
    "val_cluster_summary.csv", "train_label_summary.csv",
    "val_label_summary.csv", "metrics.csv", "summary.txt"
};

typedef struct {
    uint32_t state[8];
    uint64_t bit_count;
    unsigned char block[GS_SHA_BLOCK_SIZE];
    size_t used;
} GSSha256;

typedef struct {
    size_t k;
    char code_commit[256];
} GSKMeansMetadata;

typedef struct {
    size_t total;
    size_t occupied;
    size_t first;
    uint64_t first_count;
    size_t second;
    uint64_t second_count;
    double first_fraction;
} GSLabelStatistics;

typedef struct {
    const char *kind;
    const char *name;
    const char *path;
} GSChecksumEntry;

static uint32_t gs_rotr32(uint32_t x, unsigned int n)
{
    return (x >> n) | (x << (32u - n));
}

static void gs_sha_transform(GSSha256 *ctx, const unsigned char block[64])
{
    static const uint32_t k[64] = {
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
    uint32_t w[64];
    uint32_t a, b, c, d, e, f, g, h;
    size_t i;

    for (i = 0; i < 16; i++) {
        size_t p = i * 4u;
        w[i] = ((uint32_t)block[p] << 24) |
               ((uint32_t)block[p + 1u] << 16) |
               ((uint32_t)block[p + 2u] << 8) |
               (uint32_t)block[p + 3u];
    }
    for (i = 16; i < 64; i++) {
        uint32_t s0 = gs_rotr32(w[i - 15u], 7) ^
                      gs_rotr32(w[i - 15u], 18) ^ (w[i - 15u] >> 3);
        uint32_t s1 = gs_rotr32(w[i - 2u], 17) ^
                      gs_rotr32(w[i - 2u], 19) ^ (w[i - 2u] >> 10);
        w[i] = w[i - 16u] + s0 + w[i - 7u] + s1;
    }
    a = ctx->state[0]; b = ctx->state[1]; c = ctx->state[2];
    d = ctx->state[3]; e = ctx->state[4]; f = ctx->state[5];
    g = ctx->state[6]; h = ctx->state[7];
    for (i = 0; i < 64; i++) {
        uint32_t s1 = gs_rotr32(e, 6) ^ gs_rotr32(e, 11) ^
                      gs_rotr32(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t t1 = h + s1 + ch + k[i] + w[i];
        uint32_t s0 = gs_rotr32(a, 2) ^ gs_rotr32(a, 13) ^
                      gs_rotr32(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = s0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    ctx->state[0] += a; ctx->state[1] += b;
    ctx->state[2] += c; ctx->state[3] += d;
    ctx->state[4] += e; ctx->state[5] += f;
    ctx->state[6] += g; ctx->state[7] += h;
}

static void gs_sha_init(GSSha256 *ctx)
{
    static const uint32_t initial[8] = {
        UINT32_C(0x6a09e667), UINT32_C(0xbb67ae85),
        UINT32_C(0x3c6ef372), UINT32_C(0xa54ff53a),
        UINT32_C(0x510e527f), UINT32_C(0x9b05688c),
        UINT32_C(0x1f83d9ab), UINT32_C(0x5be0cd19)
    };
    memcpy(ctx->state, initial, sizeof(initial));
    ctx->bit_count = UINT64_C(0);
    ctx->used = 0u;
}

static int gs_sha_update(GSSha256 *ctx, const unsigned char *data, size_t size)
{
    size_t offset = 0u;
    if (ctx == NULL || (data == NULL && size != 0u) ||
        size > (UINT64_MAX - ctx->bit_count) / UINT64_C(8)) {
        return 1;
    }
    ctx->bit_count += (uint64_t)size * UINT64_C(8);
    while (offset < size) {
        size_t available = 64u - ctx->used;
        size_t take = size - offset < available ? size - offset : available;
        memcpy(ctx->block + ctx->used, data + offset, take);
        ctx->used += take;
        offset += take;
        if (ctx->used == 64u) {
            gs_sha_transform(ctx, ctx->block);
            ctx->used = 0u;
        }
    }
    return 0;
}

static void gs_sha_final(GSSha256 *ctx, unsigned char digest[32])
{
    uint64_t bits = ctx->bit_count;
    size_t i;
    ctx->block[ctx->used++] = 0x80u;
    if (ctx->used > 56u) {
        memset(ctx->block + ctx->used, 0, 64u - ctx->used);
        gs_sha_transform(ctx, ctx->block);
        ctx->used = 0u;
    }
    memset(ctx->block + ctx->used, 0, 56u - ctx->used);
    for (i = 0; i < 8u; i++) {
        ctx->block[63u - i] = (unsigned char)(bits & UINT64_C(0xff));
        bits >>= 8;
    }
    gs_sha_transform(ctx, ctx->block);
    for (i = 0; i < 8u; i++) {
        digest[i * 4u] = (unsigned char)(ctx->state[i] >> 24);
        digest[i * 4u + 1u] = (unsigned char)(ctx->state[i] >> 16);
        digest[i * 4u + 2u] = (unsigned char)(ctx->state[i] >> 8);
        digest[i * 4u + 3u] = (unsigned char)ctx->state[i];
    }
}

static int gs_sha_file(const char *path, char text[GS_SHA_TEXT_SIZE])
{
    static const char hex[] = "0123456789abcdef";
    unsigned char buffer[32768];
    unsigned char digest[32];
    GSSha256 ctx;
    FILE *file;
    size_t got;
    size_t i;
    int failed = 0;

    if (path == NULL || text == NULL || (file = fopen(path, "rb")) == NULL) {
        return 1;
    }
    gs_sha_init(&ctx);
    while ((got = fread(buffer, 1u, sizeof(buffer), file)) != 0u) {
        if (gs_sha_update(&ctx, buffer, got) != 0) {
            failed = 1;
            break;
        }
    }
    if (ferror(file) || fclose(file) != 0) {
        failed = 1;
    }
    if (failed) {
        return 1;
    }
    gs_sha_final(&ctx, digest);
    for (i = 0; i < 32u; i++) {
        text[i * 2u] = hex[digest[i] >> 4];
        text[i * 2u + 1u] = hex[digest[i] & 15u];
    }
    text[64] = '\0';
    return 0;
}

static int gs_checked_product(size_t left, size_t right, size_t *product)
{
    if (product == NULL || (right != 0u && left > SIZE_MAX / right)) {
        return 1;
    }
    *product = left * right;
    return 0;
}

static int gs_platform_valid(void)
{
    uint16_t one = UINT16_C(1);
    const unsigned char *bytes = (const unsigned char *)&one;
    return CHAR_BIT == 8 && sizeof(uint32_t) == 4u &&
           sizeof(uint64_t) == 8u && sizeof(double) == 8u &&
           FLT_RADIX == 2 && DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024 &&
           bytes[0] == 1u;
}

static int gs_value_valid(const char *value)
{
    return value != NULL && value[0] != '\0' &&
           strchr(value, '\r') == NULL && strchr(value, '\n') == NULL;
}

static int gs_parse_size(const char *text, size_t *value)
{
    char *end = NULL;
    unsigned long long parsed;
    if (!gs_value_valid(text) || value == NULL || text[0] == '-') {
        return 1;
    }
    errno = 0;
    parsed = strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' ||
        parsed > (unsigned long long)SIZE_MAX) {
        return 1;
    }
    *value = (size_t)parsed;
    return 0;
}

static int gs_join(
    const char *directory,
    const char *name,
    char *path,
    size_t capacity
)
{
    size_t a, b, need;
    int separator;
    if (directory == NULL || name == NULL || path == NULL || capacity == 0u) {
        return 1;
    }
    a = strlen(directory);
    b = strlen(name);
    separator = a != 0u && directory[a - 1u] != '/' &&
                directory[a - 1u] != '\\';
    if (a > SIZE_MAX - b || (separator && a + b == SIZE_MAX)) {
        return 1;
    }
    need = a + b + (separator ? 1u : 0u);
    if (need >= capacity) {
        return 1;
    }
    memcpy(path, directory, a);
    if (separator) {
        path[a++] = '/';
    }
    memcpy(path + a, name, b);
    path[a + b] = '\0';
    return 0;
}

static int gs_path_state(const char *path, int *is_regular, int *is_directory)
{
    struct stat info;
    if (is_regular != NULL) *is_regular = 0;
    if (is_directory != NULL) *is_directory = 0;
    errno = 0;
    if (stat(path, &info) != 0) {
        return errno == ENOENT ? 0 : -1;
    }
    if (is_regular != NULL) *is_regular = S_ISREG(info.st_mode) ? 1 : 0;
    if (is_directory != NULL) *is_directory = S_ISDIR(info.st_mode) ? 1 : 0;
    return 1;
}

static int gs_file_size(const char *path, size_t *size)
{
    struct stat info;
    if (path == NULL || size == NULL || stat(path, &info) != 0 ||
        !S_ISREG(info.st_mode) || info.st_size < 0 ||
        (uintmax_t)info.st_size > (uintmax_t)SIZE_MAX) {
        return 1;
    }
    *size = (size_t)info.st_size;
    return 0;
}

static const char *gs_basename(const char *path)
{
    const char *base = path;
    const char *cursor;
    if (path == NULL) return NULL;
    for (cursor = path; *cursor != '\0'; cursor++) {
        if (*cursor == '/' || *cursor == '\\') base = cursor + 1;
    }
    return base;
}

static int gs_canonical_existing(const char *path, char *output, size_t capacity)
{
    char *resolved;
    size_t length;
#ifdef _WIN32
    resolved = _fullpath(NULL, path, 0);
#else
    resolved = realpath(path, NULL);
#endif
    if (resolved == NULL) return 1;
    length = strlen(resolved);
    while (length > 1u &&
           (resolved[length - 1u] == '/' || resolved[length - 1u] == '\\')) {
        resolved[--length] = '\0';
    }
    if (length >= capacity) {
        free(resolved);
        return 1;
    }
    memcpy(output, resolved, length + 1u);
    free(resolved);
    for (length = 0; output[length] != '\0'; length++) {
        if (output[length] == '\\') output[length] = '/';
#ifdef _WIN32
        if (output[length] >= 'A' && output[length] <= 'Z') {
            output[length] = (char)(output[length] - 'A' + 'a');
        }
#endif
    }
    return 0;
}

static int gs_validate_kmeans_location(
    const char *base_directory,
    const char *kmeans_directory,
    size_t k
)
{
    char base[GS_PATH_SIZE];
    char actual[GS_PATH_SIZE];
    char expected[GS_PATH_SIZE];
    char suffix[64];
    int written;
    if (gs_canonical_existing(base_directory, base, sizeof(base)) != 0 ||
        gs_canonical_existing(kmeans_directory, actual, sizeof(actual)) != 0 ||
        (written = snprintf(suffix, sizeof(suffix),
                            "kmeans_seed123/k%lu", (unsigned long)k)) < 0 ||
        (size_t)written >= sizeof(suffix) ||
        gs_join(base, suffix, expected, sizeof(expected)) != 0) {
        return 1;
    }
    return strcmp(actual, expected) == 0 ? 0 : 1;
}

static int gs_parent_path(
    const char *path,
    char *parent,
    size_t capacity
)
{
    size_t length, split;
    if (!gs_value_valid(path) || parent == NULL || capacity == 0u) return 1;
    length = strlen(path);
    while (length != 0u &&
           (path[length - 1u] == '/' || path[length - 1u] == '\\')) {
        length--;
    }
    if (length == 0u) return 1;
    split = length;
    while (split != 0u &&
           path[split - 1u] != '/' && path[split - 1u] != '\\') {
        split--;
    }
    while (split > 1u &&
           (path[split - 1u] == '/' || path[split - 1u] == '\\')) {
        split--;
    }
    if (split == 0u || split >= capacity) return 1;
    memcpy(parent, path, split);
    parent[split] = '\0';
    return 0;
}

static int gs_name_index(
    const char *name,
    const char *const *names,
    size_t count,
    size_t *index
)
{
    size_t i;
    for (i = 0; i < count; i++) {
        if (strcmp(name, names[i]) == 0) {
            if (index != NULL) *index = i;
            return 1;
        }
    }
    return 0;
}

static int gs_exact_files(
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
        size_t index;
        char path[GS_PATH_SIZE];
        int regular, is_directory;
        errno = 0;
        entry = readdir(stream);
        if (entry == NULL) {
            if (errno != 0) failed = 1;
            break;
        }
        if (strcmp(entry->d_name, ".") == 0 ||
            strcmp(entry->d_name, "..") == 0) continue;
        if (!gs_name_index(entry->d_name, names, count, &index) ||
            seen[index] ||
            gs_join(directory, entry->d_name, path, sizeof(path)) != 0 ||
            gs_path_state(path, &regular, &is_directory) != 1 ||
            !regular || is_directory) {
            failed = 1;
            break;
        }
        seen[index] = 1u;
    }
    if (closedir(stream) != 0) failed = 1;
    for (i = 0; !failed && i < count; i++) {
        if (!seen[i]) failed = 1;
    }
    free(seen);
    return failed;
}

static int gs_base_inventory(const char *directory)
{
    unsigned char seen[10] = {0};
    DIR *stream = opendir(directory);
    struct dirent *entry;
    int kmeans_seen = 0;
    int evaluation_seen = 0;
    int failed = stream == NULL;
    size_t i;
    while (!failed) {
        size_t index;
        char path[GS_PATH_SIZE];
        int regular, is_directory;
        errno = 0;
        entry = readdir(stream);
        if (entry == NULL) {
            if (errno != 0) failed = 1;
            break;
        }
        if (strcmp(entry->d_name, ".") == 0 ||
            strcmp(entry->d_name, "..") == 0) continue;
        if (gs_join(directory, entry->d_name, path, sizeof(path)) != 0 ||
            gs_path_state(path, &regular, &is_directory) != 1) {
            failed = 1;
            break;
        }
        if (gs_name_index(entry->d_name, gs_base_files, 10u, &index)) {
            if (seen[index] || !regular) failed = 1;
            seen[index] = 1u;
        } else if (strcmp(entry->d_name, "kmeans_seed123") == 0 &&
                   !kmeans_seen && is_directory) {
            kmeans_seen = 1;
        } else if (strcmp(
                       entry->d_name,
                       "label_evaluation_seed123"
                   ) == 0 &&
                   !evaluation_seen && is_directory) {
            evaluation_seen = 1;
        } else {
            failed = 1;
        }
    }
    if (stream != NULL && closedir(stream) != 0) failed = 1;
    for (i = 0; !failed && i < 10u; i++) if (!seen[i]) failed = 1;
    return failed;
}

static int gs_verify_checksum_manifest(
    const char *sha_path,
    const GSChecksumEntry *entries,
    size_t count
)
{
    FILE *file;
    char line[GS_LINE_SIZE];
    size_t index = 0u;
    if (sha_path == NULL || entries == NULL ||
        (file = fopen(sha_path, "rb")) == NULL) {
        return 1;
    }
    while (index < count && fgets(line, sizeof(line), file) != NULL) {
        char kind[16], hash[65], name[GS_PATH_SIZE], extra;
        char actual[65];
        int fields = sscanf(line, "%15s %64s %4095s %c",
                            kind, hash, name, &extra);
        if (fields != 3 || strcmp(kind, entries[index].kind) != 0 ||
            strcmp(name, entries[index].name) != 0 ||
            strlen(hash) != 64u ||
            gs_sha_file(entries[index].path, actual) != 0 ||
            strcmp(hash, actual) != 0) {
            fclose(file);
            return 1;
        }
        index++;
    }
    if (index != count || fgets(line, sizeof(line), file) != NULL ||
        ferror(file) || fclose(file) != 0) {
        return 1;
    }
    return 0;
}

static int gs_build_base_entries(
    const char *manifest_path,
    const char *base,
    GSChecksumEntry entries[13],
    char paths[12][GS_PATH_SIZE]
)
{
    static const char *const input_names[4] = {
        "manifest_events_272.csv", "manifest_tensors_224_central.csv",
        "subset224_train_800_seed123.csv", "subset224_val_200_seed123.csv"
    };
    static const char *const artifact_names[9] = {
        "feature_spec.txt", "train_features_raw_f32.bin",
        "val_features_raw_f32.bin", "train_features_std_f32.bin",
        "val_features_std_f32.bin", "scaler.csv", "train_rows.csv",
        "val_rows.csv", "summary.txt"
    };
    static const char *const input_source_paths[4] = {
        "manifest_events_272.csv", "manifest_tensors_224_central.csv",
        "sakai_multires_c/configs/subset224_train_800_seed123.csv",
        "sakai_multires_c/configs/subset224_val_200_seed123.csv"
    };
    size_t i;
    if (strcmp(gs_basename(manifest_path), input_names[0]) != 0) return 1;
    entries[0].kind = "INPUT"; entries[0].name = input_names[0];
    entries[0].path = manifest_path;
    for (i = 1; i < 4u; i++) {
        if (strlen(input_source_paths[i]) >= GS_PATH_SIZE) return 1;
        strcpy(paths[i - 1u], input_source_paths[i]);
        entries[i].kind = "INPUT"; entries[i].name = input_names[i];
        entries[i].path = paths[i - 1u];
    }
    for (i = 0; i < 9u; i++) {
        if (gs_join(base, artifact_names[i], paths[i + 3u],
                    GS_PATH_SIZE) != 0) return 1;
        entries[i + 4u].kind = "ARTIFACT";
        entries[i + 4u].name = artifact_names[i];
        entries[i + 4u].path = paths[i + 3u];
    }
    return 0;
}

static int gs_validate_base_metadata(const char *base);

static int gs_validate_base(const char *manifest_path, const char *base)
{
    GSChecksumEntry entries[13];
    char paths[12][GS_PATH_SIZE];
    char sha_path[GS_PATH_SIZE];
    size_t i, actual;
    if (strcmp(
            gs_basename(base),
            "mini224_subset1000_pool14_no_pca"
        ) != 0 ||
        gs_base_inventory(base) != 0) {
        return 1;
    }
    for (i = 0; i < 10u; i++) {
        char path[GS_PATH_SIZE];
        if (gs_join(base, gs_base_files[i], path, sizeof(path)) != 0 ||
            gs_file_size(path, &actual) != 0 || actual != gs_base_sizes[i]) {
            return 1;
        }
    }
    if (gs_build_base_entries(manifest_path, base, entries, paths) != 0 ||
        gs_join(base, "sha256.txt", sha_path, sizeof(sha_path)) != 0 ||
        gs_verify_checksum_manifest(sha_path, entries, 13u) != 0 ||
        gs_validate_base_metadata(base) != 0) {
        return 1;
    }
    return 0;
}

static int gs_read_summary_value(
    const char *path,
    const char *key,
    char *value,
    size_t capacity
)
{
    FILE *file = fopen(path, "rb");
    char line[GS_LINE_SIZE];
    size_t key_length = strlen(key);
    int found = 0;
    if (file == NULL || capacity == 0u) return 1;
    while (fgets(line, sizeof(line), file) != NULL) {
        size_t length = strlen(line);
        while (length != 0u &&
               (line[length - 1u] == '\r' || line[length - 1u] == '\n')) {
            line[--length] = '\0';
        }
        if (strncmp(line, key, key_length) == 0 &&
            line[key_length] == '=') {
            const char *source = line + key_length + 1u;
            if (found || strlen(source) >= capacity) {
                fclose(file);
                return 1;
            }
            strcpy(value, source);
            found = 1;
        }
    }
    if (ferror(file) || fclose(file) != 0 || !found) return 1;
    return 0;
}

static int gs_validate_base_metadata(const char *base)
{
    static const char *const summary_keys[] = {
        "profile", "subset_profile", "train_rows", "validation_rows",
        "feature_count", "constant_feature_count", "standardization_fit",
        "standardization_ddof", "validation_transformed_without_refit",
        "scaler_unchanged_after_validation", "pca_used",
        "augmentation_used", "labels_used", "code_commit"
    };
    static const char *const summary_values[] = {
        "mini224_subset1000_pool14_no_pca",
        "subset224_seed123_prefix_800_200", "800", "200", "1024",
        "0", "train_only", "0", "true", "true", "false", "false",
        "false", GS_BASE_CODE_COMMIT
    };
    static const char *const feature_keys[] = {
        "profile", "subset_profile", "train_rows", "validation_rows",
        "feature_count", "scaler_fit", "pca", "augmentation",
        "labels_used", "code_commit"
    };
    static const char *const feature_values[] = {
        "mini224_subset1000_pool14_no_pca",
        "subset224_seed123_prefix_800_200", "800", "200", "1024",
        "train_only", "none", "false", "false", GS_BASE_CODE_COMMIT
    };
    char summary_path[GS_PATH_SIZE];
    char feature_path[GS_PATH_SIZE];
    char value[256];
    size_t i;

    if (gs_join(base, "summary.txt", summary_path, sizeof(summary_path)) != 0 ||
        gs_join(
            base, "feature_spec.txt", feature_path, sizeof(feature_path)
        ) != 0) {
        return 1;
    }
    for (i = 0; i < sizeof(summary_keys) / sizeof(summary_keys[0]); i++) {
        if (gs_read_summary_value(
                summary_path, summary_keys[i], value, sizeof(value)
            ) != 0 ||
            strcmp(value, summary_values[i]) != 0) {
            return 1;
        }
    }
    for (i = 0; i < sizeof(feature_keys) / sizeof(feature_keys[0]); i++) {
        if (gs_read_summary_value(
                feature_path, feature_keys[i], value, sizeof(value)
            ) != 0 ||
            strcmp(value, feature_values[i]) != 0) {
            return 1;
        }
    }
    return 0;
}

static int gs_validate_kmeans_summary(
    const char *directory,
    size_t k,
    GSKMeansMetadata *metadata
)
{
    static const char *const keys[] = {
        "profile", "input_profile", "subset_profile", "train_rows",
        "validation_rows", "feature_count", "k", "seed",
        "max_iterations", "converged", "fit_input",
        "validation_assigned_after_fit", "input_standardization", "pca",
        "augmentation", "labels_used", "base_code_commit", "code_commit"
    };
    static const char *const fixed[] = {
        "mini224_subset1000_pool14_no_pca_kmeans",
        "mini224_subset1000_pool14_no_pca",
        "subset224_seed123_prefix_800_200", "800", "200", "1024",
        NULL, "123", "300", "1", "train_only", "true",
        "train_only_population_ddof0", "false", "false", "false",
        GS_BASE_CODE_COMMIT, NULL
    };
    char path[GS_PATH_SIZE];
    char value[256];
    char k_text[32];
    size_t i;
    int written;
    if (metadata == NULL ||
        gs_join(directory, "summary.txt", path, sizeof(path)) != 0 ||
        (written = snprintf(k_text, sizeof(k_text), "%lu",
                            (unsigned long)k)) < 0 ||
        (size_t)written >= sizeof(k_text)) {
        return 1;
    }
    for (i = 0; i < sizeof(keys) / sizeof(keys[0]); i++) {
        if (gs_read_summary_value(path, keys[i], value, sizeof(value)) != 0) {
            return 1;
        }
        if (i == 6u) {
            if (strcmp(value, k_text) != 0) return 1;
        } else if (i == 17u) {
            if (!gs_value_valid(value) ||
                strlen(value) >= sizeof(metadata->code_commit)) return 1;
            strcpy(metadata->code_commit, value);
        } else if (strcmp(value, fixed[i]) != 0) {
            return 1;
        }
    }
    if (strcmp(metadata->code_commit, GS_KMEANS_CODE_COMMIT) != 0) {
        return 1;
    }
    metadata->k = k;
    return 0;
}

static int gs_build_kmeans_entries(
    const char *base,
    const char *directory,
    GSChecksumEntry entries[18],
    char paths[18][GS_PATH_SIZE]
)
{
    static const char *const input_names[6] = {
        "feature_spec.txt", "scaler.csv", "train_features_std_f32.bin",
        "val_features_std_f32.bin", "train_rows.csv", "val_rows.csv"
    };
    static const char *const artifact_names[12] = {
        "centroids_f64.bin", "train_assignments_u32.bin",
        "val_assignments_u32.bin", "train_squared_distances_f64.bin",
        "val_squared_distances_f64.bin", "cluster_sizes_train_u64.bin",
        "initial_centroid_indices_u64.bin", "train_assignments.csv",
        "val_assignments.csv", "cluster_sizes.csv", "initial_centroids.csv",
        "summary.txt"
    };
    size_t i;
    for (i = 0; i < 6u; i++) {
        if (gs_join(base, input_names[i], paths[i], GS_PATH_SIZE) != 0) return 1;
        entries[i].kind = "INPUT"; entries[i].name = input_names[i];
        entries[i].path = paths[i];
    }
    for (i = 0; i < 12u; i++) {
        if (gs_join(directory, artifact_names[i], paths[i + 6u],
                    GS_PATH_SIZE) != 0) return 1;
        entries[i + 6u].kind = "ARTIFACT";
        entries[i + 6u].name = artifact_names[i];
        entries[i + 6u].path = paths[i + 6u];
    }
    return 0;
}

static int gs_validate_kmeans(
    const char *base,
    const char *directory,
    size_t k,
    GSKMeansMetadata *metadata
)
{
    GSChecksumEntry entries[18];
    char paths[18][GS_PATH_SIZE];
    char sha_path[GS_PATH_SIZE];
    size_t sizes[7] = {0u, 3200u, 800u, 6400u, 1600u, 0u, 0u};
    size_t actual, k_bytes, centroid_values, centroid_bytes, i;
    char expected_base[16];
    int written;
    if (k != 22u && k != 36u) return 1;
    if ((written = snprintf(expected_base, sizeof(expected_base), "k%lu",
                            (unsigned long)k)) < 0 ||
        (size_t)written >= sizeof(expected_base) ||
        strcmp(gs_basename(directory), expected_base) != 0 ||
        strstr(gs_basename(directory), GS_PARTIAL_SUFFIX) != NULL ||
        gs_validate_kmeans_location(base, directory, k) != 0 ||
        gs_exact_files(directory, gs_kmeans_files, 13u) != 0 ||
        gs_checked_product(k, 1024u, &centroid_values) != 0 ||
        gs_checked_product(centroid_values, sizeof(double),
                           &centroid_bytes) != 0 ||
        gs_checked_product(k, sizeof(uint64_t), &k_bytes) != 0) {
        return 1;
    }
    sizes[0] = centroid_bytes; sizes[5] = k_bytes; sizes[6] = k_bytes;
    for (i = 0; i < 13u; i++) {
        char path[GS_PATH_SIZE];
        if (gs_join(directory, gs_kmeans_files[i], path, sizeof(path)) != 0 ||
            gs_file_size(path, &actual) != 0 ||
            (i < 7u ? actual != sizes[i] : actual == 0u)) {
            return 1;
        }
    }
    if (gs_build_kmeans_entries(base, directory, entries, paths) != 0 ||
        gs_join(directory, "sha256.txt", sha_path, sizeof(sha_path)) != 0 ||
        gs_verify_checksum_manifest(sha_path, entries, 18u) != 0 ||
        gs_validate_kmeans_summary(directory, k, metadata) != 0) {
        return 1;
    }
    return 0;
}

static int gs_label_index(const char *label)
{
    size_t i;
    if (label == NULL) return -1;
    for (i = 0; i < GS_LABEL_COUNT; i++) {
        if (strcmp(label, gs_labels[i]) == 0) return (int)i;
    }
    return -1;
}

static int gs_compare_strings(const void *left, const void *right)
{
    const char *const *a = (const char *const *)left;
    const char *const *b = (const char *const *)right;
    return strcmp(*a, *b);
}

static int gs_validate_manifest(const GSEventManifest *manifest)
{
    const char **ids;
    unsigned char labels[GS_LABEL_COUNT] = {0};
    size_t i, view, id_bytes;
    int failed = 0;
    if (manifest == NULL || manifest->records == NULL ||
        manifest->count != GS_EVENT_COUNT ||
        gs_checked_product(GS_EVENT_COUNT, sizeof(*ids), &id_bytes) != 0) {
        return 1;
    }
    ids = (const char **)malloc(id_bytes);
    if (ids == NULL) return 1;
    for (i = 0; i < GS_EVENT_COUNT; i++) {
        const GSEventRecord *record = &manifest->records[i];
        int label = gs_label_index(record->label);
        if (record->event_index != (int)(i + 1u) ||
            !gs_value_valid(record->event_id) || label < 0) {
            failed = 1;
            break;
        }
        labels[(size_t)label] = 1u;
        ids[i] = record->event_id;
        for (view = 0; view < GS_EVENT_VIEW_COUNT; view++) {
            if (!gs_value_valid(record->image_paths[view])) failed = 1;
        }
        if (failed) break;
    }
    if (!failed) {
        qsort(ids, GS_EVENT_COUNT, sizeof(*ids), gs_compare_strings);
        for (i = 1; i < GS_EVENT_COUNT; i++) {
            if (strcmp(ids[i - 1u], ids[i]) == 0) {
                failed = 1;
                break;
            }
        }
    }
    for (i = 0; !failed && i < GS_LABEL_COUNT; i++) {
        if (!labels[i]) failed = 1;
    }
    free(ids);
    return failed;
}

static void gs_strip_eol(char *line)
{
    size_t length = strlen(line);
    while (length != 0u &&
           (line[length - 1u] == '\r' || line[length - 1u] == '\n')) {
        line[--length] = '\0';
    }
}

static int gs_split_csv(char *line, char **fields, size_t expected)
{
    size_t count = 1u;
    char *cursor;
    if (line == NULL || fields == NULL || expected == 0u) return 1;
    gs_strip_eol(line);
    fields[0] = line;
    for (cursor = line; *cursor != '\0'; cursor++) {
        if (*cursor == ',') {
            if (count >= expected) return 1;
            *cursor = '\0';
            fields[count++] = cursor + 1;
        }
    }
    return count == expected ? 0 : 1;
}

static int gs_load_rows(
    const char *path,
    const char *split,
    size_t rows,
    const GSEventManifest *manifest,
    unsigned char *coverage,
    uint32_t *labels
)
{
    static const char header[] =
        "row_index,split_pos,manifest_index,event_index,event_id,split";
    FILE *file = fopen(path, "rb");
    char line[GS_LINE_SIZE];
    size_t row;
    int failed = file == NULL;
    if (!failed) {
        if (fgets(line, sizeof(line), file) == NULL) failed = 1;
        else {
            gs_strip_eol(line);
            if (strcmp(line, header) != 0) failed = 1;
        }
    }
    for (row = 0; !failed && row < rows; row++) {
        char *fields[6];
        size_t row_index, split_pos, manifest_index, event_index;
        const GSEventRecord *record;
        int label;
        if (fgets(line, sizeof(line), file) == NULL ||
            gs_split_csv(line, fields, 6u) != 0 ||
            gs_parse_size(fields[0], &row_index) != 0 ||
            gs_parse_size(fields[1], &split_pos) != 0 ||
            gs_parse_size(fields[2], &manifest_index) != 0 ||
            gs_parse_size(fields[3], &event_index) != 0 ||
            row_index != row || split_pos != row ||
            manifest_index >= manifest->count || coverage[manifest_index] ||
            strcmp(fields[5], split) != 0 ||
            !gs_value_valid(fields[4])) {
            failed = 1;
            break;
        }
        record = gs_event_manifest_get(manifest, manifest_index);
        label = record == NULL ? -1 : gs_label_index(record->label);
        if (record == NULL || label < 0 ||
            event_index != (size_t)record->event_index ||
            strcmp(fields[4], record->event_id) != 0) {
            failed = 1;
            break;
        }
        coverage[manifest_index] = 1u;
        labels[row] = (uint32_t)label;
    }
    if (!failed && fgets(line, sizeof(line), file) != NULL) failed = 1;
    if (file != NULL && (ferror(file) || fclose(file) != 0)) failed = 1;
    return failed;
}

static int gs_read_assignments(
    const char *path,
    uint32_t *values,
    size_t rows,
    size_t k
)
{
    FILE *file;
    size_t bytes, actual, row;
    int failed = 0;
    if (gs_checked_product(rows, sizeof(uint32_t), &bytes) != 0 ||
        gs_file_size(path, &actual) != 0 || actual != bytes ||
        (file = fopen(path, "rb")) == NULL) return 1;
    if (fread(values, sizeof(uint32_t), rows, file) != rows ||
        fgetc(file) != EOF || ferror(file)) failed = 1;
    if (fclose(file) != 0) failed = 1;
    if (failed) return 1;
    for (row = 0; row < rows; row++) if ((size_t)values[row] >= k) return 1;
    return 0;
}

static int gs_metrics_valid(
    const GSBaselineClusteringMetrics *metrics,
    size_t rows,
    size_t k
)
{
    uint64_t label_sum = UINT64_C(0);
    uint64_t cluster_sum = UINT64_C(0);
    uint64_t contingency_sum = UINT64_C(0);
    size_t cells, label, cluster;
    if (metrics == NULL || metrics->rows != rows ||
        metrics->label_count != GS_LABEL_COUNT ||
        metrics->cluster_count != k || metrics->contingency == NULL ||
        metrics->label_totals == NULL || metrics->cluster_totals == NULL ||
        gs_checked_product(GS_LABEL_COUNT, k, &cells) != 0 ||
        !isfinite(metrics->purity) ||
        !isfinite(metrics->normalized_mutual_information) ||
        !isfinite(metrics->adjusted_rand_index) ||
        !isfinite(metrics->homogeneity) ||
        !isfinite(metrics->completeness) ||
        metrics->purity < 0.0 || metrics->purity > 1.0 ||
        metrics->normalized_mutual_information < 0.0 ||
        metrics->normalized_mutual_information > 1.0 ||
        metrics->adjusted_rand_index < -1.0 ||
        metrics->adjusted_rand_index > 1.0 ||
        metrics->homogeneity < 0.0 || metrics->homogeneity > 1.0 ||
        metrics->completeness < 0.0 || metrics->completeness > 1.0) {
        return 0;
    }
    for (label = 0; label < GS_LABEL_COUNT; label++) {
        uint64_t row_sum = UINT64_C(0);
        for (cluster = 0; cluster < k; cluster++) {
            uint64_t value = metrics->contingency[label * k + cluster];
            if (row_sum > UINT64_MAX - value ||
                contingency_sum > UINT64_MAX - value) {
                return 0;
            }
            row_sum += value;
            contingency_sum += value;
        }
        if (row_sum != metrics->label_totals[label] ||
            label_sum > UINT64_MAX - row_sum) {
            return 0;
        }
        label_sum += row_sum;
    }
    for (cluster = 0; cluster < k; cluster++) {
        uint64_t column_sum = UINT64_C(0);
        for (label = 0; label < GS_LABEL_COUNT; label++) {
            uint64_t value = metrics->contingency[label * k + cluster];
            if (column_sum > UINT64_MAX - value) return 0;
            column_sum += value;
        }
        if (column_sum != metrics->cluster_totals[cluster] ||
            cluster_sum > UINT64_MAX - column_sum) {
            return 0;
        }
        cluster_sum += column_sum;
    }
    return cells != 0u &&
           label_sum == (uint64_t)rows &&
           cluster_sum == (uint64_t)rows &&
           contingency_sum == (uint64_t)rows;
}

static int gs_open_output(const char *directory, const char *name, FILE **file)
{
    char path[GS_PATH_SIZE];
    if (file == NULL ||
        gs_join(directory, name, path, sizeof(path)) != 0 ||
        (*file = fopen(path, "wb")) == NULL) return 1;
    return 0;
}

static int gs_finish_output(FILE *file, int failed)
{
    if (file == NULL) return 1;
    if (fflush(file) != 0 || ferror(file)) failed = 1;
    if (fclose(file) != 0) failed = 1;
    return failed;
}

static void gs_top_two_labels(
    const GSBaselineClusteringMetrics *metrics,
    size_t cluster,
    size_t *first,
    uint64_t *first_count,
    size_t *second,
    uint64_t *second_count
)
{
    size_t label;
    *first = SIZE_MAX; *second = SIZE_MAX;
    *first_count = 0u; *second_count = 0u;
    for (label = 0; label < GS_LABEL_COUNT; label++) {
        uint64_t count =
            metrics->contingency[label * metrics->cluster_count + cluster];
        if (*first == SIZE_MAX || count > *first_count) {
            *second = *first; *second_count = *first_count;
            *first = label; *first_count = count;
        } else if (*second == SIZE_MAX || count > *second_count) {
            *second = label; *second_count = count;
        }
    }
}

static void gs_label_statistics(
    const GSBaselineClusteringMetrics *metrics,
    size_t label,
    GSLabelStatistics *stats
)
{
    size_t cluster;
    stats->total = (size_t)metrics->label_totals[label];
    stats->occupied = 0u; stats->first = SIZE_MAX; stats->second = SIZE_MAX;
    stats->first_count = 0u; stats->second_count = 0u;
    for (cluster = 0; cluster < metrics->cluster_count; cluster++) {
        uint64_t count =
            metrics->contingency[label * metrics->cluster_count + cluster];
        if (count != 0u) stats->occupied++;
        if (stats->first == SIZE_MAX || count > stats->first_count) {
            stats->second = stats->first;
            stats->second_count = stats->first_count;
            stats->first = cluster;
            stats->first_count = count;
        } else if (stats->second == SIZE_MAX || count > stats->second_count) {
            stats->second = cluster;
            stats->second_count = count;
        }
    }
    stats->first_fraction = stats->total == 0u ? 0.0 :
        (double)stats->first_count / (double)stats->total;
}

static int gs_write_label_map(const char *directory)
{
    FILE *file = NULL;
    size_t label;
    int failed = gs_open_output(directory, "label_map.csv", &file);
    if (failed) return 1;
    if (fputs("label_index,label\n", file) == EOF) failed = 1;
    for (label = 0; !failed && label < GS_LABEL_COUNT; label++) {
        if (fprintf(file, "%lu,%s\n", (unsigned long)label,
                    gs_labels[label]) < 0) failed = 1;
    }
    return gs_finish_output(file, failed);
}

static int gs_write_contingency(
    const char *directory,
    const char *name,
    const GSBaselineClusteringMetrics *metrics
)
{
    FILE *file = NULL;
    size_t label, cluster;
    int failed = gs_open_output(directory, name, &file);
    if (failed) return 1;
    if (fputs("label_index,label", file) == EOF) failed = 1;
    for (cluster = 0; !failed && cluster < metrics->cluster_count; cluster++) {
        if (fprintf(file, ",cluster_%lu", (unsigned long)cluster) < 0) failed = 1;
    }
    if (!failed && fputs(",label_total\n", file) == EOF) failed = 1;
    for (label = 0; !failed && label < GS_LABEL_COUNT; label++) {
        if (fprintf(file, "%lu,%s", (unsigned long)label,
                    gs_labels[label]) < 0) failed = 1;
        for (cluster = 0; !failed && cluster < metrics->cluster_count; cluster++) {
            if (fprintf(file, ",%" PRIu64,
                        metrics->contingency[
                            label * metrics->cluster_count + cluster
                        ]) < 0) failed = 1;
        }
        if (!failed && fprintf(file, ",%" PRIu64 "\n",
                               metrics->label_totals[label]) < 0) failed = 1;
    }
    return gs_finish_output(file, failed);
}

static int gs_write_cluster_summary(
    const char *directory,
    const char *name,
    const GSBaselineClusteringMetrics *metrics
)
{
    FILE *file = NULL;
    size_t cluster;
    int failed = gs_open_output(directory, name, &file);
    if (failed) return 1;
    if (fputs(
            "cluster,size,dominant_label_index,dominant_label,dominant_count,"
            "cluster_purity,second_label_index,second_label,second_count\n",
            file
        ) == EOF) failed = 1;
    for (cluster = 0; !failed && cluster < metrics->cluster_count; cluster++) {
        uint64_t size = metrics->cluster_totals[cluster];
        if (size == 0u) {
            if (fprintf(file, "%lu,0,-1,NA,0,0,-1,NA,0\n",
                        (unsigned long)cluster) < 0) failed = 1;
        } else {
            size_t first, second;
            uint64_t first_count, second_count;
            gs_top_two_labels(metrics, cluster, &first, &first_count,
                              &second, &second_count);
            if (fprintf(
                    file, "%lu,%" PRIu64 ",%lu,%s,%" PRIu64
                    ",%.17g,%lu,%s,%" PRIu64 "\n",
                    (unsigned long)cluster, size, (unsigned long)first,
                    gs_labels[first], first_count,
                    (double)first_count / (double)size,
                    (unsigned long)second, gs_labels[second], second_count
                ) < 0) failed = 1;
        }
    }
    return gs_finish_output(file, failed);
}

static int gs_write_label_summary(
    const char *directory,
    const char *name,
    const GSBaselineClusteringMetrics *metrics
)
{
    FILE *file = NULL;
    size_t label;
    int failed = gs_open_output(directory, name, &file);
    if (failed) return 1;
    if (fputs(
            "label_index,label,total,occupied_clusters,dominant_cluster,"
            "dominant_count,dominant_fraction,second_cluster,second_count\n",
            file
        ) == EOF) failed = 1;
    for (label = 0; !failed && label < GS_LABEL_COUNT; label++) {
        GSLabelStatistics s;
        gs_label_statistics(metrics, label, &s);
        if (s.total == 0u) {
            if (fprintf(file, "%lu,%s,0,0,-1,0,0,-1,0\n",
                        (unsigned long)label, gs_labels[label]) < 0) failed = 1;
        } else if (fprintf(
                       file, "%lu,%s,%lu,%lu,%lu,%" PRIu64
                       ",%.17g,%lu,%" PRIu64 "\n",
                       (unsigned long)label, gs_labels[label],
                       (unsigned long)s.total, (unsigned long)s.occupied,
                       (unsigned long)s.first, s.first_count, s.first_fraction,
                       (unsigned long)s.second, s.second_count
                   ) < 0) failed = 1;
    }
    return gs_finish_output(file, failed);
}

static int gs_write_metrics(
    const char *directory,
    size_t k,
    const GSBaselineClusteringMetrics *train,
    const GSBaselineClusteringMetrics *validation
)
{
    FILE *file = NULL;
    int failed = gs_open_output(directory, "metrics.csv", &file);
    if (failed) return 1;
    if (fputs(
            "split,rows,k,purity,nmi_arithmetic,ari,homogeneity,completeness\n",
            file
        ) == EOF ||
        fprintf(file, "train,800,%lu,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                (unsigned long)k, train->purity,
                train->normalized_mutual_information,
                train->adjusted_rand_index, train->homogeneity,
                train->completeness) < 0 ||
        fprintf(file,
                "validation,200,%lu,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                (unsigned long)k, validation->purity,
                validation->normalized_mutual_information,
                validation->adjusted_rand_index, validation->homogeneity,
                validation->completeness) < 0) failed = 1;
    return gs_finish_output(file, failed);
}

static size_t gs_largest_cluster(const GSBaselineClusteringMetrics *metrics)
{
    size_t best = 0u, cluster;
    for (cluster = 1u; cluster < metrics->cluster_count; cluster++) {
        if (metrics->cluster_totals[cluster] >
            metrics->cluster_totals[best]) best = cluster;
    }
    return best;
}

static void gs_purity_extremes(
    const GSBaselineClusteringMetrics *metrics,
    size_t *minimum_cluster,
    double *minimum_purity,
    size_t *maximum_cluster,
    double *maximum_purity
)
{
    size_t cluster;
    int initialized = 0;
    *minimum_cluster = 0u; *maximum_cluster = 0u;
    *minimum_purity = 0.0; *maximum_purity = 0.0;
    for (cluster = 0; cluster < metrics->cluster_count; cluster++) {
        uint64_t size = metrics->cluster_totals[cluster];
        size_t first, second;
        uint64_t first_count, second_count;
        double purity;
        if (size == 0u) continue;
        gs_top_two_labels(metrics, cluster, &first, &first_count,
                          &second, &second_count);
        purity = (double)first_count / (double)size;
        if (!initialized || purity < *minimum_purity) {
            *minimum_cluster = cluster; *minimum_purity = purity;
        }
        if (!initialized || purity > *maximum_purity) {
            *maximum_cluster = cluster; *maximum_purity = purity;
        }
        initialized = 1;
    }
}

static int gs_write_requested_label(
    FILE *file,
    const char *name,
    size_t label,
    const GSBaselineClusteringMetrics *train,
    const GSBaselineClusteringMetrics *validation
)
{
    GSLabelStatistics a, b;
    gs_label_statistics(train, label, &a);
    gs_label_statistics(validation, label, &b);
    return fprintf(
        file,
        "%s_train_total=%lu\n%s_train_dominant_cluster=%lu\n"
        "%s_train_dominant_fraction=%.17g\n"
        "%s_train_occupied_clusters=%lu\n"
        "%s_validation_total=%lu\n%s_validation_dominant_cluster=%lu\n"
        "%s_validation_dominant_fraction=%.17g\n"
        "%s_validation_occupied_clusters=%lu\n",
        name, (unsigned long)a.total, name, (unsigned long)a.first,
        name, a.first_fraction, name, (unsigned long)a.occupied,
        name, (unsigned long)b.total, name, (unsigned long)b.first,
        name, b.first_fraction, name, (unsigned long)b.occupied
    ) < 0 ? 1 : 0;
}

static int gs_write_summary(
    const char *directory,
    size_t k,
    const char *code_commit,
    const char *input_commit,
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
    if (failed) return 1;
    largest = gs_largest_cluster(train);
    largest_size = train->cluster_totals[largest];
    gs_purity_extremes(train, &minimum, &minimum_purity,
                       &maximum, &maximum_purity);
    if (fprintf(
            file,
            "format_version=1\n"
            "profile=mini224_subset1000_pool14_no_pca_label_evaluation\n"
            "input_profile=mini224_subset1000_pool14_no_pca\n"
            "subset_profile=subset224_seed123_prefix_800_200\n"
            "kmeans_profile=mini224_subset1000_pool14_no_pca_kmeans\n"
            "evaluation=labels_after_clustering\n"
            "labels_policy=posterior_evaluation_only\n"
            "label_count=22\nclass_count=22\nevent_count=8583\n"
            "subset_rows=1000\ncluster_count=%lu\nk=%lu\nseed=123\n"
            "train_rows=800\nvalidation_rows=200\n"
            "labels_used_for_feature_extraction=false\n"
            "labels_used_for_standardization=false\n"
            "labels_used_for_pca=false\nlabels_used_for_kmeans=false\n"
            "labels_used_for_evaluation=true\npca=false\naugmentation=false\n"
            "nmi_normalization=arithmetic\n"
            "purity_definition=weighted_cluster_majority\n"
            "base_code_commit=%s\nkmeans_code_commit=%s\ncode_commit=%s\n"
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
            (unsigned long)k, (unsigned long)k, GS_BASE_CODE_COMMIT,
            input_commit, code_commit,
            train->purity, train->normalized_mutual_information,
            train->adjusted_rand_index, train->homogeneity,
            train->completeness, validation->purity,
            validation->normalized_mutual_information,
            validation->adjusted_rand_index, validation->homogeneity,
            validation->completeness, (unsigned long)largest, largest_size,
            (unsigned long)minimum, minimum_purity,
            (unsigned long)maximum, maximum_purity
        ) < 0) failed = 1;
    for (rank = 0; !failed && rank < 5u; rank++) {
        size_t label, best = SIZE_MAX;
        uint64_t count = 0u;
        for (label = 0; label < GS_LABEL_COUNT; label++) {
            uint64_t current =
                train->contingency[label * train->cluster_count + largest];
            if (!selected[label] && (best == SIZE_MAX || current > count)) {
                best = label; count = current;
            }
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
                largest_size == 0u ? 0.0 : (double)count / (double)largest_size
            ) < 0) failed = 1;
    }
    if (!failed &&
        (gs_write_requested_label(file, "Blip", 3u, train, validation) ||
         gs_write_requested_label(file, "Koi_Fish", 7u, train, validation) ||
         gs_write_requested_label(file, "Scattered_Light", 16u,
                                  train, validation) ||
         gs_write_requested_label(file, "None_of_the_Above", 12u,
                                  train, validation))) failed = 1;
    return gs_finish_output(file, failed);
}

static int gs_write_nine(
    const char *directory,
    size_t k,
    const char *code_commit,
    const char *input_commit,
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
        gs_write_summary(directory, k, code_commit, input_commit,
                         train, validation);
}

static int gs_build_evaluation_entries(
    const char *manifest_path,
    const char *base,
    const char *kmeans,
    const char *output,
    GSChecksumEntry entries[14],
    char paths[13][GS_PATH_SIZE]
)
{
    static const char *const input_names[5] = {
        "manifest_events_272.csv", "train_rows.csv", "val_rows.csv",
        "train_assignments_u32.bin", "val_assignments_u32.bin"
    };
    size_t i;
    if (strcmp(gs_basename(manifest_path), input_names[0]) != 0) return 1;
    entries[0].kind = "INPUT"; entries[0].name = input_names[0];
    entries[0].path = manifest_path;
    if (gs_join(base, input_names[1], paths[0], GS_PATH_SIZE) != 0 ||
        gs_join(base, input_names[2], paths[1], GS_PATH_SIZE) != 0 ||
        gs_join(kmeans, input_names[3], paths[2], GS_PATH_SIZE) != 0 ||
        gs_join(kmeans, input_names[4], paths[3], GS_PATH_SIZE) != 0) return 1;
    for (i = 1; i < 5u; i++) {
        entries[i].kind = "INPUT"; entries[i].name = input_names[i];
        entries[i].path = paths[i - 1u];
    }
    for (i = 0; i < 9u; i++) {
        if (gs_join(output, gs_historical_artifacts[i], paths[i + 4u],
                    GS_PATH_SIZE) != 0) return 1;
        entries[i + 5u].kind = "ARTIFACT";
        entries[i + 5u].name = gs_historical_artifacts[i];
        entries[i + 5u].path = paths[i + 4u];
    }
    return 0;
}

static int gs_write_sha256(
    const char *manifest_path,
    const char *base,
    const char *kmeans,
    const char *output
)
{
    GSChecksumEntry entries[14];
    char paths[13][GS_PATH_SIZE];
    FILE *file = NULL;
    size_t i;
    int failed;
    if (gs_build_evaluation_entries(
            manifest_path, base, kmeans, output, entries, paths
        ) != 0 ||
        gs_open_output(output, "sha256.txt", &file) != 0) return 1;
    failed = 0;
    for (i = 0; !failed && i < 14u; i++) {
        char hash[65];
        if (gs_sha_file(entries[i].path, hash) != 0 ||
            fprintf(file, "%s %s %s\n", entries[i].kind,
                    hash, entries[i].name) < 0) failed = 1;
    }
    return gs_finish_output(file, failed);
}

static int gs_validate_output(
    const char *manifest_path,
    const char *base,
    const char *kmeans,
    const char *output
)
{
    GSChecksumEntry entries[14];
    char paths[13][GS_PATH_SIZE];
    char sha_path[GS_PATH_SIZE];
    size_t i, size;
    if (gs_exact_files(output, gs_output_files, 10u) != 0) return 1;
    for (i = 0; i < 10u; i++) {
        char path[GS_PATH_SIZE];
        if (gs_join(output, gs_output_files[i], path, sizeof(path)) != 0 ||
            gs_file_size(path, &size) != 0 || size == 0u) return 1;
    }
    if (gs_build_evaluation_entries(
            manifest_path, base, kmeans, output, entries, paths
        ) != 0 ||
        gs_join(output, "sha256.txt", sha_path, sizeof(sha_path)) != 0 ||
        gs_verify_checksum_manifest(sha_path, entries, 14u) != 0) return 1;
    return 0;
}

static int gs_output_paths(
    const char *partial,
    const char *base_directory,
    size_t k,
    char *final,
    size_t capacity
)
{
    char parent[GS_PATH_SIZE];
    char actual_parent[GS_PATH_SIZE];
    char expected_parent[GS_PATH_SIZE];
    char base[GS_PATH_SIZE];
    char expected_partial_name[32];
    char expected_final_name[32];
    int written;
    size_t length, suffix = strlen(GS_PARTIAL_SUFFIX), final_length;
    size_t longest_name = strlen("train_contingency_counts.csv");
    if (!gs_value_valid(partial)) return 1;
    written = snprintf(expected_partial_name, sizeof(expected_partial_name),
                       "k%lu.partial", (unsigned long)k);
    if (written < 0 || (size_t)written >= sizeof(expected_partial_name) ||
        strcmp(gs_basename(partial), expected_partial_name) != 0 ||
        gs_parent_path(partial, parent, sizeof(parent)) != 0 ||
        strcmp(gs_basename(parent), "label_evaluation_seed123") != 0 ||
        gs_canonical_existing(parent, actual_parent, sizeof(actual_parent)) != 0 ||
        gs_canonical_existing(base_directory, base, sizeof(base)) != 0 ||
        gs_join(base, "label_evaluation_seed123", expected_parent,
                sizeof(expected_parent)) != 0 ||
        strcmp(actual_parent, expected_parent) != 0) {
        return 1;
    }
    length = strlen(partial);
    if (length <= suffix ||
        strcmp(partial + length - suffix, GS_PARTIAL_SUFFIX) != 0 ||
        length > SIZE_MAX - longest_name - 1u ||
        length + longest_name + 1u >= capacity) {
        return 1;
    }
    final_length = length - suffix;
    if (final_length == 0u || final_length >= capacity) return 1;
    memcpy(final, partial, final_length);
    final[final_length] = '\0';
    written = snprintf(expected_final_name, sizeof(expected_final_name),
                       "k%lu", (unsigned long)k);
    if (written < 0 || (size_t)written >= sizeof(expected_final_name) ||
        strcmp(gs_basename(final), expected_final_name) != 0) return 1;
    return gs_path_state(partial, NULL, NULL) == 0 &&
           gs_path_state(final, NULL, NULL) == 0 ? 0 : 1;
}

static int gs_failure_hook(const char *partial)
{
    const char *value = getenv(GS_FAILURE_ENV);
    return partial != NULL && value != NULL && strcmp(value, "1") == 0 &&
           (strcmp(gs_basename(partial), "k22.partial") == 0 ||
            strcmp(gs_basename(partial), "k36.partial") == 0);
}

static int gs_cleanup_partial(const char *partial)
{
    size_t i;
    int failed = 0;
    for (i = 0; i < 10u; i++) {
        char path[GS_PATH_SIZE];
        int state;
        if (gs_join(partial, gs_output_files[i], path, sizeof(path)) != 0) {
            failed = 1;
            continue;
        }
        state = gs_path_state(path, NULL, NULL);
        if (state < 0 || (state == 1 && remove(path) != 0)) failed = 1;
    }
    if (GS_RMDIR(partial) != 0) failed = 1;
    return failed;
}

int main(int argc, char **argv)
{
    GSEventManifest manifest;
    GSKMeansMetadata kmeans_metadata;
    GSBaselineClusteringMetrics train_metrics;
    GSBaselineClusteringMetrics validation_metrics;
    unsigned char *coverage = NULL;
    uint32_t *train_labels = NULL, *validation_labels = NULL;
    uint32_t *train_assignments = NULL, *validation_assignments = NULL;
    size_t train_bytes, validation_bytes, k, i;
    char train_rows_path[GS_PATH_SIZE], validation_rows_path[GS_PATH_SIZE];
    char train_assignments_path[GS_PATH_SIZE];
    char validation_assignments_path[GS_PATH_SIZE];
    char final_path[GS_PATH_SIZE];
    char train_hash_before[65], validation_hash_before[65];
    char train_hash_after[65], validation_hash_after[65];
    int partial_created = 0;
    int rc = 1;

    memset(&manifest, 0, sizeof(manifest));
    memset(&kmeans_metadata, 0, sizeof(kmeans_metadata));
    memset(&train_metrics, 0, sizeof(train_metrics));
    memset(&validation_metrics, 0, sizeof(validation_metrics));

    if (argc != 7 || !gs_platform_valid() ||
        setlocale(LC_NUMERIC, "C") == NULL ||
        gs_parse_size(argv[4], &k) != 0 || (k != 22u && k != 36u) ||
        !gs_value_valid(argv[6]) ||
        gs_output_paths(argv[5], argv[2], k, final_path,
                        sizeof(final_path)) != 0 ||
        gs_checked_product(GS_TRAIN_ROWS, sizeof(uint32_t), &train_bytes) ||
        gs_checked_product(GS_VALIDATION_ROWS, sizeof(uint32_t),
                           &validation_bytes)) {
        fprintf(stderr, "Errore: argomenti o piattaforma non validi.\n");
        return 1;
    }
    if (gs_validate_base(argv[1], argv[2]) != 0 ||
        gs_validate_kmeans(argv[2], argv[3], k, &kmeans_metadata) != 0 ||
        gs_join(argv[2], "train_rows.csv", train_rows_path,
                sizeof(train_rows_path)) != 0 ||
        gs_join(argv[2], "val_rows.csv", validation_rows_path,
                sizeof(validation_rows_path)) != 0 ||
        gs_join(argv[3], "train_assignments_u32.bin", train_assignments_path,
                sizeof(train_assignments_path)) != 0 ||
        gs_join(argv[3], "val_assignments_u32.bin",
                validation_assignments_path,
                sizeof(validation_assignments_path)) != 0) {
        fprintf(stderr, "Errore: directory base o k-means non valida.\n");
        goto cleanup;
    }
    printf("[PASS] base_and_kmeans_contract\n");

    coverage = (unsigned char *)calloc(GS_EVENT_COUNT, 1u);
    train_labels = (uint32_t *)malloc(train_bytes);
    validation_labels = (uint32_t *)malloc(validation_bytes);
    train_assignments = (uint32_t *)malloc(train_bytes);
    validation_assignments = (uint32_t *)malloc(validation_bytes);
    if (coverage == NULL || train_labels == NULL || validation_labels == NULL ||
        train_assignments == NULL || validation_assignments == NULL ||
        gs_read_assignments(train_assignments_path, train_assignments,
                            GS_TRAIN_ROWS, k) != 0 ||
        gs_read_assignments(validation_assignments_path,
                            validation_assignments,
                            GS_VALIDATION_ROWS, k) != 0 ||
        gs_sha_file(train_assignments_path, train_hash_before) != 0 ||
        gs_sha_file(validation_assignments_path,
                    validation_hash_before) != 0) {
        fprintf(stderr, "Errore: assignment non validi.\n");
        goto cleanup;
    }
    printf("[PASS] frozen_assignment_inputs\n");

    if (gs_event_manifest_load(argv[1], &manifest) != 0 ||
        gs_validate_manifest(&manifest) != 0 ||
        gs_load_rows(train_rows_path, "train", GS_TRAIN_ROWS, &manifest,
                     coverage, train_labels) != 0 ||
        gs_load_rows(validation_rows_path, "validation", GS_VALIDATION_ROWS,
                     &manifest, coverage, validation_labels) != 0) {
        fprintf(stderr, "Errore: manifest o row mapping non valido.\n");
        goto cleanup;
    }
    {
        size_t covered = 0u;
        for (i = 0; i < GS_EVENT_COUNT; i++) {
            if (coverage[i]) covered++;
        }
        if (covered != GS_TRAIN_ROWS + GS_VALIDATION_ROWS) {
            fprintf(stderr, "Errore: copertura subset non valida.\n");
            goto cleanup;
        }
    }
    if (gs_sha_file(train_assignments_path, train_hash_after) != 0 ||
        gs_sha_file(validation_assignments_path, validation_hash_after) != 0 ||
        strcmp(train_hash_before, train_hash_after) != 0 ||
        strcmp(validation_hash_before, validation_hash_after) != 0) {
        fprintf(stderr, "Errore: assignment modificati durante le label.\n");
        goto cleanup;
    }
    printf("[PASS] manifest_mapping_labels_and_assignments\n");

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

    if (GS_MKDIR(argv[5]) != 0) {
        fprintf(stderr, "Errore: creazione partial fallita.\n");
        goto cleanup;
    }
    partial_created = 1;
    if (gs_failure_hook(argv[5])) {
        fprintf(stderr, "Errore: failure hook di test attivato.\n");
        goto cleanup;
    }
    if (gs_write_nine(argv[5], k, argv[6], kmeans_metadata.code_commit,
                      &train_metrics, &validation_metrics) != 0 ||
        gs_write_sha256(argv[1], argv[2], argv[3], argv[5]) != 0 ||
        gs_validate_output(argv[1], argv[2], argv[3], argv[5]) != 0 ||
        gs_sha_file(train_assignments_path, train_hash_after) != 0 ||
        gs_sha_file(validation_assignments_path, validation_hash_after) != 0 ||
        strcmp(train_hash_before, train_hash_after) != 0 ||
        strcmp(validation_hash_before, validation_hash_after) != 0) {
        fprintf(stderr, "Errore: scrittura o validazione output fallita.\n");
        goto cleanup;
    }
    printf("[PASS] exact_inventory_and_checksums\n");
    if (rename(argv[5], final_path) != 0) {
        fprintf(stderr, "Errore: promozione atomica fallita.\n");
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
    printf("VALIDATION_ARI=%.17g\n", validation_metrics.adjusted_rand_index);
    printf("VALIDATION_HOMOGENEITY=%.17g\n",
           validation_metrics.homogeneity);
    printf("VALIDATION_COMPLETENESS=%.17g\n",
           validation_metrics.completeness);
    printf("MINI224 SUBSET1000 LABEL EVALUATION COMPLETED\n");
    rc = 0;

cleanup:
    if (partial_created && gs_cleanup_partial(argv[5]) != 0) {
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
