#include "baseline_clustering_metrics.h"
#include "vae_event_dataset.h"

#include <dirent.h>
#include <errno.h>
#include <float.h>
#include <limits.h>
#include <locale.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>


enum {
    GS_TEST_LABEL_COUNT = 22,
    GS_TEST_TRAIN_ROWS = 6866,
    GS_TEST_VALIDATION_ROWS = 1717,
    GS_TEST_MANIFEST_ROWS = 8583,
    GS_TEST_PATH_LEN = 4096,
    GS_TEST_LINE_LEN = 8192,
    GS_TEST_MAX_FIELDS = 64,
    GS_TEST_SHA_BLOCK = 64,
    GS_TEST_SHA_TEXT = 65
};

#define GS_TEST_PARTIAL_SUFFIX ".partial"


static const char *const gs_test_labels[GS_TEST_LABEL_COUNT] = {
    "1080Lines",
    "1400Ripples",
    "Air_Compressor",
    "Blip",
    "Chirp",
    "Extremely_Loud",
    "Helix",
    "Koi_Fish",
    "Light_Modulation",
    "Low_Frequency_Burst",
    "Low_Frequency_Lines",
    "No_Glitch",
    "None_of_the_Above",
    "Paired_Doves",
    "Power_Line",
    "Repeating_Blips",
    "Scattered_Light",
    "Scratchy",
    "Tomte",
    "Violin_Mode",
    "Wandering_Line",
    "Whistle"
};


typedef struct {
    size_t total;
    size_t occupied;
    size_t dominant;
    uint64_t dominant_count;
    double dominant_fraction;
    size_t second;
    uint64_t second_count;
} GSTestLabelStatistics;
typedef struct {
    uint32_t state[8];
    uint64_t bit_count;
    unsigned char block[GS_TEST_SHA_BLOCK];
    size_t used;
} GSTestSha256;

typedef struct {
    const char *kind;
    const char *name;
    const char *path;
} GSTestChecksumEntry;

typedef struct {
    size_t k;
    char code_commit[256];
} GSTestKMeansMetadata;

static const char *const gs_test_base_files[10] = {
    "feature_spec.txt", "train_features_raw_f32.bin",
    "val_features_raw_f32.bin", "train_features_std_f32.bin",
    "val_features_std_f32.bin", "scaler.csv", "train_rows.csv",
    "val_rows.csv", "summary.txt", "sha256.txt"
};

static const size_t gs_test_base_sizes[10] = {
    958u, 28123136u, 7032832u, 28123136u, 7032832u,
    54237u, 270713u, 74659u, 1411u, 1241u
};

static const char *const gs_test_kmeans_files[13] = {
    "centroids_f64.bin", "train_assignments_u32.bin",
    "val_assignments_u32.bin", "train_squared_distances_f64.bin",
    "val_squared_distances_f64.bin", "cluster_sizes_train_u64.bin",
    "initial_centroid_indices_u64.bin", "train_assignments.csv",
    "val_assignments.csv", "cluster_sizes.csv", "initial_centroids.csv",
    "summary.txt", "sha256.txt"
};

static const char *const gs_test_output_files[10] = {
    "label_map.csv", "metrics.csv", "summary.txt",
    "train_cluster_summary.csv", "train_contingency_counts.csv",
    "train_label_summary.csv", "val_cluster_summary.csv",
    "val_contingency_counts.csv", "val_label_summary.csv", "sha256.txt"
};

static const char *const gs_test_historical_artifacts[9] = {
    "label_map.csv", "train_contingency_counts.csv",
    "val_contingency_counts.csv", "train_cluster_summary.csv",
    "val_cluster_summary.csv", "train_label_summary.csv",
    "val_label_summary.csv", "metrics.csv", "summary.txt"
};
static uint32_t gs_test_rotr32(uint32_t x, unsigned int n)
{
    return (x >> n) | (x << (32u - n));
}

static void gs_test_sha_transform(GSTestSha256 *ctx, const unsigned char block[64])
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
        uint32_t s0 = gs_test_rotr32(w[i - 15u], 7) ^
                      gs_test_rotr32(w[i - 15u], 18) ^ (w[i - 15u] >> 3);
        uint32_t s1 = gs_test_rotr32(w[i - 2u], 17) ^
                      gs_test_rotr32(w[i - 2u], 19) ^ (w[i - 2u] >> 10);
        w[i] = w[i - 16u] + s0 + w[i - 7u] + s1;
    }
    a = ctx->state[0]; b = ctx->state[1]; c = ctx->state[2];
    d = ctx->state[3]; e = ctx->state[4]; f = ctx->state[5];
    g = ctx->state[6]; h = ctx->state[7];
    for (i = 0; i < 64; i++) {
        uint32_t s1 = gs_test_rotr32(e, 6) ^ gs_test_rotr32(e, 11) ^
                      gs_test_rotr32(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t t1 = h + s1 + ch + k[i] + w[i];
        uint32_t s0 = gs_test_rotr32(a, 2) ^ gs_test_rotr32(a, 13) ^
                      gs_test_rotr32(a, 22);
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

static void gs_test_sha_init(GSTestSha256 *ctx)
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

static int gs_test_sha_update(GSTestSha256 *ctx, const unsigned char *data, size_t size)
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
            gs_test_sha_transform(ctx, ctx->block);
            ctx->used = 0u;
        }
    }
    return 0;
}

static void gs_test_sha_final(GSTestSha256 *ctx, unsigned char digest[32])
{
    uint64_t bits = ctx->bit_count;
    size_t i;
    ctx->block[ctx->used++] = 0x80u;
    if (ctx->used > 56u) {
        memset(ctx->block + ctx->used, 0, 64u - ctx->used);
        gs_test_sha_transform(ctx, ctx->block);
        ctx->used = 0u;
    }
    memset(ctx->block + ctx->used, 0, 56u - ctx->used);
    for (i = 0; i < 8u; i++) {
        ctx->block[63u - i] = (unsigned char)(bits & UINT64_C(0xff));
        bits >>= 8;
    }
    gs_test_sha_transform(ctx, ctx->block);
    for (i = 0; i < 8u; i++) {
        digest[i * 4u] = (unsigned char)(ctx->state[i] >> 24);
        digest[i * 4u + 1u] = (unsigned char)(ctx->state[i] >> 16);
        digest[i * 4u + 2u] = (unsigned char)(ctx->state[i] >> 8);
        digest[i * 4u + 3u] = (unsigned char)ctx->state[i];
    }
}

static int gs_test_sha_file(const char *path, char text[GS_TEST_SHA_TEXT])
{
    static const char hex[] = "0123456789abcdef";
    unsigned char buffer[32768];
    unsigned char digest[32];
    GSTestSha256 ctx;
    FILE *file;
    size_t got;
    size_t i;
    int failed = 0;

    if (path == NULL || text == NULL || (file = fopen(path, "rb")) == NULL) {
        return 1;
    }
    gs_test_sha_init(&ctx);
    while ((got = fread(buffer, 1u, sizeof(buffer), file)) != 0u) {
        if (gs_test_sha_update(&ctx, buffer, got) != 0) {
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
    gs_test_sha_final(&ctx, digest);
    for (i = 0; i < 32u; i++) {
        text[i * 2u] = hex[digest[i] >> 4];
        text[i * 2u + 1u] = hex[digest[i] & 15u];
    }
    text[64] = '\0';
    return 0;
}


static int gs_test_checked_product(
    size_t left,
    size_t right,
    size_t *product
)
{
    if (product == NULL || (right != 0 && left > SIZE_MAX / right)) {
        return 1;
    }
    *product = left * right;
    return 0;
}


static int gs_test_platform_is_valid(void)
{
    uint16_t value = UINT16_C(1);
    const unsigned char *bytes = (const unsigned char *)&value;

    return CHAR_BIT == 8 && sizeof(uint32_t) == 4 &&
           sizeof(uint64_t) == 8 && sizeof(double) == 8 &&
           bytes[0] == 1u && FLT_RADIX == 2 && DBL_MANT_DIG == 53 &&
           DBL_MAX_EXP == 1024;
}


static int gs_test_label_index(const char *label)
{
    size_t index;

    for (index = 0; index < GS_TEST_LABEL_COUNT; index++) {
        if (strcmp(label, gs_test_labels[index]) == 0) {
            return (int)index;
        }
    }
    return -1;
}


static void gs_test_strip_line_ending(char *line)
{
    size_t length = strlen(line);

    while (length != 0 &&
           (line[length - 1] == '\r' || line[length - 1] == '\n')) {
        line[--length] = '\0';
    }
}


static int gs_test_split_csv(
    char *line,
    char **fields,
    size_t expected_fields
)
{
    size_t count = 0;
    char *cursor;

    if (line == NULL || fields == NULL || expected_fields == 0 ||
        expected_fields > GS_TEST_MAX_FIELDS) {
        return 1;
    }
    gs_test_strip_line_ending(line);
    fields[count++] = line;
    for (cursor = line; *cursor != '\0'; cursor++) {
        if (*cursor == ',') {
            if (count >= expected_fields) {
                return 1;
            }
            *cursor = '\0';
            fields[count++] = cursor + 1;
        }
    }
    return count == expected_fields ? 0 : 1;
}


static int gs_test_parse_size(const char *text, size_t *value)
{
    char *endptr = NULL;
    unsigned long long parsed;

    if (text == NULL || value == NULL || text[0] == '\0' ||
        text[0] == '-') {
        return 1;
    }
    errno = 0;
    parsed = strtoull(text, &endptr, 10);
    if (errno == ERANGE || endptr == text || *endptr != '\0' ||
        parsed > (unsigned long long)SIZE_MAX) {
        return 1;
    }
    *value = (size_t)parsed;
    return 0;
}


static int gs_test_parse_u64(const char *text, uint64_t *value)
{
    char *endptr = NULL;
    unsigned long long parsed;

    if (text == NULL || value == NULL || text[0] == '\0' ||
        text[0] == '-') {
        return 1;
    }
    errno = 0;
    parsed = strtoull(text, &endptr, 10);
    if (errno == ERANGE || endptr == text || *endptr != '\0') {
        return 1;
    }
    *value = (uint64_t)parsed;
    return 0;
}


static int gs_test_parse_long(const char *text, long *value)
{
    char *endptr = NULL;
    long parsed;

    errno = 0;
    parsed = strtol(text, &endptr, 10);
    if (errno == ERANGE || endptr == text || *endptr != '\0') {
        return 1;
    }
    *value = parsed;
    return 0;
}


static int gs_test_parse_double(const char *text, double *value)
{
    char *endptr = NULL;
    double parsed;

    errno = 0;
    parsed = strtod(text, &endptr);
    if (errno == ERANGE || endptr == text || *endptr != '\0' ||
        !isfinite(parsed)) {
        return 1;
    }
    *value = parsed;
    return 0;
}


static int gs_test_close(double left, double right)
{
    return isfinite(left) && isfinite(right) &&
           fabs(left - right) <= 1.0e-12 * (1.0 + fabs(right));
}

static int gs_test_metrics_consistent(
    const GSBaselineClusteringMetrics *metrics,
    size_t rows,
    size_t k
)
{
    uint64_t total = UINT64_C(0);
    size_t cells, label, cluster;
    if (metrics == NULL || metrics->rows != rows ||
        metrics->label_count != GS_TEST_LABEL_COUNT ||
        metrics->cluster_count != k || metrics->contingency == NULL ||
        metrics->label_totals == NULL || metrics->cluster_totals == NULL ||
        gs_test_checked_product(GS_TEST_LABEL_COUNT, k, &cells) != 0 ||
        cells == 0u || !isfinite(metrics->purity) ||
        !isfinite(metrics->normalized_mutual_information) ||
        !isfinite(metrics->adjusted_rand_index) ||
        !isfinite(metrics->homogeneity) ||
        !isfinite(metrics->completeness)) {
        return 0;
    }
    for (label = 0; label < GS_TEST_LABEL_COUNT; label++) {
        uint64_t row_total = UINT64_C(0);
        for (cluster = 0; cluster < k; cluster++) {
            uint64_t value = metrics->contingency[label * k + cluster];
            if (row_total > UINT64_MAX - value) return 0;
            row_total += value;
        }
        if (row_total != metrics->label_totals[label] ||
            total > UINT64_MAX - row_total) {
            return 0;
        }
        total += row_total;
    }
    if (total != (uint64_t)rows) return 0;
    for (cluster = 0; cluster < k; cluster++) {
        uint64_t column_total = UINT64_C(0);
        for (label = 0; label < GS_TEST_LABEL_COUNT; label++) {
            uint64_t value = metrics->contingency[label * k + cluster];
            if (column_total > UINT64_MAX - value) return 0;
            column_total += value;
        }
        if (column_total != metrics->cluster_totals[cluster]) return 0;
    }
    return 1;
}

static int gs_test_value_valid(const char *value)
{
    return value != NULL && value[0] != '\0' &&
           strchr(value, '\r') == NULL && strchr(value, '\n') == NULL;
}


static int gs_test_path_join(
    const char *directory,
    const char *filename,
    char *path,
    size_t path_size
)
{
    size_t directory_length;
    size_t filename_length;
    size_t required;
    int needs_separator;

    if (directory == NULL || filename == NULL || path == NULL ||
        path_size == 0) {
        return 1;
    }
    directory_length = strlen(directory);
    filename_length = strlen(filename);
    needs_separator = directory_length != 0 &&
        directory[directory_length - 1] != '/' &&
        directory[directory_length - 1] != '\\';
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
    memcpy(path, directory, directory_length);
    if (needs_separator) {
        path[directory_length++] = '/';
    }
    memcpy(path + directory_length, filename, filename_length);
    path[directory_length + filename_length] = '\0';
    return 0;
}


static int gs_test_file_size(const char *path, size_t *size)
{
    FILE *file;
    long position;
    int failed = 0;

    file = fopen(path, "rb");
    if (file == NULL) {
        return 1;
    }
    if (fseek(file, 0L, SEEK_END) != 0) {
        failed = 1;
    }
    position = failed ? -1L : ftell(file);
    if (position < 0 || fclose(file) != 0) {
        failed = 1;
    }
    if (failed) {
        return 1;
    }
    *size = (size_t)position;
    return 0;
}


static int gs_test_path_state(const char *path, int *is_regular, int *is_directory)
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
static const char *gs_test_basename(const char *path)
{
    const char *base = path;
    const char *cursor;
    if (path == NULL) return NULL;
    for (cursor = path; *cursor != '\0'; cursor++) {
        if (*cursor == '/' || *cursor == '\\') base = cursor + 1;
    }
    return base;
}

static int gs_test_canonical_existing(const char *path, char *output, size_t capacity)
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

static int gs_test_validate_kmeans_location(
    const char *base_directory,
    const char *kmeans_directory,
    size_t k
)
{
    char base[GS_TEST_PATH_LEN];
    char actual[GS_TEST_PATH_LEN];
    char expected[GS_TEST_PATH_LEN];
    char suffix[64];
    int written;
    if (gs_test_canonical_existing(base_directory, base, sizeof(base)) != 0 ||
        gs_test_canonical_existing(kmeans_directory, actual, sizeof(actual)) != 0 ||
        (written = snprintf(suffix, sizeof(suffix),
                            "kmeans_seed123/k%lu", (unsigned long)k)) < 0 ||
        (size_t)written >= sizeof(suffix) ||
        gs_test_path_join(base, suffix, expected, sizeof(expected)) != 0) {
        return 1;
    }
    return strcmp(actual, expected) == 0 ? 0 : 1;
}

static int gs_test_name_index(
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

static int gs_test_exact_files(
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
        char path[GS_TEST_PATH_LEN];
        int regular, is_directory;
        errno = 0;
        entry = readdir(stream);
        if (entry == NULL) {
            if (errno != 0) failed = 1;
            break;
        }
        if (strcmp(entry->d_name, ".") == 0 ||
            strcmp(entry->d_name, "..") == 0) continue;
        if (!gs_test_name_index(entry->d_name, names, count, &index) ||
            seen[index] ||
            gs_test_path_join(directory, entry->d_name, path, sizeof(path)) != 0 ||
            gs_test_path_state(path, &regular, &is_directory) != 1 ||
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

static int gs_test_base_inventory(const char *directory)
{
    unsigned char seen[10] = {0};
    DIR *stream = opendir(directory);
    struct dirent *entry;
    int kmeans_seen = 0;
    int failed = stream == NULL;
    size_t i;
    while (!failed) {
        size_t index;
        char path[GS_TEST_PATH_LEN];
        int regular, is_directory;
        errno = 0;
        entry = readdir(stream);
        if (entry == NULL) {
            if (errno != 0) failed = 1;
            break;
        }
        if (strcmp(entry->d_name, ".") == 0 ||
            strcmp(entry->d_name, "..") == 0) continue;
        if (gs_test_path_join(directory, entry->d_name, path, sizeof(path)) != 0 ||
            gs_test_path_state(path, &regular, &is_directory) != 1) {
            failed = 1;
            break;
        }
        if (gs_test_name_index(entry->d_name, gs_test_base_files, 10u, &index)) {
            if (seen[index] || !regular) failed = 1;
            seen[index] = 1u;
        } else if (strcmp(entry->d_name, "kmeans_seed123") == 0 &&
                   !kmeans_seen && is_directory) {
            kmeans_seen = 1;
        } else {
            failed = 1;
        }
    }
    if (stream != NULL && closedir(stream) != 0) failed = 1;
    for (i = 0; !failed && i < 10u; i++) if (!seen[i]) failed = 1;
    return failed;
}

static int gs_test_verify_checksum_manifest(
    const char *sha_path,
    const GSTestChecksumEntry *entries,
    size_t count
)
{
    FILE *file;
    char line[GS_TEST_LINE_LEN];
    size_t index = 0u;
    if (sha_path == NULL || entries == NULL ||
        (file = fopen(sha_path, "rb")) == NULL) {
        return 1;
    }
    while (index < count && fgets(line, sizeof(line), file) != NULL) {
        char kind[16], hash[65], name[GS_TEST_PATH_LEN], extra;
        char actual[65];
        int fields = sscanf(line, "%15s %64s %4095s %c",
                            kind, hash, name, &extra);
        if (fields != 3 || strcmp(kind, entries[index].kind) != 0 ||
            strcmp(name, entries[index].name) != 0 ||
            strlen(hash) != 64u ||
            gs_test_sha_file(entries[index].path, actual) != 0 ||
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

static int gs_test_build_base_entries(
    const char *manifest_path,
    const char *base,
    GSTestChecksumEntry entries[13],
    char paths[12][GS_TEST_PATH_LEN]
)
{
    static const char *const input_names[4] = {
        "manifest_events_272.csv", "manifest_tensors_112_central.csv",
        "split_train_indices_seed123.csv", "split_val_indices_seed123.csv"
    };
    static const char *const artifact_names[9] = {
        "feature_spec.txt", "train_features_raw_f32.bin",
        "val_features_raw_f32.bin", "train_features_std_f32.bin",
        "val_features_std_f32.bin", "scaler.csv", "train_rows.csv",
        "val_rows.csv", "summary.txt"
    };
    size_t i;
    if (strcmp(gs_test_basename(manifest_path), input_names[0]) != 0) return 1;
    entries[0].kind = "INPUT"; entries[0].name = input_names[0];
    entries[0].path = manifest_path;
    for (i = 1; i < 4u; i++) {
        if (strlen(input_names[i]) >= GS_TEST_PATH_LEN) return 1;
        strcpy(paths[i - 1u], input_names[i]);
        entries[i].kind = "INPUT"; entries[i].name = input_names[i];
        entries[i].path = paths[i - 1u];
    }
    for (i = 0; i < 9u; i++) {
        if (gs_test_path_join(base, artifact_names[i], paths[i + 3u],
                    GS_TEST_PATH_LEN) != 0) return 1;
        entries[i + 4u].kind = "ARTIFACT";
        entries[i + 4u].name = artifact_names[i];
        entries[i + 4u].path = paths[i + 3u];
    }
    return 0;
}

static int gs_test_validate_base(const char *manifest_path, const char *base)
{
    GSTestChecksumEntry entries[13];
    char paths[12][GS_TEST_PATH_LEN];
    char sha_path[GS_TEST_PATH_LEN];
    size_t i, actual;
    if (gs_test_base_inventory(base) != 0) return 1;
    for (i = 0; i < 10u; i++) {
        char path[GS_TEST_PATH_LEN];
        if (gs_test_path_join(base, gs_test_base_files[i], path, sizeof(path)) != 0 ||
            gs_test_file_size(path, &actual) != 0 || actual != gs_test_base_sizes[i]) {
            return 1;
        }
    }
    if (gs_test_build_base_entries(manifest_path, base, entries, paths) != 0 ||
        gs_test_path_join(base, "sha256.txt", sha_path, sizeof(sha_path)) != 0 ||
        gs_test_verify_checksum_manifest(sha_path, entries, 13u) != 0) {
        return 1;
    }
    return 0;
}

static int gs_test_read_summary_value(
    const char *path,
    const char *key,
    char *value,
    size_t capacity
)
{
    FILE *file = fopen(path, "rb");
    char line[GS_TEST_LINE_LEN];
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

static int gs_test_validate_kmeans_summary(
    const char *directory,
    size_t k,
    GSTestKMeansMetadata *metadata
)
{
    static const char *const keys[] = {
        "profile", "input_profile", "train_rows", "validation_rows",
        "feature_count", "k", "seed", "max_iterations", "converged",
        "fit_input", "validation_assigned_after_fit", "pca",
        "augmentation", "labels_used", "code_commit"
    };
    static const char *const fixed[] = {
        "mini112_pool7_no_pca_kmeans", "mini112_pool7_no_pca", "6866",
        "1717", "1024", NULL, "123", "300", "1", "train_only",
        "true", "false", "false", "false", NULL
    };
    char path[GS_TEST_PATH_LEN];
    char value[256];
    char k_text[32];
    size_t i;
    int written;
    if (metadata == NULL ||
        gs_test_path_join(directory, "summary.txt", path, sizeof(path)) != 0 ||
        (written = snprintf(k_text, sizeof(k_text), "%lu",
                            (unsigned long)k)) < 0 ||
        (size_t)written >= sizeof(k_text)) {
        return 1;
    }
    for (i = 0; i < sizeof(keys) / sizeof(keys[0]); i++) {
        if (gs_test_read_summary_value(path, keys[i], value, sizeof(value)) != 0) {
            return 1;
        }
        if (i == 5u) {
            if (strcmp(value, k_text) != 0) return 1;
        } else if (i == 14u) {
            if (!gs_test_value_valid(value) ||
                strlen(value) >= sizeof(metadata->code_commit)) return 1;
            strcpy(metadata->code_commit, value);
        } else if (strcmp(value, fixed[i]) != 0) {
            return 1;
        }
    }
    metadata->k = k;
    return 0;
}

static int gs_test_build_kmeans_entries(
    const char *base,
    const char *directory,
    GSTestChecksumEntry entries[18],
    char paths[18][GS_TEST_PATH_LEN]
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
        if (gs_test_path_join(base, input_names[i], paths[i], GS_TEST_PATH_LEN) != 0) return 1;
        entries[i].kind = "INPUT"; entries[i].name = input_names[i];
        entries[i].path = paths[i];
    }
    for (i = 0; i < 12u; i++) {
        if (gs_test_path_join(directory, artifact_names[i], paths[i + 6u],
                    GS_TEST_PATH_LEN) != 0) return 1;
        entries[i + 6u].kind = "ARTIFACT";
        entries[i + 6u].name = artifact_names[i];
        entries[i + 6u].path = paths[i + 6u];
    }
    return 0;
}

static int gs_test_validate_kmeans(
    const char *base,
    const char *directory,
    size_t k,
    GSTestKMeansMetadata *metadata
)
{
    GSTestChecksumEntry entries[18];
    char paths[18][GS_TEST_PATH_LEN];
    char sha_path[GS_TEST_PATH_LEN];
    size_t sizes[7] = {0u, 27464u, 6868u, 54928u, 13736u, 0u, 0u};
    size_t actual, k_bytes, centroid_values, centroid_bytes, i;
    char expected_base[16];
    int written;
    if (k != 22u && k != 36u) return 1;
    if ((written = snprintf(expected_base, sizeof(expected_base), "k%lu",
                            (unsigned long)k)) < 0 ||
        (size_t)written >= sizeof(expected_base) ||
        strcmp(gs_test_basename(directory), expected_base) != 0 ||
        strstr(gs_test_basename(directory), GS_TEST_PARTIAL_SUFFIX) != NULL ||
        gs_test_validate_kmeans_location(base, directory, k) != 0 ||
        gs_test_exact_files(directory, gs_test_kmeans_files, 13u) != 0 ||
        gs_test_checked_product(k, 1024u, &centroid_values) != 0 ||
        gs_test_checked_product(centroid_values, sizeof(double),
                           &centroid_bytes) != 0 ||
        gs_test_checked_product(k, sizeof(uint64_t), &k_bytes) != 0) {
        return 1;
    }
    sizes[0] = centroid_bytes; sizes[5] = k_bytes; sizes[6] = k_bytes;
    for (i = 0; i < 13u; i++) {
        char path[GS_TEST_PATH_LEN];
        if (gs_test_path_join(directory, gs_test_kmeans_files[i], path, sizeof(path)) != 0 ||
            gs_test_file_size(path, &actual) != 0 ||
            (i < 7u ? actual != sizes[i] : actual == 0u)) {
            return 1;
        }
    }
    if (gs_test_build_kmeans_entries(base, directory, entries, paths) != 0 ||
        gs_test_path_join(directory, "sha256.txt", sha_path, sizeof(sha_path)) != 0 ||
        gs_test_verify_checksum_manifest(sha_path, entries, 18u) != 0 ||
        gs_test_validate_kmeans_summary(directory, k, metadata) != 0) {
        return 1;
    }
    return 0;
}

static int gs_test_build_evaluation_entries(
    const char *manifest_path,
    const char *base,
    const char *kmeans,
    const char *output,
    GSTestChecksumEntry entries[14],
    char paths[13][GS_TEST_PATH_LEN]
)
{
    static const char *const input_names[5] = {
        "manifest_events_272.csv", "train_rows.csv", "val_rows.csv",
        "train_assignments_u32.bin", "val_assignments_u32.bin"
    };
    size_t index;
    if (strcmp(gs_test_basename(manifest_path), input_names[0]) != 0) return 1;
    entries[0].kind = "INPUT";
    entries[0].name = input_names[0];
    entries[0].path = manifest_path;
    if (gs_test_path_join(base, input_names[1], paths[0],
                          GS_TEST_PATH_LEN) != 0 ||
        gs_test_path_join(base, input_names[2], paths[1],
                          GS_TEST_PATH_LEN) != 0 ||
        gs_test_path_join(kmeans, input_names[3], paths[2],
                          GS_TEST_PATH_LEN) != 0 ||
        gs_test_path_join(kmeans, input_names[4], paths[3],
                          GS_TEST_PATH_LEN) != 0) {
        return 1;
    }
    for (index = 1; index < 5u; index++) {
        entries[index].kind = "INPUT";
        entries[index].name = input_names[index];
        entries[index].path = paths[index - 1u];
    }
    for (index = 0; index < 9u; index++) {
        if (gs_test_path_join(output, gs_test_historical_artifacts[index],
                              paths[index + 4u], GS_TEST_PATH_LEN) != 0) {
            return 1;
        }
        entries[index + 5u].kind = "ARTIFACT";
        entries[index + 5u].name = gs_test_historical_artifacts[index];
        entries[index + 5u].path = paths[index + 4u];
    }
    return 0;
}

static int gs_test_expected_files_and_checksums(
    const char *manifest,
    const char *base,
    const char *kmeans,
    const char *directory
)
{
    GSTestChecksumEntry entries[14];
    char paths[13][GS_TEST_PATH_LEN];
    char sha_path[GS_TEST_PATH_LEN];
    char partial_path[GS_TEST_PATH_LEN];
    size_t index;
    int regular, is_directory;
    if (gs_test_exact_files(directory, gs_test_output_files, 10u) != 0 ||
        gs_test_build_evaluation_entries(
            manifest, base, kmeans, directory, entries, paths
        ) != 0 ||
        gs_test_path_join(directory, "sha256.txt", sha_path,
                          sizeof(sha_path)) != 0 ||
        gs_test_verify_checksum_manifest(sha_path, entries, 14u) != 0) {
        return 1;
    }
    for (index = 0; index < 10u; index++) {
        size_t size;
        char path[GS_TEST_PATH_LEN];
        if (gs_test_path_join(directory, gs_test_output_files[index],
                              path, sizeof(path)) != 0 ||
            gs_test_file_size(path, &size) != 0 || size == 0u) {
            return 1;
        }
    }
    if (strlen(directory) > SIZE_MAX - strlen(GS_TEST_PARTIAL_SUFFIX) ||
        strlen(directory) + strlen(GS_TEST_PARTIAL_SUFFIX) >=
            sizeof(partial_path)) {
        return 1;
    }
    strcpy(partial_path, directory);
    strcat(partial_path, GS_TEST_PARTIAL_SUFFIX);
    if (gs_test_path_state(partial_path, &regular, &is_directory) != 0) {
        return 1;
    }
    return 0;
}


static int gs_test_compare_strings(const void *left, const void *right)
{
    const char *const *a = (const char *const *)left;
    const char *const *b = (const char *const *)right;
    return strcmp(*a, *b);
}

static int gs_test_validate_manifest(const GSEventManifest *manifest)
{
    unsigned char present[GS_TEST_LABEL_COUNT] = {0};
    const char **ids = NULL;
    size_t index;

    if (manifest == NULL || manifest->count != GS_TEST_MANIFEST_ROWS ||
        manifest->records == NULL) {
        return 1;
    }
    if (manifest->count > SIZE_MAX / sizeof(*ids) ||
        (ids = (const char **)malloc(manifest->count * sizeof(*ids))) == NULL) {
        return 1;
    }
    for (index = 0; index < manifest->count; index++) {
        const GSEventRecord *record = &manifest->records[index];
        int label = gs_test_label_index(record->label);
        size_t view;
        if (label < 0 || record->event_index != (int)(index + 1u) ||
            record->event_id[0] == '\0') {
            free(ids);
            return 1;
        }
        for (view = 0; view < GS_EVENT_VIEW_COUNT; view++) {
            if (record->image_paths[view][0] == '\0') {
                free(ids);
                return 1;
            }
        }
        ids[index] = record->event_id;
        present[(size_t)label] = 1;
    }
    qsort(ids, manifest->count, sizeof(*ids), gs_test_compare_strings);
    for (index = 1; index < manifest->count; index++) {
        if (strcmp(ids[index - 1u], ids[index]) == 0) {
            free(ids);
            return 1;
        }
    }
    free(ids);
    for (index = 0; index < GS_TEST_LABEL_COUNT; index++) {
        if (!present[index]) {
            return 1;
        }
    }
    return 0;
}


static int gs_test_validate_label_map(const char *directory)
{
    char path[GS_TEST_PATH_LEN];
    char line[GS_TEST_LINE_LEN];
    FILE *file;
    size_t label;
    int failed = 0;

    if (gs_test_path_join(
            directory,
            "label_map.csv",
            path,
            sizeof(path)
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
        gs_test_strip_line_ending(line);
        if (strcmp(line, "label_index,label") != 0) {
            failed = 1;
        }
    }
    for (label = 0; !failed && label < GS_TEST_LABEL_COUNT; label++) {
        char *fields[2];
        size_t parsed_label;
        if (fgets(line, sizeof(line), file) == NULL ||
            gs_test_split_csv(line, fields, 2) != 0 ||
            gs_test_parse_size(fields[0], &parsed_label) != 0 ||
            parsed_label != label ||
            strcmp(fields[1], gs_test_labels[label]) != 0) {
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


static int gs_test_load_rows(
    const char *path,
    const char *split,
    size_t rows,
    const GSEventManifest *manifest,
    unsigned char *coverage,
    uint32_t *labels
)
{
    static const char expected_header[] =
        "row_index,split_pos,manifest_index,event_index,event_id,split";
    FILE *file = fopen(path, "rb");
    char line[GS_TEST_LINE_LEN];
    size_t row;
    int failed = 0;

    if (file == NULL) {
        return 1;
    }
    if (fgets(line, sizeof(line), file) == NULL) {
        failed = 1;
    } else {
        gs_test_strip_line_ending(line);
        if (strcmp(line, expected_header) != 0) {
            failed = 1;
        }
    }
    for (row = 0; !failed && row < rows; row++) {
        char *fields[6];
        size_t row_index;
        size_t split_pos;
        size_t manifest_index;
        size_t event_index;
        const GSEventRecord *record;
        int label;

        if (fgets(line, sizeof(line), file) == NULL ||
            gs_test_split_csv(line, fields, 6) != 0 ||
            gs_test_parse_size(fields[0], &row_index) != 0 ||
            gs_test_parse_size(fields[1], &split_pos) != 0 ||
            gs_test_parse_size(fields[2], &manifest_index) != 0 ||
            gs_test_parse_size(fields[3], &event_index) != 0 ||
            row_index != row || split_pos != row ||
            manifest_index >= manifest->count || coverage[manifest_index] ||
            strcmp(fields[5], split) != 0) {
            failed = 1;
            break;
        }
        record = gs_event_manifest_get(manifest, manifest_index);
        label = record == NULL ? -1 : gs_test_label_index(record->label);
        if (record == NULL || label < 0 ||
            event_index != (size_t)record->event_index ||
            strcmp(fields[4], record->event_id) != 0) {
            failed = 1;
            break;
        }
        coverage[manifest_index] = 1;
        labels[row] = (uint32_t)label;
    }
    if (!failed && fgets(line, sizeof(line), file) != NULL) {
        failed = 1;
    }
    if (ferror(file) || fclose(file) != 0) {
        failed = 1;
    }
    return failed;
}


static int gs_test_read_assignments(
    const char *path,
    uint32_t *assignments,
    size_t rows,
    size_t k
)
{
    FILE *file;
    size_t expected_bytes;
    size_t actual_bytes;
    size_t row;
    int failed = 0;

    if (gs_test_checked_product(
            rows,
            sizeof(uint32_t),
            &expected_bytes
        ) != 0 ||
        gs_test_file_size(path, &actual_bytes) != 0 ||
        actual_bytes != expected_bytes) {
        return 1;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        return 1;
    }
    if (fread(assignments, sizeof(uint32_t), rows, file) != rows ||
        fgetc(file) != EOF || ferror(file)) {
        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    for (row = 0; !failed && row < rows; row++) {
        if ((size_t)assignments[row] >= k) {
            failed = 1;
        }
    }
    return failed;
}


static int gs_test_validate_contingency(
    const char *directory,
    const char *filename,
    const GSBaselineClusteringMetrics *metrics
)
{
    char path[GS_TEST_PATH_LEN];
    char line[GS_TEST_LINE_LEN];
    char *fields[GS_TEST_MAX_FIELDS];
    FILE *file;
    size_t label;
    size_t cluster;
    size_t expected_fields = metrics->cluster_count + 3;
    int failed = 0;

    if (expected_fields > GS_TEST_MAX_FIELDS ||
        gs_test_path_join(directory, filename, path, sizeof(path)) != 0) {
        return 1;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        return 1;
    }
    if (fgets(line, sizeof(line), file) == NULL ||
        gs_test_split_csv(line, fields, expected_fields) != 0 ||
        strcmp(fields[0], "label_index") != 0 ||
        strcmp(fields[1], "label") != 0 ||
        strcmp(fields[expected_fields - 1], "label_total") != 0) {
        failed = 1;
    }
    for (cluster = 0; !failed && cluster < metrics->cluster_count; cluster++) {
        char expected[64];
        if (snprintf(
                expected,
                sizeof(expected),
                "cluster_%lu",
                (unsigned long)cluster
            ) < 0 ||
            strcmp(fields[cluster + 2], expected) != 0) {
            failed = 1;
        }
    }

    for (label = 0; !failed && label < GS_TEST_LABEL_COUNT; label++) {
        size_t parsed_label;
        uint64_t parsed_total;

        if (fgets(line, sizeof(line), file) == NULL ||
            gs_test_split_csv(line, fields, expected_fields) != 0 ||
            gs_test_parse_size(fields[0], &parsed_label) != 0 ||
            parsed_label != label ||
            strcmp(fields[1], gs_test_labels[label]) != 0) {
            failed = 1;
            break;
        }
        for (cluster = 0; cluster < metrics->cluster_count; cluster++) {
            uint64_t parsed;
            if (gs_test_parse_u64(fields[cluster + 2], &parsed) != 0 ||
                parsed != metrics->contingency[
                    label * metrics->cluster_count + cluster
                ]) {
                failed = 1;
                break;
            }
        }
        if (!failed &&
            (gs_test_parse_u64(
                 fields[expected_fields - 1],
                 &parsed_total
             ) != 0 ||
             parsed_total != metrics->label_totals[label])) {
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


static int gs_test_validate_metric_row(
    char **fields,
    const char *split,
    size_t rows,
    size_t k,
    const GSBaselineClusteringMetrics *metrics
)
{
    size_t parsed_rows;
    size_t parsed_k;
    double values[5];
    size_t index;

    if (strcmp(fields[0], split) != 0 ||
        gs_test_parse_size(fields[1], &parsed_rows) != 0 ||
        gs_test_parse_size(fields[2], &parsed_k) != 0 ||
        parsed_rows != rows || parsed_k != k) {
        return 1;
    }
    for (index = 0; index < 5; index++) {
        if (gs_test_parse_double(fields[index + 3], &values[index]) != 0) {
            return 1;
        }
    }
    return !gs_test_close(values[0], metrics->purity) ||
        !gs_test_close(values[1], metrics->normalized_mutual_information) ||
        !gs_test_close(values[2], metrics->adjusted_rand_index) ||
        !gs_test_close(values[3], metrics->homogeneity) ||
        !gs_test_close(values[4], metrics->completeness);
}


static int gs_test_validate_metrics_csv(
    const char *directory,
    size_t k,
    const GSBaselineClusteringMetrics *train,
    const GSBaselineClusteringMetrics *validation,
    int *train_passed,
    int *validation_passed
)
{
    static const char header[] =
        "split,rows,k,purity,nmi_arithmetic,ari,homogeneity,completeness";
    char path[GS_TEST_PATH_LEN];
    char line[GS_TEST_LINE_LEN];
    char *fields[8];
    FILE *file;
    int failed = 0;

    *train_passed = 0;
    *validation_passed = 0;
    if (gs_test_path_join(
            directory,
            "metrics.csv",
            path,
            sizeof(path)
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
        gs_test_strip_line_ending(line);
        if (strcmp(line, header) != 0) {
            failed = 1;
        }
    }
    if (!failed &&
        (fgets(line, sizeof(line), file) == NULL ||
         gs_test_split_csv(line, fields, 8) != 0 ||
         gs_test_validate_metric_row(
             fields,
             "train",
             GS_TEST_TRAIN_ROWS,
             k,
             train
         ) != 0)) {
        failed = 1;
    } else if (!failed) {
        *train_passed = 1;
    }
    if (!failed &&
        (fgets(line, sizeof(line), file) == NULL ||
         gs_test_split_csv(line, fields, 8) != 0 ||
         gs_test_validate_metric_row(
             fields,
             "validation",
             GS_TEST_VALIDATION_ROWS,
             k,
             validation
         ) != 0)) {
        failed = 1;
    } else if (!failed) {
        *validation_passed = 1;
    }
    if (!failed && fgets(line, sizeof(line), file) != NULL) {
        failed = 1;
    }
    if (ferror(file) || fclose(file) != 0) {
        failed = 1;
    }
    return failed;
}


static void gs_test_top_two_labels(
    const GSBaselineClusteringMetrics *metrics,
    size_t cluster,
    size_t *first,
    uint64_t *first_count,
    size_t *second,
    uint64_t *second_count
)
{
    size_t label;

    *first = SIZE_MAX;
    *second = SIZE_MAX;
    *first_count = UINT64_C(0);
    *second_count = UINT64_C(0);
    for (label = 0; label < GS_TEST_LABEL_COUNT; label++) {
        uint64_t count =
            metrics->contingency[label * metrics->cluster_count + cluster];
        if (*first == SIZE_MAX || count > *first_count) {
            *second = *first;
            *second_count = *first_count;
            *first = label;
            *first_count = count;
        } else if (*second == SIZE_MAX || count > *second_count) {
            *second = label;
            *second_count = count;
        }
    }
}


static int gs_test_validate_cluster_summary(
    const char *directory,
    const char *filename,
    const GSBaselineClusteringMetrics *metrics
)
{
    static const char header[] =
        "cluster,size,dominant_label_index,dominant_label,dominant_count,"
        "cluster_purity,second_label_index,second_label,second_count";
    char path[GS_TEST_PATH_LEN];
    char line[GS_TEST_LINE_LEN];
    FILE *file;
    size_t cluster;
    int failed = 0;

    if (gs_test_path_join(directory, filename, path, sizeof(path)) != 0) {
        return 1;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        return 1;
    }
    if (fgets(line, sizeof(line), file) == NULL) {
        failed = 1;
    } else {
        gs_test_strip_line_ending(line);
        if (strcmp(line, header) != 0) {
            failed = 1;
        }
    }
    for (cluster = 0; !failed && cluster < metrics->cluster_count; cluster++) {
        char *fields[9];
        size_t parsed_cluster;
        uint64_t parsed_size;
        long parsed_first;
        uint64_t parsed_first_count;
        double parsed_purity;
        long parsed_second;
        uint64_t parsed_second_count;
        uint64_t size = metrics->cluster_totals[cluster];

        if (fgets(line, sizeof(line), file) == NULL ||
            gs_test_split_csv(line, fields, 9) != 0 ||
            gs_test_parse_size(fields[0], &parsed_cluster) != 0 ||
            gs_test_parse_u64(fields[1], &parsed_size) != 0 ||
            gs_test_parse_long(fields[2], &parsed_first) != 0 ||
            gs_test_parse_u64(fields[4], &parsed_first_count) != 0 ||
            gs_test_parse_double(fields[5], &parsed_purity) != 0 ||
            gs_test_parse_long(fields[6], &parsed_second) != 0 ||
            gs_test_parse_u64(fields[8], &parsed_second_count) != 0 ||
            parsed_cluster != cluster || parsed_size != size) {
            failed = 1;
            break;
        }

        if (size == 0) {
            if (parsed_first != -1 || strcmp(fields[3], "NA") != 0 ||
                parsed_first_count != 0 || !gs_test_close(parsed_purity, 0.0) ||
                parsed_second != -1 || strcmp(fields[7], "NA") != 0 ||
                parsed_second_count != 0) {
                failed = 1;
            }
        } else {
            size_t first;
            size_t second;
            uint64_t first_count;
            uint64_t second_count;
            double purity;

            gs_test_top_two_labels(
                metrics,
                cluster,
                &first,
                &first_count,
                &second,
                &second_count
            );
            purity = (double)first_count / (double)size;
            if (parsed_first != (long)first ||
                strcmp(fields[3], gs_test_labels[first]) != 0 ||
                parsed_first_count != first_count ||
                !gs_test_close(parsed_purity, purity) ||
                parsed_second != (long)second ||
                strcmp(fields[7], gs_test_labels[second]) != 0 ||
                parsed_second_count != second_count) {
                failed = 1;
            }
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


static void gs_test_label_statistics(
    const GSBaselineClusteringMetrics *metrics,
    size_t label,
    GSTestLabelStatistics *statistics
)
{
    size_t cluster;

    statistics->total = (size_t)metrics->label_totals[label];
    statistics->occupied = 0;
    statistics->dominant = SIZE_MAX;
    statistics->dominant_count = UINT64_C(0);
    statistics->second = SIZE_MAX;
    statistics->second_count = UINT64_C(0);
    for (cluster = 0; cluster < metrics->cluster_count; cluster++) {
        uint64_t count =
            metrics->contingency[label * metrics->cluster_count + cluster];
        if (count != 0) {
            statistics->occupied++;
        }
        if (statistics->dominant == SIZE_MAX ||
            count > statistics->dominant_count) {
            statistics->second = statistics->dominant;
            statistics->second_count = statistics->dominant_count;
            statistics->dominant = cluster;
            statistics->dominant_count = count;
        } else if (statistics->second == SIZE_MAX ||
                   count > statistics->second_count) {
            statistics->second = cluster;
            statistics->second_count = count;
        }
    }
    statistics->dominant_fraction = statistics->total == 0
        ? 0.0
        : (double)statistics->dominant_count / (double)statistics->total;
}


static int gs_test_validate_label_summary(
    const char *directory,
    const char *filename,
    const GSBaselineClusteringMetrics *metrics
)
{
    static const char header[] =
        "label_index,label,total,occupied_clusters,dominant_cluster,"
        "dominant_count,dominant_fraction,second_cluster,second_count";
    char path[GS_TEST_PATH_LEN];
    char line[GS_TEST_LINE_LEN];
    FILE *file;
    size_t label;
    int failed = 0;

    if (gs_test_path_join(directory, filename, path, sizeof(path)) != 0) {
        return 1;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        return 1;
    }
    if (fgets(line, sizeof(line), file) == NULL) {
        failed = 1;
    } else {
        gs_test_strip_line_ending(line);
        if (strcmp(line, header) != 0) {
            failed = 1;
        }
    }
    for (label = 0; !failed && label < GS_TEST_LABEL_COUNT; label++) {
        char *fields[9];
        size_t parsed_label;
        uint64_t parsed_total;
        size_t parsed_occupied;
        long parsed_dominant;
        uint64_t parsed_dominant_count;
        double parsed_fraction;
        long parsed_second;
        uint64_t parsed_second_count;
        GSTestLabelStatistics statistics;

        if (fgets(line, sizeof(line), file) == NULL ||
            gs_test_split_csv(line, fields, 9) != 0 ||
            gs_test_parse_size(fields[0], &parsed_label) != 0 ||
            gs_test_parse_u64(fields[2], &parsed_total) != 0 ||
            gs_test_parse_size(fields[3], &parsed_occupied) != 0 ||
            gs_test_parse_long(fields[4], &parsed_dominant) != 0 ||
            gs_test_parse_u64(fields[5], &parsed_dominant_count) != 0 ||
            gs_test_parse_double(fields[6], &parsed_fraction) != 0 ||
            gs_test_parse_long(fields[7], &parsed_second) != 0 ||
            gs_test_parse_u64(fields[8], &parsed_second_count) != 0 ||
            parsed_label != label ||
            strcmp(fields[1], gs_test_labels[label]) != 0) {
            failed = 1;
            break;
        }

        gs_test_label_statistics(metrics, label, &statistics);
        if (parsed_total != (uint64_t)statistics.total ||
            parsed_occupied != statistics.occupied ||
            parsed_dominant_count != statistics.dominant_count ||
            parsed_second_count != statistics.second_count ||
            !gs_test_close(parsed_fraction, statistics.dominant_fraction) ||
            (statistics.total == 0
                ? parsed_dominant != -1 || parsed_second != -1
                : parsed_dominant != (long)statistics.dominant ||
                  parsed_second != (long)statistics.second)) {
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


static int gs_test_metadata_contract(
    const char *directory,
    size_t k,
    const char *commit,
    const char *input_commit,
    const GSBaselineClusteringMetrics *train,
    const GSBaselineClusteringMetrics *validation
)
{
    static const char *const fixed_keys[] = {
        "format_version", "profile", "evaluation_profile", "kmeans_profile",
        "evaluation", "label_count", "event_count", "seed", "train_rows",
        "validation_rows", "labels_used_for_training",
        "labels_used_for_clustering", "labels_used_for_evaluation", "pca",
        "nmi_normalization", "purity_definition"
    };
    static const char *const fixed_values[] = {
        "1", "mini112_pool7_no_pca",
        "mini112_pool7_no_pca_label_evaluation",
        "mini112_pool7_no_pca_kmeans", "labels_after_clustering", "22",
        "8583", "123", "6866", "1717", "false", "false", "true",
        "false", "arithmetic", "weighted_cluster_majority"
    };
    static const char *const metric_keys[10] = {
        "train_purity", "train_nmi_arithmetic", "train_ari",
        "train_homogeneity", "train_completeness", "validation_purity",
        "validation_nmi_arithmetic", "validation_ari",
        "validation_homogeneity", "validation_completeness"
    };
    double expected[10];
    char path[GS_TEST_PATH_LEN];
    char value[512];
    size_t index;
    size_t parsed_k;
    if (directory == NULL || commit == NULL || input_commit == NULL ||
        train == NULL || validation == NULL ||
        gs_test_path_join(directory, "summary.txt", path, sizeof(path)) != 0) {
        return 1;
    }
    for (index = 0; index < sizeof(fixed_keys) / sizeof(fixed_keys[0]);
         index++) {
        if (gs_test_read_summary_value(
                path, fixed_keys[index], value, sizeof(value)
            ) != 0 ||
            strcmp(value, fixed_values[index]) != 0) {
            return 1;
        }
    }
    if (gs_test_read_summary_value(path, "k", value, sizeof(value)) != 0 ||
        gs_test_parse_size(value, &parsed_k) != 0 || parsed_k != k ||
        gs_test_read_summary_value(
            path, "code_commit", value, sizeof(value)
        ) != 0 || strcmp(value, commit) != 0 ||
        gs_test_read_summary_value(
            path, "input_kmeans_code_commit", value, sizeof(value)
        ) != 0 || strcmp(value, input_commit) != 0) {
        return 1;
    }
    expected[0] = train->purity;
    expected[1] = train->normalized_mutual_information;
    expected[2] = train->adjusted_rand_index;
    expected[3] = train->homogeneity;
    expected[4] = train->completeness;
    expected[5] = validation->purity;
    expected[6] = validation->normalized_mutual_information;
    expected[7] = validation->adjusted_rand_index;
    expected[8] = validation->homogeneity;
    expected[9] = validation->completeness;
    for (index = 0; index < 10u; index++) {
        double parsed;
        if (gs_test_read_summary_value(
                path, metric_keys[index], value, sizeof(value)
            ) != 0 ||
            gs_test_parse_double(value, &parsed) != 0 ||
            !gs_test_close(parsed, expected[index])) {
            return 1;
        }
    }
    for (index = 1; index <= 5u; index++) {
        char key[128];
        if (snprintf(
                key,
                sizeof(key),
                "largest_train_cluster_top_%lu_label",
                (unsigned long)index
            ) < 0 ||
            gs_test_read_summary_value(path, key, value, sizeof(value)) != 0 ||
            gs_test_label_index(value) < 0) {
            return 1;
        }
    }
    return 0;
}

static int gs_test_snapshot_hashes(
    const char *directory,
    const char *const *names,
    size_t count,
    char (*hashes)[GS_TEST_SHA_TEXT]
)
{
    size_t index;
    for (index = 0; index < count; index++) {
        char path[GS_TEST_PATH_LEN];
        if (gs_test_path_join(directory, names[index], path, sizeof(path)) != 0 ||
            gs_test_sha_file(path, hashes[index]) != 0) {
            return 1;
        }
    }
    return 0;
}

static int gs_test_hashes_equal(
    char (*before)[GS_TEST_SHA_TEXT],
    char (*after)[GS_TEST_SHA_TEXT],
    size_t count
)
{
    size_t index;
    for (index = 0; index < count; index++) {
        if (strcmp(before[index], after[index]) != 0) return 0;
    }
    return 1;
}


int main(int argc, char **argv)
{
    GSEventManifest manifest;
    GSBaselineClusteringMetrics train_metrics;
    GSBaselineClusteringMetrics validation_metrics;
    GSTestKMeansMetadata kmeans_metadata;
    GSTestKMeansMetadata other_kmeans_metadata;
    unsigned char *coverage = NULL;
    uint32_t *train_labels = NULL;
    uint32_t *validation_labels = NULL;
    uint32_t *train_assignments = NULL;
    uint32_t *validation_assignments = NULL;
    uint32_t *train_copy = NULL;
    uint32_t *validation_copy = NULL;
    char train_rows_path[GS_TEST_PATH_LEN];
    char validation_rows_path[GS_TEST_PATH_LEN];
    char train_assignments_path[GS_TEST_PATH_LEN];
    char validation_assignments_path[GS_TEST_PATH_LEN];
    char k22_directory[GS_TEST_PATH_LEN];
    char k36_directory[GS_TEST_PATH_LEN];
    char base_before[10][GS_TEST_SHA_TEXT];
    char base_after[10][GS_TEST_SHA_TEXT];
    char k22_before[13][GS_TEST_SHA_TEXT];
    char k22_after[13][GS_TEST_SHA_TEXT];
    char k36_before[13][GS_TEST_SHA_TEXT];
    char k36_after[13][GS_TEST_SHA_TEXT];
    size_t train_bytes;
    size_t validation_bytes;
    size_t k;
    size_t index;
    int train_metrics_passed = 0;
    int validation_metrics_passed = 0;
    int rc = 1;

    memset(&manifest, 0, sizeof(manifest));
    memset(&train_metrics, 0, sizeof(train_metrics));
    memset(&validation_metrics, 0, sizeof(validation_metrics));
    memset(&kmeans_metadata, 0, sizeof(kmeans_metadata));
    memset(&other_kmeans_metadata, 0, sizeof(other_kmeans_metadata));

    if (argc != 7 || !gs_test_platform_is_valid() ||
        setlocale(LC_NUMERIC, "C") == NULL ||
        !gs_test_value_valid(argv[6]) ||
        gs_test_parse_size(argv[5], &k) != 0 || (k != 22u && k != 36u) ||
        gs_test_checked_product(
            GS_TEST_TRAIN_ROWS, sizeof(uint32_t), &train_bytes
        ) != 0 ||
        gs_test_checked_product(
            GS_TEST_VALIDATION_ROWS, sizeof(uint32_t), &validation_bytes
        ) != 0 ||
        gs_test_path_join(argv[2], "train_rows.csv", train_rows_path,
                          sizeof(train_rows_path)) != 0 ||
        gs_test_path_join(argv[2], "val_rows.csv", validation_rows_path,
                          sizeof(validation_rows_path)) != 0 ||
        gs_test_path_join(argv[3], "train_assignments_u32.bin",
                          train_assignments_path,
                          sizeof(train_assignments_path)) != 0 ||
        gs_test_path_join(argv[3], "val_assignments_u32.bin",
                          validation_assignments_path,
                          sizeof(validation_assignments_path)) != 0 ||
        gs_test_path_join(argv[2], "kmeans_seed123/k22", k22_directory,
                          sizeof(k22_directory)) != 0 ||
        gs_test_path_join(argv[2], "kmeans_seed123/k36", k36_directory,
                          sizeof(k36_directory)) != 0) {
        fprintf(stderr, "ERROR: invalid validator arguments or platform.\n");
        return 1;
    }

    if (gs_test_snapshot_hashes(
            argv[2], gs_test_base_files, 10u, base_before
        ) != 0 ||
        gs_test_snapshot_hashes(
            k22_directory, gs_test_kmeans_files, 13u, k22_before
        ) != 0 ||
        gs_test_snapshot_hashes(
            k36_directory, gs_test_kmeans_files, 13u, k36_before
        ) != 0 ||
        gs_test_validate_base(argv[1], argv[2]) != 0 ||
        gs_test_validate_kmeans(argv[2], argv[3], k, &kmeans_metadata) != 0 ||
        gs_test_validate_kmeans(
            argv[2], k == 22u ? k36_directory : k22_directory,
            k == 22u ? 36u : 22u, &other_kmeans_metadata
        ) != 0) {
        fprintf(stderr, "[FAIL] frozen_input_contract\n");
        goto cleanup;
    }
    printf("[PASS] frozen_input_contract\n");

    if (gs_test_expected_files_and_checksums(
            argv[1], argv[2], argv[3], argv[4]
        ) != 0) {
        fprintf(stderr, "[FAIL] evaluation_inventory_and_checksums\n");
        goto cleanup;
    }
    printf("[PASS] evaluation_inventory_and_checksums\n");

    coverage = (unsigned char *)calloc(
        GS_TEST_MANIFEST_ROWS, sizeof(unsigned char)
    );
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
        fprintf(stderr, "[FAIL] manifest_contract\n");
        goto cleanup;
    }
    printf("[PASS] manifest_contract\n");

    if (gs_test_load_rows(
            train_rows_path, "train", GS_TEST_TRAIN_ROWS, &manifest,
            coverage, train_labels
        ) != 0 ||
        gs_test_load_rows(
            validation_rows_path, "validation", GS_TEST_VALIDATION_ROWS,
            &manifest, coverage, validation_labels
        ) != 0 ||
        gs_test_read_assignments(
            train_assignments_path, train_assignments,
            GS_TEST_TRAIN_ROWS, k
        ) != 0 ||
        gs_test_read_assignments(
            validation_assignments_path, validation_assignments,
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
        fprintf(stderr, "[FAIL] metric_recomputation\n");
        goto cleanup;
    }

    if (gs_test_validate_label_map(argv[4]) != 0) {
        fprintf(stderr, "[FAIL] label_map\n");
        goto cleanup;
    }
    printf("LABEL_MAP_MATCH=true\n");

    if (gs_test_validate_contingency(
            argv[4], "train_contingency_counts.csv", &train_metrics
        ) != 0 ||
        gs_test_validate_contingency(
            argv[4], "val_contingency_counts.csv", &validation_metrics
        ) != 0) {
        fprintf(stderr, "[FAIL] recomputed_contingencies\n");
        goto cleanup;
    }
    printf("TRAIN_CONTINGENCY_RECOMPUTED_MATCH=true\n");
    printf("VALIDATION_CONTINGENCY_RECOMPUTED_MATCH=true\n");

    if (gs_test_validate_metrics_csv(
            argv[4], k, &train_metrics, &validation_metrics,
            &train_metrics_passed, &validation_metrics_passed
        ) != 0 || !train_metrics_passed || !validation_metrics_passed) {
        fprintf(stderr, "[FAIL] recomputed_metrics\n");
        goto cleanup;
    }
    printf("TRAIN_METRICS_RECOMPUTED_MATCH=true\n");
    printf("VALIDATION_METRICS_RECOMPUTED_MATCH=true\n");

    if (gs_test_validate_cluster_summary(
            argv[4], "train_cluster_summary.csv", &train_metrics
        ) != 0 ||
        gs_test_validate_cluster_summary(
            argv[4], "val_cluster_summary.csv", &validation_metrics
        ) != 0) {
        fprintf(stderr, "[FAIL] recomputed_cluster_summaries\n");
        goto cleanup;
    }
    printf("TRAIN_CLUSTER_SUMMARY_RECOMPUTED_MATCH=true\n");
    printf("VALIDATION_CLUSTER_SUMMARY_RECOMPUTED_MATCH=true\n");

    if (gs_test_validate_label_summary(
            argv[4], "train_label_summary.csv", &train_metrics
        ) != 0 ||
        gs_test_validate_label_summary(
            argv[4], "val_label_summary.csv", &validation_metrics
        ) != 0) {
        fprintf(stderr, "[FAIL] recomputed_label_summaries\n");
        goto cleanup;
    }
    printf("TRAIN_LABEL_SUMMARY_RECOMPUTED_MATCH=true\n");
    printf("VALIDATION_LABEL_SUMMARY_RECOMPUTED_MATCH=true\n");

    if (gs_test_metadata_contract(
            argv[4], k, argv[6], kmeans_metadata.code_commit,
            &train_metrics, &validation_metrics
        ) != 0) {
        fprintf(stderr, "[FAIL] deterministic_summary_contract\n");
        goto cleanup;
    }
    printf("[PASS] deterministic_summary_contract\n");

    if (memcmp(train_assignments, train_copy, train_bytes) != 0 ||
        memcmp(validation_assignments, validation_copy,
               validation_bytes) != 0 ||
        gs_test_validate_base(argv[1], argv[2]) != 0 ||
        gs_test_validate_kmeans(argv[2], argv[3], k, &kmeans_metadata) != 0 ||
        gs_test_validate_kmeans(
            argv[2], k == 22u ? k36_directory : k22_directory,
            k == 22u ? 36u : 22u, &other_kmeans_metadata
        ) != 0 ||
        gs_test_snapshot_hashes(
            argv[2], gs_test_base_files, 10u, base_after
        ) != 0 ||
        gs_test_snapshot_hashes(
            k22_directory, gs_test_kmeans_files, 13u, k22_after
        ) != 0 ||
        gs_test_snapshot_hashes(
            k36_directory, gs_test_kmeans_files, 13u, k36_after
        ) != 0 ||
        !gs_test_hashes_equal(base_before, base_after, 10u) ||
        !gs_test_hashes_equal(k22_before, k22_after, 13u) ||
        !gs_test_hashes_equal(k36_before, k36_after, 13u)) {
        fprintf(stderr, "[FAIL] frozen_inputs_bitwise_unchanged\n");
        goto cleanup;
    }

    printf("ALL_EVALUATION_CHECKSUMS_MATCH=true\n");
    printf("ASSIGNMENTS_BITWISE_UNCHANGED=true\n");
    printf("BASE_ARTIFACTS_BITWISE_UNCHANGED=true\n");
    printf("KMEANS_ARTIFACTS_BITWISE_UNCHANGED=true\n");
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
    printf("ALL MINI112 EVALUATION ARTIFACT TESTS PASSED\n");
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
