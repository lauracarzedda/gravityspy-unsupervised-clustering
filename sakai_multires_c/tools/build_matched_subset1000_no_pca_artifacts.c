#define _CRT_SECURE_NO_WARNINGS

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "baseline_standardization.h"

enum {
    GS_PATH_CAPACITY = 4096,
    GS_LINE_CAPACITY = 4096,
    GS_EVENT_ID_CAPACITY = 128,
    GS_SOURCE_TRAIN_ROWS = 6866,
    GS_SOURCE_VALIDATION_ROWS = 1717,
    GS_MATCHED_TRAIN_ROWS = 800,
    GS_MATCHED_VALIDATION_ROWS = 200,
    GS_EVENT_ROWS = 8583,
    GS_INPUT_COUNT = 9,
    GS_ARTIFACT_COUNT = 9,
    GS_TOTAL_OUTPUT_FILES = 10
};

typedef struct {
    const char *token;
    const char *source_basename;
    const char *source_profile;
    const char *matched_basename;
    const char *matched_profile;
    const char *source_commit;
    const char *source_checksum_manifest_hash;
    const char *tensor_manifest_name;
    const char *input_shape;
    const char *extractor;
    const char *pooled_shape;
    size_t feature_count;
    size_t pooled_side;
} GSProfile;

typedef struct {
    size_t split_pos;
    size_t manifest_index;
    size_t event_index;
    char event_id[GS_EVENT_ID_CAPACITY];
} GSRow;

typedef struct {
    char source[GS_PATH_CAPACITY];
    char root[GS_PATH_CAPACITY];
    char manifest[GS_PATH_CAPACITY];
    char subset_train[GS_PATH_CAPACITY];
    char subset_validation[GS_PATH_CAPACITY];
    char output_parent[GS_PATH_CAPACITY];
    char output_partial[GS_PATH_CAPACITY];
    char output_final[GS_PATH_CAPACITY];
} GSResolved;

typedef struct {
    double train_raw_min;
    double train_raw_max;
    double validation_raw_min;
    double validation_raw_max;
    double max_abs_mean;
    double max_abs_stddev_error;
    size_t constant_count;
} GSStatistics;

typedef struct {
    uint32_t state[8];
    uint64_t bit_count;
    unsigned char buffer[64];
    size_t buffer_count;
} GSSha256;

static const GSProfile gs_profiles[] = {
    {
        "mini75_pool5",
        "mini75_pool5_no_pca",
        "mini75_pool5_no_pca",
        "mini75_pool5_no_pca",
        "matched_subset1000_mini75_pool5_no_pca",
        "02f9360b71c7034c56c7c3ec5506543e35323904",
        "b855be125e882e3d2f7169374bfb8b642208567b857c62cc0a54e5e92536c84f",
        "manifest_tensors_75_central.csv",
        "4x75x75",
        "average_pool_5x5_non_overlapping",
        "4x15x15",
        900u,
        15u
    },
    {
        "mini112_pool7",
        "mini112_pool7_no_pca",
        "mini112_pool7_no_pca",
        "mini112_pool7_no_pca",
        "matched_subset1000_mini112_pool7_no_pca",
        "e1809d6d6391cfa92cc8c36718321e853e73d963",
        "c26a9e28863526976339a8690ea0b70f35f1146b502f3234d92fc48c7df6dab1",
        "manifest_tensors_112_central.csv",
        "4x112x112",
        "average_pool_7x7_non_overlapping",
        "4x16x16",
        1024u,
        16u
    }
};

static const char *const gs_source_file_names[] = {
    "feature_spec.txt",
    "scaler.csv",
    "sha256.txt",
    "summary.txt",
    "train_features_raw_f32.bin",
    "train_features_std_f32.bin",
    "train_rows.csv",
    "val_features_raw_f32.bin",
    "val_features_std_f32.bin",
    "val_rows.csv"
};

static const char *const gs_artifact_names[GS_ARTIFACT_COUNT] = {
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

static const char *const gs_input_logical_names[GS_INPUT_COUNT] = {
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

static int gs_checked_product(size_t left, size_t right, size_t *result)
{
    if (result == NULL || (left != 0u && right > SIZE_MAX / left)) {
        return 1;
    }
    *result = left * right;
    return 0;
}

static int gs_text_value_valid(const char *value)
{
    const unsigned char *cursor = (const unsigned char *)value;
    if (value == NULL || value[0] == '\0') {
        return 0;
    }
    while (*cursor != '\0') {
        if (*cursor < 32u || *cursor == 127u) {
            return 0;
        }
        cursor++;
    }
    return 1;
}

static int gs_commit_valid(const char *value)
{
    size_t index;
    if (!gs_text_value_valid(value)) {
        return 0;
    }
    if (strcmp(value, "UNCOMMITTED_STEP11B_TEST") == 0) {
        return 1;
    }
    if (strlen(value) != 40u) {
        return 0;
    }
    for (index = 0u; index < 40u; index++) {
        if (!isxdigit((unsigned char)value[index])) {
            return 0;
        }
    }
    return 1;
}

static const GSProfile *gs_profile_from_token(const char *token)
{
    size_t index;
    if (token == NULL) {
        return NULL;
    }
    for (index = 0u; index < sizeof(gs_profiles) / sizeof(gs_profiles[0]); index++) {
        if (strcmp(token, gs_profiles[index].token) == 0) {
            return &gs_profiles[index];
        }
    }
    return NULL;
}

static int gs_path_join(
    const char *directory,
    const char *name,
    char *output,
    size_t output_capacity
)
{
    size_t directory_length;
    size_t name_length;
    int separator;
    int written;
    if (directory == NULL || name == NULL || output == NULL ||
        output_capacity == 0u) {
        return 1;
    }
    directory_length = strlen(directory);
    name_length = strlen(name);
    separator = directory_length != 0u &&
        directory[directory_length - 1u] != '/' &&
        directory[directory_length - 1u] != '\\';
    if (directory_length > SIZE_MAX - name_length - (size_t)separator) {
        return 1;
    }
    written = snprintf(
        output,
        output_capacity,
        separator ? "%s/%s" : "%s%s",
        directory,
        name
    );
    return written < 0 || (size_t)written >= output_capacity;
}

static void gs_normalize_separators(char *path)
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
static int gs_has_reparse_component(const char *path)
{
    char copy[GS_PATH_CAPACITY];
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
            DWORD attributes;
            char saved = copy[index];
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

static int gs_canonical_existing(
    const char *path,
    int require_directory,
    char resolved[GS_PATH_CAPACITY]
)
{
    struct stat information;
    size_t length;
    if (!gs_text_value_valid(path) || resolved == NULL) {
        return 1;
    }
#ifdef _WIN32
    if (_fullpath(resolved, path, GS_PATH_CAPACITY) == NULL) {
        return 1;
    }
#else
    if (realpath(path, resolved) == NULL) {
        return 1;
    }
#endif
    gs_normalize_separators(resolved);
    length = strlen(resolved);
    while (length > 1u && resolved[length - 1u] == '/') {
#ifdef _WIN32
        if (length == 3u && resolved[1] == ':') {
            break;
        }
#endif
        resolved[--length] = '\0';
    }
    if (stat(resolved, &information) != 0 ||
        (require_directory && !S_ISDIR(information.st_mode)) ||
        (!require_directory && !S_ISREG(information.st_mode))) {
        return 1;
    }
#ifdef _WIN32
    if (gs_has_reparse_component(resolved) != 0) {
        return 1;
    }
#endif
    return 0;
}

static const char *gs_basename(const char *path)
{
    const char *slash;
    const char *backslash;
    const char *last;
    if (path == NULL) {
        return NULL;
    }
    slash = strrchr(path, '/');
    backslash = strrchr(path, '\\');
    last = slash;
    if (backslash != NULL && (last == NULL || backslash > last)) {
        last = backslash;
    }
    return last == NULL ? path : last + 1;
}

static int gs_parent_path(const char *path, char *parent, size_t capacity)
{
    size_t length;
    char *last;
    if (path == NULL || parent == NULL || capacity == 0u ||
        strlen(path) >= capacity) {
        return 1;
    }
    strcpy(parent, path);
    gs_normalize_separators(parent);
    length = strlen(parent);
    while (length > 0u && parent[length - 1u] == '/') {
        parent[--length] = '\0';
    }
    last = strrchr(parent, '/');
    if (last == NULL || last == parent) {
        return 1;
    }
    *last = '\0';
    return 0;
}

static int gs_path_state(const char *path)
{
    struct stat information;
    errno = 0;
    if (path != NULL && stat(path, &information) == 0) {
        return 1;
    }
    return errno == ENOENT ? 0 : -1;
}

static int gs_resolve_layout(
    const char *manifest_argument,
    const char *source_argument,
    const GSProfile *profile,
    const char *subset_train_argument,
    const char *subset_validation_argument,
    const char *output_argument,
    GSResolved *resolved
)
{
    static const char baseline_prefix[] =
        "/sakai_multires_c/results/baseline/";
    char suffix[GS_PATH_CAPACITY];
    char expected[GS_PATH_CAPACITY];
    char canonical_expected[GS_PATH_CAPACITY];
    char output_absolute[GS_PATH_CAPACITY];
    char output_parent_argument[GS_PATH_CAPACITY];
    size_t source_length;
    size_t suffix_length;
    size_t root_length;
    int written;
    if (profile == NULL || resolved == NULL ||
        gs_canonical_existing(source_argument, 1, resolved->source) != 0) {
        return 1;
    }
    written = snprintf(
        suffix,
        sizeof(suffix),
        "%s%s",
        baseline_prefix,
        profile->source_basename
    );
    if (written < 0 || (size_t)written >= sizeof(suffix)) {
        return 1;
    }
    source_length = strlen(resolved->source);
    suffix_length = strlen(suffix);
    if (source_length <= suffix_length) {
        return 1;
    }
    root_length = source_length - suffix_length;
#ifdef _WIN32
    if (_stricmp(resolved->source + root_length, suffix) != 0) {
#else
    if (strcmp(resolved->source + root_length, suffix) != 0) {
#endif
        return 1;
    }
    if (root_length == 0u || root_length >= sizeof(resolved->root)) {
        return 1;
    }
    memcpy(resolved->root, resolved->source, root_length);
    resolved->root[root_length] = '\0';
    if (gs_canonical_existing(resolved->root, 1, canonical_expected) != 0 ||
        !gs_paths_equal(resolved->root, canonical_expected) ||
        gs_path_join(
            resolved->root,
            "sakai_multires_c/results/baseline",
            expected,
            sizeof(expected)
        ) != 0 ||
        gs_canonical_existing(expected, 1, canonical_expected) != 0 ||
        gs_path_join(canonical_expected, profile->source_basename, expected,
                     sizeof(expected)) != 0 ||
        gs_canonical_existing(expected, 1, canonical_expected) != 0 ||
        !gs_paths_equal(resolved->source, canonical_expected)) {
        return 1;
    }
    if (gs_path_join(resolved->root, "manifest_events_272.csv",
                     expected, sizeof(expected)) != 0 ||
        gs_canonical_existing(expected, 0, resolved->manifest) != 0 ||
        gs_canonical_existing(manifest_argument, 0, canonical_expected) != 0 ||
        !gs_paths_equal(resolved->manifest, canonical_expected)) {
        return 1;
    }
    if (gs_path_join(
            resolved->root,
            "sakai_multires_c/configs/subset224_train_800_seed123.csv",
            expected,
            sizeof(expected)
        ) != 0 ||
        gs_canonical_existing(expected, 0, resolved->subset_train) != 0 ||
        gs_canonical_existing(subset_train_argument, 0, canonical_expected) != 0 ||
        !gs_paths_equal(resolved->subset_train, canonical_expected)) {
        return 1;
    }
    if (gs_path_join(
            resolved->root,
            "sakai_multires_c/configs/subset224_val_200_seed123.csv",
            expected,
            sizeof(expected)
        ) != 0 ||
        gs_canonical_existing(expected, 0, resolved->subset_validation) != 0 ||
        gs_canonical_existing(subset_validation_argument, 0,
                              canonical_expected) != 0 ||
        !gs_paths_equal(resolved->subset_validation, canonical_expected)) {
        return 1;
    }
#ifdef _WIN32
    if (_fullpath(output_absolute, output_argument, sizeof(output_absolute)) == NULL) {
        return 1;
    }
#else
    if (output_argument == NULL || output_argument[0] != '/' ||
        strlen(output_argument) >= sizeof(output_absolute)) {
        return 1;
    }
    strcpy(output_absolute, output_argument);
#endif
    gs_normalize_separators(output_absolute);
    if (strcmp(gs_basename(output_absolute), profile->matched_basename) == 0) {
        return 1;
    }
    written = snprintf(
        expected,
        sizeof(expected),
        "%s.partial",
        profile->matched_basename
    );
    if (written < 0 || (size_t)written >= sizeof(expected) ||
        strcmp(gs_basename(output_absolute), expected) != 0 ||
        gs_parent_path(output_absolute, output_parent_argument,
                       sizeof(output_parent_argument)) != 0 ||
        gs_canonical_existing(output_parent_argument, 1,
                              resolved->output_parent) != 0 ||
        gs_path_join(
            resolved->root,
            "sakai_multires_c/results/baseline/matched_subset1000",
            expected,
            sizeof(expected)
        ) != 0 ||
        gs_canonical_existing(expected, 1, canonical_expected) != 0 ||
        !gs_paths_equal(resolved->output_parent, canonical_expected) ||
        gs_path_join(resolved->output_parent, gs_basename(output_absolute),
                     resolved->output_partial,
                     sizeof(resolved->output_partial)) != 0 ||
        !gs_paths_equal(output_absolute, resolved->output_partial) ||
        gs_path_join(resolved->output_parent, profile->matched_basename,
                     resolved->output_final,
                     sizeof(resolved->output_final)) != 0 ||
        gs_path_state(resolved->output_partial) != 0 ||
        gs_path_state(resolved->output_final) != 0) {
        return 1;
    }
    return 0;
}

static uint32_t gs_rotr(uint32_t value, unsigned int count)
{
    return (value >> count) | (value << (32u - count));
}

static void gs_sha_transform(GSSha256 *context, const unsigned char block[64])
{
    static const uint32_t constants[64] = {
        0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,
        0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
        0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,
        0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
        0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,
        0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
        0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,
        0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
        0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,
        0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
        0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,
        0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
        0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,
        0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
        0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,
        0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
    };
    uint32_t words[64];
    uint32_t a,b,c,d,e,f,g,h;
    size_t index;
    for (index = 0u; index < 16u; index++) {
        size_t offset = index * 4u;
        words[index] =
            ((uint32_t)block[offset] << 24) |
            ((uint32_t)block[offset + 1u] << 16) |
            ((uint32_t)block[offset + 2u] << 8) |
            (uint32_t)block[offset + 3u];
    }
    for (index = 16u; index < 64u; index++) {
        uint32_t s0 = gs_rotr(words[index - 15u], 7u) ^
            gs_rotr(words[index - 15u], 18u) ^ (words[index - 15u] >> 3);
        uint32_t s1 = gs_rotr(words[index - 2u], 17u) ^
            gs_rotr(words[index - 2u], 19u) ^ (words[index - 2u] >> 10);
        words[index] = words[index - 16u] + s0 +
            words[index - 7u] + s1;
    }
    a=context->state[0]; b=context->state[1]; c=context->state[2];
    d=context->state[3]; e=context->state[4]; f=context->state[5];
    g=context->state[6]; h=context->state[7];
    for (index = 0u; index < 64u; index++) {
        uint32_t s1=gs_rotr(e,6u)^gs_rotr(e,11u)^gs_rotr(e,25u);
        uint32_t choice=(e&f)^((~e)&g);
        uint32_t temp1=h+s1+choice+constants[index]+words[index];
        uint32_t s0=gs_rotr(a,2u)^gs_rotr(a,13u)^gs_rotr(a,22u);
        uint32_t majority=(a&b)^(a&c)^(b&c);
        uint32_t temp2=s0+majority;
        h=g; g=f; f=e; e=d+temp1; d=c; c=b; b=a; a=temp1+temp2;
    }
    context->state[0]+=a; context->state[1]+=b;
    context->state[2]+=c; context->state[3]+=d;
    context->state[4]+=e; context->state[5]+=f;
    context->state[6]+=g; context->state[7]+=h;
}

static void gs_sha_init(GSSha256 *context)
{
    static const uint32_t initial[8] = {
        0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
        0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u
    };
    memcpy(context->state, initial, sizeof(initial));
    context->bit_count = 0u;
    context->buffer_count = 0u;
}

static int gs_sha_update(GSSha256 *context, const void *data, size_t length)
{
    const unsigned char *bytes = (const unsigned char *)data;
    size_t index = 0u;
    if (context == NULL || (data == NULL && length != 0u) ||
        length > (UINT64_MAX - context->bit_count) / 8u) {
        return 1;
    }
    context->bit_count += (uint64_t)length * 8u;
    while (index < length) {
        size_t available = 64u - context->buffer_count;
        size_t take = length - index < available ? length - index : available;
        memcpy(context->buffer + context->buffer_count, bytes + index, take);
        context->buffer_count += take;
        index += take;
        if (context->buffer_count == 64u) {
            gs_sha_transform(context, context->buffer);
            context->buffer_count = 0u;
        }
    }
    return 0;
}

static void gs_sha_final(GSSha256 *context, unsigned char digest[32])
{
    uint64_t bit_count = context->bit_count;
    size_t index;
    context->buffer[context->buffer_count++] = 0x80u;
    if (context->buffer_count > 56u) {
        while (context->buffer_count < 64u) {
            context->buffer[context->buffer_count++] = 0u;
        }
        gs_sha_transform(context, context->buffer);
        context->buffer_count = 0u;
    }
    while (context->buffer_count < 56u) {
        context->buffer[context->buffer_count++] = 0u;
    }
    for (index = 0u; index < 8u; index++) {
        context->buffer[63u - index] =
            (unsigned char)(bit_count >> (index * 8u));
    }
    gs_sha_transform(context, context->buffer);
    for (index = 0u; index < 8u; index++) {
        digest[index * 4u] = (unsigned char)(context->state[index] >> 24);
        digest[index * 4u + 1u] =
            (unsigned char)(context->state[index] >> 16);
        digest[index * 4u + 2u] =
            (unsigned char)(context->state[index] >> 8);
        digest[index * 4u + 3u] =
            (unsigned char)context->state[index];
    }
}

static int gs_sha_file(const char *path, char output[65])
{
    unsigned char buffer[65536];
    unsigned char digest[32];
    static const char digits[] = "0123456789abcdef";
    GSSha256 context;
    FILE *file;
    size_t count;
    size_t index;
    int failed = 0;
    if (path == NULL || output == NULL || (file = fopen(path, "rb")) == NULL) {
        return 1;
    }
    gs_sha_init(&context);
    while ((count = fread(buffer, 1u, sizeof(buffer), file)) != 0u) {
        if (gs_sha_update(&context, buffer, count) != 0) {
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
    gs_sha_final(&context, digest);
    for (index = 0u; index < 32u; index++) {
        output[index * 2u] = digits[digest[index] >> 4];
        output[index * 2u + 1u] = digits[digest[index] & 15u];
    }
    output[64] = '\0';
    return 0;
}

static int gs_file_size(const char *path, size_t *size)
{
    struct stat information;
    if (path == NULL || size == NULL || stat(path, &information) != 0 ||
        !S_ISREG(information.st_mode) || information.st_size < 0 ||
        (uintmax_t)information.st_size > (uintmax_t)SIZE_MAX) {
        return 1;
    }
    *size = (size_t)information.st_size;
    return 0;
}

static void gs_chomp(char *line)
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

static int gs_file_has_key_value(
    const char *path,
    const char *key,
    const char *expected
)
{
    char line[GS_LINE_CAPACITY];
    char prefix[256];
    FILE *file;
    int written;
    int found = 0;
    written = snprintf(prefix, sizeof(prefix), "%s=", key);
    if (written < 0 || (size_t)written >= sizeof(prefix) ||
        (file = fopen(path, "rb")) == NULL) {
        return 0;
    }
    while (fgets(line, sizeof(line), file) != NULL) {
        gs_chomp(line);
        if (strncmp(line, prefix, strlen(prefix)) == 0) {
            found = strcmp(line + strlen(prefix), expected) == 0;
            break;
        }
    }
    if (ferror(file) || fclose(file) != 0) {
        return 0;
    }
    return found;
}

static int gs_name_in_source_inventory(const char *name)
{
    size_t index;
    for (index = 0u;
         index < sizeof(gs_source_file_names) / sizeof(gs_source_file_names[0]);
         index++) {
        if (strcmp(name, gs_source_file_names[index]) == 0) {
            return 1;
        }
    }
    return 0;
}

static int gs_validate_source_inventory(const char *source)
{
    DIR *stream;
    struct dirent *entry;
    size_t index;
    for (index = 0u;
         index < sizeof(gs_source_file_names) / sizeof(gs_source_file_names[0]);
         index++) {
        char path[GS_PATH_CAPACITY];
        size_t size;
        if (gs_path_join(source, gs_source_file_names[index], path,
                         sizeof(path)) != 0 ||
            gs_file_size(path, &size) != 0 || size == 0u) {
            return 1;
        }
    }
    stream = opendir(source);
    if (stream == NULL) {
        return 1;
    }
    while ((entry = readdir(stream)) != NULL) {
        char path[GS_PATH_CAPACITY];
        struct stat information;
        if (strcmp(entry->d_name, ".") == 0 ||
            strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        if (gs_path_join(source, entry->d_name, path, sizeof(path)) != 0 ||
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
        if (S_ISREG(information.st_mode) &&
            !gs_name_in_source_inventory(entry->d_name)) {
            closedir(stream);
            return 1;
        }
        if (S_ISDIR(information.st_mode) &&
            strcmp(entry->d_name, "kmeans_seed123") != 0 &&
            strcmp(entry->d_name, "label_evaluation_seed123") != 0) {
            closedir(stream);
            return 1;
        }
        if (!S_ISREG(information.st_mode) && !S_ISDIR(information.st_mode)) {
            closedir(stream);
            return 1;
        }
    }
    return closedir(stream) != 0;
}

static int gs_validate_source_sha(
    const GSResolved *resolved,
    const GSProfile *profile
)
{
    static const char *const artifact_order[] = {
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
    const char *input_names[4];
    char input_paths[4][GS_PATH_CAPACITY];
    char sha_path[GS_PATH_CAPACITY];
    char sha_manifest_hash[65];
    char line[GS_LINE_CAPACITY];
    FILE *file;
    size_t index;
    input_names[0] = "manifest_events_272.csv";
    input_names[1] = profile->tensor_manifest_name;
    input_names[2] = "split_train_indices_seed123.csv";
    input_names[3] = "split_val_indices_seed123.csv";
    for (index = 0u; index < 4u; index++) {
        if (gs_path_join(resolved->root, input_names[index],
                         input_paths[index], sizeof(input_paths[index])) != 0 ||
            gs_file_size(input_paths[index], &(size_t){0}) != 0) {
            return 1;
        }
    }
    if (gs_path_join(resolved->source, "sha256.txt", sha_path,
                     sizeof(sha_path)) != 0 ||
        gs_sha_file(sha_path, sha_manifest_hash) != 0 ||
        strcmp(
            sha_manifest_hash,
            profile->source_checksum_manifest_hash
        ) != 0 ||
        (file = fopen(sha_path, "rb")) == NULL) {
        return 1;
    }
    for (index = 0u; index < 4u + 9u; index++) {
        char path[GS_PATH_CAPACITY];
        char hash[65];
        char expected[GS_LINE_CAPACITY];
        const char *kind;
        const char *name;
        int written;
        if (index < 4u) {
            kind = "INPUT";
            name = input_names[index];
            strcpy(path, input_paths[index]);
        } else {
            kind = "ARTIFACT";
            name = artifact_order[index - 4u];
            if (gs_path_join(resolved->source, name, path, sizeof(path)) != 0) {
                fclose(file);
                return 1;
            }
        }
        if (gs_sha_file(path, hash) != 0 ||
            (written = snprintf(expected, sizeof(expected), "%s %s %s",
                                kind, hash, name)) < 0 ||
            (size_t)written >= sizeof(expected) ||
            fgets(line, sizeof(line), file) == NULL) {
            fclose(file);
            return 1;
        }
        gs_chomp(line);
        if (strcmp(line, expected) != 0) {
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

static int gs_parse_size(const char *text, size_t *value)
{
    uintmax_t parsed;
    char *end;
    if (text == NULL || value == NULL || text[0] == '\0' ||
        !isdigit((unsigned char)text[0])) {
        return 1;
    }
    errno = 0;
    parsed = strtoumax(text, &end, 10);
    if (errno != 0 || *end != '\0' || parsed > SIZE_MAX) {
        return 1;
    }
    *value = (size_t)parsed;
    return 0;
}

static int gs_csv_fields(char *line, char **fields, size_t capacity, size_t *count)
{
    char *read_cursor;
    char *write_cursor;
    size_t field_count = 0u;
    if (line == NULL || fields == NULL || count == NULL || capacity == 0u) {
        return 1;
    }
    read_cursor = line;
    write_cursor = line;
    for (;;) {
        int quoted = 0;
        if (field_count == capacity) {
            return 1;
        }
        fields[field_count++] = write_cursor;
        if (*read_cursor == '"') {
            quoted = 1;
            read_cursor++;
        }
        while (*read_cursor != '\0') {
            if (quoted) {
                if (*read_cursor == '"' && read_cursor[1] == '"') {
                    *write_cursor++ = '"';
                    read_cursor += 2;
                } else if (*read_cursor == '"') {
                    read_cursor++;
                    quoted = 0;
                    if (*read_cursor != ',' && *read_cursor != '\0') {
                        return 1;
                    }
                } else {
                    *write_cursor++ = *read_cursor++;
                }
            } else if (*read_cursor == ',') {
                read_cursor++;
                break;
            } else if (*read_cursor == '"') {
                return 1;
            } else {
                *write_cursor++ = *read_cursor++;
            }
        }
        if (quoted) {
            return 1;
        }
        *write_cursor++ = '\0';
        if (*read_cursor == '\0') {
            break;
        }
    }
    *count = field_count;
    return 0;
}

static int gs_copy_event_id(char output[GS_EVENT_ID_CAPACITY], const char *input)
{
    size_t length;
    if (input == NULL || input[0] == '\0') {
        return 1;
    }
    length = strlen(input);
    if (length >= GS_EVENT_ID_CAPACITY) {
        return 1;
    }
    memcpy(output, input, length + 1u);
    return 0;
}

static int gs_load_rows(
    const char *path,
    const char *split_name,
    size_t expected_count,
    int subset_schema,
    GSRow *rows
)
{
    const char *expected_header = subset_schema ?
        "split_name,split_pos,manifest_index,event_index,event_id,tensor_path" :
        "row_index,split_pos,manifest_index,event_index,event_id,split";
    char line[GS_LINE_CAPACITY];
    FILE *file;
    size_t row_index;
    if (rows == NULL || (file = fopen(path, "rb")) == NULL ||
        fgets(line, sizeof(line), file) == NULL) {
        if (file != NULL) {
            fclose(file);
        }
        return 1;
    }
    gs_chomp(line);
    if (strcmp(line, expected_header) != 0) {
        fclose(file);
        return 1;
    }
    for (row_index = 0u; row_index < expected_count; row_index++) {
        char *fields[6];
        size_t field_count;
        size_t first_value;
        if (fgets(line, sizeof(line), file) == NULL) {
            fclose(file);
            return 1;
        }
        gs_chomp(line);
        if (gs_csv_fields(line, fields, 6u, &field_count) != 0 ||
            field_count != 6u) {
            fclose(file);
            return 1;
        }
        if (subset_schema) {
            if (strcmp(fields[0], split_name) != 0 ||
                gs_parse_size(fields[1], &rows[row_index].split_pos) != 0 ||
                gs_parse_size(fields[2], &rows[row_index].manifest_index) != 0 ||
                gs_parse_size(fields[3], &rows[row_index].event_index) != 0 ||
                fields[5][0] == '\0') {
                fclose(file);
                return 1;
            }
        } else {
            if (gs_parse_size(fields[0], &first_value) != 0 ||
                first_value != row_index ||
                gs_parse_size(fields[1], &rows[row_index].split_pos) != 0 ||
                gs_parse_size(fields[2], &rows[row_index].manifest_index) != 0 ||
                gs_parse_size(fields[3], &rows[row_index].event_index) != 0 ||
                strcmp(fields[5], split_name) != 0) {
                fclose(file);
                return 1;
            }
        }
        if (rows[row_index].split_pos != row_index ||
            rows[row_index].manifest_index >= GS_EVENT_ROWS ||
            rows[row_index].event_index == 0u ||
            gs_copy_event_id(rows[row_index].event_id, fields[4]) != 0) {
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

static int gs_rows_equal(const GSRow *left, const GSRow *right)
{
    return left->split_pos == right->split_pos &&
        left->manifest_index == right->manifest_index &&
        left->event_index == right->event_index &&
        strcmp(left->event_id, right->event_id) == 0;
}

static int gs_validate_manifest_identifiers(
    const char *path,
    const GSRow *train,
    const GSRow *validation
)
{
    char line[GS_LINE_CAPACITY];
    char event_ids[GS_EVENT_ROWS][GS_EVENT_ID_CAPACITY];
    size_t event_indices[GS_EVENT_ROWS];
    unsigned char seen[GS_EVENT_ROWS] = {0};
    FILE *file;
    size_t row;
    if ((file = fopen(path, "rb")) == NULL ||
        fgets(line, sizeof(line), file) == NULL) {
        if (file != NULL) {
            fclose(file);
        }
        return 1;
    }
    for (row = 0u; row < GS_EVENT_ROWS; row++) {
        char *fields[7];
        size_t field_count;
        if (fgets(line, sizeof(line), file) == NULL) {
            fclose(file);
            return 1;
        }
        gs_chomp(line);
        if (gs_csv_fields(line, fields, 7u, &field_count) != 0 ||
            field_count != 7u ||
            gs_parse_size(fields[0], &event_indices[row]) != 0 ||
            event_indices[row] != row + 1u ||
            gs_copy_event_id(event_ids[row], fields[2]) != 0) {
            fclose(file);
            return 1;
        }
    }
    if (fgets(line, sizeof(line), file) != NULL || ferror(file) ||
        fclose(file) != 0) {
        return 1;
    }
    for (row = 0u; row < GS_MATCHED_TRAIN_ROWS; row++) {
        size_t index = train[row].manifest_index;
        if (seen[index] != 0u || train[row].event_index != event_indices[index] ||
            strcmp(train[row].event_id, event_ids[index]) != 0) {
            return 1;
        }
        seen[index] = 1u;
    }
    for (row = 0u; row < GS_MATCHED_VALIDATION_ROWS; row++) {
        size_t index = validation[row].manifest_index;
        if (seen[index] != 0u ||
            validation[row].event_index != event_indices[index] ||
            strcmp(validation[row].event_id, event_ids[index]) != 0) {
            return 1;
        }
        seen[index] = 1u;
    }
    return 0;
}

static int gs_read_raw_prefix_and_validate(
    const char *path,
    size_t source_rows,
    size_t matched_rows,
    size_t columns,
    float *prefix,
    double *minimum,
    double *maximum
)
{
    float buffer[8192];
    size_t total_values;
    size_t prefix_values;
    size_t processed = 0u;
    size_t copied = 0u;
    size_t expected_bytes;
    size_t actual_bytes;
    FILE *file;
    double local_min = DBL_MAX;
    double local_max = -DBL_MAX;
    if (prefix == NULL || minimum == NULL || maximum == NULL ||
        gs_checked_product(source_rows, columns, &total_values) != 0 ||
        gs_checked_product(matched_rows, columns, &prefix_values) != 0 ||
        gs_checked_product(total_values, sizeof(float), &expected_bytes) != 0 ||
        gs_file_size(path, &actual_bytes) != 0 || actual_bytes != expected_bytes ||
        (file = fopen(path, "rb")) == NULL) {
        return 1;
    }
    while (processed < total_values) {
        size_t remaining = total_values - processed;
        size_t request = remaining < sizeof(buffer) / sizeof(buffer[0]) ?
            remaining : sizeof(buffer) / sizeof(buffer[0]);
        size_t count = fread(buffer, sizeof(float), request, file);
        size_t index;
        if (count != request) {
            fclose(file);
            return 1;
        }
        for (index = 0u; index < count; index++) {
            double value = (double)buffer[index];
            if (!isfinite(value) || value < 0.0 || value > 1.0) {
                fclose(file);
                return 1;
            }
            if (value < local_min) local_min = value;
            if (value > local_max) local_max = value;
        }
        if (copied < prefix_values) {
            size_t take = prefix_values - copied < count ?
                prefix_values - copied : count;
            memcpy(prefix + copied, buffer, take * sizeof(float));
            copied += take;
        }
        processed += count;
    }
    if (fgetc(file) != EOF || ferror(file) || fclose(file) != 0 ||
        copied != prefix_values) {
        return 1;
    }
    *minimum = local_min;
    *maximum = local_max;
    return 0;
}

static int gs_validate_source_metadata(
    const GSResolved *resolved,
    const GSProfile *profile
)
{
    char feature_path[GS_PATH_CAPACITY];
    char summary_path[GS_PATH_CAPACITY];
    char value[64];
    int written;
    if (gs_path_join(resolved->source, "feature_spec.txt", feature_path,
                     sizeof(feature_path)) != 0 ||
        gs_path_join(resolved->source, "summary.txt", summary_path,
                     sizeof(summary_path)) != 0) {
        return 1;
    }
    written = snprintf(value, sizeof(value), "%zu", profile->feature_count);
    if (written < 0 || (size_t)written >= sizeof(value) ||
        !gs_file_has_key_value(feature_path, "profile", profile->source_profile) ||
        !gs_file_has_key_value(feature_path, "input_shape", profile->input_shape) ||
        !gs_file_has_key_value(feature_path, "feature_extractor",
                               profile->extractor) ||
        !gs_file_has_key_value(feature_path, "pooled_shape",
                               profile->pooled_shape) ||
        !gs_file_has_key_value(feature_path, "feature_count", value) ||
        !gs_file_has_key_value(feature_path, "matrix_layout", "row_major") ||
        !gs_file_has_key_value(feature_path, "matrix_dtype",
                               "float32_ieee754") ||
        !gs_file_has_key_value(feature_path, "normalization",
                               "uint8_div_255") ||
        !gs_file_has_key_value(feature_path, "code_commit",
                               profile->source_commit) ||
        !gs_file_has_key_value(summary_path, "profile",
                               profile->source_profile) ||
        !gs_file_has_key_value(summary_path, "train_rows", "6866") ||
        !gs_file_has_key_value(summary_path, "validation_rows", "1717") ||
        !gs_file_has_key_value(summary_path, "feature_count", value) ||
        !gs_file_has_key_value(summary_path, "code_commit",
                               profile->source_commit) ||
        !gs_file_has_key_value(summary_path, "raw_values_in_unit_interval",
                               "true")) {
        return 1;
    }
    return 0;
}

static int gs_validate_standardized(
    const float *train,
    const float *validation,
    const double *mean,
    const double *stddev,
    const unsigned char *constant_mask,
    size_t columns,
    GSStatistics *statistics
)
{
    size_t column;
    statistics->constant_count = 0u;
    statistics->max_abs_mean = 0.0;
    statistics->max_abs_stddev_error = 0.0;
    for (column = 0u; column < columns; column++) {
        size_t row;
        double average = 0.0;
        double variance = 0.0;
        if (!isfinite(mean[column]) || !isfinite(stddev[column]) ||
            stddev[column] < 0.0 || constant_mask[column] > 1u) {
            return 1;
        }
        for (row = 0u; row < GS_MATCHED_TRAIN_ROWS; row++) {
            float value = train[row * columns + column];
            if (!isfinite(value)) return 1;
            if (constant_mask[column] != 0u && value != 0.0f) return 1;
            average += (double)value;
        }
        for (row = 0u; row < GS_MATCHED_VALIDATION_ROWS; row++) {
            float value = validation[row * columns + column];
            if (!isfinite(value) ||
                (constant_mask[column] != 0u && value != 0.0f)) return 1;
        }
        if (constant_mask[column] != 0u) {
            statistics->constant_count++;
            continue;
        }
        average /= (double)GS_MATCHED_TRAIN_ROWS;
        for (row = 0u; row < GS_MATCHED_TRAIN_ROWS; row++) {
            double delta = (double)train[row * columns + column] - average;
            variance += delta * delta;
        }
        variance /= (double)GS_MATCHED_TRAIN_ROWS;
        if (fabs(average) > statistics->max_abs_mean) {
            statistics->max_abs_mean = fabs(average);
        }
        if (fabs(sqrt(variance) - 1.0) >
            statistics->max_abs_stddev_error) {
            statistics->max_abs_stddev_error =
                fabs(sqrt(variance) - 1.0);
        }
    }
    return statistics->max_abs_mean > 1e-6 ||
        statistics->max_abs_stddev_error > 1e-6;
}

static int gs_finish_file(FILE *file)
{
    int failed = 0;
    if (file == NULL) return 1;
    if (fflush(file) != 0 || ferror(file)) failed = 1;
    if (fclose(file) != 0) failed = 1;
    return failed;
}

static int gs_write_binary(
    const char *directory,
    const char *name,
    const float *values,
    size_t count
)
{
    char path[GS_PATH_CAPACITY];
    FILE *file;
    int failed = 0;
    if (gs_path_join(directory, name, path, sizeof(path)) != 0 ||
        (file = fopen(path, "wb")) == NULL) return 1;
    if (fwrite(values, sizeof(float), count, file) != count) failed = 1;
    if (gs_finish_file(file) != 0) failed = 1;
    return failed;
}

static int gs_write_rows(
    const char *directory,
    const char *name,
    const char *split,
    const GSRow *rows,
    size_t count
)
{
    char path[GS_PATH_CAPACITY];
    FILE *file;
    size_t index;
    int failed = 0;
    if (gs_path_join(directory, name, path, sizeof(path)) != 0 ||
        (file = fopen(path, "wb")) == NULL) return 1;
    if (fputs("row_index,split_pos,manifest_index,event_index,event_id,split\n",
              file) == EOF) failed = 1;
    for (index = 0u; !failed && index < count; index++) {
        if (strpbrk(rows[index].event_id, ",\"\r\n") != NULL ||
            fprintf(file, "%zu,%zu,%zu,%zu,%s,%s\n", index,
                    rows[index].split_pos, rows[index].manifest_index,
                    rows[index].event_index, rows[index].event_id, split) < 0) {
            failed = 1;
        }
    }
    if (gs_finish_file(file) != 0) failed = 1;
    return failed;
}

static int gs_write_scaler(
    const char *directory,
    const GSProfile *profile,
    const double *mean,
    const double *stddev,
    const unsigned char *constant_mask
)
{
    char path[GS_PATH_CAPACITY];
    FILE *file;
    size_t feature;
    int failed = 0;
    if (gs_path_join(directory, "scaler.csv", path, sizeof(path)) != 0 ||
        (file = fopen(path, "wb")) == NULL) return 1;
    if (fputs("feature_index,channel,pooled_row,pooled_col,mean,stddev,constant\n",
              file) == EOF) failed = 1;
    for (feature = 0u; !failed && feature < profile->feature_count; feature++) {
        size_t plane = profile->pooled_side * profile->pooled_side;
        size_t channel = feature / plane;
        size_t remainder = feature % plane;
        if (fprintf(file, "%zu,%zu,%zu,%zu,%.17g,%.17g,%u\n",
                    feature, channel, remainder / profile->pooled_side,
                    remainder % profile->pooled_side, mean[feature],
                    stddev[feature], (unsigned int)constant_mask[feature]) < 0) {
            failed = 1;
        }
    }
    if (gs_finish_file(file) != 0) failed = 1;
    return failed;
}

static int gs_write_feature_spec(
    const char *directory,
    const GSProfile *profile,
    const char *code_commit,
    size_t constant_count
)
{
    char path[GS_PATH_CAPACITY];
    FILE *file;
    int failed = 0;
    if (gs_path_join(directory, "feature_spec.txt", path, sizeof(path)) != 0 ||
        (file = fopen(path, "wb")) == NULL) return 1;
    if (fprintf(
            file,
            "format_version=1\n"
            "profile=%s\n"
            "source_profile=%s\n"
            "matched_subset_profile=subset224_seed123_prefix_800_200\n"
            "provenance=slice_raw_official_full_baseline\n"
            "input_shape=%s\n"
            "input_layout=channel_major\n"
            "normalization=already_applied_in_source_raw\n"
            "feature_extractor=%s\n"
            "pooled_shape=%s\n"
            "feature_count=%zu\n"
            "feature_order=channel_pooled_row_pooled_col\n"
            "matrix_layout=row_major\n"
            "matrix_dtype=float32_ieee754\n"
            "matrix_endian=little\n"
            "standardization=train_only_population_ddof0\n"
            "standardization_fit=train_only\n"
            "standardization_ddof=0\n"
            "constant_epsilon=1e-12\n"
            "constant_output=0\n"
            "constant_feature_count=%zu\n"
            "source_train_rows=6866\n"
            "source_validation_rows=1717\n"
            "train_rows=800\n"
            "validation_rows=200\n"
            "source_code_commit=%s\n"
            "matched_builder_code_commit=%s\n"
            "scaler_unchanged_after_validation=true\n"
            "pca=false\n"
            "augmentation=false\n"
            "labels_used=false\n",
            profile->matched_profile, profile->source_profile,
            profile->input_shape, profile->extractor,
            profile->pooled_shape, profile->feature_count,
            constant_count, profile->source_commit, code_commit
        ) < 0) failed = 1;
    if (gs_finish_file(file) != 0) failed = 1;
    return failed;
}

static int gs_write_summary(
    const char *directory,
    const GSProfile *profile,
    const char *code_commit,
    size_t train_bytes,
    size_t validation_bytes,
    const GSStatistics *statistics
)
{
    char path[GS_PATH_CAPACITY];
    FILE *file;
    int failed = 0;
    if (gs_path_join(directory, "summary.txt", path, sizeof(path)) != 0 ||
        (file = fopen(path, "wb")) == NULL) return 1;
    if (fprintf(
            file,
            "format_version=1\n"
            "profile=%s\n"
            "source_profile=%s\n"
            "matched_subset_profile=subset224_seed123_prefix_800_200\n"
            "provenance=slice_raw_official_full_baseline\n"
            "source_code_commit=%s\n"
            "matched_builder_code_commit=%s\n"
            "feature_count=%zu\n"
            "source_train_rows=6866\n"
            "source_validation_rows=1717\n"
            "train_rows=800\n"
            "validation_rows=200\n"
            "train_features_raw_bytes=%zu\n"
            "val_features_raw_bytes=%zu\n"
            "train_features_std_bytes=%zu\n"
            "val_features_std_bytes=%zu\n"
            "normalization=already_applied_in_source_raw\n"
            "standardization=train_only_population_ddof0\n"
            "standardization_fit=train_only\n"
            "standardization_ddof=0\n"
            "constant_epsilon=1e-12\n"
            "constant_feature_count=%zu\n"
            "train_raw_min=%.17g\n"
            "train_raw_max=%.17g\n"
            "validation_raw_min=%.17g\n"
            "validation_raw_max=%.17g\n"
            "train_std_nonconstant_max_abs_mean=%.17g\n"
            "train_std_nonconstant_max_abs_population_stddev_error=%.17g\n"
            "raw_train_is_bitwise_source_prefix=true\n"
            "raw_validation_is_bitwise_source_prefix=true\n"
            "validation_transformed_without_refit=true\n"
            "scaler_unchanged_after_validation=true\n"
            "standardized_values_finite=true\n"
            "pca=false\n"
            "augmentation=false\n"
            "labels_used=false\n"
            "kmeans_used=false\n",
            profile->matched_profile, profile->source_profile,
            profile->source_commit, code_commit, profile->feature_count,
            train_bytes, validation_bytes, train_bytes, validation_bytes,
            statistics->constant_count, statistics->train_raw_min,
            statistics->train_raw_max, statistics->validation_raw_min,
            statistics->validation_raw_max, statistics->max_abs_mean,
            statistics->max_abs_stddev_error
        ) < 0) failed = 1;
    if (gs_finish_file(file) != 0) failed = 1;
    return failed;
}

static int gs_source_input_paths(
    const GSResolved *resolved,
    char storage[GS_INPUT_COUNT][GS_PATH_CAPACITY],
    const char *paths[GS_INPUT_COUNT]
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
    strcpy(storage[0], resolved->manifest);
    strcpy(storage[1], resolved->subset_train);
    strcpy(storage[2], resolved->subset_validation);
    for (index = 0u; index < 6u; index++) {
        if (gs_path_join(resolved->source, source_names[index],
                         storage[index + 3u], GS_PATH_CAPACITY) != 0) {
            return 1;
        }
    }
    for (index = 0u; index < GS_INPUT_COUNT; index++) {
        paths[index] = storage[index];
    }
    return 0;
}

static int gs_write_checksum(
    const char *directory,
    const char *const input_paths[GS_INPUT_COUNT]
)
{
    char checksum_path[GS_PATH_CAPACITY];
    FILE *file;
    size_t index;
    int failed = 0;
    if (gs_path_join(directory, "sha256.txt", checksum_path,
                     sizeof(checksum_path)) != 0 ||
        (file = fopen(checksum_path, "wb")) == NULL) return 1;
    for (index = 0u; !failed && index < GS_INPUT_COUNT; index++) {
        char hash[65];
        if (gs_sha_file(input_paths[index], hash) != 0 ||
            fprintf(file, "INPUT %s %s\n", hash,
                    gs_input_logical_names[index]) < 0) failed = 1;
    }
    for (index = 0u; !failed && index < GS_ARTIFACT_COUNT; index++) {
        char path[GS_PATH_CAPACITY];
        char hash[65];
        if (gs_path_join(directory, gs_artifact_names[index], path,
                         sizeof(path)) != 0 ||
            gs_sha_file(path, hash) != 0 ||
            fprintf(file, "ARTIFACT %s %s\n", hash,
                    gs_artifact_names[index]) < 0) failed = 1;
    }
    if (gs_finish_file(file) != 0) failed = 1;
    return failed;
}

static int gs_validate_output(
    const char *directory,
    const char *const input_paths[GS_INPUT_COUNT],
    size_t train_bytes,
    size_t validation_bytes
)
{
    unsigned char seen[GS_TOTAL_OUTPUT_FILES] = {0};
    DIR *stream;
    struct dirent *entry;
    char checksum_path[GS_PATH_CAPACITY];
    char line[GS_LINE_CAPACITY];
    FILE *file;
    size_t index;
    stream = opendir(directory);
    if (stream == NULL) return 1;
    while ((entry = readdir(stream)) != NULL) {
        int matched = 0;
        if (strcmp(entry->d_name, ".") == 0 ||
            strcmp(entry->d_name, "..") == 0) continue;
        for (index = 0u; index < GS_ARTIFACT_COUNT; index++) {
            if (strcmp(entry->d_name, gs_artifact_names[index]) == 0) {
                if (seen[index] != 0u) {
                    closedir(stream);
                    return 1;
                }
                seen[index] = 1u;
                matched = 1;
            }
        }
        if (strcmp(entry->d_name, "sha256.txt") == 0) {
            if (seen[GS_ARTIFACT_COUNT] != 0u) {
                closedir(stream);
                return 1;
            }
            seen[GS_ARTIFACT_COUNT] = 1u;
            matched = 1;
        }
        if (!matched) {
            closedir(stream);
            return 1;
        }
    }
    if (closedir(stream) != 0) return 1;
    for (index = 0u; index < GS_TOTAL_OUTPUT_FILES; index++) {
        if (seen[index] == 0u) return 1;
    }
    for (index = 0u; index < GS_ARTIFACT_COUNT; index++) {
        char path[GS_PATH_CAPACITY];
        size_t size;
        size_t expected = 0u;
        if (index == 2u || index == 4u) expected = train_bytes;
        if (index == 3u || index == 5u) expected = validation_bytes;
        if (gs_path_join(directory, gs_artifact_names[index], path,
                         sizeof(path)) != 0 ||
            gs_file_size(path, &size) != 0 ||
            (expected != 0u && size != expected) ||
            (expected == 0u && size == 0u)) return 1;
    }
    if (gs_path_join(directory, "sha256.txt", checksum_path,
                     sizeof(checksum_path)) != 0 ||
        (file = fopen(checksum_path, "rb")) == NULL) return 1;
    for (index = 0u; index < GS_INPUT_COUNT + GS_ARTIFACT_COUNT; index++) {
        char path[GS_PATH_CAPACITY];
        char hash[65];
        char expected[GS_LINE_CAPACITY];
        const char *kind;
        const char *name;
        int written;
        if (index < GS_INPUT_COUNT) {
            kind = "INPUT";
            name = gs_input_logical_names[index];
            strcpy(path, input_paths[index]);
        } else {
            kind = "ARTIFACT";
            name = gs_artifact_names[index - GS_INPUT_COUNT];
            if (gs_path_join(directory, name, path, sizeof(path)) != 0) {
                fclose(file);
                return 1;
            }
        }
        if (gs_sha_file(path, hash) != 0 ||
            (written = snprintf(expected, sizeof(expected), "%s %s %s",
                                kind, hash, name)) < 0 ||
            (size_t)written >= sizeof(expected) ||
            fgets(line, sizeof(line), file) == NULL) {
            fclose(file);
            return 1;
        }
        gs_chomp(line);
        if (strcmp(line, expected) != 0) {
            fclose(file);
            return 1;
        }
    }
    if (fgets(line, sizeof(line), file) != NULL || ferror(file) ||
        fclose(file) != 0) return 1;
    return 0;
}

static int gs_remove_partial(const char *directory)
{
    size_t index;
    int failed = 0;
    if (directory == NULL) return 1;
    for (index = 0u; index < GS_ARTIFACT_COUNT; index++) {
        char path[GS_PATH_CAPACITY];
        if (gs_path_join(directory, gs_artifact_names[index], path,
                         sizeof(path)) == 0) {
            if (remove(path) != 0 && errno != ENOENT) failed = 1;
        } else {
            failed = 1;
        }
    }
    {
        char path[GS_PATH_CAPACITY];
        if (gs_path_join(directory, "sha256.txt", path, sizeof(path)) == 0) {
            if (remove(path) != 0 && errno != ENOENT) failed = 1;
        } else {
            failed = 1;
        }
    }
#ifdef _WIN32
    if (_rmdir(directory) != 0 && errno != ENOENT) failed = 1;
#else
    if (rmdir(directory) != 0 && errno != ENOENT) failed = 1;
#endif
    return failed;
}

static int gs_failure_hook_requested(void)
{
    const char *value = getenv("GS_MATCHED_SUBSET1000_TEST_FAIL_AFTER_PARTIAL");
    return value != NULL && strcmp(value, "1") == 0;
}

static int gs_platform_valid(void)
{
    const uint16_t value = 1u;
    return sizeof(float) == 4u && sizeof(double) == 8u &&
        FLT_RADIX == 2 && FLT_MANT_DIG == 24 &&
        *(const unsigned char *)&value == 1u;
}

#ifndef GS_MATCHED_SUBSET1000_NO_MAIN
int main(int argc, char **argv)
{
    const GSProfile *profile;
    GSResolved resolved;
    GSRow *source_train = NULL;
    GSRow *source_validation = NULL;
    GSRow subset_train[GS_MATCHED_TRAIN_ROWS];
    GSRow subset_validation[GS_MATCHED_VALIDATION_ROWS];
    float *train_raw = NULL;
    float *validation_raw = NULL;
    float *train_standardized = NULL;
    float *validation_standardized = NULL;
    double *mean = NULL;
    double *stddev = NULL;
    double *mean_copy = NULL;
    double *stddev_copy = NULL;
    unsigned char *constant_mask = NULL;
    unsigned char *constant_copy = NULL;
    GSStatistics statistics = {0};
    char path[GS_PATH_CAPACITY];
    char input_storage[GS_INPUT_COUNT][GS_PATH_CAPACITY];
    const char *input_paths[GS_INPUT_COUNT];
    size_t train_values;
    size_t validation_values;
    size_t train_bytes;
    size_t validation_bytes;
    size_t double_bytes;
    size_t index;
    int partial_created = 0;
    int result = 1;

    if (argc != 8) {
        fprintf(stderr,
            "Usage: build_matched_subset1000_no_pca_artifacts.exe "
            "<manifest_events> <source_full_no_pca_directory> "
            "<mini75_pool5|mini112_pool7> <subset_train_csv> "
            "<subset_val_csv> <output_partial_directory> <code_commit>\n");
        return 1;
    }
    profile = gs_profile_from_token(argv[3]);
    if (!gs_platform_valid() || profile == NULL ||
        !gs_commit_valid(argv[7]) ||
        gs_resolve_layout(argv[1], argv[2], profile, argv[4], argv[5],
                          argv[6], &resolved) != 0) {
        fprintf(stderr, "ERROR: invalid platform, profile, commit or canonical layout.\n");
        return 1;
    }
    printf("[PASS] canonical_source_and_output_resolution\n");
    if (gs_validate_source_inventory(resolved.source) != 0 ||
        gs_validate_source_sha(&resolved, profile) != 0 ||
        gs_validate_source_metadata(&resolved, profile) != 0) {
        fprintf(stderr, "ERROR: full source baseline validation failed.\n");
        return 1;
    }
    printf("[PASS] full_source_inventory_metadata_and_checksums\n");
    source_train = (GSRow *)calloc(GS_SOURCE_TRAIN_ROWS, sizeof(*source_train));
    source_validation =
        (GSRow *)calloc(GS_SOURCE_VALIDATION_ROWS, sizeof(*source_validation));
    if (source_train == NULL || source_validation == NULL ||
        gs_path_join(resolved.source, "train_rows.csv", path,
                     sizeof(path)) != 0 ||
        gs_load_rows(path, "train", GS_SOURCE_TRAIN_ROWS, 0,
                     source_train) != 0 ||
        gs_path_join(resolved.source, "val_rows.csv", path,
                     sizeof(path)) != 0 ||
        gs_load_rows(path, "validation", GS_SOURCE_VALIDATION_ROWS, 0,
                     source_validation) != 0 ||
        gs_load_rows(resolved.subset_train, "train", GS_MATCHED_TRAIN_ROWS,
                     1, subset_train) != 0 ||
        gs_load_rows(resolved.subset_validation, "validation",
                     GS_MATCHED_VALIDATION_ROWS, 1,
                     subset_validation) != 0) {
        fprintf(stderr, "ERROR: row-map or subset loading failed.\n");
        goto cleanup;
    }
    for (index = 0u; index < GS_MATCHED_TRAIN_ROWS; index++) {
        if (!gs_rows_equal(source_train + index, subset_train + index)) {
            fprintf(stderr, "ERROR: train subset is not exact source prefix.\n");
            goto cleanup;
        }
    }
    for (index = 0u; index < GS_MATCHED_VALIDATION_ROWS; index++) {
        if (!gs_rows_equal(source_validation + index,
                           subset_validation + index)) {
            fprintf(stderr, "ERROR: validation subset is not exact source prefix.\n");
            goto cleanup;
        }
    }
    if (gs_validate_manifest_identifiers(
            resolved.manifest, subset_train, subset_validation) != 0) {
        fprintf(stderr, "ERROR: subset identifiers or overlap invalid.\n");
        goto cleanup;
    }
    printf("[PASS] matched_subset_row_maps_and_manifest_identifiers\n");
    if (gs_checked_product(GS_MATCHED_TRAIN_ROWS, profile->feature_count,
                           &train_values) != 0 ||
        gs_checked_product(GS_MATCHED_VALIDATION_ROWS,
                           profile->feature_count, &validation_values) != 0 ||
        gs_checked_product(train_values, sizeof(float), &train_bytes) != 0 ||
        gs_checked_product(validation_values, sizeof(float),
                           &validation_bytes) != 0 ||
        gs_checked_product(profile->feature_count, sizeof(double),
                           &double_bytes) != 0) {
        fprintf(stderr, "ERROR: matrix dimension overflow.\n");
        goto cleanup;
    }
    train_raw = (float *)malloc(train_bytes);
    validation_raw = (float *)malloc(validation_bytes);
    train_standardized = (float *)malloc(train_bytes);
    validation_standardized = (float *)malloc(validation_bytes);
    mean = (double *)malloc(double_bytes);
    stddev = (double *)malloc(double_bytes);
    mean_copy = (double *)malloc(double_bytes);
    stddev_copy = (double *)malloc(double_bytes);
    constant_mask = (unsigned char *)malloc(profile->feature_count);
    constant_copy = (unsigned char *)malloc(profile->feature_count);
    if (train_raw == NULL || validation_raw == NULL ||
        train_standardized == NULL || validation_standardized == NULL ||
        mean == NULL || stddev == NULL || mean_copy == NULL ||
        stddev_copy == NULL || constant_mask == NULL ||
        constant_copy == NULL) {
        fprintf(stderr, "ERROR: allocation failure.\n");
        goto cleanup;
    }
    if (gs_path_join(resolved.source, "train_features_raw_f32.bin",
                     path, sizeof(path)) != 0 ||
        gs_read_raw_prefix_and_validate(
            path, GS_SOURCE_TRAIN_ROWS, GS_MATCHED_TRAIN_ROWS,
            profile->feature_count, train_raw,
            &statistics.train_raw_min, &statistics.train_raw_max) != 0 ||
        gs_path_join(resolved.source, "val_features_raw_f32.bin",
                     path, sizeof(path)) != 0 ||
        gs_read_raw_prefix_and_validate(
            path, GS_SOURCE_VALIDATION_ROWS, GS_MATCHED_VALIDATION_ROWS,
            profile->feature_count, validation_raw,
            &statistics.validation_raw_min,
            &statistics.validation_raw_max) != 0) {
        fprintf(stderr, "ERROR: raw source slicing or finite-value check failed.\n");
        goto cleanup;
    }
    printf("[PASS] complete_raw_source_validation_and_bitwise_prefix_slice\n");
    if (gs_baseline_standardization_fit(
            train_raw, GS_MATCHED_TRAIN_ROWS, profile->feature_count,
            mean, stddev, constant_mask,
            GS_BASELINE_STANDARDIZATION_DEFAULT_CONSTANT_EPSILON
        ) != GS_BASELINE_STANDARDIZATION_SUCCESS) {
        fprintf(stderr, "ERROR: train-only standardization fit failed.\n");
        goto cleanup;
    }
    memcpy(mean_copy, mean, double_bytes);
    memcpy(stddev_copy, stddev, double_bytes);
    memcpy(constant_copy, constant_mask, profile->feature_count);
    if (gs_baseline_standardization_transform(
            train_raw, GS_MATCHED_TRAIN_ROWS, profile->feature_count,
            mean, stddev, constant_mask, train_standardized
        ) != GS_BASELINE_STANDARDIZATION_SUCCESS ||
        gs_baseline_standardization_transform(
            validation_raw, GS_MATCHED_VALIDATION_ROWS,
            profile->feature_count, mean, stddev, constant_mask,
            validation_standardized
        ) != GS_BASELINE_STANDARDIZATION_SUCCESS ||
        memcmp(mean, mean_copy, double_bytes) != 0 ||
        memcmp(stddev, stddev_copy, double_bytes) != 0 ||
        memcmp(constant_mask, constant_copy, profile->feature_count) != 0 ||
        gs_validate_standardized(
            train_standardized, validation_standardized, mean, stddev,
            constant_mask, profile->feature_count, &statistics) != 0) {
        fprintf(stderr, "ERROR: matched standardization contract failed.\n");
        goto cleanup;
    }
    printf("[PASS] train_only_standardization_and_scaler_immutability\n");
    if (gs_source_input_paths(&resolved, input_storage, input_paths) != 0) {
        goto cleanup;
    }
#ifdef _WIN32
    if (_mkdir(resolved.output_partial) != 0) {
#else
    if (mkdir(resolved.output_partial, 0777) != 0) {
#endif
        fprintf(stderr, "ERROR: cannot create output partial.\n");
        goto cleanup;
    }
    partial_created = 1;
    if (gs_failure_hook_requested()) {
        fprintf(stderr, "ERROR: confined test failure after partial creation.\n");
        goto cleanup;
    }
    if (gs_write_feature_spec(resolved.output_partial, profile, argv[7],
                              statistics.constant_count) != 0 ||
        gs_write_scaler(resolved.output_partial, profile, mean, stddev,
                        constant_mask) != 0 ||
        gs_write_binary(resolved.output_partial,
                        "train_features_raw_f32.bin",
                        train_raw, train_values) != 0 ||
        gs_write_binary(resolved.output_partial,
                        "val_features_raw_f32.bin",
                        validation_raw, validation_values) != 0 ||
        gs_write_binary(resolved.output_partial,
                        "train_features_std_f32.bin",
                        train_standardized, train_values) != 0 ||
        gs_write_binary(resolved.output_partial,
                        "val_features_std_f32.bin",
                        validation_standardized, validation_values) != 0 ||
        gs_write_rows(resolved.output_partial, "train_rows.csv", "train",
                      subset_train, GS_MATCHED_TRAIN_ROWS) != 0 ||
        gs_write_rows(resolved.output_partial, "val_rows.csv", "validation",
                      subset_validation, GS_MATCHED_VALIDATION_ROWS) != 0 ||
        gs_write_summary(resolved.output_partial, profile, argv[7],
                         train_bytes, validation_bytes, &statistics) != 0 ||
        gs_write_checksum(resolved.output_partial, input_paths) != 0 ||
        gs_validate_output(resolved.output_partial, input_paths,
                           train_bytes, validation_bytes) != 0) {
        fprintf(stderr, "ERROR: output write or internal verification failed.\n");
        goto cleanup;
    }
    if (rename(resolved.output_partial, resolved.output_final) != 0) {
        fprintf(stderr, "ERROR: atomic promotion failed.\n");
        goto cleanup;
    }
    partial_created = 0;
    printf("[PASS] verified_partial_promoted_atomically\n");
    printf("PROFILE=%s\n", profile->matched_profile);
    printf("SOURCE_PROFILE=%s\n", profile->source_profile);
    printf("TRAIN_ROWS=%d\n", GS_MATCHED_TRAIN_ROWS);
    printf("VALIDATION_ROWS=%d\n", GS_MATCHED_VALIDATION_ROWS);
    printf("FEATURE_DIM=%zu\n", profile->feature_count);
    printf("CONSTANT_FEATURES=%zu\n", statistics.constant_count);
    printf("TRAIN_STD_MAX_ABS_MEAN=%.17g\n", statistics.max_abs_mean);
    printf("TRAIN_STD_MAX_ABS_POPULATION_STDDEV_ERROR=%.17g\n",
           statistics.max_abs_stddev_error);
    printf("SCALER_UNCHANGED_AFTER_VALIDATION=true\n");
    printf("ARTIFACT BUILD COMPLETED\n");
    result = 0;

cleanup:
    if (partial_created) {
        if (gs_remove_partial(resolved.output_partial) != 0) {
            fprintf(stderr, "ERROR: confined partial cleanup failed.\n");
            result = 1;
        } else {
            printf("[PASS] current_run_partial_cleanup\n");
        }
    }
    free(constant_copy);
    free(constant_mask);
    free(stddev_copy);
    free(mean_copy);
    free(stddev);
    free(mean);
    free(validation_standardized);
    free(train_standardized);
    free(validation_raw);
    free(train_raw);
    free(source_validation);
    free(source_train);
    return result;
}
#endif
