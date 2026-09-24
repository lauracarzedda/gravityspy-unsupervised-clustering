#include "baseline_kmeans.h"

#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/types.h>
#endif


enum {
    GS_RUN_PATH_LENGTH = 4096,
    GS_RUN_LINE_LENGTH = 8192,
    GS_RUN_TRAIN_ROWS = 6866,
    GS_RUN_VALIDATION_ROWS = 1717,
    GS_RUN_SEED = 123,
    GS_RUN_MAX_ITERATIONS = 300,
    GS_RUN_PCA_ARTIFACT_COUNT = 15
};

#define GS_RUN_PARTIAL_SUFFIX ".partial"


typedef struct {
    const char *name;
    const char *train_filename;
    const char *validation_filename;
    size_t components;
} GSPCAProfile;


typedef struct {
    uint32_t state[8];
    uint64_t bit_count;
    unsigned char buffer[64];
    size_t buffer_length;
} GSSha256;


typedef struct {
    double minimum;
    double maximum;
    double mean;
    double sum;
} GSDistanceStatistics;


static int gs_checked_product(size_t left, size_t right, size_t *product)
{
    if (product == NULL || (right != 0 && left > SIZE_MAX / right)) {
        return 1;
    }
    *product = left * right;
    return 0;
}


static int gs_checked_sum(size_t left, size_t right, size_t *sum)
{
    if (sum == NULL || left > SIZE_MAX - right) {
        return 1;
    }
    *sum = left + right;
    return 0;
}


static int gs_platform_is_valid(void)
{
    const uint16_t marker = UINT16_C(1);
    const unsigned char *bytes = (const unsigned char *)&marker;

    return sizeof(float) == 4 &&
           FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 &&
           sizeof(double) == 8 &&
           DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024 &&
           sizeof(uint32_t) == 4 && sizeof(uint64_t) == 8 &&
           bytes[0] == 1u;
}


static int gs_metadata_value_is_valid(const char *value)
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


static int gs_commit_is_valid(const char *commit)
{
    size_t index;

    if (!gs_metadata_value_is_valid(commit) || strlen(commit) != 40) {
        return 0;
    }
    for (index = 0; index < 40; index++) {
        if (!((commit[index] >= '0' && commit[index] <= '9') ||
              (commit[index] >= 'a' && commit[index] <= 'f'))) {

            return 0;
        }
    }
    return 1;
}


static int gs_parse_size_value(const char *text, size_t *value)
{
    char *end;
    unsigned long long parsed;

    if (text == NULL || value == NULL || text[0] == '\0' ||
        text[0] == '-') {

        return 1;
    }
    errno = 0;
    end = NULL;
    parsed = strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' ||
        parsed > (unsigned long long)SIZE_MAX) {

        return 1;
    }
    *value = (size_t)parsed;
    return 0;
}


static int gs_parse_u64_value(const char *text, uint64_t *value)
{
    char *end;
    unsigned long long parsed;

    if (text == NULL || value == NULL || text[0] == '\0' ||
        text[0] == '-') {

        return 1;
    }
    errno = 0;
    end = NULL;
    parsed = strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        return 1;
    }
    *value = (uint64_t)parsed;
    return 0;
}


static int gs_parse_double_value(const char *text, double *value)
{
    char *end;
    double parsed;

    if (text == NULL || value == NULL || text[0] == '\0') {
        return 1;
    }
    errno = 0;
    end = NULL;
    parsed = strtod(text, &end);
    if (errno == ERANGE || end == text || *end != '\0' ||
        !isfinite(parsed)) {

        return 1;
    }
    *value = parsed;
    return 0;
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
    size_t position = 0;
    int separator;

    if (directory == NULL || filename == NULL || path == NULL ||
        path_size == 0) {

        return 1;
    }
    directory_length = strlen(directory);
    filename_length = strlen(filename);
    separator = directory_length != 0 &&
        directory[directory_length - 1] != '/' &&
        directory[directory_length - 1] != '\\';

    if (gs_checked_sum(directory_length, filename_length, &required) != 0 ||
        (separator && gs_checked_sum(required, 1, &required) != 0) ||
        required >= path_size) {

        return 1;
    }
    if (directory_length != 0) {
        memcpy(path, directory, directory_length);
        position = directory_length;
    }
    if (separator) {
        path[position++] = '/';
    }
    if (filename_length != 0) {
        memcpy(path + position, filename, filename_length);
        position += filename_length;
    }
    path[position] = '\0';
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


static int gs_validate_output_directory(
    const char *partial_directory,
    char *final_directory,
    size_t final_size
)
{
    size_t partial_length;
    size_t suffix_length = strlen(GS_RUN_PARTIAL_SUFFIX);
    size_t final_length;

    if (!gs_metadata_value_is_valid(partial_directory) ||
        final_directory == NULL || final_size == 0) {

        return 1;
    }
    partial_length = strlen(partial_directory);
    if (partial_length <= suffix_length ||
        strcmp(
            partial_directory + partial_length - suffix_length,
            GS_RUN_PARTIAL_SUFFIX
        ) != 0) {

        fprintf(stderr, "Errore: output deve terminare con .partial.\n");
        return 1;
    }
    final_length = partial_length - suffix_length;
    if (final_length == 0 || final_length >= final_size) {
        return 1;
    }
    memcpy(final_directory, partial_directory, final_length);
    final_directory[final_length] = '\0';

    if (gs_path_state(partial_directory) != 0 ||
        gs_path_state(final_directory) != 0) {

        fprintf(stderr, "Errore: directory partial o finale gia' esistente.\n");
        return 1;
    }
    return 0;
}


static int gs_create_partial_directory(
    const char *partial_directory,
    const char *final_directory
)
{
    if (gs_path_state(partial_directory) != 0 ||
        gs_path_state(final_directory) != 0) {

        return 1;
    }
#ifdef _WIN32
    if (_mkdir(partial_directory) != 0) {
#else
    if (mkdir(partial_directory, 0777) != 0) {
#endif
        fprintf(stderr, "Errore: creazione directory partial fallita.\n");
        return 1;
    }
    return 0;
}


static int gs_file_size(const char *path, size_t *size)
{
    FILE *file;
    long position;
    int failed = 0;

    if (path == NULL || size == NULL) {
        return 1;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        return 1;
    }
    if (fseek(file, 0L, SEEK_END) != 0) {
        failed = 1;
    }
    position = failed ? -1L : ftell(file);
    if (position < 0) {
        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    if (failed) {
        return 1;
    }
    *size = (size_t)position;
    return 0;
}


static void gs_strip_line_ending(char *line)
{
    size_t length = strlen(line);

    while (length != 0 &&
           (line[length - 1] == '\r' || line[length - 1] == '\n')) {

        line[--length] = '\0';
    }
}


static int gs_split_csv_line(
    char *line,
    char **fields,
    size_t expected_fields
)
{
    size_t count = 1;
    char *cursor;

    if (line == NULL || fields == NULL || expected_fields == 0) {
        return 1;
    }
    fields[0] = line;
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


static int gs_read_thresholds(
    const char *directory,
    size_t *d90,
    size_t *d95,
    size_t *d99
)
{
    static const double expected_thresholds[3] = {0.90, 0.95, 0.99};
    char path[GS_RUN_PATH_LENGTH];
    char line[GS_RUN_LINE_LENGTH];
    size_t counts[3];
    double previous_cumulative = 0.0;
    FILE *file;
    size_t row;
    int failed = 0;

    if (d90 == NULL || d95 == NULL || d99 == NULL ||
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
        gs_strip_line_ending(line);
        if (strcmp(
                line,
                "threshold,component_count,cumulative_variance"
            ) != 0) {

            failed = 1;
        }
    }

    for (row = 0; !failed && row < 3; row++) {
        char *fields[3];
        double threshold;
        double cumulative;

        if (fgets(line, sizeof(line), file) == NULL ||
            (strchr(line, '\n') == NULL && !feof(file))) {

            failed = 1;
            break;
        }
        gs_strip_line_ending(line);
        if (gs_split_csv_line(line, fields, 3) != 0 ||
            gs_parse_double_value(fields[0], &threshold) != 0 ||
            gs_parse_size_value(fields[1], &counts[row]) != 0 ||
            gs_parse_double_value(fields[2], &cumulative) != 0 ||
            threshold != expected_thresholds[row] ||
            counts[row] == 0 ||
            cumulative < threshold || cumulative > 1.0 ||
            cumulative < previous_cumulative) {

            failed = 1;
            break;
        }
        previous_cumulative = cumulative;
    }
    if (!failed && fgets(line, sizeof(line), file) != NULL) {
        failed = 1;
    }
    if (ferror(file) || fclose(file) != 0) {
        failed = 1;
    }
    if (failed || counts[0] != 21 || counts[1] != 62 || counts[2] != 263) {
        return 1;
    }
    *d90 = counts[0];
    *d95 = counts[1];
    *d99 = counts[2];
    return 0;
}


static int gs_select_profile(
    const char *name,
    size_t d90,
    size_t d95,
    size_t d99,
    GSPCAProfile *profile
)
{
    if (name == NULL || profile == NULL ||
        d90 != 21 || d95 != 62 || d99 != 263) {

        return 1;
    }
    if (strcmp(name, "pca90") == 0) {
        profile->name = "pca90";
        profile->train_filename = "train_scores_pca90_f32.bin";
        profile->validation_filename = "val_scores_pca90_f32.bin";
        profile->components = d90;
    } else if (strcmp(name, "pca95") == 0) {
        profile->name = "pca95";
        profile->train_filename = "train_scores_pca95_f32.bin";
        profile->validation_filename = "val_scores_pca95_f32.bin";
        profile->components = d95;
    } else if (strcmp(name, "pca99") == 0) {
        profile->name = "pca99";
        profile->train_filename = "train_scores_pca99_f32.bin";
        profile->validation_filename = "val_scores_pca99_f32.bin";
        profile->components = d99;
    } else {
        return 1;
    }
    return 0;
}


static uint32_t gs_rotr32(uint32_t value, unsigned int shift)
{
    return (value >> shift) | (value << (32u - shift));
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
    uint32_t schedule[64];
    uint32_t a;
    uint32_t b;
    uint32_t c;
    uint32_t d;
    uint32_t e;
    uint32_t f;
    uint32_t g;
    uint32_t h;
    size_t index;

    for (index = 0; index < 16; index++) {
        size_t offset = index * 4;

        schedule[index] =
            ((uint32_t)block[offset] << 24) |
            ((uint32_t)block[offset + 1] << 16) |
            ((uint32_t)block[offset + 2] << 8) |
            (uint32_t)block[offset + 3];
    }
    for (index = 16; index < 64; index++) {
        uint32_t left = schedule[index - 15];
        uint32_t right = schedule[index - 2];
        uint32_t sigma0 =
            gs_rotr32(left, 7) ^ gs_rotr32(left, 18) ^ (left >> 3);
        uint32_t sigma1 =
            gs_rotr32(right, 17) ^ gs_rotr32(right, 19) ^ (right >> 10);

        schedule[index] =
            schedule[index - 16] + sigma0 +
            schedule[index - 7] + sigma1;
    }

    a = context->state[0];
    b = context->state[1];
    c = context->state[2];
    d = context->state[3];
    e = context->state[4];
    f = context->state[5];
    g = context->state[6];
    h = context->state[7];

    for (index = 0; index < 64; index++) {
        uint32_t sum1 =
            gs_rotr32(e, 6) ^ gs_rotr32(e, 11) ^ gs_rotr32(e, 25);
        uint32_t choose = (e & f) ^ ((~e) & g);
        uint32_t temporary1 =
            h + sum1 + choose + constants[index] + schedule[index];
        uint32_t sum0 =
            gs_rotr32(a, 2) ^ gs_rotr32(a, 13) ^ gs_rotr32(a, 22);
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
    static const uint32_t initial[8] = {
        UINT32_C(0x6a09e667), UINT32_C(0xbb67ae85),
        UINT32_C(0x3c6ef372), UINT32_C(0xa54ff53a),
        UINT32_C(0x510e527f), UINT32_C(0x9b05688c),
        UINT32_C(0x1f83d9ab), UINT32_C(0x5be0cd19)
    };

    memcpy(context->state, initial, sizeof(initial));
    context->bit_count = UINT64_C(0);
    context->buffer_length = 0;
}


static int gs_sha256_update(
    GSSha256 *context,
    const unsigned char *data,
    size_t length
)
{
    size_t consumed = 0;

    if (context == NULL || (data == NULL && length != 0) ||
        length > (UINT64_MAX - context->bit_count) / UINT64_C(8)) {

        return 1;
    }
    context->bit_count += (uint64_t)length * UINT64_C(8);

    while (consumed < length) {
        size_t available = 64 - context->buffer_length;
        size_t remaining = length - consumed;
        size_t amount = remaining < available ? remaining : available;

        memcpy(
            context->buffer + context->buffer_length,
            data + consumed,
            amount
        );
        context->buffer_length += amount;
        consumed += amount;
        if (context->buffer_length == 64) {
            gs_sha256_transform(context, context->buffer);
            context->buffer_length = 0;
        }
    }
    return 0;
}


static void gs_sha256_final(GSSha256 *context, unsigned char digest[32])
{
    uint64_t bit_count = context->bit_count;
    size_t index;

    context->buffer[context->buffer_length++] = 0x80u;
    if (context->buffer_length > 56) {
        while (context->buffer_length < 64) {
            context->buffer[context->buffer_length++] = 0;
        }
        gs_sha256_transform(context, context->buffer);
        context->buffer_length = 0;
    }
    while (context->buffer_length < 56) {
        context->buffer[context->buffer_length++] = 0;
    }
    for (index = 0; index < 8; index++) {
        context->buffer[63 - index] =
            (unsigned char)(bit_count >> (index * 8));
    }
    gs_sha256_transform(context, context->buffer);
    for (index = 0; index < 8; index++) {
        digest[index * 4] =
            (unsigned char)(context->state[index] >> 24);
        digest[index * 4 + 1] =
            (unsigned char)(context->state[index] >> 16);
        digest[index * 4 + 2] =
            (unsigned char)(context->state[index] >> 8);
        digest[index * 4 + 3] =
            (unsigned char)context->state[index];
    }
}


static int gs_sha256_file(const char *path, char output[65])
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
    while ((count = fread(buffer, 1, sizeof(buffer), file)) != 0) {
        if (gs_sha256_update(&context, buffer, count) != 0) {
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
    gs_sha256_final(&context, digest);
    for (index = 0; index < sizeof(digest); index++) {
        output[index * 2] = digits[digest[index] >> 4];
        output[index * 2 + 1] = digits[digest[index] & 0x0fu];
    }
    output[64] = '\0';
    return 0;
}


static int gs_lower_hex_digest(const char *text)
{
    size_t index;

    if (text == NULL || strlen(text) < 65) {
        return 0;
    }
    for (index = 0; index < 64; index++) {
        if (!((text[index] >= '0' && text[index] <= '9') ||
              (text[index] >= 'a' && text[index] <= 'f'))) {

            return 0;
        }
    }
    return text[64] == ' ';
}


static int gs_read_source_directory(
    const char *pca_directory,
    char *source_directory,
    size_t source_size
)
{
    static const char prefix[] = "source_directory=";
    char path[GS_RUN_PATH_LENGTH];
    char line[GS_RUN_LINE_LENGTH];
    FILE *file;
    int found = 0;
    int failed = 0;

    if (source_directory == NULL || source_size == 0 ||
        gs_path_join(
            pca_directory, "summary.txt", path, sizeof(path)
        ) != 0) {

        return 1;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        return 1;
    }
    while (fgets(line, sizeof(line), file) != NULL) {
        size_t length = strlen(line);

        if (length != 0 && line[length - 1] != '\n' && !feof(file)) {
            failed = 1;
            break;
        }
        gs_strip_line_ending(line);
        if (strncmp(line, prefix, sizeof(prefix) - 1) == 0) {
            const char *value = line + sizeof(prefix) - 1;

            if (found || !gs_metadata_value_is_valid(value) ||
                strlen(value) >= source_size) {

                failed = 1;
                break;
            }
            memcpy(source_directory, value, strlen(value) + 1);
            found = 1;
        }
    }
    if (ferror(file) || fclose(file) != 0) {
        failed = 1;
    }
    return failed || !found ? 1 : 0;
}


static int gs_verify_pca_artifact_hashes(const char *directory)
{
    static const char *input_names[7] = {
        "feature_spec.txt",
        "scaler.csv",
        "train_features_std_f32.bin",
        "val_features_std_f32.bin",
        "train_rows.csv",
        "val_rows.csv",
        "source_sha256.txt"
    };
    static const char *artifact_names[GS_RUN_PCA_ARTIFACT_COUNT] = {
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
        "train_scores_pca95_f32.bin",
        "val_scores_pca95_f32.bin",
        "train_scores_pca99_f32.bin",
        "val_scores_pca99_f32.bin",
        "summary.txt"
    };
    char manifest_path[GS_RUN_PATH_LENGTH];
    char source_directory[GS_RUN_PATH_LENGTH];
    char line[GS_RUN_LINE_LENGTH];
    char expected_inputs[7][65];
    char expected[GS_RUN_PCA_ARTIFACT_COUNT][65];
    int found_inputs[7] = {0};
    int found[GS_RUN_PCA_ARTIFACT_COUNT] = {0};
    FILE *file;
    size_t input_count = 0;
    size_t artifact_count = 0;
    int failed = 0;

    if (gs_path_join(
            directory, "sha256.txt",
            manifest_path, sizeof(manifest_path)
        ) != 0) {

        return 1;
    }
    file = fopen(manifest_path, "rb");
    if (file == NULL) {
        return 1;
    }

    while (fgets(line, sizeof(line), file) != NULL) {
        size_t length = strlen(line);
        const char *digest;
        const char *filename;

        if (length != 0 && line[length - 1] != '\n' && !feof(file)) {
            failed = 1;
            break;
        }
        gs_strip_line_ending(line);
        if (strncmp(line, "INPUT ", 6) == 0) {
            size_t index;
            int matched = 0;

            digest = line + 6;
            if (!gs_lower_hex_digest(digest) || digest[65] == '\0') {

                failed = 1;
                break;
            }
            filename = digest + 65;
            for (index = 0; index < 7; index++) {
                if (strcmp(filename, input_names[index]) == 0) {
                    if (found_inputs[index]) {
                        failed = 1;
                    } else {
                        memcpy(expected_inputs[index], digest, 64);
                        expected_inputs[index][64] = '\0';
                        found_inputs[index] = 1;
                        input_count++;
                    }
                    matched = 1;
                    break;
                }
            }
            if (!matched || failed) {
                failed = 1;
                break;
            }
        } else if (strncmp(line, "ARTIFACT ", 9) == 0) {
            size_t index;
            int matched = 0;

            digest = line + 9;
            if (!gs_lower_hex_digest(digest) || digest[65] == '\0') {
                failed = 1;
                break;
            }
            filename = digest + 65;
            if (strcmp(filename, "sha256.txt") == 0) {
                failed = 1;
                break;
            }
            for (index = 0; index < GS_RUN_PCA_ARTIFACT_COUNT; index++) {
                if (strcmp(filename, artifact_names[index]) == 0) {
                    if (found[index]) {
                        failed = 1;
                    } else {
                        memcpy(expected[index], digest, 64);
                        expected[index][64] = '\0';
                        found[index] = 1;
                        artifact_count++;
                    }
                    matched = 1;
                    break;
                }
            }
            if (!matched || failed) {
                failed = 1;
                break;
            }
        } else {
            failed = 1;
            break;
        }
    }
    if (ferror(file) || fclose(file) != 0) {
        failed = 1;
    }
    if (failed || input_count != 7 ||
        artifact_count != GS_RUN_PCA_ARTIFACT_COUNT) {

        return 1;
    }

    {
        size_t index;

        if (gs_read_source_directory(
                directory,
                source_directory,
                sizeof(source_directory)
            ) != 0) {

            return 1;
        }
        for (index = 0; index < 7; index++) {
            const char *filename =
                strcmp(input_names[index], "source_sha256.txt") == 0
                    ? "sha256.txt"
                    : input_names[index];
            char path[GS_RUN_PATH_LENGTH];
            char actual[65];

            if (!found_inputs[index] ||
                gs_path_join(
                    source_directory, filename, path, sizeof(path)
                ) != 0 ||
                gs_sha256_file(path, actual) != 0 ||
                strcmp(actual, expected_inputs[index]) != 0) {

                return 1;
            }
        }
        for (index = 0; index < GS_RUN_PCA_ARTIFACT_COUNT; index++) {
            char path[GS_RUN_PATH_LENGTH];
            char actual[65];

            if (!found[index] ||
                gs_path_join(
                    directory, artifact_names[index],
                    path, sizeof(path)
                ) != 0 ||
                gs_sha256_file(path, actual) != 0 ||
                strcmp(actual, expected[index]) != 0) {

                return 1;
            }
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


static int gs_open_new_output(
    const char *directory,
    const char *filename,
    FILE **file
)
{
    char path[GS_RUN_PATH_LENGTH];

    if (file == NULL ||
        gs_path_join(directory, filename, path, sizeof(path)) != 0) {

        return 1;
    }
    *file = fopen(path, "wbx");
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
    FILE *file = NULL;
    int failed = 0;

    if (values == NULL || element_size == 0 ||
        gs_open_new_output(directory, filename, &file) != 0) {

        return 1;
    }
    if (fwrite(values, element_size, count, file) != count) {
        failed = 1;
    }
    if (gs_finish_write(file) != 0) {
        failed = 1;
    }
    return failed;
}


static int gs_distance_statistics(
    const double *distances,
    size_t count,
    GSDistanceStatistics *statistics
)
{
    size_t index;
    double total = 0.0;

    if (distances == NULL || statistics == NULL || count == 0) {
        return 1;
    }
    statistics->minimum = distances[0];
    statistics->maximum = distances[0];
    for (index = 0; index < count; index++) {
        double value = distances[index];

        if (!isfinite(value) || value < 0.0) {
            return 1;
        }
        total += value;
        if (!isfinite(total)) {
            return 1;
        }
        if (value < statistics->minimum) {
            statistics->minimum = value;
        }
        if (value > statistics->maximum) {
            statistics->maximum = value;
        }
    }
    statistics->sum = total;
    statistics->mean = total / (double)count;
    return isfinite(statistics->mean) ? 0 : 1;
}


static int gs_validate_and_convert(
    const GSBaselineKMeansResult *result,
    const size_t *validation_assignments,
    size_t validation_rows,
    const double *train_distances,
    const double *validation_distances,
    uint32_t *train_u32,
    uint32_t *validation_u32,
    uint64_t *cluster_sizes_u64,
    uint64_t *initial_indices_u64,
    GSDistanceStatistics *train_statistics,
    GSDistanceStatistics *validation_statistics,
    size_t *empty_clusters,
    size_t *minimum_cluster_size,
    size_t *maximum_cluster_size
)
{
    size_t *recomputed_sizes;
    size_t row;
    size_t cluster;
    size_t total_size = 0;
    double tolerance;

    if (result == NULL || validation_assignments == NULL ||
        train_distances == NULL || validation_distances == NULL ||
        train_u32 == NULL || validation_u32 == NULL ||
        cluster_sizes_u64 == NULL || initial_indices_u64 == NULL ||
        train_statistics == NULL || validation_statistics == NULL ||
        empty_clusters == NULL || minimum_cluster_size == NULL ||
        maximum_cluster_size == NULL ||
        result->k > (size_t)UINT32_MAX) {

        return 1;
    }
    recomputed_sizes = (size_t *)calloc(result->k, sizeof(size_t));
    if (recomputed_sizes == NULL) {
        return 1;
    }
    for (row = 0; row < result->rows; row++) {
        size_t assignment = result->assignments[row];

        if (assignment >= result->k || assignment > (size_t)UINT32_MAX) {
            free(recomputed_sizes);
            return 1;
        }
        train_u32[row] = (uint32_t)assignment;
        recomputed_sizes[assignment]++;
    }
    for (row = 0; row < validation_rows; row++) {
        size_t assignment = validation_assignments[row];

        if (assignment >= result->k || assignment > (size_t)UINT32_MAX) {
            free(recomputed_sizes);
            return 1;
        }
        validation_u32[row] = (uint32_t)assignment;
    }

    *empty_clusters = 0;
    *minimum_cluster_size = result->rows;
    *maximum_cluster_size = 0;
    for (cluster = 0; cluster < result->k; cluster++) {
        size_t size = result->cluster_sizes[cluster];
        size_t left;

        if (size != recomputed_sizes[cluster] ||
            total_size > SIZE_MAX - size) {

            free(recomputed_sizes);
            return 1;
        }
        total_size += size;
        cluster_sizes_u64[cluster] = (uint64_t)size;
        if (size == 0) {
            (*empty_clusters)++;
        }
        if (size < *minimum_cluster_size) {
            *minimum_cluster_size = size;
        }
        if (size > *maximum_cluster_size) {
            *maximum_cluster_size = size;
        }
        if (result->initial_centroid_indices[cluster] >= result->rows) {
            free(recomputed_sizes);
            return 1;
        }
        initial_indices_u64[cluster] =
            (uint64_t)result->initial_centroid_indices[cluster];
        for (left = 0; left < cluster; left++) {
            if (result->initial_centroid_indices[left] ==
                result->initial_centroid_indices[cluster]) {

                free(recomputed_sizes);
                return 1;
            }
        }
    }
    free(recomputed_sizes);

    if (total_size != result->rows ||
        !isfinite(result->initial_inertia) || result->initial_inertia < 0.0 ||
        !isfinite(result->inertia) || result->inertia < 0.0 ||
        gs_distance_statistics(
            train_distances, result->rows, train_statistics
        ) != 0 ||
        gs_distance_statistics(
            validation_distances, validation_rows, validation_statistics
        ) != 0) {

        return 1;
    }
    tolerance = 1.0e-10 * (1.0 + fabs(result->inertia));
    return fabs(train_statistics->sum - result->inertia) <= tolerance ? 0 : 1;
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
        gs_open_new_output(directory, filename, &file) != 0) {

        return 1;
    }
    if (fputs("row_index,cluster,squared_distance\n", file) == EOF) {
        failed = 1;
    }
    for (row = 0; !failed && row < rows; row++) {
        if (fprintf(
                file, "%zu,%u,%.17g\n",
                row, (unsigned int)assignments[row], distances[row]
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
    const uint64_t *sizes,
    size_t k
)
{
    FILE *file = NULL;
    size_t cluster;
    int failed = 0;

    if (sizes == NULL ||
        gs_open_new_output(directory, "cluster_sizes.csv", &file) != 0) {

        return 1;
    }
    if (fputs("cluster,train_size\n", file) == EOF) {
        failed = 1;
    }
    for (cluster = 0; !failed && cluster < k; cluster++) {
        if (fprintf(
                file, "%zu,%llu\n",
                cluster, (unsigned long long)sizes[cluster]
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
        gs_open_new_output(directory, "initial_centroids.csv", &file) != 0) {

        return 1;
    }
    if (fputs("selection_order,train_row_index\n", file) == EOF) {
        failed = 1;
    }
    for (order = 0; !failed && order < k; order++) {
        if (fprintf(
                file, "%zu,%llu\n",
                order, (unsigned long long)indices[order]
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
    uint64_t seed,
    size_t max_iterations,
    const GSBaselineKMeansResult *result,
    const GSDistanceStatistics *train_statistics,
    const GSDistanceStatistics *validation_statistics,
    size_t empty_clusters,
    size_t minimum_cluster_size,
    size_t maximum_cluster_size,
    double cpu_seconds,
    const char *code_commit
)
{
    FILE *file = NULL;
    int failed = 0;

    if (profile == NULL || result == NULL || train_statistics == NULL ||
        validation_statistics == NULL ||
        gs_open_new_output(directory, "summary.txt", &file) != 0) {

        return 1;
    }
    if (fprintf(
            file,
            "format_version=1\n"
            "profile=mini75_pool5_pca_kmeans\n"
            "source_profile=mini75_pool5_pca\n"
            "algorithm=kmeans_plus_plus_lloyd\n"
            "input=train_scores_%s_f32\n"
            "validation_policy=assign_to_train_centroids\n"
            "train_rows=6866\n"
            "validation_rows=1717\n"
            "feature_count=%zu\n"
            "pca_profile=%s\n"
            "pca_components=%zu\n"
            "pca_whitening=false\n"
            "post_pca_standardization=false\n"
            "k=%zu\n"
            "seed=%llu\n"
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
            "train_squared_distance_mean=%.17g\n"
            "train_squared_distance_minimum=%.17g\n"
            "train_squared_distance_maximum=%.17g\n"
            "validation_squared_distance_mean=%.17g\n"
            "validation_squared_distance_minimum=%.17g\n"
            "validation_squared_distance_maximum=%.17g\n"
            "fit_cpu_seconds_approx=%.17g\n"
            "centroid_dtype=float64\n"
            "assignment_dtype=uint32\n"
            "distance_dtype=float64\n"
            "cluster_size_dtype=uint64\n"
            "fit_data=train_only\n"
            "fit_input=train_only\n"
            "validation_assigned_after_fit=true\n"
            "augmentation=false\n"
            "labels_used=false\n"
            "code_commit=%s\n",
            profile->name,
            profile->components,
            profile->name,
            profile->components,
            k,
            (unsigned long long)seed,
            max_iterations,
            result->iterations,
            result->converged,
            result->empty_cluster_reseeds,
            result->initial_inertia,
            result->inertia,
            validation_statistics->sum,
            empty_clusters,
            minimum_cluster_size,
            maximum_cluster_size,
            train_statistics->mean,
            train_statistics->minimum,
            train_statistics->maximum,
            validation_statistics->mean,
            validation_statistics->minimum,
            validation_statistics->maximum,
            cpu_seconds,
            code_commit
        ) < 0) {

        failed = 1;
    }
    if (gs_finish_write(file) != 0) {
        failed = 1;
    }
    return failed;
}


static void gs_print_usage(const char *program)
{
    printf(
        "Usage: %s <pca_artifact_directory> <pca_profile> <k> <seed> "
        "<max_iterations> <output_partial_directory> <code_commit>\n"
        "pca_profile: pca90 | pca95 | pca99\n"
        "fixed contract: train_rows=6866 validation_rows=1717 "
        "seed=123 max_iterations=300 k=22|36\n",
        program
    );
}


int main(int argc, char **argv)
{
    GSPCAProfile profile;
    size_t d90;
    size_t d95;
    size_t d99;
    size_t k;
    size_t max_iterations;
    uint64_t seed;
    size_t train_values;
    size_t validation_values;
    size_t centroid_values;
    size_t train_bytes;
    size_t validation_bytes;
    size_t centroid_bytes;
    size_t train_assignment_bytes;
    size_t validation_assignment_bytes;
    size_t validation_size_t_bytes;
    size_t train_distance_bytes;
    size_t validation_distance_bytes;
    size_t k_u64_bytes;
    char train_path[GS_RUN_PATH_LENGTH];
    char validation_path[GS_RUN_PATH_LENGTH];
    char final_directory[GS_RUN_PATH_LENGTH];

    float *train = NULL;
    float *validation = NULL;
    size_t *validation_assignments = NULL;
    double *train_distances = NULL;
    double *validation_distances = NULL;
    double *centroids_copy = NULL;
    uint32_t *train_u32 = NULL;
    uint32_t *validation_u32 = NULL;
    uint64_t *cluster_sizes_u64 = NULL;
    uint64_t *initial_indices_u64 = NULL;
    GSBaselineKMeansResult result;
    GSDistanceStatistics train_statistics;
    GSDistanceStatistics validation_statistics;
    size_t empty_clusters;
    size_t minimum_cluster_size;
    size_t maximum_cluster_size;
    clock_t start_clock;
    clock_t end_clock;
    double cpu_seconds;
    int rc = 1;

    memset(&result, 0, sizeof(result));
    memset(&profile, 0, sizeof(profile));

    if (argc == 2 && strcmp(argv[1], "--help") == 0) {
        gs_print_usage(argv[0]);
        return 0;
    }
    if (argc != 8) {
        gs_print_usage(argv[0]);
        return 1;
    }
    if (!gs_platform_is_valid() ||
        !gs_metadata_value_is_valid(argv[1]) ||
        gs_parse_size_value(argv[3], &k) != 0 ||
        gs_parse_u64_value(argv[4], &seed) != 0 ||
        gs_parse_size_value(argv[5], &max_iterations) != 0 ||
        !gs_commit_is_valid(argv[7]) ||
        (k != 22 && k != 36) ||
        seed != GS_RUN_SEED ||
        max_iterations != GS_RUN_MAX_ITERATIONS ||
        gs_validate_output_directory(
            argv[6], final_directory, sizeof(final_directory)
        ) != 0) {

        fprintf(stderr, "Errore: argomenti o piattaforma non validi.\n");
        return 1;
    }
    if (gs_verify_pca_artifact_hashes(argv[1]) != 0) {
        fprintf(stderr, "Errore: verifica sha256 PCA fallita.\n");
        return 1;
    }
    if (gs_read_thresholds(argv[1], &d90, &d95, &d99) != 0 ||
        gs_select_profile(argv[2], d90, d95, d99, &profile) != 0 ||
        gs_path_join(
            argv[1], profile.train_filename,
            train_path, sizeof(train_path)
        ) != 0 ||
        gs_path_join(
            argv[1], profile.validation_filename,
            validation_path, sizeof(validation_path)
        ) != 0) {

        fprintf(stderr, "Errore: profilo o thresholds PCA non validi.\n");
        return 1;
    }

    if (gs_checked_product(
            GS_RUN_TRAIN_ROWS, profile.components, &train_values
        ) != 0 ||
        gs_checked_product(
            GS_RUN_VALIDATION_ROWS, profile.components, &validation_values
        ) != 0 ||
        gs_checked_product(k, profile.components, &centroid_values) != 0 ||
        gs_checked_product(train_values, sizeof(float), &train_bytes) != 0 ||
        gs_checked_product(
            validation_values, sizeof(float), &validation_bytes
        ) != 0 ||
        gs_checked_product(
            centroid_values, sizeof(double), &centroid_bytes
        ) != 0 ||
        gs_checked_product(
            GS_RUN_TRAIN_ROWS, sizeof(uint32_t), &train_assignment_bytes
        ) != 0 ||
        gs_checked_product(
            GS_RUN_VALIDATION_ROWS,
            sizeof(uint32_t),
            &validation_assignment_bytes
        ) != 0 ||
        gs_checked_product(
            GS_RUN_VALIDATION_ROWS,
            sizeof(size_t),
            &validation_size_t_bytes
        ) != 0 ||
        gs_checked_product(
            GS_RUN_TRAIN_ROWS, sizeof(double), &train_distance_bytes
        ) != 0 ||
        gs_checked_product(
            GS_RUN_VALIDATION_ROWS,
            sizeof(double),
            &validation_distance_bytes
        ) != 0 ||
        gs_checked_product(k, sizeof(uint64_t), &k_u64_bytes) != 0) {

        fprintf(stderr, "Errore: overflow nelle dimensioni.\n");
        return 1;
    }

    train = (float *)malloc(train_bytes);
    validation = (float *)malloc(validation_bytes);
    validation_assignments = (size_t *)malloc(validation_size_t_bytes);
    train_distances = (double *)malloc(train_distance_bytes);
    validation_distances = (double *)malloc(validation_distance_bytes);
    centroids_copy = (double *)malloc(centroid_bytes);
    train_u32 = (uint32_t *)malloc(train_assignment_bytes);
    validation_u32 = (uint32_t *)malloc(validation_assignment_bytes);
    cluster_sizes_u64 = (uint64_t *)malloc(k_u64_bytes);
    initial_indices_u64 = (uint64_t *)malloc(k_u64_bytes);

    if (train == NULL || validation == NULL ||
        validation_assignments == NULL || train_distances == NULL ||
        validation_distances == NULL || centroids_copy == NULL ||
        train_u32 == NULL || validation_u32 == NULL ||
        cluster_sizes_u64 == NULL || initial_indices_u64 == NULL) {

        fprintf(stderr, "Errore: memoria insufficiente.\n");
        goto cleanup;
    }
    if (gs_read_float_matrix(
            train_path, train, train_values, train_bytes
        ) != 0 ||
        gs_read_float_matrix(
            validation_path, validation, validation_values, validation_bytes
        ) != 0) {

        fprintf(stderr, "Errore: lettura score PCA fallita.\n");
        goto cleanup;
    }
    printf("[PASS] pca_profile_and_inputs\n");
    printf(
        "KMEANS_FIT_STARTED profile=%s k=%zu rows=6866 cols=%zu\n",
        profile.name, k, profile.components
    );
    fflush(stdout);

    start_clock = clock();
    if (gs_baseline_kmeans_fit(
            train,
            GS_RUN_TRAIN_ROWS,
            profile.components,
            k,
            seed,
            max_iterations,
            &result
        ) != GS_BASELINE_KMEANS_SUCCESS) {

        fprintf(stderr, "Errore: fit k-means fallito.\n");
        goto cleanup;
    }
    end_clock = clock();
    if (start_clock == (clock_t)-1 || end_clock == (clock_t)-1 ||
        end_clock < start_clock) {

        cpu_seconds = -1.0;
    } else {
        cpu_seconds =
            (double)(end_clock - start_clock) / (double)CLOCKS_PER_SEC;
    }
    printf("KMEANS_FIT_COMPLETED\n");

    if (gs_baseline_kmeans_assign(
            train,
            GS_RUN_TRAIN_ROWS,
            profile.components,
            result.centroids,
            k,
            result.assignments,
            train_distances
        ) != GS_BASELINE_KMEANS_SUCCESS) {

        fprintf(stderr, "Errore: ricalcolo train fallito.\n");
        goto cleanup;
    }
    printf("[PASS] train_fit_only\n");

    memcpy(centroids_copy, result.centroids, centroid_bytes);
    if (gs_baseline_kmeans_assign(
            validation,
            GS_RUN_VALIDATION_ROWS,
            profile.components,
            result.centroids,
            k,
            validation_assignments,
            validation_distances
        ) != GS_BASELINE_KMEANS_SUCCESS ||
        memcmp(centroids_copy, result.centroids, centroid_bytes) != 0) {

        fprintf(stderr, "Errore: assegnazione validation fallita.\n");
        goto cleanup;
    }
    printf("[PASS] validation_assignment\n");

    if (gs_validate_and_convert(
            &result,
            validation_assignments,
            GS_RUN_VALIDATION_ROWS,
            train_distances,
            validation_distances,
            train_u32,
            validation_u32,
            cluster_sizes_u64,
            initial_indices_u64,
            &train_statistics,
            &validation_statistics,
            &empty_clusters,
            &minimum_cluster_size,
            &maximum_cluster_size
        ) != 0) {

        fprintf(stderr, "Errore: contratto numerico non valido.\n");
        goto cleanup;
    }
    printf("[PASS] numeric_validation\n");

    if (gs_create_partial_directory(argv[6], final_directory) != 0) {
        goto cleanup;
    }
    if (gs_write_binary(
            argv[6], "centroids_f64.bin",
            result.centroids, sizeof(double), centroid_values
        ) != 0 ||
        gs_write_binary(
            argv[6], "train_assignments_u32.bin",
            train_u32, sizeof(uint32_t), GS_RUN_TRAIN_ROWS
        ) != 0 ||
        gs_write_binary(
            argv[6], "val_assignments_u32.bin",
            validation_u32, sizeof(uint32_t), GS_RUN_VALIDATION_ROWS
        ) != 0 ||
        gs_write_binary(
            argv[6], "train_squared_distances_f64.bin",
            train_distances, sizeof(double), GS_RUN_TRAIN_ROWS
        ) != 0 ||
        gs_write_binary(
            argv[6], "val_squared_distances_f64.bin",
            validation_distances, sizeof(double), GS_RUN_VALIDATION_ROWS
        ) != 0 ||
        gs_write_binary(
            argv[6], "cluster_sizes_train_u64.bin",
            cluster_sizes_u64, sizeof(uint64_t), k
        ) != 0 ||
        gs_write_binary(
            argv[6], "initial_centroid_indices_u64.bin",
            initial_indices_u64, sizeof(uint64_t), k
        ) != 0 ||
        gs_write_assignment_csv(
            argv[6], "train_assignments.csv",
            train_u32, train_distances, GS_RUN_TRAIN_ROWS
        ) != 0 ||
        gs_write_assignment_csv(
            argv[6], "val_assignments.csv",
            validation_u32, validation_distances, GS_RUN_VALIDATION_ROWS
        ) != 0 ||
        gs_write_cluster_sizes_csv(argv[6], cluster_sizes_u64, k) != 0 ||
        gs_write_initial_centroids_csv(argv[6], initial_indices_u64, k) != 0 ||
        gs_write_summary(
            argv[6],
            &profile,
            k,
            seed,
            max_iterations,
            &result,
            &train_statistics,
            &validation_statistics,
            empty_clusters,
            minimum_cluster_size,
            maximum_cluster_size,
            cpu_seconds,
            argv[7]
        ) != 0) {

        fprintf(stderr, "Errore: scrittura artefatti k-means fallita.\n");
        goto cleanup;
    }

    printf("[PASS] artifacts_written\n");
    printf("PCA_PROFILE=%s\n", profile.name);
    printf("PCA_COMPONENTS=%zu\n", profile.components);
    printf("K=%zu\n", k);
    printf("INITIAL_INERTIA=%.17g\n", result.initial_inertia);
    printf("FINAL_TRAIN_INERTIA=%.17g\n", result.inertia);
    printf("VALIDATION_INERTIA=%.17g\n", validation_statistics.sum);
    printf("ITERATIONS=%zu\n", result.iterations);
    printf("CONVERGED=%d\n", result.converged);
    printf("EMPTY_CLUSTER_RESEEDS=%zu\n", result.empty_cluster_reseeds);
    printf("EMPTY_FINAL_CLUSTERS=%zu\n", empty_clusters);
    printf("CLUSTER_SIZE_MIN=%zu\n", minimum_cluster_size);
    printf("CLUSTER_SIZE_MAX=%zu\n", maximum_cluster_size);
    printf("FIT_CPU_SECONDS_APPROX=%.17g\n", cpu_seconds);
    printf("PCA KMEANS RUN COMPLETED\n");
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
    free(validation);
    free(train);
    return rc;
}
