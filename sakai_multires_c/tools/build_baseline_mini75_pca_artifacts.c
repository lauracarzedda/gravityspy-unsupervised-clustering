#include "baseline_pca.h"

#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#ifdef _WIN32
#include <direct.h>
#endif


enum {
    GS_PCA_TRAIN_ROWS = 6866,
    GS_PCA_VALIDATION_ROWS = 1717,
    GS_PCA_INPUT_DIM = 900,
    GS_PCA_PATH_LENGTH = 4096,
    GS_PCA_SHA256_HEX_LENGTH = 64,
    GS_PCA_SOURCE_HASH_COUNT = 6,
    GS_PCA_ARTIFACT_COUNT = 15
};

#define GS_PCA_PARTIAL_SUFFIX ".partial"
#define GS_PCA_SCORE_MEAN_TOLERANCE 5.0e-5
#define GS_PCA_SCORE_VARIANCE_TOLERANCE 5.0e-4


typedef struct {
    uint32_t state[8];
    uint64_t bit_count;
    unsigned char buffer[64];
    size_t buffer_length;
} GSSha256;


typedef struct {
    size_t input_dim;
    size_t component_count;
    double total_variance;
    size_t components_90;
    size_t components_95;
    size_t components_99;
} GSPCAModelMetadata;


typedef struct {
    const char *name;
    size_t size;
} GSPCAArtifactSize;


static int gs_checked_product(
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
           sizeof(double) == 8 &&
           FLT_RADIX == 2 &&
           FLT_MANT_DIG == 24 &&
           DBL_MANT_DIG == 53 &&
           FLT_MAX_EXP == 128 &&
           DBL_MAX_EXP == 1024 &&
           bytes[0] == 1;
}


static int gs_metadata_value_is_valid(const char *value)
{
    const unsigned char *cursor = (const unsigned char *)value;

    if (value == NULL || value[0] == '\0') {
        return 0;
    }
    while (*cursor != '\0') {
        if (*cursor < 32 || *cursor == 127) {
            return 0;
        }
        cursor++;
    }
    return 1;
}


static int gs_commit_is_valid(const char *commit)
{
    size_t length;
    size_t index;

    if (!gs_metadata_value_is_valid(commit)) {
        return 0;
    }
    length = strlen(commit);
    if (length < 7 || length > 64) {
        return 0;
    }

    for (index = 0; index < length; index++) {
        char value = commit[index];

        if (!((value >= '0' && value <= '9') ||
              (value >= 'a' && value <= 'f') ||
              (value >= 'A' && value <= 'F'))) {

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
    size_t position = 0;
    int separator;

    if (directory == NULL || filename == NULL || path == NULL ||
        path_size == 0) {

        return 1;
    }

    directory_length = strlen(directory);
    filename_length = strlen(filename);
    separator =
        directory_length != 0 &&
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
    size_t suffix_length = strlen(GS_PCA_PARTIAL_SUFFIX);
    size_t final_length;

    if (!gs_metadata_value_is_valid(partial_directory) ||
        final_directory == NULL ||
        final_size == 0) {

        return 1;
    }

    partial_length = strlen(partial_directory);
    if (partial_length <= suffix_length ||
        strcmp(
            partial_directory + partial_length - suffix_length,
            GS_PCA_PARTIAL_SUFFIX
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


static void gs_sha256_final(
    GSSha256 *context,
    unsigned char digest[32]
)
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


static int gs_is_hex_character(char value)
{
    return (value >= '0' && value <= '9') ||
           (value >= 'a' && value <= 'f') ||
           (value >= 'A' && value <= 'F');
}


static int gs_extract_sha256(const char *line, char output[65])
{
    size_t length;
    size_t start;

    if (line == NULL || output == NULL) {
        return 1;
    }
    length = strlen(line);

    for (start = 0; start + 64 <= length; start++) {
        size_t index;
        int valid = 1;

        if (start != 0 && gs_is_hex_character(line[start - 1])) {
            continue;
        }
        for (index = 0; index < 64; index++) {
            if (!gs_is_hex_character(line[start + index])) {
                valid = 0;
                break;
            }
        }
        if (valid &&
            (start + 64 == length ||
             !gs_is_hex_character(line[start + 64]))) {

            for (index = 0; index < 64; index++) {
                char value = line[start + index];

                output[index] =
                    value >= 'A' && value <= 'F'
                        ? (char)(value - 'A' + 'a')
                        : value;
            }
            output[64] = '\0';
            return 0;
        }
    }
    return 1;
}


static int gs_verify_source_hashes(const char *source_directory)
{
    static const char *filenames[GS_PCA_SOURCE_HASH_COUNT] = {
        "feature_spec.txt",
        "scaler.csv",
        "train_features_std_f32.bin",
        "val_features_std_f32.bin",
        "train_rows.csv",
        "val_rows.csv"
    };
    char manifest_path[GS_PCA_PATH_LENGTH];
    char line[8192];
    char expected[GS_PCA_SOURCE_HASH_COUNT][65];
    int found[GS_PCA_SOURCE_HASH_COUNT] = {0, 0, 0, 0, 0, 0};
    FILE *file;
    size_t index;
    int failed = 0;

    if (gs_path_join(
            source_directory,
            "sha256.txt",
            manifest_path,
            sizeof(manifest_path)
        ) != 0) {

        return 1;
    }
    file = fopen(manifest_path, "rb");
    if (file == NULL) {
        return 1;
    }

    while (fgets(line, sizeof(line), file) != NULL) {
        size_t length = strlen(line);

        if (length != 0 && line[length - 1] != '\n' && !feof(file)) {
            failed = 1;
            break;
        }
        while (length != 0 &&
               (line[length - 1] == '\n' || line[length - 1] == '\r')) {

            line[--length] = '\0';
        }

        for (index = 0; index < GS_PCA_SOURCE_HASH_COUNT; index++) {
            if (strstr(line, filenames[index]) != NULL) {
                if (found[index] ||
                    gs_extract_sha256(line, expected[index]) != 0) {

                    failed = 1;
                    break;
                }
                found[index] = 1;
            }
        }
        if (failed) {
            break;
        }
    }
    if (ferror(file) || fclose(file) != 0) {
        failed = 1;
    }
    if (failed) {
        return 1;
    }

    for (index = 0; index < GS_PCA_SOURCE_HASH_COUNT; index++) {
        char path[GS_PCA_PATH_LENGTH];
        char actual[65];

        if (!found[index] ||
            gs_path_join(
                source_directory,
                filenames[index],
                path,
                sizeof(path)
            ) != 0 ||
            gs_sha256_file(path, actual) != 0 ||
            strcmp(actual, expected[index]) != 0) {

            return 1;
        }
    }
    return 0;
}


static int gs_text_has_line(const char *text, const char *line)
{
    const char *cursor = text;
    size_t length = strlen(line);

    while ((cursor = strstr(cursor, line)) != NULL) {
        const char *after = cursor + length;

        if ((cursor == text || cursor[-1] == '\n') &&
            (*after == '\0' || *after == '\r' || *after == '\n')) {

            return 1;
        }
        cursor++;
    }
    return 0;
}


static int gs_read_small_text(
    const char *path,
    size_t maximum_size,
    char **text
)
{
    size_t size;
    FILE *file;
    char *buffer;
    int failed = 0;

    if (path == NULL || text == NULL ||
        gs_file_size(path, &size) != 0 ||
        size == 0 ||
        size > maximum_size ||
        size == SIZE_MAX) {

        return 1;
    }

    buffer = (char *)malloc(size + 1);
    if (buffer == NULL) {
        return 1;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        free(buffer);
        return 1;
    }
    if (fread(buffer, 1, size, file) != size ||
        fgetc(file) != EOF ||
        ferror(file) ||
        memchr(buffer, '\0', size) != NULL) {

        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    if (failed) {
        free(buffer);
        return 1;
    }

    buffer[size] = '\0';
    *text = buffer;
    return 0;
}


static int gs_validate_feature_spec(const char *source_directory)
{
    static const char *required[] = {
        "profile=mini75_pool5_no_pca",
        "feature_count=900",
        "matrix_layout=row_major",
        "matrix_dtype=float32_ieee754",
        "matrix_endian=little",
        "standardization_fit=train_only",
        "standardization_ddof=0",
        "pca=false",
        "train_rows=6866",
        "validation_rows=1717"
    };
    char path[GS_PCA_PATH_LENGTH];
    char *text = NULL;
    size_t index;
    int failed = 0;

    if (gs_path_join(
            source_directory,
            "feature_spec.txt",
            path,
            sizeof(path)
        ) != 0 ||
        gs_read_small_text(path, 1024u * 1024u, &text) != 0) {

        free(text);
        return 1;
    }

    for (index = 0;
         index < sizeof(required) / sizeof(required[0]);
         index++) {

        if (!gs_text_has_line(text, required[index])) {
            failed = 1;
        }
    }

    free(text);
    return failed;
}


static int gs_validate_csv_line_count(
    const char *path,
    size_t expected_data_rows
)
{
    FILE *file;
    char line[16384];
    size_t lines = 0;
    int failed = 0;

    if (path == NULL || expected_data_rows == SIZE_MAX) {
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
        if (memchr(line, '\0', length) == NULL) {
            lines++;
        }
    }
    if (ferror(file) || fclose(file) != 0) {
        failed = 1;
    }

    return failed || lines != expected_data_rows + 1;
}


static int gs_validate_source_files(
    const char *source_directory,
    size_t train_bytes,
    size_t validation_bytes
)
{
    static const char *nonempty_files[] = {
        "feature_spec.txt",
        "scaler.csv",
        "train_rows.csv",
        "val_rows.csv",
        "sha256.txt"
    };
    char path[GS_PCA_PATH_LENGTH];
    size_t size;
    size_t index;

    for (index = 0;
         index < sizeof(nonempty_files) / sizeof(nonempty_files[0]);
         index++) {

        if (gs_path_join(
                source_directory,
                nonempty_files[index],
                path,
                sizeof(path)
            ) != 0 ||
            gs_file_size(path, &size) != 0 ||
            size == 0) {

            return 1;
        }
    }

    if (gs_path_join(
            source_directory,
            "train_features_std_f32.bin",
            path,
            sizeof(path)
        ) != 0 ||
        gs_file_size(path, &size) != 0 ||
        size != train_bytes ||
        gs_path_join(
            source_directory,
            "val_features_std_f32.bin",
            path,
            sizeof(path)
        ) != 0 ||
        gs_file_size(path, &size) != 0 ||
        size != validation_bytes ||
        gs_path_join(
            source_directory,
            "train_rows.csv",
            path,
            sizeof(path)
        ) != 0 ||
        gs_validate_csv_line_count(path, GS_PCA_TRAIN_ROWS) != 0 ||
        gs_path_join(
            source_directory,
            "val_rows.csv",
            path,
            sizeof(path)
        ) != 0 ||
        gs_validate_csv_line_count(path, GS_PCA_VALIDATION_ROWS) != 0 ||
        gs_validate_feature_spec(source_directory) != 0) {

        return 1;
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
    size_t size;
    size_t index;
    int failed = 0;

    if (path == NULL || values == NULL ||
        gs_file_size(path, &size) != 0 ||
        size != expected_bytes) {

        return 1;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        return 1;
    }
    if (fread(values, sizeof(float), value_count, file) != value_count ||
        fgetc(file) != EOF ||
        ferror(file)) {

        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    if (failed) {
        return 1;
    }

    for (index = 0; index < value_count; index++) {
        if (!isfinite((double)values[index])) {
            return 1;
        }
    }
    return 0;
}


static int gs_model_validate(
    const GSBaselinePCA *model,
    size_t *zero_eigenvalue_count
)
{
    size_t component;
    size_t feature;
    double previous_eigenvalue = INFINITY;
    double previous_cumulative = 0.0;
    double orthonormal_tolerance =
        1024.0 * DBL_EPSILON * (double)GS_PCA_INPUT_DIM;
    size_t zeros = 0;

    if (model == NULL || zero_eigenvalue_count == NULL ||
        model->input_dim != GS_PCA_INPUT_DIM ||
        model->component_count != GS_PCA_INPUT_DIM ||
        model->mean == NULL ||
        model->components == NULL ||
        model->eigenvalues == NULL ||
        model->explained_variance_ratio == NULL ||
        model->cumulative_explained_variance_ratio == NULL ||
        !isfinite(model->total_variance) ||
        model->total_variance <= 0.0 ||
        model->components_90 == 0 ||
        model->components_90 > model->components_95 ||
        model->components_95 > model->components_99 ||
        model->components_99 > GS_PCA_INPUT_DIM) {

        return 1;
    }

    for (feature = 0; feature < GS_PCA_INPUT_DIM; feature++) {
        if (!isfinite(model->mean[feature])) {
            return 1;
        }
    }

    for (component = 0; component < GS_PCA_INPUT_DIM; component++) {
        double eigenvalue = model->eigenvalues[component];
        double ratio = model->explained_variance_ratio[component];
        double cumulative =
            model->cumulative_explained_variance_ratio[component];
        size_t maximum_feature = 0;
        double maximum_abs =
            fabs(model->components[component * GS_PCA_INPUT_DIM]);

        if (!isfinite(eigenvalue) ||
            eigenvalue < 0.0 ||
            eigenvalue > previous_eigenvalue ||
            !isfinite(ratio) ||
            ratio < 0.0 ||
            ratio > 1.0 ||
            !isfinite(cumulative) ||
            cumulative < previous_cumulative ||
            cumulative > 1.0 + 1.0e-12) {

            return 1;
        }
        if (eigenvalue == 0.0) {
            zeros++;
        }
        previous_eigenvalue = eigenvalue;
        previous_cumulative = cumulative;

        for (feature = 0; feature < GS_PCA_INPUT_DIM; feature++) {
            double loading =
                model->components[
                    component * GS_PCA_INPUT_DIM + feature
                ];
            double loading_abs;

            if (!isfinite(loading)) {
                return 1;
            }
            loading_abs = fabs(loading);
            if (loading_abs > maximum_abs) {
                maximum_abs = loading_abs;
                maximum_feature = feature;
            }
        }
        if (!(maximum_abs > 0.0) ||
            model->components[
                component * GS_PCA_INPUT_DIM + maximum_feature
            ] <= 0.0) {

            return 1;
        }
    }

    if (fabs(previous_cumulative - 1.0) > 1.0e-12) {
        return 1;
    }

    for (component = 0; component < GS_PCA_INPUT_DIM; component++) {
        size_t other_component;

        for (other_component = 0;
             other_component < GS_PCA_INPUT_DIM;
             other_component++) {

            double dot = 0.0;
            double expected =
                component == other_component ? 1.0 : 0.0;

            for (feature = 0; feature < GS_PCA_INPUT_DIM; feature++) {
                dot +=
                    model->components[
                        component * GS_PCA_INPUT_DIM + feature
                    ] *
                    model->components[
                        other_component * GS_PCA_INPUT_DIM + feature
                    ];
            }
            if (!isfinite(dot) ||
                fabs(dot - expected) > orthonormal_tolerance) {

                return 1;
            }
        }
    }

    if (model->cumulative_explained_variance_ratio[
            model->components_90 - 1
        ] < 0.90 ||
        (model->components_90 > 1 &&
         model->cumulative_explained_variance_ratio[
             model->components_90 - 2
         ] >= 0.90) ||
        model->cumulative_explained_variance_ratio[
            model->components_95 - 1
        ] < 0.95 ||
        (model->components_95 > 1 &&
         model->cumulative_explained_variance_ratio[
             model->components_95 - 2
         ] >= 0.95) ||
        model->cumulative_explained_variance_ratio[
            model->components_99 - 1
        ] < 0.99 ||
        (model->components_99 > 1 &&
         model->cumulative_explained_variance_ratio[
             model->components_99 - 2
         ] >= 0.99)) {

        return 1;
    }

    *zero_eigenvalue_count = zeros;
    return 0;
}


static int gs_snapshot_model(
    const GSBaselinePCA *model,
    double *snapshot,
    size_t snapshot_count,
    GSPCAModelMetadata *metadata
)
{
    size_t matrix_count;
    size_t required;
    size_t offset = 0;

    if (model == NULL || snapshot == NULL || metadata == NULL ||
        gs_checked_product(
            GS_PCA_INPUT_DIM,
            GS_PCA_INPUT_DIM,
            &matrix_count
        ) != 0 ||
        gs_checked_sum(matrix_count, 4u * GS_PCA_INPUT_DIM, &required) != 0 ||
        snapshot_count != required) {

        return 1;
    }

    memcpy(
        snapshot + offset,
        model->mean,
        GS_PCA_INPUT_DIM * sizeof(double)
    );
    offset += GS_PCA_INPUT_DIM;
    memcpy(
        snapshot + offset,
        model->components,
        matrix_count * sizeof(double)
    );
    offset += matrix_count;
    memcpy(
        snapshot + offset,
        model->eigenvalues,
        GS_PCA_INPUT_DIM * sizeof(double)
    );
    offset += GS_PCA_INPUT_DIM;
    memcpy(
        snapshot + offset,
        model->explained_variance_ratio,
        GS_PCA_INPUT_DIM * sizeof(double)
    );
    offset += GS_PCA_INPUT_DIM;
    memcpy(
        snapshot + offset,
        model->cumulative_explained_variance_ratio,
        GS_PCA_INPUT_DIM * sizeof(double)
    );

    metadata->input_dim = model->input_dim;
    metadata->component_count = model->component_count;
    metadata->total_variance = model->total_variance;
    metadata->components_90 = model->components_90;
    metadata->components_95 = model->components_95;
    metadata->components_99 = model->components_99;
    return 0;
}


static int gs_model_matches_snapshot(
    const GSBaselinePCA *model,
    const double *snapshot,
    size_t snapshot_count,
    const GSPCAModelMetadata *metadata
)
{
    size_t matrix_count;
    size_t required;
    size_t offset = 0;

    if (model == NULL || snapshot == NULL || metadata == NULL ||
        gs_checked_product(
            GS_PCA_INPUT_DIM,
            GS_PCA_INPUT_DIM,
            &matrix_count
        ) != 0 ||
        gs_checked_sum(matrix_count, 4u * GS_PCA_INPUT_DIM, &required) != 0 ||
        snapshot_count != required ||
        model->input_dim != metadata->input_dim ||
        model->component_count != metadata->component_count ||
        memcmp(
            &model->total_variance,
            &metadata->total_variance,
            sizeof(double)
        ) != 0 ||
        model->components_90 != metadata->components_90 ||
        model->components_95 != metadata->components_95 ||
        model->components_99 != metadata->components_99) {

        return 0;
    }

    if (memcmp(
            snapshot + offset,
            model->mean,
            GS_PCA_INPUT_DIM * sizeof(double)
        ) != 0) {

        return 0;
    }
    offset += GS_PCA_INPUT_DIM;
    if (memcmp(
            snapshot + offset,
            model->components,
            matrix_count * sizeof(double)
        ) != 0) {

        return 0;
    }
    offset += matrix_count;
    if (memcmp(
            snapshot + offset,
            model->eigenvalues,
            GS_PCA_INPUT_DIM * sizeof(double)
        ) != 0) {

        return 0;
    }
    offset += GS_PCA_INPUT_DIM;
    if (memcmp(
            snapshot + offset,
            model->explained_variance_ratio,
            GS_PCA_INPUT_DIM * sizeof(double)
        ) != 0) {

        return 0;
    }
    offset += GS_PCA_INPUT_DIM;
    return memcmp(
        snapshot + offset,
        model->cumulative_explained_variance_ratio,
        GS_PCA_INPUT_DIM * sizeof(double)
    ) == 0;
}


static int gs_validate_scores(
    const float *scores,
    size_t rows,
    size_t components
)
{
    size_t count;
    size_t index;

    if (scores == NULL || rows == 0 || components == 0 ||
        gs_checked_product(rows, components, &count) != 0) {

        return 1;
    }
    for (index = 0; index < count; index++) {
        if (!isfinite((double)scores[index])) {
            return 1;
        }
    }
    return 0;
}


static int gs_validate_train_score_statistics(
    const float *scores,
    size_t rows,
    size_t components,
    const double *eigenvalues
)
{
    size_t component;

    if (scores == NULL || rows == 0 || components == 0 ||
        eigenvalues == NULL) {

        return 1;
    }

    for (component = 0; component < components; component++) {
        double mean = 0.0;
        double variance = 0.0;
        double mean_tolerance;
        double variance_tolerance;
        size_t row;

        for (row = 0; row < rows; row++) {
            mean += (double)scores[row * components + component];
        }
        mean /= (double)rows;

        for (row = 0; row < rows; row++) {
            double centered =
                (double)scores[row * components + component] - mean;

            variance += centered * centered;
        }
        variance /= (double)rows;

        mean_tolerance =
            GS_PCA_SCORE_MEAN_TOLERANCE *
            (1.0 + sqrt(eigenvalues[component]));
        variance_tolerance =
            GS_PCA_SCORE_VARIANCE_TOLERANCE *
            (1.0 + eigenvalues[component]);

        if (!isfinite(mean) ||
            !isfinite(variance) ||
            fabs(mean) > mean_tolerance ||
            fabs(variance - eigenvalues[component]) >
                variance_tolerance) {

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
    char path[GS_PCA_PATH_LENGTH];

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


static int gs_write_score_prefix(
    const char *directory,
    const char *filename,
    const float *scores,
    size_t rows,
    size_t source_components,
    size_t output_components
)
{
    FILE *file = NULL;
    size_t row;
    int failed = 0;

    if (scores == NULL ||
        output_components == 0 ||
        output_components > source_components ||
        gs_open_new_output(directory, filename, &file) != 0) {

        return 1;
    }

    for (row = 0; row < rows; row++) {
        if (fwrite(
                scores + row * source_components,
                sizeof(float),
                output_components,
                file
            ) != output_components) {

            failed = 1;
            break;
        }
    }
    if (gs_finish_write(file) != 0) {
        failed = 1;
    }
    return failed;
}


static int gs_write_pca_spec(
    const char *directory,
    const char *code_commit
)
{
    FILE *file = NULL;
    int failed = 0;

    if (gs_open_new_output(directory, "pca_spec.txt", &file) != 0) {
        return 1;
    }
    if (fprintf(
            file,
            "format_version=1\n"
            "profile=mini75_pool5_pca\n"
            "source_profile=mini75_pool5_no_pca\n"
            "input_rows_train=6866\n"
            "input_rows_validation=1717\n"
            "input_dim=900\n"
            "input_standardization=train_only\n"
            "pca_fit=train_only\n"
            "validation_transformed_without_refit=true\n"
            "covariance_ddof=0\n"
            "backend=LAPACKE_dsyevd\n"
            "component_order=eigenvalue_descending\n"
            "component_layout=component_major\n"
            "sign_convention=max_abs_loading_positive_lowest_index_tie\n"
            "whitening=false\n"
            "threshold_90=0.90\n"
            "threshold_95=0.95\n"
            "threshold_99=0.99\n"
            "principal_configuration=pca95\n"
            "labels_used=false\n"
            "code_commit=%s\n",
            code_commit
        ) < 0) {

        failed = 1;
    }
    if (gs_finish_write(file) != 0) {
        failed = 1;
    }
    return failed;
}


static int gs_write_thresholds(
    const char *directory,
    const GSBaselinePCA *model
)
{
    FILE *file = NULL;
    int failed = 0;

    if (model == NULL ||
        gs_open_new_output(directory, "thresholds.csv", &file) != 0) {

        return 1;
    }
    if (fprintf(
            file,
            "threshold,component_count,cumulative_variance\n"
            "0.90,%lu,%.17g\n"
            "0.95,%lu,%.17g\n"
            "0.99,%lu,%.17g\n",
            (unsigned long)model->components_90,
            model->cumulative_explained_variance_ratio[
                model->components_90 - 1
            ],
            (unsigned long)model->components_95,
            model->cumulative_explained_variance_ratio[
                model->components_95 - 1
            ],
            (unsigned long)model->components_99,
            model->cumulative_explained_variance_ratio[
                model->components_99 - 1
            ]
        ) < 0) {

        failed = 1;
    }
    if (gs_finish_write(file) != 0) {
        failed = 1;
    }
    return failed;
}


static int gs_write_spectrum(
    const char *directory,
    const GSBaselinePCA *model
)
{
    FILE *file = NULL;
    size_t component;
    int failed = 0;

    if (model == NULL ||
        gs_open_new_output(directory, "spectrum.csv", &file) != 0) {

        return 1;
    }
    if (fprintf(
            file,
            "component_index,eigenvalue,explained_variance_ratio,"
            "cumulative_explained_variance_ratio\n"
        ) < 0) {

        failed = 1;
    }

    for (component = 0;
         !failed && component < model->component_count;
         component++) {

        if (fprintf(
                file,
                "%lu,%.17g,%.17g,%.17g\n",
                (unsigned long)component,
                model->eigenvalues[component],
                model->explained_variance_ratio[component],
                model->cumulative_explained_variance_ratio[component]
            ) < 0) {

            failed = 1;
        }
    }
    if (gs_finish_write(file) != 0) {
        failed = 1;
    }
    return failed;
}


static int gs_collect_artifact_size(
    const char *directory,
    GSPCAArtifactSize *artifact
)
{
    char path[GS_PCA_PATH_LENGTH];

    return artifact == NULL ||
           gs_path_join(
               directory,
               artifact->name,
               path,
               sizeof(path)
           ) != 0 ||
           gs_file_size(path, &artifact->size) != 0;
}


static int gs_format_summary(
    char *buffer,
    size_t buffer_size,
    const char *source_directory,
    const char *code_commit,
    const GSBaselinePCA *model,
    size_t zero_eigenvalue_count,
    double read_seconds,
    double fit_seconds,
    double train_transform_seconds,
    double validation_transform_seconds,
    double write_seconds,
    const GSPCAArtifactSize *artifacts,
    size_t summary_size
)
{
    int written;

    written = snprintf(
        buffer,
        buffer_size,
        "format_version=1\n"
        "profile=mini75_pool5_pca\n"
        "source_directory=%s\n"
        "source_feature_spec=%s/feature_spec.txt\n"
        "source_scaler=%s/scaler.csv\n"
        "source_train_matrix=%s/train_features_std_f32.bin\n"
        "source_validation_matrix=%s/val_features_std_f32.bin\n"
        "source_train_rows=%s/train_rows.csv\n"
        "source_validation_rows=%s/val_rows.csv\n"
        "source_sha256=%s/sha256.txt\n"
        "code_commit=%s\n"
        "backend=LAPACKE_dsyevd\n"
        "toolchain=%s\n"
        "train_rows=6866\n"
        "validation_rows=1717\n"
        "input_dim=900\n"
        "d90=%lu\n"
        "d95=%lu\n"
        "d99=%lu\n"
        "cumulative_variance_90=%.17g\n"
        "cumulative_variance_95=%.17g\n"
        "cumulative_variance_99=%.17g\n"
        "total_variance=%.17g\n"
        "maximum_eigenvalue=%.17g\n"
        "minimum_eigenvalue=%.17g\n"
        "eigenvalues_zero_after_tolerance=%lu\n"
        "pca_fit=train_only\n"
        "validation_transformed_without_refit=true\n"
        "whitening=false\n"
        "labels_used=false\n"
        "read_cpu_seconds_approx=%.9g\n"
        "fit_pca_cpu_seconds_approx=%.9g\n"
        "transform_train_cpu_seconds_approx=%.9g\n"
        "transform_validation_cpu_seconds_approx=%.9g\n"
        "write_cpu_seconds_approx=%.9g\n"
        "pca_spec_txt_bytes=%lu\n"
        "mean_f64_bin_bytes=%lu\n"
        "components_f64_bin_bytes=%lu\n"
        "eigenvalues_f64_bin_bytes=%lu\n"
        "explained_variance_ratio_f64_bin_bytes=%lu\n"
        "cumulative_explained_variance_ratio_f64_bin_bytes=%lu\n"
        "thresholds_csv_bytes=%lu\n"
        "spectrum_csv_bytes=%lu\n"
        "train_scores_pca90_f32_bin_bytes=%lu\n"
        "val_scores_pca90_f32_bin_bytes=%lu\n"
        "train_scores_pca95_f32_bin_bytes=%lu\n"
        "val_scores_pca95_f32_bin_bytes=%lu\n"
        "train_scores_pca99_f32_bin_bytes=%lu\n"
        "val_scores_pca99_f32_bin_bytes=%lu\n"
        "summary_txt_bytes=%lu\n",
        source_directory,
        source_directory,
        source_directory,
        source_directory,
        source_directory,
        source_directory,
        source_directory,
        source_directory,
        code_commit,
        __VERSION__,
        (unsigned long)model->components_90,
        (unsigned long)model->components_95,
        (unsigned long)model->components_99,
        model->cumulative_explained_variance_ratio[
            model->components_90 - 1
        ],
        model->cumulative_explained_variance_ratio[
            model->components_95 - 1
        ],
        model->cumulative_explained_variance_ratio[
            model->components_99 - 1
        ],
        model->total_variance,
        model->eigenvalues[0],
        model->eigenvalues[model->component_count - 1],
        (unsigned long)zero_eigenvalue_count,
        read_seconds,
        fit_seconds,
        train_transform_seconds,
        validation_transform_seconds,
        write_seconds,
        (unsigned long)artifacts[0].size,
        (unsigned long)artifacts[1].size,
        (unsigned long)artifacts[2].size,
        (unsigned long)artifacts[3].size,
        (unsigned long)artifacts[4].size,
        (unsigned long)artifacts[5].size,
        (unsigned long)artifacts[6].size,
        (unsigned long)artifacts[7].size,
        (unsigned long)artifacts[8].size,
        (unsigned long)artifacts[9].size,
        (unsigned long)artifacts[10].size,
        (unsigned long)artifacts[11].size,
        (unsigned long)artifacts[12].size,
        (unsigned long)artifacts[13].size,
        (unsigned long)summary_size
    );

    return written < 0 || (size_t)written >= buffer_size ? -1 : written;
}


static int gs_write_summary(
    const char *directory,
    const char *source_directory,
    const char *code_commit,
    const GSBaselinePCA *model,
    size_t zero_eigenvalue_count,
    double read_seconds,
    double fit_seconds,
    double train_transform_seconds,
    double validation_transform_seconds,
    double write_seconds,
    GSPCAArtifactSize *artifacts
)
{
    char *buffer;
    const size_t buffer_size = 32768;
    size_t expected_size = 0;
    int written = -1;
    int iteration;
    FILE *file = NULL;
    int failed = 0;

    buffer = (char *)malloc(buffer_size);
    if (buffer == NULL) {
        return 1;
    }

    for (iteration = 0; iteration < 8; iteration++) {
        written = gs_format_summary(
            buffer,
            buffer_size,
            source_directory,
            code_commit,
            model,
            zero_eigenvalue_count,
            read_seconds,
            fit_seconds,
            train_transform_seconds,
            validation_transform_seconds,
            write_seconds,
            artifacts,
            expected_size
        );
        if (written < 0) {
            free(buffer);
            return 1;
        }
        if ((size_t)written == expected_size) {
            break;
        }
        expected_size = (size_t)written;
    }
    if (written < 0 || (size_t)written != expected_size ||
        gs_open_new_output(directory, "summary.txt", &file) != 0) {

        free(buffer);
        return 1;
    }

    if (fwrite(buffer, 1, expected_size, file) != expected_size) {
        failed = 1;
    }
    if (gs_finish_write(file) != 0) {
        failed = 1;
    }
    free(buffer);
    return failed;
}


static void gs_print_usage(const char *program)
{
    printf(
        "Uso: %s <source_no_pca_directory> "
        "<output_partial_directory> <code_commit>\n"
        "Costruisce gli artefatti PCA mini75 con fit esclusivamente train.\n"
        "L'output deve essere una nuova directory con suffisso .partial.\n",
        program
    );
}


int main(int argc, char **argv)
{
    GSBaselinePCA model = {0};
    GSPCAModelMetadata metadata;
    GSPCAArtifactSize artifacts[GS_PCA_ARTIFACT_COUNT] = {
        {"pca_spec.txt", 0},
        {"mean_f64.bin", 0},
        {"components_f64.bin", 0},
        {"eigenvalues_f64.bin", 0},
        {"explained_variance_ratio_f64.bin", 0},
        {"cumulative_explained_variance_ratio_f64.bin", 0},
        {"thresholds.csv", 0},
        {"spectrum.csv", 0},
        {"train_scores_pca90_f32.bin", 0},
        {"val_scores_pca90_f32.bin", 0},
        {"train_scores_pca95_f32.bin", 0},
        {"val_scores_pca95_f32.bin", 0},
        {"train_scores_pca99_f32.bin", 0},
        {"val_scores_pca99_f32.bin", 0},
        {"summary.txt", 0}
    };
    float *train = NULL;
    float *validation = NULL;
    float *train_scores = NULL;
    float *validation_scores = NULL;
    double *model_snapshot = NULL;
    size_t train_value_count;
    size_t validation_value_count;
    size_t train_bytes;
    size_t validation_bytes;
    size_t matrix_count;
    size_t snapshot_count;
    size_t snapshot_bytes;
    size_t train_score_count;
    size_t validation_score_count;
    size_t zero_eigenvalue_count = 0;
    char train_path[GS_PCA_PATH_LENGTH];
    char validation_path[GS_PCA_PATH_LENGTH];
    char final_directory[GS_PCA_PATH_LENGTH];
    clock_t started;
    double read_seconds;
    double fit_seconds;
    double train_transform_seconds;
    double validation_transform_seconds;
    double write_seconds;
    size_t artifact;
    int rc = EXIT_FAILURE;

    if (argc == 2 && strcmp(argv[1], "--help") == 0) {
        gs_print_usage(argv[0]);
        return EXIT_SUCCESS;
    }
    if (argc != 4) {
        gs_print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    if (!gs_platform_is_valid()) {
        fprintf(stderr, "Errore: piattaforma binaria non compatibile.\n");
        return EXIT_FAILURE;
    }
    if (!gs_metadata_value_is_valid(argv[1]) ||
        !gs_metadata_value_is_valid(argv[2]) ||
        !gs_commit_is_valid(argv[3]) ||
        gs_validate_output_directory(
            argv[2],
            final_directory,
            sizeof(final_directory)
        ) != 0 ||
        gs_checked_product(
            GS_PCA_TRAIN_ROWS,
            GS_PCA_INPUT_DIM,
            &train_value_count
        ) != 0 ||
        gs_checked_product(
            GS_PCA_VALIDATION_ROWS,
            GS_PCA_INPUT_DIM,
            &validation_value_count
        ) != 0 ||
        gs_checked_product(
            train_value_count,
            sizeof(float),
            &train_bytes
        ) != 0 ||
        gs_checked_product(
            validation_value_count,
            sizeof(float),
            &validation_bytes
        ) != 0 ||
        gs_checked_product(
            GS_PCA_INPUT_DIM,
            GS_PCA_INPUT_DIM,
            &matrix_count
        ) != 0 ||
        gs_checked_sum(
            matrix_count,
            4u * GS_PCA_INPUT_DIM,
            &snapshot_count
        ) != 0 ||
        gs_checked_product(
            snapshot_count,
            sizeof(double),
            &snapshot_bytes
        ) != 0) {

        fprintf(stderr, "Errore: argomenti o dimensioni non validi.\n");
        return EXIT_FAILURE;
    }

    started = clock();
    if (gs_validate_source_files(argv[1], train_bytes, validation_bytes) != 0 ||
        gs_verify_source_hashes(argv[1]) != 0 ||
        gs_path_join(
            argv[1],
            "train_features_std_f32.bin",
            train_path,
            sizeof(train_path)
        ) != 0 ||
        gs_path_join(
            argv[1],
            "val_features_std_f32.bin",
            validation_path,
            sizeof(validation_path)
        ) != 0) {

        fprintf(stderr, "Errore: contratto o hash degli input non valido.\n");
        goto cleanup;
    }

    train = (float *)malloc(train_bytes);
    validation = (float *)malloc(validation_bytes);
    model_snapshot = (double *)malloc(snapshot_bytes);
    if (train == NULL || validation == NULL || model_snapshot == NULL ||
        gs_read_float_matrix(
            train_path,
            train,
            train_value_count,
            train_bytes
        ) != 0 ||
        gs_read_float_matrix(
            validation_path,
            validation,
            validation_value_count,
            validation_bytes
        ) != 0) {

        fprintf(stderr, "Errore: lettura delle matrici standardizzate fallita.\n");
        goto cleanup;
    }
    read_seconds = (double)(clock() - started) / (double)CLOCKS_PER_SEC;

    started = clock();
    if (gs_baseline_pca_fit(
            train,
            GS_PCA_TRAIN_ROWS,
            GS_PCA_INPUT_DIM,
            &model
        ) != GS_BASELINE_PCA_SUCCESS ||
        gs_model_validate(&model, &zero_eigenvalue_count) != 0 ||
        gs_snapshot_model(
            &model,
            model_snapshot,
            snapshot_count,
            &metadata
        ) != 0) {

        fprintf(stderr, "Errore: fit o validazione PCA train fallita.\n");
        goto cleanup;
    }
    fit_seconds = (double)(clock() - started) / (double)CLOCKS_PER_SEC;

    if (gs_checked_product(
            GS_PCA_TRAIN_ROWS,
            model.components_99,
            &train_score_count
        ) != 0 ||
        gs_checked_product(
            GS_PCA_VALIDATION_ROWS,
            model.components_99,
            &validation_score_count
        ) != 0 ||
        train_score_count > SIZE_MAX / sizeof(float) ||
        validation_score_count > SIZE_MAX / sizeof(float)) {

        fprintf(stderr, "Errore: overflow nelle dimensioni degli score.\n");
        goto cleanup;
    }

    train_scores =
        (float *)malloc(train_score_count * sizeof(float));
    validation_scores =
        (float *)malloc(validation_score_count * sizeof(float));
    if (train_scores == NULL || validation_scores == NULL) {
        fprintf(stderr, "Errore: allocazione score fallita.\n");
        goto cleanup;
    }

    started = clock();
    if (gs_baseline_pca_transform(
            &model,
            train,
            GS_PCA_TRAIN_ROWS,
            GS_PCA_INPUT_DIM,
            model.components_99,
            train_scores,
            train_score_count
        ) != GS_BASELINE_PCA_SUCCESS) {

        fprintf(stderr, "Errore: transform train fallita.\n");
        goto cleanup;
    }
    train_transform_seconds =
        (double)(clock() - started) / (double)CLOCKS_PER_SEC;

    started = clock();
    if (gs_baseline_pca_transform(
            &model,
            validation,
            GS_PCA_VALIDATION_ROWS,
            GS_PCA_INPUT_DIM,
            model.components_99,
            validation_scores,
            validation_score_count
        ) != GS_BASELINE_PCA_SUCCESS) {

        fprintf(stderr, "Errore: transform validation fallita.\n");
        goto cleanup;
    }
    validation_transform_seconds =
        (double)(clock() - started) / (double)CLOCKS_PER_SEC;

    if (!gs_model_matches_snapshot(
            &model,
            model_snapshot,
            snapshot_count,
            &metadata
        ) ||
        gs_validate_scores(
            train_scores,
            GS_PCA_TRAIN_ROWS,
            model.components_99
        ) != 0 ||
        gs_validate_scores(
            validation_scores,
            GS_PCA_VALIDATION_ROWS,
            model.components_99
        ) != 0 ||
        gs_validate_train_score_statistics(
            train_scores,
            GS_PCA_TRAIN_ROWS,
            model.components_99,
            model.eigenvalues
        ) != 0) {

        fprintf(stderr, "Errore: validazione numerica degli score fallita.\n");
        goto cleanup;
    }

    started = clock();
    if (gs_create_partial_directory(argv[2], final_directory) != 0 ||
        gs_write_pca_spec(argv[2], argv[3]) != 0 ||
        gs_write_binary(
            argv[2],
            "mean_f64.bin",
            model.mean,
            sizeof(double),
            GS_PCA_INPUT_DIM
        ) != 0 ||
        gs_write_binary(
            argv[2],
            "components_f64.bin",
            model.components,
            sizeof(double),
            matrix_count
        ) != 0 ||
        gs_write_binary(
            argv[2],
            "eigenvalues_f64.bin",
            model.eigenvalues,
            sizeof(double),
            GS_PCA_INPUT_DIM
        ) != 0 ||
        gs_write_binary(
            argv[2],
            "explained_variance_ratio_f64.bin",
            model.explained_variance_ratio,
            sizeof(double),
            GS_PCA_INPUT_DIM
        ) != 0 ||
        gs_write_binary(
            argv[2],
            "cumulative_explained_variance_ratio_f64.bin",
            model.cumulative_explained_variance_ratio,
            sizeof(double),
            GS_PCA_INPUT_DIM
        ) != 0 ||
        gs_write_thresholds(argv[2], &model) != 0 ||
        gs_write_spectrum(argv[2], &model) != 0 ||
        gs_write_score_prefix(
            argv[2],
            "train_scores_pca90_f32.bin",
            train_scores,
            GS_PCA_TRAIN_ROWS,
            model.components_99,
            model.components_90
        ) != 0 ||
        gs_write_score_prefix(
            argv[2],
            "val_scores_pca90_f32.bin",
            validation_scores,
            GS_PCA_VALIDATION_ROWS,
            model.components_99,
            model.components_90
        ) != 0 ||
        gs_write_score_prefix(
            argv[2],
            "train_scores_pca95_f32.bin",
            train_scores,
            GS_PCA_TRAIN_ROWS,
            model.components_99,
            model.components_95
        ) != 0 ||
        gs_write_score_prefix(
            argv[2],
            "val_scores_pca95_f32.bin",
            validation_scores,
            GS_PCA_VALIDATION_ROWS,
            model.components_99,
            model.components_95
        ) != 0 ||
        gs_write_score_prefix(
            argv[2],
            "train_scores_pca99_f32.bin",
            train_scores,
            GS_PCA_TRAIN_ROWS,
            model.components_99,
            model.components_99
        ) != 0 ||
        gs_write_score_prefix(
            argv[2],
            "val_scores_pca99_f32.bin",
            validation_scores,
            GS_PCA_VALIDATION_ROWS,
            model.components_99,
            model.components_99
        ) != 0) {

        fprintf(stderr, "Errore: scrittura artefatti PCA fallita.\n");
        goto cleanup;
    }

    for (artifact = 0;
         artifact + 1 < GS_PCA_ARTIFACT_COUNT;
         artifact++) {

        if (gs_collect_artifact_size(argv[2], &artifacts[artifact]) != 0) {
            fprintf(stderr, "Errore: controllo dimensioni output fallito.\n");
            goto cleanup;
        }
    }
    write_seconds =
        (double)(clock() - started) / (double)CLOCKS_PER_SEC;

    if (gs_write_summary(
            argv[2],
            argv[1],
            argv[3],
            &model,
            zero_eigenvalue_count,
            read_seconds,
            fit_seconds,
            train_transform_seconds,
            validation_transform_seconds,
            write_seconds,
            artifacts
        ) != 0 ||
        gs_collect_artifact_size(
            argv[2],
            &artifacts[GS_PCA_ARTIFACT_COUNT - 1]
        ) != 0) {

        fprintf(stderr, "Errore: scrittura summary fallita.\n");
        goto cleanup;
    }

    printf("[PASS] source_files_and_sha256\n");
    printf("[PASS] train_only_pca_fit\n");
    printf("[PASS] train_and_validation_transform\n");
    printf("[PASS] model_byte_invariance\n");
    printf("[PASS] numeric_validation\n");
    printf("[PASS] artifacts_written\n");
    printf("D90=%lu\n", (unsigned long)model.components_90);
    printf("D95=%lu\n", (unsigned long)model.components_95);
    printf("D99=%lu\n", (unsigned long)model.components_99);
    printf("PCA ARTIFACT BUILD COMPLETED\n");
    rc = EXIT_SUCCESS;

cleanup:
    free(model_snapshot);
    free(validation_scores);
    free(train_scores);
    free(validation);
    free(train);
    gs_baseline_pca_free(&model);
    return rc;
}
