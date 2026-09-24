#include "baseline_kmeans.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>


enum {
    GS_SMOKE_TRAIN_ROWS = 96,
    GS_SMOKE_VALIDATION_ROWS = 24,
    GS_SMOKE_FEATURE_COUNT = 1024,
    GS_SMOKE_K = 6,
    GS_SMOKE_SEED = 123,
    GS_SMOKE_MAX_ITERATIONS = 100,
    GS_SMOKE_SHA256_LENGTH = 64
};


typedef struct {
    uint32_t state[8];
    uint64_t completed_bits;
    unsigned char block[64];
    size_t block_size;
} GSSha256;


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
    char output[GS_SMOKE_SHA256_LENGTH + 1]
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
    output[GS_SMOKE_SHA256_LENGTH] = '\0';
    return 0;
}


static int gs_read_prefix(
    const char *path,
    size_t expected_file_size,
    float *values,
    size_t value_count
)
{
    FILE *file;
    size_t actual_size;
    size_t index;
    int failed = 0;

    if (path == NULL || values == NULL ||
        gs_file_size(path, &actual_size) != 0 ||
        actual_size != expected_file_size) {
        return 1;
    }

    file = fopen(path, "rb");
    if (file == NULL) {
        return 1;
    }
    if (fread(values, sizeof(float), value_count, file) != value_count ||
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
        if (!isfinite(values[index])) {
            return 1;
        }
    }
    return 0;
}


static int gs_result_contract_is_valid(
    const GSBaselineKMeansResult *result
)
{
    size_t centroid_count;
    size_t row;
    size_t cluster;
    size_t total = 0u;

    if (result == NULL ||
        result->rows != GS_SMOKE_TRAIN_ROWS ||
        result->cols != GS_SMOKE_FEATURE_COUNT ||
        result->k != GS_SMOKE_K ||
        result->seed != GS_SMOKE_SEED ||
        result->iterations == 0u ||
        result->iterations > GS_SMOKE_MAX_ITERATIONS ||
        result->converged != 1 ||
        result->centroids == NULL ||
        result->assignments == NULL ||
        result->cluster_sizes == NULL ||
        result->initial_centroid_indices == NULL ||
        !isfinite(result->initial_inertia) ||
        result->initial_inertia < 0.0 ||
        !isfinite(result->inertia) ||
        result->inertia < 0.0 ||
        gs_checked_product(
            result->k, result->cols, &centroid_count
        ) != 0) {
        return 0;
    }

    for (row = 0; row < centroid_count; row++) {
        if (!isfinite(result->centroids[row])) {
            return 0;
        }
    }
    for (row = 0; row < result->rows; row++) {
        if (result->assignments[row] >= result->k) {
            return 0;
        }
    }
    for (cluster = 0; cluster < result->k; cluster++) {
        if (total > SIZE_MAX - result->cluster_sizes[cluster] ||
            result->initial_centroid_indices[cluster] >= result->rows) {
            return 0;
        }
        total += result->cluster_sizes[cluster];
    }
    return total == result->rows;
}


int main(int argc, char **argv)
{
    size_t train_values;
    size_t validation_values;
    size_t train_bytes;
    size_t validation_bytes;
    size_t centroid_values;
    size_t centroid_bytes;
    size_t train_assignment_bytes;
    size_t k_size_t_bytes;
    size_t validation_assignment_bytes;
    size_t validation_distance_bytes;
    float *train = NULL;
    float *validation = NULL;
    float *train_copy = NULL;
    float *validation_copy = NULL;
    double *centroid_copy = NULL;
    size_t *validation_assignments = NULL;
    double *validation_distances = NULL;
    GSBaselineKMeansResult first;
    GSBaselineKMeansResult second;
    char train_hash_before[GS_SMOKE_SHA256_LENGTH + 1];
    char train_hash_after[GS_SMOKE_SHA256_LENGTH + 1];
    char validation_hash_before[GS_SMOKE_SHA256_LENGTH + 1];
    char validation_hash_after[GS_SMOKE_SHA256_LENGTH + 1];
    size_t row;
    int centroids_equal = 0;
    int assignments_equal = 0;
    int cluster_sizes_equal = 0;
    int initial_indices_equal = 0;
    int iterations_equal = 0;
    int inertia_equal = 0;
    int validation_centroids_unchanged = 0;
    int all_values_finite = 0;
    int rc = 1;

    memset(&first, 0, sizeof(first));
    memset(&second, 0, sizeof(second));

    if (argc != 3 || !gs_platform_is_valid() ||
        gs_checked_product(
            GS_SMOKE_TRAIN_ROWS,
            GS_SMOKE_FEATURE_COUNT,
            &train_values
        ) != 0 ||
        gs_checked_product(
            GS_SMOKE_VALIDATION_ROWS,
            GS_SMOKE_FEATURE_COUNT,
            &validation_values
        ) != 0 ||
        gs_checked_product(
            train_values, sizeof(float), &train_bytes
        ) != 0 ||
        gs_checked_product(
            validation_values, sizeof(float), &validation_bytes
        ) != 0 ||
        gs_checked_product(
            GS_SMOKE_K, GS_SMOKE_FEATURE_COUNT, &centroid_values
        ) != 0 ||
        gs_checked_product(
            centroid_values, sizeof(double), &centroid_bytes
        ) != 0 ||
        gs_checked_product(
            GS_SMOKE_TRAIN_ROWS,
            sizeof(size_t),
            &train_assignment_bytes
        ) != 0 ||
        gs_checked_product(
            GS_SMOKE_K,
            sizeof(size_t),
            &k_size_t_bytes
        ) != 0 ||
        gs_checked_product(
            GS_SMOKE_VALIDATION_ROWS,
            sizeof(size_t),
            &validation_assignment_bytes
        ) != 0 ||
        gs_checked_product(
            GS_SMOKE_VALIDATION_ROWS,
            sizeof(double),
            &validation_distance_bytes
        ) != 0) {

        fprintf(stderr, "Errore: argomenti o dimensioni smoke non validi.\n");
        return 1;
    }

    train = (float *)malloc(train_bytes);
    validation = (float *)malloc(validation_bytes);
    train_copy = (float *)malloc(train_bytes);
    validation_copy = (float *)malloc(validation_bytes);
    centroid_copy = (double *)malloc(centroid_bytes);
    validation_assignments =
        (size_t *)malloc(validation_assignment_bytes);
    validation_distances =
        (double *)malloc(validation_distance_bytes);

    if (train == NULL || validation == NULL ||
        train_copy == NULL || validation_copy == NULL ||
        centroid_copy == NULL || validation_assignments == NULL ||
        validation_distances == NULL ||
        gs_sha256_file(argv[1], train_hash_before) != 0 ||
        gs_sha256_file(argv[2], validation_hash_before) != 0 ||
        gs_read_prefix(
            argv[1], 28123136u, train, train_values
        ) != 0 ||
        gs_read_prefix(
            argv[2], 7032832u, validation, validation_values
        ) != 0) {

        fprintf(stderr, "Errore: caricamento smoke fallito.\n");
        goto cleanup;
    }

    memcpy(train_copy, train, train_bytes);
    memcpy(validation_copy, validation, validation_bytes);

    if (gs_baseline_kmeans_fit(
            train,
            GS_SMOKE_TRAIN_ROWS,
            GS_SMOKE_FEATURE_COUNT,
            GS_SMOKE_K,
            GS_SMOKE_SEED,
            GS_SMOKE_MAX_ITERATIONS,
            &first
        ) != GS_BASELINE_KMEANS_SUCCESS ||
        gs_baseline_kmeans_fit(
            train,
            GS_SMOKE_TRAIN_ROWS,
            GS_SMOKE_FEATURE_COUNT,
            GS_SMOKE_K,
            GS_SMOKE_SEED,
            GS_SMOKE_MAX_ITERATIONS,
            &second
        ) != GS_BASELINE_KMEANS_SUCCESS ||
        !gs_result_contract_is_valid(&first) ||
        !gs_result_contract_is_valid(&second)) {

        fprintf(stderr, "Errore: fit smoke non valido.\n");
        goto cleanup;
    }

    centroids_equal =
        memcmp(
            first.centroids, second.centroids, centroid_bytes
        ) == 0;
    assignments_equal =
        memcmp(
            first.assignments,
            second.assignments,
            train_assignment_bytes
        ) == 0;
    cluster_sizes_equal =
        memcmp(
            first.cluster_sizes,
            second.cluster_sizes,
            k_size_t_bytes
        ) == 0;
    initial_indices_equal =
        memcmp(
            first.initial_centroid_indices,
            second.initial_centroid_indices,
            k_size_t_bytes
        ) == 0;
    iterations_equal =
        first.iterations == second.iterations &&
        first.converged == second.converged &&
        first.empty_cluster_reseeds == second.empty_cluster_reseeds;
    inertia_equal =
        memcmp(
            &first.initial_inertia,
            &second.initial_inertia,
            sizeof(double)
        ) == 0 &&
        memcmp(
            &first.inertia, &second.inertia, sizeof(double)
        ) == 0;

    if (!centroids_equal || !assignments_equal ||
        !cluster_sizes_equal || !initial_indices_equal ||
        !iterations_equal || !inertia_equal ||
        memcmp(train, train_copy, train_bytes) != 0 ||
        memcmp(validation, validation_copy, validation_bytes) != 0) {

        fprintf(stderr, "Errore: determinismo o immutabilita' smoke falliti.\n");
        goto cleanup;
    }

    memcpy(centroid_copy, first.centroids, centroid_bytes);
    if (gs_baseline_kmeans_assign(
            validation,
            GS_SMOKE_VALIDATION_ROWS,
            GS_SMOKE_FEATURE_COUNT,
            first.centroids,
            GS_SMOKE_K,
            validation_assignments,
            validation_distances
        ) != GS_BASELINE_KMEANS_SUCCESS) {

        fprintf(stderr, "Errore: assegnazione validation smoke fallita.\n");
        goto cleanup;
    }

    validation_centroids_unchanged =
        memcmp(centroid_copy, first.centroids, centroid_bytes) == 0;
    all_values_finite = 1;
    for (row = 0; row < GS_SMOKE_VALIDATION_ROWS; row++) {
        if (validation_assignments[row] >= GS_SMOKE_K ||
            !isfinite(validation_distances[row]) ||
            validation_distances[row] < 0.0) {
            all_values_finite = 0;
            break;
        }
    }

    if (!validation_centroids_unchanged || !all_values_finite ||
        memcmp(train, train_copy, train_bytes) != 0 ||
        memcmp(validation, validation_copy, validation_bytes) != 0 ||
        gs_sha256_file(argv[1], train_hash_after) != 0 ||
        gs_sha256_file(argv[2], validation_hash_after) != 0 ||
        strcmp(train_hash_before, train_hash_after) != 0 ||
        strcmp(validation_hash_before, validation_hash_after) != 0) {

        fprintf(stderr, "Errore: contratto validation o SHA smoke fallito.\n");
        goto cleanup;
    }

    printf("MINI112_SMOKE_REPEAT_CENTROIDS_BITWISE=true\n");
    printf("MINI112_SMOKE_REPEAT_ASSIGNMENTS_BITWISE=true\n");
    printf("MINI112_SMOKE_REPEAT_CLUSTER_SIZES_BITWISE=true\n");
    printf("MINI112_SMOKE_REPEAT_INITIAL_INDICES_BITWISE=true\n");
    printf("MINI112_SMOKE_REPEAT_ITERATIONS_EQUAL=true\n");
    printf("MINI112_SMOKE_REPEAT_INERTIA_BITWISE=true\n");
    printf("MINI112_SMOKE_VALIDATION_CENTROIDS_UNCHANGED=true\n");
    printf("MINI112_SMOKE_ALL_VALUES_FINITE=true\n");
    printf("ALL MINI112 KMEANS SMOKE TESTS PASSED\n");
    rc = 0;

cleanup:
    gs_baseline_kmeans_result_free(&second);
    gs_baseline_kmeans_result_free(&first);
    free(validation_distances);
    free(validation_assignments);
    free(centroid_copy);
    free(validation_copy);
    free(train_copy);
    free(validation);
    free(train);
    return rc;
}
