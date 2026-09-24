#include "vae_c_iic_affinity.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REAL_ROWS 8583u
#define REAL_COLS 180u
#define TINY_ROWS 5u
#define TINY_COLS 4u
#define PATH_CAP 1024u

static const float synthetic[TINY_ROWS * TINY_COLS] = {
     0.125f, -0.5f,  1.25f,  0.1f,
    -0.75f,   0.25f, 1.25f,  0.375f,
     0.125f, -0.5f,  1.25f,  0.1f,
     1.5f,    0.2f, -0.75f, -0.625f,
    -1.125f,  0.75f, 0.5f,   1.75f
};
static const size_t real_indices[TINY_ROWS] = {0u, 1u, 2u, 100u, 8582u};

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "affinity check failed at line %d: %s\n", \
                __LINE__, #expr); \
        return 0; \
    } \
} while (0)

static int matrix_valid(const double *a, size_t n)
{
    size_t i, j;
    for (i = 0u; i < n; i++) {
        CHECK(memcmp(&a[i * n + i], &(double){1.0}, sizeof(double)) == 0);
        for (j = 0u; j < n; j++) {
            CHECK(isfinite(a[i * n + j]));
            CHECK(a[i * n + j] > 0.0 && a[i * n + j] <= 1.0);
            CHECK(memcmp(&a[i * n + j], &a[j * n + i], sizeof(double)) == 0);
        }
    }
    return 1;
}

static int replay_and_chunks(const float *h, size_t n, size_t d,
                             double *out)
{
    double replay[TINY_ROWS * TINY_ROWS];
    double chunks[TINY_ROWS * TINY_ROWS];
    size_t i;
    CHECK(n == TINY_ROWS);
    memset(replay, 0, sizeof(replay));
    memset(chunks, 0, sizeof(chunks));
    CHECK(vae_c_iic_affinity_compute(h, n, d, out, n * n) == VAE_C_OP_OK);
    CHECK(vae_c_iic_affinity_compute(h, n, d, replay, n * n) == VAE_C_OP_OK);
    CHECK(memcmp(out, replay, sizeof(replay)) == 0);
    for (i = 0u; i < n; i++) {
        CHECK(vae_c_iic_affinity_compute_rows(h, n, d, i, 1u,
                                              chunks, n * n) == VAE_C_OP_OK);
    }
    CHECK(memcmp(out, chunks, sizeof(chunks)) == 0);
    memset(chunks, 0, sizeof(chunks));
    CHECK(vae_c_iic_affinity_compute_rows(h, n, d, 0u, 2u,
                                          chunks, n * n) == VAE_C_OP_OK);
    CHECK(vae_c_iic_affinity_compute_rows(h, n, d, 2u, 3u,
                                          chunks, n * n) == VAE_C_OP_OK);
    CHECK(memcmp(out, chunks, sizeof(chunks)) == 0);
    CHECK(matrix_valid(out, n));
    return 1;
}

static int analytic_tests(void)
{
    const float identical[2] = {2.5f, 2.5f};
    const float one_d[2] = {0.0f, 1.0f};
    const float triangle[4] = {0.0f, 0.0f, 3.0f, 4.0f};
    const float singleton[3] = {0.0f, 1.0f, -3.0f};
    float h_copy[sizeof(synthetic) / sizeof(float)];
    double a[4];
    double synthetic_a[TINY_ROWS * TINY_ROWS];
    memcpy(h_copy, synthetic, sizeof(synthetic));
    CHECK(vae_c_iic_affinity_compute(identical, 2u, 1u, a, 4u) == VAE_C_OP_OK);
    CHECK(a[1] == 1.0 && matrix_valid(a, 2u));
    CHECK(vae_c_iic_affinity_compute(one_d, 2u, 1u, a, 4u) == VAE_C_OP_OK);
    CHECK(fabs(a[1] - exp(-1.0)) <= 2.0e-16);
    CHECK(vae_c_iic_affinity_compute(triangle, 2u, 2u, a, 4u) == VAE_C_OP_OK);
    CHECK(fabs(a[1] - exp(-5.0)) <= 2.0e-16);
    CHECK(vae_c_iic_affinity_compute(singleton, 1u, 3u, a, 4u) == VAE_C_OP_OK);
    CHECK(a[0] == 1.0);
    CHECK(replay_and_chunks(synthetic, TINY_ROWS, TINY_COLS, synthetic_a));
    CHECK(memcmp(h_copy, synthetic, sizeof(synthetic)) == 0);
    CHECK(synthetic_a[0u * TINY_ROWS + 2u] == 1.0);
    return 1;
}

static int invalid_tests(void)
{
    const float h[2] = {0.0f, 1.0f};
    const float nonfinite[2] = {0.0f, NAN};
    const float infinity[2] = {0.0f, INFINITY};
    const float extreme[2] = {0.0f, 1000.0f};
    double a[4] = {42.0, 42.0, 42.0, 42.0};
    size_t i;
    CHECK(vae_c_iic_affinity_compute(NULL, 2u, 1u, a, 4u) ==
          VAE_C_OP_INVALID_ARGUMENT);
    CHECK(vae_c_iic_affinity_compute(h, 2u, 1u, NULL, 4u) ==
          VAE_C_OP_INVALID_ARGUMENT);
    CHECK(vae_c_iic_affinity_compute(h, 0u, 1u, a, 4u) ==
          VAE_C_OP_INVALID_ARGUMENT);
    CHECK(vae_c_iic_affinity_compute(h, 2u, 0u, a, 4u) ==
          VAE_C_OP_INVALID_ARGUMENT);
    CHECK(vae_c_iic_affinity_compute(h, 2u, 1u, a, 3u) ==
          VAE_C_OP_SHAPE_MISMATCH);
    CHECK(vae_c_iic_affinity_compute(h, SIZE_MAX, 2u, a, 4u) ==
          VAE_C_OP_OVERFLOW);
    CHECK(vae_c_iic_affinity_compute_rows(h, 2u, 1u, 2u, 1u, a, 4u) ==
          VAE_C_OP_INVALID_ARGUMENT);
    CHECK(vae_c_iic_affinity_compute_rows(h, 2u, 1u, 1u, 2u, a, 4u) ==
          VAE_C_OP_INVALID_ARGUMENT);
    CHECK(vae_c_iic_affinity_compute_rows(h, 2u, 1u, 0u, 0u, a, 4u) ==
          VAE_C_OP_INVALID_ARGUMENT);
    CHECK(vae_c_iic_affinity_compute(nonfinite, 2u, 1u, a, 4u) ==
          VAE_C_OP_INVALID_ARGUMENT);
    CHECK(vae_c_iic_affinity_compute(infinity, 2u, 1u, a, 4u) ==
          VAE_C_OP_INVALID_ARGUMENT);
    for (i = 0u; i < 4u; i++) CHECK(a[i] == 42.0);
    CHECK(vae_c_iic_affinity_compute(extreme, 2u, 1u, a, 4u) ==
          VAE_C_OP_OVERFLOW);
    return 1;
}

static int path_join(char *out, size_t capacity,
                     const char *directory, const char *file)
{
    int written = snprintf(out, capacity, "%s/%s", directory, file);
    return written >= 0 && (size_t)written < capacity;
}

static int load_real(const char *directory, float *h)
{
    char payload_path[PATH_CAP], rows_path[PATH_CAP];
    FILE *payload = NULL, *row_map = NULL;
    char line[512];
    size_t row, selected = 0u;
    long expected_bytes = (long)(REAL_ROWS * REAL_COLS * sizeof(float));

    CHECK(path_join(payload_path, sizeof(payload_path), directory, "h_f32.bin"));
    CHECK(path_join(rows_path, sizeof(rows_path), directory, "rows.csv"));
    payload = fopen(payload_path, "rb");
    CHECK(payload != NULL);
    CHECK(fseek(payload, 0L, SEEK_END) == 0);
    CHECK(ftell(payload) == expected_bytes);
    for (row = 0u; row < TINY_ROWS; row++) {
        long offset = (long)(real_indices[row] * REAL_COLS * sizeof(float));
        CHECK(fseek(payload, offset, SEEK_SET) == 0);
        CHECK(fread(h + row * REAL_COLS, sizeof(float), REAL_COLS, payload) ==
              REAL_COLS);
    }
    CHECK(fclose(payload) == 0);
    row_map = fopen(rows_path, "rb");
    CHECK(row_map != NULL);
    CHECK(fgets(line, sizeof(line), row_map) != NULL);
    CHECK(strncmp(line,
        "manifest_index,event_index,event_id,source_split,split_pos",
        sizeof("manifest_index,event_index,event_id,source_split,split_pos") - 1u) == 0);
    for (row = 0u; row < REAL_ROWS; row++) {
        size_t manifest_index, event_index, split_pos;
        char event_id[128], split[32];
        CHECK(fgets(line, sizeof(line), row_map) != NULL);
        CHECK(sscanf(line, "%zu,%zu,%127[^,],%31[^,],%zu",
                     &manifest_index, &event_index, event_id,
                     split, &split_pos) == 5);
        CHECK(manifest_index == row && event_index > 0u &&
              event_id[0] != '\0');
        CHECK(strcmp(split, "train") == 0 ||
              strcmp(split, "validation") == 0);
        if (selected < TINY_ROWS && row == real_indices[selected])
            selected++;
    }
    CHECK(selected == TINY_ROWS);
    CHECK(fgets(line, sizeof(line), row_map) == NULL);
    CHECK(fclose(row_map) == 0);
    for (row = 0u; row < TINY_ROWS * REAL_COLS; row++) CHECK(isfinite(h[row]));
    return 1;
}

static int real_test(const char *directory, int dump)
{
    float h[TINY_ROWS * REAL_COLS];
    float copy[TINY_ROWS * REAL_COLS];
    double a[TINY_ROWS * TINY_ROWS];
    size_t i, d, j;
    CHECK(load_real(directory, h));
    memcpy(copy, h, sizeof(h));
    CHECK(replay_and_chunks(h, TINY_ROWS, REAL_COLS, a));
    CHECK(memcmp(h, copy, sizeof(h)) == 0);
    if (dump) {
        for (i = 0u; i < TINY_ROWS; i++) {
            for (d = 0u; d < REAL_COLS; d++)
                printf("H,%zu,%zu,%a\n", real_indices[i], d,
                       (double)h[i * REAL_COLS + d]);
        }
        for (i = 0u; i < TINY_ROWS; i++) {
            for (j = 0u; j < TINY_ROWS; j++)
                printf("A,%zu,%zu,%.17g\n", i, j, a[i * TINY_ROWS + j]);
        }
    }
    return 1;
}

static int synthetic_dump(void)
{
    double a[TINY_ROWS * TINY_ROWS];
    size_t i, j;
    CHECK(replay_and_chunks(synthetic, TINY_ROWS, TINY_COLS, a));
    for (i = 0u; i < TINY_ROWS; i++) {
        for (j = 0u; j < TINY_ROWS; j++)
            printf("A,%zu,%zu,%.17g\n", i, j, a[i * TINY_ROWS + j]);
    }
    return 1;
}

int main(int argc, char **argv)
{
    if (argc == 1) {
        if (!analytic_tests() || !invalid_tests()) return 1;
        puts("IIC_AFFINITY_SYNTHETIC_PASS");
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--dump-synthetic") == 0)
        return synthetic_dump() ? 0 : 1;
    if (argc == 3 && strcmp(argv[1], "--check-real") == 0)
        return real_test(argv[2], 0) ? 0 : 1;
    if (argc == 3 && strcmp(argv[1], "--dump-real") == 0)
        return real_test(argv[2], 1) ? 0 : 1;
    fprintf(stderr, "usage: %s [--dump-synthetic|--check-real DIR|--dump-real DIR]\n",
            argv[0]);
    return 2;
}
