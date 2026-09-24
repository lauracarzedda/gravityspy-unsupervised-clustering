#include "vae_c_iic_mini224_central.h"

#include "vae_c_mu_contract.h"
#include "vae_c_sha256.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#define test_mkdir(path) _mkdir(path)
#else
#include <sys/stat.h>
#define test_mkdir(path) mkdir((path), 0700)
#endif

#define TEST_PATH_CAP 4096u
#define TEST_TEXT_CAP 65536u

static const char artifact_relative[] =
    "sakai_multires_c/results/vae_c/embeddings/"
    "mini224_bridge_v1_subset1000_best_epoch99_posterior_mu_central_v1";
static const char *const names[6] = {
    "embedding_spec.txt", "sha256.txt", "train_mu_f32.bin",
    "train_rows.csv", "validation_mu_f32.bin", "validation_rows.csv"
};
static const char *const expected_hashes[6] = {
    "d3c67e9efcfedcb5b620893cfb07ec225505653e42267986dcb030bcbcb89ef6",
    "aa55332307b89ec902ffd217467cdd58870728f382ad8956045cf91878ed8c95",
    "f494b232cb2fbf9ca4fb25df80c47e5f80963ac1d7448afd807361409a4ab166",
    "fd5bb3c48f87e98d5ec8ed0e561d5fe077ab96a7effa0c7425a41a222cd4375c",
    "5f6aec1737549a24d20897581b4c6b57b3fae6859e1dc1e972a8b8841ede4617",
    "87a3c7fc715fe0f4a6158474e46f6903d5267a45741d338836e8fecd61df7ed8"
};

static int path_join(const char *base, const char *name,
                     char output[TEST_PATH_CAP])
{
    int size = snprintf(output, TEST_PATH_CAP, "%s/%s", base, name);
    return size >= 0 && (size_t)size < TEST_PATH_CAP;
}

static int canonical_hashes(const char *directory)
{
    char path[TEST_PATH_CAP], digest[65];
    size_t index;
    for (index = 0u; index < 6u; ++index) {
        if (!path_join(directory, names[index], path) ||
            vae_c_mu_file_sha256(path, digest) != VAE_C_MU_OK ||
            strcmp(digest, expected_hashes[index]) != 0) return 0;
    }
    return 1;
}

static int copy_file(const char *source, const char *destination)
{
    unsigned char bytes[8192];
    FILE *in = fopen(source, "rb");
    FILE *out;
    size_t n;
    int ok = 1;
    if (in == NULL) return 0;
    out = fopen(destination, "wb");
    if (out == NULL) { (void)fclose(in); return 0; }
    while ((n = fread(bytes, 1u, sizeof(bytes), in)) != 0u) {
        if (fwrite(bytes, 1u, n, out) != n) { ok = 0; break; }
    }
    if (ferror(in)) ok = 0;
    if (fclose(in) != 0) ok = 0;
    if (fclose(out) != 0) ok = 0;
    return ok;
}

static int reset_fixture(const char *original, const char *fixture)
{
    char src[TEST_PATH_CAP], dst[TEST_PATH_CAP];
    size_t index;
    for (index = 0u; index < 6u; ++index) {
        if (!path_join(original, names[index], src) ||
            !path_join(fixture, names[index], dst) ||
            !copy_file(src, dst)) return 0;
    }
    return 1;
}

static int read_text(const char *path, char text[TEST_TEXT_CAP])
{
    FILE *file = fopen(path, "rb");
    size_t n;
    if (file == NULL) return 0;
    n = fread(text, 1u, TEST_TEXT_CAP - 1u, file);
    if (ferror(file) || !feof(file)) { (void)fclose(file); return 0; }
    if (fclose(file) != 0) return 0;
    text[n] = '\0';
    return 1;
}

static int write_text(const char *path, const char *text)
{
    FILE *file = fopen(path, "wb");
    size_t n = strlen(text);
    int ok;
    if (file == NULL) return 0;
    ok = fwrite(text, 1u, n, file) == n;
    if (fclose(file) != 0) ok = 0;
    return ok;
}

static int replace_once(const char *path, const char *before,
                        const char *after)
{
    char text[TEST_TEXT_CAP], changed[TEST_TEXT_CAP];
    char *at;
    size_t prefix, suffix, after_size;
    if (!read_text(path, text) || (at = strstr(text, before)) == NULL) return 0;
    prefix = (size_t)(at - text);
    suffix = strlen(at + strlen(before));
    after_size = strlen(after);
    if (prefix + after_size + suffix >= sizeof(changed)) return 0;
    memcpy(changed, text, prefix);
    memcpy(changed + prefix, after, after_size);
    memcpy(changed + prefix + after_size,
           at + strlen(before), suffix + 1u);
    return write_text(path, changed);
}

static int refresh_manifest(const char *fixture)
{
    static const size_t order[5] = {0u, 2u, 3u, 4u, 5u};
    char path[TEST_PATH_CAP], manifest_path[TEST_PATH_CAP];
    char hash[65];
    FILE *manifest;
    size_t i;
    if (!path_join(fixture, "sha256.txt", manifest_path)) return 0;
    manifest = fopen(manifest_path, "wb");
    if (manifest == NULL) return 0;
    for (i = 0u; i < 5u; ++i) {
        const char *name = names[order[i]];
        if (!path_join(fixture, name, path) ||
            vae_c_mu_file_sha256(path, hash) != VAE_C_MU_OK ||
            fprintf(manifest, "%s  %s\n", hash, name) < 0) {
            (void)fclose(manifest);
            return 0;
        }
    }
    return fclose(manifest) == 0;
}

static int refresh_spec_hash(const char *fixture, const char *file_name,
                             const char *spec_key)
{
    char path[TEST_PATH_CAP], spec[TEST_PATH_CAP], hash[65];
    char before[256], after[256], text[TEST_TEXT_CAP], *at;
    if (!path_join(fixture, file_name, path) ||
        vae_c_mu_file_sha256(path, hash) != VAE_C_MU_OK ||
        !path_join(fixture, "embedding_spec.txt", spec) ||
        !read_text(spec, text)) return 0;
    if (snprintf(before, sizeof(before), "%s=", spec_key) < 0 ||
        snprintf(after, sizeof(after), "%s=%s", spec_key, hash) < 0 ||
        (at = strstr(text, before)) == NULL) return 0;
    {
        char old_line[256];
        char *end = strchr(at, '\n');
        size_t len;
        if (end == NULL) return 0;
        len = (size_t)(end - at);
        if (len >= sizeof(old_line)) return 0;
        memcpy(old_line, at, len);
        old_line[len] = '\0';
        return replace_once(spec, old_line, after) && refresh_manifest(fixture);
    }
}

static int mutate_matrix(const char *fixture, float value)
{
    char path[TEST_PATH_CAP];
    FILE *file;
    int ok;
    if (!path_join(fixture, "train_mu_f32.bin", path)) return 0;
    file = fopen(path, "r+b");
    if (file == NULL) return 0;
    ok = fwrite(&value, sizeof(value), 1u, file) == 1u;
    if (fclose(file) != 0) ok = 0;
    return ok && refresh_spec_hash(fixture, "train_mu_f32.bin",
                                   "train_matrix_sha256");
}

static int mutate_row(const char *fixture, const char *before,
                      const char *after)
{
    char path[TEST_PATH_CAP];
    if (!path_join(fixture, "train_rows.csv", path) ||
        !replace_once(path, before, after)) return 0;
    return refresh_spec_hash(fixture, "train_rows.csv", "train_rows_sha256");
}

static int expected_reject(const char *root, const char *fixture,
                           VAECIICMini224Central *view, const char *case_name)
{
    char error[512];
    if (vae_c_iic_mini224_central_load(root, fixture, view,
                                        error, sizeof(error)) ||
        view->mu != NULL || view->rows != NULL) {
        fprintf(stderr, "negative case accepted: %s\n", case_name);
        vae_c_iic_mini224_central_free(view);
        return 0;
    }
    printf("REJECTED_%s=true\n", case_name);
    return 1;
}

int main(int argc, char **argv)
{
    VAECIICMini224Central view;
    char original[TEST_PATH_CAP], fixture[TEST_PATH_CAP];
    char error[512], digest[65], path[TEST_PATH_CAP];
    FILE *file;
    size_t index;
    if (argc != 3 || !path_join(argv[1], artifact_relative, original) ||
        strlen(argv[2]) >= sizeof(fixture)) return 1;
    (void)strcpy(fixture, argv[2]);
    vae_c_iic_mini224_central_init(&view);
    if (!canonical_hashes(original) ||
        !vae_c_iic_mini224_central_load(argv[1], NULL, &view,
                                         error, sizeof(error))) {
        fprintf(stderr, "canonical load failed: %s\n", error);
        return 1;
    }
    if (view.row_count != 1000u || view.latent_dim != 64u ||
        strcmp(view.checkpoint_sha256,
               "15a1ecd6dd3c6d11aa0a2ac20ea0b6176d08a18c6f582e7349ed6784ab8528f3") != 0)
        return 1;
    for (index = 0u; index < view.row_count; ++index) {
        const VAECIICMini224CentralRow *row = &view.rows[index];
        const float *mu = vae_c_iic_mini224_central_row_mu(&view, index);
        size_t dim;
        if (row->iic_row_index != index ||
            row->source_split != (index < 800u ?
                VAE_C_IIC_MINI224_SOURCE_TRAIN :
                VAE_C_IIC_MINI224_SOURCE_VALIDATION) ||
            row->split_pos != (index < 800u ? index : index - 800u) ||
            row->event_index != row->manifest_index + 1u ||
            row->event_id[0] == '\0' || mu == NULL) return 1;
        for (dim = 0u; dim < 64u; ++dim) if (!isfinite(mu[dim])) return 1;
    }
    if (view.rows[0].manifest_index != 7663u ||
        view.rows[799].manifest_index != 3677u ||
        view.rows[799].source_split != VAE_C_IIC_MINI224_SOURCE_TRAIN ||
        view.rows[800].manifest_index != 6864u ||
        view.rows[999].manifest_index != 3392u ||
        view.rows[999].source_split != VAE_C_IIC_MINI224_SOURCE_VALIDATION ||
        vae_c_sha256_digest_hex_lower(view.mu, 1000u * 64u * sizeof(float),
                                      digest) != VAE_C_SHA256_OK ||
        strcmp(digest,
               "89d8dcaf1b7d64ebc77f77e5d1b3853bad9f91be901500b8de685c5d7246a240") != 0)
        return 1;
    printf("CENTRAL_MU_SHA256=%s\n", digest);
    vae_c_iic_mini224_central_free(&view);
    if (test_mkdir(fixture) != 0) {
        if (!path_join(fixture, "embedding_spec.txt", path)) return 1;
    }
    if (!reset_fixture(original, fixture) ||
        !mutate_row(fixture, "1,1,2606,2607", "1,1,7663,2607") ||
        !expected_reject(argv[1], fixture, &view, "DUPLICATE_MANIFEST")) return 1;
    if (!reset_fixture(original, fixture) ||
        !mutate_row(fixture, "1,1,2606,2607,H1_AYvIQZztiZ",
                    "1,1,2606,2607,H1_BqniZ6oLFl") ||
        !expected_reject(argv[1], fixture, &view, "DUPLICATE_EVENT_ID")) return 1;
    if (!reset_fixture(original, fixture) ||
        !mutate_row(fixture, "0,0,7663,7664", "1,0,7663,7664") ||
        !expected_reject(argv[1], fixture, &view, "ROW_ORDER")) return 1;
    if (!reset_fixture(original, fixture) ||
        !mutate_row(fixture, "0,0,7663,7664", "0,0,7663,7665") ||
        !expected_reject(argv[1], fixture, &view, "ROW_MAP_MISMATCH")) return 1;
    if (!reset_fixture(original, fixture) ||
        !path_join(fixture, "train_rows.csv", path)) return 1;
    {
        char text[TEST_TEXT_CAP], *end;
        if (!read_text(path, text)) return 1;
        end = strrchr(text, '\n');
        if (end == NULL) return 1;
        *end = '\0';
        end = strrchr(text, '\n');
        if (end == NULL) return 1;
        end[1] = '\0';
        if (!write_text(path, text) ||
            !refresh_spec_hash(fixture, "train_rows.csv", "train_rows_sha256") ||
            !expected_reject(argv[1], fixture, &view, "WRONG_COUNT")) return 1;
    }
    if (!reset_fixture(original, fixture) ||
        !path_join(fixture, "embedding_spec.txt", path) ||
        !replace_once(path, "latent_dim=64", "latent_dim=32") ||
        !refresh_manifest(fixture) ||
        !expected_reject(argv[1], fixture, &view, "WRONG_LATENT_DIM")) return 1;
    if (!reset_fixture(original, fixture) || !mutate_matrix(fixture, NAN) ||
        !expected_reject(argv[1], fixture, &view, "NAN")) return 1;
    if (!reset_fixture(original, fixture) || !mutate_matrix(fixture, INFINITY) ||
        !expected_reject(argv[1], fixture, &view, "INF")) return 1;
    if (!reset_fixture(original, fixture) ||
        !path_join(fixture, "train_mu_f32.bin", path)) return 1;
    file = fopen(path, "r+b");
    if (file == NULL || fputc(0, file) == EOF || fclose(file) != 0 ||
        !expected_reject(argv[1], fixture, &view, "CHECKSUM")) return 1;
    if (!canonical_hashes(original)) return 1;
    printf("MINI224_CENTRAL_TEST_PASS=true\n");
    return 0;
}
