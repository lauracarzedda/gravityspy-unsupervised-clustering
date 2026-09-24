#include "vae_c_iic_mini112_central.h"

#include "vae_c_mu_contract.h"
#include "vae_c_sha256.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#define central_mkdir(path) _mkdir(path)
#else
#include <sys/stat.h>
#define central_mkdir(path) mkdir((path), 0700)
#endif

#define TEST_PATH_CAP 4096u
#define TEST_SPEC_CAP 8192u

static const char *const test_files[6] = {
    "embedding_spec.txt", "sha256.txt", "train_mu_f32.bin",
    "train_rows.csv", "validation_mu_f32.bin", "validation_rows.csv"
};
static const char test_artifact_relative[] =
    "sakai_multires_c/results/vae_c/embeddings/"
    "mini112_primary_v1_full_best_epoch97_posterior_mu_central_v1";

static int test_path(const char *directory, const char *name,
                     char path[TEST_PATH_CAP])
{
    int written = snprintf(path, TEST_PATH_CAP, "%s/%s", directory, name);
    return written >= 0 && (size_t)written < TEST_PATH_CAP;
}

static int test_copy(const char *source, const char *destination)
{
    unsigned char buffer[8192];
    FILE *input = fopen(source, "rb");
    FILE *output;
    size_t count;
    int ok = 1;
    if (input == NULL) {
        return 0;
    }
    output = fopen(destination, "wb");
    if (output == NULL) {
        (void)fclose(input);
        return 0;
    }
    while ((count = fread(buffer, 1u, sizeof(buffer), input)) != 0u) {
        if (fwrite(buffer, 1u, count, output) != count) {
            ok = 0;
            break;
        }
    }
    if (ferror(input) || fclose(output) != 0) {
        ok = 0;
    }
    (void)fclose(input);
    return ok;
}

static int test_restore_fixture(const char *original, const char *fixture)
{
    char source[TEST_PATH_CAP];
    char destination[TEST_PATH_CAP];
    size_t index;
    for (index = 0u; index < 6u; ++index) {
        if (!test_path(original, test_files[index], source) ||
            !test_path(fixture, test_files[index], destination) ||
            !test_copy(source, destination)) {
            return 0;
        }
    }
    return 1;
}

static int test_replace_spec_value(const char *fixture, const char *key,
                                   const char *new_value)
{
    char path[TEST_PATH_CAP];
    char source[TEST_SPEC_CAP];
    char output[TEST_SPEC_CAP];
    char replacement[256];
    FILE *file;
    size_t bytes;
    char *found;
    char *end;
    int written;
    if (!test_path(fixture, "embedding_spec.txt", path)) {
        return 0;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        return 0;
    }
    bytes = fread(source, 1u, sizeof(source) - 1u, file);
    if (ferror(file) || !feof(file)) {
        (void)fclose(file);
        return 0;
    }
    (void)fclose(file);
    source[bytes] = '\0';
    written = snprintf(replacement, sizeof(replacement), "%s=%s\n", key,
                       new_value);
    if (written < 0 || (size_t)written >= sizeof(replacement)) {
        return 0;
    }
    found = strstr(source, key);
    if (found == NULL || (found != source && found[-1] != '\n') ||
        found[strlen(key)] != '=') {
        return 0;
    }
    end = strchr(found, '\n');
    if (end == NULL) {
        return 0;
    }
    ++end;
    if ((size_t)(found - source) + (size_t)written + strlen(end) >=
        sizeof(output)) {
        return 0;
    }
    memcpy(output, source, (size_t)(found - source));
    memcpy(output + (found - source), replacement, (size_t)written);
    (void)strcpy(output + (found - source) + written, end);
    file = fopen(path, "wb");
    if (file == NULL) {
        return 0;
    }
    bytes = strlen(output);
    return fwrite(output, 1u, bytes, file) == bytes && fclose(file) == 0;
}

static int test_refresh_hash_manifest(const char *fixture)
{
    static const char *const names[5] = {
        "embedding_spec.txt", "train_mu_f32.bin", "train_rows.csv",
        "validation_mu_f32.bin", "validation_rows.csv"
    };
    char path[TEST_PATH_CAP];
    char manifest_path[TEST_PATH_CAP];
    char hashes[5][65];
    FILE *file;
    size_t index;
    for (index = 0u; index < 5u; ++index) {
        if (!test_path(fixture, names[index], path) ||
            vae_c_mu_file_sha256(path, hashes[index]) != VAE_C_MU_OK) {
            return 0;
        }
    }
    if (!test_path(fixture, "sha256.txt", manifest_path)) {
        return 0;
    }
    file = fopen(manifest_path, "wb");
    if (file == NULL) {
        return 0;
    }
    for (index = 0u; index < 5u; ++index) {
        if (fprintf(file, "%s  %s\n", hashes[index], names[index]) < 0) {
            (void)fclose(file);
            return 0;
        }
    }
    return fclose(file) == 0;
}

static int test_write_nonfinite(const char *fixture)
{
    char path[TEST_PATH_CAP];
    char hash[65];
    FILE *file;
    const float value = NAN;
    if (!test_path(fixture, "train_mu_f32.bin", path)) {
        return 0;
    }
    file = fopen(path, "r+b");
    if (file == NULL) {
        return 0;
    }
    if (fwrite(&value, sizeof(value), 1u, file) != 1u || fclose(file) != 0 ||
        vae_c_mu_file_sha256(path, hash) != VAE_C_MU_OK ||
        !test_replace_spec_value(fixture, "train_matrix_sha256", hash) ||
        !test_refresh_hash_manifest(fixture)) {
        return 0;
    }
    return 1;
}

static int test_write_wrong_event(const char *fixture)
{
    char path[TEST_PATH_CAP];
    char hash[65];
    char content[300000];
    FILE *file;
    size_t bytes;
    char *row;
    if (!test_path(fixture, "train_rows.csv", path)) {
        return 0;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        return 0;
    }
    bytes = fread(content, 1u, sizeof(content) - 1u, file);
    if (ferror(file) || !feof(file)) {
        (void)fclose(file);
        return 0;
    }
    (void)fclose(file);
    content[bytes] = '\0';
    row = strchr(content, '\n');
    if (row == NULL || strncmp(row + 1u, "0,0,7663,", 9u) != 0) {
        return 0;
    }
    /* Keep byte count unchanged while making the first index discordant. */
    row[8] = '4';
    file = fopen(path, "wb");
    if (file == NULL) {
        return 0;
    }
    if (fwrite(content, 1u, bytes, file) != bytes || fclose(file) != 0 ||
        vae_c_mu_file_sha256(path, hash) != VAE_C_MU_OK ||
        !test_replace_spec_value(fixture, "train_rows_sha256", hash) ||
        !test_refresh_hash_manifest(fixture)) {
        return 0;
    }
    return 1;
}

int main(int argc, char **argv)
{
    VAECIICMini112Central view;
    char original[TEST_PATH_CAP];
    char fixture[TEST_PATH_CAP];
    char error[512];
    char digest[65];
    size_t index;
    size_t train_count = 0u;
    size_t validation_count = 0u;
    if (argc != 3 || !test_path(argv[1], test_artifact_relative, original) ||
        strlen(argv[2]) >= sizeof(fixture)) {
        fprintf(stderr, "usage: test_vae_c_iic_mini112_central ROOT BUILD_FIXTURE\n");
        return 1;
    }
    (void)strcpy(fixture, argv[2]);
    vae_c_iic_mini112_central_init(&view);
    if (!vae_c_iic_mini112_central_load(argv[1], NULL, &view, error,
                                       sizeof(error))) {
        fprintf(stderr, "canonical central load failed: %s\n", error);
        return 1;
    }
    if (view.row_count != VAE_C_IIC_MINI112_ROWS ||
        view.latent_dim != VAE_C_IIC_MINI112_LATENT_DIM ||
        strcmp(view.checkpoint_sha256,
               "8857090fb1477fafed6b841bb880a06d61acf38899c7ad2e5a9ef60672989a8b") != 0) {
        fprintf(stderr, "central population/checkpoint identity mismatch\n");
        vae_c_iic_mini112_central_free(&view);
        return 1;
    }
    for (index = 0u; index < view.row_count; ++index) {
        const VAECIICMini112CentralRow *row = &view.rows[index];
        const float *mu = vae_c_iic_mini112_central_row_mu(&view, index);
        size_t dimension;
        if (row->manifest_index != index || row->event_index != index + 1u ||
            row->event_id[0] == '\0' || mu == NULL) {
            fprintf(stderr, "central row identity mismatch: %llu\n",
                    (unsigned long long)index);
            vae_c_iic_mini112_central_free(&view);
            return 1;
        }
        if (row->source == VAE_C_IIC_MINI112_SOURCE_TRAIN) {
            ++train_count;
        } else if (row->source == VAE_C_IIC_MINI112_SOURCE_VALIDATION) {
            ++validation_count;
        } else {
            return 1;
        }
        for (dimension = 0u; dimension < view.latent_dim; ++dimension) {
            if (!isfinite(mu[dimension])) {
                fprintf(stderr, "non-finite central mu after assembly\n");
                vae_c_iic_mini112_central_free(&view);
                return 1;
            }
        }
    }
    if (train_count != 6866u || validation_count != 1717u ||
        vae_c_sha256_digest_hex_lower(
            view.mu, VAE_C_IIC_MINI112_ROWS *
                         VAE_C_IIC_MINI112_LATENT_DIM * sizeof(float),
            digest) != VAE_C_SHA256_OK) {
        fprintf(stderr, "central split counts/digest invalid\n");
        vae_c_iic_mini112_central_free(&view);
        return 1;
    }
    printf("MINI112_CENTRAL_MU_ROWS=8583\n");
    printf("MINI112_CENTRAL_MU_TRAIN_ROWS=%llu\n",
           (unsigned long long)train_count);
    printf("MINI112_CENTRAL_MU_VALIDATION_ROWS=%llu\n",
           (unsigned long long)validation_count);
    printf("MINI112_CENTRAL_MU_MANIFEST_ORDER_DIGEST=%s\n", digest);
    printf("CENTRAL_MU_SHA256=%s\n", digest);
    vae_c_iic_mini112_central_free(&view);

    if (central_mkdir(fixture) != 0) {
        /* Existing test fixture is safe: every named file is overwritten. */
        char probe[TEST_PATH_CAP];
        if (!test_path(fixture, "embedding_spec.txt", probe)) {
            return 1;
        }
    }
    if (!test_restore_fixture(original, fixture) ||
        !test_path(fixture, "train_mu_f32.bin", original)) {
        fprintf(stderr, "cannot create external negative-test fixture\n");
        return 1;
    }
    {
        FILE *file = fopen(original, "r+b");
        int byte;
        if (file == NULL || (byte = fgetc(file)) == EOF ||
            fseek(file, 0L, SEEK_SET) != 0 ||
            fputc(byte ^ 1, file) == EOF || fclose(file) != 0) {
            fprintf(stderr, "cannot tamper external matrix fixture\n");
            return 1;
        }
    }
    if (vae_c_iic_mini112_central_load(argv[1], fixture, &view, error,
                                       sizeof(error)) || view.mu != NULL) {
        fprintf(stderr, "checksum corruption was accepted\n");
        vae_c_iic_mini112_central_free(&view);
        return 1;
    }
    printf("MINI112_CENTRAL_MU_CORRUPT_HASH_REJECTED=true\n");

    if (!test_path(argv[1], test_artifact_relative, original) ||
        !test_restore_fixture(original, fixture) ||
        !test_write_nonfinite(fixture) ||
        vae_c_iic_mini112_central_load(argv[1], fixture, &view, error,
                                       sizeof(error))) {
        fprintf(stderr, "non-finite matrix was not rejected: %s\n", error);
        vae_c_iic_mini112_central_free(&view);
        return 1;
    }
    printf("MINI112_CENTRAL_MU_NONFINITE_REJECTED=true\n");

    if (!test_restore_fixture(original, fixture) ||
        !test_write_wrong_event(fixture) ||
        vae_c_iic_mini112_central_load(argv[1], fixture, &view, error,
                                       sizeof(error))) {
        fprintf(stderr, "row-map mismatch was not rejected: %s\n", error);
        vae_c_iic_mini112_central_free(&view);
        return 1;
    }
    printf("MINI112_CENTRAL_MU_ROW_MISMATCH_REJECTED=true\n");
    printf("MINI112_CENTRAL_MU_TEST_PASS=true\n");
    printf("CENTRAL_TEST_PASS=true\n");
    return 0;
}
