#include "vae_c_iic_mini224_central.h"

#include "vae_c_mu_contract.h"
#include "vae_event_dataset.h"
#include "vae_saved_split.h"

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CENTRAL_PATH_CAP 4096u
#define CENTRAL_LINE_CAP 1024u

static const char *const central_files[5] = {
    "embedding_spec.txt", "train_mu_f32.bin", "train_rows.csv",
    "validation_mu_f32.bin", "validation_rows.csv"
};
static const char *const central_canonical_sha256[5] = {
    "d3c67e9efcfedcb5b620893cfb07ec225505653e42267986dcb030bcbcb89ef6",
    "f494b232cb2fbf9ca4fb25df80c47e5f80963ac1d7448afd807361409a4ab166",
    "fd5bb3c48f87e98d5ec8ed0e561d5fe077ab96a7effa0c7425a41a222cd4375c",
    "5f6aec1737549a24d20897581b4c6b57b3fae6859e1dc1e972a8b8841ede4617",
    "87a3c7fc715fe0f4a6158474e46f6903d5267a45741d338836e8fecd61df7ed8"
};
static const char central_artifact_relative[] =
    "sakai_multires_c/results/vae_c/embeddings/"
    "mini224_bridge_v1_subset1000_best_epoch99_posterior_mu_central_v1";

static void central_error(char *message, size_t capacity, const char *format, ...)
{
    va_list arguments;
    if (message == NULL || capacity == 0u) {
        return;
    }
    va_start(arguments, format);
    (void)vsnprintf(message, capacity, format, arguments);
    va_end(arguments);
}

static int central_join(const char *directory, const char *name,
                        char path[CENTRAL_PATH_CAP])
{
    const size_t length = strlen(directory);
    const int separator = length > 0u && directory[length - 1u] != '/' &&
        directory[length - 1u] != '\\';
    const int written = snprintf(path, CENTRAL_PATH_CAP, "%s%s%s",
                                 directory, separator ? "/" : "", name);
    return written >= 0 && (size_t)written < CENTRAL_PATH_CAP;
}

static int central_line(FILE *file, char line[CENTRAL_LINE_CAP])
{
    size_t length;
    if (fgets(line, CENTRAL_LINE_CAP, file) == NULL) {
        return 0;
    }
    length = strlen(line);
    if (length == CENTRAL_LINE_CAP - 1u && line[length - 1u] != '\n') {
        return -1;
    }
    while (length > 0u && (line[length - 1u] == '\n' ||
                           line[length - 1u] == '\r')) {
        line[--length] = '\0';
    }
    return 1;
}

static int central_spec_value(const char *spec_path, const char *key,
                              char *value, size_t capacity)
{
    FILE *file;
    char line[CENTRAL_LINE_CAP];
    const size_t key_length = strlen(key);
    int found = 0;
    int status;
    file = fopen(spec_path, "rb");
    if (file == NULL) {
        return 0;
    }
    while ((status = central_line(file, line)) == 1) {
        if (strncmp(line, key, key_length) == 0 && line[key_length] == '=') {
            const size_t length = strlen(line + key_length + 1u);
            if (found || length >= capacity) {
                (void)fclose(file);
                return 0;
            }
            memcpy(value, line + key_length + 1u, length + 1u);
            found = 1;
        }
    }
    (void)fclose(file);
    return status == 0 && found;
}

static int central_require_spec(const char *spec_path, const char *key,
                                const char *expected, char *error,
                                size_t error_capacity)
{
    char value[CENTRAL_LINE_CAP];
    if (!central_spec_value(spec_path, key, value, sizeof(value)) ||
        strcmp(value, expected) != 0) {
        central_error(error, error_capacity, "embedding_spec mismatch: %s", key);
        return 0;
    }
    return 1;
}

static int central_require_number(const char *spec_path, const char *key,
                                  uint64_t number, char *error,
                                  size_t error_capacity)
{
    char text[32];
    const int written = snprintf(text, sizeof(text), "%llu",
                                 (unsigned long long)number);
    return written >= 0 && (size_t)written < sizeof(text) &&
        central_require_spec(spec_path, key, text, error, error_capacity);
}

static int central_file_size(const char *path, uint64_t *bytes)
{
    FILE *file = fopen(path, "rb");
    long length;
    int close_status;
    if (file == NULL) {
        return 0;
    }
    if (fseek(file, 0L, SEEK_END) != 0) {
        (void)fclose(file);
        return 0;
    }
    length = ftell(file);
    close_status = fclose(file);
    if (length < 0L || close_status != 0) {
        return 0;
    }
    *bytes = (uint64_t)length;
    return 1;
}

static int central_verify_files(const char *directory,
                                char hashes[5][65], char *error,
                                size_t error_capacity)
{
    char manifest_path[CENTRAL_PATH_CAP];
    char path[CENTRAL_PATH_CAP];
    char line[CENTRAL_LINE_CAP];
    FILE *manifest;
    size_t index;
    int status;
    if (!central_join(directory, "sha256.txt", manifest_path)) {
        central_error(error, error_capacity, "artifact path overflow");
        return 0;
    }
    manifest = fopen(manifest_path, "rb");
    if (manifest == NULL) {
        central_error(error, error_capacity, "missing artifact sha256.txt");
        return 0;
    }
    for (index = 0u; index < 5u; ++index) {
        if (!central_join(directory, central_files[index], path) ||
            vae_c_mu_file_sha256(path, hashes[index]) != VAE_C_MU_OK ||
            central_line(manifest, line) != 1 || strlen(line) != 66u +
                strlen(central_files[index]) ||
            strncmp(line, hashes[index], 64u) != 0 ||
            line[64] != ' ' || line[65] != ' ' ||
            strcmp(line + 66u, central_files[index]) != 0) {
            central_error(error, error_capacity,
                          "artifact checksum/file mismatch: %s",
                          central_files[index]);
            (void)fclose(manifest);
            return 0;
        }
    }
    status = central_line(manifest, line);
    (void)fclose(manifest);
    if (status != 0) {
        central_error(error, error_capacity, "artifact checksum has extra rows");
        return 0;
    }
    return 1;
}

static int central_validate_spec(const char *repository_root,
                                 const char *directory,
                                 const VAECMuCanonicalContract *contract,
                                 char hashes[5][65], char *error,
                                 size_t error_capacity)
{
    const VAECMuArtifactContractV1 *artifact = &contract->artifact_v1;
    char spec_path[CENTRAL_PATH_CAP];
    char checkpoint_sha256[65];
    (void)repository_root;
    if (!central_join(directory, central_files[0], spec_path)) {
        return 0;
    }
#define CENTRAL_REQUIRE(key, expected) \
    do { if (!central_require_spec(spec_path, (key), (expected), error, \
                                  error_capacity)) return 0; } while (0)
#define CENTRAL_NUMBER(key, number) \
    do { if (!central_require_number(spec_path, (key), (number), error, \
                                    error_capacity)) return 0; } while (0)
    CENTRAL_REQUIRE("format", VAE_C_MU_ARTIFACT_V1);
    CENTRAL_REQUIRE("representation", "posterior_mu");
    CENTRAL_REQUIRE("view", "central");
    CENTRAL_REQUIRE("profile_id", artifact->profile_id);
    CENTRAL_NUMBER("latent_dim", VAE_C_IIC_MINI224_LATENT_DIM);
    CENTRAL_REQUIRE("profile_config_sha256", artifact->profile_config_sha256);
    CENTRAL_REQUIRE("run_id", contract->run_id);
    CENTRAL_REQUIRE("run_config_path",
                    "sakai_multires_c/results/vae_c/training/"
                    "mini224_bridge_v1_subset1000_candidate_init123_latent456_aug789_shuffle101112_bs16_max100_pat20/"
                    "run_config.txt");
    CENTRAL_REQUIRE("run_config_sha256", contract->run_config_sha256);
    CENTRAL_REQUIRE("checkpoint_path",
                    "sakai_multires_c/results/vae_c/training/"
                    "mini224_bridge_v1_subset1000_candidate_init123_latent456_aug789_shuffle101112_bs16_max100_pat20/"
                    "best.ckpt");
    CENTRAL_REQUIRE("checkpoint_kind", "BEST");
    CENTRAL_NUMBER("checkpoint_epoch", contract->expected_epoch);
    CENTRAL_NUMBER("checkpoint_optimizer_step",
                   contract->expected_optimizer_step);
    CENTRAL_REQUIRE("checkpoint_sha256", contract->checkpoint_sha256);
    CENTRAL_REQUIRE("checkpoint_code_commit", artifact->checkpoint_code_commit);
    CENTRAL_REQUIRE("catalog_sha256", artifact->catalog_sha256);
    CENTRAL_REQUIRE("catalog_path", "sakai_multires_c/configs/vae_c_profiles_v2.csv");
    CENTRAL_REQUIRE("event_manifest_path", "manifest_events_272.csv");
    CENTRAL_REQUIRE("event_manifest_sha256", artifact->event_manifest_sha256);
    CENTRAL_REQUIRE("tensor_manifest_path", "manifest_tensors_224_central.csv");
    CENTRAL_REQUIRE("tensor_manifest_sha256", artifact->tensor_manifest_sha256);
    CENTRAL_REQUIRE("train_population_path", "sakai_multires_c/configs/subset224_train_800_seed123.csv");
    CENTRAL_REQUIRE("validation_population_path", "sakai_multires_c/configs/subset224_val_200_seed123.csv");
    CENTRAL_REQUIRE("train_split_sha256", artifact->train_split_sha256);
    CENTRAL_REQUIRE("validation_split_sha256",
                    artifact->validation_split_sha256);
    CENTRAL_REQUIRE("population_kind", "canonical_subset");
    CENTRAL_REQUIRE("official_train_split_path", "split_train_indices_seed123.csv");
    CENTRAL_REQUIRE("official_validation_split_path", "split_val_indices_seed123.csv");
    CENTRAL_REQUIRE("train_subset_sha256", contract->train_subset_sha256);
    CENTRAL_REQUIRE("validation_subset_sha256", contract->validation_subset_sha256);
    CENTRAL_REQUIRE("population_identity", artifact->population_identity);
    CENTRAL_REQUIRE("encoder_only", "true");
    CENTRAL_REQUIRE("batch_norm_mode", "EVAL");
    CENTRAL_REQUIRE("epsilon_used", "false");
    CENTRAL_REQUIRE("latent_sampled", "false");
    CENTRAL_REQUIRE("decoder_executed", "false");
    CENTRAL_REQUIRE("labels_used", "false");
    CENTRAL_REQUIRE("dtype", "float32_ieee754");
    CENTRAL_REQUIRE("endian", "little");
    CENTRAL_REQUIRE("layout", "row_major_[N,latent_dim]");
    CENTRAL_REQUIRE("latent_scaling", "none");
    CENTRAL_REQUIRE("latent_normalization", "none");
    CENTRAL_REQUIRE("pca", "false");
    CENTRAL_REQUIRE("row_order", "native_split_pos_ascending");
    CENTRAL_REQUIRE("row_map_schema",
                    "row_index,split_pos,manifest_index,event_index,event_id,split");
    CENTRAL_REQUIRE("train_matrix_file", central_files[1]);
    CENTRAL_REQUIRE("train_rows_file", central_files[2]);
    CENTRAL_NUMBER("train_rows", contract->train_count);
    CENTRAL_NUMBER("train_cols", VAE_C_IIC_MINI224_LATENT_DIM);
    CENTRAL_NUMBER("train_expected_bytes", artifact->train.expected_bytes);
    CENTRAL_REQUIRE("train_matrix_sha256", hashes[1]);
    CENTRAL_REQUIRE("train_rows_sha256", hashes[2]);
    CENTRAL_REQUIRE("validation_matrix_file", central_files[3]);
    CENTRAL_REQUIRE("validation_rows_file", central_files[4]);
    CENTRAL_NUMBER("validation_rows", contract->validation_count);
    CENTRAL_NUMBER("validation_cols", VAE_C_IIC_MINI224_LATENT_DIM);
    CENTRAL_NUMBER("validation_expected_bytes",
                   artifact->validation.expected_bytes);
    CENTRAL_REQUIRE("validation_matrix_sha256", hashes[3]);
    CENTRAL_REQUIRE("validation_rows_sha256", hashes[4]);
    CENTRAL_REQUIRE("checksum_file", "sha256.txt");
    CENTRAL_REQUIRE("checksum_manifest_excludes_self", "true");
#undef CENTRAL_REQUIRE
#undef CENTRAL_NUMBER
    if (vae_c_mu_file_sha256(contract->checkpoint_path, checkpoint_sha256) !=
            VAE_C_MU_OK ||
        strcmp(checkpoint_sha256, contract->checkpoint_sha256) != 0) {
        central_error(error, error_capacity, "BEST checkpoint SHA-256 mismatch");
        return 0;
    }
    return 1;
}

static int central_parse_u32(const char *text, uint32_t *value)
{
    char *end;
    unsigned long long parsed;
    if (text == NULL || text[0] == '\0' || text[0] == '-' ||
        text[0] == '+') {
        return 0;
    }
    errno = 0;
    parsed = strtoull(text, &end, 10);
    if (errno != 0 || *end != '\0' || parsed > UINT32_MAX) {
        return 0;
    }
    *value = (uint32_t)parsed;
    return 1;
}

static int central_parse_row(char line[CENTRAL_LINE_CAP], uint32_t *row_index,
                             VAECIICMini224CentralRow *row,
                             const char *expected_split)
{
    char *fields[6];
    char *cursor = line;
    size_t field;
    for (field = 0u; field < 5u; ++field) {
        char *comma = strchr(cursor, ',');
        if (comma == NULL) {
            return 0;
        }
        *comma = '\0';
        fields[field] = cursor;
        cursor = comma + 1u;
    }
    fields[5] = cursor;
    if (strchr(fields[5], ',') != NULL ||
        strlen(fields[4]) == 0u ||
        strlen(fields[4]) >= sizeof(row->event_id) ||
        strcmp(fields[5], expected_split) != 0 ||
        !central_parse_u32(fields[0], row_index) ||
        !central_parse_u32(fields[1], &row->split_pos) ||
        !central_parse_u32(fields[2], &row->manifest_index) ||
        !central_parse_u32(fields[3], &row->event_index)) {
        return 0;
    }
    (void)strcpy(row->event_id, fields[4]);
    row->source_split = strcmp(expected_split, "train") == 0 ?
        VAE_C_IIC_MINI224_SOURCE_TRAIN :
        VAE_C_IIC_MINI224_SOURCE_VALIDATION;
    return 1;
}

static int central_load_split(const char *directory, const char *row_name,
                              const char *matrix_name, const char *split_name,
                              const GSEventManifest *events,
                              const GSSavedSplit *split,
                              VAECIICMini224Central *candidate, size_t base,
                              unsigned char seen[VAE_C_IIC_MINI224_MANIFEST_ROWS],
                              char *error, size_t error_capacity)
{
    char row_path[CENTRAL_PATH_CAP];
    char matrix_path[CENTRAL_PATH_CAP];
    char line[CENTRAL_LINE_CAP];
    FILE *rows = NULL;
    FILE *matrix = NULL;
    size_t position;
    uint64_t bytes;
    int status;
    int ok = 0;
    if (!central_join(directory, row_name, row_path) ||
        !central_join(directory, matrix_name, matrix_path) ||
        !central_file_size(matrix_path, &bytes) ||
        bytes != (uint64_t)split->count * VAE_C_IIC_MINI224_LATENT_DIM * 4u) {
        central_error(error, error_capacity, "matrix byte count mismatch: %s",
                      matrix_name);
        return 0;
    }
    rows = fopen(row_path, "rb");
    matrix = fopen(matrix_path, "rb");
    if (rows == NULL || matrix == NULL ||
        central_line(rows, line) != 1 ||
        strcmp(line, "row_index,split_pos,manifest_index,event_index,event_id,split")
            != 0) {
        central_error(error, error_capacity, "row-map header invalid: %s",
                      row_name);
        goto cleanup;
    }
    for (position = 0u; position < split->count; ++position) {
        VAECIICMini224CentralRow row;
        uint32_t row_index;
        size_t dimension;
        float *destination;
        const GSEventRecord *event;
        memset(&row, 0, sizeof(row));
        if (central_line(rows, line) != 1 ||
            !central_parse_row(line, &row_index, &row, split_name) ||
            row_index != position || row.split_pos != position ||
            row.manifest_index >= VAE_C_IIC_MINI224_MANIFEST_ROWS ||
            seen[row.manifest_index] != 0u ||
            row.manifest_index != split->indices[position] ||
            base + position >= VAE_C_IIC_MINI224_ROWS) {
            central_error(error, error_capacity,
                          "row-map order/coverage mismatch: %s row %llu",
                          row_name, (unsigned long long)position);
            goto cleanup;
        }
        event = &events->records[row.manifest_index];
        for (size_t earlier = 0u; earlier < base + position; ++earlier) {
            if (strcmp(candidate->rows[earlier].event_id, row.event_id) == 0) {
                central_error(error, error_capacity,
                              "duplicate event_id: %s row %llu", row_name,
                              (unsigned long long)position);
                goto cleanup;
            }
        }
        if (row.event_index != (uint32_t)event->event_index ||
            strcmp(row.event_id, event->event_id) != 0) {
            central_error(error, error_capacity,
                          "row-map event identity mismatch: %s row %llu",
                          row_name, (unsigned long long)position);
            goto cleanup;
        }
        destination = candidate->mu +
            (base + position) * VAE_C_IIC_MINI224_LATENT_DIM;
        if (fread(destination, sizeof(float), VAE_C_IIC_MINI224_LATENT_DIM,
                  matrix) != VAE_C_IIC_MINI224_LATENT_DIM) {
            central_error(error, error_capacity, "matrix short read: %s",
                          matrix_name);
            goto cleanup;
        }
        for (dimension = 0u; dimension < VAE_C_IIC_MINI224_LATENT_DIM;
             ++dimension) {
            if (!isfinite(destination[dimension])) {
                central_error(error, error_capacity,
                              "non-finite central mu: %s row %llu",
                              matrix_name, (unsigned long long)position);
                goto cleanup;
            }
        }
        row.iic_row_index = (uint32_t)(base + position);
        candidate->rows[base + position] = row;
        seen[row.manifest_index] = 1u;
    }
    status = central_line(rows, line);
    if (status != 0 || fgetc(matrix) != EOF || ferror(matrix)) {
        central_error(error, error_capacity, "extra rows or matrix bytes: %s",
                      split_name);
        goto cleanup;
    }
    ok = 1;
cleanup:
    if (rows != NULL) {
        (void)fclose(rows);
    }
    if (matrix != NULL) {
        (void)fclose(matrix);
    }
    return ok;
}
void vae_c_iic_mini224_central_init(VAECIICMini224Central *view)
{
    if (view != NULL) {
        memset(view, 0, sizeof(*view));
    }
}

void vae_c_iic_mini224_central_free(VAECIICMini224Central *view)
{
    if (view != NULL) {
        free(view->mu);
        free(view->rows);
        vae_c_iic_mini224_central_init(view);
    }
}

const float *vae_c_iic_mini224_central_row_mu(
    const VAECIICMini224Central *view, size_t iic_row_index)
{
    if (view == NULL || view->mu == NULL ||
        iic_row_index >= view->row_count ||
        view->latent_dim != VAE_C_IIC_MINI224_LATENT_DIM) {
        return NULL;
    }
    return view->mu + iic_row_index * VAE_C_IIC_MINI224_LATENT_DIM;
}

int vae_c_iic_mini224_central_load(
    const char *repository_root, const char *artifact_directory_override,
    VAECIICMini224Central *view, char *error_message, size_t error_capacity)
{
    VAECMuCanonicalContract contract;
    VAECIICMini224Central candidate;
    GSEventManifest events;
    GSSavedSplit train;
    GSSavedSplit validation;
    unsigned char seen[VAE_C_IIC_MINI224_MANIFEST_ROWS] = {0};
    char directory[CENTRAL_PATH_CAP];
    char hashes[5][65];
    char hashes_after_read[5][65];
    char detail[256];
    size_t index;
    int ok = 0;
    memset(&events, 0, sizeof(events));
    memset(&train, 0, sizeof(train));
    memset(&validation, 0, sizeof(validation));
    vae_c_iic_mini224_central_init(&candidate);
    if (error_message != NULL && error_capacity != 0u) {
        error_message[0] = '\0';
    }
    if (repository_root == NULL || repository_root[0] == '\0' ||
        view == NULL || view->mu != NULL || view->rows != NULL) {
        central_error(error_message, error_capacity, "invalid load arguments");
        return 0;
    }
    if (vae_c_mu_build_canonical_contract_label_free(
            repository_root, "mini224", &contract, detail, sizeof(detail)) !=
        VAE_C_MU_OK) {
        central_error(error_message, error_capacity,
                      "canonical mini224 contract invalid: %s", detail);
        return 0;
    }
    if (contract.event_count != VAE_C_IIC_MINI224_MANIFEST_ROWS ||
        contract.train_count != VAE_C_IIC_MINI224_TRAIN_ROWS ||
        contract.validation_count != VAE_C_IIC_MINI224_VALIDATION_ROWS ||
        contract.artifact_v1.latent_dim != VAE_C_IIC_MINI224_LATENT_DIM ||
        !contract.uses_subset) {
        central_error(error_message, error_capacity,
                      "canonical mini224 population/config mismatch");
        return 0;
    }
    if (artifact_directory_override != NULL) {
        if (strlen(artifact_directory_override) >= sizeof(directory)) {
            central_error(error_message, error_capacity,
                          "artifact directory path overflow");
            return 0;
        }
        (void)strcpy(directory, artifact_directory_override);
    } else if (!central_join(repository_root, central_artifact_relative,
                             directory)) {
        central_error(error_message, error_capacity,
                      "artifact directory path overflow");
        return 0;
    }
    if (!central_verify_files(directory, hashes, error_message,
                              error_capacity) ||
        !central_validate_spec(repository_root, directory, &contract, hashes,
                               error_message, error_capacity)) {
        return 0;
    }
    if (artifact_directory_override == NULL) {
        char checksum_path[CENTRAL_PATH_CAP];
        char checksum_sha256[65];
        if (!central_join(directory, "sha256.txt", checksum_path) ||
            vae_c_mu_file_sha256(checksum_path, checksum_sha256) != VAE_C_MU_OK ||
            strcmp(checksum_sha256,
                   "aa55332307b89ec902ffd217467cdd58870728f382ad8956045cf91878ed8c95") != 0) {
            central_error(error_message, error_capacity,
                          "canonical mini224 checksum manifest mismatch");
            return 0;
        }
        for (index = 0u; index < 5u; ++index) {
            if (strcmp(hashes[index], central_canonical_sha256[index]) != 0) {
                central_error(error_message, error_capacity,
                              "canonical mini224 artifact SHA-256 mismatch: %s",
                              central_files[index]);
                return 0;
            }
        }
    }
    if (gs_event_manifest_load_label_free(contract.event_manifest_path, &events) != 0 ||
        events.count != VAE_C_IIC_MINI224_MANIFEST_ROWS ||
        gs_saved_split_load(contract.train_population_path, "train",
                            &events, &train) != 0 ||
        gs_saved_split_load(contract.validation_population_path,
                            "validation", &events, &validation) != 0 ||
        train.count != contract.train_count ||
        validation.count != contract.validation_count) {
        central_error(error_message, error_capacity,
                      "event manifest/split load mismatch");
        goto cleanup;
    }
    candidate.mu = (float *)calloc(VAE_C_IIC_MINI224_ROWS *
                                    VAE_C_IIC_MINI224_LATENT_DIM,
                                    sizeof(float));
    candidate.rows = (VAECIICMini224CentralRow *)calloc(
        VAE_C_IIC_MINI224_ROWS, sizeof(*candidate.rows));
    if (candidate.mu == NULL || candidate.rows == NULL) {
        central_error(error_message, error_capacity, "central mu allocation failed");
        goto cleanup;
    }
    if (!central_load_split(directory, central_files[2], central_files[1],
                            "train", &events, &train, &candidate, 0u, seen,
                            error_message, error_capacity) ||
        !central_load_split(directory, central_files[4], central_files[3],
                            "validation", &events, &validation, &candidate,
                            VAE_C_IIC_MINI224_TRAIN_ROWS, seen, error_message, error_capacity)) {
        goto cleanup;
    }
    for (index = 0u; index < VAE_C_IIC_MINI224_ROWS; ++index) {
        if (candidate.rows[index].iic_row_index != index ||
            candidate.rows[index].event_index !=
                candidate.rows[index].manifest_index + 1u ||
            candidate.rows[index].source_split !=
                (index < VAE_C_IIC_MINI224_TRAIN_ROWS ?
                 VAE_C_IIC_MINI224_SOURCE_TRAIN :
                 VAE_C_IIC_MINI224_SOURCE_VALIDATION)) {
            central_error(error_message, error_capacity,
                          "central mu manifest coverage mismatch at %llu",
                          (unsigned long long)index);
            goto cleanup;
        }
    }
    if (!central_verify_files(directory, hashes_after_read, error_message,
                              error_capacity)) {
        goto cleanup;
    }
    for (index = 0u; index < 5u; ++index) {
        if (strcmp(hashes[index], hashes_after_read[index]) != 0) {
            central_error(error_message, error_capacity,
                          "artifact changed while central mu was read");
            goto cleanup;
        }
    }
    candidate.row_count = VAE_C_IIC_MINI224_ROWS;
    candidate.latent_dim = VAE_C_IIC_MINI224_LATENT_DIM;
    (void)strcpy(candidate.checkpoint_sha256, contract.checkpoint_sha256);
    (void)strcpy(candidate.train_matrix_sha256, hashes[1]);
    (void)strcpy(candidate.train_rows_sha256, hashes[2]);
    (void)strcpy(candidate.validation_matrix_sha256, hashes[3]);
    (void)strcpy(candidate.validation_rows_sha256, hashes[4]);
    *view = candidate;
    vae_c_iic_mini224_central_init(&candidate);
    ok = 1;
cleanup:
    vae_c_iic_mini224_central_free(&candidate);
    gs_saved_split_free(&validation);
    gs_saved_split_free(&train);
    gs_event_manifest_free(&events);
    return ok;
}
