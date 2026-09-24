#include "vae_c_iic_checkpoint.h"
#include "vae_c_iic_mini224_perturbed.h"
#include "vae_c_iic_png_identity.h"
#include "vae_c_sha256.h"
#include "vae_c_shuffle.h"
#include "vae_augmentation.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifndef _WIN32
#include <dirent.h>
#include <sys/statvfs.h>
#endif

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

/* Validate-only is isolated above SMOKE_CODE_START. The epoch gate below is
 * debug-only and is restricted to an existing external temporary directory. */
#define PATH_CAP 4096u
#define TEXT_CAP 8192u
#define ROWS VAE_C_IIC_MINI224_ROWS
#define LOGICAL_BATCH 128u
#define ENCODER_MICROBATCH 16u
#define MAX_EPOCHS UINT64_C(100)
#define BATCHES_PER_EPOCH 8u
#define BEST_SHA256 \
    "15a1ecd6dd3c6d11aa0a2ac20ea0b6176d08a18c6f582e7349ed6784ab8528f3"
#define PNG_SHA256 \
    "5e6bf28879886f69223864076d354a5bce511b78164e96e742e6117f79f51a5c"
#define PNG_TAR_SHA256 \
    "70c766c3652f586c8da4c8cadf134a8ce83cfd07c8cd10545367f380017ddab2"

typedef struct {
    const char *repository_root;
    const char *code_commit;
    const char *output_dir;
    const char *smoke_output_dir;
    const char *debug_output_dir;
    const char *expected_best_sha256;
    size_t expected_population;
    size_t logical_batch;
    size_t encoder_microbatch;
    uint64_t max_epochs;
    bool validate_only;
    bool smoke_real_two_batch;
    bool debug_epoch_gate;
    bool debug_resume;
    bool debug_stop_seen;
    uint64_t debug_stop_after_epoch;
    bool official_run;
    bool official_preflight;
    bool official_resume;
} Options;

typedef struct {
    VAECIICMini224Perturbed input;
    VAECIICCheckpointIdentity checkpoint_identity;
    char source_identity_sha256[VAE_C_SHA256_HEX_SIZE];
    char producer_binary_sha256[VAE_C_SHA256_HEX_SIZE];
    char png_sha256[VAE_C_SHA256_HEX_SIZE];
    char run_id[VAE_C_IIC_CHECKPOINT_RUN_ID_CAP];
    char run_config[TEXT_CAP];
    char run_config_sha256[VAE_C_SHA256_HEX_SIZE];
    char official_output_dir[PATH_CAP];
} RunInput;

static bool decimal_u64(const char *text, uint64_t *value)
{
    char *end;
    unsigned long long parsed;
    if (text == NULL || text[0] == '\0' || text[0] == '-' ||
        text[0] == '+') {
        return false;
    }
    errno = 0;
    parsed = strtoull(text, &end, 10);
    if (errno != 0 || *end != '\0') {
        return false;
    }
    *value = (uint64_t)parsed;
    return (unsigned long long)*value == parsed;
}

static bool lower_hex(const char *text, size_t length)
{
    size_t index;
    if (text == NULL || strlen(text) != length) {
        return false;
    }
    for (index = 0u; index < length; ++index) {
        if (!((text[index] >= '0' && text[index] <= '9') ||
              (text[index] >= 'a' && text[index] <= 'f'))) {
            return false;
        }
    }
    return true;
}

static bool parse_options(int argc, char **argv, Options *options)
{
    int index;
    unsigned int seen = 0u;
    memset(options, 0, sizeof(*options));
    options->expected_best_sha256 = BEST_SHA256;
    options->expected_population = ROWS;
    options->logical_batch = LOGICAL_BATCH;
    options->encoder_microbatch = ENCODER_MICROBATCH;
    options->max_epochs = MAX_EPOCHS;
    for (index = 1; index < argc; ++index) {
        const char *arg = argv[index];
        if (strcmp(arg, "--validate-only") == 0 && !options->validate_only &&
            !options->smoke_real_two_batch && !options->debug_epoch_gate) {
            options->validate_only = true;
        } else if (strcmp(arg, "--smoke-real-two-batch") == 0 &&
                   !options->validate_only && !options->smoke_real_two_batch &&
                   !options->debug_epoch_gate) {
            options->smoke_real_two_batch = true;
        } else if (strcmp(arg, "--debug-epoch-gate") == 0 &&
                    !options->validate_only && !options->smoke_real_two_batch &&
                    !options->debug_epoch_gate && !options->official_run &&
                    !options->official_preflight) {
            options->debug_epoch_gate = true;
        } else if (strcmp(arg, "--official-run") == 0 &&
                   !options->validate_only && !options->smoke_real_two_batch &&
                   !options->debug_epoch_gate && !options->official_run &&
                   !options->official_preflight) {
            options->official_run = true;
        } else if (strcmp(arg, "--official-preflight") == 0 &&
                   !options->validate_only && !options->smoke_real_two_batch &&
                   !options->debug_epoch_gate && !options->official_run &&
                   !options->official_preflight) {
            options->official_preflight = true;
        } else if (strcmp(arg, "--official-resume") == 0 &&
                   !options->official_resume && !options->debug_resume) {
            options->official_resume = true;
        } else if (strcmp(arg, "--debug-output-dir") == 0 &&
                   ++index < argc && options->debug_output_dir == NULL) {
            options->debug_output_dir = argv[index];
        } else if (strcmp(arg, "--debug-stop-after-epoch") == 0 &&
                   ++index < argc && !options->debug_stop_seen) {
            if (!decimal_u64(argv[index],
                             &options->debug_stop_after_epoch) ||
                options->debug_stop_after_epoch > 1u) return false;
            options->debug_stop_seen = true;
        } else if (strcmp(arg, "--resume") == 0 && !options->debug_resume &&
                   !options->official_resume) {
            options->debug_resume = true;
        } else if (strcmp(arg, "--repository-root") == 0 &&
                   ++index < argc && options->repository_root == NULL) {
            options->repository_root = argv[index];
        } else if (strcmp(arg, "--code-commit") == 0 &&
                   ++index < argc && options->code_commit == NULL) {
            options->code_commit = argv[index];
        } else if (strcmp(arg, "--output-dir") == 0 &&
                   ++index < argc && options->output_dir == NULL) {
            options->output_dir = argv[index];
        } else if (strcmp(arg, "--smoke-output-dir") == 0 &&
                   ++index < argc && options->smoke_output_dir == NULL) {
            options->smoke_output_dir = argv[index];
        } else if (strcmp(arg, "--expected-best-sha256") == 0 &&
                   ++index < argc && (seen & 1u) == 0u) {
            options->expected_best_sha256 = argv[index];
            seen |= 1u;
        } else if (strcmp(arg, "--expected-population") == 0 &&
                   ++index < argc && (seen & 2u) == 0u) {
            uint64_t value;
            if (!decimal_u64(argv[index], &value) || value > SIZE_MAX) {
                return false;
            }
            options->expected_population = (size_t)value;
            seen |= 2u;
        } else if (strcmp(arg, "--logical-batch") == 0 &&
                   ++index < argc && (seen & 4u) == 0u) {
            uint64_t value;
            if (!decimal_u64(argv[index], &value) || value > SIZE_MAX) {
                return false;
            }
            options->logical_batch = (size_t)value;
            seen |= 4u;
        } else if (strcmp(arg, "--encoder-microbatch") == 0 &&
                   ++index < argc && (seen & 8u) == 0u) {
            uint64_t value;
            if (!decimal_u64(argv[index], &value) || value > SIZE_MAX) {
                return false;
            }
            options->encoder_microbatch = (size_t)value;
            seen |= 8u;
        } else if (strcmp(arg, "--max-epochs") == 0 &&
                   ++index < argc && (seen & 16u) == 0u) {
            if (!decimal_u64(argv[index], &options->max_epochs)) {
                return false;
            }
            seen |= 16u;
        } else {
            return false;
        }
    }
    if (options->official_run && options->debug_resume) {
        options->official_resume = true;
        options->debug_resume = false;
    }
    return ((options->validate_only && !options->smoke_real_two_batch &&
              !options->debug_epoch_gate && options->smoke_output_dir == NULL &&
              options->debug_output_dir == NULL && !options->debug_resume &&
              !options->debug_stop_seen && !options->official_run &&
              !options->official_preflight && !options->official_resume) ||
             (options->smoke_real_two_batch && !options->validate_only &&
              !options->debug_epoch_gate && options->output_dir == NULL &&
              options->smoke_output_dir != NULL &&
              options->debug_output_dir == NULL && !options->debug_resume &&
              !options->debug_stop_seen && !options->official_run &&
              !options->official_preflight && !options->official_resume) ||
             (options->debug_epoch_gate && !options->validate_only &&
              !options->smoke_real_two_batch && options->output_dir == NULL &&
              options->smoke_output_dir == NULL &&
              options->debug_output_dir != NULL &&
              options->debug_stop_seen && !options->official_run &&
              !options->official_preflight && !options->official_resume) ||
             ((options->official_run || options->official_preflight) &&
              !options->validate_only && !options->smoke_real_two_batch &&
              !options->debug_epoch_gate && options->output_dir == NULL &&
              options->smoke_output_dir == NULL &&
              options->debug_output_dir == NULL && !options->debug_resume &&
              !options->debug_stop_seen &&
              (!options->official_preflight || !options->official_resume))) &&
        options->repository_root != NULL &&
        options->code_commit != NULL &&
        lower_hex(options->code_commit, 40u) &&
        lower_hex(options->expected_best_sha256, 64u) &&
        options->logical_batch == LOGICAL_BATCH &&
        options->encoder_microbatch == ENCODER_MICROBATCH &&
        options->max_epochs == MAX_EPOCHS;
}

static bool join(char output[PATH_CAP], const char *directory,
                 const char *name)
{
    int written = snprintf(output, PATH_CAP, "%s/%s", directory, name);
    return written > 0 && (size_t)written < PATH_CAP;
}

static bool path_exists(const char *path)
{
#ifdef _WIN32
    return GetFileAttributesA(path) != INVALID_FILE_ATTRIBUTES;
#else
    struct stat entry;
    return stat(path, &entry) == 0;
#endif
}

static bool directory_exists(const char *path)
{
#ifdef _WIN32
    DWORD attributes = GetFileAttributesA(path);
    return attributes != INVALID_FILE_ATTRIBUTES &&
        (attributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
#else
    struct stat entry;
    return stat(path, &entry) == 0 && S_ISDIR(entry.st_mode);
#endif
}

static bool read_line(const char *path, char *line, size_t capacity)
{
    FILE *file = fopen(path, "rb");
    size_t length;
    bool ok;
    if (file == NULL || fgets(line, (int)capacity, file) == NULL) {
        if (file != NULL) (void)fclose(file);
        return false;
    }
    length = strlen(line);
    ok = length > 0u && length < capacity - 1u &&
        (line[length - 1u] == '\n' || feof(file));
    while (length > 0u &&
           (line[length - 1u] == '\n' || line[length - 1u] == '\r')) {
        line[--length] = '\0';
    }
    if (fgetc(file) != EOF || ferror(file)) ok = false;
    if (fclose(file) != 0) ok = false;
    return ok;
}

/* As in the frozen mini112 runner, a caller-supplied SHA is not trusted as
 * HEAD. The local branch ref must resolve to it. */
static bool verify_git_head(const char *root, const char *commit)
{
    char head_path[PATH_CAP], ref_path[PATH_CAP];
    char head_line[256], actual[256];
    const char *prefix = "ref: refs/heads/";
    return join(head_path, root, ".git/HEAD") &&
        read_line(head_path, head_line, sizeof(head_line)) &&
        strncmp(head_line, prefix, strlen(prefix)) == 0 &&
        strcmp(head_line + strlen(prefix), "vae-c-sakai-like") == 0 &&
        join(ref_path, root, ".git/refs/heads/vae-c-sakai-like") &&
        read_line(ref_path, actual, sizeof(actual)) &&
        lower_hex(actual, 40u) && strcmp(actual, commit) == 0;
}

static bool schedule(size_t batches[BATCHES_PER_EPOCH],
                     size_t chunks[7], uint64_t *max_steps)
{
    size_t index;
    size_t sum = 0u;
    for (index = 0u; index < BATCHES_PER_EPOCH; ++index) {
        batches[index] = ROWS - sum;
        if (batches[index] > LOGICAL_BATCH) {
            batches[index] = LOGICAL_BATCH;
        }
        sum += batches[index];
    }
    if (sum != ROWS || batches[7] != 104u ||
        LOGICAL_BATCH % ENCODER_MICROBATCH != 0u) {
        return false;
    }
    for (index = 0u; index < 7u; ++index) {
        chunks[index] = index < 6u ? ENCODER_MICROBATCH : 8u;
    }
    *max_steps = MAX_EPOCHS * BATCHES_PER_EPOCH;
    return *max_steps == 800u;
}

static bool source_valid(const RunInput *run)
{
    const VAECIICMini224Perturbed *input = &run->input;
    const VAECIICMini224Central *central = &input->central;
    unsigned char seen[VAE_C_IIC_MINI224_MANIFEST_ROWS] = {0};
    size_t row_index;
    if (!input->ready || central->row_count != ROWS ||
        central->latent_dim != VAE_C_IIC_MINI224_LATENT_DIM ||
        input->contract.train_count != VAE_C_IIC_MINI224_TRAIN_ROWS ||
        input->contract.validation_count !=
            VAE_C_IIC_MINI224_VALIDATION_ROWS ||
        input->contract.expected_epoch != 99u ||
        input->contract.expected_optimizer_step != 5000u ||
        input->contract.expected_identity.max_batch_size !=
            ENCODER_MICROBATCH ||
        strcmp(input->contract.profile.profile_id,
               "mini224_bridge_v1") != 0 ||
        strcmp(central->checkpoint_sha256, BEST_SHA256) != 0 ||
        strcmp(input->contract.checkpoint_sha256, BEST_SHA256) != 0) {
        return false;
    }
    for (row_index = 0u; row_index < ROWS; ++row_index) {
        const VAECIICMini224CentralRow *row = &central->rows[row_index];
        const GSEventRecord *event = gs_event_manifest_get(
            &input->events, row->manifest_index);
        size_t expected_split_pos = row_index < 800u ?
            row_index : row_index - 800u;
        VAECIICMini224Source expected_split = row_index < 800u ?
            VAE_C_IIC_MINI224_SOURCE_TRAIN :
            VAE_C_IIC_MINI224_SOURCE_VALIDATION;
        if (event == NULL ||
            row->manifest_index >= VAE_C_IIC_MINI224_MANIFEST_ROWS ||
            seen[row->manifest_index] != 0u ||
            row->iic_row_index != row_index ||
            row->split_pos != expected_split_pos ||
            row->source_split != expected_split ||
            row->event_index != (uint32_t)event->event_index ||
            strcmp(row->event_id, event->event_id) != 0 ||
            row->manifest_index !=
                input->canonical_manifest_indices[row_index] ||
            vae_c_iic_mini224_central_row_mu(central, row_index) == NULL) {
            return false;
        }
        seen[row->manifest_index] = 1u;
    }
    return true;
}

static bool build_identity(RunInput *run, const Options *options,
                           const char *executable)
{
    const VAECIICMini224Central *central = &run->input.central;
    const VAECMuCanonicalContract *contract = &run->input.contract;
    VAECIICCheckpointIdentity *identity = &run->checkpoint_identity;
    char preimage[TEXT_CAP];
    char digest[VAE_C_SHA256_HEX_SIZE];
    const char source_tag[] = "MINI224_IIC_REAL_INPUT_CONTRACT_V1";
    int written;
    size_t pair;
    if (vae_c_sha256_digest_hex_lower(
            source_tag, strlen(source_tag), run->source_identity_sha256) !=
            VAE_C_SHA256_OK ||
        vae_c_mu_file_sha256(executable, run->producer_binary_sha256) !=
            VAE_C_MU_OK) {
        return false;
    }
    written = snprintf(
        preimage, sizeof(preimage),
        "format=MINI224_IIC_REAL_RUN_V1\n"
        "source_identity_sha256=%s\n"
        "source_identity_role=contract_identifier\n"
        "producer_git_head=%s\n"
        "producer_binary_sha256=%s\n"
        "profile_id=mini224_bridge_v1\n"
        "profile_config_sha256=%s\n"
        "catalog_sha256=%s\n"
        "vae_run_config_sha256=%s\n"
        "vae_best_sha256=%s\n"
        "event_manifest_sha256=%s\n"
        "tensor_manifest_sha256=%s\n"
        "train_split_sha256=%s\n"
        "validation_split_sha256=%s\n"
        "train_subset_sha256=%s\n"
        "validation_subset_sha256=%s\n"
        "train_mu_sha256=%s\n"
        "validation_mu_sha256=%s\n"
        "train_rows_sha256=%s\n"
        "validation_rows_sha256=%s\n"
        "png_content_identity_format=%s\n"
        "png_reference_tar_sha256=%s\n"
        "population=mini224_subset1000_train800_validation200\n"
        "latent_dim=64\naugmentation=symmetric_figure:789\n"
        "shuffle_seed=101112\nlogical_batch=128\nencoder_microbatch=16\n"
        "C=36\nW=250\nK=5\npair_seeds=123,124,125,126,127\n"
        "adam=0.0005,0.9,0.999,1e-8,weight_decay0,amsgrad_false\n"
        "init_policy=splitmix64_raw_seed_main_then_over_v1\n"
        "max_epochs=100\ncheckpoint=IIC_V1_LATEST_FINAL_NO_BEST\n"
        "labels_used=false\n",
        run->source_identity_sha256, options->code_commit,
        run->producer_binary_sha256,
        contract->artifact_v1.profile_config_sha256,
        contract->artifact_v1.catalog_sha256,
        contract->run_config_sha256, central->checkpoint_sha256,
        contract->expected_identity.event_manifest_sha256,
        contract->expected_identity.tensor_manifest_sha256,
        contract->expected_identity.train_split_sha256,
        contract->expected_identity.validation_split_sha256,
        contract->train_subset_sha256,
        contract->validation_subset_sha256,
        central->train_matrix_sha256, central->validation_matrix_sha256,
        central->train_rows_sha256, central->validation_rows_sha256,
        VAE_C_IIC_PNG_CONTENT_IDENTITY_FORMAT_V1,
        PNG_TAR_SHA256);
    if (written < 0 || (size_t)written >= sizeof(preimage) ||
        !vae_c_iic_png_bind_run_preimage_v1(
            preimage, (size_t)written, run->png_sha256, digest) ||
        snprintf(run->run_id, sizeof(run->run_id),
                 "mini224_iic_v1_%s", digest) != 79) {
        return false;
    }
    written = snprintf(
        run->run_config, sizeof(run->run_config),
        "{\n"
        "  \"format\": \"MINI224_IIC_REAL_RUN_V1\",\n"
        "  \"run_id\": \"%s\",\n"
        "  \"source_identity_sha256\": \"%s\",\n"
        "  \"source_identity_role\": \"contract_identifier\",\n"
        "  \"producer_git_head\": \"%s\",\n"
        "  \"producer_state\": \"BINARY_SHA256_AUTHORITY\",\n"
        "  \"producer_binary_sha256\": \"%s\",\n"
        "  \"profile_id\": \"mini224_bridge_v1\",\n"
        "  \"profile_config_sha256\": \"%s\",\n"
        "  \"catalog_sha256\": \"%s\",\n"
        "  \"vae_run_config_sha256\": \"%s\",\n"
        "  \"vae_checkpoint_kind\": \"BEST\",\n"
        "  \"vae_checkpoint_epoch\": 99,\n"
        "  \"vae_checkpoint_step\": 5000,\n"
        "  \"vae_checkpoint_sha256\": \"%s\",\n"
        "  \"event_manifest_sha256\": \"%s\",\n"
        "  \"tensor_manifest_sha256\": \"%s\",\n"
        "  \"train_split_sha256\": \"%s\",\n"
        "  \"validation_split_sha256\": \"%s\",\n"
        "  \"train_subset_sha256\": \"%s\",\n"
        "  \"validation_subset_sha256\": \"%s\",\n"
        "  \"train_mu_sha256\": \"%s\",\n"
        "  \"validation_mu_sha256\": \"%s\",\n"
        "  \"train_rows_sha256\": \"%s\",\n"
        "  \"validation_rows_sha256\": \"%s\",\n"
        "  \"png_content_identity_format\": \"%s\",\n"
        "  \"png_content_identity_sha256\": \"%s\",\n"
        "  \"png_reference_tar_sha256\": \"%s\",\n"
        "  \"population_rows\": 1000,\n"
        "  \"population_order\": \"train800_then_validation200_native_split\",\n"
        "  \"central_view\": \"precomputed_posterior_mu\",\n"
        "  \"perturbed_view\": \"frozen_encoder_eval_posterior_mu\",\n"
        "  \"latent_dim\": 64,\n"
        "  \"augmentation_mode\": \"symmetric_figure\",\n"
        "  \"augmentation_seed\": 789,\n"
        "  \"shuffle_seed\": 101112,\n"
        "  \"logical_batch\": 128,\n"
        "  \"encoder_microbatch\": 16,\n"
        "  \"logical_batches_per_epoch\": 8,\n"
        "  \"max_optimizer_steps_per_pair\": 800,\n"
        "  \"C\": 36, \"W\": 250, \"K\": 5,\n"
        "  \"pair_seeds\": [123, 124, 125, 126, 127],\n"
        "  \"adam_lr\": 0.0005, \"adam_beta1\": 0.9,\n"
        "  \"adam_beta2\": 0.999, \"adam_epsilon\": 1e-8,\n"
        "  \"adam_weight_decay\": 0, \"adam_amsgrad\": false,\n"
        "  \"init_policy\": \"splitmix64_raw_seed_main_then_over_v1\",\n"
        "  \"max_epochs\": 100,\n"
        "  \"checkpoint_format\": \"IIC_V1\",\n"
        "  \"checkpoint_policy\": \"LATEST_EACH_EPOCH_FINAL_TERMINAL_NO_BEST\",\n"
        "  \"labels_used\": false\n"
        "}\n",
        run->run_id, run->source_identity_sha256, options->code_commit,
        run->producer_binary_sha256,
        contract->artifact_v1.profile_config_sha256,
        contract->artifact_v1.catalog_sha256,
        contract->run_config_sha256, central->checkpoint_sha256,
        contract->expected_identity.event_manifest_sha256,
        contract->expected_identity.tensor_manifest_sha256,
        contract->expected_identity.train_split_sha256,
        contract->expected_identity.validation_split_sha256,
        contract->train_subset_sha256,
        contract->validation_subset_sha256,
        central->train_matrix_sha256, central->validation_matrix_sha256,
        central->train_rows_sha256, central->validation_rows_sha256,
        VAE_C_IIC_PNG_CONTENT_IDENTITY_FORMAT_V1, run->png_sha256,
        PNG_TAR_SHA256);
    if (written < 0 || (size_t)written >= sizeof(run->run_config) ||
        vae_c_sha256_digest_hex_lower(run->run_config, (size_t)written,
                                      run->run_config_sha256) !=
            VAE_C_SHA256_OK) {
        return false;
    }
    vae_c_iic_checkpoint_identity_init(identity);
    identity->input_dim = VAE_C_IIC_MINI224_LATENT_DIM;
    identity->capacity = LOGICAL_BATCH;
    identity->batch_size = LOGICAL_BATCH;
    identity->train_rows = ROWS;
    identity->max_epochs = MAX_EPOCHS;
    identity->shuffle_seed = VAE_C_IIC_ENSEMBLE_SHUFFLE_SEED;
    for (pair = 0u; pair < VAE_C_IIC_ENSEMBLE_K; ++pair) {
        identity->seeds[pair] =
            VAE_C_IIC_ENSEMBLE_FIRST_SEED + (uint64_t)pair;
    }
    (void)strcpy(identity->run_id, run->run_id);
    (void)strcpy(identity->base_commit, options->code_commit);
    (void)strcpy(identity->producer_state, "BINARY_SHA256_AUTHORITY");
    (void)strcpy(identity->producer_binary_sha256,
                 run->producer_binary_sha256);
    return true;
}

static bool output_available(RunInput *run, const Options *options)
{
    char parent[PATH_CAP], partial[PATH_CAP], temp[PATH_CAP];
    int written;
    if (!join(parent, options->repository_root,
              "sakai_multires_c/results/vae_c/iic") ||
        !directory_exists(parent) ||
        !join(run->official_output_dir, parent, run->run_id)) {
        return false;
    }
    if (options->output_dir != NULL) {
#ifdef _WIN32
        if (_stricmp(options->output_dir, run->official_output_dir) != 0)
            return false;
#else
        if (strcmp(options->output_dir, run->official_output_dir) != 0)
            return false;
#endif
    }
    written = snprintf(partial, sizeof(partial), "%s.partial",
                       run->official_output_dir);
    if (written < 0 || (size_t)written >= sizeof(partial)) return false;
    written = snprintf(temp, sizeof(temp), "%s.tmp",
                       run->official_output_dir);
    if (written <= 0 || (size_t)written >= sizeof(temp) ||
        path_exists(partial) || path_exists(temp)) return false;
    if (options->official_run && options->official_resume)
        return directory_exists(run->official_output_dir);
    return !path_exists(run->official_output_dir);
}

static bool prepare(RunInput *run, const Options *options,
                    const char *executable)
{
    size_t png_count = 0u;
    size_t batches[BATCHES_PER_EPOCH], chunks[7];
    uint64_t max_steps;
    if (!vae_c_iic_mini224_perturbed_open(&run->input,
                                           options->repository_root) ||
        !source_valid(run) ||
        options->expected_population != run->input.central.row_count ||
        strcmp(options->expected_best_sha256,
               run->input.contract.checkpoint_sha256) != 0 ||
        !schedule(batches, chunks, &max_steps)) {
        return false;
    }
    if (!vae_c_iic_png_content_identity_v1(
            options->repository_root, &run->input.events,
            run->png_sha256, &png_count) ||
        png_count != VAE_C_IIC_MINI224_MANIFEST_ROWS *
                     GS_EVENT_VIEW_COUNT ||
        strcmp(run->png_sha256, PNG_SHA256) != 0 ||
        !build_identity(run, options, executable) ||
        !output_available(run, options)) {
        return false;
    }
    return run->checkpoint_identity.input_dim == 64u &&
        run->checkpoint_identity.capacity == LOGICAL_BATCH &&
        run->checkpoint_identity.batch_size == LOGICAL_BATCH &&
        run->checkpoint_identity.train_rows == ROWS &&
        run->checkpoint_identity.max_epochs == MAX_EPOCHS &&
        run->checkpoint_identity.shuffle_seed ==
            VAE_C_IIC_ENSEMBLE_SHUFFLE_SEED &&
        max_steps == 800u && chunks[6] == 8u;
}

/* SMOKE_CODE_START: no function above this marker performs a neural call. */
typedef struct {
    char text[32768];
    size_t used;
    bool valid;
} DigestBuilder;

static void digest_append(DigestBuilder *builder, const char *format, ...)
{
    va_list args;
    int written;
    if (!builder->valid || builder->used >= sizeof(builder->text)) return;
    va_start(args, format);
    written = vsnprintf(builder->text + builder->used,
                        sizeof(builder->text) - builder->used, format, args);
    va_end(args);
    if (written < 0 ||
        (size_t)written >= sizeof(builder->text) - builder->used) {
        builder->valid = false;
    } else {
        builder->used += (size_t)written;
    }
}

static void digest_float_blob(DigestBuilder *builder, const float *values,
                              size_t count)
{
    char sha[VAE_C_SHA256_HEX_SIZE];
    size_t index;
    if (!builder->valid || values == NULL || count > SIZE_MAX / sizeof(float)) {
        builder->valid = false;
        return;
    }
    for (index = 0u; index < count; ++index) {
        if (!isfinite(values[index])) {
            builder->valid = false;
            return;
        }
    }
    if (vae_c_sha256_digest_hex_lower(values, count * sizeof(float), sha) !=
        VAE_C_SHA256_OK) {
        builder->valid = false;
        return;
    }
    digest_append(builder, "%zu:%s\n", count, sha);
}

static bool ensemble_digests(const VAECIICEnsemble *ensemble,
                             char parameters[VAE_C_SHA256_HEX_SIZE],
                             char optimizer[VAE_C_SHA256_HEX_SIZE],
                             char batch_norm[VAE_C_SHA256_HEX_SIZE],
                             char combined[VAE_C_SHA256_HEX_SIZE])
{
    DigestBuilder p = {{0}, 0u, true};
    DigestBuilder o = {{0}, 0u, true};
    DigestBuilder b = {{0}, 0u, true};
    size_t pair, branch, item;
    char preimage[3u * VAE_C_SHA256_HEX_SIZE + 4u];
    if (vae_c_iic_ensemble_validate(ensemble) != VAE_C_OP_OK) return false;
    for (pair = 0u; pair < VAE_C_IIC_ENSEMBLE_K; ++pair) {
        const VAECIICPair *model = &ensemble->pairs[pair];
        const VAECAdam *adam = &ensemble->optimizers[pair];
        const VAECIICHead *heads[2] = {
            &model->main_head, &model->overcluster_head
        };
        digest_append(&o, "adam:%zu:%" PRIu64 ":%a:%a:%a:%a:%a:%a\n",
                      pair, adam->step, adam->learning_rate, adam->beta1,
                      adam->beta2, adam->adam_epsilon, adam->beta1_power,
                      adam->beta2_power);
        for (branch = 0u; branch < 2u; ++branch) {
            const VAECIICHead *head = heads[branch];
            const VAECParameter *parts[5] = {
                &head->linear1.weights, &head->batch_norm.gamma,
                &head->batch_norm.beta, &head->linear2.weights,
                &head->linear2.bias
            };
            digest_append(&p, "pair:%zu:branch:%zu\n", pair, branch);
            digest_append(&o, "pair:%zu:branch:%zu\n", pair, branch);
            digest_append(&b, "pair:%zu:branch:%zu:tracked:%" PRIu64 "\n",
                          pair, branch, head->num_batches_tracked);
            for (item = 0u; item < 5u; ++item) {
                digest_append(&p, "part:%zu\n", item);
                digest_float_blob(&p, parts[item]->values,
                                  parts[item]->shape.count);
                digest_append(&o, "part:%zu\n", item);
                digest_float_blob(&o, parts[item]->adam_m,
                                  parts[item]->shape.count);
                digest_float_blob(&o, parts[item]->adam_v,
                                  parts[item]->shape.count);
            }
            digest_float_blob(&b, head->batch_norm.running_mean,
                              VAE_C_IIC_HEAD_HIDDEN_DIM);
            digest_float_blob(&b, head->batch_norm.running_variance,
                              VAE_C_IIC_HEAD_HIDDEN_DIM);
        }
    }
    if (!p.valid || !o.valid || !b.valid ||
        vae_c_sha256_digest_hex_lower(p.text, p.used, parameters) !=
            VAE_C_SHA256_OK ||
        vae_c_sha256_digest_hex_lower(o.text, o.used, optimizer) !=
            VAE_C_SHA256_OK ||
        vae_c_sha256_digest_hex_lower(b.text, b.used, batch_norm) !=
            VAE_C_SHA256_OK) return false;
    if (snprintf(preimage, sizeof(preimage), "%s|%s|%s", parameters,
                 optimizer, batch_norm) != 194) return false;
    return vae_c_sha256_digest_hex_lower(preimage, strlen(preimage),
                                          combined) == VAE_C_SHA256_OK;
}

static bool monotonic_now(struct timespec *stamp)
{
#ifdef _WIN32
    LARGE_INTEGER counter, frequency;
    if (!QueryPerformanceCounter(&counter) ||
        !QueryPerformanceFrequency(&frequency) ||
        frequency.QuadPart <= 0) return false;
    stamp->tv_sec = (time_t)(counter.QuadPart / frequency.QuadPart);
    stamp->tv_nsec = (long)((counter.QuadPart % frequency.QuadPart) *
                            INT64_C(1000000000) / frequency.QuadPart);
    return true;
#else
    return clock_gettime(CLOCK_MONOTONIC, stamp) == 0;
#endif
}

static double seconds_between(const struct timespec *a,
                              const struct timespec *b)
{
    return (double)(b->tv_sec - a->tv_sec) +
        (double)(b->tv_nsec - a->tv_nsec) / 1.0e9;
}

static bool temporary_dir_safe(const char *directory)
{
    if (directory == NULL || !directory_exists(directory)) return false;
#ifdef _WIN32
    {
        char temp[PATH_CAP], output[PATH_CAP];
        DWORD temp_length = GetTempPathA(PATH_CAP, temp);
        DWORD output_length = GetFullPathNameA(directory,
                                                PATH_CAP, output, NULL);
        return temp_length > 0u && temp_length < PATH_CAP &&
            output_length > temp_length && output_length < PATH_CAP &&
            _strnicmp(temp, output, temp_length) == 0;
    }
#else
    return strncmp(directory, "/tmp/", 5u) == 0;
#endif
}

static bool valid_probabilities(const VAECIICHeadWorkspace *view,
                                size_t rows, size_t columns)
{
    size_t row, col;
    if (view->softmax_output.shape.rank != 2u ||
        view->softmax_output.shape.dims[0] != rows ||
        view->softmax_output.shape.dims[1] != columns ||
        view->softmax_storage.data == NULL) return false;
    for (row = 0u; row < rows; ++row) {
        double sum = 0.0;
        for (col = 0u; col < columns; ++col) {
            float value = view->softmax_storage.data[row * columns + col];
            if (!isfinite(value) || value < 0.0f) return false;
            sum += (double)value;
        }
        if (!isfinite(sum) || fabs(sum - 1.0) > 1.0e-4) return false;
    }
    return true;
}

typedef struct {
    uint64_t encoder_calls;
    double loader_encoder_seconds;
    double iic_seconds;
    size_t batches;
    size_t samples;
    double main_mi[VAE_C_IIC_ENSEMBLE_K];
    double over_mi[VAE_C_IIC_ENSEMBLE_K];
    double total_loss[VAE_C_IIC_ENSEMBLE_K];
    char crop_hashes[BATCHES_PER_EPOCH][VAE_C_SHA256_HEX_SIZE];
} SmokeMetrics;

static bool smoke_batch(RunInput *run, VAECIICEnsemble *ensemble,
                        const size_t *selection, size_t count,
                        uint64_t epoch, size_t batch_index,
                        SmokeMetrics *metrics)
{
    const size_t dims[2] = {count, VAE_C_IIC_MINI224_LATENT_DIM};
    VAECShape shape;
    VAECTensor a, b;
    VAECIICMini224PerturbedRow rows[LOGICAL_BATCH];
    VAECIICMini224PerturbedReport report;
    struct timespec start, encoded, trained;
    char crop_text[16384], crop_sha[VAE_C_SHA256_HEX_SIZE];
    float before_b[LOGICAL_BATCH * VAE_C_IIC_MINI224_LATENT_DIM];
    size_t crop_used = 0u, row, pair;
    uint64_t previous_step;
    bool success = false;
    vae_c_tensor_init(&a);
    vae_c_tensor_init(&b);
    memset(&report, 0, sizeof(report));
    if (count < 2u || count > LOGICAL_BATCH ||
        batch_index >= BATCHES_PER_EPOCH ||
        vae_c_shape_make(2u, dims, &shape) != VAE_C_STORAGE_OK ||
        vae_c_tensor_allocate(&a, &shape, true) != VAE_C_STORAGE_OK ||
        vae_c_tensor_allocate(&b, &shape, true) != VAE_C_STORAGE_OK ||
        !monotonic_now(&start)) goto done;
    for (row = 0u; row < count; ++row) {
        const float *mu = vae_c_iic_mini224_central_row_mu(
            &run->input.central, selection[row]);
        if (mu == NULL) goto done;
        memcpy(a.data + row * VAE_C_IIC_MINI224_LATENT_DIM, mu,
               VAE_C_IIC_MINI224_LATENT_DIM * sizeof(float));
    }
    if (!vae_c_iic_mini224_perturbed_compute(
            &run->input, selection, count, epoch, ENCODER_MICROBATCH,
            b.data, rows, &report) ||
        report.samples != count ||
        report.encoder_forward_calls !=
            (uint64_t)((count + ENCODER_MICROBATCH - 1u) /
                       ENCODER_MICROBATCH) ||
        !report.inputs_bitwise_unchanged ||
        !report.model_parameters_bn_bitwise_unchanged ||
        !report.decoder_workspace_untouched ||
        !report.optimizer_unchanged || !monotonic_now(&encoded)) goto done;
    for (row = 0u; row < count; ++row) {
        int expected_crop;
        int written;
        if (rows[row].identity.iic_row_index != selection[row] ||
            rows[row].identity.manifest_index !=
                run->input.canonical_manifest_indices[selection[row]] ||
            rows[row].epoch != epoch ||
            gs_augment_crop_left_for_event_mode(
                VAE_C_IIC_MINI224_AUGMENTATION_SEED, epoch,
                (uint64_t)rows[row].identity.manifest_index,
                GS_AUGMENT_MODE_SYMMETRIC_FIGURE, &expected_crop) != 0 ||
            expected_crop != rows[row].crop_left) goto done;
        written = snprintf(crop_text + crop_used,
                           sizeof(crop_text) - crop_used,
                           "%zu|%u|%d\n", selection[row],
                           rows[row].identity.manifest_index,
                           rows[row].crop_left);
        if (written < 0 ||
            (size_t)written >= sizeof(crop_text) - crop_used) goto done;
        crop_used += (size_t)written;
    }
    if (vae_c_sha256_digest_hex_lower(crop_text, crop_used, crop_sha) !=
        VAE_C_SHA256_OK) goto done;
    memcpy(metrics->crop_hashes[batch_index], crop_sha,
           VAE_C_SHA256_HEX_SIZE);
    memcpy(before_b, b.data, count * VAE_C_IIC_MINI224_LATENT_DIM *
           sizeof(float));
    previous_step = ensemble->optimizers[0].step;
    if (vae_c_iic_ensemble_train_step(ensemble, &a, &b) != VAE_C_OP_OK ||
        !monotonic_now(&trained)) goto done;
    for (row = 0u; row < count; ++row) {
        const float *mu = vae_c_iic_mini224_central_row_mu(
            &run->input.central, selection[row]);
        if (mu == NULL ||
            memcmp(a.data + row * VAE_C_IIC_MINI224_LATENT_DIM, mu,
                   VAE_C_IIC_MINI224_LATENT_DIM * sizeof(float)) != 0)
            goto done;
    }
    if (memcmp(before_b, b.data,
               count * VAE_C_IIC_MINI224_LATENT_DIM * sizeof(float)) != 0)
        goto done;
    for (pair = 0u; pair < VAE_C_IIC_ENSEMBLE_K; ++pair) {
        const VAECIICPairWorkspace *workspace = &ensemble->workspaces[pair];
        const VAECIICPair *model = &ensemble->pairs[pair];
        if (ensemble->optimizers[pair].step != previous_step + 1u ||
            model->main_head.num_batches_tracked !=
                2u * (previous_step + 1u) ||
            model->overcluster_head.num_batches_tracked !=
                2u * (previous_step + 1u) ||
            workspace->batch_size != 0u ||
            !isfinite(workspace->main_result.mutual_information) ||
            !isfinite(workspace->over_result.mutual_information) ||
            !isfinite(workspace->main_result.loss) ||
            !isfinite(workspace->over_result.loss) ||
            !isfinite(workspace->total_loss) ||
            workspace->total_loss != workspace->main_result.loss +
                                     workspace->over_result.loss ||
            !valid_probabilities(&workspace->main_a, count, 36u) ||
            !valid_probabilities(&workspace->main_b, count, 36u) ||
            !valid_probabilities(&workspace->over_a, count, 250u) ||
            !valid_probabilities(&workspace->over_b, count, 250u)) goto done;
        metrics->main_mi[pair] +=
            (double)workspace->main_result.mutual_information;
        metrics->over_mi[pair] +=
            (double)workspace->over_result.mutual_information;
        metrics->total_loss[pair] += (double)workspace->total_loss;
    }
    printf("SMOKE_BATCH_INDEX=%zu EPOCH=%" PRIu64 " LOGICAL_BATCH_SIZE=%zu "
           "FIRST_IIC_ROW=%zu LAST_IIC_ROW=%zu "
           "FIRST_MANIFEST_INDEX=%u LAST_MANIFEST_INDEX=%u "
           "FIRST_CROP=%d LAST_CROP=%d CROP_MAP_SHA256=%s "
           "ENCODER_FORWARD_CALLS=%" PRIu64 "\n",
           batch_index, epoch, count, selection[0], selection[count - 1u],
           rows[0].identity.manifest_index,
           rows[count - 1u].identity.manifest_index,
           rows[0].crop_left, rows[count - 1u].crop_left,
           crop_sha, report.encoder_forward_calls);
    for (pair = 0u; pair < VAE_C_IIC_ENSEMBLE_K; ++pair) {
        const VAECIICPairWorkspace *workspace = &ensemble->workspaces[pair];
        printf("SMOKE_BATCH=%zu PAIR=%zu MAIN_MI=%.9g OVER_MI=%.9g "
               "MAIN_LOSS=%.9g OVER_LOSS=%.9g TOTAL_LOSS=%.9g "
               "OPTIMIZER_STEP=%" PRIu64 " BN_MAIN=%" PRIu64
               " BN_OVER=%" PRIu64 "\n",
               batch_index, pair,
               workspace->main_result.mutual_information,
               workspace->over_result.mutual_information,
               workspace->main_result.loss, workspace->over_result.loss,
               workspace->total_loss, ensemble->optimizers[pair].step,
               ensemble->pairs[pair].main_head.num_batches_tracked,
               ensemble->pairs[pair].overcluster_head.num_batches_tracked);
    }
    metrics->encoder_calls += report.encoder_forward_calls;
    metrics->loader_encoder_seconds += seconds_between(&start, &encoded);
    metrics->iic_seconds += seconds_between(&encoded, &trained);
    ++metrics->batches;
    metrics->samples += count;
    success = true;
done:
    vae_c_tensor_free(&b);
    vae_c_tensor_free(&a);
    return success;
}

static bool run_smoke(RunInput *run, const Options *options)
{
    VAECIICEnsemble ensemble;
    SmokeMetrics metrics = {0};
    size_t permutation[ROWS];
    char initial_parameters[VAE_C_SHA256_HEX_SIZE];
    char initial_optimizer[VAE_C_SHA256_HEX_SIZE];
    char initial_bn[VAE_C_SHA256_HEX_SIZE];
    char initial_combined[VAE_C_SHA256_HEX_SIZE];
    char final_parameters[VAE_C_SHA256_HEX_SIZE];
    char final_optimizer[VAE_C_SHA256_HEX_SIZE];
    char final_bn[VAE_C_SHA256_HEX_SIZE];
    char final_combined[VAE_C_SHA256_HEX_SIZE];
    struct timespec start, finish;
    size_t pair;
    bool success = false;
    vae_c_iic_ensemble_init(&ensemble);
    if (!temporary_dir_safe(options->smoke_output_dir) ||
        vae_c_shuffle_make_permutation(
            ROWS, VAE_C_IIC_ENSEMBLE_SHUFFLE_SEED, 0u,
            permutation) != VAE_C_RNG_OK ||
        vae_c_iic_ensemble_allocate_v1(
            &ensemble, VAE_C_IIC_MINI224_LATENT_DIM,
            LOGICAL_BATCH) != VAE_C_OP_OK ||
        !ensemble_digests(&ensemble, initial_parameters,
                          initial_optimizer, initial_bn, initial_combined) ||
        !monotonic_now(&start) ||
        !smoke_batch(run, &ensemble, &permutation[0], 128u, 0u, 0u,
                     &metrics) ||
        !smoke_batch(run, &ensemble, &permutation[896], 104u, 0u, 7u,
                     &metrics) ||
        !monotonic_now(&finish) || metrics.batches != 2u ||
        metrics.samples != 232u || metrics.encoder_calls != 15u ||
        !ensemble_digests(&ensemble, final_parameters,
                          final_optimizer, final_bn, final_combined) ||
        strcmp(initial_parameters, final_parameters) == 0 ||
        strcmp(initial_optimizer, final_optimizer) == 0) goto done;
    for (pair = 0u; pair < VAE_C_IIC_ENSEMBLE_K; ++pair) {
        if (ensemble.optimizers[pair].step != 2u ||
            ensemble.pairs[pair].main_head.num_batches_tracked != 4u ||
            ensemble.pairs[pair].overcluster_head.num_batches_tracked != 4u)
            goto done;
    }
    printf("SMOKE_OUTPUT_DIR=%s\n", options->smoke_output_dir);
    printf("SMOKE_PARAMETER_VALUES_SHA256=%s\n", final_parameters);
    printf("SMOKE_OPTIMIZER_STATE_SHA256=%s\n", final_optimizer);
    printf("SMOKE_BATCH_NORM_STATE_SHA256=%s\n", final_bn);
    printf("SMOKE_FINAL_STATE_SHA256=%s\n", final_combined);
    puts("SMOKE_BATCHES=2\nSMOKE_SAMPLES=232\nSMOKE_EPOCH=0");
    puts("VAE_ENCODER_FORWARD_CALLS=15\nDECODER_CALLS=0");
    puts("IIC_HEAD_FORWARD_CALLS=40\nIIC_HEAD_BACKWARD_CALLS=40");
    puts("ADAM_STEPS_TOTAL=10\nOPTIMIZER_STEPS_PER_PAIR=2");
    puts("IIC_BN_UPDATES_PER_HEAD=4\nLATEST_CREATED=false");
    puts("FINAL_CREATED=false\nOUTPUT_CREATED=false");
    printf("SMOKE_ELAPSED_SECONDS=%.6f\n", seconds_between(&start, &finish));
    printf("SMOKE_LOADER_ENCODER_SECONDS=%.6f\n",
           metrics.loader_encoder_seconds);
    printf("SMOKE_IIC_SECONDS=%.6f\n", metrics.iic_seconds);
#ifdef _WIN32
    {
        PROCESS_MEMORY_COUNTERS_EX memory;
        memset(&memory, 0, sizeof(memory));
        memory.cb = sizeof(memory);
        if (GetProcessMemoryInfo(GetCurrentProcess(),
                (PROCESS_MEMORY_COUNTERS *)&memory, sizeof(memory)) != 0) {
            printf("PEAK_WORKING_SET_BYTES=%zu\n",
                   (size_t)memory.PeakWorkingSetSize);
            printf("PRIVATE_BYTES_AT_END=%zu\n",
                   (size_t)memory.PrivateUsage);
        }
    }
#endif
    puts("SMOKE_REAL_TWO_BATCH_PASS=true");
    success = true;
done:
    vae_c_iic_ensemble_free(&ensemble);
    return success;
}

/* MINI224-IIC-2C: a bounded full-epoch gate. It uses the normal IIC V1
 * checkpoint API, but can write only into a pre-existing external temp dir.
 * The scientific max_epochs and run identity never depend on the stop bound. */
#define DEBUG_METRICS_HEADER \
    "epoch,batches,samples,step0,step1,step2,step3,step4," \
    "main0,main1,main2,main3,main4," \
    "over0,over1,over2,over3,over4," \
    "total0,total1,total2,total3,total4\n"

static bool debug_write_new(const char *path, const char *text)
{
    FILE *file = fopen(path, "wbx");
    size_t length = strlen(text);
    bool ok;
    if (file == NULL) return false;
    ok = fwrite(text, 1u, length, file) == length && fflush(file) == 0;
    if (fclose(file) != 0) ok = false;
    return ok;
}

static bool debug_file_equals(const char *path, const char *text)
{
    FILE *file = fopen(path, "rb");
    size_t length = strlen(text);
    char *buffer;
    bool ok;
    if (file == NULL || length == SIZE_MAX) {
        if (file != NULL) (void)fclose(file);
        return false;
    }
    buffer = (char *)malloc(length + 1u);
    if (buffer == NULL) { (void)fclose(file); return false; }
    ok = fread(buffer, 1u, length + 1u, file) == length &&
        memcmp(buffer, text, length) == 0 && !ferror(file);
    if (fclose(file) != 0) ok = false;
    free(buffer);
    return ok;
}

static bool debug_metric_rows(const char *path, uint64_t expected)
{
    FILE *file = fopen(path, "rb");
    char line[4096];
    uint64_t row = 0u;
    bool ok = false;
    if (file == NULL) return false;
    if (fgets(line, sizeof(line), file) == NULL ||
        strcmp(line, DEBUG_METRICS_HEADER) != 0) goto done;
    while (fgets(line, sizeof(line), file) != NULL) {
        uint64_t epoch;
        int consumed = 0;
        if (sscanf(line, "%" SCNu64 ",8,1000,%*u,%*u,%*u,%*u,%*u,%n",
                   &epoch, &consumed) != 1 || consumed == 0 ||
            epoch != row || strlen(line) < (size_t)consumed + 2u ||
            line[strlen(line) - 1u] != '\n') goto done;
        ++row;
    }
    ok = !ferror(file) && row == expected;
done:
    if (fclose(file) != 0) ok = false;
    return ok;
}

static bool debug_append_metrics(const char *path, uint64_t epoch,
                                 const SmokeMetrics *metrics,
                                 const VAECIICEnsemble *ensemble)
{
    FILE *file = fopen(path, "ab");
    size_t pair;
    bool ok = file != NULL && metrics->batches == BATCHES_PER_EPOCH &&
        metrics->samples == ROWS;
    if (!ok) { if (file != NULL) (void)fclose(file); return false; }
    if (fprintf(file, "%" PRIu64 ",%zu,%zu", epoch, metrics->batches,
                metrics->samples) < 0) ok = false;
    for (pair = 0u; ok && pair < VAE_C_IIC_ENSEMBLE_K; ++pair)
        if (fprintf(file, ",%" PRIu64,
                    ensemble->optimizers[pair].step) < 0) ok = false;
    for (pair = 0u; ok && pair < VAE_C_IIC_ENSEMBLE_K; ++pair)
        if (fprintf(file, ",%.17g", metrics->main_mi[pair]) < 0)
            ok = false;
    for (pair = 0u; ok && pair < VAE_C_IIC_ENSEMBLE_K; ++pair)
        if (fprintf(file, ",%.17g", metrics->over_mi[pair]) < 0)
            ok = false;
    for (pair = 0u; ok && pair < VAE_C_IIC_ENSEMBLE_K; ++pair)
        if (fprintf(file, ",%.17g", metrics->total_loss[pair]) < 0)
            ok = false;
    if (ok && (fputc('\n', file) == EOF || fflush(file) != 0))
        ok = false;
    if (fclose(file) != 0) ok = false;
    return ok;
}

static bool debug_epoch(RunInput *run, VAECIICEnsemble *ensemble,
                        uint64_t epoch, const char *metrics_path)
{
    size_t permutation[ROWS], batch;
    SmokeMetrics metrics = {0};
    char permutation_sha[VAE_C_SHA256_HEX_SIZE];
    char crop_sha[VAE_C_SHA256_HEX_SIZE];
    char crop_text[BATCHES_PER_EPOCH * (VAE_C_SHA256_HEX_SIZE - 1u)];
    struct timespec start, finish;
    size_t pair;
    if (vae_c_shuffle_make_permutation(
            ROWS, VAE_C_IIC_ENSEMBLE_SHUFFLE_SEED, epoch,
            permutation) != VAE_C_RNG_OK ||
        vae_c_sha256_digest_hex_lower(permutation, sizeof(permutation),
                                      permutation_sha) != VAE_C_SHA256_OK ||
        !monotonic_now(&start)) return false;
    for (batch = 0u; batch < BATCHES_PER_EPOCH; ++batch) {
        size_t offset = batch * LOGICAL_BATCH;
        size_t count = ROWS - offset;
        if (count > LOGICAL_BATCH) count = LOGICAL_BATCH;
        if (!smoke_batch(run, ensemble, &permutation[offset], count,
                         epoch, batch, &metrics)) return false;
        memcpy(crop_text + batch * (VAE_C_SHA256_HEX_SIZE - 1u),
               metrics.crop_hashes[batch], VAE_C_SHA256_HEX_SIZE - 1u);
    }
    if (!monotonic_now(&finish) || metrics.batches != BATCHES_PER_EPOCH ||
        metrics.samples != ROWS || metrics.encoder_calls != 63u ||
        vae_c_sha256_digest_hex_lower(crop_text, sizeof(crop_text),
                                      crop_sha) != VAE_C_SHA256_OK ||
        !debug_append_metrics(metrics_path, epoch, &metrics, ensemble))
        return false;
    for (pair = 0u; pair < VAE_C_IIC_ENSEMBLE_K; ++pair) {
        if (ensemble->optimizers[pair].step !=
                (epoch + 1u) * BATCHES_PER_EPOCH ||
            ensemble->pairs[pair].main_head.num_batches_tracked !=
                2u * (epoch + 1u) * BATCHES_PER_EPOCH ||
            ensemble->pairs[pair].overcluster_head.num_batches_tracked !=
                2u * (epoch + 1u) * BATCHES_PER_EPOCH) return false;
    }
    printf("DEBUG_EPOCH=%" PRIu64 " BATCHES=%zu SAMPLES=%zu "
           "ENCODER_CALLS=%" PRIu64 " EPOCH_SECONDS=%.6f "
           "LOADER_ENCODER_SECONDS=%.6f IIC_SECONDS=%.6f\n",
           epoch, metrics.batches, metrics.samples, metrics.encoder_calls,
           seconds_between(&start, &finish), metrics.loader_encoder_seconds,
           metrics.iic_seconds);
    printf("DEBUG_EPOCH=%" PRIu64 " PERMUTATION_SHA256=%s "
           "CROP_MAP_SHA256=%s\n", epoch, permutation_sha, crop_sha);
    return true;
}

static bool run_debug_epoch_gate(RunInput *run, const Options *options)
{
    char config_path[PATH_CAP], metrics_path[PATH_CAP];
    char latest_path[PATH_CAP], final_path[PATH_CAP];
    char parameters[VAE_C_SHA256_HEX_SIZE];
    char optimizer[VAE_C_SHA256_HEX_SIZE];
    char batch_norm[VAE_C_SHA256_HEX_SIZE];
    char combined[VAE_C_SHA256_HEX_SIZE];
    char latest_sha[VAE_C_SHA256_HEX_SIZE];
    VAECIICCheckpointMetadata metadata;
    VAECIICCheckpointStatus status;
    VAECIICEnsemble ensemble;
    uint64_t epoch, start_epoch = 0u;
    size_t pair;
    bool ok = false;
    vae_c_iic_ensemble_init(&ensemble);
    vae_c_iic_checkpoint_metadata_init(&metadata);
    if (!temporary_dir_safe(options->debug_output_dir) ||
        !join(config_path, options->debug_output_dir, "run_config.json") ||
        !join(metrics_path, options->debug_output_dir, "metrics.csv") ||
        !join(latest_path, options->debug_output_dir, "latest.ckpt") ||
        !join(final_path, options->debug_output_dir, "final.ckpt") ||
        path_exists(final_path) ||
        vae_c_iic_ensemble_allocate_v1(&ensemble,
            VAE_C_IIC_MINI224_LATENT_DIM, LOGICAL_BATCH) != VAE_C_OP_OK)
        goto done;
    if (options->debug_resume) {
        if (options->debug_stop_after_epoch != 1u ||
            !debug_file_equals(config_path, run->run_config) ||
            !path_exists(latest_path)) goto done;
        status = vae_c_iic_checkpoint_load(
            latest_path, &run->checkpoint_identity, &ensemble, &metadata);
        if (status != VAE_C_IIC_CHECKPOINT_OK ||
            metadata.kind != VAE_C_IIC_CHECKPOINT_LATEST ||
            metadata.last_completed_epoch != 0u ||
            !debug_metric_rows(metrics_path, 1u)) goto done;
        start_epoch = metadata.last_completed_epoch + 1u;
        puts("RESUME_SOURCE=LATEST\nLAST_COMPLETED_EPOCH=0"
             "\nRESUME_START_EPOCH=1");
    } else {
        if (path_exists(config_path) || path_exists(metrics_path) ||
            path_exists(latest_path) ||
            !debug_write_new(config_path, run->run_config) ||
            !debug_write_new(metrics_path, DEBUG_METRICS_HEADER)) goto done;
        puts("RESUME_SOURCE=NONE\nSTART_EPOCH=0");
    }
    for (epoch = start_epoch; epoch <= options->debug_stop_after_epoch;
         ++epoch) {
        if (!debug_epoch(run, &ensemble, epoch, metrics_path)) goto done;
        metadata.kind = VAE_C_IIC_CHECKPOINT_LATEST;
        metadata.last_completed_epoch = epoch;
        status = vae_c_iic_checkpoint_save_atomic(
            latest_path, &run->checkpoint_identity, &metadata, &ensemble);
        if (status != VAE_C_IIC_CHECKPOINT_OK ||
            !ensemble_digests(&ensemble, parameters, optimizer,
                              batch_norm, combined) ||
            vae_c_mu_file_sha256(latest_path, latest_sha) != VAE_C_MU_OK ||
            !debug_metric_rows(metrics_path, epoch + 1u)) goto done;
        printf("DEBUG_EPOCH=%" PRIu64 " PARAMETERS_SHA256=%s "
               "BN_STATE_SHA256=%s ADAM_STATE_SHA256=%s "
               "FULL_SCIENTIFIC_STATE_SHA256=%s LATEST_SHA256=%s\n",
               epoch, parameters, batch_norm, optimizer, combined,
               latest_sha);
    }
    for (pair = 0u; pair < VAE_C_IIC_ENSEMBLE_K; ++pair)
        printf("PAIR=%zu OPTIMIZER_STEP=%" PRIu64 " BN_MAIN=%" PRIu64
               " BN_OVER=%" PRIu64 "\n", pair,
               ensemble.optimizers[pair].step,
               ensemble.pairs[pair].main_head.num_batches_tracked,
               ensemble.pairs[pair].overcluster_head.num_batches_tracked);
    puts("DEBUG_GATE_COMPLETE=true\nFINAL_CREATED=false");
    ok = true;
done:
    vae_c_iic_ensemble_free(&ensemble);
    return ok;
}

/* mini112 writes metrics, then LATEST, then FINAL on normal epoch 99.
 * If interrupted after terminal LATEST but before FINAL, resume loads
 * last_completed_epoch=99, for which 99 >= max_epochs(100) is false;
 * start_epoch=100 then finalizes without another optimizer step.
 * mini224 preserves the same terminal-recovery contract. */
#define OFFICIAL_METRICS_HEADER \
    "epoch,elapsed_seconds,batches,samples," \
    "optimizer_step_c0,optimizer_step_c1,optimizer_step_c2," \
    "optimizer_step_c3,optimizer_step_c4," \
    "mean_batch_main_mi_c0,mean_batch_main_mi_c1," \
    "mean_batch_main_mi_c2,mean_batch_main_mi_c3," \
    "mean_batch_main_mi_c4," \
    "mean_batch_overcluster_mi_c0,mean_batch_overcluster_mi_c1," \
    "mean_batch_overcluster_mi_c2,mean_batch_overcluster_mi_c3," \
    "mean_batch_overcluster_mi_c4," \
    "mean_batch_total_loss_c0,mean_batch_total_loss_c1," \
    "mean_batch_total_loss_c2,mean_batch_total_loss_c3," \
    "mean_batch_total_loss_c4\n"

typedef struct {
    char config[PATH_CAP], metrics[PATH_CAP], latest[PATH_CAP];
    char final[PATH_CAP];
} OfficialPaths;

static bool official_paths(const RunInput *run, OfficialPaths *paths)
{
    return join(paths->config, run->official_output_dir, "run_config.json") &&
        join(paths->metrics, run->official_output_dir, "metrics.csv") &&
        join(paths->latest, run->official_output_dir, "latest.ckpt") &&
        join(paths->final, run->official_output_dir, "final.ckpt");
}

static bool official_disk_free(const char *directory, uint64_t *bytes)
{
#ifdef _WIN32
    ULARGE_INTEGER available;
    if (!GetDiskFreeSpaceExA(directory, &available, NULL, NULL)) return false;
    *bytes = (uint64_t)available.QuadPart;
#else
    struct statvfs filesystem;
    if (statvfs(directory, &filesystem) != 0 ||
        filesystem.f_bavail > UINT64_MAX / filesystem.f_frsize)
        return false;
    *bytes = (uint64_t)filesystem.f_bavail * filesystem.f_frsize;
#endif
    /* Approximately 13 MB per checkpoint; reserve ample space for two
     * checkpoints, their atomic siblings, config and metrics. */
    return *bytes >= UINT64_C(268435456);
}

static bool official_inventory(const RunInput *run, bool *latest, bool *final)
{
    OfficialPaths paths;
    if (!official_paths(run, &paths) ||
        !directory_exists(run->official_output_dir)) return false;
    *latest = path_exists(paths.latest);
    *final = path_exists(paths.final);
    if (!path_exists(paths.config) ||
        (*latest && !path_exists(paths.metrics)))
        return false;
#ifdef _WIN32
    {
        char pattern[PATH_CAP];
        WIN32_FIND_DATAA found;
        HANDLE handle;
        if (!join(pattern, run->official_output_dir, "*")) return false;
        handle = FindFirstFileA(pattern, &found);
        if (handle == INVALID_HANDLE_VALUE) return false;
        do {
            const char *name = found.cFileName;
            if (strcmp(name, ".") == 0 || strcmp(name, "..") == 0)
                continue;
            if ((found.dwFileAttributes &
                 (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)) ||
                (strcmp(name, "run_config.json") != 0 &&
                 strcmp(name, "metrics.csv") != 0 &&
                 strcmp(name, "latest.ckpt") != 0 &&
                 strcmp(name, "final.ckpt") != 0)) {
                (void)FindClose(handle);
                return false;
            }
        } while (FindNextFileA(handle, &found));
        if (GetLastError() != ERROR_NO_MORE_FILES) {
            (void)FindClose(handle);
            return false;
        }
        (void)FindClose(handle);
    }
#else
    {
        DIR *directory = opendir(run->official_output_dir);
        struct dirent *entry;
        if (directory == NULL) return false;
        errno = 0;
        while ((entry = readdir(directory)) != NULL) {
            const char *name = entry->d_name;
            if (strcmp(name, ".") != 0 && strcmp(name, "..") != 0 &&
                strcmp(name, "run_config.json") != 0 &&
                strcmp(name, "metrics.csv") != 0 &&
                strcmp(name, "latest.ckpt") != 0 &&
                strcmp(name, "final.ckpt") != 0) {
                (void)closedir(directory);
                return false;
            }
        }
        if (errno != 0) { (void)closedir(directory); return false; }
        (void)closedir(directory);
    }
#endif
    return true;
}

static bool official_write_atomic(const char *path, const char *text)
{
    char partial[PATH_CAP];
    FILE *file;
    size_t length = strlen(text);
    int written = snprintf(partial, sizeof(partial), "%s.partial", path);
    bool ok;
    if (written <= 0 || (size_t)written >= sizeof(partial) ||
        path_exists(partial)) return false;
    file = fopen(partial, "wbx");
    if (file == NULL) return false;
    ok = fwrite(text, 1u, length, file) == length && fflush(file) == 0;
    if (fclose(file) != 0) ok = false;
    if (!ok) return false;
#ifdef _WIN32
    return MoveFileExA(partial, path, MOVEFILE_REPLACE_EXISTING) != 0;
#else
    return rename(partial, path) == 0;
#endif
}

static bool official_metric_row(const char *line, uint64_t epoch)
{
    const char *cursor = line;
    double value[24];
    size_t column;
    for (column = 0u; column < 24u; ++column) {
        char *end;
        errno = 0;
        value[column] = strtod(cursor, &end);
        if (errno != 0 || end == cursor || !isfinite(value[column]))
            return false;
        if (column == 23u) {
            if (*end != '\n' || end[1] != '\0') return false;
        } else if (*end != ',') return false;
        cursor = end + 1;
    }
    if (value[0] != (double)epoch || value[1] < 0.0 ||
        value[2] != (double)BATCHES_PER_EPOCH ||
        value[3] != (double)ROWS) return false;
    for (column = 4u; column < 9u; ++column)
        if (value[column] != (double)((epoch + 1u) * BATCHES_PER_EPOCH))
            return false;
    return true;
}

static bool official_future_epoch(const char *line, uint64_t epoch)
{
    char *end;
    unsigned long long parsed;
    if (line[0] == '+' || line[0] == '-') return false;
    errno = 0;
    parsed = strtoull(line, &end, 10);
    return errno == 0 && end != line &&
        (*end == ',' || *end == '\n' || *end == '\0') &&
        (uint64_t)parsed > epoch;
}

/* Validate everything before mutating metrics. Only future, non-authoritative
 * rows may be truncated. The last committed row must be complete. */
static bool official_reconcile_metrics(const char *path, bool has_latest,
                                       uint64_t committed_epoch)
{
    FILE *file = fopen(path, "rb");
    char line[4096];
    char *retained = NULL;
    size_t used = 0u, capacity = 4096u * 101u;
    uint64_t expected = 0u;
    bool future = false, ok = false;
    if (file == NULL) return false;
    retained = (char *)malloc(capacity);
    if (retained == NULL || fgets(line, sizeof(line), file) == NULL ||
        strcmp(line, OFFICIAL_METRICS_HEADER) != 0) goto done;
    used = strlen(OFFICIAL_METRICS_HEADER);
    memcpy(retained, OFFICIAL_METRICS_HEADER, used);
    while (fgets(line, sizeof(line), file) != NULL) {
        size_t length = strlen(line);
        if (length == 0u || length == sizeof(line) - 1u) goto done;
        if (!has_latest) { future = true; continue; }
        if (expected > committed_epoch) {
            if (!official_future_epoch(line, committed_epoch)) goto done;
            future = true;
            continue;
        }
        if (!official_metric_row(line, expected) ||
            used > capacity - length - 1u) goto done;
        memcpy(retained + used, line, length);
        used += length;
        ++expected;
    }
    if (ferror(file) || (has_latest && expected != committed_epoch + 1u))
        goto done;
    if (fclose(file) != 0) { file = NULL; goto done; }
    file = NULL;
    retained[used] = '\0';
    ok = !future || official_write_atomic(path, retained);
done:
    if (file != NULL) (void)fclose(file);
    free(retained);
    return ok;
}

static bool official_append_metrics(const char *path, uint64_t epoch,
                                    double elapsed, const SmokeMetrics *metrics,
                                    const VAECIICEnsemble *ensemble)
{
    FILE *file = fopen(path, "ab");
    size_t pair;
    bool ok = file != NULL && isfinite(elapsed) && elapsed >= 0.0 &&
        metrics->batches == BATCHES_PER_EPOCH && metrics->samples == ROWS;
    if (!ok) { if (file != NULL) (void)fclose(file); return false; }
    if (fprintf(file, "%" PRIu64 ",%.9f,%zu,%zu", epoch, elapsed,
                metrics->batches, metrics->samples) < 0) ok = false;
    for (pair = 0u; ok && pair < VAE_C_IIC_ENSEMBLE_K; ++pair)
        if (fprintf(file, ",%" PRIu64,
                    ensemble->optimizers[pair].step) < 0) ok = false;
    for (pair = 0u; ok && pair < VAE_C_IIC_ENSEMBLE_K; ++pair)
        if (fprintf(file, ",%.17g", metrics->main_mi[pair] /
                    BATCHES_PER_EPOCH) < 0) ok = false;
    for (pair = 0u; ok && pair < VAE_C_IIC_ENSEMBLE_K; ++pair)
        if (fprintf(file, ",%.17g", metrics->over_mi[pair] /
                    BATCHES_PER_EPOCH) < 0) ok = false;
    for (pair = 0u; ok && pair < VAE_C_IIC_ENSEMBLE_K; ++pair)
        if (fprintf(file, ",%.17g", metrics->total_loss[pair] /
                    BATCHES_PER_EPOCH) < 0) ok = false;
    if (ok && (fputc('\n', file) == EOF || fflush(file) != 0)) ok = false;
    if (fclose(file) != 0) ok = false;
    return ok;
}

static bool official_epoch(RunInput *run, VAECIICEnsemble *ensemble,
                           uint64_t epoch, const char *metrics_path)
{
    size_t permutation[ROWS], batch;
    SmokeMetrics metrics = {0};
    struct timespec start, finish;
    if (epoch >= MAX_EPOCHS ||
        vae_c_shuffle_make_permutation(
            ROWS, VAE_C_IIC_ENSEMBLE_SHUFFLE_SEED, epoch,
            permutation) != VAE_C_RNG_OK || !monotonic_now(&start))
        return false;
    for (batch = 0u; batch < BATCHES_PER_EPOCH; ++batch) {
        size_t offset = batch * LOGICAL_BATCH;
        size_t count = ROWS - offset;
        if (count > LOGICAL_BATCH) count = LOGICAL_BATCH;
        if (!smoke_batch(run, ensemble, &permutation[offset], count,
                         epoch, batch, &metrics)) return false;
    }
    if (!monotonic_now(&finish) ||
        !official_append_metrics(metrics_path, epoch,
            seconds_between(&start, &finish), &metrics, ensemble))
        return false;
    printf("EPOCH=%" PRIu64 " BATCHES=%zu SAMPLES=%zu STEP=%" PRIu64
           " EPOCH_SECONDS=%.6f LOADER_ENCODER_SECONDS=%.6f"
           " IIC_SECONDS=%.6f\n", epoch, metrics.batches, metrics.samples,
           ensemble->optimizers[0].step, seconds_between(&start, &finish),
           metrics.loader_encoder_seconds, metrics.iic_seconds);
    return true;
}

static bool official_make_directory(const char *path)
{
#ifdef _WIN32
    return CreateDirectoryA(path, NULL) != 0;
#else
    return mkdir(path, 0700) == 0;
#endif
}

typedef enum {
    OFFICIAL_REJECT = 0,
    OFFICIAL_FRESH,
    OFFICIAL_RESTART_ZERO,
    OFFICIAL_CONTINUE,
    OFFICIAL_FINALIZE
} OfficialAction;

static OfficialAction official_action(bool resume, bool directory_present,
                                      bool latest, bool final,
                                      bool config_matches,
                                      uint64_t completed_epoch)
{
    if (final) return OFFICIAL_REJECT;
    if (!resume)
        return directory_present ? OFFICIAL_REJECT : OFFICIAL_FRESH;
    if (!directory_present || !config_matches) return OFFICIAL_REJECT;
    if (!latest) return OFFICIAL_RESTART_ZERO;
    if (completed_epoch >= MAX_EPOCHS) return OFFICIAL_REJECT;
    return completed_epoch + 1u == MAX_EPOCHS ?
        OFFICIAL_FINALIZE : OFFICIAL_CONTINUE;
}

static bool official_run_controller(RunInput *run, const Options *options)
{
    OfficialPaths paths;
    VAECIICCheckpointMetadata metadata;
    VAECIICEnsemble ensemble;
    VAECIICCheckpointStatus status;
    uint64_t epoch, start = 0u;
    OfficialAction action;
    bool latest = false, final = false, ok = false;
    vae_c_iic_ensemble_init(&ensemble);
    vae_c_iic_checkpoint_metadata_init(&metadata);
    if (!official_paths(run, &paths)) goto done;
    if (options->official_resume) {
        if (!official_inventory(run, &latest, &final) || final ||
            !debug_file_equals(paths.config, run->run_config)) goto done;
    } else {
        if (path_exists(run->official_output_dir) ||
            !official_make_directory(run->official_output_dir) ||
            !official_write_atomic(paths.config, run->run_config) ||
            !official_write_atomic(paths.metrics, OFFICIAL_METRICS_HEADER))
            goto done;
    }
    if (vae_c_iic_ensemble_allocate_v1(
            &ensemble, VAE_C_IIC_MINI224_LATENT_DIM,
            LOGICAL_BATCH) != VAE_C_OP_OK) goto done;
    if (latest) {
        status = vae_c_iic_checkpoint_load(
            paths.latest, &run->checkpoint_identity, &ensemble, &metadata);
        if (status != VAE_C_IIC_CHECKPOINT_OK ||
            metadata.kind != VAE_C_IIC_CHECKPOINT_LATEST ||
            metadata.last_completed_epoch >= MAX_EPOCHS) goto done;
        start = metadata.last_completed_epoch + 1u;
    } else if (options->official_resume && !path_exists(paths.metrics) &&
               !official_write_atomic(paths.metrics,
                                      OFFICIAL_METRICS_HEADER)) {
        goto done;
    }
    action = official_action(options->official_resume,
                             options->official_resume, latest, final,
                             true, metadata.last_completed_epoch);
    if (action == OFFICIAL_REJECT) goto done;
    if (!official_reconcile_metrics(paths.metrics, latest,
                                    metadata.last_completed_epoch)) goto done;
    if (action == OFFICIAL_FINALIZE) {
        metadata.kind = VAE_C_IIC_CHECKPOINT_FINAL;
        status = vae_c_iic_checkpoint_save_atomic(
            paths.final, &run->checkpoint_identity, &metadata, &ensemble);
        if (status != VAE_C_IIC_CHECKPOINT_OK) goto done;
        puts("FINALIZED_FROM_TERMINAL_LATEST=true");
        ok = true;
        goto done;
    }
    printf("START_EPOCH=%" PRIu64 "\n", start);
    for (epoch = start; epoch < MAX_EPOCHS; ++epoch) {
        if (!official_epoch(run, &ensemble, epoch, paths.metrics)) goto done;
        metadata.kind = VAE_C_IIC_CHECKPOINT_LATEST;
        metadata.last_completed_epoch = epoch;
        status = vae_c_iic_checkpoint_save_atomic(
            paths.latest, &run->checkpoint_identity, &metadata, &ensemble);
        if (status != VAE_C_IIC_CHECKPOINT_OK) goto done;
    }
    metadata.kind = VAE_C_IIC_CHECKPOINT_FINAL;
    status = vae_c_iic_checkpoint_save_atomic(
        paths.final, &run->checkpoint_identity, &metadata, &ensemble);
    if (status != VAE_C_IIC_CHECKPOINT_OK) goto done;
    printf("RUN_COMPLETED=true\nFINAL_EPOCH=%" PRIu64
           "\nFINAL_OPTIMIZER_STEP=%" PRIu64 "\n",
           metadata.last_completed_epoch, ensemble.optimizers[0].step);
    ok = true;
done:
    vae_c_iic_ensemble_free(&ensemble);
    return ok;
}

int main(int argc, char **argv)
{
    Options options;
    RunInput run;
    size_t batches[BATCHES_PER_EPOCH], chunks[7];
    uint64_t max_steps;
    uint64_t disk_free_bytes = 0u;
    char official_parent[PATH_CAP];
    bool ok = false;
    if (!parse_options(argc, argv, &options)) {
        fputs("Invalid mini224 validate-only arguments\n", stderr);
        return 2;
    }
    if (options.output_dir != NULL && path_exists(options.output_dir)) {
        puts("OUTPUT_COLLISION=true");
        return 2;
    }
    if (options.smoke_real_two_batch &&
        !temporary_dir_safe(options.smoke_output_dir)) {
        fputs("Unsafe smoke output directory\n", stderr);
        return 2;
    }
    if (options.debug_epoch_gate &&
        !temporary_dir_safe(options.debug_output_dir)) {
        fputs("Unsafe debug output directory\n", stderr);
        return 2;
    }
    if (!verify_git_head(options.repository_root, options.code_commit)) {
        fputs("Branch/HEAD mismatch\n", stderr);
        return 2;
    }
    memset(&run, 0, sizeof(run));
    vae_c_iic_mini224_perturbed_init(&run.input);
    vae_c_iic_checkpoint_identity_init(&run.checkpoint_identity);
    if (prepare(&run, &options, argv[0]) &&
        schedule(batches, chunks, &max_steps) &&
        (!(options.official_run || options.official_preflight) ||
         (join(official_parent, options.repository_root,
               "sakai_multires_c/results/vae_c/iic") &&
          official_disk_free(official_parent, &disk_free_bytes)))) {
        printf("RUN_ID=%s\n", run.run_id);
        printf("RUN_CONFIG_SHA256=%s\n", run.run_config_sha256);
        printf("SOURCE_IDENTITY_SHA256=%s\n",
               run.source_identity_sha256);
        printf("PRODUCER_GIT_HEAD=%s\n", options.code_commit);
        printf("PRODUCER_BINARY_SHA256=%s\n",
               run.producer_binary_sha256);
        printf("BEST_SHA256=%s\n", run.input.contract.checkpoint_sha256);
        printf("PNG_CONTENT_IDENTITY_SHA256=%s\n", run.png_sha256);
        printf("PNG_REFERENCE_TAR_SHA256=%s\n", PNG_TAR_SHA256);
        printf("OUTPUT_DIR=%s\n", run.official_output_dir);
        puts("POPULATION_TRAIN=800\nPOPULATION_VALIDATION=200");
        puts("POPULATION_TOTAL=1000\nPOPULATION_ORDER=train800_then_validation200_native_split");
        puts("LOGICAL_BATCH_SIZE=128\nENCODER_MICROBATCH=16");
        printf("LOGICAL_BATCHES_PER_EPOCH=%u\n", BATCHES_PER_EPOCH);
        printf("LOGICAL_BATCH_SIZES=%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu\n",
               batches[0], batches[1], batches[2], batches[3],
               batches[4], batches[5], batches[6], batches[7]);
        printf("TAIL_ENCODER_CHUNKS=%zu,%zu,%zu,%zu,%zu,%zu,%zu\n",
               chunks[0], chunks[1], chunks[2], chunks[3], chunks[4],
               chunks[5], chunks[6]);
        printf("MAX_EPOCHS=%" PRIu64 "\nMAX_OPTIMIZER_STEPS_PER_PAIR=%" PRIu64
               "\n", MAX_EPOCHS, max_steps);
        puts("IIC_CHECKPOINT_FORMAT=IIC_V1");
        puts("IIC_CHECKPOINT_POLICY=LATEST_EACH_EPOCH_FINAL_TERMINAL_NO_BEST");
        puts("RESUME_AUTHORITY=LATEST\nRESUME_START=LAST_COMPLETED_EPOCH_PLUS_1");
        if (options.official_run || options.official_preflight) {
            printf("OFFICIAL_DISK_FREE_BYTES=%" PRIu64 "\n",
                   disk_free_bytes);
            puts("OFFICIAL_DISK_MINIMUM_BYTES=268435456");
        }
        if (options.validate_only) {
            puts("VALIDATE_ONLY_PASS=true");
            puts("VAE_ENCODER_FORWARD_CALLS=0\nIIC_FORWARD_CALLS=0");
            puts("IIC_BACKWARD_CALLS=0\nADAM_STEPS=0\nOPTIMIZER_STEPS=0");
            puts("DECODER_CALLS=0\nOUTPUT_CREATED=false");
            ok = true;
        } else if (options.official_preflight) {
            puts("OFFICIAL_PREFLIGHT_PASS=true");
            puts("VAE_ENCODER_FORWARD_CALLS=0\nIIC_FORWARD_CALLS=0");
            puts("IIC_BACKWARD_CALLS=0\nADAM_STEPS=0\nOPTIMIZER_STEPS=0");
            puts("DECODER_CALLS=0\nOUTPUT_CREATED=false");
            ok = true;
        } else if (options.smoke_real_two_batch) {
            ok = run_smoke(&run, &options);
        } else if (options.official_run) {
            ok = official_run_controller(&run, &options);
        } else {
            ok = run_debug_epoch_gate(&run, &options);
        }
    }
    vae_c_iic_mini224_perturbed_free(&run.input);
    if (!ok) fputs("MINI224 IIC validate-only failed\n", stderr);
    return ok ? 0 : 1;
}
