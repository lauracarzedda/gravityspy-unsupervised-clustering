#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif

#include "vae_c_iic_checkpoint.h"
#include "vae_c_iic_mini112_central.h"
#include "vae_c_iic_mini112_perturbed.h"
#include "vae_c_iic_png_identity.h"
#include "vae_c_sha256.h"
#include "vae_c_shuffle.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

/* This tool is the real-input runner. Its controller is separate from the
 * C5B6 synthetic fixture; all neural, RNG and checkpoint math stays in the
 * frozen libraries. The official 100-epoch run is not invoked by tests. */
#define PATH_CAP 2048u
#define CONFIG_CAP 8192u
#define LINE_CAP 4096u
#define ROWS VAE_C_IIC_MINI112_ROWS
#define DIM VAE_C_IIC_MINI112_LATENT_DIM
#define BATCH VAE_C_IIC_MINI112_LOGICAL_BATCH
#define BATCHES 68u
#define DEFAULT_MICROBATCH 64u
/* SHA-256 of the UTF-8 literal MINI112_IIC_REAL_INPUT_CONTRACT_V1.
 * This identifies the runner input contract, not the PNG or mu bytes. */
#define SOURCE_IDENTITY_SHA256 \
    "9fc396e09064e52983384c1502f00136153b1eaecfd192c574ea957b7ee1c212"
/* Derived independently from the verified 2026-08-26 out272_gray transfer TAR. */
#define EXPECTED_PNG_CONTENT_IDENTITY_SHA256 \
    "5e6bf28879886f69223864076d354a5bce511b78164e96e742e6117f79f51a5c"
#define PNG_REFERENCE_TAR_SHA256 \
    "70c766c3652f586c8da4c8cadf134a8ce83cfd07c8cd10545367f380017ddab2"
#define METRICS_HEADER \
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

typedef enum {
    MODE_NONE = 0,
    MODE_VALIDATE,
    MODE_PARTIAL,
    MODE_RUN
} Mode;

typedef struct {
    Mode mode;
    const char *repository_root;
    const char *output_dir;
    const char *code_commit;
    uint64_t max_epochs;
    size_t encoder_microbatch;
    bool resume;
    bool official;
} Options;

typedef struct {
    VAECIICMini112Central central;
    VAECIICMini112Perturbed perturbed;
    VAECIICCheckpointIdentity checkpoint_identity;
    char run_id[VAE_C_IIC_CHECKPOINT_RUN_ID_CAP];
    char binary_sha256[VAE_C_SHA256_HEX_SIZE];
    char png_content_identity_sha256[VAE_C_SHA256_HEX_SIZE];
    char run_config[CONFIG_CAP];
} RunInput;

typedef struct {
    double main_mi[VAE_C_IIC_ENSEMBLE_K];
    double over_mi[VAE_C_IIC_ENSEMBLE_K];
    double total_loss[VAE_C_IIC_ENSEMBLE_K];
    double loader_encoder_seconds;
    double iic_seconds;
    size_t batches;
    size_t samples;
} EpochMetrics;

static bool decimal_u64(const char *text, uint64_t *value)
{
    char *end;
    unsigned long long parsed;
    if (text == NULL || text[0] == '\0' || text[0] == '-' ||
        text[0] == '+') return false;
    errno = 0;
    parsed = strtoull(text, &end, 10);
    if (errno != 0 || *end != '\0') return false;
    *value = (uint64_t)parsed;
    return (unsigned long long)*value == parsed;
}

static bool lowercase_hex(const char *text, size_t length)
{
    size_t i;
    if (text == NULL || strlen(text) != length) return false;
    for (i = 0u; i < length; i++) {
        if (!((text[i] >= '0' && text[i] <= '9') ||
              (text[i] >= 'a' && text[i] <= 'f'))) return false;
    }
    return true;
}

static bool parse_options(int argc, char **argv, Options *options)
{
    int index;
    bool seen_microbatch = false, seen_epochs = false;
    memset(options, 0, sizeof(*options));
    options->max_epochs = 100u;
    options->encoder_microbatch = DEFAULT_MICROBATCH;
    for (index = 1; index < argc; index++) {
        const char *argument = argv[index];
        if (strcmp(argument, "--validate-only") == 0 &&
            options->mode == MODE_NONE) {
            options->mode = MODE_VALIDATE;
        } else if (strcmp(argument, "--smoke-partial") == 0 &&
                   options->mode == MODE_NONE) {
            options->mode = MODE_PARTIAL;
        } else if (strcmp(argument, "--run") == 0 &&
                   options->mode == MODE_NONE) {
            options->mode = MODE_RUN;
        } else if (strcmp(argument, "--repository-root") == 0 &&
                   ++index < argc && options->repository_root == NULL) {
            options->repository_root = argv[index];
        } else if (strcmp(argument, "--code-commit") == 0 &&
                   ++index < argc && options->code_commit == NULL) {
            options->code_commit = argv[index];
        } else if (strcmp(argument, "--output-dir") == 0 &&
                   ++index < argc && options->output_dir == NULL) {
            options->output_dir = argv[index];
        } else if (strcmp(argument, "--max-epochs") == 0 &&
                   ++index < argc && !seen_epochs) {
            if (!decimal_u64(argv[index], &options->max_epochs)) return false;
            seen_epochs = true;
        } else if (strcmp(argument, "--encoder-microbatch") == 0 &&
                   ++index < argc && !seen_microbatch) {
            uint64_t value;
            if (!decimal_u64(argv[index], &value) ||
                value > SIZE_MAX) return false;
            options->encoder_microbatch = (size_t)value;
            seen_microbatch = true;
        } else if (strcmp(argument, "--resume") == 0 &&
                   !options->resume) {
            options->resume = true;
        } else if (strcmp(argument, "--official") == 0 &&
                   !options->official) {
            options->official = true;
        } else {
            return false;
        }
    }
    if (options->mode == MODE_NONE || options->repository_root == NULL ||
        options->code_commit == NULL ||
        !lowercase_hex(options->code_commit, 40u) ||
        options->max_epochs == 0u || options->max_epochs > 100u ||
        options->encoder_microbatch == 0u ||
        options->encoder_microbatch > 64u) return false;
    if (options->mode == MODE_RUN)
        return options->output_dir != NULL;
    return options->output_dir == NULL && !options->resume &&
        !options->official && !seen_epochs && !seen_microbatch;
}

static bool join(char path[PATH_CAP], const char *directory,
                 const char *name)
{
    int written = snprintf(path, PATH_CAP, "%s/%s", directory, name);
    return written > 0 && (size_t)written < PATH_CAP;
}

static bool read_line(const char *path, char *output, size_t capacity)
{
    FILE *file = fopen(path, "rb");
    size_t length;
    bool good;
    if (file == NULL || fgets(output, (int)capacity, file) == NULL) {
        if (file != NULL) (void)fclose(file);
        return false;
    }
    length = strlen(output);
    good = length > 0u && length < capacity - 1u &&
        (output[length - 1u] == '\n' || feof(file));
    while (length > 0u && (output[length - 1u] == '\n' ||
                           output[length - 1u] == '\r'))
        output[--length] = '\0';
    if (fgetc(file) != EOF || ferror(file)) good = false;
    if (fclose(file) != 0) good = false;
    return good;
}

/* This repository has a normal loose branch ref. A packed/worktree ref is
 * rejected rather than silently treating a caller-supplied SHA as HEAD. */
static bool verify_git_head(const char *root, const char *commit)
{
    char head_path[PATH_CAP], ref_path[PATH_CAP];
    char head_line[256], actual[256];
    const char *prefix = "ref: refs/heads/";
    if (!join(head_path, root, ".git/HEAD") ||
        !read_line(head_path, head_line, sizeof(head_line)) ||
        strncmp(head_line, prefix, strlen(prefix)) != 0 ||
        strcmp(head_line + strlen(prefix), "vae-c-sakai-like") != 0 ||
        !join(ref_path, root, ".git/refs/heads/vae-c-sakai-like") ||
        !read_line(ref_path, actual, sizeof(actual)) ||
        !lowercase_hex(actual, 40u)) return false;
    return strcmp(actual, commit) == 0;
}

static bool exists(const char *path)
{
#ifdef _WIN32
    return GetFileAttributesA(path) != INVALID_FILE_ATTRIBUTES;
#else
    struct stat entry;
    return stat(path, &entry) == 0;
#endif
}

static bool make_directory(const char *path)
{
#ifdef _WIN32
    return CreateDirectoryA(path, NULL) != 0;
#else
    return mkdir(path, 0700) == 0;
#endif
}

static bool replace_file(const char *source, const char *destination)
{
#ifdef _WIN32
    return MoveFileExA(source, destination,
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)
        != 0;
#else
    return rename(source, destination) == 0;
#endif
}

static bool absolute_path(const char *input, char output[PATH_CAP])
{
#ifdef _WIN32
    DWORD size = GetFullPathNameA(input, PATH_CAP, output, NULL);
    return size > 0u && size < PATH_CAP;
#else
    size_t length = strlen(input);
    if (length == 0u || length >= PATH_CAP || input[0] != '/' ||
        strstr(input, "..") != NULL) return false;
    memcpy(output, input, length + 1u);
    return true;
#endif
}

static bool path_under(const char *child, const char *parent)
{
    size_t length = strlen(parent);
#ifdef _WIN32
    return _strnicmp(child, parent, length) == 0 &&
#else
    return strncmp(child, parent, length) == 0 &&
#endif
        (child[length] == '\0' || child[length] == '/' ||
        child[length] == '\\');
}

#ifdef _WIN32
static bool no_reparse_ancestors(char absolute[PATH_CAP])
{
    size_t index, end = strlen(absolute);
    for (index = 3u; index <= end; index++) {
        if (index == end || absolute[index] == '\\' ||
            absolute[index] == '/') {
            char saved = absolute[index];
            DWORD attributes;
            absolute[index] = '\0';
            attributes = GetFileAttributesA(absolute);
            absolute[index] = saved;
            if (attributes != INVALID_FILE_ATTRIBUTES &&
                (attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0)
                return false;
            if (index < end &&
                (attributes == INVALID_FILE_ATTRIBUTES ||
                 (attributes & FILE_ATTRIBUTE_DIRECTORY) == 0))
                return false;
        }
    }
    return true;
}
#endif

static bool output_dir_policy(const Options *options, bool *inside_repo)
{
    char root[PATH_CAP], output[PATH_CAP];
    if (!absolute_path(options->repository_root, root) ||
        !absolute_path(options->output_dir, output)) return false;
    *inside_repo = path_under(output, root);
    if (!options->official && *inside_repo) return false;
    if (options->official) return *inside_repo &&
        options->max_epochs == 100u &&
        options->encoder_microbatch == DEFAULT_MICROBATCH
#ifdef _WIN32
        && no_reparse_ancestors(output)
#endif
        ;
#ifdef _WIN32
    {
        char temp[PATH_CAP];
        DWORD length = GetTempPathA(PATH_CAP, temp);
        if (length == 0u || length >= PATH_CAP) return false;
        while (length > 0u && (temp[length - 1u] == '\\' ||
                               temp[length - 1u] == '/'))
            temp[--length] = '\0';
        if (!path_under(output, temp) ||
            strlen(output) <= length + 1u) return false;
        return no_reparse_ancestors(output);
    }
#else
    {
        const char *temp = getenv("TMPDIR");
        if (temp == NULL || temp[0] == '\0') temp = "/tmp";
        return path_under(output, temp) && strlen(output) > strlen(temp) + 1u;
    }
#endif
}

static void run_input_init(RunInput *input)
{
    memset(input, 0, sizeof(*input));
    vae_c_iic_mini112_central_init(&input->central);
    vae_c_iic_mini112_perturbed_init(&input->perturbed);
    vae_c_iic_checkpoint_identity_init(&input->checkpoint_identity);
}

static void run_input_free(RunInput *input)
{
    vae_c_iic_mini112_perturbed_free(&input->perturbed);
    vae_c_iic_mini112_central_free(&input->central);
}

static bool source_valid(const RunInput *input)
{
    size_t i;
    if (input->central.row_count != ROWS ||
        input->central.latent_dim != DIM ||
        strcmp(input->central.checkpoint_sha256,
               input->perturbed.contract.checkpoint_sha256) != 0 ||
        strcmp(input->central.checkpoint_sha256,
               "8857090fb1477fafed6b841bb880a06d61acf38899c7ad2e5a9ef60672989a8b") != 0 ||
        input->perturbed.contract.expected_epoch != 97u ||
        input->perturbed.contract.expected_optimizer_step != 10584u ||
        strcmp(input->perturbed.contract.profile.profile_id,
               "mini112_primary_v1") != 0) return false;
    for (i = 0u; i < ROWS; i++) {
        const VAECIICMini112CentralRow *row = &input->central.rows[i];
        const GSEventRecord *event = gs_event_manifest_get(
            &input->perturbed.events, i);
        if (event == NULL || row->manifest_index != i ||
            row->event_index != (uint32_t)event->event_index ||
            strcmp(row->event_id, event->event_id) != 0 ||
            vae_c_iic_mini112_central_row_mu(&input->central, i) == NULL)
            return false;
    }
    return true;
}

static bool build_identity(RunInput *input, const Options *options,
                           const char *executable)
{
    char preimage[CONFIG_CAP];
    char digest[VAE_C_SHA256_HEX_SIZE];
    const VAECIICMini112Central *central = &input->central;
    const VAECMuCanonicalContract *contract = &input->perturbed.contract;
    VAECIICCheckpointIdentity *identity = &input->checkpoint_identity;
    int written;
    size_t pair;
    if (vae_c_mu_file_sha256(executable, input->binary_sha256) !=
        VAE_C_MU_OK) return false;
    written = snprintf(
        preimage, sizeof(preimage),
        "format=MINI112_IIC_REAL_RUN_V1\n"
        "source_identity_sha256=%s\n"
        "source_identity_role=contract_identifier\n"
        "producer_git_head=%s\n"
        "producer_binary_sha256=%s\n"
        "profile_id=mini112_primary_v1\n"
        "profile_config_sha256=%s\n"
        "catalog_sha256=%s\n"
        "vae_run_config_sha256=%s\n"
        "vae_best_sha256=%s\n"
        "event_manifest_sha256=%s\n"
        "tensor_manifest_sha256=%s\n"
        "train_split_sha256=%s\n"
        "validation_split_sha256=%s\n"
        "train_mu_sha256=%s\n"
        "validation_mu_sha256=%s\n"
        "train_rows_sha256=%s\n"
        "validation_rows_sha256=%s\n"
        "png_content_identity_format=MINI75_IIC_PNG_CONTENT_IDENTITY_V1\n"
        "augmentation=symmetric_figure:789\n"
        "shuffle_seed=101112\n"
        "logical_batch=128\n"
        "encoder_microbatch=%zu\n"
        "C=36\nW=250\nK=5\n"
        "pair_seeds=123,124,125,126,127\n"
        "adam=0.0005,0.9,0.999,1e-8,weight_decay0,amsgrad_false\n"
        "init_policy=splitmix64_raw_seed_main_then_over_v1\n"
        "max_epochs=%" PRIu64 "\n"
        "checkpoint=IIC_V1_LATEST_FINAL_NO_BEST\n"
        "labels_used=false\n",
        SOURCE_IDENTITY_SHA256, options->code_commit,
        input->binary_sha256,
        contract->artifact_v1.profile_config_sha256,
        contract->artifact_v1.catalog_sha256,
        contract->run_config_sha256,
        central->checkpoint_sha256,
        contract->expected_identity.event_manifest_sha256,
        contract->expected_identity.tensor_manifest_sha256,
        contract->expected_identity.train_split_sha256,
        contract->expected_identity.validation_split_sha256,
        central->train_matrix_sha256, central->validation_matrix_sha256,
        central->train_rows_sha256, central->validation_rows_sha256,
        options->encoder_microbatch, options->max_epochs);
    if (written < 0 || (size_t)written >= sizeof(preimage) ||
        !vae_c_iic_png_bind_run_preimage_v1(
            preimage, (size_t)written,
            input->png_content_identity_sha256, digest) ||
        snprintf(input->run_id, sizeof(input->run_id),
                 "mini112_iic_v1_%s", digest) != 79) return false;
    written = snprintf(input->run_config, sizeof(input->run_config),
        "{\n"
        "  \"format\": \"MINI112_IIC_REAL_RUN_V1\",\n"
        "  \"run_id\": \"%s\",\n"
        "  \"source_identity_sha256\": \"%s\",\n"
        "  \"source_identity_role\": \"contract_identifier\",\n"
        "  \"producer_git_head\": \"%s\",\n"
        "  \"producer_state\": \"BINARY_SHA256_AUTHORITY\",\n"
        "  \"producer_binary_sha256\": \"%s\",\n"
        "  \"profile_id\": \"mini112_primary_v1\",\n"
        "  \"profile_config_sha256\": \"%s\",\n"
        "  \"catalog_sha256\": \"%s\",\n"
        "  \"vae_run_config_sha256\": \"%s\",\n"
        "  \"vae_checkpoint_kind\": \"BEST\",\n"
        "  \"vae_checkpoint_epoch\": 97,\n"
        "  \"vae_checkpoint_step\": 10584,\n"
        "  \"vae_checkpoint_sha256\": \"%s\",\n"
        "  \"event_manifest_sha256\": \"%s\",\n"
        "  \"tensor_manifest_sha256\": \"%s\",\n"
        "  \"train_split_sha256\": \"%s\",\n"
        "  \"validation_split_sha256\": \"%s\",\n"
        "  \"train_mu_sha256\": \"%s\",\n"
        "  \"validation_mu_sha256\": \"%s\",\n"
        "  \"train_rows_sha256\": \"%s\",\n"
        "  \"validation_rows_sha256\": \"%s\",\n"
        "  \"png_content_identity_format\": "
            "\"MINI75_IIC_PNG_CONTENT_IDENTITY_V1\",\n"
        "  \"png_content_identity_sha256\": \"%s\",\n"
        "  \"png_reference_tar_sha256\": \"%s\",\n"
        "  \"population_rows\": 8583,\n"
        "  \"population_order\": \"manifest_index_0_8582\",\n"
        "  \"central_view\": \"precomputed_posterior_mu\",\n"
        "  \"perturbed_view\": \"frozen_encoder_eval_posterior_mu\",\n"
        "  \"augmentation_mode\": \"symmetric_figure\",\n"
        "  \"augmentation_seed\": 789,\n"
        "  \"shuffle_seed\": 101112,\n"
        "  \"logical_batch\": 128,\n"
        "  \"encoder_microbatch\": %zu,\n"
        "  \"C\": 36, \"W\": 250, \"K\": 5,\n"
        "  \"pair_seeds\": [123, 124, 125, 126, 127],\n"
        "  \"adam_lr\": 0.0005, \"adam_beta1\": 0.9,\n"
        "  \"adam_beta2\": 0.999, \"adam_epsilon\": 1e-8,\n"
        "  \"adam_weight_decay\": 0, \"adam_amsgrad\": false,\n"
        "  \"init_policy\": \"splitmix64_raw_seed_main_then_over_v1\",\n"
        "  \"max_epochs\": %" PRIu64 ",\n"
        "  \"checkpoint_format\": \"IIC_V1\",\n"
        "  \"checkpoint_policy\": \"LATEST_EACH_EPOCH_FINAL_TERMINAL_NO_BEST\",\n"
        "  \"labels_used\": false\n"
        "}\n",
        input->run_id, SOURCE_IDENTITY_SHA256, options->code_commit,
        input->binary_sha256,
        contract->artifact_v1.profile_config_sha256,
        contract->artifact_v1.catalog_sha256,
        contract->run_config_sha256,
        central->checkpoint_sha256,
        contract->expected_identity.event_manifest_sha256,
        contract->expected_identity.tensor_manifest_sha256,
        contract->expected_identity.train_split_sha256,
        contract->expected_identity.validation_split_sha256,
        central->train_matrix_sha256, central->validation_matrix_sha256,
        central->train_rows_sha256, central->validation_rows_sha256,
        input->png_content_identity_sha256, PNG_REFERENCE_TAR_SHA256,
        options->encoder_microbatch, options->max_epochs);
    if (written < 0 || (size_t)written >= sizeof(input->run_config))
        return false;
    identity->input_dim = DIM;
    identity->capacity = BATCH;
    identity->batch_size = BATCH;
    identity->train_rows = ROWS;
    identity->max_epochs = options->max_epochs;
    identity->shuffle_seed = VAE_C_IIC_ENSEMBLE_SHUFFLE_SEED;
    for (pair = 0u; pair < VAE_C_IIC_ENSEMBLE_K; pair++)
        identity->seeds[pair] =
            VAE_C_IIC_ENSEMBLE_FIRST_SEED + (uint64_t)pair;
    (void)strcpy(identity->run_id, input->run_id);
    (void)strcpy(identity->base_commit, options->code_commit);
    (void)strcpy(identity->producer_state, "BINARY_SHA256_AUTHORITY");
    (void)strcpy(identity->producer_binary_sha256,
                 input->binary_sha256);
    return true;
}

static bool prepare_input(RunInput *input, const Options *options,
                          const char *executable)
{
    char error[512];
    size_t png_count = 0u;
    error[0] = '\0';
    if (!vae_c_iic_mini112_central_load(
            options->repository_root, NULL, &input->central,
            error, sizeof(error))) {
        fprintf(stderr, "Central mu: %s\n", error);
        return false;
    }
    if (!vae_c_iic_mini112_perturbed_open(
            &input->perturbed, options->repository_root) ||
        !source_valid(input)) {
        fputs("Canonical input/provenance mismatch\n", stderr);
        return false;
    }
    if (!vae_c_iic_png_content_identity_v1(
            options->repository_root, &input->perturbed.events,
            input->png_content_identity_sha256, &png_count) ||
        png_count != ROWS * GS_EVENT_VIEW_COUNT ||
        strcmp(input->png_content_identity_sha256,
               EXPECTED_PNG_CONTENT_IDENTITY_SHA256) != 0) {
        fputs("PNG content identity mismatch\n", stderr);
        return false;
    }
    if (!build_identity(input, options, executable)) return false;
    return true;
}

static double seconds_between(const struct timespec *before,
                              const struct timespec *after)
{
    return (double)(after->tv_sec - before->tv_sec) +
        (double)(after->tv_nsec - before->tv_nsec) / 1.0e9;
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

static void process_memory_report(void)
{
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX counters;
    memset(&counters, 0, sizeof(counters));
    counters.cb = sizeof(counters);
    if (GetProcessMemoryInfo(GetCurrentProcess(),
            (PROCESS_MEMORY_COUNTERS *)&counters,
            sizeof(counters)) != 0) {
        printf("PEAK_WORKING_SET_BYTES=%zu\n",
               (size_t)counters.PeakWorkingSetSize);
        printf("PRIVATE_BYTES_AT_END=%zu\n",
               (size_t)counters.PrivateUsage);
    }
#endif
}

static bool epoch_permutation(uint64_t epoch, size_t *permutation)
{
    return vae_c_shuffle_make_permutation(
        ROWS, VAE_C_IIC_ENSEMBLE_SHUFFLE_SEED, epoch,
        permutation) == VAE_C_RNG_OK;
}

static bool run_batch(RunInput *input, VAECIICEnsemble *ensemble,
                      const size_t *indices, size_t count, uint64_t epoch,
                      size_t microbatch, EpochMetrics *metrics)
{
    const size_t dims[2] = {count, DIM};
    VAECShape shape;
    VAECTensor a, b;
    VAECIICMini112PerturbedReport report;
    struct timespec t0, t1, t2;
    float snapshot_b[BATCH * DIM];
    uint64_t old_step;
    size_t row, pair;
    bool ok = false;
    vae_c_tensor_init(&a);
    vae_c_tensor_init(&b);
    if (count < 2u || count > BATCH ||
        vae_c_shape_make(2u, dims, &shape) != VAE_C_STORAGE_OK ||
        vae_c_tensor_allocate(&a, &shape, true) != VAE_C_STORAGE_OK ||
        vae_c_tensor_allocate(&b, &shape, true) != VAE_C_STORAGE_OK ||
        !monotonic_now(&t0)) goto done;
    for (row = 0u; row < count; row++) {
        const float *source = vae_c_iic_mini112_central_row_mu(
            &input->central, indices[row]);
        if (source == NULL) goto done;
        memcpy(&a.data[row * DIM], source, DIM * sizeof(float));
    }
    memset(&report, 0, sizeof(report));
    if (!vae_c_iic_mini112_perturbed_compute(
            &input->perturbed, indices, count, epoch, microbatch,
            b.data, NULL, &report) ||
        report.samples != count ||
        report.encoder_forward_calls !=
            (uint64_t)((count + microbatch - 1u) / microbatch) ||
        !report.inputs_bitwise_unchanged ||
        !report.model_parameters_bn_bitwise_unchanged ||
        !report.decoder_workspace_untouched ||
        !report.optimizer_unchanged || !monotonic_now(&t1)) goto done;
    memcpy(snapshot_b, b.data, count * DIM * sizeof(float));
    old_step = ensemble->optimizers[0].step;
    if (vae_c_iic_ensemble_train_step(ensemble, &a, &b) != VAE_C_OP_OK ||
        !monotonic_now(&t2)) goto done;
    for (row = 0u; row < count; row++) {
        const float *source = vae_c_iic_mini112_central_row_mu(
            &input->central, indices[row]);
        if (source == NULL ||
            memcmp(&a.data[row * DIM], source, DIM * sizeof(float)) != 0)
            goto done;
    }
    if (memcmp(snapshot_b, b.data, count * DIM * sizeof(float)) != 0)
        goto done;
    for (pair = 0u; pair < VAE_C_IIC_ENSEMBLE_K; pair++) {
        const VAECIICPairWorkspace *workspace = &ensemble->workspaces[pair];
        const VAECIICPair *model = &ensemble->pairs[pair];
        if (!isfinite(workspace->main_result.mutual_information) ||
            !isfinite(workspace->over_result.mutual_information) ||
            !isfinite(workspace->total_loss) ||
            ensemble->optimizers[pair].step != old_step + 1u ||
            model->main_head.num_batches_tracked !=
                2u * (old_step + 1u) ||
            model->overcluster_head.num_batches_tracked !=
                2u * (old_step + 1u)) goto done;
        metrics->main_mi[pair] +=
            (double)workspace->main_result.mutual_information;
        metrics->over_mi[pair] +=
            (double)workspace->over_result.mutual_information;
        metrics->total_loss[pair] += (double)workspace->total_loss;
    }
    metrics->batches++;
    metrics->samples += count;
    metrics->loader_encoder_seconds += seconds_between(&t0, &t1);
    metrics->iic_seconds += seconds_between(&t1, &t2);
    ok = true;
done:
    vae_c_tensor_free(&b);
    vae_c_tensor_free(&a);
    return ok;
}

static bool run_partial(RunInput *input, const Options *options)
{
    size_t *permutation = NULL;
    VAECIICEnsemble ensemble;
    EpochMetrics metrics;
    const size_t offsets[3] = {0u, BATCH, ROWS - 7u};
    const size_t counts[3] = {BATCH, BATCH, 7u};
    size_t index, pair;
    bool ok = false;
    vae_c_iic_ensemble_init(&ensemble);
    memset(&metrics, 0, sizeof(metrics));
    permutation = (size_t *)malloc(ROWS * sizeof(*permutation));
    if (permutation == NULL || !epoch_permutation(0u, permutation) ||
        vae_c_iic_ensemble_allocate_v1(&ensemble, DIM, BATCH) !=
            VAE_C_OP_OK) goto done;
    for (index = 0u; index < 3u; index++) {
        if (!run_batch(input, &ensemble, &permutation[offsets[index]],
                       counts[index], 0u, options->encoder_microbatch,
                       &metrics)) goto done;
    }
    if (metrics.batches != 3u || metrics.samples != 263u) goto done;
    for (pair = 0u; pair < VAE_C_IIC_ENSEMBLE_K; pair++) {
        if (ensemble.optimizers[pair].step != 3u ||
            ensemble.pairs[pair].main_head.num_batches_tracked != 6u ||
            ensemble.pairs[pair].overcluster_head.num_batches_tracked != 6u)
            goto done;
    }
    ok = true;
    puts("SMOKE_PARTIAL_PASS=true");
    puts("IIC_OPTIMIZER_STEPS_PER_PAIR=3");
    puts("LATEST_CREATED=false");
    puts("FINAL_CREATED=false");
    puts("OUTPUT_CREATED=false");
    printf("PARTIAL_LOADER_ENCODER_SECONDS=%.6f\n",
           metrics.loader_encoder_seconds);
    printf("PARTIAL_IIC_SECONDS=%.6f\n", metrics.iic_seconds);
    process_memory_report();
done:
    vae_c_iic_ensemble_free(&ensemble);
    free(permutation);
    return ok;
}

static bool write_text_atomic(const char *path, const char *content)
{
    char partial[PATH_CAP];
    FILE *file;
    size_t length = strlen(content);
    if (snprintf(partial, sizeof(partial), "%s.partial", path) < 0 ||
        strlen(path) + strlen(".partial") >= sizeof(partial) ||
        exists(partial)) return false;
    file = fopen(partial, "wb");
    if (file == NULL) return false;
    if (fwrite(content, 1u, length, file) != length ||
        fflush(file) != 0) {
        (void)fclose(file);
        return false;
    }
    if (fclose(file) != 0) return false;
    return replace_file(partial, path);
}

static bool file_equals_text(const char *path, const char *content)
{
    FILE *file = fopen(path, "rb");
    size_t length = strlen(content);
    char *readback;
    bool equal;
    if (file == NULL) return false;
    readback = (char *)malloc(length + 2u);
    if (readback == NULL) { (void)fclose(file); return false; }
    equal = fread(readback, 1u, length + 1u, file) == length &&
        memcmp(readback, content, length) == 0 && ferror(file) == 0;
    if (fclose(file) != 0) equal = false;
    free(readback);
    return equal;
}

static bool valid_metric_row(const char *line, uint64_t expected_epoch)
{
    const char *cursor = line;
    size_t column;
    double value[24];
    for (column = 0u; column < 24u; column++) {
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
    if (value[0] != (double)expected_epoch || value[1] < 0.0 ||
        value[2] != (double)BATCHES || value[3] != (double)ROWS)
        return false;
    for (column = 4u; column < 9u; column++) {
        if (value[column] != (double)((expected_epoch + 1u) * BATCHES))
            return false;
    }
    return true;
}

static bool metric_epoch_prefix(const char *line, uint64_t *epoch)
{
    char *end;
    unsigned long long parsed;
    if (line[0] == '-' || line[0] == '+') return false;
    errno = 0;
    parsed = strtoull(line, &end, 10);
    if (errno != 0 || end == line || *end != ',') return false;
    *epoch = (uint64_t)parsed;
    return (unsigned long long)*epoch == parsed;
}

static bool future_epoch_prefix(const char *line, uint64_t committed_epoch)
{
    char *end;
    unsigned long long parsed;
    if (line[0] == '-' || line[0] == '+') return false;
    errno = 0;
    parsed = strtoull(line, &end, 10);
    return errno == 0 && end != line &&
        (*end == ',' || *end == '\n' || *end == '\0') &&
        (uint64_t)parsed > committed_epoch;
}

static bool reconcile_metrics(const char *path, bool has_latest,
                              uint64_t committed_epoch)
{
    char partial[PATH_CAP], line[LINE_CAP];
    FILE *in = NULL, *out = NULL;
    uint64_t expected = 0u;
    bool future = false, ok = false;
    if (snprintf(partial, sizeof(partial), "%s.partial", path) < 0 ||
        strlen(path) + strlen(".partial") >= sizeof(partial) ||
        exists(partial)) return false;
    in = fopen(path, "rb");
    if (has_latest && in == NULL) return false;
    out = fopen(partial, "wb");
    if (out == NULL || fputs(METRICS_HEADER, out) == EOF) goto done;
    if (in != NULL) {
        if (fgets(line, sizeof(line), in) == NULL ||
            strcmp(line, METRICS_HEADER) != 0) goto done;
        while (fgets(line, sizeof(line), in) != NULL) {
            uint64_t epoch;
            if (!has_latest) {
                future = true;
                continue;
            }
            if (expected > committed_epoch) {
                if (!future_epoch_prefix(line, committed_epoch))
                    goto done;
                future = true;
                continue;
            }
            if (!metric_epoch_prefix(line, &epoch)) goto done;
            if (epoch > committed_epoch) {
                /* Missing committed rows are not a recoverable future tail. */
                if (expected != committed_epoch + 1u) goto done;
                future = true;
                continue;
            }
            if (future || !valid_metric_row(line, expected) ||
                fputs(line, out) == EOF) goto done;
            expected++;
        }
        if (ferror(in)) goto done;
    }
    if (has_latest && expected != committed_epoch + 1u) goto done;
    if (fflush(out) != 0 || fclose(out) != 0) {
        out = NULL;
        goto done;
    }
    out = NULL;
    if (in != NULL && fclose(in) != 0) { in = NULL; goto done; }
    in = NULL;
    if (has_latest && !future) {
        if (remove(partial) != 0) goto done;
    } else if (!replace_file(partial, path)) goto done;
    ok = true;
done:
    if (in != NULL) (void)fclose(in);
    if (out != NULL) (void)fclose(out);
    return ok;
}

static bool append_metrics(const char *path, uint64_t epoch,
                           double elapsed, const EpochMetrics *metrics,
                           const VAECIICEnsemble *ensemble)
{
    FILE *out = fopen(path, "ab");
    size_t pair;
    bool ok = true;
    if (out == NULL || metrics->batches != BATCHES ||
        metrics->samples != ROWS || !isfinite(elapsed)) {
        if (out != NULL) (void)fclose(out);
        return false;
    }
    if (fprintf(out, "%" PRIu64 ",%.9f,%zu,%zu", epoch, elapsed,
                metrics->batches, metrics->samples) < 0) ok = false;
    for (pair = 0u; ok && pair < VAE_C_IIC_ENSEMBLE_K; pair++)
        if (fprintf(out, ",%" PRIu64,
                    ensemble->optimizers[pair].step) < 0) ok = false;
    for (pair = 0u; ok && pair < VAE_C_IIC_ENSEMBLE_K; pair++)
        if (fprintf(out, ",%.17g", metrics->main_mi[pair] / BATCHES)
            < 0) ok = false;
    for (pair = 0u; ok && pair < VAE_C_IIC_ENSEMBLE_K; pair++)
        if (fprintf(out, ",%.17g", metrics->over_mi[pair] / BATCHES)
            < 0) ok = false;
    for (pair = 0u; ok && pair < VAE_C_IIC_ENSEMBLE_K; pair++)
        if (fprintf(out, ",%.17g", metrics->total_loss[pair] / BATCHES)
            < 0) ok = false;
    if (ok && fputc('\n', out) == EOF) ok = false;
    if (ok && fflush(out) != 0) ok = false;
    if (fclose(out) != 0) ok = false;
    return ok;
}

static bool run_epoch(RunInput *input, VAECIICEnsemble *ensemble,
                      uint64_t epoch, const Options *options,
                      const char *metrics_path)
{
    size_t *permutation = NULL, offset;
    EpochMetrics metrics;
    struct timespec start, finish;
    bool ok = false;
    memset(&metrics, 0, sizeof(metrics));
    permutation = (size_t *)malloc(ROWS * sizeof(*permutation));
    if (permutation == NULL || !epoch_permutation(epoch, permutation) ||
        !monotonic_now(&start)) goto done;
    for (offset = 0u; offset < ROWS; offset += BATCH) {
        size_t count = ROWS - offset;
        if (count > BATCH) count = BATCH;
        if (!run_batch(input, ensemble, &permutation[offset], count,
                       epoch, options->encoder_microbatch, &metrics))
            goto done;
    }
    if (!monotonic_now(&finish) ||
        !append_metrics(metrics_path, epoch,
                        seconds_between(&start, &finish), &metrics,
                        ensemble)) goto done;
    printf("EPOCH=%" PRIu64 " BATCHES=%zu SAMPLES=%zu STEP=%" PRIu64
           " EPOCH_SECONDS=%.6f LOADER_ENCODER_SECONDS=%.6f"
           " IIC_SECONDS=%.6f\n", epoch, metrics.batches,
           metrics.samples, ensemble->optimizers[0].step,
           seconds_between(&start, &finish),
           metrics.loader_encoder_seconds, metrics.iic_seconds);
    process_memory_report();
    ok = true;
done:
    free(permutation);
    return ok;
}

static bool official_output_matches(const Options *options,
                                    const RunInput *input)
{
    char expected[PATH_CAP], normalized[PATH_CAP], output[PATH_CAP];
    int written;
    if (!absolute_path(options->output_dir, output)) return false;
    written = snprintf(expected, sizeof(expected),
        "%s/sakai_multires_c/results/vae_c/iic/%s",
        options->repository_root, input->run_id);
    if (written < 0 || (size_t)written >= sizeof(expected) ||
        !absolute_path(expected, normalized)) return false;
#ifdef _WIN32
    return _stricmp(output, normalized) == 0;
#else
    return strcmp(output, normalized) == 0;
#endif
}

static bool run_controller(RunInput *input, const Options *options)
{
    char config_path[PATH_CAP], metrics_path[PATH_CAP];
    char latest_path[PATH_CAP], final_path[PATH_CAP];
    VAECIICCheckpointMetadata metadata;
    VAECIICEnsemble ensemble;
    VAECIICCheckpointStatus status;
    uint64_t epoch, start_epoch = 0u;
    bool latest, ok = false;
    vae_c_iic_ensemble_init(&ensemble);
    vae_c_iic_checkpoint_metadata_init(&metadata);
    if (!join(config_path, options->output_dir, "run_config.json") ||
        !join(metrics_path, options->output_dir, "metrics.csv") ||
        !join(latest_path, options->output_dir, "latest.ckpt") ||
        !join(final_path, options->output_dir, "final.ckpt") ||
        (options->official && !official_output_matches(options, input)))
        goto done;
    if (exists(final_path)) {
        fputs("FINAL exists; refusing execution\n", stderr);
        goto done;
    }
    if (options->resume) {
        if (!exists(options->output_dir) ||
            !file_equals_text(config_path, input->run_config))
            goto done;
    } else {
        if (!make_directory(options->output_dir) ||
            !write_text_atomic(config_path, input->run_config) ||
            !write_text_atomic(metrics_path, METRICS_HEADER))
            goto done;
    }
    latest = exists(latest_path);
    if (latest && !options->resume) goto done;
    if (vae_c_iic_ensemble_allocate_v1(
            &ensemble, DIM, BATCH) != VAE_C_OP_OK) goto done;
    if (latest) {
        status = vae_c_iic_checkpoint_load(
            latest_path, &input->checkpoint_identity, &ensemble,
            &metadata);
        if (status != VAE_C_IIC_CHECKPOINT_OK ||
            metadata.kind != VAE_C_IIC_CHECKPOINT_LATEST ||
            metadata.last_completed_epoch >= options->max_epochs)
            goto done;
        start_epoch = metadata.last_completed_epoch + 1u;
    }
    if (!reconcile_metrics(metrics_path, latest,
                           metadata.last_completed_epoch)) goto done;
    if (latest && start_epoch == options->max_epochs) {
        metadata.kind = VAE_C_IIC_CHECKPOINT_FINAL;
        status = vae_c_iic_checkpoint_save_atomic(
            final_path, &input->checkpoint_identity, &metadata,
            &ensemble);
        if (status != VAE_C_IIC_CHECKPOINT_OK) goto done;
        puts("FINALIZED_FROM_TERMINAL_LATEST=true");
        ok = true;
        goto done;
    }
    printf("START_EPOCH=%" PRIu64 "\n", start_epoch);
    for (epoch = start_epoch; epoch < options->max_epochs; epoch++) {
        if (!run_epoch(input, &ensemble, epoch, options, metrics_path))
            goto done;
        metadata.kind = VAE_C_IIC_CHECKPOINT_LATEST;
        metadata.last_completed_epoch = epoch;
        status = vae_c_iic_checkpoint_save_atomic(
            latest_path, &input->checkpoint_identity, &metadata,
            &ensemble);
        if (status != VAE_C_IIC_CHECKPOINT_OK) goto done;
    }
    metadata.kind = VAE_C_IIC_CHECKPOINT_FINAL;
    status = vae_c_iic_checkpoint_save_atomic(
        final_path, &input->checkpoint_identity, &metadata,
        &ensemble);
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
    RunInput input;
    bool inside_repo = false, ok = false;
    if (!parse_options(argc, argv, &options)) {
        fputs("Invalid runner arguments\n", stderr);
        return 2;
    }
    if (options.mode == MODE_RUN &&
        !output_dir_policy(&options, &inside_repo)) {
        if (inside_repo && !options.official)
            puts("OUTPUT_DIR_INSIDE_REPO_REJECTED=true");
        fputs("Unsafe output directory\n", stderr);
        return 2;
    }
    if (!verify_git_head(options.repository_root, options.code_commit)) {
        fputs("Branch/HEAD mismatch\n", stderr);
        return 2;
    }
    run_input_init(&input);
    if (!prepare_input(&input, &options, argv[0])) goto done;
    printf("RUN_ID=%s\n", input.run_id);
    printf("PNG_CONTENT_IDENTITY_SHA256=%s\n",
           input.png_content_identity_sha256);
    printf("PRODUCER_BINARY_SHA256=%s\n", input.binary_sha256);
    if (options.mode == MODE_VALIDATE) {
        puts("VALIDATE_ONLY_PASS=true");
        puts("ENCODER_FORWARD_CALLS=0");
        puts("IIC_OPTIMIZER_STEPS_PER_PAIR=0");
        puts("OUTPUT_CREATED=false");
        ok = true;
    } else if (options.mode == MODE_PARTIAL) {
        ok = run_partial(&input, &options);
    } else {
        ok = run_controller(&input, &options);
    }
done:
    run_input_free(&input);
    if (!ok) fputs("MINI112 IIC runner failed\n", stderr);
    return ok ? 0 : 1;
}
