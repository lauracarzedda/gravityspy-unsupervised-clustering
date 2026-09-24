#include "vae_augmentation.h"
#include "vae_c_adam.h"
#include "vae_c_checkpoint.h"
#include "vae_c_config.h"
#include "vae_c_model_forward.h"
#include "vae_c_model_train.h"
#include "vae_c_rng.h"
#include "vae_c_shuffle.h"
#include "vae_event_dataset.h"
#include "vae_multires_batch.h"
#include "vae_multires_event_loader.h"
#include "vae_saved_split.h"
#include "vae_tensor_manifest.h"
#include "vae_tensor_spec.h"

#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <direct.h>
#include <windows.h>
#else
#include <sys/types.h>
#include <unistd.h>
#endif

enum {
    TRAINING_PATH_CAP = 4096,
    TRAINING_TEXT_CAP = 16384,
    TRAINING_LINE_CAP = 4096,
    TRAINING_SHA_SIZE = 32,
    TRAINING_CHANNELS = 4,
    TRAINING_SIDE = 112,
    TRAINING_INPUT_VALUES = 4 * 112 * 112,
    TRAINING_LATENT_VALUES = 64,
    TRAINING_TRAIN_COUNT = 6866,
    TRAINING_VALIDATION_COUNT = 1717,
    TRAINING_BATCH_SIZE = 64,
    TRAINING_TRAIN_TAIL = 18,
    TRAINING_VALIDATION_TAIL = 53,
    TRAINING_TRAIN_BATCHES = 108,
    TRAINING_VALIDATION_BATCHES = 27,
    TRAINING_MAX_EPOCHS = 100,
    TRAINING_PATIENCE = 10,
    TRAINING_MEAN_EVAL_EVERY = 5,
    TRAINING_PREVIEW_EVENTS = 2
};

#define TRAINING_PROFILE_ID "mini112_primary_v1"
#define TRAINING_OUTPUT_BASENAME \
    "mini112_primary_v1_full_init123_latent456_aug789_" \
    "shuffle101112_bs64"
#define TRAINING_INIT_SEED UINT64_C(123)
#define TRAINING_LATENT_SEED UINT64_C(456)
#define TRAINING_AUGMENTATION_SEED UINT64_C(789)
#define TRAINING_SHUFFLE_SEED UINT64_C(101112)

#define TRAINING_EVENT_SHA \
    "620960E6B35DA31BC4CF1A8A6DA8679027D378E1D4B2AC73C5267C1C5311AE82"
#define TRAINING_TENSOR_SHA \
    "9EB7F1C7E0491F6DF0DC346F1D405B8B5929A3CD26E8CE92FEB3C299875D301A"
#define TRAINING_TRAIN_SPLIT_SHA \
    "8C7A513B98DBE996261BC80965500AB15AB77F498EC9440858003152E42F926F"
#define TRAINING_VALIDATION_SPLIT_SHA \
    "E63109C88DF7B3EE97317F94791F26DF3FCAF8B937921CFB27967D01C2546467"
#define TRAINING_PROFILE_SHA \
    "E11D4BD722474BFBC50432B843679EF4F07E94A875D2E5ACE5B8CD85EFF6860F"

typedef struct {
    uint32_t state[8];
    uint64_t bit_count;
    unsigned char block[64];
    size_t block_size;
} TrainingSha256;

typedef struct {
    VAECModelWorkspace workspace;
    GSMultiresBatch batch;
    uint64_t *manifest_indices;
    size_t batch_size;
} TrainingSlot;

typedef struct {
    uint64_t sample_count;
    uint64_t input_value_count;
    uint64_t latent_value_count;
    double total_sum;
    double reconstruction_sum;
    double kl_sum;
} TrainingAggregate;

typedef struct {
    char phase[64];
    int64_t epoch;
    uint64_t optimizer_step;
    uint64_t sample_count;
    uint64_t input_value_count;
    uint64_t latent_value_count;
    double total_sum;
    double total_mean_per_sample;
    double reconstruction_sum;
    double reconstruction_mean_per_sample;
    double reconstruction_mean_per_input_value;
    double kl_sum;
    double kl_mean_per_sample;
    double kl_mean_per_latent_value;
} TrainingMetricRow;

typedef struct {
    VAECProfileCatalog catalog;
    const VAECProfile *profile;
    GSEventManifest events;
    GSSavedSplit train_split;
    GSSavedSplit validation_split;
    GSTensorManifest tensor_manifest;
    GSMultiresEventLoader event_loader;
    VAECModel model;
    VAECAdam optimizer;
    TrainingSlot train_full;
    TrainingSlot train_tail;
    TrainingSlot validation_full;
    TrainingSlot validation_tail;
    TrainingSlot preview;
    size_t *permutation;
    size_t *shuffled_indices;
    unsigned char *permutation_seen;
    GSSavedSplit shuffled_split;
    VAECCheckpointRunIdentity identity;
    VAECCheckpointMetadata controller;
    FILE *metrics_file;
    char repository_root[TRAINING_PATH_CAP];
    char final_output[TRAINING_PATH_CAP];
    char partial_output[TRAINING_PATH_CAP];
    bool partial_is_owned_run;
} TrainingRun;

static const char training_metrics_header[] =
    "phase,epoch,optimizer_step,sample_count,input_value_count,"
    "latent_value_count,total_sum,total_mean_per_sample,"
    "reconstruction_sum,reconstruction_mean_per_sample,"
    "reconstruction_mean_per_input_value,kl_sum,kl_mean_per_sample,"
    "kl_mean_per_latent_value\n";

static uint32_t training_rotate_right(uint32_t value, unsigned count)
{
    return (value >> count) | (value << (32u - count));
}

static void training_sha256_transform(
    TrainingSha256 *context,
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

    for (index = 0; index < 16; index++) {
        size_t offset = index * 4;
        words[index] =
            ((uint32_t)block[offset] << 24) |
            ((uint32_t)block[offset + 1] << 16) |
            ((uint32_t)block[offset + 2] << 8) |
            (uint32_t)block[offset + 3];
    }
    for (index = 16; index < 64; index++) {
        uint32_t s0 =
            training_rotate_right(words[index - 15], 7) ^
            training_rotate_right(words[index - 15], 18) ^
            (words[index - 15] >> 3);
        uint32_t s1 =
            training_rotate_right(words[index - 2], 17) ^
            training_rotate_right(words[index - 2], 19) ^
            (words[index - 2] >> 10);
        words[index] = words[index - 16] + s0 + words[index - 7] + s1;
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
        uint32_t sum1 = training_rotate_right(e, 6) ^
            training_rotate_right(e, 11) ^ training_rotate_right(e, 25);
        uint32_t choice = (e & f) ^ ((~e) & g);
        uint32_t temporary1 =
            h + sum1 + choice + constants[index] + words[index];
        uint32_t sum0 = training_rotate_right(a, 2) ^
            training_rotate_right(a, 13) ^ training_rotate_right(a, 22);
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

static void training_sha256_init(TrainingSha256 *context)
{
    memset(context, 0, sizeof(*context));
    context->state[0] = UINT32_C(0x6a09e667);
    context->state[1] = UINT32_C(0xbb67ae85);
    context->state[2] = UINT32_C(0x3c6ef372);
    context->state[3] = UINT32_C(0xa54ff53a);
    context->state[4] = UINT32_C(0x510e527f);
    context->state[5] = UINT32_C(0x9b05688c);
    context->state[6] = UINT32_C(0x1f83d9ab);
    context->state[7] = UINT32_C(0x5be0cd19);
}

static int training_sha256_update(
    TrainingSha256 *context,
    const unsigned char *data,
    size_t size
)
{
    size_t index;

    if (context == NULL || (size != 0 && data == NULL) ||
        size > (UINT64_MAX - context->bit_count) / UINT64_C(8)) {
        return 1;
    }
    context->bit_count += (uint64_t)size * UINT64_C(8);
    for (index = 0; index < size; index++) {
        context->block[context->block_size++] = data[index];
        if (context->block_size == sizeof(context->block)) {
            training_sha256_transform(context, context->block);
            context->block_size = 0;
        }
    }
    return 0;
}

static void training_sha256_final(
    TrainingSha256 *context,
    unsigned char digest[TRAINING_SHA_SIZE]
)
{
    uint64_t bit_count = context->bit_count;
    size_t index;

    context->block[context->block_size++] = 0x80u;
    if (context->block_size > 56) {
        while (context->block_size < 64) {
            context->block[context->block_size++] = 0;
        }
        training_sha256_transform(context, context->block);
        context->block_size = 0;
    }
    while (context->block_size < 56) {
        context->block[context->block_size++] = 0;
    }
    for (index = 0; index < 8; index++) {
        context->block[63 - index] =
            (unsigned char)(bit_count >> (8 * index));
    }
    training_sha256_transform(context, context->block);
    for (index = 0; index < 8; index++) {
        digest[index * 4] = (unsigned char)(context->state[index] >> 24);
        digest[index * 4 + 1] =
            (unsigned char)(context->state[index] >> 16);
        digest[index * 4 + 2] =
            (unsigned char)(context->state[index] >> 8);
        digest[index * 4 + 3] = (unsigned char)context->state[index];
    }
}

static int training_file_sha256(const char *path, char hex[65])
{
    static const char digits[] = "0123456789ABCDEF";
    TrainingSha256 context;
    unsigned char buffer[65536];
    unsigned char digest[TRAINING_SHA_SIZE];
    FILE *file;
    size_t count;
    size_t index;

    if (path == NULL || hex == NULL) {
        return 1;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        return 1;
    }
    training_sha256_init(&context);
    while ((count = fread(buffer, 1, sizeof(buffer), file)) != 0) {
        if (training_sha256_update(&context, buffer, count) != 0) {
            (void)fclose(file);
            return 1;
        }
    }
    if (ferror(file) || fclose(file) != 0) {
        return 1;
    }
    training_sha256_final(&context, digest);
    for (index = 0; index < sizeof(digest); index++) {
        hex[index * 2] = digits[digest[index] >> 4];
        hex[index * 2 + 1] = digits[digest[index] & 0x0f];
    }
    hex[64] = '\0';
    return 0;
}

static int training_size_multiply(
    size_t left,
    size_t right,
    size_t *result
)
{
    if (result == NULL || (left != 0 && right > SIZE_MAX / left)) {
        return 1;
    }
    *result = left * right;
    return 0;
}

static int training_path_join(
    const char *directory,
    const char *name,
    char *path,
    size_t capacity
)
{
    size_t directory_length;
    size_t name_length;
    size_t needed;
    bool separator;

    if (directory == NULL || name == NULL || path == NULL || capacity == 0) {
        return 1;
    }
    directory_length = strlen(directory);
    name_length = strlen(name);
    separator = directory_length != 0 &&
        directory[directory_length - 1] != '/' &&
        directory[directory_length - 1] != '\\';
    if (directory_length > SIZE_MAX - name_length) {
        return 1;
    }
    needed = directory_length + name_length;
    if (separator) {
        if (needed == SIZE_MAX) {
            return 1;
        }
        needed++;
    }
    if (needed >= capacity) {
        return 1;
    }
    memcpy(path, directory, directory_length);
    if (separator) {
        path[directory_length++] = '/';
    }
    memcpy(path + directory_length, name, name_length + 1);
    return 0;
}

static int training_path_append(
    const char *path,
    const char *suffix,
    char *output,
    size_t capacity
)
{
    size_t path_length;
    size_t suffix_length;

    if (path == NULL || suffix == NULL || output == NULL || capacity == 0) {
        return 1;
    }
    path_length = strlen(path);
    suffix_length = strlen(suffix);
    if (path_length > SIZE_MAX - suffix_length ||
        path_length + suffix_length >= capacity) {
        return 1;
    }
    memcpy(output, path, path_length);
    memcpy(output + path_length, suffix, suffix_length + 1);
    return 0;
}

static int training_path_state(const char *path, bool *is_directory)
{
    struct stat information;

    if (path == NULL) {
        return -1;
    }
    errno = 0;
    if (stat(path, &information) == 0) {
        if (is_directory != NULL) {
#ifdef _WIN32
            *is_directory = (information.st_mode & _S_IFDIR) != 0;
#else
            *is_directory = S_ISDIR(information.st_mode);
#endif
        }
        return 1;
    }
    if (errno == ENOENT) {
        if (is_directory != NULL) {
            *is_directory = false;
        }
        return 0;
    }
    return -1;
}

static int training_ensure_directory(const char *path)
{
    bool is_directory = false;
    int state = training_path_state(path, &is_directory);

    if (state == 1) {
        return is_directory ? 0 : 1;
    }
    if (state != 0) {
        return 1;
    }
#ifdef _WIN32
    return _mkdir(path) == 0 ? 0 : 1;
#else
    return mkdir(path, 0777) == 0 ? 0 : 1;
#endif
}

static int training_atomic_replace(const char *temporary, const char *target)
{
#ifdef _WIN32
    return MoveFileExA(
        temporary,
        target,
        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH
    ) ? 0 : 1;
#else
    return rename(temporary, target) == 0 ? 0 : 1;
#endif
}

static int training_code_commit_is_valid(const char *code_commit)
{
    size_t index;

    if (code_commit == NULL || strlen(code_commit) != 40) {
        return 0;
    }
    for (index = 0; index < 40; index++) {
        unsigned char value = (unsigned char)code_commit[index];
        if (!((value >= '0' && value <= '9') ||
              (value >= 'a' && value <= 'f') ||
              (value >= 'A' && value <= 'F'))) {
            return 0;
        }
    }
    return 1;
}

static int training_set_text(char *target, size_t capacity, const char *text)
{
    size_t length;

    if (target == NULL || capacity == 0 || text == NULL) {
        return 1;
    }
    length = strlen(text);
    if (length == 0 || length >= capacity) {
        return 1;
    }
    memcpy(target, text, length + 1);
    return 0;
}

static const char *training_compiler_target(void)
{
#if defined(__MINGW64__) && defined(_WIN64)
    return "x86_64-w64-mingw32-ucrt64";
#elif defined(__MINGW32__)
    return "mingw32-windows";
#elif defined(_WIN64)
    return "x86_64-windows";
#elif defined(_WIN32)
    return "x86-windows";
#else
    return "non-windows-c11";
#endif
}

static void training_slot_init(TrainingSlot *slot)
{
    if (slot != NULL) {
        memset(slot, 0, sizeof(*slot));
        vae_c_model_workspace_init(&slot->workspace);
    }
}

static void training_slot_free(TrainingSlot *slot)
{
    if (slot != NULL) {
        free(slot->manifest_indices);
        gs_multires_batch_free(&slot->batch);
        vae_c_model_workspace_free(&slot->workspace);
        training_slot_init(slot);
    }
}

static int training_slot_allocate(
    const VAECModel *model,
    const GSTensorSpec *spec,
    size_t batch_size,
    TrainingSlot *slot
)
{
    if (model == NULL || spec == NULL || slot == NULL || batch_size == 0 ||
        vae_c_model_workspace_allocate(
            model, batch_size, &slot->workspace) != VAE_C_OP_OK ||
        gs_multires_batch_alloc(&slot->batch, spec, batch_size) != 0) {
        return 1;
    }
    slot->manifest_indices = (uint64_t *)malloc(
        batch_size * sizeof(uint64_t)
    );
    if (slot->manifest_indices == NULL) {
        return 1;
    }
    slot->batch_size = batch_size;
    return 0;
}

static void training_run_init(TrainingRun *run)
{
    if (run != NULL) {
        memset(run, 0, sizeof(*run));
        vae_c_profile_catalog_init(&run->catalog);
        vae_c_model_init(&run->model);
        vae_c_adam_init(&run->optimizer);
        training_slot_init(&run->train_full);
        training_slot_init(&run->train_tail);
        training_slot_init(&run->validation_full);
        training_slot_init(&run->validation_tail);
        training_slot_init(&run->preview);
        vae_c_checkpoint_run_identity_init(&run->identity);
        vae_c_checkpoint_metadata_init(&run->controller);
    }
}

static void training_run_free(TrainingRun *run)
{
    if (run == NULL) {
        return;
    }
    if (run->metrics_file != NULL) {
        (void)fclose(run->metrics_file);
        run->metrics_file = NULL;
    }
    free(run->permutation_seen);
    free(run->shuffled_indices);
    free(run->permutation);
    training_slot_free(&run->preview);
    training_slot_free(&run->validation_tail);
    training_slot_free(&run->validation_full);
    training_slot_free(&run->train_tail);
    training_slot_free(&run->train_full);
    gs_multires_event_loader_free(&run->event_loader);
    vae_c_adam_free(&run->optimizer);
    vae_c_model_free(&run->model);
    gs_tensor_manifest_free(&run->tensor_manifest);
    gs_saved_split_free(&run->validation_split);
    gs_saved_split_free(&run->train_split);
    gs_event_manifest_free(&run->events);
    vae_c_profile_catalog_free(&run->catalog);
    memset(run, 0, sizeof(*run));
}

static int training_profile_is_expected(const VAECProfile *profile)
{
    return profile != NULL &&
        strcmp(profile->profile_id, TRAINING_PROFILE_ID) == 0 &&
        strcmp(profile->role, "primary_c_experiment") == 0 &&
        profile->input_side == TRAINING_SIDE &&
        profile->conv_depth == 3 &&
        profile->channels[0] == 16 &&
        profile->channels[1] == 32 &&
        profile->channels[2] == 64 &&
        profile->channels[3] == 0 &&
        profile->latent_dim == TRAINING_LATENT_VALUES &&
        profile->bottleneck_side == 7 &&
        profile->expected_trainable_parameters_bn1d == UINT64_C(262852) &&
        profile->expected_forward_mac_per_sample == UINT64_C(39948288) &&
        strcmp(profile->profile_status, "primary") == 0;
}

static int training_verify_hash(
    const char *path,
    const char *expected,
    const char *name
)
{
    char actual[65];

    if (training_file_sha256(path, actual) != 0 ||
        strcmp(actual, expected) != 0) {
        fprintf(stderr, "Errore SHA-256 %s.\n", name);
        return 1;
    }
    return 0;
}

static int training_build_identity(
    TrainingRun *run,
    const char *code_commit
)
{
    VAECCheckpointRunIdentity *identity = &run->identity;

    vae_c_checkpoint_run_identity_init(identity);
    if (training_set_text(identity->profile_id,
                          sizeof(identity->profile_id),
                          TRAINING_PROFILE_ID) != 0 ||
        training_set_text(identity->profile_config_sha256,
                          sizeof(identity->profile_config_sha256),
                          TRAINING_PROFILE_SHA) != 0 ||
        training_set_text(identity->code_commit,
                          sizeof(identity->code_commit), code_commit) != 0 ||
        training_set_text(identity->dataset_id,
                          sizeof(identity->dataset_id),
                          "gravity_spy_o1_official_8583") != 0 ||
        training_set_text(identity->event_manifest_id,
                          sizeof(identity->event_manifest_id),
                          "manifest_events_272.csv") != 0 ||
        training_set_text(identity->event_manifest_sha256,
                          sizeof(identity->event_manifest_sha256),
                          TRAINING_EVENT_SHA) != 0 ||
        training_set_text(identity->tensor_manifest_id,
                          sizeof(identity->tensor_manifest_id),
                          "manifest_tensors_112_central.csv") != 0 ||
        training_set_text(identity->tensor_manifest_sha256,
                          sizeof(identity->tensor_manifest_sha256),
                          TRAINING_TENSOR_SHA) != 0 ||
        training_set_text(identity->train_split_id,
                          sizeof(identity->train_split_id),
                          "split_train_indices_seed123.csv") != 0 ||
        training_set_text(identity->train_split_sha256,
                          sizeof(identity->train_split_sha256),
                          TRAINING_TRAIN_SPLIT_SHA) != 0 ||
        training_set_text(identity->validation_split_id,
                          sizeof(identity->validation_split_id),
                          "split_val_indices_seed123.csv") != 0 ||
        training_set_text(identity->validation_split_sha256,
                          sizeof(identity->validation_split_sha256),
                          TRAINING_VALIDATION_SPLIT_SHA) != 0 ||
        training_set_text(identity->augmentation_policy,
                          sizeof(identity->augmentation_policy),
                          "stateless_event_epoch_v1+symmetric_figure+side112") != 0 ||
        training_set_text(identity->shuffle_policy,
                          sizeof(identity->shuffle_policy),
                          "fisher_yates_unbiased_v1") != 0 ||
        training_set_text(identity->latent_policy,
                          sizeof(identity->latent_policy),
                          "train_event_epoch_v1+validation_event_v1+latent64") != 0 ||
        training_set_text(identity->validation_policy,
                          sizeof(identity->validation_policy),
                          "central_stochastic_fixed_epsilon_v1+max100+patience10") != 0 ||
        training_set_text(identity->compiler_id,
                          sizeof(identity->compiler_id), "GNU C") != 0 ||
        training_set_text(identity->compiler_version,
                          sizeof(identity->compiler_version), __VERSION__) != 0 ||
        training_set_text(identity->compiler_target,
                          sizeof(identity->compiler_target),
                          training_compiler_target()) != 0) {
        return 1;
    }
    identity->initialization_seed = TRAINING_INIT_SEED;
    identity->latent_seed = TRAINING_LATENT_SEED;
    identity->augmentation_seed = TRAINING_AUGMENTATION_SEED;
    identity->shuffle_seed = TRAINING_SHUFFLE_SEED;
    identity->max_batch_size = TRAINING_BATCH_SIZE;
    identity->batch_size = TRAINING_BATCH_SIZE;
    identity->optimizer_learning_rate = VAE_C_ADAM_LEARNING_RATE_V1;
    identity->optimizer_beta1 = VAE_C_ADAM_BETA1_V1;
    identity->optimizer_beta2 = VAE_C_ADAM_BETA2_V1;
    identity->optimizer_epsilon = VAE_C_ADAM_EPSILON_V1;
    return 0;
}

static int training_load_contract(
    TrainingRun *run,
    const char *repository_root,
    const char *code_commit
)
{
    char profile_path[TRAINING_PATH_CAP];
    char event_path[TRAINING_PATH_CAP];
    char train_path[TRAINING_PATH_CAP];
    char validation_path[TRAINING_PATH_CAP];
    char tensor_path[TRAINING_PATH_CAP];
    char error_message[512];
    GSTensorSpec expected_spec;
    size_t index;
    VAECOpStatus status;

    if (strlen(repository_root) >= sizeof(run->repository_root) ||
        training_path_join(repository_root,
            "sakai_multires_c/configs/vae_c_profiles_v1.csv",
            profile_path, sizeof(profile_path)) != 0 ||
        training_path_join(repository_root, "manifest_events_272.csv",
            event_path, sizeof(event_path)) != 0 ||
        training_path_join(repository_root, "split_train_indices_seed123.csv",
            train_path, sizeof(train_path)) != 0 ||
        training_path_join(repository_root, "split_val_indices_seed123.csv",
            validation_path, sizeof(validation_path)) != 0 ||
        training_path_join(repository_root, "manifest_tensors_112_central.csv",
            tensor_path, sizeof(tensor_path)) != 0) {
        return 1;
    }
    memcpy(run->repository_root, repository_root, strlen(repository_root) + 1);
    if (training_verify_hash(event_path, TRAINING_EVENT_SHA,
                             "manifest eventi") != 0 ||
        training_verify_hash(tensor_path, TRAINING_TENSOR_SHA,
                             "manifest tensor") != 0 ||
        training_verify_hash(train_path, TRAINING_TRAIN_SPLIT_SHA,
                             "split train") != 0 ||
        training_verify_hash(validation_path, TRAINING_VALIDATION_SPLIT_SHA,
                             "split validation") != 0 ||
        training_verify_hash(profile_path, TRAINING_PROFILE_SHA,
                             "catalogo profili") != 0 ||
        training_build_identity(run, code_commit) != 0) {
        return 1;
    }
    error_message[0] = '\0';
    if (vae_c_profile_catalog_load_csv(
            profile_path, &run->catalog,
            error_message, sizeof(error_message)) != VAE_C_CONFIG_OK) {
        fprintf(stderr, "Errore catalogo profili: %s\n", error_message);
        return 1;
    }
    run->profile = vae_c_profile_catalog_find(
        &run->catalog, TRAINING_PROFILE_ID
    );
    if (!training_profile_is_expected(run->profile) ||
        gs_event_manifest_load(event_path, &run->events) != 0 ||
        run->events.count != 8583 ||
        gs_saved_split_load(train_path, "train", &run->events,
                            &run->train_split) != 0 ||
        run->train_split.count != TRAINING_TRAIN_COUNT ||
        gs_saved_split_load(validation_path, "validation", &run->events,
                            &run->validation_split) != 0 ||
        run->validation_split.count != TRAINING_VALIDATION_COUNT ||
        gs_tensor_spec_init(&expected_spec, TRAINING_SIDE) != 0 ||
        gs_tensor_manifest_load(tensor_path, &expected_spec,
                                &run->tensor_manifest) != 0 ||
        run->tensor_manifest.count != run->events.count ||
        strcmp(run->tensor_manifest.extra_column_name, "origin") != 0) {
        fprintf(stderr, "Errore: contratto dataset/profilo non conforme.\n");
        return 1;
    }
    for (index = 0; index < run->tensor_manifest.count; index++) {
        const GSTensorManifestRecord *record =
            &run->tensor_manifest.records[index];
        if (record->channels != TRAINING_CHANNELS ||
            record->height != TRAINING_SIDE ||
            record->width != TRAINING_SIDE ||
            strcmp(record->dtype, "uint8") != 0 ||
            strcmp(record->layout, "channel_major") != 0 ||
            strcmp(record->channel_order, "0.5;1.0;2.0;4.0") != 0 ||
            strcmp(record->extra_value, "derived_from_224_central") != 0) {
            fprintf(stderr, "Errore record tensor %lu.\n",
                    (unsigned long)index);
            return 1;
        }
    }
    status = vae_c_model_allocate(
        &run->model, run->profile, TRAINING_BATCH_SIZE
    );
    if (status == VAE_C_OP_OK) {
        status = vae_c_model_initialize_parameters_v1(
            &run->model, TRAINING_INIT_SEED
        );
    }
    if (status == VAE_C_OP_OK) {
        status = vae_c_adam_configure_v1(&run->optimizer);
    }
    if (status != VAE_C_OP_OK ||
        run->optimizer.learning_rate != 5.0e-4 ||
        run->optimizer.beta1 != 0.9 ||
        run->optimizer.beta2 != 0.999 ||
        run->optimizer.adam_epsilon != 1.0e-8) {
        fprintf(stderr, "Errore setup modello/Adam: %s\n",
                vae_c_op_status_name(status));
        return 1;
    }
    return 0;
}

static int training_allocate_runtime(TrainingRun *run)
{
    if (gs_multires_event_loader_init(
            &run->event_loader, TRAINING_SIDE) != 0 ||
        training_slot_allocate(&run->model, &run->tensor_manifest.spec,
            TRAINING_BATCH_SIZE, &run->train_full) != 0 ||
        training_slot_allocate(&run->model, &run->tensor_manifest.spec,
            TRAINING_TRAIN_TAIL, &run->train_tail) != 0 ||
        training_slot_allocate(&run->model, &run->tensor_manifest.spec,
            TRAINING_BATCH_SIZE, &run->validation_full) != 0 ||
        training_slot_allocate(&run->model, &run->tensor_manifest.spec,
            TRAINING_VALIDATION_TAIL, &run->validation_tail) != 0 ||
        training_slot_allocate(&run->model, &run->tensor_manifest.spec,
            TRAINING_PREVIEW_EVENTS, &run->preview) != 0) {
        return 1;
    }
    run->permutation = (size_t *)malloc(
        run->train_split.count * sizeof(size_t)
    );
    run->shuffled_indices = (size_t *)malloc(
        run->train_split.count * sizeof(size_t)
    );
    run->permutation_seen = (unsigned char *)malloc(run->train_split.count);
    if (run->permutation == NULL || run->shuffled_indices == NULL ||
        run->permutation_seen == NULL) {
        return 1;
    }
    memset(&run->shuffled_split, 0, sizeof(run->shuffled_split));
    run->shuffled_split.indices = run->shuffled_indices;
    run->shuffled_split.count = run->train_split.count;
    memcpy(run->shuffled_split.name, "train", sizeof("train"));
    return 0;
}

static int training_prepare_epoch(TrainingRun *run, uint64_t epoch)
{
    size_t index;

    if (vae_c_shuffle_make_permutation(
            run->train_split.count,
            TRAINING_SHUFFLE_SEED,
            epoch,
            run->permutation) != VAE_C_RNG_OK) {
        return 1;
    }
    memset(run->permutation_seen, 0, run->train_split.count);
    for (index = 0; index < run->train_split.count; index++) {
        size_t position = run->permutation[index];
        if (position >= run->train_split.count ||
            run->permutation_seen[position] != 0) {
            return 1;
        }
        run->permutation_seen[position] = 1;
        run->shuffled_indices[index] = run->train_split.indices[position];
    }
    return 0;
}

static int training_copy_batch_to_workspace(TrainingSlot *slot)
{
    size_t expected_values;
    size_t expected_bytes;
    size_t sample;

    if (slot == NULL || slot->batch.count != slot->batch_size ||
        training_size_multiply(slot->batch_size,
            slot->batch.spec.values_per_event, &expected_values) != 0 ||
        expected_values != slot->workspace.input.shape.count ||
        training_size_multiply(expected_values, sizeof(float),
                               &expected_bytes) != 0) {
        return 1;
    }
    memcpy(slot->workspace.input.data, slot->batch.x, expected_bytes);
    for (sample = 0; sample < slot->batch_size; sample++) {
        slot->manifest_indices[sample] =
            (uint64_t)slot->batch.manifest_indices[sample];
    }
    return 0;
}

static int training_load_train_slot(
    TrainingRun *run,
    TrainingSlot *slot,
    size_t start,
    uint64_t epoch
)
{
    size_t sample;

    if (gs_multires_batch_load_train_stateless_mode(
            &run->events, &run->shuffled_split, &run->event_loader,
            start, slot->batch_size, TRAINING_AUGMENTATION_SEED, epoch,
            GS_AUGMENT_MODE_SYMMETRIC_FIGURE, &slot->batch) != 0 ||
        slot->batch.count != slot->batch_size) {
        return 1;
    }
    for (sample = 0; sample < slot->batch_size; sample++) {
        int expected_crop = -1;
        size_t expected_index = run->shuffled_indices[start + sample];
        if (slot->batch.manifest_indices[sample] != expected_index ||
            gs_augment_crop_left_for_event_mode(
                TRAINING_AUGMENTATION_SEED, epoch,
                (uint64_t)expected_index,
                GS_AUGMENT_MODE_SYMMETRIC_FIGURE,
                &expected_crop) != 0 ||
            expected_crop < 0 || expected_crop > 48 ||
            slot->batch.crop_lefts[sample] != expected_crop) {
            return 1;
        }
    }
    return training_copy_batch_to_workspace(slot);
}

static int training_fill_validation_epsilon(TrainingSlot *slot)
{
    size_t sample;

    for (sample = 0; sample < slot->batch_size; sample++) {
        if (vae_c_rng_fill_latent_validation_epsilon_f32(
                TRAINING_LATENT_SEED,
                slot->manifest_indices[sample],
                slot->workspace.epsilon.data +
                    sample * TRAINING_LATENT_VALUES,
                TRAINING_LATENT_VALUES) != VAE_C_RNG_OK) {
            return 1;
        }
    }
    return 0;
}

static int training_load_validation_slot(
    TrainingRun *run,
    TrainingSlot *slot,
    size_t start,
    bool fill_epsilon
)
{
    size_t sample;

    if (gs_multires_batch_load_validation(
            &run->events, &run->tensor_manifest,
            &run->validation_split, start, slot->batch_size,
            &slot->batch) != 0 ||
        slot->batch.count != slot->batch_size) {
        return 1;
    }
    for (sample = 0; sample < slot->batch_size; sample++) {
        if (slot->batch.manifest_indices[sample] !=
            run->validation_split.indices[start + sample]) {
            return 1;
        }
    }
    if (training_copy_batch_to_workspace(slot) != 0) {
        return 1;
    }
    return fill_epsilon ? training_fill_validation_epsilon(slot) : 0;
}

static void training_aggregate_init(TrainingAggregate *aggregate)
{
    if (aggregate != NULL) {
        memset(aggregate, 0, sizeof(*aggregate));
    }
}

static int training_stats_are_valid(
    const VAECVAELossStats *stats,
    size_t batch_size
)
{
    return stats != NULL && stats->batch_size == batch_size &&
        stats->beta_kl == 1.0f &&
        stats->reconstruction.batch_size == batch_size &&
        stats->reconstruction.values_per_sample == TRAINING_INPUT_VALUES &&
        stats->reconstruction.total_values ==
            batch_size * TRAINING_INPUT_VALUES &&
        stats->kl.batch_size == batch_size &&
        stats->kl.latent_dim == TRAINING_LATENT_VALUES &&
        stats->kl.total_values == batch_size * TRAINING_LATENT_VALUES &&
        isfinite(stats->total_sum) &&
        isfinite(stats->reconstruction.total_sum) &&
        isfinite(stats->kl.total_sum);
}

static int training_aggregate_add(
    TrainingAggregate *aggregate,
    const VAECVAELossStats *stats
)
{
    uint64_t batch_size;
    uint64_t input_values;
    uint64_t latent_values;

    if (aggregate == NULL || stats == NULL ||
        !training_stats_are_valid(stats, stats->batch_size)) {
        return 1;
    }
    batch_size = (uint64_t)stats->batch_size;
    input_values = (uint64_t)stats->reconstruction.total_values;
    latent_values = (uint64_t)stats->kl.total_values;
    if (aggregate->sample_count > UINT64_MAX - batch_size ||
        aggregate->input_value_count > UINT64_MAX - input_values ||
        aggregate->latent_value_count > UINT64_MAX - latent_values) {
        return 1;
    }
    aggregate->sample_count += batch_size;
    aggregate->input_value_count += input_values;
    aggregate->latent_value_count += latent_values;
    aggregate->total_sum += stats->total_sum;
    aggregate->reconstruction_sum += stats->reconstruction.total_sum;
    aggregate->kl_sum += stats->kl.total_sum;
    return isfinite(aggregate->total_sum) &&
        isfinite(aggregate->reconstruction_sum) &&
        isfinite(aggregate->kl_sum) ? 0 : 1;
}

static int training_aggregate_is_complete(
    const TrainingAggregate *aggregate,
    uint64_t expected_samples
)
{
    return aggregate != NULL && expected_samples != 0 &&
        aggregate->sample_count == expected_samples &&
        aggregate->input_value_count ==
            expected_samples * TRAINING_INPUT_VALUES &&
        aggregate->latent_value_count ==
            expected_samples * TRAINING_LATENT_VALUES &&
        isfinite(aggregate->total_sum) &&
        isfinite(aggregate->reconstruction_sum) &&
        isfinite(aggregate->kl_sum) &&
        aggregate->total_sum >= 0.0 &&
        aggregate->reconstruction_sum >= 0.0 &&
        aggregate->kl_sum >= 0.0;
}

static int training_run_epoch(
    TrainingRun *run,
    uint64_t epoch,
    TrainingAggregate *aggregate
)
{
    size_t start;
    uint64_t step_before = run->optimizer.step;

    training_aggregate_init(aggregate);
    if (training_prepare_epoch(run, epoch) != 0) {
        return 1;
    }
    for (start = 0; start < run->train_split.count;) {
        size_t remaining = run->train_split.count - start;
        TrainingSlot *slot = remaining >= TRAINING_BATCH_SIZE
            ? &run->train_full : &run->train_tail;
        VAECVAELossStats stats;

        if (remaining < TRAINING_BATCH_SIZE &&
            remaining != TRAINING_TRAIN_TAIL) {
            return 1;
        }
        if (training_load_train_slot(run, slot, start, epoch) != 0) {
            return 1;
        }
        vae_c_vae_loss_stats_init(&stats);
        if (vae_c_model_train_step(
                &run->model, &slot->workspace, &run->optimizer,
                TRAINING_LATENT_SEED, epoch, slot->manifest_indices,
                slot->batch_size, &stats) != VAE_C_OP_OK ||
            training_aggregate_add(aggregate, &stats) != 0) {
            return 1;
        }
        start += slot->batch_size;
    }
    return training_aggregate_is_complete(
            aggregate, TRAINING_TRAIN_COUNT) &&
        run->optimizer.step == step_before + TRAINING_TRAIN_BATCHES
        ? 0 : 1;
}

static int training_evaluate(
    TrainingRun *run,
    VAECLatentMode latent_mode,
    TrainingAggregate *aggregate
)
{
    size_t start;

    training_aggregate_init(aggregate);
    for (start = 0; start < run->validation_split.count;) {
        size_t remaining = run->validation_split.count - start;
        TrainingSlot *slot = remaining >= TRAINING_BATCH_SIZE
            ? &run->validation_full : &run->validation_tail;
        VAECVAELossStats stats;

        if (remaining < TRAINING_BATCH_SIZE &&
            remaining != TRAINING_VALIDATION_TAIL) {
            return 1;
        }
        if (training_load_validation_slot(
                run, slot, start,
                latent_mode == VAE_C_LATENT_MODE_STOCHASTIC) != 0) {
            return 1;
        }
        vae_c_vae_loss_stats_init(&stats);
        if (vae_c_model_vae_forward(
                &run->model, VAE_C_BATCH_NORM_MODE_EVAL,
                latent_mode, &slot->workspace, &stats) != VAE_C_OP_OK ||
            training_aggregate_add(aggregate, &stats) != 0) {
            return 1;
        }
        start += slot->batch_size;
    }
    return training_aggregate_is_complete(
        aggregate, TRAINING_VALIDATION_COUNT) ? 0 : 1;
}

static int training_metric_from_aggregate(
    const char *phase,
    int64_t epoch,
    uint64_t optimizer_step,
    const TrainingAggregate *aggregate,
    TrainingMetricRow *row
)
{
    if (phase == NULL || aggregate == NULL || row == NULL ||
        strlen(phase) >= sizeof(row->phase) ||
        aggregate->sample_count == 0 ||
        aggregate->input_value_count == 0 ||
        aggregate->latent_value_count == 0) {
        return 1;
    }
    memset(row, 0, sizeof(*row));
    memcpy(row->phase, phase, strlen(phase) + 1);
    row->epoch = epoch;
    row->optimizer_step = optimizer_step;
    row->sample_count = aggregate->sample_count;
    row->input_value_count = aggregate->input_value_count;
    row->latent_value_count = aggregate->latent_value_count;
    row->total_sum = aggregate->total_sum;
    row->total_mean_per_sample =
        aggregate->total_sum / (double)aggregate->sample_count;
    row->reconstruction_sum = aggregate->reconstruction_sum;
    row->reconstruction_mean_per_sample =
        aggregate->reconstruction_sum / (double)aggregate->sample_count;
    row->reconstruction_mean_per_input_value =
        aggregate->reconstruction_sum /
        (double)aggregate->input_value_count;
    row->kl_sum = aggregate->kl_sum;
    row->kl_mean_per_sample =
        aggregate->kl_sum / (double)aggregate->sample_count;
    row->kl_mean_per_latent_value =
        aggregate->kl_sum / (double)aggregate->latent_value_count;
    return 0;
}

static int training_write_metric(
    FILE *file,
    const TrainingMetricRow *row
)
{
    if (file == NULL || row == NULL ||
        fprintf(file,
            "%s,%" PRId64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
            ",%" PRIu64 ",%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
            row->phase, row->epoch, row->optimizer_step,
            row->sample_count, row->input_value_count,
            row->latent_value_count, row->total_sum,
            row->total_mean_per_sample, row->reconstruction_sum,
            row->reconstruction_mean_per_sample,
            row->reconstruction_mean_per_input_value,
            row->kl_sum, row->kl_mean_per_sample,
            row->kl_mean_per_latent_value) < 0 ||
        fflush(file) != 0 || ferror(file)) {
        return 1;
    }
    return 0;
}

static int training_build_output_paths(TrainingRun *run)
{
    char results[TRAINING_PATH_CAP];
    char vae_c[TRAINING_PATH_CAP];
    char training[TRAINING_PATH_CAP];

    return training_path_join(run->repository_root,
               "sakai_multires_c/results", results, sizeof(results)) != 0 ||
        training_path_join(results, "vae_c", vae_c, sizeof(vae_c)) != 0 ||
        training_path_join(vae_c, "training",
                           training, sizeof(training)) != 0 ||
        training_path_join(training, TRAINING_OUTPUT_BASENAME,
                           run->final_output,
                           sizeof(run->final_output)) != 0 ||
        training_path_append(run->final_output, ".partial",
                             run->partial_output,
                             sizeof(run->partial_output)) != 0;
}

static int training_build_run_config(
    const TrainingRun *run,
    char *text,
    size_t capacity
)
{
    int written = snprintf(
        text, capacity,
        "format=vae_c_mini112_training_v1\n"
        "code_commit=%s\n"
        "profile=%s\n"
        "input_channels=4\ninput_side=112\nconv_depth=3\n"
        "channels=16,32,64\nlatent_dim=64\nbottleneck_side=7\n"
        "trainable_parameters=262852\n"
        "train_count=%d\nvalidation_count=%d\nbatch_size=%d\n"
        "train_batches_per_epoch=%d\nvalidation_batches_per_epoch=%d\n"
        "train_last_batch=%d\nvalidation_last_batch=%d\n"
        "init_seed=%" PRIu64 "\nlatent_seed=%" PRIu64 "\n"
        "augmentation_seed=%" PRIu64 "\nshuffle_seed=%" PRIu64 "\n"
        "augmentation_policy=stateless_event_epoch_v1\n"
        "augmentation_mode=symmetric_figure\n"
        "shuffle_policy=fisher_yates_unbiased_v1\n"
        "latent_train_policy=event_epoch_stateless_v1\n"
        "latent_validation_policy=event_fixed_stateless_v1\n"
        "validation_policy=central_stochastic_eval_fixed_epsilon\n"
        "diagnostic_policy=central_mean_eval\n"
        "adam_learning_rate=%.17g\nadam_beta1=%.17g\n"
        "adam_beta2=%.17g\nadam_epsilon=%.17g\nbeta_kl=1\n"
        "max_epochs=%d\nearly_stopping_patience=%d\n"
        "validation_every=1\nmean_eval_every=%d\n"
        "epoch_indexing=zero_based_0_99\n"
        "best_policy=strict_min_validation_total_mean_per_sample\n"
        "best_tie_break=earliest_epoch\n"
        "checkpoint_resume_boundary=epoch\n"
        "latest_is_resume_authority=true\nrng_state_serialized=false\n"
        "labels_used=false\n"
        "event_manifest=manifest_events_272.csv\n"
        "event_manifest_sha256=%s\n"
        "tensor_manifest=manifest_tensors_112_central.csv\n"
        "tensor_manifest_sha256=%s\n"
        "train_split=split_train_indices_seed123.csv\n"
        "train_split_sha256=%s\n"
        "validation_split=split_val_indices_seed123.csv\n"
        "validation_split_sha256=%s\n"
        "profile_catalog=sakai_multires_c/configs/vae_c_profiles_v1.csv\n"
        "profile_catalog_sha256=%s\n"
        "compiler_id=%s\ncompiler_version=%s\ncompiler_target=%s\n",
        run->identity.code_commit, TRAINING_PROFILE_ID,
        TRAINING_TRAIN_COUNT, TRAINING_VALIDATION_COUNT,
        TRAINING_BATCH_SIZE, TRAINING_TRAIN_BATCHES,
        TRAINING_VALIDATION_BATCHES, TRAINING_TRAIN_TAIL,
        TRAINING_VALIDATION_TAIL, TRAINING_INIT_SEED,
        TRAINING_LATENT_SEED, TRAINING_AUGMENTATION_SEED,
        TRAINING_SHUFFLE_SEED, run->optimizer.learning_rate,
        run->optimizer.beta1, run->optimizer.beta2,
        run->optimizer.adam_epsilon, TRAINING_MAX_EPOCHS,
        TRAINING_PATIENCE, TRAINING_MEAN_EVAL_EVERY,
        TRAINING_EVENT_SHA, TRAINING_TENSOR_SHA,
        TRAINING_TRAIN_SPLIT_SHA, TRAINING_VALIDATION_SPLIT_SHA,
        TRAINING_PROFILE_SHA, run->identity.compiler_id,
        run->identity.compiler_version, run->identity.compiler_target
    );
    return written < 0 || (size_t)written >= capacity ? 1 : 0;
}

static int training_write_exclusive_text(
    const char *path,
    const char *text
)
{
    FILE *file = fopen(path, "wbx");
    int failed = 0;
    size_t length = text == NULL ? 0 : strlen(text);

    if (file == NULL || text == NULL) {
        if (file != NULL) {
            (void)fclose(file);
        }
        return 1;
    }
    if (fwrite(text, 1, length, file) != length ||
        fflush(file) != 0 || ferror(file)) {
        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    if (failed) {
        (void)remove(path);
    }
    return failed;
}

static int training_file_equals_text(const char *path, const char *expected)
{
    FILE *file;
    size_t expected_length;
    size_t index = 0;
    int character;

    if (path == NULL || expected == NULL) {
        return 0;
    }
    expected_length = strlen(expected);
    file = fopen(path, "rb");
    if (file == NULL) {
        return 0;
    }
    while ((character = fgetc(file)) != EOF) {
        if (index >= expected_length ||
            (unsigned char)character != (unsigned char)expected[index]) {
            (void)fclose(file);
            return 0;
        }
        index++;
    }
    if (ferror(file) || fclose(file) != 0) {
        return 0;
    }
    return index == expected_length;
}

static int training_prepare_fresh_output(TrainingRun *run)
{
    char results[TRAINING_PATH_CAP];
    char vae_c[TRAINING_PATH_CAP];
    char training[TRAINING_PATH_CAP];
    char config_path[TRAINING_PATH_CAP];
    char metrics_path[TRAINING_PATH_CAP];
    char config[TRAINING_TEXT_CAP];
    bool ignored;

    if (training_path_join(run->repository_root,
            "sakai_multires_c/results", results, sizeof(results)) != 0 ||
        training_path_join(results, "vae_c", vae_c, sizeof(vae_c)) != 0 ||
        training_path_join(vae_c, "training", training, sizeof(training)) != 0 ||
        training_path_state(run->final_output, &ignored) != 0 ||
        training_path_state(run->partial_output, &ignored) != 0 ||
        training_ensure_directory(results) != 0 ||
        training_ensure_directory(vae_c) != 0 ||
        training_ensure_directory(training) != 0 ||
        training_ensure_directory(run->partial_output) != 0 ||
        training_path_join(run->partial_output, "run_config.txt",
                           config_path, sizeof(config_path)) != 0 ||
        training_path_join(run->partial_output, "metrics.csv",
                           metrics_path, sizeof(metrics_path)) != 0 ||
        training_build_run_config(run, config, sizeof(config)) != 0 ||
        training_write_exclusive_text(config_path, config) != 0 ||
        training_write_exclusive_text(metrics_path,
                                      training_metrics_header) != 0) {
        fprintf(stderr, "Errore creazione output partial.\n");
        return 1;
    }
    run->partial_is_owned_run = true;
    run->metrics_file = fopen(metrics_path, "ab");
    return run->metrics_file == NULL ? 1 : 0;
}

static int training_open_resume_output(TrainingRun *run)
{
    char config_path[TRAINING_PATH_CAP];
    char metrics_path[TRAINING_PATH_CAP];
    char config[TRAINING_TEXT_CAP];
    bool is_directory = false;

    if (training_path_state(run->final_output, &is_directory) != 0 ||
        training_path_state(run->partial_output, &is_directory) != 1 ||
        !is_directory ||
        training_path_join(run->partial_output, "run_config.txt",
                           config_path, sizeof(config_path)) != 0 ||
        training_path_join(run->partial_output, "metrics.csv",
                           metrics_path, sizeof(metrics_path)) != 0 ||
        training_build_run_config(run, config, sizeof(config)) != 0 ||
        !training_file_equals_text(config_path, config)) {
        fprintf(stderr, "Errore identity/run_config del resume.\n");
        return 1;
    }
    run->partial_is_owned_run = true;
    return 0;
}

static int training_parse_metric(
    const char *line,
    TrainingMetricRow *row
)
{
    int consumed = 0;
    int conversions;
    const char *tail;

    memset(row, 0, sizeof(*row));
    conversions = sscanf(
        line,
        "%63[^,],%" SCNd64 ",%" SCNu64 ",%" SCNu64 ",%" SCNu64
        ",%" SCNu64 ",%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf%n",
        row->phase, &row->epoch, &row->optimizer_step,
        &row->sample_count, &row->input_value_count,
        &row->latent_value_count, &row->total_sum,
        &row->total_mean_per_sample, &row->reconstruction_sum,
        &row->reconstruction_mean_per_sample,
        &row->reconstruction_mean_per_input_value,
        &row->kl_sum, &row->kl_mean_per_sample,
        &row->kl_mean_per_latent_value, &consumed
    );
    if (conversions != 14 || consumed <= 0) {
        return 1;
    }
    tail = line + consumed;
    while (*tail == '\r' || *tail == '\n') {
        tail++;
    }
    return *tail == '\0' ? 0 : 1;
}

static int training_double_close(double left, double right)
{
    double scale = fmax(1.0, fmax(fabs(left), fabs(right)));
    return isfinite(left) && isfinite(right) &&
        fabs(left - right) <= 1.0e-12 * scale;
}

static int training_metric_numeric_valid(const TrainingMetricRow *row)
{
    if (row == NULL || row->sample_count == 0 ||
        row->sample_count > UINT64_MAX / TRAINING_INPUT_VALUES ||
        row->sample_count > UINT64_MAX / TRAINING_LATENT_VALUES ||
        row->input_value_count !=
            row->sample_count * TRAINING_INPUT_VALUES ||
        row->latent_value_count !=
            row->sample_count * TRAINING_LATENT_VALUES ||
        !isfinite(row->total_sum) ||
        !isfinite(row->reconstruction_sum) ||
        !isfinite(row->kl_sum) ||
        row->total_sum < 0.0 || row->reconstruction_sum < 0.0 ||
        row->kl_sum < 0.0) {
        return 0;
    }
    return training_double_close(
            row->total_sum,
            row->reconstruction_sum + row->kl_sum) &&
        training_double_close(
            row->total_mean_per_sample,
            row->total_sum / (double)row->sample_count) &&
        training_double_close(
            row->reconstruction_mean_per_sample,
            row->reconstruction_sum / (double)row->sample_count) &&
        training_double_close(
            row->reconstruction_mean_per_input_value,
            row->reconstruction_sum / (double)row->input_value_count) &&
        training_double_close(
            row->kl_mean_per_sample,
            row->kl_sum / (double)row->sample_count) &&
        training_double_close(
            row->kl_mean_per_latent_value,
            row->kl_sum / (double)row->latent_value_count);
}

static int training_reconcile_metrics(
    TrainingRun *run,
    uint64_t latest_epoch,
    bool *final_row_present
)
{
    char metrics_path[TRAINING_PATH_CAP];
    char temporary_path[TRAINING_PATH_CAP];
    char line[TRAINING_LINE_CAP];
    FILE *input = NULL;
    FILE *output = NULL;
    bool train_seen[TRAINING_MAX_EPOCHS] = {false};
    bool validation_seen[TRAINING_MAX_EPOCHS] = {false};
    bool mean_seen[TRAINING_MAX_EPOCHS] = {false};
    double validation_means[TRAINING_MAX_EPOCHS] = {0.0};
    bool initial_stochastic = false;
    bool initial_mean = false;
    bool trailing = false;
    bool any_dropped = false;
    int result = 1;

    if (run == NULL || final_row_present == NULL ||
        latest_epoch >= TRAINING_MAX_EPOCHS) {
        return 1;
    }
    *final_row_present = false;
    if (
        training_path_join(run->partial_output, "metrics.csv",
                           metrics_path, sizeof(metrics_path)) != 0 ||
        training_path_append(metrics_path, ".reconcile.tmp",
                             temporary_path, sizeof(temporary_path)) != 0 ||
        training_path_state(temporary_path, NULL) != 0) {
        return 1;
    }
    input = fopen(metrics_path, "rb");
    output = fopen(temporary_path, "wbx");
    if (input == NULL || output == NULL ||
        fgets(line, sizeof(line), input) == NULL ||
        strcmp(line, training_metrics_header) != 0 ||
        fputs(line, output) == EOF) {
        goto cleanup;
    }
    while (fgets(line, sizeof(line), input) != NULL) {
        TrainingMetricRow row;
        bool complete_line = strchr(line, '\n') != NULL;
        bool is_train = false;
        bool is_validation = false;
        bool is_mean = false;
        bool is_final = false;

        if (!complete_line && feof(input) &&
            initial_stochastic && initial_mean) {
            size_t epoch;
            bool required_complete = true;
            for (epoch = 0; epoch <= latest_epoch; epoch++) {
                if (!train_seen[epoch] || !validation_seen[epoch]) {
                    required_complete = false;
                }
            }
            if (required_complete) {
                trailing = true;
                any_dropped = true;
                break;
            }
        }
        if (training_parse_metric(line, &row) != 0 ||
            !training_metric_numeric_valid(&row)) {
            goto cleanup;
        }
        if (row.epoch == -1) {
            if (row.optimizer_step != 0 ||
                row.sample_count != TRAINING_VALIDATION_COUNT ||
                (strcmp(row.phase,
                        "validation_stochastic_initial") != 0 &&
                 strcmp(row.phase, "validation_mean_initial") != 0)) {
                goto cleanup;
            }
        } else if (row.epoch >= 0 &&
                   row.epoch < TRAINING_MAX_EPOCHS) {
            uint64_t expected_step =
                ((uint64_t)row.epoch + UINT64_C(1)) *
                TRAINING_TRAIN_BATCHES;
            is_train = strcmp(
                row.phase, "train_stochastic_pre_update") == 0;
            is_validation = strcmp(
                row.phase, "validation_stochastic_post_epoch") == 0;
            is_mean = strcmp(
                row.phase, "validation_mean_diagnostic") == 0;
            is_final = strcmp(
                row.phase, "validation_mean_final") == 0;
            if (row.optimizer_step != expected_step ||
                (!is_train && !is_validation && !is_mean && !is_final) ||
                (is_train &&
                 row.sample_count != TRAINING_TRAIN_COUNT) ||
                (!is_train &&
                 row.sample_count != TRAINING_VALIDATION_COUNT)) {
                goto cleanup;
            }
        } else {
            goto cleanup;
        }
        if (row.epoch > (int64_t)latest_epoch) {
            trailing = true;
            any_dropped = true;
            continue;
        }
        if (trailing) {
            goto cleanup;
        }
        if (*final_row_present) {
            goto cleanup;
        }
        if (row.epoch == -1) {
            if (strcmp(row.phase, "validation_stochastic_initial") == 0 &&
                !initial_stochastic) {
                initial_stochastic = true;
            } else if (strcmp(row.phase, "validation_mean_initial") == 0 &&
                       initial_stochastic && !initial_mean) {
                initial_mean = true;
            } else {
                goto cleanup;
            }
        } else if (row.epoch >= 0 &&
                   row.epoch < TRAINING_MAX_EPOCHS) {
            size_t epoch = (size_t)row.epoch;
            if (!initial_stochastic || !initial_mean) {
                goto cleanup;
            }
            if (is_train) {
                if (train_seen[epoch] || validation_seen[epoch] ||
                    (epoch != 0 && !validation_seen[epoch - 1])) {
                    goto cleanup;
                }
                train_seen[epoch] = true;
            } else if (is_validation) {
                if (!train_seen[epoch] || validation_seen[epoch]) {
                    goto cleanup;
                }
                validation_seen[epoch] = true;
                validation_means[epoch] = row.total_mean_per_sample;
            } else if (is_mean) {
                if (!validation_seen[epoch] || mean_seen[epoch] ||
                    ((epoch + 1) % TRAINING_MEAN_EVAL_EVERY) != 0) {
                    goto cleanup;
                }
                mean_seen[epoch] = true;
            } else if (is_final) {
                if (epoch != latest_epoch || !validation_seen[epoch] ||
                    (((epoch + 1) % TRAINING_MEAN_EVAL_EVERY) == 0 &&
                     !mean_seen[epoch]) ||
                    *final_row_present) {
                    goto cleanup;
                }
                *final_row_present = true;
            } else {
                goto cleanup;
            }
        } else {
            goto cleanup;
        }
        if (fputs(line, output) == EOF) {
            goto cleanup;
        }
    }
    if (ferror(input) || !initial_stochastic || !initial_mean) {
        goto cleanup;
    }
    {
        size_t epoch;
        bool reconstructed_best_defined = false;
        uint64_t reconstructed_best_epoch = 0;
        uint64_t reconstructed_no_improvement = 0;
        double reconstructed_best = 0.0;
        bool terminal;

        for (epoch = 0; epoch <= latest_epoch; epoch++) {
            bool mean_required =
                ((epoch + 1) % TRAINING_MEAN_EVAL_EVERY) == 0;
            double current_validation = validation_means[epoch];

            if (!train_seen[epoch] || !validation_seen[epoch] ||
                mean_seen[epoch] != mean_required ||
                (epoch != 0 &&
                 reconstructed_no_improvement >= TRAINING_PATIENCE)) {
                goto cleanup;
            }
            if (!reconstructed_best_defined ||
                current_validation < reconstructed_best) {
                reconstructed_best_defined = true;
                reconstructed_best = current_validation;
                reconstructed_best_epoch = (uint64_t)epoch;
                reconstructed_no_improvement = 0;
            } else {
                reconstructed_no_improvement++;
            }
        }
        terminal = latest_epoch == TRAINING_MAX_EPOCHS - 1 ||
            reconstructed_no_improvement >= TRAINING_PATIENCE;
        if (run->controller.best_defined != reconstructed_best_defined ||
            !reconstructed_best_defined ||
            run->controller.best_epoch != reconstructed_best_epoch ||
            run->controller.best_validation_total_mean_per_sample !=
                reconstructed_best ||
            run->controller.no_improvement_count !=
                reconstructed_no_improvement ||
            (*final_row_present && !terminal)) {
            goto cleanup;
        }
    }
    if (fflush(output) != 0 || ferror(output) || fclose(output) != 0) {
        output = NULL;
        goto cleanup;
    }
    output = NULL;
    if (fclose(input) != 0) {
        input = NULL;
        goto cleanup;
    }
    input = NULL;
    if (any_dropped) {
        if (training_atomic_replace(temporary_path, metrics_path) != 0) {
            goto cleanup;
        }
        printf("METRICS_RECONCILED_TO_LATEST=true\n");
    } else if (remove(temporary_path) != 0) {
        goto cleanup;
    }
    run->metrics_file = fopen(metrics_path, "ab");
    if (run->metrics_file == NULL) {
        goto cleanup;
    }
    result = 0;

cleanup:
    if (input != NULL) {
        (void)fclose(input);
    }
    if (output != NULL) {
        (void)fclose(output);
    }
    if (result != 0) {
        (void)remove(temporary_path);
    }
    return result;
}

static int training_checkpoint_path(
    const TrainingRun *run,
    const char *name,
    char *path,
    size_t capacity
)
{
    return training_path_join(run->partial_output, name, path, capacity);
}

static int training_save_checkpoint(
    TrainingRun *run,
    const char *name,
    VAECCheckpointKind kind
)
{
    VAECCheckpointMetadata metadata = run->controller;
    char path[TRAINING_PATH_CAP];
    VAECCheckpointStatus status;

    metadata.kind = kind;
    if (training_checkpoint_path(run, name, path, sizeof(path)) != 0) {
        return 1;
    }
    status = vae_c_checkpoint_save_atomic(
        path, &run->model, &run->optimizer, &metadata
    );
    if (status != VAE_C_CHECKPOINT_OK) {
        fprintf(stderr, "Errore checkpoint %s: %s\n",
                name, vae_c_checkpoint_status_name(status));
        return 1;
    }
    return 0;
}

static int training_load_latest(TrainingRun *run)
{
    char path[TRAINING_PATH_CAP];
    VAECCheckpointStatus status;

    if (training_checkpoint_path(
            run, "latest.ckpt", path, sizeof(path)) != 0) {
        return 1;
    }
    status = vae_c_checkpoint_load(
        path, &run->identity, &run->model,
        &run->optimizer, &run->controller
    );
    if (status != VAE_C_CHECKPOINT_OK ||
        run->controller.kind != VAE_C_CHECKPOINT_KIND_LATEST ||
        run->controller.last_completed_epoch >= TRAINING_MAX_EPOCHS ||
        run->optimizer.step !=
            (run->controller.last_completed_epoch + UINT64_C(1)) *
            TRAINING_TRAIN_BATCHES ||
        run->controller.no_improvement_count > TRAINING_PATIENCE) {
        fprintf(stderr, "Errore latest checkpoint: %s\n",
                vae_c_checkpoint_status_name(status));
        return 1;
    }
    return 0;
}

static int training_validate_best_checkpoint(TrainingRun *run)
{
    char path[TRAINING_PATH_CAP];
    bool is_directory = false;
    int state;
    VAECModel candidate_model;
    VAECAdam candidate_optimizer;
    VAECCheckpointMetadata metadata;
    VAECCheckpointStatus status = VAE_C_CHECKPOINT_INVALID_STATE;
    int result = 1;

    if (training_checkpoint_path(
            run, "best.ckpt", path, sizeof(path)) != 0) {
        return 1;
    }
    state = training_path_state(path, &is_directory);
    if (!run->controller.best_defined) {
        return state == 0 ? 0 : 1;
    }
    if (state == 0 &&
        run->controller.best_epoch == run->controller.last_completed_epoch) {
        return training_save_checkpoint(
            run, "best.ckpt", VAE_C_CHECKPOINT_KIND_BEST
        );
    }
    if (state != 1 || is_directory) {
        return 1;
    }
    vae_c_model_init(&candidate_model);
    vae_c_adam_init(&candidate_optimizer);
    vae_c_checkpoint_metadata_init(&metadata);
    if (vae_c_model_allocate(
            &candidate_model, run->profile,
            TRAINING_BATCH_SIZE) != VAE_C_OP_OK ||
        vae_c_model_initialize_parameters_v1(
            &candidate_model, TRAINING_INIT_SEED) != VAE_C_OP_OK ||
        vae_c_adam_configure_v1(&candidate_optimizer) != VAE_C_OP_OK) {
        goto cleanup;
    }
    status = vae_c_checkpoint_load(
        path, &run->identity, &candidate_model,
        &candidate_optimizer, &metadata
    );
    if (status == VAE_C_CHECKPOINT_OK &&
        metadata.kind == VAE_C_CHECKPOINT_KIND_BEST &&
        metadata.best_defined &&
        metadata.last_completed_epoch == metadata.best_epoch &&
        metadata.best_epoch == run->controller.best_epoch &&
        metadata.best_validation_total_mean_per_sample ==
            run->controller.best_validation_total_mean_per_sample) {
        result = 0;
    }

cleanup:
    vae_c_adam_free(&candidate_optimizer);
    vae_c_model_free(&candidate_model);
    if (result != 0 &&
        run->controller.best_epoch == run->controller.last_completed_epoch) {
        if (training_save_checkpoint(
                run, "best.ckpt", VAE_C_CHECKPOINT_KIND_BEST) == 0) {
            printf("BEST_CHECKPOINT_REPAIRED_FROM_LATEST=true\n");
            result = 0;
        }
    }
    return result;
}

static unsigned char training_float_to_u8(float value)
{
    double scaled;
    if (!isfinite(value)) {
        return 0;
    }
    if (value < 0.0f) {
        value = 0.0f;
    } else if (value > 1.0f) {
        value = 1.0f;
    }
    scaled = floor((double)value * 255.0 + 0.5);
    return (unsigned char)scaled;
}

static float training_sigmoid(float value)
{
    if (value >= 0.0f) {
        double exponential = exp(-(double)value);
        return (float)(1.0 / (1.0 + exponential));
    }
    {
        double exponential = exp((double)value);
        return (float)(exponential / (1.0 + exponential));
    }
}

static int training_write_pgm_atomic(
    const char *path,
    const float *values,
    bool logits
)
{
    char temporary[TRAINING_PATH_CAP];
    unsigned char pixels[TRAINING_SIDE * TRAINING_SIDE];
    FILE *file;
    size_t index;
    int failed = 0;

    if (training_path_append(path, ".tmp",
                             temporary, sizeof(temporary)) != 0 ||
        training_path_state(temporary, NULL) != 0) {
        return 1;
    }
    for (index = 0; index < sizeof(pixels); index++) {
        float value = logits ? training_sigmoid(values[index]) : values[index];
        pixels[index] = training_float_to_u8(value);
    }
    file = fopen(temporary, "wbx");
    if (file == NULL) {
        return 1;
    }
    if (fprintf(file, "P5\n%d %d\n255\n",
                TRAINING_SIDE, TRAINING_SIDE) < 0 ||
        fwrite(pixels, 1, sizeof(pixels), file) != sizeof(pixels) ||
        fflush(file) != 0 || ferror(file)) {
        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    if (!failed && training_atomic_replace(temporary, path) != 0) {
        failed = 1;
    }
    if (failed) {
        (void)remove(temporary);
    }
    return failed;
}

static int training_write_preview_for_model(
    TrainingRun *run,
    VAECModel *model,
    TrainingSlot *preview,
    const char *phase,
    bool write_targets
)
{
    VAECVAELossStats stats;
    size_t sample;
    size_t channel;

    if (training_load_validation_slot(run, preview, 0, false) != 0) {
        return 1;
    }
    vae_c_vae_loss_stats_init(&stats);
    if (vae_c_model_vae_forward(
            model, VAE_C_BATCH_NORM_MODE_EVAL,
            VAE_C_LATENT_MODE_MEAN, &preview->workspace,
            &stats) != VAE_C_OP_OK ||
        !training_stats_are_valid(&stats, TRAINING_PREVIEW_EVENTS)) {
        return 1;
    }
    for (sample = 0; sample < TRAINING_PREVIEW_EVENTS; sample++) {
        uint64_t manifest_index = preview->manifest_indices[sample];
        for (channel = 0; channel < TRAINING_CHANNELS; channel++) {
            char name[256];
            char path[TRAINING_PATH_CAP];
            size_t offset =
                (sample * TRAINING_CHANNELS + channel) *
                TRAINING_SIDE * TRAINING_SIDE;
            int written = snprintf(
                name, sizeof(name),
                "reconstruction_mean_%s_manifest_%08" PRIu64 "_ch%lu.pgm",
                phase, manifest_index, (unsigned long)channel
            );
            if (written < 0 || (size_t)written >= sizeof(name) ||
                training_path_join(run->partial_output, name,
                                   path, sizeof(path)) != 0 ||
                training_write_pgm_atomic(
                    path, preview->workspace.logits.data + offset,
                    true) != 0) {
                return 1;
            }
            if (write_targets) {
                written = snprintf(
                    name, sizeof(name),
                    "target_initial_manifest_%08" PRIu64 "_ch%lu.pgm",
                    manifest_index, (unsigned long)channel
                );
                if (written < 0 || (size_t)written >= sizeof(name) ||
                    training_path_join(run->partial_output, name,
                                       path, sizeof(path)) != 0 ||
                    training_write_pgm_atomic(
                        path, preview->workspace.input.data + offset,
                        false) != 0) {
                    return 1;
                }
            }
        }
    }
    return 0;
}

static int training_write_preview(
    TrainingRun *run,
    const char *phase,
    bool write_targets
)
{
    return training_write_preview_for_model(
        run, &run->model, &run->preview, phase, write_targets
    );
}

static int training_preview_exists(
    const TrainingRun *run,
    const char *phase,
    bool targets
)
{
    size_t sample;
    size_t channel;

    for (sample = 0; sample < TRAINING_PREVIEW_EVENTS; sample++) {
        uint64_t manifest_index =
            (uint64_t)run->validation_split.indices[sample];
        for (channel = 0; channel < TRAINING_CHANNELS; channel++) {
            char name[256];
            char path[TRAINING_PATH_CAP];
            bool is_directory = false;
            int written = snprintf(
                name, sizeof(name),
                "reconstruction_mean_%s_manifest_%08" PRIu64 "_ch%lu.pgm",
                phase, manifest_index, (unsigned long)channel
            );
            if (written < 0 || (size_t)written >= sizeof(name) ||
                training_path_join(run->partial_output, name,
                                   path, sizeof(path)) != 0 ||
                training_path_state(path, &is_directory) != 1 ||
                is_directory) {
                return 0;
            }
            if (targets) {
                written = snprintf(
                    name, sizeof(name),
                    "target_initial_manifest_%08" PRIu64 "_ch%lu.pgm",
                    manifest_index, (unsigned long)channel
                );
                if (written < 0 || (size_t)written >= sizeof(name) ||
                    training_path_join(run->partial_output, name,
                                       path, sizeof(path)) != 0 ||
                    training_path_state(path, &is_directory) != 1 ||
                    is_directory) {
                    return 0;
                }
            }
        }
    }
    return 1;
}

static int training_refresh_best_preview(TrainingRun *run)
{
    char best_path[TRAINING_PATH_CAP];
    VAECModel best_model;
    VAECAdam best_optimizer;
    TrainingSlot best_preview;
    VAECCheckpointMetadata best_metadata;
    VAECCheckpointStatus status;
    int result = 1;

    if (!run->controller.best_defined) {
        return 0;
    }
    if (training_checkpoint_path(
            run, "best.ckpt", best_path, sizeof(best_path)) != 0) {
        return 1;
    }
    vae_c_model_init(&best_model);
    vae_c_adam_init(&best_optimizer);
    training_slot_init(&best_preview);
    vae_c_checkpoint_metadata_init(&best_metadata);
    if (vae_c_model_allocate(
            &best_model, run->profile,
            TRAINING_BATCH_SIZE) != VAE_C_OP_OK ||
        vae_c_model_initialize_parameters_v1(
            &best_model, TRAINING_INIT_SEED) != VAE_C_OP_OK ||
        vae_c_adam_configure_v1(&best_optimizer) != VAE_C_OP_OK) {
        goto cleanup;
    }
    status = vae_c_checkpoint_load(
        best_path, &run->identity, &best_model,
        &best_optimizer, &best_metadata
    );
    if (status != VAE_C_CHECKPOINT_OK ||
        best_metadata.kind != VAE_C_CHECKPOINT_KIND_BEST ||
        !best_metadata.best_defined ||
        best_metadata.last_completed_epoch != best_metadata.best_epoch ||
        best_metadata.best_epoch != run->controller.best_epoch ||
        best_metadata.best_validation_total_mean_per_sample !=
            run->controller.best_validation_total_mean_per_sample) {
        goto cleanup;
    }
    if (training_slot_allocate(
            &best_model, &run->tensor_manifest.spec,
            TRAINING_PREVIEW_EVENTS, &best_preview) != 0 ||
        training_write_preview_for_model(
            run, &best_model, &best_preview, "best", false) != 0) {
        goto cleanup;
    }
    result = 0;
    printf("BEST_PREVIEW_REFRESHED_FROM_CHECKPOINT=true\n");

cleanup:
    training_slot_free(&best_preview);
    vae_c_adam_free(&best_optimizer);
    vae_c_model_free(&best_model);
    return result;
}

static int training_dry_run(TrainingRun *run)
{
    float *epsilon_copy = NULL;
    size_t epsilon_bytes;
    int result = 1;

    if (training_allocate_runtime(run) != 0 ||
        training_prepare_epoch(run, UINT64_C(0)) != 0 ||
        training_load_train_slot(run, &run->train_full, 0, 0) != 0 ||
        training_load_train_slot(
            run, &run->train_tail,
            TRAINING_TRAIN_COUNT - TRAINING_TRAIN_TAIL, 0) != 0 ||
        training_load_validation_slot(
            run, &run->validation_full, 0, true) != 0 ||
        training_size_multiply(
            run->validation_full.workspace.epsilon.shape.count,
            sizeof(float), &epsilon_bytes) != 0) {
        goto cleanup;
    }
    epsilon_copy = (float *)malloc(epsilon_bytes);
    if (epsilon_copy == NULL) {
        goto cleanup;
    }
    memcpy(epsilon_copy,
           run->validation_full.workspace.epsilon.data,
           epsilon_bytes);
    if (training_fill_validation_epsilon(&run->validation_full) != 0 ||
        memcmp(epsilon_copy,
               run->validation_full.workspace.epsilon.data,
               epsilon_bytes) != 0 ||
        training_load_validation_slot(
            run, &run->validation_tail,
            TRAINING_VALIDATION_COUNT - TRAINING_VALIDATION_TAIL,
            true) != 0 ||
        run->optimizer.step != 0) {
        goto cleanup;
    }
    printf("DRY_RUN=true\nPROFILE=%s\nBATCH_SIZE=%d\n",
           TRAINING_PROFILE_ID, TRAINING_BATCH_SIZE);
    printf("FIRST_TRAIN_BATCH=%d\nLAST_TRAIN_BATCH=%d\n",
           TRAINING_BATCH_SIZE, TRAINING_TRAIN_TAIL);
    printf("FIRST_VALIDATION_BATCH=%d\nLAST_VALIDATION_BATCH=%d\n",
           TRAINING_BATCH_SIZE, TRAINING_VALIDATION_TAIL);
    printf("STATELESS_AUGMENTATION_VERIFIED=true\n");
    printf("VALIDATION_EPSILON_FIXED=true\nOPTIMIZER_STEP=0\n");
    printf("CHECKPOINTS_CREATED=0\nOUTPUT_CREATED=false\n");
    printf("MINI112 TRAINING DRY RUN PASSED\n");
    result = 0;

cleanup:
    free(epsilon_copy);
    return result;
}

static int training_smoke(TrainingRun *run)
{
    VAECVAELossStats first;
    VAECVAELossStats tail;
    VAECVAELossStats validation;
    VAECVAELossStats validation_tail;

    if (training_allocate_runtime(run) != 0 ||
        training_prepare_epoch(run, UINT64_C(0)) != 0 ||
        training_load_train_slot(run, &run->train_full, 0, 0) != 0) {
        return 1;
    }
    vae_c_vae_loss_stats_init(&first);
    if (vae_c_model_train_step(
            &run->model, &run->train_full.workspace,
            &run->optimizer, TRAINING_LATENT_SEED, 0,
            run->train_full.manifest_indices,
            TRAINING_BATCH_SIZE, &first) != VAE_C_OP_OK ||
        !training_stats_are_valid(&first, TRAINING_BATCH_SIZE) ||
        training_load_train_slot(
            run, &run->train_tail,
            TRAINING_TRAIN_COUNT - TRAINING_TRAIN_TAIL, 0) != 0) {
        return 1;
    }
    vae_c_vae_loss_stats_init(&tail);
    if (vae_c_model_train_step(
            &run->model, &run->train_tail.workspace,
            &run->optimizer, TRAINING_LATENT_SEED, 0,
            run->train_tail.manifest_indices,
            TRAINING_TRAIN_TAIL, &tail) != VAE_C_OP_OK ||
        !training_stats_are_valid(&tail, TRAINING_TRAIN_TAIL) ||
        training_load_validation_slot(
            run, &run->validation_full, 0, true) != 0) {
        return 1;
    }
    vae_c_vae_loss_stats_init(&validation);
    if (vae_c_model_vae_forward(
            &run->model, VAE_C_BATCH_NORM_MODE_EVAL,
            VAE_C_LATENT_MODE_STOCHASTIC,
            &run->validation_full.workspace,
            &validation) != VAE_C_OP_OK ||
        !training_stats_are_valid(&validation, TRAINING_BATCH_SIZE) ||
        training_load_validation_slot(
            run, &run->validation_tail,
            TRAINING_VALIDATION_COUNT - TRAINING_VALIDATION_TAIL,
            true) != 0) {
        return 1;
    }
    vae_c_vae_loss_stats_init(&validation_tail);
    if (vae_c_model_vae_forward(
            &run->model, VAE_C_BATCH_NORM_MODE_EVAL,
            VAE_C_LATENT_MODE_STOCHASTIC,
            &run->validation_tail.workspace,
            &validation_tail) != VAE_C_OP_OK ||
        !training_stats_are_valid(
            &validation_tail, TRAINING_VALIDATION_TAIL) ||
        run->optimizer.step != 2) {
        return 1;
    }
    printf("SMOKE=true\n");
    printf("TRAIN_FIRST_TOTAL_MEAN=%.17g\n", first.mean_per_sample);
    printf("TRAIN_TAIL_TOTAL_MEAN=%.17g\n", tail.mean_per_sample);
    printf("VALIDATION_TOTAL_MEAN=%.17g\n", validation.mean_per_sample);
    printf("VALIDATION_TAIL_TOTAL_MEAN=%.17g\n",
           validation_tail.mean_per_sample);
    printf("OPTIMIZER_STEP=2\nALL_LOSSES_FINITE=true\n");
    printf("CHECKPOINTS_CREATED=0\nOUTPUT_CREATED=false\n");
    printf("MINI112 TRAINING SMOKE PASSED\n");
    return 0;
}

static int training_finalize_directory(TrainingRun *run)
{
    bool ignored;

    if (run->metrics_file != NULL) {
        if (fflush(run->metrics_file) != 0 ||
            ferror(run->metrics_file) ||
            fclose(run->metrics_file) != 0) {
            run->metrics_file = NULL;
            return 1;
        }
        run->metrics_file = NULL;
    }
    if (training_path_state(run->final_output, &ignored) != 0 ||
        rename(run->partial_output, run->final_output) != 0) {
        return 1;
    }
    run->partial_is_owned_run = false;
    return 0;
}

static int training_full_run(TrainingRun *run, bool resume)
{
    uint64_t next_epoch = 0;
    bool final_row_present = false;

    if (resume) {
        if (training_open_resume_output(run) != 0 ||
            training_load_latest(run) != 0 ||
            training_validate_best_checkpoint(run) != 0 ||
            training_allocate_runtime(run) != 0 ||
            !training_preview_exists(run, "initial", true) ||
            training_refresh_best_preview(run) != 0 ||
            training_reconcile_metrics(
                run, run->controller.last_completed_epoch,
                &final_row_present) != 0) {
            return 1;
        }
        next_epoch = run->controller.last_completed_epoch + UINT64_C(1);
        printf("RESUME=true\nRESUME_NEXT_EPOCH=%" PRIu64 "\n", next_epoch);
    } else {
        TrainingAggregate initial_stochastic;
        TrainingAggregate initial_mean;
        TrainingMetricRow row;

        if (training_allocate_runtime(run) != 0 ||
            training_evaluate(run, VAE_C_LATENT_MODE_STOCHASTIC,
                              &initial_stochastic) != 0 ||
            training_evaluate(run, VAE_C_LATENT_MODE_MEAN,
                              &initial_mean) != 0 ||
            training_prepare_fresh_output(run) != 0 ||
            training_metric_from_aggregate(
                "validation_stochastic_initial", -1, 0,
                &initial_stochastic, &row) != 0 ||
            training_write_metric(run->metrics_file, &row) != 0 ||
            training_metric_from_aggregate(
                "validation_mean_initial", -1, 0,
                &initial_mean, &row) != 0 ||
            training_write_metric(run->metrics_file, &row) != 0 ||
            training_write_preview(run, "initial", true) != 0) {
            return 1;
        }
        run->controller.run_identity = run->identity;
        printf("RESUME=false\n");
        printf("INITIAL_VALIDATION_STOCHASTIC_TOTAL_MEAN=%.17g\n",
               initial_stochastic.total_sum /
               (double)initial_stochastic.sample_count);
        printf("INITIAL_VALIDATION_MEAN_TOTAL_MEAN=%.17g\n",
               initial_mean.total_sum / (double)initial_mean.sample_count);
    }
    while (next_epoch < TRAINING_MAX_EPOCHS &&
           run->controller.no_improvement_count < TRAINING_PATIENCE) {
        TrainingAggregate train;
        TrainingAggregate validation;
        TrainingAggregate mean;
        TrainingMetricRow train_row;
        TrainingMetricRow validation_row;
        TrainingMetricRow mean_row;
        bool mean_defined = false;
        bool improved;
        bool stop;
        uint64_t expected_step =
            (next_epoch + UINT64_C(1)) * TRAINING_TRAIN_BATCHES;
        double validation_mean;

        if (training_run_epoch(run, next_epoch, &train) != 0 ||
            run->optimizer.step != expected_step ||
            training_evaluate(
                run, VAE_C_LATENT_MODE_STOCHASTIC, &validation) != 0) {
            return 1;
        }
        validation_mean =
            validation.total_sum / (double)validation.sample_count;
        improved = !run->controller.best_defined ||
            validation_mean <
                run->controller.best_validation_total_mean_per_sample;
        if (improved) {
            run->controller.best_defined = true;
            run->controller.best_epoch = next_epoch;
            run->controller.best_validation_total_mean_per_sample =
                validation_mean;
            run->controller.no_improvement_count = 0;
        } else {
            run->controller.no_improvement_count++;
        }
        run->controller.run_identity = run->identity;
        run->controller.last_completed_epoch = next_epoch;
        if (((next_epoch + UINT64_C(1)) %
             TRAINING_MEAN_EVAL_EVERY) == 0) {
            if (training_evaluate(
                    run, VAE_C_LATENT_MODE_MEAN, &mean) != 0) {
                return 1;
            }
            mean_defined = true;
        }
        if (training_metric_from_aggregate(
                "train_stochastic_pre_update", (int64_t)next_epoch,
                run->optimizer.step, &train, &train_row) != 0 ||
            training_metric_from_aggregate(
                "validation_stochastic_post_epoch", (int64_t)next_epoch,
                run->optimizer.step, &validation, &validation_row) != 0 ||
            training_write_metric(run->metrics_file, &train_row) != 0 ||
            training_write_metric(run->metrics_file, &validation_row) != 0) {
            return 1;
        }
        if (mean_defined &&
            (training_metric_from_aggregate(
                 "validation_mean_diagnostic", (int64_t)next_epoch,
                 run->optimizer.step, &mean, &mean_row) != 0 ||
             training_write_metric(run->metrics_file, &mean_row) != 0)) {
            return 1;
        }
        if (training_save_checkpoint(
                run, "latest.ckpt",
                VAE_C_CHECKPOINT_KIND_LATEST) != 0) {
            return 1;
        }
        if (improved &&
            (training_save_checkpoint(
                 run, "best.ckpt", VAE_C_CHECKPOINT_KIND_BEST) != 0 ||
             training_write_preview(run, "best", false) != 0)) {
            return 1;
        }
        printf("EPOCH=%" PRIu64
               "|TRAIN_TOTAL_MEAN=%.17g|VALIDATION_TOTAL_MEAN=%.17g|"
               "BEST_EPOCH=%" PRIu64 "|NO_IMPROVEMENT=%" PRIu64 "\n",
               next_epoch,
               train.total_sum / (double)train.sample_count,
               validation_mean, run->controller.best_epoch,
               run->controller.no_improvement_count);
        stop = run->controller.no_improvement_count >= TRAINING_PATIENCE ||
            next_epoch + UINT64_C(1) >= TRAINING_MAX_EPOCHS;
        if (stop) {
            TrainingAggregate final_mean;
            TrainingMetricRow final_row;
            if (training_evaluate(
                    run, VAE_C_LATENT_MODE_MEAN, &final_mean) != 0 ||
                training_metric_from_aggregate(
                    "validation_mean_final", (int64_t)next_epoch,
                    run->optimizer.step, &final_mean, &final_row) != 0 ||
                training_write_metric(run->metrics_file, &final_row) != 0 ||
                training_write_preview(run, "final", false) != 0 ||
                training_save_checkpoint(
                    run, "final.ckpt",
                    VAE_C_CHECKPOINT_KIND_FINAL) != 0 ||
                training_finalize_directory(run) != 0) {
                return 1;
            }
            printf("TRAINING_COMPLETED=true\n");
            printf("LAST_COMPLETED_EPOCH=%" PRIu64 "\n", next_epoch);
            printf("BEST_EPOCH=%" PRIu64 "\n",
                   run->controller.best_epoch);
            printf("BEST_VALIDATION_TOTAL_MEAN=%.17g\n",
                   run->controller.best_validation_total_mean_per_sample);
            printf("OUTPUT_DIRECTORY=%s\n", run->final_output);
            return 0;
        }
        next_epoch++;
    }
    if (next_epoch > 0 &&
        (next_epoch >= TRAINING_MAX_EPOCHS ||
         run->controller.no_improvement_count >= TRAINING_PATIENCE)) {
        TrainingAggregate final_mean;
        TrainingMetricRow final_row;
        uint64_t final_epoch = run->controller.last_completed_epoch;
        if (training_evaluate(
                run, VAE_C_LATENT_MODE_MEAN, &final_mean) != 0 ||
            (!final_row_present &&
             (training_metric_from_aggregate(
                  "validation_mean_final", (int64_t)final_epoch,
                  run->optimizer.step, &final_mean, &final_row) != 0 ||
              training_write_metric(run->metrics_file, &final_row) != 0)) ||
            training_write_preview(run, "final", false) != 0 ||
            training_save_checkpoint(
                run, "final.ckpt", VAE_C_CHECKPOINT_KIND_FINAL) != 0 ||
            training_finalize_directory(run) != 0) {
            return 1;
        }
        printf("TRAINING_COMPLETED_FROM_RESUME_BOUNDARY=true\n");
        printf("OUTPUT_DIRECTORY=%s\n", run->final_output);
        return 0;
    }
    return 1;
}

static void training_print_usage(const char *program)
{
    printf(
        "Uso: %s [--dry-run|--smoke|--resume] "
        "--code-commit HASH [repository_root]\n\n"
        "--dry-run valida hash, contratto, shuffle, batch 64/18/64/53 "
        "ed epsilon senza train step o output.\n"
        "--smoke esegue in memoria due train step e validation full/tail, "
        "senza checkpoint o output.\n"
        "--resume riprende esclusivamente dalla directory .partial e dal "
        "latest.ckpt compatibile.\n"
        "Senza queste opzioni esegue il training ufficiale fino a 100 epoch; "
        "questa modalita' non va usata durante la verifica pre-commit.\n",
        program
    );
}

int main(int argc, char **argv)
{
    const char *repository_root = ".";
    const char *code_commit = NULL;
    bool dry_run = false;
    bool smoke = false;
    bool resume = false;
    bool repository_seen = false;
    TrainingRun run;
    int argument;
    int mode_count;
    int result = 1;

    for (argument = 1; argument < argc; argument++) {
        if (strcmp(argv[argument], "--help") == 0 ||
            strcmp(argv[argument], "-h") == 0) {
            training_print_usage(argv[0]);
            return 0;
        }
    }
    for (argument = 1; argument < argc; argument++) {
        if (strcmp(argv[argument], "--dry-run") == 0) {
            if (dry_run) {
                return 2;
            }
            dry_run = true;
        } else if (strcmp(argv[argument], "--smoke") == 0) {
            if (smoke) {
                return 2;
            }
            smoke = true;
        } else if (strcmp(argv[argument], "--resume") == 0) {
            if (resume) {
                return 2;
            }
            resume = true;
        } else if (strcmp(argv[argument], "--code-commit") == 0) {
            if (code_commit != NULL || argument + 1 >= argc) {
                return 2;
            }
            code_commit = argv[++argument];
        } else if (argv[argument][0] == '-' || repository_seen) {
            training_print_usage(argv[0]);
            return 2;
        } else {
            repository_root = argv[argument];
            repository_seen = true;
        }
    }
    mode_count = (dry_run ? 1 : 0) + (smoke ? 1 : 0) + (resume ? 1 : 0);
    if (mode_count > 1 || !training_code_commit_is_valid(code_commit)) {
        fprintf(stderr, "Errore opzioni o --code-commit non valido.\n");
        training_print_usage(argv[0]);
        return 2;
    }

    training_run_init(&run);
    if (training_load_contract(
            &run, repository_root, code_commit) != 0 ||
        training_build_output_paths(&run) != 0) {
        goto cleanup;
    }
    if (dry_run) {
        result = training_dry_run(&run);
    } else if (smoke) {
        result = training_smoke(&run);
    } else {
        bool final_is_directory = false;
        bool partial_is_directory = false;
        int final_state = training_path_state(
            run.final_output, &final_is_directory
        );
        int partial_state = training_path_state(
            run.partial_output, &partial_is_directory
        );

        if (final_state != 0 ||
            (partial_state == 1 && !partial_is_directory) ||
            (resume && partial_state != 1) ||
            (!resume && partial_state != 0)) {
            fprintf(stderr, "Errore stato output final/partial.\n");
            goto cleanup;
        }
        result = training_full_run(&run, resume);
    }

cleanup:
    training_run_free(&run);
    return result;
}
