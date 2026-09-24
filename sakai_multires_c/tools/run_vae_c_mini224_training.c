#include "vae_augmentation.h"
#include "vae_c_adam.h"
#include "vae_c_checkpoint.h"
#include "vae_c_config.h"
#include "vae_c_identity.h"
#include "vae_c_model_forward.h"
#include "vae_c_model_train.h"
#include "vae_c_rng.h"
#include "vae_c_sha256.h"
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
    TRAINING_CHANNELS = 4,
    TRAINING_SIDE = 224,
    TRAINING_INPUT_VALUES = 4 * 224 * 224,
    TRAINING_LATENT_VALUES = 64,
    TRAINING_TRAIN_COUNT = 800,
    TRAINING_VALIDATION_COUNT = 200,
    TRAINING_MEAN_EVAL_EVERY = 5,
    TRAINING_PREVIEW_EVENTS = 2
};

#define TRAINING_PROFILE_ID "mini224_bridge_v1"
#define TRAINING_DATASET_VERSION "legacy_luma_v1"
#define TRAINING_INIT_SEED UINT64_C(123)
#define TRAINING_LATENT_SEED UINT64_C(456)
#define TRAINING_AUGMENTATION_SEED UINT64_C(789)
#define TRAINING_SHUFFLE_SEED UINT64_C(101112)

#define TRAINING_EVENT_SHA \
    "620960e6b35da31bc4cf1a8a6da8679027d378e1d4b2ac73c5267c1c5311ae82"
#define TRAINING_TENSOR_SHA \
    "94b7e2e051bd32de72dd2cb6a5fb7eda9bcaec123cbf3a07d0d901bd19779afa"
#define TRAINING_TRAIN_SPLIT_SHA \
    "8c7a513b98dbe996261bc80965500ab15ab77f498ec9440858003152e42f926f"
#define TRAINING_VALIDATION_SPLIT_SHA \
    "e63109c88df7b3ee97317f94791f26df3fcaf8b937921cfb27967d01c2546467"
#define TRAINING_TRAIN_SUBSET_SHA \
    "80c5396a70e7f2c8e3a00680582945581ae76c301a5b9ba8c62e006bb25cdfb5"
#define TRAINING_VALIDATION_SUBSET_SHA \
    "9f5fe6c50c7398588f2de7bd4a543a54a53e7f7a0b9a84bf4004a01f065ae76f"
#define TRAINING_PROFILE_SHA \
    "7ac5e1b5f5a76265144e9a09ce8e1e1ec976ace76f664f86b432a20a45a19ad1"
#define TRAINING_DATASET_ID \
    "389fc0326101afcae22350445114e9b42caf64b059d9d6d923f25d83ce3377b0"
#define TRAINING_TRAIN_SELECTION_ID \
    "b9f311f6a655afed553a7e64affb850a92be58034449550b25e4f60685e5b0a5"
#define TRAINING_VALIDATION_SELECTION_ID \
    "cf1a6c62ca6acf45e466d6128b52018a33fbf4052aeea46f60533017abfeb831"

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
    GSSavedSplit official_train_split;
    GSSavedSplit official_validation_split;
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
    size_t batch_size;
    size_t train_last_batch;
    size_t validation_last_batch;
    uint64_t train_batches;
    uint64_t validation_batches;
    uint64_t max_epochs;
    uint64_t patience;
    char repository_root[TRAINING_PATH_CAP];
    char output_basename[256];
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

static int training_file_sha256(const char *path, char hex[65])
{
    FILE *file = NULL;
    unsigned char *data = NULL;
    long length;
    size_t size;
    int result = 1;

    if (path == NULL || hex == NULL) {
        return 1;
    }
    file = fopen(path, "rb");
    if (file == NULL ||
        fseek(file, 0, SEEK_END) != 0 ||
        (length = ftell(file)) < 0 ||
        fseek(file, 0, SEEK_SET) != 0) {
        if (file != NULL) {
            (void)fclose(file);
        }
        return 1;
    }
    size = (size_t)length;
    if ((long)size != length) {
        (void)fclose(file);
        return 1;
    }
    if (size != 0u) {
        data = (unsigned char *)malloc(size);
        if (data == NULL ||
            fread(data, 1u, size, file) != size ||
            fgetc(file) != EOF ||
            ferror(file)) {
            free(data);
            (void)fclose(file);
            return 1;
        }
    }
    if (fclose(file) != 0) {
        free(data);
        return 1;
    }
    if (vae_c_sha256_digest_hex_lower(data, size, hex) == VAE_C_SHA256_OK) {
        result = 0;
    }
    free(data);
    return result;
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
    gs_saved_split_free(&run->official_validation_split);
    gs_saved_split_free(&run->official_train_split);
    gs_event_manifest_free(&run->events);
    vae_c_profile_catalog_free(&run->catalog);
    memset(run, 0, sizeof(*run));
}

static int training_profile_is_expected(const VAECProfile *profile)
{
    return profile != NULL &&
        strcmp(profile->profile_id, TRAINING_PROFILE_ID) == 0 &&
        strcmp(profile->role, "project_bridge_candidate") == 0 &&
        profile->input_side == TRAINING_SIDE &&
        profile->conv_depth == 4 &&
        profile->channels[0] == 16 &&
        profile->channels[1] == 32 &&
        profile->channels[2] == 64 &&
        profile->channels[3] == 128 &&
        profile->latent_dim == TRAINING_LATENT_VALUES &&
        profile->bottleneck_side == 7 &&
        profile->expected_trainable_parameters_bn1d == UINT64_C(625860) &&
        profile->expected_forward_mac_per_sample == UINT64_C(220790784) &&
        strcmp(profile->profile_status, "candidate_pending_224_smoke") == 0;
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

static int training_hash_matches(
    const char *path,
    const char *expected
)
{
    char actual[VAE_C_SHA256_HEX_SIZE];

    return path != NULL && expected != NULL &&
        training_file_sha256(path, actual) == 0 &&
        strcmp(actual, expected) == 0;
}

static int training_build_identity(
    TrainingRun *run,
    const char *code_commit
)
{
    VAECCheckpointRunIdentity *identity = &run->identity;
    char validation_policy[64];
    int validation_policy_length;

    validation_policy_length = snprintf(
        validation_policy, sizeof(validation_policy),
        "m224_controller_v1+m=%016" PRIx64 "+p=%016" PRIx64,
        run->max_epochs, run->patience
    );
    if (validation_policy_length < 0 ||
        (size_t)validation_policy_length >= sizeof(validation_policy)) {
        return 1;
    }

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
                          TRAINING_DATASET_ID) != 0 ||
        training_set_text(identity->event_manifest_id,
                          sizeof(identity->event_manifest_id),
                          "manifest_events_272.csv") != 0 ||
        training_set_text(identity->event_manifest_sha256,
                          sizeof(identity->event_manifest_sha256),
                          TRAINING_EVENT_SHA) != 0 ||
        training_set_text(identity->tensor_manifest_id,
                          sizeof(identity->tensor_manifest_id),
                          "manifest_tensors_224_central.csv") != 0 ||
        training_set_text(identity->tensor_manifest_sha256,
                          sizeof(identity->tensor_manifest_sha256),
                          TRAINING_TENSOR_SHA) != 0 ||
        training_set_text(identity->train_split_id,
                          sizeof(identity->train_split_id),
                          TRAINING_TRAIN_SELECTION_ID) != 0 ||
        training_set_text(identity->train_split_sha256,
                          sizeof(identity->train_split_sha256),
                          TRAINING_TRAIN_SPLIT_SHA) != 0 ||
        training_set_text(identity->validation_split_id,
                          sizeof(identity->validation_split_id),
                          TRAINING_VALIDATION_SELECTION_ID) != 0 ||
        training_set_text(identity->validation_split_sha256,
                          sizeof(identity->validation_split_sha256),
                          TRAINING_VALIDATION_SPLIT_SHA) != 0 ||
        training_set_text(identity->augmentation_policy,
                          sizeof(identity->augmentation_policy),
                          "stateless_event_epoch_v1+symmetric_figure+side224") != 0 ||
        training_set_text(identity->shuffle_policy,
                          sizeof(identity->shuffle_policy),
                          "fisher_yates_unbiased_v1") != 0 ||
        training_set_text(identity->latent_policy,
                          sizeof(identity->latent_policy),
                          "train_event_epoch_v1+validation_event_v1+latent64") != 0 ||
        training_set_text(identity->validation_policy,
                          sizeof(identity->validation_policy),
                          validation_policy) != 0 ||
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
    identity->max_batch_size = (uint64_t)run->batch_size;
    identity->batch_size = (uint64_t)run->batch_size;
    identity->optimizer_learning_rate = VAE_C_ADAM_LEARNING_RATE_V1;
    identity->optimizer_beta1 = VAE_C_ADAM_BETA1_V1;
    identity->optimizer_beta2 = VAE_C_ADAM_BETA2_V1;
    identity->optimizer_epsilon = VAE_C_ADAM_EPSILON_V1;
    return 0;
}

static int training_subset_is_ordered_prefix(
    const GSSavedSplit *subset,
    const GSSavedSplit *official_split,
    size_t expected_count
)
{
    size_t index;

    if (subset == NULL || official_split == NULL ||
        subset->count != expected_count ||
        official_split->count < subset->count) {
        return 0;
    }
    for (index = 0; index < subset->count; index++) {
        if (subset->indices[index] != official_split->indices[index]) {
            return 0;
        }
    }
    return 1;
}

static int training_subsets_are_disjoint(
    const GSSavedSplit *train,
    const GSSavedSplit *validation
)
{
    size_t train_index;
    size_t validation_index;

    if (train == NULL || validation == NULL) {
        return 0;
    }
    for (train_index = 0; train_index < train->count; train_index++) {
        for (validation_index = 0;
             validation_index < validation->count;
             validation_index++) {
            if (train->indices[train_index] ==
                validation->indices[validation_index]) {
                return 0;
            }
        }
    }
    return 1;
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
    char train_subset_path[TRAINING_PATH_CAP];
    char validation_subset_path[TRAINING_PATH_CAP];
    char tensor_path[TRAINING_PATH_CAP];
    char dataset_identity[VAE_C_SHA256_HEX_SIZE];
    char train_identity[VAE_C_SHA256_HEX_SIZE];
    char validation_identity[VAE_C_SHA256_HEX_SIZE];
    char error_message[512];
    GSTensorSpec expected_spec;
    size_t index;

    if (run == NULL || repository_root == NULL || code_commit == NULL ||
        strlen(repository_root) >= sizeof(run->repository_root) ||
        training_path_join(repository_root,
            "sakai_multires_c/configs/vae_c_profiles_v2.csv",
            profile_path, sizeof(profile_path)) != 0 ||
        training_path_join(repository_root, "manifest_events_272.csv",
            event_path, sizeof(event_path)) != 0 ||
        training_path_join(repository_root, "split_train_indices_seed123.csv",
            train_path, sizeof(train_path)) != 0 ||
        training_path_join(repository_root, "split_val_indices_seed123.csv",
            validation_path, sizeof(validation_path)) != 0 ||
        training_path_join(repository_root,
            "sakai_multires_c/configs/subset224_train_800_seed123.csv",
            train_subset_path, sizeof(train_subset_path)) != 0 ||
        training_path_join(repository_root,
            "sakai_multires_c/configs/subset224_val_200_seed123.csv",
            validation_subset_path, sizeof(validation_subset_path)) != 0 ||
        training_path_join(repository_root, "manifest_tensors_224_central.csv",
            tensor_path, sizeof(tensor_path)) != 0) {
        return 1;
    }
    memcpy(run->repository_root, repository_root, strlen(repository_root) + 1);

    if (training_verify_hash(event_path, TRAINING_EVENT_SHA,
                             "manifest eventi") != 0 ||
        training_verify_hash(tensor_path, TRAINING_TENSOR_SHA,
                             "manifest tensor 224") != 0 ||
        training_verify_hash(train_path, TRAINING_TRAIN_SPLIT_SHA,
                             "split train ufficiale") != 0 ||
        training_verify_hash(validation_path, TRAINING_VALIDATION_SPLIT_SHA,
                             "split validation ufficiale") != 0 ||
        training_verify_hash(train_subset_path, TRAINING_TRAIN_SUBSET_SHA,
                             "subset train800") != 0 ||
        training_verify_hash(validation_subset_path,
                             TRAINING_VALIDATION_SUBSET_SHA,
                             "subset validation200") != 0 ||
        training_verify_hash(profile_path, TRAINING_PROFILE_SHA,
                             "catalogo profili V2") != 0) {
        return 1;
    }

    if (vae_c_identity_build_mini224_dataset_v1(
            TRAINING_DATASET_VERSION,
            TRAINING_EVENT_SHA,
            TRAINING_TENSOR_SHA,
            dataset_identity) != VAE_C_IDENTITY_OK ||
        vae_c_identity_build_mini224_train_selection_v1(
            dataset_identity,
            TRAINING_TRAIN_SPLIT_SHA,
            TRAINING_TRAIN_SUBSET_SHA,
            TRAINING_TRAIN_COUNT,
            train_identity) != VAE_C_IDENTITY_OK ||
        vae_c_identity_build_mini224_validation_selection_v1(
            dataset_identity,
            TRAINING_VALIDATION_SPLIT_SHA,
            TRAINING_VALIDATION_SUBSET_SHA,
            TRAINING_VALIDATION_COUNT,
            validation_identity) != VAE_C_IDENTITY_OK ||
        strcmp(dataset_identity, TRAINING_DATASET_ID) != 0 ||
        strcmp(train_identity, TRAINING_TRAIN_SELECTION_ID) != 0 ||
        strcmp(validation_identity,
               TRAINING_VALIDATION_SELECTION_ID) != 0 ||
        training_build_identity(run, code_commit) != 0) {
        fprintf(stderr, "Errore: identita' canonica mini224 non conforme.\n");
        return 1;
    }

    error_message[0] = '\0';
    if (vae_c_profile_catalog_load_csv(
            profile_path, &run->catalog,
            error_message, sizeof(error_message)) != VAE_C_CONFIG_OK) {
        fprintf(stderr, "Errore catalogo profili V2: %s\n", error_message);
        return 1;
    }
    run->profile = vae_c_profile_catalog_find(
        &run->catalog, TRAINING_PROFILE_ID
    );
    if (!training_profile_is_expected(run->profile) ||
        gs_event_manifest_load(event_path, &run->events) != 0 ||
        run->events.count != 8583u ||
        gs_saved_split_load(train_path, "train", &run->events,
                            &run->official_train_split) != 0 ||
        run->official_train_split.count != 6866u ||
        gs_saved_split_load(validation_path, "validation", &run->events,
                            &run->official_validation_split) != 0 ||
        run->official_validation_split.count != 1717u ||
        gs_saved_split_load(train_subset_path, "train", &run->events,
                            &run->train_split) != 0 ||
        gs_saved_split_load(validation_subset_path, "validation",
                            &run->events, &run->validation_split) != 0 ||
        !training_subset_is_ordered_prefix(
            &run->train_split, &run->official_train_split,
            TRAINING_TRAIN_COUNT) ||
        !training_subset_is_ordered_prefix(
            &run->validation_split, &run->official_validation_split,
            TRAINING_VALIDATION_COUNT) ||
        !training_subsets_are_disjoint(
            &run->train_split, &run->validation_split) ||
        gs_tensor_spec_init(&expected_spec, TRAINING_SIDE) != 0 ||
        gs_tensor_manifest_load(tensor_path, &expected_spec,
                                &run->tensor_manifest) != 0 ||
        run->tensor_manifest.count != run->events.count ||
        strcmp(run->tensor_manifest.extra_column_name, "crop_left") != 0) {
        fprintf(stderr, "Errore: contratto dataset/profilo non conforme.\n");
        return 1;
    }
    for (index = 0; index < run->tensor_manifest.count; index++) {
        const GSTensorManifestRecord *record =
            &run->tensor_manifest.records[index];
        const GSEventRecord *event = &run->events.records[index];

        if (record->event_index != event->event_index ||
            strcmp(record->event_id, event->event_id) != 0 ||
            record->channels != TRAINING_CHANNELS ||
            record->height != TRAINING_SIDE ||
            record->width != TRAINING_SIDE ||
            strcmp(record->dtype, "uint8") != 0 ||
            strcmp(record->layout, "channel_major") != 0 ||
            strcmp(record->channel_order, "0.5;1.0;2.0;4.0") != 0 ||
            strcmp(record->extra_value, "24") != 0) {
            fprintf(stderr, "Errore record tensor 224 %lu.\n",
                    (unsigned long)index);
            return 1;
        }
    }
    for (index = 0; index < run->validation_split.count; index++) {
        size_t manifest_index = run->validation_split.indices[index];
        const GSTensorManifestRecord *tensor_record =
            &run->tensor_manifest.records[manifest_index];
        const GSEventRecord *event = &run->events.records[manifest_index];

        if (strcmp(tensor_record->event_id, event->event_id) != 0 ||
            tensor_record->event_index != event->event_index) {
            fprintf(stderr, "Errore mapping validation centrale.\n");
            return 1;
        }
    }
    return 0;
}

static int training_allocate_model(TrainingRun *run)
{
    VAECOpStatus status;

    status = vae_c_model_allocate(
        &run->model, run->profile, run->batch_size
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
            run->batch_size, &run->train_full) != 0 ||
        training_slot_allocate(&run->model, &run->tensor_manifest.spec,
            run->train_last_batch, &run->train_tail) != 0 ||
        training_slot_allocate(&run->model, &run->tensor_manifest.spec,
            run->batch_size, &run->validation_full) != 0 ||
        training_slot_allocate(&run->model, &run->tensor_manifest.spec,
            run->validation_last_batch, &run->validation_tail) != 0 ||
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
        TrainingSlot *slot = remaining >= run->batch_size
            ? &run->train_full : &run->train_tail;
        VAECVAELossStats stats;

        if (remaining < run->batch_size &&
            remaining != run->train_last_batch) {
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
        run->optimizer.step == step_before + run->train_batches
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
        TrainingSlot *slot = remaining >= run->batch_size
            ? &run->validation_full : &run->validation_tail;
        VAECVAELossStats stats;

        if (remaining < run->batch_size &&
            remaining != run->validation_last_batch) {
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
        training_path_join(training, run->output_basename,
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
        "format=vae_c_mini224_training_v1\n"
        "run_id=%s\n"
        "code_commit=%s\n"
        "profile=%s\n"
        "profile_catalog=sakai_multires_c/configs/vae_c_profiles_v2.csv\n"
        "profile_catalog_version=v2\n"
        "profile_catalog_sha256=%s\n"
        "profile_role=project_bridge_candidate\n"
        "profile_status=candidate_pending_224_smoke\n"
        "input_channels=4\ninput_side=224\nconv_depth=4\n"
        "channels=16,32,64,128\nlatent_dim=64\nbottleneck_side=7\n"
        "trainable_parameters=625860\n"
        "forward_mac_per_sample=220790784\n"
        "train_count=%d\nvalidation_count=%d\nbatch_size=%zu\n"
        "train_batches_per_epoch=%" PRIu64 "\n"
        "validation_batches_per_epoch=%" PRIu64 "\n"
        "train_last_batch=%zu\nvalidation_last_batch=%zu\n"
        "init_seed=%" PRIu64 "\nlatent_seed=%" PRIu64 "\n"
        "augmentation_seed=%" PRIu64 "\nshuffle_seed=%" PRIu64 "\n"
        "augmentation_policy=stateless_event_epoch_v1\n"
        "augmentation_mode=symmetric_figure\n"
        "augmentation_crop_left_domain=0..48\n"
        "augmentation_shift_domain=-24..24\n"
        "shuffle_policy=fisher_yates_unbiased_v1\n"
        "latent_train_policy=event_epoch_stateless_v1\n"
        "latent_validation_policy=event_fixed_stateless_v1\n"
        "validation_policy=central_stochastic_eval_fixed_epsilon\n"
        "diagnostic_policy=central_mean_eval\n"
        "adam_learning_rate=%.17g\nadam_beta1=%.17g\n"
        "adam_beta2=%.17g\nadam_epsilon=%.17g\nbeta_kl=1\n"
        "max_epochs=%" PRIu64 "\nearly_stopping_patience=%" PRIu64 "\n"
        "validation_every=1\nmean_eval_every=%d\n"
        "epoch_indexing=zero_based\n"
        "best_policy=strict_min_validation_total_mean_per_sample\n"
        "best_tie_break=earliest_epoch\n"
        "checkpoint_resume_boundary=epoch\n"
        "checkpoint_format_version=%u\n"
        "latest_is_resume_authority=true\nrng_state_serialized=false\n"
        "labels_used=false\n"
        "dataset_version=%s\n"
        "dataset_identity_sha256=%s\n"
        "event_manifest=manifest_events_272.csv\n"
        "event_manifest_sha256=%s\n"
        "tensor_manifest=manifest_tensors_224_central.csv\n"
        "tensor_manifest_sha256=%s\n"
        "train_split=split_train_indices_seed123.csv\n"
        "train_split_sha256=%s\n"
        "train_subset=sakai_multires_c/configs/subset224_train_800_seed123.csv\n"
        "train_subset_sha256=%s\n"
        "train_selection_identity_sha256=%s\n"
        "validation_split=split_val_indices_seed123.csv\n"
        "validation_split_sha256=%s\n"
        "validation_subset=sakai_multires_c/configs/subset224_val_200_seed123.csv\n"
        "validation_subset_sha256=%s\n"
        "validation_selection_identity_sha256=%s\n"
        "compiler_id=%s\ncompiler_version=%s\ncompiler_target=%s\n",
        run->output_basename, run->identity.code_commit,
        TRAINING_PROFILE_ID,
        TRAINING_PROFILE_SHA,
        TRAINING_TRAIN_COUNT, TRAINING_VALIDATION_COUNT,
        run->batch_size, run->train_batches,
        run->validation_batches, run->train_last_batch,
        run->validation_last_batch, TRAINING_INIT_SEED,
        TRAINING_LATENT_SEED, TRAINING_AUGMENTATION_SEED,
        TRAINING_SHUFFLE_SEED, VAE_C_ADAM_LEARNING_RATE_V1,
        VAE_C_ADAM_BETA1_V1, VAE_C_ADAM_BETA2_V1,
        VAE_C_ADAM_EPSILON_V1, run->max_epochs,
        run->patience, TRAINING_MEAN_EVAL_EVERY,
        VAE_C_CHECKPOINT_VERSION,
        TRAINING_DATASET_VERSION, TRAINING_DATASET_ID,
        TRAINING_EVENT_SHA, TRAINING_TENSOR_SHA,
        TRAINING_TRAIN_SPLIT_SHA, TRAINING_TRAIN_SUBSET_SHA,
        TRAINING_TRAIN_SELECTION_ID,
        TRAINING_VALIDATION_SPLIT_SHA,
        TRAINING_VALIDATION_SUBSET_SHA,
        TRAINING_VALIDATION_SELECTION_ID,
        run->identity.compiler_id,
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
    bool *train_seen = NULL;
    bool *validation_seen = NULL;
    bool *mean_seen = NULL;
    double *validation_means = NULL;
    size_t epoch_capacity;
    bool initial_stochastic = false;
    bool initial_mean = false;
    bool trailing = false;
    bool any_dropped = false;
    int result = 1;

    if (run == NULL || final_row_present == NULL ||
        latest_epoch >= run->max_epochs ||
        run->max_epochs > (uint64_t)SIZE_MAX) {
        return 1;
    }
    epoch_capacity = (size_t)run->max_epochs;
    train_seen = (bool *)calloc(epoch_capacity, sizeof(*train_seen));
    validation_seen = (bool *)calloc(
        epoch_capacity, sizeof(*validation_seen));
    mean_seen = (bool *)calloc(epoch_capacity, sizeof(*mean_seen));
    validation_means = (double *)calloc(
        epoch_capacity, sizeof(*validation_means));
    if (train_seen == NULL || validation_seen == NULL ||
        mean_seen == NULL || validation_means == NULL) {
        goto cleanup;
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
                   (uint64_t)row.epoch < run->max_epochs) {
            uint64_t expected_step =
                ((uint64_t)row.epoch + UINT64_C(1)) *
                run->train_batches;
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
                   (uint64_t)row.epoch < run->max_epochs) {
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
                 reconstructed_no_improvement >= run->patience)) {
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
        terminal = latest_epoch == run->max_epochs - UINT64_C(1) ||
            reconstructed_no_improvement >= run->patience;
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
    free(validation_means);
    free(mean_seen);
    free(validation_seen);
    free(train_seen);
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
        run->controller.last_completed_epoch >= run->max_epochs ||
        run->optimizer.step !=
            (run->controller.last_completed_epoch + UINT64_C(1)) *
            run->train_batches ||
        run->controller.no_improvement_count > run->patience) {
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
            run->batch_size) != VAE_C_OP_OK ||
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
            run->batch_size) != VAE_C_OP_OK ||
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

static int training_configure(
    TrainingRun *run,
    size_t batch_size,
    uint64_t max_epochs,
    uint64_t patience
);

static int training_validate_only_hard_gates(TrainingRun *run)
{
    VAECConfig config;
    TrainingRun invalid_config;
    VAECProfileCatalog v1_catalog;
    GSMultiresEventLoader loader;
    GSMultiresBatch train_batch;
    GSMultiresBatch validation_batch;
    char v1_path[TRAINING_PATH_CAP];
    char v2_path[TRAINING_PATH_CAP];
    char actual_sha[VAE_C_SHA256_HEX_SIZE];
    char error_message[256];
    size_t *first_permutation = NULL;
    size_t *second_permutation = NULL;
    float epsilon_a[TRAINING_LATENT_VALUES];
    float epsilon_b[TRAINING_LATENT_VALUES];
    size_t value_count;
    size_t index;
    int crop_left = -1;
    int result = 1;

    memset(&invalid_config, 0, sizeof(invalid_config));
    vae_c_profile_catalog_init(&v1_catalog);
    memset(&loader, 0, sizeof(loader));
    memset(&train_batch, 0, sizeof(train_batch));
    memset(&validation_batch, 0, sizeof(validation_batch));
    error_message[0] = '\0';

    if (run == NULL ||
        vae_c_config_build(run->profile, &config,
                           error_message,
                           sizeof(error_message)) != VAE_C_CONFIG_OK ||
        config.input_channels != TRAINING_CHANNELS ||
        config.input_side != TRAINING_SIDE || config.conv_depth != 4u ||
        config.channels[0] != 16u || config.channels[1] != 32u ||
        config.channels[2] != 64u || config.channels[3] != 128u ||
        config.latent_dim != TRAINING_LATENT_VALUES ||
        config.bottleneck_side != 7u ||
        config.trainable_parameters_bn1d != UINT64_C(625860) ||
        config.forward_mac_per_sample != UINT64_C(220790784) ||
        training_configure(&invalid_config, 1u, 50u, 10u) == 0 ||
        training_configure(&invalid_config, 799u, 50u, 10u) == 0 ||
        training_configure(&invalid_config, 32u, 0u, 10u) == 0 ||
        training_configure(&invalid_config, 32u, 50u, 0u) == 0 ||
        training_configure(&invalid_config, 32u, 10u, 11u) == 0 ||
        !training_code_commit_is_valid(run->identity.code_commit) ||
        training_code_commit_is_valid("") ||
        training_code_commit_is_valid(
            "not-a-40-character-hexadecimal-commit") ||
        strcmp(run->output_basename,
               "mini75_experiment_v1_full_init123_latent456_aug789_"
               "shuffle101112_bs64") == 0 ||
        strcmp(run->output_basename,
               "mini112_primary_v1_full_init123_latent456_aug789_"
               "shuffle101112_bs64") == 0 ||
        vae_c_model_is_allocated(&run->model) ||
        run->optimizer.configured || run->optimizer.step != 0u ||
        training_path_join(run->repository_root,
            "sakai_multires_c/configs/vae_c_profiles_v1.csv",
            v1_path, sizeof(v1_path)) != 0 ||
        training_path_join(run->repository_root,
            "sakai_multires_c/configs/vae_c_profiles_v2.csv",
            v2_path, sizeof(v2_path)) != 0 ||
        vae_c_profile_catalog_load_csv(
            v1_path, &v1_catalog,
            error_message, sizeof(error_message)) != VAE_C_CONFIG_OK ||
        vae_c_profile_catalog_find(&v1_catalog, TRAINING_PROFILE_ID) != NULL ||
        training_file_sha256(v2_path, actual_sha) != 0 ||
        strcmp(actual_sha, TRAINING_PROFILE_SHA) != 0 ||
        training_hash_matches(
            v2_path,
            "0000000000000000000000000000000000000000000000000000000000000000")) {
        goto cleanup;
    }

    first_permutation = (size_t *)malloc(
        TRAINING_TRAIN_COUNT * sizeof(size_t));
    second_permutation = (size_t *)malloc(
        TRAINING_TRAIN_COUNT * sizeof(size_t));
    if (first_permutation == NULL || second_permutation == NULL ||
        vae_c_shuffle_make_permutation(
            TRAINING_TRAIN_COUNT, TRAINING_SHUFFLE_SEED, 0u,
            first_permutation) != VAE_C_RNG_OK ||
        vae_c_shuffle_make_permutation(
            TRAINING_TRAIN_COUNT, TRAINING_SHUFFLE_SEED, 0u,
            second_permutation) != VAE_C_RNG_OK ||
        memcmp(first_permutation, second_permutation,
               TRAINING_TRAIN_COUNT * sizeof(size_t)) != 0 ||
        gs_augment_crop_left_for_event_mode(
            TRAINING_AUGMENTATION_SEED, 0u,
            (uint64_t)run->train_split.indices[0],
            GS_AUGMENT_MODE_SYMMETRIC_FIGURE,
            &crop_left) != 0 || crop_left < 0 || crop_left > 48 ||
        vae_c_rng_fill_latent_validation_epsilon_f32(
            TRAINING_LATENT_SEED,
            (uint64_t)run->validation_split.indices[0],
            epsilon_a, TRAINING_LATENT_VALUES) != VAE_C_RNG_OK ||
        vae_c_rng_fill_latent_validation_epsilon_f32(
            TRAINING_LATENT_SEED,
            (uint64_t)run->validation_split.indices[0],
            epsilon_b, TRAINING_LATENT_VALUES) != VAE_C_RNG_OK ||
        memcmp(epsilon_a, epsilon_b, sizeof(epsilon_a)) != 0 ||
        gs_multires_event_loader_init(&loader, TRAINING_SIDE) != 0 ||
        gs_multires_batch_alloc(
            &train_batch, &run->tensor_manifest.spec, 2u) != 0 ||
        gs_multires_batch_alloc(
            &validation_batch, &run->tensor_manifest.spec, 2u) != 0 ||
        gs_multires_batch_load_train_stateless_mode(
            &run->events, &run->train_split, &loader, 0u, 2u,
            TRAINING_AUGMENTATION_SEED, 0u,
            GS_AUGMENT_MODE_SYMMETRIC_FIGURE,
            &train_batch) != 0 ||
        gs_multires_batch_load_validation(
            &run->events, &run->tensor_manifest,
            &run->validation_split, 0u, 2u,
            &validation_batch) != 0 ||
        train_batch.count != 2u || validation_batch.count != 2u) {
        goto cleanup;
    }
    for (index = 0; index < 2u; index++) {
        if (train_batch.manifest_indices[index] !=
                run->train_split.indices[index] ||
            validation_batch.manifest_indices[index] !=
                run->validation_split.indices[index] ||
            train_batch.crop_lefts[index] < 0 ||
            train_batch.crop_lefts[index] > 48 ||
            validation_batch.crop_lefts[index] != 24) {
            goto cleanup;
        }
    }
    value_count = 2u * run->tensor_manifest.spec.values_per_event;
    for (index = 0; index < value_count; index++) {
        if (!isfinite(train_batch.x[index]) ||
            train_batch.x[index] < 0.0f || train_batch.x[index] > 1.0f ||
            !isfinite(validation_batch.x[index]) ||
            validation_batch.x[index] < 0.0f ||
            validation_batch.x[index] > 1.0f) {
            goto cleanup;
        }
    }
    result = 0;

cleanup:
    free(second_permutation);
    free(first_permutation);
    gs_multires_batch_free(&validation_batch);
    gs_multires_batch_free(&train_batch);
    gs_multires_event_loader_free(&loader);
    vae_c_profile_catalog_free(&v1_catalog);
    return result;
}

static int training_validate_only(TrainingRun *run)
{
    bool final_is_directory = false;
    bool partial_is_directory = false;
    int final_state;
    int partial_state;

    if (run == NULL || run->profile == NULL ||
        run->batch_size < 2u ||
        TRAINING_TRAIN_COUNT % run->batch_size == 1u ||
        run->train_split.count != TRAINING_TRAIN_COUNT ||
        run->validation_split.count != TRAINING_VALIDATION_COUNT ||
        run->identity.max_batch_size != (uint64_t)run->batch_size ||
        run->identity.batch_size != (uint64_t)run->batch_size ||
        strcmp(run->identity.dataset_id, TRAINING_DATASET_ID) != 0 ||
        strcmp(run->identity.train_split_id,
               TRAINING_TRAIN_SELECTION_ID) != 0 ||
        strcmp(run->identity.validation_split_id,
               TRAINING_VALIDATION_SELECTION_ID) != 0 ||
        training_validate_only_hard_gates(run) != 0) {
        return 1;
    }
    final_state = training_path_state(
        run->final_output, &final_is_directory);
    partial_state = training_path_state(
        run->partial_output, &partial_is_directory);
    if (final_state < 0 || partial_state < 0 ||
        (final_state == 1 && !final_is_directory) ||
        (partial_state == 1 && !partial_is_directory)) {
        return 1;
    }

    printf("VALIDATE_ONLY=true\nPROFILE=%s\n", TRAINING_PROFILE_ID);
    printf("PROFILE_CATALOG_V2=true\nPROFILE_V1_REJECTED=true\n");
    printf("INPUT_SHAPE=4x224x224\nDEPTH=4\nCHANNELS=16,32,64,128\n");
    printf("LATENT_DIM=64\nTRAIN_COUNT=%d\nVALIDATION_COUNT=%d\n",
           TRAINING_TRAIN_COUNT, TRAINING_VALIDATION_COUNT);
    printf("BATCH_SIZE=%zu\nTRAIN_TAIL=%zu\nVALIDATION_TAIL=%zu\n",
           run->batch_size, run->train_last_batch,
           run->validation_last_batch);
    printf("MAX_EPOCHS=%" PRIu64 "\nPATIENCE=%" PRIu64 "\n",
           run->max_epochs, run->patience);
    printf("DATASET_IDENTITY=%s\nTRAIN_SELECTION_IDENTITY=%s\n"
           "VALIDATION_SELECTION_IDENTITY=%s\n",
           run->identity.dataset_id, run->identity.train_split_id,
           run->identity.validation_split_id);
    printf("ORDERED_PREFIX_TRAIN800=true\n"
           "ORDERED_PREFIX_VALIDATION200=true\n"
           "SUBSET_OVERLAP=0\n");
    printf("AUGMENTATION_GLOBAL_MANIFEST_KEY=true\n");
    printf("VALIDATION_CENTRAL_STOCHASTIC_FIXED_EPSILON=true\n");
    printf("LABELS_USED=false\nOPTIMIZER_STEP=0\n");
    printf("FORWARD_CALLS=0\nBACKWARD_CALLS=0\nADAM_STEPS=0\n");
    printf("CHECKPOINTS_CREATED=0\nOUTPUT_CREATED=false\n");
    printf("OUTPUT_FINAL_EXISTS=%s\nOUTPUT_PARTIAL_EXISTS=%s\n",
           final_state == 1 ? "true" : "false",
           partial_state == 1 ? "true" : "false");
    printf("MINI224 TRAINING VALIDATION PASSED\n");
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
    while (next_epoch < run->max_epochs &&
           run->controller.no_improvement_count < run->patience) {
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
            (next_epoch + UINT64_C(1)) * run->train_batches;
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
        stop = run->controller.no_improvement_count >= run->patience ||
            next_epoch + UINT64_C(1) >= run->max_epochs;
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
        (next_epoch >= run->max_epochs ||
         run->controller.no_improvement_count >= run->patience)) {
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

static int training_parse_u64(
    const char *text,
    uint64_t *value
)
{
    char *end = NULL;
    unsigned long long parsed;

    if (text == NULL || value == NULL || text[0] == '\0' ||
        text[0] == '+' || text[0] == '-') {
        return 1;
    }
    errno = 0;
    parsed = strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' ||
        (uint64_t)parsed != parsed) {
        return 1;
    }
    *value = (uint64_t)parsed;
    return 0;
}

static int training_configure(
    TrainingRun *run,
    size_t batch_size,
    uint64_t max_epochs,
    uint64_t patience
)
{
    size_t train_remainder;
    size_t validation_remainder;
    int written;

    if (run == NULL || batch_size < 2u ||
        batch_size > TRAINING_TRAIN_COUNT ||
        TRAINING_TRAIN_COUNT % batch_size == 1u ||
        max_epochs == 0u || max_epochs > (uint64_t)INT64_MAX ||
        patience == 0u || patience > max_epochs) {
        return 1;
    }
    run->batch_size = batch_size;
    train_remainder = TRAINING_TRAIN_COUNT % batch_size;
    validation_remainder = TRAINING_VALIDATION_COUNT % batch_size;
    run->train_last_batch = train_remainder == 0u
        ? batch_size : train_remainder;
    run->validation_last_batch = validation_remainder == 0u
        ? batch_size : validation_remainder;
    run->train_batches =
        (uint64_t)(TRAINING_TRAIN_COUNT / batch_size) +
        (train_remainder == 0u ? UINT64_C(0) : UINT64_C(1));
    run->validation_batches =
        (uint64_t)(TRAINING_VALIDATION_COUNT / batch_size) +
        (validation_remainder == 0u ? UINT64_C(0) : UINT64_C(1));
    if (run->train_batches == 0u ||
        max_epochs > UINT64_MAX / run->train_batches) {
        return 1;
    }
    run->max_epochs = max_epochs;
    run->patience = patience;
    written = snprintf(
        run->output_basename, sizeof(run->output_basename),
        "mini224_bridge_v1_subset1000_candidate_"
        "init123_latent456_aug789_shuffle101112_"
        "bs%zu_max%" PRIu64 "_pat%" PRIu64,
        batch_size, max_epochs, patience
    );
    return written < 0 ||
        (size_t)written >= sizeof(run->output_basename) ? 1 : 0;
}

static void training_print_usage(const char *program)
{
    printf(
        "Uso: %s [--validate-only|--resume] --batch-size N "
        "--max-epochs N --patience N --code-commit HASH "
        "[repository_root]\n\n"
        "--validate-only verifica profilo V2, hash, identita' canoniche, "
        "subset, mapping, tail e identity checkpoint senza allocare il "
        "modello, senza forward/backward/Adam e senza output.\n"
        "--resume riprende esclusivamente dalla directory .partial e dal "
        "latest.ckpt compatibile.\n"
        "Senza modalita' esegue il candidato di training richiesto; "
        "batch, max_epochs e patience non hanno default impliciti.\n",
        program
    );
}

#ifndef VAE_C_MINI224_RUNNER_TESTING
int main(int argc, char **argv)
{
    const char *repository_root = ".";
    const char *code_commit = NULL;
    const char *batch_text = NULL;
    const char *max_epochs_text = NULL;
    const char *patience_text = NULL;
    bool validate_only = false;
    bool resume = false;
    bool repository_seen = false;
    uint64_t batch_u64 = 0;
    uint64_t max_epochs = 0;
    uint64_t patience = 0;
    TrainingRun run;
    int argument;
    int result = 1;

    for (argument = 1; argument < argc; argument++) {
        if (strcmp(argv[argument], "--help") == 0 ||
            strcmp(argv[argument], "-h") == 0) {
            training_print_usage(argv[0]);
            return 0;
        }
    }
    for (argument = 1; argument < argc; argument++) {
        if (strcmp(argv[argument], "--validate-only") == 0) {
            if (validate_only) {
                return 2;
            }
            validate_only = true;
        } else if (strcmp(argv[argument], "--resume") == 0) {
            if (resume) {
                return 2;
            }
            resume = true;
        } else if (strcmp(argv[argument], "--batch-size") == 0) {
            if (batch_text != NULL || argument + 1 >= argc) {
                return 2;
            }
            batch_text = argv[++argument];
        } else if (strcmp(argv[argument], "--max-epochs") == 0) {
            if (max_epochs_text != NULL || argument + 1 >= argc) {
                return 2;
            }
            max_epochs_text = argv[++argument];
        } else if (strcmp(argv[argument], "--patience") == 0) {
            if (patience_text != NULL || argument + 1 >= argc) {
                return 2;
            }
            patience_text = argv[++argument];
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
    if ((validate_only && resume) ||
        !training_code_commit_is_valid(code_commit) ||
        training_parse_u64(batch_text, &batch_u64) != 0 ||
        batch_u64 > (uint64_t)SIZE_MAX ||
        training_parse_u64(max_epochs_text, &max_epochs) != 0 ||
        training_parse_u64(patience_text, &patience) != 0) {
        fprintf(stderr, "Errore opzioni mini224.\n");
        training_print_usage(argv[0]);
        return 2;
    }

    training_run_init(&run);
    if (training_configure(
            &run, (size_t)batch_u64, max_epochs, patience) != 0 ||
        training_load_contract(
            &run, repository_root, code_commit) != 0 ||
        training_build_output_paths(&run) != 0) {
        goto cleanup;
    }
    if (validate_only) {
        result = training_validate_only(&run);
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
        if (training_allocate_model(&run) != 0) {
            goto cleanup;
        }
        result = training_full_run(&run, resume);
    }

cleanup:
    training_run_free(&run);
    return result;
}
#endif
