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
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

enum {
    SMOKE_PATH_CAP = 4096,
    SMOKE_BATCH = 2,
    SMOKE_SIDE = 224,
    SMOKE_INPUT_VALUES = 4 * 224 * 224,
    SMOKE_LATENT = 64,
    SMOKE_TRAIN_COUNT = 800,
    SMOKE_VALIDATION_COUNT = 200
};

#define SMOKE_PROFILE_ID "mini224_bridge_v1"
#define SMOKE_INIT_SEED UINT64_C(123)
#define SMOKE_LATENT_SEED UINT64_C(456)
#define SMOKE_AUGMENTATION_SEED UINT64_C(789)
#define SMOKE_SHUFFLE_SEED UINT64_C(101112)

#define SMOKE_EVENT_SHA \
    "620960e6b35da31bc4cf1a8a6da8679027d378e1d4b2ac73c5267c1c5311ae82"
#define SMOKE_TENSOR_SHA \
    "94b7e2e051bd32de72dd2cb6a5fb7eda9bcaec123cbf3a07d0d901bd19779afa"
#define SMOKE_TRAIN_SPLIT_SHA \
    "8c7a513b98dbe996261bc80965500ab15ab77f498ec9440858003152e42f926f"
#define SMOKE_VALIDATION_SPLIT_SHA \
    "e63109c88df7b3ee97317f94791f26df3fcaf8b937921cfb27967d01c2546467"
#define SMOKE_TRAIN_SUBSET_SHA \
    "80c5396a70e7f2c8e3a00680582945581ae76c301a5b9ba8c62e006bb25cdfb5"
#define SMOKE_VALIDATION_SUBSET_SHA \
    "9f5fe6c50c7398588f2de7bd4a543a54a53e7f7a0b9a84bf4004a01f065ae76f"
#define SMOKE_PROFILE_SHA \
    "7ac5e1b5f5a76265144e9a09ce8e1e1ec976ace76f664f86b432a20a45a19ad1"
#define SMOKE_DATASET_ID \
    "389fc0326101afcae22350445114e9b42caf64b059d9d6d923f25d83ce3377b0"
#define SMOKE_TRAIN_SELECTION_ID \
    "b9f311f6a655afed553a7e64affb850a92be58034449550b25e4f60685e5b0a5"
#define SMOKE_VALIDATION_SELECTION_ID \
    "cf1a6c62ca6acf45e466d6128b52018a33fbf4052aeea46f60533017abfeb831"

typedef struct {
    VAECProfileCatalog catalog;
    const VAECProfile *profile;
    GSEventManifest events;
    GSSavedSplit train_subset;
    GSSavedSplit validation_subset;
    GSTensorManifest tensor_manifest;
    GSMultiresEventLoader loader;
    GSMultiresBatch train_batch;
    GSMultiresBatch validation_batch;
    VAECModel model;
    VAECAdam optimizer;
    VAECModelWorkspace forward_workspace;
    VAECModelWorkspace train_workspace;
    VAECModelWorkspace validation_workspace;
    VAECCheckpointRunIdentity identity;
} SmokeRun;

static void smoke_init(SmokeRun *run)
{
    memset(run, 0, sizeof(*run));
    vae_c_profile_catalog_init(&run->catalog);
    vae_c_model_init(&run->model);
    vae_c_adam_init(&run->optimizer);
    vae_c_model_workspace_init(&run->forward_workspace);
    vae_c_model_workspace_init(&run->train_workspace);
    vae_c_model_workspace_init(&run->validation_workspace);
    vae_c_checkpoint_run_identity_init(&run->identity);
}

static void smoke_free(SmokeRun *run)
{
    vae_c_model_workspace_free(&run->validation_workspace);
    vae_c_model_workspace_free(&run->train_workspace);
    vae_c_model_workspace_free(&run->forward_workspace);
    vae_c_adam_free(&run->optimizer);
    vae_c_model_free(&run->model);
    gs_multires_batch_free(&run->validation_batch);
    gs_multires_batch_free(&run->train_batch);
    gs_multires_event_loader_free(&run->loader);
    gs_tensor_manifest_free(&run->tensor_manifest);
    gs_saved_split_free(&run->validation_subset);
    gs_saved_split_free(&run->train_subset);
    gs_event_manifest_free(&run->events);
    vae_c_profile_catalog_free(&run->catalog);
}

static int smoke_path_join(
    const char *directory,
    const char *name,
    char path[SMOKE_PATH_CAP]
)
{
    int length = snprintf(path, SMOKE_PATH_CAP, "%s%s%s", directory,
        directory[0] != '\0' &&
        directory[strlen(directory) - 1] != '/' &&
        directory[strlen(directory) - 1] != '\\' ? "/" : "", name);
    return length < 0 || length >= SMOKE_PATH_CAP ? 1 : 0;
}

static int smoke_file_sha256(const char *path, char hex[65])
{
    FILE *file = fopen(path, "rb");
    unsigned char *data = NULL;
    long length;
    size_t size;
    int result = 1;

    if (file == NULL || fseek(file, 0, SEEK_END) != 0 ||
        (length = ftell(file)) < 0 || fseek(file, 0, SEEK_SET) != 0) {
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
        if (data == NULL || fread(data, 1u, size, file) != size ||
            fgetc(file) != EOF || ferror(file)) {
            free(data);
            (void)fclose(file);
            return 1;
        }
    }
    if (fclose(file) == 0 &&
        vae_c_sha256_digest_hex_lower(data, size, hex) == VAE_C_SHA256_OK) {
        result = 0;
    }
    free(data);
    return result;
}

static int smoke_hash_is(const char *path, const char *expected)
{
    char actual[65];
    return smoke_file_sha256(path, actual) == 0 &&
        strcmp(actual, expected) == 0;
}

static int smoke_code_commit_is_valid(const char *value)
{
    size_t index;
    if (value == NULL || strlen(value) != 40u) {
        return 0;
    }
    for (index = 0; index < 40u; index++) {
        unsigned char c = (unsigned char)value[index];
        if (!((c >= '0' && c <= '9') ||
              (c >= 'a' && c <= 'f') ||
              (c >= 'A' && c <= 'F'))) {
            return 0;
        }
    }
    return 1;
}

static int smoke_set_text(char *target, size_t capacity, const char *value)
{
    size_t length = value == NULL ? 0u : strlen(value);
    if (target == NULL || length == 0u || length >= capacity) {
        return 1;
    }
    memcpy(target, value, length + 1u);
    return 0;
}

static const char *smoke_compiler_target(void)
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

static int smoke_build_identity(SmokeRun *run, const char *code_commit)
{
    VAECCheckpointRunIdentity *identity = &run->identity;
    vae_c_checkpoint_run_identity_init(identity);
#define SMOKE_TEXT(field, value) \
    smoke_set_text(identity->field, sizeof(identity->field), value)
    if (SMOKE_TEXT(profile_id, SMOKE_PROFILE_ID) ||
        SMOKE_TEXT(profile_config_sha256, SMOKE_PROFILE_SHA) ||
        SMOKE_TEXT(code_commit, code_commit) ||
        SMOKE_TEXT(dataset_id, SMOKE_DATASET_ID) ||
        SMOKE_TEXT(event_manifest_id, "manifest_events_272.csv") ||
        SMOKE_TEXT(event_manifest_sha256, SMOKE_EVENT_SHA) ||
        SMOKE_TEXT(tensor_manifest_id,
                   "manifest_tensors_224_central.csv") ||
        SMOKE_TEXT(tensor_manifest_sha256, SMOKE_TENSOR_SHA) ||
        SMOKE_TEXT(train_split_id, SMOKE_TRAIN_SELECTION_ID) ||
        SMOKE_TEXT(train_split_sha256, SMOKE_TRAIN_SPLIT_SHA) ||
        SMOKE_TEXT(validation_split_id,
                   SMOKE_VALIDATION_SELECTION_ID) ||
        SMOKE_TEXT(validation_split_sha256,
                   SMOKE_VALIDATION_SPLIT_SHA) ||
        SMOKE_TEXT(augmentation_policy,
                   "stateless_event_epoch_v1+symmetric_figure+side224") ||
        SMOKE_TEXT(shuffle_policy, "fisher_yates_unbiased_v1") ||
        SMOKE_TEXT(latent_policy,
                   "train_event_epoch_v1+validation_event_v1+latent64") ||
        SMOKE_TEXT(validation_policy, "mini224_runtime_smoke_v1") ||
        SMOKE_TEXT(compiler_id, "GNU C") ||
        SMOKE_TEXT(compiler_version, __VERSION__) ||
        SMOKE_TEXT(compiler_target, smoke_compiler_target())) {
        return 1;
    }
#undef SMOKE_TEXT
    identity->initialization_seed = SMOKE_INIT_SEED;
    identity->latent_seed = SMOKE_LATENT_SEED;
    identity->augmentation_seed = SMOKE_AUGMENTATION_SEED;
    identity->shuffle_seed = SMOKE_SHUFFLE_SEED;
    identity->max_batch_size = SMOKE_BATCH;
    identity->batch_size = SMOKE_BATCH;
    identity->optimizer_learning_rate = VAE_C_ADAM_LEARNING_RATE_V1;
    identity->optimizer_beta1 = VAE_C_ADAM_BETA1_V1;
    identity->optimizer_beta2 = VAE_C_ADAM_BETA2_V1;
    identity->optimizer_epsilon = VAE_C_ADAM_EPSILON_V1;
    return 0;
}

static int smoke_profile_is_expected(const VAECProfile *profile)
{
    return profile != NULL &&
        strcmp(profile->profile_id, SMOKE_PROFILE_ID) == 0 &&
        profile->input_side == 224u && profile->conv_depth == 4u &&
        profile->channels[0] == 16u && profile->channels[1] == 32u &&
        profile->channels[2] == 64u && profile->channels[3] == 128u &&
        profile->latent_dim == 64u && profile->bottleneck_side == 7u &&
        profile->expected_trainable_parameters_bn1d == UINT64_C(625860) &&
        profile->expected_forward_mac_per_sample == UINT64_C(220790784) &&
        strcmp(profile->profile_status,
               "candidate_pending_224_smoke") == 0;
}

static int smoke_load_contract(
    SmokeRun *run,
    const char *root,
    const char *code_commit
)
{
    char profile_path[SMOKE_PATH_CAP];
    char event_path[SMOKE_PATH_CAP];
    char tensor_path[SMOKE_PATH_CAP];
    char train_path[SMOKE_PATH_CAP];
    char validation_path[SMOKE_PATH_CAP];
    char train_subset_path[SMOKE_PATH_CAP];
    char validation_subset_path[SMOKE_PATH_CAP];
    char dataset_identity[65];
    char train_identity[65];
    char validation_identity[65];
    char error[512];
    GSTensorSpec spec;

    if (smoke_path_join(root,
            "sakai_multires_c/configs/vae_c_profiles_v2.csv",
            profile_path) ||
        smoke_path_join(root, "manifest_events_272.csv", event_path) ||
        smoke_path_join(root, "manifest_tensors_224_central.csv",
                        tensor_path) ||
        smoke_path_join(root, "split_train_indices_seed123.csv",
                        train_path) ||
        smoke_path_join(root, "split_val_indices_seed123.csv",
                        validation_path) ||
        smoke_path_join(root,
            "sakai_multires_c/configs/subset224_train_800_seed123.csv",
            train_subset_path) ||
        smoke_path_join(root,
            "sakai_multires_c/configs/subset224_val_200_seed123.csv",
            validation_subset_path) ||
        !smoke_hash_is(profile_path, SMOKE_PROFILE_SHA) ||
        !smoke_hash_is(event_path, SMOKE_EVENT_SHA) ||
        !smoke_hash_is(tensor_path, SMOKE_TENSOR_SHA) ||
        !smoke_hash_is(train_path, SMOKE_TRAIN_SPLIT_SHA) ||
        !smoke_hash_is(validation_path, SMOKE_VALIDATION_SPLIT_SHA) ||
        !smoke_hash_is(train_subset_path, SMOKE_TRAIN_SUBSET_SHA) ||
        !smoke_hash_is(validation_subset_path,
                       SMOKE_VALIDATION_SUBSET_SHA)) {
        fprintf(stderr, "SMOKE_ERROR=canonical_path_or_sha\n");
        return 1;
    }
    if (vae_c_identity_build_mini224_dataset_v1(
            "legacy_luma_v1", SMOKE_EVENT_SHA, SMOKE_TENSOR_SHA,
            dataset_identity) != VAE_C_IDENTITY_OK ||
        vae_c_identity_build_mini224_train_selection_v1(
            dataset_identity, SMOKE_TRAIN_SPLIT_SHA,
            SMOKE_TRAIN_SUBSET_SHA, SMOKE_TRAIN_COUNT,
            train_identity) != VAE_C_IDENTITY_OK ||
        vae_c_identity_build_mini224_validation_selection_v1(
            dataset_identity, SMOKE_VALIDATION_SPLIT_SHA,
            SMOKE_VALIDATION_SUBSET_SHA, SMOKE_VALIDATION_COUNT,
            validation_identity) != VAE_C_IDENTITY_OK ||
        strcmp(dataset_identity, SMOKE_DATASET_ID) != 0 ||
        strcmp(train_identity, SMOKE_TRAIN_SELECTION_ID) != 0 ||
        strcmp(validation_identity, SMOKE_VALIDATION_SELECTION_ID) != 0 ||
        smoke_build_identity(run, code_commit)) {
        fprintf(stderr, "SMOKE_ERROR=canonical_identity\n");
        return 1;
    }
    error[0] = '\0';
    if (vae_c_profile_catalog_load_csv(profile_path, &run->catalog,
            error, sizeof(error)) != VAE_C_CONFIG_OK ||
        (run->profile = vae_c_profile_catalog_find(
            &run->catalog, SMOKE_PROFILE_ID)) == NULL ||
        !smoke_profile_is_expected(run->profile) ||
        gs_event_manifest_load(event_path, &run->events) != 0 ||
        run->events.count != 8583u ||
        gs_saved_split_load(train_subset_path, "train", &run->events,
                            &run->train_subset) != 0 ||
        run->train_subset.count != SMOKE_TRAIN_COUNT ||
        gs_saved_split_load(validation_subset_path, "validation",
                            &run->events,
                            &run->validation_subset) != 0 ||
        run->validation_subset.count != SMOKE_VALIDATION_COUNT ||
        gs_tensor_spec_init(&spec, SMOKE_SIDE) != 0 ||
        gs_tensor_manifest_load(tensor_path, &spec,
                                &run->tensor_manifest) != 0 ||
        run->tensor_manifest.count != run->events.count ||
        gs_multires_event_loader_init(&run->loader, SMOKE_SIDE) != 0) {
        fprintf(stderr, "SMOKE_ERROR=contract_load:%s\n", error);
        return 1;
    }
    return 0;
}

static int smoke_tensor_shape(
    const VAECTensor *tensor,
    size_t rank,
    size_t d0,
    size_t d1,
    size_t d2,
    size_t d3
)
{
    return tensor != NULL && tensor->shape.rank == rank &&
        tensor->shape.dims[0] == d0 && tensor->shape.dims[1] == d1 &&
        (rank < 3u || tensor->shape.dims[2] == d2) &&
        (rank < 4u || tensor->shape.dims[3] == d3);
}

static int smoke_workspace_shapes(const VAECModelWorkspace *workspace)
{
    return workspace->owned_tensor_count == 38u &&
        workspace->storage_bytes == 60620288u &&
        smoke_tensor_shape(&workspace->input, 4, 2, 4, 224, 224) &&
        smoke_tensor_shape(&workspace->encoder_conv_outputs[0],
                           4, 2, 16, 112, 112) &&
        smoke_tensor_shape(&workspace->encoder_pool_output,
                           4, 2, 16, 56, 56) &&
        smoke_tensor_shape(&workspace->encoder_conv_outputs[1],
                           4, 2, 32, 28, 28) &&
        smoke_tensor_shape(&workspace->encoder_conv_outputs[2],
                           4, 2, 64, 14, 14) &&
        smoke_tensor_shape(&workspace->encoder_conv_outputs[3],
                           4, 2, 128, 7, 7) &&
        smoke_tensor_shape(&workspace->encoder_global_avg_output,
                           2, 2, 128, 0, 0) &&
        smoke_tensor_shape(&workspace->mu, 2, 2, 64, 0, 0) &&
        smoke_tensor_shape(&workspace->logvar, 2, 2, 64, 0, 0) &&
        smoke_tensor_shape(&workspace->decoder_reshape_view,
                           4, 2, 128, 7, 7) &&
        smoke_tensor_shape(&workspace->decoder_upsample_outputs[0],
                           4, 2, 128, 14, 14) &&
        smoke_tensor_shape(&workspace->decoder_upsample_outputs[1],
                           4, 2, 128, 28, 28) &&
        smoke_tensor_shape(&workspace->decoder_upsample_outputs[2],
                           4, 2, 64, 56, 56) &&
        smoke_tensor_shape(&workspace->decoder_upsample_outputs[3],
                           4, 2, 32, 112, 112) &&
        smoke_tensor_shape(&workspace->decoder_upsample_outputs[4],
                           4, 2, 16, 224, 224) &&
        smoke_tensor_shape(&workspace->logits, 4, 2, 4, 224, 224);
}

static size_t smoke_collect_tensors(
    const VAECModelWorkspace *workspace,
    const VAECTensor *tensors[VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS]
)
{
    size_t count = 0;
    size_t index;
    tensors[count++] = &workspace->input;
    for (index = 0; index < workspace->config.conv_depth; index++) {
        tensors[count++] = &workspace->encoder_conv_outputs[index];
        tensors[count++] = &workspace->encoder_batch_norm_outputs[index];
        tensors[count++] = &workspace->encoder_relu_outputs[index];
    }
    tensors[count++] = &workspace->encoder_pool_output;
    tensors[count++] = &workspace->encoder_global_avg_output;
    tensors[count++] = &workspace->encoder_packed_output;
    tensors[count++] = &workspace->mu;
    tensors[count++] = &workspace->logvar;
    tensors[count++] = &workspace->epsilon;
    tensors[count++] = &workspace->z;
    tensors[count++] = &workspace->decoder_fc_output;
    tensors[count++] = &workspace->decoder_fc_batch_norm_output;
    tensors[count++] = &workspace->decoder_fc_relu_output;
    for (index = 0; index < workspace->config.decoder_upsample_count;
         index++) {
        tensors[count++] = &workspace->decoder_upsample_outputs[index];
    }
    for (index = 0; index < workspace->config.decoder_hidden_conv_count;
         index++) {
        tensors[count++] = &workspace->decoder_hidden_conv_outputs[index];
        tensors[count++] =
            &workspace->decoder_hidden_batch_norm_outputs[index];
        tensors[count++] = &workspace->decoder_hidden_relu_outputs[index];
    }
    tensors[count++] = &workspace->logits;
    return count;
}

static int smoke_values_finite(const float *values, size_t count)
{
    size_t index;
    if (values == NULL) {
        return 0;
    }
    for (index = 0; index < count; index++) {
        if (!isfinite(values[index])) {
            return 0;
        }
    }
    return 1;
}

static int smoke_activations_finite(const VAECModelWorkspace *workspace)
{
    const VAECTensor *tensors[VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS];
    size_t count = smoke_collect_tensors(workspace, tensors);
    size_t index;
    if (count != workspace->owned_tensor_count) {
        return 0;
    }
    for (index = 0; index < count; index++) {
        if (!smoke_values_finite(tensors[index]->data,
                                 tensors[index]->shape.count)) {
            return 0;
        }
    }
    return 1;
}

static int smoke_gradients_finite(
    const VAECModel *model,
    const VAECModelWorkspace *workspace
)
{
    const VAECTensor *tensors[VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS];
    size_t count = smoke_collect_tensors(workspace, tensors);
    size_t index;
    for (index = 0; index < model->parameter_count; index++) {
        const VAECParameter *parameter = model->parameters[index];
        if (!smoke_values_finite(parameter->gradients,
                                 parameter->shape.count)) {
            return 0;
        }
    }
    for (index = 0; index < count; index++) {
        if (tensors[index]->has_grad &&
            !smoke_values_finite(tensors[index]->grad,
                                 tensors[index]->shape.count)) {
            return 0;
        }
    }
    return 1;
}

static int smoke_stats_valid(const VAECVAELossStats *stats)
{
    return stats != NULL && stats->batch_size == SMOKE_BATCH &&
        stats->reconstruction.total_values ==
            (size_t)SMOKE_BATCH * SMOKE_INPUT_VALUES &&
        stats->kl.total_values == (size_t)SMOKE_BATCH * SMOKE_LATENT &&
        isfinite(stats->total_sum) &&
        isfinite(stats->reconstruction.total_sum) &&
        isfinite(stats->kl.total_sum);
}

static size_t smoke_parameter_float_count(const VAECModel *model)
{
    size_t total = 0;
    size_t index;
    for (index = 0; index < model->parameter_count; index++) {
        total += model->parameters[index]->shape.count;
    }
    return total;
}

static void smoke_copy_parameter_values(const VAECModel *model, float *copy)
{
    size_t offset = 0;
    size_t index;
    for (index = 0; index < model->parameter_count; index++) {
        size_t count = model->parameters[index]->shape.count;
        memcpy(copy + offset, model->parameters[index]->values,
               count * sizeof(float));
        offset += count;
    }
}

static int smoke_parameter_values_equal(
    const VAECModel *model,
    const float *copy
)
{
    size_t offset = 0;
    size_t index;
    for (index = 0; index < model->parameter_count; index++) {
        size_t count = model->parameters[index]->shape.count;
        if (memcmp(copy + offset, model->parameters[index]->values,
                   count * sizeof(float)) != 0) {
            return 0;
        }
        offset += count;
    }
    return 1;
}

static size_t smoke_bn_float_count(const VAECModel *model)
{
    size_t total = model->decoder_fc_batch_norm.num_features;
    size_t index;
    for (index = 0; index < model->config.conv_depth; index++) {
        total += model->encoder_batch_norms[index].num_features;
    }
    for (index = 0; index < model->config.decoder_hidden_conv_count;
         index++) {
        total += model->decoder_hidden_batch_norms[index].num_features;
    }
    return total * 2u;
}

static void smoke_copy_bn_state(const VAECModel *model, float *copy)
{
    size_t offset = 0;
    size_t index;
#define SMOKE_COPY_BN(layer) do { \
    size_t smoke_n = (layer).num_features; \
    memcpy(copy + offset, (layer).running_mean, smoke_n * sizeof(float)); \
    offset += smoke_n; \
    memcpy(copy + offset, (layer).running_variance, \
           smoke_n * sizeof(float)); \
    offset += smoke_n; \
} while (0)
    for (index = 0; index < model->config.conv_depth; index++) {
        SMOKE_COPY_BN(model->encoder_batch_norms[index]);
    }
    SMOKE_COPY_BN(model->decoder_fc_batch_norm);
    for (index = 0; index < model->config.decoder_hidden_conv_count;
         index++) {
        SMOKE_COPY_BN(model->decoder_hidden_batch_norms[index]);
    }
#undef SMOKE_COPY_BN
}

static int smoke_persistent_equal(
    const VAECModel *left,
    const VAECAdam *left_optimizer,
    const VAECModel *right,
    const VAECAdam *right_optimizer
)
{
    size_t index;
    if (left->parameter_count != right->parameter_count ||
        left_optimizer->learning_rate != right_optimizer->learning_rate ||
        left_optimizer->beta1 != right_optimizer->beta1 ||
        left_optimizer->beta2 != right_optimizer->beta2 ||
        left_optimizer->adam_epsilon != right_optimizer->adam_epsilon ||
        left_optimizer->step != right_optimizer->step ||
        left_optimizer->beta1_power != right_optimizer->beta1_power ||
        left_optimizer->beta2_power != right_optimizer->beta2_power) {
        return 0;
    }
    for (index = 0; index < left->parameter_count; index++) {
        const VAECParameter *a = left->parameters[index];
        const VAECParameter *b = right->parameters[index];
        size_t bytes = a->shape.count * sizeof(float);
        if (a->parameter_key != b->parameter_key ||
            !vae_c_shape_equal(&a->shape, &b->shape) ||
            memcmp(a->values, b->values, bytes) != 0 ||
            memcmp(a->adam_m, b->adam_m, bytes) != 0 ||
            memcmp(a->adam_v, b->adam_v, bytes) != 0) {
            return 0;
        }
    }
#define SMOKE_COMPARE_BN(a, b) do { \
    size_t smoke_bytes = (a).num_features * sizeof(float); \
    if ((a).num_features != (b).num_features || \
        memcmp((a).running_mean, (b).running_mean, smoke_bytes) != 0 || \
        memcmp((a).running_variance, (b).running_variance, \
               smoke_bytes) != 0) { return 0; } \
} while (0)
    for (index = 0; index < left->config.conv_depth; index++) {
        SMOKE_COMPARE_BN(left->encoder_batch_norms[index],
                         right->encoder_batch_norms[index]);
    }
    SMOKE_COMPARE_BN(left->decoder_fc_batch_norm,
                     right->decoder_fc_batch_norm);
    for (index = 0; index < left->config.decoder_hidden_conv_count;
         index++) {
        SMOKE_COMPARE_BN(left->decoder_hidden_batch_norms[index],
                         right->decoder_hidden_batch_norms[index]);
    }
#undef SMOKE_COMPARE_BN
    return 1;
}

static int smoke_path_exists(const char *path)
{
    struct stat info;
    return stat(path, &info) == 0;
}

static int smoke_fill_validation_epsilon(
    VAECModelWorkspace *workspace,
    const size_t indices[SMOKE_BATCH]
)
{
    size_t sample;
    for (sample = 0; sample < SMOKE_BATCH; sample++) {
        if (vae_c_rng_fill_latent_validation_epsilon_f32(
                SMOKE_LATENT_SEED, (uint64_t)indices[sample],
                workspace->epsilon.data + sample * SMOKE_LATENT,
                SMOKE_LATENT) != VAE_C_RNG_OK) {
            return 1;
        }
    }
    return 0;
}

static int smoke_run(
    SmokeRun *run,
    const char *checkpoint_path
)
{
    size_t *permutation = NULL;
    size_t shuffled_indices[SMOKE_BATCH];
    size_t validation_indices[SMOKE_BATCH];
    uint64_t manifest_indices[SMOKE_BATCH];
    GSSavedSplit shuffled_split;
    VAECVAELossStats forward_stats;
    VAECVAELossStats train_stats;
    VAECVAELossStats validation_stats_a;
    VAECVAELossStats validation_stats_b;
    VAECCheckpointMetadata metadata;
    VAECCheckpointMetadata loaded_metadata;
    VAECModel loaded_model;
    VAECAdam loaded_optimizer;
    float *parameter_before = NULL;
    float *parameter_after = NULL;
    float *bn_before = NULL;
    float *bn_after_train = NULL;
    float *validation_logits = NULL;
    float *input_copy = NULL;
    size_t parameter_count;
    size_t bn_count;
    size_t input_bytes = (size_t)SMOKE_BATCH * SMOKE_INPUT_VALUES *
        sizeof(float);
    size_t logits_bytes = input_bytes;
    size_t sample;
    char checkpoint_temp[SMOKE_PATH_CAP];
    bool checkpoint_created = false;
    int result = 1;

    memset(&shuffled_split, 0, sizeof(shuffled_split));
    vae_c_model_init(&loaded_model);
    vae_c_adam_init(&loaded_optimizer);
    vae_c_checkpoint_metadata_init(&metadata);
    vae_c_checkpoint_metadata_init(&loaded_metadata);

    if (snprintf(checkpoint_temp, sizeof(checkpoint_temp), "%s.tmp",
                 checkpoint_path) < 0 ||
        strlen(checkpoint_path) + 4u >= sizeof(checkpoint_temp) ||
        smoke_path_exists(checkpoint_path) ||
        smoke_path_exists(checkpoint_temp)) {
        fprintf(stderr, "SMOKE_ERROR=checkpoint_path_not_fresh\n");
        goto cleanup;
    }
    if (vae_c_model_allocate(&run->model, run->profile, SMOKE_BATCH) !=
            VAE_C_OP_OK ||
        vae_c_model_initialize_parameters_v1(
            &run->model, SMOKE_INIT_SEED) != VAE_C_OP_OK ||
        vae_c_adam_configure_v1(&run->optimizer) != VAE_C_OP_OK ||
        vae_c_model_workspace_allocate(
            &run->model, SMOKE_BATCH, &run->forward_workspace) !=
            VAE_C_OP_OK ||
        vae_c_model_workspace_allocate(
            &run->model, SMOKE_BATCH, &run->train_workspace) !=
            VAE_C_OP_OK ||
        vae_c_model_workspace_allocate(
            &run->model, SMOKE_BATCH, &run->validation_workspace) !=
            VAE_C_OP_OK ||
        !smoke_workspace_shapes(&run->forward_workspace) ||
        !smoke_workspace_shapes(&run->train_workspace) ||
        !smoke_workspace_shapes(&run->validation_workspace) ||
        run->model.parameter_count != 28u ||
        run->model.trainable_value_count != UINT64_C(625860)) {
        fprintf(stderr, "SMOKE_ERROR=runtime_allocation_or_shape\n");
        goto cleanup;
    }
    if (gs_multires_batch_alloc(&run->train_batch,
            &run->tensor_manifest.spec, SMOKE_BATCH) != 0 ||
        gs_multires_batch_alloc(&run->validation_batch,
            &run->tensor_manifest.spec, SMOKE_BATCH) != 0) {
        fprintf(stderr, "SMOKE_ERROR=batch_allocation\n");
        goto cleanup;
    }
    permutation = (size_t *)malloc(
        run->train_subset.count * sizeof(size_t));
    if (permutation == NULL ||
        vae_c_shuffle_make_permutation(run->train_subset.count,
            SMOKE_SHUFFLE_SEED, UINT64_C(0), permutation) != VAE_C_RNG_OK) {
        fprintf(stderr, "SMOKE_ERROR=shuffle\n");
        goto cleanup;
    }
    for (sample = 0; sample < SMOKE_BATCH; sample++) {
        if (permutation[sample] >= run->train_subset.count) {
            goto cleanup;
        }
        shuffled_indices[sample] =
            run->train_subset.indices[permutation[sample]];
        validation_indices[sample] = run->validation_subset.indices[sample];
    }
    shuffled_split.indices = shuffled_indices;
    shuffled_split.count = SMOKE_BATCH;
    memcpy(shuffled_split.name, "train", sizeof("train"));
    if (gs_multires_batch_load_train_stateless_mode(
            &run->events, &shuffled_split, &run->loader, 0,
            SMOKE_BATCH, SMOKE_AUGMENTATION_SEED, UINT64_C(0),
            GS_AUGMENT_MODE_SYMMETRIC_FIGURE,
            &run->train_batch) != 0 ||
        gs_multires_batch_load_validation(
            &run->events, &run->tensor_manifest,
            &run->validation_subset, 0, SMOKE_BATCH,
            &run->validation_batch) != 0) {
        fprintf(stderr, "SMOKE_ERROR=real_loader\n");
        goto cleanup;
    }
    for (sample = 0; sample < SMOKE_BATCH; sample++) {
        int expected_crop = -1;
        if (run->train_batch.manifest_indices[sample] !=
                shuffled_indices[sample] ||
            run->validation_batch.manifest_indices[sample] !=
                validation_indices[sample] ||
            gs_augment_crop_left_for_event_mode(
                SMOKE_AUGMENTATION_SEED, UINT64_C(0),
                (uint64_t)shuffled_indices[sample],
                GS_AUGMENT_MODE_SYMMETRIC_FIGURE,
                &expected_crop) != 0 ||
            run->train_batch.crop_lefts[sample] != expected_crop ||
            expected_crop < 0 || expected_crop > 48) {
            fprintf(stderr, "SMOKE_ERROR=batch_mapping\n");
            goto cleanup;
        }
        manifest_indices[sample] = (uint64_t)shuffled_indices[sample];
        printf("TRAIN_SAMPLE_%lu_SUBSET_POSITION=%lu\n",
               (unsigned long)sample,
               (unsigned long)permutation[sample]);
        printf("TRAIN_SAMPLE_%lu_MANIFEST_INDEX=%lu\n",
               (unsigned long)sample,
               (unsigned long)shuffled_indices[sample]);
        printf("TRAIN_SAMPLE_%lu_EVENT_ID=%s\n",
               (unsigned long)sample,
               run->events.records[shuffled_indices[sample]].event_id);
        printf("TRAIN_SAMPLE_%lu_CROP_LEFT=%d\n",
               (unsigned long)sample, expected_crop);
        printf("VALIDATION_SAMPLE_%lu_MANIFEST_INDEX=%lu\n",
               (unsigned long)sample,
               (unsigned long)validation_indices[sample]);
        printf("VALIDATION_SAMPLE_%lu_EVENT_ID=%s\n",
               (unsigned long)sample,
               run->events.records[validation_indices[sample]].event_id);
    }
    memcpy(run->forward_workspace.input.data, run->train_batch.x,
           input_bytes);
    memcpy(run->train_workspace.input.data, run->train_batch.x,
           input_bytes);
    memcpy(run->validation_workspace.input.data,
           run->validation_batch.x, input_bytes);
    input_copy = (float *)malloc(input_bytes);
    parameter_count = smoke_parameter_float_count(&run->model);
    bn_count = smoke_bn_float_count(&run->model);
    parameter_before = (float *)malloc(parameter_count * sizeof(float));
    parameter_after = (float *)malloc(parameter_count * sizeof(float));
    bn_before = (float *)malloc(bn_count * sizeof(float));
    bn_after_train = (float *)malloc(bn_count * sizeof(float));
    validation_logits = (float *)malloc(logits_bytes);
    if (input_copy == NULL || parameter_before == NULL ||
        parameter_after == NULL || bn_before == NULL ||
        bn_after_train == NULL || validation_logits == NULL) {
        fprintf(stderr, "SMOKE_ERROR=snapshot_allocation\n");
        goto cleanup;
    }
    memcpy(input_copy, run->train_workspace.input.data, input_bytes);
    smoke_copy_parameter_values(&run->model, parameter_before);
    smoke_copy_bn_state(&run->model, bn_before);

    for (sample = 0; sample < SMOKE_BATCH; sample++) {
        if (vae_c_rng_fill_latent_train_epsilon_f32(
                SMOKE_LATENT_SEED, UINT64_C(0), manifest_indices[sample],
                run->forward_workspace.epsilon.data +
                    sample * SMOKE_LATENT,
                SMOKE_LATENT) != VAE_C_RNG_OK) {
            goto cleanup;
        }
    }
    vae_c_vae_loss_stats_init(&forward_stats);
    if (vae_c_model_vae_forward(
            &run->model, VAE_C_BATCH_NORM_MODE_TRAIN,
            VAE_C_LATENT_MODE_STOCHASTIC, &run->forward_workspace,
            &forward_stats) != VAE_C_OP_OK ||
        !smoke_stats_valid(&forward_stats) ||
        !smoke_activations_finite(&run->forward_workspace) ||
        !smoke_parameter_values_equal(&run->model, parameter_before)) {
        fprintf(stderr, "SMOKE_ERROR=forward_train_stochastic\n");
        goto cleanup;
    }
    vae_c_vae_loss_stats_init(&train_stats);
    if (vae_c_model_train_step(
            &run->model, &run->train_workspace, &run->optimizer,
            SMOKE_LATENT_SEED, UINT64_C(0), manifest_indices,
            SMOKE_BATCH, &train_stats) != VAE_C_OP_OK ||
        run->optimizer.step != UINT64_C(1) ||
        !smoke_stats_valid(&train_stats) ||
        !smoke_activations_finite(&run->train_workspace) ||
        !smoke_gradients_finite(&run->model, &run->train_workspace) ||
        memcmp(input_copy, run->train_workspace.input.data,
               input_bytes) != 0 ||
        smoke_parameter_values_equal(&run->model, parameter_before)) {
        fprintf(stderr, "SMOKE_ERROR=train_step\n");
        goto cleanup;
    }
    smoke_copy_parameter_values(&run->model, parameter_after);
    smoke_copy_bn_state(&run->model, bn_after_train);
    if (memcmp(bn_before, bn_after_train,
               bn_count * sizeof(float)) == 0) {
        fprintf(stderr, "SMOKE_ERROR=batchnorm_train_not_updated\n");
        goto cleanup;
    }

    if (smoke_fill_validation_epsilon(
            &run->validation_workspace, validation_indices)) {
        goto cleanup;
    }
    vae_c_vae_loss_stats_init(&validation_stats_a);
    if (vae_c_model_vae_forward(
            &run->model, VAE_C_BATCH_NORM_MODE_EVAL,
            VAE_C_LATENT_MODE_STOCHASTIC,
            &run->validation_workspace,
            &validation_stats_a) != VAE_C_OP_OK ||
        !smoke_stats_valid(&validation_stats_a) ||
        !smoke_activations_finite(&run->validation_workspace) ||
        !smoke_parameter_values_equal(&run->model, parameter_after)) {
        fprintf(stderr, "SMOKE_ERROR=validation_first\n");
        goto cleanup;
    }
    memcpy(validation_logits, run->validation_workspace.logits.data,
           logits_bytes);
    if (smoke_fill_validation_epsilon(
            &run->validation_workspace, validation_indices)) {
        goto cleanup;
    }
    vae_c_vae_loss_stats_init(&validation_stats_b);
    if (vae_c_model_vae_forward(
            &run->model, VAE_C_BATCH_NORM_MODE_EVAL,
            VAE_C_LATENT_MODE_STOCHASTIC,
            &run->validation_workspace,
            &validation_stats_b) != VAE_C_OP_OK ||
        memcmp(&validation_stats_a, &validation_stats_b,
               sizeof(validation_stats_a)) != 0 ||
        memcmp(validation_logits, run->validation_workspace.logits.data,
               logits_bytes) != 0 ||
        !smoke_parameter_values_equal(&run->model, parameter_after)) {
        fprintf(stderr, "SMOKE_ERROR=validation_repeat\n");
        goto cleanup;
    }
    {
        float *bn_after_eval = (float *)malloc(bn_count * sizeof(float));
        if (bn_after_eval == NULL) {
            goto cleanup;
        }
        smoke_copy_bn_state(&run->model, bn_after_eval);
        if (memcmp(bn_after_train, bn_after_eval,
                   bn_count * sizeof(float)) != 0) {
            free(bn_after_eval);
            fprintf(stderr, "SMOKE_ERROR=validation_mutated_bn\n");
            goto cleanup;
        }
        free(bn_after_eval);
    }

    metadata.run_identity = run->identity;
    metadata.kind = VAE_C_CHECKPOINT_KIND_LATEST;
    metadata.last_completed_epoch = UINT64_C(0);
    metadata.best_defined = true;
    metadata.best_epoch = UINT64_C(0);
    metadata.best_validation_total_mean_per_sample =
        validation_stats_a.mean_per_sample;
    metadata.no_improvement_count = UINT64_C(0);
    if (vae_c_checkpoint_save_atomic(checkpoint_path, &run->model,
            &run->optimizer, &metadata) != VAE_C_CHECKPOINT_OK) {
        fprintf(stderr, "SMOKE_ERROR=checkpoint_save\n");
        goto cleanup;
    }
    checkpoint_created = true;
    if (vae_c_model_allocate(&loaded_model, run->profile, SMOKE_BATCH) !=
            VAE_C_OP_OK ||
        vae_c_model_initialize_parameters_v1(
            &loaded_model, SMOKE_INIT_SEED) != VAE_C_OP_OK ||
        vae_c_adam_configure_v1(&loaded_optimizer) != VAE_C_OP_OK ||
        vae_c_checkpoint_load(checkpoint_path, &run->identity,
            &loaded_model, &loaded_optimizer,
            &loaded_metadata) != VAE_C_CHECKPOINT_OK ||
        loaded_optimizer.step != UINT64_C(1) ||
        memcmp(&metadata, &loaded_metadata, sizeof(metadata)) != 0 ||
        !smoke_persistent_equal(&run->model, &run->optimizer,
                                &loaded_model, &loaded_optimizer)) {
        fprintf(stderr, "SMOKE_ERROR=checkpoint_roundtrip\n");
        goto cleanup;
    }

    printf("PROFILE_ID=%s\n", run->model.profile_id);
    printf("DEPTH=%lu\n", (unsigned long)run->model.config.conv_depth);
    printf("PARAMETER_TENSORS=%lu\n",
           (unsigned long)run->model.parameter_count);
    printf("TRAINABLE_PARAMETERS=%" PRIu64 "\n",
           run->model.trainable_value_count);
    printf("FORWARD_MAC_PER_SAMPLE=%" PRIu64 "\n",
           run->model.config.forward_mac_per_sample);
    printf("WORKSPACE_OWNED_TENSORS=%lu\n",
           (unsigned long)run->train_workspace.owned_tensor_count);
    printf("WORKSPACE_BATCH2_BYTES=%lu\n",
           (unsigned long)run->train_workspace.storage_bytes);
    printf("WORKSPACE_INSTANCES=3\n");
    printf("MEMORY_LOWER_BOUND_BYTES=%lu\n",
           (unsigned long)(run->forward_workspace.storage_bytes * 3u +
               parameter_count * sizeof(float) * 4u +
               input_bytes * 2u));
    printf("FORWARD_TOTAL=%.17g\n", forward_stats.total_sum);
    printf("FORWARD_RECONSTRUCTION=%.17g\n",
           forward_stats.reconstruction.total_sum);
    printf("FORWARD_KL=%.17g\n", forward_stats.kl.total_sum);
    printf("TRAIN_PREUPDATE_TOTAL=%.17g\n", train_stats.total_sum);
    printf("TRAIN_PREUPDATE_RECONSTRUCTION=%.17g\n",
           train_stats.reconstruction.total_sum);
    printf("TRAIN_PREUPDATE_KL=%.17g\n", train_stats.kl.total_sum);
    printf("VALIDATION_TOTAL=%.17g\n", validation_stats_a.total_sum);
    printf("VALIDATION_RECONSTRUCTION=%.17g\n",
           validation_stats_a.reconstruction.total_sum);
    printf("VALIDATION_KL=%.17g\n", validation_stats_a.kl.total_sum);
    printf("OPTIMIZER_STEP_BEFORE=0\n");
    printf("OPTIMIZER_STEP_AFTER=%" PRIu64 "\n", run->optimizer.step);
    printf("PROFILE_DEPTH4_RUNTIME_VALIDATED=true\n");
    printf("224_FORWARD_PASS=true\n");
    printf("224_BACKWARD_PASS=true\n");
    printf("224_TRAIN_STEP_PASS=true\n");
    printf("OPTIMIZER_STEP_ONE=true\n");
    printf("PARAMETERS_CHANGED=true\n");
    printf("INPUT_UNCHANGED=true\n");
    printf("BATCHNORM_TRAIN_UPDATED=true\n");
    printf("VALIDATION_REPEAT_BITWISE=true\n");
    printf("VALIDATION_PERSISTENT_STATE_UNCHANGED=true\n");
    printf("VALIDATION_PASS=true\n");
    printf("CHECKPOINT_OPTIMIZER_STEP=%" PRIu64 "\n",
           loaded_optimizer.step);
    printf("CHECKPOINT_ROUNDTRIP_PASS=true\n");
    printf("NO_BENCHMARK_RUN=true\n");
    printf("NO_OFFICIAL_TRAINING_RUN=true\n");
    result = 0;

cleanup:
    if (checkpoint_created && remove(checkpoint_path) != 0) {
        fprintf(stderr, "SMOKE_ERROR=checkpoint_cleanup:%d\n", errno);
        result = 1;
    }
    if (smoke_path_exists(checkpoint_temp)) {
        fprintf(stderr, "SMOKE_ERROR=checkpoint_tmp_leftover\n");
        result = 1;
    }
    free(input_copy);
    free(validation_logits);
    free(bn_after_train);
    free(bn_before);
    free(parameter_after);
    free(parameter_before);
    free(permutation);
    vae_c_adam_free(&loaded_optimizer);
    vae_c_model_free(&loaded_model);
    return result;
}

static void smoke_usage(const char *program)
{
    fprintf(stderr,
        "Usage: %s --code-commit <40hex> --checkpoint-path <external> "
        "[repository-root]\n", program);
}

int main(int argc, char **argv)
{
    SmokeRun run;
    const char *code_commit = NULL;
    const char *checkpoint_path = NULL;
    const char *root = ".";
    int argument;
    int result;

    for (argument = 1; argument < argc; argument++) {
        if (strcmp(argv[argument], "--help") == 0) {
            smoke_usage(argv[0]);
            return 0;
        }
        if (strcmp(argv[argument], "--code-commit") == 0 &&
            argument + 1 < argc) {
            code_commit = argv[++argument];
        } else if (strcmp(argv[argument], "--checkpoint-path") == 0 &&
                   argument + 1 < argc) {
            checkpoint_path = argv[++argument];
        } else if (argv[argument][0] != '-' && strcmp(root, ".") == 0) {
            root = argv[argument];
        } else {
            smoke_usage(argv[0]);
            return 2;
        }
    }
    if (!smoke_code_commit_is_valid(code_commit) ||
        checkpoint_path == NULL || checkpoint_path[0] == '\0') {
        smoke_usage(argv[0]);
        return 2;
    }
    smoke_init(&run);
    if (smoke_load_contract(&run, root, code_commit) != 0) {
        smoke_free(&run);
        return 1;
    }
    result = smoke_run(&run, checkpoint_path);
    smoke_free(&run);
    if (result == 0) {
        printf("CODE_COMMIT=%s\n", code_commit);
        printf("MINI224_RUNTIME_SMOKE_PASS=true\n");
    }
    return result;
}
