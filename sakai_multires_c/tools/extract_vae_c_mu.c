#include "vae_c_mu_contract.h"

#include "vae_c_model_forward.h"
#include "vae_c_model_workspace.h"
#include "vae_c_sha256.h"
#include "vae_event_dataset.h"
#include "vae_saved_split.h"
#include "vae_tensor_io.h"
#include "vae_tensor_manifest.h"

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#include <windows.h>
#else
#include <dirent.h>
#include <sys/types.h>
#endif


enum {
    TINY_ROWS_PER_SPLIT = 2,
    TINY_SPLIT_COUNT = 2,
    TINY_TOTAL_ROWS = TINY_ROWS_PER_SPLIT * TINY_SPLIT_COUNT,
    TINY_ROW_IDENTITY_CAP = 2048,
    ARTIFACT_SPEC_CAP = 32768,
    ARTIFACT_CHECKSUM_CAP = 1024,
    ARTIFACT_ROW_LINE_CAP = 1024
};


#define MU_OUTPUT_PARENT_REL \
    "sakai_multires_c/results/vae_c/embeddings"
#define MU_OUTPUT_SUFFIX "_posterior_mu_central_v1"
#define MU_EXTRACTOR_SOURCE_REL \
    "sakai_multires_c/tools/extract_vae_c_mu.c"
#define MU_CONTRACT_SOURCE_REL \
    "sakai_multires_c/src/vae_c_mu_contract.c"
#define MU_CONTRACT_HEADER_REL \
    "sakai_multires_c/include/vae_c_mu_contract.h"


typedef enum {
    EXTRACTOR_MODE_NONE = 0,
    EXTRACTOR_MODE_VALIDATE_ONLY,
    EXTRACTOR_MODE_TINY_FORWARD,
    EXTRACTOR_MODE_EXTRACT,
    EXTRACTOR_MODE_VERIFY_ARTIFACT
} ExtractorMode;


typedef struct {
    char split[16];
    size_t split_pos;
    size_t manifest_index;
    int event_index;
    char event_id[GS_EVENT_MAX_ID_LEN];
} TinyRowIdentity;


typedef struct {
    GSEventManifest events;
    GSSavedSplit train;
    GSSavedSplit validation;
    GSTensorManifest tensor_manifest;
    GSTensorSpec tensor_spec;

    VAECModel model;
    VAECAdam optimizer;
    VAECCheckpointMetadata metadata;
    VAECModelWorkspace workspace;

    float *canonical_inputs;
    float *canonical_mu;
    unsigned char *parameter_snapshot;
    float *batch_norm_snapshot;
    size_t parameter_snapshot_bytes;
    size_t batch_norm_snapshot_values;
    uint64_t cache_generation_before;
    VAECAdam optimizer_before;
    TinyRowIdentity rows[TINY_TOTAL_ROWS];
} TinyForwardState;


typedef struct {
    TinyForwardState core;
    float *train_mu;
    float *validation_mu;
    size_t train_mu_values;
    size_t validation_mu_values;
    size_t producer_batch_size;
    uint64_t neural_batch_loads;
    uint64_t encoder_forward_calls;
    uint64_t mu_values_created;
    char train_input_stream_sha256[VAE_C_SHA256_HEX_SIZE];
    char validation_input_stream_sha256[VAE_C_SHA256_HEX_SIZE];
} ProductiveState;


typedef struct {
    char parent[VAE_C_MU_PATH_CAP];
    char basename[VAE_C_MU_TEXT_CAP * 2u];
    char final[VAE_C_MU_PATH_CAP];
    char partial[VAE_C_MU_PATH_CAP];
} ArtifactPaths;


typedef struct {
    char train_mu[VAE_C_SHA256_HEX_SIZE];
    char validation_mu[VAE_C_SHA256_HEX_SIZE];
    char train_rows[VAE_C_SHA256_HEX_SIZE];
    char validation_rows[VAE_C_SHA256_HEX_SIZE];
    char embedding_spec[VAE_C_SHA256_HEX_SIZE];
    char checksum[VAE_C_SHA256_HEX_SIZE];
} ArtifactHashes;


typedef struct {
    char extractor[VAE_C_SHA256_HEX_SIZE];
    char contract_source[VAE_C_SHA256_HEX_SIZE];
    char contract_header[VAE_C_SHA256_HEX_SIZE];
} ProducerSourceHashes;


static void print_usage(const char *program)
{
    fprintf(
        stderr,
        "Usage:\n"
        "  %s --validate-only --run <mini75|mini112|mini224> "
        "[repository_root]\n"
        "  %s --tiny-forward --run <mini75|mini112|mini224> "
        "--batch-size <1|2> [repository_root]\n"
        "  %s --extract --run <mini75|mini112|mini224> "
        "--batch-size <N> --extractor-code-commit <40hex> "
        "[repository_root]\n"
        "  %s --verify-artifact --run <mini75|mini112|mini224> "
        "--batch-size <N> --extractor-code-commit <40hex> "
        "[repository_root]\n"
        "The tiny forward is fixed to the first and last native train rows "
        "and the first and last native validation rows, uses CENTRAL "
        "tensors and EVAL BatchNorm, and "
        "writes no mu artifact. Productive extraction uses the canonical "
        "BEST checkpoint and publishes one VAE_C_MU_ARTIFACT_V1 directory "
        "atomically through a sibling .partial directory. Verification "
        "recomputes posterior mu read-only.\n",
        program,
        program,
        program,
        program
    );
}


static int size_multiply(size_t left, size_t right, size_t *result)
{
    if (result == NULL || (left != 0u && right > SIZE_MAX / left)) {
        return 0;
    }
    *result = left * right;
    return 1;
}


static int size_add(size_t left, size_t right, size_t *result)
{
    if (result == NULL || right > SIZE_MAX - left) {
        return 0;
    }
    *result = left + right;
    return 1;
}


static int parse_batch_size(const char *text, size_t *batch_size)
{
    char *end = NULL;
    unsigned long long value;

    if (text == NULL || batch_size == NULL || text[0] == '\0') {
        return 0;
    }
    errno = 0;
    value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value == 0ull ||
        value > (unsigned long long)SIZE_MAX) {
        return 0;
    }
    *batch_size = (size_t)value;
    return 1;
}


static int hash_bytes(
    const void *data,
    size_t size,
    char hex[VAE_C_SHA256_HEX_SIZE]
)
{
    return vae_c_sha256_digest_hex_lower(data, size, hex) ==
        VAE_C_SHA256_OK;
}


static int hash_equal(const char *left, const char *right)
{
    size_t index;

    if (left == NULL || right == NULL || strlen(left) != 64u ||
        strlen(right) != 64u) {
        return 0;
    }
    for (index = 0; index < 64u; index++) {
        char a = left[index];
        char b = right[index];

        if (a >= 'A' && a <= 'F') {
            a = (char)(a - 'A' + 'a');
        }
        if (b >= 'A' && b <= 'F') {
            b = (char)(b - 'A' + 'a');
        }
        if (a != b) {
            return 0;
        }
    }
    return 1;
}


static void tiny_forward_state_init(TinyForwardState *state)
{
    if (state != NULL) {
        memset(state, 0, sizeof(*state));
        vae_c_model_init(&state->model);
        vae_c_adam_init(&state->optimizer);
        vae_c_checkpoint_metadata_init(&state->metadata);
        vae_c_model_workspace_init(&state->workspace);
    }
}


static void tiny_forward_state_free(TinyForwardState *state)
{
    if (state == NULL) {
        return;
    }
    free(state->batch_norm_snapshot);
    free(state->parameter_snapshot);
    free(state->canonical_mu);
    free(state->canonical_inputs);
    vae_c_model_workspace_free(&state->workspace);
    vae_c_adam_free(&state->optimizer);
    vae_c_model_free(&state->model);
    gs_tensor_manifest_free(&state->tensor_manifest);
    gs_saved_split_free(&state->validation);
    gs_saved_split_free(&state->train);
    gs_event_manifest_free(&state->events);
    memset(state, 0, sizeof(*state));
}


static int copy_text(char *target, size_t capacity, const char *source)
{
    size_t length;

    if (target == NULL || capacity == 0u || source == NULL) {
        return 0;
    }
    length = strlen(source);
    if (length >= capacity) {
        return 0;
    }
    memcpy(target, source, length + 1u);
    return 1;
}


static int inputs_are_finite_float01(const float *values, size_t count)
{
    size_t index;

    if (values == NULL || count == 0u) {
        return 0;
    }
    for (index = 0; index < count; index++) {
        if (!isfinite(values[index]) ||
            values[index] < 0.0f || values[index] > 1.0f) {
            return 0;
        }
    }
    return 1;
}


static int values_are_finite(const float *values, size_t count)
{
    size_t index;

    if (values == NULL || count == 0u) {
        return 0;
    }
    for (index = 0; index < count; index++) {
        if (!isfinite(values[index])) {
            return 0;
        }
    }
    return 1;
}


static int tensor_storage_is_zero(const VAECTensor *tensor)
{
    size_t index;

    if (tensor == NULL || tensor->data == NULL || tensor->shape.count == 0u) {
        return 0;
    }
    for (index = 0; index < tensor->shape.count; index++) {
        if (tensor->data[index] != 0.0f) {
            return 0;
        }
    }
    if (tensor->has_grad) {
        if (tensor->grad == NULL) {
            return 0;
        }
        for (index = 0; index < tensor->shape.count; index++) {
            if (tensor->grad[index] != 0.0f) {
                return 0;
            }
        }
    }
    return 1;
}


static int decoder_workspace_is_zero_for(
    const VAECModel *model,
    const VAECModelWorkspace *workspace
)
{
    size_t index;

    if (model == NULL || workspace == NULL) {
        return 0;
    }
    if (!tensor_storage_is_zero(&workspace->decoder_fc_output) ||
        !tensor_storage_is_zero(&workspace->decoder_fc_batch_norm_output) ||
        !tensor_storage_is_zero(&workspace->decoder_fc_relu_output) ||
        !tensor_storage_is_zero(&workspace->logits)) {
        return 0;
    }
    for (index = 0; index < model->config.decoder_upsample_count;
         index++) {
        if (!tensor_storage_is_zero(
                &workspace->decoder_upsample_outputs[index])) {
            return 0;
        }
    }
    for (index = 0;
         index < model->config.decoder_hidden_conv_count;
         index++) {
        if (!tensor_storage_is_zero(
                &workspace->decoder_hidden_conv_outputs[index]) ||
            !tensor_storage_is_zero(
                &workspace->decoder_hidden_batch_norm_outputs[index]) ||
            !tensor_storage_is_zero(
                &workspace->decoder_hidden_relu_outputs[index])) {
            return 0;
        }
    }
    return 1;
}


static int decoder_workspace_is_zero(const TinyForwardState *state)
{
    return state != NULL && decoder_workspace_is_zero_for(
        &state->model, &state->workspace
    );
}


static int parameter_snapshot_create(TinyForwardState *state)
{
    size_t total = 0u;
    size_t offset = 0u;
    size_t index;

    if (state == NULL || state->model.parameter_count == 0u) {
        return 0;
    }
    for (index = 0; index < state->model.parameter_count; index++) {
        const VAECParameter *parameter = state->model.parameters[index];

        if (parameter == NULL || parameter->owned_block == NULL ||
            parameter->owned_bytes == 0u ||
            !size_add(total, parameter->owned_bytes, &total)) {
            return 0;
        }
    }
    state->parameter_snapshot = (unsigned char *)malloc(total);
    if (state->parameter_snapshot == NULL) {
        return 0;
    }
    for (index = 0; index < state->model.parameter_count; index++) {
        const VAECParameter *parameter = state->model.parameters[index];

        memcpy(state->parameter_snapshot + offset,
               parameter->owned_block, parameter->owned_bytes);
        offset += parameter->owned_bytes;
    }
    state->parameter_snapshot_bytes = total;
    return offset == total;
}


static int parameter_snapshot_matches(const TinyForwardState *state)
{
    size_t offset = 0u;
    size_t index;

    if (state == NULL || state->parameter_snapshot == NULL ||
        state->parameter_snapshot_bytes == 0u) {
        return 0;
    }
    for (index = 0; index < state->model.parameter_count; index++) {
        const VAECParameter *parameter = state->model.parameters[index];

        if (parameter == NULL || parameter->owned_block == NULL ||
            parameter->owned_bytes >
                state->parameter_snapshot_bytes - offset ||
            memcmp(state->parameter_snapshot + offset,
                   parameter->owned_block, parameter->owned_bytes) != 0) {
            return 0;
        }
        offset += parameter->owned_bytes;
    }
    return offset == state->parameter_snapshot_bytes;
}


static size_t collect_batch_norms(
    const VAECModel *model,
    const VAECBatchNorm *layers[
        VAE_C_MODEL_MAX_ENCODER_CONVS +
        VAE_C_MODEL_MAX_DECODER_HIDDEN_CONVS + 1u]
)
{
    size_t count = 0u;
    size_t index;

    for (index = 0; index < model->config.conv_depth; index++) {
        layers[count++] = &model->encoder_batch_norms[index];
    }
    layers[count++] = &model->decoder_fc_batch_norm;
    for (index = 0; index < model->config.decoder_hidden_conv_count;
         index++) {
        layers[count++] = &model->decoder_hidden_batch_norms[index];
    }
    return count;
}


static int batch_norm_snapshot_create(TinyForwardState *state)
{
    const VAECBatchNorm *layers[
        VAE_C_MODEL_MAX_ENCODER_CONVS +
        VAE_C_MODEL_MAX_DECODER_HIDDEN_CONVS + 1u];
    size_t layer_count;
    size_t value_count = 0u;
    size_t offset = 0u;
    size_t index;

    if (state == NULL) {
        return 0;
    }
    layer_count = collect_batch_norms(&state->model, layers);
    for (index = 0; index < layer_count; index++) {
        size_t layer_values;

        if (layers[index]->running_mean == NULL ||
            layers[index]->running_variance == NULL ||
            !size_multiply(layers[index]->num_features, 2u,
                           &layer_values) ||
            !size_add(value_count, layer_values, &value_count)) {
            return 0;
        }
    }
    state->batch_norm_snapshot = (float *)malloc(
        value_count * sizeof(float)
    );
    if (state->batch_norm_snapshot == NULL) {
        return 0;
    }
    for (index = 0; index < layer_count; index++) {
        size_t bytes = layers[index]->num_features * sizeof(float);

        memcpy(state->batch_norm_snapshot + offset,
               layers[index]->running_mean, bytes);
        offset += layers[index]->num_features;
        memcpy(state->batch_norm_snapshot + offset,
               layers[index]->running_variance, bytes);
        offset += layers[index]->num_features;
    }
    state->batch_norm_snapshot_values = value_count;
    return offset == value_count;
}


static int batch_norm_snapshot_matches(const TinyForwardState *state)
{
    const VAECBatchNorm *layers[
        VAE_C_MODEL_MAX_ENCODER_CONVS +
        VAE_C_MODEL_MAX_DECODER_HIDDEN_CONVS + 1u];
    size_t layer_count;
    size_t offset = 0u;
    size_t index;

    if (state == NULL || state->batch_norm_snapshot == NULL ||
        state->batch_norm_snapshot_values == 0u) {
        return 0;
    }
    layer_count = collect_batch_norms(&state->model, layers);
    for (index = 0; index < layer_count; index++) {
        size_t count = layers[index]->num_features;
        size_t bytes = count * sizeof(float);

        if (count > state->batch_norm_snapshot_values - offset ||
            memcmp(state->batch_norm_snapshot + offset,
                   layers[index]->running_mean, bytes) != 0) {
            return 0;
        }
        offset += count;
        if (count > state->batch_norm_snapshot_values - offset ||
            memcmp(state->batch_norm_snapshot + offset,
                   layers[index]->running_variance, bytes) != 0) {
            return 0;
        }
        offset += count;
    }
    return offset == state->batch_norm_snapshot_values;
}


static int optimizer_matches_snapshot(const TinyForwardState *state)
{
    const VAECAdam *left;
    const VAECAdam *right;

    if (state == NULL) {
        return 0;
    }
    left = &state->optimizer;
    right = &state->optimizer_before;
    return left->learning_rate == right->learning_rate &&
        left->beta1 == right->beta1 &&
        left->beta2 == right->beta2 &&
        left->adam_epsilon == right->adam_epsilon &&
        left->step == right->step &&
        left->beta1_power == right->beta1_power &&
        left->beta2_power == right->beta2_power &&
        left->configured == right->configured;
}


static int resolve_tensor_path(
    const char *repository_root,
    const char *tensor_path,
    char resolved[VAE_C_MU_PATH_CAP]
)
{
    size_t root_length;
    const char *separator;
    int written;

    if (repository_root == NULL || tensor_path == NULL ||
        resolved == NULL || tensor_path[0] == '\0') {
        return 0;
    }
    if (tensor_path[0] == '/' || tensor_path[0] == '\\' ||
        (tensor_path[1] == ':' && tensor_path[2] != '\0')) {
        return copy_text(resolved, VAE_C_MU_PATH_CAP, tensor_path);
    }
    root_length = strlen(repository_root);
    separator = root_length != 0u &&
        (repository_root[root_length - 1u] == '/' ||
         repository_root[root_length - 1u] == '\\') ? "" : "/";
    written = snprintf(
        resolved, VAE_C_MU_PATH_CAP, "%s%s%s",
        repository_root, separator, tensor_path
    );
    return written >= 0 && (size_t)written < VAE_C_MU_PATH_CAP;
}


static int load_tiny_row(
    TinyForwardState *state,
    const char *repository_root,
    const GSSavedSplit *split,
    const char *split_name,
    size_t split_pos,
    size_t row_index
)
{
    size_t manifest_index;
    const GSEventRecord *event;
    const GSTensorManifestRecord *tensor;
    float *destination;
    char tensor_path[VAE_C_MU_PATH_CAP];

    if (state == NULL || repository_root == NULL || split == NULL ||
        split_name == NULL ||
        split_pos >= split->count || row_index >= TINY_TOTAL_ROWS) {
        return 0;
    }
    manifest_index = split->indices[split_pos];
    event = gs_event_manifest_get(&state->events, manifest_index);
    tensor = gs_tensor_manifest_get(&state->tensor_manifest, manifest_index);
    if (event == NULL || tensor == NULL ||
        event->event_index != tensor->event_index ||
        strcmp(event->event_id, tensor->event_id) != 0 ||
        tensor->channels != state->tensor_spec.channels ||
        tensor->height != state->tensor_spec.height ||
        tensor->width != state->tensor_spec.width ||
        strcmp(tensor->dtype, "uint8") != 0 ||
        strcmp(tensor->layout, "channel_major") != 0 ||
        strcmp(tensor->channel_order, "0.5;1.0;2.0;4.0") != 0 ||
        !resolve_tensor_path(
            repository_root, tensor->tensor_path, tensor_path)) {
        return 0;
    }
    destination = state->canonical_inputs +
        row_index * state->tensor_spec.values_per_event;
    if (gs_tensor_read_float01(
            tensor_path, &state->tensor_spec, destination) != 0 ||
        !inputs_are_finite_float01(
            destination, state->tensor_spec.values_per_event) ||
        !copy_text(state->rows[row_index].split,
                   sizeof(state->rows[row_index].split), split_name) ||
        !copy_text(state->rows[row_index].event_id,
                   sizeof(state->rows[row_index].event_id),
                   event->event_id)) {
        return 0;
    }
    state->rows[row_index].split_pos = split_pos;
    state->rows[row_index].manifest_index = manifest_index;
    state->rows[row_index].event_index = event->event_index;
    return 1;
}


static int load_source_metadata(
    const VAECMuCanonicalContract *contract,
    TinyForwardState *state
)
{
    if (contract == NULL || state == NULL ||
        contract->profile.input_side > (size_t)INT_MAX ||
        gs_event_manifest_load(
            contract->event_manifest_path, &state->events) != 0 ||
        state->events.count != contract->event_count ||
        gs_saved_split_load(
            contract->train_population_path, "train",
            &state->events, &state->train) != 0 ||
        state->train.count != contract->train_count ||
        gs_saved_split_load(
            contract->validation_population_path, "validation",
            &state->events, &state->validation) != 0 ||
        state->validation.count != contract->validation_count ||
        state->train.count < TINY_ROWS_PER_SPLIT ||
        state->validation.count < TINY_ROWS_PER_SPLIT ||
        gs_tensor_spec_init(
            &state->tensor_spec, (int)contract->profile.input_side) != 0 ||
        gs_tensor_manifest_load(
            contract->tensor_manifest_path, &state->tensor_spec,
            &state->tensor_manifest) != 0 ||
        state->tensor_manifest.count != state->events.count) {
        return 0;
    }
    return 1;
}


static int load_tiny_data(
    const VAECMuCanonicalContract *contract,
    TinyForwardState *state
)
{
    size_t input_values;
    size_t mu_values;
    size_t split_index;
    size_t sample_index;
    const GSSavedSplit *splits[TINY_SPLIT_COUNT];
    const char *split_names[TINY_SPLIT_COUNT] = {"train", "validation"};

    if (!load_source_metadata(contract, state) ||
        !size_multiply(TINY_TOTAL_ROWS,
                       state->tensor_spec.values_per_event, &input_values) ||
        !size_multiply(TINY_TOTAL_ROWS,
                       contract->profile.latent_dim, &mu_values)) {
        return 0;
    }

    state->canonical_inputs = (float *)malloc(input_values * sizeof(float));
    state->canonical_mu = (float *)malloc(mu_values * sizeof(float));
    if (state->canonical_inputs == NULL || state->canonical_mu == NULL) {
        return 0;
    }
    splits[0] = &state->train;
    splits[1] = &state->validation;
    for (split_index = 0; split_index < TINY_SPLIT_COUNT; split_index++) {
        for (sample_index = 0;
             sample_index < TINY_ROWS_PER_SPLIT;
             sample_index++) {
            size_t split_pos = sample_index == 0u ?
                0u : splits[split_index]->count - 1u;
            size_t row_index =
                split_index * TINY_ROWS_PER_SPLIT + sample_index;

            if (!load_tiny_row(
                    state, contract->repository_root,
                    splits[split_index], split_names[split_index],
                    split_pos, row_index)) {
                return 0;
            }
        }
    }
    return 1;
}


static int load_tiny_model(
    const VAECMuCanonicalContract *contract,
    size_t batch_size,
    TinyForwardState *state
)
{
    VAECOpStatus op_status;
    VAECCheckpointStatus checkpoint_status;
    char checkpoint_sha[VAE_C_CHECKPOINT_SHA256_CAP];

    if (contract == NULL || state == NULL ||
        contract->expected_identity.max_batch_size > (uint64_t)SIZE_MAX) {
        return 0;
    }
    op_status = vae_c_model_allocate(
        &state->model, &contract->profile,
        (size_t)contract->expected_identity.max_batch_size
    );
    if (op_status == VAE_C_OP_OK) {
        op_status = vae_c_model_initialize_parameters_v1(
            &state->model,
            contract->expected_identity.initialization_seed
        );
    }
    if (op_status == VAE_C_OP_OK) {
        op_status = vae_c_adam_configure_v1(&state->optimizer);
    }
    if (op_status != VAE_C_OP_OK) {
        return 0;
    }
    checkpoint_status = vae_c_checkpoint_load(
        contract->checkpoint_path,
        &contract->expected_identity,
        &state->model,
        &state->optimizer,
        &state->metadata
    );
    if (checkpoint_status != VAE_C_CHECKPOINT_OK ||
        state->metadata.kind != contract->expected_kind ||
        state->metadata.last_completed_epoch != contract->expected_epoch ||
        !state->metadata.best_defined ||
        state->metadata.best_epoch != contract->expected_epoch ||
        state->optimizer.step != contract->expected_optimizer_step ||
        strcmp(state->model.profile_id, contract->profile.profile_id) != 0 ||
        state->model.config.latent_dim != contract->profile.latent_dim ||
        vae_c_mu_file_sha256(
            contract->checkpoint_path, checkpoint_sha) != VAE_C_MU_OK ||
        !hash_equal(checkpoint_sha, contract->checkpoint_sha256) ||
        vae_c_model_workspace_allocate(
            &state->model, batch_size, &state->workspace) != VAE_C_OP_OK ||
        !decoder_workspace_is_zero(state) ||
        !tensor_storage_is_zero(&state->workspace.epsilon) ||
        !tensor_storage_is_zero(&state->workspace.z) ||
        !parameter_snapshot_create(state) ||
        !batch_norm_snapshot_create(state)) {
        return 0;
    }
    state->optimizer_before = state->optimizer;
    state->cache_generation_before = state->model.vae_cache_generation;
    return 1;
}


static int build_row_identity_hash(
    const TinyForwardState *state,
    char hex[VAE_C_SHA256_HEX_SIZE]
)
{
    char text[TINY_ROW_IDENTITY_CAP];
    size_t used = 0u;
    size_t index;

    if (state == NULL) {
        return 0;
    }
    text[0] = '\0';
    for (index = 0; index < TINY_TOTAL_ROWS; index++) {
        int written = snprintf(
            text + used,
            sizeof(text) - used,
            "%s\t%zu\t%zu\t%d\t%s\n",
            state->rows[index].split,
            state->rows[index].split_pos,
            state->rows[index].manifest_index,
            state->rows[index].event_index,
            state->rows[index].event_id
        );

        if (written < 0 || (size_t)written >= sizeof(text) - used) {
            return 0;
        }
        used += (size_t)written;
    }
    return hash_bytes(text, used, hex);
}


static int run_encoder_batches(
    const VAECMuCanonicalContract *contract,
    size_t batch_size,
    TinyForwardState *state,
    uint64_t *forward_calls
)
{
    size_t split_index;
    size_t offset;
    size_t input_values;
    size_t input_bytes;
    size_t mu_values;

    if (contract == NULL || state == NULL || forward_calls == NULL ||
        !size_multiply(batch_size,
                       state->tensor_spec.values_per_event, &input_values) ||
        !size_multiply(input_values, sizeof(float), &input_bytes) ||
        !size_multiply(batch_size,
                       contract->profile.latent_dim, &mu_values) ||
        state->workspace.input.shape.count != input_values ||
        state->workspace.mu.shape.count != mu_values ||
        state->workspace.logvar.shape.count != mu_values) {
        return 0;
    }

    *forward_calls = UINT64_C(0);
    for (split_index = 0; split_index < TINY_SPLIT_COUNT; split_index++) {
        size_t split_row_start = split_index * TINY_ROWS_PER_SPLIT;

        for (offset = 0; offset < TINY_ROWS_PER_SPLIT;
             offset += batch_size) {
            size_t source_row = split_row_start + offset;
            const float *source_input = state->canonical_inputs +
                source_row * state->tensor_spec.values_per_event;
            float *destination_mu = state->canonical_mu +
                source_row * contract->profile.latent_dim;

            memcpy(state->workspace.input.data, source_input, input_bytes);
            if (!decoder_workspace_is_zero(state) ||
                !tensor_storage_is_zero(&state->workspace.epsilon) ||
                !tensor_storage_is_zero(&state->workspace.z) ||
                vae_c_model_encoder_forward(
                    &state->model,
                    VAE_C_BATCH_NORM_MODE_EVAL,
                    &state->workspace) != VAE_C_OP_OK) {
                return 0;
            }
            (*forward_calls)++;
            if (memcmp(state->workspace.input.data,
                       source_input, input_bytes) != 0 ||
                !values_are_finite(state->workspace.mu.data, mu_values) ||
                !values_are_finite(state->workspace.logvar.data, mu_values) ||
                !tensor_storage_is_zero(&state->workspace.epsilon) ||
                !tensor_storage_is_zero(&state->workspace.z) ||
                !decoder_workspace_is_zero(state)) {
                return 0;
            }
            memcpy(destination_mu, state->workspace.mu.data,
                   mu_values * sizeof(float));
        }
    }
    return 1;
}


static void productive_state_init(ProductiveState *state)
{
    if (state != NULL) {
        memset(state, 0, sizeof(*state));
        tiny_forward_state_init(&state->core);
    }
}


static void productive_state_free(ProductiveState *state)
{
    if (state == NULL) {
        return;
    }
    free(state->validation_mu);
    free(state->train_mu);
    state->validation_mu = NULL;
    state->train_mu = NULL;
    tiny_forward_state_free(&state->core);
    memset(state, 0, sizeof(*state));
}


static int path_join(
    const char *left,
    const char *right,
    char *output,
    size_t capacity
)
{
    size_t left_length;
    const char *separator;
    int written;

    if (left == NULL || left[0] == '\0' || right == NULL ||
        right[0] == '\0' || output == NULL || capacity == 0u) {
        return 0;
    }
    left_length = strlen(left);
    separator = left[left_length - 1u] == '/' ||
        left[left_length - 1u] == '\\' ? "" : "/";
    written = snprintf(output, capacity, "%s%s%s", left, separator, right);
    return written >= 0 && (size_t)written < capacity;
}


static int path_append(
    const char *path,
    const char *suffix,
    char *output,
    size_t capacity
)
{
    size_t path_length;
    size_t suffix_length;

    if (path == NULL || suffix == NULL || output == NULL || capacity == 0u) {
        return 0;
    }
    path_length = strlen(path);
    suffix_length = strlen(suffix);
    if (path_length > SIZE_MAX - suffix_length ||
        path_length + suffix_length >= capacity) {
        return 0;
    }
    memcpy(output, path, path_length);
    memcpy(output + path_length, suffix, suffix_length + 1u);
    return 1;
}


static int path_state(const char *path, int *is_directory)
{
    struct stat information;

    if (path == NULL || path[0] == '\0') {
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
            *is_directory = 0;
        }
        return 0;
    }
    return -1;
}


static int ensure_directory(const char *path)
{
    int is_directory = 0;
    int state = path_state(path, &is_directory);

    if (state == 1) {
        return is_directory;
    }
    if (state != 0) {
        return 0;
    }
#ifdef _WIN32
    return _mkdir(path) == 0;
#else
    return mkdir(path, 0777) == 0;
#endif
}


static int publish_directory_no_replace(
    const char *partial,
    const char *final
)
{
    int is_directory = 0;

    if (path_state(final, &is_directory) != 0) {
        return 0;
    }
#ifdef _WIN32
    return MoveFileExA(partial, final, MOVEFILE_WRITE_THROUGH) != 0;
#else
    return rename(partial, final) == 0;
#endif
}


static int build_output_basename(
    const char *run_token,
    char *basename,
    size_t capacity
)
{
    const char *value = NULL;

    if (run_token != NULL && strcmp(run_token, "mini75") == 0) {
        value = "mini75_experiment_v1_full_best_epoch47"
            MU_OUTPUT_SUFFIX;
    } else if (run_token != NULL && strcmp(run_token, "mini112") == 0) {
        value = "mini112_primary_v1_full_best_epoch97"
            MU_OUTPUT_SUFFIX;
    } else if (run_token != NULL && strcmp(run_token, "mini224") == 0) {
        value = "mini224_bridge_v1_subset1000_best_epoch99"
            MU_OUTPUT_SUFFIX;
    }
    return value != NULL && copy_text(basename, capacity, value);
}


static int build_artifact_paths(
    const VAECMuCanonicalContract *contract,
    ArtifactPaths *paths
)
{
    if (contract == NULL || paths == NULL) {
        return 0;
    }
    memset(paths, 0, sizeof(*paths));
    return build_output_basename(
               contract->run_token, paths->basename,
               sizeof(paths->basename)) &&
        path_join(contract->repository_root, MU_OUTPUT_PARENT_REL,
                  paths->parent, sizeof(paths->parent)) &&
        path_join(paths->parent, paths->basename,
                  paths->final, sizeof(paths->final)) &&
        path_append(paths->final, ".partial",
                    paths->partial, sizeof(paths->partial));
}


static int output_targets_are_absent(const ArtifactPaths *paths)
{
    int is_directory = 0;

    return paths != NULL && path_state(paths->final, &is_directory) == 0 &&
        path_state(paths->partial, &is_directory) == 0;
}


static int commit_is_valid(const char *commit)
{
    size_t index;

    if (commit == NULL || strlen(commit) != 40u) {
        return 0;
    }
    for (index = 0; index < 40u; index++) {
        const char value = commit[index];

        if (!((value >= '0' && value <= '9') ||
              (value >= 'a' && value <= 'f'))) {
            return 0;
        }
    }
    return 1;
}


static int canonical_row(
    const TinyForwardState *state,
    const GSSavedSplit *split,
    size_t split_pos,
    size_t *manifest_index,
    const GSEventRecord **event,
    const GSTensorManifestRecord **tensor
)
{
    size_t index;
    const GSEventRecord *event_record;
    const GSTensorManifestRecord *tensor_record;

    if (state == NULL || split == NULL || split_pos >= split->count ||
        manifest_index == NULL || event == NULL || tensor == NULL) {
        return 0;
    }
    index = split->indices[split_pos];
    event_record = gs_event_manifest_get(&state->events, index);
    tensor_record = gs_tensor_manifest_get(&state->tensor_manifest, index);
    if (event_record == NULL || tensor_record == NULL ||
        event_record->event_index != tensor_record->event_index ||
        strcmp(event_record->event_id, tensor_record->event_id) != 0 ||
        tensor_record->channels != state->tensor_spec.channels ||
        tensor_record->height != state->tensor_spec.height ||
        tensor_record->width != state->tensor_spec.width ||
        strcmp(tensor_record->dtype, "uint8") != 0 ||
        strcmp(tensor_record->layout, "channel_major") != 0 ||
        strcmp(tensor_record->channel_order, "0.5;1.0;2.0;4.0") != 0) {
        return 0;
    }
    *manifest_index = index;
    *event = event_record;
    *tensor = tensor_record;
    return 1;
}


static int load_population_batch(
    const VAECMuCanonicalContract *contract,
    const TinyForwardState *state,
    const GSSavedSplit *split,
    size_t first_row,
    size_t row_count,
    VAECModelWorkspace *workspace
)
{
    size_t row;
    size_t expected_values;

    if (contract == NULL || state == NULL || split == NULL ||
        workspace == NULL || row_count == 0u || first_row > split->count ||
        row_count > split->count - first_row ||
        !size_multiply(row_count, state->tensor_spec.values_per_event,
                       &expected_values) ||
        workspace->input.shape.count != expected_values) {
        return 0;
    }
    for (row = 0; row < row_count; row++) {
        size_t manifest_index;
        const GSEventRecord *event;
        const GSTensorManifestRecord *tensor;
        char tensor_path[VAE_C_MU_PATH_CAP];
        float *destination = workspace->input.data +
            row * state->tensor_spec.values_per_event;

        if (!canonical_row(
                state, split, first_row + row, &manifest_index,
                &event, &tensor) ||
            !resolve_tensor_path(
                contract->repository_root, tensor->tensor_path,
                tensor_path) ||
            gs_tensor_read_float01(
                tensor_path, &state->tensor_spec, destination) != 0 ||
            !inputs_are_finite_float01(
                destination, state->tensor_spec.values_per_event)) {
            return 0;
        }
        (void)manifest_index;
        (void)event;
    }
    return 1;
}


static int generate_split_mu(
    const VAECMuCanonicalContract *contract,
    ProductiveState *state,
    const GSSavedSplit *split,
    size_t batch_size,
    float *matrix,
    size_t matrix_values
)
{
    size_t expected_values;
    size_t first_row;
    int passed = 0;

    if (contract == NULL || state == NULL || split == NULL ||
        matrix == NULL || batch_size == 0u ||
        !size_multiply(split->count, contract->profile.latent_dim,
                       &expected_values) ||
        expected_values != matrix_values) {
        goto cleanup;
    }

    for (first_row = 0u; first_row < split->count;) {
        size_t remaining = split->count - first_row;
        size_t actual_batch = remaining < batch_size ? remaining : batch_size;
        size_t input_values;
        size_t input_bytes;
        size_t mu_values;
        size_t mu_bytes;
        char input_before[VAE_C_SHA256_HEX_SIZE];
        char input_after[VAE_C_SHA256_HEX_SIZE];
        VAECModelWorkspace *workspace = &state->core.workspace;

        if (workspace->batch_size != actual_batch) {
            vae_c_model_workspace_free(workspace);
            vae_c_model_workspace_init(workspace);
            if (vae_c_model_workspace_allocate(
                    &state->core.model, actual_batch,
                    workspace) != VAE_C_OP_OK) {
                goto cleanup;
            }
        }
        if (!size_multiply(actual_batch,
                           state->core.tensor_spec.values_per_event,
                           &input_values) ||
            !size_multiply(input_values, sizeof(float), &input_bytes) ||
            !size_multiply(actual_batch, contract->profile.latent_dim,
                           &mu_values) ||
            !size_multiply(mu_values, sizeof(float), &mu_bytes) ||
            !load_population_batch(
                contract, &state->core, split, first_row,
                actual_batch, workspace) ||
            !hash_bytes(workspace->input.data, input_bytes, input_before) ||
            !decoder_workspace_is_zero_for(
                &state->core.model, workspace) ||
            !tensor_storage_is_zero(&workspace->epsilon) ||
            !tensor_storage_is_zero(&workspace->z) ||
            vae_c_model_encoder_forward(
                &state->core.model, VAE_C_BATCH_NORM_MODE_EVAL,
                workspace) != VAE_C_OP_OK ||
            !hash_bytes(workspace->input.data, input_bytes, input_after) ||
            !hash_equal(input_before, input_after) ||
            !values_are_finite(workspace->mu.data, mu_values) ||
            !values_are_finite(workspace->logvar.data, mu_values) ||
            !tensor_storage_is_zero(&workspace->epsilon) ||
            !tensor_storage_is_zero(&workspace->z) ||
            !decoder_workspace_is_zero_for(
                &state->core.model, workspace)) {
            goto cleanup;
        }
        memcpy(
            matrix + first_row * contract->profile.latent_dim,
            workspace->mu.data,
            mu_bytes
        );
        if (state->neural_batch_loads == UINT64_MAX ||
            state->encoder_forward_calls == UINT64_MAX ||
            state->mu_values_created > UINT64_MAX - (uint64_t)mu_values) {
            goto cleanup;
        }
        state->neural_batch_loads++;
        state->encoder_forward_calls++;
        state->mu_values_created += (uint64_t)mu_values;
        first_row += actual_batch;
    }
    passed = 1;

cleanup:
    return passed;
}


static int build_population_input_stream_sha256(
    const VAECMuCanonicalContract *contract,
    const TinyForwardState *state,
    const GSSavedSplit *split,
    char output[VAE_C_SHA256_HEX_SIZE]
)
{
    char *ordered_hashes = NULL;
    size_t ordered_bytes;
    size_t split_pos;
    int passed = 0;

    if (contract == NULL || state == NULL || split == NULL ||
        output == NULL ||
        !size_multiply(split->count, 64u, &ordered_bytes)) {
        return 0;
    }
    ordered_hashes = (char *)malloc(ordered_bytes);
    if (ordered_hashes == NULL) {
        return 0;
    }
    for (split_pos = 0u; split_pos < split->count; split_pos++) {
        size_t manifest_index;
        const GSEventRecord *event;
        const GSTensorManifestRecord *tensor;
        char tensor_path[VAE_C_MU_PATH_CAP];
        char tensor_sha[VAE_C_SHA256_HEX_SIZE];

        if (!canonical_row(
                state, split, split_pos, &manifest_index, &event, &tensor) ||
            !resolve_tensor_path(
                contract->repository_root, tensor->tensor_path,
                tensor_path) ||
            vae_c_mu_file_sha256(
                tensor_path, tensor_sha) != VAE_C_MU_OK) {
            goto cleanup;
        }
        memcpy(ordered_hashes + split_pos * 64u, tensor_sha, 64u);
        (void)manifest_index;
        (void)event;
    }
    passed = hash_bytes(ordered_hashes, ordered_bytes, output);

cleanup:
    free(ordered_hashes);
    return passed;
}


static int productive_generate(
    const VAECMuCanonicalContract *contract,
    size_t batch_size,
    ProductiveState *state
)
{
    size_t train_bytes;
    size_t validation_bytes;
    uint64_t expected_mu_values;
    char checkpoint_sha[VAE_C_SHA256_HEX_SIZE];
    char train_input_after[VAE_C_SHA256_HEX_SIZE];
    char validation_input_after[VAE_C_SHA256_HEX_SIZE];

    if (contract == NULL || state == NULL || batch_size == 0u ||
        contract->expected_identity.max_batch_size > (uint64_t)SIZE_MAX ||
        batch_size > (size_t)contract->expected_identity.max_batch_size ||
        !load_source_metadata(contract, &state->core) ||
        !size_multiply(state->core.train.count,
                       contract->profile.latent_dim,
                       &state->train_mu_values) ||
        !size_multiply(state->core.validation.count,
                       contract->profile.latent_dim,
                       &state->validation_mu_values) ||
        !size_multiply(state->train_mu_values, sizeof(float),
                       &train_bytes) ||
        !size_multiply(state->validation_mu_values, sizeof(float),
                       &validation_bytes) ||
        (uint64_t)train_bytes != contract->artifact_v1.train.expected_bytes ||
        (uint64_t)validation_bytes !=
            contract->artifact_v1.validation.expected_bytes ||
        !build_population_input_stream_sha256(
            contract, &state->core, &state->core.train,
            state->train_input_stream_sha256) ||
        !build_population_input_stream_sha256(
            contract, &state->core, &state->core.validation,
            state->validation_input_stream_sha256)) {
        return 0;
    }
    state->train_mu = (float *)malloc(train_bytes);
    state->validation_mu = (float *)malloc(validation_bytes);
    if (state->train_mu == NULL || state->validation_mu == NULL ||
        !load_tiny_model(contract, batch_size, &state->core) ||
        !generate_split_mu(
            contract, state, &state->core.train, batch_size,
            state->train_mu, state->train_mu_values) ||
        !generate_split_mu(
            contract, state, &state->core.validation, batch_size,
            state->validation_mu, state->validation_mu_values) ||
        !build_population_input_stream_sha256(
            contract, &state->core, &state->core.train,
            train_input_after) ||
        !build_population_input_stream_sha256(
            contract, &state->core, &state->core.validation,
            validation_input_after) ||
        !hash_equal(
            state->train_input_stream_sha256, train_input_after) ||
        !hash_equal(
            state->validation_input_stream_sha256,
            validation_input_after) ||
        !parameter_snapshot_matches(&state->core) ||
        !batch_norm_snapshot_matches(&state->core) ||
        !optimizer_matches_snapshot(&state->core) ||
        state->core.model.vae_cache_generation <
            state->core.cache_generation_before ||
        state->core.model.vae_cache_generation -
            state->core.cache_generation_before !=
                state->encoder_forward_calls ||
        !values_are_finite(state->train_mu, state->train_mu_values) ||
        !values_are_finite(
            state->validation_mu, state->validation_mu_values) ||
        vae_c_mu_file_sha256(
            contract->checkpoint_path,
            checkpoint_sha) != VAE_C_MU_OK ||
        !hash_equal(checkpoint_sha, contract->checkpoint_sha256)) {
        return 0;
    }
    expected_mu_values = contract->train_count * contract->profile.latent_dim;
    if (contract->validation_count >
            (UINT64_MAX - expected_mu_values) /
                contract->profile.latent_dim) {
        return 0;
    }
    expected_mu_values +=
        contract->validation_count * contract->profile.latent_dim;
    state->producer_batch_size = batch_size;
    return state->mu_values_created == expected_mu_values;
}


static int append_format(
    char *buffer,
    size_t capacity,
    size_t *used,
    const char *format,
    ...
)
{
    va_list arguments;
    int written;

    if (buffer == NULL || capacity == 0u || used == NULL ||
        *used >= capacity || format == NULL) {
        return 0;
    }
    va_start(arguments, format);
    written = vsnprintf(
        buffer + *used, capacity - *used, format, arguments
    );
    va_end(arguments);
    if (written < 0 || (size_t)written >= capacity - *used) {
        return 0;
    }
    *used += (size_t)written;
    return 1;
}


static int repository_relative_path(
    const char *repository_root,
    const char *absolute_path,
    char *relative,
    size_t capacity
)
{
    size_t root_length;
    const char *source;
    size_t index;

    if (repository_root == NULL || absolute_path == NULL ||
        relative == NULL || capacity == 0u) {
        return 0;
    }
    root_length = strlen(repository_root);
    if (strncmp(repository_root, absolute_path, root_length) != 0) {
        return 0;
    }
    source = absolute_path + root_length;
    if (*source == '/' || *source == '\\') {
        source++;
    } else if (*source != '\0') {
        return 0;
    }
    if (strlen(source) >= capacity) {
        return 0;
    }
    for (index = 0u; source[index] != '\0'; index++) {
        relative[index] = source[index] == '\\' ? '/' : source[index];
    }
    relative[index] = '\0';
    return 1;
}


static int file_size_u64(const char *path, uint64_t *size)
{
    struct stat information;

    if (path == NULL || size == NULL || stat(path, &information) != 0 ||
        information.st_size < 0) {
        return 0;
    }
#ifdef _WIN32
    if ((information.st_mode & _S_IFDIR) != 0) {
#else
    if (S_ISDIR(information.st_mode)) {
#endif
        return 0;
    }
    *size = (uint64_t)information.st_size;
    return 1;
}


static int finish_write(FILE *file)
{
    int failed;

    if (file == NULL) {
        return 0;
    }
    failed = fflush(file) != 0 || ferror(file);
    if (fclose(file) != 0) {
        failed = 1;
    }
    return !failed;
}


static int write_file_bytes(
    const char *directory,
    const char *filename,
    const void *data,
    size_t size
)
{
    char path[VAE_C_MU_PATH_CAP];
    FILE *file;

    if (directory == NULL || filename == NULL ||
        (size != 0u && data == NULL) ||
        !path_join(directory, filename, path, sizeof(path))) {
        return 0;
    }
    file = fopen(path, "wb");
    if (file == NULL) {
        return 0;
    }
    if (size != 0u && fwrite(data, 1u, size, file) != size) {
        (void)fclose(file);
        return 0;
    }
    return finish_write(file);
}


static int csv_value_is_safe(const char *value)
{
    size_t index;

    if (value == NULL || value[0] == '\0') {
        return 0;
    }
    for (index = 0u; value[index] != '\0'; index++) {
        if (value[index] == ',' || value[index] == '\r' ||
            value[index] == '\n') {
            return 0;
        }
    }
    return 1;
}


static int write_row_map(
    const char *directory,
    const char *filename,
    const char *split_name,
    const TinyForwardState *state,
    const GSSavedSplit *split
)
{
    char path[VAE_C_MU_PATH_CAP];
    FILE *file;
    size_t split_pos;
    int failed = 0;

    if (directory == NULL || filename == NULL || split_name == NULL ||
        state == NULL || split == NULL ||
        !csv_value_is_safe(split_name) ||
        !path_join(directory, filename, path, sizeof(path))) {
        return 0;
    }
    file = fopen(path, "wb");
    if (file == NULL) {
        return 0;
    }
    if (fputs(
            "row_index,split_pos,manifest_index,event_index,event_id,split\n",
            file) == EOF) {
        failed = 1;
    }
    for (split_pos = 0u; !failed && split_pos < split->count; split_pos++) {
        size_t manifest_index;
        const GSEventRecord *event;
        const GSTensorManifestRecord *tensor;

        if (!canonical_row(
                state, split, split_pos, &manifest_index, &event, &tensor) ||
            !csv_value_is_safe(event->event_id) ||
            fprintf(
                file, "%zu,%zu,%zu,%d,%s,%s\n",
                split_pos, split_pos, manifest_index,
                event->event_index, event->event_id, split_name) < 0) {
            failed = 1;
        }
        (void)tensor;
    }
    if (failed) {
        (void)fclose(file);
        return 0;
    }
    return finish_write(file);
}


static int hash_artifact_file(
    const char *directory,
    const char *filename,
    char output[VAE_C_SHA256_HEX_SIZE]
)
{
    char path[VAE_C_MU_PATH_CAP];

    return path_join(directory, filename, path, sizeof(path)) &&
        vae_c_mu_file_sha256(path, output) == VAE_C_MU_OK;
}


static uint64_t forward_count(uint64_t rows, uint64_t batch_size)
{
    return batch_size == 0u ? UINT64_C(0) :
        rows / batch_size + (rows % batch_size != 0u ? UINT64_C(1) :
                              UINT64_C(0));
}


static int build_embedding_spec(
    const VAECMuCanonicalContract *contract,
    const ArtifactPaths *paths,
    const ProductiveState *state,
    const ArtifactHashes *hashes,
    const char *extractor_code_commit,
    const ProducerSourceHashes *producer_sources,
    char *spec,
    size_t capacity,
    size_t *spec_size
)
{
    const VAECMuArtifactContractV1 *artifact;
    char catalog_rel[VAE_C_MU_PATH_CAP];
    char event_manifest_rel[VAE_C_MU_PATH_CAP];
    char tensor_manifest_rel[VAE_C_MU_PATH_CAP];
    char official_train_split_rel[VAE_C_MU_PATH_CAP];
    char official_validation_split_rel[VAE_C_MU_PATH_CAP];
    char train_population_rel[VAE_C_MU_PATH_CAP];
    char validation_population_rel[VAE_C_MU_PATH_CAP];
    char run_config_rel[VAE_C_MU_PATH_CAP];
    char checkpoint_rel[VAE_C_MU_PATH_CAP];
    size_t used = 0u;
    uint64_t producer_batch;

    if (contract == NULL || paths == NULL || state == NULL ||
        hashes == NULL || !commit_is_valid(extractor_code_commit) ||
        producer_sources == NULL ||
        strlen(producer_sources->extractor) != 64u ||
        strlen(producer_sources->contract_source) != 64u ||
        strlen(producer_sources->contract_header) != 64u || spec == NULL ||
        capacity == 0u || spec_size == NULL ||
        !repository_relative_path(
            contract->repository_root, contract->catalog_path,
            catalog_rel, sizeof(catalog_rel)) ||
        !repository_relative_path(
            contract->repository_root, contract->event_manifest_path,
            event_manifest_rel, sizeof(event_manifest_rel)) ||
        !repository_relative_path(
            contract->repository_root, contract->tensor_manifest_path,
            tensor_manifest_rel, sizeof(tensor_manifest_rel)) ||
        !repository_relative_path(
            contract->repository_root,
            contract->official_train_split_path,
            official_train_split_rel, sizeof(official_train_split_rel)) ||
        !repository_relative_path(
            contract->repository_root,
            contract->official_validation_split_path,
            official_validation_split_rel,
            sizeof(official_validation_split_rel)) ||
        !repository_relative_path(
            contract->repository_root, contract->train_population_path,
            train_population_rel, sizeof(train_population_rel)) ||
        !repository_relative_path(
            contract->repository_root,
            contract->validation_population_path,
            validation_population_rel, sizeof(validation_population_rel)) ||
        !repository_relative_path(
            contract->repository_root, contract->run_config_path,
            run_config_rel, sizeof(run_config_rel)) ||
        !repository_relative_path(
            contract->repository_root, contract->checkpoint_path,
            checkpoint_rel, sizeof(checkpoint_rel))) {
        return 0;
    }
    artifact = &contract->artifact_v1;
    producer_batch = contract->expected_identity.max_batch_size;
    spec[0] = '\0';
    if (!append_format(spec, capacity, &used, "format=%s\n",
                       artifact->format) ||
        !append_format(spec, capacity, &used,
                       "representation=%s\nview=%s\n",
                       artifact->representation, artifact->view) ||
        !append_format(spec, capacity, &used,
                       "profile_id=%s\nlatent_dim=%" PRIu64 "\n",
                       artifact->profile_id, artifact->latent_dim) ||
        !append_format(spec, capacity, &used,
                       "profile_config_sha256=%s\nrun_id=%s\n",
                       artifact->profile_config_sha256,
                       artifact->run_id) ||
        !append_format(spec, capacity, &used,
                       "run_config_path=%s\nrun_config_sha256=%s\n",
                       run_config_rel, artifact->run_config_sha256) ||
        !append_format(spec, capacity, &used,
                       "checkpoint_path=%s\ncheckpoint_kind=%s\n",
                       checkpoint_rel, artifact->checkpoint_kind) ||
        !append_format(spec, capacity, &used,
                       "checkpoint_epoch=%" PRIu64 "\n"
                       "checkpoint_optimizer_step=%" PRIu64 "\n",
                       artifact->checkpoint_epoch,
                       artifact->checkpoint_optimizer_step) ||
        !append_format(spec, capacity, &used,
                       "checkpoint_sha256=%s\ncheckpoint_code_commit=%s\n",
                       artifact->checkpoint_sha256,
                       artifact->checkpoint_code_commit) ||
        !append_format(spec, capacity, &used,
                       "extractor_code_commit=%s\n"
                       "extractor_source_path=%s\n"
                       "extractor_source_sha256=%s\n",
                       extractor_code_commit, MU_EXTRACTOR_SOURCE_REL,
                       producer_sources->extractor) ||
        !append_format(spec, capacity, &used,
                       "mu_contract_source_path=%s\n"
                       "mu_contract_source_sha256=%s\n"
                       "mu_contract_header_path=%s\n"
                       "mu_contract_header_sha256=%s\n",
                       MU_CONTRACT_SOURCE_REL,
                       producer_sources->contract_source,
                       MU_CONTRACT_HEADER_REL,
                       producer_sources->contract_header) ||
        !append_format(spec, capacity, &used,
                       "catalog_path=%s\ncatalog_sha256=%s\n",
                       catalog_rel, artifact->catalog_sha256) ||
        !append_format(spec, capacity, &used,
                       "event_manifest_path=%s\n"
                       "event_manifest_sha256=%s\n",
                       event_manifest_rel,
                       artifact->event_manifest_sha256) ||
        !append_format(spec, capacity, &used,
                       "tensor_manifest_path=%s\n"
                       "tensor_manifest_sha256=%s\n",
                       tensor_manifest_rel,
                       artifact->tensor_manifest_sha256) ||
        !append_format(spec, capacity, &used,
                       "train_population_path=%s\n"
                       "validation_population_path=%s\n",
                       train_population_rel, validation_population_rel) ||
        !append_format(spec, capacity, &used,
                       "official_train_split_path=%s\n"
                       "official_validation_split_path=%s\n"
                       "population_kind=%s\n",
                       official_train_split_rel,
                       official_validation_split_rel,
                       contract->uses_subset ? "canonical_subset" :
                                               "full_split") ||
        !append_format(spec, capacity, &used,
                       "train_split_sha256=%s\n"
                       "validation_split_sha256=%s\n",
                       artifact->train_split_sha256,
                       artifact->validation_split_sha256) ||
        !append_format(spec, capacity, &used,
                       "train_subset_sha256=%s\n"
                       "validation_subset_sha256=%s\n",
                       artifact->train_subset_sha256,
                       artifact->validation_subset_sha256) ||
        !append_format(spec, capacity, &used,
                       "population_identity=%s\n"
                       "checkpoint_dataset_id=%s\n",
                       artifact->population_identity,
                       artifact->checkpoint_dataset_id) ||
        !append_format(spec, capacity, &used,
                       "checkpoint_train_selection_id=%s\n"
                       "checkpoint_validation_selection_id=%s\n",
                       artifact->checkpoint_train_selection_id,
                       artifact->checkpoint_validation_selection_id) ||
        !append_format(spec, capacity, &used,
                       "encoder_only=true\nbatch_norm_mode=%s\n",
                       artifact->batch_norm_mode) ||
        !append_format(spec, capacity, &used,
                       "epsilon_used=false\nlatent_sampled=false\n"
                       "decoder_executed=false\nbackward_executed=false\n"
                       "adam_step_executed=false\n") ||
        !append_format(spec, capacity, &used,
                       "labels_used=false\n"
                       "label_dependent_selections=false\n"
                       "label_values_exported=false\n") ||
        !append_format(spec, capacity, &used,
                       "generic_manifest_parser_may_store_label_metadata=true\n") ||
        !append_format(spec, capacity, &used,
                       "dtype=%s\nendian=%s\nlayout=%s\n",
                       artifact->dtype, artifact->endian,
                       artifact->layout) ||
        !append_format(spec, capacity, &used,
                       "input_normalization=%s\nlatent_scaling=%s\n",
                       artifact->input_normalization,
                       artifact->latent_scaling) ||
        !append_format(spec, capacity, &used,
                       "latent_normalization=none\npca=false\n") ||
        !append_format(spec, capacity, &used,
                       "row_order=%s\nrow_map_schema=%s\n",
                       artifact->row_order, artifact->row_map_schema) ||
        !append_format(spec, capacity, &used,
                       "producer_batch_size=%" PRIu64 "\n",
                       producer_batch) ||
        !append_format(spec, capacity, &used,
                       "producer_train_forward_calls=%" PRIu64 "\n"
                       "producer_validation_forward_calls=%" PRIu64 "\n",
                       forward_count(artifact->train.rows, producer_batch),
                       forward_count(
                           artifact->validation.rows, producer_batch)) ||
        !append_format(spec, capacity, &used,
                       "input_stream_identity_method="
                       "sha256_of_native_order_tensor_sha256_hex_concat_v1\n") ||
        !append_format(spec, capacity, &used,
                       "train_input_stream_sha256=%s\n"
                       "validation_input_stream_sha256=%s\n",
                       state->train_input_stream_sha256,
                       state->validation_input_stream_sha256) ||
        !append_format(spec, capacity, &used,
                       "artifact_directory=%s\n",
                       paths->basename) ||
        !append_format(spec, capacity, &used,
                       "train_matrix_file=%s\ntrain_rows_file=%s\n",
                       artifact->train.matrix_file,
                       artifact->train.row_map_file) ||
        !append_format(spec, capacity, &used,
                       "train_rows=%" PRIu64 "\ntrain_cols=%" PRIu64 "\n"
                       "train_expected_bytes=%" PRIu64 "\n",
                       artifact->train.rows, artifact->train.cols,
                       artifact->train.expected_bytes) ||
        !append_format(spec, capacity, &used,
                       "train_matrix_sha256=%s\ntrain_rows_sha256=%s\n",
                       hashes->train_mu, hashes->train_rows) ||
        !append_format(spec, capacity, &used,
                       "validation_matrix_file=%s\n"
                       "validation_rows_file=%s\n",
                       artifact->validation.matrix_file,
                       artifact->validation.row_map_file) ||
        !append_format(spec, capacity, &used,
                       "validation_rows=%" PRIu64 "\n"
                       "validation_cols=%" PRIu64 "\n"
                       "validation_expected_bytes=%" PRIu64 "\n",
                       artifact->validation.rows,
                       artifact->validation.cols,
                       artifact->validation.expected_bytes) ||
        !append_format(spec, capacity, &used,
                       "validation_matrix_sha256=%s\n"
                       "validation_rows_sha256=%s\n",
                       hashes->validation_mu,
                       hashes->validation_rows) ||
        !append_format(spec, capacity, &used,
                       "checksum_file=%s\n"
                       "checksum_manifest_excludes_self=true\n",
                       artifact->checksum_file) ||
        !append_format(spec, capacity, &used,
                       "parameters_unchanged=true\n"
                       "batch_norm_running_stats_unchanged=true\n"
                       "optimizer_state_unchanged=true\n"
                       "input_batches_unchanged=true\n")) {
        return 0;
    }
    *spec_size = used;
    return 1;
}


static int build_checksum_manifest(
    const VAECMuCanonicalContract *contract,
    const ArtifactHashes *hashes,
    char *text,
    size_t capacity,
    size_t *text_size
)
{
    size_t used = 0u;
    const VAECMuArtifactContractV1 *artifact;

    if (contract == NULL || hashes == NULL || text == NULL ||
        capacity == 0u || text_size == NULL) {
        return 0;
    }
    artifact = &contract->artifact_v1;
    text[0] = '\0';
    if (!append_format(text, capacity, &used, "%s  %s\n",
                       hashes->embedding_spec,
                       artifact->embedding_spec_file) ||
        !append_format(text, capacity, &used, "%s  %s\n",
                       hashes->train_mu, artifact->train.matrix_file) ||
        !append_format(text, capacity, &used, "%s  %s\n",
                       hashes->train_rows, artifact->train.row_map_file) ||
        !append_format(text, capacity, &used, "%s  %s\n",
                       hashes->validation_mu,
                       artifact->validation.matrix_file) ||
        !append_format(text, capacity, &used, "%s  %s\n",
                       hashes->validation_rows,
                       artifact->validation.row_map_file)) {
        return 0;
    }
    *text_size = used;
    return 1;
}


static int file_equals_bytes(
    const char *path,
    const void *expected,
    size_t expected_size
)
{
    const unsigned char *bytes = (const unsigned char *)expected;
    unsigned char buffer[65536];
    size_t offset = 0u;
    FILE *file;
    uint64_t actual_size;
    int passed = 0;

    if (path == NULL || (expected_size != 0u && expected == NULL) ||
        !file_size_u64(path, &actual_size) ||
        actual_size != (uint64_t)expected_size) {
        return 0;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        return 0;
    }
    while (offset < expected_size) {
        size_t remaining = expected_size - offset;
        size_t chunk = remaining < sizeof(buffer) ? remaining : sizeof(buffer);

        if (fread(buffer, 1u, chunk, file) != chunk ||
            memcmp(buffer, bytes + offset, chunk) != 0) {
            goto cleanup;
        }
        offset += chunk;
    }
    if (fgetc(file) != EOF || ferror(file)) {
        goto cleanup;
    }
    passed = 1;

cleanup:
    if (fclose(file) != 0) {
        passed = 0;
    }
    return passed;
}


static int artifact_name_index(
    const VAECMuCanonicalContract *contract,
    const char *name
)
{
    const VAECMuArtifactContractV1 *artifact;
    const char *names[6];
    size_t index;

    if (contract == NULL || name == NULL) {
        return -1;
    }
    artifact = &contract->artifact_v1;
    names[0] = artifact->train.matrix_file;
    names[1] = artifact->validation.matrix_file;
    names[2] = artifact->train.row_map_file;
    names[3] = artifact->validation.row_map_file;
    names[4] = artifact->embedding_spec_file;
    names[5] = artifact->checksum_file;
    for (index = 0u; index < 6u; index++) {
        if (strcmp(name, names[index]) == 0) {
            return (int)index;
        }
    }
    return -1;
}


static int directory_has_exact_artifact_inventory(
    const char *directory,
    const VAECMuCanonicalContract *contract
)
{
    int seen[6] = {0, 0, 0, 0, 0, 0};
    size_t count = 0u;

    if (directory == NULL || contract == NULL) {
        return 0;
    }
#ifdef _WIN32
    {
        char pattern[VAE_C_MU_PATH_CAP];
        struct _finddata_t entry;
        intptr_t handle;

        if (!path_join(directory, "*", pattern, sizeof(pattern))) {
            return 0;
        }
        handle = _findfirst(pattern, &entry);
        if (handle == (intptr_t)-1) {
            return 0;
        }
        do {
            int index;

            if (strcmp(entry.name, ".") == 0 ||
                strcmp(entry.name, "..") == 0) {
                continue;
            }
            index = artifact_name_index(contract, entry.name);
            if (index < 0 || seen[index] ||
                (entry.attrib & _A_SUBDIR) != 0) {
                (void)_findclose(handle);
                return 0;
            }
            seen[index] = 1;
            count++;
        } while (_findnext(handle, &entry) == 0);
        if (_findclose(handle) != 0) {
            return 0;
        }
    }
#else
    {
        DIR *stream = opendir(directory);
        struct dirent *entry;

        if (stream == NULL) {
            return 0;
        }
        errno = 0;
        while ((entry = readdir(stream)) != NULL) {
            int index;

            if (strcmp(entry->d_name, ".") == 0 ||
                strcmp(entry->d_name, "..") == 0) {
                continue;
            }
            index = artifact_name_index(contract, entry->d_name);
            if (index < 0 || seen[index]) {
                (void)closedir(stream);
                return 0;
            }
            seen[index] = 1;
            count++;
        }
        if (errno != 0 || closedir(stream) != 0) {
            return 0;
        }
    }
#endif
    return count == 6u && seen[0] && seen[1] && seen[2] &&
        seen[3] && seen[4] && seen[5];
}


static int verify_row_map(
    const char *directory,
    const char *filename,
    const char *split_name,
    const TinyForwardState *state,
    const GSSavedSplit *split
)
{
    char path[VAE_C_MU_PATH_CAP];
    char actual[ARTIFACT_ROW_LINE_CAP];
    char expected[ARTIFACT_ROW_LINE_CAP];
    FILE *file;
    size_t split_pos;
    int passed = 0;

    if (directory == NULL || filename == NULL || split_name == NULL ||
        state == NULL || split == NULL ||
        !path_join(directory, filename, path, sizeof(path))) {
        return 0;
    }
    file = fopen(path, "rb");
    if (file == NULL || fgets(actual, sizeof(actual), file) == NULL ||
        strcmp(actual,
               "row_index,split_pos,manifest_index,event_index,event_id,split\n")
            != 0) {
        if (file != NULL) {
            (void)fclose(file);
        }
        return 0;
    }
    for (split_pos = 0u; split_pos < split->count; split_pos++) {
        size_t manifest_index;
        const GSEventRecord *event;
        const GSTensorManifestRecord *tensor;
        int written;

        if (!canonical_row(
                state, split, split_pos, &manifest_index, &event, &tensor)) {
            goto cleanup;
        }
        written = snprintf(
            expected, sizeof(expected), "%zu,%zu,%zu,%d,%s,%s\n",
            split_pos, split_pos, manifest_index, event->event_index,
            event->event_id, split_name
        );
        if (written < 0 || (size_t)written >= sizeof(expected) ||
            fgets(actual, sizeof(actual), file) == NULL ||
            strcmp(actual, expected) != 0) {
            goto cleanup;
        }
        (void)tensor;
    }
    if (fgetc(file) != EOF || ferror(file)) {
        goto cleanup;
    }
    passed = 1;

cleanup:
    if (fclose(file) != 0) {
        passed = 0;
    }
    return passed;
}


static int collect_artifact_hashes(
    const char *directory,
    const VAECMuCanonicalContract *contract,
    ArtifactHashes *hashes
)
{
    const VAECMuArtifactContractV1 *artifact;

    if (directory == NULL || contract == NULL || hashes == NULL) {
        return 0;
    }
    artifact = &contract->artifact_v1;
    memset(hashes, 0, sizeof(*hashes));
    return hash_artifact_file(
               directory, artifact->train.matrix_file,
               hashes->train_mu) &&
        hash_artifact_file(
               directory, artifact->validation.matrix_file,
               hashes->validation_mu) &&
        hash_artifact_file(
               directory, artifact->train.row_map_file,
               hashes->train_rows) &&
        hash_artifact_file(
               directory, artifact->validation.row_map_file,
               hashes->validation_rows) &&
        hash_artifact_file(
               directory, artifact->embedding_spec_file,
               hashes->embedding_spec) &&
        hash_artifact_file(
               directory, artifact->checksum_file,
               hashes->checksum);
}


static int verify_artifact_payload(
    const char *directory,
    const VAECMuCanonicalContract *contract,
    const ArtifactPaths *paths,
    const ProductiveState *state,
    const char *extractor_code_commit,
    const ProducerSourceHashes *producer_sources,
    ArtifactHashes *verified_hashes
)
{
    const VAECMuArtifactContractV1 *artifact;
    ArtifactHashes hashes;
    char path[VAE_C_MU_PATH_CAP];
    char expected_spec[ARTIFACT_SPEC_CAP];
    char expected_checksums[ARTIFACT_CHECKSUM_CAP];
    size_t spec_size;
    size_t checksum_size;
    size_t train_bytes;
    size_t validation_bytes;
    int is_directory = 0;

    if (directory == NULL || contract == NULL || paths == NULL ||
        state == NULL || verified_hashes == NULL ||
        path_state(directory, &is_directory) != 1 || !is_directory ||
        !directory_has_exact_artifact_inventory(directory, contract) ||
        !size_multiply(state->train_mu_values, sizeof(float),
                       &train_bytes) ||
        !size_multiply(state->validation_mu_values, sizeof(float),
                       &validation_bytes)) {
        return 0;
    }
    artifact = &contract->artifact_v1;
    if (!path_join(directory, artifact->train.matrix_file,
                   path, sizeof(path)) ||
        !file_equals_bytes(path, state->train_mu, train_bytes) ||
        !path_join(directory, artifact->validation.matrix_file,
                   path, sizeof(path)) ||
        !file_equals_bytes(path, state->validation_mu, validation_bytes) ||
        !verify_row_map(
            directory, artifact->train.row_map_file, "train",
            &state->core, &state->core.train) ||
        !verify_row_map(
            directory, artifact->validation.row_map_file, "validation",
            &state->core, &state->core.validation) ||
        !collect_artifact_hashes(directory, contract, &hashes) ||
        !build_embedding_spec(
            contract, paths, state, &hashes, extractor_code_commit,
            producer_sources, expected_spec,
            sizeof(expected_spec), &spec_size) ||
        !path_join(directory, artifact->embedding_spec_file,
                   path, sizeof(path)) ||
        !file_equals_bytes(path, expected_spec, spec_size) ||
        !build_checksum_manifest(
            contract, &hashes, expected_checksums,
            sizeof(expected_checksums), &checksum_size) ||
        !path_join(directory, artifact->checksum_file,
                   path, sizeof(path)) ||
        !file_equals_bytes(path, expected_checksums, checksum_size)) {
        return 0;
    }
    *verified_hashes = hashes;
    return 1;
}


static int write_artifact_payload(
    const VAECMuCanonicalContract *contract,
    const ArtifactPaths *paths,
    const ProductiveState *state,
    const char *extractor_code_commit,
    const ProducerSourceHashes *producer_sources,
    ArtifactHashes *verified_hashes
)
{
    const VAECMuArtifactContractV1 *artifact;
    ArtifactHashes hashes;
    char spec[ARTIFACT_SPEC_CAP];
    char checksums[ARTIFACT_CHECKSUM_CAP];
    size_t train_bytes;
    size_t validation_bytes;
    size_t spec_size;
    size_t checksum_size;
    int parent_is_directory = 0;
    int final_is_directory = 0;
    int partial_is_directory = 0;

    if (contract == NULL || paths == NULL || state == NULL ||
        verified_hashes == NULL || !output_targets_are_absent(paths) ||
        CHAR_BIT != 8 ||
        !size_multiply(state->train_mu_values, sizeof(float),
                       &train_bytes) ||
        !size_multiply(state->validation_mu_values, sizeof(float),
                       &validation_bytes) ||
        (uint64_t)train_bytes != contract->artifact_v1.train.expected_bytes ||
        (uint64_t)validation_bytes !=
            contract->artifact_v1.validation.expected_bytes) {
        return 0;
    }
    artifact = &contract->artifact_v1;
    if (!ensure_directory(paths->parent) ||
        path_state(paths->parent, &parent_is_directory) != 1 ||
        !parent_is_directory || !output_targets_are_absent(paths)) {
        return 0;
    }
#ifdef _WIN32
    if (_mkdir(paths->partial) != 0) {
#else
    if (mkdir(paths->partial, 0777) != 0) {
#endif
        return 0;
    }
    if (!write_file_bytes(
            paths->partial, artifact->train.matrix_file,
            state->train_mu, train_bytes) ||
        !write_file_bytes(
            paths->partial, artifact->validation.matrix_file,
            state->validation_mu, validation_bytes) ||
        !write_row_map(
            paths->partial, artifact->train.row_map_file, "train",
            &state->core, &state->core.train) ||
        !write_row_map(
            paths->partial, artifact->validation.row_map_file,
            "validation", &state->core, &state->core.validation) ||
        !hash_artifact_file(
            paths->partial, artifact->train.matrix_file,
            hashes.train_mu) ||
        !hash_artifact_file(
            paths->partial, artifact->validation.matrix_file,
            hashes.validation_mu) ||
        !hash_artifact_file(
            paths->partial, artifact->train.row_map_file,
            hashes.train_rows) ||
        !hash_artifact_file(
            paths->partial, artifact->validation.row_map_file,
            hashes.validation_rows) ||
        !build_embedding_spec(
            contract, paths, state, &hashes, extractor_code_commit,
            producer_sources, spec, sizeof(spec), &spec_size) ||
        !write_file_bytes(
            paths->partial, artifact->embedding_spec_file,
            spec, spec_size) ||
        !hash_artifact_file(
            paths->partial, artifact->embedding_spec_file,
            hashes.embedding_spec) ||
        !build_checksum_manifest(
            contract, &hashes, checksums, sizeof(checksums),
            &checksum_size) ||
        !write_file_bytes(
            paths->partial, artifact->checksum_file,
            checksums, checksum_size) ||
        !hash_artifact_file(
            paths->partial, artifact->checksum_file,
            hashes.checksum) ||
        !verify_artifact_payload(
            paths->partial, contract, paths, state,
            extractor_code_commit, producer_sources,
            &hashes) ||
        path_state(paths->final, &final_is_directory) != 0 ||
        path_state(paths->partial, &partial_is_directory) != 1 ||
        !partial_is_directory ||
        !publish_directory_no_replace(paths->partial, paths->final) ||
        !verify_artifact_payload(
            paths->final, contract, paths, state,
            extractor_code_commit, producer_sources,
            &hashes)) {
        return 0;
    }
    *verified_hashes = hashes;
    return 1;
}


static void print_productive_result(
    int verify_only,
    const VAECMuCanonicalContract *contract,
    const ArtifactPaths *paths,
    const ProductiveState *state,
    const ArtifactHashes *hashes,
    const char *extractor_code_commit,
    const ProducerSourceHashes *producer_sources
)
{
    printf("PRODUCTIVE_EXTRACTION=%s\n", verify_only ? "false" : "true");
    printf("VERIFY_ARTIFACT=%s\n", verify_only ? "true" : "false");
    printf("ARTIFACT_FORMAT=%s\n", contract->artifact_v1.format);
    printf("RUN=%s\n", contract->run_token);
    printf("PROFILE=%s\n", contract->profile.profile_id);
    printf("RUN_ID=%s\n", contract->run_id);
    printf("VIEW=central\nREPRESENTATION=posterior_mu\n");
    printf("CHECKPOINT_KIND=BEST\n");
    printf("CHECKPOINT_EPOCH=%" PRIu64 "\n", contract->expected_epoch);
    printf("CHECKPOINT_OPTIMIZER_STEP=%" PRIu64 "\n",
           contract->expected_optimizer_step);
    printf("CHECKPOINT_SHA256=%s\n", contract->checkpoint_sha256);
    printf("CHECKPOINT_CODE_COMMIT=%s\n",
           contract->expected_identity.code_commit);
    printf("EXTRACTOR_CODE_COMMIT=%s\n", extractor_code_commit);
    printf("EXTRACTOR_SOURCE_SHA256=%s\n", producer_sources->extractor);
    printf("MU_CONTRACT_SOURCE_SHA256=%s\n",
           producer_sources->contract_source);
    printf("MU_CONTRACT_HEADER_SHA256=%s\n",
           producer_sources->contract_header);
    printf("BATCH_SIZE=%zu\n", state->producer_batch_size);
    printf("TRAIN_ROWS=%" PRIu64 "\n", contract->train_count);
    printf("VALIDATION_ROWS=%" PRIu64 "\n", contract->validation_count);
    printf("LATENT_DIM=%zu\n", contract->profile.latent_dim);
    printf("TRAIN_EXPECTED_BYTES=%" PRIu64 "\n",
           contract->artifact_v1.train.expected_bytes);
    printf("VALIDATION_EXPECTED_BYTES=%" PRIu64 "\n",
           contract->artifact_v1.validation.expected_bytes);
    printf("TRAIN_INPUT_STREAM_SHA256=%s\n",
           state->train_input_stream_sha256);
    printf("VALIDATION_INPUT_STREAM_SHA256=%s\n",
           state->validation_input_stream_sha256);
    printf("TRAIN_MU_SHA256=%s\n", hashes->train_mu);
    printf("VALIDATION_MU_SHA256=%s\n", hashes->validation_mu);
    printf("TRAIN_ROWS_SHA256=%s\n", hashes->train_rows);
    printf("VALIDATION_ROWS_SHA256=%s\n", hashes->validation_rows);
    printf("EMBEDDING_SPEC_SHA256=%s\n", hashes->embedding_spec);
    printf("SHA256_MANIFEST_SHA256=%s\n", hashes->checksum);
    printf("NEURAL_BATCH_LOADS=%" PRIu64 "\n", state->neural_batch_loads);
    printf("ENCODER_FORWARD_CALLS=%" PRIu64 "\n",
           state->encoder_forward_calls);
    printf("MU_VALUES_CREATED=%" PRIu64 "\n", state->mu_values_created);
    printf("MU_FINITE=true\nLOGVAR_FINITE=true\n");
    printf("ROW_ORDER_NATIVE_SPLIT_POS=true\n");
    printf("EVENT_IDENTITY_VERIFIED=true\n");
    printf("INPUT_UNCHANGED=true\n");
    printf("MODEL_PARAMETER_STORAGE_UNCHANGED=true\n");
    printf("BATCH_NORM_RUNNING_STATS_UNCHANGED=true\n");
    printf("OPTIMIZER_STATE_UNCHANGED=true\n");
    printf("CHECKPOINT_UNCHANGED=true\n");
    printf("DIRECT_ENCODER_ONLY_PATH=true\n");
    printf("DECODER_EXECUTED=false\nEPSILON_USED=false\n");
    printf("LATENT_SAMPLED=false\nBACKWARD_EXECUTED=false\n");
    printf("ADAM_STEP_EXECUTED=false\n");
    printf("LABEL_DEPENDENT_SELECTIONS=0\n");
    printf("LABEL_VALUES_EXPORTED=0\nLABELS_USED=false\n");
    printf("OUTPUT_DIRECTORY=%s\n", paths->final);
    printf("PARTIAL_ABSENT=true\nARTIFACT_VERIFIED=true\n");
    printf("VAE_C_MU_PRODUCTIVE_PASS=true\n");
}


static int build_producer_source_hashes(
    const char *repository_root,
    ProducerSourceHashes *hashes
)
{
    char path[VAE_C_MU_PATH_CAP];

    if (repository_root == NULL || hashes == NULL) {
        return 0;
    }
    memset(hashes, 0, sizeof(*hashes));
    return path_join(repository_root, MU_EXTRACTOR_SOURCE_REL,
                     path, sizeof(path)) &&
        vae_c_mu_file_sha256(path, hashes->extractor) == VAE_C_MU_OK &&
        path_join(repository_root, MU_CONTRACT_SOURCE_REL,
                  path, sizeof(path)) &&
        vae_c_mu_file_sha256(
            path, hashes->contract_source) == VAE_C_MU_OK &&
        path_join(repository_root, MU_CONTRACT_HEADER_REL,
                  path, sizeof(path)) &&
        vae_c_mu_file_sha256(
            path, hashes->contract_header) == VAE_C_MU_OK;
}


static int producer_source_hashes_equal(
    const ProducerSourceHashes *left,
    const ProducerSourceHashes *right
)
{
    return left != NULL && right != NULL &&
        hash_equal(left->extractor, right->extractor) &&
        hash_equal(left->contract_source, right->contract_source) &&
        hash_equal(left->contract_header, right->contract_header);
}


static int productive_command(
    const char *repository_root,
    const char *run,
    size_t batch_size,
    const char *extractor_code_commit,
    int verify_only
)
{
    VAECMuCanonicalContract contract;
    VAECMuCanonicalContract post_contract;
    VAECMuValidationReport validation_report;
    VAECMuValidationReport post_report;
    ProductiveState state;
    ArtifactPaths paths;
    ArtifactHashes hashes;
    VAECMuStatus status;
    char error_message[512];
    ProducerSourceHashes producer_sources;
    ProducerSourceHashes producer_sources_after;
    int final_is_directory = 0;
    int partial_is_directory = 0;
    int passed = 0;

    vae_c_mu_canonical_contract_init(&contract);
    vae_c_mu_canonical_contract_init(&post_contract);
    vae_c_mu_validation_report_init(&validation_report);
    vae_c_mu_validation_report_init(&post_report);
    productive_state_init(&state);
    error_message[0] = '\0';
    if (!commit_is_valid(extractor_code_commit) || CHAR_BIT != 8) {
        fprintf(stderr, "Invalid extractor code commit or platform.\n");
        goto cleanup;
    }
    status = vae_c_mu_validate_only(
        repository_root, run, &contract, &validation_report,
        error_message, sizeof(error_message)
    );
    if (status != VAE_C_MU_OK) {
        fprintf(stderr, "MU productive preflight failed: %s: %s\n",
                vae_c_mu_status_name(status), error_message);
        goto cleanup;
    }
    if (!build_artifact_paths(&contract, &paths) ||
        !build_producer_source_hashes(
            repository_root, &producer_sources) ||
        batch_size == 0u ||
        contract.expected_identity.max_batch_size > (uint64_t)SIZE_MAX ||
        batch_size >
            (size_t)contract.expected_identity.max_batch_size) {
        fprintf(stderr, "MU productive path/batch preflight failed.\n");
        goto cleanup;
    }
    if (!verify_only &&
        batch_size != (size_t)contract.expected_identity.max_batch_size) {
        fprintf(stderr,
                "Productive extraction requires canonical max batch size.\n");
        goto cleanup;
    }
    if (verify_only) {
        if (path_state(paths.final, &final_is_directory) != 1 ||
            !final_is_directory ||
            path_state(paths.partial, &partial_is_directory) != 0) {
            fprintf(stderr,
                    "Published artifact missing or .partial present.\n");
            goto cleanup;
        }
    } else if (!output_targets_are_absent(&paths)) {
        fprintf(stderr,
                "Refusing overwrite: final or .partial output exists.\n");
        goto cleanup;
    }
    if (!productive_generate(&contract, batch_size, &state)) {
        fprintf(stderr, "MU productive encoder invariant failure.\n");
        goto cleanup;
    }
    if (verify_only) {
        if (!verify_artifact_payload(
                paths.final, &contract, &paths, &state,
                extractor_code_commit, &producer_sources,
                &hashes)) {
            fprintf(stderr, "Published artifact verification failed.\n");
            goto cleanup;
        }
    } else if (!write_artifact_payload(
                   &contract, &paths, &state,
                   extractor_code_commit, &producer_sources,
                   &hashes)) {
        fprintf(stderr,
                "Artifact write/publish failed; inspect .partial.\n");
        goto cleanup;
    }
    error_message[0] = '\0';
    status = vae_c_mu_validate_only(
        repository_root, run, &post_contract, &post_report,
        error_message, sizeof(error_message)
    );
    if (status != VAE_C_MU_OK ||
        !build_producer_source_hashes(
            repository_root, &producer_sources_after) ||
        !producer_source_hashes_equal(
            &producer_sources, &producer_sources_after) ||
        path_state(paths.final, &final_is_directory) != 1 ||
        !final_is_directory ||
        path_state(paths.partial, &partial_is_directory) != 0) {
        fprintf(stderr, "MU productive post-check failed: %s\n",
                error_message);
        goto cleanup;
    }
    print_productive_result(
        verify_only, &contract, &paths, &state, &hashes,
        extractor_code_commit, &producer_sources
    );
    passed = 1;

cleanup:
    productive_state_free(&state);
    return passed ? 0 : 1;
}


static void print_validate_only(
    const VAECMuCanonicalContract *contract,
    const VAECMuValidationReport *report
)
{
    printf("VALIDATE_ONLY=true\n");
    printf("ARTIFACT_FORMAT=%s\n", contract->artifact_v1.format);
    printf("RUN=%s\n", contract->run_token);
    printf("PROFILE=%s\n", contract->profile.profile_id);
    printf("VIEW=%s\n", contract->artifact_v1.view);
    printf("REPRESENTATION=%s\n", contract->artifact_v1.representation);
    printf("CHECKPOINT_KIND=%s\n", contract->artifact_v1.checkpoint_kind);
    printf("CHECKPOINT_EPOCH=%" PRIu64 "\n", contract->expected_epoch);
    printf("CHECKPOINT_OPTIMIZER_STEP=%" PRIu64 "\n",
           contract->expected_optimizer_step);
    printf("CHECKPOINT_SHA256=%s\n", contract->checkpoint_sha256);
    printf("PROFILE_CONFIG_SHA256=%s\n",
           contract->expected_identity.profile_config_sha256);
    printf("TRAIN_ROWS=%" PRIu64 "\n", contract->train_count);
    printf("VALIDATION_ROWS=%" PRIu64 "\n",
           contract->validation_count);
    printf("LATENT_DIM=%lu\n", (unsigned long)contract->profile.latent_dim);
    printf("TRAIN_EXPECTED_BYTES=%" PRIu64 "\n",
           contract->artifact_v1.train.expected_bytes);
    printf("VALIDATION_EXPECTED_BYTES=%" PRIu64 "\n",
           contract->artifact_v1.validation.expected_bytes);
    printf("SOURCE_CONTRACT_VALIDATED=%s\n",
           report->source_contract_validated ? "true" : "false");
    printf("CHECKPOINT_EXTERNAL_SHA256_VALID=%s\n",
           report->checkpoint_hash_verified ? "true" : "false");
    printf("CHECKPOINT_SHA256_STABLE_ACROSS_LOAD=%s\n",
           report->checkpoint_hash_reverified_after_load ? "true" : "false");
    printf("CHECKPOINT_IDENTITY_VALID=%s\n",
           report->metadata_verified ? "true" : "false");
    printf("MODEL_ALLOCATED_FOR_CHECKPOINT_LOAD=true\n");
    printf("NEURAL_WORKSPACE_ALLOCATED=%s\n",
           report->neural_workspace_allocations == 0u ? "false" : "true");
    printf("DATA_BATCH_ALLOCATED=%s\n",
           report->neural_batch_loads == 0u ? "false" : "true");
    printf("FORWARD_CALLS=%" PRIu64 "\n", report->encoder_forward_calls);
    printf("BACKWARD_CALLS=%" PRIu64 "\n", report->backward_calls);
    printf("ADAM_STEPS=%" PRIu64 "\n", report->adam_steps);
    printf("MU_VALUES_CREATED=%" PRIu64 "\n", report->mu_values_created);
    printf("LABELS_USED=%s\n",
           contract->artifact_v1.labels_used ? "true" : "false");
    printf("OUTPUT_CREATED=%s\n",
           report->artifact_files_created == 0u ? "false" : "true");
    printf("ARTIFACT_FILES_CREATED=%" PRIu64 "\n",
           report->artifact_files_created);
    printf("READY_TO_EXTRACT=true\n");
    printf("VAE_C_MU_VALIDATION_PASSED=true\n");
}


static int tiny_forward(
    const char *repository_root,
    const char *run,
    size_t batch_size
)
{
    VAECMuCanonicalContract contract;
    VAECMuValidationReport validation_report;
    TinyForwardState state;
    VAECMuStatus status;
    uint64_t forward_calls;
    size_t input_values;
    size_t mu_values;
    size_t index;
    char input_sha[VAE_C_SHA256_HEX_SIZE];
    char mu_sha[VAE_C_SHA256_HEX_SIZE];
    char row_identity_sha[VAE_C_SHA256_HEX_SIZE];
    char checkpoint_sha[VAE_C_CHECKPOINT_SHA256_CAP];
    char row_mu_sha[TINY_TOTAL_ROWS][VAE_C_SHA256_HEX_SIZE];
    char error_message[512];
    int passed = 0;

    vae_c_mu_canonical_contract_init(&contract);
    vae_c_mu_validation_report_init(&validation_report);
    tiny_forward_state_init(&state);
    error_message[0] = '\0';
    status = vae_c_mu_validate_only(
        repository_root, run, &contract, &validation_report,
        error_message, sizeof(error_message)
    );
    if (status != VAE_C_MU_OK) {
        fprintf(stderr, "MU tiny-forward preflight failed: %s: %s\n",
                vae_c_mu_status_name(status), error_message);
        goto cleanup;
    }
    if (validation_report.neural_workspace_allocations != UINT64_C(0) ||
        validation_report.neural_batch_loads != UINT64_C(0) ||
        validation_report.encoder_forward_calls != UINT64_C(0) ||
        validation_report.backward_calls != UINT64_C(0) ||
        validation_report.adam_steps != UINT64_C(0) ||
        validation_report.mu_values_created != UINT64_C(0) ||
        validation_report.artifact_files_created != UINT64_C(0) ||
        !load_tiny_data(&contract, &state) ||
        !load_tiny_model(&contract, batch_size, &state) ||
        !size_multiply(TINY_TOTAL_ROWS,
                       state.tensor_spec.values_per_event, &input_values) ||
        !size_multiply(TINY_TOTAL_ROWS,
                       contract.profile.latent_dim, &mu_values) ||
        !hash_bytes(state.canonical_inputs,
                    input_values * sizeof(float), input_sha) ||
        !build_row_identity_hash(&state, row_identity_sha) ||
        !run_encoder_batches(
            &contract, batch_size, &state, &forward_calls) ||
        !parameter_snapshot_matches(&state) ||
        !batch_norm_snapshot_matches(&state) ||
        !optimizer_matches_snapshot(&state) ||
        state.model.vae_cache_generation - state.cache_generation_before !=
            forward_calls ||
        !hash_bytes(state.canonical_mu,
                    mu_values * sizeof(float), mu_sha) ||
        vae_c_mu_file_sha256(
            contract.checkpoint_path, checkpoint_sha) != VAE_C_MU_OK ||
        !hash_equal(checkpoint_sha, contract.checkpoint_sha256)) {
        fprintf(stderr, "MU tiny-forward invariant failure for %s.\n", run);
        goto cleanup;
    }
    for (index = 0; index < TINY_TOTAL_ROWS; index++) {
        if (!hash_bytes(
                state.canonical_mu + index * contract.profile.latent_dim,
                contract.profile.latent_dim * sizeof(float),
                row_mu_sha[index])) {
            fprintf(stderr, "MU tiny-forward row hash failure.\n");
            goto cleanup;
        }
    }

    printf("TINY_FORWARD=true\n");
    printf("RUN=%s\n", contract.run_token);
    printf("PROFILE=%s\n", contract.profile.profile_id);
    printf("VIEW=central\n");
    printf("REPRESENTATION=posterior_mu\n");
    printf("CHECKPOINT_KIND=BEST\n");
    printf("CHECKPOINT_EPOCH=%" PRIu64 "\n", contract.expected_epoch);
    printf("CHECKPOINT_OPTIMIZER_STEP=%" PRIu64 "\n",
           contract.expected_optimizer_step);
    printf("CHECKPOINT_SHA256=%s\n", contract.checkpoint_sha256);
    printf("BATCH_SIZE=%zu\n", batch_size);
    printf("TRAIN_ROWS=%d\n", TINY_ROWS_PER_SPLIT);
    printf("VALIDATION_ROWS=%d\n", TINY_ROWS_PER_SPLIT);
    printf("TOTAL_ROWS=%d\n", TINY_TOTAL_ROWS);
    printf("LATENT_DIM=%zu\n", contract.profile.latent_dim);
    printf("INPUT_CANONICAL_SHA256=%s\n", input_sha);
    printf("ROW_IDENTITY_SHA256=%s\n", row_identity_sha);
    printf("MU_CANONICAL_SHA256=%s\n", mu_sha);
    for (index = 0; index < TINY_TOTAL_ROWS; index++) {
        printf("ROW_%zu=%s,%zu,%zu,%d,%s,%s\n",
               index,
               state.rows[index].split,
               state.rows[index].split_pos,
               state.rows[index].manifest_index,
               state.rows[index].event_index,
               state.rows[index].event_id,
               row_mu_sha[index]);
    }
    printf("WORKSPACE_MU_REAL=true\n");
    printf("MU_FINITE=true\n");
    printf("LOGVAR_FINITE=true\n");
    printf("ROW_ORDER_NATIVE_SPLIT_POS=true\n");
    printf("EVENT_IDENTITY_VERIFIED=true\n");
    printf("INPUT_UNCHANGED=true\n");
    printf("MODEL_PARAMETER_STORAGE_UNCHANGED=true\n");
    printf("BATCH_NORM_RUNNING_STATS_UNCHANGED=true\n");
    printf("OPTIMIZER_STATE_UNCHANGED=true\n");
    printf("CHECKPOINT_UNCHANGED=true\n");
    printf("PREFLIGHT_ZERO_ACTIVITY=true\n");
    printf("DIRECT_ENCODER_ONLY_PATH=true\n");
    printf("ENCODER_FORWARD_CALLS=%" PRIu64 "\n", forward_calls);
    printf("DECODER_EXECUTED=false\n");
    printf("DECODER_WORKSPACE_UNCHANGED_ZERO=true\n");
    printf("EPSILON_USED=false\n");
    printf("LATENT_SAMPLED=false\n");
    printf("BACKWARD_EXECUTED=false\n");
    printf("ADAM_STEP_EXECUTED=false\n");
    printf("LABEL_DEPENDENT_SELECTIONS=0\n");
    printf("LABEL_VALUES_EXPORTED=0\n");
    printf("LABELS_USED=false\n");
    printf("OUTPUT_CREATED=false\n");
    printf("ARTIFACT_FILES_CREATED=0\n");
    printf("TINY_FORWARD_PASSED=true\n");
    passed = 1;

cleanup:
    tiny_forward_state_free(&state);
    return passed ? 0 : 1;
}


int main(int argc, char **argv)
{
    const char *run = NULL;
    const char *repository_root = ".";
    const char *extractor_code_commit = NULL;
    ExtractorMode mode = EXTRACTOR_MODE_NONE;
    size_t batch_size = 0u;
    int positional_seen = 0;
    int argument;
    VAECMuCanonicalContract contract;
    VAECMuValidationReport report;
    VAECMuStatus status;
    char error_message[512];

    for (argument = 1; argument < argc; argument++) {
        if (strcmp(argv[argument], "--help") == 0 ||
            strcmp(argv[argument], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        if (strcmp(argv[argument], "--validate-only") == 0) {
            if (mode != EXTRACTOR_MODE_NONE) {
                print_usage(argv[0]);
                return 2;
            }
            mode = EXTRACTOR_MODE_VALIDATE_ONLY;
            continue;
        }
        if (strcmp(argv[argument], "--tiny-forward") == 0) {
            if (mode != EXTRACTOR_MODE_NONE) {
                print_usage(argv[0]);
                return 2;
            }
            mode = EXTRACTOR_MODE_TINY_FORWARD;
            continue;
        }
        if (strcmp(argv[argument], "--extract") == 0) {
            if (mode != EXTRACTOR_MODE_NONE) {
                print_usage(argv[0]);
                return 2;
            }
            mode = EXTRACTOR_MODE_EXTRACT;
            continue;
        }
        if (strcmp(argv[argument], "--verify-artifact") == 0) {
            if (mode != EXTRACTOR_MODE_NONE) {
                print_usage(argv[0]);
                return 2;
            }
            mode = EXTRACTOR_MODE_VERIFY_ARTIFACT;
            continue;
        }
        if (strcmp(argv[argument], "--run") == 0) {
            if (run != NULL || argument + 1 >= argc) {
                print_usage(argv[0]);
                return 2;
            }
            run = argv[++argument];
            continue;
        }
        if (strcmp(argv[argument], "--batch-size") == 0) {
            if (batch_size != 0u || argument + 1 >= argc ||
                !parse_batch_size(argv[argument + 1], &batch_size)) {
                print_usage(argv[0]);
                return 2;
            }
            argument++;
            continue;
        }
        if (strcmp(argv[argument], "--extractor-code-commit") == 0) {
            if (extractor_code_commit != NULL || argument + 1 >= argc ||
                !commit_is_valid(argv[argument + 1])) {
                print_usage(argv[0]);
                return 2;
            }
            extractor_code_commit = argv[++argument];
            continue;
        }
        if (argv[argument][0] == '-') {
            print_usage(argv[0]);
            return 2;
        }
        if (positional_seen) {
            print_usage(argv[0]);
            return 2;
        }
        repository_root = argv[argument];
        positional_seen = 1;
    }

    if (run == NULL || mode == EXTRACTOR_MODE_NONE ||
        (mode == EXTRACTOR_MODE_VALIDATE_ONLY &&
         (batch_size != 0u || extractor_code_commit != NULL)) ||
        (mode == EXTRACTOR_MODE_TINY_FORWARD &&
         ((batch_size != 1u && batch_size != 2u) ||
          extractor_code_commit != NULL)) ||
        ((mode == EXTRACTOR_MODE_EXTRACT ||
          mode == EXTRACTOR_MODE_VERIFY_ARTIFACT) &&
         (batch_size == 0u || extractor_code_commit == NULL))) {
        print_usage(argv[0]);
        return 2;
    }
    if (mode == EXTRACTOR_MODE_TINY_FORWARD) {
        return tiny_forward(repository_root, run, batch_size);
    }
    if (mode == EXTRACTOR_MODE_EXTRACT ||
        mode == EXTRACTOR_MODE_VERIFY_ARTIFACT) {
        return productive_command(
            repository_root, run, batch_size,
            extractor_code_commit,
            mode == EXTRACTOR_MODE_VERIFY_ARTIFACT
        );
    }

    vae_c_mu_canonical_contract_init(&contract);
    vae_c_mu_validation_report_init(&report);
    error_message[0] = '\0';
    status = vae_c_mu_validate_only(
        repository_root, run, &contract, &report,
        error_message, sizeof(error_message)
    );
    if (status != VAE_C_MU_OK) {
        fprintf(
            stderr, "MU validate-only failed: %s: %s\n",
            vae_c_mu_status_name(status), error_message
        );
        return 1;
    }
    print_validate_only(&contract, &report);
    return 0;
}
