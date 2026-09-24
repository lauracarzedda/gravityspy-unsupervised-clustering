#include "vae_c_model_forward.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


typedef struct {
    VAECModel model_object;
    VAECModelWorkspace workspace_object;
    unsigned char *dynamic_bytes;
    size_t dynamic_size;
} FullSnapshot;


typedef struct {
    char model_profile_id[VAE_C_PROFILE_TEXT_CAP];
    char workspace_profile_id[VAE_C_PROFILE_TEXT_CAP];
    VAECConfig model_config;
    VAECConfig workspace_config;
    VAECParameter *registry[VAE_C_MODEL_MAX_PARAMETERS];
    size_t parameter_count;
    size_t max_batch_size;
    size_t batch_size;
    uint64_t trainable_value_count;
    uint64_t initialization_seed;
    bool parameters_initialized;
    bool model_allocated;
    size_t owned_tensor_count;
    size_t logical_value_count;
    size_t storage_float_count;
    size_t storage_bytes;
    bool workspace_allocated;
    VAECTensor input_object;
    VAECTensor epsilon_object;
    unsigned char *bytes;
    size_t byte_count;
} InvariantSnapshot;


typedef struct {
    unsigned char *bytes;
    size_t byte_count;
} RunningSnapshot;


#define TEST_MODEL_BUFFER_CAPACITY \
    (4 * VAE_C_MODEL_MAX_PARAMETERS + \
     7 * (VAE_C_MAX_CONV_DEPTH + 1 + \
          VAE_C_MAX_DECODER_HIDDEN_CONVS) + 1)


static int report_check(const char *name, int passed)
{
    if (passed) {
        printf("[PASS] %s\n", name);
        return 0;
    }
    fprintf(stderr, "[FAIL] %s\n", name);
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


static int size_multiply(size_t left, size_t right, size_t *result)
{
    if (result == NULL || (left != 0 && right > SIZE_MAX / left)) {
        return 0;
    }
    *result = left * right;
    return 1;
}


static size_t collect_workspace_tensors(
    const VAECModelWorkspace *workspace,
    const VAECTensor *tensors[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ]
)
{
    size_t count = 0;
    size_t index;

    tensors[count++] = &workspace->input;
    for (index = 0; index < workspace->config.conv_depth; index++) {
        tensors[count++] = &workspace->encoder_conv_outputs[index];
        tensors[count++] =
            &workspace->encoder_batch_norm_outputs[index];
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
    for (index = 0;
         index < workspace->config.decoder_upsample_count;
         index++) {

        tensors[count++] = &workspace->decoder_upsample_outputs[index];
    }
    for (index = 0;
         index < workspace->config.decoder_hidden_conv_count;
         index++) {

        tensors[count++] =
            &workspace->decoder_hidden_conv_outputs[index];
        tensors[count++] =
            &workspace->decoder_hidden_batch_norm_outputs[index];
        tensors[count++] =
            &workspace->decoder_hidden_relu_outputs[index];
    }
    tensors[count++] = &workspace->logits;
    return count;
}


static size_t collect_batch_norms(
    const VAECModel *model,
    const VAECBatchNorm *layers[
        VAE_C_MAX_CONV_DEPTH + 1 +
        VAE_C_MAX_DECODER_HIDDEN_CONVS
    ]
)
{
    size_t count = 0;
    size_t index;

    for (index = 0; index < model->config.conv_depth; index++) {
        layers[count++] = &model->encoder_batch_norms[index];
    }
    layers[count++] = &model->decoder_fc_batch_norm;
    for (index = 0;
         index < model->config.decoder_hidden_conv_count;
         index++) {

        layers[count++] = &model->decoder_hidden_batch_norms[index];
    }
    return count;
}


static size_t collect_model_buffers(
    const VAECModel *model,
    const void *buffers[TEST_MODEL_BUFFER_CAPACITY]
)
{
    const VAECBatchNorm *batch_norms[
        VAE_C_MAX_CONV_DEPTH + 1 +
        VAE_C_MAX_DECODER_HIDDEN_CONVS
    ];
    size_t batch_norm_count;
    size_t count = 0;
    size_t index;

    if (model == NULL || buffers == NULL) {
        return SIZE_MAX;
    }
    for (index = 0; index < model->parameter_count; index++) {
        const VAECParameter *parameter = model->parameters[index];

        if (parameter == NULL ||
            count > TEST_MODEL_BUFFER_CAPACITY - 4) {

            return SIZE_MAX;
        }
        buffers[count++] = parameter->values;
        buffers[count++] = parameter->gradients;
        buffers[count++] = parameter->adam_m;
        buffers[count++] = parameter->adam_v;
    }
    batch_norm_count = collect_batch_norms(model, batch_norms);
    for (index = 0; index < batch_norm_count; index++) {
        const VAECBatchNorm *layer = batch_norms[index];

        if (count > TEST_MODEL_BUFFER_CAPACITY - 7) {
            return SIZE_MAX;
        }
        buffers[count++] = layer->running_mean;
        buffers[count++] = layer->running_variance;
        buffers[count++] = layer->batch_mean;
        buffers[count++] = layer->batch_inv_std;
        buffers[count++] = layer->work_mean;
        buffers[count++] = layer->work_inv_std;
        buffers[count++] = layer->work_unbiased_variance;
    }
    if (count >= TEST_MODEL_BUFFER_CAPACITY) {
        return SIZE_MAX;
    }
    buffers[count++] = model->encoder_max_pool.argmax_indices;
    return count;
}


static size_t count_model_buffer_aliases(const VAECModel *model)
{
    const void *buffers[TEST_MODEL_BUFFER_CAPACITY];
    size_t count = collect_model_buffers(model, buffers);
    size_t aliases = 0;
    size_t left;

    if (count == SIZE_MAX) {
        return SIZE_MAX;
    }
    for (left = 0; left < count; left++) {
        size_t right;

        if (buffers[left] == NULL) {
            return SIZE_MAX;
        }
        for (right = left + 1; right < count; right++) {
            if (buffers[left] == buffers[right]) {
                if (aliases == SIZE_MAX) {
                    return SIZE_MAX;
                }
                aliases++;
            }
        }
    }
    return aliases;
}


static int stats_matches_pre_lifecycle_alias(
    const VAECVAELossStats *stats,
    const VAECModel *model,
    const VAECModelWorkspace *workspace
)
{
    const VAECTensor *tensors[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ];
    const void *model_buffers[TEST_MODEL_BUFFER_CAPACITY];
    const void *pointer = (const void *)stats;
    size_t model_buffer_count = collect_model_buffers(
        model, model_buffers
    );
    size_t tensor_count = collect_workspace_tensors(workspace, tensors);
    size_t index;

    if (model_buffer_count == SIZE_MAX ||
        pointer == (const void *)model ||
        pointer == (const void *)model->profile_id ||
        pointer == (const void *)&model->config ||
        pointer == (const void *)&model->max_batch_size ||
        pointer == (const void *)&model->encoder_max_pool ||
        pointer == (const void *)&model->encoder_fc ||
        pointer == (const void *)&model->decoder_fc ||
        pointer == (const void *)&model->decoder_fc_batch_norm ||
        pointer == (const void *)&model->decoder_output_conv ||
        pointer == (const void *)model->parameters ||
        pointer == (const void *)&model->parameter_count ||
        pointer == (const void *)&model->trainable_value_count ||
        pointer == (const void *)&model->initialization_seed ||
        pointer == (const void *)&model->parameters_initialized ||
        pointer == (const void *)&model->allocated ||
        pointer == (const void *)workspace ||
        pointer == (const void *)workspace->profile_id ||
        pointer == (const void *)&workspace->config ||
        pointer == (const void *)&workspace->batch_size ||
        pointer == (const void *)&workspace->decoder_reshape_view ||
        pointer == (const void *)&workspace->owned_tensor_count ||
        pointer == (const void *)&workspace->logical_value_count ||
        pointer == (const void *)&workspace->storage_float_count ||
        pointer == (const void *)&workspace->storage_bytes ||
        pointer == (const void *)&workspace->allocated) {

        return 1;
    }
    for (index = 0; index < model_buffer_count; index++) {
        if (pointer == model_buffers[index]) {
            return 1;
        }
    }
    for (index = 0; index < model->config.conv_depth; index++) {
        if (pointer == (const void *)&model->encoder_convs[index] ||
            pointer ==
                (const void *)&model->encoder_batch_norms[index]) {

            return 1;
        }
    }
    for (index = 0;
         index < model->config.decoder_hidden_conv_count;
         index++) {

        if (pointer ==
                (const void *)&model->decoder_hidden_convs[index] ||
            pointer ==
                (const void *)&model->decoder_hidden_batch_norms[index]) {

            return 1;
        }
    }
    for (index = 0; index < model->parameter_count; index++) {
        if (pointer == (const void *)model->parameters[index]) {
            return 1;
        }
    }
    for (index = 0; index < tensor_count; index++) {
        const VAECTensor *tensor = tensors[index];

        if (pointer == (const void *)tensor ||
            pointer == (const void *)&tensor->shape ||
            pointer == (const void *)&tensor->owned_bytes ||
            pointer == (const void *)&tensor->owns_memory ||
            pointer == (const void *)&tensor->has_grad ||
            pointer == (const void *)tensor->data ||
            (tensor->grad != NULL &&
             pointer == (const void *)tensor->grad)) {

            return 1;
        }
    }
    return 0;
}


static size_t count_stats_lifecycle_aliases(
    const VAECVAELossStats *stats,
    const VAECModel *model,
    const VAECModelWorkspace *workspace
)
{
    const void *pointer = (const void *)stats;
    size_t count = 0;

    count += pointer == (const void *)&model->vae_cache_generation;
    count += pointer == (const void *)&workspace->vae_backward_model;
    count +=
        pointer == (const void *)&workspace->vae_backward_generation;
    return count;
}


static void full_snapshot_init(FullSnapshot *snapshot)
{
    memset(snapshot, 0, sizeof(*snapshot));
}


static void full_snapshot_free(FullSnapshot *snapshot)
{
    if (snapshot != NULL) {
        free(snapshot->dynamic_bytes);
        full_snapshot_init(snapshot);
    }
}


static int full_dynamic_size(
    const VAECModel *model,
    const VAECModelWorkspace *workspace,
    size_t *total
)
{
    const VAECBatchNorm *batch_norms[
        VAE_C_MAX_CONV_DEPTH + 1 +
        VAE_C_MAX_DECODER_HIDDEN_CONVS
    ];
    const VAECTensor *tensors[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ];
    size_t batch_norm_count;
    size_t tensor_count;
    size_t result = 0;
    size_t index;

    if (model == NULL || workspace == NULL || total == NULL) {
        return 0;
    }
    for (index = 0; index < model->parameter_count; index++) {
        if (!size_add(
                result,
                model->parameters[index]->owned_bytes,
                &result)) {

            return 0;
        }
    }
    batch_norm_count = collect_batch_norms(model, batch_norms);
    for (index = 0; index < batch_norm_count; index++) {
        if (!size_add(result, batch_norms[index]->owned_bytes, &result)) {
            return 0;
        }
    }
    if (!size_add(
            result, model->encoder_max_pool.owned_bytes, &result)) {

        return 0;
    }
    tensor_count = collect_workspace_tensors(workspace, tensors);
    for (index = 0; index < tensor_count; index++) {
        if (!size_add(result, tensors[index]->owned_bytes, &result)) {
            return 0;
        }
    }
    *total = result;
    return 1;
}


static void copy_region(
    unsigned char *destination,
    size_t *offset,
    const void *source,
    size_t byte_count
)
{
    memcpy(destination + *offset, source, byte_count);
    *offset += byte_count;
}


static int full_snapshot_capture(
    const VAECModel *model,
    const VAECModelWorkspace *workspace,
    FullSnapshot *snapshot
)
{
    const VAECBatchNorm *batch_norms[
        VAE_C_MAX_CONV_DEPTH + 1 +
        VAE_C_MAX_DECODER_HIDDEN_CONVS
    ];
    const VAECTensor *tensors[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ];
    size_t batch_norm_count;
    size_t tensor_count;
    size_t offset = 0;
    size_t index;

    if (model == NULL || workspace == NULL || snapshot == NULL) {
        return 0;
    }
    full_snapshot_free(snapshot);
    if (!full_dynamic_size(
            model, workspace, &snapshot->dynamic_size) ||
        snapshot->dynamic_size == 0) {

        return 0;
    }
    snapshot->dynamic_bytes =
        (unsigned char *)malloc(snapshot->dynamic_size);
    if (snapshot->dynamic_bytes == NULL) {
        full_snapshot_init(snapshot);
        return 0;
    }
    snapshot->model_object = *model;
    snapshot->workspace_object = *workspace;
    for (index = 0; index < model->parameter_count; index++) {
        copy_region(
            snapshot->dynamic_bytes,
            &offset,
            model->parameters[index]->owned_block,
            model->parameters[index]->owned_bytes
        );
    }
    batch_norm_count = collect_batch_norms(model, batch_norms);
    for (index = 0; index < batch_norm_count; index++) {
        copy_region(
            snapshot->dynamic_bytes,
            &offset,
            batch_norms[index]->owned_block,
            batch_norms[index]->owned_bytes
        );
    }
    copy_region(
        snapshot->dynamic_bytes,
        &offset,
        model->encoder_max_pool.argmax_indices,
        model->encoder_max_pool.owned_bytes
    );
    tensor_count = collect_workspace_tensors(workspace, tensors);
    for (index = 0; index < tensor_count; index++) {
        copy_region(
            snapshot->dynamic_bytes,
            &offset,
            tensors[index]->owned_block,
            tensors[index]->owned_bytes
        );
    }
    return offset == snapshot->dynamic_size;
}


static int region_matches(
    const unsigned char *expected,
    size_t *offset,
    const void *actual,
    size_t byte_count
)
{
    int equal = memcmp(expected + *offset, actual, byte_count) == 0;

    *offset += byte_count;
    return equal;
}


static int full_snapshot_matches(
    const VAECModel *model,
    const VAECModelWorkspace *workspace,
    const FullSnapshot *snapshot
)
{
    const VAECBatchNorm *batch_norms[
        VAE_C_MAX_CONV_DEPTH + 1 +
        VAE_C_MAX_DECODER_HIDDEN_CONVS
    ];
    const VAECTensor *tensors[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ];
    size_t batch_norm_count;
    size_t tensor_count;
    size_t offset = 0;
    size_t current_size;
    size_t index;

    if (model == NULL || workspace == NULL || snapshot == NULL ||
        snapshot->dynamic_bytes == NULL ||
        memcmp(model, &snapshot->model_object, sizeof(*model)) != 0 ||
        memcmp(
            workspace,
            &snapshot->workspace_object,
            sizeof(*workspace)) != 0 ||
        !full_dynamic_size(model, workspace, &current_size) ||
        current_size != snapshot->dynamic_size) {

        return 0;
    }
    for (index = 0; index < model->parameter_count; index++) {
        if (!region_matches(
                snapshot->dynamic_bytes,
                &offset,
                model->parameters[index]->owned_block,
                model->parameters[index]->owned_bytes)) {

            return 0;
        }
    }
    batch_norm_count = collect_batch_norms(model, batch_norms);
    for (index = 0; index < batch_norm_count; index++) {
        if (!region_matches(
                snapshot->dynamic_bytes,
                &offset,
                batch_norms[index]->owned_block,
                batch_norms[index]->owned_bytes)) {

            return 0;
        }
    }
    if (!region_matches(
            snapshot->dynamic_bytes,
            &offset,
            model->encoder_max_pool.argmax_indices,
            model->encoder_max_pool.owned_bytes)) {

        return 0;
    }
    tensor_count = collect_workspace_tensors(workspace, tensors);
    for (index = 0; index < tensor_count; index++) {
        if (!region_matches(
                snapshot->dynamic_bytes,
                &offset,
                tensors[index]->owned_block,
                tensors[index]->owned_bytes)) {

            return 0;
        }
    }
    return offset == snapshot->dynamic_size;
}


static void invariant_snapshot_init(InvariantSnapshot *snapshot)
{
    memset(snapshot, 0, sizeof(*snapshot));
}


static void invariant_snapshot_free(InvariantSnapshot *snapshot)
{
    if (snapshot != NULL) {
        free(snapshot->bytes);
        invariant_snapshot_init(snapshot);
    }
}


static int invariant_byte_count(
    const VAECModel *model,
    const VAECModelWorkspace *workspace,
    size_t *total
)
{
    const VAECTensor *tensors[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ];
    size_t tensor_count;
    size_t input_bytes;
    size_t epsilon_bytes;
    size_t result = 0;
    size_t index;

    if (!size_multiply(
            workspace->input.shape.count,
            sizeof(float),
            &input_bytes) ||
        !size_multiply(
            workspace->epsilon.shape.count,
            sizeof(float),
            &epsilon_bytes)) {

        return 0;
    }
    for (index = 0; index < model->parameter_count; index++) {
        if (!size_add(
                result,
                model->parameters[index]->owned_bytes,
                &result)) {

            return 0;
        }
    }
    if (!size_add(result, input_bytes, &result) ||
        !size_add(result, epsilon_bytes, &result)) {

        return 0;
    }
    tensor_count = collect_workspace_tensors(workspace, tensors);
    for (index = 0; index < tensor_count; index++) {
        size_t gradient_bytes;

        if (tensors[index]->grad != NULL) {
            if (!size_multiply(
                    tensors[index]->shape.count,
                    sizeof(float),
                    &gradient_bytes) ||
                !size_add(result, gradient_bytes, &result)) {

                return 0;
            }
        }
    }
    *total = result;
    return 1;
}


static int invariant_snapshot_capture(
    const VAECModel *model,
    const VAECModelWorkspace *workspace,
    InvariantSnapshot *snapshot
)
{
    const VAECTensor *tensors[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ];
    size_t tensor_count;
    size_t offset = 0;
    size_t bytes;
    size_t index;

    invariant_snapshot_free(snapshot);
    if (!invariant_byte_count(
            model, workspace, &snapshot->byte_count) ||
        snapshot->byte_count == 0) {

        return 0;
    }
    snapshot->bytes = (unsigned char *)malloc(snapshot->byte_count);
    if (snapshot->bytes == NULL) {
        invariant_snapshot_init(snapshot);
        return 0;
    }
    memcpy(snapshot->model_profile_id,
           model->profile_id,
           sizeof(snapshot->model_profile_id));
    memcpy(snapshot->workspace_profile_id,
           workspace->profile_id,
           sizeof(snapshot->workspace_profile_id));
    snapshot->model_config = model->config;
    snapshot->workspace_config = workspace->config;
    memcpy(snapshot->registry,
           model->parameters,
           sizeof(snapshot->registry));
    snapshot->parameter_count = model->parameter_count;
    snapshot->max_batch_size = model->max_batch_size;
    snapshot->batch_size = workspace->batch_size;
    snapshot->trainable_value_count = model->trainable_value_count;
    snapshot->initialization_seed = model->initialization_seed;
    snapshot->parameters_initialized = model->parameters_initialized;
    snapshot->model_allocated = model->allocated;
    snapshot->owned_tensor_count = workspace->owned_tensor_count;
    snapshot->logical_value_count = workspace->logical_value_count;
    snapshot->storage_float_count = workspace->storage_float_count;
    snapshot->storage_bytes = workspace->storage_bytes;
    snapshot->workspace_allocated = workspace->allocated;
    snapshot->input_object = workspace->input;
    snapshot->epsilon_object = workspace->epsilon;

    for (index = 0; index < model->parameter_count; index++) {
        copy_region(
            snapshot->bytes,
            &offset,
            model->parameters[index]->owned_block,
            model->parameters[index]->owned_bytes
        );
    }
    bytes = workspace->input.shape.count * sizeof(float);
    copy_region(
        snapshot->bytes, &offset, workspace->input.data, bytes
    );
    bytes = workspace->epsilon.shape.count * sizeof(float);
    copy_region(
        snapshot->bytes, &offset, workspace->epsilon.data, bytes
    );
    tensor_count = collect_workspace_tensors(workspace, tensors);
    for (index = 0; index < tensor_count; index++) {
        if (tensors[index]->grad != NULL) {
            bytes = tensors[index]->shape.count * sizeof(float);
            copy_region(
                snapshot->bytes,
                &offset,
                tensors[index]->grad,
                bytes
            );
        }
    }
    return offset == snapshot->byte_count;
}


static int invariant_snapshot_matches(
    const VAECModel *model,
    const VAECModelWorkspace *workspace,
    const InvariantSnapshot *snapshot
)
{
    const VAECTensor *tensors[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ];
    size_t tensor_count;
    size_t current_count;
    size_t offset = 0;
    size_t bytes;
    size_t index;

    if (snapshot->bytes == NULL ||
        !invariant_byte_count(model, workspace, &current_count) ||
        current_count != snapshot->byte_count ||
        memcmp(snapshot->model_profile_id,
               model->profile_id,
               sizeof(snapshot->model_profile_id)) != 0 ||
        memcmp(snapshot->workspace_profile_id,
               workspace->profile_id,
               sizeof(snapshot->workspace_profile_id)) != 0 ||
        memcmp(&snapshot->model_config,
               &model->config,
               sizeof(model->config)) != 0 ||
        memcmp(&snapshot->workspace_config,
               &workspace->config,
               sizeof(workspace->config)) != 0 ||
        memcmp(snapshot->registry,
               model->parameters,
               sizeof(snapshot->registry)) != 0 ||
        snapshot->parameter_count != model->parameter_count ||
        snapshot->max_batch_size != model->max_batch_size ||
        snapshot->batch_size != workspace->batch_size ||
        snapshot->trainable_value_count !=
            model->trainable_value_count ||
        snapshot->initialization_seed != model->initialization_seed ||
        snapshot->parameters_initialized !=
            model->parameters_initialized ||
        snapshot->model_allocated != model->allocated ||
        snapshot->owned_tensor_count != workspace->owned_tensor_count ||
        snapshot->logical_value_count !=
            workspace->logical_value_count ||
        snapshot->storage_float_count !=
            workspace->storage_float_count ||
        snapshot->storage_bytes != workspace->storage_bytes ||
        snapshot->workspace_allocated != workspace->allocated ||
        memcmp(&snapshot->input_object,
               &workspace->input,
               sizeof(workspace->input)) != 0 ||
        memcmp(&snapshot->epsilon_object,
               &workspace->epsilon,
               sizeof(workspace->epsilon)) != 0) {

        return 0;
    }
    for (index = 0; index < model->parameter_count; index++) {
        if (!region_matches(
                snapshot->bytes,
                &offset,
                model->parameters[index]->owned_block,
                model->parameters[index]->owned_bytes)) {

            return 0;
        }
    }
    bytes = workspace->input.shape.count * sizeof(float);
    if (!region_matches(
            snapshot->bytes, &offset, workspace->input.data, bytes)) {

        return 0;
    }
    bytes = workspace->epsilon.shape.count * sizeof(float);
    if (!region_matches(
            snapshot->bytes, &offset, workspace->epsilon.data, bytes)) {

        return 0;
    }
    tensor_count = collect_workspace_tensors(workspace, tensors);
    for (index = 0; index < tensor_count; index++) {
        if (tensors[index]->grad != NULL) {
            bytes = tensors[index]->shape.count * sizeof(float);
            if (!region_matches(
                    snapshot->bytes,
                    &offset,
                    tensors[index]->grad,
                    bytes)) {

                return 0;
            }
        }
    }
    return offset == snapshot->byte_count;
}


static void running_snapshot_init(RunningSnapshot *snapshot)
{
    memset(snapshot, 0, sizeof(*snapshot));
}


static void running_snapshot_free(RunningSnapshot *snapshot)
{
    if (snapshot != NULL) {
        free(snapshot->bytes);
        running_snapshot_init(snapshot);
    }
}


static int running_snapshot_capture(
    const VAECModel *model,
    RunningSnapshot *snapshot
)
{
    const VAECBatchNorm *layers[
        VAE_C_MAX_CONV_DEPTH + 1 +
        VAE_C_MAX_DECODER_HIDDEN_CONVS
    ];
    size_t count = collect_batch_norms(model, layers);
    size_t offset = 0;
    size_t index;

    running_snapshot_free(snapshot);
    for (index = 0; index < count; index++) {
        size_t bytes;

        if (!size_multiply(
                layers[index]->num_features,
                2 * sizeof(float),
                &bytes) ||
            !size_add(snapshot->byte_count, bytes,
                      &snapshot->byte_count)) {

            return 0;
        }
    }
    snapshot->bytes = (unsigned char *)malloc(snapshot->byte_count);
    if (snapshot->bytes == NULL) {
        running_snapshot_init(snapshot);
        return 0;
    }
    for (index = 0; index < count; index++) {
        size_t bytes = layers[index]->num_features * sizeof(float);

        copy_region(snapshot->bytes,
                    &offset,
                    layers[index]->running_mean,
                    bytes);
        copy_region(snapshot->bytes,
                    &offset,
                    layers[index]->running_variance,
                    bytes);
    }
    return offset == snapshot->byte_count;
}


static int running_snapshot_matches(
    const VAECModel *model,
    const RunningSnapshot *snapshot
)
{
    const VAECBatchNorm *layers[
        VAE_C_MAX_CONV_DEPTH + 1 +
        VAE_C_MAX_DECODER_HIDDEN_CONVS
    ];
    size_t count = collect_batch_norms(model, layers);
    size_t offset = 0;
    size_t index;

    if (snapshot->bytes == NULL) {
        return 0;
    }
    for (index = 0; index < count; index++) {
        size_t bytes = layers[index]->num_features * sizeof(float);

        if (!region_matches(snapshot->bytes,
                            &offset,
                            layers[index]->running_mean,
                            bytes) ||
            !region_matches(snapshot->bytes,
                            &offset,
                            layers[index]->running_variance,
                            bytes)) {

            return 0;
        }
    }
    return offset == snapshot->byte_count;
}


static int parameter_equal(
    const VAECParameter *left,
    const VAECParameter *right
)
{
    return
        left->parameter_key == right->parameter_key &&
        vae_c_shape_equal(&left->shape, &right->shape) &&
        left->owned_bytes == right->owned_bytes &&
        memcmp(left->owned_block,
               right->owned_block,
               left->owned_bytes) == 0;
}


static int batch_norm_equal(
    const VAECBatchNorm *left,
    const VAECBatchNorm *right
)
{
    return
        left->kind == right->kind &&
        left->num_features == right->num_features &&
        left->epsilon == right->epsilon &&
        left->momentum == right->momentum &&
        parameter_equal(&left->gamma, &right->gamma) &&
        parameter_equal(&left->beta, &right->beta) &&
        left->owned_bytes == right->owned_bytes &&
        memcmp(left->owned_block,
               right->owned_block,
               left->owned_bytes) == 0 &&
        left->last_sample_count == right->last_sample_count &&
        memcmp(&left->last_input_shape,
               &right->last_input_shape,
               sizeof(left->last_input_shape)) == 0 &&
        memcmp(&left->last_output_shape,
               &right->last_output_shape,
               sizeof(left->last_output_shape)) == 0 &&
        left->cache_valid == right->cache_valid;
}


static int model_equal(const VAECModel *left, const VAECModel *right)
{
    size_t index;

    if (memcmp(left->profile_id,
               right->profile_id,
               sizeof(left->profile_id)) != 0 ||
        memcmp(&left->config,
               &right->config,
               sizeof(left->config)) != 0 ||
        left->max_batch_size != right->max_batch_size ||
        left->parameter_count != right->parameter_count ||
        left->trainable_value_count != right->trainable_value_count ||
        left->initialization_seed != right->initialization_seed ||
        left->vae_cache_generation != right->vae_cache_generation ||
        left->parameters_initialized != right->parameters_initialized ||
        left->allocated != right->allocated ||
        left->encoder_max_pool.capacity !=
            right->encoder_max_pool.capacity ||
        left->encoder_max_pool.owned_bytes !=
            right->encoder_max_pool.owned_bytes ||
        memcmp(left->encoder_max_pool.argmax_indices,
               right->encoder_max_pool.argmax_indices,
               left->encoder_max_pool.owned_bytes) != 0 ||
        left->encoder_max_pool.last_count !=
            right->encoder_max_pool.last_count ||
        memcmp(&left->encoder_max_pool.last_input_shape,
               &right->encoder_max_pool.last_input_shape,
               sizeof(left->encoder_max_pool.last_input_shape)) != 0 ||
        memcmp(&left->encoder_max_pool.last_output_shape,
               &right->encoder_max_pool.last_output_shape,
               sizeof(left->encoder_max_pool.last_output_shape)) != 0 ||
        left->encoder_max_pool.cache_valid !=
            right->encoder_max_pool.cache_valid) {

        return 0;
    }
    for (index = 0; index < left->parameter_count; index++) {
        if (!parameter_equal(
                left->parameters[index], right->parameters[index])) {

            return 0;
        }
    }
    for (index = 0; index < left->config.conv_depth; index++) {
        if (!batch_norm_equal(
                &left->encoder_batch_norms[index],
                &right->encoder_batch_norms[index])) {

            return 0;
        }
    }
    if (!batch_norm_equal(&left->decoder_fc_batch_norm,
                          &right->decoder_fc_batch_norm)) {

        return 0;
    }
    for (index = 0;
         index < left->config.decoder_hidden_conv_count;
         index++) {

        if (!batch_norm_equal(
                &left->decoder_hidden_batch_norms[index],
                &right->decoder_hidden_batch_norms[index])) {

            return 0;
        }
    }
    return 1;
}


static int tensor_equal(const VAECTensor *left, const VAECTensor *right)
{
    size_t bytes;

    if (!vae_c_shape_equal(&left->shape, &right->shape) ||
        left->owns_memory != right->owns_memory ||
        left->has_grad != right->has_grad ||
        left->owned_bytes != right->owned_bytes ||
        !size_multiply(left->shape.count, sizeof(float), &bytes) ||
        memcmp(left->data, right->data, bytes) != 0) {

        return 0;
    }
    return
        left->grad == NULL ||
        memcmp(left->grad, right->grad, bytes) == 0;
}


static int workspace_equal(
    const VAECModel *left_model,
    const VAECModelWorkspace *left,
    const VAECModel *right_model,
    const VAECModelWorkspace *right
)
{
    const VAECTensor *left_tensors[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ];
    const VAECTensor *right_tensors[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ];
    size_t left_count = collect_workspace_tensors(left, left_tensors);
    size_t right_count = collect_workspace_tensors(right, right_tensors);
    size_t index;

    if (memcmp(left->profile_id,
               right->profile_id,
               sizeof(left->profile_id)) != 0 ||
        memcmp(&left->config,
               &right->config,
               sizeof(left->config)) != 0 ||
        left->batch_size != right->batch_size ||
        left->owned_tensor_count != right->owned_tensor_count ||
        left->logical_value_count != right->logical_value_count ||
        left->storage_float_count != right->storage_float_count ||
        left->storage_bytes != right->storage_bytes ||
        left->vae_backward_generation !=
            right->vae_backward_generation ||
        ((left->vae_backward_model == NULL) !=
         (right->vae_backward_model == NULL)) ||
        (left->vae_backward_model != NULL &&
         (left->vae_backward_model != left_model ||
          right->vae_backward_model != right_model)) ||
        left->allocated != right->allocated ||
        left_count != right_count ||
        !vae_c_shape_equal(
            &left->decoder_reshape_view.shape,
            &right->decoder_reshape_view.shape) ||
        left->decoder_reshape_view.owns_memory ||
        right->decoder_reshape_view.owns_memory ||
        left->decoder_reshape_view.data !=
            left->decoder_fc_relu_output.data ||
        right->decoder_reshape_view.data !=
            right->decoder_fc_relu_output.data) {

        return 0;
    }
    for (index = 0; index < left_count; index++) {
        if (!tensor_equal(left_tensors[index], right_tensors[index])) {
            return 0;
        }
    }
    return 1;
}


static int stats_equal(
    const VAECVAELossStats *left,
    const VAECVAELossStats *right
)
{
    return
        left->batch_size == right->batch_size &&
        left->beta_kl == right->beta_kl &&
        left->reconstruction.batch_size ==
            right->reconstruction.batch_size &&
        left->reconstruction.values_per_sample ==
            right->reconstruction.values_per_sample &&
        left->reconstruction.total_values ==
            right->reconstruction.total_values &&
        left->reconstruction.total_sum ==
            right->reconstruction.total_sum &&
        left->reconstruction.mean_per_sample ==
            right->reconstruction.mean_per_sample &&
        left->reconstruction.mean_per_value ==
            right->reconstruction.mean_per_value &&
        left->kl.batch_size == right->kl.batch_size &&
        left->kl.latent_dim == right->kl.latent_dim &&
        left->kl.total_values == right->kl.total_values &&
        left->kl.total_sum == right->kl.total_sum &&
        left->kl.mean_per_sample == right->kl.mean_per_sample &&
        left->kl.mean_per_dimension == right->kl.mean_per_dimension &&
        left->total_sum == right->total_sum &&
        left->mean_per_sample == right->mean_per_sample;
}


static void fill_input_and_epsilon(VAECModelWorkspace *workspace)
{
    size_t index;

    for (index = 0; index < workspace->input.shape.count; index++) {
        workspace->input.data[index] =
            (float)(index % 17) * (1.0f / 16.0f);
    }
    for (index = 0; index < workspace->epsilon.shape.count; index++) {
        int centered = (int)(index % 11) - 5;

        workspace->epsilon.data[index] = 0.125f * (float)centered;
    }
}


static int allocate_fixture(
    const VAECProfile *profile,
    size_t batch_size,
    uint64_t seed,
    VAECModel *model,
    VAECModelWorkspace *workspace
)
{
    vae_c_model_init(model);
    vae_c_model_workspace_init(workspace);
    if (profile == NULL ||
        vae_c_model_allocate(model, profile, batch_size) != VAE_C_OP_OK ||
        vae_c_model_initialize_parameters_v1(model, seed) !=
            VAE_C_OP_OK ||
        vae_c_model_workspace_allocate(model, batch_size, workspace) !=
            VAE_C_OP_OK) {

        return 0;
    }
    fill_input_and_epsilon(workspace);
    return
        vae_c_model_validate(model) == VAE_C_OP_OK &&
        vae_c_model_workspace_validate(model, workspace) == VAE_C_OP_OK;
}


static void free_fixture(
    VAECModel *model,
    VAECModelWorkspace *workspace
)
{
    vae_c_model_workspace_free(workspace);
    vae_c_model_free(model);
}


static VAECOpStatus manual_vae_forward(
    VAECModel *model,
    VAECBatchNormMode batch_norm_mode,
    VAECLatentMode latent_mode,
    VAECModelWorkspace *workspace,
    VAECVAELossStats *stats
)
{
    const VAECTensor *epsilon =
        latent_mode == VAE_C_LATENT_MODE_STOCHASTIC
            ? &workspace->epsilon
            : NULL;
    VAECOpStatus status;

    status = vae_c_model_encoder_forward(
        model, batch_norm_mode, workspace
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_reparameterize_forward(
        latent_mode,
        &workspace->mu,
        &workspace->logvar,
        epsilon,
        &workspace->z
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_model_decoder_forward(
        model, batch_norm_mode, workspace
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_vae_loss_forward(
        &workspace->logits,
        &workspace->input,
        &workspace->mu,
        &workspace->logvar,
        stats
    );
    if (status == VAE_C_OP_OK &&
        batch_norm_mode == VAE_C_BATCH_NORM_MODE_TRAIN &&
        latent_mode == VAE_C_LATENT_MODE_STOCHASTIC) {

        workspace->vae_backward_model = model;
        workspace->vae_backward_generation =
            model->vae_cache_generation;
    }
    return status;
}


static void poison_epsilon(VAECModelWorkspace *workspace)
{
    size_t index;

    for (index = 0; index < workspace->epsilon.shape.count; index++) {
        workspace->epsilon.data[index] =
            index % 2 == 0 ? NAN : INFINITY;
    }
}


static int run_equivalence(
    const VAECProfile *profile,
    size_t batch_size,
    VAECBatchNormMode batch_norm_mode,
    VAECLatentMode latent_mode,
    int epsilon_is_poisoned
)
{
    const uint64_t seed = UINT64_C(0x123456789ABCDEF0);
    VAECModel manual_model;
    VAECModel composite_model;
    VAECModelWorkspace manual_workspace;
    VAECModelWorkspace composite_workspace;
    VAECVAELossStats manual_stats;
    VAECVAELossStats composite_stats;
    InvariantSnapshot invariant;
    RunningSnapshot running;
    int passed = 0;

    invariant_snapshot_init(&invariant);
    running_snapshot_init(&running);
    vae_c_model_init(&manual_model);
    vae_c_model_init(&composite_model);
    vae_c_model_workspace_init(&manual_workspace);
    vae_c_model_workspace_init(&composite_workspace);
    if (!allocate_fixture(profile,
                          batch_size,
                          seed,
                          &manual_model,
                          &manual_workspace) ||
        !allocate_fixture(profile,
                          batch_size,
                          seed,
                          &composite_model,
                          &composite_workspace)) {

        goto cleanup;
    }
    if (epsilon_is_poisoned) {
        poison_epsilon(&manual_workspace);
        poison_epsilon(&composite_workspace);
    }
    if (!invariant_snapshot_capture(
            &composite_model, &composite_workspace, &invariant) ||
        !running_snapshot_capture(&composite_model, &running) ||
        manual_vae_forward(
            &manual_model,
            batch_norm_mode,
            latent_mode,
            &manual_workspace,
            &manual_stats) != VAE_C_OP_OK ||
        vae_c_model_vae_forward(
            &composite_model,
            batch_norm_mode,
            latent_mode,
            &composite_workspace,
            &composite_stats) != VAE_C_OP_OK) {

        goto cleanup;
    }
    passed =
        model_equal(&manual_model, &composite_model) &&
        workspace_equal(
            &manual_model,
            &manual_workspace,
            &composite_model,
            &composite_workspace
        ) &&
        stats_equal(&manual_stats, &composite_stats) &&
        invariant_snapshot_matches(
            &composite_model, &composite_workspace, &invariant) &&
        composite_stats.beta_kl == VAE_C_VAE_LOSS_BETA_KL_V1 &&
        composite_stats.batch_size == batch_size &&
        composite_stats.total_sum ==
            composite_stats.reconstruction.total_sum +
            composite_stats.kl.total_sum &&
        isfinite(composite_stats.mean_per_sample) &&
        vae_c_model_validate(&composite_model) == VAE_C_OP_OK &&
        vae_c_model_workspace_validate(
            &composite_model, &composite_workspace) == VAE_C_OP_OK;
    if (batch_norm_mode == VAE_C_BATCH_NORM_MODE_EVAL) {
        passed = passed &&
            running_snapshot_matches(&composite_model, &running);
    } else {
        passed = passed &&
            !running_snapshot_matches(&composite_model, &running);
    }
    if (latent_mode == VAE_C_LATENT_MODE_MEAN) {
        passed = passed &&
            memcmp(composite_workspace.z.data,
                   composite_workspace.mu.data,
                   composite_workspace.z.shape.count * sizeof(float)) ==
                0;
    }

cleanup:
    running_snapshot_free(&running);
    invariant_snapshot_free(&invariant);
    free_fixture(&manual_model, &manual_workspace);
    free_fixture(&composite_model, &composite_workspace);
    return passed;
}


static void make_synthetic_depth4(VAECProfile *profile)
{
    memset(profile, 0, sizeof(*profile));
    memcpy(profile->profile_id,
           "synthetic_depth4_small",
           sizeof("synthetic_depth4_small"));
    memcpy(profile->role, "synthetic", sizeof("synthetic"));
    memcpy(profile->profile_status, "test_only", sizeof("test_only"));
    profile->input_side = 32;
    profile->conv_depth = 4;
    profile->channels[0] = 2;
    profile->channels[1] = 3;
    profile->channels[2] = 4;
    profile->channels[3] = 5;
    profile->latent_dim = 3;
    profile->bottleneck_side = 1;
    profile->expected_trainable_parameters_bn1d = UINT64_C(939);
    profile->expected_forward_mac_per_sample = UINT64_C(117297);
}


static int test_manual_stochastic_train_depth3(
    const VAECProfileCatalog *catalog
)
{
    return run_equivalence(
        vae_c_profile_catalog_find(catalog, "mini75_smoke_v1"),
        2,
        VAE_C_BATCH_NORM_MODE_TRAIN,
        VAE_C_LATENT_MODE_STOCHASTIC,
        0
    );
}


static int test_stochastic_eval_fixed_reproducible(
    const VAECProfileCatalog *catalog
)
{
    const VAECProfile *profile = vae_c_profile_catalog_find(
        catalog, "mini75_smoke_v1"
    );
    const uint64_t seed = UINT64_C(987654321);
    VAECModel first_model;
    VAECModel second_model;
    VAECModelWorkspace first_workspace;
    VAECModelWorkspace second_workspace;
    VAECVAELossStats first_stats;
    VAECVAELossStats second_stats;
    int passed;

    if (!run_equivalence(profile,
                         2,
                         VAE_C_BATCH_NORM_MODE_EVAL,
                         VAE_C_LATENT_MODE_STOCHASTIC,
                         0)) {

        return 0;
    }
    vae_c_model_init(&first_model);
    vae_c_model_init(&second_model);
    vae_c_model_workspace_init(&first_workspace);
    vae_c_model_workspace_init(&second_workspace);
    passed =
        allocate_fixture(profile,
                         2,
                         seed,
                         &first_model,
                         &first_workspace) &&
        allocate_fixture(profile,
                         2,
                         seed,
                         &second_model,
                         &second_workspace) &&
        vae_c_model_vae_forward(
            &first_model,
            VAE_C_BATCH_NORM_MODE_EVAL,
            VAE_C_LATENT_MODE_STOCHASTIC,
            &first_workspace,
            &first_stats) == VAE_C_OP_OK &&
        vae_c_model_vae_forward(
            &second_model,
            VAE_C_BATCH_NORM_MODE_EVAL,
            VAE_C_LATENT_MODE_STOCHASTIC,
            &second_workspace,
            &second_stats) == VAE_C_OP_OK &&
        model_equal(&first_model, &second_model) &&
        workspace_equal(
            &first_model,
            &first_workspace,
            &second_model,
            &second_workspace
        ) &&
        stats_equal(&first_stats, &second_stats);
    free_fixture(&first_model, &first_workspace);
    free_fixture(&second_model, &second_workspace);
    return passed;
}


static int test_mean_eval_ignores_epsilon_batch1(
    const VAECProfileCatalog *catalog
)
{
    return run_equivalence(
        vae_c_profile_catalog_find(catalog, "mini75_smoke_v1"),
        1,
        VAE_C_BATCH_NORM_MODE_EVAL,
        VAE_C_LATENT_MODE_MEAN,
        1
    );
}


static int test_manual_synthetic_depth4(void)
{
    VAECProfile profile;

    make_synthetic_depth4(&profile);
    return run_equivalence(
        &profile,
        2,
        VAE_C_BATCH_NORM_MODE_TRAIN,
        VAE_C_LATENT_MODE_STOCHASTIC,
        0
    );
}


static void set_stats_sentinel(VAECVAELossStats *stats)
{
    memset(stats, 0xA5, sizeof(*stats));
}


static int rejected_atomically(
    VAECModel *model,
    VAECBatchNormMode batch_norm_mode,
    VAECLatentMode latent_mode,
    VAECModelWorkspace *workspace,
    VAECOpStatus expected_status
)
{
    FullSnapshot snapshot;
    VAECVAELossStats stats;
    VAECVAELossStats stats_before;
    VAECOpStatus status;
    int passed;

    full_snapshot_init(&snapshot);
    set_stats_sentinel(&stats);
    stats_before = stats;
    if (!full_snapshot_capture(model, workspace, &snapshot)) {
        return 0;
    }
    status = vae_c_model_vae_forward(
        model,
        batch_norm_mode,
        latent_mode,
        workspace,
        &stats
    );
    passed = status == expected_status &&
        full_snapshot_matches(model, workspace, &snapshot) &&
        memcmp(&stats, &stats_before, sizeof(stats)) == 0;
    full_snapshot_free(&snapshot);
    return passed;
}


static int rejected_atomically_any_status(
    VAECModel *model,
    VAECBatchNormMode batch_norm_mode,
    VAECLatentMode latent_mode,
    VAECModelWorkspace *workspace
)
{
    FullSnapshot snapshot;
    VAECVAELossStats stats;
    VAECVAELossStats stats_before;
    VAECOpStatus status;
    int passed;

    full_snapshot_init(&snapshot);
    set_stats_sentinel(&stats);
    stats_before = stats;
    if (!full_snapshot_capture(model, workspace, &snapshot)) {
        return 0;
    }
    status = vae_c_model_vae_forward(
        model,
        batch_norm_mode,
        latent_mode,
        workspace,
        &stats
    );
    passed = status != VAE_C_OP_OK &&
        full_snapshot_matches(model, workspace, &snapshot) &&
        memcmp(&stats, &stats_before, sizeof(stats)) == 0;
    full_snapshot_free(&snapshot);
    return passed;
}


static int test_modes_null_and_batch1_preflight(
    const VAECProfileCatalog *catalog
)
{
    const VAECProfile *profile = vae_c_profile_catalog_find(
        catalog, "mini75_smoke_v1"
    );
    VAECModel model;
    VAECModelWorkspace workspace;
    VAECVAELossStats stats;
    FullSnapshot snapshot;
    int passed;

    full_snapshot_init(&snapshot);
    if (!allocate_fixture(
            profile, 2, UINT64_C(101), &model, &workspace)) {

        return 0;
    }
    passed =
        rejected_atomically(
            &model,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            VAE_C_LATENT_MODE_MEAN,
            &workspace,
            VAE_C_OP_INVALID_ARGUMENT) &&
        rejected_atomically(
            &model,
            (VAECBatchNormMode)0,
            VAE_C_LATENT_MODE_STOCHASTIC,
            &workspace,
            VAE_C_OP_INVALID_ARGUMENT) &&
        rejected_atomically(
            &model,
            VAE_C_BATCH_NORM_MODE_EVAL,
            (VAECLatentMode)0,
            &workspace,
            VAE_C_OP_INVALID_ARGUMENT) &&
        full_snapshot_capture(&model, &workspace, &snapshot);
    set_stats_sentinel(&stats);
    if (passed) {
        passed =
            vae_c_model_vae_forward(
                NULL,
                VAE_C_BATCH_NORM_MODE_EVAL,
                VAE_C_LATENT_MODE_MEAN,
                &workspace,
                &stats) == VAE_C_OP_INVALID_ARGUMENT &&
            vae_c_model_vae_forward(
                &model,
                VAE_C_BATCH_NORM_MODE_EVAL,
                VAE_C_LATENT_MODE_MEAN,
                NULL,
                &stats) == VAE_C_OP_INVALID_ARGUMENT &&
            vae_c_model_vae_forward(
                &model,
                VAE_C_BATCH_NORM_MODE_EVAL,
                VAE_C_LATENT_MODE_MEAN,
                &workspace,
                NULL) == VAE_C_OP_INVALID_ARGUMENT &&
            full_snapshot_matches(&model, &workspace, &snapshot);
    }
    full_snapshot_free(&snapshot);
    free_fixture(&model, &workspace);
    if (!passed || !allocate_fixture(
            profile, 1, UINT64_C(102), &model, &workspace)) {

        return 0;
    }
    passed = rejected_atomically(
        &model,
        VAE_C_BATCH_NORM_MODE_TRAIN,
        VAE_C_LATENT_MODE_STOCHASTIC,
        &workspace,
        VAE_C_OP_INVALID_STATE
    );
    free_fixture(&model, &workspace);
    return passed;
}


static int test_invalid_state_input_epsilon_uninitialized(
    const VAECProfileCatalog *catalog
)
{
    const VAECProfile *profile = vae_c_profile_catalog_find(
        catalog, "mini75_smoke_v1"
    );
    const float bad_input[] = { NAN, INFINITY, -0.01f, 1.01f };
    const float bad_epsilon[] = { NAN, INFINITY };
    VAECModel model;
    VAECModelWorkspace workspace;
    size_t index;
    int passed = 1;

    if (!allocate_fixture(
            profile, 2, UINT64_C(201), &model, &workspace)) {

        return 0;
    }
    for (index = 0;
         passed && index < sizeof(bad_input) / sizeof(bad_input[0]);
         index++) {

        float saved = workspace.input.data[0];

        workspace.input.data[0] = bad_input[index];
        passed = rejected_atomically(
            &model,
            VAE_C_BATCH_NORM_MODE_EVAL,
            VAE_C_LATENT_MODE_STOCHASTIC,
            &workspace,
            VAE_C_OP_INVALID_STATE
        );
        workspace.input.data[0] = saved;
    }
    for (index = 0;
         passed &&
         index < sizeof(bad_epsilon) / sizeof(bad_epsilon[0]);
         index++) {

        float saved = workspace.epsilon.data[0];

        workspace.epsilon.data[0] = bad_epsilon[index];
        passed = rejected_atomically(
            &model,
            VAE_C_BATCH_NORM_MODE_EVAL,
            VAE_C_LATENT_MODE_STOCHASTIC,
            &workspace,
            VAE_C_OP_INVALID_STATE
        );
        workspace.epsilon.data[0] = saved;
    }
    free_fixture(&model, &workspace);

    vae_c_model_init(&model);
    vae_c_model_workspace_init(&workspace);
    if (!passed || profile == NULL ||
        vae_c_model_allocate(&model, profile, 2) != VAE_C_OP_OK ||
        vae_c_model_workspace_allocate(&model, 2, &workspace) !=
            VAE_C_OP_OK) {

        free_fixture(&model, &workspace);
        return 0;
    }
    fill_input_and_epsilon(&workspace);
    passed = rejected_atomically(
        &model,
        VAE_C_BATCH_NORM_MODE_EVAL,
        VAE_C_LATENT_MODE_STOCHASTIC,
        &workspace,
        VAE_C_OP_INVALID_STATE
    );
    free_fixture(&model, &workspace);
    return passed;
}


static int test_shape_capacity_overflow_and_aliases(
    const VAECProfileCatalog *catalog
)
{
    const VAECProfile *profile = vae_c_profile_catalog_find(
        catalog, "mini75_smoke_v1"
    );
    VAECModel model;
    VAECModelWorkspace workspace;
    VAECVAELossStats *aliased_stats;
    FullSnapshot snapshot;
    VAECTensor saved_epsilon;
    VAECShape saved_shape;
    size_t saved_capacity;
    size_t saved_target;
    VAECOpStatus status;
    int passed;

    full_snapshot_init(&snapshot);
    if (!allocate_fixture(
            profile, 2, UINT64_C(301), &model, &workspace)) {

        return 0;
    }
    aliased_stats = (VAECVAELossStats *)(void *)&workspace.z;
    passed = full_snapshot_capture(&model, &workspace, &snapshot);
    status = passed ? vae_c_model_vae_forward(
        &model,
        VAE_C_BATCH_NORM_MODE_EVAL,
        VAE_C_LATENT_MODE_STOCHASTIC,
        &workspace,
        aliased_stats
    ) : VAE_C_OP_OK;
    passed = passed && status == VAE_C_OP_INVALID_ARGUMENT &&
        full_snapshot_matches(&model, &workspace, &snapshot);
    full_snapshot_free(&snapshot);

    saved_target = workspace.config.decoder_upsample_targets[0];
    workspace.config.decoder_upsample_targets[0] = SIZE_MAX;
    passed = passed && rejected_atomically_any_status(
        &model,
        VAE_C_BATCH_NORM_MODE_EVAL,
        VAE_C_LATENT_MODE_STOCHASTIC,
        &workspace
    );
    workspace.config.decoder_upsample_targets[0] = saved_target;

    saved_capacity = model.encoder_max_pool.capacity;
    model.encoder_max_pool.capacity--;
    passed = passed && rejected_atomically_any_status(
        &model,
        VAE_C_BATCH_NORM_MODE_EVAL,
        VAE_C_LATENT_MODE_STOCHASTIC,
        &workspace
    );
    model.encoder_max_pool.capacity = saved_capacity;

    saved_shape = workspace.encoder_batch_norm_outputs[0].shape;
    workspace.encoder_batch_norm_outputs[0].shape.dims[2] = SIZE_MAX;
    passed = passed && rejected_atomically_any_status(
        &model,
        VAE_C_BATCH_NORM_MODE_EVAL,
        VAE_C_LATENT_MODE_STOCHASTIC,
        &workspace
    );
    workspace.encoder_batch_norm_outputs[0].shape = saved_shape;

    saved_epsilon = workspace.epsilon;
    if (model.encoder_fc.bias.shape.count !=
            workspace.epsilon.shape.count) {

        passed = 0;
    } else {
        workspace.epsilon.data = model.encoder_fc.bias.values;
        workspace.epsilon.owned_block = workspace.epsilon.data;
        passed = passed && rejected_atomically(
            &model,
            VAE_C_BATCH_NORM_MODE_EVAL,
            VAE_C_LATENT_MODE_STOCHASTIC,
            &workspace,
            VAE_C_OP_INVALID_ARGUMENT
        );
        workspace.epsilon = saved_epsilon;
    }
    passed = passed &&
        vae_c_model_validate(&model) == VAE_C_OP_OK &&
        vae_c_model_workspace_validate(&model, &workspace) ==
            VAE_C_OP_OK;
    free_fixture(&model, &workspace);
    return passed;
}


static int test_model_model_alias_preflight_atomic(
    const VAECProfileCatalog *catalog
)
{
    const VAECProfile *profile = vae_c_profile_catalog_find(
        catalog, "mini75_smoke_v1"
    );
    VAECModel model;
    VAECModelWorkspace workspace;
    VAECVAELossStats stats;
    VAECVAELossStats stats_before;
    FullSnapshot snapshot;
    size_t *saved_argmax_indices;
    size_t source_bytes;
    size_t alias_count;
    VAECOpStatus status;
    int atomic_rejection;
    int passed;

    full_snapshot_init(&snapshot);
    if (!allocate_fixture(
            profile, 2, UINT64_C(351), &model, &workspace) ||
        !size_multiply(
            model.decoder_fc.weights.shape.count,
            sizeof(float),
            &source_bytes) ||
        source_bytes < model.encoder_max_pool.owned_bytes ||
        model.encoder_max_pool.cache_valid ||
        count_model_buffer_aliases(&model) != 0) {

        free_fixture(&model, &workspace);
        return 0;
    }

    saved_argmax_indices = model.encoder_max_pool.argmax_indices;
    model.encoder_max_pool.argmax_indices =
        (size_t *)(void *)model.decoder_fc.weights.values;
    alias_count = count_model_buffer_aliases(&model);
    set_stats_sentinel(&stats);
    stats_before = stats;
    passed =
        alias_count == 1 &&
        vae_c_model_validate(&model) == VAE_C_OP_OK &&
        vae_c_model_workspace_validate(&model, &workspace) ==
            VAE_C_OP_OK &&
        full_snapshot_capture(&model, &workspace, &snapshot);
    status = passed
        ? vae_c_model_vae_forward(
            &model,
            VAE_C_BATCH_NORM_MODE_EVAL,
            VAE_C_LATENT_MODE_STOCHASTIC,
            &workspace,
            &stats)
        : VAE_C_OP_OK;
    atomic_rejection =
        passed &&
        status == VAE_C_OP_INVALID_ARGUMENT &&
        count_model_buffer_aliases(&model) == 1 &&
        full_snapshot_matches(&model, &workspace, &snapshot) &&
        memcmp(&stats, &stats_before, sizeof(stats)) == 0;

    model.encoder_max_pool.argmax_indices = saved_argmax_indices;
    passed =
        atomic_rejection &&
        count_model_buffer_aliases(&model) == 0 &&
        vae_c_model_validate(&model) == VAE_C_OP_OK &&
        vae_c_model_workspace_validate(&model, &workspace) ==
            VAE_C_OP_OK;
    full_snapshot_free(&snapshot);
    free_fixture(&model, &workspace);
    return passed;
}


static int run_stats_lifecycle_alias_case(
    const VAECProfile *profile,
    int alias_case
)
{
    const uint64_t generation = UINT64_C(37);
    VAECModel model;
    VAECModelWorkspace workspace;
    VAECVAELossStats ordinary_stats;
    VAECVAELossStats *aliased_stats = &ordinary_stats;
    FullSnapshot snapshot;
    VAECOpStatus status;
    int passed = 0;

    full_snapshot_init(&snapshot);
    if (!allocate_fixture(
            profile,
            2,
            UINT64_C(371) + (uint64_t)alias_case,
            &model,
            &workspace)) {

        return 0;
    }
    model.vae_cache_generation = generation;
    workspace.vae_backward_model = &model;
    workspace.vae_backward_generation = generation;

    if (alias_case == 0) {
        aliased_stats = (VAECVAELossStats *)(void *)
            &model.vae_cache_generation;
    } else if (alias_case == 1) {
        aliased_stats = (VAECVAELossStats *)(void *)
            &workspace.vae_backward_model;
    } else if (alias_case == 2) {
        aliased_stats = (VAECVAELossStats *)(void *)
            &workspace.vae_backward_generation;
    } else {
        goto cleanup;
    }

    if (count_stats_lifecycle_aliases(
            aliased_stats, &model, &workspace) != 1 ||
        stats_matches_pre_lifecycle_alias(
            aliased_stats, &model, &workspace) ||
        vae_c_model_validate(&model) != VAE_C_OP_OK ||
        vae_c_model_workspace_validate(&model, &workspace) !=
            VAE_C_OP_OK ||
        !full_snapshot_capture(&model, &workspace, &snapshot)) {

        goto cleanup;
    }

    status = vae_c_model_vae_forward(
        &model,
        VAE_C_BATCH_NORM_MODE_TRAIN,
        VAE_C_LATENT_MODE_STOCHASTIC,
        &workspace,
        aliased_stats
    );
    aliased_stats = &ordinary_stats;
    passed =
        status == VAE_C_OP_INVALID_ARGUMENT &&
        count_stats_lifecycle_aliases(
            aliased_stats, &model, &workspace) == 0 &&
        full_snapshot_matches(&model, &workspace, &snapshot) &&
        model.vae_cache_generation == generation &&
        workspace.vae_backward_model == &model &&
        workspace.vae_backward_generation == generation &&
        vae_c_model_validate(&model) == VAE_C_OP_OK &&
        vae_c_model_workspace_validate(&model, &workspace) ==
            VAE_C_OP_OK;

cleanup:
    full_snapshot_free(&snapshot);
    free_fixture(&model, &workspace);
    return passed;
}


static int test_stats_lifecycle_alias_preflight_atomic(
    const VAECProfileCatalog *catalog,
    int alias_case
)
{
    const VAECProfile *profile = vae_c_profile_catalog_find(
        catalog, "mini75_smoke_v1"
    );

    return profile != NULL &&
        run_stats_lifecycle_alias_case(profile, alias_case);
}


static int test_post_start_failure_no_global_rollback(
    const VAECProfileCatalog *catalog
)
{
    const VAECProfile *profile = vae_c_profile_catalog_find(
        catalog, "mini75_smoke_v1"
    );
    VAECModel model;
    VAECModelWorkspace workspace;
    VAECVAELossStats stats;
    VAECVAELossStats stats_before;
    FullSnapshot full;
    InvariantSnapshot invariant;
    float *z_before;
    size_t latent_dim;
    size_t index;
    VAECOpStatus status;
    int passed;

    full_snapshot_init(&full);
    invariant_snapshot_init(&invariant);
    if (!allocate_fixture(
            profile, 2, UINT64_C(401), &model, &workspace)) {

        return 0;
    }
    memset(model.encoder_fc.weights.values,
           0,
           model.encoder_fc.weights.shape.count * sizeof(float));
    latent_dim = model.config.latent_dim;
    for (index = 0; index < model.encoder_fc.bias.shape.count; index++) {
        model.encoder_fc.bias.values[index] =
            index < latent_dim ? 0.0f : 200.0f;
    }
    z_before = (float *)malloc(
        workspace.z.shape.count * sizeof(float)
    );
    if (z_before == NULL ||
        vae_c_model_validate(&model) != VAE_C_OP_OK ||
        !full_snapshot_capture(&model, &workspace, &full) ||
        !invariant_snapshot_capture(
            &model, &workspace, &invariant)) {

        free(z_before);
        full_snapshot_free(&full);
        invariant_snapshot_free(&invariant);
        free_fixture(&model, &workspace);
        return 0;
    }
    memcpy(z_before,
           workspace.z.data,
           workspace.z.shape.count * sizeof(float));
    set_stats_sentinel(&stats);
    stats_before = stats;
    status = vae_c_model_vae_forward(
        &model,
        VAE_C_BATCH_NORM_MODE_EVAL,
        VAE_C_LATENT_MODE_STOCHASTIC,
        &workspace,
        &stats
    );
    passed =
        status == VAE_C_OP_INVALID_STATE &&
        memcmp(&stats, &stats_before, sizeof(stats)) == 0 &&
        memcmp(z_before,
               workspace.z.data,
               workspace.z.shape.count * sizeof(float)) == 0 &&
        invariant_snapshot_matches(&model, &workspace, &invariant) &&
        !full_snapshot_matches(&model, &workspace, &full) &&
        model.encoder_max_pool.cache_valid;
    free(z_before);
    full_snapshot_free(&full);
    invariant_snapshot_free(&invariant);
    free_fixture(&model, &workspace);
    return passed;
}


static int backward_ready(
    const VAECModel *model,
    const VAECModelWorkspace *workspace
)
{
    return model != NULL && workspace != NULL &&
        workspace->vae_backward_model == model &&
        workspace->vae_backward_generation != UINT64_C(0) &&
        workspace->vae_backward_generation ==
            model->vae_cache_generation;
}


static int test_backward_ready_lifecycle_and_overflow(
    const VAECProfileCatalog *catalog
)
{
    const VAECProfile *profile = vae_c_profile_catalog_find(
        catalog, "mini75_smoke_v1"
    );
    VAECModel model;
    VAECModel other_model;
    VAECModelWorkspace first;
    VAECModelWorkspace second;
    VAECModelWorkspace other_workspace;
    VAECVAELossStats stats;
    VAECVAELossStats stats_before;
    FullSnapshot snapshot;
    float *bias_copy = NULL;
    size_t bias_bytes;
    size_t index;
    VAECOpStatus status;
    int passed = 0;

    full_snapshot_init(&snapshot);
    vae_c_model_init(&model);
    vae_c_model_init(&other_model);
    vae_c_model_workspace_init(&first);
    vae_c_model_workspace_init(&second);
    vae_c_model_workspace_init(&other_workspace);
    if (!allocate_fixture(
            profile, 2, UINT64_C(5001), &model, &first) ||
        vae_c_model_workspace_allocate(&model, 2, &second) !=
            VAE_C_OP_OK ||
        !allocate_fixture(
            profile,
            2,
            UINT64_C(5001),
            &other_model,
            &other_workspace)) {

        goto cleanup;
    }
    fill_input_and_epsilon(&second);

    status = vae_c_model_vae_forward(
        &model,
        VAE_C_BATCH_NORM_MODE_TRAIN,
        VAE_C_LATENT_MODE_STOCHASTIC,
        &first,
        &stats
    );
    if (status != VAE_C_OP_OK ||
        model.vae_cache_generation != UINT64_C(2) ||
        !backward_ready(&model, &first)) {

        goto cleanup;
    }
    other_model.vae_cache_generation = model.vae_cache_generation;
    if (backward_ready(&other_model, &first)) {
        goto cleanup;
    }

    first.input.grad[0] = 1.0f;
    if (vae_c_model_workspace_zero_grad(&model, &first) != VAE_C_OP_OK ||
        first.input.grad[0] != 0.0f || !backward_ready(&model, &first)) {

        goto cleanup;
    }

    set_stats_sentinel(&stats);
    stats_before = stats;
    if (!full_snapshot_capture(&model, &first, &snapshot) ||
        vae_c_model_vae_forward(
            &model,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            VAE_C_LATENT_MODE_MEAN,
            &first,
            &stats) != VAE_C_OP_INVALID_ARGUMENT ||
        !full_snapshot_matches(&model, &first, &snapshot) ||
        memcmp(&stats, &stats_before, sizeof(stats)) != 0 ||
        !backward_ready(&model, &first)) {

        goto cleanup;
    }

    if (vae_c_model_vae_forward(
            &model,
            VAE_C_BATCH_NORM_MODE_EVAL,
            VAE_C_LATENT_MODE_STOCHASTIC,
            &second,
            &stats) != VAE_C_OP_OK ||
        model.vae_cache_generation != UINT64_C(4) ||
        backward_ready(&model, &first) ||
        backward_ready(&model, &second) ||
        first.vae_backward_model != &model ||
        first.vae_backward_generation != UINT64_C(2) ||
        vae_c_model_workspace_validate(&model, &first) != VAE_C_OP_OK) {

        goto cleanup;
    }
    if (vae_c_model_vae_forward(
            &model,
            VAE_C_BATCH_NORM_MODE_EVAL,
            VAE_C_LATENT_MODE_MEAN,
            &second,
            &stats) != VAE_C_OP_OK ||
        model.vae_cache_generation != UINT64_C(6) ||
        backward_ready(&model, &second)) {

        goto cleanup;
    }

    if (vae_c_model_vae_forward(
            &model,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            VAE_C_LATENT_MODE_STOCHASTIC,
            &first,
            &stats) != VAE_C_OP_OK ||
        model.vae_cache_generation != UINT64_C(8) ||
        !backward_ready(&model, &first)) {

        goto cleanup;
    }

    bias_bytes = model.encoder_fc.bias.shape.count * sizeof(float);
    bias_copy = (float *)malloc(bias_bytes);
    if (bias_copy == NULL) {
        goto cleanup;
    }
    memcpy(bias_copy, model.encoder_fc.bias.values, bias_bytes);
    for (index = 0; index < model.encoder_fc.bias.shape.count; index++) {
        model.encoder_fc.bias.values[index] =
            index < model.config.latent_dim ? 0.0f : 200.0f;
    }
    status = vae_c_model_vae_forward(
        &model,
        VAE_C_BATCH_NORM_MODE_EVAL,
        VAE_C_LATENT_MODE_STOCHASTIC,
        &first,
        &stats
    );
    memcpy(model.encoder_fc.bias.values, bias_copy, bias_bytes);
    if (status != VAE_C_OP_INVALID_STATE ||
        model.vae_cache_generation != UINT64_C(9) ||
        backward_ready(&model, &first) ||
        first.vae_backward_model != NULL ||
        first.vae_backward_generation != UINT64_C(0)) {

        goto cleanup;
    }

    model.vae_cache_generation = UINT64_MAX;
    first.vae_backward_model = &model;
    first.vae_backward_generation = UINT64_MAX;
    set_stats_sentinel(&stats);
    stats_before = stats;
    full_snapshot_free(&snapshot);
    if (!full_snapshot_capture(&model, &first, &snapshot) ||
        vae_c_model_vae_forward(
            &model,
            VAE_C_BATCH_NORM_MODE_EVAL,
            VAE_C_LATENT_MODE_STOCHASTIC,
            &first,
            &stats) != VAE_C_OP_OVERFLOW ||
        !full_snapshot_matches(&model, &first, &snapshot) ||
        memcmp(&stats, &stats_before, sizeof(stats)) != 0) {

        goto cleanup;
    }

    model.vae_cache_generation = UINT64_MAX - UINT64_C(1);
    first.vae_backward_model = &model;
    first.vae_backward_generation = model.vae_cache_generation;
    set_stats_sentinel(&stats);
    stats_before = stats;
    full_snapshot_free(&snapshot);
    if (!full_snapshot_capture(&model, &first, &snapshot) ||
        vae_c_model_vae_forward(
            &model,
            VAE_C_BATCH_NORM_MODE_EVAL,
            VAE_C_LATENT_MODE_STOCHASTIC,
            &first,
            &stats) != VAE_C_OP_OVERFLOW ||
        !full_snapshot_matches(&model, &first, &snapshot) ||
        memcmp(&stats, &stats_before, sizeof(stats)) != 0) {

        goto cleanup;
    }

    model.vae_cache_generation = UINT64_MAX - UINT64_C(2);
    first.vae_backward_model = &model;
    first.vae_backward_generation = model.vae_cache_generation;
    if (vae_c_model_vae_forward(
            &model,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            VAE_C_LATENT_MODE_STOCHASTIC,
            &first,
            &stats) != VAE_C_OP_OK ||
        model.vae_cache_generation != UINT64_MAX ||
        !backward_ready(&model, &first)) {

        goto cleanup;
    }
    passed = 1;

cleanup:
    free(bias_copy);
    full_snapshot_free(&snapshot);
    vae_c_model_workspace_free(&second);
    free_fixture(&model, &first);
    free_fixture(&other_model, &other_workspace);
    return passed;
}


int main(int argc, char **argv)
{
    VAECProfileCatalog catalog;
    int failures = 0;

    if (argc != 2) {
        fprintf(stderr, "usage: %s <vae_c_profiles_v1.csv>\n", argv[0]);
        return 2;
    }
    vae_c_profile_catalog_init(&catalog);
    if (vae_c_profile_catalog_load_csv(
            argv[1], &catalog, NULL, 0) != VAE_C_CONFIG_OK) {

        fprintf(stderr, "failed to load profile catalog\n");
        return 2;
    }

    failures += report_check(
        "manual_stochastic_train_depth3_loss_and_boundaries",
        test_manual_stochastic_train_depth3(&catalog)
    );
    failures += report_check(
        "stochastic_eval_fixed_epsilon_manual_and_reproducible",
        test_stochastic_eval_fixed_reproducible(&catalog)
    );
    failures += report_check(
        "mean_eval_batch1_z_equals_mu_and_epsilon_ignored",
        test_mean_eval_ignores_epsilon_batch1(&catalog)
    );
    failures += report_check(
        "manual_stochastic_train_synthetic_depth4",
        test_manual_synthetic_depth4()
    );
    failures += report_check(
        "operational_modes_null_and_train_batch1_preflight_atomic",
        test_modes_null_and_batch1_preflight(&catalog)
    );
    failures += report_check(
        "input_epsilon_and_uninitialized_preflight_atomic",
        test_invalid_state_input_epsilon_uninitialized(&catalog)
    );
    failures += report_check(
        "shape_capacity_overflow_and_exact_alias_preflight_atomic",
        test_shape_capacity_overflow_and_aliases(&catalog)
    );
    failures += report_check(
        "model_model_exact_alias_preflight_atomic",
        test_model_model_alias_preflight_atomic(&catalog)
    );
    failures += report_check(
        "stats_alias_model_cache_generation_preflight_atomic",
        test_stats_lifecycle_alias_preflight_atomic(&catalog, 0)
    );
    failures += report_check(
        "stats_alias_workspace_backward_model_preflight_atomic",
        test_stats_lifecycle_alias_preflight_atomic(&catalog, 1)
    );
    failures += report_check(
        "stats_alias_workspace_backward_generation_preflight_atomic",
        test_stats_lifecycle_alias_preflight_atomic(&catalog, 2)
    );
    failures += report_check(
        "post_start_failure_has_no_global_rollback",
        test_post_start_failure_no_global_rollback(&catalog)
    );
    failures += report_check(
        "backward_ready_lifecycle_cross_workspace_and_overflow",
        test_backward_ready_lifecycle_and_overflow(&catalog)
    );

    vae_c_profile_catalog_free(&catalog);
    if (failures != 0) {
        fprintf(stderr,
                "%d VAE C FULL FORWARD TEST(S) FAILED\n",
                failures);
        return 1;
    }
    printf("VAE_FORWARD_STRONG_GUARANTEE=preflight_only\n");
    printf("VAE_FORWARD_POST_START_ROLLBACK=false\n");
    printf("VAE_FORWARD_RNG_INTERNAL=false\n");
    printf("VAE_FORWARD_BETA_KL=1\n");
    printf("ALL VAE C MODEL FULL FORWARD TESTS PASSED\n");
    return 0;
}
