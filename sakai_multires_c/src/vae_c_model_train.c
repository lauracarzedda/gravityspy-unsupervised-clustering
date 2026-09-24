#include "vae_c_model_train.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


#define TRAIN_BATCH_NORM_CAPACITY \
    (VAE_C_MAX_CONV_DEPTH + 1 + \
     VAE_C_MAX_DECODER_HIDDEN_CONVS)

#define TRAIN_MODEL_BUFFER_CAPACITY \
    (4 * VAE_C_MODEL_MAX_PARAMETERS + \
     7 * TRAIN_BATCH_NORM_CAPACITY + 1)

#define TRAIN_WORKSPACE_BUFFER_CAPACITY \
    (2 * VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS)


static int train_size_multiply(
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


static VAECOpStatus train_append_pointer(
    const void **pointers,
    size_t capacity,
    size_t *count,
    const void *pointer
)
{
    if (pointers == NULL || count == NULL || pointer == NULL) {
        return VAE_C_OP_INVALID_STATE;
    }
    if (*count >= capacity) {
        return VAE_C_OP_OVERFLOW;
    }
    pointers[*count] = pointer;
    *count += 1;
    return VAE_C_OP_OK;
}


static VAECOpStatus train_collect_batch_norms(
    const VAECModel *model,
    const VAECBatchNorm **layers,
    size_t capacity,
    size_t *count
)
{
    size_t index;

    if (model == NULL || layers == NULL || count == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    *count = 0;
    for (index = 0; index < model->config.conv_depth; index++) {
        if (*count >= capacity) {
            return VAE_C_OP_OVERFLOW;
        }
        layers[(*count)++] = &model->encoder_batch_norms[index];
    }
    if (*count >= capacity) {
        return VAE_C_OP_OVERFLOW;
    }
    layers[(*count)++] = &model->decoder_fc_batch_norm;
    for (index = 0;
         index < model->config.decoder_hidden_conv_count;
         index++) {

        if (*count >= capacity) {
            return VAE_C_OP_OVERFLOW;
        }
        layers[(*count)++] =
            &model->decoder_hidden_batch_norms[index];
    }
    return VAE_C_OP_OK;
}


static VAECOpStatus train_collect_model_buffers(
    const VAECModel *model,
    const void **buffers,
    size_t capacity,
    size_t *count
)
{
    const VAECBatchNorm *batch_norms[TRAIN_BATCH_NORM_CAPACITY];
    size_t batch_norm_count = 0;
    size_t index;
    VAECOpStatus status;

    if (model == NULL || buffers == NULL || count == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    *count = 0;
    for (index = 0; index < model->parameter_count; index++) {
        const VAECParameter *parameter = model->parameters[index];
        const void *regions[4];
        size_t region;

        if (parameter == NULL) {
            return VAE_C_OP_INVALID_STATE;
        }
        regions[0] = parameter->values;
        regions[1] = parameter->gradients;
        regions[2] = parameter->adam_m;
        regions[3] = parameter->adam_v;
        for (region = 0; region < 4; region++) {
            status = train_append_pointer(
                buffers, capacity, count, regions[region]
            );
            if (status != VAE_C_OP_OK) {
                return status;
            }
        }
    }

    status = train_collect_batch_norms(
        model,
        batch_norms,
        sizeof(batch_norms) / sizeof(batch_norms[0]),
        &batch_norm_count
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    for (index = 0; index < batch_norm_count; index++) {
        const VAECBatchNorm *layer = batch_norms[index];
        const void *regions[7] = {
            layer->running_mean,
            layer->running_variance,
            layer->batch_mean,
            layer->batch_inv_std,
            layer->work_mean,
            layer->work_inv_std,
            layer->work_unbiased_variance
        };
        size_t region;

        for (region = 0; region < 7; region++) {
            status = train_append_pointer(
                buffers, capacity, count, regions[region]
            );
            if (status != VAE_C_OP_OK) {
                return status;
            }
        }
    }
    return train_append_pointer(
        buffers,
        capacity,
        count,
        model->encoder_max_pool.argmax_indices
    );
}


static VAECOpStatus train_collect_workspace_tensors(
    const VAECModelWorkspace *workspace,
    const VAECTensor **tensors,
    size_t capacity,
    size_t *count
)
{
    size_t index;

#define TRAIN_APPEND_TENSOR(tensor_pointer) \
    do { \
        if (*count >= capacity) { \
            return VAE_C_OP_OVERFLOW; \
        } \
        tensors[*count] = (tensor_pointer); \
        *count += 1; \
    } while (0)

    if (workspace == NULL || tensors == NULL || count == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    *count = 0;
    TRAIN_APPEND_TENSOR(&workspace->input);
    for (index = 0; index < workspace->config.conv_depth; index++) {
        TRAIN_APPEND_TENSOR(&workspace->encoder_conv_outputs[index]);
        TRAIN_APPEND_TENSOR(
            &workspace->encoder_batch_norm_outputs[index]
        );
        TRAIN_APPEND_TENSOR(&workspace->encoder_relu_outputs[index]);
    }
    TRAIN_APPEND_TENSOR(&workspace->encoder_pool_output);
    TRAIN_APPEND_TENSOR(&workspace->encoder_global_avg_output);
    TRAIN_APPEND_TENSOR(&workspace->encoder_packed_output);
    TRAIN_APPEND_TENSOR(&workspace->mu);
    TRAIN_APPEND_TENSOR(&workspace->logvar);
    TRAIN_APPEND_TENSOR(&workspace->epsilon);
    TRAIN_APPEND_TENSOR(&workspace->z);
    TRAIN_APPEND_TENSOR(&workspace->decoder_fc_output);
    TRAIN_APPEND_TENSOR(&workspace->decoder_fc_batch_norm_output);
    TRAIN_APPEND_TENSOR(&workspace->decoder_fc_relu_output);
    for (index = 0;
         index < workspace->config.decoder_upsample_count;
         index++) {

        TRAIN_APPEND_TENSOR(&workspace->decoder_upsample_outputs[index]);
    }
    for (index = 0;
         index < workspace->config.decoder_hidden_conv_count;
         index++) {

        TRAIN_APPEND_TENSOR(
            &workspace->decoder_hidden_conv_outputs[index]
        );
        TRAIN_APPEND_TENSOR(
            &workspace->decoder_hidden_batch_norm_outputs[index]
        );
        TRAIN_APPEND_TENSOR(
            &workspace->decoder_hidden_relu_outputs[index]
        );
    }
    TRAIN_APPEND_TENSOR(&workspace->logits);

#undef TRAIN_APPEND_TENSOR

    return VAE_C_OP_OK;
}


static VAECOpStatus train_collect_workspace_buffers(
    const VAECModelWorkspace *workspace,
    const void **buffers,
    size_t capacity,
    size_t *count
)
{
    const VAECTensor *tensors[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ];
    size_t tensor_count = 0;
    size_t index;
    VAECOpStatus status;

    if (workspace == NULL || buffers == NULL || count == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    *count = 0;
    status = train_collect_workspace_tensors(
        workspace,
        tensors,
        sizeof(tensors) / sizeof(tensors[0]),
        &tensor_count
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    for (index = 0; index < tensor_count; index++) {
        status = train_append_pointer(
            buffers, capacity, count, tensors[index]->data
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
        if (tensors[index]->grad != NULL) {
            status = train_append_pointer(
                buffers, capacity, count, tensors[index]->grad
            );
            if (status != VAE_C_OP_OK) {
                return status;
            }
        }
    }
    return VAE_C_OP_OK;
}


static bool train_pointer_aliases_shape_metadata(
    const void *pointer,
    const VAECShape *shape
)
{
    size_t index;

    if (pointer == (const void *)shape ||
        pointer == (const void *)&shape->rank ||
        pointer == (const void *)&shape->count) {

        return true;
    }
    for (index = 0; index < VAE_C_TENSOR_MAX_RANK; index++) {
        if (pointer == (const void *)&shape->dims[index] ||
            pointer == (const void *)&shape->strides[index]) {

            return true;
        }
    }
    return false;
}


static bool train_pointer_aliases_config_metadata(
    const void *pointer,
    const VAECConfig *config
)
{
    size_t index;

    if (pointer == (const void *)config ||
        pointer == (const void *)&config->input_channels ||
        pointer == (const void *)&config->input_side ||
        pointer == (const void *)&config->conv_depth ||
        pointer == (const void *)&config->pool_side ||
        pointer == (const void *)&config->latent_dim ||
        pointer == (const void *)&config->bottleneck_side ||
        pointer == (const void *)&config->decoder_fc_features ||
        pointer == (const void *)&config->decoder_upsample_count ||
        pointer == (const void *)&config->decoder_hidden_conv_count ||
        pointer == (const void *)&config->values_per_sample ||
        pointer ==
            (const void *)&config->trainable_parameters_bn1d ||
        pointer == (const void *)&config->forward_mac_per_sample) {

        return true;
    }
    for (index = 0; index < VAE_C_MAX_CONV_DEPTH; index++) {
        if (pointer == (const void *)&config->channels[index] ||
            pointer ==
                (const void *)&config->encoder_conv_sides[index]) {

            return true;
        }
    }
    for (index = 0; index < VAE_C_MAX_DECODER_UPSAMPLES; index++) {
        if (pointer ==
                (const void *)&config->decoder_upsample_targets[index]) {

            return true;
        }
    }
    for (index = 0;
         index < VAE_C_MAX_DECODER_HIDDEN_CONVS;
         index++) {

        if (pointer ==
                (const void *)&config->decoder_hidden_conv_sides[index]) {

            return true;
        }
    }
    return false;
}


static bool train_pointer_aliases_parameter_metadata(
    const void *pointer,
    const VAECParameter *parameter
)
{
    return
        pointer == (const void *)parameter ||
        train_pointer_aliases_shape_metadata(pointer, &parameter->shape) ||
        pointer == (const void *)&parameter->parameter_key ||
        pointer == (const void *)&parameter->values ||
        pointer == (const void *)&parameter->gradients ||
        pointer == (const void *)&parameter->adam_m ||
        pointer == (const void *)&parameter->adam_v ||
        pointer == (const void *)&parameter->owned_block ||
        pointer == (const void *)&parameter->owned_bytes;
}


static bool train_pointer_aliases_dense_metadata(
    const void *pointer,
    const VAECDense *layer
)
{
    return
        pointer == (const void *)layer ||
        pointer == (const void *)&layer->in_features ||
        pointer == (const void *)&layer->out_features ||
        pointer == (const void *)&layer->with_bias ||
        train_pointer_aliases_parameter_metadata(
            pointer, &layer->weights) ||
        train_pointer_aliases_parameter_metadata(pointer, &layer->bias);
}


static bool train_pointer_aliases_conv2d_metadata(
    const void *pointer,
    const VAECConv2D *layer
)
{
    return
        pointer == (const void *)layer ||
        pointer == (const void *)&layer->in_channels ||
        pointer == (const void *)&layer->out_channels ||
        pointer == (const void *)&layer->kernel_size ||
        pointer == (const void *)&layer->stride ||
        pointer == (const void *)&layer->padding ||
        pointer == (const void *)&layer->with_bias ||
        train_pointer_aliases_parameter_metadata(
            pointer, &layer->weights) ||
        train_pointer_aliases_parameter_metadata(pointer, &layer->bias);
}


static bool train_pointer_aliases_batch_norm_metadata(
    const void *pointer,
    const VAECBatchNorm *layer
)
{
    return
        pointer == (const void *)layer ||
        pointer == (const void *)&layer->kind ||
        pointer == (const void *)&layer->num_features ||
        pointer == (const void *)&layer->epsilon ||
        pointer == (const void *)&layer->momentum ||
        train_pointer_aliases_parameter_metadata(pointer, &layer->gamma) ||
        train_pointer_aliases_parameter_metadata(pointer, &layer->beta) ||
        pointer == (const void *)&layer->running_mean ||
        pointer == (const void *)&layer->running_variance ||
        pointer == (const void *)&layer->batch_mean ||
        pointer == (const void *)&layer->batch_inv_std ||
        pointer == (const void *)&layer->work_mean ||
        pointer == (const void *)&layer->work_inv_std ||
        pointer == (const void *)&layer->work_unbiased_variance ||
        pointer == (const void *)&layer->owned_block ||
        pointer == (const void *)&layer->owned_bytes ||
        pointer == (const void *)&layer->last_sample_count ||
        train_pointer_aliases_shape_metadata(
            pointer, &layer->last_input_shape) ||
        train_pointer_aliases_shape_metadata(
            pointer, &layer->last_output_shape) ||
        pointer == (const void *)&layer->cache_valid;
}


static bool train_pointer_aliases_max_pool_metadata(
    const void *pointer,
    const VAECMaxPool2D *layer
)
{
    return
        pointer == (const void *)layer ||
        pointer == (const void *)&layer->argmax_indices ||
        pointer == (const void *)&layer->capacity ||
        pointer == (const void *)&layer->owned_bytes ||
        pointer == (const void *)&layer->last_count ||
        train_pointer_aliases_shape_metadata(
            pointer, &layer->last_input_shape) ||
        train_pointer_aliases_shape_metadata(
            pointer, &layer->last_output_shape) ||
        pointer == (const void *)&layer->cache_valid;
}


static bool train_pointer_aliases_model_object(
    const void *pointer,
    const VAECModel *model
)
{
    size_t index;

    if (pointer == (const void *)model ||
        pointer == (const void *)model->profile_id ||
        train_pointer_aliases_config_metadata(pointer, &model->config) ||
        pointer == (const void *)&model->max_batch_size ||
        train_pointer_aliases_max_pool_metadata(
            pointer, &model->encoder_max_pool) ||
        train_pointer_aliases_dense_metadata(
            pointer, &model->encoder_fc) ||
        train_pointer_aliases_dense_metadata(
            pointer, &model->decoder_fc) ||
        train_pointer_aliases_batch_norm_metadata(
            pointer, &model->decoder_fc_batch_norm) ||
        train_pointer_aliases_conv2d_metadata(
            pointer, &model->decoder_output_conv) ||
        pointer == (const void *)model->parameters ||
        pointer == (const void *)&model->parameter_count ||
        pointer == (const void *)&model->trainable_value_count ||
        pointer == (const void *)&model->initialization_seed ||
        pointer == (const void *)&model->vae_cache_generation ||
        pointer == (const void *)&model->parameters_initialized ||
        pointer == (const void *)&model->allocated) {

        return true;
    }
    for (index = 0; index < VAE_C_PROFILE_TEXT_CAP; index++) {
        if (pointer == (const void *)&model->profile_id[index]) {

            return true;
        }
    }
    for (index = 0; index < VAE_C_MODEL_MAX_ENCODER_CONVS; index++) {
        if (train_pointer_aliases_conv2d_metadata(
                pointer, &model->encoder_convs[index]) ||
            train_pointer_aliases_batch_norm_metadata(
                pointer, &model->encoder_batch_norms[index])) {

            return true;
        }
    }
    for (index = 0;
         index < VAE_C_MODEL_MAX_DECODER_HIDDEN_CONVS;
         index++) {

        if (train_pointer_aliases_conv2d_metadata(
                pointer, &model->decoder_hidden_convs[index]) ||
            train_pointer_aliases_batch_norm_metadata(
                pointer, &model->decoder_hidden_batch_norms[index])) {

            return true;
        }
    }
    for (index = 0; index < VAE_C_MODEL_MAX_PARAMETERS; index++) {
        if (pointer == (const void *)&model->parameters[index] ||
            (model->parameters[index] != NULL &&
             train_pointer_aliases_parameter_metadata(
                 pointer, model->parameters[index]))) {

            return true;
        }
    }
    return false;
}


static bool train_pointer_aliases_tensor_object(
    const void *pointer,
    const VAECTensor *tensor
)
{
    return
        pointer == (const void *)tensor ||
        train_pointer_aliases_shape_metadata(pointer, &tensor->shape) ||
        pointer == (const void *)&tensor->data ||
        pointer == (const void *)&tensor->grad ||
        pointer == (const void *)&tensor->owned_block ||
        pointer == (const void *)&tensor->owned_bytes ||
        pointer == (const void *)&tensor->owns_memory ||
        pointer == (const void *)&tensor->has_grad;
}


static bool train_pointer_aliases_workspace_object(
    const void *pointer,
    const VAECModelWorkspace *workspace
)
{
    size_t index;

    if (pointer == (const void *)workspace ||
        pointer == (const void *)workspace->profile_id ||
        train_pointer_aliases_config_metadata(
            pointer, &workspace->config) ||
        pointer == (const void *)&workspace->batch_size ||
        pointer == (const void *)&workspace->owned_tensor_count ||
        pointer == (const void *)&workspace->logical_value_count ||
        pointer == (const void *)&workspace->storage_float_count ||
        pointer == (const void *)&workspace->storage_bytes ||
        pointer == (const void *)&workspace->vae_backward_model ||
        pointer == (const void *)&workspace->vae_backward_generation ||
        pointer == (const void *)&workspace->allocated) {

        return true;
    }
    for (index = 0; index < VAE_C_PROFILE_TEXT_CAP; index++) {
        if (pointer == (const void *)&workspace->profile_id[index]) {

            return true;
        }
    }
    if (train_pointer_aliases_tensor_object(
            pointer, &workspace->input) ||
        train_pointer_aliases_tensor_object(
            pointer, &workspace->encoder_pool_output) ||
        train_pointer_aliases_tensor_object(
            pointer, &workspace->encoder_global_avg_output) ||
        train_pointer_aliases_tensor_object(
            pointer, &workspace->encoder_packed_output) ||
        train_pointer_aliases_tensor_object(pointer, &workspace->mu) ||
        train_pointer_aliases_tensor_object(pointer, &workspace->logvar) ||
        train_pointer_aliases_tensor_object(pointer, &workspace->epsilon) ||
        train_pointer_aliases_tensor_object(pointer, &workspace->z) ||
        train_pointer_aliases_tensor_object(
            pointer, &workspace->decoder_fc_output) ||
        train_pointer_aliases_tensor_object(
            pointer, &workspace->decoder_fc_batch_norm_output) ||
        train_pointer_aliases_tensor_object(
            pointer, &workspace->decoder_fc_relu_output) ||
        train_pointer_aliases_tensor_object(
            pointer, &workspace->decoder_reshape_view) ||
        train_pointer_aliases_tensor_object(pointer, &workspace->logits)) {

        return true;
    }
    for (index = 0; index < VAE_C_MODEL_MAX_ENCODER_CONVS; index++) {
        if (train_pointer_aliases_tensor_object(
                pointer, &workspace->encoder_conv_outputs[index]) ||
            train_pointer_aliases_tensor_object(
                pointer,
                &workspace->encoder_batch_norm_outputs[index]) ||
            train_pointer_aliases_tensor_object(
                pointer, &workspace->encoder_relu_outputs[index])) {

            return true;
        }
    }
    for (index = 0; index < VAE_C_MAX_DECODER_UPSAMPLES; index++) {
        if (train_pointer_aliases_tensor_object(
                pointer, &workspace->decoder_upsample_outputs[index])) {

            return true;
        }
    }
    for (index = 0;
         index < VAE_C_MODEL_MAX_DECODER_HIDDEN_CONVS;
         index++) {

        if (train_pointer_aliases_tensor_object(
                pointer,
                &workspace->decoder_hidden_conv_outputs[index]) ||
            train_pointer_aliases_tensor_object(
                pointer,
                &workspace->decoder_hidden_batch_norm_outputs[index]) ||
            train_pointer_aliases_tensor_object(
                pointer,
                &workspace->decoder_hidden_relu_outputs[index])) {

            return true;
        }
    }
    return false;
}


static bool train_pointer_aliases_optimizer_object(
    const void *pointer,
    const VAECAdam *optimizer
)
{
    return
        pointer == (const void *)optimizer ||
        pointer == (const void *)&optimizer->learning_rate ||
        pointer == (const void *)&optimizer->beta1 ||
        pointer == (const void *)&optimizer->beta2 ||
        pointer == (const void *)&optimizer->adam_epsilon ||
        pointer == (const void *)&optimizer->step ||
        pointer == (const void *)&optimizer->beta1_power ||
        pointer == (const void *)&optimizer->beta2_power ||
        pointer == (const void *)&optimizer->configured;
}


static bool train_pointer_aliases_stats_object(
    const void *pointer,
    const VAECVAELossStats *stats
)
{
    return
        pointer == (const void *)stats ||
        pointer == (const void *)&stats->batch_size ||
        pointer == (const void *)&stats->beta_kl ||
        pointer == (const void *)&stats->reconstruction ||
        pointer ==
            (const void *)&stats->reconstruction.batch_size ||
        pointer ==
            (const void *)&stats->reconstruction.values_per_sample ||
        pointer ==
            (const void *)&stats->reconstruction.total_values ||
        pointer ==
            (const void *)&stats->reconstruction.total_sum ||
        pointer ==
            (const void *)&stats->reconstruction.mean_per_sample ||
        pointer ==
            (const void *)&stats->reconstruction.mean_per_value ||
        pointer == (const void *)&stats->kl ||
        pointer == (const void *)&stats->kl.batch_size ||
        pointer == (const void *)&stats->kl.latent_dim ||
        pointer == (const void *)&stats->kl.total_values ||
        pointer == (const void *)&stats->kl.total_sum ||
        pointer == (const void *)&stats->kl.mean_per_sample ||
        pointer == (const void *)&stats->kl.mean_per_dimension ||
        pointer == (const void *)&stats->total_sum ||
        pointer == (const void *)&stats->mean_per_sample;
}


static VAECOpStatus train_validate_aliases(
    const VAECModel *model,
    const VAECModelWorkspace *workspace,
    const VAECAdam *optimizer,
    const uint64_t *manifest_indices,
    const VAECVAELossStats *stats
)
{
    const void *model_buffers[TRAIN_MODEL_BUFFER_CAPACITY];
    const void *workspace_buffers[TRAIN_WORKSPACE_BUFFER_CAPACITY];
    const void *external[3] = {
        (const void *)optimizer,
        (const void *)manifest_indices,
        (const void *)stats
    };
    size_t model_count = 0;
    size_t workspace_count = 0;
    size_t model_index;
    size_t other_model_index;
    size_t workspace_index;
    size_t external_index;
    size_t other_external_index;
    size_t manifest_index;
    VAECOpStatus status;

    if ((const void *)model == (const void *)workspace) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (train_pointer_aliases_model_object(
            (const void *)stats, model) ||
        train_pointer_aliases_workspace_object(
            (const void *)stats, workspace) ||
        train_pointer_aliases_optimizer_object(
            (const void *)stats, optimizer)) {

        return VAE_C_OP_INVALID_ARGUMENT;
    }
    for (manifest_index = 0;
         manifest_index < workspace->batch_size;
         manifest_index++) {

        if ((const void *)stats ==
                (const void *)&manifest_indices[manifest_index]) {

            return VAE_C_OP_INVALID_ARGUMENT;
        }
    }
    status = train_collect_model_buffers(
        model,
        model_buffers,
        sizeof(model_buffers) / sizeof(model_buffers[0]),
        &model_count
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = train_collect_workspace_buffers(
        workspace,
        workspace_buffers,
        sizeof(workspace_buffers) / sizeof(workspace_buffers[0]),
        &workspace_count
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }

    for (model_index = 0; model_index < model_count; model_index++) {
        for (other_model_index = model_index + 1;
             other_model_index < model_count;
             other_model_index++) {

            if (model_buffers[model_index] ==
                    model_buffers[other_model_index]) {

                return VAE_C_OP_INVALID_ARGUMENT;
            }
        }
        for (workspace_index = 0;
             workspace_index < workspace_count;
             workspace_index++) {

            if (model_buffers[model_index] ==
                    workspace_buffers[workspace_index]) {

                return VAE_C_OP_INVALID_ARGUMENT;
            }
        }
    }

    for (external_index = 0; external_index < 3; external_index++) {
        const void *pointer = external[external_index];

        for (other_external_index = external_index + 1;
             other_external_index < 3;
             other_external_index++) {

            if (pointer == external[other_external_index]) {
                return VAE_C_OP_INVALID_ARGUMENT;
            }
        }
        for (model_index = 0; model_index < model_count; model_index++) {
            if (pointer == model_buffers[model_index]) {
                return VAE_C_OP_INVALID_ARGUMENT;
            }
        }
        for (workspace_index = 0;
             workspace_index < workspace_count;
             workspace_index++) {

            if (pointer == workspace_buffers[workspace_index]) {
                return VAE_C_OP_INVALID_ARGUMENT;
            }
        }
        if (train_pointer_aliases_model_object(pointer, model) ||
            train_pointer_aliases_workspace_object(pointer, workspace)) {

            return VAE_C_OP_INVALID_ARGUMENT;
        }
    }

    if (train_pointer_aliases_optimizer_object(
            (const void *)manifest_indices, optimizer) ||
        train_pointer_aliases_optimizer_object(
            (const void *)stats, optimizer) ||
        train_pointer_aliases_stats_object(
            (const void *)optimizer, stats) ||
        train_pointer_aliases_stats_object(
            (const void *)manifest_indices, stats)) {

        return VAE_C_OP_INVALID_ARGUMENT;
    }
    return VAE_C_OP_OK;
}


static bool train_input_is_valid(const VAECTensor *input)
{
    size_t index;

    if (input == NULL || input->data == NULL) {
        return false;
    }
    for (index = 0; index < input->shape.count; index++) {
        float value = input->data[index];

        if (!isfinite(value) || value < 0.0f || value > 1.0f) {
            return false;
        }
    }
    return true;
}


static VAECOpStatus train_preflight(
    VAECModel *model,
    VAECModelWorkspace *workspace,
    VAECAdam *optimizer,
    const uint64_t *manifest_indices,
    size_t manifest_index_count,
    VAECVAELossStats *stats
)
{
    size_t expected_epsilon_count;
    VAECOpStatus status;

    if (model == NULL || workspace == NULL || optimizer == NULL ||
        manifest_indices == NULL || stats == NULL) {

        return VAE_C_OP_INVALID_ARGUMENT;
    }
    status = vae_c_model_validate(model);
    if (status != VAE_C_OP_OK) {
        return status;
    }
    if (!vae_c_model_is_allocated(model) ||
        !vae_c_model_is_initialized(model)) {

        return VAE_C_OP_INVALID_STATE;
    }
    status = vae_c_model_workspace_validate(model, workspace);
    if (status != VAE_C_OP_OK) {
        return status;
    }
    if (!vae_c_model_workspace_is_allocated(model, workspace) ||
        workspace->batch_size < 2 ||
        workspace->batch_size > model->max_batch_size ||
        manifest_index_count != workspace->batch_size ||
        workspace->epsilon.shape.rank != 2 ||
        workspace->epsilon.shape.dims[0] != workspace->batch_size ||
        workspace->epsilon.shape.dims[1] != model->config.latent_dim) {

        return VAE_C_OP_INVALID_STATE;
    }
    if (workspace->vae_backward_model == model &&
        workspace->vae_backward_generation != UINT64_C(0) &&
        workspace->vae_backward_generation ==
            model->vae_cache_generation) {

        return VAE_C_OP_INVALID_STATE;
    }
    if (!train_input_is_valid(&workspace->input)) {
        return VAE_C_OP_INVALID_STATE;
    }
    if (train_size_multiply(
            workspace->batch_size,
            model->config.latent_dim,
            &expected_epsilon_count) != 0) {

        return VAE_C_OP_OVERFLOW;
    }
    if (expected_epsilon_count != workspace->epsilon.shape.count ||
        model->vae_cache_generation > UINT64_MAX - UINT64_C(2)) {

        return model->vae_cache_generation >
                UINT64_MAX - UINT64_C(2)
            ? VAE_C_OP_OVERFLOW
            : VAE_C_OP_INVALID_STATE;
    }
    status = train_validate_aliases(
        model, workspace, optimizer, manifest_indices, stats
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    return vae_c_adam_is_configured(optimizer)
        ? VAE_C_OP_OK
        : VAE_C_OP_INVALID_STATE;
}


static VAECOpStatus train_zero_parameter_gradients(VAECModel *model)
{
    size_t index;

    for (index = 0; index < model->parameter_count; index++) {
        VAECStorageStatus status =
            vae_c_parameter_zero_grad(model->parameters[index]);

        if (status != VAE_C_STORAGE_OK) {
            return status == VAE_C_STORAGE_OVERFLOW
                ? VAE_C_OP_OVERFLOW
                : VAE_C_OP_INVALID_STATE;
        }
    }
    return VAE_C_OP_OK;
}


static VAECOpStatus train_fill_epsilon(
    VAECModelWorkspace *workspace,
    uint64_t latent_base_seed,
    uint64_t epoch,
    const uint64_t *manifest_indices
)
{
    size_t latent_dim = workspace->config.latent_dim;
    size_t sample;

    for (sample = 0; sample < workspace->batch_size; sample++) {
        size_t offset;
        VAECRngStatus status;

        if (train_size_multiply(sample, latent_dim, &offset) != 0 ||
            offset > workspace->epsilon.shape.count ||
            latent_dim > workspace->epsilon.shape.count - offset) {

            return VAE_C_OP_OVERFLOW;
        }
        status = vae_c_rng_fill_latent_train_epsilon_f32(
            latent_base_seed,
            epoch,
            manifest_indices[sample],
            workspace->epsilon.data + offset,
            latent_dim
        );
        if (status != VAE_C_RNG_OK) {
            return status == VAE_C_RNG_INVALID_ARGUMENT
                ? VAE_C_OP_INVALID_ARGUMENT
                : VAE_C_OP_INVALID_STATE;
        }
    }
    return VAE_C_OP_OK;
}


VAECOpStatus vae_c_model_train_step(
    VAECModel *model,
    VAECModelWorkspace *workspace,
    VAECAdam *optimizer,
    uint64_t latent_base_seed,
    uint64_t epoch,
    const uint64_t *manifest_indices,
    size_t manifest_index_count,
    VAECVAELossStats *stats
)
{
    VAECVAELossStats local_stats;
    VAECOpStatus status;

    status = train_preflight(
        model,
        workspace,
        optimizer,
        manifest_indices,
        manifest_index_count,
        stats
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }

    status = train_zero_parameter_gradients(model);
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = train_fill_epsilon(
        workspace,
        latent_base_seed,
        epoch,
        manifest_indices
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }

    vae_c_vae_loss_stats_init(&local_stats);
    status = vae_c_model_vae_forward(
        model,
        VAE_C_BATCH_NORM_MODE_TRAIN,
        VAE_C_LATENT_MODE_STOCHASTIC,
        workspace,
        &local_stats
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_model_workspace_zero_grad(model, workspace);
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_model_vae_backward(model, workspace, 1.0f);
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_adam_step(
        optimizer, model->parameters, model->parameter_count
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }

    *stats = local_stats;
    return VAE_C_OP_OK;
}
