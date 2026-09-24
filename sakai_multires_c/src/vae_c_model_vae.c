#include "vae_c_model_forward.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


#define VAE_FORWARD_BATCH_NORM_CAPACITY \
    (VAE_C_MAX_CONV_DEPTH + 1 + \
     VAE_C_MAX_DECODER_HIDDEN_CONVS)

#define VAE_FORWARD_MODEL_BUFFER_CAPACITY \
    (4 * VAE_C_MODEL_MAX_PARAMETERS + \
     7 * VAE_FORWARD_BATCH_NORM_CAPACITY + 1)

#define VAE_FORWARD_WORKSPACE_BUFFER_CAPACITY \
    (2 * VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS)


static int vae_forward_size_multiply(
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


static bool vae_forward_values_are_finite(
    const float *values,
    size_t count
)
{
    size_t index;

    if (values == NULL) {
        return false;
    }
    for (index = 0; index < count; index++) {
        if (!isfinite(values[index])) {
            return false;
        }
    }
    return true;
}


static bool vae_forward_input_is_valid(const VAECTensor *input)
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


static VAECOpStatus vae_forward_append_pointer(
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


static VAECOpStatus vae_forward_collect_batch_norms(
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


static VAECOpStatus vae_forward_collect_model_buffers(
    const VAECModel *model,
    const void **buffers,
    size_t capacity,
    size_t *count
)
{
    const VAECBatchNorm *batch_norms[
        VAE_FORWARD_BATCH_NORM_CAPACITY
    ];
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
            status = vae_forward_append_pointer(
                buffers, capacity, count, regions[region]
            );
            if (status != VAE_C_OP_OK) {
                return status;
            }
        }
    }

    status = vae_forward_collect_batch_norms(
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
            status = vae_forward_append_pointer(
                buffers, capacity, count, regions[region]
            );
            if (status != VAE_C_OP_OK) {
                return status;
            }
        }
    }
    return vae_forward_append_pointer(
        buffers,
        capacity,
        count,
        model->encoder_max_pool.argmax_indices
    );
}


static VAECOpStatus vae_forward_collect_workspace_tensors(
    const VAECModelWorkspace *workspace,
    const VAECTensor **tensors,
    size_t capacity,
    size_t *count
)
{
    size_t index;

#define VAE_FORWARD_APPEND_TENSOR(tensor_pointer) \
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
    VAE_FORWARD_APPEND_TENSOR(&workspace->input);
    for (index = 0; index < workspace->config.conv_depth; index++) {
        VAE_FORWARD_APPEND_TENSOR(
            &workspace->encoder_conv_outputs[index]
        );
        VAE_FORWARD_APPEND_TENSOR(
            &workspace->encoder_batch_norm_outputs[index]
        );
        VAE_FORWARD_APPEND_TENSOR(
            &workspace->encoder_relu_outputs[index]
        );
    }
    VAE_FORWARD_APPEND_TENSOR(&workspace->encoder_pool_output);
    VAE_FORWARD_APPEND_TENSOR(&workspace->encoder_global_avg_output);
    VAE_FORWARD_APPEND_TENSOR(&workspace->encoder_packed_output);
    VAE_FORWARD_APPEND_TENSOR(&workspace->mu);
    VAE_FORWARD_APPEND_TENSOR(&workspace->logvar);
    VAE_FORWARD_APPEND_TENSOR(&workspace->epsilon);
    VAE_FORWARD_APPEND_TENSOR(&workspace->z);
    VAE_FORWARD_APPEND_TENSOR(&workspace->decoder_fc_output);
    VAE_FORWARD_APPEND_TENSOR(
        &workspace->decoder_fc_batch_norm_output
    );
    VAE_FORWARD_APPEND_TENSOR(&workspace->decoder_fc_relu_output);
    for (index = 0;
         index < workspace->config.decoder_upsample_count;
         index++) {

        VAE_FORWARD_APPEND_TENSOR(
            &workspace->decoder_upsample_outputs[index]
        );
    }
    for (index = 0;
         index < workspace->config.decoder_hidden_conv_count;
         index++) {

        VAE_FORWARD_APPEND_TENSOR(
            &workspace->decoder_hidden_conv_outputs[index]
        );
        VAE_FORWARD_APPEND_TENSOR(
            &workspace->decoder_hidden_batch_norm_outputs[index]
        );
        VAE_FORWARD_APPEND_TENSOR(
            &workspace->decoder_hidden_relu_outputs[index]
        );
    }
    VAE_FORWARD_APPEND_TENSOR(&workspace->logits);

#undef VAE_FORWARD_APPEND_TENSOR

    return VAE_C_OP_OK;
}


static VAECOpStatus vae_forward_collect_workspace_buffers(
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
    status = vae_forward_collect_workspace_tensors(
        workspace,
        tensors,
        sizeof(tensors) / sizeof(tensors[0]),
        &tensor_count
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    for (index = 0; index < tensor_count; index++) {
        status = vae_forward_append_pointer(
            buffers, capacity, count, tensors[index]->data
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
        if (tensors[index]->grad != NULL) {
            status = vae_forward_append_pointer(
                buffers, capacity, count, tensors[index]->grad
            );
            if (status != VAE_C_OP_OK) {
                return status;
            }
        }
    }
    return VAE_C_OP_OK;
}


static bool vae_forward_stats_alias_model_object(
    const void *stats_pointer,
    const VAECModel *model
)
{
    size_t index;

    if (stats_pointer == (const void *)model ||
        stats_pointer == (const void *)model->profile_id ||
        stats_pointer == (const void *)&model->config ||
        stats_pointer == (const void *)&model->max_batch_size ||
        stats_pointer == (const void *)&model->encoder_max_pool ||
        stats_pointer == (const void *)&model->encoder_fc ||
        stats_pointer == (const void *)&model->decoder_fc ||
        stats_pointer == (const void *)&model->decoder_fc_batch_norm ||
        stats_pointer == (const void *)&model->decoder_output_conv ||
        stats_pointer == (const void *)model->parameters ||
        stats_pointer == (const void *)&model->parameter_count ||
        stats_pointer == (const void *)&model->trainable_value_count ||
        stats_pointer == (const void *)&model->initialization_seed ||
        stats_pointer == (const void *)&model->vae_cache_generation ||
        stats_pointer == (const void *)&model->parameters_initialized ||
        stats_pointer == (const void *)&model->allocated) {

        return true;
    }
    for (index = 0; index < model->config.conv_depth; index++) {
        if (stats_pointer == (const void *)&model->encoder_convs[index] ||
            stats_pointer ==
                (const void *)&model->encoder_batch_norms[index]) {

            return true;
        }
    }
    for (index = 0;
         index < model->config.decoder_hidden_conv_count;
         index++) {

        if (stats_pointer ==
                (const void *)&model->decoder_hidden_convs[index] ||
            stats_pointer ==
                (const void *)&model->decoder_hidden_batch_norms[index]) {

            return true;
        }
    }
    for (index = 0; index < model->parameter_count; index++) {
        if (stats_pointer == (const void *)model->parameters[index]) {
            return true;
        }
    }
    return false;
}


static bool vae_forward_stats_alias_tensor_object(
    const void *stats_pointer,
    const VAECTensor *tensor
)
{
    return
        stats_pointer == (const void *)tensor ||
        stats_pointer == (const void *)&tensor->shape ||
        stats_pointer == (const void *)&tensor->owned_bytes ||
        stats_pointer == (const void *)&tensor->owns_memory ||
        stats_pointer == (const void *)&tensor->has_grad;
}


static bool vae_forward_stats_alias_workspace_object(
    const void *stats_pointer,
    const VAECModelWorkspace *workspace,
    const VAECTensor *const *tensors,
    size_t tensor_count
)
{
    size_t index;

    if (stats_pointer == (const void *)workspace ||
        stats_pointer == (const void *)workspace->profile_id ||
        stats_pointer == (const void *)&workspace->config ||
        stats_pointer == (const void *)&workspace->batch_size ||
        stats_pointer ==
            (const void *)&workspace->decoder_reshape_view ||
        stats_pointer == (const void *)&workspace->owned_tensor_count ||
        stats_pointer == (const void *)&workspace->logical_value_count ||
        stats_pointer == (const void *)&workspace->storage_float_count ||
        stats_pointer == (const void *)&workspace->storage_bytes ||
        stats_pointer == (const void *)&workspace->vae_backward_model ||
        stats_pointer ==
            (const void *)&workspace->vae_backward_generation ||
        stats_pointer == (const void *)&workspace->allocated) {

        return true;
    }
    for (index = 0; index < tensor_count; index++) {
        if (vae_forward_stats_alias_tensor_object(
                stats_pointer, tensors[index])) {

            return true;
        }
    }
    return false;
}


static VAECOpStatus vae_forward_validate_aliases(
    const VAECModel *model,
    const VAECModelWorkspace *workspace,
    const VAECVAELossStats *stats
)
{
    const VAECTensor *tensors[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ];
    const void *model_buffers[VAE_FORWARD_MODEL_BUFFER_CAPACITY];
    const void *workspace_buffers[
        VAE_FORWARD_WORKSPACE_BUFFER_CAPACITY
    ];
    const void *stats_pointer = (const void *)stats;
    size_t model_count = 0;
    size_t workspace_count = 0;
    size_t tensor_count = 0;
    size_t workspace_index;
    size_t model_index;
    size_t other_model_index;
    VAECOpStatus status;

    status = vae_forward_collect_model_buffers(
        model,
        model_buffers,
        sizeof(model_buffers) / sizeof(model_buffers[0]),
        &model_count
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
    }
    status = vae_forward_collect_workspace_buffers(
        workspace,
        workspace_buffers,
        sizeof(workspace_buffers) / sizeof(workspace_buffers[0]),
        &workspace_count
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_forward_collect_workspace_tensors(
        workspace,
        tensors,
        sizeof(tensors) / sizeof(tensors[0]),
        &tensor_count
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }

    for (workspace_index = 0;
         workspace_index < workspace_count;
         workspace_index++) {

        if (stats_pointer == workspace_buffers[workspace_index]) {
            return VAE_C_OP_INVALID_ARGUMENT;
        }
        for (model_index = 0;
             model_index < model_count;
             model_index++) {

            if (workspace_buffers[workspace_index] ==
                    model_buffers[model_index]) {

                return VAE_C_OP_INVALID_ARGUMENT;
            }
        }
    }
    for (model_index = 0; model_index < model_count; model_index++) {
        if (stats_pointer == model_buffers[model_index]) {
            return VAE_C_OP_INVALID_ARGUMENT;
        }
    }
    if (vae_forward_stats_alias_model_object(stats_pointer, model) ||
        vae_forward_stats_alias_workspace_object(
            stats_pointer, workspace, tensors, tensor_count)) {

        return VAE_C_OP_INVALID_ARGUMENT;
    }
    return VAE_C_OP_OK;
}


static VAECOpStatus vae_forward_validate_sample_count(
    size_t batch_size,
    const VAECTensor *tensor,
    VAECBatchNormMode mode
)
{
    size_t spatial_count;
    size_t sample_count;

    if (tensor == NULL || tensor->shape.rank != 4 ||
        tensor->shape.dims[0] != batch_size ||
        vae_forward_size_multiply(
            tensor->shape.dims[2],
            tensor->shape.dims[3],
            &spatial_count) != 0 ||
        vae_forward_size_multiply(
            batch_size,
            spatial_count,
            &sample_count) != 0) {

        return VAE_C_OP_OVERFLOW;
    }
    if (sample_count == 0 ||
        (mode == VAE_C_BATCH_NORM_MODE_TRAIN && sample_count < 2)) {

        return VAE_C_OP_INVALID_STATE;
    }
    return VAE_C_OP_OK;
}


static VAECOpStatus vae_forward_validate_batch_norm_samples(
    const VAECModel *model,
    const VAECModelWorkspace *workspace,
    VAECBatchNormMode mode
)
{
    size_t index;
    VAECOpStatus status;

    for (index = 0; index < model->config.conv_depth; index++) {
        status = vae_forward_validate_sample_count(
            workspace->batch_size,
            &workspace->encoder_batch_norm_outputs[index],
            mode
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
    }
    if (workspace->batch_size == 0 ||
        (mode == VAE_C_BATCH_NORM_MODE_TRAIN &&
         workspace->batch_size < 2)) {

        return VAE_C_OP_INVALID_STATE;
    }
    for (index = 0;
         index < model->config.decoder_hidden_conv_count;
         index++) {

        status = vae_forward_validate_sample_count(
            workspace->batch_size,
            &workspace->decoder_hidden_batch_norm_outputs[index],
            mode
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
    }
    return VAE_C_OP_OK;
}


static VAECOpStatus vae_forward_preflight(
    VAECModel *model,
    VAECBatchNormMode batch_norm_mode,
    VAECLatentMode latent_mode,
    VAECModelWorkspace *workspace,
    VAECVAELossStats *stats
)
{
    VAECOpStatus status;
    size_t depth;

    if (model == NULL || workspace == NULL || stats == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (batch_norm_mode != VAE_C_BATCH_NORM_MODE_TRAIN &&
        batch_norm_mode != VAE_C_BATCH_NORM_MODE_EVAL) {

        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (latent_mode != VAE_C_LATENT_MODE_STOCHASTIC &&
        latent_mode != VAE_C_LATENT_MODE_MEAN) {

        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (latent_mode == VAE_C_LATENT_MODE_MEAN &&
        batch_norm_mode == VAE_C_BATCH_NORM_MODE_TRAIN) {

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
        workspace->batch_size == 0 ||
        workspace->batch_size > model->max_batch_size) {

        return VAE_C_OP_INVALID_STATE;
    }

    depth = model->config.conv_depth;
    if ((depth != 3 && depth != 4) ||
        workspace->config.conv_depth != depth ||
        model->config.decoder_hidden_conv_count != depth - 1 ||
        model->config.decoder_upsample_count != depth + 1 ||
        workspace->config.decoder_hidden_conv_count != depth - 1 ||
        workspace->config.decoder_upsample_count != depth + 1 ||
        workspace->input.shape.rank != 4 ||
        workspace->input.shape.dims[0] != workspace->batch_size ||
        !vae_c_shape_equal(
            &workspace->input.shape, &workspace->logits.shape) ||
        workspace->mu.shape.rank != 2 ||
        workspace->mu.shape.dims[0] != workspace->batch_size ||
        workspace->mu.shape.dims[1] != model->config.latent_dim ||
        !vae_c_shape_equal(
            &workspace->mu.shape, &workspace->logvar.shape) ||
        !vae_c_shape_equal(
            &workspace->mu.shape, &workspace->epsilon.shape) ||
        !vae_c_shape_equal(
            &workspace->mu.shape, &workspace->z.shape)) {

        return VAE_C_OP_INVALID_STATE;
    }
    if (!vae_forward_input_is_valid(&workspace->input)) {
        return VAE_C_OP_INVALID_STATE;
    }
    if (latent_mode == VAE_C_LATENT_MODE_STOCHASTIC &&
        !vae_forward_values_are_finite(
            workspace->epsilon.data,
            workspace->epsilon.shape.count)) {

        return VAE_C_OP_INVALID_STATE;
    }
    status = vae_forward_validate_batch_norm_samples(
        model, workspace, batch_norm_mode
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_forward_validate_aliases(model, workspace, stats);
    if (status != VAE_C_OP_OK) {
        return status;
    }
    return model->vae_cache_generation > UINT64_MAX - UINT64_C(2)
        ? VAE_C_OP_OVERFLOW
        : VAE_C_OP_OK;
}


VAECOpStatus vae_c_model_vae_forward(
    VAECModel *model,
    VAECBatchNormMode batch_norm_mode,
    VAECLatentMode latent_mode,
    VAECModelWorkspace *workspace,
    VAECVAELossStats *stats
)
{
    const VAECTensor *epsilon;
    VAECOpStatus status;

    status = vae_forward_preflight(
        model,
        batch_norm_mode,
        latent_mode,
        workspace,
        stats
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }

    status = vae_c_model_encoder_forward(
        model, batch_norm_mode, workspace
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    epsilon = latent_mode == VAE_C_LATENT_MODE_STOCHASTIC
        ? &workspace->epsilon
        : NULL;
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
    if (status != VAE_C_OP_OK) {
        return status;
    }
    if (batch_norm_mode == VAE_C_BATCH_NORM_MODE_TRAIN &&
        latent_mode == VAE_C_LATENT_MODE_STOCHASTIC) {

        workspace->vae_backward_model = model;
        workspace->vae_backward_generation =
            model->vae_cache_generation;
    }
    return VAE_C_OP_OK;
}
