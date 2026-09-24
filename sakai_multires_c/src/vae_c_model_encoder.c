#include "vae_c_model_forward.h"

#include "vae_c_latent.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


#define ENCODER_MODEL_REGION_CAPACITY \
    (4 * VAE_C_MODEL_MAX_PARAMETERS + \
     7 * (VAE_C_MAX_CONV_DEPTH + 1 + \
          VAE_C_MAX_DECODER_HIDDEN_CONVS) + 1)

#define ENCODER_WORKSPACE_REGION_CAPACITY \
    (2 * VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS)


static int encoder_size_multiply(
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


static bool encoder_values_are_finite(
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


static size_t encoder_collect_workspace_tensors(
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


static size_t encoder_collect_batch_norms(
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


static VAECOpStatus encoder_preflight_aliases(
    const VAECModel *model,
    const VAECModelWorkspace *workspace
)
{
    const VAECTensor *tensors[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ];
    const VAECBatchNorm *batch_norms[
        VAE_C_MAX_CONV_DEPTH + 1 +
        VAE_C_MAX_DECODER_HIDDEN_CONVS
    ];
    const void *model_regions[ENCODER_MODEL_REGION_CAPACITY];
    const void *workspace_regions[ENCODER_WORKSPACE_REGION_CAPACITY];
    size_t model_count = 0;
    size_t workspace_count = 0;
    size_t batch_norm_count;
    size_t tensor_count;
    size_t index;
    size_t other;

    for (index = 0; index < model->parameter_count; index++) {
        const VAECParameter *parameter = model->parameters[index];

        model_regions[model_count++] = parameter->values;
        model_regions[model_count++] = parameter->gradients;
        model_regions[model_count++] = parameter->adam_m;
        model_regions[model_count++] = parameter->adam_v;
    }
    batch_norm_count = encoder_collect_batch_norms(model, batch_norms);
    for (index = 0; index < batch_norm_count; index++) {
        const VAECBatchNorm *layer = batch_norms[index];

        model_regions[model_count++] = layer->running_mean;
        model_regions[model_count++] = layer->running_variance;
        model_regions[model_count++] = layer->batch_mean;
        model_regions[model_count++] = layer->batch_inv_std;
        model_regions[model_count++] = layer->work_mean;
        model_regions[model_count++] = layer->work_inv_std;
        model_regions[model_count++] = layer->work_unbiased_variance;
    }
    model_regions[model_count++] = model->encoder_max_pool.argmax_indices;

    if (model_count > ENCODER_MODEL_REGION_CAPACITY) {
        return VAE_C_OP_INVALID_STATE;
    }
    for (index = 0; index < model_count; index++) {
        if (model_regions[index] == NULL) {
            return VAE_C_OP_INVALID_STATE;
        }
        for (other = 0; other < index; other++) {
            if (model_regions[index] == model_regions[other]) {
                return VAE_C_OP_INVALID_ARGUMENT;
            }
        }
    }

    tensor_count = encoder_collect_workspace_tensors(workspace, tensors);
    if (tensor_count > VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS) {
        return VAE_C_OP_INVALID_STATE;
    }
    for (index = 0; index < tensor_count; index++) {
        workspace_regions[workspace_count++] = tensors[index]->data;
        if (tensors[index]->grad != NULL) {
            workspace_regions[workspace_count++] = tensors[index]->grad;
        }
    }
    if (workspace_count > ENCODER_WORKSPACE_REGION_CAPACITY) {
        return VAE_C_OP_INVALID_STATE;
    }
    for (index = 0; index < workspace_count; index++) {
        if (workspace_regions[index] == NULL) {
            return VAE_C_OP_INVALID_STATE;
        }
        for (other = 0; other < model_count; other++) {
            if (workspace_regions[index] == model_regions[other]) {
                return VAE_C_OP_INVALID_ARGUMENT;
            }
        }
    }
    return VAE_C_OP_OK;
}


static VAECOpStatus encoder_preflight(
    VAECModel *model,
    VAECBatchNormMode batch_norm_mode,
    VAECModelWorkspace *workspace
)
{
    VAECOpStatus status;
    size_t depth;
    size_t index;

    if (model == NULL || workspace == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (batch_norm_mode != VAE_C_BATCH_NORM_MODE_TRAIN &&
        batch_norm_mode != VAE_C_BATCH_NORM_MODE_EVAL) {

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
        workspace->config.conv_depth != depth) {

        return VAE_C_OP_INVALID_STATE;
    }

    for (index = 0; index < workspace->input.shape.count; index++) {
        float value = workspace->input.data[index];

        if (!isfinite(value) || value < 0.0f || value > 1.0f) {
            return VAE_C_OP_INVALID_STATE;
        }
    }
    for (index = 0; index < model->parameter_count; index++) {
        const VAECParameter *parameter = model->parameters[index];

        if (parameter == NULL || parameter->values == NULL ||
            !encoder_values_are_finite(
                parameter->values,
                parameter->shape.count)) {

            return VAE_C_OP_INVALID_STATE;
        }
    }

    for (index = 0; index < depth; index++) {
        size_t spatial_count;
        size_t sample_count;
        size_t side =
            workspace->encoder_batch_norm_outputs[index].shape.dims[2];

        if (encoder_size_multiply(side, side, &spatial_count) != 0 ||
            encoder_size_multiply(
                workspace->batch_size,
                spatial_count,
                &sample_count) != 0) {

            return VAE_C_OP_OVERFLOW;
        }
        if ((batch_norm_mode == VAE_C_BATCH_NORM_MODE_TRAIN &&
             sample_count < 2) ||
            (batch_norm_mode == VAE_C_BATCH_NORM_MODE_EVAL &&
             sample_count < 1)) {

            return VAE_C_OP_INVALID_STATE;
        }
    }

    status = encoder_preflight_aliases(model, workspace);
    if (status != VAE_C_OP_OK) {
        return status;
    }
    return model->vae_cache_generation == UINT64_MAX
        ? VAE_C_OP_OVERFLOW
        : VAE_C_OP_OK;
}


VAECOpStatus vae_c_model_encoder_forward(
    VAECModel *model,
    VAECBatchNormMode batch_norm_mode,
    VAECModelWorkspace *workspace
)
{
    VAECOpStatus status;
    const VAECTensor *block_input;
    size_t depth;
    size_t index;

    status = encoder_preflight(model, batch_norm_mode, workspace);
    if (status != VAE_C_OP_OK) {
        return status;
    }

    workspace->vae_backward_model = NULL;
    workspace->vae_backward_generation = UINT64_C(0);
    model->vae_cache_generation += UINT64_C(1);

    depth = model->config.conv_depth;
    status = vae_c_conv2d_forward(
        &model->encoder_convs[0],
        &workspace->input,
        &workspace->encoder_conv_outputs[0]
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_batch_norm2d_forward(
        &model->encoder_batch_norms[0],
        batch_norm_mode,
        &workspace->encoder_conv_outputs[0],
        &workspace->encoder_batch_norm_outputs[0]
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_relu_forward(
        &workspace->encoder_batch_norm_outputs[0],
        &workspace->encoder_relu_outputs[0]
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_max_pool2d_forward(
        &model->encoder_max_pool,
        &workspace->encoder_relu_outputs[0],
        &workspace->encoder_pool_output
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }

    block_input = &workspace->encoder_pool_output;
    for (index = 1; index < depth; index++) {
        status = vae_c_conv2d_forward(
            &model->encoder_convs[index],
            block_input,
            &workspace->encoder_conv_outputs[index]
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
        status = vae_c_batch_norm2d_forward(
            &model->encoder_batch_norms[index],
            batch_norm_mode,
            &workspace->encoder_conv_outputs[index],
            &workspace->encoder_batch_norm_outputs[index]
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
        status = vae_c_relu_forward(
            &workspace->encoder_batch_norm_outputs[index],
            &workspace->encoder_relu_outputs[index]
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
        block_input = &workspace->encoder_relu_outputs[index];
    }

    status = vae_c_global_avg_pool2d_forward(
        block_input,
        &workspace->encoder_global_avg_output
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_dense_forward(
        &model->encoder_fc,
        &workspace->encoder_global_avg_output,
        &workspace->encoder_packed_output
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_latent_split_forward(
        &workspace->encoder_packed_output,
        &workspace->mu,
        &workspace->logvar
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    if (!encoder_values_are_finite(
            workspace->mu.data,
            workspace->mu.shape.count) ||
        !encoder_values_are_finite(
            workspace->logvar.data,
            workspace->logvar.shape.count)) {

        return VAE_C_OP_INVALID_STATE;
    }
    return VAE_C_OP_OK;
}
