#include "vae_c_model_forward.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


#define DECODER_MODEL_REGION_CAPACITY \
    (4 * VAE_C_MODEL_MAX_PARAMETERS + \
     7 * (VAE_C_MAX_CONV_DEPTH + 1 + \
          VAE_C_MAX_DECODER_HIDDEN_CONVS) + 1)

#define DECODER_WORKSPACE_REGION_CAPACITY \
    (2 * VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS)


static int decoder_size_multiply(
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


static bool decoder_values_are_finite(
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


static VAECOpStatus decoder_append_region(
    const void **regions,
    size_t capacity,
    size_t *count,
    const void *region
)
{
    if (regions == NULL || count == NULL || region == NULL) {
        return VAE_C_OP_INVALID_STATE;
    }
    if (*count >= capacity) {
        return VAE_C_OP_OVERFLOW;
    }
    regions[*count] = region;
    *count += 1;
    return VAE_C_OP_OK;
}


static VAECOpStatus decoder_collect_workspace_tensors(
    const VAECModelWorkspace *workspace,
    const VAECTensor **tensors,
    size_t capacity,
    size_t *count
)
{
    size_t index;

#define DECODER_APPEND_TENSOR(tensor_pointer) \
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
    DECODER_APPEND_TENSOR(&workspace->input);
    for (index = 0; index < workspace->config.conv_depth; index++) {
        DECODER_APPEND_TENSOR(&workspace->encoder_conv_outputs[index]);
        DECODER_APPEND_TENSOR(
            &workspace->encoder_batch_norm_outputs[index]
        );
        DECODER_APPEND_TENSOR(&workspace->encoder_relu_outputs[index]);
    }
    DECODER_APPEND_TENSOR(&workspace->encoder_pool_output);
    DECODER_APPEND_TENSOR(&workspace->encoder_global_avg_output);
    DECODER_APPEND_TENSOR(&workspace->encoder_packed_output);
    DECODER_APPEND_TENSOR(&workspace->mu);
    DECODER_APPEND_TENSOR(&workspace->logvar);
    DECODER_APPEND_TENSOR(&workspace->epsilon);
    DECODER_APPEND_TENSOR(&workspace->z);
    DECODER_APPEND_TENSOR(&workspace->decoder_fc_output);
    DECODER_APPEND_TENSOR(&workspace->decoder_fc_batch_norm_output);
    DECODER_APPEND_TENSOR(&workspace->decoder_fc_relu_output);
    for (index = 0;
         index < workspace->config.decoder_upsample_count;
         index++) {

        DECODER_APPEND_TENSOR(
            &workspace->decoder_upsample_outputs[index]
        );
    }
    for (index = 0;
         index < workspace->config.decoder_hidden_conv_count;
         index++) {

        DECODER_APPEND_TENSOR(
            &workspace->decoder_hidden_conv_outputs[index]
        );
        DECODER_APPEND_TENSOR(
            &workspace->decoder_hidden_batch_norm_outputs[index]
        );
        DECODER_APPEND_TENSOR(
            &workspace->decoder_hidden_relu_outputs[index]
        );
    }
    DECODER_APPEND_TENSOR(&workspace->logits);

#undef DECODER_APPEND_TENSOR

    return VAE_C_OP_OK;
}


static VAECOpStatus decoder_collect_batch_norms(
    const VAECModel *model,
    const VAECBatchNorm **layers,
    size_t capacity,
    size_t *count
)
{
    size_t index;

#define DECODER_APPEND_BATCH_NORM(layer_pointer) \
    do { \
        if (*count >= capacity) { \
            return VAE_C_OP_OVERFLOW; \
        } \
        layers[*count] = (layer_pointer); \
        *count += 1; \
    } while (0)

    if (model == NULL || layers == NULL || count == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    *count = 0;
    for (index = 0; index < model->config.conv_depth; index++) {
        DECODER_APPEND_BATCH_NORM(&model->encoder_batch_norms[index]);
    }
    DECODER_APPEND_BATCH_NORM(&model->decoder_fc_batch_norm);
    for (index = 0;
         index < model->config.decoder_hidden_conv_count;
         index++) {

        DECODER_APPEND_BATCH_NORM(
            &model->decoder_hidden_batch_norms[index]
        );
    }

#undef DECODER_APPEND_BATCH_NORM

    return VAE_C_OP_OK;
}


static VAECOpStatus decoder_preflight_aliases(
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
    const void *model_regions[DECODER_MODEL_REGION_CAPACITY];
    const void *workspace_regions[DECODER_WORKSPACE_REGION_CAPACITY];
    size_t model_count = 0;
    size_t workspace_count = 0;
    size_t batch_norm_count = 0;
    size_t tensor_count = 0;
    size_t index;
    size_t other;
    VAECOpStatus status;

    for (index = 0; index < model->parameter_count; index++) {
        const VAECParameter *parameter = model->parameters[index];

        status = decoder_append_region(
            model_regions,
            DECODER_MODEL_REGION_CAPACITY,
            &model_count,
            parameter->values
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
        status = decoder_append_region(
            model_regions,
            DECODER_MODEL_REGION_CAPACITY,
            &model_count,
            parameter->gradients
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
        status = decoder_append_region(
            model_regions,
            DECODER_MODEL_REGION_CAPACITY,
            &model_count,
            parameter->adam_m
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
        status = decoder_append_region(
            model_regions,
            DECODER_MODEL_REGION_CAPACITY,
            &model_count,
            parameter->adam_v
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
    }
    status = decoder_collect_batch_norms(
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
        size_t region_index;

        for (region_index = 0; region_index < 7; region_index++) {
            status = decoder_append_region(
                model_regions,
                DECODER_MODEL_REGION_CAPACITY,
                &model_count,
                regions[region_index]
            );
            if (status != VAE_C_OP_OK) {
                return status;
            }
        }
    }
    status = decoder_append_region(
        model_regions,
        DECODER_MODEL_REGION_CAPACITY,
        &model_count,
        model->encoder_max_pool.argmax_indices
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    for (index = 0; index < model_count; index++) {
        for (other = 0; other < index; other++) {
            if (model_regions[index] == model_regions[other]) {
                return VAE_C_OP_INVALID_ARGUMENT;
            }
        }
    }

    status = decoder_collect_workspace_tensors(
        workspace,
        tensors,
        sizeof(tensors) / sizeof(tensors[0]),
        &tensor_count
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    for (index = 0; index < tensor_count; index++) {
        status = decoder_append_region(
            workspace_regions,
            DECODER_WORKSPACE_REGION_CAPACITY,
            &workspace_count,
            tensors[index]->data
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
        if (tensors[index]->grad != NULL) {
            status = decoder_append_region(
                workspace_regions,
                DECODER_WORKSPACE_REGION_CAPACITY,
                &workspace_count,
                tensors[index]->grad
            );
            if (status != VAE_C_OP_OK) {
                return status;
            }
        }
    }
    for (index = 0; index < workspace_count; index++) {
        for (other = 0; other < model_count; other++) {
            if (workspace_regions[index] == model_regions[other]) {
                return VAE_C_OP_INVALID_ARGUMENT;
            }
        }
    }
    return VAE_C_OP_OK;
}


static VAECOpStatus decoder_preflight_upsampling(
    const VAECModelWorkspace *workspace
)
{
    const VAECTensor *input = &workspace->decoder_reshape_view;
    size_t index;

    for (index = 0;
         index < workspace->config.decoder_upsample_count;
         index++) {

        const VAECTensor *output =
            &workspace->decoder_upsample_outputs[index];
        size_t input_side = input->shape.dims[2];
        size_t output_side = output->shape.dims[2];
        size_t input_square;
        size_t output_square;
        size_t input_count;
        size_t output_count;

        if (input->shape.rank != 4 || output->shape.rank != 4 ||
            input_side == 0 || output_side < input_side ||
            input->shape.dims[3] != input_side ||
            output->shape.dims[3] != output_side ||
            (output_side - 1) > SIZE_MAX / input_side ||
            decoder_size_multiply(
                input_side, input_side, &input_square) != 0 ||
            decoder_size_multiply(
                output_side, output_side, &output_square) != 0 ||
            decoder_size_multiply(
                workspace->batch_size,
                input->shape.dims[1],
                &input_count) != 0 ||
            decoder_size_multiply(
                input_count, input_square, &input_count) != 0 ||
            decoder_size_multiply(
                workspace->batch_size,
                output->shape.dims[1],
                &output_count) != 0 ||
            decoder_size_multiply(
                output_count, output_square, &output_count) != 0 ||
            input_count != input->shape.count ||
            output_count != output->shape.count) {

            return VAE_C_OP_OVERFLOW;
        }
        input = output;
        if (index >= 1 &&
            index - 1 < workspace->config.decoder_hidden_conv_count) {

            input = &workspace->decoder_hidden_relu_outputs[index - 1];
        }
    }
    return VAE_C_OP_OK;
}


static VAECOpStatus decoder_preflight(
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
        workspace->config.conv_depth != depth ||
        model->config.decoder_hidden_conv_count != depth - 1 ||
        model->config.decoder_upsample_count != depth + 1 ||
        workspace->config.decoder_hidden_conv_count != depth - 1 ||
        workspace->config.decoder_upsample_count != depth + 1) {

        return VAE_C_OP_INVALID_STATE;
    }
    if (workspace->z.shape.rank != 2 ||
        workspace->z.shape.dims[0] != workspace->batch_size ||
        workspace->z.shape.dims[1] != model->config.latent_dim ||
        workspace->z.shape.count == 0 ||
        workspace->z.owned_bytes / sizeof(float) / 2 <
            workspace->z.shape.count ||
        !decoder_values_are_finite(
            workspace->z.data,
            workspace->z.shape.count)) {

        return VAE_C_OP_INVALID_STATE;
    }
    if (batch_norm_mode == VAE_C_BATCH_NORM_MODE_TRAIN &&
        workspace->batch_size < 2) {

        return VAE_C_OP_INVALID_STATE;
    }
    for (index = 0;
         index < model->config.decoder_hidden_conv_count;
         index++) {

        const VAECTensor *output =
            &workspace->decoder_hidden_batch_norm_outputs[index];
        size_t spatial_count;
        size_t sample_count;
        size_t side = output->shape.dims[2];

        if (decoder_size_multiply(
                side, side, &spatial_count) != 0 ||
            decoder_size_multiply(
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
    status = decoder_preflight_upsampling(workspace);
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = decoder_preflight_aliases(model, workspace);
    if (status != VAE_C_OP_OK) {
        return status;
    }
    return model->vae_cache_generation == UINT64_MAX
        ? VAE_C_OP_OVERFLOW
        : VAE_C_OP_OK;
}


VAECOpStatus vae_c_model_decoder_forward(
    VAECModel *model,
    VAECBatchNormMode batch_norm_mode,
    VAECModelWorkspace *workspace
)
{
    VAECOpStatus status;
    size_t hidden_count;
    size_t index;

    status = decoder_preflight(model, batch_norm_mode, workspace);
    if (status != VAE_C_OP_OK) {
        return status;
    }

    workspace->vae_backward_model = NULL;
    workspace->vae_backward_generation = UINT64_C(0);
    model->vae_cache_generation += UINT64_C(1);

    hidden_count = model->config.decoder_hidden_conv_count;
    status = vae_c_dense_forward(
        &model->decoder_fc,
        &workspace->z,
        &workspace->decoder_fc_output
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_batch_norm1d_forward(
        &model->decoder_fc_batch_norm,
        batch_norm_mode,
        &workspace->decoder_fc_output,
        &workspace->decoder_fc_batch_norm_output
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_relu_forward(
        &workspace->decoder_fc_batch_norm_output,
        &workspace->decoder_fc_relu_output
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_nearest_upsample2d_forward(
        &workspace->decoder_reshape_view,
        &workspace->decoder_upsample_outputs[0]
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_nearest_upsample2d_forward(
        &workspace->decoder_upsample_outputs[0],
        &workspace->decoder_upsample_outputs[1]
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }

    for (index = 0; index < hidden_count; index++) {
        status = vae_c_conv2d_forward(
            &model->decoder_hidden_convs[index],
            &workspace->decoder_upsample_outputs[index + 1],
            &workspace->decoder_hidden_conv_outputs[index]
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
        status = vae_c_batch_norm2d_forward(
            &model->decoder_hidden_batch_norms[index],
            batch_norm_mode,
            &workspace->decoder_hidden_conv_outputs[index],
            &workspace->decoder_hidden_batch_norm_outputs[index]
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
        status = vae_c_relu_forward(
            &workspace->decoder_hidden_batch_norm_outputs[index],
            &workspace->decoder_hidden_relu_outputs[index]
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
        status = vae_c_nearest_upsample2d_forward(
            &workspace->decoder_hidden_relu_outputs[index],
            &workspace->decoder_upsample_outputs[index + 2]
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
    }
    status = vae_c_conv2d_forward(
        &model->decoder_output_conv,
        &workspace->decoder_upsample_outputs[
            model->config.decoder_upsample_count - 1
        ],
        &workspace->logits
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    if (!decoder_values_are_finite(
            workspace->logits.data,
            workspace->logits.shape.count)) {

        return VAE_C_OP_INVALID_STATE;
    }
    return VAE_C_OP_OK;
}
