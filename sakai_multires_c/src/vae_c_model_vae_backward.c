#include "vae_c_model_backward.h"

#include "vae_c_batchnorm.h"
#include "vae_c_conv2d.h"
#include "vae_c_dense.h"
#include "vae_c_latent.h"
#include "vae_c_objective.h"
#include "vae_c_ops.h"
#include "vae_c_spatial.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


#define VAE_BACKWARD_BATCH_NORM_CAPACITY \
    (VAE_C_MAX_CONV_DEPTH + 1 + \
     VAE_C_MAX_DECODER_HIDDEN_CONVS)

#define VAE_BACKWARD_MODEL_BUFFER_CAPACITY \
    (4 * VAE_C_MODEL_MAX_PARAMETERS + \
     7 * VAE_BACKWARD_BATCH_NORM_CAPACITY + 1)

#define VAE_BACKWARD_WORKSPACE_BUFFER_CAPACITY \
    (2 * VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS)


static int vae_backward_size_multiply(
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


static bool vae_backward_values_are_finite(
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


static bool vae_backward_values_are_zero(
    const float *values,
    size_t count
)
{
    size_t index;

    if (values == NULL) {
        return false;
    }
    for (index = 0; index < count; index++) {
        if (values[index] != 0.0f) {
            return false;
        }
    }
    return true;
}


static VAECOpStatus vae_backward_append_pointer(
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


static VAECOpStatus vae_backward_collect_workspace_tensors(
    const VAECModelWorkspace *workspace,
    const VAECTensor **tensors,
    size_t capacity,
    size_t *count
)
{
    size_t index;

#define VAE_BACKWARD_APPEND_TENSOR(tensor_pointer) \
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
    VAE_BACKWARD_APPEND_TENSOR(&workspace->input);
    for (index = 0; index < workspace->config.conv_depth; index++) {
        VAE_BACKWARD_APPEND_TENSOR(
            &workspace->encoder_conv_outputs[index]
        );
        VAE_BACKWARD_APPEND_TENSOR(
            &workspace->encoder_batch_norm_outputs[index]
        );
        VAE_BACKWARD_APPEND_TENSOR(
            &workspace->encoder_relu_outputs[index]
        );
    }
    VAE_BACKWARD_APPEND_TENSOR(&workspace->encoder_pool_output);
    VAE_BACKWARD_APPEND_TENSOR(
        &workspace->encoder_global_avg_output
    );
    VAE_BACKWARD_APPEND_TENSOR(&workspace->encoder_packed_output);
    VAE_BACKWARD_APPEND_TENSOR(&workspace->mu);
    VAE_BACKWARD_APPEND_TENSOR(&workspace->logvar);
    VAE_BACKWARD_APPEND_TENSOR(&workspace->epsilon);
    VAE_BACKWARD_APPEND_TENSOR(&workspace->z);
    VAE_BACKWARD_APPEND_TENSOR(&workspace->decoder_fc_output);
    VAE_BACKWARD_APPEND_TENSOR(
        &workspace->decoder_fc_batch_norm_output
    );
    VAE_BACKWARD_APPEND_TENSOR(&workspace->decoder_fc_relu_output);
    for (index = 0;
         index < workspace->config.decoder_upsample_count;
         index++) {

        VAE_BACKWARD_APPEND_TENSOR(
            &workspace->decoder_upsample_outputs[index]
        );
    }
    for (index = 0;
         index < workspace->config.decoder_hidden_conv_count;
         index++) {

        VAE_BACKWARD_APPEND_TENSOR(
            &workspace->decoder_hidden_conv_outputs[index]
        );
        VAE_BACKWARD_APPEND_TENSOR(
            &workspace->decoder_hidden_batch_norm_outputs[index]
        );
        VAE_BACKWARD_APPEND_TENSOR(
            &workspace->decoder_hidden_relu_outputs[index]
        );
    }
    VAE_BACKWARD_APPEND_TENSOR(&workspace->logits);

#undef VAE_BACKWARD_APPEND_TENSOR

    return VAE_C_OP_OK;
}


static VAECOpStatus vae_backward_collect_batch_norms(
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


static VAECOpStatus vae_backward_collect_model_buffers(
    const VAECModel *model,
    const void **buffers,
    size_t capacity,
    size_t *count
)
{
    const VAECBatchNorm *batch_norms[
        VAE_BACKWARD_BATCH_NORM_CAPACITY
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
            status = vae_backward_append_pointer(
                buffers, capacity, count, regions[region]
            );
            if (status != VAE_C_OP_OK) {
                return status;
            }
        }
    }

    status = vae_backward_collect_batch_norms(
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
            status = vae_backward_append_pointer(
                buffers, capacity, count, regions[region]
            );
            if (status != VAE_C_OP_OK) {
                return status;
            }
        }
    }
    return vae_backward_append_pointer(
        buffers,
        capacity,
        count,
        model->encoder_max_pool.argmax_indices
    );
}


static VAECOpStatus vae_backward_validate_aliases(
    const VAECModel *model,
    const VAECModelWorkspace *workspace
)
{
    const VAECTensor *tensors[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ];
    const void *model_buffers[VAE_BACKWARD_MODEL_BUFFER_CAPACITY];
    const void *workspace_buffers[
        VAE_BACKWARD_WORKSPACE_BUFFER_CAPACITY
    ];
    size_t model_count = 0;
    size_t workspace_count = 0;
    size_t tensor_count = 0;
    size_t model_index;
    size_t other;
    size_t tensor_index;
    VAECOpStatus status;

    status = vae_backward_collect_model_buffers(
        model,
        model_buffers,
        sizeof(model_buffers) / sizeof(model_buffers[0]),
        &model_count
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    for (model_index = 0; model_index < model_count; model_index++) {
        for (other = model_index + 1; other < model_count; other++) {
            if (model_buffers[model_index] == model_buffers[other]) {
                return VAE_C_OP_INVALID_ARGUMENT;
            }
        }
    }

    status = vae_backward_collect_workspace_tensors(
        workspace,
        tensors,
        sizeof(tensors) / sizeof(tensors[0]),
        &tensor_count
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    for (tensor_index = 0; tensor_index < tensor_count; tensor_index++) {
        const VAECTensor *tensor = tensors[tensor_index];

        status = vae_backward_append_pointer(
            workspace_buffers,
            sizeof(workspace_buffers) / sizeof(workspace_buffers[0]),
            &workspace_count,
            tensor->data
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
        if (tensor->grad != NULL) {
            status = vae_backward_append_pointer(
                workspace_buffers,
                sizeof(workspace_buffers) /
                    sizeof(workspace_buffers[0]),
                &workspace_count,
                tensor->grad
            );
            if (status != VAE_C_OP_OK) {
                return status;
            }
        }
    }
    for (tensor_index = 0;
         tensor_index < workspace_count;
         tensor_index++) {

        for (model_index = 0;
             model_index < model_count;
             model_index++) {

            if (workspace_buffers[tensor_index] ==
                    model_buffers[model_index]) {

                return VAE_C_OP_INVALID_ARGUMENT;
            }
        }
    }
    return VAE_C_OP_OK;
}


static VAECOpStatus vae_backward_validate_workspace_values(
    const VAECModelWorkspace *workspace
)
{
    const VAECTensor *tensors[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ];
    size_t count = 0;
    size_t tensor_index;
    VAECOpStatus status;

    status = vae_backward_collect_workspace_tensors(
        workspace,
        tensors,
        sizeof(tensors) / sizeof(tensors[0]),
        &count
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    for (tensor_index = 0; tensor_index < count; tensor_index++) {
        const VAECTensor *tensor = tensors[tensor_index];

        if (!vae_backward_values_are_finite(
                tensor->data, tensor->shape.count)) {

            return VAE_C_OP_INVALID_STATE;
        }
        if (tensor->has_grad &&
            !vae_backward_values_are_zero(
                tensor->grad, tensor->shape.count)) {

            return VAE_C_OP_INVALID_STATE;
        }
    }
    for (tensor_index = 0;
         tensor_index < workspace->input.shape.count;
         tensor_index++) {

        float value = workspace->input.data[tensor_index];

        if (value < 0.0f || value > 1.0f) {
            return VAE_C_OP_INVALID_STATE;
        }
    }
    return VAE_C_OP_OK;
}


static VAECOpStatus vae_backward_validate_batch_norm_cache(
    const VAECBatchNorm *layer,
    const VAECTensor *input,
    const VAECTensor *output
)
{
    size_t spatial_count;
    size_t sample_count;

    if (layer == NULL || input == NULL || output == NULL ||
        !layer->cache_valid || layer->last_sample_count < 2 ||
        !vae_c_shape_equal(&layer->last_input_shape, &input->shape) ||
        !vae_c_shape_equal(&layer->last_output_shape, &output->shape)) {

        return VAE_C_OP_INVALID_STATE;
    }
    if (layer->kind == VAE_C_BATCH_NORM_KIND_1D) {
        if (input->shape.rank != 2) {
            return VAE_C_OP_INVALID_STATE;
        }
        sample_count = input->shape.dims[0];
    } else if (layer->kind == VAE_C_BATCH_NORM_KIND_2D) {
        if (input->shape.rank != 4 ||
            vae_backward_size_multiply(
                input->shape.dims[2],
                input->shape.dims[3],
                &spatial_count) != 0 ||
            vae_backward_size_multiply(
                input->shape.dims[0],
                spatial_count,
                &sample_count) != 0) {

            return VAE_C_OP_OVERFLOW;
        }
    } else {
        return VAE_C_OP_INVALID_STATE;
    }
    return layer->last_sample_count == sample_count
        ? VAE_C_OP_OK
        : VAE_C_OP_INVALID_STATE;
}


static VAECOpStatus vae_backward_validate_caches(
    const VAECModel *model,
    const VAECModelWorkspace *workspace
)
{
    const VAECMaxPool2D *pool = &model->encoder_max_pool;
    size_t input_count;
    size_t index;
    VAECOpStatus status;

    for (index = 0; index < model->config.conv_depth; index++) {
        status = vae_backward_validate_batch_norm_cache(
            &model->encoder_batch_norms[index],
            &workspace->encoder_conv_outputs[index],
            &workspace->encoder_batch_norm_outputs[index]
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
    }
    status = vae_backward_validate_batch_norm_cache(
        &model->decoder_fc_batch_norm,
        &workspace->decoder_fc_output,
        &workspace->decoder_fc_batch_norm_output
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    for (index = 0;
         index < model->config.decoder_hidden_conv_count;
         index++) {

        status = vae_backward_validate_batch_norm_cache(
            &model->decoder_hidden_batch_norms[index],
            &workspace->decoder_hidden_conv_outputs[index],
            &workspace->decoder_hidden_batch_norm_outputs[index]
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
    }

    input_count = workspace->encoder_relu_outputs[0].shape.count;
    if (!pool->cache_valid ||
        !vae_c_shape_equal(
            &pool->last_input_shape,
            &workspace->encoder_relu_outputs[0].shape) ||
        !vae_c_shape_equal(
            &pool->last_output_shape,
            &workspace->encoder_pool_output.shape) ||
        pool->last_count != workspace->encoder_pool_output.shape.count ||
        pool->last_count > pool->capacity) {

        return VAE_C_OP_INVALID_STATE;
    }
    for (index = 0; index < pool->last_count; index++) {
        if (pool->argmax_indices[index] >= input_count) {
            return VAE_C_OP_INVALID_STATE;
        }
    }
    return VAE_C_OP_OK;
}


static VAECOpStatus vae_backward_preflight(
    VAECModel *model,
    VAECModelWorkspace *workspace,
    float upstream_gradient
)
{
    VAECOpStatus status;
    size_t depth;

    if (model == NULL || workspace == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (!isfinite(upstream_gradient)) {
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
        workspace->vae_backward_model != model ||
        workspace->vae_backward_generation == UINT64_C(0) ||
        workspace->vae_backward_generation !=
            model->vae_cache_generation) {

        return VAE_C_OP_INVALID_STATE;
    }

    depth = model->config.conv_depth;
    if ((depth != 3 && depth != 4) ||
        workspace->config.conv_depth != depth ||
        model->config.decoder_hidden_conv_count != depth - 1 ||
        model->config.decoder_upsample_count != depth + 1 ||
        workspace->config.decoder_hidden_conv_count != depth - 1 ||
        workspace->config.decoder_upsample_count != depth + 1 ||
        workspace->batch_size < 2 ||
        workspace->batch_size > model->max_batch_size ||
        workspace->input.shape.rank != 4 ||
        !vae_c_shape_equal(
            &workspace->input.shape, &workspace->logits.shape) ||
        !vae_c_shape_equal(
            &workspace->mu.shape, &workspace->logvar.shape) ||
        !vae_c_shape_equal(
            &workspace->mu.shape, &workspace->epsilon.shape) ||
        !vae_c_shape_equal(
            &workspace->mu.shape, &workspace->z.shape)) {

        return VAE_C_OP_INVALID_STATE;
    }
    status = vae_backward_validate_aliases(model, workspace);
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_backward_validate_workspace_values(workspace);
    if (status != VAE_C_OP_OK) {
        return status;
    }
    return vae_backward_validate_caches(model, workspace);
}


static VAECOpStatus vae_backward_run_decoder(
    VAECModel *model,
    VAECModelWorkspace *workspace
)
{
    size_t hidden_count = model->config.decoder_hidden_conv_count;
    size_t index;
    VAECOpStatus status;

    status = vae_c_conv2d_backward(
        &model->decoder_output_conv,
        &workspace->decoder_upsample_outputs[
            model->config.decoder_upsample_count - 1
        ],
        &workspace->logits
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    for (index = hidden_count; index-- > 0;) {
        status = vae_c_nearest_upsample2d_backward(
            &workspace->decoder_hidden_relu_outputs[index],
            &workspace->decoder_upsample_outputs[index + 2]
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
        status = vae_c_relu_backward(
            &workspace->decoder_hidden_batch_norm_outputs[index],
            &workspace->decoder_hidden_relu_outputs[index]
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
        status = vae_c_batch_norm2d_backward(
            &model->decoder_hidden_batch_norms[index],
            &workspace->decoder_hidden_conv_outputs[index],
            &workspace->decoder_hidden_batch_norm_outputs[index]
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
        status = vae_c_conv2d_backward(
            &model->decoder_hidden_convs[index],
            &workspace->decoder_upsample_outputs[index + 1],
            &workspace->decoder_hidden_conv_outputs[index]
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
    }
    status = vae_c_nearest_upsample2d_backward(
        &workspace->decoder_upsample_outputs[0],
        &workspace->decoder_upsample_outputs[1]
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_nearest_upsample2d_backward(
        &workspace->decoder_reshape_view,
        &workspace->decoder_upsample_outputs[0]
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_relu_backward(
        &workspace->decoder_fc_batch_norm_output,
        &workspace->decoder_fc_relu_output
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_batch_norm1d_backward(
        &model->decoder_fc_batch_norm,
        &workspace->decoder_fc_output,
        &workspace->decoder_fc_batch_norm_output
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    return vae_c_dense_backward(
        &model->decoder_fc,
        &workspace->z,
        &workspace->decoder_fc_output
    );
}


static VAECOpStatus vae_backward_run_encoder(
    VAECModel *model,
    VAECModelWorkspace *workspace
)
{
    size_t depth = model->config.conv_depth;
    size_t index;
    VAECOpStatus status;

    status = vae_c_dense_backward(
        &model->encoder_fc,
        &workspace->encoder_global_avg_output,
        &workspace->encoder_packed_output
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_global_avg_pool2d_backward(
        &workspace->encoder_relu_outputs[depth - 1],
        &workspace->encoder_global_avg_output
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    for (index = depth; index-- > 1;) {
        const VAECTensor *input = index == 1
            ? &workspace->encoder_pool_output
            : &workspace->encoder_relu_outputs[index - 1];

        status = vae_c_relu_backward(
            &workspace->encoder_batch_norm_outputs[index],
            &workspace->encoder_relu_outputs[index]
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
        status = vae_c_batch_norm2d_backward(
            &model->encoder_batch_norms[index],
            &workspace->encoder_conv_outputs[index],
            &workspace->encoder_batch_norm_outputs[index]
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
        status = vae_c_conv2d_backward(
            &model->encoder_convs[index],
            input,
            &workspace->encoder_conv_outputs[index]
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
    }
    status = vae_c_max_pool2d_backward(
        &model->encoder_max_pool,
        &workspace->encoder_relu_outputs[0],
        &workspace->encoder_pool_output
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_relu_backward(
        &workspace->encoder_batch_norm_outputs[0],
        &workspace->encoder_relu_outputs[0]
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_batch_norm2d_backward(
        &model->encoder_batch_norms[0],
        &workspace->encoder_conv_outputs[0],
        &workspace->encoder_batch_norm_outputs[0]
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    return vae_c_conv2d_backward(
        &model->encoder_convs[0],
        &workspace->input,
        &workspace->encoder_conv_outputs[0]
    );
}


VAECOpStatus vae_c_model_vae_backward(
    VAECModel *model,
    VAECModelWorkspace *workspace,
    float upstream_gradient
)
{
    VAECOpStatus status;

    status = vae_backward_preflight(
        model, workspace, upstream_gradient
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }

    workspace->vae_backward_model = NULL;
    workspace->vae_backward_generation = UINT64_C(0);

    status = vae_c_vae_loss_backward(
        &workspace->logits,
        &workspace->input,
        &workspace->mu,
        &workspace->logvar,
        upstream_gradient
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_backward_run_decoder(model, workspace);
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_reparameterize_backward(
        VAE_C_LATENT_MODE_STOCHASTIC,
        &workspace->mu,
        &workspace->logvar,
        &workspace->epsilon,
        &workspace->z
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_latent_split_backward(
        &workspace->encoder_packed_output,
        &workspace->mu,
        &workspace->logvar
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    return vae_backward_run_encoder(model, workspace);
}
