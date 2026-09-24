#include "vae_c_iic_pair_train.h"

#include <math.h>
#include <stdint.h>


#define IIC_PAIR_TRAIN_PARAMETER_COUNT 10


static void train_parameters(
    VAECIICPair *pair,
    VAECParameter *parameters[IIC_PAIR_TRAIN_PARAMETER_COUNT]
)
{
    parameters[0] = &pair->main_head.linear1.weights;
    parameters[1] = &pair->main_head.batch_norm.gamma;
    parameters[2] = &pair->main_head.batch_norm.beta;
    parameters[3] = &pair->main_head.linear2.weights;
    parameters[4] = &pair->main_head.linear2.bias;
    parameters[5] = &pair->overcluster_head.linear1.weights;
    parameters[6] = &pair->overcluster_head.batch_norm.gamma;
    parameters[7] = &pair->overcluster_head.batch_norm.beta;
    parameters[8] = &pair->overcluster_head.linear2.weights;
    parameters[9] = &pair->overcluster_head.linear2.bias;
}


/* The pair's input preflight is private. Check inputs, storage ownership,
 * aliases, gradients and head/BatchNorm counter headroom before zero_grad.
 * Neural operations and Adam remain delegated to their frozen cores. */
static bool train_pointer_aliases_head(
    const VAECIICHead *head,
    const void *pointer
)
{
    const VAECParameter *parameters[5] = {
        &head->linear1.weights, &head->batch_norm.gamma,
        &head->batch_norm.beta, &head->linear2.weights,
        &head->linear2.bias
    };
    const void *bn_buffers[8] = {
        head->batch_norm.running_mean,
        head->batch_norm.running_variance,
        head->batch_norm.batch_mean,
        head->batch_norm.batch_inv_std,
        head->batch_norm.work_mean,
        head->batch_norm.work_inv_std,
        head->batch_norm.work_unbiased_variance,
        head->batch_norm.owned_block
    };
    size_t index;

    for (index = 0; index < 5; index++) {
        const VAECParameter *parameter = parameters[index];
        if (pointer == parameter->values ||
            pointer == parameter->gradients ||
            pointer == parameter->adam_m ||
            pointer == parameter->adam_v ||
            pointer == parameter->owned_block) {
            return true;
        }
    }
    for (index = 0; index < 8; index++) {
        if (pointer == bn_buffers[index]) {
            return true;
        }
    }
    return false;
}


static bool train_pointer_aliases_workspace(
    const VAECIICHeadWorkspace *workspace,
    const void *pointer
)
{
    const VAECTensor *tensors[5] = {
        &workspace->linear1_storage,
        &workspace->batch_norm_storage,
        &workspace->relu_storage,
        &workspace->logits_storage,
        &workspace->softmax_storage
    };
    size_t index;

    for (index = 0; index < 5; index++) {
        if (pointer == tensors[index]->data ||
            pointer == tensors[index]->grad ||
            pointer == tensors[index]->owned_block) {
            return true;
        }
    }
    return false;
}


static bool train_input_aliases_owned_buffers(
    const VAECIICPair *pair,
    const VAECIICPairWorkspace *workspace,
    const VAECTensor *input
)
{
    const VAECIICHeadWorkspace *views[4] = {
        &workspace->main_a, &workspace->over_a,
        &workspace->main_b, &workspace->over_b
    };
    size_t index;

    if (train_pointer_aliases_head(&pair->main_head, input->data) ||
        train_pointer_aliases_head(&pair->main_head, input->grad) ||
        train_pointer_aliases_head(
            &pair->overcluster_head, input->data) ||
        train_pointer_aliases_head(
            &pair->overcluster_head, input->grad)) {
        return true;
    }
    for (index = 0; index < 4; index++) {
        if (train_pointer_aliases_workspace(views[index], input->data) ||
            train_pointer_aliases_workspace(views[index], input->grad)) {
            return true;
        }
    }
    return false;
}


static VAECOpStatus train_input_preflight(
    const VAECIICPair *pair,
    const VAECIICPairWorkspace *workspace,
    const VAECTensor *input_a,
    const VAECTensor *input_b
)
{
    const VAECTensor *inputs[2] = {input_a, input_b};
    size_t batch_size;
    size_t index;

    if (input_a == NULL || input_b == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (input_a->shape.rank != 2 || input_b->shape.rank != 2 ||
        input_a->shape.dims[1] != pair->input_dim ||
        input_b->shape.dims[1] != pair->input_dim ||
        input_a->shape.dims[0] != input_b->shape.dims[0]) {
        return VAE_C_OP_SHAPE_MISMATCH;
    }
    batch_size = input_a->shape.dims[0];
    if (batch_size < 2 || batch_size > workspace->capacity) {
        return VAE_C_OP_SHAPE_MISMATCH;
    }
    if (pair->main_head.forward_generation > UINT64_MAX - 2 ||
        pair->overcluster_head.forward_generation > UINT64_MAX - 2 ||
        pair->main_head.num_batches_tracked > UINT64_MAX - 2 ||
        pair->overcluster_head.num_batches_tracked > UINT64_MAX - 2) {
        return VAE_C_OP_OVERFLOW;
    }
    for (index = 0; index < 2; index++) {
        const VAECTensor *input = inputs[index];
        const size_t dims[2] = {batch_size, pair->input_dim};
        VAECShape expected;
        size_t element;

        if (vae_c_shape_make(2, dims, &expected) != VAE_C_STORAGE_OK ||
            !vae_c_shape_equal(&input->shape, &expected) ||
            input->data == NULL) {
            return VAE_C_OP_INVALID_STATE;
        }
        if (!input->has_grad || input->grad == NULL) {
            return VAE_C_OP_GRADIENT_NOT_AVAILABLE;
        }
        if (input->data == input->grad) {
            return VAE_C_OP_INVALID_STATE;
        }
        if (input->owns_memory) {
            size_t required_bytes;

            if (input->owned_block == NULL ||
                input->data != input->owned_block ||
                expected.count > SIZE_MAX / (2 * sizeof(float))) {
                return VAE_C_OP_INVALID_STATE;
            }
            required_bytes = expected.count * 2 * sizeof(float);
            if (input->owned_bytes != required_bytes ||
                input->grad != input->data + expected.count) {
                return VAE_C_OP_INVALID_STATE;
            }
        } else if (input->owned_block != NULL ||
                   input->owned_bytes != 0) {
            return VAE_C_OP_INVALID_STATE;
        }
        if (train_input_aliases_owned_buffers(pair, workspace, input)) {
            return VAE_C_OP_INVALID_ARGUMENT;
        }
        for (element = 0; element < expected.count; element++) {
            if (!isfinite(input->data[element]) ||
                !isfinite(input->grad[element])) {
                return VAE_C_OP_INVALID_STATE;
            }
        }
    }
    if (input_a != input_b &&
        (input_a->data == input_b->grad ||
         input_a->grad == input_b->data)) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    return VAE_C_OP_OK;
}


static VAECOpStatus train_gradients_finite(
    VAECParameter *const parameters[IIC_PAIR_TRAIN_PARAMETER_COUNT]
)
{
    size_t parameter_index;

    for (parameter_index = 0;
         parameter_index < IIC_PAIR_TRAIN_PARAMETER_COUNT;
         parameter_index++) {
        const VAECParameter *parameter = parameters[parameter_index];
        size_t element;

        for (element = 0; element < parameter->shape.count; element++) {
            if (!isfinite(parameter->gradients[element])) {
                return VAE_C_OP_INVALID_STATE;
            }
        }
    }
    return VAE_C_OP_OK;
}


VAECOpStatus vae_c_iic_pair_train_step(
    VAECIICPair *pair,
    VAECIICPairWorkspace *workspace,
    VAECAdam *optimizer,
    const VAECTensor *input_a,
    const VAECTensor *input_b
)
{
    VAECParameter *parameters[IIC_PAIR_TRAIN_PARAMETER_COUNT];
    VAECOpStatus status;

    if (pair == NULL || workspace == NULL || optimizer == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (vae_c_iic_pair_validate(pair) != VAE_C_OP_OK ||
        vae_c_iic_pair_workspace_validate(pair, workspace) != VAE_C_OP_OK ||
        !vae_c_adam_is_configured(optimizer)) {
        return VAE_C_OP_INVALID_STATE;
    }
    status = train_input_preflight(pair, workspace, input_a, input_b);
    if (status != VAE_C_OP_OK) {
        return status;
    }

    status = vae_c_iic_pair_zero_grad(pair);
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_iic_pair_forward_train(
        pair, input_a, input_b, workspace
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_iic_pair_backward(
        pair, input_a, input_b, workspace,
        VAE_C_IIC_PAIR_BACKWARD_A_THEN_B
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }

    train_parameters(pair, parameters);
    status = train_gradients_finite(parameters);
    if (status != VAE_C_OP_OK) {
        return status;
    }
    return vae_c_adam_step(
        optimizer, parameters, IIC_PAIR_TRAIN_PARAMETER_COUNT
    );
}
