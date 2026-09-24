#include "vae_c_iic_pair.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>


static bool pair_product(size_t left, size_t right, size_t *product)
{
    if (product == NULL || left == 0 || right == 0 ||
        left > SIZE_MAX / right) {
        return false;
    }
    *product = left * right;
    return true;
}


static VAECOpStatus pair_loss_status(VAECIICLossStatus status)
{
    switch (status) {
        case VAE_C_IIC_LOSS_OK:
            return VAE_C_OP_OK;
        case VAE_C_IIC_LOSS_ERROR_NULL_ARGUMENT:
        case VAE_C_IIC_LOSS_ERROR_ALIASING:
        case VAE_C_IIC_LOSS_ERROR_DIMENSIONS:
            return VAE_C_OP_INVALID_ARGUMENT;
        case VAE_C_IIC_LOSS_ERROR_SHAPE_MISMATCH:
        case VAE_C_IIC_LOSS_ERROR_CAPACITY:
            return VAE_C_OP_SHAPE_MISMATCH;
        case VAE_C_IIC_LOSS_ERROR_OVERFLOW:
            return VAE_C_OP_OVERFLOW;
        default:
            return VAE_C_OP_INVALID_STATE;
    }
}


void vae_c_iic_pair_init(VAECIICPair *pair)
{
    if (pair != NULL) {
        memset(pair, 0, sizeof(*pair));
    }
}


void vae_c_iic_pair_free(VAECIICPair *pair)
{
    if (pair == NULL) {
        return;
    }
    vae_c_iic_head_free(&pair->main_head);
    vae_c_iic_head_free(&pair->overcluster_head);
    vae_c_iic_pair_init(pair);
}


bool vae_c_iic_pair_is_empty(const VAECIICPair *pair)
{
    return pair != NULL && !pair->allocated && pair->input_dim == 0 &&
        pair->main_output_dim == 0 && pair->overcluster_output_dim == 0 &&
        vae_c_iic_head_is_empty(&pair->main_head) &&
        vae_c_iic_head_is_empty(&pair->overcluster_head);
}


static bool pair_heads_distinct(const VAECIICPair *pair)
{
    const VAECParameter *main_parameters[5] = {
        &pair->main_head.linear1.weights,
        &pair->main_head.batch_norm.gamma,
        &pair->main_head.batch_norm.beta,
        &pair->main_head.linear2.weights,
        &pair->main_head.linear2.bias
    };
    const VAECParameter *over_parameters[5] = {
        &pair->overcluster_head.linear1.weights,
        &pair->overcluster_head.batch_norm.gamma,
        &pair->overcluster_head.batch_norm.beta,
        &pair->overcluster_head.linear2.weights,
        &pair->overcluster_head.linear2.bias
    };
    size_t i;
    size_t j;

    if (pair->main_head.batch_norm.owned_block ==
        pair->overcluster_head.batch_norm.owned_block) {
        return false;
    }
    for (i = 0; i < 5; i++) {
        if (main_parameters[i]->owned_block ==
                pair->overcluster_head.batch_norm.owned_block ||
            over_parameters[i]->owned_block ==
                pair->main_head.batch_norm.owned_block) {
            return false;
        }
        for (j = 0; j < 5; j++) {
            if (main_parameters[i]->owned_block ==
                over_parameters[j]->owned_block) {
                return false;
            }
        }
    }
    return true;
}


static bool pair_pointer_aliases_head(
    const VAECIICHead *head,
    const void *pointer
)
{
    const VAECParameter *parameters[5] = {
        &head->linear1.weights, &head->batch_norm.gamma,
        &head->batch_norm.beta, &head->linear2.weights,
        &head->linear2.bias
    };
    const void *batch_norm_buffers[8] = {
        head->batch_norm.running_mean,
        head->batch_norm.running_variance,
        head->batch_norm.batch_mean,
        head->batch_norm.batch_inv_std,
        head->batch_norm.work_mean,
        head->batch_norm.work_inv_std,
        head->batch_norm.work_unbiased_variance,
        head->batch_norm.owned_block
    };
    size_t i;

    if (pointer == NULL) {
        return false;
    }
    for (i = 0; i < 5; i++) {
        if (pointer == parameters[i]->values ||
            pointer == parameters[i]->gradients ||
            pointer == parameters[i]->adam_m ||
            pointer == parameters[i]->adam_v ||
            pointer == parameters[i]->owned_block) {
            return true;
        }
    }
    for (i = 0; i < 8; i++) {
        if (pointer == batch_norm_buffers[i]) {
            return true;
        }
    }
    return false;
}


static bool pair_pointer_aliases_workspace(
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
    size_t i;

    if (pointer == NULL) {
        return false;
    }
    for (i = 0; i < 5; i++) {
        if (pointer == tensors[i]->data ||
            pointer == tensors[i]->grad ||
            pointer == tensors[i]->owned_block) {
            return true;
        }
    }
    return false;
}


static bool pair_input_aliases_owned_buffers(
    const VAECIICPair *pair,
    const VAECIICPairWorkspace *workspace,
    const VAECTensor *input
)
{
    const VAECIICHeadWorkspace *views[4] = {
        &workspace->main_a, &workspace->over_a,
        &workspace->main_b, &workspace->over_b
    };
    size_t i;

    if (pair_pointer_aliases_head(&pair->main_head, input->data) ||
        pair_pointer_aliases_head(&pair->main_head, input->grad) ||
        pair_pointer_aliases_head(
            &pair->overcluster_head, input->data) ||
        pair_pointer_aliases_head(
            &pair->overcluster_head, input->grad)) {
        return true;
    }
    for (i = 0; i < 4; i++) {
        if (pair_pointer_aliases_workspace(views[i], input->data) ||
            pair_pointer_aliases_workspace(views[i], input->grad)) {
            return true;
        }
    }
    return false;
}


VAECOpStatus vae_c_iic_pair_validate(const VAECIICPair *pair)
{
    if (pair == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (!pair->allocated || pair->input_dim == 0 ||
        pair->main_output_dim == 0 || pair->overcluster_output_dim == 0 ||
        vae_c_iic_head_validate(&pair->main_head) != VAE_C_OP_OK ||
        vae_c_iic_head_validate(&pair->overcluster_head) != VAE_C_OP_OK ||
        pair->main_head.input_dim != pair->input_dim ||
        pair->overcluster_head.input_dim != pair->input_dim ||
        pair->main_head.output_dim != pair->main_output_dim ||
        pair->overcluster_head.output_dim !=
            pair->overcluster_output_dim ||
        !pair_heads_distinct(pair)) {
        return VAE_C_OP_INVALID_STATE;
    }
    return VAE_C_OP_OK;
}


VAECOpStatus vae_c_iic_pair_allocate(
    VAECIICPair *pair,
    size_t input_dim,
    size_t main_output_dim,
    size_t overcluster_output_dim
)
{
    VAECIICPair candidate;
    VAECOpStatus status;

    if (pair == NULL || input_dim == 0 || main_output_dim == 0 ||
        overcluster_output_dim == 0) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (!vae_c_iic_pair_is_empty(pair)) {
        return VAE_C_OP_OBJECT_NOT_EMPTY;
    }
    vae_c_iic_pair_init(&candidate);
    status = vae_c_iic_head_allocate(
        &candidate.main_head, input_dim, main_output_dim
    );
    if (status == VAE_C_OP_OK) {
        status = vae_c_iic_head_allocate(
            &candidate.overcluster_head, input_dim,
            overcluster_output_dim
        );
    }
    if (status != VAE_C_OP_OK) {
        vae_c_iic_pair_free(&candidate);
        return status;
    }
    candidate.input_dim = input_dim;
    candidate.main_output_dim = main_output_dim;
    candidate.overcluster_output_dim = overcluster_output_dim;
    candidate.allocated = true;
    if (vae_c_iic_pair_validate(&candidate) != VAE_C_OP_OK) {
        vae_c_iic_pair_free(&candidate);
        return VAE_C_OP_INVALID_STATE;
    }
    *pair = candidate;
    return VAE_C_OP_OK;
}


VAECOpStatus vae_c_iic_pair_zero_grad(VAECIICPair *pair)
{
    VAECOpStatus status;

    if (pair == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (vae_c_iic_pair_is_empty(pair)) {
        return VAE_C_OP_OK;
    }
    if (vae_c_iic_pair_validate(pair) != VAE_C_OP_OK) {
        return VAE_C_OP_INVALID_STATE;
    }
    status = vae_c_iic_head_zero_grad(&pair->main_head);
    if (status != VAE_C_OP_OK) {
        return status;
    }
    return vae_c_iic_head_zero_grad(&pair->overcluster_head);
}


void vae_c_iic_pair_workspace_init(VAECIICPairWorkspace *workspace)
{
    if (workspace != NULL) {
        memset(workspace, 0, sizeof(*workspace));
    }
}


void vae_c_iic_pair_workspace_free(VAECIICPairWorkspace *workspace)
{
    if (workspace == NULL) {
        return;
    }
    vae_c_iic_head_workspace_free(&workspace->main_a);
    vae_c_iic_head_workspace_free(&workspace->over_a);
    vae_c_iic_head_workspace_free(&workspace->main_b);
    vae_c_iic_head_workspace_free(&workspace->over_b);
    free(workspace->main_joint);
    free(workspace->main_marginal_i);
    free(workspace->main_marginal_j);
    free(workspace->over_joint);
    free(workspace->over_marginal_i);
    free(workspace->over_marginal_j);
    free(workspace->main_a_loss_grad);
    free(workspace->main_b_loss_grad);
    free(workspace->over_a_loss_grad);
    free(workspace->over_b_loss_grad);
    vae_c_iic_pair_workspace_init(workspace);
}


bool vae_c_iic_pair_workspace_is_empty(
    const VAECIICPairWorkspace *workspace
)
{
    return workspace != NULL && !workspace->allocated &&
        workspace->capacity == 0 && workspace->batch_size == 0 &&
        vae_c_iic_head_workspace_is_empty(&workspace->main_a) &&
        vae_c_iic_head_workspace_is_empty(&workspace->over_a) &&
        vae_c_iic_head_workspace_is_empty(&workspace->main_b) &&
        vae_c_iic_head_workspace_is_empty(&workspace->over_b) &&
        workspace->main_joint == NULL &&
        workspace->main_marginal_i == NULL &&
        workspace->main_marginal_j == NULL &&
        workspace->over_joint == NULL &&
        workspace->over_marginal_i == NULL &&
        workspace->over_marginal_j == NULL &&
        workspace->main_a_loss_grad == NULL &&
        workspace->main_b_loss_grad == NULL &&
        workspace->over_a_loss_grad == NULL &&
        workspace->over_b_loss_grad == NULL &&
        workspace->backward_pair == NULL;
}


static bool pair_loss_buffers_valid(
    const VAECIICLossResult *result,
    const float *joint,
    const float *marginal_i,
    const float *marginal_j,
    size_t class_count
)
{
    size_t joint_count;

    return pair_product(class_count, class_count, &joint_count) &&
        joint != NULL && marginal_i != NULL && marginal_j != NULL &&
        result->joint == joint && result->joint_capacity == joint_count &&
        result->marginal_i == marginal_i &&
        result->marginal_i_capacity == class_count &&
        result->marginal_j == marginal_j &&
        result->marginal_j_capacity == class_count;
}


static bool pair_workspaces_distinct(
    const VAECIICPairWorkspace *workspace
)
{
    const VAECIICHeadWorkspace *views[4] = {
        &workspace->main_a, &workspace->over_a,
        &workspace->main_b, &workspace->over_b
    };
    const VAECTensor *tensors[4][5];
    size_t i;
    size_t j;
    size_t k;
    size_t m;

    for (i = 0; i < 4; i++) {
        tensors[i][0] = &views[i]->linear1_storage;
        tensors[i][1] = &views[i]->batch_norm_storage;
        tensors[i][2] = &views[i]->relu_storage;
        tensors[i][3] = &views[i]->logits_storage;
        tensors[i][4] = &views[i]->softmax_storage;
    }
    for (i = 0; i < 4; i++) {
        for (j = i + 1; j < 4; j++) {
            for (k = 0; k < 5; k++) {
                for (m = 0; m < 5; m++) {
                    if (tensors[i][k]->owned_block ==
                        tensors[j][m]->owned_block) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}


VAECOpStatus vae_c_iic_pair_workspace_validate(
    const VAECIICPair *pair,
    const VAECIICPairWorkspace *workspace
)
{
    size_t main_gradient_count;
    size_t over_gradient_count;

    if (pair == NULL || workspace == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (!pair_product(
            workspace->capacity, pair->main_output_dim,
            &main_gradient_count) ||
        !pair_product(
            workspace->capacity, pair->overcluster_output_dim,
            &over_gradient_count) ||
        main_gradient_count > SIZE_MAX / sizeof(float) ||
        over_gradient_count > SIZE_MAX / sizeof(float)) {
        return VAE_C_OP_INVALID_STATE;
    }
    if (vae_c_iic_pair_validate(pair) != VAE_C_OP_OK ||
        !workspace->allocated || workspace->capacity < 2 ||
        workspace->batch_size > workspace->capacity ||
        vae_c_iic_head_workspace_validate(
            &pair->main_head, &workspace->main_a) != VAE_C_OP_OK ||
        vae_c_iic_head_workspace_validate(
            &pair->main_head, &workspace->main_b) != VAE_C_OP_OK ||
        vae_c_iic_head_workspace_validate(
            &pair->overcluster_head, &workspace->over_a) != VAE_C_OP_OK ||
        vae_c_iic_head_workspace_validate(
            &pair->overcluster_head, &workspace->over_b) != VAE_C_OP_OK ||
        workspace->main_a.capacity != workspace->capacity ||
        workspace->main_b.capacity != workspace->capacity ||
        workspace->over_a.capacity != workspace->capacity ||
        workspace->over_b.capacity != workspace->capacity ||
        !pair_workspaces_distinct(workspace) ||
        !pair_loss_buffers_valid(
            &workspace->main_result, workspace->main_joint,
            workspace->main_marginal_i, workspace->main_marginal_j,
            pair->main_output_dim) ||
        !pair_loss_buffers_valid(
            &workspace->over_result, workspace->over_joint,
            workspace->over_marginal_i, workspace->over_marginal_j,
            pair->overcluster_output_dim) ||
        workspace->main_a_loss_grad == NULL ||
        workspace->main_b_loss_grad == NULL ||
        workspace->over_a_loss_grad == NULL ||
        workspace->over_b_loss_grad == NULL) {
        return VAE_C_OP_INVALID_STATE;
    }
    return VAE_C_OP_OK;
}


static VAECOpStatus pair_allocate_loss_buffers(
    VAECIICPairWorkspace *workspace,
    size_t capacity,
    size_t main_count,
    size_t over_count
)
{
    size_t main_joint_count;
    size_t over_joint_count;
    size_t main_gradient_count;
    size_t over_gradient_count;

    if (!pair_product(main_count, main_count, &main_joint_count) ||
        !pair_product(over_count, over_count, &over_joint_count) ||
        !pair_product(capacity, main_count, &main_gradient_count) ||
        !pair_product(capacity, over_count, &over_gradient_count) ||
        main_joint_count > SIZE_MAX / sizeof(float) ||
        over_joint_count > SIZE_MAX / sizeof(float) ||
        main_gradient_count > SIZE_MAX / sizeof(float) ||
        over_gradient_count > SIZE_MAX / sizeof(float) ||
        main_count > SIZE_MAX / sizeof(float) ||
        over_count > SIZE_MAX / sizeof(float)) {
        return VAE_C_OP_OVERFLOW;
    }
    workspace->main_joint = calloc(main_joint_count, sizeof(float));
    workspace->main_marginal_i = calloc(main_count, sizeof(float));
    workspace->main_marginal_j = calloc(main_count, sizeof(float));
    workspace->over_joint = calloc(over_joint_count, sizeof(float));
    workspace->over_marginal_i = calloc(over_count, sizeof(float));
    workspace->over_marginal_j = calloc(over_count, sizeof(float));
    workspace->main_a_loss_grad = calloc(
        main_gradient_count, sizeof(float)
    );
    workspace->main_b_loss_grad = calloc(
        main_gradient_count, sizeof(float)
    );
    workspace->over_a_loss_grad = calloc(
        over_gradient_count, sizeof(float)
    );
    workspace->over_b_loss_grad = calloc(
        over_gradient_count, sizeof(float)
    );
    if (workspace->main_joint == NULL ||
        workspace->main_marginal_i == NULL ||
        workspace->main_marginal_j == NULL ||
        workspace->over_joint == NULL ||
        workspace->over_marginal_i == NULL ||
        workspace->over_marginal_j == NULL ||
        workspace->main_a_loss_grad == NULL ||
        workspace->main_b_loss_grad == NULL ||
        workspace->over_a_loss_grad == NULL ||
        workspace->over_b_loss_grad == NULL) {
        return VAE_C_OP_ALLOCATION_FAILED;
    }
    vae_c_iic_loss_result_init(
        &workspace->main_result, workspace->main_joint,
        main_joint_count, workspace->main_marginal_i, main_count,
        workspace->main_marginal_j, main_count
    );
    vae_c_iic_loss_result_init(
        &workspace->over_result, workspace->over_joint,
        over_joint_count, workspace->over_marginal_i, over_count,
        workspace->over_marginal_j, over_count
    );
    return VAE_C_OP_OK;
}


VAECOpStatus vae_c_iic_pair_workspace_allocate(
    const VAECIICPair *pair,
    size_t capacity,
    VAECIICPairWorkspace *workspace
)
{
    VAECIICPairWorkspace candidate;
    VAECOpStatus status;

    if (pair == NULL || workspace == NULL || capacity < 2) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (!vae_c_iic_pair_workspace_is_empty(workspace)) {
        return VAE_C_OP_OBJECT_NOT_EMPTY;
    }
    if (vae_c_iic_pair_validate(pair) != VAE_C_OP_OK) {
        return VAE_C_OP_INVALID_STATE;
    }
    vae_c_iic_pair_workspace_init(&candidate);
    status = vae_c_iic_head_workspace_allocate(
        &pair->main_head, capacity, &candidate.main_a
    );
    if (status == VAE_C_OP_OK) {
        status = vae_c_iic_head_workspace_allocate(
            &pair->overcluster_head, capacity, &candidate.over_a
        );
    }
    if (status == VAE_C_OP_OK) {
        status = vae_c_iic_head_workspace_allocate(
            &pair->main_head, capacity, &candidate.main_b
        );
    }
    if (status == VAE_C_OP_OK) {
        status = vae_c_iic_head_workspace_allocate(
            &pair->overcluster_head, capacity, &candidate.over_b
        );
    }
    if (status == VAE_C_OP_OK) {
        status = pair_allocate_loss_buffers(
            &candidate, capacity, pair->main_output_dim,
            pair->overcluster_output_dim
        );
    }
    if (status != VAE_C_OP_OK) {
        vae_c_iic_pair_workspace_free(&candidate);
        return status;
    }
    candidate.capacity = capacity;
    candidate.allocated = true;
    if (vae_c_iic_pair_workspace_validate(pair, &candidate) !=
        VAE_C_OP_OK) {
        vae_c_iic_pair_workspace_free(&candidate);
        return VAE_C_OP_INVALID_STATE;
    }
    *workspace = candidate;
    return VAE_C_OP_OK;
}


VAECOpStatus vae_c_iic_pair_workspace_zero_grad(
    const VAECIICPair *pair,
    VAECIICPairWorkspace *workspace
)
{
    VAECOpStatus status;

    if (pair == NULL || workspace == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (vae_c_iic_pair_workspace_validate(pair, workspace) !=
        VAE_C_OP_OK) {
        return VAE_C_OP_INVALID_STATE;
    }
    status = vae_c_iic_head_workspace_zero_grad(
        &pair->main_head, &workspace->main_a
    );
    if (status == VAE_C_OP_OK) {
        status = vae_c_iic_head_workspace_zero_grad(
            &pair->overcluster_head, &workspace->over_a
        );
    }
    if (status == VAE_C_OP_OK) {
        status = vae_c_iic_head_workspace_zero_grad(
            &pair->main_head, &workspace->main_b
        );
    }
    if (status == VAE_C_OP_OK) {
        status = vae_c_iic_head_workspace_zero_grad(
            &pair->overcluster_head, &workspace->over_b
        );
    }
    return status;
}


static VAECOpStatus pair_input_preflight(
    const VAECIICPair *pair,
    const VAECIICPairWorkspace *workspace,
    const VAECTensor *input_a,
    const VAECTensor *input_b
)
{
    const VAECTensor *inputs[2] = {input_a, input_b};
    size_t index;
    size_t batch_size;

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
    for (index = 0; index < 2; index++) {
        const VAECTensor *input = inputs[index];
        VAECShape expected;
        const size_t dims[2] = {batch_size, pair->input_dim};
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
        if (pair_input_aliases_owned_buffers(pair, workspace, input)) {
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


static void pair_invalidate_context(VAECIICPairWorkspace *workspace)
{
    workspace->backward_pair = NULL;
    workspace->backward_input_a = NULL;
    workspace->backward_input_b = NULL;
    workspace->main_a_generation = 0;
    workspace->main_b_generation = 0;
    workspace->over_a_generation = 0;
    workspace->over_b_generation = 0;
    workspace->batch_size = 0;
}


VAECOpStatus vae_c_iic_pair_forward_train(
    VAECIICPair *pair,
    const VAECTensor *input_a,
    const VAECTensor *input_b,
    VAECIICPairWorkspace *workspace
)
{
    VAECOpStatus status;
    VAECIICLossStatus loss_status;
    size_t batch_size;
    float total_loss;

    if (pair == NULL || workspace == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (vae_c_iic_pair_workspace_validate(pair, workspace) !=
        VAE_C_OP_OK) {
        return VAE_C_OP_INVALID_STATE;
    }
    status = pair_input_preflight(pair, workspace, input_a, input_b);
    if (status != VAE_C_OP_OK) {
        return status;
    }
    if (pair->main_head.forward_generation > UINT64_MAX - 2 ||
        pair->overcluster_head.forward_generation > UINT64_MAX - 2 ||
        pair->main_head.num_batches_tracked > UINT64_MAX - 2 ||
        pair->overcluster_head.num_batches_tracked > UINT64_MAX - 2) {
        return VAE_C_OP_OVERFLOW;
    }

    batch_size = input_a->shape.dims[0];
    pair_invalidate_context(workspace);
    workspace->main_result.valid = false;
    workspace->over_result.valid = false;
    workspace->total_loss = 0.0f;

    status = vae_c_iic_head_forward(
        &pair->main_head, VAE_C_BATCH_NORM_MODE_TRAIN,
        input_a, &workspace->main_a
    );
    if (status == VAE_C_OP_OK) {
        status = vae_c_iic_head_forward(
            &pair->overcluster_head, VAE_C_BATCH_NORM_MODE_TRAIN,
            input_a, &workspace->over_a
        );
    }
    if (status == VAE_C_OP_OK) {
        status = vae_c_iic_head_forward(
            &pair->main_head, VAE_C_BATCH_NORM_MODE_TRAIN,
            input_b, &workspace->main_b
        );
    }
    if (status == VAE_C_OP_OK) {
        status = vae_c_iic_head_forward(
            &pair->overcluster_head, VAE_C_BATCH_NORM_MODE_TRAIN,
            input_b, &workspace->over_b
        );
    }
    if (status != VAE_C_OP_OK) {
        return status;
    }

    loss_status = vae_c_iic_loss_forward(
        workspace->main_a.softmax_output.data, batch_size,
        pair->main_output_dim,
        workspace->main_b.softmax_output.data, batch_size,
        pair->main_output_dim, &workspace->main_result
    );
    if (loss_status != VAE_C_IIC_LOSS_OK) {
        return pair_loss_status(loss_status);
    }
    loss_status = vae_c_iic_loss_forward(
        workspace->over_a.softmax_output.data, batch_size,
        pair->overcluster_output_dim,
        workspace->over_b.softmax_output.data, batch_size,
        pair->overcluster_output_dim, &workspace->over_result
    );
    if (loss_status != VAE_C_IIC_LOSS_OK) {
        return pair_loss_status(loss_status);
    }
    total_loss = workspace->main_result.loss + workspace->over_result.loss;
    if (!isfinite(total_loss)) {
        return VAE_C_OP_INVALID_STATE;
    }

    workspace->batch_size = batch_size;
    workspace->total_loss = total_loss;
    workspace->backward_pair = pair;
    workspace->backward_input_a = input_a;
    workspace->backward_input_b = input_b;
    workspace->main_a_generation = workspace->main_a.backward_generation;
    workspace->main_b_generation = workspace->main_b.backward_generation;
    workspace->over_a_generation = workspace->over_a.backward_generation;
    workspace->over_b_generation = workspace->over_b.backward_generation;
    return VAE_C_OP_OK;
}


static bool pair_ticket_matches(
    const VAECIICHead *head,
    const VAECIICHeadWorkspace *head_workspace,
    const VAECTensor *input,
    uint64_t generation,
    size_t batch_size
)
{
    return generation != 0 && generation <= head->forward_generation &&
        head_workspace->backward_head == head &&
        head_workspace->backward_input == input &&
        head_workspace->backward_generation == generation &&
        head_workspace->batch_norm_cache.valid &&
        head_workspace->batch_size == batch_size &&
        head_workspace->softmax_output.shape.dims[0] == batch_size;
}


static bool pair_result_matches(
    const VAECIICLossResult *result,
    const VAECIICHeadWorkspace *view_a,
    const VAECIICHeadWorkspace *view_b,
    size_t batch_size,
    size_t class_count
)
{
    return result->valid && result->batch_size == batch_size &&
        result->class_count == class_count &&
        result->probabilities_a_data == view_a->softmax_output.data &&
        result->probabilities_b_data == view_b->softmax_output.data &&
        isfinite(result->mutual_information) &&
        isfinite(result->loss) &&
        result->loss == -result->mutual_information;
}


static VAECOpStatus pair_backward_preflight(
    const VAECIICPair *pair,
    const VAECTensor *input_a,
    const VAECTensor *input_b,
    const VAECIICPairWorkspace *workspace,
    VAECIICPairBackwardOrder order
)
{
    VAECOpStatus status;
    size_t batch_size;

    if (order != VAE_C_IIC_PAIR_BACKWARD_A_THEN_B &&
        order != VAE_C_IIC_PAIR_BACKWARD_B_THEN_A) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    status = pair_input_preflight(pair, workspace, input_a, input_b);
    if (status != VAE_C_OP_OK) {
        return status;
    }
    batch_size = input_a->shape.dims[0];
    if (workspace->backward_pair != pair ||
        workspace->backward_input_a != input_a ||
        workspace->backward_input_b != input_b ||
        workspace->batch_size != batch_size ||
        !pair_ticket_matches(
            &pair->main_head, &workspace->main_a, input_a,
            workspace->main_a_generation, batch_size) ||
        !pair_ticket_matches(
            &pair->main_head, &workspace->main_b, input_b,
            workspace->main_b_generation, batch_size) ||
        !pair_ticket_matches(
            &pair->overcluster_head, &workspace->over_a, input_a,
            workspace->over_a_generation, batch_size) ||
        !pair_ticket_matches(
            &pair->overcluster_head, &workspace->over_b, input_b,
            workspace->over_b_generation, batch_size) ||
        !pair_result_matches(
            &workspace->main_result, &workspace->main_a,
            &workspace->main_b, batch_size, pair->main_output_dim) ||
        !pair_result_matches(
            &workspace->over_result, &workspace->over_a,
            &workspace->over_b, batch_size,
            pair->overcluster_output_dim) ||
        !isfinite(workspace->total_loss) ||
        workspace->total_loss !=
            workspace->main_result.loss + workspace->over_result.loss) {
        return VAE_C_OP_INVALID_STATE;
    }
    return VAE_C_OP_OK;
}


VAECOpStatus vae_c_iic_pair_backward(
    VAECIICPair *pair,
    const VAECTensor *input_a,
    const VAECTensor *input_b,
    VAECIICPairWorkspace *workspace,
    VAECIICPairBackwardOrder order
)
{
    VAECOpStatus status;
    VAECIICLossStatus loss_status;
    size_t batch_size;

    if (pair == NULL || workspace == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (vae_c_iic_pair_workspace_validate(pair, workspace) !=
        VAE_C_OP_OK) {
        return VAE_C_OP_INVALID_STATE;
    }
    status = pair_backward_preflight(
        pair, input_a, input_b, workspace, order
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    batch_size = workspace->batch_size;
    /* C5B1 validates its complete forward result, including every joint
     * and marginal value, before writing these private gradient buffers.
     * Doing both calls here keeps model gradients unchanged if either result
     * has become invalid. */
    loss_status = vae_c_iic_loss_backward(
        workspace->main_a.softmax_output.data, batch_size,
        pair->main_output_dim,
        workspace->main_b.softmax_output.data, batch_size,
        pair->main_output_dim, &workspace->main_result,
        workspace->main_a_loss_grad,
        workspace->capacity * pair->main_output_dim,
        workspace->main_b_loss_grad,
        workspace->capacity * pair->main_output_dim
    );
    if (loss_status != VAE_C_IIC_LOSS_OK) {
        return pair_loss_status(loss_status);
    }
    loss_status = vae_c_iic_loss_backward(
        workspace->over_a.softmax_output.data, batch_size,
        pair->overcluster_output_dim,
        workspace->over_b.softmax_output.data, batch_size,
        pair->overcluster_output_dim, &workspace->over_result,
        workspace->over_a_loss_grad,
        workspace->capacity * pair->overcluster_output_dim,
        workspace->over_b_loss_grad,
        workspace->capacity * pair->overcluster_output_dim
    );
    if (loss_status != VAE_C_IIC_LOSS_OK) {
        return pair_loss_status(loss_status);
    }
    pair_invalidate_context(workspace);
    status = vae_c_iic_pair_workspace_zero_grad(pair, workspace);
    if (status != VAE_C_OP_OK) {
        return status;
    }
    memcpy(
        workspace->main_a.softmax_output.grad,
        workspace->main_a_loss_grad,
        batch_size * pair->main_output_dim * sizeof(float)
    );
    memcpy(
        workspace->main_b.softmax_output.grad,
        workspace->main_b_loss_grad,
        batch_size * pair->main_output_dim * sizeof(float)
    );
    memcpy(
        workspace->over_a.softmax_output.grad,
        workspace->over_a_loss_grad,
        batch_size * pair->overcluster_output_dim * sizeof(float)
    );
    memcpy(
        workspace->over_b.softmax_output.grad,
        workspace->over_b_loss_grad,
        batch_size * pair->overcluster_output_dim * sizeof(float)
    );

    if (order == VAE_C_IIC_PAIR_BACKWARD_A_THEN_B) {
        status = vae_c_iic_head_backward(
            &pair->main_head, input_a, &workspace->main_a
        );
        if (status == VAE_C_OP_OK) {
            status = vae_c_iic_head_backward(
                &pair->main_head, input_b, &workspace->main_b
            );
        }
        if (status == VAE_C_OP_OK) {
            status = vae_c_iic_head_backward(
                &pair->overcluster_head, input_a, &workspace->over_a
            );
        }
        if (status == VAE_C_OP_OK) {
            status = vae_c_iic_head_backward(
                &pair->overcluster_head, input_b, &workspace->over_b
            );
        }
    } else {
        status = vae_c_iic_head_backward(
            &pair->main_head, input_b, &workspace->main_b
        );
        if (status == VAE_C_OP_OK) {
            status = vae_c_iic_head_backward(
                &pair->main_head, input_a, &workspace->main_a
            );
        }
        if (status == VAE_C_OP_OK) {
            status = vae_c_iic_head_backward(
                &pair->overcluster_head, input_b, &workspace->over_b
            );
        }
        if (status == VAE_C_OP_OK) {
            status = vae_c_iic_head_backward(
                &pair->overcluster_head, input_a, &workspace->over_a
            );
        }
    }
    return status;
}
