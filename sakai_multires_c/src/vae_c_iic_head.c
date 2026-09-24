#include "vae_c_iic_head.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>


typedef struct {
    VAECTensor linear1_output;
    VAECTensor batch_norm_output;
    VAECTensor relu_output;
    VAECTensor logits;
    VAECTensor softmax_output;
} VAECIICHeadActiveViews;


static int iic_head_size_multiply(
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


static VAECOpStatus iic_head_map_storage_status(
    VAECStorageStatus status
)
{
    switch (status) {
        case VAE_C_STORAGE_OK:
            return VAE_C_OP_OK;
        case VAE_C_STORAGE_INVALID_ARGUMENT:
            return VAE_C_OP_INVALID_ARGUMENT;
        case VAE_C_STORAGE_INVALID_RANK:
        case VAE_C_STORAGE_INVALID_DIMENSION:
        case VAE_C_STORAGE_SHAPE_MISMATCH:
            return VAE_C_OP_SHAPE_MISMATCH;
        case VAE_C_STORAGE_OVERFLOW:
            return VAE_C_OP_OVERFLOW;
        case VAE_C_STORAGE_ALLOCATION_FAILED:
            return VAE_C_OP_ALLOCATION_FAILED;
        case VAE_C_STORAGE_OBJECT_NOT_EMPTY:
            return VAE_C_OP_OBJECT_NOT_EMPTY;
        case VAE_C_STORAGE_GRADIENT_NOT_AVAILABLE:
            return VAE_C_OP_GRADIENT_NOT_AVAILABLE;
        case VAE_C_STORAGE_INVALID_STATE:
        case VAE_C_STORAGE_OUT_OF_RANGE:
        default:
            return VAE_C_OP_INVALID_STATE;
    }
}


static bool iic_head_shape_is_zero(const VAECShape *shape)
{
    size_t index;

    if (shape == NULL || shape->rank != 0 || shape->count != 0) {
        return false;
    }
    for (index = 0; index < VAE_C_TENSOR_MAX_RANK; index++) {
        if (shape->dims[index] != 0 || shape->strides[index] != 0) {
            return false;
        }
    }
    return true;
}


static VAECOpStatus iic_head_validate_shape(const VAECShape *shape)
{
    VAECShape canonical;
    VAECStorageStatus status;

    if (shape == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    status = vae_c_shape_make(shape->rank, shape->dims, &canonical);
    if (status != VAE_C_STORAGE_OK) {
        return iic_head_map_storage_status(status);
    }
    return
        memcmp(shape, &canonical, sizeof(canonical)) == 0
            ? VAE_C_OP_OK
            : VAE_C_OP_INVALID_STATE;
}


static bool iic_head_all_finite(const float *values, size_t count)
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


static bool iic_head_all_finite_zero(const float *values, size_t count)
{
    size_t index;

    if (values == NULL) {
        return false;
    }
    for (index = 0; index < count; index++) {
        if (!isfinite(values[index]) || values[index] != 0.0f) {
            return false;
        }
    }
    return true;
}


static VAECOpStatus iic_head_validate_parameter(
    const VAECParameter *parameter,
    uint64_t expected_key
)
{
    size_t total_elements;
    size_t expected_bytes;
    size_t index;

    if (parameter == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (iic_head_validate_shape(&parameter->shape) != VAE_C_OP_OK ||
        parameter->shape.count == 0 ||
        parameter->parameter_key != expected_key ||
        expected_key == UINT64_C(0) ||
        parameter->owned_block == NULL ||
        parameter->values != parameter->owned_block ||
        parameter->gradients !=
            parameter->values + parameter->shape.count ||
        parameter->adam_m !=
            parameter->gradients + parameter->shape.count ||
        parameter->adam_v !=
            parameter->adam_m + parameter->shape.count ||
        iic_head_size_multiply(
            parameter->shape.count,
            4,
            &total_elements) != 0 ||
        iic_head_size_multiply(
            total_elements,
            sizeof(float),
            &expected_bytes) != 0 ||
        parameter->owned_bytes != expected_bytes ||
        !iic_head_all_finite(
            parameter->owned_block,
            total_elements)) {

        return VAE_C_OP_INVALID_STATE;
    }
    for (index = 0; index < parameter->shape.count; index++) {
        if (parameter->adam_v[index] < 0.0f) {
            return VAE_C_OP_INVALID_STATE;
        }
    }
    return VAE_C_OP_OK;
}


static bool iic_head_parameters_alias(
    const VAECParameter *left,
    const VAECParameter *right
)
{
    const void *left_regions[5];
    const void *right_regions[5];
    size_t left_index;
    size_t right_index;

    if (left == NULL || right == NULL || left == right) {
        return true;
    }
    left_regions[0] = left->values;
    left_regions[1] = left->gradients;
    left_regions[2] = left->adam_m;
    left_regions[3] = left->adam_v;
    left_regions[4] = left->owned_block;
    right_regions[0] = right->values;
    right_regions[1] = right->gradients;
    right_regions[2] = right->adam_m;
    right_regions[3] = right->adam_v;
    right_regions[4] = right->owned_block;

    for (left_index = 0; left_index < 5; left_index++) {
        for (right_index = 0; right_index < 5; right_index++) {
            if (left_regions[left_index] == right_regions[right_index]) {
                return true;
            }
        }
    }
    return false;
}


static VAECOpStatus iic_head_validate_dense(
    const VAECDense *layer,
    size_t input_dim,
    size_t output_dim,
    bool with_bias,
    uint64_t weight_key,
    uint64_t bias_key
)
{
    if (layer == NULL ||
        layer->in_features != input_dim ||
        layer->out_features != output_dim ||
        layer->with_bias != with_bias ||
        iic_head_validate_parameter(
            &layer->weights,
            weight_key) != VAE_C_OP_OK ||
        layer->weights.shape.rank != 2 ||
        layer->weights.shape.dims[0] != output_dim ||
        layer->weights.shape.dims[1] != input_dim) {

        return VAE_C_OP_INVALID_STATE;
    }
    if (!with_bias) {
        return
            vae_c_parameter_is_empty(&layer->bias)
                ? VAE_C_OP_OK
                : VAE_C_OP_INVALID_STATE;
    }
    if (iic_head_validate_parameter(&layer->bias, bias_key) !=
            VAE_C_OP_OK ||
        layer->bias.shape.rank != 1 ||
        layer->bias.shape.dims[0] != output_dim ||
        iic_head_parameters_alias(&layer->weights, &layer->bias)) {

        return VAE_C_OP_INVALID_STATE;
    }
    return VAE_C_OP_OK;
}


static bool iic_head_batch_norm_buffers_unique(
    const VAECBatchNorm *layer
)
{
    const void *buffers[15];
    size_t left;
    size_t right;

    buffers[0] = layer->gamma.values;
    buffers[1] = layer->gamma.gradients;
    buffers[2] = layer->gamma.adam_m;
    buffers[3] = layer->gamma.adam_v;
    buffers[4] = layer->beta.values;
    buffers[5] = layer->beta.gradients;
    buffers[6] = layer->beta.adam_m;
    buffers[7] = layer->beta.adam_v;
    buffers[8] = layer->running_mean;
    buffers[9] = layer->running_variance;
    buffers[10] = layer->batch_mean;
    buffers[11] = layer->batch_inv_std;
    buffers[12] = layer->work_mean;
    buffers[13] = layer->work_inv_std;
    buffers[14] = layer->work_unbiased_variance;

    for (left = 0; left < 15; left++) {
        if (buffers[left] == NULL) {
            return false;
        }
        for (right = left + 1; right < 15; right++) {
            if (buffers[left] == buffers[right]) {
                return false;
            }
        }
    }
    return true;
}


static VAECOpStatus iic_head_validate_batch_norm_cache(
    const VAECBatchNorm *layer
)
{
    size_t feature;

    if (!layer->cache_valid) {
        return
            layer->last_sample_count == 0 &&
            iic_head_shape_is_zero(&layer->last_input_shape) &&
            iic_head_shape_is_zero(&layer->last_output_shape)
                ? VAE_C_OP_OK
                : VAE_C_OP_INVALID_STATE;
    }
    if (layer->last_sample_count < 2 ||
        iic_head_validate_shape(&layer->last_input_shape) !=
            VAE_C_OP_OK ||
        iic_head_validate_shape(&layer->last_output_shape) !=
            VAE_C_OP_OK ||
        !vae_c_shape_equal(
            &layer->last_input_shape,
            &layer->last_output_shape) ||
        layer->last_input_shape.rank != 2 ||
        layer->last_input_shape.dims[0] != layer->last_sample_count ||
        layer->last_input_shape.dims[1] != layer->num_features) {

        return VAE_C_OP_INVALID_STATE;
    }
    for (feature = 0; feature < layer->num_features; feature++) {
        if (!isfinite(layer->batch_mean[feature]) ||
            !isfinite(layer->batch_inv_std[feature]) ||
            layer->batch_inv_std[feature] <= 0.0f) {

            return VAE_C_OP_INVALID_STATE;
        }
    }
    return VAE_C_OP_OK;
}


static void iic_head_capture_batch_norm_cache(
    const VAECBatchNorm *layer,
    VAECIICHeadBatchNormCache *cache
)
{
    memcpy(
        cache->batch_mean,
        layer->batch_mean,
        sizeof(cache->batch_mean)
    );
    memcpy(
        cache->batch_inv_std,
        layer->batch_inv_std,
        sizeof(cache->batch_inv_std)
    );
    cache->last_sample_count = layer->last_sample_count;
    cache->last_input_shape = layer->last_input_shape;
    cache->last_output_shape = layer->last_output_shape;
    cache->valid = layer->cache_valid;
}


static void iic_head_restore_batch_norm_cache(
    VAECBatchNorm *layer,
    const VAECIICHeadBatchNormCache *cache
)
{
    memcpy(
        layer->batch_mean,
        cache->batch_mean,
        sizeof(cache->batch_mean)
    );
    memcpy(
        layer->batch_inv_std,
        cache->batch_inv_std,
        sizeof(cache->batch_inv_std)
    );
    layer->last_sample_count = cache->last_sample_count;
    layer->last_input_shape = cache->last_input_shape;
    layer->last_output_shape = cache->last_output_shape;
    layer->cache_valid = cache->valid;
}


static VAECOpStatus iic_head_validate_batch_norm(
    const VAECBatchNorm *layer
)
{
    size_t total_elements;
    size_t expected_bytes;
    size_t feature;

    if (layer == NULL ||
        layer->kind != VAE_C_BATCH_NORM_KIND_1D ||
        layer->num_features != VAE_C_IIC_HEAD_HIDDEN_DIM ||
        layer->epsilon != VAE_C_BATCH_NORM_EPSILON ||
        layer->momentum != VAE_C_BATCH_NORM_MOMENTUM ||
        iic_head_validate_parameter(
            &layer->gamma,
            VAE_C_IIC_HEAD_PARAM_KEY_BATCH_NORM_GAMMA) !=
                VAE_C_OP_OK ||
        iic_head_validate_parameter(
            &layer->beta,
            VAE_C_IIC_HEAD_PARAM_KEY_BATCH_NORM_BETA) !=
                VAE_C_OP_OK ||
        layer->gamma.shape.rank != 1 ||
        layer->gamma.shape.dims[0] != VAE_C_IIC_HEAD_HIDDEN_DIM ||
        layer->beta.shape.rank != 1 ||
        layer->beta.shape.dims[0] != VAE_C_IIC_HEAD_HIDDEN_DIM ||
        layer->owned_block == NULL ||
        iic_head_size_multiply(
            VAE_C_IIC_HEAD_HIDDEN_DIM,
            7,
            &total_elements) != 0 ||
        iic_head_size_multiply(
            total_elements,
            sizeof(float),
            &expected_bytes) != 0 ||
        layer->owned_bytes != expected_bytes ||
        layer->running_mean != layer->owned_block ||
        layer->running_variance !=
            layer->running_mean + VAE_C_IIC_HEAD_HIDDEN_DIM ||
        layer->batch_mean !=
            layer->running_variance + VAE_C_IIC_HEAD_HIDDEN_DIM ||
        layer->batch_inv_std !=
            layer->batch_mean + VAE_C_IIC_HEAD_HIDDEN_DIM ||
        layer->work_mean !=
            layer->batch_inv_std + VAE_C_IIC_HEAD_HIDDEN_DIM ||
        layer->work_inv_std !=
            layer->work_mean + VAE_C_IIC_HEAD_HIDDEN_DIM ||
        layer->work_unbiased_variance !=
            layer->work_inv_std + VAE_C_IIC_HEAD_HIDDEN_DIM ||
        !iic_head_batch_norm_buffers_unique(layer) ||
        !iic_head_all_finite(layer->owned_block, total_elements)) {

        return VAE_C_OP_INVALID_STATE;
    }
    for (feature = 0;
         feature < VAE_C_IIC_HEAD_HIDDEN_DIM;
         feature++) {

        if (layer->running_variance[feature] < 0.0f) {
            return VAE_C_OP_INVALID_STATE;
        }
    }
    return iic_head_validate_batch_norm_cache(layer);
}


static bool iic_head_parameter_blocks_unique(const VAECIICHead *head)
{
    const VAECParameter *parameters[5];
    const void *batch_norm_buffers[7];
    size_t left;
    size_t right;
    size_t buffer_index;

    parameters[0] = &head->linear1.weights;
    parameters[1] = &head->batch_norm.gamma;
    parameters[2] = &head->batch_norm.beta;
    parameters[3] = &head->linear2.weights;
    parameters[4] = &head->linear2.bias;
    batch_norm_buffers[0] = head->batch_norm.running_mean;
    batch_norm_buffers[1] = head->batch_norm.running_variance;
    batch_norm_buffers[2] = head->batch_norm.batch_mean;
    batch_norm_buffers[3] = head->batch_norm.batch_inv_std;
    batch_norm_buffers[4] = head->batch_norm.work_mean;
    batch_norm_buffers[5] = head->batch_norm.work_inv_std;
    batch_norm_buffers[6] = head->batch_norm.work_unbiased_variance;

    for (left = 0; left < 5; left++) {
        for (right = left + 1; right < 5; right++) {
            if (iic_head_parameters_alias(
                    parameters[left],
                    parameters[right])) {

                return false;
            }
        }
        for (buffer_index = 0; buffer_index < 7; buffer_index++) {
            if (parameters[left]->values ==
                    batch_norm_buffers[buffer_index] ||
                parameters[left]->gradients ==
                    batch_norm_buffers[buffer_index] ||
                parameters[left]->adam_m ==
                    batch_norm_buffers[buffer_index] ||
                parameters[left]->adam_v ==
                    batch_norm_buffers[buffer_index] ||
                parameters[left]->owned_block ==
                    batch_norm_buffers[buffer_index]) {

                return false;
            }
        }
    }
    return true;
}


void vae_c_iic_head_init(VAECIICHead *head)
{
    if (head != NULL) {
        memset(head, 0, sizeof(*head));
    }
}


void vae_c_iic_head_free(VAECIICHead *head)
{
    if (head == NULL) {
        return;
    }
    vae_c_dense_free(&head->linear1);
    vae_c_batch_norm_free(&head->batch_norm);
    vae_c_dense_free(&head->linear2);
    vae_c_iic_head_init(head);
}


bool vae_c_iic_head_is_empty(const VAECIICHead *head)
{
    return
        head != NULL &&
        head->input_dim == 0 &&
        head->output_dim == 0 &&
        vae_c_dense_is_empty(&head->linear1) &&
        vae_c_batch_norm_is_empty(&head->batch_norm) &&
        vae_c_dense_is_empty(&head->linear2) &&
        head->num_batches_tracked == UINT64_C(0) &&
        head->forward_generation == UINT64_C(0) &&
        !head->allocated;
}


VAECOpStatus vae_c_iic_head_validate(const VAECIICHead *head)
{
    if (head == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (!head->allocated ||
        head->input_dim == 0 ||
        head->output_dim == 0 ||
        iic_head_validate_dense(
            &head->linear1,
            head->input_dim,
            VAE_C_IIC_HEAD_HIDDEN_DIM,
            false,
            VAE_C_IIC_HEAD_PARAM_KEY_LINEAR1_WEIGHT,
            UINT64_C(0)) != VAE_C_OP_OK ||
        iic_head_validate_batch_norm(&head->batch_norm) !=
            VAE_C_OP_OK ||
        iic_head_validate_dense(
            &head->linear2,
            VAE_C_IIC_HEAD_HIDDEN_DIM,
            head->output_dim,
            true,
            VAE_C_IIC_HEAD_PARAM_KEY_LINEAR2_WEIGHT,
            VAE_C_IIC_HEAD_PARAM_KEY_LINEAR2_BIAS) != VAE_C_OP_OK ||
        !iic_head_parameter_blocks_unique(head)) {

        return VAE_C_OP_INVALID_STATE;
    }
    return VAE_C_OP_OK;
}


bool vae_c_iic_head_is_allocated(const VAECIICHead *head)
{
    return vae_c_iic_head_validate(head) == VAE_C_OP_OK;
}


VAECOpStatus vae_c_iic_head_allocate(
    VAECIICHead *head,
    size_t input_dim,
    size_t output_dim
)
{
    VAECIICHead candidate;
    VAECOpStatus status;

    if (head == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (!vae_c_iic_head_is_empty(head)) {
        return VAE_C_OP_OBJECT_NOT_EMPTY;
    }
    if (input_dim == 0 || output_dim == 0) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    vae_c_iic_head_init(&candidate);
    status = vae_c_dense_allocate(
        &candidate.linear1,
        input_dim,
        VAE_C_IIC_HEAD_HIDDEN_DIM,
        false,
        VAE_C_IIC_HEAD_PARAM_KEY_LINEAR1_WEIGHT,
        UINT64_C(0)
    );
    if (status != VAE_C_OP_OK) {
        vae_c_iic_head_free(&candidate);
        return status;
    }
    status = vae_c_batch_norm1d_allocate(
        &candidate.batch_norm,
        VAE_C_IIC_HEAD_HIDDEN_DIM,
        VAE_C_IIC_HEAD_PARAM_KEY_BATCH_NORM_GAMMA,
        VAE_C_IIC_HEAD_PARAM_KEY_BATCH_NORM_BETA
    );
    if (status != VAE_C_OP_OK) {
        vae_c_iic_head_free(&candidate);
        return status;
    }
    status = vae_c_dense_allocate(
        &candidate.linear2,
        VAE_C_IIC_HEAD_HIDDEN_DIM,
        output_dim,
        true,
        VAE_C_IIC_HEAD_PARAM_KEY_LINEAR2_WEIGHT,
        VAE_C_IIC_HEAD_PARAM_KEY_LINEAR2_BIAS
    );
    if (status != VAE_C_OP_OK) {
        vae_c_iic_head_free(&candidate);
        return status;
    }

    candidate.input_dim = input_dim;
    candidate.output_dim = output_dim;
    candidate.allocated = true;
    if (vae_c_iic_head_validate(&candidate) != VAE_C_OP_OK) {
        vae_c_iic_head_free(&candidate);
        return VAE_C_OP_INVALID_STATE;
    }
    *head = candidate;
    return VAE_C_OP_OK;
}


VAECOpStatus vae_c_iic_head_zero_grad(VAECIICHead *head)
{
    VAECOpStatus status;

    if (head == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (vae_c_iic_head_is_empty(head)) {
        return VAE_C_OP_OK;
    }
    if (vae_c_iic_head_validate(head) != VAE_C_OP_OK) {
        return VAE_C_OP_INVALID_STATE;
    }
    status = vae_c_dense_zero_grad(&head->linear1);
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_batch_norm_zero_grad(&head->batch_norm);
    if (status != VAE_C_OP_OK) {
        return status;
    }
    return vae_c_dense_zero_grad(&head->linear2);
}


static VAECOpStatus iic_head_validate_tensor(
    const VAECTensor *tensor
)
{
    size_t arrays;
    size_t total_elements;
    size_t expected_bytes;

    if (tensor == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (iic_head_validate_shape(&tensor->shape) != VAE_C_OP_OK ||
        tensor->data == NULL ||
        (tensor->has_grad &&
         (tensor->grad == NULL || tensor->grad == tensor->data)) ||
        (!tensor->has_grad && tensor->grad != NULL)) {

        return VAE_C_OP_INVALID_STATE;
    }
    if (!tensor->owns_memory) {
        return
            tensor->owned_block == NULL && tensor->owned_bytes == 0
                ? VAE_C_OP_OK
                : VAE_C_OP_INVALID_STATE;
    }
    arrays = tensor->has_grad ? 2 : 1;
    if (tensor->owned_block == NULL ||
        tensor->data != tensor->owned_block ||
        iic_head_size_multiply(
            tensor->shape.count,
            arrays,
            &total_elements) != 0 ||
        iic_head_size_multiply(
            total_elements,
            sizeof(float),
            &expected_bytes) != 0 ||
        tensor->owned_bytes != expected_bytes ||
        (tensor->has_grad &&
         tensor->grad != tensor->data + tensor->shape.count)) {

        return VAE_C_OP_INVALID_STATE;
    }
    return VAE_C_OP_OK;
}


static bool iic_head_owned_tensor_matches(
    const VAECTensor *tensor,
    size_t rows,
    size_t columns
)
{
    VAECShape expected;
    size_t dims[2];
    size_t total_elements;
    size_t expected_bytes;

    dims[0] = rows;
    dims[1] = columns;
    if (tensor == NULL ||
        vae_c_shape_make(2, dims, &expected) != VAE_C_STORAGE_OK ||
        iic_head_validate_tensor(tensor) != VAE_C_OP_OK ||
        !vae_c_shape_equal(&tensor->shape, &expected) ||
        !tensor->owns_memory || !tensor->has_grad ||
        iic_head_size_multiply(
            expected.count,
            2,
            &total_elements) != 0 ||
        iic_head_size_multiply(
            total_elements,
            sizeof(float),
            &expected_bytes) != 0) {

        return false;
    }
    return tensor->owned_bytes == expected_bytes;
}


static bool iic_head_active_view_matches(
    const VAECTensor *view,
    const VAECTensor *storage,
    size_t rows,
    size_t columns
)
{
    VAECShape expected;
    size_t dims[2];

    dims[0] = rows;
    dims[1] = columns;
    return
        view != NULL && storage != NULL &&
        vae_c_shape_make(2, dims, &expected) == VAE_C_STORAGE_OK &&
        iic_head_validate_tensor(view) == VAE_C_OP_OK &&
        vae_c_shape_equal(&view->shape, &expected) &&
        !view->owns_memory && view->has_grad &&
        view->data == storage->data &&
        view->grad == storage->grad;
}


static bool iic_head_workspace_storage_unique(
    const VAECIICHeadWorkspace *workspace
)
{
    const VAECTensor *tensors[5];
    const void *buffers[10];
    size_t tensor_index;
    size_t left;
    size_t right;

    tensors[0] = &workspace->linear1_storage;
    tensors[1] = &workspace->batch_norm_storage;
    tensors[2] = &workspace->relu_storage;
    tensors[3] = &workspace->logits_storage;
    tensors[4] = &workspace->softmax_storage;
    for (tensor_index = 0; tensor_index < 5; tensor_index++) {
        buffers[2 * tensor_index] = tensors[tensor_index]->data;
        buffers[2 * tensor_index + 1] = tensors[tensor_index]->grad;
    }
    for (left = 0; left < 10; left++) {
        if (buffers[left] == NULL) {
            return false;
        }
        for (right = left + 1; right < 10; right++) {
            if (buffers[left] == buffers[right]) {
                return false;
            }
        }
    }
    return true;
}


void vae_c_iic_head_workspace_init(VAECIICHeadWorkspace *workspace)
{
    if (workspace != NULL) {
        memset(workspace, 0, sizeof(*workspace));
    }
}


static void iic_head_workspace_clear_views(
    VAECIICHeadWorkspace *workspace
)
{
    vae_c_tensor_init(&workspace->linear1_output);
    vae_c_tensor_init(&workspace->batch_norm_output);
    vae_c_tensor_init(&workspace->relu_output);
    vae_c_tensor_init(&workspace->logits);
    vae_c_tensor_init(&workspace->softmax_output);
}


void vae_c_iic_head_workspace_free(VAECIICHeadWorkspace *workspace)
{
    if (workspace == NULL) {
        return;
    }
    iic_head_workspace_clear_views(workspace);
    vae_c_tensor_free(&workspace->linear1_storage);
    vae_c_tensor_free(&workspace->batch_norm_storage);
    vae_c_tensor_free(&workspace->relu_storage);
    vae_c_tensor_free(&workspace->logits_storage);
    vae_c_tensor_free(&workspace->softmax_storage);
    vae_c_iic_head_workspace_init(workspace);
}


bool vae_c_iic_head_workspace_is_empty(
    const VAECIICHeadWorkspace *workspace
)
{
    return
        workspace != NULL &&
        workspace->capacity == 0 &&
        workspace->input_dim == 0 &&
        workspace->output_dim == 0 &&
        workspace->batch_size == 0 &&
        vae_c_tensor_is_empty(&workspace->linear1_storage) &&
        vae_c_tensor_is_empty(&workspace->batch_norm_storage) &&
        vae_c_tensor_is_empty(&workspace->relu_storage) &&
        vae_c_tensor_is_empty(&workspace->logits_storage) &&
        vae_c_tensor_is_empty(&workspace->softmax_storage) &&
        vae_c_tensor_is_empty(&workspace->linear1_output) &&
        vae_c_tensor_is_empty(&workspace->batch_norm_output) &&
        vae_c_tensor_is_empty(&workspace->relu_output) &&
        vae_c_tensor_is_empty(&workspace->logits) &&
        vae_c_tensor_is_empty(&workspace->softmax_output) &&
        workspace->backward_head == NULL &&
        workspace->backward_input == NULL &&
        workspace->backward_generation == UINT64_C(0) &&
        !workspace->batch_norm_cache.valid &&
        !workspace->allocated;
}


VAECOpStatus vae_c_iic_head_workspace_validate(
    const VAECIICHead *head,
    const VAECIICHeadWorkspace *workspace
)
{
    if (head == NULL || workspace == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (vae_c_iic_head_validate(head) != VAE_C_OP_OK ||
        !workspace->allocated ||
        workspace->capacity == 0 ||
        workspace->input_dim != head->input_dim ||
        workspace->output_dim != head->output_dim ||
        !iic_head_owned_tensor_matches(
            &workspace->linear1_storage,
            workspace->capacity,
            VAE_C_IIC_HEAD_HIDDEN_DIM) ||
        !iic_head_owned_tensor_matches(
            &workspace->batch_norm_storage,
            workspace->capacity,
            VAE_C_IIC_HEAD_HIDDEN_DIM) ||
        !iic_head_owned_tensor_matches(
            &workspace->relu_storage,
            workspace->capacity,
            VAE_C_IIC_HEAD_HIDDEN_DIM) ||
        !iic_head_owned_tensor_matches(
            &workspace->logits_storage,
            workspace->capacity,
            workspace->output_dim) ||
        !iic_head_owned_tensor_matches(
            &workspace->softmax_storage,
            workspace->capacity,
            workspace->output_dim) ||
        !iic_head_workspace_storage_unique(workspace) ||
        ((workspace->backward_head == NULL) !=
         (workspace->backward_input == NULL)) ||
        ((workspace->backward_head == NULL) !=
         (workspace->backward_generation == UINT64_C(0))) ||
        ((workspace->backward_head == NULL) !=
         !workspace->batch_norm_cache.valid)) {

        return VAE_C_OP_INVALID_STATE;
    }

    if (workspace->batch_size == 0) {
        return
            vae_c_tensor_is_empty(&workspace->linear1_output) &&
            vae_c_tensor_is_empty(&workspace->batch_norm_output) &&
            vae_c_tensor_is_empty(&workspace->relu_output) &&
            vae_c_tensor_is_empty(&workspace->logits) &&
            vae_c_tensor_is_empty(&workspace->softmax_output) &&
            workspace->backward_head == NULL
                ? VAE_C_OP_OK
                : VAE_C_OP_INVALID_STATE;
    }
    if (workspace->batch_size > workspace->capacity ||
        !iic_head_active_view_matches(
            &workspace->linear1_output,
            &workspace->linear1_storage,
            workspace->batch_size,
            VAE_C_IIC_HEAD_HIDDEN_DIM) ||
        !iic_head_active_view_matches(
            &workspace->batch_norm_output,
            &workspace->batch_norm_storage,
            workspace->batch_size,
            VAE_C_IIC_HEAD_HIDDEN_DIM) ||
        !iic_head_active_view_matches(
            &workspace->relu_output,
            &workspace->relu_storage,
            workspace->batch_size,
            VAE_C_IIC_HEAD_HIDDEN_DIM) ||
        !iic_head_active_view_matches(
            &workspace->logits,
            &workspace->logits_storage,
            workspace->batch_size,
            workspace->output_dim) ||
        !iic_head_active_view_matches(
            &workspace->softmax_output,
            &workspace->softmax_storage,
            workspace->batch_size,
            workspace->output_dim)) {

        return VAE_C_OP_INVALID_STATE;
    }
    if (workspace->batch_norm_cache.valid) {
        size_t feature;

        if (workspace->batch_norm_cache.last_sample_count !=
                workspace->batch_size ||
            !vae_c_shape_equal(
                &workspace->batch_norm_cache.last_input_shape,
                &workspace->linear1_output.shape) ||
            !vae_c_shape_equal(
                &workspace->batch_norm_cache.last_output_shape,
                &workspace->batch_norm_output.shape) ||
            !iic_head_all_finite(
                workspace->batch_norm_cache.batch_mean,
                VAE_C_IIC_HEAD_HIDDEN_DIM)) {

            return VAE_C_OP_INVALID_STATE;
        }
        for (feature = 0; feature < VAE_C_IIC_HEAD_HIDDEN_DIM;
             feature++) {
            float inverse_std =
                workspace->batch_norm_cache.batch_inv_std[feature];

            if (!isfinite(inverse_std) || inverse_std <= 0.0f) {
                return VAE_C_OP_INVALID_STATE;
            }
        }
    }
    return VAE_C_OP_OK;
}


bool vae_c_iic_head_workspace_is_allocated(
    const VAECIICHead *head,
    const VAECIICHeadWorkspace *workspace
)
{
    return
        vae_c_iic_head_workspace_validate(head, workspace) ==
        VAE_C_OP_OK;
}


static VAECOpStatus iic_head_allocate_tensor(
    VAECTensor *tensor,
    size_t rows,
    size_t columns
)
{
    VAECShape shape;
    VAECStorageStatus status;
    const size_t dims[2] = {rows, columns};

    status = vae_c_shape_make(2, dims, &shape);
    if (status != VAE_C_STORAGE_OK) {
        return iic_head_map_storage_status(status);
    }
    return iic_head_map_storage_status(
        vae_c_tensor_allocate(tensor, &shape, true)
    );
}


VAECOpStatus vae_c_iic_head_workspace_allocate(
    const VAECIICHead *head,
    size_t capacity,
    VAECIICHeadWorkspace *workspace
)
{
    VAECIICHeadWorkspace candidate;
    VAECOpStatus status;

    if (head == NULL || workspace == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (!vae_c_iic_head_workspace_is_empty(workspace)) {
        return VAE_C_OP_OBJECT_NOT_EMPTY;
    }
    if (vae_c_iic_head_validate(head) != VAE_C_OP_OK) {
        return VAE_C_OP_INVALID_STATE;
    }
    if (capacity == 0) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    vae_c_iic_head_workspace_init(&candidate);
    status = iic_head_allocate_tensor(
        &candidate.linear1_storage,
        capacity,
        VAE_C_IIC_HEAD_HIDDEN_DIM
    );
    if (status != VAE_C_OP_OK) {
        vae_c_iic_head_workspace_free(&candidate);
        return status;
    }
    status = iic_head_allocate_tensor(
        &candidate.batch_norm_storage,
        capacity,
        VAE_C_IIC_HEAD_HIDDEN_DIM
    );
    if (status != VAE_C_OP_OK) {
        vae_c_iic_head_workspace_free(&candidate);
        return status;
    }
    status = iic_head_allocate_tensor(
        &candidate.relu_storage,
        capacity,
        VAE_C_IIC_HEAD_HIDDEN_DIM
    );
    if (status != VAE_C_OP_OK) {
        vae_c_iic_head_workspace_free(&candidate);
        return status;
    }
    status = iic_head_allocate_tensor(
        &candidate.logits_storage,
        capacity,
        head->output_dim
    );
    if (status != VAE_C_OP_OK) {
        vae_c_iic_head_workspace_free(&candidate);
        return status;
    }
    status = iic_head_allocate_tensor(
        &candidate.softmax_storage,
        capacity,
        head->output_dim
    );
    if (status != VAE_C_OP_OK) {
        vae_c_iic_head_workspace_free(&candidate);
        return status;
    }

    candidate.capacity = capacity;
    candidate.input_dim = head->input_dim;
    candidate.output_dim = head->output_dim;
    candidate.allocated = true;
    if (vae_c_iic_head_workspace_validate(head, &candidate) !=
            VAE_C_OP_OK) {

        vae_c_iic_head_workspace_free(&candidate);
        return VAE_C_OP_INVALID_STATE;
    }
    *workspace = candidate;
    return VAE_C_OP_OK;
}


VAECOpStatus vae_c_iic_head_workspace_zero_grad(
    const VAECIICHead *head,
    VAECIICHeadWorkspace *workspace
)
{
    VAECTensor *tensors[5];
    size_t index;

    if (head == NULL || workspace == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (vae_c_iic_head_workspace_validate(head, workspace) !=
            VAE_C_OP_OK) {

        return VAE_C_OP_INVALID_STATE;
    }
    tensors[0] = &workspace->linear1_storage;
    tensors[1] = &workspace->batch_norm_storage;
    tensors[2] = &workspace->relu_storage;
    tensors[3] = &workspace->logits_storage;
    tensors[4] = &workspace->softmax_storage;
    for (index = 0; index < 5; index++) {
        VAECStorageStatus status = vae_c_tensor_zero_grad(tensors[index]);

        if (status != VAE_C_STORAGE_OK) {
            return iic_head_map_storage_status(status);
        }
    }
    return VAE_C_OP_OK;
}


static bool iic_head_pointer_is_head_buffer(
    const VAECIICHead *head,
    const void *pointer
)
{
    const VAECParameter *parameters[5];
    const void *batch_norm_buffers[7];
    size_t parameter_index;
    size_t buffer_index;

    if (head == NULL || pointer == NULL) {
        return false;
    }
    parameters[0] = &head->linear1.weights;
    parameters[1] = &head->batch_norm.gamma;
    parameters[2] = &head->batch_norm.beta;
    parameters[3] = &head->linear2.weights;
    parameters[4] = &head->linear2.bias;
    for (parameter_index = 0;
         parameter_index < 5;
         parameter_index++) {

        const VAECParameter *parameter = parameters[parameter_index];

        if (pointer == parameter->values ||
            pointer == parameter->gradients ||
            pointer == parameter->adam_m ||
            pointer == parameter->adam_v ||
            pointer == parameter->owned_block) {

            return true;
        }
    }
    batch_norm_buffers[0] = head->batch_norm.running_mean;
    batch_norm_buffers[1] = head->batch_norm.running_variance;
    batch_norm_buffers[2] = head->batch_norm.batch_mean;
    batch_norm_buffers[3] = head->batch_norm.batch_inv_std;
    batch_norm_buffers[4] = head->batch_norm.work_mean;
    batch_norm_buffers[5] = head->batch_norm.work_inv_std;
    batch_norm_buffers[6] = head->batch_norm.work_unbiased_variance;
    for (buffer_index = 0; buffer_index < 7; buffer_index++) {
        if (pointer == batch_norm_buffers[buffer_index]) {
            return true;
        }
    }
    return pointer == head->batch_norm.owned_block;
}


static bool iic_head_pointer_is_workspace_buffer(
    const VAECIICHeadWorkspace *workspace,
    const void *pointer
)
{
    const VAECTensor *tensors[5];
    size_t index;

    if (workspace == NULL || pointer == NULL) {
        return false;
    }
    tensors[0] = &workspace->linear1_storage;
    tensors[1] = &workspace->batch_norm_storage;
    tensors[2] = &workspace->relu_storage;
    tensors[3] = &workspace->logits_storage;
    tensors[4] = &workspace->softmax_storage;
    for (index = 0; index < 5; index++) {
        if (pointer == tensors[index]->data ||
            pointer == tensors[index]->grad ||
            pointer == tensors[index]->owned_block) {

            return true;
        }
    }
    return false;
}


static bool iic_head_workspace_aliases_head(
    const VAECIICHead *head,
    const VAECIICHeadWorkspace *workspace
)
{
    const VAECTensor *tensors[5];
    size_t index;

    tensors[0] = &workspace->linear1_storage;
    tensors[1] = &workspace->batch_norm_storage;
    tensors[2] = &workspace->relu_storage;
    tensors[3] = &workspace->logits_storage;
    tensors[4] = &workspace->softmax_storage;
    for (index = 0; index < 5; index++) {
        if (iic_head_pointer_is_head_buffer(head, tensors[index]->data) ||
            iic_head_pointer_is_head_buffer(head, tensors[index]->grad) ||
            iic_head_pointer_is_head_buffer(
                head,
                tensors[index]->owned_block)) {

            return true;
        }
    }
    return false;
}


static VAECOpStatus iic_head_validate_external_input(
    const VAECIICHead *head,
    const VAECIICHeadWorkspace *workspace,
    VAECBatchNormMode mode,
    const VAECTensor *input,
    bool require_gradient
)
{
    if (mode != VAE_C_BATCH_NORM_MODE_TRAIN &&
        mode != VAE_C_BATCH_NORM_MODE_EVAL) {

        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (iic_head_validate_tensor(input) != VAE_C_OP_OK) {
        return VAE_C_OP_INVALID_STATE;
    }
    if (input->shape.rank != 2 ||
        input->shape.dims[1] != head->input_dim ||
        input->shape.dims[0] > workspace->capacity ||
        (mode == VAE_C_BATCH_NORM_MODE_TRAIN &&
         input->shape.dims[0] < 2)) {

        return VAE_C_OP_SHAPE_MISMATCH;
    }
    if (require_gradient && !input->has_grad) {
        return VAE_C_OP_GRADIENT_NOT_AVAILABLE;
    }
    if (!iic_head_all_finite(input->data, input->shape.count)) {
        return VAE_C_OP_INVALID_STATE;
    }
    if (iic_head_pointer_is_head_buffer(head, input->data) ||
        iic_head_pointer_is_head_buffer(head, input->grad) ||
        iic_head_pointer_is_workspace_buffer(workspace, input->data) ||
        iic_head_pointer_is_workspace_buffer(workspace, input->grad) ||
        iic_head_workspace_aliases_head(head, workspace)) {

        return VAE_C_OP_INVALID_ARGUMENT;
    }
    return VAE_C_OP_OK;
}


static VAECOpStatus iic_head_make_view(
    const VAECTensor *storage,
    size_t rows,
    size_t columns,
    VAECTensor *view
)
{
    VAECShape shape;
    VAECStorageStatus status;
    const size_t dims[2] = {rows, columns};

    status = vae_c_shape_make(2, dims, &shape);
    if (status != VAE_C_STORAGE_OK) {
        return iic_head_map_storage_status(status);
    }
    return iic_head_map_storage_status(
        vae_c_tensor_make_contiguous_view(storage, 0, &shape, view)
    );
}


static VAECOpStatus iic_head_make_active_views(
    const VAECIICHeadWorkspace *workspace,
    size_t batch_size,
    VAECIICHeadActiveViews *views
)
{
    VAECOpStatus status;

    if (workspace == NULL || views == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    memset(views, 0, sizeof(*views));
    status = iic_head_make_view(
        &workspace->linear1_storage,
        batch_size,
        VAE_C_IIC_HEAD_HIDDEN_DIM,
        &views->linear1_output
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = iic_head_make_view(
        &workspace->batch_norm_storage,
        batch_size,
        VAE_C_IIC_HEAD_HIDDEN_DIM,
        &views->batch_norm_output
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = iic_head_make_view(
        &workspace->relu_storage,
        batch_size,
        VAE_C_IIC_HEAD_HIDDEN_DIM,
        &views->relu_output
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = iic_head_make_view(
        &workspace->logits_storage,
        batch_size,
        workspace->output_dim,
        &views->logits
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    return iic_head_make_view(
        &workspace->softmax_storage,
        batch_size,
        workspace->output_dim,
        &views->softmax_output
    );
}


static void iic_head_commit_active_views(
    VAECIICHeadWorkspace *workspace,
    size_t batch_size,
    const VAECIICHeadActiveViews *views
)
{
    iic_head_workspace_clear_views(workspace);
    workspace->linear1_output = views->linear1_output;
    workspace->batch_norm_output = views->batch_norm_output;
    workspace->relu_output = views->relu_output;
    workspace->logits = views->logits;
    workspace->softmax_output = views->softmax_output;
    workspace->batch_size = batch_size;
}


static bool iic_head_tensors_have_exact_alias(
    const VAECTensor *left,
    const VAECTensor *right
)
{
    return
        left == right ||
        left->data == right->data ||
        (left->grad != NULL &&
         (left->grad == right->data || left->grad == right->grad)) ||
        (right->grad != NULL && left->data == right->grad);
}


static VAECOpStatus iic_head_softmax_forward(
    const VAECTensor *logits,
    VAECTensor *output
)
{
    size_t batch_size;
    size_t output_dim;
    size_t row;

    if (logits == NULL || output == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (iic_head_validate_tensor(logits) != VAE_C_OP_OK ||
        iic_head_validate_tensor(output) != VAE_C_OP_OK) {

        return VAE_C_OP_INVALID_STATE;
    }
    if (iic_head_tensors_have_exact_alias(logits, output)) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (logits->shape.rank != 2 ||
        !vae_c_shape_equal(&logits->shape, &output->shape)) {

        return VAE_C_OP_SHAPE_MISMATCH;
    }
    if (!iic_head_all_finite(logits->data, logits->shape.count)) {
        return VAE_C_OP_INVALID_STATE;
    }

    batch_size = logits->shape.dims[0];
    output_dim = logits->shape.dims[1];
    for (row = 0; row < batch_size; row++) {
        size_t offset = row * output_dim;
        size_t column;
        float maximum = logits->data[offset];
        float denominator = 0.0f;

        for (column = 1; column < output_dim; column++) {
            float value = logits->data[offset + column];

            if (value > maximum) {
                maximum = value;
            }
        }
        for (column = 0; column < output_dim; column++) {
            denominator += expf(logits->data[offset + column] - maximum);
        }
        if (!isfinite(denominator) || denominator <= 0.0f) {
            return VAE_C_OP_INVALID_STATE;
        }
        for (column = 0; column < output_dim; column++) {
            float probability =
                expf(logits->data[offset + column] - maximum) /
                denominator;

            if (!isfinite(probability) || probability < 0.0f ||
                probability > 1.0f) {

                return VAE_C_OP_INVALID_STATE;
            }
        }
    }

    for (row = 0; row < batch_size; row++) {
        size_t offset = row * output_dim;
        size_t column;
        float maximum = logits->data[offset];
        float denominator = 0.0f;

        for (column = 1; column < output_dim; column++) {
            float value = logits->data[offset + column];

            if (value > maximum) {
                maximum = value;
            }
        }
        for (column = 0; column < output_dim; column++) {
            denominator += expf(logits->data[offset + column] - maximum);
        }
        for (column = 0; column < output_dim; column++) {
            output->data[offset + column] =
                expf(logits->data[offset + column] - maximum) /
                denominator;
        }
    }
    return VAE_C_OP_OK;
}


static bool iic_head_softmax_rows_valid(const VAECTensor *softmax)
{
    size_t batch_size;
    size_t output_dim;
    size_t row;

    if (softmax == NULL || softmax->shape.rank != 2 ||
        !iic_head_all_finite(softmax->data, softmax->shape.count)) {

        return false;
    }
    batch_size = softmax->shape.dims[0];
    output_dim = softmax->shape.dims[1];
    for (row = 0; row < batch_size; row++) {
        size_t offset = row * output_dim;
        size_t column;
        float sum = 0.0f;

        for (column = 0; column < output_dim; column++) {
            float probability = softmax->data[offset + column];

            if (probability < 0.0f || probability > 1.0f) {
                return false;
            }
            sum += probability;
        }
        if (!isfinite(sum) || fabsf(sum - 1.0f) > 1.0e-4f) {
            return false;
        }
    }
    return true;
}


static VAECOpStatus iic_head_softmax_backward(
    const VAECTensor *softmax,
    VAECTensor *logits
)
{
    size_t batch_size;
    size_t output_dim;
    size_t row;

    if (softmax == NULL || logits == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (iic_head_validate_tensor(softmax) != VAE_C_OP_OK ||
        iic_head_validate_tensor(logits) != VAE_C_OP_OK) {

        return VAE_C_OP_INVALID_STATE;
    }
    if (!softmax->has_grad || !logits->has_grad) {
        return VAE_C_OP_GRADIENT_NOT_AVAILABLE;
    }
    if (iic_head_tensors_have_exact_alias(softmax, logits)) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (!vae_c_shape_equal(&softmax->shape, &logits->shape)) {
        return VAE_C_OP_SHAPE_MISMATCH;
    }
    if (!iic_head_softmax_rows_valid(softmax) ||
        !iic_head_all_finite(softmax->grad, softmax->shape.count) ||
        !iic_head_all_finite(logits->grad, logits->shape.count)) {

        return VAE_C_OP_INVALID_STATE;
    }

    batch_size = softmax->shape.dims[0];
    output_dim = softmax->shape.dims[1];
    for (row = 0; row < batch_size; row++) {
        size_t offset = row * output_dim;
        size_t column;
        float dot = 0.0f;

        for (column = 0; column < output_dim; column++) {
            float product =
                softmax->grad[offset + column] *
                softmax->data[offset + column];

            if (!isfinite(product)) {
                return VAE_C_OP_INVALID_STATE;
            }
            dot += product;
        }
        if (!isfinite(dot)) {
            return VAE_C_OP_INVALID_STATE;
        }
        for (column = 0; column < output_dim; column++) {
            float contribution =
                softmax->data[offset + column] *
                (softmax->grad[offset + column] - dot);
            float accumulated =
                logits->grad[offset + column] + contribution;

            if (!isfinite(contribution) || !isfinite(accumulated)) {
                return VAE_C_OP_INVALID_STATE;
            }
        }
    }

    for (row = 0; row < batch_size; row++) {
        size_t offset = row * output_dim;
        size_t column;
        float dot = 0.0f;

        for (column = 0; column < output_dim; column++) {
            dot +=
                softmax->grad[offset + column] *
                softmax->data[offset + column];
        }
        for (column = 0; column < output_dim; column++) {
            logits->grad[offset + column] +=
                softmax->data[offset + column] *
                (softmax->grad[offset + column] - dot);
        }
    }
    return VAE_C_OP_OK;
}


VAECOpStatus vae_c_iic_head_forward(
    VAECIICHead *head,
    VAECBatchNormMode mode,
    const VAECTensor *input,
    VAECIICHeadWorkspace *workspace
)
{
    VAECIICHeadActiveViews views;
    VAECOpStatus status;
    size_t batch_size;
    bool is_train;

    if (head == NULL || input == NULL || workspace == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (vae_c_iic_head_validate(head) != VAE_C_OP_OK ||
        vae_c_iic_head_workspace_validate(head, workspace) !=
            VAE_C_OP_OK) {

        return VAE_C_OP_INVALID_STATE;
    }
    is_train = mode == VAE_C_BATCH_NORM_MODE_TRAIN;
    status = iic_head_validate_external_input(
        head,
        workspace,
        mode,
        input,
        is_train
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    if (head->forward_generation == UINT64_MAX ||
        (is_train && head->num_batches_tracked == UINT64_MAX)) {

        return VAE_C_OP_OVERFLOW;
    }

    batch_size = input->shape.dims[0];
    status = iic_head_make_active_views(
        workspace,
        batch_size,
        &views
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }

    /* Reusing this workspace invalidates only its own earlier context. */
    workspace->backward_head = NULL;
    workspace->backward_input = NULL;
    workspace->backward_generation = UINT64_C(0);
    workspace->batch_norm_cache.valid = false;
    head->forward_generation++;

    status = vae_c_dense_forward(
        &head->linear1,
        input,
        &views.linear1_output
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    if (!iic_head_all_finite(
            views.linear1_output.data,
            views.linear1_output.shape.count)) {

        return VAE_C_OP_INVALID_STATE;
    }

    status = vae_c_batch_norm1d_forward(
        &head->batch_norm,
        mode,
        &views.linear1_output,
        &views.batch_norm_output
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    if (is_train) {
        /* This counter belongs to the BatchNorm state and advances exactly
         * when its TRAIN operation has completed successfully. */
        head->num_batches_tracked++;
    }
    if (!iic_head_all_finite(
            views.batch_norm_output.data,
            views.batch_norm_output.shape.count)) {

        return VAE_C_OP_INVALID_STATE;
    }

    status = vae_c_relu_forward(
        &views.batch_norm_output,
        &views.relu_output
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    if (!iic_head_all_finite(
            views.relu_output.data,
            views.relu_output.shape.count)) {

        return VAE_C_OP_INVALID_STATE;
    }

    status = vae_c_dense_forward(
        &head->linear2,
        &views.relu_output,
        &views.logits
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    if (!iic_head_all_finite(
            views.logits.data,
            views.logits.shape.count)) {

        return VAE_C_OP_INVALID_STATE;
    }

    status = iic_head_softmax_forward(
        &views.logits,
        &views.softmax_output
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    if (!iic_head_softmax_rows_valid(&views.softmax_output)) {
        return VAE_C_OP_INVALID_STATE;
    }

    iic_head_commit_active_views(workspace, batch_size, &views);
    if (is_train) {
        iic_head_capture_batch_norm_cache(
            &head->batch_norm,
            &workspace->batch_norm_cache
        );
        workspace->backward_head = head;
        workspace->backward_input = input;
        workspace->backward_generation = head->forward_generation;
    }
    return VAE_C_OP_OK;
}


static bool iic_head_active_data_are_finite(
    const VAECIICHeadWorkspace *workspace
)
{
    return
        iic_head_all_finite(
            workspace->linear1_output.data,
            workspace->linear1_output.shape.count) &&
        iic_head_all_finite(
            workspace->batch_norm_output.data,
            workspace->batch_norm_output.shape.count) &&
        iic_head_all_finite(
            workspace->relu_output.data,
            workspace->relu_output.shape.count) &&
        iic_head_all_finite(
            workspace->logits.data,
            workspace->logits.shape.count) &&
        iic_head_softmax_rows_valid(&workspace->softmax_output);
}


static bool iic_head_intermediate_gradients_are_zero(
    const VAECIICHeadWorkspace *workspace
)
{
    return
        iic_head_all_finite_zero(
            workspace->linear1_output.grad,
            workspace->linear1_output.shape.count) &&
        iic_head_all_finite_zero(
            workspace->batch_norm_output.grad,
            workspace->batch_norm_output.shape.count) &&
        iic_head_all_finite_zero(
            workspace->relu_output.grad,
            workspace->relu_output.shape.count) &&
        iic_head_all_finite_zero(
            workspace->logits.grad,
            workspace->logits.shape.count);
}


static bool iic_head_active_gradients_are_finite(
    const VAECIICHeadWorkspace *workspace
)
{
    return
        iic_head_all_finite(
            workspace->linear1_output.grad,
            workspace->linear1_output.shape.count) &&
        iic_head_all_finite(
            workspace->batch_norm_output.grad,
            workspace->batch_norm_output.shape.count) &&
        iic_head_all_finite(
            workspace->relu_output.grad,
            workspace->relu_output.shape.count) &&
        iic_head_all_finite(
            workspace->logits.grad,
            workspace->logits.shape.count) &&
        iic_head_all_finite(
            workspace->softmax_output.grad,
            workspace->softmax_output.shape.count);
}


VAECOpStatus vae_c_iic_head_backward(
    VAECIICHead *head,
    const VAECTensor *input,
    VAECIICHeadWorkspace *workspace
)
{
    VAECOpStatus status;
    VAECIICHeadBatchNormCache current_batch_norm_cache;

    if (head == NULL || input == NULL || workspace == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (vae_c_iic_head_validate(head) != VAE_C_OP_OK ||
        vae_c_iic_head_workspace_validate(head, workspace) !=
            VAE_C_OP_OK) {

        return VAE_C_OP_INVALID_STATE;
    }
    status = iic_head_validate_external_input(
        head,
        workspace,
        VAE_C_BATCH_NORM_MODE_TRAIN,
        input,
        true
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    if (workspace->batch_size != input->shape.dims[0]) {
        return VAE_C_OP_SHAPE_MISMATCH;
    }
    if (workspace->backward_head != head ||
        workspace->backward_input != input ||
        workspace->backward_generation == UINT64_C(0) ||
        workspace->backward_generation > head->forward_generation ||
        !workspace->batch_norm_cache.valid ||
        !iic_head_active_data_are_finite(workspace) ||
        !iic_head_intermediate_gradients_are_zero(workspace) ||
        !iic_head_all_finite(
            workspace->softmax_output.grad,
            workspace->softmax_output.shape.count) ||
        !iic_head_all_finite(input->grad, input->shape.count)) {

        return VAE_C_OP_INVALID_STATE;
    }

    /* The ticket is one-shot even if a primitive reports a later error. */
    workspace->backward_head = NULL;
    workspace->backward_input = NULL;
    workspace->backward_generation = UINT64_C(0);
    workspace->batch_norm_cache.valid = false;

    status = iic_head_softmax_backward(
        &workspace->softmax_output,
        &workspace->logits
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_dense_backward(
        &head->linear2,
        &workspace->relu_output,
        &workspace->logits
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    if (!iic_head_all_finite(
            workspace->relu_output.grad,
            workspace->relu_output.shape.count) ||
        vae_c_iic_head_validate(head) != VAE_C_OP_OK) {

        return VAE_C_OP_INVALID_STATE;
    }

    status = vae_c_relu_backward(
        &workspace->batch_norm_output,
        &workspace->relu_output
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    if (!iic_head_all_finite(
            workspace->batch_norm_output.grad,
            workspace->batch_norm_output.shape.count)) {

        return VAE_C_OP_INVALID_STATE;
    }

    /* The primitive keeps only one cache.  Install this context's batch
     * statistics for its backward, then restore the head's latest cache. */
    iic_head_capture_batch_norm_cache(
        &head->batch_norm,
        &current_batch_norm_cache
    );
    iic_head_restore_batch_norm_cache(
        &head->batch_norm,
        &workspace->batch_norm_cache
    );
    head->batch_norm.cache_valid = true;
    status = vae_c_batch_norm1d_backward(
        &head->batch_norm,
        &workspace->linear1_output,
        &workspace->batch_norm_output
    );
    iic_head_restore_batch_norm_cache(
        &head->batch_norm,
        &current_batch_norm_cache
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    if (!iic_head_all_finite(
            workspace->linear1_output.grad,
            workspace->linear1_output.shape.count) ||
        vae_c_iic_head_validate(head) != VAE_C_OP_OK) {

        return VAE_C_OP_INVALID_STATE;
    }

    status = vae_c_dense_backward(
        &head->linear1,
        input,
        &workspace->linear1_output
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    if (!iic_head_all_finite(input->grad, input->shape.count) ||
        !iic_head_active_gradients_are_finite(workspace) ||
        vae_c_iic_head_validate(head) != VAE_C_OP_OK) {

        return VAE_C_OP_INVALID_STATE;
    }
    return VAE_C_OP_OK;
}
