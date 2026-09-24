#include "vae_c_tensor.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>


static int vae_c_size_multiply(
    size_t left,
    size_t right,
    size_t *result
)
{
    if (result == NULL) {
        return 1;
    }

    if (left != 0 && right > SIZE_MAX / left) {
        return 1;
    }

    *result = left * right;
    return 0;
}


static int vae_c_size_add(
    size_t left,
    size_t right,
    size_t *result
)
{
    if (result == NULL || right > SIZE_MAX - left) {
        return 1;
    }

    *result = left + right;
    return 0;
}


static VAECStorageStatus vae_c_shape_canonicalize(
    const VAECShape *shape,
    VAECShape *canonical
)
{
    VAECStorageStatus status;
    size_t index;

    if (shape == NULL || canonical == NULL) {
        return VAE_C_STORAGE_INVALID_ARGUMENT;
    }

    status = vae_c_shape_make(shape->rank, shape->dims, canonical);
    if (status != VAE_C_STORAGE_OK) {
        return status;
    }

    if (shape->count != canonical->count) {
        return VAE_C_STORAGE_INVALID_STATE;
    }

    for (index = 0; index < shape->rank; index++) {
        if (shape->dims[index] != canonical->dims[index] ||
            shape->strides[index] != canonical->strides[index]) {

            return VAE_C_STORAGE_INVALID_STATE;
        }
    }

    for (index = shape->rank;
         index < VAE_C_TENSOR_MAX_RANK;
         index++) {

        if (shape->dims[index] != 0 || shape->strides[index] != 0) {
            return VAE_C_STORAGE_INVALID_STATE;
        }
    }

    return VAE_C_STORAGE_OK;
}


static bool vae_c_shape_is_zero_initialized(const VAECShape *shape)
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


static VAECStorageStatus vae_c_tensor_validate_nonempty(
    const VAECTensor *tensor
)
{
    VAECShape canonical;
    size_t array_count;
    size_t total_elements;
    size_t expected_bytes;
    VAECStorageStatus status;

    if (tensor == NULL) {
        return VAE_C_STORAGE_INVALID_ARGUMENT;
    }

    status = vae_c_shape_canonicalize(&tensor->shape, &canonical);
    if (status != VAE_C_STORAGE_OK) {
        return VAE_C_STORAGE_INVALID_STATE;
    }

    if (tensor->data == NULL) {
        return VAE_C_STORAGE_INVALID_STATE;
    }

    if (tensor->has_grad) {
        if (tensor->grad == NULL) {
            return VAE_C_STORAGE_INVALID_STATE;
        }
    } else if (tensor->grad != NULL) {
        return VAE_C_STORAGE_INVALID_STATE;
    }

    if (!tensor->owns_memory) {
        if (tensor->owned_block != NULL || tensor->owned_bytes != 0) {
            return VAE_C_STORAGE_INVALID_STATE;
        }
        return VAE_C_STORAGE_OK;
    }

    if (tensor->owned_block == NULL ||
        tensor->data != tensor->owned_block) {

        return VAE_C_STORAGE_INVALID_STATE;
    }

    array_count = tensor->has_grad ? 2 : 1;
    if (vae_c_size_multiply(
            tensor->shape.count,
            array_count,
            &total_elements) != 0 ||
        vae_c_size_multiply(
            total_elements,
            sizeof(float),
            &expected_bytes) != 0) {

        return VAE_C_STORAGE_INVALID_STATE;
    }

    if (tensor->owned_bytes != expected_bytes) {
        return VAE_C_STORAGE_INVALID_STATE;
    }

    if (tensor->has_grad &&
        tensor->grad != tensor->data + tensor->shape.count) {

        return VAE_C_STORAGE_INVALID_STATE;
    }

    return VAE_C_STORAGE_OK;
}


void vae_c_shape_init(VAECShape *shape)
{
    if (shape != NULL) {
        memset(shape, 0, sizeof(*shape));
    }
}


VAECStorageStatus vae_c_shape_make(
    size_t rank,
    const size_t *dims,
    VAECShape *shape
)
{
    VAECShape candidate;
    size_t running_count;
    size_t index;

    if (dims == NULL || shape == NULL) {
        return VAE_C_STORAGE_INVALID_ARGUMENT;
    }

    if (rank == 0 || rank > VAE_C_TENSOR_MAX_RANK) {
        return VAE_C_STORAGE_INVALID_RANK;
    }

    vae_c_shape_init(&candidate);
    candidate.rank = rank;
    running_count = 1;

    for (index = rank; index > 0; index--) {
        size_t dimension = dims[index - 1];

        if (dimension == 0) {
            return VAE_C_STORAGE_INVALID_DIMENSION;
        }

        candidate.dims[index - 1] = dimension;
        candidate.strides[index - 1] = running_count;

        if (vae_c_size_multiply(
                running_count,
                dimension,
                &running_count) != 0) {

            return VAE_C_STORAGE_OVERFLOW;
        }
    }

    candidate.count = running_count;
    *shape = candidate;
    return VAE_C_STORAGE_OK;
}


bool vae_c_shape_equal(const VAECShape *left, const VAECShape *right)
{
    VAECShape canonical_left;
    VAECShape canonical_right;
    size_t index;

    if (left == NULL || right == NULL) {
        return false;
    }

    if (vae_c_shape_canonicalize(left, &canonical_left) !=
            VAE_C_STORAGE_OK ||
        vae_c_shape_canonicalize(right, &canonical_right) !=
            VAE_C_STORAGE_OK) {

        return false;
    }

    if (left->rank != right->rank || left->count != right->count) {

        return false;
    }

    for (index = 0; index < left->rank; index++) {
        if (left->dims[index] != right->dims[index] ||
            left->strides[index] != right->strides[index]) {

            return false;
        }
    }

    return true;
}


VAECStorageStatus vae_c_shape_offset(
    const VAECShape *shape,
    const size_t *indices,
    size_t index_count,
    size_t *offset
)
{
    VAECShape canonical;
    VAECStorageStatus status;
    size_t candidate_offset;
    size_t index;

    if (shape == NULL || indices == NULL || offset == NULL) {
        return VAE_C_STORAGE_INVALID_ARGUMENT;
    }

    status = vae_c_shape_canonicalize(shape, &canonical);
    if (status != VAE_C_STORAGE_OK) {
        return status;
    }

    if (index_count != shape->rank) {
        return VAE_C_STORAGE_SHAPE_MISMATCH;
    }

    candidate_offset = 0;

    for (index = 0; index < shape->rank; index++) {
        size_t term;

        if (indices[index] >= shape->dims[index]) {
            return VAE_C_STORAGE_OUT_OF_RANGE;
        }

        if (vae_c_size_multiply(
                indices[index],
                shape->strides[index],
                &term) != 0 ||
            vae_c_size_add(
                candidate_offset,
                term,
                &candidate_offset) != 0) {

            return VAE_C_STORAGE_OVERFLOW;
        }
    }

    if (candidate_offset >= shape->count) {
        return VAE_C_STORAGE_INVALID_STATE;
    }

    *offset = candidate_offset;
    return VAE_C_STORAGE_OK;
}


void vae_c_tensor_init(VAECTensor *tensor)
{
    if (tensor != NULL) {
        memset(tensor, 0, sizeof(*tensor));
    }
}


void vae_c_tensor_free(VAECTensor *tensor)
{
    if (tensor == NULL) {
        return;
    }

    if (tensor->owns_memory) {
        free(tensor->owned_block);
    }

    vae_c_tensor_init(tensor);
}


bool vae_c_tensor_is_empty(const VAECTensor *tensor)
{
    if (tensor == NULL) {
        return false;
    }

    return
        vae_c_shape_is_zero_initialized(&tensor->shape) &&
        tensor->data == NULL &&
        tensor->grad == NULL &&
        tensor->owned_block == NULL &&
        tensor->owned_bytes == 0 &&
        !tensor->owns_memory &&
        !tensor->has_grad;
}


VAECStorageStatus vae_c_tensor_allocate(
    VAECTensor *tensor,
    const VAECShape *shape,
    bool with_gradient
)
{
    VAECShape canonical;
    VAECTensor candidate;
    VAECStorageStatus status;
    size_t array_count;
    size_t total_elements;
    size_t total_bytes;
    float *block;

    if (tensor == NULL || shape == NULL) {
        return VAE_C_STORAGE_INVALID_ARGUMENT;
    }

    if (!vae_c_tensor_is_empty(tensor)) {
        return VAE_C_STORAGE_OBJECT_NOT_EMPTY;
    }

    status = vae_c_shape_canonicalize(shape, &canonical);
    if (status != VAE_C_STORAGE_OK) {
        return status;
    }

    array_count = with_gradient ? 2 : 1;

    if (vae_c_size_multiply(
            canonical.count,
            array_count,
            &total_elements) != 0 ||
        vae_c_size_multiply(
            total_elements,
            sizeof(float),
            &total_bytes) != 0) {

        return VAE_C_STORAGE_OVERFLOW;
    }

    block = (float *)calloc(total_elements, sizeof(float));
    if (block == NULL) {
        return VAE_C_STORAGE_ALLOCATION_FAILED;
    }

    vae_c_tensor_init(&candidate);
    candidate.shape = canonical;
    candidate.data = block;
    candidate.grad =
        with_gradient
            ? block + canonical.count
            : NULL;
    candidate.owned_block = block;
    candidate.owned_bytes = total_bytes;
    candidate.owns_memory = true;
    candidate.has_grad = with_gradient;

    *tensor = candidate;
    return VAE_C_STORAGE_OK;
}


VAECStorageStatus vae_c_tensor_make_contiguous_view(
    const VAECTensor *source,
    size_t element_offset,
    const VAECShape *view_shape,
    VAECTensor *view
)
{
    VAECShape canonical_view_shape;
    VAECTensor candidate;
    VAECStorageStatus status;
    size_t end_offset;

    if (source == NULL || view_shape == NULL || view == NULL) {
        return VAE_C_STORAGE_INVALID_ARGUMENT;
    }

    if (source == view) {
        return VAE_C_STORAGE_INVALID_ARGUMENT;
    }

    if (!vae_c_tensor_is_empty(view)) {
        return VAE_C_STORAGE_OBJECT_NOT_EMPTY;
    }

    status = vae_c_tensor_validate_nonempty(source);
    if (status != VAE_C_STORAGE_OK) {
        return VAE_C_STORAGE_INVALID_STATE;
    }

    status = vae_c_shape_canonicalize(
        view_shape,
        &canonical_view_shape
    );
    if (status != VAE_C_STORAGE_OK) {
        return status;
    }

    if (vae_c_size_add(
            element_offset,
            canonical_view_shape.count,
            &end_offset) != 0) {

        return VAE_C_STORAGE_OVERFLOW;
    }

    if (end_offset > source->shape.count) {
        return VAE_C_STORAGE_OUT_OF_RANGE;
    }

    vae_c_tensor_init(&candidate);
    candidate.shape = canonical_view_shape;
    candidate.data = source->data + element_offset;
    candidate.grad =
        source->has_grad
            ? source->grad + element_offset
            : NULL;
    candidate.owned_block = NULL;
    candidate.owned_bytes = 0;
    candidate.owns_memory = false;
    candidate.has_grad = source->has_grad;

    *view = candidate;
    return VAE_C_STORAGE_OK;
}


VAECStorageStatus vae_c_tensor_zero_data(VAECTensor *tensor)
{
    VAECStorageStatus status;
    size_t index;

    if (tensor == NULL) {
        return VAE_C_STORAGE_INVALID_ARGUMENT;
    }

    if (vae_c_tensor_is_empty(tensor)) {
        return VAE_C_STORAGE_OK;
    }

    status = vae_c_tensor_validate_nonempty(tensor);
    if (status != VAE_C_STORAGE_OK) {
        return VAE_C_STORAGE_INVALID_STATE;
    }

    for (index = 0; index < tensor->shape.count; index++) {
        tensor->data[index] = 0.0f;
    }

    return VAE_C_STORAGE_OK;
}


VAECStorageStatus vae_c_tensor_zero_grad(VAECTensor *tensor)
{
    VAECStorageStatus status;
    size_t index;

    if (tensor == NULL) {
        return VAE_C_STORAGE_INVALID_ARGUMENT;
    }

    if (vae_c_tensor_is_empty(tensor)) {
        return VAE_C_STORAGE_OK;
    }

    status = vae_c_tensor_validate_nonempty(tensor);
    if (status != VAE_C_STORAGE_OK) {
        return VAE_C_STORAGE_INVALID_STATE;
    }

    if (!tensor->has_grad) {
        return VAE_C_STORAGE_GRADIENT_NOT_AVAILABLE;
    }

    for (index = 0; index < tensor->shape.count; index++) {
        tensor->grad[index] = 0.0f;
    }

    return VAE_C_STORAGE_OK;
}


const char *vae_c_storage_status_name(VAECStorageStatus status)
{
    switch (status) {
        case VAE_C_STORAGE_OK:
            return "success";
        case VAE_C_STORAGE_INVALID_ARGUMENT:
            return "invalid_argument";
        case VAE_C_STORAGE_INVALID_RANK:
            return "invalid_rank";
        case VAE_C_STORAGE_INVALID_DIMENSION:
            return "invalid_dimension";
        case VAE_C_STORAGE_OVERFLOW:
            return "overflow";
        case VAE_C_STORAGE_ALLOCATION_FAILED:
            return "allocation_failed";
        case VAE_C_STORAGE_OBJECT_NOT_EMPTY:
            return "object_not_empty";
        case VAE_C_STORAGE_INVALID_STATE:
            return "invalid_state";
        case VAE_C_STORAGE_SHAPE_MISMATCH:
            return "shape_mismatch";
        case VAE_C_STORAGE_OUT_OF_RANGE:
            return "out_of_range";
        case VAE_C_STORAGE_GRADIENT_NOT_AVAILABLE:
            return "gradient_not_available";
        default:
            return "unknown";
    }
}
