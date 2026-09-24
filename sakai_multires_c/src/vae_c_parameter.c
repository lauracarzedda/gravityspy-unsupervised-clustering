#include "vae_c_parameter.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>


static int vae_c_parameter_size_multiply(
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


static VAECStorageStatus vae_c_parameter_canonical_shape(
    const VAECShape *shape,
    VAECShape *canonical
)
{
    VAECStorageStatus status;

    if (shape == NULL || canonical == NULL) {
        return VAE_C_STORAGE_INVALID_ARGUMENT;
    }

    status = vae_c_shape_make(shape->rank, shape->dims, canonical);
    if (status != VAE_C_STORAGE_OK) {
        return status;
    }

    if (!vae_c_shape_equal(shape, canonical)) {
        return VAE_C_STORAGE_INVALID_STATE;
    }

    return VAE_C_STORAGE_OK;
}


static bool vae_c_parameter_shape_is_zero_initialized(
    const VAECShape *shape
)
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


static VAECStorageStatus vae_c_parameter_validate_nonempty(
    const VAECParameter *parameter
)
{
    VAECShape canonical;
    size_t total_elements;
    size_t expected_bytes;
    VAECStorageStatus status;

    if (parameter == NULL) {
        return VAE_C_STORAGE_INVALID_ARGUMENT;
    }

    status = vae_c_parameter_canonical_shape(
        &parameter->shape,
        &canonical
    );
    if (status != VAE_C_STORAGE_OK) {
        return VAE_C_STORAGE_INVALID_STATE;
    }

    if (parameter->owned_block == NULL ||
        parameter->values != parameter->owned_block ||
        parameter->gradients !=
            parameter->values + parameter->shape.count ||
        parameter->adam_m !=
            parameter->gradients + parameter->shape.count ||
        parameter->adam_v !=
            parameter->adam_m + parameter->shape.count) {

        return VAE_C_STORAGE_INVALID_STATE;
    }

    if (vae_c_parameter_size_multiply(
            parameter->shape.count,
            4,
            &total_elements) != 0 ||
        vae_c_parameter_size_multiply(
            total_elements,
            sizeof(float),
            &expected_bytes) != 0 ||
        parameter->owned_bytes != expected_bytes) {

        return VAE_C_STORAGE_INVALID_STATE;
    }

    return VAE_C_STORAGE_OK;
}


void vae_c_parameter_init(VAECParameter *parameter)
{
    if (parameter != NULL) {
        memset(parameter, 0, sizeof(*parameter));
    }
}


void vae_c_parameter_free(VAECParameter *parameter)
{
    if (parameter == NULL) {
        return;
    }

    free(parameter->owned_block);
    vae_c_parameter_init(parameter);
}


bool vae_c_parameter_is_empty(const VAECParameter *parameter)
{
    if (parameter == NULL) {
        return false;
    }

    return
        vae_c_parameter_shape_is_zero_initialized(&parameter->shape) &&
        parameter->parameter_key == UINT64_C(0) &&
        parameter->values == NULL &&
        parameter->gradients == NULL &&
        parameter->adam_m == NULL &&
        parameter->adam_v == NULL &&
        parameter->owned_block == NULL &&
        parameter->owned_bytes == 0;
}


VAECStorageStatus vae_c_parameter_allocate(
    VAECParameter *parameter,
    const VAECShape *shape,
    uint64_t parameter_key
)
{
    VAECShape canonical;
    VAECParameter candidate;
    VAECStorageStatus status;
    size_t total_elements;
    size_t total_bytes;
    float *block;

    if (parameter == NULL || shape == NULL) {
        return VAE_C_STORAGE_INVALID_ARGUMENT;
    }

    if (!vae_c_parameter_is_empty(parameter)) {
        return VAE_C_STORAGE_OBJECT_NOT_EMPTY;
    }

    status = vae_c_parameter_canonical_shape(shape, &canonical);
    if (status != VAE_C_STORAGE_OK) {
        return status;
    }

    if (vae_c_parameter_size_multiply(
            canonical.count,
            4,
            &total_elements) != 0 ||
        vae_c_parameter_size_multiply(
            total_elements,
            sizeof(float),
            &total_bytes) != 0) {

        return VAE_C_STORAGE_OVERFLOW;
    }

    block = (float *)calloc(total_elements, sizeof(float));
    if (block == NULL) {
        return VAE_C_STORAGE_ALLOCATION_FAILED;
    }

    vae_c_parameter_init(&candidate);
    candidate.shape = canonical;
    candidate.parameter_key = parameter_key;
    candidate.values = block;
    candidate.gradients = block + canonical.count;
    candidate.adam_m = candidate.gradients + canonical.count;
    candidate.adam_v = candidate.adam_m + canonical.count;
    candidate.owned_block = block;
    candidate.owned_bytes = total_bytes;

    *parameter = candidate;
    return VAE_C_STORAGE_OK;
}


VAECStorageStatus vae_c_parameter_zero_grad(VAECParameter *parameter)
{
    VAECStorageStatus status;
    size_t index;

    if (parameter == NULL) {
        return VAE_C_STORAGE_INVALID_ARGUMENT;
    }

    if (vae_c_parameter_is_empty(parameter)) {
        return VAE_C_STORAGE_OK;
    }

    status = vae_c_parameter_validate_nonempty(parameter);
    if (status != VAE_C_STORAGE_OK) {
        return VAE_C_STORAGE_INVALID_STATE;
    }

    for (index = 0; index < parameter->shape.count; index++) {
        parameter->gradients[index] = 0.0f;
    }

    return VAE_C_STORAGE_OK;
}


VAECStorageStatus vae_c_parameter_zero_optimizer_state(
    VAECParameter *parameter
)
{
    VAECStorageStatus status;
    size_t index;

    if (parameter == NULL) {
        return VAE_C_STORAGE_INVALID_ARGUMENT;
    }

    if (vae_c_parameter_is_empty(parameter)) {
        return VAE_C_STORAGE_OK;
    }

    status = vae_c_parameter_validate_nonempty(parameter);
    if (status != VAE_C_STORAGE_OK) {
        return VAE_C_STORAGE_INVALID_STATE;
    }

    for (index = 0; index < parameter->shape.count; index++) {
        parameter->adam_m[index] = 0.0f;
        parameter->adam_v[index] = 0.0f;
    }

    return VAE_C_STORAGE_OK;
}
