#include "vae_c_latent.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


static int vae_c_latent_size_multiply(
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


static VAECOpStatus vae_c_latent_validate_shape(const VAECShape *shape)
{
    VAECShape canonical;
    VAECStorageStatus storage_status;
    size_t index;

    if (shape == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    storage_status = vae_c_shape_make(
        shape->rank,
        shape->dims,
        &canonical
    );
    if (storage_status != VAE_C_STORAGE_OK) {
        return
            storage_status == VAE_C_STORAGE_OVERFLOW
                ? VAE_C_OP_OVERFLOW
                : VAE_C_OP_INVALID_STATE;
    }

    if (shape->count != canonical.count) {
        return VAE_C_OP_INVALID_STATE;
    }

    for (index = 0; index < shape->rank; index++) {
        if (shape->dims[index] != canonical.dims[index] ||
            shape->strides[index] != canonical.strides[index]) {

            return VAE_C_OP_INVALID_STATE;
        }
    }

    for (index = shape->rank;
         index < VAE_C_TENSOR_MAX_RANK;
         index++) {

        if (shape->dims[index] != 0 || shape->strides[index] != 0) {
            return VAE_C_OP_INVALID_STATE;
        }
    }

    return VAE_C_OP_OK;
}


static VAECOpStatus vae_c_latent_validate_tensor(
    const VAECTensor *tensor
)
{
    VAECOpStatus status;
    size_t array_count;
    size_t total_elements;
    size_t expected_bytes;

    if (tensor == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    status = vae_c_latent_validate_shape(&tensor->shape);
    if (status != VAE_C_OP_OK) {
        return status;
    }

    if (tensor->data == NULL) {
        return VAE_C_OP_INVALID_STATE;
    }

    if (tensor->has_grad) {
        if (tensor->grad == NULL || tensor->grad == tensor->data) {
            return VAE_C_OP_INVALID_STATE;
        }
    } else if (tensor->grad != NULL) {
        return VAE_C_OP_INVALID_STATE;
    }

    if (!tensor->owns_memory) {
        if (tensor->owned_block != NULL || tensor->owned_bytes != 0) {
            return VAE_C_OP_INVALID_STATE;
        }
        return VAE_C_OP_OK;
    }

    if (tensor->owned_block == NULL ||
        tensor->data != tensor->owned_block) {

        return VAE_C_OP_INVALID_STATE;
    }

    array_count = tensor->has_grad ? 2 : 1;
    if (vae_c_latent_size_multiply(
            tensor->shape.count,
            array_count,
            &total_elements) != 0 ||
        vae_c_latent_size_multiply(
            total_elements,
            sizeof(float),
            &expected_bytes) != 0) {

        return VAE_C_OP_INVALID_STATE;
    }

    if (tensor->owned_bytes != expected_bytes) {
        return VAE_C_OP_INVALID_STATE;
    }

    if (tensor->has_grad &&
        tensor->grad != tensor->data + tensor->shape.count) {

        return VAE_C_OP_INVALID_STATE;
    }

    return VAE_C_OP_OK;
}


static bool vae_c_latent_tensors_have_exact_alias(
    const VAECTensor *const *tensors,
    size_t tensor_count
)
{
    size_t left_index;

    for (left_index = 0;
         left_index < tensor_count;
         left_index++) {

        const VAECTensor *left = tensors[left_index];
        size_t right_index;

        for (right_index = left_index + 1;
             right_index < tensor_count;
             right_index++) {

            const VAECTensor *right = tensors[right_index];

            if (left == right ||
                left->data == right->data ||
                (left->grad != NULL &&
                 (left->grad == right->data ||
                  left->grad == right->grad)) ||
                (right->grad != NULL &&
                 left->data == right->grad)) {

                return true;
            }
        }
    }

    return false;
}


static VAECOpStatus vae_c_latent_validate_index_extent(
    const VAECShape *shape
)
{
    size_t row_start;
    size_t maximum_index;

    if (shape == NULL || shape->rank != 2) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    if (vae_c_latent_size_multiply(
            shape->dims[0] - 1,
            shape->strides[0],
            &row_start) != 0 ||
        shape->dims[1] - 1 > SIZE_MAX - row_start) {

        return VAE_C_OP_OVERFLOW;
    }

    maximum_index = row_start + shape->dims[1] - 1;
    return
        maximum_index < shape->count
            ? VAE_C_OP_OK
            : VAE_C_OP_INVALID_STATE;
}


static VAECOpStatus vae_c_latent_validate_split(
    const VAECTensor *packed,
    const VAECTensor *mu,
    const VAECTensor *logvar,
    bool require_gradients,
    size_t *latent_count
)
{
    const VAECTensor *tensors[3];
    VAECOpStatus status;
    size_t doubled_latent_count;
    size_t index;

    if (packed == NULL || mu == NULL || logvar == NULL ||
        latent_count == NULL) {

        return VAE_C_OP_INVALID_ARGUMENT;
    }

    tensors[0] = packed;
    tensors[1] = mu;
    tensors[2] = logvar;

    for (index = 0; index < 3; index++) {
        status = vae_c_latent_validate_tensor(tensors[index]);
        if (status != VAE_C_OP_OK) {
            return status;
        }
    }

    if (vae_c_latent_tensors_have_exact_alias(tensors, 3)) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    if (packed->shape.rank != 2 ||
        mu->shape.rank != 2 ||
        logvar->shape.rank != 2 ||
        packed->shape.dims[0] != mu->shape.dims[0] ||
        packed->shape.dims[0] != logvar->shape.dims[0] ||
        mu->shape.dims[1] != logvar->shape.dims[1]) {

        return VAE_C_OP_SHAPE_MISMATCH;
    }

    if (vae_c_latent_size_multiply(
            mu->shape.dims[1],
            2,
            &doubled_latent_count) != 0) {

        return VAE_C_OP_OVERFLOW;
    }

    if (packed->shape.dims[1] != doubled_latent_count) {
        return VAE_C_OP_SHAPE_MISMATCH;
    }

    for (index = 0; index < 3; index++) {
        status = vae_c_latent_validate_index_extent(
            &tensors[index]->shape
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
    }

    if (require_gradients &&
        (!packed->has_grad || !mu->has_grad || !logvar->has_grad)) {

        return VAE_C_OP_GRADIENT_NOT_AVAILABLE;
    }

    *latent_count = mu->shape.dims[1];
    return VAE_C_OP_OK;
}


static bool vae_c_latent_mode_is_valid(VAECLatentMode mode)
{
    return
        mode == VAE_C_LATENT_MODE_STOCHASTIC ||
        mode == VAE_C_LATENT_MODE_MEAN;
}


static VAECOpStatus vae_c_latent_validate_reparameterization(
    VAECLatentMode mode,
    const VAECTensor *mu,
    const VAECTensor *logvar,
    const VAECTensor *epsilon,
    const VAECTensor *z,
    bool require_gradients
)
{
    const VAECTensor *tensors[4];
    size_t tensor_count;
    VAECOpStatus status;
    size_t index;

    if (!vae_c_latent_mode_is_valid(mode)) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    if (mu == NULL || logvar == NULL || z == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    if ((mode == VAE_C_LATENT_MODE_STOCHASTIC && epsilon == NULL) ||
        (mode == VAE_C_LATENT_MODE_MEAN && epsilon != NULL)) {

        return VAE_C_OP_INVALID_ARGUMENT;
    }

    tensors[0] = mu;
    tensors[1] = logvar;
    tensors[2] = z;
    tensor_count = 3;

    if (epsilon != NULL) {
        tensors[3] = epsilon;
        tensor_count = 4;
    }

    for (index = 0; index < tensor_count; index++) {
        status = vae_c_latent_validate_tensor(tensors[index]);
        if (status != VAE_C_OP_OK) {
            return status;
        }
    }

    if (vae_c_latent_tensors_have_exact_alias(
            tensors,
            tensor_count)) {

        return VAE_C_OP_INVALID_ARGUMENT;
    }

    if (mu->shape.rank != 2 ||
        logvar->shape.rank != 2 ||
        z->shape.rank != 2 ||
        !vae_c_shape_equal(&mu->shape, &logvar->shape) ||
        !vae_c_shape_equal(&mu->shape, &z->shape) ||
        (epsilon != NULL &&
         (epsilon->shape.rank != 2 ||
          !vae_c_shape_equal(&mu->shape, &epsilon->shape)))) {

        return VAE_C_OP_SHAPE_MISMATCH;
    }

    for (index = 0; index < tensor_count; index++) {
        status = vae_c_latent_validate_index_extent(
            &tensors[index]->shape
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
    }

    if (require_gradients &&
        (!mu->has_grad || !logvar->has_grad || !z->has_grad)) {

        return VAE_C_OP_GRADIENT_NOT_AVAILABLE;
    }

    return VAE_C_OP_OK;
}


VAECOpStatus vae_c_latent_split_forward(
    const VAECTensor *packed,
    VAECTensor *mu,
    VAECTensor *logvar
)
{
    VAECOpStatus status;
    size_t latent_count;
    size_t index;
    size_t batch_index;
    size_t latent_index;

    status = vae_c_latent_validate_split(
        packed,
        mu,
        logvar,
        false,
        &latent_count
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }

    for (index = 0; index < packed->shape.count; index++) {
        if (!isfinite(packed->data[index])) {
            return VAE_C_OP_INVALID_STATE;
        }
    }

    for (batch_index = 0;
         batch_index < packed->shape.dims[0];
         batch_index++) {

        size_t packed_row = batch_index * packed->shape.strides[0];
        size_t output_row = batch_index * mu->shape.strides[0];

        for (latent_index = 0;
             latent_index < latent_count;
             latent_index++) {

            mu->data[output_row + latent_index] =
                packed->data[packed_row + latent_index];
            logvar->data[output_row + latent_index] =
                packed->data[
                    packed_row + latent_count + latent_index
                ];
        }
    }

    return VAE_C_OP_OK;
}


VAECOpStatus vae_c_latent_split_backward(
    const VAECTensor *packed,
    const VAECTensor *mu,
    const VAECTensor *logvar
)
{
    VAECOpStatus status;
    size_t latent_count;
    size_t batch_index;
    size_t latent_index;

    status = vae_c_latent_validate_split(
        packed,
        mu,
        logvar,
        true,
        &latent_count
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }

    for (batch_index = 0;
         batch_index < packed->shape.dims[0];
         batch_index++) {

        size_t packed_row = batch_index * packed->shape.strides[0];
        size_t output_row = batch_index * mu->shape.strides[0];

        for (latent_index = 0;
             latent_index < latent_count;
             latent_index++) {

            size_t mu_index = output_row + latent_index;
            size_t packed_mu_index = packed_row + latent_index;
            size_t packed_logvar_index =
                packed_row + latent_count + latent_index;
            float mu_upstream = mu->grad[mu_index];
            float logvar_upstream = logvar->grad[mu_index];
            float next_mu_gradient;
            float next_logvar_gradient;

            if (!isfinite(mu_upstream) ||
                !isfinite(logvar_upstream)) {

                return VAE_C_OP_INVALID_STATE;
            }

            next_mu_gradient =
                packed->grad[packed_mu_index] + mu_upstream;
            next_logvar_gradient =
                packed->grad[packed_logvar_index] +
                logvar_upstream;

            if (!isfinite(next_mu_gradient) ||
                !isfinite(next_logvar_gradient)) {

                return VAE_C_OP_INVALID_STATE;
            }
        }
    }

    for (batch_index = 0;
         batch_index < packed->shape.dims[0];
         batch_index++) {

        size_t packed_row = batch_index * packed->shape.strides[0];
        size_t output_row = batch_index * mu->shape.strides[0];

        for (latent_index = 0;
             latent_index < latent_count;
             latent_index++) {

            packed->grad[packed_row + latent_index] +=
                mu->grad[output_row + latent_index];
            packed->grad[
                packed_row + latent_count + latent_index
            ] += logvar->grad[output_row + latent_index];
        }
    }

    return VAE_C_OP_OK;
}


VAECOpStatus vae_c_reparameterize_forward(
    VAECLatentMode mode,
    const VAECTensor *mu,
    const VAECTensor *logvar,
    const VAECTensor *epsilon,
    VAECTensor *z
)
{
    VAECOpStatus status;
    size_t index;

    status = vae_c_latent_validate_reparameterization(
        mode,
        mu,
        logvar,
        epsilon,
        z,
        false
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }

    for (index = 0; index < mu->shape.count; index++) {
        float mu_value = mu->data[index];
        float logvar_value = logvar->data[index];

        if (!isfinite(mu_value) || !isfinite(logvar_value)) {
            return VAE_C_OP_INVALID_STATE;
        }

        if (mode == VAE_C_LATENT_MODE_STOCHASTIC) {
            float epsilon_value = epsilon->data[index];
            float half_logvar = 0.5f * logvar_value;
            float sigma = expf(half_logvar);
            float result = mu_value + sigma * epsilon_value;

            if (!isfinite(epsilon_value) ||
                !isfinite(half_logvar) ||
                !isfinite(sigma) ||
                !(sigma > 0.0f) ||
                !isfinite(result)) {

                return VAE_C_OP_INVALID_STATE;
            }
        }
    }

    if (mode == VAE_C_LATENT_MODE_MEAN) {
        for (index = 0; index < mu->shape.count; index++) {
            z->data[index] = mu->data[index];
        }
    } else {
        for (index = 0; index < mu->shape.count; index++) {
            float sigma = expf(0.5f * logvar->data[index]);

            z->data[index] =
                mu->data[index] + sigma * epsilon->data[index];
        }
    }

    return VAE_C_OP_OK;
}


VAECOpStatus vae_c_reparameterize_backward(
    VAECLatentMode mode,
    const VAECTensor *mu,
    const VAECTensor *logvar,
    const VAECTensor *epsilon,
    const VAECTensor *z
)
{
    VAECOpStatus status;
    size_t index;

    status = vae_c_latent_validate_reparameterization(
        mode,
        mu,
        logvar,
        epsilon,
        z,
        true
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }

    for (index = 0; index < mu->shape.count; index++) {
        float mu_value = mu->data[index];
        float logvar_value = logvar->data[index];
        float upstream = z->grad[index];
        float next_mu_gradient;

        if (!isfinite(mu_value) ||
            !isfinite(logvar_value) ||
            !isfinite(upstream)) {

            return VAE_C_OP_INVALID_STATE;
        }

        next_mu_gradient = mu->grad[index] + upstream;
        if (!isfinite(next_mu_gradient)) {
            return VAE_C_OP_INVALID_STATE;
        }

        if (mode == VAE_C_LATENT_MODE_STOCHASTIC) {
            float epsilon_value = epsilon->data[index];
            float half_logvar = 0.5f * logvar_value;
            float sigma = expf(half_logvar);
            float contribution =
                upstream * 0.5f * sigma * epsilon_value;
            float next_logvar_gradient =
                logvar->grad[index] + contribution;

            if (!isfinite(epsilon_value) ||
                !isfinite(half_logvar) ||
                !isfinite(sigma) ||
                !(sigma > 0.0f) ||
                !isfinite(contribution) ||
                !isfinite(next_logvar_gradient)) {

                return VAE_C_OP_INVALID_STATE;
            }
        }
    }

    if (mode == VAE_C_LATENT_MODE_MEAN) {
        for (index = 0; index < mu->shape.count; index++) {
            mu->grad[index] += z->grad[index];
        }
    } else {
        for (index = 0; index < mu->shape.count; index++) {
            float sigma = expf(0.5f * logvar->data[index]);

            mu->grad[index] += z->grad[index];
            logvar->grad[index] +=
                z->grad[index] *
                0.5f *
                sigma *
                epsilon->data[index];
        }
    }

    return VAE_C_OP_OK;
}
