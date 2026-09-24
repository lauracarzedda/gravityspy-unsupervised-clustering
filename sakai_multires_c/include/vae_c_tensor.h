#ifndef VAE_C_TENSOR_H
#define VAE_C_TENSOR_H

#include <stdbool.h>
#include <stddef.h>


#define VAE_C_TENSOR_MAX_RANK 4


typedef enum {
    VAE_C_STORAGE_OK = 0,
    VAE_C_STORAGE_INVALID_ARGUMENT,
    VAE_C_STORAGE_INVALID_RANK,
    VAE_C_STORAGE_INVALID_DIMENSION,
    VAE_C_STORAGE_OVERFLOW,
    VAE_C_STORAGE_ALLOCATION_FAILED,
    VAE_C_STORAGE_OBJECT_NOT_EMPTY,
    VAE_C_STORAGE_INVALID_STATE,
    VAE_C_STORAGE_SHAPE_MISMATCH,
    VAE_C_STORAGE_OUT_OF_RANGE,
    VAE_C_STORAGE_GRADIENT_NOT_AVAILABLE
} VAECStorageStatus;


typedef struct {
    size_t rank;
    size_t dims[VAE_C_TENSOR_MAX_RANK];
    size_t strides[VAE_C_TENSOR_MAX_RANK];
    size_t count;
} VAECShape;


typedef struct {
    VAECShape shape;

    float *data;
    float *grad;

    float *owned_block;
    size_t owned_bytes;

    bool owns_memory;
    bool has_grad;
} VAECTensor;


void vae_c_shape_init(VAECShape *shape);

VAECStorageStatus vae_c_shape_make(
    size_t rank,
    const size_t *dims,
    VAECShape *shape
);

bool vae_c_shape_equal(const VAECShape *left, const VAECShape *right);

VAECStorageStatus vae_c_shape_offset(
    const VAECShape *shape,
    const size_t *indices,
    size_t index_count,
    size_t *offset
);

void vae_c_tensor_init(VAECTensor *tensor);

void vae_c_tensor_free(VAECTensor *tensor);

bool vae_c_tensor_is_empty(const VAECTensor *tensor);

VAECStorageStatus vae_c_tensor_allocate(
    VAECTensor *tensor,
    const VAECShape *shape,
    bool with_gradient
);

/*
 * Contiguous views do not own memory and do not keep their source alive.
 * The source must outlive every use of the view. Using a view after its
 * source has been freed is undefined behavior. Arbitrary-stride views are
 * deliberately outside this API.
 */
VAECStorageStatus vae_c_tensor_make_contiguous_view(
    const VAECTensor *source,
    size_t element_offset,
    const VAECShape *view_shape,
    VAECTensor *view
);

VAECStorageStatus vae_c_tensor_zero_data(VAECTensor *tensor);

VAECStorageStatus vae_c_tensor_zero_grad(VAECTensor *tensor);

const char *vae_c_storage_status_name(VAECStorageStatus status);

#endif
