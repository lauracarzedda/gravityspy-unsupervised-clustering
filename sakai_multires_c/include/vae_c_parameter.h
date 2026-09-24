#ifndef VAE_C_PARAMETER_H
#define VAE_C_PARAMETER_H

#include "vae_c_tensor.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


typedef struct {
    VAECShape shape;
    uint64_t parameter_key;

    float *values;
    float *gradients;
    float *adam_m;
    float *adam_v;

    float *owned_block;
    size_t owned_bytes;
} VAECParameter;


void vae_c_parameter_init(VAECParameter *parameter);

void vae_c_parameter_free(VAECParameter *parameter);

bool vae_c_parameter_is_empty(const VAECParameter *parameter);

VAECStorageStatus vae_c_parameter_allocate(
    VAECParameter *parameter,
    const VAECShape *shape,
    uint64_t parameter_key
);

VAECStorageStatus vae_c_parameter_zero_grad(VAECParameter *parameter);

/*
 * This function only clears the memory reserved for Adam m and v.
 * It does not implement an Adam update or any optimizer behavior.
 */
VAECStorageStatus vae_c_parameter_zero_optimizer_state(
    VAECParameter *parameter
);

#endif
