#ifndef VAE_RESIZE_U8_H
#define VAE_RESIZE_U8_H

#include <stdint.h>

#include "vae_tensor_spec.h"

int gs_resize_tensor_u8(
    const uint8_t *src,
    const GSTensorSpec *src_spec,
    uint8_t *dst,
    const GSTensorSpec *dst_spec
);

#endif