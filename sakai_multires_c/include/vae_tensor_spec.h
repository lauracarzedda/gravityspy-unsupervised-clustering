#ifndef VAE_TENSOR_SPEC_H
#define VAE_TENSOR_SPEC_H

#include <stddef.h>

#define GS_MULTI_CHANNELS 4

typedef struct {
    int channels;
    int height;
    int width;

    size_t pixels_per_channel;
    size_t values_per_event;
    size_t bytes_u8;
    size_t bytes_f32;
} GSTensorSpec;

int gs_tensor_spec_init(GSTensorSpec *spec, int side);

int gs_tensor_spec_is_valid(const GSTensorSpec *spec);

const char *gs_tensor_spec_name(const GSTensorSpec *spec);

void gs_tensor_spec_print(const GSTensorSpec *spec);

#endif