#ifndef VAE_TENSOR_IO_H
#define VAE_TENSOR_IO_H

#include <stdint.h>

#include "vae_tensor_spec.h"


/*
   Legge esattamente un tensore uint8 della shape indicata.
*/
int gs_tensor_read_u8(
    const char *path,
    const GSTensorSpec *spec,
    uint8_t *out_u8
);


/*
   Converte un tensore già letto:

   uint8 -> float [0,1]
*/
int gs_tensor_u8_to_float01(
    const uint8_t *input_u8,
    const GSTensorSpec *spec,
    float *output_f32
);


/*
   Legge direttamente un .bin come float [0,1].
*/
int gs_tensor_read_float01(
    const char *path,
    const GSTensorSpec *spec,
    float *out_f32
);

#endif