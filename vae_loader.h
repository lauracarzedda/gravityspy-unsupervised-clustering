#ifndef VAE_LOADER_H
#define VAE_LOADER_H

#include <stddef.h>

#define GS_CHANNELS 4
#define GS_HEIGHT   224
#define GS_WIDTH    224

#define GS_PIXELS_PER_CHANNEL ((size_t)GS_HEIGHT * (size_t)GS_WIDTH)
#define GS_TENSOR_SIZE        ((size_t)GS_CHANNELS * GS_PIXELS_PER_CHANNEL)

/*
   Alloca un tensore float32 normalizzato per il VAE:
   shape logica: 4 x 224 x 224
   layout: channel-major
*/
float *gs_alloc_tensor_f32(void);

/*
   Legge un file .bin uint8 4x224x224 e lo converte in float [0,1].

   Parametri:
   - bin_path: percorso del file .bin
   - out: array float già allocato, lungo GS_TENSOR_SIZE

   Ritorna:
   - 0 se tutto va bene
   - diverso da 0 se c'è errore
*/
int gs_load_bin_u8_as_float01(const char *bin_path, float *out);

/*
   Nome leggibile del canale temporale.
*/
const char *gs_channel_name(int c);

#endif