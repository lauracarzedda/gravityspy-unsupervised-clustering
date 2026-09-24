#ifndef VAE_EVENT_LOADER_H
#define VAE_EVENT_LOADER_H

#include <stddef.h>
#include <stdint.h>

#include "vae_event_dataset.h"
#include "vae_loader.h"


/*
   Geometria del crop temporale:

   sorgente: 272 pixel
   output:   224 pixel
   margine totale disponibile: 48 pixel
*/
#define GS_EVENT_MIN_CROP_LEFT 0
#define GS_EVENT_MAX_CROP_LEFT \
    (GS_EVENT_SOURCE_WIDTH - GS_WIDTH)


/*
   Crop centrale:

   (272 - 224) / 2 = 24
*/
#define GS_SAKAI_CENTRAL_CROP_LEFT \
    (GS_EVENT_MAX_CROP_LEFT / 2)


/*
   Intervallo legacy usato attualmente dal generatore casuale:

       crop_left = 0 ... 24

   Queste costanti restano separate dall'intervallo geometrico
   completo 0 ... 48.
*/
#define GS_SAKAI_MIN_CROP_LEFT GS_EVENT_MIN_CROP_LEFT
#define GS_SAKAI_MAX_CROP_LEFT GS_SAKAI_CENTRAL_CROP_LEFT


/*
   Controlla che crop_left appartenga all'intervallo
   geometricamente valido 0-48.
*/
int gs_event_crop_left_is_valid(int crop_left);


/*
   Carica le quattro PNG 272x224 di un evento,
   applica lo stesso crop_left a tutte le viste
   e produce un tensore uint8:

       4 x 224 x 224

   Layout:
       channel-major

   out_u8 deve avere GS_TENSOR_SIZE elementi.
*/
int gs_event_load_crop_u8(
    const GSEventRecord *record,
    int crop_left,
    uint8_t *out_u8
);


/*
   Come sopra, ma converte anche:

       uint8 0-255
       ->
       float [0,1]

   out_f32 deve avere GS_TENSOR_SIZE elementi.
*/
int gs_event_load_crop_float01(
    const GSEventRecord *record,
    int crop_left,
    float *out_f32
);


/*
   Funzioni comode per il crop centrale fisso:
       crop_left = 24
*/
int gs_event_load_central_u8(
    const GSEventRecord *record,
    uint8_t *out_u8
);

int gs_event_load_central_float01(
    const GSEventRecord *record,
    float *out_f32
);

#endif