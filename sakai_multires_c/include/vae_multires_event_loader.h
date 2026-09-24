#ifndef VAE_MULTIRES_EVENT_LOADER_H
#define VAE_MULTIRES_EVENT_LOADER_H

#include <stdint.h>

#include "vae_augmentation.h"
#include "vae_event_dataset.h"
#include "vae_tensor_spec.h"


typedef struct {
    GSTensorSpec source_spec;
    GSTensorSpec target_spec;
    uint8_t *crop224_u8;
    uint8_t *target_u8;
    int initialized;
} GSMultiresEventLoader;


int gs_multires_event_loader_init(
    GSMultiresEventLoader *loader,
    int target_side
);

void gs_multires_event_loader_free(
    GSMultiresEventLoader *loader
);

const GSTensorSpec *gs_multires_event_loader_spec(
    const GSMultiresEventLoader *loader
);

int gs_multires_event_load_crop_u8(
    GSMultiresEventLoader *loader,
    const GSEventRecord *record,
    int crop_left,
    uint8_t *out_u8
);

int gs_multires_event_load_crop_float01(
    GSMultiresEventLoader *loader,
    const GSEventRecord *record,
    int crop_left,
    float *out_f32
);

int gs_multires_event_load_central_u8(
    GSMultiresEventLoader *loader,
    const GSEventRecord *record,
    uint8_t *out_u8
);

int gs_multires_event_load_central_float01(
    GSMultiresEventLoader *loader,
    const GSEventRecord *record,
    float *out_f32
);

int gs_multires_event_load_random_float01_mode(
    GSMultiresEventLoader *loader,
    const GSEventRecord *record,
    GSAugmentRng *rng,
    GSAugmentMode mode,
    int *used_crop_left,
    float *out_f32
);

/* Compatibilita: usa legacy_text, crop_left 0..24. */
int gs_multires_event_load_random_float01(
    GSMultiresEventLoader *loader,
    const GSEventRecord *record,
    GSAugmentRng *rng,
    int *used_crop_left,
    float *out_f32
);

#endif
