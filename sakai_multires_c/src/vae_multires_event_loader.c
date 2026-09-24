#include "vae_multires_event_loader.h"
#include "vae_resize_u8.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


static void gs_loader_zero(GSMultiresEventLoader *loader)
{
    if (loader != NULL) {
        memset(loader, 0, sizeof(*loader));
    }
}


static int gs_loader_ready(const GSMultiresEventLoader *loader)
{
    return loader != NULL &&
           loader->initialized &&
           loader->crop224_u8 != NULL &&
           loader->target_u8 != NULL &&
           gs_tensor_spec_is_valid(&loader->source_spec) &&
           gs_tensor_spec_is_valid(&loader->target_spec);
}


int gs_multires_event_loader_init(
    GSMultiresEventLoader *loader,
    int target_side
)
{
    if (loader == NULL) {
        return 1;
    }

    gs_loader_zero(loader);

    if (gs_tensor_spec_init(&loader->source_spec, 224) != 0 ||
        gs_tensor_spec_init(&loader->target_spec, target_side) != 0) {
        return 1;
    }

    loader->crop224_u8 = (uint8_t *)malloc(loader->source_spec.bytes_u8);
    loader->target_u8 = (uint8_t *)malloc(loader->target_spec.bytes_u8);

    if (loader->crop224_u8 == NULL || loader->target_u8 == NULL) {
        gs_multires_event_loader_free(loader);
        return 1;
    }

    loader->initialized = 1;
    return 0;
}


void gs_multires_event_loader_free(GSMultiresEventLoader *loader)
{
    if (loader == NULL) {
        return;
    }

    free(loader->crop224_u8);
    free(loader->target_u8);
    gs_loader_zero(loader);
}


const GSTensorSpec *gs_multires_event_loader_spec(
    const GSMultiresEventLoader *loader
)
{
    return gs_loader_ready(loader) ? &loader->target_spec : NULL;
}


int gs_multires_event_load_crop_u8(
    GSMultiresEventLoader *loader,
    const GSEventRecord *record,
    int crop_left,
    uint8_t *out_u8
)
{
    if (!gs_loader_ready(loader) || record == NULL || out_u8 == NULL) {
        return 1;
    }

    if (gs_event_load_crop_u8(record, crop_left, loader->crop224_u8) != 0) {
        return 1;
    }

    return gs_resize_tensor_u8(
        loader->crop224_u8,
        &loader->source_spec,
        out_u8,
        &loader->target_spec
    );
}


int gs_multires_event_load_crop_float01(
    GSMultiresEventLoader *loader,
    const GSEventRecord *record,
    int crop_left,
    float *out_f32
)
{
    size_t i;

    if (!gs_loader_ready(loader) || out_f32 == NULL) {
        return 1;
    }

    if (gs_multires_event_load_crop_u8(
            loader,
            record,
            crop_left,
            loader->target_u8
        ) != 0) {
        return 1;
    }

    for (i = 0; i < loader->target_spec.values_per_event; i++) {
        out_f32[i] = (float)loader->target_u8[i] / 255.0f;
    }

    return 0;
}


int gs_multires_event_load_central_u8(
    GSMultiresEventLoader *loader,
    const GSEventRecord *record,
    uint8_t *out_u8
)
{
    return gs_multires_event_load_crop_u8(
        loader,
        record,
        GS_SAKAI_CENTRAL_CROP_LEFT,
        out_u8
    );
}


int gs_multires_event_load_central_float01(
    GSMultiresEventLoader *loader,
    const GSEventRecord *record,
    float *out_f32
)
{
    return gs_multires_event_load_crop_float01(
        loader,
        record,
        GS_SAKAI_CENTRAL_CROP_LEFT,
        out_f32
    );
}


int gs_multires_event_load_random_float01_mode(
    GSMultiresEventLoader *loader,
    const GSEventRecord *record,
    GSAugmentRng *rng,
    GSAugmentMode mode,
    int *used_crop_left,
    float *out_f32
)
{
    int crop_left;

    if (rng == NULL || used_crop_left == NULL) {
        return 1;
    }

    if (gs_augment_random_crop_left_mode(rng, mode, &crop_left) != 0) {
        return 1;
    }

    if (gs_multires_event_load_crop_float01(
            loader,
            record,
            crop_left,
            out_f32
        ) != 0) {
        return 1;
    }

    *used_crop_left = crop_left;
    return 0;
}


int gs_multires_event_load_random_float01(
    GSMultiresEventLoader *loader,
    const GSEventRecord *record,
    GSAugmentRng *rng,
    int *used_crop_left,
    float *out_f32
)
{
    return gs_multires_event_load_random_float01_mode(
        loader,
        record,
        rng,
        GS_AUGMENT_MODE_LEGACY_TEXT,
        used_crop_left,
        out_f32
    );
}
