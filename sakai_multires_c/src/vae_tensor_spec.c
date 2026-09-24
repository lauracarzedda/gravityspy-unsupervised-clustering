#include "vae_tensor_spec.h"

#include <stdio.h>


int gs_tensor_spec_init(GSTensorSpec *spec, int side)
{
    if (spec == NULL) {
        printf("Errore: spec NULL.\n");
        return 1;
    }

    if (side != 75 && side != 112 && side != 224) {
        printf("Errore: risoluzione non supportata: %d.\n", side);
        printf("Risoluzioni supportate: 75, 112, 224.\n");
        return 1;
    }

    spec->channels = GS_MULTI_CHANNELS;
    spec->height = side;
    spec->width = side;

    spec->pixels_per_channel =
        (size_t)spec->height * (size_t)spec->width;

    spec->values_per_event =
        (size_t)spec->channels * spec->pixels_per_channel;

    spec->bytes_u8 = spec->values_per_event;

    spec->bytes_f32 =
        spec->values_per_event * sizeof(float);

    return 0;
}


int gs_tensor_spec_is_valid(const GSTensorSpec *spec)
{
    if (spec == NULL) {
        return 0;
    }

    if (spec->channels != GS_MULTI_CHANNELS) {
        return 0;
    }

    if (spec->height <= 0 || spec->width <= 0) {
        return 0;
    }

    if (spec->height != spec->width) {
        return 0;
    }

    if (spec->height != 75 &&
        spec->height != 112 &&
        spec->height != 224) {

        return 0;
    }

    if (spec->pixels_per_channel !=
        (size_t)spec->height * (size_t)spec->width) {

        return 0;
    }

    if (spec->values_per_event !=
        (size_t)spec->channels *
        spec->pixels_per_channel) {

        return 0;
    }

    return 1;
}


const char *gs_tensor_spec_name(const GSTensorSpec *spec)
{
    if (!gs_tensor_spec_is_valid(spec)) {
        return "invalid";
    }

    if (spec->height == 75) {
        return "mini75";
    }

    if (spec->height == 112) {
        return "mini112";
    }

    return "224";
}


void gs_tensor_spec_print(const GSTensorSpec *spec)
{
    if (!gs_tensor_spec_is_valid(spec)) {
        printf("Tensor spec non valida.\n");
        return;
    }

    printf("Profilo: %s\n",
           gs_tensor_spec_name(spec));

    printf("Shape:   %d x %d x %d\n",
           spec->channels,
           spec->height,
           spec->width);

    printf("Valori per evento: %lu\n",
           (unsigned long)spec->values_per_event);

    printf("Byte uint8:        %lu\n",
           (unsigned long)spec->bytes_u8);

    printf("Byte float32:      %lu\n",
           (unsigned long)spec->bytes_f32);
}