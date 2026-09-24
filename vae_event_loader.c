#include "vae_event_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/*
   L'implementazione di stb_image deve apparire
   in un solo file .c del programma finale.
*/
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


int gs_event_crop_left_is_valid(int crop_left)
{
    if (crop_left < GS_EVENT_MIN_CROP_LEFT) {
        return 0;
    }

    if (crop_left > GS_EVENT_MAX_CROP_LEFT) {
        return 0;
    }

    /*
       Controllo generale aggiuntivo:
       il crop non deve oltrepassare la larghezza sorgente.
    */
    if (crop_left + GS_WIDTH > GS_EVENT_SOURCE_WIDTH) {
        return 0;
    }

    return 1;
}


int gs_event_load_crop_u8(
    const GSEventRecord *record,
    int crop_left,
    uint8_t *out_u8
)
{
    size_t view;

    if (record == NULL) {
        printf("Errore: record evento NULL.\n");
        return 1;
    }

    if (out_u8 == NULL) {
        printf("Errore: buffer output uint8 NULL.\n");
        return 1;
    }

    if (!gs_event_crop_left_is_valid(crop_left)) {
        printf("Errore: crop_left non valido.\n");
        printf("crop_left trovato: %d\n", crop_left);
        printf("Intervallo geometrico atteso: %d-%d\n",
               GS_EVENT_MIN_CROP_LEFT,
               GS_EVENT_MAX_CROP_LEFT);
        return 1;
    }

    /*
       Carica una vista temporale alla volta.

       L'output finale è channel-major:
       prima tutto il canale 0,
       poi tutto il canale 1,
       poi il canale 2,
       poi il canale 3.
    */
    for (view = 0; view < GS_EVENT_VIEW_COUNT; view++) {
        const char *path;
        unsigned char *image;

        int width;
        int height;
        int original_channels;

        size_t y;
        size_t channel_offset;

        path = record->image_paths[view];

        /*
           Forziamo il caricamento a un solo canale grayscale.

           original_channels ci dice comunque quanti canali
           erano presenti realmente nel file PNG.
        */
        image = stbi_load(
            path,
            &width,
            &height,
            &original_channels,
            1
        );

        if (image == NULL) {
            const char *reason;

            reason = stbi_failure_reason();

            printf("Errore: impossibile caricare la PNG:\n");
            printf("%s\n", path);
            printf("Motivo STB: %s\n",
                   reason != NULL ? reason : "sconosciuto");

            return 1;
        }

        if (width != GS_EVENT_SOURCE_WIDTH ||
            height != GS_EVENT_SOURCE_HEIGHT) {

            printf("Errore: dimensione PNG non valida.\n");
            printf("File: %s\n", path);
            printf("Trovata: %d x %d\n", width, height);
            printf("Attesa:  %d x %d\n",
                   GS_EVENT_SOURCE_WIDTH,
                   GS_EVENT_SOURCE_HEIGHT);

            stbi_image_free(image);
            return 1;
        }

        /*
           Le PNG create nel preprocessing ufficiale
           devono essere monocanale.
        */
        if (original_channels != 1) {
            printf("Errore: PNG non monocanale.\n");
            printf("File: %s\n", path);
            printf("Canali originali trovati: %d\n",
                   original_channels);

            stbi_image_free(image);
            return 1;
        }

        channel_offset = view * GS_PIXELS_PER_CHANNEL;

        /*
           Per ogni riga:
           copia 224 pixel partendo dalla colonna crop_left.
        */
        for (y = 0; y < GS_HEIGHT; y++) {
            const unsigned char *src_row;
            uint8_t *dst_row;

            src_row =
                image +
                y * GS_EVENT_SOURCE_WIDTH +
                (size_t)crop_left;

            dst_row =
                out_u8 +
                channel_offset +
                y * GS_WIDTH;

            memcpy(
                dst_row,
                src_row,
                GS_WIDTH * sizeof(uint8_t)
            );
        }

        stbi_image_free(image);
    }

    return 0;
}


int gs_event_load_crop_float01(
    const GSEventRecord *record,
    int crop_left,
    float *out_f32
)
{
    uint8_t *temporary_u8;
    size_t i;
    int rc;

    if (out_f32 == NULL) {
        printf("Errore: buffer output float NULL.\n");
        return 1;
    }

    temporary_u8 = (uint8_t *)malloc(
        GS_TENSOR_SIZE * sizeof(uint8_t)
    );

    if (temporary_u8 == NULL) {
        printf("Errore: memoria insufficiente per buffer temporaneo.\n");
        return 1;
    }

    rc = gs_event_load_crop_u8(
        record,
        crop_left,
        temporary_u8
    );

    if (rc != 0) {
        free(temporary_u8);
        return 1;
    }

    /*
       Normalizzazione pixel per pixel:
       0-255 -> 0.0-1.0
    */
    for (i = 0; i < GS_TENSOR_SIZE; i++) {
        out_f32[i] = (float)temporary_u8[i] / 255.0f;
    }

    free(temporary_u8);

    return 0;
}


int gs_event_load_central_u8(
    const GSEventRecord *record,
    uint8_t *out_u8
)
{
    return gs_event_load_crop_u8(
        record,
        GS_SAKAI_CENTRAL_CROP_LEFT,
        out_u8
    );
}


int gs_event_load_central_float01(
    const GSEventRecord *record,
    float *out_f32
)
{
    return gs_event_load_crop_float01(
        record,
        GS_SAKAI_CENTRAL_CROP_LEFT,
        out_f32
    );
}