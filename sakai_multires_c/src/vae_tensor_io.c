#include "vae_tensor_io.h"

#include <stdio.h>
#include <stdlib.h>


int gs_tensor_read_u8(
    const char *path,
    const GSTensorSpec *spec,
    uint8_t *out_u8
)
{
    FILE *fp;
    size_t bytes_read;
    int extra;

    if (path == NULL) {
        printf("Errore: path NULL.\n");
        return 1;
    }

    if (!gs_tensor_spec_is_valid(spec)) {
        printf("Errore: spec non valida.\n");
        return 1;
    }

    if (out_u8 == NULL) {
        printf("Errore: out_u8 NULL.\n");
        return 1;
    }

    fp = fopen(path, "rb");

    if (fp == NULL) {
        printf("Errore: impossibile aprire:\n%s\n",
               path);

        return 1;
    }

    bytes_read = fread(
        out_u8,
        1,
        spec->bytes_u8,
        fp
    );

    extra = fgetc(fp);

    fclose(fp);

    if (bytes_read != spec->bytes_u8) {
        printf("Errore: dimensione .bin non valida.\n");
        printf("File:   %s\n", path);
        printf("Attesi: %lu byte\n",
               (unsigned long)spec->bytes_u8);
        printf("Letti:  %lu byte\n",
               (unsigned long)bytes_read);

        return 1;
    }

    if (extra != EOF) {
        printf("Errore: il file contiene byte aggiuntivi.\n");
        printf("File: %s\n", path);

        return 1;
    }

    return 0;
}


int gs_tensor_u8_to_float01(
    const uint8_t *input_u8,
    const GSTensorSpec *spec,
    float *output_f32
)
{
    size_t i;

    if (input_u8 == NULL ||
        output_f32 == NULL) {

        printf("Errore: buffer NULL nella normalizzazione.\n");
        return 1;
    }

    if (!gs_tensor_spec_is_valid(spec)) {
        printf("Errore: spec non valida.\n");
        return 1;
    }

    for (i = 0;
         i < spec->values_per_event;
         i++) {

        output_f32[i] =
            (float)input_u8[i] / 255.0f;
    }

    return 0;
}


int gs_tensor_read_float01(
    const char *path,
    const GSTensorSpec *spec,
    float *out_f32
)
{
    uint8_t *temporary_u8;
    int rc;

    if (out_f32 == NULL) {
        printf("Errore: out_f32 NULL.\n");
        return 1;
    }

    if (!gs_tensor_spec_is_valid(spec)) {
        printf("Errore: spec non valida.\n");
        return 1;
    }

    temporary_u8 = (uint8_t *)malloc(
        spec->bytes_u8
    );

    if (temporary_u8 == NULL) {
        printf("Errore: memoria insufficiente.\n");
        return 1;
    }

    rc = gs_tensor_read_u8(
        path,
        spec,
        temporary_u8
    );

    if (rc == 0) {
        rc = gs_tensor_u8_to_float01(
            temporary_u8,
            spec,
            out_f32
        );
    }

    free(temporary_u8);

    return rc;
}