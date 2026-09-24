#include "vae_tensor_spec.h"
#include "vae_resize_u8.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>


static int read_exact_u8(
    const char *path,
    uint8_t *buffer,
    size_t expected_size
)
{
    FILE *fp;
    size_t nread;
    int extra;

    fp = fopen(path, "rb");

    if (fp == NULL) {
        printf("Errore: impossibile aprire %s\n", path);
        return 1;
    }

    nread = fread(
        buffer,
        1,
        expected_size,
        fp
    );

    extra = fgetc(fp);

    fclose(fp);

    if (nread != expected_size || extra != EOF) {
        printf("Errore: dimensione non valida per %s\n",
               path);

        printf("Attesi %lu byte, letti %lu.\n",
               (unsigned long)expected_size,
               (unsigned long)nread);

        return 1;
    }

    return 0;
}


static size_t compare_u8(
    const uint8_t *a,
    const uint8_t *b,
    size_t count,
    int *max_difference
)
{
    size_t i;
    size_t different;

    different = 0;
    *max_difference = 0;

    for (i = 0; i < count; i++) {
        int d;

        d = (int)a[i] - (int)b[i];

        if (d < 0) {
            d = -d;
        }

        if (d != 0) {
            different++;
        }

        if (d > *max_difference) {
            *max_difference = d;
        }
    }

    return different;
}


int main(void)
{
    GSTensorSpec spec224;
    GSTensorSpec spec112;
    GSTensorSpec spec75;

    uint8_t *src224;
    uint8_t *generated112;
    uint8_t *generated75;
    uint8_t *saved112;
    uint8_t *saved75;

    size_t diff112;
    size_t diff75;

    int max112;
    int max75;
    int rc;

    if (gs_tensor_spec_init(&spec224, 224) != 0 ||
        gs_tensor_spec_init(&spec112, 112) != 0 ||
        gs_tensor_spec_init(&spec75, 75) != 0) {

        return 1;
    }

    src224 = (uint8_t *)malloc(spec224.bytes_u8);
    generated112 = (uint8_t *)malloc(spec112.bytes_u8);
    generated75 = (uint8_t *)malloc(spec75.bytes_u8);
    saved112 = (uint8_t *)malloc(spec112.bytes_u8);
    saved75 = (uint8_t *)malloc(spec75.bytes_u8);

    if (src224 == NULL ||
        generated112 == NULL ||
        generated75 == NULL ||
        saved112 == NULL ||
        saved75 == NULL) {

        printf("Errore: memoria insufficiente.\n");

        free(src224);
        free(generated112);
        free(generated75);
        free(saved112);
        free(saved75);

        return 1;
    }

    rc = read_exact_u8(
        "tensors_224_central_uint8/event_000001.bin",
        src224,
        spec224.bytes_u8
    );

    if (rc == 0) {
        rc = read_exact_u8(
            "tensors_112_central_uint8/event_000001.bin",
            saved112,
            spec112.bytes_u8
        );
    }

    if (rc == 0) {
        rc = read_exact_u8(
            "tensors_75_central_uint8/event_000001.bin",
            saved75,
            spec75.bytes_u8
        );
    }

    if (rc != 0) {
        free(src224);
        free(generated112);
        free(generated75);
        free(saved112);
        free(saved75);

        return 1;
    }

    rc = gs_resize_tensor_u8(
        src224,
        &spec224,
        generated112,
        &spec112
    );

    if (rc == 0) {
        rc = gs_resize_tensor_u8(
            src224,
            &spec224,
            generated75,
            &spec75
        );
    }

    if (rc != 0) {
        free(src224);
        free(generated112);
        free(generated75);
        free(saved112);
        free(saved75);

        return 1;
    }

    diff112 = compare_u8(
        generated112,
        saved112,
        spec112.values_per_event,
        &max112
    );

    diff75 = compare_u8(
        generated75,
        saved75,
        spec75.values_per_event,
        &max75
    );

    printf("Confronto resize centrale con i .bin ufficiali:\n");

    printf(
        "112x112: byte diversi = %lu, differenza massima = %d\n",
        (unsigned long)diff112,
        max112
    );

    printf(
        "75x75:   byte diversi = %lu, differenza massima = %d\n",
        (unsigned long)diff75,
        max75
    );

    free(src224);
    free(generated112);
    free(generated75);
    free(saved112);
    free(saved75);

    if (diff112 != 0 || diff75 != 0) {
        printf(
            "ERRORE: il resize non coincide "
            "con i dataset ridotti.\n"
        );

        return 1;
    }

    printf(
        "OK: resize multirisoluzione identico "
        "ai .bin centrali 112 e 75.\n"
    );

    return 0;
}