#include "vae_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>


float *gs_alloc_tensor_f32(void)
{
    /*
       calloc inizializza a zero.
       Il tensore contiene 4 * 224 * 224 float.
    */
    return (float *)calloc(GS_TENSOR_SIZE, sizeof(float));
}


const char *gs_channel_name(int c)
{
    if (c == 0) return "0.5 s";
    if (c == 1) return "1.0 s";
    if (c == 2) return "2.0 s";
    if (c == 3) return "4.0 s";
    return "canale sconosciuto";
}


int gs_load_bin_u8_as_float01(const char *bin_path, float *out)
{
    FILE *fp;
    uint8_t *tmp;
    size_t nread;
    int extra;
    size_t i;

    if (bin_path == NULL) {
        printf("Errore: bin_path NULL.\n");
        return 1;
    }

    if (out == NULL) {
        printf("Errore: buffer di output NULL.\n");
        return 1;
    }

    fp = fopen(bin_path, "rb");
    if (fp == NULL) {
        printf("Errore: impossibile aprire il file:\n%s\n", bin_path);
        return 1;
    }

    tmp = (uint8_t *)malloc(GS_TENSOR_SIZE * sizeof(uint8_t));
    if (tmp == NULL) {
        printf("Errore: memoria insufficiente per buffer uint8.\n");
        fclose(fp);
        return 1;
    }

    /*
       Ogni file deve contenere esattamente:
       4 * 224 * 224 = 200704 byte
    */
    nread = fread(tmp, sizeof(uint8_t), GS_TENSOR_SIZE, fp);

    if (nread != GS_TENSOR_SIZE) {
        printf("Errore: dimensione file non valida.\n");
        printf("Attesi: %lu byte\n", (unsigned long)GS_TENSOR_SIZE);
        printf("Letti:  %lu byte\n", (unsigned long)nread);

        free(tmp);
        fclose(fp);
        return 1;
    }

    /*
       Controllo che il file non sia più grande del previsto.
       Se c'è ancora un byte leggibile, il .bin non è nel formato atteso.
    */
    extra = fgetc(fp);
    if (extra != EOF) {
        printf("Errore: il file contiene più dati del previsto.\n");
        printf("Formato atteso: uint8 4x224x224, channel-major.\n");

        free(tmp);
        fclose(fp);
        return 1;
    }

    /*
       Normalizzazione per il VAE:
       uint8 0..255 -> float 0..1
    */
    for (i = 0; i < GS_TENSOR_SIZE; i++) {
        out[i] = (float)tmp[i] / 255.0f;
    }

    free(tmp);
    fclose(fp);

    return 0;
}