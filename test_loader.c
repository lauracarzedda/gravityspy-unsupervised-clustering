#include "vae_loader.h"

#include <stdio.h>
#include <stdlib.h>


int main(int argc, char **argv)
{
    float *x;
    int rc;
    int c;
    size_t i;

    if (argc < 2) {
        printf("Uso:\n");
        printf("  test_loader.exe percorso_file.bin\n");
        return 1;
    }

    x = gs_alloc_tensor_f32();
    if (x == NULL) {
        printf("Errore: impossibile allocare tensore float.\n");
        return 1;
    }

    rc = gs_load_bin_u8_as_float01(argv[1], x);
    if (rc != 0) {
        free(x);
        return 1;
    }

    printf("Tensor caricato correttamente.\n");
    printf("Shape logica: %d x %d x %d\n", GS_CHANNELS, GS_HEIGHT, GS_WIDTH);
    printf("Layout: channel-major\n");
    printf("Valori normalizzati in [0,1]\n\n");

    for (c = 0; c < GS_CHANNELS; c++) {
        float minv = 999.0f;
        float maxv = -999.0f;
        double sum = 0.0;

        size_t offset = (size_t)c * GS_PIXELS_PER_CHANNEL;

        for (i = 0; i < GS_PIXELS_PER_CHANNEL; i++) {
            float v = x[offset + i];

            if (v < minv) minv = v;
            if (v > maxv) maxv = v;
            sum += (double)v;
        }

        printf("Canale %d (%s): min = %.6f, max = %.6f, mean = %.6f\n",
               c,
               gs_channel_name(c),
               minv,
               maxv,
               sum / (double)GS_PIXELS_PER_CHANNEL);
    }

    free(x);
    return 0;
}