#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

/*
 * Preview di un tensore binario 4x224x224.
 *
 * Il file .bin contiene un evento completo Gravity Spy:
 *
 *   canale 0 -> vista temporale 0.5 s
 *   canale 1 -> vista temporale 1.0 s
 *   canale 2 -> vista temporale 2.0 s
 *   canale 3 -> vista temporale 4.0 s
 *
 * Ogni canale è una immagine 224x224 in scala di grigi.
 *
 * Il programma legge il file .bin e salva quattro PNG di controllo.
 */

#define NUM_CHANNELS 4
#define WIDTH 224
#define HEIGHT 224

#define TENSOR_SIZE (NUM_CHANNELS * WIDTH * HEIGHT)


int main(int argc, char **argv)
{
    const char *tensor_path;
    const char *duration_names[NUM_CHANNELS];

    FILE *f;
    unsigned char *tensor;
    size_t read_count;

    char output_path[512];

    int c;
    int ok;

    if (argc < 2) {
        printf("Uso:\n");
        printf("  05_preview_tensor_bin.exe tensors_224_central_uint8/event_000001.bin\n");
        return 1;
    }

    tensor_path = argv[1];

    duration_names[0] = "0_5";
    duration_names[1] = "1_0";
    duration_names[2] = "2_0";
    duration_names[3] = "4_0";

    tensor = (unsigned char *)malloc((size_t)TENSOR_SIZE);

    if (tensor == NULL) {
        printf("Errore: memoria insufficiente.\n");
        return 1;
    }

    f = fopen(tensor_path, "rb");

    if (f == NULL) {
        printf("Errore: impossibile aprire il file tensor:\n%s\n", tensor_path);
        free(tensor);
        return 1;
    }

    read_count = fread(tensor, 1, (size_t)TENSOR_SIZE, f);

    fclose(f);

    if (read_count != (size_t)TENSOR_SIZE) {
        printf("Errore: dimensione letta non corretta.\n");
        printf("Atteso: %d byte\n", TENSOR_SIZE);
        printf("Letto:   %lu byte\n", (unsigned long)read_count);
        free(tensor);
        return 1;
    }

    printf("Tensor letto correttamente.\n");
    printf("Dimensione: 4 x 224 x 224 = %d byte\n", TENSOR_SIZE);

    for (c = 0; c < NUM_CHANNELS; c++) {
        sprintf(
            output_path,
            "tensor_bin_preview/event_000001_channel_%s.png",
            duration_names[c]
        );

        /*
         * Layout del file:
         * canale 0: primi 224*224 byte
         * canale 1: successivi 224*224 byte
         * canale 2: successivi 224*224 byte
         * canale 3: successivi 224*224 byte
         */
        ok = stbi_write_png(
            output_path,
            WIDTH,
            HEIGHT,
            1,
            tensor + c * WIDTH * HEIGHT,
            WIDTH
        );

        if (!ok) {
            printf("Errore salvataggio: %s\n", output_path);
        } else {
            printf("Creato: %s\n", output_path);
        }
    }

    free(tensor);

    printf("\nPreview tensor completata.\n");

    return 0;
}