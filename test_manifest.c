#include "vae_dataset.h"
#include "vae_loader.h"

#include <stdio.h>
#include <stdlib.h>


int main(int argc, char **argv)
{
    const char *manifest_path;
    GSManifest manifest;
    float *x;
    size_t i;
    int rc;

    float minv;
    float maxv;
    double sum;

    if (argc < 2) {
        manifest_path = "manifest_tensors_224_central.csv";
    } else {
        manifest_path = argv[1];
    }

    rc = gs_manifest_load(manifest_path, &manifest);
    if (rc != 0) {
        return 1;
    }

    printf("Manifest caricato correttamente.\n");
    printf("Numero eventi: %lu\n\n", (unsigned long)manifest.count);

    printf("Primi 3 record:\n\n");

    for (i = 0; i < 3 && i < manifest.count; i++) {
        printf("---- Record %lu ----\n", (unsigned long)i);
        gs_manifest_print_record(&manifest.records[i]);
        printf("\n");
    }

    /*
       Ora proviamo a caricare il primo tensore indicato dal manifest.
    */
    x = gs_alloc_tensor_f32();
    if (x == NULL) {
        printf("Errore: impossibile allocare tensore float.\n");
        gs_manifest_free(&manifest);
        return 1;
    }

    rc = gs_manifest_load_tensor(&manifest, 0, x);
    if (rc != 0) {
        free(x);
        gs_manifest_free(&manifest);
        return 1;
    }

    minv = 999.0f;
    maxv = -999.0f;
    sum = 0.0;

    for (i = 0; i < GS_TENSOR_SIZE; i++) {
        float v = x[i];

        if (v < minv) minv = v;
        if (v > maxv) maxv = v;

        sum += (double)v;
    }

    printf("Primo tensore caricato dal manifest.\n");
    printf("Shape logica: %d x %d x %d\n", GS_CHANNELS, GS_HEIGHT, GS_WIDTH);
    printf("Valori: min = %.6f, max = %.6f, mean = %.6f\n",
           minv,
           maxv,
           sum / (double)GS_TENSOR_SIZE);

    free(x);
    gs_manifest_free(&manifest);

    return 0;
}