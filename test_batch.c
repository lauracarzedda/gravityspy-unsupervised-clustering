#include "vae_loader.h"
#include "vae_dataset.h"
#include "vae_batch.h"

#include <stdio.h>
#include <stdlib.h>


int main(int argc, char **argv)
{
    const char *manifest_path;
    size_t batch_size;
    size_t start_index;

    GSManifest manifest;
    GSBatch batch;

    int rc;

    /*
       Uso:
       test_batch.exe manifest.csv batch_size start_index

       Esempio:
       test_batch.exe manifest_tensors_224_central.csv 4 0
    */
    manifest_path = "manifest_tensors_224_central.csv";
    batch_size = 4;
    start_index = 0;

    if (argc >= 2) {
        manifest_path = argv[1];
    }

    if (argc >= 3) {
        batch_size = (size_t)strtoul(argv[2], NULL, 10);
    }

    if (argc >= 4) {
        start_index = (size_t)strtoul(argv[3], NULL, 10);
    }

    printf("Manifest:    %s\n", manifest_path);
    printf("Batch size:  %lu\n", (unsigned long)batch_size);
    printf("Start index: %lu\n\n", (unsigned long)start_index);

    rc = gs_manifest_load(manifest_path, &manifest);
    if (rc != 0) {
        return 1;
    }

    printf("Manifest caricato correttamente.\n");
    printf("Numero eventi nel manifest: %lu\n\n", (unsigned long)manifest.count);

    rc = gs_batch_alloc(&batch, batch_size);
    if (rc != 0) {
        gs_manifest_free(&manifest);
        return 1;
    }

    rc = gs_batch_load_sequential(&manifest, start_index, batch_size, &batch);
    if (rc != 0) {
        gs_batch_free(&batch);
        gs_manifest_free(&manifest);
        return 1;
    }

    gs_batch_print_summary(&batch);

    gs_batch_free(&batch);
    gs_manifest_free(&manifest);

    return 0;
}