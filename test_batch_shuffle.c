#include "vae_loader.h"
#include "vae_dataset.h"
#include "vae_batch.h"
#include "vae_shuffle.h"

#include <stdio.h>
#include <stdlib.h>


int main(int argc, char **argv)
{
    const char *manifest_path;
    size_t batch_size;
    unsigned int seed;

    GSManifest manifest;
    GSBatch batch;

    size_t *indices;
    size_t i;
    int rc;

    /*
       Uso:
       test_batch_shuffle.exe manifest.csv batch_size seed

       Esempio:
       test_batch_shuffle.exe manifest_tensors_224_central.csv 4 123
    */
    manifest_path = "manifest_tensors_224_central.csv";
    batch_size = 4;
    seed = 123;

    indices = NULL;

    if (argc >= 2) {
        manifest_path = argv[1];
    }

    if (argc >= 3) {
        batch_size = (size_t)strtoul(argv[2], NULL, 10);
    }

    if (argc >= 4) {
        seed = (unsigned int)strtoul(argv[3], NULL, 10);
    }

    printf("Manifest:   %s\n", manifest_path);
    printf("Batch size: %lu\n", (unsigned long)batch_size);
    printf("Seed:       %u\n\n", seed);

    rc = gs_manifest_load(manifest_path, &manifest);
    if (rc != 0) {
        return 1;
    }

    printf("Manifest caricato correttamente.\n");
    printf("Numero eventi nel manifest: %lu\n\n", (unsigned long)manifest.count);

    rc = gs_indices_alloc(&indices, manifest.count);
    if (rc != 0) {
        gs_manifest_free(&manifest);
        return 1;
    }

    gs_indices_fill_sequential(indices, manifest.count);

    printf("Primi 10 indici prima dello shuffle:\n");
    for (i = 0; i < 10 && i < manifest.count; i++) {
        const GSManifestRecord *record;

        record = gs_manifest_get(&manifest, indices[i]);

        printf("  pos=%lu -> manifest_index=%lu, label=%s, event_id=%s\n",
               (unsigned long)i,
               (unsigned long)indices[i],
               record->label,
               record->event_id);
    }

    printf("\n");

    gs_indices_shuffle(indices, manifest.count, seed);

    printf("Primi 10 indici dopo lo shuffle:\n");
    for (i = 0; i < 10 && i < manifest.count; i++) {
        const GSManifestRecord *record;

        record = gs_manifest_get(&manifest, indices[i]);

        printf("  pos=%lu -> manifest_index=%lu, label=%s, event_id=%s\n",
               (unsigned long)i,
               (unsigned long)indices[i],
               record->label,
               record->event_id);
    }

    printf("\n");

    rc = gs_batch_alloc(&batch, batch_size);
    if (rc != 0) {
        free(indices);
        gs_manifest_free(&manifest);
        return 1;
    }

    /*
       Carichiamo il primo batch usando gli indici mescolati.
       start_pos = 0.
    */
    rc = gs_batch_load_from_indices(
        &manifest,
        indices,
        manifest.count,
        0,
        batch_size,
        &batch
    );

    if (rc != 0) {
        gs_batch_free(&batch);
        free(indices);
        gs_manifest_free(&manifest);
        return 1;
    }

    printf("Batch caricato usando indici shuffled:\n\n");
    gs_batch_print_summary(&batch);

    gs_batch_free(&batch);
    free(indices);
    gs_manifest_free(&manifest);

    return 0;
}