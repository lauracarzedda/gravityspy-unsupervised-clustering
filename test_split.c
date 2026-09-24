#include "vae_loader.h"
#include "vae_dataset.h"
#include "vae_batch.h"
#include "vae_shuffle.h"
#include "vae_split.h"

#include <stdio.h>
#include <stdlib.h>


int main(int argc, char **argv)
{
    const char *manifest_path;
    double train_rate;
    size_t batch_size;
    unsigned int seed;

    GSManifest manifest;
    GSSplit split;
    GSBatch batch;

    int rc;

    /*
       Uso:
       test_split.exe manifest.csv train_rate batch_size seed

       Esempio:
       test_split.exe manifest_tensors_224_central.csv 0.8 4 123
    */
    manifest_path = "manifest_tensors_224_central.csv";
    train_rate = 0.8;
    batch_size = 4;
    seed = 123;

    if (argc >= 2) {
        manifest_path = argv[1];
    }

    if (argc >= 3) {
        train_rate = atof(argv[2]);
    }

    if (argc >= 4) {
        batch_size = (size_t)strtoul(argv[3], NULL, 10);
    }

    if (argc >= 5) {
        seed = (unsigned int)strtoul(argv[4], NULL, 10);
    }

    printf("Manifest:    %s\n", manifest_path);
    printf("Train rate:  %.3f\n", train_rate);
    printf("Batch size:  %lu\n", (unsigned long)batch_size);
    printf("Seed:        %u\n\n", seed);

    rc = gs_manifest_load(manifest_path, &manifest);
    if (rc != 0) {
        return 1;
    }

    printf("Manifest caricato correttamente.\n");
    printf("Numero eventi nel manifest: %lu\n\n", (unsigned long)manifest.count);

    rc = gs_split_train_val(&manifest, train_rate, seed, &split);
    if (rc != 0) {
        gs_manifest_free(&manifest);
        return 1;
    }

    gs_split_print_summary(&manifest, &split, 5);

    printf("\n");

    rc = gs_batch_alloc(&batch, batch_size);
    if (rc != 0) {
        gs_split_free(&split);
        gs_manifest_free(&manifest);
        return 1;
    }

    /*
       Test: primo batch dal train set.
       Le label sono stampate solo per controllo.
       Il futuro VAE userà solo batch.x.
    */
    printf("=== Primo batch TRAIN ===\n\n");

    rc = gs_batch_load_from_indices(
        &manifest,
        split.train_indices,
        split.train_count,
        0,
        batch_size,
        &batch
    );

    if (rc != 0) {
        gs_batch_free(&batch);
        gs_split_free(&split);
        gs_manifest_free(&manifest);
        return 1;
    }

    gs_batch_print_summary(&batch);

    printf("\n");

    /*
       Test: primo batch dal validation set.
    */
    printf("=== Primo batch VALIDATION ===\n\n");

    rc = gs_batch_load_from_indices(
        &manifest,
        split.val_indices,
        split.val_count,
        0,
        batch_size,
        &batch
    );

    if (rc != 0) {
        gs_batch_free(&batch);
        gs_split_free(&split);
        gs_manifest_free(&manifest);
        return 1;
    }

    gs_batch_print_summary(&batch);

    gs_batch_free(&batch);
    gs_split_free(&split);
    gs_manifest_free(&manifest);

    return 0;
}