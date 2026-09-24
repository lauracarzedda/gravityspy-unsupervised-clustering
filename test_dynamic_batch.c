#include "vae_dynamic_batch.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


static int parse_mode(const char *text, GSAugmentMode *mode)
{
    if (strcmp(text, "legacy_text") == 0) {
        *mode = GS_AUGMENT_MODE_LEGACY_TEXT;
        return 0;
    }

    if (strcmp(text, "symmetric_figure") == 0) {
        *mode = GS_AUGMENT_MODE_SYMMETRIC_FIGURE;
        return 0;
    }

    return 1;
}


int main(int argc, char **argv)
{
    const char *manifest_path = "manifest_events_272.csv";
    const char *train_path = "split_train_indices_seed123.csv";
    const char *val_path = "split_val_indices_seed123.csv";
    size_t batch_size = 4;
    unsigned int seed = 456u;
    GSAugmentMode mode = GS_AUGMENT_MODE_LEGACY_TEXT;

    GSEventManifest manifest = {0};
    GSSavedSplit train = {0};
    GSSavedSplit val = {0};
    GSDynamicBatch batch = {0};
    GSAugmentRng rng;

    int min_crop;
    int max_crop;
    size_t i;
    int rc = 1;

    if (argc >= 2) manifest_path = argv[1];
    if (argc >= 3) train_path = argv[2];
    if (argc >= 4) val_path = argv[3];
    if (argc >= 5) batch_size = (size_t)strtoul(argv[4], NULL, 10);
    if (argc >= 6) seed = (unsigned int)strtoul(argv[5], NULL, 10);

    if (argc >= 7 && parse_mode(argv[6], &mode) != 0) {
        printf("Modo non valido. Usa legacy_text o symmetric_figure.\n");
        return 1;
    }

    if (batch_size == 0 ||
        gs_augment_mode_crop_range(mode, &min_crop, &max_crop) != 0) {
        return 1;
    }

    printf("Modo train: %s (%d-%d)\n",
           gs_augment_mode_name(mode), min_crop, max_crop);

    if (gs_event_manifest_load(manifest_path, &manifest) != 0) goto cleanup;
    if (gs_saved_split_load(train_path, "train", &manifest, &train) != 0) goto cleanup;
    if (gs_saved_split_load(val_path, "validation", &manifest, &val) != 0) goto cleanup;
    if (gs_dynamic_batch_alloc(&batch, batch_size) != 0) goto cleanup;

    gs_augment_rng_seed(&rng, seed);

    if (gs_dynamic_batch_load_train_mode(
            &manifest, &train, 0, batch_size,
            &rng, mode, &batch
        ) != 0) goto cleanup;

    for (i = 0; i < batch.count; i++) {
        if (batch.crop_lefts[i] < min_crop ||
            batch.crop_lefts[i] > max_crop) {

            printf("Errore: crop train fuori intervallo.\n");
            goto cleanup;
        }
    }

    gs_dynamic_batch_print_summary("TRAIN", &batch);

    if (gs_dynamic_batch_load_validation(
            &manifest, &val, 0, batch_size, &batch
        ) != 0) goto cleanup;

    for (i = 0; i < batch.count; i++) {
        if (batch.crop_lefts[i] != GS_SAKAI_CENTRAL_CROP_LEFT) {
            printf("Errore: validation non centrale.\n");
            goto cleanup;
        }
    }

    gs_dynamic_batch_print_summary("VALIDATION", &batch);
    printf("OK: train=%s, validation=crop_left_24.\n",
           gs_augment_mode_name(mode));

    rc = 0;

cleanup:
    gs_dynamic_batch_free(&batch);
    gs_saved_split_free(&val);
    gs_saved_split_free(&train);
    gs_event_manifest_free(&manifest);
    return rc;
}
