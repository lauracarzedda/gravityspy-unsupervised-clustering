#include "vae_dynamic_batch.h"

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define GS_CROP_COUNT (GS_EVENT_MAX_CROP_LEFT + 1)


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


static int run_train(
    const GSEventManifest *manifest,
    const GSSavedSplit *split,
    size_t batch_size,
    unsigned int seed,
    size_t max_batches,
    GSAugmentMode mode
)
{
    GSDynamicBatch batch = {0};
    GSAugmentRng rng;
    size_t crop_counts[GS_CROP_COUNT] = {0};
    size_t start = 0;
    size_t batches = 0;
    size_t events = 0;
    size_t values = 0;
    double sum = 0.0;
    float global_min = FLT_MAX;
    float global_max = -FLT_MAX;
    int min_crop;
    int max_crop;
    size_t i;

    if (gs_augment_mode_crop_range(mode, &min_crop, &max_crop) != 0 ||
        gs_dynamic_batch_alloc(&batch, batch_size) != 0) {
        return 1;
    }

    gs_augment_rng_seed(&rng, seed);

    while (start < split->count) {
        size_t current = split->count - start;
        GSDynamicBatchStats stats;

        if (current > batch_size) current = batch_size;

        if (gs_dynamic_batch_load_train_mode(
                manifest, split, start, current,
                &rng, mode, &batch
            ) != 0 ||
            gs_dynamic_batch_compute_stats(&batch, &stats) != 0) {

            gs_dynamic_batch_free(&batch);
            return 1;
        }

        if (stats.min_value < 0.0f || stats.max_value > 1.0f) {
            printf("Errore: train fuori [0,1].\n");
            gs_dynamic_batch_free(&batch);
            return 1;
        }

        if (stats.min_value < global_min) global_min = stats.min_value;
        if (stats.max_value > global_max) global_max = stats.max_value;

        sum += stats.mean_value * (double)(batch.count * GS_TENSOR_SIZE);
        values += batch.count * GS_TENSOR_SIZE;

        for (i = 0; i < batch.count; i++) {
            int crop = batch.crop_lefts[i];

            if (crop < min_crop || crop > max_crop) {
                printf("Errore: crop train %d fuori %d-%d.\n",
                       crop, min_crop, max_crop);
                gs_dynamic_batch_free(&batch);
                return 1;
            }

            crop_counts[crop]++;
        }

        events += batch.count;
        start += batch.count;
        batches++;

        if (max_batches > 0 && batches >= max_batches) break;
    }

    {
        size_t distinct = 0;

        for (i = (size_t)min_crop; i <= (size_t)max_crop; i++) {
            if (crop_counts[i] > 0) distinct++;
        }

        printf("TRAIN %s\n", gs_augment_mode_name(mode));
        printf("  batch=%lu eventi=%lu\n",
               (unsigned long)batches,
               (unsigned long)events);
        printf("  min/max/mean=%.6f / %.6f / %.6f\n",
               global_min,
               global_max,
               values ? sum / (double)values : 0.0);
        printf("  crop=%d-%d distinti=%lu/%d centro24=%lu\n",
               min_crop,
               max_crop,
               (unsigned long)distinct,
               max_crop - min_crop + 1,
               (unsigned long)crop_counts[GS_SAKAI_CENTRAL_CROP_LEFT]);

        if (max_batches == 0 &&
            (events != split->count ||
             distinct != (size_t)(max_crop - min_crop + 1))) {

            printf("Errore: train completo non coerente.\n");
            gs_dynamic_batch_free(&batch);
            return 1;
        }
    }

    gs_dynamic_batch_free(&batch);
    return 0;
}


static int run_validation(
    const GSEventManifest *manifest,
    const GSSavedSplit *split,
    size_t batch_size,
    size_t max_batches
)
{
    GSDynamicBatch batch = {0};
    size_t start = 0;
    size_t batches = 0;
    size_t events = 0;
    size_t central = 0;
    size_t values = 0;
    double sum = 0.0;
    float global_min = FLT_MAX;
    float global_max = -FLT_MAX;
    size_t i;

    if (gs_dynamic_batch_alloc(&batch, batch_size) != 0) {
        return 1;
    }

    while (start < split->count) {
        size_t current = split->count - start;
        GSDynamicBatchStats stats;

        if (current > batch_size) current = batch_size;

        if (gs_dynamic_batch_load_validation(
                manifest, split, start, current, &batch
            ) != 0 ||
            gs_dynamic_batch_compute_stats(&batch, &stats) != 0) {

            gs_dynamic_batch_free(&batch);
            return 1;
        }

        if (stats.min_value < 0.0f || stats.max_value > 1.0f) {
            printf("Errore: validation fuori [0,1].\n");
            gs_dynamic_batch_free(&batch);
            return 1;
        }

        if (stats.min_value < global_min) global_min = stats.min_value;
        if (stats.max_value > global_max) global_max = stats.max_value;

        sum += stats.mean_value * (double)(batch.count * GS_TENSOR_SIZE);
        values += batch.count * GS_TENSOR_SIZE;

        for (i = 0; i < batch.count; i++) {
            if (batch.crop_lefts[i] != GS_SAKAI_CENTRAL_CROP_LEFT) {
                printf("Errore: validation non centrale.\n");
                gs_dynamic_batch_free(&batch);
                return 1;
            }
            central++;
        }

        events += batch.count;
        start += batch.count;
        batches++;

        if (max_batches > 0 && batches >= max_batches) break;
    }

    printf("VALIDATION\n");
    printf("  batch=%lu eventi=%lu centrali24=%lu\n",
           (unsigned long)batches,
           (unsigned long)events,
           (unsigned long)central);
    printf("  min/max/mean=%.6f / %.6f / %.6f\n",
           global_min,
           global_max,
           values ? sum / (double)values : 0.0);

    if (max_batches == 0 &&
        (events != split->count || central != split->count)) {

        printf("Errore: validation completa non coerente.\n");
        gs_dynamic_batch_free(&batch);
        return 1;
    }

    gs_dynamic_batch_free(&batch);
    return 0;
}


int main(int argc, char **argv)
{
    const char *manifest_path = "manifest_events_272.csv";
    const char *train_path = "split_train_indices_seed123.csv";
    const char *val_path = "split_val_indices_seed123.csv";
    size_t batch_size = 32;
    unsigned int seed = 456u;
    size_t max_batches = 3;
    GSAugmentMode mode = GS_AUGMENT_MODE_LEGACY_TEXT;

    GSEventManifest manifest = {0};
    GSSavedSplit train = {0};
    GSSavedSplit val = {0};
    int rc = 1;

    if (argc >= 2) manifest_path = argv[1];
    if (argc >= 3) train_path = argv[2];
    if (argc >= 4) val_path = argv[3];
    if (argc >= 5) batch_size = (size_t)strtoul(argv[4], NULL, 10);
    if (argc >= 6) seed = (unsigned int)strtoul(argv[5], NULL, 10);
    if (argc >= 7) max_batches = (size_t)strtoul(argv[6], NULL, 10);

    if (argc >= 8 && parse_mode(argv[7], &mode) != 0) {
        printf("Modo non valido. Usa legacy_text o symmetric_figure.\n");
        return 1;
    }

    if (batch_size == 0) return 1;

    printf("Modo train: %s\n", gs_augment_mode_name(mode));

    if (gs_event_manifest_load(manifest_path, &manifest) != 0) goto cleanup;
    if (gs_saved_split_load(train_path, "train", &manifest, &train) != 0) goto cleanup;
    if (gs_saved_split_load(val_path, "validation", &manifest, &val) != 0) goto cleanup;

    if (run_train(&manifest, &train, batch_size, seed, max_batches, mode) != 0) {
        goto cleanup;
    }

    if (run_validation(&manifest, &val, batch_size, max_batches) != 0) {
        goto cleanup;
    }

    printf("OK: dry-run completato.\n");
    rc = 0;

cleanup:
    gs_saved_split_free(&val);
    gs_saved_split_free(&train);
    gs_event_manifest_free(&manifest);
    return rc;
}
