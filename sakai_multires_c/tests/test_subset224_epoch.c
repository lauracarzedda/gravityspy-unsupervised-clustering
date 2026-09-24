#include "vae_multires_batch.h"

#include <float.h>
#include <math.h>
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


static int check_subset(
    const GSEventManifest *events,
    const GSSavedSplit *train,
    const GSSavedSplit *validation
)
{
    unsigned char *owner;
    size_t i;
    size_t overlap = 0;
    size_t selected = 0;

    owner = (unsigned char *)calloc(events->count, 1);
    if (owner == NULL) return 1;

    for (i = 0; i < train->count; i++) {
        if (train->indices[i] >= events->count || owner[train->indices[i]] != 0) {
            overlap++;
        } else {
            owner[train->indices[i]] = 1;
        }
    }

    for (i = 0; i < validation->count; i++) {
        if (validation->indices[i] >= events->count ||
            owner[validation->indices[i]] != 0) {
            overlap++;
        } else {
            owner[validation->indices[i]] = 2;
        }
    }

    for (i = 0; i < events->count; i++) {
        if (owner[i] != 0) selected++;
    }

    free(owner);

    printf("Subset: train=%lu validation=%lu unici=%lu overlap=%lu\n",
           (unsigned long)train->count,
           (unsigned long)validation->count,
           (unsigned long)selected,
           (unsigned long)overlap);

    return overlap == 0 && selected == train->count + validation->count ? 0 : 1;
}


static int run_subset(
    const GSEventManifest *events,
    const GSSavedSplit *train,
    const GSSavedSplit *validation,
    const GSTensorManifest *tensors,
    size_t batch_size,
    unsigned int seed,
    GSAugmentMode mode
)
{
    GSTensorSpec spec;
    GSMultiresEventLoader loader = {0};
    GSMultiresBatch batch = {0};
    GSAugmentRng rng;
    size_t crop_counts[GS_CROP_COUNT] = {0};
    size_t start;
    size_t batches;
    size_t events_loaded;
    size_t values;
    size_t central_count;
    size_t distinct;
    size_t i;
    int min_crop;
    int max_crop;
    float global_min;
    float global_max;
    double sum;
    double mean;
    int rc = 1;

    if (gs_tensor_spec_init(&spec, 224) != 0 ||
        gs_multires_event_loader_init(&loader, 224) != 0 ||
        gs_multires_batch_alloc(&batch, &spec, batch_size) != 0 ||
        gs_augment_mode_crop_range(mode, &min_crop, &max_crop) != 0) {
        goto cleanup;
    }

    gs_augment_rng_seed(&rng, seed);
    start = 0;
    batches = 0;
    events_loaded = 0;
    values = 0;
    global_min = FLT_MAX;
    global_max = -FLT_MAX;
    sum = 0.0;

    while (start < train->count) {
        size_t wanted = train->count - start;
        GSMultiresBatchStats stats;

        if (wanted > batch_size) wanted = batch_size;

        if (gs_multires_batch_load_train_mode(
                events, train, &loader, start, wanted, &rng, mode, &batch
            ) != 0 ||
            gs_multires_batch_compute_stats(&batch, &stats) != 0) {
            goto cleanup;
        }

        if (stats.min_value < global_min) global_min = stats.min_value;
        if (stats.max_value > global_max) global_max = stats.max_value;

        for (i = 0; i < batch.count; i++) {
            int crop = batch.crop_lefts[i];
            if (crop < min_crop || crop > max_crop) goto cleanup;
            crop_counts[crop]++;
        }

        sum += stats.mean_value *
               (double)(batch.count * spec.values_per_event);
        values += batch.count * spec.values_per_event;
        events_loaded += batch.count;
        start += batch.count;
        batches++;
    }

    mean = sum / (double)values;
    distinct = 0;
    for (i = (size_t)min_crop; i <= (size_t)max_crop; i++) {
        if (crop_counts[i] != 0) distinct++;
    }

    printf("mini224_1000 TRAIN %s\n", gs_augment_mode_name(mode));
    printf("  batch=%lu eventi=%lu min/max/mean=%.6f / %.6f / %.6f\n",
           (unsigned long)batches,
           (unsigned long)events_loaded,
           global_min, global_max, mean);
    printf("  crop=%d-%d distinti=%lu/%d centro24=%lu\n",
           min_crop, max_crop,
           (unsigned long)distinct,
           max_crop - min_crop + 1,
           (unsigned long)crop_counts[GS_SAKAI_CENTRAL_CROP_LEFT]);

    if (events_loaded != train->count ||
        distinct != (size_t)(max_crop - min_crop + 1)) {
        goto cleanup;
    }

    if (mode == GS_AUGMENT_MODE_LEGACY_TEXT && seed == 456u &&
        fabs(mean - 0.193387) > 0.000002) {
        printf("Errore: regressione media train legacy.\n");
        goto cleanup;
    }

    start = 0;
    batches = 0;
    events_loaded = 0;
    values = 0;
    central_count = 0;
    global_min = FLT_MAX;
    global_max = -FLT_MAX;
    sum = 0.0;

    while (start < validation->count) {
        size_t wanted = validation->count - start;
        GSMultiresBatchStats stats;

        if (wanted > batch_size) wanted = batch_size;

        if (gs_multires_batch_load_validation(
                events, tensors, validation, start, wanted, &batch
            ) != 0 ||
            gs_multires_batch_compute_stats(&batch, &stats) != 0) {
            goto cleanup;
        }

        if (stats.min_value < global_min) global_min = stats.min_value;
        if (stats.max_value > global_max) global_max = stats.max_value;

        for (i = 0; i < batch.count; i++) {
            if (batch.crop_lefts[i] != GS_SAKAI_CENTRAL_CROP_LEFT) goto cleanup;
            central_count++;
        }

        sum += stats.mean_value *
               (double)(batch.count * spec.values_per_event);
        values += batch.count * spec.values_per_event;
        events_loaded += batch.count;
        start += batch.count;
        batches++;
    }

    mean = sum / (double)values;
    printf("mini224_1000 VALIDATION\n");
    printf("  batch=%lu eventi=%lu centrali24=%lu\n",
           (unsigned long)batches,
           (unsigned long)events_loaded,
           (unsigned long)central_count);
    printf("  min/max/mean=%.6f / %.6f / %.6f\n",
           global_min, global_max, mean);

    if (events_loaded != validation->count ||
        central_count != validation->count ||
        fabs(mean - 0.192118) > 0.000002) {
        goto cleanup;
    }

    rc = 0;

cleanup:
    gs_multires_batch_free(&batch);
    gs_multires_event_loader_free(&loader);
    return rc;
}


int main(int argc, char **argv)
{
    const char *train_path =
        "sakai_multires_c/configs/subset224_train_800_seed123.csv";
    const char *validation_path =
        "sakai_multires_c/configs/subset224_val_200_seed123.csv";
    size_t batch_size = 32;
    unsigned int seed = 456u;
    GSAugmentMode mode = GS_AUGMENT_MODE_LEGACY_TEXT;
    GSEventManifest events = {0};
    GSSavedSplit train = {0};
    GSSavedSplit validation = {0};
    GSTensorSpec spec;
    GSTensorManifest tensors = {0};
    int rc = 1;

    if (argc >= 2) train_path = argv[1];
    if (argc >= 3) validation_path = argv[2];
    if (argc >= 4) batch_size = (size_t)strtoul(argv[3], NULL, 10);
    if (argc >= 5) seed = (unsigned int)strtoul(argv[4], NULL, 10);
    if (argc >= 6 && parse_mode(argv[5], &mode) != 0) {
        printf("Uso: test_subset224_epoch.exe [train_csv] [val_csv] [batch] [seed] "
               "[legacy_text|symmetric_figure]\n");
        return 1;
    }

    if (batch_size == 0 ||
        gs_tensor_spec_init(&spec, 224) != 0 ||
        gs_event_manifest_load("manifest_events_272.csv", &events) != 0 ||
        gs_saved_split_load(train_path, "train", &events, &train) != 0 ||
        gs_saved_split_load(validation_path, "validation", &events, &validation) != 0 ||
        check_subset(&events, &train, &validation) != 0 ||
        gs_tensor_manifest_load(
            "manifest_tensors_224_central.csv", &spec, &tensors
        ) != 0) {
        goto cleanup;
    }

    printf("Modalita train: %s\n", gs_augment_mode_name(mode));

    if (run_subset(
            &events, &train, &validation, &tensors,
            batch_size, seed, mode
        ) != 0) {
        goto cleanup;
    }

    printf("OK: dry-run mini224_1000 completato.\n");
    rc = 0;

cleanup:
    gs_tensor_manifest_free(&tensors);
    gs_saved_split_free(&validation);
    gs_saved_split_free(&train);
    gs_event_manifest_free(&events);
    return rc;
}
