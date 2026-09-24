#include "vae_dynamic_batch.h"

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


static void gs_batch_zero(GSDynamicBatch *batch)
{
    if (batch == NULL) {
        return;
    }

    memset(batch, 0, sizeof(*batch));
}


static void gs_copy_slot(
    char *base,
    size_t slot_size,
    size_t index,
    const char *source
)
{
    char *destination;

    destination = base + index * slot_size;
    strncpy(destination, source, slot_size - 1);
    destination[slot_size - 1] = '\0';
}


static int gs_validate_load(
    const GSEventManifest *manifest,
    const GSSavedSplit *split,
    size_t start_pos,
    size_t batch_size,
    const GSDynamicBatch *batch
)
{
    if (manifest == NULL || split == NULL || split->indices == NULL) {
        printf("Errore: manifest o split non valido.\n");
        return 1;
    }

    if (batch == NULL || batch->x == NULL) {
        printf("Errore: batch non allocato.\n");
        return 1;
    }

    if (batch_size == 0 || batch_size > batch->capacity) {
        printf("Errore: batch_size non valido.\n");
        return 1;
    }

    if (start_pos >= split->count) {
        printf("Errore: start_pos fuori dallo split.\n");
        return 1;
    }

    return 0;
}


static void gs_store_metadata(
    GSDynamicBatch *batch,
    size_t item,
    size_t manifest_index,
    const GSEventRecord *record,
    int crop_left
)
{
    batch->manifest_indices[item] = manifest_index;
    batch->event_indices[item] = record->event_index;
    batch->crop_lefts[item] = crop_left;

    gs_copy_slot(
        batch->labels,
        GS_EVENT_MAX_LABEL_LEN,
        item,
        record->label
    );

    gs_copy_slot(
        batch->event_ids,
        GS_EVENT_MAX_ID_LEN,
        item,
        record->event_id
    );
}


int gs_dynamic_batch_alloc(
    GSDynamicBatch *batch,
    size_t capacity
)
{
    if (batch == NULL || capacity == 0) {
        return 1;
    }

    gs_batch_zero(batch);

    batch->x = (float *)malloc(
        capacity * GS_TENSOR_SIZE * sizeof(float)
    );
    batch->manifest_indices = (size_t *)malloc(
        capacity * sizeof(size_t)
    );
    batch->event_indices = (int *)malloc(
        capacity * sizeof(int)
    );
    batch->crop_lefts = (int *)malloc(
        capacity * sizeof(int)
    );
    batch->labels = (char *)calloc(
        capacity,
        GS_EVENT_MAX_LABEL_LEN
    );
    batch->event_ids = (char *)calloc(
        capacity,
        GS_EVENT_MAX_ID_LEN
    );

    if (batch->x == NULL ||
        batch->manifest_indices == NULL ||
        batch->event_indices == NULL ||
        batch->crop_lefts == NULL ||
        batch->labels == NULL ||
        batch->event_ids == NULL) {

        gs_dynamic_batch_free(batch);
        return 1;
    }

    batch->capacity = capacity;
    return 0;
}


void gs_dynamic_batch_free(GSDynamicBatch *batch)
{
    if (batch == NULL) {
        return;
    }

    free(batch->x);
    free(batch->manifest_indices);
    free(batch->event_indices);
    free(batch->crop_lefts);
    free(batch->labels);
    free(batch->event_ids);

    gs_batch_zero(batch);
}


float *gs_dynamic_batch_item_ptr(
    GSDynamicBatch *batch,
    size_t item_index
)
{
    if (batch == NULL || item_index >= batch->capacity) {
        return NULL;
    }

    return batch->x + item_index * GS_TENSOR_SIZE;
}


const char *gs_dynamic_batch_label(
    const GSDynamicBatch *batch,
    size_t item_index
)
{
    if (batch == NULL || item_index >= batch->capacity) {
        return NULL;
    }

    return batch->labels + item_index * GS_EVENT_MAX_LABEL_LEN;
}


const char *gs_dynamic_batch_event_id(
    const GSDynamicBatch *batch,
    size_t item_index
)
{
    if (batch == NULL || item_index >= batch->capacity) {
        return NULL;
    }

    return batch->event_ids + item_index * GS_EVENT_MAX_ID_LEN;
}


int gs_dynamic_batch_load_train_mode(
    const GSEventManifest *event_manifest,
    const GSSavedSplit *train_split,
    size_t start_pos,
    size_t batch_size,
    GSAugmentRng *rng,
    GSAugmentMode mode,
    GSDynamicBatch *batch
)
{
    size_t actual_count;
    size_t i;
    int min_crop;
    int max_crop;

    if (gs_validate_load(
            event_manifest,
            train_split,
            start_pos,
            batch_size,
            batch
        ) != 0 || rng == NULL) {

        return 1;
    }

    if (gs_augment_mode_crop_range(mode, &min_crop, &max_crop) != 0) {
        return 1;
    }

    (void)min_crop;
    (void)max_crop;

    actual_count = train_split->count - start_pos;
    if (actual_count > batch_size) {
        actual_count = batch_size;
    }

    batch->count = 0;

    for (i = 0; i < actual_count; i++) {
        size_t manifest_index;
        const GSEventRecord *record;
        float *destination;
        int crop_left;

        manifest_index = train_split->indices[start_pos + i];
        record = gs_event_manifest_get(event_manifest, manifest_index);
        destination = gs_dynamic_batch_item_ptr(batch, i);

        if (record == NULL || destination == NULL) {
            return 1;
        }

        if (gs_event_load_random_crop_float01_mode(
                record,
                rng,
                mode,
                &crop_left,
                destination
            ) != 0) {

            return 1;
        }

        gs_store_metadata(
            batch,
            i,
            manifest_index,
            record,
            crop_left
        );

        batch->count++;
    }

    return 0;
}


int gs_dynamic_batch_load_train(
    const GSEventManifest *event_manifest,
    const GSSavedSplit *train_split,
    size_t start_pos,
    size_t batch_size,
    GSAugmentRng *rng,
    GSDynamicBatch *batch
)
{
    return gs_dynamic_batch_load_train_mode(
        event_manifest,
        train_split,
        start_pos,
        batch_size,
        rng,
        GS_AUGMENT_MODE_LEGACY_TEXT,
        batch
    );
}


int gs_dynamic_batch_load_validation(
    const GSEventManifest *event_manifest,
    const GSSavedSplit *val_split,
    size_t start_pos,
    size_t batch_size,
    GSDynamicBatch *batch
)
{
    size_t actual_count;
    size_t i;

    if (gs_validate_load(
            event_manifest,
            val_split,
            start_pos,
            batch_size,
            batch
        ) != 0) {

        return 1;
    }

    actual_count = val_split->count - start_pos;
    if (actual_count > batch_size) {
        actual_count = batch_size;
    }

    batch->count = 0;

    for (i = 0; i < actual_count; i++) {
        size_t manifest_index;
        const GSEventRecord *record;
        float *destination;

        manifest_index = val_split->indices[start_pos + i];
        record = gs_event_manifest_get(event_manifest, manifest_index);
        destination = gs_dynamic_batch_item_ptr(batch, i);

        if (record == NULL || destination == NULL) {
            return 1;
        }

        if (gs_event_load_central_float01(record, destination) != 0) {
            return 1;
        }

        gs_store_metadata(
            batch,
            i,
            manifest_index,
            record,
            GS_SAKAI_CENTRAL_CROP_LEFT
        );

        batch->count++;
    }

    return 0;
}


int gs_dynamic_batch_compute_stats(
    const GSDynamicBatch *batch,
    GSDynamicBatchStats *stats
)
{
    size_t total_values;
    size_t i;
    double sum;

    if (batch == NULL || batch->x == NULL ||
        batch->count == 0 || stats == NULL) {

        return 1;
    }

    total_values = batch->count * GS_TENSOR_SIZE;
    stats->min_value = FLT_MAX;
    stats->max_value = -FLT_MAX;
    sum = 0.0;

    for (i = 0; i < total_values; i++) {
        float value;

        value = batch->x[i];
        if (value < stats->min_value) stats->min_value = value;
        if (value > stats->max_value) stats->max_value = value;
        sum += (double)value;
    }

    stats->mean_value = sum / (double)total_values;
    return 0;
}


void gs_dynamic_batch_print_summary(
    const char *name,
    const GSDynamicBatch *batch
)
{
    GSDynamicBatchStats stats;
    size_t i;

    if (name == NULL || batch == NULL) {
        return;
    }

    printf("=== %s ===\n", name);
    printf("Shape: %lu x %d x %d x %d\n",
           (unsigned long)batch->count,
           GS_CHANNELS,
           GS_HEIGHT,
           GS_WIDTH);

    if (gs_dynamic_batch_compute_stats(batch, &stats) == 0) {
        printf("Min/max/mean: %.6f / %.6f / %.6f\n",
               stats.min_value,
               stats.max_value,
               stats.mean_value);
    }

    for (i = 0; i < batch->count; i++) {
        printf("[%lu] manifest=%lu crop=%d event_id=%s label=%s\n",
               (unsigned long)i,
               (unsigned long)batch->manifest_indices[i],
               batch->crop_lefts[i],
               gs_dynamic_batch_event_id(batch, i),
               gs_dynamic_batch_label(batch, i));
    }

    printf("\n");
}
