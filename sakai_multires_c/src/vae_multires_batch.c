#include "vae_multires_batch.h"
#include "vae_tensor_io.h"

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


static void gs_batch_zero(GSMultiresBatch *batch)
{
    if (batch != NULL) {
        memset(batch, 0, sizeof(*batch));
    }
}


static int gs_specs_equal(const GSTensorSpec *a, const GSTensorSpec *b)
{
    return gs_tensor_spec_is_valid(a) &&
           gs_tensor_spec_is_valid(b) &&
           a->channels == b->channels &&
           a->height == b->height &&
           a->width == b->width &&
           a->values_per_event == b->values_per_event;
}


static void gs_copy_slot(
    char *base,
    size_t slot_size,
    size_t index,
    const char *source
)
{
    snprintf(base + index * slot_size, slot_size, "%s", source);
}


static int gs_validate_load(
    const GSEventManifest *manifest,
    const GSSavedSplit *split,
    size_t start_pos,
    size_t batch_size,
    const GSMultiresBatch *batch
)
{
    if (manifest == NULL || manifest->records == NULL ||
        split == NULL || split->indices == NULL ||
        batch == NULL || batch->x == NULL ||
        !gs_tensor_spec_is_valid(&batch->spec)) {
        return 1;
    }

    if (batch_size == 0 || batch_size > batch->capacity ||
        start_pos >= split->count) {
        return 1;
    }

    return 0;
}


static void gs_store_metadata(
    GSMultiresBatch *batch,
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


int gs_multires_batch_alloc(
    GSMultiresBatch *batch,
    const GSTensorSpec *spec,
    size_t capacity
)
{
    if (batch == NULL || !gs_tensor_spec_is_valid(spec) || capacity == 0) {
        return 1;
    }

    gs_batch_zero(batch);
    batch->spec = *spec;

    batch->x = (float *)malloc(
        capacity * spec->values_per_event * sizeof(float)
    );
    batch->manifest_indices = (size_t *)malloc(capacity * sizeof(size_t));
    batch->event_indices = (int *)malloc(capacity * sizeof(int));
    batch->crop_lefts = (int *)malloc(capacity * sizeof(int));
    batch->labels = (char *)calloc(capacity, GS_EVENT_MAX_LABEL_LEN);
    batch->event_ids = (char *)calloc(capacity, GS_EVENT_MAX_ID_LEN);

    if (batch->x == NULL ||
        batch->manifest_indices == NULL ||
        batch->event_indices == NULL ||
        batch->crop_lefts == NULL ||
        batch->labels == NULL ||
        batch->event_ids == NULL) {
        gs_multires_batch_free(batch);
        return 1;
    }

    batch->capacity = capacity;
    return 0;
}


void gs_multires_batch_free(GSMultiresBatch *batch)
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


float *gs_multires_batch_item_ptr(
    GSMultiresBatch *batch,
    size_t item_index
)
{
    if (batch == NULL || batch->x == NULL || item_index >= batch->capacity) {
        return NULL;
    }

    return batch->x + item_index * batch->spec.values_per_event;
}


const float *gs_multires_batch_item_const_ptr(
    const GSMultiresBatch *batch,
    size_t item_index
)
{
    if (batch == NULL || batch->x == NULL || item_index >= batch->capacity) {
        return NULL;
    }

    return batch->x + item_index * batch->spec.values_per_event;
}


const char *gs_multires_batch_label(
    const GSMultiresBatch *batch,
    size_t item_index
)
{
    if (batch == NULL || batch->labels == NULL || item_index >= batch->capacity) {
        return NULL;
    }

    return batch->labels + item_index * GS_EVENT_MAX_LABEL_LEN;
}


const char *gs_multires_batch_event_id(
    const GSMultiresBatch *batch,
    size_t item_index
)
{
    if (batch == NULL || batch->event_ids == NULL || item_index >= batch->capacity) {
        return NULL;
    }

    return batch->event_ids + item_index * GS_EVENT_MAX_ID_LEN;
}


int gs_multires_batch_load_train_mode(
    const GSEventManifest *event_manifest,
    const GSSavedSplit *train_split,
    GSMultiresEventLoader *event_loader,
    size_t start_pos,
    size_t batch_size,
    GSAugmentRng *rng,
    GSAugmentMode mode,
    GSMultiresBatch *batch
)
{
    const GSTensorSpec *loader_spec;
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
        ) != 0 || event_loader == NULL || rng == NULL) {
        return 1;
    }

    if (gs_augment_mode_crop_range(mode, &min_crop, &max_crop) != 0) {
        return 1;
    }

    (void)min_crop;
    (void)max_crop;

    loader_spec = gs_multires_event_loader_spec(event_loader);
    if (!gs_specs_equal(loader_spec, &batch->spec)) {
        return 1;
    }

    actual_count = train_split->count - start_pos;
    if (actual_count > batch_size) {
        actual_count = batch_size;
    }

    batch->count = 0;

    for (i = 0; i < actual_count; i++) {
        size_t manifest_index = train_split->indices[start_pos + i];
        const GSEventRecord *record =
            gs_event_manifest_get(event_manifest, manifest_index);
        float *destination = gs_multires_batch_item_ptr(batch, i);
        int crop_left;

        if (record == NULL || destination == NULL) {
            return 1;
        }

        if (gs_multires_event_load_random_float01_mode(
                event_loader,
                record,
                rng,
                mode,
                &crop_left,
                destination
            ) != 0) {
            return 1;
        }

        gs_store_metadata(batch, i, manifest_index, record, crop_left);
        batch->count++;
    }

    return 0;
}


int gs_multires_batch_load_train_stateless_mode(
    const GSEventManifest *event_manifest,
    const GSSavedSplit *train_split,
    GSMultiresEventLoader *event_loader,
    size_t start_pos,
    size_t batch_size,
    uint64_t augmentation_base_seed,
    uint64_t epoch,
    GSAugmentMode mode,
    GSMultiresBatch *batch
)
{
    const GSTensorSpec *loader_spec;
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
        ) != 0 || event_loader == NULL) {

        return 1;
    }

    if (gs_augment_mode_crop_range(mode, &min_crop, &max_crop) != 0) {
        return 1;
    }

    (void)min_crop;
    (void)max_crop;

    loader_spec = gs_multires_event_loader_spec(event_loader);
    if (!gs_specs_equal(loader_spec, &batch->spec)) {
        return 1;
    }

    actual_count = train_split->count - start_pos;
    if (actual_count > batch_size) {
        actual_count = batch_size;
    }

    batch->count = 0;

    for (i = 0; i < actual_count; i++) {
        size_t manifest_index = train_split->indices[start_pos + i];
        uint64_t manifest_key = (uint64_t)manifest_index;
        const GSEventRecord *record =
            gs_event_manifest_get(event_manifest, manifest_index);
        float *destination = gs_multires_batch_item_ptr(batch, i);
        int crop_left;

        if ((size_t)manifest_key != manifest_index ||
            record == NULL || destination == NULL) {

            return 1;
        }

        if (gs_augment_crop_left_for_event_mode(
                augmentation_base_seed,
                epoch,
                manifest_key,
                mode,
                &crop_left
            ) != 0) {

            return 1;
        }

        if (gs_multires_event_load_crop_float01(
                event_loader,
                record,
                crop_left,
                destination
            ) != 0) {

            return 1;
        }

        gs_store_metadata(batch, i, manifest_index, record, crop_left);
        batch->count++;
    }

    return 0;
}


int gs_multires_batch_load_train(
    const GSEventManifest *event_manifest,
    const GSSavedSplit *train_split,
    GSMultiresEventLoader *event_loader,
    size_t start_pos,
    size_t batch_size,
    GSAugmentRng *rng,
    GSMultiresBatch *batch
)
{
    return gs_multires_batch_load_train_mode(
        event_manifest,
        train_split,
        event_loader,
        start_pos,
        batch_size,
        rng,
        GS_AUGMENT_MODE_LEGACY_TEXT,
        batch
    );
}


int gs_multires_batch_load_validation(
    const GSEventManifest *event_manifest,
    const GSTensorManifest *tensor_manifest,
    const GSSavedSplit *val_split,
    size_t start_pos,
    size_t batch_size,
    GSMultiresBatch *batch
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
        ) != 0 || tensor_manifest == NULL || tensor_manifest->records == NULL) {
        return 1;
    }

    if (!gs_specs_equal(&tensor_manifest->spec, &batch->spec) ||
        tensor_manifest->count != event_manifest->count) {
        return 1;
    }

    actual_count = val_split->count - start_pos;
    if (actual_count > batch_size) {
        actual_count = batch_size;
    }

    batch->count = 0;

    for (i = 0; i < actual_count; i++) {
        size_t manifest_index = val_split->indices[start_pos + i];
        const GSEventRecord *event_record =
            gs_event_manifest_get(event_manifest, manifest_index);
        const GSTensorManifestRecord *tensor_record =
            gs_tensor_manifest_get(tensor_manifest, manifest_index);
        float *destination = gs_multires_batch_item_ptr(batch, i);

        if (event_record == NULL || tensor_record == NULL || destination == NULL) {
            return 1;
        }

        if (event_record->event_index != tensor_record->event_index ||
            strcmp(event_record->event_id, tensor_record->event_id) != 0) {
            return 1;
        }

        if (gs_tensor_read_float01(
                tensor_record->tensor_path,
                &batch->spec,
                destination
            ) != 0) {
            return 1;
        }

        gs_store_metadata(
            batch,
            i,
            manifest_index,
            event_record,
            GS_SAKAI_CENTRAL_CROP_LEFT
        );

        batch->count++;
    }

    return 0;
}


int gs_multires_batch_compute_stats(
    const GSMultiresBatch *batch,
    GSMultiresBatchStats *stats
)
{
    size_t total_values;
    size_t i;
    double sum = 0.0;

    if (batch == NULL || batch->x == NULL || batch->count == 0 || stats == NULL) {
        return 1;
    }

    total_values = batch->count * batch->spec.values_per_event;
    stats->min_value = FLT_MAX;
    stats->max_value = -FLT_MAX;

    for (i = 0; i < total_values; i++) {
        float value = batch->x[i];

        if (value < stats->min_value) {
            stats->min_value = value;
        }
        if (value > stats->max_value) {
            stats->max_value = value;
        }
        sum += (double)value;
    }

    stats->mean_value = sum / (double)total_values;
    return 0;
}


void gs_multires_batch_print_summary(
    const char *name,
    const GSMultiresBatch *batch
)
{
    GSMultiresBatchStats stats;
    size_t i;

    if (name == NULL || batch == NULL) {
        return;
    }

    printf("=== %s ===\n", name);
    printf("Shape: %lu x %d x %d x %d\n",
           (unsigned long)batch->count,
           batch->spec.channels,
           batch->spec.height,
           batch->spec.width);

    if (gs_multires_batch_compute_stats(batch, &stats) == 0) {
        printf("Min/max/mean: %.6f / %.6f / %.6f\n",
               stats.min_value,
               stats.max_value,
               stats.mean_value);
    }

    for (i = 0; i < batch->count; i++) {
        printf("[%lu] manifest=%lu crop=%d shift=%+d event_id=%s label=%s\n",
               (unsigned long)i,
               (unsigned long)batch->manifest_indices[i],
               batch->crop_lefts[i],
               batch->crop_lefts[i] - GS_SAKAI_CENTRAL_CROP_LEFT,
               gs_multires_batch_event_id(batch, i),
               gs_multires_batch_label(batch, i));
    }

    printf("\n");
}
