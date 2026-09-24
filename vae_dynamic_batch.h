#ifndef VAE_DYNAMIC_BATCH_H
#define VAE_DYNAMIC_BATCH_H

#include <stddef.h>

#include "vae_augmentation.h"
#include "vae_event_dataset.h"
#include "vae_saved_split.h"


typedef struct {
    size_t capacity;
    size_t count;

    float *x;
    size_t *manifest_indices;
    int *event_indices;
    int *crop_lefts;

    char *labels;
    char *event_ids;
} GSDynamicBatch;


typedef struct {
    float min_value;
    float max_value;
    double mean_value;
} GSDynamicBatchStats;


int gs_dynamic_batch_alloc(
    GSDynamicBatch *batch,
    size_t capacity
);

void gs_dynamic_batch_free(
    GSDynamicBatch *batch
);

float *gs_dynamic_batch_item_ptr(
    GSDynamicBatch *batch,
    size_t item_index
);

const char *gs_dynamic_batch_label(
    const GSDynamicBatch *batch,
    size_t item_index
);

const char *gs_dynamic_batch_event_id(
    const GSDynamicBatch *batch,
    size_t item_index
);


int gs_dynamic_batch_load_train_mode(
    const GSEventManifest *event_manifest,
    const GSSavedSplit *train_split,
    size_t start_pos,
    size_t batch_size,
    GSAugmentRng *rng,
    GSAugmentMode mode,
    GSDynamicBatch *batch
);

/* Compatibilita: usa sempre legacy_text, crop_left 0..24. */
int gs_dynamic_batch_load_train(
    const GSEventManifest *event_manifest,
    const GSSavedSplit *train_split,
    size_t start_pos,
    size_t batch_size,
    GSAugmentRng *rng,
    GSDynamicBatch *batch
);

int gs_dynamic_batch_load_validation(
    const GSEventManifest *event_manifest,
    const GSSavedSplit *val_split,
    size_t start_pos,
    size_t batch_size,
    GSDynamicBatch *batch
);

int gs_dynamic_batch_compute_stats(
    const GSDynamicBatch *batch,
    GSDynamicBatchStats *stats
);

void gs_dynamic_batch_print_summary(
    const char *name,
    const GSDynamicBatch *batch
);

#endif
