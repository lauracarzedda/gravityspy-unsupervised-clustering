#ifndef VAE_MULTIRES_BATCH_H
#define VAE_MULTIRES_BATCH_H

#include <stddef.h>
#include <stdint.h>

#include "vae_multires_event_loader.h"
#include "vae_saved_split.h"
#include "vae_tensor_manifest.h"


typedef struct {
    GSTensorSpec spec;
    size_t capacity;
    size_t count;

    float *x;
    size_t *manifest_indices;
    int *event_indices;
    int *crop_lefts;
    char *labels;
    char *event_ids;
} GSMultiresBatch;


typedef struct {
    float min_value;
    float max_value;
    double mean_value;
} GSMultiresBatchStats;


int gs_multires_batch_alloc(
    GSMultiresBatch *batch,
    const GSTensorSpec *spec,
    size_t capacity
);

void gs_multires_batch_free(GSMultiresBatch *batch);

float *gs_multires_batch_item_ptr(
    GSMultiresBatch *batch,
    size_t item_index
);

const float *gs_multires_batch_item_const_ptr(
    const GSMultiresBatch *batch,
    size_t item_index
);

const char *gs_multires_batch_label(
    const GSMultiresBatch *batch,
    size_t item_index
);

const char *gs_multires_batch_event_id(
    const GSMultiresBatch *batch,
    size_t item_index
);

int gs_multires_batch_load_train_mode(
    const GSEventManifest *event_manifest,
    const GSSavedSplit *train_split,
    GSMultiresEventLoader *event_loader,
    size_t start_pos,
    size_t batch_size,
    GSAugmentRng *rng,
    GSAugmentMode mode,
    GSMultiresBatch *batch
);


/*
   Loader train stateless per evento/epoca.

   Il crop di ogni elemento dipende esclusivamente da:
       augmentation_base_seed, epoch, manifest_index, mode.

   Non usa GSAugmentRng e non dipende dall'ordine o dalla composizione
   del batch.
*/
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
);

/* Compatibilita: usa legacy_text, crop_left 0..24. */
int gs_multires_batch_load_train(
    const GSEventManifest *event_manifest,
    const GSSavedSplit *train_split,
    GSMultiresEventLoader *event_loader,
    size_t start_pos,
    size_t batch_size,
    GSAugmentRng *rng,
    GSMultiresBatch *batch
);

int gs_multires_batch_load_validation(
    const GSEventManifest *event_manifest,
    const GSTensorManifest *tensor_manifest,
    const GSSavedSplit *val_split,
    size_t start_pos,
    size_t batch_size,
    GSMultiresBatch *batch
);

int gs_multires_batch_compute_stats(
    const GSMultiresBatch *batch,
    GSMultiresBatchStats *stats
);

void gs_multires_batch_print_summary(
    const char *name,
    const GSMultiresBatch *batch
);

#endif
