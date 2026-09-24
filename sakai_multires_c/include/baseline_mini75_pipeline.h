#ifndef BASELINE_MINI75_PIPELINE_H
#define BASELINE_MINI75_PIPELINE_H

#include <stddef.h>

#include "vae_event_dataset.h"
#include "vae_saved_split.h"
#include "vae_tensor_manifest.h"


enum {
    GS_BASELINE_MINI75_OFFICIAL_EVENT_COUNT = 8583,
    GS_BASELINE_MINI75_OFFICIAL_TRAIN_COUNT = 6866,
    GS_BASELINE_MINI75_OFFICIAL_VALIDATION_COUNT = 1717,

    /* Buffer locale usato per data_root + tensor_path mini75. */
    GS_BASELINE_MINI75_JOINED_PATH_LEN = 4096
};


enum {
    GS_BASELINE_MINI75_PIPELINE_SUCCESS = 0,
    GS_BASELINE_MINI75_PIPELINE_ERROR_NULL_ARGUMENT = 1,
    GS_BASELINE_MINI75_PIPELINE_ERROR_COUNTS = 2,
    GS_BASELINE_MINI75_PIPELINE_ERROR_ALIGNMENT = 3,
    GS_BASELINE_MINI75_PIPELINE_ERROR_RANGE = 4,
    GS_BASELINE_MINI75_PIPELINE_ERROR_BUFFER_SIZE = 5,
    GS_BASELINE_MINI75_PIPELINE_ERROR_OVERFLOW = 6,
    GS_BASELINE_MINI75_PIPELINE_ERROR_PATH = 7,
    GS_BASELINE_MINI75_PIPELINE_ERROR_TENSOR_READ = 8,
    GS_BASELINE_MINI75_PIPELINE_ERROR_TENSOR_VALUE = 9,
    GS_BASELINE_MINI75_PIPELINE_ERROR_FEATURE_EXTRACTION = 10
};


/*
   Valida globalmente dataset e split ufficiali mini75:
   conteggi, specifica tensoriale, allineamento event_index/event_id,
   origin, indici e assenza di overlap.

   Le label non fanno parte dell'interfaccia e non vengono usate.
*/
int gs_baseline_mini75_validate_alignment(
    const GSEventManifest *event_manifest,
    const GSTensorManifest *tensor_manifest,
    const GSSavedSplit *train_split,
    const GSSavedSplit *validation_split
);


/*
   Estrae un intervallo contiguo nell'ordine dello split.

   Per ogni riga usa split->indices[split_pos] direttamente come indice
   del manifest eventi e del manifest mini75. Il path viene costruito
   in un buffer locale fisso da GS_BASELINE_MINI75_JOINED_PATH_LEN byte,
   usando '/' come separatore compatibile anche con Windows.

   tensor_buffer deve contenere esattamente 22500 float.
   raw_features deve avere spazio per almeno row_count * 900 float.
   Il modulo non esegue allocazioni dinamiche.
*/
int gs_baseline_mini75_extract_split_rows(
    const char *data_root,
    const GSEventManifest *event_manifest,
    const GSTensorManifest *tensor_manifest,
    const GSSavedSplit *split,
    size_t first_row,
    size_t row_count,
    float *tensor_buffer,
    size_t tensor_buffer_count,
    float *raw_features,
    size_t raw_features_count
);

#endif
