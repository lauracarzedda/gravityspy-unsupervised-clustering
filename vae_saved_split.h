#ifndef VAE_SAVED_SPLIT_H
#define VAE_SAVED_SPLIT_H

#include <stddef.h>

#include "vae_event_dataset.h"


#define GS_SAVED_SPLIT_NAME_LEN 32


/*
   Uno split già salvato su CSV.

   indices contiene manifest_index zero-based,
   riferiti a manifest_events_272.csv.
*/
typedef struct {
    size_t *indices;
    size_t count;

    char name[GS_SAVED_SPLIT_NAME_LEN];
} GSSavedSplit;


/*
   Legge uno dei file:

   split_train_indices_seed123.csv
   split_val_indices_seed123.csv

   expected_split_name deve essere:
   "train"
   oppure
   "validation"

   Il caricamento controlla anche che:
   - manifest_index sia valido;
   - event_index coincida;
   - event_id coincida con manifest_events_272.csv;
   - non ci siano duplicati nello stesso split.
*/
int gs_saved_split_load(
    const char *csv_path,
    const char *expected_split_name,
    const GSEventManifest *event_manifest,
    GSSavedSplit *split
);


/*
   Libera la memoria dello split.
*/
void gs_saved_split_free(
    GSSavedSplit *split
);

#endif