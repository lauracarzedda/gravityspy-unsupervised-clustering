#ifndef VAE_EVENT_DATASET_H
#define VAE_EVENT_DATASET_H

#include <stddef.h>


/*
   Numero delle viste temporali di ogni evento Gravity Spy.
*/
#define GS_EVENT_VIEW_COUNT 4


/*
   Dimensioni delle immagini indicate da manifest_events_272.csv.

   In C:
   width  = direzione temporale = 272
   height = direzione frequenza = 224
*/
#define GS_EVENT_SOURCE_WIDTH  272
#define GS_EVENT_SOURCE_HEIGHT 224


/*
   Dimensioni massime dei campi testuali.
*/
#define GS_EVENT_MAX_LABEL_LEN 64
#define GS_EVENT_MAX_ID_LEN    128
#define GS_EVENT_MAX_PATH_LEN  512


/*
   Un record corrisponde a una riga dati di:

   manifest_events_272.csv

   Ogni record rappresenta un evento completo con quattro viste:
   image_paths[0] = 0.5 s
   image_paths[1] = 1.0 s
   image_paths[2] = 2.0 s
   image_paths[3] = 4.0 s
*/
typedef struct {
    int event_index;

    char label[GS_EVENT_MAX_LABEL_LEN];
    char event_id[GS_EVENT_MAX_ID_LEN];

    char image_paths[GS_EVENT_VIEW_COUNT][GS_EVENT_MAX_PATH_LEN];

} GSEventRecord;


/*
   Intero manifest eventi caricato in memoria.
*/
typedef struct {
    GSEventRecord *records;
    size_t count;
} GSEventManifest;


/*
   Legge manifest_events_272.csv e costruisce
   un array di GSEventRecord.

   Ritorna:
   0 = successo
   diverso da 0 = errore
*/
int gs_event_manifest_load(
    const char *csv_path,
    GSEventManifest *manifest
);

/* Loads event identities and image paths without inspecting or copying the
 * label value. The CSV delimiter parser still locates the column boundary. */
int gs_event_manifest_load_label_free(
    const char *csv_path,
    GSEventManifest *manifest
);


/*
   Libera la memoria del manifest.
*/
void gs_event_manifest_free(
    GSEventManifest *manifest
);


/*
   Restituisce il record zero-based:

   index = 0 -> primo evento
   index = 1 -> secondo evento
*/
const GSEventRecord *gs_event_manifest_get(
    const GSEventManifest *manifest,
    size_t index
);


/*
   Restituisce il nome leggibile della vista temporale.

   0 -> "0.5 s"
   1 -> "1.0 s"
   2 -> "2.0 s"
   3 -> "4.0 s"
*/
const char *gs_event_view_name(
    size_t view_index
);


/*
   Restituisce il percorso della vista richiesta.
*/
const char *gs_event_image_path(
    const GSEventRecord *record,
    size_t view_index
);


/*
   Stampa un record in forma leggibile.
*/
void gs_event_manifest_print_record(
    const GSEventRecord *record
);


/*
   Controlla che tutti i file PNG indicati dal manifest esistano.

   checked_files:
   numero totale dei percorsi controllati.

   missing_files:
   numero di file non trovati.

   max_missing_to_print:
   numero massimo di percorsi mancanti da stampare nel terminale.
*/
int gs_event_manifest_check_files(
    const GSEventManifest *manifest,
    size_t *checked_files,
    size_t *missing_files,
    size_t max_missing_to_print
);

#endif
