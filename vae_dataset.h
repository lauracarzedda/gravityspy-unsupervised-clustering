#ifndef VAE_DATASET_H
#define VAE_DATASET_H

#include <stddef.h>
#include "vae_loader.h"

#define GS_MAX_LABEL_LEN          64
#define GS_MAX_EVENT_ID_LEN       128
#define GS_MAX_PATH_LEN           512
#define GS_MAX_DTYPE_LEN          32
#define GS_MAX_LAYOUT_LEN         64
#define GS_MAX_CHANNEL_ORDER_LEN  64

/*
   Una riga del manifest_tensors_224_central.csv.

   Nota:
   label è conservata per valutazione finale,
   non per il training non supervisionato.
*/
typedef struct {
    int event_index;

    char label[GS_MAX_LABEL_LEN];
    char event_id[GS_MAX_EVENT_ID_LEN];
    char tensor_path[GS_MAX_PATH_LEN];

    int channels;
    int height;
    int width;

    char dtype[GS_MAX_DTYPE_LEN];
    char layout[GS_MAX_LAYOUT_LEN];
    char channel_order[GS_MAX_CHANNEL_ORDER_LEN];

    int crop_left;
} GSManifestRecord;


/*
   Manifest completo in memoria.
*/
typedef struct {
    GSManifestRecord *records;
    size_t count;
} GSManifest;


/*
   Carica il manifest CSV dei tensori centrali 224.
*/
int gs_manifest_load(const char *csv_path, GSManifest *manifest);


/*
   Libera la memoria del manifest.
*/
void gs_manifest_free(GSManifest *manifest);


/*
   Ritorna il record numero index.
   index è zero-based:
   index = 0 -> prima riga dati, cioè event_000001.bin.
*/
const GSManifestRecord *gs_manifest_get(const GSManifest *manifest, size_t index);


/*
   Stampa un riassunto leggibile di un record.
*/
void gs_manifest_print_record(const GSManifestRecord *record);


/*
   Carica il tensore associato al record index.
   Produce float [0,1] usando il loader già creato.
*/
int gs_manifest_load_tensor(const GSManifest *manifest, size_t index, float *out);

#endif