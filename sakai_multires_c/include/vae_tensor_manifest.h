#ifndef VAE_TENSOR_MANIFEST_H
#define VAE_TENSOR_MANIFEST_H

#include <stddef.h>

#include "vae_tensor_spec.h"


#define GS_TMAN_LABEL_LEN         64
#define GS_TMAN_EVENT_ID_LEN      128
#define GS_TMAN_PATH_LEN          512
#define GS_TMAN_DTYPE_LEN         32
#define GS_TMAN_LAYOUT_LEN        64
#define GS_TMAN_CHANNEL_ORDER_LEN 64
#define GS_TMAN_EXTRA_NAME_LEN    32
#define GS_TMAN_EXTRA_VALUE_LEN   128


/*
   Record generico dei manifest:

   manifest_tensors_224_central.csv
   manifest_tensors_112_central.csv
   manifest_tensors_75_central.csv
*/
typedef struct {
    int event_index;

    char label[GS_TMAN_LABEL_LEN];
    char event_id[GS_TMAN_EVENT_ID_LEN];
    char tensor_path[GS_TMAN_PATH_LEN];

    int channels;
    int height;
    int width;

    char dtype[GS_TMAN_DTYPE_LEN];
    char layout[GS_TMAN_LAYOUT_LEN];
    char channel_order[GS_TMAN_CHANNEL_ORDER_LEN];

    /*
       Per 224:
       crop_left = 24

       Per 112/75:
       origin = derived_from_224_central
    */
    char extra_value[GS_TMAN_EXTRA_VALUE_LEN];

} GSTensorManifestRecord;


/*
   Manifest completo in RAM.
*/
typedef struct {
    GSTensorManifestRecord *records;
    size_t count;

    GSTensorSpec spec;

    /*
       Nome dell'ultima colonna:
       "crop_left" oppure "origin".
    */
    char extra_column_name[GS_TMAN_EXTRA_NAME_LEN];

} GSTensorManifest;


/*
   Legge e valida un manifest tensoriale.

   expected_spec indica la shape attesa:
   75, 112 oppure 224.
*/
int gs_tensor_manifest_load(
    const char *csv_path,
    const GSTensorSpec *expected_spec,
    GSTensorManifest *manifest
);

/* Same validation without inspecting or copying the label value. */
int gs_tensor_manifest_load_label_free(
    const char *csv_path,
    const GSTensorSpec *expected_spec,
    GSTensorManifest *manifest
);


/*
   Libera memoria.
*/
void gs_tensor_manifest_free(
    GSTensorManifest *manifest
);


/*
   Restituisce il record zero-based.
*/
const GSTensorManifestRecord *gs_tensor_manifest_get(
    const GSTensorManifest *manifest,
    size_t index
);

#endif
