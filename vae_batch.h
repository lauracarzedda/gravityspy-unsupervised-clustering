#ifndef VAE_BATCH_H
#define VAE_BATCH_H

#include <stddef.h>

#include "vae_loader.h"
#include "vae_dataset.h"


/*
   Batch di tensori per il VAE.

   Shape logica:
   count x 4 x 224 x 224

   Dove:
   - count è il numero effettivo di eventi caricati nel batch
   - capacity è il massimo numero di eventi allocati
   - x contiene tutti i valori float normalizzati [0,1]
*/
typedef struct {
    size_t capacity;
    size_t count;

    /*
       Array float del batch.
       Dimensione allocata:
       capacity * GS_TENSOR_SIZE
    */
    float *x;

    /*
       Indici zero-based nel manifest.
       Esempio:
       manifest_indices[0] = 0 significa prima riga dati del manifest.
    */
    size_t *manifest_indices;

    /*
       event_index originale letto dal CSV.
       Esempio:
       event_indices[0] = 1 per event_000001.bin.
    */
    int *event_indices;

    /*
       Label e event_id vengono conservati solo per controlli
       e valutazione futura. Non saranno usati per allenare il VAE.
    */
    char *labels;
    char *event_ids;

} GSBatch;


/*
   Statistiche globali del batch.
   Servono solo per controllo, non per il training.
*/
typedef struct {
    float min_value;
    float max_value;
    double mean_value;
} GSBatchStats;


/*
   Alloca memoria per un batch.
*/
int gs_batch_alloc(GSBatch *batch, size_t capacity);


/*
   Libera la memoria del batch.
*/
void gs_batch_free(GSBatch *batch);


/*
   Restituisce il puntatore al tensore item_index dentro il batch.
*/
float *gs_batch_item_ptr(GSBatch *batch, size_t item_index);


/*
   Versione const del puntatore al tensore.
*/
const float *gs_batch_item_const_ptr(const GSBatch *batch, size_t item_index);


/*
   Restituisce label ed event_id dell'elemento item_index.
*/
const char *gs_batch_label(const GSBatch *batch, size_t item_index);
const char *gs_batch_event_id(const GSBatch *batch, size_t item_index);


/*
   Carica un batch sequenziale dal manifest.

   Esempio:
   start_index = 0, batch_size = 4
   carica gli eventi manifest 0, 1, 2, 3.

   Per ora non facciamo shuffle.
   Lo shuffle verrà dopo.
*/
int gs_batch_load_sequential(
    const GSManifest *manifest,
    size_t start_index,
    size_t batch_size,
    GSBatch *batch
);


/*
   Calcola min, max, mean su tutto il batch.
*/
int gs_batch_compute_stats(const GSBatch *batch, GSBatchStats *stats);


/*
   Stampa riassunto leggibile del batch.
*/
void gs_batch_print_summary(const GSBatch *batch);

#endif

/*
   Carica un batch usando una lista di indici.

   Serve per lo shuffle:
   gli indici non sono più 0,1,2,3...
   ma possono essere mescolati, per esempio:
   3812, 17, 7900, 204...

   indices contiene indici zero-based del manifest.
*/
int gs_batch_load_from_indices(
    const GSManifest *manifest,
    const size_t *indices,
    size_t index_count,
    size_t start_pos,
    size_t batch_size,
    GSBatch *batch
);