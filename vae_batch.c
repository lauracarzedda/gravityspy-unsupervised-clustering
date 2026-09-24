#include "vae_batch.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>


/*
   Porta la struttura batch in uno stato vuoto/sicuro.
*/
static void gs_batch_zero(GSBatch *batch)
{
    if (batch == NULL) {
        return;
    }

    batch->capacity = 0;
    batch->count = 0;

    batch->x = NULL;
    batch->manifest_indices = NULL;
    batch->event_indices = NULL;
    batch->labels = NULL;
    batch->event_ids = NULL;
}


/*
   Copia una stringa dentro uno slot dell'array labels/event_ids.

   Esempio:
   labels è un blocco unico:
   [slot 0][slot 1][slot 2]...

   Ogni slot ha lunghezza GS_MAX_LABEL_LEN.
*/
static void gs_copy_to_slot(
    char *base,
    size_t slot_size,
    size_t index,
    const char *src
)
{
    char *dst;

    if (base == NULL || slot_size == 0) {
        return;
    }

    dst = base + index * slot_size;

    if (src == NULL) {
        dst[0] = '\0';
        return;
    }

    strncpy(dst, src, slot_size - 1);
    dst[slot_size - 1] = '\0';
}


/*
   Alloca memoria per un batch.

   Shape logica:
   capacity x 4 x 224 x 224

   Il batch conterrà tensori float normalizzati [0,1].
*/
int gs_batch_alloc(GSBatch *batch, size_t capacity)
{
    size_t total_values;

    if (batch == NULL) {
        printf("Errore: batch NULL.\n");
        return 1;
    }

    if (capacity == 0) {
        printf("Errore: capacity batch uguale a zero.\n");
        return 1;
    }

    gs_batch_zero(batch);

    total_values = capacity * GS_TENSOR_SIZE;

    batch->x = (float *)malloc(total_values * sizeof(float));
    batch->manifest_indices = (size_t *)malloc(capacity * sizeof(size_t));
    batch->event_indices = (int *)malloc(capacity * sizeof(int));

    batch->labels = (char *)calloc(capacity, GS_MAX_LABEL_LEN);
    batch->event_ids = (char *)calloc(capacity, GS_MAX_EVENT_ID_LEN);

    if (batch->x == NULL ||
        batch->manifest_indices == NULL ||
        batch->event_indices == NULL ||
        batch->labels == NULL ||
        batch->event_ids == NULL) {

        printf("Errore: memoria insufficiente per allocare il batch.\n");
        gs_batch_free(batch);
        return 1;
    }

    batch->capacity = capacity;
    batch->count = 0;

    return 0;
}


/*
   Libera tutta la memoria associata al batch.
*/
void gs_batch_free(GSBatch *batch)
{
    if (batch == NULL) {
        return;
    }

    free(batch->x);
    free(batch->manifest_indices);
    free(batch->event_indices);
    free(batch->labels);
    free(batch->event_ids);

    gs_batch_zero(batch);
}


/*
   Restituisce il puntatore al tensore item_index dentro il batch.

   Ogni elemento del batch occupa:
   GS_TENSOR_SIZE = 4 * 224 * 224 float.
*/
float *gs_batch_item_ptr(GSBatch *batch, size_t item_index)
{
    if (batch == NULL) {
        return NULL;
    }

    if (item_index >= batch->capacity) {
        return NULL;
    }

    return batch->x + item_index * GS_TENSOR_SIZE;
}


/*
   Versione const del puntatore al tensore.
*/
const float *gs_batch_item_const_ptr(const GSBatch *batch, size_t item_index)
{
    if (batch == NULL) {
        return NULL;
    }

    if (item_index >= batch->capacity) {
        return NULL;
    }

    return batch->x + item_index * GS_TENSOR_SIZE;
}


/*
   Restituisce la label associata a un elemento del batch.
   La label NON sarà usata per allenare il VAE.
   Serve solo per controlli e valutazione futura.
*/
const char *gs_batch_label(const GSBatch *batch, size_t item_index)
{
    if (batch == NULL || batch->labels == NULL) {
        return NULL;
    }

    if (item_index >= batch->capacity) {
        return NULL;
    }

    return batch->labels + item_index * GS_MAX_LABEL_LEN;
}


/*
   Restituisce event_id associato a un elemento del batch.
*/
const char *gs_batch_event_id(const GSBatch *batch, size_t item_index)
{
    if (batch == NULL || batch->event_ids == NULL) {
        return NULL;
    }

    if (item_index >= batch->capacity) {
        return NULL;
    }

    return batch->event_ids + item_index * GS_MAX_EVENT_ID_LEN;
}


/*
   Carica un batch sequenziale dal manifest.

   Esempio:
   start_index = 0, batch_size = 4

   Carica:
   manifest[0]
   manifest[1]
   manifest[2]
   manifest[3]

   Questo NON fa shuffle.
*/
int gs_batch_load_sequential(
    const GSManifest *manifest,
    size_t start_index,
    size_t batch_size,
    GSBatch *batch
)
{
    size_t available;
    size_t actual_count;
    size_t i;

    if (manifest == NULL) {
        printf("Errore: manifest NULL.\n");
        return 1;
    }

    if (batch == NULL) {
        printf("Errore: batch NULL.\n");
        return 1;
    }

    if (batch_size == 0) {
        printf("Errore: batch_size uguale a zero.\n");
        return 1;
    }

    if (batch_size > batch->capacity) {
        printf("Errore: batch_size maggiore della capacity allocata.\n");
        printf("batch_size = %lu\n", (unsigned long)batch_size);
        printf("capacity   = %lu\n", (unsigned long)batch->capacity);
        return 1;
    }

    if (start_index >= manifest->count) {
        printf("Errore: start_index fuori dal manifest.\n");
        printf("start_index    = %lu\n", (unsigned long)start_index);
        printf("manifest count = %lu\n", (unsigned long)manifest->count);
        return 1;
    }

    available = manifest->count - start_index;

    if (batch_size <= available) {
        actual_count = batch_size;
    } else {
        actual_count = available;
    }

    batch->count = 0;

    for (i = 0; i < actual_count; i++) {
        size_t manifest_index;
        const GSManifestRecord *record;
        float *dst;
        int rc;

        manifest_index = start_index + i;
        record = gs_manifest_get(manifest, manifest_index);

        if (record == NULL) {
            printf("Errore: record NULL al manifest_index %lu.\n",
                   (unsigned long)manifest_index);
            return 1;
        }

        dst = gs_batch_item_ptr(batch, i);
        if (dst == NULL) {
            printf("Errore: puntatore batch NULL all'elemento %lu.\n",
                   (unsigned long)i);
            return 1;
        }

        /*
           Passaggio fondamentale:
           file .bin uint8 -> tensore float normalizzato [0,1].
        */
        rc = gs_manifest_load_tensor(manifest, manifest_index, dst);
        if (rc != 0) {
            printf("Errore: caricamento tensore fallito per manifest_index %lu.\n",
                   (unsigned long)manifest_index);
            return 1;
        }

        batch->manifest_indices[i] = manifest_index;
        batch->event_indices[i] = record->event_index;

        gs_copy_to_slot(batch->labels, GS_MAX_LABEL_LEN, i, record->label);
        gs_copy_to_slot(batch->event_ids, GS_MAX_EVENT_ID_LEN, i, record->event_id);

        batch->count++;
    }

    return 0;
}


/*
   Carica un batch usando una lista di indici.

   Questa funzione serve per lo shuffle.

   Invece di caricare:
   0, 1, 2, 3

   può caricare, per esempio:
   3812, 17, 7900, 204

   Importante:
   vengono mescolati solo gli indici degli eventi.
   Non vengono mescolati pixel, canali o immagini.
*/
int gs_batch_load_from_indices(
    const GSManifest *manifest,
    const size_t *indices,
    size_t index_count,
    size_t start_pos,
    size_t batch_size,
    GSBatch *batch
)
{
    size_t available;
    size_t actual_count;
    size_t i;

    if (manifest == NULL) {
        printf("Errore: manifest NULL.\n");
        return 1;
    }

    if (indices == NULL) {
        printf("Errore: indices NULL.\n");
        return 1;
    }

    if (batch == NULL) {
        printf("Errore: batch NULL.\n");
        return 1;
    }

    if (batch_size == 0) {
        printf("Errore: batch_size uguale a zero.\n");
        return 1;
    }

    if (batch_size > batch->capacity) {
        printf("Errore: batch_size maggiore della capacity allocata.\n");
        printf("batch_size = %lu\n", (unsigned long)batch_size);
        printf("capacity   = %lu\n", (unsigned long)batch->capacity);
        return 1;
    }

    if (start_pos >= index_count) {
        printf("Errore: start_pos fuori dalla lista degli indici.\n");
        printf("start_pos   = %lu\n", (unsigned long)start_pos);
        printf("index_count = %lu\n", (unsigned long)index_count);
        return 1;
    }

    available = index_count - start_pos;

    if (batch_size <= available) {
        actual_count = batch_size;
    } else {
        actual_count = available;
    }

    batch->count = 0;

    for (i = 0; i < actual_count; i++) {
        size_t manifest_index;
        const GSManifestRecord *record;
        float *dst;
        int rc;

        /*
           Qui sta la differenza rispetto al caricamento sequenziale:
           non usiamo start_index + i,
           ma indices[start_pos + i].
        */
        manifest_index = indices[start_pos + i];

        if (manifest_index >= manifest->count) {
            printf("Errore: indice manifest non valido nella lista.\n");
            printf("manifest_index = %lu\n", (unsigned long)manifest_index);
            printf("manifest count = %lu\n", (unsigned long)manifest->count);
            return 1;
        }

        record = gs_manifest_get(manifest, manifest_index);

        if (record == NULL) {
            printf("Errore: record NULL al manifest_index %lu.\n",
                   (unsigned long)manifest_index);
            return 1;
        }

        dst = gs_batch_item_ptr(batch, i);
        if (dst == NULL) {
            printf("Errore: puntatore batch NULL all'elemento %lu.\n",
                   (unsigned long)i);
            return 1;
        }

        /*
           Anche qui:
           file .bin uint8 -> tensore float normalizzato [0,1].
        */
        rc = gs_manifest_load_tensor(manifest, manifest_index, dst);
        if (rc != 0) {
            printf("Errore: caricamento tensore fallito per manifest_index %lu.\n",
                   (unsigned long)manifest_index);
            return 1;
        }

        batch->manifest_indices[i] = manifest_index;
        batch->event_indices[i] = record->event_index;

        gs_copy_to_slot(batch->labels, GS_MAX_LABEL_LEN, i, record->label);
        gs_copy_to_slot(batch->event_ids, GS_MAX_EVENT_ID_LEN, i, record->event_id);

        batch->count++;
    }

    return 0;
}


/*
   Calcola statistiche globali su tutto il batch:
   min, max e mean.

   Queste statistiche servono solo per controllo.
   Il VAE userà tutti i valori del batch, non solo queste tre statistiche.
*/
int gs_batch_compute_stats(const GSBatch *batch, GSBatchStats *stats)
{
    size_t total_values;
    size_t i;

    if (batch == NULL) {
        printf("Errore: batch NULL.\n");
        return 1;
    }

    if (stats == NULL) {
        printf("Errore: stats NULL.\n");
        return 1;
    }

    if (batch->count == 0) {
        printf("Errore: batch vuoto.\n");
        return 1;
    }

    if (batch->x == NULL) {
        printf("Errore: batch->x NULL.\n");
        return 1;
    }

    total_values = batch->count * GS_TENSOR_SIZE;

    stats->min_value = FLT_MAX;
    stats->max_value = -FLT_MAX;
    stats->mean_value = 0.0;

    for (i = 0; i < total_values; i++) {
        float v = batch->x[i];

        if (v < stats->min_value) {
            stats->min_value = v;
        }

        if (v > stats->max_value) {
            stats->max_value = v;
        }

        stats->mean_value += (double)v;
    }

    stats->mean_value /= (double)total_values;

    return 0;
}


/*
   Stampa un riassunto leggibile del batch.
*/
void gs_batch_print_summary(const GSBatch *batch)
{
    GSBatchStats stats;
    size_t i;
    int rc;

    if (batch == NULL) {
        printf("Batch NULL.\n");
        return;
    }

    printf("Batch caricato correttamente.\n");
    printf("Shape logica: %lu x %d x %d x %d\n",
           (unsigned long)batch->count,
           GS_CHANNELS,
           GS_HEIGHT,
           GS_WIDTH);

    printf("Elementi nel batch: %lu\n", (unsigned long)batch->count);
    printf("Capacity allocata:  %lu\n", (unsigned long)batch->capacity);
    printf("Valori per evento:  %lu\n", (unsigned long)GS_TENSOR_SIZE);
    printf("Valori totali:      %lu\n",
           (unsigned long)(batch->count * GS_TENSOR_SIZE));

    rc = gs_batch_compute_stats(batch, &stats);
    if (rc == 0) {
        printf("Statistiche globali batch:\n");
        printf("  min  = %.6f\n", stats.min_value);
        printf("  max  = %.6f\n", stats.max_value);
        printf("  mean = %.6f\n", stats.mean_value);
    }

    printf("\nEventi nel batch:\n");

    for (i = 0; i < batch->count; i++) {
        printf("  [%lu] manifest_index=%lu, event_index=%d, label=%s, event_id=%s\n",
               (unsigned long)i,
               (unsigned long)batch->manifest_indices[i],
               batch->event_indices[i],
               gs_batch_label(batch, i),
               gs_batch_event_id(batch, i));
    }
}