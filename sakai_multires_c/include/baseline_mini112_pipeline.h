#ifndef BASELINE_MINI112_PIPELINE_H
#define BASELINE_MINI112_PIPELINE_H

#include <stddef.h>


enum {
    GS_BASELINE_MINI112_CHANNELS = 4,
    GS_BASELINE_MINI112_SIDE = 112,
    GS_BASELINE_MINI112_POOL_SIDE = 7,
    GS_BASELINE_MINI112_POOLED_SIDE =
        GS_BASELINE_MINI112_SIDE / GS_BASELINE_MINI112_POOL_SIDE,
    GS_BASELINE_MINI112_INPUT_COUNT =
        GS_BASELINE_MINI112_CHANNELS *
        GS_BASELINE_MINI112_SIDE *
        GS_BASELINE_MINI112_SIDE,
    GS_BASELINE_MINI112_TENSOR_BYTES =
        GS_BASELINE_MINI112_INPUT_COUNT,
    GS_BASELINE_MINI112_FEATURE_COUNT =
        GS_BASELINE_MINI112_CHANNELS *
        GS_BASELINE_MINI112_POOLED_SIDE *
        GS_BASELINE_MINI112_POOLED_SIDE,
    GS_BASELINE_MINI112_OFFICIAL_EVENT_COUNT = 8583,
    GS_BASELINE_MINI112_OFFICIAL_TRAIN_COUNT = 6866,
    GS_BASELINE_MINI112_OFFICIAL_VALIDATION_COUNT = 1717,
    GS_BASELINE_MINI112_PATH_LENGTH = 4096
};


enum {
    GS_BASELINE_MINI112_PIPELINE_SUCCESS = 0,
    GS_BASELINE_MINI112_PIPELINE_ERROR_NULL_ARGUMENT = 1,
    GS_BASELINE_MINI112_PIPELINE_ERROR_STATE = 2,
    GS_BASELINE_MINI112_PIPELINE_ERROR_ALLOCATION = 3,
    GS_BASELINE_MINI112_PIPELINE_ERROR_SPEC = 4,
    GS_BASELINE_MINI112_PIPELINE_ERROR_EVENT_MANIFEST = 5,
    GS_BASELINE_MINI112_PIPELINE_ERROR_TENSOR_MANIFEST = 6,
    GS_BASELINE_MINI112_PIPELINE_ERROR_TRAIN_SPLIT = 7,
    GS_BASELINE_MINI112_PIPELINE_ERROR_VALIDATION_SPLIT = 8,
    GS_BASELINE_MINI112_PIPELINE_ERROR_COUNTS = 9,
    GS_BASELINE_MINI112_PIPELINE_ERROR_ALIGNMENT = 10,
    GS_BASELINE_MINI112_PIPELINE_ERROR_DUPLICATE_PATH = 11,
    GS_BASELINE_MINI112_PIPELINE_ERROR_PATH = 12,
    GS_BASELINE_MINI112_PIPELINE_ERROR_FILE_NOT_FOUND = 13,
    GS_BASELINE_MINI112_PIPELINE_ERROR_FILE_SIZE = 14,
    GS_BASELINE_MINI112_PIPELINE_ERROR_RANGE = 15,
    GS_BASELINE_MINI112_PIPELINE_ERROR_OVERFLOW = 16,
    GS_BASELINE_MINI112_PIPELINE_ERROR_BUFFER_SIZE = 17,
    GS_BASELINE_MINI112_PIPELINE_ERROR_TENSOR_READ = 18,
    GS_BASELINE_MINI112_PIPELINE_ERROR_TENSOR_VALUE = 19,
    GS_BASELINE_MINI112_PIPELINE_ERROR_FEATURE_EXTRACTION = 20
};


/*
   Le funzioni che restituiscono int usano SUCCESS oppure uno dei codici
   di errore dichiarati sopra. Il testo diagnostico associato e'
   disponibile tramite gs_baseline_mini112_pipeline_error_string.
*/


/*
   Contesto opaco. Puo' essere inizializzato anche come {0}.
   Non espone label, manifest o path interni.
*/
typedef struct {
    void *implementation;
} GSBaselineMini112Pipeline;


/*
   Inizializza un contesto nuovo. Un contesto gia' aperto deve essere
   prima liberato con gs_baseline_mini112_pipeline_free.
*/
void gs_baseline_mini112_pipeline_init(
    GSBaselineMini112Pipeline *pipeline
);


/*
   Carica e valida globalmente manifest eventi, manifest mini112 e split
   ufficiali. I percorsi sono forniti dal chiamante. data_root viene
   anteposto esclusivamente ai tensor_path del manifest mini112.

   pipeline deve essere stato inizializzato con {0} oppure con init.
   Una seconda open senza free restituisce ERROR_STATE e lascia il
   contesto aperto invariato. Se open fallisce, il contesto resta vuoto,
   coerente e liberabile; dopo free lo stesso contesto puo' essere aperto
   nuovamente.

   La validazione controlla conteggi, specifica tensoriale, allineamento
   degli identificativi non-label, copertura e disgiunzione degli split,
   unicita' dei tensor_path, presenza e dimensione di tutti i tensori.
   I payload tensoriali non vengono letti durante questa funzione.
*/
int gs_baseline_mini112_pipeline_open(
    GSBaselineMini112Pipeline *pipeline,
    const char *event_manifest_path,
    const char *tensor_manifest_path,
    const char *train_split_path,
    const char *validation_split_path,
    const char *data_root
);


/*
   Restituisce i conteggi validati senza esporre metadati o label.
   Richiede un contesto aperto; altrimenti restituisce ERROR_STATE.
*/
int gs_baseline_mini112_pipeline_counts(
    const GSBaselineMini112Pipeline *pipeline,
    size_t *event_count,
    size_t *train_count,
    size_t *validation_count
);


/*
   Restituiscono il manifest_index zero-based associato a split_pos.
   Sono funzioni diagnostiche non-label e non restituiscono tensor_path.
   Richiedono un contesto aperto e split_pos nell'intervallo dello split.
*/
int gs_baseline_mini112_train_manifest_index(
    const GSBaselineMini112Pipeline *pipeline,
    size_t split_pos,
    size_t *manifest_index
);

int gs_baseline_mini112_validation_manifest_index(
    const GSBaselineMini112Pipeline *pipeline,
    size_t split_pos,
    size_t *manifest_index
);


/*
   Estraggono un intervallo contiguo preservando l'ordine ufficiale dello
   split. Richiedono un contesto aperto. L'output e' una matrice
   row-major [row_count][GS_BASELINE_MINI112_FEATURE_COUNT]. Ogni riga
   contiene esattamente
   GS_BASELINE_MINI112_FEATURE_COUNT float in ordine:

       channel, pooled_row, pooled_column.

   row_count == 0 viene rifiutato con ERROR_RANGE. feature_capacity e'
   espressa in numero di float e deve essere almeno
   row_count * GS_BASELINE_MINI112_FEATURE_COUNT.

   La pipeline mantiene soltanto il buffer temporaneo di un tensore e
   non alloca la matrice di output.

   Una chiamata multi-riga non e' atomica: se fallisce dopo una o piu'
   righe completate, il buffer di output puo' essere parzialmente scritto
   e deve essere considerato non valido nel suo complesso.
*/
int gs_baseline_mini112_extract_train_rows(
    GSBaselineMini112Pipeline *pipeline,
    size_t first_row,
    size_t row_count,
    float *features,
    size_t feature_capacity
);

int gs_baseline_mini112_extract_validation_rows(
    GSBaselineMini112Pipeline *pipeline,
    size_t first_row,
    size_t row_count,
    float *features,
    size_t feature_capacity
);


/*
   Restituisce un messaggio diagnostico statico per un codice pipeline.
*/
const char *gs_baseline_mini112_pipeline_error_string(int error_code);


/*
   Libera tutte le risorse, azzera il contesto ed e' idempotente su un
   contesto inizializzato, aperto o gia' liberato.
*/
void gs_baseline_mini112_pipeline_free(
    GSBaselineMini112Pipeline *pipeline
);

#endif
