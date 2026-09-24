#ifndef BASELINE_AVERAGE_POOL_FEATURES_H
#define BASELINE_AVERAGE_POOL_FEATURES_H

#include <stddef.h>


enum {
    GS_BASELINE_AVERAGE_POOL_SUCCESS = 0,
    GS_BASELINE_AVERAGE_POOL_ERROR_NULL_ARGUMENT = 1,
    GS_BASELINE_AVERAGE_POOL_ERROR_DIMENSIONS = 2,
    GS_BASELINE_AVERAGE_POOL_ERROR_DIVISIBILITY = 3,
    GS_BASELINE_AVERAGE_POOL_ERROR_OVERFLOW = 4,
    GS_BASELINE_AVERAGE_POOL_ERROR_BUFFER_SIZE = 5
};


/*
   Calcola le dimensioni di un average pooling quadrato non sovrapposto.

   L'immagine e' quadrata, lo stride coincide con pool_side e non viene
   applicato padding. image_side deve quindi essere divisibile per
   pool_side.

   In caso di successo:
   - input_count e' channels * image_side * image_side;
   - pooled_side e' image_side / pool_side;
   - feature_count e' channels * pooled_side * pooled_side.

   Gli output non vengono modificati in caso di errore.
*/
int gs_baseline_average_pool_dimensions(
    size_t channels,
    size_t image_side,
    size_t pool_side,
    size_t *input_count,
    size_t *pooled_side,
    size_t *feature_count
);


/*
   Estrae feature tramite average pooling quadrato non sovrapposto.

   input e' channel-major:
       channel, input_row, input_column.

   features mantiene lo stesso ordinamento:
       channel, pooled_row, pooled_column.

   L'accumulazione avviene in double e il risultato viene salvato in
   float. Il kernel non alloca memoria e non modifica input. Le regioni
   di memoria di input e features non devono sovrapporsi; l'uso in-place
   non e' supportato.

   input_count deve coincidere con la dimensione calcolata da
   gs_baseline_average_pool_dimensions. feature_capacity deve essere
   almeno pari al feature_count calcolato dalla stessa funzione.
*/
int gs_baseline_average_pool_extract(
    const float *input,
    size_t input_count,
    size_t channels,
    size_t image_side,
    size_t pool_side,
    float *features,
    size_t feature_capacity
);

#endif
