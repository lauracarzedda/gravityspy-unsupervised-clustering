#ifndef BASELINE_MINI75_FEATURES_H
#define BASELINE_MINI75_FEATURES_H

#include <stddef.h>


enum {
    GS_BASELINE_MINI75_CHANNELS = 4,
    GS_BASELINE_MINI75_SIDE = 75,
    GS_BASELINE_MINI75_POOL_SIDE = 5,
    GS_BASELINE_MINI75_POOLED_SIDE = 15,
    GS_BASELINE_MINI75_INPUT_COUNT = 22500,
    GS_BASELINE_MINI75_FEATURE_COUNT = 900
};


enum {
    GS_BASELINE_MINI75_SUCCESS = 0,
    GS_BASELINE_MINI75_ERROR_NULL_ARGUMENT = 1,
    GS_BASELINE_MINI75_ERROR_COUNT = 2
};


/*
   Estrae 900 feature tramite average pooling 5x5 non sovrapposto,
   separatamente sui quattro canali di un tensore channel-major 4x75x75.

   I buffer sono forniti dal chiamante e non viene allocata memoria.
   Il buffer input non viene modificato.

   Ritorna:
   GS_BASELINE_MINI75_SUCCESS             successo
   GS_BASELINE_MINI75_ERROR_NULL_ARGUMENT input o features NULL
   GS_BASELINE_MINI75_ERROR_COUNT         dimensione input/output errata
*/
int gs_baseline_mini75_pool5_extract(
    const float *input,
    size_t input_count,
    float *features,
    size_t feature_count
);

#endif
