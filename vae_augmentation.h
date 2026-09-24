#ifndef VAE_AUGMENTATION_H
#define VAE_AUGMENTATION_H

#include <stdint.h>

#include "vae_event_loader.h"


/*
   Generatore casuale dedicato alla perturbazione temporale.

   È separato dal seed usato per lo split train/validation.
*/
typedef struct {
    uint32_t state;
} GSAugmentRng;


/*
   Politiche di perturbazione disponibili.

   LEGACY_TEXT:
       crop_left = 0 ... 24
       Mantiene esattamente il comportamento precedente.

   SYMMETRIC_FIGURE:
       crop_left = 0 ... 48
       Il centro è 24.
       L'offset della finestra rispetto al centro è -24 ... +24.
*/
typedef enum {
    GS_AUGMENT_MODE_LEGACY_TEXT = 0,
    GS_AUGMENT_MODE_SYMMETRIC_FIGURE = 1
} GSAugmentMode;


/*
   Restituisce il nome leggibile della modalità.
*/
const char *gs_augment_mode_name(
    GSAugmentMode mode
);


/*
   Restituisce l'intervallo inclusivo di crop_left
   associato alla modalità.

   Ritorna:
   0 = successo
   diverso da 0 = modalità o puntatori non validi
*/
int gs_augment_mode_crop_range(
    GSAugmentMode mode,
    int *min_crop_left,
    int *max_crop_left
);


/*
   Policy augmentation_stateless_v1, deterministica per evento/epoca.

   La chiave e':
       domain tag, base_seed, epoch, manifest_index, mode, stream index.

   Non conserva stato e non dipende dall'ordine o dalla composizione
   del batch. Il mapping all'intervallo del mode usa rejection sampling
   uint64_t senza modulo bias.

   crop_left viene modificato soltanto in caso di successo.

   Ritorna:
   0 = successo
   diverso da 0 = argomento o modalita' non validi
*/
int gs_augment_crop_left_for_event_mode(
    uint64_t base_seed,
    uint64_t epoch,
    uint64_t manifest_index,
    GSAugmentMode mode,
    int *crop_left
);


/*
   Inizializza il generatore con un seed.

   Stesso seed:
   stessa sequenza di crop casuali.
*/
void gs_augment_rng_seed(
    GSAugmentRng *rng,
    uint32_t seed
);


/*
   Genera il prossimo numero pseudo-casuale uint32.
*/
uint32_t gs_augment_rng_next(
    GSAugmentRng *rng
);


/*
   Estrae un crop_left casuale secondo la modalità richiesta.
*/
int gs_augment_random_crop_left_mode(
    GSAugmentRng *rng,
    GSAugmentMode mode,
    int *crop_left
);


/*
   Wrapper di compatibilità.

   Mantiene il comportamento storico:
       crop_left = 0 ... 24
*/
int gs_augment_random_crop_left(
    GSAugmentRng *rng,
    int *crop_left
);


/*
   Sceglie un solo crop_left casuale secondo la modalità
   e lo applica alle quattro viste dello stesso evento.

   Produce:
       4 x 224 x 224 float [0,1]
*/
int gs_event_load_random_crop_float01_mode(
    const GSEventRecord *record,
    GSAugmentRng *rng,
    GSAugmentMode mode,
    int *used_crop_left,
    float *out_f32
);


/*
   Wrapper di compatibilità.

   Mantiene il comportamento storico:
       modalità LEGACY_TEXT
       crop_left = 0 ... 24
*/
int gs_event_load_random_crop_float01(
    const GSEventRecord *record,
    GSAugmentRng *rng,
    int *used_crop_left,
    float *out_f32
);

#endif
