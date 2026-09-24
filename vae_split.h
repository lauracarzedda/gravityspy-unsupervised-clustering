#ifndef VAE_SPLIT_H
#define VAE_SPLIT_H

#include <stddef.h>

#include "vae_dataset.h"


/*
   Split train/validation basato sugli indici del manifest.

   Importante:
   gli indici rappresentano eventi completi.
   Non stiamo dividendo singole immagini.
*/
typedef struct {
    size_t *train_indices;
    size_t train_count;

    size_t *val_indices;
    size_t val_count;

    size_t total_count;
    double train_rate;
    unsigned int seed;
} GSSplit;


/*
   Crea uno split train/validation.

   Esempio:
   train_rate = 0.8
   seed = 123

   Procedura:
   1. crea indici 0,1,2,...
   2. li mescola
   3. assegna i primi 80% al train
   4. assegna il restante 20% alla validation
*/
int gs_split_train_val(
    const GSManifest *manifest,
    double train_rate,
    unsigned int seed,
    GSSplit *split
);


/*
   Libera memoria dello split.
*/
void gs_split_free(GSSplit *split);


/*
   Stampa un riassunto dello split.
   Le label vengono stampate solo per controllo.
*/
void gs_split_print_summary(
    const GSManifest *manifest,
    const GSSplit *split,
    size_t preview_count
);


/*
   Salva una lista di indici su CSV.

   Nota importante:
   NON salva le label.
   Salva solo informazioni tecniche utili per ricostruire lo split.
*/
int gs_split_save_indices_csv(
    const char *csv_path,
    const char *split_name,
    const GSManifest *manifest,
    const size_t *indices,
    size_t count
);


/*
   Salva train e validation split in due CSV separati.
*/
int gs_split_save_train_val_csv(
    const GSManifest *manifest,
    const GSSplit *split,
    const char *train_csv_path,
    const char *val_csv_path
);

#endif