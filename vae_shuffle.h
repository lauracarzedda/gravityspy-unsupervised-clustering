#ifndef VAE_SHUFFLE_H
#define VAE_SHUFFLE_H

#include <stddef.h>

/*
   Alloca una lista di indici.
*/
int gs_indices_alloc(size_t **indices, size_t count);

/*
   Riempie la lista con:
   0, 1, 2, 3, ...
*/
void gs_indices_fill_sequential(size_t *indices, size_t count);

/*
   Mescola gli indici con Fisher-Yates shuffle.
   Il seed serve per rendere il test riproducibile.
*/
void gs_indices_shuffle(size_t *indices, size_t count, unsigned int seed);

#endif