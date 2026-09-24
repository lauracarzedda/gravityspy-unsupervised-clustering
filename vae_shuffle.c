#include "vae_shuffle.h"

#include <stdio.h>
#include <stdlib.h>


int gs_indices_alloc(size_t **indices, size_t count)
{
    if (indices == NULL) {
        printf("Errore: indices NULL.\n");
        return 1;
    }

    if (count == 0) {
        printf("Errore: count uguale a zero.\n");
        return 1;
    }

    *indices = (size_t *)malloc(count * sizeof(size_t));

    if (*indices == NULL) {
        printf("Errore: memoria insufficiente per gli indici.\n");
        return 1;
    }

    return 0;
}


void gs_indices_fill_sequential(size_t *indices, size_t count)
{
    size_t i;

    if (indices == NULL) {
        return;
    }

    for (i = 0; i < count; i++) {
        indices[i] = i;
    }
}


void gs_indices_shuffle(size_t *indices, size_t count, unsigned int seed)
{
    size_t i;

    if (indices == NULL) {
        return;
    }

    if (count <= 1) {
        return;
    }

    srand(seed);

    /*
       Fisher-Yates shuffle.
       Mescola solo gli indici degli eventi.
       Non tocca immagini, pixel o canali.
    */
    for (i = count - 1; i > 0; i--) {
        size_t j;
        size_t tmp;

        j = (size_t)(rand() % (int)(i + 1));

        tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }
}