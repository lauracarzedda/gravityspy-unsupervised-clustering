#ifndef BASELINE_KMEANS_H
#define BASELINE_KMEANS_H

#include <stddef.h>
#include <stdint.h>


enum {
    GS_BASELINE_KMEANS_SUCCESS = 0,
    GS_BASELINE_KMEANS_ERROR_NULL_ARGUMENT = 1,
    GS_BASELINE_KMEANS_ERROR_DIMENSIONS = 2,
    GS_BASELINE_KMEANS_ERROR_OVERFLOW = 3,
    GS_BASELINE_KMEANS_ERROR_NONFINITE_INPUT = 4,
    GS_BASELINE_KMEANS_ERROR_ALLOCATION = 5,
    GS_BASELINE_KMEANS_ERROR_NUMERIC = 6
};


typedef struct {
    size_t rows;
    size_t cols;
    size_t k;
    size_t iterations;
    size_t empty_cluster_reseeds;
    int converged;
    uint64_t seed;
    double initial_inertia;
    double inertia;

    double *centroids;
    size_t *assignments;
    size_t *cluster_sizes;
    size_t *initial_centroid_indices;
} GSBaselineKMeansResult;


/*
   Esegue k-means++ deterministico seguito dalle iterazioni di Lloyd.
   L'input x e' una matrice float row-major rows x cols e non viene
   modificato. Il risultato e il workspace sono allocati internamente.

   result viene inizializzato a uno stato sicuro prima dell'uso. Un
   risultato precedente deve essere liberato dal chiamante prima di
   riutilizzare la stessa struttura.
*/
int gs_baseline_kmeans_fit(
    const float *x,
    size_t rows,
    size_t cols,
    size_t k,
    uint64_t seed,
    size_t max_iterations,
    GSBaselineKMeansResult *result
);


/*
   Assegna ogni riga al centro piu' vicino usando distanza euclidea
   quadratica in double. In caso di parita' vince il cluster con indice
   minore. squared_distances puo' essere NULL.
*/
int gs_baseline_kmeans_assign(
    const float *x,
    size_t rows,
    size_t cols,
    const double *centroids,
    size_t k,
    size_t *assignments,
    double *squared_distances
);


/*
   Libera tutti gli output, azzera i campi ed e' idempotente su un
   risultato valido o gia' liberato.
*/
void gs_baseline_kmeans_result_free(
    GSBaselineKMeansResult *result
);

#endif
