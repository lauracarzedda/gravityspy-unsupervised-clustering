#ifndef BASELINE_CLUSTERING_METRICS_H
#define BASELINE_CLUSTERING_METRICS_H

#include <stddef.h>
#include <stdint.h>


enum {
    GS_BASELINE_CLUSTERING_METRICS_SUCCESS = 0,
    GS_BASELINE_CLUSTERING_METRICS_ERROR_NULL_ARGUMENT = 1,
    GS_BASELINE_CLUSTERING_METRICS_ERROR_DIMENSIONS = 2,
    GS_BASELINE_CLUSTERING_METRICS_ERROR_RANGE = 3,
    GS_BASELINE_CLUSTERING_METRICS_ERROR_OVERFLOW = 4,
    GS_BASELINE_CLUSTERING_METRICS_ERROR_ALLOCATION = 5,
    GS_BASELINE_CLUSTERING_METRICS_ERROR_NUMERIC = 6
};


typedef struct {
    size_t rows;
    size_t label_count;
    size_t cluster_count;

    uint64_t *contingency;
    uint64_t *label_totals;
    uint64_t *cluster_totals;

    double purity;
    double normalized_mutual_information;
    double adjusted_rand_index;
    double homogeneity;
    double completeness;
} GSBaselineClusteringMetrics;


/*
   Calcola metriche esterne da label vere e assegnazioni cluster uint32_t.
   La contingenza risultante e' label-major row-major, con indice
   label * cluster_count + cluster. Gli input non vengono modificati.

   result viene sempre inizializzato a uno stato sicuro quando non e'
   NULL. Un risultato precedente deve essere liberato prima di riusare
   la stessa struttura.
*/
int gs_baseline_clustering_metrics_compute(
    const uint32_t *labels,
    const uint32_t *clusters,
    size_t rows,
    size_t label_count,
    size_t cluster_count,
    GSBaselineClusteringMetrics *result
);


/* Libera gli output, azzera tutti i campi ed e' idempotente. */
void gs_baseline_clustering_metrics_free(
    GSBaselineClusteringMetrics *result
);

#endif
