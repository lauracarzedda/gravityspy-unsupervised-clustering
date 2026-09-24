#ifndef BASELINE_PCA_H
#define BASELINE_PCA_H

#include <stddef.h>


enum {
    GS_BASELINE_PCA_SUCCESS = 0,
    GS_BASELINE_PCA_ERROR_NULL_ARGUMENT = 1,
    GS_BASELINE_PCA_ERROR_DIMENSIONS = 2,
    GS_BASELINE_PCA_ERROR_OVERFLOW = 3,
    GS_BASELINE_PCA_ERROR_NONFINITE = 4,
    GS_BASELINE_PCA_ERROR_ALLOCATION = 5,
    GS_BASELINE_PCA_ERROR_LAPACKE = 6,
    GS_BASELINE_PCA_ERROR_NUMERIC = 7,
    GS_BASELINE_PCA_ERROR_MODEL = 8,
    GS_BASELINE_PCA_ERROR_BUFFER = 9
};


typedef struct {
    size_t input_dim;
    size_t component_count;

    double *mean;
    double *components;
    double *eigenvalues;
    double *explained_variance_ratio;
    double *cumulative_explained_variance_ratio;

    double total_variance;

    size_t components_90;
    size_t components_95;
    size_t components_99;
} GSBaselinePCA;


/*
   Esegue un fit PCA completo su una matrice float32 row-major.
   La covarianza usa il divisore rows (ddof=0). Il modello deve essere
   inizializzato a zero oppure contenere un precedente modello valido.
   Un errore non modifica un precedente modello valido.
*/
int gs_baseline_pca_fit(
    const float *x,
    size_t rows,
    size_t cols,
    GSBaselinePCA *model
);


/*
   Trasforma una matrice float32 row-major senza modificare il modello.
   scores e' row-major con dimensioni rows x requested_components.
*/
int gs_baseline_pca_transform(
    const GSBaselinePCA *model,
    const float *x,
    size_t rows,
    size_t cols,
    size_t requested_components,
    float *scores,
    size_t scores_count
);


/* Restituisce il numero minimo di componenti per una soglia in (0, 1]. */
int gs_baseline_pca_components_for_threshold(
    const GSBaselinePCA *model,
    double threshold,
    size_t *component_count
);


/* Libera tutti gli output, azzera la struttura ed e' idempotente. */
void gs_baseline_pca_free(
    GSBaselinePCA *model
);

#endif
