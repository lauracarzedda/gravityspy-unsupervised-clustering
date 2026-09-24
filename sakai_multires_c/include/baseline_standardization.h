#ifndef BASELINE_STANDARDIZATION_H
#define BASELINE_STANDARDIZATION_H

#include <stddef.h>


#define GS_BASELINE_STANDARDIZATION_DEFAULT_CONSTANT_EPSILON 1.0e-12

enum {
    GS_BASELINE_STANDARDIZATION_DDOF = 0
};


enum {
    GS_BASELINE_STANDARDIZATION_SUCCESS = 0,
    GS_BASELINE_STANDARDIZATION_ERROR_NULL_ARGUMENT = 1,
    GS_BASELINE_STANDARDIZATION_ERROR_DIMENSIONS = 2,
    GS_BASELINE_STANDARDIZATION_ERROR_EPSILON = 3,
    GS_BASELINE_STANDARDIZATION_ERROR_NONFINITE_INPUT = 4,
    GS_BASELINE_STANDARDIZATION_ERROR_INVALID_SCALER = 5,
    GS_BASELINE_STANDARDIZATION_ERROR_NUMERIC = 6
};


/*
   Calcola media e deviazione standard population (ddof = 0),
   colonna per colonna, su una matrice float row-major.

   Una colonna e' marcata costante quando:
       stddev[j] <= constant_epsilon

   Non viene allocata memoria internamente e x non viene modificato.

   Nella pipeline reale:
   1. fit viene eseguito soltanto sulle 6866 righe train;
   2. lo stesso scaler viene applicato alle righe train;
   3. lo stesso scaler, senza refit, viene applicato alle 1717 righe
      validation.

   Ritorna GS_BASELINE_STANDARDIZATION_SUCCESS in caso di successo;
   gli altri codici documentano rispettivamente argomenti NULL,
   dimensioni non valide, epsilon non valido, input non finito,
   scaler non valido o errore numerico.
*/
int gs_baseline_standardization_fit(
    const float *x,
    size_t rows,
    size_t cols,
    double *mean,
    double *stddev,
    unsigned char *constant_mask,
    double constant_epsilon
);


/*
   Applica a una matrice float row-major uno scaler gia' calcolato.

   Per colonne costanti produce 0. Per le altre produce:
       (x[i,j] - mean[j]) / stddev[j]

   Non ricalcola statistiche e non modifica x, mean, stddev o
   constant_mask. Non viene allocata memoria internamente.
*/
int gs_baseline_standardization_transform(
    const float *x,
    size_t rows,
    size_t cols,
    const double *mean,
    const double *stddev,
    const unsigned char *constant_mask,
    float *output
);

#endif
