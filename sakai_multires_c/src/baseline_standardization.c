#include "baseline_standardization.h"

#include <math.h>
#include <stdint.h>


static int gs_standardization_dimensions_valid(
    size_t rows,
    size_t cols
)
{
    if (rows == 0 || cols == 0) {
        return 0;
    }

    if (rows > SIZE_MAX / cols) {
        return 0;
    }

    return 1;
}


static int gs_standardization_input_is_finite(
    const float *x,
    size_t value_count
)
{
    size_t index;

    for (index = 0; index < value_count; index++) {
        if (!isfinite(x[index])) {
            return 0;
        }
    }

    return 1;
}


int gs_baseline_standardization_fit(
    const float *x,
    size_t rows,
    size_t cols,
    double *mean,
    double *stddev,
    unsigned char *constant_mask,
    double constant_epsilon
)
{
    size_t row;
    size_t col;
    size_t value_count;

    if (x == NULL || mean == NULL || stddev == NULL ||
        constant_mask == NULL) {

        return GS_BASELINE_STANDARDIZATION_ERROR_NULL_ARGUMENT;
    }

    if (!gs_standardization_dimensions_valid(rows, cols)) {
        return GS_BASELINE_STANDARDIZATION_ERROR_DIMENSIONS;
    }

    if (!isfinite(constant_epsilon) || constant_epsilon < 0.0) {
        return GS_BASELINE_STANDARDIZATION_ERROR_EPSILON;
    }

    value_count = rows * cols;
    if (!gs_standardization_input_is_finite(x, value_count)) {
        return GS_BASELINE_STANDARDIZATION_ERROR_NONFINITE_INPUT;
    }

    for (col = 0; col < cols; col++) {
        mean[col] = 0.0;
        stddev[col] = 0.0;
        constant_mask[col] = 0;
    }

    /*
       Welford colonna per colonna.
       Durante il ciclo stddev contiene M2; dopo il ciclo contiene
       la deviazione standard population.
    */
    for (row = 0; row < rows; row++) {
        double count = (double)(row + 1);

        for (col = 0; col < cols; col++) {
            double value = (double)x[row * cols + col];
            double delta = value - mean[col];
            double delta_after_update;

            mean[col] += delta / count;
            delta_after_update = value - mean[col];
            stddev[col] += delta * delta_after_update;
        }
    }

    for (col = 0; col < cols; col++) {
        double variance = stddev[col] / (double)rows;

        if (!isfinite(mean[col]) ||
            !isfinite(variance) ||
            variance < 0.0) {

            return GS_BASELINE_STANDARDIZATION_ERROR_NUMERIC;
        }

        stddev[col] = sqrt(variance);

        if (!isfinite(stddev[col])) {
            return GS_BASELINE_STANDARDIZATION_ERROR_NUMERIC;
        }

        constant_mask[col] =
            stddev[col] <= constant_epsilon ? 1u : 0u;
    }

    return GS_BASELINE_STANDARDIZATION_SUCCESS;
}


int gs_baseline_standardization_transform(
    const float *x,
    size_t rows,
    size_t cols,
    const double *mean,
    const double *stddev,
    const unsigned char *constant_mask,
    float *output
)
{
    size_t row;
    size_t col;
    size_t value_count;

    if (x == NULL || mean == NULL || stddev == NULL ||
        constant_mask == NULL || output == NULL) {

        return GS_BASELINE_STANDARDIZATION_ERROR_NULL_ARGUMENT;
    }

    if (!gs_standardization_dimensions_valid(rows, cols)) {
        return GS_BASELINE_STANDARDIZATION_ERROR_DIMENSIONS;
    }

    value_count = rows * cols;
    if (!gs_standardization_input_is_finite(x, value_count)) {
        return GS_BASELINE_STANDARDIZATION_ERROR_NONFINITE_INPUT;
    }

    for (col = 0; col < cols; col++) {
        if (!isfinite(mean[col]) ||
            !isfinite(stddev[col]) ||
            stddev[col] < 0.0 ||
            constant_mask[col] > 1u ||
            (constant_mask[col] == 0u && stddev[col] <= 0.0)) {

            return GS_BASELINE_STANDARDIZATION_ERROR_INVALID_SCALER;
        }
    }

    for (row = 0; row < rows; row++) {
        for (col = 0; col < cols; col++) {
            size_t index = row * cols + col;

            if (constant_mask[col] != 0u) {
                output[index] = 0.0f;
            } else {
                double transformed =
                    ((double)x[index] - mean[col]) / stddev[col];
                float transformed_float;

                if (!isfinite(transformed)) {
                    return GS_BASELINE_STANDARDIZATION_ERROR_NUMERIC;
                }

                transformed_float = (float)transformed;
                if (!isfinite(transformed_float)) {
                    return GS_BASELINE_STANDARDIZATION_ERROR_NUMERIC;
                }

                output[index] = transformed_float;
            }
        }
    }

    return GS_BASELINE_STANDARDIZATION_SUCCESS;
}
