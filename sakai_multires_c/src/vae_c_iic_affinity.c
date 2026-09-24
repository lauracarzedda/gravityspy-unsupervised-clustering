#include "vae_c_iic_affinity.h"

#include <math.h>
#include <stdint.h>

static VAECOpStatus preflight(
    const float *h, size_t rows, size_t cols,
    size_t first_row, size_t row_count,
    const double *affinity, size_t affinity_capacity
)
{
    size_t h_count, a_count, i;

    if (h == NULL || affinity == NULL ||
        (const void *)h == (const void *)affinity ||
        rows == 0u || cols == 0u || row_count == 0u ||
        first_row >= rows || row_count > rows - first_row)
        return VAE_C_OP_INVALID_ARGUMENT;

    if (rows > SIZE_MAX / cols || rows > SIZE_MAX / rows)
        return VAE_C_OP_OVERFLOW;
    h_count = rows * cols;
    a_count = rows * rows;
    if (h_count > SIZE_MAX / sizeof(float) ||
        a_count > SIZE_MAX / sizeof(double))
        return VAE_C_OP_OVERFLOW;
    if (affinity_capacity < a_count)
        return VAE_C_OP_SHAPE_MISMATCH;

    for (i = 0u; i < h_count; i++) {
        if (!isfinite(h[i])) return VAE_C_OP_INVALID_ARGUMENT;
    }
    return VAE_C_OP_OK;
}

VAECOpStatus vae_c_iic_affinity_compute_rows(
    const float *h, size_t rows, size_t cols,
    size_t first_row, size_t row_count,
    double *affinity, size_t affinity_capacity
)
{
    VAECOpStatus status = preflight(h, rows, cols, first_row, row_count,
                                   affinity, affinity_capacity);
    size_t i, j, d;

    if (status != VAE_C_OP_OK) return status;

    for (i = first_row; i < first_row + row_count; i++) {
        affinity[i * rows + i] = 1.0;
        for (j = i + 1u; j < rows; j++) {
            double sum = 0.0;
            double distance, value;
            for (d = 0u; d < cols; d++) {
                double difference = (double)h[i * cols + d] -
                                    (double)h[j * cols + d];
                sum += difference * difference;
            }
            if (!isfinite(sum)) return VAE_C_OP_OVERFLOW;
            distance = sqrt(sum);
            value = exp(-distance);
            if (!isfinite(value) || value <= 0.0 || value > 1.0)
                return VAE_C_OP_OVERFLOW;
            affinity[i * rows + j] = value;
            affinity[j * rows + i] = value;
        }
    }
    return VAE_C_OP_OK;
}

VAECOpStatus vae_c_iic_affinity_compute(
    const float *h, size_t rows, size_t cols,
    double *affinity, size_t affinity_capacity
)
{
    return vae_c_iic_affinity_compute_rows(h, rows, cols, 0u, rows,
                                           affinity, affinity_capacity);
}
