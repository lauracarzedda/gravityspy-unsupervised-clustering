#ifndef VAE_C_IIC_AFFINITY_H
#define VAE_C_IIC_AFFINITY_H

#include "vae_c_ops.h"

#include <stddef.h>

/* Dense A[i,j] = exp(-||H[i,:] - H[j,:]||_2). H is row-major float32;
 * A is caller-owned row-major float64. Capacity is measured in doubles.
 * Each input float is converted to double before subtraction. The diagonal
 * is set to exactly 1.0 and every lower-triangle entry is copied from its
 * upper-triangle counterpart, never independently recomputed.
 *
 * The caller may place A directly in a mapped output file: this core does
 * not allocate H64, distance, affinity, or block-sized temporary arrays.
 * H and A must be separate objects. Exact pointer equality is rejected;
 * partial overlap of different C objects is not detected.
 *
 * Invalid shape/capacity and non-finite H are rejected before any output
 * write. For extreme finite H, exp(-distance) can underflow to zero; the
 * function returns OVERFLOW instead of clipping or publishing zero. In that
 * exceptional case A may have been partially written and must be discarded.
 */
VAECOpStatus vae_c_iic_affinity_compute(
    const float *h,
    size_t rows,
    size_t cols,
    double *affinity,
    size_t affinity_capacity
);

/* Processes upper rows [first_row, first_row + row_count), writing their
 * diagonal, upper entries, and exact lower mirrors into the same full NxN
 * caller-owned output. This permits bounded-chunk traversal without a second
 * dense matrix. The caller must cover every upper row once before publishing.
 */
VAECOpStatus vae_c_iic_affinity_compute_rows(
    const float *h,
    size_t rows,
    size_t cols,
    size_t first_row,
    size_t row_count,
    double *affinity,
    size_t affinity_capacity
);

#endif
