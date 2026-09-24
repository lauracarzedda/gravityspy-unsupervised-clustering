#ifndef VAE_C_IIC_LOSS_H
#define VAE_C_IIC_LOSS_H

#include <stdbool.h>
#include <stddef.h>


#define VAE_C_IIC_LOSS_REFERENCE_CONTRACT_SHA256 \
    "b0b108727286c8202b4ebfcae987d6776a80eb3a8f993c992a5a328f785179f1"

/* Match the frozen IIC head's float32, left-to-right softmax row check.
 * A shared constant would require changing that frozen producer. */
#define VAE_C_IIC_PROBABILITY_ROW_RTOL 1.0e-4f
#define VAE_C_IIC_PROBABILITY_ROW_ATOL 1.0e-7f


typedef enum {
    VAE_C_IIC_LOSS_OK = 0,
    VAE_C_IIC_LOSS_ERROR_NULL_ARGUMENT,
    VAE_C_IIC_LOSS_ERROR_DIMENSIONS,
    VAE_C_IIC_LOSS_ERROR_SHAPE_MISMATCH,
    VAE_C_IIC_LOSS_ERROR_CAPACITY,
    VAE_C_IIC_LOSS_ERROR_OVERFLOW,
    VAE_C_IIC_LOSS_ERROR_NONFINITE,
    VAE_C_IIC_LOSS_ERROR_NEGATIVE_PROBABILITY,
    VAE_C_IIC_LOSS_ERROR_ROW_SUM,
    VAE_C_IIC_LOSS_ERROR_ALIASING,
    VAE_C_IIC_LOSS_ERROR_INVALID_FORWARD_STATE
} VAECIICLossStatus;


typedef struct {
    float *joint;
    size_t joint_capacity;

    float *marginal_i;
    size_t marginal_i_capacity;

    float *marginal_j;
    size_t marginal_j_capacity;

    size_t batch_size;
    size_t class_count;

    float mutual_information;
    float loss;

    const float *probabilities_a_data;
    const float *probabilities_b_data;
    bool valid;
} VAECIICLossResult;


/*
 * Bind caller-owned output buffers to a reusable result object.  The joint
 * buffer requires class_count * class_count floats; each marginal buffer
 * requires class_count floats.  No allocation is performed by this module.
 */
void vae_c_iic_loss_result_init(
    VAECIICLossResult *result,
    float *joint,
    size_t joint_capacity,
    float *marginal_i,
    size_t marginal_i_capacity,
    float *marginal_j,
    size_t marginal_j_capacity
);


/*
 * Compute the Notebook06 reference objective in float32:
 *
 *   P = probabilities_a^T * probabilities_b / batch_size
 *   MI = sum_{Pij > 0} Pij * (log(Pij) - log(Pi) - log(Pj))
 *   loss = -MI
 *
 * The paired inputs must be row-major matrices with identical non-zero
 * [batch_size, class_count] shapes.  Values must be finite, non-negative,
 * and each float32 row sum must meet the IIC head-compatible tolerance.
 * The joint is
 * neither symmetrized nor renormalized, and no epsilon is added to the MI.
 *
 * Input buffers and the three result buffers must not alias.  Exact pointer
 * aliasing is rejected; partial overlap cannot be detected portably.  Input
 * buffers are never modified.  A failed call leaves result->valid false.
 */
VAECIICLossStatus vae_c_iic_loss_forward(
    const float *probabilities_a,
    size_t probabilities_a_rows,
    size_t probabilities_a_columns,
    const float *probabilities_b,
    size_t probabilities_b_rows,
    size_t probabilities_b_columns,
    VAECIICLossResult *result
);


/*
 * Compute d(loss)/d(probabilities_a) and d(loss)/d(probabilities_b) from a
 * successful matching forward result.  Gradients are row-major and are
 * overwritten, not accumulated.  Output gradients must not alias inputs,
 * each other, or result buffers.  Exact pointer aliasing is rejected;
 * partial overlap cannot be detected portably.
 */
VAECIICLossStatus vae_c_iic_loss_backward(
    const float *probabilities_a,
    size_t probabilities_a_rows,
    size_t probabilities_a_columns,
    const float *probabilities_b,
    size_t probabilities_b_rows,
    size_t probabilities_b_columns,
    const VAECIICLossResult *result,
    float *probabilities_a_gradient,
    size_t probabilities_a_gradient_capacity,
    float *probabilities_b_gradient,
    size_t probabilities_b_gradient_capacity
);


const char *vae_c_iic_loss_status_name(VAECIICLossStatus status);

#endif
