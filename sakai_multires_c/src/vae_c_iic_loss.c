#include "vae_c_iic_loss.h"

#include <math.h>
#include <stdint.h>


static void vae_c_iic_loss_result_invalidate(VAECIICLossResult *result)
{
    if (result == NULL) {
        return;
    }

    result->batch_size = 0;
    result->class_count = 0;
    result->mutual_information = 0.0f;
    result->loss = 0.0f;
    result->probabilities_a_data = NULL;
    result->probabilities_b_data = NULL;
    result->valid = false;
}


static int vae_c_iic_loss_size_product(
    size_t left,
    size_t right,
    size_t *product
)
{
    if (product == NULL) {
        return 0;
    }
    if (left != 0 && right > SIZE_MAX / left) {
        return 0;
    }

    *product = left * right;
    return 1;
}


static VAECIICLossStatus vae_c_iic_loss_validate_probabilities(
    const float *probabilities,
    size_t rows,
    size_t columns
)
{
    size_t row;

    if (probabilities == NULL) {
        return VAE_C_IIC_LOSS_ERROR_NULL_ARGUMENT;
    }
    if (rows == 0 || columns == 0) {
        return VAE_C_IIC_LOSS_ERROR_DIMENSIONS;
    }

    for (row = 0; row < rows; row++) {
        size_t column;
        float row_sum = 0.0f;

        for (column = 0; column < columns; column++) {
            float value = probabilities[row * columns + column];

            if (!isfinite(value)) {
                return VAE_C_IIC_LOSS_ERROR_NONFINITE;
            }
            if (value < 0.0f) {
                return VAE_C_IIC_LOSS_ERROR_NEGATIVE_PROBABILITY;
            }

            /* Mirror the frozen head producer's left-to-right float32 sum. */
            row_sum += value;
            if (!isfinite(row_sum)) {
                return VAE_C_IIC_LOSS_ERROR_NONFINITE;
            }
        }

        if (fabsf(row_sum - 1.0f) >
            VAE_C_IIC_PROBABILITY_ROW_ATOL +
                VAE_C_IIC_PROBABILITY_ROW_RTOL) {

            return VAE_C_IIC_LOSS_ERROR_ROW_SUM;
        }
    }

    return VAE_C_IIC_LOSS_OK;
}


static int vae_c_iic_loss_exact_alias(
    const void *left,
    const void *right
)
{
    return left != NULL && right != NULL && left == right;
}


static VAECIICLossStatus vae_c_iic_loss_validate_result_buffers(
    const float *probabilities_a,
    const float *probabilities_b,
    size_t class_count,
    const VAECIICLossResult *result
)
{
    size_t joint_values;

    if (result == NULL || result->joint == NULL ||
        result->marginal_i == NULL || result->marginal_j == NULL) {

        return VAE_C_IIC_LOSS_ERROR_NULL_ARGUMENT;
    }
    if (!vae_c_iic_loss_size_product(
            class_count, class_count, &joint_values
        )) {

        return VAE_C_IIC_LOSS_ERROR_OVERFLOW;
    }
    if (result->joint_capacity < joint_values ||
        result->marginal_i_capacity < class_count ||
        result->marginal_j_capacity < class_count) {

        return VAE_C_IIC_LOSS_ERROR_CAPACITY;
    }
    if (vae_c_iic_loss_exact_alias(probabilities_a, result->joint) ||
        vae_c_iic_loss_exact_alias(probabilities_a, result->marginal_i) ||
        vae_c_iic_loss_exact_alias(probabilities_a, result->marginal_j) ||
        vae_c_iic_loss_exact_alias(probabilities_b, result->joint) ||
        vae_c_iic_loss_exact_alias(probabilities_b, result->marginal_i) ||
        vae_c_iic_loss_exact_alias(probabilities_b, result->marginal_j) ||
        vae_c_iic_loss_exact_alias(result->joint, result->marginal_i) ||
        vae_c_iic_loss_exact_alias(result->joint, result->marginal_j) ||
        vae_c_iic_loss_exact_alias(
            result->marginal_i, result->marginal_j
        )) {

        return VAE_C_IIC_LOSS_ERROR_ALIASING;
    }

    return VAE_C_IIC_LOSS_OK;
}


void vae_c_iic_loss_result_init(
    VAECIICLossResult *result,
    float *joint,
    size_t joint_capacity,
    float *marginal_i,
    size_t marginal_i_capacity,
    float *marginal_j,
    size_t marginal_j_capacity
)
{
    if (result == NULL) {
        return;
    }

    result->joint = joint;
    result->joint_capacity = joint_capacity;
    result->marginal_i = marginal_i;
    result->marginal_i_capacity = marginal_i_capacity;
    result->marginal_j = marginal_j;
    result->marginal_j_capacity = marginal_j_capacity;
    vae_c_iic_loss_result_invalidate(result);
}


VAECIICLossStatus vae_c_iic_loss_forward(
    const float *probabilities_a,
    size_t probabilities_a_rows,
    size_t probabilities_a_columns,
    const float *probabilities_b,
    size_t probabilities_b_rows,
    size_t probabilities_b_columns,
    VAECIICLossResult *result
)
{
    VAECIICLossStatus status;
    size_t input_values;
    size_t i;
    float inverse_batch;
    float mutual_information = 0.0f;

    if (result == NULL) {
        return VAE_C_IIC_LOSS_ERROR_NULL_ARGUMENT;
    }
    vae_c_iic_loss_result_invalidate(result);

    if (probabilities_a == NULL || probabilities_b == NULL) {
        return VAE_C_IIC_LOSS_ERROR_NULL_ARGUMENT;
    }
    if (probabilities_a_rows == 0 || probabilities_a_columns == 0 ||
        probabilities_b_rows == 0 || probabilities_b_columns == 0) {

        return VAE_C_IIC_LOSS_ERROR_DIMENSIONS;
    }
    if (probabilities_a_rows != probabilities_b_rows ||
        probabilities_a_columns != probabilities_b_columns) {

        return VAE_C_IIC_LOSS_ERROR_SHAPE_MISMATCH;
    }
    if (!vae_c_iic_loss_size_product(
            probabilities_a_rows,
            probabilities_a_columns,
            &input_values
        )) {

        return VAE_C_IIC_LOSS_ERROR_OVERFLOW;
    }
    (void)input_values;

    status = vae_c_iic_loss_validate_result_buffers(
        probabilities_a,
        probabilities_b,
        probabilities_a_columns,
        result
    );
    if (status != VAE_C_IIC_LOSS_OK) {
        return status;
    }

    status = vae_c_iic_loss_validate_probabilities(
        probabilities_a,
        probabilities_a_rows,
        probabilities_a_columns
    );
    if (status != VAE_C_IIC_LOSS_OK) {
        return status;
    }
    status = vae_c_iic_loss_validate_probabilities(
        probabilities_b,
        probabilities_b_rows,
        probabilities_b_columns
    );
    if (status != VAE_C_IIC_LOSS_OK) {
        return status;
    }

    inverse_batch = 1.0f / (float)probabilities_a_rows;
    if (!isfinite(inverse_batch) || inverse_batch <= 0.0f) {
        return VAE_C_IIC_LOSS_ERROR_OVERFLOW;
    }

    for (i = 0; i < probabilities_a_columns; i++) {
        size_t j;

        for (j = 0; j < probabilities_a_columns; j++) {
            size_t row;
            float sum = 0.0f;

            for (row = 0; row < probabilities_a_rows; row++) {
                sum += probabilities_a[
                    row * probabilities_a_columns + i
                ] * probabilities_b[
                    row * probabilities_b_columns + j
                ];
            }

            result->joint[i * probabilities_a_columns + j] =
                sum * inverse_batch;
            if (!isfinite(
                    result->joint[i * probabilities_a_columns + j]
                )) {

                return VAE_C_IIC_LOSS_ERROR_NONFINITE;
            }
        }
    }

    for (i = 0; i < probabilities_a_columns; i++) {
        size_t j;
        float row_sum = 0.0f;
        float column_sum = 0.0f;

        for (j = 0; j < probabilities_a_columns; j++) {
            row_sum += result->joint[i * probabilities_a_columns + j];
            column_sum += result->joint[
                j * probabilities_a_columns + i
            ];
        }

        result->marginal_i[i] = row_sum;
        result->marginal_j[i] = column_sum;
        if (!isfinite(row_sum) || !isfinite(column_sum)) {
            return VAE_C_IIC_LOSS_ERROR_NONFINITE;
        }
    }

    for (i = 0; i < probabilities_a_columns; i++) {
        size_t j;

        for (j = 0; j < probabilities_a_columns; j++) {
            float joint_value =
                result->joint[i * probabilities_a_columns + j];

            if (joint_value > 0.0f) {
                float term;

                if (result->marginal_i[i] <= 0.0f ||
                    result->marginal_j[j] <= 0.0f) {

                    return VAE_C_IIC_LOSS_ERROR_INVALID_FORWARD_STATE;
                }

                term = joint_value * (
                    logf(joint_value) -
                    logf(result->marginal_i[i]) -
                    logf(result->marginal_j[j])
                );
                if (!isfinite(term)) {
                    return VAE_C_IIC_LOSS_ERROR_NONFINITE;
                }

                mutual_information += term;
                if (!isfinite(mutual_information)) {
                    return VAE_C_IIC_LOSS_ERROR_NONFINITE;
                }
            }
        }
    }

    result->batch_size = probabilities_a_rows;
    result->class_count = probabilities_a_columns;
    result->mutual_information = mutual_information;
    result->loss = -mutual_information;
    result->probabilities_a_data = probabilities_a;
    result->probabilities_b_data = probabilities_b;
    result->valid = true;
    return VAE_C_IIC_LOSS_OK;
}


static VAECIICLossStatus vae_c_iic_loss_validate_forward_result(
    const VAECIICLossResult *result
)
{
    size_t joint_values;
    size_t i;

    if (result == NULL || !result->valid || result->joint == NULL ||
        result->marginal_i == NULL || result->marginal_j == NULL ||
        result->batch_size == 0 || result->class_count == 0 ||
        !isfinite(result->mutual_information) || !isfinite(result->loss) ||
        result->loss != -result->mutual_information) {

        return VAE_C_IIC_LOSS_ERROR_INVALID_FORWARD_STATE;
    }
    if (!vae_c_iic_loss_size_product(
            result->class_count, result->class_count, &joint_values
        ) ||
        result->joint_capacity < joint_values ||
        result->marginal_i_capacity < result->class_count ||
        result->marginal_j_capacity < result->class_count) {

        return VAE_C_IIC_LOSS_ERROR_INVALID_FORWARD_STATE;
    }

    for (i = 0; i < result->class_count; i++) {
        size_t j;
        float row_sum = 0.0f;
        float column_sum = 0.0f;

        if (!isfinite(result->marginal_i[i]) ||
            !isfinite(result->marginal_j[i]) ||
            result->marginal_i[i] < 0.0f ||
            result->marginal_j[i] < 0.0f) {

            return VAE_C_IIC_LOSS_ERROR_INVALID_FORWARD_STATE;
        }

        for (j = 0; j < result->class_count; j++) {
            float row_value = result->joint[
                i * result->class_count + j
            ];
            float column_value = result->joint[
                j * result->class_count + i
            ];

            if (!isfinite(row_value) || row_value < 0.0f ||
                !isfinite(column_value) || column_value < 0.0f) {

                return VAE_C_IIC_LOSS_ERROR_INVALID_FORWARD_STATE;
            }
            row_sum += row_value;
            column_sum += column_value;
        }

        if (!isfinite(row_sum) || !isfinite(column_sum) ||
            fabsf(row_sum - result->marginal_i[i]) >
                VAE_C_IIC_PROBABILITY_ROW_ATOL ||
            fabsf(column_sum - result->marginal_j[i]) >
                VAE_C_IIC_PROBABILITY_ROW_ATOL) {

            return VAE_C_IIC_LOSS_ERROR_INVALID_FORWARD_STATE;
        }
    }

    return VAE_C_IIC_LOSS_OK;
}


static float vae_c_iic_loss_joint_gradient(
    const VAECIICLossResult *result,
    size_t i,
    size_t j
)
{
    float joint_value = result->joint[i * result->class_count + j];
    float gradient = 0.0f;

    if (result->marginal_i[i] > 0.0f) {
        gradient += 1.0f;
    }
    if (result->marginal_j[j] > 0.0f) {
        gradient += 1.0f;
    }
    if (joint_value > 0.0f) {
        gradient +=
            logf(result->marginal_i[i]) +
            logf(result->marginal_j[j]) -
            logf(joint_value) -
            1.0f;
    }

    return gradient;
}


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
)
{
    VAECIICLossStatus status;
    size_t input_values;
    size_t i;
    float inverse_batch;

    if (probabilities_a == NULL || probabilities_b == NULL ||
        result == NULL || probabilities_a_gradient == NULL ||
        probabilities_b_gradient == NULL) {

        return VAE_C_IIC_LOSS_ERROR_NULL_ARGUMENT;
    }
    if (probabilities_a_rows == 0 || probabilities_a_columns == 0 ||
        probabilities_b_rows == 0 || probabilities_b_columns == 0) {

        return VAE_C_IIC_LOSS_ERROR_DIMENSIONS;
    }
    if (probabilities_a_rows != probabilities_b_rows ||
        probabilities_a_columns != probabilities_b_columns) {

        return VAE_C_IIC_LOSS_ERROR_SHAPE_MISMATCH;
    }
    if (!vae_c_iic_loss_size_product(
            probabilities_a_rows,
            probabilities_a_columns,
            &input_values
        )) {

        return VAE_C_IIC_LOSS_ERROR_OVERFLOW;
    }
    if (probabilities_a_gradient_capacity < input_values ||
        probabilities_b_gradient_capacity < input_values) {

        return VAE_C_IIC_LOSS_ERROR_CAPACITY;
    }
    if (vae_c_iic_loss_exact_alias(
            probabilities_a_gradient, probabilities_b_gradient
        ) ||
        vae_c_iic_loss_exact_alias(
            probabilities_a_gradient, probabilities_a
        ) ||
        vae_c_iic_loss_exact_alias(
            probabilities_a_gradient, probabilities_b
        ) ||
        vae_c_iic_loss_exact_alias(
            probabilities_b_gradient, probabilities_a
        ) ||
        vae_c_iic_loss_exact_alias(
            probabilities_b_gradient, probabilities_b
        ) ||
        vae_c_iic_loss_exact_alias(
            probabilities_a_gradient, result->joint
        ) ||
        vae_c_iic_loss_exact_alias(
            probabilities_a_gradient, result->marginal_i
        ) ||
        vae_c_iic_loss_exact_alias(
            probabilities_a_gradient, result->marginal_j
        ) ||
        vae_c_iic_loss_exact_alias(
            probabilities_b_gradient, result->joint
        ) ||
        vae_c_iic_loss_exact_alias(
            probabilities_b_gradient, result->marginal_i
        ) ||
        vae_c_iic_loss_exact_alias(
            probabilities_b_gradient, result->marginal_j
        )) {

        return VAE_C_IIC_LOSS_ERROR_ALIASING;
    }

    status = vae_c_iic_loss_validate_probabilities(
        probabilities_a,
        probabilities_a_rows,
        probabilities_a_columns
    );
    if (status != VAE_C_IIC_LOSS_OK) {
        return status;
    }
    status = vae_c_iic_loss_validate_probabilities(
        probabilities_b,
        probabilities_b_rows,
        probabilities_b_columns
    );
    if (status != VAE_C_IIC_LOSS_OK) {
        return status;
    }
    status = vae_c_iic_loss_validate_forward_result(result);
    if (status != VAE_C_IIC_LOSS_OK) {
        return status;
    }
    if (result->batch_size != probabilities_a_rows ||
        result->class_count != probabilities_a_columns ||
        result->probabilities_a_data != probabilities_a ||
        result->probabilities_b_data != probabilities_b) {

        return VAE_C_IIC_LOSS_ERROR_INVALID_FORWARD_STATE;
    }

    inverse_batch = 1.0f / (float)probabilities_a_rows;

    for (i = 0; i < result->class_count; i++) {
        size_t j;

        for (j = 0; j < result->class_count; j++) {
            if (!isfinite(vae_c_iic_loss_joint_gradient(result, i, j))) {
                return VAE_C_IIC_LOSS_ERROR_NONFINITE;
            }
        }
    }

    for (i = 0; i < probabilities_a_rows; i++) {
        size_t class_a;

        for (class_a = 0; class_a < result->class_count; class_a++) {
            size_t class_b;
            float sum = 0.0f;

            for (class_b = 0;
                 class_b < result->class_count;
                 class_b++) {

                sum += vae_c_iic_loss_joint_gradient(
                    result, class_a, class_b
                ) * probabilities_b[
                    i * probabilities_b_columns + class_b
                ];
            }
            probabilities_a_gradient[
                i * probabilities_a_columns + class_a
            ] = sum * inverse_batch;
        }

        for (size_t class_b = 0;
             class_b < result->class_count;
             class_b++) {
            size_t class_a;
            float sum = 0.0f;

            for (class_a = 0;
                 class_a < result->class_count;
                 class_a++) {

                sum += probabilities_a[
                    i * probabilities_a_columns + class_a
                ] * vae_c_iic_loss_joint_gradient(
                    result, class_a, class_b
                );
            }
            probabilities_b_gradient[
                i * probabilities_b_columns + class_b
            ] = sum * inverse_batch;
        }
    }

    for (i = 0; i < input_values; i++) {
        if (!isfinite(probabilities_a_gradient[i]) ||
            !isfinite(probabilities_b_gradient[i])) {

            return VAE_C_IIC_LOSS_ERROR_NONFINITE;
        }
    }

    return VAE_C_IIC_LOSS_OK;
}


const char *vae_c_iic_loss_status_name(VAECIICLossStatus status)
{
    switch (status) {
        case VAE_C_IIC_LOSS_OK:
            return "VAE_C_IIC_LOSS_OK";
        case VAE_C_IIC_LOSS_ERROR_NULL_ARGUMENT:
            return "VAE_C_IIC_LOSS_ERROR_NULL_ARGUMENT";
        case VAE_C_IIC_LOSS_ERROR_DIMENSIONS:
            return "VAE_C_IIC_LOSS_ERROR_DIMENSIONS";
        case VAE_C_IIC_LOSS_ERROR_SHAPE_MISMATCH:
            return "VAE_C_IIC_LOSS_ERROR_SHAPE_MISMATCH";
        case VAE_C_IIC_LOSS_ERROR_CAPACITY:
            return "VAE_C_IIC_LOSS_ERROR_CAPACITY";
        case VAE_C_IIC_LOSS_ERROR_OVERFLOW:
            return "VAE_C_IIC_LOSS_ERROR_OVERFLOW";
        case VAE_C_IIC_LOSS_ERROR_NONFINITE:
            return "VAE_C_IIC_LOSS_ERROR_NONFINITE";
        case VAE_C_IIC_LOSS_ERROR_NEGATIVE_PROBABILITY:
            return "VAE_C_IIC_LOSS_ERROR_NEGATIVE_PROBABILITY";
        case VAE_C_IIC_LOSS_ERROR_ROW_SUM:
            return "VAE_C_IIC_LOSS_ERROR_ROW_SUM";
        case VAE_C_IIC_LOSS_ERROR_ALIASING:
            return "VAE_C_IIC_LOSS_ERROR_ALIASING";
        case VAE_C_IIC_LOSS_ERROR_INVALID_FORWARD_STATE:
            return "VAE_C_IIC_LOSS_ERROR_INVALID_FORWARD_STATE";
        default:
            return "VAE_C_IIC_LOSS_ERROR_UNKNOWN";
    }
}
