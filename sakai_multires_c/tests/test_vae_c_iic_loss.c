#include "vae_c_iic_loss.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define GOLDEN_BATCH 4
#define GOLDEN_CLASSES 3
#define GOLDEN_VALUES (GOLDEN_BATCH * GOLDEN_CLASSES)
#define GOLDEN_JOINT_VALUES (GOLDEN_CLASSES * GOLDEN_CLASSES)


static const float golden_probabilities_a[GOLDEN_VALUES] = {
    0.60f, 0.30f, 0.10f,
    0.20f, 0.50f, 0.30f,
    0.10f, 0.20f, 0.70f,
    0.55f, 0.15f, 0.30f
};

static const float golden_probabilities_b[GOLDEN_VALUES] = {
    0.50f, 0.35f, 0.15f,
    0.25f, 0.45f, 0.30f,
    0.05f, 0.25f, 0.70f,
    0.40f, 0.20f, 0.40f
};

/*
 * Generated from the exact Notebook06 A2R float32 forward graph.  The
 * gradients were cross-checked independently with scalar reverse-mode
 * Python autodiff (maximum error 5.56e-17 in float64) and central finite
 * differences (maximum error 1.19e-10 in float64).
 */
static const float golden_joint[GOLDEN_JOINT_VALUES] = {
    0.14375001f, 0.10875001f, 0.11000000f,
    0.08625000f, 0.10250000f, 0.09875000f,
    0.07000000f, 0.10125000f, 0.17875000f
};

static const float golden_marginal_i[GOLDEN_CLASSES] = {
    0.36250000f, 0.28750000f, 0.35000000f
};

static const float golden_marginal_j[GOLDEN_CLASSES] = {
    0.30000000f, 0.31250000f, 0.38750000f
};

static const float golden_gradient_a[GOLDEN_VALUES] = {
    0.22786282f, 0.24299052f, 0.29708370f,
    0.25549132f, 0.24421795f, 0.26331833f,
    0.29185185f, 0.26286668f, 0.21157615f,
    0.24858914f, 0.25546965f, 0.26679656f
};

static const float golden_gradient_b[GOLDEN_VALUES] = {
    0.21828301f, 0.24816875f, 0.28881770f,
    0.26645870f, 0.24135649f, 0.25659212f,
    0.31398080f, 0.25793922f, 0.21382569f,
    0.24204408f, 0.25646010f, 0.26743448f
};


static int report_check(const char *name, int passed)
{
    if (!passed) {
        fprintf(stderr, "[FAIL] %s\n", name);
        return 1;
    }

    printf("[PASS] %s\n", name);
    return 0;
}


static int close_float(float actual, float expected, float tolerance)
{
    return isfinite(actual) && isfinite(expected) &&
           fabsf(actual - expected) <= tolerance;
}


static int array_close(
    const float *actual,
    const float *expected,
    size_t count,
    float tolerance
)
{
    size_t index;

    for (index = 0; index < count; index++) {
        if (!close_float(actual[index], expected[index], tolerance)) {
            fprintf(
                stderr,
                "mismatch[%zu]: actual=%.9g expected=%.9g\n",
                index,
                (double)actual[index],
                (double)expected[index]
            );
            return 0;
        }
    }

    return 1;
}


static VAECIICLossStatus golden_forward(
    const float *probabilities_a,
    const float *probabilities_b,
    float *joint,
    float *marginal_i,
    float *marginal_j,
    VAECIICLossResult *result
)
{
    vae_c_iic_loss_result_init(
        result,
        joint,
        GOLDEN_JOINT_VALUES,
        marginal_i,
        GOLDEN_CLASSES,
        marginal_j,
        GOLDEN_CLASSES
    );

    return vae_c_iic_loss_forward(
        probabilities_a,
        GOLDEN_BATCH,
        GOLDEN_CLASSES,
        probabilities_b,
        GOLDEN_BATCH,
        GOLDEN_CLASSES,
        result
    );
}


static int test_reference_golden_vector(void)
{
    float probabilities_a[GOLDEN_VALUES];
    float probabilities_b[GOLDEN_VALUES];
    float probabilities_a_before[GOLDEN_VALUES];
    float probabilities_b_before[GOLDEN_VALUES];
    float joint[GOLDEN_JOINT_VALUES];
    float marginal_i[GOLDEN_CLASSES];
    float marginal_j[GOLDEN_CLASSES];
    float gradient_a[GOLDEN_VALUES];
    float gradient_b[GOLDEN_VALUES];
    VAECIICLossResult result;

    memcpy(
        probabilities_a,
        golden_probabilities_a,
        sizeof(probabilities_a)
    );
    memcpy(
        probabilities_b,
        golden_probabilities_b,
        sizeof(probabilities_b)
    );
    memcpy(
        probabilities_a_before,
        probabilities_a,
        sizeof(probabilities_a)
    );
    memcpy(
        probabilities_b_before,
        probabilities_b,
        sizeof(probabilities_b)
    );

    if (golden_forward(
            probabilities_a,
            probabilities_b,
            joint,
            marginal_i,
            marginal_j,
            &result
        ) != VAE_C_IIC_LOSS_OK ||
        vae_c_iic_loss_backward(
            probabilities_a,
            GOLDEN_BATCH,
            GOLDEN_CLASSES,
            probabilities_b,
            GOLDEN_BATCH,
            GOLDEN_CLASSES,
            &result,
            gradient_a,
            GOLDEN_VALUES,
            gradient_b,
            GOLDEN_VALUES
        ) != VAE_C_IIC_LOSS_OK) {

        return 0;
    }

    printf("GOLDEN_MI=%.9g\n", (double)result.mutual_information);
    printf("GOLDEN_LOSS=%.9g\n", (double)result.loss);

    return result.valid && result.batch_size == GOLDEN_BATCH &&
        result.class_count == GOLDEN_CLASSES &&
        array_close(joint, golden_joint, GOLDEN_JOINT_VALUES, 2.0e-7f) &&
        array_close(
            marginal_i,
            golden_marginal_i,
            GOLDEN_CLASSES,
            2.0e-7f
        ) &&
        array_close(
            marginal_j,
            golden_marginal_j,
            GOLDEN_CLASSES,
            2.0e-7f
        ) &&
        close_float(result.mutual_information, 0.023528768f, 2.0e-7f) &&
        close_float(result.loss, -0.023528768f, 2.0e-7f) &&
        array_close(
            gradient_a,
            golden_gradient_a,
            GOLDEN_VALUES,
            2.0e-6f
        ) &&
        array_close(
            gradient_b,
            golden_gradient_b,
            GOLDEN_VALUES,
            2.0e-6f
        ) &&
        memcmp(
            probabilities_a,
            probabilities_a_before,
            sizeof(probabilities_a)
        ) == 0 &&
        memcmp(
            probabilities_b,
            probabilities_b_before,
            sizeof(probabilities_b)
        ) == 0;
}


static int test_zero_joint_cells(void)
{
    const float probabilities_a[] = {
        1.0f, 0.0f,
        1.0f, 0.0f,
        0.0f, 1.0f
    };
    const float probabilities_b[] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        0.0f, 1.0f
    };
    const float expected_joint[] = {
        0.33333334f, 0.33333334f,
        0.0f, 0.33333334f
    };
    const float expected_gradient_a[] = {
        0.19817829f, 0.66666670f,
        0.42922735f, 0.19817830f,
        0.42922735f, 0.19817830f
    };
    const float expected_gradient_b[] = {
        0.19817829f, 0.42922735f,
        0.19817829f, 0.42922735f,
        0.66666670f, 0.19817830f
    };
    float joint[4];
    float marginal_i[2];
    float marginal_j[2];
    float gradient_a[6];
    float gradient_b[6];
    VAECIICLossResult result;

    vae_c_iic_loss_result_init(
        &result, joint, 4, marginal_i, 2, marginal_j, 2
    );
    if (vae_c_iic_loss_forward(
            probabilities_a,
            3,
            2,
            probabilities_b,
            3,
            2,
            &result
        ) != VAE_C_IIC_LOSS_OK ||
        vae_c_iic_loss_backward(
            probabilities_a,
            3,
            2,
            probabilities_b,
            3,
            2,
            &result,
            gradient_a,
            6,
            gradient_b,
            6
        ) != VAE_C_IIC_LOSS_OK) {

        return 0;
    }

    return array_close(joint, expected_joint, 4, 2.0e-7f) &&
        joint[2] == 0.0f &&
        close_float(result.mutual_information, 0.17441598f, 2.0e-7f) &&
        close_float(result.loss, -0.17441598f, 2.0e-7f) &&
        array_close(
            gradient_a, expected_gradient_a, 6, 2.0e-6f
        ) &&
        array_close(
            gradient_b, expected_gradient_b, 6, 2.0e-6f
        ) &&
        memcmp(joint, (float[]){
            0.33333334f, 0.33333334f, 0.0f, 0.33333334f
        }, sizeof(joint)) == 0;
}


static int evaluate_loss(
    const float *probabilities_a,
    const float *probabilities_b,
    float *loss
)
{
    float joint[GOLDEN_JOINT_VALUES];
    float marginal_i[GOLDEN_CLASSES];
    float marginal_j[GOLDEN_CLASSES];
    VAECIICLossResult result;

    if (golden_forward(
            probabilities_a,
            probabilities_b,
            joint,
            marginal_i,
            marginal_j,
            &result
        ) != VAE_C_IIC_LOSS_OK) {

        return 0;
    }

    *loss = result.loss;
    return 1;
}


static int test_simplex_finite_difference(void)
{
    const float epsilon = 5.0e-3f;
    float joint[GOLDEN_JOINT_VALUES];
    float marginal_i[GOLDEN_CLASSES];
    float marginal_j[GOLDEN_CLASSES];
    float gradient_a[GOLDEN_VALUES];
    float gradient_b[GOLDEN_VALUES];
    VAECIICLossResult result;
    float maximum_error = 0.0f;
    size_t view;

    if (golden_forward(
            golden_probabilities_a,
            golden_probabilities_b,
            joint,
            marginal_i,
            marginal_j,
            &result
        ) != VAE_C_IIC_LOSS_OK ||
        vae_c_iic_loss_backward(
            golden_probabilities_a,
            GOLDEN_BATCH,
            GOLDEN_CLASSES,
            golden_probabilities_b,
            GOLDEN_BATCH,
            GOLDEN_CLASSES,
            &result,
            gradient_a,
            GOLDEN_VALUES,
            gradient_b,
            GOLDEN_VALUES
        ) != VAE_C_IIC_LOSS_OK) {

        return 0;
    }

    for (view = 0; view < 2; view++) {
        size_t row;

        for (row = 0; row < GOLDEN_BATCH; row++) {
            size_t column;

            for (column = 0; column + 1 < GOLDEN_CLASSES; column++) {
                float perturbed_a_plus[GOLDEN_VALUES];
                float perturbed_a_minus[GOLDEN_VALUES];
                float perturbed_b_plus[GOLDEN_VALUES];
                float perturbed_b_minus[GOLDEN_VALUES];
                float loss_plus;
                float loss_minus;
                float numeric;
                float analytic;
                float error;

                memcpy(
                    perturbed_a_plus,
                    golden_probabilities_a,
                    sizeof(perturbed_a_plus)
                );
                memcpy(
                    perturbed_a_minus,
                    golden_probabilities_a,
                    sizeof(perturbed_a_minus)
                );
                memcpy(
                    perturbed_b_plus,
                    golden_probabilities_b,
                    sizeof(perturbed_b_plus)
                );
                memcpy(
                    perturbed_b_minus,
                    golden_probabilities_b,
                    sizeof(perturbed_b_minus)
                );

                if (view == 0) {
                    perturbed_a_plus[row * GOLDEN_CLASSES + column] +=
                        epsilon;
                    perturbed_a_plus[
                        row * GOLDEN_CLASSES + GOLDEN_CLASSES - 1
                    ] -= epsilon;
                    perturbed_a_minus[row * GOLDEN_CLASSES + column] -=
                        epsilon;
                    perturbed_a_minus[
                        row * GOLDEN_CLASSES + GOLDEN_CLASSES - 1
                    ] += epsilon;
                } else {
                    perturbed_b_plus[row * GOLDEN_CLASSES + column] +=
                        epsilon;
                    perturbed_b_plus[
                        row * GOLDEN_CLASSES + GOLDEN_CLASSES - 1
                    ] -= epsilon;
                    perturbed_b_minus[row * GOLDEN_CLASSES + column] -=
                        epsilon;
                    perturbed_b_minus[
                        row * GOLDEN_CLASSES + GOLDEN_CLASSES - 1
                    ] += epsilon;
                }

                if (!evaluate_loss(
                        perturbed_a_plus,
                        perturbed_b_plus,
                        &loss_plus
                    ) ||
                    !evaluate_loss(
                        perturbed_a_minus,
                        perturbed_b_minus,
                        &loss_minus
                    )) {

                    return 0;
                }

                numeric = (loss_plus - loss_minus) / (2.0f * epsilon);
                if (view == 0) {
                    analytic =
                        gradient_a[row * GOLDEN_CLASSES + column] -
                        gradient_a[
                            row * GOLDEN_CLASSES + GOLDEN_CLASSES - 1
                        ];
                } else {
                    analytic =
                        gradient_b[row * GOLDEN_CLASSES + column] -
                        gradient_b[
                            row * GOLDEN_CLASSES + GOLDEN_CLASSES - 1
                        ];
                }

                error = fabsf(numeric - analytic);
                if (error > maximum_error) {
                    maximum_error = error;
                }
            }
        }
    }

    printf("FINITE_DIFFERENCE_MAX_ABS=%.9g\n", (double)maximum_error);
    return maximum_error <= 3.0e-5f;
}


static int test_deterministic_repeat(void)
{
    float joint_a[GOLDEN_JOINT_VALUES];
    float joint_b[GOLDEN_JOINT_VALUES];
    float marginal_i_a[GOLDEN_CLASSES];
    float marginal_i_b[GOLDEN_CLASSES];
    float marginal_j_a[GOLDEN_CLASSES];
    float marginal_j_b[GOLDEN_CLASSES];
    float gradient_a_first[GOLDEN_VALUES];
    float gradient_a_second[GOLDEN_VALUES];
    float gradient_b_first[GOLDEN_VALUES];
    float gradient_b_second[GOLDEN_VALUES];
    VAECIICLossResult first;
    VAECIICLossResult second;

    if (golden_forward(
            golden_probabilities_a,
            golden_probabilities_b,
            joint_a,
            marginal_i_a,
            marginal_j_a,
            &first
        ) != VAE_C_IIC_LOSS_OK ||
        golden_forward(
            golden_probabilities_a,
            golden_probabilities_b,
            joint_b,
            marginal_i_b,
            marginal_j_b,
            &second
        ) != VAE_C_IIC_LOSS_OK ||
        vae_c_iic_loss_backward(
            golden_probabilities_a,
            GOLDEN_BATCH,
            GOLDEN_CLASSES,
            golden_probabilities_b,
            GOLDEN_BATCH,
            GOLDEN_CLASSES,
            &first,
            gradient_a_first,
            GOLDEN_VALUES,
            gradient_b_first,
            GOLDEN_VALUES
        ) != VAE_C_IIC_LOSS_OK ||
        vae_c_iic_loss_backward(
            golden_probabilities_a,
            GOLDEN_BATCH,
            GOLDEN_CLASSES,
            golden_probabilities_b,
            GOLDEN_BATCH,
            GOLDEN_CLASSES,
            &second,
            gradient_a_second,
            GOLDEN_VALUES,
            gradient_b_second,
            GOLDEN_VALUES
        ) != VAE_C_IIC_LOSS_OK) {

        return 0;
    }

    return memcmp(joint_a, joint_b, sizeof(joint_a)) == 0 &&
        memcmp(marginal_i_a, marginal_i_b, sizeof(marginal_i_a)) == 0 &&
        memcmp(marginal_j_a, marginal_j_b, sizeof(marginal_j_a)) == 0 &&
        memcmp(
            gradient_a_first,
            gradient_a_second,
            sizeof(gradient_a_first)
        ) == 0 &&
        memcmp(
            gradient_b_first,
            gradient_b_second,
            sizeof(gradient_b_first)
        ) == 0 &&
        memcmp(
            &first.mutual_information,
            &second.mutual_information,
            sizeof(float)
        ) == 0 &&
        memcmp(&first.loss, &second.loss, sizeof(float)) == 0;
}


static int invalid_forward_status(
    const float *probabilities_a,
    size_t rows_a,
    size_t columns_a,
    const float *probabilities_b,
    size_t rows_b,
    size_t columns_b,
    size_t joint_capacity,
    VAECIICLossStatus expected
)
{
    float joint[GOLDEN_JOINT_VALUES];
    float marginal_i[GOLDEN_CLASSES];
    float marginal_j[GOLDEN_CLASSES];
    VAECIICLossResult result;
    VAECIICLossStatus actual;

    vae_c_iic_loss_result_init(
        &result,
        joint,
        joint_capacity,
        marginal_i,
        GOLDEN_CLASSES,
        marginal_j,
        GOLDEN_CLASSES
    );
    actual = vae_c_iic_loss_forward(
        probabilities_a,
        rows_a,
        columns_a,
        probabilities_b,
        rows_b,
        columns_b,
        &result
    );

    return actual == expected && !result.valid;
}


static int test_invalid_inputs(void)
{
    float nan_a[GOLDEN_VALUES];
    float inf_b[GOLDEN_VALUES];
    float negative_a[GOLDEN_VALUES];
    float unnormalized_b[GOLDEN_VALUES];
    float joint[GOLDEN_JOINT_VALUES];
    float marginal_i[GOLDEN_CLASSES];
    float marginal_j[GOLDEN_CLASSES];
    float gradient_a[GOLDEN_VALUES];
    float gradient_b[GOLDEN_VALUES];
    VAECIICLossResult result;

    memcpy(nan_a, golden_probabilities_a, sizeof(nan_a));
    memcpy(inf_b, golden_probabilities_b, sizeof(inf_b));
    memcpy(negative_a, golden_probabilities_a, sizeof(negative_a));
    memcpy(
        unnormalized_b,
        golden_probabilities_b,
        sizeof(unnormalized_b)
    );
    nan_a[1] = NAN;
    inf_b[5] = INFINITY;
    negative_a[0] = -0.1f;
    negative_a[1] = 0.4f;
    unnormalized_b[0] += 0.01f;

    vae_c_iic_loss_result_init(
        &result,
        joint,
        GOLDEN_JOINT_VALUES,
        marginal_i,
        GOLDEN_CLASSES,
        marginal_j,
        GOLDEN_CLASSES
    );

    if (vae_c_iic_loss_forward(
            NULL,
            GOLDEN_BATCH,
            GOLDEN_CLASSES,
            golden_probabilities_b,
            GOLDEN_BATCH,
            GOLDEN_CLASSES,
            &result
        ) != VAE_C_IIC_LOSS_ERROR_NULL_ARGUMENT ||
        !invalid_forward_status(
            golden_probabilities_a,
            0,
            GOLDEN_CLASSES,
            golden_probabilities_b,
            0,
            GOLDEN_CLASSES,
            GOLDEN_JOINT_VALUES,
            VAE_C_IIC_LOSS_ERROR_DIMENSIONS
        ) ||
        !invalid_forward_status(
            golden_probabilities_a,
            GOLDEN_BATCH,
            GOLDEN_CLASSES,
            golden_probabilities_b,
            GOLDEN_BATCH - 1,
            GOLDEN_CLASSES,
            GOLDEN_JOINT_VALUES,
            VAE_C_IIC_LOSS_ERROR_SHAPE_MISMATCH
        ) ||
        !invalid_forward_status(
            golden_probabilities_a,
            GOLDEN_BATCH,
            GOLDEN_CLASSES,
            golden_probabilities_b,
            GOLDEN_BATCH,
            GOLDEN_CLASSES,
            GOLDEN_JOINT_VALUES - 1,
            VAE_C_IIC_LOSS_ERROR_CAPACITY
        ) ||
        !invalid_forward_status(
            nan_a,
            GOLDEN_BATCH,
            GOLDEN_CLASSES,
            golden_probabilities_b,
            GOLDEN_BATCH,
            GOLDEN_CLASSES,
            GOLDEN_JOINT_VALUES,
            VAE_C_IIC_LOSS_ERROR_NONFINITE
        ) ||
        !invalid_forward_status(
            golden_probabilities_a,
            GOLDEN_BATCH,
            GOLDEN_CLASSES,
            inf_b,
            GOLDEN_BATCH,
            GOLDEN_CLASSES,
            GOLDEN_JOINT_VALUES,
            VAE_C_IIC_LOSS_ERROR_NONFINITE
        ) ||
        !invalid_forward_status(
            negative_a,
            GOLDEN_BATCH,
            GOLDEN_CLASSES,
            golden_probabilities_b,
            GOLDEN_BATCH,
            GOLDEN_CLASSES,
            GOLDEN_JOINT_VALUES,
            VAE_C_IIC_LOSS_ERROR_NEGATIVE_PROBABILITY
        ) ||
        !invalid_forward_status(
            golden_probabilities_a,
            GOLDEN_BATCH,
            GOLDEN_CLASSES,
            unnormalized_b,
            GOLDEN_BATCH,
            GOLDEN_CLASSES,
            GOLDEN_JOINT_VALUES,
            VAE_C_IIC_LOSS_ERROR_ROW_SUM
        )) {

        return 0;
    }

    vae_c_iic_loss_result_init(
        &result,
        joint,
        GOLDEN_JOINT_VALUES,
        marginal_i,
        GOLDEN_CLASSES,
        marginal_j,
        GOLDEN_CLASSES
    );
    if (vae_c_iic_loss_backward(
            golden_probabilities_a,
            GOLDEN_BATCH,
            GOLDEN_CLASSES,
            golden_probabilities_b,
            GOLDEN_BATCH,
            GOLDEN_CLASSES,
            &result,
            gradient_a,
            GOLDEN_VALUES,
            gradient_b,
            GOLDEN_VALUES
        ) != VAE_C_IIC_LOSS_ERROR_INVALID_FORWARD_STATE) {

        return 0;
    }

    if (golden_forward(
            golden_probabilities_a,
            golden_probabilities_b,
            joint,
            marginal_i,
            marginal_j,
            &result
        ) != VAE_C_IIC_LOSS_OK) {

        return 0;
    }

    return vae_c_iic_loss_backward(
        golden_probabilities_a,
        GOLDEN_BATCH,
        GOLDEN_CLASSES,
        golden_probabilities_b,
        GOLDEN_BATCH,
        GOLDEN_CLASSES,
        &result,
        gradient_a,
        GOLDEN_VALUES - 1,
        gradient_b,
        GOLDEN_VALUES
    ) == VAE_C_IIC_LOSS_ERROR_CAPACITY;
}


static int test_main_and_overcluster_dimensions(void)
{
    const size_t class_counts[] = {36, 250};
    size_t case_index;

    for (case_index = 0;
         case_index < sizeof(class_counts) / sizeof(class_counts[0]);
         case_index++) {
        size_t classes = class_counts[case_index];
        size_t input_values = 2 * classes;
        size_t joint_values = classes * classes;
        float *probabilities_a = malloc(input_values * sizeof(float));
        float *probabilities_b = malloc(input_values * sizeof(float));
        float *joint = malloc(joint_values * sizeof(float));
        float *marginal_i = malloc(classes * sizeof(float));
        float *marginal_j = malloc(classes * sizeof(float));
        float *gradient_a = malloc(input_values * sizeof(float));
        float *gradient_b = malloc(input_values * sizeof(float));
        VAECIICLossResult result;
        VAECIICLossStatus forward_status;
        VAECIICLossStatus backward_status;
        size_t index;
        int failed = 0;

        if (probabilities_a == NULL || probabilities_b == NULL ||
            joint == NULL || marginal_i == NULL || marginal_j == NULL ||
            gradient_a == NULL || gradient_b == NULL) {

            free(gradient_b);
            free(gradient_a);
            free(marginal_j);
            free(marginal_i);
            free(joint);
            free(probabilities_b);
            free(probabilities_a);
            return 0;
        }

        for (index = 0; index < input_values; index++) {
            probabilities_a[index] = 1.0f / (float)classes;
            probabilities_b[index] = 1.0f / (float)classes;
        }

        vae_c_iic_loss_result_init(
            &result,
            joint,
            joint_values,
            marginal_i,
            classes,
            marginal_j,
            classes
        );
        forward_status = vae_c_iic_loss_forward(
            probabilities_a,
            2,
            classes,
            probabilities_b,
            2,
            classes,
            &result
        );
        backward_status = forward_status == VAE_C_IIC_LOSS_OK
            ? vae_c_iic_loss_backward(
                probabilities_a,
                2,
                classes,
                probabilities_b,
                2,
                classes,
                &result,
                gradient_a,
                input_values,
                gradient_b,
                input_values
            )
            : forward_status;
        if (forward_status != VAE_C_IIC_LOSS_OK ||
            backward_status != VAE_C_IIC_LOSS_OK ||
            !isfinite(result.loss) || !isfinite(result.mutual_information)) {

            fprintf(
                stderr,
                "class_count=%zu forward=%s backward=%s\n",
                classes,
                vae_c_iic_loss_status_name(forward_status),
                vae_c_iic_loss_status_name(backward_status)
            );
            failed = 1;
        }

        free(gradient_b);
        free(gradient_a);
        free(marginal_j);
        free(marginal_i);
        free(joint);
        free(probabilities_b);
        free(probabilities_a);

        if (failed) {
            return 0;
        }
    }

    return 1;
}


int main(void)
{
    int failures = 0;

    failures += report_check(
        "reference_golden_vector",
        test_reference_golden_vector()
    );
    failures += report_check("zero_joint_cells", test_zero_joint_cells());
    failures += report_check(
        "simplex_finite_difference",
        test_simplex_finite_difference()
    );
    failures += report_check(
        "deterministic_repeat",
        test_deterministic_repeat()
    );
    failures += report_check("invalid_inputs", test_invalid_inputs());
    failures += report_check(
        "main_and_overcluster_dimensions",
        test_main_and_overcluster_dimensions()
    );

    if (failures != 0) {
        fprintf(stderr, "%d VAE C IIC LOSS TEST(S) FAILED\n", failures);
        return 1;
    }

    printf(
        "REFERENCE_CONTRACT_SHA256=%s\n",
        VAE_C_IIC_LOSS_REFERENCE_CONTRACT_SHA256
    );
    printf("JOINT_SYMMETRIZED=false\n");
    printf("EXTRA_RENORMALIZATION=false\n");
    printf("MI_EPSILON_ADDED=false\n");
    printf("ZERO_POLICY=Pij_gt_0_only\n");
    printf("ALL VAE C IIC LOSS TESTS PASSED\n");
    return 0;
}
