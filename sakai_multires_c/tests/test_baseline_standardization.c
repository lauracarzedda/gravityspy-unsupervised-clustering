#include "baseline_standardization.h"

#include <math.h>
#include <stdio.h>
#include <string.h>


#define GS_TEST_DOUBLE_TOLERANCE 1.0e-12
#define GS_TEST_FLOAT_TOLERANCE  1.0e-6


static int nearly_equal_double(
    double actual,
    double expected,
    double tolerance
)
{
    double difference = actual - expected;

    if (difference < 0.0) {
        difference = -difference;
    }

    return difference <= tolerance;
}


static int test_known_mean_std(void)
{
    const float x[8] = {
        1.0f, 2.0f,
        2.0f, 4.0f,
        3.0f, 6.0f,
        4.0f, 8.0f
    };

    double mean[2];
    double stddev[2];
    unsigned char constant_mask[2];

    if (gs_baseline_standardization_fit(
            x, 4, 2,
            mean, stddev, constant_mask,
            GS_BASELINE_STANDARDIZATION_DEFAULT_CONSTANT_EPSILON
        ) != GS_BASELINE_STANDARDIZATION_SUCCESS) {

        return 1;
    }

    if (!nearly_equal_double(mean[0], 2.5, GS_TEST_DOUBLE_TOLERANCE) ||
        !nearly_equal_double(mean[1], 5.0, GS_TEST_DOUBLE_TOLERANCE) ||
        !nearly_equal_double(
            stddev[0], sqrt(1.25), GS_TEST_DOUBLE_TOLERANCE
        ) ||
        !nearly_equal_double(
            stddev[1], sqrt(5.0), GS_TEST_DOUBLE_TOLERANCE
        ) ||
        constant_mask[0] != 0u ||
        constant_mask[1] != 0u) {

        return 1;
    }

    return 0;
}


static int test_standardized_train_statistics(void)
{
    const float x[15] = {
        1.0f, 10.0f, -2.0f,
        2.0f,  8.0f, -1.0f,
        3.0f,  6.0f,  0.0f,
        4.0f,  4.0f,  1.0f,
        5.0f,  2.0f,  2.0f
    };

    double mean[3];
    double stddev[3];
    unsigned char constant_mask[3];
    float output[15];
    size_t col;

    if (gs_baseline_standardization_fit(
            x, 5, 3,
            mean, stddev, constant_mask,
            GS_BASELINE_STANDARDIZATION_DEFAULT_CONSTANT_EPSILON
        ) != GS_BASELINE_STANDARDIZATION_SUCCESS ||

        gs_baseline_standardization_transform(
            x, 5, 3,
            mean, stddev, constant_mask,
            output
        ) != GS_BASELINE_STANDARDIZATION_SUCCESS) {

        return 1;
    }

    for (col = 0; col < 3; col++) {
        size_t row;
        double standardized_mean = 0.0;
        double standardized_variance = 0.0;

        if (constant_mask[col] != 0u) {
            return 1;
        }

        for (row = 0; row < 5; row++) {
            standardized_mean += (double)output[row * 3 + col];
        }
        standardized_mean /= 5.0;

        for (row = 0; row < 5; row++) {
            double difference =
                (double)output[row * 3 + col] - standardized_mean;

            standardized_variance += difference * difference;
        }
        standardized_variance /= 5.0;

        if (!nearly_equal_double(
                standardized_mean,
                0.0,
                GS_TEST_FLOAT_TOLERANCE
            ) ||
            !nearly_equal_double(
                sqrt(standardized_variance),
                1.0,
                GS_TEST_FLOAT_TOLERANCE
            )) {

            return 1;
        }
    }

    return 0;
}


static int test_constant_column(void)
{
    const float x[12] = {
        1.0f, 7.0f, 10.0f,
        2.0f, 7.0f, 20.0f,
        3.0f, 7.0f, 30.0f,
        4.0f, 7.0f, 40.0f
    };

    double mean[3];
    double stddev[3];
    unsigned char constant_mask[3];
    float output[12];
    size_t row;

    if (gs_baseline_standardization_fit(
            x, 4, 3,
            mean, stddev, constant_mask,
            GS_BASELINE_STANDARDIZATION_DEFAULT_CONSTANT_EPSILON
        ) != GS_BASELINE_STANDARDIZATION_SUCCESS ||

        gs_baseline_standardization_transform(
            x, 4, 3,
            mean, stddev, constant_mask,
            output
        ) != GS_BASELINE_STANDARDIZATION_SUCCESS) {

        return 1;
    }

    if (constant_mask[0] != 0u ||
        constant_mask[1] != 1u ||
        constant_mask[2] != 0u ||
        stddev[1] != 0.0) {

        return 1;
    }

    for (row = 0; row < 4; row++) {
        if (output[row * 3 + 1] != 0.0f) {
            return 1;
        }
    }

    return 0;
}


static int test_train_only_no_refit_on_validation(void)
{
    const float train[6] = {
        1.0f, 10.0f,
        2.0f, 20.0f,
        3.0f, 30.0f
    };

    const float validation[4] = {
        4.0f, 40.0f,
        5.0f, 50.0f
    };

    double mean[2];
    double stddev[2];
    unsigned char constant_mask[2];
    double mean_copy[2];
    double stddev_copy[2];
    unsigned char constant_mask_copy[2];
    float output[4];
    size_t index;

    if (gs_baseline_standardization_fit(
            train, 3, 2,
            mean, stddev, constant_mask,
            GS_BASELINE_STANDARDIZATION_DEFAULT_CONSTANT_EPSILON
        ) != GS_BASELINE_STANDARDIZATION_SUCCESS) {

        return 1;
    }

    memcpy(mean_copy, mean, sizeof(mean));
    memcpy(stddev_copy, stddev, sizeof(stddev));
    memcpy(constant_mask_copy, constant_mask, sizeof(constant_mask));

    if (gs_baseline_standardization_transform(
            validation, 2, 2,
            mean, stddev, constant_mask,
            output
        ) != GS_BASELINE_STANDARDIZATION_SUCCESS) {

        return 1;
    }

    for (index = 0; index < 4; index++) {
        size_t col = index % 2;
        double expected =
            ((double)validation[index] - mean_copy[col]) /
            stddev_copy[col];

        if (!nearly_equal_double(
                (double)output[index],
                expected,
                GS_TEST_FLOAT_TOLERANCE
            )) {

            return 1;
        }
    }

    if (memcmp(mean, mean_copy, sizeof(mean)) != 0 ||
        memcmp(stddev, stddev_copy, sizeof(stddev)) != 0 ||
        memcmp(
            constant_mask,
            constant_mask_copy,
            sizeof(constant_mask)
        ) != 0) {

        return 1;
    }

    return 0;
}


static int test_invalid_arguments(void)
{
    const float valid_x[4] = {
        1.0f, 2.0f,
        3.0f, 4.0f
    };

    float invalid_x[4];
    float output[4];
    double mean[2] = {2.0, 3.0};
    double stddev[2] = {1.0, 1.0};
    unsigned char constant_mask[2] = {0u, 0u};

    memcpy(invalid_x, valid_x, sizeof(valid_x));

    if (gs_baseline_standardization_fit(
            NULL, 2, 2, mean, stddev, constant_mask,
            GS_BASELINE_STANDARDIZATION_DEFAULT_CONSTANT_EPSILON
        ) != GS_BASELINE_STANDARDIZATION_ERROR_NULL_ARGUMENT ||
        gs_baseline_standardization_fit(
            valid_x, 2, 2, NULL, stddev, constant_mask,
            GS_BASELINE_STANDARDIZATION_DEFAULT_CONSTANT_EPSILON
        ) != GS_BASELINE_STANDARDIZATION_ERROR_NULL_ARGUMENT ||
        gs_baseline_standardization_fit(
            valid_x, 2, 2, mean, NULL, constant_mask,
            GS_BASELINE_STANDARDIZATION_DEFAULT_CONSTANT_EPSILON
        ) != GS_BASELINE_STANDARDIZATION_ERROR_NULL_ARGUMENT ||
        gs_baseline_standardization_fit(
            valid_x, 2, 2, mean, stddev, NULL,
            GS_BASELINE_STANDARDIZATION_DEFAULT_CONSTANT_EPSILON
        ) != GS_BASELINE_STANDARDIZATION_ERROR_NULL_ARGUMENT ||
        gs_baseline_standardization_fit(
            valid_x, 0, 2, mean, stddev, constant_mask,
            GS_BASELINE_STANDARDIZATION_DEFAULT_CONSTANT_EPSILON
        ) != GS_BASELINE_STANDARDIZATION_ERROR_DIMENSIONS ||
        gs_baseline_standardization_fit(
            valid_x, 2, 0, mean, stddev, constant_mask,
            GS_BASELINE_STANDARDIZATION_DEFAULT_CONSTANT_EPSILON
        ) != GS_BASELINE_STANDARDIZATION_ERROR_DIMENSIONS ||
        gs_baseline_standardization_fit(
            valid_x, 2, 2, mean, stddev, constant_mask, -1.0
        ) != GS_BASELINE_STANDARDIZATION_ERROR_EPSILON ||
        gs_baseline_standardization_fit(
            valid_x, 2, 2, mean, stddev, constant_mask, NAN
        ) != GS_BASELINE_STANDARDIZATION_ERROR_EPSILON ||
        gs_baseline_standardization_fit(
            valid_x, 2, 2, mean, stddev, constant_mask, INFINITY
        ) != GS_BASELINE_STANDARDIZATION_ERROR_EPSILON) {

        return 1;
    }

    invalid_x[0] = NAN;
    if (gs_baseline_standardization_fit(
            invalid_x, 2, 2, mean, stddev, constant_mask,
            GS_BASELINE_STANDARDIZATION_DEFAULT_CONSTANT_EPSILON
        ) != GS_BASELINE_STANDARDIZATION_ERROR_NONFINITE_INPUT) {

        return 1;
    }

    invalid_x[0] = INFINITY;
    if (gs_baseline_standardization_fit(
            invalid_x, 2, 2, mean, stddev, constant_mask,
            GS_BASELINE_STANDARDIZATION_DEFAULT_CONSTANT_EPSILON
        ) != GS_BASELINE_STANDARDIZATION_ERROR_NONFINITE_INPUT) {

        return 1;
    }

    memcpy(invalid_x, valid_x, sizeof(valid_x));

    if (gs_baseline_standardization_transform(
            NULL, 2, 2, mean, stddev, constant_mask, output
        ) != GS_BASELINE_STANDARDIZATION_ERROR_NULL_ARGUMENT ||
        gs_baseline_standardization_transform(
            valid_x, 2, 2, NULL, stddev, constant_mask, output
        ) != GS_BASELINE_STANDARDIZATION_ERROR_NULL_ARGUMENT ||
        gs_baseline_standardization_transform(
            valid_x, 2, 2, mean, NULL, constant_mask, output
        ) != GS_BASELINE_STANDARDIZATION_ERROR_NULL_ARGUMENT ||
        gs_baseline_standardization_transform(
            valid_x, 2, 2, mean, stddev, NULL, output
        ) != GS_BASELINE_STANDARDIZATION_ERROR_NULL_ARGUMENT ||
        gs_baseline_standardization_transform(
            valid_x, 2, 2, mean, stddev, constant_mask, NULL
        ) != GS_BASELINE_STANDARDIZATION_ERROR_NULL_ARGUMENT ||
        gs_baseline_standardization_transform(
            valid_x, 0, 2, mean, stddev, constant_mask, output
        ) != GS_BASELINE_STANDARDIZATION_ERROR_DIMENSIONS ||
        gs_baseline_standardization_transform(
            valid_x, 2, 0, mean, stddev, constant_mask, output
        ) != GS_BASELINE_STANDARDIZATION_ERROR_DIMENSIONS) {

        return 1;
    }

    invalid_x[0] = NAN;
    if (gs_baseline_standardization_transform(
            invalid_x, 2, 2,
            mean, stddev, constant_mask, output
        ) != GS_BASELINE_STANDARDIZATION_ERROR_NONFINITE_INPUT) {

        return 1;
    }

    invalid_x[0] = INFINITY;
    if (gs_baseline_standardization_transform(
            invalid_x, 2, 2,
            mean, stddev, constant_mask, output
        ) != GS_BASELINE_STANDARDIZATION_ERROR_NONFINITE_INPUT) {

        return 1;
    }

    memcpy(invalid_x, valid_x, sizeof(valid_x));

    mean[0] = NAN;
    if (gs_baseline_standardization_transform(
            valid_x, 2, 2,
            mean, stddev, constant_mask, output
        ) != GS_BASELINE_STANDARDIZATION_ERROR_INVALID_SCALER) {

        return 1;
    }
    mean[0] = 2.0;

    stddev[0] = INFINITY;
    if (gs_baseline_standardization_transform(
            valid_x, 2, 2,
            mean, stddev, constant_mask, output
        ) != GS_BASELINE_STANDARDIZATION_ERROR_INVALID_SCALER) {

        return 1;
    }

    stddev[0] = 0.0;
    if (gs_baseline_standardization_transform(
            valid_x, 2, 2,
            mean, stddev, constant_mask, output
        ) != GS_BASELINE_STANDARDIZATION_ERROR_INVALID_SCALER) {

        return 1;
    }

    stddev[0] = 1.0;
    constant_mask[0] = 2u;
    if (gs_baseline_standardization_transform(
            valid_x, 2, 2,
            mean, stddev, constant_mask, output
        ) != GS_BASELINE_STANDARDIZATION_ERROR_INVALID_SCALER) {

        return 1;
    }

    return 0;
}


static int test_input_and_scaler_not_modified(void)
{
    const float x[6] = {
        1.0f, 4.0f,
        2.0f, 5.0f,
        3.0f, 6.0f
    };

    float x_copy[6];
    float output[6];
    double mean[2];
    double stddev[2];
    unsigned char constant_mask[2];
    double mean_copy[2];
    double stddev_copy[2];
    unsigned char constant_mask_copy[2];

    memcpy(x_copy, x, sizeof(x));

    if (gs_baseline_standardization_fit(
            x, 3, 2,
            mean, stddev, constant_mask,
            GS_BASELINE_STANDARDIZATION_DEFAULT_CONSTANT_EPSILON
        ) != GS_BASELINE_STANDARDIZATION_SUCCESS ||
        memcmp(x, x_copy, sizeof(x)) != 0) {

        return 1;
    }

    memcpy(mean_copy, mean, sizeof(mean));
    memcpy(stddev_copy, stddev, sizeof(stddev));
    memcpy(constant_mask_copy, constant_mask, sizeof(constant_mask));

    if (gs_baseline_standardization_transform(
            x, 3, 2,
            mean, stddev, constant_mask,
            output
        ) != GS_BASELINE_STANDARDIZATION_SUCCESS ||
        memcmp(x, x_copy, sizeof(x)) != 0 ||
        memcmp(mean, mean_copy, sizeof(mean)) != 0 ||
        memcmp(stddev, stddev_copy, sizeof(stddev)) != 0 ||
        memcmp(
            constant_mask,
            constant_mask_copy,
            sizeof(constant_mask)
        ) != 0) {

        return 1;
    }

    return 0;
}


static int test_deterministic_repeat(void)
{
    const float x[12] = {
         1.0f,  2.0f, 3.0f,
         4.0f,  5.0f, 3.0f,
         7.0f,  8.0f, 3.0f,
        10.0f, 11.0f, 3.0f
    };

    double mean_a[3];
    double mean_b[3];
    double stddev_a[3];
    double stddev_b[3];
    unsigned char mask_a[3];
    unsigned char mask_b[3];
    float output_a[12];
    float output_b[12];

    if (gs_baseline_standardization_fit(
            x, 4, 3,
            mean_a, stddev_a, mask_a,
            GS_BASELINE_STANDARDIZATION_DEFAULT_CONSTANT_EPSILON
        ) != GS_BASELINE_STANDARDIZATION_SUCCESS ||

        gs_baseline_standardization_fit(
            x, 4, 3,
            mean_b, stddev_b, mask_b,
            GS_BASELINE_STANDARDIZATION_DEFAULT_CONSTANT_EPSILON
        ) != GS_BASELINE_STANDARDIZATION_SUCCESS ||

        memcmp(mean_a, mean_b, sizeof(mean_a)) != 0 ||
        memcmp(stddev_a, stddev_b, sizeof(stddev_a)) != 0 ||
        memcmp(mask_a, mask_b, sizeof(mask_a)) != 0 ||

        gs_baseline_standardization_transform(
            x, 4, 3,
            mean_a, stddev_a, mask_a,
            output_a
        ) != GS_BASELINE_STANDARDIZATION_SUCCESS ||

        gs_baseline_standardization_transform(
            x, 4, 3,
            mean_b, stddev_b, mask_b,
            output_b
        ) != GS_BASELINE_STANDARDIZATION_SUCCESS ||

        memcmp(output_a, output_b, sizeof(output_a)) != 0) {

        return 1;
    }

    return 0;
}


int main(void)
{
    if (test_known_mean_std() != 0) {
        return 1;
    }
    printf("[PASS] known_mean_std\n");

    if (test_standardized_train_statistics() != 0) {
        return 1;
    }
    printf("[PASS] standardized_train_statistics\n");

    if (test_constant_column() != 0) {
        return 1;
    }
    printf("[PASS] constant_column\n");

    if (test_train_only_no_refit_on_validation() != 0) {
        return 1;
    }
    printf("[PASS] train_only_no_refit_on_validation\n");

    if (test_invalid_arguments() != 0) {
        return 1;
    }
    printf("[PASS] invalid_arguments\n");

    if (test_input_and_scaler_not_modified() != 0) {
        return 1;
    }
    printf("[PASS] input_and_scaler_not_modified\n");

    if (test_deterministic_repeat() != 0) {
        return 1;
    }
    printf("[PASS] deterministic_repeat\n");

    printf("DDOF=%d\n", GS_BASELINE_STANDARDIZATION_DDOF);
    printf("CONSTANT_EPSILON=1e-12\n");
    printf("ALL TESTS PASSED\n");

    return 0;
}
