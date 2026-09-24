#include "baseline_average_pool_features.h"
#include "baseline_mini75_features.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define GS_TEST_REFERENCE_TOLERANCE 2.0e-7


static void gs_test_fill_reproducible(
    float *values,
    size_t count,
    uint32_t initial_state
)
{
    uint32_t state = initial_state;
    size_t index;

    for (index = 0; index < count; index++) {
        state = state * UINT32_C(1664525) + UINT32_C(1013904223);
        values[index] =
            (float)(state & UINT32_C(0x00ffffff)) / 16777215.0f;
    }
}


static void gs_test_reference_average_pool(
    const float *input,
    size_t channels,
    size_t image_side,
    size_t pool_side,
    float *features
)
{
    size_t pooled_side = image_side / pool_side;
    size_t channel;

    for (channel = 0; channel < channels; channel++) {
        size_t pooled_row;

        for (pooled_row = 0;
             pooled_row < pooled_side;
             pooled_row++) {

            size_t pooled_column;

            for (pooled_column = 0;
                 pooled_column < pooled_side;
                 pooled_column++) {

                size_t row_begin = pooled_row * pool_side;
                size_t row_end = row_begin + pool_side;
                size_t column_begin = pooled_column * pool_side;
                size_t column_end = column_begin + pool_side;
                size_t input_row;
                long double sum = 0.0L;

                for (input_row = row_begin;
                     input_row < row_end;
                     input_row++) {

                    size_t input_column;

                    for (input_column = column_begin;
                         input_column < column_end;
                         input_column++) {

                        size_t input_index =
                            (channel * image_side + input_row) *
                                image_side +
                            input_column;

                        sum += (long double)input[input_index];
                    }
                }

                features[
                    (channel * pooled_side + pooled_row) *
                        pooled_side +
                    pooled_column
                ] = (float)(
                    sum /
                    ((long double)pool_side * (long double)pool_side)
                );
            }
        }
    }
}


static int gs_test_dimensions(
    size_t channels,
    size_t image_side,
    size_t pool_side,
    size_t expected_input_count,
    size_t expected_pooled_side,
    size_t expected_feature_count
)
{
    size_t input_count = 0;
    size_t pooled_side = 0;
    size_t feature_count = 0;

    if (gs_baseline_average_pool_dimensions(
            channels,
            image_side,
            pool_side,
            &input_count,
            &pooled_side,
            &feature_count
        ) != GS_BASELINE_AVERAGE_POOL_SUCCESS ||
        input_count != expected_input_count ||
        pooled_side != expected_pooled_side ||
        feature_count != expected_feature_count) {

        return 1;
    }

    return 0;
}


static int test_mini112_dimensions(void)
{
    return gs_test_dimensions(4, 112, 7, 50176, 16, 1024);
}


static int test_mini75_dimensions(void)
{
    return gs_test_dimensions(4, 75, 5, 22500, 15, 900);
}


static int test_constant_per_channel_and_output_order(void)
{
    const size_t channels = 3;
    const size_t image_side = 8;
    const size_t pool_side = 2;
    const size_t pixels_per_channel = 64;
    const size_t features_per_channel = 16;
    const float channel_values[3] = {0.125f, 0.5f, 0.875f};
    float input[192];
    float features[48];
    size_t channel;

    for (channel = 0; channel < channels; channel++) {
        size_t index;

        for (index = 0; index < pixels_per_channel; index++) {
            input[channel * pixels_per_channel + index] =
                channel_values[channel];
        }
    }

    if (gs_baseline_average_pool_extract(
            input,
            192,
            channels,
            image_side,
            pool_side,
            features,
            48
        ) != GS_BASELINE_AVERAGE_POOL_SUCCESS) {

        return 1;
    }

    for (channel = 0; channel < channels; channel++) {
        size_t index;

        for (index = 0; index < features_per_channel; index++) {
            if (features[channel * features_per_channel + index] !=
                channel_values[channel]) {

                return 1;
            }
        }
    }

    return 0;
}


static int test_known_spatial_pattern(void)
{
    const size_t channels = 2;
    const size_t image_side = 6;
    const size_t pool_side = 3;
    const size_t pooled_side = 2;
    float input[72];
    float features[8];
    size_t channel;

    for (channel = 0; channel < channels; channel++) {
        size_t input_row;

        for (input_row = 0;
             input_row < image_side;
             input_row++) {

            size_t input_column;

            for (input_column = 0;
                 input_column < image_side;
                 input_column++) {

                input[
                    (channel * image_side + input_row) * image_side +
                    input_column
                ] = (float)(
                    channel * 100 + input_row * 10 + input_column
                );
            }
        }
    }

    if (gs_baseline_average_pool_extract(
            input,
            72,
            channels,
            image_side,
            pool_side,
            features,
            8
        ) != GS_BASELINE_AVERAGE_POOL_SUCCESS) {

        return 1;
    }

    for (channel = 0; channel < channels; channel++) {
        size_t pooled_row;

        for (pooled_row = 0;
             pooled_row < pooled_side;
             pooled_row++) {

            size_t pooled_column;

            for (pooled_column = 0;
                 pooled_column < pooled_side;
                 pooled_column++) {

                float expected = (float)(
                    channel * 100 +
                    (pooled_row * pool_side + 1) * 10 +
                    pooled_column * pool_side + 1
                );

                size_t feature_index =
                    (channel * pooled_side + pooled_row) *
                        pooled_side +
                    pooled_column;

                if (features[feature_index] != expected) {
                    return 1;
                }
            }
        }
    }

    if (features[0] != 11.0f || features[7] != 144.0f) {
        return 1;
    }

    return 0;
}


static int test_deterministic_mini112_reference_input_and_range(void)
{
    const size_t input_count = 50176;
    const size_t feature_count = 1024;
    float *input = NULL;
    float *input_copy = NULL;
    float *features = NULL;
    float *features_repeat = NULL;
    float *reference = NULL;
    double maximum_difference = 0.0;
    size_t index;
    int rc = 1;

    input = (float *)malloc(input_count * sizeof(float));
    input_copy = (float *)malloc(input_count * sizeof(float));
    features = (float *)malloc(feature_count * sizeof(float));
    features_repeat = (float *)malloc(feature_count * sizeof(float));
    reference = (float *)malloc(feature_count * sizeof(float));

    if (input == NULL || input_copy == NULL || features == NULL ||
        features_repeat == NULL || reference == NULL) {

        goto cleanup;
    }

    gs_test_fill_reproducible(
        input,
        input_count,
        UINT32_C(0x12345678)
    );
    memcpy(input_copy, input, input_count * sizeof(float));

    if (gs_baseline_average_pool_extract(
            input,
            input_count,
            4,
            112,
            7,
            features,
            feature_count
        ) != GS_BASELINE_AVERAGE_POOL_SUCCESS ||
        gs_baseline_average_pool_extract(
            input,
            input_count,
            4,
            112,
            7,
            features_repeat,
            feature_count
        ) != GS_BASELINE_AVERAGE_POOL_SUCCESS) {

        goto cleanup;
    }

    gs_test_reference_average_pool(
        input,
        4,
        112,
        7,
        reference
    );

    if (memcmp(
            input,
            input_copy,
            input_count * sizeof(float)
        ) != 0 ||
        memcmp(
            features,
            features_repeat,
            feature_count * sizeof(float)
        ) != 0) {

        goto cleanup;
    }

    for (index = 0; index < feature_count; index++) {
        double difference =
            fabs((double)features[index] - (double)reference[index]);

        if (difference > maximum_difference) {
            maximum_difference = difference;
        }

        if (difference > GS_TEST_REFERENCE_TOLERANCE ||
            !isfinite(features[index]) ||
            features[index] < 0.0f ||
            features[index] > 1.0f) {

            goto cleanup;
        }
    }

    printf(
        "MINI112_REFERENCE_MAX_ABS_DIFF=%.9g\n",
        maximum_difference
    );
    rc = 0;

cleanup:
    free(reference);
    free(features_repeat);
    free(features);
    free(input_copy);
    free(input);
    return rc;
}


static int test_mini75_bitwise_regression_and_input(void)
{
    const size_t input_count = 22500;
    const size_t feature_count = 900;
    float *input = NULL;
    float *input_copy = NULL;
    float *generic_features = NULL;
    float *historical_features = NULL;
    double maximum_difference = 0.0;
    int bitwise_equal;
    size_t index;
    int rc = 1;

    input = (float *)malloc(input_count * sizeof(float));
    input_copy = (float *)malloc(input_count * sizeof(float));
    generic_features = (float *)malloc(feature_count * sizeof(float));
    historical_features = (float *)malloc(
        feature_count * sizeof(float)
    );

    if (input == NULL || input_copy == NULL ||
        generic_features == NULL || historical_features == NULL) {

        goto cleanup;
    }

    gs_test_fill_reproducible(
        input,
        input_count,
        UINT32_C(0x9e3779b9)
    );
    memcpy(input_copy, input, input_count * sizeof(float));

    if (gs_baseline_average_pool_extract(
            input,
            input_count,
            4,
            75,
            5,
            generic_features,
            feature_count
        ) != GS_BASELINE_AVERAGE_POOL_SUCCESS ||
        gs_baseline_mini75_pool5_extract(
            input,
            input_count,
            historical_features,
            feature_count
        ) != GS_BASELINE_MINI75_SUCCESS ||
        memcmp(
            input,
            input_copy,
            input_count * sizeof(float)
        ) != 0) {

        goto cleanup;
    }

    for (index = 0; index < feature_count; index++) {
        double difference = fabs(
            (double)generic_features[index] -
            (double)historical_features[index]
        );

        if (difference > maximum_difference) {
            maximum_difference = difference;
        }
    }

    bitwise_equal = memcmp(
        generic_features,
        historical_features,
        feature_count * sizeof(float)
    ) == 0;

    printf(
        "MINI75_BITWISE_EQUAL=%s\n",
        bitwise_equal ? "true" : "false"
    );
    printf(
        "MINI75_MAX_ABS_DIFF=%.9g\n",
        maximum_difference
    );

    if (!bitwise_equal) {
        goto cleanup;
    }

    rc = 0;

cleanup:
    free(historical_features);
    free(generic_features);
    free(input_copy);
    free(input);
    return rc;
}


static int test_error_contract(void)
{
    float input[16] = {0.0f};
    float features[4] = {
        -1.0f, -1.0f, -1.0f, -1.0f
    };
    size_t input_count = 123;
    size_t pooled_side = 123;
    size_t feature_count = 123;

    if (gs_baseline_average_pool_dimensions(
            1, 4, 2, NULL, &pooled_side, &feature_count
        ) != GS_BASELINE_AVERAGE_POOL_ERROR_NULL_ARGUMENT ||
        gs_baseline_average_pool_dimensions(
            1, 4, 2, &input_count, NULL, &feature_count
        ) != GS_BASELINE_AVERAGE_POOL_ERROR_NULL_ARGUMENT ||
        gs_baseline_average_pool_dimensions(
            1, 4, 2, &input_count, &pooled_side, NULL
        ) != GS_BASELINE_AVERAGE_POOL_ERROR_NULL_ARGUMENT ||
        gs_baseline_average_pool_extract(
            NULL, 16, 1, 4, 2, features, 4
        ) != GS_BASELINE_AVERAGE_POOL_ERROR_NULL_ARGUMENT ||
        gs_baseline_average_pool_extract(
            input, 16, 1, 4, 2, NULL, 4
        ) != GS_BASELINE_AVERAGE_POOL_ERROR_NULL_ARGUMENT) {

        return 1;
    }

    if (gs_baseline_average_pool_dimensions(
            0, 4, 2, &input_count, &pooled_side, &feature_count
        ) != GS_BASELINE_AVERAGE_POOL_ERROR_DIMENSIONS ||
        gs_baseline_average_pool_dimensions(
            1, 0, 2, &input_count, &pooled_side, &feature_count
        ) != GS_BASELINE_AVERAGE_POOL_ERROR_DIMENSIONS ||
        gs_baseline_average_pool_dimensions(
            1, 4, 0, &input_count, &pooled_side, &feature_count
        ) != GS_BASELINE_AVERAGE_POOL_ERROR_DIMENSIONS ||
        gs_baseline_average_pool_dimensions(
            1, 5, 2, &input_count, &pooled_side, &feature_count
        ) != GS_BASELINE_AVERAGE_POOL_ERROR_DIVISIBILITY) {

        return 1;
    }

    if (gs_baseline_average_pool_extract(
            input, 15, 1, 4, 2, features, 4
        ) != GS_BASELINE_AVERAGE_POOL_ERROR_BUFFER_SIZE ||
        gs_baseline_average_pool_extract(
            input, 16, 1, 4, 2, features, 3
        ) != GS_BASELINE_AVERAGE_POOL_ERROR_BUFFER_SIZE ||
        features[0] != -1.0f ||
        features[1] != -1.0f ||
        features[2] != -1.0f ||
        features[3] != -1.0f) {

        return 1;
    }

    if (gs_baseline_average_pool_dimensions(
            SIZE_MAX,
            2,
            1,
            &input_count,
            &pooled_side,
            &feature_count
        ) != GS_BASELINE_AVERAGE_POOL_ERROR_OVERFLOW ||
        gs_baseline_average_pool_dimensions(
            1,
            SIZE_MAX,
            1,
            &input_count,
            &pooled_side,
            &feature_count
        ) != GS_BASELINE_AVERAGE_POOL_ERROR_OVERFLOW) {

        return 1;
    }

    return 0;
}


int main(void)
{
    if (test_mini112_dimensions() != 0) {
        return 1;
    }
    printf("[PASS] mini112_dimensions\n");

    if (test_mini75_dimensions() != 0) {
        return 1;
    }
    printf("[PASS] mini75_dimensions\n");

    if (test_constant_per_channel_and_output_order() != 0) {
        return 1;
    }
    printf("[PASS] constant_per_channel_and_output_order\n");

    if (test_known_spatial_pattern() != 0) {
        return 1;
    }
    printf("[PASS] known_spatial_pattern_first_and_last_blocks\n");

    if (test_deterministic_mini112_reference_input_and_range() != 0) {
        return 1;
    }
    printf(
        "[PASS] deterministic_mini112_reference_input_immutability_range\n"
    );

    if (test_mini75_bitwise_regression_and_input() != 0) {
        return 1;
    }
    printf("[PASS] mini75_bitwise_regression_and_input_immutability\n");

    if (test_error_contract() != 0) {
        return 1;
    }
    printf("[PASS] null_dimension_divisibility_buffer_overflow_errors\n");

    printf("MINI112_FEATURE_DIM=1024\n");
    printf("MINI75_FEATURE_DIM=900\n");
    printf("ALL AVERAGE POOL TESTS PASSED\n");
    return 0;
}
