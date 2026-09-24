#include "baseline_average_pool_features.h"

#include <stdint.h>


static int gs_average_pool_checked_product(
    size_t left,
    size_t right,
    size_t *product
)
{
    if (product == NULL) {
        return 1;
    }

    if (left != 0 && right > SIZE_MAX / left) {
        return 1;
    }

    *product = left * right;
    return 0;
}


int gs_baseline_average_pool_dimensions(
    size_t channels,
    size_t image_side,
    size_t pool_side,
    size_t *input_count,
    size_t *pooled_side,
    size_t *feature_count
)
{
    size_t pixels_per_channel;
    size_t local_input_count;
    size_t local_pooled_side;
    size_t pooled_values_per_channel;
    size_t local_feature_count;

    if (input_count == NULL ||
        pooled_side == NULL ||
        feature_count == NULL) {

        return GS_BASELINE_AVERAGE_POOL_ERROR_NULL_ARGUMENT;
    }

    if (channels == 0 || image_side == 0 || pool_side == 0) {
        return GS_BASELINE_AVERAGE_POOL_ERROR_DIMENSIONS;
    }

    if (image_side % pool_side != 0) {
        return GS_BASELINE_AVERAGE_POOL_ERROR_DIVISIBILITY;
    }

    if (gs_average_pool_checked_product(
            image_side,
            image_side,
            &pixels_per_channel
        ) != 0 ||
        gs_average_pool_checked_product(
            channels,
            pixels_per_channel,
            &local_input_count
        ) != 0) {

        return GS_BASELINE_AVERAGE_POOL_ERROR_OVERFLOW;
    }

    local_pooled_side = image_side / pool_side;

    if (gs_average_pool_checked_product(
            local_pooled_side,
            local_pooled_side,
            &pooled_values_per_channel
        ) != 0 ||
        gs_average_pool_checked_product(
            channels,
            pooled_values_per_channel,
            &local_feature_count
        ) != 0) {

        return GS_BASELINE_AVERAGE_POOL_ERROR_OVERFLOW;
    }

    *input_count = local_input_count;
    *pooled_side = local_pooled_side;
    *feature_count = local_feature_count;
    return GS_BASELINE_AVERAGE_POOL_SUCCESS;
}


int gs_baseline_average_pool_extract(
    const float *input,
    size_t input_count,
    size_t channels,
    size_t image_side,
    size_t pool_side,
    float *features,
    size_t feature_capacity
)
{
    size_t required_input_count;
    size_t pooled_side;
    size_t required_feature_count;
    size_t pixels_per_channel;
    size_t pooled_values_per_channel;
    size_t pool_area;
    size_t channel;
    int dimensions_rc;

    if (input == NULL || features == NULL) {
        return GS_BASELINE_AVERAGE_POOL_ERROR_NULL_ARGUMENT;
    }

    dimensions_rc = gs_baseline_average_pool_dimensions(
        channels,
        image_side,
        pool_side,
        &required_input_count,
        &pooled_side,
        &required_feature_count
    );

    if (dimensions_rc != GS_BASELINE_AVERAGE_POOL_SUCCESS) {
        return dimensions_rc;
    }

    if (input_count != required_input_count) {
        return GS_BASELINE_AVERAGE_POOL_ERROR_BUFFER_SIZE;
    }

    if (feature_capacity < required_feature_count) {
        return GS_BASELINE_AVERAGE_POOL_ERROR_BUFFER_SIZE;
    }

    if (gs_average_pool_checked_product(
            image_side,
            image_side,
            &pixels_per_channel
        ) != 0 ||
        gs_average_pool_checked_product(
            pooled_side,
            pooled_side,
            &pooled_values_per_channel
        ) != 0 ||
        gs_average_pool_checked_product(
            pool_side,
            pool_side,
            &pool_area
        ) != 0) {

        return GS_BASELINE_AVERAGE_POOL_ERROR_OVERFLOW;
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

                size_t local_row;
                size_t feature_index;
                double sum = 0.0;

                for (local_row = 0;
                     local_row < pool_side;
                     local_row++) {

                    size_t input_row =
                        pooled_row * pool_side + local_row;
                    size_t local_column;

                    for (local_column = 0;
                         local_column < pool_side;
                         local_column++) {

                        size_t input_column =
                            pooled_column * pool_side + local_column;

                        size_t input_index =
                            channel * pixels_per_channel +
                            input_row * image_side +
                            input_column;

                        sum += (double)input[input_index];
                    }
                }

                feature_index =
                    channel * pooled_values_per_channel +
                    pooled_row * pooled_side +
                    pooled_column;

                features[feature_index] =
                    (float)(sum / (double)pool_area);
            }
        }
    }

    return GS_BASELINE_AVERAGE_POOL_SUCCESS;
}
