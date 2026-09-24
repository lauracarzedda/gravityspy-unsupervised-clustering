#include "baseline_mini75_features.h"


int gs_baseline_mini75_pool5_extract(
    const float *input,
    size_t input_count,
    float *features,
    size_t feature_count
)
{
    size_t channel;
    size_t pooled_row;
    size_t pooled_col;

    if (input == NULL || features == NULL) {
        return GS_BASELINE_MINI75_ERROR_NULL_ARGUMENT;
    }

    if (input_count != GS_BASELINE_MINI75_INPUT_COUNT ||
        feature_count != GS_BASELINE_MINI75_FEATURE_COUNT) {

        return GS_BASELINE_MINI75_ERROR_COUNT;
    }

    for (channel = 0;
         channel < GS_BASELINE_MINI75_CHANNELS;
         channel++) {

        for (pooled_row = 0;
             pooled_row < GS_BASELINE_MINI75_POOLED_SIDE;
             pooled_row++) {

            for (pooled_col = 0;
                 pooled_col < GS_BASELINE_MINI75_POOLED_SIDE;
                 pooled_col++) {

                size_t local_row;
                size_t local_col;
                size_t feature_index;
                double sum = 0.0;

                for (local_row = 0;
                     local_row < GS_BASELINE_MINI75_POOL_SIDE;
                     local_row++) {

                    size_t input_row =
                        pooled_row * GS_BASELINE_MINI75_POOL_SIDE +
                        local_row;

                    for (local_col = 0;
                         local_col < GS_BASELINE_MINI75_POOL_SIDE;
                         local_col++) {

                        size_t input_col =
                            pooled_col * GS_BASELINE_MINI75_POOL_SIDE +
                            local_col;

                        size_t input_index =
                            channel *
                                GS_BASELINE_MINI75_SIDE *
                                GS_BASELINE_MINI75_SIDE +
                            input_row * GS_BASELINE_MINI75_SIDE +
                            input_col;

                        sum += (double)input[input_index];
                    }
                }

                feature_index =
                    channel *
                        GS_BASELINE_MINI75_POOLED_SIDE *
                        GS_BASELINE_MINI75_POOLED_SIDE +
                    pooled_row * GS_BASELINE_MINI75_POOLED_SIDE +
                    pooled_col;

                features[feature_index] = (float)(sum / 25.0);
            }
        }
    }

    return GS_BASELINE_MINI75_SUCCESS;
}
