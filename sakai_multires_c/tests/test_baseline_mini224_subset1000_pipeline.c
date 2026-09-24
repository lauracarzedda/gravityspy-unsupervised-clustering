#include "baseline_mini224_subset1000_pipeline.h"

#include "baseline_average_pool_features.h"
#include "baseline_standardization.h"
#include "vae_event_dataset.h"
#include "vae_saved_split.h"
#include "vae_tensor_io.h"
#include "vae_tensor_manifest.h"
#include "vae_tensor_spec.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


enum {
    GS_TEST_TRAIN_ROWS = 8,
    GS_TEST_VALIDATION_ROWS = 4,
    GS_TEST_GUARD_VALUES = 8
};


#define GS_TEST_GUARD_VALUE (-12345.25f)


typedef int (*GSManifestIndexGetter)(
    const GSBaselineMini224Subset1000Pipeline *,
    size_t,
    size_t *
);


static int gs_test_join_path(
    const char *data_root,
    const char *relative_path,
    char *joined_path,
    size_t joined_path_size
)
{
    size_t root_length;
    size_t relative_length;
    size_t required;
    size_t position = 0;
    int needs_separator;

    if (data_root == NULL || relative_path == NULL ||
        joined_path == NULL || joined_path_size == 0) {

        return 1;
    }

    root_length = strlen(data_root);
    relative_length = strlen(relative_path);

    needs_separator =
        root_length != 0 &&
        data_root[root_length - 1] != '/' &&
        data_root[root_length - 1] != '\\';

    if (root_length > SIZE_MAX - relative_length) {
        return 1;
    }

    required = root_length + relative_length;
    if (needs_separator) {
        if (required == SIZE_MAX) {
            return 1;
        }
        required++;
    }

    if (required >= joined_path_size) {
        return 1;
    }

    if (root_length != 0) {
        memcpy(joined_path, data_root, root_length);
        position = root_length;
    }

    if (needs_separator) {
        joined_path[position++] = '/';
    }

    if (relative_length != 0) {
        memcpy(
            joined_path + position,
            relative_path,
            relative_length
        );
        position += relative_length;
    }

    joined_path[position] = '\0';
    return 0;
}


static int gs_test_values_are_finite(
    const float *values,
    size_t count
)
{
    size_t index;

    if (values == NULL) {
        return 0;
    }

    for (index = 0; index < count; index++) {
        if (!isfinite(values[index])) {
            return 0;
        }
    }

    return 1;
}


static int gs_test_values_are_float01(
    const float *values,
    size_t count
)
{
    size_t index;

    if (!gs_test_values_are_finite(values, count)) {
        return 0;
    }

    for (index = 0; index < count; index++) {
        if (values[index] < 0.0f || values[index] > 1.0f) {
            return 0;
        }
    }

    return 1;
}


static int gs_test_guards_are_unchanged(
    const float *values,
    size_t value_count
)
{
    size_t index;

    for (index = 0; index < GS_TEST_GUARD_VALUES; index++) {
        if (values[value_count + index] != GS_TEST_GUARD_VALUE) {
            return 0;
        }
    }

    return 1;
}


static int gs_test_read_first_split_tensor_path(
    const char *split_path,
    char *tensor_path,
    size_t tensor_path_size
)
{
    FILE *file;
    char line[4096];
    char *field;
    char *end;
    size_t length;

    if (split_path == NULL || tensor_path == NULL ||
        tensor_path_size == 0) {

        return 1;
    }

    file = fopen(split_path, "r");
    if (file == NULL) {
        return 1;
    }

    if (fgets(line, sizeof(line), file) == NULL ||
        fgets(line, sizeof(line), file) == NULL) {

        fclose(file);
        return 1;
    }

    fclose(file);

    field = strrchr(line, ',');
    if (field == NULL) {
        return 1;
    }
    field++;

    while (*field == ' ' || *field == '\t') {
        field++;
    }

    end = field + strlen(field);
    while (end > field &&
           (end[-1] == '\n' || end[-1] == '\r' ||
            end[-1] == ' ' || end[-1] == '\t')) {

        end--;
    }
    *end = '\0';

    if (field[0] == '"') {
        field++;
        end = field + strlen(field);

        if (end == field || end[-1] != '"') {
            return 1;
        }

        end[-1] = '\0';
    }

    length = strlen(field);
    if (length == 0 || length >= tensor_path_size) {
        return 1;
    }

    memcpy(tensor_path, field, length + 1);
    return 0;
}


static int test_constants_and_api_contract(void)
{
    typedef int (*GSOpenSignature)(
        GSBaselineMini224Subset1000Pipeline *,
        const char *,
        const char *,
        const char *,
        const char *,
        const char *
    );

    typedef int (*GSExtractSignature)(
        GSBaselineMini224Subset1000Pipeline *,
        size_t,
        size_t,
        float *,
        size_t
    );

    GSOpenSignature open_function =
        gs_baseline_mini224_subset1000_pipeline_open;

    GSExtractSignature train_function =
        gs_baseline_mini224_subset1000_extract_train_rows;

    GSExtractSignature validation_function =
        gs_baseline_mini224_subset1000_extract_validation_rows;

    if (GS_BASELINE_MINI224_SUBSET1000_CHANNELS != 4 ||
        GS_BASELINE_MINI224_SUBSET1000_SIDE != 224 ||
        GS_BASELINE_MINI224_SUBSET1000_POOL_SIDE != 14 ||
        GS_BASELINE_MINI224_SUBSET1000_POOLED_SIDE != 16 ||
        GS_BASELINE_MINI224_SUBSET1000_INPUT_COUNT != 200704 ||
        GS_BASELINE_MINI224_SUBSET1000_TENSOR_BYTES != 200704 ||
        GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT != 1024 ||
        GS_BASELINE_MINI224_SUBSET1000_OFFICIAL_EVENT_COUNT != 8583 ||
        GS_BASELINE_MINI224_SUBSET1000_OFFICIAL_TRAIN_COUNT != 800 ||
        GS_BASELINE_MINI224_SUBSET1000_OFFICIAL_VALIDATION_COUNT != 200 ||
        GS_BASELINE_MINI224_SUBSET1000_FULL_TRAIN_COUNT != 6866 ||
        GS_BASELINE_MINI224_SUBSET1000_FULL_VALIDATION_COUNT != 1717 ||
        open_function == NULL ||
        train_function == NULL ||
        validation_function == NULL ||
        gs_baseline_mini224_subset1000_pipeline_error_string(
            GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_OVERFLOW
        ) == NULL) {

        return 1;
    }

    return 0;
}


static int test_invalid_initialization_and_free(
    const char *tensor_manifest_path,
    const char *train_split_path,
    const char *validation_split_path,
    const char *data_root
)
{
    GSBaselineMini224Subset1000Pipeline pipeline = {0};
    char *long_root = NULL;
    size_t event_count = 0;
    size_t train_count = 0;
    size_t validation_count = 0;
    float one_row[GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT];
    int rc = 1;

    gs_baseline_mini224_subset1000_pipeline_init(&pipeline);

    if (gs_baseline_mini224_subset1000_pipeline_counts(
            &pipeline,
            &event_count,
            &train_count,
            &validation_count
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_STATE ||
        gs_baseline_mini224_subset1000_extract_train_rows(
            &pipeline,
            0,
            1,
            one_row,
            GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_STATE ||
        gs_baseline_mini224_subset1000_pipeline_open(
            NULL,
            "__missing_manifest_events__.csv",
            tensor_manifest_path,
            train_split_path,
            validation_split_path,
            data_root
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_NULL_ARGUMENT ||
        gs_baseline_mini224_subset1000_pipeline_open(
            &pipeline,
            NULL,
            tensor_manifest_path,
            train_split_path,
            validation_split_path,
            data_root
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_NULL_ARGUMENT) {

        goto cleanup;
    }

    if (gs_baseline_mini224_subset1000_pipeline_open(
            &pipeline,
            "__missing_manifest_events__.csv",
            tensor_manifest_path,
            train_split_path,
            validation_split_path,
            data_root
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_EVENT_MANIFEST ||
        pipeline.implementation != NULL) {

        goto cleanup;
    }

    long_root = (char *)malloc(
        (GS_BASELINE_MINI224_SUBSET1000_PATH_LENGTH + 1u) * sizeof(char)
    );
    if (long_root == NULL) {
        goto cleanup;
    }

    memset(long_root, 'x', GS_BASELINE_MINI224_SUBSET1000_PATH_LENGTH);
    long_root[GS_BASELINE_MINI224_SUBSET1000_PATH_LENGTH] = '\0';

    if (gs_baseline_mini224_subset1000_pipeline_open(
            &pipeline,
            "__unused__.csv",
            "__unused__.csv",
            "__unused__.csv",
            "__unused__.csv",
            long_root
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_PATH ||
        pipeline.implementation != NULL) {

        goto cleanup;
    }

    gs_baseline_mini224_subset1000_pipeline_free(&pipeline);
    gs_baseline_mini224_subset1000_pipeline_free(&pipeline);

    if (pipeline.implementation != NULL) {
        goto cleanup;
    }

    rc = 0;

cleanup:
    free(long_root);
    gs_baseline_mini224_subset1000_pipeline_free(&pipeline);
    gs_baseline_mini224_subset1000_pipeline_free(&pipeline);
    return rc;
}


static int test_global_counts(
    const GSBaselineMini224Subset1000Pipeline *pipeline
)
{
    size_t event_count = 0;
    size_t train_count = 0;
    size_t validation_count = 0;

    if (gs_baseline_mini224_subset1000_pipeline_counts(
            pipeline,
            &event_count,
            &train_count,
            &validation_count
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_SUCCESS ||
        event_count != GS_BASELINE_MINI224_SUBSET1000_OFFICIAL_EVENT_COUNT ||
        train_count != GS_BASELINE_MINI224_SUBSET1000_OFFICIAL_TRAIN_COUNT ||
        validation_count !=
            GS_BASELINE_MINI224_SUBSET1000_OFFICIAL_VALIDATION_COUNT) {

        return 1;
    }

    return 0;
}


static int test_subset_tensor_path_is_exact(
    const GSBaselineMini224Subset1000Pipeline *pipeline,
    const GSTensorManifest *tensor_manifest,
    const GSSavedSplit *train_split,
    const char *train_split_path
)
{
    char split_tensor_path[GS_TMAN_PATH_LEN];
    size_t pipeline_manifest_index;
    size_t manifest_index;
    const GSTensorManifestRecord *tensor_record;

    if (pipeline == NULL || tensor_manifest == NULL ||
        train_split == NULL || train_split->indices == NULL) {

        return 1;
    }

    manifest_index = train_split->indices[0];

    if (gs_baseline_mini224_subset1000_train_manifest_index(
            pipeline,
            0,
            &pipeline_manifest_index
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_SUCCESS ||
        pipeline_manifest_index != manifest_index ||
        gs_test_read_first_split_tensor_path(
            train_split_path,
            split_tensor_path,
            sizeof(split_tensor_path)
        ) != 0) {

        return 1;
    }

    tensor_record = gs_tensor_manifest_get(
        tensor_manifest,
        manifest_index
    );

    if (tensor_record == NULL ||
        strcmp(split_tensor_path, tensor_record->tensor_path) != 0 ||
        strstr(
            split_tensor_path,
            "tensors_224_central_uint8/"
        ) == NULL ||
        strstr(
            tensor_record->tensor_path,
            "tensors_224_central_uint8/"
        ) == NULL) {

        return 1;
    }

    printf("SPLIT_TENSOR_PATH=%s\n", split_tensor_path);
    printf("PIPELINE_TENSOR_PATH=%s\n", tensor_record->tensor_path);
    printf("SUBSET_224_PATH_MATCHED=true\n");
    return 0;
}


static int gs_test_direct_compare(
    GSBaselineMini224Subset1000Pipeline *pipeline,
    const GSTensorManifest *tensor_manifest,
    const GSSavedSplit *split,
    GSManifestIndexGetter index_getter,
    const char *data_root,
    const float *pipeline_matrix,
    size_t first_row,
    size_t row_count,
    float *tensor_buffer,
    float *direct_features,
    double *maximum_difference
)
{
    size_t row;

    if (pipeline == NULL || tensor_manifest == NULL ||
        split == NULL || split->indices == NULL ||
        index_getter == NULL || data_root == NULL ||
        pipeline_matrix == NULL || tensor_buffer == NULL ||
        direct_features == NULL || maximum_difference == NULL) {

        return 1;
    }

    if (first_row >= split->count ||
        row_count > split->count - first_row) {

        return 1;
    }

    for (row = 0; row < row_count; row++) {
        size_t split_pos = first_row + row;
        size_t manifest_index = split->indices[split_pos];
        size_t pipeline_manifest_index;
        const GSTensorManifestRecord *tensor_record;
        char joined_path[GS_BASELINE_MINI224_SUBSET1000_PATH_LENGTH];
        const float *pipeline_row =
            pipeline_matrix +
            row * GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT;
        size_t feature;

        if (index_getter(
                pipeline,
                split_pos,
                &pipeline_manifest_index
            ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_SUCCESS ||
            pipeline_manifest_index != manifest_index) {

            return 1;
        }

        tensor_record = gs_tensor_manifest_get(
            tensor_manifest,
            manifest_index
        );

        if (tensor_record == NULL ||
            gs_test_join_path(
                data_root,
                tensor_record->tensor_path,
                joined_path,
                sizeof(joined_path)
            ) != 0 ||
            gs_tensor_read_float01(
                joined_path,
                &tensor_manifest->spec,
                tensor_buffer
            ) != 0 ||
            gs_baseline_average_pool_extract(
                tensor_buffer,
                GS_BASELINE_MINI224_SUBSET1000_INPUT_COUNT,
                GS_BASELINE_MINI224_SUBSET1000_CHANNELS,
                GS_BASELINE_MINI224_SUBSET1000_SIDE,
                GS_BASELINE_MINI224_SUBSET1000_POOL_SIDE,
                direct_features,
                GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT
            ) != GS_BASELINE_AVERAGE_POOL_SUCCESS) {

            return 1;
        }

        for (feature = 0;
             feature < GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT;
             feature++) {

            double difference = fabs(
                (double)pipeline_row[feature] -
                (double)direct_features[feature]
            );

            if (difference > *maximum_difference) {
                *maximum_difference = difference;
            }
        }

        if (memcmp(
                pipeline_row,
                direct_features,
                GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT * sizeof(float)
            ) != 0) {

            return 1;
        }
    }

    return 0;
}


static int test_repeat_and_chunking(
    GSBaselineMini224Subset1000Pipeline *pipeline,
    const float *reference_matrix
)
{
    const size_t value_count =
        GS_TEST_TRAIN_ROWS * GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT;

    const size_t first_chunk_count =
        3u * GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT;

    const size_t second_chunk_count =
        5u * GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT;

    float *repeat_matrix = NULL;
    float *chunked_matrix = NULL;
    int rc = 1;

    if (pipeline == NULL || reference_matrix == NULL) {
        return 1;
    }

    repeat_matrix = (float *)malloc(value_count * sizeof(float));
    chunked_matrix = (float *)malloc(value_count * sizeof(float));

    if (repeat_matrix == NULL || chunked_matrix == NULL) {
        goto cleanup;
    }

    if (gs_baseline_mini224_subset1000_extract_train_rows(
            pipeline,
            0,
            GS_TEST_TRAIN_ROWS,
            repeat_matrix,
            value_count
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_SUCCESS ||
        memcmp(
            reference_matrix,
            repeat_matrix,
            value_count * sizeof(float)
        ) != 0) {

        goto cleanup;
    }

    printf("MINI224_SUBSET1000_REPEAT_BITWISE=true\n");

    if (gs_baseline_mini224_subset1000_extract_train_rows(
            pipeline,
            0,
            3,
            chunked_matrix,
            first_chunk_count
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_SUCCESS ||
        gs_baseline_mini224_subset1000_extract_train_rows(
            pipeline,
            3,
            5,
            chunked_matrix + first_chunk_count,
            second_chunk_count
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_SUCCESS ||
        memcmp(
            reference_matrix,
            chunked_matrix,
            value_count * sizeof(float)
        ) != 0) {

        goto cleanup;
    }

    printf("MINI224_SUBSET1000_CHUNKING_BITWISE=true\n");
    rc = 0;

cleanup:
    free(chunked_matrix);
    free(repeat_matrix);
    return rc;
}


static int test_split_end_boundaries(
    GSBaselineMini224Subset1000Pipeline *pipeline,
    const GSTensorManifest *tensor_manifest,
    const GSSavedSplit *train_split,
    const GSSavedSplit *validation_split,
    const char *data_root,
    float *tensor_buffer,
    float *direct_features
)
{
    const size_t guarded_value_count =
        GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT +
        2u * GS_TEST_GUARD_VALUES;

    const size_t exact_end_value_count =
        2u * GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT;

    const size_t train_last =
        GS_BASELINE_MINI224_SUBSET1000_OFFICIAL_TRAIN_COUNT - 1u;

    const size_t validation_last =
        GS_BASELINE_MINI224_SUBSET1000_OFFICIAL_VALIDATION_COUNT - 1u;

    float *train_guarded = NULL;
    float *validation_guarded = NULL;
    float *train_exact_end = NULL;
    float *validation_exact_end = NULL;
    float *train_output;
    float *validation_output;
    float train_guard_before[GS_TEST_GUARD_VALUES];
    float train_guard_after[GS_TEST_GUARD_VALUES];
    float validation_guard_before[GS_TEST_GUARD_VALUES];
    float validation_guard_after[GS_TEST_GUARD_VALUES];
    double train_maximum_difference = 0.0;
    double validation_maximum_difference = 0.0;
    size_t train_manifest_index;
    size_t validation_manifest_index;
    size_t index;
    int rc = 1;

    if (pipeline == NULL || tensor_manifest == NULL ||
        train_split == NULL || validation_split == NULL ||
        train_split->indices == NULL ||
        validation_split->indices == NULL ||
        data_root == NULL || tensor_buffer == NULL ||
        direct_features == NULL) {

        return 1;
    }

    train_guarded = (float *)malloc(
        guarded_value_count * sizeof(float)
    );
    validation_guarded = (float *)malloc(
        guarded_value_count * sizeof(float)
    );
    train_exact_end = (float *)malloc(
        exact_end_value_count * sizeof(float)
    );
    validation_exact_end = (float *)malloc(
        exact_end_value_count * sizeof(float)
    );

    if (train_guarded == NULL || validation_guarded == NULL ||
        train_exact_end == NULL || validation_exact_end == NULL) {

        goto cleanup;
    }

    train_output = train_guarded + GS_TEST_GUARD_VALUES;
    validation_output =
        validation_guarded + GS_TEST_GUARD_VALUES;

    for (index = 0; index < guarded_value_count; index++) {
        train_guarded[index] = GS_TEST_GUARD_VALUE;
        validation_guarded[index] = GS_TEST_GUARD_VALUE;
    }

    for (index = 0;
         index < GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT;
         index++) {

        train_output[index] = NAN;
        validation_output[index] = NAN;
    }

    memcpy(
        train_guard_before,
        train_guarded,
        sizeof(train_guard_before)
    );
    memcpy(
        train_guard_after,
        train_output + GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT,
        sizeof(train_guard_after)
    );
    memcpy(
        validation_guard_before,
        validation_guarded,
        sizeof(validation_guard_before)
    );
    memcpy(
        validation_guard_after,
        validation_output + GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT,
        sizeof(validation_guard_after)
    );

    if (gs_baseline_mini224_subset1000_train_manifest_index(
            pipeline,
            train_last,
            &train_manifest_index
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_SUCCESS ||
        train_manifest_index != train_split->indices[train_last] ||
        gs_baseline_mini224_subset1000_validation_manifest_index(
            pipeline,
            validation_last,
            &validation_manifest_index
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_SUCCESS ||
        validation_manifest_index !=
            validation_split->indices[validation_last] ||
        gs_baseline_mini224_subset1000_extract_train_rows(
            pipeline,
            train_last,
            1,
            train_output,
            GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_SUCCESS ||
        gs_baseline_mini224_subset1000_extract_validation_rows(
            pipeline,
            validation_last,
            1,
            validation_output,
            GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_SUCCESS ||
        !gs_test_values_are_float01(
            train_output,
            GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT
        ) ||
        !gs_test_values_are_float01(
            validation_output,
            GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT
        ) ||
        memcmp(
            train_guarded,
            train_guard_before,
            sizeof(train_guard_before)
        ) != 0 ||
        memcmp(
            train_output + GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT,
            train_guard_after,
            sizeof(train_guard_after)
        ) != 0 ||
        memcmp(
            validation_guarded,
            validation_guard_before,
            sizeof(validation_guard_before)
        ) != 0 ||
        memcmp(
            validation_output + GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT,
            validation_guard_after,
            sizeof(validation_guard_after)
        ) != 0) {

        goto cleanup;
    }

    if (gs_test_direct_compare(
            pipeline,
            tensor_manifest,
            train_split,
            gs_baseline_mini224_subset1000_train_manifest_index,
            data_root,
            train_output,
            train_last,
            1,
            tensor_buffer,
            direct_features,
            &train_maximum_difference
        ) != 0 ||
        gs_test_direct_compare(
            pipeline,
            tensor_manifest,
            validation_split,
            gs_baseline_mini224_subset1000_validation_manifest_index,
            data_root,
            validation_output,
            validation_last,
            1,
            tensor_buffer,
            direct_features,
            &validation_maximum_difference
        ) != 0) {

        goto cleanup;
    }

    printf("LAST_TRAIN_MANIFEST_INDEX=%lu\n",
           (unsigned long)train_manifest_index);
    printf("LAST_VALIDATION_MANIFEST_INDEX=%lu\n",
           (unsigned long)validation_manifest_index);
    printf("LAST_TRAIN_DIRECT_BITWISE=true\n");
    printf("LAST_VALIDATION_DIRECT_BITWISE=true\n");
    printf(
        "LAST_TRAIN_DIRECT_MAX_ABS_DIFF=%.9g\n",
        train_maximum_difference
    );
    printf(
        "LAST_VALIDATION_DIRECT_MAX_ABS_DIFF=%.9g\n",
        validation_maximum_difference
    );
    printf("FULL_GUARDS_UNCHANGED=true\n");

    if (gs_baseline_mini224_subset1000_extract_train_rows(
            pipeline,
            train_last - 1u,
            2,
            train_exact_end,
            exact_end_value_count
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_SUCCESS ||
        gs_baseline_mini224_subset1000_extract_validation_rows(
            pipeline,
            validation_last - 1u,
            2,
            validation_exact_end,
            exact_end_value_count
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_SUCCESS ||
        !gs_test_values_are_float01(
            train_exact_end,
            exact_end_value_count
        ) ||
        !gs_test_values_are_float01(
            validation_exact_end,
            exact_end_value_count
        ) ||
        gs_baseline_mini224_subset1000_extract_train_rows(
            pipeline,
            train_last,
            2,
            train_exact_end,
            exact_end_value_count
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_RANGE ||
        gs_baseline_mini224_subset1000_extract_validation_rows(
            pipeline,
            validation_last,
            2,
            validation_exact_end,
            exact_end_value_count
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_RANGE) {

        goto cleanup;
    }

    printf("TRAIN_EXACT_END_INTERVAL=true\n");
    printf("VALIDATION_EXACT_END_INTERVAL=true\n");
    rc = 0;

cleanup:
    free(validation_exact_end);
    free(train_exact_end);
    free(validation_guarded);
    free(train_guarded);
    return rc;
}


static int test_context_lifecycle(
    const char *event_manifest_path,
    const char *tensor_manifest_path,
    const char *train_split_path,
    const char *validation_split_path,
    const char *data_root
)
{
    GSBaselineMini224Subset1000Pipeline pipeline = {0};
    float one_row[GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT];
    size_t event_count = 0;
    size_t train_count = 0;
    size_t validation_count = 0;
    int rc = 1;

    gs_baseline_mini224_subset1000_pipeline_init(&pipeline);

    if (gs_baseline_mini224_subset1000_pipeline_open(
            &pipeline,
            event_manifest_path,
            tensor_manifest_path,
            train_split_path,
            validation_split_path,
            data_root
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_SUCCESS ||
        gs_baseline_mini224_subset1000_pipeline_counts(
            &pipeline,
            &event_count,
            &train_count,
            &validation_count
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_SUCCESS ||
        event_count != GS_BASELINE_MINI224_SUBSET1000_OFFICIAL_EVENT_COUNT ||
        train_count != GS_BASELINE_MINI224_SUBSET1000_OFFICIAL_TRAIN_COUNT ||
        validation_count !=
            GS_BASELINE_MINI224_SUBSET1000_OFFICIAL_VALIDATION_COUNT ||
        gs_baseline_mini224_subset1000_pipeline_open(
            &pipeline,
            event_manifest_path,
            tensor_manifest_path,
            train_split_path,
            validation_split_path,
            data_root
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_STATE ||
        gs_baseline_mini224_subset1000_pipeline_counts(
            &pipeline,
            &event_count,
            &train_count,
            &validation_count
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_SUCCESS) {

        goto cleanup;
    }

    printf("SECOND_OPEN_WITHOUT_FREE_REJECTED=true\n");

    gs_baseline_mini224_subset1000_pipeline_free(&pipeline);
    gs_baseline_mini224_subset1000_pipeline_free(&pipeline);

    if (pipeline.implementation != NULL ||
        gs_baseline_mini224_subset1000_pipeline_open(
            &pipeline,
            event_manifest_path,
            tensor_manifest_path,
            train_split_path,
            validation_split_path,
            data_root
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_SUCCESS ||
        gs_baseline_mini224_subset1000_extract_train_rows(
            &pipeline,
            0,
            1,
            one_row,
            GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_SUCCESS ||
        !gs_test_values_are_float01(
            one_row,
            GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT
        )) {

        goto cleanup;
    }

    gs_baseline_mini224_subset1000_pipeline_free(&pipeline);

    if (pipeline.implementation != NULL) {
        goto cleanup;
    }

    printf("REOPEN_AFTER_FREE=true\n");
    rc = 0;

cleanup:
    gs_baseline_mini224_subset1000_pipeline_free(&pipeline);
    gs_baseline_mini224_subset1000_pipeline_free(&pipeline);
    return rc;
}


static int test_train_only_standardization(
    const float *train_raw,
    const float *validation_raw
)
{
    const size_t train_value_count =
        GS_TEST_TRAIN_ROWS * GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT;

    const size_t validation_value_count =
        GS_TEST_VALIDATION_ROWS *
        GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT;

    float *train_standardized = NULL;
    float *validation_standardized = NULL;
    double *mean = NULL;
    double *stddev = NULL;
    double *mean_copy = NULL;
    double *stddev_copy = NULL;
    unsigned char *constant_mask = NULL;
    unsigned char *constant_mask_copy = NULL;
    int rc = 1;

    train_standardized = (float *)malloc(
        train_value_count * sizeof(float)
    );
    validation_standardized = (float *)malloc(
        validation_value_count * sizeof(float)
    );
    mean = (double *)malloc(
        GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT * sizeof(double)
    );
    stddev = (double *)malloc(
        GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT * sizeof(double)
    );
    mean_copy = (double *)malloc(
        GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT * sizeof(double)
    );
    stddev_copy = (double *)malloc(
        GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT * sizeof(double)
    );
    constant_mask = (unsigned char *)malloc(
        GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT * sizeof(unsigned char)
    );
    constant_mask_copy = (unsigned char *)malloc(
        GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT * sizeof(unsigned char)
    );

    if (train_standardized == NULL ||
        validation_standardized == NULL ||
        mean == NULL || stddev == NULL ||
        mean_copy == NULL || stddev_copy == NULL ||
        constant_mask == NULL || constant_mask_copy == NULL) {

        goto cleanup;
    }

    if (gs_baseline_standardization_fit(
            train_raw,
            GS_TEST_TRAIN_ROWS,
            GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT,
            mean,
            stddev,
            constant_mask,
            GS_BASELINE_STANDARDIZATION_DEFAULT_CONSTANT_EPSILON
        ) != GS_BASELINE_STANDARDIZATION_SUCCESS ||
        gs_baseline_standardization_transform(
            train_raw,
            GS_TEST_TRAIN_ROWS,
            GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT,
            mean,
            stddev,
            constant_mask,
            train_standardized
        ) != GS_BASELINE_STANDARDIZATION_SUCCESS) {

        goto cleanup;
    }

    memcpy(
        mean_copy,
        mean,
        GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT * sizeof(double)
    );
    memcpy(
        stddev_copy,
        stddev,
        GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT * sizeof(double)
    );
    memcpy(
        constant_mask_copy,
        constant_mask,
        GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT * sizeof(unsigned char)
    );

    if (gs_baseline_standardization_transform(
            validation_raw,
            GS_TEST_VALIDATION_ROWS,
            GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT,
            mean,
            stddev,
            constant_mask,
            validation_standardized
        ) != GS_BASELINE_STANDARDIZATION_SUCCESS ||
        memcmp(
            mean,
            mean_copy,
            GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT * sizeof(double)
        ) != 0 ||
        memcmp(
            stddev,
            stddev_copy,
            GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT * sizeof(double)
        ) != 0 ||
        memcmp(
            constant_mask,
            constant_mask_copy,
            GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT * sizeof(unsigned char)
        ) != 0 ||
        !gs_test_values_are_finite(
            train_standardized,
            train_value_count
        ) ||
        !gs_test_values_are_finite(
            validation_standardized,
            validation_value_count
        )) {

        goto cleanup;
    }

    printf("SCALER_UNCHANGED_AFTER_VALIDATION=true\n");
    rc = 0;

cleanup:
    free(constant_mask_copy);
    free(constant_mask);
    free(stddev_copy);
    free(mean_copy);
    free(stddev);
    free(mean);
    free(validation_standardized);
    free(train_standardized);
    return rc;
}


static int test_invalid_ranges_buffers_and_overflow(
    GSBaselineMini224Subset1000Pipeline *pipeline
)
{
    GSBaselineMini224Subset1000Pipeline empty_pipeline = {0};
    float one_row[GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT];
    size_t manifest_index;

    gs_baseline_mini224_subset1000_pipeline_init(&empty_pipeline);

    if (gs_baseline_mini224_subset1000_extract_train_rows(
            NULL,
            0,
            1,
            one_row,
            GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_NULL_ARGUMENT ||
        gs_baseline_mini224_subset1000_extract_train_rows(
            pipeline,
            0,
            1,
            NULL,
            GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_NULL_ARGUMENT ||
        gs_baseline_mini224_subset1000_extract_train_rows(
            &empty_pipeline,
            0,
            1,
            one_row,
            GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_STATE ||
        gs_baseline_mini224_subset1000_extract_train_rows(
            pipeline,
            0,
            0,
            one_row,
            GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_RANGE ||
        gs_baseline_mini224_subset1000_extract_train_rows(
            pipeline,
            GS_BASELINE_MINI224_SUBSET1000_OFFICIAL_TRAIN_COUNT,
            1,
            one_row,
            GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_RANGE ||
        gs_baseline_mini224_subset1000_extract_train_rows(
            pipeline,
            GS_BASELINE_MINI224_SUBSET1000_OFFICIAL_TRAIN_COUNT - 1,
            2,
            one_row,
            GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_RANGE ||
        gs_baseline_mini224_subset1000_extract_validation_rows(
            pipeline,
            GS_BASELINE_MINI224_SUBSET1000_OFFICIAL_VALIDATION_COUNT,
            1,
            one_row,
            GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_RANGE ||
        gs_baseline_mini224_subset1000_extract_validation_rows(
            pipeline,
            GS_BASELINE_MINI224_SUBSET1000_OFFICIAL_VALIDATION_COUNT - 1,
            2,
            one_row,
            GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_RANGE ||
        gs_baseline_mini224_subset1000_extract_train_rows(
            pipeline,
            0,
            1,
            one_row,
            GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT - 1
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_BUFFER_SIZE ||
        gs_baseline_mini224_subset1000_extract_train_rows(
            pipeline,
            0,
            SIZE_MAX,
            one_row,
            GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_OVERFLOW ||
        gs_baseline_mini224_subset1000_train_manifest_index(
            pipeline,
            GS_BASELINE_MINI224_SUBSET1000_OFFICIAL_TRAIN_COUNT,
            &manifest_index
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_RANGE ||
        gs_baseline_mini224_subset1000_train_manifest_index(
            pipeline,
            0,
            NULL
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_NULL_ARGUMENT) {

        gs_baseline_mini224_subset1000_pipeline_free(&empty_pipeline);
        return 1;
    }

    gs_baseline_mini224_subset1000_pipeline_free(&empty_pipeline);
    gs_baseline_mini224_subset1000_pipeline_free(&empty_pipeline);
    return 0;
}


static int test_synthetic_pool14_contract(void)
{
    float *input = NULL;
    float *input_copy = NULL;
    float features[GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT];
    size_t input_count = 0u;
    size_t pooled_side = 0u;
    size_t feature_count = 0u;
    size_t channel;
    size_t row;
    size_t col;
    int rc = 1;

    input = (float *)malloc(
        GS_BASELINE_MINI224_SUBSET1000_INPUT_COUNT * sizeof(float)
    );
    input_copy = (float *)malloc(
        GS_BASELINE_MINI224_SUBSET1000_INPUT_COUNT * sizeof(float)
    );

    if (input == NULL || input_copy == NULL ||
        gs_baseline_average_pool_dimensions(
            GS_BASELINE_MINI224_SUBSET1000_CHANNELS,
            GS_BASELINE_MINI224_SUBSET1000_SIDE,
            GS_BASELINE_MINI224_SUBSET1000_POOL_SIDE,
            &input_count,
            &pooled_side,
            &feature_count
        ) != GS_BASELINE_AVERAGE_POOL_SUCCESS ||
        input_count != GS_BASELINE_MINI224_SUBSET1000_INPUT_COUNT ||
        pooled_side != GS_BASELINE_MINI224_SUBSET1000_POOLED_SIDE ||
        feature_count != GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT) {

        goto cleanup;
    }

    for (channel = 0;
         channel < GS_BASELINE_MINI224_SUBSET1000_CHANNELS;
         channel++) {

        for (row = 0;
             row < GS_BASELINE_MINI224_SUBSET1000_SIDE;
             row++) {

            for (col = 0;
                 col < GS_BASELINE_MINI224_SUBSET1000_SIDE;
                 col++) {

                unsigned int value =
                    (unsigned int)(channel * 50u) +
                    (unsigned int)(
                        (row / GS_BASELINE_MINI224_SUBSET1000_POOL_SIDE) *
                        2u
                    ) +
                    (unsigned int)(
                        col / GS_BASELINE_MINI224_SUBSET1000_POOL_SIDE
                    );

                size_t index =
                    channel *
                        GS_BASELINE_MINI224_SUBSET1000_SIDE *
                        GS_BASELINE_MINI224_SUBSET1000_SIDE +
                    row * GS_BASELINE_MINI224_SUBSET1000_SIDE +
                    col;

                input[index] = (float)value / 255.0f;
            }
        }
    }

    memcpy(
        input_copy,
        input,
        GS_BASELINE_MINI224_SUBSET1000_INPUT_COUNT * sizeof(float)
    );

    if (gs_baseline_average_pool_extract(
            input,
            GS_BASELINE_MINI224_SUBSET1000_INPUT_COUNT,
            GS_BASELINE_MINI224_SUBSET1000_CHANNELS,
            GS_BASELINE_MINI224_SUBSET1000_SIDE,
            GS_BASELINE_MINI224_SUBSET1000_POOL_SIDE,
            features,
            GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT
        ) != GS_BASELINE_AVERAGE_POOL_SUCCESS ||
        memcmp(
            input,
            input_copy,
            GS_BASELINE_MINI224_SUBSET1000_INPUT_COUNT * sizeof(float)
        ) != 0) {

        goto cleanup;
    }

    for (channel = 0;
         channel < GS_BASELINE_MINI224_SUBSET1000_CHANNELS;
         channel++) {

        size_t pooled_row;

        for (pooled_row = 0;
             pooled_row < GS_BASELINE_MINI224_SUBSET1000_POOLED_SIDE;
             pooled_row++) {

            size_t pooled_col;

            for (pooled_col = 0;
                 pooled_col < GS_BASELINE_MINI224_SUBSET1000_POOLED_SIDE;
                 pooled_col++) {

                unsigned int value =
                    (unsigned int)(channel * 50u) +
                    (unsigned int)(pooled_row * 2u) +
                    (unsigned int)pooled_col;

                size_t feature =
                    channel *
                        GS_BASELINE_MINI224_SUBSET1000_POOLED_SIDE *
                        GS_BASELINE_MINI224_SUBSET1000_POOLED_SIDE +
                    pooled_row *
                        GS_BASELINE_MINI224_SUBSET1000_POOLED_SIDE +
                    pooled_col;

                if (features[feature] != (float)value / 255.0f) {
                    goto cleanup;
                }
            }
        }
    }

    rc = 0;

cleanup:
    free(input_copy);
    free(input);
    return rc;
}


int main(int argc, char **argv)
{
    GSBaselineMini224Subset1000Pipeline pipeline = {0};
    GSEventManifest independent_events = {0};
    GSTensorManifest independent_tensors = {0};
    GSSavedSplit independent_train = {0};
    GSSavedSplit independent_validation = {0};
    GSTensorSpec independent_spec;

    float *train_matrix = NULL;
    float *validation_matrix = NULL;
    float *direct_tensor = NULL;
    float *direct_features = NULL;

    const size_t train_value_count =
        GS_TEST_TRAIN_ROWS * GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT;

    const size_t validation_value_count =
        GS_TEST_VALIDATION_ROWS *
        GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT;

    double maximum_difference = 0.0;
    size_t index;
    int rc = 1;

    if (argc != 6) {
        fprintf(
            stderr,
            "Uso: test_baseline_mini224_subset1000_pipeline.exe "
            "manifest_events_272.csv "
            "manifest_tensors_224_central.csv "
            "subset224_train_800_seed123.csv "
            "subset224_val_200_seed123.csv data_root\n"
        );
        return 1;
    }

    if (test_constants_and_api_contract() != 0 ||
        test_synthetic_pool14_contract() != 0) {
        fprintf(stderr, "Errore: contratto costanti/API mini224_subset1000.\n");
        goto cleanup;
    }
    printf("[PASS] mini224_subset1000_constants_and_label_free_api\n");
    printf("[PASS] synthetic_pool14_normalization_order_edges_no_crop\n");

    if (test_invalid_initialization_and_free(
            argv[2],
            argv[3],
            argv[4],
            argv[5]
        ) != 0) {

        fprintf(stderr, "Errore: inizializzazione/free invalidi.\n");
        goto cleanup;
    }
    printf("[PASS] invalid_initialization_paths_and_idempotent_free\n");

    gs_baseline_mini224_subset1000_pipeline_init(&pipeline);

    if (gs_baseline_mini224_subset1000_pipeline_open(
            &pipeline,
            argv[1],
            argv[2],
            argv[3],
            argv[4],
            argv[5]
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_SUCCESS ||
        test_global_counts(&pipeline) != 0) {

        fprintf(stderr, "Errore: validazione globale mini224_subset1000.\n");
        goto cleanup;
    }
    printf("[PASS] global_official_counts_alignment_files_and_sizes\n");

    if (gs_tensor_spec_init(
            &independent_spec,
            GS_BASELINE_MINI224_SUBSET1000_SIDE
        ) != 0 ||
        gs_event_manifest_load(argv[1], &independent_events) != 0 ||
        gs_tensor_manifest_load(
            argv[2],
            &independent_spec,
            &independent_tensors
        ) != 0 ||
        gs_saved_split_load(
            argv[3],
            "train",
            &independent_events,
            &independent_train
        ) != 0 ||
        gs_saved_split_load(
            argv[4],
            "validation",
            &independent_events,
            &independent_validation
        ) != 0) {

        fprintf(stderr, "Errore: caricamento diretto indipendente.\n");
        goto cleanup;
    }

    if (test_subset_tensor_path_is_exact(
            &pipeline,
            &independent_tensors,
            &independent_train,
            argv[3]
        ) != 0) {

        fprintf(stderr, "Errore: prova tensor_path subset 224.\n");
        goto cleanup;
    }
    printf("[PASS] subset_224_tensor_path_exact\n");

    train_matrix = (float *)malloc(
        (train_value_count + GS_TEST_GUARD_VALUES) * sizeof(float)
    );
    validation_matrix = (float *)malloc(
        (validation_value_count + GS_TEST_GUARD_VALUES) * sizeof(float)
    );
    direct_tensor = (float *)malloc(
        GS_BASELINE_MINI224_SUBSET1000_INPUT_COUNT * sizeof(float)
    );
    direct_features = (float *)malloc(
        GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT * sizeof(float)
    );

    if (train_matrix == NULL || validation_matrix == NULL ||
        direct_tensor == NULL || direct_features == NULL) {

        fprintf(stderr, "Errore: memoria insufficiente nel test.\n");
        goto cleanup;
    }

    for (index = 0;
         index < train_value_count + GS_TEST_GUARD_VALUES;
         index++) {

        train_matrix[index] =
            index < train_value_count ? NAN : GS_TEST_GUARD_VALUE;
    }

    for (index = 0;
         index < validation_value_count + GS_TEST_GUARD_VALUES;
         index++) {

        validation_matrix[index] =
            index < validation_value_count ? NAN : GS_TEST_GUARD_VALUE;
    }

    if (gs_baseline_mini224_subset1000_extract_train_rows(
            &pipeline,
            0,
            GS_TEST_TRAIN_ROWS,
            train_matrix,
            train_value_count + GS_TEST_GUARD_VALUES
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_SUCCESS ||
        !gs_test_values_are_float01(train_matrix, train_value_count) ||
        !gs_test_guards_are_unchanged(
            train_matrix,
            train_value_count
        )) {

        fprintf(stderr, "Errore: estrazione ridotta train mini224_subset1000.\n");
        goto cleanup;
    }
    printf("[PASS] extract_first_8_train_rows_shape_order_range_guard\n");

    if (gs_baseline_mini224_subset1000_extract_validation_rows(
            &pipeline,
            0,
            GS_TEST_VALIDATION_ROWS,
            validation_matrix,
            validation_value_count + GS_TEST_GUARD_VALUES
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_SUCCESS ||
        !gs_test_values_are_float01(
            validation_matrix,
            validation_value_count
        ) ||
        !gs_test_guards_are_unchanged(
            validation_matrix,
            validation_value_count
        )) {

        fprintf(
            stderr,
            "Errore: estrazione ridotta validation mini224_subset1000.\n"
        );
        goto cleanup;
    }
    printf(
        "[PASS] extract_first_4_validation_rows_shape_order_range_guard\n"
    );

    if (gs_test_direct_compare(
            &pipeline,
            &independent_tensors,
            &independent_train,
            gs_baseline_mini224_subset1000_train_manifest_index,
            argv[5],
            train_matrix,
            0,
            GS_TEST_TRAIN_ROWS,
            direct_tensor,
            direct_features,
            &maximum_difference
        ) != 0 ||
        gs_test_direct_compare(
            &pipeline,
            &independent_tensors,
            &independent_validation,
            gs_baseline_mini224_subset1000_validation_manifest_index,
            argv[5],
            validation_matrix,
            0,
            GS_TEST_VALIDATION_ROWS,
            direct_tensor,
            direct_features,
            &maximum_difference
        ) != 0) {

        fprintf(stderr, "Errore: confronto diretto non bitwise.\n");
        goto cleanup;
    }

    printf("DIRECT_COMPARE_ROWS=12\n");
    printf("DIRECT_COMPARE_BITWISE=true\n");
    printf(
        "DIRECT_COMPARE_MAX_ABS_DIFF=%.9g\n",
        maximum_difference
    );
    printf("[PASS] direct_bitwise_comparison_all_12_rows\n");

    if (test_repeat_and_chunking(
            &pipeline,
            train_matrix
        ) != 0) {

        fprintf(stderr, "Errore: determinismo/chunking mini224_subset1000.\n");
        goto cleanup;
    }
    printf("[PASS] repeated_and_chunked_train_extraction_bitwise\n");

    if (test_split_end_boundaries(
            &pipeline,
            &independent_tensors,
            &independent_train,
            &independent_validation,
            argv[5],
            direct_tensor,
            direct_features
        ) != 0) {

        fprintf(stderr, "Errore: frontiera finale degli split.\n");
        goto cleanup;
    }
    printf("[PASS] split_end_boundaries_direct_compare_and_guards\n");

    if (test_train_only_standardization(
            train_matrix,
            validation_matrix
        ) != 0) {

        fprintf(stderr, "Errore: interoperabilita' standardizzazione.\n");
        goto cleanup;
    }
    printf("[PASS] train_only_standardization_no_validation_refit\n");

    if (test_invalid_ranges_buffers_and_overflow(&pipeline) != 0) {
        fprintf(stderr, "Errore: contratto range/buffer/overflow.\n");
        goto cleanup;
    }
    printf("[PASS] null_zero_range_buffer_overflow_and_state_errors\n");

    if (test_context_lifecycle(
            argv[1],
            argv[2],
            argv[3],
            argv[4],
            argv[5]
        ) != 0) {

        fprintf(stderr, "Errore: ciclo di vita/reopen mini224_subset1000.\n");
        goto cleanup;
    }
    printf("[PASS] lifecycle_second_open_free_reopen_extract\n");

    printf("EVENT_COUNT=%d\n", GS_BASELINE_MINI224_SUBSET1000_OFFICIAL_EVENT_COUNT);
    printf("TRAIN_COUNT=%d\n", GS_BASELINE_MINI224_SUBSET1000_OFFICIAL_TRAIN_COUNT);
    printf(
        "VALIDATION_COUNT=%d\n",
        GS_BASELINE_MINI224_SUBSET1000_OFFICIAL_VALIDATION_COUNT
    );
    printf("TRAIN_ROWS_TESTED=%d\n", GS_TEST_TRAIN_ROWS);
    printf("VALIDATION_ROWS_TESTED=%d\n", GS_TEST_VALIDATION_ROWS);
    printf("FEATURE_DIM=%d\n", GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT);
    printf("ALL MINI224_SUBSET1000 PIPELINE TESTS PASSED\n");

    rc = 0;

cleanup:
    free(direct_features);
    free(direct_tensor);
    free(validation_matrix);
    free(train_matrix);

    gs_saved_split_free(&independent_validation);
    gs_saved_split_free(&independent_train);
    gs_tensor_manifest_free(&independent_tensors);
    gs_event_manifest_free(&independent_events);

    gs_baseline_mini224_subset1000_pipeline_free(&pipeline);
    gs_baseline_mini224_subset1000_pipeline_free(&pipeline);

    return rc;
}
