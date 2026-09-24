#include "baseline_mini112_pipeline.h"

#include "baseline_average_pool_features.h"
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
#include <sys/stat.h>


typedef struct {
    GSEventManifest event_manifest;
    GSTensorManifest tensor_manifest;
    GSSavedSplit train_split;
    GSSavedSplit validation_split;
    GSTensorSpec tensor_spec;
    char data_root[GS_BASELINE_MINI112_PATH_LENGTH];
    float *tensor_buffer;
} GSBaselineMini112PipelineImplementation;


static int gs_mini112_checked_product(
    size_t left,
    size_t right,
    size_t *product
)
{
    if (product == NULL) {
        return GS_BASELINE_MINI112_PIPELINE_ERROR_NULL_ARGUMENT;
    }

    if (left != 0 && right > SIZE_MAX / left) {
        return GS_BASELINE_MINI112_PIPELINE_ERROR_OVERFLOW;
    }

    *product = left * right;
    return GS_BASELINE_MINI112_PIPELINE_SUCCESS;
}


static int gs_mini112_join_path(
    const char *data_root,
    const char *relative_path,
    char *joined_path,
    size_t joined_path_size
)
{
    size_t root_length;
    size_t relative_length;
    size_t required;
    size_t position;
    int needs_separator;

    if (data_root == NULL || relative_path == NULL ||
        joined_path == NULL || joined_path_size == 0) {

        return GS_BASELINE_MINI112_PIPELINE_ERROR_NULL_ARGUMENT;
    }

    root_length = strlen(data_root);
    relative_length = strlen(relative_path);

    needs_separator =
        root_length != 0 &&
        data_root[root_length - 1] != '/' &&
        data_root[root_length - 1] != '\\';

    if (root_length > SIZE_MAX - relative_length) {
        return GS_BASELINE_MINI112_PIPELINE_ERROR_OVERFLOW;
    }

    required = root_length + relative_length;

    if (needs_separator) {
        if (required == SIZE_MAX) {
            return GS_BASELINE_MINI112_PIPELINE_ERROR_OVERFLOW;
        }
        required++;
    }

    if (required >= joined_path_size) {
        return GS_BASELINE_MINI112_PIPELINE_ERROR_PATH;
    }

    position = 0;

    if (root_length != 0) {
        memcpy(joined_path, data_root, root_length);
        position = root_length;
    }

    if (needs_separator) {
        joined_path[position] = '/';
        position++;
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
    return GS_BASELINE_MINI112_PIPELINE_SUCCESS;
}


static int gs_mini112_spec_is_valid(const GSTensorSpec *spec)
{
    return spec != NULL &&
           gs_tensor_spec_is_valid(spec) &&
           spec->channels == GS_BASELINE_MINI112_CHANNELS &&
           spec->height == GS_BASELINE_MINI112_SIDE &&
           spec->width == GS_BASELINE_MINI112_SIDE &&
           spec->pixels_per_channel ==
               (size_t)GS_BASELINE_MINI112_SIDE *
               (size_t)GS_BASELINE_MINI112_SIDE &&
           spec->values_per_event == GS_BASELINE_MINI112_INPUT_COUNT &&
           spec->bytes_u8 == GS_BASELINE_MINI112_TENSOR_BYTES &&
           spec->bytes_f32 ==
               GS_BASELINE_MINI112_INPUT_COUNT * sizeof(float);
}


static int gs_mini112_tensor_record_is_valid(
    const GSTensorManifestRecord *record
)
{
    return record != NULL &&
           record->event_id[0] != '\0' &&
           record->tensor_path[0] != '\0' &&
           record->channels == GS_BASELINE_MINI112_CHANNELS &&
           record->height == GS_BASELINE_MINI112_SIDE &&
           record->width == GS_BASELINE_MINI112_SIDE &&
           strcmp(record->dtype, "uint8") == 0 &&
           strcmp(record->layout, "channel_major") == 0 &&
           strcmp(record->channel_order, "0.5;1.0;2.0;4.0") == 0 &&
           strcmp(record->extra_value, "derived_from_224_central") == 0;
}


static int gs_mini112_compare_paths(
    const void *left,
    const void *right
)
{
    const char *const *left_path = (const char *const *)left;
    const char *const *right_path = (const char *const *)right;

    return strcmp(*left_path, *right_path);
}


static void gs_mini112_implementation_free(
    GSBaselineMini112PipelineImplementation *implementation
)
{
    if (implementation == NULL) {
        return;
    }

    free(implementation->tensor_buffer);
    implementation->tensor_buffer = NULL;

    gs_saved_split_free(&implementation->validation_split);
    gs_saved_split_free(&implementation->train_split);
    gs_tensor_manifest_free(&implementation->tensor_manifest);
    gs_event_manifest_free(&implementation->event_manifest);

    free(implementation);
}


static int gs_mini112_validate_global(
    const GSBaselineMini112PipelineImplementation *implementation
)
{
    const char **path_references = NULL;
    unsigned char *owner = NULL;
    size_t pointer_bytes;
    size_t index;
    size_t expected_input_count;
    size_t expected_pooled_side;
    size_t expected_feature_count;
    int result = GS_BASELINE_MINI112_PIPELINE_ERROR_ALIGNMENT;

    if (implementation == NULL ||
        implementation->event_manifest.records == NULL ||
        implementation->tensor_manifest.records == NULL ||
        implementation->train_split.indices == NULL ||
        implementation->validation_split.indices == NULL) {

        return GS_BASELINE_MINI112_PIPELINE_ERROR_STATE;
    }

    if (implementation->event_manifest.count !=
            GS_BASELINE_MINI112_OFFICIAL_EVENT_COUNT ||
        implementation->tensor_manifest.count !=
            GS_BASELINE_MINI112_OFFICIAL_EVENT_COUNT ||
        implementation->train_split.count !=
            GS_BASELINE_MINI112_OFFICIAL_TRAIN_COUNT ||
        implementation->validation_split.count !=
            GS_BASELINE_MINI112_OFFICIAL_VALIDATION_COUNT) {

        return GS_BASELINE_MINI112_PIPELINE_ERROR_COUNTS;
    }

    if (strcmp(implementation->train_split.name, "train") != 0 ||
        strcmp(
            implementation->validation_split.name,
            "validation"
        ) != 0 ||
        strcmp(
            implementation->tensor_manifest.extra_column_name,
            "origin"
        ) != 0 ||
        !gs_mini112_spec_is_valid(&implementation->tensor_spec) ||
        !gs_mini112_spec_is_valid(
            &implementation->tensor_manifest.spec
        )) {

        return GS_BASELINE_MINI112_PIPELINE_ERROR_SPEC;
    }

    if (gs_baseline_average_pool_dimensions(
            GS_BASELINE_MINI112_CHANNELS,
            GS_BASELINE_MINI112_SIDE,
            GS_BASELINE_MINI112_POOL_SIDE,
            &expected_input_count,
            &expected_pooled_side,
            &expected_feature_count
        ) != GS_BASELINE_AVERAGE_POOL_SUCCESS ||
        expected_input_count != GS_BASELINE_MINI112_INPUT_COUNT ||
        expected_pooled_side != GS_BASELINE_MINI112_POOLED_SIDE ||
        expected_feature_count != GS_BASELINE_MINI112_FEATURE_COUNT) {

        return GS_BASELINE_MINI112_PIPELINE_ERROR_SPEC;
    }

    if (gs_mini112_checked_product(
            implementation->tensor_manifest.count,
            sizeof(*path_references),
            &pointer_bytes
        ) != GS_BASELINE_MINI112_PIPELINE_SUCCESS) {

        return GS_BASELINE_MINI112_PIPELINE_ERROR_OVERFLOW;
    }

    path_references = (const char **)malloc(pointer_bytes);
    owner = (unsigned char *)calloc(
        GS_BASELINE_MINI112_OFFICIAL_EVENT_COUNT,
        sizeof(unsigned char)
    );

    if (path_references == NULL || owner == NULL) {
        result = GS_BASELINE_MINI112_PIPELINE_ERROR_ALLOCATION;
        goto cleanup;
    }

    for (index = 0;
         index < GS_BASELINE_MINI112_OFFICIAL_EVENT_COUNT;
         index++) {

        const GSEventRecord *event_record =
            gs_event_manifest_get(
                &implementation->event_manifest,
                index
            );

        const GSTensorManifestRecord *tensor_record =
            gs_tensor_manifest_get(
                &implementation->tensor_manifest,
                index
            );

        char joined_path[GS_BASELINE_MINI112_PATH_LENGTH];
        struct stat file_information;
        int join_result;

        if (event_record == NULL || tensor_record == NULL ||
            event_record->event_id[0] == '\0' ||
            (size_t)event_record->event_index != index + 1 ||
            (size_t)tensor_record->event_index != index + 1 ||
            event_record->event_index != tensor_record->event_index ||
            strcmp(event_record->event_id, tensor_record->event_id) != 0 ||
            !gs_mini112_tensor_record_is_valid(tensor_record)) {

            result = GS_BASELINE_MINI112_PIPELINE_ERROR_ALIGNMENT;
            goto cleanup;
        }

        path_references[index] = tensor_record->tensor_path;

        join_result = gs_mini112_join_path(
            implementation->data_root,
            tensor_record->tensor_path,
            joined_path,
            sizeof(joined_path)
        );

        if (join_result != GS_BASELINE_MINI112_PIPELINE_SUCCESS) {
            result = join_result;
            goto cleanup;
        }

        if (stat(joined_path, &file_information) != 0) {
            result = GS_BASELINE_MINI112_PIPELINE_ERROR_FILE_NOT_FOUND;
            goto cleanup;
        }

        if (file_information.st_size < 0 ||
            (uintmax_t)file_information.st_size !=
                (uintmax_t)GS_BASELINE_MINI112_TENSOR_BYTES) {

            result = GS_BASELINE_MINI112_PIPELINE_ERROR_FILE_SIZE;
            goto cleanup;
        }
    }

    qsort(
        path_references,
        implementation->tensor_manifest.count,
        sizeof(*path_references),
        gs_mini112_compare_paths
    );

    for (index = 1;
         index < implementation->tensor_manifest.count;
         index++) {

        if (strcmp(
                path_references[index - 1],
                path_references[index]
            ) == 0) {

            result = GS_BASELINE_MINI112_PIPELINE_ERROR_DUPLICATE_PATH;
            goto cleanup;
        }
    }

    for (index = 0;
         index < implementation->train_split.count;
         index++) {

        size_t manifest_index =
            implementation->train_split.indices[index];

        if (manifest_index >=
                GS_BASELINE_MINI112_OFFICIAL_EVENT_COUNT) {

            result = GS_BASELINE_MINI112_PIPELINE_ERROR_RANGE;
            goto cleanup;
        }

        if (owner[manifest_index] != 0u) {
            result = GS_BASELINE_MINI112_PIPELINE_ERROR_ALIGNMENT;
            goto cleanup;
        }

        owner[manifest_index] = 1u;
    }

    for (index = 0;
         index < implementation->validation_split.count;
         index++) {

        size_t manifest_index =
            implementation->validation_split.indices[index];

        if (manifest_index >=
                GS_BASELINE_MINI112_OFFICIAL_EVENT_COUNT) {

            result = GS_BASELINE_MINI112_PIPELINE_ERROR_RANGE;
            goto cleanup;
        }

        if (owner[manifest_index] != 0u) {
            result = GS_BASELINE_MINI112_PIPELINE_ERROR_ALIGNMENT;
            goto cleanup;
        }

        owner[manifest_index] = 2u;
    }

    for (index = 0;
         index < GS_BASELINE_MINI112_OFFICIAL_EVENT_COUNT;
         index++) {

        if (owner[index] == 0u) {
            result = GS_BASELINE_MINI112_PIPELINE_ERROR_ALIGNMENT;
            goto cleanup;
        }
    }

    result = GS_BASELINE_MINI112_PIPELINE_SUCCESS;

cleanup:
    free(owner);
    free(path_references);
    return result;
}


void gs_baseline_mini112_pipeline_init(
    GSBaselineMini112Pipeline *pipeline
)
{
    if (pipeline == NULL) {
        return;
    }

    pipeline->implementation = NULL;
}


int gs_baseline_mini112_pipeline_open(
    GSBaselineMini112Pipeline *pipeline,
    const char *event_manifest_path,
    const char *tensor_manifest_path,
    const char *train_split_path,
    const char *validation_split_path,
    const char *data_root
)
{
    GSBaselineMini112PipelineImplementation *implementation;
    size_t root_length;
    size_t tensor_buffer_bytes;
    int result;

    if (pipeline == NULL || event_manifest_path == NULL ||
        tensor_manifest_path == NULL || train_split_path == NULL ||
        validation_split_path == NULL || data_root == NULL) {

        return GS_BASELINE_MINI112_PIPELINE_ERROR_NULL_ARGUMENT;
    }

    if (pipeline->implementation != NULL) {
        return GS_BASELINE_MINI112_PIPELINE_ERROR_STATE;
    }

    root_length = strlen(data_root);
    if (root_length >= GS_BASELINE_MINI112_PATH_LENGTH) {
        return GS_BASELINE_MINI112_PIPELINE_ERROR_PATH;
    }

    implementation =
        (GSBaselineMini112PipelineImplementation *)calloc(
            1,
            sizeof(*implementation)
        );

    if (implementation == NULL) {
        return GS_BASELINE_MINI112_PIPELINE_ERROR_ALLOCATION;
    }

    memcpy(
        implementation->data_root,
        data_root,
        root_length + 1
    );

    if (gs_tensor_spec_init(
            &implementation->tensor_spec,
            GS_BASELINE_MINI112_SIDE
        ) != 0) {

        result = GS_BASELINE_MINI112_PIPELINE_ERROR_SPEC;
        goto fail;
    }

    if (gs_event_manifest_load(
            event_manifest_path,
            &implementation->event_manifest
        ) != 0) {

        result = GS_BASELINE_MINI112_PIPELINE_ERROR_EVENT_MANIFEST;
        goto fail;
    }

    if (gs_tensor_manifest_load(
            tensor_manifest_path,
            &implementation->tensor_spec,
            &implementation->tensor_manifest
        ) != 0) {

        result = GS_BASELINE_MINI112_PIPELINE_ERROR_TENSOR_MANIFEST;
        goto fail;
    }

    if (gs_saved_split_load(
            train_split_path,
            "train",
            &implementation->event_manifest,
            &implementation->train_split
        ) != 0) {

        result = GS_BASELINE_MINI112_PIPELINE_ERROR_TRAIN_SPLIT;
        goto fail;
    }

    if (gs_saved_split_load(
            validation_split_path,
            "validation",
            &implementation->event_manifest,
            &implementation->validation_split
        ) != 0) {

        result = GS_BASELINE_MINI112_PIPELINE_ERROR_VALIDATION_SPLIT;
        goto fail;
    }

    result = gs_mini112_validate_global(implementation);
    if (result != GS_BASELINE_MINI112_PIPELINE_SUCCESS) {
        goto fail;
    }

    result = gs_mini112_checked_product(
        GS_BASELINE_MINI112_INPUT_COUNT,
        sizeof(float),
        &tensor_buffer_bytes
    );

    if (result != GS_BASELINE_MINI112_PIPELINE_SUCCESS) {
        goto fail;
    }

    implementation->tensor_buffer =
        (float *)malloc(tensor_buffer_bytes);

    if (implementation->tensor_buffer == NULL) {
        result = GS_BASELINE_MINI112_PIPELINE_ERROR_ALLOCATION;
        goto fail;
    }

    pipeline->implementation = implementation;
    return GS_BASELINE_MINI112_PIPELINE_SUCCESS;

fail:
    gs_mini112_implementation_free(implementation);
    return result;
}


static const GSBaselineMini112PipelineImplementation *
gs_mini112_const_implementation(
    const GSBaselineMini112Pipeline *pipeline
)
{
    if (pipeline == NULL || pipeline->implementation == NULL) {
        return NULL;
    }

    return (const GSBaselineMini112PipelineImplementation *)
        pipeline->implementation;
}


int gs_baseline_mini112_pipeline_counts(
    const GSBaselineMini112Pipeline *pipeline,
    size_t *event_count,
    size_t *train_count,
    size_t *validation_count
)
{
    const GSBaselineMini112PipelineImplementation *implementation =
        gs_mini112_const_implementation(pipeline);

    if (event_count == NULL || train_count == NULL ||
        validation_count == NULL) {

        return GS_BASELINE_MINI112_PIPELINE_ERROR_NULL_ARGUMENT;
    }

    if (implementation == NULL) {
        return GS_BASELINE_MINI112_PIPELINE_ERROR_STATE;
    }

    *event_count = implementation->event_manifest.count;
    *train_count = implementation->train_split.count;
    *validation_count = implementation->validation_split.count;

    return GS_BASELINE_MINI112_PIPELINE_SUCCESS;
}


static int gs_mini112_manifest_index(
    const GSBaselineMini112Pipeline *pipeline,
    const GSSavedSplit *split,
    size_t split_pos,
    size_t *manifest_index
)
{
    if (pipeline == NULL || manifest_index == NULL) {
        return GS_BASELINE_MINI112_PIPELINE_ERROR_NULL_ARGUMENT;
    }

    if (pipeline->implementation == NULL || split == NULL ||
        split->indices == NULL) {

        return GS_BASELINE_MINI112_PIPELINE_ERROR_STATE;
    }

    if (split_pos >= split->count) {
        return GS_BASELINE_MINI112_PIPELINE_ERROR_RANGE;
    }

    *manifest_index = split->indices[split_pos];
    return GS_BASELINE_MINI112_PIPELINE_SUCCESS;
}


int gs_baseline_mini112_train_manifest_index(
    const GSBaselineMini112Pipeline *pipeline,
    size_t split_pos,
    size_t *manifest_index
)
{
    const GSBaselineMini112PipelineImplementation *implementation =
        gs_mini112_const_implementation(pipeline);

    return gs_mini112_manifest_index(
        pipeline,
        implementation == NULL ? NULL : &implementation->train_split,
        split_pos,
        manifest_index
    );
}


int gs_baseline_mini112_validation_manifest_index(
    const GSBaselineMini112Pipeline *pipeline,
    size_t split_pos,
    size_t *manifest_index
)
{
    const GSBaselineMini112PipelineImplementation *implementation =
        gs_mini112_const_implementation(pipeline);

    return gs_mini112_manifest_index(
        pipeline,
        implementation == NULL ?
            NULL :
            &implementation->validation_split,
        split_pos,
        manifest_index
    );
}


static int gs_mini112_extract_rows(
    GSBaselineMini112Pipeline *pipeline,
    const GSSavedSplit *split,
    size_t first_row,
    size_t row_count,
    float *features,
    size_t feature_capacity
)
{
    GSBaselineMini112PipelineImplementation *implementation;
    size_t required_features;
    size_t local_row;
    int product_result;

    if (pipeline == NULL || features == NULL) {
        return GS_BASELINE_MINI112_PIPELINE_ERROR_NULL_ARGUMENT;
    }

    implementation =
        (GSBaselineMini112PipelineImplementation *)
            pipeline->implementation;

    if (implementation == NULL || split == NULL ||
        split->indices == NULL ||
        implementation->tensor_buffer == NULL) {

        return GS_BASELINE_MINI112_PIPELINE_ERROR_STATE;
    }

    if (row_count == 0) {
        return GS_BASELINE_MINI112_PIPELINE_ERROR_RANGE;
    }

    product_result = gs_mini112_checked_product(
        row_count,
        GS_BASELINE_MINI112_FEATURE_COUNT,
        &required_features
    );

    if (product_result != GS_BASELINE_MINI112_PIPELINE_SUCCESS) {
        return product_result;
    }

    if (first_row >= split->count ||
        row_count > split->count - first_row) {

        return GS_BASELINE_MINI112_PIPELINE_ERROR_RANGE;
    }

    if (feature_capacity < required_features) {
        return GS_BASELINE_MINI112_PIPELINE_ERROR_BUFFER_SIZE;
    }

    for (local_row = 0; local_row < row_count; local_row++) {
        size_t split_pos = first_row + local_row;
        size_t manifest_index = split->indices[split_pos];
        const GSEventRecord *event_record;
        const GSTensorManifestRecord *tensor_record;
        float *destination;
        char joined_path[GS_BASELINE_MINI112_PATH_LENGTH];
        size_t value_index;
        int join_result;

        if (manifest_index >= implementation->event_manifest.count ||
            manifest_index >= implementation->tensor_manifest.count) {

            return GS_BASELINE_MINI112_PIPELINE_ERROR_RANGE;
        }

        event_record = gs_event_manifest_get(
            &implementation->event_manifest,
            manifest_index
        );

        tensor_record = gs_tensor_manifest_get(
            &implementation->tensor_manifest,
            manifest_index
        );

        if (event_record == NULL || tensor_record == NULL ||
            event_record->event_index != tensor_record->event_index ||
            strcmp(event_record->event_id, tensor_record->event_id) != 0 ||
            !gs_mini112_tensor_record_is_valid(tensor_record)) {

            return GS_BASELINE_MINI112_PIPELINE_ERROR_ALIGNMENT;
        }

        join_result = gs_mini112_join_path(
            implementation->data_root,
            tensor_record->tensor_path,
            joined_path,
            sizeof(joined_path)
        );

        if (join_result != GS_BASELINE_MINI112_PIPELINE_SUCCESS) {
            return join_result;
        }

        if (gs_tensor_read_float01(
                joined_path,
                &implementation->tensor_spec,
                implementation->tensor_buffer
            ) != 0) {

            return GS_BASELINE_MINI112_PIPELINE_ERROR_TENSOR_READ;
        }

        for (value_index = 0;
             value_index < GS_BASELINE_MINI112_INPUT_COUNT;
             value_index++) {

            if (!isfinite(implementation->tensor_buffer[value_index]) ||
                implementation->tensor_buffer[value_index] < 0.0f ||
                implementation->tensor_buffer[value_index] > 1.0f) {

                return GS_BASELINE_MINI112_PIPELINE_ERROR_TENSOR_VALUE;
            }
        }

        destination =
            features +
            local_row * GS_BASELINE_MINI112_FEATURE_COUNT;

        if (gs_baseline_average_pool_extract(
                implementation->tensor_buffer,
                GS_BASELINE_MINI112_INPUT_COUNT,
                GS_BASELINE_MINI112_CHANNELS,
                GS_BASELINE_MINI112_SIDE,
                GS_BASELINE_MINI112_POOL_SIDE,
                destination,
                GS_BASELINE_MINI112_FEATURE_COUNT
            ) != GS_BASELINE_AVERAGE_POOL_SUCCESS) {

            return
                GS_BASELINE_MINI112_PIPELINE_ERROR_FEATURE_EXTRACTION;
        }

        for (value_index = 0;
             value_index < GS_BASELINE_MINI112_FEATURE_COUNT;
             value_index++) {

            if (!isfinite(destination[value_index]) ||
                destination[value_index] < 0.0f ||
                destination[value_index] > 1.0f) {

                return GS_BASELINE_MINI112_PIPELINE_ERROR_TENSOR_VALUE;
            }
        }
    }

    return GS_BASELINE_MINI112_PIPELINE_SUCCESS;
}


int gs_baseline_mini112_extract_train_rows(
    GSBaselineMini112Pipeline *pipeline,
    size_t first_row,
    size_t row_count,
    float *features,
    size_t feature_capacity
)
{
    GSBaselineMini112PipelineImplementation *implementation;

    if (pipeline == NULL) {
        return GS_BASELINE_MINI112_PIPELINE_ERROR_NULL_ARGUMENT;
    }

    implementation =
        (GSBaselineMini112PipelineImplementation *)
            pipeline->implementation;

    return gs_mini112_extract_rows(
        pipeline,
        implementation == NULL ? NULL : &implementation->train_split,
        first_row,
        row_count,
        features,
        feature_capacity
    );
}


int gs_baseline_mini112_extract_validation_rows(
    GSBaselineMini112Pipeline *pipeline,
    size_t first_row,
    size_t row_count,
    float *features,
    size_t feature_capacity
)
{
    GSBaselineMini112PipelineImplementation *implementation;

    if (pipeline == NULL) {
        return GS_BASELINE_MINI112_PIPELINE_ERROR_NULL_ARGUMENT;
    }

    implementation =
        (GSBaselineMini112PipelineImplementation *)
            pipeline->implementation;

    return gs_mini112_extract_rows(
        pipeline,
        implementation == NULL ?
            NULL :
            &implementation->validation_split,
        first_row,
        row_count,
        features,
        feature_capacity
    );
}


const char *gs_baseline_mini112_pipeline_error_string(int error_code)
{
    switch (error_code) {
        case GS_BASELINE_MINI112_PIPELINE_SUCCESS:
            return "success";
        case GS_BASELINE_MINI112_PIPELINE_ERROR_NULL_ARGUMENT:
            return "null argument";
        case GS_BASELINE_MINI112_PIPELINE_ERROR_STATE:
            return "pipeline state is not ready";
        case GS_BASELINE_MINI112_PIPELINE_ERROR_ALLOCATION:
            return "memory allocation failed";
        case GS_BASELINE_MINI112_PIPELINE_ERROR_SPEC:
            return "mini112 tensor or pooling specification is invalid";
        case GS_BASELINE_MINI112_PIPELINE_ERROR_EVENT_MANIFEST:
            return "event manifest load failed";
        case GS_BASELINE_MINI112_PIPELINE_ERROR_TENSOR_MANIFEST:
            return "mini112 tensor manifest load failed";
        case GS_BASELINE_MINI112_PIPELINE_ERROR_TRAIN_SPLIT:
            return "train split load failed";
        case GS_BASELINE_MINI112_PIPELINE_ERROR_VALIDATION_SPLIT:
            return "validation split load failed";
        case GS_BASELINE_MINI112_PIPELINE_ERROR_COUNTS:
            return "official counts are invalid";
        case GS_BASELINE_MINI112_PIPELINE_ERROR_ALIGNMENT:
            return "manifest or split alignment is invalid";
        case GS_BASELINE_MINI112_PIPELINE_ERROR_DUPLICATE_PATH:
            return "mini112 tensor path is duplicated";
        case GS_BASELINE_MINI112_PIPELINE_ERROR_PATH:
            return "resolved path is too long";
        case GS_BASELINE_MINI112_PIPELINE_ERROR_FILE_NOT_FOUND:
            return "mini112 tensor file is missing";
        case GS_BASELINE_MINI112_PIPELINE_ERROR_FILE_SIZE:
            return "mini112 tensor file size is invalid";
        case GS_BASELINE_MINI112_PIPELINE_ERROR_RANGE:
            return "requested split interval is invalid";
        case GS_BASELINE_MINI112_PIPELINE_ERROR_OVERFLOW:
            return "size calculation overflow";
        case GS_BASELINE_MINI112_PIPELINE_ERROR_BUFFER_SIZE:
            return "buffer capacity is insufficient";
        case GS_BASELINE_MINI112_PIPELINE_ERROR_TENSOR_READ:
            return "mini112 tensor read failed";
        case GS_BASELINE_MINI112_PIPELINE_ERROR_TENSOR_VALUE:
            return "tensor or feature value is invalid";
        case GS_BASELINE_MINI112_PIPELINE_ERROR_FEATURE_EXTRACTION:
            return "average pooling feature extraction failed";
        default:
            return "unknown mini112 pipeline error";
    }
}


void gs_baseline_mini112_pipeline_free(
    GSBaselineMini112Pipeline *pipeline
)
{
    if (pipeline == NULL) {
        return;
    }

    gs_mini112_implementation_free(
        (GSBaselineMini112PipelineImplementation *)
            pipeline->implementation
    );

    pipeline->implementation = NULL;
}
