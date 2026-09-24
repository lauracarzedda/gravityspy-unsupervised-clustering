#include "baseline_mini75_pipeline.h"

#include "baseline_mini75_features.h"
#include "vae_tensor_io.h"

#include <math.h>
#include <stdint.h>
#include <string.h>


static int gs_pipeline_spec_is_mini75(
    const GSTensorSpec *spec
)
{
    return spec != NULL &&
           gs_tensor_spec_is_valid(spec) &&
           spec->channels == GS_BASELINE_MINI75_CHANNELS &&
           spec->height == GS_BASELINE_MINI75_SIDE &&
           spec->width == GS_BASELINE_MINI75_SIDE &&
           spec->values_per_event == GS_BASELINE_MINI75_INPUT_COUNT &&
           spec->bytes_u8 == GS_BASELINE_MINI75_INPUT_COUNT &&
           spec->bytes_f32 ==
               GS_BASELINE_MINI75_INPUT_COUNT * sizeof(float);
}


static int gs_pipeline_tensor_record_is_mini75(
    const GSTensorManifestRecord *record
)
{
    return record != NULL &&
           record->event_id[0] != '\0' &&
           record->tensor_path[0] != '\0' &&
           record->channels == GS_BASELINE_MINI75_CHANNELS &&
           record->height == GS_BASELINE_MINI75_SIDE &&
           record->width == GS_BASELINE_MINI75_SIDE &&
           strcmp(record->dtype, "uint8") == 0 &&
           strcmp(record->layout, "channel_major") == 0 &&
           strcmp(record->channel_order, "0.5;1.0;2.0;4.0") == 0 &&
           strcmp(record->extra_value, "derived_from_224_central") == 0;
}


static int gs_pipeline_join_path(
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

        return GS_BASELINE_MINI75_PIPELINE_ERROR_NULL_ARGUMENT;
    }

    root_length = strlen(data_root);
    relative_length = strlen(relative_path);

    needs_separator =
        root_length != 0 &&
        data_root[root_length - 1] != '/' &&
        data_root[root_length - 1] != '\\';

    if (root_length > SIZE_MAX - relative_length) {
        return GS_BASELINE_MINI75_PIPELINE_ERROR_OVERFLOW;
    }

    required = root_length + relative_length;

    if (needs_separator) {
        if (required == SIZE_MAX) {
            return GS_BASELINE_MINI75_PIPELINE_ERROR_OVERFLOW;
        }
        required++;
    }

    if (required >= joined_path_size) {
        return GS_BASELINE_MINI75_PIPELINE_ERROR_PATH;
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
    return GS_BASELINE_MINI75_PIPELINE_SUCCESS;
}


int gs_baseline_mini75_validate_alignment(
    const GSEventManifest *event_manifest,
    const GSTensorManifest *tensor_manifest,
    const GSSavedSplit *train_split,
    const GSSavedSplit *validation_split
)
{
    unsigned char owner[GS_BASELINE_MINI75_OFFICIAL_EVENT_COUNT];
    size_t index;

    if (event_manifest == NULL || tensor_manifest == NULL ||
        train_split == NULL || validation_split == NULL) {

        return GS_BASELINE_MINI75_PIPELINE_ERROR_NULL_ARGUMENT;
    }

    if (event_manifest->records == NULL ||
        tensor_manifest->records == NULL ||
        train_split->indices == NULL ||
        validation_split->indices == NULL) {

        return GS_BASELINE_MINI75_PIPELINE_ERROR_NULL_ARGUMENT;
    }

    if (event_manifest->count !=
            GS_BASELINE_MINI75_OFFICIAL_EVENT_COUNT ||
        tensor_manifest->count !=
            GS_BASELINE_MINI75_OFFICIAL_EVENT_COUNT ||
        train_split->count !=
            GS_BASELINE_MINI75_OFFICIAL_TRAIN_COUNT ||
        validation_split->count !=
            GS_BASELINE_MINI75_OFFICIAL_VALIDATION_COUNT) {

        return GS_BASELINE_MINI75_PIPELINE_ERROR_COUNTS;
    }

    if (strcmp(train_split->name, "train") != 0 ||
        strcmp(validation_split->name, "validation") != 0 ||
        strcmp(tensor_manifest->extra_column_name, "origin") != 0 ||
        !gs_pipeline_spec_is_mini75(&tensor_manifest->spec)) {

        return GS_BASELINE_MINI75_PIPELINE_ERROR_ALIGNMENT;
    }

    for (index = 0;
         index < GS_BASELINE_MINI75_OFFICIAL_EVENT_COUNT;
         index++) {

        const GSEventRecord *event_record =
            gs_event_manifest_get(event_manifest, index);

        const GSTensorManifestRecord *tensor_record =
            gs_tensor_manifest_get(tensor_manifest, index);

        if (event_record == NULL || tensor_record == NULL ||
            event_record->event_index != tensor_record->event_index ||
            strcmp(event_record->event_id, tensor_record->event_id) != 0 ||
            !gs_pipeline_tensor_record_is_mini75(tensor_record)) {

            return GS_BASELINE_MINI75_PIPELINE_ERROR_ALIGNMENT;
        }
    }

    memset(owner, 0, sizeof(owner));

    for (index = 0; index < train_split->count; index++) {
        size_t manifest_index = train_split->indices[index];

        if (manifest_index >= GS_BASELINE_MINI75_OFFICIAL_EVENT_COUNT) {
            return GS_BASELINE_MINI75_PIPELINE_ERROR_RANGE;
        }

        if (owner[manifest_index] != 0u) {
            return GS_BASELINE_MINI75_PIPELINE_ERROR_ALIGNMENT;
        }

        owner[manifest_index] = 1u;
    }

    for (index = 0; index < validation_split->count; index++) {
        size_t manifest_index = validation_split->indices[index];

        if (manifest_index >= GS_BASELINE_MINI75_OFFICIAL_EVENT_COUNT) {
            return GS_BASELINE_MINI75_PIPELINE_ERROR_RANGE;
        }

        if (owner[manifest_index] != 0u) {
            return GS_BASELINE_MINI75_PIPELINE_ERROR_ALIGNMENT;
        }

        owner[manifest_index] = 2u;
    }

    for (index = 0;
         index < GS_BASELINE_MINI75_OFFICIAL_EVENT_COUNT;
         index++) {

        if (owner[index] == 0u) {
            return GS_BASELINE_MINI75_PIPELINE_ERROR_ALIGNMENT;
        }
    }

    return GS_BASELINE_MINI75_PIPELINE_SUCCESS;
}


int gs_baseline_mini75_extract_split_rows(
    const char *data_root,
    const GSEventManifest *event_manifest,
    const GSTensorManifest *tensor_manifest,
    const GSSavedSplit *split,
    size_t first_row,
    size_t row_count,
    float *tensor_buffer,
    size_t tensor_buffer_count,
    float *raw_features,
    size_t raw_features_count
)
{
    size_t required_features;
    size_t local_row;

    if (data_root == NULL || event_manifest == NULL ||
        tensor_manifest == NULL || split == NULL ||
        tensor_buffer == NULL || raw_features == NULL) {

        return GS_BASELINE_MINI75_PIPELINE_ERROR_NULL_ARGUMENT;
    }

    if (event_manifest->records == NULL ||
        tensor_manifest->records == NULL ||
        split->indices == NULL) {

        return GS_BASELINE_MINI75_PIPELINE_ERROR_NULL_ARGUMENT;
    }

    if (!gs_pipeline_spec_is_mini75(&tensor_manifest->spec) ||
        strcmp(tensor_manifest->extra_column_name, "origin") != 0) {

        return GS_BASELINE_MINI75_PIPELINE_ERROR_ALIGNMENT;
    }

    if (first_row > split->count) {
        return GS_BASELINE_MINI75_PIPELINE_ERROR_RANGE;
    }

    if (row_count > split->count - first_row) {
        return GS_BASELINE_MINI75_PIPELINE_ERROR_RANGE;
    }

    if (tensor_buffer_count != GS_BASELINE_MINI75_INPUT_COUNT) {
        return GS_BASELINE_MINI75_PIPELINE_ERROR_BUFFER_SIZE;
    }

    if (row_count > SIZE_MAX / GS_BASELINE_MINI75_FEATURE_COUNT) {
        return GS_BASELINE_MINI75_PIPELINE_ERROR_OVERFLOW;
    }

    required_features =
        row_count * GS_BASELINE_MINI75_FEATURE_COUNT;

    if (required_features > raw_features_count) {
        return GS_BASELINE_MINI75_PIPELINE_ERROR_BUFFER_SIZE;
    }

    for (local_row = 0; local_row < row_count; local_row++) {
        size_t split_pos = first_row + local_row;
        size_t manifest_index = split->indices[split_pos];
        const GSEventRecord *event_record;
        const GSTensorManifestRecord *tensor_record;
        float *destination;
        char joined_path[GS_BASELINE_MINI75_JOINED_PATH_LEN];
        size_t value_index;
        int rc;

        if (manifest_index >= event_manifest->count ||
            manifest_index >= tensor_manifest->count) {

            return GS_BASELINE_MINI75_PIPELINE_ERROR_RANGE;
        }

        event_record =
            gs_event_manifest_get(event_manifest, manifest_index);

        tensor_record =
            gs_tensor_manifest_get(tensor_manifest, manifest_index);

        if (event_record == NULL || tensor_record == NULL ||
            event_record->event_index != tensor_record->event_index ||
            strcmp(event_record->event_id, tensor_record->event_id) != 0 ||
            !gs_pipeline_tensor_record_is_mini75(tensor_record)) {

            return GS_BASELINE_MINI75_PIPELINE_ERROR_ALIGNMENT;
        }

        rc = gs_pipeline_join_path(
            data_root,
            tensor_record->tensor_path,
            joined_path,
            sizeof(joined_path)
        );

        if (rc != GS_BASELINE_MINI75_PIPELINE_SUCCESS) {
            return rc;
        }

        if (gs_tensor_read_float01(
                joined_path,
                &tensor_manifest->spec,
                tensor_buffer
            ) != 0) {

            return GS_BASELINE_MINI75_PIPELINE_ERROR_TENSOR_READ;
        }

        for (value_index = 0;
             value_index < GS_BASELINE_MINI75_INPUT_COUNT;
             value_index++) {

            if (!isfinite(tensor_buffer[value_index]) ||
                tensor_buffer[value_index] < 0.0f ||
                tensor_buffer[value_index] > 1.0f) {

                return GS_BASELINE_MINI75_PIPELINE_ERROR_TENSOR_VALUE;
            }
        }

        destination =
            raw_features +
            local_row * GS_BASELINE_MINI75_FEATURE_COUNT;

        if (gs_baseline_mini75_pool5_extract(
                tensor_buffer,
                tensor_buffer_count,
                destination,
                GS_BASELINE_MINI75_FEATURE_COUNT
            ) != GS_BASELINE_MINI75_SUCCESS) {

            return GS_BASELINE_MINI75_PIPELINE_ERROR_FEATURE_EXTRACTION;
        }

        for (value_index = 0;
             value_index < GS_BASELINE_MINI75_FEATURE_COUNT;
             value_index++) {

            if (!isfinite(destination[value_index]) ||
                destination[value_index] < 0.0f ||
                destination[value_index] > 1.0f) {

                return GS_BASELINE_MINI75_PIPELINE_ERROR_TENSOR_VALUE;
            }
        }
    }

    return GS_BASELINE_MINI75_PIPELINE_SUCCESS;
}
