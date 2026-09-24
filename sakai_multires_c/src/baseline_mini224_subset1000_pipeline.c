#include "baseline_mini224_subset1000_pipeline.h"

#include "baseline_average_pool_features.h"
#include "vae_event_dataset.h"
#include "vae_saved_split.h"
#include "vae_tensor_io.h"
#include "vae_tensor_manifest.h"
#include "vae_tensor_spec.h"

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
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
    GSSavedSplit official_train_split;
    GSSavedSplit official_validation_split;
    GSTensorSpec tensor_spec;
    char data_root[GS_BASELINE_MINI224_SUBSET1000_PATH_LENGTH];
    float *tensor_buffer;
} GSBaselineMini224Subset1000PipelineImplementation;


static int gs_mini224_subset1000_checked_product(
    size_t left,
    size_t right,
    size_t *product
)
{
    if (product == NULL) {
        return GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_NULL_ARGUMENT;
    }

    if (left != 0 && right > SIZE_MAX / left) {
        return GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_OVERFLOW;
    }

    *product = left * right;
    return GS_BASELINE_MINI224_SUBSET1000_PIPELINE_SUCCESS;
}


static int gs_mini224_subset1000_join_path(
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

        return GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_NULL_ARGUMENT;
    }

    root_length = strlen(data_root);
    relative_length = strlen(relative_path);

    needs_separator =
        root_length != 0 &&
        data_root[root_length - 1] != '/' &&
        data_root[root_length - 1] != '\\';

    if (root_length > SIZE_MAX - relative_length) {
        return GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_OVERFLOW;
    }

    required = root_length + relative_length;

    if (needs_separator) {
        if (required == SIZE_MAX) {
            return GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_OVERFLOW;
        }
        required++;
    }

    if (required >= joined_path_size) {
        return GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_PATH;
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
    return GS_BASELINE_MINI224_SUBSET1000_PIPELINE_SUCCESS;
}


static int gs_mini224_subset1000_sibling_path(
    const char *path,
    const char *sibling_name,
    char *sibling_path,
    size_t sibling_path_size
)
{
    const char *slash;
    const char *backslash;
    const char *separator;
    size_t directory_length;
    size_t name_length;

    if (path == NULL || sibling_name == NULL || sibling_path == NULL ||
        sibling_path_size == 0 || sibling_name[0] == '\0') {

        return GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_NULL_ARGUMENT;
    }

    slash = strrchr(path, '/');
    backslash = strrchr(path, '\\');
    separator = slash;
    if (separator == NULL ||
        (backslash != NULL && backslash > separator)) {

        separator = backslash;
    }

    directory_length =
        separator == NULL ? 0u : (size_t)(separator - path + 1);
    name_length = strlen(sibling_name);

    if (directory_length > SIZE_MAX - name_length ||
        directory_length + name_length >= sibling_path_size) {

        return GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_PATH;
    }

    if (directory_length != 0u) {
        memcpy(sibling_path, path, directory_length);
    }
    memcpy(sibling_path + directory_length, sibling_name, name_length + 1u);
    return GS_BASELINE_MINI224_SUBSET1000_PIPELINE_SUCCESS;
}


static void gs_mini224_subset1000_remove_line_ending(char *line)
{
    size_t length;

    if (line == NULL) {
        return;
    }

    length = strlen(line);
    while (length != 0u &&
           (line[length - 1u] == '\r' || line[length - 1u] == '\n')) {

        line[--length] = '\0';
    }
}


static int gs_mini224_subset1000_csv_line(
    char *line,
    char **fields,
    size_t field_capacity
)
{
    char *read;
    char *write;
    size_t count = 0u;

    if (line == NULL || fields == NULL || field_capacity == 0u) {
        return -1;
    }

    read = line;
    while (*read != '\0') {
        int quoted = 0;
        int has_separator;

        if (count >= field_capacity) {
            return -1;
        }

        write = read;
        fields[count++] = write;

        if (*read == '"') {
            quoted = 1;
            read++;
            fields[count - 1u] = write;
        }

        while (*read != '\0') {
            if (quoted && *read == '"') {
                if (read[1] == '"') {
                    *write++ = '"';
                    read += 2;
                    continue;
                }

                read++;
                quoted = 0;
                if (*read != '\0' && *read != ',') {
                    return -1;
                }
                break;
            }

            if (!quoted && *read == ',') {
                break;
            }

            *write++ = *read++;
        }

        if (quoted) {
            return -1;
        }

        has_separator = *read == ',';
        *write = '\0';
        if (has_separator) {
            read++;
            if (*read == '\0') {
                if (count >= field_capacity) {
                    return -1;
                }
                fields[count++] = read;
            }
        }
    }

    return (int)count;
}


static int gs_mini224_subset1000_parse_size(
    const char *text,
    size_t *value
)
{
    char *end = NULL;
    uintmax_t parsed;

    if (text == NULL || value == NULL || text[0] == '\0' ||
        text[0] == '-') {

        return 1;
    }

    errno = 0;
    parsed = strtoumax(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' ||
        parsed > (uintmax_t)SIZE_MAX) {

        return 1;
    }

    *value = (size_t)parsed;
    return 0;
}


static int gs_mini224_subset1000_validate_subset_csv(
    const char *csv_path,
    const char *split_name,
    const GSSavedSplit *split,
    const GSEventManifest *events,
    const GSTensorManifest *tensors
)
{
    static const char expected_header[] =
        "split_name,split_pos,manifest_index,event_index,event_id,tensor_path";
    FILE *file = NULL;
    char line[4096];
    size_t row = 0u;
    int failed = 0;

    if (csv_path == NULL || split_name == NULL || split == NULL ||
        events == NULL || tensors == NULL || split->indices == NULL) {

        return 1;
    }

    file = fopen(csv_path, "rb");
    if (file == NULL || fgets(line, sizeof(line), file) == NULL) {
        failed = 1;
        goto cleanup;
    }

    gs_mini224_subset1000_remove_line_ending(line);
    if (strcmp(line, expected_header) != 0) {
        failed = 1;
        goto cleanup;
    }

    while (!failed && fgets(line, sizeof(line), file) != NULL) {
        char *fields[7];
        size_t split_pos;
        size_t manifest_index;
        size_t event_index;
        const GSEventRecord *event_record;
        const GSTensorManifestRecord *tensor_record;
        int field_count;

        if (strchr(line, '\n') == NULL && !feof(file)) {
            failed = 1;
            break;
        }

        gs_mini224_subset1000_remove_line_ending(line);
        if (line[0] == '\0') {
            failed = 1;
            break;
        }

        field_count = gs_mini224_subset1000_csv_line(
            line,
            fields,
            sizeof(fields) / sizeof(fields[0])
        );

        if (field_count != 6 || row >= split->count ||
            strcmp(fields[0], split_name) != 0 ||
            gs_mini224_subset1000_parse_size(fields[1], &split_pos) != 0 ||
            gs_mini224_subset1000_parse_size(
                fields[2],
                &manifest_index
            ) != 0 ||
            gs_mini224_subset1000_parse_size(fields[3], &event_index) != 0 ||
            split_pos != row ||
            manifest_index != split->indices[row] ||
            manifest_index >= events->count ||
            manifest_index >= tensors->count) {

            failed = 1;
            break;
        }

        event_record = gs_event_manifest_get(events, manifest_index);
        tensor_record = gs_tensor_manifest_get(tensors, manifest_index);

        if (event_record == NULL || tensor_record == NULL ||
            event_index != (size_t)event_record->event_index ||
            event_record->event_index != tensor_record->event_index ||
            strcmp(fields[4], event_record->event_id) != 0 ||
            strcmp(fields[4], tensor_record->event_id) != 0 ||
            strcmp(fields[5], tensor_record->tensor_path) != 0) {

            failed = 1;
            break;
        }

        row++;
    }

    if (ferror(file) || row != split->count) {
        failed = 1;
    }

cleanup:
    if (file != NULL && fclose(file) != 0) {
        failed = 1;
    }
    return failed;
}


static int gs_mini224_subset1000_spec_is_valid(const GSTensorSpec *spec)
{
    return spec != NULL &&
           gs_tensor_spec_is_valid(spec) &&
           spec->channels == GS_BASELINE_MINI224_SUBSET1000_CHANNELS &&
           spec->height == GS_BASELINE_MINI224_SUBSET1000_SIDE &&
           spec->width == GS_BASELINE_MINI224_SUBSET1000_SIDE &&
           spec->pixels_per_channel ==
               (size_t)GS_BASELINE_MINI224_SUBSET1000_SIDE *
               (size_t)GS_BASELINE_MINI224_SUBSET1000_SIDE &&
           spec->values_per_event == GS_BASELINE_MINI224_SUBSET1000_INPUT_COUNT &&
           spec->bytes_u8 == GS_BASELINE_MINI224_SUBSET1000_TENSOR_BYTES &&
           spec->bytes_f32 ==
               GS_BASELINE_MINI224_SUBSET1000_INPUT_COUNT * sizeof(float);
}


static int gs_mini224_subset1000_tensor_record_is_valid(
    const GSTensorManifestRecord *record
)
{
    return record != NULL &&
           record->event_id[0] != '\0' &&
           record->tensor_path[0] != '\0' &&
           record->channels == GS_BASELINE_MINI224_SUBSET1000_CHANNELS &&
           record->height == GS_BASELINE_MINI224_SUBSET1000_SIDE &&
           record->width == GS_BASELINE_MINI224_SUBSET1000_SIDE &&
           strcmp(record->dtype, "uint8") == 0 &&
           strcmp(record->layout, "channel_major") == 0 &&
           strcmp(record->channel_order, "0.5;1.0;2.0;4.0") == 0 &&
           strcmp(record->extra_value, "24") == 0;
}


static int gs_mini224_subset1000_compare_paths(
    const void *left,
    const void *right
)
{
    const char *const *left_path = (const char *const *)left;
    const char *const *right_path = (const char *const *)right;

    return strcmp(*left_path, *right_path);
}


static void gs_mini224_subset1000_implementation_free(
    GSBaselineMini224Subset1000PipelineImplementation *implementation
)
{
    if (implementation == NULL) {
        return;
    }

    free(implementation->tensor_buffer);
    implementation->tensor_buffer = NULL;

    gs_saved_split_free(&implementation->official_validation_split);
    gs_saved_split_free(&implementation->official_train_split);
    gs_saved_split_free(&implementation->validation_split);
    gs_saved_split_free(&implementation->train_split);
    gs_tensor_manifest_free(&implementation->tensor_manifest);
    gs_event_manifest_free(&implementation->event_manifest);

    free(implementation);
}


static int gs_mini224_subset1000_validate_global(
    const GSBaselineMini224Subset1000PipelineImplementation *implementation
)
{
    const char **path_references = NULL;
    unsigned char *owner = NULL;
    size_t pointer_bytes;
    size_t index;
    size_t expected_input_count;
    size_t expected_pooled_side;
    size_t expected_feature_count;
    int result = GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_ALIGNMENT;

    if (implementation == NULL ||
        implementation->event_manifest.records == NULL ||
        implementation->tensor_manifest.records == NULL ||
        implementation->train_split.indices == NULL ||
        implementation->validation_split.indices == NULL ||
        implementation->official_train_split.indices == NULL ||
        implementation->official_validation_split.indices == NULL) {

        return GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_STATE;
    }

    if (implementation->event_manifest.count !=
            GS_BASELINE_MINI224_SUBSET1000_OFFICIAL_EVENT_COUNT ||
        implementation->tensor_manifest.count !=
            GS_BASELINE_MINI224_SUBSET1000_OFFICIAL_EVENT_COUNT ||
        implementation->train_split.count !=
            GS_BASELINE_MINI224_SUBSET1000_OFFICIAL_TRAIN_COUNT ||
        implementation->validation_split.count !=
            GS_BASELINE_MINI224_SUBSET1000_OFFICIAL_VALIDATION_COUNT ||
        implementation->official_train_split.count !=
            GS_BASELINE_MINI224_SUBSET1000_FULL_TRAIN_COUNT ||
        implementation->official_validation_split.count !=
            GS_BASELINE_MINI224_SUBSET1000_FULL_VALIDATION_COUNT) {

        return GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_COUNTS;
    }

    if (strcmp(implementation->train_split.name, "train") != 0 ||
        strcmp(
            implementation->validation_split.name,
            "validation"
        ) != 0 ||
        strcmp(
            implementation->tensor_manifest.extra_column_name,
            "crop_left"
        ) != 0 ||
        !gs_mini224_subset1000_spec_is_valid(&implementation->tensor_spec) ||
        !gs_mini224_subset1000_spec_is_valid(
            &implementation->tensor_manifest.spec
        )) {

        return GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_SPEC;
    }

    if (gs_baseline_average_pool_dimensions(
            GS_BASELINE_MINI224_SUBSET1000_CHANNELS,
            GS_BASELINE_MINI224_SUBSET1000_SIDE,
            GS_BASELINE_MINI224_SUBSET1000_POOL_SIDE,
            &expected_input_count,
            &expected_pooled_side,
            &expected_feature_count
        ) != GS_BASELINE_AVERAGE_POOL_SUCCESS ||
        expected_input_count != GS_BASELINE_MINI224_SUBSET1000_INPUT_COUNT ||
        expected_pooled_side != GS_BASELINE_MINI224_SUBSET1000_POOLED_SIDE ||
        expected_feature_count != GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT) {

        return GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_SPEC;
    }

    if (gs_mini224_subset1000_checked_product(
            implementation->tensor_manifest.count,
            sizeof(*path_references),
            &pointer_bytes
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_SUCCESS) {

        return GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_OVERFLOW;
    }

    path_references = (const char **)malloc(pointer_bytes);
    owner = (unsigned char *)calloc(
        GS_BASELINE_MINI224_SUBSET1000_OFFICIAL_EVENT_COUNT,
        sizeof(unsigned char)
    );

    if (path_references == NULL || owner == NULL) {
        result = GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_ALLOCATION;
        goto cleanup;
    }

    for (index = 0;
         index < GS_BASELINE_MINI224_SUBSET1000_OFFICIAL_EVENT_COUNT;
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

        char joined_path[GS_BASELINE_MINI224_SUBSET1000_PATH_LENGTH];
        struct stat file_information;
        int join_result;

        if (event_record == NULL || tensor_record == NULL ||
            event_record->event_id[0] == '\0' ||
            (size_t)event_record->event_index != index + 1 ||
            (size_t)tensor_record->event_index != index + 1 ||
            event_record->event_index != tensor_record->event_index ||
            strcmp(event_record->event_id, tensor_record->event_id) != 0 ||
            !gs_mini224_subset1000_tensor_record_is_valid(tensor_record)) {

            result = GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_ALIGNMENT;
            goto cleanup;
        }

        path_references[index] = tensor_record->tensor_path;

        join_result = gs_mini224_subset1000_join_path(
            implementation->data_root,
            tensor_record->tensor_path,
            joined_path,
            sizeof(joined_path)
        );

        if (join_result != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_SUCCESS) {
            result = join_result;
            goto cleanup;
        }

        if (stat(joined_path, &file_information) != 0) {
            result = GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_FILE_NOT_FOUND;
            goto cleanup;
        }

        if (file_information.st_size < 0 ||
            (uintmax_t)file_information.st_size !=
                (uintmax_t)GS_BASELINE_MINI224_SUBSET1000_TENSOR_BYTES) {

            result = GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_FILE_SIZE;
            goto cleanup;
        }
    }

    qsort(
        path_references,
        implementation->tensor_manifest.count,
        sizeof(*path_references),
        gs_mini224_subset1000_compare_paths
    );

    for (index = 1;
         index < implementation->tensor_manifest.count;
         index++) {

        if (strcmp(
                path_references[index - 1],
                path_references[index]
            ) == 0) {

            result = GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_DUPLICATE_PATH;
            goto cleanup;
        }
    }

    for (index = 0;
         index < implementation->train_split.count;
         index++) {

        size_t manifest_index =
            implementation->train_split.indices[index];

        if (manifest_index >=
                GS_BASELINE_MINI224_SUBSET1000_OFFICIAL_EVENT_COUNT) {

            result = GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_RANGE;
            goto cleanup;
        }

        if (owner[manifest_index] != 0u) {
            result = GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_ALIGNMENT;
            goto cleanup;
        }

        if (manifest_index !=
                implementation->official_train_split.indices[index]) {

            result = GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_ALIGNMENT;
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
                GS_BASELINE_MINI224_SUBSET1000_OFFICIAL_EVENT_COUNT) {

            result = GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_RANGE;
            goto cleanup;
        }

        if (owner[manifest_index] != 0u) {
            result = GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_ALIGNMENT;
            goto cleanup;
        }

        if (manifest_index !=
                implementation->official_validation_split.indices[index]) {

            result = GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_ALIGNMENT;
            goto cleanup;
        }

        owner[manifest_index] = 2u;
    }

    result = GS_BASELINE_MINI224_SUBSET1000_PIPELINE_SUCCESS;

cleanup:
    free(owner);
    free(path_references);
    return result;
}


void gs_baseline_mini224_subset1000_pipeline_init(
    GSBaselineMini224Subset1000Pipeline *pipeline
)
{
    if (pipeline == NULL) {
        return;
    }

    pipeline->implementation = NULL;
}


int gs_baseline_mini224_subset1000_pipeline_open(
    GSBaselineMini224Subset1000Pipeline *pipeline,
    const char *event_manifest_path,
    const char *tensor_manifest_path,
    const char *train_subset_path,
    const char *validation_subset_path,
    const char *data_root
)
{
    GSBaselineMini224Subset1000PipelineImplementation *implementation;
    char official_train_path[GS_BASELINE_MINI224_SUBSET1000_PATH_LENGTH];
    char official_validation_path[
        GS_BASELINE_MINI224_SUBSET1000_PATH_LENGTH
    ];
    size_t root_length;
    size_t tensor_buffer_bytes;
    int result;

    if (pipeline == NULL || event_manifest_path == NULL ||
        tensor_manifest_path == NULL || train_subset_path == NULL ||
        validation_subset_path == NULL || data_root == NULL) {

        return GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_NULL_ARGUMENT;
    }

    if (pipeline->implementation != NULL) {
        return GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_STATE;
    }

    root_length = strlen(data_root);
    if (root_length >= GS_BASELINE_MINI224_SUBSET1000_PATH_LENGTH) {
        return GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_PATH;
    }

    implementation =
        (GSBaselineMini224Subset1000PipelineImplementation *)calloc(
            1,
            sizeof(*implementation)
        );

    if (implementation == NULL) {
        return GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_ALLOCATION;
    }

    memcpy(
        implementation->data_root,
        data_root,
        root_length + 1
    );

    if (gs_tensor_spec_init(
            &implementation->tensor_spec,
            GS_BASELINE_MINI224_SUBSET1000_SIDE
        ) != 0) {

        result = GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_SPEC;
        goto fail;
    }

    if (gs_event_manifest_load(
            event_manifest_path,
            &implementation->event_manifest
        ) != 0) {

        result = GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_EVENT_MANIFEST;
        goto fail;
    }

    if (gs_mini224_subset1000_sibling_path(
            event_manifest_path,
            "split_train_indices_seed123.csv",
            official_train_path,
            sizeof(official_train_path)
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_SUCCESS ||
        gs_mini224_subset1000_sibling_path(
            event_manifest_path,
            "split_val_indices_seed123.csv",
            official_validation_path,
            sizeof(official_validation_path)
        ) != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_SUCCESS ||
        gs_saved_split_load(
            official_train_path,
            "train",
            &implementation->event_manifest,
            &implementation->official_train_split
        ) != 0 ||
        gs_saved_split_load(
            official_validation_path,
            "validation",
            &implementation->event_manifest,
            &implementation->official_validation_split
        ) != 0) {

        result = GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_ALIGNMENT;
        goto fail;
    }

    if (gs_tensor_manifest_load(
            tensor_manifest_path,
            &implementation->tensor_spec,
            &implementation->tensor_manifest
        ) != 0) {

        result = GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_TENSOR_MANIFEST;
        goto fail;
    }

    if (gs_saved_split_load(
            train_subset_path,
            "train",
            &implementation->event_manifest,
            &implementation->train_split
        ) != 0) {

        result = GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_TRAIN_SPLIT;
        goto fail;
    }

    if (gs_saved_split_load(
            validation_subset_path,
            "validation",
            &implementation->event_manifest,
            &implementation->validation_split
        ) != 0) {

        result = GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_VALIDATION_SPLIT;
        goto fail;
    }

    result = gs_mini224_subset1000_validate_global(implementation);
    if (result != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_SUCCESS) {
        goto fail;
    }

    if (gs_mini224_subset1000_validate_subset_csv(
            train_subset_path,
            "train",
            &implementation->train_split,
            &implementation->event_manifest,
            &implementation->tensor_manifest
        ) != 0 ||
        gs_mini224_subset1000_validate_subset_csv(
            validation_subset_path,
            "validation",
            &implementation->validation_split,
            &implementation->event_manifest,
            &implementation->tensor_manifest
        ) != 0) {

        result = GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_ALIGNMENT;
        goto fail;
    }

    result = gs_mini224_subset1000_checked_product(
        GS_BASELINE_MINI224_SUBSET1000_INPUT_COUNT,
        sizeof(float),
        &tensor_buffer_bytes
    );

    if (result != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_SUCCESS) {
        goto fail;
    }

    implementation->tensor_buffer =
        (float *)malloc(tensor_buffer_bytes);

    if (implementation->tensor_buffer == NULL) {
        result = GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_ALLOCATION;
        goto fail;
    }

    pipeline->implementation = implementation;
    return GS_BASELINE_MINI224_SUBSET1000_PIPELINE_SUCCESS;

fail:
    gs_mini224_subset1000_implementation_free(implementation);
    return result;
}


static const GSBaselineMini224Subset1000PipelineImplementation *
gs_mini224_subset1000_const_implementation(
    const GSBaselineMini224Subset1000Pipeline *pipeline
)
{
    if (pipeline == NULL || pipeline->implementation == NULL) {
        return NULL;
    }

    return (const GSBaselineMini224Subset1000PipelineImplementation *)
        pipeline->implementation;
}


int gs_baseline_mini224_subset1000_pipeline_counts(
    const GSBaselineMini224Subset1000Pipeline *pipeline,
    size_t *event_count,
    size_t *train_count,
    size_t *validation_count
)
{
    const GSBaselineMini224Subset1000PipelineImplementation *implementation =
        gs_mini224_subset1000_const_implementation(pipeline);

    if (event_count == NULL || train_count == NULL ||
        validation_count == NULL) {

        return GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_NULL_ARGUMENT;
    }

    if (implementation == NULL) {
        return GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_STATE;
    }

    *event_count = implementation->event_manifest.count;
    *train_count = implementation->train_split.count;
    *validation_count = implementation->validation_split.count;

    return GS_BASELINE_MINI224_SUBSET1000_PIPELINE_SUCCESS;
}


static int gs_mini224_subset1000_manifest_index(
    const GSBaselineMini224Subset1000Pipeline *pipeline,
    const GSSavedSplit *split,
    size_t split_pos,
    size_t *manifest_index
)
{
    if (pipeline == NULL || manifest_index == NULL) {
        return GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_NULL_ARGUMENT;
    }

    if (pipeline->implementation == NULL || split == NULL ||
        split->indices == NULL) {

        return GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_STATE;
    }

    if (split_pos >= split->count) {
        return GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_RANGE;
    }

    *manifest_index = split->indices[split_pos];
    return GS_BASELINE_MINI224_SUBSET1000_PIPELINE_SUCCESS;
}


int gs_baseline_mini224_subset1000_train_manifest_index(
    const GSBaselineMini224Subset1000Pipeline *pipeline,
    size_t split_pos,
    size_t *manifest_index
)
{
    const GSBaselineMini224Subset1000PipelineImplementation *implementation =
        gs_mini224_subset1000_const_implementation(pipeline);

    return gs_mini224_subset1000_manifest_index(
        pipeline,
        implementation == NULL ? NULL : &implementation->train_split,
        split_pos,
        manifest_index
    );
}


int gs_baseline_mini224_subset1000_validation_manifest_index(
    const GSBaselineMini224Subset1000Pipeline *pipeline,
    size_t split_pos,
    size_t *manifest_index
)
{
    const GSBaselineMini224Subset1000PipelineImplementation *implementation =
        gs_mini224_subset1000_const_implementation(pipeline);

    return gs_mini224_subset1000_manifest_index(
        pipeline,
        implementation == NULL ?
            NULL :
            &implementation->validation_split,
        split_pos,
        manifest_index
    );
}


static int gs_mini224_subset1000_extract_rows(
    GSBaselineMini224Subset1000Pipeline *pipeline,
    const GSSavedSplit *split,
    size_t first_row,
    size_t row_count,
    float *features,
    size_t feature_capacity
)
{
    GSBaselineMini224Subset1000PipelineImplementation *implementation;
    size_t required_features;
    size_t local_row;
    int product_result;

    if (pipeline == NULL || features == NULL) {
        return GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_NULL_ARGUMENT;
    }

    implementation =
        (GSBaselineMini224Subset1000PipelineImplementation *)
            pipeline->implementation;

    if (implementation == NULL || split == NULL ||
        split->indices == NULL ||
        implementation->tensor_buffer == NULL) {

        return GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_STATE;
    }

    if (row_count == 0) {
        return GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_RANGE;
    }

    product_result = gs_mini224_subset1000_checked_product(
        row_count,
        GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT,
        &required_features
    );

    if (product_result != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_SUCCESS) {
        return product_result;
    }

    if (first_row >= split->count ||
        row_count > split->count - first_row) {

        return GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_RANGE;
    }

    if (feature_capacity < required_features) {
        return GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_BUFFER_SIZE;
    }

    for (local_row = 0; local_row < row_count; local_row++) {
        size_t split_pos = first_row + local_row;
        size_t manifest_index = split->indices[split_pos];
        const GSEventRecord *event_record;
        const GSTensorManifestRecord *tensor_record;
        float *destination;
        char joined_path[GS_BASELINE_MINI224_SUBSET1000_PATH_LENGTH];
        size_t value_index;
        int join_result;

        if (manifest_index >= implementation->event_manifest.count ||
            manifest_index >= implementation->tensor_manifest.count) {

            return GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_RANGE;
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
            !gs_mini224_subset1000_tensor_record_is_valid(tensor_record)) {

            return GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_ALIGNMENT;
        }

        join_result = gs_mini224_subset1000_join_path(
            implementation->data_root,
            tensor_record->tensor_path,
            joined_path,
            sizeof(joined_path)
        );

        if (join_result != GS_BASELINE_MINI224_SUBSET1000_PIPELINE_SUCCESS) {
            return join_result;
        }

        if (gs_tensor_read_float01(
                joined_path,
                &implementation->tensor_spec,
                implementation->tensor_buffer
            ) != 0) {

            return GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_TENSOR_READ;
        }

        for (value_index = 0;
             value_index < GS_BASELINE_MINI224_SUBSET1000_INPUT_COUNT;
             value_index++) {

            if (!isfinite(implementation->tensor_buffer[value_index]) ||
                implementation->tensor_buffer[value_index] < 0.0f ||
                implementation->tensor_buffer[value_index] > 1.0f) {

                return GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_TENSOR_VALUE;
            }
        }

        destination =
            features +
            local_row * GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT;

        if (gs_baseline_average_pool_extract(
                implementation->tensor_buffer,
                GS_BASELINE_MINI224_SUBSET1000_INPUT_COUNT,
                GS_BASELINE_MINI224_SUBSET1000_CHANNELS,
                GS_BASELINE_MINI224_SUBSET1000_SIDE,
                GS_BASELINE_MINI224_SUBSET1000_POOL_SIDE,
                destination,
                GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT
            ) != GS_BASELINE_AVERAGE_POOL_SUCCESS) {

            return
                GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_FEATURE_EXTRACTION;
        }

        for (value_index = 0;
             value_index < GS_BASELINE_MINI224_SUBSET1000_FEATURE_COUNT;
             value_index++) {

            if (!isfinite(destination[value_index]) ||
                destination[value_index] < 0.0f ||
                destination[value_index] > 1.0f) {

                return GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_TENSOR_VALUE;
            }
        }
    }

    return GS_BASELINE_MINI224_SUBSET1000_PIPELINE_SUCCESS;
}


int gs_baseline_mini224_subset1000_extract_train_rows(
    GSBaselineMini224Subset1000Pipeline *pipeline,
    size_t first_row,
    size_t row_count,
    float *features,
    size_t feature_capacity
)
{
    GSBaselineMini224Subset1000PipelineImplementation *implementation;

    if (pipeline == NULL) {
        return GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_NULL_ARGUMENT;
    }

    implementation =
        (GSBaselineMini224Subset1000PipelineImplementation *)
            pipeline->implementation;

    return gs_mini224_subset1000_extract_rows(
        pipeline,
        implementation == NULL ? NULL : &implementation->train_split,
        first_row,
        row_count,
        features,
        feature_capacity
    );
}


int gs_baseline_mini224_subset1000_extract_validation_rows(
    GSBaselineMini224Subset1000Pipeline *pipeline,
    size_t first_row,
    size_t row_count,
    float *features,
    size_t feature_capacity
)
{
    GSBaselineMini224Subset1000PipelineImplementation *implementation;

    if (pipeline == NULL) {
        return GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_NULL_ARGUMENT;
    }

    implementation =
        (GSBaselineMini224Subset1000PipelineImplementation *)
            pipeline->implementation;

    return gs_mini224_subset1000_extract_rows(
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


const char *gs_baseline_mini224_subset1000_pipeline_error_string(int error_code)
{
    switch (error_code) {
        case GS_BASELINE_MINI224_SUBSET1000_PIPELINE_SUCCESS:
            return "success";
        case GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_NULL_ARGUMENT:
            return "null argument";
        case GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_STATE:
            return "pipeline state is not ready";
        case GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_ALLOCATION:
            return "memory allocation failed";
        case GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_SPEC:
            return "mini224_subset1000 tensor or pooling specification is invalid";
        case GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_EVENT_MANIFEST:
            return "event manifest load failed";
        case GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_TENSOR_MANIFEST:
            return "mini224_subset1000 tensor manifest load failed";
        case GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_TRAIN_SPLIT:
            return "train split load failed";
        case GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_VALIDATION_SPLIT:
            return "validation split load failed";
        case GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_COUNTS:
            return "official counts are invalid";
        case GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_ALIGNMENT:
            return "manifest or split alignment is invalid";
        case GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_DUPLICATE_PATH:
            return "mini224_subset1000 tensor path is duplicated";
        case GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_PATH:
            return "resolved path is too long";
        case GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_FILE_NOT_FOUND:
            return "mini224_subset1000 tensor file is missing";
        case GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_FILE_SIZE:
            return "mini224_subset1000 tensor file size is invalid";
        case GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_RANGE:
            return "requested split interval is invalid";
        case GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_OVERFLOW:
            return "size calculation overflow";
        case GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_BUFFER_SIZE:
            return "buffer capacity is insufficient";
        case GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_TENSOR_READ:
            return "mini224_subset1000 tensor read failed";
        case GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_TENSOR_VALUE:
            return "tensor or feature value is invalid";
        case GS_BASELINE_MINI224_SUBSET1000_PIPELINE_ERROR_FEATURE_EXTRACTION:
            return "average pooling feature extraction failed";
        default:
            return "unknown mini224_subset1000 pipeline error";
    }
}


void gs_baseline_mini224_subset1000_pipeline_free(
    GSBaselineMini224Subset1000Pipeline *pipeline
)
{
    if (pipeline == NULL) {
        return;
    }

    gs_mini224_subset1000_implementation_free(
        (GSBaselineMini224Subset1000PipelineImplementation *)
            pipeline->implementation
    );

    pipeline->implementation = NULL;
}
