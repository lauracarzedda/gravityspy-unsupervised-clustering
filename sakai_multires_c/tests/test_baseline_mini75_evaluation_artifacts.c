#include "baseline_clustering_metrics.h"
#include "vae_event_dataset.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


enum {
    GS_TEST_LABEL_COUNT = 22,
    GS_TEST_TRAIN_ROWS = 6866,
    GS_TEST_VALIDATION_ROWS = 1717,
    GS_TEST_MANIFEST_ROWS = 8583,
    GS_TEST_PATH_LEN = 4096,
    GS_TEST_LINE_LEN = 4096,
    GS_TEST_MAX_FIELDS = 64
};


#define GS_TEST_CODE_COMMIT "e701884ca8e006800483aff7b84318b3c7d2a7b0"


static const char *const gs_test_labels[GS_TEST_LABEL_COUNT] = {
    "1080Lines",
    "1400Ripples",
    "Air_Compressor",
    "Blip",
    "Chirp",
    "Extremely_Loud",
    "Helix",
    "Koi_Fish",
    "Light_Modulation",
    "Low_Frequency_Burst",
    "Low_Frequency_Lines",
    "No_Glitch",
    "None_of_the_Above",
    "Paired_Doves",
    "Power_Line",
    "Repeating_Blips",
    "Scattered_Light",
    "Scratchy",
    "Tomte",
    "Violin_Mode",
    "Wandering_Line",
    "Whistle"
};


typedef struct {
    size_t total;
    size_t occupied;
    size_t dominant;
    uint64_t dominant_count;
    double dominant_fraction;
    size_t second;
    uint64_t second_count;
} GSTestLabelStatistics;


static int gs_test_checked_product(
    size_t left,
    size_t right,
    size_t *product
)
{
    if (product == NULL || (right != 0 && left > SIZE_MAX / right)) {
        return 1;
    }
    *product = left * right;
    return 0;
}


static int gs_test_platform_is_valid(void)
{
    uint16_t value = UINT16_C(1);
    const unsigned char *bytes = (const unsigned char *)&value;

    return sizeof(uint32_t) == 4 && sizeof(uint64_t) == 8 &&
           bytes[0] == 1u;
}


static int gs_test_label_index(const char *label)
{
    size_t index;

    for (index = 0; index < GS_TEST_LABEL_COUNT; index++) {
        if (strcmp(label, gs_test_labels[index]) == 0) {
            return (int)index;
        }
    }
    return -1;
}


static void gs_test_strip_line_ending(char *line)
{
    size_t length = strlen(line);

    while (length != 0 &&
           (line[length - 1] == '\r' || line[length - 1] == '\n')) {
        line[--length] = '\0';
    }
}


static int gs_test_split_csv(
    char *line,
    char **fields,
    size_t expected_fields
)
{
    size_t count = 0;
    char *cursor;

    if (line == NULL || fields == NULL || expected_fields == 0 ||
        expected_fields > GS_TEST_MAX_FIELDS) {
        return 1;
    }
    gs_test_strip_line_ending(line);
    fields[count++] = line;
    for (cursor = line; *cursor != '\0'; cursor++) {
        if (*cursor == ',') {
            if (count >= expected_fields) {
                return 1;
            }
            *cursor = '\0';
            fields[count++] = cursor + 1;
        }
    }
    return count == expected_fields ? 0 : 1;
}


static int gs_test_parse_size(const char *text, size_t *value)
{
    char *endptr = NULL;
    unsigned long long parsed;

    if (text == NULL || value == NULL || text[0] == '\0' ||
        text[0] == '-') {
        return 1;
    }
    errno = 0;
    parsed = strtoull(text, &endptr, 10);
    if (errno == ERANGE || endptr == text || *endptr != '\0' ||
        parsed > (unsigned long long)SIZE_MAX) {
        return 1;
    }
    *value = (size_t)parsed;
    return 0;
}


static int gs_test_parse_u64(const char *text, uint64_t *value)
{
    char *endptr = NULL;
    unsigned long long parsed;

    if (text == NULL || value == NULL || text[0] == '\0' ||
        text[0] == '-') {
        return 1;
    }
    errno = 0;
    parsed = strtoull(text, &endptr, 10);
    if (errno == ERANGE || endptr == text || *endptr != '\0') {
        return 1;
    }
    *value = (uint64_t)parsed;
    return 0;
}


static int gs_test_parse_long(const char *text, long *value)
{
    char *endptr = NULL;
    long parsed;

    errno = 0;
    parsed = strtol(text, &endptr, 10);
    if (errno == ERANGE || endptr == text || *endptr != '\0') {
        return 1;
    }
    *value = parsed;
    return 0;
}


static int gs_test_parse_double(const char *text, double *value)
{
    char *endptr = NULL;
    double parsed;

    errno = 0;
    parsed = strtod(text, &endptr);
    if (errno == ERANGE || endptr == text || *endptr != '\0' ||
        !isfinite(parsed)) {
        return 1;
    }
    *value = parsed;
    return 0;
}


static int gs_test_close(double left, double right)
{
    return isfinite(left) && isfinite(right) &&
           fabs(left - right) <= 1.0e-12;
}


static int gs_test_path_join(
    const char *directory,
    const char *filename,
    char *path,
    size_t path_size
)
{
    size_t directory_length;
    size_t filename_length;
    size_t required;
    int needs_separator;

    if (directory == NULL || filename == NULL || path == NULL ||
        path_size == 0) {
        return 1;
    }
    directory_length = strlen(directory);
    filename_length = strlen(filename);
    needs_separator = directory_length != 0 &&
        directory[directory_length - 1] != '/' &&
        directory[directory_length - 1] != '\\';
    if (directory_length > SIZE_MAX - filename_length) {
        return 1;
    }
    required = directory_length + filename_length;
    if (needs_separator) {
        if (required == SIZE_MAX) {
            return 1;
        }
        required++;
    }
    if (required >= path_size) {
        return 1;
    }
    memcpy(path, directory, directory_length);
    if (needs_separator) {
        path[directory_length++] = '/';
    }
    memcpy(path + directory_length, filename, filename_length);
    path[directory_length + filename_length] = '\0';
    return 0;
}


static int gs_test_file_size(const char *path, size_t *size)
{
    FILE *file;
    long position;
    int failed = 0;

    file = fopen(path, "rb");
    if (file == NULL) {
        return 1;
    }
    if (fseek(file, 0L, SEEK_END) != 0) {
        failed = 1;
    }
    position = failed ? -1L : ftell(file);
    if (position < 0 || fclose(file) != 0) {
        failed = 1;
    }
    if (failed) {
        return 1;
    }
    *size = (size_t)position;
    return 0;
}


static int gs_test_expected_files(const char *directory)
{
    static const char *const files[] = {
        "label_map.csv",
        "train_contingency_counts.csv",
        "val_contingency_counts.csv",
        "train_cluster_summary.csv",
        "val_cluster_summary.csv",
        "train_label_summary.csv",
        "val_label_summary.csv",
        "metrics.csv",
        "summary.txt"
    };
    char path[GS_TEST_PATH_LEN];
    size_t index;

    for (index = 0; index < sizeof(files) / sizeof(files[0]); index++) {
        size_t size;
        if (gs_test_path_join(
                directory,
                files[index],
                path,
                sizeof(path)
            ) != 0 ||
            gs_test_file_size(path, &size) != 0 || size == 0) {
            return 1;
        }
    }
    return 0;
}


static int gs_test_validate_manifest(const GSEventManifest *manifest)
{
    unsigned char present[GS_TEST_LABEL_COUNT] = {0};
    size_t index;

    if (manifest == NULL || manifest->count != GS_TEST_MANIFEST_ROWS ||
        manifest->records == NULL) {
        return 1;
    }
    for (index = 0; index < manifest->count; index++) {
        int label = gs_test_label_index(manifest->records[index].label);
        if (label < 0) {
            return 1;
        }
        present[(size_t)label] = 1;
    }
    for (index = 0; index < GS_TEST_LABEL_COUNT; index++) {
        if (!present[index]) {
            return 1;
        }
    }
    return 0;
}


static int gs_test_validate_label_map(const char *directory)
{
    char path[GS_TEST_PATH_LEN];
    char line[GS_TEST_LINE_LEN];
    FILE *file;
    size_t label;
    int failed = 0;

    if (gs_test_path_join(
            directory,
            "label_map.csv",
            path,
            sizeof(path)
        ) != 0) {
        return 1;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        return 1;
    }
    if (fgets(line, sizeof(line), file) == NULL) {
        failed = 1;
    } else {
        gs_test_strip_line_ending(line);
        if (strcmp(line, "label_index,label") != 0) {
            failed = 1;
        }
    }
    for (label = 0; !failed && label < GS_TEST_LABEL_COUNT; label++) {
        char *fields[2];
        size_t parsed_label;
        if (fgets(line, sizeof(line), file) == NULL ||
            gs_test_split_csv(line, fields, 2) != 0 ||
            gs_test_parse_size(fields[0], &parsed_label) != 0 ||
            parsed_label != label ||
            strcmp(fields[1], gs_test_labels[label]) != 0) {
            failed = 1;
        }
    }
    if (!failed && fgets(line, sizeof(line), file) != NULL) {
        failed = 1;
    }
    if (ferror(file) || fclose(file) != 0) {
        failed = 1;
    }
    return failed;
}


static int gs_test_load_rows(
    const char *path,
    const char *split,
    size_t rows,
    const GSEventManifest *manifest,
    unsigned char *coverage,
    uint32_t *labels
)
{
    static const char expected_header[] =
        "row_index,split_pos,manifest_index,event_index,event_id,split";
    FILE *file = fopen(path, "rb");
    char line[GS_TEST_LINE_LEN];
    size_t row;
    int failed = 0;

    if (file == NULL) {
        return 1;
    }
    if (fgets(line, sizeof(line), file) == NULL) {
        failed = 1;
    } else {
        gs_test_strip_line_ending(line);
        if (strcmp(line, expected_header) != 0) {
            failed = 1;
        }
    }
    for (row = 0; !failed && row < rows; row++) {
        char *fields[6];
        size_t row_index;
        size_t split_pos;
        size_t manifest_index;
        size_t event_index;
        const GSEventRecord *record;
        int label;

        if (fgets(line, sizeof(line), file) == NULL ||
            gs_test_split_csv(line, fields, 6) != 0 ||
            gs_test_parse_size(fields[0], &row_index) != 0 ||
            gs_test_parse_size(fields[1], &split_pos) != 0 ||
            gs_test_parse_size(fields[2], &manifest_index) != 0 ||
            gs_test_parse_size(fields[3], &event_index) != 0 ||
            row_index != row || split_pos != row ||
            manifest_index >= manifest->count || coverage[manifest_index] ||
            strcmp(fields[5], split) != 0) {
            failed = 1;
            break;
        }
        record = gs_event_manifest_get(manifest, manifest_index);
        label = record == NULL ? -1 : gs_test_label_index(record->label);
        if (record == NULL || label < 0 ||
            event_index != (size_t)record->event_index ||
            strcmp(fields[4], record->event_id) != 0) {
            failed = 1;
            break;
        }
        coverage[manifest_index] = 1;
        labels[row] = (uint32_t)label;
    }
    if (!failed && fgets(line, sizeof(line), file) != NULL) {
        failed = 1;
    }
    if (ferror(file) || fclose(file) != 0) {
        failed = 1;
    }
    return failed;
}


static int gs_test_read_assignments(
    const char *path,
    uint32_t *assignments,
    size_t rows,
    size_t k
)
{
    FILE *file;
    size_t expected_bytes;
    size_t actual_bytes;
    size_t row;
    int failed = 0;

    if (gs_test_checked_product(
            rows,
            sizeof(uint32_t),
            &expected_bytes
        ) != 0 ||
        gs_test_file_size(path, &actual_bytes) != 0 ||
        actual_bytes != expected_bytes) {
        return 1;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        return 1;
    }
    if (fread(assignments, sizeof(uint32_t), rows, file) != rows ||
        fgetc(file) != EOF || ferror(file)) {
        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    for (row = 0; !failed && row < rows; row++) {
        if ((size_t)assignments[row] >= k) {
            failed = 1;
        }
    }
    return failed;
}


static int gs_test_validate_contingency(
    const char *directory,
    const char *filename,
    const GSBaselineClusteringMetrics *metrics
)
{
    char path[GS_TEST_PATH_LEN];
    char line[GS_TEST_LINE_LEN];
    char *fields[GS_TEST_MAX_FIELDS];
    FILE *file;
    size_t label;
    size_t cluster;
    size_t expected_fields = metrics->cluster_count + 3;
    int failed = 0;

    if (expected_fields > GS_TEST_MAX_FIELDS ||
        gs_test_path_join(directory, filename, path, sizeof(path)) != 0) {
        return 1;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        return 1;
    }
    if (fgets(line, sizeof(line), file) == NULL ||
        gs_test_split_csv(line, fields, expected_fields) != 0 ||
        strcmp(fields[0], "label_index") != 0 ||
        strcmp(fields[1], "label") != 0 ||
        strcmp(fields[expected_fields - 1], "label_total") != 0) {
        failed = 1;
    }
    for (cluster = 0; !failed && cluster < metrics->cluster_count; cluster++) {
        char expected[64];
        if (snprintf(
                expected,
                sizeof(expected),
                "cluster_%lu",
                (unsigned long)cluster
            ) < 0 ||
            strcmp(fields[cluster + 2], expected) != 0) {
            failed = 1;
        }
    }

    for (label = 0; !failed && label < GS_TEST_LABEL_COUNT; label++) {
        size_t parsed_label;
        uint64_t parsed_total;

        if (fgets(line, sizeof(line), file) == NULL ||
            gs_test_split_csv(line, fields, expected_fields) != 0 ||
            gs_test_parse_size(fields[0], &parsed_label) != 0 ||
            parsed_label != label ||
            strcmp(fields[1], gs_test_labels[label]) != 0) {
            failed = 1;
            break;
        }
        for (cluster = 0; cluster < metrics->cluster_count; cluster++) {
            uint64_t parsed;
            if (gs_test_parse_u64(fields[cluster + 2], &parsed) != 0 ||
                parsed != metrics->contingency[
                    label * metrics->cluster_count + cluster
                ]) {
                failed = 1;
                break;
            }
        }
        if (!failed &&
            (gs_test_parse_u64(
                 fields[expected_fields - 1],
                 &parsed_total
             ) != 0 ||
             parsed_total != metrics->label_totals[label])) {
            failed = 1;
        }
    }
    if (!failed && fgets(line, sizeof(line), file) != NULL) {
        failed = 1;
    }
    if (ferror(file) || fclose(file) != 0) {
        failed = 1;
    }
    return failed;
}


static int gs_test_validate_metric_row(
    char **fields,
    const char *split,
    size_t rows,
    size_t k,
    const GSBaselineClusteringMetrics *metrics
)
{
    size_t parsed_rows;
    size_t parsed_k;
    double values[5];
    size_t index;

    if (strcmp(fields[0], split) != 0 ||
        gs_test_parse_size(fields[1], &parsed_rows) != 0 ||
        gs_test_parse_size(fields[2], &parsed_k) != 0 ||
        parsed_rows != rows || parsed_k != k) {
        return 1;
    }
    for (index = 0; index < 5; index++) {
        if (gs_test_parse_double(fields[index + 3], &values[index]) != 0) {
            return 1;
        }
    }
    return !gs_test_close(values[0], metrics->purity) ||
        !gs_test_close(values[1], metrics->normalized_mutual_information) ||
        !gs_test_close(values[2], metrics->adjusted_rand_index) ||
        !gs_test_close(values[3], metrics->homogeneity) ||
        !gs_test_close(values[4], metrics->completeness);
}


static int gs_test_validate_metrics_csv(
    const char *directory,
    size_t k,
    const GSBaselineClusteringMetrics *train,
    const GSBaselineClusteringMetrics *validation,
    int *train_passed,
    int *validation_passed
)
{
    static const char header[] =
        "split,rows,k,purity,nmi_arithmetic,ari,homogeneity,completeness";
    char path[GS_TEST_PATH_LEN];
    char line[GS_TEST_LINE_LEN];
    char *fields[8];
    FILE *file;
    int failed = 0;

    *train_passed = 0;
    *validation_passed = 0;
    if (gs_test_path_join(
            directory,
            "metrics.csv",
            path,
            sizeof(path)
        ) != 0) {
        return 1;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        return 1;
    }
    if (fgets(line, sizeof(line), file) == NULL) {
        failed = 1;
    } else {
        gs_test_strip_line_ending(line);
        if (strcmp(line, header) != 0) {
            failed = 1;
        }
    }
    if (!failed &&
        (fgets(line, sizeof(line), file) == NULL ||
         gs_test_split_csv(line, fields, 8) != 0 ||
         gs_test_validate_metric_row(
             fields,
             "train",
             GS_TEST_TRAIN_ROWS,
             k,
             train
         ) != 0)) {
        failed = 1;
    } else if (!failed) {
        *train_passed = 1;
    }
    if (!failed &&
        (fgets(line, sizeof(line), file) == NULL ||
         gs_test_split_csv(line, fields, 8) != 0 ||
         gs_test_validate_metric_row(
             fields,
             "validation",
             GS_TEST_VALIDATION_ROWS,
             k,
             validation
         ) != 0)) {
        failed = 1;
    } else if (!failed) {
        *validation_passed = 1;
    }
    if (!failed && fgets(line, sizeof(line), file) != NULL) {
        failed = 1;
    }
    if (ferror(file) || fclose(file) != 0) {
        failed = 1;
    }
    return failed;
}


static void gs_test_top_two_labels(
    const GSBaselineClusteringMetrics *metrics,
    size_t cluster,
    size_t *first,
    uint64_t *first_count,
    size_t *second,
    uint64_t *second_count
)
{
    size_t label;

    *first = SIZE_MAX;
    *second = SIZE_MAX;
    *first_count = UINT64_C(0);
    *second_count = UINT64_C(0);
    for (label = 0; label < GS_TEST_LABEL_COUNT; label++) {
        uint64_t count =
            metrics->contingency[label * metrics->cluster_count + cluster];
        if (*first == SIZE_MAX || count > *first_count) {
            *second = *first;
            *second_count = *first_count;
            *first = label;
            *first_count = count;
        } else if (*second == SIZE_MAX || count > *second_count) {
            *second = label;
            *second_count = count;
        }
    }
}


static int gs_test_validate_cluster_summary(
    const char *directory,
    const char *filename,
    const GSBaselineClusteringMetrics *metrics
)
{
    static const char header[] =
        "cluster,size,dominant_label_index,dominant_label,dominant_count,"
        "cluster_purity,second_label_index,second_label,second_count";
    char path[GS_TEST_PATH_LEN];
    char line[GS_TEST_LINE_LEN];
    FILE *file;
    size_t cluster;
    int failed = 0;

    if (gs_test_path_join(directory, filename, path, sizeof(path)) != 0) {
        return 1;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        return 1;
    }
    if (fgets(line, sizeof(line), file) == NULL) {
        failed = 1;
    } else {
        gs_test_strip_line_ending(line);
        if (strcmp(line, header) != 0) {
            failed = 1;
        }
    }
    for (cluster = 0; !failed && cluster < metrics->cluster_count; cluster++) {
        char *fields[9];
        size_t parsed_cluster;
        uint64_t parsed_size;
        long parsed_first;
        uint64_t parsed_first_count;
        double parsed_purity;
        long parsed_second;
        uint64_t parsed_second_count;
        uint64_t size = metrics->cluster_totals[cluster];

        if (fgets(line, sizeof(line), file) == NULL ||
            gs_test_split_csv(line, fields, 9) != 0 ||
            gs_test_parse_size(fields[0], &parsed_cluster) != 0 ||
            gs_test_parse_u64(fields[1], &parsed_size) != 0 ||
            gs_test_parse_long(fields[2], &parsed_first) != 0 ||
            gs_test_parse_u64(fields[4], &parsed_first_count) != 0 ||
            gs_test_parse_double(fields[5], &parsed_purity) != 0 ||
            gs_test_parse_long(fields[6], &parsed_second) != 0 ||
            gs_test_parse_u64(fields[8], &parsed_second_count) != 0 ||
            parsed_cluster != cluster || parsed_size != size) {
            failed = 1;
            break;
        }

        if (size == 0) {
            if (parsed_first != -1 || strcmp(fields[3], "NA") != 0 ||
                parsed_first_count != 0 || !gs_test_close(parsed_purity, 0.0) ||
                parsed_second != -1 || strcmp(fields[7], "NA") != 0 ||
                parsed_second_count != 0) {
                failed = 1;
            }
        } else {
            size_t first;
            size_t second;
            uint64_t first_count;
            uint64_t second_count;
            double purity;

            gs_test_top_two_labels(
                metrics,
                cluster,
                &first,
                &first_count,
                &second,
                &second_count
            );
            purity = (double)first_count / (double)size;
            if (parsed_first != (long)first ||
                strcmp(fields[3], gs_test_labels[first]) != 0 ||
                parsed_first_count != first_count ||
                !gs_test_close(parsed_purity, purity) ||
                parsed_second != (long)second ||
                strcmp(fields[7], gs_test_labels[second]) != 0 ||
                parsed_second_count != second_count) {
                failed = 1;
            }
        }
    }
    if (!failed && fgets(line, sizeof(line), file) != NULL) {
        failed = 1;
    }
    if (ferror(file) || fclose(file) != 0) {
        failed = 1;
    }
    return failed;
}


static void gs_test_label_statistics(
    const GSBaselineClusteringMetrics *metrics,
    size_t label,
    GSTestLabelStatistics *statistics
)
{
    size_t cluster;

    statistics->total = (size_t)metrics->label_totals[label];
    statistics->occupied = 0;
    statistics->dominant = SIZE_MAX;
    statistics->dominant_count = UINT64_C(0);
    statistics->second = SIZE_MAX;
    statistics->second_count = UINT64_C(0);
    for (cluster = 0; cluster < metrics->cluster_count; cluster++) {
        uint64_t count =
            metrics->contingency[label * metrics->cluster_count + cluster];
        if (count != 0) {
            statistics->occupied++;
        }
        if (statistics->dominant == SIZE_MAX ||
            count > statistics->dominant_count) {
            statistics->second = statistics->dominant;
            statistics->second_count = statistics->dominant_count;
            statistics->dominant = cluster;
            statistics->dominant_count = count;
        } else if (statistics->second == SIZE_MAX ||
                   count > statistics->second_count) {
            statistics->second = cluster;
            statistics->second_count = count;
        }
    }
    statistics->dominant_fraction = statistics->total == 0
        ? 0.0
        : (double)statistics->dominant_count / (double)statistics->total;
}


static int gs_test_validate_label_summary(
    const char *directory,
    const char *filename,
    const GSBaselineClusteringMetrics *metrics
)
{
    static const char header[] =
        "label_index,label,total,occupied_clusters,dominant_cluster,"
        "dominant_count,dominant_fraction,second_cluster,second_count";
    char path[GS_TEST_PATH_LEN];
    char line[GS_TEST_LINE_LEN];
    FILE *file;
    size_t label;
    int failed = 0;

    if (gs_test_path_join(directory, filename, path, sizeof(path)) != 0) {
        return 1;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        return 1;
    }
    if (fgets(line, sizeof(line), file) == NULL) {
        failed = 1;
    } else {
        gs_test_strip_line_ending(line);
        if (strcmp(line, header) != 0) {
            failed = 1;
        }
    }
    for (label = 0; !failed && label < GS_TEST_LABEL_COUNT; label++) {
        char *fields[9];
        size_t parsed_label;
        uint64_t parsed_total;
        size_t parsed_occupied;
        long parsed_dominant;
        uint64_t parsed_dominant_count;
        double parsed_fraction;
        long parsed_second;
        uint64_t parsed_second_count;
        GSTestLabelStatistics statistics;

        if (fgets(line, sizeof(line), file) == NULL ||
            gs_test_split_csv(line, fields, 9) != 0 ||
            gs_test_parse_size(fields[0], &parsed_label) != 0 ||
            gs_test_parse_u64(fields[2], &parsed_total) != 0 ||
            gs_test_parse_size(fields[3], &parsed_occupied) != 0 ||
            gs_test_parse_long(fields[4], &parsed_dominant) != 0 ||
            gs_test_parse_u64(fields[5], &parsed_dominant_count) != 0 ||
            gs_test_parse_double(fields[6], &parsed_fraction) != 0 ||
            gs_test_parse_long(fields[7], &parsed_second) != 0 ||
            gs_test_parse_u64(fields[8], &parsed_second_count) != 0 ||
            parsed_label != label ||
            strcmp(fields[1], gs_test_labels[label]) != 0) {
            failed = 1;
            break;
        }

        gs_test_label_statistics(metrics, label, &statistics);
        if (parsed_total != (uint64_t)statistics.total ||
            parsed_occupied != statistics.occupied ||
            parsed_dominant_count != statistics.dominant_count ||
            parsed_second_count != statistics.second_count ||
            !gs_test_close(parsed_fraction, statistics.dominant_fraction) ||
            (statistics.total == 0
                ? parsed_dominant != -1 || parsed_second != -1
                : parsed_dominant != (long)statistics.dominant ||
                  parsed_second != (long)statistics.second)) {
            failed = 1;
        }
    }
    if (!failed && fgets(line, sizeof(line), file) != NULL) {
        failed = 1;
    }
    if (ferror(file) || fclose(file) != 0) {
        failed = 1;
    }
    return failed;
}


static int gs_test_read_text(
    const char *directory,
    const char *filename,
    char **text
)
{
    char path[GS_TEST_PATH_LEN];
    size_t size;
    FILE *file;
    char *buffer;
    int failed = 0;

    if (gs_test_path_join(directory, filename, path, sizeof(path)) != 0 ||
        gs_test_file_size(path, &size) != 0 || size == 0 ||
        size == SIZE_MAX) {
        return 1;
    }
    buffer = (char *)malloc(size + 1);
    if (buffer == NULL) {
        return 1;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        free(buffer);
        return 1;
    }
    if (fread(buffer, 1, size, file) != size || fgetc(file) != EOF ||
        ferror(file) || memchr(buffer, '\0', size) != NULL) {
        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    if (failed) {
        free(buffer);
        return 1;
    }
    buffer[size] = '\0';
    *text = buffer;
    return 0;
}


static int gs_test_text_has_line(const char *text, const char *line)
{
    const char *cursor = text;
    size_t length = strlen(line);

    while ((cursor = strstr(cursor, line)) != NULL) {
        const char *after = cursor + length;
        if ((cursor == text || cursor[-1] == '\n') &&
            (*after == '\0' || *after == '\r' || *after == '\n')) {
            return 1;
        }
        cursor++;
    }
    return 0;
}


static int gs_test_text_has_key(const char *text, const char *key)
{
    const char *cursor = text;
    size_t length = strlen(key);

    while ((cursor = strstr(cursor, key)) != NULL) {
        if ((cursor == text || cursor[-1] == '\n') &&
            cursor[length] != '\0' && cursor[length] != '\r' &&
            cursor[length] != '\n') {
            return 1;
        }
        cursor++;
    }
    return 0;
}


static int gs_test_metadata_contract(
    const char *directory,
    size_t k,
    const char *commit,
    int check_usage
)
{
    static const char *const fixed[] = {
        "format_version=1",
        "profile=mini75_pool5_no_pca",
        "evaluation=labels_after_clustering",
        "label_count=22",
        "seed=123",
        "train_rows=6866",
        "validation_rows=1717",
        "nmi_normalization=arithmetic",
        "purity_definition=weighted_cluster_majority"
    };
    static const char *const metric_keys[] = {
        "train_purity=",
        "train_nmi_arithmetic=",
        "train_ari=",
        "train_homogeneity=",
        "train_completeness=",
        "validation_purity=",
        "validation_nmi_arithmetic=",
        "validation_ari=",
        "validation_homogeneity=",
        "validation_completeness=",
        "largest_train_cluster=",
        "largest_train_cluster_size=",
        "minimum_purity_train_cluster=",
        "minimum_train_cluster_purity=",
        "maximum_purity_train_cluster=",
        "maximum_train_cluster_purity=",
        "Blip_train_total=",
        "Blip_validation_total=",
        "Koi_Fish_train_total=",
        "Koi_Fish_validation_total=",
        "Scattered_Light_train_total=",
        "Scattered_Light_validation_total=",
        "None_of_the_Above_train_total=",
        "None_of_the_Above_validation_total="
    };
    char k_line[64];
    char commit_line[128];
    char *text = NULL;
    size_t index;
    int failed = 0;

    if (snprintf(k_line, sizeof(k_line), "k=%lu", (unsigned long)k) < 0 ||
        snprintf(
            commit_line,
            sizeof(commit_line),
            "code_commit=%s",
            commit
        ) < 0 ||
        gs_test_read_text(directory, "summary.txt", &text) != 0) {
        free(text);
        return 1;
    }
    for (index = 0; index < sizeof(fixed) / sizeof(fixed[0]); index++) {
        if (!gs_test_text_has_line(text, fixed[index])) {
            failed = 1;
        }
    }
    for (index = 0;
         index < sizeof(metric_keys) / sizeof(metric_keys[0]);
         index++) {
        if (!gs_test_text_has_key(text, metric_keys[index])) {
            failed = 1;
        }
    }
    for (index = 1; index <= 5; index++) {
        char key[128];
        if (snprintf(
                key,
                sizeof(key),
                "largest_train_cluster_top_%lu_label=",
                (unsigned long)index
            ) < 0 ||
            !gs_test_text_has_key(text, key)) {
            failed = 1;
        }
    }
    if (!gs_test_text_has_line(text, k_line) ||
        !gs_test_text_has_line(text, commit_line)) {
        failed = 1;
    }
    if (check_usage &&
        (!gs_test_text_has_line(text, "labels_used_for_training=false") ||
         !gs_test_text_has_line(text, "labels_used_for_clustering=false") ||
         !gs_test_text_has_line(text, "labels_used_for_evaluation=true"))) {
        failed = 1;
    }
    free(text);
    return failed;
}


int main(int argc, char **argv)
{
    GSEventManifest manifest;
    GSBaselineClusteringMetrics train_metrics;
    GSBaselineClusteringMetrics validation_metrics;
    unsigned char *coverage = NULL;
    uint32_t *train_labels = NULL;
    uint32_t *validation_labels = NULL;
    uint32_t *train_assignments = NULL;
    uint32_t *validation_assignments = NULL;
    uint32_t *train_copy = NULL;
    uint32_t *validation_copy = NULL;
    size_t train_bytes;
    size_t validation_bytes;
    size_t k;
    size_t index;
    int train_metrics_passed = 0;
    int validation_metrics_passed = 0;
    int rc = 1;

    memset(&manifest, 0, sizeof(manifest));
    memset(&train_metrics, 0, sizeof(train_metrics));
    memset(&validation_metrics, 0, sizeof(validation_metrics));

    if (argc != 9 || !gs_test_platform_is_valid() ||
        gs_test_parse_size(argv[6], &k) != 0 ||
        (k != 22 && k != 36) ||
        strcmp(argv[8], GS_TEST_CODE_COMMIT) != 0 ||
        gs_test_checked_product(
            GS_TEST_TRAIN_ROWS,
            sizeof(uint32_t),
            &train_bytes
        ) != 0 ||
        gs_test_checked_product(
            GS_TEST_VALIDATION_ROWS,
            sizeof(uint32_t),
            &validation_bytes
        ) != 0) {
        fprintf(stderr, "Errore: argomenti validatore non validi.\n");
        return 1;
    }

    if (gs_test_expected_files(argv[7]) != 0) {
        fprintf(stderr, "[FAIL] expected_files\n");
        goto cleanup;
    }
    printf("[PASS] expected_files\n");

    coverage = (unsigned char *)calloc(
        GS_TEST_MANIFEST_ROWS,
        sizeof(unsigned char)
    );
    train_labels = (uint32_t *)malloc(train_bytes);
    validation_labels = (uint32_t *)malloc(validation_bytes);
    train_assignments = (uint32_t *)malloc(train_bytes);
    validation_assignments = (uint32_t *)malloc(validation_bytes);
    train_copy = (uint32_t *)malloc(train_bytes);
    validation_copy = (uint32_t *)malloc(validation_bytes);

    if (coverage == NULL || train_labels == NULL ||
        validation_labels == NULL || train_assignments == NULL ||
        validation_assignments == NULL || train_copy == NULL ||
        validation_copy == NULL ||
        gs_event_manifest_load(argv[1], &manifest) != 0 ||
        gs_test_validate_manifest(&manifest) != 0 ||
        gs_test_validate_label_map(argv[7]) != 0) {
        fprintf(stderr, "[FAIL] official_label_map\n");
        goto cleanup;
    }
    printf("[PASS] official_label_map\n");

    if (gs_test_load_rows(
            argv[2],
            "train",
            GS_TEST_TRAIN_ROWS,
            &manifest,
            coverage,
            train_labels
        ) != 0 ||
        gs_test_load_rows(
            argv[3],
            "validation",
            GS_TEST_VALIDATION_ROWS,
            &manifest,
            coverage,
            validation_labels
        ) != 0 ||
        gs_test_read_assignments(
            argv[4],
            train_assignments,
            GS_TEST_TRAIN_ROWS,
            k
        ) != 0 ||
        gs_test_read_assignments(
            argv[5],
            validation_assignments,
            GS_TEST_VALIDATION_ROWS,
            k
        ) != 0) {
        fprintf(stderr, "[FAIL] row_mapping_and_coverage\n");
        goto cleanup;
    }
    for (index = 0; index < GS_TEST_MANIFEST_ROWS; index++) {
        if (!coverage[index]) {
            fprintf(stderr, "[FAIL] row_mapping_and_coverage\n");
            goto cleanup;
        }
    }
    memcpy(train_copy, train_assignments, train_bytes);
    memcpy(validation_copy, validation_assignments, validation_bytes);
    printf("[PASS] row_mapping_and_coverage\n");

    if (gs_baseline_clustering_metrics_compute(
            train_labels,
            train_assignments,
            GS_TEST_TRAIN_ROWS,
            GS_TEST_LABEL_COUNT,
            k,
            &train_metrics
        ) != GS_BASELINE_CLUSTERING_METRICS_SUCCESS ||
        gs_baseline_clustering_metrics_compute(
            validation_labels,
            validation_assignments,
            GS_TEST_VALIDATION_ROWS,
            GS_TEST_LABEL_COUNT,
            k,
            &validation_metrics
        ) != GS_BASELINE_CLUSTERING_METRICS_SUCCESS ||
        gs_test_validate_contingency(
            argv[7],
            "train_contingency_counts.csv",
            &train_metrics
        ) != 0 ||
        gs_test_validate_contingency(
            argv[7],
            "val_contingency_counts.csv",
            &validation_metrics
        ) != 0) {
        fprintf(stderr, "[FAIL] contingency_contract\n");
        goto cleanup;
    }
    printf("[PASS] contingency_contract\n");

    if (gs_test_validate_metrics_csv(
            argv[7],
            k,
            &train_metrics,
            &validation_metrics,
            &train_metrics_passed,
            &validation_metrics_passed
        ) != 0 || !train_metrics_passed || !validation_metrics_passed) {
        fprintf(stderr, "[FAIL] recomputed metrics\n");
        goto cleanup;
    }
    printf("[PASS] recomputed_train_metrics\n");
    printf("[PASS] recomputed_validation_metrics\n");

    if (gs_test_validate_cluster_summary(
            argv[7],
            "train_cluster_summary.csv",
            &train_metrics
        ) != 0 ||
        gs_test_validate_cluster_summary(
            argv[7],
            "val_cluster_summary.csv",
            &validation_metrics
        ) != 0) {
        fprintf(stderr, "[FAIL] cluster_summary_contract\n");
        goto cleanup;
    }
    printf("[PASS] cluster_summary_contract\n");

    if (gs_test_validate_label_summary(
            argv[7],
            "train_label_summary.csv",
            &train_metrics
        ) != 0 ||
        gs_test_validate_label_summary(
            argv[7],
            "val_label_summary.csv",
            &validation_metrics
        ) != 0) {
        fprintf(stderr, "[FAIL] label_summary_contract\n");
        goto cleanup;
    }
    printf("[PASS] label_summary_contract\n");

    if (gs_test_metadata_contract(argv[7], k, argv[8], 0) != 0) {
        fprintf(stderr, "[FAIL] metadata_contract\n");
        goto cleanup;
    }
    printf("[PASS] metadata_contract\n");

    if (gs_test_metadata_contract(argv[7], k, argv[8], 1) != 0 ||
        memcmp(train_assignments, train_copy, train_bytes) != 0 ||
        memcmp(validation_assignments, validation_copy,
               validation_bytes) != 0) {
        fprintf(stderr, "[FAIL] evaluation_only_label_usage\n");
        goto cleanup;
    }
    printf("[PASS] evaluation_only_label_usage\n");
    printf("K=%lu\n", (unsigned long)k);
    printf("TRAIN_ROWS=6866\n");
    printf("VALIDATION_ROWS=1717\n");
    printf("LABEL_COUNT=22\n");
    printf("ALL EVALUATION ARTIFACT TESTS PASSED\n");
    rc = 0;

cleanup:
    gs_baseline_clustering_metrics_free(&validation_metrics);
    gs_baseline_clustering_metrics_free(&train_metrics);
    gs_event_manifest_free(&manifest);
    free(validation_copy);
    free(train_copy);
    free(validation_assignments);
    free(train_assignments);
    free(validation_labels);
    free(train_labels);
    free(coverage);
    return rc;
}
