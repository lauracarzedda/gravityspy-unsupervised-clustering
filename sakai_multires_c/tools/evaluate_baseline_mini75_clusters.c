#include "baseline_clustering_metrics.h"
#include "vae_event_dataset.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/types.h>
#endif


enum {
    GS_EVAL_LABEL_COUNT = 22,
    GS_EVAL_TRAIN_ROWS = 6866,
    GS_EVAL_VALIDATION_ROWS = 1717,
    GS_EVAL_MANIFEST_ROWS = 8583,
    GS_EVAL_PATH_LEN = 4096,
    GS_EVAL_LINE_LEN = 4096
};


#define GS_EVAL_PARTIAL_SUFFIX ".partial"
#define GS_EVAL_CODE_COMMIT "e701884ca8e006800483aff7b84318b3c7d2a7b0"


static const char *const gs_eval_labels[GS_EVAL_LABEL_COUNT] = {
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
    size_t occupied_clusters;
    size_t dominant_cluster;
    uint64_t dominant_count;
    double dominant_fraction;
    size_t second_cluster;
    uint64_t second_count;
} GSLabelStatistics;


static int gs_eval_checked_product(
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


static int gs_eval_platform_is_valid(void)
{
    uint16_t value = UINT16_C(1);
    const unsigned char *bytes = (const unsigned char *)&value;

    return sizeof(uint32_t) == 4 && sizeof(uint64_t) == 8 &&
           bytes[0] == 1u;
}


static int gs_eval_label_index(const char *label)
{
    size_t index;

    if (label == NULL) {
        return -1;
    }
    for (index = 0; index < GS_EVAL_LABEL_COUNT; index++) {
        if (strcmp(label, gs_eval_labels[index]) == 0) {
            return (int)index;
        }
    }
    return -1;
}


static void gs_eval_strip_line_ending(char *line)
{
    size_t length = strlen(line);

    while (length != 0 &&
           (line[length - 1] == '\r' || line[length - 1] == '\n')) {
        line[--length] = '\0';
    }
}


static int gs_eval_split_csv(
    char *line,
    char **fields,
    size_t expected_fields
)
{
    size_t count = 0;
    char *cursor;

    if (line == NULL || fields == NULL || expected_fields == 0) {
        return 1;
    }
    gs_eval_strip_line_ending(line);
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


static int gs_eval_parse_size(const char *text, size_t *value)
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


static int gs_eval_path_join(
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


static int gs_eval_path_state(const char *path)
{
    struct stat information;

    errno = 0;
    if (stat(path, &information) == 0) {
        return 1;
    }
    return errno == ENOENT ? 0 : -1;
}


static int gs_eval_validate_output_paths(
    const char *partial_path,
    char *final_path,
    size_t final_path_size
)
{
    size_t partial_length;
    size_t suffix_length = strlen(GS_EVAL_PARTIAL_SUFFIX);
    size_t final_length;

    if (partial_path == NULL || partial_path[0] == '\0' ||
        strchr(partial_path, '\r') != NULL ||
        strchr(partial_path, '\n') != NULL) {
        return 1;
    }
    partial_length = strlen(partial_path);
    if (partial_length <= suffix_length ||
        strcmp(
            partial_path + partial_length - suffix_length,
            GS_EVAL_PARTIAL_SUFFIX
        ) != 0) {
        return 1;
    }
    final_length = partial_length - suffix_length;
    if (final_length == 0 || final_length >= final_path_size) {
        return 1;
    }
    memcpy(final_path, partial_path, final_length);
    final_path[final_length] = '\0';
    return gs_eval_path_state(partial_path) == 0 &&
           gs_eval_path_state(final_path) == 0 ? 0 : 1;
}


static int gs_eval_create_directory(
    const char *partial_path,
    const char *final_path
)
{
    if (gs_eval_path_state(partial_path) != 0 ||
        gs_eval_path_state(final_path) != 0) {
        return 1;
    }
#ifdef _WIN32
    return _mkdir(partial_path) == 0 ? 0 : 1;
#else
    return mkdir(partial_path, 0777) == 0 ? 0 : 1;
#endif
}


static int gs_eval_validate_manifest(const GSEventManifest *manifest)
{
    unsigned char present[GS_EVAL_LABEL_COUNT] = {0};
    size_t index;

    if (manifest == NULL || manifest->records == NULL ||
        manifest->count != GS_EVAL_MANIFEST_ROWS) {
        return 1;
    }
    for (index = 0; index < manifest->count; index++) {
        int label_index = gs_eval_label_index(manifest->records[index].label);

        if (label_index < 0) {
            return 1;
        }
        present[(size_t)label_index] = 1;
    }
    for (index = 0; index < GS_EVAL_LABEL_COUNT; index++) {
        if (!present[index]) {
            return 1;
        }
    }
    return 0;
}


static int gs_eval_load_rows(
    const char *path,
    const char *expected_split,
    size_t expected_rows,
    const GSEventManifest *manifest,
    unsigned char *coverage,
    uint32_t *labels
)
{
    static const char expected_header[] =
        "row_index,split_pos,manifest_index,event_index,event_id,split";
    FILE *file;
    char line[GS_EVAL_LINE_LEN];
    size_t row;
    int failed = 0;

    file = fopen(path, "rb");
    if (file == NULL) {
        return 1;
    }
    if (fgets(line, sizeof(line), file) == NULL) {
        failed = 1;
    } else {
        gs_eval_strip_line_ending(line);
        if (strcmp(line, expected_header) != 0) {
            failed = 1;
        }
    }

    for (row = 0; !failed && row < expected_rows; row++) {
        char *fields[6];
        size_t row_index;
        size_t split_pos;
        size_t manifest_index;
        size_t event_index;
        const GSEventRecord *record;
        int label_index;

        if (fgets(line, sizeof(line), file) == NULL ||
            gs_eval_split_csv(line, fields, 6) != 0 ||
            gs_eval_parse_size(fields[0], &row_index) != 0 ||
            gs_eval_parse_size(fields[1], &split_pos) != 0 ||
            gs_eval_parse_size(fields[2], &manifest_index) != 0 ||
            gs_eval_parse_size(fields[3], &event_index) != 0 ||
            row_index != row || split_pos != row ||
            manifest_index >= manifest->count || coverage[manifest_index] ||
            strcmp(fields[5], expected_split) != 0) {
            failed = 1;
            break;
        }

        record = gs_event_manifest_get(manifest, manifest_index);
        label_index = record == NULL ? -1 : gs_eval_label_index(record->label);
        if (record == NULL || label_index < 0 ||
            event_index != (size_t)record->event_index ||
            strcmp(fields[4], record->event_id) != 0) {
            failed = 1;
            break;
        }

        coverage[manifest_index] = 1;
        labels[row] = (uint32_t)label_index;
    }

    if (!failed && fgets(line, sizeof(line), file) != NULL) {
        failed = 1;
    }
    if (ferror(file) || fclose(file) != 0) {
        failed = 1;
    }
    return failed;
}


static int gs_eval_file_size(const char *path, size_t *size)
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


static int gs_eval_read_assignments(
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

    if (gs_eval_checked_product(
            rows,
            sizeof(uint32_t),
            &expected_bytes
        ) != 0 ||
        gs_eval_file_size(path, &actual_bytes) != 0 ||
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
    if (failed) {
        return 1;
    }
    for (row = 0; row < rows; row++) {
        if ((size_t)assignments[row] >= k) {
            return 1;
        }
    }
    return 0;
}


static int gs_eval_metrics_are_valid(
    const GSBaselineClusteringMetrics *metrics,
    size_t rows,
    size_t k
)
{
    uint64_t total = UINT64_C(0);
    size_t label;

    if (metrics == NULL || metrics->rows != rows ||
        metrics->label_count != GS_EVAL_LABEL_COUNT ||
        metrics->cluster_count != k || metrics->contingency == NULL ||
        metrics->label_totals == NULL || metrics->cluster_totals == NULL ||
        !isfinite(metrics->purity) ||
        !isfinite(metrics->normalized_mutual_information) ||
        !isfinite(metrics->adjusted_rand_index) ||
        !isfinite(metrics->homogeneity) ||
        !isfinite(metrics->completeness) ||
        metrics->purity < 0.0 || metrics->purity > 1.0 ||
        metrics->normalized_mutual_information < 0.0 ||
        metrics->normalized_mutual_information > 1.0 ||
        metrics->adjusted_rand_index < -1.0 ||
        metrics->adjusted_rand_index > 1.0 ||
        metrics->homogeneity < 0.0 || metrics->homogeneity > 1.0 ||
        metrics->completeness < 0.0 || metrics->completeness > 1.0) {
        return 0;
    }

    for (label = 0; label < GS_EVAL_LABEL_COUNT; label++) {
        if (total > UINT64_MAX - metrics->label_totals[label]) {
            return 0;
        }
        total += metrics->label_totals[label];
    }
    return total == (uint64_t)rows;
}


static int gs_eval_open_output(
    const char *directory,
    const char *filename,
    FILE **file
)
{
    char path[GS_EVAL_PATH_LEN];

    if (file == NULL ||
        gs_eval_path_join(directory, filename, path, sizeof(path)) != 0) {
        return 1;
    }
    *file = fopen(path, "wb");
    return *file == NULL ? 1 : 0;
}


static int gs_eval_finish_output(FILE *file, int failed)
{
    if (file == NULL) {
        return 1;
    }
    if (fflush(file) != 0 || ferror(file)) {
        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    return failed;
}


static int gs_eval_write_label_map(const char *directory)
{
    FILE *file = NULL;
    size_t label;
    int failed = 0;

    if (gs_eval_open_output(directory, "label_map.csv", &file) != 0) {
        return 1;
    }
    if (fputs("label_index,label\n", file) == EOF) {
        failed = 1;
    }
    for (label = 0; !failed && label < GS_EVAL_LABEL_COUNT; label++) {
        if (fprintf(file, "%lu,%s\n", (unsigned long)label,
                    gs_eval_labels[label]) < 0) {
            failed = 1;
        }
    }
    return gs_eval_finish_output(file, failed);
}


static int gs_eval_write_contingency(
    const char *directory,
    const char *filename,
    const GSBaselineClusteringMetrics *metrics
)
{
    FILE *file = NULL;
    size_t label;
    size_t cluster;
    int failed = 0;

    if (gs_eval_open_output(directory, filename, &file) != 0) {
        return 1;
    }
    if (fputs("label_index,label", file) == EOF) {
        failed = 1;
    }
    for (cluster = 0; !failed && cluster < metrics->cluster_count; cluster++) {
        if (fprintf(file, ",cluster_%lu", (unsigned long)cluster) < 0) {
            failed = 1;
        }
    }
    if (!failed && fputs(",label_total\n", file) == EOF) {
        failed = 1;
    }

    for (label = 0; !failed && label < GS_EVAL_LABEL_COUNT; label++) {
        if (fprintf(file, "%lu,%s", (unsigned long)label,
                    gs_eval_labels[label]) < 0) {
            failed = 1;
        }
        for (cluster = 0; !failed && cluster < metrics->cluster_count;
             cluster++) {
            if (fprintf(
                    file,
                    ",%lu",
                    (unsigned long)metrics->contingency[
                        label * metrics->cluster_count + cluster
                    ]
                ) < 0) {
                failed = 1;
            }
        }
        if (!failed && fprintf(
                file,
                ",%lu\n",
                (unsigned long)metrics->label_totals[label]
            ) < 0) {
            failed = 1;
        }
    }
    return gs_eval_finish_output(file, failed);
}


static void gs_eval_top_two_labels(
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

    for (label = 0; label < GS_EVAL_LABEL_COUNT; label++) {
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


static int gs_eval_write_cluster_summary(
    const char *directory,
    const char *filename,
    const GSBaselineClusteringMetrics *metrics
)
{
    static const char header[] =
        "cluster,size,dominant_label_index,dominant_label,dominant_count,"
        "cluster_purity,second_label_index,second_label,second_count\n";
    FILE *file = NULL;
    size_t cluster;
    int failed = 0;

    if (gs_eval_open_output(directory, filename, &file) != 0) {
        return 1;
    }
    if (fputs(header, file) == EOF) {
        failed = 1;
    }

    for (cluster = 0; !failed && cluster < metrics->cluster_count; cluster++) {
        uint64_t size = metrics->cluster_totals[cluster];

        if (size == 0) {
            if (fprintf(
                    file,
                    "%lu,0,-1,NA,0,0,-1,NA,0\n",
                    (unsigned long)cluster
                ) < 0) {
                failed = 1;
            }
        } else {
            size_t first;
            size_t second;
            uint64_t first_count;
            uint64_t second_count;
            double purity;

            gs_eval_top_two_labels(
                metrics,
                cluster,
                &first,
                &first_count,
                &second,
                &second_count
            );
            purity = (double)first_count / (double)size;
            if (fprintf(
                    file,
                    "%lu,%lu,%lu,%s,%lu,%.17g,%lu,%s,%lu\n",
                    (unsigned long)cluster,
                    (unsigned long)size,
                    (unsigned long)first,
                    gs_eval_labels[first],
                    (unsigned long)first_count,
                    purity,
                    (unsigned long)second,
                    gs_eval_labels[second],
                    (unsigned long)second_count
                ) < 0) {
                failed = 1;
            }
        }
    }
    return gs_eval_finish_output(file, failed);
}


static void gs_eval_label_statistics(
    const GSBaselineClusteringMetrics *metrics,
    size_t label,
    GSLabelStatistics *statistics
)
{
    size_t cluster;

    statistics->total = (size_t)metrics->label_totals[label];
    statistics->occupied_clusters = 0;
    statistics->dominant_cluster = SIZE_MAX;
    statistics->dominant_count = UINT64_C(0);
    statistics->second_cluster = SIZE_MAX;
    statistics->second_count = UINT64_C(0);

    for (cluster = 0; cluster < metrics->cluster_count; cluster++) {
        uint64_t count =
            metrics->contingency[label * metrics->cluster_count + cluster];

        if (count != 0) {
            statistics->occupied_clusters++;
        }
        if (statistics->dominant_cluster == SIZE_MAX ||
            count > statistics->dominant_count) {
            statistics->second_cluster = statistics->dominant_cluster;
            statistics->second_count = statistics->dominant_count;
            statistics->dominant_cluster = cluster;
            statistics->dominant_count = count;
        } else if (statistics->second_cluster == SIZE_MAX ||
                   count > statistics->second_count) {
            statistics->second_cluster = cluster;
            statistics->second_count = count;
        }
    }

    statistics->dominant_fraction = statistics->total == 0
        ? 0.0
        : (double)statistics->dominant_count / (double)statistics->total;
}


static int gs_eval_write_label_summary(
    const char *directory,
    const char *filename,
    const GSBaselineClusteringMetrics *metrics
)
{
    static const char header[] =
        "label_index,label,total,occupied_clusters,dominant_cluster,"
        "dominant_count,dominant_fraction,second_cluster,second_count\n";
    FILE *file = NULL;
    size_t label;
    int failed = 0;

    if (gs_eval_open_output(directory, filename, &file) != 0) {
        return 1;
    }
    if (fputs(header, file) == EOF) {
        failed = 1;
    }
    for (label = 0; !failed && label < GS_EVAL_LABEL_COUNT; label++) {
        GSLabelStatistics statistics;

        gs_eval_label_statistics(metrics, label, &statistics);
        if (statistics.total == 0) {
            if (fprintf(
                    file,
                    "%lu,%s,0,0,-1,0,0,-1,0\n",
                    (unsigned long)label,
                    gs_eval_labels[label]
                ) < 0) {
                failed = 1;
            }
        } else if (fprintf(
                file,
                "%lu,%s,%lu,%lu,%lu,%lu,%.17g,%lu,%lu\n",
                (unsigned long)label,
                gs_eval_labels[label],
                (unsigned long)statistics.total,
                (unsigned long)statistics.occupied_clusters,
                (unsigned long)statistics.dominant_cluster,
                (unsigned long)statistics.dominant_count,
                statistics.dominant_fraction,
                (unsigned long)statistics.second_cluster,
                (unsigned long)statistics.second_count
            ) < 0) {
            failed = 1;
        }
    }
    return gs_eval_finish_output(file, failed);
}


static int gs_eval_write_metrics(
    const char *directory,
    size_t k,
    const GSBaselineClusteringMetrics *train,
    const GSBaselineClusteringMetrics *validation
)
{
    FILE *file = NULL;
    int failed = 0;

    if (gs_eval_open_output(directory, "metrics.csv", &file) != 0) {
        return 1;
    }
    if (fputs(
            "split,rows,k,purity,nmi_arithmetic,ari,homogeneity,completeness\n",
            file
        ) == EOF ||
        fprintf(
            file,
            "train,6866,%lu,%.17g,%.17g,%.17g,%.17g,%.17g\n",
            (unsigned long)k,
            train->purity,
            train->normalized_mutual_information,
            train->adjusted_rand_index,
            train->homogeneity,
            train->completeness
        ) < 0 ||
        fprintf(
            file,
            "validation,1717,%lu,%.17g,%.17g,%.17g,%.17g,%.17g\n",
            (unsigned long)k,
            validation->purity,
            validation->normalized_mutual_information,
            validation->adjusted_rand_index,
            validation->homogeneity,
            validation->completeness
        ) < 0) {
        failed = 1;
    }
    return gs_eval_finish_output(file, failed);
}


static size_t gs_eval_largest_cluster(
    const GSBaselineClusteringMetrics *metrics
)
{
    size_t best = 0;
    size_t cluster;

    for (cluster = 1; cluster < metrics->cluster_count; cluster++) {
        if (metrics->cluster_totals[cluster] > metrics->cluster_totals[best]) {
            best = cluster;
        }
    }
    return best;
}


static void gs_eval_purity_extremes(
    const GSBaselineClusteringMetrics *metrics,
    size_t *minimum_cluster,
    double *minimum_purity,
    size_t *maximum_cluster,
    double *maximum_purity
)
{
    size_t cluster;

    *minimum_cluster = 0;
    *maximum_cluster = 0;
    *minimum_purity = 1.0;
    *maximum_purity = 0.0;

    for (cluster = 0; cluster < metrics->cluster_count; cluster++) {
        uint64_t size = metrics->cluster_totals[cluster];
        size_t first;
        size_t second;
        uint64_t first_count;
        uint64_t second_count;
        double purity;

        gs_eval_top_two_labels(
            metrics,
            cluster,
            &first,
            &first_count,
            &second,
            &second_count
        );
        purity = size == 0 ? 0.0 : (double)first_count / (double)size;
        if (cluster == 0 || purity < *minimum_purity) {
            *minimum_cluster = cluster;
            *minimum_purity = purity;
        }
        if (cluster == 0 || purity > *maximum_purity) {
            *maximum_cluster = cluster;
            *maximum_purity = purity;
        }
    }
}


static int gs_eval_write_requested_label(
    FILE *file,
    const char *name,
    size_t label,
    const GSBaselineClusteringMetrics *train,
    const GSBaselineClusteringMetrics *validation
)
{
    GSLabelStatistics train_statistics;
    GSLabelStatistics validation_statistics;

    gs_eval_label_statistics(train, label, &train_statistics);
    gs_eval_label_statistics(validation, label, &validation_statistics);

    return fprintf(
        file,
        "%s_train_total=%lu\n"
        "%s_train_dominant_cluster=%lu\n"
        "%s_train_dominant_fraction=%.17g\n"
        "%s_train_occupied_clusters=%lu\n"
        "%s_validation_total=%lu\n"
        "%s_validation_dominant_cluster=%lu\n"
        "%s_validation_dominant_fraction=%.17g\n"
        "%s_validation_occupied_clusters=%lu\n",
        name,
        (unsigned long)train_statistics.total,
        name,
        (unsigned long)train_statistics.dominant_cluster,
        name,
        train_statistics.dominant_fraction,
        name,
        (unsigned long)train_statistics.occupied_clusters,
        name,
        (unsigned long)validation_statistics.total,
        name,
        (unsigned long)validation_statistics.dominant_cluster,
        name,
        validation_statistics.dominant_fraction,
        name,
        (unsigned long)validation_statistics.occupied_clusters
    ) < 0 ? 1 : 0;
}


static int gs_eval_write_summary(
    const char *directory,
    size_t k,
    const char *code_commit,
    const GSBaselineClusteringMetrics *train,
    const GSBaselineClusteringMetrics *validation
)
{
    FILE *file = NULL;
    size_t largest;
    uint64_t largest_size;
    size_t minimum_cluster;
    size_t maximum_cluster;
    double minimum_purity;
    double maximum_purity;
    unsigned char selected[GS_EVAL_LABEL_COUNT] = {0};
    size_t rank;
    int failed = 0;

    if (gs_eval_open_output(directory, "summary.txt", &file) != 0) {
        return 1;
    }

    largest = gs_eval_largest_cluster(train);
    largest_size = train->cluster_totals[largest];
    gs_eval_purity_extremes(
        train,
        &minimum_cluster,
        &minimum_purity,
        &maximum_cluster,
        &maximum_purity
    );

    if (fprintf(
            file,
            "format_version=1\n"
            "profile=mini75_pool5_no_pca\n"
            "evaluation=labels_after_clustering\n"
            "label_count=22\n"
            "k=%lu\n"
            "seed=123\n"
            "train_rows=6866\n"
            "validation_rows=1717\n"
            "labels_used_for_training=false\n"
            "labels_used_for_clustering=false\n"
            "labels_used_for_evaluation=true\n"
            "nmi_normalization=arithmetic\n"
            "purity_definition=weighted_cluster_majority\n"
            "code_commit=%s\n"
            "train_purity=%.17g\n"
            "train_nmi_arithmetic=%.17g\n"
            "train_ari=%.17g\n"
            "train_homogeneity=%.17g\n"
            "train_completeness=%.17g\n"
            "validation_purity=%.17g\n"
            "validation_nmi_arithmetic=%.17g\n"
            "validation_ari=%.17g\n"
            "validation_homogeneity=%.17g\n"
            "validation_completeness=%.17g\n"
            "largest_train_cluster=%lu\n"
            "largest_train_cluster_size=%lu\n"
            "minimum_purity_train_cluster=%lu\n"
            "minimum_train_cluster_purity=%.17g\n"
            "maximum_purity_train_cluster=%lu\n"
            "maximum_train_cluster_purity=%.17g\n",
            (unsigned long)k,
            code_commit,
            train->purity,
            train->normalized_mutual_information,
            train->adjusted_rand_index,
            train->homogeneity,
            train->completeness,
            validation->purity,
            validation->normalized_mutual_information,
            validation->adjusted_rand_index,
            validation->homogeneity,
            validation->completeness,
            (unsigned long)largest,
            (unsigned long)largest_size,
            (unsigned long)minimum_cluster,
            minimum_purity,
            (unsigned long)maximum_cluster,
            maximum_purity
        ) < 0) {
        failed = 1;
    }

    for (rank = 0; !failed && rank < 5; rank++) {
        size_t label;
        size_t best = SIZE_MAX;
        uint64_t best_count = UINT64_C(0);

        for (label = 0; label < GS_EVAL_LABEL_COUNT; label++) {
            uint64_t count = train->contingency[
                label * train->cluster_count + largest
            ];
            if (!selected[label] &&
                (best == SIZE_MAX || count > best_count)) {
                best = label;
                best_count = count;
            }
        }
        selected[best] = 1;
        if (fprintf(
                file,
                "largest_train_cluster_top_%lu_label_index=%lu\n"
                "largest_train_cluster_top_%lu_label=%s\n"
                "largest_train_cluster_top_%lu_count=%lu\n"
                "largest_train_cluster_top_%lu_fraction=%.17g\n",
                (unsigned long)(rank + 1),
                (unsigned long)best,
                (unsigned long)(rank + 1),
                gs_eval_labels[best],
                (unsigned long)(rank + 1),
                (unsigned long)best_count,
                (unsigned long)(rank + 1),
                largest_size == 0
                    ? 0.0
                    : (double)best_count / (double)largest_size
            ) < 0) {
            failed = 1;
        }
    }

    if (!failed &&
        (gs_eval_write_requested_label(file, "Blip", 3, train, validation) ||
         gs_eval_write_requested_label(file, "Koi_Fish", 7, train, validation) ||
         gs_eval_write_requested_label(
             file,
             "Scattered_Light",
             16,
             train,
             validation
         ) ||
         gs_eval_write_requested_label(
             file,
             "None_of_the_Above",
             12,
             train,
             validation
         ))) {
        failed = 1;
    }

    return gs_eval_finish_output(file, failed);
}


static int gs_eval_write_all(
    const char *directory,
    size_t k,
    const char *code_commit,
    const GSBaselineClusteringMetrics *train,
    const GSBaselineClusteringMetrics *validation
)
{
    return gs_eval_write_label_map(directory) ||
        gs_eval_write_contingency(
            directory,
            "train_contingency_counts.csv",
            train
        ) ||
        gs_eval_write_contingency(
            directory,
            "val_contingency_counts.csv",
            validation
        ) ||
        gs_eval_write_cluster_summary(
            directory,
            "train_cluster_summary.csv",
            train
        ) ||
        gs_eval_write_cluster_summary(
            directory,
            "val_cluster_summary.csv",
            validation
        ) ||
        gs_eval_write_label_summary(
            directory,
            "train_label_summary.csv",
            train
        ) ||
        gs_eval_write_label_summary(
            directory,
            "val_label_summary.csv",
            validation
        ) ||
        gs_eval_write_metrics(directory, k, train, validation) ||
        gs_eval_write_summary(
            directory,
            k,
            code_commit,
            train,
            validation
        );
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
    uint32_t *train_assignments_copy = NULL;
    uint32_t *validation_assignments_copy = NULL;
    size_t train_bytes;
    size_t validation_bytes;
    size_t k;
    size_t index;
    char final_path[GS_EVAL_PATH_LEN];
    int rc = 1;

    memset(&manifest, 0, sizeof(manifest));
    memset(&train_metrics, 0, sizeof(train_metrics));
    memset(&validation_metrics, 0, sizeof(validation_metrics));

    if (argc != 9 || !gs_eval_platform_is_valid() ||
        gs_eval_parse_size(argv[6], &k) != 0 ||
        (k != 22 && k != 36) ||
        strcmp(argv[8], GS_EVAL_CODE_COMMIT) != 0 ||
        gs_eval_validate_output_paths(
            argv[7],
            final_path,
            sizeof(final_path)
        ) != 0 ||
        gs_eval_checked_product(
            GS_EVAL_TRAIN_ROWS,
            sizeof(uint32_t),
            &train_bytes
        ) != 0 ||
        gs_eval_checked_product(
            GS_EVAL_VALIDATION_ROWS,
            sizeof(uint32_t),
            &validation_bytes
        ) != 0) {
        fprintf(stderr, "Errore: argomenti o piattaforma non validi.\n");
        return 1;
    }

    coverage = (unsigned char *)calloc(
        GS_EVAL_MANIFEST_ROWS,
        sizeof(unsigned char)
    );
    train_labels = (uint32_t *)malloc(train_bytes);
    validation_labels = (uint32_t *)malloc(validation_bytes);
    train_assignments = (uint32_t *)malloc(train_bytes);
    validation_assignments = (uint32_t *)malloc(validation_bytes);
    train_assignments_copy = (uint32_t *)malloc(train_bytes);
    validation_assignments_copy = (uint32_t *)malloc(validation_bytes);

    if (coverage == NULL || train_labels == NULL ||
        validation_labels == NULL || train_assignments == NULL ||
        validation_assignments == NULL || train_assignments_copy == NULL ||
        validation_assignments_copy == NULL ||
        gs_event_manifest_load(argv[1], &manifest) != 0 ||
        gs_eval_validate_manifest(&manifest) != 0 ||
        gs_eval_load_rows(
            argv[2],
            "train",
            GS_EVAL_TRAIN_ROWS,
            &manifest,
            coverage,
            train_labels
        ) != 0 ||
        gs_eval_load_rows(
            argv[3],
            "validation",
            GS_EVAL_VALIDATION_ROWS,
            &manifest,
            coverage,
            validation_labels
        ) != 0) {
        fprintf(stderr, "Errore: manifest o row mapping non validi.\n");
        goto cleanup;
    }
    for (index = 0; index < GS_EVAL_MANIFEST_ROWS; index++) {
        if (!coverage[index]) {
            fprintf(stderr, "Errore: copertura manifest incompleta.\n");
            goto cleanup;
        }
    }
    printf("[PASS] manifest_labels_and_row_mapping\n");

    if (gs_eval_read_assignments(
            argv[4],
            train_assignments,
            GS_EVAL_TRAIN_ROWS,
            k
        ) != 0 ||
        gs_eval_read_assignments(
            argv[5],
            validation_assignments,
            GS_EVAL_VALIDATION_ROWS,
            k
        ) != 0) {
        fprintf(stderr, "Errore: assegnazioni non valide.\n");
        goto cleanup;
    }
    memcpy(train_assignments_copy, train_assignments, train_bytes);
    memcpy(validation_assignments_copy, validation_assignments,
           validation_bytes);
    printf("[PASS] existing_assignments\n");

    if (gs_baseline_clustering_metrics_compute(
            train_labels,
            train_assignments,
            GS_EVAL_TRAIN_ROWS,
            GS_EVAL_LABEL_COUNT,
            k,
            &train_metrics
        ) != GS_BASELINE_CLUSTERING_METRICS_SUCCESS ||
        !gs_eval_metrics_are_valid(
            &train_metrics,
            GS_EVAL_TRAIN_ROWS,
            k
        ) ||
        gs_baseline_clustering_metrics_compute(
            validation_labels,
            validation_assignments,
            GS_EVAL_VALIDATION_ROWS,
            GS_EVAL_LABEL_COUNT,
            k,
            &validation_metrics
        ) != GS_BASELINE_CLUSTERING_METRICS_SUCCESS ||
        !gs_eval_metrics_are_valid(
            &validation_metrics,
            GS_EVAL_VALIDATION_ROWS,
            k
        ) ||
        memcmp(train_assignments, train_assignments_copy, train_bytes) != 0 ||
        memcmp(
            validation_assignments,
            validation_assignments_copy,
            validation_bytes
        ) != 0) {
        fprintf(stderr, "Errore: calcolo metriche non valido.\n");
        goto cleanup;
    }
    printf("[PASS] posterior_metrics\n");

    if (gs_eval_create_directory(argv[7], final_path) != 0 ||
        gs_eval_write_all(
            argv[7],
            k,
            argv[8],
            &train_metrics,
            &validation_metrics
        ) != 0 ||
        memcmp(train_assignments, train_assignments_copy, train_bytes) != 0 ||
        memcmp(
            validation_assignments,
            validation_assignments_copy,
            validation_bytes
        ) != 0) {
        fprintf(stderr, "Errore: scrittura valutazione fallita.\n");
        goto cleanup;
    }
    printf("[PASS] evaluation_artifacts_written\n");
    printf("K=%lu\n", (unsigned long)k);
    printf("TRAIN_PURITY=%.17g\n", train_metrics.purity);
    printf(
        "TRAIN_NMI_ARITHMETIC=%.17g\n",
        train_metrics.normalized_mutual_information
    );
    printf("TRAIN_ARI=%.17g\n", train_metrics.adjusted_rand_index);
    printf("VALIDATION_PURITY=%.17g\n", validation_metrics.purity);
    printf(
        "VALIDATION_NMI_ARITHMETIC=%.17g\n",
        validation_metrics.normalized_mutual_information
    );
    printf(
        "VALIDATION_ARI=%.17g\n",
        validation_metrics.adjusted_rand_index
    );
    printf("LABEL EVALUATION COMPLETED\n");
    rc = 0;

cleanup:
    gs_baseline_clustering_metrics_free(&validation_metrics);
    gs_baseline_clustering_metrics_free(&train_metrics);
    gs_event_manifest_free(&manifest);
    free(validation_assignments_copy);
    free(train_assignments_copy);
    free(validation_assignments);
    free(train_assignments);
    free(validation_labels);
    free(train_labels);
    free(coverage);
    return rc;
}
