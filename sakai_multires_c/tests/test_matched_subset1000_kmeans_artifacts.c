#define GS_MATCHED_KMEANS_NO_MAIN
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "../tools/run_matched_subset1000_kmeans.c"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

typedef struct {
    size_t iterations;
    int converged;
    size_t reseeds;
    double initial_inertia;
    double train_inertia;
    double validation_inertia;
    size_t empty_clusters;
    size_t minimum_cluster_size;
    size_t maximum_cluster_size;
} ArtifactSummary;

static int artifact_regular_non_reparse(const char *path)
{
    struct stat information;
    if (path == NULL || stat(path, &information) != 0 ||
        !S_ISREG(information.st_mode)) {
        return 0;
    }
#ifdef _WIN32
    {
        DWORD attributes = GetFileAttributesA(path);
        if (attributes == INVALID_FILE_ATTRIBUTES ||
            (attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0u) {
            return 0;
        }
    }
#endif
    return 1;
}

static int artifact_directory_non_reparse(const char *path)
{
    struct stat information;
    if (path == NULL || stat(path, &information) != 0 ||
        !S_ISDIR(information.st_mode)) {
        return 0;
    }
#ifdef _WIN32
    {
        DWORD attributes = GetFileAttributesA(path);
        if (attributes == INVALID_FILE_ATTRIBUTES ||
            (attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0u) {
            return 0;
        }
    }
#endif
    return 1;
}

static int artifact_base_inventory_downstream_aware(const char *directory)
{
    unsigned char seen[KM_BASE_FILE_COUNT] = {0};
    DIR *stream = NULL;
    struct dirent *entry;
    size_t index;
    int kmeans_seen = 0;
    int evaluation_seen = 0;
    int failed = 0;

    stream = opendir(directory);
    if (stream == NULL) {
        return 1;
    }
    while (!failed) {
        char path[GS_PATH_CAPACITY];
        size_t matched;

        errno = 0;
        entry = readdir(stream);
        if (entry == NULL) {
            if (errno != 0) {
                failed = 1;
            }
            break;
        }
        if (strcmp(entry->d_name, ".") == 0 ||
            strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        if (gs_path_join(
                directory, entry->d_name, path, sizeof(path)
            ) != 0) {
            failed = 1;
        } else if (km_name_index(
                       entry->d_name,
                       km_base_names,
                       KM_BASE_FILE_COUNT,
                       &matched
                   )) {
            if (seen[matched] ||
                !artifact_regular_non_reparse(path)) {
                failed = 1;
            } else {
                seen[matched] = 1u;
            }
        } else if (strcmp(entry->d_name, "kmeans_seed123") == 0 &&
                   !kmeans_seen &&
                   artifact_directory_non_reparse(path)) {
            kmeans_seen = 1;
        } else if (strcmp(
                       entry->d_name, "label_evaluation_seed123"
                   ) == 0 &&
                   !evaluation_seen &&
                   artifact_directory_non_reparse(path)) {
            evaluation_seen = 1;
        } else {
            failed = 1;
        }
    }
    if (closedir(stream) != 0) {
        failed = 1;
    }
    for (index = 0u; !failed && index < KM_BASE_FILE_COUNT; index++) {
        if (!seen[index]) {
            failed = 1;
        }
    }
    if (!failed && !kmeans_seen) {
        failed = 1;
    }
    return failed;
}

static int artifact_resolve_base_downstream_aware(
    const char *argument,
    const char *token,
    KMBase *base
)
{
    static const char prefix[] =
        "/sakai_multires_c/results/baseline/matched_subset1000/";
    char suffix[GS_PATH_CAPACITY];
    char expected[GS_PATH_CAPACITY];
    char canonical[GS_PATH_CAPACITY];
    size_t base_length;
    size_t suffix_length;
    size_t root_length;
    int written;

    if (base == NULL ||
        (base->profile = gs_profile_from_token(token)) == NULL ||
        gs_canonical_existing(argument, 1, base->base) != 0) {
        return 1;
    }
    written = snprintf(
        suffix, sizeof(suffix), "%s%s",
        prefix, base->profile->matched_basename
    );
    if (written < 0 || (size_t)written >= sizeof(suffix)) {
        return 1;
    }
    base_length = strlen(base->base);
    suffix_length = strlen(suffix);
    if (base_length <= suffix_length) {
        return 1;
    }
    root_length = base_length - suffix_length;
#ifdef _WIN32
    if (_stricmp(base->base + root_length, suffix) != 0) {
#else
    if (strcmp(base->base + root_length, suffix) != 0) {
#endif
        return 1;
    }
    if (root_length == 0u || root_length >= sizeof(base->root)) {
        return 1;
    }
    memcpy(base->root, base->base, root_length);
    base->root[root_length] = '\0';
    if (gs_canonical_existing(base->root, 1, canonical) != 0 ||
        !gs_paths_equal(base->root, canonical) ||
        gs_path_join(
            base->root,
            "sakai_multires_c/results/baseline/matched_subset1000",
            expected, sizeof(expected)
        ) != 0 ||
        gs_canonical_existing(expected, 1, canonical) != 0 ||
        gs_path_join(
            canonical, base->profile->matched_basename,
            expected, sizeof(expected)
        ) != 0 ||
        gs_canonical_existing(expected, 1, canonical) != 0 ||
        !gs_paths_equal(base->base, canonical) ||
        gs_path_join(
            base->root, "sakai_multires_c/results/baseline",
            expected, sizeof(expected)
        ) != 0 ||
        gs_canonical_existing(expected, 1, canonical) != 0 ||
        gs_path_join(
            canonical, base->profile->source_basename,
            expected, sizeof(expected)
        ) != 0 ||
        gs_canonical_existing(expected, 1, base->source) != 0 ||
        gs_path_join(
            base->root,
            "sakai_multires_c/configs/subset224_train_800_seed123.csv",
            expected, sizeof(expected)
        ) != 0 ||
        gs_canonical_existing(expected, 0, base->subset_train) != 0 ||
        gs_path_join(
            base->root,
            "sakai_multires_c/configs/subset224_val_200_seed123.csv",
            expected, sizeof(expected)
        ) != 0 ||
        gs_canonical_existing(
            expected, 0, base->subset_validation
        ) != 0) {
        return 1;
    }
    return artifact_base_inventory_downstream_aware(base->base);
}

static int artifact_prepare_base_downstream_aware(
    const char *directory,
    const char *token,
    KMBase *base
)
{
    double *mean = NULL;
    double *stddev = NULL;
    unsigned char *constant = NULL;
    GSStatistics statistics;
    char path[GS_PATH_CAPACITY];
    size_t index;
    int failed = 0;

    if (base == NULL) {
        return 1;
    }
    memset(base, 0, sizeof(*base));
    memset(&statistics, 0, sizeof(statistics));
    if (artifact_resolve_base_downstream_aware(
            directory, token, base
        ) != 0 ||
        km_validate_base_checksum(base) != 0 ||
        km_validate_metadata(base) != 0 ||
        km_validate_rows(base) != 0) {
        return 1;
    }
    for (index = 0u; index < KM_INPUT_COUNT; index++) {
        if (gs_path_join(
                base->base, km_input_names[index],
                base->input_storage[index], GS_PATH_CAPACITY
            ) != 0 ||
            gs_canonical_existing(
                base->input_storage[index], 0, path
            ) != 0 ||
            !gs_paths_equal(base->input_storage[index], path) ||
            gs_sha_file(
                base->input_storage[index],
                base->input_hashes[index]
            ) != 0) {
            return 1;
        }
        base->input_paths[index] = base->input_storage[index];
    }
    mean = (double *)malloc(base->feature_count * sizeof(*mean));
    stddev = (double *)malloc(base->feature_count * sizeof(*stddev));
    constant = (unsigned char *)malloc(
        base->feature_count * sizeof(*constant)
    );
    if (mean == NULL || stddev == NULL || constant == NULL ||
        km_load_scaler(base, mean, stddev, constant) != 0 ||
        km_read_matrix(
            base->input_paths[2], base->train_rows,
            base->feature_count, &base->train
        ) != 0 ||
        km_read_matrix(
            base->input_paths[3], base->validation_rows,
            base->feature_count, &base->validation
        ) != 0 ||
        gs_validate_standardized(
            base->train, base->validation,
            mean, stddev, constant,
            base->feature_count, &statistics
        ) != 0 ||
        statistics.constant_count != 0u) {
        failed = 1;
    }
    free(constant);
    free(stddev);
    free(mean);
    if (failed) {
        km_base_free(base);
    }
    return failed;
}

static int artifact_resolve_final(
    const KMBase *base,
    const char *argument,
    size_t k,
    char final[GS_PATH_CAPACITY]
)
{
    char canonical[GS_PATH_CAPACITY];
    char expected_parent[GS_PATH_CAPACITY];
    char canonical_expected_parent[GS_PATH_CAPACITY];
    char expected[GS_PATH_CAPACITY];
    char partial[GS_PATH_CAPACITY];
    char name[16];
    char partial_name[32];
    int written;

    written = snprintf(name, sizeof(name), "k%zu", k);
    if (written < 0 || (size_t)written >= sizeof(name) ||
        gs_canonical_existing(argument, 1, canonical) != 0 ||
        strcmp(gs_basename(canonical), name) != 0 ||
        gs_path_join(
            base->base, "kmeans_seed123",
            expected_parent, sizeof(expected_parent)
        ) != 0 ||
        gs_canonical_existing(
            expected_parent, 1, canonical_expected_parent
        ) != 0 ||
        gs_path_join(
            canonical_expected_parent, name, expected, sizeof(expected)
        ) != 0 ||
        gs_canonical_existing(expected, 1, final) != 0 ||
        !gs_paths_equal(canonical, final)) {
        return 1;
    }
    written = snprintf(
        partial_name, sizeof(partial_name), "k%zu.partial", k
    );
    if (written < 0 || (size_t)written >= sizeof(partial_name) ||
        gs_path_join(
            canonical_expected_parent,
            partial_name, partial, sizeof(partial)
        ) != 0 ||
        gs_path_state(partial) != 0) {
        return 1;
    }
    return 0;
}

static int artifact_read_binary(
    const char *directory,
    const char *name,
    void *values,
    size_t element_size,
    size_t count
)
{
    char path[GS_PATH_CAPACITY];
    size_t bytes;
    size_t actual;
    FILE *file = NULL;

    if (values == NULL || element_size == 0u ||
        gs_checked_product(element_size, count, &bytes) != 0 ||
        gs_path_join(directory, name, path, sizeof(path)) != 0 ||
        gs_file_size(path, &actual) != 0 || actual != bytes ||
        (file = fopen(path, "rb")) == NULL) {
        return 1;
    }
    {
        int failed = 0;
        if (fread(values, element_size, count, file) != count ||
            fgetc(file) != EOF || ferror(file)) {
            failed = 1;
        }
        if (fclose(file) != 0) {
            failed = 1;
        }
        if (failed) {
            return 1;
        }
    }
    return 0;
}

static int artifact_parse_double(
    const char *path,
    const char *key,
    double *value
)
{
    char text[128];
    char *end = NULL;
    if (km_file_value(path, key, text, sizeof(text)) != 0) {
        return 1;
    }
    errno = 0;
    *value = strtod(text, &end);
    return errno == ERANGE || end == text || *end != '\0' ||
        !isfinite(*value) || *value < 0.0;
}

static int artifact_load_summary(
    const char *directory,
    const KMBase *base,
    size_t k,
    const char *commit,
    ArtifactSummary *summary
)
{
    char path[GS_PATH_CAPACITY];
    char value[128];
    char expected[128];
    size_t parsed;

    if (gs_path_join(
            directory, "summary.txt", path, sizeof(path)
        ) != 0 ||
        snprintf(
            expected, sizeof(expected),
            "matched_subset1000_%s_no_pca_kmeans",
            base->profile->token
        ) < 0 ||
        !gs_file_has_key_value(path, "format_version", "1") ||
        !gs_file_has_key_value(path, "profile", expected) ||
        !gs_file_has_key_value(
            path, "input_profile", base->profile->matched_profile
        ) ||
        !gs_file_has_key_value(
            path, "matched_subset_profile",
            "subset224_seed123_prefix_800_200"
        ) ||
        !gs_file_has_key_value(
            path, "source_profile", base->profile->source_profile
        ) ||
        !gs_file_has_key_value(
            path, "algorithm", "kmeans_plus_plus_lloyd"
        ) ||
        !gs_file_has_key_value(
            path, "input", "train_features_std_f32"
        ) ||
        !gs_file_has_key_value(
            path, "validation_policy", "assign_to_train_centroids"
        ) ||
        !gs_file_has_key_value(path, "train_rows", "800") ||
        !gs_file_has_key_value(path, "validation_rows", "200") ||
        !gs_file_has_key_value(path, "seed", "123") ||
        !gs_file_has_key_value(path, "max_iterations", "300") ||
        !gs_file_has_key_value(
            path, "standardization",
            "train_only_population_ddof0"
        ) ||
        !gs_file_has_key_value(path, "fit_input", "train_only") ||
        !gs_file_has_key_value(
            path, "validation_assigned_after_fit", "true"
        ) ||
        !gs_file_has_key_value(path, "pca", "false") ||
        !gs_file_has_key_value(path, "whitening", "false") ||
        !gs_file_has_key_value(path, "augmentation", "false") ||
        !gs_file_has_key_value(path, "labels_used", "false") ||
        !gs_file_has_key_value(
            path, "matched_base_code_commit",
            base->matched_base_commit
        ) ||
        !gs_file_has_key_value(path, "code_commit", commit)) {
        return 1;
    }
    if (km_file_value(path, "feature_count", value, sizeof(value)) != 0 ||
        km_parse_size(value, &parsed) != 0 ||
        parsed != base->feature_count ||
        km_file_value(path, "k", value, sizeof(value)) != 0 ||
        km_parse_size(value, &parsed) != 0 || parsed != k ||
        km_file_value(path, "iterations", value, sizeof(value)) != 0 ||
        km_parse_size(value, &summary->iterations) != 0 ||
        km_file_value(path, "converged", value, sizeof(value)) != 0 ||
        km_parse_size(value, &parsed) != 0 || parsed > 1u ||
        km_file_value(
            path, "empty_cluster_reseeds", value, sizeof(value)
        ) != 0 ||
        km_parse_size(value, &summary->reseeds) != 0 ||
        km_file_value(
            path, "empty_final_clusters", value, sizeof(value)
        ) != 0 ||
        km_parse_size(value, &summary->empty_clusters) != 0 ||
        km_file_value(
            path, "minimum_cluster_size", value, sizeof(value)
        ) != 0 ||
        km_parse_size(value, &summary->minimum_cluster_size) != 0 ||
        km_file_value(
            path, "maximum_cluster_size", value, sizeof(value)
        ) != 0 ||
        km_parse_size(value, &summary->maximum_cluster_size) != 0 ||
        artifact_parse_double(
            path, "initial_inertia", &summary->initial_inertia
        ) != 0 ||
        artifact_parse_double(
            path, "final_train_inertia", &summary->train_inertia
        ) != 0 ||
        artifact_parse_double(
            path, "validation_inertia", &summary->validation_inertia
        ) != 0) {
        return 1;
    }
    summary->converged = (int)parsed;
    return summary->converged != 1 ||
        summary->iterations == 0u ||
        summary->iterations > KM_MAX_ITERATIONS;
}

static int artifact_validate_csv(
    const char *directory,
    const char *name,
    const char *header,
    size_t data_rows
)
{
    char path[GS_PATH_CAPACITY];
    char line[GS_LINE_CAPACITY];
    FILE *file = NULL;
    size_t rows = 0u;

    if (gs_path_join(directory, name, path, sizeof(path)) != 0 ||
        (file = fopen(path, "rb")) == NULL ||
        fgets(line, sizeof(line), file) == NULL) {
        if (file != NULL) {
            fclose(file);
        }
        return 1;
    }
    gs_chomp(line);
    if (strcmp(line, header) != 0) {
        fclose(file);
        return 1;
    }
    while (fgets(line, sizeof(line), file) != NULL) {
        if (strchr(line, '\n') == NULL && !feof(file)) {
            fclose(file);
            return 1;
        }
        rows++;
    }
    if (ferror(file) || fclose(file) != 0 || rows != data_rows) {
        return 1;
    }
    return 0;
}

static int artifact_assignment_csv_matches(
    const char *directory,
    const char *name,
    const uint32_t *assignments,
    const double *distances,
    size_t rows
)
{
    char path[GS_PATH_CAPACITY];
    char line[GS_LINE_CAPACITY];
    FILE *file = NULL;
    size_t row;

    if (assignments == NULL || distances == NULL ||
        gs_path_join(directory, name, path, sizeof(path)) != 0 ||
        (file = fopen(path, "rb")) == NULL ||
        fgets(line, sizeof(line), file) == NULL) {
        if (file != NULL) {
            fclose(file);
        }
        return 1;
    }
    gs_chomp(line);
    if (strcmp(line, "row_index,cluster_id,squared_distance") != 0) {
        fclose(file);
        return 1;
    }
    for (row = 0u; row < rows; row++) {
        char *fields[3];
        size_t count;
        size_t parsed_row;
        size_t parsed_cluster;
        char *end = NULL;
        double parsed_distance;
        if (fgets(line, sizeof(line), file) == NULL) {
            fclose(file);
            return 1;
        }
        gs_chomp(line);
        if (gs_csv_fields(line, fields, 3u, &count) != 0 || count != 3u ||
            km_parse_size(fields[0], &parsed_row) != 0 ||
            km_parse_size(fields[1], &parsed_cluster) != 0 ||
            parsed_row != row || parsed_cluster != assignments[row]) {
            fclose(file);
            return 1;
        }
        errno = 0;
        parsed_distance = strtod(fields[2], &end);
        if (errno == ERANGE || end == fields[2] || *end != '\0' ||
            !isfinite(parsed_distance) || parsed_distance < 0.0 ||
            memcmp(
                &parsed_distance, &distances[row], sizeof(double)
            ) != 0) {
            fclose(file);
            return 1;
        }
    }
    {
        int extra = fgets(line, sizeof(line), file) != NULL;
        int read_failed = ferror(file);
        int close_failed = fclose(file) != 0;
        return extra || read_failed || close_failed;
    }
}

static int artifact_u64_csv_matches(
    const char *directory,
    const char *name,
    const uint64_t *values,
    size_t count,
    const char *header
)
{
    char path[GS_PATH_CAPACITY];
    char line[GS_LINE_CAPACITY];
    FILE *file = NULL;
    size_t index;

    if (values == NULL ||
        gs_path_join(directory, name, path, sizeof(path)) != 0 ||
        (file = fopen(path, "rb")) == NULL ||
        fgets(line, sizeof(line), file) == NULL) {
        if (file != NULL) {
            fclose(file);
        }
        return 1;
    }
    gs_chomp(line);
    if (strcmp(line, header) != 0) {
        fclose(file);
        return 1;
    }
    for (index = 0u; index < count; index++) {
        char *fields[2];
        size_t field_count;
        size_t parsed_index;
        uint64_t parsed_value;
        if (fgets(line, sizeof(line), file) == NULL) {
            fclose(file);
            return 1;
        }
        gs_chomp(line);
        if (gs_csv_fields(line, fields, 2u, &field_count) != 0 ||
            field_count != 2u ||
            km_parse_size(fields[0], &parsed_index) != 0 ||
            km_parse_u64(fields[1], &parsed_value) != 0 ||
            parsed_index != index || parsed_value != values[index]) {
            fclose(file);
            return 1;
        }
    }
    {
        int extra = fgets(line, sizeof(line), file) != NULL;
        int read_failed = ferror(file);
        int close_failed = fclose(file) != 0;
        return extra || read_failed || close_failed;
    }
}

static int artifact_centroids_from_train(
    const KMBase *base,
    size_t k,
    const uint32_t *assignments,
    const uint64_t *sizes,
    double *centroids
)
{
    size_t centroid_count;
    size_t row;
    size_t cluster;

    if (gs_checked_product(k, base->feature_count, &centroid_count) != 0) {
        return 1;
    }
    memset(centroids, 0, centroid_count * sizeof(*centroids));
    for (row = 0u; row < base->train_rows; row++) {
        size_t column;
        cluster = assignments[row];
        if (cluster >= k) {
            return 1;
        }
        for (column = 0u; column < base->feature_count; column++) {
            centroids[cluster * base->feature_count + column] +=
                (double)base->train[row * base->feature_count + column];
        }
    }
    for (cluster = 0u; cluster < k; cluster++) {
        size_t column;
        if (sizes[cluster] == 0u) {
            return 1;
        }
        for (column = 0u; column < base->feature_count; column++) {
            centroids[cluster * base->feature_count + column] /=
                (double)sizes[cluster];
        }
    }
    return 0;
}

int main(int argc, char **argv)
{
    KMBase base;
    ArtifactSummary summary;
    GSBaselineKMeansResult refit;
    size_t k;
    size_t centroid_count;
    size_t centroid_bytes;
    size_t train_size_t_bytes;
    size_t validation_size_t_bytes;
    size_t train_distance_bytes;
    size_t validation_distance_bytes;
    char final[GS_PATH_CAPACITY];
    double *centroids = NULL;
    double *centroids_rebuilt = NULL;
    double *centroids_copy = NULL;
    double *train_distances = NULL;
    double *validation_distances = NULL;
    double *train_distances_recomputed = NULL;
    double *validation_distances_recomputed = NULL;
    uint32_t *train_u32 = NULL;
    uint32_t *validation_u32 = NULL;
    uint64_t *sizes = NULL;
    uint64_t *indices = NULL;
    uint64_t *sizes_recomputed = NULL;
    size_t *train_assignments = NULL;
    size_t *validation_assignments = NULL;
    size_t *refit_validation_assignments = NULL;
    double *refit_validation_distances = NULL;
    double train_inertia = 0.0;
    double validation_inertia = 0.0;
    size_t row;
    size_t cluster;
    size_t empty = 0u;
    size_t minimum;
    size_t maximum = 0u;
    int rc = 1;

    memset(&base, 0, sizeof(base));
    memset(&summary, 0, sizeof(summary));
    memset(&refit, 0, sizeof(refit));
    if (setlocale(LC_ALL, "C") == NULL || argc != 6 ||
        km_parse_size(argv[4], &k) != 0 ||
        (k != 22u && k != 36u) ||
        !km_commit_valid(argv[5]) ||
        artifact_prepare_base_downstream_aware(
            argv[1], argv[2], &base
        ) != 0 ||
        artifact_resolve_final(&base, argv[3], k, final) != 0 ||
        km_output_inventory_valid(final) != 0 ||
        km_verify_checksum(final, &base) != 0 ||
        km_validate_output_sizes(final, &base, k) != 0 ||
        artifact_load_summary(
            final, &base, k, argv[5], &summary
        ) != 0 ||
        artifact_validate_csv(
            final, "train_assignments.csv",
            "row_index,cluster_id,squared_distance",
            base.train_rows
        ) != 0 ||
        artifact_validate_csv(
            final, "val_assignments.csv",
            "row_index,cluster_id,squared_distance",
            base.validation_rows
        ) != 0 ||
        artifact_validate_csv(
            final, "cluster_sizes.csv",
            "cluster_id,train_count", k
        ) != 0 ||
        artifact_validate_csv(
            final, "initial_centroids.csv",
            "centroid_order,train_row_index", k
        ) != 0 ||
        gs_checked_product(k, base.feature_count, &centroid_count) != 0 ||
        gs_checked_product(
            centroid_count, sizeof(double), &centroid_bytes
        ) != 0 ||
        gs_checked_product(
            base.train_rows, sizeof(size_t), &train_size_t_bytes
        ) != 0 ||
        gs_checked_product(
            base.validation_rows, sizeof(size_t),
            &validation_size_t_bytes
        ) != 0 ||
        gs_checked_product(
            base.train_rows, sizeof(double), &train_distance_bytes
        ) != 0 ||
        gs_checked_product(
            base.validation_rows, sizeof(double),
            &validation_distance_bytes
        ) != 0) {
        fprintf(
            stderr,
            "Usage: test_matched_subset1000_kmeans_artifacts.exe "
            "<matched_base_directory> "
            "<mini75_pool5|mini112_pool7> "
            "<kmeans_final_directory> <22|36> "
            "<expected_code_commit>\n"
        );
        goto cleanup;
    }

#define ALLOCATE(name, type, count) \
    do { \
        name = (type *)malloc((count) * sizeof(type)); \
        if (name == NULL) goto cleanup; \
    } while (0)
    ALLOCATE(centroids, double, centroid_count);
    ALLOCATE(centroids_rebuilt, double, centroid_count);
    ALLOCATE(centroids_copy, double, centroid_count);
    ALLOCATE(train_distances, double, base.train_rows);
    ALLOCATE(validation_distances, double, base.validation_rows);
    ALLOCATE(train_distances_recomputed, double, base.train_rows);
    ALLOCATE(validation_distances_recomputed, double, base.validation_rows);
    ALLOCATE(train_u32, uint32_t, base.train_rows);
    ALLOCATE(validation_u32, uint32_t, base.validation_rows);
    ALLOCATE(sizes, uint64_t, k);
    ALLOCATE(indices, uint64_t, k);
    ALLOCATE(sizes_recomputed, uint64_t, k);
    ALLOCATE(train_assignments, size_t, base.train_rows);
    ALLOCATE(validation_assignments, size_t, base.validation_rows);
    ALLOCATE(refit_validation_assignments, size_t, base.validation_rows);
    ALLOCATE(refit_validation_distances, double, base.validation_rows);
#undef ALLOCATE

    if (artifact_read_binary(
            final, "centroids_f64.bin",
            centroids, sizeof(double), centroid_count
        ) != 0 ||
        artifact_read_binary(
            final, "train_assignments_u32.bin",
            train_u32, sizeof(uint32_t), base.train_rows
        ) != 0 ||
        artifact_read_binary(
            final, "val_assignments_u32.bin",
            validation_u32, sizeof(uint32_t), base.validation_rows
        ) != 0 ||
        artifact_read_binary(
            final, "train_squared_distances_f64.bin",
            train_distances, sizeof(double), base.train_rows
        ) != 0 ||
        artifact_read_binary(
            final, "val_squared_distances_f64.bin",
            validation_distances, sizeof(double), base.validation_rows
        ) != 0 ||
        artifact_read_binary(
            final, "cluster_sizes_train_u64.bin",
            sizes, sizeof(uint64_t), k
        ) != 0 ||
        artifact_read_binary(
            final, "initial_centroid_indices_u64.bin",
            indices, sizeof(uint64_t), k
        ) != 0) {
        fprintf(stderr, "ERROR: binary artifact read failed.\n");
        goto cleanup;
    }
    if (artifact_assignment_csv_matches(
            final, "train_assignments.csv",
            train_u32, train_distances, base.train_rows
        ) != 0 ||
        artifact_assignment_csv_matches(
            final, "val_assignments.csv",
            validation_u32, validation_distances, base.validation_rows
        ) != 0 ||
        artifact_u64_csv_matches(
            final, "cluster_sizes.csv", sizes, k,
            "cluster_id,train_count"
        ) != 0 ||
        artifact_u64_csv_matches(
            final, "initial_centroids.csv", indices, k,
            "centroid_order,train_row_index"
        ) != 0) {
        fprintf(stderr, "ERROR: CSV/binary semantic mismatch.\n");
        goto cleanup;
    }
    memset(sizes_recomputed, 0, k * sizeof(*sizes_recomputed));
    for (row = 0u; row < base.train_rows; row++) {
        if (train_u32[row] >= k) {
            goto cleanup;
        }
        train_assignments[row] = (size_t)train_u32[row];
        sizes_recomputed[train_u32[row]]++;
    }
    for (row = 0u; row < base.validation_rows; row++) {
        if (validation_u32[row] >= k) {
            goto cleanup;
        }
        validation_assignments[row] = (size_t)validation_u32[row];
    }
    for (cluster = 0u; cluster < k; cluster++) {
        size_t inner;
        if (indices[cluster] >= base.train_rows) {
            goto cleanup;
        }
        for (inner = 0u; inner < cluster; inner++) {
            if (indices[inner] == indices[cluster]) {
                goto cleanup;
            }
        }
    }
    if (gs_baseline_kmeans_assign(
            base.train, base.train_rows, base.feature_count,
            centroids, k, train_assignments,
            train_distances_recomputed
        ) != GS_BASELINE_KMEANS_SUCCESS ||
        gs_baseline_kmeans_assign(
            base.validation, base.validation_rows, base.feature_count,
            centroids, k, validation_assignments,
            validation_distances_recomputed
        ) != GS_BASELINE_KMEANS_SUCCESS ||
        memcmp(
            train_distances, train_distances_recomputed,
            train_distance_bytes
        ) != 0 ||
        memcmp(
            validation_distances, validation_distances_recomputed,
            validation_distance_bytes
        ) != 0 ||
        memcmp(sizes, sizes_recomputed, k * sizeof(*sizes)) != 0 ||
        artifact_centroids_from_train(
            &base, k, train_u32, sizes, centroids_rebuilt
        ) != 0 ||
        memcmp(centroids, centroids_rebuilt, centroid_bytes) != 0) {
        fprintf(stderr, "ERROR: recomputation mismatch.\n");
        goto cleanup;
    }
    for (row = 0u; row < base.train_rows; row++) {
        if ((size_t)train_u32[row] != train_assignments[row]) {
            goto cleanup;
        }
        train_inertia += train_distances_recomputed[row];
    }
    for (row = 0u; row < base.validation_rows; row++) {
        if ((size_t)validation_u32[row] != validation_assignments[row]) {
            goto cleanup;
        }
        validation_inertia += validation_distances_recomputed[row];
    }
    minimum = base.train_rows;
    for (cluster = 0u; cluster < k; cluster++) {
        if (sizes[cluster] == 0u) {
            empty++;
        }
        if (sizes[cluster] < minimum) {
            minimum = (size_t)sizes[cluster];
        }
        if (sizes[cluster] > maximum) {
            maximum = (size_t)sizes[cluster];
        }
    }
    if (memcmp(
            &train_inertia, &summary.train_inertia, sizeof(double)
        ) != 0 ||
        memcmp(
            &validation_inertia,
            &summary.validation_inertia,
            sizeof(double)
        ) != 0 ||
        empty != summary.empty_clusters ||
        minimum != summary.minimum_cluster_size ||
        maximum != summary.maximum_cluster_size) {
        fprintf(stderr, "ERROR: summary numeric mismatch.\n");
        goto cleanup;
    }

    if (gs_baseline_kmeans_fit(
            base.train, base.train_rows, base.feature_count,
            k, KM_SEED, KM_MAX_ITERATIONS, &refit
        ) != GS_BASELINE_KMEANS_SUCCESS ||
        refit.converged != 1 ||
        memcmp(refit.centroids, centroids, centroid_bytes) != 0 ||
        memcmp(
            refit.assignments, train_assignments, train_size_t_bytes
        ) != 0 ||
        refit.iterations != summary.iterations ||
        refit.converged != summary.converged ||
        refit.empty_cluster_reseeds != summary.reseeds ||
        memcmp(
            &refit.initial_inertia,
            &summary.initial_inertia,
            sizeof(double)
        ) != 0 ||
        memcmp(
            &refit.inertia,
            &summary.train_inertia,
            sizeof(double)
        ) != 0) {
        fprintf(stderr, "ERROR: full train-only refit mismatch.\n");
        goto cleanup;
    }
    for (cluster = 0u; cluster < k; cluster++) {
        if ((uint64_t)refit.cluster_sizes[cluster] != sizes[cluster] ||
            (uint64_t)refit.initial_centroid_indices[cluster] !=
                indices[cluster]) {
            goto cleanup;
        }
    }
    memcpy(centroids_copy, refit.centroids, centroid_bytes);
    if (gs_baseline_kmeans_assign(
            base.validation, base.validation_rows, base.feature_count,
            refit.centroids, k, refit_validation_assignments,
            refit_validation_distances
        ) != GS_BASELINE_KMEANS_SUCCESS ||
        memcmp(
            centroids_copy, refit.centroids, centroid_bytes
        ) != 0 ||
        memcmp(
            refit_validation_assignments,
            validation_assignments,
            validation_size_t_bytes
        ) != 0 ||
        memcmp(
            refit_validation_distances,
            validation_distances,
            validation_distance_bytes
        ) != 0) {
        fprintf(stderr, "ERROR: refit validation mismatch.\n");
        goto cleanup;
    }

    printf("TRAIN_ASSIGNMENTS_RECOMPUTED_MATCH=true\n");
    printf("VALIDATION_ASSIGNMENTS_RECOMPUTED_MATCH=true\n");
    printf("TRAIN_DISTANCES_RECOMPUTED_MATCH=true\n");
    printf("VALIDATION_DISTANCES_RECOMPUTED_MATCH=true\n");
    printf("CENTROIDS_RECOMPUTED_FROM_TRAIN_MATCH=true\n");
    printf("TRAIN_INERTIA_RECOMPUTED_MATCH=true\n");
    printf("VALIDATION_INERTIA_RECOMPUTED_MATCH=true\n");
    printf("CLUSTER_SIZES_RECOMPUTED_MATCH=true\n");
    printf("INITIAL_INDICES_VALID=true\n");
    printf("FULL_TRAIN_ONLY_REFIT_BITWISE=true\n");
    printf("VALIDATION_CENTROIDS_UNCHANGED=true\n");
    printf("ALL_MATCHED_KMEANS_CHECKSUMS_MATCH=true\n");
    printf("ALL MATCHED SUBSET1000 KMEANS ARTIFACT TESTS PASSED\n");
    rc = 0;

cleanup:
    gs_baseline_kmeans_result_free(&refit);
    free(refit_validation_distances);
    free(refit_validation_assignments);
    free(validation_assignments);
    free(train_assignments);
    free(sizes_recomputed);
    free(indices);
    free(sizes);
    free(validation_u32);
    free(train_u32);
    free(validation_distances_recomputed);
    free(train_distances_recomputed);
    free(validation_distances);
    free(train_distances);
    free(centroids_copy);
    free(centroids_rebuilt);
    free(centroids);
    km_base_free(&base);
    return rc;
}
