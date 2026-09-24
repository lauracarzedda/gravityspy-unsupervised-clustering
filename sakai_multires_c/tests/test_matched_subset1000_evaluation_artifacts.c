#define main gs_historical_mini224_evaluation_validator_main
#include "test_baseline_mini224_subset1000_evaluation_artifacts.c"
#undef main

#define GS_MATCHED_EVALUATION_NO_MAIN
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "../tools/evaluate_matched_subset1000_clusters.c"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

static int mv_parse_double_value(
    const char *directory,
    const char *key,
    double *value
)
{
    char text[128];
    char *end = NULL;
    if (value == NULL ||
        me_read_value(
            directory, "summary.txt", key, text, sizeof(text)
        ) != 0) {
        return 1;
    }
    errno = 0;
    *value = strtod(text, &end);
    return errno == ERANGE || end == text || *end != '\0' ||
        !isfinite(*value);
}

static int mv_metric_matches(
    const char *directory,
    const char *key,
    double expected
)
{
    double actual;
    return mv_parse_double_value(directory, key, &actual) == 0 &&
           gs_test_close(actual, expected) ? 0 : 1;
}

static int mv_evaluation_parent_inventory(const char *parent)
{
    DIR *stream = NULL;
    struct dirent *entry;
    unsigned char seen[2] = {0};
    int failed = 0;
    size_t index;

    stream = opendir(parent);
    if (stream == NULL) {
        return 1;
    }
    while (!failed) {
        char path[GS_PATH_SIZE];
        size_t selected;
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
        if (strcmp(entry->d_name, "k22") == 0) {
            selected = 0u;
        } else if (strcmp(entry->d_name, "k36") == 0) {
            selected = 1u;
        } else {
            failed = 1;
            continue;
        }
        if (seen[selected] ||
            gs_join(parent, entry->d_name, path, sizeof(path)) != 0 ||
            !me_directory_non_reparse(path)) {
            failed = 1;
        } else {
            seen[selected] = 1u;
        }
    }
    if (closedir(stream) != 0) {
        failed = 1;
    }
    for (index = 0u; index < 2u; index++) {
        char partial_name[32];
        char partial[GS_PATH_SIZE];
        int written = snprintf(
            partial_name, sizeof(partial_name),
            "k%lu.partial",
            (unsigned long)(index == 0u ? 22u : 36u)
        );
        if (written < 0 || (size_t)written >= sizeof(partial_name) ||
            gs_join(parent, partial_name, partial, sizeof(partial)) != 0 ||
            gs_path_state(partial, NULL, NULL) != 0) {
            failed = 1;
        }
    }
    return failed;
}

static int mv_resolve_evaluation(
    const MEBase *base,
    const char *argument,
    size_t k,
    char evaluation[GS_PATH_SIZE]
)
{
    char parent_argument[GS_PATH_SIZE];
    char parent[GS_PATH_SIZE];
    char expected_parent[GS_PATH_SIZE];
    char expected[GS_PATH_SIZE];
    char name[16];
    int written;

    written = snprintf(name, sizeof(name), "k%lu", (unsigned long)k);
    if (base == NULL || written < 0 ||
        (size_t)written >= sizeof(name) ||
        me_canonical_existing(argument, 1, evaluation) != 0 ||
        strcmp(gs_basename(evaluation), name) != 0 ||
        gs_parent_path(
            evaluation, parent_argument, sizeof(parent_argument)
        ) != 0 ||
        me_canonical_existing(parent_argument, 1, parent) != 0 ||
        gs_join(
            base->base, "label_evaluation_seed123",
            expected_parent, sizeof(expected_parent)
        ) != 0 ||
        !me_paths_equal(parent, expected_parent) ||
        gs_join(parent, name, expected, sizeof(expected)) != 0 ||
        !me_paths_equal(evaluation, expected) ||
        mv_evaluation_parent_inventory(parent) != 0 ||
        me_exact_regular_files(
            evaluation,
            me_output_files,
            ME_EVALUATION_FILE_COUNT
        ) != 0) {
        return 1;
    }
    return 0;
}

static int mv_validate_checksums(
    const MEBase *base,
    const MEKMeans *kmeans,
    const char *evaluation
)
{
    GSChecksumEntry entries[14];
    char paths[ME_EVALUATION_PATH_COUNT][GS_PATH_SIZE];
    char sha_path[GS_PATH_SIZE];
    size_t index;

    if (me_build_evaluation_entries(
            base, kmeans, evaluation, entries, paths
        ) != 0 ||
        gs_join(
            evaluation, "sha256.txt", sha_path, sizeof(sha_path)
        ) != 0 ||
        gs_verify_checksum_manifest(sha_path, entries, 14u) != 0) {
        return 1;
    }
    for (index = 0u; index < ME_EVALUATION_FILE_COUNT; index++) {
        char path[GS_PATH_SIZE];
        size_t size;
        if (gs_join(
                evaluation, me_output_files[index],
                path, sizeof(path)
            ) != 0 ||
            gs_file_size(path, &size) != 0 || size == 0u) {
            return 1;
        }
    }
    return 0;
}

static int mv_validate_summary(
    const char *evaluation,
    const MEBase *base,
    const MEKMeans *kmeans,
    const char *expected_commit,
    const GSBaselineClusteringMetrics *train,
    const GSBaselineClusteringMetrics *validation
)
{
    char profile[256];
    char kmeans_profile[256];
    char feature_count[32];
    char k_text[32];
    int written;

    written = snprintf(
        profile, sizeof(profile),
        "%s_label_evaluation", base->profile->matched_profile
    );
    if (written < 0 || (size_t)written >= sizeof(profile)) {
        return 1;
    }
    written = snprintf(
        kmeans_profile, sizeof(kmeans_profile),
        "%s_kmeans", base->profile->matched_profile
    );
    if (written < 0 || (size_t)written >= sizeof(kmeans_profile)) {
        return 1;
    }
    written = snprintf(
        feature_count, sizeof(feature_count), "%lu",
        (unsigned long)base->profile->feature_count
    );
    if (written < 0 || (size_t)written >= sizeof(feature_count)) {
        return 1;
    }
    written = snprintf(
        k_text, sizeof(k_text), "%lu", (unsigned long)kmeans->k
    );
    if (written < 0 || (size_t)written >= sizeof(k_text) ||
        me_expect_value(
            evaluation, "summary.txt", "format_version", "1"
        ) ||
        me_expect_value(
            evaluation, "summary.txt", "profile", profile
        ) ||
        me_expect_value(
            evaluation, "summary.txt", "input_profile",
            base->profile->matched_profile
        ) ||
        me_expect_value(
            evaluation, "summary.txt", "matched_subset_profile",
            "subset224_seed123_prefix_800_200"
        ) ||
        me_expect_value(
            evaluation, "summary.txt", "source_profile",
            base->profile->source_profile
        ) ||
        me_expect_value(
            evaluation, "summary.txt", "feature_count", feature_count
        ) ||
        me_expect_value(
            evaluation, "summary.txt", "standardization",
            "train_only_population_ddof0"
        ) ||
        me_expect_value(
            evaluation, "summary.txt", "kmeans_profile", kmeans_profile
        ) ||
        me_expect_value(
            evaluation, "summary.txt", "labels_policy",
            "posterior_evaluation_only"
        ) ||
        me_expect_value(
            evaluation, "summary.txt", "subset_rows", "1000"
        ) ||
        me_expect_value(
            evaluation, "summary.txt", "class_count", "22"
        ) ||
        me_expect_value(
            evaluation, "summary.txt", "cluster_count", k_text
        ) ||
        me_expect_value(evaluation, "summary.txt", "k", k_text) ||
        me_expect_value(evaluation, "summary.txt", "seed", "123") ||
        me_expect_value(
            evaluation, "summary.txt", "train_rows", "800"
        ) ||
        me_expect_value(
            evaluation, "summary.txt", "validation_rows", "200"
        ) ||
        me_expect_value(
            evaluation, "summary.txt",
            "labels_used_for_feature_extraction", "false"
        ) ||
        me_expect_value(
            evaluation, "summary.txt",
            "labels_used_for_standardization", "false"
        ) ||
        me_expect_value(
            evaluation, "summary.txt",
            "labels_used_for_pca", "false"
        ) ||
        me_expect_value(
            evaluation, "summary.txt",
            "labels_used_for_kmeans", "false"
        ) ||
        me_expect_value(
            evaluation, "summary.txt",
            "labels_used_for_evaluation", "true"
        ) ||
        me_expect_value(evaluation, "summary.txt", "pca", "false") ||
        me_expect_value(
            evaluation, "summary.txt", "augmentation", "false"
        ) ||
        me_expect_value(
            evaluation, "summary.txt", "matched_base_code_commit",
            base->base_commit
        ) ||
        me_expect_value(
            evaluation, "summary.txt", "kmeans_code_commit",
            kmeans->code_commit
        ) ||
        me_expect_value(
            evaluation, "summary.txt", "evaluator_code_commit",
            expected_commit
        ) ||
        me_expect_value(
            evaluation, "summary.txt", "code_commit", expected_commit
        ) ||
        mv_metric_matches(
            evaluation, "train_purity", train->purity
        ) ||
        mv_metric_matches(
            evaluation, "train_nmi_arithmetic",
            train->normalized_mutual_information
        ) ||
        mv_metric_matches(
            evaluation, "train_ari", train->adjusted_rand_index
        ) ||
        mv_metric_matches(
            evaluation, "train_homogeneity", train->homogeneity
        ) ||
        mv_metric_matches(
            evaluation, "train_completeness", train->completeness
        ) ||
        mv_metric_matches(
            evaluation, "validation_purity", validation->purity
        ) ||
        mv_metric_matches(
            evaluation, "validation_nmi_arithmetic",
            validation->normalized_mutual_information
        ) ||
        mv_metric_matches(
            evaluation, "validation_ari",
            validation->adjusted_rand_index
        ) ||
        mv_metric_matches(
            evaluation, "validation_homogeneity",
            validation->homogeneity
        ) ||
        mv_metric_matches(
            evaluation, "validation_completeness",
            validation->completeness
        )) {
        return 1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    MEBase base;
    MEKMeans kmeans;
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
    char train_rows_path[GS_PATH_SIZE];
    char validation_rows_path[GS_PATH_SIZE];
    char train_assignments_path[GS_PATH_SIZE];
    char validation_assignments_path[GS_PATH_SIZE];
    char evaluation[GS_PATH_SIZE];
    char train_hash_before[65];
    char validation_hash_before[65];
    char train_hash_after[65];
    char validation_hash_after[65];
    size_t train_bytes = 0u;
    size_t validation_bytes = 0u;
    size_t k = 0u;
    size_t index = 0u;
    int train_metrics_match = 0;
    int validation_metrics_match = 0;
    int rc = 1;

    memset(&base, 0, sizeof(base));
    memset(&kmeans, 0, sizeof(kmeans));
    memset(&manifest, 0, sizeof(manifest));
    memset(&train_metrics, 0, sizeof(train_metrics));
    memset(&validation_metrics, 0, sizeof(validation_metrics));

    if (argc != 8) {
        fprintf(
            stderr,
            "Usage: test_matched_subset1000_evaluation_artifacts.exe "
            "<manifest_events> <matched_base_directory> "
            "<mini75_pool5|mini112_pool7> <kmeans_directory> "
            "<evaluation_directory> <22|36> "
            "<expected_code_commit>\n"
        );
        return 1;
    }
    if (!gs_test_platform_is_valid() ||
        setlocale(LC_NUMERIC, "C") == NULL) {
        fprintf(stderr, "ERROR: invalid platform or locale.\n");
        return 1;
    }
    if (gs_parse_size(argv[6], &k) != 0 ||
        (k != 22u && k != 36u) ||
        !me_commit_valid(argv[7]) ||
        gs_checked_product(
            ME_TRAIN_ROWS, sizeof(uint32_t), &train_bytes
        ) != 0 ||
        gs_checked_product(
            ME_VALIDATION_ROWS, sizeof(uint32_t), &validation_bytes
        ) != 0) {
        fprintf(stderr, "ERROR: invalid numeric or commit arguments.\n");
        return 1;
    }

    if (me_resolve_base(argv[1], argv[2], argv[3], &base) != 0) {
        fprintf(stderr, "[FAIL] canonical_base_resolution\n");
        goto cleanup;
    }
    if (me_validate_base(&base) != 0) {
        fprintf(stderr, "[FAIL] matched_base_contract\n");
        goto cleanup;
    }
    if (me_validate_kmeans(&base, argv[4], k, &kmeans) != 0) {
        fprintf(stderr, "[FAIL] matched_kmeans_contract\n");
        goto cleanup;
    }
    if (mv_resolve_evaluation(
            &base, argv[5], k, evaluation
        ) != 0) {
        fprintf(stderr, "[FAIL] canonical_evaluation_resolution\n");
        goto cleanup;
    }
    if (gs_join(
            base.base, "train_rows.csv",
            train_rows_path, sizeof(train_rows_path)
        ) != 0 ||
        gs_join(
            base.base, "val_rows.csv",
            validation_rows_path, sizeof(validation_rows_path)
        ) != 0 ||
        gs_join(
            kmeans.directory, "train_assignments_u32.bin",
            train_assignments_path, sizeof(train_assignments_path)
        ) != 0 ||
        gs_join(
            kmeans.directory, "val_assignments_u32.bin",
            validation_assignments_path,
            sizeof(validation_assignments_path)
        ) != 0) {
        fprintf(stderr, "[FAIL] frozen_input_contract\n");
        goto cleanup;
    }
    printf("[PASS] frozen_input_contract\n");

    train_assignments = (uint32_t *)malloc(train_bytes);
    validation_assignments = (uint32_t *)malloc(validation_bytes);
    train_copy = (uint32_t *)malloc(train_bytes);
    validation_copy = (uint32_t *)malloc(validation_bytes);
    if (train_assignments == NULL || validation_assignments == NULL ||
        train_copy == NULL || validation_copy == NULL ||
        gs_read_assignments(
            train_assignments_path, train_assignments,
            ME_TRAIN_ROWS, k
        ) != 0 ||
        gs_read_assignments(
            validation_assignments_path, validation_assignments,
            ME_VALIDATION_ROWS, k
        ) != 0 ||
        gs_sha_file(
            train_assignments_path, train_hash_before
        ) != 0 ||
        gs_sha_file(
            validation_assignments_path, validation_hash_before
        ) != 0 ||
        mv_validate_checksums(&base, &kmeans, evaluation) != 0) {
        fprintf(stderr, "[FAIL] frozen_assignment_and_checksum_contract\n");
        goto cleanup;
    }
    memcpy(train_copy, train_assignments, train_bytes);
    memcpy(validation_copy, validation_assignments, validation_bytes);
    printf("[PASS] frozen_assignment_and_checksum_contract\n");

    coverage = (unsigned char *)calloc(GS_EVENT_COUNT, 1u);
    if (coverage == NULL ||
        me_validate_prelabel_rows(
            &base, train_rows_path, validation_rows_path, coverage
        ) != 0) {
        fprintf(stderr, "[FAIL] prelabel_row_mapping_and_coverage\n");
        goto cleanup;
    }
    printf("[PASS] prelabel_row_mapping_and_coverage\n");
    memset(coverage, 0, GS_EVENT_COUNT);
    train_labels = (uint32_t *)malloc(train_bytes);
    validation_labels = (uint32_t *)malloc(validation_bytes);
    if (train_labels == NULL || validation_labels == NULL ||
        gs_event_manifest_load(base.manifest, &manifest) != 0 ||
        gs_test_validate_manifest(&manifest) != 0 ||
        gs_test_load_rows(
            train_rows_path, "train", ME_TRAIN_ROWS,
            &manifest, coverage, train_labels
        ) != 0 ||
        gs_test_load_rows(
            validation_rows_path, "validation", ME_VALIDATION_ROWS,
            &manifest, coverage, validation_labels
        ) != 0) {
        fprintf(stderr, "[FAIL] independent_manifest_and_row_mapping\n");
        goto cleanup;
    }
    {
        size_t covered = 0u;
        for (index = 0u; index < GS_EVENT_COUNT; index++) {
            if (coverage[index] == 1u) {
                covered++;
            } else if (coverage[index] != 0u) {
                goto cleanup;
            }
        }
        if (covered != ME_SUBSET_ROWS) {
            fprintf(stderr, "[FAIL] independent_manifest_and_row_mapping\n");
            goto cleanup;
        }
    }
    printf("[PASS] independent_manifest_and_row_mapping\n");

    if (gs_baseline_clustering_metrics_compute(
            train_labels, train_assignments, ME_TRAIN_ROWS,
            GS_LABEL_COUNT, k, &train_metrics
        ) != GS_BASELINE_CLUSTERING_METRICS_SUCCESS ||
        gs_baseline_clustering_metrics_compute(
            validation_labels, validation_assignments,
            ME_VALIDATION_ROWS, GS_LABEL_COUNT, k,
            &validation_metrics
        ) != GS_BASELINE_CLUSTERING_METRICS_SUCCESS ||
        !gs_test_metrics_consistent(
            &train_metrics, ME_TRAIN_ROWS, k
        ) ||
        !gs_test_metrics_consistent(
            &validation_metrics, ME_VALIDATION_ROWS, k
        ) ||
        gs_test_validate_label_map(evaluation) != 0) {
        fprintf(stderr, "[FAIL] independent_label_and_metric_rebuild\n");
        goto cleanup;
    }

    if (gs_test_validate_contingency(
            evaluation, "train_contingency_counts.csv", &train_metrics
        ) != 0 ||
        gs_test_validate_contingency(
            evaluation, "val_contingency_counts.csv", &validation_metrics
        ) != 0) {
        fprintf(stderr, "[FAIL] recomputed_contingencies\n");
        goto cleanup;
    }
    printf("TRAIN_CONTINGENCY_RECOMPUTED_MATCH=true\n");
    printf("VALIDATION_CONTINGENCY_RECOMPUTED_MATCH=true\n");

    if (gs_test_validate_metrics_csv(
            evaluation, k, &train_metrics, &validation_metrics,
            &train_metrics_match, &validation_metrics_match
        ) != 0 ||
        !train_metrics_match || !validation_metrics_match) {
        fprintf(stderr, "[FAIL] recomputed_metrics\n");
        goto cleanup;
    }
    printf("TRAIN_METRICS_RECOMPUTED_MATCH=true\n");
    printf("VALIDATION_METRICS_RECOMPUTED_MATCH=true\n");

    if (gs_test_validate_cluster_summary(
            evaluation, "train_cluster_summary.csv", &train_metrics
        ) != 0 ||
        gs_test_validate_cluster_summary(
            evaluation, "val_cluster_summary.csv", &validation_metrics
        ) != 0) {
        fprintf(stderr, "[FAIL] recomputed_cluster_summaries\n");
        goto cleanup;
    }
    printf("TRAIN_CLUSTER_SUMMARY_RECOMPUTED_MATCH=true\n");
    printf("VALIDATION_CLUSTER_SUMMARY_RECOMPUTED_MATCH=true\n");

    if (gs_test_validate_label_summary(
            evaluation, "train_label_summary.csv", &train_metrics
        ) != 0 ||
        gs_test_validate_label_summary(
            evaluation, "val_label_summary.csv", &validation_metrics
        ) != 0) {
        fprintf(stderr, "[FAIL] recomputed_label_summaries\n");
        goto cleanup;
    }
    printf("TRAIN_LABEL_SUMMARY_RECOMPUTED_MATCH=true\n");
    printf("VALIDATION_LABEL_SUMMARY_RECOMPUTED_MATCH=true\n");

    if (mv_validate_summary(
            evaluation, &base, &kmeans, argv[7],
            &train_metrics, &validation_metrics
        ) != 0 ||
        memcmp(train_assignments, train_copy, train_bytes) != 0 ||
        memcmp(
            validation_assignments, validation_copy, validation_bytes
        ) != 0 ||
        gs_sha_file(train_assignments_path, train_hash_after) != 0 ||
        gs_sha_file(
            validation_assignments_path, validation_hash_after
        ) != 0 ||
        strcmp(train_hash_before, train_hash_after) != 0 ||
        strcmp(validation_hash_before, validation_hash_after) != 0 ||
        me_validate_base(&base) != 0 ||
        me_validate_kmeans(&base, argv[4], k, &kmeans) != 0 ||
        mv_validate_checksums(&base, &kmeans, evaluation) != 0) {
        fprintf(stderr, "[FAIL] posterior_only_and_frozen_sources\n");
        goto cleanup;
    }

    printf("SOURCE_ASSIGNMENTS_UNCHANGED=true\n");
    printf("LABELS_POSTERIOR_ONLY=true\n");
    printf("ALL_MATCHED_EVALUATION_CHECKSUMS_MATCH=true\n");
    printf("K=%lu\n", (unsigned long)k);
    printf("TRAIN_PURITY=%.17g\n", train_metrics.purity);
    printf(
        "TRAIN_NMI_ARITHMETIC=%.17g\n",
        train_metrics.normalized_mutual_information
    );
    printf("TRAIN_ARI=%.17g\n", train_metrics.adjusted_rand_index);
    printf("TRAIN_HOMOGENEITY=%.17g\n", train_metrics.homogeneity);
    printf("TRAIN_COMPLETENESS=%.17g\n", train_metrics.completeness);
    printf("VALIDATION_PURITY=%.17g\n", validation_metrics.purity);
    printf(
        "VALIDATION_NMI_ARITHMETIC=%.17g\n",
        validation_metrics.normalized_mutual_information
    );
    printf(
        "VALIDATION_ARI=%.17g\n",
        validation_metrics.adjusted_rand_index
    );
    printf(
        "VALIDATION_HOMOGENEITY=%.17g\n",
        validation_metrics.homogeneity
    );
    printf(
        "VALIDATION_COMPLETENESS=%.17g\n",
        validation_metrics.completeness
    );
    printf(
        "ALL MATCHED SUBSET1000 EVALUATION ARTIFACT TESTS PASSED\n"
    );
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
