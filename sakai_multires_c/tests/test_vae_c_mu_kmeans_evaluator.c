#define GS_VAE_C_MU_KMEANS_EVALUATOR_NO_MAIN

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4505)
#endif
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "../tools/evaluate_vae_c_mu_kmeans.c"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#include <errno.h>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif


enum { TEST_EVK_PATH_CAP = 4096 };


static int test_make_directory(const char *path)
{
#ifdef _WIN32
    return _mkdir(path) == 0 || errno == EEXIST;
#else
    return mkdir(path, 0777) == 0 || errno == EEXIST;
#endif
}


static int test_path_join(
    const char *directory,
    const char *name,
    char *path,
    size_t path_size
)
{
    int written = snprintf(path, path_size, "%s/%s", directory, name);

    return written >= 0 && (size_t)written < path_size;
}


static int test_write_text(
    const char *directory,
    const char *name,
    const char *text,
    char path[TEST_EVK_PATH_CAP]
)
{
    FILE *file;
    size_t length = strlen(text);
    int failed = 0;

    if (!test_path_join(directory, name, path, TEST_EVK_PATH_CAP)) {
        return 0;
    }
    file = fopen(path, "wb");
    if (file == NULL) {
        return 0;
    }
    if ((length != 0u && fwrite(text, 1u, length, file) != length) ||
        fflush(file) != 0) {
        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    return !failed;
}


static int test_write_binary_assignments(
    const char *directory,
    const char *name,
    const uint32_t *values,
    size_t count,
    char path[TEST_EVK_PATH_CAP]
)
{
    FILE *file;
    int failed = 0;

    if (!test_path_join(directory, name, path, TEST_EVK_PATH_CAP)) {
        return 0;
    }
    file = fopen(path, "wb");
    if (file == NULL) {
        return 0;
    }
    if ((count != 0u &&
         fwrite(values, sizeof(*values), count, file) != count) ||
        fflush(file) != 0) {
        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    return !failed;
}


static int test_make_child_directory(
    const char *parent,
    const char *name,
    char path[TEST_EVK_PATH_CAP]
)
{
    return test_path_join(parent, name, path, TEST_EVK_PATH_CAP) &&
        test_make_directory(path);
}


typedef enum {
    TEST_METADATA_VALID = 0,
    TEST_METADATA_DUPLICATE_IDENTICAL,
    TEST_METADATA_DUPLICATE_CONTRADICTORY
} TestMetadataVariant;


static int test_write_metadata_shape(
    const char *directory,
    const char *name,
    TestMetadataVariant variant,
    char path[TEST_EVK_PATH_CAP]
)
{
    FILE *file;
    size_t index;
    int failed = 0;

    if (!test_path_join(directory, name, path, TEST_EVK_PATH_CAP)) {
        return 0;
    }
    file = fopen(path, "wb");
    if (file == NULL) {
        return 0;
    }
    for (index = 0u; index < EVK_METADATA_LINE_COUNT; index++) {
        if (index == EVK_METADATA_LINE_COUNT - 1u &&
            variant == TEST_METADATA_DUPLICATE_IDENTICAL) {
            if (fprintf(file, "key_00=value_00\n") < 0) {
                failed = 1;
                break;
            }
        } else if (index == EVK_METADATA_LINE_COUNT - 1u &&
                   variant == TEST_METADATA_DUPLICATE_CONTRADICTORY) {
            if (fprintf(file, "key_00=contradictory_value\n") < 0) {
                failed = 1;
                break;
            }
        } else if (fprintf(
                       file, "key_%02zu=value_%02zu\n", index, index
                   ) < 0) {
            failed = 1;
            break;
        }
    }
    if (fflush(file) != 0 || ferror(file)) {
        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    return !failed;
}


static int test_metadata_shape_contract(const char *directory)
{
    char valid[TEST_EVK_PATH_CAP];
    char duplicate[TEST_EVK_PATH_CAP];
    char contradictory[TEST_EVK_PATH_CAP];

    return test_write_metadata_shape(
            directory, "metadata_shape_valid.txt", TEST_METADATA_VALID,
            valid
        ) &&
        test_write_metadata_shape(
            directory,
            "metadata_shape_duplicate.txt",
            TEST_METADATA_DUPLICATE_IDENTICAL,
            duplicate
        ) &&
        test_write_metadata_shape(
            directory,
            "metadata_shape_contradictory.txt",
            TEST_METADATA_DUPLICATE_CONTRADICTORY,
            contradictory
        ) &&
        evk_validate_metadata_shape(valid) &&
        !evk_validate_metadata_shape(duplicate) &&
        !evk_validate_metadata_shape(contradictory);
}


static int test_hash_contract(const char *directory)
{
    static const char expected[] =
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad";
    static const char wrong[] =
        "00000000000000000000000000000000"
        "00000000000000000000000000000000";
    char path[TEST_EVK_PATH_CAP];

    return test_write_text(directory, "hash_probe.bin", "abc", path) &&
        evk_verify_hash(path, expected) && !evk_verify_hash(path, wrong) &&
        !evk_verify_hash(path, "not-a-sha256");
}


static int test_inventory_contract(const char *directory)
{
    char inventory[TEST_EVK_PATH_CAP];
    char path[TEST_EVK_PATH_CAP];
    size_t index;

    if (!test_make_child_directory(
            directory, "inventory_probe", inventory
        )) {
        return 0;
    }
    for (index = 0u; index < EVK_OUTPUT_FILE_COUNT; index++) {
        if (!test_write_text(
                inventory, evk_output_names[index], "", path
            )) {
            return 0;
        }
    }
    if (!evk_directory_has_exact_inventory(inventory)) {
        return 0;
    }
    if (!test_path_join(
            inventory,
            evk_output_names[EVK_OUTPUT_FILE_COUNT - 1u],
            path,
            sizeof(path)
        ) ||
        remove(path) != 0 || evk_directory_has_exact_inventory(inventory) ||
        !test_write_text(
            inventory,
            evk_output_names[EVK_OUTPUT_FILE_COUNT - 1u],
            "",
            path
        ) ||
        !test_write_text(inventory, "unexpected.extra", "", path) ||
        evk_directory_has_exact_inventory(inventory) || remove(path) != 0 ||
        !evk_directory_has_exact_inventory(inventory)) {
        return 0;
    }
    return 1;
}


static int test_partial_presence_contract(const char *directory)
{
    char path[TEST_EVK_PATH_CAP];

    if (!test_path_join(
            directory, "partial_presence_probe.partial", path, sizeof(path)
        )) {
        return 0;
    }
    errno = 0;
    if (remove(path) != 0 && errno != ENOENT) {
        return 0;
    }
    if (!evk_path_absent(path) ||
        !test_write_text(
            directory, "partial_presence_probe.partial", "partial", path
        ) ||
        evk_path_absent(path) || remove(path) != 0 ||
        !evk_path_absent(path)) {
        return 0;
    }
    return 1;
}


static int test_invalid_cli_contract(const char *repository_root)
{
    char program[] = "evaluate_vae_c_mu_kmeans";
    char validate_only[] = "--validate-only";
    char scope_option[] = "--scope";
    char full[] = "full";
    char profile_option[] = "--profile";
    char mini75[] = "mini75";
    char mini224[] = "mini224";
    char k_option[] = "--k";
    char invalid_k[] = "23";
    char valid_k[] = "22";
    char root[TEST_EVK_PATH_CAP];
    char *invalid_k_argv[9];
    char *invalid_profile_argv[9];

    if (snprintf(root, sizeof(root), "%s", repository_root) < 0 ||
        strlen(repository_root) >= sizeof(root)) {
        return 0;
    }
    invalid_k_argv[0] = program;
    invalid_k_argv[1] = validate_only;
    invalid_k_argv[2] = scope_option;
    invalid_k_argv[3] = full;
    invalid_k_argv[4] = profile_option;
    invalid_k_argv[5] = mini75;
    invalid_k_argv[6] = k_option;
    invalid_k_argv[7] = invalid_k;
    invalid_k_argv[8] = root;

    invalid_profile_argv[0] = program;
    invalid_profile_argv[1] = validate_only;
    invalid_profile_argv[2] = scope_option;
    invalid_profile_argv[3] = full;
    invalid_profile_argv[4] = profile_option;
    invalid_profile_argv[5] = mini224;
    invalid_profile_argv[6] = k_option;
    invalid_profile_argv[7] = valid_k;
    invalid_profile_argv[8] = root;

    return evk_main(9, invalid_k_argv) == EXIT_FAILURE &&
        evk_main(9, invalid_profile_argv) == EXIT_FAILURE;
}


static void test_manifest_init(
    GSEventManifest *manifest,
    GSEventRecord records[3]
)
{
    static const char *const labels[3] = {
        "1080Lines", "Wandering_Line", "Whistle"
    };
    static const char *const ids[3] = {"H1_A", "H1_B", "H1_C"};
    size_t index;

    memset(records, 0, 3u * sizeof(*records));
    for (index = 0u; index < 3u; index++) {
        records[index].event_index = (int)index + 1;
        (void)snprintf(
            records[index].label,
            sizeof(records[index].label),
            "%s",
            labels[index]
        );
        (void)snprintf(
            records[index].event_id,
            sizeof(records[index].event_id),
            "%s",
            ids[index]
        );
    }
    manifest->records = records;
    manifest->count = 3u;
}


static void test_expected_rows_init(EVKExpectedRow expected[2])
{
    memset(expected, 0, 2u * sizeof(*expected));
    expected[0].split_pos = UINT64_C(0);
    expected[0].manifest_index = UINT64_C(0);
    expected[0].event_index = UINT64_C(1);
    (void)snprintf(expected[0].event_id, sizeof(expected[0].event_id), "H1_A");
    (void)snprintf(
        expected[0].split, sizeof(expected[0].split), "validation"
    );

    expected[1].split_pos = UINT64_C(1);
    expected[1].manifest_index = UINT64_C(2);
    expected[1].event_index = UINT64_C(3);
    (void)snprintf(expected[1].event_id, sizeof(expected[1].event_id), "H1_C");
    (void)snprintf(
        expected[1].split, sizeof(expected[1].split), "validation"
    );
}


static int test_label_contract(void)
{
    size_t index = SIZE_MAX;

    return evk_label_index("1080Lines", &index) && index == 0u &&
        evk_label_index("Wandering_Line", &index) && index == 20u &&
        evk_label_index("Whistle", &index) && index == 21u &&
        !evk_label_index("Not_A_Gravity_Spy_Class", &index) &&
        !evk_label_index(NULL, &index) &&
        !evk_label_index("1080Lines", NULL);
}


static int test_k_contract(void)
{
    return evk_validate_k(22u) && evk_validate_k(36u) &&
        !evk_validate_k(0u) && !evk_validate_k(21u) &&
        !evk_validate_k(23u) && !evk_validate_k(37u);
}


static int test_valid_join_and_zero_label_row(const char *directory)
{
    static const char csv[] =
        "row_index,split_pos,manifest_index,event_index,event_id,split,cluster_id,squared_distance\n"
        "0,0,0,1,H1_A,validation,0,1.25\n"
        "1,1,2,3,H1_C,validation,21,2.5\n";
    GSEventManifest manifest;
    GSEventRecord records[3];
    EVKExpectedRow expected[2];
    EVKPopulationReport report;
    uint32_t assignments[2] = {UINT32_MAX, UINT32_MAX};
    uint32_t bad_assignments[2] = {0u, 20u};
    char csv_path[TEST_EVK_PATH_CAP];
    char binary_path[TEST_EVK_PATH_CAP];
    char bad_binary_path[TEST_EVK_PATH_CAP];

    test_manifest_init(&manifest, records);
    test_expected_rows_init(expected);
    memset(&report, 0, sizeof(report));

    return test_write_text(
            directory, "valid_assignments.csv", csv, csv_path
        ) &&
        evk_validate_assignment_csv(
            csv_path,
            expected,
            2u,
            &manifest,
            "validation",
            22u,
            assignments,
            NULL,
            &report
        ) &&
        assignments[0] == 0u && assignments[1] == 21u &&
        report.rows == 2u && report.label_counts[0] == UINT64_C(1) &&
        report.label_counts[20] == UINT64_C(0) &&
        report.label_counts[21] == UINT64_C(1) &&
        report.used_clusters == 2u &&
        test_write_binary_assignments(
            directory,
            "valid_assignments_u32.bin",
            assignments,
            2u,
            binary_path
        ) &&
        evk_validate_assignment_binary(binary_path, assignments, 2u) &&
        test_write_binary_assignments(
            directory,
            "bad_assignments_u32.bin",
            bad_assignments,
            2u,
            bad_binary_path
        ) &&
        !evk_validate_assignment_binary(
            bad_binary_path, assignments, 2u
        );
}


static int test_csv_rejected(
    const char *directory,
    const char *name,
    const char *csv,
    GSEventManifest *manifest,
    const EVKExpectedRow expected[2],
    const char *expected_split,
    size_t k
)
{
    EVKPopulationReport report;
    uint32_t assignments[2] = {0u, 0u};
    char path[TEST_EVK_PATH_CAP];

    memset(&report, 0, sizeof(report));
    return test_write_text(directory, name, csv, path) &&
        !evk_validate_assignment_csv(
            path,
            expected,
            2u,
            manifest,
            expected_split,
            k,
            assignments,
            NULL,
            &report
        );
}


static int test_negative_joins(const char *directory)
{
    static const char bad_event_id[] =
        "row_index,split_pos,manifest_index,event_index,event_id,split,cluster_id,squared_distance\n"
        "0,0,0,1,H1_WRONG,validation,0,1.25\n"
        "1,1,2,3,H1_C,validation,21,2.5\n";
    static const char bad_event_index[] =
        "row_index,split_pos,manifest_index,event_index,event_id,split,cluster_id,squared_distance\n"
        "0,0,0,2,H1_A,validation,0,1.25\n"
        "1,1,2,3,H1_C,validation,21,2.5\n";
    static const char bad_manifest_index[] =
        "row_index,split_pos,manifest_index,event_index,event_id,split,cluster_id,squared_distance\n"
        "0,0,1,2,H1_B,validation,0,1.25\n"
        "1,1,2,3,H1_C,validation,21,2.5\n";
    static const char bad_split[] =
        "row_index,split_pos,manifest_index,event_index,event_id,split,cluster_id,squared_distance\n"
        "0,0,0,1,H1_A,train,0,1.25\n"
        "1,1,2,3,H1_C,validation,21,2.5\n";
    static const char bad_split_pos[] =
        "row_index,split_pos,manifest_index,event_index,event_id,split,cluster_id,squared_distance\n"
        "0,1,0,1,H1_A,validation,0,1.25\n"
        "1,1,2,3,H1_C,validation,21,2.5\n";
    static const char invalid_cluster[] =
        "row_index,split_pos,manifest_index,event_index,event_id,split,cluster_id,squared_distance\n"
        "0,0,0,1,H1_A,validation,22,1.25\n"
        "1,1,2,3,H1_C,validation,21,2.5\n";
    static const char nonfinite_distance[] =
        "row_index,split_pos,manifest_index,event_index,event_id,split,cluster_id,squared_distance\n"
        "0,0,0,1,H1_A,validation,0,nan\n"
        "1,1,2,3,H1_C,validation,21,2.5\n";
    GSEventManifest manifest;
    GSEventRecord records[3];
    EVKExpectedRow expected[2];
    char saved_label[GS_EVENT_MAX_LABEL_LEN];
    int passed;

    test_manifest_init(&manifest, records);
    test_expected_rows_init(expected);
    passed =
        test_csv_rejected(
            directory, "bad_event_id.csv", bad_event_id,
            &manifest, expected, "validation", 22u
        ) &&
        test_csv_rejected(
            directory, "bad_event_index.csv", bad_event_index,
            &manifest, expected, "validation", 22u
        ) &&
        test_csv_rejected(
            directory, "bad_manifest_index.csv", bad_manifest_index,
            &manifest, expected, "validation", 22u
        ) &&
        test_csv_rejected(
            directory, "bad_split.csv", bad_split,
            &manifest, expected, "validation", 22u
        ) &&
        test_csv_rejected(
            directory, "bad_split_pos.csv", bad_split_pos,
            &manifest, expected, "validation", 22u
        ) &&
        test_csv_rejected(
            directory, "invalid_cluster.csv", invalid_cluster,
            &manifest, expected, "validation", 22u
        ) &&
        test_csv_rejected(
            directory, "nonfinite_distance.csv", nonfinite_distance,
            &manifest, expected, "validation", 22u
        );
    if (!passed) {
        return 0;
    }

    (void)snprintf(saved_label, sizeof(saved_label), "%s", records[0].label);
    (void)snprintf(
        records[0].label, sizeof(records[0].label), "Unknown_Label"
    );
    passed = test_csv_rejected(
        directory,
        "unknown_label.csv",
        "row_index,split_pos,manifest_index,event_index,event_id,split,cluster_id,squared_distance\n"
        "0,0,0,1,H1_A,validation,0,1.25\n"
        "1,1,2,3,H1_C,validation,21,2.5\n",
        &manifest,
        expected,
        "validation",
        22u
    );
    (void)snprintf(records[0].label, sizeof(records[0].label), "%s", saved_label);
    return passed;
}


static int test_fixed_contingency_zero_row(void)
{
    const uint32_t labels[4] = {0u, 21u, 0u, 21u};
    const uint32_t clusters[4] = {0u, 1u, 0u, 1u};
    const size_t cluster_counts[2] = {22u, 36u};
    size_t case_index;

    for (case_index = 0u; case_index < 2u; case_index++) {
        GSBaselineClusteringMetrics metrics;
        const size_t cluster_count = cluster_counts[case_index];
        size_t cluster;

        memset(&metrics, 0, sizeof(metrics));
        if (gs_baseline_clustering_metrics_compute(
                labels, clusters, 4u, 22u, cluster_count, &metrics
            ) != 0 ||
            metrics.rows != 4u || metrics.label_count != 22u ||
            metrics.cluster_count != cluster_count ||
            metrics.contingency == NULL ||
            !evk_metrics_valid(&metrics, 4u, cluster_count)) {
            gs_baseline_clustering_metrics_free(&metrics);
            return 0;
        }
        for (cluster = 0u; cluster < cluster_count; cluster++) {
            if (metrics.contingency[
                    20u * cluster_count + cluster
                ] != UINT64_C(0)) {
                gs_baseline_clustering_metrics_free(&metrics);
                return 0;
            }
        }
        {
            double saved_purity = metrics.purity;
            uint64_t saved_total = metrics.label_totals[0];

            metrics.purity = NAN;
            if (evk_metrics_valid(&metrics, 4u, cluster_count)) {
                gs_baseline_clustering_metrics_free(&metrics);
                return 0;
            }
            metrics.purity = saved_purity;
            metrics.label_totals[0] = saved_total + UINT64_C(1);
            if (evk_metrics_valid(&metrics, 4u, cluster_count)) {
                gs_baseline_clustering_metrics_free(&metrics);
                return 0;
            }
            metrics.label_totals[0] = saved_total;
            if (!evk_metrics_valid(&metrics, 4u, cluster_count)) {
                gs_baseline_clustering_metrics_free(&metrics);
                return 0;
            }
        }
        gs_baseline_clustering_metrics_free(&metrics);
    }
    return 1;
}


int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <external-build-fixture-directory>\n", argv[0]);
        return 2;
    }
    if (!test_make_directory(argv[1])) {
        fprintf(stderr, "cannot create external-build fixture directory\n");
        return 1;
    }
    if (!test_metadata_shape_contract(argv[1])) {
        fprintf(stderr, "strict metadata shape test failed\n");
        return 1;
    }
    printf("[PASS] duplicate_and_contradictory_metadata_rejected\n");
    if (!test_hash_contract(argv[1])) {
        fprintf(stderr, "direct SHA-256 verification test failed\n");
        return 1;
    }
    printf("[PASS] direct_sha256_match_and_mismatch\n");
    if (!test_inventory_contract(argv[1])) {
        fprintf(stderr, "exact inventory test failed\n");
        return 1;
    }
    printf("[PASS] missing_and_extra_inventory_rejected\n");
    if (!test_partial_presence_contract(argv[1])) {
        fprintf(stderr, "partial presence/absence test failed\n");
        return 1;
    }
    printf("[PASS] partial_presence_and_absence\n");
    if (!test_invalid_cli_contract(argv[1])) {
        fprintf(stderr, "invalid CLI contract test failed\n");
        return 1;
    }
    printf("[PASS] invalid_k_and_full_mini224_cli_rejected\n");
    if (!test_label_contract()) {
        fprintf(stderr, "label contract test failed\n");
        return 1;
    }
    printf("[PASS] canonical_label_map\n");
    if (!test_k_contract()) {
        fprintf(stderr, "K contract test failed\n");
        return 1;
    }
    printf("[PASS] invalid_k_rejected\n");
    if (!test_valid_join_and_zero_label_row(argv[1])) {
        fprintf(stderr, "valid label join / zero-row test failed\n");
        return 1;
    }
    printf("[PASS] manifest_index_join_and_zero_label_row\n");
    if (!test_negative_joins(argv[1])) {
        fprintf(stderr, "negative row-map/label tests failed\n");
        return 1;
    }
    printf("[PASS] discordant_label_and_row_maps_rejected\n");
    printf("[PASS] invalid_cluster_id_rejected\n");
    if (!test_fixed_contingency_zero_row()) {
        fprintf(stderr, "fixed contingency zero-row test failed\n");
        return 1;
    }
    printf("[PASS] fixed_22_by_k_zero_label_row_synthetic\n");
    printf("[PASS] finite_metric_ranges_and_contingency_totals\n");
    printf("OFFICIAL_METRICS_COMPUTE_CALLS=0\n");
    printf("SYNTHETIC_METRICS_COMPUTE_CALLS=2\n");
    printf("FIXTURES_CONFINED_TO_BUILD_TREE=true\n");
    printf("OFFICIAL_OUTPUT_CREATED=false\n");
    printf("VAE_C_MU_KMEANS_EVALUATOR_UNIT_TEST_PASS=true\n");
    return 0;
}
