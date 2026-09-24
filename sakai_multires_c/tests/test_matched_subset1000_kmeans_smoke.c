#define GS_MATCHED_KMEANS_NO_MAIN
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "../tools/run_matched_subset1000_kmeans.c"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

enum {
    SMOKE_TRAIN_ROWS = 128,
    SMOKE_VALIDATION_ROWS = 32,
    SMOKE_K = 4,
    SMOKE_MAX_ITERATIONS = 100
};

typedef struct {
    GSBaselineKMeansResult fit;
    size_t *train_assignments;
    size_t *validation_assignments;
    double *train_distances;
    double *validation_distances;
    uint64_t cluster_sizes[SMOKE_K];
    double validation_inertia;
} SmokeRun;

static void smoke_run_free(SmokeRun *run)
{
    if (run == NULL) {
        return;
    }
    free(run->validation_distances);
    free(run->train_distances);
    free(run->validation_assignments);
    free(run->train_assignments);
    gs_baseline_kmeans_result_free(&run->fit);
    memset(run, 0, sizeof(*run));
}

static int smoke_run_once(const KMBase *base, SmokeRun *run)
{
    size_t train_bytes;
    size_t validation_bytes;
    size_t row;

    memset(run, 0, sizeof(*run));
    if (gs_checked_product(
            SMOKE_TRAIN_ROWS, sizeof(size_t), &train_bytes
        ) != 0 ||
        gs_checked_product(
            SMOKE_VALIDATION_ROWS, sizeof(size_t), &validation_bytes
        ) != 0) {
        return 1;
    }
    run->train_assignments = (size_t *)malloc(train_bytes);
    run->validation_assignments = (size_t *)malloc(validation_bytes);
    run->train_distances =
        (double *)malloc(SMOKE_TRAIN_ROWS * sizeof(double));
    run->validation_distances =
        (double *)malloc(SMOKE_VALIDATION_ROWS * sizeof(double));
    if (run->train_assignments == NULL ||
        run->validation_assignments == NULL ||
        run->train_distances == NULL ||
        run->validation_distances == NULL ||
        gs_baseline_kmeans_fit(
            base->train,
            SMOKE_TRAIN_ROWS,
            base->feature_count,
            SMOKE_K,
            KM_SEED,
            SMOKE_MAX_ITERATIONS,
            &run->fit
        ) != GS_BASELINE_KMEANS_SUCCESS ||
        gs_baseline_kmeans_assign(
            base->train,
            SMOKE_TRAIN_ROWS,
            base->feature_count,
            run->fit.centroids,
            SMOKE_K,
            run->train_assignments,
            run->train_distances
        ) != GS_BASELINE_KMEANS_SUCCESS ||
        gs_baseline_kmeans_assign(
            base->validation,
            SMOKE_VALIDATION_ROWS,
            base->feature_count,
            run->fit.centroids,
            SMOKE_K,
            run->validation_assignments,
            run->validation_distances
        ) != GS_BASELINE_KMEANS_SUCCESS) {
        return 1;
    }
    for (row = 0u; row < SMOKE_TRAIN_ROWS; row++) {
        if (run->train_assignments[row] >= SMOKE_K ||
            !isfinite(run->train_distances[row]) ||
            run->train_distances[row] < 0.0) {
            return 1;
        }
        run->cluster_sizes[run->train_assignments[row]]++;
    }
    for (row = 0u; row < SMOKE_VALIDATION_ROWS; row++) {
        if (run->validation_assignments[row] >= SMOKE_K ||
            !isfinite(run->validation_distances[row]) ||
            run->validation_distances[row] < 0.0) {
            return 1;
        }
        run->validation_inertia += run->validation_distances[row];
        if (!isfinite(run->validation_inertia)) {
            return 1;
        }
    }
    return 0;
}

int main(int argc, char **argv)
{
    KMBase base;
    SmokeRun first;
    SmokeRun second;
    double *centroids_copy = NULL;
    size_t centroid_count;
    size_t centroid_bytes;
    size_t train_size_t_bytes;
    size_t validation_size_t_bytes;
    size_t train_distance_bytes;
    size_t validation_distance_bytes;
    int rc = 1;

    memset(&base, 0, sizeof(base));
    memset(&first, 0, sizeof(first));
    memset(&second, 0, sizeof(second));
    if (setlocale(LC_ALL, "C") == NULL || argc != 3 ||
        km_prepare_base(argv[1], argv[2], &base) != 0 ||
        gs_checked_product(
            SMOKE_K, base.feature_count, &centroid_count
        ) != 0 ||
        gs_checked_product(
            centroid_count, sizeof(double), &centroid_bytes
        ) != 0 ||
        gs_checked_product(
            SMOKE_TRAIN_ROWS, sizeof(size_t), &train_size_t_bytes
        ) != 0 ||
        gs_checked_product(
            SMOKE_VALIDATION_ROWS, sizeof(size_t),
            &validation_size_t_bytes
        ) != 0 ||
        gs_checked_product(
            SMOKE_TRAIN_ROWS, sizeof(double), &train_distance_bytes
        ) != 0 ||
        gs_checked_product(
            SMOKE_VALIDATION_ROWS, sizeof(double),
            &validation_distance_bytes
        ) != 0 ||
        smoke_run_once(&base, &first) != 0 ||
        smoke_run_once(&base, &second) != 0) {
        fprintf(
            stderr,
            "Usage: test_matched_subset1000_kmeans_smoke.exe "
            "<matched_base_directory> "
            "<mini75_pool5|mini112_pool7>\n"
        );
        goto cleanup;
    }
    centroids_copy = (double *)malloc(centroid_bytes);
    if (centroids_copy == NULL) {
        goto cleanup;
    }
    memcpy(centroids_copy, first.fit.centroids, centroid_bytes);
    if (gs_baseline_kmeans_assign(
            base.validation,
            SMOKE_VALIDATION_ROWS,
            base.feature_count,
            first.fit.centroids,
            SMOKE_K,
            first.validation_assignments,
            first.validation_distances
        ) != GS_BASELINE_KMEANS_SUCCESS ||
        memcmp(
            centroids_copy, first.fit.centroids, centroid_bytes
        ) != 0 ||
        memcmp(
            first.fit.centroids, second.fit.centroids, centroid_bytes
        ) != 0 ||
        memcmp(
            first.train_assignments, second.train_assignments,
            train_size_t_bytes
        ) != 0 ||
        memcmp(
            first.validation_assignments, second.validation_assignments,
            validation_size_t_bytes
        ) != 0 ||
        memcmp(
            first.train_distances, second.train_distances,
            train_distance_bytes
        ) != 0 ||
        memcmp(
            first.validation_distances, second.validation_distances,
            validation_distance_bytes
        ) != 0 ||
        memcmp(
            first.cluster_sizes, second.cluster_sizes,
            sizeof(first.cluster_sizes)
        ) != 0 ||
        memcmp(
            first.fit.initial_centroid_indices,
            second.fit.initial_centroid_indices,
            SMOKE_K * sizeof(size_t)
        ) != 0 ||
        first.fit.iterations != second.fit.iterations ||
        first.fit.converged != second.fit.converged ||
        first.fit.empty_cluster_reseeds !=
            second.fit.empty_cluster_reseeds ||
        memcmp(
            &first.fit.initial_inertia,
            &second.fit.initial_inertia,
            sizeof(double)
        ) != 0 ||
        memcmp(
            &first.fit.inertia,
            &second.fit.inertia,
            sizeof(double)
        ) != 0 ||
        memcmp(
            &first.validation_inertia,
            &second.validation_inertia,
            sizeof(double)
        ) != 0) {
        fprintf(stderr, "ERROR: deterministic smoke contract failed.\n");
        goto cleanup;
    }

    printf("PROFILE_TOKEN=%s\n", base.profile->token);
    printf("TRAIN_ROWS_SMOKE=128\n");
    printf("VALIDATION_ROWS_SMOKE=32\n");
    printf("FEATURE_COUNT=%zu\n", base.feature_count);
    printf("K_SMOKE=4\n");
    printf("MATCHED_KMEANS_SMOKE_REPEAT_CENTROIDS_BITWISE=true\n");
    printf("MATCHED_KMEANS_SMOKE_REPEAT_ASSIGNMENTS_BITWISE=true\n");
    printf("MATCHED_KMEANS_SMOKE_REPEAT_DISTANCES_BITWISE=true\n");
    printf("MATCHED_KMEANS_SMOKE_REPEAT_CLUSTER_SIZES_BITWISE=true\n");
    printf("MATCHED_KMEANS_SMOKE_REPEAT_INITIAL_INDICES_BITWISE=true\n");
    printf("MATCHED_KMEANS_SMOKE_REPEAT_INERTIA_BITWISE=true\n");
    printf("MATCHED_KMEANS_SMOKE_VALIDATION_CENTROIDS_UNCHANGED=true\n");
    printf("MATCHED_KMEANS_SMOKE_ALL_VALUES_FINITE=true\n");
    printf("ALL MATCHED SUBSET1000 KMEANS SMOKE TESTS PASSED\n");
    rc = 0;

cleanup:
    free(centroids_copy);
    smoke_run_free(&second);
    smoke_run_free(&first);
    km_base_free(&base);
    return rc;
}
