#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

#define GS_MATCHED_SUBSET1000_NO_MAIN
#include "../tools/build_matched_subset1000_no_pca_artifacts.c"

#include "baseline_mini75_features.h"
#include "baseline_mini75_pipeline.h"
#include "baseline_mini112_pipeline.h"
#include "vae_event_dataset.h"
#include "vae_saved_split.h"
#include "vae_tensor_manifest.h"
#include "vae_tensor_spec.h"

static int test_resolve_inputs(
    const char *source_argument,
    const GSProfile *profile,
    const char *subset_train_argument,
    const char *subset_validation_argument,
    GSResolved *resolved
)
{
    static const char prefix[] =
        "/sakai_multires_c/results/baseline/";
    char suffix[GS_PATH_CAPACITY];
    char expected[GS_PATH_CAPACITY];
    char canonical[GS_PATH_CAPACITY];
    size_t source_length;
    size_t suffix_length;
    size_t root_length;
    int written;
    if (profile == NULL || resolved == NULL ||
        gs_canonical_existing(source_argument, 1, resolved->source) != 0) {
        return 1;
    }
    written = snprintf(suffix, sizeof(suffix), "%s%s",
                       prefix, profile->source_basename);
    if (written < 0 || (size_t)written >= sizeof(suffix)) return 1;
    source_length = strlen(resolved->source);
    suffix_length = strlen(suffix);
    if (source_length <= suffix_length) return 1;
    root_length = source_length - suffix_length;
#ifdef _WIN32
    if (_stricmp(resolved->source + root_length, suffix) != 0) return 1;
#else
    if (strcmp(resolved->source + root_length, suffix) != 0) return 1;
#endif
    if (root_length == 0u || root_length >= sizeof(resolved->root)) return 1;
    memcpy(resolved->root, resolved->source, root_length);
    resolved->root[root_length] = '\0';
    if (gs_canonical_existing(resolved->root, 1, canonical) != 0 ||
        !gs_paths_equal(resolved->root, canonical) ||
        gs_path_join(
            resolved->root,
            "sakai_multires_c/results/baseline",
            expected,
            sizeof(expected)
        ) != 0 ||
        gs_canonical_existing(expected, 1, canonical) != 0 ||
        gs_path_join(canonical, profile->source_basename,
                     expected, sizeof(expected)) != 0 ||
        gs_canonical_existing(expected, 1, canonical) != 0 ||
        !gs_paths_equal(resolved->source, canonical) ||
        gs_path_join(resolved->root, "manifest_events_272.csv",
                     expected, sizeof(expected)) != 0 ||
        gs_canonical_existing(expected, 0, resolved->manifest) != 0 ||
        gs_path_join(
            resolved->root,
            "sakai_multires_c/configs/subset224_train_800_seed123.csv",
            expected,
            sizeof(expected)
        ) != 0 ||
        gs_canonical_existing(expected, 0, resolved->subset_train) != 0 ||
        gs_canonical_existing(subset_train_argument, 0, canonical) != 0 ||
        !gs_paths_equal(resolved->subset_train, canonical) ||
        gs_path_join(
            resolved->root,
            "sakai_multires_c/configs/subset224_val_200_seed123.csv",
            expected,
            sizeof(expected)
        ) != 0 ||
        gs_canonical_existing(expected, 0, resolved->subset_validation) != 0 ||
        gs_canonical_existing(subset_validation_argument, 0, canonical) != 0 ||
        !gs_paths_equal(resolved->subset_validation, canonical)) {
        return 1;
    }
    return 0;
}

static int test_hash_source_inputs(
    const GSResolved *resolved,
    char hashes[4][65]
)
{
    char path[GS_PATH_CAPACITY];
    if (gs_path_join(resolved->source, "train_features_raw_f32.bin",
                     path, sizeof(path)) != 0 ||
        gs_sha_file(path, hashes[0]) != 0 ||
        gs_path_join(resolved->source, "val_features_raw_f32.bin",
                     path, sizeof(path)) != 0 ||
        gs_sha_file(path, hashes[1]) != 0 ||
        gs_path_join(resolved->source, "train_rows.csv",
                     path, sizeof(path)) != 0 ||
        gs_sha_file(path, hashes[2]) != 0 ||
        gs_path_join(resolved->source, "val_rows.csv",
                     path, sizeof(path)) != 0 ||
        gs_sha_file(path, hashes[3]) != 0) {
        return 1;
    }
    return 0;
}

static int test_direct_sample_mini75(
    const GSResolved *resolved,
    const float *train_raw,
    const float *validation_raw
)
{
    static const size_t train_positions[8] =
        {0u,1u,2u,3u,796u,797u,798u,799u};
    static const size_t validation_positions[8] =
        {0u,1u,2u,3u,196u,197u,198u,199u};
    GSEventManifest events = {0};
    GSTensorManifest tensors = {0};
    GSSavedSplit train = {0};
    GSSavedSplit validation = {0};
    GSTensorSpec spec = {0};
    float *tensor_buffer = NULL;
    float *features = NULL;
    char path[GS_PATH_CAPACITY];
    size_t index;
    int failed = 1;
    if (gs_tensor_spec_init(&spec, 75) != 0 ||
        gs_event_manifest_load(resolved->manifest, &events) != 0 ||
        gs_path_join(resolved->root, "manifest_tensors_75_central.csv",
                     path, sizeof(path)) != 0 ||
        gs_tensor_manifest_load(path, &spec, &tensors) != 0 ||
        gs_path_join(resolved->root, "split_train_indices_seed123.csv",
                     path, sizeof(path)) != 0 ||
        gs_saved_split_load(path, "train", &events, &train) != 0 ||
        gs_path_join(resolved->root, "split_val_indices_seed123.csv",
                     path, sizeof(path)) != 0 ||
        gs_saved_split_load(path, "validation", &events, &validation) != 0) {
        goto cleanup;
    }
    tensor_buffer = (float *)malloc(
        GS_BASELINE_MINI75_INPUT_COUNT * sizeof(float));
    features = (float *)malloc(
        GS_BASELINE_MINI75_FEATURE_COUNT * sizeof(float));
    if (tensor_buffer == NULL || features == NULL) goto cleanup;
    for (index = 0u; index < 8u; index++) {
        size_t position = train_positions[index];
        if (gs_baseline_mini75_extract_split_rows(
                resolved->root, &events, &tensors, &train,
                position, 1u, tensor_buffer,
                GS_BASELINE_MINI75_INPUT_COUNT, features,
                GS_BASELINE_MINI75_FEATURE_COUNT
            ) != GS_BASELINE_MINI75_PIPELINE_SUCCESS ||
            memcmp(features,
                   train_raw + position * GS_BASELINE_MINI75_FEATURE_COUNT,
                   GS_BASELINE_MINI75_FEATURE_COUNT * sizeof(float)) != 0) {
            goto cleanup;
        }
    }
    for (index = 0u; index < 8u; index++) {
        size_t position = validation_positions[index];
        if (gs_baseline_mini75_extract_split_rows(
                resolved->root, &events, &tensors, &validation,
                position, 1u, tensor_buffer,
                GS_BASELINE_MINI75_INPUT_COUNT, features,
                GS_BASELINE_MINI75_FEATURE_COUNT
            ) != GS_BASELINE_MINI75_PIPELINE_SUCCESS ||
            memcmp(features,
                   validation_raw +
                       position * GS_BASELINE_MINI75_FEATURE_COUNT,
                   GS_BASELINE_MINI75_FEATURE_COUNT * sizeof(float)) != 0) {
            goto cleanup;
        }
    }
    failed = 0;
cleanup:
    free(features);
    free(tensor_buffer);
    gs_saved_split_free(&validation);
    gs_saved_split_free(&train);
    gs_tensor_manifest_free(&tensors);
    gs_event_manifest_free(&events);
    return failed;
}

static int test_direct_sample_mini112(
    const GSResolved *resolved,
    const float *train_raw,
    const float *validation_raw
)
{
    static const size_t train_positions[8] =
        {0u,1u,2u,3u,796u,797u,798u,799u};
    static const size_t validation_positions[8] =
        {0u,1u,2u,3u,196u,197u,198u,199u};
    GSBaselineMini112Pipeline pipeline = {0};
    float *features = NULL;
    char tensor_manifest[GS_PATH_CAPACITY];
    char train_split[GS_PATH_CAPACITY];
    char validation_split[GS_PATH_CAPACITY];
    size_t index;
    int failed = 1;
    gs_baseline_mini112_pipeline_init(&pipeline);
    if (gs_path_join(resolved->root, "manifest_tensors_112_central.csv",
                     tensor_manifest, sizeof(tensor_manifest)) != 0 ||
        gs_path_join(resolved->root, "split_train_indices_seed123.csv",
                     train_split, sizeof(train_split)) != 0 ||
        gs_path_join(resolved->root, "split_val_indices_seed123.csv",
                     validation_split, sizeof(validation_split)) != 0 ||
        gs_baseline_mini112_pipeline_open(
            &pipeline, resolved->manifest, tensor_manifest,
            train_split, validation_split, resolved->root
        ) != GS_BASELINE_MINI112_PIPELINE_SUCCESS) {
        goto cleanup;
    }
    features = (float *)malloc(
        GS_BASELINE_MINI112_FEATURE_COUNT * sizeof(float));
    if (features == NULL) goto cleanup;
    for (index = 0u; index < 8u; index++) {
        size_t position = train_positions[index];
        if (gs_baseline_mini112_extract_train_rows(
                &pipeline, position, 1u, features,
                GS_BASELINE_MINI112_FEATURE_COUNT
            ) != GS_BASELINE_MINI112_PIPELINE_SUCCESS ||
            memcmp(features,
                   train_raw + position * GS_BASELINE_MINI112_FEATURE_COUNT,
                   GS_BASELINE_MINI112_FEATURE_COUNT * sizeof(float)) != 0) {
            goto cleanup;
        }
    }
    for (index = 0u; index < 8u; index++) {
        size_t position = validation_positions[index];
        if (gs_baseline_mini112_extract_validation_rows(
                &pipeline, position, 1u, features,
                GS_BASELINE_MINI112_FEATURE_COUNT
            ) != GS_BASELINE_MINI112_PIPELINE_SUCCESS ||
            memcmp(features,
                   validation_raw +
                       position * GS_BASELINE_MINI112_FEATURE_COUNT,
                   GS_BASELINE_MINI112_FEATURE_COUNT * sizeof(float)) != 0) {
            goto cleanup;
        }
    }
    failed = 0;
cleanup:
    free(features);
    gs_baseline_mini112_pipeline_free(&pipeline);
    gs_baseline_mini112_pipeline_free(&pipeline);
    return failed;
}

int main(int argc, char **argv)
{
    const GSProfile *profile;
    GSResolved resolved;
    GSRow *source_train = NULL;
    GSRow *source_validation = NULL;
    GSRow subset_train[GS_MATCHED_TRAIN_ROWS];
    GSRow subset_validation[GS_MATCHED_VALIDATION_ROWS];
    float *train_raw = NULL;
    float *validation_raw = NULL;
    float *train_std_a = NULL;
    float *validation_std_a = NULL;
    float *train_std_b = NULL;
    float *validation_std_b = NULL;
    double *mean_a = NULL;
    double *stddev_a = NULL;
    double *mean_b = NULL;
    double *stddev_b = NULL;
    unsigned char *mask_a = NULL;
    unsigned char *mask_b = NULL;
    GSStatistics statistics = {0};
    char path[GS_PATH_CAPACITY];
    char hashes_before[4][65];
    char hashes_after[4][65];
    size_t train_values;
    size_t validation_values;
    size_t train_bytes;
    size_t validation_bytes;
    size_t double_bytes;
    size_t index;
    int direct_result;
    int result = 1;

    if (argc != 5) {
        fprintf(stderr,
            "Usage: test_matched_subset1000_no_pca_pipeline.exe "
            "<source_full_no_pca_directory> "
            "<mini75_pool5|mini112_pool7> "
            "<subset_train_csv> <subset_val_csv>\n");
        return 1;
    }
    profile = gs_profile_from_token(argv[2]);
    if (!gs_platform_valid() || profile == NULL ||
        test_resolve_inputs(argv[1], profile, argv[3], argv[4],
                            &resolved) != 0 ||
        gs_validate_source_inventory(resolved.source) != 0 ||
        gs_validate_source_sha(&resolved, profile) != 0 ||
        gs_validate_source_metadata(&resolved, profile) != 0 ||
        test_hash_source_inputs(&resolved, hashes_before) != 0) {
        fprintf(stderr, "ERROR: pipeline source validation failed.\n");
        return 1;
    }
    source_train = (GSRow *)calloc(GS_SOURCE_TRAIN_ROWS, sizeof(*source_train));
    source_validation =
        (GSRow *)calloc(GS_SOURCE_VALIDATION_ROWS, sizeof(*source_validation));
    if (source_train == NULL || source_validation == NULL ||
        gs_path_join(resolved.source, "train_rows.csv", path,
                     sizeof(path)) != 0 ||
        gs_load_rows(path, "train", GS_SOURCE_TRAIN_ROWS, 0,
                     source_train) != 0 ||
        gs_path_join(resolved.source, "val_rows.csv", path,
                     sizeof(path)) != 0 ||
        gs_load_rows(path, "validation", GS_SOURCE_VALIDATION_ROWS, 0,
                     source_validation) != 0 ||
        gs_load_rows(resolved.subset_train, "train",
                     GS_MATCHED_TRAIN_ROWS, 1, subset_train) != 0 ||
        gs_load_rows(resolved.subset_validation, "validation",
                     GS_MATCHED_VALIDATION_ROWS, 1,
                     subset_validation) != 0) {
        goto cleanup;
    }
    for (index = 0u; index < GS_MATCHED_TRAIN_ROWS; index++) {
        if (!gs_rows_equal(source_train + index, subset_train + index)) {
            goto cleanup;
        }
    }
    for (index = 0u; index < GS_MATCHED_VALIDATION_ROWS; index++) {
        if (!gs_rows_equal(source_validation + index,
                           subset_validation + index)) {
            goto cleanup;
        }
    }
    if (gs_validate_manifest_identifiers(
            resolved.manifest, subset_train, subset_validation) != 0 ||
        gs_checked_product(GS_MATCHED_TRAIN_ROWS, profile->feature_count,
                           &train_values) != 0 ||
        gs_checked_product(GS_MATCHED_VALIDATION_ROWS,
                           profile->feature_count, &validation_values) != 0 ||
        gs_checked_product(train_values, sizeof(float), &train_bytes) != 0 ||
        gs_checked_product(validation_values, sizeof(float),
                           &validation_bytes) != 0 ||
        gs_checked_product(profile->feature_count, sizeof(double),
                           &double_bytes) != 0) {
        goto cleanup;
    }
    train_raw = (float *)malloc(train_bytes);
    validation_raw = (float *)malloc(validation_bytes);
    train_std_a = (float *)malloc(train_bytes);
    validation_std_a = (float *)malloc(validation_bytes);
    train_std_b = (float *)malloc(train_bytes);
    validation_std_b = (float *)malloc(validation_bytes);
    mean_a = (double *)malloc(double_bytes);
    stddev_a = (double *)malloc(double_bytes);
    mean_b = (double *)malloc(double_bytes);
    stddev_b = (double *)malloc(double_bytes);
    mask_a = (unsigned char *)malloc(profile->feature_count);
    mask_b = (unsigned char *)malloc(profile->feature_count);
    if (train_raw == NULL || validation_raw == NULL ||
        train_std_a == NULL || validation_std_a == NULL ||
        train_std_b == NULL || validation_std_b == NULL ||
        mean_a == NULL || stddev_a == NULL ||
        mean_b == NULL || stddev_b == NULL ||
        mask_a == NULL || mask_b == NULL) {
        goto cleanup;
    }
    if (gs_path_join(resolved.source, "train_features_raw_f32.bin",
                     path, sizeof(path)) != 0 ||
        gs_read_raw_prefix_and_validate(
            path, GS_SOURCE_TRAIN_ROWS, GS_MATCHED_TRAIN_ROWS,
            profile->feature_count, train_raw,
            &statistics.train_raw_min, &statistics.train_raw_max) != 0 ||
        gs_path_join(resolved.source, "val_features_raw_f32.bin",
                     path, sizeof(path)) != 0 ||
        gs_read_raw_prefix_and_validate(
            path, GS_SOURCE_VALIDATION_ROWS, GS_MATCHED_VALIDATION_ROWS,
            profile->feature_count, validation_raw,
            &statistics.validation_raw_min,
            &statistics.validation_raw_max) != 0) {
        goto cleanup;
    }
    if (gs_baseline_standardization_fit(
            train_raw, GS_MATCHED_TRAIN_ROWS, profile->feature_count,
            mean_a, stddev_a, mask_a,
            GS_BASELINE_STANDARDIZATION_DEFAULT_CONSTANT_EPSILON
        ) != GS_BASELINE_STANDARDIZATION_SUCCESS ||
        gs_baseline_standardization_transform(
            train_raw, GS_MATCHED_TRAIN_ROWS, profile->feature_count,
            mean_a, stddev_a, mask_a, train_std_a
        ) != GS_BASELINE_STANDARDIZATION_SUCCESS ||
        gs_baseline_standardization_transform(
            validation_raw, GS_MATCHED_VALIDATION_ROWS,
            profile->feature_count, mean_a, stddev_a, mask_a,
            validation_std_a
        ) != GS_BASELINE_STANDARDIZATION_SUCCESS ||
        gs_baseline_standardization_fit(
            train_raw, GS_MATCHED_TRAIN_ROWS, profile->feature_count,
            mean_b, stddev_b, mask_b,
            GS_BASELINE_STANDARDIZATION_DEFAULT_CONSTANT_EPSILON
        ) != GS_BASELINE_STANDARDIZATION_SUCCESS ||
        gs_baseline_standardization_transform(
            train_raw, GS_MATCHED_TRAIN_ROWS, profile->feature_count,
            mean_b, stddev_b, mask_b, train_std_b
        ) != GS_BASELINE_STANDARDIZATION_SUCCESS ||
        gs_baseline_standardization_transform(
            validation_raw, GS_MATCHED_VALIDATION_ROWS,
            profile->feature_count, mean_b, stddev_b, mask_b,
            validation_std_b
        ) != GS_BASELINE_STANDARDIZATION_SUCCESS ||
        memcmp(mean_a, mean_b, double_bytes) != 0 ||
        memcmp(stddev_a, stddev_b, double_bytes) != 0 ||
        memcmp(mask_a, mask_b, profile->feature_count) != 0 ||
        memcmp(train_std_a, train_std_b, train_bytes) != 0 ||
        memcmp(validation_std_a, validation_std_b,
               validation_bytes) != 0 ||
        gs_validate_standardized(
            train_std_a, validation_std_a, mean_a, stddev_a,
            mask_a, profile->feature_count, &statistics) != 0) {
        goto cleanup;
    }
    if (gs_baseline_standardization_fit(
            train_raw, 0u, profile->feature_count,
            mean_b, stddev_b, mask_b,
            GS_BASELINE_STANDARDIZATION_DEFAULT_CONSTANT_EPSILON
        ) == GS_BASELINE_STANDARDIZATION_SUCCESS ||
        gs_baseline_standardization_transform(
            train_raw, GS_MATCHED_TRAIN_ROWS, 0u,
            mean_a, stddev_a, mask_a, train_std_b
        ) == GS_BASELINE_STANDARDIZATION_SUCCESS ||
        gs_checked_product(SIZE_MAX, 2u, &index) == 0) {
        goto cleanup;
    }
    direct_result = strcmp(profile->token, "mini75_pool5") == 0 ?
        test_direct_sample_mini75(&resolved, train_raw, validation_raw) :
        test_direct_sample_mini112(&resolved, train_raw, validation_raw);
    if (direct_result != 0 ||
        test_hash_source_inputs(&resolved, hashes_after) != 0 ||
        memcmp(hashes_before, hashes_after, sizeof(hashes_before)) != 0) {
        goto cleanup;
    }
    printf("[PASS] exact_row_maps_800_200\n");
    printf("[PASS] full_raw_prefix_bitwise_and_finite\n");
    printf("[PASS] row_major_feature_dimension_%zu\n",
           profile->feature_count);
    printf("[PASS] deterministic_train_only_standardization\n");
    printf("[PASS] validation_without_refit_and_scaler_immutable\n");
    printf("[PASS] source_inputs_immutable\n");
    printf("[PASS] invalid_dimensions_and_overflow_rejected\n");
    printf("[PASS] idempotent_pipeline_cleanup\n");
    if (strcmp(profile->token, "mini75_pool5") == 0) {
        printf("MINI75_MATCHED_PIPELINE_VALID=true\n");
        printf("MINI75_DIRECT_SAMPLE_MATCH=true\n");
    } else {
        printf("MINI112_MATCHED_PIPELINE_VALID=true\n");
        printf("MINI112_DIRECT_SAMPLE_MATCH=true\n");
    }
    printf("ALL MATCHED SUBSET1000 PIPELINE TESTS PASSED\n");
    result = 0;

cleanup:
    free(mask_b);
    free(mask_a);
    free(stddev_b);
    free(mean_b);
    free(stddev_a);
    free(mean_a);
    free(validation_std_b);
    free(train_std_b);
    free(validation_std_a);
    free(train_std_a);
    free(validation_raw);
    free(train_raw);
    free(source_validation);
    free(source_train);
    if (result != 0) {
        fprintf(stderr, "MATCHED SUBSET1000 PIPELINE TEST FAILED\n");
    }
    return result;
}
