#include "baseline_mini75_pipeline.h"

#include "baseline_mini75_features.h"
#include "baseline_standardization.h"
#include "vae_tensor_spec.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


enum {
    GS_TEST_TRAIN_ROWS = 8,
    GS_TEST_VALIDATION_ROWS = 4
};


#define GS_TEST_STANDARDIZED_TOLERANCE 1.0e-5


static int gs_test_values_are_finite(
    const float *values,
    size_t count
)
{
    size_t index;

    if (values == NULL) {
        return 0;
    }

    for (index = 0; index < count; index++) {
        if (!isfinite(values[index])) {
            return 0;
        }
    }

    return 1;
}


static int gs_test_values_are_float01(
    const float *values,
    size_t count
)
{
    size_t index;

    if (!gs_test_values_are_finite(values, count)) {
        return 0;
    }

    for (index = 0; index < count; index++) {
        if (values[index] < 0.0f || values[index] > 1.0f) {
            return 0;
        }
    }

    return 1;
}


static int gs_test_extract_and_check_order(
    const char *data_root,
    const GSEventManifest *events,
    const GSTensorManifest *tensors,
    const GSSavedSplit *split,
    size_t rows,
    float *tensor_buffer,
    float *raw_matrix,
    float *single_row
)
{
    size_t row;
    size_t matrix_count =
        rows * GS_BASELINE_MINI75_FEATURE_COUNT;

    if (gs_baseline_mini75_extract_split_rows(
            data_root,
            events,
            tensors,
            split,
            0,
            rows,
            tensor_buffer,
            GS_BASELINE_MINI75_INPUT_COUNT,
            raw_matrix,
            matrix_count
        ) != GS_BASELINE_MINI75_PIPELINE_SUCCESS ||
        !gs_test_values_are_float01(raw_matrix, matrix_count)) {

        return 1;
    }

    for (row = 0; row < rows; row++) {
        if (gs_baseline_mini75_extract_split_rows(
                data_root,
                events,
                tensors,
                split,
                row,
                1,
                tensor_buffer,
                GS_BASELINE_MINI75_INPUT_COUNT,
                single_row,
                GS_BASELINE_MINI75_FEATURE_COUNT
            ) != GS_BASELINE_MINI75_PIPELINE_SUCCESS ||

            memcmp(
                raw_matrix +
                    row * GS_BASELINE_MINI75_FEATURE_COUNT,
                single_row,
                GS_BASELINE_MINI75_FEATURE_COUNT * sizeof(float)
            ) != 0) {

            return 1;
        }
    }

    return 0;
}


static int gs_test_train_only_standardization(
    const float *train_raw,
    const float *validation_raw,
    float *train_standardized,
    float *validation_standardized,
    double *mean,
    double *stddev,
    unsigned char *constant_mask,
    double *mean_copy,
    double *stddev_copy,
    unsigned char *constant_mask_copy
)
{
    size_t feature;
    size_t nonconstant_count = 0;

    if (gs_baseline_standardization_fit(
            train_raw,
            GS_TEST_TRAIN_ROWS,
            GS_BASELINE_MINI75_FEATURE_COUNT,
            mean,
            stddev,
            constant_mask,
            GS_BASELINE_STANDARDIZATION_DEFAULT_CONSTANT_EPSILON
        ) != GS_BASELINE_STANDARDIZATION_SUCCESS) {

        return 1;
    }

    memcpy(
        mean_copy,
        mean,
        GS_BASELINE_MINI75_FEATURE_COUNT * sizeof(double)
    );
    memcpy(
        stddev_copy,
        stddev,
        GS_BASELINE_MINI75_FEATURE_COUNT * sizeof(double)
    );
    memcpy(
        constant_mask_copy,
        constant_mask,
        GS_BASELINE_MINI75_FEATURE_COUNT * sizeof(unsigned char)
    );

    if (gs_baseline_standardization_transform(
            train_raw,
            GS_TEST_TRAIN_ROWS,
            GS_BASELINE_MINI75_FEATURE_COUNT,
            mean,
            stddev,
            constant_mask,
            train_standardized
        ) != GS_BASELINE_STANDARDIZATION_SUCCESS ||

        gs_baseline_standardization_transform(
            validation_raw,
            GS_TEST_VALIDATION_ROWS,
            GS_BASELINE_MINI75_FEATURE_COUNT,
            mean,
            stddev,
            constant_mask,
            validation_standardized
        ) != GS_BASELINE_STANDARDIZATION_SUCCESS) {

        return 1;
    }

    if (memcmp(
            mean,
            mean_copy,
            GS_BASELINE_MINI75_FEATURE_COUNT * sizeof(double)
        ) != 0 ||
        memcmp(
            stddev,
            stddev_copy,
            GS_BASELINE_MINI75_FEATURE_COUNT * sizeof(double)
        ) != 0 ||
        memcmp(
            constant_mask,
            constant_mask_copy,
            GS_BASELINE_MINI75_FEATURE_COUNT * sizeof(unsigned char)
        ) != 0 ||
        !gs_test_values_are_finite(
            train_standardized,
            GS_TEST_TRAIN_ROWS * GS_BASELINE_MINI75_FEATURE_COUNT
        ) ||
        !gs_test_values_are_finite(
            validation_standardized,
            GS_TEST_VALIDATION_ROWS *
                GS_BASELINE_MINI75_FEATURE_COUNT
        )) {

        return 1;
    }

    for (feature = 0;
         feature < GS_BASELINE_MINI75_FEATURE_COUNT;
         feature++) {

        size_t row;

        if (constant_mask[feature] != 0u) {
            for (row = 0; row < GS_TEST_TRAIN_ROWS; row++) {
                if (train_standardized[
                        row * GS_BASELINE_MINI75_FEATURE_COUNT + feature
                    ] != 0.0f) {

                    return 1;
                }
            }

            for (row = 0; row < GS_TEST_VALIDATION_ROWS; row++) {
                if (validation_standardized[
                        row * GS_BASELINE_MINI75_FEATURE_COUNT + feature
                    ] != 0.0f) {

                    return 1;
                }
            }
        } else {
            double standardized_mean = 0.0;
            double standardized_variance = 0.0;

            nonconstant_count++;

            for (row = 0; row < GS_TEST_TRAIN_ROWS; row++) {
                standardized_mean +=
                    (double)train_standardized[
                        row * GS_BASELINE_MINI75_FEATURE_COUNT + feature
                    ];
            }
            standardized_mean /= (double)GS_TEST_TRAIN_ROWS;

            for (row = 0; row < GS_TEST_TRAIN_ROWS; row++) {
                double difference =
                    (double)train_standardized[
                        row * GS_BASELINE_MINI75_FEATURE_COUNT + feature
                    ] - standardized_mean;

                standardized_variance += difference * difference;
            }
            standardized_variance /= (double)GS_TEST_TRAIN_ROWS;

            if (fabs(standardized_mean) >
                    GS_TEST_STANDARDIZED_TOLERANCE ||
                fabs(sqrt(standardized_variance) - 1.0) >
                    GS_TEST_STANDARDIZED_TOLERANCE) {

                return 1;
            }
        }
    }

    return nonconstant_count == 0 ? 1 : 0;
}


static int gs_test_invalid_ranges_and_buffers(
    const char *data_root,
    const GSEventManifest *events,
    const GSTensorManifest *tensors,
    const GSSavedSplit *split,
    float *tensor_buffer,
    float *raw_features
)
{
    if (gs_baseline_mini75_extract_split_rows(
            data_root, events, tensors, split,
            split->count + 1, 0,
            tensor_buffer, GS_BASELINE_MINI75_INPUT_COUNT,
            raw_features, GS_BASELINE_MINI75_FEATURE_COUNT
        ) == GS_BASELINE_MINI75_PIPELINE_SUCCESS ||

        gs_baseline_mini75_extract_split_rows(
            data_root, events, tensors, split,
            split->count, 1,
            tensor_buffer, GS_BASELINE_MINI75_INPUT_COUNT,
            raw_features, GS_BASELINE_MINI75_FEATURE_COUNT
        ) == GS_BASELINE_MINI75_PIPELINE_SUCCESS ||

        gs_baseline_mini75_extract_split_rows(
            data_root, events, tensors, split,
            0, 1,
            tensor_buffer, GS_BASELINE_MINI75_INPUT_COUNT - 1,
            raw_features, GS_BASELINE_MINI75_FEATURE_COUNT
        ) == GS_BASELINE_MINI75_PIPELINE_SUCCESS ||

        gs_baseline_mini75_extract_split_rows(
            data_root, events, tensors, split,
            0, 1,
            tensor_buffer, GS_BASELINE_MINI75_INPUT_COUNT,
            raw_features, GS_BASELINE_MINI75_FEATURE_COUNT - 1
        ) == GS_BASELINE_MINI75_PIPELINE_SUCCESS ||

        gs_baseline_mini75_extract_split_rows(
            NULL, events, tensors, split,
            0, 1,
            tensor_buffer, GS_BASELINE_MINI75_INPUT_COUNT,
            raw_features, GS_BASELINE_MINI75_FEATURE_COUNT
        ) == GS_BASELINE_MINI75_PIPELINE_SUCCESS ||

        gs_baseline_mini75_extract_split_rows(
            data_root, NULL, tensors, split,
            0, 1,
            tensor_buffer, GS_BASELINE_MINI75_INPUT_COUNT,
            raw_features, GS_BASELINE_MINI75_FEATURE_COUNT
        ) == GS_BASELINE_MINI75_PIPELINE_SUCCESS ||

        gs_baseline_mini75_extract_split_rows(
            data_root, events, NULL, split,
            0, 1,
            tensor_buffer, GS_BASELINE_MINI75_INPUT_COUNT,
            raw_features, GS_BASELINE_MINI75_FEATURE_COUNT
        ) == GS_BASELINE_MINI75_PIPELINE_SUCCESS ||

        gs_baseline_mini75_extract_split_rows(
            data_root, events, tensors, NULL,
            0, 1,
            tensor_buffer, GS_BASELINE_MINI75_INPUT_COUNT,
            raw_features, GS_BASELINE_MINI75_FEATURE_COUNT
        ) == GS_BASELINE_MINI75_PIPELINE_SUCCESS ||

        gs_baseline_mini75_extract_split_rows(
            data_root, events, tensors, split,
            0, 1,
            NULL, GS_BASELINE_MINI75_INPUT_COUNT,
            raw_features, GS_BASELINE_MINI75_FEATURE_COUNT
        ) == GS_BASELINE_MINI75_PIPELINE_SUCCESS ||

        gs_baseline_mini75_extract_split_rows(
            data_root, events, tensors, split,
            0, 1,
            tensor_buffer, GS_BASELINE_MINI75_INPUT_COUNT,
            NULL, GS_BASELINE_MINI75_FEATURE_COUNT
        ) == GS_BASELINE_MINI75_PIPELINE_SUCCESS) {

        return 1;
    }

    return 0;
}


static int gs_test_label_independence_by_interface(void)
{
    typedef int (*GSValidateSignature)(
        const GSEventManifest *,
        const GSTensorManifest *,
        const GSSavedSplit *,
        const GSSavedSplit *
    );

    typedef int (*GSExtractSignature)(
        const char *,
        const GSEventManifest *,
        const GSTensorManifest *,
        const GSSavedSplit *,
        size_t,
        size_t,
        float *,
        size_t,
        float *,
        size_t
    );

    GSValidateSignature validate_function =
        gs_baseline_mini75_validate_alignment;

    GSExtractSignature extract_function =
        gs_baseline_mini75_extract_split_rows;

    return validate_function == NULL || extract_function == NULL ? 1 : 0;
}


int main(int argc, char **argv)
{
    GSEventManifest events = {0};
    GSTensorManifest tensors = {0};
    GSSavedSplit train = {0};
    GSSavedSplit validation = {0};
    GSTensorSpec spec;

    float *tensor_buffer = NULL;
    float *train_raw_a = NULL;
    float *train_raw_b = NULL;
    float *validation_raw = NULL;
    float *single_row = NULL;
    float *train_standardized = NULL;
    float *validation_standardized = NULL;

    double *mean = NULL;
    double *stddev = NULL;
    double *mean_copy = NULL;
    double *stddev_copy = NULL;

    unsigned char *constant_mask = NULL;
    unsigned char *constant_mask_copy = NULL;

    size_t train_feature_count =
        GS_TEST_TRAIN_ROWS * GS_BASELINE_MINI75_FEATURE_COUNT;

    size_t validation_feature_count =
        GS_TEST_VALIDATION_ROWS * GS_BASELINE_MINI75_FEATURE_COUNT;

    const char *data_root;
    int rc = 1;

    if (argc != 6) {
        fprintf(stderr,
                "Uso: test_baseline_mini75_pipeline.exe "
                "manifest_events_272.csv "
                "manifest_tensors_75_central.csv "
                "split_train_indices_seed123.csv "
                "split_val_indices_seed123.csv data_root\n");
        return 1;
    }

    data_root = argv[5];

    if (gs_tensor_spec_init(&spec, 75) != 0 ||
        gs_event_manifest_load(argv[1], &events) != 0 ||
        gs_tensor_manifest_load(argv[2], &spec, &tensors) != 0 ||
        gs_saved_split_load(argv[3], "train", &events, &train) != 0 ||
        gs_saved_split_load(
            argv[4],
            "validation",
            &events,
            &validation
        ) != 0) {

        fprintf(stderr, "Errore: caricamento input di integrazione.\n");
        goto cleanup;
    }

    tensor_buffer = (float *)malloc(
        GS_BASELINE_MINI75_INPUT_COUNT * sizeof(float)
    );
    train_raw_a = (float *)malloc(train_feature_count * sizeof(float));
    train_raw_b = (float *)malloc(train_feature_count * sizeof(float));
    validation_raw = (float *)malloc(
        validation_feature_count * sizeof(float)
    );
    single_row = (float *)malloc(
        GS_BASELINE_MINI75_FEATURE_COUNT * sizeof(float)
    );
    train_standardized = (float *)malloc(
        train_feature_count * sizeof(float)
    );
    validation_standardized = (float *)malloc(
        validation_feature_count * sizeof(float)
    );

    mean = (double *)malloc(
        GS_BASELINE_MINI75_FEATURE_COUNT * sizeof(double)
    );
    stddev = (double *)malloc(
        GS_BASELINE_MINI75_FEATURE_COUNT * sizeof(double)
    );
    mean_copy = (double *)malloc(
        GS_BASELINE_MINI75_FEATURE_COUNT * sizeof(double)
    );
    stddev_copy = (double *)malloc(
        GS_BASELINE_MINI75_FEATURE_COUNT * sizeof(double)
    );

    constant_mask = (unsigned char *)malloc(
        GS_BASELINE_MINI75_FEATURE_COUNT * sizeof(unsigned char)
    );
    constant_mask_copy = (unsigned char *)malloc(
        GS_BASELINE_MINI75_FEATURE_COUNT * sizeof(unsigned char)
    );

    if (tensor_buffer == NULL ||
        train_raw_a == NULL ||
        train_raw_b == NULL ||
        validation_raw == NULL ||
        single_row == NULL ||
        train_standardized == NULL ||
        validation_standardized == NULL ||
        mean == NULL ||
        stddev == NULL ||
        mean_copy == NULL ||
        stddev_copy == NULL ||
        constant_mask == NULL ||
        constant_mask_copy == NULL) {

        fprintf(stderr, "Errore: memoria insufficiente per il test.\n");
        goto cleanup;
    }

    if (gs_baseline_mini75_validate_alignment(
            &events,
            &tensors,
            &train,
            &validation
        ) != GS_BASELINE_MINI75_PIPELINE_SUCCESS) {

        fprintf(stderr, "Errore: allineamento ufficiale non valido.\n");
        goto cleanup;
    }
    printf("[PASS] official_counts_and_alignment\n");

    if (gs_test_extract_and_check_order(
            data_root,
            &events,
            &tensors,
            &train,
            GS_TEST_TRAIN_ROWS,
            tensor_buffer,
            train_raw_a,
            single_row
        ) != 0) {

        fprintf(stderr, "Errore: estrazione prime righe train.\n");
        goto cleanup;
    }
    printf("[PASS] extract_first_8_train_rows\n");

    if (gs_test_extract_and_check_order(
            data_root,
            &events,
            &tensors,
            &validation,
            GS_TEST_VALIDATION_ROWS,
            tensor_buffer,
            validation_raw,
            single_row
        ) != 0) {

        fprintf(stderr, "Errore: estrazione prime righe validation.\n");
        goto cleanup;
    }
    printf("[PASS] extract_first_4_validation_rows\n");

    if (gs_baseline_mini75_extract_split_rows(
            data_root,
            &events,
            &tensors,
            &train,
            0,
            GS_TEST_TRAIN_ROWS,
            tensor_buffer,
            GS_BASELINE_MINI75_INPUT_COUNT,
            train_raw_b,
            train_feature_count
        ) != GS_BASELINE_MINI75_PIPELINE_SUCCESS ||
        memcmp(
            train_raw_a,
            train_raw_b,
            train_feature_count * sizeof(float)
        ) != 0) {

        fprintf(stderr, "Errore: estrazione non deterministica.\n");
        goto cleanup;
    }
    printf("[PASS] deterministic_repeat\n");

    if (gs_test_train_only_standardization(
            train_raw_a,
            validation_raw,
            train_standardized,
            validation_standardized,
            mean,
            stddev,
            constant_mask,
            mean_copy,
            stddev_copy,
            constant_mask_copy
        ) != 0) {

        fprintf(stderr, "Errore: standardizzazione train-only reale.\n");
        goto cleanup;
    }
    printf("[PASS] train_only_standardization_real_rows\n");

    if (gs_test_invalid_ranges_and_buffers(
            data_root,
            &events,
            &tensors,
            &train,
            tensor_buffer,
            single_row
        ) != 0) {

        fprintf(stderr, "Errore: gestione range/buffer invalidi.\n");
        goto cleanup;
    }
    printf("[PASS] invalid_ranges_and_buffers\n");

    if (gs_test_label_independence_by_interface() != 0) {
        fprintf(stderr, "Errore: interfaccia pipeline non valida.\n");
        goto cleanup;
    }
    printf("[PASS] label_independence_by_interface\n");

    printf("TRAIN_ROWS_TESTED=%d\n", GS_TEST_TRAIN_ROWS);
    printf("VALIDATION_ROWS_TESTED=%d\n", GS_TEST_VALIDATION_ROWS);
    printf("FEATURE_DIM=%d\n", GS_BASELINE_MINI75_FEATURE_COUNT);
    printf("ALL TESTS PASSED\n");

    rc = 0;

cleanup:
    free(constant_mask_copy);
    free(constant_mask);
    free(stddev_copy);
    free(mean_copy);
    free(stddev);
    free(mean);
    free(validation_standardized);
    free(train_standardized);
    free(single_row);
    free(validation_raw);
    free(train_raw_b);
    free(train_raw_a);
    free(tensor_buffer);

    gs_saved_split_free(&validation);
    gs_saved_split_free(&train);
    gs_tensor_manifest_free(&tensors);
    gs_event_manifest_free(&events);

    return rc;
}
