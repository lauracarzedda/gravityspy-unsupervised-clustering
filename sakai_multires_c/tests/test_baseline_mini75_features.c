#include "baseline_mini75_features.h"
#include "vae_tensor_io.h"
#include "vae_tensor_manifest.h"
#include "vae_tensor_spec.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define GS_TEST_TOLERANCE 1.0e-6


static int gs_test_nearly_equal(
    float actual,
    double expected
)
{
    double difference = (double)actual - expected;

    if (difference < 0.0) {
        difference = -difference;
    }

    return difference <= GS_TEST_TOLERANCE;
}


static int test_constants_and_dimensions(void)
{
    if (GS_BASELINE_MINI75_CHANNELS != 4 ||
        GS_BASELINE_MINI75_SIDE != 75 ||
        GS_BASELINE_MINI75_POOL_SIDE != 5 ||
        GS_BASELINE_MINI75_POOLED_SIDE != 15 ||
        GS_BASELINE_MINI75_INPUT_COUNT != 22500 ||
        GS_BASELINE_MINI75_FEATURE_COUNT != 900) {

        fprintf(stderr, "Errore: costanti mini75 non valide.\n");
        return 1;
    }

    return 0;
}


static int test_constant_per_channel(void)
{
    const float channel_values[GS_BASELINE_MINI75_CHANNELS] = {
        0.125f,
        0.250f,
        0.500f,
        0.750f
    };

    float *input;
    float *features;
    size_t channel;
    size_t index;
    int rc = 1;

    input = (float *)malloc(
        GS_BASELINE_MINI75_INPUT_COUNT * sizeof(float)
    );

    features = (float *)malloc(
        GS_BASELINE_MINI75_FEATURE_COUNT * sizeof(float)
    );

    if (input == NULL || features == NULL) {
        fprintf(stderr, "Errore: allocazione test costanti.\n");
        goto cleanup;
    }

    for (channel = 0;
         channel < GS_BASELINE_MINI75_CHANNELS;
         channel++) {

        size_t channel_offset =
            channel * GS_BASELINE_MINI75_SIDE * GS_BASELINE_MINI75_SIDE;

        for (index = 0;
             index < GS_BASELINE_MINI75_SIDE * GS_BASELINE_MINI75_SIDE;
             index++) {

            input[channel_offset + index] = channel_values[channel];
        }
    }

    if (gs_baseline_mini75_pool5_extract(
            input,
            GS_BASELINE_MINI75_INPUT_COUNT,
            features,
            GS_BASELINE_MINI75_FEATURE_COUNT
        ) != GS_BASELINE_MINI75_SUCCESS) {

        fprintf(stderr, "Errore: estrazione test costanti.\n");
        goto cleanup;
    }

    for (channel = 0;
         channel < GS_BASELINE_MINI75_CHANNELS;
         channel++) {

        size_t feature_offset =
            channel *
            GS_BASELINE_MINI75_POOLED_SIDE *
            GS_BASELINE_MINI75_POOLED_SIDE;

        for (index = 0;
             index < GS_BASELINE_MINI75_POOLED_SIDE *
                     GS_BASELINE_MINI75_POOLED_SIDE;
             index++) {

            if (!gs_test_nearly_equal(
                    features[feature_offset + index],
                    (double)channel_values[channel]
                )) {

                fprintf(stderr,
                        "Errore: ordine/valore canale %lu, feature %lu.\n",
                        (unsigned long)channel,
                        (unsigned long)index);
                goto cleanup;
            }
        }
    }

    rc = 0;

cleanup:
    free(features);
    free(input);
    return rc;
}


static int test_spatial_ramp_all_features(void)
{
    float *input;
    float *features;
    size_t index;
    size_t channel;
    size_t pooled_row;
    size_t pooled_col;
    int rc = 1;

    input = (float *)malloc(
        GS_BASELINE_MINI75_INPUT_COUNT * sizeof(float)
    );

    features = (float *)malloc(
        GS_BASELINE_MINI75_FEATURE_COUNT * sizeof(float)
    );

    if (input == NULL || features == NULL) {
        fprintf(stderr, "Errore: allocazione test rampa.\n");
        goto cleanup;
    }

    for (index = 0;
         index < GS_BASELINE_MINI75_INPUT_COUNT;
         index++) {

        input[index] = (float)((double)index / 22499.0);
    }

    if (gs_baseline_mini75_pool5_extract(
            input,
            GS_BASELINE_MINI75_INPUT_COUNT,
            features,
            GS_BASELINE_MINI75_FEATURE_COUNT
        ) != GS_BASELINE_MINI75_SUCCESS) {

        fprintf(stderr, "Errore: estrazione test rampa.\n");
        goto cleanup;
    }

    for (channel = 0;
         channel < GS_BASELINE_MINI75_CHANNELS;
         channel++) {

        for (pooled_row = 0;
             pooled_row < GS_BASELINE_MINI75_POOLED_SIDE;
             pooled_row++) {

            for (pooled_col = 0;
                 pooled_col < GS_BASELINE_MINI75_POOLED_SIDE;
                 pooled_col++) {

                size_t feature_index =
                    channel * 225 + pooled_row * 15 + pooled_col;

                double expected =
                    ((double)(channel * 5625) +
                     (double)(pooled_row * 5 + 2) * 75.0 +
                     (double)(pooled_col * 5 + 2)) /
                    22499.0;

                if (!gs_test_nearly_equal(
                        features[feature_index],
                        expected
                    )) {

                    fprintf(stderr,
                            "Errore: rampa feature %lu.\n",
                            (unsigned long)feature_index);
                    goto cleanup;
                }
            }
        }
    }

    rc = 0;

cleanup:
    free(features);
    free(input);
    return rc;
}


static int test_invalid_arguments(void)
{
    float *input;
    float *features;
    int rc = 1;

    input = (float *)malloc(
        GS_BASELINE_MINI75_INPUT_COUNT * sizeof(float)
    );

    features = (float *)malloc(
        GS_BASELINE_MINI75_FEATURE_COUNT * sizeof(float)
    );

    if (input == NULL || features == NULL) {
        fprintf(stderr, "Errore: allocazione test argomenti.\n");
        goto cleanup;
    }

    if (gs_baseline_mini75_pool5_extract(
            NULL,
            GS_BASELINE_MINI75_INPUT_COUNT,
            features,
            GS_BASELINE_MINI75_FEATURE_COUNT
        ) != GS_BASELINE_MINI75_ERROR_NULL_ARGUMENT ||

        gs_baseline_mini75_pool5_extract(
            input,
            GS_BASELINE_MINI75_INPUT_COUNT,
            NULL,
            GS_BASELINE_MINI75_FEATURE_COUNT
        ) != GS_BASELINE_MINI75_ERROR_NULL_ARGUMENT ||

        gs_baseline_mini75_pool5_extract(
            input,
            GS_BASELINE_MINI75_INPUT_COUNT - 1,
            features,
            GS_BASELINE_MINI75_FEATURE_COUNT
        ) != GS_BASELINE_MINI75_ERROR_COUNT ||

        gs_baseline_mini75_pool5_extract(
            input,
            GS_BASELINE_MINI75_INPUT_COUNT,
            features,
            GS_BASELINE_MINI75_FEATURE_COUNT - 1
        ) != GS_BASELINE_MINI75_ERROR_COUNT) {

        fprintf(stderr, "Errore: gestione argomenti invalidi.\n");
        goto cleanup;
    }

    rc = 0;

cleanup:
    free(features);
    free(input);
    return rc;
}


static int test_input_not_modified(void)
{
    float *input;
    float *input_copy;
    float *features;
    size_t index;
    int rc = 1;

    input = (float *)malloc(
        GS_BASELINE_MINI75_INPUT_COUNT * sizeof(float)
    );

    input_copy = (float *)malloc(
        GS_BASELINE_MINI75_INPUT_COUNT * sizeof(float)
    );

    features = (float *)malloc(
        GS_BASELINE_MINI75_FEATURE_COUNT * sizeof(float)
    );

    if (input == NULL || input_copy == NULL || features == NULL) {
        fprintf(stderr, "Errore: allocazione test input immutabile.\n");
        goto cleanup;
    }

    for (index = 0;
         index < GS_BASELINE_MINI75_INPUT_COUNT;
         index++) {

        input[index] = (float)(index % 257) / 256.0f;
    }

    memcpy(
        input_copy,
        input,
        GS_BASELINE_MINI75_INPUT_COUNT * sizeof(float)
    );

    if (gs_baseline_mini75_pool5_extract(
            input,
            GS_BASELINE_MINI75_INPUT_COUNT,
            features,
            GS_BASELINE_MINI75_FEATURE_COUNT
        ) != GS_BASELINE_MINI75_SUCCESS) {

        fprintf(stderr, "Errore: estrazione test input immutabile.\n");
        goto cleanup;
    }

    if (memcmp(
            input,
            input_copy,
            GS_BASELINE_MINI75_INPUT_COUNT * sizeof(float)
        ) != 0) {

        fprintf(stderr, "Errore: l'estrattore ha modificato l'input.\n");
        goto cleanup;
    }

    rc = 0;

cleanup:
    free(features);
    free(input_copy);
    free(input);
    return rc;
}


static int test_real_mini75_integration(
    const char *manifest_path
)
{
    GSTensorSpec spec;
    GSTensorManifest manifest = {0};
    const GSTensorManifestRecord *record;
    float *input = NULL;
    float *input_copy = NULL;
    float *features_a = NULL;
    float *features_b = NULL;
    size_t index;
    int rc = 1;

    if (manifest_path == NULL ||
        gs_tensor_spec_init(&spec, 75) != 0 ||
        spec.values_per_event != GS_BASELINE_MINI75_INPUT_COUNT ||
        gs_tensor_manifest_load(manifest_path, &spec, &manifest) != 0) {

        fprintf(stderr, "Errore: inizializzazione integrazione mini75.\n");
        goto cleanup;
    }

    record = gs_tensor_manifest_get(&manifest, 0);
    if (record == NULL) {
        fprintf(stderr, "Errore: primo record mini75 assente.\n");
        goto cleanup;
    }

    input = (float *)malloc(
        GS_BASELINE_MINI75_INPUT_COUNT * sizeof(float)
    );

    input_copy = (float *)malloc(
        GS_BASELINE_MINI75_INPUT_COUNT * sizeof(float)
    );

    features_a = (float *)malloc(
        GS_BASELINE_MINI75_FEATURE_COUNT * sizeof(float)
    );

    features_b = (float *)malloc(
        GS_BASELINE_MINI75_FEATURE_COUNT * sizeof(float)
    );

    if (input == NULL || input_copy == NULL ||
        features_a == NULL || features_b == NULL) {

        fprintf(stderr, "Errore: allocazione integrazione mini75.\n");
        goto cleanup;
    }

    if (gs_tensor_read_float01(
            record->tensor_path,
            &spec,
            input
        ) != 0) {

        fprintf(stderr, "Errore: lettura primo tensore mini75.\n");
        goto cleanup;
    }

    for (index = 0;
         index < GS_BASELINE_MINI75_INPUT_COUNT;
         index++) {

        if (!isfinite(input[index]) ||
            input[index] < 0.0f ||
            input[index] > 1.0f) {

            fprintf(stderr,
                    "Errore: input reale non finito o fuori [0,1].\n");
            goto cleanup;
        }
    }

    memcpy(
        input_copy,
        input,
        GS_BASELINE_MINI75_INPUT_COUNT * sizeof(float)
    );

    if (gs_baseline_mini75_pool5_extract(
            input,
            GS_BASELINE_MINI75_INPUT_COUNT,
            features_a,
            GS_BASELINE_MINI75_FEATURE_COUNT
        ) != GS_BASELINE_MINI75_SUCCESS ||

        gs_baseline_mini75_pool5_extract(
            input,
            GS_BASELINE_MINI75_INPUT_COUNT,
            features_b,
            GS_BASELINE_MINI75_FEATURE_COUNT
        ) != GS_BASELINE_MINI75_SUCCESS) {

        fprintf(stderr, "Errore: estrazione integrazione mini75.\n");
        goto cleanup;
    }

    if (memcmp(
            features_a,
            features_b,
            GS_BASELINE_MINI75_FEATURE_COUNT * sizeof(float)
        ) != 0) {

        fprintf(stderr, "Errore: estrazioni reali non deterministiche.\n");
        goto cleanup;
    }

    if (memcmp(
            input,
            input_copy,
            GS_BASELINE_MINI75_INPUT_COUNT * sizeof(float)
        ) != 0) {

        fprintf(stderr, "Errore: input reale modificato.\n");
        goto cleanup;
    }

    for (index = 0;
         index < GS_BASELINE_MINI75_FEATURE_COUNT;
         index++) {

        if (!isfinite(features_a[index]) ||
            features_a[index] < 0.0f ||
            features_a[index] > 1.0f) {

            fprintf(stderr,
                    "Errore: feature reale non finita o fuori [0,1].\n");
            goto cleanup;
        }
    }

    rc = 0;

cleanup:
    free(features_b);
    free(features_a);
    free(input_copy);
    free(input);
    gs_tensor_manifest_free(&manifest);
    return rc;
}


int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr,
                "Uso: test_baseline_mini75_features.exe "
                "manifest_tensors_75_central.csv\n");
        return 1;
    }

    if (test_constants_and_dimensions() != 0) {
        return 1;
    }
    printf("[PASS] constants_and_dimensions\n");

    if (test_constant_per_channel() != 0) {
        return 1;
    }
    printf("[PASS] constant_per_channel\n");

    if (test_spatial_ramp_all_features() != 0) {
        return 1;
    }
    printf("[PASS] spatial_ramp_all_900_features\n");

    if (test_invalid_arguments() != 0) {
        return 1;
    }
    printf("[PASS] invalid_arguments\n");

    if (test_input_not_modified() != 0) {
        return 1;
    }
    printf("[PASS] input_not_modified\n");

    if (test_real_mini75_integration(argv[1]) != 0) {
        return 1;
    }
    printf("[PASS] real_mini75_integration\n");

    printf("FEATURE_DIM=%d\n", GS_BASELINE_MINI75_FEATURE_COUNT);
    printf("ALL TESTS PASSED\n");

    return 0;
}
