#include "vae_tensor_manifest.h"
#include "vae_tensor_io.h"
#include "vae_event_dataset.h"

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>


static void compute_stats(
    const float *values,
    size_t count,
    float *min_value,
    float *max_value,
    double *mean_value
)
{
    size_t i;
    double sum;

    *min_value = FLT_MAX;
    *max_value = -FLT_MAX;
    sum = 0.0;

    for (i = 0; i < count; i++) {
        float value;

        value = values[i];

        if (value < *min_value) {
            *min_value = value;
        }

        if (value > *max_value) {
            *max_value = value;
        }

        sum += (double)value;
    }

    *mean_value =
        sum / (double)count;
}


static int test_one_manifest(
    const char *manifest_path,
    const GSTensorSpec *spec,
    const GSEventManifest *event_manifest,
    GSTensorManifest *tensor_manifest
)
{
    size_t i;
    size_t alignment_errors;

    const GSTensorManifestRecord *sample_record;

    float *sample;

    float min_value;
    float max_value;
    double mean_value;

    int rc;

    rc = gs_tensor_manifest_load(
        manifest_path,
        spec,
        tensor_manifest
    );

    if (rc != 0) {
        return 1;
    }

    if (tensor_manifest->count !=
        event_manifest->count) {

        printf("Errore: conteggio manifest diverso.\n");
        return 1;
    }

    alignment_errors = 0;

    for (i = 0;
         i < tensor_manifest->count;
         i++) {

        const GSTensorManifestRecord *tensor_record;
        const GSEventRecord *event_record;

        tensor_record =
            gs_tensor_manifest_get(
                tensor_manifest,
                i
            );

        event_record =
            gs_event_manifest_get(
                event_manifest,
                i
            );

        if (tensor_record == NULL ||
            event_record == NULL) {

            alignment_errors++;
            continue;
        }

        if (tensor_record->event_index !=
                event_record->event_index ||

            strcmp(
                tensor_record->event_id,
                event_record->event_id
            ) != 0) {

            alignment_errors++;
        }
    }

    printf("Manifest %s:\n",
           gs_tensor_spec_name(spec));

    printf("  record              = %lu\n",
           (unsigned long)tensor_manifest->count);

    printf("  ultima colonna      = %s\n",
           tensor_manifest->extra_column_name);

    printf("  errori allineamento = %lu\n",
           (unsigned long)alignment_errors);

    if (alignment_errors != 0) {
        return 1;
    }

    sample_record =
        gs_tensor_manifest_get(
            tensor_manifest,
            0
        );

    if (sample_record == NULL) {
        return 1;
    }

    sample = (float *)malloc(
        spec->bytes_f32
    );

    if (sample == NULL) {
        printf("Errore: memoria insufficiente.\n");
        return 1;
    }

    rc = gs_tensor_read_float01(
        sample_record->tensor_path,
        spec,
        sample
    );

    if (rc != 0) {
        free(sample);
        return 1;
    }

    compute_stats(
        sample,
        spec->values_per_event,
        &min_value,
        &max_value,
        &mean_value
    );

    printf("  sample path         = %s\n",
           sample_record->tensor_path);

    printf("  shape               = %d x %d x %d\n",
           spec->channels,
           spec->height,
           spec->width);

    printf("  valori              = %lu\n",
           (unsigned long)spec->values_per_event);

    printf("  min/max/mean        = %.6f / %.6f / %.6f\n\n",
           min_value,
           max_value,
           mean_value);

    free(sample);

    if (min_value < 0.0f ||
        max_value > 1.0f) {

        return 1;
    }

    return 0;
}


int main(void)
{
    GSEventManifest event_manifest;

    GSTensorSpec spec224;
    GSTensorSpec spec112;
    GSTensorSpec spec75;

    GSTensorManifest manifest224;
    GSTensorManifest manifest112;
    GSTensorManifest manifest75;

    int rc;

    manifest224.records = NULL;
    manifest112.records = NULL;
    manifest75.records = NULL;

    rc = gs_event_manifest_load(
        "manifest_events_272.csv",
        &event_manifest
    );

    if (rc != 0) {
        return 1;
    }

    if (gs_tensor_spec_init(&spec224, 224) != 0 ||
        gs_tensor_spec_init(&spec112, 112) != 0 ||
        gs_tensor_spec_init(&spec75, 75) != 0) {

        gs_event_manifest_free(&event_manifest);
        return 1;
    }

    rc = test_one_manifest(
        "manifest_tensors_224_central.csv",
        &spec224,
        &event_manifest,
        &manifest224
    );

    if (rc == 0) {
        rc = test_one_manifest(
            "manifest_tensors_112_central.csv",
            &spec112,
            &event_manifest,
            &manifest112
        );
    }

    if (rc == 0) {
        rc = test_one_manifest(
            "manifest_tensors_75_central.csv",
            &spec75,
            &event_manifest,
            &manifest75
        );
    }

    if (rc != 0) {
        if (manifest224.records != NULL) {
            gs_tensor_manifest_free(&manifest224);
        }

        if (manifest112.records != NULL) {
            gs_tensor_manifest_free(&manifest112);
        }

        if (manifest75.records != NULL) {
            gs_tensor_manifest_free(&manifest75);
        }

        gs_event_manifest_free(&event_manifest);

        printf("ERRORE: test manifest/IO fallito.\n");
        return 1;
    }

    gs_tensor_manifest_free(&manifest75);
    gs_tensor_manifest_free(&manifest112);
    gs_tensor_manifest_free(&manifest224);

    gs_event_manifest_free(&event_manifest);

    printf("OK: manifest e lettura .bin multirisoluzione verificati.\n");
    printf("La validation potra' usare i .bin centrali 224, 112 e 75.\n");

    return 0;
}