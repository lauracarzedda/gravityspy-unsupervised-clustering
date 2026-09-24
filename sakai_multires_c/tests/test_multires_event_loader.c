#include "vae_multires_event_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>


static int read_exact_u8(
    const char *path,
    uint8_t *buffer,
    size_t expected_size
)
{
    FILE *fp;
    size_t nread;
    int extra;

    fp = fopen(path, "rb");

    if (fp == NULL) {
        printf("Errore: impossibile aprire %s\n", path);
        return 1;
    }

    nread = fread(
        buffer,
        1,
        expected_size,
        fp
    );

    extra = fgetc(fp);

    fclose(fp);

    if (nread != expected_size || extra != EOF) {
        printf("Errore: dimensione non valida per %s\n",
               path);

        return 1;
    }

    return 0;
}


static size_t compare_u8(
    const uint8_t *a,
    const uint8_t *b,
    size_t count,
    int *max_difference
)
{
    size_t i;
    size_t different;

    different = 0;
    *max_difference = 0;

    for (i = 0; i < count; i++) {
        int difference;

        difference = (int)a[i] - (int)b[i];

        if (difference < 0) {
            difference = -difference;
        }

        if (difference != 0) {
            different++;
        }

        if (difference > *max_difference) {
            *max_difference = difference;
        }
    }

    return different;
}


static void compute_float_stats(
    const float *x,
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

        value = x[i];

        if (value < *min_value) {
            *min_value = value;
        }

        if (value > *max_value) {
            *max_value = value;
        }

        sum += (double)value;
    }

    *mean_value = sum / (double)count;
}


static int test_resolution(
    const GSEventRecord *record,
    int target_side,
    const char *official_bin_path
)
{
    GSMultiresEventLoader loader;
    GSAugmentRng rng;

    const GSTensorSpec *spec;

    uint8_t *generated_central;
    uint8_t *official_central;

    float *random_tensor;
    float *central_tensor;

    size_t different_bytes;
    size_t changed_values;

    int max_difference;
    int crop_left;

    float min_value;
    float max_value;
    double mean_value;

    size_t i;
    int rc;

    rc = gs_multires_event_loader_init(
        &loader,
        target_side
    );

    if (rc != 0) {
        return 1;
    }

    spec = gs_multires_event_loader_spec(&loader);

    if (spec == NULL) {
        gs_multires_event_loader_free(&loader);
        return 1;
    }

    generated_central = (uint8_t *)malloc(
        spec->bytes_u8
    );

    official_central = (uint8_t *)malloc(
        spec->bytes_u8
    );

    random_tensor = (float *)malloc(
        spec->bytes_f32
    );

    central_tensor = (float *)malloc(
        spec->bytes_f32
    );

    if (generated_central == NULL ||
        official_central == NULL ||
        random_tensor == NULL ||
        central_tensor == NULL) {

        printf("Errore: memoria insufficiente.\n");

        free(generated_central);
        free(official_central);
        free(random_tensor);
        free(central_tensor);

        gs_multires_event_loader_free(&loader);

        return 1;
    }

    /*
       Verifica completa:

       PNG 272
       -> crop centrale 224
       -> resize target
       -> confronto con il .bin ufficiale.
    */
    rc = gs_multires_event_load_central_u8(
        &loader,
        record,
        generated_central
    );

    if (rc == 0) {
        rc = read_exact_u8(
            official_bin_path,
            official_central,
            spec->bytes_u8
        );
    }

    if (rc != 0) {
        free(generated_central);
        free(official_central);
        free(random_tensor);
        free(central_tensor);

        gs_multires_event_loader_free(&loader);

        return 1;
    }

    different_bytes = compare_u8(
        generated_central,
        official_central,
        spec->values_per_event,
        &max_difference
    );

    /*
       Test di un campione perturbato ridotto.
       Lo stesso seed viene reinizializzato per ogni risoluzione,
       quindi il crop_left deve essere uguale.
    */
    gs_augment_rng_seed(&rng, 456u);

    rc = gs_multires_event_load_random_float01(
        &loader,
        record,
        &rng,
        &crop_left,
        random_tensor
    );

    if (rc == 0) {
        rc = gs_multires_event_load_central_float01(
            &loader,
            record,
            central_tensor
        );
    }

    if (rc != 0) {
        free(generated_central);
        free(official_central);
        free(random_tensor);
        free(central_tensor);

        gs_multires_event_loader_free(&loader);

        return 1;
    }

    changed_values = 0;

    for (i = 0; i < spec->values_per_event; i++) {
        if (random_tensor[i] != central_tensor[i]) {
            changed_values++;
        }
    }

    compute_float_stats(
        random_tensor,
        spec->values_per_event,
        &min_value,
        &max_value,
        &mean_value
    );

    printf("Risoluzione %d x %d:\n",
           target_side,
           target_side);

    printf("  valori per evento       = %lu\n",
           (unsigned long)spec->values_per_event);

    printf("  byte centrali diversi   = %lu\n",
           (unsigned long)different_bytes);

    printf("  differenza massima      = %d\n",
           max_difference);

    printf("  crop casuale usato      = %d\n",
           crop_left);

    printf("  valori random/centrale diversi = %lu su %lu\n",
           (unsigned long)changed_values,
           (unsigned long)spec->values_per_event);

    printf("  random min/max/mean     = %.6f / %.6f / %.6f\n\n",
           min_value,
           max_value,
           mean_value);

    free(generated_central);
    free(official_central);
    free(random_tensor);
    free(central_tensor);

    gs_multires_event_loader_free(&loader);

    if (different_bytes != 0 ||
        max_difference != 0 ||
        crop_left < 0 ||
        crop_left > 24 ||
        min_value < 0.0f ||
        max_value > 1.0f) {

        return 1;
    }

    return 0;
}


int main(int argc, char **argv)
{
    const char *manifest_path;

    GSEventManifest manifest;
    const GSEventRecord *record;

    int rc;

    manifest_path = "manifest_events_272.csv";

    if (argc >= 2) {
        manifest_path = argv[1];
    }

    rc = gs_event_manifest_load(
        manifest_path,
        &manifest
    );

    if (rc != 0) {
        return 1;
    }

    record = gs_event_manifest_get(
        &manifest,
        0
    );

    if (record == NULL) {
        gs_event_manifest_free(&manifest);
        return 1;
    }

    printf("Evento di test:\n");
    printf("  event_index = %d\n", record->event_index);
    printf("  event_id    = %s\n\n", record->event_id);

    rc = test_resolution(
        record,
        224,
        "tensors_224_central_uint8/event_000001.bin"
    );

    if (rc == 0) {
        rc = test_resolution(
            record,
            112,
            "tensors_112_central_uint8/event_000001.bin"
        );
    }

    if (rc == 0) {
        rc = test_resolution(
            record,
            75,
            "tensors_75_central_uint8/event_000001.bin"
        );
    }

    gs_event_manifest_free(&manifest);

    if (rc != 0) {
        printf("ERRORE: test multirisoluzione fallito.\n");
        return 1;
    }

    printf(
        "OK: loader evento multirisoluzione verificato.\n"
    );

    printf(
        "Le perturbate 224, 112 e 75 possono essere "
        "generate dinamicamente in RAM.\n"
    );

    return 0;
}