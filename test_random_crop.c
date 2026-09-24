#include "vae_event_dataset.h"
#include "vae_event_loader.h"
#include "vae_augmentation.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>


#define GS_TEST_MAX_CROP_VALUES 49
#define GS_TEST_KNOWN_SEQUENCE_COUNT 12


static const int GS_LEGACY_SEED456_SEQUENCE[
    GS_TEST_KNOWN_SEQUENCE_COUNT
] = {
    0, 20, 20, 2, 0, 14, 11, 16, 16, 5, 17, 2
};


static void compute_stats(
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
        float v;

        v = x[i];

        if (v < *min_value) {
            *min_value = v;
        }

        if (v > *max_value) {
            *max_value = v;
        }

        sum += (double)v;
    }

    *mean_value = sum / (double)count;
}


static int test_mode_sequence(
    GSAugmentMode mode,
    uint32_t seed,
    size_t draws
)
{
    GSAugmentRng rng_a;
    GSAugmentRng rng_b;

    int min_crop_left;
    int max_crop_left;
    int seen[GS_TEST_MAX_CROP_VALUES];

    size_t i;
    size_t unique_count;
    size_t out_of_range_count;

    int observed_min;
    int observed_max;

    if (gs_augment_mode_crop_range(
            mode,
            &min_crop_left,
            &max_crop_left
        ) != 0) {

        return 1;
    }

    if (min_crop_left < 0 ||
        max_crop_left >= GS_TEST_MAX_CROP_VALUES) {

        printf("Errore: intervallo non gestibile dal test.\n");
        return 1;
    }

    for (i = 0; i < GS_TEST_MAX_CROP_VALUES; i++) {
        seen[i] = 0;
    }

    gs_augment_rng_seed(&rng_a, seed);
    gs_augment_rng_seed(&rng_b, seed);

    unique_count = 0;
    out_of_range_count = 0;
    observed_min = max_crop_left;
    observed_max = min_crop_left;

    printf("Modalità: %s\n", gs_augment_mode_name(mode));
    printf("Intervallo atteso: %d-%d\n",
           min_crop_left,
           max_crop_left);

    printf("Prime estrazioni:\n");

    for (i = 0; i < draws; i++) {
        int crop_a;
        int crop_b;

        if (gs_augment_random_crop_left_mode(
                &rng_a,
                mode,
                &crop_a
            ) != 0) {

            return 1;
        }

        if (gs_augment_random_crop_left_mode(
                &rng_b,
                mode,
                &crop_b
            ) != 0) {

            return 1;
        }

        if (crop_a != crop_b) {
            printf("Errore: sequenza non riproducibile.\n");
            printf("Estrazione %lu: %d contro %d\n",
                   (unsigned long)i,
                   crop_a,
                   crop_b);

            return 1;
        }

        if (crop_a < min_crop_left ||
            crop_a > max_crop_left) {

            out_of_range_count++;
        } else {
            if (!seen[crop_a]) {
                seen[crop_a] = 1;
                unique_count++;
            }

            if (crop_a < observed_min) {
                observed_min = crop_a;
            }

            if (crop_a > observed_max) {
                observed_max = crop_a;
            }
        }

        if (i < 12) {
            printf(
                "  %lu: crop_left=%d, offset_finestra=%d px\n",
                (unsigned long)i,
                crop_a,
                crop_a - GS_SAKAI_CENTRAL_CROP_LEFT
            );
        }

        /*
           Controllo di regressione:
           la modalità legacy con seed 456 deve continuare
           a produrre la sequenza già validata.
        */
        if (mode == GS_AUGMENT_MODE_LEGACY_TEXT &&
            seed == 456u &&
            i < GS_TEST_KNOWN_SEQUENCE_COUNT &&
            crop_a != GS_LEGACY_SEED456_SEQUENCE[i]) {

            printf("Errore: regressione nella sequenza legacy.\n");
            printf("Estrazione %lu: trovato %d, atteso %d\n",
                   (unsigned long)i,
                   crop_a,
                   GS_LEGACY_SEED456_SEQUENCE[i]);

            return 1;
        }
    }

    printf("Valori distinti osservati: %lu/%d\n",
           (unsigned long)unique_count,
           max_crop_left - min_crop_left + 1);

    printf("Min/max osservati: %d/%d\n",
           observed_min,
           observed_max);

    printf("Fuori intervallo: %lu\n\n",
           (unsigned long)out_of_range_count);

    if (out_of_range_count != 0) {
        printf("Errore: osservati crop fuori intervallo.\n");
        return 1;
    }

    /*
       Con un numero elevato di estrazioni devono apparire
       tutti i valori dell'intervallo.
    */
    if (draws >= 100000u &&
        unique_count !=
            (size_t)(max_crop_left - min_crop_left + 1)) {

        printf("Errore: non sono comparsi tutti i crop possibili.\n");
        return 1;
    }

    return 0;
}


static int test_tensor_wrapper(
    const GSEventRecord *record,
    GSAugmentMode mode,
    uint32_t seed
)
{
    GSAugmentRng rng;

    float *random_tensor;
    float *fixed_tensor;
    float *central_tensor;

    int used_crop_left;

    size_t i;
    size_t random_fixed_differences;
    size_t random_central_differences;

    float min_value;
    float max_value;
    double mean_value;

    int rc;

    random_tensor = (float *)malloc(
        GS_TENSOR_SIZE * sizeof(float)
    );

    fixed_tensor = (float *)malloc(
        GS_TENSOR_SIZE * sizeof(float)
    );

    central_tensor = (float *)malloc(
        GS_TENSOR_SIZE * sizeof(float)
    );

    if (random_tensor == NULL ||
        fixed_tensor == NULL ||
        central_tensor == NULL) {

        printf("Errore: memoria insufficiente.\n");

        free(random_tensor);
        free(fixed_tensor);
        free(central_tensor);

        return 1;
    }

    gs_augment_rng_seed(&rng, seed);

    rc = gs_event_load_random_crop_float01_mode(
        record,
        &rng,
        mode,
        &used_crop_left,
        random_tensor
    );

    if (rc != 0) {
        free(random_tensor);
        free(fixed_tensor);
        free(central_tensor);
        return 1;
    }

    rc = gs_event_load_crop_float01(
        record,
        used_crop_left,
        fixed_tensor
    );

    if (rc != 0) {
        free(random_tensor);
        free(fixed_tensor);
        free(central_tensor);
        return 1;
    }

    rc = gs_event_load_central_float01(
        record,
        central_tensor
    );

    if (rc != 0) {
        free(random_tensor);
        free(fixed_tensor);
        free(central_tensor);
        return 1;
    }

    random_fixed_differences = 0;
    random_central_differences = 0;

    for (i = 0; i < GS_TENSOR_SIZE; i++) {
        if (random_tensor[i] != fixed_tensor[i]) {
            random_fixed_differences++;
        }

        if (random_tensor[i] != central_tensor[i]) {
            random_central_differences++;
        }
    }

    compute_stats(
        random_tensor,
        GS_TENSOR_SIZE,
        &min_value,
        &max_value,
        &mean_value
    );

    printf("Wrapper tensore - modalità %s:\n",
           gs_augment_mode_name(mode));

    printf("  crop_left usato            = %d\n",
           used_crop_left);

    printf("  offset finestra dal centro = %d px\n",
           used_crop_left -
           GS_SAKAI_CENTRAL_CROP_LEFT);

    printf("  differenze random/fisso    = %lu\n",
           (unsigned long)random_fixed_differences);

    printf("  differenze random/centrale = %lu su %lu\n",
           (unsigned long)random_central_differences,
           (unsigned long)GS_TENSOR_SIZE);

    printf("  min/max/mean               = %.6f / %.6f / %.6f\n\n",
           min_value,
           max_value,
           mean_value);

    free(random_tensor);
    free(fixed_tensor);
    free(central_tensor);

    if (random_fixed_differences != 0) {
        printf("Errore: wrapper casuale diverso dal crop fisso.\n");
        return 1;
    }

    return 0;
}


int main(int argc, char **argv)
{
    const char *manifest_path;

    size_t event_position;
    uint32_t augmentation_seed;
    size_t draws;

    GSEventManifest manifest;
    const GSEventRecord *record;

    int rc;

    manifest_path = "manifest_events_272.csv";
    event_position = 0;
    augmentation_seed = 456u;
    draws = 100000u;

    if (argc >= 2) {
        manifest_path = argv[1];
    }

    if (argc >= 3) {
        event_position = (size_t)strtoul(
            argv[2],
            NULL,
            10
        );
    }

    if (argc >= 4) {
        augmentation_seed = (uint32_t)strtoul(
            argv[3],
            NULL,
            10
        );
    }

    if (argc >= 5) {
        draws = (size_t)strtoul(
            argv[4],
            NULL,
            10
        );
    }

    if (draws == 0) {
        draws = 100000u;
    }

    printf("Manifest eventi:  %s\n", manifest_path);
    printf("Posizione evento:  %lu\n",
           (unsigned long)event_position);
    printf("Seed augmentation: %lu\n",
           (unsigned long)augmentation_seed);
    printf("Estrazioni test:   %lu\n\n",
           (unsigned long)draws);

    rc = gs_event_manifest_load(
        manifest_path,
        &manifest
    );

    if (rc != 0) {
        return 1;
    }

    record = gs_event_manifest_get(
        &manifest,
        event_position
    );

    if (record == NULL) {
        printf("Errore: posizione evento non valida.\n");
        printf("Eventi disponibili: %lu\n",
               (unsigned long)manifest.count);

        gs_event_manifest_free(&manifest);
        return 1;
    }

    printf("Evento selezionato:\n");
    printf("  event_index = %d\n", record->event_index);
    printf("  event_id    = %s\n", record->event_id);
    printf("  label       = %s, solo per controllo\n\n",
           record->label);

    rc = test_mode_sequence(
        GS_AUGMENT_MODE_LEGACY_TEXT,
        augmentation_seed,
        draws
    );

    if (rc != 0) {
        gs_event_manifest_free(&manifest);
        return 1;
    }

    rc = test_mode_sequence(
        GS_AUGMENT_MODE_SYMMETRIC_FIGURE,
        augmentation_seed,
        draws
    );

    if (rc != 0) {
        gs_event_manifest_free(&manifest);
        return 1;
    }

    rc = test_tensor_wrapper(
        record,
        GS_AUGMENT_MODE_LEGACY_TEXT,
        augmentation_seed
    );

    if (rc != 0) {
        gs_event_manifest_free(&manifest);
        return 1;
    }

    rc = test_tensor_wrapper(
        record,
        GS_AUGMENT_MODE_SYMMETRIC_FIGURE,
        augmentation_seed
    );

    if (rc != 0) {
        gs_event_manifest_free(&manifest);
        return 1;
    }

    printf("OK: entrambe le modalità di augmentation funzionano.\n");
    printf("La modalità legacy mantiene il comportamento 0-24.\n");
    printf("La modalità simmetrica usa il dominio 0-48.\n");
    printf("La validation centrale resta crop_left=24.\n");

    gs_event_manifest_free(&manifest);

    return 0;
}
