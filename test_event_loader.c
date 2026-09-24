#include "vae_event_dataset.h"
#include "vae_event_loader.h"
#include "vae_dataset.h"
#include "vae_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <float.h>


static int load_bin_raw_u8(
    const char *bin_path,
    uint8_t *out_u8
)
{
    FILE *fp;
    size_t nread;
    int extra;

    if (bin_path == NULL || out_u8 == NULL) {
        printf("Errore: argomenti NULL in load_bin_raw_u8.\n");
        return 1;
    }

    fp = fopen(bin_path, "rb");

    if (fp == NULL) {
        printf("Errore: impossibile aprire il file binario:\n");
        printf("%s\n", bin_path);
        return 1;
    }

    nread = fread(
        out_u8,
        sizeof(uint8_t),
        GS_TENSOR_SIZE,
        fp
    );

    if (nread != GS_TENSOR_SIZE) {
        printf("Errore: dimensione .bin non valida.\n");
        printf("File: %s\n", bin_path);
        printf("Byte attesi: %lu\n",
               (unsigned long)GS_TENSOR_SIZE);
        printf("Byte letti:  %lu\n",
               (unsigned long)nread);

        fclose(fp);
        return 1;
    }

    extra = fgetc(fp);

    if (extra != EOF) {
        printf("Errore: il .bin contiene byte aggiuntivi.\n");
        printf("File: %s\n", bin_path);

        fclose(fp);
        return 1;
    }

    fclose(fp);

    return 0;
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


int main(int argc, char **argv)
{
    const char *event_manifest_path;
    const char *tensor_manifest_path;

    size_t events_to_test;
    int shifted_crop_left;

    GSEventManifest event_manifest;
    GSManifest tensor_manifest;

    uint8_t *dynamic_central_u8;
    uint8_t *saved_bin_u8;

    float *central_f32;
    float *shifted_f32;

    size_t i;
    size_t total_different_bytes;

    int rc;

    /*
       Uso:

       test_event_loader.exe
           manifest_events_272.csv
           manifest_tensors_224_central.csv
           numero_eventi_da_testare
           crop_left_spostato

       Esempio:
       test_event_loader.exe manifest_events_272.csv
           manifest_tensors_224_central.csv 5 7
    */

    event_manifest_path = "manifest_events_272.csv";
    tensor_manifest_path = "manifest_tensors_224_central.csv";

    events_to_test = 5;
    shifted_crop_left = 7;

    if (argc >= 2) {
        event_manifest_path = argv[1];
    }

    if (argc >= 3) {
        tensor_manifest_path = argv[2];
    }

    if (argc >= 4) {
        events_to_test = (size_t)strtoul(
            argv[3],
            NULL,
            10
        );
    }

    if (argc >= 5) {
        shifted_crop_left = (int)strtol(
            argv[4],
            NULL,
            10
        );
    }

    printf("Manifest eventi: %s\n", event_manifest_path);
    printf("Manifest tensori: %s\n", tensor_manifest_path);
    printf("Eventi da testare: %lu\n",
           (unsigned long)events_to_test);
    printf("Crop centrale left: %d\n",
           GS_SAKAI_CENTRAL_CROP_LEFT);
    printf("Crop spostato di test left: %d\n\n",
           shifted_crop_left);

    if (!gs_event_crop_left_is_valid(shifted_crop_left)) {
        printf("Errore: crop_left spostato non valido.\n");
        return 1;
    }

    rc = gs_event_manifest_load(
        event_manifest_path,
        &event_manifest
    );

    if (rc != 0) {
        return 1;
    }

    rc = gs_manifest_load(
        tensor_manifest_path,
        &tensor_manifest
    );

    if (rc != 0) {
        gs_event_manifest_free(&event_manifest);
        return 1;
    }

    if (event_manifest.count != tensor_manifest.count) {
        printf("Errore: i due manifest hanno conteggi diversi.\n");
        printf("Eventi PNG: %lu\n",
               (unsigned long)event_manifest.count);
        printf("Tensori:    %lu\n",
               (unsigned long)tensor_manifest.count);

        gs_manifest_free(&tensor_manifest);
        gs_event_manifest_free(&event_manifest);
        return 1;
    }

    if (events_to_test == 0 ||
        events_to_test > event_manifest.count) {

        events_to_test = event_manifest.count;
    }

    dynamic_central_u8 = (uint8_t *)malloc(
        GS_TENSOR_SIZE * sizeof(uint8_t)
    );

    saved_bin_u8 = (uint8_t *)malloc(
        GS_TENSOR_SIZE * sizeof(uint8_t)
    );

    central_f32 = (float *)malloc(
        GS_TENSOR_SIZE * sizeof(float)
    );

    shifted_f32 = (float *)malloc(
        GS_TENSOR_SIZE * sizeof(float)
    );

    if (dynamic_central_u8 == NULL ||
        saved_bin_u8 == NULL ||
        central_f32 == NULL ||
        shifted_f32 == NULL) {

        printf("Errore: memoria insufficiente per il test.\n");

        free(dynamic_central_u8);
        free(saved_bin_u8);
        free(central_f32);
        free(shifted_f32);

        gs_manifest_free(&tensor_manifest);
        gs_event_manifest_free(&event_manifest);

        return 1;
    }

    total_different_bytes = 0;

    printf("Confronto crop centrale PNG vs .bin:\n\n");

    for (i = 0; i < events_to_test; i++) {
        const GSEventRecord *event_record;
        const GSManifestRecord *tensor_record;

        size_t j;
        size_t different_bytes;
        int max_difference;

        event_record = gs_event_manifest_get(
            &event_manifest,
            i
        );

        tensor_record = gs_manifest_get(
            &tensor_manifest,
            i
        );

        if (event_record == NULL || tensor_record == NULL) {
            printf("Errore: record NULL all'indice %lu.\n",
                   (unsigned long)i);

            free(dynamic_central_u8);
            free(saved_bin_u8);
            free(central_f32);
            free(shifted_f32);

            gs_manifest_free(&tensor_manifest);
            gs_event_manifest_free(&event_manifest);

            return 1;
        }

        /*
           Verifica che i due manifest indichino lo stesso evento.
        */
        if (event_record->event_index !=
                tensor_record->event_index ||

            strcmp(
                event_record->event_id,
                tensor_record->event_id
            ) != 0) {

            printf("Errore: manifest non allineati all'indice %lu.\n",
                   (unsigned long)i);

            printf("Event manifest: index=%d, id=%s\n",
                   event_record->event_index,
                   event_record->event_id);

            printf("Tensor manifest: index=%d, id=%s\n",
                   tensor_record->event_index,
                   tensor_record->event_id);

            free(dynamic_central_u8);
            free(saved_bin_u8);
            free(central_f32);
            free(shifted_f32);

            gs_manifest_free(&tensor_manifest);
            gs_event_manifest_free(&event_manifest);

            return 1;
        }

        rc = gs_event_load_central_u8(
            event_record,
            dynamic_central_u8
        );

        if (rc != 0) {
            free(dynamic_central_u8);
            free(saved_bin_u8);
            free(central_f32);
            free(shifted_f32);

            gs_manifest_free(&tensor_manifest);
            gs_event_manifest_free(&event_manifest);

            return 1;
        }

        rc = load_bin_raw_u8(
            tensor_record->tensor_path,
            saved_bin_u8
        );

        if (rc != 0) {
            free(dynamic_central_u8);
            free(saved_bin_u8);
            free(central_f32);
            free(shifted_f32);

            gs_manifest_free(&tensor_manifest);
            gs_event_manifest_free(&event_manifest);

            return 1;
        }

        different_bytes = 0;
        max_difference = 0;

        for (j = 0; j < GS_TENSOR_SIZE; j++) {
            int difference;

            difference =
                (int)dynamic_central_u8[j] -
                (int)saved_bin_u8[j];

            if (difference < 0) {
                difference = -difference;
            }

            if (difference != 0) {
                different_bytes++;
            }

            if (difference > max_difference) {
                max_difference = difference;
            }
        }

        total_different_bytes += different_bytes;

        printf(
            "Evento %d, id=%s: byte diversi=%lu, differenza massima=%d\n",
            event_record->event_index,
            event_record->event_id,
            (unsigned long)different_bytes,
            max_difference
        );
    }

    printf("\nTotale byte diversi nei test: %lu\n",
           (unsigned long)total_different_bytes);

    if (total_different_bytes != 0) {
        printf("Errore: il crop centrale dinamico non coincide con i .bin.\n");

        free(dynamic_central_u8);
        free(saved_bin_u8);
        free(central_f32);
        free(shifted_f32);

        gs_manifest_free(&tensor_manifest);
        gs_event_manifest_free(&event_manifest);

        return 1;
    }

    /*
       Test manuale di un crop spostato sul primo evento.
       Questo test usa un crop_left esplicito; la generazione casuale del crop e' verificata separatamente.
    */
    rc = gs_event_load_central_float01(
        &event_manifest.records[0],
        central_f32
    );

    if (rc != 0) {
        free(dynamic_central_u8);
        free(saved_bin_u8);
        free(central_f32);
        free(shifted_f32);

        gs_manifest_free(&tensor_manifest);
        gs_event_manifest_free(&event_manifest);

        return 1;
    }

    rc = gs_event_load_crop_float01(
        &event_manifest.records[0],
        shifted_crop_left,
        shifted_f32
    );

    if (rc != 0) {
        free(dynamic_central_u8);
        free(saved_bin_u8);
        free(central_f32);
        free(shifted_f32);

        gs_manifest_free(&tensor_manifest);
        gs_event_manifest_free(&event_manifest);

        return 1;
    }

    {
        float central_min;
        float central_max;
        double central_mean;

        float shifted_min;
        float shifted_max;
        double shifted_mean;

        size_t changed_values;
        size_t j;

        compute_float_stats(
            central_f32,
            GS_TENSOR_SIZE,
            &central_min,
            &central_max,
            &central_mean
        );

        compute_float_stats(
            shifted_f32,
            GS_TENSOR_SIZE,
            &shifted_min,
            &shifted_max,
            &shifted_mean
        );

        changed_values = 0;

        for (j = 0; j < GS_TENSOR_SIZE; j++) {
            if (central_f32[j] != shifted_f32[j]) {
                changed_values++;
            }
        }

        printf("\nTest crop spostato sul primo evento:\n");
        printf("  crop centrale left = %d\n",
               GS_SAKAI_CENTRAL_CROP_LEFT);
        printf("  crop spostato left = %d\n",
               shifted_crop_left);

        printf("  valori diversi tra i due crop = %lu su %lu\n",
               (unsigned long)changed_values,
               (unsigned long)GS_TENSOR_SIZE);

        printf("  centrale min/max/mean = %.6f / %.6f / %.6f\n",
               central_min,
               central_max,
               central_mean);

        printf("  spostato min/max/mean = %.6f / %.6f / %.6f\n",
               shifted_min,
               shifted_max,
               shifted_mean);
    }

    printf("\nOK: loader dinamico di un evento completato.\n");
    printf("Il crop centrale da PNG coincide esattamente con il .bin.\n");
    printf("Il crop spostato usa lo stesso crop_left sui quattro canali.\n");
    printf("Questo test usa un crop_left configurabile; la generazione casuale del crop e' verificata separatamente.\n");

    free(dynamic_central_u8);
    free(saved_bin_u8);
    free(central_f32);
    free(shifted_f32);

    gs_manifest_free(&tensor_manifest);
    gs_event_manifest_free(&event_manifest);

    return 0;
}