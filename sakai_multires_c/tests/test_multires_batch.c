#include "vae_multires_batch.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


static int parse_mode(const char *text, GSAugmentMode *mode)
{
    if (strcmp(text, "legacy_text") == 0) {
        *mode = GS_AUGMENT_MODE_LEGACY_TEXT;
        return 0;
    }

    if (strcmp(text, "symmetric_figure") == 0) {
        *mode = GS_AUGMENT_MODE_SYMMETRIC_FIGURE;
        return 0;
    }

    return 1;
}


static size_t compare_exact(const float *a, const float *b, size_t count)
{
    size_t i;
    size_t differences = 0;

    for (i = 0; i < count; i++) {
        if (a[i] != b[i]) {
            differences++;
        }
    }

    return differences;
}


static int test_profile(
    int side,
    const char *tensor_manifest_path,
    const GSEventManifest *events,
    const GSSavedSplit *train,
    const GSSavedSplit *validation,
    GSAugmentMode mode
)
{
    GSTensorSpec spec;
    GSTensorManifest tensors = {0};
    GSMultiresEventLoader loader = {0};
    GSMultiresBatch batch = {0};
    GSAugmentRng rng;
    float *central = NULL;
    int min_crop;
    int max_crop;
    size_t i;
    size_t differences;
    int rc = 1;

    if (gs_tensor_spec_init(&spec, side) != 0 ||
        gs_tensor_manifest_load(tensor_manifest_path, &spec, &tensors) != 0 ||
        gs_multires_event_loader_init(&loader, side) != 0 ||
        gs_multires_batch_alloc(&batch, &spec, 4) != 0 ||
        gs_augment_mode_crop_range(mode, &min_crop, &max_crop) != 0) {
        goto cleanup;
    }

    gs_augment_rng_seed(&rng, 456u);

    if (gs_multires_batch_load_train_mode(
            events, train, &loader, 0, 4, &rng, mode, &batch
        ) != 0) {
        goto cleanup;
    }

    for (i = 0; i < batch.count; i++) {
        if (batch.crop_lefts[i] < min_crop || batch.crop_lefts[i] > max_crop) {
            goto cleanup;
        }
    }

    printf("%s %s train crop:",
           gs_tensor_spec_name(&spec),
           gs_augment_mode_name(mode));
    for (i = 0; i < batch.count; i++) {
        printf(" %d(%+d)",
               batch.crop_lefts[i],
               batch.crop_lefts[i] - GS_SAKAI_CENTRAL_CROP_LEFT);
    }
    printf("\n");

    if (gs_multires_batch_load_validation(
            events, &tensors, validation, 0, 4, &batch
        ) != 0) {
        goto cleanup;
    }

    for (i = 0; i < batch.count; i++) {
        if (batch.crop_lefts[i] != GS_SAKAI_CENTRAL_CROP_LEFT) {
            goto cleanup;
        }
    }

    central = (float *)malloc(spec.bytes_f32);
    if (central == NULL) {
        goto cleanup;
    }

    if (gs_multires_event_load_central_float01(
            &loader,
            gs_event_manifest_get(events, batch.manifest_indices[0]),
            central
        ) != 0) {
        goto cleanup;
    }

    differences = compare_exact(
        central,
        gs_multires_batch_item_const_ptr(&batch, 0),
        spec.values_per_event
    );

    printf("%s validation crop=24, differenze dinamico/bin=%lu\n",
           gs_tensor_spec_name(&spec),
           (unsigned long)differences);

    if (differences != 0) {
        goto cleanup;
    }

    rc = 0;

cleanup:
    free(central);
    gs_multires_batch_free(&batch);
    gs_multires_event_loader_free(&loader);
    gs_tensor_manifest_free(&tensors);
    return rc;
}


int main(int argc, char **argv)
{
    GSAugmentMode mode = GS_AUGMENT_MODE_LEGACY_TEXT;
    GSEventManifest events = {0};
    GSSavedSplit train = {0};
    GSSavedSplit validation = {0};
    int rc = 1;

    if (argc >= 2 && parse_mode(argv[1], &mode) != 0) {
        printf("Uso: test_multires_batch.exe [legacy_text|symmetric_figure]\n");
        return 1;
    }

    if (gs_event_manifest_load("manifest_events_272.csv", &events) != 0 ||
        gs_saved_split_load(
            "split_train_indices_seed123.csv", "train", &events, &train
        ) != 0 ||
        gs_saved_split_load(
            "split_val_indices_seed123.csv", "validation", &events, &validation
        ) != 0) {
        goto cleanup;
    }

    printf("Modalita train: %s\n", gs_augment_mode_name(mode));

    if (test_profile(
            75, "manifest_tensors_75_central.csv",
            &events, &train, &validation, mode
        ) != 0 ||
        test_profile(
            112, "manifest_tensors_112_central.csv",
            &events, &train, &validation, mode
        ) != 0 ||
        test_profile(
            224, "manifest_tensors_224_central.csv",
            &events, &train, &validation, mode
        ) != 0) {
        goto cleanup;
    }

    printf("OK: batch multirisoluzione %s.\n", gs_augment_mode_name(mode));
    rc = 0;

cleanup:
    gs_saved_split_free(&validation);
    gs_saved_split_free(&train);
    gs_event_manifest_free(&events);
    return rc;
}
