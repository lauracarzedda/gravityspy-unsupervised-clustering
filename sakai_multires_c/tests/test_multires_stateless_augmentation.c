#include "vae_multires_batch.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


typedef struct {
    uint64_t base_seed;
    uint64_t epoch;
    uint64_t manifest_index;
    GSAugmentMode mode;
    int expected_crop_left;
} GSGoldenCrop;


static int fail(const char *test_name, const char *detail)
{
    fprintf(stderr, "[FAIL] %s: %s\n", test_name, detail);
    return 1;
}


static int find_manifest_item(
    const GSMultiresBatch *batch,
    size_t manifest_index,
    size_t *item_index
)
{
    size_t i;

    if (batch == NULL || item_index == NULL) {
        return 1;
    }

    for (i = 0; i < batch->count; i++) {
        if (batch->manifest_indices[i] == manifest_index) {
            *item_index = i;
            return 0;
        }
    }

    return 1;
}


static int compare_batch_items(
    const GSMultiresBatch *a,
    size_t a_item,
    const GSMultiresBatch *b,
    size_t b_item
)
{
    const float *a_data = gs_multires_batch_item_const_ptr(a, a_item);
    const float *b_data = gs_multires_batch_item_const_ptr(b, b_item);

    if (a_data == NULL || b_data == NULL ||
        a->spec.bytes_f32 != b->spec.bytes_f32) {
        return 1;
    }

    return memcmp(a_data, b_data, a->spec.bytes_f32) != 0;
}


static int test_golden_vectors(void)
{
    static const GSGoldenCrop vectors[] = {
        {UINT64_C(0), UINT64_C(0), UINT64_C(0),
         GS_AUGMENT_MODE_LEGACY_TEXT, 7},
        {UINT64_C(123), UINT64_C(0), UINT64_C(7663),
         GS_AUGMENT_MODE_SYMMETRIC_FIGURE, 36},
        {UINT64_C(789), UINT64_C(79), UINT64_C(4224),
         GS_AUGMENT_MODE_SYMMETRIC_FIGURE, 25},
        {UINT64_C(456), UINT64_C(3), UINT64_C(42),
         GS_AUGMENT_MODE_LEGACY_TEXT, 0},
        {UINT64_MAX, UINT64_MAX, UINT64_MAX,
         GS_AUGMENT_MODE_SYMMETRIC_FIGURE, 40}
    };
    size_t i;

    for (i = 0; i < sizeof(vectors) / sizeof(vectors[0]); i++) {
        int crop_left = -1;

        if (gs_augment_crop_left_for_event_mode(
                vectors[i].base_seed,
                vectors[i].epoch,
                vectors[i].manifest_index,
                vectors[i].mode,
                &crop_left
            ) != 0 || crop_left != vectors[i].expected_crop_left) {
            return fail("golden_vectors", "stateless vector mismatch");
        }
    }

    printf("[PASS] golden_vectors_stateless_v1\n");
    return 0;
}


static int test_stateless_contract(void)
{
    uint64_t key;
    int epoch_changes_crop = 0;
    int manifest_changes_crop = 0;
    int reference;

    for (key = UINT64_C(0); key < UINT64_C(512); key++) {
        int legacy_a;
        int legacy_b;
        int symmetric_a;
        int symmetric_b;

        if (gs_augment_crop_left_for_event_mode(
                UINT64_C(123), UINT64_C(17), key,
                GS_AUGMENT_MODE_LEGACY_TEXT, &legacy_a
            ) != 0 ||
            gs_augment_crop_left_for_event_mode(
                UINT64_C(123), UINT64_C(17), key,
                GS_AUGMENT_MODE_LEGACY_TEXT, &legacy_b
            ) != 0 ||
            gs_augment_crop_left_for_event_mode(
                UINT64_C(123), UINT64_C(17), key,
                GS_AUGMENT_MODE_SYMMETRIC_FIGURE, &symmetric_a
            ) != 0 ||
            gs_augment_crop_left_for_event_mode(
                UINT64_C(123), UINT64_C(17), key,
                GS_AUGMENT_MODE_SYMMETRIC_FIGURE, &symmetric_b
            ) != 0) {
            return fail("stateless_contract", "valid call failed");
        }

        if (legacy_a != legacy_b || symmetric_a != symmetric_b ||
            legacy_a < 0 || legacy_a > 24 ||
            symmetric_a < 0 || symmetric_a > 48 ||
            symmetric_a - GS_SAKAI_CENTRAL_CROP_LEFT < -24 ||
            symmetric_a - GS_SAKAI_CENTRAL_CROP_LEFT > 24) {
            return fail("stateless_contract", "range or replica mismatch");
        }
    }

    reference = -1;
    if (gs_augment_crop_left_for_event_mode(
            UINT64_C(789), UINT64_C(0), UINT64_C(7663),
            GS_AUGMENT_MODE_SYMMETRIC_FIGURE, &reference
        ) != 0) {
        return fail("stateless_contract", "reference derivation failed");
    }

    for (key = UINT64_C(1); key < UINT64_C(128); key++) {
        int by_epoch;
        int by_manifest;

        if (gs_augment_crop_left_for_event_mode(
                UINT64_C(789), key, UINT64_C(7663),
                GS_AUGMENT_MODE_SYMMETRIC_FIGURE, &by_epoch
            ) != 0 ||
            gs_augment_crop_left_for_event_mode(
                UINT64_C(789), UINT64_C(0), UINT64_C(7663) + key,
                GS_AUGMENT_MODE_SYMMETRIC_FIGURE, &by_manifest
            ) != 0) {
            return fail("stateless_contract", "key derivation failed");
        }

        if (by_epoch != reference) {
            epoch_changes_crop = 1;
        }
        if (by_manifest != reference) {
            manifest_changes_crop = 1;
        }
    }

    if (!epoch_changes_crop || !manifest_changes_crop) {
        return fail("stateless_contract", "key field has no observed effect");
    }

    {
        int unchanged = 731;

        if (gs_augment_crop_left_for_event_mode(
                UINT64_C(1), UINT64_C(2), UINT64_C(3),
                (GSAugmentMode)99, &unchanged
            ) == 0 || unchanged != 731 ||
            gs_augment_crop_left_for_event_mode(
                UINT64_C(1), UINT64_C(2), UINT64_C(3),
                GS_AUGMENT_MODE_LEGACY_TEXT, NULL
            ) == 0) {
            return fail("stateless_contract", "invalid argument contract");
        }
    }

    printf("[PASS] pure_replica_ranges_keys_invalid_args\n");
    return 0;
}


static int test_sequential_regression(void)
{
    static const int expected[] = {
        0, 20, 20, 2, 0, 14, 11, 16, 16, 5, 17, 2
    };
    GSAugmentRng rng;
    size_t i;

    gs_augment_rng_seed(&rng, 456u);
    for (i = 0; i < sizeof(expected) / sizeof(expected[0]); i++) {
        int crop_left = -1;

        if (gs_augment_random_crop_left_mode(
                &rng,
                GS_AUGMENT_MODE_LEGACY_TEXT,
                &crop_left
            ) != 0 || crop_left != expected[i]) {
            return fail("sequential_regression", "golden sequence changed");
        }
    }

    printf("[PASS] sequential_seed456_golden_unchanged\n");
    return 0;
}


static int test_loader_invariance(void)
{
    static size_t order_a_indices[] = {7663, 2606, 7841, 4067};
    static size_t order_b_indices[] = {4067, 7663, 7841, 2606};
    static size_t composition_indices[] = {999, 2606, 5554};
    GSSavedSplit order_a = {order_a_indices, 4, "order_a"};
    GSSavedSplit order_b = {order_b_indices, 4, "order_b"};
    GSSavedSplit composition = {
        composition_indices, 3, "composition"
    };
    GSEventManifest events = {0};
    GSTensorSpec spec;
    GSMultiresEventLoader loader_a = {0};
    GSMultiresEventLoader loader_b = {0};
    GSMultiresBatch full_a = {0};
    GSMultiresBatch full_b = {0};
    GSMultiresBatch replica = {0};
    GSMultiresBatch boundary = {0};
    GSMultiresBatch changed_composition = {0};
    float *explicit_data = NULL;
    size_t i;
    int rc = 1;

    if (gs_event_manifest_load("manifest_events_272.csv", &events) != 0 ||
        gs_tensor_spec_init(&spec, 75) != 0 ||
        gs_multires_event_loader_init(&loader_a, 75) != 0 ||
        gs_multires_event_loader_init(&loader_b, 75) != 0 ||
        gs_multires_batch_alloc(&full_a, &spec, 4) != 0 ||
        gs_multires_batch_alloc(&full_b, &spec, 4) != 0 ||
        gs_multires_batch_alloc(&replica, &spec, 4) != 0 ||
        gs_multires_batch_alloc(&boundary, &spec, 2) != 0 ||
        gs_multires_batch_alloc(&changed_composition, &spec, 3) != 0) {
        fail("loader_invariance", "fixture setup failed");
        goto cleanup;
    }

    explicit_data = (float *)malloc(spec.bytes_f32);
    if (explicit_data == NULL) {
        fail("loader_invariance", "explicit buffer allocation failed");
        goto cleanup;
    }

    if (gs_multires_batch_load_train_stateless_mode(
            &events, &order_a, &loader_a, 0, 4,
            UINT64_C(789), UINT64_C(17),
            GS_AUGMENT_MODE_SYMMETRIC_FIGURE, &full_a
        ) != 0 ||
        gs_multires_batch_load_train_stateless_mode(
            &events, &order_b, &loader_b, 0, 4,
            UINT64_C(789), UINT64_C(17),
            GS_AUGMENT_MODE_SYMMETRIC_FIGURE, &full_b
        ) != 0 ||
        gs_multires_batch_load_train_stateless_mode(
            &events, &order_a, &loader_b, 0, 4,
            UINT64_C(789), UINT64_C(17),
            GS_AUGMENT_MODE_SYMMETRIC_FIGURE, &replica
        ) != 0) {
        fail("loader_invariance", "full load failed");
        goto cleanup;
    }

    for (i = 0; i < full_a.count; i++) {
        size_t b_item;
        const GSEventRecord *record;
        const float *loaded;
        size_t value_index;
        int direct_crop = -1;

        if (find_manifest_item(
                &full_b, full_a.manifest_indices[i], &b_item
            ) != 0 ||
            full_a.crop_lefts[i] != full_b.crop_lefts[b_item] ||
            compare_batch_items(&full_a, i, &full_b, b_item) != 0 ||
            full_a.crop_lefts[i] != replica.crop_lefts[i] ||
            compare_batch_items(&full_a, i, &replica, i) != 0) {
            fail("loader_invariance", "order or replica changed mapping");
            goto cleanup;
        }

        if (gs_augment_crop_left_for_event_mode(
                UINT64_C(789), UINT64_C(17),
                (uint64_t)full_a.manifest_indices[i],
                GS_AUGMENT_MODE_SYMMETRIC_FIGURE,
                &direct_crop
            ) != 0 || direct_crop != full_a.crop_lefts[i]) {
            fail("loader_invariance", "metadata differs from pure API");
            goto cleanup;
        }

        record = gs_event_manifest_get(&events, full_a.manifest_indices[i]);
        if (record == NULL ||
            full_a.event_indices[i] != record->event_index ||
            gs_multires_event_load_crop_float01(
                &loader_a, record, direct_crop, explicit_data
            ) != 0) {
            fail("loader_invariance", "explicit crop load failed");
            goto cleanup;
        }

        loaded = gs_multires_batch_item_const_ptr(&full_a, i);
        if (loaded == NULL ||
            memcmp(loaded, explicit_data, spec.bytes_f32) != 0) {
            fail("loader_invariance", "loader data differs from explicit crop");
            goto cleanup;
        }

        for (value_index = 0;
             value_index < spec.values_per_event;
             value_index++) {
            if (loaded[value_index] < 0.0f || loaded[value_index] > 1.0f) {
                fail("loader_invariance", "loader output outside [0,1]");
                goto cleanup;
            }
        }
    }

    if (gs_multires_batch_load_train_stateless_mode(
            &events, &order_a, &loader_b, 0, 2,
            UINT64_C(789), UINT64_C(17),
            GS_AUGMENT_MODE_SYMMETRIC_FIGURE, &boundary
        ) != 0) {
        fail("loader_invariance", "first boundary load failed");
        goto cleanup;
    }

    for (i = 0; i < 2; i++) {
        if (boundary.manifest_indices[i] != full_a.manifest_indices[i] ||
            boundary.crop_lefts[i] != full_a.crop_lefts[i] ||
            compare_batch_items(&boundary, i, &full_a, i) != 0) {
            fail("loader_invariance", "first batch boundary changed mapping");
            goto cleanup;
        }
    }

    if (gs_multires_batch_load_train_stateless_mode(
            &events, &order_a, &loader_b, 2, 2,
            UINT64_C(789), UINT64_C(17),
            GS_AUGMENT_MODE_SYMMETRIC_FIGURE, &boundary
        ) != 0) {
        fail("loader_invariance", "second boundary load failed");
        goto cleanup;
    }

    for (i = 0; i < 2; i++) {
        if (boundary.manifest_indices[i] != full_a.manifest_indices[i + 2] ||
            boundary.crop_lefts[i] != full_a.crop_lefts[i + 2] ||
            compare_batch_items(&boundary, i, &full_a, i + 2) != 0) {
            fail("loader_invariance", "second batch boundary changed mapping");
            goto cleanup;
        }
    }

    if (gs_multires_batch_load_train_stateless_mode(
            &events, &composition, &loader_b, 0, 3,
            UINT64_C(789), UINT64_C(17),
            GS_AUGMENT_MODE_SYMMETRIC_FIGURE, &changed_composition
        ) != 0) {
        fail("loader_invariance", "changed composition load failed");
        goto cleanup;
    }

    {
        size_t a_item;
        size_t composition_item;

        if (find_manifest_item(&full_a, 2606, &a_item) != 0 ||
            find_manifest_item(
                &changed_composition, 2606, &composition_item
            ) != 0 ||
            full_a.crop_lefts[a_item] !=
                changed_composition.crop_lefts[composition_item] ||
            compare_batch_items(
                &full_a, a_item, &changed_composition, composition_item
            ) != 0) {
            fail("loader_invariance", "composition changed common event");
            goto cleanup;
        }
    }

    printf("MANIFEST_INDEX_TO_CROP_INVARIANT=true\n");
    printf("[PASS] loader_order_batch_composition_explicit_crop\n");
    rc = 0;

cleanup:
    free(explicit_data);
    gs_multires_batch_free(&changed_composition);
    gs_multires_batch_free(&boundary);
    gs_multires_batch_free(&replica);
    gs_multires_batch_free(&full_b);
    gs_multires_batch_free(&full_a);
    gs_multires_event_loader_free(&loader_b);
    gs_multires_event_loader_free(&loader_a);
    gs_event_manifest_free(&events);
    return rc;
}


int main(void)
{
    if (test_golden_vectors() != 0 ||
        test_stateless_contract() != 0 ||
        test_sequential_regression() != 0 ||
        test_loader_invariance() != 0) {
        return 1;
    }

    printf("ALL STATELESS AUGMENTATION TESTS PASSED\n");
    return 0;
}
