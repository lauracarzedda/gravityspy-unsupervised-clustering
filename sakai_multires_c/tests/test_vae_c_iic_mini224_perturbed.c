#include "vae_c_iic_mini224_perturbed.h"

#include "vae_augmentation.h"
#include "vae_c_sha256.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_ROWS 16u
#define TEST_VALUES (TEST_ROWS * VAE_C_IIC_MINI224_LATENT_DIM)

static bool report_ok(const VAECIICMini224PerturbedReport *report,
                      uint64_t calls)
{
    return report->samples == TEST_ROWS &&
           report->encoder_forward_calls == calls &&
           report->inputs_bitwise_unchanged &&
           report->model_parameters_bn_bitwise_unchanged &&
           report->decoder_workspace_untouched &&
           report->optimizer_unchanged;
}

static bool same_row(const VAECIICMini224PerturbedRow *row,
                     const VAECIICMini224CentralRow *expected,
                     uint64_t epoch)
{
    int crop;
    return row->identity.iic_row_index == expected->iic_row_index &&
           row->identity.manifest_index == expected->manifest_index &&
           row->identity.event_index == expected->event_index &&
           row->identity.split_pos == expected->split_pos &&
           row->identity.source_split == expected->source_split &&
           strcmp(row->identity.event_id, expected->event_id) == 0 &&
           row->epoch == epoch &&
           gs_augment_crop_left_for_event_mode(
               VAE_C_IIC_MINI224_AUGMENTATION_SEED, epoch,
               expected->manifest_index,
               GS_AUGMENT_MODE_SYMMETRIC_FIGURE, &crop) == 0 &&
           crop >= 0 && crop <= 48 && row->crop_left == crop;
}

static bool conversion_and_channel_order(VAECIICMini224Perturbed *source,
                                         size_t iic_row, int crop)
{
    const VAECIICMini224CentralRow *row = &source->central.rows[iic_row];
    const GSEventRecord *record = gs_event_manifest_get(
        &source->events, row->manifest_index);
    const GSTensorSpec *spec = gs_multires_event_loader_spec(&source->loader);
    GSEventRecord resolved;
    uint8_t *bytes = NULL;
    float *values = NULL;
    size_t view;
    size_t index;
    bool ok = false;
    if (record == NULL || spec == NULL ||
        spec->values_per_event != 4u * 224u * 224u) {
        return false;
    }
    memset(&resolved, 0, sizeof(resolved));
    resolved.event_index = record->event_index;
    memcpy(resolved.event_id, record->event_id, sizeof(resolved.event_id));
    for (view = 0u; view < GS_EVENT_VIEW_COUNT; ++view) {
        int written = snprintf(resolved.image_paths[view],
                               sizeof(resolved.image_paths[view]),
                               "%s/%s", source->repository_root,
                               record->image_paths[view]);
        if (written < 0 ||
            (size_t)written >= sizeof(resolved.image_paths[view])) {
            return false;
        }
    }
    bytes = (uint8_t *)malloc(spec->values_per_event);
    values = (float *)malloc(spec->values_per_event * sizeof(float));
    if (bytes == NULL || values == NULL ||
        gs_multires_event_load_crop_u8(&source->loader, &resolved, crop,
                                       bytes) != 0 ||
        gs_multires_event_load_crop_float01(&source->loader, &resolved,
                                             crop, values) != 0) {
        goto done;
    }
    for (index = 0u; index < spec->values_per_event; ++index) {
        if (values[index] != (float)bytes[index] / 255.0f) {
            goto done;
        }
    }
    ok = true;
done:
    free(values);
    free(bytes);
    return ok;
}

static bool negative_gates(VAECIICMini224Perturbed *source,
                           const char *repository_root)
{
    VAECMuCanonicalContract wrong_contract;
    VAECCheckpointMetadata metadata;
    char error[256];
    size_t selected = 0u;
    size_t invalid_row = VAE_C_IIC_MINI224_ROWS;
    float output[VAE_C_IIC_MINI224_LATENT_DIM];
    VAECIICMini224CentralRow saved_row;
    size_t saved_shape;
    char saved_path[GS_EVENT_MAX_PATH_LEN];
    uint32_t outside = 0u;
    size_t index;
    bool found_outside = false;
    output[0] = -123.0f;
    if (vae_c_iic_mini224_perturbed_compute(source, &selected, 1u, 0u,
                                             17u, output, NULL, NULL) ||
        vae_c_iic_mini224_perturbed_compute(source, &invalid_row, 1u, 0u,
                                             16u, output, NULL, NULL) ||
        output[0] != -123.0f) {
        return false;
    }
    if (vae_c_mu_build_canonical_contract_label_free(
            repository_root, "mini112", &wrong_contract, error,
            sizeof(error)) != VAE_C_MU_OK) {
        return false;
    }
    vae_c_checkpoint_metadata_init(&metadata);
    if (vae_c_checkpoint_load(wrong_contract.checkpoint_path,
                              &source->contract.expected_identity,
                              &source->model, &source->optimizer,
                              &metadata) == VAE_C_CHECKPOINT_OK) {
        return false;
    }
    source->contract.profile.input_side = 112u;
    if (vae_c_iic_mini224_perturbed_compute(source, &selected, 1u, 0u,
                                             16u, output, NULL, NULL)) {
        source->contract.profile.input_side = 224u;
        return false;
    }
    source->contract.profile.input_side = 224u;
    saved_row = source->central.rows[0];
    for (outside = 0u; outside < VAE_C_IIC_MINI224_MANIFEST_ROWS;
         ++outside) {
        bool present = false;
        for (index = 0u; index < source->central.row_count; ++index) {
            if (source->central.rows[index].manifest_index == outside) {
                present = true;
                break;
            }
        }
        if (!present) {
            found_outside = true;
            break;
        }
    }
    if (!found_outside) {
        return false;
    }
    source->central.rows[0].manifest_index = outside;
    source->central.rows[0].event_index =
        (uint32_t)source->events.records[outside].event_index;
    (void)strcpy(source->central.rows[0].event_id,
                 source->events.records[outside].event_id);
    if (vae_c_iic_mini224_perturbed_compute(source, &selected, 1u, 0u,
                                             16u, output, NULL, NULL)) {
        source->central.rows[0] = saved_row;
        return false;
    }
    source->central.rows[0] = saved_row;
    memcpy(saved_path,
           source->events.records[saved_row.manifest_index].image_paths[0],
           sizeof(saved_path));
    (void)strcpy(source->events.records[saved_row.manifest_index]
                     .image_paths[0], "missing/mini224_iic_test.png");
    if (vae_c_iic_mini224_perturbed_compute(source, &selected, 1u, 0u,
                                             16u, output, NULL, NULL)) {
        memcpy(source->events.records[saved_row.manifest_index]
                   .image_paths[0], saved_path, sizeof(saved_path));
        return false;
    }
    memcpy(source->events.records[saved_row.manifest_index].image_paths[0],
           saved_path, sizeof(saved_path));
    saved_shape = source->loader.target_spec.values_per_event;
    source->loader.target_spec.values_per_event = saved_shape - 1u;
    if (vae_c_iic_mini224_perturbed_compute(source, &selected, 1u, 0u,
                                             16u, output, NULL, NULL)) {
        source->loader.target_spec.values_per_event = saved_shape;
        return false;
    }
    source->loader.target_spec.values_per_event = saved_shape;
    return output[0] == -123.0f;
}

int main(int argc, char **argv)
{
    VAECIICMini224Perturbed source;
    VAECIICMini224PerturbedReport report;
    size_t rows[TEST_ROWS];
    size_t reverse[TEST_ROWS];
    float mu16[TEST_VALUES];
    float mu8[TEST_VALUES];
    float mu4[TEST_VALUES];
    float mu_reverse[TEST_VALUES];
    float mu_epoch1[VAE_C_IIC_MINI224_LATENT_DIM];
    VAECIICMini224PerturbedRow identity16[TEST_ROWS];
    VAECIICMini224PerturbedRow identity8[TEST_ROWS];
    VAECIICMini224PerturbedRow identity4[TEST_ROWS];
    VAECIICMini224PerturbedRow identity_reverse[TEST_ROWS];
    VAECIICMini224PerturbedRow identity_epoch1;
    char digest[VAE_C_SHA256_HEX_SIZE];
    size_t index;
    size_t changed = TEST_ROWS;
    bool success = false;
    if (argc != 2) {
        return 2;
    }
    vae_c_iic_mini224_perturbed_init(&source);
    if (!vae_c_iic_mini224_perturbed_open(&source, argv[1])) {
        goto done;
    }
    for (index = 0u; index < TEST_ROWS; ++index) {
        rows[index] = index == TEST_ROWS - 1u ?
            VAE_C_IIC_MINI224_TRAIN_ROWS : index;
    }
    for (index = 0u; index < TEST_ROWS; ++index) {
        reverse[index] = rows[TEST_ROWS - 1u - index];
    }
    if (!vae_c_iic_mini224_perturbed_compute(
            &source, rows, TEST_ROWS, 0u, 16u, mu16, identity16,
            &report) || !report_ok(&report, 1u) ||
        !vae_c_iic_mini224_perturbed_compute(
            &source, rows, TEST_ROWS, 0u, 8u, mu8, identity8,
            &report) || !report_ok(&report, 2u) ||
        !vae_c_iic_mini224_perturbed_compute(
            &source, rows, TEST_ROWS, 0u, 4u, mu4, identity4,
            &report) || !report_ok(&report, 4u) ||
        memcmp(mu16, mu8, sizeof(mu16)) != 0 ||
        memcmp(mu16, mu4, sizeof(mu16)) != 0 ||
        !vae_c_iic_mini224_perturbed_compute(
            &source, reverse, TEST_ROWS, 0u, 8u, mu_reverse,
            identity_reverse, &report) || !report_ok(&report, 2u)) {
        goto done;
    }
    for (index = 0u; index < TEST_ROWS; ++index) {
        const size_t original = TEST_ROWS - 1u - index;
        size_t latent;
        if (!same_row(&identity16[index], &source.central.rows[rows[index]],
                      0u) ||
            !same_row(&identity8[index], &source.central.rows[rows[index]],
                      0u) ||
            !same_row(&identity4[index], &source.central.rows[rows[index]],
                      0u) ||
            !same_row(&identity_reverse[index],
                      &source.central.rows[reverse[index]], 0u) ||
            identity16[index].crop_left != identity8[index].crop_left ||
            identity16[index].crop_left != identity4[index].crop_left ||
            identity_reverse[index].crop_left !=
                identity16[original].crop_left ||
            memcmp(mu_reverse + index * VAE_C_IIC_MINI224_LATENT_DIM,
                   mu16 + original * VAE_C_IIC_MINI224_LATENT_DIM,
                   VAE_C_IIC_MINI224_LATENT_DIM * sizeof(float)) != 0) {
            goto done;
        }
        for (latent = 0u; latent < VAE_C_IIC_MINI224_LATENT_DIM;
             ++latent) {
            if (!isfinite(mu16[index * VAE_C_IIC_MINI224_LATENT_DIM +
                               latent])) {
                goto done;
            }
        }
        if (changed == TEST_ROWS) {
            int next_crop;
            if (gs_augment_crop_left_for_event_mode(
                    VAE_C_IIC_MINI224_AUGMENTATION_SEED, 1u,
                    source.central.rows[rows[index]].manifest_index,
                    GS_AUGMENT_MODE_SYMMETRIC_FIGURE, &next_crop) != 0) {
                goto done;
            }
            if (next_crop != identity16[index].crop_left) {
                changed = index;
            }
        }
    }
    if (changed == TEST_ROWS ||
        !conversion_and_channel_order(&source, rows[0],
                                       identity16[0].crop_left) ||
        !vae_c_iic_mini224_perturbed_compute(
            &source, &rows[changed], 1u, 1u, 1u, mu_epoch1,
            &identity_epoch1, &report) ||
        !same_row(&identity_epoch1,
                  &source.central.rows[rows[changed]], 1u) ||
        identity_epoch1.crop_left == identity16[changed].crop_left ||
        !negative_gates(&source, argv[1]) ||
        vae_c_sha256_digest_hex_lower(mu16, sizeof(mu16), digest) !=
            VAE_C_SHA256_OK) {
        goto done;
    }
    printf("MINI224_PERTURBED_ROWS=%u\n", TEST_ROWS);
    puts("MINI224_PERTURBED_MICROBATCH_16_8_4_BITWISE=true");
    puts("MINI224_PERTURBED_ORDER_AND_CROP_INDEPENDENT=true");
    puts("MINI224_PERTURBED_EVAL_FROZEN=true");
    puts("MINI224_PERTURBED_DECODER_UNTOUCHED=true");
    puts("MINI224_PERTURBED_NEGATIVES_PASS=true");
    printf("MINI224_PERTURBED_MU_SHA256=%s\n", digest);
    success = true;
done:
    vae_c_iic_mini224_perturbed_free(&source);
    if (!success) {
        fputs("MINI224_PERTURBED_TEST_PASS=false\n", stderr);
        return 1;
    }
    puts("MINI224_PERTURBED_TEST_PASS=true");
    return 0;
}
