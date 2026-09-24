#include "vae_c_iic_mini75_perturbed.h"

#include "vae_augmentation.h"
#include "vae_c_sha256.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static bool report_ok(
    const VAECIICMini75PerturbedReport *report,
    size_t rows,
    uint64_t calls
)
{
    return report->samples == rows &&
           report->encoder_forward_calls == calls &&
           report->inputs_bitwise_unchanged &&
           report->model_parameters_bn_bitwise_unchanged &&
           report->decoder_workspace_untouched &&
           report->optimizer_unchanged;
}

int main(int argc, char **argv)
{
    VAECIICMini75Perturbed source;
    VAECIICMini75PerturbedReport report;
    size_t indices[VAE_C_IIC_MINI75_LOGICAL_BATCH];
    size_t tail_indices[7];
    float mu64[VAE_C_IIC_MINI75_LOGICAL_BATCH *
               VAE_C_IIC_MINI75_LATENT_DIM];
    float mu32[VAE_C_IIC_MINI75_LOGICAL_BATCH *
               VAE_C_IIC_MINI75_LATENT_DIM];
    float tail7[7u * VAE_C_IIC_MINI75_LATENT_DIM];
    float tail3[7u * VAE_C_IIC_MINI75_LATENT_DIM];
    int crops64[VAE_C_IIC_MINI75_LOGICAL_BATCH];
    int crops32[VAE_C_IIC_MINI75_LOGICAL_BATCH];
    int crop_tail7[7];
    int crop_tail3[7];
    char digest[VAE_C_SHA256_HEX_SIZE];
    size_t index;
    bool differing_epoch_crop = false;
    int success = 0;

    if (argc != 2) {
        return 2;
    }
    vae_c_iic_mini75_perturbed_init(&source);
    if (!vae_c_iic_mini75_perturbed_open(&source, argv[1])) {
        goto done;
    }
    for (index = 0u; index < VAE_C_IIC_MINI75_LOGICAL_BATCH; index++) {
        indices[index] = index;
    }
    for (index = 0u; index < 7u; index++) {
        tail_indices[index] = VAE_C_IIC_MINI75_ROWS - 7u + index;
    }
    if (!vae_c_iic_mini75_perturbed_compute(
            &source, indices, VAE_C_IIC_MINI75_LOGICAL_BATCH, 0u,
            64u, mu64, crops64, &report) ||
        !report_ok(&report, VAE_C_IIC_MINI75_LOGICAL_BATCH, 2u) ||
        !vae_c_iic_mini75_perturbed_compute(
            &source, indices, VAE_C_IIC_MINI75_LOGICAL_BATCH, 0u,
            32u, mu32, crops32, &report) ||
        !report_ok(&report, VAE_C_IIC_MINI75_LOGICAL_BATCH, 4u) ||
        memcmp(mu64, mu32, sizeof(mu64)) != 0 ||
        memcmp(crops64, crops32, sizeof(crops64)) != 0) {
        goto done;
    }
    for (index = 0u; index < VAE_C_IIC_MINI75_LOGICAL_BATCH; index++) {
        int expected_crop;
        int next_epoch_crop;
        size_t latent;
        if (gs_augment_crop_left_for_event_mode(
                VAE_C_IIC_MINI75_AUGMENTATION_SEED, 0u,
                (uint64_t)indices[index],
                GS_AUGMENT_MODE_SYMMETRIC_FIGURE,
                &expected_crop) != 0 ||
            gs_augment_crop_left_for_event_mode(
                VAE_C_IIC_MINI75_AUGMENTATION_SEED, 1u,
                (uint64_t)indices[index],
                GS_AUGMENT_MODE_SYMMETRIC_FIGURE,
                &next_epoch_crop) != 0 ||
            crops64[index] != expected_crop || expected_crop < 0 ||
            expected_crop > 48) {
            goto done;
        }
        if (next_epoch_crop != expected_crop) {
            differing_epoch_crop = true;
        }
        for (latent = 0u; latent < VAE_C_IIC_MINI75_LATENT_DIM;
             latent++) {
            if (!isfinite(mu64[index * VAE_C_IIC_MINI75_LATENT_DIM +
                               latent])) {
                goto done;
            }
        }
    }
    if (!differing_epoch_crop ||
        !vae_c_iic_mini75_perturbed_compute(
            &source, tail_indices, 7u, 1u, 7u, tail7,
            crop_tail7, &report) ||
        !report_ok(&report, 7u, 1u) ||
        !vae_c_iic_mini75_perturbed_compute(
            &source, tail_indices, 7u, 1u, 3u, tail3,
            crop_tail3, &report) ||
        !report_ok(&report, 7u, 3u) ||
        memcmp(tail7, tail3, sizeof(tail7)) != 0 ||
        memcmp(crop_tail7, crop_tail3, sizeof(crop_tail7)) != 0) {
        goto done;
    }
    indices[0] = VAE_C_IIC_MINI75_ROWS;
    mu32[0] = -123.0f;
    if (vae_c_iic_mini75_perturbed_compute(
            &source, indices, 1u, 0u, 1u, mu32, NULL,
            &report) || mu32[0] != -123.0f ||
        vae_c_iic_mini75_perturbed_compute(
            &source, tail_indices, 7u, 0u, 65u, mu32,
            NULL, &report)) {
        goto done;
    }
    if (vae_c_sha256_digest_hex_lower(
            mu64, sizeof(mu64), digest) != VAE_C_SHA256_OK) {
        goto done;
    }
    printf("PERTURBED_B128_MICROBATCH_64_32_BITWISE=true\n");
    printf("PERTURBED_TAIL7_MICROBATCH_7_3_BITWISE=true\n");
    printf("PERTURBED_EVAL_PARAMS_BN_UNCHANGED=true\n");
    printf("PERTURBED_DECODER_UNTOUCHED=true\n");
    printf("PERTURBED_SHA256=%s\n", digest);
    success = 1;
done:
    vae_c_iic_mini75_perturbed_free(&source);
    if (!success) {
        fputs("PERTURBED_TEST_PASS=false\n", stderr);
        return 1;
    }
    puts("PERTURBED_TEST_PASS=true");
    return 0;
}
