#include "vae_c_iic_mini112_perturbed.h"

#include "vae_augmentation.h"
#include "vae_c_sha256.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool report_ok(
    const VAECIICMini112PerturbedReport *report,
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

/* Technical benchmark only: two full logical batches and the final tail. */
static int benchmark(const char *microbatch_text, const char *repository_root)
{
    VAECIICMini112Perturbed source;
    VAECIICMini112PerturbedReport report;
    size_t indices[VAE_C_IIC_MINI112_LOGICAL_BATCH];
    float mu[263u * VAE_C_IIC_MINI112_LATENT_DIM];
    char digest[VAE_C_SHA256_HEX_SIZE];
    char *end;
    unsigned long microbatch;
    size_t group;
    size_t index;
    size_t offset = 0u;
    int success = 0;
    microbatch = strtoul(microbatch_text, &end, 10);
    if (*end != '\0' || (microbatch != 64u && microbatch != 32u &&
                          microbatch != 16u)) {
        return 2;
    }
    vae_c_iic_mini112_perturbed_init(&source);
    if (!vae_c_iic_mini112_perturbed_open(&source, repository_root)) {
        goto done;
    }
    for (group = 0u; group < 3u; ++group) {
        const size_t count = group == 2u ? 7u : 128u;
        const size_t first = group == 2u ?
            VAE_C_IIC_MINI112_ROWS - 7u : group * 128u;
        for (index = 0u; index < count; ++index) {
            indices[index] = first + index;
        }
        if (!vae_c_iic_mini112_perturbed_compute(
                &source, indices, count, 0u, (size_t)microbatch,
                mu + offset * VAE_C_IIC_MINI112_LATENT_DIM,
                NULL, &report) ||
            !report_ok(&report, count,
                       (count + (size_t)microbatch - 1u) /
                           (size_t)microbatch)) {
            goto done;
        }
        offset += count;
    }
    if (offset != 263u ||
        vae_c_sha256_digest_hex_lower(mu, sizeof(mu), digest) !=
            VAE_C_SHA256_OK) {
        goto done;
    }
    printf("BENCH_MICROBATCH=%lu\nBENCH_SAMPLES=%llu\nBENCH_MU_SHA256=%s\n",
           microbatch, (unsigned long long)offset, digest);
    success = 1;
done:
    vae_c_iic_mini112_perturbed_free(&source);
    return success ? 0 : 1;
}

int main(int argc, char **argv)
{
    VAECIICMini112Perturbed source;
    VAECIICMini112PerturbedReport report;
    size_t indices[VAE_C_IIC_MINI112_LOGICAL_BATCH];
    size_t reverse_indices[VAE_C_IIC_MINI112_LOGICAL_BATCH];
    size_t tail_indices[7];
    float mu64[VAE_C_IIC_MINI112_LOGICAL_BATCH *
               VAE_C_IIC_MINI112_LATENT_DIM];
    float mu32[VAE_C_IIC_MINI112_LOGICAL_BATCH *
               VAE_C_IIC_MINI112_LATENT_DIM];
    float mu16[VAE_C_IIC_MINI112_LOGICAL_BATCH *
               VAE_C_IIC_MINI112_LATENT_DIM];
    float mu_reverse[VAE_C_IIC_MINI112_LOGICAL_BATCH *
                     VAE_C_IIC_MINI112_LATENT_DIM];
    float tail7[7u * VAE_C_IIC_MINI112_LATENT_DIM];
    float tail3[7u * VAE_C_IIC_MINI112_LATENT_DIM];
    int crops64[VAE_C_IIC_MINI112_LOGICAL_BATCH];
    int crops32[VAE_C_IIC_MINI112_LOGICAL_BATCH];
    int crops16[VAE_C_IIC_MINI112_LOGICAL_BATCH];
    int crops_reverse[VAE_C_IIC_MINI112_LOGICAL_BATCH];
    int crop_tail7[7];
    int crop_tail3[7];
    char digest[VAE_C_SHA256_HEX_SIZE];
    size_t index;
    bool differing_epoch_crop = false;
    int success = 0;

    if (argc == 4 && strcmp(argv[1], "--benchmark") == 0) {
        return benchmark(argv[2], argv[3]);
    }
    if (argc != 2) {
        return 2;
    }
    vae_c_iic_mini112_perturbed_init(&source);
    if (!vae_c_iic_mini112_perturbed_open(&source, argv[1])) {
        goto done;
    }
    for (index = 0u; index < VAE_C_IIC_MINI112_LOGICAL_BATCH; index++) {
        indices[index] = index;
        reverse_indices[index] = VAE_C_IIC_MINI112_LOGICAL_BATCH - 1u - index;
    }
    for (index = 0u; index < 7u; index++) {
        tail_indices[index] = VAE_C_IIC_MINI112_ROWS - 7u + index;
    }
    if (!vae_c_iic_mini112_perturbed_compute(
            &source, indices, VAE_C_IIC_MINI112_LOGICAL_BATCH, 0u,
            64u, mu64, crops64, &report) ||
        !report_ok(&report, VAE_C_IIC_MINI112_LOGICAL_BATCH, 2u) ||
        !vae_c_iic_mini112_perturbed_compute(
            &source, indices, VAE_C_IIC_MINI112_LOGICAL_BATCH, 0u,
            32u, mu32, crops32, &report) ||
        !report_ok(&report, VAE_C_IIC_MINI112_LOGICAL_BATCH, 4u) ||
        !vae_c_iic_mini112_perturbed_compute(
            &source, indices, VAE_C_IIC_MINI112_LOGICAL_BATCH, 0u,
            16u, mu16, crops16, &report) ||
        !report_ok(&report, VAE_C_IIC_MINI112_LOGICAL_BATCH, 8u) ||
        memcmp(mu64, mu32, sizeof(mu64)) != 0 ||
        memcmp(mu64, mu16, sizeof(mu64)) != 0 ||
        memcmp(crops64, crops32, sizeof(crops64)) != 0 ||
        memcmp(crops64, crops16, sizeof(crops64)) != 0 ||
        !vae_c_iic_mini112_perturbed_compute(
            &source, reverse_indices, VAE_C_IIC_MINI112_LOGICAL_BATCH,
            0u, 32u, mu_reverse, crops_reverse, &report)) {
        goto done;
    }
    for (index = 0u; index < VAE_C_IIC_MINI112_LOGICAL_BATCH; index++) {
        const size_t original = VAE_C_IIC_MINI112_LOGICAL_BATCH - 1u - index;
        if (memcmp(mu_reverse + index * VAE_C_IIC_MINI112_LATENT_DIM,
                   mu64 + original * VAE_C_IIC_MINI112_LATENT_DIM,
                   VAE_C_IIC_MINI112_LATENT_DIM * sizeof(float)) != 0 ||
            crops_reverse[index] != crops64[original]) {
            goto done;
        }
    }
    for (index = 0u; index < VAE_C_IIC_MINI112_LOGICAL_BATCH; index++) {
        int expected_crop;
        int next_epoch_crop;
        size_t latent;
        if (gs_augment_crop_left_for_event_mode(
                VAE_C_IIC_MINI112_AUGMENTATION_SEED, 0u,
                (uint64_t)indices[index],
                GS_AUGMENT_MODE_SYMMETRIC_FIGURE,
                &expected_crop) != 0 ||
            gs_augment_crop_left_for_event_mode(
                VAE_C_IIC_MINI112_AUGMENTATION_SEED, 1u,
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
        for (latent = 0u; latent < VAE_C_IIC_MINI112_LATENT_DIM;
             latent++) {
            if (!isfinite(mu64[index * VAE_C_IIC_MINI112_LATENT_DIM +
                               latent])) {
                goto done;
            }
        }
    }
    if (!differing_epoch_crop ||
        !vae_c_iic_mini112_perturbed_compute(
            &source, tail_indices, 7u, 1u, 7u, tail7,
            crop_tail7, &report) ||
        !report_ok(&report, 7u, 1u) ||
        !vae_c_iic_mini112_perturbed_compute(
            &source, tail_indices, 7u, 1u, 3u, tail3,
            crop_tail3, &report) ||
        !report_ok(&report, 7u, 3u) ||
        memcmp(tail7, tail3, sizeof(tail7)) != 0 ||
        memcmp(crop_tail7, crop_tail3, sizeof(crop_tail7)) != 0) {
        goto done;
    }
    indices[0] = VAE_C_IIC_MINI112_ROWS;
    mu32[0] = -123.0f;
    if (vae_c_iic_mini112_perturbed_compute(
            &source, indices, 1u, 0u, 1u, mu32, NULL,
            &report) || mu32[0] != -123.0f ||
        vae_c_iic_mini112_perturbed_compute(
            &source, tail_indices, 7u, 0u, 65u, mu32,
            NULL, &report)) {
        goto done;
    }
    if (vae_c_sha256_digest_hex_lower(
            mu64, sizeof(mu64), digest) != VAE_C_SHA256_OK) {
        goto done;
    }
    printf("PERTURBED_B128_MICROBATCH_64_32_16_BITWISE=true\n");
    printf("PERTURBED_ORDER_INDEPENDENT=true\n");
    printf("PERTURBED_TAIL7_MICROBATCH_7_3_BITWISE=true\n");
    printf("PERTURBED_EVAL_PARAMS_BN_UNCHANGED=true\n");
    printf("PERTURBED_DECODER_UNTOUCHED=true\n");
    printf("PERTURBED_SHA256=%s\n", digest);
    success = 1;
done:
    vae_c_iic_mini112_perturbed_free(&source);
    if (!success) {
        fputs("PERTURBED_TEST_PASS=false\n", stderr);
        return 1;
    }
    puts("PERTURBED_TEST_PASS=true");
    return 0;
}
