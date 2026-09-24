#include "vae_c_iic_mini75_central.h"
#include "vae_c_iic_mini75_perturbed.h"

#include "vae_c_iic_ensemble.h"
#include "vae_c_sha256.h"
#include "vae_c_shuffle.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

#define MINI75_IIC_EPOCHS UINT64_C(100)
#define MINI75_IIC_SHUFFLE_SEED UINT64_C(101112)
#define MINI75_IIC_BATCH 128u
#define MINI75_IIC_ENCODER_MICROBATCH 64u

typedef enum {
    MODE_VALIDATE = 1,
    MODE_SMOKE = 2,
    MODE_BENCHMARK = 3
} RunMode;

static bool parse_args(
    int argc,
    char **argv,
    RunMode *mode,
    const char **repository_root,
    size_t *benchmark_batches
)
{
    if (argc < 3 || mode == NULL || repository_root == NULL ||
        benchmark_batches == NULL) {
        return false;
    }
    *repository_root = argv[argc - 1];
    *benchmark_batches = 0u;
    if (argc == 3 && strcmp(argv[1], "--validate-only") == 0) {
        *mode = MODE_VALIDATE;
        return true;
    }
    if (argc == 3 && strcmp(argv[1], "--smoke") == 0) {
        *mode = MODE_SMOKE;
        return true;
    }
    if (argc == 4 && strcmp(argv[1], "--benchmark-batches") == 0 &&
        strlen(argv[2]) == 1u && argv[2][0] >= '1' &&
        argv[2][0] <= '4') {
        *mode = MODE_BENCHMARK;
        *benchmark_batches = (size_t)(argv[2][0] - '0');
        return true;
    }
    return false;
}

static bool validate_permutations(
    const VAECIICMini75Central *central,
    const VAECIICMini75Perturbed *perturbed,
    size_t *epoch0
)
{
    size_t *epoch1 = NULL;
    unsigned char *seen = NULL;
    size_t index;
    bool differs = false;
    bool result = false;

    epoch1 = (size_t *)malloc(VAE_C_IIC_MINI75_ROWS * sizeof(size_t));
    seen = (unsigned char *)calloc(VAE_C_IIC_MINI75_ROWS,
                                   sizeof(unsigned char));
    if (epoch0 == NULL || epoch1 == NULL || seen == NULL ||
        vae_c_shuffle_make_permutation(
            VAE_C_IIC_MINI75_ROWS, MINI75_IIC_SHUFFLE_SEED,
            0u, epoch0) != VAE_C_RNG_OK ||
        vae_c_shuffle_make_permutation(
            VAE_C_IIC_MINI75_ROWS, MINI75_IIC_SHUFFLE_SEED,
            1u, epoch1) != VAE_C_RNG_OK) {
        goto done;
    }
    for (index = 0u; index < VAE_C_IIC_MINI75_ROWS; index++) {
        const VAECIICMini75CentralRow *row;
        const GSEventRecord *event;
        size_t manifest_index = epoch0[index];
        if (manifest_index >= VAE_C_IIC_MINI75_ROWS ||
            seen[manifest_index] != 0u ||
            epoch1[index] >= VAE_C_IIC_MINI75_ROWS) {
            goto done;
        }
        seen[manifest_index] = 1u;
        row = &central->rows[manifest_index];
        event = gs_event_manifest_get(&perturbed->events, manifest_index);
        if (event == NULL || row->manifest_index != manifest_index ||
            row->event_index != (uint32_t)event->event_index ||
            strcmp(row->event_id, event->event_id) != 0 ||
            vae_c_iic_mini75_central_row_mu(
                central, manifest_index) == NULL) {
            goto done;
        }
        if (epoch1[index] != manifest_index) {
            differs = true;
        }
    }
    memset(seen, 0, VAE_C_IIC_MINI75_ROWS);
    for (index = 0u; index < VAE_C_IIC_MINI75_ROWS; index++) {
        if (seen[epoch1[index]] != 0u) {
            goto done;
        }
        seen[epoch1[index]] = 1u;
    }
    result = differs;
done:
    free(seen);
    free(epoch1);
    return result;
}

static bool source_identity(
    const VAECIICMini75Central *central,
    const VAECIICMini75Perturbed *perturbed,
    char digest[VAE_C_SHA256_HEX_SIZE],
    char run_id[96]
)
{
    char preimage[2048];
    int written = snprintf(
        preimage, sizeof(preimage),
        "format=MINI75_IIC_REAL_INPUT_V1\n"
        "population=transductive_manifest_index_0_8582\n"
        "profile=mini75_experiment_v1\n"
        "vae_checkpoint_kind=BEST\n"
        "vae_checkpoint_epoch=47\n"
        "vae_checkpoint_step=5184\n"
        "vae_checkpoint_sha256=%s\n"
        "event_manifest_sha256=%s\n"
        "tensor_manifest_sha256=%s\n"
        "train_split_sha256=%s\n"
        "validation_split_sha256=%s\n"
        "train_mu_sha256=%s\n"
        "validation_mu_sha256=%s\n"
        "train_rows_sha256=%s\n"
        "validation_rows_sha256=%s\n"
        "augmentation=stateless_symmetric_figure_seed789\n"
        "shuffle=epoch_stateless_seed101112\n"
        "central_view=precomputed_posterior_mu\n"
        "perturbed_view=frozen_encoder_eval_posterior_mu\n"
        "main=36\nover=250\npairs=5\n"
        "pair_seeds=123,124,125,126,127\n"
        "adam=0.0005,0.9,0.999,1e-8\n"
        "max_epochs=100\nlogical_batch=128\n"
        "encoder_microbatch=64\n"
        "init_policy=splitmix64_raw_seed_main_then_over_v1\n",
        central->checkpoint_sha256,
        perturbed->contract.expected_identity.event_manifest_sha256,
        perturbed->contract.expected_identity.tensor_manifest_sha256,
        perturbed->contract.expected_identity.train_split_sha256,
        perturbed->contract.expected_identity.validation_split_sha256,
        central->train_matrix_sha256,
        central->validation_matrix_sha256,
        central->train_rows_sha256,
        central->validation_rows_sha256
    );
    if (written < 0 || (size_t)written >= sizeof(preimage) ||
        vae_c_sha256_digest_hex_lower(
            preimage, (size_t)written, digest) != VAE_C_SHA256_OK ||
        snprintf(run_id, 96u, "mini75_iic_v1_%s", digest) != 78) {
        return false;
    }
    return true;
}

static bool smoke(
    VAECIICMini75Perturbed *perturbed,
    const VAECIICMini75Central *central,
    const size_t *permutation
)
{
    float first[MINI75_IIC_BATCH * VAE_C_IIC_MINI75_LATENT_DIM];
    float second[MINI75_IIC_BATCH * VAE_C_IIC_MINI75_LATENT_DIM];
    float tail_a[7u * VAE_C_IIC_MINI75_LATENT_DIM];
    float tail_b[7u * VAE_C_IIC_MINI75_LATENT_DIM];
    int crops_a[MINI75_IIC_BATCH];
    int crops_b[MINI75_IIC_BATCH];
    VAECIICMini75PerturbedReport report;
    size_t index;

    if (!vae_c_iic_mini75_perturbed_compute(
            perturbed, permutation, MINI75_IIC_BATCH,
            0u, 64u, first, crops_a, &report) ||
        report.encoder_forward_calls != 2u ||
        !report.inputs_bitwise_unchanged ||
        !report.model_parameters_bn_bitwise_unchanged ||
        !report.decoder_workspace_untouched ||
        !report.optimizer_unchanged ||
        !vae_c_iic_mini75_perturbed_compute(
            perturbed, permutation, MINI75_IIC_BATCH,
            0u, 32u, second, crops_b, &report) ||
        report.encoder_forward_calls != 4u ||
        memcmp(first, second, sizeof(first)) != 0 ||
        memcmp(crops_a, crops_b, sizeof(crops_a)) != 0 ||
        !vae_c_iic_mini75_perturbed_compute(
            perturbed, permutation +
                VAE_C_IIC_MINI75_ROWS - 7u,
            7u, 0u, 7u, tail_a, NULL, &report) ||
        !vae_c_iic_mini75_perturbed_compute(
            perturbed, permutation +
                VAE_C_IIC_MINI75_ROWS - 7u,
            7u, 0u, 3u, tail_b, NULL, &report) ||
        memcmp(tail_a, tail_b, sizeof(tail_a)) != 0) {
        return false;
    }
    for (index = 0u; index < MINI75_IIC_BATCH; index++) {
        const float *central_mu = vae_c_iic_mini75_central_row_mu(
            central, permutation[index]);
        size_t latent;
        if (central_mu == NULL || crops_a[index] < 0 ||
            crops_a[index] > 48) {
            return false;
        }
        for (latent = 0u; latent < VAE_C_IIC_MINI75_LATENT_DIM;
             latent++) {
            if (!isfinite(central_mu[latent]) ||
                !isfinite(first[index *
                                VAE_C_IIC_MINI75_LATENT_DIM + latent])) {
                return false;
            }
        }
    }
    puts("MICROBATCH_64_32_BITWISE=true");
    puts("TAIL7_MICROBATCH_INVARIANT=true");
    puts("CENTRAL_PERTURBED_MANIFEST_PAIRING=true");
    puts("ENCODER_EVAL_PARAMS_BN_UNCHANGED=true");
    puts("DECODER_EXECUTED=false");
    return true;
}

static double elapsed_seconds(const struct timespec *first,
                              const struct timespec *last)
{
    return (double)(last->tv_sec - first->tv_sec) +
           (double)(last->tv_nsec - first->tv_nsec) / 1.0e9;
}

static bool benchmark(
    VAECIICMini75Perturbed *perturbed,
    const size_t *permutation,
    size_t batches
)
{
    float mu[MINI75_IIC_BATCH * VAE_C_IIC_MINI75_LATENT_DIM];
    VAECIICMini75PerturbedReport report;
    struct timespec first;
    struct timespec last;
    size_t batch;
    double seconds;
    if (timespec_get(&first, TIME_UTC) != TIME_UTC) {
        return false;
    }
    for (batch = 0u; batch < batches; batch++) {
        if (!vae_c_iic_mini75_perturbed_compute(
                perturbed, permutation + batch * MINI75_IIC_BATCH,
                MINI75_IIC_BATCH, 0u, MINI75_IIC_ENCODER_MICROBATCH,
                mu, NULL, &report)) {
            return false;
        }
    }
    if (timespec_get(&last, TIME_UTC) != TIME_UTC) {
        return false;
    }
    seconds = elapsed_seconds(&first, &last);
    if (!isfinite(seconds) || seconds <= 0.0) {
        return false;
    }
    printf("BENCHMARK_BATCHES=%zu\n", batches);
    printf("BENCHMARK_SAMPLES=%zu\n", batches * MINI75_IIC_BATCH);
    printf("BENCHMARK_SECONDS=%.6f\n", seconds);
    printf("BENCHMARK_PERTURBED_SAMPLE_PER_SECOND=%.6f\n",
           (double)(batches * MINI75_IIC_BATCH) / seconds);
#ifdef _WIN32
    {
        PROCESS_MEMORY_COUNTERS_EX counters;
        memset(&counters, 0, sizeof(counters));
        counters.cb = sizeof(counters);
        if (GetProcessMemoryInfo(
                GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&counters,
                sizeof(counters)) != 0) {
            printf("PEAK_WORKING_SET_BYTES=%zu\n",
                   (size_t)counters.PeakWorkingSetSize);
            printf("PRIVATE_BYTES_AT_END=%zu\n",
                   (size_t)counters.PrivateUsage);
        }
    }
#endif
    return true;
}

int main(int argc, char **argv)
{
    RunMode mode;
    const char *repository_root;
    size_t benchmark_batches;
    VAECIICMini75Central central;
    VAECIICMini75Perturbed perturbed;
    size_t *permutation = NULL;
    char error[512];
    char digest[VAE_C_SHA256_HEX_SIZE];
    char run_id[96];
    int success = 0;

    if (!parse_args(argc, argv, &mode, &repository_root,
                    &benchmark_batches)) {
        fputs("Usage: validate_vae_c_iic_mini75_input "
              "<--validate-only|--smoke|--benchmark-batches 1..4> "
              "<repository-root>\n", stderr);
        return 2;
    }
    vae_c_iic_mini75_central_init(&central);
    vae_c_iic_mini75_perturbed_init(&perturbed);
    error[0] = '\0';
    permutation = (size_t *)malloc(
        VAE_C_IIC_MINI75_ROWS * sizeof(size_t));
    if (permutation == NULL) {
        fputs("MINI75_IIC_INPUT_VALIDATION_FAILED: permutation allocation\n", stderr);
        goto done;
    }
    if (!vae_c_iic_mini75_central_load(
            repository_root, NULL, &central,
            error, sizeof(error))) {
        fprintf(stderr, "MINI75_IIC_INPUT_VALIDATION_FAILED: central: %s\n", error);
        goto done;
    }
    if (
        central.row_count != VAE_C_IIC_MINI75_ROWS ||
        central.latent_dim != VAE_C_IIC_MINI75_LATENT_DIM) {
        fputs("MINI75_IIC_INPUT_VALIDATION_FAILED: central shape\n", stderr);
        goto done;
    }
    if (!vae_c_iic_mini75_perturbed_open(&perturbed, repository_root)) {
        fputs("MINI75_IIC_INPUT_VALIDATION_FAILED: perturbed source\n", stderr);
        goto done;
    }
    if (
        strcmp(central.checkpoint_sha256,
               perturbed.contract.checkpoint_sha256) != 0) {
        fputs("MINI75_IIC_INPUT_VALIDATION_FAILED: checkpoint identity\n", stderr);
        goto done;
    }
    if (!validate_permutations(&central, &perturbed, permutation)) {
        fputs("MINI75_IIC_INPUT_VALIDATION_FAILED: permutations or row pairing\n", stderr);
        goto done;
    }
    if (!source_identity(&central, &perturbed, digest, run_id)) {
        fputs("MINI75_IIC_INPUT_VALIDATION_FAILED: source identity\n", stderr);
        goto done;
    }
    printf("VALIDATE_ONLY=%s\n", mode == MODE_VALIDATE ? "true" : "false");
    printf("POPULATION_ROWS=%zu\n", central.row_count);
    printf("ROW_ORDER=manifest_index_0_8582\n");
    printf("CENTRAL_SOURCE=precomputed_train_validation_mu\n");
    printf("PERTURBED_SOURCE=out272_gray_stateless_symmetric_figure\n");
    printf("VAE_CHECKPOINT_KIND=BEST\nVAE_CHECKPOINT_EPOCH=47\n");
    printf("VAE_CHECKPOINT_STEP=5184\nVAE_CHECKPOINT_SHA256=%s\n",
           central.checkpoint_sha256);
    printf("TRAIN_MU_SHA256=%s\nVALIDATION_MU_SHA256=%s\n",
           central.train_matrix_sha256,
           central.validation_matrix_sha256);
    printf("TRAIN_ROWS_SHA256=%s\nVALIDATION_ROWS_SHA256=%s\n",
           central.train_rows_sha256,
           central.validation_rows_sha256);
    printf("EVENT_MANIFEST_SHA256=%s\nTENSOR_MANIFEST_SHA256=%s\n",
           perturbed.contract.expected_identity.event_manifest_sha256,
           perturbed.contract.expected_identity.tensor_manifest_sha256);
    printf("SOURCE_IDENTITY_SHA256=%s\nRUN_ID_PROPOSED=%s\n",
           digest, run_id);
    printf("K=%d\nC=%d\nW=%d\n", VAE_C_IIC_ENSEMBLE_K,
           VAE_C_IIC_ENSEMBLE_MAIN_DIM,
           VAE_C_IIC_ENSEMBLE_OVER_DIM);
    printf("MAX_EPOCHS=100\nLOGICAL_BATCH=128\n");
    printf("ENCODER_MICROBATCH=64\nTAIL=7\n");
    printf("SHUFFLE_SEED=101112\nPAIR_SEEDS=123,124,125,126,127\n");
    printf("AUGMENTATION_MODE=symmetric_figure\nAUGMENTATION_SEED=789\n");
    puts("SAME_PERMUTATION_FOR_ALL_PAIRS=true");
    puts("RUN_CONFIG_PRODUCER_HEAD_REQUIRED=true");
    puts("CHECKPOINT_FORMAT=IIC_V1_LATEST_FINAL_NO_BEST");
    if (mode == MODE_VALIDATE) {
        puts("ENCODER_FORWARD_CALLS=0");
        puts("IIC_OPTIMIZER_STEPS=0");
        puts("OUTPUT_CREATED=false");
    } else if (!smoke(&perturbed, &central, permutation)) {
        fputs("MINI75_IIC_SMOKE_FAILED\n", stderr);
        goto done;
    } else if (mode == MODE_BENCHMARK &&
               !benchmark(&perturbed, permutation,
                          benchmark_batches)) {
        fputs("MINI75_IIC_BENCHMARK_FAILED\n", stderr);
        goto done;
    }
    puts("MINI75_IIC_INPUT_PASS=true");
    success = 1;
done:
    free(permutation);
    vae_c_iic_mini75_perturbed_free(&perturbed);
    vae_c_iic_mini75_central_free(&central);
    return success ? 0 : 1;
}
