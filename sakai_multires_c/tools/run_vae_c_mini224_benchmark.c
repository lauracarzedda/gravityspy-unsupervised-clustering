#if !defined(_WIN32)
#define _POSIX_C_SOURCE 200809L
#endif

#include "vae_augmentation.h"
#include "vae_c_adam.h"
#include "vae_c_config.h"
#include "vae_c_model_forward.h"
#include "vae_c_model_train.h"
#include "vae_c_rng.h"
#include "vae_c_sha256.h"
#include "vae_event_dataset.h"
#include "vae_multires_batch.h"
#include "vae_multires_event_loader.h"
#include "vae_saved_split.h"
#include "vae_tensor_manifest.h"
#include "vae_tensor_spec.h"

#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#endif

enum {
    BENCH_PATH_CAP = 4096,
    BENCH_SIDE = 224,
    BENCH_INPUT_VALUES = 4 * 224 * 224,
    BENCH_LATENT = 64,
    BENCH_TRAIN_SUBSET_COUNT = 800,
    BENCH_VALIDATION_SUBSET_COUNT = 200,
    BENCH_TRAIN_SAMPLE_COUNT = 128,
    BENCH_VALIDATION_SAMPLE_COUNT = 64,
    BENCH_MAX_BATCH = 32,
    BENCH_MAX_MEASUREMENTS = 32,
    BENCH_MEMORY_SAMPLE_CAP = 128
};

#define BENCH_PROFILE_ID "mini224_bridge_v1"
#define BENCH_INIT_SEED UINT64_C(123)
#define BENCH_LATENT_SEED UINT64_C(456)
#define BENCH_AUGMENTATION_SEED UINT64_C(789)
#define BENCH_SHUFFLE_SEED UINT64_C(101112)
#define BENCH_EPOCH UINT64_C(0)

#define BENCH_EVENT_SHA \
    "620960e6b35da31bc4cf1a8a6da8679027d378e1d4b2ac73c5267c1c5311ae82"
#define BENCH_TENSOR_SHA \
    "94b7e2e051bd32de72dd2cb6a5fb7eda9bcaec123cbf3a07d0d901bd19779afa"
#define BENCH_TRAIN_SUBSET_SHA \
    "80c5396a70e7f2c8e3a00680582945581ae76c301a5b9ba8c62e006bb25cdfb5"
#define BENCH_VALIDATION_SUBSET_SHA \
    "9f5fe6c50c7398588f2de7bd4a543a54a53e7f7a0b9a84bf4004a01f065ae76f"
#define BENCH_PROFILE_SHA \
    "7ac5e1b5f5a76265144e9a09ce8e1e1ec976ace76f664f86b432a20a45a19ad1"

typedef struct {
    VAECProfileCatalog catalog;
    const VAECProfile *profile;
    GSEventManifest events;
    GSSavedSplit train_subset;
    GSSavedSplit validation_subset;
    GSTensorManifest tensor_manifest;
    GSMultiresEventLoader loader;
} BenchmarkContract;

typedef struct {
    VAECModel model;
    VAECAdam optimizer;
    VAECModelWorkspace workspace;
    GSMultiresBatch batch;
    uint64_t *manifest_indices;
    float *validation_logits;
} BenchmarkCandidate;

typedef struct {
    double loader[BENCH_MAX_MEASUREMENTS];
    double neural[BENCH_MAX_MEASUREMENTS];
    double end_to_end[BENCH_MAX_MEASUREMENTS];
    size_t count;
    double loader_total;
    double neural_total;
    double end_to_end_total;
} BenchmarkTimes;

typedef struct {
    uint64_t working_samples[BENCH_MEMORY_SAMPLE_CAP];
    uint64_t private_samples[BENCH_MEMORY_SAMPLE_CAP];
    size_t sample_count;
    uint64_t working_start;
    uint64_t working_end;
    uint64_t working_peak;
    uint64_t private_start;
    uint64_t private_end;
    uint64_t private_peak;
    uint64_t train_private_first;
    uint64_t train_private_last;
} BenchmarkMemory;

typedef struct {
    BenchmarkTimes train_times;
    BenchmarkTimes validation_times;
    BenchmarkMemory memory;
    double train_total;
    double train_reconstruction;
    double train_kl;
    double validation_total;
    double validation_reconstruction;
    double validation_kl;
    uint64_t lower_bound_bytes;
    uint64_t optimizer_step_before;
    uint64_t optimizer_step_after;
    bool parameters_changed;
    bool validation_repeat_bitwise;
    bool validation_persistent_unchanged;
} BenchmarkResult;

static void benchmark_contract_init(BenchmarkContract *contract)
{
    memset(contract, 0, sizeof(*contract));
    vae_c_profile_catalog_init(&contract->catalog);
}

static void benchmark_contract_free(BenchmarkContract *contract)
{
    gs_multires_event_loader_free(&contract->loader);
    gs_tensor_manifest_free(&contract->tensor_manifest);
    gs_saved_split_free(&contract->validation_subset);
    gs_saved_split_free(&contract->train_subset);
    gs_event_manifest_free(&contract->events);
    vae_c_profile_catalog_free(&contract->catalog);
    memset(contract, 0, sizeof(*contract));
}

static void benchmark_candidate_init(BenchmarkCandidate *candidate)
{
    memset(candidate, 0, sizeof(*candidate));
    vae_c_model_init(&candidate->model);
    vae_c_adam_init(&candidate->optimizer);
    vae_c_model_workspace_init(&candidate->workspace);
}

static void benchmark_candidate_free(BenchmarkCandidate *candidate)
{
    free(candidate->validation_logits);
    free(candidate->manifest_indices);
    gs_multires_batch_free(&candidate->batch);
    vae_c_model_workspace_free(&candidate->workspace);
    vae_c_adam_free(&candidate->optimizer);
    vae_c_model_free(&candidate->model);
    memset(candidate, 0, sizeof(*candidate));
}

static int benchmark_path_join(
    const char *directory,
    const char *name,
    char path[BENCH_PATH_CAP]
)
{
    int length;
    size_t directory_length;

    if (directory == NULL || name == NULL || path == NULL) {
        return 1;
    }
    directory_length = strlen(directory);
    length = snprintf(path, BENCH_PATH_CAP, "%s%s%s", directory,
        directory_length != 0u &&
        directory[directory_length - 1u] != '/' &&
        directory[directory_length - 1u] != '\\' ? "/" : "", name);
    return length < 0 || length >= BENCH_PATH_CAP ? 1 : 0;
}

static int benchmark_file_sha256(const char *path, char hex[65])
{
    FILE *file = NULL;
    unsigned char *data = NULL;
    long length;
    size_t size;
    int result = 1;

    file = fopen(path, "rb");
    if (file == NULL || fseek(file, 0, SEEK_END) != 0 ||
        (length = ftell(file)) < 0 || fseek(file, 0, SEEK_SET) != 0) {
        if (file != NULL) {
            (void)fclose(file);
        }
        return 1;
    }
    size = (size_t)length;
    if ((long)size != length) {
        (void)fclose(file);
        return 1;
    }
    if (size != 0u) {
        data = (unsigned char *)malloc(size);
        if (data == NULL || fread(data, 1u, size, file) != size ||
            fgetc(file) != EOF || ferror(file)) {
            free(data);
            (void)fclose(file);
            return 1;
        }
    }
    if (fclose(file) == 0 &&
        vae_c_sha256_digest_hex_lower(data, size, hex) == VAE_C_SHA256_OK) {
        result = 0;
    }
    free(data);
    return result;
}

static int benchmark_hash_is(const char *path, const char *expected)
{
    char actual[65];
    return benchmark_file_sha256(path, actual) == 0 &&
        strcmp(actual, expected) == 0;
}

static int benchmark_profile_is_expected(const VAECProfile *profile)
{
    return profile != NULL &&
        strcmp(profile->profile_id, BENCH_PROFILE_ID) == 0 &&
        profile->input_side == BENCH_SIDE && profile->conv_depth == 4u &&
        profile->channels[0] == 16u && profile->channels[1] == 32u &&
        profile->channels[2] == 64u && profile->channels[3] == 128u &&
        profile->latent_dim == BENCH_LATENT &&
        profile->bottleneck_side == 7u &&
        profile->expected_trainable_parameters_bn1d == UINT64_C(625860) &&
        profile->expected_forward_mac_per_sample == UINT64_C(220790784);
}

static int benchmark_load_contract(
    BenchmarkContract *contract,
    const char *root
)
{
    char profile_path[BENCH_PATH_CAP];
    char event_path[BENCH_PATH_CAP];
    char tensor_path[BENCH_PATH_CAP];
    char train_subset_path[BENCH_PATH_CAP];
    char validation_subset_path[BENCH_PATH_CAP];
    char error[512];
    GSTensorSpec spec;

    if (benchmark_path_join(root,
            "sakai_multires_c/configs/vae_c_profiles_v2.csv",
            profile_path) ||
        benchmark_path_join(root, "manifest_events_272.csv", event_path) ||
        benchmark_path_join(root, "manifest_tensors_224_central.csv",
                            tensor_path) ||
        benchmark_path_join(root,
            "sakai_multires_c/configs/subset224_train_800_seed123.csv",
            train_subset_path) ||
        benchmark_path_join(root,
            "sakai_multires_c/configs/subset224_val_200_seed123.csv",
            validation_subset_path) ||
        !benchmark_hash_is(profile_path, BENCH_PROFILE_SHA) ||
        !benchmark_hash_is(event_path, BENCH_EVENT_SHA) ||
        !benchmark_hash_is(tensor_path, BENCH_TENSOR_SHA) ||
        !benchmark_hash_is(train_subset_path, BENCH_TRAIN_SUBSET_SHA) ||
        !benchmark_hash_is(validation_subset_path,
                           BENCH_VALIDATION_SUBSET_SHA)) {
        fprintf(stderr, "BENCHMARK_ERROR=canonical_path_or_sha\n");
        return 1;
    }
    error[0] = '\0';
    if (vae_c_profile_catalog_load_csv(profile_path, &contract->catalog,
            error, sizeof(error)) != VAE_C_CONFIG_OK ||
        (contract->profile = vae_c_profile_catalog_find(
            &contract->catalog, BENCH_PROFILE_ID)) == NULL ||
        !benchmark_profile_is_expected(contract->profile) ||
        gs_event_manifest_load(event_path, &contract->events) != 0 ||
        contract->events.count != 8583u ||
        gs_saved_split_load(train_subset_path, "train", &contract->events,
                            &contract->train_subset) != 0 ||
        contract->train_subset.count != BENCH_TRAIN_SUBSET_COUNT ||
        gs_saved_split_load(validation_subset_path, "validation",
                            &contract->events,
                            &contract->validation_subset) != 0 ||
        contract->validation_subset.count !=
            BENCH_VALIDATION_SUBSET_COUNT ||
        gs_tensor_spec_init(&spec, BENCH_SIDE) != 0 ||
        gs_tensor_manifest_load(tensor_path, &spec,
                                &contract->tensor_manifest) != 0 ||
        contract->tensor_manifest.count != contract->events.count ||
        contract->tensor_manifest.spec.values_per_event !=
            BENCH_INPUT_VALUES ||
        gs_multires_event_loader_init(&contract->loader, BENCH_SIDE) != 0) {
        fprintf(stderr, "BENCHMARK_ERROR=contract_load:%s\n", error);
        return 1;
    }
    return 0;
}

static int benchmark_batch_is_allowed(size_t batch_size)
{
    return batch_size == 4u || batch_size == 8u ||
        batch_size == 16u || batch_size == 32u;
}

static double benchmark_now(void)
{
#if defined(_WIN32)
    LARGE_INTEGER counter;
    LARGE_INTEGER frequency;
    if (!QueryPerformanceFrequency(&frequency) ||
        !QueryPerformanceCounter(&counter) || frequency.QuadPart <= 0) {
        return -1.0;
    }
    return (double)counter.QuadPart / (double)frequency.QuadPart;
#else
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return -1.0;
    }
    return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
#endif
}

static int benchmark_memory_read(
    uint64_t *working,
    uint64_t *peak_working,
    uint64_t *private_bytes
)
{
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX counters;
    memset(&counters, 0, sizeof(counters));
    counters.cb = sizeof(counters);
    if (!GetProcessMemoryInfo(GetCurrentProcess(),
            (PROCESS_MEMORY_COUNTERS *)&counters,
            (DWORD)sizeof(counters))) {
        return 1;
    }
    *working = (uint64_t)counters.WorkingSetSize;
    *peak_working = (uint64_t)counters.PeakWorkingSetSize;
    *private_bytes = (uint64_t)counters.PrivateUsage;
#else
    *working = UINT64_C(0);
    *peak_working = UINT64_C(0);
    *private_bytes = UINT64_C(0);
#endif
    return 0;
}

static int benchmark_memory_sample(BenchmarkMemory *memory)
{
    uint64_t working;
    uint64_t peak_working;
    uint64_t private_bytes;

    if (memory->sample_count >= BENCH_MEMORY_SAMPLE_CAP ||
        benchmark_memory_read(&working, &peak_working, &private_bytes)) {
        return 1;
    }
    if (memory->sample_count == 0u) {
        memory->working_start = working;
        memory->private_start = private_bytes;
    }
    memory->working_samples[memory->sample_count] = working;
    memory->private_samples[memory->sample_count] = private_bytes;
    memory->sample_count++;
    memory->working_end = working;
    memory->private_end = private_bytes;
    if (peak_working > memory->working_peak) {
        memory->working_peak = peak_working;
    }
    if (private_bytes > memory->private_peak) {
        memory->private_peak = private_bytes;
    }
    return 0;
}

static uint64_t benchmark_u64_median(
    const uint64_t *values,
    size_t count
)
{
    uint64_t copy[BENCH_MEMORY_SAMPLE_CAP];
    size_t index;
    size_t position;

    if (count == 0u || count > BENCH_MEMORY_SAMPLE_CAP) {
        return UINT64_C(0);
    }
    memcpy(copy, values, count * sizeof(uint64_t));
    for (index = 1u; index < count; index++) {
        uint64_t value = copy[index];
        position = index;
        while (position > 0u && copy[position - 1u] > value) {
            copy[position] = copy[position - 1u];
            position--;
        }
        copy[position] = value;
    }
    if ((count & 1u) != 0u) {
        return copy[count / 2u];
    }
    return copy[count / 2u - 1u] / UINT64_C(2) +
        copy[count / 2u] / UINT64_C(2) +
        ((copy[count / 2u - 1u] & UINT64_C(1)) +
         (copy[count / 2u] & UINT64_C(1))) / UINT64_C(2);
}

static double benchmark_double_median(const double *values, size_t count)
{
    double copy[BENCH_MAX_MEASUREMENTS];
    size_t index;
    size_t position;

    if (count == 0u || count > BENCH_MAX_MEASUREMENTS) {
        return NAN;
    }
    memcpy(copy, values, count * sizeof(double));
    for (index = 1u; index < count; index++) {
        double value = copy[index];
        position = index;
        while (position > 0u && copy[position - 1u] > value) {
            copy[position] = copy[position - 1u];
            position--;
        }
        copy[position] = value;
    }
    if ((count & 1u) != 0u) {
        return copy[count / 2u];
    }
    return 0.5 * (copy[count / 2u - 1u] + copy[count / 2u]);
}

static int benchmark_values_finite(const float *values, size_t count)
{
    size_t index;
    if (values == NULL) {
        return 0;
    }
    for (index = 0u; index < count; index++) {
        if (!isfinite(values[index])) {
            return 0;
        }
    }
    return 1;
}

static int benchmark_values_unit_interval(
    const float *values,
    size_t count
)
{
    size_t index;
    if (values == NULL) {
        return 0;
    }
    for (index = 0u; index < count; index++) {
        if (!isfinite(values[index]) ||
            values[index] < 0.0f || values[index] > 1.0f) {
            return 0;
        }
    }
    return 1;
}

static size_t benchmark_collect_tensors(
    const VAECModelWorkspace *workspace,
    const VAECTensor *tensors[VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS]
)
{
    size_t count = 0u;
    size_t index;
    tensors[count++] = &workspace->input;
    for (index = 0u; index < workspace->config.conv_depth; index++) {
        tensors[count++] = &workspace->encoder_conv_outputs[index];
        tensors[count++] = &workspace->encoder_batch_norm_outputs[index];
        tensors[count++] = &workspace->encoder_relu_outputs[index];
    }
    tensors[count++] = &workspace->encoder_pool_output;
    tensors[count++] = &workspace->encoder_global_avg_output;
    tensors[count++] = &workspace->encoder_packed_output;
    tensors[count++] = &workspace->mu;
    tensors[count++] = &workspace->logvar;
    tensors[count++] = &workspace->epsilon;
    tensors[count++] = &workspace->z;
    tensors[count++] = &workspace->decoder_fc_output;
    tensors[count++] = &workspace->decoder_fc_batch_norm_output;
    tensors[count++] = &workspace->decoder_fc_relu_output;
    for (index = 0u; index < workspace->config.decoder_upsample_count;
         index++) {
        tensors[count++] = &workspace->decoder_upsample_outputs[index];
    }
    for (index = 0u; index < workspace->config.decoder_hidden_conv_count;
         index++) {
        tensors[count++] = &workspace->decoder_hidden_conv_outputs[index];
        tensors[count++] =
            &workspace->decoder_hidden_batch_norm_outputs[index];
        tensors[count++] = &workspace->decoder_hidden_relu_outputs[index];
    }
    tensors[count++] = &workspace->logits;
    return count;
}

static int benchmark_gradients_finite(
    const VAECModel *model,
    const VAECModelWorkspace *workspace
)
{
    const VAECTensor *tensors[VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS];
    size_t tensor_count = benchmark_collect_tensors(workspace, tensors);
    size_t index;

    if (tensor_count != workspace->owned_tensor_count) {
        return 0;
    }
    for (index = 0u; index < model->parameter_count; index++) {
        const VAECParameter *parameter = model->parameters[index];
        if (!benchmark_values_finite(parameter->gradients,
                                     parameter->shape.count)) {
            return 0;
        }
    }
    for (index = 0u; index < tensor_count; index++) {
        if (tensors[index]->has_grad &&
            !benchmark_values_finite(tensors[index]->grad,
                                     tensors[index]->shape.count)) {
            return 0;
        }
    }
    return 1;
}

static int benchmark_parameters_finite(const VAECModel *model)
{
    size_t index;
    for (index = 0u; index < model->parameter_count; index++) {
        const VAECParameter *parameter = model->parameters[index];
        if (!benchmark_values_finite(parameter->values,
                                     parameter->shape.count) ||
            !benchmark_values_finite(parameter->adam_m,
                                     parameter->shape.count) ||
            !benchmark_values_finite(parameter->adam_v,
                                     parameter->shape.count)) {
            return 0;
        }
    }
    return 1;
}

static int benchmark_stats_valid(
    const VAECVAELossStats *stats,
    size_t batch_size
)
{
    return stats != NULL && stats->batch_size == batch_size &&
        stats->reconstruction.total_values ==
            batch_size * (size_t)BENCH_INPUT_VALUES &&
        stats->kl.total_values == batch_size * (size_t)BENCH_LATENT &&
        isfinite(stats->total_sum) &&
        isfinite(stats->reconstruction.total_sum) &&
        isfinite(stats->kl.total_sum);
}

static size_t benchmark_parameter_float_count(const VAECModel *model)
{
    size_t total = 0u;
    size_t index;
    for (index = 0u; index < model->parameter_count; index++) {
        total += model->parameters[index]->shape.count;
    }
    return total;
}

static void benchmark_copy_parameters(const VAECModel *model, float *copy)
{
    size_t offset = 0u;
    size_t index;
    for (index = 0u; index < model->parameter_count; index++) {
        size_t count = model->parameters[index]->shape.count;
        memcpy(copy + offset, model->parameters[index]->values,
               count * sizeof(float));
        offset += count;
    }
}

static int benchmark_parameters_equal(
    const VAECModel *model,
    const float *copy
)
{
    size_t offset = 0u;
    size_t index;
    for (index = 0u; index < model->parameter_count; index++) {
        size_t count = model->parameters[index]->shape.count;
        if (memcmp(copy + offset, model->parameters[index]->values,
                   count * sizeof(float)) != 0) {
            return 0;
        }
        offset += count;
    }
    return 1;
}

static size_t benchmark_bn_float_count(const VAECModel *model)
{
    size_t total = model->decoder_fc_batch_norm.num_features;
    size_t index;
    for (index = 0u; index < model->config.conv_depth; index++) {
        total += model->encoder_batch_norms[index].num_features;
    }
    for (index = 0u; index < model->config.decoder_hidden_conv_count;
         index++) {
        total += model->decoder_hidden_batch_norms[index].num_features;
    }
    return total * 2u;
}

static void benchmark_copy_bn_state(const VAECModel *model, float *copy)
{
    size_t offset = 0u;
    size_t index;
#define BENCH_COPY_BN(layer) do { \
    size_t bench_n = (layer).num_features; \
    memcpy(copy + offset, (layer).running_mean, \
           bench_n * sizeof(float)); \
    offset += bench_n; \
    memcpy(copy + offset, (layer).running_variance, \
           bench_n * sizeof(float)); \
    offset += bench_n; \
} while (0)
    for (index = 0u; index < model->config.conv_depth; index++) {
        BENCH_COPY_BN(model->encoder_batch_norms[index]);
    }
    BENCH_COPY_BN(model->decoder_fc_batch_norm);
    for (index = 0u; index < model->config.decoder_hidden_conv_count;
         index++) {
        BENCH_COPY_BN(model->decoder_hidden_batch_norms[index]);
    }
#undef BENCH_COPY_BN
}

static int benchmark_candidate_allocate(
    BenchmarkCandidate *candidate,
    const BenchmarkContract *contract,
    size_t batch_size
)
{
    size_t input_count = batch_size * (size_t)BENCH_INPUT_VALUES;
    if (vae_c_model_allocate(&candidate->model, contract->profile,
            batch_size) != VAE_C_OP_OK ||
        vae_c_model_initialize_parameters_v1(
            &candidate->model, BENCH_INIT_SEED) != VAE_C_OP_OK ||
        vae_c_adam_configure_v1(&candidate->optimizer) != VAE_C_OP_OK ||
        vae_c_model_workspace_allocate(&candidate->model, batch_size,
            &candidate->workspace) != VAE_C_OP_OK ||
        gs_multires_batch_alloc(&candidate->batch,
            &contract->tensor_manifest.spec, batch_size) != 0) {
        return 1;
    }
    candidate->manifest_indices =
        (uint64_t *)malloc(batch_size * sizeof(uint64_t));
    candidate->validation_logits =
        (float *)malloc(input_count * sizeof(float));
    return candidate->manifest_indices == NULL ||
        candidate->validation_logits == NULL ? 1 : 0;
}

static int benchmark_validate_train_batch(
    const BenchmarkContract *contract,
    const BenchmarkCandidate *candidate,
    size_t start,
    size_t batch_size
)
{
    size_t sample;
    size_t input_count = batch_size * (size_t)BENCH_INPUT_VALUES;
    if (candidate->batch.count != batch_size ||
        !benchmark_values_unit_interval(candidate->batch.x, input_count)) {
        return 1;
    }
    for (sample = 0u; sample < batch_size; sample++) {
        size_t expected_manifest = contract->train_subset.indices[start + sample];
        int expected_crop = -1;
        if (candidate->batch.manifest_indices[sample] != expected_manifest ||
            gs_augment_crop_left_for_event_mode(
                BENCH_AUGMENTATION_SEED, BENCH_EPOCH,
                (uint64_t)expected_manifest,
                GS_AUGMENT_MODE_SYMMETRIC_FIGURE,
                &expected_crop) != 0 ||
            expected_crop < 0 || expected_crop > 48 ||
            candidate->batch.crop_lefts[sample] != expected_crop) {
            return 1;
        }
    }
    return 0;
}

static int benchmark_validate_validation_batch(
    const BenchmarkContract *contract,
    const BenchmarkCandidate *candidate,
    size_t start,
    size_t batch_size
)
{
    size_t sample;
    size_t input_count = batch_size * (size_t)BENCH_INPUT_VALUES;
    if (candidate->batch.count != batch_size ||
        !benchmark_values_unit_interval(candidate->batch.x, input_count)) {
        return 1;
    }
    for (sample = 0u; sample < batch_size; sample++) {
        if (candidate->batch.manifest_indices[sample] !=
                contract->validation_subset.indices[start + sample]) {
            return 1;
        }
    }
    return 0;
}

static int benchmark_fill_validation_epsilon(
    BenchmarkCandidate *candidate,
    size_t batch_size
)
{
    size_t sample;
    for (sample = 0u; sample < batch_size; sample++) {
        size_t manifest_index = candidate->batch.manifest_indices[sample];
        if (vae_c_rng_fill_latent_validation_epsilon_f32(
                BENCH_LATENT_SEED, (uint64_t)manifest_index,
                candidate->workspace.epsilon.data +
                    sample * (size_t)BENCH_LATENT,
                BENCH_LATENT) != VAE_C_RNG_OK) {
            return 1;
        }
    }
    return 0;
}

static int benchmark_warmup(
    BenchmarkContract *contract,
    BenchmarkCandidate *candidate,
    size_t batch_size
)
{
    VAECVAELossStats stats;
    size_t sample;
    size_t input_bytes = batch_size * (size_t)BENCH_INPUT_VALUES *
        sizeof(float);

    if (gs_multires_batch_load_train_stateless_mode(
            &contract->events, &contract->train_subset, &contract->loader,
            0u, batch_size, BENCH_AUGMENTATION_SEED, BENCH_EPOCH,
            GS_AUGMENT_MODE_SYMMETRIC_FIGURE,
            &candidate->batch) != 0 ||
        benchmark_validate_train_batch(
            contract, candidate, 0u, batch_size)) {
        return 1;
    }
    memcpy(candidate->workspace.input.data, candidate->batch.x, input_bytes);
    for (sample = 0u; sample < batch_size; sample++) {
        candidate->manifest_indices[sample] =
            (uint64_t)candidate->batch.manifest_indices[sample];
    }
    vae_c_vae_loss_stats_init(&stats);
    if (vae_c_model_train_step(&candidate->model, &candidate->workspace,
            &candidate->optimizer, BENCH_LATENT_SEED, BENCH_EPOCH,
            candidate->manifest_indices, batch_size,
            &stats) != VAE_C_OP_OK ||
        candidate->optimizer.step != UINT64_C(1) ||
        !benchmark_stats_valid(&stats, batch_size)) {
        return 1;
    }
    if (gs_multires_batch_load_validation(
            &contract->events, &contract->tensor_manifest,
            &contract->validation_subset, 0u, batch_size,
            &candidate->batch) != 0 ||
        benchmark_validate_validation_batch(
            contract, candidate, 0u, batch_size)) {
        return 1;
    }
    memcpy(candidate->workspace.input.data, candidate->batch.x, input_bytes);
    if (benchmark_fill_validation_epsilon(candidate, batch_size)) {
        return 1;
    }
    vae_c_vae_loss_stats_init(&stats);
    return vae_c_model_vae_forward(&candidate->model,
        VAE_C_BATCH_NORM_MODE_EVAL, VAE_C_LATENT_MODE_STOCHASTIC,
        &candidate->workspace, &stats) != VAE_C_OP_OK ||
        !benchmark_stats_valid(&stats, batch_size) ? 1 : 0;
}

static int benchmark_warm_loader_cache(
    BenchmarkContract *contract,
    BenchmarkCandidate *candidate,
    size_t batch_size
)
{
    size_t start;

    for (start = 0u; start < BENCH_TRAIN_SAMPLE_COUNT;
         start += batch_size) {
        if (gs_multires_batch_load_train_stateless_mode(
                &contract->events, &contract->train_subset,
                &contract->loader, start, batch_size,
                BENCH_AUGMENTATION_SEED, BENCH_EPOCH,
                GS_AUGMENT_MODE_SYMMETRIC_FIGURE,
                &candidate->batch) != 0 ||
            benchmark_validate_train_batch(
                contract, candidate, start, batch_size)) {
            return 1;
        }
    }
    for (start = 0u; start < BENCH_VALIDATION_SAMPLE_COUNT;
         start += batch_size) {
        if (gs_multires_batch_load_validation(
                &contract->events, &contract->tensor_manifest,
                &contract->validation_subset, start, batch_size,
                &candidate->batch) != 0 ||
            benchmark_validate_validation_batch(
                contract, candidate, start, batch_size)) {
            return 1;
        }
    }
    return 0;
}

static int benchmark_record_time(
    BenchmarkTimes *times,
    double loader,
    double neural,
    double end_to_end
)
{
    size_t index = times->count;
    if (index >= BENCH_MAX_MEASUREMENTS || loader < 0.0 ||
        neural < 0.0 || end_to_end < 0.0 ||
        !isfinite(loader) || !isfinite(neural) ||
        !isfinite(end_to_end)) {
        return 1;
    }
    times->loader[index] = loader;
    times->neural[index] = neural;
    times->end_to_end[index] = end_to_end;
    times->loader_total += loader;
    times->neural_total += neural;
    times->end_to_end_total += end_to_end;
    times->count++;
    return 0;
}

static int benchmark_train_pass(
    BenchmarkContract *contract,
    BenchmarkCandidate *candidate,
    size_t batch_size,
    BenchmarkResult *result
)
{
    size_t start;
    size_t sample;
    size_t parameter_count = benchmark_parameter_float_count(
        &candidate->model);
    size_t input_bytes = batch_size * (size_t)BENCH_INPUT_VALUES *
        sizeof(float);
    float *parameters_before = (float *)malloc(
        parameter_count * sizeof(float));
    int failed = 0;

    if (parameters_before == NULL) {
        return 1;
    }
    benchmark_copy_parameters(&candidate->model, parameters_before);
    result->optimizer_step_before = candidate->optimizer.step;
    for (start = 0u; start < BENCH_TRAIN_SAMPLE_COUNT;
         start += batch_size) {
        VAECVAELossStats stats;
        double e2e_begin = benchmark_now();
        double loader_begin = e2e_begin;
        double loader_end;
        double neural_begin;
        double neural_end;
        double e2e_end;

        if (e2e_begin < 0.0 ||
            gs_multires_batch_load_train_stateless_mode(
                &contract->events, &contract->train_subset,
                &contract->loader, start, batch_size,
                BENCH_AUGMENTATION_SEED, BENCH_EPOCH,
                GS_AUGMENT_MODE_SYMMETRIC_FIGURE,
                &candidate->batch) != 0) {
            failed = 1;
            break;
        }
        loader_end = benchmark_now();
        if (loader_end < 0.0) {
            failed = 1;
            break;
        }
        memcpy(candidate->workspace.input.data,
               candidate->batch.x, input_bytes);
        for (sample = 0u; sample < batch_size; sample++) {
            candidate->manifest_indices[sample] =
                (uint64_t)candidate->batch.manifest_indices[sample];
        }
        vae_c_vae_loss_stats_init(&stats);
        neural_begin = benchmark_now();
        if (neural_begin < 0.0 ||
            vae_c_model_train_step(&candidate->model,
                &candidate->workspace, &candidate->optimizer,
                BENCH_LATENT_SEED, BENCH_EPOCH,
                candidate->manifest_indices, batch_size,
                &stats) != VAE_C_OP_OK) {
            failed = 1;
            break;
        }
        neural_end = benchmark_now();
        e2e_end = neural_end;
        if (neural_end < 0.0 ||
            benchmark_record_time(&result->train_times,
                loader_end - loader_begin, neural_end - neural_begin,
                e2e_end - e2e_begin) ||
            benchmark_validate_train_batch(
                contract, candidate, start, batch_size) ||
            !benchmark_stats_valid(&stats, batch_size) ||
            !benchmark_gradients_finite(
                &candidate->model, &candidate->workspace) ||
            !benchmark_parameters_finite(&candidate->model) ||
            memcmp(candidate->workspace.input.data,
                   candidate->batch.x, input_bytes) != 0 ||
            benchmark_memory_sample(&result->memory)) {
            failed = 1;
            break;
        }
        result->train_total += stats.total_sum;
        result->train_reconstruction += stats.reconstruction.total_sum;
        result->train_kl += stats.kl.total_sum;
        if (start == 0u) {
            result->memory.train_private_first =
                result->memory.private_end;
        }
        result->memory.train_private_last = result->memory.private_end;
    }
    result->optimizer_step_after = candidate->optimizer.step;
    result->parameters_changed =
        !benchmark_parameters_equal(&candidate->model, parameters_before);
    if (!result->parameters_changed ||
        result->train_times.count != BENCH_TRAIN_SAMPLE_COUNT / batch_size ||
        result->optimizer_step_after - result->optimizer_step_before !=
            (uint64_t)result->train_times.count) {
        failed = 1;
    }
    free(parameters_before);
    return failed;
}

static int benchmark_validation_pass(
    BenchmarkContract *contract,
    BenchmarkCandidate *candidate,
    size_t batch_size,
    BenchmarkResult *result
)
{
    size_t start;
    size_t parameter_count = benchmark_parameter_float_count(
        &candidate->model);
    size_t bn_count = benchmark_bn_float_count(&candidate->model);
    size_t input_bytes = batch_size * (size_t)BENCH_INPUT_VALUES *
        sizeof(float);
    float *parameters_before = (float *)malloc(
        parameter_count * sizeof(float));
    float *bn_before = (float *)malloc(bn_count * sizeof(float));
    uint64_t optimizer_step = candidate->optimizer.step;
    int failed = 0;

    if (parameters_before == NULL || bn_before == NULL) {
        free(bn_before);
        free(parameters_before);
        return 1;
    }
    benchmark_copy_parameters(&candidate->model, parameters_before);
    benchmark_copy_bn_state(&candidate->model, bn_before);
    result->validation_repeat_bitwise = true;
    for (start = 0u; start < BENCH_VALIDATION_SAMPLE_COUNT;
         start += batch_size) {
        VAECVAELossStats stats_a;
        VAECVAELossStats stats_b;
        double e2e_begin = benchmark_now();
        double loader_begin = e2e_begin;
        double loader_end;
        double neural_begin;
        double neural_end;
        double e2e_end;

        if (e2e_begin < 0.0 ||
            gs_multires_batch_load_validation(
                &contract->events, &contract->tensor_manifest,
                &contract->validation_subset, start, batch_size,
                &candidate->batch) != 0) {
            failed = 1;
            break;
        }
        loader_end = benchmark_now();
        if (loader_end < 0.0) {
            failed = 1;
            break;
        }
        memcpy(candidate->workspace.input.data,
               candidate->batch.x, input_bytes);
        if (benchmark_fill_validation_epsilon(candidate, batch_size)) {
            failed = 1;
            break;
        }
        vae_c_vae_loss_stats_init(&stats_a);
        neural_begin = benchmark_now();
        if (neural_begin < 0.0 ||
            vae_c_model_vae_forward(&candidate->model,
                VAE_C_BATCH_NORM_MODE_EVAL,
                VAE_C_LATENT_MODE_STOCHASTIC,
                &candidate->workspace, &stats_a) != VAE_C_OP_OK) {
            failed = 1;
            break;
        }
        neural_end = benchmark_now();
        e2e_end = neural_end;
        if (neural_end < 0.0 ||
            benchmark_record_time(&result->validation_times,
                loader_end - loader_begin, neural_end - neural_begin,
                e2e_end - e2e_begin) ||
            benchmark_validate_validation_batch(
                contract, candidate, start, batch_size) ||
            !benchmark_stats_valid(&stats_a, batch_size) ||
            !benchmark_values_finite(candidate->workspace.logits.data,
                candidate->workspace.logits.shape.count)) {
            failed = 1;
            break;
        }
        memcpy(candidate->validation_logits,
               candidate->workspace.logits.data, input_bytes);
        if (benchmark_fill_validation_epsilon(candidate, batch_size)) {
            failed = 1;
            break;
        }
        vae_c_vae_loss_stats_init(&stats_b);
        if (vae_c_model_vae_forward(&candidate->model,
                VAE_C_BATCH_NORM_MODE_EVAL,
                VAE_C_LATENT_MODE_STOCHASTIC,
                &candidate->workspace, &stats_b) != VAE_C_OP_OK ||
            memcmp(&stats_a, &stats_b, sizeof(stats_a)) != 0 ||
            memcmp(candidate->validation_logits,
                   candidate->workspace.logits.data, input_bytes) != 0) {
            result->validation_repeat_bitwise = false;
            failed = 1;
            break;
        }
        result->validation_total += stats_a.total_sum;
        result->validation_reconstruction +=
            stats_a.reconstruction.total_sum;
        result->validation_kl += stats_a.kl.total_sum;
        if (benchmark_memory_sample(&result->memory)) {
            failed = 1;
            break;
        }
    }
    result->validation_persistent_unchanged =
        candidate->optimizer.step == optimizer_step &&
        benchmark_parameters_equal(&candidate->model, parameters_before);
    if (result->validation_persistent_unchanged) {
        float *bn_after = (float *)malloc(bn_count * sizeof(float));
        if (bn_after == NULL) {
            result->validation_persistent_unchanged = false;
        } else {
            benchmark_copy_bn_state(&candidate->model, bn_after);
            result->validation_persistent_unchanged =
                memcmp(bn_before, bn_after,
                       bn_count * sizeof(float)) == 0;
            free(bn_after);
        }
    }
    if (!result->validation_repeat_bitwise ||
        !result->validation_persistent_unchanged ||
        result->validation_times.count !=
            BENCH_VALIDATION_SAMPLE_COUNT / batch_size) {
        failed = 1;
    }
    free(bn_before);
    free(parameters_before);
    return failed;
}

static void benchmark_u64_le(unsigned char target[8], uint64_t value)
{
    size_t byte;
    for (byte = 0u; byte < 8u; byte++) {
        target[byte] = (unsigned char)(value >> (8u * byte));
    }
}

static void benchmark_u32_le(unsigned char target[4], uint32_t value)
{
    size_t byte;
    for (byte = 0u; byte < 4u; byte++) {
        target[byte] = (unsigned char)(value >> (8u * byte));
    }
}

static int benchmark_fairness_digests(
    const BenchmarkContract *contract,
    char train_sequence[65],
    char train_crop_map[65],
    char validation_sequence[65]
)
{
    unsigned char train_bytes[BENCH_TRAIN_SAMPLE_COUNT * 8u];
    unsigned char crop_bytes[BENCH_TRAIN_SAMPLE_COUNT * 12u];
    unsigned char validation_bytes[BENCH_VALIDATION_SAMPLE_COUNT * 8u];
    size_t index;

    for (index = 0u; index < BENCH_TRAIN_SAMPLE_COUNT; index++) {
        uint64_t manifest = (uint64_t)contract->train_subset.indices[index];
        int crop = -1;
        if (gs_augment_crop_left_for_event_mode(
                BENCH_AUGMENTATION_SEED, BENCH_EPOCH, manifest,
                GS_AUGMENT_MODE_SYMMETRIC_FIGURE, &crop) != 0 ||
            crop < 0 || crop > 48) {
            return 1;
        }
        benchmark_u64_le(train_bytes + index * 8u, manifest);
        benchmark_u64_le(crop_bytes + index * 12u, manifest);
        benchmark_u32_le(crop_bytes + index * 12u + 8u,
                         (uint32_t)crop);
    }
    for (index = 0u; index < BENCH_VALIDATION_SAMPLE_COUNT; index++) {
        benchmark_u64_le(validation_bytes + index * 8u,
            (uint64_t)contract->validation_subset.indices[index]);
    }
    return vae_c_sha256_digest_hex_lower(
            train_bytes, sizeof(train_bytes), train_sequence) !=
            VAE_C_SHA256_OK ||
        vae_c_sha256_digest_hex_lower(
            crop_bytes, sizeof(crop_bytes), train_crop_map) !=
            VAE_C_SHA256_OK ||
        vae_c_sha256_digest_hex_lower(
            validation_bytes, sizeof(validation_bytes),
            validation_sequence) != VAE_C_SHA256_OK ? 1 : 0;
}

static int benchmark_run(
    BenchmarkContract *contract,
    size_t batch_size,
    BenchmarkResult *result
)
{
    BenchmarkCandidate candidate;
    size_t parameter_count;
    size_t batch_values;
    uint64_t parameter_bytes;
    uint64_t batch_bytes;
    char train_sequence[65];
    char train_crop_map[65];
    char validation_sequence[65];
    int status = 1;

    benchmark_candidate_init(&candidate);
    memset(result, 0, sizeof(*result));
    if (benchmark_candidate_allocate(
            &candidate, contract, batch_size) ||
        benchmark_fairness_digests(contract, train_sequence,
            train_crop_map, validation_sequence)) {
        fprintf(stderr, "BENCHMARK_ERROR=allocation_or_fairness_digest\n");
        goto cleanup;
    }
    parameter_count = benchmark_parameter_float_count(&candidate.model);
    batch_values = batch_size * (size_t)BENCH_INPUT_VALUES;
    parameter_bytes = (uint64_t)parameter_count * UINT64_C(4) *
        (uint64_t)sizeof(float);
    batch_bytes = (uint64_t)batch_values * (uint64_t)sizeof(float);
    result->lower_bound_bytes =
        (uint64_t)candidate.workspace.storage_bytes + parameter_bytes +
        batch_bytes * UINT64_C(2) +
        (uint64_t)parameter_count * (uint64_t)sizeof(float);

    if (benchmark_warmup(contract, &candidate, batch_size) ||
        benchmark_warm_loader_cache(contract, &candidate, batch_size)) {
        fprintf(stderr, "BENCHMARK_ERROR=warmup\n");
        goto cleanup;
    }
    memset(&result->memory, 0, sizeof(result->memory));
    if (benchmark_memory_sample(&result->memory) ||
        benchmark_train_pass(contract, &candidate, batch_size, result) ||
        benchmark_validation_pass(
            contract, &candidate, batch_size, result) ||
        benchmark_memory_sample(&result->memory)) {
        fprintf(stderr, "BENCHMARK_ERROR=timed_pass_or_numeric_gate\n");
        goto cleanup;
    }

    printf("PROFILE_ID=%s\n", BENCH_PROFILE_ID);
    printf("BATCH_SIZE=%lu\n", (unsigned long)batch_size);
    printf("INIT_SEED=%" PRIu64 "\n", BENCH_INIT_SEED);
    printf("LATENT_SEED=%" PRIu64 "\n", BENCH_LATENT_SEED);
    printf("AUGMENTATION_SEED=%" PRIu64 "\n",
           BENCH_AUGMENTATION_SEED);
    printf("SHUFFLE_SEED=%" PRIu64 "\n", BENCH_SHUFFLE_SEED);
    printf("BENCHMARK_ORDER=canonical_prefix_no_shuffle\n");
    printf("EPOCH_KEY=%" PRIu64 "\n", BENCH_EPOCH);
    printf("TRAIN_SAMPLE_COUNT=%u\n", BENCH_TRAIN_SAMPLE_COUNT);
    printf("VALIDATION_SAMPLE_COUNT=%u\n",
           BENCH_VALIDATION_SAMPLE_COUNT);
    printf("TRAIN_EVENT_SEQUENCE_SHA256=%s\n", train_sequence);
    printf("TRAIN_CROP_MAP_SHA256=%s\n", train_crop_map);
    printf("VALIDATION_EVENT_SEQUENCE_SHA256=%s\n",
           validation_sequence);
    printf("TRAIN_EVENT_SEQUENCE_VALID=true\n");
    printf("VALIDATION_EVENT_SEQUENCE_VALID=true\n");
    printf("CROP_EVENT_EPOCH_KEY_VALID=true\n");
    printf("WARMUP_TRAIN_STEPS=1\n");
    printf("TIMED_TRAIN_STEPS=%lu\n",
           (unsigned long)result->train_times.count);
    printf("TIMED_VALIDATION_BATCHES=%lu\n",
           (unsigned long)result->validation_times.count);
    printf("OPTIMIZER_STEP_BEFORE_TIMED=%" PRIu64 "\n",
           result->optimizer_step_before);
    printf("OPTIMIZER_STEP_AFTER_TIMED=%" PRIu64 "\n",
           result->optimizer_step_after);
    printf("TRAIN_LOADER_SECONDS=%.9f\n",
           result->train_times.loader_total);
    printf("TRAIN_NEURAL_SECONDS=%.9f\n",
           result->train_times.neural_total);
    printf("TRAIN_END_TO_END_SECONDS=%.9f\n",
           result->train_times.end_to_end_total);
    printf("TRAIN_LOADER_MEDIAN_BATCH_SECONDS=%.9f\n",
           benchmark_double_median(result->train_times.loader,
                                   result->train_times.count));
    printf("TRAIN_NEURAL_MEDIAN_BATCH_SECONDS=%.9f\n",
           benchmark_double_median(result->train_times.neural,
                                   result->train_times.count));
    printf("TRAIN_END_TO_END_MEDIAN_BATCH_SECONDS=%.9f\n",
           benchmark_double_median(result->train_times.end_to_end,
                                   result->train_times.count));
    printf("TRAIN_NEURAL_SAMPLES_PER_SECOND=%.9f\n",
           (double)BENCH_TRAIN_SAMPLE_COUNT /
               result->train_times.neural_total);
    printf("TRAIN_END_TO_END_SAMPLES_PER_SECOND=%.9f\n",
           (double)BENCH_TRAIN_SAMPLE_COUNT /
               result->train_times.end_to_end_total);
    printf("VALIDATION_LOADER_SECONDS=%.9f\n",
           result->validation_times.loader_total);
    printf("VALIDATION_NEURAL_SECONDS=%.9f\n",
           result->validation_times.neural_total);
    printf("VALIDATION_END_TO_END_SECONDS=%.9f\n",
           result->validation_times.end_to_end_total);
    printf("VALIDATION_LOADER_MEDIAN_BATCH_SECONDS=%.9f\n",
           benchmark_double_median(result->validation_times.loader,
                                   result->validation_times.count));
    printf("VALIDATION_NEURAL_MEDIAN_BATCH_SECONDS=%.9f\n",
           benchmark_double_median(result->validation_times.neural,
                                   result->validation_times.count));
    printf("VALIDATION_END_TO_END_MEDIAN_BATCH_SECONDS=%.9f\n",
           benchmark_double_median(result->validation_times.end_to_end,
                                   result->validation_times.count));
    printf("VALIDATION_NEURAL_SAMPLES_PER_SECOND=%.9f\n",
           (double)BENCH_VALIDATION_SAMPLE_COUNT /
               result->validation_times.neural_total);
    printf("VALIDATION_END_TO_END_SAMPLES_PER_SECOND=%.9f\n",
           (double)BENCH_VALIDATION_SAMPLE_COUNT /
               result->validation_times.end_to_end_total);
    printf("TRAIN_TOTAL_SUM=%.17g\n", result->train_total);
    printf("TRAIN_RECONSTRUCTION_SUM=%.17g\n",
           result->train_reconstruction);
    printf("TRAIN_KL_SUM=%.17g\n", result->train_kl);
    printf("VALIDATION_TOTAL_SUM=%.17g\n", result->validation_total);
    printf("VALIDATION_RECONSTRUCTION_SUM=%.17g\n",
           result->validation_reconstruction);
    printf("VALIDATION_KL_SUM=%.17g\n", result->validation_kl);
    printf("MEMORY_LOWER_BOUND_BYTES=%" PRIu64 "\n",
           result->lower_bound_bytes);
    printf("MEMORY_WORKING_START_BYTES=%" PRIu64 "\n",
           result->memory.working_start);
    printf("MEMORY_WORKING_END_BYTES=%" PRIu64 "\n",
           result->memory.working_end);
    printf("MEMORY_WORKING_MEDIAN_BYTES=%" PRIu64 "\n",
           benchmark_u64_median(result->memory.working_samples,
                                result->memory.sample_count));
    printf("MEMORY_PEAK_WORKING_SET_BYTES=%" PRIu64 "\n",
           result->memory.working_peak);
    printf("MEMORY_PRIVATE_START_BYTES=%" PRIu64 "\n",
           result->memory.private_start);
    printf("MEMORY_PRIVATE_END_BYTES=%" PRIu64 "\n",
           result->memory.private_end);
    printf("MEMORY_PRIVATE_MEDIAN_BYTES=%" PRIu64 "\n",
           benchmark_u64_median(result->memory.private_samples,
                                result->memory.sample_count));
    printf("MEMORY_PEAK_PRIVATE_BYTES=%" PRIu64 "\n",
           result->memory.private_peak);
    printf("MEMORY_TRAIN_PRIVATE_FIRST_BYTES=%" PRIu64 "\n",
           result->memory.train_private_first);
    printf("MEMORY_TRAIN_PRIVATE_LAST_BYTES=%" PRIu64 "\n",
           result->memory.train_private_last);
    printf("PARAMETERS_CHANGED=%s\n",
           result->parameters_changed ? "true" : "false");
    printf("GRADIENTS_FINITE=true\n");
    printf("PARAMETERS_FINITE=true\n");
    printf("INPUT_UNCHANGED=true\n");
    printf("VALIDATION_REPEAT_BITWISE=%s\n",
           result->validation_repeat_bitwise ? "true" : "false");
    printf("VALIDATION_PERSISTENT_STATE_UNCHANGED=%s\n",
           result->validation_persistent_unchanged ? "true" : "false");
    printf("NUMERIC_GATE_PASS=true\n");
    printf("NO_RESULTS_WRITTEN=true\n");
    printf("NO_OFFICIAL_TRAINING_RUN=true\n");
    printf("BENCHMARK_PASS=true\n");
    status = 0;

cleanup:
    benchmark_candidate_free(&candidate);
    return status;
}

static void benchmark_usage(const char *program)
{
    printf("Usage: %s --batch <4|8|16|32> <repository-root>\n", program);
    printf("Runs one manual mini224_bridge_v1 performance candidate.\n");
}

int main(int argc, char **argv)
{
    BenchmarkContract contract;
    BenchmarkResult result;
    char *end = NULL;
    unsigned long parsed;
    size_t batch_size;
    int status = EXIT_FAILURE;

    if (argc == 2 && strcmp(argv[1], "--help") == 0) {
        benchmark_usage(argv[0]);
        return EXIT_SUCCESS;
    }
    if (argc != 4 || strcmp(argv[1], "--batch") != 0) {
        benchmark_usage(argv[0]);
        return EXIT_FAILURE;
    }
    parsed = strtoul(argv[2], &end, 10);
    if (end == argv[2] || *end != '\0' || parsed > (unsigned long)SIZE_MAX) {
        fprintf(stderr, "BENCHMARK_ERROR=invalid_batch\n");
        return EXIT_FAILURE;
    }
    batch_size = (size_t)parsed;
    if (!benchmark_batch_is_allowed(batch_size)) {
        fprintf(stderr, "BENCHMARK_ERROR=unsupported_batch\n");
        return EXIT_FAILURE;
    }

    benchmark_contract_init(&contract);
    if (benchmark_load_contract(&contract, argv[3]) == 0 &&
        benchmark_run(&contract, batch_size, &result) == 0) {
        status = EXIT_SUCCESS;
    }
    benchmark_contract_free(&contract);
    return status;
}
