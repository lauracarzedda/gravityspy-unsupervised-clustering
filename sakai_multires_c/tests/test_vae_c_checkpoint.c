#include "vae_c_checkpoint.h"

#include "vae_c_config.h"
#include "vae_c_model_backward.h"
#include "vae_c_model_forward.h"
#include "vae_c_model_train.h"
#include "vae_c_model_workspace.h"
#include "vae_c_rng.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


enum {
    TEST_PATH_CAP = 1024,
    TEST_HEADER_SIZE = 72,
    TEST_DIGEST_SIZE = 32,
    TEST_MAX_BATCH_NORMS =
        VAE_C_MAX_CONV_DEPTH + 1 + VAE_C_MAX_DECODER_HIDDEN_CONVS
};


/* Test-only symbol emitted by gravityspy_vae_core when BUILD_TESTING is on. */
int vae_c_checkpoint_sha256_test_only(
    const unsigned char *data,
    size_t size,
    unsigned char digest[TEST_DIGEST_SIZE]
);


typedef struct {
    VAECModel model;
    VAECModelWorkspace workspace;
    VAECAdam optimizer;
} TestFixture;


typedef struct {
    unsigned char *bytes;
    size_t size;
} TestBytes;


typedef struct {
    TestBytes bytes;
} TestStateSnapshot;


static int report_check(const char *name, int passed)
{
    if (passed) {
        printf("[PASS] %s\n", name);
        return 0;
    }
    printf("[FAIL] %s\n", name);
    return 1;
}


static int test_sha256_standard_golden_vectors(void)
{
    static const unsigned char expected_empty[TEST_DIGEST_SIZE] = {
        0xe3u, 0xb0u, 0xc4u, 0x42u, 0x98u, 0xfcu, 0x1cu, 0x14u,
        0x9au, 0xfbu, 0xf4u, 0xc8u, 0x99u, 0x6fu, 0xb9u, 0x24u,
        0x27u, 0xaeu, 0x41u, 0xe4u, 0x64u, 0x9bu, 0x93u, 0x4cu,
        0xa4u, 0x95u, 0x99u, 0x1bu, 0x78u, 0x52u, 0xb8u, 0x55u
    };
    static const unsigned char expected_abc[TEST_DIGEST_SIZE] = {
        0xbau, 0x78u, 0x16u, 0xbfu, 0x8fu, 0x01u, 0xcfu, 0xeau,
        0x41u, 0x41u, 0x40u, 0xdeu, 0x5du, 0xaeu, 0x22u, 0x23u,
        0xb0u, 0x03u, 0x61u, 0xa3u, 0x96u, 0x17u, 0x7au, 0x9cu,
        0xb4u, 0x10u, 0xffu, 0x61u, 0xf2u, 0x00u, 0x15u, 0xadu
    };
    static const unsigned char abc[] = {'a', 'b', 'c'};
    unsigned char digest[TEST_DIGEST_SIZE];

    if (!vae_c_checkpoint_sha256_test_only(NULL, 0u, digest) ||
        memcmp(digest, expected_empty, sizeof(digest)) != 0 ||
        !vae_c_checkpoint_sha256_test_only(abc, sizeof(abc), digest) ||
        memcmp(digest, expected_abc, sizeof(digest)) != 0) {

        return 0;
    }
    return 1;
}


static void test_bytes_init(TestBytes *bytes)
{
    memset(bytes, 0, sizeof(*bytes));
}


static void test_bytes_free(TestBytes *bytes)
{
    free(bytes->bytes);
    test_bytes_init(bytes);
}


static int test_bytes_append(
    TestBytes *bytes,
    const void *source,
    size_t count
)
{
    unsigned char *candidate;

    if (bytes == NULL || (count != 0 && source == NULL) ||
        bytes->size > SIZE_MAX - count) {
        return 0;
    }
    candidate = (unsigned char *)realloc(
        bytes->bytes, bytes->size + count
    );
    if (candidate == NULL && count != 0) {
        return 0;
    }
    bytes->bytes = candidate;
    if (count != 0) {
        memcpy(bytes->bytes + bytes->size, source, count);
        bytes->size += count;
    }
    return 1;
}


static void fixture_init(TestFixture *fixture)
{
    vae_c_model_init(&fixture->model);
    vae_c_model_workspace_init(&fixture->workspace);
    vae_c_adam_init(&fixture->optimizer);
}


static void fixture_free(TestFixture *fixture)
{
    vae_c_model_workspace_free(&fixture->workspace);
    vae_c_model_free(&fixture->model);
    vae_c_adam_free(&fixture->optimizer);
}


static int fixture_allocate(
    TestFixture *fixture,
    const VAECProfile *profile,
    size_t batch_size,
    uint64_t initialization_seed,
    int allocate_workspace
)
{
    fixture_init(fixture);
    if (vae_c_model_allocate(
            &fixture->model, profile, batch_size) != VAE_C_OP_OK ||
        vae_c_model_initialize_parameters_v1(
            &fixture->model, initialization_seed) != VAE_C_OP_OK ||
        vae_c_adam_configure_v1(&fixture->optimizer) != VAE_C_OP_OK) {
        return 0;
    }
    if (allocate_workspace &&
        vae_c_model_workspace_allocate(
            &fixture->model, batch_size, &fixture->workspace) !=
                VAE_C_OP_OK) {
        return 0;
    }
    return 1;
}


static void fill_input(VAECModelWorkspace *workspace, unsigned salt)
{
    size_t index;

    for (index = 0; index < workspace->input.shape.count; index++) {
        unsigned long value = (unsigned long)(
            (index * 37u + (size_t)salt * 53u + 11u) % 997u
        );

        workspace->input.data[index] =
            0.05f + 0.9f * (float)value / 996.0f;
    }
}


static void seed_parameter_gradients(VAECModel *model, float scale)
{
    size_t parameter_index;

    for (parameter_index = 0;
         parameter_index < model->parameter_count;
         parameter_index++) {
        VAECParameter *parameter = model->parameters[parameter_index];
        size_t index;

        for (index = 0; index < parameter->shape.count; index++) {
            parameter->gradients[index] = scale *
                (float)(1u + (parameter_index + index) % 11u);
        }
    }
}


static size_t collect_batch_norms_const(
    const VAECModel *model,
    const VAECBatchNorm *layers[TEST_MAX_BATCH_NORMS]
)
{
    size_t count = 0;
    size_t index;

    for (index = 0; index < model->config.conv_depth; index++) {
        layers[count++] = &model->encoder_batch_norms[index];
    }
    layers[count++] = &model->decoder_fc_batch_norm;
    for (index = 0;
         index < model->config.decoder_hidden_conv_count;
         index++) {
        layers[count++] = &model->decoder_hidden_batch_norms[index];
    }
    return count;
}


static int persistent_state_equal(
    const TestFixture *left,
    const TestFixture *right
)
{
    const VAECBatchNorm *left_batch_norms[TEST_MAX_BATCH_NORMS];
    const VAECBatchNorm *right_batch_norms[TEST_MAX_BATCH_NORMS];
    size_t left_bn_count;
    size_t right_bn_count;
    size_t index;

    if (strcmp(left->model.profile_id, right->model.profile_id) != 0 ||
        memcmp(&left->model.config,
               &right->model.config,
               sizeof(left->model.config)) != 0 ||
        left->model.max_batch_size != right->model.max_batch_size ||
        left->model.initialization_seed !=
            right->model.initialization_seed ||
        left->model.parameter_count != right->model.parameter_count ||
        left->model.trainable_value_count !=
            right->model.trainable_value_count ||
        left->optimizer.learning_rate != right->optimizer.learning_rate ||
        left->optimizer.beta1 != right->optimizer.beta1 ||
        left->optimizer.beta2 != right->optimizer.beta2 ||
        left->optimizer.adam_epsilon != right->optimizer.adam_epsilon ||
        left->optimizer.step != right->optimizer.step ||
        left->optimizer.beta1_power != right->optimizer.beta1_power ||
        left->optimizer.beta2_power != right->optimizer.beta2_power ||
        left->optimizer.configured != right->optimizer.configured) {
        return 0;
    }
    for (index = 0; index < left->model.parameter_count; index++) {
        const VAECParameter *left_parameter =
            left->model.parameters[index];
        const VAECParameter *right_parameter =
            right->model.parameters[index];
        size_t bytes = left_parameter->shape.count * sizeof(float);

        if (left_parameter->parameter_key !=
                right_parameter->parameter_key ||
            memcmp(&left_parameter->shape,
                   &right_parameter->shape,
                   sizeof(left_parameter->shape)) != 0 ||
            memcmp(left_parameter->values,
                   right_parameter->values, bytes) != 0 ||
            memcmp(left_parameter->adam_m,
                   right_parameter->adam_m, bytes) != 0 ||
            memcmp(left_parameter->adam_v,
                   right_parameter->adam_v, bytes) != 0) {
            return 0;
        }
    }
    left_bn_count = collect_batch_norms_const(
        &left->model, left_batch_norms
    );
    right_bn_count = collect_batch_norms_const(
        &right->model, right_batch_norms
    );
    if (left_bn_count != right_bn_count) {
        return 0;
    }
    for (index = 0; index < left_bn_count; index++) {
        size_t bytes = left_batch_norms[index]->num_features *
            sizeof(float);

        if (left_batch_norms[index]->kind !=
                right_batch_norms[index]->kind ||
            left_batch_norms[index]->num_features !=
                right_batch_norms[index]->num_features ||
            memcmp(left_batch_norms[index]->running_mean,
                   right_batch_norms[index]->running_mean,
                   bytes) != 0 ||
            memcmp(left_batch_norms[index]->running_variance,
                   right_batch_norms[index]->running_variance,
                   bytes) != 0) {
            return 0;
        }
    }
    return 1;
}


static int gradient_state_equal(
    const VAECModel *model,
    const float *snapshot,
    size_t count
)
{
    size_t offset = 0;
    size_t parameter_index;

    for (parameter_index = 0;
         parameter_index < model->parameter_count;
         parameter_index++) {
        const VAECParameter *parameter = model->parameters[parameter_index];

        if (offset > count - parameter->shape.count ||
            memcmp(parameter->gradients,
                   snapshot + offset,
                   parameter->shape.count * sizeof(float)) != 0) {
            return 0;
        }
        offset += parameter->shape.count;
    }
    return offset == count;
}


static float *copy_parameter_gradients(
    const VAECModel *model,
    size_t *count
)
{
    size_t total = (size_t)model->trainable_value_count;
    float *snapshot;
    size_t offset = 0;
    size_t parameter_index;

    if ((uint64_t)total != model->trainable_value_count || count == NULL ||
        total > SIZE_MAX / sizeof(float)) {
        return NULL;
    }
    snapshot = (float *)malloc(total * sizeof(float));
    if (snapshot == NULL) {
        return NULL;
    }
    for (parameter_index = 0;
         parameter_index < model->parameter_count;
         parameter_index++) {
        const VAECParameter *parameter = model->parameters[parameter_index];

        memcpy(snapshot + offset,
               parameter->gradients,
               parameter->shape.count * sizeof(float));
        offset += parameter->shape.count;
    }
    *count = total;
    return snapshot;
}


static uint64_t fnv_bytes(uint64_t hash, const void *memory, size_t size)
{
    const unsigned char *bytes = (const unsigned char *)memory;
    size_t index;

    for (index = 0; index < size; index++) {
        hash ^= (uint64_t)bytes[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}


static uint64_t cache_state_hash(const VAECModel *model)
{
    const VAECBatchNorm *batch_norms[TEST_MAX_BATCH_NORMS];
    size_t batch_norm_count = collect_batch_norms_const(model, batch_norms);
    uint64_t hash = UINT64_C(1469598103934665603);
    size_t index;

    for (index = 0; index < batch_norm_count; index++) {
        const VAECBatchNorm *layer = batch_norms[index];
        size_t transient_values = layer->num_features * 5u;
        const float *transient_start = layer->batch_mean;

        hash = fnv_bytes(
            hash,
            transient_start,
            transient_values * sizeof(float)
        );
        hash = fnv_bytes(hash, &layer->last_sample_count,
                         sizeof(layer->last_sample_count));
        hash = fnv_bytes(hash, &layer->last_input_shape,
                         sizeof(layer->last_input_shape));
        hash = fnv_bytes(hash, &layer->last_output_shape,
                         sizeof(layer->last_output_shape));
        hash = fnv_bytes(hash, &layer->cache_valid,
                         sizeof(layer->cache_valid));
    }
    hash = fnv_bytes(
        hash,
        model->encoder_max_pool.argmax_indices,
        model->encoder_max_pool.owned_bytes
    );
    hash = fnv_bytes(hash, &model->encoder_max_pool.last_count,
                     sizeof(model->encoder_max_pool.last_count));
    hash = fnv_bytes(hash, &model->encoder_max_pool.last_input_shape,
                     sizeof(model->encoder_max_pool.last_input_shape));
    hash = fnv_bytes(hash, &model->encoder_max_pool.last_output_shape,
                     sizeof(model->encoder_max_pool.last_output_shape));
    hash = fnv_bytes(hash, &model->encoder_max_pool.cache_valid,
                     sizeof(model->encoder_max_pool.cache_valid));
    return hash;
}


static int state_snapshot_capture(
    const VAECModel *model,
    const VAECAdam *optimizer,
    TestStateSnapshot *snapshot
)
{
    const VAECBatchNorm *batch_norms[TEST_MAX_BATCH_NORMS];
    size_t batch_norm_count = collect_batch_norms_const(model, batch_norms);
    size_t index;

    test_bytes_init(&snapshot->bytes);
    if (!test_bytes_append(
            &snapshot->bytes, model, sizeof(*model)) ||
        !test_bytes_append(
            &snapshot->bytes, optimizer, sizeof(*optimizer))) {
        return 0;
    }
    for (index = 0; index < model->parameter_count; index++) {
        const VAECParameter *parameter = model->parameters[index];

        if (!test_bytes_append(
                &snapshot->bytes,
                parameter->owned_block,
                parameter->owned_bytes)) {
            return 0;
        }
    }
    for (index = 0; index < batch_norm_count; index++) {
        const VAECBatchNorm *layer = batch_norms[index];

        if (!test_bytes_append(
                &snapshot->bytes, layer->owned_block, layer->owned_bytes)) {
            return 0;
        }
    }
    return test_bytes_append(
        &snapshot->bytes,
        model->encoder_max_pool.argmax_indices,
        model->encoder_max_pool.owned_bytes
    );
}


static int state_snapshot_matches(
    const TestStateSnapshot *before,
    const VAECModel *model,
    const VAECAdam *optimizer
)
{
    TestStateSnapshot after;
    int equal;

    if (!state_snapshot_capture(model, optimizer, &after)) {
        return 0;
    }
    equal = before->bytes.size == after.bytes.size &&
        memcmp(before->bytes.bytes,
               after.bytes.bytes,
               before->bytes.size) == 0;
    test_bytes_free(&after.bytes);
    return equal;
}


static void state_snapshot_free(TestStateSnapshot *snapshot)
{
    test_bytes_free(&snapshot->bytes);
}


static int make_path(
    char path[TEST_PATH_CAP],
    const char *directory,
    const char *name
)
{
    int written = snprintf(path, TEST_PATH_CAP, "%s/%s", directory, name);
    return written > 0 && (size_t)written < TEST_PATH_CAP;
}


static int read_file(const char *path, TestBytes *bytes)
{
    FILE *file;
    long length;
    size_t size;
    int failed = 0;

    test_bytes_init(bytes);
    file = fopen(path, "rb");
    if (file == NULL || fseek(file, 0, SEEK_END) != 0 ||
        (length = ftell(file)) < 0 || fseek(file, 0, SEEK_SET) != 0) {
        if (file != NULL) {
            (void)fclose(file);
        }
        return 0;
    }
    size = (size_t)length;
    bytes->bytes = (unsigned char *)malloc(size);
    if (bytes->bytes == NULL) {
        (void)fclose(file);
        return 0;
    }
    if (fread(bytes->bytes, 1u, size, file) != size || ferror(file)) {
        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    if (failed) {
        test_bytes_free(bytes);
        return 0;
    }
    bytes->size = size;
    return 1;
}


static int write_file(
    const char *path,
    const unsigned char *bytes,
    size_t size
)
{
    FILE *file = fopen(path, "wb");
    int failed;

    if (file == NULL) {
        return 0;
    }
    failed = fwrite(bytes, 1u, size, file) != size ||
        fflush(file) != 0 || ferror(file);
    if (fclose(file) != 0) {
        failed = 1;
    }
    return !failed;
}


static uint64_t read_u64_le(const unsigned char *bytes)
{
    uint64_t value = UINT64_C(0);
    size_t index;

    for (index = 0; index < 8u; index++) {
        value |= (uint64_t)bytes[index] << (8u * index);
    }
    return value;
}


static void write_u32_le(unsigned char *bytes, uint32_t value)
{
    bytes[0] = (unsigned char)value;
    bytes[1] = (unsigned char)(value >> 8);
    bytes[2] = (unsigned char)(value >> 16);
    bytes[3] = (unsigned char)(value >> 24);
}


static void write_u64_le(unsigned char *bytes, uint64_t value)
{
    size_t index;

    for (index = 0; index < 8u; index++) {
        bytes[index] = (unsigned char)(value >> (8u * index));
    }
}


static void fill_hex(char *destination, size_t capacity, char digit)
{
    size_t length = capacity - 1u;

    memset(destination, digit, length);
    destination[length] = '\0';
}


static void fill_identity(
    VAECCheckpointRunIdentity *identity,
    const VAECModel *model,
    uint64_t initialization_seed
)
{
    vae_c_checkpoint_run_identity_init(identity);
    (void)snprintf(identity->profile_id, sizeof(identity->profile_id),
                   "%s", model->profile_id);
    fill_hex(identity->profile_config_sha256,
             sizeof(identity->profile_config_sha256), 'A');
    fill_hex(identity->code_commit, sizeof(identity->code_commit), 'b');
    (void)snprintf(identity->dataset_id, sizeof(identity->dataset_id),
                   "synthetic_checkpoint_dataset_v1");
    (void)snprintf(identity->event_manifest_id,
                   sizeof(identity->event_manifest_id),
                   "manifest_events_synthetic.csv");
    fill_hex(identity->event_manifest_sha256,
             sizeof(identity->event_manifest_sha256), 'C');
    (void)snprintf(identity->tensor_manifest_id,
                   sizeof(identity->tensor_manifest_id),
                   "manifest_tensors_synthetic.csv");
    fill_hex(identity->tensor_manifest_sha256,
             sizeof(identity->tensor_manifest_sha256), 'D');
    (void)snprintf(identity->train_split_id,
                   sizeof(identity->train_split_id),
                   "split_train_synthetic.csv");
    fill_hex(identity->train_split_sha256,
             sizeof(identity->train_split_sha256), 'E');
    (void)snprintf(identity->validation_split_id,
                   sizeof(identity->validation_split_id),
                   "split_validation_synthetic.csv");
    fill_hex(identity->validation_split_sha256,
             sizeof(identity->validation_split_sha256), 'F');
    (void)snprintf(identity->augmentation_policy,
                   sizeof(identity->augmentation_policy),
                   "augmentation_stateless_v1");
    (void)snprintf(identity->shuffle_policy,
                   sizeof(identity->shuffle_policy),
                   "fisher_yates_shuffle_v1");
    (void)snprintf(identity->latent_policy,
                   sizeof(identity->latent_policy),
                   "latent_train_per_event_epoch_v1");
    (void)snprintf(identity->validation_policy,
                   sizeof(identity->validation_policy),
                   "central_fixed_epsilon_v1");
    (void)snprintf(identity->compiler_id,
                   sizeof(identity->compiler_id), "GNU");
    (void)snprintf(identity->compiler_version,
                   sizeof(identity->compiler_version), "16.1.0");
    (void)snprintf(identity->compiler_target,
                   sizeof(identity->compiler_target),
                   "x86_64-w64-mingw32");
    identity->initialization_seed = initialization_seed;
    identity->latent_seed = UINT64_C(456);
    identity->augmentation_seed = UINT64_C(789);
    identity->shuffle_seed = UINT64_C(101112);
    identity->max_batch_size = (uint64_t)model->max_batch_size;
    identity->batch_size = (uint64_t)model->max_batch_size;
    identity->optimizer_learning_rate = VAE_C_ADAM_LEARNING_RATE_V1;
    identity->optimizer_beta1 = VAE_C_ADAM_BETA1_V1;
    identity->optimizer_beta2 = VAE_C_ADAM_BETA2_V1;
    identity->optimizer_epsilon = VAE_C_ADAM_EPSILON_V1;
}


static void fill_metadata(
    VAECCheckpointMetadata *metadata,
    const VAECModel *model,
    uint64_t initialization_seed,
    VAECCheckpointKind kind,
    uint64_t epoch
)
{
    vae_c_checkpoint_metadata_init(metadata);
    fill_identity(&metadata->run_identity, model, initialization_seed);
    metadata->kind = kind;
    metadata->last_completed_epoch = epoch;
    metadata->best_defined = true;
    metadata->best_epoch = epoch;
    metadata->best_validation_total_mean_per_sample = 123.25;
    metadata->no_improvement_count = UINT64_C(0);
}


static int metadata_equal(
    const VAECCheckpointMetadata *left,
    const VAECCheckpointMetadata *right
)
{
    return memcmp(left, right, sizeof(*left)) == 0;
}


static int run_train_epoch(
    TestFixture *fixture,
    uint64_t latent_seed,
    uint64_t epoch,
    const uint64_t manifest_indices[2],
    unsigned input_salt
)
{
    VAECVAELossStats stats;

    fill_input(&fixture->workspace, input_salt);
    vae_c_vae_loss_stats_init(&stats);
    return vae_c_model_train_step(
        &fixture->model,
        &fixture->workspace,
        &fixture->optimizer,
        latent_seed,
        epoch,
        manifest_indices,
        2u,
        &stats
    ) == VAE_C_OP_OK && isfinite(stats.mean_per_sample);
}


static int issue_backward_ticket(
    TestFixture *fixture,
    uint64_t latent_seed,
    uint64_t epoch,
    const uint64_t manifest_indices[2]
)
{
    VAECVAELossStats stats;
    size_t sample;
    size_t latent_dim = fixture->model.config.latent_dim;

    fill_input(&fixture->workspace, 91u);
    for (sample = 0; sample < 2u; sample++) {
        if (vae_c_rng_fill_latent_train_epsilon_f32(
                latent_seed,
                epoch,
                manifest_indices[sample],
                fixture->workspace.epsilon.data + sample * latent_dim,
                latent_dim) != VAE_C_RNG_OK) {
            return 0;
        }
    }
    vae_c_vae_loss_stats_init(&stats);
    return vae_c_model_vae_forward(
        &fixture->model,
        VAE_C_BATCH_NORM_MODE_TRAIN,
        VAE_C_LATENT_MODE_STOCHASTIC,
        &fixture->workspace,
        &stats
    ) == VAE_C_OP_OK &&
        fixture->workspace.vae_backward_model == &fixture->model &&
        fixture->workspace.vae_backward_generation ==
            fixture->model.vae_cache_generation;
}


static void make_synthetic_depth4(VAECProfile *profile)
{
    memset(profile, 0, sizeof(*profile));
    memcpy(profile->profile_id,
           "synthetic_depth4_checkpoint",
           sizeof("synthetic_depth4_checkpoint"));
    memcpy(profile->role, "synthetic", sizeof("synthetic"));
    memcpy(profile->profile_status, "test_only", sizeof("test_only"));
    profile->input_side = 32;
    profile->conv_depth = 4;
    profile->channels[0] = 2;
    profile->channels[1] = 3;
    profile->channels[2] = 4;
    profile->channels[3] = 5;
    profile->latent_dim = 3;
    profile->bottleneck_side = 1;
    profile->expected_trainable_parameters_bn1d = UINT64_C(939);
    profile->expected_forward_mac_per_sample = UINT64_C(117297);
}


static int test_round_trip_and_lifecycle(
    const VAECProfile *profile,
    const char *fixture_directory
)
{
    const uint64_t manifest_indices[2] = {UINT64_C(17), UINT64_C(31)};
    TestFixture source;
    TestFixture destination;
    VAECCheckpointMetadata metadata;
    VAECCheckpointMetadata loaded;
    float *destination_gradients = NULL;
    size_t destination_gradient_count = 0;
    uint64_t destination_cache_hash;
    uint64_t ticket_generation;
    uint64_t generation_before_load;
    char path[TEST_PATH_CAP];
    int passed = 0;

    fixture_init(&source);
    fixture_init(&destination);
    vae_c_checkpoint_metadata_init(&loaded);
    if (!make_path(path, fixture_directory, "round_trip.ckpt") ||
        !fixture_allocate(&source, profile, 2u, UINT64_C(123), 1) ||
        !fixture_allocate(&destination, profile, 2u, UINT64_C(123), 1) ||
        !run_train_epoch(
            &source, UINT64_C(456), UINT64_C(0), manifest_indices, 3u) ||
        !issue_backward_ticket(
            &destination, UINT64_C(456), UINT64_C(9), manifest_indices)) {
        goto cleanup;
    }
    seed_parameter_gradients(&destination.model, 0.003f);
    destination_gradients = copy_parameter_gradients(
        &destination.model, &destination_gradient_count
    );
    if (destination_gradients == NULL) {
        goto cleanup;
    }
    destination_cache_hash = cache_state_hash(&destination.model);
    ticket_generation = destination.workspace.vae_backward_generation;
    generation_before_load = destination.model.vae_cache_generation;
    fill_metadata(
        &metadata,
        &source.model,
        UINT64_C(123),
        VAE_C_CHECKPOINT_KIND_LATEST,
        UINT64_C(0)
    );
    if (vae_c_checkpoint_save_atomic(
            path, &source.model, &source.optimizer, &metadata) !=
                VAE_C_CHECKPOINT_OK ||
        vae_c_checkpoint_load(
            path,
            &metadata.run_identity,
            &destination.model,
            &destination.optimizer,
            &loaded) != VAE_C_CHECKPOINT_OK ||
        !persistent_state_equal(&source, &destination) ||
        !metadata_equal(&metadata, &loaded) ||
        !gradient_state_equal(
            &destination.model,
            destination_gradients,
            destination_gradient_count) ||
        cache_state_hash(&destination.model) != destination_cache_hash ||
        destination.workspace.vae_backward_generation != ticket_generation ||
        destination.model.vae_cache_generation !=
            generation_before_load + UINT64_C(1) ||
        destination.model.vae_cache_generation == ticket_generation ||
        vae_c_model_vae_backward(
            &destination.model,
            &destination.workspace,
            1.0f) != VAE_C_OP_INVALID_STATE ||
        vae_c_model_validate(&destination.model) != VAE_C_OP_OK ||
        !issue_backward_ticket(
            &destination, UINT64_C(456), UINT64_C(10), manifest_indices) ||
        destination.model.vae_cache_generation !=
            generation_before_load + UINT64_C(3)) {
        goto cleanup;
    }
    passed = 1;

cleanup:
    (void)remove(path);
    free(destination_gradients);
    fixture_free(&source);
    fixture_free(&destination);
    return passed;
}


static int test_provenance_not_hard_gate_and_best_undefined(
    const VAECProfile *profile,
    const char *fixture_directory
)
{
    TestFixture source;
    TestFixture destination;
    VAECCheckpointMetadata metadata;
    VAECCheckpointMetadata loaded;
    VAECCheckpointRunIdentity expected;
    char path[TEST_PATH_CAP];
    int passed = 0;

    fixture_init(&source);
    fixture_init(&destination);
    if (!make_path(path, fixture_directory, "provenance.ckpt") ||
        !fixture_allocate(&source, profile, 2u, UINT64_C(123), 0) ||
        !fixture_allocate(&destination, profile, 2u, UINT64_C(123), 0)) {
        goto cleanup;
    }
    fill_metadata(
        &metadata,
        &source.model,
        UINT64_C(123),
        VAE_C_CHECKPOINT_KIND_FINAL,
        UINT64_C(4)
    );
    metadata.best_defined = false;
    metadata.best_epoch = UINT64_C(0);
    metadata.best_validation_total_mean_per_sample = 0.0;
    metadata.no_improvement_count = UINT64_C(0);
    expected = metadata.run_identity;
    (void)snprintf(expected.compiler_id, sizeof(expected.compiler_id),
                   "different-compiler-provenance");
    (void)snprintf(expected.compiler_version,
                   sizeof(expected.compiler_version), "99.0");
    (void)snprintf(expected.compiler_target,
                   sizeof(expected.compiler_target), "different-target");
    if (vae_c_checkpoint_save_atomic(
            path, &source.model, &source.optimizer, &metadata) !=
                VAE_C_CHECKPOINT_OK ||
        vae_c_checkpoint_load(
            path,
            &expected,
            &destination.model,
            &destination.optimizer,
            &loaded) != VAE_C_CHECKPOINT_OK ||
        !metadata_equal(&metadata, &loaded) ||
        !persistent_state_equal(&source, &destination)) {
        goto cleanup;
    }
    passed = 1;

cleanup:
    (void)remove(path);
    fixture_free(&source);
    fixture_free(&destination);
    return passed;
}


static int test_byte_identical_saves(
    const VAECProfile *profile,
    const char *fixture_directory
)
{
    TestFixture source;
    VAECCheckpointMetadata metadata;
    TestBytes first;
    TestBytes second;
    char first_path[TEST_PATH_CAP] = {0};
    char second_path[TEST_PATH_CAP] = {0};
    int passed = 0;

    fixture_init(&source);
    test_bytes_init(&first);
    test_bytes_init(&second);
    if (!make_path(first_path, fixture_directory, "deterministic_a.ckpt") ||
        !make_path(second_path, fixture_directory, "deterministic_b.ckpt") ||
        !fixture_allocate(&source, profile, 2u, UINT64_C(123), 0)) {
        goto cleanup;
    }
    fill_metadata(
        &metadata, &source.model, UINT64_C(123),
        VAE_C_CHECKPOINT_KIND_LATEST, UINT64_C(7)
    );
    source.model.parameters[0]->values[0] += 0.03125f;
    source.model.parameters[0]->adam_m[0] = -0.125f;
    source.model.parameters[0]->adam_v[0] = 0.0625f;
    if (vae_c_checkpoint_save_atomic(
            first_path, &source.model, &source.optimizer, &metadata) !=
                VAE_C_CHECKPOINT_OK ||
        vae_c_checkpoint_save_atomic(
            second_path, &source.model, &source.optimizer, &metadata) !=
                VAE_C_CHECKPOINT_OK ||
        !read_file(first_path, &first) ||
        !read_file(second_path, &second) ||
        first.size != second.size ||
        first.size < TEST_DIGEST_SIZE ||
        memcmp(first.bytes, second.bytes, first.size) != 0 ||
        memcmp(first.bytes + first.size - TEST_DIGEST_SIZE,
               second.bytes + second.size - TEST_DIGEST_SIZE,
               TEST_DIGEST_SIZE) != 0) {

        goto cleanup;
    }
    passed = 1;

cleanup:
    (void)remove(first_path);
    (void)remove(second_path);
    test_bytes_free(&first);
    test_bytes_free(&second);
    fixture_free(&source);
    return passed;
}


static int load_failure_is_atomic(
    const char *path,
    const VAECCheckpointRunIdentity *identity,
    TestFixture *destination
)
{
    TestStateSnapshot before;
    VAECCheckpointMetadata output;
    VAECCheckpointMetadata sentinel;
    VAECCheckpointStatus status;
    int passed;

    memset(&sentinel, 0xA5, sizeof(sentinel));
    output = sentinel;
    if (!state_snapshot_capture(
            &destination->model, &destination->optimizer, &before)) {
        return 0;
    }
    status = vae_c_checkpoint_load(
        path,
        identity,
        &destination->model,
        &destination->optimizer,
        &output
    );
    passed = status != VAE_C_CHECKPOINT_OK &&
        state_snapshot_matches(
            &before, &destination->model, &destination->optimizer) &&
        memcmp(&output, &sentinel, sizeof(output)) == 0;
    state_snapshot_free(&before);
    return passed;
}


static int test_lifecycle_generation_boundaries(
    const VAECProfile *profile,
    const char *fixture_directory
)
{
    const uint64_t manifest_indices[2] = {UINT64_C(17), UINT64_C(31)};
    const uint64_t rejected_generations[3] = {
        UINT64_MAX - UINT64_C(2),
        UINT64_MAX - UINT64_C(1),
        UINT64_MAX
    };
    TestFixture source;
    TestFixture destination;
    VAECCheckpointMetadata metadata;
    VAECCheckpointMetadata loaded;
    char path[TEST_PATH_CAP] = {0};
    size_t index;
    int passed = 0;

    fixture_init(&source);
    fixture_init(&destination);
    vae_c_checkpoint_metadata_init(&loaded);
    if (!make_path(path, fixture_directory, "generation_boundary.ckpt") ||
        !fixture_allocate(&source, profile, 2u, UINT64_C(123), 0) ||
        !fixture_allocate(&destination, profile, 2u, UINT64_C(123), 1)) {
        goto cleanup;
    }
    fill_metadata(
        &metadata, &source.model, UINT64_C(123),
        VAE_C_CHECKPOINT_KIND_LATEST, UINT64_C(4)
    );
    if (vae_c_checkpoint_save_atomic(
            path, &source.model, &source.optimizer, &metadata) !=
                VAE_C_CHECKPOINT_OK) {
        goto cleanup;
    }

    destination.model.vae_cache_generation = UINT64_MAX - UINT64_C(3);
    destination.workspace.vae_backward_model = &destination.model;
    destination.workspace.vae_backward_generation = UINT64_C(19);
    if (vae_c_checkpoint_load(
            path,
            &metadata.run_identity,
            &destination.model,
            &destination.optimizer,
            &loaded) != VAE_C_CHECKPOINT_OK ||
        destination.model.vae_cache_generation !=
            UINT64_MAX - UINT64_C(2) ||
        destination.workspace.vae_backward_model != &destination.model ||
        destination.workspace.vae_backward_generation != UINT64_C(19) ||
        destination.workspace.vae_backward_generation ==
            destination.model.vae_cache_generation ||
        !issue_backward_ticket(
            &destination, UINT64_C(456), UINT64_C(5), manifest_indices) ||
        destination.model.vae_cache_generation != UINT64_MAX ||
        destination.workspace.vae_backward_generation != UINT64_MAX) {

        goto cleanup;
    }

    destination.workspace.vae_backward_model = NULL;
    destination.workspace.vae_backward_generation = UINT64_C(0);
    for (index = 0; index < 3u; index++) {
        destination.model.vae_cache_generation = rejected_generations[index];
        if (!load_failure_is_atomic(
                path, &metadata.run_identity, &destination)) {
            goto cleanup;
        }
    }
    passed = 1;

cleanup:
    (void)remove(path);
    fixture_free(&source);
    fixture_free(&destination);
    return passed;
}


static int write_mutated_and_expect_failure(
    const char *path,
    const unsigned char *source,
    size_t source_size,
    size_t offset,
    const unsigned char *replacement,
    size_t replacement_size,
    size_t output_size,
    const VAECCheckpointRunIdentity *identity,
    TestFixture *destination
)
{
    unsigned char *candidate;
    int passed;

    if (output_size == 0 || source_size == 0 ||
        output_size < replacement_size ||
        offset > output_size - replacement_size) {
        return 0;
    }
    candidate = (unsigned char *)malloc(output_size);
    if (candidate == NULL) {
        return 0;
    }
    memset(candidate, 0, output_size);
    memcpy(candidate, source,
           source_size < output_size ? source_size : output_size);
    memcpy(candidate + offset, replacement, replacement_size);
    passed = write_file(path, candidate, output_size) &&
        load_failure_is_atomic(path, identity, destination);
    free(candidate);
    (void)remove(path);
    return passed;
}


static int write_mutated_with_valid_digest_and_expect_failure(
    const char *path,
    const unsigned char *source,
    size_t source_size,
    size_t offset,
    const unsigned char *replacement,
    size_t replacement_size,
    const VAECCheckpointRunIdentity *identity,
    TestFixture *destination
)
{
    unsigned char *candidate;
    int passed;

    if (source == NULL || source_size < TEST_DIGEST_SIZE ||
        replacement == NULL ||
        offset > source_size - TEST_DIGEST_SIZE ||
        replacement_size >
            source_size - TEST_DIGEST_SIZE - offset) {

        return 0;
    }
    candidate = (unsigned char *)malloc(source_size);
    if (candidate == NULL) {
        return 0;
    }
    memcpy(candidate, source, source_size);
    memcpy(candidate + offset, replacement, replacement_size);
    passed = vae_c_checkpoint_sha256_test_only(
            candidate,
            source_size - TEST_DIGEST_SIZE,
            candidate + source_size - TEST_DIGEST_SIZE) &&
        write_file(path, candidate, source_size) &&
        load_failure_is_atomic(path, identity, destination);
    free(candidate);
    (void)remove(path);
    return passed;
}


static int test_negative_files_and_transaction(
    const VAECProfile *profile,
    const char *fixture_directory
)
{
    TestFixture source;
    TestFixture destination;
    VAECCheckpointMetadata metadata;
    VAECCheckpointRunIdentity mismatched_identity;
    TestBytes valid;
    char valid_path[TEST_PATH_CAP];
    char corrupt_path[TEST_PATH_CAP];
    uint64_t identity_length;
    uint64_t model_length;
    uint64_t parameter_length;
    uint64_t batch_norm_length;
    uint64_t adam_length;
    size_t model_start;
    size_t parameter_start;
    size_t batch_norm_start;
    size_t adam_start;
    size_t controller_start;
    uint64_t first_parameter_count;
    uint64_t first_batch_norm_features;
    unsigned char replacement[8];
    int passed = 0;

    fixture_init(&source);
    fixture_init(&destination);
    test_bytes_init(&valid);
    if (!make_path(valid_path, fixture_directory, "negative_source.ckpt") ||
        !make_path(corrupt_path, fixture_directory, "negative_case.ckpt") ||
        !fixture_allocate(&source, profile, 2u, UINT64_C(123), 1) ||
        !fixture_allocate(&destination, profile, 2u, UINT64_C(123), 1)) {
        goto cleanup;
    }
    fill_metadata(
        &metadata, &source.model, UINT64_C(123),
        VAE_C_CHECKPOINT_KIND_BEST, UINT64_C(3)
    );
    if (vae_c_checkpoint_save_atomic(
            valid_path, &source.model, &source.optimizer, &metadata) !=
                VAE_C_CHECKPOINT_OK ||
        !read_file(valid_path, &valid) ||
        valid.size <= TEST_HEADER_SIZE + TEST_DIGEST_SIZE) {
        goto cleanup;
    }
    identity_length = read_u64_le(valid.bytes + 24u);
    model_length = read_u64_le(valid.bytes + 32u);
    parameter_length = read_u64_le(valid.bytes + 40u);
    batch_norm_length = read_u64_le(valid.bytes + 48u);
    adam_length = read_u64_le(valid.bytes + 56u);
    if (identity_length > SIZE_MAX || model_length > SIZE_MAX ||
        parameter_length > SIZE_MAX || batch_norm_length > SIZE_MAX ||
        adam_length > SIZE_MAX) {
        goto cleanup;
    }
    model_start = TEST_HEADER_SIZE + (size_t)identity_length;
    parameter_start = model_start + (size_t)model_length;
    batch_norm_start = parameter_start + (size_t)parameter_length;
    adam_start = batch_norm_start + (size_t)batch_norm_length;
    controller_start = adam_start + (size_t)adam_length;
    if (adam_start >= valid.size - TEST_DIGEST_SIZE ||
        controller_start > valid.size - TEST_DIGEST_SIZE - 16u ||
        parameter_start > valid.size - 64u ||
        batch_norm_start > valid.size - 32u) {
        goto cleanup;
    }
    first_parameter_count = read_u64_le(valid.bytes + parameter_start + 56u);
    first_batch_norm_features = read_u64_le(
        valid.bytes + batch_norm_start + 24u
    );
    if (first_parameter_count == 0 || first_batch_norm_features == 0 ||
        first_parameter_count > SIZE_MAX ||
        first_batch_norm_features > SIZE_MAX) {
        goto cleanup;
    }

    replacement[0] = (unsigned char)'X';
    if (!write_mutated_and_expect_failure(
            corrupt_path, valid.bytes, valid.size, 0u,
            replacement, 1u, valid.size,
            &metadata.run_identity, &destination)) {
        goto cleanup;
    }
    write_u32_le(replacement, UINT32_C(2));
    if (!write_mutated_and_expect_failure(
            corrupt_path, valid.bytes, valid.size, 8u,
            replacement, 4u, valid.size,
            &metadata.run_identity, &destination)) {
        goto cleanup;
    }
    replacement[0] = (unsigned char)(
        valid.bytes[valid.size - 1u] ^ 0x80u
    );
    if (!write_mutated_and_expect_failure(
            corrupt_path, valid.bytes, valid.size, valid.size - 1u,
            replacement, 1u, valid.size,
            &metadata.run_identity, &destination) ||
        !write_file(corrupt_path, valid.bytes, valid.size - 1u) ||
        !load_failure_is_atomic(
            corrupt_path, &metadata.run_identity, &destination)) {
        goto cleanup;
    }
    (void)remove(corrupt_path);
    replacement[0] = 0u;
    if (!write_mutated_and_expect_failure(
            corrupt_path, valid.bytes, valid.size, valid.size,
            replacement, 1u, valid.size + 1u,
            &metadata.run_identity, &destination)) {
        goto cleanup;
    }

    write_u64_le(replacement, UINT64_MAX);
    if (!write_mutated_and_expect_failure(
            corrupt_path, valid.bytes, valid.size, 24u,
            replacement, 8u, valid.size,
            &metadata.run_identity, &destination)) {
        goto cleanup;
    }
    write_u64_le(replacement, (uint64_t)source.model.parameter_count + 1u);
    if (!write_mutated_and_expect_failure(
            corrupt_path, valid.bytes, valid.size, parameter_start,
            replacement, 8u, valid.size,
            &metadata.run_identity, &destination)) {
        goto cleanup;
    }
    write_u64_le(replacement,
                 source.model.parameters[0]->parameter_key + UINT64_C(100));
    if (!write_mutated_and_expect_failure(
            corrupt_path, valid.bytes, valid.size, parameter_start + 8u,
            replacement, 8u, valid.size,
            &metadata.run_identity, &destination)) {
        goto cleanup;
    }
    write_u64_le(
        replacement,
        (uint64_t)source.model.parameters[0]->shape.dims[0] + UINT64_C(1)
    );
    if (!write_mutated_and_expect_failure(
            corrupt_path, valid.bytes, valid.size, parameter_start + 24u,
            replacement, 8u, valid.size,
            &metadata.run_identity, &destination)) {
        goto cleanup;
    }
    write_u64_le(replacement, UINT64_C(99));
    if (!write_mutated_and_expect_failure(
            corrupt_path, valid.bytes, valid.size, batch_norm_start + 8u,
            replacement, 8u, valid.size,
            &metadata.run_identity, &destination)) {
        goto cleanup;
    }
    write_u64_le(replacement, first_batch_norm_features + UINT64_C(1));
    if (!write_mutated_and_expect_failure(
            corrupt_path, valid.bytes, valid.size, batch_norm_start + 24u,
            replacement, 8u, valid.size,
            &metadata.run_identity, &destination)) {
        goto cleanup;
    }
    write_u64_le(replacement, (uint64_t)source.model.max_batch_size + 1u);
    if (!write_mutated_and_expect_failure(
            corrupt_path, valid.bytes, valid.size, model_start + 8u,
            replacement, 8u, valid.size,
            &metadata.run_identity, &destination)) {
        goto cleanup;
    }
    write_u64_le(replacement, (uint64_t)source.model.config.input_side + 1u);
    if (!write_mutated_and_expect_failure(
            corrupt_path, valid.bytes, valid.size, model_start + 48u,
            replacement, 8u, valid.size,
            &metadata.run_identity, &destination)) {
        goto cleanup;
    }

    write_u32_le(replacement, UINT32_C(0x7fc00000));
    if (!write_mutated_and_expect_failure(
            corrupt_path, valid.bytes, valid.size, parameter_start + 64u,
            replacement, 4u, valid.size,
            &metadata.run_identity, &destination)) {
        goto cleanup;
    }
    write_u32_le(replacement, UINT32_C(0x7f800000));
    if (!write_mutated_and_expect_failure(
            corrupt_path, valid.bytes, valid.size,
            parameter_start + 64u + (size_t)first_parameter_count * 4u,
            replacement, 4u, valid.size,
            &metadata.run_identity, &destination)) {
        goto cleanup;
    }
    write_u32_le(replacement, UINT32_C(0xbf800000));
    if (!write_mutated_and_expect_failure(
            corrupt_path, valid.bytes, valid.size,
            parameter_start + 64u +
                (size_t)first_parameter_count * 8u,
            replacement, 4u, valid.size,
            &metadata.run_identity, &destination) ||
        !write_mutated_and_expect_failure(
            corrupt_path, valid.bytes, valid.size,
            batch_norm_start + 32u +
                (size_t)first_batch_norm_features * 4u,
            replacement, 4u, valid.size,
            &metadata.run_identity, &destination)) {
        goto cleanup;
    }
    write_u64_le(replacement, UINT64_C(0));
    if (!write_mutated_and_expect_failure(
            corrupt_path, valid.bytes, valid.size, adam_start + 8u,
            replacement, 8u, valid.size,
            &metadata.run_identity, &destination)) {
        goto cleanup;
    }

    write_u64_le(replacement, UINT64_MAX);
    if (!write_mutated_with_valid_digest_and_expect_failure(
            corrupt_path, valid.bytes, valid.size,
            controller_start + 8u,
            replacement, 8u,
            &metadata.run_identity, &destination)) {
        goto cleanup;
    }

    {
        VAECCheckpointMetadata overflow_metadata = metadata;
        TestBytes preserved;

        test_bytes_init(&preserved);
        overflow_metadata.last_completed_epoch = UINT64_MAX;
        if (vae_c_checkpoint_save_atomic(
                corrupt_path,
                &source.model,
                &source.optimizer,
                &overflow_metadata) != VAE_C_CHECKPOINT_INVALID_ARGUMENT ||
            read_file(corrupt_path, &preserved)) {
            test_bytes_free(&preserved);
            goto cleanup;
        }
        test_bytes_free(&preserved);
    }

    mismatched_identity = metadata.run_identity;
    (void)snprintf(mismatched_identity.dataset_id,
                   sizeof(mismatched_identity.dataset_id),
                   "incompatible-dataset");
    if (!load_failure_is_atomic(
            valid_path, &mismatched_identity, &destination)) {
        goto cleanup;
    }
    mismatched_identity = metadata.run_identity;
    (void)snprintf(mismatched_identity.profile_id,
                   sizeof(mismatched_identity.profile_id),
                   "incompatible-profile");
    if (!load_failure_is_atomic(
            valid_path, &mismatched_identity, &destination)) {
        goto cleanup;
    }
    passed = 1;

cleanup:
    (void)remove(corrupt_path);
    (void)remove(valid_path);
    test_bytes_free(&valid);
    fixture_free(&source);
    fixture_free(&destination);
    return passed;
}


static int test_atomic_save_contract(
    const VAECProfile *profile,
    const char *fixture_directory
)
{
    TestFixture source;
    VAECCheckpointMetadata metadata;
    VAECCheckpointMetadata invalid_metadata;
    TestBytes before;
    TestBytes after;
    TestBytes temporary;
    static const unsigned char temporary_sentinel[] = {
        0x12u, 0x34u, 0x56u, 0x78u
    };
    char path[TEST_PATH_CAP];
    char temporary_path[TEST_PATH_CAP];
    int passed = 0;

    fixture_init(&source);
    test_bytes_init(&before);
    test_bytes_init(&after);
    test_bytes_init(&temporary);
    if (!make_path(path, fixture_directory, "atomic.ckpt") ||
        snprintf(temporary_path,
                 sizeof(temporary_path), "%s.tmp", path) <= 0 ||
        !fixture_allocate(&source, profile, 2u, UINT64_C(123), 0)) {
        goto cleanup;
    }
    fill_metadata(
        &metadata, &source.model, UINT64_C(123),
        VAE_C_CHECKPOINT_KIND_LATEST, UINT64_C(0)
    );
    if (vae_c_checkpoint_save_atomic(
            path, &source.model, &source.optimizer, &metadata) !=
                VAE_C_CHECKPOINT_OK ||
        !read_file(path, &before) ||
        !write_file(
            temporary_path,
            temporary_sentinel,
            sizeof(temporary_sentinel)) ||
        vae_c_checkpoint_save_atomic(
            path, &source.model, &source.optimizer, &metadata) !=
                VAE_C_CHECKPOINT_TEMP_EXISTS ||
        !read_file(path, &after) ||
        before.size != after.size ||
        memcmp(before.bytes, after.bytes, before.size) != 0 ||
        !read_file(temporary_path, &temporary) ||
        temporary.size != sizeof(temporary_sentinel) ||
        memcmp(temporary.bytes,
               temporary_sentinel,
               sizeof(temporary_sentinel)) != 0) {
        goto cleanup;
    }
    (void)remove(temporary_path);
    test_bytes_free(&after);
    test_bytes_free(&temporary);
    invalid_metadata = metadata;
    invalid_metadata.run_identity.code_commit[0] = 'Z';
    if (vae_c_checkpoint_save_atomic(
            path,
            &source.model,
            &source.optimizer,
            &invalid_metadata) == VAE_C_CHECKPOINT_OK ||
        !read_file(path, &after) ||
        before.size != after.size ||
        memcmp(before.bytes, after.bytes, before.size) != 0 ||
        read_file(temporary_path, &temporary)) {
        goto cleanup;
    }

    metadata.kind = VAE_C_CHECKPOINT_KIND_FINAL;
    metadata.last_completed_epoch = UINT64_C(1);
    metadata.best_epoch = UINT64_C(1);
    source.model.parameters[0]->values[0] += 0.125f;
    if (vae_c_checkpoint_save_atomic(
            path, &source.model, &source.optimizer, &metadata) !=
                VAE_C_CHECKPOINT_OK ||
        !read_file(path, &after) ||
        (after.size == before.size &&
         memcmp(after.bytes, before.bytes, before.size) == 0)) {
        goto cleanup;
    }
    passed = 1;

cleanup:
    (void)remove(temporary_path);
    (void)remove(path);
    test_bytes_free(&before);
    test_bytes_free(&after);
    test_bytes_free(&temporary);
    fixture_free(&source);
    return passed;
}


static int test_deterministic_resume(const char *fixture_directory)
{
    const uint64_t manifest_indices[2] = {UINT64_C(101), UINT64_C(303)};
    VAECProfile profile;
    TestFixture uninterrupted;
    TestFixture checkpoint_source;
    TestFixture resumed;
    VAECCheckpointMetadata metadata;
    VAECCheckpointMetadata loaded;
    char path[TEST_PATH_CAP];
    int passed = 0;

    make_synthetic_depth4(&profile);
    fixture_init(&uninterrupted);
    fixture_init(&checkpoint_source);
    fixture_init(&resumed);
    if (!make_path(path, fixture_directory, "resume.ckpt") ||
        !fixture_allocate(
            &uninterrupted, &profile, 2u, UINT64_C(123), 1) ||
        !fixture_allocate(
            &checkpoint_source, &profile, 2u, UINT64_C(123), 1) ||
        !fixture_allocate(&resumed, &profile, 2u, UINT64_C(123), 0) ||
        !run_train_epoch(
            &uninterrupted, UINT64_C(456), UINT64_C(0),
            manifest_indices, 7u) ||
        !run_train_epoch(
            &checkpoint_source, UINT64_C(456), UINT64_C(0),
            manifest_indices, 7u) ||
        !persistent_state_equal(&uninterrupted, &checkpoint_source)) {
        goto cleanup;
    }
    fill_metadata(
        &metadata,
        &checkpoint_source.model,
        UINT64_C(123),
        VAE_C_CHECKPOINT_KIND_LATEST,
        UINT64_C(0)
    );
    if (vae_c_checkpoint_save_atomic(
            path,
            &checkpoint_source.model,
            &checkpoint_source.optimizer,
            &metadata) != VAE_C_CHECKPOINT_OK ||
        vae_c_checkpoint_load(
            path,
            &metadata.run_identity,
            &resumed.model,
            &resumed.optimizer,
            &loaded) != VAE_C_CHECKPOINT_OK ||
        !metadata_equal(&metadata, &loaded) ||
        vae_c_model_workspace_allocate(
            &resumed.model, 2u, &resumed.workspace) != VAE_C_OP_OK ||
        !run_train_epoch(
            &uninterrupted, UINT64_C(456), UINT64_C(1),
            manifest_indices, 13u) ||
        !run_train_epoch(
            &resumed, UINT64_C(456), UINT64_C(1),
            manifest_indices, 13u) ||
        !persistent_state_equal(&uninterrupted, &resumed)) {
        goto cleanup;
    }
    passed = 1;

cleanup:
    (void)remove(path);
    fixture_free(&uninterrupted);
    fixture_free(&checkpoint_source);
    fixture_free(&resumed);
    return passed;
}


int main(int argc, char **argv)
{
    VAECProfileCatalog catalog;
    const VAECProfile *profile;
    int failures = 0;

    if (argc != 3) {
        fprintf(stderr,
                "usage: %s <vae_c_profiles_v1.csv> <fixture_directory>\n",
                argv[0]);
        return 2;
    }
    vae_c_profile_catalog_init(&catalog);
    if (vae_c_profile_catalog_load_csv(
            argv[1], &catalog, NULL, 0) != VAE_C_CONFIG_OK) {
        fprintf(stderr, "failed to load VAE profile catalog\n");
        return 2;
    }
    profile = vae_c_profile_catalog_find(&catalog, "mini75_smoke_v1");
    if (profile == NULL) {
        vae_c_profile_catalog_free(&catalog);
        fprintf(stderr, "mini75_smoke_v1 profile not found\n");
        return 2;
    }

    failures += report_check(
        "sha256_standard_golden_vectors",
        test_sha256_standard_golden_vectors()
    );
    failures += report_check(
        "bitwise_round_trip_transient_exclusion_and_lifecycle",
        test_round_trip_and_lifecycle(profile, argv[2])
    );
    failures += report_check(
        "provenance_is_metadata_and_best_undefined_round_trip",
        test_provenance_not_hard_gate_and_best_undefined(profile, argv[2])
    );
    failures += report_check(
        "byte_identical_checkpoint_saves",
        test_byte_identical_saves(profile, argv[2])
    );
    failures += report_check(
        "lifecycle_generation_boundaries",
        test_lifecycle_generation_boundaries(profile, argv[2])
    );
    failures += report_check(
        "corruption_incompatibility_and_transactional_load",
        test_negative_files_and_transaction(profile, argv[2])
    );
    failures += report_check(
        "atomic_replace_target_preservation_and_tmp_refusal",
        test_atomic_save_contract(profile, argv[2])
    );
    failures += report_check(
        "deterministic_epoch_boundary_resume_depth4",
        test_deterministic_resume(argv[2])
    );

    vae_c_profile_catalog_free(&catalog);
    if (failures != 0) {
        fprintf(stderr, "%d VAE C CHECKPOINT TEST(S) FAILED\n", failures);
        return 1;
    }
    printf("CHECKPOINT_FORMAT_VERSION=1\n");
    printf("CHECKPOINT_ENDIANNESS=little\n");
    printf("CHECKPOINT_INTEGRITY=SHA-256\n");
    printf("CHECKPOINT_SHA256_GOLDEN_VECTORS=true\n");
    printf("CHECKPOINT_FILE_DETERMINISTIC=true\n");
    printf("CHECKPOINT_LOAD_TRANSACTIONAL=true\n");
    printf("CHECKPOINT_GENERATION_WRAPAROUND=false\n");
    printf("CHECKPOINT_EPOCH_UINT64_MAX_REJECTED=true\n");
    printf("CHECKPOINT_RESUME_BOUNDARY=epoch\n");
    printf("CHECKPOINT_RNG_STATE_SERIALIZED=false\n");
    printf("CHECKPOINT_WORKSPACE_SERIALIZED=false\n");
    printf("CHECKPOINT_DETERMINISTIC_RESUME_BITWISE=true\n");
    printf("ALL VAE C CHECKPOINT TESTS PASSED\n");
    return 0;
}
