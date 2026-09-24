#include "vae_c_model_train.h"

#include "vae_c_config.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


typedef struct {
    VAECModel model;
    VAECModelWorkspace workspace;
    VAECAdam optimizer;
    VAECVAELossStats stats;
} Fixture;


typedef enum {
    TEST_STATS_ALIAS_MODEL_PARAMETER_KEY = 0,
    TEST_STATS_ALIAS_WORKSPACE_TENSOR_DATA_FIELD,
    TEST_STATS_ALIAS_WORKSPACE_TENSOR_GRAD_FIELD,
    TEST_STATS_ALIAS_WORKSPACE_TENSOR_OWNED_BLOCK_FIELD,
    TEST_STATS_ALIAS_WORKSPACE_TICKET_FIELD,
    TEST_STATS_ALIAS_OPTIMIZER_STEP,
    TEST_STATS_ALIAS_MANIFEST_INDEX_ONE,
    TEST_STATS_ALIAS_CASE_COUNT
} TestStatsAliasCase;


static int report_check(const char *name, int passed)
{
    if (passed) {
        printf("[PASS] %s\n", name);
        return 0;
    }
    printf("[FAIL] %s\n", name);
    return 1;
}


static uint64_t hash_bytes(
    uint64_t hash,
    const void *memory,
    size_t byte_count
)
{
    const unsigned char *bytes = (const unsigned char *)memory;
    size_t index;

    for (index = 0; index < byte_count; index++) {
        hash ^= (uint64_t)bytes[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}


static size_t collect_workspace_tensors(
    const VAECModelWorkspace *workspace,
    const VAECTensor *tensors[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ]
)
{
    size_t count = 0;
    size_t index;

    tensors[count++] = &workspace->input;
    for (index = 0; index < workspace->config.conv_depth; index++) {
        tensors[count++] = &workspace->encoder_conv_outputs[index];
        tensors[count++] =
            &workspace->encoder_batch_norm_outputs[index];
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
    for (index = 0;
         index < workspace->config.decoder_upsample_count;
         index++) {

        tensors[count++] = &workspace->decoder_upsample_outputs[index];
    }
    for (index = 0;
         index < workspace->config.decoder_hidden_conv_count;
         index++) {

        tensors[count++] =
            &workspace->decoder_hidden_conv_outputs[index];
        tensors[count++] =
            &workspace->decoder_hidden_batch_norm_outputs[index];
        tensors[count++] =
            &workspace->decoder_hidden_relu_outputs[index];
    }
    tensors[count++] = &workspace->logits;
    return count;
}


static size_t collect_batch_norms(
    const VAECModel *model,
    const VAECBatchNorm *layers[
        VAE_C_MAX_CONV_DEPTH + 1 +
        VAE_C_MAX_DECODER_HIDDEN_CONVS
    ]
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


static void fixture_init(Fixture *fixture)
{
    vae_c_model_init(&fixture->model);
    vae_c_model_workspace_init(&fixture->workspace);
    vae_c_adam_init(&fixture->optimizer);
    vae_c_vae_loss_stats_init(&fixture->stats);
}


static void fixture_free(Fixture *fixture)
{
    vae_c_model_workspace_free(&fixture->workspace);
    vae_c_model_free(&fixture->model);
    vae_c_adam_free(&fixture->optimizer);
    vae_c_vae_loss_stats_init(&fixture->stats);
}


static void fill_input(VAECModelWorkspace *workspace)
{
    size_t index;

    for (index = 0; index < workspace->input.shape.count; index++) {
        unsigned long value =
            (unsigned long)((index * 37u + 11u) % 997u);

        workspace->input.data[index] =
            0.1f + 0.8f * (float)value / 996.0f;
    }
}


static void seed_parameter_gradients(VAECModel *model)
{
    size_t parameter_index;

    for (parameter_index = 0;
         parameter_index < model->parameter_count;
         parameter_index++) {

        VAECParameter *parameter = model->parameters[parameter_index];
        size_t element;

        for (element = 0; element < parameter->shape.count; element++) {
            parameter->gradients[element] =
                0.01f * (float)(1 + (parameter_index + element) % 7);
        }
    }
}


static void seed_workspace_gradients(VAECModelWorkspace *workspace)
{
    const VAECTensor *tensors[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ];
    size_t count = collect_workspace_tensors(workspace, tensors);
    size_t tensor_index;

    for (tensor_index = 0; tensor_index < count; tensor_index++) {
        VAECTensor *tensor = (VAECTensor *)(void *)tensors[tensor_index];
        size_t element;

        if (!tensor->has_grad) {
            continue;
        }
        for (element = 0; element < tensor->shape.count; element++) {
            tensor->grad[element] =
                0.02f * (float)(1 + (tensor_index + element) % 5);
        }
    }
}


static int fixture_allocate(
    Fixture *fixture,
    const VAECProfile *profile,
    uint64_t initialization_seed
)
{
    fixture_init(fixture);
    if (profile == NULL ||
        vae_c_model_allocate(&fixture->model, profile, 2) !=
            VAE_C_OP_OK ||
        vae_c_model_initialize_parameters_v1(
            &fixture->model, initialization_seed) != VAE_C_OP_OK ||
        vae_c_model_workspace_allocate(
            &fixture->model, 2, &fixture->workspace) != VAE_C_OP_OK ||
        vae_c_adam_configure_v1(&fixture->optimizer) != VAE_C_OP_OK) {

        return 0;
    }
    fill_input(&fixture->workspace);
    return
        vae_c_model_validate(&fixture->model) == VAE_C_OP_OK &&
        vae_c_model_workspace_validate(
            &fixture->model, &fixture->workspace) == VAE_C_OP_OK &&
        vae_c_adam_is_configured(&fixture->optimizer);
}


static void make_synthetic_depth4(VAECProfile *profile)
{
    memset(profile, 0, sizeof(*profile));
    memcpy(profile->profile_id,
           "synthetic_depth4_small",
           sizeof("synthetic_depth4_small"));
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


static int stats_equal(
    const VAECVAELossStats *left,
    const VAECVAELossStats *right
)
{
    return
        left->batch_size == right->batch_size &&
        left->beta_kl == right->beta_kl &&
        left->reconstruction.batch_size ==
            right->reconstruction.batch_size &&
        left->reconstruction.values_per_sample ==
            right->reconstruction.values_per_sample &&
        left->reconstruction.total_values ==
            right->reconstruction.total_values &&
        left->reconstruction.total_sum ==
            right->reconstruction.total_sum &&
        left->reconstruction.mean_per_sample ==
            right->reconstruction.mean_per_sample &&
        left->reconstruction.mean_per_value ==
            right->reconstruction.mean_per_value &&
        left->kl.batch_size == right->kl.batch_size &&
        left->kl.latent_dim == right->kl.latent_dim &&
        left->kl.total_values == right->kl.total_values &&
        left->kl.total_sum == right->kl.total_sum &&
        left->kl.mean_per_sample == right->kl.mean_per_sample &&
        left->kl.mean_per_dimension == right->kl.mean_per_dimension &&
        left->total_sum == right->total_sum &&
        left->mean_per_sample == right->mean_per_sample;
}


static void set_stats_sentinel(VAECVAELossStats *stats)
{
    memset(stats, 0x5A, sizeof(*stats));
}


static int batch_norm_equal(
    const VAECBatchNorm *left,
    const VAECBatchNorm *right
)
{
    return
        left->kind == right->kind &&
        left->num_features == right->num_features &&
        left->epsilon == right->epsilon &&
        left->momentum == right->momentum &&
        left->owned_bytes == right->owned_bytes &&
        memcmp(left->owned_block,
               right->owned_block,
               left->owned_bytes) == 0 &&
        left->last_sample_count == right->last_sample_count &&
        memcmp(&left->last_input_shape,
               &right->last_input_shape,
               sizeof(left->last_input_shape)) == 0 &&
        memcmp(&left->last_output_shape,
               &right->last_output_shape,
               sizeof(left->last_output_shape)) == 0 &&
        left->cache_valid == right->cache_valid;
}


static int fixtures_equal(
    const Fixture *left,
    const Fixture *right
)
{
    const VAECTensor *left_tensors[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ];
    const VAECTensor *right_tensors[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ];
    const VAECBatchNorm *left_batch_norms[
        VAE_C_MAX_CONV_DEPTH + 1 +
        VAE_C_MAX_DECODER_HIDDEN_CONVS
    ];
    const VAECBatchNorm *right_batch_norms[
        VAE_C_MAX_CONV_DEPTH + 1 +
        VAE_C_MAX_DECODER_HIDDEN_CONVS
    ];
    size_t tensor_count;
    size_t right_tensor_count;
    size_t batch_norm_count;
    size_t right_batch_norm_count;
    size_t index;

    if (memcmp(left->model.profile_id,
               right->model.profile_id,
               sizeof(left->model.profile_id)) != 0 ||
        memcmp(&left->model.config,
               &right->model.config,
               sizeof(left->model.config)) != 0 ||
        left->model.parameter_count != right->model.parameter_count ||
        left->model.vae_cache_generation !=
            right->model.vae_cache_generation ||
        memcmp(&left->optimizer,
               &right->optimizer,
               sizeof(left->optimizer)) != 0 ||
        !stats_equal(&left->stats, &right->stats)) {

        return 0;
    }
    for (index = 0; index < left->model.parameter_count; index++) {
        const VAECParameter *left_parameter =
            left->model.parameters[index];
        const VAECParameter *right_parameter =
            right->model.parameters[index];

        if (left_parameter->parameter_key !=
                right_parameter->parameter_key ||
            memcmp(&left_parameter->shape,
                   &right_parameter->shape,
                   sizeof(left_parameter->shape)) != 0 ||
            left_parameter->owned_bytes != right_parameter->owned_bytes ||
            memcmp(left_parameter->owned_block,
                   right_parameter->owned_block,
                   left_parameter->owned_bytes) != 0) {

            return 0;
        }
    }
    batch_norm_count = collect_batch_norms(
        &left->model, left_batch_norms
    );
    right_batch_norm_count = collect_batch_norms(
        &right->model, right_batch_norms
    );
    if (batch_norm_count != right_batch_norm_count) {
        return 0;
    }
    for (index = 0; index < batch_norm_count; index++) {
        if (!batch_norm_equal(
                left_batch_norms[index], right_batch_norms[index])) {

            return 0;
        }
    }
    if (left->model.encoder_max_pool.last_count !=
            right->model.encoder_max_pool.last_count ||
        left->model.encoder_max_pool.cache_valid !=
            right->model.encoder_max_pool.cache_valid ||
        memcmp(&left->model.encoder_max_pool.last_input_shape,
               &right->model.encoder_max_pool.last_input_shape,
               sizeof(VAECShape)) != 0 ||
        memcmp(&left->model.encoder_max_pool.last_output_shape,
               &right->model.encoder_max_pool.last_output_shape,
               sizeof(VAECShape)) != 0 ||
        memcmp(left->model.encoder_max_pool.argmax_indices,
               right->model.encoder_max_pool.argmax_indices,
               left->model.encoder_max_pool.owned_bytes) != 0) {

        return 0;
    }

    tensor_count = collect_workspace_tensors(
        &left->workspace, left_tensors
    );
    right_tensor_count = collect_workspace_tensors(
        &right->workspace, right_tensors
    );
    if (tensor_count != right_tensor_count ||
        left->workspace.vae_backward_model != NULL ||
        right->workspace.vae_backward_model != NULL ||
        left->workspace.vae_backward_generation != UINT64_C(0) ||
        right->workspace.vae_backward_generation != UINT64_C(0)) {

        return 0;
    }
    for (index = 0; index < tensor_count; index++) {
        if (memcmp(&left_tensors[index]->shape,
                   &right_tensors[index]->shape,
                   sizeof(VAECShape)) != 0 ||
            left_tensors[index]->owned_bytes !=
                right_tensors[index]->owned_bytes ||
            memcmp(left_tensors[index]->owned_block,
                   right_tensors[index]->owned_block,
                   left_tensors[index]->owned_bytes) != 0) {

            return 0;
        }
    }
    return 1;
}


static uint64_t fixture_state_hash(const Fixture *fixture)
{
    const VAECTensor *tensors[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ];
    const VAECBatchNorm *batch_norms[
        VAE_C_MAX_CONV_DEPTH + 1 +
        VAE_C_MAX_DECODER_HIDDEN_CONVS
    ];
    uint64_t hash = UINT64_C(1469598103934665603);
    size_t tensor_count = collect_workspace_tensors(
        &fixture->workspace, tensors
    );
    size_t batch_norm_count = collect_batch_norms(
        &fixture->model, batch_norms
    );
    size_t index;

    hash = hash_bytes(hash, &fixture->model, sizeof(fixture->model));
    hash = hash_bytes(
        hash, &fixture->workspace, sizeof(fixture->workspace)
    );
    hash = hash_bytes(
        hash, &fixture->optimizer, sizeof(fixture->optimizer)
    );
    hash = hash_bytes(hash, &fixture->stats, sizeof(fixture->stats));
    for (index = 0; index < fixture->model.parameter_count; index++) {
        const VAECParameter *parameter =
            fixture->model.parameters[index];

        hash = hash_bytes(
            hash, parameter->owned_block, parameter->owned_bytes
        );
    }
    for (index = 0; index < batch_norm_count; index++) {
        hash = hash_bytes(
            hash,
            batch_norms[index]->owned_block,
            batch_norms[index]->owned_bytes
        );
    }
    hash = hash_bytes(
        hash,
        fixture->model.encoder_max_pool.argmax_indices,
        fixture->model.encoder_max_pool.owned_bytes
    );
    for (index = 0; index < tensor_count; index++) {
        hash = hash_bytes(
            hash, tensors[index]->owned_block, tensors[index]->owned_bytes
        );
    }
    return hash;
}


static uint64_t update_state_hash(const Fixture *fixture)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    size_t parameter_index;

    hash = hash_bytes(
        hash, &fixture->optimizer, sizeof(fixture->optimizer)
    );
    for (parameter_index = 0;
         parameter_index < fixture->model.parameter_count;
         parameter_index++) {

        const VAECParameter *parameter =
            fixture->model.parameters[parameter_index];
        size_t bytes = parameter->shape.count * sizeof(float);

        hash = hash_bytes(hash, parameter->values, bytes);
        hash = hash_bytes(hash, parameter->adam_m, bytes);
        hash = hash_bytes(hash, parameter->adam_v, bytes);
    }
    return hash;
}


static uint64_t parameter_value_hash(const VAECModel *model)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    size_t index;

    for (index = 0; index < model->parameter_count; index++) {
        const VAECParameter *parameter = model->parameters[index];

        hash = hash_bytes(
            hash,
            parameter->values,
            parameter->shape.count * sizeof(float)
        );
    }
    return hash;
}


static int zero_parameter_gradients(VAECModel *model)
{
    size_t index;

    for (index = 0; index < model->parameter_count; index++) {
        if (vae_c_parameter_zero_grad(model->parameters[index]) !=
                VAE_C_STORAGE_OK) {

            return 0;
        }
    }
    return 1;
}


static int fill_epsilon_expected(
    VAECModelWorkspace *workspace,
    uint64_t base_seed,
    uint64_t epoch,
    const uint64_t *manifest_indices
)
{
    size_t sample;
    size_t latent_dim = workspace->config.latent_dim;

    for (sample = 0; sample < workspace->batch_size; sample++) {
        if (vae_c_rng_fill_latent_train_epsilon_f32(
                base_seed,
                epoch,
                manifest_indices[sample],
                workspace->epsilon.data + sample * latent_dim,
                latent_dim) != VAE_C_RNG_OK) {

            return 0;
        }
    }
    return 1;
}


static VAECOpStatus manual_train_step(
    Fixture *fixture,
    uint64_t base_seed,
    uint64_t epoch,
    const uint64_t *manifest_indices
)
{
    VAECVAELossStats local_stats;
    VAECOpStatus status;

    if (!zero_parameter_gradients(&fixture->model) ||
        !fill_epsilon_expected(
            &fixture->workspace,
            base_seed,
            epoch,
            manifest_indices)) {

        return VAE_C_OP_INVALID_STATE;
    }
    vae_c_vae_loss_stats_init(&local_stats);
    status = vae_c_model_vae_forward(
        &fixture->model,
        VAE_C_BATCH_NORM_MODE_TRAIN,
        VAE_C_LATENT_MODE_STOCHASTIC,
        &fixture->workspace,
        &local_stats
    );
    if (status == VAE_C_OP_OK) {
        status = vae_c_model_workspace_zero_grad(
            &fixture->model, &fixture->workspace
        );
    }
    if (status == VAE_C_OP_OK) {
        status = vae_c_model_vae_backward(
            &fixture->model, &fixture->workspace, 1.0f
        );
    }
    if (status == VAE_C_OP_OK) {
        status = vae_c_adam_step(
            &fixture->optimizer,
            fixture->model.parameters,
            fixture->model.parameter_count
        );
    }
    if (status == VAE_C_OP_OK) {
        fixture->stats = local_stats;
    }
    return status;
}


static int epsilon_matches_expected(
    const VAECModelWorkspace *workspace,
    uint64_t base_seed,
    uint64_t epoch,
    const uint64_t *manifest_indices
)
{
    size_t bytes = workspace->epsilon.shape.count * sizeof(float);
    float *expected = (float *)malloc(bytes);
    size_t sample;
    size_t latent_dim = workspace->config.latent_dim;
    int passed = expected != NULL;

    for (sample = 0; passed && sample < workspace->batch_size; sample++) {
        passed = vae_c_rng_fill_latent_train_epsilon_f32(
            base_seed,
            epoch,
            manifest_indices[sample],
            expected + sample * latent_dim,
            latent_dim
        ) == VAE_C_RNG_OK;
    }
    if (passed) {
        passed = memcmp(
            workspace->epsilon.data, expected, bytes
        ) == 0;
    }
    free(expected);
    return passed;
}


static int run_manual_equivalence(const VAECProfile *profile)
{
    const uint64_t base_seed = UINT64_C(0x123456789ABCDEF0);
    const uint64_t epoch = UINT64_C(17);
    const uint64_t manifest_indices[2] = {UINT64_C(41), UINT64_C(99)};
    const uint64_t manifest_before[2] = {UINT64_C(41), UINT64_C(99)};
    Fixture manual;
    Fixture composite;
    float *input_before = NULL;
    size_t input_bytes;
    uint64_t values_before;
    int passed = 0;

    fixture_init(&manual);
    fixture_init(&composite);
    if (!fixture_allocate(&manual, profile, UINT64_C(7001)) ||
        !fixture_allocate(&composite, profile, UINT64_C(7001))) {

        goto cleanup;
    }
    seed_parameter_gradients(&manual.model);
    seed_parameter_gradients(&composite.model);
    seed_workspace_gradients(&manual.workspace);
    seed_workspace_gradients(&composite.workspace);
    input_bytes = composite.workspace.input.shape.count * sizeof(float);
    input_before = (float *)malloc(input_bytes);
    if (input_before == NULL) {
        goto cleanup;
    }
    memcpy(input_before, composite.workspace.input.data, input_bytes);
    values_before = parameter_value_hash(&composite.model);

    if (manual_train_step(
            &manual, base_seed, epoch, manifest_indices) != VAE_C_OP_OK ||
        vae_c_model_train_step(
            &composite.model,
            &composite.workspace,
            &composite.optimizer,
            base_seed,
            epoch,
            manifest_indices,
            2,
            &composite.stats) != VAE_C_OP_OK) {

        goto cleanup;
    }
    passed =
        fixtures_equal(&manual, &composite) &&
        memcmp(manifest_indices,
               manifest_before,
               sizeof(manifest_indices)) == 0 &&
        memcmp(composite.workspace.input.data,
               input_before,
               input_bytes) == 0 &&
        epsilon_matches_expected(
            &composite.workspace,
            base_seed,
            epoch,
            manifest_indices) &&
        parameter_value_hash(&composite.model) != values_before &&
        composite.optimizer.step == UINT64_C(1) &&
        composite.workspace.vae_backward_model == NULL &&
        composite.workspace.vae_backward_generation == UINT64_C(0) &&
        composite.stats.batch_size == 2 &&
        isfinite(composite.stats.total_sum) &&
        isfinite(composite.stats.mean_per_sample);

cleanup:
    free(input_before);
    fixture_free(&manual);
    fixture_free(&composite);
    return passed;
}


static int test_manual_depth3(const VAECProfileCatalog *catalog)
{
    return run_manual_equivalence(
        vae_c_profile_catalog_find(catalog, "mini75_smoke_v1")
    );
}


static int test_manual_depth4(void)
{
    VAECProfile profile;

    make_synthetic_depth4(&profile);
    return run_manual_equivalence(&profile);
}


static int test_rng_determinism_and_order(
    const VAECProfileCatalog *catalog
)
{
    const VAECProfile *profile = vae_c_profile_catalog_find(
        catalog, "mini75_smoke_v1"
    );
    const uint64_t base_seed = UINT64_C(456);
    const uint64_t epoch = UINT64_MAX;
    const uint64_t ordered[2] = {UINT64_C(17), UINT64_C(8123)};
    const uint64_t reversed[2] = {UINT64_C(8123), UINT64_C(17)};
    const uint64_t ordered_before[2] = {
        UINT64_C(17), UINT64_C(8123)
    };
    const uint64_t reversed_before[2] = {
        UINT64_C(8123), UINT64_C(17)
    };
    Fixture first;
    Fixture repeat;
    Fixture swapped;
    size_t row_bytes;
    int passed = 0;

    fixture_init(&first);
    fixture_init(&repeat);
    fixture_init(&swapped);
    if (!fixture_allocate(&first, profile, UINT64_C(8001)) ||
        !fixture_allocate(&repeat, profile, UINT64_C(8001)) ||
        !fixture_allocate(&swapped, profile, UINT64_C(8001))) {

        goto cleanup;
    }
    if (vae_c_model_train_step(
            &first.model,
            &first.workspace,
            &first.optimizer,
            base_seed,
            epoch,
            ordered,
            2,
            &first.stats) != VAE_C_OP_OK ||
        vae_c_model_train_step(
            &repeat.model,
            &repeat.workspace,
            &repeat.optimizer,
            base_seed,
            epoch,
            ordered,
            2,
            &repeat.stats) != VAE_C_OP_OK ||
        vae_c_model_train_step(
            &swapped.model,
            &swapped.workspace,
            &swapped.optimizer,
            base_seed,
            epoch,
            reversed,
            2,
            &swapped.stats) != VAE_C_OP_OK) {

        goto cleanup;
    }
    row_bytes = first.workspace.config.latent_dim * sizeof(float);
    passed =
        fixtures_equal(&first, &repeat) &&
        memcmp(first.workspace.epsilon.data,
               swapped.workspace.epsilon.data +
                   swapped.workspace.config.latent_dim,
               row_bytes) == 0 &&
        memcmp(first.workspace.epsilon.data +
                   first.workspace.config.latent_dim,
               swapped.workspace.epsilon.data,
               row_bytes) == 0 &&
        epsilon_matches_expected(
            &first.workspace, base_seed, epoch, ordered) &&
        epsilon_matches_expected(
            &swapped.workspace, base_seed, epoch, reversed) &&
        memcmp(ordered, ordered_before, sizeof(ordered)) == 0 &&
        memcmp(reversed, reversed_before, sizeof(reversed)) == 0;

cleanup:
    fixture_free(&first);
    fixture_free(&repeat);
    fixture_free(&swapped);
    return passed;
}


static int test_ticket_lifecycle(const VAECProfileCatalog *catalog)
{
    const VAECProfile *profile = vae_c_profile_catalog_find(
        catalog, "mini75_smoke_v1"
    );
    const uint64_t manifest_indices[2] = {UINT64_C(3), UINT64_C(4)};
    Fixture ready;
    Fixture stale;
    Fixture manual;
    uint64_t ready_hash;
    const VAECModel *ticket_model;
    uint64_t ticket_generation;
    int passed = 0;

    fixture_init(&ready);
    fixture_init(&stale);
    fixture_init(&manual);
    if (!fixture_allocate(&ready, profile, UINT64_C(9001)) ||
        !fixture_allocate(&stale, profile, UINT64_C(9001)) ||
        !fixture_allocate(&manual, profile, UINT64_C(9001)) ||
        !fill_epsilon_expected(
            &ready.workspace,
            UINT64_C(11),
            UINT64_C(2),
            manifest_indices) ||
        vae_c_model_vae_forward(
            &ready.model,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            VAE_C_LATENT_MODE_STOCHASTIC,
            &ready.workspace,
            &ready.stats) != VAE_C_OP_OK) {

        goto cleanup;
    }
    ready_hash = fixture_state_hash(&ready);
    if (vae_c_model_train_step(
            &ready.model,
            &ready.workspace,
            &ready.optimizer,
            UINT64_C(11),
            UINT64_C(2),
            manifest_indices,
            2,
            &ready.stats) != VAE_C_OP_INVALID_STATE ||
        fixture_state_hash(&ready) != ready_hash) {

        goto cleanup;
    }

    stale.workspace.vae_backward_model = &stale.model;
    stale.workspace.vae_backward_generation = UINT64_C(1);
    if (stale.model.vae_cache_generation != UINT64_C(0) ||
        vae_c_model_workspace_validate(
            &stale.model, &stale.workspace) != VAE_C_OP_OK ||
        vae_c_model_train_step(
            &stale.model,
            &stale.workspace,
            &stale.optimizer,
            UINT64_C(11),
            UINT64_C(2),
            manifest_indices,
            2,
            &stale.stats) != VAE_C_OP_OK ||
        stale.workspace.vae_backward_model != NULL ||
        stale.workspace.vae_backward_generation != UINT64_C(0)) {

        goto cleanup;
    }

    if (!zero_parameter_gradients(&manual.model) ||
        !fill_epsilon_expected(
            &manual.workspace,
            UINT64_C(11),
            UINT64_C(2),
            manifest_indices) ||
        vae_c_model_vae_forward(
            &manual.model,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            VAE_C_LATENT_MODE_STOCHASTIC,
            &manual.workspace,
            &manual.stats) != VAE_C_OP_OK) {

        goto cleanup;
    }
    ticket_model = manual.workspace.vae_backward_model;
    ticket_generation = manual.workspace.vae_backward_generation;
    if (ticket_model != &manual.model ||
        ticket_generation != manual.model.vae_cache_generation ||
        vae_c_model_workspace_zero_grad(
            &manual.model, &manual.workspace) != VAE_C_OP_OK ||
        manual.workspace.vae_backward_model != ticket_model ||
        manual.workspace.vae_backward_generation != ticket_generation ||
        vae_c_model_vae_backward(
            &manual.model, &manual.workspace, 1.0f) != VAE_C_OP_OK ||
        manual.workspace.vae_backward_model != NULL ||
        manual.workspace.vae_backward_generation != UINT64_C(0)) {

        goto cleanup;
    }
    passed = 1;

cleanup:
    fixture_free(&ready);
    fixture_free(&stale);
    fixture_free(&manual);
    return passed;
}


static int expect_atomic_failure(
    Fixture *fixture,
    VAECAdam *optimizer,
    const uint64_t *manifest_indices,
    size_t manifest_index_count,
    VAECVAELossStats *stats,
    VAECOpStatus expected_status
)
{
    uint64_t before = fixture_state_hash(fixture);
    VAECOpStatus status = vae_c_model_train_step(
        &fixture->model,
        &fixture->workspace,
        optimizer,
        UINT64_C(123),
        UINT64_C(7),
        manifest_indices,
        manifest_index_count,
        stats
    );

    return status == expected_status &&
        fixture_state_hash(fixture) == before;
}


static const void *stats_alias_target(
    Fixture *fixture,
    const uint64_t *manifest_indices,
    TestStatsAliasCase alias_case
)
{
    switch (alias_case) {
        case TEST_STATS_ALIAS_MODEL_PARAMETER_KEY:
            return (const void *)
                &fixture->model.parameters[0]->parameter_key;
        case TEST_STATS_ALIAS_WORKSPACE_TENSOR_DATA_FIELD:
            return (const void *)&fixture->workspace.input.data;
        case TEST_STATS_ALIAS_WORKSPACE_TENSOR_GRAD_FIELD:
            return (const void *)&fixture->workspace.input.grad;
        case TEST_STATS_ALIAS_WORKSPACE_TENSOR_OWNED_BLOCK_FIELD:
            return (const void *)&fixture->workspace.input.owned_block;
        case TEST_STATS_ALIAS_WORKSPACE_TICKET_FIELD:
            return (const void *)
                &fixture->workspace.vae_backward_generation;
        case TEST_STATS_ALIAS_OPTIMIZER_STEP:
            return (const void *)&fixture->optimizer.step;
        case TEST_STATS_ALIAS_MANIFEST_INDEX_ONE:
            return (const void *)&manifest_indices[1];
        default:
            return NULL;
    }
}


static size_t count_selected_stats_aliases(
    Fixture *fixture,
    const uint64_t *manifest_indices,
    const void *pointer
)
{
    size_t count = 0;
    size_t alias_case;

    for (alias_case = 0;
         alias_case < (size_t)TEST_STATS_ALIAS_CASE_COUNT;
         alias_case++) {

        if (pointer == stats_alias_target(
                fixture,
                manifest_indices,
                (TestStatsAliasCase)alias_case)) {

            count++;
        }
    }
    return count;
}


static int run_stats_metadata_alias_case(
    const VAECProfile *profile,
    TestStatsAliasCase alias_case
)
{
    uint64_t manifest_indices[2] = {UINT64_C(71), UINT64_C(103)};
    const uint64_t manifest_before[2] = {
        UINT64_C(71), UINT64_C(103)
    };
    Fixture fixture;
    VAECVAELossStats *stats_argument;
    const void *target;
    uint64_t before;
    VAECOpStatus status;
    size_t alias_count;
    int passed = 0;

    fixture_init(&fixture);
    if (!fixture_allocate(&fixture, profile, UINT64_C(10501))) {
        goto cleanup;
    }
    seed_parameter_gradients(&fixture.model);
    seed_workspace_gradients(&fixture.workspace);
    target = stats_alias_target(
        &fixture, manifest_indices, alias_case
    );
    if (target == NULL ||
        (uintptr_t)target % _Alignof(VAECVAELossStats) != 0) {

        goto cleanup;
    }
    alias_count = count_selected_stats_aliases(
        &fixture, manifest_indices, target
    );
    before = fixture_state_hash(&fixture);
    stats_argument = (VAECVAELossStats *)(void *)target;
    status = vae_c_model_train_step(
        &fixture.model,
        &fixture.workspace,
        &fixture.optimizer,
        UINT64_C(123),
        UINT64_C(7),
        manifest_indices,
        2,
        stats_argument
    );
    stats_argument = &fixture.stats;

    passed =
        alias_count == 1 &&
        status == VAE_C_OP_INVALID_ARGUMENT &&
        fixture_state_hash(&fixture) == before &&
        memcmp(
            manifest_indices,
            manifest_before,
            sizeof(manifest_indices)) == 0 &&
        stats_argument == &fixture.stats &&
        vae_c_model_validate(&fixture.model) == VAE_C_OP_OK &&
        vae_c_model_workspace_validate(
            &fixture.model, &fixture.workspace) == VAE_C_OP_OK &&
        vae_c_adam_is_configured(&fixture.optimizer);

cleanup:
    fixture_free(&fixture);
    return passed;
}


static int test_stats_metadata_alias_preflight_atomic(
    const VAECProfileCatalog *catalog
)
{
    const VAECProfile *profile = vae_c_profile_catalog_find(
        catalog, "mini75_smoke_v1"
    );
    size_t alias_case;

    if (profile == NULL) {
        return 0;
    }
    for (alias_case = 0;
         alias_case < (size_t)TEST_STATS_ALIAS_CASE_COUNT;
         alias_case++) {

        if (!run_stats_metadata_alias_case(
                profile, (TestStatsAliasCase)alias_case)) {

            return 0;
        }
    }
    return 1;
}


static int test_preflight_negative_alias_overflow(
    const VAECProfileCatalog *catalog
)
{
    const VAECProfile *profile = vae_c_profile_catalog_find(
        catalog, "mini75_smoke_v1"
    );
    const uint64_t manifest_indices[2] = {UINT64_C(5), UINT64_C(6)};
    Fixture fixture;
    float saved_float;
    uint64_t saved_generation;
    uint64_t saved_step;
    int passed = 0;

    fixture_init(&fixture);
    if (!fixture_allocate(&fixture, profile, UINT64_C(10001))) {
        goto cleanup;
    }
    if (!expect_atomic_failure(
            &fixture,
            &fixture.optimizer,
            manifest_indices,
            1,
            &fixture.stats,
            VAE_C_OP_INVALID_STATE)) {

        goto cleanup;
    }

    saved_float = fixture.workspace.input.data[0];
    fixture.workspace.input.data[0] = NAN;
    if (!expect_atomic_failure(
            &fixture,
            &fixture.optimizer,
            manifest_indices,
            2,
            &fixture.stats,
            VAE_C_OP_INVALID_STATE)) {

        goto cleanup;
    }
    fixture.workspace.input.data[0] = saved_float;

    saved_float = fixture.model.parameters[0]->gradients[0];
    fixture.model.parameters[0]->gradients[0] = INFINITY;
    if (!expect_atomic_failure(
            &fixture,
            &fixture.optimizer,
            manifest_indices,
            2,
            &fixture.stats,
            VAE_C_OP_INVALID_STATE)) {

        goto cleanup;
    }
    fixture.model.parameters[0]->gradients[0] = saved_float;

    saved_step = fixture.optimizer.step;
    fixture.optimizer.step = UINT64_MAX;
    if (!expect_atomic_failure(
            &fixture,
            &fixture.optimizer,
            manifest_indices,
            2,
            &fixture.stats,
            VAE_C_OP_INVALID_STATE)) {

        goto cleanup;
    }
    fixture.optimizer.step = saved_step;

    saved_generation = fixture.model.vae_cache_generation;
    fixture.model.vae_cache_generation = UINT64_MAX - UINT64_C(1);
    if (!expect_atomic_failure(
            &fixture,
            &fixture.optimizer,
            manifest_indices,
            2,
            &fixture.stats,
            VAE_C_OP_OVERFLOW)) {

        goto cleanup;
    }
    fixture.model.vae_cache_generation = saved_generation;

    if (!expect_atomic_failure(
            &fixture,
            &fixture.optimizer,
            (const uint64_t *)(const void *)
                fixture.workspace.epsilon.data,
            2,
            &fixture.stats,
            VAE_C_OP_INVALID_ARGUMENT) ||
        !expect_atomic_failure(
            &fixture,
            &fixture.optimizer,
            manifest_indices,
            2,
            (VAECVAELossStats *)(void *)
                fixture.model.parameters[0]->gradients,
            VAE_C_OP_INVALID_ARGUMENT) ||
        !expect_atomic_failure(
            &fixture,
            (VAECAdam *)(void *)fixture.workspace.input.data,
            manifest_indices,
            2,
            &fixture.stats,
            VAE_C_OP_INVALID_ARGUMENT)) {

        goto cleanup;
    }

    if (vae_c_model_train_step(
            NULL,
            &fixture.workspace,
            &fixture.optimizer,
            UINT64_C(0),
            UINT64_C(0),
            manifest_indices,
            2,
            &fixture.stats) != VAE_C_OP_INVALID_ARGUMENT ||
        vae_c_model_train_step(
            &fixture.model,
            &fixture.workspace,
            &fixture.optimizer,
            UINT64_C(0),
            UINT64_C(0),
            NULL,
            2,
            &fixture.stats) != VAE_C_OP_INVALID_ARGUMENT ||
        vae_c_model_train_step(
            &fixture.model,
            &fixture.workspace,
            &fixture.optimizer,
            UINT64_C(0),
            UINT64_C(0),
            manifest_indices,
            2,
            NULL) != VAE_C_OP_INVALID_ARGUMENT) {

        goto cleanup;
    }
    passed = 1;

cleanup:
    fixture_free(&fixture);
    return passed;
}


static void set_final_hidden_constant(
    VAECModel *model,
    float constant
)
{
    size_t hidden_index = model->config.decoder_hidden_conv_count - 1;
    VAECBatchNorm *layer =
        &model->decoder_hidden_batch_norms[hidden_index];
    size_t index;

    for (index = 0; index < layer->num_features; index++) {
        layer->gamma.values[index] = 0.0f;
        layer->beta.values[index] = constant;
    }
}


static int test_failure_boundaries(const VAECProfileCatalog *catalog)
{
    const VAECProfile *profile = vae_c_profile_catalog_find(
        catalog, "mini75_smoke_v1"
    );
    const uint64_t manifest_indices[2] = {UINT64_C(31), UINT64_C(47)};
    Fixture forward_failure;
    Fixture backward_failure;
    Fixture adam_failure;
    VAECVAELossStats sentinel;
    VAECOpStatus forward_status;
    VAECOpStatus backward_status;
    VAECOpStatus adam_status;
    uint64_t before;
    size_t index;
    int passed = 0;

    fixture_init(&forward_failure);
    fixture_init(&backward_failure);
    fixture_init(&adam_failure);
    set_stats_sentinel(&sentinel);
    if (!fixture_allocate(
            &forward_failure, profile, UINT64_C(11001)) ||
        !fixture_allocate(
            &backward_failure, profile, UINT64_C(11002)) ||
        !fixture_allocate(
            &adam_failure, profile, UINT64_C(11003))) {

        goto cleanup;
    }

    for (index = 0;
         index < forward_failure.model.encoder_convs[0].weights.shape.count;
         index++) {

        forward_failure.model.encoder_convs[0].weights.values[index] =
            FLT_MAX;
    }
    forward_failure.stats = sentinel;
    before = update_state_hash(&forward_failure);
    forward_status = vae_c_model_train_step(
            &forward_failure.model,
            &forward_failure.workspace,
            &forward_failure.optimizer,
            UINT64_C(77),
            UINT64_C(5),
            manifest_indices,
            2,
            &forward_failure.stats);
    if (forward_status == VAE_C_OP_OK ||
        update_state_hash(&forward_failure) != before ||
        !stats_equal(&forward_failure.stats, &sentinel) ||
        forward_failure.optimizer.step != UINT64_C(0)) {

        printf("FORWARD_FAILURE_DIAGNOSTIC status=%s generation=%llu "
               "step=%llu\n",
               vae_c_op_status_name(forward_status),
               (unsigned long long)
                   forward_failure.model.vae_cache_generation,
               (unsigned long long)forward_failure.optimizer.step);
        goto cleanup;
    }

    for (index = 0;
         index < backward_failure.workspace.input.shape.count;
         index++) {

        backward_failure.workspace.input.data[index] = 0.0f;
    }
    set_final_hidden_constant(&backward_failure.model, 0.25f);
    for (index = 0;
         index < backward_failure.model.decoder_output_conv.weights
             .shape.count;
         index++) {

        backward_failure.model.decoder_output_conv.weights.values[index] =
            0.0f;
    }
    for (index = 0;
         index < backward_failure.model.decoder_output_conv.out_channels;
         index++) {

        size_t weight_index =
            index * backward_failure.model.decoder_output_conv.weights
                .shape.strides[0];

        backward_failure.model.decoder_output_conv.weights
            .values[weight_index] = FLT_MAX;
    }
    for (index = 0;
         index < backward_failure.model.decoder_output_conv.bias
             .shape.count;
         index++) {

        backward_failure.model.decoder_output_conv.bias.values[index] =
            100.0f;
    }
    backward_failure.stats = sentinel;
    before = update_state_hash(&backward_failure);
    backward_status = vae_c_model_train_step(
            &backward_failure.model,
            &backward_failure.workspace,
            &backward_failure.optimizer,
            UINT64_C(77),
            UINT64_C(5),
            manifest_indices,
            2,
            &backward_failure.stats);
    if (backward_status == VAE_C_OP_OK ||
        backward_failure.model.vae_cache_generation != UINT64_C(2) ||
        backward_failure.workspace.vae_backward_model != NULL ||
        backward_failure.workspace.vae_backward_generation !=
            UINT64_C(0) ||
        update_state_hash(&backward_failure) != before ||
        !stats_equal(&backward_failure.stats, &sentinel) ||
        backward_failure.optimizer.step != UINT64_C(0)) {

        printf("BACKWARD_FAILURE_DIAGNOSTIC status=%s generation=%llu "
               "step=%llu ticket=%llu update_same=%s stats_same=%s\n",
               vae_c_op_status_name(backward_status),
               (unsigned long long)
                   backward_failure.model.vae_cache_generation,
               (unsigned long long)backward_failure.optimizer.step,
               (unsigned long long)
                   backward_failure.workspace.vae_backward_generation,
               update_state_hash(&backward_failure) == before
                   ? "true" : "false",
               stats_equal(&backward_failure.stats, &sentinel)
                   ? "true" : "false");
        goto cleanup;
    }

    for (index = 0;
         index < adam_failure.workspace.input.shape.count;
         index++) {

        adam_failure.workspace.input.data[index] = 0.0f;
    }
    set_final_hidden_constant(&adam_failure.model, 1.0e20f);
    for (index = 0;
         index < adam_failure.model.decoder_output_conv.weights
             .shape.count;
         index++) {

        adam_failure.model.decoder_output_conv.weights.values[index] =
            0.0f;
    }
    adam_failure.stats = sentinel;
    before = update_state_hash(&adam_failure);
    adam_status = vae_c_model_train_step(
            &adam_failure.model,
            &adam_failure.workspace,
            &adam_failure.optimizer,
            UINT64_C(77),
            UINT64_C(5),
            manifest_indices,
            2,
            &adam_failure.stats);
    if (adam_status == VAE_C_OP_OK ||
        adam_failure.model.vae_cache_generation != UINT64_C(2) ||
        adam_failure.workspace.vae_backward_model != NULL ||
        adam_failure.workspace.vae_backward_generation != UINT64_C(0) ||
        update_state_hash(&adam_failure) != before ||
        !stats_equal(&adam_failure.stats, &sentinel) ||
        adam_failure.optimizer.step != UINT64_C(0)) {

        printf("ADAM_FAILURE_DIAGNOSTIC status=%s generation=%llu "
               "step=%llu ticket=%llu update_same=%s stats_same=%s\n",
               vae_c_op_status_name(adam_status),
               (unsigned long long)
                   adam_failure.model.vae_cache_generation,
               (unsigned long long)adam_failure.optimizer.step,
               (unsigned long long)
                   adam_failure.workspace.vae_backward_generation,
               update_state_hash(&adam_failure) == before
                   ? "true" : "false",
               stats_equal(&adam_failure.stats, &sentinel)
                   ? "true" : "false");
        goto cleanup;
    }
    passed = 1;

cleanup:
    fixture_free(&forward_failure);
    fixture_free(&backward_failure);
    fixture_free(&adam_failure);
    return passed;
}


int main(int argc, char **argv)
{
    VAECProfileCatalog catalog;
    int failures = 0;

    if (argc != 2) {
        fprintf(stderr, "usage: %s <vae_c_profiles_v1.csv>\n", argv[0]);
        return 2;
    }
    vae_c_profile_catalog_init(&catalog);
    if (vae_c_profile_catalog_load_csv(
            argv[1], &catalog, NULL, 0) != VAE_C_CONFIG_OK) {

        fprintf(stderr, "failed to load profile catalog\n");
        return 2;
    }

    failures += report_check(
        "manual_train_step_depth3",
        test_manual_depth3(&catalog)
    );
    failures += report_check(
        "manual_train_step_synthetic_depth4",
        test_manual_depth4()
    );
    failures += report_check(
        "event_epsilon_determinism_and_batch_order_independence",
        test_rng_determinism_and_order(&catalog)
    );
    failures += report_check(
        "ticket_ready_rejection_stale_acceptance_and_consumption",
        test_ticket_lifecycle(&catalog)
    );
    failures += report_check(
        "preflight_null_numeric_count_overflow_and_exact_alias",
        test_preflight_negative_alias_overflow(&catalog)
    );
    failures += report_check(
        "stats_metadata_exact_alias_preflight_atomic",
        test_stats_metadata_alias_preflight_atomic(&catalog)
    );
    failures += report_check(
        "forward_backward_and_adam_failure_boundaries",
        test_failure_boundaries(&catalog)
    );

    vae_c_profile_catalog_free(&catalog);
    if (failures != 0) {
        fprintf(stderr, "%d VAE C TRAINING TEST(S) FAILED\n", failures);
        return 1;
    }
    printf("VAE_TRAIN_PARAMETER_ACCUMULATION=false\n");
    printf("VAE_TRAIN_EPSILON_DOMAIN=LATENT_TRAIN\n");
    printf("VAE_TRAIN_EPSILON_BATCH_ORDER_INDEPENDENT=true\n");
    printf("VAE_TRAIN_FORWARD_MODE=STOCHASTIC_TRAIN\n");
    printf("VAE_TRAIN_BACKWARD_UPSTREAM=1\n");
    printf("VAE_TRAIN_STATS=pre_update\n");
    printf("VAE_TRAIN_STATS_PUBLISHED_AFTER_ADAM=true\n");
    printf("VAE_TRAIN_STRONG_GUARANTEE=preflight_only\n");
    printf("VAE_TRAIN_POST_PREFLIGHT_ROLLBACK=false\n");
    printf("ALL VAE C MODEL TRAINING TESTS PASSED\n");
    return 0;
}
