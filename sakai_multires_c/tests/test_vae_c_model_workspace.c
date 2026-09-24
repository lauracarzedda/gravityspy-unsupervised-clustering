#include "vae_c_model_workspace.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define TEST_MODEL_MAX_BATCH_NORM_BLOCKS \
    (VAE_C_MAX_CONV_DEPTH + 1 + VAE_C_MAX_DECODER_HIDDEN_CONVS)


typedef struct {
    const void *original_pointer;
    size_t byte_count;
    unsigned char *copy;
} ByteSnapshot;


typedef struct {
    VAECModel metadata;

    ByteSnapshot parameter_blocks[VAE_C_MODEL_MAX_PARAMETERS];
    size_t parameter_block_count;

    ByteSnapshot batch_norm_blocks[
        TEST_MODEL_MAX_BATCH_NORM_BLOCKS
    ];
    size_t batch_norm_block_count;

    ByteSnapshot max_pool_block;
    bool captured;
} ModelDeepSnapshot;


static int report_check(const char *name, int passed)
{
    if (passed) {
        printf("[PASS] %s\n", name);
        return 0;
    }
    fprintf(stderr, "[FAIL] %s\n", name);
    return 1;
}


static void byte_snapshot_init(ByteSnapshot *snapshot)
{
    if (snapshot != NULL) {
        memset(snapshot, 0, sizeof(*snapshot));
    }
}


static void byte_snapshot_free(ByteSnapshot *snapshot)
{
    if (snapshot != NULL) {
        free(snapshot->copy);
        byte_snapshot_init(snapshot);
    }
}


static int byte_snapshot_capture(
    const void *pointer,
    size_t byte_count,
    ByteSnapshot *snapshot
)
{
    unsigned char *copy;

    if (pointer == NULL || byte_count == 0 || snapshot == NULL ||
        snapshot->copy != NULL) {

        return 0;
    }
    copy = (unsigned char *)malloc(byte_count);
    if (copy == NULL) {
        return 0;
    }
    memcpy(copy, pointer, byte_count);
    snapshot->original_pointer = pointer;
    snapshot->byte_count = byte_count;
    snapshot->copy = copy;
    return 1;
}


static int byte_snapshot_matches(
    const void *pointer,
    size_t byte_count,
    const ByteSnapshot *snapshot
)
{
    return
        snapshot != NULL && snapshot->copy != NULL &&
        pointer == snapshot->original_pointer &&
        byte_count == snapshot->byte_count &&
        memcmp(pointer, snapshot->copy, byte_count) == 0;
}


static size_t collect_model_batch_norms(
    const VAECModel *model,
    const VAECBatchNorm *layers[TEST_MODEL_MAX_BATCH_NORM_BLOCKS]
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


static void model_deep_snapshot_init(ModelDeepSnapshot *snapshot)
{
    if (snapshot != NULL) {
        memset(snapshot, 0, sizeof(*snapshot));
    }
}


static void model_deep_snapshot_free(ModelDeepSnapshot *snapshot)
{
    size_t index;

    if (snapshot == NULL) {
        return;
    }
    for (index = 0;
         index < VAE_C_MODEL_MAX_PARAMETERS;
         index++) {

        byte_snapshot_free(&snapshot->parameter_blocks[index]);
    }
    for (index = 0;
         index < TEST_MODEL_MAX_BATCH_NORM_BLOCKS;
         index++) {

        byte_snapshot_free(&snapshot->batch_norm_blocks[index]);
    }
    byte_snapshot_free(&snapshot->max_pool_block);
    model_deep_snapshot_init(snapshot);
}


static int model_deep_snapshot_capture(
    const VAECModel *model,
    ModelDeepSnapshot *snapshot
)
{
    const VAECBatchNorm *batch_norms[
        TEST_MODEL_MAX_BATCH_NORM_BLOCKS
    ];
    ModelDeepSnapshot candidate;
    size_t batch_norm_count;
    size_t index;

    if (model == NULL || snapshot == NULL || snapshot->captured ||
        model->parameter_count > VAE_C_MODEL_MAX_PARAMETERS) {

        return 0;
    }
    model_deep_snapshot_init(&candidate);
    candidate.metadata = *model;
    candidate.parameter_block_count = model->parameter_count;
    for (index = 0; index < model->parameter_count; index++) {
        const VAECParameter *parameter = model->parameters[index];

        if (parameter == NULL ||
            !byte_snapshot_capture(
                parameter->owned_block,
                parameter->owned_bytes,
                &candidate.parameter_blocks[index])) {

            model_deep_snapshot_free(&candidate);
            return 0;
        }
    }
    batch_norm_count = collect_model_batch_norms(model, batch_norms);
    if (batch_norm_count > TEST_MODEL_MAX_BATCH_NORM_BLOCKS) {
        model_deep_snapshot_free(&candidate);
        return 0;
    }
    candidate.batch_norm_block_count = batch_norm_count;
    for (index = 0; index < batch_norm_count; index++) {
        if (!byte_snapshot_capture(
                batch_norms[index]->owned_block,
                batch_norms[index]->owned_bytes,
                &candidate.batch_norm_blocks[index])) {

            model_deep_snapshot_free(&candidate);
            return 0;
        }
    }
    if (!byte_snapshot_capture(
            model->encoder_max_pool.argmax_indices,
            model->encoder_max_pool.owned_bytes,
            &candidate.max_pool_block)) {

        model_deep_snapshot_free(&candidate);
        return 0;
    }
    candidate.captured = true;
    *snapshot = candidate;
    model_deep_snapshot_init(&candidate);
    return 1;
}


static int model_deep_snapshot_matches(
    const VAECModel *model,
    const ModelDeepSnapshot *snapshot
)
{
    const VAECBatchNorm *batch_norms[
        TEST_MODEL_MAX_BATCH_NORM_BLOCKS
    ];
    size_t batch_norm_count;
    size_t index;

    if (model == NULL || snapshot == NULL || !snapshot->captured ||
        model->parameter_count != snapshot->parameter_block_count) {

        return 0;
    }
    for (index = 0; index < model->parameter_count; index++) {
        const VAECParameter *parameter = model->parameters[index];

        if (parameter == NULL ||
            parameter != snapshot->metadata.parameters[index] ||
            !byte_snapshot_matches(
                parameter->owned_block,
                parameter->owned_bytes,
                &snapshot->parameter_blocks[index])) {

            return 0;
        }
    }
    batch_norm_count = collect_model_batch_norms(model, batch_norms);
    if (batch_norm_count != snapshot->batch_norm_block_count) {
        return 0;
    }
    for (index = 0; index < batch_norm_count; index++) {
        if (!byte_snapshot_matches(
                batch_norms[index]->owned_block,
                batch_norms[index]->owned_bytes,
                &snapshot->batch_norm_blocks[index])) {

            return 0;
        }
    }
    if (!byte_snapshot_matches(
            model->encoder_max_pool.argmax_indices,
            model->encoder_max_pool.owned_bytes,
            &snapshot->max_pool_block)) {

        return 0;
    }
    return memcmp(model, &snapshot->metadata, sizeof(*model)) == 0;
}


static size_t collect_owned_const(
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


static size_t collect_owned_mutable(
    VAECModelWorkspace *workspace,
    VAECTensor *tensors[VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS]
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


typedef struct {
    VAECModelWorkspace metadata;
    ByteSnapshot owned_blocks[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ];
    size_t owned_block_count;
    bool captured;
} WorkspaceDeepSnapshot;


static void workspace_deep_snapshot_init(
    WorkspaceDeepSnapshot *snapshot
)
{
    if (snapshot != NULL) {
        memset(snapshot, 0, sizeof(*snapshot));
    }
}


static void workspace_deep_snapshot_free(
    WorkspaceDeepSnapshot *snapshot
)
{
    size_t index;

    if (snapshot == NULL) {
        return;
    }
    for (index = 0;
         index < VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS;
         index++) {

        byte_snapshot_free(&snapshot->owned_blocks[index]);
    }
    workspace_deep_snapshot_init(snapshot);
}


static int workspace_deep_snapshot_capture(
    const VAECModelWorkspace *workspace,
    WorkspaceDeepSnapshot *snapshot
)
{
    const VAECTensor *owned[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ];
    WorkspaceDeepSnapshot candidate;
    size_t count;
    size_t index;

    if (workspace == NULL || snapshot == NULL || snapshot->captured) {
        return 0;
    }
    workspace_deep_snapshot_init(&candidate);
    candidate.metadata = *workspace;
    count = collect_owned_const(workspace, owned);
    if (count > VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS) {
        return 0;
    }
    candidate.owned_block_count = count;
    for (index = 0; index < count; index++) {
        size_t array_count = owned[index]->has_grad ? 2 : 1;
        size_t float_count;
        size_t byte_count;

        if (owned[index]->shape.count > SIZE_MAX / array_count) {
            workspace_deep_snapshot_free(&candidate);
            return 0;
        }
        float_count = owned[index]->shape.count * array_count;
        if (float_count > SIZE_MAX / sizeof(float)) {
            workspace_deep_snapshot_free(&candidate);
            return 0;
        }
        byte_count = float_count * sizeof(float);
        if (!byte_snapshot_capture(
                owned[index]->owned_block,
                byte_count,
                &candidate.owned_blocks[index])) {

            workspace_deep_snapshot_free(&candidate);
            return 0;
        }
    }
    candidate.captured = true;
    *snapshot = candidate;
    workspace_deep_snapshot_init(&candidate);
    return 1;
}


static int workspace_deep_snapshot_matches(
    const VAECModelWorkspace *workspace,
    const WorkspaceDeepSnapshot *snapshot
)
{
    const VAECTensor *owned[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ];
    size_t count;
    size_t index;

    if (workspace == NULL || snapshot == NULL || !snapshot->captured ||
        memcmp(workspace,
               &snapshot->metadata,
               sizeof(*workspace)) != 0) {

        return 0;
    }
    count = collect_owned_const(workspace, owned);
    if (count != snapshot->owned_block_count) {
        return 0;
    }
    for (index = 0; index < count; index++) {
        size_t array_count = owned[index]->has_grad ? 2 : 1;
        size_t byte_count =
            owned[index]->shape.count * array_count * sizeof(float);

        if (!byte_snapshot_matches(
                owned[index]->owned_block,
                byte_count,
                &snapshot->owned_blocks[index])) {

            return 0;
        }
    }
    return 1;
}


static int shape_is(
    const VAECTensor *tensor,
    size_t rank,
    size_t d0,
    size_t d1,
    size_t d2,
    size_t d3
)
{
    const size_t dims[4] = {d0, d1, d2, d3};
    VAECShape expected;

    return
        vae_c_shape_make(rank, dims, &expected) == VAE_C_STORAGE_OK &&
        vae_c_shape_equal(&tensor->shape, &expected);
}


static int all_tensor_storage_zero(const VAECModelWorkspace *workspace)
{
    const VAECTensor *owned[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ];
    size_t count = collect_owned_const(workspace, owned);
    size_t index;
    size_t value;

    for (index = 0; index < count; index++) {
        for (value = 0; value < owned[index]->shape.count; value++) {
            if (owned[index]->data[value] != 0.0f ||
                (owned[index]->has_grad &&
                 owned[index]->grad[value] != 0.0f)) {

                return 0;
            }
        }
    }
    return 1;
}


static int all_batch_dimensions_match(
    const VAECModelWorkspace *workspace
)
{
    const VAECTensor *owned[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ];
    size_t count = collect_owned_const(workspace, owned);
    size_t index;

    for (index = 0; index < count; index++) {
        if (owned[index]->shape.dims[0] != workspace->batch_size) {
            return 0;
        }
    }
    return
        workspace->decoder_reshape_view.shape.dims[0] ==
            workspace->batch_size;
}


static int generic_shapes_match(const VAECModelWorkspace *workspace)
{
    size_t depth = workspace->config.conv_depth;
    size_t index;

    if (!shape_is(&workspace->input, 4,
                  workspace->batch_size, 4,
                  workspace->config.input_side,
                  workspace->config.input_side) ||
        !shape_is(&workspace->encoder_pool_output, 4,
                  workspace->batch_size,
                  workspace->config.channels[0],
                  workspace->config.pool_side,
                  workspace->config.pool_side) ||
        !shape_is(&workspace->encoder_global_avg_output, 2,
                  workspace->batch_size,
                  workspace->config.channels[depth - 1], 0, 0) ||
        !shape_is(&workspace->encoder_packed_output, 2,
                  workspace->batch_size,
                  2 * workspace->config.latent_dim, 0, 0) ||
        !shape_is(&workspace->mu, 2,
                  workspace->batch_size,
                  workspace->config.latent_dim, 0, 0) ||
        !shape_is(&workspace->logvar, 2,
                  workspace->batch_size,
                  workspace->config.latent_dim, 0, 0) ||
        !shape_is(&workspace->epsilon, 2,
                  workspace->batch_size,
                  workspace->config.latent_dim, 0, 0) ||
        !shape_is(&workspace->z, 2,
                  workspace->batch_size,
                  workspace->config.latent_dim, 0, 0) ||
        !shape_is(&workspace->decoder_fc_output, 2,
                  workspace->batch_size,
                  workspace->config.decoder_fc_features, 0, 0) ||
        !shape_is(&workspace->decoder_fc_batch_norm_output, 2,
                  workspace->batch_size,
                  workspace->config.decoder_fc_features, 0, 0) ||
        !shape_is(&workspace->decoder_fc_relu_output, 2,
                  workspace->batch_size,
                  workspace->config.decoder_fc_features, 0, 0) ||
        !shape_is(&workspace->decoder_reshape_view, 4,
                  workspace->batch_size,
                  workspace->config.channels[depth - 1],
                  workspace->config.bottleneck_side,
                  workspace->config.bottleneck_side) ||
        !shape_is(&workspace->logits, 4,
                  workspace->batch_size, 4,
                  workspace->config.input_side,
                  workspace->config.input_side)) {

        return 0;
    }
    for (index = 0; index < depth; index++) {
        size_t side = workspace->config.encoder_conv_sides[index];
        size_t channels = workspace->config.channels[index];

        if (!shape_is(&workspace->encoder_conv_outputs[index], 4,
                      workspace->batch_size, channels, side, side) ||
            !shape_is(
                &workspace->encoder_batch_norm_outputs[index], 4,
                workspace->batch_size, channels, side, side) ||
            !shape_is(&workspace->encoder_relu_outputs[index], 4,
                      workspace->batch_size, channels, side, side)) {

            return 0;
        }
    }
    for (index = 0;
         index < workspace->config.decoder_upsample_count;
         index++) {

        size_t channels = index < 2
            ? workspace->config.channels[depth - 1]
            : workspace->config.channels[depth - index];
        size_t side = workspace->config.decoder_upsample_targets[index];

        if (!shape_is(&workspace->decoder_upsample_outputs[index], 4,
                      workspace->batch_size, channels, side, side)) {

            return 0;
        }
    }
    for (index = 0;
         index < workspace->config.decoder_hidden_conv_count;
         index++) {

        size_t channels = workspace->config.channels[depth - 2 - index];
        size_t side = workspace->config.decoder_hidden_conv_sides[index];

        if (!shape_is(
                &workspace->decoder_hidden_conv_outputs[index], 4,
                workspace->batch_size, channels, side, side) ||
            !shape_is(
                &workspace->decoder_hidden_batch_norm_outputs[index], 4,
                workspace->batch_size, channels, side, side) ||
            !shape_is(
                &workspace->decoder_hidden_relu_outputs[index], 4,
                workspace->batch_size, channels, side, side)) {

            return 0;
        }
    }
    return all_batch_dimensions_match(workspace);
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


static int test_lifecycle_basics(void)
{
    VAECModelWorkspace workspace;
    VAECModelWorkspace before;

    vae_c_model_workspace_init(&workspace);
    before = workspace;
    if (!vae_c_model_workspace_is_empty(&workspace) ||
        vae_c_model_workspace_is_empty(NULL) ||
        vae_c_model_workspace_is_allocated(NULL, &workspace) ||
        vae_c_model_workspace_is_allocated(NULL, NULL) ||
        vae_c_model_workspace_validate(NULL, &workspace) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        vae_c_model_workspace_validate(NULL, NULL) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        vae_c_model_workspace_allocate(NULL, 1, &workspace) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        vae_c_model_workspace_allocate(NULL, 1, NULL) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        vae_c_model_workspace_zero_grad(NULL, NULL) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        memcmp(&workspace, &before, sizeof(workspace)) != 0 ||
        vae_c_model_workspace_zero_grad(NULL, &workspace) !=
            VAE_C_OP_OK) {

        return 0;
    }
    vae_c_model_workspace_free(&workspace);
    vae_c_model_workspace_free(&workspace);
    vae_c_model_workspace_free(NULL);
    return vae_c_model_workspace_is_empty(&workspace);
}


static int test_official_profiles(const VAECProfileCatalog *catalog)
{
    const char *ids[4] = {
        "mini75_smoke_v1",
        "mini75_experiment_v1",
        "mini112_primary_v1",
        "mini112_scaleup_v1"
    };
    const size_t logical[4] = {454416, 818448, 1782144, 3363584};
    const size_t storage[4] = {908768, 1636832, 3564160, 6726912};
    size_t profile_index;

    for (profile_index = 0; profile_index < 4; profile_index++) {
        const VAECProfile *profile = vae_c_profile_catalog_find(
            catalog, ids[profile_index]
        );
        VAECModel model;
        VAECModelWorkspace workspace;
        ModelDeepSnapshot model_before;

        vae_c_model_init(&model);
        vae_c_model_workspace_init(&workspace);
        model_deep_snapshot_init(&model_before);
        if (profile == NULL ||
            vae_c_model_allocate(&model, profile, 2) != VAE_C_OP_OK) {

            return 0;
        }
        if (!model_deep_snapshot_capture(&model, &model_before) ||
            vae_c_model_workspace_allocate(&model, 2, &workspace) !=
                VAE_C_OP_OK ||
            !model_deep_snapshot_matches(&model, &model_before) ||
            vae_c_model_workspace_validate(&model, &workspace) !=
                VAE_C_OP_OK ||
            !model_deep_snapshot_matches(&model, &model_before) ||
            !vae_c_model_workspace_is_allocated(&model, &workspace) ||
            !generic_shapes_match(&workspace) ||
            workspace.owned_tensor_count != 31 ||
            workspace.logical_value_count != logical[profile_index] ||
            workspace.storage_float_count != storage[profile_index] ||
            workspace.storage_bytes !=
                storage[profile_index] * sizeof(float) ||
            workspace.epsilon.has_grad || workspace.epsilon.grad != NULL ||
            workspace.decoder_reshape_view.owns_memory ||
            workspace.decoder_reshape_view.data !=
                workspace.decoder_fc_relu_output.data ||
            workspace.decoder_reshape_view.grad !=
                workspace.decoder_fc_relu_output.grad ||
            !all_tensor_storage_zero(&workspace) ||
            vae_c_model_workspace_zero_grad(&model, &workspace) !=
                VAE_C_OP_OK ||
            !model_deep_snapshot_matches(&model, &model_before)) {

            vae_c_model_workspace_free(&workspace);
            model_deep_snapshot_free(&model_before);
            vae_c_model_free(&model);
            return 0;
        }
        vae_c_model_workspace_free(&workspace);
        if (!vae_c_model_workspace_is_empty(&workspace) ||
            !model_deep_snapshot_matches(&model, &model_before)) {

            model_deep_snapshot_free(&model_before);
            vae_c_model_free(&model);
            return 0;
        }
        vae_c_model_workspace_free(&workspace);
        if (!model_deep_snapshot_matches(&model, &model_before)) {
            model_deep_snapshot_free(&model_before);
            vae_c_model_free(&model);
            return 0;
        }
        model_deep_snapshot_free(&model_before);
        model_deep_snapshot_free(&model_before);
        vae_c_model_free(&model);
    }
    return 1;
}


static int test_exact_official_shapes(const VAECProfileCatalog *catalog)
{
    const VAECProfile *smoke = vae_c_profile_catalog_find(
        catalog, "mini75_smoke_v1"
    );
    const VAECProfile *primary = vae_c_profile_catalog_find(
        catalog, "mini112_primary_v1"
    );
    VAECModel model;
    VAECModelWorkspace workspace;
    int passed;

    vae_c_model_init(&model);
    vae_c_model_workspace_init(&workspace);
    if (smoke == NULL || primary == NULL ||
        vae_c_model_allocate(&model, smoke, 2) != VAE_C_OP_OK ||
        vae_c_model_workspace_allocate(&model, 2, &workspace) !=
            VAE_C_OP_OK) {

        return 0;
    }
    passed =
        shape_is(&workspace.input, 4, 2, 4, 75, 75) &&
        shape_is(&workspace.encoder_conv_outputs[0], 4, 2, 8, 38, 38) &&
        shape_is(&workspace.encoder_pool_output, 4, 2, 8, 19, 19) &&
        shape_is(&workspace.encoder_conv_outputs[1], 4, 2, 16, 10, 10) &&
        shape_is(&workspace.encoder_conv_outputs[2], 4, 2, 32, 5, 5) &&
        shape_is(&workspace.encoder_global_avg_output, 2, 2, 32, 0, 0) &&
        shape_is(&workspace.encoder_packed_output, 2, 2, 64, 0, 0) &&
        shape_is(&workspace.mu, 2, 2, 32, 0, 0) &&
        shape_is(&workspace.decoder_fc_output, 2, 2, 800, 0, 0) &&
        shape_is(&workspace.decoder_reshape_view, 4, 2, 32, 5, 5) &&
        shape_is(&workspace.decoder_upsample_outputs[0], 4,
                 2, 32, 10, 10) &&
        shape_is(&workspace.decoder_upsample_outputs[1], 4,
                 2, 32, 19, 19) &&
        shape_is(&workspace.decoder_upsample_outputs[2], 4,
                 2, 16, 38, 38) &&
        shape_is(&workspace.decoder_upsample_outputs[3], 4,
                 2, 8, 75, 75) &&
        shape_is(&workspace.decoder_hidden_conv_outputs[0], 4,
                 2, 16, 19, 19) &&
        shape_is(&workspace.decoder_hidden_relu_outputs[1], 4,
                 2, 8, 38, 38) &&
        shape_is(&workspace.logits, 4, 2, 4, 75, 75);
    vae_c_model_workspace_free(&workspace);
    vae_c_model_free(&model);
    if (!passed) {
        return 0;
    }

    vae_c_model_init(&model);
    vae_c_model_workspace_init(&workspace);
    if (vae_c_model_allocate(&model, primary, 2) != VAE_C_OP_OK ||
        vae_c_model_workspace_allocate(&model, 2, &workspace) !=
            VAE_C_OP_OK) {

        return 0;
    }
    passed =
        shape_is(&workspace.input, 4, 2, 4, 112, 112) &&
        shape_is(&workspace.encoder_conv_outputs[0], 4, 2, 16, 56, 56) &&
        shape_is(&workspace.encoder_pool_output, 4, 2, 16, 28, 28) &&
        shape_is(&workspace.encoder_conv_outputs[1], 4, 2, 32, 14, 14) &&
        shape_is(&workspace.encoder_conv_outputs[2], 4, 2, 64, 7, 7) &&
        shape_is(&workspace.encoder_global_avg_output, 2, 2, 64, 0, 0) &&
        shape_is(&workspace.encoder_packed_output, 2, 2, 128, 0, 0) &&
        shape_is(&workspace.mu, 2, 2, 64, 0, 0) &&
        shape_is(&workspace.decoder_fc_output, 2, 2, 3136, 0, 0) &&
        shape_is(&workspace.decoder_reshape_view, 4, 2, 64, 7, 7) &&
        shape_is(&workspace.decoder_upsample_outputs[0], 4,
                 2, 64, 14, 14) &&
        shape_is(&workspace.decoder_upsample_outputs[1], 4,
                 2, 64, 28, 28) &&
        shape_is(&workspace.decoder_upsample_outputs[2], 4,
                 2, 32, 56, 56) &&
        shape_is(&workspace.decoder_upsample_outputs[3], 4,
                 2, 16, 112, 112) &&
        shape_is(&workspace.decoder_hidden_conv_outputs[0], 4,
                 2, 32, 28, 28) &&
        shape_is(&workspace.decoder_hidden_relu_outputs[1], 4,
                 2, 16, 56, 56) &&
        shape_is(&workspace.logits, 4, 2, 4, 112, 112);
    vae_c_model_workspace_free(&workspace);
    vae_c_model_free(&model);
    return passed;
}


static int test_synthetic_depth4(void)
{
    VAECProfile profile;
    VAECModel model;
    VAECModelWorkspace workspace;
    VAECTensor *owned[VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS];
    size_t offsets[VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS];
    ModelDeepSnapshot model_before;
    float *data_copy = NULL;
    size_t count;
    size_t total = 0;
    size_t index;
    size_t value;
    int passed;

    make_synthetic_depth4(&profile);
    vae_c_model_init(&model);
    vae_c_model_workspace_init(&workspace);
    model_deep_snapshot_init(&model_before);
    if (vae_c_model_allocate(&model, &profile, 2) != VAE_C_OP_OK ||
        vae_c_model_workspace_allocate(&model, 2, &workspace) !=
            VAE_C_OP_OK ||
        vae_c_model_workspace_validate(&model, &workspace) !=
            VAE_C_OP_OK) {

        return 0;
    }
    passed =
        generic_shapes_match(&workspace) &&
        workspace.owned_tensor_count == 38 &&
        workspace.logical_value_count == 31154 &&
        workspace.storage_float_count == 62302 &&
        workspace.storage_bytes == 62302 * sizeof(float) &&
        shape_is(&workspace.encoder_conv_outputs[0], 4, 2, 2, 16, 16) &&
        shape_is(&workspace.encoder_pool_output, 4, 2, 2, 8, 8) &&
        shape_is(&workspace.encoder_conv_outputs[1], 4, 2, 3, 4, 4) &&
        shape_is(&workspace.encoder_conv_outputs[2], 4, 2, 4, 2, 2) &&
        shape_is(&workspace.encoder_conv_outputs[3], 4, 2, 5, 1, 1) &&
        shape_is(&workspace.encoder_global_avg_output, 2, 2, 5, 0, 0) &&
        shape_is(&workspace.encoder_packed_output, 2, 2, 6, 0, 0) &&
        shape_is(&workspace.mu, 2, 2, 3, 0, 0) &&
        shape_is(&workspace.decoder_fc_output, 2, 2, 5, 0, 0) &&
        shape_is(&workspace.decoder_reshape_view, 4, 2, 5, 1, 1) &&
        shape_is(&workspace.decoder_upsample_outputs[0], 4, 2, 5, 2, 2) &&
        shape_is(&workspace.decoder_upsample_outputs[1], 4, 2, 5, 4, 4) &&
        shape_is(&workspace.decoder_upsample_outputs[2], 4, 2, 4, 8, 8) &&
        shape_is(&workspace.decoder_upsample_outputs[3], 4,
                 2, 3, 16, 16) &&
        shape_is(&workspace.decoder_upsample_outputs[4], 4,
                 2, 2, 32, 32) &&
        shape_is(&workspace.decoder_hidden_conv_outputs[0], 4,
                 2, 4, 4, 4) &&
        shape_is(&workspace.decoder_hidden_conv_outputs[1], 4,
                 2, 3, 8, 8) &&
        shape_is(&workspace.decoder_hidden_relu_outputs[2], 4,
                 2, 2, 16, 16) &&
        shape_is(&workspace.logits, 4, 2, 4, 32, 32);

    count = collect_owned_mutable(&workspace, owned);
    for (index = 0; index < count; index++) {
        offsets[index] = total;
        if (owned[index]->shape.count > SIZE_MAX - total) {
            passed = 0;
            break;
        }
        total += owned[index]->shape.count;
    }
    if (passed) {
        data_copy = (float *)malloc(total * sizeof(float));
        if (data_copy == NULL) {
            passed = 0;
        }
    }
    for (index = 0; passed && index < count; index++) {
        for (value = 0; value < owned[index]->shape.count; value++) {
            owned[index]->data[value] =
                (float)(index + 1) + (float)(value % 19) / 32.0f;
            data_copy[offsets[index] + value] = owned[index]->data[value];
            if (owned[index]->has_grad) {
                owned[index]->grad[value] =
                    (float)(index + 2) + (float)(value % 17) / 16.0f;
            }
        }
    }
    if (passed &&
        !model_deep_snapshot_capture(&model, &model_before)) {

        passed = 0;
    }
    if (passed &&
        (vae_c_model_workspace_zero_grad(&model, &workspace) !=
             VAE_C_OP_OK ||
         !model_deep_snapshot_matches(&model, &model_before) ||
         workspace.epsilon.has_grad || workspace.epsilon.grad != NULL ||
         workspace.decoder_reshape_view.grad == NULL)) {

        passed = 0;
    }
    for (index = 0; passed && index < count; index++) {
        if (memcmp(owned[index]->data,
                   data_copy + offsets[index],
                   owned[index]->shape.count * sizeof(float)) != 0) {

            passed = 0;
            break;
        }
        if (owned[index]->has_grad) {
            for (value = 0; value < owned[index]->shape.count; value++) {
                if (owned[index]->grad[value] != 0.0f) {
                    passed = 0;
                    break;
                }
            }
        }
    }
    if (passed && workspace.decoder_reshape_view.grad[0] != 0.0f) {
        passed = 0;
    }
    vae_c_model_workspace_free(&workspace);
    passed = passed &&
        vae_c_model_workspace_is_empty(&workspace) &&
        !vae_c_model_workspace_is_allocated(&model, &workspace) &&
        model_deep_snapshot_matches(&model, &model_before);
    vae_c_model_workspace_free(&workspace);
    passed = passed &&
        vae_c_model_workspace_is_empty(&workspace) &&
        !vae_c_model_workspace_is_allocated(&model, &workspace) &&
        model_deep_snapshot_matches(&model, &model_before);
    free(data_copy);
    model_deep_snapshot_free(&model_before);
    model_deep_snapshot_free(&model_before);
    vae_c_model_free(&model);
    return passed;
}


static int test_batch_and_allocate_guarantee(
    const VAECProfileCatalog *catalog
)
{
    const VAECProfile *profile = vae_c_profile_catalog_find(
        catalog, "mini75_smoke_v1"
    );
    VAECModel model;
    VAECModelWorkspace workspace;
    VAECModelWorkspace workspace_before;
    ModelDeepSnapshot model_before;
    size_t batches[2] = {1, 3};
    size_t index;

    vae_c_model_init(&model);
    model_deep_snapshot_init(&model_before);
    if (profile == NULL ||
        vae_c_model_allocate(&model, profile, 3) != VAE_C_OP_OK) {

        return 0;
    }
    if (!model_deep_snapshot_capture(&model, &model_before)) {
        vae_c_model_free(&model);
        return 0;
    }
    for (index = 0; index < 2; index++) {
        vae_c_model_workspace_init(&workspace);
        if (vae_c_model_workspace_allocate(
                &model, batches[index], &workspace) != VAE_C_OP_OK ||
            !all_batch_dimensions_match(&workspace) ||
            !model_deep_snapshot_matches(&model, &model_before)) {

            vae_c_model_workspace_free(&workspace);
            model_deep_snapshot_free(&model_before);
            vae_c_model_free(&model);
            return 0;
        }
        vae_c_model_workspace_free(&workspace);
    }

    if (vae_c_model_initialize_parameters_v1(
            &model, UINT64_C(987654321)) != VAE_C_OP_OK) {

        model_deep_snapshot_free(&model_before);
        vae_c_model_free(&model);
        return 0;
    }
    model_deep_snapshot_free(&model_before);
    if (!model_deep_snapshot_capture(&model, &model_before)) {
        vae_c_model_free(&model);
        return 0;
    }
    vae_c_model_workspace_init(&workspace);
    if (vae_c_model_workspace_allocate(&model, 2, &workspace) !=
            VAE_C_OP_OK ||
        !model_deep_snapshot_matches(&model, &model_before)) {

        vae_c_model_workspace_free(&workspace);
        model_deep_snapshot_free(&model_before);
        vae_c_model_free(&model);
        return 0;
    }
    vae_c_model_workspace_free(&workspace);

    vae_c_model_workspace_init(&workspace);
    workspace_before = workspace;
    if (vae_c_model_workspace_allocate(&model, 0, &workspace) ==
            VAE_C_OP_OK ||
        memcmp(&workspace, &workspace_before, sizeof(workspace)) != 0 ||
        vae_c_model_workspace_allocate(&model, 4, &workspace) ==
            VAE_C_OP_OK ||
        memcmp(&workspace, &workspace_before, sizeof(workspace)) != 0 ||
        !model_deep_snapshot_matches(&model, &model_before) ||
        vae_c_model_workspace_allocate(&model, 2, &workspace) !=
            VAE_C_OP_OK) {

        vae_c_model_workspace_free(&workspace);
        model_deep_snapshot_free(&model_before);
        vae_c_model_free(&model);
        return 0;
    }
    workspace_before = workspace;
    if (vae_c_model_workspace_allocate(&model, 2, &workspace) !=
            VAE_C_OP_OBJECT_NOT_EMPTY ||
        memcmp(&workspace, &workspace_before, sizeof(workspace)) != 0 ||
        !model_deep_snapshot_matches(&model, &model_before)) {

        vae_c_model_workspace_free(&workspace);
        model_deep_snapshot_free(&model_before);
        vae_c_model_free(&model);
        return 0;
    }
    vae_c_model_workspace_free(&workspace);

    model.encoder_max_pool.capacity -= 1;
    model_deep_snapshot_free(&model_before);
    if (!model_deep_snapshot_capture(&model, &model_before)) {
        model.encoder_max_pool.capacity += 1;
        vae_c_model_free(&model);
        return 0;
    }
    workspace_before = workspace;
    if (vae_c_model_workspace_allocate(&model, 2, &workspace) ==
            VAE_C_OP_OK ||
        memcmp(&workspace, &workspace_before, sizeof(workspace)) != 0 ||
        !model_deep_snapshot_matches(&model, &model_before)) {

        vae_c_model_workspace_free(&workspace);
        model.encoder_max_pool.capacity += 1;
        model_deep_snapshot_free(&model_before);
        vae_c_model_free(&model);
        return 0;
    }
    model.encoder_max_pool.capacity += 1;
    model_deep_snapshot_free(&model_before);
    vae_c_model_free(&model);
    return 1;
}


static int stack_pressure_view(const VAECModelWorkspace *workspace)
{
    volatile unsigned char pressure[32768];
    size_t index;

    for (index = 0; index < sizeof(pressure); index++) {
        pressure[index] = (unsigned char)(index % 251);
    }
    return
        pressure[250] == 250 &&
        workspace->decoder_reshape_view.data ==
            workspace->decoder_fc_relu_output.data &&
        workspace->decoder_reshape_view.grad ==
            workspace->decoder_fc_relu_output.grad;
}


static int test_reshape_view(const VAECProfileCatalog *catalog)
{
    const VAECProfile *profile = vae_c_profile_catalog_find(
        catalog, "mini75_smoke_v1"
    );
    VAECModel model;
    VAECModelWorkspace workspace;
    VAECShape view_shape;
    float *source_data;
    float *source_grad;
    int passed;

    vae_c_model_init(&model);
    vae_c_model_workspace_init(&workspace);
    if (profile == NULL ||
        vae_c_model_allocate(&model, profile, 2) != VAE_C_OP_OK ||
        vae_c_model_workspace_allocate(&model, 2, &workspace) !=
            VAE_C_OP_OK) {

        return 0;
    }
    source_data = workspace.decoder_fc_relu_output.data;
    source_grad = workspace.decoder_fc_relu_output.grad;
    view_shape = workspace.decoder_reshape_view.shape;
    workspace.decoder_reshape_view.data[7] = 19.0f;
    workspace.decoder_reshape_view.grad[9] = -3.0f;
    passed =
        workspace.decoder_fc_relu_output.data[7] == 19.0f &&
        workspace.decoder_fc_relu_output.grad[9] == -3.0f &&
        workspace.decoder_reshape_view.shape.count ==
            workspace.decoder_fc_relu_output.shape.count &&
        !workspace.decoder_reshape_view.owns_memory &&
        workspace.decoder_reshape_view.owned_block == NULL &&
        workspace.decoder_reshape_view.owned_bytes == 0 &&
        workspace.decoder_reshape_view.has_grad &&
        stack_pressure_view(&workspace);

    vae_c_tensor_free(&workspace.decoder_reshape_view);
    passed = passed &&
        workspace.decoder_fc_relu_output.data == source_data &&
        workspace.decoder_fc_relu_output.grad == source_grad &&
        workspace.decoder_fc_relu_output.data[7] == 19.0f &&
        vae_c_tensor_make_contiguous_view(
            &workspace.decoder_fc_relu_output,
            0,
            &view_shape,
            &workspace.decoder_reshape_view) == VAE_C_STORAGE_OK &&
        vae_c_model_workspace_validate(&model, &workspace) == VAE_C_OP_OK;
    vae_c_model_workspace_free(&workspace);
    vae_c_model_free(&model);
    return passed;
}


static int test_zero_grad_atomic(const VAECProfileCatalog *catalog)
{
    const VAECProfile *profile = vae_c_profile_catalog_find(
        catalog, "mini75_smoke_v1"
    );
    VAECModel model;
    VAECModelWorkspace workspace;
    VAECTensor *owned[VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS];
    size_t offsets[VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS];
    size_t count;
    size_t index;
    size_t value;
    size_t total = 0;
    float *data_copy;
    float *grad_copy;
    ModelDeepSnapshot model_before;
    WorkspaceDeepSnapshot corrupted_before;
    size_t saved_bytes;
    int passed = 1;

    vae_c_model_init(&model);
    vae_c_model_workspace_init(&workspace);
    model_deep_snapshot_init(&model_before);
    workspace_deep_snapshot_init(&corrupted_before);
    if (profile == NULL ||
        vae_c_model_allocate(&model, profile, 2) != VAE_C_OP_OK ||
        vae_c_model_workspace_allocate(&model, 2, &workspace) !=
            VAE_C_OP_OK) {

        return 0;
    }
    count = collect_owned_mutable(&workspace, owned);
    for (index = 0; index < count; index++) {
        offsets[index] = total;
        total += owned[index]->shape.count;
    }
    data_copy = (float *)malloc(total * sizeof(float));
    grad_copy = (float *)malloc(total * sizeof(float));
    if (data_copy == NULL || grad_copy == NULL) {
        free(data_copy);
        free(grad_copy);
        vae_c_model_workspace_free(&workspace);
        vae_c_model_free(&model);
        return 0;
    }
    for (index = 0; index < count; index++) {
        for (value = 0; value < owned[index]->shape.count; value++) {
            owned[index]->data[value] =
                (float)(index + 1) + (float)(value % 17) / 32.0f;
            data_copy[offsets[index] + value] = owned[index]->data[value];
            if (owned[index]->has_grad) {
                owned[index]->grad[value] =
                    (float)(index + 2) + (float)(value % 13) / 16.0f;
                grad_copy[offsets[index] + value] =
                    owned[index]->grad[value];
            } else {
                grad_copy[offsets[index] + value] = 0.0f;
            }
        }
    }
    if (!model_deep_snapshot_capture(&model, &model_before) ||
        vae_c_model_workspace_zero_grad(&model, &workspace) !=
            VAE_C_OP_OK ||
        !model_deep_snapshot_matches(&model, &model_before)) {

        passed = 0;
    }
    for (index = 0; passed && index < count; index++) {
        if (memcmp(owned[index]->data,
                   data_copy + offsets[index],
                   owned[index]->shape.count * sizeof(float)) != 0) {

            passed = 0;
        }
        if (owned[index]->has_grad) {
            for (value = 0; value < owned[index]->shape.count; value++) {
                if (owned[index]->grad[value] != 0.0f) {
                    passed = 0;
                    break;
                }
            }
        }
    }
    if (passed &&
        (workspace.epsilon.grad != NULL || workspace.epsilon.has_grad ||
         workspace.decoder_reshape_view.grad[0] != 0.0f ||
         vae_c_model_workspace_zero_grad(&model, &workspace) !=
             VAE_C_OP_OK)) {

        passed = 0;
    }

    for (index = 0; index < count; index++) {
        if (owned[index]->has_grad) {
            for (value = 0; value < owned[index]->shape.count; value++) {
                owned[index]->grad[value] =
                    grad_copy[offsets[index] + value];
            }
        }
    }
    saved_bytes = workspace.logits.owned_bytes;
    workspace.logits.owned_bytes -= sizeof(float);
    if (!workspace_deep_snapshot_capture(
            &workspace, &corrupted_before) ||
        vae_c_model_workspace_zero_grad(&model, &workspace) ==
            VAE_C_OP_OK ||
        !workspace_deep_snapshot_matches(
            &workspace, &corrupted_before) ||
        !model_deep_snapshot_matches(&model, &model_before)) {

        passed = 0;
    }
    for (index = 0; passed && index < count; index++) {
        if (memcmp(owned[index]->data,
                   data_copy + offsets[index],
                   owned[index]->shape.count * sizeof(float)) != 0 ||
            (owned[index]->has_grad &&
             memcmp(owned[index]->grad,
                    grad_copy + offsets[index],
                    owned[index]->shape.count * sizeof(float)) != 0)) {

            passed = 0;
        }
    }
    workspace_deep_snapshot_free(&corrupted_before);
    workspace.logits.owned_bytes = saved_bytes;
    passed = passed &&
        vae_c_model_workspace_validate(&model, &workspace) == VAE_C_OP_OK &&
        vae_c_model_workspace_is_allocated(&model, &workspace) &&
        model_deep_snapshot_matches(&model, &model_before);
    free(data_copy);
    free(grad_copy);
    vae_c_model_workspace_free(&workspace);
    workspace_deep_snapshot_free(&corrupted_before);
    model_deep_snapshot_free(&model_before);
    vae_c_model_free(&model);
    return passed;
}


static int test_backward_ticket_lifecycle(
    const VAECProfileCatalog *catalog
)
{
    const VAECProfile *profile = vae_c_profile_catalog_find(
        catalog, "mini75_smoke_v1"
    );
    VAECModel model;
    VAECModelWorkspace workspace;
    int passed;

    vae_c_model_init(&model);
    vae_c_model_workspace_init(&workspace);
    if (profile == NULL ||
        vae_c_model_allocate(&model, profile, 2) != VAE_C_OP_OK ||
        vae_c_model_workspace_allocate(&model, 2, &workspace) !=
            VAE_C_OP_OK) {

        vae_c_model_workspace_free(&workspace);
        vae_c_model_free(&model);
        return 0;
    }

    passed =
        model.vae_cache_generation == UINT64_C(0) &&
        workspace.vae_backward_model == NULL &&
        workspace.vae_backward_generation == UINT64_C(0);

    model.vae_cache_generation = UINT64_C(7);
    workspace.vae_backward_model = &model;
    workspace.vae_backward_generation = UINT64_C(7);
    workspace.input.grad[0] = 1.0f;
    passed = passed &&
        vae_c_model_workspace_validate(&model, &workspace) == VAE_C_OP_OK &&
        vae_c_model_workspace_zero_grad(&model, &workspace) == VAE_C_OP_OK &&
        workspace.input.grad[0] == 0.0f &&
        workspace.vae_backward_model == &model &&
        workspace.vae_backward_generation == UINT64_C(7);

    model.vae_cache_generation = UINT64_C(8);
    passed = passed &&
        vae_c_model_workspace_validate(&model, &workspace) == VAE_C_OP_OK &&
        workspace.vae_backward_model == &model &&
        workspace.vae_backward_generation == UINT64_C(7);

    workspace.vae_backward_model = NULL;
    passed = passed &&
        vae_c_model_workspace_validate(&model, &workspace) ==
            VAE_C_OP_INVALID_STATE;
    workspace.vae_backward_model = &model;
    workspace.vae_backward_generation = UINT64_C(0);
    passed = passed &&
        vae_c_model_workspace_validate(&model, &workspace) ==
            VAE_C_OP_INVALID_STATE;

    workspace.vae_backward_model = NULL;
    workspace.vae_backward_generation = UINT64_C(0);
    passed = passed &&
        vae_c_model_workspace_validate(&model, &workspace) == VAE_C_OP_OK;
    vae_c_model_workspace_free(&workspace);
    passed = passed && vae_c_model_workspace_is_empty(&workspace) &&
        workspace.vae_backward_model == NULL &&
        workspace.vae_backward_generation == UINT64_C(0);
    vae_c_model_free(&model);
    return passed;
}


static int rejection_is_pure(
    const VAECModel *model,
    VAECModelWorkspace *workspace
)
{
    VAECModelWorkspace before = *workspace;
    ModelDeepSnapshot model_before;
    int passed;

    model_deep_snapshot_init(&model_before);
    if (!model_deep_snapshot_capture(model, &model_before)) {
        return 0;
    }
    passed =
        vae_c_model_workspace_validate(model, workspace) != VAE_C_OP_OK &&
        memcmp(workspace, &before, sizeof(*workspace)) == 0 &&
        !vae_c_model_workspace_is_allocated(model, workspace) &&
        memcmp(workspace, &before, sizeof(*workspace)) == 0 &&
        model_deep_snapshot_matches(model, &model_before);
    model_deep_snapshot_free(&model_before);
    return passed;
}


static int test_validation_corruptions(
    const VAECProfileCatalog *catalog
)
{
    const VAECProfile *profile = vae_c_profile_catalog_find(
        catalog, "mini75_smoke_v1"
    );
    VAECModel model;
    VAECModelWorkspace workspace;
    VAECTensor saved_tensor;
    VAECShape saved_shape;
    float *saved_pointer;
    size_t saved_size;
    bool saved_bool;
    char saved_char;
    int passed = 1;

    vae_c_model_init(&model);
    vae_c_model_workspace_init(&workspace);
    if (profile == NULL ||
        vae_c_model_allocate(&model, profile, 2) != VAE_C_OP_OK ||
        vae_c_model_workspace_allocate(&model, 2, &workspace) !=
            VAE_C_OP_OK) {

        return 0;
    }

    saved_char = workspace.profile_id[0];
    workspace.profile_id[0] ^= 1;
    passed = passed && rejection_is_pure(&model, &workspace);
    workspace.profile_id[0] = saved_char;

    saved_size = workspace.config.pool_side;
    workspace.config.pool_side += 1;
    passed = passed && rejection_is_pure(&model, &workspace);
    workspace.config.pool_side = saved_size;

    saved_size = workspace.batch_size;
    workspace.batch_size = 0;
    passed = passed && rejection_is_pure(&model, &workspace);
    workspace.batch_size = saved_size;
    workspace.batch_size = model.max_batch_size + 1;
    passed = passed && rejection_is_pure(&model, &workspace);
    workspace.batch_size = saved_size;

    saved_bool = workspace.allocated;
    workspace.allocated = false;
    passed = passed && rejection_is_pure(&model, &workspace);
    workspace.allocated = saved_bool;

    saved_shape = workspace.input.shape;
    workspace.input.shape.dims[3] += 1;
    passed = passed && rejection_is_pure(&model, &workspace);
    workspace.input.shape = saved_shape;

    saved_bool = workspace.input.has_grad;
    workspace.input.has_grad = false;
    passed = passed && rejection_is_pure(&model, &workspace);
    workspace.input.has_grad = saved_bool;

    saved_size = workspace.input.owned_bytes;
    workspace.input.owned_bytes -= sizeof(float);
    passed = passed && rejection_is_pure(&model, &workspace);
    workspace.input.owned_bytes = saved_size;

    saved_bool = workspace.epsilon.has_grad;
    saved_pointer = workspace.epsilon.grad;
    workspace.epsilon.has_grad = true;
    workspace.epsilon.grad = workspace.epsilon.data;
    passed = passed && rejection_is_pure(&model, &workspace);
    workspace.epsilon.has_grad = saved_bool;
    workspace.epsilon.grad = saved_pointer;

    saved_tensor = workspace.mu;
    vae_c_tensor_init(&workspace.mu);
    passed = passed && rejection_is_pure(&model, &workspace);
    workspace.mu = saved_tensor;

    saved_tensor = workspace.encoder_conv_outputs[3];
    workspace.encoder_conv_outputs[3] = workspace.input;
    passed = passed && rejection_is_pure(&model, &workspace);
    workspace.encoder_conv_outputs[3] = saved_tensor;

    saved_bool = workspace.decoder_reshape_view.owns_memory;
    saved_pointer = workspace.decoder_reshape_view.owned_block;
    saved_size = workspace.decoder_reshape_view.owned_bytes;
    workspace.decoder_reshape_view.owns_memory = true;
    workspace.decoder_reshape_view.owned_block =
        workspace.decoder_reshape_view.data;
    workspace.decoder_reshape_view.owned_bytes = sizeof(float);
    passed = passed && rejection_is_pure(&model, &workspace);
    workspace.decoder_reshape_view.owns_memory = saved_bool;
    workspace.decoder_reshape_view.owned_block = saved_pointer;
    workspace.decoder_reshape_view.owned_bytes = saved_size;

    saved_pointer = workspace.decoder_reshape_view.data;
    workspace.decoder_reshape_view.data = workspace.input.data;
    passed = passed && rejection_is_pure(&model, &workspace);
    workspace.decoder_reshape_view.data = saved_pointer;

    saved_pointer = workspace.decoder_reshape_view.grad;
    workspace.decoder_reshape_view.grad = workspace.input.grad;
    passed = passed && rejection_is_pure(&model, &workspace);
    workspace.decoder_reshape_view.grad = saved_pointer;

    saved_shape = workspace.decoder_reshape_view.shape;
    workspace.decoder_reshape_view.shape.dims[3] += 1;
    passed = passed && rejection_is_pure(&model, &workspace);
    workspace.decoder_reshape_view.shape = saved_shape;

    workspace.owned_tensor_count += 1;
    passed = passed && rejection_is_pure(&model, &workspace);
    workspace.owned_tensor_count -= 1;
    workspace.logical_value_count += 1;
    passed = passed && rejection_is_pure(&model, &workspace);
    workspace.logical_value_count -= 1;
    workspace.storage_float_count += 1;
    passed = passed && rejection_is_pure(&model, &workspace);
    workspace.storage_float_count -= 1;
    workspace.storage_bytes += 1;
    passed = passed && rejection_is_pure(&model, &workspace);
    workspace.storage_bytes -= 1;

    saved_tensor = workspace.logits;
    workspace.logits = workspace.input;
    passed = passed && rejection_is_pure(&model, &workspace);
    workspace.logits = saved_tensor;

    saved_tensor = workspace.epsilon;
    workspace.epsilon.owned_block = workspace.input.grad;
    workspace.epsilon.data = workspace.input.grad;
    passed = passed && rejection_is_pure(&model, &workspace);
    workspace.epsilon = saved_tensor;
    passed = passed &&
        vae_c_model_workspace_validate(&model, &workspace) == VAE_C_OP_OK &&
        vae_c_model_workspace_is_allocated(&model, &workspace);

    saved_size = model.encoder_max_pool.capacity;
    model.encoder_max_pool.capacity =
        workspace.encoder_pool_output.shape.count - 1;
    passed = passed && rejection_is_pure(&model, &workspace);
    model.encoder_max_pool.capacity = saved_size;

    passed = passed &&
        vae_c_model_workspace_validate(&model, &workspace) == VAE_C_OP_OK;
    vae_c_model_workspace_free(&workspace);
    vae_c_model_free(&model);
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
        "workspace_lifecycle_basics", test_lifecycle_basics()
    );
    failures += report_check(
        "workspace_official_profiles", test_official_profiles(&catalog)
    );
    failures += report_check(
        "workspace_exact_official_shapes",
        test_exact_official_shapes(&catalog)
    );
    failures += report_check(
        "workspace_synthetic_depth4", test_synthetic_depth4()
    );
    failures += report_check(
        "workspace_batch_and_allocate_guarantee",
        test_batch_and_allocate_guarantee(&catalog)
    );
    failures += report_check(
        "workspace_reshape_view", test_reshape_view(&catalog)
    );
    failures += report_check(
        "workspace_zero_grad_atomic", test_zero_grad_atomic(&catalog)
    );
    failures += report_check(
        "workspace_backward_ticket_lifecycle",
        test_backward_ticket_lifecycle(&catalog)
    );
    failures += report_check(
        "workspace_validation_corruptions",
        test_validation_corruptions(&catalog)
    );

    vae_c_profile_catalog_free(&catalog);
    if (failures != 0) {
        fprintf(stderr, "%d VAE C WORKSPACE TEST(S) FAILED\n", failures);
        return 1;
    }
    printf("ALL VAE C MODEL WORKSPACE TESTS PASSED\n");
    return 0;
}
