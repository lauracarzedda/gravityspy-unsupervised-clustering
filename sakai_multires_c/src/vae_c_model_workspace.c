#include "vae_c_model_workspace.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>


static int size_multiply(size_t left, size_t right, size_t *result)
{
    if (result == NULL || (left != 0 && right > SIZE_MAX / left)) {
        return 1;
    }
    *result = left * right;
    return 0;
}


static int size_add(size_t left, size_t right, size_t *result)
{
    if (result == NULL || right > SIZE_MAX - left) {
        return 1;
    }
    *result = left + right;
    return 0;
}


static VAECOpStatus storage_to_op(VAECStorageStatus status)
{
    switch (status) {
        case VAE_C_STORAGE_OK:
            return VAE_C_OP_OK;
        case VAE_C_STORAGE_INVALID_ARGUMENT:
            return VAE_C_OP_INVALID_ARGUMENT;
        case VAE_C_STORAGE_OVERFLOW:
            return VAE_C_OP_OVERFLOW;
        case VAE_C_STORAGE_ALLOCATION_FAILED:
            return VAE_C_OP_ALLOCATION_FAILED;
        case VAE_C_STORAGE_OBJECT_NOT_EMPTY:
            return VAE_C_OP_OBJECT_NOT_EMPTY;
        case VAE_C_STORAGE_SHAPE_MISMATCH:
            return VAE_C_OP_SHAPE_MISMATCH;
        case VAE_C_STORAGE_GRADIENT_NOT_AVAILABLE:
            return VAE_C_OP_GRADIENT_NOT_AVAILABLE;
        default:
            return VAE_C_OP_INVALID_STATE;
    }
}


static bool bytes_are_zero(const void *object, size_t byte_count)
{
    const unsigned char *bytes = (const unsigned char *)object;
    size_t index;

    if (object == NULL) {
        return false;
    }
    for (index = 0; index < byte_count; index++) {
        if (bytes[index] != 0) {
            return false;
        }
    }
    return true;
}


static bool config_equal(const VAECConfig *left, const VAECConfig *right)
{
    size_t index;

    if (left == NULL || right == NULL ||
        left->input_channels != right->input_channels ||
        left->input_side != right->input_side ||
        left->conv_depth != right->conv_depth ||
        left->pool_side != right->pool_side ||
        left->latent_dim != right->latent_dim ||
        left->bottleneck_side != right->bottleneck_side ||
        left->decoder_fc_features != right->decoder_fc_features ||
        left->decoder_upsample_count != right->decoder_upsample_count ||
        left->decoder_hidden_conv_count !=
            right->decoder_hidden_conv_count ||
        left->values_per_sample != right->values_per_sample ||
        left->trainable_parameters_bn1d !=
            right->trainable_parameters_bn1d ||
        left->forward_mac_per_sample != right->forward_mac_per_sample) {

        return false;
    }
    for (index = 0; index < VAE_C_MAX_CONV_DEPTH; index++) {
        if (left->channels[index] != right->channels[index] ||
            left->encoder_conv_sides[index] !=
                right->encoder_conv_sides[index]) {

            return false;
        }
    }
    for (index = 0; index < VAE_C_MAX_DECODER_UPSAMPLES; index++) {
        if (left->decoder_upsample_targets[index] !=
            right->decoder_upsample_targets[index]) {

            return false;
        }
    }
    for (index = 0;
         index < VAE_C_MAX_DECODER_HIDDEN_CONVS;
         index++) {

        if (left->decoder_hidden_conv_sides[index] !=
            right->decoder_hidden_conv_sides[index]) {

            return false;
        }
    }
    return true;
}


static VAECOpStatus make_shape(
    size_t rank,
    const size_t *dims,
    VAECShape *shape
)
{
    return storage_to_op(vae_c_shape_make(rank, dims, shape));
}


static VAECOpStatus allocate_tensor(
    VAECTensor *tensor,
    size_t rank,
    const size_t *dims,
    bool with_gradient
)
{
    VAECShape shape;
    VAECOpStatus status = make_shape(rank, dims, &shape);

    if (status != VAE_C_OP_OK) {
        return status;
    }
    return storage_to_op(
        vae_c_tensor_allocate(tensor, &shape, with_gradient)
    );
}


static bool tensor_matches(
    const VAECTensor *tensor,
    size_t rank,
    const size_t *dims,
    bool with_gradient
)
{
    VAECShape expected;
    size_t arrays;
    size_t floats;
    size_t bytes;

    if (tensor == NULL ||
        vae_c_shape_make(rank, dims, &expected) != VAE_C_STORAGE_OK ||
        !vae_c_shape_equal(&tensor->shape, &expected) ||
        !tensor->owns_memory || tensor->owned_block == NULL ||
        tensor->data != tensor->owned_block ||
        tensor->has_grad != with_gradient) {

        return false;
    }
    if ((with_gradient &&
         tensor->grad != tensor->data + tensor->shape.count) ||
        (!with_gradient && tensor->grad != NULL)) {

        return false;
    }
    arrays = with_gradient ? 2 : 1;
    if (size_multiply(tensor->shape.count, arrays, &floats) != 0 ||
        size_multiply(floats, sizeof(float), &bytes) != 0) {

        return false;
    }
    return tensor->owned_bytes == bytes;
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


void vae_c_model_workspace_init(VAECModelWorkspace *workspace)
{
    if (workspace != NULL) {
        memset(workspace, 0, sizeof(*workspace));
    }
}


void vae_c_model_workspace_free(VAECModelWorkspace *workspace)
{
    size_t index;

    if (workspace == NULL) {
        return;
    }

    /* The reshape is non-owning, so discard it before its source. */
    vae_c_tensor_init(&workspace->decoder_reshape_view);
    vae_c_tensor_free(&workspace->input);
    for (index = 0; index < VAE_C_MODEL_MAX_ENCODER_CONVS; index++) {
        vae_c_tensor_free(&workspace->encoder_conv_outputs[index]);
        vae_c_tensor_free(
            &workspace->encoder_batch_norm_outputs[index]
        );
        vae_c_tensor_free(&workspace->encoder_relu_outputs[index]);
    }
    vae_c_tensor_free(&workspace->encoder_pool_output);
    vae_c_tensor_free(&workspace->encoder_global_avg_output);
    vae_c_tensor_free(&workspace->encoder_packed_output);
    vae_c_tensor_free(&workspace->mu);
    vae_c_tensor_free(&workspace->logvar);
    vae_c_tensor_free(&workspace->epsilon);
    vae_c_tensor_free(&workspace->z);
    vae_c_tensor_free(&workspace->decoder_fc_output);
    vae_c_tensor_free(&workspace->decoder_fc_batch_norm_output);
    vae_c_tensor_free(&workspace->decoder_fc_relu_output);
    for (index = 0; index < VAE_C_MAX_DECODER_UPSAMPLES; index++) {
        vae_c_tensor_free(&workspace->decoder_upsample_outputs[index]);
    }
    for (index = 0;
         index < VAE_C_MODEL_MAX_DECODER_HIDDEN_CONVS;
         index++) {

        vae_c_tensor_free(
            &workspace->decoder_hidden_conv_outputs[index]
        );
        vae_c_tensor_free(
            &workspace->decoder_hidden_batch_norm_outputs[index]
        );
        vae_c_tensor_free(
            &workspace->decoder_hidden_relu_outputs[index]
        );
    }
    vae_c_tensor_free(&workspace->logits);
    vae_c_model_workspace_init(workspace);
}


bool vae_c_model_workspace_is_empty(
    const VAECModelWorkspace *workspace
)
{
    size_t index;

    if (workspace == NULL ||
        !bytes_are_zero(workspace->profile_id,
                        sizeof(workspace->profile_id)) ||
        !bytes_are_zero(&workspace->config,
                        sizeof(workspace->config)) ||
        workspace->batch_size != 0 ||
        !vae_c_tensor_is_empty(&workspace->input) ||
        !vae_c_tensor_is_empty(&workspace->encoder_pool_output) ||
        !vae_c_tensor_is_empty(&workspace->encoder_global_avg_output) ||
        !vae_c_tensor_is_empty(&workspace->encoder_packed_output) ||
        !vae_c_tensor_is_empty(&workspace->mu) ||
        !vae_c_tensor_is_empty(&workspace->logvar) ||
        !vae_c_tensor_is_empty(&workspace->epsilon) ||
        !vae_c_tensor_is_empty(&workspace->z) ||
        !vae_c_tensor_is_empty(&workspace->decoder_fc_output) ||
        !vae_c_tensor_is_empty(
            &workspace->decoder_fc_batch_norm_output) ||
        !vae_c_tensor_is_empty(&workspace->decoder_fc_relu_output) ||
        !vae_c_tensor_is_empty(&workspace->decoder_reshape_view) ||
        !vae_c_tensor_is_empty(&workspace->logits) ||
        workspace->owned_tensor_count != 0 ||
        workspace->logical_value_count != 0 ||
        workspace->storage_float_count != 0 ||
        workspace->storage_bytes != 0 ||
        workspace->vae_backward_model != NULL ||
        workspace->vae_backward_generation != UINT64_C(0) ||
        workspace->allocated) {

        return false;
    }
    for (index = 0; index < VAE_C_MODEL_MAX_ENCODER_CONVS; index++) {
        if (!vae_c_tensor_is_empty(
                &workspace->encoder_conv_outputs[index]) ||
            !vae_c_tensor_is_empty(
                &workspace->encoder_batch_norm_outputs[index]) ||
            !vae_c_tensor_is_empty(
                &workspace->encoder_relu_outputs[index])) {

            return false;
        }
    }
    for (index = 0; index < VAE_C_MAX_DECODER_UPSAMPLES; index++) {
        if (!vae_c_tensor_is_empty(
                &workspace->decoder_upsample_outputs[index])) {

            return false;
        }
    }
    for (index = 0;
         index < VAE_C_MODEL_MAX_DECODER_HIDDEN_CONVS;
         index++) {

        if (!vae_c_tensor_is_empty(
                &workspace->decoder_hidden_conv_outputs[index]) ||
            !vae_c_tensor_is_empty(
                &workspace->decoder_hidden_batch_norm_outputs[index]) ||
            !vae_c_tensor_is_empty(
                &workspace->decoder_hidden_relu_outputs[index])) {

            return false;
        }
    }
    return true;
}


VAECOpStatus vae_c_model_workspace_validate(
    const VAECModel *model,
    const VAECModelWorkspace *workspace
)
{
    const VAECTensor *owned[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ];
    size_t dims[4];
    size_t depth;
    size_t index;
    size_t other;
    size_t owned_count;
    size_t expected_owned_count;
    size_t logical_count = 0;
    size_t storage_float_count = 0;
    size_t storage_bytes = 0;
    size_t expected_bytes;
    size_t decoder_features;
    size_t square;

    if (model == NULL || workspace == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (vae_c_model_validate(model) != VAE_C_OP_OK ||
        !workspace->allocated ||
        memcmp(workspace->profile_id,
               model->profile_id,
               sizeof(workspace->profile_id)) != 0 ||
        !config_equal(&workspace->config, &model->config) ||
        ((workspace->vae_backward_model == NULL) !=
         (workspace->vae_backward_generation == UINT64_C(0))) ||
        workspace->batch_size == 0 ||
        workspace->batch_size > model->max_batch_size) {

        return VAE_C_OP_INVALID_STATE;
    }

    depth = workspace->config.conv_depth;
    if (depth < 3 || depth > VAE_C_MAX_CONV_DEPTH ||
        workspace->config.decoder_upsample_count != depth + 1 ||
        workspace->config.decoder_hidden_conv_count != depth - 1 ||
        size_multiply(depth, 7, &expected_owned_count) != 0 ||
        size_add(expected_owned_count, 10, &expected_owned_count) != 0 ||
        expected_owned_count >
            VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS ||
        size_multiply(workspace->config.bottleneck_side,
                      workspace->config.bottleneck_side,
                      &square) != 0 ||
        size_multiply(workspace->config.channels[depth - 1],
                      square,
                      &decoder_features) != 0 ||
        decoder_features != workspace->config.decoder_fc_features) {

        return VAE_C_OP_INVALID_STATE;
    }

    dims[0] = workspace->batch_size;
    dims[1] = VAE_C_INPUT_CHANNELS;
    dims[2] = workspace->config.input_side;
    dims[3] = workspace->config.input_side;
    if (!tensor_matches(&workspace->input, 4, dims, true)) {
        return VAE_C_OP_INVALID_STATE;
    }

    for (index = 0; index < depth; index++) {
        dims[1] = workspace->config.channels[index];
        dims[2] = workspace->config.encoder_conv_sides[index];
        dims[3] = dims[2];
        if (!tensor_matches(
                &workspace->encoder_conv_outputs[index],
                4, dims, true) ||
            !tensor_matches(
                &workspace->encoder_batch_norm_outputs[index],
                4, dims, true) ||
            !tensor_matches(
                &workspace->encoder_relu_outputs[index],
                4, dims, true)) {

            return VAE_C_OP_INVALID_STATE;
        }
    }
    for (index = depth;
         index < VAE_C_MODEL_MAX_ENCODER_CONVS;
         index++) {

        if (!vae_c_tensor_is_empty(
                &workspace->encoder_conv_outputs[index]) ||
            !vae_c_tensor_is_empty(
                &workspace->encoder_batch_norm_outputs[index]) ||
            !vae_c_tensor_is_empty(
                &workspace->encoder_relu_outputs[index])) {

            return VAE_C_OP_INVALID_STATE;
        }
    }

    dims[1] = workspace->config.channels[0];
    dims[2] = workspace->config.pool_side;
    dims[3] = dims[2];
    if (!tensor_matches(
            &workspace->encoder_pool_output, 4, dims, true) ||
        workspace->encoder_pool_output.shape.count >
            model->encoder_max_pool.capacity) {

        return VAE_C_OP_INVALID_STATE;
    }
    dims[1] = workspace->config.channels[depth - 1];
    if (!tensor_matches(
            &workspace->encoder_global_avg_output, 2, dims, true)) {

        return VAE_C_OP_INVALID_STATE;
    }
    if (workspace->config.latent_dim > SIZE_MAX / 2) {
        return VAE_C_OP_OVERFLOW;
    }
    dims[1] = workspace->config.latent_dim * 2;
    if (!tensor_matches(
            &workspace->encoder_packed_output, 2, dims, true)) {

        return VAE_C_OP_INVALID_STATE;
    }
    dims[1] = workspace->config.latent_dim;
    if (!tensor_matches(&workspace->mu, 2, dims, true) ||
        !tensor_matches(&workspace->logvar, 2, dims, true) ||
        !tensor_matches(&workspace->epsilon, 2, dims, false) ||
        !tensor_matches(&workspace->z, 2, dims, true)) {

        return VAE_C_OP_INVALID_STATE;
    }

    dims[1] = workspace->config.decoder_fc_features;
    if (!tensor_matches(&workspace->decoder_fc_output, 2, dims, true) ||
        !tensor_matches(
            &workspace->decoder_fc_batch_norm_output,
            2, dims, true) ||
        !tensor_matches(
            &workspace->decoder_fc_relu_output,
            2, dims, true)) {

        return VAE_C_OP_INVALID_STATE;
    }
    dims[1] = workspace->config.channels[depth - 1];
    dims[2] = workspace->config.bottleneck_side;
    dims[3] = dims[2];
    {
        VAECShape expected_view;

        if (vae_c_shape_make(4, dims, &expected_view) !=
                VAE_C_STORAGE_OK ||
            !vae_c_shape_equal(
                &workspace->decoder_reshape_view.shape,
                &expected_view) ||
            workspace->decoder_reshape_view.data !=
                workspace->decoder_fc_relu_output.data ||
            workspace->decoder_reshape_view.grad !=
                workspace->decoder_fc_relu_output.grad ||
            workspace->decoder_reshape_view.owns_memory ||
            workspace->decoder_reshape_view.owned_block != NULL ||
            workspace->decoder_reshape_view.owned_bytes != 0 ||
            !workspace->decoder_reshape_view.has_grad) {

            return VAE_C_OP_INVALID_STATE;
        }
    }

    for (index = 0;
         index < workspace->config.decoder_upsample_count;
         index++) {

        dims[1] = index < 2
            ? workspace->config.channels[depth - 1]
            : workspace->config.channels[depth - index];
        dims[2] = workspace->config.decoder_upsample_targets[index];
        dims[3] = dims[2];
        if (!tensor_matches(
                &workspace->decoder_upsample_outputs[index],
                4, dims, true)) {

            return VAE_C_OP_INVALID_STATE;
        }
    }
    for (index = workspace->config.decoder_upsample_count;
         index < VAE_C_MAX_DECODER_UPSAMPLES;
         index++) {

        if (!vae_c_tensor_is_empty(
                &workspace->decoder_upsample_outputs[index])) {

            return VAE_C_OP_INVALID_STATE;
        }
    }

    for (index = 0;
         index < workspace->config.decoder_hidden_conv_count;
         index++) {

        dims[1] = workspace->config.channels[depth - 2 - index];
        dims[2] = workspace->config.decoder_hidden_conv_sides[index];
        dims[3] = dims[2];
        if (!tensor_matches(
                &workspace->decoder_hidden_conv_outputs[index],
                4, dims, true) ||
            !tensor_matches(
                &workspace->decoder_hidden_batch_norm_outputs[index],
                4, dims, true) ||
            !tensor_matches(
                &workspace->decoder_hidden_relu_outputs[index],
                4, dims, true)) {

            return VAE_C_OP_INVALID_STATE;
        }
    }
    for (index = workspace->config.decoder_hidden_conv_count;
         index < VAE_C_MODEL_MAX_DECODER_HIDDEN_CONVS;
         index++) {

        if (!vae_c_tensor_is_empty(
                &workspace->decoder_hidden_conv_outputs[index]) ||
            !vae_c_tensor_is_empty(
                &workspace->decoder_hidden_batch_norm_outputs[index]) ||
            !vae_c_tensor_is_empty(
                &workspace->decoder_hidden_relu_outputs[index])) {

            return VAE_C_OP_INVALID_STATE;
        }
    }

    dims[1] = VAE_C_INPUT_CHANNELS;
    dims[2] = workspace->config.input_side;
    dims[3] = dims[2];
    if (!tensor_matches(&workspace->logits, 4, dims, true) ||
        !vae_c_shape_equal(
            &workspace->input.shape, &workspace->logits.shape)) {

        return VAE_C_OP_INVALID_STATE;
    }

    owned_count = collect_owned_const(workspace, owned);
    if (owned_count != expected_owned_count ||
        workspace->owned_tensor_count != owned_count) {

        return VAE_C_OP_INVALID_STATE;
    }
    for (index = 0; index < owned_count; index++) {
        size_t tensor_floats;

        if (owned[index]->owned_block == NULL ||
            size_add(logical_count,
                     owned[index]->shape.count,
                     &logical_count) != 0 ||
            size_multiply(owned[index]->shape.count,
                          owned[index]->has_grad ? 2 : 1,
                          &tensor_floats) != 0 ||
            size_add(storage_float_count,
                     tensor_floats,
                     &storage_float_count) != 0 ||
            size_add(storage_bytes,
                     owned[index]->owned_bytes,
                     &storage_bytes) != 0) {

            return VAE_C_OP_INVALID_STATE;
        }
        for (other = 0; other < index; other++) {
            if (owned[index]->owned_block == owned[other]->owned_block ||
                owned[index]->data == owned[other]->data ||
                (owned[index]->grad != NULL &&
                 owned[other]->grad != NULL &&
                 owned[index]->grad == owned[other]->grad) ||
                (owned[index]->grad != NULL &&
                 owned[index]->grad == owned[other]->data) ||
                (owned[other]->grad != NULL &&
                 owned[index]->data == owned[other]->grad)) {

                return VAE_C_OP_INVALID_STATE;
            }
        }
    }
    if (size_multiply(storage_float_count,
                      sizeof(float),
                      &expected_bytes) != 0 ||
        workspace->logical_value_count != logical_count ||
        workspace->storage_float_count != storage_float_count ||
        workspace->storage_bytes != storage_bytes ||
        storage_bytes != expected_bytes) {

        return VAE_C_OP_INVALID_STATE;
    }
    return VAE_C_OP_OK;
}


bool vae_c_model_workspace_is_allocated(
    const VAECModel *model,
    const VAECModelWorkspace *workspace
)
{
    return vae_c_model_workspace_validate(model, workspace) ==
        VAE_C_OP_OK;
}


static VAECOpStatus calculate_counts(VAECModelWorkspace *workspace)
{
    const VAECTensor *owned[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ];
    size_t count;
    size_t index;
    size_t logical = 0;
    size_t storage_floats = 0;
    size_t storage_bytes = 0;

    count = collect_owned_const(workspace, owned);
    for (index = 0; index < count; index++) {
        size_t tensor_floats;

        if (size_add(logical,
                     owned[index]->shape.count,
                     &logical) != 0 ||
            size_multiply(owned[index]->shape.count,
                          owned[index]->has_grad ? 2 : 1,
                          &tensor_floats) != 0 ||
            size_add(storage_floats,
                     tensor_floats,
                     &storage_floats) != 0 ||
            size_add(storage_bytes,
                     owned[index]->owned_bytes,
                     &storage_bytes) != 0) {

            return VAE_C_OP_OVERFLOW;
        }
    }
    workspace->owned_tensor_count = count;
    workspace->logical_value_count = logical;
    workspace->storage_float_count = storage_floats;
    workspace->storage_bytes = storage_bytes;
    return VAE_C_OP_OK;
}


VAECOpStatus vae_c_model_workspace_allocate(
    const VAECModel *model,
    size_t batch_size,
    VAECModelWorkspace *workspace
)
{
    VAECModelWorkspace candidate;
    VAECOpStatus status;
    VAECShape view_shape;
    size_t dims[4];
    size_t depth;
    size_t index;

    if (model == NULL || workspace == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (!vae_c_model_workspace_is_empty(workspace)) {
        return VAE_C_OP_OBJECT_NOT_EMPTY;
    }
    if (vae_c_model_validate(model) != VAE_C_OP_OK ||
        batch_size == 0 || batch_size > model->max_batch_size) {

        return VAE_C_OP_INVALID_STATE;
    }

    vae_c_model_workspace_init(&candidate);
    memcpy(candidate.profile_id,
           model->profile_id,
           sizeof(candidate.profile_id));
    candidate.config = model->config;
    candidate.batch_size = batch_size;
    depth = candidate.config.conv_depth;
    dims[0] = batch_size;

    dims[1] = VAE_C_INPUT_CHANNELS;
    dims[2] = candidate.config.input_side;
    dims[3] = dims[2];
    status = allocate_tensor(&candidate.input, 4, dims, true);
    if (status != VAE_C_OP_OK) {
        goto fail;
    }
    for (index = 0; index < depth; index++) {
        dims[1] = candidate.config.channels[index];
        dims[2] = candidate.config.encoder_conv_sides[index];
        dims[3] = dims[2];
        status = allocate_tensor(
            &candidate.encoder_conv_outputs[index], 4, dims, true
        );
        if (status == VAE_C_OP_OK) {
            status = allocate_tensor(
                &candidate.encoder_batch_norm_outputs[index],
                4, dims, true
            );
        }
        if (status == VAE_C_OP_OK) {
            status = allocate_tensor(
                &candidate.encoder_relu_outputs[index], 4, dims, true
            );
        }
        if (status != VAE_C_OP_OK) {
            goto fail;
        }
    }

    dims[1] = candidate.config.channels[0];
    dims[2] = candidate.config.pool_side;
    dims[3] = dims[2];
    status = allocate_tensor(
        &candidate.encoder_pool_output, 4, dims, true
    );
    if (status != VAE_C_OP_OK) {
        goto fail;
    }
    dims[1] = candidate.config.channels[depth - 1];
    status = allocate_tensor(
        &candidate.encoder_global_avg_output, 2, dims, true
    );
    if (status != VAE_C_OP_OK) {
        goto fail;
    }
    if (candidate.config.latent_dim > SIZE_MAX / 2) {
        status = VAE_C_OP_OVERFLOW;
        goto fail;
    }
    dims[1] = candidate.config.latent_dim * 2;
    status = allocate_tensor(
        &candidate.encoder_packed_output, 2, dims, true
    );
    if (status != VAE_C_OP_OK) {
        goto fail;
    }
    dims[1] = candidate.config.latent_dim;
    status = allocate_tensor(&candidate.mu, 2, dims, true);
    if (status == VAE_C_OP_OK) {
        status = allocate_tensor(&candidate.logvar, 2, dims, true);
    }
    if (status == VAE_C_OP_OK) {
        status = allocate_tensor(&candidate.epsilon, 2, dims, false);
    }
    if (status == VAE_C_OP_OK) {
        status = allocate_tensor(&candidate.z, 2, dims, true);
    }
    if (status != VAE_C_OP_OK) {
        goto fail;
    }

    dims[1] = candidate.config.decoder_fc_features;
    status = allocate_tensor(
        &candidate.decoder_fc_output, 2, dims, true
    );
    if (status == VAE_C_OP_OK) {
        status = allocate_tensor(
            &candidate.decoder_fc_batch_norm_output, 2, dims, true
        );
    }
    if (status == VAE_C_OP_OK) {
        status = allocate_tensor(
            &candidate.decoder_fc_relu_output, 2, dims, true
        );
    }
    if (status != VAE_C_OP_OK) {
        goto fail;
    }
    dims[1] = candidate.config.channels[depth - 1];
    dims[2] = candidate.config.bottleneck_side;
    dims[3] = dims[2];
    status = make_shape(4, dims, &view_shape);
    if (status == VAE_C_OP_OK) {
        status = storage_to_op(vae_c_tensor_make_contiguous_view(
            &candidate.decoder_fc_relu_output,
            0,
            &view_shape,
            &candidate.decoder_reshape_view
        ));
    }
    if (status != VAE_C_OP_OK) {
        goto fail;
    }

    for (index = 0;
         index < candidate.config.decoder_upsample_count;
         index++) {

        dims[1] = index < 2
            ? candidate.config.channels[depth - 1]
            : candidate.config.channels[depth - index];
        dims[2] = candidate.config.decoder_upsample_targets[index];
        dims[3] = dims[2];
        status = allocate_tensor(
            &candidate.decoder_upsample_outputs[index],
            4, dims, true
        );
        if (status != VAE_C_OP_OK) {
            goto fail;
        }
    }
    for (index = 0;
         index < candidate.config.decoder_hidden_conv_count;
         index++) {

        dims[1] = candidate.config.channels[depth - 2 - index];
        dims[2] = candidate.config.decoder_hidden_conv_sides[index];
        dims[3] = dims[2];
        status = allocate_tensor(
            &candidate.decoder_hidden_conv_outputs[index],
            4, dims, true
        );
        if (status == VAE_C_OP_OK) {
            status = allocate_tensor(
                &candidate.decoder_hidden_batch_norm_outputs[index],
                4, dims, true
            );
        }
        if (status == VAE_C_OP_OK) {
            status = allocate_tensor(
                &candidate.decoder_hidden_relu_outputs[index],
                4, dims, true
            );
        }
        if (status != VAE_C_OP_OK) {
            goto fail;
        }
    }
    dims[1] = VAE_C_INPUT_CHANNELS;
    dims[2] = candidate.config.input_side;
    dims[3] = dims[2];
    status = allocate_tensor(&candidate.logits, 4, dims, true);
    if (status != VAE_C_OP_OK) {
        goto fail;
    }

    status = calculate_counts(&candidate);
    if (status != VAE_C_OP_OK) {
        goto fail;
    }
    candidate.allocated = true;
    status = vae_c_model_workspace_validate(model, &candidate);
    if (status != VAE_C_OP_OK) {
        goto fail;
    }

    *workspace = candidate;
    vae_c_model_workspace_init(&candidate);
    return VAE_C_OP_OK;

fail:
    vae_c_model_workspace_free(&candidate);
    return status;
}


static void workspace_zero_tensor_grad_commit(VAECTensor *tensor)
{
    size_t index;

    for (index = 0; index < tensor->shape.count; index++) {
        tensor->grad[index] = 0.0f;
    }
}


VAECOpStatus vae_c_model_workspace_zero_grad(
    const VAECModel *model,
    VAECModelWorkspace *workspace
)
{
    VAECTensor *owned[VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS];
    size_t count;
    size_t index;

    if (workspace == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (vae_c_model_workspace_is_empty(workspace)) {
        return VAE_C_OP_OK;
    }
    if (vae_c_model_workspace_validate(model, workspace) !=
            VAE_C_OP_OK) {

        return model == NULL
            ? VAE_C_OP_INVALID_ARGUMENT
            : VAE_C_OP_INVALID_STATE;
    }

    /* Full validation above makes the following write phase atomic. */
    count = collect_owned_mutable(workspace, owned);
    for (index = 0; index < count; index++) {
        if (owned[index]->has_grad) {
            workspace_zero_tensor_grad_commit(owned[index]);
        }
    }
    return VAE_C_OP_OK;
}
