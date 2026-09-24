#include "vae_c_model_forward.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define TEST_MAX_BLOCKS \
    (VAE_C_MODEL_MAX_PARAMETERS + \
     VAE_C_MAX_CONV_DEPTH + 1 + \
     VAE_C_MAX_DECODER_HIDDEN_CONVS + 1 + \
     3 * VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS)


typedef struct {
    const void *pointer;
    size_t byte_count;
    unsigned char *copy;
    char label[128];
} BlockSnapshot;


typedef struct {
    VAECModel model_metadata;
    VAECModelWorkspace workspace_metadata;
    BlockSnapshot blocks[TEST_MAX_BLOCKS];
    size_t block_count;
    bool captured;
} FullSnapshot;


typedef struct {
    float *running_mean;
    float *running_variance;
    float *batch_mean;
    float *batch_inv_std;
    float *work_mean;
    float *work_inv_std;
    float *work_unbiased_variance;
    float *owned_block;
} BatchNormPointers;


typedef struct {
    char message[256];
} ComparisonDiagnostic;


static void comparison_diagnostic_clear(ComparisonDiagnostic *diagnostic)
{
    if (diagnostic != NULL) {
        diagnostic->message[0] = '\0';
    }
}


static int comparison_difference(
    ComparisonDiagnostic *diagnostic,
    const char *path,
    const char *field
)
{
    if (diagnostic != NULL) {
        (void)snprintf(
            diagnostic->message,
            sizeof(diagnostic->message),
            "%s.%s differs",
            path,
            field
        );
    }
    return 0;
}


static void indexed_path(
    char *destination,
    size_t capacity,
    const char *category,
    size_t index
)
{
    if (destination != NULL && capacity != 0) {
        (void)snprintf(
            destination, capacity, "%s[%zu]", category, index
        );
    }
}


static int report_check(const char *name, int passed)
{
    if (passed) {
        printf("[PASS] %s\n", name);
        return 0;
    }
    fprintf(stderr, "[FAIL] %s\n", name);
    return 1;
}


static int size_multiply(size_t left, size_t right, size_t *result)
{
    if (result == NULL || (left != 0 && right > SIZE_MAX / left)) {
        return 0;
    }
    *result = left * right;
    return 1;
}


static int shape_equal(const VAECShape *left, const VAECShape *right)
{
    size_t index;

    if (left == NULL || right == NULL ||
        left->rank != right->rank || left->count != right->count) {

        return 0;
    }
    for (index = 0; index < VAE_C_TENSOR_MAX_RANK; index++) {
        if (left->dims[index] != right->dims[index] ||
            left->strides[index] != right->strides[index]) {

            return 0;
        }
    }
    return 1;
}


static int parameter_equal(
    const VAECParameter *left,
    const VAECParameter *right
)
{
    return left != NULL && right != NULL &&
        left->parameter_key == right->parameter_key &&
        shape_equal(&left->shape, &right->shape) &&
        left->owned_bytes == right->owned_bytes &&
        left->owned_block != NULL && right->owned_block != NULL &&
        memcmp(left->owned_block,
               right->owned_block,
               left->owned_bytes) == 0;
}


static const VAECParameter *test_parameter_by_key(
    const VAECModel *model,
    uint64_t key
)
{
    size_t index;
    size_t offset;

    if (model == NULL) {
        return NULL;
    }
    if (key >= UINT64_C(1) && key <= UINT64_C(12)) {
        index = (size_t)((key - UINT64_C(1)) / UINT64_C(3));
        offset = (size_t)((key - UINT64_C(1)) % UINT64_C(3));
        if (offset == 0) {
            return &model->encoder_convs[index].weights;
        }
        return offset == 1
            ? &model->encoder_batch_norms[index].gamma
            : &model->encoder_batch_norms[index].beta;
    }
    switch (key) {
    case VAE_C_MODEL_PARAM_KEY_ENCODER_FC_WEIGHT:
        return &model->encoder_fc.weights;
    case VAE_C_MODEL_PARAM_KEY_ENCODER_FC_BIAS:
        return &model->encoder_fc.bias;
    case VAE_C_MODEL_PARAM_KEY_DECODER_FC_WEIGHT:
        return &model->decoder_fc.weights;
    case VAE_C_MODEL_PARAM_KEY_DECODER_FC_BN_GAMMA:
        return &model->decoder_fc_batch_norm.gamma;
    case VAE_C_MODEL_PARAM_KEY_DECODER_FC_BN_BETA:
        return &model->decoder_fc_batch_norm.beta;
    case VAE_C_MODEL_PARAM_KEY_DECODER_OUTPUT_WEIGHT:
        return &model->decoder_output_conv.weights;
    case VAE_C_MODEL_PARAM_KEY_DECODER_OUTPUT_BIAS:
        return &model->decoder_output_conv.bias;
    default:
        break;
    }
    if (key >= UINT64_C(18) && key <= UINT64_C(26)) {
        index = (size_t)((key - UINT64_C(18)) / UINT64_C(3));
        offset = (size_t)((key - UINT64_C(18)) % UINT64_C(3));
        if (offset == 0) {
            return &model->decoder_hidden_convs[index].weights;
        }
        return offset == 1
            ? &model->decoder_hidden_batch_norms[index].gamma
            : &model->decoder_hidden_batch_norms[index].beta;
    }
    return NULL;
}


static size_t test_expected_parameter_keys(
    const VAECModel *model,
    uint64_t keys[VAE_C_MODEL_MAX_PARAMETERS]
)
{
    size_t count = 0;
    size_t index;

    if (model == NULL || keys == NULL ||
        (model->config.conv_depth != 3 && model->config.conv_depth != 4) ||
        model->config.decoder_hidden_conv_count !=
            model->config.conv_depth - 1) {

        return 0;
    }
    for (index = 0; index < model->config.conv_depth; index++) {
        keys[count++] = UINT64_C(1) + UINT64_C(3) * (uint64_t)index;
        keys[count++] = UINT64_C(2) + UINT64_C(3) * (uint64_t)index;
        keys[count++] = UINT64_C(3) + UINT64_C(3) * (uint64_t)index;
    }
    keys[count++] = VAE_C_MODEL_PARAM_KEY_ENCODER_FC_WEIGHT;
    keys[count++] = VAE_C_MODEL_PARAM_KEY_ENCODER_FC_BIAS;
    keys[count++] = VAE_C_MODEL_PARAM_KEY_DECODER_FC_WEIGHT;
    keys[count++] = VAE_C_MODEL_PARAM_KEY_DECODER_FC_BN_GAMMA;
    keys[count++] = VAE_C_MODEL_PARAM_KEY_DECODER_FC_BN_BETA;
    for (index = 0; index < model->config.decoder_hidden_conv_count; index++) {
        keys[count++] = UINT64_C(18) + UINT64_C(3) * (uint64_t)index;
        keys[count++] = UINT64_C(19) + UINT64_C(3) * (uint64_t)index;
        keys[count++] = UINT64_C(20) + UINT64_C(3) * (uint64_t)index;
    }
    keys[count++] = VAE_C_MODEL_PARAM_KEY_DECODER_OUTPUT_WEIGHT;
    keys[count++] = VAE_C_MODEL_PARAM_KEY_DECODER_OUTPUT_BIAS;
    return count;
}


static int independent_registries_equal(
    const VAECModel *left,
    const VAECModel *right,
    ComparisonDiagnostic *diagnostic
)
{
    uint64_t left_keys[VAE_C_MODEL_MAX_PARAMETERS];
    uint64_t right_keys[VAE_C_MODEL_MAX_PARAMETERS];
    size_t left_count;
    size_t right_count;
    size_t index;
    char path[128];

    comparison_diagnostic_clear(diagnostic);
    if (left == NULL || right == NULL) {
        return comparison_difference(diagnostic, "model.registry", "object");
    }
    left_count = test_expected_parameter_keys(left, left_keys);
    right_count = test_expected_parameter_keys(right, right_keys);
    if (left_count == 0 || right_count == 0 ||
        left_count != right_count ||
        left->parameter_count != left_count ||
        right->parameter_count != right_count) {

        return comparison_difference(
            diagnostic, "model.registry", "parameter_count"
        );
    }
    for (index = 0; index < left_count; index++) {
        const VAECParameter *left_expected;
        const VAECParameter *right_expected;
        const VAECParameter *left_entry = left->parameters[index];
        const VAECParameter *right_entry = right->parameters[index];

        (void)snprintf(
            path,
            sizeof(path),
            "model.parameters[%zu]{key=%llu}",
            index,
            (unsigned long long)left_keys[index]
        );
        if (left_keys[index] != right_keys[index]) {
            return comparison_difference(diagnostic, path, "order");
        }
        left_expected = test_parameter_by_key(left, left_keys[index]);
        right_expected = test_parameter_by_key(right, right_keys[index]);
        if (left_entry == NULL || left_expected == NULL) {
            return comparison_difference(
                diagnostic, path, "left_non_null"
            );
        }
        if (right_entry == NULL || right_expected == NULL) {
            return comparison_difference(
                diagnostic, path, "right_non_null"
            );
        }
        if (left_entry != left_expected) {
            return comparison_difference(
                diagnostic, path, "left_self_pointer"
            );
        }
        if (right_entry != right_expected) {
            return comparison_difference(
                diagnostic, path, "right_self_pointer"
            );
        }
        if (left_entry->parameter_key != left_keys[index] ||
            right_entry->parameter_key != right_keys[index]) {

            return comparison_difference(diagnostic, path, "parameter_key");
        }
        if (left_entry == right_entry ||
            left_entry->owned_block == right_entry->owned_block) {

            return comparison_difference(
                diagnostic, path, "cross_model_storage_alias"
            );
        }
        if (!parameter_equal(left_entry, right_entry)) {
            return comparison_difference(diagnostic, path, "contents");
        }
    }
    for (index = left_count; index < VAE_C_MODEL_MAX_PARAMETERS; index++) {
        if (left->parameters[index] != NULL ||
            right->parameters[index] != NULL) {

            return comparison_difference(
                diagnostic, "model.registry", "unused_entry"
            );
        }
    }
    return 1;
}


static int tensor_metadata_equal(
    const VAECTensor *left,
    const VAECTensor *right
)
{
    return left != NULL && right != NULL &&
        shape_equal(&left->shape, &right->shape) &&
        left->owned_bytes == right->owned_bytes &&
        left->owns_memory == right->owns_memory &&
        left->has_grad == right->has_grad;
}


static int tensor_equal(
    const VAECTensor *left,
    const VAECTensor *right
)
{
    return tensor_metadata_equal(left, right) &&
        left->owned_block != NULL && right->owned_block != NULL &&
        memcmp(left->owned_block,
               right->owned_block,
               left->owned_bytes) == 0;
}


static int batch_norm_equal(
    const VAECBatchNorm *left,
    const VAECBatchNorm *right
)
{
    return left != NULL && right != NULL &&
        left->kind == right->kind &&
        left->num_features == right->num_features &&
        left->epsilon == right->epsilon &&
        left->momentum == right->momentum &&
        parameter_equal(&left->gamma, &right->gamma) &&
        parameter_equal(&left->beta, &right->beta) &&
        left->owned_bytes == right->owned_bytes &&
        left->owned_block != NULL && right->owned_block != NULL &&
        memcmp(left->owned_block,
               right->owned_block,
               left->owned_bytes) == 0 &&
        left->last_sample_count == right->last_sample_count &&
        shape_equal(&left->last_input_shape, &right->last_input_shape) &&
        shape_equal(&left->last_output_shape,
                    &right->last_output_shape) &&
        left->cache_valid == right->cache_valid;
}


static int shape_metadata_exact(
    const VAECShape *left,
    const VAECShape *right,
    ComparisonDiagnostic *diagnostic,
    const char *path
)
{
    size_t index;
    char field[48];

    if (left == NULL || right == NULL) {
        return comparison_difference(diagnostic, path, "object");
    }
    if (left->rank != right->rank) {
        return comparison_difference(diagnostic, path, "rank");
    }
    for (index = 0; index < VAE_C_TENSOR_MAX_RANK; index++) {
        if (left->dims[index] != right->dims[index]) {
            (void)snprintf(field, sizeof(field), "dims[%zu]", index);
            return comparison_difference(diagnostic, path, field);
        }
        if (left->strides[index] != right->strides[index]) {
            (void)snprintf(field, sizeof(field), "strides[%zu]", index);
            return comparison_difference(diagnostic, path, field);
        }
    }
    if (left->count != right->count) {
        return comparison_difference(diagnostic, path, "count");
    }
    return 1;
}


static int config_metadata_exact(
    const VAECConfig *left,
    const VAECConfig *right,
    ComparisonDiagnostic *diagnostic,
    const char *path
)
{
    size_t index;
    char field[64];

    if (left == NULL || right == NULL) {
        return comparison_difference(diagnostic, path, "object");
    }
#define CHECK_CONFIG_FIELD(name) \
    do { \
        if (left->name != right->name) { \
            return comparison_difference(diagnostic, path, #name); \
        } \
    } while (0)
    CHECK_CONFIG_FIELD(input_channels);
    CHECK_CONFIG_FIELD(input_side);
    CHECK_CONFIG_FIELD(conv_depth);
    CHECK_CONFIG_FIELD(pool_side);
    CHECK_CONFIG_FIELD(latent_dim);
    CHECK_CONFIG_FIELD(bottleneck_side);
    CHECK_CONFIG_FIELD(decoder_fc_features);
    CHECK_CONFIG_FIELD(decoder_upsample_count);
    CHECK_CONFIG_FIELD(decoder_hidden_conv_count);
    CHECK_CONFIG_FIELD(values_per_sample);
    CHECK_CONFIG_FIELD(trainable_parameters_bn1d);
    CHECK_CONFIG_FIELD(forward_mac_per_sample);
#undef CHECK_CONFIG_FIELD
    for (index = 0; index < VAE_C_MAX_CONV_DEPTH; index++) {
        if (left->channels[index] != right->channels[index]) {
            (void)snprintf(field, sizeof(field), "channels[%zu]", index);
            return comparison_difference(diagnostic, path, field);
        }
        if (left->encoder_conv_sides[index] !=
            right->encoder_conv_sides[index]) {

            (void)snprintf(
                field, sizeof(field), "encoder_conv_sides[%zu]", index
            );
            return comparison_difference(diagnostic, path, field);
        }
    }
    for (index = 0; index < VAE_C_MAX_DECODER_UPSAMPLES; index++) {
        if (left->decoder_upsample_targets[index] !=
            right->decoder_upsample_targets[index]) {

            (void)snprintf(
                field,
                sizeof(field),
                "decoder_upsample_targets[%zu]",
                index
            );
            return comparison_difference(diagnostic, path, field);
        }
    }
    for (index = 0; index < VAE_C_MAX_DECODER_HIDDEN_CONVS; index++) {
        if (left->decoder_hidden_conv_sides[index] !=
            right->decoder_hidden_conv_sides[index]) {

            (void)snprintf(
                field,
                sizeof(field),
                "decoder_hidden_conv_sides[%zu]",
                index
            );
            return comparison_difference(diagnostic, path, field);
        }
    }
    return 1;
}


static int parameter_metadata_exact(
    const VAECParameter *left,
    const VAECParameter *right,
    ComparisonDiagnostic *diagnostic,
    const char *path
)
{
    char shape_path[128];

    if (left == NULL || right == NULL) {
        return comparison_difference(diagnostic, path, "object");
    }
    (void)snprintf(shape_path, sizeof(shape_path), "%s.shape", path);
    if (!shape_metadata_exact(
            &left->shape, &right->shape, diagnostic, shape_path)) {

        return 0;
    }
#define CHECK_PARAMETER_FIELD(name) \
    do { \
        if (left->name != right->name) { \
            return comparison_difference(diagnostic, path, #name); \
        } \
    } while (0)
    CHECK_PARAMETER_FIELD(parameter_key);
    CHECK_PARAMETER_FIELD(values);
    CHECK_PARAMETER_FIELD(gradients);
    CHECK_PARAMETER_FIELD(adam_m);
    CHECK_PARAMETER_FIELD(adam_v);
    CHECK_PARAMETER_FIELD(owned_block);
    CHECK_PARAMETER_FIELD(owned_bytes);
#undef CHECK_PARAMETER_FIELD
    return 1;
}


static int dense_metadata_exact(
    const VAECDense *left,
    const VAECDense *right,
    ComparisonDiagnostic *diagnostic,
    const char *path
)
{
    char child_path[128];

    if (left == NULL || right == NULL) {
        return comparison_difference(diagnostic, path, "object");
    }
    if (left->in_features != right->in_features) {
        return comparison_difference(diagnostic, path, "in_features");
    }
    if (left->out_features != right->out_features) {
        return comparison_difference(diagnostic, path, "out_features");
    }
    if (left->with_bias != right->with_bias) {
        return comparison_difference(diagnostic, path, "with_bias");
    }
    (void)snprintf(child_path, sizeof(child_path), "%s.weights", path);
    if (!parameter_metadata_exact(
            &left->weights, &right->weights, diagnostic, child_path)) {

        return 0;
    }
    (void)snprintf(child_path, sizeof(child_path), "%s.bias", path);
    return parameter_metadata_exact(
        &left->bias, &right->bias, diagnostic, child_path
    );
}


static int conv_metadata_exact(
    const VAECConv2D *left,
    const VAECConv2D *right,
    ComparisonDiagnostic *diagnostic,
    const char *path
)
{
    char child_path[128];

    if (left == NULL || right == NULL) {
        return comparison_difference(diagnostic, path, "object");
    }
#define CHECK_CONV_FIELD(name) \
    do { \
        if (left->name != right->name) { \
            return comparison_difference(diagnostic, path, #name); \
        } \
    } while (0)
    CHECK_CONV_FIELD(in_channels);
    CHECK_CONV_FIELD(out_channels);
    CHECK_CONV_FIELD(kernel_size);
    CHECK_CONV_FIELD(stride);
    CHECK_CONV_FIELD(padding);
    CHECK_CONV_FIELD(with_bias);
#undef CHECK_CONV_FIELD
    (void)snprintf(child_path, sizeof(child_path), "%s.weights", path);
    if (!parameter_metadata_exact(
            &left->weights, &right->weights, diagnostic, child_path)) {

        return 0;
    }
    (void)snprintf(child_path, sizeof(child_path), "%s.bias", path);
    return parameter_metadata_exact(
        &left->bias, &right->bias, diagnostic, child_path
    );
}


static int batch_norm_metadata_base_exact(
    const VAECBatchNorm *left,
    const VAECBatchNorm *right,
    ComparisonDiagnostic *diagnostic,
    const char *path
)
{
    char child_path[128];

    if (left == NULL || right == NULL) {
        return comparison_difference(diagnostic, path, "object");
    }
#define CHECK_BATCH_NORM_FIELD(name) \
    do { \
        if (left->name != right->name) { \
            return comparison_difference(diagnostic, path, #name); \
        } \
    } while (0)
    CHECK_BATCH_NORM_FIELD(kind);
    CHECK_BATCH_NORM_FIELD(num_features);
    CHECK_BATCH_NORM_FIELD(epsilon);
    CHECK_BATCH_NORM_FIELD(momentum);
    (void)snprintf(child_path, sizeof(child_path), "%s.gamma", path);
    if (!parameter_metadata_exact(
            &left->gamma, &right->gamma, diagnostic, child_path)) {

        return 0;
    }
    (void)snprintf(child_path, sizeof(child_path), "%s.beta", path);
    if (!parameter_metadata_exact(
            &left->beta, &right->beta, diagnostic, child_path)) {

        return 0;
    }
    CHECK_BATCH_NORM_FIELD(running_mean);
    CHECK_BATCH_NORM_FIELD(running_variance);
    CHECK_BATCH_NORM_FIELD(batch_mean);
    CHECK_BATCH_NORM_FIELD(batch_inv_std);
    CHECK_BATCH_NORM_FIELD(work_mean);
    CHECK_BATCH_NORM_FIELD(work_inv_std);
    CHECK_BATCH_NORM_FIELD(work_unbiased_variance);
    CHECK_BATCH_NORM_FIELD(owned_block);
    CHECK_BATCH_NORM_FIELD(owned_bytes);
#undef CHECK_BATCH_NORM_FIELD
    return 1;
}


static int batch_norm_metadata_exact(
    const VAECBatchNorm *left,
    const VAECBatchNorm *right,
    ComparisonDiagnostic *diagnostic,
    const char *path
)
{
    char child_path[128];

    if (!batch_norm_metadata_base_exact(
            left, right, diagnostic, path)) {

        return 0;
    }
    if (left->last_sample_count != right->last_sample_count) {
        return comparison_difference(
            diagnostic, path, "last_sample_count"
        );
    }
    (void)snprintf(
        child_path, sizeof(child_path), "%s.last_input_shape", path
    );
    if (!shape_metadata_exact(
            &left->last_input_shape,
            &right->last_input_shape,
            diagnostic,
            child_path)) {

        return 0;
    }
    (void)snprintf(
        child_path, sizeof(child_path), "%s.last_output_shape", path
    );
    if (!shape_metadata_exact(
            &left->last_output_shape,
            &right->last_output_shape,
            diagnostic,
            child_path)) {

        return 0;
    }
    if (left->cache_valid != right->cache_valid) {
        return comparison_difference(diagnostic, path, "cache_valid");
    }
    return 1;
}


static int max_pool_metadata_exact(
    const VAECMaxPool2D *left,
    const VAECMaxPool2D *right,
    ComparisonDiagnostic *diagnostic,
    const char *path
)
{
    char child_path[128];

    if (left == NULL || right == NULL) {
        return comparison_difference(diagnostic, path, "object");
    }
#define CHECK_MAX_POOL_FIELD(name) \
    do { \
        if (left->name != right->name) { \
            return comparison_difference(diagnostic, path, #name); \
        } \
    } while (0)
    CHECK_MAX_POOL_FIELD(argmax_indices);
    CHECK_MAX_POOL_FIELD(capacity);
    CHECK_MAX_POOL_FIELD(owned_bytes);
    CHECK_MAX_POOL_FIELD(last_count);
    (void)snprintf(
        child_path, sizeof(child_path), "%s.last_input_shape", path
    );
    if (!shape_metadata_exact(
            &left->last_input_shape,
            &right->last_input_shape,
            diagnostic,
            child_path)) {

        return 0;
    }
    (void)snprintf(
        child_path, sizeof(child_path), "%s.last_output_shape", path
    );
    if (!shape_metadata_exact(
            &left->last_output_shape,
            &right->last_output_shape,
            diagnostic,
            child_path)) {

        return 0;
    }
    CHECK_MAX_POOL_FIELD(cache_valid);
#undef CHECK_MAX_POOL_FIELD
    return 1;
}


static int tensor_metadata_exact(
    const VAECTensor *left,
    const VAECTensor *right,
    ComparisonDiagnostic *diagnostic,
    const char *path
)
{
    char shape_path[128];

    if (left == NULL || right == NULL) {
        return comparison_difference(diagnostic, path, "object");
    }
    (void)snprintf(shape_path, sizeof(shape_path), "%s.shape", path);
    if (!shape_metadata_exact(
            &left->shape, &right->shape, diagnostic, shape_path)) {

        return 0;
    }
#define CHECK_TENSOR_FIELD(name) \
    do { \
        if (left->name != right->name) { \
            return comparison_difference(diagnostic, path, #name); \
        } \
    } while (0)
    CHECK_TENSOR_FIELD(data);
    CHECK_TENSOR_FIELD(grad);
    CHECK_TENSOR_FIELD(owned_block);
    CHECK_TENSOR_FIELD(owned_bytes);
    CHECK_TENSOR_FIELD(owns_memory);
    CHECK_TENSOR_FIELD(has_grad);
#undef CHECK_TENSOR_FIELD
    return 1;
}


static int model_metadata_compare(
    const VAECModel *left,
    const VAECModel *right,
    bool allow_decoder_batch_norm_cache,
    ComparisonDiagnostic *diagnostic
)
{
    size_t index;
    char path[128];

    comparison_diagnostic_clear(diagnostic);
    if (left == NULL || right == NULL) {
        return comparison_difference(diagnostic, "model", "object");
    }
    if (memcmp(
            left->profile_id,
            right->profile_id,
            VAE_C_PROFILE_TEXT_CAP) != 0) {

        return comparison_difference(diagnostic, "model", "profile_id");
    }
    if (!config_metadata_exact(
            &left->config, &right->config, diagnostic, "model.config")) {

        return 0;
    }
#define CHECK_MODEL_FIELD(name) \
    do { \
        if (left->name != right->name) { \
            return comparison_difference(diagnostic, "model", #name); \
        } \
    } while (0)
    CHECK_MODEL_FIELD(max_batch_size);
    if (!max_pool_metadata_exact(
            &left->encoder_max_pool,
            &right->encoder_max_pool,
            diagnostic,
            "model.encoder_max_pool")) {

        return 0;
    }
    if (!dense_metadata_exact(
            &left->encoder_fc,
            &right->encoder_fc,
            diagnostic,
            "model.encoder_fc") ||
        !dense_metadata_exact(
            &left->decoder_fc,
            &right->decoder_fc,
            diagnostic,
            "model.decoder_fc")) {

        return 0;
    }
    if (allow_decoder_batch_norm_cache) {
        if (!batch_norm_metadata_base_exact(
                &left->decoder_fc_batch_norm,
                &right->decoder_fc_batch_norm,
                diagnostic,
                "model.decoder_fc_batch_norm")) {

            return 0;
        }
    } else if (!batch_norm_metadata_exact(
                   &left->decoder_fc_batch_norm,
                   &right->decoder_fc_batch_norm,
                   diagnostic,
                   "model.decoder_fc_batch_norm")) {

        return 0;
    }
    if (!conv_metadata_exact(
            &left->decoder_output_conv,
            &right->decoder_output_conv,
            diagnostic,
            "model.decoder_output_conv")) {

        return 0;
    }
    CHECK_MODEL_FIELD(parameter_count);
    CHECK_MODEL_FIELD(trainable_value_count);
    CHECK_MODEL_FIELD(initialization_seed);
    CHECK_MODEL_FIELD(vae_cache_generation);
    CHECK_MODEL_FIELD(parameters_initialized);
    CHECK_MODEL_FIELD(allocated);
#undef CHECK_MODEL_FIELD
    for (index = 0; index < VAE_C_MODEL_MAX_ENCODER_CONVS; index++) {
        indexed_path(
            path, sizeof(path), "model.encoder_convs", index
        );
        if (!conv_metadata_exact(
                &left->encoder_convs[index],
                &right->encoder_convs[index],
                diagnostic,
                path)) {

            return 0;
        }
        indexed_path(
            path, sizeof(path), "model.encoder_batch_norms", index
        );
        if (!batch_norm_metadata_exact(
                &left->encoder_batch_norms[index],
                &right->encoder_batch_norms[index],
                diagnostic,
                path)) {

            return 0;
        }
    }
    for (index = 0;
         index < VAE_C_MODEL_MAX_DECODER_HIDDEN_CONVS;
         index++) {

        indexed_path(
            path, sizeof(path), "model.decoder_hidden_convs", index
        );
        if (!conv_metadata_exact(
                &left->decoder_hidden_convs[index],
                &right->decoder_hidden_convs[index],
                diagnostic,
                path)) {

            return 0;
        }
        indexed_path(
            path,
            sizeof(path),
            "model.decoder_hidden_batch_norms",
            index
        );
        if (allow_decoder_batch_norm_cache &&
            index < left->config.decoder_hidden_conv_count) {

            if (!batch_norm_metadata_base_exact(
                    &left->decoder_hidden_batch_norms[index],
                    &right->decoder_hidden_batch_norms[index],
                    diagnostic,
                    path)) {

                return 0;
            }
        } else if (!batch_norm_metadata_exact(
                       &left->decoder_hidden_batch_norms[index],
                       &right->decoder_hidden_batch_norms[index],
                       diagnostic,
                       path)) {

            return 0;
        }
    }
    for (index = 0; index < VAE_C_MODEL_MAX_PARAMETERS; index++) {
        uint64_t expected_key = right->parameters[index] == NULL
            ? UINT64_C(0)
            : right->parameters[index]->parameter_key;

        if (left->parameters[index] != right->parameters[index]) {
            (void)snprintf(
                path,
                sizeof(path),
                "model.parameters[%zu]{key=%llu}",
                index,
                (unsigned long long)expected_key
            );
            return comparison_difference(diagnostic, path, "pointer");
        }
    }
    return 1;
}


static int model_metadata_exact(
    const VAECModel *left,
    const VAECModel *right,
    ComparisonDiagnostic *diagnostic
)
{
    return model_metadata_compare(left, right, false, diagnostic);
}


static int workspace_metadata_compare(
    const VAECModelWorkspace *left,
    const VAECModelWorkspace *right,
    bool require_valid_reshape_relation,
    ComparisonDiagnostic *diagnostic
)
{
    size_t index;
    char path[128];

    comparison_diagnostic_clear(diagnostic);
    if (left == NULL || right == NULL) {
        return comparison_difference(diagnostic, "workspace", "object");
    }
    if (memcmp(
            left->profile_id,
            right->profile_id,
            VAE_C_PROFILE_TEXT_CAP) != 0) {

        return comparison_difference(
            diagnostic, "workspace", "profile_id"
        );
    }
    if (!config_metadata_exact(
            &left->config,
            &right->config,
            diagnostic,
            "workspace.config")) {

        return 0;
    }
#define CHECK_WORKSPACE_FIELD(name) \
    do { \
        if (left->name != right->name) { \
            return comparison_difference( \
                diagnostic, "workspace", #name \
            ); \
        } \
    } while (0)
    CHECK_WORKSPACE_FIELD(batch_size);
    if (!tensor_metadata_exact(
            &left->input, &right->input, diagnostic, "workspace.input")) {

        return 0;
    }
    for (index = 0; index < VAE_C_MODEL_MAX_ENCODER_CONVS; index++) {
        indexed_path(
            path, sizeof(path), "workspace.encoder_conv_outputs", index
        );
        if (!tensor_metadata_exact(
                &left->encoder_conv_outputs[index],
                &right->encoder_conv_outputs[index],
                diagnostic,
                path)) {

            return 0;
        }
        indexed_path(
            path,
            sizeof(path),
            "workspace.encoder_batch_norm_outputs",
            index
        );
        if (!tensor_metadata_exact(
                &left->encoder_batch_norm_outputs[index],
                &right->encoder_batch_norm_outputs[index],
                diagnostic,
                path)) {

            return 0;
        }
        indexed_path(
            path, sizeof(path), "workspace.encoder_relu_outputs", index
        );
        if (!tensor_metadata_exact(
                &left->encoder_relu_outputs[index],
                &right->encoder_relu_outputs[index],
                diagnostic,
                path)) {

            return 0;
        }
    }
#define CHECK_WORKSPACE_TENSOR(name) \
    do { \
        if (!tensor_metadata_exact( \
                &left->name, \
                &right->name, \
                diagnostic, \
                "workspace." #name)) { \
            return 0; \
        } \
    } while (0)
    CHECK_WORKSPACE_TENSOR(encoder_pool_output);
    CHECK_WORKSPACE_TENSOR(encoder_global_avg_output);
    CHECK_WORKSPACE_TENSOR(encoder_packed_output);
    CHECK_WORKSPACE_TENSOR(mu);
    CHECK_WORKSPACE_TENSOR(logvar);
    CHECK_WORKSPACE_TENSOR(epsilon);
    CHECK_WORKSPACE_TENSOR(z);
    CHECK_WORKSPACE_TENSOR(decoder_fc_output);
    CHECK_WORKSPACE_TENSOR(decoder_fc_batch_norm_output);
    CHECK_WORKSPACE_TENSOR(decoder_fc_relu_output);
    CHECK_WORKSPACE_TENSOR(decoder_reshape_view);
    for (index = 0; index < VAE_C_MAX_DECODER_UPSAMPLES; index++) {
        indexed_path(
            path, sizeof(path), "workspace.decoder_upsample_outputs", index
        );
        if (!tensor_metadata_exact(
                &left->decoder_upsample_outputs[index],
                &right->decoder_upsample_outputs[index],
                diagnostic,
                path)) {

            return 0;
        }
    }
    for (index = 0;
         index < VAE_C_MODEL_MAX_DECODER_HIDDEN_CONVS;
         index++) {

        indexed_path(
            path,
            sizeof(path),
            "workspace.decoder_hidden_conv_outputs",
            index
        );
        if (!tensor_metadata_exact(
                &left->decoder_hidden_conv_outputs[index],
                &right->decoder_hidden_conv_outputs[index],
                diagnostic,
                path)) {

            return 0;
        }
        indexed_path(
            path,
            sizeof(path),
            "workspace.decoder_hidden_batch_norm_outputs",
            index
        );
        if (!tensor_metadata_exact(
                &left->decoder_hidden_batch_norm_outputs[index],
                &right->decoder_hidden_batch_norm_outputs[index],
                diagnostic,
                path)) {

            return 0;
        }
        indexed_path(
            path,
            sizeof(path),
            "workspace.decoder_hidden_relu_outputs",
            index
        );
        if (!tensor_metadata_exact(
                &left->decoder_hidden_relu_outputs[index],
                &right->decoder_hidden_relu_outputs[index],
                diagnostic,
                path)) {

            return 0;
        }
    }
    CHECK_WORKSPACE_TENSOR(logits);
#undef CHECK_WORKSPACE_TENSOR
    CHECK_WORKSPACE_FIELD(owned_tensor_count);
    CHECK_WORKSPACE_FIELD(logical_value_count);
    CHECK_WORKSPACE_FIELD(storage_float_count);
    CHECK_WORKSPACE_FIELD(storage_bytes);
    CHECK_WORKSPACE_FIELD(vae_backward_model);
    CHECK_WORKSPACE_FIELD(vae_backward_generation);
    CHECK_WORKSPACE_FIELD(allocated);
#undef CHECK_WORKSPACE_FIELD
    if (!require_valid_reshape_relation) {
        return 1;
    }
    if (left->decoder_reshape_view.owns_memory ||
        left->decoder_reshape_view.owned_block != NULL ||
        left->decoder_reshape_view.owned_bytes != 0) {

        return comparison_difference(
            diagnostic,
            "workspace.decoder_reshape_view",
            "non_owning_relation"
        );
    }
    if (left->decoder_reshape_view.data !=
        left->decoder_fc_relu_output.data) {

        return comparison_difference(
            diagnostic,
            "workspace.decoder_reshape_view",
            "data_owner_alias"
        );
    }
    if (left->decoder_reshape_view.grad !=
        left->decoder_fc_relu_output.grad) {

        return comparison_difference(
            diagnostic,
            "workspace.decoder_reshape_view",
            "grad_owner_alias"
        );
    }
    if (right->decoder_reshape_view.owns_memory ||
        right->decoder_reshape_view.owned_block != NULL ||
        right->decoder_reshape_view.owned_bytes != 0 ||
        right->decoder_reshape_view.data !=
            right->decoder_fc_relu_output.data ||
        right->decoder_reshape_view.grad !=
            right->decoder_fc_relu_output.grad) {

        return comparison_difference(
            diagnostic,
            "workspace.decoder_reshape_view",
            "snapshot_owner_alias"
        );
    }
    return 1;
}


static int workspace_metadata_exact(
    const VAECModelWorkspace *left,
    const VAECModelWorkspace *right,
    ComparisonDiagnostic *diagnostic
)
{
    return workspace_metadata_compare(
        left, right, false, diagnostic
    );
}


static size_t collect_workspace_tensors(
    const VAECModelWorkspace *workspace,
    const VAECTensor **tensors
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


static size_t collect_workspace_tensors_fixed(
    const VAECModelWorkspace *workspace,
    const VAECTensor **tensors
)
{
    size_t count = 0;
    size_t index;

    tensors[count++] = &workspace->input;
    for (index = 0; index < VAE_C_MODEL_MAX_ENCODER_CONVS; index++) {
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
    for (index = 0; index < VAE_C_MAX_DECODER_UPSAMPLES; index++) {
        tensors[count++] = &workspace->decoder_upsample_outputs[index];
    }
    for (index = 0;
         index < VAE_C_MODEL_MAX_DECODER_HIDDEN_CONVS;
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


static void block_snapshot_free(BlockSnapshot *snapshot)
{
    if (snapshot != NULL) {
        free(snapshot->copy);
        memset(snapshot, 0, sizeof(*snapshot));
    }
}


static int snapshot_append(
    FullSnapshot *snapshot,
    const void *pointer,
    size_t byte_count,
    const char *label
)
{
    BlockSnapshot *block;

    if (snapshot == NULL || pointer == NULL || byte_count == 0 ||
        label == NULL || label[0] == '\0' ||
        snapshot->block_count >= TEST_MAX_BLOCKS) {

        return 0;
    }
    block = &snapshot->blocks[snapshot->block_count];
    block->copy = (unsigned char *)malloc(byte_count);
    if (block->copy == NULL) {
        return 0;
    }
    memcpy(block->copy, pointer, byte_count);
    block->pointer = pointer;
    block->byte_count = byte_count;
    (void)snprintf(block->label, sizeof(block->label), "%s", label);
    snapshot->block_count++;
    return 1;
}


static void full_snapshot_init(FullSnapshot *snapshot)
{
    if (snapshot != NULL) {
        memset(snapshot, 0, sizeof(*snapshot));
    }
}


static void full_snapshot_free(FullSnapshot *snapshot)
{
    size_t index;

    if (snapshot == NULL) {
        return;
    }
    for (index = 0; index < snapshot->block_count; index++) {
        block_snapshot_free(&snapshot->blocks[index]);
    }
    full_snapshot_init(snapshot);
}


static int full_snapshot_capture(
    const VAECModel *model,
    const VAECModelWorkspace *workspace,
    FullSnapshot *snapshot
)
{
    const VAECTensor *tensors[VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS];
    const VAECBatchNorm *batch_norms[
        VAE_C_MAX_CONV_DEPTH + 1 +
        VAE_C_MAX_DECODER_HIDDEN_CONVS
    ];
    FullSnapshot candidate;
    size_t count;
    size_t index;
    char label[128];

    if (model == NULL || workspace == NULL || snapshot == NULL ||
        snapshot->captured) {

        return 0;
    }
    full_snapshot_init(&candidate);
    /*
     * Intentional non-owning metadata snapshots: these structure values are
     * never validated or freed as independent owners.  Their pointed-to
     * storage is captured separately below and remains owned by the fixture.
     */
    candidate.model_metadata = *model;
    candidate.workspace_metadata = *workspace;
    for (index = 0; index < model->parameter_count; index++) {
        const VAECParameter *parameter = model->parameters[index];

        if (parameter != NULL) {
            (void)snprintf(
                label,
                sizeof(label),
                "model.parameters[%zu]{key=%llu}.owned_block",
                index,
                (unsigned long long)parameter->parameter_key
            );
        }
        if (parameter == NULL || !snapshot_append(
                &candidate,
                parameter->owned_block,
                parameter->owned_bytes,
                label)) {

            full_snapshot_free(&candidate);
            return 0;
        }
    }
    count = 0;
    for (index = 0; index < VAE_C_MAX_CONV_DEPTH; index++) {
        batch_norms[count++] = &model->encoder_batch_norms[index];
    }
    batch_norms[count++] = &model->decoder_fc_batch_norm;
    for (index = 0;
         index < VAE_C_MAX_DECODER_HIDDEN_CONVS;
         index++) {

        batch_norms[count++] = &model->decoder_hidden_batch_norms[index];
    }
    for (index = 0; index < count; index++) {
        if (index < VAE_C_MAX_CONV_DEPTH) {
            (void)snprintf(
                label,
                sizeof(label),
                "model.encoder_batch_norms[%zu].owned_block",
                index
            );
        } else if (index == VAE_C_MAX_CONV_DEPTH) {
            (void)snprintf(
                label,
                sizeof(label),
                "model.decoder_fc_batch_norm.owned_block"
            );
        } else {
            (void)snprintf(
                label,
                sizeof(label),
                "model.decoder_hidden_batch_norms[%zu].owned_block",
                index - VAE_C_MAX_CONV_DEPTH - 1
            );
        }
        if (batch_norms[index]->owned_block != NULL &&
            !snapshot_append(
                &candidate,
                batch_norms[index]->owned_block,
                batch_norms[index]->owned_bytes,
                label)) {

            full_snapshot_free(&candidate);
            return 0;
        }
    }
    if (!snapshot_append(
            &candidate,
            model->encoder_max_pool.argmax_indices,
            model->encoder_max_pool.owned_bytes,
            "model.encoder_max_pool.argmax_indices")) {

        full_snapshot_free(&candidate);
        return 0;
    }
    count = collect_workspace_tensors_fixed(workspace, tensors);
    for (index = 0; index < count; index++) {
        size_t data_bytes;

        if (!size_multiply(
                tensors[index]->shape.count,
                sizeof(float),
                &data_bytes)) {

            full_snapshot_free(&candidate);
            return 0;
        }
        (void)snprintf(
            label,
            sizeof(label),
            "workspace.owned_tensors[%zu].owned_block",
            index
        );
        if (tensors[index]->owned_block != NULL &&
            !snapshot_append(
                &candidate,
                tensors[index]->owned_block,
                tensors[index]->owned_bytes,
                label)) {

            full_snapshot_free(&candidate);
            return 0;
        }
        (void)snprintf(
            label,
            sizeof(label),
            "workspace.owned_tensors[%zu].data",
            index
        );
        if (data_bytes != 0 && !snapshot_append(
                &candidate,
                tensors[index]->data,
                data_bytes,
                label)) {

            full_snapshot_free(&candidate);
            return 0;
        }
        (void)snprintf(
            label,
            sizeof(label),
            "workspace.owned_tensors[%zu].grad",
            index
        );
        if (tensors[index]->has_grad && data_bytes != 0 &&
            !snapshot_append(
                &candidate,
                tensors[index]->grad,
                data_bytes,
                label)) {

            full_snapshot_free(&candidate);
            return 0;
        }
    }
    candidate.captured = true;
    *snapshot = candidate;
    full_snapshot_init(&candidate);
    return 1;
}


static const BlockSnapshot *find_snapshot(
    const FullSnapshot *snapshot,
    const void *pointer,
    size_t byte_count
)
{
    size_t index;

    if (snapshot == NULL || !snapshot->captured) {
        return NULL;
    }
    for (index = 0; index < snapshot->block_count; index++) {
        if (snapshot->blocks[index].pointer == pointer &&
            snapshot->blocks[index].byte_count == byte_count) {

            return &snapshot->blocks[index];
        }
    }
    return NULL;
}


static int block_matches(
    const FullSnapshot *snapshot,
    const void *pointer,
    size_t byte_count
)
{
    const BlockSnapshot *block =
        find_snapshot(snapshot, pointer, byte_count);

    return block != NULL &&
        memcmp(pointer, block->copy, byte_count) == 0;
}


static int full_snapshot_matches_with_diagnostic(
    const VAECModel *model,
    const VAECModelWorkspace *workspace,
    const FullSnapshot *snapshot,
    ComparisonDiagnostic *diagnostic
)
{
    size_t index;

    comparison_diagnostic_clear(diagnostic);
    if (model == NULL || workspace == NULL || snapshot == NULL ||
        !snapshot->captured) {

        return comparison_difference(
            diagnostic, "snapshot", "capture_state"
        );
    }
    if (!model_metadata_exact(
            model, &snapshot->model_metadata, diagnostic)) {

        return 0;
    }
    if (!workspace_metadata_exact(
            workspace, &snapshot->workspace_metadata, diagnostic)) {

        return 0;
    }
    for (index = 0; index < snapshot->block_count; index++) {
        const BlockSnapshot *block = &snapshot->blocks[index];

        if (memcmp(block->pointer, block->copy, block->byte_count) != 0) {
            return comparison_difference(
                diagnostic, block->label, "contents"
            );
        }
    }
    return 1;
}


static int full_snapshot_matches(
    const VAECModel *model,
    const VAECModelWorkspace *workspace,
    const FullSnapshot *snapshot
)
{
    ComparisonDiagnostic diagnostic;
    int passed = full_snapshot_matches_with_diagnostic(
        model, workspace, snapshot, &diagnostic
    );

    if (!passed) {
        fprintf(stderr, "snapshot mismatch: %s\n", diagnostic.message);
    }
    return passed;
}


static int call_rejected_without_mutation(
    VAECModel *model,
    VAECBatchNormMode mode,
    VAECModelWorkspace *workspace,
    VAECOpStatus expected
)
{
    FullSnapshot snapshot;
    VAECOpStatus status;
    int passed;

    full_snapshot_init(&snapshot);
    if (!full_snapshot_capture(model, workspace, &snapshot)) {
        return 0;
    }
    status = vae_c_model_decoder_forward(model, mode, workspace);
    passed = status == expected &&
        full_snapshot_matches(model, workspace, &snapshot);
    full_snapshot_free(&snapshot);
    return passed;
}


static void fill_z(VAECModelWorkspace *workspace)
{
    size_t index;

    for (index = 0; index < workspace->z.shape.count; index++) {
        int centered = (int)(index % 17u) - 8;

        workspace->z.data[index] =
            0.375f * (float)centered +
            (index % 11u == 0 ? 2.5f : 0.0f);
    }
}


static int values_finite(const float *values, size_t count)
{
    size_t index;

    if (values == NULL) {
        return 0;
    }
    for (index = 0; index < count; index++) {
        if (!isfinite(values[index])) {
            return 0;
        }
    }
    return 1;
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


static int allocate_fixture(
    const VAECProfile *profile,
    size_t max_batch_size,
    size_t batch_size,
    uint64_t seed,
    VAECModel *model,
    VAECModelWorkspace *workspace
)
{
    vae_c_model_init(model);
    vae_c_model_workspace_init(workspace);
    if (profile == NULL ||
        vae_c_model_allocate(model, profile, max_batch_size) !=
            VAE_C_OP_OK ||
        vae_c_model_initialize_parameters_v1(model, seed) !=
            VAE_C_OP_OK ||
        vae_c_model_workspace_allocate(
            model, batch_size, workspace) != VAE_C_OP_OK) {

        return 0;
    }
    fill_z(workspace);
    return vae_c_model_validate(model) == VAE_C_OP_OK &&
        vae_c_model_workspace_validate(model, workspace) == VAE_C_OP_OK;
}


static void free_fixture(
    VAECModel *model,
    VAECModelWorkspace *workspace
)
{
    vae_c_model_workspace_free(workspace);
    vae_c_model_free(model);
}


static int allocate_pair(
    const VAECProfile *profile,
    size_t batch_size,
    VAECModel *first_model,
    VAECModelWorkspace *first_workspace,
    VAECModel *second_model,
    VAECModelWorkspace *second_workspace
)
{
    const uint64_t seed = UINT64_C(0x123456789ABCDEF0);

    return allocate_fixture(
        profile, batch_size, batch_size, seed,
        first_model, first_workspace
    ) && allocate_fixture(
        profile, batch_size, batch_size, seed,
        second_model, second_workspace
    );
}


static VAECOpStatus manual_decoder_forward(
    VAECModel *model,
    VAECBatchNormMode mode,
    VAECModelWorkspace *workspace
)
{
    VAECOpStatus status;
    size_t index;

    status = vae_c_dense_forward(
        &model->decoder_fc,
        &workspace->z,
        &workspace->decoder_fc_output
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_batch_norm1d_forward(
        &model->decoder_fc_batch_norm,
        mode,
        &workspace->decoder_fc_output,
        &workspace->decoder_fc_batch_norm_output
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_relu_forward(
        &workspace->decoder_fc_batch_norm_output,
        &workspace->decoder_fc_relu_output
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_nearest_upsample2d_forward(
        &workspace->decoder_reshape_view,
        &workspace->decoder_upsample_outputs[0]
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    status = vae_c_nearest_upsample2d_forward(
        &workspace->decoder_upsample_outputs[0],
        &workspace->decoder_upsample_outputs[1]
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    for (index = 0;
         index < model->config.decoder_hidden_conv_count;
         index++) {

        status = vae_c_conv2d_forward(
            &model->decoder_hidden_convs[index],
            &workspace->decoder_upsample_outputs[index + 1],
            &workspace->decoder_hidden_conv_outputs[index]
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
        status = vae_c_batch_norm2d_forward(
            &model->decoder_hidden_batch_norms[index],
            mode,
            &workspace->decoder_hidden_conv_outputs[index],
            &workspace->decoder_hidden_batch_norm_outputs[index]
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
        status = vae_c_relu_forward(
            &workspace->decoder_hidden_batch_norm_outputs[index],
            &workspace->decoder_hidden_relu_outputs[index]
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
        status = vae_c_nearest_upsample2d_forward(
            &workspace->decoder_hidden_relu_outputs[index],
            &workspace->decoder_upsample_outputs[index + 2]
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
    }
    return vae_c_conv2d_forward(
        &model->decoder_output_conv,
        &workspace->decoder_upsample_outputs[
            model->config.decoder_upsample_count - 1
        ],
        &workspace->logits
    );
}


static int decoder_results_equal(
    const VAECModel *left_model,
    const VAECModelWorkspace *left_workspace,
    const VAECModel *right_model,
    const VAECModelWorkspace *right_workspace
)
{
    size_t index;

    if (!tensor_equal(&left_workspace->z, &right_workspace->z) ||
        !tensor_equal(&left_workspace->decoder_fc_output,
                      &right_workspace->decoder_fc_output) ||
        !tensor_equal(&left_workspace->decoder_fc_batch_norm_output,
                      &right_workspace->decoder_fc_batch_norm_output) ||
        !tensor_equal(&left_workspace->decoder_fc_relu_output,
                      &right_workspace->decoder_fc_relu_output) ||
        !tensor_equal(&left_workspace->logits,
                      &right_workspace->logits) ||
        !shape_equal(&left_workspace->decoder_reshape_view.shape,
                     &right_workspace->decoder_reshape_view.shape) ||
        left_workspace->decoder_reshape_view.owns_memory ||
        right_workspace->decoder_reshape_view.owns_memory ||
        left_workspace->decoder_reshape_view.data !=
            left_workspace->decoder_fc_relu_output.data ||
        right_workspace->decoder_reshape_view.data !=
            right_workspace->decoder_fc_relu_output.data ||
        left_workspace->decoder_reshape_view.grad !=
            left_workspace->decoder_fc_relu_output.grad ||
        right_workspace->decoder_reshape_view.grad !=
            right_workspace->decoder_fc_relu_output.grad ||
        !batch_norm_equal(&left_model->decoder_fc_batch_norm,
                          &right_model->decoder_fc_batch_norm)) {

        return 0;
    }
    for (index = 0;
         index < left_model->config.decoder_upsample_count;
         index++) {

        if (!tensor_equal(
                &left_workspace->decoder_upsample_outputs[index],
                &right_workspace->decoder_upsample_outputs[index])) {

            return 0;
        }
    }
    for (index = 0;
         index < left_model->config.decoder_hidden_conv_count;
         index++) {

        if (!tensor_equal(
                &left_workspace->decoder_hidden_conv_outputs[index],
                &right_workspace->decoder_hidden_conv_outputs[index]) ||
            !tensor_equal(
                &left_workspace->decoder_hidden_batch_norm_outputs[index],
                &right_workspace->decoder_hidden_batch_norm_outputs[index]) ||
            !tensor_equal(
                &left_workspace->decoder_hidden_relu_outputs[index],
                &right_workspace->decoder_hidden_relu_outputs[index]) ||
            !batch_norm_equal(
                &left_model->decoder_hidden_batch_norms[index],
                &right_model->decoder_hidden_batch_norms[index])) {

            return 0;
        }
    }
    {
        ComparisonDiagnostic diagnostic;

        return independent_registries_equal(
            left_model, right_model, &diagnostic
        );
    }
}


static int run_manual_equivalence(
    const VAECProfile *profile,
    size_t batch_size,
    VAECBatchNormMode mode
)
{
    VAECModel composite_model;
    VAECModel manual_model;
    VAECModelWorkspace composite_workspace;
    VAECModelWorkspace manual_workspace;
    int passed;

    vae_c_model_init(&composite_model);
    vae_c_model_init(&manual_model);
    vae_c_model_workspace_init(&composite_workspace);
    vae_c_model_workspace_init(&manual_workspace);
    if (!allocate_pair(
            profile,
            batch_size,
            &composite_model,
            &composite_workspace,
            &manual_model,
            &manual_workspace)) {

        free_fixture(&composite_model, &composite_workspace);
        free_fixture(&manual_model, &manual_workspace);
        return 0;
    }
    passed = vae_c_model_decoder_forward(
        &composite_model, mode, &composite_workspace
    ) == VAE_C_OP_OK && manual_decoder_forward(
        &manual_model, mode, &manual_workspace
    ) == VAE_C_OP_OK && decoder_results_equal(
        &composite_model,
        &composite_workspace,
        &manual_model,
        &manual_workspace
    );
    free_fixture(&composite_model, &composite_workspace);
    free_fixture(&manual_model, &manual_workspace);
    return passed;
}


static int test_manual_depth3(
    const VAECProfileCatalog *catalog
)
{
    const VAECProfile *mini75 = vae_c_profile_catalog_find(
        catalog, "mini75_smoke_v1"
    );

    return mini75 != NULL &&
        run_manual_equivalence(
            mini75, 2, VAE_C_BATCH_NORM_MODE_TRAIN) &&
        run_manual_equivalence(
            mini75, 2, VAE_C_BATCH_NORM_MODE_EVAL);
}


static int test_manual_depth4(void)
{
    VAECProfile depth4;

    make_synthetic_depth4(&depth4);
    return run_manual_equivalence(
        &depth4, 2, VAE_C_BATCH_NORM_MODE_TRAIN
    ) && run_manual_equivalence(
        &depth4, 1, VAE_C_BATCH_NORM_MODE_EVAL
    );
}


static int profile_mode_case(
    const VAECProfile *profile,
    size_t batch_size,
    VAECBatchNormMode mode
)
{
    VAECModel model;
    VAECModelWorkspace workspace;
    size_t index;
    int passed;

    vae_c_model_init(&model);
    vae_c_model_workspace_init(&workspace);
    if (!allocate_fixture(
            profile, batch_size, batch_size, UINT64_C(7123),
            &model, &workspace)) {

        free_fixture(&model, &workspace);
        return 0;
    }
    passed = vae_c_model_decoder_forward(
        &model, mode, &workspace
    ) == VAE_C_OP_OK && workspace.logits.shape.rank == 4 &&
        workspace.logits.shape.dims[0] == batch_size &&
        workspace.logits.shape.dims[1] == VAE_C_INPUT_CHANNELS &&
        workspace.logits.shape.dims[2] == profile->input_side &&
        workspace.logits.shape.dims[3] == profile->input_side &&
        values_finite(workspace.logits.data, workspace.logits.shape.count);
    for (index = 0;
         passed && index < model.config.decoder_hidden_conv_count;
         index++) {

        passed = workspace.decoder_hidden_conv_outputs[index].shape.dims[2]
                == model.config.decoder_hidden_conv_sides[index] &&
            workspace.decoder_upsample_outputs[index + 2].shape.dims[2]
                == model.config.decoder_upsample_targets[index + 2];
    }
    free_fixture(&model, &workspace);
    return passed;
}


static int test_official_profiles(
    const VAECProfileCatalog *catalog
)
{
    const char *ids[4] = {
        "mini75_smoke_v1",
        "mini75_experiment_v1",
        "mini112_primary_v1",
        "mini112_scaleup_v1"
    };
    size_t index;

    for (index = 0; index < 4; index++) {
        const VAECProfile *profile =
            vae_c_profile_catalog_find(catalog, ids[index]);

        if (profile == NULL ||
            !profile_mode_case(
                profile, 1, VAE_C_BATCH_NORM_MODE_EVAL) ||
            !profile_mode_case(
                profile, 2, VAE_C_BATCH_NORM_MODE_EVAL) ||
            !profile_mode_case(
                profile, 2, VAE_C_BATCH_NORM_MODE_TRAIN)) {

            return 0;
        }
    }
    return 1;
}


static int test_batch_one(const VAECProfileCatalog *catalog)
{
    const VAECProfile *profile = vae_c_profile_catalog_find(
        catalog, "mini75_smoke_v1"
    );
    VAECModel model;
    VAECModelWorkspace workspace;
    int passed;

    vae_c_model_init(&model);
    vae_c_model_workspace_init(&workspace);
    if (!allocate_fixture(
            profile, 2, 1, UINT64_C(912), &model, &workspace)) {

        free_fixture(&model, &workspace);
        return 0;
    }
    passed = call_rejected_without_mutation(
        &model,
        VAE_C_BATCH_NORM_MODE_TRAIN,
        &workspace,
        VAE_C_OP_INVALID_STATE
    ) && vae_c_model_decoder_forward(
        &model, VAE_C_BATCH_NORM_MODE_EVAL, &workspace
    ) == VAE_C_OP_OK;
    free_fixture(&model, &workspace);
    return passed;
}


static int test_z_contract(const VAECProfileCatalog *catalog)
{
    const VAECProfile *profile = vae_c_profile_catalog_find(
        catalog, "mini75_smoke_v1"
    );
    VAECModel model;
    VAECModelWorkspace workspace;
    unsigned char *z_copy;
    size_t z_bytes;
    float saved;
    size_t saved_rank;
    size_t saved_owned_bytes;
    size_t index;
    int passed = 1;

    vae_c_model_init(&model);
    vae_c_model_workspace_init(&workspace);
    if (!allocate_fixture(
            profile, 2, 2, UINT64_C(1122), &model, &workspace) ||
        !size_multiply(workspace.z.shape.count, sizeof(float), &z_bytes)) {

        free_fixture(&model, &workspace);
        return 0;
    }
    z_copy = (unsigned char *)malloc(workspace.z.owned_bytes);
    if (z_copy == NULL) {
        free_fixture(&model, &workspace);
        return 0;
    }
    memcpy(z_copy, workspace.z.owned_block, workspace.z.owned_bytes);
    passed = workspace.z.data[0] < 0.0f &&
        workspace.z.data[11] > 1.0f &&
        vae_c_model_decoder_forward(
            &model, VAE_C_BATCH_NORM_MODE_EVAL, &workspace) ==
            VAE_C_OP_OK &&
        memcmp(z_copy, workspace.z.owned_block,
               workspace.z.owned_bytes) == 0;

    saved = workspace.z.data[0];
    for (index = 0; passed && index < 3; index++) {
        workspace.z.data[0] = index == 0
            ? NAN
            : (index == 1 ? INFINITY : -INFINITY);
        passed = call_rejected_without_mutation(
            &model,
            VAE_C_BATCH_NORM_MODE_EVAL,
            &workspace,
            VAE_C_OP_INVALID_STATE
        );
    }
    workspace.z.data[0] = saved;
    saved_rank = workspace.z.shape.rank;
    workspace.z.shape.rank = 1;
    passed = passed && call_rejected_without_mutation(
        &model,
        VAE_C_BATCH_NORM_MODE_EVAL,
        &workspace,
        VAE_C_OP_INVALID_STATE
    );
    workspace.z.shape.rank = saved_rank;
    saved_owned_bytes = workspace.z.owned_bytes;
    workspace.z.owned_bytes = z_bytes;
    passed = passed && call_rejected_without_mutation(
        &model,
        VAE_C_BATCH_NORM_MODE_EVAL,
        &workspace,
        VAE_C_OP_INVALID_STATE
    );
    workspace.z.owned_bytes = saved_owned_bytes;
    passed = passed && vae_c_model_workspace_validate(
        &model, &workspace
    ) == VAE_C_OP_OK;
    free(z_copy);
    free_fixture(&model, &workspace);
    return passed;
}


static int test_linear_logits(const VAECProfileCatalog *catalog)
{
    const VAECProfile *profile = vae_c_profile_catalog_find(
        catalog, "mini75_smoke_v1"
    );
    const float biases[VAE_C_INPUT_CHANNELS] = {
        -2.0f, 0.0f, 2.0f, 4.0f
    };
    VAECModel model;
    VAECModelWorkspace workspace;
    size_t channel;
    size_t spatial;
    size_t sample;
    int passed;

    vae_c_model_init(&model);
    vae_c_model_workspace_init(&workspace);
    if (!allocate_fixture(
            profile, 2, 2, UINT64_C(31415), &model, &workspace)) {

        free_fixture(&model, &workspace);
        return 0;
    }
    memset(model.decoder_output_conv.weights.values,
           0,
           model.decoder_output_conv.weights.shape.count * sizeof(float));
    memcpy(model.decoder_output_conv.bias.values,
           biases,
           sizeof(biases));
    passed = vae_c_model_decoder_forward(
        &model, VAE_C_BATCH_NORM_MODE_EVAL, &workspace
    ) == VAE_C_OP_OK;
    spatial = profile->input_side * profile->input_side;
    for (sample = 0; passed && sample < 2; sample++) {
        for (channel = 0;
             passed && channel < VAE_C_INPUT_CHANNELS;
             channel++) {

            size_t index;
            size_t base =
                (sample * VAE_C_INPUT_CHANNELS + channel) * spatial;

            for (index = 0; index < spatial; index++) {
                if (workspace.logits.data[base + index] !=
                    biases[channel]) {

                    passed = 0;
                    break;
                }
            }
        }
    }
    free_fixture(&model, &workspace);
    return passed;
}


static int test_eval_determinism(const VAECProfileCatalog *catalog)
{
    const VAECProfile *profile = vae_c_profile_catalog_find(
        catalog, "mini75_smoke_v1"
    );
    VAECModel first_model;
    VAECModel second_model;
    VAECModelWorkspace first_workspace;
    VAECModelWorkspace second_workspace;
    int passed;

    vae_c_model_init(&first_model);
    vae_c_model_init(&second_model);
    vae_c_model_workspace_init(&first_workspace);
    vae_c_model_workspace_init(&second_workspace);
    if (!allocate_pair(
            profile, 2,
            &first_model, &first_workspace,
            &second_model, &second_workspace)) {

        free_fixture(&first_model, &first_workspace);
        free_fixture(&second_model, &second_workspace);
        return 0;
    }
    passed = vae_c_model_decoder_forward(
        &first_model, VAE_C_BATCH_NORM_MODE_EVAL, &first_workspace
    ) == VAE_C_OP_OK && vae_c_model_decoder_forward(
        &second_model, VAE_C_BATCH_NORM_MODE_EVAL, &second_workspace
    ) == VAE_C_OP_OK && decoder_results_equal(
        &first_model, &first_workspace, &second_model, &second_workspace
    );
    free_fixture(&first_model, &first_workspace);
    free_fixture(&second_model, &second_workspace);
    return passed;
}


static int tensor_is_decoder_output(
    const VAECModelWorkspace *workspace,
    const VAECTensor *tensor
)
{
    size_t index;

    if (tensor == &workspace->decoder_fc_output ||
        tensor == &workspace->decoder_fc_batch_norm_output ||
        tensor == &workspace->decoder_fc_relu_output ||
        tensor == &workspace->logits) {

        return 1;
    }
    for (index = 0;
         index < workspace->config.decoder_upsample_count;
         index++) {

        if (tensor == &workspace->decoder_upsample_outputs[index]) {
            return 1;
        }
    }
    for (index = 0;
         index < workspace->config.decoder_hidden_conv_count;
         index++) {

        if (tensor == &workspace->decoder_hidden_conv_outputs[index] ||
            tensor ==
                &workspace->decoder_hidden_batch_norm_outputs[index] ||
            tensor == &workspace->decoder_hidden_relu_outputs[index]) {

            return 1;
        }
    }
    return 0;
}


static int batch_norm_mutation_allowed(
    const VAECBatchNorm *current,
    const VAECBatchNorm *before,
    const FullSnapshot *snapshot,
    VAECBatchNormMode mode,
    ComparisonDiagnostic *diagnostic,
    const char *path
)
{
    const BlockSnapshot *block;
    size_t segment_bytes;
    size_t segment;
    size_t index;
    char field[64];

    if (!batch_norm_metadata_base_exact(
            current, before, diagnostic, path)) {

        return 0;
    }
    if (current->num_features == 0) {
        return comparison_difference(diagnostic, path, "num_features");
    }
    block = find_snapshot(
        snapshot, before->owned_block, before->owned_bytes
    );
    if (block == NULL) {
        return comparison_difference(diagnostic, path, "owned_snapshot");
    }
    if (mode == VAE_C_BATCH_NORM_MODE_TRAIN) {
        if (!current->cache_valid) {
            return comparison_difference(diagnostic, path, "cache_valid");
        }
        if (current->last_sample_count < 2) {
            return comparison_difference(
                diagnostic, path, "last_sample_count"
            );
        }
        if (!shape_equal(
                &current->last_input_shape,
                &current->last_output_shape)) {

            return comparison_difference(
                diagnostic, path, "last_input_output_shape_relation"
            );
        }
        return 1;
    }
    if (mode != VAE_C_BATCH_NORM_MODE_EVAL || current->cache_valid ||
        current->last_sample_count != 0 ||
        current->last_input_shape.rank != 0 ||
        current->last_input_shape.count != 0 ||
        current->last_output_shape.rank != 0 ||
        current->last_output_shape.count != 0) {

        return comparison_difference(
            diagnostic, path, "eval_cache_invalidation"
        );
    }
    for (index = 0; index < VAE_C_TENSOR_MAX_RANK; index++) {
        if (current->last_input_shape.dims[index] != 0 ||
            current->last_input_shape.strides[index] != 0 ||
            current->last_output_shape.dims[index] != 0 ||
            current->last_output_shape.strides[index] != 0) {

            (void)snprintf(
                field,
                sizeof(field),
                "eval_cache_shape_slot[%zu]",
                index
            );
            return comparison_difference(diagnostic, path, field);
        }
    }
    segment_bytes = current->num_features * sizeof(float);
    for (segment = 0; segment < 7; segment++) {
        if (segment != 5 && memcmp(
                (const unsigned char *)current->owned_block +
                    segment * segment_bytes,
                block->copy + segment * segment_bytes,
                segment_bytes) != 0) {

            (void)snprintf(
                field,
                sizeof(field),
                "owned_block_segment[%zu]",
                segment
            );
            return comparison_difference(diagnostic, path, field);
        }
    }
    return 1;
}


static int valid_mutation_boundary(
    const VAECProfile *profile,
    VAECBatchNormMode mode
)
{
    const VAECTensor *tensors[VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS];
    ComparisonDiagnostic diagnostic;
    FullSnapshot snapshot;
    VAECModel model;
    VAECModelWorkspace workspace;
    VAECModel expected_model;
    VAECModelWorkspace expected_workspace;
    size_t tensor_count;
    size_t index;
    char path[128];
    int passed = 1;

    full_snapshot_init(&snapshot);
    vae_c_model_init(&model);
    vae_c_model_workspace_init(&workspace);
    if (!allocate_fixture(
            profile, 2, 2, UINT64_C(271828), &model, &workspace)) {

        free_fixture(&model, &workspace);
        return 0;
    }
    for (index = 0; index < model.parameter_count; index++) {
        VAECParameter *parameter = model.parameters[index];
        size_t value;

        for (value = 0; value < parameter->shape.count; value++) {
            parameter->gradients[value] =
                0.01f * (float)((value % 7u) + 1u);
            parameter->adam_m[value] =
                -0.02f * (float)((value % 5u) + 1u);
            parameter->adam_v[value] =
                0.03f * (float)((value % 3u) + 1u);
        }
    }
    model.vae_cache_generation = UINT64_C(31);
    workspace.vae_backward_model = &model;
    workspace.vae_backward_generation = UINT64_C(31);
    tensor_count = collect_workspace_tensors(&workspace, tensors);
    for (index = 0; index < tensor_count; index++) {
        size_t value;

        if (tensor_is_decoder_output(&workspace, tensors[index])) {
            for (value = 0; value < tensors[index]->shape.count; value++) {
                tensors[index]->data[value] = NAN;
            }
        }
        if (tensors[index]->grad != NULL) {
            for (value = 0; value < tensors[index]->shape.count; value++) {
                tensors[index]->grad[value] =
                    0.125f + 0.001f * (float)(value % 19u);
            }
        }
    }
    if (!full_snapshot_capture(&model, &workspace, &snapshot) ||
        vae_c_model_decoder_forward(&model, mode, &workspace) !=
            VAE_C_OP_OK) {

        full_snapshot_free(&snapshot);
        free_fixture(&model, &workspace);
        return 0;
    }

    expected_model = snapshot.model_metadata;
    expected_model.vae_cache_generation += UINT64_C(1);
    expected_workspace = snapshot.workspace_metadata;
    expected_workspace.vae_backward_model = NULL;
    expected_workspace.vae_backward_generation = UINT64_C(0);
    comparison_diagnostic_clear(&diagnostic);
    passed = model_metadata_compare(
        &model, &expected_model, true, &diagnostic
    ) && workspace_metadata_compare(
        &workspace, &expected_workspace, true, &diagnostic
    );
    for (index = 0; passed && index < model.parameter_count; index++) {
        passed = block_matches(
            &snapshot,
            model.parameters[index]->owned_block,
            model.parameters[index]->owned_bytes
        );
        if (!passed) {
            (void)snprintf(
                path,
                sizeof(path),
                "model.parameters[%zu]{key=%llu}",
                index,
                (unsigned long long)model.parameters[index]->parameter_key
            );
            (void)comparison_difference(
                &diagnostic, path, "owned_block.contents"
            );
        }
    }
    for (index = 0; passed && index < model.config.conv_depth; index++) {
        passed = block_matches(
            &snapshot,
            model.encoder_batch_norms[index].owned_block,
            model.encoder_batch_norms[index].owned_bytes
        );
        if (!passed) {
            indexed_path(
                path,
                sizeof(path),
                "model.encoder_batch_norms",
                index
            );
            (void)comparison_difference(
                &diagnostic, path, "owned_block.contents"
            );
        }
    }
    if (passed && !block_matches(
            &snapshot,
            model.encoder_max_pool.argmax_indices,
            model.encoder_max_pool.owned_bytes)) {

        passed = comparison_difference(
            &diagnostic,
            "model.encoder_max_pool",
            "argmax_indices.contents"
        );
    }
    if (passed) {
        passed = batch_norm_mutation_allowed(
            &model.decoder_fc_batch_norm,
            &snapshot.model_metadata.decoder_fc_batch_norm,
            &snapshot,
            mode,
            &diagnostic,
            "model.decoder_fc_batch_norm"
        );
    }
    for (index = 0;
         passed && index < model.config.decoder_hidden_conv_count;
         index++) {

        indexed_path(
            path,
            sizeof(path),
            "model.decoder_hidden_batch_norms",
            index
        );
        passed = batch_norm_mutation_allowed(
            &model.decoder_hidden_batch_norms[index],
            &snapshot.model_metadata.decoder_hidden_batch_norms[index],
            &snapshot,
            mode,
            &diagnostic,
            path
        );
    }
    for (index = 0; passed && index < tensor_count; index++) {
        const VAECTensor *tensor = tensors[index];
        const BlockSnapshot *block = find_snapshot(
            &snapshot, tensor->owned_block, tensor->owned_bytes
        );
        size_t data_bytes = tensor->shape.count * sizeof(float);

        if (block == NULL) {
            (void)snprintf(
                path,
                sizeof(path),
                "workspace.active_tensors[%zu]",
                index
            );
            passed = comparison_difference(
                &diagnostic, path, "owned_snapshot"
            );
        } else if (tensor_is_decoder_output(&workspace, tensor)) {
            passed = tensor->has_grad &&
                values_finite(tensor->data, tensor->shape.count) &&
                memcmp(
                    (const unsigned char *)tensor->owned_block +
                        data_bytes,
                    block->copy + data_bytes,
                    data_bytes) == 0;
            if (!passed) {
                (void)snprintf(
                    path,
                    sizeof(path),
                    "workspace.active_tensors[%zu]",
                    index
                );
                (void)comparison_difference(
                    &diagnostic,
                    path,
                    "decoder_data_finite_or_grad_unchanged"
                );
            }
        } else {
            passed = memcmp(
                tensor->owned_block,
                block->copy,
                tensor->owned_bytes
            ) == 0;
            if (!passed) {
                (void)snprintf(
                    path,
                    sizeof(path),
                    "workspace.active_tensors[%zu]",
                    index
                );
                (void)comparison_difference(
                    &diagnostic, path, "owned_block.contents"
                );
            }
        }
    }
    if (!passed) {
        fprintf(
            stderr,
            "decoder mutation boundary mismatch (%s): %s\n",
            mode == VAE_C_BATCH_NORM_MODE_TRAIN ? "TRAIN" : "EVAL",
            diagnostic.message
        );
    }
    full_snapshot_free(&snapshot);
    free_fixture(&model, &workspace);
    return passed;
}


static int test_mutation_boundaries(
    const VAECProfileCatalog *catalog
)
{
    const VAECProfile *depth3 = vae_c_profile_catalog_find(
        catalog, "mini75_smoke_v1"
    );
    VAECProfile depth4;

    make_synthetic_depth4(&depth4);
    return depth3 != NULL && valid_mutation_boundary(
        depth3, VAE_C_BATCH_NORM_MODE_TRAIN
    ) && valid_mutation_boundary(
        depth3, VAE_C_BATCH_NORM_MODE_EVAL
    ) && valid_mutation_boundary(
        &depth4, VAE_C_BATCH_NORM_MODE_TRAIN
    ) && valid_mutation_boundary(
        &depth4, VAE_C_BATCH_NORM_MODE_EVAL
    );
}


static int diagnostic_contains(
    const ComparisonDiagnostic *diagnostic,
    const char *text
)
{
    return diagnostic != NULL && text != NULL &&
        strstr(diagnostic->message, text) != NULL;
}


static int independent_max_pools_equal(
    const VAECMaxPool2D *left,
    const VAECMaxPool2D *right,
    ComparisonDiagnostic *diagnostic
)
{
    if (left == NULL || right == NULL) {
        return comparison_difference(
            diagnostic, "model.encoder_max_pool", "object"
        );
    }
    if (left->capacity != right->capacity ||
        left->owned_bytes != right->owned_bytes ||
        left->last_count != right->last_count ||
        left->cache_valid != right->cache_valid ||
        !shape_equal(&left->last_input_shape, &right->last_input_shape) ||
        !shape_equal(&left->last_output_shape, &right->last_output_shape)) {

        return comparison_difference(
            diagnostic, "model.encoder_max_pool", "metadata"
        );
    }
    if (left->argmax_indices == NULL || right->argmax_indices == NULL ||
        left->argmax_indices == right->argmax_indices ||
        memcmp(left->argmax_indices,
               right->argmax_indices,
               left->owned_bytes) != 0) {

        return comparison_difference(
            diagnostic, "model.encoder_max_pool", "owned_storage"
        );
    }
    return 1;
}


static int independent_models_equal(
    const VAECModel *left,
    const VAECModel *right,
    ComparisonDiagnostic *diagnostic
)
{
    size_t index;
    char path[128];

    comparison_diagnostic_clear(diagnostic);
    if (left == NULL || right == NULL) {
        return comparison_difference(diagnostic, "model", "object");
    }
    if (memcmp(
            left->profile_id,
            right->profile_id,
            VAE_C_PROFILE_TEXT_CAP) != 0) {

        return comparison_difference(diagnostic, "model", "profile_id");
    }
    if (!config_metadata_exact(
            &left->config, &right->config, diagnostic, "model.config")) {

        return 0;
    }
#define CHECK_INDEPENDENT_MODEL_FIELD(name) \
    do { \
        if (left->name != right->name) { \
            return comparison_difference(diagnostic, "model", #name); \
        } \
    } while (0)
    CHECK_INDEPENDENT_MODEL_FIELD(max_batch_size);
    CHECK_INDEPENDENT_MODEL_FIELD(parameter_count);
    CHECK_INDEPENDENT_MODEL_FIELD(trainable_value_count);
    CHECK_INDEPENDENT_MODEL_FIELD(initialization_seed);
    CHECK_INDEPENDENT_MODEL_FIELD(vae_cache_generation);
    CHECK_INDEPENDENT_MODEL_FIELD(parameters_initialized);
    CHECK_INDEPENDENT_MODEL_FIELD(allocated);
#undef CHECK_INDEPENDENT_MODEL_FIELD
    if (!independent_max_pools_equal(
            &left->encoder_max_pool,
            &right->encoder_max_pool,
            diagnostic)) {

        return 0;
    }
    for (index = 0; index < left->config.conv_depth; index++) {
        (void)snprintf(
            path,
            sizeof(path),
            "model.encoder_batch_norms[%zu]",
            index
        );
        if (!batch_norm_equal(
                &left->encoder_batch_norms[index],
                &right->encoder_batch_norms[index])) {

            if (left->encoder_batch_norms[index].cache_valid !=
                right->encoder_batch_norms[index].cache_valid) {

                return comparison_difference(
                    diagnostic, path, "cache_valid"
                );
            }
            return comparison_difference(diagnostic, path, "state");
        }
        if (left->encoder_batch_norms[index].owned_block ==
            right->encoder_batch_norms[index].owned_block) {

            return comparison_difference(
                diagnostic, path, "cross_model_storage_alias"
            );
        }
    }
    if (!batch_norm_equal(
            &left->decoder_fc_batch_norm,
            &right->decoder_fc_batch_norm) ||
        left->decoder_fc_batch_norm.owned_block ==
            right->decoder_fc_batch_norm.owned_block) {

        return comparison_difference(
            diagnostic, "model.decoder_fc_batch_norm", "state"
        );
    }
    for (index = 0;
         index < left->config.decoder_hidden_conv_count;
         index++) {

        if (!batch_norm_equal(
                &left->decoder_hidden_batch_norms[index],
                &right->decoder_hidden_batch_norms[index]) ||
            left->decoder_hidden_batch_norms[index].owned_block ==
                right->decoder_hidden_batch_norms[index].owned_block) {

            (void)snprintf(
                path,
                sizeof(path),
                "model.decoder_hidden_batch_norms[%zu]",
                index
            );
            return comparison_difference(diagnostic, path, "state");
        }
    }
    return independent_registries_equal(left, right, diagnostic);
}


static int independent_tensor_equal(
    const VAECTensor *left,
    const VAECTensor *right,
    ComparisonDiagnostic *diagnostic,
    const char *path
)
{
    char shape_path[160];

    if (left == NULL || right == NULL) {
        return comparison_difference(diagnostic, path, "object");
    }
    (void)snprintf(shape_path, sizeof(shape_path), "%s.shape", path);
    if (!shape_metadata_exact(
            &left->shape, &right->shape, diagnostic, shape_path)) {

        return 0;
    }
    if (left->owned_bytes != right->owned_bytes) {
        return comparison_difference(diagnostic, path, "owned_bytes");
    }
    if (left->owns_memory != right->owns_memory) {
        return comparison_difference(diagnostic, path, "owns_memory");
    }
    if (left->has_grad != right->has_grad) {
        return comparison_difference(diagnostic, path, "has_grad");
    }
    if (!left->owns_memory) {
        return 1;
    }
    if (left->owned_block == NULL || right->owned_block == NULL ||
        left->owned_block == right->owned_block) {

        return comparison_difference(
            diagnostic, path, "independent_owned_storage"
        );
    }
    if (memcmp(
            left->owned_block,
            right->owned_block,
            left->owned_bytes) != 0) {

        return comparison_difference(diagnostic, path, "contents");
    }
    return 1;
}


static int independent_workspaces_equal(
    const VAECModelWorkspace *left,
    const VAECModelWorkspace *right,
    ComparisonDiagnostic *diagnostic
)
{
    const VAECTensor *left_tensors[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ];
    const VAECTensor *right_tensors[
        VAE_C_MODEL_WORKSPACE_MAX_OWNED_TENSORS
    ];
    size_t left_count;
    size_t right_count;
    size_t index;
    char path[128];

    comparison_diagnostic_clear(diagnostic);
    if (left == NULL || right == NULL) {
        return comparison_difference(diagnostic, "workspace", "object");
    }
    if (memcmp(
            left->profile_id,
            right->profile_id,
            VAE_C_PROFILE_TEXT_CAP) != 0) {

        return comparison_difference(
            diagnostic, "workspace", "profile_id"
        );
    }
    if (!config_metadata_exact(
            &left->config,
            &right->config,
            diagnostic,
            "workspace.config")) {

        return 0;
    }
#define CHECK_INDEPENDENT_WORKSPACE_FIELD(name) \
    do { \
        if (left->name != right->name) { \
            return comparison_difference(diagnostic, "workspace", #name); \
        } \
    } while (0)
    CHECK_INDEPENDENT_WORKSPACE_FIELD(batch_size);
    CHECK_INDEPENDENT_WORKSPACE_FIELD(owned_tensor_count);
    CHECK_INDEPENDENT_WORKSPACE_FIELD(logical_value_count);
    CHECK_INDEPENDENT_WORKSPACE_FIELD(storage_float_count);
    CHECK_INDEPENDENT_WORKSPACE_FIELD(storage_bytes);
    CHECK_INDEPENDENT_WORKSPACE_FIELD(vae_backward_model);
    CHECK_INDEPENDENT_WORKSPACE_FIELD(vae_backward_generation);
    CHECK_INDEPENDENT_WORKSPACE_FIELD(allocated);
#undef CHECK_INDEPENDENT_WORKSPACE_FIELD
    left_count = collect_workspace_tensors_fixed(left, left_tensors);
    right_count = collect_workspace_tensors_fixed(right, right_tensors);
    if (left_count != right_count) {
        return comparison_difference(
            diagnostic, "workspace", "tensor_count"
        );
    }
    for (index = 0; index < left_count; index++) {
        if (left_tensors[index] == &left->z) {
            (void)snprintf(path, sizeof(path), "workspace.z");
        } else {
            (void)snprintf(
                path, sizeof(path), "workspace.fixed_tensors[%zu]", index
            );
        }
        if (!independent_tensor_equal(
                left_tensors[index],
                right_tensors[index],
                diagnostic,
                path)) {

            return 0;
        }
    }
    if (left->decoder_reshape_view.data !=
            left->decoder_fc_relu_output.data ||
        left->decoder_reshape_view.grad !=
            left->decoder_fc_relu_output.grad ||
        right->decoder_reshape_view.data !=
            right->decoder_fc_relu_output.data ||
        right->decoder_reshape_view.grad !=
            right->decoder_fc_relu_output.grad) {

        return comparison_difference(
            diagnostic, "workspace.decoder_reshape_view", "owner_alias"
        );
    }
    return 1;
}


static int independent_fixture_pair_valid_and_equal(
    const VAECModel *left_model,
    const VAECModelWorkspace *left_workspace,
    const VAECModel *right_model,
    const VAECModelWorkspace *right_workspace,
    ComparisonDiagnostic *diagnostic
)
{
    return vae_c_model_validate(left_model) == VAE_C_OP_OK &&
        vae_c_model_workspace_validate(left_model, left_workspace) ==
            VAE_C_OP_OK &&
        vae_c_model_validate(right_model) == VAE_C_OP_OK &&
        vae_c_model_workspace_validate(right_model, right_workspace) ==
            VAE_C_OP_OK &&
        independent_models_equal(left_model, right_model, diagnostic) &&
        independent_workspaces_equal(
            left_workspace, right_workspace, diagnostic
        );
}


static int test_snapshot_comparator_regressions(
    const VAECProfileCatalog *catalog
)
{
    const VAECProfile *profile = vae_c_profile_catalog_find(
        catalog, "mini75_smoke_v1"
    );
    ComparisonDiagnostic diagnostic;
    VAECModel left_model;
    VAECModel right_model;
    VAECModelWorkspace left_workspace;
    VAECModelWorkspace right_workspace;
    int passed = 1;

    vae_c_model_init(&left_model);
    vae_c_model_init(&right_model);
    vae_c_model_workspace_init(&left_workspace);
    vae_c_model_workspace_init(&right_workspace);
    if (profile == NULL || !allocate_pair(
            profile,
            2,
            &left_model,
            &left_workspace,
            &right_model,
            &right_workspace) ||
        !independent_fixture_pair_valid_and_equal(
            &left_model,
            &left_workspace,
            &right_model,
            &right_workspace,
            &diagnostic)) {

        free_fixture(&left_model, &left_workspace);
        free_fixture(&right_model, &right_workspace);
        return 0;
    }

    if (passed) {
        size_t saved = right_model.max_batch_size;
        int detected;
        int restored;

        passed = independent_fixture_pair_valid_and_equal(
            &left_model,
            &left_workspace,
            &right_model,
            &right_workspace,
            &diagnostic
        );
        if (passed) {
            right_model.max_batch_size++;
            detected = !independent_models_equal(
                &left_model, &right_model, &diagnostic
            ) && diagnostic_contains(
                &diagnostic, "model.max_batch_size"
            );
            right_model.max_batch_size = saved;
            restored = independent_fixture_pair_valid_and_equal(
                &left_model,
                &left_workspace,
                &right_model,
                &right_workspace,
                &diagnostic
            );
            passed = detected && restored;
        }
    }
    if (passed) {
        size_t saved = right_workspace.batch_size;
        int detected;
        int restored;

        passed = independent_fixture_pair_valid_and_equal(
            &left_model,
            &left_workspace,
            &right_model,
            &right_workspace,
            &diagnostic
        );
        if (passed) {
            right_workspace.batch_size++;
            detected = !independent_workspaces_equal(
                &left_workspace, &right_workspace, &diagnostic
            ) && diagnostic_contains(
                &diagnostic, "workspace.batch_size"
            );
            right_workspace.batch_size = saved;
            restored = independent_fixture_pair_valid_and_equal(
                &left_model,
                &left_workspace,
                &right_model,
                &right_workspace,
                &diagnostic
            );
            passed = detected && restored;
        }
    }
    if (passed) {
        size_t saved = right_workspace.z.shape.strides[1];
        int detected;
        int restored;

        passed = independent_fixture_pair_valid_and_equal(
            &left_model,
            &left_workspace,
            &right_model,
            &right_workspace,
            &diagnostic
        );
        if (passed) {
            right_workspace.z.shape.strides[1]++;
            detected = !independent_workspaces_equal(
                &left_workspace, &right_workspace, &diagnostic
            ) && diagnostic_contains(
                &diagnostic, "workspace.z.shape.strides[1]"
            );
            right_workspace.z.shape.strides[1] = saved;
            restored = independent_fixture_pair_valid_and_equal(
                &left_model,
                &left_workspace,
                &right_model,
                &right_workspace,
                &diagnostic
            );
            passed = detected && restored;
        }
    }
    if (passed) {
        bool saved = right_model.encoder_batch_norms[0].cache_valid;
        int detected;
        int restored;

        passed = independent_fixture_pair_valid_and_equal(
            &left_model,
            &left_workspace,
            &right_model,
            &right_workspace,
            &diagnostic
        );
        if (passed) {
            right_model.encoder_batch_norms[0].cache_valid = !saved;
            detected = !independent_models_equal(
                &left_model, &right_model, &diagnostic
            ) && diagnostic_contains(
                &diagnostic,
                "model.encoder_batch_norms[0].cache_valid"
            );
            right_model.encoder_batch_norms[0].cache_valid = saved;
            restored = independent_fixture_pair_valid_and_equal(
                &left_model,
                &left_workspace,
                &right_model,
                &right_workspace,
                &diagnostic
            );
            passed = detected && restored;
        }
    }
    if (passed) {
        VAECParameter *saved = right_model.parameters[0];
        int detected;
        int restored;

        passed = independent_fixture_pair_valid_and_equal(
            &left_model,
            &left_workspace,
            &right_model,
            &right_workspace,
            &diagnostic
        ) && parameter_equal(
            left_model.parameters[0], right_model.parameters[0]
        );
        if (passed) {
            right_model.parameters[0] = left_model.parameters[0];
            detected = parameter_equal(
                left_model.parameters[0], right_model.parameters[0]
            ) && !independent_models_equal(
                &left_model, &right_model, &diagnostic
            ) && diagnostic_contains(
                &diagnostic,
                "model.parameters[0]{key=1}.right_self_pointer"
            );
            right_model.parameters[0] = saved;
            restored = independent_fixture_pair_valid_and_equal(
                &left_model,
                &left_workspace,
                &right_model,
                &right_workspace,
                &diagnostic
            );
            passed = detected && restored;
        }
    }

    free_fixture(&left_model, &left_workspace);
    free_fixture(&right_model, &right_workspace);
    return passed;
}


static void batch_norm_pointers_capture(
    const VAECBatchNorm *layer,
    BatchNormPointers *pointers
)
{
    pointers->running_mean = layer->running_mean;
    pointers->running_variance = layer->running_variance;
    pointers->batch_mean = layer->batch_mean;
    pointers->batch_inv_std = layer->batch_inv_std;
    pointers->work_mean = layer->work_mean;
    pointers->work_inv_std = layer->work_inv_std;
    pointers->work_unbiased_variance = layer->work_unbiased_variance;
    pointers->owned_block = layer->owned_block;
}


static void batch_norm_pointers_restore(
    VAECBatchNorm *layer,
    const BatchNormPointers *pointers
)
{
    layer->running_mean = pointers->running_mean;
    layer->running_variance = pointers->running_variance;
    layer->batch_mean = pointers->batch_mean;
    layer->batch_inv_std = pointers->batch_inv_std;
    layer->work_mean = pointers->work_mean;
    layer->work_inv_std = pointers->work_inv_std;
    layer->work_unbiased_variance = pointers->work_unbiased_variance;
    layer->owned_block = pointers->owned_block;
}


static void batch_norm_rebind(VAECBatchNorm *layer, float *base)
{
    size_t features = layer->num_features;

    layer->owned_block = base;
    layer->running_mean = base;
    layer->running_variance = base + features;
    layer->batch_mean = base + 2 * features;
    layer->batch_inv_std = base + 3 * features;
    layer->work_mean = base + 4 * features;
    layer->work_inv_std = base + 5 * features;
    layer->work_unbiased_variance = base + 6 * features;
}


static int expand_tensor_backing(VAECTensor *tensor, size_t multiplier)
{
    float *expanded;
    size_t float_count;

    if (tensor == NULL || !tensor->owns_memory || !tensor->has_grad ||
        multiplier < 2 || !size_multiply(
            tensor->shape.count, multiplier, &float_count)) {

        return 0;
    }
    expanded = (float *)calloc(float_count, sizeof(float));
    if (expanded == NULL) {
        return 0;
    }
    memcpy(expanded, tensor->owned_block, tensor->owned_bytes);
    free(tensor->owned_block);
    tensor->owned_block = expanded;
    tensor->data = expanded;
    tensor->grad = expanded + tensor->shape.count;
    return 1;
}


static int test_aliases(const VAECProfileCatalog *catalog)
{
    enum {
        ALIAS_VALUES = 0,
        ALIAS_GRADIENTS,
        ALIAS_ADAM_M,
        ALIAS_ADAM_V,
        ALIAS_BN_STATE,
        ALIAS_DATA_CASE_COUNT
    };
    const VAECProfile *profile = vae_c_profile_catalog_find(
        catalog, "mini75_smoke_v1"
    );
    VAECModel model;
    VAECModelWorkspace workspace;
    VAECParameter *parameter;
    size_t alias_case;
    int passed = 1;

    vae_c_model_init(&model);
    vae_c_model_workspace_init(&workspace);
    if (!allocate_fixture(
            profile, 2, 1, UINT64_C(777), &model, &workspace)) {

        free_fixture(&model, &workspace);
        return 0;
    }
    parameter = &model.encoder_batch_norms[2].gamma;
    if (parameter->shape.count != workspace.epsilon.shape.count) {
        free_fixture(&model, &workspace);
        return 0;
    }
    for (alias_case = 0;
         alias_case < (size_t)ALIAS_DATA_CASE_COUNT;
         alias_case++) {

        float *saved_data = workspace.epsilon.data;
        float *saved_owned_block = workspace.epsilon.owned_block;
        FullSnapshot snapshot;
        const float *alias_pointer;
        VAECOpStatus status;

        full_snapshot_init(&snapshot);
        switch (alias_case) {
        case ALIAS_VALUES:
            alias_pointer = parameter->values;
            break;
        case ALIAS_GRADIENTS:
            alias_pointer = parameter->gradients;
            break;
        case ALIAS_ADAM_M:
            alias_pointer = parameter->adam_m;
            break;
        case ALIAS_ADAM_V:
            alias_pointer = parameter->adam_v;
            break;
        case ALIAS_BN_STATE:
            alias_pointer = model.encoder_batch_norms[2].running_mean;
            break;
        default:
            alias_pointer = NULL;
            break;
        }
        workspace.epsilon.data = (float *)(void *)alias_pointer;
        workspace.epsilon.owned_block = workspace.epsilon.data;
        passed = passed && alias_pointer != NULL &&
            vae_c_model_validate(&model) == VAE_C_OP_OK &&
            vae_c_model_workspace_validate(&model, &workspace) ==
                VAE_C_OP_OK &&
            full_snapshot_capture(&model, &workspace, &snapshot);
        status = passed ? vae_c_model_decoder_forward(
            &model, VAE_C_BATCH_NORM_MODE_EVAL, &workspace
        ) : VAE_C_OP_OK;
        passed = passed && status == VAE_C_OP_INVALID_ARGUMENT &&
            full_snapshot_matches(&model, &workspace, &snapshot);
        full_snapshot_free(&snapshot);
        workspace.epsilon.data = saved_data;
        workspace.epsilon.owned_block = saved_owned_block;
    }

    if (passed) {
        VAECTensor *tensor = &workspace.decoder_fc_output;
        VAECBatchNorm *layer = &model.decoder_fc_batch_norm;
        BatchNormPointers saved_pointers;
        FullSnapshot snapshot;
        size_t expanded_bytes;
        unsigned char *expanded_copy = NULL;
        VAECOpStatus status;

        full_snapshot_init(&snapshot);
        batch_norm_pointers_capture(layer, &saved_pointers);
        passed = expand_tensor_backing(tensor, 8) &&
            size_multiply(
                tensor->shape.count, 8 * sizeof(float),
                &expanded_bytes);
        if (passed) {
            expanded_copy = (unsigned char *)malloc(expanded_bytes);
            passed = expanded_copy != NULL;
        }
        if (passed) {
            batch_norm_rebind(layer, tensor->grad);
            memcpy(expanded_copy, tensor->owned_block, expanded_bytes);
            passed = vae_c_model_validate(&model) == VAE_C_OP_OK &&
                vae_c_model_workspace_validate(&model, &workspace) ==
                    VAE_C_OP_OK &&
                full_snapshot_capture(&model, &workspace, &snapshot);
        }
        status = passed ? vae_c_model_decoder_forward(
            &model, VAE_C_BATCH_NORM_MODE_EVAL, &workspace
        ) : VAE_C_OP_OK;
        passed = passed && status == VAE_C_OP_INVALID_ARGUMENT &&
            full_snapshot_matches(&model, &workspace, &snapshot) &&
            memcmp(tensor->owned_block,
                   expanded_copy,
                   expanded_bytes) == 0;
        full_snapshot_free(&snapshot);
        batch_norm_pointers_restore(layer, &saved_pointers);
        free(expanded_copy);
    }

    if (passed) {
        float *saved_data = workspace.z.data;
        float *saved_grad = workspace.z.grad;
        float *saved_owned = workspace.z.owned_block;
        FullSnapshot snapshot;

        full_snapshot_init(&snapshot);
        workspace.z.data = workspace.decoder_fc_output.data;
        workspace.z.grad = workspace.decoder_fc_output.data +
            workspace.z.shape.count;
        workspace.z.owned_block = workspace.z.data;
        passed = full_snapshot_capture(&model, &workspace, &snapshot) &&
            vae_c_model_decoder_forward(
                &model, VAE_C_BATCH_NORM_MODE_EVAL, &workspace) ==
                VAE_C_OP_INVALID_STATE &&
            full_snapshot_matches(&model, &workspace, &snapshot);
        full_snapshot_free(&snapshot);
        workspace.z.data = saved_data;
        workspace.z.grad = saved_grad;
        workspace.z.owned_block = saved_owned;
    }

    if (passed) {
        float *saved_data = workspace.decoder_reshape_view.data;
        FullSnapshot snapshot;

        full_snapshot_init(&snapshot);
        workspace.decoder_reshape_view.data = workspace.z.data;
        passed = full_snapshot_capture(&model, &workspace, &snapshot) &&
            vae_c_model_decoder_forward(
                &model, VAE_C_BATCH_NORM_MODE_EVAL, &workspace) ==
                VAE_C_OP_INVALID_STATE &&
            full_snapshot_matches(&model, &workspace, &snapshot);
        full_snapshot_free(&snapshot);
        workspace.decoder_reshape_view.data = saved_data;
    }
    passed = passed && vae_c_model_validate(&model) == VAE_C_OP_OK &&
        vae_c_model_workspace_validate(&model, &workspace) == VAE_C_OP_OK;
    free_fixture(&model, &workspace);
    return passed;
}


static int test_negative_preflight(
    const VAECProfileCatalog *catalog
)
{
    const VAECProfile *profile = vae_c_profile_catalog_find(
        catalog, "mini75_smoke_v1"
    );
    VAECModel model;
    VAECModelWorkspace workspace;
    VAECModel empty_model;
    VAECModelWorkspace empty_workspace;
    VAECConfig saved_config;
    VAECShape saved_shape;
    FullSnapshot initial_snapshot;
    float saved_value;
    bool saved_owns;
    size_t saved_stride;
    int passed;

    vae_c_model_init(&model);
    vae_c_model_init(&empty_model);
    vae_c_model_workspace_init(&workspace);
    vae_c_model_workspace_init(&empty_workspace);
    full_snapshot_init(&initial_snapshot);
    if (!allocate_fixture(
            profile, 2, 2, UINT64_C(999), &model, &workspace)) {

        free_fixture(&model, &workspace);
        return 0;
    }
    passed = full_snapshot_capture(
        &model, &workspace, &initial_snapshot
    ) && vae_c_model_decoder_forward(
        NULL, VAE_C_BATCH_NORM_MODE_EVAL, &workspace
    ) == VAE_C_OP_INVALID_ARGUMENT &&
        full_snapshot_matches(&model, &workspace, &initial_snapshot) &&
        vae_c_model_decoder_forward(
        &model, VAE_C_BATCH_NORM_MODE_EVAL, NULL
    ) == VAE_C_OP_INVALID_ARGUMENT &&
        full_snapshot_matches(&model, &workspace, &initial_snapshot) &&
        call_rejected_without_mutation(
            &model,
            (VAECBatchNormMode)0,
            &workspace,
            VAE_C_OP_INVALID_ARGUMENT
        ) && vae_c_model_decoder_forward(
        &empty_model, VAE_C_BATCH_NORM_MODE_EVAL, &workspace
    ) != VAE_C_OP_OK && vae_c_model_decoder_forward(
        &model, VAE_C_BATCH_NORM_MODE_EVAL, &empty_workspace
    ) != VAE_C_OP_OK;
    full_snapshot_free(&initial_snapshot);
    if (!passed) {
        fprintf(stderr, "negative subcase failed: null_empty_mode\n");
    }

    saved_config = model.config;
    model.config.conv_depth = 2;
    passed = passed && call_rejected_without_mutation(
        &model, VAE_C_BATCH_NORM_MODE_EVAL, &workspace,
        VAE_C_OP_INVALID_STATE
    );
    if (!passed) {
        fprintf(stderr, "negative subcase failed: depth\n");
    }
    model.config = saved_config;
    saved_config = model.config;
    model.config.decoder_hidden_conv_count++;
    passed = passed && call_rejected_without_mutation(
        &model, VAE_C_BATCH_NORM_MODE_EVAL, &workspace,
        VAE_C_OP_INVALID_STATE
    );
    if (!passed) {
        fprintf(stderr, "negative subcase failed: decoder_count\n");
    }
    model.config = saved_config;

    saved_shape = workspace.decoder_fc_output.shape;
    workspace.decoder_fc_output.shape.rank = 1;
    passed = passed && call_rejected_without_mutation(
        &model, VAE_C_BATCH_NORM_MODE_EVAL, &workspace,
        VAE_C_OP_INVALID_STATE
    );
    if (!passed) {
        fprintf(stderr, "negative subcase failed: decoder_slot\n");
    }
    workspace.decoder_fc_output.shape = saved_shape;
    saved_shape = workspace.logits.shape;
    workspace.logits.shape.dims[2]--;
    passed = passed && call_rejected_without_mutation(
        &model, VAE_C_BATCH_NORM_MODE_EVAL, &workspace,
        VAE_C_OP_INVALID_STATE
    );
    if (!passed) {
        fprintf(stderr, "negative subcase failed: logits_shape\n");
    }
    workspace.logits.shape = saved_shape;

    saved_owns = workspace.decoder_reshape_view.owns_memory;
    workspace.decoder_reshape_view.owns_memory = true;
    passed = passed && call_rejected_without_mutation(
        &model, VAE_C_BATCH_NORM_MODE_EVAL, &workspace,
        VAE_C_OP_INVALID_STATE
    );
    if (!passed) {
        fprintf(stderr, "negative subcase failed: reshape_ownership\n");
    }
    workspace.decoder_reshape_view.owns_memory = saved_owns;
    saved_stride = workspace.decoder_reshape_view.shape.strides[1];
    workspace.decoder_reshape_view.shape.strides[1]++;
    passed = passed && call_rejected_without_mutation(
        &model, VAE_C_BATCH_NORM_MODE_EVAL, &workspace,
        VAE_C_OP_INVALID_STATE
    );
    if (!passed) {
        fprintf(stderr, "negative subcase failed: reshape_stride\n");
    }
    workspace.decoder_reshape_view.shape.strides[1] = saved_stride;
    saved_shape = workspace.decoder_reshape_view.shape;
    workspace.decoder_reshape_view.shape.dims[2]++;
    passed = passed && call_rejected_without_mutation(
        &model, VAE_C_BATCH_NORM_MODE_EVAL, &workspace,
        VAE_C_OP_INVALID_STATE
    );
    if (!passed) {
        fprintf(stderr, "negative subcase failed: reshape_shape\n");
    }
    workspace.decoder_reshape_view.shape = saved_shape;

    saved_config = workspace.config;
    workspace.config.decoder_upsample_targets[0] = SIZE_MAX;
    passed = passed && call_rejected_without_mutation(
        &model, VAE_C_BATCH_NORM_MODE_EVAL, &workspace,
        VAE_C_OP_INVALID_STATE
    );
    if (!passed) {
        fprintf(stderr, "negative subcase failed: overflow_config\n");
    }
    workspace.config = saved_config;

    saved_value = model.decoder_fc.weights.values[0];
    model.decoder_fc.weights.values[0] = NAN;
    passed = passed && call_rejected_without_mutation(
        &model, VAE_C_BATCH_NORM_MODE_EVAL, &workspace,
        VAE_C_OP_INVALID_STATE
    );
    if (!passed) {
        fprintf(stderr, "negative subcase failed: parameter_nan\n");
    }
    model.decoder_fc.weights.values[0] = INFINITY;
    passed = passed && call_rejected_without_mutation(
        &model, VAE_C_BATCH_NORM_MODE_EVAL, &workspace,
        VAE_C_OP_INVALID_STATE
    );
    if (!passed) {
        fprintf(stderr, "negative subcase failed: parameter_inf\n");
    }
    model.decoder_fc.weights.values[0] = saved_value;

    passed = passed && vae_c_model_validate(&model) == VAE_C_OP_OK &&
        vae_c_model_workspace_validate(&model, &workspace) == VAE_C_OP_OK;
    free_fixture(&model, &workspace);

    vae_c_model_init(&model);
    vae_c_model_workspace_init(&workspace);
    if (passed && profile != NULL &&
        vae_c_model_allocate(&model, profile, 2) == VAE_C_OP_OK &&
        vae_c_model_workspace_allocate(&model, 2, &workspace) ==
            VAE_C_OP_OK) {

        fill_z(&workspace);
        passed = call_rejected_without_mutation(
            &model, VAE_C_BATCH_NORM_MODE_EVAL, &workspace,
            VAE_C_OP_INVALID_STATE
        );
    } else {
        fprintf(stderr, "negative subcase failed: uninitialized_setup\n");
        passed = 0;
    }
    free_fixture(&model, &workspace);
    return passed;
}


static int test_generation_ticket_and_overflow(
    const VAECProfileCatalog *catalog
)
{
    const VAECProfile *profile = vae_c_profile_catalog_find(
        catalog, "mini75_smoke_v1"
    );
    VAECModel model;
    VAECModelWorkspace workspace;
    FullSnapshot snapshot;
    int passed;

    full_snapshot_init(&snapshot);
    vae_c_model_init(&model);
    vae_c_model_workspace_init(&workspace);
    if (!allocate_fixture(
            profile, 2, 2, UINT64_C(161803), &model, &workspace)) {

        free_fixture(&model, &workspace);
        return 0;
    }

    model.vae_cache_generation = UINT64_C(41);
    workspace.vae_backward_model = &model;
    workspace.vae_backward_generation = UINT64_C(41);
    passed = vae_c_model_decoder_forward(
        &model, VAE_C_BATCH_NORM_MODE_EVAL, &workspace
    ) == VAE_C_OP_OK &&
        model.vae_cache_generation == UINT64_C(42) &&
        workspace.vae_backward_model == NULL &&
        workspace.vae_backward_generation == UINT64_C(0);

    model.vae_cache_generation = UINT64_MAX;
    workspace.vae_backward_model = &model;
    workspace.vae_backward_generation = UINT64_MAX;
    passed = passed && full_snapshot_capture(&model, &workspace, &snapshot) &&
        vae_c_model_decoder_forward(
            &model, VAE_C_BATCH_NORM_MODE_EVAL, &workspace
        ) == VAE_C_OP_OVERFLOW &&
        full_snapshot_matches(&model, &workspace, &snapshot);

    full_snapshot_free(&snapshot);
    free_fixture(&model, &workspace);
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
        "manual_equivalence_depth3_train_eval",
        test_manual_depth3(&catalog)
    );
    failures += report_check(
        "manual_equivalence_synthetic_depth4_train_eval",
        test_manual_depth4()
    );
    failures += report_check(
        "all_official_profiles_eval_b1_b2_train_b2",
        test_official_profiles(&catalog)
    );
    failures += report_check(
        "batch1_train_atomic_eval_valid",
        test_batch_one(&catalog)
    );
    failures += report_check(
        "z_finite_unbounded_immutable_and_nonfinite_atomic",
        test_z_contract(&catalog)
    );
    failures += report_check(
        "linear_logits_no_final_activation_or_normalization",
        test_linear_logits(&catalog)
    );
    failures += report_check(
        "eval_determinism_independent_models",
        test_eval_determinism(&catalog)
    );
    failures += report_check(
        "deep_train_eval_mutation_boundaries",
        test_mutation_boundaries(&catalog)
    );
    failures += report_check(
        "field_comparators_detect_model_workspace_tensor_bn_registry",
        test_snapshot_comparator_regressions(&catalog)
    );
    failures += report_check(
        "isolated_exact_aliases_and_reshape_relation_atomic",
        test_aliases(&catalog)
    );
    failures += report_check(
        "negative_structural_numeric_overflow_preflight_atomic",
        test_negative_preflight(&catalog)
    );
    failures += report_check(
        "generation_ticket_and_overflow",
        test_generation_ticket_and_overflow(&catalog)
    );

    vae_c_profile_catalog_free(&catalog);
    if (failures != 0) {
        fprintf(stderr, "%d VAE C DECODER FORWARD TEST(S) FAILED\n",
                failures);
        return 1;
    }
    printf("DECODER_FORWARD_STRONG_GUARANTEE=preflight_only\n");
    printf("DECODER_FORWARD_POST_START_ROLLBACK=false\n");
    printf("POST_START_FAILURE_INJECTION=not_forced_portably\n");
    printf("ALL VAE C MODEL DECODER FORWARD TESTS PASSED\n");
    return 0;
}
