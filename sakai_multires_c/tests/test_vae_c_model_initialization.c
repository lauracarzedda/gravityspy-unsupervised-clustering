#include "vae_c_init.h"
#include "vae_c_model.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


typedef enum {
    TEST_INIT_ZERO = 1,
    TEST_INIT_ONE,
    TEST_INIT_HE,
    TEST_INIT_XAVIER
} TestInitKind;


static int report_check(const char *name, int passed)
{
    if (passed) {
        printf("[PASS] %s\n", name);
        return 0;
    }
    fprintf(stderr, "[FAIL] %s\n", name);
    return 1;
}


static int all_equal(const float *values, size_t count, float expected)
{
    size_t index;

    if (values == NULL) {
        return 0;
    }
    for (index = 0; index < count; index++) {
        if (values[index] != expected) {
            return 0;
        }
    }
    return 1;
}


static uint64_t hash_bytes(uint64_t hash, const void *data, size_t size)
{
    const unsigned char *bytes = (const unsigned char *)data;
    size_t index;

    for (index = 0; index < size; index++) {
        hash ^= (uint64_t)bytes[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}


static uint64_t hash_parameter_block(
    uint64_t hash,
    const VAECParameter *parameter
)
{
    if (parameter->owned_block == NULL) {
        return hash;
    }
    return hash_bytes(hash, parameter->owned_block, parameter->owned_bytes);
}


static uint64_t hash_model_buffers(const VAECModel *model)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    size_t index;

    if (model == NULL) {
        return UINT64_C(0);
    }
    for (index = 0; index < VAE_C_MODEL_MAX_ENCODER_CONVS; index++) {
        hash = hash_parameter_block(hash, &model->encoder_convs[index].weights);
        hash = hash_parameter_block(hash, &model->encoder_convs[index].bias);
        hash = hash_parameter_block(
            hash,
            &model->encoder_batch_norms[index].gamma
        );
        hash = hash_parameter_block(
            hash,
            &model->encoder_batch_norms[index].beta
        );
    }
    hash = hash_parameter_block(hash, &model->encoder_fc.weights);
    hash = hash_parameter_block(hash, &model->encoder_fc.bias);
    hash = hash_parameter_block(hash, &model->decoder_fc.weights);
    hash = hash_parameter_block(hash, &model->decoder_fc.bias);
    hash = hash_parameter_block(hash, &model->decoder_fc_batch_norm.gamma);
    hash = hash_parameter_block(hash, &model->decoder_fc_batch_norm.beta);
    for (index = 0;
         index < VAE_C_MODEL_MAX_DECODER_HIDDEN_CONVS;
         index++) {

        hash = hash_parameter_block(
            hash,
            &model->decoder_hidden_convs[index].weights
        );
        hash = hash_parameter_block(
            hash,
            &model->decoder_hidden_convs[index].bias
        );
        hash = hash_parameter_block(
            hash,
            &model->decoder_hidden_batch_norms[index].gamma
        );
        hash = hash_parameter_block(
            hash,
            &model->decoder_hidden_batch_norms[index].beta
        );
    }
    hash = hash_parameter_block(hash, &model->decoder_output_conv.weights);
    hash = hash_parameter_block(hash, &model->decoder_output_conv.bias);
    for (index = 0; index < model->config.conv_depth; index++) {
        const VAECBatchNorm *layer = &model->encoder_batch_norms[index];

        if (layer->owned_block != NULL) {
            hash = hash_bytes(hash, layer->owned_block, layer->owned_bytes);
        }
    }
    if (model->decoder_fc_batch_norm.owned_block != NULL) {
        hash = hash_bytes(
            hash,
            model->decoder_fc_batch_norm.owned_block,
            model->decoder_fc_batch_norm.owned_bytes
        );
    }
    for (index = 0;
         index < model->config.decoder_hidden_conv_count;
         index++) {

        const VAECBatchNorm *layer =
            &model->decoder_hidden_batch_norms[index];

        if (layer->owned_block != NULL) {
            hash = hash_bytes(hash, layer->owned_block, layer->owned_bytes);
        }
    }
    if (model->encoder_max_pool.argmax_indices != NULL) {
        hash = hash_bytes(
            hash,
            model->encoder_max_pool.argmax_indices,
            model->encoder_max_pool.owned_bytes
        );
    }
    return hash;
}


static int initialization_failure_preserves_and_reports(
    VAECModel *model,
    uint64_t seed,
    VAECOpStatus *observed_status
)
{
    VAECModel before;
    uint64_t buffers_before;
    VAECOpStatus status;

    before = *model;
    buffers_before = hash_model_buffers(model);
    status = vae_c_model_initialize_parameters_v1(model, seed);
    if (observed_status != NULL) {
        *observed_status = status;
    }
    return
        status != VAE_C_OP_OK &&
        memcmp(model, &before, sizeof(*model)) == 0 &&
        hash_model_buffers(model) == buffers_before;
}


static int initialization_failure_preserves(
    VAECModel *model,
    uint64_t seed
)
{
    return initialization_failure_preserves_and_reports(
        model,
        seed,
        NULL
    );
}


static TestInitKind kind_for_key(uint64_t key)
{
    switch (key) {
        case VAE_C_MODEL_PARAM_KEY_ENCODER_CONV1_WEIGHT:
        case VAE_C_MODEL_PARAM_KEY_ENCODER_CONV2_WEIGHT:
        case VAE_C_MODEL_PARAM_KEY_ENCODER_CONV3_WEIGHT:
        case VAE_C_MODEL_PARAM_KEY_ENCODER_CONV4_WEIGHT:
        case VAE_C_MODEL_PARAM_KEY_DECODER_FC_WEIGHT:
        case VAE_C_MODEL_PARAM_KEY_DECODER_CONV1_WEIGHT:
        case VAE_C_MODEL_PARAM_KEY_DECODER_CONV2_WEIGHT:
        case VAE_C_MODEL_PARAM_KEY_DECODER_CONV3_WEIGHT:
            return TEST_INIT_HE;

        case VAE_C_MODEL_PARAM_KEY_ENCODER_FC_WEIGHT:
        case VAE_C_MODEL_PARAM_KEY_DECODER_OUTPUT_WEIGHT:
            return TEST_INIT_XAVIER;

        case VAE_C_MODEL_PARAM_KEY_ENCODER_BN1_GAMMA:
        case VAE_C_MODEL_PARAM_KEY_ENCODER_BN2_GAMMA:
        case VAE_C_MODEL_PARAM_KEY_ENCODER_BN3_GAMMA:
        case VAE_C_MODEL_PARAM_KEY_ENCODER_BN4_GAMMA:
        case VAE_C_MODEL_PARAM_KEY_DECODER_FC_BN_GAMMA:
        case VAE_C_MODEL_PARAM_KEY_DECODER_BN1_GAMMA:
        case VAE_C_MODEL_PARAM_KEY_DECODER_BN2_GAMMA:
        case VAE_C_MODEL_PARAM_KEY_DECODER_BN3_GAMMA:
            return TEST_INIT_ONE;

        default:
            return TEST_INIT_ZERO;
    }
}


static int key_is_dense_weight(uint64_t key)
{
    return
        key == VAE_C_MODEL_PARAM_KEY_ENCODER_FC_WEIGHT ||
        key == VAE_C_MODEL_PARAM_KEY_DECODER_FC_WEIGHT;
}


static int compute_fans(
    const VAECParameter *parameter,
    size_t *fan_in,
    size_t *fan_out
)
{
    size_t area;

    if (parameter == NULL || fan_in == NULL || fan_out == NULL) {
        return 0;
    }
    if (key_is_dense_weight(parameter->parameter_key)) {
        if (parameter->shape.rank != 2) {
            return 0;
        }
        *fan_in = parameter->shape.dims[1];
        *fan_out = parameter->shape.dims[0];
        return *fan_in > 0 && *fan_out > 0;
    }
    if (parameter->shape.rank != 4 ||
        (parameter->shape.dims[2] != 0 &&
         parameter->shape.dims[3] >
            SIZE_MAX / parameter->shape.dims[2])) {

        return 0;
    }
    area = parameter->shape.dims[2] * parameter->shape.dims[3];
    if (area == 0 ||
        parameter->shape.dims[1] > SIZE_MAX / area ||
        parameter->shape.dims[0] > SIZE_MAX / area) {

        return 0;
    }
    *fan_in = parameter->shape.dims[1] * area;
    *fan_out = parameter->shape.dims[0] * area;
    return *fan_in > 0 && *fan_out > 0;
}


static VAECParameter *parameter_by_key(VAECModel *model, uint64_t key)
{
    size_t index;

    for (index = 0; index < model->parameter_count; index++) {
        if (model->parameters[index]->parameter_key == key) {
            return model->parameters[index];
        }
    }
    return NULL;
}


static int batch_norm_state_matches(const VAECBatchNorm *layer)
{
    return
        layer != NULL &&
        layer->num_features > 0 &&
        all_equal(layer->running_mean, layer->num_features, 0.0f) &&
        all_equal(layer->running_variance, layer->num_features, 1.0f) &&
        !layer->cache_valid;
}


static int verify_parameter_reference(
    const VAECParameter *parameter,
    uint64_t seed
)
{
    TestInitKind kind;
    VAECInitStatus status;
    float *reference;
    double bound;
    size_t fan_in = 0;
    size_t fan_out = 0;
    size_t fan_sum;
    size_t index;
    int any_nonzero = 0;
    int passed = 1;

    kind = kind_for_key(parameter->parameter_key);
    reference = (float *)malloc(parameter->shape.count * sizeof(float));
    if (reference == NULL) {
        return 0;
    }
    if (kind == TEST_INIT_ZERO) {
        status = vae_c_init_zero_f32(reference, parameter->shape.count);
    } else if (kind == TEST_INIT_ONE) {
        status = vae_c_init_one_f32(reference, parameter->shape.count);
    } else if (!compute_fans(parameter, &fan_in, &fan_out)) {
        free(reference);
        return 0;
    } else if (kind == TEST_INIT_HE) {
        status = vae_c_init_he_uniform_f32(
            reference,
            parameter->shape.count,
            seed,
            parameter->parameter_key,
            fan_in
        );
    } else {
        status = vae_c_init_xavier_uniform_f32(
            reference,
            parameter->shape.count,
            seed,
            parameter->parameter_key,
            fan_in,
            fan_out
        );
    }
    if (status != VAE_C_INIT_OK ||
        memcmp(
            reference,
            parameter->values,
            parameter->shape.count * sizeof(float)) != 0) {

        free(reference);
        return 0;
    }
    if (kind == TEST_INIT_HE) {
        bound = sqrt(6.0 / (double)fan_in);
    } else if (kind == TEST_INIT_XAVIER) {
        if (fan_in > SIZE_MAX - fan_out) {
            free(reference);
            return 0;
        }
        fan_sum = fan_in + fan_out;
        bound = sqrt(6.0 / (double)fan_sum);
    } else {
        bound = 0.0;
    }
    for (index = 0; index < parameter->shape.count; index++) {
        double value = (double)parameter->values[index];

        if (!isfinite(value) ||
            ((kind == TEST_INIT_HE || kind == TEST_INIT_XAVIER) &&
             (value < -bound || value > bound))) {

            passed = 0;
        }
        if (value != 0.0) {
            any_nonzero = 1;
        }
    }
    if ((kind == TEST_INIT_HE || kind == TEST_INIT_XAVIER) &&
        parameter->shape.count >= 16 && !any_nonzero) {

        passed = 0;
    }
    free(reference);
    return passed;
}


static int verify_initialized_model(
    const VAECModel *model,
    uint64_t seed,
    int compare_references
)
{
    size_t index;

    if (model == NULL ||
        vae_c_model_validate(model) != VAE_C_OP_OK ||
        !vae_c_model_is_allocated(model) ||
        !vae_c_model_is_initialized(model) ||
        !model->parameters_initialized ||
        model->initialization_seed != seed) {

        return 0;
    }
    for (index = 0; index < model->parameter_count; index++) {
        const VAECParameter *parameter = model->parameters[index];
        TestInitKind kind = kind_for_key(parameter->parameter_key);
        float expected = kind == TEST_INIT_ONE ? 1.0f : 0.0f;

        if (!all_equal(
                parameter->gradients,
                parameter->shape.count,
                0.0f) ||
            !all_equal(parameter->adam_m, parameter->shape.count, 0.0f) ||
            !all_equal(parameter->adam_v, parameter->shape.count, 0.0f) ||
            ((kind == TEST_INIT_ZERO || kind == TEST_INIT_ONE) &&
             !all_equal(parameter->values, parameter->shape.count, expected)) ||
            (compare_references &&
             !verify_parameter_reference(parameter, seed))) {

            return 0;
        }
    }
    for (index = 0; index < model->config.conv_depth; index++) {
        if (!batch_norm_state_matches(
                &model->encoder_batch_norms[index])) {

            return 0;
        }
    }
    if (!batch_norm_state_matches(&model->decoder_fc_batch_norm)) {
        return 0;
    }
    for (index = 0;
         index < model->config.decoder_hidden_conv_count;
         index++) {

        if (!batch_norm_state_matches(
                &model->decoder_hidden_batch_norms[index])) {

            return 0;
        }
    }
    return 1;
}


static int parameter_semantically_equal(
    const VAECParameter *left,
    const VAECParameter *right
)
{
    size_t bytes;

    if (left->parameter_key != right->parameter_key ||
        memcmp(&left->shape, &right->shape, sizeof(left->shape)) != 0 ||
        left->owned_bytes != right->owned_bytes) {

        return 0;
    }
    bytes = left->shape.count * sizeof(float);
    return
        memcmp(left->values, right->values, bytes) == 0 &&
        memcmp(left->gradients, right->gradients, bytes) == 0 &&
        memcmp(left->adam_m, right->adam_m, bytes) == 0 &&
        memcmp(left->adam_v, right->adam_v, bytes) == 0;
}


static int conv_metadata_equal(
    const VAECConv2D *left,
    const VAECConv2D *right
)
{
    return
        left->in_channels == right->in_channels &&
        left->out_channels == right->out_channels &&
        left->kernel_size == right->kernel_size &&
        left->stride == right->stride &&
        left->padding == right->padding &&
        left->with_bias == right->with_bias;
}


static int dense_metadata_equal(
    const VAECDense *left,
    const VAECDense *right
)
{
    return
        left->in_features == right->in_features &&
        left->out_features == right->out_features &&
        left->with_bias == right->with_bias;
}


static int batch_norm_semantically_equal(
    const VAECBatchNorm *left,
    const VAECBatchNorm *right
)
{
    size_t bytes;

    if (left->kind != right->kind ||
        left->num_features != right->num_features ||
        left->epsilon != right->epsilon ||
        left->momentum != right->momentum ||
        left->cache_valid != right->cache_valid ||
        left->last_sample_count != right->last_sample_count ||
        memcmp(
            &left->last_input_shape,
            &right->last_input_shape,
            sizeof(left->last_input_shape)) != 0 ||
        memcmp(
            &left->last_output_shape,
            &right->last_output_shape,
            sizeof(left->last_output_shape)) != 0) {

        return 0;
    }
    bytes = left->num_features * sizeof(float);
    return
        memcmp(left->running_mean, right->running_mean, bytes) == 0 &&
        memcmp(left->running_variance, right->running_variance, bytes) == 0 &&
        memcmp(left->batch_mean, right->batch_mean, bytes) == 0 &&
        memcmp(left->batch_inv_std, right->batch_inv_std, bytes) == 0 &&
        memcmp(left->work_mean, right->work_mean, bytes) == 0 &&
        memcmp(left->work_inv_std, right->work_inv_std, bytes) == 0 &&
        memcmp(
            left->work_unbiased_variance,
            right->work_unbiased_variance,
            bytes) == 0;
}


static int models_semantically_equal(
    const VAECModel *left,
    const VAECModel *right
)
{
    size_t index;

    if (memcmp(left->profile_id, right->profile_id, sizeof(left->profile_id)) !=
            0 ||
        memcmp(&left->config, &right->config, sizeof(left->config)) != 0 ||
        left->max_batch_size != right->max_batch_size ||
        left->parameter_count != right->parameter_count ||
        left->trainable_value_count != right->trainable_value_count ||
        left->initialization_seed != right->initialization_seed ||
        left->vae_cache_generation != right->vae_cache_generation ||
        left->parameters_initialized != right->parameters_initialized ||
        left->allocated != right->allocated ||
        left->encoder_max_pool.capacity != right->encoder_max_pool.capacity ||
        left->encoder_max_pool.cache_valid !=
            right->encoder_max_pool.cache_valid) {

        return 0;
    }
    for (index = 0; index < left->parameter_count; index++) {
        if (!parameter_semantically_equal(
                left->parameters[index],
                right->parameters[index])) {

            return 0;
        }
    }
    for (index = 0; index < left->config.conv_depth; index++) {
        if (!conv_metadata_equal(
                &left->encoder_convs[index],
                &right->encoder_convs[index]) ||
            !batch_norm_semantically_equal(
                &left->encoder_batch_norms[index],
                &right->encoder_batch_norms[index])) {

            return 0;
        }
    }
    if (!batch_norm_semantically_equal(
            &left->decoder_fc_batch_norm,
            &right->decoder_fc_batch_norm) ||
        !dense_metadata_equal(&left->encoder_fc, &right->encoder_fc) ||
        !dense_metadata_equal(&left->decoder_fc, &right->decoder_fc)) {

        return 0;
    }
    for (index = 0;
         index < left->config.decoder_hidden_conv_count;
         index++) {

        if (!conv_metadata_equal(
                &left->decoder_hidden_convs[index],
                &right->decoder_hidden_convs[index]) ||
            !batch_norm_semantically_equal(
                &left->decoder_hidden_batch_norms[index],
                &right->decoder_hidden_batch_norms[index])) {

            return 0;
        }
    }
    return conv_metadata_equal(
        &left->decoder_output_conv,
        &right->decoder_output_conv
    );
}


static void make_synthetic_depth4(VAECProfile *profile)
{
    memset(profile, 0, sizeof(*profile));
    memcpy(
        profile->profile_id,
        "synthetic_depth4_small",
        sizeof("synthetic_depth4_small")
    );
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


static int test_official_profiles(const VAECProfileCatalog *catalog)
{
    const char *ids[4] = {
        "mini75_smoke_v1",
        "mini75_experiment_v1",
        "mini112_primary_v1",
        "mini112_scaleup_v1"
    };
    const uint64_t seed = UINT64_C(0x1020304050607080);
    size_t index;

    if (catalog == NULL || catalog->count != 4) {
        return 0;
    }
    for (index = 0; index < 4; index++) {
        const VAECProfile *profile = vae_c_profile_catalog_find(
            catalog,
            ids[index]
        );
        VAECModel first;
        VAECModel replica;
        int passed;

        vae_c_model_init(&first);
        vae_c_model_init(&replica);
        passed =
            profile != NULL &&
            vae_c_model_allocate(&first, profile, 2) == VAE_C_OP_OK &&
            vae_c_model_allocate(&replica, profile, 2) == VAE_C_OP_OK &&
            vae_c_model_validate(&first) == VAE_C_OP_OK &&
            vae_c_model_is_allocated(&first) &&
            !vae_c_model_is_initialized(&first) &&
            first.initialization_seed == UINT64_C(0) &&
            first.vae_cache_generation == UINT64_C(0) &&
            !first.parameters_initialized &&
            vae_c_model_initialize_parameters_v1(&first, seed) ==
                VAE_C_OP_OK &&
            vae_c_model_initialize_parameters_v1(&replica, seed) ==
                VAE_C_OP_OK &&
            verify_initialized_model(&first, seed, index == 0) &&
            verify_initialized_model(&replica, seed, 0) &&
            models_semantically_equal(&first, &replica);

        vae_c_model_free(&first);
        vae_c_model_free(&replica);
        if (!passed || !vae_c_model_is_empty(&first) ||
            vae_c_model_is_initialized(&first) ||
            first.initialization_seed != UINT64_C(0) ||
            first.vae_cache_generation != UINT64_C(0) ||
            first.parameters_initialized) {

            return 0;
        }
    }
    return 1;
}


static int parameter_values_differ(
    const VAECParameter *left,
    const VAECParameter *right
)
{
    return
        left != NULL && right != NULL &&
        left->shape.count == right->shape.count &&
        memcmp(
            left->values,
            right->values,
            left->shape.count * sizeof(float)) != 0;
}


static int fixed_parameters_equal(
    const VAECModel *left,
    const VAECModel *right
)
{
    size_t index;

    for (index = 0; index < left->parameter_count; index++) {
        const VAECParameter *a = left->parameters[index];
        const VAECParameter *b = right->parameters[index];
        TestInitKind kind = kind_for_key(a->parameter_key);

        if (a->parameter_key != b->parameter_key ||
            ((kind == TEST_INIT_ZERO || kind == TEST_INIT_ONE) &&
             memcmp(
                a->values,
                b->values,
                a->shape.count * sizeof(float)) != 0) ||
            memcmp(
                a->gradients,
                b->gradients,
                a->shape.count * sizeof(float)) != 0 ||
            memcmp(
                a->adam_m,
                b->adam_m,
                a->shape.count * sizeof(float)) != 0 ||
            memcmp(
                a->adam_v,
                b->adam_v,
                a->shape.count * sizeof(float)) != 0) {

            return 0;
        }
    }
    return 1;
}


static int test_seed_behavior(const VAECProfileCatalog *catalog)
{
    const VAECProfile *profile = vae_c_profile_catalog_find(
        catalog,
        "mini75_smoke_v1"
    );
    VAECModel first;
    VAECModel second;
    VAECModel zero_seed;
    VAECModel zero_seed_replica;
    int passed;

    vae_c_model_init(&first);
    vae_c_model_init(&second);
    vae_c_model_init(&zero_seed);
    vae_c_model_init(&zero_seed_replica);
    passed =
        profile != NULL &&
        vae_c_model_allocate(&first, profile, 2) == VAE_C_OP_OK &&
        vae_c_model_allocate(&second, profile, 2) == VAE_C_OP_OK &&
        vae_c_model_initialize_parameters_v1(&first, UINT64_C(111)) ==
            VAE_C_OP_OK &&
        vae_c_model_initialize_parameters_v1(&second, UINT64_C(222)) ==
            VAE_C_OP_OK &&
        parameter_values_differ(
            parameter_by_key(
                &first,
                VAE_C_MODEL_PARAM_KEY_ENCODER_CONV1_WEIGHT),
            parameter_by_key(
                &second,
                VAE_C_MODEL_PARAM_KEY_ENCODER_CONV1_WEIGHT)) &&
        parameter_values_differ(
            parameter_by_key(
                &first,
                VAE_C_MODEL_PARAM_KEY_DECODER_FC_WEIGHT),
            parameter_by_key(
                &second,
                VAE_C_MODEL_PARAM_KEY_DECODER_FC_WEIGHT)) &&
        parameter_values_differ(
            parameter_by_key(
                &first,
                VAE_C_MODEL_PARAM_KEY_ENCODER_FC_WEIGHT),
            parameter_by_key(
                &second,
                VAE_C_MODEL_PARAM_KEY_ENCODER_FC_WEIGHT)) &&
        parameter_values_differ(
            parameter_by_key(
                &first,
                VAE_C_MODEL_PARAM_KEY_DECODER_OUTPUT_WEIGHT),
            parameter_by_key(
                &second,
                VAE_C_MODEL_PARAM_KEY_DECODER_OUTPUT_WEIGHT)) &&
        fixed_parameters_equal(&first, &second) &&
        verify_initialized_model(&first, UINT64_C(111), 1) &&
        verify_initialized_model(&second, UINT64_C(222), 1) &&
        vae_c_model_allocate(&zero_seed, profile, 2) == VAE_C_OP_OK &&
        vae_c_model_initialize_parameters_v1(&zero_seed, UINT64_C(0)) ==
            VAE_C_OP_OK &&
        verify_initialized_model(&zero_seed, UINT64_C(0), 1) &&
        vae_c_model_allocate(&zero_seed_replica, profile, 2) ==
            VAE_C_OP_OK &&
        vae_c_model_initialize_parameters_v1(
            &zero_seed_replica,
            UINT64_C(0)) == VAE_C_OP_OK &&
        verify_initialized_model(&zero_seed_replica, UINT64_C(0), 1) &&
        zero_seed.parameters[0] != zero_seed_replica.parameters[0] &&
        zero_seed.parameters[0]->owned_block !=
            zero_seed_replica.parameters[0]->owned_block &&
        models_semantically_equal(&zero_seed, &zero_seed_replica);

    vae_c_model_free(&first);
    vae_c_model_free(&second);
    vae_c_model_free(&zero_seed);
    vae_c_model_free(&zero_seed_replica);
    return
        passed &&
        vae_c_model_is_empty(&zero_seed_replica) &&
        !vae_c_model_is_allocated(&zero_seed_replica) &&
        !vae_c_model_is_initialized(&zero_seed_replica);
}


static int test_synthetic_depth4(void)
{
    const uint64_t seed = UINT64_C(987654321);
    VAECProfile profile;
    VAECModel first;
    VAECModel replica;
    size_t index;
    int passed;

    make_synthetic_depth4(&profile);
    vae_c_model_init(&first);
    vae_c_model_init(&replica);
    passed =
        vae_c_model_allocate(&first, &profile, 2) == VAE_C_OP_OK &&
        vae_c_model_allocate(&replica, &profile, 2) == VAE_C_OP_OK &&
        first.parameter_count == 28 &&
        first.trainable_value_count == UINT64_C(939) &&
        vae_c_model_initialize_parameters_v1(&first, seed) == VAE_C_OP_OK &&
        vae_c_model_initialize_parameters_v1(&replica, seed) == VAE_C_OP_OK &&
        verify_initialized_model(&first, seed, 1) &&
        verify_initialized_model(&replica, seed, 1) &&
        models_semantically_equal(&first, &replica) &&
        kind_for_key(10) == TEST_INIT_HE &&
        kind_for_key(11) == TEST_INIT_ONE &&
        kind_for_key(12) == TEST_INIT_ZERO &&
        kind_for_key(24) == TEST_INIT_HE &&
        kind_for_key(25) == TEST_INIT_ONE &&
        kind_for_key(26) == TEST_INIT_ZERO;

    for (index = 0; passed && index < 28; index++) {
        if (first.parameters[index]->parameter_key != (uint64_t)(index + 1)) {
            passed = 0;
        }
    }
    vae_c_model_free(&first);
    vae_c_model_free(&replica);
    return
        passed &&
        vae_c_model_is_empty(&first) &&
        !vae_c_model_is_initialized(&first);
}


static int test_lifecycle_and_second_initialization(
    const VAECProfileCatalog *catalog
)
{
    const VAECProfile *profile = vae_c_profile_catalog_find(
        catalog,
        "mini75_smoke_v1"
    );
    VAECParameter *registry_before[VAE_C_MODEL_MAX_PARAMETERS];
    VAECModel model;
    size_t index;
    int passed;

    vae_c_model_init(&model);
    if (profile == NULL || !vae_c_model_is_empty(&model) ||
        vae_c_model_is_allocated(&model) ||
        vae_c_model_is_initialized(&model) ||
        vae_c_model_allocate(&model, profile, 2) != VAE_C_OP_OK) {

        return 0;
    }
    for (index = 0; index < model.parameter_count; index++) {
        registry_before[index] = model.parameters[index];
    }
    passed =
        vae_c_model_validate(&model) == VAE_C_OP_OK &&
        vae_c_model_is_allocated(&model) &&
        !vae_c_model_is_initialized(&model) &&
        vae_c_model_initialize_parameters_v1(
            &model,
            UINT64_C(5555)) == VAE_C_OP_OK &&
        verify_initialized_model(&model, UINT64_C(5555), 1) &&
        initialization_failure_preserves(&model, UINT64_C(7777));

    for (index = 0; passed && index < model.parameter_count; index++) {
        if (model.parameters[index] != registry_before[index]) {
            passed = 0;
        }
    }
    vae_c_model_free(&model);
    return
        passed &&
        vae_c_model_is_empty(&model) &&
        !vae_c_model_is_allocated(&model) &&
        !vae_c_model_is_initialized(&model) &&
        model.initialization_seed == UINT64_C(0) &&
        !model.parameters_initialized;
}


static int test_negative_states(const VAECProfileCatalog *catalog)
{
    const VAECProfile *profile = vae_c_profile_catalog_find(
        catalog,
        "mini75_smoke_v1"
    );
    VAECModel empty;
    VAECModel model;
    VAECParameter *saved_parameter;
    VAECOpStatus observed_status;
    uint64_t saved_key;
    uint64_t saved_count;
    size_t saved_size;
    float original_bias;
    int passed = 1;

    vae_c_model_init(&empty);
    if (vae_c_model_initialize_parameters_v1(NULL, UINT64_C(1)) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        !initialization_failure_preserves(&empty, UINT64_C(1))) {

        return 0;
    }
    vae_c_model_init(&model);
    if (profile == NULL ||
        vae_c_model_allocate(&model, profile, 2) != VAE_C_OP_OK) {

        return 0;
    }

    model.parameters_initialized = true;
    if (!initialization_failure_preserves(&model, UINT64_C(1))) {
        passed = 0;
    }
    model.parameters_initialized = false;

    model.initialization_seed = UINT64_C(1);
    if (!initialization_failure_preserves(&model, UINT64_C(1)) ||
        vae_c_model_validate(&model) == VAE_C_OP_OK) {

        passed = 0;
    }
    model.initialization_seed = UINT64_C(0);

    saved_size = model.encoder_convs[0].stride;
    model.encoder_convs[0].stride = 1;
    if (!initialization_failure_preserves(&model, UINT64_C(1))) {
        passed = 0;
    }
    model.encoder_convs[0].stride = saved_size;

    saved_parameter = model.parameters[1];
    model.parameters[1] = model.parameters[2];
    if (!initialization_failure_preserves(&model, UINT64_C(1))) {
        passed = 0;
    }
    model.parameters[1] = saved_parameter;

    saved_parameter = model.parameters[0];
    model.parameters[0] = model.parameters[1];
    if (!initialization_failure_preserves(&model, UINT64_C(1))) {
        passed = 0;
    }
    model.parameters[0] = saved_parameter;

    saved_key = model.parameters[0]->parameter_key;
    model.parameters[0]->parameter_key = UINT64_C(999);
    if (!initialization_failure_preserves(&model, UINT64_C(1))) {
        passed = 0;
    }
    model.parameters[0]->parameter_key = saved_key;

    saved_count = model.trainable_value_count;
    model.trainable_value_count = UINT64_MAX;
    if (!initialization_failure_preserves(&model, UINT64_C(1))) {
        passed = 0;
    }
    model.trainable_value_count = saved_count;

    saved_size = model.parameters[1]->shape.count;
    model.parameters[1]->shape.count = SIZE_MAX;
    if (!initialization_failure_preserves(&model, UINT64_C(1))) {
        passed = 0;
    }
    model.parameters[1]->shape.count = saved_size;

    saved_size = model.encoder_fc.weights.shape.dims[1];
    model.encoder_fc.weights.shape.dims[1] = 0;
    if (!initialization_failure_preserves(&model, UINT64_C(1))) {
        passed = 0;
    }
    model.encoder_fc.weights.shape.dims[1] = saved_size;

    saved_size = model.encoder_fc.weights.shape.dims[0];
    model.encoder_fc.weights.shape.dims[0] = 0;
    if (!initialization_failure_preserves(&model, UINT64_C(1))) {
        passed = 0;
    }
    model.encoder_fc.weights.shape.dims[0] = saved_size;

    saved_size = model.decoder_output_conv.weights.shape.dims[1];
    model.decoder_output_conv.weights.shape.dims[1] = SIZE_MAX;
    model.decoder_output_conv.weights.shape.dims[2] = 2;
    if (!initialization_failure_preserves(&model, UINT64_C(1))) {
        passed = 0;
    }
    model.decoder_output_conv.weights.shape.dims[1] = saved_size;
    model.decoder_output_conv.weights.shape.dims[2] =
        VAE_C_CONV2D_KERNEL_SIZE;

    saved_size = model.encoder_fc.weights.shape.dims[1];
    model.encoder_fc.weights.shape.dims[1] = SIZE_MAX;
    model.encoder_fc.weights.shape.dims[0] = 1;
    if (!initialization_failure_preserves(&model, UINT64_C(1))) {
        passed = 0;
    }
    model.encoder_fc.weights.shape.dims[1] = saved_size;
    model.encoder_fc.weights.shape.dims[0] = model.encoder_fc.out_features;

    model.parameters[0]->gradients[0] = 1.0f;
    if (!initialization_failure_preserves(&model, UINT64_C(1))) {
        passed = 0;
    }
    model.parameters[0]->gradients[0] = 0.0f;

    model.parameters[0]->adam_m[0] = 1.0f;
    if (!initialization_failure_preserves(&model, UINT64_C(1))) {
        passed = 0;
    }
    model.parameters[0]->adam_m[0] = 0.0f;

    model.parameters[0]->adam_v[0] = 1.0f;
    if (!initialization_failure_preserves(&model, UINT64_C(1))) {
        passed = 0;
    }
    model.parameters[0]->adam_v[0] = 0.0f;

    model.parameters[0]->values[0] = 1.0f;
    if (!initialization_failure_preserves(&model, UINT64_C(1))) {
        passed = 0;
    }
    model.parameters[0]->values[0] = 0.0f;

    if (!model.encoder_fc.with_bias ||
        model.encoder_fc.bias.values == NULL ||
        model.encoder_fc.bias.shape.count == 0 ||
        model.parameters_initialized ||
        model.initialization_seed != UINT64_C(0) ||
        vae_c_model_validate(&model) != VAE_C_OP_OK) {

        passed = 0;
    } else {
        original_bias = model.encoder_fc.bias.values[0];
        if (original_bias != 0.0f) {
            passed = 0;
        } else {
            model.encoder_fc.bias.values[0] = 1.0f;
            if (!initialization_failure_preserves_and_reports(
                    &model,
                    UINT64_C(1),
                    &observed_status) ||
                observed_status != VAE_C_OP_INVALID_STATE ||
                model.encoder_fc.bias.values[0] != 1.0f ||
                model.parameters_initialized ||
                model.initialization_seed != UINT64_C(0)) {

                passed = 0;
            }
            model.encoder_fc.bias.values[0] = original_bias;
            if (model.encoder_fc.bias.values[0] != 0.0f ||
                vae_c_model_validate(&model) != VAE_C_OP_OK ||
                !vae_c_model_is_allocated(&model) ||
                vae_c_model_is_initialized(&model)) {

                passed = 0;
            }
        }
    }

    model.encoder_batch_norms[0].gamma.values[0] = 2.0f;
    if (!initialization_failure_preserves(&model, UINT64_C(1))) {
        passed = 0;
    }
    model.encoder_batch_norms[0].gamma.values[0] = 1.0f;

    model.encoder_batch_norms[0].beta.values[0] = 1.0f;
    if (!initialization_failure_preserves(&model, UINT64_C(1))) {
        passed = 0;
    }
    model.encoder_batch_norms[0].beta.values[0] = 0.0f;

    model.encoder_batch_norms[0].running_mean[0] = 1.0f;
    if (!initialization_failure_preserves(&model, UINT64_C(1))) {
        passed = 0;
    }
    model.encoder_batch_norms[0].running_mean[0] = 0.0f;

    model.encoder_batch_norms[0].running_variance[0] = 2.0f;
    if (!initialization_failure_preserves(&model, UINT64_C(1))) {
        passed = 0;
    }
    model.encoder_batch_norms[0].running_variance[0] = 1.0f;

    model.encoder_batch_norms[0].cache_valid = true;
    if (!initialization_failure_preserves(&model, UINT64_C(1))) {
        passed = 0;
    }
    model.encoder_batch_norms[0].cache_valid = false;

    if (vae_c_model_validate(&model) != VAE_C_OP_OK ||
        !vae_c_model_is_allocated(&model) ||
        vae_c_model_is_initialized(&model) ||
        vae_c_model_initialize_parameters_v1(
            &model,
            UINT64_C(444)) != VAE_C_OP_OK ||
        !initialization_failure_preserves(&model, UINT64_C(445))) {

        passed = 0;
    }
    vae_c_model_free(&model);
    return passed && vae_c_model_is_empty(&model);
}


int main(int argc, char **argv)
{
    VAECProfileCatalog catalog;
    VAECConfigStatus load_status;
    int failures = 0;

    if (argc != 2) {
        fprintf(stderr, "usage: %s <vae_c_profiles_v1.csv>\n", argv[0]);
        return 2;
    }
    vae_c_profile_catalog_init(&catalog);
    load_status = vae_c_profile_catalog_load_csv(
        argv[1],
        &catalog,
        NULL,
        0
    );
    if (load_status != VAE_C_CONFIG_OK) {
        fprintf(stderr, "failed to load profile catalog\n");
        vae_c_profile_catalog_free(&catalog);
        return 2;
    }

    failures += report_check(
        "model_initialization_official_profiles",
        test_official_profiles(&catalog)
    );
    failures += report_check(
        "model_initialization_seed_behavior",
        test_seed_behavior(&catalog)
    );
    failures += report_check(
        "model_initialization_synthetic_depth4",
        test_synthetic_depth4()
    );
    failures += report_check(
        "model_initialization_lifecycle",
        test_lifecycle_and_second_initialization(&catalog)
    );
    failures += report_check(
        "model_initialization_negative_states",
        test_negative_states(&catalog)
    );

    vae_c_profile_catalog_free(&catalog);
    if (failures != 0) {
        fprintf(stderr, "%d VAE C MODEL INIT TEST(S) FAILED\n", failures);
        return 1;
    }
    printf("ALL VAE C MODEL INITIALIZATION TESTS PASSED\n");
    return 0;
}
