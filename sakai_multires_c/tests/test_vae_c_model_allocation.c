#include "vae_c_model.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>


static int report_check(const char *name, int passed)
{
    if (passed) {
        printf("[PASS] %s\n", name);
        return 0;
    }
    fprintf(stderr, "[FAIL] %s\n", name);
    return 1;
}


static int all_float_equal(
    const float *values,
    size_t count,
    float expected
)
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


static int parameter_state_matches(
    const VAECParameter *parameter,
    float expected_value
)
{
    return
        parameter != NULL &&
        parameter->shape.count > 0 &&
        all_float_equal(
            parameter->values,
            parameter->shape.count,
            expected_value) &&
        all_float_equal(
            parameter->gradients,
            parameter->shape.count,
            0.0f) &&
        all_float_equal(
            parameter->adam_m,
            parameter->shape.count,
            0.0f) &&
        all_float_equal(
            parameter->adam_v,
            parameter->shape.count,
            0.0f);
}


static int key_is_batch_norm_gamma(uint64_t key)
{
    return
        key == VAE_C_MODEL_PARAM_KEY_ENCODER_BN1_GAMMA ||
        key == VAE_C_MODEL_PARAM_KEY_ENCODER_BN2_GAMMA ||
        key == VAE_C_MODEL_PARAM_KEY_ENCODER_BN3_GAMMA ||
        key == VAE_C_MODEL_PARAM_KEY_ENCODER_BN4_GAMMA ||
        key == VAE_C_MODEL_PARAM_KEY_DECODER_FC_BN_GAMMA ||
        key == VAE_C_MODEL_PARAM_KEY_DECODER_BN1_GAMMA ||
        key == VAE_C_MODEL_PARAM_KEY_DECODER_BN2_GAMMA ||
        key == VAE_C_MODEL_PARAM_KEY_DECODER_BN3_GAMMA;
}


static const VAECParameter *expected_parameter_by_key(
    const VAECModel *model,
    uint64_t key
)
{
    switch (key) {
        case VAE_C_MODEL_PARAM_KEY_ENCODER_CONV1_WEIGHT:
            return &model->encoder_convs[0].weights;
        case VAE_C_MODEL_PARAM_KEY_ENCODER_BN1_GAMMA:
            return &model->encoder_batch_norms[0].gamma;
        case VAE_C_MODEL_PARAM_KEY_ENCODER_BN1_BETA:
            return &model->encoder_batch_norms[0].beta;
        case VAE_C_MODEL_PARAM_KEY_ENCODER_CONV2_WEIGHT:
            return &model->encoder_convs[1].weights;
        case VAE_C_MODEL_PARAM_KEY_ENCODER_BN2_GAMMA:
            return &model->encoder_batch_norms[1].gamma;
        case VAE_C_MODEL_PARAM_KEY_ENCODER_BN2_BETA:
            return &model->encoder_batch_norms[1].beta;
        case VAE_C_MODEL_PARAM_KEY_ENCODER_CONV3_WEIGHT:
            return &model->encoder_convs[2].weights;
        case VAE_C_MODEL_PARAM_KEY_ENCODER_BN3_GAMMA:
            return &model->encoder_batch_norms[2].gamma;
        case VAE_C_MODEL_PARAM_KEY_ENCODER_BN3_BETA:
            return &model->encoder_batch_norms[2].beta;
        case VAE_C_MODEL_PARAM_KEY_ENCODER_CONV4_WEIGHT:
            return &model->encoder_convs[3].weights;
        case VAE_C_MODEL_PARAM_KEY_ENCODER_BN4_GAMMA:
            return &model->encoder_batch_norms[3].gamma;
        case VAE_C_MODEL_PARAM_KEY_ENCODER_BN4_BETA:
            return &model->encoder_batch_norms[3].beta;
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
        case VAE_C_MODEL_PARAM_KEY_DECODER_CONV1_WEIGHT:
            return &model->decoder_hidden_convs[0].weights;
        case VAE_C_MODEL_PARAM_KEY_DECODER_BN1_GAMMA:
            return &model->decoder_hidden_batch_norms[0].gamma;
        case VAE_C_MODEL_PARAM_KEY_DECODER_BN1_BETA:
            return &model->decoder_hidden_batch_norms[0].beta;
        case VAE_C_MODEL_PARAM_KEY_DECODER_CONV2_WEIGHT:
            return &model->decoder_hidden_convs[1].weights;
        case VAE_C_MODEL_PARAM_KEY_DECODER_BN2_GAMMA:
            return &model->decoder_hidden_batch_norms[1].gamma;
        case VAE_C_MODEL_PARAM_KEY_DECODER_BN2_BETA:
            return &model->decoder_hidden_batch_norms[1].beta;
        case VAE_C_MODEL_PARAM_KEY_DECODER_CONV3_WEIGHT:
            return &model->decoder_hidden_convs[2].weights;
        case VAE_C_MODEL_PARAM_KEY_DECODER_BN3_GAMMA:
            return &model->decoder_hidden_batch_norms[2].gamma;
        case VAE_C_MODEL_PARAM_KEY_DECODER_BN3_BETA:
            return &model->decoder_hidden_batch_norms[2].beta;
        case VAE_C_MODEL_PARAM_KEY_DECODER_OUTPUT_WEIGHT:
            return &model->decoder_output_conv.weights;
        case VAE_C_MODEL_PARAM_KEY_DECODER_OUTPUT_BIAS:
            return &model->decoder_output_conv.bias;
        default:
            return NULL;
    }
}


static size_t expected_keys_for_depth(
    size_t depth,
    uint64_t keys[VAE_C_MODEL_MAX_PARAMETERS]
)
{
    size_t count = 0;
    size_t index;

    for (index = 0; index < depth; index++) {
        keys[count++] = UINT64_C(1) + UINT64_C(3) * (uint64_t)index;
        keys[count++] = UINT64_C(2) + UINT64_C(3) * (uint64_t)index;
        keys[count++] = UINT64_C(3) + UINT64_C(3) * (uint64_t)index;
    }
    keys[count++] = VAE_C_MODEL_PARAM_KEY_ENCODER_FC_WEIGHT;
    keys[count++] = VAE_C_MODEL_PARAM_KEY_ENCODER_FC_BIAS;
    keys[count++] = VAE_C_MODEL_PARAM_KEY_DECODER_FC_WEIGHT;
    keys[count++] = VAE_C_MODEL_PARAM_KEY_DECODER_FC_BN_GAMMA;
    keys[count++] = VAE_C_MODEL_PARAM_KEY_DECODER_FC_BN_BETA;
    for (index = 0; index < depth - 1; index++) {
        keys[count++] = UINT64_C(18) + UINT64_C(3) * (uint64_t)index;
        keys[count++] = UINT64_C(19) + UINT64_C(3) * (uint64_t)index;
        keys[count++] = UINT64_C(20) + UINT64_C(3) * (uint64_t)index;
    }
    keys[count++] = VAE_C_MODEL_PARAM_KEY_DECODER_OUTPUT_WEIGHT;
    keys[count++] = VAE_C_MODEL_PARAM_KEY_DECODER_OUTPUT_BIAS;
    return count;
}


static int registry_matches_model(const VAECModel *model)
{
    uint64_t keys[VAE_C_MODEL_MAX_PARAMETERS];
    uint64_t total = UINT64_C(0);
    size_t expected_count;
    size_t index;
    size_t other;

    expected_count = expected_keys_for_depth(model->config.conv_depth, keys);
    if (model->parameter_count != expected_count ||
        expected_count != 6 * model->config.conv_depth + 4) {

        return 0;
    }
    for (index = 0; index < expected_count; index++) {
        const VAECParameter *parameter = model->parameters[index];

        if (parameter == NULL ||
            parameter != expected_parameter_by_key(model, keys[index]) ||
            parameter->parameter_key != keys[index] ||
            parameter->parameter_key == UINT64_C(0) ||
            (index > 0 &&
             model->parameters[index - 1]->parameter_key >=
                 parameter->parameter_key) ||
            (const void *)parameter == (const void *)model->profile_id ||
            (const void *)parameter == (const void *)&model->config ||
            (uint64_t)parameter->shape.count > UINT64_MAX - total) {

            return 0;
        }
        for (other = 0; other < index; other++) {
            if (parameter == model->parameters[other]) {
                return 0;
            }
        }
        total += (uint64_t)parameter->shape.count;
    }
    for (index = expected_count;
         index < VAE_C_MODEL_MAX_PARAMETERS;
         index++) {

        if (model->parameters[index] != NULL) {
            return 0;
        }
    }
    return
        total == model->trainable_value_count &&
        total == model->config.trainable_parameters_bn1d;
}


static int batch_norm_initial_state(const VAECBatchNorm *layer)
{
    size_t count = layer->num_features;

    return
        count > 0 &&
        parameter_state_matches(&layer->gamma, 1.0f) &&
        parameter_state_matches(&layer->beta, 0.0f) &&
        all_float_equal(layer->running_mean, count, 0.0f) &&
        all_float_equal(layer->running_variance, count, 1.0f) &&
        all_float_equal(layer->batch_mean, count, 0.0f) &&
        all_float_equal(layer->batch_inv_std, count, 0.0f) &&
        all_float_equal(layer->work_mean, count, 0.0f) &&
        all_float_equal(layer->work_inv_std, count, 0.0f) &&
        all_float_equal(layer->work_unbiased_variance, count, 0.0f) &&
        layer->last_sample_count == 0 &&
        !layer->cache_valid;
}


static int model_initial_state_matches(const VAECModel *model)
{
    size_t index;

    if (model->vae_cache_generation != UINT64_C(0)) {
        return 0;
    }
    for (index = 0; index < model->parameter_count; index++) {
        float expected =
            key_is_batch_norm_gamma(
                model->parameters[index]->parameter_key)
                ? 1.0f
                : 0.0f;

        if (!parameter_state_matches(model->parameters[index], expected)) {
            return 0;
        }
    }
    for (index = 0; index < model->config.conv_depth; index++) {
        if (!batch_norm_initial_state(
                &model->encoder_batch_norms[index])) {

            return 0;
        }
    }
    if (!batch_norm_initial_state(&model->decoder_fc_batch_norm)) {
        return 0;
    }
    for (index = 0;
         index < model->config.decoder_hidden_conv_count;
         index++) {

        if (!batch_norm_initial_state(
                &model->decoder_hidden_batch_norms[index])) {

            return 0;
        }
    }
    return 1;
}


static int stack_pressure_preserves_registry(VAECModel *model)
{
    volatile unsigned char pressure[32768];
    size_t index;

    for (index = 0; index < sizeof(pressure); index++) {
        pressure[index] = (unsigned char)(index % 251);
    }
    return
        pressure[0] == 0 &&
        pressure[250] == 250 &&
        vae_c_model_validate(model) == VAE_C_OP_OK &&
        registry_matches_model(model);
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
    memcpy(
        profile->profile_status,
        "test_only",
        sizeof("test_only")
    );
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
    const uint64_t totals[4] = {
        UINT64_C(41572),
        UINT64_C(106116),
        UINT64_C(262852),
        UINT64_C(1035652)
    };
    size_t index;

    if (catalog == NULL || catalog->count != 4) {
        return 0;
    }
    for (index = 0; index < 4; index++) {
        const VAECProfile *profile = vae_c_profile_catalog_find(
            catalog,
            ids[index]
        );
        VAECModel model;

        vae_c_model_init(&model);
        if (profile == NULL ||
            vae_c_model_allocate(&model, profile, 4) != VAE_C_OP_OK ||
            model.config.conv_depth != 3 ||
            model.parameter_count != 22 ||
            model.trainable_value_count != totals[index] ||
            vae_c_model_validate(&model) != VAE_C_OP_OK ||
            !vae_c_model_is_allocated(&model) ||
            !registry_matches_model(&model) ||
            !model_initial_state_matches(&model) ||
            !stack_pressure_preserves_registry(&model)) {

            vae_c_model_free(&model);
            return 0;
        }
        vae_c_model_free(&model);
        vae_c_model_free(&model);
        if (!vae_c_model_is_empty(&model) ||
            vae_c_model_is_allocated(&model)) {

            return 0;
        }
    }
    return 1;
}


static int test_mini75_smoke_structure(
    const VAECProfileCatalog *catalog
)
{
    const uint64_t expected_keys[22] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9,
        13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 27, 28
    };
    const VAECProfile *profile = vae_c_profile_catalog_find(
        catalog,
        "mini75_smoke_v1"
    );
    VAECModel model;
    size_t index;
    int passed;

    vae_c_model_init(&model);
    if (profile == NULL ||
        vae_c_model_allocate(&model, profile, 4) != VAE_C_OP_OK) {

        return 0;
    }
    passed =
        model.encoder_convs[0].in_channels == 4 &&
        model.encoder_convs[0].out_channels == 8 &&
        model.encoder_convs[0].stride == 2 &&
        !model.encoder_convs[0].with_bias &&
        model.encoder_batch_norms[0].kind ==
            VAE_C_BATCH_NORM_KIND_2D &&
        model.encoder_batch_norms[0].num_features == 8 &&
        model.encoder_convs[1].in_channels == 8 &&
        model.encoder_convs[1].out_channels == 16 &&
        model.encoder_convs[1].stride == 2 &&
        !model.encoder_convs[1].with_bias &&
        model.encoder_batch_norms[1].num_features == 16 &&
        model.encoder_convs[2].in_channels == 16 &&
        model.encoder_convs[2].out_channels == 32 &&
        model.encoder_convs[2].stride == 2 &&
        !model.encoder_convs[2].with_bias &&
        model.encoder_batch_norms[2].num_features == 32 &&
        model.encoder_max_pool.capacity == 4 * 8 * 19 * 19 &&
        model.encoder_fc.in_features == 32 &&
        model.encoder_fc.out_features == 64 &&
        model.encoder_fc.with_bias &&
        model.decoder_fc.in_features == 32 &&
        model.decoder_fc.out_features == 800 &&
        !model.decoder_fc.with_bias &&
        model.decoder_fc_batch_norm.kind ==
            VAE_C_BATCH_NORM_KIND_1D &&
        model.decoder_fc_batch_norm.num_features == 800 &&
        model.decoder_hidden_convs[0].in_channels == 32 &&
        model.decoder_hidden_convs[0].out_channels == 16 &&
        model.decoder_hidden_convs[0].stride == 1 &&
        !model.decoder_hidden_convs[0].with_bias &&
        model.decoder_hidden_batch_norms[0].num_features == 16 &&
        model.decoder_hidden_convs[1].in_channels == 16 &&
        model.decoder_hidden_convs[1].out_channels == 8 &&
        model.decoder_hidden_convs[1].stride == 1 &&
        !model.decoder_hidden_convs[1].with_bias &&
        model.decoder_hidden_batch_norms[1].num_features == 8 &&
        model.decoder_output_conv.in_channels == 8 &&
        model.decoder_output_conv.out_channels == 4 &&
        model.decoder_output_conv.stride == 1 &&
        model.decoder_output_conv.with_bias;

    for (index = 0; passed && index < 22; index++) {
        if (model.parameters[index]->parameter_key !=
            expected_keys[index]) {

            passed = 0;
        }
    }
    passed =
        passed &&
        vae_c_conv2d_is_empty(&model.encoder_convs[3]) &&
        vae_c_batch_norm_is_empty(&model.encoder_batch_norms[3]) &&
        vae_c_conv2d_is_empty(&model.decoder_hidden_convs[2]) &&
        vae_c_batch_norm_is_empty(
            &model.decoder_hidden_batch_norms[2]) &&
        vae_c_model_validate(&model) == VAE_C_OP_OK;

    vae_c_model_free(&model);
    return passed;
}


static int test_mini112_primary_structure(
    const VAECProfileCatalog *catalog
)
{
    const VAECProfile *profile = vae_c_profile_catalog_find(
        catalog,
        "mini112_primary_v1"
    );
    VAECModel model;
    int passed;

    vae_c_model_init(&model);
    if (profile == NULL ||
        vae_c_model_allocate(&model, profile, 4) != VAE_C_OP_OK) {

        return 0;
    }
    passed =
        model.encoder_convs[0].in_channels == 4 &&
        model.encoder_convs[0].out_channels == 16 &&
        model.encoder_convs[1].in_channels == 16 &&
        model.encoder_convs[1].out_channels == 32 &&
        model.encoder_convs[2].in_channels == 32 &&
        model.encoder_convs[2].out_channels == 64 &&
        model.encoder_fc.in_features == 64 &&
        model.encoder_fc.out_features == 128 &&
        model.decoder_fc.in_features == 64 &&
        model.decoder_fc.out_features == 3136 &&
        model.decoder_hidden_convs[0].in_channels == 64 &&
        model.decoder_hidden_convs[0].out_channels == 32 &&
        model.decoder_hidden_convs[1].in_channels == 32 &&
        model.decoder_hidden_convs[1].out_channels == 16 &&
        model.decoder_output_conv.in_channels == 16 &&
        model.decoder_output_conv.out_channels == 4 &&
        model.encoder_max_pool.capacity == 4 * 16 * 28 * 28 &&
        model.parameter_count == 22 &&
        model.trainable_value_count == UINT64_C(262852) &&
        vae_c_model_validate(&model) == VAE_C_OP_OK;

    vae_c_model_free(&model);
    return passed;
}


static int test_synthetic_depth4(void)
{
    const size_t expected_encoder[4] = {16, 4, 2, 1};
    const size_t expected_targets[5] = {2, 4, 8, 16, 32};
    const size_t expected_hidden[3] = {4, 8, 16};
    VAECProfile profile;
    VAECConfig config;
    VAECModel model;
    size_t index;
    int passed = 1;

    make_synthetic_depth4(&profile);
    if (vae_c_config_build(&profile, &config, NULL, 0) !=
            VAE_C_CONFIG_OK) {

        return 0;
    }
    for (index = 0; index < 4; index++) {
        if (config.encoder_conv_sides[index] != expected_encoder[index]) {
            passed = 0;
        }
    }
    for (index = 0; index < 5; index++) {
        if (config.decoder_upsample_targets[index] !=
            expected_targets[index]) {

            passed = 0;
        }
    }
    for (index = 0; index < 3; index++) {
        if (config.decoder_hidden_conv_sides[index] !=
            expected_hidden[index]) {

            passed = 0;
        }
    }
    if (!passed || config.pool_side != 8 ||
        config.decoder_upsample_count != 5 ||
        config.decoder_hidden_conv_count != 3) {

        return 0;
    }

    vae_c_model_init(&model);
    if (vae_c_model_allocate(&model, &profile, 2) != VAE_C_OP_OK) {
        return 0;
    }
    passed =
        model.parameter_count == 28 &&
        model.trainable_value_count == UINT64_C(939) &&
        model.encoder_max_pool.capacity == 2 * 2 * 8 * 8 &&
        registry_matches_model(&model) &&
        model_initial_state_matches(&model) &&
        stack_pressure_preserves_registry(&model) &&
        vae_c_model_validate(&model) == VAE_C_OP_OK;
    for (index = 0; passed && index < 28; index++) {
        if (model.parameters[index]->parameter_key !=
            (uint64_t)(index + 1)) {

            passed = 0;
        }
    }
    for (index = 0; passed && index < 4; index++) {
        if (vae_c_conv2d_is_empty(&model.encoder_convs[index]) ||
            vae_c_batch_norm_is_empty(
                &model.encoder_batch_norms[index])) {

            passed = 0;
        }
    }
    for (index = 0; passed && index < 3; index++) {
        if (vae_c_conv2d_is_empty(
                &model.decoder_hidden_convs[index]) ||
            vae_c_batch_norm_is_empty(
                &model.decoder_hidden_batch_norms[index])) {

            passed = 0;
        }
    }
    vae_c_model_free(&model);
    return passed && vae_c_model_is_empty(&model);
}


static int expect_allocate_failure_preserves(
    const VAECProfile *profile,
    size_t max_batch_size
)
{
    VAECModel model;
    VAECModel model_before;
    VAECProfile profile_before;
    VAECOpStatus status;

    vae_c_model_init(&model);
    model_before = model;
    if (profile != NULL) {
        profile_before = *profile;
    }
    status = vae_c_model_allocate(&model, profile, max_batch_size);
    return
        status != VAE_C_OP_OK &&
        memcmp(&model, &model_before, sizeof(model)) == 0 &&
        vae_c_model_is_empty(&model) &&
        (profile == NULL ||
         memcmp(profile, &profile_before, sizeof(*profile)) == 0);
}


static int test_lifecycle_and_allocate_errors(
    const VAECProfileCatalog *catalog
)
{
    const VAECProfile *smoke = vae_c_profile_catalog_find(
        catalog,
        "mini75_smoke_v1"
    );
    VAECProfile invalid;
    VAECModel model;
    VAECModel before;
    int passed = 1;

    if (smoke == NULL) {
        return 0;
    }
    vae_c_model_init(&model);
    if (!vae_c_model_is_empty(&model) ||
        model.vae_cache_generation != UINT64_C(0) ||
        vae_c_model_is_allocated(&model) ||
        vae_c_model_is_empty(NULL) ||
        vae_c_model_is_allocated(NULL) ||
        vae_c_model_validate(NULL) != VAE_C_OP_INVALID_ARGUMENT ||
        vae_c_model_allocate(NULL, smoke, 2) !=
            VAE_C_OP_INVALID_ARGUMENT ||
        !expect_allocate_failure_preserves(NULL, 2) ||
        !expect_allocate_failure_preserves(smoke, 0)) {

        return 0;
    }

    invalid = *smoke;
    memset(invalid.profile_id, 0, sizeof(invalid.profile_id));
    passed = passed && expect_allocate_failure_preserves(&invalid, 2);

    invalid = *smoke;
    memset(invalid.profile_id, 'x', sizeof(invalid.profile_id));
    passed = passed && expect_allocate_failure_preserves(&invalid, 2);

    invalid = *smoke;
    invalid.conv_depth = 2;
    passed = passed && expect_allocate_failure_preserves(&invalid, 2);

    invalid = *smoke;
    invalid.channels[1] = 0;
    passed = passed && expect_allocate_failure_preserves(&invalid, 2);

    invalid = *smoke;
    invalid.latent_dim = SIZE_MAX / 2 + 1;
    passed = passed && expect_allocate_failure_preserves(&invalid, 2);

    invalid = *smoke;
    invalid.expected_trainable_parameters_bn1d += UINT64_C(1);
    passed = passed && expect_allocate_failure_preserves(&invalid, 2);

    invalid = *smoke;
    invalid.expected_forward_mac_per_sample += UINT64_C(1);
    passed = passed && expect_allocate_failure_preserves(&invalid, 2);

    passed = passed && expect_allocate_failure_preserves(smoke, SIZE_MAX);
    if (!passed ||
        vae_c_model_allocate(&model, smoke, 2) != VAE_C_OP_OK ||
        vae_c_model_is_empty(&model) ||
        !vae_c_model_is_allocated(&model)) {

        vae_c_model_free(&model);
        return 0;
    }
    before = model;
    if (vae_c_model_allocate(&model, smoke, 2) !=
            VAE_C_OP_OBJECT_NOT_EMPTY ||
        memcmp(&model, &before, sizeof(model)) != 0) {

        vae_c_model_free(&model);
        return 0;
    }
    vae_c_model_free(&model);
    if (!vae_c_model_is_empty(&model) ||
        model.vae_cache_generation != UINT64_C(0) ||
        vae_c_model_is_allocated(&model)) {

        return 0;
    }
    vae_c_model_free(&model);
    if (!vae_c_model_is_empty(&model) ||
        vae_c_model_allocate(&model, smoke, 2) != VAE_C_OP_OK ||
        !vae_c_model_is_allocated(&model)) {

        vae_c_model_free(&model);
        return 0;
    }
    vae_c_model_free(&model);

    vae_c_model_init(&model);
    model.vae_cache_generation = UINT64_C(1);
    if (vae_c_model_is_empty(&model)) {
        return 0;
    }
    vae_c_model_init(&model);
    model.max_batch_size = 1;
    return
        !vae_c_model_is_empty(&model) &&
        !vae_c_model_is_allocated(&model);
}


static int corrupted_model_is_rejected_without_mutation(VAECModel *model)
{
    VAECModel before = *model;

    return
        vae_c_model_validate(model) != VAE_C_OP_OK &&
        memcmp(model, &before, sizeof(*model)) == 0 &&
        !vae_c_model_is_allocated(model) &&
        memcmp(model, &before, sizeof(*model)) == 0 &&
        !vae_c_model_is_empty(model) &&
        memcmp(model, &before, sizeof(*model)) == 0;
}


static int test_validation_corruptions(
    const VAECProfileCatalog *catalog
)
{
    const size_t registry_index_a = 1;
    const size_t registry_index_b = 2;
    const VAECProfile *smoke = vae_c_profile_catalog_find(
        catalog,
        "mini75_smoke_v1"
    );
    VAECModel model;
    VAECParameter *saved_pointer;
    VAECParameter *saved_pointer_a;
    VAECParameter *saved_pointer_b;
    uint64_t saved_key;
    uint64_t saved_key_a;
    uint64_t saved_key_b;
    size_t original_num_features;
    size_t saved_size;
    bool saved_bool;
    VAECBatchNormKind saved_kind;
    int passed = 1;

    vae_c_model_init(&model);
    if (smoke == NULL ||
        vae_c_model_allocate(&model, smoke, 2) != VAE_C_OP_OK) {

        return 0;
    }

    saved_size = model.encoder_convs[0].stride;
    model.encoder_convs[0].stride = 1;
    passed = passed && corrupted_model_is_rejected_without_mutation(&model);
    model.encoder_convs[0].stride = saved_size;

    saved_bool = model.encoder_convs[0].with_bias;
    model.encoder_convs[0].with_bias = true;
    passed = passed && corrupted_model_is_rejected_without_mutation(&model);
    model.encoder_convs[0].with_bias = saved_bool;

    saved_kind = model.encoder_batch_norms[0].kind;
    model.encoder_batch_norms[0].kind = VAE_C_BATCH_NORM_KIND_1D;
    passed = passed && corrupted_model_is_rejected_without_mutation(&model);
    model.encoder_batch_norms[0].kind = saved_kind;

    original_num_features = model.encoder_batch_norms[0].num_features;
    if (vae_c_model_validate(&model) != VAE_C_OP_OK ||
        !vae_c_model_is_allocated(&model) ||
        original_num_features == 0 ||
        original_num_features == SIZE_MAX) {

        passed = 0;
    } else {
        model.encoder_batch_norms[0].num_features =
            original_num_features + 1;
        if (model.encoder_batch_norms[0].num_features == 0 ||
            model.encoder_batch_norms[0].num_features ==
                original_num_features ||
            !corrupted_model_is_rejected_without_mutation(&model) ||
            !corrupted_model_is_rejected_without_mutation(&model)) {

            passed = 0;
        }
        model.encoder_batch_norms[0].num_features = original_num_features;
        if (vae_c_model_validate(&model) != VAE_C_OP_OK ||
            !vae_c_model_is_allocated(&model)) {

            passed = 0;
        }
    }

    saved_size = model.encoder_max_pool.capacity;
    model.encoder_max_pool.capacity -= 1;
    passed = passed && corrupted_model_is_rejected_without_mutation(&model);
    model.encoder_max_pool.capacity = saved_size;

    saved_pointer = model.parameters[0];
    model.parameters[0] = NULL;
    passed = passed && corrupted_model_is_rejected_without_mutation(&model);
    model.parameters[0] = saved_pointer;

    saved_pointer = model.parameters[0];
    model.parameters[0] = model.parameters[1];
    passed = passed && corrupted_model_is_rejected_without_mutation(&model);
    model.parameters[0] = saved_pointer;

    if (registry_index_b >= model.parameter_count ||
        model.parameters[registry_index_a] == NULL ||
        model.parameters[registry_index_b] == NULL ||
        model.parameters[registry_index_a] ==
            model.parameters[registry_index_b] ||
        model.parameters[registry_index_a]->parameter_key >=
            model.parameters[registry_index_b]->parameter_key ||
        vae_c_model_validate(&model) != VAE_C_OP_OK ||
        !vae_c_model_is_allocated(&model) ||
        !registry_matches_model(&model)) {

        passed = 0;
    } else {
        saved_pointer_a = model.parameters[registry_index_a];
        saved_pointer_b = model.parameters[registry_index_b];
        saved_key_a = saved_pointer_a->parameter_key;
        saved_key_b = saved_pointer_b->parameter_key;

        model.parameters[registry_index_a] = saved_pointer_b;
        model.parameters[registry_index_b] = saved_pointer_a;
        if (model.parameters[registry_index_a]->parameter_key ==
                model.parameters[registry_index_b]->parameter_key ||
            model.parameters[registry_index_a]->parameter_key <=
                model.parameters[registry_index_b]->parameter_key ||
            saved_pointer_a->parameter_key != saved_key_a ||
            saved_pointer_b->parameter_key != saved_key_b ||
            !corrupted_model_is_rejected_without_mutation(&model) ||
            !corrupted_model_is_rejected_without_mutation(&model)) {

            passed = 0;
        }

        model.parameters[registry_index_a] = saved_pointer_a;
        model.parameters[registry_index_b] = saved_pointer_b;
        if (model.parameters[registry_index_a] != saved_pointer_a ||
            model.parameters[registry_index_b] != saved_pointer_b ||
            model.parameters[registry_index_a]->parameter_key >=
                model.parameters[registry_index_b]->parameter_key ||
            model.parameters[registry_index_a]->parameter_key != saved_key_a ||
            model.parameters[registry_index_b]->parameter_key != saved_key_b ||
            vae_c_model_validate(&model) != VAE_C_OP_OK ||
            !vae_c_model_is_allocated(&model) ||
            !registry_matches_model(&model)) {

            passed = 0;
        }
    }

    saved_pointer = model.parameters[0];
    model.parameters[0] = (VAECParameter *)(void *)model.profile_id;
    passed = passed && corrupted_model_is_rejected_without_mutation(&model);
    model.parameters[0] = saved_pointer;

    saved_key = model.parameters[1]->parameter_key;
    model.parameters[1]->parameter_key = model.parameters[0]->parameter_key;
    passed = passed && corrupted_model_is_rejected_without_mutation(&model);
    model.parameters[1]->parameter_key = saved_key;

    saved_size = model.parameter_count;
    model.parameter_count -= 1;
    passed = passed && corrupted_model_is_rejected_without_mutation(&model);
    model.parameter_count = saved_size;

    model.trainable_value_count -= UINT64_C(1);
    passed = passed && corrupted_model_is_rejected_without_mutation(&model);
    model.trainable_value_count += UINT64_C(1);

    model.encoder_convs[3].in_channels = 1;
    passed = passed && corrupted_model_is_rejected_without_mutation(&model);
    model.encoder_convs[3].in_channels = 0;

    model.config.pool_side += 1;
    passed = passed && corrupted_model_is_rejected_without_mutation(&model);
    model.config.pool_side -= 1;

    if (passed && vae_c_model_validate(&model) != VAE_C_OP_OK) {
        passed = 0;
    }
    vae_c_model_free(&model);

    vae_c_model_init(&model);
    model.allocated = true;
    passed =
        passed &&
        corrupted_model_is_rejected_without_mutation(&model);
    vae_c_model_init(&model);
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
        return 2;
    }

    failures += report_check(
        "model_official_profiles",
        test_official_profiles(&catalog)
    );
    failures += report_check(
        "model_mini75_smoke_structure",
        test_mini75_smoke_structure(&catalog)
    );
    failures += report_check(
        "model_mini112_primary_structure",
        test_mini112_primary_structure(&catalog)
    );
    failures += report_check(
        "model_synthetic_depth4",
        test_synthetic_depth4()
    );
    failures += report_check(
        "model_lifecycle_and_allocate_errors",
        test_lifecycle_and_allocate_errors(&catalog)
    );
    failures += report_check(
        "model_validation_corruptions",
        test_validation_corruptions(&catalog)
    );

    vae_c_profile_catalog_free(&catalog);
    if (failures != 0) {
        fprintf(stderr, "%d VAE C MODEL TEST(S) FAILED\n", failures);
        return 1;
    }
    printf("ALL VAE C MODEL ALLOCATION TESTS PASSED\n");
    return 0;
}
