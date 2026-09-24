#include "vae_c_config.h"
#include "vae_c_model.h"
#include "vae_c_model_workspace.h"

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


static int size_array_matches(
    const size_t *actual,
    const size_t *expected,
    size_t count
)
{
    size_t index;

    for (index = 0; index < count; index++) {
        if (actual[index] != expected[index]) {
            return 0;
        }
    }

    return 1;
}


static int profiles_semantically_equal(
    const VAECProfile *left,
    const VAECProfile *right
)
{
    if (left == NULL || right == NULL) {
        return 0;
    }

    return
        strcmp(left->profile_id, right->profile_id) == 0 &&
        strcmp(left->role, right->role) == 0 &&
        left->input_side == right->input_side &&
        left->conv_depth == right->conv_depth &&
        size_array_matches(
            left->channels,
            right->channels,
            VAE_C_MAX_CONV_DEPTH) &&
        left->latent_dim == right->latent_dim &&
        left->bottleneck_side == right->bottleneck_side &&
        left->expected_trainable_parameters_bn1d ==
            right->expected_trainable_parameters_bn1d &&
        left->expected_forward_mac_per_sample ==
            right->expected_forward_mac_per_sample &&
        strcmp(left->profile_status, right->profile_status) == 0;
}


static int tensor_shape_is(
    const VAECTensor *tensor,
    size_t rank,
    size_t dim0,
    size_t dim1,
    size_t dim2,
    size_t dim3
)
{
    const size_t expected[VAE_C_TENSOR_MAX_RANK] =
        {dim0, dim1, dim2, dim3};
    size_t index;

    if (tensor == NULL || tensor->shape.rank != rank) {
        return 0;
    }

    for (index = 0; index < VAE_C_TENSOR_MAX_RANK; index++) {
        if (tensor->shape.dims[index] != expected[index]) {
            return 0;
        }
    }

    return 1;
}


static int load_catalog(
    const char *path,
    VAECProfileCatalog *catalog
)
{
    char error_message[256];

    error_message[0] = '\0';
    return
        vae_c_profile_catalog_load_csv(
            path,
            catalog,
            error_message,
            sizeof(error_message)) == VAE_C_CONFIG_OK &&
        error_message[0] == '\0';
}


static int test_catalog_versioning(
    const VAECProfileCatalog *v1,
    const VAECProfileCatalog *v2
)
{
    static const char *expected_ids[5] = {
        "mini75_smoke_v1",
        "mini75_experiment_v1",
        "mini112_primary_v1",
        "mini112_scaleup_v1",
        "mini224_bridge_v1"
    };
    size_t index;

    if (v1 == NULL || v2 == NULL || v1->count != 4 || v2->count != 5) {
        return 0;
    }

    for (index = 0; index < 5; index++) {
        if (strcmp(v2->items[index].profile_id, expected_ids[index]) != 0 ||
            vae_c_profile_catalog_find(v2, expected_ids[index]) !=
                &v2->items[index]) {

            return 0;
        }
    }

    for (index = 0; index < 4; index++) {
        if (!profiles_semantically_equal(&v1->items[index], &v2->items[index])) {
            return 0;
        }
    }

    return vae_c_profile_catalog_find(v1, "mini224_bridge_v1") == NULL;
}


static int test_bridge_config(const VAECProfileCatalog *v2)
{
    static const size_t expected_channels[VAE_C_MAX_CONV_DEPTH] =
        {16, 32, 64, 128};
    static const size_t expected_encoder[VAE_C_MAX_CONV_DEPTH] =
        {112, 28, 14, 7};
    static const size_t expected_targets[VAE_C_MAX_DECODER_UPSAMPLES] =
        {14, 28, 56, 112, 224};
    static const size_t expected_hidden[VAE_C_MAX_DECODER_HIDDEN_CONVS] =
        {28, 56, 112};
    const VAECProfile *profile;
    VAECConfig config;
    char error_message[256];

    profile = vae_c_profile_catalog_find(v2, "mini224_bridge_v1");
    if (profile == NULL ||
        strcmp(profile->role, "project_bridge_candidate") != 0 ||
        strcmp(profile->profile_status, "candidate_pending_224_smoke") != 0 ||
        profile->input_side != 224 || profile->conv_depth != 4 ||
        !size_array_matches(
            profile->channels,
            expected_channels,
            VAE_C_MAX_CONV_DEPTH) ||
        profile->latent_dim != 64 || profile->bottleneck_side != 7 ||
        profile->expected_trainable_parameters_bn1d != UINT64_C(625860) ||
        profile->expected_forward_mac_per_sample != UINT64_C(220790784)) {

        return 0;
    }

    error_message[0] = '\0';
    if (vae_c_config_build(
            profile,
            &config,
            error_message,
            sizeof(error_message)) != VAE_C_CONFIG_OK ||
        error_message[0] != '\0') {

        return 0;
    }

    return
        config.input_channels == 4 && config.input_side == 224 &&
        config.conv_depth == 4 &&
        size_array_matches(
            config.channels,
            expected_channels,
            VAE_C_MAX_CONV_DEPTH) &&
        size_array_matches(
            config.encoder_conv_sides,
            expected_encoder,
            VAE_C_MAX_CONV_DEPTH) &&
        config.pool_side == 56 && config.latent_dim == 64 &&
        config.bottleneck_side == 7 && config.decoder_fc_features == 6272 &&
        config.decoder_upsample_count == 5 &&
        size_array_matches(
            config.decoder_upsample_targets,
            expected_targets,
            VAE_C_MAX_DECODER_UPSAMPLES) &&
        config.decoder_hidden_conv_count == 3 &&
        size_array_matches(
            config.decoder_hidden_conv_sides,
            expected_hidden,
            VAE_C_MAX_DECODER_HIDDEN_CONVS) &&
        config.values_per_sample == UINT64_C(200704) &&
        config.trainable_parameters_bn1d == UINT64_C(625860) &&
        config.forward_mac_per_sample == UINT64_C(220790784);
}


static int test_bridge_workspace(const VAECProfileCatalog *v2)
{
    const VAECProfile *profile =
        vae_c_profile_catalog_find(v2, "mini224_bridge_v1");
    VAECModel model;
    VAECModelWorkspace workspace;
    int passed;

    vae_c_model_init(&model);
    vae_c_model_workspace_init(&workspace);

    if (profile == NULL ||
        vae_c_model_allocate(&model, profile, 2) != VAE_C_OP_OK ||
        vae_c_model_workspace_allocate(&model, 2, &workspace) != VAE_C_OP_OK) {

        vae_c_model_workspace_free(&workspace);
        vae_c_model_free(&model);
        return 0;
    }

    passed =
        vae_c_model_workspace_validate(&model, &workspace) == VAE_C_OP_OK &&
        workspace.owned_tensor_count == 38 &&
        workspace.storage_float_count == 15155072 &&
        workspace.storage_bytes == 60620288 &&
        tensor_shape_is(&workspace.input, 4, 2, 4, 224, 224) &&
        tensor_shape_is(
            &workspace.encoder_conv_outputs[0], 4, 2, 16, 112, 112) &&
        tensor_shape_is(
            &workspace.encoder_pool_output, 4, 2, 16, 56, 56) &&
        tensor_shape_is(
            &workspace.encoder_conv_outputs[1], 4, 2, 32, 28, 28) &&
        tensor_shape_is(
            &workspace.encoder_conv_outputs[2], 4, 2, 64, 14, 14) &&
        tensor_shape_is(
            &workspace.encoder_conv_outputs[3], 4, 2, 128, 7, 7) &&
        tensor_shape_is(
            &workspace.encoder_global_avg_output, 2, 2, 128, 0, 0) &&
        tensor_shape_is(
            &workspace.encoder_packed_output, 2, 2, 128, 0, 0) &&
        tensor_shape_is(&workspace.mu, 2, 2, 64, 0, 0) &&
        tensor_shape_is(&workspace.logvar, 2, 2, 64, 0, 0) &&
        tensor_shape_is(
            &workspace.decoder_fc_output, 2, 2, 6272, 0, 0) &&
        tensor_shape_is(
            &workspace.decoder_reshape_view, 4, 2, 128, 7, 7) &&
        tensor_shape_is(
            &workspace.decoder_upsample_outputs[0], 4, 2, 128, 14, 14) &&
        tensor_shape_is(
            &workspace.decoder_upsample_outputs[1], 4, 2, 128, 28, 28) &&
        tensor_shape_is(
            &workspace.decoder_hidden_conv_outputs[0], 4, 2, 64, 28, 28) &&
        tensor_shape_is(
            &workspace.decoder_upsample_outputs[2], 4, 2, 64, 56, 56) &&
        tensor_shape_is(
            &workspace.decoder_hidden_conv_outputs[1], 4, 2, 32, 56, 56) &&
        tensor_shape_is(
            &workspace.decoder_upsample_outputs[3], 4, 2, 32, 112, 112) &&
        tensor_shape_is(
            &workspace.decoder_hidden_conv_outputs[2], 4, 2, 16, 112, 112) &&
        tensor_shape_is(
            &workspace.decoder_upsample_outputs[4], 4, 2, 16, 224, 224) &&
        tensor_shape_is(&workspace.logits, 4, 2, 4, 224, 224);

    vae_c_model_workspace_free(&workspace);
    vae_c_model_free(&model);
    return passed;
}


static int test_invalid_depth4_rejected(const VAECProfileCatalog *v2)
{
    const VAECProfile *profile =
        vae_c_profile_catalog_find(v2, "mini224_bridge_v1");
    VAECProfile invalid;
    VAECConfig config;

    if (profile == NULL) {
        return 0;
    }

    invalid = *profile;
    invalid.channels[3] = 0;
    if (vae_c_config_build(&invalid, &config, NULL, 0) !=
        VAE_C_CONFIG_VALUE_OUT_OF_RANGE) {

        return 0;
    }

    invalid = *profile;
    invalid.conv_depth = VAE_C_MAX_CONV_DEPTH + 1;
    return vae_c_config_build(&invalid, &config, NULL, 0) ==
        VAE_C_CONFIG_VALUE_OUT_OF_RANGE;
}


int main(int argc, char **argv)
{
    VAECProfileCatalog v1;
    VAECProfileCatalog v2;
    int failures = 0;

    if (argc != 3) {
        fprintf(
            stderr,
            "usage: %s <vae_c_profiles_v1.csv> <vae_c_profiles_v2.csv>\n",
            argc > 0 ? argv[0] : "test_vae_c_profiles_v2");
        return 2;
    }

    vae_c_profile_catalog_init(&v1);
    vae_c_profile_catalog_init(&v2);
    if (!load_catalog(argv[1], &v1) || !load_catalog(argv[2], &v2)) {
        fprintf(stderr, "failed to load VAE C profile catalogs\n");
        vae_c_profile_catalog_free(&v2);
        vae_c_profile_catalog_free(&v1);
        return 1;
    }

    failures += report_check(
        "v1_v2_catalog_versioning",
        test_catalog_versioning(&v1, &v2));
    failures += report_check(
        "mini224_bridge_config_known_answer",
        test_bridge_config(&v2));
    failures += report_check(
        "mini224_bridge_workspace_known_answer",
        test_bridge_workspace(&v2));
    failures += report_check(
        "invalid_depth4_rejected",
        test_invalid_depth4_rejected(&v2));

    vae_c_profile_catalog_free(&v2);
    vae_c_profile_catalog_free(&v1);

    if (failures != 0) {
        fprintf(stderr, "%d VAE C PROFILE V2 TEST(S) FAILED\n", failures);
        return 1;
    }

    printf("ALL VAE C PROFILE V2 TESTS PASSED\n");
    return 0;
}
