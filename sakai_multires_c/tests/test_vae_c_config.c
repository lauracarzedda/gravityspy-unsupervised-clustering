#include "vae_c_config.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define VAE_C_TEST_HEADER                                                   \
    "profile_id,role,input_side,conv_depth,c1,c2,c3,c4,latent_dim,"         \
    "bottleneck_side,trainable_parameters_bn1d,forward_mac_per_sample,"     \
    "status"

#define VAE_C_TEST_SMOKE_ROW                                                \
    "mini75_smoke_v1,engineering_smoke,75,3,8,16,32,0,32,5,41572,"        \
    "5620896,primary_smoke"


typedef struct {
    const char *profile_id;
    const char *role;
    size_t input_side;
    size_t conv_depth;
    size_t channels[VAE_C_MAX_CONV_DEPTH];
    size_t latent_dim;
    size_t bottleneck_side;
    uint64_t parameters;
    uint64_t mac;
    const char *status;
} ExpectedProfile;


static int report_check(const char *name, int passed)
{
    if (passed) {
        printf("[PASS] %s\n", name);
        return 0;
    } else {
        fprintf(stderr, "[FAIL] %s\n", name);
        return 1;
    }
}


static int make_fixture_path(
    const char *fixture_directory,
    const char *file_name,
    char *path,
    size_t path_capacity
)
{
    int length;

    if (fixture_directory == NULL ||
        file_name == NULL ||
        path == NULL ||
        path_capacity == 0) {

        return 0;
    }

    length = snprintf(
        path,
        path_capacity,
        "%s/%s",
        fixture_directory,
        file_name
    );

    return length >= 0 && (size_t)length < path_capacity;
}


static int write_fixture(const char *path, const char *contents)
{
    FILE *stream;
    size_t length;
    int close_status;

    if (path == NULL || contents == NULL) {
        return 0;
    }

    stream = fopen(path, "wb");
    if (stream == NULL) {
        return 0;
    }

    length = strlen(contents);
    if (fwrite(contents, 1, length, stream) != length) {
        (void)fclose(stream);
        return 0;
    }

    close_status = fclose(stream);
    return close_status == 0;
}


static int expect_load_failure(
    const char *fixture_directory,
    const char *file_name,
    const char *contents,
    VAECConfigStatus expected_status
)
{
    VAECProfileCatalog catalog;
    VAECConfigStatus status;
    char path[1024];
    char error_message[256];
    int result;

    vae_c_profile_catalog_init(&catalog);

    if (!make_fixture_path(
            fixture_directory,
            file_name,
            path,
            sizeof(path))) {

        return 0;
    }

    if (contents != NULL && !write_fixture(path, contents)) {
        return 0;
    }

    error_message[0] = '\0';
    status = vae_c_profile_catalog_load_csv(
        path,
        &catalog,
        error_message,
        sizeof(error_message)
    );

    result =
        status == expected_status &&
        catalog.items == NULL &&
        catalog.count == 0 &&
        error_message[0] != '\0';

    vae_c_profile_catalog_free(&catalog);
    return result;
}


static int profile_matches(
    const VAECProfile *profile,
    const ExpectedProfile *expected
)
{
    size_t index;

    if (profile == NULL || expected == NULL) {
        return 0;
    }

    if (strcmp(profile->profile_id, expected->profile_id) != 0 ||
        strcmp(profile->role, expected->role) != 0 ||
        profile->input_side != expected->input_side ||
        profile->conv_depth != expected->conv_depth ||
        profile->latent_dim != expected->latent_dim ||
        profile->bottleneck_side != expected->bottleneck_side ||
        profile->expected_trainable_parameters_bn1d !=
            expected->parameters ||
        profile->expected_forward_mac_per_sample != expected->mac ||
        strcmp(profile->profile_status, expected->status) != 0) {

        return 0;
    }

    for (index = 0; index < VAE_C_MAX_CONV_DEPTH; index++) {
        if (profile->channels[index] != expected->channels[index]) {
            return 0;
        }
    }

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


static int config_matches_depth3(
    const VAECConfig *config,
    size_t input_side,
    const size_t channels[VAE_C_MAX_CONV_DEPTH],
    size_t latent_dim,
    size_t bottleneck_side,
    uint64_t expected_parameters,
    uint64_t expected_mac
)
{
    const size_t encoder_sides_75[VAE_C_MAX_CONV_DEPTH] =
        {38, 10, 5, 0};
    const size_t encoder_sides_112[VAE_C_MAX_CONV_DEPTH] =
        {56, 14, 7, 0};
    const size_t targets_75[VAE_C_MAX_DECODER_UPSAMPLES] =
        {10, 19, 38, 75, 0};
    const size_t targets_112[VAE_C_MAX_DECODER_UPSAMPLES] =
        {14, 28, 56, 112, 0};
    const size_t hidden_75[VAE_C_MAX_DECODER_HIDDEN_CONVS] =
        {19, 38, 0};
    const size_t hidden_112[VAE_C_MAX_DECODER_HIDDEN_CONVS] =
        {28, 56, 0};
    const size_t *encoder_sides;
    const size_t *targets;
    const size_t *hidden_sides;
    size_t expected_pool_side;
    uint64_t expected_values;

    if (config == NULL) {
        return 0;
    }

    if (input_side == 75) {
        encoder_sides = encoder_sides_75;
        targets = targets_75;
        hidden_sides = hidden_75;
        expected_pool_side = 19;
        expected_values = 22500;
    } else if (input_side == 112) {
        encoder_sides = encoder_sides_112;
        targets = targets_112;
        hidden_sides = hidden_112;
        expected_pool_side = 28;
        expected_values = 50176;
    } else {
        return 0;
    }

    return
        config->input_channels == VAE_C_INPUT_CHANNELS &&
        config->input_side == input_side &&
        config->conv_depth == 3 &&
        size_array_matches(
            config->channels,
            channels,
            VAE_C_MAX_CONV_DEPTH) &&
        size_array_matches(
            config->encoder_conv_sides,
            encoder_sides,
            VAE_C_MAX_CONV_DEPTH) &&
        config->pool_side == expected_pool_side &&
        config->latent_dim == latent_dim &&
        config->bottleneck_side == bottleneck_side &&
        config->decoder_fc_features ==
            channels[2] * bottleneck_side * bottleneck_side &&
        config->decoder_upsample_count == 4 &&
        size_array_matches(
            config->decoder_upsample_targets,
            targets,
            VAE_C_MAX_DECODER_UPSAMPLES) &&
        config->decoder_hidden_conv_count == 2 &&
        size_array_matches(
            config->decoder_hidden_conv_sides,
            hidden_sides,
            VAE_C_MAX_DECODER_HIDDEN_CONVS) &&
        config->values_per_sample == expected_values &&
        config->trainable_parameters_bn1d == expected_parameters &&
        config->forward_mac_per_sample == expected_mac;
}


static int test_official_catalog(
    const char *official_csv,
    VAECProfileCatalog *catalog
)
{
    static const ExpectedProfile expected[] = {
        {
            "mini75_smoke_v1",
            "engineering_smoke",
            75,
            3,
            {8, 16, 32, 0},
            32,
            5,
            UINT64_C(41572),
            UINT64_C(5620896),
            "primary_smoke"
        },
        {
            "mini75_experiment_v1",
            "optional_full_split_experiment",
            75,
            3,
            {16, 32, 64, 0},
            32,
            5,
            UINT64_C(106116),
            UINT64_C(18356544),
            "optional_after_smoke"
        },
        {
            "mini112_primary_v1",
            "primary_c_experiment",
            112,
            3,
            {16, 32, 64, 0},
            64,
            7,
            UINT64_C(262852),
            UINT64_C(39948288),
            "primary"
        },
        {
            "mini112_scaleup_v1",
            "optional_scaleup",
            112,
            3,
            {32, 64, 128, 0},
            128,
            7,
            UINT64_C(1035652),
            UINT64_C(141729792),
            "optional_after_benchmark"
        }
    };
    VAECConfigStatus status;
    char error_message[256];
    size_t index;

    error_message[0] = '\0';
    status = vae_c_profile_catalog_load_csv(
        official_csv,
        catalog,
        error_message,
        sizeof(error_message)
    );

    if (status != VAE_C_CONFIG_OK ||
        error_message[0] != '\0' ||
        catalog->count != sizeof(expected) / sizeof(expected[0])) {

        return 0;
    }

    for (index = 0; index < catalog->count; index++) {
        if (!profile_matches(&catalog->items[index], &expected[index]) ||
            vae_c_profile_catalog_find(
                catalog,
                expected[index].profile_id) != &catalog->items[index]) {

            return 0;
        }
    }

    return
        vae_c_profile_catalog_find(catalog, "missing_profile") == NULL &&
        vae_c_profile_catalog_find(NULL, "mini75_smoke_v1") == NULL &&
        vae_c_profile_catalog_find(catalog, NULL) == NULL;
}


static int test_official_configs(const VAECProfileCatalog *catalog)
{
    size_t index;

    if (catalog == NULL || catalog->count != 4) {
        return 0;
    }

    for (index = 0; index < catalog->count; index++) {
        VAECConfig config;
        VAECConfigStatus status;
        char error_message[256];
        const VAECProfile *profile;

        profile = &catalog->items[index];
        memset(&config, 0xA5, sizeof(config));
        error_message[0] = '\0';
        status = vae_c_config_build(
            profile,
            &config,
            error_message,
            sizeof(error_message)
        );

        if (status != VAE_C_CONFIG_OK ||
            error_message[0] != '\0' ||
            !config_matches_depth3(
                &config,
                profile->input_side,
                profile->channels,
                profile->latent_dim,
                profile->bottleneck_side,
                profile->expected_trainable_parameters_bn1d,
                profile->expected_forward_mac_per_sample)) {

            return 0;
        }
    }

    return 1;
}


static int test_depth4_config(void)
{
    const size_t expected_encoder[VAE_C_MAX_CONV_DEPTH] =
        {112, 28, 14, 7};
    const size_t expected_targets[VAE_C_MAX_DECODER_UPSAMPLES] =
        {14, 28, 56, 112, 224};
    const size_t expected_hidden[VAE_C_MAX_DECODER_HIDDEN_CONVS] =
        {28, 56, 112};
    VAECProfile profile;
    VAECConfig config;
    VAECConfigStatus status;
    char error_message[256];

    memset(&profile, 0, sizeof(profile));
    (void)snprintf(
        profile.profile_id,
        sizeof(profile.profile_id),
        "%s",
        "synthetic_depth4"
    );
    (void)snprintf(
        profile.role,
        sizeof(profile.role),
        "%s",
        "synthetic"
    );
    (void)snprintf(
        profile.profile_status,
        sizeof(profile.profile_status),
        "%s",
        "test_only"
    );

    profile.input_side = 224;
    profile.conv_depth = 4;
    profile.channels[0] = 64;
    profile.channels[1] = 128;
    profile.channels[2] = 256;
    profile.channels[3] = 512;
    profile.latent_dim = 512;
    profile.bottleneck_side = 7;
    profile.expected_trainable_parameters_bn1d =
        UINT64_C(16524548);
    profile.expected_forward_mac_per_sample =
        UINT64_C(3105816576);

    error_message[0] = '\0';
    status = vae_c_config_build(
        &profile,
        &config,
        error_message,
        sizeof(error_message)
    );

    return
        status == VAE_C_CONFIG_OK &&
        error_message[0] == '\0' &&
        config.input_channels == 4 &&
        config.input_side == 224 &&
        config.conv_depth == 4 &&
        size_array_matches(
            config.channels,
            profile.channels,
            VAE_C_MAX_CONV_DEPTH) &&
        size_array_matches(
            config.encoder_conv_sides,
            expected_encoder,
            VAE_C_MAX_CONV_DEPTH) &&
        config.pool_side == 56 &&
        config.latent_dim == 512 &&
        config.bottleneck_side == 7 &&
        config.decoder_fc_features == 25088 &&
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
        config.trainable_parameters_bn1d == UINT64_C(16524548) &&
        config.forward_mac_per_sample == UINT64_C(3105816576);
}


static int test_valid_line_endings(const char *fixture_directory)
{
    static const char crlf_contents[] =
        VAE_C_TEST_HEADER "\r\n"
        VAE_C_TEST_SMOKE_ROW "\r\n";
    static const char no_final_newline_contents[] =
        VAE_C_TEST_HEADER "\n"
        VAE_C_TEST_SMOKE_ROW;
    const char *file_names[] = {
        "valid_crlf.csv",
        "valid_no_final_newline.csv"
    };
    const char *contents[] = {
        crlf_contents,
        no_final_newline_contents
    };
    size_t index;

    for (index = 0; index < sizeof(file_names) / sizeof(file_names[0]);
         index++) {
        char path[1024];
        char error_message[256];
        VAECProfileCatalog catalog;
        VAECConfigStatus status;

        vae_c_profile_catalog_init(&catalog);

        if (!make_fixture_path(
                fixture_directory,
                file_names[index],
                path,
                sizeof(path)) ||
            !write_fixture(path, contents[index])) {

            return 0;
        }

        error_message[0] = '\0';
        status = vae_c_profile_catalog_load_csv(
            path,
            &catalog,
            error_message,
            sizeof(error_message)
        );

        if (status != VAE_C_CONFIG_OK ||
            error_message[0] != '\0' ||
            catalog.count != 1 ||
            strcmp(catalog.items[0].profile_id, "mini75_smoke_v1") != 0) {

            vae_c_profile_catalog_free(&catalog);
            return 0;
        }

        vae_c_profile_catalog_free(&catalog);
    }

    return 1;
}


static int test_malformed_csv(const char *fixture_directory)
{
    static const struct {
        const char *name;
        const char *contents;
        VAECConfigStatus status;
    } cases[] = {
        {
            "bad_header.csv",
            "not,the,official,header\n" VAE_C_TEST_SMOKE_ROW "\n",
            VAE_C_CONFIG_CSV_FORMAT_ERROR
        },
        {
            "too_few_fields.csv",
            VAE_C_TEST_HEADER "\n"
            "mini75_smoke_v1,engineering_smoke,75,3,8,16,32,0,32,5,"
            "41572,5620896\n",
            VAE_C_CONFIG_CSV_FORMAT_ERROR
        },
        {
            "too_many_fields.csv",
            VAE_C_TEST_HEADER "\n" VAE_C_TEST_SMOKE_ROW ",extra\n",
            VAE_C_CONFIG_CSV_FORMAT_ERROR
        },
        {
            "empty_profile_id.csv",
            VAE_C_TEST_HEADER "\n"
            ",engineering_smoke,75,3,8,16,32,0,32,5,41572,5620896,"
            "primary_smoke\n",
            VAE_C_CONFIG_CSV_FORMAT_ERROR
        },
        {
            "empty_role.csv",
            VAE_C_TEST_HEADER "\n"
            "mini75_smoke_v1,,75,3,8,16,32,0,32,5,41572,5620896,"
            "primary_smoke\n",
            VAE_C_CONFIG_CSV_FORMAT_ERROR
        },
        {
            "profile_id_too_long.csv",
            VAE_C_TEST_HEADER "\n"
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,"
            "engineering_smoke,75,3,8,16,32,0,32,5,41572,5620896,"
            "primary_smoke\n",
            VAE_C_CONFIG_VALUE_OUT_OF_RANGE
        },
        {
            "nonnumeric.csv",
            VAE_C_TEST_HEADER "\n"
            "mini75_smoke_v1,engineering_smoke,75x,3,8,16,32,0,32,5,"
            "41572,5620896,primary_smoke\n",
            VAE_C_CONFIG_CSV_FORMAT_ERROR
        },
        {
            "negative.csv",
            VAE_C_TEST_HEADER "\n"
            "mini75_smoke_v1,engineering_smoke,-75,3,8,16,32,0,32,5,"
            "41572,5620896,primary_smoke\n",
            VAE_C_CONFIG_CSV_FORMAT_ERROR
        },
        {
            "plus_sign.csv",
            VAE_C_TEST_HEADER "\n"
            "mini75_smoke_v1,engineering_smoke,+75,3,8,16,32,0,32,5,"
            "41572,5620896,primary_smoke\n",
            VAE_C_CONFIG_CSV_FORMAT_ERROR
        },
        {
            "overflow.csv",
            VAE_C_TEST_HEADER "\n"
            "mini75_smoke_v1,engineering_smoke,18446744073709551616,3,"
            "8,16,32,0,32,5,41572,5620896,primary_smoke\n",
            VAE_C_CONFIG_VALUE_OUT_OF_RANGE
        },
        {
            "duplicate.csv",
            VAE_C_TEST_HEADER "\n"
            VAE_C_TEST_SMOKE_ROW "\n"
            VAE_C_TEST_SMOKE_ROW "\n",
            VAE_C_CONFIG_DUPLICATE_PROFILE_ID
        },
        {
            "whitespace.csv",
            VAE_C_TEST_HEADER "\n"
            "mini75_smoke_v1, engineering_smoke,75,3,8,16,32,0,32,5,"
            "41572,5620896,primary_smoke\n",
            VAE_C_CONFIG_CSV_FORMAT_ERROR
        },
        {
            "quotes.csv",
            VAE_C_TEST_HEADER "\n"
            "\"mini75_smoke_v1\",engineering_smoke,75,3,8,16,32,0,32,"
            "5,41572,5620896,primary_smoke\n",
            VAE_C_CONFIG_CSV_FORMAT_ERROR
        },
        {
            "blank_line.csv",
            VAE_C_TEST_HEADER "\n" VAE_C_TEST_SMOKE_ROW "\n\n",
            VAE_C_CONFIG_CSV_FORMAT_ERROR
        }
    };
    size_t index;
    char long_path[1024];
    char *long_contents;
    size_t header_length;
    size_t long_length;
    int result;

    for (index = 0; index < sizeof(cases) / sizeof(cases[0]); index++) {
        if (!expect_load_failure(
                fixture_directory,
                cases[index].name,
                cases[index].contents,
                cases[index].status)) {

            return 0;
        }
    }

    if (!make_fixture_path(
            fixture_directory,
            "line_too_long.csv",
            long_path,
            sizeof(long_path))) {

        return 0;
    }

    header_length = strlen(VAE_C_TEST_HEADER);
    long_length = header_length + 1 + 3000 + 1;
    long_contents = (char *)malloc(long_length + 1);
    if (long_contents == NULL) {
        return 0;
    }

    memcpy(long_contents, VAE_C_TEST_HEADER, header_length);
    long_contents[header_length] = '\n';
    memset(long_contents + header_length + 1, 'a', 3000);
    long_contents[long_length - 1] = '\n';
    long_contents[long_length] = '\0';

    result = expect_load_failure(
        fixture_directory,
        "line_too_long.csv",
        long_contents,
        VAE_C_CONFIG_CSV_FORMAT_ERROR
    );
    free(long_contents);

    return result;
}


static int test_invalid_profiles(const char *fixture_directory)
{
    static const struct {
        const char *name;
        const char *row;
        VAECConfigStatus status;
    } cases[] = {
        {
            "depth_low.csv",
            "depth_low,synthetic,75,2,8,16,0,0,32,5,1,1,test",
            VAE_C_CONFIG_VALUE_OUT_OF_RANGE
        },
        {
            "depth_high.csv",
            "depth_high,synthetic,224,5,8,16,32,64,32,7,1,1,test",
            VAE_C_CONFIG_VALUE_OUT_OF_RANGE
        },
        {
            "depth3_c4_nonzero.csv",
            "depth3_c4,synthetic,75,3,8,16,32,1,32,5,1,1,test",
            VAE_C_CONFIG_VALUE_OUT_OF_RANGE
        },
        {
            "depth4_c4_zero.csv",
            "depth4_c4,synthetic,224,4,32,64,128,0,256,7,1,1,test",
            VAE_C_CONFIG_VALUE_OUT_OF_RANGE
        },
        {
            "used_channel_zero.csv",
            "channel_zero,synthetic,75,3,8,0,32,0,32,5,1,1,test",
            VAE_C_CONFIG_VALUE_OUT_OF_RANGE
        },
        {
            "bottleneck_mismatch.csv",
            "bad_bottleneck,synthetic,75,3,8,16,32,0,32,4,1,1,test",
            VAE_C_CONFIG_SHAPE_MISMATCH
        },
        {
            "parameter_mismatch.csv",
            "bad_parameters,synthetic,75,3,8,16,32,0,32,5,41573,"
            "5620896,test",
            VAE_C_CONFIG_COUNT_MISMATCH
        },
        {
            "mac_mismatch.csv",
            "bad_mac,synthetic,75,3,8,16,32,0,32,5,41572,5620897,"
            "test",
            VAE_C_CONFIG_COUNT_MISMATCH
        }
    };
    size_t index;

    for (index = 0; index < sizeof(cases) / sizeof(cases[0]); index++) {
        char contents[1024];
        int length;

        length = snprintf(
            contents,
            sizeof(contents),
            "%s\n%s\n",
            VAE_C_TEST_HEADER,
            cases[index].row
        );
        if (length < 0 || (size_t)length >= sizeof(contents) ||
            !expect_load_failure(
                fixture_directory,
                cases[index].name,
                contents,
                cases[index].status)) {

            return 0;
        }
    }

    return 1;
}


static int test_catalog_guarantees(
    const char *official_csv,
    const char *fixture_directory
)
{
    VAECProfileCatalog catalog;
    VAECConfigStatus status;
    VAECProfile *original_items;
    size_t original_count;
    char original_id[VAE_C_PROFILE_TEXT_CAP];
    char bad_path[1024];
    char missing_path[1024];
    char error_message[256];
    char tiny_error[2] = {'X', 'Y'};
    int result;

    vae_c_profile_catalog_init(&catalog);
    status = vae_c_profile_catalog_load_csv(
        official_csv,
        &catalog,
        error_message,
        sizeof(error_message)
    );
    if (status != VAE_C_CONFIG_OK || catalog.count != 4) {
        vae_c_profile_catalog_free(&catalog);
        return 0;
    }

    original_items = catalog.items;
    original_count = catalog.count;
    (void)snprintf(
        original_id,
        sizeof(original_id),
        "%s",
        catalog.items[0].profile_id
    );

    if (!make_fixture_path(
            fixture_directory,
            "strong_guarantee_bad.csv",
            bad_path,
            sizeof(bad_path)) ||
        !write_fixture(bad_path, "bad,header\n") ||
        !make_fixture_path(
            fixture_directory,
            "does_not_exist.csv",
            missing_path,
            sizeof(missing_path))) {

        vae_c_profile_catalog_free(&catalog);
        return 0;
    }

    error_message[0] = '\0';
    status = vae_c_profile_catalog_load_csv(
        bad_path,
        &catalog,
        error_message,
        sizeof(error_message)
    );
    result =
        status == VAE_C_CONFIG_CSV_FORMAT_ERROR &&
        error_message[0] != '\0' &&
        catalog.items == original_items &&
        catalog.count == original_count &&
        strcmp(catalog.items[0].profile_id, original_id) == 0;

    status = vae_c_profile_catalog_load_csv(
        missing_path,
        &catalog,
        NULL,
        0
    );
    result =
        result &&
        status == VAE_C_CONFIG_IO_ERROR &&
        catalog.items == original_items &&
        catalog.count == original_count;

    status = vae_c_profile_catalog_load_csv(
        missing_path,
        &catalog,
        tiny_error,
        sizeof(tiny_error)
    );
    result =
        result &&
        status == VAE_C_CONFIG_IO_ERROR &&
        tiny_error[sizeof(tiny_error) - 1] == '\0' &&
        catalog.items == original_items &&
        catalog.count == original_count;

    vae_c_profile_catalog_free(&catalog);
    vae_c_profile_catalog_free(&catalog);

    return
        result &&
        catalog.items == NULL &&
        catalog.count == 0;
}


static int test_api_contract(void)
{
    VAECProfileCatalog catalog;
    VAECProfile profile;
    VAECConfig config;
    VAECConfig original_config;
    VAECConfigStatus status;
    char error_message[64];
    size_t index;
    int result;

    vae_c_profile_catalog_init(&catalog);
    result = catalog.items == NULL && catalog.count == 0;

    memset(&profile, 0, sizeof(profile));
    memset(&config, 0x5A, sizeof(config));
    original_config = config;
    error_message[0] = 'X';

    status = vae_c_config_build(
        &profile,
        &config,
        error_message,
        sizeof(error_message)
    );
    result =
        result &&
        status != VAE_C_CONFIG_OK &&
        memcmp(&config, &original_config, sizeof(config)) == 0 &&
        error_message[0] != '\0';

    result =
        result &&
        vae_c_config_build(NULL, &config, NULL, 0) ==
            VAE_C_CONFIG_INVALID_ARGUMENT &&
        vae_c_config_build(&profile, NULL, NULL, 0) ==
            VAE_C_CONFIG_INVALID_ARGUMENT &&
        vae_c_profile_catalog_load_csv(NULL, &catalog, NULL, 0) ==
            VAE_C_CONFIG_INVALID_ARGUMENT &&
        vae_c_profile_catalog_load_csv("unused", NULL, NULL, 0) ==
            VAE_C_CONFIG_INVALID_ARGUMENT;

    for (index = VAE_C_CONFIG_OK;
         index <= VAE_C_CONFIG_COUNT_MISMATCH;
         index++) {

        if (strcmp(
                vae_c_config_status_name((VAECConfigStatus)index),
                "unknown") == 0) {

            result = 0;
        }
    }

    result =
        result &&
        strcmp(
            vae_c_config_status_name((VAECConfigStatus)999),
            "unknown") == 0;

    vae_c_profile_catalog_free(&catalog);
    vae_c_profile_catalog_free(NULL);
    return result;
}


int main(int argc, char **argv)
{
    VAECProfileCatalog official_catalog;
    int failures;

    if (argc != 3) {
        fprintf(
            stderr,
            "usage: %s <official_profiles_csv> <fixture_directory>\n",
            argc > 0 ? argv[0] : "test_vae_c_config"
        );
        return 2;
    }

    vae_c_profile_catalog_init(&official_catalog);
    failures = 0;

    failures += report_check(
        "official_catalog",
        test_official_catalog(argv[1], &official_catalog)
    );
    failures += report_check(
        "official_derived_configs",
        test_official_configs(&official_catalog)
    );
    failures += report_check("synthetic_depth4", test_depth4_config());
    failures += report_check(
        "valid_line_endings",
        test_valid_line_endings(argv[2])
    );
    failures += report_check(
        "malformed_csv_rejected",
        test_malformed_csv(argv[2])
    );
    failures += report_check(
        "invalid_profiles_rejected",
        test_invalid_profiles(argv[2])
    );
    failures += report_check(
        "catalog_strong_guarantee",
        test_catalog_guarantees(argv[1], argv[2])
    );
    failures += report_check("api_lifecycle_contract", test_api_contract());

    vae_c_profile_catalog_free(&official_catalog);
    vae_c_profile_catalog_free(&official_catalog);

    if (failures != 0) {
        fprintf(stderr, "%d VAE C CONFIG TEST(S) FAILED\n", failures);
        return 1;
    }

    printf("ALL VAE C CONFIG TESTS PASSED\n");
    return 0;
}
