#include "vae_c_config.h"

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>


#define VAE_C_CONFIG_CSV_FIELD_COUNT 13
#define VAE_C_CONFIG_LINE_CAP 2048


static const char vae_c_config_csv_header[] =
    "profile_id,role,input_side,conv_depth,c1,c2,c3,c4,latent_dim,"
    "bottleneck_side,trainable_parameters_bn1d,"
    "forward_mac_per_sample,status";


static void vae_c_error_clear(
    char *error_message,
    size_t error_message_capacity
)
{
    if (error_message != NULL && error_message_capacity > 0) {
        error_message[0] = '\0';
    }
}


static void vae_c_error_set(
    char *error_message,
    size_t error_message_capacity,
    const char *format,
    ...
)
{
    va_list arguments;

    if (error_message == NULL ||
        error_message_capacity == 0 ||
        format == NULL) {
        return;
    }

    va_start(arguments, format);
    (void)vsnprintf(
        error_message,
        error_message_capacity,
        format,
        arguments
    );
    va_end(arguments);

    error_message[error_message_capacity - 1] = '\0';
}


static int vae_c_size_mul(
    size_t left,
    size_t right,
    size_t *result
)
{
    if (result == NULL) {
        return 1;
    }

    if (left != 0 && right > SIZE_MAX / left) {
        return 1;
    }

    *result = left * right;
    return 0;
}


static int vae_c_u64_add(
    uint64_t left,
    uint64_t right,
    uint64_t *result
)
{
    if (result == NULL || right > UINT64_MAX - left) {
        return 1;
    }

    *result = left + right;
    return 0;
}


static int vae_c_u64_mul(
    uint64_t left,
    uint64_t right,
    uint64_t *result
)
{
    if (result == NULL) {
        return 1;
    }

    if (left != 0 && right > UINT64_MAX / left) {
        return 1;
    }

    *result = left * right;
    return 0;
}


static int vae_c_u64_accumulate_product(
    uint64_t *total,
    const uint64_t *factors,
    size_t factor_count
)
{
    uint64_t product;
    size_t i;

    if (total == NULL || factors == NULL || factor_count == 0) {
        return 1;
    }

    product = 1;

    for (i = 0; i < factor_count; i++) {
        if (vae_c_u64_mul(product, factors[i], &product) != 0) {
            return 1;
        }
    }

    return vae_c_u64_add(*total, product, total);
}


static int vae_c_text_is_terminated_and_nonempty(
    const char text[VAE_C_PROFILE_TEXT_CAP]
)
{
    return text != NULL &&
           text[0] != '\0' &&
           memchr(text, '\0', VAE_C_PROFILE_TEXT_CAP) != NULL;
}


static VAECConfigStatus vae_c_validate_profile_fields(
    const VAECProfile *profile,
    char *error_message,
    size_t error_message_capacity
)
{
    size_t i;

    if (profile == NULL) {
        vae_c_error_set(
            error_message,
            error_message_capacity,
            "profile is NULL"
        );
        return VAE_C_CONFIG_INVALID_ARGUMENT;
    }

    if (!vae_c_text_is_terminated_and_nonempty(profile->profile_id)) {
        vae_c_error_set(
            error_message,
            error_message_capacity,
            "field profile_id is empty or not terminated"
        );
        return VAE_C_CONFIG_VALUE_OUT_OF_RANGE;
    }

    if (!vae_c_text_is_terminated_and_nonempty(profile->role)) {
        vae_c_error_set(
            error_message,
            error_message_capacity,
            "field role is empty or not terminated"
        );
        return VAE_C_CONFIG_VALUE_OUT_OF_RANGE;
    }

    if (!vae_c_text_is_terminated_and_nonempty(profile->profile_status)) {
        vae_c_error_set(
            error_message,
            error_message_capacity,
            "field status is empty or not terminated"
        );
        return VAE_C_CONFIG_VALUE_OUT_OF_RANGE;
    }

    if (profile->input_side == 0) {
        vae_c_error_set(
            error_message,
            error_message_capacity,
            "field input_side must be greater than zero"
        );
        return VAE_C_CONFIG_VALUE_OUT_OF_RANGE;
    }

    if (profile->conv_depth != 3 && profile->conv_depth != 4) {
        vae_c_error_set(
            error_message,
            error_message_capacity,
            "field conv_depth must be 3 or 4"
        );
        return VAE_C_CONFIG_VALUE_OUT_OF_RANGE;
    }

    for (i = 0; i < profile->conv_depth; i++) {
        if (profile->channels[i] == 0) {
            vae_c_error_set(
                error_message,
                error_message_capacity,
                "field c%lu must be greater than zero",
                (unsigned long)(i + 1)
            );
            return VAE_C_CONFIG_VALUE_OUT_OF_RANGE;
        }
    }

    for (i = profile->conv_depth;
         i < VAE_C_MAX_CONV_DEPTH;
         i++) {

        if (profile->channels[i] != 0) {
            vae_c_error_set(
                error_message,
                error_message_capacity,
                "field c%lu must be zero for depth %lu",
                (unsigned long)(i + 1),
                (unsigned long)profile->conv_depth
            );
            return VAE_C_CONFIG_VALUE_OUT_OF_RANGE;
        }
    }

    if (profile->latent_dim == 0) {
        vae_c_error_set(
            error_message,
            error_message_capacity,
            "field latent_dim must be greater than zero"
        );
        return VAE_C_CONFIG_VALUE_OUT_OF_RANGE;
    }

    if (profile->bottleneck_side == 0) {
        vae_c_error_set(
            error_message,
            error_message_capacity,
            "field bottleneck_side must be greater than zero"
        );
        return VAE_C_CONFIG_VALUE_OUT_OF_RANGE;
    }

    if (profile->expected_trainable_parameters_bn1d == 0) {
        vae_c_error_set(
            error_message,
            error_message_capacity,
            "field trainable_parameters_bn1d must be greater than zero"
        );
        return VAE_C_CONFIG_VALUE_OUT_OF_RANGE;
    }

    if (profile->expected_forward_mac_per_sample == 0) {
        vae_c_error_set(
            error_message,
            error_message_capacity,
            "field forward_mac_per_sample must be greater than zero"
        );
        return VAE_C_CONFIG_VALUE_OUT_OF_RANGE;
    }

    return VAE_C_CONFIG_OK;
}


static VAECConfigStatus vae_c_conv_down_side(
    size_t input_side,
    size_t *output_side,
    char *error_message,
    size_t error_message_capacity
)
{
    size_t padded;
    size_t numerator;

    if (output_side == NULL) {
        return VAE_C_CONFIG_INVALID_ARGUMENT;
    }

    if (input_side > SIZE_MAX - 2) {
        vae_c_error_set(
            error_message,
            error_message_capacity,
            "overflow in convolution shape"
        );
        return VAE_C_CONFIG_VALUE_OUT_OF_RANGE;
    }

    padded = input_side + 2;

    if (padded < 3) {
        vae_c_error_set(
            error_message,
            error_message_capacity,
            "input side is too small for convolution"
        );
        return VAE_C_CONFIG_SHAPE_MISMATCH;
    }

    numerator = padded - 3;
    *output_side = numerator / 2 + 1;

    if (*output_side == 0) {
        vae_c_error_set(
            error_message,
            error_message_capacity,
            "convolution produced a zero side"
        );
        return VAE_C_CONFIG_SHAPE_MISMATCH;
    }

    return VAE_C_CONFIG_OK;
}


static VAECConfigStatus vae_c_pool_side(
    size_t input_side,
    size_t *output_side,
    char *error_message,
    size_t error_message_capacity
)
{
    if (output_side == NULL) {
        return VAE_C_CONFIG_INVALID_ARGUMENT;
    }

    if (input_side < 2) {
        vae_c_error_set(
            error_message,
            error_message_capacity,
            "input side is too small for max-pooling"
        );
        return VAE_C_CONFIG_SHAPE_MISMATCH;
    }

    *output_side = (input_side - 2) / 2 + 1;

    if (*output_side == 0) {
        vae_c_error_set(
            error_message,
            error_message_capacity,
            "max-pooling produced a zero side"
        );
        return VAE_C_CONFIG_SHAPE_MISMATCH;
    }

    return VAE_C_CONFIG_OK;
}


static VAECConfigStatus vae_c_compute_parameter_count(
    const VAECConfig *config,
    uint64_t *parameter_count,
    char *error_message,
    size_t error_message_capacity
)
{
    uint64_t total;
    size_t i;

    if (config == NULL || parameter_count == NULL) {
        return VAE_C_CONFIG_INVALID_ARGUMENT;
    }

    total = 0;

    for (i = 0; i < config->conv_depth; i++) {
        uint64_t input_channels;
        uint64_t output_channels;
        uint64_t conv_factors[3];
        uint64_t bn_factors[2];

        input_channels =
            i == 0
                ? (uint64_t)VAE_C_INPUT_CHANNELS
                : (uint64_t)config->channels[i - 1];

        output_channels = (uint64_t)config->channels[i];

        conv_factors[0] = 9;
        conv_factors[1] = input_channels;
        conv_factors[2] = output_channels;
        bn_factors[0] = 2;
        bn_factors[1] = output_channels;

        if (vae_c_u64_accumulate_product(
                &total,
                conv_factors,
                sizeof(conv_factors) / sizeof(conv_factors[0])
            ) != 0 ||
            vae_c_u64_accumulate_product(
                &total,
                bn_factors,
                2
            ) != 0) {

            goto overflow;
        }
    }

    {
        uint64_t encoder_fc_factors[3];
        uint64_t encoder_fc_bias_factors[2];
        uint64_t decoder_fc_factors[2];
        uint64_t decoder_bn_factors[2];

        encoder_fc_factors[0] =
            (uint64_t)config->channels[config->conv_depth - 1];
        encoder_fc_factors[1] = 2;
        encoder_fc_factors[2] = (uint64_t)config->latent_dim;

        encoder_fc_bias_factors[0] = 2;
        encoder_fc_bias_factors[1] = (uint64_t)config->latent_dim;

        decoder_fc_factors[0] = (uint64_t)config->latent_dim;
        decoder_fc_factors[1] =
            (uint64_t)config->decoder_fc_features;

        decoder_bn_factors[0] = 2;
        decoder_bn_factors[1] =
            (uint64_t)config->decoder_fc_features;

        if (vae_c_u64_accumulate_product(
                &total,
                encoder_fc_factors,
                3
            ) != 0 ||
            vae_c_u64_accumulate_product(
                &total,
                encoder_fc_bias_factors,
                2
            ) != 0 ||
            vae_c_u64_accumulate_product(
                &total,
                decoder_fc_factors,
                2
            ) != 0 ||
            vae_c_u64_accumulate_product(
                &total,
                decoder_bn_factors,
                2
            ) != 0) {

            goto overflow;
        }
    }

    for (i = 0; i < config->decoder_hidden_conv_count; i++) {
        size_t input_index;
        size_t output_index;
        uint64_t conv_factors[3];
        uint64_t bn_factors[2];

        input_index = config->conv_depth - 1 - i;
        output_index = input_index - 1;

        conv_factors[0] = 9;
        conv_factors[1] =
            (uint64_t)config->channels[input_index];
        conv_factors[2] =
            (uint64_t)config->channels[output_index];

        bn_factors[0] = 2;
        bn_factors[1] =
            (uint64_t)config->channels[output_index];

        if (vae_c_u64_accumulate_product(
                &total,
                conv_factors,
                3
            ) != 0 ||
            vae_c_u64_accumulate_product(
                &total,
                bn_factors,
                2
            ) != 0) {

            goto overflow;
        }
    }

    {
        uint64_t final_conv_factors[3];

        final_conv_factors[0] = 9;
        final_conv_factors[1] = (uint64_t)config->channels[0];
        final_conv_factors[2] = (uint64_t)VAE_C_INPUT_CHANNELS;

        if (vae_c_u64_accumulate_product(
                &total,
                final_conv_factors,
                3
            ) != 0 ||
            vae_c_u64_add(
                total,
                (uint64_t)VAE_C_INPUT_CHANNELS,
                &total
            ) != 0) {

            goto overflow;
        }
    }

    *parameter_count = total;
    return VAE_C_CONFIG_OK;

overflow:
    vae_c_error_set(
        error_message,
        error_message_capacity,
        "overflow while counting trainable parameters"
    );
    return VAE_C_CONFIG_VALUE_OUT_OF_RANGE;
}


static VAECConfigStatus vae_c_compute_mac_count(
    const VAECConfig *config,
    uint64_t *mac_count,
    char *error_message,
    size_t error_message_capacity
)
{
    uint64_t total;
    size_t i;

    if (config == NULL || mac_count == NULL) {
        return VAE_C_CONFIG_INVALID_ARGUMENT;
    }

    total = 0;

    for (i = 0; i < config->conv_depth; i++) {
        uint64_t input_channels;
        uint64_t factors[6];

        input_channels =
            i == 0
                ? (uint64_t)VAE_C_INPUT_CHANNELS
                : (uint64_t)config->channels[i - 1];

        factors[0] = (uint64_t)config->encoder_conv_sides[i];
        factors[1] = (uint64_t)config->encoder_conv_sides[i];
        factors[2] = (uint64_t)config->channels[i];
        factors[3] = input_channels;
        factors[4] = 3;
        factors[5] = 3;

        if (vae_c_u64_accumulate_product(
                &total,
                factors,
                6
            ) != 0) {

            goto overflow;
        }
    }

    {
        uint64_t encoder_fc_factors[3];
        uint64_t decoder_fc_factors[2];

        encoder_fc_factors[0] =
            (uint64_t)config->channels[config->conv_depth - 1];
        encoder_fc_factors[1] = 2;
        encoder_fc_factors[2] = (uint64_t)config->latent_dim;

        decoder_fc_factors[0] = (uint64_t)config->latent_dim;
        decoder_fc_factors[1] =
            (uint64_t)config->decoder_fc_features;

        if (vae_c_u64_accumulate_product(
                &total,
                encoder_fc_factors,
                3
            ) != 0 ||
            vae_c_u64_accumulate_product(
                &total,
                decoder_fc_factors,
                2
            ) != 0) {

            goto overflow;
        }
    }

    for (i = 0; i < config->decoder_hidden_conv_count; i++) {
        size_t input_index;
        size_t output_index;
        uint64_t factors[6];

        input_index = config->conv_depth - 1 - i;
        output_index = input_index - 1;

        factors[0] =
            (uint64_t)config->decoder_hidden_conv_sides[i];
        factors[1] =
            (uint64_t)config->decoder_hidden_conv_sides[i];
        factors[2] =
            (uint64_t)config->channels[output_index];
        factors[3] =
            (uint64_t)config->channels[input_index];
        factors[4] = 3;
        factors[5] = 3;

        if (vae_c_u64_accumulate_product(
                &total,
                factors,
                6
            ) != 0) {

            goto overflow;
        }
    }

    {
        uint64_t final_factors[6];

        final_factors[0] = (uint64_t)config->input_side;
        final_factors[1] = (uint64_t)config->input_side;
        final_factors[2] = (uint64_t)VAE_C_INPUT_CHANNELS;
        final_factors[3] = (uint64_t)config->channels[0];
        final_factors[4] = 3;
        final_factors[5] = 3;

        if (vae_c_u64_accumulate_product(
                &total,
                final_factors,
                6
            ) != 0) {

            goto overflow;
        }
    }

    *mac_count = total;
    return VAE_C_CONFIG_OK;

overflow:
    vae_c_error_set(
        error_message,
        error_message_capacity,
        "overflow while counting forward MAC"
    );
    return VAE_C_CONFIG_VALUE_OUT_OF_RANGE;
}


void vae_c_profile_catalog_init(VAECProfileCatalog *catalog)
{
    if (catalog != NULL) {
        catalog->items = NULL;
        catalog->count = 0;
    }
}


void vae_c_profile_catalog_free(VAECProfileCatalog *catalog)
{
    if (catalog == NULL) {
        return;
    }

    free(catalog->items);
    catalog->items = NULL;
    catalog->count = 0;
}


const VAECProfile *vae_c_profile_catalog_find(
    const VAECProfileCatalog *catalog,
    const char *profile_id
)
{
    size_t i;

    if (catalog == NULL ||
        profile_id == NULL ||
        catalog->items == NULL) {

        return NULL;
    }

    for (i = 0; i < catalog->count; i++) {
        if (strcmp(catalog->items[i].profile_id, profile_id) == 0) {
            return &catalog->items[i];
        }
    }

    return NULL;
}


VAECConfigStatus vae_c_config_build(
    const VAECProfile *profile,
    VAECConfig *config,
    char *error_message,
    size_t error_message_capacity
)
{
    VAECConfig candidate;
    VAECConfigStatus status;
    size_t current_side;
    size_t squared_side;
    size_t index;
    size_t target_index;
    uint64_t values;
    uint64_t parameters;
    uint64_t mac;

    vae_c_error_clear(error_message, error_message_capacity);

    if (profile == NULL || config == NULL) {
        vae_c_error_set(
            error_message,
            error_message_capacity,
            "profile and config must be non-NULL"
        );
        return VAE_C_CONFIG_INVALID_ARGUMENT;
    }

    status = vae_c_validate_profile_fields(
        profile,
        error_message,
        error_message_capacity
    );

    if (status != VAE_C_CONFIG_OK) {
        return status;
    }

    memset(&candidate, 0, sizeof(candidate));

    candidate.input_channels = VAE_C_INPUT_CHANNELS;
    candidate.input_side = profile->input_side;
    candidate.conv_depth = profile->conv_depth;
    candidate.latent_dim = profile->latent_dim;
    candidate.bottleneck_side = profile->bottleneck_side;

    for (index = 0; index < profile->conv_depth; index++) {
        candidate.channels[index] = profile->channels[index];
    }

    status = vae_c_conv_down_side(
        candidate.input_side,
        &candidate.encoder_conv_sides[0],
        error_message,
        error_message_capacity
    );

    if (status != VAE_C_CONFIG_OK) {
        return status;
    }

    status = vae_c_pool_side(
        candidate.encoder_conv_sides[0],
        &candidate.pool_side,
        error_message,
        error_message_capacity
    );

    if (status != VAE_C_CONFIG_OK) {
        return status;
    }

    current_side = candidate.pool_side;

    for (index = 1; index < candidate.conv_depth; index++) {
        status = vae_c_conv_down_side(
            current_side,
            &candidate.encoder_conv_sides[index],
            error_message,
            error_message_capacity
        );

        if (status != VAE_C_CONFIG_OK) {
            return status;
        }

        current_side = candidate.encoder_conv_sides[index];
    }

    if (current_side != candidate.bottleneck_side) {
        vae_c_error_set(
            error_message,
            error_message_capacity,
            "bottleneck mismatch: derived %lu, expected %lu",
            (unsigned long)current_side,
            (unsigned long)candidate.bottleneck_side
        );
        return VAE_C_CONFIG_SHAPE_MISMATCH;
    }

    if (vae_c_size_mul(
            candidate.bottleneck_side,
            candidate.bottleneck_side,
            &squared_side
        ) != 0 ||
        vae_c_size_mul(
            candidate.channels[candidate.conv_depth - 1],
            squared_side,
            &candidate.decoder_fc_features
        ) != 0) {

        vae_c_error_set(
            error_message,
            error_message_capacity,
            "overflow in decoder_fc_features"
        );
        return VAE_C_CONFIG_VALUE_OUT_OF_RANGE;
    }

    target_index = 0;

    for (index = candidate.conv_depth - 1; index > 1; index--) {
        if (target_index >= VAE_C_MAX_DECODER_UPSAMPLES) {
            vae_c_error_set(
                error_message,
                error_message_capacity,
                "too many decoder upsampling targets"
            );
            return VAE_C_CONFIG_VALUE_OUT_OF_RANGE;
        }

        candidate.decoder_upsample_targets[target_index] =
            candidate.encoder_conv_sides[index - 1];
        target_index++;
    }

    if (target_index + 3 > VAE_C_MAX_DECODER_UPSAMPLES) {
        vae_c_error_set(
            error_message,
            error_message_capacity,
            "too many decoder upsampling targets"
        );
        return VAE_C_CONFIG_VALUE_OUT_OF_RANGE;
    }

    candidate.decoder_upsample_targets[target_index++] =
        candidate.pool_side;
    candidate.decoder_upsample_targets[target_index++] =
        candidate.encoder_conv_sides[0];
    candidate.decoder_upsample_targets[target_index++] =
        candidate.input_side;

    candidate.decoder_upsample_count = target_index;
    candidate.decoder_hidden_conv_count =
        candidate.conv_depth - 1;

    if (candidate.decoder_hidden_conv_count >
        VAE_C_MAX_DECODER_HIDDEN_CONVS) {

        vae_c_error_set(
            error_message,
            error_message_capacity,
            "too many decoder hidden convolutions"
        );
        return VAE_C_CONFIG_VALUE_OUT_OF_RANGE;
    }

    for (index = 0;
         index < candidate.decoder_hidden_conv_count;
         index++) {

        candidate.decoder_hidden_conv_sides[index] =
            candidate.decoder_upsample_targets[index + 1];
    }

    if (vae_c_u64_mul(
            (uint64_t)candidate.input_side,
            (uint64_t)candidate.input_side,
            &values
        ) != 0 ||
        vae_c_u64_mul(
            values,
            (uint64_t)VAE_C_INPUT_CHANNELS,
            &values
        ) != 0) {

        vae_c_error_set(
            error_message,
            error_message_capacity,
            "overflow in values_per_sample"
        );
        return VAE_C_CONFIG_VALUE_OUT_OF_RANGE;
    }

    candidate.values_per_sample = values;

    status = vae_c_compute_parameter_count(
        &candidate,
        &parameters,
        error_message,
        error_message_capacity
    );

    if (status != VAE_C_CONFIG_OK) {
        return status;
    }

    candidate.trainable_parameters_bn1d = parameters;

    if (parameters != profile->expected_trainable_parameters_bn1d) {
        vae_c_error_set(
            error_message,
            error_message_capacity,
            "parameter count mismatch: derived %llu, expected %llu",
            (unsigned long long)parameters,
            (unsigned long long)
                profile->expected_trainable_parameters_bn1d
        );
        return VAE_C_CONFIG_COUNT_MISMATCH;
    }

    status = vae_c_compute_mac_count(
        &candidate,
        &mac,
        error_message,
        error_message_capacity
    );

    if (status != VAE_C_CONFIG_OK) {
        return status;
    }

    candidate.forward_mac_per_sample = mac;

    if (mac != profile->expected_forward_mac_per_sample) {
        vae_c_error_set(
            error_message,
            error_message_capacity,
            "MAC count mismatch: derived %llu, expected %llu",
            (unsigned long long)mac,
            (unsigned long long)
                profile->expected_forward_mac_per_sample
        );
        return VAE_C_CONFIG_COUNT_MISMATCH;
    }

    *config = candidate;
    return VAE_C_CONFIG_OK;
}


static int vae_c_strip_line_ending(
    char *line,
    size_t line_capacity,
    FILE *stream,
    size_t line_number,
    char *error_message,
    size_t error_message_capacity
)
{
    size_t length;

    if (line == NULL || line_capacity == 0 || stream == NULL) {
        return 1;
    }

    length = strlen(line);

    if (length == line_capacity - 1 &&
        length > 0 &&
        line[length - 1] != '\n' &&
        !feof(stream)) {

        vae_c_error_set(
            error_message,
            error_message_capacity,
            "line %lu: line is too long",
            (unsigned long)line_number
        );
        return 1;
    }

    if (length > 0 && line[length - 1] == '\n') {
        line[--length] = '\0';

        if (length > 0 && line[length - 1] == '\r') {
            line[--length] = '\0';
        }
    }

    if (strchr(line, '\r') != NULL ||
        strchr(line, '\n') != NULL) {

        vae_c_error_set(
            error_message,
            error_message_capacity,
            "line %lu: invalid line ending",
            (unsigned long)line_number
        );
        return 1;
    }

    return 0;
}


static VAECConfigStatus vae_c_split_csv_fields(
    char *line,
    char *fields[VAE_C_CONFIG_CSV_FIELD_COUNT],
    size_t line_number,
    char *error_message,
    size_t error_message_capacity
)
{
    char *cursor;
    size_t count;
    size_t i;

    if (line == NULL || fields == NULL) {
        return VAE_C_CONFIG_INVALID_ARGUMENT;
    }

    if (line[0] == '\0') {
        vae_c_error_set(
            error_message,
            error_message_capacity,
            "line %lu: empty data row",
            (unsigned long)line_number
        );
        return VAE_C_CONFIG_CSV_FORMAT_ERROR;
    }

    fields[0] = line;
    count = 1;

    for (cursor = line; *cursor != '\0'; cursor++) {
        if (*cursor == '"') {
            vae_c_error_set(
                error_message,
                error_message_capacity,
                "line %lu: CSV quoting is not supported",
                (unsigned long)line_number
            );
            return VAE_C_CONFIG_CSV_FORMAT_ERROR;
        }

        if (*cursor == ',') {
            *cursor = '\0';

            if (count >= VAE_C_CONFIG_CSV_FIELD_COUNT) {
                vae_c_error_set(
                    error_message,
                    error_message_capacity,
                    "line %lu: too many CSV fields",
                    (unsigned long)line_number
                );
                return VAE_C_CONFIG_CSV_FORMAT_ERROR;
            }

            fields[count++] = cursor + 1;
        }
    }

    if (count != VAE_C_CONFIG_CSV_FIELD_COUNT) {
        vae_c_error_set(
            error_message,
            error_message_capacity,
            "line %lu: expected 13 fields, found %lu",
            (unsigned long)line_number,
            (unsigned long)count
        );
        return VAE_C_CONFIG_CSV_FORMAT_ERROR;
    }

    for (i = 0; i < count; i++) {
        size_t length;

        length = strlen(fields[i]);

        if (length == 0) {
            vae_c_error_set(
                error_message,
                error_message_capacity,
                "line %lu, field %lu: field is empty",
                (unsigned long)line_number,
                (unsigned long)(i + 1)
            );
            return VAE_C_CONFIG_CSV_FORMAT_ERROR;
        }

        if (isspace((unsigned char)fields[i][0]) ||
            isspace((unsigned char)fields[i][length - 1])) {

            vae_c_error_set(
                error_message,
                error_message_capacity,
                "line %lu, field %lu: surrounding whitespace is forbidden",
                (unsigned long)line_number,
                (unsigned long)(i + 1)
            );
            return VAE_C_CONFIG_CSV_FORMAT_ERROR;
        }
    }

    return VAE_C_CONFIG_OK;
}


static VAECConfigStatus vae_c_copy_csv_text(
    const char *source,
    char destination[VAE_C_PROFILE_TEXT_CAP],
    size_t line_number,
    const char *field_name,
    char *error_message,
    size_t error_message_capacity
)
{
    size_t length;

    if (source == NULL || destination == NULL || field_name == NULL) {
        return VAE_C_CONFIG_INVALID_ARGUMENT;
    }

    length = strlen(source);

    if (length == 0 || length >= VAE_C_PROFILE_TEXT_CAP) {
        vae_c_error_set(
            error_message,
            error_message_capacity,
            "line %lu, field %s: text length must be 1..63",
            (unsigned long)line_number,
            field_name
        );
        return VAE_C_CONFIG_VALUE_OUT_OF_RANGE;
    }

    memcpy(destination, source, length + 1);
    return VAE_C_CONFIG_OK;
}


static VAECConfigStatus vae_c_parse_u64_field(
    const char *text,
    uint64_t *value,
    size_t line_number,
    const char *field_name,
    char *error_message,
    size_t error_message_capacity
)
{
    unsigned long long parsed;
    char *end_pointer;
    const unsigned char *cursor;

    if (text == NULL || value == NULL || field_name == NULL) {
        return VAE_C_CONFIG_INVALID_ARGUMENT;
    }

    if (text[0] == '+' || text[0] == '-') {
        vae_c_error_set(
            error_message,
            error_message_capacity,
            "line %lu, field %s: signs are forbidden",
            (unsigned long)line_number,
            field_name
        );
        return VAE_C_CONFIG_CSV_FORMAT_ERROR;
    }

    for (cursor = (const unsigned char *)text;
         *cursor != '\0';
         cursor++) {

        if (!isdigit(*cursor)) {
            vae_c_error_set(
                error_message,
                error_message_capacity,
                "line %lu, field %s: expected unsigned decimal integer",
                (unsigned long)line_number,
                field_name
            );
            return VAE_C_CONFIG_CSV_FORMAT_ERROR;
        }
    }

    errno = 0;
    end_pointer = NULL;
    parsed = strtoull(text, &end_pointer, 10);

    if (errno == ERANGE ||
        end_pointer == text ||
        end_pointer == NULL ||
        *end_pointer != '\0') {

        vae_c_error_set(
            error_message,
            error_message_capacity,
            "line %lu, field %s: unsigned integer overflow",
            (unsigned long)line_number,
            field_name
        );
        return VAE_C_CONFIG_VALUE_OUT_OF_RANGE;
    }

#if ULLONG_MAX > UINT64_MAX
    if (parsed > (unsigned long long)UINT64_MAX) {
        vae_c_error_set(
            error_message,
            error_message_capacity,
            "line %lu, field %s: unsigned integer overflow",
            (unsigned long)line_number,
            field_name
        );
        return VAE_C_CONFIG_VALUE_OUT_OF_RANGE;
    }
#endif

    *value = (uint64_t)parsed;
    return VAE_C_CONFIG_OK;
}


static VAECConfigStatus vae_c_parse_size_field(
    const char *text,
    size_t *value,
    size_t line_number,
    const char *field_name,
    char *error_message,
    size_t error_message_capacity
)
{
    uint64_t parsed;
    VAECConfigStatus status;

    status = vae_c_parse_u64_field(
        text,
        &parsed,
        line_number,
        field_name,
        error_message,
        error_message_capacity
    );

    if (status != VAE_C_CONFIG_OK) {
        return status;
    }

    if (parsed > SIZE_MAX) {
        vae_c_error_set(
            error_message,
            error_message_capacity,
            "line %lu, field %s: value does not fit size_t",
            (unsigned long)line_number,
            field_name
        );
        return VAE_C_CONFIG_VALUE_OUT_OF_RANGE;
    }

    *value = (size_t)parsed;
    return VAE_C_CONFIG_OK;
}


static VAECConfigStatus vae_c_parse_profile(
    char *fields[VAE_C_CONFIG_CSV_FIELD_COUNT],
    size_t line_number,
    VAECProfile *profile,
    char *error_message,
    size_t error_message_capacity
)
{
    static const char *channel_names[VAE_C_MAX_CONV_DEPTH] = {
        "c1", "c2", "c3", "c4"
    };

    VAECConfigStatus status;
    size_t i;

    if (fields == NULL || profile == NULL) {
        return VAE_C_CONFIG_INVALID_ARGUMENT;
    }

    memset(profile, 0, sizeof(*profile));

    status = vae_c_copy_csv_text(
        fields[0],
        profile->profile_id,
        line_number,
        "profile_id",
        error_message,
        error_message_capacity
    );
    if (status != VAE_C_CONFIG_OK) return status;

    status = vae_c_copy_csv_text(
        fields[1],
        profile->role,
        line_number,
        "role",
        error_message,
        error_message_capacity
    );
    if (status != VAE_C_CONFIG_OK) return status;

    status = vae_c_parse_size_field(
        fields[2],
        &profile->input_side,
        line_number,
        "input_side",
        error_message,
        error_message_capacity
    );
    if (status != VAE_C_CONFIG_OK) return status;

    status = vae_c_parse_size_field(
        fields[3],
        &profile->conv_depth,
        line_number,
        "conv_depth",
        error_message,
        error_message_capacity
    );
    if (status != VAE_C_CONFIG_OK) return status;

    for (i = 0; i < VAE_C_MAX_CONV_DEPTH; i++) {
        status = vae_c_parse_size_field(
            fields[4 + i],
            &profile->channels[i],
            line_number,
            channel_names[i],
            error_message,
            error_message_capacity
        );
        if (status != VAE_C_CONFIG_OK) return status;
    }

    status = vae_c_parse_size_field(
        fields[8],
        &profile->latent_dim,
        line_number,
        "latent_dim",
        error_message,
        error_message_capacity
    );
    if (status != VAE_C_CONFIG_OK) return status;

    status = vae_c_parse_size_field(
        fields[9],
        &profile->bottleneck_side,
        line_number,
        "bottleneck_side",
        error_message,
        error_message_capacity
    );
    if (status != VAE_C_CONFIG_OK) return status;

    status = vae_c_parse_u64_field(
        fields[10],
        &profile->expected_trainable_parameters_bn1d,
        line_number,
        "trainable_parameters_bn1d",
        error_message,
        error_message_capacity
    );
    if (status != VAE_C_CONFIG_OK) return status;

    status = vae_c_parse_u64_field(
        fields[11],
        &profile->expected_forward_mac_per_sample,
        line_number,
        "forward_mac_per_sample",
        error_message,
        error_message_capacity
    );
    if (status != VAE_C_CONFIG_OK) return status;

    status = vae_c_copy_csv_text(
        fields[12],
        profile->profile_status,
        line_number,
        "status",
        error_message,
        error_message_capacity
    );
    if (status != VAE_C_CONFIG_OK) return status;

    return vae_c_validate_profile_fields(
        profile,
        error_message,
        error_message_capacity
    );
}


static int vae_c_catalog_has_profile_id(
    const VAECProfileCatalog *catalog,
    const char *profile_id
)
{
    return vae_c_profile_catalog_find(catalog, profile_id) != NULL;
}


static VAECConfigStatus vae_c_catalog_append(
    VAECProfileCatalog *catalog,
    size_t *capacity,
    const VAECProfile *profile,
    char *error_message,
    size_t error_message_capacity
)
{
    VAECProfile *new_items;
    size_t new_capacity;

    if (catalog == NULL || capacity == NULL || profile == NULL) {
        return VAE_C_CONFIG_INVALID_ARGUMENT;
    }

    if (catalog->count == *capacity) {
        if (*capacity == 0) {
            new_capacity = 4;
        } else {
            if (*capacity > SIZE_MAX / 2) {
                vae_c_error_set(
                    error_message,
                    error_message_capacity,
                    "profile catalog capacity overflow"
                );
                return VAE_C_CONFIG_VALUE_OUT_OF_RANGE;
            }
            new_capacity = *capacity * 2;
        }

        if (new_capacity > SIZE_MAX / sizeof(VAECProfile)) {
            vae_c_error_set(
                error_message,
                error_message_capacity,
                "profile catalog byte-size overflow"
            );
            return VAE_C_CONFIG_VALUE_OUT_OF_RANGE;
        }

        new_items = (VAECProfile *)realloc(
            catalog->items,
            new_capacity * sizeof(VAECProfile)
        );

        if (new_items == NULL) {
            vae_c_error_set(
                error_message,
                error_message_capacity,
                "profile catalog allocation failed"
            );
            return VAE_C_CONFIG_ALLOCATION_FAILED;
        }

        catalog->items = new_items;
        *capacity = new_capacity;
    }

    catalog->items[catalog->count] = *profile;
    catalog->count++;
    return VAE_C_CONFIG_OK;
}


VAECConfigStatus vae_c_profile_catalog_load_csv(
    const char *path,
    VAECProfileCatalog *catalog,
    char *error_message,
    size_t error_message_capacity
)
{
    VAECProfileCatalog candidate;
    VAECConfigStatus status;
    FILE *stream;
    char line[VAE_C_CONFIG_LINE_CAP];
    size_t line_number;
    size_t capacity;

    vae_c_error_clear(error_message, error_message_capacity);

    if (path == NULL || catalog == NULL) {
        vae_c_error_set(
            error_message,
            error_message_capacity,
            "path and catalog must be non-NULL"
        );
        return VAE_C_CONFIG_INVALID_ARGUMENT;
    }

    stream = fopen(path, "rb");

    if (stream == NULL) {
        vae_c_error_set(
            error_message,
            error_message_capacity,
            "cannot open CSV: %s",
            path
        );
        return VAE_C_CONFIG_IO_ERROR;
    }

    vae_c_profile_catalog_init(&candidate);
    capacity = 0;
    line_number = 1;
    status = VAE_C_CONFIG_OK;

    if (fgets(line, sizeof(line), stream) == NULL) {
        if (ferror(stream)) {
            status = VAE_C_CONFIG_IO_ERROR;
            vae_c_error_set(
                error_message,
                error_message_capacity,
                "I/O error while reading CSV header"
            );
        } else {
            status = VAE_C_CONFIG_CSV_FORMAT_ERROR;
            vae_c_error_set(
                error_message,
                error_message_capacity,
                "CSV is empty"
            );
        }
        goto fail;
    }

    if (vae_c_strip_line_ending(
            line,
            sizeof(line),
            stream,
            line_number,
            error_message,
            error_message_capacity
        ) != 0) {

        status = VAE_C_CONFIG_CSV_FORMAT_ERROR;
        goto fail;
    }

    if (strcmp(line, vae_c_config_csv_header) != 0) {
        vae_c_error_set(
            error_message,
            error_message_capacity,
            "line 1: CSV header does not match the v1 contract"
        );
        status = VAE_C_CONFIG_CSV_FORMAT_ERROR;
        goto fail;
    }

    while (fgets(line, sizeof(line), stream) != NULL) {
        char *fields[VAE_C_CONFIG_CSV_FIELD_COUNT];
        VAECProfile profile;
        VAECConfig derived_config;
        char detail[256];

        line_number++;

        if (vae_c_strip_line_ending(
                line,
                sizeof(line),
                stream,
                line_number,
                error_message,
                error_message_capacity
            ) != 0) {

            status = VAE_C_CONFIG_CSV_FORMAT_ERROR;
            goto fail;
        }

        status = vae_c_split_csv_fields(
            line,
            fields,
            line_number,
            error_message,
            error_message_capacity
        );

        if (status != VAE_C_CONFIG_OK) {
            goto fail;
        }

        status = vae_c_parse_profile(
            fields,
            line_number,
            &profile,
            error_message,
            error_message_capacity
        );

        if (status != VAE_C_CONFIG_OK) {
            goto fail;
        }

        if (vae_c_catalog_has_profile_id(
                &candidate,
                profile.profile_id
            )) {

            vae_c_error_set(
                error_message,
                error_message_capacity,
                "line %lu, field profile_id: duplicate value %s",
                (unsigned long)line_number,
                profile.profile_id
            );
            status = VAE_C_CONFIG_DUPLICATE_PROFILE_ID;
            goto fail;
        }

        detail[0] = '\0';
        status = vae_c_config_build(
            &profile,
            &derived_config,
            detail,
            sizeof(detail)
        );

        if (status != VAE_C_CONFIG_OK) {
            vae_c_error_set(
                error_message,
                error_message_capacity,
                "line %lu: %s",
                (unsigned long)line_number,
                detail
            );
            goto fail;
        }

        status = vae_c_catalog_append(
            &candidate,
            &capacity,
            &profile,
            error_message,
            error_message_capacity
        );

        if (status != VAE_C_CONFIG_OK) {
            goto fail;
        }
    }

    if (ferror(stream)) {
        vae_c_error_set(
            error_message,
            error_message_capacity,
            "I/O error while reading CSV data"
        );
        status = VAE_C_CONFIG_IO_ERROR;
        goto fail;
    }

    if (candidate.count == 0) {
        vae_c_error_set(
            error_message,
            error_message_capacity,
            "CSV contains no data rows"
        );
        status = VAE_C_CONFIG_CSV_FORMAT_ERROR;
        goto fail;
    }

    if (fclose(stream) != 0) {
        vae_c_error_set(
            error_message,
            error_message_capacity,
            "I/O error while closing CSV"
        );
        vae_c_profile_catalog_free(&candidate);
        return VAE_C_CONFIG_IO_ERROR;
    }

    vae_c_profile_catalog_free(catalog);
    *catalog = candidate;
    return VAE_C_CONFIG_OK;

fail:
    (void)fclose(stream);
    vae_c_profile_catalog_free(&candidate);
    return status;
}


const char *vae_c_config_status_name(VAECConfigStatus status)
{
    switch (status) {
        case VAE_C_CONFIG_OK:
            return "success";
        case VAE_C_CONFIG_INVALID_ARGUMENT:
            return "invalid_argument";
        case VAE_C_CONFIG_IO_ERROR:
            return "io_error";
        case VAE_C_CONFIG_CSV_FORMAT_ERROR:
            return "csv_format_error";
        case VAE_C_CONFIG_VALUE_OUT_OF_RANGE:
            return "value_out_of_range";
        case VAE_C_CONFIG_ALLOCATION_FAILED:
            return "allocation_failed";
        case VAE_C_CONFIG_DUPLICATE_PROFILE_ID:
            return "duplicate_profile_id";
        case VAE_C_CONFIG_SHAPE_MISMATCH:
            return "shape_mismatch";
        case VAE_C_CONFIG_COUNT_MISMATCH:
            return "count_mismatch";
        default:
            return "unknown";
    }
}
