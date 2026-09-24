#ifndef VAE_C_CONFIG_H
#define VAE_C_CONFIG_H

#include <stddef.h>
#include <stdint.h>


enum {
    VAE_C_INPUT_CHANNELS = 4,
    VAE_C_MAX_CONV_DEPTH = 4,
    VAE_C_MAX_DECODER_UPSAMPLES = 5,
    VAE_C_MAX_DECODER_HIDDEN_CONVS = 3,
    VAE_C_PROFILE_TEXT_CAP = 64
};


typedef enum {
    VAE_C_CONFIG_OK = 0,
    VAE_C_CONFIG_INVALID_ARGUMENT,
    VAE_C_CONFIG_IO_ERROR,
    VAE_C_CONFIG_CSV_FORMAT_ERROR,
    VAE_C_CONFIG_VALUE_OUT_OF_RANGE,
    VAE_C_CONFIG_ALLOCATION_FAILED,
    VAE_C_CONFIG_DUPLICATE_PROFILE_ID,
    VAE_C_CONFIG_SHAPE_MISMATCH,
    VAE_C_CONFIG_COUNT_MISMATCH
} VAECConfigStatus;


typedef struct {
    char profile_id[VAE_C_PROFILE_TEXT_CAP];
    char role[VAE_C_PROFILE_TEXT_CAP];

    size_t input_side;
    size_t conv_depth;
    size_t channels[VAE_C_MAX_CONV_DEPTH];

    size_t latent_dim;
    size_t bottleneck_side;

    uint64_t expected_trainable_parameters_bn1d;
    uint64_t expected_forward_mac_per_sample;

    char profile_status[VAE_C_PROFILE_TEXT_CAP];
} VAECProfile;


typedef struct {
    VAECProfile *items;
    size_t count;
} VAECProfileCatalog;


typedef struct {
    size_t input_channels;
    size_t input_side;
    size_t conv_depth;
    size_t channels[VAE_C_MAX_CONV_DEPTH];

    size_t encoder_conv_sides[VAE_C_MAX_CONV_DEPTH];
    size_t pool_side;

    size_t latent_dim;
    size_t bottleneck_side;
    size_t decoder_fc_features;

    size_t decoder_upsample_count;
    size_t decoder_upsample_targets[VAE_C_MAX_DECODER_UPSAMPLES];

    size_t decoder_hidden_conv_count;
    size_t decoder_hidden_conv_sides[VAE_C_MAX_DECODER_HIDDEN_CONVS];

    uint64_t values_per_sample;
    uint64_t trainable_parameters_bn1d;
    uint64_t forward_mac_per_sample;
} VAECConfig;


void vae_c_profile_catalog_init(VAECProfileCatalog *catalog);

void vae_c_profile_catalog_free(VAECProfileCatalog *catalog);

VAECConfigStatus vae_c_profile_catalog_load_csv(
    const char *path,
    VAECProfileCatalog *catalog,
    char *error_message,
    size_t error_message_capacity
);

const VAECProfile *vae_c_profile_catalog_find(
    const VAECProfileCatalog *catalog,
    const char *profile_id
);

VAECConfigStatus vae_c_config_build(
    const VAECProfile *profile,
    VAECConfig *config,
    char *error_message,
    size_t error_message_capacity
);

const char *vae_c_config_status_name(VAECConfigStatus status);

#endif
