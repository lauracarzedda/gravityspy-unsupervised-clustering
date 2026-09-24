#ifndef VAE_C_IDENTITY_H
#define VAE_C_IDENTITY_H

#include "vae_c_sha256.h"

#include <stdint.h>


#define VAE_C_IDENTITY_DATASET_VERSION_CAP 64u


typedef enum {
    VAE_C_IDENTITY_OK = 0,
    VAE_C_IDENTITY_INVALID_ARGUMENT,
    VAE_C_IDENTITY_INVALID_COMPONENT,
    VAE_C_IDENTITY_OVERFLOW,
    VAE_C_IDENTITY_HASH_FAILED
} VAECIdentityStatus;


/*
 * PROJECT: canonical mini224 identities are SHA-256 digests of fixed
 * key=value\n preimages.  SHA inputs must be exactly 64 lowercase hex
 * characters.  Dataset versions are non-empty ASCII identifiers containing
 * only letters, digits, '.', '_' or '-'.  Outputs are published only after
 * complete validation and hashing succeeds.
 */
VAECIdentityStatus vae_c_identity_build_mini224_dataset_v1(
    const char *dataset_version,
    const char *event_manifest_sha256,
    const char *tensor_manifest_sha256,
    char identity_sha256[VAE_C_SHA256_HEX_SIZE]
);

VAECIdentityStatus vae_c_identity_build_mini224_train_selection_v1(
    const char *dataset_identity_sha256,
    const char *official_split_sha256,
    const char *subset_sha256,
    uint64_t subset_count,
    char identity_sha256[VAE_C_SHA256_HEX_SIZE]
);

VAECIdentityStatus vae_c_identity_build_mini224_validation_selection_v1(
    const char *dataset_identity_sha256,
    const char *official_split_sha256,
    const char *subset_sha256,
    uint64_t subset_count,
    char identity_sha256[VAE_C_SHA256_HEX_SIZE]
);

const char *vae_c_identity_status_name(VAECIdentityStatus status);

#endif
