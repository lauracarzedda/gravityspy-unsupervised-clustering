#ifndef VAE_C_IIC_CHECKPOINT_H
#define VAE_C_IIC_CHECKPOINT_H

#include "vae_c_iic_ensemble.h"

#include <stddef.h>
#include <stdint.h>

#define VAE_C_IIC_CHECKPOINT_VERSION 1u
#define VAE_C_IIC_CHECKPOINT_RUN_ID_CAP 96u
#define VAE_C_IIC_CHECKPOINT_COMMIT_CAP 65u
#define VAE_C_IIC_CHECKPOINT_PRODUCER_CAP 96u
#define VAE_C_IIC_CHECKPOINT_SHA256_CAP 65u

typedef enum {
    VAE_C_IIC_CHECKPOINT_LATEST = 1,
    VAE_C_IIC_CHECKPOINT_FINAL = 2
} VAECIICCheckpointKind;

/* The caller independently constructs this identity; it is never learned
 * from a checkpoint under verification. producer_state must identify the
 * actual producer provenance; uncommitted code must not be attributed to
 * base_commit. */
typedef struct {
    size_t input_dim;
    size_t capacity;
    size_t batch_size;
    size_t train_rows;
    uint64_t max_epochs;
    uint64_t shuffle_seed;
    uint64_t seeds[VAE_C_IIC_ENSEMBLE_K];
    char run_id[VAE_C_IIC_CHECKPOINT_RUN_ID_CAP];
    char base_commit[VAE_C_IIC_CHECKPOINT_COMMIT_CAP];
    char producer_state[VAE_C_IIC_CHECKPOINT_PRODUCER_CAP];
    char producer_binary_sha256[VAE_C_IIC_CHECKPOINT_SHA256_CAP];
} VAECIICCheckpointIdentity;

typedef struct {
    VAECIICCheckpointKind kind;
    uint64_t last_completed_epoch;
} VAECIICCheckpointMetadata;

typedef enum {
    VAE_C_IIC_CHECKPOINT_OK = 0,
    VAE_C_IIC_CHECKPOINT_INVALID_ARGUMENT,
    VAE_C_IIC_CHECKPOINT_INVALID_STATE,
    VAE_C_IIC_CHECKPOINT_IDENTITY_MISMATCH,
    VAE_C_IIC_CHECKPOINT_CORRUPT,
    VAE_C_IIC_CHECKPOINT_IO_ERROR,
    VAE_C_IIC_CHECKPOINT_EXISTS,
    VAE_C_IIC_CHECKPOINT_ALLOCATION_FAILED
} VAECIICCheckpointStatus;

void vae_c_iic_checkpoint_identity_init(VAECIICCheckpointIdentity *identity);
void vae_c_iic_checkpoint_metadata_init(VAECIICCheckpointMetadata *metadata);

/* LATEST atomically replaces an existing LATEST. FINAL refuses overwrite.
 * The exclusive sibling path is path.partial. Neither form serializes a
 * workspace, backward ticket, pointer, gradient, or RNG stream. */
VAECIICCheckpointStatus vae_c_iic_checkpoint_save_atomic(
    const char *path,
    const VAECIICCheckpointIdentity *identity,
    const VAECIICCheckpointMetadata *metadata,
    const VAECIICEnsemble *ensemble
);

/* Validates complete file/checksum/identity/config/numerics into private
 * scratch before modifying the live ensemble. The destination must already
 * be allocated with matching dimensions and have no live backward context.
 * On success, all gradients are zero and no backward context remains. */
VAECIICCheckpointStatus vae_c_iic_checkpoint_load(
    const char *path,
    const VAECIICCheckpointIdentity *expected_identity,
    VAECIICEnsemble *ensemble,
    VAECIICCheckpointMetadata *metadata_out
);

const char *vae_c_iic_checkpoint_status_name(VAECIICCheckpointStatus status);

#endif
