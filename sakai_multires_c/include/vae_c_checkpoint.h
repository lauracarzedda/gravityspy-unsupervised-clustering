#ifndef VAE_C_CHECKPOINT_H
#define VAE_C_CHECKPOINT_H

#include "vae_c_adam.h"
#include "vae_c_model.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


#define VAE_C_CHECKPOINT_VERSION 1u
#define VAE_C_CHECKPOINT_CODE_COMMIT_CAP 41u
#define VAE_C_CHECKPOINT_SHA256_CAP 65u
#define VAE_C_CHECKPOINT_ID_CAP 128u
#define VAE_C_CHECKPOINT_POLICY_CAP 64u
#define VAE_C_CHECKPOINT_PROVENANCE_CAP 128u


typedef enum {
    VAE_C_CHECKPOINT_OK = 0,
    VAE_C_CHECKPOINT_INVALID_ARGUMENT,
    VAE_C_CHECKPOINT_INVALID_STATE,
    VAE_C_CHECKPOINT_IO_ERROR,
    VAE_C_CHECKPOINT_TEMP_EXISTS,
    VAE_C_CHECKPOINT_FORMAT_ERROR,
    VAE_C_CHECKPOINT_UNSUPPORTED_VERSION,
    VAE_C_CHECKPOINT_CHECKSUM_MISMATCH,
    VAE_C_CHECKPOINT_INCOMPATIBLE,
    VAE_C_CHECKPOINT_OVERFLOW,
    VAE_C_CHECKPOINT_ALLOCATION_FAILED
} VAECCheckpointStatus;


typedef enum {
    VAE_C_CHECKPOINT_KIND_LATEST = 1,
    VAE_C_CHECKPOINT_KIND_BEST = 2,
    VAE_C_CHECKPOINT_KIND_FINAL = 3
} VAECCheckpointKind;


/*
 * PROJECT: immutable identity of one training run.  Text hashes are
 * lowercase or uppercase hexadecimal SHA-256 strings.  Compiler fields are
 * provenance only: the checkpoint core records and round-trips them but does
 * not use them as an automatic compatibility gate.
 */
typedef struct {
    char profile_id[VAE_C_PROFILE_TEXT_CAP];
    char profile_config_sha256[VAE_C_CHECKPOINT_SHA256_CAP];
    char code_commit[VAE_C_CHECKPOINT_CODE_COMMIT_CAP];

    char dataset_id[VAE_C_CHECKPOINT_ID_CAP];
    char event_manifest_id[VAE_C_CHECKPOINT_ID_CAP];
    char event_manifest_sha256[VAE_C_CHECKPOINT_SHA256_CAP];
    char tensor_manifest_id[VAE_C_CHECKPOINT_ID_CAP];
    char tensor_manifest_sha256[VAE_C_CHECKPOINT_SHA256_CAP];
    char train_split_id[VAE_C_CHECKPOINT_ID_CAP];
    char train_split_sha256[VAE_C_CHECKPOINT_SHA256_CAP];
    char validation_split_id[VAE_C_CHECKPOINT_ID_CAP];
    char validation_split_sha256[VAE_C_CHECKPOINT_SHA256_CAP];

    char augmentation_policy[VAE_C_CHECKPOINT_POLICY_CAP];
    char shuffle_policy[VAE_C_CHECKPOINT_POLICY_CAP];
    char latent_policy[VAE_C_CHECKPOINT_POLICY_CAP];
    char validation_policy[VAE_C_CHECKPOINT_POLICY_CAP];

    char compiler_id[VAE_C_CHECKPOINT_PROVENANCE_CAP];
    char compiler_version[VAE_C_CHECKPOINT_PROVENANCE_CAP];
    char compiler_target[VAE_C_CHECKPOINT_PROVENANCE_CAP];

    uint64_t initialization_seed;
    uint64_t latent_seed;
    uint64_t augmentation_seed;
    uint64_t shuffle_seed;
    uint64_t max_batch_size;
    uint64_t batch_size;

    double optimizer_learning_rate;
    double optimizer_beta1;
    double optimizer_beta2;
    double optimizer_epsilon;
} VAECCheckpointRunIdentity;


typedef struct {
    VAECCheckpointRunIdentity run_identity;
    VAECCheckpointKind kind;
    uint64_t last_completed_epoch;

    bool best_defined;
    uint64_t best_epoch;
    double best_validation_total_mean_per_sample;
    uint64_t no_improvement_count;
} VAECCheckpointMetadata;


void vae_c_checkpoint_run_identity_init(
    VAECCheckpointRunIdentity *identity
);

void vae_c_checkpoint_metadata_init(VAECCheckpointMetadata *metadata);

/*
 * PROJECT: write a complete v1 checkpoint to path.tmp, close and re-read it,
 * validate the encoded state and SHA-256, then atomically publish it at path.
 * A pre-existing path.tmp is never removed or overwritten.
 */
VAECCheckpointStatus vae_c_checkpoint_save_atomic(
    const char *path,
    const VAECModel *model,
    const VAECAdam *optimizer,
    const VAECCheckpointMetadata *metadata
);

/*
 * PROJECT: the destination model must already be allocated and initialized
 * with a compatible profile and maximum batch size; optimizer must be
 * configured.  Parsing and validation use temporary storage.  On failure,
 * model, optimizer, and loaded_metadata are unchanged.
 *
 * Successful load restores only persistent state.  It invalidates the
 * current model cache generation by advancing it exactly once, so every
 * existing backward ticket is stale without generation wraparound.  Load is
 * rejected atomically unless the destination generation has room for this
 * increment and the next full VAE forward (+2).  Training code must
 * discard/recreate its workspace after resume.
 * The output metadata object must not overlap path, expected_identity,
 * model, optimizer, or their owned storage. Exact top-level aliases are
 * rejected; partial overlap between distinct C objects is outside contract.
 */
VAECCheckpointStatus vae_c_checkpoint_load(
    const char *path,
    const VAECCheckpointRunIdentity *expected_identity,
    VAECModel *model,
    VAECAdam *optimizer,
    VAECCheckpointMetadata *loaded_metadata
);

const char *vae_c_checkpoint_status_name(VAECCheckpointStatus status);

#endif
