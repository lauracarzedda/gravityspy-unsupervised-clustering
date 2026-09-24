#ifndef VAE_C_MU_CONTRACT_H
#define VAE_C_MU_CONTRACT_H

#include "vae_c_checkpoint.h"
#include "vae_c_config.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


#define VAE_C_MU_ARTIFACT_V1 "VAE_C_MU_ARTIFACT_V1"
#define VAE_C_MU_PATH_CAP 4096u
#define VAE_C_MU_TEXT_CAP 128u


typedef enum {
    VAE_C_MU_OK = 0,
    VAE_C_MU_INVALID_ARGUMENT,
    VAE_C_MU_UNSUPPORTED_PROFILE,
    VAE_C_MU_PATH_OVERFLOW,
    VAE_C_MU_IO_ERROR,
    VAE_C_MU_HASH_MISMATCH,
    VAE_C_MU_RUN_CONFIG_MISMATCH,
    VAE_C_MU_PROFILE_MISMATCH,
    VAE_C_MU_DATASET_MISMATCH,
    VAE_C_MU_IDENTITY_MISMATCH,
    VAE_C_MU_PLATFORM_MISMATCH,
    VAE_C_MU_MODEL_SETUP_FAILED,
    VAE_C_MU_CHECKPOINT_LOAD_FAILED,
    VAE_C_MU_CHECKPOINT_METADATA_MISMATCH,
    VAE_C_MU_OVERFLOW
} VAECMuStatus;


typedef struct {
    char matrix_file[VAE_C_MU_TEXT_CAP];
    char row_map_file[VAE_C_MU_TEXT_CAP];
    uint64_t rows;
    uint64_t cols;
    uint64_t expected_bytes;
} VAECMuArtifactSplitV1;


/*
 * PROJECT: contract for a deterministic posterior-mu artifact.  The view is
 * explicit so a future perturbed view cannot be mistaken for the central
 * contract.  Construction and checkpoint validation remain write-free;
 * productive publication is owned by the extractor tool.
 */
typedef struct {
    char format[VAE_C_MU_TEXT_CAP];
    char embedding_spec_file[VAE_C_MU_TEXT_CAP];
    char checksum_file[VAE_C_MU_TEXT_CAP];
    char representation[VAE_C_MU_TEXT_CAP];
    char view[VAE_C_MU_TEXT_CAP];
    char row_order[VAE_C_MU_TEXT_CAP];
    char row_map_schema[256];
    char profile_id[VAE_C_PROFILE_TEXT_CAP];
    uint64_t latent_dim;
    char profile_config_sha256[VAE_C_CHECKPOINT_SHA256_CAP];
    char run_id[VAE_C_MU_TEXT_CAP];
    char run_config_sha256[VAE_C_CHECKPOINT_SHA256_CAP];

    char checkpoint_kind[VAE_C_MU_TEXT_CAP];
    uint64_t checkpoint_epoch;
    uint64_t checkpoint_optimizer_step;
    char checkpoint_sha256[VAE_C_CHECKPOINT_SHA256_CAP];
    char checkpoint_code_commit[VAE_C_CHECKPOINT_CODE_COMMIT_CAP];

    char catalog_sha256[VAE_C_CHECKPOINT_SHA256_CAP];
    char event_manifest_sha256[VAE_C_CHECKPOINT_SHA256_CAP];
    char tensor_manifest_sha256[VAE_C_CHECKPOINT_SHA256_CAP];
    char train_split_sha256[VAE_C_CHECKPOINT_SHA256_CAP];
    char validation_split_sha256[VAE_C_CHECKPOINT_SHA256_CAP];
    char train_subset_sha256[VAE_C_CHECKPOINT_SHA256_CAP];
    char validation_subset_sha256[VAE_C_CHECKPOINT_SHA256_CAP];
    char population_identity[VAE_C_MU_TEXT_CAP];
    char checkpoint_dataset_id[VAE_C_CHECKPOINT_ID_CAP];
    char checkpoint_train_selection_id[VAE_C_CHECKPOINT_ID_CAP];
    char checkpoint_validation_selection_id[VAE_C_CHECKPOINT_ID_CAP];

    bool encoder_only;
    char batch_norm_mode[VAE_C_MU_TEXT_CAP];
    bool epsilon_used;
    bool latent_sampled;
    bool decoder_executed;
    bool labels_used;

    char dtype[VAE_C_MU_TEXT_CAP];
    char endian[VAE_C_MU_TEXT_CAP];
    char layout[VAE_C_MU_TEXT_CAP];
    char input_normalization[VAE_C_MU_TEXT_CAP];
    char latent_scaling[VAE_C_MU_TEXT_CAP];
    VAECMuArtifactSplitV1 train;
    VAECMuArtifactSplitV1 validation;
} VAECMuArtifactContractV1;


/*
 * Fully resolved expected contract.  Every identity field is constructed
 * from the canonical registry plus independently hashed repository/run
 * inputs before the checkpoint is opened.  Tests may copy and mutate this
 * value to prove that negative gates reject mismatches.
 */
typedef struct {
    char run_token[VAE_C_MU_TEXT_CAP];
    char repository_root[VAE_C_MU_PATH_CAP];
    char run_id[VAE_C_MU_TEXT_CAP];

    char catalog_path[VAE_C_MU_PATH_CAP];
    char event_manifest_path[VAE_C_MU_PATH_CAP];
    char tensor_manifest_path[VAE_C_MU_PATH_CAP];
    char official_train_split_path[VAE_C_MU_PATH_CAP];
    char official_validation_split_path[VAE_C_MU_PATH_CAP];
    char train_population_path[VAE_C_MU_PATH_CAP];
    char validation_population_path[VAE_C_MU_PATH_CAP];
    char run_config_path[VAE_C_MU_PATH_CAP];
    char checkpoint_path[VAE_C_MU_PATH_CAP];

    char run_config_sha256[VAE_C_CHECKPOINT_SHA256_CAP];
    char checkpoint_sha256[VAE_C_CHECKPOINT_SHA256_CAP];
    char train_subset_sha256[VAE_C_CHECKPOINT_SHA256_CAP];
    char validation_subset_sha256[VAE_C_CHECKPOINT_SHA256_CAP];

    VAECProfile profile;
    VAECCheckpointRunIdentity expected_identity;
    VAECCheckpointKind expected_kind;
    uint64_t expected_epoch;
    uint64_t expected_optimizer_step;
    uint64_t event_count;
    uint64_t train_count;
    uint64_t validation_count;
    bool uses_subset;

    VAECMuArtifactContractV1 artifact_v1;
} VAECMuCanonicalContract;


typedef struct {
    bool source_contract_validated;
    bool checkpoint_hash_verified;
    bool checkpoint_hash_reverified_after_load;
    bool checkpoint_load_called;
    bool checkpoint_loaded;
    bool metadata_verified;
    VAECCheckpointStatus checkpoint_status;
    VAECCheckpointMetadata loaded_metadata;
    uint64_t loaded_optimizer_step;

    uint64_t neural_workspace_allocations;
    uint64_t neural_batch_loads;
    uint64_t encoder_forward_calls;
    uint64_t backward_calls;
    uint64_t adam_steps;
    uint64_t mu_values_created;
    uint64_t artifact_files_created;
} VAECMuValidationReport;


void vae_c_mu_canonical_contract_init(VAECMuCanonicalContract *contract);

void vae_c_mu_validation_report_init(VAECMuValidationReport *report);

/* Accepts canonical run tokens mini75/mini112/mini224 or their profile IDs. */
VAECMuStatus vae_c_mu_build_canonical_contract(
    const char *repository_root,
    const char *run_or_profile,
    VAECMuCanonicalContract *contract,
    char *error_message,
    size_t error_message_capacity
);

/* Same canonical verification, but the event-manifest parser does not copy
 * or inspect the Gravity Spy label column. Intended for label-free IIC. */
VAECMuStatus vae_c_mu_build_canonical_contract_label_free(
    const char *repository_root,
    const char *run_or_profile,
    VAECMuCanonicalContract *contract,
    char *error_message,
    size_t error_message_capacity
);

/*
 * Hash and load the checkpoint through vae_c_checkpoint_load, then enforce
 * the canonical BEST/epoch/optimizer-step postconditions.  This function
 * allocates no model workspace or data batch and contains no forward path.
 */
VAECMuStatus vae_c_mu_validate_checkpoint(
    const VAECMuCanonicalContract *contract,
    VAECMuValidationReport *report,
    char *error_message,
    size_t error_message_capacity
);

VAECMuStatus vae_c_mu_validate_only(
    const char *repository_root,
    const char *run_or_profile,
    VAECMuCanonicalContract *contract,
    VAECMuValidationReport *report,
    char *error_message,
    size_t error_message_capacity
);

VAECMuStatus vae_c_mu_validate_only_label_free(
    const char *repository_root,
    const char *run_or_profile,
    VAECMuCanonicalContract *contract,
    VAECMuValidationReport *report,
    char *error_message,
    size_t error_message_capacity
);

/* Shared one-shot file hashing built on the repository SHA-256 primitive. */
VAECMuStatus vae_c_mu_file_sha256(
    const char *path,
    char sha256[VAE_C_CHECKPOINT_SHA256_CAP]
);

const char *vae_c_mu_status_name(VAECMuStatus status);

#endif
