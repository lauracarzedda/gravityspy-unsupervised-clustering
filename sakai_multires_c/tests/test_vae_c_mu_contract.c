#include "vae_c_mu_contract.h"
#include "vae_c_model_forward.h"

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


enum {
    TEST_PATH_CAP = 4096,
    TEST_ERROR_CAP = 512,
    TEST_CANONICAL_ASSET_COUNT = 6
};


static uint64_t intercepted_encoder_forward_calls = UINT64_C(0);


/*
 * Test link seam: if the validate-only library ever starts calling the
 * encoder, this definition intercepts the call before the production encoder
 * object is pulled from the static core library.
 */
VAECOpStatus vae_c_model_encoder_forward(
    VAECModel *model,
    VAECBatchNormMode batch_norm_mode,
    VAECModelWorkspace *workspace
)
{
    (void)model;
    (void)batch_norm_mode;
    (void)workspace;
    intercepted_encoder_forward_calls++;
    return VAE_C_OP_INVALID_STATE;
}


static int report_check(const char *name, int passed)
{
    if (passed) {
        printf("[PASS] %s\n", name);
        return 0;
    }
    printf("[FAIL] %s\n", name);
    return 1;
}


static int join_path(
    const char *root,
    const char *relative,
    char path[TEST_PATH_CAP]
)
{
    size_t length;
    const char *separator;
    int result;

    if (root == NULL || relative == NULL || path == NULL) {
        return 0;
    }
    length = strlen(root);
    separator = length != 0u &&
        (root[length - 1u] == '/' || root[length - 1u] == '\\') ? "" : "/";
    result = snprintf(path, TEST_PATH_CAP, "%s%s%s",
                      root, separator, relative);
    return result >= 0 && (size_t)result < TEST_PATH_CAP;
}


static int file_exists(const char *path)
{
    FILE *file = fopen(path, "rb");

    if (file == NULL) {
        return 0;
    }
    return fclose(file) == 0;
}


static int canonical_ignored_asset_count(const char *root)
{
    static const char *const paths[] = {
        "sakai_multires_c/results/vae_c/training/"
            "mini75_experiment_v1_full_init123_latent456_aug789_"
            "shuffle101112_bs64/run_config.txt",
        "sakai_multires_c/results/vae_c/training/"
            "mini75_experiment_v1_full_init123_latent456_aug789_"
            "shuffle101112_bs64/best.ckpt",
        "sakai_multires_c/results/vae_c/training/"
            "mini112_primary_v1_full_init123_latent456_aug789_"
            "shuffle101112_bs64/run_config.txt",
        "sakai_multires_c/results/vae_c/training/"
            "mini112_primary_v1_full_init123_latent456_aug789_"
            "shuffle101112_bs64/best.ckpt",
        "sakai_multires_c/results/vae_c/training/"
            "mini224_bridge_v1_subset1000_candidate_init123_latent456_"
            "aug789_shuffle101112_bs16_max100_pat20/run_config.txt",
        "sakai_multires_c/results/vae_c/training/"
            "mini224_bridge_v1_subset1000_candidate_init123_latent456_"
            "aug789_shuffle101112_bs16_max100_pat20/best.ckpt"
    };
    char path[TEST_PATH_CAP];
    size_t index;

    int present = 0;

    for (index = 0; index < sizeof(paths) / sizeof(paths[0]); index++) {
        if (join_path(root, paths[index], path) && file_exists(path)) {
            present++;
        }
    }
    return present;
}


static int no_inference_or_artifact(const VAECMuValidationReport *report)
{
    return report != NULL &&
        report->neural_workspace_allocations == UINT64_C(0) &&
        report->neural_batch_loads == UINT64_C(0) &&
        report->encoder_forward_calls == UINT64_C(0) &&
        report->backward_calls == UINT64_C(0) &&
        report->adam_steps == UINT64_C(0) &&
        report->mu_values_created == UINT64_C(0) &&
        report->artifact_files_created == UINT64_C(0) &&
        intercepted_encoder_forward_calls == UINT64_C(0);
}


static int validate_positive(
    const VAECMuCanonicalContract *contract,
    uint64_t epoch,
    uint64_t step
)
{
    VAECMuValidationReport report;
    char error[TEST_ERROR_CAP];

    vae_c_mu_validation_report_init(&report);
    error[0] = '\0';
    return vae_c_mu_validate_checkpoint(
               contract, &report, error, sizeof(error)) == VAE_C_MU_OK &&
        report.source_contract_validated &&
        report.checkpoint_hash_verified &&
        report.checkpoint_hash_reverified_after_load &&
        report.checkpoint_load_called &&
        report.checkpoint_loaded && report.metadata_verified &&
        report.checkpoint_status == VAE_C_CHECKPOINT_OK &&
        report.loaded_metadata.kind == VAE_C_CHECKPOINT_KIND_BEST &&
        report.loaded_metadata.last_completed_epoch == epoch &&
        report.loaded_optimizer_step == step &&
        no_inference_or_artifact(&report);
}


static void mutate_text(char *text)
{
    if (text != NULL && text[0] != '\0') {
        text[0] = text[0] == '0' ? '1' : '0';
    }
}


static int sha256_equal_case_insensitive(
    const char *left,
    const char *right
)
{
    size_t index;

    if (left == NULL || right == NULL || strlen(left) != 64u ||
        strlen(right) != 64u) {
        return 0;
    }
    for (index = 0; index < 64u; index++) {
        if (tolower((unsigned char)left[index]) !=
            tolower((unsigned char)right[index])) {
            return 0;
        }
    }
    return 1;
}


static int expect_checkpoint_failure(
    const VAECMuCanonicalContract *contract,
    VAECMuStatus expected_status,
    VAECCheckpointStatus expected_checkpoint_status,
    int expected_load_called
)
{
    VAECMuValidationReport report;
    VAECMuStatus status;
    char error[TEST_ERROR_CAP];

    vae_c_mu_validation_report_init(&report);
    error[0] = '\0';
    status = vae_c_mu_validate_checkpoint(
        contract, &report, error, sizeof(error)
    );
    return status == expected_status &&
        report.checkpoint_load_called == (expected_load_called != 0) &&
        (!expected_load_called ||
         report.checkpoint_status == expected_checkpoint_status) &&
        !report.metadata_verified && no_inference_or_artifact(&report);
}


static int copy_with_corrupt_digest(
    const char *source_path,
    const char *destination_path
)
{
    FILE *source = NULL;
    FILE *destination = NULL;
    unsigned char *bytes = NULL;
    long length;
    size_t size;
    int passed = 0;

    source = fopen(source_path, "rb");
    if (source == NULL || fseek(source, 0, SEEK_END) != 0 ||
        (length = ftell(source)) <= 0 || fseek(source, 0, SEEK_SET) != 0) {
        goto cleanup;
    }
    size = (size_t)length;
    if ((long)size != length) {
        goto cleanup;
    }
    bytes = (unsigned char *)malloc(size);
    if (bytes == NULL) {
        goto cleanup;
    }
    {
        int read_ok = fread(bytes, 1u, size, source) == size;
        int close_status = fclose(source);

        source = NULL;
        if (!read_ok || close_status != 0) {
            goto cleanup;
        }
    }
    bytes[size - 1u] ^= 0x01u;
    destination = fopen(destination_path, "wb");
    if (destination == NULL) {
        goto cleanup;
    }
    {
        int write_ok = fwrite(bytes, 1u, size, destination) == size;
        int close_status = fclose(destination);

        destination = NULL;
        if (!write_ok || close_status != 0) {
            goto cleanup;
        }
    }
    passed = 1;

cleanup:
    if (destination != NULL) {
        (void)fclose(destination);
    }
    if (source != NULL) {
        (void)fclose(source);
    }
    free(bytes);
    return passed;
}


static int artifact_schema_is_valid(
    const VAECMuCanonicalContract *contract,
    uint64_t expected_train_bytes,
    uint64_t expected_validation_bytes
)
{
    const VAECMuArtifactContractV1 *artifact = &contract->artifact_v1;

    return strcmp(artifact->format, VAE_C_MU_ARTIFACT_V1) == 0 &&
        strcmp(artifact->embedding_spec_file, "embedding_spec.txt") == 0 &&
        strcmp(artifact->checksum_file, "sha256.txt") == 0 &&
        strcmp(artifact->representation, "posterior_mu") == 0 &&
        strcmp(artifact->view, "central") == 0 &&
        strcmp(artifact->row_order, "native_split_pos_ascending") == 0 &&
        strcmp(artifact->row_map_schema,
               "row_index,split_pos,manifest_index,event_index,event_id,split") == 0 &&
        strcmp(artifact->profile_id, contract->profile.profile_id) == 0 &&
        artifact->latent_dim == contract->profile.latent_dim &&
        sha256_equal_case_insensitive(
            artifact->profile_config_sha256,
            contract->expected_identity.profile_config_sha256) &&
        strcmp(artifact->run_id, contract->run_id) == 0 &&
        strcmp(artifact->run_config_sha256,
               contract->run_config_sha256) == 0 &&
        strcmp(artifact->checkpoint_kind, "BEST") == 0 &&
        artifact->checkpoint_epoch == contract->expected_epoch &&
        artifact->checkpoint_optimizer_step ==
            contract->expected_optimizer_step &&
        strcmp(artifact->checkpoint_sha256,
               contract->checkpoint_sha256) == 0 &&
        strcmp(artifact->checkpoint_code_commit,
               contract->expected_identity.code_commit) == 0 &&
        sha256_equal_case_insensitive(
            artifact->catalog_sha256,
            contract->expected_identity.profile_config_sha256) &&
        sha256_equal_case_insensitive(
            artifact->event_manifest_sha256,
            contract->expected_identity.event_manifest_sha256) &&
        sha256_equal_case_insensitive(
            artifact->tensor_manifest_sha256,
            contract->expected_identity.tensor_manifest_sha256) &&
        sha256_equal_case_insensitive(
            artifact->train_split_sha256,
            contract->expected_identity.train_split_sha256) &&
        sha256_equal_case_insensitive(
            artifact->validation_split_sha256,
            contract->expected_identity.validation_split_sha256) &&
        strcmp(artifact->population_identity,
               contract->uses_subset
                   ? "mini224_subset_seed123_prefix_800_200_v1"
                   : "gravity_spy_o1_full_split_seed123_v1") == 0 &&
        strcmp(artifact->checkpoint_dataset_id,
               contract->expected_identity.dataset_id) == 0 &&
        strcmp(artifact->checkpoint_train_selection_id,
               contract->expected_identity.train_split_id) == 0 &&
        strcmp(artifact->checkpoint_validation_selection_id,
               contract->expected_identity.validation_split_id) == 0 &&
        (contract->uses_subset
             ? sha256_equal_case_insensitive(
                   artifact->train_subset_sha256,
                   contract->train_subset_sha256) &&
               sha256_equal_case_insensitive(
                   artifact->validation_subset_sha256,
                   contract->validation_subset_sha256)
             : strcmp(artifact->train_subset_sha256,
                      "not_applicable_full_split") == 0 &&
               strcmp(artifact->validation_subset_sha256,
                      "not_applicable_full_split") == 0) &&
        strcmp(artifact->batch_norm_mode, "EVAL") == 0 &&
        artifact->encoder_only && !artifact->epsilon_used &&
        !artifact->latent_sampled && !artifact->decoder_executed &&
        !artifact->labels_used &&
        strcmp(artifact->dtype, "float32_ieee754") == 0 &&
        strcmp(artifact->endian, "little") == 0 &&
        strcmp(artifact->layout, "row_major_[N,latent_dim]") == 0 &&
        strcmp(artifact->input_normalization, "uint8_div_255") == 0 &&
        strcmp(artifact->latent_scaling, "none") == 0 &&
        strcmp(artifact->train.matrix_file, "train_mu_f32.bin") == 0 &&
        strcmp(artifact->train.row_map_file, "train_rows.csv") == 0 &&
        strcmp(artifact->validation.matrix_file,
               "validation_mu_f32.bin") == 0 &&
        strcmp(artifact->validation.row_map_file,
               "validation_rows.csv") == 0 &&
        artifact->train.rows == contract->train_count &&
        artifact->validation.rows == contract->validation_count &&
        artifact->train.cols == contract->profile.latent_dim &&
        artifact->validation.cols == contract->profile.latent_dim &&
        artifact->train.expected_bytes == expected_train_bytes &&
        artifact->validation.expected_bytes == expected_validation_bytes;
}


int main(int argc, char **argv)
{
    VAECMuCanonicalContract mini75;
    VAECMuCanonicalContract mini112;
    VAECMuCanonicalContract mini224;
    VAECMuCanonicalContract negative;
    VAECMuValidationReport report;
    VAECMuStatus status;
    char corrupt_path[TEST_PATH_CAP];
    char corrupt_sha[VAE_C_CHECKPOINT_SHA256_CAP];
    char error[TEST_ERROR_CAP];
    int failures = 0;
    int cleanup_ok = 1;
    int canonical_asset_count;

    if (argc != 3) {
        fprintf(stderr, "Usage: %s REPOSITORY_ROOT FIXTURE_DIRECTORY\n",
                argv[0]);
        return 2;
    }
    canonical_asset_count = canonical_ignored_asset_count(argv[1]);
    if (canonical_asset_count == 0) {
        printf("[SKIP] canonical ignored run artifacts are absent\n");
        return 77;
    }
    if (canonical_asset_count != TEST_CANONICAL_ASSET_COUNT) {
        printf("[FAIL] canonical ignored run artifacts are partially present "
               "(%d/%d)\n", canonical_asset_count,
               TEST_CANONICAL_ASSET_COUNT);
        return 1;
    }

    vae_c_mu_canonical_contract_init(&mini75);
    vae_c_mu_canonical_contract_init(&mini112);
    vae_c_mu_canonical_contract_init(&mini224);
    error[0] = '\0';
    failures += report_check(
        "build canonical mini75 identity from external sources",
        vae_c_mu_build_canonical_contract(
            argv[1], "mini75", &mini75, error, sizeof(error)) == VAE_C_MU_OK
    );
    error[0] = '\0';
    failures += report_check(
        "build canonical mini112 identity from external sources",
        vae_c_mu_build_canonical_contract(
            argv[1], "mini112", &mini112, error, sizeof(error)) == VAE_C_MU_OK
    );
    error[0] = '\0';
    failures += report_check(
        "build canonical mini224 identity/subset from external sources",
        vae_c_mu_build_canonical_contract(
            argv[1], "mini224", &mini224, error, sizeof(error)) == VAE_C_MU_OK
    );
    if (failures != 0) {
        return 1;
    }

    failures += report_check(
        "positive canonical mini75 BEST epoch47 step5184",
        validate_positive(&mini75, UINT64_C(47), UINT64_C(5184))
    );
    failures += report_check(
        "positive canonical mini112 BEST epoch97 step10584",
        validate_positive(&mini112, UINT64_C(97), UINT64_C(10584))
    );
    failures += report_check(
        "positive canonical mini224 BEST epoch99 step5000",
        validate_positive(&mini224, UINT64_C(99), UINT64_C(5000))
    );

    failures += report_check(
        "VAE_C_MU_ARTIFACT_V1 mini75 schema",
        artifact_schema_is_valid(
            &mini75, UINT64_C(878848), UINT64_C(219776))
    );
    failures += report_check(
        "VAE_C_MU_ARTIFACT_V1 mini112 schema",
        artifact_schema_is_valid(
            &mini112, UINT64_C(1757696), UINT64_C(439552))
    );
    failures += report_check(
        "VAE_C_MU_ARTIFACT_V1 mini224 schema",
        artifact_schema_is_valid(
            &mini224, UINT64_C(204800), UINT64_C(51200))
    );

    negative = mini75;
    memcpy(negative.checkpoint_path, mini112.checkpoint_path,
           sizeof(negative.checkpoint_path));
    memcpy(negative.checkpoint_sha256, mini112.checkpoint_sha256,
           sizeof(negative.checkpoint_sha256));
    failures += report_check(
        "negative wrong checkpoint for profile",
        expect_checkpoint_failure(
            &negative, VAE_C_MU_CHECKPOINT_LOAD_FAILED,
            VAE_C_CHECKPOINT_INCOMPATIBLE, 1)
    );

    negative = mini224;
    mutate_text(negative.expected_identity.dataset_id);
    failures += report_check(
        "negative dataset identity mismatch",
        expect_checkpoint_failure(
            &negative, VAE_C_MU_CHECKPOINT_LOAD_FAILED,
            VAE_C_CHECKPOINT_INCOMPATIBLE, 1)
    );

    negative = mini75;
    mutate_text(negative.expected_identity.train_split_sha256);
    failures += report_check(
        "negative official split identity mismatch",
        expect_checkpoint_failure(
            &negative, VAE_C_MU_CHECKPOINT_LOAD_FAILED,
            VAE_C_CHECKPOINT_INCOMPATIBLE, 1)
    );

    negative = mini224;
    mutate_text(negative.expected_identity.train_split_id);
    failures += report_check(
        "negative subset selection identity mismatch",
        expect_checkpoint_failure(
            &negative, VAE_C_MU_CHECKPOINT_LOAD_FAILED,
            VAE_C_CHECKPOINT_INCOMPATIBLE, 1)
    );

    negative = mini112;
    mutate_text(negative.checkpoint_sha256);
    failures += report_check(
        "negative external checkpoint SHA-256 mismatch before load",
        expect_checkpoint_failure(
            &negative, VAE_C_MU_HASH_MISMATCH,
            VAE_C_CHECKPOINT_INVALID_STATE, 0)
    );

    negative = mini75;
    negative.expected_kind = VAE_C_CHECKPOINT_KIND_FINAL;
    failures += report_check(
        "negative checkpoint kind mismatch",
        expect_checkpoint_failure(
            &negative, VAE_C_MU_CHECKPOINT_METADATA_MISMATCH,
            VAE_C_CHECKPOINT_OK, 1)
    );

    negative = mini112;
    negative.expected_epoch++;
    failures += report_check(
        "negative checkpoint epoch mismatch",
        expect_checkpoint_failure(
            &negative, VAE_C_MU_CHECKPOINT_METADATA_MISMATCH,
            VAE_C_CHECKPOINT_OK, 1)
    );

    negative = mini224;
    negative.expected_optimizer_step++;
    failures += report_check(
        "negative checkpoint optimizer step mismatch",
        expect_checkpoint_failure(
            &negative, VAE_C_MU_CHECKPOINT_METADATA_MISMATCH,
            VAE_C_CHECKPOINT_OK, 1)
    );

    negative = mini112;
    mutate_text(negative.expected_identity.profile_config_sha256);
    failures += report_check(
        "negative catalog/profile config identity mismatch",
        expect_checkpoint_failure(
            &negative, VAE_C_MU_CHECKPOINT_LOAD_FAILED,
            VAE_C_CHECKPOINT_INCOMPATIBLE, 1)
    );

    if (!join_path(argv[2], "corrupt_best.ckpt", corrupt_path)) {
        failures += report_check("corrupt fixture path", 0);
    } else {
        (void)remove(corrupt_path);
        negative = mini75;
        if (!copy_with_corrupt_digest(mini75.checkpoint_path, corrupt_path) ||
            vae_c_mu_file_sha256(corrupt_path, corrupt_sha) != VAE_C_MU_OK) {
            failures += report_check("create isolated corrupt fixture", 0);
        } else {
            memcpy(negative.checkpoint_path, corrupt_path,
                   strlen(corrupt_path) + 1u);
            memcpy(negative.checkpoint_sha256, corrupt_sha,
                   sizeof(negative.checkpoint_sha256));
            failures += report_check(
                "negative corrupt checkpoint internal checksum",
                expect_checkpoint_failure(
                    &negative, VAE_C_MU_CHECKPOINT_LOAD_FAILED,
                    VAE_C_CHECKPOINT_CHECKSUM_MISMATCH, 1)
            );
        }
        cleanup_ok = remove(corrupt_path) == 0;
        failures += report_check("isolated corrupt fixture cleanup", cleanup_ok);
    }

    vae_c_mu_validation_report_init(&report);
    error[0] = '\0';
    status = vae_c_mu_validate_only(
        argv[1], "unsupported", &negative, &report,
        error, sizeof(error)
    );
    failures += report_check(
        "negative unsupported profile before checkpoint load",
        status == VAE_C_MU_UNSUPPORTED_PROFILE &&
        !report.checkpoint_load_called && no_inference_or_artifact(&report)
    );

    if (failures != 0) {
        printf("test_vae_c_mu_contract: %d failure(s)\n", failures);
        return 1;
    }
    printf("test_vae_c_mu_contract: PASS\n");
    return 0;
}
