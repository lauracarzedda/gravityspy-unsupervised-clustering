#include "vae_c_iic_checkpoint.h"

#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* This inspector has no output path and only invokes the checkpoint load API.
 * Identity values must come from an independently verified run config. */
static bool number(const char *text, uint64_t *value)
{
    char *end;
    unsigned long long parsed;
    if (text == NULL || text[0] == '\0' || text[0] == '-' ||
        text[0] == '+') return false;
    errno = 0;
    parsed = strtoull(text, &end, 10);
    if (errno != 0 || *end != '\0') return false;
    *value = (uint64_t)parsed;
    return (unsigned long long)*value == parsed;
}

static bool copy_text(char *dst, size_t capacity, const char *src)
{
    size_t length;
    if (src == NULL) return false;
    length = strlen(src);
    if (length == 0u || length >= capacity) return false;
    memcpy(dst, src, length + 1u);
    return true;
}

static bool identity_from_args(
    int argc, char **argv, const char **checkpoint,
    VAECIICCheckpointIdentity *identity,
    VAECIICCheckpointKind *expected_kind, uint64_t *expected_epoch
)
{
    uint64_t parsed, seen = 0u;
    int index;
    static const char *const names[] = {
        "--checkpoint", "--input-dim", "--capacity", "--batch-size",
        "--train-rows", "--max-epochs", "--shuffle-seed", "--run-id",
        "--base-commit", "--producer-state", "--producer-binary-sha256",
        "--expected-kind", "--expected-epoch"
    };
    const uint64_t all = (UINT64_C(1) <<
        (sizeof(names) / sizeof(names[0]))) - UINT64_C(1);
    if (argc != 27) return false;
    vae_c_iic_checkpoint_identity_init(identity);
    for (index = 1; index < argc; index += 2) {
        size_t option;
        const char *value = argv[index + 1];
        for (option = 0u; option < sizeof(names) / sizeof(names[0]); option++) {
            if (strcmp(argv[index], names[option]) == 0) break;
        }
        if (option == sizeof(names) / sizeof(names[0]) ||
            (seen & (UINT64_C(1) << option)) != 0u) return false;
        seen |= UINT64_C(1) << option;
        switch (option) {
            case 0: *checkpoint = value; break;
            case 1: case 2: case 3: case 4: case 5: case 6:
            case 12:
                if (!number(value, &parsed)) return false;
                if (option <= 4 && parsed > SIZE_MAX) return false;
                if (option == 1) identity->input_dim = (size_t)parsed;
                if (option == 2) identity->capacity = (size_t)parsed;
                if (option == 3) identity->batch_size = (size_t)parsed;
                if (option == 4) identity->train_rows = (size_t)parsed;
                if (option == 5) identity->max_epochs = parsed;
                if (option == 6) identity->shuffle_seed = parsed;
                if (option == 12) *expected_epoch = parsed;
                break;
            case 7:
                if (!copy_text(identity->run_id, sizeof(identity->run_id),
                               value)) return false;
                break;
            case 8:
                if (!copy_text(identity->base_commit,
                               sizeof(identity->base_commit), value))
                    return false;
                break;
            case 9:
                if (!copy_text(identity->producer_state,
                               sizeof(identity->producer_state), value))
                    return false;
                break;
            case 10:
                if (!copy_text(identity->producer_binary_sha256,
                               sizeof(identity->producer_binary_sha256),
                               value)) return false;
                break;
            case 11:
                if (strcmp(value, "LATEST") == 0)
                    *expected_kind = VAE_C_IIC_CHECKPOINT_LATEST;
                else if (strcmp(value, "FINAL") == 0)
                    *expected_kind = VAE_C_IIC_CHECKPOINT_FINAL;
                else return false;
                break;
            default: return false;
        }
    }
    for (index = 0; index < VAE_C_IIC_ENSEMBLE_K; index++) {
        identity->seeds[index] = VAE_C_IIC_ENSEMBLE_FIRST_SEED +
            (uint64_t)index;
    }
    return seen == all && *checkpoint != NULL && (*checkpoint)[0] != '\0';
}

int main(int argc, char **argv)
{
    const char *checkpoint = NULL;
    VAECIICCheckpointIdentity identity;
    VAECIICCheckpointMetadata metadata;
    VAECIICCheckpointKind expected_kind = 0;
    VAECIICEnsemble *ensemble;
    VAECIICCheckpointStatus status;
    uint64_t expected_epoch = UINT64_MAX;
    size_t pair_index;
    if (!identity_from_args(argc, argv, &checkpoint, &identity,
                            &expected_kind, &expected_epoch)) {
        fputs("Invalid explicit checkpoint identity arguments\n", stderr);
        return 2;
    }
    ensemble = (VAECIICEnsemble *)malloc(sizeof(*ensemble));
    if (ensemble == NULL) return 2;
    vae_c_iic_ensemble_init(ensemble);
    if (vae_c_iic_ensemble_allocate_v1(
            ensemble, identity.input_dim, identity.capacity) != VAE_C_OP_OK) {
        free(ensemble);
        return 2;
    }
    vae_c_iic_checkpoint_metadata_init(&metadata);
    status = vae_c_iic_checkpoint_load(
        checkpoint, &identity, ensemble, &metadata);
    if (status != VAE_C_IIC_CHECKPOINT_OK ||
        metadata.kind != expected_kind ||
        metadata.last_completed_epoch != expected_epoch) {
        fprintf(stderr, "LOAD_PASS=false API_STATUS=%s\n",
                vae_c_iic_checkpoint_status_name(status));
        vae_c_iic_ensemble_free(ensemble);
        free(ensemble);
        return 1;
    }
    puts("LOAD_PASS=true");
    puts("IDENTITY_CONFIG_MATCH=true");
    puts("STATE_FINITE=true");
    printf("K=%u\n", (unsigned)VAE_C_IIC_ENSEMBLE_K);
    printf("KIND=%s\n", metadata.kind == VAE_C_IIC_CHECKPOINT_LATEST ?
           "LATEST" : "FINAL");
    printf("EPOCH=%" PRIu64 "\n", metadata.last_completed_epoch);
    for (pair_index = 0u; pair_index < VAE_C_IIC_ENSEMBLE_K;
         pair_index++) {
        const VAECIICPair *pair = &ensemble->pairs[pair_index];
        printf("PAIR%zu_STEP=%" PRIu64 "\n", pair_index,
               ensemble->optimizers[pair_index].step);
        printf("PAIR%zu_MAIN_BN_BATCHES=%" PRIu64 "\n", pair_index,
               pair->main_head.num_batches_tracked);
        printf("PAIR%zu_OVER_BN_BATCHES=%" PRIu64 "\n", pair_index,
               pair->overcluster_head.num_batches_tracked);
    }
    vae_c_iic_ensemble_free(ensemble);
    free(ensemble);
    return 0;
}
