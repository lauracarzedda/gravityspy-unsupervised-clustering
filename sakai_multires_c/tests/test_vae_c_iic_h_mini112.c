#include "vae_c_iic_checkpoint.h"
#include "vae_c_iic_h.h"
#include "vae_c_iic_mini112_central.h"
#include "vae_c_mu_contract.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REAL_ROWS 127u
#define REAL_DIM VAE_C_IIC_MINI112_LATENT_DIM
#define REAL_COLS VAE_C_IIC_H_COLUMNS
#define REAL_RUN_ID "mini112_iic_v1_0cecf5cd656d0d61a237d50fcf96a7890fb2e6d052ef403051bc9bc17aed8fba"
#define REAL_FINAL_SHA "c4a989377434ddc2f589cd76362e9263078411423e30144d4137a686bcf76f60"

static uint64_t digest(uint64_t hash, const void *data, size_t bytes)
{
    const unsigned char *p = (const unsigned char *)data;
    size_t i;
    for (i = 0u; i < bytes; i++) {
        hash ^= p[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static uint64_t persistent_digest(const VAECIICEnsemble *ensemble)
{
    uint64_t hash = UINT64_C(14695981039346656037);
    size_t k, branch, j;
    for (k = 0u; k < VAE_C_IIC_ENSEMBLE_K; k++) {
        const VAECIICHead *heads[2] = {
            &ensemble->pairs[k].main_head, &ensemble->pairs[k].overcluster_head
        };
        const VAECAdam *adam = &ensemble->optimizers[k];
        hash = digest(hash, &adam->step, sizeof(adam->step));
        hash = digest(hash, &adam->beta1_power, sizeof(adam->beta1_power));
        hash = digest(hash, &adam->beta2_power, sizeof(adam->beta2_power));
        for (branch = 0u; branch < 2u; branch++) {
            const VAECIICHead *head = heads[branch];
            const VAECParameter *params[5] = {
                &head->linear1.weights, &head->batch_norm.gamma,
                &head->batch_norm.beta, &head->linear2.weights,
                &head->linear2.bias
            };
            for (j = 0u; j < 5u; j++) {
                const VAECParameter *p = params[j];
                hash = digest(hash, p->values, p->shape.count * sizeof(float));
                hash = digest(hash, p->gradients, p->shape.count * sizeof(float));
                hash = digest(hash, p->adam_m, p->shape.count * sizeof(float));
                hash = digest(hash, p->adam_v, p->shape.count * sizeof(float));
            }
            hash = digest(hash, head->batch_norm.running_mean,
                          VAE_C_IIC_HEAD_HIDDEN_DIM * sizeof(float));
            hash = digest(hash, head->batch_norm.running_variance,
                          VAE_C_IIC_HEAD_HIDDEN_DIM * sizeof(float));
            hash = digest(hash, &head->num_batches_tracked,
                          sizeof(head->num_batches_tracked));
        }
    }
    return hash;
}

static int forward_rows(VAECIICEnsemble *ensemble, const float *mu,
                        size_t rows, size_t chunk, float *h)
{
    VAECIICHWorkspace workspace;
    size_t start;
    int ok = 0;
    vae_c_iic_h_workspace_init(&workspace);
    if (vae_c_iic_h_workspace_allocate(ensemble, chunk, &workspace) !=
        VAE_C_OP_OK) goto done;
    for (start = 0u; start < rows; start += chunk) {
        VAECTensor input;
        size_t size = rows - start, dims[2];
        if (size > chunk) size = chunk;
        dims[0] = size;
        dims[1] = REAL_DIM;
        vae_c_tensor_init(&input);
        if (vae_c_shape_make(2u, dims, &input.shape) != VAE_C_STORAGE_OK)
            goto done;
        input.data = (float *)(mu + start * REAL_DIM);
        if (vae_c_iic_h_forward(ensemble, &input, &workspace,
                h + start * REAL_COLS, size * REAL_COLS) != VAE_C_OP_OK)
            goto done;
    }
    ok = 1;
done:
    vae_c_iic_h_workspace_free(&workspace);
    return ok;
}

int main(int argc, char **argv)
{
    VAECIICEnsemble ensemble;
    VAECIICCheckpointIdentity identity;
    VAECIICCheckpointMetadata metadata;
    VAECIICMini112Central central;
    float mu[REAL_ROWS * REAL_DIM], original[REAL_ROWS * REAL_DIM];
    float whole[REAL_ROWS * REAL_COLS], chunk64[REAL_ROWS * REAL_COLS];
    float chunk32[REAL_ROWS * REAL_COLS];
    char path[4096], sha[65], after_sha[65], error[512];
    uint64_t before;
    size_t k, row, c;
    int n, passed = 0;
    if (argc != 2) return 2;
    vae_c_iic_ensemble_init(&ensemble);
    vae_c_iic_checkpoint_identity_init(&identity);
    vae_c_iic_checkpoint_metadata_init(&metadata);
    vae_c_iic_mini112_central_init(&central);
    n = snprintf(path, sizeof(path),
        "%s/sakai_multires_c/results/vae_c/iic/%s/final.ckpt",
        argv[1], REAL_RUN_ID);
    if (n < 0 || (size_t)n >= sizeof(path) ||
        vae_c_mu_file_sha256(path, sha) != VAE_C_MU_OK ||
        strcmp(sha, REAL_FINAL_SHA) != 0) goto done;
    identity.input_dim = REAL_DIM;
    identity.capacity = 128u;
    identity.batch_size = 128u;
    identity.train_rows = VAE_C_IIC_MINI112_ROWS;
    identity.max_epochs = 100u;
    identity.shuffle_seed = VAE_C_IIC_ENSEMBLE_SHUFFLE_SEED;
    for (k = 0u; k < VAE_C_IIC_ENSEMBLE_K; k++)
        identity.seeds[k] = VAE_C_IIC_ENSEMBLE_FIRST_SEED + (uint64_t)k;
    (void)snprintf(identity.run_id, sizeof(identity.run_id), "%s", REAL_RUN_ID);
    (void)snprintf(identity.base_commit, sizeof(identity.base_commit), "%s",
                   "887046ce505dcca7f09bc8e51ea2b59332d8b4f6");
    (void)snprintf(identity.producer_state, sizeof(identity.producer_state),
                   "%s", "BINARY_SHA256_AUTHORITY");
    (void)snprintf(identity.producer_binary_sha256,
                   sizeof(identity.producer_binary_sha256), "%s",
                   "c41b49f6671678a94c6ee6b62538785249ec3bbc5a000822917f4b975d7f3333");
    if (vae_c_iic_ensemble_allocate_v1(&ensemble, REAL_DIM, 128u) !=
            VAE_C_OP_OK ||
        vae_c_iic_checkpoint_load(path, &identity, &ensemble, &metadata) !=
            VAE_C_IIC_CHECKPOINT_OK ||
        metadata.kind != VAE_C_IIC_CHECKPOINT_FINAL ||
        metadata.last_completed_epoch != 99u) goto done;
    for (k = 0u; k < VAE_C_IIC_ENSEMBLE_K; k++) {
        if (ensemble.optimizers[k].step != 6800u ||
            ensemble.pairs[k].main_head.num_batches_tracked != 13600u ||
            ensemble.pairs[k].overcluster_head.num_batches_tracked != 13600u)
            goto done;
    }
    if (!vae_c_iic_mini112_central_load(argv[1], NULL, &central,
            error, sizeof(error)) || central.row_count != 8583u ||
        central.latent_dim != REAL_DIM) goto done;
    for (row = 0u; row < REAL_ROWS; row++) {
        size_t index = row == REAL_ROWS - 1u ? 8582u : row * 67u;
        const float *source = vae_c_iic_mini112_central_row_mu(&central, index);
        if (source == NULL || central.rows[index].manifest_index != index)
            goto done;
        memcpy(mu + row * REAL_DIM, source, REAL_DIM * sizeof(float));
    }
    memcpy(original, mu, sizeof(mu));
    before = persistent_digest(&ensemble);
    if (!forward_rows(&ensemble, mu, REAL_ROWS, REAL_ROWS, whole) ||
        !forward_rows(&ensemble, mu, REAL_ROWS, 64u, chunk64) ||
        !forward_rows(&ensemble, mu, REAL_ROWS, 32u, chunk32) ||
        memcmp(whole, chunk64, sizeof(whole)) != 0 ||
        memcmp(whole, chunk32, sizeof(whole)) != 0 ||
        memcmp(mu, original, sizeof(mu)) != 0 ||
        persistent_digest(&ensemble) != before ||
        !forward_rows(&ensemble, mu, 7u, 7u, chunk64) ||
        memcmp(whole, chunk64, 7u * REAL_COLS * sizeof(float)) != 0)
        goto done;
    for (row = 0u; row < REAL_ROWS; row++) {
        double full_sum = 0.0;
        for (k = 0u; k < VAE_C_IIC_ENSEMBLE_K; k++) {
            double block = 0.0;
            for (c = 0u; c < VAE_C_IIC_ENSEMBLE_MAIN_DIM; c++) {
                double value = whole[row * REAL_COLS +
                    k * VAE_C_IIC_ENSEMBLE_MAIN_DIM + c];
                if (!isfinite(value) || value < 0.0 || value > 1.0) goto done;
                block += value;
            }
            if (fabs(block - 1.0) > 1.0e-4) goto done;
            full_sum += block;
        }
        if (fabs(full_sum - 5.0) > 5.0e-4) goto done;
    }
    if (persistent_digest(&ensemble) != before ||
        vae_c_mu_file_sha256(path, after_sha) != VAE_C_MU_OK ||
        strcmp(sha, after_sha) != 0) goto done;
    printf("MINI112_H_REAL_TINY_DIGEST=%016llx\n",
           (unsigned long long)digest(UINT64_C(14695981039346656037),
                                      whole, sizeof(whole)));
    puts("MINI112_H_REAL_FINAL_TINY_PASS=true");
    puts("MINI112_H_BATCH_INDEPENDENT=BITWISE");
    passed = 1;
done:
    vae_c_iic_mini112_central_free(&central);
    vae_c_iic_ensemble_free(&ensemble);
    return passed ? 0 : 1;
}
