#include "vae_c_iic_h.h"
#include "vae_c_iic_checkpoint.h"
#include "vae_c_iic_mini75_central.h"
#include "vae_c_mu_contract.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define TEST_DIM 3u
#define TEST_ROWS 4u
#define TEST_PATH_CAP 1024u
#define REAL_RUN_ID \
    "mini75_iic_v1_737da555866b822e14c545c4f2687869f41a52cd1340b9e941f3a1ce77763eb9"
#define REAL_FINAL_SHA \
    "6df9bb278210861b449adec454c4f649fee8be1398082418096c2679aee907d5"

static uint64_t hash_bytes(uint64_t h, const void *data, size_t bytes)
{
    const unsigned char *p = (const unsigned char *)data;
    size_t i;
    for (i = 0u; i < bytes; i++) {
        h ^= p[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

static uint64_t hash_parameter(uint64_t h, const VAECParameter *p)
{
    const float *const arrays[4] = {
        p->values, p->gradients, p->adam_m, p->adam_v
    };
    size_t a;
    for (a = 0u; a < 4u; a++)
        h = hash_bytes(h, arrays[a], p->shape.count * sizeof(float));
    return h;
}

/* Persistent state plus parameter/optimizer buffers; transient EVAL caches and
 * forward-generation tickets are intentionally excluded. */
static uint64_t persistent_digest(const VAECIICEnsemble *ensemble)
{
    uint64_t h = UINT64_C(14695981039346656037);
    size_t k, branch;
    for (k = 0u; k < VAE_C_IIC_ENSEMBLE_K; k++) {
        const VAECIICPair *pair = &ensemble->pairs[k];
        const VAECIICHead *heads[2] = {
            &pair->main_head, &pair->overcluster_head
        };
        const VAECAdam *adam = &ensemble->optimizers[k];
        h = hash_bytes(h, &adam->step, sizeof(adam->step));
        h = hash_bytes(h, &adam->beta1_power, sizeof(adam->beta1_power));
        h = hash_bytes(h, &adam->beta2_power, sizeof(adam->beta2_power));
        for (branch = 0u; branch < 2u; branch++) {
            const VAECIICHead *head = heads[branch];
            h = hash_parameter(h, &head->linear1.weights);
            h = hash_parameter(h, &head->batch_norm.gamma);
            h = hash_parameter(h, &head->batch_norm.beta);
            h = hash_parameter(h, &head->linear2.weights);
            h = hash_parameter(h, &head->linear2.bias);
            h = hash_bytes(h, head->batch_norm.running_mean,
                VAE_C_IIC_HEAD_HIDDEN_DIM * sizeof(float));
            h = hash_bytes(h, head->batch_norm.running_variance,
                VAE_C_IIC_HEAD_HIDDEN_DIM * sizeof(float));
            h = hash_bytes(h, &head->num_batches_tracked,
                sizeof(head->num_batches_tracked));
        }
    }
    return h;
}

static int borrowed(const float *data, size_t rows, size_t dim,
                    VAECTensor *tensor)
{
    size_t dims[2] = {rows, dim};
    vae_c_tensor_init(tensor);
    if (vae_c_shape_make(2u, dims, &tensor->shape) != VAE_C_STORAGE_OK)
        return 0;
    tensor->data = (float *)data;
    return 1;
}

static int probabilities_valid(const float *h, size_t rows)
{
    size_t row, k, c;
    for (row = 0u; row < rows; row++) {
        float full_sum = 0.0f;
        for (k = 0u; k < VAE_C_IIC_ENSEMBLE_K; k++) {
            float sum = 0.0f;
            for (c = 0u; c < VAE_C_IIC_ENSEMBLE_MAIN_DIM; c++) {
                float p = h[row * VAE_C_IIC_H_COLUMNS +
                    k * VAE_C_IIC_ENSEMBLE_MAIN_DIM + c];
                if (!isfinite(p) || p < 0.0f || p > 1.0f) return 0;
                sum += p;
            }
            if (fabsf(sum - 1.0f) > 1.0e-4f) return 0;
            full_sum += sum;
        }
        if (fabsf(full_sum - 5.0f) > 5.0e-4f) return 0;
    }
    return 1;
}

static void set_explicit_main_fixture(VAECIICEnsemble *ensemble)
{
    size_t k, hidden, d, c;
    for (k = 0u; k < VAE_C_IIC_ENSEMBLE_K; k++) {
        VAECIICHead *head = &ensemble->pairs[k].main_head;
        for (hidden = 0u; hidden < VAE_C_IIC_HEAD_HIDDEN_DIM; hidden++) {
            for (d = 0u; d < TEST_DIM; d++) {
                head->linear1.weights.values[hidden * TEST_DIM + d] =
                    0.006f * (float)((int)((hidden * 3u + d * 7u + k) % 17u) - 8);
            }
            head->batch_norm.gamma.values[hidden] =
                1.0f + 0.001f * (float)((int)(hidden % 7u) - 3);
            head->batch_norm.beta.values[hidden] =
                0.01f * (float)((int)(hidden % 5u) - 2);
            head->batch_norm.running_mean[hidden] =
                0.002f * (float)((int)(hidden % 13u) - 6);
            head->batch_norm.running_variance[hidden] =
                1.0f + 0.001f * (float)(hidden % 17u);
        }
        for (c = 0u; c < VAE_C_IIC_ENSEMBLE_MAIN_DIM; c++) {
            for (hidden = 0u; hidden < VAE_C_IIC_HEAD_HIDDEN_DIM;
                 hidden++) {
                head->linear2.weights.values[
                    c * VAE_C_IIC_HEAD_HIDDEN_DIM + hidden] =
                    0.0002f *
                    (float)((int)((c * 7u + hidden * 3u + k) % 19u) - 9);
            }
            head->linear2.bias.values[c] =
                0.003f * ((float)c - 17.0f) + 0.005f * (float)k;
        }
    }
}

static int chunk_compare(VAECIICEnsemble *ensemble, const float *mu,
                         size_t dim, size_t rows, const float *whole,
                         VAECIICHWorkspace *workspace)
{
    const size_t chunks[3] = {1u, 2u, 1u};
    float chunked[TEST_ROWS * VAE_C_IIC_H_COLUMNS];
    size_t start = 0u, j;
    if (rows != TEST_ROWS) return 0;
    for (j = 0u; j < 3u; j++) {
        VAECTensor input;
        if (!borrowed(mu + start * dim, chunks[j], dim, &input) ||
            vae_c_iic_h_forward(ensemble, &input, workspace,
                chunked + start * VAE_C_IIC_H_COLUMNS,
                chunks[j] * VAE_C_IIC_H_COLUMNS) != VAE_C_OP_OK)
            return 0;
        start += chunks[j];
    }
    return memcmp(whole, chunked, sizeof(chunked)) == 0;
}

static int synthetic_test(int dump)
{
    VAECIICEnsemble ensemble;
    VAECIICHWorkspace large, small;
    VAECTensor input;
    float mu[TEST_ROWS * TEST_DIM];
    float original[TEST_ROWS * TEST_DIM];
    float h[TEST_ROWS * VAE_C_IIC_H_COLUMNS];
    float repeat[TEST_ROWS * VAE_C_IIC_H_COLUMNS];
    float changed[TEST_ROWS * VAE_C_IIC_H_COLUMNS];
    uint64_t before, after;
    size_t row, d, k, c;
    int passed = 0;
    vae_c_iic_ensemble_init(&ensemble);
    vae_c_iic_h_workspace_init(&large);
    vae_c_iic_h_workspace_init(&small);
    if (vae_c_iic_ensemble_allocate_v1(&ensemble, TEST_DIM, TEST_ROWS) !=
            VAE_C_OP_OK) goto done;
    set_explicit_main_fixture(&ensemble);
    if (vae_c_iic_h_workspace_allocate(&ensemble, TEST_ROWS, &large) !=
            VAE_C_OP_OK ||
        vae_c_iic_h_workspace_allocate(&ensemble, 2u, &small) !=
            VAE_C_OP_OK) goto done;
    for (row = 0u; row < TEST_ROWS; row++) {
        for (d = 0u; d < TEST_DIM; d++)
            mu[row * TEST_DIM + d] =
                0.1f * (float)((row + 1u) * (d + 1u)) - 0.25f;
    }
    memcpy(original, mu, sizeof(mu));
    if (!borrowed(mu, TEST_ROWS, TEST_DIM, &input)) goto done;
    before = persistent_digest(&ensemble);
    if (vae_c_iic_h_forward(&ensemble, &input, &large, h,
            TEST_ROWS * VAE_C_IIC_H_COLUMNS) != VAE_C_OP_OK ||
        !probabilities_valid(h, TEST_ROWS)) goto done;
    after = persistent_digest(&ensemble);
    if (before != after || memcmp(mu, original, sizeof(mu)) != 0) goto done;
    for (row = 0u; row < TEST_ROWS; row++) {
        for (k = 0u; k < VAE_C_IIC_ENSEMBLE_K; k++) {
            for (c = 0u; c < VAE_C_IIC_ENSEMBLE_MAIN_DIM; c++) {
                if (h[row * VAE_C_IIC_H_COLUMNS +
                      k * VAE_C_IIC_ENSEMBLE_MAIN_DIM + c] !=
                    large.main[k].softmax_output.data[
                      row * VAE_C_IIC_ENSEMBLE_MAIN_DIM + c]) goto done;
            }
        }
    }
    if (vae_c_iic_h_forward(&ensemble, &input, &large, repeat,
            TEST_ROWS * VAE_C_IIC_H_COLUMNS) != VAE_C_OP_OK ||
        memcmp(h, repeat, sizeof(h)) != 0 ||
        !chunk_compare(&ensemble, mu, TEST_DIM, TEST_ROWS, h, &small) ||
        persistent_digest(&ensemble) != before) goto done;
    ensemble.pairs[0].overcluster_head.linear2.bias.values[0] += 1.0f;
    if (vae_c_iic_h_forward(&ensemble, &input, &large, repeat,
            TEST_ROWS * VAE_C_IIC_H_COLUMNS) != VAE_C_OP_OK ||
        memcmp(h, repeat, sizeof(h)) != 0) goto done;
    ensemble.pairs[0].main_head.linear2.bias.values[0] += 1.0f;
    if (vae_c_iic_h_forward(&ensemble, &input, &large, changed,
            TEST_ROWS * VAE_C_IIC_H_COLUMNS) != VAE_C_OP_OK ||
        memcmp(h, changed, sizeof(h)) == 0) goto done;
    ensemble.pairs[0].main_head.linear2.bias.values[0] -= 1.0f;
    memset(repeat, 0xA5, sizeof(repeat));
    if (vae_c_iic_h_forward(&ensemble, &input, &large, repeat, 1u) ==
            VAE_C_OP_OK ||
        vae_c_iic_h_forward(&ensemble, &input, &large, mu,
            TEST_ROWS * VAE_C_IIC_H_COLUMNS) == VAE_C_OP_OK ||
        vae_c_iic_h_forward(&ensemble, NULL, &large, repeat,
            TEST_ROWS * VAE_C_IIC_H_COLUMNS) == VAE_C_OP_OK) goto done;
    for (row = 0u; row < sizeof(repeat); row++) {
        if (((unsigned char *)repeat)[row] != 0xA5u) goto done;
    }
    mu[0] = NAN;
    if (vae_c_iic_h_forward(&ensemble, &input, &large, repeat,
            TEST_ROWS * VAE_C_IIC_H_COLUMNS) == VAE_C_OP_OK) goto done;
    mu[0] = original[0];
    if (dump) {
        printf("GOLDEN_VALUES=[");
        for (row = 0u; row < TEST_ROWS * VAE_C_IIC_H_COLUMNS; row++)
            printf("%s%.9g", row == 0u ? "" : ",", (double)h[row]);
        puts("]");
    }
    printf("SYNTHETIC_H_DIGEST=%016llx\n",
           (unsigned long long)hash_bytes(UINT64_C(14695981039346656037),
                                           h, sizeof(h)));
    puts("H_SYNTHETIC_PASS=true");
    passed = 1;
done:
    vae_c_iic_h_workspace_free(&small);
    vae_c_iic_h_workspace_free(&large);
    vae_c_iic_ensemble_free(&ensemble);
    return passed;
}

static int real_tiny_test(const char *repository_root)
{
    VAECIICEnsemble ensemble;
    VAECIICHWorkspace whole_workspace, chunk_workspace;
    VAECIICMini75Central central;
    VAECIICCheckpointIdentity identity;
    VAECIICCheckpointMetadata metadata;
    VAECTensor input;
    char path[TEST_PATH_CAP], sha[65], error[512];
    const size_t indices[TEST_ROWS] = {0u, 37u, 4000u, 8582u};
    float mu[TEST_ROWS * VAE_C_IIC_MINI75_LATENT_DIM];
    float h[TEST_ROWS * VAE_C_IIC_H_COLUMNS];
    float repeat[TEST_ROWS * VAE_C_IIC_H_COLUMNS];
    uint64_t before;
    size_t k, row;
    int passed = 0, length;
    vae_c_iic_ensemble_init(&ensemble);
    vae_c_iic_h_workspace_init(&whole_workspace);
    vae_c_iic_h_workspace_init(&chunk_workspace);
    vae_c_iic_mini75_central_init(&central);
    vae_c_iic_checkpoint_identity_init(&identity);
    vae_c_iic_checkpoint_metadata_init(&metadata);
    length = snprintf(path, sizeof(path),
        "%s/sakai_multires_c/results/vae_c/iic/%s/final.ckpt",
        repository_root, REAL_RUN_ID);
    if (length < 0 || (size_t)length >= sizeof(path) ||
        vae_c_mu_file_sha256(path, sha) != VAE_C_MU_OK ||
        strcmp(sha, REAL_FINAL_SHA) != 0) goto done;
    identity.input_dim = VAE_C_IIC_MINI75_LATENT_DIM;
    identity.capacity = 128u;
    identity.batch_size = 128u;
    identity.train_rows = VAE_C_IIC_MINI75_ROWS;
    identity.max_epochs = 100u;
    identity.shuffle_seed = VAE_C_IIC_ENSEMBLE_SHUFFLE_SEED;
    for (k = 0u; k < VAE_C_IIC_ENSEMBLE_K; k++)
        identity.seeds[k] = VAE_C_IIC_ENSEMBLE_FIRST_SEED + (uint64_t)k;
    (void)snprintf(identity.run_id, sizeof(identity.run_id), "%s", REAL_RUN_ID);
    (void)snprintf(identity.base_commit, sizeof(identity.base_commit), "%s",
                   "ba3f2e5b244054b6c11a053c749da3c29f69c298");
    (void)snprintf(identity.producer_state, sizeof(identity.producer_state),
                   "%s", "BINARY_SHA256_AUTHORITY");
    (void)snprintf(identity.producer_binary_sha256,
                   sizeof(identity.producer_binary_sha256), "%s",
                   "0371657dbb4b7dff2157b93228499a92d1297983fc1c21891d14f4b775b4fcd5");
    if (vae_c_iic_ensemble_allocate_v1(&ensemble,
            VAE_C_IIC_MINI75_LATENT_DIM, 128u) != VAE_C_OP_OK ||
        vae_c_iic_checkpoint_load(path, &identity, &ensemble,
            &metadata) != VAE_C_IIC_CHECKPOINT_OK ||
        metadata.kind != VAE_C_IIC_CHECKPOINT_FINAL ||
        metadata.last_completed_epoch != 99u) goto done;
    for (k = 0u; k < VAE_C_IIC_ENSEMBLE_K; k++) {
        if (ensemble.optimizers[k].step != 6800u ||
            ensemble.pairs[k].main_head.num_batches_tracked != 13600u ||
            ensemble.pairs[k].overcluster_head.num_batches_tracked != 13600u)
            goto done;
    }
    if (!vae_c_iic_mini75_central_load(
            repository_root, NULL, &central, error, sizeof(error)) ||
        central.row_count != 8583u || central.latent_dim != 32u)
        goto done;
    for (row = 0u; row < TEST_ROWS; row++) {
        const float *source = vae_c_iic_mini75_central_row_mu(
            &central, indices[row]);
        if (source == NULL || central.rows[indices[row]].manifest_index !=
                indices[row]) goto done;
        memcpy(mu + row * 32u, source, 32u * sizeof(float));
    }
    if (!borrowed(mu, TEST_ROWS, 32u, &input) ||
        vae_c_iic_h_workspace_allocate(&ensemble, TEST_ROWS,
            &whole_workspace) != VAE_C_OP_OK ||
        vae_c_iic_h_workspace_allocate(&ensemble, 2u,
            &chunk_workspace) != VAE_C_OP_OK) goto done;
    before = persistent_digest(&ensemble);
    if (vae_c_iic_h_forward(&ensemble, &input, &whole_workspace, h,
            TEST_ROWS * VAE_C_IIC_H_COLUMNS) != VAE_C_OP_OK ||
        !probabilities_valid(h, TEST_ROWS) ||
        !chunk_compare(&ensemble, mu, 32u, TEST_ROWS, h,
            &chunk_workspace) ||
        vae_c_iic_h_forward(&ensemble, &input, &whole_workspace, repeat,
            TEST_ROWS * VAE_C_IIC_H_COLUMNS) != VAE_C_OP_OK ||
        memcmp(h, repeat, sizeof(h)) != 0 ||
        persistent_digest(&ensemble) != before) goto done;
    printf("REAL_FINAL_H_DIGEST=%016llx\n",
           (unsigned long long)hash_bytes(UINT64_C(14695981039346656037),
                                           h, sizeof(h)));
    puts("REAL_FINAL_TINY_PASS=true");
    passed = 1;
done:
    vae_c_iic_h_workspace_free(&chunk_workspace);
    vae_c_iic_h_workspace_free(&whole_workspace);
    vae_c_iic_mini75_central_free(&central);
    vae_c_iic_ensemble_free(&ensemble);
    return passed;
}

int main(int argc, char **argv)
{
    if (argc == 1) return synthetic_test(0) ? 0 : 1;
    if (argc == 2 && strcmp(argv[1], "--golden-dump") == 0)
        return synthetic_test(1) ? 0 : 1;
    if (argc == 3 && strcmp(argv[1], "--real-tiny") == 0)
        return real_tiny_test(argv[2]) ? 0 : 1;
    fputs("usage: test_vae_c_iic_h [--golden-dump|--real-tiny ROOT]\n",
          stderr);
    return 2;
}
