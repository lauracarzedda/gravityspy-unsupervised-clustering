#include "vae_c_iic_checkpoint.h"
#include "vae_c_sha256.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <process.h>
#include <direct.h>
#define TEST_PID _getpid
#define TEST_MKDIR(path) _mkdir(path)
#define TEST_RMDIR(path) _rmdir(path)
#else
#include <unistd.h>
#define TEST_PID getpid
#define TEST_MKDIR(path) mkdir(path, 0700)
#define TEST_RMDIR(path) rmdir(path)
#endif

#define CHECK(condition) do {                                                 \
    if (!(condition)) {                                                       \
        fprintf(stderr, "IIC checkpoint test failed at line %d: %s\n",      \
                __LINE__, #condition);                                       \
        ok = false;                                                           \
        goto done;                                                            \
    }                                                                         \
} while (0)

static VAECParameter *parameter_at(VAECIICHead *head, size_t index)
{
    switch (index) {
        case 0: return &head->linear1.weights;
        case 1: return &head->batch_norm.gamma;
        case 2: return &head->batch_norm.beta;
        case 3: return &head->linear2.weights;
        default: return &head->linear2.bias;
    }
}

static VAECIICHead *head_at(VAECIICPair *pair, size_t index)
{
    return index == 0 ? &pair->main_head : &pair->overcluster_head;
}

static void populate_state(VAECIICEnsemble *ensemble)
{
    size_t pair_index;
    for (pair_index = 0; pair_index < VAE_C_IIC_ENSEMBLE_K;
         pair_index++) {
        VAECAdam *adam = &ensemble->optimizers[pair_index];
        size_t head_index;
        adam->step = 3;
        adam->beta1_power = 1.0;
        adam->beta2_power = 1.0;
        for (head_index = 0; head_index < 2u; head_index++) {
            VAECIICHead *head = head_at(
                &ensemble->pairs[pair_index], head_index
            );
            size_t index;
            head->num_batches_tracked = 6;
            for (index = 0; index < VAE_C_IIC_HEAD_HIDDEN_DIM; index++) {
                head->batch_norm.running_mean[index] =
                    (float)(pair_index + head_index + index) * 0.0001f;
                head->batch_norm.running_variance[index] =
                    1.0f + (float)index * 0.00001f;
            }
            for (index = 0; index < 5u; index++) {
                VAECParameter *parameter = parameter_at(head, index);
                size_t element;
                for (element = 0; element < parameter->shape.count;
                     element++) {
                    parameter->values[element] =
                        0.001f * (float)(index + 1u) +
                        0.000001f * (float)(element + pair_index);
                    parameter->adam_m[element] =
                        0.0001f * (float)(head_index + 1u);
                    parameter->adam_v[element] =
                        0.000001f * (float)(index + 1u);
                }
            }
        }
        /* One optimizer state per pair, not one per head. */
        adam->beta1_power = adam->beta1 * adam->beta1 * adam->beta1;
        adam->beta2_power = adam->beta2 * adam->beta2 * adam->beta2;
    }
}

static void set_step(VAECIICEnsemble *ensemble, uint64_t step)
{
    size_t pair_index;
    for (pair_index = 0; pair_index < VAE_C_IIC_ENSEMBLE_K;
         pair_index++) {
        VAECAdam *adam = &ensemble->optimizers[pair_index];
        uint64_t index;
        adam->step = step;
        adam->beta1_power = 1.0;
        adam->beta2_power = 1.0;
        for (index = 0; index < step; index++) {
            adam->beta1_power *= adam->beta1;
            adam->beta2_power *= adam->beta2;
        }
        ensemble->pairs[pair_index].main_head.num_batches_tracked = 2u * step;
        ensemble->pairs[pair_index].overcluster_head.num_batches_tracked =
            2u * step;
    }
}

static bool same_state(
    VAECIICEnsemble *left,
    VAECIICEnsemble *right
)
{
    size_t pair_index;
    for (pair_index = 0; pair_index < VAE_C_IIC_ENSEMBLE_K;
         pair_index++) {
        size_t head_index;
        if (left->optimizers[pair_index].step !=
                right->optimizers[pair_index].step ||
            left->optimizers[pair_index].beta1_power !=
                right->optimizers[pair_index].beta1_power ||
            left->optimizers[pair_index].beta2_power !=
                right->optimizers[pair_index].beta2_power) {
            return false;
        }
        for (head_index = 0; head_index < 2u; head_index++) {
            VAECIICHead *a = head_at(&left->pairs[pair_index], head_index);
            VAECIICHead *b = head_at(&right->pairs[pair_index], head_index);
            size_t parameter_index;
            if (a->num_batches_tracked != b->num_batches_tracked ||
                memcmp(a->batch_norm.running_mean,
                       b->batch_norm.running_mean,
                       VAE_C_IIC_HEAD_HIDDEN_DIM * sizeof(float)) != 0 ||
                memcmp(a->batch_norm.running_variance,
                       b->batch_norm.running_variance,
                       VAE_C_IIC_HEAD_HIDDEN_DIM * sizeof(float)) != 0) {
                return false;
            }
            for (parameter_index = 0; parameter_index < 5u;
                 parameter_index++) {
                VAECParameter *p = parameter_at(a, parameter_index);
                VAECParameter *q = parameter_at(b, parameter_index);
                size_t bytes = p->shape.count * sizeof(float);
                if (memcmp(p->values, q->values, bytes) != 0 ||
                    memcmp(p->adam_m, q->adam_m, bytes) != 0 ||
                    memcmp(p->adam_v, q->adam_v, bytes) != 0) {
                    return false;
                }
            }
        }
    }
    return true;
}

static bool read_bytes(const char *path, unsigned char **bytes, size_t *size)
{
    FILE *file = fopen(path, "rb");
    long length;
    if (file == NULL || fseek(file, 0, SEEK_END) != 0 ||
        (length = ftell(file)) < 0 || fseek(file, 0, SEEK_SET) != 0) {
        if (file != NULL) (void)fclose(file);
        return false;
    }
    *bytes = (unsigned char *)malloc((size_t)length);
    if (*bytes == NULL) {
        (void)fclose(file);
        return false;
    }
    *size = (size_t)length;
    if (fread(*bytes, 1u, *size, file) != *size) {
        (void)fclose(file);
        free(*bytes);
        *bytes = NULL;
        return false;
    }
    if (fclose(file) != 0) {
        free(*bytes);
        *bytes = NULL;
        return false;
    }
    return true;
}

static bool file_absent(const char *path)
{
    FILE *file = fopen(path, "rb");
    if (file == NULL) return true;
    (void)fclose(file);
    return false;
}

static bool gradients_zero(VAECIICEnsemble *ensemble)
{
    size_t pair_index;
    for (pair_index = 0; pair_index < VAE_C_IIC_ENSEMBLE_K;
         pair_index++) {
        size_t head_index;
        VAECIICPairWorkspace *workspace =
            &ensemble->workspaces[pair_index];
        VAECIICHeadWorkspace *views[4] = {
            &workspace->main_a, &workspace->over_a,
            &workspace->main_b, &workspace->over_b
        };
        size_t view_index;
        if (workspace->backward_pair != NULL) return false;
        for (head_index = 0; head_index < 2u; head_index++) {
            VAECIICHead *head = head_at(
                &ensemble->pairs[pair_index], head_index
            );
            size_t parameter_index;
            for (parameter_index = 0; parameter_index < 5u;
                 parameter_index++) {
                VAECParameter *parameter =
                    parameter_at(head, parameter_index);
                size_t element;
                for (element = 0; element < parameter->shape.count; element++) {
                    if (parameter->gradients[element] != 0.0f) return false;
                }
            }
        }
        for (view_index = 0; view_index < 4u; view_index++) {
            VAECIICHeadWorkspace *view = views[view_index];
            VAECTensor *tensors[5] = {
                &view->linear1_storage, &view->batch_norm_storage,
                &view->relu_storage, &view->logits_storage,
                &view->softmax_storage
            };
            size_t tensor_index;
            if (view->backward_head != NULL ||
                view->backward_generation != 0) return false;
            for (tensor_index = 0; tensor_index < 5u; tensor_index++) {
                size_t element;
                for (element = 0; element < tensors[tensor_index]->shape.count;
                     element++) {
                    if (tensors[tensor_index]->grad[element] != 0.0f) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

static bool write_bytes(
    const char *path, const unsigned char *bytes, size_t size
)
{
    FILE *file = fopen(path, "wb");
    bool result;
    if (file == NULL) return false;
    result = fwrite(bytes, 1u, size, file) == size && fflush(file) == 0;
    if (fclose(file) != 0) result = false;
    return result;
}

int main(int argc, char **argv)
{
    VAECIICEnsemble *source = NULL;
    VAECIICEnsemble *target = NULL;
    VAECIICCheckpointIdentity identity;
    VAECIICCheckpointIdentity wrong_identity;
    VAECIICCheckpointMetadata metadata;
    VAECIICCheckpointMetadata loaded;
    unsigned char *bytes = NULL;
    size_t size = 0;
    char latest[1024] = {0};
    char final[1024] = {0};
    char bad[1024] = {0};
    char partial[sizeof(latest) + sizeof(".partial")] = {0};
    char fixture_dir[900] = {0};
    bool fixture_created = false;
    bool ok = true;
    size_t index;
    int path_written;
    uint8_t digest[VAE_C_SHA256_DIGEST_SIZE];

    if (argc != 2 || strlen(argv[1]) > 800u) {
        fprintf(stderr, "Expected external build-directory fixture root\n");
        return 2;
    }
    path_written = snprintf(fixture_dir, sizeof(fixture_dir),
                            "%s/iic_ckpt_fixture_%ld", argv[1],
                            (long)TEST_PID());
    CHECK(path_written >= 0 && (size_t)path_written < sizeof(fixture_dir));
    CHECK(TEST_MKDIR(fixture_dir) == 0);
    fixture_created = true;
    path_written = snprintf(latest, sizeof(latest), "%s/latest.bin",
                            fixture_dir);
    CHECK(path_written >= 0 && (size_t)path_written < sizeof(latest));
    path_written = snprintf(final, sizeof(final), "%s/final.bin",
                            fixture_dir);
    CHECK(path_written >= 0 && (size_t)path_written < sizeof(final));
    path_written = snprintf(bad, sizeof(bad), "%s/bad.bin", fixture_dir);
    CHECK(path_written >= 0 && (size_t)path_written < sizeof(bad));
    path_written = snprintf(partial, sizeof(partial), "%s.partial", latest);
    CHECK(path_written >= 0 && (size_t)path_written < sizeof(partial));
    source = (VAECIICEnsemble *)malloc(sizeof(*source));
    target = (VAECIICEnsemble *)malloc(sizeof(*target));
    if (source != NULL) vae_c_iic_ensemble_init(source);
    if (target != NULL) vae_c_iic_ensemble_init(target);
    CHECK(source != NULL && target != NULL);
    CHECK(vae_c_iic_ensemble_allocate_v1(source, 3u, 8u) == VAE_C_OP_OK);
    CHECK(vae_c_iic_ensemble_allocate_v1(target, 3u, 8u) == VAE_C_OP_OK);
    populate_state(source);
    vae_c_iic_checkpoint_identity_init(&identity);
    identity.input_dim = 3u;
    identity.capacity = 8u;
    identity.batch_size = 8u;
    identity.train_rows = 23u;
    identity.max_epochs = 3u;
    identity.shuffle_seed = VAE_C_IIC_ENSEMBLE_SHUFFLE_SEED;
    for (index = 0; index < VAE_C_IIC_ENSEMBLE_K; index++) {
        identity.seeds[index] = VAE_C_IIC_ENSEMBLE_FIRST_SEED + index;
    }
    (void)strcpy(identity.run_id, "synthetic-checkpoint-fixture");
    (void)strcpy(identity.base_commit,
                 "2bbc7c9b670f7778ce5b25779a7933dcf9ddf891");
    (void)strcpy(identity.producer_state, "UNCOMMITTED_MINI_C5B6_TEST");
    memset(identity.producer_binary_sha256, 'a', 64u);
    identity.producer_binary_sha256[64] = 0;
    vae_c_iic_checkpoint_metadata_init(&metadata);
    metadata.kind = VAE_C_IIC_CHECKPOINT_LATEST;
    metadata.last_completed_epoch = 0;

    CHECK(vae_c_iic_checkpoint_save_atomic(
        latest, &identity, &metadata, source) == VAE_C_IIC_CHECKPOINT_OK);
    CHECK(vae_c_iic_checkpoint_save_atomic(
        latest, &identity, &metadata, source) == VAE_C_IIC_CHECKPOINT_OK);
    target->pairs[0].main_head.linear1.weights.gradients[0] = 7.0f;
    target->workspaces[0].main_a.linear1_storage.grad[0] = 9.0f;
    CHECK(vae_c_iic_checkpoint_load(
        latest, &identity, target, &loaded) == VAE_C_IIC_CHECKPOINT_OK);
    CHECK(loaded.kind == VAE_C_IIC_CHECKPOINT_LATEST &&
          loaded.last_completed_epoch == 0 && same_state(source, target));
    CHECK(gradients_zero(target));
    CHECK(vae_c_iic_ensemble_validate(target) == VAE_C_OP_OK);
    printf("IIC_CHECKPOINT_ROUNDTRIP_BITWISE=true\n");
    printf("IIC_CHECKPOINT_LATEST_ATOMIC_REPLACE=true\n");

    wrong_identity = identity;
    (void)strcpy(wrong_identity.run_id, "wrong-fixture");
    CHECK(vae_c_iic_checkpoint_load(
        latest, &wrong_identity, target, &loaded) ==
        VAE_C_IIC_CHECKPOINT_IDENTITY_MISMATCH);
    CHECK(same_state(source, target));
    wrong_identity = identity;
    wrong_identity.max_epochs++;
    CHECK(vae_c_iic_checkpoint_load(
        latest, &wrong_identity, target, &loaded) ==
        VAE_C_IIC_CHECKPOINT_IDENTITY_MISMATCH);
    CHECK(same_state(source, target));
    wrong_identity = identity;
    wrong_identity.batch_size = 7u;
    CHECK(vae_c_iic_checkpoint_load(
        latest, &wrong_identity, target, &loaded) ==
        VAE_C_IIC_CHECKPOINT_IDENTITY_MISMATCH);
    CHECK(same_state(source, target));
    wrong_identity = identity;
    wrong_identity.seeds[0] = 999u;
    CHECK(vae_c_iic_checkpoint_load(
        latest, &wrong_identity, target, &loaded) ==
        VAE_C_IIC_CHECKPOINT_INVALID_ARGUMENT);
    CHECK(same_state(source, target));
    wrong_identity = identity;
    wrong_identity.producer_binary_sha256[0] = 'b';
    CHECK(vae_c_iic_checkpoint_load(
        latest, &wrong_identity, target, &loaded) ==
        VAE_C_IIC_CHECKPOINT_IDENTITY_MISMATCH);
    CHECK(same_state(source, target));
    wrong_identity = identity;
    wrong_identity.producer_binary_sha256[0] = 'A';
    CHECK(vae_c_iic_checkpoint_load(
        latest, &wrong_identity, target, &loaded) ==
        VAE_C_IIC_CHECKPOINT_INVALID_ARGUMENT);
    CHECK(same_state(source, target));
    printf("IIC_CHECKPOINT_IDENTITY_MISMATCH_ATOMIC=true\n");
    printf("IIC_CHECKPOINT_CONFIG_SEED_MISMATCH_REJECTED=true\n");
    printf("IIC_CHECKPOINT_PRODUCER_SHA256_MISMATCH_REJECTED=true\n");

    CHECK(read_bytes(latest, &bytes, &size));
    CHECK(size > 64u);
    bytes[size / 2u] ^= 0x01u;
    CHECK(write_bytes(bad, bytes, size));
    CHECK(vae_c_iic_checkpoint_load(
        bad, &identity, target, &loaded) == VAE_C_IIC_CHECKPOINT_CORRUPT);
    CHECK(same_state(source, target));
    bytes[size / 2u] ^= 0x01u;
    CHECK(write_bytes(bad, bytes, size - 1u));
    CHECK(vae_c_iic_checkpoint_load(
        bad, &identity, target, &loaded) == VAE_C_IIC_CHECKPOINT_CORRUPT);
    CHECK(same_state(source, target));
    printf("IIC_CHECKPOINT_CORRUPT_TRUNCATED_REJECTED=true\n");

    /* Recalculate the checksum after a deliberate format-level K mismatch. */
    bytes[12] = 4u;
    CHECK(vae_c_sha256_digest(bytes, size - VAE_C_SHA256_DIGEST_SIZE,
                              digest) == VAE_C_SHA256_OK);
    memcpy(bytes + size - VAE_C_SHA256_DIGEST_SIZE,
           digest, VAE_C_SHA256_DIGEST_SIZE);
    CHECK(write_bytes(bad, bytes, size));
    CHECK(vae_c_iic_checkpoint_load(
        bad, &identity, target, &loaded) ==
        VAE_C_IIC_CHECKPOINT_IDENTITY_MISMATCH);
    CHECK(same_state(source, target));
    printf("IIC_CHECKPOINT_K_FORMAT_MISMATCH_REJECTED=true\n");

    bytes[12] = VAE_C_IIC_ENSEMBLE_K;
    bytes[16] = 37u;
    CHECK(vae_c_sha256_digest(bytes, size - VAE_C_SHA256_DIGEST_SIZE,
                              digest) == VAE_C_SHA256_OK);
    memcpy(bytes + size - VAE_C_SHA256_DIGEST_SIZE,
           digest, VAE_C_SHA256_DIGEST_SIZE);
    CHECK(write_bytes(bad, bytes, size));
    CHECK(vae_c_iic_checkpoint_load(
        bad, &identity, target, &loaded) ==
        VAE_C_IIC_CHECKPOINT_IDENTITY_MISMATCH);
    CHECK(same_state(source, target));
    printf("IIC_CHECKPOINT_C_FORMAT_MISMATCH_REJECTED=true\n");

    bytes[16] = VAE_C_IIC_ENSEMBLE_MAIN_DIM;
    bytes[20] = 249u;
    CHECK(vae_c_sha256_digest(bytes, size - VAE_C_SHA256_DIGEST_SIZE,
                              digest) == VAE_C_SHA256_OK);
    memcpy(bytes + size - VAE_C_SHA256_DIGEST_SIZE,
           digest, VAE_C_SHA256_DIGEST_SIZE);
    CHECK(write_bytes(bad, bytes, size));
    CHECK(vae_c_iic_checkpoint_load(
        bad, &identity, target, &loaded) ==
        VAE_C_IIC_CHECKPOINT_IDENTITY_MISMATCH);
    CHECK(same_state(source, target));
    printf("IIC_CHECKPOINT_W_FORMAT_MISMATCH_REJECTED=true\n");

    bytes[20] = VAE_C_IIC_ENSEMBLE_OVER_DIM;
    bytes[36] ^= 0x01u;
    CHECK(vae_c_sha256_digest(bytes, size - VAE_C_SHA256_DIGEST_SIZE,
                              digest) == VAE_C_SHA256_OK);
    memcpy(bytes + size - VAE_C_SHA256_DIGEST_SIZE,
           digest, VAE_C_SHA256_DIGEST_SIZE);
    CHECK(write_bytes(bad, bytes, size));
    CHECK(vae_c_iic_checkpoint_load(
        bad, &identity, target, &loaded) ==
        VAE_C_IIC_CHECKPOINT_IDENTITY_MISMATCH);
    CHECK(same_state(source, target));
    printf("IIC_CHECKPOINT_ADAM_FORMAT_MISMATCH_REJECTED=true\n");

    metadata.kind = VAE_C_IIC_CHECKPOINT_FINAL;
    metadata.last_completed_epoch = 2u;
    CHECK(vae_c_iic_checkpoint_save_atomic(
        final, &identity, &metadata, source) != VAE_C_IIC_CHECKPOINT_OK);
    printf("IIC_CHECKPOINT_EPOCH_STEP_MISMATCH_REJECTED=true\n");
    metadata.last_completed_epoch = 0u;
    CHECK(vae_c_iic_checkpoint_save_atomic(
        final, &identity, &metadata, source) != VAE_C_IIC_CHECKPOINT_OK);
    printf("IIC_CHECKPOINT_NONTERMINAL_FINAL_REJECTED=true\n");
    set_step(source, 9u);
    metadata.last_completed_epoch = 2u;
    CHECK(vae_c_iic_checkpoint_save_atomic(
        final, &identity, &metadata, source) == VAE_C_IIC_CHECKPOINT_OK);
    CHECK(vae_c_iic_checkpoint_save_atomic(
        final, &identity, &metadata, source) == VAE_C_IIC_CHECKPOINT_EXISTS);
    CHECK(vae_c_iic_checkpoint_load(
        final, &identity, target, &loaded) == VAE_C_IIC_CHECKPOINT_OK);
    CHECK(loaded.kind == VAE_C_IIC_CHECKPOINT_FINAL &&
          loaded.last_completed_epoch == 2u && same_state(source, target));
    CHECK(vae_c_iic_ensemble_validate(target) == VAE_C_OP_OK);
    CHECK(file_absent(partial));
    printf("IIC_CHECKPOINT_FINAL_NO_OVERWRITE=true\n");

done:
    free(bytes);
    if (source != NULL) {
        vae_c_iic_ensemble_free(source);
        free(source);
    }
    if (target != NULL) {
        vae_c_iic_ensemble_free(target);
        free(target);
    }
    if (fixture_created) {
        if (latest[0] != 0) (void)remove(latest);
        if (final[0] != 0) (void)remove(final);
        if (bad[0] != 0) (void)remove(bad);
        if (partial[0] != 0) (void)remove(partial);
        (void)TEST_RMDIR(fixture_dir);
    }
    if (ok) {
        puts("IIC_CHECKPOINT_TEST_PASS=true");
    }
    return ok ? 0 : 1;
}
