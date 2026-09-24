#include "vae_c_iic_checkpoint.h"
#include "vae_c_iic_h.h"
#ifndef H_CENTRAL_HEADER
#define H_CENTRAL_HEADER "vae_c_iic_mini75_central.h"
#define H_CENTRAL_TYPE VAECIICMini75Central
#define H_CENTRAL_ROW_TYPE VAECIICMini75CentralRow
#define H_CENTRAL_INIT vae_c_iic_mini75_central_init
#define H_CENTRAL_LOAD vae_c_iic_mini75_central_load
#define H_CENTRAL_FREE vae_c_iic_mini75_central_free
#define H_SOURCE_TRAIN VAE_C_IIC_MINI75_SOURCE_TRAIN
#define H_SOURCE_VALIDATION VAE_C_IIC_MINI75_SOURCE_VALIDATION
#define H_ROWS VAE_C_IIC_MINI75_ROWS
#define H_LATENT_DIM VAE_C_IIC_MINI75_LATENT_DIM
#define H_RUN_ID "mini75_iic_v1_737da555866b822e14c545c4f2687869f41a52cd1340b9e941f3a1ce77763eb9"
#define H_FINAL_SHA "6df9bb278210861b449adec454c4f649fee8be1398082418096c2679aee907d5"
#define H_RUN_CONFIG_SHA "5323b1c00c8f6a4105a4a2f015ab377a523fb94a8b1e97163b88b45cd61a4b5b"
#define H_EMBEDDING_DIR "mini75_experiment_v1_full_best_epoch47_posterior_mu_central_v1"
#define H_TENSOR_MANIFEST "manifest_tensors_75_central.csv"
#define H_PROFILE_ID "mini75_experiment_v1"
#define H_IIC_BASE_COMMIT "ba3f2e5b244054b6c11a053c749da3c29f69c298"
#define H_IIC_BINARY_SHA "0371657dbb4b7dff2157b93228499a92d1297983fc1c21891d14f4b775b4fcd5"
#endif
#ifndef H_FINAL_STEP
#define H_FINAL_STEP 6800u
#endif
#ifndef H_FINAL_BN_BATCHES
#define H_FINAL_BN_BATCHES 13600u
#endif
#ifndef H_ROW_ORDER
#define H_ROW_ORDER "manifest_index_0_8582"
#endif
#ifndef H_ROW_SOURCE
#define H_ROW_SOURCE(row) ((row)->source)
#endif
#ifndef H_ROW_VALID
#define H_ROW_VALID(row, index) \
    ((row)->manifest_index == (index) && \
     (row)->event_index == (index) + 1u)
#endif
#include H_CENTRAL_HEADER
#include "vae_c_mu_contract.h"

#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

#define H_PATH_CAP 4096u
#define H_COLS VAE_C_IIC_H_COLUMNS
#define H_PAYLOAD_BYTES (H_ROWS * H_COLS * sizeof(float))

enum {
    SRC_FINAL, SRC_RUN_CONFIG, SRC_MU_SPEC, SRC_MU_TRAIN,
    SRC_ROWS_TRAIN, SRC_MU_VALIDATION, SRC_ROWS_VALIDATION,
    SRC_MU_MANIFEST, SRC_EVENT_MANIFEST, SRC_TENSOR_MANIFEST,
    SRC_TRAIN_SPLIT, SRC_VALIDATION_SPLIT,
#ifdef H_TRAIN_SUBSET
    SRC_TRAIN_SUBSET, SRC_VALIDATION_SUBSET,
#endif
    SRC_COUNT
};

static const char *const source_relative[SRC_COUNT] = {
    "sakai_multires_c/results/vae_c/iic/" H_RUN_ID "/final.ckpt",
    "sakai_multires_c/results/vae_c/iic/" H_RUN_ID "/run_config.json",
    "sakai_multires_c/results/vae_c/embeddings/" H_EMBEDDING_DIR "/embedding_spec.txt",
    "sakai_multires_c/results/vae_c/embeddings/" H_EMBEDDING_DIR "/train_mu_f32.bin",
    "sakai_multires_c/results/vae_c/embeddings/" H_EMBEDDING_DIR "/train_rows.csv",
    "sakai_multires_c/results/vae_c/embeddings/" H_EMBEDDING_DIR "/validation_mu_f32.bin",
    "sakai_multires_c/results/vae_c/embeddings/" H_EMBEDDING_DIR "/validation_rows.csv",
    "sakai_multires_c/results/vae_c/embeddings/" H_EMBEDDING_DIR "/sha256.txt",
    "manifest_events_272.csv", H_TENSOR_MANIFEST,
    "split_train_indices_seed123.csv", "split_val_indices_seed123.csv"
#ifdef H_TRAIN_SUBSET
    , H_TRAIN_SUBSET, H_VALIDATION_SUBSET
#endif
};
static const char *const source_keys[SRC_COUNT] = {
    "iic_final", "iic_run_config", "embedding_spec", "train_mu",
    "train_rows", "validation_mu", "validation_rows",
    "mu_sha256_manifest", "event_manifest", "tensor_manifest",
    "train_split", "validation_split"
#ifdef H_TRAIN_SUBSET
    , "train_subset", "validation_subset"
#endif
};
static const char *const artifact_names[3] = {
    "h_f32.bin", "rows.csv", "h_spec.txt"
};

typedef struct {
    const char *repository_root;
    const char *output_dir;
    const char *producer_commit;
    const char *producer_binary_sha256;
    size_t chunk_size;
    int validate_only;
} Options;

typedef struct {
    char path[H_PATH_CAP];
    char sha[65];
} Source;

static int joined(const char *base, const char *name, char path[H_PATH_CAP])
{
    int n = snprintf(path, H_PATH_CAP, "%s/%s", base, name);
    return n >= 0 && (size_t)n < H_PATH_CAP;
}

static int sha_file(const char *path, char sha[65])
{
    return vae_c_mu_file_sha256(path, sha) == VAE_C_MU_OK;
}

static int lower_hex(const char *text, size_t length)
{
    size_t i;
    if (text == NULL || strlen(text) != length) return 0;
    for (i = 0u; i < length; i++) {
        if (!((text[i] >= '0' && text[i] <= '9') ||
              (text[i] >= 'a' && text[i] <= 'f'))) return 0;
    }
    return 1;
}

static int parse_options(int argc, char **argv, Options *options)
{
    int i;
    memset(options, 0, sizeof(*options));
    for (i = 1; i < argc; i++) {
        const char *key = argv[i];
        const char *value;
        char *end;
        unsigned long long parsed;
        if (strcmp(key, "--validate-only") == 0) {
            if (options->validate_only) return 0;
            options->validate_only = 1;
            continue;
        }
        if (++i >= argc) return 0;
        value = argv[i];
        if (strcmp(key, "--repository-root") == 0 &&
            options->repository_root == NULL) options->repository_root = value;
        else if (strcmp(key, "--output-dir") == 0 &&
                 options->output_dir == NULL) options->output_dir = value;
        else if (strcmp(key, "--producer-commit") == 0 &&
                 options->producer_commit == NULL) options->producer_commit = value;
        else if (strcmp(key, "--producer-binary-sha256") == 0 &&
                 options->producer_binary_sha256 == NULL)
            options->producer_binary_sha256 = value;
        else if (strcmp(key, "--chunk-size") == 0 &&
                 options->chunk_size == 0u) {
            errno = 0;
            parsed = strtoull(value, &end, 10);
            if (errno != 0 || value[0] == '\0' || *end != '\0' ||
                parsed == 0u || parsed > 128u) return 0;
            options->chunk_size = (size_t)parsed;
        } else return 0;
    }
    return options->repository_root != NULL && options->output_dir != NULL &&
           options->repository_root[0] != '\0' &&
           options->output_dir[0] != '\0' &&
           options->chunk_size != 0u &&
           lower_hex(options->producer_commit, 40u) &&
           lower_hex(options->producer_binary_sha256, 64u);
}

static int exists(const char *path)
{
    struct stat info;
    return stat(path, &info) == 0;
}

static int source_hashes(const char *root, Source sources[SRC_COUNT])
{
    size_t i;
    for (i = 0u; i < SRC_COUNT; i++) {
        if (!joined(root, source_relative[i], sources[i].path) ||
            !sha_file(sources[i].path, sources[i].sha)) return 0;
    }
    return strcmp(sources[SRC_FINAL].sha, H_FINAL_SHA) == 0 &&
           strcmp(sources[SRC_RUN_CONFIG].sha, H_RUN_CONFIG_SHA) == 0;
}

static int sources_unchanged(const Source sources[SRC_COUNT])
{
    size_t i;
    char current[65];
    for (i = 0u; i < SRC_COUNT; i++) {
        if (!sha_file(sources[i].path, current) ||
            strcmp(current, sources[i].sha) != 0) return 0;
    }
    return 1;
}

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
    const float *arrays[4] = {p->values, p->gradients, p->adam_m, p->adam_v};
    size_t i;
    for (i = 0u; i < 4u; i++)
        h = hash_bytes(h, arrays[i], p->shape.count * sizeof(float));
    return h;
}

/* This checks persistent state, including overcluster, without using it in H. */
static uint64_t persistent_digest(const VAECIICEnsemble *ensemble)
{
    uint64_t h = UINT64_C(14695981039346656037);
    size_t k, branch;
    for (k = 0u; k < VAE_C_IIC_ENSEMBLE_K; k++) {
        const VAECIICHead *heads[2] = {
            &ensemble->pairs[k].main_head,
            &ensemble->pairs[k].overcluster_head
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

static int expected_identity(VAECIICCheckpointIdentity *identity)
{
    size_t k;
    vae_c_iic_checkpoint_identity_init(identity);
    identity->input_dim = H_LATENT_DIM;
    identity->capacity = 128u;
    identity->batch_size = 128u;
    identity->train_rows = H_ROWS;
    identity->max_epochs = 100u;
    identity->shuffle_seed = VAE_C_IIC_ENSEMBLE_SHUFFLE_SEED;
    for (k = 0u; k < VAE_C_IIC_ENSEMBLE_K; k++)
        identity->seeds[k] = VAE_C_IIC_ENSEMBLE_FIRST_SEED + (uint64_t)k;
    return snprintf(identity->run_id, sizeof(identity->run_id), "%s", H_RUN_ID) > 0 &&
        snprintf(identity->base_commit, sizeof(identity->base_commit), "%s",
                 H_IIC_BASE_COMMIT) > 0 &&
        snprintf(identity->producer_state, sizeof(identity->producer_state),
                 "%s", "BINARY_SHA256_AUTHORITY") > 0 &&
        snprintf(identity->producer_binary_sha256,
                 sizeof(identity->producer_binary_sha256), "%s",
                 H_IIC_BINARY_SHA) > 0;
}

static int platform_ok(void)
{
    const uint32_t one = 1u;
    return sizeof(float) == 4u && FLT_RADIX == 2 && FLT_MANT_DIG == 24 &&
        FLT_MAX_EXP == 128 && *(const unsigned char *)&one == 1u;
}

static int make_h(VAECIICEnsemble *ensemble,
                  const H_CENTRAL_TYPE *central, size_t chunk,
                  float *h, double *minimum, double *maximum,
                  double *max_block_error, double *max_row_error)
{
    VAECIICHWorkspace workspace;
    size_t start, row, k, c;
    uint64_t state_before;
    int ok = 0;
    vae_c_iic_h_workspace_init(&workspace);
    if (vae_c_iic_h_workspace_allocate(ensemble, chunk, &workspace) !=
        VAE_C_OP_OK) return 0;
    state_before = persistent_digest(ensemble);
    for (start = 0u; start < H_ROWS; start += chunk) {
        VAECTensor input;
        size_t rows = H_ROWS - start, dims[2];
        if (rows > chunk) rows = chunk;
        dims[0] = rows;
        dims[1] = H_LATENT_DIM;
        vae_c_tensor_init(&input);
        if (vae_c_shape_make(2u, dims, &input.shape) != VAE_C_STORAGE_OK)
            goto done;
        input.data = central->mu + start * H_LATENT_DIM;
        if (vae_c_iic_h_forward(ensemble, &input, &workspace,
                h + start * H_COLS, rows * H_COLS) != VAE_C_OP_OK) goto done;
    }
    if (persistent_digest(ensemble) != state_before) goto done;
    *minimum = 1.0;
    *maximum = 0.0;
    *max_block_error = 0.0;
    *max_row_error = 0.0;
    for (row = 0u; row < H_ROWS; row++) {
        double row_sum = 0.0;
        for (k = 0u; k < VAE_C_IIC_ENSEMBLE_K; k++) {
            double block_sum = 0.0;
            for (c = 0u; c < VAE_C_IIC_ENSEMBLE_MAIN_DIM; c++) {
                double value = h[row * H_COLS +
                    k * VAE_C_IIC_ENSEMBLE_MAIN_DIM + c];
                if (!isfinite(value) || value < 0.0 || value > 1.0)
                    goto done;
                if (value < *minimum) *minimum = value;
                if (value > *maximum) *maximum = value;
                block_sum += value;
            }
            if (fabs(block_sum - 1.0) > *max_block_error)
                *max_block_error = fabs(block_sum - 1.0);
            row_sum += block_sum;
        }
        if (fabs(row_sum - 5.0) > *max_row_error)
            *max_row_error = fabs(row_sum - 5.0);
    }
    ok = *max_block_error <= 1.0e-4 && *max_row_error <= 5.0e-4;
done:
    vae_c_iic_h_workspace_free(&workspace);
    return ok;
}

static int write_h(const char *directory, const float *h, char sha[65])
{
    char path[H_PATH_CAP];
    FILE *file;
    int ok;
    if (!joined(directory, artifact_names[0], path)) return 0;
    file = fopen(path, "wb");
    if (file == NULL) return 0;
    ok = fwrite(h, sizeof(float), H_ROWS * H_COLS, file) == H_ROWS * H_COLS;
    if (fflush(file) != 0) ok = 0;
    if (fclose(file) != 0) ok = 0;
    return ok && sha_file(path, sha);
}

static int write_rows(const char *directory,
                      const H_CENTRAL_TYPE *central, char sha[65])
{
    char path[H_PATH_CAP];
    FILE *file;
    size_t row;
    int ok = 1;
    if (!joined(directory, artifact_names[1], path)) return 0;
    file = fopen(path, "wb");
    if (file == NULL) return 0;
    if (fprintf(file, "manifest_index,event_index,event_id,source_split,split_pos\n") < 0)
        ok = 0;
    for (row = 0u; ok && row < H_ROWS; row++) {
        const H_CENTRAL_ROW_TYPE *r = &central->rows[row];
        const char *split = H_ROW_SOURCE(r) == H_SOURCE_TRAIN ?
            "train" : "validation";
        if (!H_ROW_VALID(r, row) ||
            (H_ROW_SOURCE(r) != H_SOURCE_TRAIN &&
             H_ROW_SOURCE(r) != H_SOURCE_VALIDATION) ||
            strpbrk(r->event_id, ",\"\r\n") != NULL ||
            fprintf(file, "%u,%u,%s,%s,%u\n", r->manifest_index,
                    r->event_index, r->event_id, split, r->split_pos) < 0)
            ok = 0;
    }
    if (fflush(file) != 0) ok = 0;
    if (fclose(file) != 0) ok = 0;
    return ok && sha_file(path, sha);
}

static int write_spec(const char *directory, const Options *options,
                      const Source sources[SRC_COUNT], const char *h_sha,
                      const char *rows_sha, char spec_sha[65])
{
    char path[H_PATH_CAP];
    FILE *file;
    size_t i;
    int ok;
    if (!joined(directory, artifact_names[2], path)) return 0;
    file = fopen(path, "wb");
    if (file == NULL) return 0;
    ok = fprintf(file,
        "format=VAE_C_IIC_H_ARTIFACT_V1\n"
        "representation=main_head_probabilities\n"
        "view=central\n"
        "profile_id=" H_PROFILE_ID "\n"
        "rows=%u\ncols=180\ndtype=float32_ieee754\nendian=little\n"
        "input_latent_dim=%u\n"
        "layout=row_major_[row,classifier,class]\n"
        "payload_file=h_f32.bin\npayload_bytes=%llu\npayload_sha256=%s\n"
        "row_map_file=rows.csv\nrow_map_sha256=%s\n"
        "row_map_schema=manifest_index,event_index,event_id,source_split,split_pos\n"
        "row_order=" H_ROW_ORDER "\n"
        "K=5\nC=36\nclassifier_seeds=123,124,125,126,127\n"
        "classifier_order=123,124,125,126,127\n"
        "formula=H[i,36*k+c]=p_k,c(mu_i)\n"
        "main_only=true\novercluster_used=false\nlogits_used=false\n"
        "averaging=false\nargmax=false\naugmentation=false\n"
        "batch_norm_mode=EVAL\nbatch_norm_frozen=true\n"
        "optimizer_steps_during_production=0\n"
        "gradient_steps_during_production=0\nlabel_free=true\n"
        "iic_run_id=%s\niic_checkpoint_kind=FINAL\n"
        "iic_checkpoint_epoch=99\niic_checkpoint_optimizer_step=%u\n"
        "iic_final_sha256=%s\niic_run_config_sha256=%s\n"
        "producer_commit=%s\nproducer_binary_sha256=%s\n",
        (unsigned)H_ROWS, (unsigned)H_LATENT_DIM,
        (unsigned long long)H_PAYLOAD_BYTES, h_sha, rows_sha, H_RUN_ID,
        (unsigned)H_FINAL_STEP, sources[SRC_FINAL].sha,
        sources[SRC_RUN_CONFIG].sha, options->producer_commit,
        options->producer_binary_sha256) >= 0;
    for (i = 0u; ok && i < SRC_COUNT; i++) {
        ok = fprintf(file, "source_%s_path=%s\nsource_%s_sha256=%s\n",
            source_keys[i], source_relative[i], source_keys[i],
            sources[i].sha) >= 0;
    }
    if (ok) ok = fprintf(file, "checksum_file=sha256.txt\n"
                                "checksum_manifest_excludes_self=true\n") >= 0;
    if (fflush(file) != 0) ok = 0;
    if (fclose(file) != 0) ok = 0;
    return ok && sha_file(path, spec_sha);
}

static int write_manifest(const char *directory, char hashes[3][65])
{
    char path[H_PATH_CAP];
    FILE *file;
    size_t i;
    int ok = 1;
    if (!joined(directory, "sha256.txt", path)) return 0;
    file = fopen(path, "wb");
    if (file == NULL) return 0;
    for (i = 0u; i < 3u; i++) {
        if (fprintf(file, "%s  %s\n", hashes[i], artifact_names[i]) < 0)
            ok = 0;
    }
    if (fflush(file) != 0) ok = 0;
    if (fclose(file) != 0) ok = 0;
    return ok;
}

static void clean_own_partial(const char *directory)
{
    char path[H_PATH_CAP];
    size_t i;
    for (i = 0u; i < 3u; i++) {
        if (joined(directory, artifact_names[i], path)) (void)remove(path);
    }
    if (joined(directory, "sha256.txt", path)) (void)remove(path);
#ifdef _WIN32
    (void)_rmdir(directory);
#else
    (void)rmdir(directory);
#endif
}

static int publish_directory(const char *partial, const char *final)
{
    if (exists(final)) return 0;
#ifdef _WIN32
    return MoveFileExA(partial, final, MOVEFILE_WRITE_THROUGH) != 0;
#else
    return rename(partial, final) == 0;
#endif
}

int main(int argc, char **argv)
{
    Options options;
    Source sources[SRC_COUNT];
    VAECIICEnsemble ensemble;
    VAECIICCheckpointIdentity identity;
    VAECIICCheckpointMetadata metadata;
    H_CENTRAL_TYPE central;
    float *h = NULL;
    char partial[H_PATH_CAP], own_binary_sha[65], hashes[3][65],
         error[512];
    double minimum = 0.0, maximum = 0.0, block_error = 0.0,
           row_error = 0.0;
    size_t k;
    int partial_owned = 0, status = 1;
    vae_c_iic_ensemble_init(&ensemble);
    vae_c_iic_checkpoint_identity_init(&identity);
    vae_c_iic_checkpoint_metadata_init(&metadata);
    H_CENTRAL_INIT(&central);
    if (!parse_options(argc, argv, &options) || !platform_ok() ||
        snprintf(partial, sizeof(partial), "%s.partial", options.output_dir) < 0 ||
        strlen(options.output_dir) + strlen(".partial") >= sizeof(partial) ||
        exists(options.output_dir) || exists(partial) ||
        !sha_file(argv[0], own_binary_sha) ||
        strcmp(own_binary_sha, options.producer_binary_sha256) != 0 ||
        !source_hashes(options.repository_root, sources) ||
        !expected_identity(&identity)) {
        fputs("H preflight failed (arguments, platform, output, binary or sources)\n",
              stderr);
        goto done;
    }
    if (vae_c_iic_ensemble_allocate_v1(&ensemble, H_LATENT_DIM, 128u) != VAE_C_OP_OK ||
        vae_c_iic_checkpoint_load(sources[SRC_FINAL].path, &identity,
            &ensemble, &metadata) != VAE_C_IIC_CHECKPOINT_OK ||
        metadata.kind != VAE_C_IIC_CHECKPOINT_FINAL ||
        metadata.last_completed_epoch != 99u) {
        fputs("H FINAL checkpoint identity/load failed\n", stderr);
        goto done;
    }
    for (k = 0u; k < VAE_C_IIC_ENSEMBLE_K; k++) {
        if (ensemble.optimizers[k].step != H_FINAL_STEP ||
            ensemble.pairs[k].main_head.num_batches_tracked !=
                H_FINAL_BN_BATCHES ||
            ensemble.pairs[k].overcluster_head.num_batches_tracked !=
                H_FINAL_BN_BATCHES) {
            fputs("H FINAL counters mismatch\n", stderr);
            goto done;
        }
    }
    if (!H_CENTRAL_LOAD(options.repository_root, NULL,
            &central, error, sizeof(error)) || central.row_count != H_ROWS ||
        central.latent_dim != H_LATENT_DIM ||
        strcmp(central.train_matrix_sha256, sources[SRC_MU_TRAIN].sha) != 0 ||
        strcmp(central.validation_matrix_sha256,
               sources[SRC_MU_VALIDATION].sha) != 0 ||
        strcmp(central.train_rows_sha256, sources[SRC_ROWS_TRAIN].sha) != 0 ||
        strcmp(central.validation_rows_sha256,
               sources[SRC_ROWS_VALIDATION].sha) != 0 ||
        !sources_unchanged(sources)) {
        fprintf(stderr, "H central mu identity/load failed: %s\n", error);
        goto done;
    }
    if (options.validate_only) {
        puts("H_INPUT_VALIDATED=true\nH_FORWARD_CALLS=0\nOUTPUT_CREATED=false");
        status = 0;
        goto done;
    }
    h = (float *)malloc(H_PAYLOAD_BYTES);
    if (h == NULL || !make_h(&ensemble, &central, options.chunk_size, h,
                            &minimum, &maximum, &block_error, &row_error) ||
        !sources_unchanged(sources) || exists(options.output_dir) ||
        exists(partial)) {
        fputs("H inference/integrity failed\n", stderr);
        goto done;
    }
#ifdef _WIN32
    if (_mkdir(partial) != 0) goto done;
#else
    if (mkdir(partial, 0777) != 0) goto done;
#endif
    partial_owned = 1;
    if (!write_h(partial, h, hashes[0]) ||
        !write_rows(partial, &central, hashes[1]) ||
        !write_spec(partial, &options, sources, hashes[0], hashes[1], hashes[2]) ||
        !write_manifest(partial, hashes) || !sources_unchanged(sources) ||
        !publish_directory(partial, options.output_dir)) {
        fputs("H artifact write/publish failed\n", stderr);
        goto done;
    }
    partial_owned = 0;
    printf("H_CREATED=true\nH_ROWS=%u\nH_COLS=%u\nH_PAYLOAD_BYTES=%llu\n"
           "H_MIN=%.9g\nH_MAX=%.9g\nH_MAX_BLOCK_ERROR=%.9g\n"
           "H_MAX_ROW_ERROR=%.9g\nH_PAYLOAD_SHA256=%s\n"
           "H_ROW_MAP_SHA256=%s\nH_METADATA_SHA256=%s\n",
           (unsigned)H_ROWS, (unsigned)H_COLS,
           (unsigned long long)H_PAYLOAD_BYTES, minimum, maximum,
           block_error, row_error, hashes[0], hashes[1], hashes[2]);
    status = 0;
done:
    if (partial_owned) clean_own_partial(partial);
    free(h);
    H_CENTRAL_FREE(&central);
    vae_c_iic_ensemble_free(&ensemble);
    return status;
}
