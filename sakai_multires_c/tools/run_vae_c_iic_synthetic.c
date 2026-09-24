#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif

#include "vae_c_iic_checkpoint.h"
#include "vae_c_sha256.h"
#include "vae_c_shuffle.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#include <sys/stat.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

/* Deliberately synthetic-only: no real dataset, encoder, labels or results. */
#define ROWS 23u
#define DIM 4u
#define BATCH 8u
#define BATCHES 3u
#define PATH_CAP 1024u
#define MAX_EXECUTABLE_BYTES (64u * 1024u * 1024u)
#define HEADER "epoch,pair,main_loss,over_loss,total_loss,optimizer_step\n"

typedef struct {
    char directory[PATH_CAP];
    uint64_t max_epochs;
    uint64_t stop_after;
    bool resume;
    bool stop_requested;
} Options;

static bool number(const char *text, uint64_t *value)
{
    char *end;
    unsigned long long parsed;
    if (text == NULL || text[0] == '\0' || text[0] == '-') return false;
    errno = 0;
    parsed = strtoull(text, &end, 10);
    if (errno != 0 || *end != '\0') return false;
    *value = (uint64_t)parsed;
    return (unsigned long long)*value == parsed;
}

static bool options_parse(int argc, char **argv, Options *opt)
{
    int i;
    bool dir = false, budget = false;
    memset(opt, 0, sizeof(*opt));
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--output-dir") == 0 && ++i < argc && !dir) {
            size_t length = strlen(argv[i]);
            if (length == 0 || length >= PATH_CAP) return false;
            memcpy(opt->directory, argv[i], length + 1);
            dir = true;
        } else if (strcmp(argv[i], "--max-epochs") == 0 &&
                   ++i < argc && !budget) {
            if (!number(argv[i], &opt->max_epochs) ||
                opt->max_epochs == 0 || opt->max_epochs > 100) return false;
            budget = true;
        } else if (strcmp(argv[i], "--stop-after") == 0 &&
                   ++i < argc && !opt->stop_requested) {
            if (!number(argv[i], &opt->stop_after)) return false;
            opt->stop_requested = true;
        } else if (strcmp(argv[i], "--resume") == 0 && !opt->resume) {
            opt->resume = true;
        } else {
            return false;
        }
    }
    return dir && budget &&
        (!opt->stop_requested || opt->stop_after < opt->max_epochs);
}

static bool under_temp(const char *path)
{
#ifdef _WIN32
    char root[PATH_CAP], absolute[PATH_CAP];
    DWORD n = GetTempPathA(PATH_CAP, root);
    DWORD m;
    size_t index;
    if (n == 0 || n >= PATH_CAP) return false;
    m = GetFullPathNameA(path, PATH_CAP, absolute, NULL);
    if (m <= n || m >= PATH_CAP ||
        absolute[1] != ':' ||
        _strnicmp(root, absolute, n) != 0) return false;
    /* A lexical prefix is insufficient: a temp junction could redirect
     * writes into the repository. Reject every reparse-point ancestor,
     * including ancestors of the configured temp directory. */
    for (index = 3u; index <= (size_t)m; index++) {
        if (index == (size_t)m || absolute[index] == '\\' ||
            absolute[index] == '/') {
            char saved = absolute[index];
            DWORD attributes;
            absolute[index] = '\0';
            attributes = GetFileAttributesA(absolute);
            absolute[index] = saved;
            if (attributes != INVALID_FILE_ATTRIBUTES &&
                (attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0)
                return false;
            if (index < (size_t)m &&
                (attributes == INVALID_FILE_ATTRIBUTES ||
                 (attributes & FILE_ATTRIBUTE_DIRECTORY) == 0))
                return false;
        }
    }
    return true;
#else
    const char *root = getenv("TMPDIR");
    size_t n;
    if (root == NULL || *root == '\0') root = "/tmp";
    n = strlen(root);
    return path[0] == '/' && strncmp(path, root, n) == 0 &&
        path[n] == '/';
#endif
}

static bool join(char path[PATH_CAP], const char *directory,
                 const char *name)
{
    int n = snprintf(path, PATH_CAP, "%s/%s", directory, name);
    return n > 0 && n < (int)PATH_CAP;
}

static bool exists(const char *path)
{
#ifdef _WIN32
    return GetFileAttributesA(path) != INVALID_FILE_ATTRIBUTES;
#else
    struct stat st;
    return stat(path, &st) == 0;
#endif
}

static bool create_directory(const char *path)
{
#ifdef _WIN32
    return CreateDirectoryA(path, NULL) != 0;
#else
    return mkdir(path, 0700) == 0;
#endif
}

static bool replace(const char *source, const char *destination)
{
#ifdef _WIN32
    return MoveFileExA(source, destination,
        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    return rename(source, destination) == 0;
#endif
}

static bool executable_sha256(char hex[VAE_C_SHA256_HEX_SIZE])
{
    char path[PATH_CAP];
    FILE *input = NULL;
    unsigned char *bytes = NULL;
    long length;
    bool success = false;
#ifdef _WIN32
    DWORD n = GetModuleFileNameA(NULL, path, PATH_CAP);
    if (n == 0 || n >= PATH_CAP) return false;
#elif defined(__APPLE__)
    uint32_t n = PATH_CAP;
    if (_NSGetExecutablePath(path, &n) != 0) return false;
#else
    ssize_t n = readlink("/proc/self/exe", path, PATH_CAP - 1u);
    if (n <= 0 || (size_t)n >= PATH_CAP) return false;
    path[n] = '\0';
#endif
    input = fopen(path, "rb");
    if (input == NULL || fseek(input, 0, SEEK_END) != 0) goto done;
    length = ftell(input);
    if (length <= 0 || (unsigned long)length > MAX_EXECUTABLE_BYTES ||
        fseek(input, 0, SEEK_SET) != 0) goto done;
    bytes = malloc((size_t)length);
    if (bytes == NULL ||
        fread(bytes, 1, (size_t)length, input) != (size_t)length ||
        fgetc(input) != EOF || ferror(input)) goto done;
    success = vae_c_sha256_digest_hex_lower(
        bytes, (size_t)length, hex) == VAE_C_SHA256_OK;
done:
    free(bytes);
    if (input != NULL && fclose(input) != 0) success = false;
    return success;
}

static bool make_identity(VAECIICCheckpointIdentity *identity,
                          uint64_t max_epochs)
{
    size_t k;
    vae_c_iic_checkpoint_identity_init(identity);
    identity->input_dim = DIM;
    identity->capacity = BATCH;
    identity->batch_size = BATCH;
    identity->train_rows = ROWS;
    identity->max_epochs = max_epochs;
    identity->shuffle_seed = VAE_C_IIC_ENSEMBLE_SHUFFLE_SEED;
    for (k = 0; k < VAE_C_IIC_ENSEMBLE_K; k++)
        identity->seeds[k] = VAE_C_IIC_ENSEMBLE_FIRST_SEED + (uint64_t)k;
    strcpy(identity->run_id, "iic_k5_synthetic_c5b6_v1");
    strcpy(identity->base_commit,
           "2bbc7c9b670f7778ce5b25779a7933dcf9ddf891");
    strcpy(identity->producer_state, "WIP_UNCOMMITTED_MINI_C5B6");
    return executable_sha256(identity->producer_binary_sha256);
}

static bool make_batch(const size_t permutation[ROWS], size_t offset,
                       size_t count, VAECTensor *a, VAECTensor *b)
{
    const size_t dims[2] = {count, DIM};
    VAECShape shape;
    size_t row;
    if (vae_c_shape_make(2, dims, &shape) != VAE_C_STORAGE_OK ||
        vae_c_tensor_allocate(a, &shape, true) != VAE_C_STORAGE_OK ||
        vae_c_tensor_allocate(b, &shape, true) != VAE_C_STORAGE_OK)
        return false;
    for (row = 0; row < count; row++) {
        size_t feature, event = permutation[offset + row];
        for (feature = 0; feature < DIM; feature++) {
            size_t i = row * DIM + feature;
            int base = (int)((event * 17 + feature * 5) % 31) - 15;
            int perturbation = (int)((event + feature) % 5) - 2;
            a->data[i] = (float)base * 0.0625f;
            b->data[i] = a->data[i] * 0.75f +
                (float)perturbation * 0.03125f;
        }
    }
    return true;
}

/* Only future metric rows can be dropped. Every row through LATEST is
 * validated and copied verbatim before an atomic replacement. */
static bool reconcile(const char *path, bool latest, uint64_t last_epoch)
{
    char partial[PATH_CAP], line[256];
    FILE *in = NULL, *out = NULL;
    size_t committed = 0;
    bool future = false, success = false;
    if (snprintf(partial, PATH_CAP, "%s.partial", path) >= (int)PATH_CAP)
        return false;
    in = fopen(path, "rb");
    if (latest && in == NULL) return false;
    out = fopen(partial, "wb");
    if (out == NULL || fputs(HEADER, out) == EOF) goto done;
    if (in != NULL) {
        if (fgets(line, sizeof(line), in) == NULL ||
            (strcmp(line, HEADER) != 0 &&
             (latest ||
              strcmp(line,
                  "epoch,pair,main_loss,over_loss,total_loss,optimizer_step\r\n")
                  != 0))) goto done;
        while (fgets(line, sizeof(line), in) != NULL) {
            unsigned long long e, step;
            unsigned pair;
            double main_loss, over_loss, total_loss;
            char *after_epoch;
            int consumed = -1;
            int columns;
            if (strchr(line, '\n') == NULL || line[0] == '-') goto done;
            errno = 0;
            e = strtoull(line, &after_epoch, 10);
            if (errno != 0 || after_epoch == line || *after_epoch != ',')
                goto done;
            if (!latest || (uint64_t)e > last_epoch) {
                future = true;
                continue;
            }
            if (future) goto done;
            columns = sscanf(line, "%llu,%u,%lf,%lf,%lf,%llu%n",
                &e, &pair, &main_loss, &over_loss, &total_loss,
                &step, &consumed);
            if (columns != 6 || consumed < 0 ||
                line[consumed] != '\n' || line[consumed + 1] != '\0' ||
                e != committed / VAE_C_IIC_ENSEMBLE_K ||
                pair != committed % VAE_C_IIC_ENSEMBLE_K ||
                step != (e + 1u) * BATCHES ||
                !isfinite(main_loss) || !isfinite(over_loss) ||
                !isfinite(total_loss) ||
                fabs(total_loss - main_loss - over_loss) > 1.0e-3 ||
                fputs(line, out) == EOF) goto done;
            committed++;
        }
        if (ferror(in)) goto done;
    }
    if (latest &&
        committed != (size_t)(last_epoch + 1u) * VAE_C_IIC_ENSEMBLE_K)
        goto done;
    if (fflush(out) != 0) goto done;
    if (fclose(out) != 0) { out = NULL; goto done; }
    out = NULL;
    if (in != NULL && fclose(in) != 0) { in = NULL; goto done; }
    in = NULL;
    if (latest && !future) {
        if (remove(partial) != 0) goto done;
        success = true;
        goto done;
    }
    if (!replace(partial, path)) goto done;
    success = true;
done:
    if (in != NULL) fclose(in);
    if (out != NULL) fclose(out);
    if (!success) remove(partial);
    return success;
}

static bool run_epoch(VAECIICEnsemble *ensemble, uint64_t epoch,
                      const char *metrics)
{
    size_t permutation[ROWS], offset, k;
    double main_loss[5] = {0}, over_loss[5] = {0}, total_loss[5] = {0};
    FILE *out;
    if (vae_c_shuffle_make_permutation(
            ROWS, VAE_C_IIC_ENSEMBLE_SHUFFLE_SEED, epoch,
            permutation) != VAE_C_RNG_OK) return false;
    for (offset = 0; offset < ROWS; offset += BATCH) {
        size_t remaining = ROWS - offset;
        size_t count = remaining < BATCH ? remaining : BATCH;
        VAECTensor a, b;
        bool ok;
        vae_c_tensor_init(&a);
        vae_c_tensor_init(&b);
        ok = make_batch(permutation, offset, count, &a, &b) &&
            vae_c_iic_ensemble_train_step(ensemble, &a, &b) == VAE_C_OP_OK;
        for (k = 0; ok && k < VAE_C_IIC_ENSEMBLE_K; k++) {
            const VAECIICPairWorkspace *w = &ensemble->workspaces[k];
            if (!isfinite(w->main_result.loss) ||
                !isfinite(w->over_result.loss) ||
                !isfinite(w->total_loss) ||
                ensemble->optimizers[k].step !=
                    epoch * BATCHES + offset / BATCH + 1u) {
                ok = false;
                break;
            }
            main_loss[k] += (double)w->main_result.loss / BATCHES;
            over_loss[k] += (double)w->over_result.loss / BATCHES;
            total_loss[k] += (double)w->total_loss / BATCHES;
        }
        vae_c_tensor_free(&b);
        vae_c_tensor_free(&a);
        if (!ok) return false;
    }
    out = fopen(metrics, "ab");
    if (out == NULL) return false;
    for (k = 0; k < VAE_C_IIC_ENSEMBLE_K; k++) {
        if (fprintf(out, "%" PRIu64 ",%zu,%.17g,%.17g,%.17g,%" PRIu64 "\n",
                    epoch, k, main_loss[k], over_loss[k], total_loss[k],
                    ensemble->optimizers[k].step) < 0) {
            fclose(out);
            return false;
        }
    }
    if (fflush(out) != 0) { fclose(out); return false; }
    return fclose(out) == 0;
}

int main(int argc, char **argv)
{
    Options opt;
    VAECIICCheckpointIdentity identity;
    VAECIICCheckpointMetadata metadata;
    VAECIICEnsemble ensemble;
    char latest_path[PATH_CAP], final_path[PATH_CAP], metrics[PATH_CAP];
    uint64_t epoch, start = 0;
    bool latest;
    int result = 1;
    if (!options_parse(argc, argv, &opt) ||
        !under_temp(opt.directory) ||
        !join(latest_path, opt.directory, "latest.ckpt") ||
        !join(final_path, opt.directory, "final.ckpt") ||
        !join(metrics, opt.directory, "metrics.csv") ||
        !make_identity(&identity, opt.max_epochs)) {
        fprintf(stderr, "Invalid synthetic-only arguments/output path\n");
        return 2;
    }
    printf("IIC_SYNTHETIC_PRODUCER_BINARY_SHA256=%s\n",
           identity.producer_binary_sha256);
    if (exists(final_path)) {
        fprintf(stderr, "FINAL exists; refusing execution\n");
        return 2;
    }
    if (opt.resume) {
        if (!exists(opt.directory)) return 2;
    } else if (!create_directory(opt.directory)) {
        fprintf(stderr, "Output directory exists or cannot be created\n");
        return 2;
    }
    latest = exists(latest_path);
    if (latest && !opt.resume) return 2;
    vae_c_iic_checkpoint_metadata_init(&metadata);
    vae_c_iic_ensemble_init(&ensemble);
    if (vae_c_iic_ensemble_allocate_v1(
            &ensemble, DIM, BATCH) != VAE_C_OP_OK) goto done;
    if (latest) {
        VAECIICCheckpointStatus status = vae_c_iic_checkpoint_load(
            latest_path, &identity, &ensemble, &metadata);
        if (status != VAE_C_IIC_CHECKPOINT_OK ||
            metadata.kind != VAE_C_IIC_CHECKPOINT_LATEST ||
            metadata.last_completed_epoch >= opt.max_epochs) {
            fprintf(stderr, "LATEST load failed: %s\n",
                vae_c_iic_checkpoint_status_name(status));
            goto done;
        }
        start = metadata.last_completed_epoch + 1u;
    }
    if (!reconcile(metrics, latest, metadata.last_completed_epoch)) {
        fprintf(stderr, "Metrics reconciliation failed\n");
        goto done;
    }
    if (latest && start == opt.max_epochs) {
        metadata.kind = VAE_C_IIC_CHECKPOINT_FINAL;
        if (vae_c_iic_checkpoint_save_atomic(
                final_path, &identity, &metadata, &ensemble) !=
            VAE_C_IIC_CHECKPOINT_OK) goto done;
        puts("IIC_SYNTHETIC_FINALIZED_FROM_LATEST=true");
        result = 0;
        goto done;
    }
    if (latest)
        printf("IIC_SYNTHETIC_RESUME_FROM_EPOCH=%" PRIu64 "\n", start);
    else
        puts("IIC_SYNTHETIC_RESTART_FROM_EPOCH_0=true");
    for (epoch = start; epoch < opt.max_epochs; epoch++) {
        VAECIICCheckpointStatus status;
        if (!run_epoch(&ensemble, epoch, metrics)) goto done;
        metadata.kind = VAE_C_IIC_CHECKPOINT_LATEST;
        metadata.last_completed_epoch = epoch;
        status = vae_c_iic_checkpoint_save_atomic(
            latest_path, &identity, &metadata, &ensemble);
        if (status != VAE_C_IIC_CHECKPOINT_OK) {
            fprintf(stderr, "LATEST save failed: %s\n",
                vae_c_iic_checkpoint_status_name(status));
            goto done;
        }
        printf("IIC_SYNTHETIC_EPOCH=%" PRIu64 " OPTIMIZER_STEP=%" PRIu64 "\n",
               epoch, ensemble.optimizers[0].step);
        if (opt.stop_requested && epoch == opt.stop_after) {
            puts("IIC_SYNTHETIC_CONTROLLED_STOP=true");
            result = 0;
            goto done;
        }
    }
    metadata.kind = VAE_C_IIC_CHECKPOINT_FINAL;
    if (vae_c_iic_checkpoint_save_atomic(
            final_path, &identity, &metadata, &ensemble) !=
        VAE_C_IIC_CHECKPOINT_OK) goto done;
    puts("IIC_SYNTHETIC_COMPLETED=true");
    result = 0;
done:
    vae_c_iic_ensemble_free(&ensemble);
    return result;
}
