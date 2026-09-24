#include "vae_c_iic_affinity.h"
#include "vae_c_mu_contract.h"

#include <errno.h>
#include <float.h>
#include <limits.h>
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
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#define PATH_CAP 4096u
#ifndef CANONICAL_ROWS
#define CANONICAL_ROWS 8583u
#endif
#define CANONICAL_COLS 180u
#ifndef CANONICAL_ROW_VALID
#define CANONICAL_ROW_VALID(manifest_index, event_index, split, split_pos, row) \
    ((manifest_index) == (row))
#endif
#define DEFAULT_BLOCK_ROWS 512u
#define CORE_COMMIT "3619ba1bd6705cd31fe3fc0b8edf16d18b0b3a8d"
#define NOTEBOOK_SHA "81894341420d94c993c916663e4ce83578f53734d7d1c35f9c0425e3b4198c8c"
#ifndef RUN_ID
#define RUN_ID "mini75_iic_v1_737da555866b822e14c545c4f2687869f41a52cd1340b9e941f3a1ce77763eb9"
#define H_SHA "1807ea44b7e3c884b04f7a35195731806916374caf8297e0294e7aaa55abeb44"
#define ROWS_SHA "6e09d72e3b54f213a2096edffca627fd35f1f0e35778539f05a810a788d7c642"
#define H_SPEC_SHA "d2cb16a590971d0156bac3db81e42f7167c7db6df435c131c2e285b6ffbfb739"
#define H_MANIFEST_SHA "e9a308e87a6662b09287f626b7b93994fa393f32c2ade384aed7c962c0739216"
#define FINAL_SHA "6df9bb278210861b449adec454c4f649fee8be1398082418096c2679aee907d5"
#define SOURCE_MODE "canonical_mini75"
#endif
#define H_REL "sakai_multires_c/results/vae_c/iic/" RUN_ID "_h_main_v1"
#define RUN_REL "sakai_multires_c/results/vae_c/iic/" RUN_ID

typedef struct {
    const char *repository_root;
    const char *output_dir;
    const char *producer_commit;
    const char *producer_binary_sha256;
    const char *fixture_dir;
    size_t fixture_rows, fixture_cols, block_rows;
    int validate_only;
} Options;

typedef struct {
    char h_path[PATH_CAP], rows_path[PATH_CAP], spec_path[PATH_CAP];
    char manifest_path[PATH_CAP], final_path[PATH_CAP];
    char h_sha[65], rows_sha[65], spec_sha[65];
    char manifest_sha[65], final_sha[65];
    size_t rows, cols;
    int fixture;
} Sources;

typedef struct {
#ifdef _WIN32
    HANDLE file, mapping;
#else
    int file;
#endif
    void *view;
    size_t bytes;
    int writable;
} MappedFile;

typedef struct {
    double minimum, maximum, off_minimum, off_maximum;
    uint64_t ones, off_ones;
} Statistics;

static int joined(const char *base, const char *name, char path[PATH_CAP])
{
    int n = snprintf(path, PATH_CAP, "%s/%s", base, name);
    return n >= 0 && (size_t)n < PATH_CAP;
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

static int parse_size(const char *text, size_t *out)
{
    unsigned long long value;
    char *end;
    if (text == NULL || text[0] == '\0' || text[0] == '-') return 0;
    errno = 0;
    value = strtoull(text, &end, 10);
    if (errno != 0 || *end != '\0' || value == 0u ||
        value > (unsigned long long)SIZE_MAX) return 0;
    *out = (size_t)value;
    return 1;
}

static int parse_options(int argc, char **argv, Options *options)
{
    int i;
    memset(options, 0, sizeof(*options));
    options->block_rows = DEFAULT_BLOCK_ROWS;
    for (i = 1; i < argc; i++) {
        const char *key = argv[i];
        const char *value;
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
        else if (strcmp(key, "--fixture-dir") == 0 &&
                 options->fixture_dir == NULL) options->fixture_dir = value;
        else if (strcmp(key, "--fixture-rows") == 0 &&
                 options->fixture_rows == 0u) {
            if (!parse_size(value, &options->fixture_rows)) return 0;
        } else if (strcmp(key, "--fixture-cols") == 0 &&
                   options->fixture_cols == 0u) {
            if (!parse_size(value, &options->fixture_cols)) return 0;
        } else if (strcmp(key, "--block-rows") == 0) {
            if (!parse_size(value, &options->block_rows)) return 0;
        } else return 0;
    }
    if (options->repository_root == NULL || options->output_dir == NULL ||
        options->repository_root[0] == '\0' || options->output_dir[0] == '\0' ||
        !lower_hex(options->producer_commit, 40u) ||
        !lower_hex(options->producer_binary_sha256, 64u)) return 0;
    if (options->fixture_dir != NULL) {
        if (options->fixture_rows == 0u || options->fixture_rows > 32u ||
            options->fixture_cols == 0u || options->fixture_cols > 256u)
            return 0;
    } else if (options->fixture_rows != 0u || options->fixture_cols != 0u)
        return 0;
    return 1;
}

static int platform_ok(void)
{
    const uint16_t one = 1u;
    return sizeof(size_t) >= 8u && sizeof(float) == 4u &&
           sizeof(double) == 8u && FLT_RADIX == 2 &&
           FLT_MANT_DIG == 24 && DBL_MANT_DIG == 53 &&
           *(const unsigned char *)&one == 1u;
}

static int exists(const char *path)
{
    struct stat info;
    return stat(path, &info) == 0;
}

static int initialize_sources(const Options *options, Sources *sources)
{
    memset(sources, 0, sizeof(*sources));
    sources->fixture = options->fixture_dir != NULL;
    sources->rows = sources->fixture ? options->fixture_rows : CANONICAL_ROWS;
    sources->cols = sources->fixture ? options->fixture_cols : CANONICAL_COLS;
    if (sources->fixture) {
        if (!joined(options->fixture_dir, "h_f32.bin", sources->h_path) ||
            !joined(options->fixture_dir, "rows.csv", sources->rows_path) ||
            !joined(options->fixture_dir, "h_spec.txt", sources->spec_path))
            return 0;
    } else {
        char h_dir[PATH_CAP], run_dir[PATH_CAP];
        if (!joined(options->repository_root, H_REL, h_dir) ||
            !joined(options->repository_root, RUN_REL, run_dir) ||
            !joined(h_dir, "h_f32.bin", sources->h_path) ||
            !joined(h_dir, "rows.csv", sources->rows_path) ||
            !joined(h_dir, "h_spec.txt", sources->spec_path) ||
            !joined(h_dir, "sha256.txt", sources->manifest_path) ||
            !joined(run_dir, "final.ckpt", sources->final_path))
            return 0;
    }
    if (!sha_file(sources->h_path, sources->h_sha) ||
        !sha_file(sources->rows_path, sources->rows_sha) ||
        !sha_file(sources->spec_path, sources->spec_sha)) return 0;
    if (sources->fixture) return 1;
    return sha_file(sources->manifest_path, sources->manifest_sha) &&
           sha_file(sources->final_path, sources->final_sha) &&
           strcmp(sources->h_sha, H_SHA) == 0 &&
           strcmp(sources->rows_sha, ROWS_SHA) == 0 &&
           strcmp(sources->spec_sha, H_SPEC_SHA) == 0 &&
           strcmp(sources->manifest_sha, H_MANIFEST_SHA) == 0 &&
           strcmp(sources->final_sha, FINAL_SHA) == 0;
}

static int sources_unchanged(const Sources *sources)
{
    char current[65];
    if (!sha_file(sources->h_path, current) ||
        strcmp(current, sources->h_sha) != 0 ||
        !sha_file(sources->rows_path, current) ||
        strcmp(current, sources->rows_sha) != 0 ||
        !sha_file(sources->spec_path, current) ||
        strcmp(current, sources->spec_sha) != 0) return 0;
    if (sources->fixture) return 1;
    return sha_file(sources->manifest_path, current) &&
           strcmp(current, sources->manifest_sha) == 0 &&
           sha_file(sources->final_path, current) &&
           strcmp(current, sources->final_sha) == 0;
}

static int validate_rows(const char *path, size_t expected_rows)
{
    FILE *file = fopen(path, "rb");
    char line[512];
    size_t row;
    int valid = 1;
    if (file == NULL) return 0;
    if (fgets(line, sizeof(line), file) == NULL ||
        strncmp(line, "manifest_index,event_index,event_id,source_split,split_pos",
            sizeof("manifest_index,event_index,event_id,source_split,split_pos") - 1u) != 0)
        valid = 0;
    for (row = 0u; valid && row < expected_rows; row++) {
        size_t manifest_index, event_index, split_pos;
        char event_id[128], split[32];
        if (fgets(line, sizeof(line), file) == NULL ||
            sscanf(line, "%zu,%zu,%127[^,],%31[^,],%zu",
                   &manifest_index, &event_index, event_id, split,
                   &split_pos) != 5 ||
            !CANONICAL_ROW_VALID(manifest_index, event_index, split,
                                 split_pos, row) ||
            event_index == 0u || event_id[0] == '\0' ||
            (strcmp(split, "train") != 0 &&
             strcmp(split, "validation") != 0)) valid = 0;
    }
    if (valid && fgets(line, sizeof(line), file) != NULL) valid = 0;
    if (ferror(file)) valid = 0;
    if (fclose(file) != 0) valid = 0;
    return valid;
}

static int load_h(const Sources *sources, float **out)
{
    FILE *file;
    float *h;
    size_t count;
    long length;
    if (sources->rows > SIZE_MAX / sources->cols) return 0;
    count = sources->rows * sources->cols;
    if (count > SIZE_MAX / sizeof(float) || count * sizeof(float) > LONG_MAX)
        return 0;
    file = fopen(sources->h_path, "rb");
    if (file == NULL) return 0;
    if (fseek(file, 0L, SEEK_END) != 0) { fclose(file); return 0; }
    length = ftell(file);
    if (length != (long)(count * sizeof(float)) ||
        fseek(file, 0L, SEEK_SET) != 0) { fclose(file); return 0; }
    h = (float *)malloc(count * sizeof(float));
    if (h == NULL) { fclose(file); return 0; }
    {
        int ok = fread(h, sizeof(float), count, file) == count && !ferror(file);
        if (fclose(file) != 0) ok = 0;
        if (!ok) { free(h); return 0; }
    }
    for (size_t i = 0u; i < count; i++) {
        if (!isfinite(h[i])) { free(h); return 0; }
    }
    *out = h;
    return 1;
}

static int copy_file(const char *source, const char *target)
{
    unsigned char buffer[65536];
    FILE *input = fopen(source, "rb"), *output;
    size_t count;
    int ok = 1;
    if (input == NULL) return 0;
    output = fopen(target, "wb");
    if (output == NULL) { fclose(input); return 0; }
    while ((count = fread(buffer, 1u, sizeof(buffer), input)) != 0u) {
        if (fwrite(buffer, 1u, count, output) != count) { ok = 0; break; }
    }
    if (ferror(input) || fflush(output) != 0) ok = 0;
    if (fclose(input) != 0) ok = 0;
    if (fclose(output) != 0) ok = 0;
    return ok;
}

static void mapped_init(MappedFile *mapped)
{
    memset(mapped, 0, sizeof(*mapped));
#ifdef _WIN32
    mapped->file = INVALID_HANDLE_VALUE;
#else
    mapped->file = -1;
#endif
}

static int mapped_open(MappedFile *mapped, const char *path,
                       size_t bytes, int writable)
{
    mapped_init(mapped);
    mapped->bytes = bytes;
    mapped->writable = writable;
#ifdef _WIN32
    LARGE_INTEGER length;
    DWORD disposition = writable ? CREATE_NEW : OPEN_EXISTING;
    mapped->file = CreateFileA(path, writable ? GENERIC_READ | GENERIC_WRITE :
        GENERIC_READ, writable ? 0u : FILE_SHARE_READ, NULL, disposition,
        FILE_ATTRIBUTE_NORMAL, NULL);
    if (mapped->file == INVALID_HANDLE_VALUE) return 0;
    length.QuadPart = (LONGLONG)bytes;
    if (writable) {
        if (!SetFilePointerEx(mapped->file, length, NULL, FILE_BEGIN) ||
            !SetEndOfFile(mapped->file)) return 0;
    } else {
        LARGE_INTEGER actual;
        if (!GetFileSizeEx(mapped->file, &actual) ||
            actual.QuadPart != length.QuadPart) return 0;
    }
    mapped->mapping = CreateFileMappingA(mapped->file, NULL,
        writable ? PAGE_READWRITE : PAGE_READONLY,
        (DWORD)((uint64_t)bytes >> 32), (DWORD)((uint64_t)bytes), NULL);
    if (mapped->mapping == NULL) return 0;
    mapped->view = MapViewOfFile(mapped->mapping,
        writable ? FILE_MAP_WRITE : FILE_MAP_READ, 0u, 0u, bytes);
#else
    mapped->file = open(path, writable ? O_RDWR | O_CREAT | O_EXCL : O_RDONLY,
                        0666);
    if (mapped->file < 0) return 0;
    if (writable) {
        if (ftruncate(mapped->file, (off_t)bytes) != 0) return 0;
    } else {
        struct stat info;
        if (fstat(mapped->file, &info) != 0 ||
            info.st_size != (off_t)bytes) return 0;
    }
    mapped->view = mmap(NULL, bytes,
        writable ? PROT_READ | PROT_WRITE : PROT_READ,
        MAP_SHARED, mapped->file, 0);
    if (mapped->view == MAP_FAILED) mapped->view = NULL;
#endif
    return mapped->view != NULL;
}

static int mapped_close(MappedFile *mapped)
{
    int ok = 1;
#ifdef _WIN32
    if (mapped->view != NULL) {
        if (mapped->writable && !FlushViewOfFile(mapped->view, 0u)) ok = 0;
        if (!UnmapViewOfFile(mapped->view)) ok = 0;
    }
    if (mapped->writable && mapped->file != INVALID_HANDLE_VALUE &&
        !FlushFileBuffers(mapped->file)) ok = 0;
    if (mapped->mapping != NULL && !CloseHandle(mapped->mapping)) ok = 0;
    if (mapped->file != INVALID_HANDLE_VALUE && !CloseHandle(mapped->file)) ok = 0;
#else
    if (mapped->view != NULL) {
        if (mapped->writable && msync(mapped->view, mapped->bytes, MS_SYNC) != 0)
            ok = 0;
        if (munmap(mapped->view, mapped->bytes) != 0) ok = 0;
    }
    if (mapped->writable && mapped->file >= 0 && fsync(mapped->file) != 0)
        ok = 0;
    if (mapped->file >= 0 && close(mapped->file) != 0) ok = 0;
#endif
    mapped_init(mapped);
    return ok;
}

static int validate_matrix(const char *path, size_t rows, size_t bytes,
                           Statistics *stats)
{
    MappedFile mapped;
    const double *a;
    size_t i, j;
    mapped_init(&mapped);
    if (!mapped_open(&mapped, path, bytes, 0)) {
        (void)mapped_close(&mapped);
        return 0;
    }
    a = (const double *)mapped.view;
    stats->minimum = 1.0;
    stats->maximum = 0.0;
    stats->off_minimum = 1.0;
    stats->off_maximum = 0.0;
    stats->ones = 0u;
    stats->off_ones = 0u;
    for (i = 0u; i < rows; i++) {
        for (j = 0u; j < rows; j++) {
            double value = a[i * rows + j];
            if (!isfinite(value) || value <= 0.0 || value > 1.0 ||
                (i == j && memcmp(&value, &(double){1.0}, sizeof(double)) != 0) ||
                (j > i && memcmp(&value, &a[j * rows + i],
                                 sizeof(double)) != 0)) {
                (void)mapped_close(&mapped);
                return 0;
            }
            if (value < stats->minimum) stats->minimum = value;
            if (value > stats->maximum) stats->maximum = value;
            if (value == 1.0) stats->ones++;
            if (i != j) {
                if (value < stats->off_minimum) stats->off_minimum = value;
                if (value > stats->off_maximum) stats->off_maximum = value;
                if (value == 1.0) stats->off_ones++;
            }
        }
    }
    return mapped_close(&mapped);
}

static int write_spec(const char *directory, const Options *options,
                      const Sources *sources, const char *binary_sha,
                      const char *payload_sha, const char *rows_sha,
                      size_t bytes, char spec_sha[65])
{
    char path[PATH_CAP];
    FILE *file;
    int ok;
    if (!joined(directory, "affinity_spec.txt", path)) return 0;
    file = fopen(path, "wb");
    if (file == NULL) return 0;
    ok = fprintf(file,
        "format=VAE_C_IIC_AFFINITY_ARTIFACT_V1\n"
        "source_mode=%s\nrows=%zu\ncols=%zu\n"
        "dtype=float64_ieee754\nendian=little\nlayout=row_major\n"
        "payload_file=affinity_f64.bin\npayload_bytes=%zu\n"
        "payload_sha256=%s\nrow_map_file=rows.csv\nrow_map_sha256=%s\n"
        "formula=exp(-euclidean_l2(H_i,H_j))\n"
        "input_conversion=float32_to_float64_before_subtraction\n"
        "diagonal=exact_1.0\nsymmetry=upper_once_lower_exact_copy\n"
        "gamma=none\nnormalization=none\nclipping=none\n"
        "h_payload_sha256=%s\nh_row_map_sha256=%s\nh_metadata_sha256=%s\n"
        "iic_run_id=%s\niic_final_sha256=%s\n"
        "affinity_core_commit=%s\nproducer_commit=%s\n"
        "producer_binary_sha256=%s\nblock_rows=%zu\n"
        "notebook06_reference_sha256=%s\nlabel_free=true\n"
        "checksum_file=sha256.txt\nchecksum_manifest_excludes_self=true\n",
        sources->fixture ? "test_fixture" : SOURCE_MODE,
        sources->rows, sources->rows, bytes, payload_sha, rows_sha,
        sources->h_sha, sources->rows_sha, sources->spec_sha,
        sources->fixture ? "test_fixture" : RUN_ID,
        sources->fixture ? "not_applicable" : sources->final_sha,
        CORE_COMMIT, options->producer_commit, binary_sha,
        options->block_rows, NOTEBOOK_SHA) >= 0;
    if (fflush(file) != 0) ok = 0;
    if (fclose(file) != 0) ok = 0;
    return ok && sha_file(path, spec_sha);
}

static int write_manifest(const char *directory,
                          const char *payload_sha, const char *rows_sha,
                          const char *spec_sha)
{
    char path[PATH_CAP];
    FILE *file;
    int ok;
    if (!joined(directory, "sha256.txt", path)) return 0;
    file = fopen(path, "wb");
    if (file == NULL) return 0;
    ok = fprintf(file, "%s  affinity_f64.bin\n%s  rows.csv\n"
                       "%s  affinity_spec.txt\n",
                 payload_sha, rows_sha, spec_sha) >= 0;
    if (fflush(file) != 0) ok = 0;
    if (fclose(file) != 0) ok = 0;
    return ok;
}

static void clean_own_partial(const char *directory)
{
    const char *names[] = {"affinity_f64.bin", "rows.csv",
                           "affinity_spec.txt", "sha256.txt"};
    char path[PATH_CAP];
    size_t i;
    for (i = 0u; i < 4u; i++) {
        if (joined(directory, names[i], path)) (void)remove(path);
    }
#ifdef _WIN32
    (void)_rmdir(directory);
#else
    (void)rmdir(directory);
#endif
}

static int publish(const char *partial, const char *final)
{
    if (exists(final)) return 0;
#ifdef _WIN32
    return MoveFileExA(partial, final, MOVEFILE_WRITE_THROUGH) != 0;
#else
    return rename(partial, final) == 0;
#endif
}

static int parent_free_space(const char *output_dir, uint64_t required,
                             uint64_t *free_bytes)
{
    char parent[PATH_CAP];
    char *slash, *backslash, *last;
    int n = snprintf(parent, sizeof(parent), "%s", output_dir);
    if (n < 0 || (size_t)n >= sizeof(parent)) return 0;
    slash = strrchr(parent, '/');
    backslash = strrchr(parent, '\\');
    if (slash == NULL) last = backslash;
    else if (backslash == NULL) last = slash;
    else last = slash > backslash ? slash : backslash;
    if (last == NULL) return 0;
    *last = '\0';
#ifdef _WIN32
    ULARGE_INTEGER available;
    if (!GetDiskFreeSpaceExA(parent, &available, NULL, NULL)) return 0;
    *free_bytes = (uint64_t)available.QuadPart;
#else
    (void)required;
    *free_bytes = UINT64_MAX;
#endif
    return *free_bytes >= required;
}

int main(int argc, char **argv)
{
    Options options;
    Sources sources;
    MappedFile mapped;
    Statistics stats;
    float *h = NULL;
    char partial[PATH_CAP], payload_path[PATH_CAP], rows_path[PATH_CAP];
    char own_binary_sha[65], payload_sha[65], rows_sha[65], spec_sha[65];
    size_t elements, bytes, start;
    uint64_t free_bytes = 0u, required;
    int partial_owned = 0, status = 1;
    mapped_init(&mapped);
    if (!parse_options(argc, argv, &options) || !platform_ok() ||
        options.block_rows == 0u ||
        options.block_rows > (options.fixture_dir ? options.fixture_rows :
                              CANONICAL_ROWS) ||
        snprintf(partial, sizeof(partial), "%s.partial", options.output_dir) < 0 ||
        strlen(options.output_dir) + strlen(".partial") >= sizeof(partial) ||
        exists(options.output_dir) || exists(partial) ||
        !sha_file(argv[0], own_binary_sha) ||
        strcmp(own_binary_sha, options.producer_binary_sha256) != 0 ||
        !initialize_sources(&options, &sources) ||
        !validate_rows(sources.rows_path, sources.rows) ||
        sources.rows > SIZE_MAX / sources.rows ||
        sources.rows * sources.rows > SIZE_MAX / sizeof(double) ||
        !sources_unchanged(&sources)) {
        fputs("affinity preflight failed\n", stderr);
        goto done;
    }
    elements = sources.rows * sources.rows;
    bytes = elements * sizeof(double);
    if (bytes > UINT64_MAX / 2u - UINT64_C(67108864)) {
        fputs("affinity disk requirement overflow\n", stderr);
        goto done;
    }
    required = (uint64_t)bytes * 2u + UINT64_C(67108864);
    if (!parent_free_space(options.output_dir, required, &free_bytes) ||
        !load_h(&sources, &h)) {
        fputs("affinity source/disk preflight failed\n", stderr);
        goto done;
    }
    if (options.validate_only) {
        printf("AFFINITY_INPUT_VALIDATED=true\nAFFINITY_COMPUTE_CALLS=0\n"
               "OUTPUT_CREATED=false\nFREE_BYTES=%llu\n",
               (unsigned long long)free_bytes);
        status = 0;
        goto done;
    }
#ifdef _WIN32
    if (_mkdir(partial) != 0) goto done;
#else
    if (mkdir(partial, 0777) != 0) goto done;
#endif
    partial_owned = 1;
    if (!joined(partial, "affinity_f64.bin", payload_path) ||
        !joined(partial, "rows.csv", rows_path) ||
        !mapped_open(&mapped, payload_path, bytes, 1)) {
        fputs("affinity output mapping failed\n", stderr);
        goto done;
    }
    for (start = 0u; start < sources.rows; start += options.block_rows) {
        size_t count = sources.rows - start;
        if (count > options.block_rows) count = options.block_rows;
        if (vae_c_iic_affinity_compute_rows(h, sources.rows, sources.cols,
                start, count, (double *)mapped.view, elements) != VAE_C_OP_OK) {
            fputs("affinity numerical core failed\n", stderr);
            goto done;
        }
    }
    if (!mapped_close(&mapped) ||
        !validate_matrix(payload_path, sources.rows, bytes, &stats) ||
        !sha_file(payload_path, payload_sha) ||
        !copy_file(sources.rows_path, rows_path) ||
        !sha_file(rows_path, rows_sha) ||
        strcmp(rows_sha, sources.rows_sha) != 0 ||
        !validate_rows(rows_path, sources.rows) ||
        !write_spec(partial, &options, &sources, own_binary_sha,
                    payload_sha, rows_sha, bytes, spec_sha) ||
        !write_manifest(partial, payload_sha, rows_sha, spec_sha) ||
        !sources_unchanged(&sources) || exists(options.output_dir) ||
        !publish(partial, options.output_dir)) {
        fputs("affinity audit/publish failed\n", stderr);
        goto done;
    }
    partial_owned = 0;
    printf("AFFINITY_CREATED=true\nROWS=%zu\nCOLS=%zu\n"
           "PAYLOAD_BYTES=%zu\nPAYLOAD_SHA256=%s\n"
           "ROW_MAP_SHA256=%s\nMETADATA_SHA256=%s\n"
           "MIN=%.17g\nMAX=%.17g\nOFFDIAG_MIN=%.17g\n"
           "OFFDIAG_MAX=%.17g\nONES=%llu\nOFFDIAG_ONES=%llu\n",
           sources.rows, sources.rows, bytes, payload_sha, rows_sha,
           spec_sha, stats.minimum, stats.maximum, stats.off_minimum,
           stats.off_maximum, (unsigned long long)stats.ones,
           (unsigned long long)stats.off_ones);
    status = 0;
done:
    if (mapped.view != NULL ||
#ifdef _WIN32
        mapped.file != INVALID_HANDLE_VALUE
#else
        mapped.file >= 0
#endif
       ) (void)mapped_close(&mapped);
    if (partial_owned) clean_own_partial(partial);
    free(h);
    return status;
}
