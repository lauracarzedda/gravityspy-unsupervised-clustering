#define _CRT_SECURE_NO_WARNINGS

#include "baseline_kmeans.h"
#include "vae_c_mu_contract.h"

#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <locale.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#include <windows.h>
#else
#include <dirent.h>
#include <unistd.h>
#endif


enum {
    KMU_FILE_COUNT = 6,
    KMU_CHECKSUM_ENTRY_COUNT = 5,
    KMU_PATH_CAP = 4096,
    KMU_LINE_CAP = 1024,
    KMU_EVENT_ID_CAP = 128,
    KMU_DATASET_ROWS = 8583,
    KMU_FULL_TRAIN_ROWS = 6866,
    KMU_FULL_VALIDATION_ROWS = 1717,
    KMU_MATCHED_TRAIN_ROWS = 800,
    KMU_MATCHED_VALIDATION_ROWS = 200,
    KMU_SEED = 123,
    KMU_MAX_ITERATIONS = 300,
    KMU_OUTPUT_FILE_COUNT = 13,
    KMU_OUTPUT_ARTIFACT_COUNT = 12,
    KMU_CODE_FILE_COUNT = 3
};


#define KMU_TRAIN_SPLIT_SHA256 \
    "8c7a513b98dbe996261bc80965500ab15ab77f498ec9440858003152e42f926f"
#define KMU_VALIDATION_SPLIT_SHA256 \
    "e63109c88df7b3ee97317f94791f26df3fcaf8b937921cfb27967d01c2546467"
#define KMU_TRAIN_SUBSET_SHA256 \
    "80c5396a70e7f2c8e3a00680582945581ae76c301a5b9ba8c62e006bb25cdfb5"
#define KMU_VALIDATION_SUBSET_SHA256 \
    "9f5fe6c50c7398588f2de7bd4a543a54a53e7f7a0b9a84bf4004a01f065ae76f"
#define KMU_RUNNER_BASE_COMMIT \
    "7f700f648fa465697b83bcfab553713435f7eb42"
#define KMU_KMEANS_SOURCE_SHA256 \
    "987ac13f792fe21139baa10c02e4356147af83bc34203c0090ac3379e58a2eee"
#define KMU_KMEANS_HEADER_SHA256 \
    "5d60d0621b3866ce74e3fccf68361917f4374c74fe2a2ed58450c9978c2f61c8"


typedef enum {
    KMU_SCHEMA_MINI_V1 = 0,
    KMU_SCHEMA_FULL224_V1
} KMUSchema;


typedef enum {
    KMU_SCOPE_NONE = 0,
    KMU_SCOPE_FULL,
    KMU_SCOPE_MATCHED_SUBSET1000
} KMUScope;


typedef struct {
    const char *token;
    const char *profile_id;
    const char *relative_directory;
    KMUSchema schema;
    size_t train_rows;
    size_t validation_rows;
    size_t columns;
    const char *names[KMU_FILE_COUNT];
    const char *hashes[KMU_FILE_COUNT];
} KMUSourceDescriptor;


typedef struct {
    uint64_t split_pos;
    uint64_t manifest_index;
    uint64_t event_index;
    char event_id[KMU_EVENT_ID_CAP];
    char split[16];
} KMURowIdentity;


typedef struct {
    KMURowIdentity *train;
    KMURowIdentity *validation;
    size_t train_count;
    size_t validation_count;
} KMURowMaps;


typedef struct {
    KMUScope scope;
    const KMUSourceDescriptor *source;
    size_t k;
    uint64_t seed;
    size_t max_iterations;
    const char *repository_root;
    const char *full224_bundle;
    const char *output_root;
    const char *code_commit;
} KMURequest;


typedef struct {
    double initial_inertia;
    double train_inertia;
    double validation_inertia;
    size_t empty_final_clusters;
    size_t minimum_cluster_size;
    size_t maximum_cluster_size;
} KMUNumericSummary;


typedef struct {
    char runner[65];
    char core_source[65];
    char core_header[65];
} KMUCodeHashes;


typedef struct {
    size_t fit_calls;
    size_t assign_calls;
    int converged;
    int output_created;
} KMURunReport;


enum {
    KMU_METADATA_FILE = 0,
    KMU_CHECKSUM_FILE = 1,
    KMU_TRAIN_MATRIX_FILE = 2,
    KMU_VALIDATION_MATRIX_FILE = 3,
    KMU_TRAIN_ROWS_FILE = 4,
    KMU_VALIDATION_ROWS_FILE = 5
};


static const char *const kmu_output_names[KMU_OUTPUT_FILE_COUNT] = {
    "centroids_f64.bin",
    "train_assignments_u32.bin",
    "validation_assignments_u32.bin",
    "train_squared_distances_f64.bin",
    "validation_squared_distances_f64.bin",
    "cluster_sizes_train_u64.bin",
    "initial_centroid_indices_u64.bin",
    "train_assignments.csv",
    "validation_assignments.csv",
    "cluster_sizes.csv",
    "initial_centroids.csv",
    "metadata.txt",
    "sha256.txt"
};


static const KMUSourceDescriptor kmu_sources[] = {
    {
        "mini75",
        "mini75_experiment_v1",
        "sakai_multires_c/results/vae_c/embeddings/"
        "mini75_experiment_v1_full_best_epoch47_posterior_mu_central_v1",
        KMU_SCHEMA_MINI_V1,
        KMU_FULL_TRAIN_ROWS,
        KMU_FULL_VALIDATION_ROWS,
        32u,
        {
            "embedding_spec.txt", "sha256.txt", "train_mu_f32.bin",
            "validation_mu_f32.bin", "train_rows.csv",
            "validation_rows.csv"
        },
        {
            "16e5f356ebd7bd961d349a1844a555e9a8358eded735ad6f676cfc401e3aaabf",
            "a60e1533506ce16dd770ce4f9765f33ad44ea717ad25469dfad6d77910c7df17",
            "a8e82c3a0d25ffbd649640e1df01438fe97fc070ebb0def50227c5021fa52d52",
            "548596df86d7189a05ce3fb84631f403383546ba4ab1a44119c13852447b13d7",
            "3999c0ae72a0011b9433249dc335fc95ad159b4c84ec941012fd77e8f7f692c1",
            "fe62ecfba8a723938a54caf95e5c5daf2a2ca5380acaec0f208599564da7a222"
        }
    },
    {
        "mini112",
        "mini112_primary_v1",
        "sakai_multires_c/results/vae_c/embeddings/"
        "mini112_primary_v1_full_best_epoch97_posterior_mu_central_v1",
        KMU_SCHEMA_MINI_V1,
        KMU_FULL_TRAIN_ROWS,
        KMU_FULL_VALIDATION_ROWS,
        64u,
        {
            "embedding_spec.txt", "sha256.txt", "train_mu_f32.bin",
            "validation_mu_f32.bin", "train_rows.csv",
            "validation_rows.csv"
        },
        {
            "8a6969853197e32a5998b7e71011d689e9e67e5dcb0144592f626a4652f27243",
            "3dbefbbdc71f80347bf18927aa81bf46e037323de069076798c77dea635d3118",
            "1399f4658da79b368fa03dd130030511865d2c121f93dcda590dec5d2eede775",
            "1add3b9e6f9ca548856119f0b6e49a7dd1e706a59e62dcd6760f547a86022be7",
            "3999c0ae72a0011b9433249dc335fc95ad159b4c84ec941012fd77e8f7f692c1",
            "fe62ecfba8a723938a54caf95e5c5daf2a2ca5380acaec0f208599564da7a222"
        }
    },
    {
        "mini224",
        "mini224_bridge_v1",
        "sakai_multires_c/results/vae_c/embeddings/"
        "mini224_bridge_v1_subset1000_best_epoch99_posterior_mu_central_v1",
        KMU_SCHEMA_MINI_V1,
        KMU_MATCHED_TRAIN_ROWS,
        KMU_MATCHED_VALIDATION_ROWS,
        64u,
        {
            "embedding_spec.txt", "sha256.txt", "train_mu_f32.bin",
            "validation_mu_f32.bin", "train_rows.csv",
            "validation_rows.csv"
        },
        {
            "d3c67e9efcfedcb5b620893cfb07ec225505653e42267986dcb030bcbcb89ef6",
            "aa55332307b89ec902ffd217467cdd58870728f382ad8956045cf91878ed8c95",
            "f494b232cb2fbf9ca4fb25df80c47e5f80963ac1d7448afd807361409a4ab166",
            "5f6aec1737549a24d20897581b4c6b57b3fae6859e1dc1e972a8b8841ede4617",
            "fd5bb3c48f87e98d5ec8ed0e561d5fe077ab96a7effa0c7425a41a222cd4375c",
            "87a3c7fc715fe0f4a6158474e46f6903d5267a45741d338836e8fecd61df7ed8"
        }
    },
    {
        "full224",
        "sakai224_full_v1",
        NULL,
        KMU_SCHEMA_FULL224_V1,
        KMU_FULL_TRAIN_ROWS,
        KMU_FULL_VALIDATION_ROWS,
        512u,
        {
            "metadata.json", "SHA256SUMS.txt",
            "embeddings_mu_train_f32.bin",
            "embeddings_mu_validation_f32.bin", "row_map_train.csv",
            "row_map_validation.csv"
        },
        {
            "075883af82276431269e57a064da7fcdbd40e402fe8d58b8b1a3c63250a8c1e1",
            "2d6205547e6a063b49178595de9a24918bd0add1a4586c800e7519bf842f951d",
            "b620fd9711719a7bd40293b6536ea6feb235cbadb1d1e37dca0e46651332ad5e",
            "c1692b3e7a4c86a7f4d6d7172117e11083d2bd2394947838a24d1e2b81f27cf9",
            "4b25c7f41a945fc8195e93c11e60da3b2e5b8be8298f6919008fc57dc39cc9e8",
            "f5a0e48c79db4d466f834cbc54bdbf7e8c34d3fa45f545f58b37c71627c97b48"
        }
    }
};


static int kmu_error(const char *format, ...)
{
    va_list arguments;

    fputs("ERROR: ", stderr);
    va_start(arguments, format);
    vfprintf(stderr, format, arguments);
    va_end(arguments);
    fputc('\n', stderr);
    return 0;
}


static int kmu_copy_text(char *target, size_t capacity, const char *source)
{
    size_t length;

    if (target == NULL || capacity == 0u || source == NULL) {
        return 0;
    }
    length = strlen(source);
    if (length >= capacity) {
        return 0;
    }
    memcpy(target, source, length + 1u);
    return 1;
}


static int kmu_path_join(
    const char *base,
    const char *name,
    char output[KMU_PATH_CAP]
)
{
    size_t base_length;
    int written;

    if (base == NULL || name == NULL || output == NULL || base[0] == '\0' ||
        name[0] == '\0') {
        return 0;
    }
    base_length = strlen(base);
    written = snprintf(
        output,
        KMU_PATH_CAP,
        "%s%s%s",
        base,
        (base[base_length - 1u] == '/' || base[base_length - 1u] == '\\')
            ? "" : "/",
        name
    );
    return written >= 0 && (size_t)written < KMU_PATH_CAP;
}


static int kmu_file_index(
    const KMUSourceDescriptor *source,
    const char *name
)
{
    size_t index;

    for (index = 0u; index < KMU_FILE_COUNT; index++) {
        if (strcmp(source->names[index], name) == 0) {
            return (int)index;
        }
    }
    return -1;
}


static int kmu_directory_has_exact_inventory(
    const char *directory,
    const KMUSourceDescriptor *source
)
{
    unsigned char seen[KMU_FILE_COUNT] = {0};
    size_t count = 0u;

#ifdef _WIN32
    WIN32_FIND_DATAA entry;
    HANDLE search;
    DWORD find_error;
    char pattern[KMU_PATH_CAP];

    if (!kmu_path_join(directory, "*", pattern)) {
        return kmu_error("artifact inventory path overflow");
    }
    search = FindFirstFileA(pattern, &entry);
    if (search == INVALID_HANDLE_VALUE) {
        return kmu_error("cannot enumerate artifact directory: %s", directory);
    }
    do {
        int index;
        if (strcmp(entry.cFileName, ".") == 0 ||
            strcmp(entry.cFileName, "..") == 0) {
            continue;
        }
        index = kmu_file_index(source, entry.cFileName);
        if (index < 0 || seen[index] != 0u ||
            (entry.dwFileAttributes &
             (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)) != 0u) {
            FindClose(search);
            return kmu_error("unexpected artifact entry: %s", entry.cFileName);
        }
        seen[index] = 1u;
        count++;
    } while (FindNextFileA(search, &entry) != 0);
    find_error = GetLastError();
    if (FindClose(search) == 0 || find_error != ERROR_NO_MORE_FILES) {
        return kmu_error("cannot complete artifact directory inventory");
    }
#else
    DIR *stream = opendir(directory);
    struct dirent *entry;

    if (stream == NULL) {
        return kmu_error("cannot enumerate artifact directory: %s", directory);
    }
    for (;;) {
        int index;
        char path[KMU_PATH_CAP];
        struct stat status;
        errno = 0;
        entry = readdir(stream);
        if (entry == NULL) {
            if (errno != 0) {
                closedir(stream);
                return kmu_error(
                    "cannot complete artifact directory inventory"
                );
            }
            break;
        }
        if (strcmp(entry->d_name, ".") == 0 ||
            strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        index = kmu_file_index(source, entry->d_name);
        if (index < 0 || seen[index] != 0u ||
            !kmu_path_join(directory, entry->d_name, path) ||
            stat(path, &status) != 0 || !S_ISREG(status.st_mode)) {
            closedir(stream);
            return kmu_error("unexpected artifact entry: %s", entry->d_name);
        }
        seen[index] = 1u;
        count++;
    }
    if (closedir(stream) != 0) {
        return kmu_error("cannot close artifact directory inventory");
    }
#endif

    if (count != KMU_FILE_COUNT) {
        return kmu_error("artifact inventory has %zu files, expected %d",
                         count, KMU_FILE_COUNT);
    }
    return 1;
}


static int kmu_hash_is_lower_hex(const char *hash)
{
    size_t index;

    if (hash == NULL || strlen(hash) != 64u) {
        return 0;
    }
    for (index = 0u; index < 64u; index++) {
        if (!((hash[index] >= '0' && hash[index] <= '9') ||
              (hash[index] >= 'a' && hash[index] <= 'f'))) {
            return 0;
        }
    }
    return 1;
}


static int kmu_verify_file_hash(
    const char *directory,
    const KMUSourceDescriptor *source,
    size_t file_index
)
{
    char path[KMU_PATH_CAP];
    char actual[VAE_C_CHECKPOINT_SHA256_CAP];

    if (!kmu_hash_is_lower_hex(source->hashes[file_index]) ||
        !kmu_path_join(directory, source->names[file_index], path) ||
        vae_c_mu_file_sha256(path, actual) != VAE_C_MU_OK ||
        strcmp(actual, source->hashes[file_index]) != 0) {
        return kmu_error("SHA-256 mismatch: %s", source->names[file_index]);
    }
    return 1;
}


static int kmu_strip_line(char *line)
{
    size_t length;

    if (line == NULL) {
        return 0;
    }
    length = strlen(line);
    while (length > 0u &&
           (line[length - 1u] == '\n' || line[length - 1u] == '\r')) {
        line[--length] = '\0';
    }
    return 1;
}


static int kmu_validate_checksum_manifest(
    const char *directory,
    const KMUSourceDescriptor *source
)
{
    char path[KMU_PATH_CAP];
    char line[KMU_LINE_CAP];
    unsigned char seen[KMU_FILE_COUNT] = {0};
    size_t count = 0u;
    FILE *file = NULL;
    int failed;

    if (!kmu_path_join(directory, source->names[KMU_CHECKSUM_FILE], path)) {
        return kmu_error("checksum path overflow");
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        return kmu_error("cannot open checksum manifest");
    }
    while (fgets(line, sizeof(line), file) != NULL) {
        int index;
        size_t position;

        if (strchr(line, '\n') == NULL && !feof(file)) {
            fclose(file);
            return kmu_error("checksum line is too long");
        }
        kmu_strip_line(line);
        if (strlen(line) < 67u || line[64] != ' ' || line[65] != ' ') {
            fclose(file);
            return kmu_error("malformed checksum line");
        }
        for (position = 0u; position < 64u; position++) {
            if (!((line[position] >= '0' && line[position] <= '9') ||
                  (line[position] >= 'a' && line[position] <= 'f'))) {
                fclose(file);
                return kmu_error("non-canonical checksum text");
            }
        }
        line[64] = '\0';
        index = kmu_file_index(source, line + 66u);
        if (index < 0 || index == KMU_CHECKSUM_FILE || seen[index] != 0u ||
            strcmp(line, source->hashes[index]) != 0) {
            fclose(file);
            return kmu_error("unexpected checksum manifest entry");
        }
        seen[index] = 1u;
        count++;
    }
    failed = ferror(file) || count != KMU_CHECKSUM_ENTRY_COUNT;
    if (fclose(file) != 0) {
        failed = 1;
    }
    if (failed) {
        return kmu_error("checksum manifest inventory mismatch");
    }
    for (count = 0u; count < KMU_FILE_COUNT; count++) {
        if (count != KMU_CHECKSUM_FILE && seen[count] == 0u) {
            return kmu_error("checksum manifest omits an artifact file");
        }
    }
    return 1;
}


static int kmu_file_has_exact_line(const char *path, const char *expected)
{
    char line[KMU_LINE_CAP];
    FILE *file = fopen(path, "rb");

    if (file == NULL) {
        return 0;
    }
    while (fgets(line, sizeof(line), file) != NULL) {
        if (strchr(line, '\n') == NULL && !feof(file)) {
            fclose(file);
            return 0;
        }
        kmu_strip_line(line);
        if (strcmp(line, expected) == 0) {
            fclose(file);
            return 1;
        }
    }
    fclose(file);
    return 0;
}


static int kmu_file_contains(const char *path, const char *expected)
{
    char buffer[8192];
    size_t count;
    FILE *file = fopen(path, "rb");
    int failed;

    if (file == NULL) {
        return 0;
    }
    count = fread(buffer, 1u, sizeof(buffer) - 1u, file);
    failed = ferror(file) || !feof(file);
    if (fclose(file) != 0) {
        failed = 1;
    }
    if (failed) {
        return 0;
    }
    buffer[count] = '\0';
    return strstr(buffer, expected) != NULL;
}


static int kmu_expect_spec_line(
    const char *path,
    const char *key,
    const char *value
)
{
    char expected[KMU_LINE_CAP];
    int written = snprintf(expected, sizeof(expected), "%s=%s", key, value);

    return written >= 0 && (size_t)written < sizeof(expected) &&
        kmu_file_has_exact_line(path, expected);
}


static int kmu_validate_metadata(
    const char *directory,
    const KMUSourceDescriptor *source
)
{
    char path[KMU_PATH_CAP];
    char value[128];

    if (!kmu_path_join(directory, source->names[KMU_METADATA_FILE], path)) {
        return kmu_error("metadata path overflow");
    }
    if (source->schema == KMU_SCHEMA_MINI_V1) {
        snprintf(value, sizeof(value), "%zu", source->columns);
        if (!kmu_expect_spec_line(path, "format", "VAE_C_MU_ARTIFACT_V1") ||
            !kmu_expect_spec_line(path, "representation", "posterior_mu") ||
            !kmu_expect_spec_line(path, "view", "central") ||
            !kmu_expect_spec_line(path, "profile_id", source->profile_id) ||
            !kmu_expect_spec_line(path, "latent_dim", value) ||
            !kmu_expect_spec_line(path, "dtype", "float32_ieee754") ||
            !kmu_expect_spec_line(path, "endian", "little") ||
            !kmu_expect_spec_line(path, "layout", "row_major_[N,latent_dim]") ||
            !kmu_expect_spec_line(path, "latent_scaling", "none") ||
            !kmu_expect_spec_line(path, "latent_normalization", "none") ||
            !kmu_expect_spec_line(path, "pca", "false") ||
            !kmu_expect_spec_line(path, "labels_used", "false") ||
            !kmu_expect_spec_line(
                path, "label_dependent_selections", "false"
            ) ||
            !kmu_expect_spec_line(path, "label_values_exported", "false") ||
            !kmu_expect_spec_line(
                path, "row_map_schema",
                "row_index,split_pos,manifest_index,event_index,event_id,split"
            )) {
            return kmu_error("mini mu embedding format mismatch");
        }
        snprintf(value, sizeof(value), "%zu", source->train_rows);
        if (!kmu_expect_spec_line(path, "train_rows", value)) {
            return kmu_error("mini train row count metadata mismatch");
        }
        snprintf(value, sizeof(value), "%zu", source->validation_rows);
        if (!kmu_expect_spec_line(path, "validation_rows", value)) {
            return kmu_error("mini validation row count metadata mismatch");
        }
        snprintf(value, sizeof(value), "%zu", source->columns);
        if (!kmu_expect_spec_line(path, "train_cols", value) ||
            !kmu_expect_spec_line(path, "validation_cols", value)) {
            return kmu_error("mini mu column count metadata mismatch");
        }
        return 1;
    }

    if (!kmu_file_contains(path, "\"artifact_format_version\": 1") ||
        !kmu_file_contains(path, "\"artifact_id\": \"sakai224_full_mu_epoch100_v1\"") ||
        !kmu_file_contains(path, "\"canonical_feature\": \"mu\"") ||
        !kmu_file_contains(path, "\"dtype\": \"float32\"") ||
        !kmu_file_contains(path, "\"endianness\": \"little\"") ||
        !kmu_file_contains(path, "\"matrix_layout\": \"row_major\"") ||
        !kmu_file_contains(path, "\"latent_dim\": 512") ||
        !kmu_file_contains(path, "\"values_per_row\": 512") ||
        !kmu_file_contains(path, "\"train_rows\": 6866") ||
        !kmu_file_contains(path, "\"validation_rows\": 1717") ||
        !kmu_file_contains(path, "\"pca_applied\": false") ||
        !kmu_file_contains(path, "\"standardization_applied\": false") ||
        !kmu_file_contains(path, "\"labels_in_embedding_artifact\": false") ||
        !kmu_file_contains(path, "\"train_binary\": \"embeddings_mu_train_f32.bin\"") ||
        !kmu_file_contains(path, "\"validation_binary\": \"embeddings_mu_validation_f32.bin\"") ||
        !kmu_file_contains(path, "\"train_row_map\": \"row_map_train.csv\"") ||
        !kmu_file_contains(path, "\"validation_row_map\": \"row_map_validation.csv\"")) {
        return kmu_error("full224 mu metadata format mismatch");
    }
    return 1;
}


static int kmu_platform_is_f32_little_endian(void)
{
    float one = 1.0f;
    const unsigned char *bytes = (const unsigned char *)&one;

    return sizeof(float) == 4u && FLT_RADIX == 2 &&
        bytes[0] == 0x00u && bytes[1] == 0x00u &&
        bytes[2] == 0x80u && bytes[3] == 0x3fu;
}


static int kmu_expected_matrix_bytes(
    size_t rows,
    size_t columns,
    uint64_t *bytes
)
{
    if (rows == 0u || columns == 0u || rows > SIZE_MAX / columns ||
        rows * columns > SIZE_MAX / sizeof(float)) {
        return 0;
    }
    *bytes = (uint64_t)(rows * columns * sizeof(float));
    return 1;
}


static int kmu_validate_finite_matrix(
    const char *directory,
    const char *name,
    size_t rows,
    size_t columns
)
{
    float values[4096];
    size_t expected_values;
    size_t total = 0u;
    uint64_t expected_bytes;
    char path[KMU_PATH_CAP];
    struct stat status;
    FILE *file;
    int failed;

    if (!kmu_platform_is_f32_little_endian() ||
        !kmu_expected_matrix_bytes(rows, columns, &expected_bytes) ||
        !kmu_path_join(directory, name, path) || stat(path, &status) != 0 ||
        status.st_size < 0 || (uint64_t)status.st_size != expected_bytes) {
        return kmu_error("matrix byte contract mismatch: %s", name);
    }
    expected_values = rows * columns;
    file = fopen(path, "rb");
    if (file == NULL) {
        return kmu_error("cannot open matrix: %s", name);
    }
    while (total < expected_values) {
        size_t requested = expected_values - total;
        size_t read_count;
        size_t index;
        if (requested > sizeof(values) / sizeof(values[0])) {
            requested = sizeof(values) / sizeof(values[0]);
        }
        read_count = fread(values, sizeof(float), requested, file);
        if (read_count != requested) {
            fclose(file);
            return kmu_error("short matrix read: %s", name);
        }
        for (index = 0u; index < read_count; index++) {
            if (!isfinite(values[index])) {
                fclose(file);
                return kmu_error("non-finite mu value: %s", name);
            }
        }
        total += read_count;
    }
    failed = fgetc(file) != EOF || ferror(file);
    if (fclose(file) != 0) {
        failed = 1;
    }
    if (failed) {
        return kmu_error("matrix trailing data or read error: %s", name);
    }
    return 1;
}


static int kmu_parse_u64(const char *text, uint64_t *value)
{
    char *end = NULL;
    const unsigned char *cursor;
    uintmax_t parsed;

    if (text == NULL || value == NULL || text[0] == '\0') {
        return 0;
    }
    for (cursor = (const unsigned char *)text; *cursor != '\0'; cursor++) {
        if (!isdigit(*cursor)) {
            return 0;
        }
    }
    errno = 0;
    parsed = strtoumax(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' ||
        parsed > UINT64_MAX) {
        return 0;
    }
    *value = (uint64_t)parsed;
    return 1;
}


static size_t kmu_split_csv(char *line, char **fields, size_t capacity)
{
    size_t count = 0u;
    char *read_cursor = line;
    char *write_cursor = line;

    if (line == NULL || fields == NULL || capacity == 0u) {
        return 0u;
    }
    for (;;) {
        int quoted = 0;
        int closed_quote = 0;

        if (count >= capacity) {
            return 0u;
        }
        fields[count++] = write_cursor;
        if (*read_cursor == '"') {
            quoted = 1;
            read_cursor++;
        }
        while (*read_cursor != '\0') {
            if (quoted && *read_cursor == '"') {
                if (read_cursor[1] == '"') {
                    *write_cursor++ = '"';
                    read_cursor += 2;
                    continue;
                }
                read_cursor++;
                closed_quote = 1;
                break;
            }
            if (!quoted && *read_cursor == ',') {
                break;
            }
            if (!quoted && *read_cursor == '"') {
                return 0u;
            }
            *write_cursor++ = *read_cursor++;
        }
        if (quoted && !closed_quote) {
            return 0u;
        }
        if (closed_quote && *read_cursor != '\0' && *read_cursor != ',') {
            return 0u;
        }
        if (*read_cursor == ',') {
            read_cursor++;
            *write_cursor++ = '\0';
            continue;
        }
        *write_cursor = '\0';
        break;
    }
    return count;
}


static int kmu_event_id_valid(const char *event_id)
{
    size_t index;
    size_t length;

    if (event_id == NULL) {
        return 0;
    }
    length = strlen(event_id);
    if (length == 0u || length >= KMU_EVENT_ID_CAP) {
        return 0;
    }
    for (index = 0u; index < length; index++) {
        unsigned char value = (unsigned char)event_id[index];
        if (!(isalnum(value) || value == '_')) {
            return 0;
        }
    }
    return 1;
}


static int kmu_parse_row(
    char *line,
    KMUSchema schema,
    const char *expected_split,
    size_t expected_position,
    KMURowIdentity *row
)
{
    char *fields[6];
    size_t field_count = kmu_split_csv(line, fields, 6u);
    uint64_t row_index = 0u;

    memset(row, 0, sizeof(*row));
    if (schema == KMU_SCHEMA_MINI_V1) {
        if (field_count != 6u || !kmu_parse_u64(fields[0], &row_index) ||
            !kmu_parse_u64(fields[1], &row->split_pos) ||
            !kmu_parse_u64(fields[2], &row->manifest_index) ||
            !kmu_parse_u64(fields[3], &row->event_index) ||
            row_index != (uint64_t)expected_position ||
            strcmp(fields[5], expected_split) != 0 ||
            !kmu_copy_text(row->split, sizeof(row->split), fields[5]) ||
            !kmu_event_id_valid(fields[4]) ||
            !kmu_copy_text(row->event_id, sizeof(row->event_id), fields[4])) {
            return 0;
        }
    } else {
        if (field_count != 5u || strcmp(fields[0], expected_split) != 0 ||
            !kmu_parse_u64(fields[1], &row->split_pos) ||
            !kmu_parse_u64(fields[2], &row->manifest_index) ||
            !kmu_parse_u64(fields[3], &row->event_index) ||
            !kmu_copy_text(row->split, sizeof(row->split), fields[0]) ||
            !kmu_event_id_valid(fields[4]) ||
            !kmu_copy_text(row->event_id, sizeof(row->event_id), fields[4])) {
            return 0;
        }
    }
    return row->split_pos == (uint64_t)expected_position &&
        row->manifest_index < KMU_DATASET_ROWS &&
        row->event_index == row->manifest_index + UINT64_C(1);
}


static int kmu_load_row_map(
    const char *directory,
    const char *name,
    KMUSchema schema,
    const char *split,
    size_t expected_rows,
    KMURowIdentity **rows_out
)
{
    const char *expected_header = schema == KMU_SCHEMA_MINI_V1
        ? "row_index,split_pos,manifest_index,event_index,event_id,split"
        : "split,split_pos,manifest_index,event_index,event_id";
    KMURowIdentity *rows;
    char path[KMU_PATH_CAP];
    char line[KMU_LINE_CAP];
    size_t count = 0u;
    FILE *file;
    int failed;

    *rows_out = NULL;
    if (!kmu_path_join(directory, name, path)) {
        return kmu_error("row-map path overflow");
    }
    file = fopen(path, "rb");
    if (file == NULL || fgets(line, sizeof(line), file) == NULL) {
        if (file != NULL) {
            fclose(file);
        }
        return kmu_error("cannot read row-map header: %s", name);
    }
    kmu_strip_line(line);
    if (strcmp(line, expected_header) != 0) {
        fclose(file);
        return kmu_error("row-map schema mismatch: %s", name);
    }
    if (expected_rows > SIZE_MAX / sizeof(*rows)) {
        fclose(file);
        return kmu_error("row-map allocation overflow");
    }
    rows = (KMURowIdentity *)calloc(expected_rows, sizeof(*rows));
    if (rows == NULL) {
        fclose(file);
        return kmu_error("row-map allocation failed");
    }
    while (fgets(line, sizeof(line), file) != NULL) {
        if (strchr(line, '\n') == NULL && !feof(file)) {
            free(rows);
            fclose(file);
            return kmu_error("row-map line too long: %s", name);
        }
        kmu_strip_line(line);
        if (count >= expected_rows ||
            !kmu_parse_row(line, schema, split, count, &rows[count])) {
            free(rows);
            fclose(file);
            return kmu_error("invalid row-map row: %s at %zu", name, count);
        }
        count++;
    }
    failed = ferror(file) || count != expected_rows;
    if (fclose(file) != 0) {
        failed = 1;
    }
    if (failed) {
        free(rows);
        return kmu_error("row-map row count mismatch: %s", name);
    }
    *rows_out = rows;
    return 1;
}


static void kmu_row_maps_free(KMURowMaps *maps)
{
    if (maps == NULL) {
        return;
    }
    free(maps->validation);
    free(maps->train);
    memset(maps, 0, sizeof(*maps));
}


static int kmu_rows_are_unique(const KMURowMaps *maps)
{
    unsigned char *seen = (unsigned char *)calloc(KMU_DATASET_ROWS, 1u);
    size_t index;

    if (seen == NULL) {
        return 0;
    }
    for (index = 0u; index < maps->train_count; index++) {
        size_t manifest = (size_t)maps->train[index].manifest_index;
        if (seen[manifest] != 0u) {
            free(seen);
            return 0;
        }
        seen[manifest] = 1u;
    }
    for (index = 0u; index < maps->validation_count; index++) {
        size_t manifest = (size_t)maps->validation[index].manifest_index;
        if (seen[manifest] != 0u) {
            free(seen);
            return 0;
        }
        seen[manifest] = 1u;
    }
    free(seen);
    return 1;
}


static int kmu_load_row_maps(
    const char *directory,
    const KMUSourceDescriptor *source,
    KMURowMaps *maps
)
{
    memset(maps, 0, sizeof(*maps));
    maps->train_count = source->train_rows;
    maps->validation_count = source->validation_rows;
    if (!kmu_load_row_map(
            directory, source->names[KMU_TRAIN_ROWS_FILE], source->schema,
            "train", source->train_rows, &maps->train
        ) ||
        !kmu_load_row_map(
            directory, source->names[KMU_VALIDATION_ROWS_FILE],
            source->schema, "validation", source->validation_rows,
            &maps->validation
        ) ||
        !kmu_rows_are_unique(maps)) {
        kmu_row_maps_free(maps);
        return kmu_error("row-map uniqueness/disjointness failed");
    }
    return 1;
}


static int kmu_row_equal(
    const KMURowIdentity *left,
    const KMURowIdentity *right
)
{
    return left->split_pos == right->split_pos &&
        left->manifest_index == right->manifest_index &&
        left->event_index == right->event_index &&
        strcmp(left->event_id, right->event_id) == 0 &&
        strcmp(left->split, right->split) == 0;
}


static int kmu_rows_prefix_equal(
    const KMURowIdentity *left,
    size_t left_count,
    const KMURowIdentity *right,
    size_t right_count,
    size_t compared_count
)
{
    size_t index;

    if (left == NULL || right == NULL || compared_count > left_count ||
        compared_count > right_count) {
        return 0;
    }
    for (index = 0u; index < compared_count; index++) {
        if (!kmu_row_equal(&left[index], &right[index])) {
            return 0;
        }
    }
    return 1;
}


static int kmu_validate_artifact(
    const char *directory,
    const KMUSourceDescriptor *source,
    KMURowMaps *maps
)
{
    size_t index;

    if (!kmu_directory_has_exact_inventory(directory, source)) {
        return 0;
    }
    for (index = 0u; index < KMU_FILE_COUNT; index++) {
        if (!kmu_verify_file_hash(directory, source, index)) {
            return 0;
        }
    }
    if (!kmu_validate_checksum_manifest(directory, source) ||
        !kmu_validate_metadata(directory, source) ||
        !kmu_validate_finite_matrix(
            directory, source->names[KMU_TRAIN_MATRIX_FILE],
            source->train_rows, source->columns
        ) ||
        !kmu_validate_finite_matrix(
            directory, source->names[KMU_VALIDATION_MATRIX_FILE],
            source->validation_rows, source->columns
        ) ||
        !kmu_load_row_maps(directory, source, maps)) {
        return 0;
    }
    return 1;
}


static const KMUSourceDescriptor *kmu_find_source(const char *token)
{
    size_t index;

    for (index = 0u; index < sizeof(kmu_sources) / sizeof(kmu_sources[0]);
         index++) {
        if (strcmp(kmu_sources[index].token, token) == 0) {
            return &kmu_sources[index];
        }
    }
    return NULL;
}


static int kmu_resolve_directory(
    const KMUSourceDescriptor *source,
    const char *repository_root,
    const char *full224_bundle,
    char directory[KMU_PATH_CAP]
)
{
    if (source->schema == KMU_SCHEMA_FULL224_V1) {
        return full224_bundle != NULL &&
            kmu_copy_text(directory, KMU_PATH_CAP, full224_bundle);
    }
    return kmu_path_join(repository_root, source->relative_directory,
                         directory);
}


static int kmu_verify_path_hash(const char *path, const char *expected)
{
    char actual[VAE_C_CHECKPOINT_SHA256_CAP];

    return kmu_hash_is_lower_hex(expected) &&
        vae_c_mu_file_sha256(path, actual) == VAE_C_MU_OK &&
        strcmp(actual, expected) == 0;
}


static int kmu_load_population_file(
    const char *repository_root,
    const char *relative_path,
    const char *expected_sha256,
    const char *split,
    size_t expected_rows,
    KMURowIdentity **rows_out
)
{
    const char *header =
        "split_name,split_pos,manifest_index,event_index,event_id,tensor_path";
    char path[KMU_PATH_CAP];
    char line[KMU_LINE_CAP];
    KMURowIdentity *rows = NULL;
    size_t count = 0u;
    FILE *file = NULL;
    int failed;

    *rows_out = NULL;
    if (!kmu_path_join(repository_root, relative_path, path) ||
        !kmu_verify_path_hash(path, expected_sha256)) {
        return kmu_error("canonical population SHA-256 mismatch: %s",
                         relative_path);
    }
    file = fopen(path, "rb");
    if (file == NULL || fgets(line, sizeof(line), file) == NULL) {
        if (file != NULL) {
            fclose(file);
        }
        return kmu_error("cannot read canonical population: %s",
                         relative_path);
    }
    kmu_strip_line(line);
    if (strcmp(line, header) != 0 ||
        expected_rows > SIZE_MAX / sizeof(*rows)) {
        fclose(file);
        return kmu_error("canonical population schema mismatch: %s",
                         relative_path);
    }
    rows = (KMURowIdentity *)calloc(expected_rows, sizeof(*rows));
    if (rows == NULL) {
        fclose(file);
        return kmu_error("canonical population allocation failed");
    }
    while (fgets(line, sizeof(line), file) != NULL) {
        char *fields[6];
        size_t field_count;

        if (strchr(line, '\n') == NULL && !feof(file)) {
            free(rows);
            fclose(file);
            return kmu_error("canonical population line too long");
        }
        kmu_strip_line(line);
        field_count = kmu_split_csv(line, fields, 6u);
        if (count >= expected_rows || field_count != 6u ||
            strcmp(fields[0], split) != 0 ||
            !kmu_parse_u64(fields[1], &rows[count].split_pos) ||
            !kmu_parse_u64(fields[2], &rows[count].manifest_index) ||
            !kmu_parse_u64(fields[3], &rows[count].event_index) ||
            rows[count].split_pos != (uint64_t)count ||
            rows[count].manifest_index >= KMU_DATASET_ROWS ||
            rows[count].event_index !=
                rows[count].manifest_index + UINT64_C(1) ||
            !kmu_event_id_valid(fields[4]) ||
            !kmu_copy_text(
                rows[count].event_id, sizeof(rows[count].event_id), fields[4]
            ) ||
            !kmu_copy_text(
                rows[count].split, sizeof(rows[count].split), fields[0]
            )) {
            free(rows);
            fclose(file);
            return kmu_error("canonical population row mismatch: %s at %zu",
                             relative_path, count);
        }
        /* fields[5] is historical tensor_path and is deliberately ignored. */
        count++;
    }
    failed = ferror(file) || count != expected_rows;
    if (fclose(file) != 0) {
        failed = 1;
    }
    if (failed) {
        free(rows);
        return kmu_error("canonical population row count mismatch: %s",
                         relative_path);
    }
    *rows_out = rows;
    return 1;
}


static int kmu_load_canonical_populations(
    const char *repository_root,
    KMURowMaps *full,
    KMURowMaps *matched
)
{
    memset(full, 0, sizeof(*full));
    memset(matched, 0, sizeof(*matched));
    full->train_count = KMU_FULL_TRAIN_ROWS;
    full->validation_count = KMU_FULL_VALIDATION_ROWS;
    matched->train_count = KMU_MATCHED_TRAIN_ROWS;
    matched->validation_count = KMU_MATCHED_VALIDATION_ROWS;
    if (!kmu_load_population_file(
            repository_root, "split_train_indices_seed123.csv",
            KMU_TRAIN_SPLIT_SHA256, "train", full->train_count, &full->train
        ) ||
        !kmu_load_population_file(
            repository_root, "split_val_indices_seed123.csv",
            KMU_VALIDATION_SPLIT_SHA256, "validation",
            full->validation_count, &full->validation
        ) ||
        !kmu_load_population_file(
            repository_root,
            "sakai_multires_c/configs/subset224_train_800_seed123.csv",
            KMU_TRAIN_SUBSET_SHA256, "train", matched->train_count,
            &matched->train
        ) ||
        !kmu_load_population_file(
            repository_root,
            "sakai_multires_c/configs/subset224_val_200_seed123.csv",
            KMU_VALIDATION_SUBSET_SHA256, "validation",
            matched->validation_count, &matched->validation
        ) ||
        !kmu_rows_are_unique(full) || !kmu_rows_are_unique(matched)) {
        kmu_row_maps_free(matched);
        kmu_row_maps_free(full);
        return 0;
    }
    return 1;
}


static int kmu_population_selection_matches(
    const KMURowMaps *source,
    const KMURowMaps *full,
    const KMURowMaps *matched,
    KMUScope scope
)
{
    if (!kmu_rows_prefix_equal(
            matched->train, matched->train_count,
            full->train, full->train_count, KMU_MATCHED_TRAIN_ROWS
        ) ||
        !kmu_rows_prefix_equal(
            matched->validation, matched->validation_count,
            full->validation, full->validation_count,
            KMU_MATCHED_VALIDATION_ROWS
        )) {
        return 0;
    }
    if (scope == KMU_SCOPE_FULL) {
        return source->train_count == KMU_FULL_TRAIN_ROWS &&
            source->validation_count == KMU_FULL_VALIDATION_ROWS &&
            kmu_rows_prefix_equal(
                source->train, source->train_count,
                full->train, full->train_count, KMU_FULL_TRAIN_ROWS
            ) &&
            kmu_rows_prefix_equal(
                source->validation, source->validation_count,
                full->validation, full->validation_count,
                KMU_FULL_VALIDATION_ROWS
            );
    }
    return scope == KMU_SCOPE_MATCHED_SUBSET1000 &&
        source->train_count >= KMU_MATCHED_TRAIN_ROWS &&
        source->validation_count >= KMU_MATCHED_VALIDATION_ROWS &&
        kmu_rows_prefix_equal(
            source->train, source->train_count,
            matched->train, matched->train_count, KMU_MATCHED_TRAIN_ROWS
        ) &&
        kmu_rows_prefix_equal(
            source->validation, source->validation_count,
            matched->validation, matched->validation_count,
            KMU_MATCHED_VALIDATION_ROWS
        );
}


static int kmu_validate_selection(
    const KMURequest *request,
    const KMURowMaps *source_rows
)
{
    KMURowMaps full;
    KMURowMaps matched;
    int passed;

    memset(&full, 0, sizeof(full));
    memset(&matched, 0, sizeof(matched));
    if (!kmu_load_canonical_populations(
            request->repository_root, &full, &matched
        )) {
        return 0;
    }
    passed = kmu_population_selection_matches(
        source_rows, &full, &matched, request->scope
    );
    kmu_row_maps_free(&matched);
    kmu_row_maps_free(&full);
    if (!passed) {
        return kmu_error("label-free population selection mismatch");
    }
    return 1;
}


static int kmu_validate_request(const KMURequest *request)
{
    KMURowMaps source_rows;
    char directory[KMU_PATH_CAP];
    size_t selected_train;
    size_t selected_validation;

    memset(&source_rows, 0, sizeof(source_rows));
    if (request == NULL || request->source == NULL ||
        request->repository_root == NULL ||
        (request->scope != KMU_SCOPE_FULL &&
         request->scope != KMU_SCOPE_MATCHED_SUBSET1000) ||
        (request->k != 22u && request->k != 36u) ||
        request->seed != KMU_SEED ||
        request->max_iterations != KMU_MAX_ITERATIONS ||
        (request->scope == KMU_SCOPE_FULL &&
         strcmp(request->source->token, "mini224") == 0) ||
        (request->source->schema == KMU_SCHEMA_FULL224_V1 &&
         request->full224_bundle == NULL) ||
        (request->source->schema != KMU_SCHEMA_FULL224_V1 &&
         request->full224_bundle != NULL)) {
        return kmu_error("invalid or non-canonical validate-only request");
    }
    if (!kmu_resolve_directory(
            request->source, request->repository_root,
            request->full224_bundle, directory
        ) ||
        !kmu_validate_artifact(directory, request->source, &source_rows) ||
        !kmu_validate_selection(request, &source_rows)) {
        kmu_row_maps_free(&source_rows);
        return 0;
    }

    selected_train = request->scope == KMU_SCOPE_FULL
        ? KMU_FULL_TRAIN_ROWS : KMU_MATCHED_TRAIN_ROWS;
    selected_validation = request->scope == KMU_SCOPE_FULL
        ? KMU_FULL_VALIDATION_ROWS : KMU_MATCHED_VALIDATION_ROWS;

    printf("MODE=VALIDATE_ONLY\n");
    printf("SCOPE=%s\n", request->scope == KMU_SCOPE_FULL
        ? "full" : "matched-subset1000");
    printf("PROFILE=%s\n", request->source->token);
    printf("PROFILE_ID=%s\n", request->source->profile_id);
    printf("SOURCE_FORMAT=%s\n",
           request->source->schema == KMU_SCHEMA_MINI_V1
               ? "VAE_C_MU_ARTIFACT_V1" : "SAKAI224_FULL_MU_ARTIFACT_V1");
    printf("SOURCE_DIRECTORY=%s\n", directory);
    printf("LATENT_DIM=%zu\n", request->source->columns);
    printf("SOURCE_TRAIN_ROWS=%zu\n", request->source->train_rows);
    printf("SOURCE_VALIDATION_ROWS=%zu\n",
           request->source->validation_rows);
    printf("SELECTED_TRAIN_ROWS=%zu\n", selected_train);
    printf("SELECTED_VALIDATION_ROWS=%zu\n", selected_validation);
    printf("K=%zu\n", request->k);
    printf("SEED=%" PRIu64 "\n", request->seed);
    printf("SINGLE_INITIALIZATION=true\n");
    printf("MAX_ITERATIONS=%zu\n", request->max_iterations);
    printf("FIT_POPULATION=train_only\n");
    printf("VALIDATION_ASSIGNMENT=frozen_train_centroids\n");
    printf("MATCHED_SELECTION=new_fit_on_first_800_train_200_validation\n");
    printf("ASSIGNMENT_SLICING=false\n");
    printf("RAW_MU=true\n");
    printf("PCA_APPLIED=false\n");
    printf("SCALING_APPLIED=false\n");
    printf("STANDARDIZATION_APPLIED=false\n");
    printf("SHA256_VERIFIED=true\n");
    printf("FORMAT_VERIFIED=true\n");
    printf("FINITE_MU_VERIFIED=true\n");
    printf("ROW_MAP_VERIFIED=true\n");
    printf("SELECTION_VERIFIED=true\n");
    printf("LABELS_USED=false\n");
    printf("KMEANS_BACKEND=gs_baseline_kmeans_fit\n");
    printf("NONCONVERGED_PUBLICATION_ALLOWED=false\n");
    printf("FIT_CALLS=0\n");
    printf("ASSIGN_CALLS=0\n");
    printf("OUTPUT_CREATED=false\n");
    printf("VAE_C_MU_KMEANS_VALIDATE_ONLY_PASS=true\n");

    kmu_row_maps_free(&source_rows);
    return 1;
}


static int kmu_checked_product(
    size_t left,
    size_t right,
    size_t *product
)
{
    if (product == NULL || (right != 0u && left > SIZE_MAX / right)) {
        return 0;
    }
    *product = left * right;
    return 1;
}


static int kmu_platform_is_output_compatible(void)
{
    const uint32_t integer = UINT32_C(0x01020304);
    const double one = 1.0;
    const unsigned char *integer_bytes =
        (const unsigned char *)&integer;
    const unsigned char *double_bytes = (const unsigned char *)&one;

    return kmu_platform_is_f32_little_endian() &&
        sizeof(uint32_t) == 4u && sizeof(uint64_t) == 8u &&
        sizeof(double) == 8u && DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024 &&
        integer_bytes[0] == 0x04u && integer_bytes[1] == 0x03u &&
        integer_bytes[2] == 0x02u && integer_bytes[3] == 0x01u &&
        double_bytes[0] == 0x00u && double_bytes[1] == 0x00u &&
        double_bytes[2] == 0x00u && double_bytes[3] == 0x00u &&
        double_bytes[4] == 0x00u && double_bytes[5] == 0x00u &&
        double_bytes[6] == 0xf0u && double_bytes[7] == 0x3fu;
}


static int kmu_code_commit_valid(const char *value)
{
    size_t index;

    if (value == NULL || strlen(value) != 40u) {
        return 0;
    }
    for (index = 0u; index < 40u; index++) {
        if (!((value[index] >= '0' && value[index] <= '9') ||
              (value[index] >= 'a' && value[index] <= 'f'))) {
            return 0;
        }
    }
    return strcmp(value, KMU_RUNNER_BASE_COMMIT) == 0;
}


static int kmu_path_state(const char *path)
{
    struct stat status;

    if (path == NULL || path[0] == '\0') {
        return -1;
    }
    errno = 0;
    if (stat(path, &status) != 0) {
        return errno == ENOENT ? 0 : -1;
    }
    if (S_ISREG(status.st_mode)) {
        return 1;
    }
    if (S_ISDIR(status.st_mode)) {
        return 2;
    }
    return 3;
}


static int kmu_existing_output_root_is_safe(const char *path)
{
    if (kmu_path_state(path) != 2) {
        return 0;
    }
#ifdef _WIN32
    {
        DWORD attributes = GetFileAttributesA(path);
        if (attributes == INVALID_FILE_ATTRIBUTES ||
            (attributes & FILE_ATTRIBUTE_DIRECTORY) == 0u ||
            (attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0u) {
            return 0;
        }
    }
#endif
    return 1;
}


static int kmu_resolve_existing_directory(
    const char *path,
    char resolved[KMU_PATH_CAP]
)
{
    size_t length;

    if (path == NULL || resolved == NULL || kmu_path_state(path) != 2) {
        return 0;
    }
#ifdef _WIN32
    if (_fullpath(resolved, path, KMU_PATH_CAP) == NULL) {
        return 0;
    }
#else
    if (realpath(path, resolved) == NULL) {
        return 0;
    }
#endif
    length = strlen(resolved);
    while (length > 1u &&
           (resolved[length - 1u] == '/' ||
            resolved[length - 1u] == '\\')) {
#ifdef _WIN32
        if (length == 3u && resolved[1] == ':') {
            break;
        }
#endif
        resolved[--length] = '\0';
    }
    return length > 0u;
}


static int kmu_path_is_same_or_descendant(
    const char *candidate,
    const char *parent
)
{
    size_t parent_length;
    int prefix_equal;

    if (candidate == NULL || parent == NULL) {
        return 0;
    }
    parent_length = strlen(parent);
#ifdef _WIN32
    prefix_equal = _strnicmp(candidate, parent, parent_length) == 0;
#else
    prefix_equal = strncmp(candidate, parent, parent_length) == 0;
#endif
    if (!prefix_equal) {
        return 0;
    }
    return candidate[parent_length] == '\0' ||
        candidate[parent_length] == '/' ||
        candidate[parent_length] == '\\';
}


static int kmu_output_root_is_separate_from_source(
    const char *output_root,
    const char *source_directory
)
{
    char resolved_output[KMU_PATH_CAP];
    char resolved_source[KMU_PATH_CAP];

    if (!kmu_resolve_existing_directory(output_root, resolved_output) ||
        !kmu_resolve_existing_directory(source_directory, resolved_source) ||
        kmu_path_is_same_or_descendant(resolved_output, resolved_source)) {
        return kmu_error(
            "output root must be separate from the source bundle"
        );
    }
    return 1;
}


static int kmu_resolve_output_paths(
    const KMURequest *request,
    char partial[KMU_PATH_CAP],
    char final[KMU_PATH_CAP]
)
{
    const char *scope;
    char name[256];
    int written;

    if (request == NULL || request->source == NULL ||
        !kmu_existing_output_root_is_safe(request->output_root)) {
        return kmu_error("output root must be an existing safe directory");
    }
    scope = request->scope == KMU_SCOPE_FULL
        ? "full" : "matched_subset1000";
    written = snprintf(
        name,
        sizeof(name),
        "%s_%s_k%zu_seed123_maxiter300",
        scope,
        request->source->token,
        request->k
    );
    if (written < 0 || (size_t)written >= sizeof(name) ||
        !kmu_path_join(request->output_root, name, final)) {
        return kmu_error("deterministic output path overflow");
    }
    written = snprintf(partial, KMU_PATH_CAP, "%s.partial", final);
    if (written < 0 || (size_t)written >= KMU_PATH_CAP) {
        return kmu_error("partial output path overflow");
    }
    if (kmu_path_state(final) != 0 || kmu_path_state(partial) != 0) {
        return kmu_error(
            "refusing overwrite: final or partial output already exists"
        );
    }
    return 1;
}


static int kmu_compute_code_hashes(
    const char *repository_root,
    KMUCodeHashes *hashes
)
{
    static const char *const paths[KMU_CODE_FILE_COUNT] = {
        "sakai_multires_c/tools/run_vae_c_mu_kmeans.c",
        "sakai_multires_c/src/baseline_kmeans.c",
        "sakai_multires_c/include/baseline_kmeans.h"
    };
    char full_path[KMU_PATH_CAP];
    char *outputs[KMU_CODE_FILE_COUNT];
    size_t index;

    if (repository_root == NULL || hashes == NULL) {
        return 0;
    }
    outputs[0] = hashes->runner;
    outputs[1] = hashes->core_source;
    outputs[2] = hashes->core_header;
    for (index = 0u; index < KMU_CODE_FILE_COUNT; index++) {
        if (!kmu_path_join(repository_root, paths[index], full_path) ||
            vae_c_mu_file_sha256(full_path, outputs[index]) != VAE_C_MU_OK) {
            return kmu_error("cannot hash code source: %s", paths[index]);
        }
    }
    if (strcmp(hashes->core_source, KMU_KMEANS_SOURCE_SHA256) != 0 ||
        strcmp(hashes->core_header, KMU_KMEANS_HEADER_SHA256) != 0) {
        return kmu_error("k-means core/header differs from frozen base");
    }
    return 1;
}


static int kmu_code_hashes_unchanged(
    const char *repository_root,
    const KMUCodeHashes *expected
)
{
    KMUCodeHashes actual;

    return expected != NULL &&
        kmu_compute_code_hashes(repository_root, &actual) &&
        memcmp(&actual, expected, sizeof(actual)) == 0;
}


static int kmu_read_matrix_prefix(
    const char *directory,
    const char *name,
    size_t source_rows,
    size_t selected_rows,
    size_t columns,
    float **matrix_out
)
{
    char path[KMU_PATH_CAP];
    struct stat status;
    uint64_t source_bytes;
    size_t value_count;
    size_t byte_count;
    float *matrix;
    FILE *file;
    size_t index;
    int failed = 0;

    if (matrix_out == NULL || selected_rows == 0u ||
        selected_rows > source_rows ||
        !kmu_platform_is_output_compatible() ||
        !kmu_expected_matrix_bytes(source_rows, columns, &source_bytes) ||
        !kmu_checked_product(selected_rows, columns, &value_count) ||
        !kmu_checked_product(value_count, sizeof(float), &byte_count) ||
        !kmu_path_join(directory, name, path) || stat(path, &status) != 0 ||
        status.st_size < 0 || (uint64_t)status.st_size != source_bytes) {
        return kmu_error("selected matrix contract mismatch: %s", name);
    }
    *matrix_out = NULL;
    matrix = (float *)malloc(byte_count);
    file = fopen(path, "rb");
    if (matrix == NULL || file == NULL) {
        free(matrix);
        if (file != NULL) {
            fclose(file);
        }
        return kmu_error("cannot allocate/read selected matrix: %s", name);
    }
    if (fread(matrix, sizeof(float), value_count, file) != value_count) {
        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    for (index = 0u; !failed && index < value_count; index++) {
        if (!isfinite(matrix[index])) {
            failed = 1;
        }
    }
    if (failed) {
        free(matrix);
        return kmu_error("selected matrix read/non-finite failure: %s", name);
    }
    *matrix_out = matrix;
    return 1;
}


static int kmu_sum_distances(
    const double *distances,
    size_t rows,
    double *sum
)
{
    size_t row;
    double total = 0.0;

    if (distances == NULL || sum == NULL) {
        return 0;
    }
    for (row = 0u; row < rows; row++) {
        if (!isfinite(distances[row]) || distances[row] < 0.0) {
            return 0;
        }
        total += distances[row];
        if (!isfinite(total)) {
            return 0;
        }
    }
    *sum = total;
    return 1;
}


static int kmu_output_name_index(const char *name)
{
    size_t index;

    for (index = 0u; index < KMU_OUTPUT_FILE_COUNT; index++) {
        if (strcmp(name, kmu_output_names[index]) == 0) {
            return (int)index;
        }
    }
    return -1;
}


static int kmu_output_inventory_valid(
    const char *directory,
    int require_all
)
{
    unsigned char seen[KMU_OUTPUT_FILE_COUNT] = {0};
    size_t count = 0u;

#ifdef _WIN32
    WIN32_FIND_DATAA entry;
    HANDLE search;
    DWORD find_error;
    char pattern[KMU_PATH_CAP];

    if (!kmu_path_join(directory, "*", pattern)) {
        return 0;
    }
    search = FindFirstFileA(pattern, &entry);
    if (search == INVALID_HANDLE_VALUE) {
        return !require_all && GetLastError() == ERROR_FILE_NOT_FOUND;
    }
    do {
        int index;
        if (strcmp(entry.cFileName, ".") == 0 ||
            strcmp(entry.cFileName, "..") == 0) {
            continue;
        }
        index = kmu_output_name_index(entry.cFileName);
        if (index < 0 || seen[index] != 0u ||
            (entry.dwFileAttributes &
             (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)) !=
                0u) {
            FindClose(search);
            return 0;
        }
        seen[index] = 1u;
        count++;
    } while (FindNextFileA(search, &entry) != 0);
    find_error = GetLastError();
    if (FindClose(search) == 0 || find_error != ERROR_NO_MORE_FILES) {
        return 0;
    }
#else
    {
        DIR *stream = opendir(directory);
        struct dirent *entry;
        if (stream == NULL) {
            return 0;
        }
        for (;;) {
            char path[KMU_PATH_CAP];
            struct stat status;
            int index;
            errno = 0;
            entry = readdir(stream);
            if (entry == NULL) {
                if (errno != 0) {
                    closedir(stream);
                    return 0;
                }
                break;
            }
            if (strcmp(entry->d_name, ".") == 0 ||
                strcmp(entry->d_name, "..") == 0) {
                continue;
            }
            index = kmu_output_name_index(entry->d_name);
            if (index < 0 || seen[index] != 0u ||
                !kmu_path_join(directory, entry->d_name, path) ||
                stat(path, &status) != 0 || !S_ISREG(status.st_mode)) {
                closedir(stream);
                return 0;
            }
            seen[index] = 1u;
            count++;
        }
        if (closedir(stream) != 0) {
            return 0;
        }
    }
#endif

    if (!require_all) {
        return count <= KMU_OUTPUT_FILE_COUNT;
    }
    if (count != KMU_OUTPUT_FILE_COUNT) {
        return 0;
    }
    for (count = 0u; count < KMU_OUTPUT_FILE_COUNT; count++) {
        if (seen[count] == 0u) {
            return 0;
        }
    }
    return 1;
}


static int kmu_finish_output_file(FILE *file)
{
    int failed = 0;

    if (file == NULL || fflush(file) != 0 || ferror(file)) {
        failed = 1;
    }
    if (file != NULL && fclose(file) != 0) {
        failed = 1;
    }
    return !failed;
}


static int kmu_open_output_file(
    const char *directory,
    const char *name,
    FILE **file
)
{
    char path[KMU_PATH_CAP];

    if (directory == NULL || name == NULL || file == NULL ||
        kmu_output_name_index(name) < 0 ||
        !kmu_path_join(directory, name, path) || kmu_path_state(path) != 0) {
        return 0;
    }
    *file = fopen(path, "wb");
    return *file != NULL;
}


static int kmu_write_binary(
    const char *directory,
    const char *name,
    const void *values,
    size_t element_size,
    size_t count
)
{
    FILE *file = NULL;
    int passed;

    if (values == NULL || element_size == 0u || count == 0u ||
        !kmu_open_output_file(directory, name, &file)) {
        return 0;
    }
    passed = fwrite(values, element_size, count, file) == count;
    if (!kmu_finish_output_file(file)) {
        passed = 0;
    }
    return passed;
}


static int kmu_write_assignment_csv(
    const char *directory,
    const char *name,
    const KMURowIdentity *rows,
    const uint32_t *assignments,
    const double *distances,
    size_t count
)
{
    FILE *file = NULL;
    size_t row;
    int passed = 1;

    if (rows == NULL || assignments == NULL || distances == NULL ||
        !kmu_open_output_file(directory, name, &file)) {
        return 0;
    }
    if (fputs(
            "row_index,split_pos,manifest_index,event_index,event_id,split,"
            "cluster_id,squared_distance\n",
            file
        ) == EOF) {
        passed = 0;
    }
    for (row = 0u; passed && row < count; row++) {
        if (rows[row].split_pos != (uint64_t)row ||
            !kmu_event_id_valid(rows[row].event_id) ||
            (strcmp(rows[row].split, "train") != 0 &&
             strcmp(rows[row].split, "validation") != 0) ||
            !isfinite(distances[row]) || distances[row] < 0.0 ||
            fprintf(
                file,
                "%zu,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%s,%s,%" PRIu32
                ",%.17g\n",
                row,
                rows[row].split_pos,
                rows[row].manifest_index,
                rows[row].event_index,
                rows[row].event_id,
                rows[row].split,
                assignments[row],
                distances[row]
            ) < 0) {
            passed = 0;
        }
    }
    if (!kmu_finish_output_file(file)) {
        passed = 0;
    }
    return passed;
}


static int kmu_write_cluster_sizes_csv(
    const char *directory,
    const uint64_t *sizes,
    size_t k
)
{
    FILE *file = NULL;
    size_t cluster;
    int passed = 1;

    if (sizes == NULL ||
        !kmu_open_output_file(directory, "cluster_sizes.csv", &file)) {
        return 0;
    }
    if (fputs("cluster_id,train_count\n", file) == EOF) {
        passed = 0;
    }
    for (cluster = 0u; passed && cluster < k; cluster++) {
        if (fprintf(
                file, "%zu,%" PRIu64 "\n", cluster, sizes[cluster]
            ) < 0) {
            passed = 0;
        }
    }
    if (!kmu_finish_output_file(file)) {
        passed = 0;
    }
    return passed;
}


static int kmu_write_initial_centroids_csv(
    const char *directory,
    const uint64_t *indices,
    size_t k
)
{
    FILE *file = NULL;
    size_t cluster;
    int passed = 1;

    if (indices == NULL ||
        !kmu_open_output_file(directory, "initial_centroids.csv", &file)) {
        return 0;
    }
    if (fputs("centroid_order,train_row_index\n", file) == EOF) {
        passed = 0;
    }
    for (cluster = 0u; passed && cluster < k; cluster++) {
        if (fprintf(
                file, "%zu,%" PRIu64 "\n", cluster, indices[cluster]
            ) < 0) {
            passed = 0;
        }
    }
    if (!kmu_finish_output_file(file)) {
        passed = 0;
    }
    return passed;
}


static int kmu_remove_created_partial(const char *directory)
{
    size_t index;

    if (kmu_path_state(directory) != 2 ||
        !kmu_output_inventory_valid(directory, 0)) {
        return 0;
    }
    for (index = 0u; index < KMU_OUTPUT_FILE_COUNT; index++) {
        char path[KMU_PATH_CAP];
        if (!kmu_path_join(directory, kmu_output_names[index], path)) {
            return 0;
        }
        errno = 0;
        if (remove(path) != 0 && errno != ENOENT) {
            return 0;
        }
    }
#ifdef _WIN32
    return _rmdir(directory) == 0;
#else
    return rmdir(directory) == 0;
#endif
}


static int kmu_write_metadata(
    const char *directory,
    const KMURequest *request,
    size_t selected_train,
    size_t selected_validation,
    const GSBaselineKMeansResult *result,
    const KMUNumericSummary *summary,
    const KMUCodeHashes *code_hashes
)
{
    const char *scope;
    const char *source_format;
    const char *source_bundle;
    const char *population;
    FILE *file = NULL;
    int passed;

    if (request == NULL || request->source == NULL || result == NULL ||
        summary == NULL || code_hashes == NULL ||
        !kmu_open_output_file(directory, "metadata.txt", &file)) {
        return 0;
    }
    scope = request->scope == KMU_SCOPE_FULL
        ? "full" : "matched_subset1000";
    source_format = request->source->schema == KMU_SCHEMA_MINI_V1
        ? "VAE_C_MU_ARTIFACT_V1" : "SAKAI224_FULL_MU_ARTIFACT_V1";
    source_bundle = request->source->relative_directory != NULL
        ? request->source->relative_directory
        : "sakai224_full_mu_epoch100_v1";
    population = request->scope == KMU_SCOPE_FULL
        ? "gravity_spy_o1_full_split_seed123_v1"
        : "canonical_prefix_800_train_200_validation_seed123_v1";

    passed = fprintf(
        file,
        "format=VAE_C_MU_KMEANS_ARTIFACT_V1\n"
        "scope=%s\n"
        "profile=%s\n"
        "profile_id=%s\n"
        "source_format=%s\n"
        "source_bundle=%s\n"
        "representation=posterior_mu\n"
        "view=central\n"
        "input_dtype=float32_ieee754\n"
        "input_endian=little\n"
        "input_layout=row_major_[N,latent_dim]\n"
        "latent_dim=%zu\n"
        "source_train_rows=%zu\n"
        "source_validation_rows=%zu\n"
        "selected_train_rows=%zu\n"
        "selected_validation_rows=%zu\n"
        "population_identity=%s\n"
        "row_order=native_split_pos_ascending\n"
        "row_identity=split_pos,manifest_index,event_index,event_id,split\n"
        "algorithm=kmeans_plus_plus_lloyd\n"
        "backend=gs_baseline_kmeans_fit\n"
        "input=raw_posterior_mu\n"
        "fit_input=train_only\n"
        "validation_policy=assign_to_frozen_train_centroids\n"
        "matched_refit=%s\n"
        "assignment_slicing=false\n"
        "k=%zu\n"
        "seed=%" PRIu64 "\n"
        "initializations=1\n"
        "max_iterations=%zu\n"
        "iterations=%zu\n"
        "converged=1\n"
        "empty_cluster_reseeds=%zu\n"
        "initial_train_inertia=%.17g\n"
        "final_train_inertia=%.17g\n"
        "validation_inertia=%.17g\n"
        "empty_final_clusters=%zu\n"
        "minimum_cluster_size=%zu\n"
        "maximum_cluster_size=%zu\n"
        "centroid_dtype=float64_ieee754\n"
        "centroid_endian=little\n"
        "centroid_layout=row_major_[k,latent_dim]\n"
        "assignment_dtype=uint32\n"
        "assignment_endian=little\n"
        "distance_dtype=float64_ieee754\n"
        "distance_endian=little\n"
        "cluster_size_dtype=uint64\n"
        "initial_index_dtype=uint64\n"
        "raw_mu=true\n"
        "pca_applied=false\n"
        "scaling_applied=false\n"
        "standardization_applied=false\n"
        "labels_used=false\n"
        "label_values_exported=false\n"
        "source_metadata_sha256=%s\n"
        "source_checksum_sha256=%s\n"
        "source_train_mu_sha256=%s\n"
        "source_validation_mu_sha256=%s\n"
        "source_train_rows_sha256=%s\n"
        "source_validation_rows_sha256=%s\n"
        "full_train_split_sha256=%s\n"
        "full_validation_split_sha256=%s\n"
        "matched_train_subset_sha256=%s\n"
        "matched_validation_subset_sha256=%s\n"
        "runner_base_code_commit=%s\n"
        "requested_code_commit=%s\n"
        "runner_source_sha256=%s\n"
        "kmeans_core_source_sha256=%s\n"
        "kmeans_core_header_sha256=%s\n"
        "checksum_manifest_excludes_self=true\n"
        "nonconverged_publication_allowed=false\n",
        scope,
        request->source->token,
        request->source->profile_id,
        source_format,
        source_bundle,
        request->source->columns,
        request->source->train_rows,
        request->source->validation_rows,
        selected_train,
        selected_validation,
        population,
        request->scope == KMU_SCOPE_MATCHED_SUBSET1000 ? "true" : "false",
        request->k,
        request->seed,
        request->max_iterations,
        result->iterations,
        result->empty_cluster_reseeds,
        summary->initial_inertia,
        summary->train_inertia,
        summary->validation_inertia,
        summary->empty_final_clusters,
        summary->minimum_cluster_size,
        summary->maximum_cluster_size,
        request->source->hashes[KMU_METADATA_FILE],
        request->source->hashes[KMU_CHECKSUM_FILE],
        request->source->hashes[KMU_TRAIN_MATRIX_FILE],
        request->source->hashes[KMU_VALIDATION_MATRIX_FILE],
        request->source->hashes[KMU_TRAIN_ROWS_FILE],
        request->source->hashes[KMU_VALIDATION_ROWS_FILE],
        KMU_TRAIN_SPLIT_SHA256,
        KMU_VALIDATION_SPLIT_SHA256,
        KMU_TRAIN_SUBSET_SHA256,
        KMU_VALIDATION_SUBSET_SHA256,
        KMU_RUNNER_BASE_COMMIT,
        request->code_commit,
        code_hashes->runner,
        code_hashes->core_source,
        code_hashes->core_header
    ) >= 0;
    if (!kmu_finish_output_file(file)) {
        passed = 0;
    }
    return passed;
}


static int kmu_write_sha256_manifest(
    const char *directory,
    const KMURequest *request,
    const KMUCodeHashes *code_hashes
)
{
    static const char *const canonical_names[4] = {
        "split_train_indices_seed123.csv",
        "split_val_indices_seed123.csv",
        "sakai_multires_c/configs/subset224_train_800_seed123.csv",
        "sakai_multires_c/configs/subset224_val_200_seed123.csv"
    };
    static const char *const canonical_hashes[4] = {
        KMU_TRAIN_SPLIT_SHA256,
        KMU_VALIDATION_SPLIT_SHA256,
        KMU_TRAIN_SUBSET_SHA256,
        KMU_VALIDATION_SUBSET_SHA256
    };
    static const char *const code_names[KMU_CODE_FILE_COUNT] = {
        "sakai_multires_c/tools/run_vae_c_mu_kmeans.c",
        "sakai_multires_c/src/baseline_kmeans.c",
        "sakai_multires_c/include/baseline_kmeans.h"
    };
    const char *code_values[KMU_CODE_FILE_COUNT];
    FILE *file = NULL;
    size_t index;
    int passed = 1;

    if (request == NULL || request->source == NULL || code_hashes == NULL ||
        !kmu_open_output_file(directory, "sha256.txt", &file)) {
        return 0;
    }
    code_values[0] = code_hashes->runner;
    code_values[1] = code_hashes->core_source;
    code_values[2] = code_hashes->core_header;

    for (index = 0u; passed && index < KMU_FILE_COUNT; index++) {
        if (fprintf(
                file, "SOURCE %s %s\n",
                request->source->hashes[index],
                request->source->names[index]
            ) < 0) {
            passed = 0;
        }
    }
    for (index = 0u; passed && index < 4u; index++) {
        if (fprintf(
                file, "CANONICAL %s %s\n",
                canonical_hashes[index], canonical_names[index]
            ) < 0) {
            passed = 0;
        }
    }
    for (index = 0u; passed && index < KMU_CODE_FILE_COUNT; index++) {
        if (fprintf(
                file, "CODE %s %s\n", code_values[index], code_names[index]
            ) < 0) {
            passed = 0;
        }
    }
    for (index = 0u; passed && index < KMU_OUTPUT_ARTIFACT_COUNT; index++) {
        char path[KMU_PATH_CAP];
        char hash[65];
        if (!kmu_path_join(directory, kmu_output_names[index], path) ||
            vae_c_mu_file_sha256(path, hash) != VAE_C_MU_OK ||
            fprintf(
                file, "ARTIFACT %s %s\n", hash, kmu_output_names[index]
            ) < 0) {
            passed = 0;
        }
    }
    if (!kmu_finish_output_file(file)) {
        passed = 0;
    }
    return passed;
}


static int kmu_expected_manifest_line(
    char line[KMU_LINE_CAP],
    const char *kind,
    const char *hash,
    const char *name
)
{
    int written;

    if (kind == NULL || !kmu_hash_is_lower_hex(hash) || name == NULL) {
        return 0;
    }
    written = snprintf(line, KMU_LINE_CAP, "%s %s %s", kind, hash, name);
    return written >= 0 && (size_t)written < KMU_LINE_CAP;
}


static int kmu_verify_sha256_manifest(
    const char *directory,
    const KMURequest *request,
    const KMUCodeHashes *code_hashes
)
{
    static const char *const canonical_names[4] = {
        "split_train_indices_seed123.csv",
        "split_val_indices_seed123.csv",
        "sakai_multires_c/configs/subset224_train_800_seed123.csv",
        "sakai_multires_c/configs/subset224_val_200_seed123.csv"
    };
    static const char *const canonical_hashes[4] = {
        KMU_TRAIN_SPLIT_SHA256,
        KMU_VALIDATION_SPLIT_SHA256,
        KMU_TRAIN_SUBSET_SHA256,
        KMU_VALIDATION_SUBSET_SHA256
    };
    static const char *const code_names[KMU_CODE_FILE_COUNT] = {
        "sakai_multires_c/tools/run_vae_c_mu_kmeans.c",
        "sakai_multires_c/src/baseline_kmeans.c",
        "sakai_multires_c/include/baseline_kmeans.h"
    };
    const char *code_values[KMU_CODE_FILE_COUNT];
    char path[KMU_PATH_CAP];
    char line[KMU_LINE_CAP];
    char expected[KMU_LINE_CAP];
    FILE *file;
    size_t index;
    int passed = 1;

    if (request == NULL || request->source == NULL || code_hashes == NULL ||
        !kmu_path_join(directory, "sha256.txt", path) ||
        (file = fopen(path, "rb")) == NULL) {
        return 0;
    }
    code_values[0] = code_hashes->runner;
    code_values[1] = code_hashes->core_source;
    code_values[2] = code_hashes->core_header;

#define KMU_VERIFY_MANIFEST_ENTRY(kind_value, hash_value, name_value) \
    do { \
        if (fgets(line, sizeof(line), file) == NULL || \
            !kmu_strip_line(line) || \
            !kmu_expected_manifest_line( \
                expected, (kind_value), (hash_value), (name_value) \
            ) || \
            strcmp(line, expected) != 0) { \
            passed = 0; \
        } \
    } while (0)

    for (index = 0u; passed && index < KMU_FILE_COUNT; index++) {
        KMU_VERIFY_MANIFEST_ENTRY(
            "SOURCE",
            request->source->hashes[index],
            request->source->names[index]
        );
    }
    for (index = 0u; passed && index < 4u; index++) {
        KMU_VERIFY_MANIFEST_ENTRY(
            "CANONICAL", canonical_hashes[index], canonical_names[index]
        );
    }
    for (index = 0u; passed && index < KMU_CODE_FILE_COUNT; index++) {
        KMU_VERIFY_MANIFEST_ENTRY(
            "CODE", code_values[index], code_names[index]
        );
    }
    for (index = 0u; passed && index < KMU_OUTPUT_ARTIFACT_COUNT; index++) {
        char artifact_path[KMU_PATH_CAP];
        char hash[65];
        if (!kmu_path_join(
                directory, kmu_output_names[index], artifact_path
            ) ||
            vae_c_mu_file_sha256(artifact_path, hash) != VAE_C_MU_OK) {
            passed = 0;
            break;
        }
        KMU_VERIFY_MANIFEST_ENTRY(
            "ARTIFACT", hash, kmu_output_names[index]
        );
    }
#undef KMU_VERIFY_MANIFEST_ENTRY

    if (passed && fgets(line, sizeof(line), file) != NULL) {
        passed = 0;
    }
    if (ferror(file) || fclose(file) != 0) {
        passed = 0;
    }
    return passed;
}


static int kmu_file_equals_memory(
    const char *directory,
    const char *name,
    const void *expected,
    size_t bytes
)
{
    unsigned char buffer[4096];
    const unsigned char *reference = (const unsigned char *)expected;
    char path[KMU_PATH_CAP];
    struct stat status;
    FILE *file;
    size_t offset = 0u;
    int passed = 1;

    if (expected == NULL || bytes == 0u ||
        !kmu_path_join(directory, name, path) || stat(path, &status) != 0 ||
        status.st_size < 0 || (uint64_t)status.st_size != (uint64_t)bytes ||
        (file = fopen(path, "rb")) == NULL) {
        return 0;
    }
    while (passed && offset < bytes) {
        size_t chunk = bytes - offset;
        if (chunk > sizeof(buffer)) {
            chunk = sizeof(buffer);
        }
        if (fread(buffer, 1u, chunk, file) != chunk ||
            memcmp(buffer, reference + offset, chunk) != 0) {
            passed = 0;
        }
        offset += chunk;
    }
    if (passed && fgetc(file) != EOF) {
        passed = 0;
    }
    if (ferror(file) || fclose(file) != 0) {
        passed = 0;
    }
    return passed;
}


static int kmu_protected_inputs_unchanged(
    const KMURequest *request,
    const char *source_directory,
    const KMUCodeHashes *code_hashes
)
{
    static const char *const canonical_paths[4] = {
        "split_train_indices_seed123.csv",
        "split_val_indices_seed123.csv",
        "sakai_multires_c/configs/subset224_train_800_seed123.csv",
        "sakai_multires_c/configs/subset224_val_200_seed123.csv"
    };
    static const char *const canonical_hashes[4] = {
        KMU_TRAIN_SPLIT_SHA256,
        KMU_VALIDATION_SPLIT_SHA256,
        KMU_TRAIN_SUBSET_SHA256,
        KMU_VALIDATION_SUBSET_SHA256
    };
    size_t index;

    if (request == NULL || request->source == NULL ||
        source_directory == NULL || code_hashes == NULL) {
        return 0;
    }
    if (!kmu_directory_has_exact_inventory(
            source_directory, request->source
        )) {
        return kmu_error("source bundle inventory changed during fit");
    }
    for (index = 0u; index < KMU_FILE_COUNT; index++) {
        if (!kmu_verify_file_hash(
                source_directory, request->source, index
            )) {
            return 0;
        }
    }
    for (index = 0u; index < 4u; index++) {
        char path[KMU_PATH_CAP];
        if (!kmu_path_join(
                request->repository_root, canonical_paths[index], path
            ) ||
            !kmu_verify_path_hash(path, canonical_hashes[index])) {
            return kmu_error("canonical population changed during fit");
        }
    }
    return kmu_code_hashes_unchanged(
        request->repository_root, code_hashes
    );
}


static int kmu_verify_assignment_csv(
    const char *directory,
    const char *name,
    const KMURowIdentity *rows,
    const uint32_t *assignments,
    const double *distances,
    size_t count,
    size_t k
)
{
    static const char *const header =
        "row_index,split_pos,manifest_index,event_index,event_id,split,"
        "cluster_id,squared_distance";
    char path[KMU_PATH_CAP];
    char line[KMU_LINE_CAP];
    char expected[KMU_LINE_CAP];
    FILE *file = NULL;
    size_t row;
    int passed = 1;

    if (rows == NULL || assignments == NULL || distances == NULL ||
        !kmu_path_join(directory, name, path) ||
        (file = fopen(path, "rb")) == NULL ||
        fgets(line, sizeof(line), file) == NULL) {
        if (file != NULL) {
            fclose(file);
        }
        return 0;
    }
    kmu_strip_line(line);
    if (strcmp(line, header) != 0) {
        passed = 0;
    }
    for (row = 0u; passed && row < count; row++) {
        int written;
        if (assignments[row] >= k || !isfinite(distances[row]) ||
            distances[row] < 0.0 ||
            fgets(line, sizeof(line), file) == NULL) {
            passed = 0;
            break;
        }
        kmu_strip_line(line);
        written = snprintf(
            expected,
            sizeof(expected),
            "%zu,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%s,%s,%" PRIu32
            ",%.17g",
            row,
            rows[row].split_pos,
            rows[row].manifest_index,
            rows[row].event_index,
            rows[row].event_id,
            rows[row].split,
            assignments[row],
            distances[row]
        );
        if (written < 0 || (size_t)written >= sizeof(expected) ||
            strcmp(line, expected) != 0) {
            passed = 0;
        }
    }
    if (passed && fgets(line, sizeof(line), file) != NULL) {
        passed = 0;
    }
    if (ferror(file) || fclose(file) != 0) {
        passed = 0;
    }
    return passed;
}


static int kmu_verify_metadata(
    const char *directory,
    const KMURequest *request,
    size_t selected_train,
    size_t selected_validation,
    const KMUCodeHashes *code_hashes
)
{
    char path[KMU_PATH_CAP];
    char expected[KMU_LINE_CAP];
    const char *scope = request->scope == KMU_SCOPE_FULL
        ? "full" : "matched_subset1000";

    if (!kmu_path_join(directory, "metadata.txt", path) ||
        !kmu_file_has_exact_line(path, "format=VAE_C_MU_KMEANS_ARTIFACT_V1")) {
        return 0;
    }
#define KMU_VERIFY_METADATA_TEXT(key_name, text_value) \
    do { \
        int written = snprintf( \
            expected, sizeof(expected), "%s=%s", (key_name), (text_value) \
        ); \
        if (written < 0 || (size_t)written >= sizeof(expected) || \
            !kmu_file_has_exact_line(path, expected)) { \
            return 0; \
        } \
    } while (0)
#define KMU_VERIFY_METADATA_SIZE(key_name, size_value) \
    do { \
        int written = snprintf( \
            expected, sizeof(expected), "%s=%zu", (key_name), (size_value) \
        ); \
        if (written < 0 || (size_t)written >= sizeof(expected) || \
            !kmu_file_has_exact_line(path, expected)) { \
            return 0; \
        } \
    } while (0)
    KMU_VERIFY_METADATA_TEXT("scope", scope);
    KMU_VERIFY_METADATA_TEXT("profile", request->source->token);
    KMU_VERIFY_METADATA_TEXT("profile_id", request->source->profile_id);
    KMU_VERIFY_METADATA_SIZE("latent_dim", request->source->columns);
    KMU_VERIFY_METADATA_SIZE("selected_train_rows", selected_train);
    KMU_VERIFY_METADATA_SIZE(
        "selected_validation_rows", selected_validation
    );
    KMU_VERIFY_METADATA_SIZE("k", request->k);
    KMU_VERIFY_METADATA_SIZE("seed", (size_t)request->seed);
    KMU_VERIFY_METADATA_SIZE("max_iterations", request->max_iterations);
    KMU_VERIFY_METADATA_TEXT("converged", "1");
    KMU_VERIFY_METADATA_TEXT(
        "source_metadata_sha256",
        request->source->hashes[KMU_METADATA_FILE]
    );
    KMU_VERIFY_METADATA_TEXT(
        "source_train_mu_sha256",
        request->source->hashes[KMU_TRAIN_MATRIX_FILE]
    );
    KMU_VERIFY_METADATA_TEXT(
        "source_validation_mu_sha256",
        request->source->hashes[KMU_VALIDATION_MATRIX_FILE]
    );
    KMU_VERIFY_METADATA_TEXT(
        "runner_base_code_commit", KMU_RUNNER_BASE_COMMIT
    );
    KMU_VERIFY_METADATA_TEXT("requested_code_commit", request->code_commit);
    KMU_VERIFY_METADATA_TEXT(
        "runner_source_sha256", code_hashes->runner
    );
    KMU_VERIFY_METADATA_TEXT(
        "kmeans_core_source_sha256", code_hashes->core_source
    );
    KMU_VERIFY_METADATA_TEXT(
        "kmeans_core_header_sha256", code_hashes->core_header
    );
    KMU_VERIFY_METADATA_TEXT("labels_used", "false");
    KMU_VERIFY_METADATA_TEXT("assignment_slicing", "false");
    KMU_VERIFY_METADATA_TEXT(
        "validation_policy", "assign_to_frozen_train_centroids"
    );
#undef KMU_VERIFY_METADATA_SIZE
#undef KMU_VERIFY_METADATA_TEXT
    return 1;
}


static int kmu_fit_loaded_and_publish(
    const KMURequest *request,
    const char *source_directory,
    const KMURowMaps *source_rows,
    const float *train,
    const float *validation,
    size_t selected_train,
    size_t selected_validation,
    const KMUCodeHashes *code_hashes,
    const char *partial,
    const char *final,
    KMURunReport *report
)
{
    GSBaselineKMeansResult result;
    KMUNumericSummary summary;
    size_t train_values;
    size_t validation_values;
    size_t centroid_values;
    size_t train_float_bytes;
    size_t validation_float_bytes;
    size_t centroid_bytes;
    size_t train_size_t_bytes;
    size_t validation_size_t_bytes;
    size_t train_u32_bytes;
    size_t validation_u32_bytes;
    size_t train_f64_bytes;
    size_t validation_f64_bytes;
    size_t k_u64_bytes;
    float *train_copy = NULL;
    float *validation_copy = NULL;
    size_t *train_assignments = NULL;
    size_t *validation_assignments = NULL;
    double *train_distances = NULL;
    double *validation_distances = NULL;
    double *centroids_copy = NULL;
    uint32_t *train_u32 = NULL;
    uint32_t *validation_u32 = NULL;
    uint64_t *cluster_sizes_u64 = NULL;
    uint64_t *initial_indices_u64 = NULL;
    size_t row;
    size_t cluster;
    size_t total_cluster_size = 0u;
    int partial_created = 0;
    int passed = 0;
    int fit_status;

    memset(&result, 0, sizeof(result));
    memset(&summary, 0, sizeof(summary));
    if (report != NULL) {
        memset(report, 0, sizeof(*report));
    }
    if (request == NULL || request->source == NULL ||
        source_directory == NULL || source_rows == NULL ||
        train == NULL || validation == NULL || code_hashes == NULL ||
        partial == NULL || final == NULL || selected_train == 0u ||
        selected_validation == 0u || request->code_commit == NULL ||
        request->source->columns == 0u ||
        source_rows->train == NULL || source_rows->validation == NULL ||
        source_rows->train_count < selected_train ||
        source_rows->validation_count < selected_validation ||
        request->k == 0u || request->k > selected_train ||
        request->k > (size_t)UINT32_MAX ||
        !kmu_platform_is_output_compatible() ||
        !kmu_output_root_is_separate_from_source(
            request->output_root, source_directory
        ) ||
        !kmu_checked_product(
            selected_train, request->source->columns, &train_values
        ) ||
        !kmu_checked_product(
            selected_validation,
            request->source->columns,
            &validation_values
        ) ||
        !kmu_checked_product(
            request->k, request->source->columns, &centroid_values
        ) ||
        !kmu_checked_product(
            train_values, sizeof(float), &train_float_bytes
        ) ||
        !kmu_checked_product(
            validation_values, sizeof(float), &validation_float_bytes
        ) ||
        !kmu_checked_product(
            centroid_values, sizeof(double), &centroid_bytes
        ) ||
        !kmu_checked_product(
            selected_train, sizeof(size_t), &train_size_t_bytes
        ) ||
        !kmu_checked_product(
            selected_validation,
            sizeof(size_t),
            &validation_size_t_bytes
        ) ||
        !kmu_checked_product(
            selected_train, sizeof(uint32_t), &train_u32_bytes
        ) ||
        !kmu_checked_product(
            selected_validation,
            sizeof(uint32_t),
            &validation_u32_bytes
        ) ||
        !kmu_checked_product(
            selected_train, sizeof(double), &train_f64_bytes
        ) ||
        !kmu_checked_product(
            selected_validation, sizeof(double), &validation_f64_bytes
        ) ||
        !kmu_checked_product(
            request->k, sizeof(uint64_t), &k_u64_bytes
        )) {
        kmu_error("fit allocation/dimension contract failed");
        goto cleanup;
    }

    train_copy = (float *)malloc(train_float_bytes);
    validation_copy = (float *)malloc(validation_float_bytes);
    train_assignments = (size_t *)malloc(train_size_t_bytes);
    validation_assignments = (size_t *)malloc(validation_size_t_bytes);
    train_distances = (double *)malloc(train_f64_bytes);
    validation_distances = (double *)malloc(validation_f64_bytes);
    centroids_copy = (double *)malloc(centroid_bytes);
    train_u32 = (uint32_t *)malloc(train_u32_bytes);
    validation_u32 = (uint32_t *)malloc(validation_u32_bytes);
    cluster_sizes_u64 = (uint64_t *)malloc(k_u64_bytes);
    initial_indices_u64 = (uint64_t *)malloc(k_u64_bytes);
    if (train_copy == NULL || validation_copy == NULL ||
        train_assignments == NULL || validation_assignments == NULL ||
        train_distances == NULL || validation_distances == NULL ||
        centroids_copy == NULL || train_u32 == NULL ||
        validation_u32 == NULL || cluster_sizes_u64 == NULL ||
        initial_indices_u64 == NULL) {
        kmu_error("fit workspace allocation failed");
        goto cleanup;
    }
    memcpy(train_copy, train, train_float_bytes);
    memcpy(validation_copy, validation, validation_float_bytes);

    if (kmu_path_state(final) != 0 || kmu_path_state(partial) != 0) {
        kmu_error("output appeared before fit");
        goto cleanup;
    }
    if (report != NULL) {
        report->fit_calls++;
    }
    fit_status = gs_baseline_kmeans_fit(
        train,
        selected_train,
        request->source->columns,
        request->k,
        request->seed,
        request->max_iterations,
        &result
    );
    if (fit_status != GS_BASELINE_KMEANS_SUCCESS) {
        kmu_error("train-only k-means fit failed with status %d", fit_status);
        goto cleanup;
    }
    if (report != NULL) {
        report->converged = result.converged;
    }
    if (result.converged != 1) {
        kmu_error("k-means did not converge; publication refused");
        goto cleanup;
    }
    if (result.rows != selected_train ||
        result.cols != request->source->columns ||
        result.k != request->k || result.seed != request->seed ||
        result.iterations == 0u ||
        result.iterations > request->max_iterations ||
        result.centroids == NULL || result.assignments == NULL ||
        result.cluster_sizes == NULL ||
        result.initial_centroid_indices == NULL ||
        !isfinite(result.initial_inertia) || result.initial_inertia < 0.0 ||
        !isfinite(result.inertia) || result.inertia < 0.0) {
        kmu_error("fit result contract failed");
        goto cleanup;
    }
    for (row = 0u; row < centroid_values; row++) {
        if (!isfinite(result.centroids[row])) {
            kmu_error("non-finite centroid");
            goto cleanup;
        }
    }

    if (report != NULL) {
        report->assign_calls++;
    }
    if (gs_baseline_kmeans_assign(
            train,
            selected_train,
            request->source->columns,
            result.centroids,
            request->k,
            train_assignments,
            train_distances
        ) != GS_BASELINE_KMEANS_SUCCESS) {
        kmu_error("train reassignment failed");
        goto cleanup;
    }
    memcpy(centroids_copy, result.centroids, centroid_bytes);
    if (report != NULL) {
        report->assign_calls++;
    }
    if (gs_baseline_kmeans_assign(
            validation,
            selected_validation,
            request->source->columns,
            result.centroids,
            request->k,
            validation_assignments,
            validation_distances
        ) != GS_BASELINE_KMEANS_SUCCESS ||
        memcmp(centroids_copy, result.centroids, centroid_bytes) != 0 ||
        memcmp(train_copy, train, train_float_bytes) != 0 ||
        memcmp(validation_copy, validation, validation_float_bytes) != 0) {
        kmu_error("frozen-centroid assignment or input immutability failed");
        goto cleanup;
    }

    memset(cluster_sizes_u64, 0, k_u64_bytes);
    summary.initial_inertia = result.initial_inertia;
    summary.minimum_cluster_size = selected_train;
    for (row = 0u; row < selected_train; row++) {
        size_t assignment = train_assignments[row];
        if (assignment >= request->k ||
            assignment != result.assignments[row] ||
            assignment > (size_t)UINT32_MAX) {
            kmu_error("train assignment contract failed");
            goto cleanup;
        }
        train_u32[row] = (uint32_t)assignment;
        if (cluster_sizes_u64[assignment] == UINT64_MAX) {
            kmu_error("cluster-size overflow");
            goto cleanup;
        }
        cluster_sizes_u64[assignment]++;
    }
    for (row = 0u; row < selected_validation; row++) {
        size_t assignment = validation_assignments[row];
        if (assignment >= request->k ||
            assignment > (size_t)UINT32_MAX) {
            kmu_error("validation assignment contract failed");
            goto cleanup;
        }
        validation_u32[row] = (uint32_t)assignment;
    }
    if (!kmu_sum_distances(
            train_distances, selected_train, &summary.train_inertia
        ) ||
        !kmu_sum_distances(
            validation_distances,
            selected_validation,
            &summary.validation_inertia
        ) ||
        memcmp(
            &summary.train_inertia, &result.inertia, sizeof(double)
        ) != 0) {
        kmu_error("inertia recomputation failed");
        goto cleanup;
    }
    for (cluster = 0u; cluster < request->k; cluster++) {
        size_t inner;
        size_t size;
        if (result.initial_centroid_indices[cluster] >= selected_train) {
            kmu_error("initial centroid index out of range");
            goto cleanup;
        }
        for (inner = 0u; inner < cluster; inner++) {
            if (result.initial_centroid_indices[inner] ==
                result.initial_centroid_indices[cluster]) {
                kmu_error("duplicate initial centroid index");
                goto cleanup;
            }
        }
        initial_indices_u64[cluster] =
            (uint64_t)result.initial_centroid_indices[cluster];
        if (cluster_sizes_u64[cluster] !=
            (uint64_t)result.cluster_sizes[cluster]) {
            kmu_error("cluster-size result mismatch");
            goto cleanup;
        }
        size = (size_t)cluster_sizes_u64[cluster];
        if (size == 0u) {
            summary.empty_final_clusters++;
        }
        if (size < summary.minimum_cluster_size) {
            summary.minimum_cluster_size = size;
        }
        if (size > summary.maximum_cluster_size) {
            summary.maximum_cluster_size = size;
        }
        if (size > SIZE_MAX - total_cluster_size) {
            kmu_error("cluster-size total overflow");
            goto cleanup;
        }
        total_cluster_size += size;
    }
    if (total_cluster_size != selected_train) {
        kmu_error("cluster-size total mismatch");
        goto cleanup;
    }

    if (kmu_path_state(final) != 0 || kmu_path_state(partial) != 0) {
        kmu_error("output appeared before partial creation");
        goto cleanup;
    }
#ifdef _WIN32
    if (_mkdir(partial) != 0) {
#else
    if (mkdir(partial, 0777) != 0) {
#endif
        kmu_error("partial directory creation failed");
        goto cleanup;
    }
    partial_created = 1;

    if (!kmu_write_binary(
            partial,
            "centroids_f64.bin",
            result.centroids,
            sizeof(double),
            centroid_values
        ) ||
        !kmu_write_binary(
            partial,
            "train_assignments_u32.bin",
            train_u32,
            sizeof(uint32_t),
            selected_train
        ) ||
        !kmu_write_binary(
            partial,
            "validation_assignments_u32.bin",
            validation_u32,
            sizeof(uint32_t),
            selected_validation
        ) ||
        !kmu_write_binary(
            partial,
            "train_squared_distances_f64.bin",
            train_distances,
            sizeof(double),
            selected_train
        ) ||
        !kmu_write_binary(
            partial,
            "validation_squared_distances_f64.bin",
            validation_distances,
            sizeof(double),
            selected_validation
        ) ||
        !kmu_write_binary(
            partial,
            "cluster_sizes_train_u64.bin",
            cluster_sizes_u64,
            sizeof(uint64_t),
            request->k
        ) ||
        !kmu_write_binary(
            partial,
            "initial_centroid_indices_u64.bin",
            initial_indices_u64,
            sizeof(uint64_t),
            request->k
        ) ||
        !kmu_write_assignment_csv(
            partial,
            "train_assignments.csv",
            source_rows->train,
            train_u32,
            train_distances,
            selected_train
        ) ||
        !kmu_write_assignment_csv(
            partial,
            "validation_assignments.csv",
            source_rows->validation,
            validation_u32,
            validation_distances,
            selected_validation
        ) ||
        !kmu_write_cluster_sizes_csv(
            partial, cluster_sizes_u64, request->k
        ) ||
        !kmu_write_initial_centroids_csv(
            partial, initial_indices_u64, request->k
        ) ||
        !kmu_write_metadata(
            partial,
            request,
            selected_train,
            selected_validation,
            &result,
            &summary,
            code_hashes
        ) ||
        !kmu_write_sha256_manifest(partial, request, code_hashes)) {
        kmu_error("artifact write failed");
        goto cleanup;
    }

    if (!kmu_output_inventory_valid(partial, 1) ||
        !kmu_file_equals_memory(
            partial, "centroids_f64.bin",
            result.centroids, centroid_bytes
        ) ||
        !kmu_file_equals_memory(
            partial, "train_assignments_u32.bin",
            train_u32, train_u32_bytes
        ) ||
        !kmu_file_equals_memory(
            partial, "validation_assignments_u32.bin",
            validation_u32, validation_u32_bytes
        ) ||
        !kmu_file_equals_memory(
            partial, "train_squared_distances_f64.bin",
            train_distances, train_f64_bytes
        ) ||
        !kmu_file_equals_memory(
            partial, "validation_squared_distances_f64.bin",
            validation_distances, validation_f64_bytes
        ) ||
        !kmu_file_equals_memory(
            partial, "cluster_sizes_train_u64.bin",
            cluster_sizes_u64, k_u64_bytes
        ) ||
        !kmu_file_equals_memory(
            partial, "initial_centroid_indices_u64.bin",
            initial_indices_u64, k_u64_bytes
        ) ||
        !kmu_verify_assignment_csv(
            partial,
            "train_assignments.csv",
            source_rows->train,
            train_u32,
            train_distances,
            selected_train,
            request->k
        ) ||
        !kmu_verify_assignment_csv(
            partial,
            "validation_assignments.csv",
            source_rows->validation,
            validation_u32,
            validation_distances,
            selected_validation,
            request->k
        ) ||
        !kmu_verify_metadata(
            partial,
            request,
            selected_train,
            selected_validation,
            code_hashes
        ) ||
        !kmu_verify_sha256_manifest(partial, request, code_hashes)) {
        kmu_error("artifact readback validation failed");
        goto cleanup;
    }
    if (!kmu_protected_inputs_unchanged(
            request, source_directory, code_hashes
        )) {
        kmu_error("protected input changed during fit");
        goto cleanup;
    }
    if (kmu_path_state(final) != 0 || kmu_path_state(partial) != 2 ||
        rename(partial, final) != 0) {
        kmu_error("atomic final promotion failed");
        goto cleanup;
    }
    partial_created = 0;
    if (report != NULL) {
        report->output_created = 1;
    }
    passed = 1;

cleanup:
    if (partial_created && !kmu_remove_created_partial(partial)) {
        kmu_error("confined current-run partial cleanup failed");
        passed = 0;
    }
    gs_baseline_kmeans_result_free(&result);
    free(initial_indices_u64);
    free(cluster_sizes_u64);
    free(validation_u32);
    free(train_u32);
    free(centroids_copy);
    free(validation_distances);
    free(train_distances);
    free(validation_assignments);
    free(train_assignments);
    free(validation_copy);
    free(train_copy);
    return passed;
}


static void kmu_print_fit_report(
    const KMURunReport *report,
    const char *final_path
)
{
    if (report == NULL) {
        return;
    }
    printf("FIT_CALLS=%zu\n", report->fit_calls);
    printf("ASSIGN_CALLS=%zu\n", report->assign_calls);
    printf("CONVERGED=%d\n", report->converged);
    printf("OUTPUT_CREATED=%s\n",
           report->output_created ? "true" : "false");
    if (report->output_created && final_path != NULL) {
        printf("OUTPUT_DIRECTORY=%s\n", final_path);
    }
}


static int kmu_fit_request(const KMURequest *request)
{
    KMURowMaps source_rows;
    KMUCodeHashes code_hashes;
    KMURunReport report;
    char source_directory[KMU_PATH_CAP];
    char partial[KMU_PATH_CAP];
    char final[KMU_PATH_CAP];
    float *train = NULL;
    float *validation = NULL;
    size_t selected_train;
    size_t selected_validation;
    int passed = 0;

    memset(&source_rows, 0, sizeof(source_rows));
    memset(&code_hashes, 0, sizeof(code_hashes));
    memset(&report, 0, sizeof(report));
    memset(partial, 0, sizeof(partial));
    memset(final, 0, sizeof(final));
    if (request == NULL || request->source == NULL ||
        request->repository_root == NULL || request->output_root == NULL ||
        !kmu_code_commit_valid(request->code_commit) ||
        (request->scope != KMU_SCOPE_FULL &&
         request->scope != KMU_SCOPE_MATCHED_SUBSET1000) ||
        (request->k != 22u && request->k != 36u) ||
        request->seed != KMU_SEED ||
        request->max_iterations != KMU_MAX_ITERATIONS ||
        (request->scope == KMU_SCOPE_FULL &&
         strcmp(request->source->token, "mini224") == 0) ||
        (request->source->schema == KMU_SCHEMA_FULL224_V1 &&
         request->full224_bundle == NULL) ||
        (request->source->schema != KMU_SCHEMA_FULL224_V1 &&
         request->full224_bundle != NULL)) {
        kmu_error("invalid or non-canonical fit request");
        kmu_print_fit_report(&report, NULL);
        return 0;
    }

    /*
       Risolvere sorgente e output, verificarne la separazione e controllare
       l'assenza dell'output precede deliberatamente qualunque caricamento o
       chiamata k-means.
    */
    if (!kmu_resolve_directory(
            request->source,
            request->repository_root,
            request->full224_bundle,
            source_directory
        ) ||
        !kmu_output_root_is_separate_from_source(
            request->output_root, source_directory
        ) ||
        !kmu_resolve_output_paths(request, partial, final)) {
        kmu_print_fit_report(&report, NULL);
        return 0;
    }
    if (!kmu_validate_artifact(
            source_directory, request->source, &source_rows
        ) ||
        !kmu_validate_selection(request, &source_rows) ||
        !kmu_compute_code_hashes(
            request->repository_root, &code_hashes
        )) {
        goto cleanup;
    }
    selected_train = request->scope == KMU_SCOPE_FULL
        ? KMU_FULL_TRAIN_ROWS : KMU_MATCHED_TRAIN_ROWS;
    selected_validation = request->scope == KMU_SCOPE_FULL
        ? KMU_FULL_VALIDATION_ROWS : KMU_MATCHED_VALIDATION_ROWS;
    if (!kmu_read_matrix_prefix(
            source_directory,
            request->source->names[KMU_TRAIN_MATRIX_FILE],
            request->source->train_rows,
            selected_train,
            request->source->columns,
            &train
        ) ||
        !kmu_read_matrix_prefix(
            source_directory,
            request->source->names[KMU_VALIDATION_MATRIX_FILE],
            request->source->validation_rows,
            selected_validation,
            request->source->columns,
            &validation
        )) {
        goto cleanup;
    }

    printf("MODE=FIT\n");
    printf("SCOPE=%s\n", request->scope == KMU_SCOPE_FULL
        ? "full" : "matched-subset1000");
    printf("PROFILE=%s\n", request->source->token);
    printf("SELECTED_TRAIN_ROWS=%zu\n", selected_train);
    printf("SELECTED_VALIDATION_ROWS=%zu\n", selected_validation);
    printf("K=%zu\n", request->k);
    printf("SEED=%" PRIu64 "\n", request->seed);
    printf("MAX_ITERATIONS=%zu\n", request->max_iterations);
    printf("KMEANS_FIT_STARTED\n");
    fflush(stdout);
    passed = kmu_fit_loaded_and_publish(
        request,
        source_directory,
        &source_rows,
        train,
        validation,
        selected_train,
        selected_validation,
        &code_hashes,
        partial,
        final,
        &report
    );
    if (passed) {
        printf("KMEANS_FIT_AND_ATOMIC_PUBLICATION_COMPLETED\n");
    }

cleanup:
    kmu_print_fit_report(&report, passed ? final : NULL);
    free(validation);
    free(train);
    kmu_row_maps_free(&source_rows);
    return passed;
}


static int kmu_parse_size(const char *text, size_t *value)
{
    uint64_t parsed;

    if (!kmu_parse_u64(text, &parsed) || parsed > SIZE_MAX) {
        return 0;
    }
    *value = (size_t)parsed;
    return 1;
}


static void kmu_usage(const char *program)
{
    fprintf(
        stderr,
        "Usage: %s <--validate-only|--fit> "
        "--scope <full|matched-subset1000> "
        "--profile <mini75|mini112|mini224|full224> --k <22|36> "
        "--seed 123 --max-iterations 300 "
        "[--full224-bundle <directory>] "
        "[--output-root <existing-directory> --code-commit <40hex>] "
        "<repository_root>\n"
        "       --output-root and --code-commit are required only for --fit.\n",
        program
    );
}


#ifndef VAE_C_MU_KMEANS_NO_MAIN
int main(int argc, char **argv)
{
    KMURequest request;
    const char *scope_text = NULL;
    const char *profile_text = NULL;
    int validate_only_seen = 0;
    int fit_seen = 0;
    int k_seen = 0;
    int seed_seen = 0;
    int max_iterations_seen = 0;
    int positional_seen = 0;
    int argument;

    if (setlocale(LC_ALL, "C") == NULL) {
        kmu_error("C locale unavailable");
        return 1;
    }
    memset(&request, 0, sizeof(request));
    for (argument = 1; argument < argc; argument++) {
        if (strcmp(argv[argument], "--validate-only") == 0) {
            if (validate_only_seen || fit_seen) {
                kmu_usage(argv[0]);
                return 2;
            }
            validate_only_seen = 1;
        } else if (strcmp(argv[argument], "--fit") == 0) {
            if (fit_seen || validate_only_seen) {
                kmu_usage(argv[0]);
                return 2;
            }
            fit_seen = 1;
        } else if (strcmp(argv[argument], "--scope") == 0 &&
                   argument + 1 < argc && scope_text == NULL) {
            scope_text = argv[++argument];
        } else if (strcmp(argv[argument], "--profile") == 0 &&
                   argument + 1 < argc && profile_text == NULL) {
            profile_text = argv[++argument];
        } else if (strcmp(argv[argument], "--k") == 0 &&
                   argument + 1 < argc && !k_seen) {
            k_seen = 1;
            if (!kmu_parse_size(argv[++argument], &request.k)) {
                kmu_usage(argv[0]);
                return 2;
            }
        } else if (strcmp(argv[argument], "--seed") == 0 &&
                   argument + 1 < argc && !seed_seen) {
            seed_seen = 1;
            if (!kmu_parse_u64(argv[++argument], &request.seed)) {
                kmu_usage(argv[0]);
                return 2;
            }
        } else if (strcmp(argv[argument], "--max-iterations") == 0 &&
                   argument + 1 < argc && !max_iterations_seen) {
            max_iterations_seen = 1;
            if (!kmu_parse_size(argv[++argument], &request.max_iterations)) {
                kmu_usage(argv[0]);
                return 2;
            }
        } else if (strcmp(argv[argument], "--full224-bundle") == 0 &&
                   argument + 1 < argc && request.full224_bundle == NULL) {
            request.full224_bundle = argv[++argument];
        } else if (strcmp(argv[argument], "--output-root") == 0 &&
                   argument + 1 < argc && request.output_root == NULL) {
            request.output_root = argv[++argument];
        } else if (strcmp(argv[argument], "--code-commit") == 0 &&
                   argument + 1 < argc && request.code_commit == NULL) {
            request.code_commit = argv[++argument];
        } else if (argv[argument][0] == '-' || positional_seen) {
            kmu_usage(argv[0]);
            return 2;
        } else {
            request.repository_root = argv[argument];
            positional_seen = 1;
        }
    }

    if ((!validate_only_seen && !fit_seen) ||
        scope_text == NULL || profile_text == NULL ||
        request.repository_root == NULL ||
        (validate_only_seen &&
         (request.output_root != NULL || request.code_commit != NULL)) ||
        (fit_seen &&
         (request.output_root == NULL || request.code_commit == NULL))) {
        kmu_usage(argv[0]);
        return 2;
    }
    request.scope = strcmp(scope_text, "full") == 0
        ? KMU_SCOPE_FULL
        : strcmp(scope_text, "matched-subset1000") == 0
            ? KMU_SCOPE_MATCHED_SUBSET1000 : KMU_SCOPE_NONE;
    request.source = kmu_find_source(profile_text);
    if (validate_only_seen) {
        return kmu_validate_request(&request) ? 0 : 1;
    }
    return kmu_fit_request(&request) ? 0 : 1;
}
#endif
