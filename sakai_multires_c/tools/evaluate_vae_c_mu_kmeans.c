#define _CRT_SECURE_NO_WARNINGS

#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include "baseline_clustering_metrics.h"
#include "vae_c_mu_contract.h"
#include "vae_event_dataset.h"
#include "vae_saved_split.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <locale.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
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
    EVK_LABEL_COUNT = 22,
    EVK_MANIFEST_ROWS = 8583,
    EVK_FULL_TRAIN_ROWS = 6866,
    EVK_FULL_VALIDATION_ROWS = 1717,
    EVK_MATCHED_TRAIN_ROWS = 800,
    EVK_MATCHED_VALIDATION_ROWS = 200,
    EVK_PATH_CAP = 4096,
    EVK_LINE_CAP = 8192,
    EVK_EVENT_ID_CAP = 128,
    EVK_OUTPUT_FILE_COUNT = 13,
    EVK_HASHED_ARTIFACT_COUNT = 12,
    EVK_SOURCE_FILE_COUNT = 6,
    EVK_CANONICAL_FILE_COUNT = 4,
    EVK_CODE_FILE_COUNT = 3,
    EVK_SHA_RECORD_COUNT = 25,
    EVK_METADATA_LINE_COUNT = 71,
    EVK_METADATA_KEY_CAP = 128,
    EVK_WANDERING_LINE_INDEX = 20,
    EVK_EVALUATION_OUTPUT_FILE_COUNT = 7,
    EVK_EVALUATION_ARTIFACT_COUNT = 6,
    EVK_EVALUATION_SOURCE_FILE_COUNT = 6,
    EVK_EVALUATION_CANONICAL_FILE_COUNT = 5,
    EVK_EVALUATION_CODE_FILE_COUNT = 7,
    EVK_EVALUATION_SHA_RECORD_COUNT = 24,
    EVK_EVALUATION_METADATA_LINE_COUNT = 45
};


#define EVK_OUTPUT_ROOT \
    "sakai_multires_c/results/vae_c/kmeans_mu_raw_v1"
#define EVK_EVALUATION_FORMAT \
    "VAE_C_MU_KMEANS_LABEL_EVALUATION_V1"
#define EVK_MANIFEST_PATH "manifest_events_272.csv"
#define EVK_TRAIN_SPLIT_PATH "split_train_indices_seed123.csv"
#define EVK_VALIDATION_SPLIT_PATH "split_val_indices_seed123.csv"
#define EVK_TRAIN_SUBSET_PATH \
    "sakai_multires_c/configs/subset224_train_800_seed123.csv"
#define EVK_VALIDATION_SUBSET_PATH \
    "sakai_multires_c/configs/subset224_val_200_seed123.csv"

#define EVK_MANIFEST_SHA256 \
    "620960e6b35da31bc4cf1a8a6da8679027d378e1d4b2ac73c5267c1c5311ae82"
#define EVK_TRAIN_SPLIT_SHA256 \
    "8c7a513b98dbe996261bc80965500ab15ab77f498ec9440858003152e42f926f"
#define EVK_VALIDATION_SPLIT_SHA256 \
    "e63109c88df7b3ee97317f94791f26df3fcaf8b937921cfb27967d01c2546467"
#define EVK_TRAIN_SUBSET_SHA256 \
    "80c5396a70e7f2c8e3a00680582945581ae76c301a5b9ba8c62e006bb25cdfb5"
#define EVK_VALIDATION_SUBSET_SHA256 \
    "9f5fe6c50c7398588f2de7bd4a543a54a53e7f7a0b9a84bf4004a01f065ae76f"
#define EVK_PRODUCER_SHA256 \
    "efae4e00b8c9155a41608e5b8688b89b5cca28c1b04dbb11688d0af469b4cde5"
#define EVK_KMEANS_SOURCE_SHA256 \
    "987ac13f792fe21139baa10c02e4356147af83bc34203c0090ac3379e58a2eee"
#define EVK_KMEANS_HEADER_SHA256 \
    "5d60d0621b3866ce74e3fccf68361917f4374c74fe2a2ed58450c9978c2f61c8"


typedef enum {
    EVK_SCOPE_NONE = 0,
    EVK_SCOPE_FULL,
    EVK_SCOPE_MATCHED_SUBSET1000
} EVKScope;


typedef enum {
    EVK_MODE_NONE = 0,
    EVK_MODE_VALIDATE_ONLY,
    EVK_MODE_EVALUATE
} EVKMode;


typedef struct {
    const char *token;
    const char *profile_id;
    const char *source_bundle;
    size_t latent_dim;
    size_t source_train_rows;
    size_t source_validation_rows;
    int supports_full;
    const char *source_names[EVK_SOURCE_FILE_COUNT];
    const char *source_hashes[EVK_SOURCE_FILE_COUNT];
} EVKProfile;


typedef struct {
    uint64_t split_pos;
    uint64_t manifest_index;
    uint64_t event_index;
    char event_id[EVK_EVENT_ID_CAP];
    char split[16];
} EVKExpectedRow;


typedef struct {
    size_t rows;
    size_t label_counts[EVK_LABEL_COUNT];
    size_t used_clusters;
} EVKPopulationReport;


typedef struct {
    EVKMode mode;
    EVKScope scope;
    const EVKProfile *profile;
    size_t k;
    const char *output_root;
    const char *code_commit;
    const char *repository_root;
} EVKRequest;


typedef struct {
    GSEventManifest manifest;
    EVKExpectedRow *train_rows;
    EVKExpectedRow *validation_rows;
    uint32_t *train_assignments;
    uint32_t *validation_assignments;
    uint32_t *train_labels;
    uint32_t *validation_labels;
    size_t train_count;
    size_t validation_count;
    EVKPopulationReport train_report;
    EVKPopulationReport validation_report;
    char source_directory[EVK_PATH_CAP];
} EVKValidatedData;


typedef struct {
    GSBaselineClusteringMetrics train;
    GSBaselineClusteringMetrics validation;
    size_t compute_calls;
} EVKMetricsPair;


typedef struct {
    char source[EVK_EVALUATION_SOURCE_FILE_COUNT][65];
    char canonical[EVK_EVALUATION_CANONICAL_FILE_COUNT][65];
    char code[EVK_EVALUATION_CODE_FILE_COUNT][65];
} EVKProvenanceHashes;


static const char *const evk_labels[EVK_LABEL_COUNT] = {
    "1080Lines",
    "1400Ripples",
    "Air_Compressor",
    "Blip",
    "Chirp",
    "Extremely_Loud",
    "Helix",
    "Koi_Fish",
    "Light_Modulation",
    "Low_Frequency_Burst",
    "Low_Frequency_Lines",
    "No_Glitch",
    "None_of_the_Above",
    "Paired_Doves",
    "Power_Line",
    "Repeating_Blips",
    "Scattered_Light",
    "Scratchy",
    "Tomte",
    "Violin_Mode",
    "Wandering_Line",
    "Whistle"
};


static const char *const evk_output_names[EVK_OUTPUT_FILE_COUNT] = {
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


static const char *const
evk_evaluation_output_names[EVK_EVALUATION_OUTPUT_FILE_COUNT] = {
    "label_map.csv",
    "metrics.csv",
    "train_contingency_counts.csv",
    "validation_contingency_counts.csv",
    "evaluator_source.c",
    "metadata.txt",
    "sha256.txt"
};


static const char *const
evk_evaluation_source_names[EVK_EVALUATION_SOURCE_FILE_COUNT] = {
    "metadata.txt",
    "sha256.txt",
    "train_assignments.csv",
    "validation_assignments.csv",
    "train_assignments_u32.bin",
    "validation_assignments_u32.bin"
};


static const char *const
evk_evaluation_canonical_names[EVK_EVALUATION_CANONICAL_FILE_COUNT] = {
    EVK_MANIFEST_PATH,
    EVK_TRAIN_SPLIT_PATH,
    EVK_VALIDATION_SPLIT_PATH,
    EVK_TRAIN_SUBSET_PATH,
    EVK_VALIDATION_SUBSET_PATH
};


static const char *const
evk_evaluation_canonical_hashes[EVK_EVALUATION_CANONICAL_FILE_COUNT] = {
    EVK_MANIFEST_SHA256,
    EVK_TRAIN_SPLIT_SHA256,
    EVK_VALIDATION_SPLIT_SHA256,
    EVK_TRAIN_SUBSET_SHA256,
    EVK_VALIDATION_SUBSET_SHA256
};


static const char *const
evk_evaluation_code_names[EVK_EVALUATION_CODE_FILE_COUNT] = {
    "sakai_multires_c/tools/evaluate_vae_c_mu_kmeans.c",
    "sakai_multires_c/src/baseline_clustering_metrics.c",
    "sakai_multires_c/include/baseline_clustering_metrics.h",
    "vae_event_dataset.c",
    "vae_event_dataset.h",
    "vae_saved_split.c",
    "vae_saved_split.h"
};


static const EVKProfile evk_profiles[] = {
    {
        "mini75", "mini75_experiment_v1",
        "sakai_multires_c/results/vae_c/embeddings/"
        "mini75_experiment_v1_full_best_epoch47_posterior_mu_central_v1",
        32u, EVK_FULL_TRAIN_ROWS, EVK_FULL_VALIDATION_ROWS, 1,
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
        "mini112", "mini112_primary_v1",
        "sakai_multires_c/results/vae_c/embeddings/"
        "mini112_primary_v1_full_best_epoch97_posterior_mu_central_v1",
        64u, EVK_FULL_TRAIN_ROWS, EVK_FULL_VALIDATION_ROWS, 1,
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
        "mini224", "mini224_bridge_v1",
        "sakai_multires_c/results/vae_c/embeddings/"
        "mini224_bridge_v1_subset1000_best_epoch99_posterior_mu_central_v1",
        64u, EVK_MATCHED_TRAIN_ROWS, EVK_MATCHED_VALIDATION_ROWS, 0,
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
        "full224", "sakai224_full_v1", "sakai224_full_mu_epoch100_v1",
        512u, EVK_FULL_TRAIN_ROWS, EVK_FULL_VALIDATION_ROWS, 1,
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


static int evk_error(const char *format, ...)
{
    va_list arguments;

    fputs("ERROR: ", stderr);
    va_start(arguments, format);
    vfprintf(stderr, format, arguments);
    va_end(arguments);
    fputc('\n', stderr);
    return 0;
}


static int evk_path_join(
    const char *base,
    const char *name,
    char output[EVK_PATH_CAP]
)
{
    size_t length;
    int written;

    if (base == NULL || name == NULL || output == NULL || base[0] == '\0' ||
        name[0] == '\0') {
        return 0;
    }
    length = strlen(base);
    written = snprintf(
        output, EVK_PATH_CAP, "%s%s%s", base,
        (base[length - 1u] == '/' || base[length - 1u] == '\\') ? "" : "/",
        name
    );
    return written >= 0 && (size_t)written < EVK_PATH_CAP;
}


static void evk_strip_line(char *line)
{
    size_t length;

    if (line == NULL) {
        return;
    }
    length = strlen(line);
    while (length > 0u &&
           (line[length - 1u] == '\r' || line[length - 1u] == '\n')) {
        line[--length] = '\0';
    }
}


static int evk_parse_u64(const char *text, uint64_t *value)
{
    const unsigned char *cursor;
    char *end = NULL;
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


static int evk_parse_finite_nonnegative_double(
    const char *text,
    double *value
)
{
    char *end = NULL;
    double parsed;

    if (text == NULL || value == NULL || text[0] == '\0') {
        return 0;
    }
    errno = 0;
    parsed = strtod(text, &end);
    if (errno == ERANGE || end == text || *end != '\0' ||
        !isfinite(parsed) || parsed < 0.0) {
        return 0;
    }
    *value = parsed;
    return 1;
}


static size_t evk_split_csv(char *line, char **fields, size_t capacity)
{
    char *read_cursor;
    char *write_cursor;
    size_t count = 0u;

    if (line == NULL || fields == NULL || capacity == 0u) {
        return 0u;
    }
    read_cursor = line;
    write_cursor = line;
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


static int evk_label_index(const char *label, size_t *index_out)
{
    size_t index;

    if (label == NULL || index_out == NULL) {
        return 0;
    }
    for (index = 0u; index < EVK_LABEL_COUNT; index++) {
        if (strcmp(label, evk_labels[index]) == 0) {
            *index_out = index;
            return 1;
        }
    }
    return 0;
}


static int evk_validate_k(size_t k)
{
    return k == 22u || k == 36u;
}


static int evk_hash_is_lower_hex(const char *hash)
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


static int evk_commit_is_lower_hex(const char *commit)
{
    size_t index;

    if (commit == NULL || strlen(commit) != 40u) {
        return 0;
    }
    for (index = 0u; index < 40u; index++) {
        if (!((commit[index] >= '0' && commit[index] <= '9') ||
              (commit[index] >= 'a' && commit[index] <= 'f'))) {
            return 0;
        }
    }
    return 1;
}


static int evk_verify_hash(const char *path, const char *expected)
{
    char actual[VAE_C_CHECKPOINT_SHA256_CAP];
    struct stat status;

    if (path == NULL || expected == NULL) {
        return 0;
    }
#ifdef _WIN32
    if (stat(path, &status) != 0 ||
        (status.st_mode & _S_IFMT) != _S_IFREG) {
        return 0;
    }
#else
    if (lstat(path, &status) != 0 || !S_ISREG(status.st_mode)) {
        return 0;
    }
#endif
    return evk_hash_is_lower_hex(expected) &&
        vae_c_mu_file_sha256(path, actual) == VAE_C_MU_OK &&
        strcmp(actual, expected) == 0;
}


static int evk_file_has_exact_line(const char *path, const char *expected)
{
    char line[EVK_LINE_CAP];
    size_t matches = 0u;
    FILE *file = fopen(path, "rb");
    int failed = 0;

    if (file == NULL) {
        return 0;
    }
    while (fgets(line, sizeof(line), file) != NULL) {
        if (strchr(line, '\n') == NULL && !feof(file)) {
            failed = 1;
            break;
        }
        evk_strip_line(line);
        if (strcmp(line, expected) == 0) {
            matches++;
        }
    }
    if (ferror(file)) {
        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    return !failed && matches == 1u;
}


static int evk_validate_metadata_shape(const char *path)
{
    char keys[128][EVK_METADATA_KEY_CAP];
    char line[EVK_LINE_CAP];
    size_t line_count = 0u;
    FILE *file;
    int failed = 0;
    size_t expected_lines = EVK_METADATA_LINE_COUNT;

    if (path == NULL) {
        return 0;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        return 0;
    }
    while (fgets(line, sizeof(line), file) != NULL) {
        char *equals;
        const unsigned char *cursor;
        size_t key_length;
        size_t prior;

        if (strchr(line, '\n') == NULL && !feof(file)) {
            failed = 1;
            break;
        }
        evk_strip_line(line);
        equals = strchr(line, '=');
        if (line_count >= expected_lines || equals == NULL ||
            equals == line || equals[1] == '\0' || strchr(equals + 1, '=') != NULL) {
            failed = 1;
            break;
        }
        key_length = (size_t)(equals - line);
        if (key_length >= EVK_METADATA_KEY_CAP) {
            failed = 1;
            break;
        }
        for (cursor = (const unsigned char *)line;
             cursor < (const unsigned char *)equals; cursor++) {
            if (!(islower(*cursor) || isdigit(*cursor) || *cursor == '_')) {
                failed = 1;
                break;
            }
        }
        if (failed) {
            break;
        }
        memcpy(keys[line_count], line, key_length);
        keys[line_count][key_length] = '\0';
        for (prior = 0u; prior < line_count; prior++) {
            if (strcmp(keys[prior], keys[line_count]) == 0) {
                failed = 1;
                break;
            }
        }
        if (failed) {
            break;
        }
        line_count++;
    }
    if (ferror(file) || line_count != expected_lines) {
        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    return !failed;
}


static int evk_validate_key_value_shape(
    const char *path,
    size_t expected_lines
)
{
    char keys[128][EVK_METADATA_KEY_CAP];
    char line[EVK_LINE_CAP];
    size_t line_count = 0u;
    FILE *file;
    int failed = 0;

    if (path == NULL || expected_lines == 0u || expected_lines > 128u) {
        return 0;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        return 0;
    }
    while (fgets(line, sizeof(line), file) != NULL) {
        char *equals;
        const unsigned char *cursor;
        size_t key_length;
        size_t prior;

        if (strchr(line, '\n') == NULL && !feof(file)) {
            failed = 1;
            break;
        }
        evk_strip_line(line);
        equals = strchr(line, '=');
        if (line_count >= expected_lines || equals == NULL || equals == line ||
            equals[1] == '\0' || strchr(equals + 1, '=') != NULL) {
            failed = 1;
            break;
        }
        key_length = (size_t)(equals - line);
        if (key_length >= EVK_METADATA_KEY_CAP) {
            failed = 1;
            break;
        }
        for (cursor = (const unsigned char *)line;
             cursor < (const unsigned char *)equals; cursor++) {
            if (!(islower(*cursor) || isdigit(*cursor) || *cursor == '_')) {
                failed = 1;
                break;
            }
        }
        if (failed) {
            break;
        }
        memcpy(keys[line_count], line, key_length);
        keys[line_count][key_length] = '\0';
        for (prior = 0u; prior < line_count; prior++) {
            if (strcmp(keys[prior], keys[line_count]) == 0) {
                failed = 1;
                break;
            }
        }
        if (failed) {
            break;
        }
        line_count++;
    }
    if (ferror(file) || line_count != expected_lines) {
        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    return !failed;
}


static int evk_expect_metadata(
    const char *path,
    const char *key,
    const char *value
)
{
    char expected[EVK_LINE_CAP];
    int written = snprintf(expected, sizeof(expected), "%s=%s", key, value);

    return written >= 0 && (size_t)written < sizeof(expected) &&
        evk_file_has_exact_line(path, expected);
}


static int evk_output_index(const char *name)
{
    size_t index;

    for (index = 0u; index < EVK_OUTPUT_FILE_COUNT; index++) {
        if (strcmp(name, evk_output_names[index]) == 0) {
            return (int)index;
        }
    }
    return -1;
}


static int evk_directory_has_exact_inventory(const char *directory)
{
    unsigned char seen[EVK_OUTPUT_FILE_COUNT] = {0};
    size_t count = 0u;

#ifdef _WIN32
    WIN32_FIND_DATAA entry;
    HANDLE search;
    DWORD find_error;
    char pattern[EVK_PATH_CAP];

    if (!evk_path_join(directory, "*", pattern)) {
        return evk_error("output inventory path overflow");
    }
    search = FindFirstFileA(pattern, &entry);
    if (search == INVALID_HANDLE_VALUE) {
        return evk_error("cannot enumerate output directory: %s", directory);
    }
    do {
        int index;
        if (strcmp(entry.cFileName, ".") == 0 ||
            strcmp(entry.cFileName, "..") == 0) {
            continue;
        }
        index = evk_output_index(entry.cFileName);
        if (index < 0 || seen[index] != 0u ||
            (entry.dwFileAttributes &
             (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)) !=
                0u) {
            FindClose(search);
            return evk_error("unexpected output entry: %s", entry.cFileName);
        }
        seen[index] = 1u;
        count++;
    } while (FindNextFileA(search, &entry) != 0);
    find_error = GetLastError();
    if (FindClose(search) == 0 || find_error != ERROR_NO_MORE_FILES) {
        return evk_error("cannot complete output inventory");
    }
#else
    DIR *stream = opendir(directory);
    struct dirent *entry;

    if (stream == NULL) {
        return evk_error("cannot enumerate output directory: %s", directory);
    }
    for (;;) {
        errno = 0;
        entry = readdir(stream);
        if (entry == NULL) {
            if (errno != 0) {
                closedir(stream);
                return evk_error("cannot complete output inventory");
            }
            break;
        }
        if (strcmp(entry->d_name, ".") != 0 &&
            strcmp(entry->d_name, "..") != 0) {
            char path[EVK_PATH_CAP];
            struct stat status;
            int index = evk_output_index(entry->d_name);
            if (index < 0 || seen[index] != 0u ||
                !evk_path_join(directory, entry->d_name, path) ||
                lstat(path, &status) != 0 || !S_ISREG(status.st_mode)) {
                closedir(stream);
                return evk_error("unexpected output entry: %s", entry->d_name);
            }
            seen[index] = 1u;
            count++;
        }
    }
    if (closedir(stream) != 0) {
        return evk_error("cannot close output inventory");
    }
#endif

    if (count != EVK_OUTPUT_FILE_COUNT) {
        return evk_error("output has %zu files, expected %d", count,
                         EVK_OUTPUT_FILE_COUNT);
    }
    return 1;
}


static int evk_path_absent(const char *path)
{
    if (path == NULL) {
        return 0;
    }
#ifdef _WIN32
    {
        DWORD error;
        DWORD attributes = GetFileAttributesA(path);

        if (attributes != INVALID_FILE_ATTRIBUTES) {
            return 0;
        }
        error = GetLastError();
        return error == ERROR_FILE_NOT_FOUND ||
            error == ERROR_PATH_NOT_FOUND;
    }
#else
    {
        struct stat status;

        errno = 0;
        return lstat(path, &status) != 0 && errno == ENOENT;
    }
#endif
}


static int evk_validate_manifest(const GSEventManifest *manifest)
{
    unsigned char present[EVK_LABEL_COUNT] = {0};
    size_t index;

    if (manifest == NULL || manifest->records == NULL ||
        manifest->count != EVK_MANIFEST_ROWS) {
        return 0;
    }
    for (index = 0u; index < manifest->count; index++) {
        size_t label;
        if (manifest->records[index].event_index != (int)(index + 1u) ||
            manifest->records[index].event_id[0] == '\0' ||
            !evk_label_index(manifest->records[index].label, &label)) {
            return 0;
        }
        present[label] = 1u;
    }
    for (index = 0u; index < EVK_LABEL_COUNT; index++) {
        if (present[index] == 0u) {
            return 0;
        }
    }
    return 1;
}


static int evk_rows_from_split(
    const GSSavedSplit *split,
    const GSEventManifest *manifest,
    const char *split_name,
    EVKExpectedRow **rows_out
)
{
    EVKExpectedRow *rows;
    size_t index;

    *rows_out = NULL;
    if (split == NULL || split->indices == NULL || split->count == 0u ||
        manifest == NULL || manifest->records == NULL ||
        split_name == NULL || split->count > SIZE_MAX / sizeof(*rows)) {
        return 0;
    }
    rows = (EVKExpectedRow *)calloc(split->count, sizeof(*rows));
    if (rows == NULL) {
        return 0;
    }
    for (index = 0u; index < split->count; index++) {
        const GSEventRecord *record;
        size_t manifest_index = split->indices[index];
        int copied;

        if (manifest_index >= manifest->count) {
            free(rows);
            return 0;
        }
        record = &manifest->records[manifest_index];
        rows[index].split_pos = (uint64_t)index;
        rows[index].manifest_index = (uint64_t)manifest_index;
        rows[index].event_index = (uint64_t)record->event_index;
        copied = snprintf(rows[index].event_id, sizeof(rows[index].event_id),
                          "%s", record->event_id);
        if (copied < 0 || (size_t)copied >= sizeof(rows[index].event_id)) {
            free(rows);
            return 0;
        }
        copied = snprintf(rows[index].split, sizeof(rows[index].split), "%s",
                          split_name);
        if (copied < 0 || (size_t)copied >= sizeof(rows[index].split)) {
            free(rows);
            return 0;
        }
    }
    *rows_out = rows;
    return 1;
}


static int evk_validate_assignment_csv(
    const char *path,
    const EVKExpectedRow *expected,
    size_t expected_count,
    const GSEventManifest *manifest,
    const char *expected_split,
    size_t k,
    uint32_t *assignments_out,
    uint32_t *labels_out,
    EVKPopulationReport *report
)
{
    static const char header[] =
        "row_index,split_pos,manifest_index,event_index,event_id,split,"
        "cluster_id,squared_distance";
    unsigned char *used_clusters = NULL;
    char line[EVK_LINE_CAP];
    FILE *file = NULL;
    size_t row = 0u;
    int failed = 0;

    if (path == NULL || expected == NULL || expected_count == 0u ||
        manifest == NULL || manifest->records == NULL ||
        expected_split == NULL || assignments_out == NULL || report == NULL ||
        !evk_validate_k(k)) {
        return 0;
    }
    memset(report, 0, sizeof(*report));
    used_clusters = (unsigned char *)calloc(k, 1u);
    if (used_clusters == NULL) {
        return 0;
    }
    file = fopen(path, "rb");
    if (file == NULL || fgets(line, sizeof(line), file) == NULL) {
        failed = 1;
    } else {
        evk_strip_line(line);
        if (strcmp(line, header) != 0) {
            failed = 1;
        }
    }
    while (!failed && fgets(line, sizeof(line), file) != NULL) {
        char *fields[8];
        uint64_t row_index;
        uint64_t split_pos;
        uint64_t manifest_index;
        uint64_t event_index;
        uint64_t cluster;
        double squared_distance;
        const GSEventRecord *record;
        size_t label;

        if (strchr(line, '\n') == NULL && !feof(file)) {
            failed = 1;
            break;
        }
        evk_strip_line(line);
        if (row >= expected_count ||
            evk_split_csv(line, fields, 8u) != 8u ||
            !evk_parse_u64(fields[0], &row_index) ||
            !evk_parse_u64(fields[1], &split_pos) ||
            !evk_parse_u64(fields[2], &manifest_index) ||
            !evk_parse_u64(fields[3], &event_index) ||
            !evk_parse_u64(fields[6], &cluster) ||
            !evk_parse_finite_nonnegative_double(
                fields[7], &squared_distance
            ) ||
            row_index != (uint64_t)row ||
            split_pos != expected[row].split_pos ||
            manifest_index != expected[row].manifest_index ||
            event_index != expected[row].event_index ||
            strcmp(fields[4], expected[row].event_id) != 0 ||
            strcmp(fields[5], expected_split) != 0 ||
            strcmp(fields[5], expected[row].split) != 0 ||
            manifest_index >= manifest->count || cluster >= (uint64_t)k) {
            failed = 1;
            break;
        }
        record = &manifest->records[(size_t)manifest_index];
        if ((uint64_t)record->event_index != event_index ||
            strcmp(record->event_id, fields[4]) != 0 ||
            !evk_label_index(record->label, &label)) {
            failed = 1;
            break;
        }
        assignments_out[row] = (uint32_t)cluster;
        if (labels_out != NULL) {
            labels_out[row] = (uint32_t)label;
        }
        report->label_counts[label]++;
        if (used_clusters[(size_t)cluster] == 0u) {
            used_clusters[(size_t)cluster] = 1u;
            report->used_clusters++;
        }
        report->rows++;
        row++;
        (void)squared_distance;
    }
    if (file == NULL) {
        failed = 1;
    } else {
        if (ferror(file) || row != expected_count) {
            failed = 1;
        }
        if (fclose(file) != 0) {
            failed = 1;
        }
    }
    free(used_clusters);
    return !failed;
}


static int evk_platform_is_u32_little_endian(void)
{
    uint32_t one = UINT32_C(1);
    const unsigned char *bytes = (const unsigned char *)&one;

    return sizeof(uint32_t) == 4u && bytes[0] == 1u && bytes[1] == 0u &&
        bytes[2] == 0u && bytes[3] == 0u;
}


static int evk_validate_assignment_binary(
    const char *path,
    const uint32_t *assignments,
    size_t rows
)
{
    uint32_t values[1024];
    size_t offset = 0u;
    struct stat status;
    FILE *file;
    int failed = 0;

    if (path == NULL || assignments == NULL || rows == 0u ||
        !evk_platform_is_u32_little_endian() ||
        rows > SIZE_MAX / sizeof(uint32_t) ||
#ifdef _WIN32
        stat(path, &status) != 0 ||
#else
        lstat(path, &status) != 0 ||
#endif
#ifdef _WIN32
        (status.st_mode & _S_IFMT) != _S_IFREG ||
#else
        !S_ISREG(status.st_mode) ||
#endif
        status.st_size < 0 ||
        (uint64_t)status.st_size != (uint64_t)(rows * sizeof(uint32_t))) {
        return 0;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        return 0;
    }
    while (offset < rows) {
        size_t count = rows - offset;
        size_t index;
        if (count > sizeof(values) / sizeof(values[0])) {
            count = sizeof(values) / sizeof(values[0]);
        }
        if (fread(values, sizeof(values[0]), count, file) != count) {
            failed = 1;
            break;
        }
        for (index = 0u; index < count; index++) {
            if (values[index] != assignments[offset + index]) {
                failed = 1;
                break;
            }
        }
        if (failed) {
            break;
        }
        offset += count;
    }
    if (!failed && (fgetc(file) != EOF || ferror(file))) {
        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    return !failed;
}


static const EVKProfile *evk_find_profile(const char *token)
{
    size_t index;

    if (token == NULL) {
        return NULL;
    }
    for (index = 0u; index < sizeof(evk_profiles) / sizeof(evk_profiles[0]);
         index++) {
        if (strcmp(token, evk_profiles[index].token) == 0) {
            return &evk_profiles[index];
        }
    }
    return NULL;
}


static int evk_parse_request(int argc, char **argv, EVKRequest *request)
{
    uint64_t k;
    int evaluate;

    if (argv == NULL || request == NULL ||
        (argc != 9 && argc != 13) ||
        (strcmp(argv[1], "--validate-only") != 0 &&
         strcmp(argv[1], "--evaluate") != 0) ||
        strcmp(argv[2], "--scope") != 0 ||
        strcmp(argv[4], "--profile") != 0 || strcmp(argv[6], "--k") != 0 ||
        !evk_parse_u64(argv[7], &k) || k > SIZE_MAX) {
        return 0;
    }
    evaluate = strcmp(argv[1], "--evaluate") == 0;
    if ((!evaluate && argc != 9) ||
        (evaluate &&
         (argc != 13 || strcmp(argv[8], "--output-root") != 0 ||
          strcmp(argv[10], "--repository-head") != 0 ||
          argv[9][0] == '\0' || !evk_commit_is_lower_hex(argv[11])))) {
        return 0;
    }
    memset(request, 0, sizeof(*request));
    request->mode = evaluate ? EVK_MODE_EVALUATE : EVK_MODE_VALIDATE_ONLY;
    if (strcmp(argv[3], "full") == 0) {
        request->scope = EVK_SCOPE_FULL;
    } else if (strcmp(argv[3], "matched_subset1000") == 0) {
        request->scope = EVK_SCOPE_MATCHED_SUBSET1000;
    } else {
        return 0;
    }
    request->profile = evk_find_profile(argv[5]);
    request->k = (size_t)k;
    request->output_root = evaluate ? argv[9] : NULL;
    request->code_commit = evaluate ? argv[11] : NULL;
    request->repository_root = argv[evaluate ? 12 : 8];
    return request->profile != NULL && evk_validate_k(request->k) &&
        request->repository_root[0] != '\0' &&
        (request->scope != EVK_SCOPE_FULL || request->profile->supports_full);
}


static int evk_make_output_basename(
    const EVKRequest *request,
    char output[256]
)
{
    const char *scope = request->scope == EVK_SCOPE_FULL
        ? "full" : "matched_subset1000";
    int written;

    written = snprintf(
        output, 256u, "%s_%s_k%zu_seed123_maxiter300", scope,
        request->profile->token, request->k
    );
    return written >= 0 && (size_t)written < 256u;
}


static int evk_make_output_directory(
    const EVKRequest *request,
    char output[EVK_PATH_CAP]
)
{
    char root[EVK_PATH_CAP];
    char name[256];

    return evk_path_join(request->repository_root, EVK_OUTPUT_ROOT, root) &&
        evk_make_output_basename(request, name) &&
        evk_path_join(root, name, output);
}


static int evk_validate_metadata(
    const EVKRequest *request,
    const char *output_directory,
    size_t train_rows,
    size_t validation_rows
)
{
    char path[EVK_PATH_CAP];
    char value[EVK_PATH_CAP];
    const char *scope = request->scope == EVK_SCOPE_FULL
        ? "full" : "matched_subset1000";
    const char *population = request->scope == EVK_SCOPE_FULL
        ? "gravity_spy_o1_full_split_seed123_v1"
        : "canonical_prefix_800_train_200_validation_seed123_v1";
    const char *source_format = strcmp(request->profile->token, "full224") == 0
        ? "SAKAI224_FULL_MU_ARTIFACT_V1" : "VAE_C_MU_ARTIFACT_V1";
    size_t index;

    if (!evk_path_join(output_directory, "metadata.txt", path) ||
        !evk_validate_metadata_shape(path)) {
        return evk_error("metadata structure mismatch");
    }
    if (!evk_expect_metadata(path, "format", "VAE_C_MU_KMEANS_ARTIFACT_V1") ||
        !evk_expect_metadata(path, "scope", scope) ||
        !evk_expect_metadata(path, "profile", request->profile->token) ||
        !evk_expect_metadata(path, "profile_id", request->profile->profile_id) ||
        !evk_expect_metadata(path, "source_format", source_format) ||
        !evk_expect_metadata(path, "source_bundle", request->profile->source_bundle) ||
        !evk_expect_metadata(path, "representation", "posterior_mu") ||
        !evk_expect_metadata(path, "view", "central") ||
        !evk_expect_metadata(path, "input_dtype", "float32_ieee754") ||
        !evk_expect_metadata(path, "input_endian", "little") ||
        !evk_expect_metadata(path, "input_layout", "row_major_[N,latent_dim]") ||
        !evk_expect_metadata(path, "population_identity", population) ||
        !evk_expect_metadata(path, "row_order", "native_split_pos_ascending") ||
        !evk_expect_metadata(
            path, "row_identity",
            "split_pos,manifest_index,event_index,event_id,split"
        ) ||
        !evk_expect_metadata(path, "algorithm", "kmeans_plus_plus_lloyd") ||
        !evk_expect_metadata(path, "backend", "gs_baseline_kmeans_fit") ||
        !evk_expect_metadata(path, "input", "raw_posterior_mu") ||
        !evk_expect_metadata(path, "fit_input", "train_only") ||
        !evk_expect_metadata(
            path, "validation_policy", "assign_to_frozen_train_centroids"
        ) ||
        !evk_expect_metadata(
            path, "matched_refit",
            request->scope == EVK_SCOPE_FULL ? "false" : "true"
        ) ||
        !evk_expect_metadata(path, "assignment_slicing", "false") ||
        !evk_expect_metadata(path, "seed", "123") ||
        !evk_expect_metadata(path, "initializations", "1") ||
        !evk_expect_metadata(path, "max_iterations", "300") ||
        !evk_expect_metadata(path, "converged", "1") ||
        !evk_expect_metadata(path, "raw_mu", "true") ||
        !evk_expect_metadata(path, "pca_applied", "false") ||
        !evk_expect_metadata(path, "scaling_applied", "false") ||
        !evk_expect_metadata(path, "standardization_applied", "false") ||
        !evk_expect_metadata(path, "labels_used", "false") ||
        !evk_expect_metadata(path, "label_values_exported", "false") ||
        !evk_expect_metadata(
            path, "nonconverged_publication_allowed", "false"
        ) ||
        !evk_expect_metadata(path, "runner_source_sha256", EVK_PRODUCER_SHA256) ||
        !evk_expect_metadata(
            path, "kmeans_core_source_sha256", EVK_KMEANS_SOURCE_SHA256
        ) ||
        !evk_expect_metadata(
            path, "kmeans_core_header_sha256", EVK_KMEANS_HEADER_SHA256
        ) ||
        !evk_expect_metadata(
            path, "full_train_split_sha256", EVK_TRAIN_SPLIT_SHA256
        ) ||
        !evk_expect_metadata(
            path, "full_validation_split_sha256",
            EVK_VALIDATION_SPLIT_SHA256
        ) ||
        !evk_expect_metadata(
            path, "matched_train_subset_sha256", EVK_TRAIN_SUBSET_SHA256
        ) ||
        !evk_expect_metadata(
            path, "matched_validation_subset_sha256",
            EVK_VALIDATION_SUBSET_SHA256
        )) {
        return evk_error("metadata contract mismatch");
    }

    snprintf(value, sizeof(value), "%zu", request->profile->latent_dim);
    if (!evk_expect_metadata(path, "latent_dim", value)) {
        return evk_error("metadata latent_dim mismatch");
    }
    snprintf(value, sizeof(value), "%zu", request->profile->source_train_rows);
    if (!evk_expect_metadata(path, "source_train_rows", value)) {
        return evk_error("metadata source train rows mismatch");
    }
    snprintf(value, sizeof(value), "%zu",
             request->profile->source_validation_rows);
    if (!evk_expect_metadata(path, "source_validation_rows", value)) {
        return evk_error("metadata source validation rows mismatch");
    }
    snprintf(value, sizeof(value), "%zu", train_rows);
    if (!evk_expect_metadata(path, "selected_train_rows", value)) {
        return evk_error("metadata selected train rows mismatch");
    }
    snprintf(value, sizeof(value), "%zu", validation_rows);
    if (!evk_expect_metadata(path, "selected_validation_rows", value)) {
        return evk_error("metadata selected validation rows mismatch");
    }
    snprintf(value, sizeof(value), "%zu", request->k);
    if (!evk_expect_metadata(path, "k", value)) {
        return evk_error("metadata k mismatch");
    }
    for (index = 0u; index < EVK_SOURCE_FILE_COUNT; index++) {
        static const char *const keys[EVK_SOURCE_FILE_COUNT] = {
            "source_metadata_sha256", "source_checksum_sha256",
            "source_train_mu_sha256", "source_validation_mu_sha256",
            "source_train_rows_sha256", "source_validation_rows_sha256"
        };
        if (!evk_expect_metadata(
                path, keys[index], request->profile->source_hashes[index]
            )) {
            return evk_error("metadata source hash mismatch: %s", keys[index]);
        }
    }
    return 1;
}


static int evk_expected_category_entry(
    const EVKRequest *request,
    const char *category,
    const char *name,
    const char *hash,
    const char *output_directory,
    unsigned char artifact_seen[EVK_HASHED_ARTIFACT_COUNT],
    unsigned char source_seen[EVK_SOURCE_FILE_COUNT],
    unsigned char canonical_seen[EVK_CANONICAL_FILE_COUNT],
    unsigned char code_seen[EVK_CODE_FILE_COUNT]
)
{
    size_t index;
    char path[EVK_PATH_CAP];

    if (strcmp(category, "ARTIFACT") == 0) {
        for (index = 0u; index < EVK_HASHED_ARTIFACT_COUNT; index++) {
            if (strcmp(name, evk_output_names[index]) == 0) {
                if (artifact_seen[index] != 0u ||
                    !evk_path_join(output_directory, name, path) ||
                    !evk_verify_hash(path, hash)) {
                    return 0;
                }
                artifact_seen[index] = 1u;
                return 1;
            }
        }
        return 0;
    }

    if (strcmp(category, "SOURCE") == 0) {
        for (index = 0u; index < EVK_SOURCE_FILE_COUNT; index++) {
            if (strcmp(name, request->profile->source_names[index]) == 0) {
                if (source_seen[index] != 0u ||
                    strcmp(hash, request->profile->source_hashes[index]) != 0) {
                    return 0;
                }
                source_seen[index] = 1u;
                return 1;
            }
        }
        return 0;
    }

    if (strcmp(category, "CANONICAL") == 0) {
        static const char *const names[EVK_CANONICAL_FILE_COUNT] = {
            EVK_TRAIN_SPLIT_PATH,
            EVK_VALIDATION_SPLIT_PATH,
            EVK_TRAIN_SUBSET_PATH,
            EVK_VALIDATION_SUBSET_PATH
        };
        static const char *const hashes[EVK_CANONICAL_FILE_COUNT] = {
            EVK_TRAIN_SPLIT_SHA256,
            EVK_VALIDATION_SPLIT_SHA256,
            EVK_TRAIN_SUBSET_SHA256,
            EVK_VALIDATION_SUBSET_SHA256
        };
        for (index = 0u; index < EVK_CANONICAL_FILE_COUNT; index++) {
            if (strcmp(name, names[index]) == 0) {
                if (canonical_seen[index] != 0u ||
                    strcmp(hash, hashes[index]) != 0 ||
                    !evk_path_join(request->repository_root, names[index], path) ||
                    !evk_verify_hash(path, hashes[index])) {
                    return 0;
                }
                canonical_seen[index] = 1u;
                return 1;
            }
        }
        return 0;
    }

    if (strcmp(category, "CODE") == 0) {
        static const char *const names[EVK_CODE_FILE_COUNT] = {
            "sakai_multires_c/tools/run_vae_c_mu_kmeans.c",
            "sakai_multires_c/src/baseline_kmeans.c",
            "sakai_multires_c/include/baseline_kmeans.h"
        };
        static const char *const actual_paths[EVK_CODE_FILE_COUNT] = {
            "sakai_multires_c/docs/provenance/"
            "run_vae_c_mu_kmeans_producer_efae4e.c",
            "sakai_multires_c/src/baseline_kmeans.c",
            "sakai_multires_c/include/baseline_kmeans.h"
        };
        static const char *const hashes[EVK_CODE_FILE_COUNT] = {
            EVK_PRODUCER_SHA256,
            EVK_KMEANS_SOURCE_SHA256,
            EVK_KMEANS_HEADER_SHA256
        };
        for (index = 0u; index < EVK_CODE_FILE_COUNT; index++) {
            if (strcmp(name, names[index]) == 0) {
                if (code_seen[index] != 0u || strcmp(hash, hashes[index]) != 0 ||
                    !evk_path_join(
                        request->repository_root, actual_paths[index], path
                    ) || !evk_verify_hash(path, hashes[index])) {
                    return 0;
                }
                code_seen[index] = 1u;
                return 1;
            }
        }
        return 0;
    }
    return 0;
}


static int evk_validate_sha_manifest(
    const EVKRequest *request,
    const char *output_directory
)
{
    unsigned char artifact_seen[EVK_HASHED_ARTIFACT_COUNT] = {0};
    unsigned char source_seen[EVK_SOURCE_FILE_COUNT] = {0};
    unsigned char canonical_seen[EVK_CANONICAL_FILE_COUNT] = {0};
    unsigned char code_seen[EVK_CODE_FILE_COUNT] = {0};
    char path[EVK_PATH_CAP];
    char line[EVK_LINE_CAP];
    size_t count = 0u;
    FILE *file;
    int failed = 0;

    if (!evk_path_join(output_directory, "sha256.txt", path)) {
        return evk_error("checksum manifest path overflow");
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        return evk_error("cannot open output checksum manifest");
    }
    while (fgets(line, sizeof(line), file) != NULL) {
        char *category;
        char *hash;
        char *name;
        char *space;

        if (strchr(line, '\n') == NULL && !feof(file)) {
            failed = 1;
            break;
        }
        evk_strip_line(line);
        category = line;
        space = strchr(category, ' ');
        if (space == NULL) {
            failed = 1;
            break;
        }
        *space = '\0';
        hash = space + 1;
        space = strchr(hash, ' ');
        if (space == NULL) {
            failed = 1;
            break;
        }
        *space = '\0';
        name = space + 1;
        if (name[0] == '\0' || strchr(name, ' ') != NULL ||
            !evk_hash_is_lower_hex(hash) ||
            !evk_expected_category_entry(
                request, category, name, hash, output_directory,
                artifact_seen, source_seen, canonical_seen, code_seen
            )) {
            failed = 1;
            break;
        }
        count++;
    }
    if (ferror(file) || count != EVK_SHA_RECORD_COUNT) {
        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    if (failed) {
        return evk_error("output checksum manifest mismatch");
    }
    return 1;
}


static int evk_load_population(
    const EVKRequest *request,
    GSEventManifest *manifest,
    EVKExpectedRow **train_rows,
    EVKExpectedRow **validation_rows,
    size_t *train_count,
    size_t *validation_count
)
{
    GSSavedSplit full_train = {0};
    GSSavedSplit full_validation = {0};
    GSSavedSplit matched_train = {0};
    GSSavedSplit matched_validation = {0};
    char manifest_path[EVK_PATH_CAP];
    char full_train_path[EVK_PATH_CAP];
    char full_validation_path[EVK_PATH_CAP];
    char matched_train_path[EVK_PATH_CAP];
    char matched_validation_path[EVK_PATH_CAP];
    const GSSavedSplit *selected_train;
    const GSSavedSplit *selected_validation;
    size_t index;
    int ok = 0;

    memset(manifest, 0, sizeof(*manifest));
    *train_rows = NULL;
    *validation_rows = NULL;
    if (!evk_path_join(request->repository_root, EVK_MANIFEST_PATH,
                       manifest_path) ||
        !evk_path_join(request->repository_root, EVK_TRAIN_SPLIT_PATH,
                       full_train_path) ||
        !evk_path_join(request->repository_root, EVK_VALIDATION_SPLIT_PATH,
                       full_validation_path) ||
        !evk_path_join(request->repository_root, EVK_TRAIN_SUBSET_PATH,
                       matched_train_path) ||
        !evk_path_join(request->repository_root, EVK_VALIDATION_SUBSET_PATH,
                       matched_validation_path) ||
        !evk_verify_hash(manifest_path, EVK_MANIFEST_SHA256) ||
        !evk_verify_hash(full_train_path, EVK_TRAIN_SPLIT_SHA256) ||
        !evk_verify_hash(full_validation_path, EVK_VALIDATION_SPLIT_SHA256) ||
        !evk_verify_hash(matched_train_path, EVK_TRAIN_SUBSET_SHA256) ||
        !evk_verify_hash(matched_validation_path,
                         EVK_VALIDATION_SUBSET_SHA256) ||
        gs_event_manifest_load(manifest_path, manifest) != 0 ||
        !evk_validate_manifest(manifest) ||
        gs_saved_split_load(
            full_train_path, "train", manifest, &full_train
        ) != 0 ||
        gs_saved_split_load(
            full_validation_path, "validation", manifest, &full_validation
        ) != 0 ||
        gs_saved_split_load(
            matched_train_path, "train", manifest, &matched_train
        ) != 0 ||
        gs_saved_split_load(
            matched_validation_path, "validation", manifest,
            &matched_validation
        ) != 0 ||
        full_train.count != EVK_FULL_TRAIN_ROWS ||
        full_validation.count != EVK_FULL_VALIDATION_ROWS ||
        matched_train.count != EVK_MATCHED_TRAIN_ROWS ||
        matched_validation.count != EVK_MATCHED_VALIDATION_ROWS) {
        goto cleanup;
    }
    for (index = 0u; index < matched_train.count; index++) {
        if (matched_train.indices[index] != full_train.indices[index]) {
            goto cleanup;
        }
    }
    for (index = 0u; index < matched_validation.count; index++) {
        if (matched_validation.indices[index] != full_validation.indices[index]) {
            goto cleanup;
        }
    }
    selected_train = request->scope == EVK_SCOPE_FULL
        ? &full_train : &matched_train;
    selected_validation = request->scope == EVK_SCOPE_FULL
        ? &full_validation : &matched_validation;
    *train_count = selected_train->count;
    *validation_count = selected_validation->count;
    if (!evk_rows_from_split(
            selected_train, manifest, "train", train_rows
        ) ||
        !evk_rows_from_split(
            selected_validation, manifest, "validation", validation_rows
        )) {
        free(*train_rows);
        free(*validation_rows);
        *train_rows = NULL;
        *validation_rows = NULL;
        goto cleanup;
    }
    ok = 1;

cleanup:
    gs_saved_split_free(&matched_validation);
    gs_saved_split_free(&matched_train);
    gs_saved_split_free(&full_validation);
    gs_saved_split_free(&full_train);
    if (!ok) {
        gs_event_manifest_free(manifest);
    }
    return ok;
}


static void evk_metrics_pair_free(EVKMetricsPair *pair)
{
    if (pair == NULL) {
        return;
    }
    gs_baseline_clustering_metrics_free(&pair->validation);
    gs_baseline_clustering_metrics_free(&pair->train);
}


static int evk_metric_unit_interval(double value)
{
    const double tolerance = 1.0e-12;

    return isfinite(value) && value >= -tolerance &&
        value <= 1.0 + tolerance;
}


static int evk_metrics_valid(
    const GSBaselineClusteringMetrics *metrics,
    size_t expected_rows,
    size_t expected_k
)
{
    uint64_t grand_total = 0u;
    uint64_t label_total_sum = 0u;
    uint64_t cluster_total_sum = 0u;
    size_t label;
    size_t cluster;

    if (metrics == NULL || expected_rows == 0u ||
        !evk_validate_k(expected_k) || metrics->rows != expected_rows ||
        metrics->label_count != EVK_LABEL_COUNT ||
        metrics->cluster_count != expected_k || metrics->contingency == NULL ||
        metrics->label_totals == NULL || metrics->cluster_totals == NULL ||
        !evk_metric_unit_interval(metrics->purity) ||
        !evk_metric_unit_interval(metrics->normalized_mutual_information) ||
        !isfinite(metrics->adjusted_rand_index) ||
        metrics->adjusted_rand_index < -1.0 - 1.0e-12 ||
        metrics->adjusted_rand_index > 1.0 + 1.0e-12 ||
        !evk_metric_unit_interval(metrics->homogeneity) ||
        !evk_metric_unit_interval(metrics->completeness)) {
        return 0;
    }
    for (label = 0u; label < EVK_LABEL_COUNT; label++) {
        uint64_t row_total = 0u;
        for (cluster = 0u; cluster < expected_k; cluster++) {
            uint64_t value =
                metrics->contingency[label * expected_k + cluster];
            if (UINT64_MAX - row_total < value ||
                UINT64_MAX - grand_total < value) {
                return 0;
            }
            row_total += value;
            grand_total += value;
        }
        if (row_total != metrics->label_totals[label] ||
            UINT64_MAX - label_total_sum < row_total) {
            return 0;
        }
        label_total_sum += row_total;
    }
    for (cluster = 0u; cluster < expected_k; cluster++) {
        uint64_t column_total = 0u;
        for (label = 0u; label < EVK_LABEL_COUNT; label++) {
            uint64_t value =
                metrics->contingency[label * expected_k + cluster];
            if (UINT64_MAX - column_total < value) {
                return 0;
            }
            column_total += value;
        }
        if (column_total != metrics->cluster_totals[cluster] ||
            UINT64_MAX - cluster_total_sum < column_total) {
            return 0;
        }
        cluster_total_sum += column_total;
    }
    return grand_total == (uint64_t)expected_rows &&
        label_total_sum == (uint64_t)expected_rows &&
        cluster_total_sum == (uint64_t)expected_rows;
}


static int evk_compute_metrics(
    const uint32_t *train_labels,
    const uint32_t *train_assignments,
    size_t train_rows,
    const uint32_t *validation_labels,
    const uint32_t *validation_assignments,
    size_t validation_rows,
    size_t k,
    EVKScope scope,
    EVKMetricsPair *pair
)
{
    uint32_t *train_labels_copy = NULL;
    uint32_t *train_assignments_copy = NULL;
    uint32_t *validation_labels_copy = NULL;
    uint32_t *validation_assignments_copy = NULL;
    size_t cluster;
    int passed = 0;

    if (train_labels == NULL || train_assignments == NULL ||
        validation_labels == NULL || validation_assignments == NULL ||
        pair == NULL || !evk_validate_k(k) || train_rows == 0u ||
        validation_rows == 0u || train_rows > SIZE_MAX / sizeof(uint32_t) ||
        validation_rows > SIZE_MAX / sizeof(uint32_t)) {
        return 0;
    }
    memset(pair, 0, sizeof(*pair));
    train_labels_copy = (uint32_t *)malloc(train_rows * sizeof(uint32_t));
    train_assignments_copy =
        (uint32_t *)malloc(train_rows * sizeof(uint32_t));
    validation_labels_copy =
        (uint32_t *)malloc(validation_rows * sizeof(uint32_t));
    validation_assignments_copy =
        (uint32_t *)malloc(validation_rows * sizeof(uint32_t));
    if (train_labels_copy == NULL || train_assignments_copy == NULL ||
        validation_labels_copy == NULL ||
        validation_assignments_copy == NULL) {
        evk_error("metric input snapshot allocation failed");
        goto cleanup;
    }
    memcpy(train_labels_copy, train_labels, train_rows * sizeof(uint32_t));
    memcpy(
        train_assignments_copy, train_assignments,
        train_rows * sizeof(uint32_t)
    );
    memcpy(
        validation_labels_copy, validation_labels,
        validation_rows * sizeof(uint32_t)
    );
    memcpy(
        validation_assignments_copy, validation_assignments,
        validation_rows * sizeof(uint32_t)
    );
    pair->compute_calls++;
    if (gs_baseline_clustering_metrics_compute(
            train_labels, train_assignments, train_rows, EVK_LABEL_COUNT, k,
            &pair->train
        ) != GS_BASELINE_CLUSTERING_METRICS_SUCCESS) {
        evk_error("train metric computation failed");
        goto cleanup;
    }
    pair->compute_calls++;
    if (gs_baseline_clustering_metrics_compute(
            validation_labels, validation_assignments, validation_rows,
            EVK_LABEL_COUNT, k, &pair->validation
        ) != GS_BASELINE_CLUSTERING_METRICS_SUCCESS ||
        !evk_metrics_valid(&pair->train, train_rows, k) ||
        !evk_metrics_valid(&pair->validation, validation_rows, k)) {
        evk_error("computed metric contract mismatch");
        goto cleanup;
    }
    if (memcmp(
            train_labels_copy, train_labels,
            train_rows * sizeof(uint32_t)
        ) != 0 || memcmp(
            train_assignments_copy, train_assignments,
            train_rows * sizeof(uint32_t)
        ) != 0 || memcmp(
            validation_labels_copy, validation_labels,
            validation_rows * sizeof(uint32_t)
        ) != 0 || memcmp(
            validation_assignments_copy, validation_assignments,
            validation_rows * sizeof(uint32_t)
        ) != 0) {
        evk_error("metrics backend modified an input array");
        goto cleanup;
    }
    if (scope == EVK_SCOPE_MATCHED_SUBSET1000) {
        for (cluster = 0u; cluster < k; cluster++) {
            if (pair->validation.contingency[
                    EVK_WANDERING_LINE_INDEX * k + cluster
                ] != 0u) {
                evk_error(
                    "matched validation Wandering_Line row is not zero"
                );
                goto cleanup;
            }
        }
    }
    passed = pair->compute_calls == 2u;

cleanup:
    free(validation_assignments_copy);
    free(validation_labels_copy);
    free(train_assignments_copy);
    free(train_labels_copy);
    if (!passed) {
        evk_metrics_pair_free(pair);
    }
    return passed;
}


static void evk_validated_data_free(EVKValidatedData *data)
{
    if (data == NULL) {
        return;
    }
    free(data->validation_labels);
    free(data->train_labels);
    free(data->validation_assignments);
    free(data->train_assignments);
    free(data->validation_rows);
    free(data->train_rows);
    gs_event_manifest_free(&data->manifest);
    memset(data, 0, sizeof(*data));
}


static int evk_load_validated_data(
    const EVKRequest *request,
    EVKValidatedData *data
)
{
    char path[EVK_PATH_CAP];
    char partial_path[EVK_PATH_CAP];
    int written;
    int ok = 0;

    if (request == NULL || data == NULL) {
        return 0;
    }
    memset(data, 0, sizeof(*data));
    if (!evk_make_output_directory(request, data->source_directory)) {
        return evk_error("output directory path overflow");
    }
    written = snprintf(partial_path, sizeof(partial_path), "%s.partial",
                       data->source_directory);
    if (written < 0 || (size_t)written >= sizeof(partial_path) ||
        !evk_path_absent(partial_path) ||
        !evk_directory_has_exact_inventory(data->source_directory) ||
        !evk_load_population(
            request, &data->manifest, &data->train_rows,
            &data->validation_rows, &data->train_count,
            &data->validation_count
        ) ||
        !evk_validate_metadata(
            request, data->source_directory, data->train_count,
            data->validation_count
        ) ||
        !evk_validate_sha_manifest(request, data->source_directory)) {
        goto cleanup;
    }
    data->train_assignments =
        (uint32_t *)calloc(data->train_count, sizeof(uint32_t));
    data->validation_assignments =
        (uint32_t *)calloc(data->validation_count, sizeof(uint32_t));
    data->train_labels =
        (uint32_t *)calloc(data->train_count, sizeof(uint32_t));
    data->validation_labels =
        (uint32_t *)calloc(data->validation_count, sizeof(uint32_t));
    if (data->train_assignments == NULL ||
        data->validation_assignments == NULL ||
        data->train_labels == NULL || data->validation_labels == NULL) {
        evk_error("assignment validation allocation failed");
        goto cleanup;
    }
    if (!evk_path_join(
            data->source_directory, "train_assignments.csv", path
        ) ||
        !evk_validate_assignment_csv(
            path, data->train_rows, data->train_count, &data->manifest,
            "train", request->k, data->train_assignments,
            data->train_labels, &data->train_report
        ) ||
        !evk_path_join(
            data->source_directory, "train_assignments_u32.bin", path
        ) ||
        !evk_validate_assignment_binary(
            path, data->train_assignments, data->train_count
        ) ||
        !evk_path_join(
            data->source_directory, "validation_assignments.csv", path
        ) ||
        !evk_validate_assignment_csv(
            path, data->validation_rows, data->validation_count,
            &data->manifest, "validation", request->k,
            data->validation_assignments, data->validation_labels,
            &data->validation_report
        ) ||
        !evk_path_join(
            data->source_directory, "validation_assignments_u32.bin", path
        ) ||
        !evk_validate_assignment_binary(
            path, data->validation_assignments, data->validation_count
        )) {
        evk_error("assignment/label join validation failed");
        goto cleanup;
    }
    if (request->scope == EVK_SCOPE_MATCHED_SUBSET1000 &&
        data->validation_report.label_counts[EVK_WANDERING_LINE_INDEX] != 0u) {
        evk_error("matched validation Wandering_Line contract mismatch");
        goto cleanup;
    }
    {
        size_t label;
        for (label = 0u; label < EVK_LABEL_COUNT; label++) {
            if (data->train_report.label_counts[label] == 0u ||
                (request->scope == EVK_SCOPE_FULL &&
                 data->validation_report.label_counts[label] == 0u) ||
                (request->scope == EVK_SCOPE_MATCHED_SUBSET1000 &&
                 label != EVK_WANDERING_LINE_INDEX &&
                 data->validation_report.label_counts[label] == 0u)) {
                evk_error("canonical label coverage mismatch at label %zu",
                          label);
                goto cleanup;
            }
        }
    }
    ok = 1;

cleanup:
    if (!ok) {
        evk_validated_data_free(data);
    }
    return ok;
}


static int evk_run_validate_only(const EVKRequest *request)
{
    EVKValidatedData data;

    if (!evk_load_validated_data(request, &data)) {
        return 0;
    }
    printf("SCOPE=%s\n", request->scope == EVK_SCOPE_FULL
           ? "full" : "matched_subset1000");
    printf("PROFILE=%s\n", request->profile->token);
    printf("K=%zu\n", request->k);
    printf("TRAIN_ROWS_VALIDATED=%zu\n", data.train_report.rows);
    printf("VALIDATION_ROWS_VALIDATED=%zu\n", data.validation_report.rows);
    printf("LABEL_JOIN=manifest_index_cross_checked\n");
    printf("LABEL_COUNT=%d\n", EVK_LABEL_COUNT);
    printf("CONTINGENCY_ROWS=%d\n", EVK_LABEL_COUNT);
    printf("CONTINGENCY_COLS=%zu\n", request->k);
    if (request->scope == EVK_SCOPE_MATCHED_SUBSET1000) {
        printf("WANDERING_LINE_MATCHED_VALIDATION_COUNT=0\n");
        printf("WANDERING_LINE_ZERO_ROW_ACCEPTED=true\n");
    }
    printf("METRICS_PLANNED=purity,nmi_arithmetic,ari,homogeneity,completeness\n");
    printf("METRICS_COMPUTE_CALLS=0\n");
    printf("OUTPUT_CREATED=false\n");
    printf("VALIDATE_ONLY_PASS=true\n");
    evk_validated_data_free(&data);
    return 1;
}


typedef struct {
    char root[EVK_PATH_CAP];
    char final[EVK_PATH_CAP];
    char partial[EVK_PATH_CAP];
    int root_missing;
    int final_exists;
    int partial_exists;
} EVKPublishPaths;


static int evk_path_state(const char *path)
{
    if (path == NULL || path[0] == '\0') {
        return 3;
    }
#ifdef _WIN32
    {
        DWORD error;
        DWORD attributes = GetFileAttributesA(path);
        if (attributes == INVALID_FILE_ATTRIBUTES) {
            error = GetLastError();
            return error == ERROR_FILE_NOT_FOUND ||
                error == ERROR_PATH_NOT_FOUND ? 0 : 3;
        }
        if ((attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0u) {
            return 3;
        }
        return (attributes & FILE_ATTRIBUTE_DIRECTORY) != 0u ? 2 : 1;
    }
#else
    {
        struct stat status;
        errno = 0;
        if (lstat(path, &status) != 0) {
            return errno == ENOENT ? 0 : 3;
        }
        if (S_ISLNK(status.st_mode)) {
            return 3;
        }
        if (S_ISDIR(status.st_mode)) {
            return 2;
        }
        return S_ISREG(status.st_mode) ? 1 : 3;
    }
#endif
}


static int evk_parent_directory(
    const char *path,
    char parent[EVK_PATH_CAP]
)
{
    const char *slash;
    const char *backslash;
    const char *separator;
    size_t length;

    if (path == NULL || parent == NULL || path[0] == '\0') {
        return 0;
    }
    slash = strrchr(path, '/');
    backslash = strrchr(path, '\\');
    separator = slash;
    if (backslash != NULL && (separator == NULL || backslash > separator)) {
        separator = backslash;
    }
    if (separator == NULL) {
        return snprintf(parent, EVK_PATH_CAP, ".") == 1;
    }
    length = (size_t)(separator - path);
#ifdef _WIN32
    if (length == 2u && path[1] == ':') {
        length++;
    }
#endif
    if (length == 0u) {
        length = 1u;
    }
    if (length >= EVK_PATH_CAP) {
        return 0;
    }
    memcpy(parent, path, length);
    parent[length] = '\0';
    return 1;
}


static int evk_absolute_path(
    const char *path,
    char absolute[EVK_PATH_CAP]
)
{
    size_t length;

    if (path == NULL || absolute == NULL || path[0] == '\0') {
        return 0;
    }
#ifdef _WIN32
    {
        DWORD result = GetFullPathNameA(path, EVK_PATH_CAP, absolute, NULL);
        if (result == 0u || result >= EVK_PATH_CAP) {
            return 0;
        }
    }
#else
    if (realpath(path, absolute) == NULL) {
        char parent[EVK_PATH_CAP];
        char resolved_parent[EVK_PATH_CAP];
        const char *name;
        const char *slash = strrchr(path, '/');
        if (!evk_parent_directory(path, parent) ||
            realpath(parent, resolved_parent) == NULL) {
            return 0;
        }
        name = slash == NULL ? path : slash + 1;
        if (name[0] == '\0' || strcmp(name, ".") == 0 ||
            strcmp(name, "..") == 0 ||
            !evk_path_join(resolved_parent, name, absolute)) {
            return 0;
        }
    }
#endif
    length = strlen(absolute);
    while (length > 1u &&
           (absolute[length - 1u] == '/' || absolute[length - 1u] == '\\')) {
#ifdef _WIN32
        if (length == 3u && absolute[1] == ':') {
            break;
        }
#endif
        absolute[--length] = '\0';
    }
    return 1;
}


static int evk_path_character_equal(char left, char right)
{
#ifdef _WIN32
    if (left == '\\') {
        left = '/';
    }
    if (right == '\\') {
        right = '/';
    }
    return tolower((unsigned char)left) == tolower((unsigned char)right);
#else
    return left == right;
#endif
}


static int evk_path_is_same_or_descendant(
    const char *path,
    const char *ancestor
)
{
    size_t index;
    size_t ancestor_length;

    if (path == NULL || ancestor == NULL) {
        return 0;
    }
    ancestor_length = strlen(ancestor);
    if (strlen(path) < ancestor_length) {
        return 0;
    }
    for (index = 0u; index < ancestor_length; index++) {
        if (!evk_path_character_equal(path[index], ancestor[index])) {
            return 0;
        }
    }
    return path[ancestor_length] == '\0' || path[ancestor_length] == '/' ||
        path[ancestor_length] == '\\';
}


static int evk_prepare_publish_paths(
    const EVKRequest *request,
    EVKPublishPaths *paths
)
{
    char input_root[EVK_PATH_CAP];
    char input_absolute[EVK_PATH_CAP];
    char output_absolute[EVK_PATH_CAP];
    char parent[EVK_PATH_CAP];
    char basename[256];
    int requested_root_state;
    int root_state;
    int written;

    if (paths != NULL) {
        memset(paths, 0, sizeof(*paths));
    }
    requested_root_state = request != NULL && request->output_root != NULL
        ? evk_path_state(request->output_root) : 3;
    if (request == NULL || paths == NULL || request->output_root == NULL ||
        (requested_root_state != 0 && requested_root_state != 2) ||
        !evk_path_join(request->repository_root, EVK_OUTPUT_ROOT, input_root) ||
        !evk_absolute_path(input_root, input_absolute) ||
        !evk_absolute_path(request->output_root, output_absolute) ||
        evk_path_is_same_or_descendant(output_absolute, input_absolute) ||
        !evk_make_output_basename(request, basename)) {
        return evk_error("unsafe or overlapping evaluation output root");
    }
    if (snprintf(paths->root, sizeof(paths->root), "%s", output_absolute) < 0 ||
        strlen(output_absolute) >= sizeof(paths->root) ||
        !evk_path_join(paths->root, basename, paths->final)) {
        return evk_error("evaluation output path overflow");
    }
    written = snprintf(
        paths->partial, sizeof(paths->partial), "%s.partial", paths->final
    );
    if (written < 0 || (size_t)written >= sizeof(paths->partial)) {
        return evk_error("evaluation partial path overflow");
    }
    root_state = evk_path_state(paths->root);
    if (root_state == 0) {
        if (!evk_parent_directory(paths->root, parent) ||
            evk_path_state(parent) != 2) {
            return evk_error("evaluation output parent is not a safe directory");
        }
        paths->root_missing = 1;
    } else if (root_state != 2) {
        return evk_error("evaluation output root is not a safe directory");
    }
    paths->final_exists = !evk_path_absent(paths->final);
    paths->partial_exists = !evk_path_absent(paths->partial);
    if (paths->final_exists || paths->partial_exists) {
        return evk_error("evaluation output already exists");
    }
    return 1;
}


static int evk_create_directory(const char *path)
{
    if (path == NULL || evk_path_state(path) != 0) {
        return 0;
    }
#ifdef _WIN32
    return CreateDirectoryA(path, NULL) != 0 && evk_path_state(path) == 2;
#else
    return mkdir(path, 0777) == 0 && evk_path_state(path) == 2;
#endif
}


static int evk_evaluation_output_index(const char *name)
{
    size_t index;

    for (index = 0u; index < EVK_EVALUATION_OUTPUT_FILE_COUNT; index++) {
        if (strcmp(name, evk_evaluation_output_names[index]) == 0) {
            return (int)index;
        }
    }
    return -1;
}


static int evk_evaluation_inventory_valid(
    const char *directory,
    int require_all
)
{
    unsigned char seen[EVK_EVALUATION_OUTPUT_FILE_COUNT] = {0};
    size_t count = 0u;

#ifdef _WIN32
    WIN32_FIND_DATAA entry;
    HANDLE search;
    DWORD find_error;
    char pattern[EVK_PATH_CAP];

    if (!evk_path_join(directory, "*", pattern)) {
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
        index = evk_evaluation_output_index(entry.cFileName);
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
            char path[EVK_PATH_CAP];
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
            index = evk_evaluation_output_index(entry->d_name);
            if (index < 0 || seen[index] != 0u ||
                !evk_path_join(directory, entry->d_name, path) ||
                lstat(path, &status) != 0 || !S_ISREG(status.st_mode)) {
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
        return count <= EVK_EVALUATION_OUTPUT_FILE_COUNT;
    }
    if (count != EVK_EVALUATION_OUTPUT_FILE_COUNT) {
        return 0;
    }
    for (count = 0u; count < EVK_EVALUATION_OUTPUT_FILE_COUNT; count++) {
        if (seen[count] == 0u) {
            return 0;
        }
    }
    return 1;
}


static int evk_finish_output_file(FILE *file)
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


static int evk_open_output_file(
    const char *directory,
    const char *name,
    FILE **file
)
{
    char path[EVK_PATH_CAP];

    if (directory == NULL || name == NULL || file == NULL ||
        evk_evaluation_output_index(name) < 0 ||
        !evk_path_join(directory, name, path) || evk_path_state(path) != 0) {
        return 0;
    }
    *file = fopen(path, "wb");
    return *file != NULL;
}


static int evk_remove_created_partial(const char *directory)
{
    size_t index;

    if (evk_path_state(directory) != 2 ||
        !evk_evaluation_inventory_valid(directory, 0)) {
        return 0;
    }
    for (index = 0u; index < EVK_EVALUATION_OUTPUT_FILE_COUNT; index++) {
        char path[EVK_PATH_CAP];
        if (!evk_path_join(
                directory, evk_evaluation_output_names[index], path
            )) {
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


static int evk_publish_directory_no_replace(
    const char *partial,
    const char *final
)
{
    if (partial == NULL || final == NULL || evk_path_state(partial) != 2 ||
        evk_path_state(final) != 0) {
        return 0;
    }
#ifdef _WIN32
    return MoveFileExA(partial, final, MOVEFILE_WRITE_THROUGH) != 0;
#else
    return rename(partial, final) == 0;
#endif
}


static int evk_collect_provenance(
    const EVKRequest *request,
    const EVKValidatedData *data,
    EVKProvenanceHashes *hashes
)
{
    size_t index;

    if (request == NULL || data == NULL || hashes == NULL) {
        return 0;
    }
    memset(hashes, 0, sizeof(*hashes));
    for (index = 0u; index < EVK_EVALUATION_SOURCE_FILE_COUNT; index++) {
        char path[EVK_PATH_CAP];
        if (!evk_path_join(
                data->source_directory, evk_evaluation_source_names[index], path
            ) || vae_c_mu_file_sha256(path, hashes->source[index]) !=
                VAE_C_MU_OK) {
            return evk_error("cannot hash frozen clustering input");
        }
    }
    for (index = 0u; index < EVK_EVALUATION_CANONICAL_FILE_COUNT; index++) {
        char path[EVK_PATH_CAP];
        if (!evk_path_join(
                request->repository_root,
                evk_evaluation_canonical_names[index], path
            ) || vae_c_mu_file_sha256(path, hashes->canonical[index]) !=
                VAE_C_MU_OK || strcmp(
                    hashes->canonical[index],
                    evk_evaluation_canonical_hashes[index]
                ) != 0) {
            return evk_error("canonical evaluation input hash mismatch");
        }
    }
    for (index = 0u; index < EVK_EVALUATION_CODE_FILE_COUNT; index++) {
        char path[EVK_PATH_CAP];
        if (!evk_path_join(
                request->repository_root, evk_evaluation_code_names[index], path
            ) || vae_c_mu_file_sha256(path, hashes->code[index]) !=
                VAE_C_MU_OK) {
            return evk_error("cannot hash evaluator code provenance");
        }
    }
    return 1;
}


static int evk_provenance_unchanged(
    const EVKRequest *request,
    const EVKValidatedData *data,
    const EVKProvenanceHashes *expected
)
{
    EVKProvenanceHashes actual;

    return evk_directory_has_exact_inventory(data->source_directory) &&
        evk_validate_metadata(
            request, data->source_directory, data->train_count,
            data->validation_count
        ) &&
        evk_validate_sha_manifest(request, data->source_directory) &&
        evk_collect_provenance(request, data, &actual) &&
        memcmp(&actual, expected, sizeof(actual)) == 0;
}


static int evk_write_label_map(const char *directory)
{
    FILE *file = NULL;
    size_t label;
    int passed = 1;

    if (!evk_open_output_file(directory, "label_map.csv", &file)) {
        return 0;
    }
    if (fputs("label_index,label\n", file) == EOF) {
        passed = 0;
    }
    for (label = 0u; passed && label < EVK_LABEL_COUNT; label++) {
        if (fprintf(file, "%zu,%s\n", label, evk_labels[label]) < 0) {
            passed = 0;
        }
    }
    if (!evk_finish_output_file(file)) {
        passed = 0;
    }
    return passed;
}


static int evk_format_metrics_line(
    char line[EVK_LINE_CAP],
    const char *split,
    const GSBaselineClusteringMetrics *metrics
)
{
    int written;

    if (line == NULL || split == NULL || metrics == NULL) {
        return 0;
    }
    written = snprintf(
        line, EVK_LINE_CAP,
        "%s,%zu,%zu,%zu,%.17g,%.17g,%.17g,%.17g,%.17g",
        split, metrics->rows, metrics->label_count, metrics->cluster_count,
        metrics->purity, metrics->normalized_mutual_information,
        metrics->adjusted_rand_index, metrics->homogeneity,
        metrics->completeness
    );
    return written >= 0 && (size_t)written < EVK_LINE_CAP;
}


static int evk_write_metrics(
    const char *directory,
    const EVKMetricsPair *pair
)
{
    static const char header[] =
        "split,rows,label_count,cluster_count,purity,nmi_arithmetic,ari,"
        "homogeneity,completeness";
    char line[EVK_LINE_CAP];
    FILE *file = NULL;
    int passed = 1;

    if (pair == NULL ||
        !evk_open_output_file(directory, "metrics.csv", &file)) {
        return 0;
    }
    if (fprintf(file, "%s\n", header) < 0 ||
        !evk_format_metrics_line(line, "train", &pair->train) ||
        fprintf(file, "%s\n", line) < 0 ||
        !evk_format_metrics_line(line, "validation", &pair->validation) ||
        fprintf(file, "%s\n", line) < 0) {
        passed = 0;
    }
    if (!evk_finish_output_file(file)) {
        passed = 0;
    }
    return passed;
}


static int evk_write_contingency(
    const char *directory,
    const char *name,
    const GSBaselineClusteringMetrics *metrics
)
{
    FILE *file = NULL;
    size_t label;
    size_t cluster;
    int passed = 1;

    if (metrics == NULL || !evk_open_output_file(directory, name, &file)) {
        return 0;
    }
    if (fputs("label_index,label", file) == EOF) {
        passed = 0;
    }
    for (cluster = 0u; passed && cluster < metrics->cluster_count; cluster++) {
        if (fprintf(file, ",cluster_%zu", cluster) < 0) {
            passed = 0;
        }
    }
    if (passed && fputs(",row_total\n", file) == EOF) {
        passed = 0;
    }
    for (label = 0u; passed && label < EVK_LABEL_COUNT; label++) {
        if (fprintf(file, "%zu,%s", label, evk_labels[label]) < 0) {
            passed = 0;
            break;
        }
        for (cluster = 0u; passed && cluster < metrics->cluster_count;
             cluster++) {
            if (fprintf(
                    file, ",%" PRIu64,
                    metrics->contingency[
                        label * metrics->cluster_count + cluster
                    ]
                ) < 0) {
                passed = 0;
            }
        }
        if (passed && fprintf(
                file, ",%" PRIu64 "\n", metrics->label_totals[label]
            ) < 0) {
            passed = 0;
        }
    }
    if (!evk_finish_output_file(file)) {
        passed = 0;
    }
    return passed;
}


static int evk_files_equal(const char *left_path, const char *right_path)
{
    unsigned char left_buffer[8192];
    unsigned char right_buffer[8192];
    FILE *left = NULL;
    FILE *right = NULL;
    int passed = 1;

    if (left_path == NULL || right_path == NULL ||
        evk_path_state(left_path) != 1 || evk_path_state(right_path) != 1 ||
        (left = fopen(left_path, "rb")) == NULL ||
        (right = fopen(right_path, "rb")) == NULL) {
        if (right != NULL) {
            fclose(right);
        }
        if (left != NULL) {
            fclose(left);
        }
        return 0;
    }
    for (;;) {
        size_t left_count = fread(left_buffer, 1u, sizeof(left_buffer), left);
        size_t right_count = fread(
            right_buffer, 1u, sizeof(right_buffer), right
        );
        if (left_count != right_count ||
            memcmp(left_buffer, right_buffer, left_count) != 0) {
            passed = 0;
            break;
        }
        if (left_count < sizeof(left_buffer)) {
            if (ferror(left) || ferror(right) || !feof(left) || !feof(right)) {
                passed = 0;
            }
            break;
        }
    }
    if (fclose(right) != 0) {
        passed = 0;
    }
    if (fclose(left) != 0) {
        passed = 0;
    }
    return passed;
}


static int evk_write_evaluator_source_snapshot(
    const char *directory,
    const EVKRequest *request,
    const EVKProvenanceHashes *hashes
)
{
    unsigned char buffer[8192];
    char source_path[EVK_PATH_CAP];
    char output_path[EVK_PATH_CAP];
    char output_hash[65];
    FILE *source = NULL;
    FILE *output = NULL;
    int passed = 1;

    if (directory == NULL || request == NULL || hashes == NULL ||
        !evk_path_join(
            request->repository_root, evk_evaluation_code_names[0], source_path
        ) || evk_path_state(source_path) != 1 ||
        !evk_path_join(directory, "evaluator_source.c", output_path) ||
        (source = fopen(source_path, "rb")) == NULL ||
        !evk_open_output_file(directory, "evaluator_source.c", &output)) {
        if (source != NULL) {
            fclose(source);
        }
        return 0;
    }
    for (;;) {
        size_t count = fread(buffer, 1u, sizeof(buffer), source);
        if (count > 0u && fwrite(buffer, 1u, count, output) != count) {
            passed = 0;
            break;
        }
        if (count < sizeof(buffer)) {
            if (ferror(source) || !feof(source)) {
                passed = 0;
            }
            break;
        }
    }
    if (fclose(source) != 0) {
        passed = 0;
    }
    if (!evk_finish_output_file(output)) {
        passed = 0;
    }
    return passed &&
        vae_c_mu_file_sha256(output_path, output_hash) == VAE_C_MU_OK &&
        strcmp(output_hash, hashes->code[0]) == 0 &&
        evk_files_equal(source_path, output_path);
}


static int evk_verify_evaluator_source_snapshot(
    const char *directory,
    const EVKRequest *request,
    const EVKProvenanceHashes *hashes
)
{
    char source_path[EVK_PATH_CAP];
    char snapshot_path[EVK_PATH_CAP];

    return directory != NULL && request != NULL && hashes != NULL &&
        evk_path_join(
            request->repository_root, evk_evaluation_code_names[0], source_path
        ) && evk_path_join(
            directory, "evaluator_source.c", snapshot_path
        ) && evk_verify_hash(snapshot_path, hashes->code[0]) &&
        evk_files_equal(source_path, snapshot_path);
}


static int evk_source_relative_directory(
    const EVKRequest *request,
    char output[EVK_PATH_CAP]
)
{
    char basename[256];

    return request != NULL && evk_make_output_basename(request, basename) &&
        evk_path_join(EVK_OUTPUT_ROOT, basename, output);
}


static int evk_write_metadata(
    const char *directory,
    const EVKRequest *request,
    const EVKValidatedData *data,
    const EVKProvenanceHashes *hashes
)
{
    const char *scope;
    const char *population;
    char source_relative[EVK_PATH_CAP];
    FILE *file = NULL;
    int passed;

    if (request == NULL || data == NULL || hashes == NULL ||
        !evk_source_relative_directory(request, source_relative) ||
        !evk_open_output_file(directory, "metadata.txt", &file)) {
        return 0;
    }
    scope = request->scope == EVK_SCOPE_FULL
        ? "full" : "matched_subset1000";
    population = request->scope == EVK_SCOPE_FULL
        ? "gravity_spy_o1_full_split_seed123_v1"
        : "canonical_prefix_800_train_200_validation_seed123_v1";
    passed = fprintf(
        file,
        "format=%s\n"
        "scope=%s\n"
        "profile=%s\n"
        "profile_id=%s\n"
        "source_kmeans_bundle=%s\n"
        "population_identity=%s\n"
        "row_order=native_split_pos_ascending\n"
        "row_identity=split_pos,manifest_index,event_index,event_id,split\n"
        "label_source=%s\n"
        "label_source_sha256=%s\n"
        "label_join=manifest_index_cross_checked_event_id_event_index_split_split_pos\n"
        "labels_used_for_clustering=false\n"
        "labels_used_for_evaluation=true\n"
        "assignments_modified=false\n"
        "selection_post_hoc=false\n"
        "label_count=%d\n"
        "k=%zu\n"
        "train_rows=%zu\n"
        "validation_rows=%zu\n"
        "contingency_layout=label_major_[22,k]\n"
        "metrics=purity,nmi_arithmetic,ari,homogeneity,completeness\n"
        "nmi_average_method=arithmetic\n"
        "source_metadata_sha256=%s\n"
        "source_checksum_sha256=%s\n"
        "source_train_assignments_csv_sha256=%s\n"
        "source_validation_assignments_csv_sha256=%s\n"
        "source_train_assignments_bin_sha256=%s\n"
        "source_validation_assignments_bin_sha256=%s\n"
        "full_train_split_sha256=%s\n"
        "full_validation_split_sha256=%s\n"
        "matched_train_subset_sha256=%s\n"
        "matched_validation_subset_sha256=%s\n"
        "repository_head=%s\n"
        "evaluator_source_snapshot=evaluator_source.c\n"
        "evaluator_source_provenance=embedded_exact_copy_sha256\n"
        "evaluator_source_sha256=%s\n"
        "metrics_backend_source_sha256=%s\n"
        "metrics_backend_header_sha256=%s\n"
        "event_dataset_source_sha256=%s\n"
        "event_dataset_header_sha256=%s\n"
        "saved_split_source_sha256=%s\n"
        "saved_split_header_sha256=%s\n"
        "output_file_count=%d\n"
        "checksum_manifest_excludes_self=true\n"
        "deterministic_serialization=c_locale_lf_17g\n",
        EVK_EVALUATION_FORMAT,
        scope,
        request->profile->token,
        request->profile->profile_id,
        source_relative,
        population,
        EVK_MANIFEST_PATH,
        hashes->canonical[0],
        EVK_LABEL_COUNT,
        request->k,
        data->train_count,
        data->validation_count,
        hashes->source[0],
        hashes->source[1],
        hashes->source[2],
        hashes->source[3],
        hashes->source[4],
        hashes->source[5],
        hashes->canonical[1],
        hashes->canonical[2],
        hashes->canonical[3],
        hashes->canonical[4],
        request->code_commit,
        hashes->code[0],
        hashes->code[1],
        hashes->code[2],
        hashes->code[3],
        hashes->code[4],
        hashes->code[5],
        hashes->code[6],
        EVK_EVALUATION_OUTPUT_FILE_COUNT
    ) >= 0;
    if (!evk_finish_output_file(file)) {
        passed = 0;
    }
    return passed;
}


static int evk_expected_manifest_line(
    char line[EVK_LINE_CAP],
    const char *category,
    const char *hash,
    const char *name
)
{
    int written;

    if (line == NULL || category == NULL || !evk_hash_is_lower_hex(hash) ||
        name == NULL || name[0] == '\0' || strchr(name, ' ') != NULL) {
        return 0;
    }
    written = snprintf(
        line, EVK_LINE_CAP, "%s %s %s", category, hash, name
    );
    return written >= 0 && (size_t)written < EVK_LINE_CAP;
}


static int evk_write_evaluation_sha256(
    const char *directory,
    const EVKRequest *request,
    const EVKProvenanceHashes *hashes
)
{
    char source_relative[EVK_PATH_CAP];
    FILE *file = NULL;
    size_t index;
    int passed = 1;

    if (request == NULL || hashes == NULL ||
        !evk_source_relative_directory(request, source_relative) ||
        !evk_open_output_file(directory, "sha256.txt", &file)) {
        return 0;
    }
    for (index = 0u; passed &&
         index < EVK_EVALUATION_SOURCE_FILE_COUNT; index++) {
        char name[EVK_PATH_CAP];
        if (!evk_path_join(
                source_relative, evk_evaluation_source_names[index], name
            ) || fprintf(
                file, "SOURCE %s %s\n", hashes->source[index], name
            ) < 0) {
            passed = 0;
        }
    }
    for (index = 0u; passed &&
         index < EVK_EVALUATION_CANONICAL_FILE_COUNT; index++) {
        if (fprintf(
                file, "CANONICAL %s %s\n", hashes->canonical[index],
                evk_evaluation_canonical_names[index]
            ) < 0) {
            passed = 0;
        }
    }
    for (index = 0u; passed && index < EVK_EVALUATION_CODE_FILE_COUNT;
         index++) {
        if (fprintf(
                file, "CODE %s %s\n", hashes->code[index],
                evk_evaluation_code_names[index]
            ) < 0) {
            passed = 0;
        }
    }
    for (index = 0u; passed && index < EVK_EVALUATION_ARTIFACT_COUNT;
         index++) {
        char path[EVK_PATH_CAP];
        char hash[65];
        if (!evk_path_join(
                directory, evk_evaluation_output_names[index], path
            ) || vae_c_mu_file_sha256(path, hash) != VAE_C_MU_OK ||
            fprintf(
                file, "ARTIFACT %s %s\n", hash,
                evk_evaluation_output_names[index]
            ) < 0) {
            passed = 0;
        }
    }
    if (!evk_finish_output_file(file)) {
        passed = 0;
    }
    return passed;
}


static int evk_read_expected_line(FILE *file, const char *expected)
{
    char line[EVK_LINE_CAP];

    if (file == NULL || expected == NULL ||
        fgets(line, sizeof(line), file) == NULL ||
        (strchr(line, '\n') == NULL && !feof(file))) {
        return 0;
    }
    evk_strip_line(line);
    return strcmp(line, expected) == 0;
}


static int evk_verify_label_map(const char *directory)
{
    char path[EVK_PATH_CAP];
    char expected[EVK_LINE_CAP];
    FILE *file = NULL;
    size_t label;
    int passed = 1;

    if (!evk_path_join(directory, "label_map.csv", path) ||
        (file = fopen(path, "rb")) == NULL ||
        !evk_read_expected_line(file, "label_index,label")) {
        if (file != NULL) {
            fclose(file);
        }
        return 0;
    }
    for (label = 0u; passed && label < EVK_LABEL_COUNT; label++) {
        int written = snprintf(
            expected, sizeof(expected), "%zu,%s", label, evk_labels[label]
        );
        if (written < 0 || (size_t)written >= sizeof(expected) ||
            !evk_read_expected_line(file, expected)) {
            passed = 0;
        }
    }
    if (passed && fgets(expected, sizeof(expected), file) != NULL) {
        passed = 0;
    }
    if (ferror(file)) {
        passed = 0;
    }
    if (fclose(file) != 0) {
        passed = 0;
    }
    return passed;
}


static int evk_verify_metrics(
    const char *directory,
    const EVKMetricsPair *pair
)
{
    static const char header[] =
        "split,rows,label_count,cluster_count,purity,nmi_arithmetic,ari,"
        "homogeneity,completeness";
    char path[EVK_PATH_CAP];
    char expected[EVK_LINE_CAP];
    char extra[2];
    FILE *file = NULL;
    int passed = 1;

    if (pair == NULL || !evk_path_join(directory, "metrics.csv", path) ||
        (file = fopen(path, "rb")) == NULL ||
        !evk_read_expected_line(file, header) ||
        !evk_format_metrics_line(expected, "train", &pair->train) ||
        !evk_read_expected_line(file, expected) ||
        !evk_format_metrics_line(
            expected, "validation", &pair->validation
        ) || !evk_read_expected_line(file, expected)) {
        passed = 0;
    }
    if (file != NULL && passed && fgets(extra, sizeof(extra), file) != NULL) {
        passed = 0;
    }
    if (file != NULL) {
        if (ferror(file)) {
            passed = 0;
        }
        if (fclose(file) != 0) {
            passed = 0;
        }
    }
    return passed;
}


static int evk_format_contingency_header(
    char output[EVK_LINE_CAP],
    size_t k
)
{
    size_t used;
    size_t cluster;
    int written;

    if (output == NULL || !evk_validate_k(k)) {
        return 0;
    }
    written = snprintf(output, EVK_LINE_CAP, "label_index,label");
    if (written < 0 || (size_t)written >= EVK_LINE_CAP) {
        return 0;
    }
    used = (size_t)written;
    for (cluster = 0u; cluster < k; cluster++) {
        written = snprintf(
            output + used, EVK_LINE_CAP - used, ",cluster_%zu", cluster
        );
        if (written < 0 || (size_t)written >= EVK_LINE_CAP - used) {
            return 0;
        }
        used += (size_t)written;
    }
    written = snprintf(output + used, EVK_LINE_CAP - used, ",row_total");
    return written >= 0 && (size_t)written < EVK_LINE_CAP - used;
}


static int evk_format_contingency_row(
    char output[EVK_LINE_CAP],
    const GSBaselineClusteringMetrics *metrics,
    size_t label
)
{
    size_t used;
    size_t cluster;
    int written;

    if (output == NULL || metrics == NULL || label >= EVK_LABEL_COUNT) {
        return 0;
    }
    written = snprintf(
        output, EVK_LINE_CAP, "%zu,%s", label, evk_labels[label]
    );
    if (written < 0 || (size_t)written >= EVK_LINE_CAP) {
        return 0;
    }
    used = (size_t)written;
    for (cluster = 0u; cluster < metrics->cluster_count; cluster++) {
        written = snprintf(
            output + used, EVK_LINE_CAP - used, ",%" PRIu64,
            metrics->contingency[label * metrics->cluster_count + cluster]
        );
        if (written < 0 || (size_t)written >= EVK_LINE_CAP - used) {
            return 0;
        }
        used += (size_t)written;
    }
    written = snprintf(
        output + used, EVK_LINE_CAP - used, ",%" PRIu64,
        metrics->label_totals[label]
    );
    return written >= 0 && (size_t)written < EVK_LINE_CAP - used;
}


static int evk_verify_contingency(
    const char *directory,
    const char *name,
    const GSBaselineClusteringMetrics *metrics
)
{
    char path[EVK_PATH_CAP];
    char expected[EVK_LINE_CAP];
    FILE *file = NULL;
    size_t label;
    int passed = 1;

    if (metrics == NULL || !evk_path_join(directory, name, path) ||
        (file = fopen(path, "rb")) == NULL ||
        !evk_format_contingency_header(expected, metrics->cluster_count) ||
        !evk_read_expected_line(file, expected)) {
        if (file != NULL) {
            fclose(file);
        }
        return 0;
    }
    for (label = 0u; passed && label < EVK_LABEL_COUNT; label++) {
        if (!evk_format_contingency_row(expected, metrics, label) ||
            !evk_read_expected_line(file, expected)) {
            passed = 0;
        }
    }
    if (passed && fgets(expected, sizeof(expected), file) != NULL) {
        passed = 0;
    }
    if (ferror(file)) {
        passed = 0;
    }
    if (fclose(file) != 0) {
        passed = 0;
    }
    return passed;
}


static int evk_verify_evaluation_metadata(
    const char *directory,
    const EVKRequest *request,
    const EVKValidatedData *data,
    const EVKProvenanceHashes *hashes
)
{
    char path[EVK_PATH_CAP];
    char source_relative[EVK_PATH_CAP];
    char value[64];
    const char *scope;
    const char *population;

    if (request == NULL || data == NULL || hashes == NULL ||
        !evk_path_join(directory, "metadata.txt", path) ||
        !evk_source_relative_directory(request, source_relative) ||
        !evk_validate_key_value_shape(
            path, EVK_EVALUATION_METADATA_LINE_COUNT
        )) {
        return 0;
    }
    scope = request->scope == EVK_SCOPE_FULL
        ? "full" : "matched_subset1000";
    population = request->scope == EVK_SCOPE_FULL
        ? "gravity_spy_o1_full_split_seed123_v1"
        : "canonical_prefix_800_train_200_validation_seed123_v1";
#define EVK_META_EXPECT(key_value, expected_value) \
    do { \
        if (!evk_expect_metadata(path, (key_value), (expected_value))) { \
            return 0; \
        } \
    } while (0)
    EVK_META_EXPECT("format", EVK_EVALUATION_FORMAT);
    EVK_META_EXPECT("scope", scope);
    EVK_META_EXPECT("profile", request->profile->token);
    EVK_META_EXPECT("profile_id", request->profile->profile_id);
    EVK_META_EXPECT("source_kmeans_bundle", source_relative);
    EVK_META_EXPECT("population_identity", population);
    EVK_META_EXPECT("row_order", "native_split_pos_ascending");
    EVK_META_EXPECT(
        "row_identity", "split_pos,manifest_index,event_index,event_id,split"
    );
    EVK_META_EXPECT("label_source", EVK_MANIFEST_PATH);
    EVK_META_EXPECT("label_source_sha256", hashes->canonical[0]);
    EVK_META_EXPECT(
        "label_join",
        "manifest_index_cross_checked_event_id_event_index_split_split_pos"
    );
    EVK_META_EXPECT("labels_used_for_clustering", "false");
    EVK_META_EXPECT("labels_used_for_evaluation", "true");
    EVK_META_EXPECT("assignments_modified", "false");
    EVK_META_EXPECT("selection_post_hoc", "false");
    snprintf(value, sizeof(value), "%d", EVK_LABEL_COUNT);
    EVK_META_EXPECT("label_count", value);
    snprintf(value, sizeof(value), "%zu", request->k);
    EVK_META_EXPECT("k", value);
    snprintf(value, sizeof(value), "%zu", data->train_count);
    EVK_META_EXPECT("train_rows", value);
    snprintf(value, sizeof(value), "%zu", data->validation_count);
    EVK_META_EXPECT("validation_rows", value);
    EVK_META_EXPECT("contingency_layout", "label_major_[22,k]");
    EVK_META_EXPECT(
        "metrics", "purity,nmi_arithmetic,ari,homogeneity,completeness"
    );
    EVK_META_EXPECT("nmi_average_method", "arithmetic");
    EVK_META_EXPECT("source_metadata_sha256", hashes->source[0]);
    EVK_META_EXPECT("source_checksum_sha256", hashes->source[1]);
    EVK_META_EXPECT("source_train_assignments_csv_sha256", hashes->source[2]);
    EVK_META_EXPECT(
        "source_validation_assignments_csv_sha256", hashes->source[3]
    );
    EVK_META_EXPECT("source_train_assignments_bin_sha256", hashes->source[4]);
    EVK_META_EXPECT(
        "source_validation_assignments_bin_sha256", hashes->source[5]
    );
    EVK_META_EXPECT("full_train_split_sha256", hashes->canonical[1]);
    EVK_META_EXPECT("full_validation_split_sha256", hashes->canonical[2]);
    EVK_META_EXPECT("matched_train_subset_sha256", hashes->canonical[3]);
    EVK_META_EXPECT(
        "matched_validation_subset_sha256", hashes->canonical[4]
    );
    EVK_META_EXPECT("repository_head", request->code_commit);
    EVK_META_EXPECT("evaluator_source_snapshot", "evaluator_source.c");
    EVK_META_EXPECT(
        "evaluator_source_provenance", "embedded_exact_copy_sha256"
    );
    EVK_META_EXPECT("evaluator_source_sha256", hashes->code[0]);
    EVK_META_EXPECT("metrics_backend_source_sha256", hashes->code[1]);
    EVK_META_EXPECT("metrics_backend_header_sha256", hashes->code[2]);
    EVK_META_EXPECT("event_dataset_source_sha256", hashes->code[3]);
    EVK_META_EXPECT("event_dataset_header_sha256", hashes->code[4]);
    EVK_META_EXPECT("saved_split_source_sha256", hashes->code[5]);
    EVK_META_EXPECT("saved_split_header_sha256", hashes->code[6]);
    snprintf(value, sizeof(value), "%d", EVK_EVALUATION_OUTPUT_FILE_COUNT);
    EVK_META_EXPECT("output_file_count", value);
    EVK_META_EXPECT("checksum_manifest_excludes_self", "true");
    EVK_META_EXPECT("deterministic_serialization", "c_locale_lf_17g");
#undef EVK_META_EXPECT
    return 1;
}


static int evk_verify_evaluation_sha256(
    const char *directory,
    const EVKRequest *request,
    const EVKProvenanceHashes *hashes
)
{
    char path[EVK_PATH_CAP];
    char source_relative[EVK_PATH_CAP];
    char line[EVK_LINE_CAP];
    char expected[EVK_LINE_CAP];
    FILE *file = NULL;
    size_t index;
    size_t count = 0u;
    int passed = 1;

    if (request == NULL || hashes == NULL ||
        !evk_path_join(directory, "sha256.txt", path) ||
        !evk_source_relative_directory(request, source_relative) ||
        (file = fopen(path, "rb")) == NULL) {
        return 0;
    }
#define EVK_VERIFY_SHA_LINE(category_value, hash_value, name_value) \
    do { \
        if (!evk_expected_manifest_line( \
                expected, (category_value), (hash_value), (name_value) \
            ) || !evk_read_expected_line(file, expected)) { \
            passed = 0; \
        } else { \
            count++; \
        } \
    } while (0)
    for (index = 0u; passed &&
         index < EVK_EVALUATION_SOURCE_FILE_COUNT; index++) {
        char name[EVK_PATH_CAP];
        if (!evk_path_join(
                source_relative, evk_evaluation_source_names[index], name
            )) {
            passed = 0;
            break;
        }
        EVK_VERIFY_SHA_LINE("SOURCE", hashes->source[index], name);
    }
    for (index = 0u; passed &&
         index < EVK_EVALUATION_CANONICAL_FILE_COUNT; index++) {
        EVK_VERIFY_SHA_LINE(
            "CANONICAL", hashes->canonical[index],
            evk_evaluation_canonical_names[index]
        );
    }
    for (index = 0u; passed && index < EVK_EVALUATION_CODE_FILE_COUNT;
         index++) {
        EVK_VERIFY_SHA_LINE(
            "CODE", hashes->code[index], evk_evaluation_code_names[index]
        );
    }
    for (index = 0u; passed && index < EVK_EVALUATION_ARTIFACT_COUNT;
         index++) {
        char artifact_path[EVK_PATH_CAP];
        char hash[65];
        if (!evk_path_join(
                directory, evk_evaluation_output_names[index], artifact_path
            ) || vae_c_mu_file_sha256(artifact_path, hash) != VAE_C_MU_OK) {
            passed = 0;
            break;
        }
        EVK_VERIFY_SHA_LINE(
            "ARTIFACT", hash, evk_evaluation_output_names[index]
        );
    }
#undef EVK_VERIFY_SHA_LINE
    if (passed && fgets(line, sizeof(line), file) != NULL) {
        passed = 0;
    }
    if (ferror(file)) {
        passed = 0;
    }
    if (fclose(file) != 0) {
        passed = 0;
    }
    return passed && count == EVK_EVALUATION_SHA_RECORD_COUNT;
}


static int evk_write_evaluation_artifacts(
    const char *directory,
    const EVKRequest *request,
    const EVKValidatedData *data,
    const EVKMetricsPair *pair,
    const EVKProvenanceHashes *hashes
)
{
    return evk_write_label_map(directory) &&
        evk_write_metrics(directory, pair) &&
        evk_write_contingency(
            directory, "train_contingency_counts.csv", &pair->train
        ) &&
        evk_write_contingency(
            directory, "validation_contingency_counts.csv",
            &pair->validation
        ) &&
        evk_write_evaluator_source_snapshot(directory, request, hashes) &&
        evk_write_metadata(directory, request, data, hashes) &&
        evk_write_evaluation_sha256(directory, request, hashes);
}


static int evk_verify_evaluation_artifacts(
    const char *directory,
    const EVKRequest *request,
    const EVKValidatedData *data,
    const EVKMetricsPair *pair,
    const EVKProvenanceHashes *hashes
)
{
    return evk_evaluation_inventory_valid(directory, 1) &&
        evk_verify_label_map(directory) &&
        evk_verify_metrics(directory, pair) &&
        evk_verify_contingency(
            directory, "train_contingency_counts.csv", &pair->train
        ) &&
        evk_verify_contingency(
            directory, "validation_contingency_counts.csv",
            &pair->validation
        ) &&
        evk_verify_evaluator_source_snapshot(directory, request, hashes) &&
        evk_verify_evaluation_metadata(directory, request, data, hashes) &&
        evk_verify_evaluation_sha256(directory, request, hashes);
}


static int evk_run_evaluate(const EVKRequest *request)
{
    EVKValidatedData data;
    EVKMetricsPair pair;
    EVKProvenanceHashes hashes;
    EVKPublishPaths paths;
    int paths_ready = 0;
    int root_created = 0;
    int partial_created = 0;
    int published = 0;
    int passed = 0;

    memset(&data, 0, sizeof(data));
    memset(&pair, 0, sizeof(pair));
    memset(&hashes, 0, sizeof(hashes));
    memset(&paths, 0, sizeof(paths));
    paths_ready = evk_prepare_publish_paths(request, &paths);
    if (!paths_ready) {
        goto cleanup;
    }
    if (!evk_load_validated_data(request, &data) ||
        !evk_collect_provenance(request, &data, &hashes) ||
        !evk_compute_metrics(
            data.train_labels, data.train_assignments, data.train_count,
            data.validation_labels, data.validation_assignments,
            data.validation_count, request->k, request->scope, &pair
        )) {
        goto cleanup;
    }

    /* Output creation starts only after both metric computations pass. */
    if (paths.root_missing) {
        if (!evk_create_directory(paths.root)) {
            evk_error("cannot create evaluation output root");
            goto cleanup;
        }
        root_created = 1;
    } else if (evk_path_state(paths.root) != 2) {
        evk_error("evaluation output root changed during computation");
        goto cleanup;
    }
    if (!evk_path_absent(paths.final) || !evk_path_absent(paths.partial)) {
        paths.final_exists = !evk_path_absent(paths.final);
        paths.partial_exists = !evk_path_absent(paths.partial);
        evk_error("evaluation output appeared during computation");
        goto cleanup;
    }
    if (!evk_create_directory(paths.partial)) {
        evk_error("cannot create evaluation partial directory");
        goto cleanup;
    }
    partial_created = 1;
    if (!evk_write_evaluation_artifacts(
            paths.partial, request, &data, &pair, &hashes
        ) || !evk_verify_evaluation_artifacts(
            paths.partial, request, &data, &pair, &hashes
        ) || !evk_provenance_unchanged(request, &data, &hashes)) {
        evk_error("evaluation artifact write/readback gate failed");
        goto cleanup;
    }
    if (!evk_publish_directory_no_replace(paths.partial, paths.final)) {
        paths.final_exists = !evk_path_absent(paths.final);
        evk_error("atomic no-replace evaluation publish failed");
        goto cleanup;
    }
    partial_created = 0;
    published = 1;
    if (!evk_verify_evaluation_artifacts(
            paths.final, request, &data, &pair, &hashes
        ) || !evk_provenance_unchanged(request, &data, &hashes) ||
        !evk_path_absent(paths.partial)) {
        evk_error("published evaluation readback gate failed");
        goto cleanup;
    }
    passed = 1;

cleanup:
    if (partial_created && !evk_remove_created_partial(paths.partial)) {
        evk_error("failed to remove evaluator-created partial output");
    }
    if (!passed && root_created && evk_path_state(paths.root) == 2) {
#ifdef _WIN32
        if (_rmdir(paths.root) != 0 && errno != ENOTEMPTY) {
            evk_error("failed to remove evaluator-created empty output root");
        }
#else
        if (rmdir(paths.root) != 0 && errno != ENOTEMPTY) {
            evk_error("failed to remove evaluator-created empty output root");
        }
#endif
    }
    printf("METRICS_COMPUTE_CALLS=%zu\n", pair.compute_calls);
    printf("OUTPUT_CREATED=%s\n", published ? "true" : "false");
    printf("OVERWRITE_REJECTED=%s\n",
           paths.final_exists ? "true" : "false");
    printf("PARTIAL_REJECTED=%s\n",
           paths.partial_exists ? "true" : "false");
    if (passed) {
        printf("SCOPE=%s\n", request->scope == EVK_SCOPE_FULL
               ? "full" : "matched_subset1000");
        printf("PROFILE=%s\n", request->profile->token);
        printf("K=%zu\n", request->k);
        printf("TRAIN_ROWS=%zu\n", data.train_count);
        printf("VALIDATION_ROWS=%zu\n", data.validation_count);
        printf("LABEL_COUNT=%d\n", EVK_LABEL_COUNT);
        printf("CONTINGENCY_ROWS=%d\n", EVK_LABEL_COUNT);
        printf("CONTINGENCY_COLS=%zu\n", request->k);
        printf("TRAIN_PURITY=%.17g\n", pair.train.purity);
        printf("TRAIN_NMI_ARITHMETIC=%.17g\n",
               pair.train.normalized_mutual_information);
        printf("TRAIN_ARI=%.17g\n", pair.train.adjusted_rand_index);
        printf("TRAIN_HOMOGENEITY=%.17g\n", pair.train.homogeneity);
        printf("TRAIN_COMPLETENESS=%.17g\n", pair.train.completeness);
        printf("VALIDATION_PURITY=%.17g\n", pair.validation.purity);
        printf("VALIDATION_NMI_ARITHMETIC=%.17g\n",
               pair.validation.normalized_mutual_information);
        printf("VALIDATION_ARI=%.17g\n",
               pair.validation.adjusted_rand_index);
        printf("VALIDATION_HOMOGENEITY=%.17g\n",
               pair.validation.homogeneity);
        printf("VALIDATION_COMPLETENESS=%.17g\n",
               pair.validation.completeness);
        if (request->scope == EVK_SCOPE_MATCHED_SUBSET1000) {
            printf("WANDERING_LINE_MATCHED_VALIDATION_COUNT=0\n");
            printf("WANDERING_LINE_ZERO_ROW=true\n");
        }
        printf("OUTPUT_PATH=%s\n", paths.final);
        printf("EVALUATION_PASS=true\n");
    } else {
        printf("EVALUATION_PASS=false\n");
    }
    evk_metrics_pair_free(&pair);
    evk_validated_data_free(&data);
    return passed;
}


static int evk_main(int argc, char **argv)
{
    EVKRequest request;

    if (setlocale(LC_ALL, "C") == NULL) {
        return EXIT_FAILURE;
    }
    if (!evk_parse_request(argc, argv, &request)) {
        fprintf(
            stderr,
            "Usage: %s --validate-only --scope <full|matched_subset1000> "
            "--profile <mini75|mini112|mini224|full224> --k <22|36> "
            "<repository-root>\n"
            "   or: %s --evaluate --scope <full|matched_subset1000> "
            "--profile <mini75|mini112|mini224|full224> --k <22|36> "
            "--output-root <path> --repository-head <40-lowercase-hex> "
            "<repository-root>\n",
            argc > 0 && argv != NULL ? argv[0] :
                "evaluate_vae_c_mu_kmeans",
            argc > 0 && argv != NULL ? argv[0] :
                "evaluate_vae_c_mu_kmeans"
        );
        return EXIT_FAILURE;
    }
    if (request.mode == EVK_MODE_VALIDATE_ONLY) {
        return evk_run_validate_only(&request) ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    return evk_run_evaluate(&request) ? EXIT_SUCCESS : EXIT_FAILURE;
}


#ifndef GS_VAE_C_MU_KMEANS_EVALUATOR_NO_MAIN
int main(int argc, char **argv)
{
    return evk_main(argc, argv);
}
#endif
