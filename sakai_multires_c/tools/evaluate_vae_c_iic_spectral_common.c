#include "baseline_clustering_metrics.h"
#include "vae_c_mu_contract.h"
#include "vae_c_sha256.h"
#include "vae_event_dataset.h"
#include "vae_saved_split.h"

#include <inttypes.h>
#include <locale.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <errno.h>
#include <unistd.h>
#endif

enum {
    CE_MANIFEST_N = 8583,
    CE_MINI224_N = 1000,
    CE_LABELS = 22,
    CE_CLUSTERS = 36,
    CE_PATH_CAP = 4096,
    CE_LINE_CAP = 1024
};

#define CE_MANIFEST_SHA "620960e6b35da31bc4cf1a8a6da8679027d378e1d4b2ac73c5267c1c5311ae82"
#define CE_FULL_ROW_CONTRACT_SHA "4aa38fd5c51e0fd1f6b91f299c99dfc16ebe0b8c4d7146484f6ea291ebb6c9e5"

typedef enum { CE_MINI_BINARY, CE_FULL_NPY } CEInputKind;

typedef struct {
    const char *name;
    const char *role;
    CEInputKind kind;
    size_t rows;
    const char *relative_directory;
    const char *partition_name;
    const char *partition_sha256;
    const char *row_name;
    const char *row_sha256;
    const char *metadata_name;
    const char *metadata_sha256;
    const char *partition_identity;
    const char *population;
} CEProfile;

typedef struct {
    GSEventManifest manifest;
    uint32_t *labels;
    uint32_t *clusters;
    size_t *manifest_indices;
    size_t rows;
} CEData;

static const char *const ce_label_names[CE_LABELS] = {
    "1080Lines", "1400Ripples", "Air_Compressor", "Blip", "Chirp",
    "Extremely_Loud", "Helix", "Koi_Fish", "Light_Modulation",
    "Low_Frequency_Burst", "Low_Frequency_Lines", "No_Glitch",
    "None_of_the_Above", "Paired_Doves", "Power_Line", "Repeating_Blips",
    "Scattered_Light", "Scratchy", "Tomte", "Violin_Mode",
    "Wandering_Line", "Whistle"
};

static const CEProfile ce_profiles[] = {
    {
        "mini75_primary", "PRIMARY", CE_MINI_BINARY, CE_MANIFEST_N,
        "sakai_multires_c/results/vae_c/iic/mini75_iic_v1_737da555866b822e14c545c4f2687869f41a52cd1340b9e941f3a1ce77763eb9_spectral_primary_v1",
        "cluster_labels_i32.bin",
        "cc9e47a9d1a63c245bb53393c0fb65aa014015a95bbc395f3dc0f41554879ffd",
        "rows.csv",
        "6e09d72e3b54f213a2096edffca627fd35f1f0e35778539f05a810a788d7c642",
        "spectral_spec.txt",
        "12298c81739bb6002821fb90e6cc64819deb6d8c2347b68bba9c19c7afc4c0d3",
        "mini75_spectral_primary_v1", "transductive_8583"
    },
    {
        "mini112_primary", "PRIMARY", CE_MINI_BINARY, CE_MANIFEST_N,
        "sakai_multires_c/results/vae_c/iic/mini112_iic_v1_0cecf5cd656d0d61a237d50fcf96a7890fb2e6d052ef403051bc9bc17aed8fba_spectral_primary_v1",
        "cluster_labels_i32.bin",
        "17200b8d006962e39a7b49b07030042a0980970f57085b6318165cc30e738d3f",
        "rows.csv",
        "6e09d72e3b54f213a2096edffca627fd35f1f0e35778539f05a810a788d7c642",
        "spectral_spec.txt",
        "0918521d8c541ea365ab4c31903d44b2ce25dd4be6ed96ad595c63ce3beadc7f",
        "mini112_spectral_primary_v1", "transductive_8583"
    },
    {
        "mini224_1000_primary", "PRIMARY", CE_MINI_BINARY, CE_MINI224_N,
        "sakai_multires_c/results/vae_c/iic/mini224_iic_v1_4236df3f9e82f0528fcea947b2e15e42fab8159c0d4b1ed03189e4a85daae8a5_spectral_primary_v1",
        "cluster_labels_i32.bin",
        "61bcc7a38b49c210626bd5fcf191f7b9171017df85fc7a1373fd674001af75d2",
        "rows.csv",
        "d7f5101cfe3f3ae31f24b12833f0125873888ff9ed33c4c33c428b90955db4c2",
        "spectral_spec.txt",
        "e63823ca338194b0f11bc40a6b96a386fecff1af44f1516e53f8c3676d2c52d9",
        "mini224_1000_spectral_primary_v1", "train800_then_validation200"
    },
    {
        "full224_primary", "PRIMARY", CE_FULL_NPY, CE_MANIFEST_N,
        "PRIMARY", "partition.npy",
        "5a01318f703441bbf291f7b32478043f6d0376dd81265078d6dbeac43e306ac0",
        NULL, CE_FULL_ROW_CONTRACT_SHA, "spectral_metadata.json",
        "1662aa04e340bc94d538bf222a9316e9f37489b14470090124c93df6590713ed",
        "sakai224_full_iic_symmetric_v1", "transductive_8583"
    },
    {
        "full224_sensitivity", "SENSITIVITY", CE_FULL_NPY, CE_MANIFEST_N,
        "SENSITIVITY", "partition.npy",
        "69edcfaf5006214c992ce39fb359d29841c78ab5fc13fe1f58901d06915a3632",
        NULL, CE_FULL_ROW_CONTRACT_SHA, "spectral_metadata.json",
        "88271aedb7f00ecbb611d88633d74f63a11cd197f88e3f950e7c13009a9492f5",
        "sakai224_full_iic_legacy_text_v1", "transductive_8583"
    }
};

static const char *const ce_output_names[] = {
    "metrics.csv", "contingency_22x36.csv", "cluster_summary.csv",
    "label_summary.csv", "evaluation_spec.txt"
};

static int ce_join(char out[CE_PATH_CAP], const char *left, const char *right)
{
    int written;
    if (out == NULL || left == NULL || right == NULL) return 0;
    written = snprintf(out, CE_PATH_CAP, "%s/%s", left, right);
    return written > 0 && written < CE_PATH_CAP;
}

static int ce_path_exists(const char *path)
{
    struct stat st;
    return path != NULL && stat(path, &st) == 0;
}

static int ce_hash(const char *path, char hash[VAE_C_SHA256_HEX_SIZE])
{
    return vae_c_mu_file_sha256(path, hash) == VAE_C_MU_OK;
}

static int ce_hash_matches(const char *path, const char *expected)
{
    char actual[VAE_C_SHA256_HEX_SIZE];
    return expected != NULL && ce_hash(path, actual) &&
           strcmp(actual, expected) == 0;
}

static const CEProfile *ce_find_profile(const char *name)
{
    size_t index;
    for (index = 0; index < sizeof(ce_profiles) / sizeof(ce_profiles[0]);
         index++) {
        if (strcmp(name, ce_profiles[index].name) == 0)
            return &ce_profiles[index];
    }
    return NULL;
}

static int ce_label_index(const char *name)
{
    size_t index;
    for (index = 0; index < CE_LABELS; index++)
        if (strcmp(name, ce_label_names[index]) == 0) return (int)index;
    return -1;
}

static void ce_trim(char *line)
{
    size_t length = strlen(line);
    while (length > 0u && (line[length - 1u] == '\n' ||
           line[length - 1u] == '\r')) line[--length] = '\0';
}

static int ce_parse_row(const char *line, unsigned long long *manifest_index,
                        unsigned long long *event_index, char event_id[128],
                        char split[16], unsigned long long *split_pos)
{
    int consumed = -1;
    return line != NULL && manifest_index != NULL && event_index != NULL &&
        event_id != NULL && split != NULL && split_pos != NULL &&
        sscanf(line, "%llu,%llu,%127[^,],%15[^,],%llu%n", manifest_index,
               event_index, event_id, split, split_pos, &consumed) == 5 &&
        consumed > 0 && line[consumed] == '\0';
}

static int ce_i32_byte_count_matches(uint64_t bytes, size_t rows)
{
    return rows <= UINT64_MAX / UINT64_C(4) &&
           bytes == (uint64_t)rows * UINT64_C(4);
}

static int ce_read_i32(const char *path, uint32_t *values, size_t count)
{
    struct stat status;
    FILE *file = fopen(path, "rb");
    size_t index;
    int passed = file != NULL && stat(path, &status) == 0 &&
        status.st_size >= 0 &&
        ce_i32_byte_count_matches((uint64_t)status.st_size, count);
    if (!passed) return 0;
    for (index = 0; index < count && passed; index++) {
        unsigned char bytes[4];
        passed = fread(bytes, 1u, 4u, file) == 4u;
        if (passed) {
            values[index] = (uint32_t)bytes[0] |
                ((uint32_t)bytes[1] << 8u) |
                ((uint32_t)bytes[2] << 16u) |
                ((uint32_t)bytes[3] << 24u);
        }
    }
    if (passed) passed = fgetc(file) == EOF && !ferror(file);
    if (fclose(file) != 0) passed = 0;
    return passed;
}

static int ce_read_npy_i32(const char *path, uint32_t *values, size_t count)
{
    FILE *file = fopen(path, "rb");
    unsigned char prefix[12];
    char *header = NULL;
    size_t prefix_size, header_length, got;
    int passed = file != NULL;
    if (!passed) return 0;
    got = fread(prefix, 1u, 10u, file);
    passed = got == 10u && prefix[0] == 0x93u &&
        memcmp(prefix + 1u, "NUMPY", 5u) == 0;
    if (!passed) goto done;
    if (prefix[6] == 1u) {
        prefix_size = 10u;
        header_length = (size_t)prefix[8] | ((size_t)prefix[9] << 8u);
    } else if (prefix[6] == 2u || prefix[6] == 3u) {
        if (fread(prefix + 10u, 1u, 2u, file) != 2u) {
            passed = 0; goto done;
        }
        prefix_size = 12u;
        header_length = (size_t)prefix[8] | ((size_t)prefix[9] << 8u) |
            ((size_t)prefix[10] << 16u) | ((size_t)prefix[11] << 24u);
    } else { passed = 0; goto done; }
    (void)prefix_size;
    if (header_length == 0u || header_length > 4096u) {
        passed = 0; goto done;
    }
    header = (char *)calloc(header_length + 1u, 1u);
    if (header == NULL || fread(header, 1u, header_length, file) !=
        header_length) { passed = 0; goto done; }
    {
        char shape[64];
        int written = snprintf(shape, sizeof(shape), "(%zu,", count);
        passed = written > 0 && written < (int)sizeof(shape) &&
            (strstr(header, "'descr': '<i4'") != NULL ||
             strstr(header, "\"descr\": \"<i4\"") != NULL) &&
            strstr(header, "False") != NULL && strstr(header, shape) != NULL;
    }
    if (passed) {
        size_t index;
        for (index = 0; index < count && passed; index++) {
            unsigned char bytes[4];
            passed = fread(bytes, 1u, 4u, file) == 4u;
            if (passed) values[index] = (uint32_t)bytes[0] |
                ((uint32_t)bytes[1] << 8u) |
                ((uint32_t)bytes[2] << 16u) |
                ((uint32_t)bytes[3] << 24u);
        }
    }
    if (passed) passed = fgetc(file) == EOF && !ferror(file);
done:
    free(header);
    if (fclose(file) != 0) passed = 0;
    return passed;
}

static int ce_validate_identity(size_t row, size_t manifest_index,
                                unsigned long long event_index,
                                const char *event_id,
                                const GSEventManifest *manifest,
                                unsigned char seen[CE_MANIFEST_N])
{
    const GSEventRecord *record;
    (void)row;
    if (manifest_index >= manifest->count || seen[manifest_index]) return 0;
    record = &manifest->records[manifest_index];
    if (event_index != (unsigned long long)record->event_index ||
        strcmp(event_id, record->event_id) != 0) return 0;
    seen[manifest_index] = 1u;
    return 1;
}

static int ce_validate_mini224_subset(const char *root,
                                      const GSEventManifest *manifest,
                                      const size_t *indices)
{
    char path[CE_PATH_CAP];
    GSSavedSplit train = {0}, validation = {0};
    size_t index;
    int passed = ce_join(path, root,
        "sakai_multires_c/configs/subset224_train_800_seed123.csv") &&
        ce_hash_matches(path,
        "80c5396a70e7f2c8e3a00680582945581ae76c301a5b9ba8c62e006bb25cdfb5") &&
        gs_saved_split_load(path, "train", manifest, &train) == 0 &&
        train.count == 800u && ce_join(path, root,
        "sakai_multires_c/configs/subset224_val_200_seed123.csv") &&
        ce_hash_matches(path,
        "9f5fe6c50c7398588f2de7bd4a543a54a53e7f7a0b9a84bf4004a01f065ae76f") &&
        gs_saved_split_load(path, "validation", manifest, &validation) == 0 &&
        validation.count == 200u;
    for (index = 0; index < 800u && passed; index++)
        if (indices[index] != train.indices[index]) passed = 0;
    for (index = 0; index < 200u && passed; index++)
        if (indices[800u + index] != validation.indices[index]) passed = 0;
    gs_saved_split_free(&validation);
    gs_saved_split_free(&train);
    return passed;
}

static int ce_load_mini_rows(const char *path, const CEProfile *profile,
                             CEData *data, const char *root)
{
    FILE *file = fopen(path, "rb");
    char line[CE_LINE_CAP];
    unsigned char seen[CE_MANIFEST_N] = {0};
    size_t row;
    int passed = file != NULL && fgets(line, sizeof(line), file) != NULL;
    if (!passed) goto done;
    ce_trim(line);
    if (strcmp(line,
        "manifest_index,event_index,event_id,source_split,split_pos") != 0) {
        passed = 0; goto done;
    }
    for (row = 0; row < profile->rows && passed; row++) {
        unsigned long long manifest_index, event_index, split_pos;
        char event_id[128], split[16];
        if (fgets(line, sizeof(line), file) == NULL) { passed = 0; break; }
        ce_trim(line);
        passed = ce_parse_row(line, &manifest_index, &event_index, event_id,
            split, &split_pos) && manifest_index < CE_MANIFEST_N &&
            ce_validate_identity(row, (size_t)manifest_index, event_index,
                event_id, &data->manifest, seen);
        if (!passed) break;
        if (profile->rows == CE_MANIFEST_N) {
            passed = manifest_index == row &&
                ((strcmp(split, "train") == 0 && split_pos < 6866u) ||
                 (strcmp(split, "validation") == 0 && split_pos < 1717u));
        } else if (row < 800u) {
            passed = strcmp(split, "train") == 0 && split_pos == row;
        } else {
            passed = strcmp(split, "validation") == 0 &&
                split_pos == row - 800u;
        }
        if (passed) data->manifest_indices[row] = (size_t)manifest_index;
    }
    if (passed) passed = fgets(line, sizeof(line), file) == NULL &&
        !ferror(file);
    if (passed && profile->rows == CE_MINI224_N)
        passed = ce_validate_mini224_subset(root, &data->manifest,
                                            data->manifest_indices);
done:
    if (file != NULL && fclose(file) != 0) passed = 0;
    return passed;
}

static void ce_data_free(CEData *data)
{
    if (data == NULL) return;
    free(data->manifest_indices);
    free(data->clusters);
    free(data->labels);
    gs_event_manifest_free(&data->manifest);
    memset(data, 0, sizeof(*data));
}

static int ce_load(const CEProfile *profile, const char *root,
                   const char *full_root, CEData *data)
{
    char manifest_path[CE_PATH_CAP], directory[CE_PATH_CAP];
    char partition_path[CE_PATH_CAP], metadata_path[CE_PATH_CAP];
    char rows_path[CE_PATH_CAP];
    size_t row;
    size_t cluster_counts[CE_CLUSTERS] = {0};
    int passed = 0;
    memset(data, 0, sizeof(*data));
    if (!ce_join(manifest_path, root, "manifest_events_272.csv") ||
        !ce_hash_matches(manifest_path, CE_MANIFEST_SHA) ||
        gs_event_manifest_load(manifest_path, &data->manifest) != 0 ||
        data->manifest.count != CE_MANIFEST_N) goto done;
    data->rows = profile->rows;
    data->labels = (uint32_t *)calloc(data->rows, sizeof(uint32_t));
    data->clusters = (uint32_t *)calloc(data->rows, sizeof(uint32_t));
    data->manifest_indices = (size_t *)calloc(data->rows, sizeof(size_t));
    if (data->labels == NULL || data->clusters == NULL ||
        data->manifest_indices == NULL) goto done;
    if (profile->kind == CE_MINI_BINARY) {
        if (!ce_join(directory, root, profile->relative_directory) ||
            !ce_join(partition_path, directory, profile->partition_name) ||
            !ce_join(rows_path, directory, profile->row_name) ||
            !ce_join(metadata_path, directory, profile->metadata_name) ||
            !ce_hash_matches(partition_path, profile->partition_sha256) ||
            !ce_hash_matches(rows_path, profile->row_sha256) ||
            !ce_hash_matches(metadata_path, profile->metadata_sha256) ||
            !ce_read_i32(partition_path, data->clusters, data->rows) ||
            !ce_load_mini_rows(rows_path, profile, data, root)) goto done;
    } else {
        char contract_path[CE_PATH_CAP];
        if (full_root == NULL ||
            !ce_join(directory, full_root, profile->relative_directory) ||
            !ce_join(partition_path, directory, profile->partition_name) ||
            !ce_join(metadata_path, directory, profile->metadata_name) ||
            !ce_hash_matches(partition_path, profile->partition_sha256) ||
            !ce_hash_matches(metadata_path, profile->metadata_sha256) ||
            !ce_join(contract_path, full_root,
                "../notebook06_final_v1/spectral/H_construction_contract.json") ||
            !ce_hash_matches(contract_path, CE_FULL_ROW_CONTRACT_SHA) ||
            !ce_read_npy_i32(partition_path, data->clusters, data->rows))
            goto done;
        for (row = 0; row < data->rows; row++) {
            const GSEventRecord *record = &data->manifest.records[row];
            if (record->event_index != (int)(row + 1u) ||
                record->event_id[0] == '\0') goto done;
            data->manifest_indices[row] = row;
        }
    }
    for (row = 0; row < data->rows; row++) {
        const GSEventRecord *record =
            &data->manifest.records[data->manifest_indices[row]];
        int label = ce_label_index(record->label);
        if (label < 0 || data->clusters[row] >= CE_CLUSTERS) goto done;
        data->labels[row] = (uint32_t)label;
        cluster_counts[data->clusters[row]]++;
    }
    for (row = 0; row < CE_CLUSTERS; row++)
        if (cluster_counts[row] == 0u) goto done;
    passed = 1;
done:
    if (!passed) ce_data_free(data);
    return passed;
}

static int ce_metrics_valid(const GSBaselineClusteringMetrics *metrics,
                            size_t rows)
{
    uint64_t total = 0u;
    size_t index;
    if (metrics == NULL || metrics->rows != rows ||
        metrics->label_count != CE_LABELS ||
        metrics->cluster_count != CE_CLUSTERS ||
        metrics->contingency == NULL || metrics->label_totals == NULL ||
        metrics->cluster_totals == NULL || !isfinite(metrics->purity) ||
        !isfinite(metrics->normalized_mutual_information) ||
        !isfinite(metrics->adjusted_rand_index) ||
        !isfinite(metrics->homogeneity) ||
        !isfinite(metrics->completeness)) return 0;
    for (index = 0; index < CE_LABELS * CE_CLUSTERS; index++)
        total += metrics->contingency[index];
    if (total != (uint64_t)rows) return 0;
    for (index = 0; index < CE_CLUSTERS; index++)
        if (metrics->cluster_totals[index] == 0u) return 0;
    return 1;
}

static int ce_contingency_sum_matches(
    const GSBaselineClusteringMetrics *metrics, size_t expected_rows)
{
    uint64_t total = 0u;
    size_t index;
    if (metrics == NULL || metrics->contingency == NULL) return 0;
    for (index = 0; index < metrics->label_count * metrics->cluster_count;
         index++) total += metrics->contingency[index];
    return total == (uint64_t)expected_rows;
}

static FILE *ce_open_output(const char *directory, const char *name)
{
    char path[CE_PATH_CAP];
    return ce_join(path, directory, name) ? fopen(path, "wb") : NULL;
}

static int ce_write_metrics(const char *directory, const CEProfile *profile,
                            const GSBaselineClusteringMetrics *metrics)
{
    FILE *file = ce_open_output(directory, "metrics.csv");
    int passed = file != NULL;
    if (passed) passed = fprintf(file,
        "profile,role,population,n_events,purity,nmi_arithmetic,ari,homogeneity,completeness\n"
        "%s,%s,%s,%zu,%.17g,%.17g,%.17g,%.17g,%.17g\n",
        profile->name, profile->role, profile->population, metrics->rows,
        metrics->purity, metrics->normalized_mutual_information,
        metrics->adjusted_rand_index, metrics->homogeneity,
        metrics->completeness) > 0;
    if (file != NULL && fclose(file) != 0) passed = 0;
    return passed;
}

static int ce_write_contingency(const char *directory,
                                const GSBaselineClusteringMetrics *metrics)
{
    FILE *file = ce_open_output(directory, "contingency_22x36.csv");
    size_t label, cluster;
    int passed = file != NULL;
    if (passed && fprintf(file, "label_index,label") < 0) passed = 0;
    for (cluster = 0; cluster < CE_CLUSTERS && passed; cluster++)
        if (fprintf(file, ",cluster_%zu", cluster) < 0) passed = 0;
    if (passed && fprintf(file, ",row_total\n") < 0) passed = 0;
    for (label = 0; label < CE_LABELS && passed; label++) {
        if (fprintf(file, "%zu,%s", label, ce_label_names[label]) < 0)
            passed = 0;
        for (cluster = 0; cluster < CE_CLUSTERS && passed; cluster++)
            if (fprintf(file, ",%" PRIu64,
                metrics->contingency[label * CE_CLUSTERS + cluster]) < 0)
                passed = 0;
        if (passed && fprintf(file, ",%" PRIu64 "\n",
            metrics->label_totals[label]) < 0) passed = 0;
    }
    if (file != NULL && fclose(file) != 0) passed = 0;
    return passed;
}

static int ce_write_cluster_summary(
    const char *directory, const GSBaselineClusteringMetrics *metrics)
{
    FILE *file = ce_open_output(directory, "cluster_summary.csv");
    size_t cluster, label, left, right;
    int passed = file != NULL;
    if (passed && fprintf(file, "cluster_id,size,dominant_label_index,"
        "dominant_label,dominant_count,dominant_fraction") < 0) passed = 0;
    for (left = 1; left <= 3u && passed; left++)
        if (fprintf(file, ",top%zu_label_index,top%zu_label,top%zu_count,"
            "top%zu_fraction", left, left, left, left) < 0) passed = 0;
    if (passed && fputc('\n', file) == EOF) passed = 0;
    for (cluster = 0; cluster < CE_CLUSTERS && passed; cluster++) {
        size_t order[CE_LABELS];
        uint64_t size = metrics->cluster_totals[cluster];
        for (label = 0; label < CE_LABELS; label++) order[label] = label;
        for (left = 0; left < CE_LABELS; left++) {
            for (right = left + 1u; right < CE_LABELS; right++) {
                uint64_t a = metrics->contingency[
                    order[left] * CE_CLUSTERS + cluster];
                uint64_t b = metrics->contingency[
                    order[right] * CE_CLUSTERS + cluster];
                if (b > a || (b == a && order[right] < order[left])) {
                    size_t swap = order[left]; order[left] = order[right];
                    order[right] = swap;
                }
            }
        }
        if (fprintf(file, "%zu,%" PRIu64 ",%zu,%s,%" PRIu64 ",%.17g",
            cluster, size, order[0], ce_label_names[order[0]],
            metrics->contingency[order[0] * CE_CLUSTERS + cluster],
            (double)metrics->contingency[order[0] * CE_CLUSTERS + cluster] /
            (double)size) < 0) passed = 0;
        for (left = 0; left < 3u && passed; left++) {
            uint64_t count = metrics->contingency[
                order[left] * CE_CLUSTERS + cluster];
            if (fprintf(file, ",%zu,%s,%" PRIu64 ",%.17g", order[left],
                ce_label_names[order[left]], count,
                (double)count / (double)size) < 0) passed = 0;
        }
        if (passed && fputc('\n', file) == EOF) passed = 0;
    }
    if (file != NULL && fclose(file) != 0) passed = 0;
    return passed;
}

static int ce_write_label_summary(
    const char *directory, const GSBaselineClusteringMetrics *metrics)
{
    FILE *file = ce_open_output(directory, "label_summary.csv");
    size_t label, cluster;
    int passed = file != NULL;
    if (passed && fprintf(file, "label_index,label,total_count") < 0)
        passed = 0;
    for (cluster = 0; cluster < CE_CLUSTERS && passed; cluster++)
        if (fprintf(file, ",cluster_%zu", cluster) < 0) passed = 0;
    if (passed && fprintf(file, ",principal_cluster_id,"
        "principal_cluster_count,principal_fraction\n") < 0) passed = 0;
    for (label = 0; label < CE_LABELS && passed; label++) {
        size_t best = 0u;
        uint64_t total = metrics->label_totals[label];
        if (fprintf(file, "%zu,%s,%" PRIu64, label, ce_label_names[label],
            total) < 0) passed = 0;
        for (cluster = 0; cluster < CE_CLUSTERS && passed; cluster++) {
            uint64_t count = metrics->contingency[
                label * CE_CLUSTERS + cluster];
            if (count > metrics->contingency[label * CE_CLUSTERS + best])
                best = cluster;
            if (fprintf(file, ",%" PRIu64, count) < 0) passed = 0;
        }
        if (passed && fprintf(file, ",%zu,%" PRIu64 ",%.17g\n", best,
            metrics->contingency[label * CE_CLUSTERS + best],
            total == 0u ? 0.0 :
            (double)metrics->contingency[label * CE_CLUSTERS + best] /
            (double)total) < 0) passed = 0;
    }
    if (file != NULL && fclose(file) != 0) passed = 0;
    return passed;
}

static int ce_hex(const char *text, size_t length);

static int ce_worktree_state_valid(const char *state)
{
    return state != NULL &&
        (strcmp(state, "clean_commit") == 0 ||
         strcmp(state, "uncommitted_wip") == 0);
}

static int ce_write_evaluator_provenance(FILE *file, const char *commit,
                                         const char *source_sha,
                                         const char *binary_sha,
                                         const char *worktree_state)
{
    return file != NULL && ce_hex(commit, 40u) &&
        ce_hex(source_sha, 64u) && ce_hex(binary_sha, 64u) &&
        ce_worktree_state_valid(worktree_state) &&
        fprintf(file,
            "evaluator_commit=%s\n"
            "evaluator_worktree_state=%s\n"
            "evaluator_source_sha256=%s\n"
            "evaluator_binary_sha256=%s\n",
            commit, worktree_state, source_sha, binary_sha) > 0;
}

static int ce_write_spec(const char *directory, const CEProfile *profile,
                         const char *commit, const char *source_sha,
                         const char *binary_sha,
                         const char *worktree_state)
{
    FILE *file = ce_open_output(directory, "evaluation_spec.txt");
    size_t index;
    int passed = file != NULL;
    if (passed) passed = fprintf(file,
        "format=VAE_C_IIC_COMMON_POSTHOC_EVALUATION_V1\n"
        "profile=%s\nrole=%s\npopulation=%s\nrows=%zu\n"
        "partition_identity=%s\npartition_sha256=%s\n"
        "partition_metadata_sha256=%s\n"
        "row_map_identity=%s\nrow_map_sha256=%s\n"
        "label_source=manifest_events_272.csv\nlabel_source_sha256=%s\n"
        "label_join=%s\n"
        "label_count=22\ncluster_count=36\n"
        "cluster_order=0..35_canonical_unpermuted\n"
        "contingency_layout=label_major_22x36\n"
        "metrics_module=baseline_clustering_metrics\n"
        "purity_definition=weighted_cluster_majority_fraction\n"
        "nmi_average_method=arithmetic\n"
        "metrics=weighted_purity,nmi_arithmetic,ari,homogeneity,completeness\n",
        profile->name, profile->role, profile->population, profile->rows,
        profile->partition_identity, profile->partition_sha256,
        profile->metadata_sha256,
        profile->kind == CE_FULL_NPY ?
            "implicit_manifest_index_0_to_8582" : "rows.csv",
        profile->row_sha256, CE_MANIFEST_SHA,
        profile->kind == CE_FULL_NPY
            ? "frozen_implicit_manifest_index_order_then_manifest_event_identity"
            : "rows_csv_manifest_index_event_id_event_index_cross_check") > 0;
    if (passed) passed = ce_write_evaluator_provenance(file, commit,
        source_sha, binary_sha, worktree_state);
    if (passed) passed = fprintf(file,
        "posthoc=true\npartitions_frozen=true\n"
        "labels_used_for_training=false\nlabels_used_for_evaluation=true\n"
        "mini224_population_exception=%s\nlabel_order=",
        profile->rows == CE_MINI224_N ? "true" : "false") > 0;
    for (index = 0; index < CE_LABELS && passed; index++)
        if (fprintf(file, "%s%s", index == 0u ? "" : ",",
                    ce_label_names[index]) < 0) passed = 0;
    if (passed && fputc('\n', file) == EOF) passed = 0;
    if (file != NULL && fclose(file) != 0) passed = 0;
    return passed;
}

static int ce_write_manifest(const char *directory)
{
    FILE *file = ce_open_output(directory, "sha256.txt");
    size_t index;
    int passed = file != NULL;
    for (index = 0; index < sizeof(ce_output_names) /
         sizeof(ce_output_names[0]) && passed; index++) {
        char path[CE_PATH_CAP], hash[VAE_C_SHA256_HEX_SIZE];
        if (!ce_join(path, directory, ce_output_names[index]) ||
            !ce_hash(path, hash) || fprintf(file, "%s  %s\n", hash,
            ce_output_names[index]) < 0) passed = 0;
    }
    if (file != NULL && fclose(file) != 0) passed = 0;
    return passed;
}

static int ce_make_directory(const char *path)
{
#ifdef _WIN32
    return CreateDirectoryA(path, NULL) != 0;
#else
    return mkdir(path, 0777) == 0;
#endif
}

static int ce_publish(const char *partial, const char *final)
{
    if (ce_path_exists(final)) return 0;
#ifdef _WIN32
    return MoveFileExA(partial, final, MOVEFILE_WRITE_THROUGH) != 0;
#else
    return rename(partial, final) == 0;
#endif
}

static int ce_write_outputs(const char *output, const CEProfile *profile,
                            const GSBaselineClusteringMetrics *metrics,
                            const char *commit, const char *source_sha,
                            const char *binary_sha,
                            const char *worktree_state)
{
    char partial[CE_PATH_CAP];
    int written = snprintf(partial, sizeof(partial), "%s.partial", output);
    return written > 0 && written < (int)sizeof(partial) &&
        !ce_path_exists(output) && !ce_path_exists(partial) &&
        ce_make_directory(partial) &&
        ce_write_metrics(partial, profile, metrics) &&
        ce_write_contingency(partial, metrics) &&
        ce_write_cluster_summary(partial, metrics) &&
        ce_write_label_summary(partial, metrics) &&
        ce_write_spec(partial, profile, commit, source_sha, binary_sha,
                      worktree_state) &&
        ce_write_manifest(partial) && ce_publish(partial, output);
}

static int ce_hex(const char *text, size_t length)
{
    size_t index;
    if (text == NULL || strlen(text) != length) return 0;
    for (index = 0; index < length; index++)
        if (!((text[index] >= '0' && text[index] <= '9') ||
              (text[index] >= 'a' && text[index] <= 'f'))) return 0;
    return 1;
}

static int ce_self_test(void)
{
    uint32_t labels[8] = {0, 0, 1, 1, 1, 0, 1, 0};
    uint32_t clusters[8] = {0, 1, 1, 2, 2, 0, 2, 0};
    uint32_t sparse_labels[4] = {0, 0, 21, 21};
    uint32_t sparse_clusters[4] = {0, 1, 0, 1};
    GSBaselineClusteringMetrics metrics = {0}, sparse = {0};
    GSEventRecord records[2] = {{0}};
    GSEventManifest manifest = {records, 2u};
    unsigned char seen[CE_MANIFEST_N] = {0};
    unsigned long long mi, ei, pos;
    char id[128], split[16];
    static const char test_commit[] =
        "0123456789abcdef0123456789abcdef01234567";
    static const char test_sha[] =
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    int passed;
    records[0].event_index = 1;
    records[1].event_index = 2;
    (void)snprintf(records[0].event_id, sizeof(records[0].event_id), "A");
    (void)snprintf(records[1].event_id, sizeof(records[1].event_id), "B");
    passed = ce_parse_row("0,1,A,train,0", &mi, &ei, id, split, &pos) &&
        mi == 0u && ei == 1u && pos == 0u && strcmp(id, "A") == 0 &&
        ce_validate_identity(0u, 0u, 1u, "A", &manifest, seen) &&
        !ce_validate_identity(1u, 0u, 1u, "A", &manifest, seen) &&
        !ce_validate_identity(1u, 1u, 2u, "wrong", &manifest, seen) &&
        !ce_parse_row("0,1,A,train,0,extra", &mi, &ei, id, split, &pos);
    if (!passed || gs_baseline_clustering_metrics_compute(labels, clusters,
        8u, 2u, 3u, &metrics) != GS_BASELINE_CLUSTERING_METRICS_SUCCESS)
        return 0;
    passed = metrics.contingency[0] == 3u &&
        metrics.contingency[1] == 1u && metrics.contingency[2] == 0u &&
        metrics.contingency[3] == 0u && metrics.contingency[4] == 1u &&
        metrics.contingency[5] == 3u && fabs(metrics.purity - 0.875) < 1e-15 &&
        isfinite(metrics.normalized_mutual_information) &&
        ce_contingency_sum_matches(&metrics, 8u) &&
        !ce_contingency_sum_matches(&metrics, 7u) &&
        ce_i32_byte_count_matches(UINT64_C(32), 8u) &&
        !ce_i32_byte_count_matches(UINT64_C(28), 8u);
    gs_baseline_clustering_metrics_free(&metrics);
    if (!passed || gs_baseline_clustering_metrics_compute(sparse_labels,
        sparse_clusters, 4u, CE_LABELS, CE_CLUSTERS, &sparse) !=
        GS_BASELINE_CLUSTERING_METRICS_SUCCESS) return 0;
    passed = sparse.label_totals[20] == 0u && sparse.label_totals[21] == 2u;
    gs_baseline_clustering_metrics_free(&sparse);
    clusters[0] = 36u;
    passed = passed && gs_baseline_clustering_metrics_compute(labels,
        clusters, 8u, 2u, 36u, &metrics) ==
        GS_BASELINE_CLUSTERING_METRICS_ERROR_RANGE;
    gs_baseline_clustering_metrics_free(&metrics);
    if (passed) {
        FILE *file = tmpfile();
        char buffer[512] = {0};
        passed = file != NULL && ce_write_evaluator_provenance(file,
            test_commit, test_sha, test_sha, "clean_commit") &&
            fflush(file) == 0 && fseek(file, 0L, SEEK_SET) == 0 &&
            fread(buffer, 1u, sizeof(buffer) - 1u, file) > 0u &&
            strstr(buffer, "evaluator_commit=0123456789abcdef0123456789abcdef01234567\n") != NULL &&
            strstr(buffer, "evaluator_worktree_state=clean_commit\n") != NULL &&
            strstr(buffer, "evaluator_binary_sha256=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\n") != NULL;
        if (file != NULL && fclose(file) != 0) passed = 0;
    }
    if (passed) {
        FILE *file = tmpfile();
        char buffer[512] = {0};
        passed = file != NULL && ce_write_evaluator_provenance(file,
            test_commit, test_sha, test_sha, "uncommitted_wip") &&
            fflush(file) == 0 && fseek(file, 0L, SEEK_SET) == 0 &&
            fread(buffer, 1u, sizeof(buffer) - 1u, file) > 0u &&
            strstr(buffer, "evaluator_worktree_state=uncommitted_wip\n") != NULL;
        if (file != NULL && fclose(file) != 0) passed = 0;
    }
    passed = passed &&
        !ce_write_evaluator_provenance(stdout, "invalid", test_sha,
                                       test_sha, "clean_commit") &&
        !ce_write_evaluator_provenance(stdout, test_commit, test_sha,
                                       "invalid", "clean_commit") &&
        !ce_write_evaluator_provenance(stdout, test_commit, test_sha,
                                       test_sha, "unknown") &&
        !ce_write_evaluator_provenance(stdout, test_commit, test_sha,
                                       test_sha, "");
    return passed;
}

static const char *ce_arg(int argc, char **argv, const char *name)
{
    int index;
    for (index = 2; index + 1 < argc; index += 2)
        if (strcmp(argv[index], name) == 0) return argv[index + 1];
    return NULL;
}

int main(int argc, char **argv)
{
    const char *profile_name, *root, *full_root, *output;
    const char *commit, *source_sha, *binary_sha, *worktree_state;
    const CEProfile *profile;
    CEData data = {0};
    GSBaselineClusteringMetrics metrics = {0};
    int evaluate, passed = 0;
    (void)setlocale(LC_NUMERIC, "C");
    if (argc == 2 && strcmp(argv[1], "--self-test") == 0) {
        passed = ce_self_test();
        printf("COMMON_EVALUATOR_SELF_TEST_PASS=%s\n",
               passed ? "true" : "false");
        return passed ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (argc < 8 || ((argc - 2) % 2) != 0 ||
        (strcmp(argv[1], "--validate-only") != 0 &&
         strcmp(argv[1], "--evaluate") != 0)) {
        fprintf(stderr, "usage: %s --validate-only|--evaluate --profile NAME "
            "--repository-root ROOT --full224-root ROOT [--output-dir DIR "
            "--producer-commit SHA40 --producer-source-sha256 SHA64 "
            "--producer-binary-sha256 SHA64 "
            "--evaluator-worktree-state clean_commit|uncommitted_wip]\n",
            argv[0]);
        return EXIT_FAILURE;
    }
    evaluate = strcmp(argv[1], "--evaluate") == 0;
    profile_name = ce_arg(argc, argv, "--profile");
    root = ce_arg(argc, argv, "--repository-root");
    full_root = ce_arg(argc, argv, "--full224-root");
    output = ce_arg(argc, argv, "--output-dir");
    commit = ce_arg(argc, argv, "--producer-commit");
    source_sha = ce_arg(argc, argv, "--producer-source-sha256");
    binary_sha = ce_arg(argc, argv, "--producer-binary-sha256");
    worktree_state = ce_arg(argc, argv, "--evaluator-worktree-state");
    profile = profile_name == NULL ? NULL : ce_find_profile(profile_name);
    if (profile == NULL || root == NULL || full_root == NULL ||
        (evaluate && (output == NULL || !ce_hex(commit, 40u) ||
         !ce_hex(source_sha, 64u) || !ce_hex(binary_sha, 64u) ||
         !ce_worktree_state_valid(worktree_state) ||
         ce_path_exists(output)))) return EXIT_FAILURE;
    if (!ce_load(profile, root, full_root, &data)) {
        fprintf(stderr, "input SHA, shape, cluster range, or label join failed\n");
        goto done;
    }
    printf("PROFILE=%s\nROLE=%s\nPOPULATION=%s\nN=%zu\n"
           "LABEL_JOIN_PASS=true\nCLUSTER_IDS_0_TO_35_PASS=true\n"
           "ALL_36_CLUSTERS_OCCUPIED=true\nPARTITION_MODIFIED=false\n",
           profile->name, profile->role, profile->population, profile->rows);
    if (!evaluate) {
        printf("METRICS_COMPUTED=false\nOUTPUT_CREATED=false\n"
               "VALIDATE_ONLY_PASS=true\n");
        passed = 1; goto done;
    }
    if (gs_baseline_clustering_metrics_compute(data.labels, data.clusters,
        data.rows, CE_LABELS, CE_CLUSTERS, &metrics) !=
        GS_BASELINE_CLUSTERING_METRICS_SUCCESS ||
        !ce_metrics_valid(&metrics, data.rows) ||
        !ce_write_outputs(output, profile, &metrics, commit, source_sha,
                          binary_sha, worktree_state)) {
        fprintf(stderr, "metric computation or atomic publish failed\n");
        goto done;
    }
    printf("PURITY=%.17g\nNMI_ARITHMETIC=%.17g\nARI=%.17g\n"
           "HOMOGENEITY=%.17g\nCOMPLETENESS=%.17g\n"
           "CONTINGENCY_ROWS=22\nCONTINGENCY_COLS=36\n"
           "CONTINGENCY_SUM=%zu\nMETRICS_COMPUTED=true\nOUTPUT_CREATED=true\n",
           metrics.purity, metrics.normalized_mutual_information,
           metrics.adjusted_rand_index, metrics.homogeneity,
           metrics.completeness, data.rows);
    passed = 1;
done:
    gs_baseline_clustering_metrics_free(&metrics);
    ce_data_free(&data);
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
