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
#endif

enum { EV_N = 8583, EV_TRAIN = 6866, EV_VALIDATION = 1717,
       EV_LABELS = 22, EV_CLUSTERS = 36, EV_PATH_CAP = 4096 };

#define EV_RUN_ID "mini75_iic_v1_737da555866b822e14c545c4f2687869f41a52cd1340b9e941f3a1ce77763eb9"
#define EV_SPECTRAL_DIR "sakai_multires_c/results/vae_c/iic/" EV_RUN_ID "_spectral_primary_v1"
#define EV_MANIFEST_SHA "620960e6b35da31bc4cf1a8a6da8679027d378e1d4b2ac73c5267c1c5311ae82"
#define EV_TRAIN_SHA "8c7a513b98dbe996261bc80965500ab15ab77f498ec9440858003152e42f926f"
#define EV_VALIDATION_SHA "e63109c88df7b3ee97317f94791f26df3fcaf8b937921cfb27967d01c2546467"
#define EV_SPECTRAL_LABELS_SHA "cc9e47a9d1a63c245bb53393c0fb65aa014015a95bbc395f3dc0f41554879ffd"
#define EV_SPECTRAL_ROWS_SHA "6e09d72e3b54f213a2096edffca627fd35f1f0e35778539f05a810a788d7c642"
#define EV_SPECTRAL_SPEC_SHA "12298c81739bb6002821fb90e6cc64819deb6d8c2347b68bba9c19c7afc4c0d3"

typedef struct {
    GSEventManifest manifest;
    GSSavedSplit train_split;
    GSSavedSplit validation_split;
    uint32_t full_labels[EV_N], full_clusters[EV_N];
    uint32_t train_labels[EV_TRAIN], train_clusters[EV_TRAIN];
    uint32_t validation_labels[EV_VALIDATION], validation_clusters[EV_VALIDATION];
} EVEvaluationData;

static const char *const ev_labels[EV_LABELS] = {
    "1080Lines", "1400Ripples", "Air_Compressor", "Blip", "Chirp",
    "Extremely_Loud", "Helix", "Koi_Fish", "Light_Modulation",
    "Low_Frequency_Burst", "Low_Frequency_Lines", "No_Glitch",
    "None_of_the_Above", "Paired_Doves", "Power_Line", "Repeating_Blips",
    "Scattered_Light", "Scratchy", "Tomte", "Violin_Mode",
    "Wandering_Line", "Whistle"
};

static const char *const ev_spectral_names[] = {
    "cluster_labels_i32.bin", "eigenvalues_f64.bin", "eigenvectors_f64.bin",
    "environment.txt", "rows.csv", "sha256.txt",
    "spectral_embedding_f64.bin", "spectral_spec.txt"
};

static const char *const ev_spectral_hashes[] = {
    EV_SPECTRAL_LABELS_SHA,
    "b6fcfdf7467b09e2591eb5de8130b20804aaa90a5e0c026fd67039317d369046",
    "e2b30186a7537c3adceef602e47743f29b5175da9a3adb7747187eafa349bb4d",
    "14407699bec9cd11defc8cabcc4fded1899c90d23a0854fc2f9987751486a30d",
    EV_SPECTRAL_ROWS_SHA,
    "9bab23875f9172ec8c45f432830932d351ad7e95ad2be10da1d4317755bf9fe4",
    "5a5e82f9224434d6c2b92456a0a65be56c3d054a43ca15852b5679a283105acb",
    EV_SPECTRAL_SPEC_SHA
};

static const char *const ev_output_names[] = {
    "metrics_full.csv", "metrics_train.csv", "metrics_validation.csv",
    "contingency_full.csv", "contingency_train.csv",
    "contingency_validation.csv", "cluster_summary_full.csv",
    "label_summary_full.csv", "evaluation_spec.txt"
};

static int ev_join(char path[EV_PATH_CAP], const char *dir, const char *name)
{
    int written;
    if (path == NULL || dir == NULL || name == NULL) return 0;
    written = snprintf(path, EV_PATH_CAP, "%s/%s", dir, name);
    return written > 0 && written < EV_PATH_CAP;
}

static int ev_hash_matches(const char *path, const char *expected)
{
    char actual[VAE_C_SHA256_HEX_SIZE];
    return vae_c_mu_file_sha256(path, actual) == VAE_C_MU_OK &&
           strcmp(actual, expected) == 0;
}

static int ev_label_index(const char *name)
{
    size_t i;
    for (i = 0; i < EV_LABELS; i++) {
        if (strcmp(name, ev_labels[i]) == 0) return (int)i;
    }
    return -1;
}

static void ev_trim_line(char *line)
{
    size_t len = strlen(line);
    while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
        line[--len] = '\0';
}

static int ev_parse_row(const char *line, unsigned long long *manifest_index,
                        unsigned long long *event_index, char event_id[128],
                        char split[16], unsigned long long *split_pos)
{
    int consumed = -1;
    if (line == NULL || manifest_index == NULL || event_index == NULL ||
        event_id == NULL || split == NULL || split_pos == NULL) return 0;
    return sscanf(line, "%llu,%llu,%127[^,],%15[^,],%llu%n",
                  manifest_index, event_index, event_id, split,
                  split_pos, &consumed) == 5 && consumed > 0 &&
           line[consumed] == '\0';
}

static int ev_read_clusters(const char *path, uint32_t clusters[EV_N])
{
    FILE *file = fopen(path, "rb");
    size_t i;
    int passed = file != NULL;
    if (!passed) return 0;
    for (i = 0; i < EV_N && passed; i++) {
        unsigned char b[4];
        uint32_t value;
        passed = fread(b, 1u, 4u, file) == 4u;
        if (!passed) break;
        value = (uint32_t)b[0] | ((uint32_t)b[1] << 8u) |
                ((uint32_t)b[2] << 16u) | ((uint32_t)b[3] << 24u);
        passed = value < EV_CLUSTERS;
        clusters[i] = value;
    }
    if (passed) passed = fgetc(file) == EOF && !ferror(file);
    if (fclose(file) != 0) passed = 0;
    return passed;
}

static int ev_validate_input_hashes(const char *root, char spectral[EV_PATH_CAP])
{
    static const char *const names[] = {
        "manifest_events_272.csv", "split_train_indices_seed123.csv",
        "split_val_indices_seed123.csv"
    };
    static const char *const hashes[] = {
        EV_MANIFEST_SHA, EV_TRAIN_SHA, EV_VALIDATION_SHA
    };
    char path[EV_PATH_CAP];
    size_t i;
    if (!ev_join(spectral, root, EV_SPECTRAL_DIR)) return 0;
    for (i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
        if (!ev_join(path, root, names[i]) || !ev_hash_matches(path, hashes[i]))
            return 0;
    }
    for (i = 0; i < sizeof(ev_spectral_names) /
                    sizeof(ev_spectral_names[0]); i++) {
        if (!ev_join(path, spectral, ev_spectral_names[i]) ||
            !ev_hash_matches(path, ev_spectral_hashes[i])) return 0;
    }
    return 1;
}

static int ev_load_data(const char *root, EVEvaluationData *data)
{
    char spectral[EV_PATH_CAP], path[EV_PATH_CAP], line[512];
    FILE *rows = NULL;
    unsigned char seen_train[EV_TRAIN] = {0};
    unsigned char seen_validation[EV_VALIDATION] = {0};
    size_t label_counts[EV_LABELS] = {0}, cluster_counts[EV_CLUSTERS] = {0};
    size_t i;
    int passed = 0;
    if (!ev_validate_input_hashes(root, spectral) ||
        !ev_join(path, root, "manifest_events_272.csv") ||
        gs_event_manifest_load(path, &data->manifest) != 0 ||
        data->manifest.count != EV_N) return 0;
    if (!ev_join(path, root, "split_train_indices_seed123.csv") ||
        gs_saved_split_load(path, "train", &data->manifest,
                            &data->train_split) != 0 ||
        data->train_split.count != EV_TRAIN ||
        !ev_join(path, root, "split_val_indices_seed123.csv") ||
        gs_saved_split_load(path, "validation", &data->manifest,
                            &data->validation_split) != 0 ||
        data->validation_split.count != EV_VALIDATION) return 0;
    if (!ev_join(path, spectral, "cluster_labels_i32.bin") ||
        !ev_read_clusters(path, data->full_clusters) ||
        !ev_join(path, spectral, "rows.csv")) return 0;
    rows = fopen(path, "rb");
    if (rows == NULL || fgets(line, sizeof(line), rows) == NULL) goto done;
    ev_trim_line(line);
    if (strcmp(line, "manifest_index,event_index,event_id,source_split,split_pos") != 0)
        goto done;
    for (i = 0; i < EV_N; i++) {
        unsigned long long mi, ei, pos;
        char id[128], split[16];
        const GSEventRecord *record = &data->manifest.records[i];
        int label;
        uint32_t cluster = data->full_clusters[i];
        if (fgets(line, sizeof(line), rows) == NULL) goto done;
        ev_trim_line(line);
        if (!ev_parse_row(line, &mi, &ei, id, split, &pos) ||
            mi != (unsigned long long)i ||
            ei != (unsigned long long)record->event_index ||
            strcmp(id, record->event_id) != 0 ||
            record->event_index != (int)(i + 1u)) goto done;
        label = ev_label_index(record->label);
        if (label < 0) goto done;
        data->full_labels[i] = (uint32_t)label;
        label_counts[label]++;
        cluster_counts[cluster]++;
        if (strcmp(split, "train") == 0) {
            if (pos >= EV_TRAIN || seen_train[pos] ||
                data->train_split.indices[pos] != i) goto done;
            seen_train[pos] = 1u;
            data->train_labels[pos] = (uint32_t)label;
            data->train_clusters[pos] = cluster;
        } else if (strcmp(split, "validation") == 0) {
            if (pos >= EV_VALIDATION || seen_validation[pos] ||
                data->validation_split.indices[pos] != i) goto done;
            seen_validation[pos] = 1u;
            data->validation_labels[pos] = (uint32_t)label;
            data->validation_clusters[pos] = cluster;
        } else goto done;
    }
    if (fgets(line, sizeof(line), rows) != NULL || ferror(rows)) goto done;
    for (i = 0; i < EV_LABELS; i++) if (label_counts[i] == 0u) goto done;
    for (i = 0; i < EV_CLUSTERS; i++) if (cluster_counts[i] == 0u) goto done;
    for (i = 0; i < EV_TRAIN; i++) if (!seen_train[i]) goto done;
    for (i = 0; i < EV_VALIDATION; i++) if (!seen_validation[i]) goto done;
    passed = 1;
done:
    if (rows != NULL && fclose(rows) != 0) passed = 0;
    return passed;
}

static void ev_free_data(EVEvaluationData *data)
{
    if (data == NULL) return;
    gs_saved_split_free(&data->validation_split);
    gs_saved_split_free(&data->train_split);
    gs_event_manifest_free(&data->manifest);
    free(data);
}

static int ev_metrics_valid(const GSBaselineClusteringMetrics *m, size_t n)
{
    uint64_t total = 0u;
    size_t i;
    if (m->rows != n || m->label_count != EV_LABELS ||
        m->cluster_count != EV_CLUSTERS || m->contingency == NULL ||
        m->label_totals == NULL || m->cluster_totals == NULL ||
        !isfinite(m->purity) || !isfinite(m->normalized_mutual_information) ||
        !isfinite(m->adjusted_rand_index) || !isfinite(m->homogeneity) ||
        !isfinite(m->completeness)) return 0;
    for (i = 0; i < EV_LABELS * EV_CLUSTERS; i++) total += m->contingency[i];
    return total == (uint64_t)n;
}

static int ev_compute(const EVEvaluationData *d,
                      GSBaselineClusteringMetrics *full,
                      GSBaselineClusteringMetrics *train,
                      GSBaselineClusteringMetrics *validation)
{
    size_t i;
    if (gs_baseline_clustering_metrics_compute(d->full_labels,
            d->full_clusters, EV_N, EV_LABELS, EV_CLUSTERS, full) != 0 ||
        gs_baseline_clustering_metrics_compute(d->train_labels,
            d->train_clusters, EV_TRAIN, EV_LABELS, EV_CLUSTERS, train) != 0 ||
        gs_baseline_clustering_metrics_compute(d->validation_labels,
            d->validation_clusters, EV_VALIDATION, EV_LABELS, EV_CLUSTERS,
            validation) != 0 ||
        !ev_metrics_valid(full, EV_N) ||
        !ev_metrics_valid(train, EV_TRAIN) ||
        !ev_metrics_valid(validation, EV_VALIDATION)) return 0;
    for (i = 0; i < EV_LABELS * EV_CLUSTERS; i++) {
        if (full->contingency[i] !=
            train->contingency[i] + validation->contingency[i]) return 0;
    }
    for (i = 0; i < EV_CLUSTERS; i++) {
        if (full->cluster_totals[i] == 0u) return 0;
    }
    return 1;
}

static FILE *ev_open_output(const char *dir, const char *name)
{
    char path[EV_PATH_CAP];
    if (!ev_join(path, dir, name)) return NULL;
    return fopen(path, "wb");
}

static int ev_write_metrics(const char *dir, const char *name,
                            const char *population,
                            const GSBaselineClusteringMetrics *m)
{
    FILE *f = ev_open_output(dir, name);
    int passed;
    if (f == NULL) return 0;
    passed = fprintf(f,
        "population,n_events,purity,nmi_arithmetic,ari,homogeneity,completeness\n"
        "%s,%zu,%.17g,%.17g,%.17g,%.17g,%.17g\n",
        population, m->rows, m->purity, m->normalized_mutual_information,
        m->adjusted_rand_index, m->homogeneity, m->completeness) > 0;
    if (fclose(f) != 0) passed = 0;
    return passed;
}

static int ev_write_contingency(const char *dir, const char *name,
                                const GSBaselineClusteringMetrics *m)
{
    FILE *f = ev_open_output(dir, name);
    size_t l, c;
    int passed = f != NULL;
    if (!passed) return 0;
    if (fprintf(f, "label_index,label") < 0) passed = 0;
    for (c = 0; c < EV_CLUSTERS && passed; c++)
        if (fprintf(f, ",cluster_%zu", c) < 0) passed = 0;
    if (passed && fprintf(f, ",row_total\n") < 0) passed = 0;
    for (l = 0; l < EV_LABELS && passed; l++) {
        if (fprintf(f, "%zu,%s", l, ev_labels[l]) < 0) passed = 0;
        for (c = 0; c < EV_CLUSTERS && passed; c++) {
            if (fprintf(f, ",%" PRIu64,
                    m->contingency[l * EV_CLUSTERS + c]) < 0) passed = 0;
        }
        if (passed && fprintf(f, ",%" PRIu64 "\n",
                              m->label_totals[l]) < 0) passed = 0;
    }
    if (fclose(f) != 0) passed = 0;
    return passed;
}

static int ev_write_cluster_summary(const char *dir,
                                    const GSBaselineClusteringMetrics *m)
{
    FILE *f = ev_open_output(dir, "cluster_summary_full.csv");
    size_t c, l, a, b;
    int passed = f != NULL;
    if (!passed) return 0;
    if (fprintf(f, "cluster_id,size,dominant_label_index,dominant_label,"
                   "dominant_count,dominant_fraction") < 0) passed = 0;
    for (a = 1; a <= 3 && passed; a++) {
        if (fprintf(f, ",top%zu_label_index,top%zu_label,top%zu_count,"
                       "top%zu_fraction", a, a, a, a) < 0) passed = 0;
    }
    if (passed && fputc('\n', f) == EOF) passed = 0;
    for (c = 0; c < EV_CLUSTERS && passed; c++) {
        size_t order[EV_LABELS];
        uint64_t size = m->cluster_totals[c];
        for (l = 0; l < EV_LABELS; l++) order[l] = l;
        for (a = 0; a < EV_LABELS; a++) {
            for (b = a + 1; b < EV_LABELS; b++) {
                uint64_t ca = m->contingency[order[a] * EV_CLUSTERS + c];
                uint64_t cb = m->contingency[order[b] * EV_CLUSTERS + c];
                if (cb > ca || (cb == ca && order[b] < order[a])) {
                    size_t swap = order[a]; order[a] = order[b]; order[b] = swap;
                }
            }
        }
        if (fprintf(f, "%zu,%" PRIu64 ",%zu,%s,%" PRIu64 ",%.17g",
            c, size, order[0], ev_labels[order[0]],
            m->contingency[order[0] * EV_CLUSTERS + c],
            (double)m->contingency[order[0] * EV_CLUSTERS + c] /
            (double)size) < 0) passed = 0;
        for (a = 0; a < 3 && passed; a++) {
            uint64_t count = m->contingency[order[a] * EV_CLUSTERS + c];
            if (fprintf(f, ",%zu,%s,%" PRIu64 ",%.17g", order[a],
                ev_labels[order[a]], count, (double)count / (double)size) < 0)
                passed = 0;
        }
        if (passed && fputc('\n', f) == EOF) passed = 0;
    }
    if (fclose(f) != 0) passed = 0;
    return passed;
}

static int ev_write_label_summary(const char *dir,
                                  const GSBaselineClusteringMetrics *m)
{
    FILE *f = ev_open_output(dir, "label_summary_full.csv");
    size_t l, c;
    int passed = f != NULL;
    if (!passed) return 0;
    if (fprintf(f, "label_index,label,total_count") < 0) passed = 0;
    for (c = 0; c < EV_CLUSTERS && passed; c++)
        if (fprintf(f, ",cluster_%zu", c) < 0) passed = 0;
    if (passed && fprintf(f, ",principal_cluster_id,principal_cluster_count,"
                            "principal_fraction\n") < 0) passed = 0;
    for (l = 0; l < EV_LABELS && passed; l++) {
        size_t best = 0;
        uint64_t total = m->label_totals[l];
        if (fprintf(f, "%zu,%s,%" PRIu64, l, ev_labels[l], total) < 0)
            passed = 0;
        for (c = 0; c < EV_CLUSTERS && passed; c++) {
            uint64_t count = m->contingency[l * EV_CLUSTERS + c];
            if (count > m->contingency[l * EV_CLUSTERS + best]) best = c;
            if (fprintf(f, ",%" PRIu64, count) < 0) passed = 0;
        }
        if (passed && fprintf(f, ",%zu,%" PRIu64 ",%.17g\n", best,
                m->contingency[l * EV_CLUSTERS + best],
                (double)m->contingency[l * EV_CLUSTERS + best] /
                (double)total) < 0) passed = 0;
    }
    if (fclose(f) != 0) passed = 0;
    return passed;
}

static int ev_write_spec(const char *dir, const char *commit)
{
    FILE *f = ev_open_output(dir, "evaluation_spec.txt");
    size_t i;
    int passed = f != NULL;
    if (!passed) return 0;
    if (fprintf(f,
        "format=MINI75_IIC_SPECTRAL_LABEL_EVALUATION_V1\n"
        "producer_commit=%s\n"
        "spectral_labels_sha256=%s\n"
        "spectral_rows_sha256=%s\n"
        "spectral_spec_sha256=%s\n"
        "label_source=manifest_events_272.csv\n"
        "label_source_sha256=%s\n"
        "train_split_sha256=%s\n"
        "validation_split_sha256=%s\n"
        "full_rows=8583\ntrain_rows=6866\nvalidation_rows=1717\n"
        "label_count=22\ncluster_count=36\n"
        "cluster_order=0..35_canonical_unpermuted\n"
        "metrics_module=baseline_clustering_metrics\n"
        "metrics=weighted_purity,nmi_arithmetic,ari,homogeneity,completeness\n"
        "contingency_layout=label_major_22x36\n"
        "post_hoc_only=true\ntransductive_full_population=true\n"
        "validation_out_of_sample=false\n"
        "spectral_partition_modified=false\n"
        "labels_used_for_clustering=false\nlabels_used_for_evaluation=true\n"
        "top3_tie_break=lower_label_index\n"
        "principal_cluster_tie_break=lower_cluster_id\n"
        "label_order=",
        commit, EV_SPECTRAL_LABELS_SHA, EV_SPECTRAL_ROWS_SHA,
        EV_SPECTRAL_SPEC_SHA, EV_MANIFEST_SHA, EV_TRAIN_SHA,
        EV_VALIDATION_SHA) < 0) passed = 0;
    for (i = 0; i < EV_LABELS && passed; i++) {
        if (fprintf(f, "%s%s", i == 0 ? "" : ",", ev_labels[i]) < 0)
            passed = 0;
    }
    if (passed && fputc('\n', f) == EOF) passed = 0;
    if (fclose(f) != 0) passed = 0;
    return passed;
}

static int ev_write_sha256(const char *dir)
{
    FILE *f = ev_open_output(dir, "sha256.txt");
    size_t i;
    int passed = f != NULL;
    if (!passed) return 0;
    for (i = 0; i < sizeof(ev_output_names) /
                    sizeof(ev_output_names[0]) && passed; i++) {
        char path[EV_PATH_CAP], hash[VAE_C_SHA256_HEX_SIZE];
        if (!ev_join(path, dir, ev_output_names[i]) ||
            vae_c_mu_file_sha256(path, hash) != VAE_C_MU_OK ||
            fprintf(f, "%s  %s\n", hash, ev_output_names[i]) < 0)
            passed = 0;
    }
    if (fclose(f) != 0) passed = 0;
    return passed;
}

static int ev_path_exists(const char *path)
{
    struct stat st;
    return stat(path, &st) == 0;
}

static int ev_make_directory(const char *path)
{
#ifdef _WIN32
    return CreateDirectoryA(path, NULL) != 0;
#else
    return mkdir(path, 0777) == 0;
#endif
}

static int ev_publish(const char *partial, const char *final)
{
    if (ev_path_exists(final)) return 0;
#ifdef _WIN32
    return MoveFileExA(partial, final, MOVEFILE_WRITE_THROUGH) != 0;
#else
    return rename(partial, final) == 0;
#endif
}

static int ev_write_outputs(const char *final, const char *commit,
                            const GSBaselineClusteringMetrics *full,
                            const GSBaselineClusteringMetrics *train,
                            const GSBaselineClusteringMetrics *validation)
{
    char partial[EV_PATH_CAP];
    int written = snprintf(partial, sizeof(partial), "%s.partial", final);
    if (written <= 0 || written >= (int)sizeof(partial) ||
        ev_path_exists(final) || ev_path_exists(partial) ||
        !ev_make_directory(partial)) return 0;
    if (!ev_write_metrics(partial, "metrics_full.csv", "full", full) ||
        !ev_write_metrics(partial, "metrics_train.csv", "train", train) ||
        !ev_write_metrics(partial, "metrics_validation.csv", "validation",
                          validation) ||
        !ev_write_contingency(partial, "contingency_full.csv", full) ||
        !ev_write_contingency(partial, "contingency_train.csv", train) ||
        !ev_write_contingency(partial, "contingency_validation.csv",
                              validation) ||
        !ev_write_cluster_summary(partial, full) ||
        !ev_write_label_summary(partial, full) ||
        !ev_write_spec(partial, commit) ||
        !ev_write_sha256(partial) ||
        !ev_publish(partial, final)) return 0;
    return 1;
}

static int ev_self_test(void)
{
    uint32_t labels[6] = {0, 0, 1, 1, 1, 0};
    uint32_t clusters[6] = {0, 1, 1, 2, 2, 0};
    GSBaselineClusteringMetrics m = {0};
    unsigned long long mi, ei, pos;
    char id[128], split[16];
    int passed = ev_parse_row("0,1,event_A,train,0", &mi, &ei, id,
                              split, &pos) && mi == 0u && ei == 1u &&
                 pos == 0u && strcmp(id, "event_A") == 0 &&
                 strcmp(split, "train") == 0 &&
                 !ev_parse_row("0,1,event_A,train,0,extra", &mi, &ei,
                               id, split, &pos);
    if (!passed || gs_baseline_clustering_metrics_compute(labels, clusters,
            6u, 2u, 3u, &m) != GS_BASELINE_CLUSTERING_METRICS_SUCCESS)
        return 0;
    passed = m.contingency[0] == 2u && m.contingency[1] == 1u &&
             m.contingency[2] == 0u && m.contingency[3] == 0u &&
             m.contingency[4] == 1u && m.contingency[5] == 2u &&
             fabs(m.purity - 5.0 / 6.0) < 1e-12 &&
             isfinite(m.normalized_mutual_information) &&
             isfinite(m.adjusted_rand_index);
    gs_baseline_clustering_metrics_free(&m);
    clusters[5] = 3u;
    passed = passed && gs_baseline_clustering_metrics_compute(labels,
        clusters, 6u, 2u, 3u, &m) ==
        GS_BASELINE_CLUSTERING_METRICS_ERROR_RANGE;
    gs_baseline_clustering_metrics_free(&m);
    return passed;
}

int main(int argc, char **argv)
{
    EVEvaluationData *data = NULL;
    GSBaselineClusteringMetrics full = {0}, train = {0}, validation = {0};
    const char *root = NULL, *output = NULL, *commit = NULL;
    int evaluate = 0, passed = 0;
    size_t i;
    (void)setlocale(LC_NUMERIC, "C");
    if (argc == 2 && strcmp(argv[1], "--self-test") == 0) {
        passed = ev_self_test();
        printf("SPECTRAL_EVALUATOR_TINY_PASS=%s\n", passed ? "true" : "false");
        return passed ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (argc == 4 && strcmp(argv[1], "--validate-only") == 0 &&
        strcmp(argv[2], "--repository-root") == 0) {
        root = argv[3];
    } else if (argc == 8 && strcmp(argv[1], "--evaluate") == 0 &&
               strcmp(argv[2], "--repository-root") == 0 &&
               strcmp(argv[4], "--output-dir") == 0 &&
               strcmp(argv[6], "--producer-commit") == 0) {
        evaluate = 1; root = argv[3]; output = argv[5]; commit = argv[7];
        if (strlen(commit) != 40u) return EXIT_FAILURE;
        for (i = 0; i < 40u; i++) {
            if (!((commit[i] >= '0' && commit[i] <= '9') ||
                  (commit[i] >= 'a' && commit[i] <= 'f')))
                return EXIT_FAILURE;
        }
        if (ev_path_exists(output)) return EXIT_FAILURE;
    } else {
        fprintf(stderr, "usage: %s --validate-only --repository-root ROOT | "
                "--evaluate --repository-root ROOT --output-dir DIR "
                "--producer-commit SHA40 | --self-test\n", argv[0]);
        return EXIT_FAILURE;
    }
    data = (EVEvaluationData *)calloc(1u, sizeof(*data));
    if (data == NULL) return EXIT_FAILURE;
    if (!ev_load_data(root, data)) {
        fprintf(stderr, "spectral/label alignment or source SHA mismatch\n");
        goto done;
    }
    printf("LABEL_ALIGNMENT_PASS=true\nFULL_N=8583\nTRAIN_N=6866\n"
           "VALIDATION_N=1717\nLABEL_COUNT=22\nCLUSTER_COUNT=36\n");
    if (!evaluate) {
        printf("METRICS_COMPUTED=false\nOUTPUT_CREATED=false\n");
        passed = 1;
        goto done;
    }
    if (!ev_compute(data, &full, &train, &validation) ||
        !ev_write_outputs(output, commit, &full, &train, &validation)) {
        fprintf(stderr, "metric computation or atomic publish failed\n");
        goto done;
    }
    printf("FULL_PURITY=%.17g\nFULL_NMI=%.17g\nFULL_ARI=%.17g\n"
           "FULL_HOMOGENEITY=%.17g\nFULL_COMPLETENESS=%.17g\n",
           full.purity, full.normalized_mutual_information,
           full.adjusted_rand_index, full.homogeneity, full.completeness);
    printf("TRAIN_PURITY=%.17g\nTRAIN_NMI=%.17g\nTRAIN_ARI=%.17g\n"
           "TRAIN_HOMOGENEITY=%.17g\nTRAIN_COMPLETENESS=%.17g\n",
           train.purity, train.normalized_mutual_information,
           train.adjusted_rand_index, train.homogeneity, train.completeness);
    printf("VALIDATION_PURITY=%.17g\nVALIDATION_NMI=%.17g\n"
           "VALIDATION_ARI=%.17g\nVALIDATION_HOMOGENEITY=%.17g\n"
           "VALIDATION_COMPLETENESS=%.17g\n",
           validation.purity, validation.normalized_mutual_information,
           validation.adjusted_rand_index, validation.homogeneity,
           validation.completeness);
    printf("OUTPUT_CREATED=true\n");
    passed = 1;
done:
    gs_baseline_clustering_metrics_free(&validation);
    gs_baseline_clustering_metrics_free(&train);
    gs_baseline_clustering_metrics_free(&full);
    ev_free_data(data);
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
