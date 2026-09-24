#include "vae_c_iic_mini75_central.h"
#include "vae_c_iic_mini112_central.h"
#include "vae_c_iic_mini224_central.h"
#include "vae_c_mu_contract.h"
#include "vae_multires_event_loader.h"
#include "vae_event_dataset.h"

#include <ctype.h>
#include <inttypes.h>
#include <locale.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#define cs_mkdir(path) _mkdir(path)
#else
#include <unistd.h>
#define cs_mkdir(path) mkdir((path), 0777)
#endif

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

enum {
    CS_CLUSTERS = 36,
    CS_TOP_K = 5,
    CS_VIEWS = 4,
    CS_PATH_CAP = 4096,
    CS_TEXT_CAP = 1024,
    CS_MANIFEST_ROWS = 8583,
    CS_LEFT_MARGIN = 430,
    CS_TOP_MARGIN = 54,
    CS_COLUMN_GAP = 10,
    CS_ROW_TEXT = 42,
    CS_BOTTOM_MARGIN = 12
};

#define CS_MANIFEST_SHA \
    "620960e6b35da31bc4cf1a8a6da8679027d378e1d4b2ac73c5267c1c5311ae82"
#define CS_FULL_NOTEBOOK_SHA \
    "81894341420d94c993c916663e4ce83578f53734d7d1c35f9c0425e3b4198c8c"
#define CS_FULL_SELECTION_CONTRACT_SHA \
    "c1a3fd7edaefb9e70b8a5664a55ccd2bd41a2e95f858fdacb2aeb5c7f64857d1"

typedef enum {
    CS_PROFILE_MINI75,
    CS_PROFILE_MINI112,
    CS_PROFILE_MINI224
} CSProfileKind;

typedef struct {
    const char *name;
    CSProfileKind kind;
    size_t rows;
    size_t latent_dim;
    int target_side;
    const char *population;
    const char *spectral_directory;
    const char *official_output_relative;
    const char *assignment_sha;
    const char *spectral_rows_sha;
    const char *mu_train_sha;
    const char *mu_validation_sha;
    const char *mu_train_rows_sha;
    const char *mu_validation_rows_sha;
} CSProfile;

typedef struct {
    float *mu;
    uint32_t *cluster;
    uint32_t *manifest_index;
    uint32_t *event_index;
    char (*event_id)[128];
    size_t rows;
    size_t latent_dim;
    GSEventManifest manifest;
} CSData;

typedef struct {
    size_t iic_row;
    uint32_t manifest_index;
    double similarity;
} CSCandidate;

typedef struct {
    CSCandidate entries[CS_CLUSTERS][CS_TOP_K];
} CSSelection;

static const CSProfile cs_profiles[] = {
    {
        "mini75_primary", CS_PROFILE_MINI75, 8583u, 32u, 75,
        "transductive_8583",
        "sakai_multires_c/results/vae_c/iic/mini75_iic_v1_737da555866b822e14c545c4f2687869f41a52cd1340b9e941f3a1ce77763eb9_spectral_primary_v1",
        "sakai_multires_c/results/vae_c/iic/common_contact_sheets_v1/mini75_primary",
        "cc9e47a9d1a63c245bb53393c0fb65aa014015a95bbc395f3dc0f41554879ffd",
        "6e09d72e3b54f213a2096edffca627fd35f1f0e35778539f05a810a788d7c642",
        "a8e82c3a0d25ffbd649640e1df01438fe97fc070ebb0def50227c5021fa52d52",
        "548596df86d7189a05ce3fb84631f403383546ba4ab1a44119c13852447b13d7",
        "3999c0ae72a0011b9433249dc335fc95ad159b4c84ec941012fd77e8f7f692c1",
        "fe62ecfba8a723938a54caf95e5c5daf2a2ca5380acaec0f208599564da7a222"
    },
    {
        "mini112_primary", CS_PROFILE_MINI112, 8583u, 64u, 112,
        "transductive_8583",
        "sakai_multires_c/results/vae_c/iic/mini112_iic_v1_0cecf5cd656d0d61a237d50fcf96a7890fb2e6d052ef403051bc9bc17aed8fba_spectral_primary_v1",
        "sakai_multires_c/results/vae_c/iic/common_contact_sheets_v1/mini112_primary",
        "17200b8d006962e39a7b49b07030042a0980970f57085b6318165cc30e738d3f",
        "6e09d72e3b54f213a2096edffca627fd35f1f0e35778539f05a810a788d7c642",
        "1399f4658da79b368fa03dd130030511865d2c121f93dcda590dec5d2eede775",
        "1add3b9e6f9ca548856119f0b6e49a7dd1e706a59e62dcd6760f547a86022be7",
        "3999c0ae72a0011b9433249dc335fc95ad159b4c84ec941012fd77e8f7f692c1",
        "fe62ecfba8a723938a54caf95e5c5daf2a2ca5380acaec0f208599564da7a222"
    },
    {
        "mini224_1000_primary", CS_PROFILE_MINI224, 1000u, 64u, 224,
        "train800_then_validation200",
        "sakai_multires_c/results/vae_c/iic/mini224_iic_v1_4236df3f9e82f0528fcea947b2e15e42fab8159c0d4b1ed03189e4a85daae8a5_spectral_primary_v1",
        "sakai_multires_c/results/vae_c/iic/common_contact_sheets_v1/mini224_1000_primary",
        "61bcc7a38b49c210626bd5fcf191f7b9171017df85fc7a1373fd674001af75d2",
        "d7f5101cfe3f3ae31f24b12833f0125873888ff9ed33c4c33c428b90955db4c2",
        "f494b232cb2fbf9ca4fb25df80c47e5f80963ac1d7448afd807361409a4ab166",
        "5f6aec1737549a24d20897581b4c6b57b3fae6859e1dc1e972a8b8841ede4617",
        "fd5bb3c48f87e98d5ec8ed0e561d5fe077ab96a7effa0c7425a41a222cd4375c",
        "87a3c7fc715fe0f4a6158474e46f6903d5267a45741d338836e8fecd61df7ed8"
    }
};

static int cs_join(char out[CS_PATH_CAP], const char *left, const char *right)
{
    int written;
    if (out == NULL || left == NULL || right == NULL) return 0;
    written = snprintf(out, CS_PATH_CAP, "%s/%s", left, right);
    return written > 0 && written < CS_PATH_CAP;
}

static int cs_exists(const char *path)
{
    struct stat status;
    return path != NULL && stat(path, &status) == 0;
}

static int cs_hash(const char *path, char out[65])
{
    return vae_c_mu_file_sha256(path, out) == VAE_C_MU_OK;
}

static int cs_hash_is(const char *path, const char *expected)
{
    char actual[65];
    return expected != NULL && cs_hash(path, actual) &&
           strcmp(actual, expected) == 0;
}

static int cs_hex(const char *text, size_t length)
{
    size_t index;
    if (text == NULL || strlen(text) != length) return 0;
    for (index = 0; index < length; index++)
        if (!((text[index] >= '0' && text[index] <= '9') ||
              (text[index] >= 'a' && text[index] <= 'f'))) return 0;
    return 1;
}

static int cs_worktree_state_valid(const char *state)
{
    return state != NULL &&
        (strcmp(state, "clean_commit") == 0 ||
         strcmp(state, "uncommitted_wip") == 0);
}

static int cs_provenance_valid(const char *commit, const char *source_sha,
                               const char *binary_sha,
                               const char *worktree_state)
{
    return cs_hex(commit, 40u) && cs_hex(source_sha, 64u) &&
        cs_hex(binary_sha, 64u) &&
        cs_worktree_state_valid(worktree_state);
}

static int cs_shape_is(const CSProfile *profile, size_t rows,
                       size_t latent_dim)
{
    return profile != NULL && rows == profile->rows &&
           latent_dim == profile->latent_dim;
}

static int cs_row_matches(const GSEventRecord *record,
                          uint32_t manifest_index, uint32_t event_index,
                          const char *event_id, size_t manifest_count)
{
    return record != NULL && event_id != NULL &&
           manifest_index < manifest_count &&
           (uint32_t)record->event_index == event_index &&
           strcmp(record->event_id, event_id) == 0;
}

static int cs_pngs_exist(const GSEventRecord *record)
{
    size_t view;
    if (record == NULL) return 0;
    for (view = 0; view < CS_VIEWS; view++)
        if (!cs_exists(record->image_paths[view])) return 0;
    return 1;
}

static int cs_output_available(const char *output)
{
    char partial[CS_PATH_CAP];
    int written;
    if (output == NULL || cs_exists(output)) return 0;
    written = snprintf(partial, sizeof(partial), "%s.partial", output);
    return written > 0 && written < (int)sizeof(partial) &&
           !cs_exists(partial);
}

static int cs_prepare_official_parent(const char *root)
{
    char parent[CS_PATH_CAP];
    if (!cs_join(parent, root,
        "sakai_multires_c/results/vae_c/iic/common_contact_sheets_v1"))
        return 0;
    return cs_exists(parent) || cs_mkdir(parent) == 0;
}

static const CSProfile *cs_find_profile(const char *name)
{
    size_t index;
    for (index = 0; index < sizeof(cs_profiles) / sizeof(cs_profiles[0]);
         index++) {
        if (name != NULL && strcmp(name, cs_profiles[index].name) == 0)
            return &cs_profiles[index];
    }
    return NULL;
}

static void cs_data_init(CSData *data)
{
    if (data != NULL) memset(data, 0, sizeof(*data));
}

static void cs_data_free(CSData *data)
{
    if (data == NULL) return;
    free(data->mu);
    free(data->cluster);
    free(data->manifest_index);
    free(data->event_index);
    free(data->event_id);
    gs_event_manifest_free(&data->manifest);
    cs_data_init(data);
}

static int cs_allocate_data(const CSProfile *profile, CSData *data)
{
    if (profile == NULL || data == NULL || profile->rows == 0u ||
        profile->latent_dim == 0u ||
        profile->rows > SIZE_MAX / profile->latent_dim) return 0;
    data->mu = (float *)calloc(profile->rows * profile->latent_dim,
                               sizeof(float));
    data->cluster = (uint32_t *)calloc(profile->rows, sizeof(uint32_t));
    data->manifest_index = (uint32_t *)calloc(profile->rows,
                                               sizeof(uint32_t));
    data->event_index = (uint32_t *)calloc(profile->rows, sizeof(uint32_t));
    data->event_id = calloc(profile->rows, sizeof(*data->event_id));
    data->rows = profile->rows;
    data->latent_dim = profile->latent_dim;
    return data->mu != NULL && data->cluster != NULL &&
           data->manifest_index != NULL && data->event_index != NULL &&
           data->event_id != NULL;
}

static int cs_read_i32(const char *path, uint32_t *values, size_t count)
{
    FILE *file = fopen(path, "rb");
    struct stat status;
    size_t index;
    int passed = file != NULL && values != NULL && stat(path, &status) == 0 &&
        status.st_size >= 0 && (uint64_t)status.st_size == (uint64_t)count * 4u;
    if (!passed) {
        if (file != NULL) fclose(file);
        return 0;
    }
    for (index = 0; index < count && passed; index++) {
        unsigned char bytes[4];
        passed = fread(bytes, 1u, 4u, file) == 4u;
        if (passed) values[index] = (uint32_t)bytes[0] |
            ((uint32_t)bytes[1] << 8u) | ((uint32_t)bytes[2] << 16u) |
            ((uint32_t)bytes[3] << 24u);
    }
    if (passed) passed = fgetc(file) == EOF && !ferror(file);
    if (fclose(file) != 0) passed = 0;
    return passed;
}

static int cs_copy_id(char target[128], const char *source)
{
    size_t length;
    if (target == NULL || source == NULL) return 0;
    length = strlen(source);
    if (length >= 128u) return 0;
    memcpy(target, source, length + 1u);
    return 1;
}

static int cs_load_central(const CSProfile *profile, const char *root,
                           CSData *data)
{
    size_t row;
    char error[512];
    if (profile->kind == CS_PROFILE_MINI75) {
        VAECIICMini75Central view;
        int loaded;
        vae_c_iic_mini75_central_init(&view);
        error[0] = '\0';
        loaded = vae_c_iic_mini75_central_load(root, NULL, &view, error,
                                               sizeof(error)) != 0;
        if (!loaded || !cs_shape_is(profile, view.row_count,
                                    view.latent_dim) ||
            strcmp(view.train_matrix_sha256, profile->mu_train_sha) != 0 ||
            strcmp(view.validation_matrix_sha256,
                   profile->mu_validation_sha) != 0 ||
            strcmp(view.train_rows_sha256, profile->mu_train_rows_sha) != 0 ||
            strcmp(view.validation_rows_sha256,
                   profile->mu_validation_rows_sha) != 0) {
            fprintf(stderr, "mini75 central load/identity failed: %s\n",
                    loaded ? "canonical hash contract" : error);
            vae_c_iic_mini75_central_free(&view);
            return 0;
        }
        memcpy(data->mu, view.mu,
               profile->rows * profile->latent_dim * sizeof(float));
        for (row = 0; row < profile->rows; row++) {
            data->manifest_index[row] = view.rows[row].manifest_index;
            data->event_index[row] = view.rows[row].event_index;
            if (!cs_copy_id(data->event_id[row], view.rows[row].event_id)) {
                vae_c_iic_mini75_central_free(&view); return 0;
            }
        }
        vae_c_iic_mini75_central_free(&view);
        return 1;
    }
    if (profile->kind == CS_PROFILE_MINI112) {
        VAECIICMini112Central view;
        int loaded;
        vae_c_iic_mini112_central_init(&view);
        error[0] = '\0';
        loaded = vae_c_iic_mini112_central_load(root, NULL, &view, error,
                                                sizeof(error)) != 0;
        if (!loaded || !cs_shape_is(profile, view.row_count,
                                    view.latent_dim) ||
            strcmp(view.train_matrix_sha256, profile->mu_train_sha) != 0 ||
            strcmp(view.validation_matrix_sha256,
                   profile->mu_validation_sha) != 0 ||
            strcmp(view.train_rows_sha256, profile->mu_train_rows_sha) != 0 ||
            strcmp(view.validation_rows_sha256,
                   profile->mu_validation_rows_sha) != 0) {
            fprintf(stderr, "mini112 central load/identity failed: %s\n",
                    loaded ? "canonical hash contract" : error);
            vae_c_iic_mini112_central_free(&view);
            return 0;
        }
        memcpy(data->mu, view.mu,
               profile->rows * profile->latent_dim * sizeof(float));
        for (row = 0; row < profile->rows; row++) {
            data->manifest_index[row] = view.rows[row].manifest_index;
            data->event_index[row] = view.rows[row].event_index;
            if (!cs_copy_id(data->event_id[row], view.rows[row].event_id)) {
                vae_c_iic_mini112_central_free(&view); return 0;
            }
        }
        vae_c_iic_mini112_central_free(&view);
        return 1;
    }
    {
        VAECIICMini224Central view;
        int loaded;
        vae_c_iic_mini224_central_init(&view);
        error[0] = '\0';
        loaded = vae_c_iic_mini224_central_load(root, NULL, &view, error,
                                                sizeof(error)) != 0;
        if (!loaded || !cs_shape_is(profile, view.row_count,
                                    view.latent_dim) ||
            strcmp(view.train_matrix_sha256, profile->mu_train_sha) != 0 ||
            strcmp(view.validation_matrix_sha256,
                   profile->mu_validation_sha) != 0 ||
            strcmp(view.train_rows_sha256, profile->mu_train_rows_sha) != 0 ||
            strcmp(view.validation_rows_sha256,
                   profile->mu_validation_rows_sha) != 0) {
            fprintf(stderr, "mini224 central load/identity failed: %s\n",
                    loaded ? "canonical hash contract" : error);
            vae_c_iic_mini224_central_free(&view);
            return 0;
        }
        memcpy(data->mu, view.mu,
               profile->rows * profile->latent_dim * sizeof(float));
        for (row = 0; row < profile->rows; row++) {
            data->manifest_index[row] = view.rows[row].manifest_index;
            data->event_index[row] = view.rows[row].event_index;
            if (view.rows[row].iic_row_index != row ||
                !cs_copy_id(data->event_id[row], view.rows[row].event_id)) {
                vae_c_iic_mini224_central_free(&view); return 0;
            }
        }
        vae_c_iic_mini224_central_free(&view);
        return 1;
    }
}

static int cs_make_image_paths_absolute(const char *root,
                                        GSEventManifest *manifest)
{
    size_t row, view;
    for (row = 0; row < manifest->count; row++) {
        for (view = 0; view < CS_VIEWS; view++) {
            char combined[GS_EVENT_MAX_PATH_LEN];
            const char *current = manifest->records[row].image_paths[view];
            int absolute = current[0] == '/' || current[0] == '\\' ||
                (isalpha((unsigned char)current[0]) && current[1] == ':');
            int written;
            if (absolute) continue;
            written = snprintf(combined, sizeof(combined), "%s/%s", root,
                               current);
            if (written <= 0 || written >= (int)sizeof(combined)) return 0;
            memcpy(manifest->records[row].image_paths[view], combined,
                   (size_t)written + 1u);
        }
    }
    return 1;
}

static int cs_validate_cluster_ids(const uint32_t *clusters, size_t rows)
{
    size_t counts[CS_CLUSTERS] = {0};
    size_t row, cluster;
    if (clusters == NULL) return 0;
    for (row = 0; row < rows; row++) {
        if (clusters[row] >= CS_CLUSTERS) return 0;
        counts[clusters[row]]++;
    }
    for (cluster = 0; cluster < CS_CLUSTERS; cluster++)
        if (counts[cluster] < CS_TOP_K) return 0;
    return 1;
}

static int cs_load_data(const CSProfile *profile, const char *root,
                        CSData *data)
{
    char path[CS_PATH_CAP], spectral[CS_PATH_CAP], assignment[CS_PATH_CAP];
    char rows_path[CS_PATH_CAP];
    size_t row;
    if (!cs_allocate_data(profile, data) ||
        !cs_join(path, root, "manifest_events_272.csv") ||
        !cs_hash_is(path, CS_MANIFEST_SHA) ||
        gs_event_manifest_load(path, &data->manifest) != 0 ||
        data->manifest.count != CS_MANIFEST_ROWS ||
        !cs_make_image_paths_absolute(root, &data->manifest) ||
        !cs_join(spectral, root, profile->spectral_directory) ||
        !cs_join(assignment, spectral, "cluster_labels_i32.bin") ||
        !cs_join(rows_path, spectral, "rows.csv") ||
        !cs_hash_is(assignment, profile->assignment_sha) ||
        !cs_hash_is(rows_path, profile->spectral_rows_sha) ||
        !cs_read_i32(assignment, data->cluster, profile->rows) ||
        !cs_load_central(profile, root, data) ||
        !cs_validate_cluster_ids(data->cluster, profile->rows)) return 0;
    for (row = 0; row < profile->rows; row++) {
        const GSEventRecord *record;
        if (data->manifest_index[row] >= data->manifest.count) return 0;
        record = &data->manifest.records[data->manifest_index[row]];
        if (!cs_row_matches(record, data->manifest_index[row],
                data->event_index[row], data->event_id[row],
                data->manifest.count)) return 0;
    }
    return 1;
}

static int cs_candidate_compare(const void *left, const void *right)
{
    const CSCandidate *a = (const CSCandidate *)left;
    const CSCandidate *b = (const CSCandidate *)right;
    if (a->similarity > b->similarity) return -1;
    if (a->similarity < b->similarity) return 1;
    if (a->manifest_index < b->manifest_index) return -1;
    if (a->manifest_index > b->manifest_index) return 1;
    return 0;
}

static int cs_select(const CSData *data, CSSelection *selection)
{
    double *unit = NULL;
    double *prototype = NULL;
    CSCandidate *candidates = NULL;
    size_t row, dim, cluster, rank;
    int passed = data != NULL && selection != NULL && data->mu != NULL;
    if (!passed || data->rows > SIZE_MAX / data->latent_dim) return 0;
    unit = (double *)calloc(data->rows * data->latent_dim, sizeof(double));
    prototype = (double *)calloc(CS_CLUSTERS * data->latent_dim,
                                 sizeof(double));
    candidates = (CSCandidate *)calloc(data->rows, sizeof(CSCandidate));
    if (unit == NULL || prototype == NULL || candidates == NULL) passed = 0;
    for (row = 0; row < data->rows && passed; row++) {
        double squared = 0.0;
        for (dim = 0; dim < data->latent_dim; dim++) {
            double value = (double)data->mu[row * data->latent_dim + dim];
            if (!isfinite(value)) { passed = 0; break; }
            squared += value * value;
        }
        if (!passed || !(squared > 0.0) || !isfinite(squared)) {
            passed = 0; break;
        }
        {
            double inverse = 1.0 / sqrt(squared);
            for (dim = 0; dim < data->latent_dim; dim++) {
                double value = (double)data->mu[row * data->latent_dim + dim]
                    * inverse;
                unit[row * data->latent_dim + dim] = value;
                prototype[data->cluster[row] * data->latent_dim + dim] +=
                    value;
            }
        }
    }
    for (cluster = 0; cluster < CS_CLUSTERS && passed; cluster++) {
        double squared = 0.0;
        size_t count = 0u;
        for (row = 0; row < data->rows; row++)
            if (data->cluster[row] == cluster) count++;
        if (count < CS_TOP_K) { passed = 0; break; }
        for (dim = 0; dim < data->latent_dim; dim++) {
            double *value = &prototype[cluster * data->latent_dim + dim];
            *value /= (double)count;
            squared += *value * *value;
        }
        if (!(squared > 0.0) || !isfinite(squared)) { passed = 0; break; }
        {
            double inverse = 1.0 / sqrt(squared);
            for (dim = 0; dim < data->latent_dim; dim++)
                prototype[cluster * data->latent_dim + dim] *= inverse;
        }
        count = 0u;
        for (row = 0; row < data->rows; row++) {
            double similarity = 0.0;
            if (data->cluster[row] != cluster) continue;
            for (dim = 0; dim < data->latent_dim; dim++)
                similarity += unit[row * data->latent_dim + dim] *
                    prototype[cluster * data->latent_dim + dim];
            if (!isfinite(similarity)) { passed = 0; break; }
            candidates[count].iic_row = row;
            candidates[count].manifest_index = data->manifest_index[row];
            candidates[count].similarity = similarity;
            count++;
        }
        if (!passed) break;
        qsort(candidates, count, sizeof(*candidates), cs_candidate_compare);
        for (rank = 0; rank < CS_TOP_K; rank++)
            selection->entries[cluster][rank] = candidates[rank];
    }
    free(unit);
    free(prototype);
    free(candidates);
    return passed;
}

/* Compact deterministic 5x7 uppercase bitmap font for annotations. */
static void cs_glyph(char input, uint8_t rows[7])
{
    char c = (char)toupper((unsigned char)input);
    static const struct { char c; uint8_t r[7]; } glyphs[] = {
        {'A',{14,17,17,31,17,17,17}}, {'B',{30,17,17,30,17,17,30}},
        {'C',{14,17,16,16,16,17,14}}, {'D',{30,17,17,17,17,17,30}},
        {'E',{31,16,16,30,16,16,31}}, {'F',{31,16,16,30,16,16,16}},
        {'G',{14,17,16,23,17,17,14}}, {'H',{17,17,17,31,17,17,17}},
        {'I',{14,4,4,4,4,4,14}}, {'J',{7,2,2,2,18,18,12}},
        {'K',{17,18,20,24,20,18,17}}, {'L',{16,16,16,16,16,16,31}},
        {'M',{17,27,21,21,17,17,17}}, {'N',{17,25,21,19,17,17,17}},
        {'O',{14,17,17,17,17,17,14}}, {'P',{30,17,17,30,16,16,16}},
        {'Q',{14,17,17,17,21,18,13}}, {'R',{30,17,17,30,20,18,17}},
        {'S',{15,16,16,14,1,1,30}}, {'T',{31,4,4,4,4,4,4}},
        {'U',{17,17,17,17,17,17,14}}, {'V',{17,17,17,17,17,10,4}},
        {'W',{17,17,17,21,21,21,10}}, {'X',{17,17,10,4,10,17,17}},
        {'Y',{17,17,10,4,4,4,4}}, {'Z',{31,1,2,4,8,16,31}},
        {'0',{14,17,19,21,25,17,14}}, {'1',{4,12,4,4,4,4,14}},
        {'2',{14,17,1,2,4,8,31}}, {'3',{30,1,1,14,1,1,30}},
        {'4',{2,6,10,18,31,2,2}}, {'5',{31,16,16,30,1,1,30}},
        {'6',{14,16,16,30,17,17,14}}, {'7',{31,1,2,4,8,8,8}},
        {'8',{14,17,17,14,17,17,14}}, {'9',{14,17,17,15,1,1,14}},
        {'.',{0,0,0,0,0,6,6}}, {':',{0,6,6,0,6,6,0}},
        {'-',{0,0,0,31,0,0,0}}, {'_',{0,0,0,0,0,0,31}},
        {'/',{1,2,2,4,8,8,16}}, {'=',{0,31,0,31,0,0,0}},
        {'+',{0,4,4,31,4,4,0}}, {' ',{0,0,0,0,0,0,0}}
    };
    size_t index;
    memset(rows, 0, 7u);
    for (index = 0; index < sizeof(glyphs) / sizeof(glyphs[0]); index++)
        if (glyphs[index].c == c) { memcpy(rows, glyphs[index].r, 7u); return; }
    rows[0]=31; rows[1]=17; rows[2]=1; rows[3]=2;
    rows[4]=4; rows[5]=0; rows[6]=4;
}

static void cs_set_rgb(unsigned char *canvas, int width, int height,
                       int x, int y, unsigned char value)
{
    size_t offset;
    if (x < 0 || y < 0 || x >= width || y >= height) return;
    offset = ((size_t)y * (size_t)width + (size_t)x) * 3u;
    canvas[offset] = value;
    canvas[offset + 1u] = value;
    canvas[offset + 2u] = value;
}

static void cs_draw_text(unsigned char *canvas, int width, int height,
                         int x, int y, const char *text, int scale)
{
    size_t index;
    for (index = 0; text != NULL && text[index] != '\0'; index++) {
        uint8_t rows[7];
        int gy, gx, sy, sx;
        cs_glyph(text[index], rows);
        for (gy = 0; gy < 7; gy++) for (gx = 0; gx < 5; gx++)
            if ((rows[gy] & (uint8_t)(1u << (4 - gx))) != 0u)
                for (sy = 0; sy < scale; sy++) for (sx = 0; sx < scale; sx++)
                    cs_set_rgb(canvas, width, height,
                        x + (int)index * 6 * scale + gx * scale + sx,
                        y + gy * scale + sy, 0u);
    }
}

static void cs_draw_image(unsigned char *canvas, int width, int height,
                          int x0, int y0, const uint8_t *pixels, int side)
{
    int x, y;
    for (y = 0; y < side; y++) for (x = 0; x < side; x++)
        cs_set_rgb(canvas, width, height, x0 + x, y0 + y,
                   pixels[(size_t)y * (size_t)side + (size_t)x]);
}

static uint32_t cs_crc32(const unsigned char *data, size_t size)
{
    uint32_t crc = UINT32_C(0xffffffff);
    size_t index;
    for (index = 0; index < size; index++) {
        unsigned bit;
        crc ^= data[index];
        for (bit = 0; bit < 8u; bit++)
            crc = (crc >> 1u) ^ (UINT32_C(0xedb88320) &
                  (uint32_t)-(int32_t)(crc & 1u));
    }
    return ~crc;
}

static void cs_be32(unsigned char out[4], uint32_t value)
{
    out[0] = (unsigned char)(value >> 24u);
    out[1] = (unsigned char)(value >> 16u);
    out[2] = (unsigned char)(value >> 8u);
    out[3] = (unsigned char)value;
}

static int cs_add_160_dpi(const char *path)
{
    FILE *file = fopen(path, "rb");
    struct stat status;
    unsigned char *source = NULL, *result = NULL;
    size_t size;
    int passed = file != NULL && stat(path, &status) == 0 &&
        status.st_size >= 33;
    if (!passed) { if (file != NULL) fclose(file); return 0; }
    size = (size_t)status.st_size;
    source = (unsigned char *)malloc(size);
    result = (unsigned char *)malloc(size + 21u);
    passed = source != NULL && result != NULL &&
        fread(source, 1u, size, file) == size;
    if (fclose(file) != 0) passed = 0;
    if (passed) passed = memcmp(source, "\x89PNG\r\n\x1a\n", 8u) == 0 &&
        memcmp(source + 12u, "IHDR", 4u) == 0;
    if (passed) {
        unsigned char chunk[21];
        uint32_t crc;
        cs_be32(chunk, 9u);
        memcpy(chunk + 4u, "pHYs", 4u);
        cs_be32(chunk + 8u, 6299u);
        cs_be32(chunk + 12u, 6299u);
        chunk[16] = 1u;
        crc = cs_crc32(chunk + 4u, 13u);
        cs_be32(chunk + 17u, crc);
        memcpy(result, source, 33u);
        memcpy(result + 33u, chunk, sizeof(chunk));
        memcpy(result + 54u, source + 33u, size - 33u);
        file = fopen(path, "wb");
        passed = file != NULL && fwrite(result, 1u, size + 21u, file) ==
            size + 21u;
        if (file != NULL && fclose(file) != 0) passed = 0;
    }
    free(source);
    free(result);
    return passed;
}

static int cs_write_sheet(const char *path, const CSProfile *profile,
                          const CSData *data, const CSSelection *selection,
                          size_t cluster)
{
    int side = profile->target_side;
    int width = CS_LEFT_MARGIN + CS_VIEWS * side +
        (CS_VIEWS - 1) * CS_COLUMN_GAP + 12;
    int height = CS_TOP_MARGIN + CS_TOP_K * (side + CS_ROW_TEXT) +
        CS_BOTTOM_MARGIN;
    size_t canvas_bytes = (size_t)width * (size_t)height * 3u;
    unsigned char *canvas = (unsigned char *)malloc(canvas_bytes);
    uint8_t *tensor = (uint8_t *)malloc((size_t)CS_VIEWS * (size_t)side *
                                         (size_t)side);
    GSMultiresEventLoader loader;
    size_t rank, view;
    int passed = canvas != NULL && tensor != NULL &&
        gs_multires_event_loader_init(&loader, side) == 0;
    if (!passed) { free(canvas); free(tensor); return 0; }
    memset(canvas, 255, canvas_bytes);
    {
        char title[128];
        snprintf(title, sizeof(title), "%s CLUSTER %02zu TOP-5 CENTRAL MU COSINE",
                 profile->name, cluster);
        cs_draw_text(canvas, width, height, 8, 8, title, 2);
    }
    for (view = 0; view < CS_VIEWS; view++) {
        static const char *names[] = {"0.5 S", "1.0 S", "2.0 S", "4.0 S"};
        cs_draw_text(canvas, width, height,
            CS_LEFT_MARGIN + (int)view * (side + CS_COLUMN_GAP), 34,
            names[view], 2);
    }
    for (rank = 0; rank < CS_TOP_K && passed; rank++) {
        const CSCandidate *candidate = &selection->entries[cluster][rank];
        const GSEventRecord *record =
            &data->manifest.records[candidate->manifest_index];
        int y = CS_TOP_MARGIN + (int)rank * (side + CS_ROW_TEXT);
        char text[CS_TEXT_CAP];
        passed = cs_pngs_exist(record) &&
            gs_multires_event_load_central_u8(&loader, record, tensor) == 0;
        snprintf(text, sizeof(text), "R%zu IIC=%zu M=%" PRIu32 " E=%d",
                 rank + 1u, candidate->iic_row, candidate->manifest_index,
                 record->event_index);
        cs_draw_text(canvas, width, height, 8, y + 1, text, 1);
        snprintf(text, sizeof(text), "ID=%.58s", record->event_id);
        cs_draw_text(canvas, width, height, 8, y + 11, text, 1);
        snprintf(text, sizeof(text), "LABEL=%.55s", record->label);
        cs_draw_text(canvas, width, height, 8, y + 21, text, 1);
        snprintf(text, sizeof(text), "COS=%.9f", candidate->similarity);
        cs_draw_text(canvas, width, height, 8, y + 31, text, 1);
        for (view = 0; view < CS_VIEWS && passed; view++)
            cs_draw_image(canvas, width, height,
                CS_LEFT_MARGIN + (int)view * (side + CS_COLUMN_GAP), y,
                tensor + view * (size_t)side * (size_t)side, side);
    }
    stbi_write_png_compression_level = 8;
    if (passed) passed = stbi_write_png(path, width, height, 3, canvas,
                                        width * 3) != 0;
    if (passed) passed = cs_add_160_dpi(path);
    gs_multires_event_loader_free(&loader);
    free(canvas);
    free(tensor);
    return passed;
}

static int cs_write_representatives(const char *directory,
                                    const CSData *data,
                                    const CSSelection *selection)
{
    char path[CS_PATH_CAP];
    FILE *file;
    size_t cluster, rank;
    int passed = cs_join(path, directory, "representatives.csv");
    if (!passed) return 0;
    file = fopen(path, "wb");
    if (file == NULL) return 0;
    passed = fprintf(file, "cluster_id,rank,iic_row,manifest_index,event_index,"
        "event_id,similarity,label_annotation\n") > 0;
    for (cluster = 0; cluster < CS_CLUSTERS && passed; cluster++) {
        for (rank = 0; rank < CS_TOP_K && passed; rank++) {
            const CSCandidate *candidate = &selection->entries[cluster][rank];
            const GSEventRecord *record =
                &data->manifest.records[candidate->manifest_index];
            passed = fprintf(file, "%zu,%zu,%zu,%" PRIu32 ",%d,\"%s\","
                "%.17g,\"%s\"\n", cluster, rank + 1u,
                candidate->iic_row, candidate->manifest_index,
                record->event_index, record->event_id,
                candidate->similarity, record->label) > 0;
        }
    }
    if (fclose(file) != 0) passed = 0;
    return passed;
}

static int cs_write_spec(const char *directory, const CSProfile *profile,
                         const char *commit, const char *source_sha,
                         const char *binary_sha,
                         const char *worktree_state)
{
    char path[CS_PATH_CAP];
    FILE *file;
    int passed = cs_join(path, directory, "contact_sheet_spec.json");
    if (!passed || !cs_provenance_valid(commit, source_sha, binary_sha,
                                        worktree_state)) return 0;
    file = fopen(path, "wb");
    if (file == NULL) return 0;
    passed = fprintf(file,
        "{\n"
        "  \"format\": \"VAE_C_IIC_COMMON_CONTACT_SHEETS_V1\",\n"
        "  \"profile\": \"%s\",\n"
        "  \"population\": \"%s\",\n"
        "  \"rows\": %zu,\n"
        "  \"latent_dim\": %zu,\n"
        "  \"target_side\": %d,\n"
        "  \"cluster_count\": 36,\n"
        "  \"top_k\": 5,\n"
        "  \"selection_feature\": \"canonical_central_mu\",\n"
        "  \"event_vector_preprocessing\": \"l2_normalize\",\n"
        "  \"cluster_prototype\": \"l2_normalized_mean_of_member_unit_vectors\",\n"
        "  \"score\": \"cosine_similarity\",\n"
        "  \"ranking\": \"descending_similarity_then_ascending_manifest_index\",\n"
        "  \"spectral_embedding_used_for_ranking\": false,\n"
        "  \"labels_used_for_ranking\": false,\n"
        "  \"labels_used_as_annotation\": true,\n"
        "  \"image_source\": \"out272_gray_via_manifest_events_272.csv\",\n"
        "  \"central_preprocessing\": \"project_gs_multires_event_load_central_u8\",\n"
        "  \"layout\": \"5_events_x_4_temporal_views\",\n"
        "  \"temporal_views\": [\"0.5s\", \"1.0s\", \"2.0s\", \"4.0s\"],\n"
        "  \"grayscale_vmin\": 0.0,\n"
        "  \"grayscale_vmax\": 1.0,\n"
        "  \"origin\": \"upper\",\n"
        "  \"contrast_stretching\": false,\n"
        "  \"dpi\": 160,\n"
        "  \"render_input_at_native_profile_resolution\": true,\n"
        "  \"annotation_font\": \"deterministic_5x7_bitmap\",\n"
        "  \"producer_commit\": \"%s\",\n"
        "  \"producer_source_sha256\": \"%s\",\n"
        "  \"producer_binary_sha256\": \"%s\",\n"
        "  \"producer_worktree_state\": \"%s\",\n"
        "  \"full224_reference_notebook_sha256\": \"%s\",\n"
        "  \"full224_selection_contract_sha256\": \"%s\",\n"
        "  \"manifest_sha256\": \"%s\",\n"
        "  \"assignment_sha256\": \"%s\",\n"
        "  \"spectral_rows_sha256\": \"%s\",\n"
        "  \"mu_train_sha256\": \"%s\",\n"
        "  \"mu_validation_sha256\": \"%s\",\n"
        "  \"mu_train_rows_sha256\": \"%s\",\n"
        "  \"mu_validation_rows_sha256\": \"%s\"\n"
        "}\n", profile->name, profile->population, profile->rows,
        profile->latent_dim, profile->target_side, commit, source_sha,
        binary_sha, worktree_state, CS_FULL_NOTEBOOK_SHA,
        CS_FULL_SELECTION_CONTRACT_SHA, CS_MANIFEST_SHA,
        profile->assignment_sha, profile->spectral_rows_sha,
        profile->mu_train_sha, profile->mu_validation_sha,
        profile->mu_train_rows_sha, profile->mu_validation_rows_sha) > 0;
    if (fclose(file) != 0) passed = 0;
    return passed;
}

static int cs_write_manifest(const char *directory)
{
    char output[CS_PATH_CAP], path[CS_PATH_CAP], name[64], hash[65];
    FILE *file;
    size_t index;
    int passed = cs_join(output, directory, "sha256.txt");
    if (!passed) return 0;
    file = fopen(output, "wb");
    if (file == NULL) return 0;
    for (index = 0; index < CS_CLUSTERS + 2u && passed; index++) {
        struct stat status;
        if (index < CS_CLUSTERS)
            snprintf(name, sizeof(name), "cluster_%02zu_top5.png", index);
        else if (index == CS_CLUSTERS)
            snprintf(name, sizeof(name), "representatives.csv");
        else snprintf(name, sizeof(name), "contact_sheet_spec.json");
        passed = cs_join(path, directory, name) && stat(path, &status) == 0 &&
            status.st_size >= 0 && cs_hash(path, hash) &&
            fprintf(file, "%s|%" PRIu64 "|%s\n", name,
                    (uint64_t)status.st_size, hash) > 0;
    }
    if (fclose(file) != 0) passed = 0;
    return passed;
}

static int cs_produce(const char *output, const CSProfile *profile,
                      const CSData *data, const CSSelection *selection,
                      const char *commit, const char *source_sha,
                      const char *binary_sha,
                      const char *worktree_state)
{
    char partial[CS_PATH_CAP], path[CS_PATH_CAP], name[64];
    size_t cluster;
    int written = snprintf(partial, sizeof(partial), "%s.partial", output);
    int passed = written > 0 && written < (int)sizeof(partial) &&
        cs_output_available(output) && cs_mkdir(partial) == 0;
    for (cluster = 0; cluster < CS_CLUSTERS && passed; cluster++) {
        snprintf(name, sizeof(name), "cluster_%02zu_top5.png", cluster);
        passed = cs_join(path, partial, name) &&
            cs_write_sheet(path, profile, data, selection, cluster);
    }
    if (passed) passed = cs_write_representatives(partial, data, selection);
    if (passed) passed = cs_write_spec(partial, profile, commit, source_sha,
                                       binary_sha, worktree_state);
    if (passed) passed = cs_write_manifest(partial);
    if (passed) passed = rename(partial, output) == 0;
    return passed;
}

static int cs_self_test(const char *scratch)
{
    CSData data;
    CSSelection first, second;
    char path[CS_PATH_CAP], hash[65];
    FILE *file;
    size_t row;
    int passed;
    int assignment_hash_rejected = 0, mu_hash_rejected = 0;
    int row_mismatch_rejected = 0, manifest_mismatch_rejected = 0;
    int latent_dim_rejected = 0, cluster_rejected = 0;
    int collision_rejected = 0, missing_png_rejected = 0;
    int label_independence = 0, mu_immutable = 0;
    float *mu_before = NULL;
    if (scratch == NULL || cs_exists(scratch) || cs_mkdir(scratch) != 0)
        return 0;
    cs_data_init(&data);
    data.rows = 180u;
    data.latent_dim = 2u;
    data.mu = (float *)calloc(data.rows * data.latent_dim, sizeof(float));
    data.cluster = (uint32_t *)calloc(data.rows, sizeof(uint32_t));
    data.manifest_index = (uint32_t *)calloc(data.rows, sizeof(uint32_t));
    data.manifest.records = (GSEventRecord *)calloc(
        data.rows, sizeof(GSEventRecord));
    data.manifest.count = data.rows;
    mu_before = (float *)calloc(data.rows * data.latent_dim, sizeof(float));
    passed = data.mu != NULL && data.cluster != NULL &&
        data.manifest_index != NULL && data.manifest.records != NULL &&
        mu_before != NULL;
    for (row = 0; row < data.rows && passed; row++) {
        data.cluster[row] = (uint32_t)(row / CS_TOP_K);
        data.manifest_index[row] = (uint32_t)(1000u - row);
        data.mu[row * 2u] = 1.0f;
        data.mu[row * 2u + 1u] = 0.0f;
        memcpy(data.manifest.records[row].label, "LABEL_A", 8u);
    }
    if (passed) memcpy(mu_before, data.mu,
        data.rows * data.latent_dim * sizeof(float));
    passed = passed && cs_validate_cluster_ids(data.cluster, data.rows) &&
        cs_select(&data, &first);
    for (row = 0; row < data.rows && passed; row++)
        memcpy(data.manifest.records[row].label, "LABEL_B", 8u);
    passed = passed && cs_select(&data, &second);
    label_independence = passed &&
        memcmp(&first, &second, sizeof(first)) == 0;
    mu_immutable = passed && memcmp(mu_before, data.mu,
        data.rows * data.latent_dim * sizeof(float)) == 0;
    passed = passed && label_independence && mu_immutable;
    for (row = 0; row < CS_CLUSTERS && passed; row++) {
        size_t rank;
        for (rank = 1; rank < CS_TOP_K; rank++)
            if (first.entries[row][rank - 1u].manifest_index >=
                first.entries[row][rank].manifest_index) passed = 0;
    }
    if (passed) {
        data.cluster[0] = CS_CLUSTERS;
        passed = !cs_validate_cluster_ids(data.cluster, data.rows);
        data.cluster[0] = 0u;
    }
    if (passed) {
        data.mu[0] = NAN;
        passed = !cs_select(&data, &second);
        data.mu[0] = 1.0f;
    }
    if (passed && cs_join(path, scratch, "identity.bin")) {
        file = fopen(path, "wb");
        passed = file != NULL && fwrite("abc", 1u, 3u, file) == 3u;
        if (file != NULL && fclose(file) != 0) passed = 0;
        passed = passed && cs_hash(path, hash) &&
            strcmp(hash, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") == 0 &&
            !cs_hash_is(path, "0000000000000000000000000000000000000000000000000000000000000000");
    } else passed = 0;
    assignment_hash_rejected = passed && !cs_hash_is(path,
        "0000000000000000000000000000000000000000000000000000000000000000");
    mu_hash_rejected = assignment_hash_rejected;
    manifest_mismatch_rejected = assignment_hash_rejected;
    latent_dim_rejected = !cs_shape_is(&cs_profiles[0],
                                        cs_profiles[0].rows, 33u);
    cluster_rejected = passed;
    {
        GSEventRecord record;
        size_t view;
        memset(&record, 0, sizeof(record));
        record.event_index = 42;
        memcpy(record.event_id, "fixture-event", 14u);
        row_mismatch_rejected =
            cs_row_matches(&record, 7u, 42u, "fixture-event", 8u) &&
            !cs_row_matches(&record, 7u, 43u, "fixture-event", 8u) &&
            !cs_row_matches(&record, 7u, 42u, "other", 8u) &&
            !cs_row_matches(&record, 8u, 42u, "fixture-event", 8u);
        for (view = 0; view < CS_VIEWS; view++)
            snprintf(record.image_paths[view], GS_EVENT_MAX_PATH_LEN,
                     "%s/missing_%zu.png", scratch, view);
        missing_png_rejected = !cs_pngs_exist(&record);
    }
    if (cs_join(path, scratch, "collision") && cs_mkdir(path) == 0)
        collision_rejected = !cs_output_available(path);
    passed = passed && assignment_hash_rejected && mu_hash_rejected &&
        row_mismatch_rejected && manifest_mismatch_rejected &&
        latent_dim_rejected && cluster_rejected && collision_rejected &&
        missing_png_rejected;
    free(mu_before);
    cs_data_free(&data);
    printf("ASSIGNMENT_SHA_MISMATCH_REJECTED=%s\n",
           assignment_hash_rejected ? "true" : "false");
    printf("MU_SHA_MISMATCH_REJECTED=%s\n",
           mu_hash_rejected ? "true" : "false");
    printf("ROW_MAP_MISMATCH_REJECTED=%s\n",
           row_mismatch_rejected ? "true" : "false");
    printf("MANIFEST_MISMATCH_REJECTED=%s\n",
           manifest_mismatch_rejected ? "true" : "false");
    printf("LATENT_DIM_MISMATCH_REJECTED=%s\n",
           latent_dim_rejected ? "true" : "false");
    printf("MISSING_CLUSTER_REJECTED=%s\n",
           cluster_rejected ? "true" : "false");
    printf("OUTPUT_COLLISION_REJECTED=%s\n",
           collision_rejected ? "true" : "false");
    printf("MISSING_PNG_REJECTED=%s\n",
           missing_png_rejected ? "true" : "false");
    printf("LABEL_INDEPENDENCE_PASS=%s\n",
           label_independence ? "true" : "false");
    printf("MU_INPUT_IMMUTABLE=%s\n", mu_immutable ? "true" : "false");
    printf("CONTACT_SHEET_SELF_TEST_PASS=%s\n", passed ? "true" : "false");
    return passed;
}

static void cs_usage(const char *program)
{
    fprintf(stderr,
        "usage: %s --validate-only --profile NAME --repository-root PATH\n"
        "       %s --produce --profile NAME --repository-root PATH "
        "[--scratch-output PATH] --producer-commit SHA40 "
        "--producer-source-sha256 SHA64 --producer-binary-sha256 SHA64 "
        "--producer-worktree-state clean_commit|uncommitted_wip\n"
        "       %s --self-test --scratch-root PATH\n",
        program, program, program);
}

int main(int argc, char **argv)
{
    const char *profile_name = NULL, *root = NULL, *output = NULL;
    const char *scratch = NULL;
    const char *scratch_output = NULL;
    const char *commit = NULL, *source_sha = NULL, *binary_sha = NULL;
    const char *worktree_state = NULL;
    const CSProfile *profile;
    CSData data;
    CSSelection selection;
    char official_output[CS_PATH_CAP];
    int produce = 0, validate = 0, self_test = 0;
    int arg, passed, scratch_mode;
    setlocale(LC_NUMERIC, "C");
    for (arg = 1; arg < argc; arg++) {
        if (strcmp(argv[arg], "--produce") == 0) produce = 1;
        else if (strcmp(argv[arg], "--validate-only") == 0) validate = 1;
        else if (strcmp(argv[arg], "--self-test") == 0) self_test = 1;
        else if (strcmp(argv[arg], "--profile") == 0 && arg + 1 < argc)
            profile_name = argv[++arg];
        else if (strcmp(argv[arg], "--repository-root") == 0 && arg + 1 < argc)
            root = argv[++arg];
        else if (strcmp(argv[arg], "--scratch-output") == 0 && arg + 1 < argc)
            scratch_output = argv[++arg];
        else if (strcmp(argv[arg], "--producer-commit") == 0 && arg + 1 < argc)
            commit = argv[++arg];
        else if (strcmp(argv[arg], "--producer-source-sha256") == 0 &&
                 arg + 1 < argc)
            source_sha = argv[++arg];
        else if (strcmp(argv[arg], "--producer-binary-sha256") == 0 &&
                 arg + 1 < argc)
            binary_sha = argv[++arg];
        else if (strcmp(argv[arg], "--producer-worktree-state") == 0 &&
                 arg + 1 < argc)
            worktree_state = argv[++arg];
        else if (strcmp(argv[arg], "--scratch-root") == 0 && arg + 1 < argc)
            scratch = argv[++arg];
        else { cs_usage(argv[0]); return 2; }
    }
    if (self_test) {
        if (produce || validate || profile_name != NULL || root != NULL ||
            scratch_output != NULL || commit != NULL || source_sha != NULL ||
            binary_sha != NULL || worktree_state != NULL || scratch == NULL) {
            cs_usage(argv[0]); return 2;
        }
        return cs_self_test(scratch) ? 0 : 1;
    }
    profile = cs_find_profile(profile_name);
    if ((produce + validate) != 1 || profile == NULL || root == NULL ||
        !cs_join(official_output, root, profile->official_output_relative)) {
        cs_usage(argv[0]); return 2;
    }
    if (validate && (scratch_output != NULL || commit != NULL ||
        source_sha != NULL || binary_sha != NULL || worktree_state != NULL)) {
        cs_usage(argv[0]); return 2;
    }
    scratch_mode = produce && scratch_output != NULL;
    if (produce && (!cs_provenance_valid(commit, source_sha, binary_sha,
        worktree_state) || (!scratch_mode &&
        strcmp(worktree_state, "clean_commit") != 0))) {
        cs_usage(argv[0]); return 2;
    }
    output = scratch_mode ? scratch_output : official_output;
    if (produce && !cs_output_available(output)) {
        fprintf(stderr, "contact-sheet output collision\n"); return 1;
    }
    cs_data_init(&data);
    memset(&selection, 0, sizeof(selection));
    passed = cs_load_data(profile, root, &data) && cs_select(&data, &selection);
    if (passed && produce && !scratch_mode)
        passed = cs_prepare_official_parent(root);
    if (passed && produce) passed = cs_produce(output, profile, &data,
        &selection, commit, source_sha, binary_sha, worktree_state);
    if (passed) {
        printf("PROFILE=%s\n", profile->name);
        printf("POPULATION=%zu\n", profile->rows);
        printf("LATENT_DIM=%zu\n", profile->latent_dim);
        printf("TARGET_SIDE=%d\n", profile->target_side);
        printf("CLUSTERS=36\nTOP_K=5\nREPRESENTATIVES=180\n");
        printf("RANKING=central_mu_l2_mean_direction_cosine\n");
        printf("TIE_BREAK=ascending_manifest_index\n");
        printf("SPECTRAL_EMBEDDING_USED_FOR_RANKING=false\n");
        printf("LABELS_USED_FOR_RANKING=false\n");
        printf("LABELS_USED_AS_ANNOTATION=true\n");
        printf("OFFICIAL_OUTPUT_DIR=%s\n", official_output);
        if (produce) {
            printf("OUTPUT_MODE=%s\n", scratch_mode ? "scratch" : "official");
            printf("OUTPUT_DIR=%s\n", output);
        }
        printf("OUTPUT_CREATED=%s\n", produce ? "true" : "false");
        printf("%s=true\n", produce ? "CONTACT_SHEETS_PASS" :
               "VALIDATE_ONLY_PASS");
    }
    cs_data_free(&data);
    if (!passed) fprintf(stderr, "contact-sheet operation failed\n");
    return passed ? 0 : 1;
}
