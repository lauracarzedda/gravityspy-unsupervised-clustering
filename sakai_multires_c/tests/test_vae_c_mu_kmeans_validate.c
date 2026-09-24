#include "baseline_kmeans.h"


static size_t test_fit_calls = 0u;
static size_t test_assign_calls = 0u;


static int test_instrumented_kmeans_fit(
    const float *x,
    size_t rows,
    size_t cols,
    size_t k,
    uint64_t seed,
    size_t max_iterations,
    GSBaselineKMeansResult *result
);


static int test_instrumented_kmeans_assign(
    const float *x,
    size_t rows,
    size_t cols,
    const double *centroids,
    size_t k,
    size_t *assignments,
    double *squared_distances
);


#define gs_baseline_kmeans_fit test_instrumented_kmeans_fit
#define gs_baseline_kmeans_assign test_instrumented_kmeans_assign
#define VAE_C_MU_KMEANS_NO_MAIN

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "../tools/run_vae_c_mu_kmeans.c"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#undef gs_baseline_kmeans_assign
#undef gs_baseline_kmeans_fit

#include <errno.h>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif


static int test_instrumented_kmeans_fit(
    const float *x,
    size_t rows,
    size_t cols,
    size_t k,
    uint64_t seed,
    size_t max_iterations,
    GSBaselineKMeansResult *result
)
{
    test_fit_calls++;
    return gs_baseline_kmeans_fit(
        x, rows, cols, k, seed, max_iterations, result
    );
}


static int test_instrumented_kmeans_assign(
    const float *x,
    size_t rows,
    size_t cols,
    const double *centroids,
    size_t k,
    size_t *assignments,
    double *squared_distances
)
{
    test_assign_calls++;
    return gs_baseline_kmeans_assign(
        x, rows, cols, centroids, k, assignments, squared_distances
    );
}


typedef enum {
    FIXTURE_VALID = 0,
    FIXTURE_BAD_FORMAT,
    FIXTURE_NONFINITE,
    FIXTURE_BAD_ROW,
    FIXTURE_BAD_SIZE
} FixtureVariant;


typedef struct {
    KMUSourceDescriptor descriptor;
    char hashes[KMU_FILE_COUNT][VAE_C_CHECKPOINT_SHA256_CAP];
    char directory[KMU_PATH_CAP];
} Fixture;


enum {
    PRODUCTIVE_SOURCE_TRAIN_ROWS = 1000,
    PRODUCTIVE_SOURCE_VALIDATION_ROWS = 250,
    PRODUCTIVE_SELECTED_TRAIN_ROWS = 800,
    PRODUCTIVE_SELECTED_VALIDATION_ROWS = 200,
    PRODUCTIVE_COLUMNS = 2,
    PRODUCTIVE_K = 22
};


typedef struct {
    Fixture source;
    KMURowMaps rows;
    float *train;
    float *validation;
} ProductiveFixture;


static int fixture_make_directory(const char *path)
{
#ifdef _WIN32
    if (_mkdir(path) == 0 || errno == EEXIST) {
#else
    if (mkdir(path, 0777) == 0 || errno == EEXIST) {
#endif
        return 1;
    }
    return 0;
}


static int fixture_write_bytes(
    const char *directory,
    const char *name,
    const void *data,
    size_t size
)
{
    char path[KMU_PATH_CAP];
    FILE *file;
    int failed = 0;

    if (!kmu_path_join(directory, name, path)) {
        return 0;
    }
    file = fopen(path, "wb");
    if (file == NULL) {
        return 0;
    }
    if ((size > 0u && fwrite(data, 1u, size, file) != size) ||
        fflush(file) != 0) {
        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    return !failed;
}


static int fixture_hash(
    const char *directory,
    const char *name,
    char output[VAE_C_CHECKPOINT_SHA256_CAP]
)
{
    char path[KMU_PATH_CAP];

    return kmu_path_join(directory, name, path) &&
        vae_c_mu_file_sha256(path, output) == VAE_C_MU_OK;
}


static int fixture_write_row_maps(
    const char *directory,
    FixtureVariant variant
)
{
    const char *train_valid =
        "row_index,split_pos,manifest_index,event_index,event_id,split\n"
        "0,0,0,1,H1_A,train\n"
        "1,1,1,2,H1_B,train\n"
        "2,2,2,3,H1_C,train\n"
        "3,3,3,4,H1_D,train\n";
    const char *train_bad =
        "row_index,split_pos,manifest_index,event_index,event_id,split\n"
        "0,0,0,1,H1_A,train\n"
        "1,2,1,2,H1_B,train\n"
        "2,2,2,3,H1_C,train\n"
        "3,3,3,4,H1_D,train\n";
    const char *validation =
        "row_index,split_pos,manifest_index,event_index,event_id,split\n"
        "0,0,4,5,H1_E,validation\n"
        "1,1,5,6,H1_F,validation\n";
    const char *train = variant == FIXTURE_BAD_ROW ? train_bad : train_valid;

    return fixture_write_bytes(
               directory, "train_rows.csv", train, strlen(train)
           ) &&
        fixture_write_bytes(
               directory, "validation_rows.csv", validation,
               strlen(validation)
        );
}


static int fixture_write_matrices(
    const char *directory,
    FixtureVariant variant
)
{
    float train[] = {
        0.0f, 1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f, 7.0f
    };
    const float validation[] = {8.0f, 9.0f, 10.0f, 11.0f};
    size_t train_values = sizeof(train) / sizeof(train[0]);

    if (variant == FIXTURE_NONFINITE) {
        train[3] = NAN;
    } else if (variant == FIXTURE_BAD_SIZE) {
        train_values--;
    }
    return fixture_write_bytes(
               directory, "train_mu_f32.bin", train,
               train_values * sizeof(float)
           ) &&
        fixture_write_bytes(
               directory, "validation_mu_f32.bin", validation,
               sizeof(validation)
        );
}


static int fixture_write_spec(
    Fixture *fixture,
    FixtureVariant variant
)
{
    char spec[4096];
    int written = snprintf(
        spec,
        sizeof(spec),
        "format=%s\n"
        "representation=posterior_mu\n"
        "view=central\n"
        "profile_id=fixture_profile\n"
        "latent_dim=2\n"
        "dtype=float32_ieee754\n"
        "endian=little\n"
        "layout=row_major_[N,latent_dim]\n"
        "latent_scaling=none\n"
        "latent_normalization=none\n"
        "pca=false\n"
        "labels_used=false\n"
        "label_dependent_selections=false\n"
        "label_values_exported=false\n"
        "row_map_schema=row_index,split_pos,manifest_index,event_index,event_id,split\n"
        "train_rows=4\n"
        "validation_rows=2\n"
        "train_cols=2\n"
        "validation_cols=2\n"
        "train_matrix_sha256=%s\n"
        "validation_matrix_sha256=%s\n"
        "train_rows_sha256=%s\n"
        "validation_rows_sha256=%s\n",
        variant == FIXTURE_BAD_FORMAT
            ? "VAE_C_MU_ARTIFACT_BROKEN" : "VAE_C_MU_ARTIFACT_V1",
        fixture->hashes[KMU_TRAIN_MATRIX_FILE],
        fixture->hashes[KMU_VALIDATION_MATRIX_FILE],
        fixture->hashes[KMU_TRAIN_ROWS_FILE],
        fixture->hashes[KMU_VALIDATION_ROWS_FILE]
    );

    return written >= 0 && (size_t)written < sizeof(spec) &&
        fixture_write_bytes(
            fixture->directory, "embedding_spec.txt", spec, (size_t)written
        );
}


static int fixture_write_checksum(Fixture *fixture)
{
    char manifest[2048];
    int written = snprintf(
        manifest,
        sizeof(manifest),
        "%s  embedding_spec.txt\n"
        "%s  train_mu_f32.bin\n"
        "%s  train_rows.csv\n"
        "%s  validation_mu_f32.bin\n"
        "%s  validation_rows.csv\n",
        fixture->hashes[KMU_METADATA_FILE],
        fixture->hashes[KMU_TRAIN_MATRIX_FILE],
        fixture->hashes[KMU_TRAIN_ROWS_FILE],
        fixture->hashes[KMU_VALIDATION_MATRIX_FILE],
        fixture->hashes[KMU_VALIDATION_ROWS_FILE]
    );

    return written >= 0 && (size_t)written < sizeof(manifest) &&
        fixture_write_bytes(
            fixture->directory, "sha256.txt", manifest, (size_t)written
        );
}


static int fixture_build(
    const char *parent,
    const char *name,
    FixtureVariant variant,
    Fixture *fixture
)
{
    size_t index;
    static const char *const names[KMU_FILE_COUNT] = {
        "embedding_spec.txt", "sha256.txt", "train_mu_f32.bin",
        "validation_mu_f32.bin", "train_rows.csv", "validation_rows.csv"
    };

    memset(fixture, 0, sizeof(*fixture));
    if (!kmu_path_join(parent, name, fixture->directory) ||
        !fixture_make_directory(parent) ||
        !fixture_make_directory(fixture->directory)) {
        return 0;
    }
    fixture->descriptor.token = "fixture";
    fixture->descriptor.profile_id = "fixture_profile";
    fixture->descriptor.relative_directory = NULL;
    fixture->descriptor.schema = KMU_SCHEMA_MINI_V1;
    fixture->descriptor.train_rows = 4u;
    fixture->descriptor.validation_rows = 2u;
    fixture->descriptor.columns = 2u;
    for (index = 0u; index < KMU_FILE_COUNT; index++) {
        fixture->descriptor.names[index] = names[index];
        fixture->descriptor.hashes[index] = fixture->hashes[index];
    }
    if (!fixture_write_matrices(fixture->directory, variant) ||
        !fixture_write_row_maps(fixture->directory, variant)) {
        return 0;
    }
    for (index = KMU_TRAIN_MATRIX_FILE;
         index <= KMU_VALIDATION_ROWS_FILE; index++) {
        if (!fixture_hash(
                fixture->directory, names[index], fixture->hashes[index]
            )) {
            return 0;
        }
    }
    if (!fixture_write_spec(fixture, variant) ||
        !fixture_hash(
            fixture->directory, names[KMU_METADATA_FILE],
            fixture->hashes[KMU_METADATA_FILE]
        ) ||
        !fixture_write_checksum(fixture) ||
        !fixture_hash(
            fixture->directory, names[KMU_CHECKSUM_FILE],
            fixture->hashes[KMU_CHECKSUM_FILE]
        )) {
        return 0;
    }
    return 1;
}


static void productive_fixture_free(ProductiveFixture *fixture)
{
    if (fixture == NULL) {
        return;
    }
    kmu_row_maps_free(&fixture->rows);
    free(fixture->validation);
    free(fixture->train);
    memset(fixture, 0, sizeof(*fixture));
}


static int productive_write_row_map(
    const char *directory,
    const char *name,
    const char *split,
    size_t rows,
    size_t manifest_offset
)
{
    char path[KMU_PATH_CAP];
    FILE *file;
    size_t row;
    int failed = 0;

    if (!kmu_path_join(directory, name, path)) {
        return 0;
    }
    file = fopen(path, "wb");
    if (file == NULL || fputs(
            "row_index,split_pos,manifest_index,event_index,event_id,split\n",
            file
        ) == EOF) {
        if (file != NULL) {
            (void)fclose(file);
        }
        return 0;
    }
    for (row = 0u; row < rows; row++) {
        const size_t manifest_index = manifest_offset + row;
        if (fprintf(
                file,
                "%zu,%zu,%zu,%zu,H1_SYN_%c_%04zu,%s\n",
                row,
                row,
                manifest_index,
                manifest_index + 1u,
                strcmp(split, "train") == 0 ? 'T' : 'V',
                row,
                split
            ) < 0) {
            failed = 1;
            break;
        }
    }
    if (fflush(file) != 0 || ferror(file)) {
        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    return !failed;
}


static int productive_write_spec(ProductiveFixture *fixture)
{
    char spec[4096];
    int written = snprintf(
        spec,
        sizeof(spec),
        "format=VAE_C_MU_ARTIFACT_V1\n"
        "representation=posterior_mu\n"
        "view=central\n"
        "profile_id=synthetic_productive_profile\n"
        "latent_dim=%d\n"
        "dtype=float32_ieee754\n"
        "endian=little\n"
        "layout=row_major_[N,latent_dim]\n"
        "latent_scaling=none\n"
        "latent_normalization=none\n"
        "pca=false\n"
        "labels_used=false\n"
        "label_dependent_selections=false\n"
        "label_values_exported=false\n"
        "row_map_schema=row_index,split_pos,manifest_index,event_index,event_id,split\n"
        "train_rows=%d\n"
        "validation_rows=%d\n"
        "train_cols=%d\n"
        "validation_cols=%d\n"
        "train_matrix_sha256=%s\n"
        "validation_matrix_sha256=%s\n"
        "train_rows_sha256=%s\n"
        "validation_rows_sha256=%s\n",
        PRODUCTIVE_COLUMNS,
        PRODUCTIVE_SOURCE_TRAIN_ROWS,
        PRODUCTIVE_SOURCE_VALIDATION_ROWS,
        PRODUCTIVE_COLUMNS,
        PRODUCTIVE_COLUMNS,
        fixture->source.hashes[KMU_TRAIN_MATRIX_FILE],
        fixture->source.hashes[KMU_VALIDATION_MATRIX_FILE],
        fixture->source.hashes[KMU_TRAIN_ROWS_FILE],
        fixture->source.hashes[KMU_VALIDATION_ROWS_FILE]
    );

    return written >= 0 && (size_t)written < sizeof(spec) &&
        fixture_write_bytes(
            fixture->source.directory,
            "embedding_spec.txt",
            spec,
            (size_t)written
        );
}


static int productive_fixture_build(
    const char *parent,
    ProductiveFixture *fixture
)
{
    static const char *const names[KMU_FILE_COUNT] = {
        "embedding_spec.txt", "sha256.txt", "train_mu_f32.bin",
        "validation_mu_f32.bin", "train_rows.csv", "validation_rows.csv"
    };
    const size_t train_values =
        PRODUCTIVE_SOURCE_TRAIN_ROWS * PRODUCTIVE_COLUMNS;
    const size_t validation_values =
        PRODUCTIVE_SOURCE_VALIDATION_ROWS * PRODUCTIVE_COLUMNS;
    size_t row;
    size_t index;

    memset(fixture, 0, sizeof(*fixture));
    if (!kmu_path_join(
            parent, "productive_source", fixture->source.directory
        ) ||
        !fixture_make_directory(parent) ||
        !fixture_make_directory(fixture->source.directory)) {
        return 0;
    }
    fixture->source.descriptor.token = "synthetic_productive";
    fixture->source.descriptor.profile_id = "synthetic_productive_profile";
    fixture->source.descriptor.relative_directory =
        "external_build_fixture/productive_source";
    fixture->source.descriptor.schema = KMU_SCHEMA_MINI_V1;
    fixture->source.descriptor.train_rows = PRODUCTIVE_SOURCE_TRAIN_ROWS;
    fixture->source.descriptor.validation_rows =
        PRODUCTIVE_SOURCE_VALIDATION_ROWS;
    fixture->source.descriptor.columns = PRODUCTIVE_COLUMNS;
    for (index = 0u; index < KMU_FILE_COUNT; index++) {
        fixture->source.descriptor.names[index] = names[index];
        fixture->source.descriptor.hashes[index] =
            fixture->source.hashes[index];
    }

    fixture->train = (float *)malloc(train_values * sizeof(float));
    fixture->validation =
        (float *)malloc(validation_values * sizeof(float));
    if (fixture->train == NULL || fixture->validation == NULL) {
        productive_fixture_free(fixture);
        return 0;
    }
    for (row = 0u; row < PRODUCTIVE_SOURCE_TRAIN_ROWS; row++) {
        if (row < PRODUCTIVE_SELECTED_TRAIN_ROWS) {
            const size_t cluster = row % PRODUCTIVE_K;
            const size_t repetition = row / PRODUCTIVE_K;
            fixture->train[row * PRODUCTIVE_COLUMNS] =
                (float)(cluster * 16u) + (float)(repetition % 13u) * 0.01f;
            fixture->train[row * PRODUCTIVE_COLUMNS + 1u] =
                (float)(cluster * 7u) - (float)(repetition % 11u) * 0.02f;
        } else {
            const size_t tail = row - PRODUCTIVE_SELECTED_TRAIN_ROWS;
            fixture->train[row * PRODUCTIVE_COLUMNS] =
                10000.0f + (float)(tail * 3u);
            fixture->train[row * PRODUCTIVE_COLUMNS + 1u] =
                -10000.0f - (float)(tail * 5u);
        }
    }
    for (row = 0u; row < PRODUCTIVE_SOURCE_VALIDATION_ROWS; row++) {
        const size_t cluster = row % PRODUCTIVE_K;
        fixture->validation[row * PRODUCTIVE_COLUMNS] =
            (float)(cluster * 16u) + 0.25f;
        fixture->validation[row * PRODUCTIVE_COLUMNS + 1u] =
            (float)(cluster * 7u) - 0.50f;
    }

    if (!fixture_write_bytes(
            fixture->source.directory,
            "train_mu_f32.bin",
            fixture->train,
            train_values * sizeof(float)
        ) ||
        !fixture_write_bytes(
            fixture->source.directory,
            "validation_mu_f32.bin",
            fixture->validation,
            validation_values * sizeof(float)
        ) ||
        !productive_write_row_map(
            fixture->source.directory,
            "train_rows.csv",
            "train",
            PRODUCTIVE_SOURCE_TRAIN_ROWS,
            0u
        ) ||
        !productive_write_row_map(
            fixture->source.directory,
            "validation_rows.csv",
            "validation",
            PRODUCTIVE_SOURCE_VALIDATION_ROWS,
            PRODUCTIVE_SOURCE_TRAIN_ROWS
        )) {
        productive_fixture_free(fixture);
        return 0;
    }
    for (index = KMU_TRAIN_MATRIX_FILE;
         index <= KMU_VALIDATION_ROWS_FILE; index++) {
        if (!fixture_hash(
                fixture->source.directory,
                names[index],
                fixture->source.hashes[index]
            )) {
            productive_fixture_free(fixture);
            return 0;
        }
    }
    if (!productive_write_spec(fixture) ||
        !fixture_hash(
            fixture->source.directory,
            names[KMU_METADATA_FILE],
            fixture->source.hashes[KMU_METADATA_FILE]
        ) ||
        !fixture_write_checksum(&fixture->source) ||
        !fixture_hash(
            fixture->source.directory,
            names[KMU_CHECKSUM_FILE],
            fixture->source.hashes[KMU_CHECKSUM_FILE]
        ) ||
        !kmu_validate_artifact(
            fixture->source.directory,
            &fixture->source.descriptor,
            &fixture->rows
        )) {
        productive_fixture_free(fixture);
        return 0;
    }
    return 1;
}


static int test_artifact_passes(const Fixture *fixture)
{
    KMURowMaps maps;
    int result;

    memset(&maps, 0, sizeof(maps));
    result = kmu_validate_artifact(
        fixture->directory, &fixture->descriptor, &maps
    );
    kmu_row_maps_free(&maps);
    return result;
}


static int test_hash_mismatch(const char *parent)
{
    Fixture fixture;
    char path[KMU_PATH_CAP];
    float replacement = 42.0f;
    FILE *file;
    int failed = 0;

    if (!fixture_build(parent, "bad_hash", FIXTURE_VALID, &fixture) ||
        !kmu_path_join(fixture.directory, "train_mu_f32.bin", path)) {
        return 0;
    }
    file = fopen(path, "r+b");
    if (file == NULL) {
        return 0;
    }
    if (fwrite(&replacement, sizeof(replacement), 1u, file) != 1u ||
        fflush(file) != 0) {
        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    if (failed) {
        return 0;
    }
    return !test_artifact_passes(&fixture);
}


static int test_variant_rejected(
    const char *parent,
    const char *name,
    FixtureVariant variant
)
{
    Fixture fixture;

    return fixture_build(parent, name, variant, &fixture) &&
        !test_artifact_passes(&fixture);
}


static int test_selection_comparator(void)
{
    KMURowMaps full;
    KMURowMaps matched;
    KMURowMaps source;
    size_t index;
    int passed;

    memset(&full, 0, sizeof(full));
    memset(&matched, 0, sizeof(matched));
    memset(&source, 0, sizeof(source));
    full.train_count = KMU_MATCHED_TRAIN_ROWS;
    full.validation_count = KMU_MATCHED_VALIDATION_ROWS;
    matched.train_count = KMU_MATCHED_TRAIN_ROWS;
    matched.validation_count = KMU_MATCHED_VALIDATION_ROWS;
    source.train_count = KMU_MATCHED_TRAIN_ROWS;
    source.validation_count = KMU_MATCHED_VALIDATION_ROWS;
    full.train = (KMURowIdentity *)calloc(
        full.train_count, sizeof(*full.train)
    );
    full.validation = (KMURowIdentity *)calloc(
        full.validation_count, sizeof(*full.validation)
    );
    matched.train = (KMURowIdentity *)calloc(
        matched.train_count, sizeof(*matched.train)
    );
    matched.validation = (KMURowIdentity *)calloc(
        matched.validation_count, sizeof(*matched.validation)
    );
    source.train = (KMURowIdentity *)calloc(
        source.train_count, sizeof(*source.train)
    );
    source.validation = (KMURowIdentity *)calloc(
        source.validation_count, sizeof(*source.validation)
    );
    if (full.train == NULL || full.validation == NULL ||
        matched.train == NULL || matched.validation == NULL ||
        source.train == NULL || source.validation == NULL) {
        kmu_row_maps_free(&source);
        kmu_row_maps_free(&matched);
        kmu_row_maps_free(&full);
        return 0;
    }
    for (index = 0u; index < full.train_count; index++) {
        full.train[index].split_pos = (uint64_t)index;
        full.train[index].manifest_index = (uint64_t)index;
        full.train[index].event_index = (uint64_t)index + UINT64_C(1);
        snprintf(full.train[index].event_id, KMU_EVENT_ID_CAP,
                 "H1_T%zu", index);
        strcpy(full.train[index].split, "train");
        matched.train[index] = full.train[index];
        source.train[index] = full.train[index];
    }
    for (index = 0u; index < full.validation_count; index++) {
        full.validation[index].split_pos = (uint64_t)index;
        full.validation[index].manifest_index =
            (uint64_t)KMU_MATCHED_TRAIN_ROWS + (uint64_t)index;
        full.validation[index].event_index =
            full.validation[index].manifest_index + UINT64_C(1);
        snprintf(full.validation[index].event_id, KMU_EVENT_ID_CAP,
                 "H1_V%zu", index);
        strcpy(full.validation[index].split, "validation");
        matched.validation[index] = full.validation[index];
        source.validation[index] = full.validation[index];
    }
    passed = kmu_population_selection_matches(
        &source, &full, &matched, KMU_SCOPE_MATCHED_SUBSET1000
    );
    strcpy(source.train[17].event_id, "H1_WRONG");
    passed = passed && !kmu_population_selection_matches(
        &source, &full, &matched, KMU_SCOPE_MATCHED_SUBSET1000
    );
    kmu_row_maps_free(&source);
    kmu_row_maps_free(&matched);
    kmu_row_maps_free(&full);
    return passed;
}


static int test_strict_numeric_parser(void)
{
    uint64_t value = 0u;
    size_t size = 0u;

    return kmu_parse_u64("123", &value) && value == UINT64_C(123) &&
        kmu_parse_size("300", &size) && size == 300u &&
        !kmu_parse_u64("", &value) &&
        !kmu_parse_u64("+123", &value) &&
        !kmu_parse_u64("-123", &value) &&
        !kmu_parse_u64(" 123", &value) &&
        !kmu_parse_u64("123 ", &value) &&
        !kmu_parse_u64("12x", &value) &&
        !kmu_parse_u64("18446744073709551616", &value);
}


static int productive_paths(
    const char *parent,
    const char *name,
    char partial[KMU_PATH_CAP],
    char final[KMU_PATH_CAP]
)
{
    int written;

    if (!kmu_path_join(parent, name, final)) {
        return 0;
    }
    written = snprintf(partial, KMU_PATH_CAP, "%s.partial", final);
    return written >= 0 && (size_t)written < KMU_PATH_CAP;
}


static int productive_read_exact(
    const char *directory,
    const char *name,
    void *data,
    size_t bytes
)
{
    char path[KMU_PATH_CAP];
    FILE *file;
    int passed;

    if (data == NULL || bytes == 0u ||
        !kmu_path_join(directory, name, path) ||
        (file = fopen(path, "rb")) == NULL) {
        return 0;
    }
    passed = fread(data, 1u, bytes, file) == bytes &&
        fgetc(file) == EOF && !ferror(file);
    if (fclose(file) != 0) {
        passed = 0;
    }
    return passed;
}


static int productive_output_hashes(
    const char *directory,
    char hashes[KMU_OUTPUT_FILE_COUNT][VAE_C_CHECKPOINT_SHA256_CAP]
)
{
    size_t index;

    if (!kmu_output_inventory_valid(directory, 1)) {
        return 0;
    }
    for (index = 0u; index < KMU_OUTPUT_FILE_COUNT; index++) {
        char path[KMU_PATH_CAP];
        if (!kmu_path_join(directory, kmu_output_names[index], path) ||
            vae_c_mu_file_sha256(path, hashes[index]) != VAE_C_MU_OK) {
            return 0;
        }
    }
    return 1;
}


static int productive_hashes_equal(
    char left[KMU_OUTPUT_FILE_COUNT][VAE_C_CHECKPOINT_SHA256_CAP],
    char right[KMU_OUTPUT_FILE_COUNT][VAE_C_CHECKPOINT_SHA256_CAP]
)
{
    size_t index;

    for (index = 0u; index < KMU_OUTPUT_FILE_COUNT; index++) {
        if (strcmp(left[index], right[index]) != 0) {
            return 0;
        }
    }
    return 1;
}


static int productive_request_init(
    const char *repository_root,
    const char *fixture_parent,
    ProductiveFixture *fixture,
    size_t max_iterations,
    KMURequest *request,
    KMUCodeHashes *code_hashes
)
{
    memset(request, 0, sizeof(*request));
    memset(code_hashes, 0, sizeof(*code_hashes));
    request->scope = KMU_SCOPE_MATCHED_SUBSET1000;
    request->source = &fixture->source.descriptor;
    request->k = PRODUCTIVE_K;
    request->seed = KMU_SEED;
    request->max_iterations = max_iterations;
    request->repository_root = repository_root;
    request->output_root = fixture_parent;
    request->code_commit = KMU_RUNNER_BASE_COMMIT;
    return kmu_compute_code_hashes(repository_root, code_hashes);
}


static int productive_verify_output(
    const char *final,
    const ProductiveFixture *fixture,
    const KMURequest *request,
    const KMUCodeHashes *code_hashes
)
{
    const size_t centroid_count = PRODUCTIVE_K * PRODUCTIVE_COLUMNS;
    GSBaselineKMeansResult expected;
    GSBaselineKMeansResult full;
    double *centroids = NULL;
    double *train_distances = NULL;
    double *validation_distances = NULL;
    double *expected_train_distances = NULL;
    double *expected_validation_distances = NULL;
    uint32_t *train_assignments = NULL;
    uint32_t *validation_assignments = NULL;
    uint64_t *cluster_sizes = NULL;
    uint64_t *initial_indices = NULL;
    size_t *reassigned_train = NULL;
    size_t *reassigned_validation = NULL;
    size_t row;
    int full_is_distinct = 0;
    int passed = 0;
    char metadata_path[KMU_PATH_CAP];

    memset(&expected, 0, sizeof(expected));
    memset(&full, 0, sizeof(full));
    centroids = (double *)malloc(centroid_count * sizeof(double));
    train_distances = (double *)malloc(
        PRODUCTIVE_SELECTED_TRAIN_ROWS * sizeof(double)
    );
    validation_distances = (double *)malloc(
        PRODUCTIVE_SELECTED_VALIDATION_ROWS * sizeof(double)
    );
    expected_train_distances = (double *)malloc(
        PRODUCTIVE_SELECTED_TRAIN_ROWS * sizeof(double)
    );
    expected_validation_distances = (double *)malloc(
        PRODUCTIVE_SELECTED_VALIDATION_ROWS * sizeof(double)
    );
    train_assignments = (uint32_t *)malloc(
        PRODUCTIVE_SELECTED_TRAIN_ROWS * sizeof(uint32_t)
    );
    validation_assignments = (uint32_t *)malloc(
        PRODUCTIVE_SELECTED_VALIDATION_ROWS * sizeof(uint32_t)
    );
    cluster_sizes = (uint64_t *)malloc(PRODUCTIVE_K * sizeof(uint64_t));
    initial_indices = (uint64_t *)malloc(PRODUCTIVE_K * sizeof(uint64_t));
    reassigned_train = (size_t *)malloc(
        PRODUCTIVE_SELECTED_TRAIN_ROWS * sizeof(size_t)
    );
    reassigned_validation = (size_t *)malloc(
        PRODUCTIVE_SELECTED_VALIDATION_ROWS * sizeof(size_t)
    );
    if (centroids == NULL || train_distances == NULL ||
        validation_distances == NULL || expected_train_distances == NULL ||
        expected_validation_distances == NULL || train_assignments == NULL ||
        validation_assignments == NULL || cluster_sizes == NULL ||
        initial_indices == NULL || reassigned_train == NULL ||
        reassigned_validation == NULL) {
        goto cleanup;
    }

    if (gs_baseline_kmeans_fit(
            fixture->train,
            PRODUCTIVE_SELECTED_TRAIN_ROWS,
            PRODUCTIVE_COLUMNS,
            PRODUCTIVE_K,
            KMU_SEED,
            KMU_MAX_ITERATIONS,
            &expected
        ) != GS_BASELINE_KMEANS_SUCCESS || expected.converged != 1 ||
        gs_baseline_kmeans_fit(
            fixture->train,
            PRODUCTIVE_SOURCE_TRAIN_ROWS,
            PRODUCTIVE_COLUMNS,
            PRODUCTIVE_K,
            KMU_SEED,
            KMU_MAX_ITERATIONS,
            &full
        ) != GS_BASELINE_KMEANS_SUCCESS || full.converged != 1 ||
        !productive_read_exact(
            final, "centroids_f64.bin", centroids,
            centroid_count * sizeof(double)
        ) ||
        !productive_read_exact(
            final, "train_assignments_u32.bin", train_assignments,
            PRODUCTIVE_SELECTED_TRAIN_ROWS * sizeof(uint32_t)
        ) ||
        !productive_read_exact(
            final, "validation_assignments_u32.bin", validation_assignments,
            PRODUCTIVE_SELECTED_VALIDATION_ROWS * sizeof(uint32_t)
        ) ||
        !productive_read_exact(
            final, "train_squared_distances_f64.bin", train_distances,
            PRODUCTIVE_SELECTED_TRAIN_ROWS * sizeof(double)
        ) ||
        !productive_read_exact(
            final, "validation_squared_distances_f64.bin",
            validation_distances,
            PRODUCTIVE_SELECTED_VALIDATION_ROWS * sizeof(double)
        ) ||
        !productive_read_exact(
            final, "cluster_sizes_train_u64.bin", cluster_sizes,
            PRODUCTIVE_K * sizeof(uint64_t)
        ) ||
        !productive_read_exact(
            final, "initial_centroid_indices_u64.bin", initial_indices,
            PRODUCTIVE_K * sizeof(uint64_t)
        ) ||
        memcmp(
            centroids, expected.centroids, centroid_count * sizeof(double)
        ) != 0 ||
        gs_baseline_kmeans_assign(
            fixture->train,
            PRODUCTIVE_SELECTED_TRAIN_ROWS,
            PRODUCTIVE_COLUMNS,
            centroids,
            PRODUCTIVE_K,
            reassigned_train,
            expected_train_distances
        ) != GS_BASELINE_KMEANS_SUCCESS ||
        gs_baseline_kmeans_assign(
            fixture->validation,
            PRODUCTIVE_SELECTED_VALIDATION_ROWS,
            PRODUCTIVE_COLUMNS,
            centroids,
            PRODUCTIVE_K,
            reassigned_validation,
            expected_validation_distances
        ) != GS_BASELINE_KMEANS_SUCCESS) {
        goto cleanup;
    }
    for (row = 0u; row < PRODUCTIVE_SELECTED_TRAIN_ROWS; row++) {
        if (train_assignments[row] != (uint32_t)expected.assignments[row] ||
            train_assignments[row] != (uint32_t)reassigned_train[row] ||
            memcmp(
                &train_distances[row],
                &expected_train_distances[row],
                sizeof(double)
            ) != 0) {
            goto cleanup;
        }
    }
    for (row = 0u; row < PRODUCTIVE_SELECTED_VALIDATION_ROWS; row++) {
        if (validation_assignments[row] !=
                (uint32_t)reassigned_validation[row] ||
            memcmp(
                &validation_distances[row],
                &expected_validation_distances[row],
                sizeof(double)
            ) != 0) {
            goto cleanup;
        }
    }
    for (row = 0u; row < PRODUCTIVE_K; row++) {
        if (cluster_sizes[row] != (uint64_t)expected.cluster_sizes[row] ||
            initial_indices[row] !=
                (uint64_t)expected.initial_centroid_indices[row] ||
            initial_indices[row] >= PRODUCTIVE_SELECTED_TRAIN_ROWS) {
            goto cleanup;
        }
    }
    if (memcmp(
            expected.centroids,
            full.centroids,
            centroid_count * sizeof(double)
        ) != 0) {
        full_is_distinct = 1;
    }
    if (!full_is_distinct ||
        !kmu_output_inventory_valid(final, 1) ||
        !kmu_verify_sha256_manifest(final, request, code_hashes) ||
        !kmu_verify_metadata(
            final,
            request,
            PRODUCTIVE_SELECTED_TRAIN_ROWS,
            PRODUCTIVE_SELECTED_VALIDATION_ROWS,
            code_hashes
        ) ||
        !kmu_verify_assignment_csv(
            final,
            "train_assignments.csv",
            fixture->rows.train,
            train_assignments,
            train_distances,
            PRODUCTIVE_SELECTED_TRAIN_ROWS,
            PRODUCTIVE_K
        ) ||
        !kmu_verify_assignment_csv(
            final,
            "validation_assignments.csv",
            fixture->rows.validation,
            validation_assignments,
            validation_distances,
            PRODUCTIVE_SELECTED_VALIDATION_ROWS,
            PRODUCTIVE_K
        ) ||
        !kmu_path_join(final, "metadata.txt", metadata_path) ||
        !kmu_file_has_exact_line(metadata_path, "matched_refit=true") ||
        !kmu_file_has_exact_line(metadata_path, "assignment_slicing=false") ||
        !kmu_file_has_exact_line(metadata_path, "labels_used=false") ||
        !kmu_file_has_exact_line(
            metadata_path,
            "validation_policy=assign_to_frozen_train_centroids"
        )) {
        goto cleanup;
    }
    passed = 1;

cleanup:
    gs_baseline_kmeans_result_free(&full);
    gs_baseline_kmeans_result_free(&expected);
    free(reassigned_validation);
    free(reassigned_train);
    free(initial_indices);
    free(cluster_sizes);
    free(validation_assignments);
    free(train_assignments);
    free(expected_validation_distances);
    free(expected_train_distances);
    free(validation_distances);
    free(train_distances);
    free(centroids);
    return passed;
}


static int productive_publish_case(
    const char *repository_root,
    const char *fixture_parent,
    const char *output_name
)
{
    ProductiveFixture fixture;
    KMURequest request;
    KMUCodeHashes code_hashes;
    KMURunReport report;
    char partial[KMU_PATH_CAP];
    char final[KMU_PATH_CAP];
    float *train_copy = NULL;
    float *validation_copy = NULL;
    const size_t train_bytes =
        PRODUCTIVE_SOURCE_TRAIN_ROWS * PRODUCTIVE_COLUMNS * sizeof(float);
    const size_t validation_bytes = PRODUCTIVE_SOURCE_VALIDATION_ROWS *
        PRODUCTIVE_COLUMNS * sizeof(float);
    int passed = 0;

    memset(&fixture, 0, sizeof(fixture));
    memset(&report, 0, sizeof(report));
    if (!productive_fixture_build(fixture_parent, &fixture) ||
        !productive_request_init(
            repository_root,
            fixture_parent,
            &fixture,
            KMU_MAX_ITERATIONS,
            &request,
            &code_hashes
        ) ||
        !productive_paths(fixture_parent, output_name, partial, final)) {
        goto cleanup;
    }
    train_copy = (float *)malloc(train_bytes);
    validation_copy = (float *)malloc(validation_bytes);
    if (train_copy == NULL || validation_copy == NULL) {
        goto cleanup;
    }
    memcpy(train_copy, fixture.train, train_bytes);
    memcpy(validation_copy, fixture.validation, validation_bytes);
    test_fit_calls = 0u;
    test_assign_calls = 0u;
    if (!kmu_fit_loaded_and_publish(
            &request,
            fixture.source.directory,
            &fixture.rows,
            fixture.train,
            fixture.validation,
            PRODUCTIVE_SELECTED_TRAIN_ROWS,
            PRODUCTIVE_SELECTED_VALIDATION_ROWS,
            &code_hashes,
            partial,
            final,
            &report
        ) ||
        report.fit_calls != 1u || report.assign_calls != 2u ||
        report.converged != 1 || report.output_created != 1 ||
        test_fit_calls != 1u || test_assign_calls != 2u ||
        kmu_path_state(final) != 2 || kmu_path_state(partial) != 0 ||
        memcmp(train_copy, fixture.train, train_bytes) != 0 ||
        memcmp(validation_copy, fixture.validation, validation_bytes) != 0 ||
        !productive_verify_output(
            final, &fixture, &request, &code_hashes
        )) {
        goto cleanup;
    }
    printf("SYNTHETIC_PUBLISH_CONVERGED=true\n");
    printf("SYNTHETIC_FIT_CALLS=1\n");
    printf("SYNTHETIC_ASSIGN_CALLS=2\n");
    printf("SYNTHETIC_MATCHED_REFIT_PREFIX_800=true\n");
    printf("SYNTHETIC_VALIDATION_FROZEN_CENTROIDS=true\n");
    printf("SYNTHETIC_INPUT_UNCHANGED=true\n");
    printf("SYNTHETIC_OUTPUT_INVENTORY=13\n");
    printf("SYNTHETIC_OUTPUT_CREATED=true\n");
    printf("SYNTHETIC_PUBLISH_PASS=true\n");
    passed = 1;

cleanup:
    free(validation_copy);
    free(train_copy);
    productive_fixture_free(&fixture);
    return passed;
}


static int productive_overwrite_case(
    const char *repository_root,
    const char *fixture_parent,
    const char *output_name
)
{
    ProductiveFixture fixture;
    KMURequest request;
    KMUCodeHashes code_hashes;
    KMURunReport report;
    char before[KMU_OUTPUT_FILE_COUNT][VAE_C_CHECKPOINT_SHA256_CAP];
    char after[KMU_OUTPUT_FILE_COUNT][VAE_C_CHECKPOINT_SHA256_CAP];
    char partial[KMU_PATH_CAP];
    char final[KMU_PATH_CAP];
    int call_result;
    int passed = 0;

    memset(&fixture, 0, sizeof(fixture));
    memset(&report, 0, sizeof(report));
    if (!productive_fixture_build(fixture_parent, &fixture) ||
        !productive_request_init(
            repository_root,
            fixture_parent,
            &fixture,
            KMU_MAX_ITERATIONS,
            &request,
            &code_hashes
        ) ||
        !productive_paths(fixture_parent, output_name, partial, final) ||
        !productive_output_hashes(final, before)) {
        goto cleanup;
    }
    test_fit_calls = 0u;
    test_assign_calls = 0u;
    call_result = kmu_fit_loaded_and_publish(
        &request,
        fixture.source.directory,
        &fixture.rows,
        fixture.train,
        fixture.validation,
        PRODUCTIVE_SELECTED_TRAIN_ROWS,
        PRODUCTIVE_SELECTED_VALIDATION_ROWS,
        &code_hashes,
        partial,
        final,
        &report
    );
    if (call_result || report.fit_calls != 0u || report.assign_calls != 0u ||
        report.output_created != 0 || test_fit_calls != 0u ||
        test_assign_calls != 0u || kmu_path_state(partial) != 0 ||
        !productive_output_hashes(final, after) ||
        !productive_hashes_equal(before, after)) {
        goto cleanup;
    }
    printf("SYNTHETIC_OVERWRITE_REFUSED=true\n");
    printf("SYNTHETIC_OVERWRITE_FIT_CALLS=0\n");
    printf("SYNTHETIC_OVERWRITE_OUTPUT_UNCHANGED=true\n");
    printf("SYNTHETIC_OVERWRITE_PASS=true\n");
    passed = 1;

cleanup:
    productive_fixture_free(&fixture);
    return passed;
}


static int productive_nonpublish_case(
    const char *repository_root,
    const char *fixture_parent,
    const char *output_name,
    int nonfinite
)
{
    ProductiveFixture fixture;
    KMURequest request;
    KMUCodeHashes code_hashes;
    KMURunReport report;
    char partial[KMU_PATH_CAP];
    char final[KMU_PATH_CAP];
    int call_result;
    int passed = 0;

    memset(&fixture, 0, sizeof(fixture));
    memset(&report, 0, sizeof(report));
    if (!productive_fixture_build(fixture_parent, &fixture) ||
        !productive_request_init(
            repository_root,
            fixture_parent,
            &fixture,
            nonfinite ? KMU_MAX_ITERATIONS : 1u,
            &request,
            &code_hashes
        ) ||
        !productive_paths(fixture_parent, output_name, partial, final)) {
        goto cleanup;
    }
    if (nonfinite) {
        fixture.train[0] = NAN;
    }
    test_fit_calls = 0u;
    test_assign_calls = 0u;
    call_result = kmu_fit_loaded_and_publish(
        &request,
        fixture.source.directory,
        &fixture.rows,
        fixture.train,
        fixture.validation,
        PRODUCTIVE_SELECTED_TRAIN_ROWS,
        PRODUCTIVE_SELECTED_VALIDATION_ROWS,
        &code_hashes,
        partial,
        final,
        &report
    );
    if (call_result || report.fit_calls != 1u || report.assign_calls != 0u ||
        report.converged != 0 || report.output_created != 0 ||
        test_fit_calls != 1u || test_assign_calls != 0u ||
        kmu_path_state(final) != 0 || kmu_path_state(partial) != 0) {
        goto cleanup;
    }
    if (nonfinite) {
        printf("SYNTHETIC_INVALID_INPUT_REJECTED=true\n");
        printf("SYNTHETIC_INVALID_INPUT_OUTPUT_CREATED=false\n");
        printf("SYNTHETIC_INVALID_INPUT_PASS=true\n");
    } else {
        printf("SYNTHETIC_NONCONVERGED=1\n");
        printf("SYNTHETIC_NONCONVERGED_OUTPUT_CREATED=false\n");
        printf("SYNTHETIC_NONCONVERGED_PARTIAL_PRESENT=false\n");
        printf("SYNTHETIC_NONCONVERGENCE_PASS=true\n");
    }
    passed = 1;

cleanup:
    productive_fixture_free(&fixture);
    return passed;
}


static int productive_overlap_case(
    const char *repository_root,
    const char *fixture_parent,
    const char *output_name
)
{
    ProductiveFixture fixture;
    KMURequest request;
    KMUCodeHashes code_hashes;
    KMURunReport report;
    char partial[KMU_PATH_CAP];
    char final[KMU_PATH_CAP];
    int call_result;
    int passed = 0;

    memset(&fixture, 0, sizeof(fixture));
    memset(&report, 0, sizeof(report));
    if (!productive_fixture_build(fixture_parent, &fixture) ||
        !productive_request_init(
            repository_root,
            fixture_parent,
            &fixture,
            KMU_MAX_ITERATIONS,
            &request,
            &code_hashes
        ) ||
        !productive_paths(
            fixture.source.directory, output_name, partial, final
        )) {
        goto cleanup;
    }
    request.output_root = fixture.source.directory;
    test_fit_calls = 0u;
    test_assign_calls = 0u;
    call_result = kmu_fit_loaded_and_publish(
        &request,
        fixture.source.directory,
        &fixture.rows,
        fixture.train,
        fixture.validation,
        PRODUCTIVE_SELECTED_TRAIN_ROWS,
        PRODUCTIVE_SELECTED_VALIDATION_ROWS,
        &code_hashes,
        partial,
        final,
        &report
    );
    if (call_result || report.fit_calls != 0u || report.assign_calls != 0u ||
        report.output_created != 0 || test_fit_calls != 0u ||
        test_assign_calls != 0u || kmu_path_state(final) != 0 ||
        kmu_path_state(partial) != 0 ||
        !kmu_directory_has_exact_inventory(
            fixture.source.directory, &fixture.source.descriptor
        ) ||
        !test_artifact_passes(&fixture.source)) {
        goto cleanup;
    }
    printf("SYNTHETIC_SOURCE_OUTPUT_OVERLAP_REFUSED=true\n");
    printf("SYNTHETIC_SOURCE_OUTPUT_OVERLAP_FIT_CALLS=0\n");
    printf("SYNTHETIC_SOURCE_INVENTORY_UNCHANGED=true\n");
    printf("SYNTHETIC_SOURCE_OUTPUT_OVERLAP_PASS=true\n");
    passed = 1;

cleanup:
    productive_fixture_free(&fixture);
    return passed;
}


static int productive_repository_root_case(
    const char *repository_root,
    const char *fixture_parent,
    const char *output_name
)
{
    ProductiveFixture fixture;
    KMURequest request;
    KMUCodeHashes code_hashes;
    KMURunReport report;
    char partial[KMU_PATH_CAP];
    char final[KMU_PATH_CAP];
    char resolved_partial[KMU_PATH_CAP];
    char resolved_final[KMU_PATH_CAP];
    int resolve_result;
    int call_result;
    int passed = 0;

    memset(&fixture, 0, sizeof(fixture));
    memset(&report, 0, sizeof(report));
    memset(resolved_partial, 0, sizeof(resolved_partial));
    memset(resolved_final, 0, sizeof(resolved_final));
    if (!productive_fixture_build(fixture_parent, &fixture) ||
        !productive_request_init(
            repository_root,
            fixture_parent,
            &fixture,
            KMU_MAX_ITERATIONS,
            &request,
            &code_hashes
        ) ||
        !productive_paths(fixture_parent, output_name, partial, final)) {
        goto cleanup;
    }
    request.output_root = repository_root;
    test_fit_calls = 0u;
    test_assign_calls = 0u;
    resolve_result = kmu_resolve_output_paths(
        &request, resolved_partial, resolved_final
    );
    call_result = kmu_fit_loaded_and_publish(
        &request,
        fixture.source.directory,
        &fixture.rows,
        fixture.train,
        fixture.validation,
        PRODUCTIVE_SELECTED_TRAIN_ROWS,
        PRODUCTIVE_SELECTED_VALIDATION_ROWS,
        &code_hashes,
        partial,
        final,
        &report
    );
    if (resolve_result || call_result || report.fit_calls != 0u ||
        report.assign_calls != 0u || report.output_created != 0 ||
        test_fit_calls != 0u || test_assign_calls != 0u ||
        kmu_path_state(final) != 0 || kmu_path_state(partial) != 0 ||
        !kmu_directory_has_exact_inventory(
            fixture.source.directory, &fixture.source.descriptor
        ) ||
        !test_artifact_passes(&fixture.source)) {
        goto cleanup;
    }
    printf("SYNTHETIC_REPOSITORY_OUTPUT_ROOT_REFUSED=true\n");
    printf("SYNTHETIC_REPOSITORY_OUTPUT_ROOT_FIT_CALLS=0\n");
    printf("SYNTHETIC_REPOSITORY_OUTPUT_ROOT_SOURCE_UNCHANGED=true\n");
    printf("SYNTHETIC_REPOSITORY_OUTPUT_ROOT_PASS=true\n");
    passed = 1;

cleanup:
    productive_fixture_free(&fixture);
    return passed;
}


int main(int argc, char **argv)
{
    Fixture valid;
    KMURowMaps canonical_full;
    KMURowMaps canonical_matched;
    int passed;

    if (argc == 5 && strcmp(argv[1], "--productive-publish") == 0) {
        return productive_publish_case(argv[2], argv[3], argv[4]) ? 0 : 1;
    }
    if (argc == 5 && strcmp(argv[1], "--productive-overwrite") == 0) {
        return productive_overwrite_case(argv[2], argv[3], argv[4]) ? 0 : 1;
    }
    if (argc == 5 &&
        strcmp(argv[1], "--productive-nonconverged") == 0) {
        return productive_nonpublish_case(
            argv[2], argv[3], argv[4], 0
        ) ? 0 : 1;
    }
    if (argc == 5 && strcmp(argv[1], "--productive-invalid") == 0) {
        return productive_nonpublish_case(
            argv[2], argv[3], argv[4], 1
        ) ? 0 : 1;
    }
    if (argc == 5 && strcmp(argv[1], "--productive-overlap") == 0) {
        return productive_overlap_case(argv[2], argv[3], argv[4]) ? 0 : 1;
    }
    if (argc == 5 &&
        strcmp(argv[1], "--productive-repository-root") == 0) {
        return productive_repository_root_case(
            argv[2], argv[3], argv[4]
        ) ? 0 : 1;
    }
    if (argc != 3) {
        fprintf(stderr,
                "Usage: %s <repository-root> "
                "<external-build-fixture-directory>\n"
                "   or: %s --productive-{publish|overwrite|"
                "nonconverged|invalid|overlap|repository-root} "
                "<repository-root> "
                "<external-build-fixture-directory> <output-name>\n",
                argv[0],
                argv[0]);
        return 2;
    }
    memset(&canonical_full, 0, sizeof(canonical_full));
    memset(&canonical_matched, 0, sizeof(canonical_matched));
    if (!kmu_load_canonical_populations(
            argv[1], &canonical_full, &canonical_matched
        ) ||
        !kmu_population_selection_matches(
            &canonical_matched, &canonical_full, &canonical_matched,
            KMU_SCOPE_MATCHED_SUBSET1000
        )) {
        fprintf(stderr, "canonical split/subset identity failed\n");
        kmu_row_maps_free(&canonical_matched);
        kmu_row_maps_free(&canonical_full);
        return 1;
    }
    kmu_row_maps_free(&canonical_matched);
    kmu_row_maps_free(&canonical_full);
    printf("[PASS] canonical_split_subset_sha_and_identity\n");

    passed = fixture_build(argv[2], "valid", FIXTURE_VALID, &valid) &&
        test_artifact_passes(&valid);
    if (!passed) {
        fprintf(stderr, "valid synthetic artifact was rejected\n");
        return 1;
    }
    printf("[PASS] synthetic_valid_artifact\n");

    if (!test_hash_mismatch(argv[2])) {
        fprintf(stderr, "SHA mismatch was not rejected\n");
        return 1;
    }
    printf("[PASS] synthetic_sha_mismatch_rejected\n");

    if (!test_variant_rejected(
            argv[2], "bad_format", FIXTURE_BAD_FORMAT
        )) {
        fprintf(stderr, "format mismatch was not rejected\n");
        return 1;
    }
    printf("[PASS] synthetic_format_mismatch_rejected\n");

    if (!test_variant_rejected(
            argv[2], "nonfinite", FIXTURE_NONFINITE
        )) {
        fprintf(stderr, "non-finite matrix was not rejected\n");
        return 1;
    }
    printf("[PASS] synthetic_nonfinite_rejected\n");

    if (!test_variant_rejected(argv[2], "bad_row", FIXTURE_BAD_ROW)) {
        fprintf(stderr, "row-map mismatch was not rejected\n");
        return 1;
    }
    printf("[PASS] synthetic_row_map_mismatch_rejected\n");

    if (!test_variant_rejected(argv[2], "bad_size", FIXTURE_BAD_SIZE)) {
        fprintf(stderr, "matrix size mismatch was not rejected\n");
        return 1;
    }
    printf("[PASS] synthetic_invalid_matrix_size_rejected\n");

    if (!test_selection_comparator()) {
        fprintf(stderr, "matched prefix selection comparator failed\n");
        return 1;
    }
    printf("[PASS] synthetic_label_free_selection_contract\n");

    if (!test_strict_numeric_parser()) {
        fprintf(stderr, "strict numeric parser contract failed\n");
        return 1;
    }
    printf("[PASS] strict_numeric_parser_contract\n");
    printf("FIT_CALLS=0\n");
    printf("ASSIGN_CALLS=0\n");
    printf("OUTPUTS_OUTSIDE_BUILD_FIXTURE=0\n");
    printf("VAE_C_MU_KMEANS_VALIDATE_TEST_PASS=true\n");
    return 0;
}
