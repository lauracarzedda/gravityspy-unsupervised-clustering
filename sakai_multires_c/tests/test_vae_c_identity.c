#include "vae_c_identity.h"

#include "vae_c_checkpoint.h"
#include "vae_c_config.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define TEST_EVENT_MANIFEST_SHA256 \
    "620960e6b35da31bc4cf1a8a6da8679027d378e1d4b2ac73c5267c1c5311ae82"
#define TEST_TENSOR_MANIFEST_SHA256 \
    "94b7e2e051bd32de72dd2cb6a5fb7eda9bcaec123cbf3a07d0d901bd19779afa"
#define TEST_TRAIN_SPLIT_SHA256 \
    "8c7a513b98dbe996261bc80965500ab15ab77f498ec9440858003152e42f926f"
#define TEST_VALIDATION_SPLIT_SHA256 \
    "e63109c88df7b3ee97317f94791f26df3fcaf8b937921cfb27967d01c2546467"
#define TEST_TRAIN_SUBSET_SHA256 \
    "80c5396a70e7f2c8e3a00680582945581ae76c301a5b9ba8c62e006bb25cdfb5"
#define TEST_VALIDATION_SUBSET_SHA256 \
    "9f5fe6c50c7398588f2de7bd4a543a54a53e7f7a0b9a84bf4004a01f065ae76f"
#define TEST_PROFILE_CATALOG_SHA256 \
    "7ac5e1b5f5a76265144e9a09ce8e1e1ec976ace76f664f86b432a20a45a19ad1"

#define TEST_DATASET_IDENTITY_SHA256 \
    "389fc0326101afcae22350445114e9b42caf64b059d9d6d923f25d83ce3377b0"
#define TEST_TRAIN_IDENTITY_SHA256 \
    "b9f311f6a655afed553a7e64affb850a92be58034449550b25e4f60685e5b0a5"
#define TEST_VALIDATION_IDENTITY_SHA256 \
    "cf1a6c62ca6acf45e466d6128b52018a33fbf4052aeea46f60533017abfeb831"


enum {
    TEST_PATH_CAP = 1024,
    TEST_SOURCE_PATH_COUNT = 7,
    TEST_IDENTITY_MISMATCH_COUNT = 6
};


static const char test_dataset_preimage[] =
    "namespace=mini224_dataset_identity_v1\n"
    "kind=dataset\n"
    "dataset_version=legacy_luma_v1\n"
    "event_manifest_sha256=" TEST_EVENT_MANIFEST_SHA256 "\n"
    "tensor_manifest_sha256=" TEST_TENSOR_MANIFEST_SHA256 "\n";

static const char test_train_preimage[] =
    "namespace=mini224_train_selection_identity_v1\n"
    "kind=train_selection\n"
    "dataset_identity_sha256=" TEST_DATASET_IDENTITY_SHA256 "\n"
    "official_split_sha256=" TEST_TRAIN_SPLIT_SHA256 "\n"
    "subset_sha256=" TEST_TRAIN_SUBSET_SHA256 "\n"
    "subset_count=800\n";

static const char test_validation_preimage[] =
    "namespace=mini224_validation_selection_identity_v1\n"
    "kind=validation_selection\n"
    "dataset_identity_sha256=" TEST_DATASET_IDENTITY_SHA256 "\n"
    "official_split_sha256=" TEST_VALIDATION_SPLIT_SHA256 "\n"
    "subset_sha256=" TEST_VALIDATION_SUBSET_SHA256 "\n"
    "subset_count=200\n";


typedef struct {
    VAECModel model;
    VAECAdam optimizer;
} TestCheckpointFixture;


static int report_check(const char *name, int passed)
{
    if (passed) {
        printf("[PASS] %s\n", name);
        return 0;
    }
    printf("[FAIL] %s\n", name);
    return 1;
}


static int text_is_lower_sha256(const char *text)
{
    size_t index;

    if (text == NULL || strlen(text) != 64u) {
        return 0;
    }
    for (index = 0; index < 64u; index++) {
        if (!((text[index] >= '0' && text[index] <= '9') ||
              (text[index] >= 'a' && text[index] <= 'f'))) {
            return 0;
        }
    }
    return 1;
}


static int canonical_preimage_is_valid(const char *text, size_t size)
{
    size_t index;

    if (text == NULL || size == 0u || text[size - 1u] != '\n' ||
        strstr(text, "timestamp") != NULL || strstr(text, ":\\") != NULL ||
        strchr(text, '/') != NULL) {
        return 0;
    }
    for (index = 0; index < size; index++) {
        unsigned char byte = (unsigned char)text[index];

        if (byte == (unsigned char)'\r' || byte == 0xefu || byte == 0xbbu ||
            byte == 0xbfu || byte > 0x7fu) {
            return 0;
        }
    }
    return 1;
}


static int hash_file(const char *path, char digest[VAE_C_SHA256_HEX_SIZE])
{
    FILE *file = NULL;
    unsigned char *bytes = NULL;
    long length;
    size_t size;
    int passed = 0;

    if (path == NULL || digest == NULL ||
        (file = fopen(path, "rb")) == NULL) {
        return 0;
    }
    if (fseek(file, 0, SEEK_END) != 0 ||
        (length = ftell(file)) < 0 ||
        fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return 0;
    }
    size = (size_t)length;
    if ((long)size != length) {
        fclose(file);
        return 0;
    }
    bytes = (unsigned char *)malloc(size == 0u ? 1u : size);
    if (bytes != NULL &&
        (size == 0u || fread(bytes, 1u, size, file) == size)) {
        int close_status = fclose(file);

        file = NULL;
        if (close_status == 0 &&
            vae_c_sha256_digest_hex_lower(bytes, size, digest) ==
                VAE_C_SHA256_OK) {
            passed = 1;
        }
    }
    if (file != NULL) {
        (void)fclose(file);
    }
    free(bytes);
    return passed;
}


static int test_source_sha256(const char *const paths[TEST_SOURCE_PATH_COUNT])
{
    static const char *const expected[TEST_SOURCE_PATH_COUNT] = {
        TEST_EVENT_MANIFEST_SHA256,
        TEST_TENSOR_MANIFEST_SHA256,
        TEST_TRAIN_SPLIT_SHA256,
        TEST_VALIDATION_SPLIT_SHA256,
        TEST_TRAIN_SUBSET_SHA256,
        TEST_VALIDATION_SUBSET_SHA256,
        TEST_PROFILE_CATALOG_SHA256
    };
    char digest[VAE_C_SHA256_HEX_SIZE];
    size_t index;

    for (index = 0; index < TEST_SOURCE_PATH_COUNT; index++) {
        if (!hash_file(paths[index], digest) ||
            strcmp(digest, expected[index]) != 0) {
            return 0;
        }
    }
    return 1;
}


static int build_real_identities(
    char dataset[VAE_C_SHA256_HEX_SIZE],
    char train[VAE_C_SHA256_HEX_SIZE],
    char validation[VAE_C_SHA256_HEX_SIZE]
)
{
    return
        vae_c_identity_build_mini224_dataset_v1(
            "legacy_luma_v1",
            TEST_EVENT_MANIFEST_SHA256,
            TEST_TENSOR_MANIFEST_SHA256,
            dataset) == VAE_C_IDENTITY_OK &&
        vae_c_identity_build_mini224_train_selection_v1(
            dataset,
            TEST_TRAIN_SPLIT_SHA256,
            TEST_TRAIN_SUBSET_SHA256,
            UINT64_C(800),
            train) == VAE_C_IDENTITY_OK &&
        vae_c_identity_build_mini224_validation_selection_v1(
            dataset,
            TEST_VALIDATION_SPLIT_SHA256,
            TEST_VALIDATION_SUBSET_SHA256,
            UINT64_C(200),
            validation) == VAE_C_IDENTITY_OK;
}


static int test_known_answers_and_formatting(void)
{
    char dataset[VAE_C_SHA256_HEX_SIZE];
    char train[VAE_C_SHA256_HEX_SIZE];
    char validation[VAE_C_SHA256_HEX_SIZE];
    char direct[VAE_C_SHA256_HEX_SIZE];

    if (!build_real_identities(dataset, train, validation) ||
        strcmp(dataset, TEST_DATASET_IDENTITY_SHA256) != 0 ||
        strcmp(train, TEST_TRAIN_IDENTITY_SHA256) != 0 ||
        strcmp(validation, TEST_VALIDATION_IDENTITY_SHA256) != 0 ||
        !text_is_lower_sha256(dataset) || !text_is_lower_sha256(train) ||
        !text_is_lower_sha256(validation) ||
        strcmp(dataset, train) == 0 || strcmp(dataset, validation) == 0 ||
        strcmp(train, validation) == 0 ||
        !canonical_preimage_is_valid(
            test_dataset_preimage, sizeof(test_dataset_preimage) - 1u) ||
        !canonical_preimage_is_valid(
            test_train_preimage, sizeof(test_train_preimage) - 1u) ||
        !canonical_preimage_is_valid(
            test_validation_preimage,
            sizeof(test_validation_preimage) - 1u)) {
        return 0;
    }
    if (vae_c_sha256_digest_hex_lower(
            test_dataset_preimage,
            sizeof(test_dataset_preimage) - 1u,
            direct) != VAE_C_SHA256_OK ||
        strcmp(direct, dataset) != 0 ||
        vae_c_sha256_digest_hex_lower(
            test_train_preimage,
            sizeof(test_train_preimage) - 1u,
            direct) != VAE_C_SHA256_OK ||
        strcmp(direct, train) != 0 ||
        vae_c_sha256_digest_hex_lower(
            test_validation_preimage,
            sizeof(test_validation_preimage) - 1u,
            direct) != VAE_C_SHA256_OK ||
        strcmp(direct, validation) != 0) {
        return 0;
    }
    return 1;
}


static void mutate_sha(const char *source, char destination[65])
{
    memcpy(destination, source, 65u);
    destination[0] = destination[0] == '0' ? '1' : '0';
}


static int test_determinism_domain_and_sensitivity(void)
{
    char dataset[VAE_C_SHA256_HEX_SIZE];
    char train[VAE_C_SHA256_HEX_SIZE];
    char validation[VAE_C_SHA256_HEX_SIZE];
    char dataset_repeat[VAE_C_SHA256_HEX_SIZE];
    char train_repeat[VAE_C_SHA256_HEX_SIZE];
    char validation_repeat[VAE_C_SHA256_HEX_SIZE];
    char candidate[VAE_C_SHA256_HEX_SIZE];
    char changed[VAE_C_SHA256_HEX_SIZE];

    if (!build_real_identities(dataset, train, validation) ||
        !build_real_identities(
            dataset_repeat, train_repeat, validation_repeat)) {
        return 0;
    }
    if (strcmp(dataset_repeat, dataset) != 0 ||
        strcmp(train_repeat, train) != 0 ||
        strcmp(validation_repeat, validation) != 0) {
        return 0;
    }
    if (vae_c_identity_build_mini224_dataset_v1(
            "legacy_luma_v2", TEST_EVENT_MANIFEST_SHA256,
            TEST_TENSOR_MANIFEST_SHA256, candidate) != VAE_C_IDENTITY_OK ||
        strcmp(candidate, dataset) == 0) {
        return 0;
    }
    mutate_sha(TEST_EVENT_MANIFEST_SHA256, changed);
    if (vae_c_identity_build_mini224_dataset_v1(
            "legacy_luma_v1", changed, TEST_TENSOR_MANIFEST_SHA256,
            candidate) != VAE_C_IDENTITY_OK || strcmp(candidate, dataset) == 0) {
        return 0;
    }
    mutate_sha(TEST_TENSOR_MANIFEST_SHA256, changed);
    if (vae_c_identity_build_mini224_dataset_v1(
            "legacy_luma_v1", TEST_EVENT_MANIFEST_SHA256, changed,
            candidate) != VAE_C_IDENTITY_OK || strcmp(candidate, dataset) == 0) {
        return 0;
    }
    mutate_sha(dataset, changed);
    if (vae_c_identity_build_mini224_train_selection_v1(
            changed, TEST_TRAIN_SPLIT_SHA256, TEST_TRAIN_SUBSET_SHA256,
            UINT64_C(800), candidate) != VAE_C_IDENTITY_OK ||
        strcmp(candidate, train) == 0) {
        return 0;
    }
    mutate_sha(TEST_TRAIN_SPLIT_SHA256, changed);
    if (vae_c_identity_build_mini224_train_selection_v1(
            dataset, changed, TEST_TRAIN_SUBSET_SHA256, UINT64_C(800),
            candidate) != VAE_C_IDENTITY_OK || strcmp(candidate, train) == 0) {
        return 0;
    }
    mutate_sha(TEST_TRAIN_SUBSET_SHA256, changed);
    if (vae_c_identity_build_mini224_train_selection_v1(
            dataset, TEST_TRAIN_SPLIT_SHA256, changed, UINT64_C(800),
            candidate) != VAE_C_IDENTITY_OK || strcmp(candidate, train) == 0 ||
        vae_c_identity_build_mini224_train_selection_v1(
            dataset, TEST_TRAIN_SPLIT_SHA256, TEST_TRAIN_SUBSET_SHA256,
            UINT64_C(801), candidate) != VAE_C_IDENTITY_OK ||
        strcmp(candidate, train) == 0) {
        return 0;
    }
    if (vae_c_identity_build_mini224_train_selection_v1(
            dataset, TEST_TRAIN_SPLIT_SHA256, TEST_TRAIN_SUBSET_SHA256,
            UINT64_C(800), candidate) != VAE_C_IDENTITY_OK ||
        strcmp(candidate, train) != 0) {
        return 0;
    }
    if (vae_c_identity_build_mini224_train_selection_v1(
            dataset, TEST_TRAIN_SPLIT_SHA256, TEST_TRAIN_SUBSET_SHA256,
            UINT64_C(800), candidate) != VAE_C_IDENTITY_OK ||
        vae_c_identity_build_mini224_validation_selection_v1(
            dataset, TEST_TRAIN_SPLIT_SHA256, TEST_TRAIN_SUBSET_SHA256,
            UINT64_C(800), changed) != VAE_C_IDENTITY_OK ||
        strcmp(candidate, changed) == 0 || strcmp(validation, train) == 0) {
        return 0;
    }
    return 1;
}


#define CHECK_INVALID_COMPONENT(expression) \
    do { \
        memset(output, 'Q', sizeof(output)); \
        memcpy(before, output, sizeof(output)); \
        if ((expression) != VAE_C_IDENTITY_INVALID_COMPONENT || \
            memcmp(output, before, sizeof(output)) != 0) { \
            return 0; \
        } \
    } while (0)


static int test_invalid_inputs_are_atomic(void)
{
    char output[VAE_C_SHA256_HEX_SIZE];
    char before[VAE_C_SHA256_HEX_SIZE];
    char uppercase[VAE_C_SHA256_HEX_SIZE];
    char invalid[VAE_C_SHA256_HEX_SIZE];
    char short_sha[VAE_C_SHA256_HEX_SIZE];
    char no_terminator[VAE_C_SHA256_HEX_SIZE];
    char non_ascii_version[2] = {(char)0x80, '\0'};

    memcpy(uppercase, TEST_EVENT_MANIFEST_SHA256, sizeof(uppercase));
    uppercase[0] = 'A';
    memcpy(invalid, TEST_EVENT_MANIFEST_SHA256, sizeof(invalid));
    invalid[7] = 'g';
    memcpy(short_sha, TEST_EVENT_MANIFEST_SHA256, sizeof(short_sha));
    short_sha[63] = '\0';
    memset(no_terminator, 'a', sizeof(no_terminator));

    CHECK_INVALID_COMPONENT(vae_c_identity_build_mini224_dataset_v1(
        "", TEST_EVENT_MANIFEST_SHA256, TEST_TENSOR_MANIFEST_SHA256, output));
    CHECK_INVALID_COMPONENT(vae_c_identity_build_mini224_dataset_v1(
        "legacy\nluma", TEST_EVENT_MANIFEST_SHA256,
        TEST_TENSOR_MANIFEST_SHA256, output));
    CHECK_INVALID_COMPONENT(vae_c_identity_build_mini224_dataset_v1(
        "legacy\rluma", TEST_EVENT_MANIFEST_SHA256,
        TEST_TENSOR_MANIFEST_SHA256, output));
    CHECK_INVALID_COMPONENT(vae_c_identity_build_mini224_dataset_v1(
        "legacy/luma", TEST_EVENT_MANIFEST_SHA256,
        TEST_TENSOR_MANIFEST_SHA256, output));
    CHECK_INVALID_COMPONENT(vae_c_identity_build_mini224_dataset_v1(
        non_ascii_version, TEST_EVENT_MANIFEST_SHA256,
        TEST_TENSOR_MANIFEST_SHA256, output));
    CHECK_INVALID_COMPONENT(vae_c_identity_build_mini224_dataset_v1(
        "legacy_luma_v1", "", TEST_TENSOR_MANIFEST_SHA256, output));
    CHECK_INVALID_COMPONENT(vae_c_identity_build_mini224_dataset_v1(
        "legacy_luma_v1", uppercase, TEST_TENSOR_MANIFEST_SHA256, output));
    CHECK_INVALID_COMPONENT(vae_c_identity_build_mini224_dataset_v1(
        "legacy_luma_v1", invalid, TEST_TENSOR_MANIFEST_SHA256, output));
    CHECK_INVALID_COMPONENT(vae_c_identity_build_mini224_dataset_v1(
        "legacy_luma_v1", short_sha, TEST_TENSOR_MANIFEST_SHA256, output));
    CHECK_INVALID_COMPONENT(vae_c_identity_build_mini224_dataset_v1(
        "legacy_luma_v1", no_terminator,
        TEST_TENSOR_MANIFEST_SHA256, output));
    CHECK_INVALID_COMPONENT(vae_c_identity_build_mini224_dataset_v1(
        "legacy_luma_v1", NULL, TEST_TENSOR_MANIFEST_SHA256, output));
    CHECK_INVALID_COMPONENT(vae_c_identity_build_mini224_dataset_v1(
        "legacy_luma_v1", TEST_EVENT_MANIFEST_SHA256, NULL, output));
    CHECK_INVALID_COMPONENT(
        vae_c_identity_build_mini224_train_selection_v1(
            TEST_DATASET_IDENTITY_SHA256, TEST_TRAIN_SPLIT_SHA256,
            TEST_TRAIN_SUBSET_SHA256, UINT64_C(0), output));
    CHECK_INVALID_COMPONENT(
        vae_c_identity_build_mini224_validation_selection_v1(
            uppercase, TEST_VALIDATION_SPLIT_SHA256,
            TEST_VALIDATION_SUBSET_SHA256, UINT64_C(200), output));
    CHECK_INVALID_COMPONENT(
        vae_c_identity_build_mini224_train_selection_v1(
            TEST_DATASET_IDENTITY_SHA256, NULL,
            TEST_TRAIN_SUBSET_SHA256, UINT64_C(800), output));
    CHECK_INVALID_COMPONENT(
        vae_c_identity_build_mini224_train_selection_v1(
            TEST_DATASET_IDENTITY_SHA256, TEST_TRAIN_SPLIT_SHA256,
            NULL, UINT64_C(800), output));

    memset(output, 'Q', sizeof(output));
    memcpy(before, output, sizeof(output));
    if (vae_c_identity_build_mini224_dataset_v1(
            NULL, TEST_EVENT_MANIFEST_SHA256,
            TEST_TENSOR_MANIFEST_SHA256, output) !=
                VAE_C_IDENTITY_INVALID_COMPONENT ||
        memcmp(output, before, sizeof(output)) != 0 ||
        vae_c_identity_build_mini224_train_selection_v1(
            NULL, TEST_TRAIN_SPLIT_SHA256, TEST_TRAIN_SUBSET_SHA256,
            UINT64_C(800), output) != VAE_C_IDENTITY_INVALID_COMPONENT ||
        memcmp(output, before, sizeof(output)) != 0 ||
        vae_c_identity_build_mini224_dataset_v1(
            "legacy_luma_v1", TEST_EVENT_MANIFEST_SHA256,
            TEST_TENSOR_MANIFEST_SHA256, NULL) !=
                VAE_C_IDENTITY_INVALID_ARGUMENT ||
        vae_c_identity_build_mini224_train_selection_v1(
            TEST_DATASET_IDENTITY_SHA256, TEST_TRAIN_SPLIT_SHA256,
            TEST_TRAIN_SUBSET_SHA256, UINT64_C(800), NULL) !=
                VAE_C_IDENTITY_INVALID_ARGUMENT ||
        strcmp(vae_c_identity_status_name(VAE_C_IDENTITY_OK), "ok") != 0 ||
        strcmp(vae_c_identity_status_name(
            (VAECIdentityStatus)999), "unknown") != 0) {
        return 0;
    }
    return 1;
}

#undef CHECK_INVALID_COMPONENT


static void checkpoint_fixture_init(TestCheckpointFixture *fixture)
{
    vae_c_model_init(&fixture->model);
    vae_c_adam_init(&fixture->optimizer);
}


static void checkpoint_fixture_free(TestCheckpointFixture *fixture)
{
    vae_c_model_free(&fixture->model);
    vae_c_adam_free(&fixture->optimizer);
}


static int checkpoint_fixture_allocate(
    TestCheckpointFixture *fixture,
    const VAECProfile *profile
)
{
    checkpoint_fixture_init(fixture);
    return
        vae_c_model_allocate(&fixture->model, profile, 1u) == VAE_C_OP_OK &&
        vae_c_model_initialize_parameters_v1(
            &fixture->model, UINT64_C(123)) == VAE_C_OP_OK &&
        vae_c_adam_configure_v1(&fixture->optimizer) == VAE_C_OP_OK;
}


static void set_text(char *destination, size_t capacity, const char *source)
{
    (void)snprintf(destination, capacity, "%s", source);
}


static void fill_mini224_checkpoint_identity(
    VAECCheckpointRunIdentity *identity,
    const VAECModel *model
)
{
    vae_c_checkpoint_run_identity_init(identity);
    set_text(identity->profile_id, sizeof(identity->profile_id),
             "mini224_bridge_v1");
    set_text(identity->profile_config_sha256,
             sizeof(identity->profile_config_sha256),
             TEST_PROFILE_CATALOG_SHA256);
    set_text(identity->code_commit, sizeof(identity->code_commit),
             "0123456789abcdef0123456789abcdef01234567");
    set_text(identity->dataset_id, sizeof(identity->dataset_id),
             TEST_DATASET_IDENTITY_SHA256);
    set_text(identity->event_manifest_id,
             sizeof(identity->event_manifest_id),
             "manifest_events_272.csv");
    set_text(identity->event_manifest_sha256,
             sizeof(identity->event_manifest_sha256),
             TEST_EVENT_MANIFEST_SHA256);
    set_text(identity->tensor_manifest_id,
             sizeof(identity->tensor_manifest_id),
             "manifest_tensors_224_central.csv");
    set_text(identity->tensor_manifest_sha256,
             sizeof(identity->tensor_manifest_sha256),
             TEST_TENSOR_MANIFEST_SHA256);
    set_text(identity->train_split_id, sizeof(identity->train_split_id),
             TEST_TRAIN_IDENTITY_SHA256);
    set_text(identity->train_split_sha256,
             sizeof(identity->train_split_sha256),
             TEST_TRAIN_SPLIT_SHA256);
    set_text(identity->validation_split_id,
             sizeof(identity->validation_split_id),
             TEST_VALIDATION_IDENTITY_SHA256);
    set_text(identity->validation_split_sha256,
             sizeof(identity->validation_split_sha256),
             TEST_VALIDATION_SPLIT_SHA256);
    set_text(identity->augmentation_policy,
             sizeof(identity->augmentation_policy),
             "stateless_event_epoch_v1");
    set_text(identity->shuffle_policy, sizeof(identity->shuffle_policy),
             "fisher_yates_shuffle_v1");
    set_text(identity->latent_policy, sizeof(identity->latent_policy),
             "latent_train_per_event_epoch_v1");
    set_text(identity->validation_policy,
             sizeof(identity->validation_policy),
             "central_fixed_epsilon_v1");
    set_text(identity->compiler_id, sizeof(identity->compiler_id), "test");
    set_text(identity->compiler_version,
             sizeof(identity->compiler_version), "test");
    set_text(identity->compiler_target,
             sizeof(identity->compiler_target), "test");
    identity->initialization_seed = UINT64_C(123);
    identity->latent_seed = UINT64_C(456);
    identity->augmentation_seed = UINT64_C(789);
    identity->shuffle_seed = UINT64_C(101112);
    identity->max_batch_size = (uint64_t)model->max_batch_size;
    identity->batch_size = (uint64_t)model->max_batch_size;
    identity->optimizer_learning_rate = VAE_C_ADAM_LEARNING_RATE_V1;
    identity->optimizer_beta1 = VAE_C_ADAM_BETA1_V1;
    identity->optimizer_beta2 = VAE_C_ADAM_BETA2_V1;
    identity->optimizer_epsilon = VAE_C_ADAM_EPSILON_V1;
}


static int checkpoint_identity_equal(
    const VAECCheckpointRunIdentity *left,
    const VAECCheckpointRunIdentity *right
)
{
#define CHECK_TEXT(field) \
    (strcmp(left->field, right->field) == 0)
    return
        CHECK_TEXT(profile_id) && CHECK_TEXT(profile_config_sha256) &&
        CHECK_TEXT(code_commit) && CHECK_TEXT(dataset_id) &&
        CHECK_TEXT(event_manifest_id) && CHECK_TEXT(event_manifest_sha256) &&
        CHECK_TEXT(tensor_manifest_id) &&
        CHECK_TEXT(tensor_manifest_sha256) && CHECK_TEXT(train_split_id) &&
        CHECK_TEXT(train_split_sha256) && CHECK_TEXT(validation_split_id) &&
        CHECK_TEXT(validation_split_sha256) &&
        CHECK_TEXT(augmentation_policy) && CHECK_TEXT(shuffle_policy) &&
        CHECK_TEXT(latent_policy) && CHECK_TEXT(validation_policy) &&
        CHECK_TEXT(compiler_id) && CHECK_TEXT(compiler_version) &&
        CHECK_TEXT(compiler_target) &&
        left->initialization_seed == right->initialization_seed &&
        left->latent_seed == right->latent_seed &&
        left->augmentation_seed == right->augmentation_seed &&
        left->shuffle_seed == right->shuffle_seed &&
        left->max_batch_size == right->max_batch_size &&
        left->batch_size == right->batch_size &&
        left->optimizer_learning_rate == right->optimizer_learning_rate &&
        left->optimizer_beta1 == right->optimizer_beta1 &&
        left->optimizer_beta2 == right->optimizer_beta2 &&
        left->optimizer_epsilon == right->optimizer_epsilon;
#undef CHECK_TEXT
}


static int checkpoint_metadata_equal(
    const VAECCheckpointMetadata *left,
    const VAECCheckpointMetadata *right
)
{
    return checkpoint_identity_equal(
            &left->run_identity, &right->run_identity) &&
        left->kind == right->kind &&
        left->last_completed_epoch == right->last_completed_epoch &&
        left->best_defined == right->best_defined &&
        left->best_epoch == right->best_epoch &&
        left->best_validation_total_mean_per_sample ==
            right->best_validation_total_mean_per_sample &&
        left->no_improvement_count == right->no_improvement_count;
}


static void mutate_checkpoint_expected_identity(
    VAECCheckpointRunIdentity *identity,
    size_t mismatch
)
{
    char *field;

    switch (mismatch) {
        case 0u: field = identity->dataset_id; break;
        case 1u: field = identity->train_split_id; break;
        case 2u: field = identity->validation_split_id; break;
        case 3u: field = identity->profile_id; break;
        case 4u: field = identity->profile_config_sha256; break;
        default: field = identity->code_commit; break;
    }
    field[0] = field[0] == '0' ? '1' : '0';
}


static int make_checkpoint_path(
    char path[TEST_PATH_CAP],
    const char *directory
)
{
    int written = snprintf(
        path, TEST_PATH_CAP, "%s/%s", directory,
        "mini224_identity_round_trip.ckpt"
    );

    return written > 0 && (size_t)written < TEST_PATH_CAP;
}


static int test_checkpoint_mapping_round_trip(
    const VAECProfile *profile,
    const char *fixture_directory
)
{
    TestCheckpointFixture source;
    TestCheckpointFixture destination;
    VAECCheckpointMetadata metadata;
    VAECCheckpointMetadata loaded;
    VAECCheckpointMetadata loaded_before;
    VAECCheckpointRunIdentity mismatch_identity;
    char path[TEST_PATH_CAP] = {0};
    char temporary_path[TEST_PATH_CAP] = {0};
    float source_value;
    uint64_t generation;
    size_t mismatch;
    int passed = 0;

    checkpoint_fixture_init(&source);
    checkpoint_fixture_init(&destination);
    vae_c_checkpoint_metadata_init(&metadata);
    vae_c_checkpoint_metadata_init(&loaded);
    if (!make_checkpoint_path(path, fixture_directory)) {
        goto cleanup;
    }
    {
        int written = snprintf(
            temporary_path, sizeof(temporary_path), "%s.tmp", path
        );

        if (written <= 0 || (size_t)written >= sizeof(temporary_path)) {
            goto cleanup;
        }
    }
    (void)remove(path);
    (void)remove(temporary_path);
    if (!checkpoint_fixture_allocate(&source, profile) ||
        !checkpoint_fixture_allocate(&destination, profile)) {
        goto cleanup;
    }
    source.model.parameters[0]->values[0] += 0.25f;
    source_value = source.model.parameters[0]->values[0];
    fill_mini224_checkpoint_identity(&metadata.run_identity, &source.model);
    metadata.kind = VAE_C_CHECKPOINT_KIND_LATEST;
    metadata.last_completed_epoch = UINT64_C(0);
    if (vae_c_checkpoint_save_atomic(
            path, &source.model, &source.optimizer, &metadata) !=
                VAE_C_CHECKPOINT_OK ||
        vae_c_checkpoint_load(
            path, &metadata.run_identity, &destination.model,
            &destination.optimizer, &loaded) != VAE_C_CHECKPOINT_OK ||
        !checkpoint_metadata_equal(&metadata, &loaded) ||
        destination.model.parameters[0]->values[0] != source_value ||
        strcmp(loaded.run_identity.dataset_id,
               TEST_DATASET_IDENTITY_SHA256) != 0 ||
        strcmp(loaded.run_identity.train_split_id,
               TEST_TRAIN_IDENTITY_SHA256) != 0 ||
        strcmp(loaded.run_identity.validation_split_id,
               TEST_VALIDATION_IDENTITY_SHA256) != 0 ||
        strcmp(loaded.run_identity.event_manifest_sha256,
               TEST_EVENT_MANIFEST_SHA256) != 0 ||
        strcmp(loaded.run_identity.tensor_manifest_sha256,
               TEST_TENSOR_MANIFEST_SHA256) != 0 ||
        strcmp(loaded.run_identity.train_split_sha256,
               TEST_TRAIN_SPLIT_SHA256) != 0 ||
        strcmp(loaded.run_identity.validation_split_sha256,
               TEST_VALIDATION_SPLIT_SHA256) != 0) {
        goto cleanup;
    }
    generation = destination.model.vae_cache_generation;
    for (mismatch = 0; mismatch < TEST_IDENTITY_MISMATCH_COUNT; mismatch++) {
        mismatch_identity = metadata.run_identity;
        mutate_checkpoint_expected_identity(&mismatch_identity, mismatch);
        vae_c_checkpoint_metadata_init(&loaded);
        loaded.kind = VAE_C_CHECKPOINT_KIND_FINAL;
        loaded.last_completed_epoch = UINT64_C(77);
        loaded_before = loaded;
        if (vae_c_checkpoint_load(
                path, &mismatch_identity, &destination.model,
                &destination.optimizer, &loaded) !=
                    VAE_C_CHECKPOINT_INCOMPATIBLE ||
            memcmp(&loaded, &loaded_before, sizeof(loaded)) != 0 ||
            destination.model.vae_cache_generation != generation ||
            destination.model.parameters[0]->values[0] != source_value) {
            goto cleanup;
        }
    }
    passed = 1;

cleanup:
    (void)remove(path);
    (void)remove(temporary_path);
    checkpoint_fixture_free(&source);
    checkpoint_fixture_free(&destination);
    return passed;
}


int main(int argc, char **argv)
{
    VAECProfileCatalog catalog;
    const VAECProfile *profile;
    const char *source_paths[TEST_SOURCE_PATH_COUNT];
    int failures = 0;

    if (argc != 9) {
        fprintf(stderr,
                "usage: %s <profiles_v2> <event_manifest> <tensor_manifest> "
                "<train_split> <validation_split> <train_subset> "
                "<validation_subset> <fixture_directory>\n",
                argv[0]);
        return 2;
    }
    source_paths[0] = argv[2];
    source_paths[1] = argv[3];
    source_paths[2] = argv[4];
    source_paths[3] = argv[5];
    source_paths[4] = argv[6];
    source_paths[5] = argv[7];
    source_paths[6] = argv[1];

    vae_c_profile_catalog_init(&catalog);
    if (vae_c_profile_catalog_load_csv(
            argv[1], &catalog, NULL, 0u) != VAE_C_CONFIG_OK) {
        fprintf(stderr, "failed to load VAE V2 profile catalog\n");
        return 2;
    }
    profile = vae_c_profile_catalog_find(&catalog, "mini224_bridge_v1");
    if (profile == NULL) {
        vae_c_profile_catalog_free(&catalog);
        fprintf(stderr, "mini224_bridge_v1 profile not found\n");
        return 2;
    }

    failures += report_check(
        "canonical_source_sha256_values",
        test_source_sha256(source_paths)
    );
    failures += report_check(
        "real_known_answers_and_exact_preimage_formatting",
        test_known_answers_and_formatting()
    );
    failures += report_check(
        "determinism_domain_separation_and_component_sensitivity",
        test_determinism_domain_and_sensitivity()
    );
    failures += report_check(
        "invalid_components_preserve_output",
        test_invalid_inputs_are_atomic()
    );
    failures += report_check(
        "checkpoint_v1_mapping_round_trip_and_isolated_mismatches",
        test_checkpoint_mapping_round_trip(profile, argv[8])
    );

    vae_c_profile_catalog_free(&catalog);
    if (failures != 0) {
        fprintf(stderr, "%d VAE C IDENTITY TEST(S) FAILED\n", failures);
        return 1;
    }
    printf("MINI224_DATASET_IDENTITY_V1=%s\n",
           TEST_DATASET_IDENTITY_SHA256);
    printf("MINI224_TRAIN_SELECTION_IDENTITY_V1=%s\n",
           TEST_TRAIN_IDENTITY_SHA256);
    printf("MINI224_VALIDATION_SELECTION_IDENTITY_V1=%s\n",
           TEST_VALIDATION_IDENTITY_SHA256);
    printf("CHECKPOINT_V1_COMPOSITE_IDENTITY_COMPATIBLE=true\n");
    printf("MINI224_IDENTITY_CANONICAL=true\n");
    printf("ALL VAE C IDENTITY TESTS PASSED\n");
    return 0;
}
