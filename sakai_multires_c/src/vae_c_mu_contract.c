#include "vae_c_mu_contract.h"

#include "vae_c_adam.h"
#include "vae_c_identity.h"
#include "vae_c_model.h"
#include "vae_c_sha256.h"
#include "vae_event_dataset.h"
#include "vae_saved_split.h"
#include "vae_tensor_manifest.h"
#include "vae_tensor_spec.h"

#include <ctype.h>
#include <float.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


enum {
    MU_EVENT_COUNT = 8583,
    MU_FULL_TRAIN_COUNT = 6866,
    MU_FULL_VALIDATION_COUNT = 1717,
    MU_CHANNELS = 4,
    MU_LINE_CAP = 2048
};


#define MU_EVENT_MANIFEST_REL "manifest_events_272.csv"
#define MU_EVENT_MANIFEST_SHA \
    "620960e6b35da31bc4cf1a8a6da8679027d378e1d4b2ac73c5267c1c5311ae82"
#define MU_TRAIN_SPLIT_REL "split_train_indices_seed123.csv"
#define MU_TRAIN_SPLIT_SHA \
    "8c7a513b98dbe996261bc80965500ab15ab77f498ec9440858003152e42f926f"
#define MU_VALIDATION_SPLIT_REL "split_val_indices_seed123.csv"
#define MU_VALIDATION_SPLIT_SHA \
    "e63109c88df7b3ee97317f94791f26df3fcaf8b937921cfb27967d01c2546467"


typedef struct {
    const char *run_token;
    const char *profile_id;
    const char *run_id;
    const char *run_directory_rel;
    const char *run_config_format;
    const char *run_config_sha256;
    const char *checkpoint_sha256;
    const char *code_commit;

    const char *catalog_rel;
    const char *catalog_sha256;
    const char *tensor_manifest_rel;
    const char *tensor_manifest_id;
    const char *tensor_manifest_sha256;

    const char *profile_role;
    const char *profile_status;
    size_t input_side;
    size_t conv_depth;
    size_t channels[VAE_C_MAX_CONV_DEPTH];
    size_t latent_dim;
    size_t bottleneck_side;
    uint64_t trainable_parameters;
    uint64_t forward_mac_per_sample;

    uint64_t train_count;
    uint64_t validation_count;
    uint64_t batch_size;
    uint64_t max_epochs;
    uint64_t patience;
    uint64_t checkpoint_epoch;
    uint64_t checkpoint_optimizer_step;

    const char *augmentation_identity_policy;
    const char *latent_identity_policy;
    const char *validation_identity_policy;
    bool uppercase_identity_hashes;

    bool uses_subset;
    const char *dataset_version;
    const char *dataset_identity;
    const char *train_subset_rel;
    const char *train_subset_sha256;
    const char *validation_subset_rel;
    const char *validation_subset_sha256;
    const char *train_selection_identity;
    const char *validation_selection_identity;
    const char *population_identity;
    const char *tensor_extra_column;
    const char *tensor_extra_value;
} MuCanonicalDescriptor;


static const MuCanonicalDescriptor mu_descriptors[] = {
    {
        "mini75",
        "mini75_experiment_v1",
        "mini75_experiment_v1_full_init123_latent456_aug789_shuffle101112_bs64",
        "sakai_multires_c/results/vae_c/training/"
            "mini75_experiment_v1_full_init123_latent456_aug789_"
            "shuffle101112_bs64",
        "vae_c_mini75_training_v1",
        "7a874870d54e38f8517436ad1a29481e46816846b161643b3f425aacb3b3881c",
        "7926f3a2557d7b02a9b429627c622e5c11559c1d5a273d2e8129a68e5c7ab121",
        "6fab046a02cf9c85dc5c2dfa2c4f9346372f5fbb",
        "sakai_multires_c/configs/vae_c_profiles_v1.csv",
        "e11d4bd722474bfbc50432b843679ef4f07e94a875d2e5ace5b8cd85eff6860f",
        "manifest_tensors_75_central.csv",
        "manifest_tensors_75_central.csv",
        "e3bc690924ab1fdd5997b67e01e2c326b2f34425c6b1ebc4295b16ca0ec5ab23",
        "optional_full_split_experiment",
        "optional_after_smoke",
        75u, 3u, {16u, 32u, 64u, 0u}, 32u, 5u,
        UINT64_C(106116), UINT64_C(18356544),
        UINT64_C(6866), UINT64_C(1717), UINT64_C(64),
        UINT64_C(50), UINT64_C(10), UINT64_C(47), UINT64_C(5184),
        "stateless_event_epoch_v1+symmetric_figure",
        "train_event_epoch_v1+validation_event_v1",
        "central_stochastic_fixed_epsilon_v1",
        true,
        false,
        NULL, "gravity_spy_o1_official_8583",
        NULL, NULL, NULL, NULL, NULL, NULL,
        "gravity_spy_o1_full_split_seed123_v1",
        "origin", "derived_from_224_central"
    },
    {
        "mini112",
        "mini112_primary_v1",
        "mini112_primary_v1_full_init123_latent456_aug789_shuffle101112_bs64",
        "sakai_multires_c/results/vae_c/training/"
            "mini112_primary_v1_full_init123_latent456_aug789_"
            "shuffle101112_bs64",
        "vae_c_mini112_training_v1",
        "c4df52196cb66beed1034a0f12aaa6fd781c06df778507c882f2b01f0cb2e661",
        "8857090fb1477fafed6b841bb880a06d61acf38899c7ad2e5a9ef60672989a8b",
        "5fd8ed6babc99324c1da07ac81706c94497025c3",
        "sakai_multires_c/configs/vae_c_profiles_v1.csv",
        "e11d4bd722474bfbc50432b843679ef4f07e94a875d2e5ace5b8cd85eff6860f",
        "manifest_tensors_112_central.csv",
        "manifest_tensors_112_central.csv",
        "9eb7f1c7e0491f6df0dc346f1d405b8b5929a3cd26e8ce92feb3c299875d301a",
        "primary_c_experiment",
        "primary",
        112u, 3u, {16u, 32u, 64u, 0u}, 64u, 7u,
        UINT64_C(262852), UINT64_C(39948288),
        UINT64_C(6866), UINT64_C(1717), UINT64_C(64),
        UINT64_C(100), UINT64_C(10), UINT64_C(97), UINT64_C(10584),
        "stateless_event_epoch_v1+symmetric_figure+side112",
        "train_event_epoch_v1+validation_event_v1+latent64",
        "central_stochastic_fixed_epsilon_v1+max100+patience10",
        true,
        false,
        NULL, "gravity_spy_o1_official_8583",
        NULL, NULL, NULL, NULL, NULL, NULL,
        "gravity_spy_o1_full_split_seed123_v1",
        "origin", "derived_from_224_central"
    },
    {
        "mini224",
        "mini224_bridge_v1",
        "mini224_bridge_v1_subset1000_candidate_init123_latent456_"
            "aug789_shuffle101112_bs16_max100_pat20",
        "sakai_multires_c/results/vae_c/training/"
            "mini224_bridge_v1_subset1000_candidate_init123_latent456_"
            "aug789_shuffle101112_bs16_max100_pat20",
        "vae_c_mini224_training_v1",
        "55ab0d59bbcef555cddf72368e6255c40c57ffa54dd2fea0da4aa87b463c542a",
        "15a1ecd6dd3c6d11aa0a2ac20ea0b6176d08a18c6f582e7349ed6784ab8528f3",
        "99f95aacca5e2a146585ccac9f7e2e2bc1cc8256",
        "sakai_multires_c/configs/vae_c_profiles_v2.csv",
        "7ac5e1b5f5a76265144e9a09ce8e1e1ec976ace76f664f86b432a20a45a19ad1",
        "manifest_tensors_224_central.csv",
        "manifest_tensors_224_central.csv",
        "94b7e2e051bd32de72dd2cb6a5fb7eda9bcaec123cbf3a07d0d901bd19779afa",
        "project_bridge_candidate",
        "candidate_pending_224_smoke",
        224u, 4u, {16u, 32u, 64u, 128u}, 64u, 7u,
        UINT64_C(625860), UINT64_C(220790784),
        UINT64_C(800), UINT64_C(200), UINT64_C(16),
        UINT64_C(100), UINT64_C(20), UINT64_C(99), UINT64_C(5000),
        "stateless_event_epoch_v1+symmetric_figure+side224",
        "train_event_epoch_v1+validation_event_v1+latent64",
        NULL,
        false,
        true,
        "legacy_luma_v1",
        "389fc0326101afcae22350445114e9b42caf64b059d9d6d923f25d83ce3377b0",
        "sakai_multires_c/configs/subset224_train_800_seed123.csv",
        "80c5396a70e7f2c8e3a00680582945581ae76c301a5b9ba8c62e006bb25cdfb5",
        "sakai_multires_c/configs/subset224_val_200_seed123.csv",
        "9f5fe6c50c7398588f2de7bd4a543a54a53e7f7a0b9a84bf4004a01f065ae76f",
        "b9f311f6a655afed553a7e64affb850a92be58034449550b25e4f60685e5b0a5",
        "cf1a6c62ca6acf45e466d6128b52018a33fbf4052aeea46f60533017abfeb831",
        "mini224_subset_seed123_prefix_800_200_v1",
        "crop_left", "24"
    }
};


static void mu_error(
    char *message,
    size_t capacity,
    const char *format,
    ...
)
{
    va_list arguments;

    if (message == NULL || capacity == 0u) {
        return;
    }
    va_start(arguments, format);
    (void)vsnprintf(message, capacity, format, arguments);
    va_end(arguments);
    message[capacity - 1u] = '\0';
}


static int mu_copy_text(char *target, size_t capacity, const char *source)
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


static int mu_join_path(
    const char *root,
    const char *relative,
    char path[VAE_C_MU_PATH_CAP]
)
{
    size_t root_length;
    int result;
    const char *separator;

    if (root == NULL || root[0] == '\0' || relative == NULL ||
        relative[0] == '\0' || path == NULL) {
        return 0;
    }
    root_length = strlen(root);
    separator = root_length != 0u &&
        (root[root_length - 1u] == '/' || root[root_length - 1u] == '\\')
        ? "" : "/";
    result = snprintf(
        path, VAE_C_MU_PATH_CAP, "%s%s%s", root, separator, relative
    );
    return result >= 0 && (size_t)result < VAE_C_MU_PATH_CAP;
}


static int mu_hash_equal(const char *left, const char *right)
{
    size_t index;

    if (left == NULL || right == NULL || strlen(left) != 64u ||
        strlen(right) != 64u) {
        return 0;
    }
    for (index = 0; index < 64u; index++) {
        if (tolower((unsigned char)left[index]) !=
            tolower((unsigned char)right[index])) {
            return 0;
        }
    }
    return 1;
}


static int mu_hash_with_case(
    const char *lower_hash,
    bool uppercase,
    char output[VAE_C_CHECKPOINT_SHA256_CAP]
)
{
    size_t index;

    if (lower_hash == NULL || strlen(lower_hash) != 64u || output == NULL) {
        return 0;
    }
    for (index = 0; index < 64u; index++) {
        unsigned char byte = (unsigned char)lower_hash[index];

        if (!((byte >= (unsigned char)'0' && byte <= (unsigned char)'9') ||
              (byte >= (unsigned char)'a' && byte <= (unsigned char)'f'))) {
            return 0;
        }
        output[index] = uppercase ? (char)toupper(byte) : (char)byte;
    }
    output[64] = '\0';
    return 1;
}


VAECMuStatus vae_c_mu_file_sha256(
    const char *path,
    char sha256[VAE_C_CHECKPOINT_SHA256_CAP]
)
{
    FILE *file = NULL;
    unsigned char *bytes = NULL;
    long length;
    size_t size;
    char candidate[VAE_C_CHECKPOINT_SHA256_CAP];
    VAECMuStatus status = VAE_C_MU_IO_ERROR;

    if (path == NULL || path[0] == '\0' || sha256 == NULL) {
        return VAE_C_MU_INVALID_ARGUMENT;
    }
    file = fopen(path, "rb");
    if (file == NULL || fseek(file, 0, SEEK_END) != 0 ||
        (length = ftell(file)) < 0 || fseek(file, 0, SEEK_SET) != 0) {
        if (file != NULL) {
            (void)fclose(file);
        }
        return VAE_C_MU_IO_ERROR;
    }
    size = (size_t)length;
    if ((long)size != length) {
        (void)fclose(file);
        return VAE_C_MU_OVERFLOW;
    }
    bytes = (unsigned char *)malloc(size == 0u ? 1u : size);
    if (bytes == NULL) {
        (void)fclose(file);
        return VAE_C_MU_IO_ERROR;
    }
    {
        int read_ok = size == 0u || fread(bytes, 1u, size, file) == size;
        int close_status = fclose(file);

        file = NULL;
        if (read_ok && close_status == 0 &&
            vae_c_sha256_digest_hex_lower(bytes, size, candidate) ==
                VAE_C_SHA256_OK) {
            memcpy(sha256, candidate, sizeof(candidate));
            status = VAE_C_MU_OK;
        }
    }
    if (file != NULL) {
        (void)fclose(file);
    }
    free(bytes);
    return status;
}


static VAECMuStatus mu_verify_hash(
    const char *path,
    const char *expected,
    const char *description,
    char *error_message,
    size_t error_capacity
)
{
    char actual[VAE_C_CHECKPOINT_SHA256_CAP];
    VAECMuStatus status = vae_c_mu_file_sha256(path, actual);

    if (status != VAE_C_MU_OK) {
        mu_error(error_message, error_capacity,
                 "cannot hash %s: %s", description, path);
        return status;
    }
    if (!mu_hash_equal(actual, expected)) {
        mu_error(error_message, error_capacity,
                 "SHA-256 mismatch for %s", description);
        return VAE_C_MU_HASH_MISMATCH;
    }
    return VAE_C_MU_OK;
}


static const MuCanonicalDescriptor *mu_find_descriptor(const char *name)
{
    size_t index;

    if (name == NULL) {
        return NULL;
    }
    for (index = 0;
         index < sizeof(mu_descriptors) / sizeof(mu_descriptors[0]);
         index++) {
        if (strcmp(name, mu_descriptors[index].run_token) == 0 ||
            strcmp(name, mu_descriptors[index].profile_id) == 0) {
            return &mu_descriptors[index];
        }
    }
    return NULL;
}


static int mu_run_config_value(
    const char *path,
    const char *key,
    char *value,
    size_t value_capacity
)
{
    FILE *file;
    char line[MU_LINE_CAP];
    size_t key_length;
    int found = 0;

    if (path == NULL || key == NULL || key[0] == '\0' || value == NULL ||
        value_capacity == 0u || (file = fopen(path, "rb")) == NULL) {
        return 0;
    }
    key_length = strlen(key);
    while (fgets(line, sizeof(line), file) != NULL) {
        size_t length = strlen(line);
        const char *source;
        size_t source_length;

        if (length != 0u && line[length - 1u] != '\n' && !feof(file)) {
            found = 0;
            break;
        }
        while (length != 0u &&
               (line[length - 1u] == '\n' || line[length - 1u] == '\r')) {
            line[--length] = '\0';
        }
        if (length <= key_length || line[key_length] != '=' ||
            memcmp(line, key, key_length) != 0) {
            continue;
        }
        if (found) {
            found = 0;
            break;
        }
        source = line + key_length + 1u;
        source_length = strlen(source);
        if (source_length >= value_capacity) {
            found = 0;
            break;
        }
        memcpy(value, source, source_length + 1u);
        found = 1;
    }
    if (ferror(file) || fclose(file) != 0) {
        return 0;
    }
    return found;
}


static int mu_run_config_expect(
    const char *path,
    const char *key,
    const char *expected
)
{
    char actual[MU_LINE_CAP];

    return mu_run_config_value(path, key, actual, sizeof(actual)) &&
        strcmp(actual, expected) == 0;
}


static int mu_run_config_expect_u64(
    const char *path,
    const char *key,
    uint64_t expected
)
{
    char actual[MU_LINE_CAP];
    char *end = NULL;
    unsigned long long value;

    if (!mu_run_config_value(path, key, actual, sizeof(actual)) ||
        actual[0] == '-' || actual[0] == '+') {
        return 0;
    }
    value = strtoull(actual, &end, 10);
    return end != NULL && end[0] == '\0' &&
        (uint64_t)value == expected;
}


static int mu_run_config_expect_double(
    const char *path,
    const char *key,
    double expected
)
{
    char actual[MU_LINE_CAP];
    char *end = NULL;
    double value;

    if (!mu_run_config_value(path, key, actual, sizeof(actual))) {
        return 0;
    }
    value = strtod(actual, &end);
    return end != NULL && end[0] == '\0' && value == expected;
}


static int mu_validate_run_config(
    const MuCanonicalDescriptor *descriptor,
    const VAECMuCanonicalContract *contract
)
{
    char event_hash[VAE_C_CHECKPOINT_SHA256_CAP];
    char tensor_hash[VAE_C_CHECKPOINT_SHA256_CAP];
    char train_hash[VAE_C_CHECKPOINT_SHA256_CAP];
    char validation_hash[VAE_C_CHECKPOINT_SHA256_CAP];
    char catalog_hash[VAE_C_CHECKPOINT_SHA256_CAP];
    const char *path = contract->run_config_path;

    if (!mu_hash_with_case(MU_EVENT_MANIFEST_SHA,
                           descriptor->uppercase_identity_hashes,
                           event_hash) ||
        !mu_hash_with_case(descriptor->tensor_manifest_sha256,
                           descriptor->uppercase_identity_hashes,
                           tensor_hash) ||
        !mu_hash_with_case(MU_TRAIN_SPLIT_SHA,
                           descriptor->uppercase_identity_hashes,
                           train_hash) ||
        !mu_hash_with_case(MU_VALIDATION_SPLIT_SHA,
                           descriptor->uppercase_identity_hashes,
                           validation_hash) ||
        !mu_hash_with_case(descriptor->catalog_sha256,
                           descriptor->uppercase_identity_hashes,
                           catalog_hash)) {
        return 0;
    }

    if (!mu_run_config_expect(path, "format", descriptor->run_config_format) ||
        !mu_run_config_expect(path, "code_commit", descriptor->code_commit) ||
        !mu_run_config_expect(path, "profile", descriptor->profile_id) ||
        !mu_run_config_expect_u64(path, "train_count",
                                  descriptor->train_count) ||
        !mu_run_config_expect_u64(path, "validation_count",
                                  descriptor->validation_count) ||
        !mu_run_config_expect_u64(path, "batch_size",
                                  descriptor->batch_size) ||
        !mu_run_config_expect_u64(path, "init_seed", UINT64_C(123)) ||
        !mu_run_config_expect_u64(path, "latent_seed", UINT64_C(456)) ||
        !mu_run_config_expect_u64(path, "augmentation_seed", UINT64_C(789)) ||
        !mu_run_config_expect_u64(path, "shuffle_seed", UINT64_C(101112)) ||
        !mu_run_config_expect_u64(path, "max_epochs",
                                  descriptor->max_epochs) ||
        !mu_run_config_expect_u64(path, "early_stopping_patience",
                                  descriptor->patience) ||
        !mu_run_config_expect_double(path, "adam_learning_rate",
                                     VAE_C_ADAM_LEARNING_RATE_V1) ||
        !mu_run_config_expect_double(path, "adam_beta1",
                                     VAE_C_ADAM_BETA1_V1) ||
        !mu_run_config_expect_double(path, "adam_beta2",
                                     VAE_C_ADAM_BETA2_V1) ||
        !mu_run_config_expect_double(path, "adam_epsilon",
                                     VAE_C_ADAM_EPSILON_V1) ||
        !mu_run_config_expect(path, "labels_used", "false") ||
        !mu_run_config_expect(path, "event_manifest",
                              MU_EVENT_MANIFEST_REL) ||
        !mu_run_config_expect(path, "event_manifest_sha256", event_hash) ||
        !mu_run_config_expect(path, "tensor_manifest",
                              descriptor->tensor_manifest_id) ||
        !mu_run_config_expect(path, "tensor_manifest_sha256", tensor_hash) ||
        !mu_run_config_expect(path, "train_split", MU_TRAIN_SPLIT_REL) ||
        !mu_run_config_expect(path, "train_split_sha256", train_hash) ||
        !mu_run_config_expect(path, "validation_split",
                              MU_VALIDATION_SPLIT_REL) ||
        !mu_run_config_expect(path, "validation_split_sha256",
                              validation_hash) ||
        !mu_run_config_expect(path, "profile_catalog",
                              descriptor->catalog_rel) ||
        !mu_run_config_expect(path, "profile_catalog_sha256", catalog_hash) ||
        !mu_run_config_expect(path, "compiler_id", "GNU C") ||
        !mu_run_config_expect(path, "compiler_version", "16.1.0") ||
        !mu_run_config_expect(path, "compiler_target",
                              "x86_64-w64-mingw32-ucrt64")) {
        return 0;
    }

    if (descriptor->uses_subset) {
        return
            mu_run_config_expect(path, "run_id", descriptor->run_id) &&
            mu_run_config_expect(path, "dataset_version",
                                 descriptor->dataset_version) &&
            mu_run_config_expect(path, "dataset_identity_sha256",
                                 descriptor->dataset_identity) &&
            mu_run_config_expect(path, "train_subset",
                                 descriptor->train_subset_rel) &&
            mu_run_config_expect(path, "train_subset_sha256",
                                 descriptor->train_subset_sha256) &&
            mu_run_config_expect(path, "train_selection_identity_sha256",
                                 descriptor->train_selection_identity) &&
            mu_run_config_expect(path, "validation_subset",
                                 descriptor->validation_subset_rel) &&
            mu_run_config_expect(path, "validation_subset_sha256",
                                 descriptor->validation_subset_sha256) &&
            mu_run_config_expect(
                path, "validation_selection_identity_sha256",
                descriptor->validation_selection_identity
            );
    }
    return 1;
}


static int mu_profile_matches(
    const VAECProfile *profile,
    const MuCanonicalDescriptor *descriptor
)
{
    size_t index;

    if (profile == NULL || descriptor == NULL ||
        strcmp(profile->profile_id, descriptor->profile_id) != 0 ||
        strcmp(profile->role, descriptor->profile_role) != 0 ||
        profile->input_side != descriptor->input_side ||
        profile->conv_depth != descriptor->conv_depth ||
        profile->latent_dim != descriptor->latent_dim ||
        profile->bottleneck_side != descriptor->bottleneck_side ||
        profile->expected_trainable_parameters_bn1d !=
            descriptor->trainable_parameters ||
        profile->expected_forward_mac_per_sample !=
            descriptor->forward_mac_per_sample ||
        strcmp(profile->profile_status, descriptor->profile_status) != 0) {
        return 0;
    }
    for (index = 0; index < VAE_C_MAX_CONV_DEPTH; index++) {
        if (profile->channels[index] != descriptor->channels[index]) {
            return 0;
        }
    }
    return 1;
}


static int mu_splits_cover_manifest(
    const GSSavedSplit *train,
    const GSSavedSplit *validation,
    size_t event_count
)
{
    unsigned char *seen;
    size_t index;
    int valid = 1;

    if (train == NULL || validation == NULL ||
        train->count + validation->count != event_count) {
        return 0;
    }
    seen = (unsigned char *)calloc(event_count, sizeof(*seen));
    if (seen == NULL) {
        return 0;
    }
    for (index = 0; index < train->count && valid; index++) {
        size_t manifest_index = train->indices[index];

        if (manifest_index >= event_count || seen[manifest_index]) {
            valid = 0;
        } else {
            seen[manifest_index] = 1u;
        }
    }
    for (index = 0; index < validation->count && valid; index++) {
        size_t manifest_index = validation->indices[index];

        if (manifest_index >= event_count || seen[manifest_index]) {
            valid = 0;
        } else {
            seen[manifest_index] = 1u;
        }
    }
    for (index = 0; index < event_count && valid; index++) {
        if (!seen[index]) {
            valid = 0;
        }
    }
    free(seen);
    return valid;
}


static int mu_subset_is_prefix(
    const GSSavedSplit *subset,
    const GSSavedSplit *full,
    size_t expected_count
)
{
    size_t index;

    if (subset == NULL || full == NULL || subset->count != expected_count ||
        full->count < subset->count) {
        return 0;
    }
    for (index = 0; index < subset->count; index++) {
        if (subset->indices[index] != full->indices[index]) {
            return 0;
        }
    }
    return 1;
}


static int mu_splits_are_disjoint(
    const GSSavedSplit *left,
    const GSSavedSplit *right,
    size_t event_count
)
{
    unsigned char *seen;
    size_t index;
    int valid = 1;

    if (left == NULL || right == NULL) {
        return 0;
    }
    seen = (unsigned char *)calloc(event_count, sizeof(*seen));
    if (seen == NULL) {
        return 0;
    }
    for (index = 0; index < left->count && valid; index++) {
        if (left->indices[index] >= event_count) {
            valid = 0;
        } else {
            seen[left->indices[index]] = 1u;
        }
    }
    for (index = 0; index < right->count && valid; index++) {
        if (right->indices[index] >= event_count ||
            seen[right->indices[index]]) {
            valid = 0;
        }
    }
    free(seen);
    return valid;
}


static int mu_tensor_manifest_matches(
    const GSTensorManifest *tensor_manifest,
    const GSEventManifest *events,
    const MuCanonicalDescriptor *descriptor
)
{
    size_t index;

    if (tensor_manifest == NULL || events == NULL || descriptor == NULL ||
        tensor_manifest->count != events->count ||
        strcmp(tensor_manifest->extra_column_name,
               descriptor->tensor_extra_column) != 0) {
        return 0;
    }
    for (index = 0; index < tensor_manifest->count; index++) {
        const GSTensorManifestRecord *tensor =
            &tensor_manifest->records[index];
        const GSEventRecord *event = &events->records[index];

        /* Labels are intentionally neither inspected nor compared. */
        if (tensor->event_index != event->event_index ||
            strcmp(tensor->event_id, event->event_id) != 0 ||
            tensor->channels != MU_CHANNELS ||
            tensor->height != (int)descriptor->input_side ||
            tensor->width != (int)descriptor->input_side ||
            strcmp(tensor->dtype, "uint8") != 0 ||
            strcmp(tensor->layout, "channel_major") != 0 ||
            strcmp(tensor->channel_order, "0.5;1.0;2.0;4.0") != 0 ||
            strcmp(tensor->extra_value, descriptor->tensor_extra_value) != 0) {
            return 0;
        }
    }
    return 1;
}


static VAECMuStatus mu_validate_population_sources(
    const MuCanonicalDescriptor *descriptor,
    const VAECMuCanonicalContract *contract,
    VAECProfile *profile,
    char *error_message,
    size_t error_capacity,
    int label_free
)
{
    VAECProfileCatalog catalog;
    GSEventManifest events = {NULL, 0u};
    GSSavedSplit official_train = {NULL, 0u, {0}};
    GSSavedSplit official_validation = {NULL, 0u, {0}};
    GSSavedSplit train_population = {NULL, 0u, {0}};
    GSSavedSplit validation_population = {NULL, 0u, {0}};
    GSTensorManifest tensor_manifest;
    GSTensorSpec tensor_spec;
    const VAECProfile *catalog_profile;
    char config_error[512];
    int valid = 0;
    int train_population_loaded = 0;
    int validation_population_loaded = 0;
    VAECMuStatus failure_status = VAE_C_MU_DATASET_MISMATCH;

    vae_c_profile_catalog_init(&catalog);
    memset(&tensor_manifest, 0, sizeof(tensor_manifest));
    config_error[0] = '\0';
    if (vae_c_profile_catalog_load_csv(
            contract->catalog_path, &catalog,
            config_error, sizeof(config_error)) != VAE_C_CONFIG_OK) {
        mu_error(error_message, error_capacity,
                 "profile catalog invalid: %s", config_error);
        failure_status = VAE_C_MU_PROFILE_MISMATCH;
        goto cleanup;
    }
    catalog_profile = vae_c_profile_catalog_find(
        &catalog, descriptor->profile_id
    );
    if (!mu_profile_matches(catalog_profile, descriptor)) {
        mu_error(error_message, error_capacity,
                 "canonical profile config mismatch");
        failure_status = VAE_C_MU_PROFILE_MISMATCH;
        goto cleanup;
    }
    if ((label_free ? gs_event_manifest_load_label_free(
             contract->event_manifest_path, &events) :
             gs_event_manifest_load(contract->event_manifest_path, &events)) != 0 ||
        events.count != MU_EVENT_COUNT) {
        mu_error(error_message, error_capacity,
                 "event manifest contract mismatch");
        goto cleanup;
    }
    if (gs_saved_split_load(
            contract->official_train_split_path, "train", &events,
            &official_train) != 0 ||
        official_train.count != MU_FULL_TRAIN_COUNT ||
        gs_saved_split_load(
            contract->official_validation_split_path, "validation", &events,
            &official_validation) != 0 ||
        official_validation.count != MU_FULL_VALIDATION_COUNT ||
        !mu_splits_cover_manifest(
            &official_train, &official_validation, events.count)) {
        mu_error(error_message, error_capacity,
                 "official split contract mismatch");
        goto cleanup;
    }
    if (descriptor->uses_subset) {
        if (gs_saved_split_load(
                contract->train_population_path, "train", &events,
                &train_population) != 0) {
            mu_error(error_message, error_capacity,
                     "train subset contract mismatch");
            goto cleanup;
        }
        train_population_loaded = 1;
        if (gs_saved_split_load(
                contract->validation_population_path, "validation", &events,
                &validation_population) != 0) {
            mu_error(error_message, error_capacity,
                     "validation subset contract mismatch");
            goto cleanup;
        }
        validation_population_loaded = 1;
        if (!mu_subset_is_prefix(
                &train_population, &official_train,
                (size_t)descriptor->train_count) ||
            !mu_subset_is_prefix(
                &validation_population, &official_validation,
                (size_t)descriptor->validation_count) ||
            !mu_splits_are_disjoint(
                &train_population, &validation_population, events.count)) {
            mu_error(error_message, error_capacity,
                     "canonical subset identity/order mismatch");
            goto cleanup;
        }
    }
    if (gs_tensor_spec_init(&tensor_spec, (int)descriptor->input_side) != 0 ||
        (label_free ? gs_tensor_manifest_load_label_free(
             contract->tensor_manifest_path, &tensor_spec,
             &tensor_manifest) : gs_tensor_manifest_load(
             contract->tensor_manifest_path, &tensor_spec,
             &tensor_manifest)) != 0 ||
        !mu_tensor_manifest_matches(
            &tensor_manifest, &events, descriptor)) {
        mu_error(error_message, error_capacity,
                 "central tensor manifest contract mismatch");
        goto cleanup;
    }
    *profile = *catalog_profile;
    valid = 1;

cleanup:
    gs_tensor_manifest_free(&tensor_manifest);
    if (validation_population_loaded) {
        gs_saved_split_free(&validation_population);
    }
    if (train_population_loaded) {
        gs_saved_split_free(&train_population);
    }
    gs_saved_split_free(&official_validation);
    gs_saved_split_free(&official_train);
    gs_event_manifest_free(&events);
    vae_c_profile_catalog_free(&catalog);
    return valid ? VAE_C_MU_OK : failure_status;
}


static int mu_build_expected_identity(
    const MuCanonicalDescriptor *descriptor,
    VAECCheckpointRunIdentity *identity,
    char *error_message,
    size_t error_capacity
)
{
    char profile_hash[VAE_C_CHECKPOINT_SHA256_CAP];
    char event_hash[VAE_C_CHECKPOINT_SHA256_CAP];
    char tensor_hash[VAE_C_CHECKPOINT_SHA256_CAP];
    char train_hash[VAE_C_CHECKPOINT_SHA256_CAP];
    char validation_hash[VAE_C_CHECKPOINT_SHA256_CAP];
    char dataset_identity[VAE_C_CHECKPOINT_ID_CAP];
    char train_identity[VAE_C_CHECKPOINT_ID_CAP];
    char validation_identity[VAE_C_CHECKPOINT_ID_CAP];
    char validation_policy[VAE_C_CHECKPOINT_POLICY_CAP];
    int policy_length;

    if (!mu_hash_with_case(descriptor->catalog_sha256,
                           descriptor->uppercase_identity_hashes,
                           profile_hash) ||
        !mu_hash_with_case(MU_EVENT_MANIFEST_SHA,
                           descriptor->uppercase_identity_hashes,
                           event_hash) ||
        !mu_hash_with_case(descriptor->tensor_manifest_sha256,
                           descriptor->uppercase_identity_hashes,
                           tensor_hash) ||
        !mu_hash_with_case(MU_TRAIN_SPLIT_SHA,
                           descriptor->uppercase_identity_hashes,
                           train_hash) ||
        !mu_hash_with_case(MU_VALIDATION_SPLIT_SHA,
                           descriptor->uppercase_identity_hashes,
                           validation_hash)) {
        return 0;
    }

    if (descriptor->uses_subset) {
        if (vae_c_identity_build_mini224_dataset_v1(
                descriptor->dataset_version,
                MU_EVENT_MANIFEST_SHA,
                descriptor->tensor_manifest_sha256,
                dataset_identity) != VAE_C_IDENTITY_OK ||
            vae_c_identity_build_mini224_train_selection_v1(
                dataset_identity,
                MU_TRAIN_SPLIT_SHA,
                descriptor->train_subset_sha256,
                descriptor->train_count,
                train_identity) != VAE_C_IDENTITY_OK ||
            vae_c_identity_build_mini224_validation_selection_v1(
                dataset_identity,
                MU_VALIDATION_SPLIT_SHA,
                descriptor->validation_subset_sha256,
                descriptor->validation_count,
                validation_identity) != VAE_C_IDENTITY_OK ||
            strcmp(dataset_identity, descriptor->dataset_identity) != 0 ||
            strcmp(train_identity,
                   descriptor->train_selection_identity) != 0 ||
            strcmp(validation_identity,
                   descriptor->validation_selection_identity) != 0) {
            mu_error(error_message, error_capacity,
                     "mini224 composite identity mismatch");
            return 0;
        }
        policy_length = snprintf(
            validation_policy, sizeof(validation_policy),
            "m224_controller_v1+m=%016" PRIx64 "+p=%016" PRIx64,
            descriptor->max_epochs, descriptor->patience
        );
        if (policy_length < 0 ||
            (size_t)policy_length >= sizeof(validation_policy)) {
            return 0;
        }
    } else {
        if (!mu_copy_text(dataset_identity, sizeof(dataset_identity),
                          descriptor->dataset_identity) ||
            !mu_copy_text(train_identity, sizeof(train_identity),
                          MU_TRAIN_SPLIT_REL) ||
            !mu_copy_text(validation_identity, sizeof(validation_identity),
                          MU_VALIDATION_SPLIT_REL) ||
            !mu_copy_text(validation_policy, sizeof(validation_policy),
                          descriptor->validation_identity_policy)) {
            return 0;
        }
    }

    vae_c_checkpoint_run_identity_init(identity);
    if (!mu_copy_text(identity->profile_id, sizeof(identity->profile_id),
                      descriptor->profile_id) ||
        !mu_copy_text(identity->profile_config_sha256,
                      sizeof(identity->profile_config_sha256), profile_hash) ||
        !mu_copy_text(identity->code_commit, sizeof(identity->code_commit),
                      descriptor->code_commit) ||
        !mu_copy_text(identity->dataset_id, sizeof(identity->dataset_id),
                      dataset_identity) ||
        !mu_copy_text(identity->event_manifest_id,
                      sizeof(identity->event_manifest_id),
                      MU_EVENT_MANIFEST_REL) ||
        !mu_copy_text(identity->event_manifest_sha256,
                      sizeof(identity->event_manifest_sha256), event_hash) ||
        !mu_copy_text(identity->tensor_manifest_id,
                      sizeof(identity->tensor_manifest_id),
                      descriptor->tensor_manifest_id) ||
        !mu_copy_text(identity->tensor_manifest_sha256,
                      sizeof(identity->tensor_manifest_sha256), tensor_hash) ||
        !mu_copy_text(identity->train_split_id,
                      sizeof(identity->train_split_id), train_identity) ||
        !mu_copy_text(identity->train_split_sha256,
                      sizeof(identity->train_split_sha256), train_hash) ||
        !mu_copy_text(identity->validation_split_id,
                      sizeof(identity->validation_split_id),
                      validation_identity) ||
        !mu_copy_text(identity->validation_split_sha256,
                      sizeof(identity->validation_split_sha256),
                      validation_hash) ||
        !mu_copy_text(identity->augmentation_policy,
                      sizeof(identity->augmentation_policy),
                      descriptor->augmentation_identity_policy) ||
        !mu_copy_text(identity->shuffle_policy,
                      sizeof(identity->shuffle_policy),
                      "fisher_yates_unbiased_v1") ||
        !mu_copy_text(identity->latent_policy,
                      sizeof(identity->latent_policy),
                      descriptor->latent_identity_policy) ||
        !mu_copy_text(identity->validation_policy,
                      sizeof(identity->validation_policy),
                      validation_policy) ||
        !mu_copy_text(identity->compiler_id,
                      sizeof(identity->compiler_id), "GNU C") ||
        !mu_copy_text(identity->compiler_version,
                      sizeof(identity->compiler_version), "16.1.0") ||
        !mu_copy_text(identity->compiler_target,
                      sizeof(identity->compiler_target),
                      "x86_64-w64-mingw32-ucrt64")) {
        return 0;
    }
    identity->initialization_seed = UINT64_C(123);
    identity->latent_seed = UINT64_C(456);
    identity->augmentation_seed = UINT64_C(789);
    identity->shuffle_seed = UINT64_C(101112);
    identity->max_batch_size = descriptor->batch_size;
    identity->batch_size = descriptor->batch_size;
    identity->optimizer_learning_rate = VAE_C_ADAM_LEARNING_RATE_V1;
    identity->optimizer_beta1 = VAE_C_ADAM_BETA1_V1;
    identity->optimizer_beta2 = VAE_C_ADAM_BETA2_V1;
    identity->optimizer_epsilon = VAE_C_ADAM_EPSILON_V1;
    return 1;
}


static int mu_platform_is_binary32_little_endian(void)
{
    const uint16_t value = UINT16_C(1);

    return sizeof(float) == 4u && FLT_RADIX == 2 && FLT_MANT_DIG == 24 &&
        FLT_MAX_EXP == 128 && *((const unsigned char *)&value) == 1u;
}


static int mu_expected_bytes(
    uint64_t rows,
    uint64_t cols,
    uint64_t *bytes
)
{
    if (bytes == NULL || (cols != 0u && rows > UINT64_MAX / cols)) {
        return 0;
    }
    rows *= cols;
    if (rows > UINT64_MAX / UINT64_C(4)) {
        return 0;
    }
    *bytes = rows * UINT64_C(4);
    return 1;
}


static int mu_build_artifact_v1(
    const MuCanonicalDescriptor *descriptor,
    VAECMuArtifactContractV1 *artifact
)
{
    memset(artifact, 0, sizeof(*artifact));
    if (!mu_copy_text(artifact->format, sizeof(artifact->format),
                      VAE_C_MU_ARTIFACT_V1) ||
        !mu_copy_text(artifact->embedding_spec_file,
                      sizeof(artifact->embedding_spec_file),
                      "embedding_spec.txt") ||
        !mu_copy_text(artifact->checksum_file,
                      sizeof(artifact->checksum_file), "sha256.txt") ||
        !mu_copy_text(artifact->representation,
                      sizeof(artifact->representation), "posterior_mu") ||
        !mu_copy_text(artifact->view, sizeof(artifact->view), "central") ||
        !mu_copy_text(artifact->row_order, sizeof(artifact->row_order),
                      "native_split_pos_ascending") ||
        !mu_copy_text(
            artifact->row_map_schema, sizeof(artifact->row_map_schema),
            "row_index,split_pos,manifest_index,event_index,event_id,split"
        ) ||
        !mu_copy_text(artifact->profile_id, sizeof(artifact->profile_id),
                      descriptor->profile_id) ||
        !mu_copy_text(artifact->profile_config_sha256,
                      sizeof(artifact->profile_config_sha256),
                      descriptor->catalog_sha256) ||
        !mu_copy_text(artifact->run_id, sizeof(artifact->run_id),
                      descriptor->run_id) ||
        !mu_copy_text(artifact->run_config_sha256,
                      sizeof(artifact->run_config_sha256),
                      descriptor->run_config_sha256) ||
        !mu_copy_text(artifact->checkpoint_kind,
                      sizeof(artifact->checkpoint_kind), "BEST") ||
        !mu_copy_text(artifact->checkpoint_sha256,
                      sizeof(artifact->checkpoint_sha256),
                      descriptor->checkpoint_sha256) ||
        !mu_copy_text(artifact->checkpoint_code_commit,
                      sizeof(artifact->checkpoint_code_commit),
                      descriptor->code_commit) ||
        !mu_copy_text(artifact->catalog_sha256,
                      sizeof(artifact->catalog_sha256),
                      descriptor->catalog_sha256) ||
        !mu_copy_text(artifact->event_manifest_sha256,
                      sizeof(artifact->event_manifest_sha256),
                      MU_EVENT_MANIFEST_SHA) ||
        !mu_copy_text(artifact->tensor_manifest_sha256,
                      sizeof(artifact->tensor_manifest_sha256),
                      descriptor->tensor_manifest_sha256) ||
        !mu_copy_text(artifact->train_split_sha256,
                      sizeof(artifact->train_split_sha256),
                      MU_TRAIN_SPLIT_SHA) ||
        !mu_copy_text(artifact->validation_split_sha256,
                      sizeof(artifact->validation_split_sha256),
                      MU_VALIDATION_SPLIT_SHA) ||
        !mu_copy_text(artifact->population_identity,
                      sizeof(artifact->population_identity),
                      descriptor->population_identity) ||
        !mu_copy_text(artifact->checkpoint_dataset_id,
                      sizeof(artifact->checkpoint_dataset_id),
                      descriptor->dataset_identity) ||
        !mu_copy_text(artifact->batch_norm_mode,
                      sizeof(artifact->batch_norm_mode), "EVAL") ||
        !mu_copy_text(artifact->dtype, sizeof(artifact->dtype),
                      "float32_ieee754") ||
        !mu_copy_text(artifact->endian, sizeof(artifact->endian), "little") ||
        !mu_copy_text(artifact->layout, sizeof(artifact->layout),
                      "row_major_[N,latent_dim]") ||
        !mu_copy_text(artifact->input_normalization,
                      sizeof(artifact->input_normalization),
                      "uint8_div_255") ||
        !mu_copy_text(artifact->latent_scaling,
                      sizeof(artifact->latent_scaling), "none") ||
        !mu_copy_text(artifact->train.matrix_file,
                      sizeof(artifact->train.matrix_file),
                      "train_mu_f32.bin") ||
        !mu_copy_text(artifact->train.row_map_file,
                      sizeof(artifact->train.row_map_file),
                      "train_rows.csv") ||
        !mu_copy_text(artifact->validation.matrix_file,
                      sizeof(artifact->validation.matrix_file),
                      "validation_mu_f32.bin") ||
        !mu_copy_text(artifact->validation.row_map_file,
                      sizeof(artifact->validation.row_map_file),
                      "validation_rows.csv")) {
        return 0;
    }
    if (descriptor->uses_subset) {
        if (!mu_copy_text(artifact->checkpoint_train_selection_id,
                          sizeof(artifact->checkpoint_train_selection_id),
                          descriptor->train_selection_identity) ||
            !mu_copy_text(
                artifact->checkpoint_validation_selection_id,
                sizeof(artifact->checkpoint_validation_selection_id),
                descriptor->validation_selection_identity)) {
            return 0;
        }
    } else if (!mu_copy_text(
                   artifact->checkpoint_train_selection_id,
                   sizeof(artifact->checkpoint_train_selection_id),
                   MU_TRAIN_SPLIT_REL) ||
               !mu_copy_text(
                   artifact->checkpoint_validation_selection_id,
                   sizeof(artifact->checkpoint_validation_selection_id),
                   MU_VALIDATION_SPLIT_REL)) {
        return 0;
    }
    if (descriptor->uses_subset) {
        if (!mu_copy_text(artifact->train_subset_sha256,
                          sizeof(artifact->train_subset_sha256),
                          descriptor->train_subset_sha256) ||
            !mu_copy_text(artifact->validation_subset_sha256,
                          sizeof(artifact->validation_subset_sha256),
                          descriptor->validation_subset_sha256)) {
            return 0;
        }
    } else if (!mu_copy_text(
                   artifact->train_subset_sha256,
                   sizeof(artifact->train_subset_sha256),
                   "not_applicable_full_split") ||
               !mu_copy_text(
                   artifact->validation_subset_sha256,
                   sizeof(artifact->validation_subset_sha256),
                   "not_applicable_full_split")) {
        return 0;
    }

    artifact->latent_dim = descriptor->latent_dim;
    artifact->checkpoint_epoch = descriptor->checkpoint_epoch;
    artifact->checkpoint_optimizer_step =
        descriptor->checkpoint_optimizer_step;
    artifact->encoder_only = true;
    artifact->epsilon_used = false;
    artifact->latent_sampled = false;
    artifact->decoder_executed = false;
    artifact->labels_used = false;
    artifact->train.rows = descriptor->train_count;
    artifact->train.cols = descriptor->latent_dim;
    artifact->validation.rows = descriptor->validation_count;
    artifact->validation.cols = descriptor->latent_dim;
    return mu_expected_bytes(
               artifact->train.rows, artifact->train.cols,
               &artifact->train.expected_bytes) &&
        mu_expected_bytes(
               artifact->validation.rows, artifact->validation.cols,
               &artifact->validation.expected_bytes);
}


void vae_c_mu_canonical_contract_init(VAECMuCanonicalContract *contract)
{
    if (contract != NULL) {
        memset(contract, 0, sizeof(*contract));
    }
}


void vae_c_mu_validation_report_init(VAECMuValidationReport *report)
{
    if (report != NULL) {
        memset(report, 0, sizeof(*report));
        report->checkpoint_status = VAE_C_CHECKPOINT_INVALID_STATE;
        vae_c_checkpoint_metadata_init(&report->loaded_metadata);
    }
}


static VAECMuStatus mu_build_canonical_contract_impl(
    const char *repository_root,
    const char *run_or_profile,
    VAECMuCanonicalContract *contract,
    char *error_message,
    size_t error_message_capacity,
    int label_free
)
{
    const MuCanonicalDescriptor *descriptor;
    VAECMuCanonicalContract candidate;
    char relative[VAE_C_MU_PATH_CAP];
    int length;
    VAECMuStatus status;

    if (error_message != NULL && error_message_capacity != 0u) {
        error_message[0] = '\0';
    }
    if (repository_root == NULL || repository_root[0] == '\0' ||
        run_or_profile == NULL || run_or_profile[0] == '\0' ||
        contract == NULL) {
        mu_error(error_message, error_message_capacity,
                 "invalid canonical-contract arguments");
        return VAE_C_MU_INVALID_ARGUMENT;
    }
    descriptor = mu_find_descriptor(run_or_profile);
    if (descriptor == NULL) {
        mu_error(error_message, error_message_capacity,
                 "unsupported canonical run/profile: %s", run_or_profile);
        return VAE_C_MU_UNSUPPORTED_PROFILE;
    }
    vae_c_mu_canonical_contract_init(&candidate);
    if (!mu_copy_text(candidate.run_token, sizeof(candidate.run_token),
                      descriptor->run_token) ||
        !mu_copy_text(candidate.repository_root,
                      sizeof(candidate.repository_root), repository_root) ||
        !mu_copy_text(candidate.run_id, sizeof(candidate.run_id),
                      descriptor->run_id) ||
        !mu_join_path(repository_root, descriptor->catalog_rel,
                      candidate.catalog_path) ||
        !mu_join_path(repository_root, MU_EVENT_MANIFEST_REL,
                      candidate.event_manifest_path) ||
        !mu_join_path(repository_root, descriptor->tensor_manifest_rel,
                      candidate.tensor_manifest_path) ||
        !mu_join_path(repository_root, MU_TRAIN_SPLIT_REL,
                      candidate.official_train_split_path) ||
        !mu_join_path(repository_root, MU_VALIDATION_SPLIT_REL,
                      candidate.official_validation_split_path)) {
        mu_error(error_message, error_message_capacity,
                 "canonical path overflow");
        return VAE_C_MU_PATH_OVERFLOW;
    }
    if (descriptor->uses_subset) {
        if (!mu_join_path(repository_root, descriptor->train_subset_rel,
                          candidate.train_population_path) ||
            !mu_join_path(repository_root, descriptor->validation_subset_rel,
                          candidate.validation_population_path) ||
            !mu_copy_text(candidate.train_subset_sha256,
                          sizeof(candidate.train_subset_sha256),
                          descriptor->train_subset_sha256) ||
            !mu_copy_text(candidate.validation_subset_sha256,
                          sizeof(candidate.validation_subset_sha256),
                          descriptor->validation_subset_sha256)) {
            mu_error(error_message, error_message_capacity,
                     "canonical subset path overflow");
            return VAE_C_MU_PATH_OVERFLOW;
        }
    } else if (!mu_copy_text(
                   candidate.train_population_path,
                   sizeof(candidate.train_population_path),
                   candidate.official_train_split_path) ||
               !mu_copy_text(
                   candidate.validation_population_path,
                   sizeof(candidate.validation_population_path),
                   candidate.official_validation_split_path)) {
        return VAE_C_MU_PATH_OVERFLOW;
    }
    length = snprintf(relative, sizeof(relative), "%s/run_config.txt",
                      descriptor->run_directory_rel);
    if (length < 0 || (size_t)length >= sizeof(relative) ||
        !mu_join_path(repository_root, relative,
                      candidate.run_config_path)) {
        return VAE_C_MU_PATH_OVERFLOW;
    }
    length = snprintf(relative, sizeof(relative), "%s/best.ckpt",
                      descriptor->run_directory_rel);
    if (length < 0 || (size_t)length >= sizeof(relative) ||
        !mu_join_path(repository_root, relative,
                      candidate.checkpoint_path)) {
        return VAE_C_MU_PATH_OVERFLOW;
    }

#define MU_VERIFY_SOURCE(path, hash, description)                            \
    do {                                                                     \
        status = mu_verify_hash(                                             \
            (path), (hash), (description), error_message,                   \
            error_message_capacity);                                         \
        if (status != VAE_C_MU_OK) {                                         \
            return status;                                                   \
        }                                                                    \
    } while (0)

    MU_VERIFY_SOURCE(candidate.catalog_path, descriptor->catalog_sha256,
                     "profile catalog");
    MU_VERIFY_SOURCE(candidate.event_manifest_path, MU_EVENT_MANIFEST_SHA,
                     "event manifest");
    MU_VERIFY_SOURCE(candidate.tensor_manifest_path,
                     descriptor->tensor_manifest_sha256,
                     "central tensor manifest");
    MU_VERIFY_SOURCE(candidate.official_train_split_path,
                     MU_TRAIN_SPLIT_SHA, "official train split");
    MU_VERIFY_SOURCE(candidate.official_validation_split_path,
                     MU_VALIDATION_SPLIT_SHA,
                     "official validation split");
    if (descriptor->uses_subset) {
        MU_VERIFY_SOURCE(candidate.train_population_path,
                         descriptor->train_subset_sha256,
                         "canonical train subset");
        MU_VERIFY_SOURCE(candidate.validation_population_path,
                         descriptor->validation_subset_sha256,
                         "canonical validation subset");
    }
    MU_VERIFY_SOURCE(candidate.run_config_path,
                     descriptor->run_config_sha256, "run config");
#undef MU_VERIFY_SOURCE

    if (!mu_validate_run_config(descriptor, &candidate)) {
        mu_error(error_message, error_message_capacity,
                 "run config fields mismatch canonical registry");
        return VAE_C_MU_RUN_CONFIG_MISMATCH;
    }
    status = mu_validate_population_sources(
        descriptor, &candidate, &candidate.profile,
        error_message, error_message_capacity, label_free
    );
    if (status != VAE_C_MU_OK) {
        return status;
    }
    if (!mu_build_expected_identity(
            descriptor, &candidate.expected_identity,
            error_message, error_message_capacity)) {
        if (error_message != NULL && error_message_capacity != 0u &&
            error_message[0] == '\0') {
            mu_error(error_message, error_message_capacity,
                     "cannot construct expected checkpoint identity");
        }
        return VAE_C_MU_IDENTITY_MISMATCH;
    }
    if (!mu_platform_is_binary32_little_endian()) {
        mu_error(error_message, error_message_capacity,
                 "platform is not IEEE-754 binary32 little-endian");
        return VAE_C_MU_PLATFORM_MISMATCH;
    }
    if (!mu_build_artifact_v1(descriptor, &candidate.artifact_v1) ||
        !mu_copy_text(candidate.run_config_sha256,
                      sizeof(candidate.run_config_sha256),
                      descriptor->run_config_sha256) ||
        !mu_copy_text(candidate.checkpoint_sha256,
                      sizeof(candidate.checkpoint_sha256),
                      descriptor->checkpoint_sha256)) {
        return VAE_C_MU_OVERFLOW;
    }
    candidate.expected_kind = VAE_C_CHECKPOINT_KIND_BEST;
    candidate.expected_epoch = descriptor->checkpoint_epoch;
    candidate.expected_optimizer_step =
        descriptor->checkpoint_optimizer_step;
    candidate.event_count = MU_EVENT_COUNT;
    candidate.train_count = descriptor->train_count;
    candidate.validation_count = descriptor->validation_count;
    candidate.uses_subset = descriptor->uses_subset;
    *contract = candidate;
    return VAE_C_MU_OK;
}


static int mu_identity_equal_full(
    const VAECCheckpointRunIdentity *left,
    const VAECCheckpointRunIdentity *right
)
{
    return left != NULL && right != NULL &&
        strcmp(left->profile_id, right->profile_id) == 0 &&
        strcmp(left->profile_config_sha256,
               right->profile_config_sha256) == 0 &&
        strcmp(left->code_commit, right->code_commit) == 0 &&
        strcmp(left->dataset_id, right->dataset_id) == 0 &&
        strcmp(left->event_manifest_id, right->event_manifest_id) == 0 &&
        strcmp(left->event_manifest_sha256,
               right->event_manifest_sha256) == 0 &&
        strcmp(left->tensor_manifest_id, right->tensor_manifest_id) == 0 &&
        strcmp(left->tensor_manifest_sha256,
               right->tensor_manifest_sha256) == 0 &&
        strcmp(left->train_split_id, right->train_split_id) == 0 &&
        strcmp(left->train_split_sha256,
               right->train_split_sha256) == 0 &&
        strcmp(left->validation_split_id,
               right->validation_split_id) == 0 &&
        strcmp(left->validation_split_sha256,
               right->validation_split_sha256) == 0 &&
        strcmp(left->augmentation_policy,
               right->augmentation_policy) == 0 &&
        strcmp(left->shuffle_policy, right->shuffle_policy) == 0 &&
        strcmp(left->latent_policy, right->latent_policy) == 0 &&
        strcmp(left->validation_policy, right->validation_policy) == 0 &&
        strcmp(left->compiler_id, right->compiler_id) == 0 &&
        strcmp(left->compiler_version, right->compiler_version) == 0 &&
        strcmp(left->compiler_target, right->compiler_target) == 0 &&
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
}


VAECMuStatus vae_c_mu_build_canonical_contract(
    const char *repository_root,
    const char *run_or_profile,
    VAECMuCanonicalContract *contract,
    char *error_message,
    size_t error_message_capacity
)
{
    return mu_build_canonical_contract_impl(
        repository_root, run_or_profile, contract, error_message,
        error_message_capacity, 0);
}


VAECMuStatus vae_c_mu_build_canonical_contract_label_free(
    const char *repository_root,
    const char *run_or_profile,
    VAECMuCanonicalContract *contract,
    char *error_message,
    size_t error_message_capacity
)
{
    return mu_build_canonical_contract_impl(
        repository_root, run_or_profile, contract, error_message,
        error_message_capacity, 1);
}


VAECMuStatus vae_c_mu_validate_checkpoint(
    const VAECMuCanonicalContract *contract,
    VAECMuValidationReport *report,
    char *error_message,
    size_t error_message_capacity
)
{
    VAECMuValidationReport candidate;
    VAECModel model;
    VAECAdam optimizer;
    char actual_sha[VAE_C_CHECKPOINT_SHA256_CAP];
    VAECMuStatus status;
    VAECOpStatus op_status;

    if (error_message != NULL && error_message_capacity != 0u) {
        error_message[0] = '\0';
    }
    if (contract == NULL || report == NULL ||
        contract->checkpoint_path[0] == '\0' ||
        contract->checkpoint_sha256[0] == '\0') {
        mu_error(error_message, error_message_capacity,
                 "invalid checkpoint-validation arguments");
        return VAE_C_MU_INVALID_ARGUMENT;
    }
    vae_c_mu_validation_report_init(&candidate);
    candidate.source_contract_validated = true;
    status = vae_c_mu_file_sha256(contract->checkpoint_path, actual_sha);
    if (status != VAE_C_MU_OK) {
        mu_error(error_message, error_message_capacity,
                 "cannot hash canonical checkpoint");
        *report = candidate;
        return status;
    }
    if (!mu_hash_equal(actual_sha, contract->checkpoint_sha256)) {
        mu_error(error_message, error_message_capacity,
                 "canonical checkpoint external SHA-256 mismatch");
        *report = candidate;
        return VAE_C_MU_HASH_MISMATCH;
    }
    candidate.checkpoint_hash_verified = true;

    vae_c_model_init(&model);
    vae_c_adam_init(&optimizer);
    op_status = vae_c_model_allocate(
        &model, &contract->profile,
        (size_t)contract->expected_identity.max_batch_size
    );
    if (op_status == VAE_C_OP_OK) {
        op_status = vae_c_model_initialize_parameters_v1(
            &model, contract->expected_identity.initialization_seed
        );
    }
    if (op_status == VAE_C_OP_OK) {
        op_status = vae_c_adam_configure_v1(&optimizer);
    }
    if (op_status != VAE_C_OP_OK) {
        mu_error(error_message, error_message_capacity,
                 "model/Adam setup failed: %s",
                 vae_c_op_status_name(op_status));
        vae_c_model_free(&model);
        vae_c_adam_free(&optimizer);
        *report = candidate;
        return VAE_C_MU_MODEL_SETUP_FAILED;
    }

    candidate.checkpoint_load_called = true;
    candidate.checkpoint_status = vae_c_checkpoint_load(
        contract->checkpoint_path,
        &contract->expected_identity,
        &model,
        &optimizer,
        &candidate.loaded_metadata
    );
    if (candidate.checkpoint_status != VAE_C_CHECKPOINT_OK) {
        mu_error(error_message, error_message_capacity,
                 "checkpoint load failed: %s",
                 vae_c_checkpoint_status_name(candidate.checkpoint_status));
        vae_c_model_free(&model);
        vae_c_adam_free(&optimizer);
        *report = candidate;
        return VAE_C_MU_CHECKPOINT_LOAD_FAILED;
    }
    candidate.checkpoint_loaded = true;
    candidate.loaded_optimizer_step = optimizer.step;
    status = vae_c_mu_file_sha256(contract->checkpoint_path, actual_sha);
    if (status != VAE_C_MU_OK ||
        !mu_hash_equal(actual_sha, contract->checkpoint_sha256)) {
        mu_error(error_message, error_message_capacity,
                 "canonical checkpoint changed during validation");
        vae_c_model_free(&model);
        vae_c_adam_free(&optimizer);
        *report = candidate;
        return status == VAE_C_MU_OK ? VAE_C_MU_HASH_MISMATCH : status;
    }
    candidate.checkpoint_hash_reverified_after_load = true;
    if (!mu_identity_equal_full(
            &candidate.loaded_metadata.run_identity,
            &contract->expected_identity) ||
        candidate.loaded_metadata.kind != contract->expected_kind ||
        candidate.loaded_metadata.last_completed_epoch !=
            contract->expected_epoch ||
        !candidate.loaded_metadata.best_defined ||
        candidate.loaded_metadata.best_epoch != contract->expected_epoch ||
        optimizer.step != contract->expected_optimizer_step ||
        strcmp(model.profile_id, contract->profile.profile_id) != 0 ||
        model.config.latent_dim != contract->profile.latent_dim ||
        !model.parameters_initialized || !optimizer.configured) {
        mu_error(error_message, error_message_capacity,
                 "checkpoint metadata/profile/step mismatch");
        vae_c_model_free(&model);
        vae_c_adam_free(&optimizer);
        *report = candidate;
        return VAE_C_MU_CHECKPOINT_METADATA_MISMATCH;
    }
    candidate.metadata_verified = true;
    vae_c_model_free(&model);
    vae_c_adam_free(&optimizer);
    *report = candidate;
    return VAE_C_MU_OK;
}


static VAECMuStatus mu_validate_only_impl(
    const char *repository_root,
    const char *run_or_profile,
    VAECMuCanonicalContract *contract,
    VAECMuValidationReport *report,
    char *error_message,
    size_t error_message_capacity,
    int label_free
)
{
    VAECMuCanonicalContract candidate_contract;
    VAECMuValidationReport candidate_report;
    VAECMuStatus status;

    if (contract == NULL || report == NULL) {
        return VAE_C_MU_INVALID_ARGUMENT;
    }
    vae_c_mu_canonical_contract_init(&candidate_contract);
    vae_c_mu_validation_report_init(&candidate_report);
    status = mu_build_canonical_contract_impl(
        repository_root, run_or_profile, &candidate_contract,
        error_message, error_message_capacity, label_free
    );
    if (status != VAE_C_MU_OK) {
        return status;
    }
    status = vae_c_mu_validate_checkpoint(
        &candidate_contract, &candidate_report,
        error_message, error_message_capacity
    );
    *contract = candidate_contract;
    *report = candidate_report;
    return status;
}


VAECMuStatus vae_c_mu_validate_only(
    const char *repository_root,
    const char *run_or_profile,
    VAECMuCanonicalContract *contract,
    VAECMuValidationReport *report,
    char *error_message,
    size_t error_message_capacity
)
{
    return mu_validate_only_impl(repository_root, run_or_profile, contract,
                                 report, error_message,
                                 error_message_capacity, 0);
}


VAECMuStatus vae_c_mu_validate_only_label_free(
    const char *repository_root,
    const char *run_or_profile,
    VAECMuCanonicalContract *contract,
    VAECMuValidationReport *report,
    char *error_message,
    size_t error_message_capacity
)
{
    return mu_validate_only_impl(repository_root, run_or_profile, contract,
                                 report, error_message,
                                 error_message_capacity, 1);
}


const char *vae_c_mu_status_name(VAECMuStatus status)
{
    switch (status) {
        case VAE_C_MU_OK:
            return "ok";
        case VAE_C_MU_INVALID_ARGUMENT:
            return "invalid_argument";
        case VAE_C_MU_UNSUPPORTED_PROFILE:
            return "unsupported_profile";
        case VAE_C_MU_PATH_OVERFLOW:
            return "path_overflow";
        case VAE_C_MU_IO_ERROR:
            return "io_error";
        case VAE_C_MU_HASH_MISMATCH:
            return "hash_mismatch";
        case VAE_C_MU_RUN_CONFIG_MISMATCH:
            return "run_config_mismatch";
        case VAE_C_MU_PROFILE_MISMATCH:
            return "profile_mismatch";
        case VAE_C_MU_DATASET_MISMATCH:
            return "dataset_mismatch";
        case VAE_C_MU_IDENTITY_MISMATCH:
            return "identity_mismatch";
        case VAE_C_MU_PLATFORM_MISMATCH:
            return "platform_mismatch";
        case VAE_C_MU_MODEL_SETUP_FAILED:
            return "model_setup_failed";
        case VAE_C_MU_CHECKPOINT_LOAD_FAILED:
            return "checkpoint_load_failed";
        case VAE_C_MU_CHECKPOINT_METADATA_MISMATCH:
            return "checkpoint_metadata_mismatch";
        case VAE_C_MU_OVERFLOW:
            return "overflow";
        default:
            return "unknown";
    }
}
