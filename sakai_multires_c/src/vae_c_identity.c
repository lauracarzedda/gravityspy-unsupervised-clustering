#include "vae_c_identity.h"

#include <stddef.h>
#include <string.h>


enum {
    IDENTITY_PREIMAGE_CAP = 512
};


typedef struct {
    char bytes[IDENTITY_PREIMAGE_CAP];
    size_t size;
} IdentityBuilder;


static int identity_dataset_version_is_valid(const char *value)
{
    size_t index;

    if (value == NULL || value[0] == '\0') {
        return 0;
    }
    for (index = 0; index < VAE_C_IDENTITY_DATASET_VERSION_CAP; index++) {
        unsigned char byte = (unsigned char)value[index];

        if (byte == '\0') {
            return 1;
        }
        if (!((byte >= (unsigned char)'A' && byte <= (unsigned char)'Z') ||
              (byte >= (unsigned char)'a' && byte <= (unsigned char)'z') ||
              (byte >= (unsigned char)'0' && byte <= (unsigned char)'9') ||
              byte == (unsigned char)'.' || byte == (unsigned char)'_' ||
              byte == (unsigned char)'-')) {
            return 0;
        }
    }
    return 0;
}


static int identity_sha256_text_is_valid(const char *value)
{
    size_t index;

    if (value == NULL) {
        return 0;
    }
    for (index = 0; index < 64u; index++) {
        char byte = value[index];

        if (!((byte >= '0' && byte <= '9') ||
              (byte >= 'a' && byte <= 'f'))) {
            return 0;
        }
    }
    return value[64] == '\0';
}


static int identity_builder_append(
    IdentityBuilder *builder,
    const char *bytes,
    size_t size
)
{
    if (builder == NULL || (size != 0u && bytes == NULL) ||
        size > sizeof(builder->bytes) ||
        builder->size > sizeof(builder->bytes) - size) {
        return 0;
    }
    if (size != 0u) {
        memcpy(builder->bytes + builder->size, bytes, size);
        builder->size += size;
    }
    return 1;
}


static int identity_builder_append_text(
    IdentityBuilder *builder,
    const char *text
)
{
    return text != NULL &&
        identity_builder_append(builder, text, strlen(text));
}


static int identity_builder_append_line(
    IdentityBuilder *builder,
    const char *key,
    const char *value
)
{
    return
        identity_builder_append_text(builder, key) &&
        identity_builder_append(builder, "=", 1u) &&
        identity_builder_append_text(builder, value) &&
        identity_builder_append(builder, "\n", 1u);
}


static int identity_builder_append_count_line(
    IdentityBuilder *builder,
    uint64_t value
)
{
    char reverse[20];
    char decimal[20];
    size_t digits = 0;
    size_t index;

    do {
        reverse[digits++] = (char)('0' + value % UINT64_C(10));
        value /= UINT64_C(10);
    } while (value != UINT64_C(0));
    for (index = 0; index < digits; index++) {
        decimal[index] = reverse[digits - index - 1u];
    }
    return
        identity_builder_append_text(builder, "subset_count=") &&
        identity_builder_append(builder, decimal, digits) &&
        identity_builder_append(builder, "\n", 1u);
}


static VAECIdentityStatus identity_hash_builder(
    const IdentityBuilder *builder,
    char identity_sha256[VAE_C_SHA256_HEX_SIZE]
)
{
    char candidate[VAE_C_SHA256_HEX_SIZE];
    VAECSha256Status status;

    status = vae_c_sha256_digest_hex_lower(
        builder->bytes, builder->size, candidate
    );
    if (status != VAE_C_SHA256_OK) {
        return VAE_C_IDENTITY_HASH_FAILED;
    }
    memcpy(identity_sha256, candidate, sizeof(candidate));
    return VAE_C_IDENTITY_OK;
}


VAECIdentityStatus vae_c_identity_build_mini224_dataset_v1(
    const char *dataset_version,
    const char *event_manifest_sha256,
    const char *tensor_manifest_sha256,
    char identity_sha256[VAE_C_SHA256_HEX_SIZE]
)
{
    IdentityBuilder builder = {{0}, 0u};

    if (identity_sha256 == NULL) {
        return VAE_C_IDENTITY_INVALID_ARGUMENT;
    }
    if (!identity_dataset_version_is_valid(dataset_version) ||
        !identity_sha256_text_is_valid(event_manifest_sha256) ||
        !identity_sha256_text_is_valid(tensor_manifest_sha256)) {
        return VAE_C_IDENTITY_INVALID_COMPONENT;
    }
    if (!identity_builder_append_line(
            &builder, "namespace", "mini224_dataset_identity_v1") ||
        !identity_builder_append_line(&builder, "kind", "dataset") ||
        !identity_builder_append_line(
            &builder, "dataset_version", dataset_version) ||
        !identity_builder_append_line(
            &builder, "event_manifest_sha256", event_manifest_sha256) ||
        !identity_builder_append_line(
            &builder, "tensor_manifest_sha256", tensor_manifest_sha256)) {
        return VAE_C_IDENTITY_OVERFLOW;
    }
    return identity_hash_builder(&builder, identity_sha256);
}


static VAECIdentityStatus identity_build_mini224_selection_v1(
    const char *namespace_value,
    const char *kind_value,
    const char *dataset_identity_sha256,
    const char *official_split_sha256,
    const char *subset_sha256,
    uint64_t subset_count,
    char identity_sha256[VAE_C_SHA256_HEX_SIZE]
)
{
    IdentityBuilder builder = {{0}, 0u};

    if (identity_sha256 == NULL) {
        return VAE_C_IDENTITY_INVALID_ARGUMENT;
    }
    if (!identity_sha256_text_is_valid(dataset_identity_sha256) ||
        !identity_sha256_text_is_valid(official_split_sha256) ||
        !identity_sha256_text_is_valid(subset_sha256) ||
        subset_count == UINT64_C(0)) {
        return VAE_C_IDENTITY_INVALID_COMPONENT;
    }
    if (!identity_builder_append_line(
            &builder, "namespace", namespace_value) ||
        !identity_builder_append_line(&builder, "kind", kind_value) ||
        !identity_builder_append_line(
            &builder,
            "dataset_identity_sha256",
            dataset_identity_sha256) ||
        !identity_builder_append_line(
            &builder, "official_split_sha256", official_split_sha256) ||
        !identity_builder_append_line(
            &builder, "subset_sha256", subset_sha256) ||
        !identity_builder_append_count_line(&builder, subset_count)) {
        return VAE_C_IDENTITY_OVERFLOW;
    }
    return identity_hash_builder(&builder, identity_sha256);
}


VAECIdentityStatus vae_c_identity_build_mini224_train_selection_v1(
    const char *dataset_identity_sha256,
    const char *official_split_sha256,
    const char *subset_sha256,
    uint64_t subset_count,
    char identity_sha256[VAE_C_SHA256_HEX_SIZE]
)
{
    return identity_build_mini224_selection_v1(
        "mini224_train_selection_identity_v1",
        "train_selection",
        dataset_identity_sha256,
        official_split_sha256,
        subset_sha256,
        subset_count,
        identity_sha256
    );
}


VAECIdentityStatus vae_c_identity_build_mini224_validation_selection_v1(
    const char *dataset_identity_sha256,
    const char *official_split_sha256,
    const char *subset_sha256,
    uint64_t subset_count,
    char identity_sha256[VAE_C_SHA256_HEX_SIZE]
)
{
    return identity_build_mini224_selection_v1(
        "mini224_validation_selection_identity_v1",
        "validation_selection",
        dataset_identity_sha256,
        official_split_sha256,
        subset_sha256,
        subset_count,
        identity_sha256
    );
}


const char *vae_c_identity_status_name(VAECIdentityStatus status)
{
    switch (status) {
        case VAE_C_IDENTITY_OK:
            return "ok";
        case VAE_C_IDENTITY_INVALID_ARGUMENT:
            return "invalid_argument";
        case VAE_C_IDENTITY_INVALID_COMPONENT:
            return "invalid_component";
        case VAE_C_IDENTITY_OVERFLOW:
            return "overflow";
        case VAE_C_IDENTITY_HASH_FAILED:
            return "hash_failed";
        default:
            return "unknown";
    }
}
