#include "vae_c_checkpoint.h"
#include "vae_c_sha256.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif


enum {
    CHECKPOINT_HEADER_SIZE = 72,
    CHECKPOINT_DIGEST_SIZE = 32,
    CHECKPOINT_FORMAT_FLAGS = 3,
    CHECKPOINT_MAX_BATCH_NORMS =
        VAE_C_MAX_CONV_DEPTH + 1 + VAE_C_MAX_DECODER_HIDDEN_CONVS
};


typedef struct {
    unsigned char *data;
    size_t size;
    size_t capacity;
} CheckpointBuffer;


typedef struct {
    const unsigned char *data;
    size_t size;
    size_t position;
} CheckpointReader;


typedef struct {
    uint64_t identity;
    const VAECBatchNorm *layer;
} CheckpointBatchNormConst;


typedef struct {
    uint64_t identity;
    VAECBatchNorm *layer;
} CheckpointBatchNormMutable;


typedef struct {
    VAECCheckpointMetadata metadata;
    uint64_t initialization_seed;
    VAECAdam optimizer;

    float *parameter_values;
    float *parameter_m;
    float *parameter_v;
    size_t parameter_offsets[VAE_C_MODEL_MAX_PARAMETERS];
    size_t parameter_value_count;

    float *batch_norm_mean;
    float *batch_norm_variance;
    size_t batch_norm_offsets[CHECKPOINT_MAX_BATCH_NORMS];
    size_t batch_norm_value_count;
} CheckpointImage;


static const unsigned char checkpoint_magic[8] = {
    'V', 'A', 'E', 'C', 'C', 'K', 'P', 'T'
};


static int checkpoint_size_add(size_t left, size_t right, size_t *result)
{
    if (result == NULL || left > SIZE_MAX - right) {
        return 0;
    }
    *result = left + right;
    return 1;
}


static int checkpoint_size_multiply(
    size_t left,
    size_t right,
    size_t *result
)
{
    if (result == NULL || (left != 0 && right > SIZE_MAX / left)) {
        return 0;
    }
    *result = left * right;
    return 1;
}


static int checkpoint_ieee754_supported(void)
{
    return
        sizeof(float) == 4 &&
        sizeof(double) == 8 &&
        FLT_RADIX == 2 &&
        FLT_MANT_DIG == 24 &&
        DBL_MANT_DIG == 53;
}


static void checkpoint_buffer_init(CheckpointBuffer *buffer)
{
    if (buffer != NULL) {
        memset(buffer, 0, sizeof(*buffer));
    }
}


static void checkpoint_buffer_free(CheckpointBuffer *buffer)
{
    if (buffer != NULL) {
        free(buffer->data);
        checkpoint_buffer_init(buffer);
    }
}


static VAECCheckpointStatus checkpoint_buffer_reserve(
    CheckpointBuffer *buffer,
    size_t additional
)
{
    size_t required;
    size_t capacity;
    unsigned char *candidate;

    if (buffer == NULL) {
        return VAE_C_CHECKPOINT_INVALID_ARGUMENT;
    }
    if (!checkpoint_size_add(buffer->size, additional, &required)) {
        return VAE_C_CHECKPOINT_OVERFLOW;
    }
    if (required <= buffer->capacity) {
        return VAE_C_CHECKPOINT_OK;
    }
    capacity = buffer->capacity == 0 ? 256u : buffer->capacity;
    while (capacity < required) {
        if (capacity > SIZE_MAX / 2u) {
            capacity = required;
            break;
        }
        capacity *= 2u;
    }
    candidate = (unsigned char *)realloc(buffer->data, capacity);
    if (candidate == NULL) {
        return VAE_C_CHECKPOINT_ALLOCATION_FAILED;
    }
    buffer->data = candidate;
    buffer->capacity = capacity;
    return VAE_C_CHECKPOINT_OK;
}


static VAECCheckpointStatus checkpoint_buffer_append(
    CheckpointBuffer *buffer,
    const void *data,
    size_t size
)
{
    VAECCheckpointStatus status;

    if (buffer == NULL || (size != 0 && data == NULL)) {
        return VAE_C_CHECKPOINT_INVALID_ARGUMENT;
    }
    status = checkpoint_buffer_reserve(buffer, size);
    if (status != VAE_C_CHECKPOINT_OK) {
        return status;
    }
    if (size != 0) {
        memcpy(buffer->data + buffer->size, data, size);
        buffer->size += size;
    }
    return VAE_C_CHECKPOINT_OK;
}


static VAECCheckpointStatus checkpoint_append_u8(
    CheckpointBuffer *buffer,
    uint8_t value
)
{
    return checkpoint_buffer_append(buffer, &value, 1u);
}


static VAECCheckpointStatus checkpoint_append_u32(
    CheckpointBuffer *buffer,
    uint32_t value
)
{
    unsigned char bytes[4];

    bytes[0] = (unsigned char)(value & UINT32_C(0xff));
    bytes[1] = (unsigned char)((value >> 8) & UINT32_C(0xff));
    bytes[2] = (unsigned char)((value >> 16) & UINT32_C(0xff));
    bytes[3] = (unsigned char)((value >> 24) & UINT32_C(0xff));
    return checkpoint_buffer_append(buffer, bytes, sizeof(bytes));
}


static VAECCheckpointStatus checkpoint_append_u64(
    CheckpointBuffer *buffer,
    uint64_t value
)
{
    unsigned char bytes[8];
    size_t index;

    for (index = 0; index < sizeof(bytes); index++) {
        bytes[index] = (unsigned char)(value >> (8u * index));
    }
    return checkpoint_buffer_append(buffer, bytes, sizeof(bytes));
}


static VAECCheckpointStatus checkpoint_append_f32(
    CheckpointBuffer *buffer,
    float value
)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    return checkpoint_append_u32(buffer, bits);
}


static VAECCheckpointStatus checkpoint_append_f64(
    CheckpointBuffer *buffer,
    double value
)
{
    uint64_t bits;

    memcpy(&bits, &value, sizeof(bits));
    return checkpoint_append_u64(buffer, bits);
}


static VAECCheckpointStatus checkpoint_append_zeros(
    CheckpointBuffer *buffer,
    size_t count
)
{
    static const unsigned char zeros[8] = {0};

    while (count != 0) {
        size_t chunk = count < sizeof(zeros) ? count : sizeof(zeros);
        VAECCheckpointStatus status = checkpoint_buffer_append(
            buffer, zeros, chunk
        );

        if (status != VAE_C_CHECKPOINT_OK) {
            return status;
        }
        count -= chunk;
    }
    return VAE_C_CHECKPOINT_OK;
}


static int checkpoint_text_length(
    const char *text,
    size_t capacity,
    size_t *length
)
{
    const char *terminator;

    if (text == NULL || capacity == 0 || length == NULL) {
        return 0;
    }
    terminator = (const char *)memchr(text, '\0', capacity);
    if (terminator == NULL || terminator == text) {
        return 0;
    }
    *length = (size_t)(terminator - text);
    return 1;
}


static int checkpoint_is_hex(const char *text, size_t length)
{
    size_t index;

    for (index = 0; index < length; index++) {
        unsigned char value = (unsigned char)text[index];

        if (!((value >= (unsigned char)'0' && value <= (unsigned char)'9') ||
              (value >= (unsigned char)'a' && value <= (unsigned char)'f') ||
              (value >= (unsigned char)'A' && value <= (unsigned char)'F'))) {

            return 0;
        }
    }
    return 1;
}


static int checkpoint_validate_hex_text(
    const char *text,
    size_t capacity,
    size_t expected_length
)
{
    size_t length;

    return checkpoint_text_length(text, capacity, &length) &&
        length == expected_length && checkpoint_is_hex(text, length);
}


static VAECCheckpointStatus checkpoint_append_text(
    CheckpointBuffer *buffer,
    const char *text,
    size_t capacity
)
{
    size_t length;
    VAECCheckpointStatus status;

    if (!checkpoint_text_length(text, capacity, &length) ||
        length > UINT32_MAX) {

        return VAE_C_CHECKPOINT_INVALID_ARGUMENT;
    }
    status = checkpoint_append_u32(buffer, (uint32_t)length);
    return status == VAE_C_CHECKPOINT_OK
        ? checkpoint_buffer_append(buffer, text, length)
        : status;
}


static int checkpoint_reader_take(
    CheckpointReader *reader,
    size_t count,
    const unsigned char **bytes
)
{
    size_t end;

    if (reader == NULL || bytes == NULL ||
        !checkpoint_size_add(reader->position, count, &end) ||
        end > reader->size) {

        return 0;
    }
    *bytes = reader->data + reader->position;
    reader->position = end;
    return 1;
}


static int checkpoint_read_u8(CheckpointReader *reader, uint8_t *value)
{
    const unsigned char *bytes;

    if (value == NULL || !checkpoint_reader_take(reader, 1u, &bytes)) {
        return 0;
    }
    *value = bytes[0];
    return 1;
}


static int checkpoint_read_u32(CheckpointReader *reader, uint32_t *value)
{
    const unsigned char *bytes;

    if (value == NULL || !checkpoint_reader_take(reader, 4u, &bytes)) {
        return 0;
    }
    *value =
        (uint32_t)bytes[0] |
        ((uint32_t)bytes[1] << 8) |
        ((uint32_t)bytes[2] << 16) |
        ((uint32_t)bytes[3] << 24);
    return 1;
}


static int checkpoint_read_u64(CheckpointReader *reader, uint64_t *value)
{
    const unsigned char *bytes;
    uint64_t result = UINT64_C(0);
    size_t index;

    if (value == NULL || !checkpoint_reader_take(reader, 8u, &bytes)) {
        return 0;
    }
    for (index = 0; index < 8u; index++) {
        result |= (uint64_t)bytes[index] << (8u * index);
    }
    *value = result;
    return 1;
}


static int checkpoint_read_f32(CheckpointReader *reader, float *value)
{
    uint32_t bits;

    if (value == NULL || !checkpoint_read_u32(reader, &bits)) {
        return 0;
    }
    memcpy(value, &bits, sizeof(bits));
    return 1;
}


static int checkpoint_read_f64(CheckpointReader *reader, double *value)
{
    uint64_t bits;

    if (value == NULL || !checkpoint_read_u64(reader, &bits)) {
        return 0;
    }
    memcpy(value, &bits, sizeof(bits));
    return 1;
}


static int checkpoint_read_zero_bytes(
    CheckpointReader *reader,
    size_t count
)
{
    const unsigned char *bytes;
    size_t index;

    if (!checkpoint_reader_take(reader, count, &bytes)) {
        return 0;
    }
    for (index = 0; index < count; index++) {
        if (bytes[index] != 0u) {
            return 0;
        }
    }
    return 1;
}


static int checkpoint_read_text(
    CheckpointReader *reader,
    char *text,
    size_t capacity
)
{
    uint32_t encoded_length;
    const unsigned char *bytes;
    size_t length;

    if (text == NULL || capacity == 0 ||
        !checkpoint_read_u32(reader, &encoded_length)) {

        return 0;
    }
    length = (size_t)encoded_length;
    if ((uint32_t)length != encoded_length || length == 0 ||
        length >= capacity ||
        !checkpoint_reader_take(reader, length, &bytes) ||
        memchr(bytes, '\0', length) != NULL) {

        return 0;
    }
    memset(text, 0, capacity);
    memcpy(text, bytes, length);
    return 1;
}


#ifdef VAE_C_CHECKPOINT_TEST_HOOKS
int vae_c_checkpoint_sha256_test_only(
    const unsigned char *data,
    size_t size,
    unsigned char digest[CHECKPOINT_DIGEST_SIZE]
)
{
    return vae_c_sha256_digest(data, size, digest) == VAE_C_SHA256_OK;
}
#endif


void vae_c_checkpoint_run_identity_init(
    VAECCheckpointRunIdentity *identity
)
{
    if (identity != NULL) {
        memset(identity, 0, sizeof(*identity));
    }
}


void vae_c_checkpoint_metadata_init(VAECCheckpointMetadata *metadata)
{
    if (metadata != NULL) {
        memset(metadata, 0, sizeof(*metadata));
    }
}


static int checkpoint_identity_is_valid(
    const VAECCheckpointRunIdentity *identity
)
{
    size_t unused;

    if (identity == NULL ||
        !checkpoint_text_length(
            identity->profile_id, sizeof(identity->profile_id), &unused) ||
        !checkpoint_validate_hex_text(
            identity->profile_config_sha256,
            sizeof(identity->profile_config_sha256), 64u) ||
        !checkpoint_validate_hex_text(
            identity->code_commit,
            sizeof(identity->code_commit), 40u) ||
        !checkpoint_text_length(
            identity->dataset_id, sizeof(identity->dataset_id), &unused) ||
        !checkpoint_text_length(
            identity->event_manifest_id,
            sizeof(identity->event_manifest_id), &unused) ||
        !checkpoint_validate_hex_text(
            identity->event_manifest_sha256,
            sizeof(identity->event_manifest_sha256), 64u) ||
        !checkpoint_text_length(
            identity->tensor_manifest_id,
            sizeof(identity->tensor_manifest_id), &unused) ||
        !checkpoint_validate_hex_text(
            identity->tensor_manifest_sha256,
            sizeof(identity->tensor_manifest_sha256), 64u) ||
        !checkpoint_text_length(
            identity->train_split_id,
            sizeof(identity->train_split_id), &unused) ||
        !checkpoint_validate_hex_text(
            identity->train_split_sha256,
            sizeof(identity->train_split_sha256), 64u) ||
        !checkpoint_text_length(
            identity->validation_split_id,
            sizeof(identity->validation_split_id), &unused) ||
        !checkpoint_validate_hex_text(
            identity->validation_split_sha256,
            sizeof(identity->validation_split_sha256), 64u) ||
        !checkpoint_text_length(
            identity->augmentation_policy,
            sizeof(identity->augmentation_policy), &unused) ||
        !checkpoint_text_length(
            identity->shuffle_policy,
            sizeof(identity->shuffle_policy), &unused) ||
        !checkpoint_text_length(
            identity->latent_policy,
            sizeof(identity->latent_policy), &unused) ||
        !checkpoint_text_length(
            identity->validation_policy,
            sizeof(identity->validation_policy), &unused) ||
        !checkpoint_text_length(
            identity->compiler_id,
            sizeof(identity->compiler_id), &unused) ||
        !checkpoint_text_length(
            identity->compiler_version,
            sizeof(identity->compiler_version), &unused) ||
        !checkpoint_text_length(
            identity->compiler_target,
            sizeof(identity->compiler_target), &unused) ||
        identity->max_batch_size == UINT64_C(0) ||
        identity->batch_size == UINT64_C(0) ||
        identity->batch_size > identity->max_batch_size ||
        !isfinite(identity->optimizer_learning_rate) ||
        identity->optimizer_learning_rate <= 0.0 ||
        !isfinite(identity->optimizer_beta1) ||
        identity->optimizer_beta1 <= 0.0 ||
        identity->optimizer_beta1 >= 1.0 ||
        !isfinite(identity->optimizer_beta2) ||
        identity->optimizer_beta2 <= 0.0 ||
        identity->optimizer_beta2 >= 1.0 ||
        !isfinite(identity->optimizer_epsilon) ||
        identity->optimizer_epsilon <= 0.0) {

        return 0;
    }
    return 1;
}


static int checkpoint_kind_is_valid(VAECCheckpointKind kind)
{
    return
        kind == VAE_C_CHECKPOINT_KIND_LATEST ||
        kind == VAE_C_CHECKPOINT_KIND_BEST ||
        kind == VAE_C_CHECKPOINT_KIND_FINAL;
}


static int checkpoint_metadata_is_valid(
    const VAECCheckpointMetadata *metadata
)
{
    if (metadata == NULL ||
        !checkpoint_identity_is_valid(&metadata->run_identity) ||
        !checkpoint_kind_is_valid(metadata->kind) ||
        metadata->last_completed_epoch == UINT64_MAX) {

        return 0;
    }
    if (!metadata->best_defined) {
        return metadata->best_epoch == UINT64_C(0) &&
            metadata->best_validation_total_mean_per_sample == 0.0 &&
            metadata->no_improvement_count == UINT64_C(0);
    }
    return
        metadata->best_epoch <= metadata->last_completed_epoch &&
        isfinite(metadata->best_validation_total_mean_per_sample) &&
        metadata->best_validation_total_mean_per_sample >= 0.0;
}


static int checkpoint_identity_equal(
    const VAECCheckpointRunIdentity *left,
    const VAECCheckpointRunIdentity *right,
    int compare_provenance
)
{
    if (left == NULL || right == NULL) {
        return 0;
    }
    return
        strcmp(left->profile_id, right->profile_id) == 0 &&
        strcmp(left->profile_config_sha256,
               right->profile_config_sha256) == 0 &&
        strcmp(left->code_commit, right->code_commit) == 0 &&
        strcmp(left->dataset_id, right->dataset_id) == 0 &&
        strcmp(left->event_manifest_id, right->event_manifest_id) == 0 &&
        strcmp(left->event_manifest_sha256,
               right->event_manifest_sha256) == 0 &&
        strcmp(left->tensor_manifest_id,
               right->tensor_manifest_id) == 0 &&
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
        (!compare_provenance ||
         (strcmp(left->compiler_id, right->compiler_id) == 0 &&
          strcmp(left->compiler_version, right->compiler_version) == 0 &&
          strcmp(left->compiler_target, right->compiler_target) == 0)) &&
        left->initialization_seed == right->initialization_seed &&
        left->latent_seed == right->latent_seed &&
        left->augmentation_seed == right->augmentation_seed &&
        left->shuffle_seed == right->shuffle_seed &&
        left->max_batch_size == right->max_batch_size &&
        left->batch_size == right->batch_size &&
        left->optimizer_learning_rate ==
            right->optimizer_learning_rate &&
        left->optimizer_beta1 == right->optimizer_beta1 &&
        left->optimizer_beta2 == right->optimizer_beta2 &&
        left->optimizer_epsilon == right->optimizer_epsilon;
}


static int checkpoint_metadata_equal(
    const VAECCheckpointMetadata *left,
    const VAECCheckpointMetadata *right
)
{
    return left != NULL && right != NULL &&
        checkpoint_identity_equal(
            &left->run_identity, &right->run_identity, 1) &&
        left->kind == right->kind &&
        left->last_completed_epoch == right->last_completed_epoch &&
        left->best_defined == right->best_defined &&
        left->best_epoch == right->best_epoch &&
        left->best_validation_total_mean_per_sample ==
            right->best_validation_total_mean_per_sample &&
        left->no_improvement_count == right->no_improvement_count;
}


static int checkpoint_optimizer_config_matches_identity(
    const VAECAdam *optimizer,
    const VAECCheckpointRunIdentity *identity
)
{
    return optimizer != NULL && identity != NULL &&
        optimizer->learning_rate == identity->optimizer_learning_rate &&
        optimizer->beta1 == identity->optimizer_beta1 &&
        optimizer->beta2 == identity->optimizer_beta2 &&
        optimizer->adam_epsilon == identity->optimizer_epsilon;
}


static int checkpoint_model_matches_identity(
    const VAECModel *model,
    const VAECCheckpointRunIdentity *identity
)
{
    return model != NULL && identity != NULL &&
        strcmp(model->profile_id, identity->profile_id) == 0 &&
        model->initialization_seed == identity->initialization_seed &&
        (uint64_t)model->max_batch_size == identity->max_batch_size &&
        (size_t)identity->max_batch_size == model->max_batch_size;
}


static size_t checkpoint_collect_batch_norms_const(
    const VAECModel *model,
    CheckpointBatchNormConst layers[CHECKPOINT_MAX_BATCH_NORMS]
)
{
    size_t count = 0;
    size_t index;

    for (index = 0; index < model->config.conv_depth; index++) {
        layers[count].identity = UINT64_C(1) + (uint64_t)index;
        layers[count++].layer = &model->encoder_batch_norms[index];
    }
    layers[count].identity = UINT64_C(5);
    layers[count++].layer = &model->decoder_fc_batch_norm;
    for (index = 0;
         index < model->config.decoder_hidden_conv_count;
         index++) {

        layers[count].identity = UINT64_C(6) + (uint64_t)index;
        layers[count++].layer =
            &model->decoder_hidden_batch_norms[index];
    }
    return count;
}


static size_t checkpoint_collect_batch_norms_mutable(
    VAECModel *model,
    CheckpointBatchNormMutable layers[CHECKPOINT_MAX_BATCH_NORMS]
)
{
    size_t count = 0;
    size_t index;

    for (index = 0; index < model->config.conv_depth; index++) {
        layers[count].identity = UINT64_C(1) + (uint64_t)index;
        layers[count++].layer = &model->encoder_batch_norms[index];
    }
    layers[count].identity = UINT64_C(5);
    layers[count++].layer = &model->decoder_fc_batch_norm;
    for (index = 0;
         index < model->config.decoder_hidden_conv_count;
         index++) {

        layers[count].identity = UINT64_C(6) + (uint64_t)index;
        layers[count++].layer =
            &model->decoder_hidden_batch_norms[index];
    }
    return count;
}


static VAECCheckpointStatus checkpoint_encode_identity(
    CheckpointBuffer *buffer,
    const VAECCheckpointRunIdentity *identity
)
{
    VAECCheckpointStatus status;

#define CHECKPOINT_APPEND_ID_TEXT(field)                                      \
    do {                                                                       \
        status = checkpoint_append_text(                                      \
            buffer, identity->field, sizeof(identity->field));                \
        if (status != VAE_C_CHECKPOINT_OK) {                                  \
            return status;                                                     \
        }                                                                      \
    } while (0)

    CHECKPOINT_APPEND_ID_TEXT(profile_id);
    CHECKPOINT_APPEND_ID_TEXT(profile_config_sha256);
    CHECKPOINT_APPEND_ID_TEXT(code_commit);
    CHECKPOINT_APPEND_ID_TEXT(dataset_id);
    CHECKPOINT_APPEND_ID_TEXT(event_manifest_id);
    CHECKPOINT_APPEND_ID_TEXT(event_manifest_sha256);
    CHECKPOINT_APPEND_ID_TEXT(tensor_manifest_id);
    CHECKPOINT_APPEND_ID_TEXT(tensor_manifest_sha256);
    CHECKPOINT_APPEND_ID_TEXT(train_split_id);
    CHECKPOINT_APPEND_ID_TEXT(train_split_sha256);
    CHECKPOINT_APPEND_ID_TEXT(validation_split_id);
    CHECKPOINT_APPEND_ID_TEXT(validation_split_sha256);
    CHECKPOINT_APPEND_ID_TEXT(augmentation_policy);
    CHECKPOINT_APPEND_ID_TEXT(shuffle_policy);
    CHECKPOINT_APPEND_ID_TEXT(latent_policy);
    CHECKPOINT_APPEND_ID_TEXT(validation_policy);
    CHECKPOINT_APPEND_ID_TEXT(compiler_id);
    CHECKPOINT_APPEND_ID_TEXT(compiler_version);
    CHECKPOINT_APPEND_ID_TEXT(compiler_target);
#undef CHECKPOINT_APPEND_ID_TEXT

#define CHECKPOINT_APPEND_ID_U64(field)                                      \
    do {                                                                       \
        status = checkpoint_append_u64(buffer, identity->field);               \
        if (status != VAE_C_CHECKPOINT_OK) {                                  \
            return status;                                                     \
        }                                                                      \
    } while (0)

    CHECKPOINT_APPEND_ID_U64(initialization_seed);
    CHECKPOINT_APPEND_ID_U64(latent_seed);
    CHECKPOINT_APPEND_ID_U64(augmentation_seed);
    CHECKPOINT_APPEND_ID_U64(shuffle_seed);
    CHECKPOINT_APPEND_ID_U64(max_batch_size);
    CHECKPOINT_APPEND_ID_U64(batch_size);
#undef CHECKPOINT_APPEND_ID_U64

    status = checkpoint_append_f64(
        buffer, identity->optimizer_learning_rate
    );
    if (status == VAE_C_CHECKPOINT_OK) {
        status = checkpoint_append_f64(buffer, identity->optimizer_beta1);
    }
    if (status == VAE_C_CHECKPOINT_OK) {
        status = checkpoint_append_f64(buffer, identity->optimizer_beta2);
    }
    if (status == VAE_C_CHECKPOINT_OK) {
        status = checkpoint_append_f64(buffer, identity->optimizer_epsilon);
    }
    return status;
}


static VAECCheckpointStatus checkpoint_decode_identity(
    CheckpointReader *reader,
    VAECCheckpointRunIdentity *identity
)
{
    if (reader == NULL || identity == NULL) {
        return VAE_C_CHECKPOINT_INVALID_ARGUMENT;
    }
    vae_c_checkpoint_run_identity_init(identity);

#define CHECKPOINT_READ_ID_TEXT(field)                                        \
    do {                                                                       \
        if (!checkpoint_read_text(                                            \
                reader, identity->field, sizeof(identity->field))) {          \
            return VAE_C_CHECKPOINT_FORMAT_ERROR;                             \
        }                                                                      \
    } while (0)

    CHECKPOINT_READ_ID_TEXT(profile_id);
    CHECKPOINT_READ_ID_TEXT(profile_config_sha256);
    CHECKPOINT_READ_ID_TEXT(code_commit);
    CHECKPOINT_READ_ID_TEXT(dataset_id);
    CHECKPOINT_READ_ID_TEXT(event_manifest_id);
    CHECKPOINT_READ_ID_TEXT(event_manifest_sha256);
    CHECKPOINT_READ_ID_TEXT(tensor_manifest_id);
    CHECKPOINT_READ_ID_TEXT(tensor_manifest_sha256);
    CHECKPOINT_READ_ID_TEXT(train_split_id);
    CHECKPOINT_READ_ID_TEXT(train_split_sha256);
    CHECKPOINT_READ_ID_TEXT(validation_split_id);
    CHECKPOINT_READ_ID_TEXT(validation_split_sha256);
    CHECKPOINT_READ_ID_TEXT(augmentation_policy);
    CHECKPOINT_READ_ID_TEXT(shuffle_policy);
    CHECKPOINT_READ_ID_TEXT(latent_policy);
    CHECKPOINT_READ_ID_TEXT(validation_policy);
    CHECKPOINT_READ_ID_TEXT(compiler_id);
    CHECKPOINT_READ_ID_TEXT(compiler_version);
    CHECKPOINT_READ_ID_TEXT(compiler_target);
#undef CHECKPOINT_READ_ID_TEXT

#define CHECKPOINT_READ_ID_U64(field)                                         \
    do {                                                                       \
        if (!checkpoint_read_u64(reader, &identity->field)) {                 \
            return VAE_C_CHECKPOINT_FORMAT_ERROR;                             \
        }                                                                      \
    } while (0)

    CHECKPOINT_READ_ID_U64(initialization_seed);
    CHECKPOINT_READ_ID_U64(latent_seed);
    CHECKPOINT_READ_ID_U64(augmentation_seed);
    CHECKPOINT_READ_ID_U64(shuffle_seed);
    CHECKPOINT_READ_ID_U64(max_batch_size);
    CHECKPOINT_READ_ID_U64(batch_size);
#undef CHECKPOINT_READ_ID_U64

    if (!checkpoint_read_f64(reader, &identity->optimizer_learning_rate) ||
        !checkpoint_read_f64(reader, &identity->optimizer_beta1) ||
        !checkpoint_read_f64(reader, &identity->optimizer_beta2) ||
        !checkpoint_read_f64(reader, &identity->optimizer_epsilon) ||
        reader->position != reader->size ||
        !checkpoint_identity_is_valid(identity)) {

        return VAE_C_CHECKPOINT_FORMAT_ERROR;
    }
    return VAE_C_CHECKPOINT_OK;
}


static VAECCheckpointStatus checkpoint_encode_config(
    CheckpointBuffer *buffer,
    const VAECConfig *config
)
{
    VAECCheckpointStatus status;
    size_t index;

#define CHECKPOINT_APPEND_CONFIG_SIZE(field)                                  \
    do {                                                                       \
        status = checkpoint_append_u64(buffer, (uint64_t)config->field);       \
        if (status != VAE_C_CHECKPOINT_OK ||                                  \
            (size_t)(uint64_t)config->field != config->field) {               \
            return status == VAE_C_CHECKPOINT_OK                              \
                ? VAE_C_CHECKPOINT_OVERFLOW : status;                         \
        }                                                                      \
    } while (0)

    CHECKPOINT_APPEND_CONFIG_SIZE(input_channels);
    CHECKPOINT_APPEND_CONFIG_SIZE(input_side);
    CHECKPOINT_APPEND_CONFIG_SIZE(conv_depth);
    for (index = 0; index < VAE_C_MAX_CONV_DEPTH; index++) {
        status = checkpoint_append_u64(buffer, (uint64_t)config->channels[index]);
        if (status != VAE_C_CHECKPOINT_OK ||
            (size_t)(uint64_t)config->channels[index] != config->channels[index]) {
            return status == VAE_C_CHECKPOINT_OK
                ? VAE_C_CHECKPOINT_OVERFLOW : status;
        }
    }
    for (index = 0; index < VAE_C_MAX_CONV_DEPTH; index++) {
        status = checkpoint_append_u64(
            buffer, (uint64_t)config->encoder_conv_sides[index]
        );
        if (status != VAE_C_CHECKPOINT_OK ||
            (size_t)(uint64_t)config->encoder_conv_sides[index] !=
                config->encoder_conv_sides[index]) {
            return status == VAE_C_CHECKPOINT_OK
                ? VAE_C_CHECKPOINT_OVERFLOW : status;
        }
    }
    CHECKPOINT_APPEND_CONFIG_SIZE(pool_side);
    CHECKPOINT_APPEND_CONFIG_SIZE(latent_dim);
    CHECKPOINT_APPEND_CONFIG_SIZE(bottleneck_side);
    CHECKPOINT_APPEND_CONFIG_SIZE(decoder_fc_features);
    CHECKPOINT_APPEND_CONFIG_SIZE(decoder_upsample_count);
    for (index = 0; index < VAE_C_MAX_DECODER_UPSAMPLES; index++) {
        status = checkpoint_append_u64(
            buffer, (uint64_t)config->decoder_upsample_targets[index]
        );
        if (status != VAE_C_CHECKPOINT_OK ||
            (size_t)(uint64_t)config->decoder_upsample_targets[index] !=
                config->decoder_upsample_targets[index]) {
            return status == VAE_C_CHECKPOINT_OK
                ? VAE_C_CHECKPOINT_OVERFLOW : status;
        }
    }
    CHECKPOINT_APPEND_CONFIG_SIZE(decoder_hidden_conv_count);
    for (index = 0; index < VAE_C_MAX_DECODER_HIDDEN_CONVS; index++) {
        status = checkpoint_append_u64(
            buffer, (uint64_t)config->decoder_hidden_conv_sides[index]
        );
        if (status != VAE_C_CHECKPOINT_OK ||
            (size_t)(uint64_t)config->decoder_hidden_conv_sides[index] !=
                config->decoder_hidden_conv_sides[index]) {
            return status == VAE_C_CHECKPOINT_OK
                ? VAE_C_CHECKPOINT_OVERFLOW : status;
        }
    }
#undef CHECKPOINT_APPEND_CONFIG_SIZE

    status = checkpoint_append_u64(buffer, config->values_per_sample);
    if (status == VAE_C_CHECKPOINT_OK) {
        status = checkpoint_append_u64(
            buffer, config->trainable_parameters_bn1d
        );
    }
    if (status == VAE_C_CHECKPOINT_OK) {
        status = checkpoint_append_u64(
            buffer, config->forward_mac_per_sample
        );
    }
    return status;
}


static int checkpoint_read_size_matches(
    CheckpointReader *reader,
    size_t expected
)
{
    uint64_t encoded;

    return checkpoint_read_u64(reader, &encoded) &&
        encoded == (uint64_t)expected && (size_t)encoded == expected;
}


static VAECCheckpointStatus checkpoint_decode_config_matches(
    CheckpointReader *reader,
    const VAECConfig *config
)
{
    uint64_t value;
    size_t index;

    if (!checkpoint_read_size_matches(reader, config->input_channels) ||
        !checkpoint_read_size_matches(reader, config->input_side) ||
        !checkpoint_read_size_matches(reader, config->conv_depth)) {
        return VAE_C_CHECKPOINT_INCOMPATIBLE;
    }
    for (index = 0; index < VAE_C_MAX_CONV_DEPTH; index++) {
        if (!checkpoint_read_size_matches(reader, config->channels[index])) {
            return VAE_C_CHECKPOINT_INCOMPATIBLE;
        }
    }
    for (index = 0; index < VAE_C_MAX_CONV_DEPTH; index++) {
        if (!checkpoint_read_size_matches(
                reader, config->encoder_conv_sides[index])) {
            return VAE_C_CHECKPOINT_INCOMPATIBLE;
        }
    }
    if (!checkpoint_read_size_matches(reader, config->pool_side) ||
        !checkpoint_read_size_matches(reader, config->latent_dim) ||
        !checkpoint_read_size_matches(reader, config->bottleneck_side) ||
        !checkpoint_read_size_matches(reader, config->decoder_fc_features) ||
        !checkpoint_read_size_matches(
            reader, config->decoder_upsample_count)) {
        return VAE_C_CHECKPOINT_INCOMPATIBLE;
    }
    for (index = 0; index < VAE_C_MAX_DECODER_UPSAMPLES; index++) {
        if (!checkpoint_read_size_matches(
                reader, config->decoder_upsample_targets[index])) {
            return VAE_C_CHECKPOINT_INCOMPATIBLE;
        }
    }
    if (!checkpoint_read_size_matches(
            reader, config->decoder_hidden_conv_count)) {
        return VAE_C_CHECKPOINT_INCOMPATIBLE;
    }
    for (index = 0; index < VAE_C_MAX_DECODER_HIDDEN_CONVS; index++) {
        if (!checkpoint_read_size_matches(
                reader, config->decoder_hidden_conv_sides[index])) {
            return VAE_C_CHECKPOINT_INCOMPATIBLE;
        }
    }
    if (!checkpoint_read_u64(reader, &value) ||
        value != config->values_per_sample ||
        !checkpoint_read_u64(reader, &value) ||
        value != config->trainable_parameters_bn1d ||
        !checkpoint_read_u64(reader, &value) ||
        value != config->forward_mac_per_sample) {

        return VAE_C_CHECKPOINT_INCOMPATIBLE;
    }
    return VAE_C_CHECKPOINT_OK;
}


static VAECCheckpointStatus checkpoint_encode_model(
    CheckpointBuffer *buffer,
    const VAECModel *model,
    size_t batch_norm_count
)
{
    VAECCheckpointStatus status;

    status = checkpoint_append_u64(buffer, model->initialization_seed);
    if (status == VAE_C_CHECKPOINT_OK) {
        status = checkpoint_append_u64(buffer, (uint64_t)model->max_batch_size);
    }
    if (status == VAE_C_CHECKPOINT_OK) {
        status = checkpoint_append_u64(buffer, (uint64_t)model->parameter_count);
    }
    if (status == VAE_C_CHECKPOINT_OK) {
        status = checkpoint_append_u64(buffer, model->trainable_value_count);
    }
    if (status == VAE_C_CHECKPOINT_OK) {
        status = checkpoint_append_u64(buffer, (uint64_t)batch_norm_count);
    }
    if (status == VAE_C_CHECKPOINT_OK) {
        status = checkpoint_encode_config(buffer, &model->config);
    }
    return status;
}


static VAECCheckpointStatus checkpoint_encode_parameters(
    CheckpointBuffer *buffer,
    const VAECModel *model
)
{
    VAECCheckpointStatus status;
    size_t parameter_index;

    status = checkpoint_append_u64(buffer, (uint64_t)model->parameter_count);
    if (status != VAE_C_CHECKPOINT_OK) {
        return status;
    }
    for (parameter_index = 0;
         parameter_index < model->parameter_count;
         parameter_index++) {

        const VAECParameter *parameter = model->parameters[parameter_index];
        size_t index;

        status = checkpoint_append_u64(buffer, parameter->parameter_key);
        if (status == VAE_C_CHECKPOINT_OK) {
            status = checkpoint_append_u32(
                buffer, (uint32_t)parameter->shape.rank
            );
        }
        if (status == VAE_C_CHECKPOINT_OK) {
            status = checkpoint_append_u32(buffer, UINT32_C(0));
        }
        for (index = 0;
             status == VAE_C_CHECKPOINT_OK && index < 4u;
             index++) {
            status = checkpoint_append_u64(
                buffer, (uint64_t)parameter->shape.dims[index]
            );
        }
        if (status == VAE_C_CHECKPOINT_OK) {
            status = checkpoint_append_u64(
                buffer, (uint64_t)parameter->shape.count
            );
        }
        for (index = 0;
             status == VAE_C_CHECKPOINT_OK &&
                 index < parameter->shape.count;
             index++) {
            status = checkpoint_append_f32(buffer, parameter->values[index]);
        }
        for (index = 0;
             status == VAE_C_CHECKPOINT_OK &&
                 index < parameter->shape.count;
             index++) {
            status = checkpoint_append_f32(buffer, parameter->adam_m[index]);
        }
        for (index = 0;
             status == VAE_C_CHECKPOINT_OK &&
                 index < parameter->shape.count;
             index++) {
            status = checkpoint_append_f32(buffer, parameter->adam_v[index]);
        }
        if (status != VAE_C_CHECKPOINT_OK) {
            return status;
        }
    }
    return VAE_C_CHECKPOINT_OK;
}


static VAECCheckpointStatus checkpoint_encode_batch_norms(
    CheckpointBuffer *buffer,
    const CheckpointBatchNormConst *layers,
    size_t count
)
{
    VAECCheckpointStatus status;
    size_t layer_index;

    status = checkpoint_append_u64(buffer, (uint64_t)count);
    if (status != VAE_C_CHECKPOINT_OK) {
        return status;
    }
    for (layer_index = 0; layer_index < count; layer_index++) {
        const VAECBatchNorm *layer = layers[layer_index].layer;
        size_t feature;

        status = checkpoint_append_u64(buffer, layers[layer_index].identity);
        if (status == VAE_C_CHECKPOINT_OK) {
            status = checkpoint_append_u32(buffer, (uint32_t)layer->kind);
        }
        if (status == VAE_C_CHECKPOINT_OK) {
            status = checkpoint_append_u32(buffer, UINT32_C(0));
        }
        if (status == VAE_C_CHECKPOINT_OK) {
            status = checkpoint_append_u64(
                buffer, (uint64_t)layer->num_features
            );
        }
        for (feature = 0;
             status == VAE_C_CHECKPOINT_OK &&
                 feature < layer->num_features;
             feature++) {
            status = checkpoint_append_f32(
                buffer, layer->running_mean[feature]
            );
        }
        for (feature = 0;
             status == VAE_C_CHECKPOINT_OK &&
                 feature < layer->num_features;
             feature++) {
            status = checkpoint_append_f32(
                buffer, layer->running_variance[feature]
            );
        }
        if (status != VAE_C_CHECKPOINT_OK) {
            return status;
        }
    }
    return VAE_C_CHECKPOINT_OK;
}


static VAECCheckpointStatus checkpoint_encode_adam(
    CheckpointBuffer *buffer,
    const VAECAdam *optimizer
)
{
    VAECCheckpointStatus status;

    status = checkpoint_append_u8(buffer, optimizer->configured ? 1u : 0u);
    if (status == VAE_C_CHECKPOINT_OK) {
        status = checkpoint_append_zeros(buffer, 7u);
    }
    if (status == VAE_C_CHECKPOINT_OK) {
        status = checkpoint_append_f64(buffer, optimizer->learning_rate);
    }
    if (status == VAE_C_CHECKPOINT_OK) {
        status = checkpoint_append_f64(buffer, optimizer->beta1);
    }
    if (status == VAE_C_CHECKPOINT_OK) {
        status = checkpoint_append_f64(buffer, optimizer->beta2);
    }
    if (status == VAE_C_CHECKPOINT_OK) {
        status = checkpoint_append_f64(buffer, optimizer->adam_epsilon);
    }
    if (status == VAE_C_CHECKPOINT_OK) {
        status = checkpoint_append_u64(buffer, optimizer->step);
    }
    if (status == VAE_C_CHECKPOINT_OK) {
        status = checkpoint_append_f64(buffer, optimizer->beta1_power);
    }
    if (status == VAE_C_CHECKPOINT_OK) {
        status = checkpoint_append_f64(buffer, optimizer->beta2_power);
    }
    return status;
}


static VAECCheckpointStatus checkpoint_encode_controller(
    CheckpointBuffer *buffer,
    const VAECCheckpointMetadata *metadata
)
{
    VAECCheckpointStatus status;

    status = checkpoint_append_u32(buffer, (uint32_t)metadata->kind);
    if (status == VAE_C_CHECKPOINT_OK) {
        status = checkpoint_append_u8(
            buffer, metadata->best_defined ? 1u : 0u
        );
    }
    if (status == VAE_C_CHECKPOINT_OK) {
        status = checkpoint_append_zeros(buffer, 3u);
    }
    if (status == VAE_C_CHECKPOINT_OK) {
        status = checkpoint_append_u64(
            buffer, metadata->last_completed_epoch
        );
    }
    if (status == VAE_C_CHECKPOINT_OK) {
        status = checkpoint_append_u64(buffer, metadata->best_epoch);
    }
    if (status == VAE_C_CHECKPOINT_OK) {
        status = checkpoint_append_f64(
            buffer, metadata->best_validation_total_mean_per_sample
        );
    }
    if (status == VAE_C_CHECKPOINT_OK) {
        status = checkpoint_append_u64(
            buffer, metadata->no_improvement_count
        );
    }
    return status;
}


static void checkpoint_image_init(CheckpointImage *image)
{
    if (image != NULL) {
        memset(image, 0, sizeof(*image));
    }
}


static void checkpoint_image_free(CheckpointImage *image)
{
    if (image != NULL) {
        free(image->parameter_values);
        free(image->parameter_m);
        free(image->parameter_v);
        free(image->batch_norm_mean);
        free(image->batch_norm_variance);
        checkpoint_image_init(image);
    }
}


static VAECCheckpointStatus checkpoint_image_allocate(
    CheckpointImage *image,
    const VAECModel *model,
    const CheckpointBatchNormConst *batch_norms,
    size_t batch_norm_count
)
{
    size_t parameter_count;
    size_t parameter_bytes;
    size_t batch_norm_values = 0;
    size_t batch_norm_bytes;
    size_t index;

    if (image == NULL || model == NULL || batch_norms == NULL ||
        model->trainable_value_count > SIZE_MAX) {
        return VAE_C_CHECKPOINT_OVERFLOW;
    }
    parameter_count = (size_t)model->trainable_value_count;
    if ((uint64_t)parameter_count != model->trainable_value_count ||
        !checkpoint_size_multiply(
            parameter_count, sizeof(float), &parameter_bytes)) {
        return VAE_C_CHECKPOINT_OVERFLOW;
    }
    for (index = 0; index < batch_norm_count; index++) {
        if (!checkpoint_size_add(
                batch_norm_values,
                batch_norms[index].layer->num_features,
                &batch_norm_values)) {
            return VAE_C_CHECKPOINT_OVERFLOW;
        }
    }
    if (!checkpoint_size_multiply(
            batch_norm_values, sizeof(float), &batch_norm_bytes)) {
        return VAE_C_CHECKPOINT_OVERFLOW;
    }
    image->parameter_values = (float *)malloc(parameter_bytes);
    image->parameter_m = (float *)malloc(parameter_bytes);
    image->parameter_v = (float *)malloc(parameter_bytes);
    image->batch_norm_mean = (float *)malloc(batch_norm_bytes);
    image->batch_norm_variance = (float *)malloc(batch_norm_bytes);
    if (image->parameter_values == NULL || image->parameter_m == NULL ||
        image->parameter_v == NULL || image->batch_norm_mean == NULL ||
        image->batch_norm_variance == NULL) {
        checkpoint_image_free(image);
        return VAE_C_CHECKPOINT_ALLOCATION_FAILED;
    }
    image->parameter_value_count = parameter_count;
    image->batch_norm_value_count = batch_norm_values;
    return VAE_C_CHECKPOINT_OK;
}


static VAECCheckpointStatus checkpoint_decode_model_matches(
    CheckpointReader *reader,
    const VAECModel *model,
    size_t expected_batch_norm_count,
    CheckpointImage *image
)
{
    uint64_t max_batch_size;
    uint64_t parameter_count;
    uint64_t trainable_count;
    uint64_t batch_norm_count;
    VAECCheckpointStatus status;

    if (!checkpoint_read_u64(reader, &image->initialization_seed) ||
        !checkpoint_read_u64(reader, &max_batch_size) ||
        !checkpoint_read_u64(reader, &parameter_count) ||
        !checkpoint_read_u64(reader, &trainable_count) ||
        !checkpoint_read_u64(reader, &batch_norm_count)) {
        return VAE_C_CHECKPOINT_FORMAT_ERROR;
    }
    if (max_batch_size != (uint64_t)model->max_batch_size ||
        (size_t)max_batch_size != model->max_batch_size ||
        parameter_count != (uint64_t)model->parameter_count ||
        (size_t)parameter_count != model->parameter_count ||
        trainable_count != model->trainable_value_count ||
        batch_norm_count != (uint64_t)expected_batch_norm_count ||
        (size_t)batch_norm_count != expected_batch_norm_count) {
        return VAE_C_CHECKPOINT_INCOMPATIBLE;
    }
    status = checkpoint_decode_config_matches(reader, &model->config);
    if (status != VAE_C_CHECKPOINT_OK) {
        return status;
    }
    return reader->position == reader->size
        ? VAE_C_CHECKPOINT_OK
        : VAE_C_CHECKPOINT_FORMAT_ERROR;
}


static VAECCheckpointStatus checkpoint_decode_parameters(
    CheckpointReader *reader,
    const VAECModel *model,
    CheckpointImage *image
)
{
    uint64_t record_count;
    size_t offset = 0;
    size_t parameter_index;

    if (!checkpoint_read_u64(reader, &record_count)) {
        return VAE_C_CHECKPOINT_FORMAT_ERROR;
    }
    if (record_count != (uint64_t)model->parameter_count ||
        (size_t)record_count != model->parameter_count) {
        return VAE_C_CHECKPOINT_INCOMPATIBLE;
    }
    for (parameter_index = 0;
         parameter_index < model->parameter_count;
         parameter_index++) {
        const VAECParameter *parameter = model->parameters[parameter_index];
        uint64_t key;
        uint32_t rank;
        uint32_t reserved;
        uint64_t count;
        size_t index;

        if (!checkpoint_read_u64(reader, &key) ||
            !checkpoint_read_u32(reader, &rank) ||
            !checkpoint_read_u32(reader, &reserved) || reserved != 0u ||
            key != parameter->parameter_key ||
            rank != (uint32_t)parameter->shape.rank) {
            return VAE_C_CHECKPOINT_INCOMPATIBLE;
        }
        for (index = 0; index < 4u; index++) {
            uint64_t dimension;

            if (!checkpoint_read_u64(reader, &dimension) ||
                dimension != (uint64_t)parameter->shape.dims[index] ||
                (size_t)dimension != parameter->shape.dims[index]) {
                return VAE_C_CHECKPOINT_INCOMPATIBLE;
            }
        }
        if (!checkpoint_read_u64(reader, &count) ||
            count != (uint64_t)parameter->shape.count ||
            (size_t)count != parameter->shape.count ||
            offset > image->parameter_value_count - parameter->shape.count) {
            return VAE_C_CHECKPOINT_INCOMPATIBLE;
        }
        image->parameter_offsets[parameter_index] = offset;
        for (index = 0; index < parameter->shape.count; index++) {
            float value;
            if (!checkpoint_read_f32(reader, &value) || !isfinite(value)) {
                return VAE_C_CHECKPOINT_FORMAT_ERROR;
            }
            image->parameter_values[offset + index] = value;
        }
        for (index = 0; index < parameter->shape.count; index++) {
            float value;
            if (!checkpoint_read_f32(reader, &value) || !isfinite(value)) {
                return VAE_C_CHECKPOINT_FORMAT_ERROR;
            }
            image->parameter_m[offset + index] = value;
        }
        for (index = 0; index < parameter->shape.count; index++) {
            float value;
            if (!checkpoint_read_f32(reader, &value) ||
                !isfinite(value) || value < 0.0f) {
                return VAE_C_CHECKPOINT_FORMAT_ERROR;
            }
            image->parameter_v[offset + index] = value;
        }
        offset += parameter->shape.count;
    }
    return reader->position == reader->size &&
        offset == image->parameter_value_count
        ? VAE_C_CHECKPOINT_OK
        : VAE_C_CHECKPOINT_FORMAT_ERROR;
}


static VAECCheckpointStatus checkpoint_decode_batch_norms(
    CheckpointReader *reader,
    const CheckpointBatchNormConst *layers,
    size_t layer_count,
    CheckpointImage *image
)
{
    uint64_t record_count;
    size_t offset = 0;
    size_t layer_index;

    if (!checkpoint_read_u64(reader, &record_count)) {
        return VAE_C_CHECKPOINT_FORMAT_ERROR;
    }
    if (record_count != (uint64_t)layer_count ||
        (size_t)record_count != layer_count) {
        return VAE_C_CHECKPOINT_INCOMPATIBLE;
    }
    for (layer_index = 0; layer_index < layer_count; layer_index++) {
        const VAECBatchNorm *layer = layers[layer_index].layer;
        uint64_t identity;
        uint32_t kind;
        uint32_t reserved;
        uint64_t feature_count;
        size_t feature;

        if (!checkpoint_read_u64(reader, &identity) ||
            !checkpoint_read_u32(reader, &kind) ||
            !checkpoint_read_u32(reader, &reserved) ||
            !checkpoint_read_u64(reader, &feature_count) ||
            reserved != 0u || identity != layers[layer_index].identity ||
            kind != (uint32_t)layer->kind ||
            feature_count != (uint64_t)layer->num_features ||
            (size_t)feature_count != layer->num_features ||
            offset > image->batch_norm_value_count - layer->num_features) {
            return VAE_C_CHECKPOINT_INCOMPATIBLE;
        }
        image->batch_norm_offsets[layer_index] = offset;
        for (feature = 0; feature < layer->num_features; feature++) {
            float value;
            if (!checkpoint_read_f32(reader, &value) || !isfinite(value)) {
                return VAE_C_CHECKPOINT_FORMAT_ERROR;
            }
            image->batch_norm_mean[offset + feature] = value;
        }
        for (feature = 0; feature < layer->num_features; feature++) {
            float value;
            if (!checkpoint_read_f32(reader, &value) ||
                !isfinite(value) || value < 0.0f) {
                return VAE_C_CHECKPOINT_FORMAT_ERROR;
            }
            image->batch_norm_variance[offset + feature] = value;
        }
        offset += layer->num_features;
    }
    return reader->position == reader->size &&
        offset == image->batch_norm_value_count
        ? VAE_C_CHECKPOINT_OK
        : VAE_C_CHECKPOINT_FORMAT_ERROR;
}


static VAECCheckpointStatus checkpoint_decode_adam(
    CheckpointReader *reader,
    const VAECCheckpointRunIdentity *identity,
    CheckpointImage *image
)
{
    uint8_t configured;

    vae_c_adam_init(&image->optimizer);
    if (!checkpoint_read_u8(reader, &configured) || configured != 1u ||
        !checkpoint_read_zero_bytes(reader, 7u) ||
        !checkpoint_read_f64(reader, &image->optimizer.learning_rate) ||
        !checkpoint_read_f64(reader, &image->optimizer.beta1) ||
        !checkpoint_read_f64(reader, &image->optimizer.beta2) ||
        !checkpoint_read_f64(reader, &image->optimizer.adam_epsilon) ||
        !checkpoint_read_u64(reader, &image->optimizer.step) ||
        !checkpoint_read_f64(reader, &image->optimizer.beta1_power) ||
        !checkpoint_read_f64(reader, &image->optimizer.beta2_power) ||
        reader->position != reader->size) {
        return VAE_C_CHECKPOINT_FORMAT_ERROR;
    }
    image->optimizer.configured = true;
    if (!vae_c_adam_is_configured(&image->optimizer)) {
        return VAE_C_CHECKPOINT_FORMAT_ERROR;
    }
    return checkpoint_optimizer_config_matches_identity(
        &image->optimizer, identity
    ) ? VAE_C_CHECKPOINT_OK : VAE_C_CHECKPOINT_INCOMPATIBLE;
}


static VAECCheckpointStatus checkpoint_decode_controller(
    CheckpointReader *reader,
    CheckpointImage *image
)
{
    uint32_t kind;
    uint8_t best_defined;

    if (!checkpoint_read_u32(reader, &kind) ||
        !checkpoint_read_u8(reader, &best_defined) ||
        best_defined > 1u ||
        !checkpoint_read_zero_bytes(reader, 3u) ||
        !checkpoint_read_u64(
            reader, &image->metadata.last_completed_epoch) ||
        !checkpoint_read_u64(reader, &image->metadata.best_epoch) ||
        !checkpoint_read_f64(
            reader,
            &image->metadata.best_validation_total_mean_per_sample) ||
        !checkpoint_read_u64(
            reader, &image->metadata.no_improvement_count) ||
        reader->position != reader->size) {
        return VAE_C_CHECKPOINT_FORMAT_ERROR;
    }
    image->metadata.kind = (VAECCheckpointKind)kind;
    image->metadata.best_defined = best_defined != 0u;
    return checkpoint_metadata_is_valid(&image->metadata)
        ? VAE_C_CHECKPOINT_OK
        : VAE_C_CHECKPOINT_FORMAT_ERROR;
}


static VAECCheckpointStatus checkpoint_make_section(
    const unsigned char *data,
    size_t file_size,
    size_t offset,
    uint64_t encoded_length,
    CheckpointReader *reader,
    size_t *next_offset
)
{
    size_t length = (size_t)encoded_length;
    size_t end;

    if ((uint64_t)length != encoded_length ||
        !checkpoint_size_add(offset, length, &end) || end > file_size ||
        reader == NULL || next_offset == NULL) {
        return VAE_C_CHECKPOINT_OVERFLOW;
    }
    reader->data = data + offset;
    reader->size = length;
    reader->position = 0;
    *next_offset = end;
    return VAE_C_CHECKPOINT_OK;
}


static VAECCheckpointStatus checkpoint_parse_bytes(
    const unsigned char *data,
    size_t size,
    const VAECCheckpointRunIdentity *expected_identity,
    const VAECModel *model,
    const VAECAdam *destination_optimizer,
    CheckpointImage *image
)
{
    CheckpointReader header;
    CheckpointReader identity_reader;
    CheckpointReader model_reader;
    CheckpointReader parameter_reader;
    CheckpointReader batch_norm_reader;
    CheckpointReader adam_reader;
    CheckpointReader controller_reader;
    CheckpointBatchNormConst batch_norms[CHECKPOINT_MAX_BATCH_NORMS];
    uint32_t version;
    uint32_t flags;
    uint64_t total_size;
    uint64_t section_lengths[6];
    unsigned char digest[CHECKPOINT_DIGEST_SIZE];
    size_t offset = CHECKPOINT_HEADER_SIZE;
    size_t next_offset;
    size_t index;
    size_t batch_norm_count;
    VAECCheckpointStatus status;
    const unsigned char *header_magic;

    if (data == NULL || expected_identity == NULL || model == NULL ||
        destination_optimizer == NULL || image == NULL ||
        size < CHECKPOINT_HEADER_SIZE + CHECKPOINT_DIGEST_SIZE) {
        return VAE_C_CHECKPOINT_FORMAT_ERROR;
    }
    header.data = data;
    header.size = CHECKPOINT_HEADER_SIZE;
    header.position = 0;
    if (!checkpoint_reader_take(&header, 8u, &header_magic) ||
        memcmp(header_magic, checkpoint_magic, 8u) != 0) {
        return VAE_C_CHECKPOINT_FORMAT_ERROR;
    }
    if (!checkpoint_read_u32(&header, &version)) {
        return VAE_C_CHECKPOINT_FORMAT_ERROR;
    }
    if (version != VAE_C_CHECKPOINT_VERSION) {
        return VAE_C_CHECKPOINT_UNSUPPORTED_VERSION;
    }
    if (!checkpoint_read_u32(&header, &flags) ||
        flags != CHECKPOINT_FORMAT_FLAGS ||
        !checkpoint_read_u64(&header, &total_size)) {
        return VAE_C_CHECKPOINT_FORMAT_ERROR;
    }
    if (total_size != (uint64_t)size || (size_t)total_size != size) {
        return VAE_C_CHECKPOINT_FORMAT_ERROR;
    }
    for (index = 0; index < 6u; index++) {
        if (!checkpoint_read_u64(&header, &section_lengths[index])) {
            return VAE_C_CHECKPOINT_FORMAT_ERROR;
        }
    }
    if (header.position != header.size) {
        return VAE_C_CHECKPOINT_FORMAT_ERROR;
    }

#define CHECKPOINT_MAKE_NEXT_SECTION(reader_name, length_index)               \
    do {                                                                       \
        status = checkpoint_make_section(                                     \
            data, size, offset, section_lengths[length_index],                \
            &reader_name, &next_offset);                                      \
        if (status != VAE_C_CHECKPOINT_OK) {                                  \
            return status;                                                     \
        }                                                                      \
        offset = next_offset;                                                   \
    } while (0)

    CHECKPOINT_MAKE_NEXT_SECTION(identity_reader, 0);
    CHECKPOINT_MAKE_NEXT_SECTION(model_reader, 1);
    CHECKPOINT_MAKE_NEXT_SECTION(parameter_reader, 2);
    CHECKPOINT_MAKE_NEXT_SECTION(batch_norm_reader, 3);
    CHECKPOINT_MAKE_NEXT_SECTION(adam_reader, 4);
    CHECKPOINT_MAKE_NEXT_SECTION(controller_reader, 5);
#undef CHECKPOINT_MAKE_NEXT_SECTION

    if (offset != size - CHECKPOINT_DIGEST_SIZE) {
        return VAE_C_CHECKPOINT_FORMAT_ERROR;
    }

    vae_c_checkpoint_metadata_init(&image->metadata);
    status = checkpoint_decode_identity(
        &identity_reader, &image->metadata.run_identity
    );
    if (status != VAE_C_CHECKPOINT_OK) {
        return status;
    }
    if (!checkpoint_identity_equal(
            &image->metadata.run_identity, expected_identity, 0)) {
        return VAE_C_CHECKPOINT_INCOMPATIBLE;
    }
    batch_norm_count = checkpoint_collect_batch_norms_const(
        model, batch_norms
    );
    status = checkpoint_image_allocate(
        image, model, batch_norms, batch_norm_count
    );
    if (status != VAE_C_CHECKPOINT_OK) {
        return status;
    }
    status = checkpoint_decode_model_matches(
        &model_reader, model, batch_norm_count, image
    );
    if (status == VAE_C_CHECKPOINT_OK &&
        image->initialization_seed !=
            image->metadata.run_identity.initialization_seed) {
        status = VAE_C_CHECKPOINT_INCOMPATIBLE;
    }
    if (status == VAE_C_CHECKPOINT_OK) {
        status = checkpoint_decode_parameters(
            &parameter_reader, model, image
        );
    }
    if (status == VAE_C_CHECKPOINT_OK) {
        status = checkpoint_decode_batch_norms(
            &batch_norm_reader, batch_norms, batch_norm_count, image
        );
    }
    if (status == VAE_C_CHECKPOINT_OK) {
        status = checkpoint_decode_adam(
            &adam_reader, &image->metadata.run_identity, image
        );
    }
    if (status == VAE_C_CHECKPOINT_OK) {
        status = checkpoint_decode_controller(&controller_reader, image);
    }
    if (status != VAE_C_CHECKPOINT_OK) {
        return status;
    }
    if (!checkpoint_optimizer_config_matches_identity(
            destination_optimizer, expected_identity)) {
        return VAE_C_CHECKPOINT_INCOMPATIBLE;
    }
    if (vae_c_sha256_digest(
            data,
            size - CHECKPOINT_DIGEST_SIZE,
            digest
        ) != VAE_C_SHA256_OK) {
        return VAE_C_CHECKPOINT_OVERFLOW;
    }
    return memcmp(
        digest,
        data + size - CHECKPOINT_DIGEST_SIZE,
        CHECKPOINT_DIGEST_SIZE
    ) == 0 ? VAE_C_CHECKPOINT_OK : VAE_C_CHECKPOINT_CHECKSUM_MISMATCH;
}


static int checkpoint_image_matches_source(
    const CheckpointImage *image,
    const VAECModel *model,
    const VAECAdam *optimizer,
    const VAECCheckpointMetadata *metadata
)
{
    CheckpointBatchNormConst batch_norms[CHECKPOINT_MAX_BATCH_NORMS];
    size_t batch_norm_count = checkpoint_collect_batch_norms_const(
        model, batch_norms
    );
    size_t parameter_index;
    size_t batch_norm_index;

    if (!checkpoint_metadata_equal(&image->metadata, metadata) ||
        image->initialization_seed != model->initialization_seed ||
        image->optimizer.learning_rate != optimizer->learning_rate ||
        image->optimizer.beta1 != optimizer->beta1 ||
        image->optimizer.beta2 != optimizer->beta2 ||
        image->optimizer.adam_epsilon != optimizer->adam_epsilon ||
        image->optimizer.step != optimizer->step ||
        image->optimizer.beta1_power != optimizer->beta1_power ||
        image->optimizer.beta2_power != optimizer->beta2_power ||
        image->optimizer.configured != optimizer->configured) {
        return 0;
    }
    for (parameter_index = 0;
         parameter_index < model->parameter_count;
         parameter_index++) {
        const VAECParameter *parameter = model->parameters[parameter_index];
        size_t offset = image->parameter_offsets[parameter_index];
        size_t bytes = parameter->shape.count * sizeof(float);

        if (memcmp(image->parameter_values + offset,
                   parameter->values, bytes) != 0 ||
            memcmp(image->parameter_m + offset,
                   parameter->adam_m, bytes) != 0 ||
            memcmp(image->parameter_v + offset,
                   parameter->adam_v, bytes) != 0) {
            return 0;
        }
    }
    for (batch_norm_index = 0;
         batch_norm_index < batch_norm_count;
         batch_norm_index++) {
        const VAECBatchNorm *layer = batch_norms[batch_norm_index].layer;
        size_t offset = image->batch_norm_offsets[batch_norm_index];
        size_t bytes = layer->num_features * sizeof(float);

        if (memcmp(image->batch_norm_mean + offset,
                   layer->running_mean, bytes) != 0 ||
            memcmp(image->batch_norm_variance + offset,
                   layer->running_variance, bytes) != 0) {
            return 0;
        }
    }
    return 1;
}


static VAECCheckpointStatus checkpoint_build_file(
    const VAECModel *model,
    const VAECAdam *optimizer,
    const VAECCheckpointMetadata *metadata,
    CheckpointBuffer *file
)
{
    CheckpointBuffer sections[6];
    CheckpointBatchNormConst batch_norms[CHECKPOINT_MAX_BATCH_NORMS];
    size_t batch_norm_count;
    size_t total_size = CHECKPOINT_HEADER_SIZE + CHECKPOINT_DIGEST_SIZE;
    size_t index;
    VAECCheckpointStatus status = VAE_C_CHECKPOINT_OK;
    unsigned char digest[CHECKPOINT_DIGEST_SIZE];

    for (index = 0; index < 6u; index++) {
        checkpoint_buffer_init(&sections[index]);
    }
    batch_norm_count = checkpoint_collect_batch_norms_const(
        model, batch_norms
    );
    status = checkpoint_encode_identity(
        &sections[0], &metadata->run_identity
    );
    if (status == VAE_C_CHECKPOINT_OK) {
        status = checkpoint_encode_model(
            &sections[1], model, batch_norm_count
        );
    }
    if (status == VAE_C_CHECKPOINT_OK) {
        status = checkpoint_encode_parameters(&sections[2], model);
    }
    if (status == VAE_C_CHECKPOINT_OK) {
        status = checkpoint_encode_batch_norms(
            &sections[3], batch_norms, batch_norm_count
        );
    }
    if (status == VAE_C_CHECKPOINT_OK) {
        status = checkpoint_encode_adam(&sections[4], optimizer);
    }
    if (status == VAE_C_CHECKPOINT_OK) {
        status = checkpoint_encode_controller(&sections[5], metadata);
    }
    for (index = 0; status == VAE_C_CHECKPOINT_OK && index < 6u; index++) {
        if (!checkpoint_size_add(total_size, sections[index].size, &total_size)) {
            status = VAE_C_CHECKPOINT_OVERFLOW;
        }
    }
    if (status == VAE_C_CHECKPOINT_OK) {
        status = checkpoint_buffer_append(
            file, checkpoint_magic, sizeof(checkpoint_magic)
        );
    }
    if (status == VAE_C_CHECKPOINT_OK) {
        status = checkpoint_append_u32(file, VAE_C_CHECKPOINT_VERSION);
    }
    if (status == VAE_C_CHECKPOINT_OK) {
        status = checkpoint_append_u32(file, CHECKPOINT_FORMAT_FLAGS);
    }
    if (status == VAE_C_CHECKPOINT_OK) {
        status = (size_t)(uint64_t)total_size == total_size
            ? checkpoint_append_u64(file, (uint64_t)total_size)
            : VAE_C_CHECKPOINT_OVERFLOW;
    }
    for (index = 0; status == VAE_C_CHECKPOINT_OK && index < 6u; index++) {
        status = (size_t)(uint64_t)sections[index].size ==
                sections[index].size
            ? checkpoint_append_u64(
                file, (uint64_t)sections[index].size)
            : VAE_C_CHECKPOINT_OVERFLOW;
    }
    for (index = 0; status == VAE_C_CHECKPOINT_OK && index < 6u; index++) {
        status = checkpoint_buffer_append(
            file, sections[index].data, sections[index].size
        );
    }
    if (status == VAE_C_CHECKPOINT_OK &&
        (file->size + CHECKPOINT_DIGEST_SIZE != total_size ||
         vae_c_sha256_digest(
             file->data,
             file->size,
             digest
         ) != VAE_C_SHA256_OK)) {
        status = VAE_C_CHECKPOINT_OVERFLOW;
    }
    if (status == VAE_C_CHECKPOINT_OK) {
        status = checkpoint_buffer_append(
            file, digest, sizeof(digest)
        );
    }
    for (index = 0; index < 6u; index++) {
        checkpoint_buffer_free(&sections[index]);
    }
    return status;
}


static VAECCheckpointStatus checkpoint_read_file(
    const char *path,
    CheckpointBuffer *buffer
)
{
    FILE *file;
    long length;
    size_t size;
    int failed = 0;

    file = fopen(path, "rb");
    if (file == NULL) {
        return VAE_C_CHECKPOINT_IO_ERROR;
    }
    if (fseek(file, 0, SEEK_END) != 0 || (length = ftell(file)) < 0 ||
        fseek(file, 0, SEEK_SET) != 0) {
        (void)fclose(file);
        return VAE_C_CHECKPOINT_IO_ERROR;
    }
    size = (size_t)length;
    if ((long)size != length || size == 0) {
        (void)fclose(file);
        return VAE_C_CHECKPOINT_OVERFLOW;
    }
    buffer->data = (unsigned char *)malloc(size);
    if (buffer->data == NULL) {
        (void)fclose(file);
        return VAE_C_CHECKPOINT_ALLOCATION_FAILED;
    }
    buffer->capacity = size;
    buffer->size = size;
    if (fread(buffer->data, 1u, size, file) != size ||
        fgetc(file) != EOF || ferror(file)) {
        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    if (failed) {
        checkpoint_buffer_free(buffer);
        return VAE_C_CHECKPOINT_IO_ERROR;
    }
    return VAE_C_CHECKPOINT_OK;
}


static VAECCheckpointStatus checkpoint_write_new_file(
    const char *path,
    const unsigned char *data,
    size_t size,
    int *created
)
{
    FILE *file;
    int failed = 0;

    if (created == NULL) {
        return VAE_C_CHECKPOINT_INVALID_ARGUMENT;
    }
    *created = 0;
    file = fopen(path, "wbx");
    if (file == NULL) {
        FILE *existing = fopen(path, "rb");
        if (existing != NULL) {
            (void)fclose(existing);
            return VAE_C_CHECKPOINT_TEMP_EXISTS;
        }
        return VAE_C_CHECKPOINT_IO_ERROR;
    }
    *created = 1;
    if (fwrite(data, 1u, size, file) != size ||
        fflush(file) != 0 || ferror(file)) {
        failed = 1;
    }
    if (fclose(file) != 0) {
        failed = 1;
    }
    return failed
        ? VAE_C_CHECKPOINT_IO_ERROR
        : VAE_C_CHECKPOINT_OK;
}


static int checkpoint_publish_file(const char *temporary, const char *target)
{
#ifdef _WIN32
    DWORD attributes = GetFileAttributesA(target);

    if (attributes != INVALID_FILE_ATTRIBUTES) {
        return ReplaceFileA(
            target, temporary, NULL, REPLACEFILE_WRITE_THROUGH, NULL, NULL
        ) != 0;
    }
    return MoveFileExA(
        temporary,
        target,
        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH
    ) != 0;
#else
    return rename(temporary, target) == 0;
#endif
}


static VAECCheckpointStatus checkpoint_validate_common_inputs(
    const VAECModel *model,
    const VAECAdam *optimizer,
    const VAECCheckpointRunIdentity *identity
)
{
    if (!checkpoint_ieee754_supported()) {
        return VAE_C_CHECKPOINT_INVALID_STATE;
    }
    if (model == NULL || optimizer == NULL || identity == NULL ||
        !checkpoint_identity_is_valid(identity)) {
        return VAE_C_CHECKPOINT_INVALID_ARGUMENT;
    }
    if (vae_c_model_validate(model) != VAE_C_OP_OK ||
        !vae_c_model_is_initialized(model) ||
        !vae_c_adam_is_configured(optimizer)) {
        return VAE_C_CHECKPOINT_INVALID_STATE;
    }
    if (!checkpoint_model_matches_identity(model, identity) ||
        !checkpoint_optimizer_config_matches_identity(
            optimizer, identity)) {
        return VAE_C_CHECKPOINT_INCOMPATIBLE;
    }
    return VAE_C_CHECKPOINT_OK;
}


VAECCheckpointStatus vae_c_checkpoint_save_atomic(
    const char *path,
    const VAECModel *model,
    const VAECAdam *optimizer,
    const VAECCheckpointMetadata *metadata
)
{
    CheckpointBuffer file;
    CheckpointBuffer verification_bytes;
    CheckpointImage verification_image;
    VAECCheckpointStatus status;
    char *temporary = NULL;
    size_t path_length;
    size_t temporary_size;
    int temporary_created = 0;

    checkpoint_buffer_init(&file);
    checkpoint_buffer_init(&verification_bytes);
    checkpoint_image_init(&verification_image);
    if (path == NULL || path[0] == '\0' || metadata == NULL ||
        !checkpoint_metadata_is_valid(metadata)) {
        return VAE_C_CHECKPOINT_INVALID_ARGUMENT;
    }
    status = checkpoint_validate_common_inputs(
        model, optimizer, &metadata->run_identity
    );
    if (status != VAE_C_CHECKPOINT_OK) {
        return status;
    }
    path_length = strlen(path);
    if (!checkpoint_size_add(path_length, 5u, &temporary_size)) {
        return VAE_C_CHECKPOINT_OVERFLOW;
    }
    temporary = (char *)malloc(temporary_size);
    if (temporary == NULL) {
        return VAE_C_CHECKPOINT_ALLOCATION_FAILED;
    }
    memcpy(temporary, path, path_length);
    memcpy(temporary + path_length, ".tmp", 5u);

    status = checkpoint_build_file(model, optimizer, metadata, &file);
    if (status == VAE_C_CHECKPOINT_OK) {
        status = checkpoint_write_new_file(
            temporary, file.data, file.size, &temporary_created
        );
    }
    if (status == VAE_C_CHECKPOINT_OK) {
        status = checkpoint_read_file(temporary, &verification_bytes);
    }
    if (status == VAE_C_CHECKPOINT_OK) {
        status = checkpoint_parse_bytes(
            verification_bytes.data,
            verification_bytes.size,
            &metadata->run_identity,
            model,
            optimizer,
            &verification_image
        );
    }
    if (status == VAE_C_CHECKPOINT_OK &&
        !checkpoint_image_matches_source(
            &verification_image, model, optimizer, metadata)) {
        status = VAE_C_CHECKPOINT_FORMAT_ERROR;
    }
    if (status == VAE_C_CHECKPOINT_OK &&
        !checkpoint_publish_file(temporary, path)) {
        status = VAE_C_CHECKPOINT_IO_ERROR;
    }
    if (status == VAE_C_CHECKPOINT_OK) {
        temporary_created = 0;
    }
    if (temporary_created) {
        (void)remove(temporary);
    }
    free(temporary);
    checkpoint_buffer_free(&file);
    checkpoint_buffer_free(&verification_bytes);
    checkpoint_image_free(&verification_image);
    return status;
}


static void checkpoint_commit_image(
    const CheckpointImage *image,
    VAECModel *model,
    VAECAdam *optimizer,
    VAECCheckpointMetadata *loaded_metadata
)
{
    CheckpointBatchNormMutable batch_norms[CHECKPOINT_MAX_BATCH_NORMS];
    size_t batch_norm_count = checkpoint_collect_batch_norms_mutable(
        model, batch_norms
    );
    size_t parameter_index;
    size_t batch_norm_index;

    for (parameter_index = 0;
         parameter_index < model->parameter_count;
         parameter_index++) {
        VAECParameter *parameter = model->parameters[parameter_index];
        size_t offset = image->parameter_offsets[parameter_index];
        size_t bytes = parameter->shape.count * sizeof(float);

        memcpy(parameter->values, image->parameter_values + offset, bytes);
        memcpy(parameter->adam_m, image->parameter_m + offset, bytes);
        memcpy(parameter->adam_v, image->parameter_v + offset, bytes);
    }
    for (batch_norm_index = 0;
         batch_norm_index < batch_norm_count;
         batch_norm_index++) {
        VAECBatchNorm *layer = batch_norms[batch_norm_index].layer;
        size_t offset = image->batch_norm_offsets[batch_norm_index];
        size_t bytes = layer->num_features * sizeof(float);

        memcpy(layer->running_mean, image->batch_norm_mean + offset, bytes);
        memcpy(layer->running_variance,
               image->batch_norm_variance + offset, bytes);
    }
    *optimizer = image->optimizer;
    model->initialization_seed = image->initialization_seed;
    model->parameters_initialized = true;
    model->vae_cache_generation += UINT64_C(1);
    *loaded_metadata = image->metadata;
}


VAECCheckpointStatus vae_c_checkpoint_load(
    const char *path,
    const VAECCheckpointRunIdentity *expected_identity,
    VAECModel *model,
    VAECAdam *optimizer,
    VAECCheckpointMetadata *loaded_metadata
)
{
    CheckpointBuffer file;
    CheckpointImage image;
    VAECCheckpointStatus status;

    checkpoint_buffer_init(&file);
    checkpoint_image_init(&image);
    if (path == NULL || path[0] == '\0' || expected_identity == NULL ||
        model == NULL || optimizer == NULL || loaded_metadata == NULL ||
        (const void *)loaded_metadata == (const void *)path ||
        (const void *)loaded_metadata == (const void *)model ||
        (const void *)loaded_metadata == (const void *)optimizer ||
        (const void *)loaded_metadata == (const void *)expected_identity) {
        return VAE_C_CHECKPOINT_INVALID_ARGUMENT;
    }
    status = checkpoint_validate_common_inputs(
        model, optimizer, expected_identity
    );
    if (status != VAE_C_CHECKPOINT_OK) {
        return status;
    }
    if (model->vae_cache_generation > UINT64_MAX - UINT64_C(3)) {
        return VAE_C_CHECKPOINT_OVERFLOW;
    }
    status = checkpoint_read_file(path, &file);
    if (status == VAE_C_CHECKPOINT_OK) {
        status = checkpoint_parse_bytes(
            file.data,
            file.size,
            expected_identity,
            model,
            optimizer,
            &image
        );
    }
    if (status == VAE_C_CHECKPOINT_OK) {
        checkpoint_commit_image(
            &image, model, optimizer, loaded_metadata
        );
    }
    checkpoint_buffer_free(&file);
    checkpoint_image_free(&image);
    return status;
}


const char *vae_c_checkpoint_status_name(VAECCheckpointStatus status)
{
    switch (status) {
        case VAE_C_CHECKPOINT_OK:
            return "ok";
        case VAE_C_CHECKPOINT_INVALID_ARGUMENT:
            return "invalid_argument";
        case VAE_C_CHECKPOINT_INVALID_STATE:
            return "invalid_state";
        case VAE_C_CHECKPOINT_IO_ERROR:
            return "io_error";
        case VAE_C_CHECKPOINT_TEMP_EXISTS:
            return "temp_exists";
        case VAE_C_CHECKPOINT_FORMAT_ERROR:
            return "format_error";
        case VAE_C_CHECKPOINT_UNSUPPORTED_VERSION:
            return "unsupported_version";
        case VAE_C_CHECKPOINT_CHECKSUM_MISMATCH:
            return "checksum_mismatch";
        case VAE_C_CHECKPOINT_INCOMPATIBLE:
            return "incompatible";
        case VAE_C_CHECKPOINT_OVERFLOW:
            return "overflow";
        case VAE_C_CHECKPOINT_ALLOCATION_FAILED:
            return "allocation_failed";
        default:
            return "unknown";
    }
}
