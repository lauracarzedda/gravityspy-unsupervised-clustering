#include "vae_c_iic_checkpoint.h"

#include "vae_c_sha256.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#else
#include <unistd.h>
#endif

#define IIC_CKPT_MAX_FILE_BYTES (UINT64_C(512) * UINT64_C(1024) * UINT64_C(1024))
#define IIC_CKPT_DIGEST_BYTES 32u
#define IIC_CKPT_PARAMETER_COUNT 5u

static const unsigned char iic_ckpt_magic[8] = {
    'I', 'I', 'C', 'K', 'P', 'T', '1', 0
};
static const char iic_ckpt_policy[] =
    "C5B5_SPLITMIX64_RAW_SEED_MAIN_OVER_LINEAR_BOUND_V1";

typedef struct {
    unsigned char *data;
    size_t size;
    size_t capacity;
} IICBuffer;

typedef struct {
    const unsigned char *data;
    size_t size;
    size_t cursor;
} IICReader;

static void buffer_free(IICBuffer *buffer)
{
    free(buffer->data);
    memset(buffer, 0, sizeof(*buffer));
}

static bool buffer_append(IICBuffer *buffer, const void *data, size_t length)
{
    size_t required;
    size_t capacity;
    unsigned char *replacement;

    if (buffer == NULL || length > SIZE_MAX - buffer->size) {
        return false;
    }
    required = buffer->size + length;
    if (required > IIC_CKPT_MAX_FILE_BYTES) {
        return false;
    }
    if (required > buffer->capacity) {
        capacity = buffer->capacity == 0 ? 256u : buffer->capacity;
        while (capacity < required) {
            if (capacity > SIZE_MAX / 2u) {
                capacity = required;
                break;
            }
            capacity *= 2u;
        }
        replacement = (unsigned char *)realloc(buffer->data, capacity);
        if (replacement == NULL) {
            return false;
        }
        buffer->data = replacement;
        buffer->capacity = capacity;
    }
    if (length != 0) {
        if (data == NULL) {
            memset(buffer->data + buffer->size, 0, length);
        } else {
            memcpy(buffer->data + buffer->size, data, length);
        }
    }
    buffer->size = required;
    return true;
}

static bool put_u32(IICBuffer *buffer, uint32_t value)
{
    unsigned char bytes[4];
    unsigned index;
    for (index = 0; index < 4u; index++) {
        bytes[index] = (unsigned char)(value >> (8u * index));
    }
    return buffer_append(buffer, bytes, sizeof(bytes));
}

static bool put_u64(IICBuffer *buffer, uint64_t value)
{
    unsigned char bytes[8];
    unsigned index;
    for (index = 0; index < 8u; index++) {
        bytes[index] = (unsigned char)(value >> (8u * index));
    }
    return buffer_append(buffer, bytes, sizeof(bytes));
}

static bool put_f32(IICBuffer *buffer, float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return put_u32(buffer, bits);
}

static bool put_f64(IICBuffer *buffer, double value)
{
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return put_u64(buffer, bits);
}

static bool take(IICReader *reader, size_t length, const unsigned char **part)
{
    if (reader == NULL || part == NULL || length > reader->size - reader->cursor) {
        return false;
    }
    *part = reader->data + reader->cursor;
    reader->cursor += length;
    return true;
}

static bool get_u32(IICReader *reader, uint32_t *value)
{
    const unsigned char *part;
    unsigned index;
    uint32_t result = 0;
    if (value == NULL || !take(reader, 4u, &part)) {
        return false;
    }
    for (index = 0; index < 4u; index++) {
        result |= (uint32_t)part[index] << (8u * index);
    }
    *value = result;
    return true;
}

static bool get_u64(IICReader *reader, uint64_t *value)
{
    const unsigned char *part;
    unsigned index;
    uint64_t result = 0;
    if (value == NULL || !take(reader, 8u, &part)) {
        return false;
    }
    for (index = 0; index < 8u; index++) {
        result |= (uint64_t)part[index] << (8u * index);
    }
    *value = result;
    return true;
}

static bool get_f32(IICReader *reader, float *value)
{
    uint32_t bits;
    if (value == NULL || !get_u32(reader, &bits)) {
        return false;
    }
    memcpy(value, &bits, sizeof(bits));
    return isfinite(*value);
}

static bool get_f64(IICReader *reader, double *value)
{
    uint64_t bits;
    if (value == NULL || !get_u64(reader, &bits)) {
        return false;
    }
    memcpy(value, &bits, sizeof(bits));
    return isfinite(*value);
}

static bool text_valid(const char *text, size_t capacity)
{
    size_t index;
    if (text == NULL || capacity == 0 || text[0] == 0) {
        return false;
    }
    for (index = 0; index < capacity; index++) {
        unsigned char character = (unsigned char)text[index];
        if (character == 0) {
            return true;
        }
        if (character < 32u || character > 126u) {
            return false;
        }
    }
    return false;
}

static bool hex_text_valid(
    const char *text, size_t capacity, size_t expected_length
)
{
    size_t index;
    if (!text_valid(text, capacity) || expected_length >= capacity) {
        return false;
    }
    for (index = 0; index < expected_length; index++) {
        char character = text[index];
        if (!((character >= '0' && character <= '9') ||
              (character >= 'a' && character <= 'f'))) {
            return false;
        }
    }
    return text[expected_length] == 0;
}

static bool put_text(IICBuffer *buffer, const char *text, size_t capacity)
{
    size_t length = strlen(text);
    return length < capacity && length <= UINT32_MAX &&
        put_u32(buffer, (uint32_t)length) &&
        buffer_append(buffer, text, length);
}

static bool get_text_matches(IICReader *reader, const char *expected)
{
    uint32_t length;
    const unsigned char *part;
    size_t expected_length = strlen(expected);
    return get_u32(reader, &length) && length == expected_length &&
        take(reader, length, &part) &&
        memcmp(part, expected, length) == 0;
}

static bool ieee_supported(void)
{
    float one_f = 1.0f;
    double one_d = 1.0;
    uint32_t bits_f;
    uint64_t bits_d;
    if (sizeof(float) != 4u || sizeof(double) != 8u) {
        return false;
    }
    memcpy(&bits_f, &one_f, 4u);
    memcpy(&bits_d, &one_d, 8u);
    return bits_f == UINT32_C(0x3f800000) &&
        bits_d == UINT64_C(0x3ff0000000000000);
}

void vae_c_iic_checkpoint_identity_init(VAECIICCheckpointIdentity *identity)
{
    if (identity != NULL) {
        memset(identity, 0, sizeof(*identity));
    }
}

void vae_c_iic_checkpoint_metadata_init(VAECIICCheckpointMetadata *metadata)
{
    if (metadata != NULL) {
        memset(metadata, 0, sizeof(*metadata));
    }
}

static bool identity_valid(const VAECIICCheckpointIdentity *identity)
{
    size_t index;
    if (!ieee_supported() || identity == NULL || identity->input_dim == 0 ||
        identity->capacity < 2 || identity->batch_size < 2 ||
        identity->batch_size > identity->capacity ||
        identity->train_rows < identity->batch_size ||
        identity->max_epochs == 0 ||
        identity->shuffle_seed != VAE_C_IIC_ENSEMBLE_SHUFFLE_SEED ||
        !text_valid(identity->run_id, sizeof(identity->run_id)) ||
        !hex_text_valid(identity->base_commit,
                        sizeof(identity->base_commit), 40u) ||
        !text_valid(identity->producer_state, sizeof(identity->producer_state)) ||
        !hex_text_valid(identity->producer_binary_sha256,
                        sizeof(identity->producer_binary_sha256), 64u) ||
        strcmp(identity->producer_state, identity->base_commit) == 0) {
        return false;
    }
    for (index = 0; index < VAE_C_IIC_ENSEMBLE_K; index++) {
        if (identity->seeds[index] !=
            VAE_C_IIC_ENSEMBLE_FIRST_SEED + (uint64_t)index) {
            return false;
        }
    }
    return identity->train_rows % identity->batch_size != 1u;
}

static bool expected_step(
    const VAECIICCheckpointIdentity *identity,
    const VAECIICCheckpointMetadata *metadata,
    uint64_t *step
)
{
    uint64_t batches;
    uint64_t epochs;
    if (identity == NULL || metadata == NULL || step == NULL ||
        !identity_valid(identity) ||
        (metadata->kind != VAE_C_IIC_CHECKPOINT_LATEST &&
         metadata->kind != VAE_C_IIC_CHECKPOINT_FINAL) ||
        metadata->last_completed_epoch >= identity->max_epochs ||
        (metadata->kind == VAE_C_IIC_CHECKPOINT_FINAL &&
         metadata->last_completed_epoch + 1u != identity->max_epochs) ||
        identity->train_rows > UINT64_MAX / 2u ||
        identity->batch_size > UINT64_MAX / 2u) {
        return false;
    }
    batches = (uint64_t)(identity->train_rows / identity->batch_size);
    if (identity->train_rows % identity->batch_size != 0u) {
        batches++;
    }
    epochs = metadata->last_completed_epoch + UINT64_C(1);
    if (batches == 0 || epochs > UINT64_MAX / batches) {
        return false;
    }
    *step = epochs * batches;
    return *step < UINT64_MAX / 2u;
}

static VAECParameter *head_parameter(VAECIICHead *head, size_t index)
{
    switch (index) {
        case 0: return &head->linear1.weights;
        case 1: return &head->batch_norm.gamma;
        case 2: return &head->batch_norm.beta;
        case 3: return &head->linear2.weights;
        case 4: return &head->linear2.bias;
        default: return NULL;
    }
}

static const VAECParameter *head_parameter_const(
    const VAECIICHead *head, size_t index
)
{
    return head_parameter((VAECIICHead *)head, index);
}

static VAECIICHead *pair_head(VAECIICPair *pair, size_t index)
{
    return index == 0 ? &pair->main_head : &pair->overcluster_head;
}

static const VAECIICHead *pair_head_const(
    const VAECIICPair *pair, size_t index
)
{
    return pair_head((VAECIICPair *)pair, index);
}

static bool no_active_context(const VAECIICPairWorkspace *workspace)
{
    const VAECIICHeadWorkspace *views[4] = {
        &workspace->main_a, &workspace->over_a,
        &workspace->main_b, &workspace->over_b
    };
    size_t index;
    if (workspace->backward_pair != NULL ||
        workspace->backward_input_a != NULL ||
        workspace->backward_input_b != NULL) {
        return false;
    }
    for (index = 0; index < 4; index++) {
        if (views[index]->backward_head != NULL ||
            views[index]->backward_input != NULL) {
            return false;
        }
    }
    return true;
}

static bool state_valid(
    const VAECIICEnsemble *ensemble,
    const VAECIICCheckpointIdentity *identity,
    uint64_t step
)
{
    size_t pair_index;
    if (vae_c_iic_ensemble_validate(ensemble) != VAE_C_OP_OK ||
        ensemble->input_dim != identity->input_dim ||
        ensemble->capacity != identity->capacity) {
        return false;
    }
    for (pair_index = 0; pair_index < VAE_C_IIC_ENSEMBLE_K;
         pair_index++) {
        const VAECAdam *adam = &ensemble->optimizers[pair_index];
        const VAECIICPair *pair = &ensemble->pairs[pair_index];
        size_t head_index;
        if (ensemble->seeds[pair_index] != identity->seeds[pair_index] ||
            !no_active_context(&ensemble->workspaces[pair_index]) ||
            !adam->configured || adam->step != step ||
            adam->learning_rate != VAE_C_ADAM_LEARNING_RATE_V1 ||
            adam->beta1 != VAE_C_ADAM_BETA1_V1 ||
            adam->beta2 != VAE_C_ADAM_BETA2_V1 ||
            adam->adam_epsilon != VAE_C_ADAM_EPSILON_V1 ||
            !isfinite(adam->beta1_power) ||
            !isfinite(adam->beta2_power) ||
            fabs(adam->beta1_power - pow(adam->beta1, (double)step)) > 1.0e-12 ||
            fabs(adam->beta2_power - pow(adam->beta2, (double)step)) > 1.0e-12) {
            return false;
        }
        for (head_index = 0; head_index < 2u; head_index++) {
            const VAECIICHead *head = pair_head_const(pair, head_index);
            if (head->num_batches_tracked != 2u * step) {
                return false;
            }
        }
    }
    return true;
}

static bool put_identity(
    IICBuffer *buffer,
    const VAECIICCheckpointIdentity *identity
)
{
    size_t index;
    if (!put_u32(buffer, VAE_C_IIC_ENSEMBLE_K) ||
        !put_u32(buffer, VAE_C_IIC_ENSEMBLE_MAIN_DIM) ||
        !put_u32(buffer, VAE_C_IIC_ENSEMBLE_OVER_DIM) ||
        !put_u32(buffer, VAE_C_IIC_HEAD_HIDDEN_DIM) ||
        !put_f32(buffer, VAE_C_BATCH_NORM_EPSILON) ||
        !put_f32(buffer, VAE_C_BATCH_NORM_MOMENTUM) ||
        !put_f64(buffer, VAE_C_ADAM_LEARNING_RATE_V1) ||
        !put_f64(buffer, VAE_C_ADAM_BETA1_V1) ||
        !put_f64(buffer, VAE_C_ADAM_BETA2_V1) ||
        !put_f64(buffer, VAE_C_ADAM_EPSILON_V1) ||
        !put_text(buffer, iic_ckpt_policy, sizeof(iic_ckpt_policy)) ||
        !put_u64(buffer, (uint64_t)identity->input_dim) ||
        !put_u64(buffer, (uint64_t)identity->capacity) ||
        !put_u64(buffer, (uint64_t)identity->batch_size) ||
        !put_u64(buffer, (uint64_t)identity->train_rows) ||
        !put_u64(buffer, identity->max_epochs) ||
        !put_u64(buffer, identity->shuffle_seed)) {
        return false;
    }
    for (index = 0; index < VAE_C_IIC_ENSEMBLE_K; index++) {
        if (!put_u64(buffer, identity->seeds[index])) {
            return false;
        }
    }
    return put_text(buffer, identity->run_id, sizeof(identity->run_id)) &&
        put_text(buffer, identity->base_commit,
                 sizeof(identity->base_commit)) &&
        put_text(buffer, identity->producer_state,
                 sizeof(identity->producer_state)) &&
        put_text(buffer, identity->producer_binary_sha256,
                 sizeof(identity->producer_binary_sha256));
}

static bool get_u32_matches(IICReader *reader, uint32_t expected)
{
    uint32_t value;
    return get_u32(reader, &value) && value == expected;
}

static bool get_u64_matches(IICReader *reader, uint64_t expected)
{
    uint64_t value;
    return get_u64(reader, &value) && value == expected;
}

static bool get_f32_matches(IICReader *reader, float expected)
{
    float value;
    return get_f32(reader, &value) && value == expected;
}

static bool get_f64_matches(IICReader *reader, double expected)
{
    double value;
    return get_f64(reader, &value) && value == expected;
}

static bool identity_matches(
    IICReader *reader,
    const VAECIICCheckpointIdentity *identity
)
{
    size_t index;
    if (!get_u32_matches(reader, VAE_C_IIC_ENSEMBLE_K) ||
        !get_u32_matches(reader, VAE_C_IIC_ENSEMBLE_MAIN_DIM) ||
        !get_u32_matches(reader, VAE_C_IIC_ENSEMBLE_OVER_DIM) ||
        !get_u32_matches(reader, VAE_C_IIC_HEAD_HIDDEN_DIM) ||
        !get_f32_matches(reader, VAE_C_BATCH_NORM_EPSILON) ||
        !get_f32_matches(reader, VAE_C_BATCH_NORM_MOMENTUM) ||
        !get_f64_matches(reader, VAE_C_ADAM_LEARNING_RATE_V1) ||
        !get_f64_matches(reader, VAE_C_ADAM_BETA1_V1) ||
        !get_f64_matches(reader, VAE_C_ADAM_BETA2_V1) ||
        !get_f64_matches(reader, VAE_C_ADAM_EPSILON_V1) ||
        !get_text_matches(reader, iic_ckpt_policy) ||
        !get_u64_matches(reader, (uint64_t)identity->input_dim) ||
        !get_u64_matches(reader, (uint64_t)identity->capacity) ||
        !get_u64_matches(reader, (uint64_t)identity->batch_size) ||
        !get_u64_matches(reader, (uint64_t)identity->train_rows) ||
        !get_u64_matches(reader, identity->max_epochs) ||
        !get_u64_matches(reader, identity->shuffle_seed)) {
        return false;
    }
    for (index = 0; index < VAE_C_IIC_ENSEMBLE_K; index++) {
        if (!get_u64_matches(reader, identity->seeds[index])) {
            return false;
        }
    }
    return get_text_matches(reader, identity->run_id) &&
        get_text_matches(reader, identity->base_commit) &&
        get_text_matches(reader, identity->producer_state) &&
        get_text_matches(reader, identity->producer_binary_sha256);
}

static bool put_state(
    IICBuffer *buffer,
    const VAECIICEnsemble *ensemble
)
{
    size_t pair_index;
    for (pair_index = 0; pair_index < VAE_C_IIC_ENSEMBLE_K;
         pair_index++) {
        const VAECAdam *adam = &ensemble->optimizers[pair_index];
        const VAECIICPair *pair = &ensemble->pairs[pair_index];
        size_t head_index;
        if (!put_u64(buffer, adam->step) ||
            !put_f64(buffer, adam->beta1_power) ||
            !put_f64(buffer, adam->beta2_power)) {
            return false;
        }
        for (head_index = 0; head_index < 2u; head_index++) {
            const VAECIICHead *head = pair_head_const(pair, head_index);
            size_t index;
            if (!put_u64(buffer, head->num_batches_tracked)) {
                return false;
            }
            for (index = 0; index < VAE_C_IIC_HEAD_HIDDEN_DIM; index++) {
                if (!put_f32(buffer, head->batch_norm.running_mean[index])) {
                    return false;
                }
            }
            for (index = 0; index < VAE_C_IIC_HEAD_HIDDEN_DIM; index++) {
                if (!put_f32(buffer, head->batch_norm.running_variance[index])) {
                    return false;
                }
            }
            for (index = 0; index < IIC_CKPT_PARAMETER_COUNT; index++) {
                const VAECParameter *parameter =
                    head_parameter_const(head, index);
                size_t element;
                size_t dimension;
                if (!put_u32(buffer, (uint32_t)parameter->parameter_key) ||
                    !put_u32(buffer, (uint32_t)parameter->shape.rank)) {
                    return false;
                }
                for (dimension = 0; dimension < parameter->shape.rank;
                     dimension++) {
                    if (!put_u64(buffer,
                                 (uint64_t)parameter->shape.dims[dimension])) {
                        return false;
                    }
                }
                if (!put_u64(buffer, (uint64_t)parameter->shape.count)) {
                    return false;
                }
                for (element = 0; element < parameter->shape.count; element++) {
                    if (!put_f32(buffer, parameter->values[element])) {
                        return false;
                    }
                }
                for (element = 0; element < parameter->shape.count; element++) {
                    if (!put_f32(buffer, parameter->adam_m[element])) {
                        return false;
                    }
                }
                for (element = 0; element < parameter->shape.count; element++) {
                    if (!put_f32(buffer, parameter->adam_v[element])) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

static bool get_state(
    IICReader *reader,
    VAECIICEnsemble *staged
)
{
    size_t pair_index;
    for (pair_index = 0; pair_index < VAE_C_IIC_ENSEMBLE_K;
         pair_index++) {
        VAECAdam *adam = &staged->optimizers[pair_index];
        VAECIICPair *pair = &staged->pairs[pair_index];
        size_t head_index;
        if (!get_u64(reader, &adam->step) ||
            !get_f64(reader, &adam->beta1_power) ||
            !get_f64(reader, &adam->beta2_power)) {
            return false;
        }
        for (head_index = 0; head_index < 2u; head_index++) {
            VAECIICHead *head = pair_head(pair, head_index);
            size_t index;
            if (!get_u64(reader, &head->num_batches_tracked)) {
                return false;
            }
            for (index = 0; index < VAE_C_IIC_HEAD_HIDDEN_DIM; index++) {
                if (!get_f32(reader, &head->batch_norm.running_mean[index])) {
                    return false;
                }
            }
            for (index = 0; index < VAE_C_IIC_HEAD_HIDDEN_DIM; index++) {
                if (!get_f32(reader, &head->batch_norm.running_variance[index]) ||
                    head->batch_norm.running_variance[index] < 0.0f) {
                    return false;
                }
            }
            for (index = 0; index < IIC_CKPT_PARAMETER_COUNT; index++) {
                VAECParameter *parameter = head_parameter(head, index);
                size_t element;
                size_t dimension;
                if (!get_u32_matches(reader,
                        (uint32_t)parameter->parameter_key) ||
                    !get_u32_matches(reader,
                        (uint32_t)parameter->shape.rank)) {
                    return false;
                }
                for (dimension = 0; dimension < parameter->shape.rank;
                     dimension++) {
                    if (!get_u64_matches(reader,
                            (uint64_t)parameter->shape.dims[dimension])) {
                        return false;
                    }
                }
                if (!get_u64_matches(reader,
                        (uint64_t)parameter->shape.count)) {
                    return false;
                }
                for (element = 0; element < parameter->shape.count; element++) {
                    if (!get_f32(reader, &parameter->values[element])) {
                        return false;
                    }
                }
                for (element = 0; element < parameter->shape.count; element++) {
                    if (!get_f32(reader, &parameter->adam_m[element])) {
                        return false;
                    }
                }
                for (element = 0; element < parameter->shape.count; element++) {
                    if (!get_f32(reader, &parameter->adam_v[element]) ||
                        parameter->adam_v[element] < 0.0f) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

static VAECIICCheckpointStatus build_file(
    const VAECIICCheckpointIdentity *identity,
    const VAECIICCheckpointMetadata *metadata,
    const VAECIICEnsemble *ensemble,
    IICBuffer *buffer
)
{
    uint8_t digest[IIC_CKPT_DIGEST_BYTES];
    if (!buffer_append(buffer, iic_ckpt_magic, sizeof(iic_ckpt_magic)) ||
        !put_u32(buffer, VAE_C_IIC_CHECKPOINT_VERSION) ||
        !put_identity(buffer, identity) ||
        !put_u32(buffer, (uint32_t)metadata->kind) ||
        !put_u64(buffer, metadata->last_completed_epoch) ||
        !put_state(buffer, ensemble)) {
        return VAE_C_IIC_CHECKPOINT_ALLOCATION_FAILED;
    }
    if (vae_c_sha256_digest(buffer->data, buffer->size, digest) !=
        VAE_C_SHA256_OK ||
        !buffer_append(buffer, digest, sizeof(digest))) {
        return VAE_C_IIC_CHECKPOINT_ALLOCATION_FAILED;
    }
    return VAE_C_IIC_CHECKPOINT_OK;
}

static VAECIICCheckpointStatus parse_file(
    const IICBuffer *buffer,
    const VAECIICCheckpointIdentity *expected_identity,
    VAECIICEnsemble *staged,
    VAECIICCheckpointMetadata *metadata
)
{
    IICReader reader;
    uint8_t digest[IIC_CKPT_DIGEST_BYTES];
    const unsigned char *magic;
    uint32_t kind;
    uint64_t step;
    size_t payload_size;
    if (buffer->size < sizeof(iic_ckpt_magic) + 4u + IIC_CKPT_DIGEST_BYTES) {
        return VAE_C_IIC_CHECKPOINT_CORRUPT;
    }
    payload_size = buffer->size - IIC_CKPT_DIGEST_BYTES;
    if (vae_c_sha256_digest(buffer->data, payload_size, digest) !=
            VAE_C_SHA256_OK ||
        memcmp(buffer->data + payload_size, digest, sizeof(digest)) != 0) {
        return VAE_C_IIC_CHECKPOINT_CORRUPT;
    }
    reader.data = buffer->data;
    reader.size = payload_size;
    reader.cursor = 0;
    if (!take(&reader, sizeof(iic_ckpt_magic), &magic) ||
        memcmp(magic, iic_ckpt_magic, sizeof(iic_ckpt_magic)) != 0 ||
        !get_u32_matches(&reader, VAE_C_IIC_CHECKPOINT_VERSION)) {
        return VAE_C_IIC_CHECKPOINT_CORRUPT;
    }
    if (!identity_matches(&reader, expected_identity)) {
        return VAE_C_IIC_CHECKPOINT_IDENTITY_MISMATCH;
    }
    if (!get_u32(&reader, &kind) ||
        !get_u64(&reader, &metadata->last_completed_epoch)) {
        return VAE_C_IIC_CHECKPOINT_CORRUPT;
    }
    metadata->kind = (VAECIICCheckpointKind)kind;
    if (!expected_step(expected_identity, metadata, &step)) {
        return VAE_C_IIC_CHECKPOINT_CORRUPT;
    }
    if (vae_c_iic_ensemble_allocate_v1(
            staged, expected_identity->input_dim,
            expected_identity->capacity) != VAE_C_OP_OK) {
        return VAE_C_IIC_CHECKPOINT_ALLOCATION_FAILED;
    }
    if (!get_state(&reader, staged) || reader.cursor != reader.size ||
        !state_valid(staged, expected_identity, step)) {
        return VAE_C_IIC_CHECKPOINT_CORRUPT;
    }
    return VAE_C_IIC_CHECKPOINT_OK;
}

static VAECIICCheckpointStatus read_file(const char *path, IICBuffer *buffer)
{
    FILE *file;
    long length;
    bool failed = false;
    file = fopen(path, "rb");
    if (file == NULL) {
        return VAE_C_IIC_CHECKPOINT_IO_ERROR;
    }
    if (fseek(file, 0, SEEK_END) != 0 ||
        (length = ftell(file)) < 0 ||
        (uint64_t)length > IIC_CKPT_MAX_FILE_BYTES ||
        fseek(file, 0, SEEK_SET) != 0 ||
        !buffer_append(buffer, NULL, (size_t)length)) {
        (void)fclose(file);
        return VAE_C_IIC_CHECKPOINT_IO_ERROR;
    }
    if (fread(buffer->data, 1u, (size_t)length, file) !=
            (size_t)length ||
        fgetc(file) != EOF || ferror(file)) {
        failed = true;
    }
    if (fclose(file) != 0) {
        failed = true;
    }
    return failed ? VAE_C_IIC_CHECKPOINT_IO_ERROR :
        VAE_C_IIC_CHECKPOINT_OK;
}

static bool file_exists(const char *path)
{
    FILE *file = fopen(path, "rb");
    if (file != NULL) {
        (void)fclose(file);
        return true;
    }
    return false;
}

static bool publish_file(
    const char *temporary,
    const char *target,
    VAECIICCheckpointKind kind
)
{
#ifdef _WIN32
    if (kind == VAE_C_IIC_CHECKPOINT_FINAL) {
        return MoveFileExA(temporary, target, MOVEFILE_WRITE_THROUGH) != 0;
    }
    if (file_exists(target)) {
        return ReplaceFileA(
            target, temporary, NULL, REPLACEFILE_WRITE_THROUGH, NULL, NULL
        ) != 0;
    }
    return MoveFileExA(temporary, target, MOVEFILE_WRITE_THROUGH) != 0;
#else
    if (kind == VAE_C_IIC_CHECKPOINT_FINAL) {
        if (link(temporary, target) != 0) {
            return false;
        }
        return remove(temporary) == 0;
    }
    return rename(temporary, target) == 0;
#endif
}

static void clear_view_gradients(VAECIICHeadWorkspace *view)
{
    VAECTensor *tensors[5] = {
        &view->linear1_storage,
        &view->batch_norm_storage,
        &view->relu_storage,
        &view->logits_storage,
        &view->softmax_storage
    };
    size_t index;
    for (index = 0; index < 5u; index++) {
        memset(tensors[index]->grad, 0,
               tensors[index]->shape.count * sizeof(float));
    }
    view->backward_head = NULL;
    view->backward_input = NULL;
    view->backward_generation = 0;
    view->batch_norm_cache.valid = false;
}

static void copy_persistent_state(
    VAECIICEnsemble *destination,
    const VAECIICEnsemble *source
)
{
    size_t pair_index;
    for (pair_index = 0; pair_index < VAE_C_IIC_ENSEMBLE_K;
         pair_index++) {
        VAECIICPair *pair_dst = &destination->pairs[pair_index];
        const VAECIICPair *pair_src = &source->pairs[pair_index];
        VAECIICPairWorkspace *workspace =
            &destination->workspaces[pair_index];
        size_t head_index;
        destination->optimizers[pair_index] = source->optimizers[pair_index];
        for (head_index = 0; head_index < 2u; head_index++) {
            VAECIICHead *head_dst = pair_head(pair_dst, head_index);
            const VAECIICHead *head_src =
                pair_head_const(pair_src, head_index);
            size_t parameter_index;
            memcpy(head_dst->batch_norm.running_mean,
                   head_src->batch_norm.running_mean,
                   VAE_C_IIC_HEAD_HIDDEN_DIM * sizeof(float));
            memcpy(head_dst->batch_norm.running_variance,
                   head_src->batch_norm.running_variance,
                   VAE_C_IIC_HEAD_HIDDEN_DIM * sizeof(float));
            head_dst->num_batches_tracked = head_src->num_batches_tracked;
            head_dst->forward_generation = 0;
            head_dst->batch_norm.cache_valid = false;
            head_dst->batch_norm.last_sample_count = 0;
            vae_c_shape_init(&head_dst->batch_norm.last_input_shape);
            vae_c_shape_init(&head_dst->batch_norm.last_output_shape);
            for (parameter_index = 0;
                 parameter_index < IIC_CKPT_PARAMETER_COUNT;
                 parameter_index++) {
                VAECParameter *parameter_dst =
                    head_parameter(head_dst, parameter_index);
                const VAECParameter *parameter_src =
                    head_parameter_const(head_src, parameter_index);
                size_t bytes = parameter_dst->shape.count * sizeof(float);
                memcpy(parameter_dst->values, parameter_src->values, bytes);
                memset(parameter_dst->gradients, 0, bytes);
                memcpy(parameter_dst->adam_m, parameter_src->adam_m, bytes);
                memcpy(parameter_dst->adam_v, parameter_src->adam_v, bytes);
            }
        }
        workspace->batch_size = 0;
        clear_view_gradients(&workspace->main_a);
        clear_view_gradients(&workspace->over_a);
        clear_view_gradients(&workspace->main_b);
        clear_view_gradients(&workspace->over_b);
        memset(workspace->main_a_loss_grad, 0,
               workspace->capacity * pair_dst->main_output_dim *
                   sizeof(float));
        memset(workspace->main_b_loss_grad, 0,
               workspace->capacity * pair_dst->main_output_dim *
                   sizeof(float));
        memset(workspace->over_a_loss_grad, 0,
               workspace->capacity * pair_dst->overcluster_output_dim *
                   sizeof(float));
        memset(workspace->over_b_loss_grad, 0,
               workspace->capacity * pair_dst->overcluster_output_dim *
                   sizeof(float));
        workspace->backward_pair = NULL;
        workspace->backward_input_a = NULL;
        workspace->backward_input_b = NULL;
        workspace->main_a_generation = 0;
        workspace->main_b_generation = 0;
        workspace->over_a_generation = 0;
        workspace->over_b_generation = 0;
    }
}

VAECIICCheckpointStatus vae_c_iic_checkpoint_save_atomic(
    const char *path,
    const VAECIICCheckpointIdentity *identity,
    const VAECIICCheckpointMetadata *metadata,
    const VAECIICEnsemble *ensemble
)
{
    IICBuffer file = {0};
    IICBuffer verification = {0};
    VAECIICEnsemble *staged = NULL;
    VAECIICCheckpointMetadata checked_metadata;
    VAECIICCheckpointStatus status;
    uint64_t step;
    char *temporary = NULL;
    size_t path_length;
    FILE *output = NULL;
    bool created = false;

    if (path == NULL || path[0] == 0 || ensemble == NULL ||
        !expected_step(identity, metadata, &step) ||
        !state_valid(ensemble, identity, step)) {
        return VAE_C_IIC_CHECKPOINT_INVALID_ARGUMENT;
    }
    if (metadata->kind == VAE_C_IIC_CHECKPOINT_FINAL && file_exists(path)) {
        return VAE_C_IIC_CHECKPOINT_EXISTS;
    }
    path_length = strlen(path);
    if (path_length > SIZE_MAX - sizeof(".partial")) {
        return VAE_C_IIC_CHECKPOINT_INVALID_ARGUMENT;
    }
    temporary = (char *)malloc(path_length + sizeof(".partial"));
    if (temporary == NULL) {
        return VAE_C_IIC_CHECKPOINT_ALLOCATION_FAILED;
    }
    memcpy(temporary, path, path_length);
    memcpy(temporary + path_length, ".partial", sizeof(".partial"));
    if (file_exists(temporary)) {
        status = VAE_C_IIC_CHECKPOINT_EXISTS;
        goto done;
    }
    status = build_file(identity, metadata, ensemble, &file);
    if (status != VAE_C_IIC_CHECKPOINT_OK) {
        goto done;
    }
    output = fopen(temporary, "wbx");
    if (output == NULL) {
        status = file_exists(temporary) ? VAE_C_IIC_CHECKPOINT_EXISTS :
            VAE_C_IIC_CHECKPOINT_IO_ERROR;
        goto done;
    }
    created = true;
    if (fwrite(file.data, 1u, file.size, output) != file.size ||
        fflush(output) != 0 || ferror(output)) {
        status = VAE_C_IIC_CHECKPOINT_IO_ERROR;
    } else {
        status = VAE_C_IIC_CHECKPOINT_OK;
    }
#ifdef _WIN32
    if (status == VAE_C_IIC_CHECKPOINT_OK &&
        _commit(_fileno(output)) != 0) {
        status = VAE_C_IIC_CHECKPOINT_IO_ERROR;
    }
#endif
    if (fclose(output) != 0) {
        status = VAE_C_IIC_CHECKPOINT_IO_ERROR;
    }
    output = NULL;
    if (status != VAE_C_IIC_CHECKPOINT_OK) {
        goto done;
    }
    status = read_file(temporary, &verification);
    if (status != VAE_C_IIC_CHECKPOINT_OK ||
        verification.size != file.size ||
        memcmp(verification.data, file.data, file.size) != 0) {
        status = VAE_C_IIC_CHECKPOINT_IO_ERROR;
        goto done;
    }
    staged = (VAECIICEnsemble *)malloc(sizeof(*staged));
    if (staged == NULL) {
        status = VAE_C_IIC_CHECKPOINT_ALLOCATION_FAILED;
        goto done;
    }
    vae_c_iic_ensemble_init(staged);
    status = parse_file(&verification, identity, staged, &checked_metadata);
    if (status != VAE_C_IIC_CHECKPOINT_OK ||
        checked_metadata.kind != metadata->kind ||
        checked_metadata.last_completed_epoch !=
            metadata->last_completed_epoch) {
        if (status == VAE_C_IIC_CHECKPOINT_OK) {
            status = VAE_C_IIC_CHECKPOINT_CORRUPT;
        }
        goto done;
    }
    if (!publish_file(temporary, path, metadata->kind)) {
        status = metadata->kind == VAE_C_IIC_CHECKPOINT_FINAL &&
            file_exists(path) ? VAE_C_IIC_CHECKPOINT_EXISTS :
            VAE_C_IIC_CHECKPOINT_IO_ERROR;
        goto done;
    }
    created = false;
done:
    if (output != NULL) {
        (void)fclose(output);
    }
    if (created) {
        (void)remove(temporary);
    }
    if (staged != NULL) {
        vae_c_iic_ensemble_free(staged);
        free(staged);
    }
    buffer_free(&file);
    buffer_free(&verification);
    free(temporary);
    return status;
}

VAECIICCheckpointStatus vae_c_iic_checkpoint_load(
    const char *path,
    const VAECIICCheckpointIdentity *expected_identity,
    VAECIICEnsemble *ensemble,
    VAECIICCheckpointMetadata *metadata_out
)
{
    IICBuffer file = {0};
    VAECIICEnsemble *staged;
    VAECIICCheckpointMetadata metadata;
    VAECIICCheckpointStatus status;
    size_t pair_index;
    if (path == NULL || path[0] == 0 || !identity_valid(expected_identity) ||
        ensemble == NULL || metadata_out == NULL ||
        vae_c_iic_ensemble_validate(ensemble) != VAE_C_OP_OK ||
        ensemble->input_dim != expected_identity->input_dim ||
        ensemble->capacity != expected_identity->capacity) {
        return VAE_C_IIC_CHECKPOINT_INVALID_ARGUMENT;
    }
    for (pair_index = 0; pair_index < VAE_C_IIC_ENSEMBLE_K;
         pair_index++) {
        if (ensemble->seeds[pair_index] != expected_identity->seeds[pair_index] ||
            !no_active_context(&ensemble->workspaces[pair_index])) {
            return VAE_C_IIC_CHECKPOINT_INVALID_STATE;
        }
    }
    staged = (VAECIICEnsemble *)malloc(sizeof(*staged));
    if (staged == NULL) {
        return VAE_C_IIC_CHECKPOINT_ALLOCATION_FAILED;
    }
    vae_c_iic_ensemble_init(staged);
    status = read_file(path, &file);
    if (status == VAE_C_IIC_CHECKPOINT_OK) {
        status = parse_file(&file, expected_identity, staged, &metadata);
    }
    if (status == VAE_C_IIC_CHECKPOINT_OK) {
        copy_persistent_state(ensemble, staged);
        *metadata_out = metadata;
    }
    vae_c_iic_ensemble_free(staged);
    free(staged);
    buffer_free(&file);
    return status;
}

const char *vae_c_iic_checkpoint_status_name(VAECIICCheckpointStatus status)
{
    switch (status) {
        case VAE_C_IIC_CHECKPOINT_OK: return "OK";
        case VAE_C_IIC_CHECKPOINT_INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case VAE_C_IIC_CHECKPOINT_INVALID_STATE: return "INVALID_STATE";
        case VAE_C_IIC_CHECKPOINT_IDENTITY_MISMATCH: return "IDENTITY_MISMATCH";
        case VAE_C_IIC_CHECKPOINT_CORRUPT: return "CORRUPT";
        case VAE_C_IIC_CHECKPOINT_IO_ERROR: return "IO_ERROR";
        case VAE_C_IIC_CHECKPOINT_EXISTS: return "EXISTS";
        case VAE_C_IIC_CHECKPOINT_ALLOCATION_FAILED: return "ALLOCATION_FAILED";
        default: return "UNKNOWN";
    }
}
