#include "vae_c_model.h"

#include "vae_c_init.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>


typedef enum {
    MODEL_INIT_ZERO = 1,
    MODEL_INIT_ONE,
    MODEL_INIT_HE,
    MODEL_INIT_XAVIER
} ModelInitKind;


typedef struct {
    VAECParameter *parameter;
    size_t scratch_offset;
    size_t count;

    ModelInitKind kind;

    size_t fan_in;
    size_t fan_out;
} ModelInitEntry;


static int model_init_size_multiply(
    size_t left,
    size_t right,
    size_t *result
)
{
    if (result == NULL || (left != 0 && right > SIZE_MAX / left)) {
        return 1;
    }
    *result = left * right;
    return 0;
}


static bool model_init_all_equal(
    const float *values,
    size_t count,
    float expected
)
{
    size_t index;

    if (values == NULL) {
        return false;
    }
    for (index = 0; index < count; index++) {
        if (values[index] != expected) {
            return false;
        }
    }
    return true;
}


static VAECOpStatus model_init_kind_for_key(
    uint64_t key,
    ModelInitKind *kind
)
{
    if (kind == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    switch (key) {
        case VAE_C_MODEL_PARAM_KEY_ENCODER_CONV1_WEIGHT:
        case VAE_C_MODEL_PARAM_KEY_ENCODER_CONV2_WEIGHT:
        case VAE_C_MODEL_PARAM_KEY_ENCODER_CONV3_WEIGHT:
        case VAE_C_MODEL_PARAM_KEY_ENCODER_CONV4_WEIGHT:
        case VAE_C_MODEL_PARAM_KEY_DECODER_FC_WEIGHT:
        case VAE_C_MODEL_PARAM_KEY_DECODER_CONV1_WEIGHT:
        case VAE_C_MODEL_PARAM_KEY_DECODER_CONV2_WEIGHT:
        case VAE_C_MODEL_PARAM_KEY_DECODER_CONV3_WEIGHT:
            *kind = MODEL_INIT_HE;
            return VAE_C_OP_OK;

        case VAE_C_MODEL_PARAM_KEY_ENCODER_FC_WEIGHT:
        case VAE_C_MODEL_PARAM_KEY_DECODER_OUTPUT_WEIGHT:
            *kind = MODEL_INIT_XAVIER;
            return VAE_C_OP_OK;

        case VAE_C_MODEL_PARAM_KEY_ENCODER_BN1_GAMMA:
        case VAE_C_MODEL_PARAM_KEY_ENCODER_BN2_GAMMA:
        case VAE_C_MODEL_PARAM_KEY_ENCODER_BN3_GAMMA:
        case VAE_C_MODEL_PARAM_KEY_ENCODER_BN4_GAMMA:
        case VAE_C_MODEL_PARAM_KEY_DECODER_FC_BN_GAMMA:
        case VAE_C_MODEL_PARAM_KEY_DECODER_BN1_GAMMA:
        case VAE_C_MODEL_PARAM_KEY_DECODER_BN2_GAMMA:
        case VAE_C_MODEL_PARAM_KEY_DECODER_BN3_GAMMA:
            *kind = MODEL_INIT_ONE;
            return VAE_C_OP_OK;

        case VAE_C_MODEL_PARAM_KEY_ENCODER_BN1_BETA:
        case VAE_C_MODEL_PARAM_KEY_ENCODER_BN2_BETA:
        case VAE_C_MODEL_PARAM_KEY_ENCODER_BN3_BETA:
        case VAE_C_MODEL_PARAM_KEY_ENCODER_BN4_BETA:
        case VAE_C_MODEL_PARAM_KEY_ENCODER_FC_BIAS:
        case VAE_C_MODEL_PARAM_KEY_DECODER_FC_BN_BETA:
        case VAE_C_MODEL_PARAM_KEY_DECODER_BN1_BETA:
        case VAE_C_MODEL_PARAM_KEY_DECODER_BN2_BETA:
        case VAE_C_MODEL_PARAM_KEY_DECODER_BN3_BETA:
        case VAE_C_MODEL_PARAM_KEY_DECODER_OUTPUT_BIAS:
            *kind = MODEL_INIT_ZERO;
            return VAE_C_OP_OK;

        default:
            return VAE_C_OP_INVALID_STATE;
    }
}


static bool model_init_key_is_dense_weight(uint64_t key)
{
    return
        key == VAE_C_MODEL_PARAM_KEY_ENCODER_FC_WEIGHT ||
        key == VAE_C_MODEL_PARAM_KEY_DECODER_FC_WEIGHT;
}


static VAECOpStatus model_init_compute_fans(
    const VAECParameter *parameter,
    ModelInitKind kind,
    size_t *fan_in,
    size_t *fan_out
)
{
    size_t kernel_area;

    if (parameter == NULL || fan_in == NULL || fan_out == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    *fan_in = 0;
    *fan_out = 0;
    if (kind != MODEL_INIT_HE && kind != MODEL_INIT_XAVIER) {
        return VAE_C_OP_OK;
    }
    if (model_init_key_is_dense_weight(parameter->parameter_key)) {
        if (parameter->shape.rank != 2) {
            return VAE_C_OP_INVALID_STATE;
        }
        *fan_in = parameter->shape.dims[1];
        *fan_out = parameter->shape.dims[0];
    } else {
        if (parameter->shape.rank != 4) {
            return VAE_C_OP_INVALID_STATE;
        }
        if (model_init_size_multiply(
                parameter->shape.dims[2],
                parameter->shape.dims[3],
                &kernel_area) != 0 ||
            model_init_size_multiply(
                parameter->shape.dims[1],
                kernel_area,
                fan_in) != 0 ||
            model_init_size_multiply(
                parameter->shape.dims[0],
                kernel_area,
                fan_out) != 0) {

            return VAE_C_OP_OVERFLOW;
        }
    }
    if (*fan_in == 0 ||
        (kind == MODEL_INIT_XAVIER &&
         (*fan_out == 0 || *fan_in > SIZE_MAX - *fan_out))) {

        return
            kind == MODEL_INIT_XAVIER &&
            *fan_in > SIZE_MAX - *fan_out
                ? VAE_C_OP_OVERFLOW
                : VAE_C_OP_INVALID_STATE;
    }
    return VAE_C_OP_OK;
}


static VAECOpStatus model_init_build_plan(
    VAECModel *model,
    size_t total_values,
    ModelInitEntry entries[VAE_C_MODEL_MAX_PARAMETERS],
    size_t *entry_count
)
{
    size_t offset;
    size_t index;
    size_t other;
    uint64_t previous_key;

    if (model == NULL || entries == NULL || entry_count == NULL ||
        model->parameter_count == 0 ||
        model->parameter_count > VAE_C_MODEL_MAX_PARAMETERS) {

        return VAE_C_OP_INVALID_ARGUMENT;
    }
    offset = 0;
    previous_key = UINT64_C(0);
    for (index = 0; index < model->parameter_count; index++) {
        VAECParameter *parameter = model->parameters[index];
        ModelInitKind kind;
        VAECOpStatus status;
        size_t next_offset;

        if (parameter == NULL || parameter->shape.count == 0 ||
            parameter->parameter_key == UINT64_C(0) ||
            parameter->values == NULL || parameter->gradients == NULL ||
            parameter->adam_m == NULL || parameter->adam_v == NULL ||
            parameter->owned_block == NULL ||
            (index > 0 && parameter->parameter_key <= previous_key)) {

            return VAE_C_OP_INVALID_STATE;
        }
        for (other = 0; other < index; other++) {
            if (parameter == model->parameters[other]) {
                return VAE_C_OP_INVALID_STATE;
            }
        }
        status = model_init_kind_for_key(parameter->parameter_key, &kind);
        if (status != VAE_C_OP_OK) {
            return status;
        }
        if (offset > SIZE_MAX - parameter->shape.count) {
            return VAE_C_OP_OVERFLOW;
        }
        next_offset = offset + parameter->shape.count;
        if (next_offset > total_values) {
            return VAE_C_OP_INVALID_STATE;
        }
        entries[index].parameter = parameter;
        entries[index].scratch_offset = offset;
        entries[index].count = parameter->shape.count;
        entries[index].kind = kind;
        status = model_init_compute_fans(
            parameter,
            kind,
            &entries[index].fan_in,
            &entries[index].fan_out
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }
        previous_key = parameter->parameter_key;
        offset = next_offset;
    }
    if (offset != total_values) {
        return VAE_C_OP_INVALID_STATE;
    }
    *entry_count = model->parameter_count;
    return VAE_C_OP_OK;
}


static bool model_init_batch_norm_is_pristine(
    const VAECBatchNorm *layer
)
{
    return
        layer != NULL &&
        layer->num_features > 0 &&
        model_init_all_equal(
            layer->running_mean,
            layer->num_features,
            0.0f) &&
        model_init_all_equal(
            layer->running_variance,
            layer->num_features,
            1.0f) &&
        !layer->cache_valid;
}


static bool model_init_state_is_pristine(
    const VAECModel *model,
    const ModelInitEntry *entries,
    size_t entry_count
)
{
    size_t index;

    for (index = 0; index < entry_count; index++) {
        const ModelInitEntry *entry = &entries[index];
        float expected =
            entry->kind == MODEL_INIT_ONE ? 1.0f : 0.0f;

        if (!model_init_all_equal(
                entry->parameter->values,
                entry->count,
                expected) ||
            !model_init_all_equal(
                entry->parameter->gradients,
                entry->count,
                0.0f) ||
            !model_init_all_equal(
                entry->parameter->adam_m,
                entry->count,
                0.0f) ||
            !model_init_all_equal(
                entry->parameter->adam_v,
                entry->count,
                0.0f)) {

            return false;
        }
    }
    for (index = 0; index < model->config.conv_depth; index++) {
        if (!model_init_batch_norm_is_pristine(
                &model->encoder_batch_norms[index])) {

            return false;
        }
    }
    if (!model_init_batch_norm_is_pristine(
            &model->decoder_fc_batch_norm)) {

        return false;
    }
    for (index = 0;
         index < model->config.decoder_hidden_conv_count;
         index++) {

        if (!model_init_batch_norm_is_pristine(
                &model->decoder_hidden_batch_norms[index])) {

            return false;
        }
    }
    return !model->encoder_max_pool.cache_valid;
}


static bool model_init_scratch_aliases_batch_norm(
    const float *scratch,
    const VAECBatchNorm *layer
)
{
    const void *candidate = (const void *)scratch;

    return
        candidate == (const void *)layer->running_mean ||
        candidate == (const void *)layer->running_variance ||
        candidate == (const void *)layer->batch_mean ||
        candidate == (const void *)layer->batch_inv_std ||
        candidate == (const void *)layer->work_mean ||
        candidate == (const void *)layer->work_inv_std ||
        candidate == (const void *)layer->work_unbiased_variance ||
        candidate == (const void *)layer->owned_block;
}


static bool model_init_scratch_aliases_model(
    const float *scratch,
    const VAECModel *model
)
{
    size_t index;

    if ((const void *)scratch == (const void *)model ||
        (const void *)scratch ==
            (const void *)model->encoder_max_pool.argmax_indices) {

        return true;
    }
    for (index = 0; index < model->parameter_count; index++) {
        const VAECParameter *parameter = model->parameters[index];

        if ((const void *)scratch == (const void *)parameter->values ||
            (const void *)scratch == (const void *)parameter->gradients ||
            (const void *)scratch == (const void *)parameter->adam_m ||
            (const void *)scratch == (const void *)parameter->adam_v ||
            (const void *)scratch == (const void *)parameter->owned_block) {

            return true;
        }
    }
    for (index = 0; index < model->config.conv_depth; index++) {
        if (model_init_scratch_aliases_batch_norm(
                scratch,
                &model->encoder_batch_norms[index])) {

            return true;
        }
    }
    if (model_init_scratch_aliases_batch_norm(
            scratch,
            &model->decoder_fc_batch_norm)) {

        return true;
    }
    for (index = 0;
         index < model->config.decoder_hidden_conv_count;
         index++) {

        if (model_init_scratch_aliases_batch_norm(
                scratch,
                &model->decoder_hidden_batch_norms[index])) {

            return true;
        }
    }
    return false;
}


static VAECOpStatus model_init_map_status(VAECInitStatus status)
{
    if (status == VAE_C_INIT_OK) {
        return VAE_C_OP_OK;
    }
    if (status == VAE_C_INIT_FAN_SUM_OVERFLOW) {
        return VAE_C_OP_OVERFLOW;
    }
    return VAE_C_OP_INVALID_STATE;
}


static VAECOpStatus model_init_fill_entry(
    float *values,
    const ModelInitEntry *entry,
    uint64_t base_seed
)
{
    VAECInitStatus status;

    if (entry->kind == MODEL_INIT_ZERO) {
        status = vae_c_init_zero_f32(values, entry->count);
    } else if (entry->kind == MODEL_INIT_ONE) {
        status = vae_c_init_one_f32(values, entry->count);
    } else if (entry->kind == MODEL_INIT_HE) {
        status = vae_c_init_he_uniform_f32(
            values,
            entry->count,
            base_seed,
            entry->parameter->parameter_key,
            entry->fan_in
        );
    } else if (entry->kind == MODEL_INIT_XAVIER) {
        status = vae_c_init_xavier_uniform_f32(
            values,
            entry->count,
            base_seed,
            entry->parameter->parameter_key,
            entry->fan_in,
            entry->fan_out
        );
    } else {
        return VAE_C_OP_INVALID_STATE;
    }
    return model_init_map_status(status);
}


static VAECOpStatus model_init_validate_scratch_entry(
    const float *values,
    const ModelInitEntry *entry
)
{
    double bound;
    size_t fan_sum;
    size_t index;

    if (entry->kind == MODEL_INIT_HE) {
        bound = sqrt(6.0 / (double)entry->fan_in);
    } else if (entry->kind == MODEL_INIT_XAVIER) {
        if (entry->fan_in > SIZE_MAX - entry->fan_out) {
            return VAE_C_OP_OVERFLOW;
        }
        fan_sum = entry->fan_in + entry->fan_out;
        bound = sqrt(6.0 / (double)fan_sum);
    } else {
        bound = 0.0;
    }
    if ((entry->kind == MODEL_INIT_HE ||
         entry->kind == MODEL_INIT_XAVIER) &&
        (!isfinite(bound) || bound <= 0.0)) {

        return VAE_C_OP_INVALID_STATE;
    }
    for (index = 0; index < entry->count; index++) {
        double value = (double)values[index];

        if (!isfinite(value) ||
            (entry->kind == MODEL_INIT_ZERO && value != 0.0) ||
            (entry->kind == MODEL_INIT_ONE && value != 1.0) ||
            ((entry->kind == MODEL_INIT_HE ||
              entry->kind == MODEL_INIT_XAVIER) &&
             (value < -bound || value > bound))) {

            return VAE_C_OP_INVALID_STATE;
        }
    }
    return VAE_C_OP_OK;
}


static VAECOpStatus model_init_fill_scratch(
    float *scratch,
    const ModelInitEntry *entries,
    size_t entry_count,
    uint64_t base_seed
)
{
    size_t index;

    for (index = 0; index < entry_count; index++) {
        const ModelInitEntry *entry = &entries[index];
        float *values = scratch + entry->scratch_offset;
        VAECOpStatus status = model_init_fill_entry(
            values,
            entry,
            base_seed
        );

        if (status != VAE_C_OP_OK) {
            return status;
        }
        status = model_init_validate_scratch_entry(values, entry);
        if (status != VAE_C_OP_OK) {
            return status;
        }
    }
    return VAE_C_OP_OK;
}


bool vae_c_model_is_initialized(const VAECModel *model)
{
    return
        model != NULL &&
        model->parameters_initialized &&
        vae_c_model_validate(model) == VAE_C_OP_OK;
}


VAECOpStatus vae_c_model_initialize_parameters_v1(
    VAECModel *model,
    uint64_t base_seed
)
{
    ModelInitEntry entries[VAE_C_MODEL_MAX_PARAMETERS];
    VAECOpStatus status;
    float *scratch;
    uint64_t total_as_u64;
    size_t total_values;
    size_t entry_count;
    size_t index;

    if (model == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (vae_c_model_validate(model) != VAE_C_OP_OK) {
        return VAE_C_OP_INVALID_STATE;
    }
    if (model->parameters_initialized) {
        return VAE_C_OP_OBJECT_NOT_EMPTY;
    }
    if (model->initialization_seed != UINT64_C(0)) {
        return VAE_C_OP_INVALID_STATE;
    }
    total_as_u64 = model->trainable_value_count;
    total_values = (size_t)total_as_u64;
    if ((uint64_t)total_values != total_as_u64 ||
        total_values == 0 ||
        total_values > SIZE_MAX / sizeof(float)) {

        return VAE_C_OP_OVERFLOW;
    }
    status = model_init_build_plan(
        model,
        total_values,
        entries,
        &entry_count
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    if (entry_count != model->parameter_count ||
        !model_init_state_is_pristine(model, entries, entry_count)) {

        return VAE_C_OP_INVALID_STATE;
    }

    scratch = (float *)calloc(total_values, sizeof(float));
    if (scratch == NULL) {
        return VAE_C_OP_ALLOCATION_FAILED;
    }
    if (model_init_scratch_aliases_model(scratch, model)) {
        status = VAE_C_OP_INVALID_STATE;
    } else {
        status = model_init_fill_scratch(
            scratch,
            entries,
            entry_count,
            base_seed
        );
    }

    if (status == VAE_C_OP_OK) {
        for (index = 0; index < entry_count; index++) {
            const ModelInitEntry *entry = &entries[index];

            memcpy(
                entry->parameter->values,
                scratch + entry->scratch_offset,
                entry->count * sizeof(float)
            );
        }
        model->initialization_seed = base_seed;
        model->parameters_initialized = true;
    }
    free(scratch);
    return status;
}
