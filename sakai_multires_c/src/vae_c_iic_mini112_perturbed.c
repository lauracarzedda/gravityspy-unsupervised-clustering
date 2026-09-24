#include "vae_c_iic_mini112_perturbed.h"

#include "vae_augmentation.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool finite_unit_interval(const float *values, size_t count)
{
    size_t index;
    for (index = 0u; index < count; index++) {
        if (!isfinite(values[index]) || values[index] < 0.0f ||
            values[index] > 1.0f) {
            return false;
        }
    }
    return true;
}

static bool finite_values(const float *values, size_t count)
{
    size_t index;
    for (index = 0u; index < count; index++) {
        if (!isfinite(values[index])) {
            return false;
        }
    }
    return true;
}

static bool tensor_zero(const VAECTensor *tensor)
{
    size_t index;
    if (tensor == NULL || tensor->data == NULL) {
        return false;
    }
    for (index = 0u; index < tensor->shape.count; index++) {
        if (tensor->data[index] != 0.0f) {
            return false;
        }
    }
    return true;
}

static bool decoder_untouched(
    const VAECModel *model,
    const VAECModelWorkspace *workspace
)
{
    size_t index;
    if (!tensor_zero(&workspace->decoder_fc_output) ||
        !tensor_zero(&workspace->decoder_fc_batch_norm_output) ||
        !tensor_zero(&workspace->decoder_fc_relu_output) ||
        !tensor_zero(&workspace->logits) ||
        !tensor_zero(&workspace->epsilon) ||
        !tensor_zero(&workspace->z)) {
        return false;
    }
    for (index = 0u; index < model->config.decoder_upsample_count;
         index++) {
        if (!tensor_zero(&workspace->decoder_upsample_outputs[index])) {
            return false;
        }
    }
    for (index = 0u; index < model->config.decoder_hidden_conv_count;
         index++) {
        if (!tensor_zero(&workspace->decoder_hidden_conv_outputs[index]) ||
            !tensor_zero(
                &workspace->decoder_hidden_batch_norm_outputs[index]) ||
            !tensor_zero(&workspace->decoder_hidden_relu_outputs[index])) {
            return false;
        }
    }
    return true;
}

static bool state_float_count(const VAECModel *model, size_t *count)
{
    size_t total = 0u;
    size_t index;
    if (model == NULL || count == NULL) {
        return false;
    }
    for (index = 0u; index < model->parameter_count; index++) {
        const VAECParameter *parameter = model->parameters[index];
        if (parameter == NULL || parameter->values == NULL ||
            parameter->shape.count > SIZE_MAX - total) {
            return false;
        }
        total += parameter->shape.count;
    }
    for (index = 0u; index < model->config.conv_depth; index++) {
        const VAECBatchNorm *bn = &model->encoder_batch_norms[index];
        if (bn->running_mean == NULL || bn->running_variance == NULL ||
            bn->num_features > (SIZE_MAX - total) / 2u) {
            return false;
        }
        total += 2u * bn->num_features;
    }
    if (total == 0u || total > SIZE_MAX / sizeof(float)) {
        return false;
    }
    *count = total;
    return true;
}

static bool state_transfer(
    const VAECModel *model,
    float *snapshot,
    size_t snapshot_count,
    bool compare
)
{
    size_t offset = 0u;
    size_t index;
    for (index = 0u; index < model->parameter_count; index++) {
        const VAECParameter *parameter = model->parameters[index];
        size_t count = parameter->shape.count;
        size_t bytes = count * sizeof(float);
        if (count > snapshot_count - offset ||
            (compare && memcmp(snapshot + offset, parameter->values,
                               bytes) != 0)) {
            return false;
        }
        if (!compare) {
            memcpy(snapshot + offset, parameter->values, bytes);
        }
        offset += count;
    }
    for (index = 0u; index < model->config.conv_depth; index++) {
        const VAECBatchNorm *bn = &model->encoder_batch_norms[index];
        size_t count = bn->num_features;
        size_t bytes = count * sizeof(float);
        if (count > snapshot_count - offset ||
            (compare && memcmp(snapshot + offset, bn->running_mean,
                               bytes) != 0)) {
            return false;
        }
        if (!compare) {
            memcpy(snapshot + offset, bn->running_mean, bytes);
        }
        offset += count;
        if (count > snapshot_count - offset ||
            (compare && memcmp(snapshot + offset, bn->running_variance,
                               bytes) != 0)) {
            return false;
        }
        if (!compare) {
            memcpy(snapshot + offset, bn->running_variance, bytes);
        }
        offset += count;
    }
    return offset == snapshot_count;
}

void vae_c_iic_mini112_perturbed_init(VAECIICMini112Perturbed *source)
{
    if (source == NULL) {
        return;
    }
    memset(source, 0, sizeof(*source));
    vae_c_mu_canonical_contract_init(&source->contract);
    vae_c_model_init(&source->model);
    vae_c_adam_init(&source->optimizer);
    vae_c_checkpoint_metadata_init(&source->metadata);
}

void vae_c_iic_mini112_perturbed_free(VAECIICMini112Perturbed *source)
{
    if (source == NULL) {
        return;
    }
    gs_multires_event_loader_free(&source->loader);
    gs_event_manifest_free(&source->events);
    vae_c_model_free(&source->model);
    vae_c_adam_free(&source->optimizer);
    vae_c_iic_mini112_perturbed_init(source);
}

bool vae_c_iic_mini112_perturbed_open(
    VAECIICMini112Perturbed *source,
    const char *repository_root
)
{
    VAECMuValidationReport validation;
    char error[512];
    char checkpoint_sha[VAE_C_CHECKPOINT_SHA256_CAP];
    size_t root_length;
    if (source == NULL || repository_root == NULL || source->ready) {
        return false;
    }
    root_length = strlen(repository_root);
    if (root_length == 0u ||
        root_length >= sizeof(source->repository_root)) {
        return false;
    }
    vae_c_mu_validation_report_init(&validation);
    if (vae_c_mu_validate_only_label_free(
            repository_root, "mini112", &source->contract, &validation,
            error, sizeof(error)) != VAE_C_MU_OK ||
        !validation.checkpoint_loaded ||
        source->contract.profile.input_side != 112u ||
        source->contract.profile.latent_dim !=
            VAE_C_IIC_MINI112_LATENT_DIM ||
        source->contract.event_count != VAE_C_IIC_MINI112_ROWS ||
        source->contract.expected_identity.max_batch_size !=
            VAE_C_IIC_MINI112_ENCODER_MAX_BATCH ||
        gs_event_manifest_load_label_free(
            source->contract.event_manifest_path, &source->events) != 0 ||
        source->events.count != VAE_C_IIC_MINI112_ROWS ||
        gs_multires_event_loader_init(&source->loader, 112) != 0 ||
        vae_c_model_allocate(
            &source->model, &source->contract.profile,
            VAE_C_IIC_MINI112_ENCODER_MAX_BATCH) != VAE_C_OP_OK ||
        vae_c_model_initialize_parameters_v1(
            &source->model,
            source->contract.expected_identity.initialization_seed) !=
            VAE_C_OP_OK ||
        vae_c_adam_configure_v1(&source->optimizer) != VAE_C_OP_OK ||
        vae_c_checkpoint_load(
            source->contract.checkpoint_path,
            &source->contract.expected_identity,
            &source->model,
            &source->optimizer,
            &source->metadata) != VAE_C_CHECKPOINT_OK ||
        source->metadata.kind != source->contract.expected_kind ||
        source->metadata.last_completed_epoch !=
            source->contract.expected_epoch ||
        source->optimizer.step !=
            source->contract.expected_optimizer_step ||
        vae_c_mu_file_sha256(
            source->contract.checkpoint_path, checkpoint_sha) !=
            VAE_C_MU_OK ||
        strcmp(checkpoint_sha, source->contract.checkpoint_sha256) != 0) {
        vae_c_iic_mini112_perturbed_free(source);
        return false;
    }
    memcpy(source->repository_root, repository_root, root_length + 1u);
    source->ready = true;
    return true;
}

bool vae_c_iic_mini112_perturbed_compute(
    VAECIICMini112Perturbed *source,
    const size_t *manifest_indices,
    size_t count,
    uint64_t epoch,
    size_t encoder_microbatch,
    float *mu_out,
    int *crop_left_out,
    VAECIICMini112PerturbedReport *report
)
{
    const GSTensorSpec *spec;
    float *scratch_mu = NULL;
    float *snapshot = NULL;
    float *input_copy = NULL;
    int *scratch_crop = NULL;
    size_t state_count = 0u;
    size_t offset;
    uint64_t calls = 0u;
    VAECAdam optimizer_before;
    bool success = false;
    bool input_unchanged = true;
    bool decoder_clean = true;

    if (report != NULL) {
        memset(report, 0, sizeof(*report));
    }
    if (source == NULL || !source->ready || manifest_indices == NULL ||
        mu_out == NULL || count == 0u ||
        count > VAE_C_IIC_MINI112_LOGICAL_BATCH ||
        encoder_microbatch == 0u ||
        encoder_microbatch > VAE_C_IIC_MINI112_ENCODER_MAX_BATCH) {
        return false;
    }
    for (offset = 0u; offset < count; offset++) {
        if (manifest_indices[offset] >= source->events.count ||
            gs_event_manifest_get(
                &source->events, manifest_indices[offset]) == NULL) {
            return false;
        }
    }
    spec = gs_multires_event_loader_spec(&source->loader);
    if (spec == NULL || spec->values_per_event != 4u * 112u * 112u ||
        !state_float_count(&source->model, &state_count)) {
        return false;
    }
    scratch_mu = (float *)malloc(count *
                                 VAE_C_IIC_MINI112_LATENT_DIM * sizeof(float));
    scratch_crop = (int *)malloc(count * sizeof(int));
    snapshot = (float *)malloc(state_count * sizeof(float));
    input_copy = (float *)malloc(encoder_microbatch *
                                 spec->values_per_event * sizeof(float));
    if (scratch_mu == NULL || scratch_crop == NULL || snapshot == NULL ||
        input_copy == NULL ||
        !state_transfer(&source->model, snapshot, state_count, false)) {
        goto done;
    }
    optimizer_before = source->optimizer;
    for (offset = 0u; offset < count; offset += encoder_microbatch) {
        VAECModelWorkspace workspace;
        size_t batch = count - offset;
        size_t item;
        size_t input_count;
        if (batch > encoder_microbatch) {
            batch = encoder_microbatch;
        }
        input_count = batch * spec->values_per_event;
        vae_c_model_workspace_init(&workspace);
        if (vae_c_model_workspace_allocate(
                &source->model, batch, &workspace) != VAE_C_OP_OK ||
            workspace.input.shape.count != input_count ||
            workspace.mu.shape.count !=
                batch * VAE_C_IIC_MINI112_LATENT_DIM ||
            !decoder_untouched(&source->model, &workspace)) {
            vae_c_model_workspace_free(&workspace);
            goto done;
        }
        for (item = 0u; item < batch; item++) {
            size_t manifest_index = manifest_indices[offset + item];
            const GSEventRecord *record = gs_event_manifest_get(
                &source->events, manifest_index);
            GSEventRecord resolved_record;
            size_t view;
            int crop_left;
            memset(&resolved_record, 0, sizeof(resolved_record));
            resolved_record.event_index = record->event_index;
            memcpy(resolved_record.event_id, record->event_id,
                   sizeof(resolved_record.event_id));
            for (view = 0u; view < GS_EVENT_VIEW_COUNT; view++) {
                int written;
                const char *relative_path = record->image_paths[view];
                if (relative_path[0] == '\0' ||
                    relative_path[0] == '/' ||
                    relative_path[0] == '\\' ||
                    strchr(relative_path, ':') != NULL ||
                    strstr(relative_path, "..") != NULL) {
                    vae_c_model_workspace_free(&workspace);
                    goto done;
                }
                written = snprintf(
                    resolved_record.image_paths[view],
                    sizeof(resolved_record.image_paths[view]),
                    "%s/%s", source->repository_root, relative_path);
                if (written < 0 ||
                    (size_t)written >=
                        sizeof(resolved_record.image_paths[view])) {
                    vae_c_model_workspace_free(&workspace);
                    goto done;
                }
            }
            if (gs_augment_crop_left_for_event_mode(
                    VAE_C_IIC_MINI112_AUGMENTATION_SEED, epoch,
                    (uint64_t)manifest_index,
                    GS_AUGMENT_MODE_SYMMETRIC_FIGURE,
                    &crop_left) != 0 ||
                gs_multires_event_load_crop_float01(
                    &source->loader, &resolved_record, crop_left,
                    workspace.input.data + item *
                        spec->values_per_event) != 0) {
                vae_c_model_workspace_free(&workspace);
                goto done;
            }
            scratch_crop[offset + item] = crop_left;
        }
        if (!finite_unit_interval(workspace.input.data, input_count)) {
            vae_c_model_workspace_free(&workspace);
            goto done;
        }
        memcpy(input_copy, workspace.input.data,
               input_count * sizeof(float));
        if (vae_c_model_encoder_forward(
                &source->model, VAE_C_BATCH_NORM_MODE_EVAL,
                &workspace) != VAE_C_OP_OK) {
            vae_c_model_workspace_free(&workspace);
            goto done;
        }
        calls++;
        input_unchanged = input_unchanged &&
            memcmp(input_copy, workspace.input.data,
                   input_count * sizeof(float)) == 0;
        decoder_clean = decoder_clean &&
            decoder_untouched(&source->model, &workspace);
        if (!input_unchanged || !decoder_clean ||
            !finite_values(workspace.mu.data,
                           batch * VAE_C_IIC_MINI112_LATENT_DIM)) {
            vae_c_model_workspace_free(&workspace);
            goto done;
        }
        memcpy(scratch_mu + offset * VAE_C_IIC_MINI112_LATENT_DIM,
               workspace.mu.data,
               batch * VAE_C_IIC_MINI112_LATENT_DIM * sizeof(float));
        vae_c_model_workspace_free(&workspace);
    }
    if (!state_transfer(&source->model, snapshot, state_count, true) ||
        memcmp(&optimizer_before, &source->optimizer,
               sizeof(optimizer_before)) != 0) {
        goto done;
    }
    memcpy(mu_out, scratch_mu,
           count * VAE_C_IIC_MINI112_LATENT_DIM * sizeof(float));
    if (crop_left_out != NULL) {
        memcpy(crop_left_out, scratch_crop, count * sizeof(int));
    }
    success = true;
    if (report != NULL) {
        report->encoder_forward_calls = calls;
        report->samples = count;
        report->inputs_bitwise_unchanged = input_unchanged;
        report->model_parameters_bn_bitwise_unchanged = true;
        report->decoder_workspace_untouched = decoder_clean;
        report->optimizer_unchanged = true;
    }
done:
    free(input_copy);
    free(snapshot);
    free(scratch_crop);
    free(scratch_mu);
    return success;
}
