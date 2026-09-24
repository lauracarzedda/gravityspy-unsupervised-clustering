#include "vae_c_iic_h.h"
#include "vae_c_iic_loss.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static bool input_and_output_disjoint(const VAECTensor *input,
                                      const float *output, size_t output_count)
{
    uintptr_t input_start = (uintptr_t)input->data;
    uintptr_t output_start = (uintptr_t)output;
    size_t input_bytes, output_bytes;
    if (input->shape.count > SIZE_MAX / sizeof(float) ||
        output_count > SIZE_MAX / sizeof(float)) return false;
    input_bytes = input->shape.count * sizeof(float);
    output_bytes = output_count * sizeof(float);
    if (input_start > UINTPTR_MAX - input_bytes ||
        output_start > UINTPTR_MAX - output_bytes) return false;
    return input_start + input_bytes <= output_start ||
           output_start + output_bytes <= input_start;
}

static VAECOpStatus validate_main_heads(const VAECIICEnsemble *ensemble)
{
    size_t k;
    if (ensemble == NULL || !ensemble->allocated ||
        ensemble->input_dim == 0u) return VAE_C_OP_INVALID_STATE;
    for (k = 0u; k < VAE_C_IIC_ENSEMBLE_K; k++) {
        const VAECIICHead *head = &ensemble->pairs[k].main_head;
        if (ensemble->seeds[k] !=
                VAE_C_IIC_ENSEMBLE_FIRST_SEED + (uint64_t)k ||
            head->input_dim != ensemble->input_dim ||
            head->output_dim != VAE_C_IIC_ENSEMBLE_MAIN_DIM ||
            vae_c_iic_head_validate(head) != VAE_C_OP_OK)
            return VAE_C_OP_INVALID_STATE;
    }
    return VAE_C_OP_OK;
}

void vae_c_iic_h_workspace_init(VAECIICHWorkspace *workspace)
{
    size_t k;
    if (workspace == NULL) return;
    memset(workspace, 0, sizeof(*workspace));
    for (k = 0u; k < VAE_C_IIC_ENSEMBLE_K; k++)
        vae_c_iic_head_workspace_init(&workspace->main[k]);
}

void vae_c_iic_h_workspace_free(VAECIICHWorkspace *workspace)
{
    size_t k;
    if (workspace == NULL) return;
    for (k = 0u; k < VAE_C_IIC_ENSEMBLE_K; k++)
        vae_c_iic_head_workspace_free(&workspace->main[k]);
    free(workspace->row_major_scratch);
    vae_c_iic_h_workspace_init(workspace);
}

VAECOpStatus vae_c_iic_h_workspace_allocate(
    const VAECIICEnsemble *ensemble, size_t capacity,
    VAECIICHWorkspace *workspace)
{
    size_t k;
    VAECOpStatus status;
    if (workspace == NULL || ensemble == NULL || capacity == 0u)
        return VAE_C_OP_INVALID_ARGUMENT;
    if (workspace->allocated || workspace->row_major_scratch != NULL)
        return VAE_C_OP_OBJECT_NOT_EMPTY;
    if (capacity > SIZE_MAX / VAE_C_IIC_H_COLUMNS / sizeof(float))
        return VAE_C_OP_OVERFLOW;
    status = validate_main_heads(ensemble);
    if (status != VAE_C_OP_OK) return status;
    workspace->row_major_scratch = (float *)malloc(
        capacity * VAE_C_IIC_H_COLUMNS * sizeof(float));
    if (workspace->row_major_scratch == NULL)
        return VAE_C_OP_ALLOCATION_FAILED;
    for (k = 0u; k < VAE_C_IIC_ENSEMBLE_K; k++) {
        status = vae_c_iic_head_workspace_allocate(
            &ensemble->pairs[k].main_head, capacity, &workspace->main[k]);
        if (status != VAE_C_OP_OK) {
            vae_c_iic_h_workspace_free(workspace);
            return status;
        }
    }
    workspace->capacity = capacity;
    workspace->input_dim = ensemble->input_dim;
    workspace->allocated = true;
    return VAE_C_OP_OK;
}

VAECOpStatus vae_c_iic_h_forward(
    VAECIICEnsemble *ensemble, const VAECTensor *central_mu,
    VAECIICHWorkspace *workspace, float *output,
    size_t output_float_capacity)
{
    size_t rows, k, row, c, i, needed;
    VAECOpStatus status;
    if (ensemble == NULL || central_mu == NULL || workspace == NULL ||
        output == NULL) return VAE_C_OP_INVALID_ARGUMENT;
    status = validate_main_heads(ensemble);
    if (status != VAE_C_OP_OK) return status;
    if (!workspace->allocated || workspace->row_major_scratch == NULL ||
        workspace->input_dim != ensemble->input_dim ||
        central_mu->shape.rank != 2u || central_mu->data == NULL ||
        central_mu->shape.dims[1] != ensemble->input_dim ||
        central_mu->shape.strides[0] != ensemble->input_dim ||
        central_mu->shape.strides[1] != 1u)
        return VAE_C_OP_SHAPE_MISMATCH;
    rows = central_mu->shape.dims[0];
    if (rows == 0u || rows > workspace->capacity ||
        rows > SIZE_MAX / ensemble->input_dim ||
        central_mu->shape.count != rows * ensemble->input_dim ||
        rows > SIZE_MAX / VAE_C_IIC_H_COLUMNS)
        return VAE_C_OP_SHAPE_MISMATCH;
    needed = rows * VAE_C_IIC_H_COLUMNS;
    if (output_float_capacity < needed ||
        !input_and_output_disjoint(central_mu, output, needed))
        return VAE_C_OP_INVALID_ARGUMENT;
    for (i = 0u; i < central_mu->shape.count; i++) {
        if (!isfinite(central_mu->data[i])) return VAE_C_OP_INVALID_STATE;
    }
    for (k = 0u; k < VAE_C_IIC_ENSEMBLE_K; k++) {
        if (!vae_c_iic_head_workspace_is_allocated(
                &ensemble->pairs[k].main_head, &workspace->main[k]) ||
            workspace->main[k].capacity < rows)
            return VAE_C_OP_INVALID_STATE;
    }
    for (k = 0u; k < VAE_C_IIC_ENSEMBLE_K; k++) {
        const VAECTensor *probabilities;
        status = vae_c_iic_head_forward(
            &ensemble->pairs[k].main_head, VAE_C_BATCH_NORM_MODE_EVAL,
            central_mu, &workspace->main[k]);
        if (status != VAE_C_OP_OK) return status;
        probabilities = &workspace->main[k].softmax_output;
        for (row = 0u; row < rows; row++) {
            float sum = 0.0f;
            for (c = 0u; c < VAE_C_IIC_ENSEMBLE_MAIN_DIM; c++) {
                float value = probabilities->data[
                    row * VAE_C_IIC_ENSEMBLE_MAIN_DIM + c];
                if (!isfinite(value) || value < 0.0f || value > 1.0f)
                    return VAE_C_OP_INVALID_STATE;
                workspace->row_major_scratch[
                    row * VAE_C_IIC_H_COLUMNS +
                    k * VAE_C_IIC_ENSEMBLE_MAIN_DIM + c] = value;
                sum += value;
            }
            if (!isfinite(sum) ||
                fabsf(sum - 1.0f) > VAE_C_IIC_PROBABILITY_ROW_RTOL)
                return VAE_C_OP_INVALID_STATE;
        }
    }
    memcpy(output, workspace->row_major_scratch, needed * sizeof(float));
    return VAE_C_OP_OK;
}
