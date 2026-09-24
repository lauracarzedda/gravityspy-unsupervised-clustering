#include "vae_c_iic_ensemble.h"

#include "vae_c_rng.h"

#include <math.h>
#include <string.h>


static VAECOpStatus fill_linear_uniform(
    VAECParameter *parameter,
    size_t fan_in,
    VAECRng *rng
)
{
    double bound;
    size_t index;

    if (parameter == NULL || rng == NULL || fan_in == 0 ||
        parameter->values == NULL || parameter->shape.count == 0) {
        return VAE_C_OP_INVALID_STATE;
    }
    bound = 1.0 / sqrt((double)fan_in);
    if (!isfinite(bound) || bound <= 0.0) {
        return VAE_C_OP_INVALID_STATE;
    }
    for (index = 0; index < parameter->shape.count; index++) {
        float unit;
        double value;

        if (vae_c_rng_uniform_f32(rng, &unit) != VAE_C_RNG_OK) {
            return VAE_C_OP_INVALID_STATE;
        }
        value = (2.0 * (double)unit - 1.0) * bound;
        if (!isfinite(value)) {
            return VAE_C_OP_INVALID_STATE;
        }
        parameter->values[index] = (float)value;
    }
    return VAE_C_OP_OK;
}


static VAECOpStatus initialize_head(VAECIICHead *head, VAECRng *rng)
{
    VAECOpStatus status;
    size_t index;

    status = fill_linear_uniform(
        &head->linear1.weights, head->input_dim, rng
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }

    /* BatchNorm reset performs no RNG draw. */
    for (index = 0; index < VAE_C_IIC_HEAD_HIDDEN_DIM; index++) {
        head->batch_norm.gamma.values[index] = 1.0f;
        head->batch_norm.beta.values[index] = 0.0f;
        head->batch_norm.running_mean[index] = 0.0f;
        head->batch_norm.running_variance[index] = 1.0f;
    }
    head->num_batches_tracked = 0;

    status = fill_linear_uniform(
        &head->linear2.weights, VAE_C_IIC_HEAD_HIDDEN_DIM, rng
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }
    return fill_linear_uniform(
        &head->linear2.bias, VAE_C_IIC_HEAD_HIDDEN_DIM, rng
    );
}


static VAECOpStatus initialize_pair(VAECIICPair *pair, uint64_t seed)
{
    VAECRng rng;
    VAECOpStatus status;

    if (vae_c_iic_pair_validate(pair) != VAE_C_OP_OK ||
        vae_c_rng_init(&rng, seed) != VAE_C_RNG_OK) {
        return VAE_C_OP_INVALID_STATE;
    }
    status = initialize_head(&pair->main_head, &rng);
    if (status != VAE_C_OP_OK) {
        return status;
    }
    return initialize_head(&pair->overcluster_head, &rng);
}


void vae_c_iic_ensemble_init(VAECIICEnsemble *ensemble)
{
    size_t index;

    if (ensemble == NULL) {
        return;
    }
    memset(ensemble, 0, sizeof(*ensemble));
    for (index = 0; index < VAE_C_IIC_ENSEMBLE_K; index++) {
        vae_c_iic_pair_init(&ensemble->pairs[index]);
        vae_c_iic_pair_workspace_init(&ensemble->workspaces[index]);
        vae_c_adam_init(&ensemble->optimizers[index]);
    }
}


void vae_c_iic_ensemble_free(VAECIICEnsemble *ensemble)
{
    size_t index;

    if (ensemble == NULL) {
        return;
    }
    for (index = 0; index < VAE_C_IIC_ENSEMBLE_K; index++) {
        vae_c_iic_pair_workspace_free(&ensemble->workspaces[index]);
        vae_c_iic_pair_free(&ensemble->pairs[index]);
        vae_c_adam_free(&ensemble->optimizers[index]);
    }
    vae_c_iic_ensemble_init(ensemble);
}


bool vae_c_iic_ensemble_is_empty(const VAECIICEnsemble *ensemble)
{
    size_t index;

    if (ensemble == NULL || ensemble->allocated ||
        ensemble->input_dim != 0 || ensemble->capacity != 0) {
        return false;
    }
    for (index = 0; index < VAE_C_IIC_ENSEMBLE_K; index++) {
        if (ensemble->seeds[index] != 0 ||
            !vae_c_iic_pair_is_empty(&ensemble->pairs[index]) ||
            !vae_c_iic_pair_workspace_is_empty(&ensemble->workspaces[index]) ||
            !vae_c_adam_is_empty(&ensemble->optimizers[index])) {
            return false;
        }
    }
    return true;
}


VAECOpStatus vae_c_iic_ensemble_validate(const VAECIICEnsemble *ensemble)
{
    size_t index;

    if (ensemble == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (!ensemble->allocated || ensemble->input_dim == 0 ||
        ensemble->capacity < 2) {
        return VAE_C_OP_INVALID_STATE;
    }
    for (index = 0; index < VAE_C_IIC_ENSEMBLE_K; index++) {
        const VAECIICPair *pair = &ensemble->pairs[index];
        const VAECIICPairWorkspace *workspace =
            &ensemble->workspaces[index];
        size_t prior;

        if (pair->input_dim != ensemble->input_dim ||
            pair->main_output_dim != VAE_C_IIC_ENSEMBLE_MAIN_DIM ||
            pair->overcluster_output_dim != VAE_C_IIC_ENSEMBLE_OVER_DIM ||
            workspace->capacity != ensemble->capacity ||
            vae_c_iic_pair_validate(pair) != VAE_C_OP_OK ||
            vae_c_iic_pair_workspace_validate(pair, workspace) !=
                VAE_C_OP_OK ||
            !vae_c_adam_is_configured(&ensemble->optimizers[index])) {
            return VAE_C_OP_INVALID_STATE;
        }
        for (prior = 0; prior < index; prior++) {
            if (ensemble->seeds[index] == ensemble->seeds[prior]) {
                return VAE_C_OP_INVALID_STATE;
            }
        }
    }
    return VAE_C_OP_OK;
}


VAECOpStatus vae_c_iic_ensemble_allocate_seeded(
    VAECIICEnsemble *ensemble,
    size_t input_dim,
    size_t capacity,
    const uint64_t seeds[VAE_C_IIC_ENSEMBLE_K]
)
{
    size_t index;

    if (ensemble == NULL || seeds == NULL || input_dim == 0 ||
        capacity < 2) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (!vae_c_iic_ensemble_is_empty(ensemble)) {
        return VAE_C_OP_OBJECT_NOT_EMPTY;
    }
    for (index = 0; index < VAE_C_IIC_ENSEMBLE_K; index++) {
        size_t prior;

        for (prior = 0; prior < index; prior++) {
            if (seeds[index] == seeds[prior]) {
                return VAE_C_OP_INVALID_ARGUMENT;
            }
        }
    }

    /* Build directly in final storage: workspaces can retain pair pointers. */
    for (index = 0; index < VAE_C_IIC_ENSEMBLE_K; index++) {
        VAECOpStatus status;

        status = vae_c_iic_pair_allocate(
            &ensemble->pairs[index], input_dim,
            VAE_C_IIC_ENSEMBLE_MAIN_DIM, VAE_C_IIC_ENSEMBLE_OVER_DIM
        );
        if (status == VAE_C_OP_OK) {
            status = initialize_pair(&ensemble->pairs[index], seeds[index]);
        }
        if (status == VAE_C_OP_OK) {
            status = vae_c_iic_pair_workspace_allocate(
                &ensemble->pairs[index], capacity,
                &ensemble->workspaces[index]
            );
        }
        if (status == VAE_C_OP_OK) {
            status = vae_c_adam_configure_v1(&ensemble->optimizers[index]);
        }
        if (status != VAE_C_OP_OK) {
            vae_c_iic_ensemble_free(ensemble);
            return status;
        }
        ensemble->seeds[index] = seeds[index];
    }
    ensemble->input_dim = input_dim;
    ensemble->capacity = capacity;
    ensemble->allocated = true;
    if (vae_c_iic_ensemble_validate(ensemble) != VAE_C_OP_OK) {
        vae_c_iic_ensemble_free(ensemble);
        return VAE_C_OP_INVALID_STATE;
    }
    return VAE_C_OP_OK;
}


VAECOpStatus vae_c_iic_ensemble_allocate_v1(
    VAECIICEnsemble *ensemble,
    size_t input_dim,
    size_t capacity
)
{
    const uint64_t seeds[VAE_C_IIC_ENSEMBLE_K] = {
        UINT64_C(123), UINT64_C(124), UINT64_C(125),
        UINT64_C(126), UINT64_C(127)
    };

    return vae_c_iic_ensemble_allocate_seeded(
        ensemble, input_dim, capacity, seeds
    );
}


VAECOpStatus vae_c_iic_ensemble_train_step(
    VAECIICEnsemble *ensemble,
    const VAECTensor *input_a,
    const VAECTensor *input_b
)
{
    size_t index;

    if (ensemble == NULL || input_a == NULL || input_b == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }
    if (vae_c_iic_ensemble_validate(ensemble) != VAE_C_OP_OK) {
        return VAE_C_OP_INVALID_STATE;
    }
    for (index = 0; index < VAE_C_IIC_ENSEMBLE_K; index++) {
        VAECOpStatus status = vae_c_iic_pair_train_step(
            &ensemble->pairs[index], &ensemble->workspaces[index],
            &ensemble->optimizers[index], input_a, input_b
        );

        if (status != VAE_C_OP_OK) {
            return status;
        }
    }
    return VAE_C_OP_OK;
}
