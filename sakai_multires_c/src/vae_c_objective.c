#include "vae_c_objective.h"

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>


static int vae_c_objective_size_multiply(
    size_t left,
    size_t right,
    size_t *result
)
{
    if (result == NULL) {
        return 1;
    }

    if (left != 0 && right > SIZE_MAX / left) {
        return 1;
    }

    *result = left * right;
    return 0;
}


static VAECOpStatus vae_c_objective_validate_shape(
    const VAECShape *shape
)
{
    VAECShape canonical;
    VAECStorageStatus storage_status;
    size_t index;

    if (shape == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    storage_status = vae_c_shape_make(
        shape->rank,
        shape->dims,
        &canonical
    );
    if (storage_status != VAE_C_STORAGE_OK) {
        return
            storage_status == VAE_C_STORAGE_OVERFLOW
                ? VAE_C_OP_OVERFLOW
                : VAE_C_OP_INVALID_STATE;
    }

    if (shape->count != canonical.count) {
        return VAE_C_OP_INVALID_STATE;
    }

    for (index = 0; index < shape->rank; index++) {
        if (shape->dims[index] != canonical.dims[index] ||
            shape->strides[index] != canonical.strides[index]) {

            return VAE_C_OP_INVALID_STATE;
        }
    }

    for (index = shape->rank;
         index < VAE_C_TENSOR_MAX_RANK;
         index++) {

        if (shape->dims[index] != 0 || shape->strides[index] != 0) {
            return VAE_C_OP_INVALID_STATE;
        }
    }

    return VAE_C_OP_OK;
}


static VAECOpStatus vae_c_objective_validate_tensor(
    const VAECTensor *tensor
)
{
    VAECOpStatus status;
    size_t array_count;
    size_t total_elements;
    size_t expected_bytes;

    if (tensor == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    status = vae_c_objective_validate_shape(&tensor->shape);
    if (status != VAE_C_OP_OK) {
        return status;
    }

    if (tensor->data == NULL) {
        return VAE_C_OP_INVALID_STATE;
    }

    if (tensor->has_grad) {
        if (tensor->grad == NULL || tensor->grad == tensor->data) {
            return VAE_C_OP_INVALID_STATE;
        }
    } else if (tensor->grad != NULL) {
        return VAE_C_OP_INVALID_STATE;
    }

    if (!tensor->owns_memory) {
        if (tensor->owned_block != NULL || tensor->owned_bytes != 0) {
            return VAE_C_OP_INVALID_STATE;
        }
        return VAE_C_OP_OK;
    }

    if (tensor->owned_block == NULL ||
        tensor->data != tensor->owned_block) {

        return VAE_C_OP_INVALID_STATE;
    }

    array_count = tensor->has_grad ? 2 : 1;
    if (vae_c_objective_size_multiply(
            tensor->shape.count,
            array_count,
            &total_elements) != 0 ||
        vae_c_objective_size_multiply(
            total_elements,
            sizeof(float),
            &expected_bytes) != 0) {

        return VAE_C_OP_INVALID_STATE;
    }

    if (tensor->owned_bytes != expected_bytes) {
        return VAE_C_OP_INVALID_STATE;
    }

    if (tensor->has_grad &&
        tensor->grad != tensor->data + tensor->shape.count) {

        return VAE_C_OP_INVALID_STATE;
    }

    return VAE_C_OP_OK;
}


static bool vae_c_objective_tensors_have_exact_alias(
    const VAECTensor *logits,
    const VAECTensor *targets
)
{
    if (logits == targets || logits->data == targets->data) {
        return true;
    }

    if (logits->grad != NULL &&
        (logits->grad == targets->data ||
         logits->grad == targets->grad)) {

        return true;
    }

    return
        targets->grad != NULL && logits->data == targets->grad;
}


static bool vae_c_objective_stats_has_exact_alias(
    const VAECBCEWithLogitsStats *stats,
    const VAECTensor *logits,
    const VAECTensor *targets
)
{
    const void *stats_pointer = (const void *)stats;

    return
        stats_pointer == (const void *)logits ||
        stats_pointer == (const void *)targets ||
        stats_pointer == (const void *)logits->data ||
        stats_pointer == (const void *)logits->grad ||
        stats_pointer == (const void *)targets->data ||
        stats_pointer == (const void *)targets->grad;
}


static VAECOpStatus vae_c_objective_validate_pair(
    const VAECTensor *logits,
    const VAECTensor *targets,
    bool require_logit_gradient,
    size_t *batch_size,
    size_t *values_per_sample,
    size_t *total_values
)
{
    VAECOpStatus status;
    size_t channel_height;
    size_t candidate_values_per_sample;
    size_t candidate_total_values;

    if (logits == NULL || targets == NULL ||
        batch_size == NULL || values_per_sample == NULL ||
        total_values == NULL) {

        return VAE_C_OP_INVALID_ARGUMENT;
    }

    status = vae_c_objective_validate_tensor(logits);
    if (status != VAE_C_OP_OK) {
        return status;
    }

    status = vae_c_objective_validate_tensor(targets);
    if (status != VAE_C_OP_OK) {
        return status;
    }

    if (vae_c_objective_tensors_have_exact_alias(logits, targets)) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    if (logits->shape.rank != 4 ||
        targets->shape.rank != 4 ||
        !vae_c_shape_equal(&logits->shape, &targets->shape)) {

        return VAE_C_OP_SHAPE_MISMATCH;
    }

    if (vae_c_objective_size_multiply(
            logits->shape.dims[1],
            logits->shape.dims[2],
            &channel_height) != 0 ||
        vae_c_objective_size_multiply(
            channel_height,
            logits->shape.dims[3],
            &candidate_values_per_sample) != 0 ||
        vae_c_objective_size_multiply(
            logits->shape.dims[0],
            candidate_values_per_sample,
            &candidate_total_values) != 0) {

        return VAE_C_OP_OVERFLOW;
    }

    if (candidate_total_values != logits->shape.count ||
        candidate_total_values != targets->shape.count) {

        return VAE_C_OP_INVALID_STATE;
    }

    if (require_logit_gradient && !logits->has_grad) {
        return VAE_C_OP_GRADIENT_NOT_AVAILABLE;
    }

    *batch_size = logits->shape.dims[0];
    *values_per_sample = candidate_values_per_sample;
    *total_values = candidate_total_values;
    return VAE_C_OP_OK;
}


static float vae_c_objective_stable_sigmoid_unchecked(float logit)
{
    float exponential;

    if (logit >= 0.0f) {
        exponential = expf(-logit);
        return 1.0f / (1.0f + exponential);
    }

    exponential = expf(logit);
    return exponential / (1.0f + exponential);
}


static VAECOpStatus vae_c_objective_stable_sigmoid(
    float logit,
    float *sigmoid_value
)
{
    float candidate;

    if (sigmoid_value == NULL || !isfinite(logit)) {
        return VAE_C_OP_INVALID_STATE;
    }

    candidate = vae_c_objective_stable_sigmoid_unchecked(logit);
    if (!isfinite(candidate) ||
        candidate < 0.0f || candidate > 1.0f) {

        return VAE_C_OP_INVALID_STATE;
    }

    *sigmoid_value = candidate;
    return VAE_C_OP_OK;
}


typedef struct {
    VAECTensor *logits;
    const VAECTensor *targets;
    size_t total_values;
    float upstream_gradient;
    float batch_size_f32;
} VAECBCEBackwardContext;


static VAECOpStatus vae_c_objective_bce_backward_prevalidate(
    VAECTensor *logits,
    const VAECTensor *targets,
    float upstream_gradient,
    VAECBCEBackwardContext *context
)
{
    VAECBCEBackwardContext candidate;
    VAECOpStatus status;
    size_t batch_size;
    size_t values_per_sample;
    size_t total_values;
    size_t index;
    float batch_size_f32;

    if (!isfinite(upstream_gradient)) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    if (context == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    status = vae_c_objective_validate_pair(
        logits,
        targets,
        true,
        &batch_size,
        &values_per_sample,
        &total_values
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }

    batch_size_f32 = (float)batch_size;
    if (!isfinite(batch_size_f32) || !(batch_size_f32 > 0.0f)) {
        return VAE_C_OP_OVERFLOW;
    }

    for (index = 0; index < total_values; index++) {
        float logit = logits->data[index];
        float target = targets->data[index];
        float sigmoid_value;
        float contribution;
        float next_gradient;

        if (!isfinite(logit) ||
            !isfinite(target) ||
            target < 0.0f || target > 1.0f) {

            return VAE_C_OP_INVALID_STATE;
        }

        status = vae_c_objective_stable_sigmoid(
            logit,
            &sigmoid_value
        );
        if (status != VAE_C_OP_OK) {
            return status;
        }

        contribution =
            upstream_gradient *
            (sigmoid_value - target) /
            batch_size_f32;
        next_gradient = logits->grad[index] + contribution;

        if (!isfinite(contribution) ||
            !isfinite(logits->grad[index]) ||
            !isfinite(next_gradient)) {

            return VAE_C_OP_INVALID_STATE;
        }
    }

    candidate.logits = logits;
    candidate.targets = targets;
    candidate.total_values = total_values;
    candidate.upstream_gradient = upstream_gradient;
    candidate.batch_size_f32 = batch_size_f32;
    *context = candidate;
    (void)values_per_sample;
    return VAE_C_OP_OK;
}


static void vae_c_objective_bce_backward_commit(
    const VAECBCEBackwardContext *context
)
{
    size_t index;

    for (index = 0; index < context->total_values; index++) {
        float logit = context->logits->data[index];
        float target = context->targets->data[index];
        float sigmoid_value =
            vae_c_objective_stable_sigmoid_unchecked(logit);

        context->logits->grad[index] +=
            context->upstream_gradient *
            (sigmoid_value - target) /
            context->batch_size_f32;
    }
}


void vae_c_bce_with_logits_stats_init(
    VAECBCEWithLogitsStats *stats
)
{
    if (stats != NULL) {
        memset(stats, 0, sizeof(*stats));
    }
}


VAECOpStatus vae_c_bce_with_logits_forward(
    const VAECTensor *logits,
    const VAECTensor *targets,
    VAECBCEWithLogitsStats *stats
)
{
    VAECBCEWithLogitsStats candidate;
    VAECOpStatus status;
    size_t batch_size;
    size_t values_per_sample;
    size_t total_values;
    size_t index;
    double total_sum;

    if (stats == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    status = vae_c_objective_validate_pair(
        logits,
        targets,
        false,
        &batch_size,
        &values_per_sample,
        &total_values
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }

    if (vae_c_objective_stats_has_exact_alias(
            stats,
            logits,
            targets)) {

        return VAE_C_OP_INVALID_ARGUMENT;
    }

    total_sum = 0.0;

    for (index = 0; index < total_values; index++) {
        double logit = (double)logits->data[index];
        double target = (double)targets->data[index];
        double loss;
        double next_total_sum;

        if (!isfinite(logit) ||
            !isfinite(target) ||
            target < 0.0 || target > 1.0) {

            return VAE_C_OP_INVALID_STATE;
        }

        loss =
            fmax(logit, 0.0) -
            logit * target +
            log1p(exp(-fabs(logit)));
        next_total_sum = total_sum + loss;

        if (!isfinite(loss) || loss < 0.0 ||
            !isfinite(next_total_sum)) {

            return VAE_C_OP_INVALID_STATE;
        }

        total_sum = next_total_sum;
    }

    vae_c_bce_with_logits_stats_init(&candidate);
    candidate.batch_size = batch_size;
    candidate.values_per_sample = values_per_sample;
    candidate.total_values = total_values;
    candidate.total_sum = total_sum;
    candidate.mean_per_sample = total_sum / (double)batch_size;
    candidate.mean_per_value = total_sum / (double)total_values;

    if (!isfinite(candidate.mean_per_sample) ||
        !isfinite(candidate.mean_per_value)) {

        return VAE_C_OP_INVALID_STATE;
    }

    *stats = candidate;
    return VAE_C_OP_OK;
}


VAECOpStatus vae_c_bce_with_logits_backward(
    VAECTensor *logits,
    const VAECTensor *targets,
    float upstream_gradient
)
{
    VAECBCEBackwardContext context;
    VAECOpStatus status;

    status = vae_c_objective_bce_backward_prevalidate(
        logits,
        targets,
        upstream_gradient,
        &context
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }

    vae_c_objective_bce_backward_commit(&context);
    return VAE_C_OP_OK;
}


static bool vae_c_objective_kl_stats_has_exact_alias(
    const VAECKLGaussianStats *stats,
    const VAECTensor *mu,
    const VAECTensor *logvar
)
{
    const void *stats_pointer = (const void *)stats;

    return
        stats_pointer == (const void *)mu ||
        stats_pointer == (const void *)logvar ||
        stats_pointer == (const void *)mu->data ||
        stats_pointer == (const void *)mu->grad ||
        stats_pointer == (const void *)logvar->data ||
        stats_pointer == (const void *)logvar->grad;
}


static VAECOpStatus vae_c_objective_validate_kl_pair(
    const VAECTensor *mu,
    const VAECTensor *logvar,
    bool require_gradients,
    size_t *batch_size,
    size_t *latent_dim,
    size_t *total_values
)
{
    VAECOpStatus status;
    size_t candidate_total_values;

    if (mu == NULL || logvar == NULL ||
        batch_size == NULL || latent_dim == NULL ||
        total_values == NULL) {

        return VAE_C_OP_INVALID_ARGUMENT;
    }

    status = vae_c_objective_validate_tensor(mu);
    if (status != VAE_C_OP_OK) {
        return status;
    }

    status = vae_c_objective_validate_tensor(logvar);
    if (status != VAE_C_OP_OK) {
        return status;
    }

    if (vae_c_objective_tensors_have_exact_alias(mu, logvar)) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    if (mu->shape.rank != 2 ||
        logvar->shape.rank != 2 ||
        !vae_c_shape_equal(&mu->shape, &logvar->shape)) {

        return VAE_C_OP_SHAPE_MISMATCH;
    }

    if (vae_c_objective_size_multiply(
            mu->shape.dims[0],
            mu->shape.dims[1],
            &candidate_total_values) != 0) {

        return VAE_C_OP_OVERFLOW;
    }

    if (candidate_total_values != mu->shape.count ||
        candidate_total_values != logvar->shape.count) {

        return VAE_C_OP_INVALID_STATE;
    }

    if (require_gradients &&
        (!mu->has_grad || !logvar->has_grad)) {

        return VAE_C_OP_GRADIENT_NOT_AVAILABLE;
    }

    *batch_size = mu->shape.dims[0];
    *latent_dim = mu->shape.dims[1];
    *total_values = candidate_total_values;
    return VAE_C_OP_OK;
}


typedef struct {
    VAECTensor *mu;
    VAECTensor *logvar;
    size_t total_values;
    float upstream_scale;
} VAECKLBackwardContext;


static VAECOpStatus vae_c_objective_kl_backward_prevalidate(
    VAECTensor *mu,
    VAECTensor *logvar,
    float upstream_gradient,
    VAECKLBackwardContext *context
)
{
    VAECKLBackwardContext candidate;
    VAECOpStatus status;
    size_t batch_size;
    size_t latent_dim;
    size_t total_values;
    size_t index;
    float batch_size_f32;
    float upstream_scale;

    if (!isfinite(upstream_gradient)) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    if (context == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    status = vae_c_objective_validate_kl_pair(
        mu,
        logvar,
        true,
        &batch_size,
        &latent_dim,
        &total_values
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }

    batch_size_f32 = (float)batch_size;
    if (!isfinite(batch_size_f32) || !(batch_size_f32 > 0.0f)) {
        return VAE_C_OP_OVERFLOW;
    }

    upstream_scale = upstream_gradient / batch_size_f32;
    if (!isfinite(upstream_scale)) {
        return VAE_C_OP_INVALID_STATE;
    }

    for (index = 0; index < total_values; index++) {
        float mu_value = mu->data[index];
        float logvar_value = logvar->data[index];
        float exponential_minus_one;
        float mu_contribution;
        float logvar_contribution;
        float next_mu_gradient;
        float next_logvar_gradient;

        if (!isfinite(mu_value) || !isfinite(logvar_value) ||
            !isfinite(mu->grad[index]) ||
            !isfinite(logvar->grad[index])) {

            return VAE_C_OP_INVALID_STATE;
        }

        exponential_minus_one = expm1f(logvar_value);
        if (!isfinite(exponential_minus_one)) {
            return VAE_C_OP_INVALID_STATE;
        }

        mu_contribution = upstream_scale * mu_value;
        logvar_contribution =
            upstream_scale * 0.5f * exponential_minus_one;
        next_mu_gradient = mu->grad[index] + mu_contribution;
        next_logvar_gradient =
            logvar->grad[index] + logvar_contribution;

        if (!isfinite(mu_contribution) ||
            !isfinite(logvar_contribution) ||
            !isfinite(next_mu_gradient) ||
            !isfinite(next_logvar_gradient)) {

            return VAE_C_OP_INVALID_STATE;
        }
    }

    candidate.mu = mu;
    candidate.logvar = logvar;
    candidate.total_values = total_values;
    candidate.upstream_scale = upstream_scale;
    *context = candidate;
    (void)latent_dim;
    return VAE_C_OP_OK;
}


static void vae_c_objective_kl_backward_commit(
    const VAECKLBackwardContext *context
)
{
    size_t index;

    for (index = 0; index < context->total_values; index++) {
        float exponential_minus_one =
            expm1f(context->logvar->data[index]);

        context->mu->grad[index] +=
            context->upstream_scale * context->mu->data[index];
        context->logvar->grad[index] +=
            context->upstream_scale * 0.5f * exponential_minus_one;
    }
}


void vae_c_kl_gaussian_stats_init(
    VAECKLGaussianStats *stats
)
{
    if (stats != NULL) {
        memset(stats, 0, sizeof(*stats));
    }
}


VAECOpStatus vae_c_kl_gaussian_forward(
    const VAECTensor *mu,
    const VAECTensor *logvar,
    VAECKLGaussianStats *stats
)
{
    VAECKLGaussianStats candidate;
    VAECOpStatus status;
    size_t batch_size;
    size_t latent_dim;
    size_t total_values;
    size_t index;
    double total_sum;

    if (stats == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    status = vae_c_objective_validate_kl_pair(
        mu,
        logvar,
        false,
        &batch_size,
        &latent_dim,
        &total_values
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }

    if (vae_c_objective_kl_stats_has_exact_alias(
            stats,
            mu,
            logvar)) {

        return VAE_C_OP_INVALID_ARGUMENT;
    }

    total_sum = 0.0;

    for (index = 0; index < total_values; index++) {
        float mu_f32 = mu->data[index];
        float logvar_f32 = logvar->data[index];
        float exponential_minus_one_f32;
        double mu_f64;
        double logvar_f64;
        double exponential_minus_one_f64;
        double variance_term;
        double kl_element;
        double next_total_sum;

        if (!isfinite(mu_f32) || !isfinite(logvar_f32)) {
            return VAE_C_OP_INVALID_STATE;
        }

        exponential_minus_one_f32 = expm1f(logvar_f32);
        if (!isfinite(exponential_minus_one_f32)) {
            return VAE_C_OP_INVALID_STATE;
        }

        mu_f64 = (double)mu_f32;
        logvar_f64 = (double)logvar_f32;
        exponential_minus_one_f64 = expm1(logvar_f64);
        if (!isfinite(exponential_minus_one_f64)) {
            return VAE_C_OP_INVALID_STATE;
        }

        variance_term = exponential_minus_one_f64 - logvar_f64;
        kl_element = 0.5 * (mu_f64 * mu_f64 + variance_term);
        next_total_sum = total_sum + kl_element;

        if (!isfinite(variance_term) || variance_term < 0.0 ||
            !isfinite(kl_element) || kl_element < 0.0 ||
            !isfinite(next_total_sum)) {

            return VAE_C_OP_INVALID_STATE;
        }

        total_sum = next_total_sum;
    }

    vae_c_kl_gaussian_stats_init(&candidate);
    candidate.batch_size = batch_size;
    candidate.latent_dim = latent_dim;
    candidate.total_values = total_values;
    candidate.total_sum = total_sum;
    candidate.mean_per_sample = total_sum / (double)batch_size;
    candidate.mean_per_dimension = total_sum / (double)total_values;

    if (!isfinite(candidate.mean_per_sample) ||
        !isfinite(candidate.mean_per_dimension)) {

        return VAE_C_OP_INVALID_STATE;
    }

    *stats = candidate;
    return VAE_C_OP_OK;
}


VAECOpStatus vae_c_kl_gaussian_backward(
    VAECTensor *mu,
    VAECTensor *logvar,
    float upstream_gradient
)
{
    VAECKLBackwardContext context;
    VAECOpStatus status;

    status = vae_c_objective_kl_backward_prevalidate(
        mu,
        logvar,
        upstream_gradient,
        &context
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }

    vae_c_objective_kl_backward_commit(&context);
    return VAE_C_OP_OK;
}


static bool vae_c_objective_vae_inputs_have_exact_alias(
    const VAECTensor *logits,
    const VAECTensor *targets,
    const VAECTensor *mu,
    const VAECTensor *logvar
)
{
    return
        vae_c_objective_tensors_have_exact_alias(logits, targets) ||
        vae_c_objective_tensors_have_exact_alias(logits, mu) ||
        vae_c_objective_tensors_have_exact_alias(logits, logvar) ||
        vae_c_objective_tensors_have_exact_alias(targets, mu) ||
        vae_c_objective_tensors_have_exact_alias(targets, logvar) ||
        vae_c_objective_tensors_have_exact_alias(mu, logvar);
}


static bool vae_c_objective_pointer_has_exact_alias_with_tensor(
    const void *pointer,
    const VAECTensor *tensor
)
{
    return
        pointer == (const void *)tensor ||
        pointer == (const void *)tensor->data ||
        pointer == (const void *)tensor->grad;
}


static bool vae_c_objective_vae_stats_has_exact_alias(
    const VAECVAELossStats *stats,
    const VAECTensor *logits,
    const VAECTensor *targets,
    const VAECTensor *mu,
    const VAECTensor *logvar
)
{
    const void *stats_pointer = (const void *)stats;

    return
        vae_c_objective_pointer_has_exact_alias_with_tensor(
            stats_pointer,
            logits) ||
        vae_c_objective_pointer_has_exact_alias_with_tensor(
            stats_pointer,
            targets) ||
        vae_c_objective_pointer_has_exact_alias_with_tensor(
            stats_pointer,
            mu) ||
        vae_c_objective_pointer_has_exact_alias_with_tensor(
            stats_pointer,
            logvar);
}


static VAECOpStatus vae_c_objective_validate_vae_inputs(
    const VAECTensor *logits,
    const VAECTensor *targets,
    const VAECTensor *mu,
    const VAECTensor *logvar,
    bool require_gradients,
    size_t *batch_size
)
{
    VAECOpStatus status;
    size_t reconstruction_batch_size;
    size_t values_per_sample;
    size_t reconstruction_total_values;
    size_t kl_batch_size;
    size_t latent_dim;
    size_t kl_total_values;

    if (batch_size == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    status = vae_c_objective_validate_pair(
        logits,
        targets,
        require_gradients,
        &reconstruction_batch_size,
        &values_per_sample,
        &reconstruction_total_values
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }

    status = vae_c_objective_validate_kl_pair(
        mu,
        logvar,
        require_gradients,
        &kl_batch_size,
        &latent_dim,
        &kl_total_values
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }

    if (vae_c_objective_vae_inputs_have_exact_alias(
            logits,
            targets,
            mu,
            logvar)) {

        return VAE_C_OP_INVALID_ARGUMENT;
    }

    if (reconstruction_batch_size != kl_batch_size) {
        return VAE_C_OP_SHAPE_MISMATCH;
    }

    *batch_size = reconstruction_batch_size;
    (void)values_per_sample;
    (void)reconstruction_total_values;
    (void)latent_dim;
    (void)kl_total_values;
    return VAE_C_OP_OK;
}


static bool vae_c_objective_doubles_equivalent(
    double left,
    double right
)
{
    double scale;

    if (!isfinite(left) || !isfinite(right)) {
        return false;
    }

    scale = fmax(fmax(fabs(left), fabs(right)), 1.0);
    return fabs(left - right) <= 8.0 * DBL_EPSILON * scale;
}


void vae_c_vae_loss_stats_init(
    VAECVAELossStats *stats
)
{
    if (stats != NULL) {
        memset(stats, 0, sizeof(*stats));
    }
}


VAECOpStatus vae_c_vae_loss_forward(
    const VAECTensor *logits,
    const VAECTensor *targets,
    const VAECTensor *mu,
    const VAECTensor *logvar,
    VAECVAELossStats *stats
)
{
    VAECBCEWithLogitsStats reconstruction_temp;
    VAECKLGaussianStats kl_temp;
    VAECVAELossStats combined_temp;
    VAECOpStatus status;
    size_t batch_size;
    double weighted_kl_total;
    double component_mean_per_sample;

    if (stats == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    status = vae_c_objective_validate_vae_inputs(
        logits,
        targets,
        mu,
        logvar,
        false,
        &batch_size
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }

    if (vae_c_objective_vae_stats_has_exact_alias(
            stats,
            logits,
            targets,
            mu,
            logvar)) {

        return VAE_C_OP_INVALID_ARGUMENT;
    }

    if (!isfinite(VAE_C_VAE_LOSS_BETA_KL_V1) ||
        VAE_C_VAE_LOSS_BETA_KL_V1 != 1.0f) {

        return VAE_C_OP_INVALID_STATE;
    }

    vae_c_bce_with_logits_stats_init(&reconstruction_temp);
    vae_c_kl_gaussian_stats_init(&kl_temp);
    vae_c_vae_loss_stats_init(&combined_temp);

    status = vae_c_bce_with_logits_forward(
        logits,
        targets,
        &reconstruction_temp
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }

    status = vae_c_kl_gaussian_forward(mu, logvar, &kl_temp);
    if (status != VAE_C_OP_OK) {
        return status;
    }

    if (reconstruction_temp.batch_size != batch_size ||
        kl_temp.batch_size != batch_size ||
        reconstruction_temp.batch_size != kl_temp.batch_size) {

        return VAE_C_OP_SHAPE_MISMATCH;
    }

    weighted_kl_total =
        (double)VAE_C_VAE_LOSS_BETA_KL_V1 * kl_temp.total_sum;
    combined_temp.total_sum =
        reconstruction_temp.total_sum + weighted_kl_total;
    combined_temp.mean_per_sample =
        combined_temp.total_sum / (double)batch_size;
    component_mean_per_sample =
        reconstruction_temp.mean_per_sample +
        (double)VAE_C_VAE_LOSS_BETA_KL_V1 *
            kl_temp.mean_per_sample;

    if (!isfinite(weighted_kl_total) ||
        !isfinite(combined_temp.total_sum) ||
        !isfinite(combined_temp.mean_per_sample) ||
        !vae_c_objective_doubles_equivalent(
            combined_temp.mean_per_sample,
            component_mean_per_sample)) {

        return VAE_C_OP_INVALID_STATE;
    }

    combined_temp.batch_size = batch_size;
    combined_temp.beta_kl = VAE_C_VAE_LOSS_BETA_KL_V1;
    combined_temp.reconstruction = reconstruction_temp;
    combined_temp.kl = kl_temp;
    *stats = combined_temp;
    return VAE_C_OP_OK;
}


VAECOpStatus vae_c_vae_loss_backward(
    VAECTensor *logits,
    const VAECTensor *targets,
    VAECTensor *mu,
    VAECTensor *logvar,
    float upstream_gradient
)
{
    VAECBCEBackwardContext reconstruction_context;
    VAECKLBackwardContext kl_context;
    VAECOpStatus status;
    size_t batch_size;
    float upstream_kl;

    if (!isfinite(upstream_gradient)) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    status = vae_c_objective_validate_vae_inputs(
        logits,
        targets,
        mu,
        logvar,
        true,
        &batch_size
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }

    if (!isfinite(VAE_C_VAE_LOSS_BETA_KL_V1) ||
        VAE_C_VAE_LOSS_BETA_KL_V1 != 1.0f) {

        return VAE_C_OP_INVALID_STATE;
    }

    upstream_kl =
        upstream_gradient * VAE_C_VAE_LOSS_BETA_KL_V1;
    if (!isfinite(upstream_kl)) {
        return VAE_C_OP_INVALID_STATE;
    }

    status = vae_c_objective_bce_backward_prevalidate(
        logits,
        targets,
        upstream_gradient,
        &reconstruction_context
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }

    status = vae_c_objective_kl_backward_prevalidate(
        mu,
        logvar,
        upstream_kl,
        &kl_context
    );
    if (status != VAE_C_OP_OK) {
        return status;
    }

    vae_c_objective_bce_backward_commit(&reconstruction_context);
    vae_c_objective_kl_backward_commit(&kl_context);
    (void)batch_size;
    return VAE_C_OP_OK;
}
