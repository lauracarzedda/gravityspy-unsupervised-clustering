#include "vae_c_adam.h"

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>


typedef struct {
    double bias_correction1;
    double bias_correction2;
    double next_beta1_power;
    double next_beta2_power;
    uint64_t next_step;
} VAECAdamStepState;


static int adam_size_multiply(
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


static VAECOpStatus adam_validate_shape(const VAECShape *shape)
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

    if (shape->count == 0 || shape->count != canonical.count) {
        return VAE_C_OP_INVALID_STATE;
    }

    for (index = 0; index < VAE_C_TENSOR_MAX_RANK; index++) {
        if (shape->dims[index] != canonical.dims[index] ||
            shape->strides[index] != canonical.strides[index]) {

            return VAE_C_OP_INVALID_STATE;
        }
    }

    return VAE_C_OP_OK;
}


static VAECOpStatus adam_validate_parameter(
    const VAECParameter *parameter
)
{
    VAECOpStatus status;
    size_t total_elements;
    size_t expected_bytes;

    if (parameter == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    status = adam_validate_shape(&parameter->shape);
    if (status != VAE_C_OP_OK) {
        return status;
    }

    if (parameter->values == NULL ||
        parameter->gradients == NULL ||
        parameter->adam_m == NULL ||
        parameter->adam_v == NULL ||
        parameter->owned_block == NULL ||
        parameter->values != parameter->owned_block ||
        parameter->gradients !=
            parameter->values + parameter->shape.count ||
        parameter->adam_m !=
            parameter->gradients + parameter->shape.count ||
        parameter->adam_v !=
            parameter->adam_m + parameter->shape.count) {

        return VAE_C_OP_INVALID_STATE;
    }

    if (adam_size_multiply(
            parameter->shape.count,
            4,
            &total_elements) != 0 ||
        adam_size_multiply(
            total_elements,
            sizeof(float),
            &expected_bytes) != 0 ||
        parameter->owned_bytes != expected_bytes) {

        return VAE_C_OP_INVALID_STATE;
    }

    return VAE_C_OP_OK;
}


static bool adam_parameters_have_exact_alias(
    const VAECParameter *left,
    const VAECParameter *right
)
{
    const float *left_regions[5];
    const float *right_regions[5];
    size_t left_index;
    size_t right_index;

    if (left == right) {
        return true;
    }

    left_regions[0] = left->values;
    left_regions[1] = left->gradients;
    left_regions[2] = left->adam_m;
    left_regions[3] = left->adam_v;
    left_regions[4] = left->owned_block;

    right_regions[0] = right->values;
    right_regions[1] = right->gradients;
    right_regions[2] = right->adam_m;
    right_regions[3] = right->adam_v;
    right_regions[4] = right->owned_block;

    for (left_index = 0; left_index < 5; left_index++) {
        for (right_index = 0; right_index < 5; right_index++) {
            if (left_regions[left_index] ==
                    right_regions[right_index]) {

                return true;
            }
        }
    }

    return false;
}


static bool adam_optimizer_aliases_parameter(
    const VAECAdam *optimizer,
    const VAECParameter *parameter
)
{
    const void *optimizer_address = (const void *)optimizer;

    return
        optimizer_address == (const void *)parameter->values ||
        optimizer_address == (const void *)parameter->gradients ||
        optimizer_address == (const void *)parameter->adam_m ||
        optimizer_address == (const void *)parameter->adam_v ||
        optimizer_address == (const void *)parameter->owned_block;
}


static VAECOpStatus adam_validate_optimizer(
    const VAECAdam *optimizer
)
{
    if (optimizer == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    if (!optimizer->configured ||
        !isfinite(optimizer->learning_rate) ||
        optimizer->learning_rate <= 0.0 ||
        optimizer->learning_rate != VAE_C_ADAM_LEARNING_RATE_V1 ||
        !isfinite(optimizer->beta1) ||
        optimizer->beta1 <= 0.0 ||
        optimizer->beta1 >= 1.0 ||
        optimizer->beta1 != VAE_C_ADAM_BETA1_V1 ||
        !isfinite(optimizer->beta2) ||
        optimizer->beta2 <= 0.0 ||
        optimizer->beta2 >= 1.0 ||
        optimizer->beta2 != VAE_C_ADAM_BETA2_V1 ||
        !isfinite(optimizer->adam_epsilon) ||
        optimizer->adam_epsilon <= 0.0 ||
        optimizer->adam_epsilon != VAE_C_ADAM_EPSILON_V1 ||
        !isfinite(optimizer->beta1_power) ||
        optimizer->beta1_power < 0.0 ||
        optimizer->beta1_power > 1.0 ||
        !isfinite(optimizer->beta2_power) ||
        optimizer->beta2_power < 0.0 ||
        optimizer->beta2_power > 1.0 ||
        optimizer->step == UINT64_MAX) {

        return VAE_C_OP_INVALID_STATE;
    }

    if (optimizer->step == UINT64_C(0)) {
        if (optimizer->beta1_power != 1.0 ||
            optimizer->beta2_power != 1.0) {

            return VAE_C_OP_INVALID_STATE;
        }
    } else if (optimizer->beta1_power >= 1.0 ||
               optimizer->beta2_power >= 1.0) {

        return VAE_C_OP_INVALID_STATE;
    }

    return VAE_C_OP_OK;
}


static VAECOpStatus adam_prepare_step_state(
    const VAECAdam *optimizer,
    VAECAdamStepState *state
)
{
    VAECAdamStepState candidate;

    if (optimizer == NULL || state == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    candidate.next_step = optimizer->step + UINT64_C(1);
    candidate.next_beta1_power =
        optimizer->beta1_power * optimizer->beta1;
    candidate.next_beta2_power =
        optimizer->beta2_power * optimizer->beta2;
    candidate.bias_correction1 =
        1.0 - candidate.next_beta1_power;
    candidate.bias_correction2 =
        1.0 - candidate.next_beta2_power;

    if (!isfinite(candidate.next_beta1_power) ||
        candidate.next_beta1_power < 0.0 ||
        candidate.next_beta1_power > 1.0 ||
        !isfinite(candidate.next_beta2_power) ||
        candidate.next_beta2_power < 0.0 ||
        candidate.next_beta2_power > 1.0 ||
        !isfinite(candidate.bias_correction1) ||
        candidate.bias_correction1 <= 0.0 ||
        !isfinite(candidate.bias_correction2) ||
        candidate.bias_correction2 <= 0.0) {

        return VAE_C_OP_INVALID_STATE;
    }

    *state = candidate;
    return VAE_C_OP_OK;
}


static VAECOpStatus adam_compute_candidate(
    const VAECAdam *optimizer,
    double bias_correction1,
    double bias_correction2,
    float theta,
    float gradient,
    float m_old,
    float v_old,
    float *theta_candidate,
    float *m_candidate,
    float *v_candidate
)
{
    double m_candidate_d;
    double v_candidate_d;
    double m_hat;
    double v_hat;
    double root_v_hat;
    double denominator;
    double update;
    double theta_candidate_d;
    float rounded_m;
    float rounded_v;
    float rounded_theta;

    if (optimizer == NULL ||
        theta_candidate == NULL ||
        m_candidate == NULL ||
        v_candidate == NULL) {

        return VAE_C_OP_INVALID_ARGUMENT;
    }

    m_candidate_d =
        optimizer->beta1 * (double)m_old +
        (1.0 - optimizer->beta1) * (double)gradient;
    v_candidate_d =
        optimizer->beta2 * (double)v_old +
        (1.0 - optimizer->beta2) *
            (double)gradient * (double)gradient;

    if (!isfinite(m_candidate_d) ||
        m_candidate_d < -(double)FLT_MAX ||
        m_candidate_d > (double)FLT_MAX ||
        !isfinite(v_candidate_d) ||
        v_candidate_d < 0.0 ||
        v_candidate_d > (double)FLT_MAX) {

        return VAE_C_OP_INVALID_STATE;
    }

    rounded_m = (float)m_candidate_d;
    rounded_v = (float)v_candidate_d;
    if (!isfinite(rounded_m) ||
        !isfinite(rounded_v) ||
        rounded_v < 0.0f) {

        return VAE_C_OP_INVALID_STATE;
    }

    m_hat = (double)rounded_m / bias_correction1;
    v_hat = (double)rounded_v / bias_correction2;
    if (!isfinite(m_hat) ||
        !isfinite(v_hat) ||
        v_hat < 0.0) {

        return VAE_C_OP_INVALID_STATE;
    }

    root_v_hat = sqrt(v_hat);
    denominator = root_v_hat + optimizer->adam_epsilon;
    if (!isfinite(root_v_hat) ||
        !isfinite(denominator) ||
        denominator <= 0.0) {

        return VAE_C_OP_INVALID_STATE;
    }

    update = optimizer->learning_rate * m_hat / denominator;
    theta_candidate_d = (double)theta - update;
    if (!isfinite(update) ||
        !isfinite(theta_candidate_d) ||
        theta_candidate_d < -(double)FLT_MAX ||
        theta_candidate_d > (double)FLT_MAX) {

        return VAE_C_OP_INVALID_STATE;
    }

    rounded_theta = (float)theta_candidate_d;
    if (!isfinite(rounded_theta)) {
        return VAE_C_OP_INVALID_STATE;
    }

    *theta_candidate = rounded_theta;
    *m_candidate = rounded_m;
    *v_candidate = rounded_v;
    return VAE_C_OP_OK;
}


void vae_c_adam_init(VAECAdam *optimizer)
{
    if (optimizer != NULL) {
        memset(optimizer, 0, sizeof(*optimizer));
    }
}


void vae_c_adam_free(VAECAdam *optimizer)
{
    vae_c_adam_init(optimizer);
}


bool vae_c_adam_is_empty(const VAECAdam *optimizer)
{
    return
        optimizer != NULL &&
        optimizer->learning_rate == 0.0 &&
        optimizer->beta1 == 0.0 &&
        optimizer->beta2 == 0.0 &&
        optimizer->adam_epsilon == 0.0 &&
        optimizer->step == UINT64_C(0) &&
        optimizer->beta1_power == 0.0 &&
        optimizer->beta2_power == 0.0 &&
        !optimizer->configured;
}


bool vae_c_adam_is_configured(const VAECAdam *optimizer)
{
    return adam_validate_optimizer(optimizer) == VAE_C_OP_OK;
}


VAECOpStatus vae_c_adam_configure_v1(VAECAdam *optimizer)
{
    VAECAdam candidate;

    if (optimizer == NULL) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    if (!vae_c_adam_is_empty(optimizer)) {
        return VAE_C_OP_OBJECT_NOT_EMPTY;
    }

    vae_c_adam_init(&candidate);
    candidate.learning_rate = VAE_C_ADAM_LEARNING_RATE_V1;
    candidate.beta1 = VAE_C_ADAM_BETA1_V1;
    candidate.beta2 = VAE_C_ADAM_BETA2_V1;
    candidate.adam_epsilon = VAE_C_ADAM_EPSILON_V1;
    candidate.step = UINT64_C(0);
    candidate.beta1_power = 1.0;
    candidate.beta2_power = 1.0;
    candidate.configured = true;

    *optimizer = candidate;
    return VAE_C_OP_OK;
}


VAECOpStatus vae_c_adam_step(
    VAECAdam *optimizer,
    VAECParameter *const *parameters,
    size_t parameter_count
)
{
    VAECAdamStepState step_state;
    VAECOpStatus status;
    size_t parameter_index;
    size_t other_index;
    size_t element_index;

    status = adam_validate_optimizer(optimizer);
    if (status != VAE_C_OP_OK) {
        return status;
    }

    if (parameters == NULL || parameter_count == 0) {
        return VAE_C_OP_INVALID_ARGUMENT;
    }

    for (parameter_index = 0;
         parameter_index < parameter_count;
         parameter_index++) {

        status = adam_validate_parameter(parameters[parameter_index]);
        if (status != VAE_C_OP_OK) {
            return status;
        }

        if (adam_optimizer_aliases_parameter(
                optimizer,
                parameters[parameter_index])) {

            return VAE_C_OP_INVALID_ARGUMENT;
        }
    }

    for (parameter_index = 0;
         parameter_index < parameter_count;
         parameter_index++) {

        for (other_index = parameter_index + 1;
             other_index < parameter_count;
             other_index++) {

            if (adam_parameters_have_exact_alias(
                    parameters[parameter_index],
                    parameters[other_index])) {

                return VAE_C_OP_INVALID_ARGUMENT;
            }
        }
    }

    status = adam_prepare_step_state(optimizer, &step_state);
    if (status != VAE_C_OP_OK) {
        return status;
    }

    for (parameter_index = 0;
         parameter_index < parameter_count;
         parameter_index++) {

        const VAECParameter *parameter = parameters[parameter_index];

        for (element_index = 0;
             element_index < parameter->shape.count;
             element_index++) {

            if (!isfinite(parameter->values[element_index]) ||
                !isfinite(parameter->gradients[element_index]) ||
                !isfinite(parameter->adam_m[element_index]) ||
                !isfinite(parameter->adam_v[element_index]) ||
                parameter->adam_v[element_index] < 0.0f) {

                return VAE_C_OP_INVALID_STATE;
            }
        }
    }

    /*
     * PROJECT: complete numeric prevalidation.  No model or optimizer
     * storage has been written when this phase returns an error.
     */
    for (parameter_index = 0;
         parameter_index < parameter_count;
         parameter_index++) {

        const VAECParameter *parameter = parameters[parameter_index];

        for (element_index = 0;
             element_index < parameter->shape.count;
             element_index++) {

            float theta_candidate;
            float m_candidate;
            float v_candidate;

            status = adam_compute_candidate(
                optimizer,
                step_state.bias_correction1,
                step_state.bias_correction2,
                parameter->values[element_index],
                parameter->gradients[element_index],
                parameter->adam_m[element_index],
                parameter->adam_v[element_index],
                &theta_candidate,
                &m_candidate,
                &v_candidate
            );
            if (status != VAE_C_OP_OK) {
                return status;
            }
        }
    }

    /*
     * PROJECT: the shared pure helper is deliberately recomputed here.
     * All possible failures were exhausted above, so commit has no error
     * return and gradients remain untouched.
     */
    for (parameter_index = 0;
         parameter_index < parameter_count;
         parameter_index++) {

        VAECParameter *parameter = parameters[parameter_index];

        for (element_index = 0;
             element_index < parameter->shape.count;
             element_index++) {

            float theta_candidate;
            float m_candidate;
            float v_candidate;

            (void)adam_compute_candidate(
                optimizer,
                step_state.bias_correction1,
                step_state.bias_correction2,
                parameter->values[element_index],
                parameter->gradients[element_index],
                parameter->adam_m[element_index],
                parameter->adam_v[element_index],
                &theta_candidate,
                &m_candidate,
                &v_candidate
            );

            parameter->adam_m[element_index] = m_candidate;
            parameter->adam_v[element_index] = v_candidate;
            parameter->values[element_index] = theta_candidate;
        }
    }

    optimizer->step = step_state.next_step;
    optimizer->beta1_power = step_state.next_beta1_power;
    optimizer->beta2_power = step_state.next_beta2_power;
    return VAE_C_OP_OK;
}
