#include "vae_c_iic_pair_train.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define FIXTURE_INPUT_DIM 3
#define FIXTURE_ACTIVE_HIDDEN 4
#define PAIR_PARAMETER_COUNT 10

typedef struct {
    VAECIICPair pair;
    VAECIICPairWorkspace workspace;
    VAECAdam optimizer;
    VAECTensor input_a;
    VAECTensor input_b;
} TrainFixture;

typedef struct {
    float main_mean[VAE_C_IIC_HEAD_HIDDEN_DIM];
    float main_variance[VAE_C_IIC_HEAD_HIDDEN_DIM];
    float over_mean[VAE_C_IIC_HEAD_HIDDEN_DIM];
    float over_variance[VAE_C_IIC_HEAD_HIDDEN_DIM];
    uint64_t main_count;
    uint64_t over_count;
} BNState;

static const float fixture_input_a[12] = {
    0.5f, -1.0f, 0.25f,
    -0.75f, 0.5f, 1.0f,
    1.25f, 0.25f, -0.5f,
    -0.25f, -0.75f, 0.75f
};

static int report(const char *name, int passed)
{
    printf("[%s] %s\n", passed ? "PASS" : "FAIL", name);
    return passed ? 0 : 1;
}

static void registry(VAECIICPair *pair, VAECParameter *parameters[10])
{
    VAECIICHead *main_head = &pair->main_head;
    VAECIICHead *over_head = &pair->overcluster_head;
    parameters[0] = &main_head->linear1.weights;
    parameters[1] = &main_head->batch_norm.gamma;
    parameters[2] = &main_head->batch_norm.beta;
    parameters[3] = &main_head->linear2.weights;
    parameters[4] = &main_head->linear2.bias;
    parameters[5] = &over_head->linear1.weights;
    parameters[6] = &over_head->batch_norm.gamma;
    parameters[7] = &over_head->batch_norm.beta;
    parameters[8] = &over_head->linear2.weights;
    parameters[9] = &over_head->linear2.bias;
}

static void init_fixture(TrainFixture *fixture)
{
    memset(fixture, 0, sizeof(*fixture));
    vae_c_iic_pair_init(&fixture->pair);
    vae_c_iic_pair_workspace_init(&fixture->workspace);
    vae_c_adam_init(&fixture->optimizer);
    vae_c_tensor_init(&fixture->input_a);
    vae_c_tensor_init(&fixture->input_b);
}

static void free_fixture(TrainFixture *fixture)
{
    vae_c_tensor_free(&fixture->input_b);
    vae_c_tensor_free(&fixture->input_a);
    vae_c_adam_free(&fixture->optimizer);
    vae_c_iic_pair_workspace_free(&fixture->workspace);
    vae_c_iic_pair_free(&fixture->pair);
}

static void initialize_head(VAECIICHead *head, int is_over)
{
    static const float main_linear1[4][3] = {
        {0.5f, -0.25f, 0.125f},
        {-0.375f, 0.25f, 0.5f},
        {0.25f, 0.75f, -0.5f},
        {-0.125f, 0.5f, 0.375f}
    };
    static const float main_gamma[4] = {
        1.125f, -0.875f, 0.75f, 1.25f
    };
    static const float main_beta[4] = {
        0.125f, 0.25f, -0.0625f, -0.125f
    };
    static const float main_mean[4] = {
        0.25f, -0.125f, 0.375f, -0.25f
    };
    static const float main_variance[4] = {
        1.5f, 0.75f, 2.0f, 1.25f
    };
    static const float main_linear2[3][4] = {
        {0.5f, -0.25f, 0.125f, 0.0625f},
        {-0.125f, 0.375f, 0.25f, -0.5f},
        {0.25f, 0.125f, -0.375f, 0.625f}
    };
    static const float main_bias[3] = {
        0.0625f, -0.125f, 0.1875f
    };
    size_t hidden;
    size_t feature;
    size_t output;

    /* Identical to the explicit C5B3 pair fixture, not official init. */
    for (hidden = 0; hidden < VAE_C_IIC_HEAD_HIDDEN_DIM; hidden++) {
        head->batch_norm.gamma.values[hidden] = 1.0f;
        head->batch_norm.beta.values[hidden] = -1.0f;
        head->batch_norm.running_mean[hidden] = 0.0f;
        head->batch_norm.running_variance[hidden] = 1.0f;
        if (hidden >= FIXTURE_ACTIVE_HIDDEN) {
            continue;
        }
        if (!is_over) {
            memcpy(head->linear1.weights.values + hidden * FIXTURE_INPUT_DIM,
                   main_linear1[hidden], sizeof(main_linear1[hidden]));
            head->batch_norm.gamma.values[hidden] = main_gamma[hidden];
            head->batch_norm.beta.values[hidden] = main_beta[hidden];
            head->batch_norm.running_mean[hidden] = main_mean[hidden];
            head->batch_norm.running_variance[hidden] =
                main_variance[hidden];
        } else {
            for (feature = 0; feature < FIXTURE_INPUT_DIM; feature++) {
                head->linear1.weights.values[
                    hidden * FIXTURE_INPUT_DIM + feature] =
                    (float)((int)((3 * hidden + feature) % 7) - 3) *
                    0.125f;
            }
            head->batch_norm.gamma.values[hidden] =
                1.0f + (float)((int)(hidden % 3) - 1) * 0.125f;
            head->batch_norm.beta.values[hidden] =
                (float)((int)((2 * hidden) % 5) - 2) * 0.125f;
            head->batch_norm.running_mean[hidden] =
                (float)((int)((hidden + 1) % 4) - 1) * 0.125f;
            head->batch_norm.running_variance[hidden] =
                1.0f + (float)(hidden + 1) * 0.25f;
        }
    }
    for (output = 0; output < head->output_dim; output++) {
        if (!is_over && output < 3) {
            memcpy(head->linear2.weights.values +
                       output * VAE_C_IIC_HEAD_HIDDEN_DIM,
                   main_linear2[output], sizeof(main_linear2[output]));
            head->linear2.bias.values[output] = main_bias[output];
        } else {
            head->linear2.bias.values[output] =
                (float)((int)(output % 3) - 1) * 0.0625f;
            for (hidden = 0; hidden < FIXTURE_ACTIVE_HIDDEN; hidden++) {
                head->linear2.weights.values[
                    output * VAE_C_IIC_HEAD_HIDDEN_DIM + hidden] =
                    (float)((int)((5 * output + 2 * hidden) % 9) - 4) *
                    0.125f;
            }
        }
    }
}

static int allocate_fixture(
    TrainFixture *fixture,
    size_t batch,
    size_t capacity,
    size_t main_dim,
    size_t over_dim
)
{
    const size_t input_dims[2] = {batch, FIXTURE_INPUT_DIM};
    VAECShape input_shape;
    size_t index;

    init_fixture(fixture);
    if (vae_c_iic_pair_allocate(
            &fixture->pair, FIXTURE_INPUT_DIM, main_dim,
            over_dim) != VAE_C_OP_OK ||
        vae_c_iic_pair_workspace_allocate(
            &fixture->pair, capacity,
            &fixture->workspace) != VAE_C_OP_OK ||
        vae_c_adam_configure_v1(&fixture->optimizer) != VAE_C_OP_OK ||
        vae_c_shape_make(2, input_dims, &input_shape) !=
            VAE_C_STORAGE_OK ||
        vae_c_tensor_allocate(
            &fixture->input_a, &input_shape,
            true) != VAE_C_STORAGE_OK ||
        vae_c_tensor_allocate(
            &fixture->input_b, &input_shape,
            true) != VAE_C_STORAGE_OK) {
        return 0;
    }
    initialize_head(&fixture->pair.main_head, 0);
    initialize_head(&fixture->pair.overcluster_head, 1);
    for (index = 0; index < input_shape.count; index++) {
        float value = index < 12 ? fixture_input_a[index] :
            (float)((int)(index % 11) - 5) * 0.1875f;
        fixture->input_a.data[index] = value;
        fixture->input_b.data[index] = -0.375f * value +
            (float)((int)(index % 5) - 2) * 0.125f;
    }
    return 1;
}

static void snapshot_bn(const VAECIICPair *pair, BNState *state)
{
    const size_t bytes = VAE_C_IIC_HEAD_HIDDEN_DIM * sizeof(float);
    memcpy(state->main_mean, pair->main_head.batch_norm.running_mean,
           bytes);
    memcpy(state->main_variance,
           pair->main_head.batch_norm.running_variance, bytes);
    memcpy(state->over_mean,
           pair->overcluster_head.batch_norm.running_mean, bytes);
    memcpy(state->over_variance,
           pair->overcluster_head.batch_norm.running_variance, bytes);
    state->main_count = pair->main_head.num_batches_tracked;
    state->over_count = pair->overcluster_head.num_batches_tracked;
}

static int bn_matches(const VAECIICPair *pair, const BNState *state)
{
    const size_t bytes = VAE_C_IIC_HEAD_HIDDEN_DIM * sizeof(float);
    return memcmp(state->main_mean,
                  pair->main_head.batch_norm.running_mean, bytes) == 0 &&
        memcmp(state->main_variance,
               pair->main_head.batch_norm.running_variance, bytes) == 0 &&
        memcmp(state->over_mean,
               pair->overcluster_head.batch_norm.running_mean, bytes) == 0 &&
        memcmp(state->over_variance,
               pair->overcluster_head.batch_norm.running_variance, bytes) ==
            0 &&
        state->main_count == pair->main_head.num_batches_tracked &&
        state->over_count == pair->overcluster_head.num_batches_tracked;
}

static int inputs_unchanged(const TrainFixture *fixture)
{
    size_t index;
    for (index = 0; index < fixture->input_a.shape.count; index++) {
        float expected_a = index < 12 ? fixture_input_a[index] :
            (float)((int)(index % 11) - 5) * 0.1875f;
        float expected_b = -0.375f * expected_a +
            (float)((int)(index % 5) - 2) * 0.125f;
        if (fixture->input_a.data[index] != expected_a ||
            fixture->input_b.data[index] != expected_b) {
            return 0;
        }
    }
    return 1;
}

static int optimizer_equal(const VAECAdam *left, const VAECAdam *right)
{
    return left->learning_rate == right->learning_rate &&
        left->beta1 == right->beta1 && left->beta2 == right->beta2 &&
        left->adam_epsilon == right->adam_epsilon &&
        left->step == right->step &&
        left->beta1_power == right->beta1_power &&
        left->beta2_power == right->beta2_power &&
        left->configured == right->configured;
}

static int pair_parameters_equal(
    VAECIICPair *left,
    VAECIICPair *right,
    int compare_gradients
)
{
    VAECParameter *left_parameters[PAIR_PARAMETER_COUNT];
    VAECParameter *right_parameters[PAIR_PARAMETER_COUNT];
    size_t parameter_index;
    registry(left, left_parameters);
    registry(right, right_parameters);
    for (parameter_index = 0; parameter_index < PAIR_PARAMETER_COUNT;
         parameter_index++) {
        VAECParameter *a = left_parameters[parameter_index];
        VAECParameter *b = right_parameters[parameter_index];
        const size_t bytes = a->shape.count * sizeof(float);
        if (a->shape.count != b->shape.count ||
            memcmp(a->values, b->values, bytes) != 0 ||
            memcmp(a->adam_m, b->adam_m, bytes) != 0 ||
            memcmp(a->adam_v, b->adam_v, bytes) != 0 ||
            (compare_gradients &&
             memcmp(a->gradients, b->gradients, bytes) != 0)) {
            return 0;
        }
    }
    return 1;
}

static int parameters_finite(VAECIICPair *pair)
{
    VAECParameter *parameters[PAIR_PARAMETER_COUNT];
    size_t p;
    registry(pair, parameters);
    for (p = 0; p < PAIR_PARAMETER_COUNT; p++) {
        VAECParameter *parameter = parameters[p];
        size_t i;
        for (i = 0; i < parameter->shape.count; i++) {
            if (!isfinite(parameter->values[i]) ||
                !isfinite(parameter->gradients[i]) ||
                !isfinite(parameter->adam_m[i]) ||
                !isfinite(parameter->adam_v[i]) ||
                parameter->adam_v[i] < 0.0f) {
                return 0;
            }
        }
    }
    return 1;
}

static int moments_zero(VAECIICPair *pair)
{
    VAECParameter *parameters[PAIR_PARAMETER_COUNT];
    size_t p;
    registry(pair, parameters);
    for (p = 0; p < PAIR_PARAMETER_COUNT; p++) {
        size_t i;
        for (i = 0; i < parameters[p]->shape.count; i++) {
            if (parameters[p]->adam_m[i] != 0.0f ||
                parameters[p]->adam_v[i] != 0.0f) {
                return 0;
            }
        }
    }
    return 1;
}

static int each_head_updated(
    VAECIICPair *updated,
    VAECIICPair *initial
)
{
    VAECParameter *now[PAIR_PARAMETER_COUNT];
    VAECParameter *before[PAIR_PARAMETER_COUNT];
    size_t p;
    size_t branch_changes[2] = {0, 0};
    size_t branch_moments[2] = {0, 0};
    registry(updated, now);
    registry(initial, before);
    for (p = 0; p < PAIR_PARAMETER_COUNT; p++) {
        size_t i;
        for (i = 0; i < now[p]->shape.count; i++) {
            if (now[p]->values[i] != before[p]->values[i]) {
                branch_changes[p / 5]++;
            }
            if (now[p]->adam_m[i] != 0.0f &&
                now[p]->adam_v[i] > 0.0f) {
                branch_moments[p / 5]++;
            }
        }
    }
    return branch_changes[0] > 0 && branch_changes[1] > 0 &&
        branch_moments[0] > 0 && branch_moments[1] > 0;
}

static int manual_step(TrainFixture *fixture, BNState *before_adam)
{
    VAECParameter *parameters[PAIR_PARAMETER_COUNT];
    registry(&fixture->pair, parameters);
    if (vae_c_iic_pair_zero_grad(&fixture->pair) != VAE_C_OP_OK ||
        vae_c_iic_pair_forward_train(
            &fixture->pair, &fixture->input_a, &fixture->input_b,
            &fixture->workspace) != VAE_C_OP_OK ||
        vae_c_iic_pair_backward(
            &fixture->pair, &fixture->input_a, &fixture->input_b,
            &fixture->workspace,
            VAE_C_IIC_PAIR_BACKWARD_A_THEN_B) != VAE_C_OP_OK) {
        return 0;
    }
    snapshot_bn(&fixture->pair, before_adam);
    return vae_c_adam_step(
               &fixture->optimizer, parameters,
               PAIR_PARAMETER_COUNT) == VAE_C_OP_OK &&
        bn_matches(&fixture->pair, before_adam);
}

static int test_two_steps(void)
{
    TrainFixture automated;
    TrainFixture manual;
    TrainFixture initial;
    BNState before_adam;
    int passed = 0;
    size_t step;
    init_fixture(&automated);
    init_fixture(&manual);
    init_fixture(&initial);
    if (!allocate_fixture(&automated, 4, 7, 3, 4) ||
        !allocate_fixture(&manual, 4, 7, 3, 4) ||
        !allocate_fixture(&initial, 4, 7, 3, 4)) {
        goto cleanup;
    }
    if (!moments_zero(&automated.pair) ||
        automated.optimizer.step != 0 ||
        !pair_parameters_equal(&automated.pair, &manual.pair, 1)) {
        goto cleanup;
    }
    for (step = 1; step <= 2; step++) {
        VAECParameter *poison[PAIR_PARAMETER_COUNT];
        size_t p;
        registry(&automated.pair, poison);
        /* A nonzero stale gradient makes an omitted zero_grad observable. */
        for (p = 0; p < PAIR_PARAMETER_COUNT; p++) {
            poison[p]->gradients[0] = 123.0f;
        }
        if (!manual_step(&manual, &before_adam) ||
            vae_c_iic_pair_train_step(
                &automated.pair, &automated.workspace,
                &automated.optimizer, &automated.input_a,
                &automated.input_b) != VAE_C_OP_OK) {
            goto cleanup;
        }
        printf("TRAIN_NUMERIC step=%zu main_MI=%.9g main_loss=%.9g "
               "over_MI=%.9g over_loss=%.9g total_loss=%.9g\n",
               step,
               (double)automated.workspace.main_result.mutual_information,
               (double)automated.workspace.main_result.loss,
               (double)automated.workspace.over_result.mutual_information,
               (double)automated.workspace.over_result.loss,
               (double)automated.workspace.total_loss);
        if (!pair_parameters_equal(&automated.pair, &manual.pair, 1) ||
            !optimizer_equal(&automated.optimizer, &manual.optimizer) ||
            !bn_matches(&automated.pair, &before_adam) ||
            automated.optimizer.step != (uint64_t)step ||
            automated.pair.main_head.num_batches_tracked != 2 * step ||
            automated.pair.overcluster_head.num_batches_tracked !=
                2 * step ||
            automated.workspace.main_result.mutual_information !=
                manual.workspace.main_result.mutual_information ||
            automated.workspace.main_result.loss !=
                manual.workspace.main_result.loss ||
            automated.workspace.over_result.mutual_information !=
                manual.workspace.over_result.mutual_information ||
            automated.workspace.over_result.loss !=
                manual.workspace.over_result.loss ||
            automated.workspace.total_loss !=
                manual.workspace.total_loss ||
            !isfinite(automated.workspace.total_loss) ||
            !parameters_finite(&automated.pair) ||
            !inputs_unchanged(&automated) ||
            !inputs_unchanged(&manual) ||
            !each_head_updated(&automated.pair, &initial.pair)) {
            goto cleanup;
        }
    }
    passed = 1;
cleanup:
    free_fixture(&initial);
    free_fixture(&manual);
    free_fixture(&automated);
    return report("two consecutive pair steps, Adam/BN/gradients", passed);
}

static int test_invalid_preflight(void)
{
    TrainFixture fixture;
    TrainFixture initial;
    BNState initial_bn;
    VAECTensor wrong_shape;
    VAECTensor aliased_input;
    VAECShape shape;
    const size_t dims[2] = {4, 2};
    VAECAdam optimizer_before;
    int passed = 0;
    init_fixture(&fixture);
    init_fixture(&initial);
    vae_c_tensor_init(&wrong_shape);
    if (!allocate_fixture(&fixture, 4, 4, 3, 4) ||
        !allocate_fixture(&initial, 4, 4, 3, 4) ||
        vae_c_shape_make(2, dims, &shape) != VAE_C_STORAGE_OK ||
        vae_c_tensor_allocate(&wrong_shape, &shape, true) !=
            VAE_C_STORAGE_OK) {
        goto cleanup;
    }
    snapshot_bn(&fixture.pair, &initial_bn);
    optimizer_before = fixture.optimizer;
    if (vae_c_iic_pair_train_step(
            &fixture.pair, &fixture.workspace, &fixture.optimizer,
            &wrong_shape, &fixture.input_b) == VAE_C_OP_OK ||
        !pair_parameters_equal(&fixture.pair, &initial.pair, 1) ||
        !bn_matches(&fixture.pair, &initial_bn) ||
        !optimizer_equal(&fixture.optimizer, &optimizer_before)) {
        goto cleanup;
    }
    fixture.input_a.data[0] = NAN;
    if (vae_c_iic_pair_train_step(
            &fixture.pair, &fixture.workspace, &fixture.optimizer,
            &fixture.input_a, &fixture.input_b) == VAE_C_OP_OK ||
        !pair_parameters_equal(&fixture.pair, &initial.pair, 1) ||
        !bn_matches(&fixture.pair, &initial_bn) ||
        !optimizer_equal(&fixture.optimizer, &optimizer_before)) {
        goto cleanup;
    }
    fixture.input_a.data[0] = fixture_input_a[0];
    fixture.pair.main_head.linear1.weights.gradients[0] = 37.0f;
    initial.pair.main_head.linear1.weights.gradients[0] = 37.0f;
    aliased_input = fixture.input_a;
    aliased_input.data =
        fixture.pair.main_head.linear1.weights.gradients;
    aliased_input.owns_memory = false;
    aliased_input.owned_block = NULL;
    aliased_input.owned_bytes = 0;
    if (vae_c_iic_pair_train_step(
            &fixture.pair, &fixture.workspace, &fixture.optimizer,
            &aliased_input, &fixture.input_b) == VAE_C_OP_OK ||
        !pair_parameters_equal(&fixture.pair, &initial.pair, 1) ||
        !bn_matches(&fixture.pair, &initial_bn) ||
        !optimizer_equal(&fixture.optimizer, &optimizer_before)) {
        goto cleanup;
    }
    fixture.optimizer.step = UINT64_MAX;
    optimizer_before = fixture.optimizer;
    if (vae_c_iic_pair_train_step(
            &fixture.pair, &fixture.workspace, &fixture.optimizer,
            &fixture.input_a, &fixture.input_b) == VAE_C_OP_OK ||
        !pair_parameters_equal(&fixture.pair, &initial.pair, 1) ||
        !bn_matches(&fixture.pair, &initial_bn) ||
        !optimizer_equal(&fixture.optimizer, &optimizer_before)) {
        goto cleanup;
    }
    passed = inputs_unchanged(&fixture);
cleanup:
    vae_c_tensor_free(&wrong_shape);
    free_fixture(&initial);
    free_fixture(&fixture);
    return report("shape/NaN/alias/step-overflow preflight is atomic", passed);
}

static int test_tail_real_dimensions(void)
{
    TrainFixture fixture;
    int passed = 0;
    init_fixture(&fixture);
    if (!allocate_fixture(&fixture, 7, 11, 36, 250)) {
        goto cleanup;
    }
    if (vae_c_iic_pair_train_step(
            &fixture.pair, &fixture.workspace, &fixture.optimizer,
            &fixture.input_a, &fixture.input_b) != VAE_C_OP_OK) {
        goto cleanup;
    }
    passed = fixture.workspace.batch_size == 0 &&
        fixture.workspace.main_result.class_count == 36 &&
        fixture.workspace.over_result.class_count == 250 &&
        fixture.optimizer.step == 1 &&
        fixture.pair.main_head.num_batches_tracked == 2 &&
        fixture.pair.overcluster_head.num_batches_tracked == 2 &&
        parameters_finite(&fixture.pair) &&
        isfinite(fixture.workspace.total_loss) &&
        inputs_unchanged(&fixture);
cleanup:
    free_fixture(&fixture);
    return report("tail B=7 and real C=36/W=250 training step", passed);
}

static int test_branch_independence(void)
{
    TrainFixture reference;
    TrainFixture changed_main;
    VAECParameter *reference_parameters[PAIR_PARAMETER_COUNT];
    VAECParameter *changed_parameters[PAIR_PARAMETER_COUNT];
    const size_t bn_bytes = VAE_C_IIC_HEAD_HIDDEN_DIM * sizeof(float);
    size_t p;
    int passed = 0;
    init_fixture(&reference);
    init_fixture(&changed_main);
    if (!allocate_fixture(&reference, 4, 4, 3, 4) ||
        !allocate_fixture(&changed_main, 4, 4, 3, 4)) {
        goto cleanup;
    }
    changed_main.pair.main_head.linear1.weights.values[0] += 0.5f;
    if (vae_c_iic_pair_train_step(
            &reference.pair, &reference.workspace, &reference.optimizer,
            &reference.input_a, &reference.input_b) != VAE_C_OP_OK ||
        vae_c_iic_pair_train_step(
            &changed_main.pair, &changed_main.workspace,
            &changed_main.optimizer, &changed_main.input_a,
            &changed_main.input_b) != VAE_C_OP_OK) {
        goto cleanup;
    }
    registry(&reference.pair, reference_parameters);
    registry(&changed_main.pair, changed_parameters);
    passed = reference.optimizer.step == 1 &&
        optimizer_equal(&reference.optimizer, &changed_main.optimizer) &&
        reference.workspace.over_result.mutual_information ==
            changed_main.workspace.over_result.mutual_information &&
        reference.workspace.over_result.loss ==
            changed_main.workspace.over_result.loss &&
        memcmp(reference.pair.overcluster_head.batch_norm.running_mean,
               changed_main.pair.overcluster_head.batch_norm.running_mean,
               bn_bytes) == 0 &&
        memcmp(reference.pair.overcluster_head.batch_norm.running_variance,
               changed_main.pair.overcluster_head.batch_norm.running_variance,
               bn_bytes) == 0 &&
        reference.pair.overcluster_head.num_batches_tracked ==
            changed_main.pair.overcluster_head.num_batches_tracked &&
        reference_parameters[0]->values[0] !=
            changed_parameters[0]->values[0];
    for (p = 5; passed && p < PAIR_PARAMETER_COUNT; p++) {
        const size_t bytes = reference_parameters[p]->shape.count *
            sizeof(float);
        passed = memcmp(reference_parameters[p]->values,
                        changed_parameters[p]->values, bytes) == 0 &&
            memcmp(reference_parameters[p]->gradients,
                   changed_parameters[p]->gradients, bytes) == 0 &&
            memcmp(reference_parameters[p]->adam_m,
                   changed_parameters[p]->adam_m, bytes) == 0 &&
            memcmp(reference_parameters[p]->adam_v,
                   changed_parameters[p]->adam_v, bytes) == 0;
    }
cleanup:
    free_fixture(&changed_main);
    free_fixture(&reference);
    return report("main/over independent with one shared optimizer", passed);
}

static void dump_float_array(
    size_t step,
    const char *phase,
    const char *branch,
    const char *field,
    const float *values,
    size_t count
)
{
    size_t index;
    for (index = 0; index < count; index++) {
        printf("C11 step=%zu phase=%s branch=%s field=%s index=%zu "
               "value=%.9g\n", step, phase, branch, field, index,
               (double)values[index]);
    }
}

static void dump_bn_state(
    size_t step,
    const char *phase,
    const VAECIICPair *pair
)
{
    const VAECIICHead *heads[2] = {
        &pair->main_head, &pair->overcluster_head
    };
    const char *branches[2] = {"main", "overcluster"};
    size_t branch;
    for (branch = 0; branch < 2; branch++) {
        dump_float_array(
            step, phase, branches[branch], "running_mean",
            heads[branch]->batch_norm.running_mean,
            VAE_C_IIC_HEAD_HIDDEN_DIM
        );
        dump_float_array(
            step, phase, branches[branch], "running_var",
            heads[branch]->batch_norm.running_variance,
            VAE_C_IIC_HEAD_HIDDEN_DIM
        );
        printf("C11 step=%zu phase=%s branch=%s "
               "field=num_batches_tracked value=%llu\n",
               step, phase, branches[branch],
               (unsigned long long)heads[branch]->num_batches_tracked);
    }
}

static void dump_parameters(
    size_t step,
    const char *phase,
    VAECIICPair *pair
)
{
    static const char *names[5] = {
        "hidden_weight", "bn_gamma", "bn_beta", "output_weight",
        "output_bias"
    };
    VAECParameter *parameters[PAIR_PARAMETER_COUNT];
    size_t p;
    registry(pair, parameters);
    for (p = 0; p < PAIR_PARAMETER_COUNT; p++) {
        const char *branch = p < 5 ? "main" : "overcluster";
        char field[64];
        const char *name = names[p % 5];
        if (strcmp(phase, "pre_step") == 0) {
            (void)snprintf(field, sizeof(field), "%s.gradient", name);
            dump_float_array(step, phase, branch, field,
                             parameters[p]->gradients,
                             parameters[p]->shape.count);
        } else {
            (void)snprintf(field, sizeof(field), "%s.value", name);
            dump_float_array(step, phase, branch, field,
                             parameters[p]->values,
                             parameters[p]->shape.count);
            (void)snprintf(field, sizeof(field), "%s.m", name);
            dump_float_array(step, phase, branch, field,
                             parameters[p]->adam_m,
                             parameters[p]->shape.count);
            (void)snprintf(field, sizeof(field), "%s.v", name);
            dump_float_array(step, phase, branch, field,
                             parameters[p]->adam_v,
                             parameters[p]->shape.count);
        }
    }
}

static int run_dump_golden(void)
{
    TrainFixture fixture;
    VAECParameter *parameters[PAIR_PARAMETER_COUNT];
    size_t step;
    int success = 0;
    init_fixture(&fixture);
    if (!allocate_fixture(&fixture, 4, 7, 3, 4)) {
        goto cleanup;
    }
    registry(&fixture.pair, parameters);
    for (step = 1; step <= 2; step++) {
        if (vae_c_iic_pair_zero_grad(&fixture.pair) != VAE_C_OP_OK ||
            vae_c_iic_pair_forward_train(
                &fixture.pair, &fixture.input_a, &fixture.input_b,
                &fixture.workspace) != VAE_C_OP_OK ||
            vae_c_iic_pair_backward(
                &fixture.pair, &fixture.input_a, &fixture.input_b,
                &fixture.workspace,
                VAE_C_IIC_PAIR_BACKWARD_A_THEN_B) != VAE_C_OP_OK) {
            goto cleanup;
        }
        printf("C11 step=%zu phase=pre_step field=main.mi value=%.9g\n",
               step,
               (double)fixture.workspace.main_result.mutual_information);
        printf("C11 step=%zu phase=pre_step field=main.loss value=%.9g\n",
               step, (double)fixture.workspace.main_result.loss);
        printf("C11 step=%zu phase=pre_step "
               "field=overcluster.mi value=%.9g\n", step,
               (double)fixture.workspace.over_result.mutual_information);
        printf("C11 step=%zu phase=pre_step "
               "field=overcluster.loss value=%.9g\n", step,
               (double)fixture.workspace.over_result.loss);
        printf("C11 step=%zu phase=pre_step "
               "field=total_loss value=%.9g\n", step,
               (double)fixture.workspace.total_loss);
        dump_parameters(step, "pre_step", &fixture.pair);
        dump_bn_state(step, "before_adam", &fixture.pair);
        if (vae_c_adam_step(
                &fixture.optimizer, parameters,
                PAIR_PARAMETER_COUNT) != VAE_C_OP_OK) {
            goto cleanup;
        }
        dump_parameters(step, "after_step", &fixture.pair);
        dump_bn_state(step, "after_adam", &fixture.pair);
        printf("C11 step=%zu phase=after_step "
               "field=optimizer.step value=%llu\n", step,
               (unsigned long long)fixture.optimizer.step);
        printf("C11 step=%zu phase=after_step "
               "field=optimizer.beta1_power value=%.17g\n", step,
               fixture.optimizer.beta1_power);
        printf("C11 step=%zu phase=after_step "
               "field=optimizer.beta2_power value=%.17g\n", step,
               fixture.optimizer.beta2_power);
        printf("C11 step=%zu phase=after_step "
               "field=input_unchanged value=%d\n", step,
               inputs_unchanged(&fixture));
    }
    success = 1;
cleanup:
    free_fixture(&fixture);
    if (!success) {
        fprintf(stderr, "C11 golden dump failed\n");
    }
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(int argc, char **argv)
{
    int failures = 0;
    if (argc == 2 && strcmp(argv[1], "--dump-golden") == 0) {
        return run_dump_golden();
    }
    if (argc != 1) {
        fprintf(stderr, "usage: test_vae_c_iic_pair_train "
                "[--dump-golden]\n");
        return EXIT_FAILURE;
    }
    failures += test_two_steps();
    failures += test_invalid_preflight();
    failures += test_tail_real_dimensions();
    failures += test_branch_independence();
    if (failures == 0) {
        printf("ALL VAE C IIC PAIR TRAIN TESTS PASSED\n");
    }
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
