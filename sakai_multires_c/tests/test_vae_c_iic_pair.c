#include "vae_c_iic_pair.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define TEST_INPUT_DIM 3
#define TEST_MAIN_DIM 3
#define TEST_OVER_DIM 4
#define TEST_ACTIVE_HIDDEN 4

typedef struct {
    VAECIICPair pair;
    VAECIICPairWorkspace workspace;
    VAECTensor input_a;
    VAECTensor input_b;
} Fixture;

static const float input_a_golden[12] = {
    0.5f, -1.0f, 0.25f,
    -0.75f, 0.5f, 1.0f,
    1.25f, 0.25f, -0.5f,
    -0.25f, -0.75f, 0.75f
};

static int check(const char *name, int passed)
{
    printf("[%s] %s\n", passed ? "PASS" : "FAIL", name);
    return passed ? 0 : 1;
}

static int close_f32(float left, float right, float absolute_tolerance)
{
    return isfinite(left) && isfinite(right) &&
           fabsf(left - right) <= absolute_tolerance;
}

static int array_close(
    const float *left,
    const float *right,
    size_t count,
    float tolerance
)
{
    size_t index;
    for (index = 0; index < count; index++) {
        if (!close_f32(left[index], right[index], tolerance)) {
            fprintf(stderr, "gradient mismatch index=%zu left=%g right=%g\n",
                    index, (double)left[index], (double)right[index]);
            return 0;
        }
    }
    return 1;
}

static int all_finite(const float *values, size_t count)
{
    size_t index;
    for (index = 0; index < count; index++) {
        if (!isfinite(values[index])) {
            return 0;
        }
    }
    return 1;
}

static int make_input(size_t batch, VAECTensor *input)
{
    const size_t dims[2] = {batch, TEST_INPUT_DIM};
    VAECShape shape;
    return vae_c_shape_make(2, dims, &shape) == VAE_C_STORAGE_OK &&
           vae_c_tensor_allocate(input, &shape, true) == VAE_C_STORAGE_OK;
}

static void set_main_parameters(VAECIICHead *head)
{
    static const float linear1[4][3] = {
        {0.5f, -0.25f, 0.125f},
        {-0.375f, 0.25f, 0.5f},
        {0.25f, 0.75f, -0.5f},
        {-0.125f, 0.5f, 0.375f}
    };
    static const float gamma[4] = {1.125f, -0.875f, 0.75f, 1.25f};
    static const float beta[4] = {0.125f, 0.25f, -0.0625f, -0.125f};
    static const float mean[4] = {0.25f, -0.125f, 0.375f, -0.25f};
    static const float variance[4] = {1.5f, 0.75f, 2.0f, 1.25f};
    static const float linear2[3][4] = {
        {0.5f, -0.25f, 0.125f, 0.0625f},
        {-0.125f, 0.375f, 0.25f, -0.5f},
        {0.25f, 0.125f, -0.375f, 0.625f}
    };
    static const float bias[3] = {0.0625f, -0.125f, 0.1875f};
    size_t hidden;
    size_t output;
    memset(head->linear1.weights.values, 0,
           head->linear1.weights.shape.count * sizeof(float));
    memset(head->linear2.weights.values, 0,
           head->linear2.weights.shape.count * sizeof(float));
    for (hidden = 0; hidden < VAE_C_IIC_HEAD_HIDDEN_DIM; hidden++) {
        head->batch_norm.gamma.values[hidden] = 1.0f;
        head->batch_norm.beta.values[hidden] = -1.0f;
        head->batch_norm.running_mean[hidden] = 0.0f;
        head->batch_norm.running_variance[hidden] = 1.0f;
        if (hidden < TEST_ACTIVE_HIDDEN) {
            memcpy(head->linear1.weights.values + hidden * TEST_INPUT_DIM,
                   linear1[hidden], sizeof(linear1[hidden]));
            head->batch_norm.gamma.values[hidden] = gamma[hidden];
            head->batch_norm.beta.values[hidden] = beta[hidden];
            head->batch_norm.running_mean[hidden] = mean[hidden];
            head->batch_norm.running_variance[hidden] = variance[hidden];
        }
    }
    for (output = 0; output < TEST_MAIN_DIM; output++) {
        memcpy(head->linear2.weights.values + output *
                   VAE_C_IIC_HEAD_HIDDEN_DIM,
               linear2[output], sizeof(linear2[output]));
        head->linear2.bias.values[output] = bias[output];
    }
}

static void set_over_parameters(VAECIICHead *head)
{
    size_t hidden;
    size_t feature;
    size_t output;
    memset(head->linear1.weights.values, 0,
           head->linear1.weights.shape.count * sizeof(float));
    memset(head->linear2.weights.values, 0,
           head->linear2.weights.shape.count * sizeof(float));
    for (hidden = 0; hidden < VAE_C_IIC_HEAD_HIDDEN_DIM; hidden++) {
        head->batch_norm.gamma.values[hidden] = 1.0f;
        head->batch_norm.beta.values[hidden] = -1.0f;
        head->batch_norm.running_mean[hidden] = 0.0f;
        head->batch_norm.running_variance[hidden] = 1.0f;
        if (hidden < TEST_ACTIVE_HIDDEN) {
            for (feature = 0; feature < TEST_INPUT_DIM; feature++) {
                head->linear1.weights.values[hidden * TEST_INPUT_DIM +
                                             feature] =
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
        head->linear2.bias.values[output] =
            (float)((int)(output % 3) - 1) * 0.0625f;
        for (hidden = 0; hidden < TEST_ACTIVE_HIDDEN; hidden++) {
            head->linear2.weights.values[
                output * VAE_C_IIC_HEAD_HIDDEN_DIM + hidden] =
                (float)((int)((5 * output + 2 * hidden) % 9) - 4) * 0.125f;
        }
    }
}

static void fixture_init(Fixture *fixture)
{
    memset(fixture, 0, sizeof(*fixture));
    vae_c_iic_pair_init(&fixture->pair);
    vae_c_iic_pair_workspace_init(&fixture->workspace);
    vae_c_tensor_init(&fixture->input_a);
    vae_c_tensor_init(&fixture->input_b);
}

static void fixture_free(Fixture *fixture)
{
    vae_c_tensor_free(&fixture->input_b);
    vae_c_tensor_free(&fixture->input_a);
    vae_c_iic_pair_workspace_free(&fixture->workspace);
    vae_c_iic_pair_free(&fixture->pair);
}

static int fixture_allocate(
    Fixture *fixture,
    size_t batch,
    size_t capacity,
    size_t main_dim,
    size_t over_dim
)
{
    size_t index;
    fixture_init(fixture);
    if (vae_c_iic_pair_allocate(&fixture->pair, TEST_INPUT_DIM,
                                main_dim, over_dim) != VAE_C_OP_OK ||
        vae_c_iic_pair_workspace_allocate(&fixture->pair, capacity,
                                          &fixture->workspace) != VAE_C_OP_OK ||
        !make_input(batch, &fixture->input_a) ||
        !make_input(batch, &fixture->input_b)) {
        return 0;
    }
    set_main_parameters(&fixture->pair.main_head);
    set_over_parameters(&fixture->pair.overcluster_head);
    for (index = 0; index < batch * TEST_INPUT_DIM; index++) {
        float value = index < 12 ? input_a_golden[index] :
            (float)((int)(index % 11) - 5) * 0.1875f;
        fixture->input_a.data[index] = value;
        fixture->input_b.data[index] = -0.375f * value +
            (float)((int)(index % 5) - 2) * 0.125f;
    }
    return 1;
}

static int probability_rows(const VAECIICHeadWorkspace *workspace)
{
    size_t row;
    size_t column;
    for (row = 0; row < workspace->batch_size; row++) {
        float sum = 0.0f;
        for (column = 0; column < workspace->output_dim; column++) {
            float probability = workspace->softmax_output.data[
                row * workspace->output_dim + column];
            if (!isfinite(probability) || probability < 0.0f) {
                return 0;
            }
            sum += probability;
        }
        if (!close_f32(sum, 1.0f, 2.0e-6f)) {
            return 0;
        }
    }
    return 1;
}

static int head_gradients_close(
    const VAECIICHead *left,
    const VAECIICHead *right,
    float tolerance
)
{
    return
        array_close(left->linear1.weights.gradients,
                    right->linear1.weights.gradients,
                    left->linear1.weights.shape.count, tolerance) &&
        array_close(left->batch_norm.gamma.gradients,
                    right->batch_norm.gamma.gradients,
                    VAE_C_IIC_HEAD_HIDDEN_DIM, tolerance) &&
        array_close(left->batch_norm.beta.gradients,
                    right->batch_norm.beta.gradients,
                    VAE_C_IIC_HEAD_HIDDEN_DIM, tolerance) &&
        array_close(left->linear2.weights.gradients,
                    right->linear2.weights.gradients,
                    left->linear2.weights.shape.count, tolerance) &&
        array_close(left->linear2.bias.gradients,
                    right->linear2.bias.gradients,
                    left->output_dim, tolerance);
}

static int head_gradients_finite(const VAECIICHead *head)
{
    return all_finite(head->linear1.weights.gradients,
                      head->linear1.weights.shape.count) &&
           all_finite(head->batch_norm.gamma.gradients,
                      VAE_C_IIC_HEAD_HIDDEN_DIM) &&
           all_finite(head->batch_norm.beta.gradients,
                      VAE_C_IIC_HEAD_HIDDEN_DIM) &&
           all_finite(head->linear2.weights.gradients,
                      head->linear2.weights.shape.count) &&
           all_finite(head->linear2.bias.gradients, head->output_dim);
}

static int run_golden_case(
    VAECIICPairBackwardOrder order,
    Fixture *fixture
)
{
    if (!fixture_allocate(fixture, 4, 7, TEST_MAIN_DIM, TEST_OVER_DIM)) {
        return 0;
    }
    return vae_c_iic_pair_forward_train(
               &fixture->pair, &fixture->input_a, &fixture->input_b,
               &fixture->workspace) == VAE_C_OP_OK &&
           vae_c_iic_pair_zero_grad(&fixture->pair) == VAE_C_OP_OK &&
           vae_c_tensor_zero_grad(&fixture->input_a) == VAE_C_STORAGE_OK &&
           vae_c_tensor_zero_grad(&fixture->input_b) == VAE_C_STORAGE_OK &&
           vae_c_iic_pair_backward(
               &fixture->pair, &fixture->input_a, &fixture->input_b,
               &fixture->workspace, order) == VAE_C_OP_OK;
}

static int test_golden_and_backward_orders(void)
{
    Fixture a_first;
    Fixture b_first;
    int passed = 0;
    fixture_init(&a_first);
    fixture_init(&b_first);
    if (!run_golden_case(VAE_C_IIC_PAIR_BACKWARD_A_THEN_B, &a_first) ||
        !run_golden_case(VAE_C_IIC_PAIR_BACKWARD_B_THEN_A, &b_first)) {
        goto cleanup;
    }
    printf("PAIR_NUMERIC main_MI=%.9g over_MI=%.9g main_loss=%.9g "
           "over_loss=%.9g total_loss=%.9g\n",
           (double)a_first.workspace.main_result.mutual_information,
           (double)a_first.workspace.over_result.mutual_information,
           (double)a_first.workspace.main_result.loss,
           (double)a_first.workspace.over_result.loss,
           (double)a_first.workspace.total_loss);
    passed =
        a_first.pair.main_head.num_batches_tracked == 2 &&
        a_first.pair.overcluster_head.num_batches_tracked == 2 &&
        a_first.pair.main_head.forward_generation == 2 &&
        a_first.pair.overcluster_head.forward_generation == 2 &&
        close_f32(a_first.workspace.total_loss,
                  a_first.workspace.main_result.loss +
                      a_first.workspace.over_result.loss, 1.0e-7f) &&
        isfinite(a_first.workspace.total_loss) &&
        probability_rows(&a_first.workspace.main_a) &&
        probability_rows(&a_first.workspace.main_b) &&
        probability_rows(&a_first.workspace.over_a) &&
        probability_rows(&a_first.workspace.over_b) &&
        a_first.workspace.main_result.class_count == TEST_MAIN_DIM &&
        a_first.workspace.over_result.class_count == TEST_OVER_DIM &&
        array_close(a_first.input_a.grad, b_first.input_a.grad, 12,
                    3.0e-5f) &&
        array_close(a_first.input_b.grad, b_first.input_b.grad, 12,
                    3.0e-5f) &&
        head_gradients_close(&a_first.pair.main_head,
                             &b_first.pair.main_head, 3.0e-5f) &&
        head_gradients_close(&a_first.pair.overcluster_head,
                             &b_first.pair.overcluster_head, 3.0e-5f) &&
        head_gradients_finite(&a_first.pair.main_head) &&
        head_gradients_finite(&a_first.pair.overcluster_head);
    if (passed) {
        size_t index;
        for (index = 0; index < 12; index++) {
            if (a_first.input_a.data[index] != input_a_golden[index] ||
                a_first.input_b.data[index] !=
                    -0.375f * input_a_golden[index] +
                        (float)((int)(index % 5) - 2) * 0.125f) {
                passed = 0;
            }
        }
    }

cleanup:
    fixture_free(&b_first);
    fixture_free(&a_first);
    return check("golden fixture, reference order, two BN updates, "
                 "A/B backward order and input preservation", passed);
}

static int test_lifecycle_atomicity(void)
{
    Fixture fixture;
    float input_grad_before[12];
    float main_weight_grad_before;
    float over_weight_grad_before;
    int passed = 0;
    fixture_init(&fixture);
    if (!run_golden_case(VAE_C_IIC_PAIR_BACKWARD_A_THEN_B, &fixture)) {
        goto cleanup;
    }
    memcpy(input_grad_before, fixture.input_a.grad,
           sizeof(input_grad_before));
    main_weight_grad_before =
        fixture.pair.main_head.linear1.weights.gradients[0];
    over_weight_grad_before =
        fixture.pair.overcluster_head.linear1.weights.gradients[0];
    passed = vae_c_iic_pair_backward(
                 &fixture.pair, &fixture.input_a, &fixture.input_b,
                 &fixture.workspace,
                 VAE_C_IIC_PAIR_BACKWARD_A_THEN_B) != VAE_C_OP_OK &&
             memcmp(input_grad_before, fixture.input_a.grad,
                    sizeof(input_grad_before)) == 0 &&
             fixture.pair.main_head.linear1.weights.gradients[0] ==
                 main_weight_grad_before &&
             fixture.pair.overcluster_head.linear1.weights.gradients[0] ==
                 over_weight_grad_before;
cleanup:
    fixture_free(&fixture);
    return check("consumed context rejects retry atomically", passed);
}

static int test_stale_and_invalid_order(void)
{
    Fixture fixture;
    int passed = 0;
    fixture_init(&fixture);
    if (!fixture_allocate(&fixture, 4, 4, TEST_MAIN_DIM,
                          TEST_OVER_DIM) ||
        vae_c_iic_pair_forward_train(
            &fixture.pair, &fixture.input_a, &fixture.input_b,
            &fixture.workspace) != VAE_C_OP_OK) {
        goto cleanup;
    }
    passed = vae_c_iic_pair_backward(
                 &fixture.pair, &fixture.input_a, &fixture.input_b,
                 &fixture.workspace,
                 (VAECIICPairBackwardOrder)0) != VAE_C_OP_OK &&
             fixture.pair.main_head.linear1.weights.gradients[0] == 0.0f &&
             fixture.pair.overcluster_head.linear1.weights.gradients[0] ==
                 0.0f &&
             fixture.input_a.grad[0] == 0.0f;
    if (!passed || vae_c_iic_head_forward(
            &fixture.pair.main_head, VAE_C_BATCH_NORM_MODE_TRAIN,
            &fixture.input_a, &fixture.workspace.main_a) != VAE_C_OP_OK) {
        passed = 0;
        goto cleanup;
    }
    passed = vae_c_iic_pair_backward(
                 &fixture.pair, &fixture.input_a, &fixture.input_b,
                 &fixture.workspace,
                 VAE_C_IIC_PAIR_BACKWARD_A_THEN_B) != VAE_C_OP_OK &&
             fixture.pair.main_head.linear1.weights.gradients[0] == 0.0f &&
             fixture.pair.overcluster_head.linear1.weights.gradients[0] ==
                 0.0f &&
             fixture.input_a.grad[0] == 0.0f;
cleanup:
    fixture_free(&fixture);
    return check("invalid order and stale context reject atomically",
                 passed);
}

static int test_manual_branch_accumulation(void)
{
    Fixture pair_fixture;
    Fixture manual;
    size_t batch = 4;
    int passed = 0;
    fixture_init(&pair_fixture);
    fixture_init(&manual);
    if (!run_golden_case(VAE_C_IIC_PAIR_BACKWARD_A_THEN_B,
                         &pair_fixture) ||
        !fixture_allocate(&manual, batch, batch, TEST_MAIN_DIM,
                          TEST_OVER_DIM) ||
        vae_c_iic_pair_forward_train(
            &manual.pair, &manual.input_a, &manual.input_b,
            &manual.workspace) != VAE_C_OP_OK ||
        vae_c_iic_pair_zero_grad(&manual.pair) != VAE_C_OP_OK ||
        vae_c_iic_pair_workspace_zero_grad(
            &manual.pair, &manual.workspace) != VAE_C_OP_OK) {
        goto cleanup;
    }
    if (vae_c_iic_loss_backward(
            manual.workspace.main_a.softmax_output.data, batch,
            TEST_MAIN_DIM, manual.workspace.main_b.softmax_output.data,
            batch, TEST_MAIN_DIM, &manual.workspace.main_result,
            manual.workspace.main_a.softmax_output.grad,
            manual.workspace.main_a.softmax_output.shape.count,
            manual.workspace.main_b.softmax_output.grad,
            manual.workspace.main_b.softmax_output.shape.count) !=
            VAE_C_IIC_LOSS_OK ||
        vae_c_iic_loss_backward(
            manual.workspace.over_a.softmax_output.data, batch,
            TEST_OVER_DIM, manual.workspace.over_b.softmax_output.data,
            batch, TEST_OVER_DIM, &manual.workspace.over_result,
            manual.workspace.over_a.softmax_output.grad,
            manual.workspace.over_a.softmax_output.shape.count,
            manual.workspace.over_b.softmax_output.grad,
            manual.workspace.over_b.softmax_output.shape.count) !=
            VAE_C_IIC_LOSS_OK ||
        vae_c_iic_head_backward(
            &manual.pair.main_head, &manual.input_a,
            &manual.workspace.main_a) != VAE_C_OP_OK ||
        vae_c_iic_head_backward(
            &manual.pair.overcluster_head, &manual.input_a,
            &manual.workspace.over_a) != VAE_C_OP_OK) {
        goto cleanup;
    }
    /* One view contributes to both independent heads and to input A. */
    if (!all_finite(manual.input_a.grad, batch * TEST_INPUT_DIM) ||
        !all_finite(manual.pair.main_head.linear1.weights.gradients,
                    manual.pair.main_head.linear1.weights.shape.count) ||
        !all_finite(manual.pair.overcluster_head.linear1.weights.gradients,
                    manual.pair.overcluster_head.linear1.weights.shape.count) ||
        vae_c_iic_head_backward(
            &manual.pair.main_head, &manual.input_b,
            &manual.workspace.main_b) != VAE_C_OP_OK ||
        vae_c_iic_head_backward(
            &manual.pair.overcluster_head, &manual.input_b,
            &manual.workspace.over_b) != VAE_C_OP_OK) {
        goto cleanup;
    }
    passed =
        head_gradients_close(&pair_fixture.pair.main_head,
                             &manual.pair.main_head, 3.0e-5f) &&
        head_gradients_close(&pair_fixture.pair.overcluster_head,
                             &manual.pair.overcluster_head, 3.0e-5f) &&
        array_close(pair_fixture.input_a.grad, manual.input_a.grad,
                    batch * TEST_INPUT_DIM, 3.0e-5f) &&
        array_close(pair_fixture.input_b.grad, manual.input_b.grad,
                    batch * TEST_INPUT_DIM, 3.0e-5f);
cleanup:
    fixture_free(&manual);
    fixture_free(&pair_fixture);
    return check("pair gradients equal explicit A+B branch accumulation",
                 passed);
}

static int test_invalid_forward_preflight(void)
{
    Fixture fixture;
    int passed = 1;
    fixture_init(&fixture);
    if (!fixture_allocate(&fixture, 4, 4, TEST_MAIN_DIM,
                          TEST_OVER_DIM)) {
        passed = 0;
        goto cleanup;
    }
    fixture.input_b.data[0] = NAN;
    passed = vae_c_iic_pair_forward_train(
                 &fixture.pair, &fixture.input_a, &fixture.input_b,
                 &fixture.workspace) != VAE_C_OP_OK &&
             fixture.pair.main_head.num_batches_tracked == 0 &&
             fixture.pair.overcluster_head.num_batches_tracked == 0;
    fixture.input_b.data[0] = 0.25f;
    fixture.input_b.shape.dims[1] = 4;
    passed = passed && vae_c_iic_pair_forward_train(
                 &fixture.pair, &fixture.input_a, &fixture.input_b,
                 &fixture.workspace) != VAE_C_OP_OK &&
             fixture.pair.main_head.num_batches_tracked == 0;
cleanup:
    fixture_free(&fixture);
    if (!check("NaN and invalid shape rejected before any BN update",
               passed)) {
        return 0;
    }
    fixture_init(&fixture);
    passed = fixture_allocate(&fixture, 7, 4, TEST_MAIN_DIM,
                              TEST_OVER_DIM);
    if (passed) {
        passed = vae_c_iic_pair_forward_train(
                     &fixture.pair, &fixture.input_a, &fixture.input_b,
                     &fixture.workspace) != VAE_C_OP_OK &&
                 fixture.pair.main_head.num_batches_tracked == 0;
    }
    fixture_free(&fixture);
    return check("capacity mismatch rejected before any BN update",
                 passed);
}

static int test_workspace_input_alias_preflight(void)
{
    Fixture fixture;
    VAECTensor aliased_input;
    int passed = 0;
    fixture_init(&fixture);
    if (!fixture_allocate(&fixture, 4, 4, TEST_MAIN_DIM,
                          TEST_OVER_DIM)) {
        goto cleanup;
    }
    aliased_input = fixture.input_a;
    aliased_input.data = fixture.workspace.over_a.softmax_storage.data;
    aliased_input.owned_block = NULL;
    aliased_input.owned_bytes = 0;
    aliased_input.owns_memory = false;
    passed = vae_c_iic_pair_forward_train(
                 &fixture.pair, &aliased_input, &fixture.input_b,
                 &fixture.workspace) != VAE_C_OP_OK &&
             fixture.pair.main_head.num_batches_tracked == 0 &&
             fixture.pair.overcluster_head.num_batches_tracked == 0 &&
             fixture.pair.main_head.forward_generation == 0 &&
             fixture.pair.overcluster_head.forward_generation == 0;
cleanup:
    fixture_free(&fixture);
    return check("input/workspace alias rejects before any head forward",
                 passed);
}

static int test_corrupt_over_result_atomicity(void)
{
    Fixture fixture;
    uint64_t main_a_generation;
    uint64_t main_b_generation;
    uint64_t over_a_generation;
    uint64_t over_b_generation;
    int passed = 0;
    fixture_init(&fixture);
    if (!fixture_allocate(&fixture, 4, 4, TEST_MAIN_DIM,
                          TEST_OVER_DIM) ||
        vae_c_iic_pair_forward_train(
            &fixture.pair, &fixture.input_a, &fixture.input_b,
            &fixture.workspace) != VAE_C_OP_OK) {
        goto cleanup;
    }
    main_a_generation = fixture.workspace.main_a.backward_generation;
    main_b_generation = fixture.workspace.main_b.backward_generation;
    over_a_generation = fixture.workspace.over_a.backward_generation;
    over_b_generation = fixture.workspace.over_b.backward_generation;
    fixture.workspace.over_result.joint[0] = NAN;
    passed = vae_c_iic_pair_backward(
                 &fixture.pair, &fixture.input_a, &fixture.input_b,
                 &fixture.workspace,
                 VAE_C_IIC_PAIR_BACKWARD_A_THEN_B) != VAE_C_OP_OK &&
             fixture.workspace.backward_pair == &fixture.pair &&
             fixture.workspace.main_a.backward_generation ==
                 main_a_generation &&
             fixture.workspace.main_b.backward_generation ==
                 main_b_generation &&
             fixture.workspace.over_a.backward_generation ==
                 over_a_generation &&
             fixture.workspace.over_b.backward_generation ==
                 over_b_generation &&
             fixture.pair.main_head.linear1.weights.gradients[0] == 0.0f &&
             fixture.pair.overcluster_head.linear1.weights.gradients[0] ==
                 0.0f &&
             fixture.workspace.main_a.softmax_output.grad[0] == 0.0f &&
             fixture.workspace.main_b.softmax_output.grad[0] == 0.0f &&
             fixture.input_a.grad[0] == 0.0f &&
             fixture.input_b.grad[0] == 0.0f;
cleanup:
    fixture_free(&fixture);
    return check("corrupt over joint rejects before gradients/tickets change",
                 passed);
}

static int test_tail_and_real_dimensions(void)
{
    Fixture fixture;
    VAECOpStatus forward_status;
    VAECOpStatus zero_status;
    VAECOpStatus backward_status;
    VAECOpStatus retry_status;
    float input_a_before[7 * TEST_INPUT_DIM];
    float input_b_before[7 * TEST_INPUT_DIM];
    int forward_batch_size_valid;
    int passed = 0;
    fixture_init(&fixture);
    if (!fixture_allocate(&fixture, 7, 7, 36, 250)) {
        goto cleanup;
    }
    memcpy(input_a_before, fixture.input_a.data, sizeof(input_a_before));
    memcpy(input_b_before, fixture.input_b.data, sizeof(input_b_before));
    forward_status = vae_c_iic_pair_forward_train(
        &fixture.pair, &fixture.input_a, &fixture.input_b,
        &fixture.workspace
    );
    forward_batch_size_valid = forward_status == VAE_C_OP_OK &&
        fixture.workspace.batch_size == 7;
    zero_status = vae_c_iic_pair_zero_grad(&fixture.pair);
    backward_status = forward_status == VAE_C_OP_OK ?
        vae_c_iic_pair_backward(
            &fixture.pair, &fixture.input_a, &fixture.input_b,
            &fixture.workspace,
            VAE_C_IIC_PAIR_BACKWARD_A_THEN_B
        ) : VAE_C_OP_INVALID_STATE;
    if (forward_status != VAE_C_OP_OK ||
        zero_status != VAE_C_OP_OK ||
        backward_status != VAE_C_OP_OK) {
        fprintf(stderr, "TAIL_STATUS forward=%s zero=%s backward=%s\n",
                vae_c_op_status_name(forward_status),
                vae_c_op_status_name(zero_status),
                vae_c_op_status_name(backward_status));
        fprintf(stderr, "TAIL_PROGRESS main_tracked=%llu over_tracked=%llu "
                "main_result=%d over_result=%d main_prob_rows=%d "
                "over_prob_rows=%d\n",
                (unsigned long long)fixture.pair.main_head.num_batches_tracked,
                (unsigned long long)fixture.pair.overcluster_head.num_batches_tracked,
                fixture.workspace.main_result.valid,
                fixture.workspace.over_result.valid,
                probability_rows(&fixture.workspace.main_a),
                probability_rows(&fixture.workspace.over_a));
        if (fixture.pair.overcluster_head.num_batches_tracked == 2) {
            VAECIICLossStatus loss_status = vae_c_iic_loss_forward(
                fixture.workspace.over_a.softmax_output.data, 7, 250,
                fixture.workspace.over_b.softmax_output.data, 7, 250,
                &fixture.workspace.over_result
            );
            fprintf(stderr, "TAIL_OVER_LOSS=%s\n",
                    vae_c_iic_loss_status_name(loss_status));
        }
        goto cleanup;
    }
    retry_status = vae_c_iic_pair_backward(
        &fixture.pair, &fixture.input_a, &fixture.input_b,
        &fixture.workspace, VAE_C_IIC_PAIR_BACKWARD_A_THEN_B
    );
    passed = fixture.pair.main_head.num_batches_tracked == 2 &&
             fixture.pair.overcluster_head.num_batches_tracked == 2 &&
             forward_batch_size_valid &&
             fixture.workspace.batch_size == 0 &&
             fixture.workspace.backward_pair == NULL &&
             fixture.workspace.backward_input_a == NULL &&
             fixture.workspace.backward_input_b == NULL &&
             fixture.workspace.main_a_generation == 0 &&
             fixture.workspace.main_b_generation == 0 &&
             fixture.workspace.over_a_generation == 0 &&
             fixture.workspace.over_b_generation == 0 &&
             retry_status == VAE_C_OP_INVALID_STATE &&
             fixture.workspace.main_result.class_count == 36 &&
             fixture.workspace.over_result.class_count == 250 &&
             probability_rows(&fixture.workspace.main_a) &&
             probability_rows(&fixture.workspace.over_b) &&
             memcmp(input_a_before, fixture.input_a.data,
                    sizeof(input_a_before)) == 0 &&
             memcmp(input_b_before, fixture.input_b.data,
                    sizeof(input_b_before)) == 0 &&
             all_finite(fixture.input_a.grad, 7 * TEST_INPUT_DIM) &&
             all_finite(fixture.input_b.grad, 7 * TEST_INPUT_DIM) &&
             head_gradients_finite(&fixture.pair.main_head) &&
             head_gradients_finite(&fixture.pair.overcluster_head);
cleanup:
    fixture_free(&fixture);
    return check("tail B=7 and real C=36/W=250 forward-backward", passed);
}

static int test_head_independence(void)
{
    Fixture original;
    Fixture changed;
    int passed = 0;
    fixture_init(&original);
    fixture_init(&changed);
    if (!fixture_allocate(&original, 4, 4, TEST_MAIN_DIM,
                          TEST_OVER_DIM) ||
        !fixture_allocate(&changed, 4, 4, TEST_MAIN_DIM,
                          TEST_OVER_DIM)) {
        goto cleanup;
    }
    changed.pair.main_head.linear1.weights.values[0] += 0.5f;
    if (vae_c_iic_pair_forward_train(
            &original.pair, &original.input_a, &original.input_b,
            &original.workspace) != VAE_C_OP_OK ||
        vae_c_iic_pair_forward_train(
            &changed.pair, &changed.input_a, &changed.input_b,
            &changed.workspace) != VAE_C_OP_OK) {
        goto cleanup;
    }
    passed =
        memcmp(original.workspace.over_a.softmax_output.data,
               changed.workspace.over_a.softmax_output.data,
               4 * TEST_OVER_DIM * sizeof(float)) == 0 &&
        memcmp(original.workspace.over_b.softmax_output.data,
               changed.workspace.over_b.softmax_output.data,
               4 * TEST_OVER_DIM * sizeof(float)) == 0 &&
        memcmp(original.pair.overcluster_head.batch_norm.running_mean,
               changed.pair.overcluster_head.batch_norm.running_mean,
               VAE_C_IIC_HEAD_HIDDEN_DIM * sizeof(float)) == 0;
cleanup:
    fixture_free(&changed);
    fixture_free(&original);
    return check("changing main leaves over output and BN unchanged",
                 passed);
}

int main(void)
{
    int failures = 0;
    failures += test_golden_and_backward_orders();
    failures += test_lifecycle_atomicity();
    failures += test_stale_and_invalid_order();
    failures += test_manual_branch_accumulation();
    failures += test_invalid_forward_preflight();
    failures += test_workspace_input_alias_preflight();
    failures += test_corrupt_over_result_atomicity();
    failures += test_tail_and_real_dimensions();
    failures += test_head_independence();
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
