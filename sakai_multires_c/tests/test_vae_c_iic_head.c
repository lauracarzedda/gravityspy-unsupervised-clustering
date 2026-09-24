#include "vae_c_iic_head.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define GOLDEN_BATCH 4
#define GOLDEN_INPUT_DIM 3
#define GOLDEN_OUTPUT_DIM 3
#define GOLDEN_ACTIVE_HIDDEN 4


typedef struct {
    double max_absolute;
    double max_relative;
    size_t worst_index;
} ErrorStats;


static const float golden_input[12] = {
    0.5f, -1.0f, 0.25f,
    -0.75f, 0.5f, 1.0f,
    1.25f, 0.25f, -0.5f,
    -0.25f, -0.75f, 0.75f
};

static const float golden_linear1[16] = {
    0.53125f, -0.3125f, -0.75f, -0.46875f,
    -0.375f, 0.90625f, -0.3125f, 0.71875f,
    0.5f, -0.65625f, 0.75f, -0.21875f,
    0.15625f, 0.28125f, -1.0f, -0.0625f
};

static const float golden_batch_norm[16] = {
    1.136945787f, 0.789850484f, -0.535218007f, -1.423229631f,
    -1.657952101f, -1.001993675f, -0.044991926f, 1.921361961f,
    1.040569998f, 1.295242426f, 1.145557130f, -0.719105085f,
    -0.019563684f, -0.083099235f, -0.815347197f, -0.279027244f
};

static const float golden_relu[16] = {
    1.136945787f, 0.789850484f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.921361961f,
    1.040569998f, 1.295242426f, 1.145557130f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f
};

static const float golden_logits[12] = {
    0.433510273f, 0.029075708f, 0.570467757f,
    0.182585123f, -1.085680980f, 1.388351225f,
    0.402169034f, 0.517033942f, 0.179963879f,
    0.0625f, -0.125f, 0.1875f
};

static const float golden_probabilities[12] = {
    0.355349176f, 0.237143712f, 0.407507112f,
    0.216420455f, 0.060883181f, 0.722696364f,
    0.342175963f, 0.383826269f, 0.273997768f,
    0.337589485f, 0.279871513f, 0.382539002f
};

static const float golden_coefficient[12] = {
    0.25f, -0.375f, 0.5f,
    -0.5f, 0.625f, 0.125f,
    0.375f, 0.25f, -0.625f,
    0.75f, -0.25f, 0.3125f
};

static const float golden_running_mean1[4] = {
    0.2453125f, -0.10703125f, 0.3046875f, -0.22578125f
};

static const float golden_running_variance1[4] = {
    1.367740885f, 0.722224935f, 1.859733073f, 1.151261393f
};

static const float golden_running_mean2[4] = {
    0.24109375f, -0.090859375f, 0.24140625f, -0.203984375f
};

static const float golden_running_variance2[4] = {
    1.248707682f, 0.697227376f, 1.733492839f, 1.062396647f
};

static const float golden_input_gradient[12] = {
    0.036989892f, -0.011847416f, 0.028032377f,
    -0.000562631f, -0.023332095f, 0.021029413f,
    -0.029671277f, 0.022705217f, -0.022998393f,
    -0.006755983f, 0.012474294f, -0.026063398f
};

static const float golden_linear1_gradient[12] = {
    -0.051321803f, -0.076845901f, 0.051628776f,
    -0.002881240f, -0.013928682f, 0.004800854f,
    0.032638396f, -0.025666703f, -0.022186376f,
    0.000498748f, 0.000045350f, 0.000111850f
};

static const float golden_gamma_gradient[4] = {
    0.049333230f, 0.051750214f, 0.164841968f, 0.035849613f
};

static const float golden_beta_gradient[4] = {
    0.054767711f, -0.062894859f, 0.102339097f, 0.021898382f
};

static const float golden_linear2_gradient[12] = {
    0.133363322f, 0.155705876f, 0.126208853f, -0.216301807f,
    -0.077346713f, -0.010461794f, 0.086609369f, 0.070751153f,
    -0.056016609f, -0.145244082f, -0.212818223f, 0.145550653f
};

static const float golden_linear2_bias_gradient[3] = {
    0.165042247f, -0.179501948f, 0.014459701f
};

static const float golden_eval_batch_norm[16] = {
    0.417113966f, 0.482256079f, -0.627242715f, -0.446089588f,
    -0.495250602f, -0.794865820f, -0.378026072f, 0.994028953f,
    0.385653119f, 0.842469947f, 0.227214348f, -0.142906737f,
    0.039583800f, -0.139931513f, -0.769652225f, 0.046582545f
};

static const float golden_eval_relu[16] = {
    0.417113966f, 0.482256079f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.994028953f,
    0.385653119f, 0.842469947f, 0.227214348f, 0.0f,
    0.039583800f, 0.0f, 0.0f, 0.046582545f
};

static const float golden_eval_logits[12] = {
    0.150492963f, 0.003706784f, 0.352060501f,
    0.124626810f, -0.622014476f, 0.808768095f,
    0.073110866f, 0.199523177f, 0.304016643f,
    0.085203309f, -0.153239247f, 0.226510040f
};

static const float golden_eval_probabilities[12] = {
    0.323960357f, 0.279732831f, 0.396306812f,
    0.289349772f, 0.137138992f, 0.573511235f,
    0.294595023f, 0.334291685f, 0.371113292f,
    0.340178655f, 0.268011095f, 0.391810250f
};


static int report_check(const char *name, int passed)
{
    printf("[%s] %s\n", passed ? "PASS" : "FAIL", name);
    return passed ? 0 : 1;
}


static int make_tensor(
    size_t rows,
    size_t columns,
    int with_gradient,
    VAECTensor *tensor
)
{
    const size_t dims[2] = {rows, columns};
    VAECShape shape;

    vae_c_tensor_init(tensor);
    return
        vae_c_shape_make(2, dims, &shape) == VAE_C_STORAGE_OK &&
        vae_c_tensor_allocate(tensor, &shape, with_gradient != 0) ==
            VAE_C_STORAGE_OK;
}


static int all_finite(const float *values, size_t count)
{
    size_t index;

    if (values == NULL) {
        return 0;
    }
    for (index = 0; index < count; index++) {
        if (!isfinite(values[index])) {
            return 0;
        }
    }
    return 1;
}


static uint64_t fnv1a64_update(
    uint64_t hash,
    const void *data,
    size_t size
)
{
    const unsigned char *bytes = (const unsigned char *)data;
    size_t index;

    for (index = 0; index < size; index++) {
        hash ^= (uint64_t)bytes[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}


static int compare_values(
    const char *name,
    const float *actual,
    const float *expected,
    size_t count,
    double absolute_tolerance,
    double relative_tolerance
)
{
    ErrorStats stats = {0.0, 0.0, 0};
    size_t index;
    int passed = 1;

    for (index = 0; index < count; index++) {
        double a = (double)actual[index];
        double e = (double)expected[index];
        double absolute = fabs(a - e);
        double scale = fmax(fabs(a), fabs(e));
        double relative = scale > 0.0 ? absolute / scale : 0.0;

        if (!isfinite(a) || !isfinite(e) || !isfinite(absolute) ||
            !isfinite(scale) || !isfinite(relative)) {

            passed = 0;
            continue;
        }
        if (absolute > stats.max_absolute) {
            stats.max_absolute = absolute;
            stats.worst_index = index;
        }
        if (relative > stats.max_relative) {
            stats.max_relative = relative;
        }
        if (absolute > absolute_tolerance + relative_tolerance * scale) {
            passed = 0;
        }
    }
    printf(
        "%s_MAX_ABS=%.9g MAX_REL=%.9g WORST_INDEX=%zu\n",
        name,
        stats.max_absolute,
        stats.max_relative,
        stats.worst_index
    );
    return passed;
}


static int compare_active_hidden(
    const char *name,
    const VAECTensor *tensor,
    const float expected_active[GOLDEN_BATCH * GOLDEN_ACTIVE_HIDDEN],
    float expected_inactive,
    double absolute_tolerance,
    double relative_tolerance
)
{
    float expected[GOLDEN_BATCH * VAE_C_IIC_HEAD_HIDDEN_DIM];
    size_t row;
    size_t hidden;

    for (row = 0; row < GOLDEN_BATCH; row++) {
        for (hidden = 0; hidden < VAE_C_IIC_HEAD_HIDDEN_DIM; hidden++) {
            expected[row * VAE_C_IIC_HEAD_HIDDEN_DIM + hidden] =
                hidden < GOLDEN_ACTIVE_HIDDEN
                    ? expected_active[
                        row * GOLDEN_ACTIVE_HIDDEN + hidden]
                    : expected_inactive;
        }
    }
    return compare_values(
        name,
        tensor->data,
        expected,
        GOLDEN_BATCH * VAE_C_IIC_HEAD_HIDDEN_DIM,
        absolute_tolerance,
        relative_tolerance
    );
}


static int compare_parameter_rows(
    const char *name,
    const float *actual,
    size_t rows,
    size_t columns,
    size_t active_rows,
    size_t active_columns,
    const float *expected_active,
    double absolute_tolerance,
    double relative_tolerance
)
{
    ErrorStats stats = {0.0, 0.0, 0};
    size_t row;
    size_t column;
    int passed = 1;

    for (row = 0; row < rows; row++) {
        for (column = 0; column < columns; column++) {
            size_t index = row * columns + column;
            float expected =
                row < active_rows && column < active_columns
                    ? expected_active[row * active_columns + column]
                    : 0.0f;
            double a = (double)actual[index];
            double e = (double)expected;
            double absolute = fabs(a - e);
            double scale = fmax(fabs(a), fabs(e));
            double relative = scale > 0.0 ? absolute / scale : 0.0;

            if (!isfinite(a) || !isfinite(e) || !isfinite(absolute) ||
                !isfinite(scale) || !isfinite(relative)) {

                passed = 0;
                continue;
            }
            if (absolute > stats.max_absolute) {
                stats.max_absolute = absolute;
                stats.worst_index = index;
            }
            if (relative > stats.max_relative) {
                stats.max_relative = relative;
            }
            if (absolute > absolute_tolerance + relative_tolerance * scale) {
                passed = 0;
            }
        }
    }
    printf(
        "%s_MAX_ABS=%.9g MAX_REL=%.9g WORST_INDEX=%zu\n",
        name,
        stats.max_absolute,
        stats.max_relative,
        stats.worst_index
    );
    return passed;
}


static void reset_golden_running_state(VAECIICHead *head)
{
    static const float mean[4] = {0.25f, -0.125f, 0.375f, -0.25f};
    static const float variance[4] = {1.5f, 0.75f, 2.0f, 1.25f};
    size_t hidden;

    for (hidden = 0; hidden < VAE_C_IIC_HEAD_HIDDEN_DIM; hidden++) {
        head->batch_norm.running_mean[hidden] =
            hidden < 4 ? mean[hidden] : 0.0f;
        head->batch_norm.running_variance[hidden] =
            hidden < 4 ? variance[hidden] : 1.0f;
    }
    head->num_batches_tracked = 0;
}


static void set_golden_parameters(VAECIICHead *head)
{
    static const float weights1[12] = {
        0.5f, -0.25f, 0.125f,
        -0.375f, 0.25f, 0.5f,
        0.25f, 0.75f, -0.5f,
        -0.125f, 0.5f, 0.375f
    };
    static const float gamma[4] = {1.125f, -0.875f, 0.75f, 1.25f};
    static const float beta[4] = {0.125f, 0.25f, -0.0625f, -0.125f};
    static const float weights2[12] = {
        0.5f, -0.25f, 0.125f, 0.0625f,
        -0.125f, 0.375f, 0.25f, -0.5f,
        0.25f, 0.125f, -0.375f, 0.625f
    };
    static const float bias2[3] = {0.0625f, -0.125f, 0.1875f};
    size_t hidden;
    size_t output;

    memset(
        head->linear1.weights.values,
        0,
        head->linear1.weights.shape.count * sizeof(float)
    );
    memcpy(head->linear1.weights.values, weights1, sizeof(weights1));
    for (hidden = 0; hidden < VAE_C_IIC_HEAD_HIDDEN_DIM; hidden++) {
        head->batch_norm.gamma.values[hidden] =
            hidden < 4 ? gamma[hidden] : 1.0f;
        head->batch_norm.beta.values[hidden] =
            hidden < 4 ? beta[hidden] : -1.0f;
    }
    memset(
        head->linear2.weights.values,
        0,
        head->linear2.weights.shape.count * sizeof(float)
    );
    for (output = 0; output < GOLDEN_OUTPUT_DIM; output++) {
        memcpy(
            head->linear2.weights.values +
                output * VAE_C_IIC_HEAD_HIDDEN_DIM,
            weights2 + output * GOLDEN_ACTIVE_HIDDEN,
            GOLDEN_ACTIVE_HIDDEN * sizeof(float)
        );
    }
    memcpy(head->linear2.bias.values, bias2, sizeof(bias2));
    reset_golden_running_state(head);
}


static int initialize_golden_fixture(
    VAECIICHead *head,
    VAECIICHeadWorkspace *workspace,
    VAECTensor *input
)
{
    size_t index;

    vae_c_iic_head_init(head);
    vae_c_iic_head_workspace_init(workspace);
    vae_c_tensor_init(input);
    if (vae_c_iic_head_allocate(
            head,
            GOLDEN_INPUT_DIM,
            GOLDEN_OUTPUT_DIM) != VAE_C_OP_OK ||
        vae_c_iic_head_workspace_allocate(head, 7, workspace) !=
            VAE_C_OP_OK ||
        !make_tensor(GOLDEN_BATCH, GOLDEN_INPUT_DIM, 1, input)) {

        return 0;
    }
    set_golden_parameters(head);
    for (index = 0; index < 12; index++) {
        input->data[index] = golden_input[index];
    }
    return 1;
}


static int check_running_vector(
    const char *name,
    const float *actual,
    const float expected_active[4],
    float expected_inactive
)
{
    float expected[VAE_C_IIC_HEAD_HIDDEN_DIM];
    size_t hidden;

    for (hidden = 0; hidden < VAE_C_IIC_HEAD_HIDDEN_DIM; hidden++) {
        expected[hidden] =
            hidden < GOLDEN_ACTIVE_HIDDEN
                ? expected_active[hidden]
                : expected_inactive;
    }
    return compare_values(
        name,
        actual,
        expected,
        VAE_C_IIC_HEAD_HIDDEN_DIM,
        3.0e-6,
        3.0e-6
    );
}


static int test_lifecycle_and_layout(void)
{
    VAECIICHead head;
    VAECIICHeadWorkspace workspace;
    size_t hidden;
    int passed = 1;

    vae_c_iic_head_init(&head);
    vae_c_iic_head_workspace_init(&workspace);
    if (!vae_c_iic_head_is_empty(&head) ||
        !vae_c_iic_head_workspace_is_empty(&workspace) ||
        vae_c_iic_head_allocate(&head, 5, 36) != VAE_C_OP_OK ||
        vae_c_iic_head_workspace_allocate(&head, 7, &workspace) !=
            VAE_C_OP_OK) {

        passed = 0;
        goto cleanup;
    }
    passed =
        vae_c_iic_head_validate(&head) == VAE_C_OP_OK &&
        vae_c_iic_head_workspace_validate(&head, &workspace) ==
            VAE_C_OP_OK &&
        head.input_dim == 5 &&
        head.output_dim == 36 &&
        head.linear1.in_features == 5 &&
        head.linear1.out_features == VAE_C_IIC_HEAD_HIDDEN_DIM &&
        !head.linear1.with_bias &&
        head.linear2.in_features == VAE_C_IIC_HEAD_HIDDEN_DIM &&
        head.linear2.out_features == 36 &&
        head.linear2.with_bias &&
        head.batch_norm.epsilon == 1.0e-5f &&
        head.batch_norm.momentum == 0.1f &&
        head.num_batches_tracked == 0 &&
        workspace.capacity == 7;
    for (hidden = 0;
         passed && hidden < VAE_C_IIC_HEAD_HIDDEN_DIM;
         hidden++) {

        passed =
            head.batch_norm.gamma.values[hidden] == 1.0f &&
            head.batch_norm.beta.values[hidden] == 0.0f &&
            head.batch_norm.running_mean[hidden] == 0.0f &&
            head.batch_norm.running_variance[hidden] == 1.0f;
    }

cleanup:
    vae_c_iic_head_workspace_free(&workspace);
    vae_c_iic_head_free(&head);
    return
        passed &&
        vae_c_iic_head_is_empty(&head) &&
        vae_c_iic_head_workspace_is_empty(&workspace);
}


static int test_embedded_reference_golden_train_backward_eval(void)
{
    VAECIICHead head;
    VAECIICHeadWorkspace workspace;
    VAECTensor input;
    float input_snapshot[12];
    float linear1_values_snapshot[
        VAE_C_IIC_HEAD_HIDDEN_DIM * GOLDEN_INPUT_DIM];
    float gamma_values_snapshot[VAE_C_IIC_HEAD_HIDDEN_DIM];
    float beta_values_snapshot[VAE_C_IIC_HEAD_HIDDEN_DIM];
    float linear2_values_snapshot[
        GOLDEN_OUTPUT_DIM * VAE_C_IIC_HEAD_HIDDEN_DIM];
    float linear2_bias_snapshot[GOLDEN_OUTPUT_DIM];
    float persistent_snapshot[4 * VAE_C_IIC_HEAD_HIDDEN_DIM];
    uint64_t counter_snapshot;
    uint64_t state_digest;
    double loss = 0.0;
    size_t index;
    int passed = 0;

    if (!initialize_golden_fixture(&head, &workspace, &input)) {
        return 0;
    }
    memcpy(input_snapshot, input.data, sizeof(input_snapshot));
    memcpy(
        linear1_values_snapshot,
        head.linear1.weights.values,
        sizeof(linear1_values_snapshot));
    memcpy(
        gamma_values_snapshot,
        head.batch_norm.gamma.values,
        sizeof(gamma_values_snapshot));
    memcpy(
        beta_values_snapshot,
        head.batch_norm.beta.values,
        sizeof(beta_values_snapshot));
    memcpy(
        linear2_values_snapshot,
        head.linear2.weights.values,
        sizeof(linear2_values_snapshot));
    memcpy(
        linear2_bias_snapshot,
        head.linear2.bias.values,
        sizeof(linear2_bias_snapshot));
    if (vae_c_iic_head_forward(
            &head,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &input,
            &workspace) != VAE_C_OP_OK) {

        goto cleanup;
    }
    passed =
        compare_active_hidden(
            "GOLDEN_LINEAR1",
            &workspace.linear1_output,
            golden_linear1,
            0.0f,
            2.0e-6,
            2.0e-6) &&
        compare_active_hidden(
            "GOLDEN_BATCH_NORM",
            &workspace.batch_norm_output,
            golden_batch_norm,
            -1.0f,
            2.0e-5,
            2.0e-5) &&
        compare_active_hidden(
            "GOLDEN_RELU",
            &workspace.relu_output,
            golden_relu,
            0.0f,
            2.0e-5,
            2.0e-5) &&
        compare_values(
            "GOLDEN_LOGITS",
            workspace.logits.data,
            golden_logits,
            12,
            2.0e-5,
            2.0e-5) &&
        compare_values(
            "GOLDEN_SOFTMAX",
            workspace.softmax_output.data,
            golden_probabilities,
            12,
            2.0e-5,
            2.0e-5) &&
        check_running_vector(
            "GOLDEN_RUNNING_MEAN_TRAIN1",
            head.batch_norm.running_mean,
            golden_running_mean1,
            0.0f) &&
        check_running_vector(
            "GOLDEN_RUNNING_VAR_TRAIN1",
            head.batch_norm.running_variance,
            golden_running_variance1,
            0.9f) &&
        head.num_batches_tracked == 1 &&
        memcmp(input.data, input_snapshot, sizeof(input_snapshot)) == 0;
    if (!passed) {
        goto cleanup;
    }
    for (index = 0; index < 12; index++) {
        loss +=
            (double)workspace.softmax_output.data[index] *
            (double)golden_coefficient[index];
    }
    printf("GOLDEN_TEST_LOSS=%.9g\n", loss);
    if (fabs(loss - 0.5796323861574104) > 2.0e-6 ||
        vae_c_iic_head_zero_grad(&head) != VAE_C_OP_OK ||
        vae_c_iic_head_workspace_zero_grad(&head, &workspace) !=
            VAE_C_OP_OK ||
        vae_c_tensor_zero_grad(&input) != VAE_C_STORAGE_OK) {

        passed = 0;
        goto cleanup;
    }
    memcpy(
        workspace.softmax_output.grad,
        golden_coefficient,
        sizeof(golden_coefficient)
    );
    if (vae_c_iic_head_backward(&head, &input, &workspace) !=
            VAE_C_OP_OK) {

        passed = 0;
        goto cleanup;
    }
    passed =
        compare_values(
            "GOLDEN_GRAD_INPUT",
            input.grad,
            golden_input_gradient,
            12,
            4.0e-5,
            4.0e-5) &&
        compare_parameter_rows(
            "GOLDEN_GRAD_LINEAR1",
            head.linear1.weights.gradients,
            VAE_C_IIC_HEAD_HIDDEN_DIM,
            GOLDEN_INPUT_DIM,
            GOLDEN_ACTIVE_HIDDEN,
            GOLDEN_INPUT_DIM,
            golden_linear1_gradient,
            5.0e-5,
            5.0e-5) &&
        compare_parameter_rows(
            "GOLDEN_GRAD_GAMMA",
            head.batch_norm.gamma.gradients,
            1,
            VAE_C_IIC_HEAD_HIDDEN_DIM,
            1,
            GOLDEN_ACTIVE_HIDDEN,
            golden_gamma_gradient,
            5.0e-5,
            5.0e-5) &&
        compare_parameter_rows(
            "GOLDEN_GRAD_BETA",
            head.batch_norm.beta.gradients,
            1,
            VAE_C_IIC_HEAD_HIDDEN_DIM,
            1,
            GOLDEN_ACTIVE_HIDDEN,
            golden_beta_gradient,
            5.0e-5,
            5.0e-5) &&
        compare_parameter_rows(
            "GOLDEN_GRAD_LINEAR2",
            head.linear2.weights.gradients,
            GOLDEN_OUTPUT_DIM,
            VAE_C_IIC_HEAD_HIDDEN_DIM,
            GOLDEN_OUTPUT_DIM,
            GOLDEN_ACTIVE_HIDDEN,
            golden_linear2_gradient,
            5.0e-5,
            5.0e-5) &&
        compare_values(
            "GOLDEN_GRAD_LINEAR2_BIAS",
            head.linear2.bias.gradients,
            golden_linear2_bias_gradient,
            GOLDEN_OUTPUT_DIM,
            5.0e-5,
            5.0e-5) &&
        memcmp(input.data, input_snapshot, sizeof(input_snapshot)) == 0;
    if (!passed || vae_c_iic_head_forward(
            &head,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &input,
            &workspace) != VAE_C_OP_OK) {

        passed = 0;
        goto cleanup;
    }
    passed =
        head.num_batches_tracked == 2 &&
        check_running_vector(
            "GOLDEN_RUNNING_MEAN_TRAIN2",
            head.batch_norm.running_mean,
            golden_running_mean2,
            0.0f) &&
        check_running_vector(
            "GOLDEN_RUNNING_VAR_TRAIN2",
            head.batch_norm.running_variance,
            golden_running_variance2,
            0.81f);
    if (!passed) {
        goto cleanup;
    }
    memcpy(
        persistent_snapshot,
        head.batch_norm.gamma.values,
        VAE_C_IIC_HEAD_HIDDEN_DIM * sizeof(float)
    );
    memcpy(
        persistent_snapshot + VAE_C_IIC_HEAD_HIDDEN_DIM,
        head.batch_norm.beta.values,
        VAE_C_IIC_HEAD_HIDDEN_DIM * sizeof(float)
    );
    memcpy(
        persistent_snapshot + 2 * VAE_C_IIC_HEAD_HIDDEN_DIM,
        head.batch_norm.running_mean,
        VAE_C_IIC_HEAD_HIDDEN_DIM * sizeof(float)
    );
    memcpy(
        persistent_snapshot + 3 * VAE_C_IIC_HEAD_HIDDEN_DIM,
        head.batch_norm.running_variance,
        VAE_C_IIC_HEAD_HIDDEN_DIM * sizeof(float)
    );
    counter_snapshot = head.num_batches_tracked;
    if (vae_c_iic_head_forward(
            &head,
            VAE_C_BATCH_NORM_MODE_EVAL,
            &input,
            &workspace) != VAE_C_OP_OK) {

        passed = 0;
        goto cleanup;
    }
    passed =
        compare_active_hidden(
            "GOLDEN_EVAL_BATCH_NORM",
            &workspace.batch_norm_output,
            golden_eval_batch_norm,
            -1.0f,
            2.0e-5,
            2.0e-5) &&
        compare_active_hidden(
            "GOLDEN_EVAL_RELU",
            &workspace.relu_output,
            golden_eval_relu,
            0.0f,
            2.0e-5,
            2.0e-5) &&
        compare_values(
            "GOLDEN_EVAL_LOGITS",
            workspace.logits.data,
            golden_eval_logits,
            12,
            2.0e-5,
            2.0e-5) &&
        compare_values(
            "GOLDEN_EVAL_SOFTMAX",
            workspace.softmax_output.data,
            golden_eval_probabilities,
            12,
            2.0e-5,
            2.0e-5) &&
        memcmp(
            persistent_snapshot,
            head.batch_norm.gamma.values,
            VAE_C_IIC_HEAD_HIDDEN_DIM * sizeof(float)) == 0 &&
        memcmp(
            persistent_snapshot + VAE_C_IIC_HEAD_HIDDEN_DIM,
            head.batch_norm.beta.values,
            VAE_C_IIC_HEAD_HIDDEN_DIM * sizeof(float)) == 0 &&
        memcmp(
            persistent_snapshot + 2 * VAE_C_IIC_HEAD_HIDDEN_DIM,
            head.batch_norm.running_mean,
            VAE_C_IIC_HEAD_HIDDEN_DIM * sizeof(float)) == 0 &&
        memcmp(
            persistent_snapshot + 3 * VAE_C_IIC_HEAD_HIDDEN_DIM,
            head.batch_norm.running_variance,
            VAE_C_IIC_HEAD_HIDDEN_DIM * sizeof(float)) == 0 &&
        head.num_batches_tracked == counter_snapshot &&
        memcmp(input.data, input_snapshot, sizeof(input_snapshot)) == 0 &&
        memcmp(
            head.linear1.weights.values,
            linear1_values_snapshot,
            sizeof(linear1_values_snapshot)) == 0 &&
        memcmp(
            head.batch_norm.gamma.values,
            gamma_values_snapshot,
            sizeof(gamma_values_snapshot)) == 0 &&
        memcmp(
            head.batch_norm.beta.values,
            beta_values_snapshot,
            sizeof(beta_values_snapshot)) == 0 &&
        memcmp(
            head.linear2.weights.values,
            linear2_values_snapshot,
            sizeof(linear2_values_snapshot)) == 0 &&
        memcmp(
            head.linear2.bias.values,
            linear2_bias_snapshot,
            sizeof(linear2_bias_snapshot)) == 0;
    if (passed) {
        state_digest = UINT64_C(14695981039346656037);
        state_digest = fnv1a64_update(
            state_digest,
            input.data,
            input.shape.count * sizeof(float));
        state_digest = fnv1a64_update(
            state_digest,
            input.grad,
            input.shape.count * sizeof(float));
        state_digest = fnv1a64_update(
            state_digest,
            head.linear1.weights.owned_block,
            head.linear1.weights.owned_bytes);
        state_digest = fnv1a64_update(
            state_digest,
            head.batch_norm.gamma.owned_block,
            head.batch_norm.gamma.owned_bytes);
        state_digest = fnv1a64_update(
            state_digest,
            head.batch_norm.beta.owned_block,
            head.batch_norm.beta.owned_bytes);
        state_digest = fnv1a64_update(
            state_digest,
            head.batch_norm.running_mean,
            2 * VAE_C_IIC_HEAD_HIDDEN_DIM * sizeof(float));
        state_digest = fnv1a64_update(
            state_digest,
            &head.num_batches_tracked,
            sizeof(head.num_batches_tracked));
        state_digest = fnv1a64_update(
            state_digest,
            head.linear2.weights.owned_block,
            head.linear2.weights.owned_bytes);
        state_digest = fnv1a64_update(
            state_digest,
            head.linear2.bias.owned_block,
            head.linear2.bias.owned_bytes);
        state_digest = fnv1a64_update(
            state_digest,
            workspace.linear1_output.data,
            workspace.linear1_output.shape.count * sizeof(float));
        state_digest = fnv1a64_update(
            state_digest,
            workspace.batch_norm_output.data,
            workspace.batch_norm_output.shape.count * sizeof(float));
        state_digest = fnv1a64_update(
            state_digest,
            workspace.relu_output.data,
            workspace.relu_output.shape.count * sizeof(float));
        state_digest = fnv1a64_update(
            state_digest,
            workspace.logits.data,
            workspace.logits.shape.count * sizeof(float));
        state_digest = fnv1a64_update(
            state_digest,
            workspace.softmax_output.data,
            workspace.softmax_output.shape.count * sizeof(float));
        printf(
            "GOLDEN_STATE_FNV64=%016llx\n",
            (unsigned long long)state_digest
        );
    }

cleanup:
    vae_c_tensor_free(&input);
    vae_c_iic_head_workspace_free(&workspace);
    vae_c_iic_head_free(&head);
    return passed;
}


static double golden_objective(
    VAECIICHead *head,
    VAECTensor *input,
    VAECIICHeadWorkspace *workspace
)
{
    double result = NAN;
    size_t index;

    reset_golden_running_state(head);
    if (vae_c_iic_head_forward(
            head,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            input,
            workspace) != VAE_C_OP_OK) {

        return NAN;
    }
    result = 0.0;
    for (index = 0; index < 12; index++) {
        result +=
            (double)workspace->softmax_output.data[index] *
            (double)golden_coefficient[index];
    }
    return result;
}


static int finite_difference_one(
    const char *name,
    VAECIICHead *head,
    VAECTensor *input,
    VAECIICHeadWorkspace *workspace,
    float *value,
    double analytic,
    double *maximum_error
)
{
    const float step = 1.0e-3f;
    float original = *value;
    double plus;
    double minus;
    double numeric;
    double error;

    *value = original + step;
    plus = golden_objective(head, input, workspace);
    *value = original - step;
    minus = golden_objective(head, input, workspace);
    *value = original;
    numeric = (plus - minus) / (2.0 * (double)step);
    error = fabs(numeric - analytic);
    if (error > *maximum_error) {
        *maximum_error = error;
    }
    printf(
        "FINITE_DIFFERENCE_%s_ANALYTIC=%.9g NUMERIC=%.9g ABS=%.9g\n",
        name,
        analytic,
        numeric,
        error
    );
    return
        isfinite(plus) &&
        isfinite(minus) &&
        isfinite(numeric) &&
        isfinite(analytic) &&
        error <= 2.5e-4 + 5.0e-3 * fmax(fabs(numeric), fabs(analytic));
}


static int test_representative_finite_difference(void)
{
    VAECIICHead head;
    VAECIICHeadWorkspace workspace;
    VAECTensor input;
    double maximum_error = 0.0;
    double analytic[6];
    size_t index;
    int passed = 0;

    if (!initialize_golden_fixture(&head, &workspace, &input)) {
        return 0;
    }
    if (!isfinite(golden_objective(&head, &input, &workspace)) ||
        vae_c_iic_head_zero_grad(&head) != VAE_C_OP_OK ||
        vae_c_iic_head_workspace_zero_grad(&head, &workspace) !=
            VAE_C_OP_OK ||
        vae_c_tensor_zero_grad(&input) != VAE_C_STORAGE_OK) {

        goto cleanup;
    }
    for (index = 0; index < 12; index++) {
        workspace.softmax_output.grad[index] = golden_coefficient[index];
    }
    if (vae_c_iic_head_backward(&head, &input, &workspace) !=
            VAE_C_OP_OK) {

        goto cleanup;
    }
    analytic[0] = input.grad[0];
    analytic[1] = head.linear1.weights.gradients[0];
    analytic[2] = head.batch_norm.gamma.gradients[0];
    analytic[3] = head.batch_norm.beta.gradients[0];
    analytic[4] = head.linear2.weights.gradients[0];
    analytic[5] = head.linear2.bias.gradients[0];

    passed =
        finite_difference_one(
            "INPUT_0", &head, &input, &workspace, &input.data[0],
            analytic[0], &maximum_error) &&
        finite_difference_one(
            "LINEAR1_WEIGHT_0", &head, &input, &workspace,
            &head.linear1.weights.values[0],
            analytic[1], &maximum_error) &&
        finite_difference_one(
            "BN_GAMMA_0", &head, &input, &workspace,
            &head.batch_norm.gamma.values[0],
            analytic[2], &maximum_error) &&
        finite_difference_one(
            "BN_BETA_0", &head, &input, &workspace,
            &head.batch_norm.beta.values[0],
            analytic[3], &maximum_error) &&
        finite_difference_one(
            "LINEAR2_WEIGHT_0", &head, &input, &workspace,
            &head.linear2.weights.values[0],
            analytic[4], &maximum_error) &&
        finite_difference_one(
            "LINEAR2_BIAS_0", &head, &input, &workspace,
            &head.linear2.bias.values[0],
            analytic[5], &maximum_error);
    printf("FINITE_DIFFERENCE_MAX_ABS=%.9g\n", maximum_error);

cleanup:
    vae_c_tensor_free(&input);
    vae_c_iic_head_workspace_free(&workspace);
    vae_c_iic_head_free(&head);
    return passed;
}


static void set_dimension_parameters(VAECIICHead *head)
{
    size_t index;

    for (index = 0; index < head->linear1.weights.shape.count; index++) {
        int centered = (int)(index % 7) - 3;
        head->linear1.weights.values[index] =
            (float)centered * 2.0e-3f;
    }
    for (index = 0; index < VAE_C_IIC_HEAD_HIDDEN_DIM; index++) {
        head->batch_norm.gamma.values[index] =
            1.0f + (float)((int)(index % 5) - 2) * 0.01f;
        head->batch_norm.beta.values[index] =
            (float)((int)(index % 3) - 1) * 0.02f;
    }
    for (index = 0; index < head->linear2.weights.shape.count; index++) {
        int centered = (int)(index % 11) - 5;
        head->linear2.weights.values[index] =
            (float)centered * 1.0e-3f;
    }
    for (index = 0; index < head->output_dim; index++) {
        head->linear2.bias.values[index] =
            (float)((int)(index % 5) - 2) * 0.01f;
    }
}


static int run_dimension_case(size_t output_dim)
{
    const size_t input_dim = 5;
    const size_t batch = 7;
    VAECIICHead head;
    VAECIICHeadWorkspace workspace;
    VAECTensor input;
    float input_snapshot[7 * 5];
    size_t row;
    size_t column;
    int passed = 0;

    vae_c_iic_head_init(&head);
    vae_c_iic_head_workspace_init(&workspace);
    vae_c_tensor_init(&input);
    if (vae_c_iic_head_allocate(&head, input_dim, output_dim) !=
            VAE_C_OP_OK ||
        vae_c_iic_head_workspace_allocate(&head, 9, &workspace) !=
            VAE_C_OP_OK ||
        !make_tensor(batch, input_dim, 1, &input)) {

        goto cleanup;
    }
    set_dimension_parameters(&head);
    for (row = 0; row < batch; row++) {
        for (column = 0; column < input_dim; column++) {
            input.data[row * input_dim + column] =
                (float)((int)(row * 3 + column * 5) % 13 - 6) * 0.125f;
        }
    }
    memcpy(input_snapshot, input.data, sizeof(input_snapshot));
    if (vae_c_iic_head_forward(
            &head,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &input,
            &workspace) != VAE_C_OP_OK ||
        head.num_batches_tracked != 1 ||
        workspace.batch_size != batch ||
        !all_finite(
            workspace.softmax_output.data,
            batch * output_dim)) {

        goto cleanup;
    }
    for (row = 0; row < batch; row++) {
        double sum = 0.0;

        for (column = 0; column < output_dim; column++) {
            sum += workspace.softmax_output.data[row * output_dim + column];
        }
        if (fabs(sum - 1.0) > 2.0e-5) {
            goto cleanup;
        }
    }
    if (vae_c_iic_head_zero_grad(&head) != VAE_C_OP_OK ||
        vae_c_iic_head_workspace_zero_grad(&head, &workspace) !=
            VAE_C_OP_OK ||
        vae_c_tensor_zero_grad(&input) != VAE_C_STORAGE_OK) {

        goto cleanup;
    }
    for (row = 0; row < batch; row++) {
        for (column = 0; column < output_dim; column++) {
            workspace.softmax_output.grad[row * output_dim + column] =
                (float)((int)((row + column) % 7) - 3) * 0.05f;
        }
    }
    if (vae_c_iic_head_backward(&head, &input, &workspace) !=
            VAE_C_OP_OK ||
        !all_finite(input.grad, batch * input_dim) ||
        !all_finite(
            head.linear1.weights.gradients,
            head.linear1.weights.shape.count) ||
        !all_finite(
            head.batch_norm.gamma.gradients,
            VAE_C_IIC_HEAD_HIDDEN_DIM) ||
        !all_finite(
            head.batch_norm.beta.gradients,
            VAE_C_IIC_HEAD_HIDDEN_DIM) ||
        !all_finite(
            head.linear2.weights.gradients,
            head.linear2.weights.shape.count) ||
        !all_finite(head.linear2.bias.gradients, output_dim) ||
        memcmp(input.data, input_snapshot, sizeof(input_snapshot)) != 0) {

        goto cleanup;
    }
    passed = 1;

cleanup:
    vae_c_tensor_free(&input);
    vae_c_iic_head_workspace_free(&workspace);
    vae_c_iic_head_free(&head);
    return passed;
}


static int test_tail_and_output_dimensions(void)
{
    int main_pass = run_dimension_case(36);
    int overcluster_pass = run_dimension_case(250);

    printf("TAIL_BATCH=7\n");
    printf("OUTPUT_DIM_36=%s\n", main_pass ? "PASS" : "FAIL");
    printf("OUTPUT_DIM_250=%s\n", overcluster_pass ? "PASS" : "FAIL");
    return main_pass && overcluster_pass;
}


typedef struct {
    float linear1[VAE_C_IIC_HEAD_HIDDEN_DIM * GOLDEN_INPUT_DIM];
    float gamma[VAE_C_IIC_HEAD_HIDDEN_DIM];
    float beta[VAE_C_IIC_HEAD_HIDDEN_DIM];
    float linear2[GOLDEN_OUTPUT_DIM * VAE_C_IIC_HEAD_HIDDEN_DIM];
    float bias[GOLDEN_OUTPUT_DIM];
    float input[GOLDEN_BATCH * GOLDEN_INPUT_DIM];
} DualGradientSnapshot;


static void set_dual_input(VAECTensor *input)
{
    size_t index;

    for (index = 0; index < input->shape.count; index++) {
        input->data[index] =
            -0.375f * golden_input[index] +
            (float)((int)(index % 5) - 2) * 0.125f;
    }
}


static void seed_dual_upstream(
    VAECIICHeadWorkspace *workspace,
    int branch_b
)
{
    size_t index;

    for (index = 0; index < GOLDEN_BATCH * GOLDEN_OUTPUT_DIM; index++) {
        workspace->softmax_output.grad[index] = branch_b
            ? -0.625f * golden_coefficient[index] +
                (float)((int)(index % 3) - 1) * 0.03125f
            : golden_coefficient[index];
    }
}


static void capture_dual_gradients(
    const VAECIICHead *head,
    const VAECTensor *input,
    DualGradientSnapshot *snapshot
)
{
    memcpy(snapshot->linear1, head->linear1.weights.gradients,
           sizeof(snapshot->linear1));
    memcpy(snapshot->gamma, head->batch_norm.gamma.gradients,
           sizeof(snapshot->gamma));
    memcpy(snapshot->beta, head->batch_norm.beta.gradients,
           sizeof(snapshot->beta));
    memcpy(snapshot->linear2, head->linear2.weights.gradients,
           sizeof(snapshot->linear2));
    memcpy(snapshot->bias, head->linear2.bias.gradients,
           sizeof(snapshot->bias));
    memcpy(snapshot->input, input->grad, sizeof(snapshot->input));
}


static int compare_dual_parameter_gradients(
    const char *name,
    const DualGradientSnapshot *actual,
    const DualGradientSnapshot *expected
)
{
    return
        compare_values(name, actual->linear1, expected->linear1,
                       VAE_C_IIC_HEAD_HIDDEN_DIM * GOLDEN_INPUT_DIM,
                       2.0e-5, 2.0e-5) &&
        compare_values(name, actual->gamma, expected->gamma,
                       VAE_C_IIC_HEAD_HIDDEN_DIM, 2.0e-5, 2.0e-5) &&
        compare_values(name, actual->beta, expected->beta,
                       VAE_C_IIC_HEAD_HIDDEN_DIM, 2.0e-5, 2.0e-5) &&
        compare_values(name, actual->linear2, expected->linear2,
                       GOLDEN_OUTPUT_DIM * VAE_C_IIC_HEAD_HIDDEN_DIM,
                       2.0e-5, 2.0e-5) &&
        compare_values(name, actual->bias, expected->bias,
                       GOLDEN_OUTPUT_DIM, 2.0e-5, 2.0e-5);
}


static void add_dual_parameter_gradients(
    const DualGradientSnapshot *a,
    const DualGradientSnapshot *b,
    DualGradientSnapshot *sum
)
{
    size_t index;

    for (index = 0; index < VAE_C_IIC_HEAD_HIDDEN_DIM *
             GOLDEN_INPUT_DIM; index++) {
        sum->linear1[index] = a->linear1[index] + b->linear1[index];
    }
    for (index = 0; index < VAE_C_IIC_HEAD_HIDDEN_DIM; index++) {
        sum->gamma[index] = a->gamma[index] + b->gamma[index];
        sum->beta[index] = a->beta[index] + b->beta[index];
    }
    for (index = 0; index < GOLDEN_OUTPUT_DIM *
             VAE_C_IIC_HEAD_HIDDEN_DIM; index++) {
        sum->linear2[index] = a->linear2[index] + b->linear2[index];
    }
    for (index = 0; index < GOLDEN_OUTPUT_DIM; index++) {
        sum->bias[index] = a->bias[index] + b->bias[index];
    }
}


static int dual_snapshot_identical(
    const DualGradientSnapshot *a,
    const DualGradientSnapshot *b
)
{
    return
        memcmp(a->linear1, b->linear1, sizeof(a->linear1)) == 0 &&
        memcmp(a->gamma, b->gamma, sizeof(a->gamma)) == 0 &&
        memcmp(a->beta, b->beta, sizeof(a->beta)) == 0 &&
        memcmp(a->linear2, b->linear2, sizeof(a->linear2)) == 0 &&
        memcmp(a->bias, b->bias, sizeof(a->bias)) == 0 &&
        memcmp(a->input, b->input, sizeof(a->input)) == 0;
}


static int run_single_dual_reference(
    int branch_b,
    DualGradientSnapshot *snapshot
)
{
    VAECIICHead head;
    VAECIICHeadWorkspace workspace;
    VAECTensor input;
    int passed = 0;

    if (!initialize_golden_fixture(&head, &workspace, &input)) {
        goto cleanup;
    }
    if (branch_b) {
        set_dual_input(&input);
    }
    if (vae_c_iic_head_forward(&head, VAE_C_BATCH_NORM_MODE_TRAIN,
                               &input, &workspace) != VAE_C_OP_OK ||
        vae_c_iic_head_zero_grad(&head) != VAE_C_OP_OK ||
        vae_c_iic_head_workspace_zero_grad(&head, &workspace) !=
            VAE_C_OP_OK ||
        vae_c_tensor_zero_grad(&input) != VAE_C_STORAGE_OK) {
        goto cleanup;
    }
    seed_dual_upstream(&workspace, branch_b);
    if (vae_c_iic_head_backward(&head, &input, &workspace) != VAE_C_OP_OK) {
        goto cleanup;
    }
    capture_dual_gradients(&head, &input, snapshot);
    passed = 1;

cleanup:
    vae_c_tensor_free(&input);
    vae_c_iic_head_workspace_free(&workspace);
    vae_c_iic_head_free(&head);
    return passed;
}


static int run_dual_backward_order(
    int b_first,
    const DualGradientSnapshot *reference_a,
    const DualGradientSnapshot *reference_b,
    const DualGradientSnapshot *reference_sum
)
{
    VAECIICHead head;
    VAECIICHeadWorkspace workspace_a;
    VAECIICHeadWorkspace workspace_b;
    VAECTensor input_a;
    VAECTensor input_b;
    DualGradientSnapshot first_grad;
    DualGradientSnapshot total_grad;
    float a_probabilities[GOLDEN_BATCH * GOLDEN_OUTPUT_DIM];
    float a_mean[VAE_C_IIC_HEAD_HIDDEN_DIM];
    float a_inv_std[VAE_C_IIC_HEAD_HIDDEN_DIM];
    float latest_mean[VAE_C_IIC_HEAD_HIDDEN_DIM];
    float latest_inv_std[VAE_C_IIC_HEAD_HIDDEN_DIM];
    VAECShape latest_input_shape;
    VAECShape latest_output_shape;
    size_t latest_sample_count;
    float running_mean[VAE_C_IIC_HEAD_HIDDEN_DIM];
    float running_variance[VAE_C_IIC_HEAD_HIDDEN_DIM];
    float input_a_data[GOLDEN_BATCH * GOLDEN_INPUT_DIM];
    float input_b_data[GOLDEN_BATCH * GOLDEN_INPUT_DIM];
    const VAECTensor *first_input;
    const VAECTensor *second_input;
    VAECIICHeadWorkspace *first_workspace;
    VAECIICHeadWorkspace *second_workspace;
    const DualGradientSnapshot *first_reference;
    int passed = 0;

    vae_c_iic_head_workspace_init(&workspace_b);
    vae_c_tensor_init(&input_b);
    if (!initialize_golden_fixture(&head, &workspace_a, &input_a) ||
        vae_c_iic_head_workspace_allocate(&head, 7, &workspace_b) !=
            VAE_C_OP_OK ||
        !make_tensor(GOLDEN_BATCH, GOLDEN_INPUT_DIM, 1, &input_b)) {
        goto cleanup;
    }
    set_dual_input(&input_b);
    memcpy(input_a_data, input_a.data, sizeof(input_a_data));
    memcpy(input_b_data, input_b.data, sizeof(input_b_data));
    if (vae_c_iic_head_forward(&head, VAE_C_BATCH_NORM_MODE_TRAIN,
                               &input_a, &workspace_a) != VAE_C_OP_OK ||
        head.num_batches_tracked != 1 ||
        !workspace_a.batch_norm_cache.valid) {
        goto cleanup;
    }
    memcpy(a_probabilities, workspace_a.softmax_output.data,
           sizeof(a_probabilities));
    memcpy(a_mean, workspace_a.batch_norm_cache.batch_mean,
           sizeof(a_mean));
    memcpy(a_inv_std, workspace_a.batch_norm_cache.batch_inv_std,
           sizeof(a_inv_std));
    if (vae_c_iic_head_forward(&head, VAE_C_BATCH_NORM_MODE_TRAIN,
                               &input_b, &workspace_b) != VAE_C_OP_OK ||
        head.num_batches_tracked != 2 ||
        !workspace_a.batch_norm_cache.valid ||
        !workspace_b.batch_norm_cache.valid ||
        memcmp(a_probabilities, workspace_a.softmax_output.data,
               sizeof(a_probabilities)) != 0 ||
        memcmp(a_mean, workspace_a.batch_norm_cache.batch_mean,
               sizeof(a_mean)) != 0 ||
        memcmp(a_inv_std, workspace_a.batch_norm_cache.batch_inv_std,
               sizeof(a_inv_std)) != 0 ||
        memcmp(a_mean, workspace_b.batch_norm_cache.batch_mean,
               sizeof(a_mean)) == 0) {
        goto cleanup;
    }
    memcpy(running_mean, head.batch_norm.running_mean,
           sizeof(running_mean));
    memcpy(running_variance, head.batch_norm.running_variance,
           sizeof(running_variance));
    memcpy(latest_mean, head.batch_norm.batch_mean, sizeof(latest_mean));
    memcpy(latest_inv_std, head.batch_norm.batch_inv_std,
           sizeof(latest_inv_std));
    latest_input_shape = head.batch_norm.last_input_shape;
    latest_output_shape = head.batch_norm.last_output_shape;
    latest_sample_count = head.batch_norm.last_sample_count;
    if (vae_c_iic_head_zero_grad(&head) != VAE_C_OP_OK ||
        vae_c_iic_head_workspace_zero_grad(&head, &workspace_a) !=
            VAE_C_OP_OK ||
        vae_c_iic_head_workspace_zero_grad(&head, &workspace_b) !=
            VAE_C_OP_OK ||
        vae_c_tensor_zero_grad(&input_a) != VAE_C_STORAGE_OK ||
        vae_c_tensor_zero_grad(&input_b) != VAE_C_STORAGE_OK) {
        goto cleanup;
    }
    seed_dual_upstream(&workspace_a, 0);
    seed_dual_upstream(&workspace_b, 1);
    first_input = b_first ? &input_b : &input_a;
    second_input = b_first ? &input_a : &input_b;
    first_workspace = b_first ? &workspace_b : &workspace_a;
    second_workspace = b_first ? &workspace_a : &workspace_b;
    first_reference = b_first ? reference_b : reference_a;
    if (vae_c_iic_head_backward(&head, first_input, first_workspace) !=
            VAE_C_OP_OK) {
        goto cleanup;
    }
    capture_dual_gradients(&head, first_input, &first_grad);
    if (!compare_dual_parameter_gradients("DUAL_FIRST_GRAD",
                                          &first_grad, first_reference) ||
        !compare_values("DUAL_FIRST_INPUT_GRAD", first_grad.input,
                        first_reference->input,
                        GOLDEN_BATCH * GOLDEN_INPUT_DIM,
                        2.0e-5, 2.0e-5) ||
        memcmp(latest_mean, head.batch_norm.batch_mean,
               sizeof(latest_mean)) != 0 ||
        memcmp(latest_inv_std, head.batch_norm.batch_inv_std,
               sizeof(latest_inv_std)) != 0 ||
        head.batch_norm.last_sample_count != latest_sample_count ||
        !vae_c_shape_equal(&head.batch_norm.last_input_shape,
                           &latest_input_shape) ||
        !vae_c_shape_equal(&head.batch_norm.last_output_shape,
                           &latest_output_shape) ||
        vae_c_iic_head_backward(&head, first_input, first_workspace) ==
            VAE_C_OP_OK) {
        goto cleanup;
    }
    capture_dual_gradients(&head, first_input, &total_grad);
    if (!compare_dual_parameter_gradients("DUAL_CONSUME_ATOMIC",
                                          &total_grad, &first_grad) ||
        vae_c_iic_head_backward(&head, second_input, second_workspace) !=
            VAE_C_OP_OK) {
        goto cleanup;
    }
    capture_dual_gradients(&head, second_input, &total_grad);
    if (!compare_dual_parameter_gradients("DUAL_TOTAL_GRAD",
                                          &total_grad, reference_sum) ||
        !compare_values("DUAL_SECOND_INPUT_GRAD", total_grad.input,
                        (b_first ? reference_a : reference_b)->input,
                        GOLDEN_BATCH * GOLDEN_INPUT_DIM,
                        2.0e-5, 2.0e-5) ||
        memcmp(running_mean, head.batch_norm.running_mean,
               sizeof(running_mean)) != 0 ||
        memcmp(running_variance, head.batch_norm.running_variance,
               sizeof(running_variance)) != 0 ||
        memcmp(latest_mean, head.batch_norm.batch_mean,
               sizeof(latest_mean)) != 0 ||
        memcmp(latest_inv_std, head.batch_norm.batch_inv_std,
               sizeof(latest_inv_std)) != 0 ||
        head.num_batches_tracked != 2 ||
        memcmp(input_a_data, input_a.data, sizeof(input_a_data)) != 0 ||
        memcmp(input_b_data, input_b.data, sizeof(input_b_data)) != 0) {
        goto cleanup;
    }
    passed = 1;

cleanup:
    vae_c_tensor_free(&input_b);
    vae_c_tensor_free(&input_a);
    vae_c_iic_head_workspace_free(&workspace_b);
    vae_c_iic_head_workspace_free(&workspace_a);
    vae_c_iic_head_free(&head);
    return passed;
}


static int test_dual_train_contexts(void)
{
    DualGradientSnapshot reference_a;
    DualGradientSnapshot reference_b;
    DualGradientSnapshot reference_sum;

    if (!run_single_dual_reference(0, &reference_a) ||
        !run_single_dual_reference(1, &reference_b)) {
        return 0;
    }
    add_dual_parameter_gradients(&reference_a, &reference_b,
                                 &reference_sum);
    return run_dual_backward_order(0, &reference_a, &reference_b,
                                   &reference_sum) &&
           run_dual_backward_order(1, &reference_a, &reference_b,
                                   &reference_sum);
}


static int test_context_replacement_is_atomic(void)
{
    VAECIICHead head;
    VAECIICHeadWorkspace workspace;
    VAECTensor input_a;
    VAECTensor input_b;
    DualGradientSnapshot before;
    DualGradientSnapshot after;
    int passed = 0;

    vae_c_tensor_init(&input_b);
    if (!initialize_golden_fixture(&head, &workspace, &input_a) ||
        !make_tensor(GOLDEN_BATCH, GOLDEN_INPUT_DIM, 1, &input_b)) {
        goto cleanup;
    }
    set_dual_input(&input_b);
    if (vae_c_iic_head_forward(&head, VAE_C_BATCH_NORM_MODE_TRAIN,
                               &input_a, &workspace) != VAE_C_OP_OK ||
        vae_c_iic_head_forward(&head, VAE_C_BATCH_NORM_MODE_TRAIN,
                               &input_b, &workspace) != VAE_C_OP_OK ||
        head.num_batches_tracked != 2 ||
        vae_c_iic_head_zero_grad(&head) != VAE_C_OP_OK ||
        vae_c_iic_head_workspace_zero_grad(&head, &workspace) !=
            VAE_C_OP_OK ||
        vae_c_tensor_zero_grad(&input_a) != VAE_C_STORAGE_OK ||
        vae_c_tensor_zero_grad(&input_b) != VAE_C_STORAGE_OK) {
        goto cleanup;
    }
    seed_dual_upstream(&workspace, 1);
    capture_dual_gradients(&head, &input_a, &before);
    if (vae_c_iic_head_backward(&head, &input_a, &workspace) ==
            VAE_C_OP_OK) {
        goto cleanup;
    }
    capture_dual_gradients(&head, &input_a, &after);
    if (!dual_snapshot_identical(&before, &after) ||
        vae_c_iic_head_backward(&head, &input_b, &workspace) !=
            VAE_C_OP_OK ||
        !all_finite(input_b.grad, GOLDEN_BATCH * GOLDEN_INPUT_DIM)) {
        goto cleanup;
    }
    passed = 1;

cleanup:
    vae_c_tensor_free(&input_b);
    vae_c_tensor_free(&input_a);
    vae_c_iic_head_workspace_free(&workspace);
    vae_c_iic_head_free(&head);
    return passed;
}


static int run_dual_tail_dimension_case(size_t output_dim)
{
    const size_t input_dim = 5;
    const size_t batch = 7;
    VAECIICHead head;
    VAECIICHeadWorkspace workspace_a;
    VAECIICHeadWorkspace workspace_b;
    VAECTensor input_a;
    VAECTensor input_b;
    size_t index;
    int passed = 0;

    vae_c_iic_head_init(&head);
    vae_c_iic_head_workspace_init(&workspace_a);
    vae_c_iic_head_workspace_init(&workspace_b);
    vae_c_tensor_init(&input_a);
    vae_c_tensor_init(&input_b);
    if (vae_c_iic_head_allocate(&head, input_dim, output_dim) !=
            VAE_C_OP_OK ||
        vae_c_iic_head_workspace_allocate(&head, 9, &workspace_a) !=
            VAE_C_OP_OK ||
        vae_c_iic_head_workspace_allocate(&head, 9, &workspace_b) !=
            VAE_C_OP_OK ||
        !make_tensor(batch, input_dim, 1, &input_a) ||
        !make_tensor(batch, input_dim, 1, &input_b)) {
        goto cleanup;
    }
    set_dimension_parameters(&head);
    for (index = 0; index < batch * input_dim; index++) {
        input_a.data[index] =
            (float)((int)(index % 13) - 6) * 0.125f;
        input_b.data[index] =
            (float)((int)((index * 3) % 17) - 8) * 0.09375f;
    }
    if (vae_c_iic_head_forward(&head, VAE_C_BATCH_NORM_MODE_TRAIN,
                               &input_a, &workspace_a) != VAE_C_OP_OK ||
        vae_c_iic_head_forward(&head, VAE_C_BATCH_NORM_MODE_TRAIN,
                               &input_b, &workspace_b) != VAE_C_OP_OK ||
        head.num_batches_tracked != 2 ||
        vae_c_iic_head_zero_grad(&head) != VAE_C_OP_OK ||
        vae_c_iic_head_workspace_zero_grad(&head, &workspace_a) !=
            VAE_C_OP_OK ||
        vae_c_iic_head_workspace_zero_grad(&head, &workspace_b) !=
            VAE_C_OP_OK) {
        goto cleanup;
    }
    for (index = 0; index < batch * output_dim; index++) {
        workspace_a.softmax_output.grad[index] =
            (float)((int)(index % 7) - 3) * 0.03125f;
        workspace_b.softmax_output.grad[index] =
            (float)((int)(index % 11) - 5) * 0.0234375f;
    }
    if (vae_c_iic_head_backward(&head, &input_b, &workspace_b) !=
            VAE_C_OP_OK ||
        vae_c_iic_head_backward(&head, &input_a, &workspace_a) !=
            VAE_C_OP_OK ||
        !all_finite(input_a.grad, batch * input_dim) ||
        !all_finite(input_b.grad, batch * input_dim) ||
        !all_finite(head.linear1.weights.gradients,
                    head.linear1.weights.shape.count) ||
        !all_finite(head.batch_norm.gamma.gradients,
                    VAE_C_IIC_HEAD_HIDDEN_DIM) ||
        !all_finite(head.linear2.weights.gradients,
                    head.linear2.weights.shape.count)) {
        goto cleanup;
    }
    passed = 1;

cleanup:
    vae_c_tensor_free(&input_b);
    vae_c_tensor_free(&input_a);
    vae_c_iic_head_workspace_free(&workspace_b);
    vae_c_iic_head_workspace_free(&workspace_a);
    vae_c_iic_head_free(&head);
    return passed;
}


static int test_dual_tail_and_output_dimensions(void)
{
    int main_pass = run_dual_tail_dimension_case(36);
    int overcluster_pass = run_dual_tail_dimension_case(250);

    printf("DUAL_TAIL_BATCH=7\n");
    printf("DUAL_OUTPUT_DIM_36=%s\n", main_pass ? "PASS" : "FAIL");
    printf("DUAL_OUTPUT_DIM_250=%s\n",
           overcluster_pass ? "PASS" : "FAIL");
    return main_pass && overcluster_pass;
}


static int test_invalid_inputs(void)
{
    VAECIICHead head;
    VAECIICHead fresh;
    VAECIICHeadWorkspace workspace;
    VAECIICHeadWorkspace small_workspace;
    VAECTensor input;
    VAECTensor singleton;
    VAECTensor too_large;
    VAECTensor wrong_columns;
    float saved;
    uint64_t saved_generation;
    uint64_t saved_counter;
    size_t index;
    int passed = 0;

    vae_c_iic_head_init(&head);
    vae_c_iic_head_init(&fresh);
    vae_c_iic_head_workspace_init(&workspace);
    vae_c_iic_head_workspace_init(&small_workspace);
    vae_c_tensor_init(&input);
    vae_c_tensor_init(&singleton);
    vae_c_tensor_init(&too_large);
    vae_c_tensor_init(&wrong_columns);
    if (vae_c_iic_head_allocate(&head, 3, 3) != VAE_C_OP_OK ||
        vae_c_iic_head_workspace_allocate(&head, 4, &workspace) !=
            VAE_C_OP_OK ||
        vae_c_iic_head_workspace_allocate(&head, 3, &small_workspace) !=
            VAE_C_OP_OK ||
        !make_tensor(4, 3, 1, &input) ||
        !make_tensor(1, 3, 1, &singleton) ||
        !make_tensor(4, 3, 1, &too_large) ||
        !make_tensor(4, 2, 1, &wrong_columns)) {

        goto cleanup;
    }
    set_golden_parameters(&head);
    for (index = 0; index < 12; index++) {
        input.data[index] = golden_input[index];
    }
    if (vae_c_iic_head_forward(
            NULL,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &input,
            &workspace) == VAE_C_OP_OK ||
        vae_c_iic_head_forward(
            &head,
            (VAECBatchNormMode)99,
            &input,
            &workspace) == VAE_C_OP_OK ||
        vae_c_iic_head_forward(
            &head,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            NULL,
            &workspace) == VAE_C_OP_OK ||
        vae_c_iic_head_forward(
            &head,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &singleton,
            &workspace) == VAE_C_OP_OK ||
        vae_c_iic_head_forward(
            &head,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &too_large,
            &small_workspace) == VAE_C_OP_OK ||
        vae_c_iic_head_forward(
            &head,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &wrong_columns,
            &workspace) == VAE_C_OP_OK ||
        vae_c_iic_head_backward(&fresh, &input, &workspace) ==
            VAE_C_OP_OK) {

        goto cleanup;
    }
    saved = input.data[0];
    input.data[0] = NAN;
    if (vae_c_iic_head_forward(
            &head,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &input,
            &workspace) == VAE_C_OP_OK) {

        goto cleanup;
    }
    input.data[0] = INFINITY;
    if (vae_c_iic_head_forward(
            &head,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &input,
            &workspace) == VAE_C_OP_OK) {

        goto cleanup;
    }
    input.data[0] = saved;
    saved = head.linear2.weights.values[0];
    head.linear2.weights.values[0] = NAN;
    if (vae_c_iic_head_forward(
            &head,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &input,
            &workspace) == VAE_C_OP_OK) {

        goto cleanup;
    }
    head.linear2.weights.values[0] = saved;
    saved = head.batch_norm.running_variance[0];
    head.batch_norm.running_variance[0] = INFINITY;
    if (vae_c_iic_head_forward(
            &head,
            VAE_C_BATCH_NORM_MODE_EVAL,
            &input,
            &workspace) == VAE_C_OP_OK) {

        goto cleanup;
    }
    head.batch_norm.running_variance[0] = saved;
    if (vae_c_iic_head_forward(
            &head,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &input,
            &workspace) != VAE_C_OP_OK ||
        vae_c_iic_head_workspace_zero_grad(&head, &workspace) !=
            VAE_C_OP_OK) {

        goto cleanup;
    }
    workspace.softmax_output.grad[0] = NAN;
    if (vae_c_iic_head_backward(&head, &input, &workspace) ==
            VAE_C_OP_OK) {

        goto cleanup;
    }
    workspace.softmax_output.grad[0] = INFINITY;
    if (vae_c_iic_head_backward(&head, &input, &workspace) ==
            VAE_C_OP_OK) {

        goto cleanup;
    }
    workspace.softmax_output.grad[0] = 0.0f;
    if (vae_c_iic_head_forward(
            &head,
            VAE_C_BATCH_NORM_MODE_EVAL,
            &input,
            &workspace) != VAE_C_OP_OK ||
        vae_c_iic_head_backward(&head, &input, &workspace) ==
            VAE_C_OP_OK) {

        goto cleanup;
    }
    saved_generation = head.forward_generation;
    head.forward_generation = UINT64_MAX;
    if (vae_c_iic_head_forward(
            &head,
            VAE_C_BATCH_NORM_MODE_EVAL,
            &input,
            &workspace) != VAE_C_OP_OVERFLOW) {

        goto cleanup;
    }
    head.forward_generation = saved_generation;
    saved_counter = head.num_batches_tracked;
    head.num_batches_tracked = UINT64_MAX;
    if (vae_c_iic_head_forward(
            &head,
            VAE_C_BATCH_NORM_MODE_TRAIN,
            &input,
            &workspace) != VAE_C_OP_OVERFLOW) {

        goto cleanup;
    }
    head.num_batches_tracked = saved_counter;
    passed = 1;

cleanup:
    vae_c_tensor_free(&wrong_columns);
    vae_c_tensor_free(&too_large);
    vae_c_tensor_free(&singleton);
    vae_c_tensor_free(&input);
    vae_c_iic_head_workspace_free(&small_workspace);
    vae_c_iic_head_workspace_free(&workspace);
    vae_c_iic_head_free(&fresh);
    vae_c_iic_head_free(&head);
    return passed;
}


int main(void)
{
    int failures = 0;

    failures += report_check("lifecycle_and_layout", test_lifecycle_and_layout());
    failures += report_check(
        "embedded_reference_golden_train_backward_eval",
        test_embedded_reference_golden_train_backward_eval()
    );
    failures += report_check(
        "representative_finite_difference",
        test_representative_finite_difference()
    );
    failures += report_check(
        "tail_and_output_dimensions",
        test_tail_and_output_dimensions()
    );
    failures += report_check("dual_train_contexts", test_dual_train_contexts());
    failures += report_check(
        "context_replacement_is_atomic",
        test_context_replacement_is_atomic()
    );
    failures += report_check(
        "dual_tail_and_output_dimensions",
        test_dual_tail_and_output_dimensions()
    );
    failures += report_check("invalid_inputs", test_invalid_inputs());

    printf("REFERENCE_NOTEBOOK_SHA256="
           "81894341420d94c993c916663e4ce83578f53734d7d1c35f9c0425e3b4198c8c\n");
    printf("REFERENCE_ARCHITECTURE_SHA256="
           "12fd00c3dddc6474249a86899b14797cc891e815126cfb6989eb96e489d5e758\n");
    printf("BATCH_VARIANCE=biased\n");
    printf("RUNNING_VARIANCE_UPDATE=unbiased\n");
    printf("BN_EPSILON=1e-5\n");
    printf("BN_MOMENTUM=0.1\n");

    if (failures != 0) {
        fprintf(stderr, "%d VAE C IIC HEAD TEST(S) FAILED\n", failures);
        return 1;
    }
    printf("ALL VAE C IIC HEAD TESTS PASSED\n");
    return 0;
}
