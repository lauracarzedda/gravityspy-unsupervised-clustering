#include "vae_c_iic_head.h"
#include "vae_c_iic_loss.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define PROBE_BATCH 7
#define PROBE_INPUT_DIM 3
#define PROBE_OUTPUT_DIM 250
#define PROBE_ACTIVE_HIDDEN 4


static const float first_four_inputs[12] = {
    0.5f, -1.0f, 0.25f,
    -0.75f, 0.5f, 1.0f,
    1.25f, 0.25f, -0.5f,
    -0.25f, -0.75f, 0.75f
};


/* Exact over-head parameter fixture preserved from the C5B3 test. */
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
        if (hidden < PROBE_ACTIVE_HIDDEN) {
            for (feature = 0; feature < PROBE_INPUT_DIM; feature++) {
                head->linear1.weights.values[
                    hidden * PROBE_INPUT_DIM + feature
                ] = (float)((int)((3 * hidden + feature) % 7) - 3) *
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
        for (hidden = 0; hidden < PROBE_ACTIVE_HIDDEN; hidden++) {
            head->linear2.weights.values[
                output * VAE_C_IIC_HEAD_HIDDEN_DIM + hidden
            ] = (float)((int)((5 * output + 2 * hidden) % 9) - 4) *
                0.125f;
        }
    }
}


static int make_input(VAECTensor *input, int perturbed)
{
    const size_t dims[2] = {PROBE_BATCH, PROBE_INPUT_DIM};
    VAECShape shape;
    size_t index;

    if (vae_c_shape_make(2, dims, &shape) != VAE_C_STORAGE_OK ||
        vae_c_tensor_allocate(input, &shape, true) != VAE_C_STORAGE_OK) {
        return 0;
    }
    for (index = 0; index < PROBE_BATCH * PROBE_INPUT_DIM; index++) {
        float value = index < 12 ? first_four_inputs[index] :
            (float)((int)(index % 11) - 5) * 0.1875f;
        input->data[index] = perturbed ?
            -0.375f * value +
                (float)((int)(index % 5) - 2) * 0.125f :
            value;
    }
    return 1;
}


typedef struct {
    double min_sum;
    double max_sum;
    double max_deviation;
    double min_denominator;
    double max_denominator;
    size_t worst_row;
    char worst_view;
    size_t zero_probabilities;
    size_t zero_exponentials;
    size_t negative_probabilities;
    size_t nonfinite_probabilities;
    size_t nonfinite_denominators;
} ProbeStats;


static void probe_rows(
    char view,
    const VAECIICHeadWorkspace *workspace,
    ProbeStats *stats
)
{
    size_t row;

    for (row = 0; row < PROBE_BATCH; row++) {
        size_t column;
        float sum_f32 = 0.0f;
        double sum_double = 0.0;
        float max_logit = -INFINITY;
        float denominator_f32 = 0.0f;
        double deviation;

        for (column = 0; column < PROBE_OUTPUT_DIM; column++) {
            float logit = workspace->logits.data[
                row * PROBE_OUTPUT_DIM + column
            ];
            if (logit > max_logit) {
                max_logit = logit;
            }
        }
        for (column = 0; column < PROBE_OUTPUT_DIM; column++) {
            float probability = workspace->softmax_output.data[
                row * PROBE_OUTPUT_DIM + column
            ];
            float exponential = expf(workspace->logits.data[
                row * PROBE_OUTPUT_DIM + column
            ] - max_logit);

            if (!isfinite(probability)) {
                stats->nonfinite_probabilities++;
            } else {
                sum_f32 += probability;
                sum_double += (double)probability;
                if (probability < 0.0f) {
                    stats->negative_probabilities++;
                }
                if (probability == 0.0f) {
                    stats->zero_probabilities++;
                }
            }
            if (exponential == 0.0f) {
                stats->zero_exponentials++;
            }
            denominator_f32 += exponential;
        }
        if (!isfinite(denominator_f32) || denominator_f32 <= 0.0f) {
            stats->nonfinite_denominators++;
        } else {
            if ((double)denominator_f32 < stats->min_denominator) {
                stats->min_denominator = (double)denominator_f32;
            }
            if ((double)denominator_f32 > stats->max_denominator) {
                stats->max_denominator = (double)denominator_f32;
            }
        }

        deviation = fabs(sum_double - 1.0);
        if (sum_double < stats->min_sum) {
            stats->min_sum = sum_double;
        }
        if (sum_double > stats->max_sum) {
            stats->max_sum = sum_double;
        }
        if (deviation > stats->max_deviation) {
            stats->max_deviation = deviation;
            stats->worst_row = row;
            stats->worst_view = view;
        }
        printf("ROW view=%c index=%zu sum_f32=%.9g dev_f32=%.9g "
               "sum_double=%.17g dev_double=%.17g denominator_f32=%.9g\n",
               view, row, (double)sum_f32,
               fabs((double)sum_f32 - 1.0), sum_double, deviation,
               (double)denominator_f32);
    }
}


int main(int argc, char **argv)
{
    const int audit = argc == 2 && strcmp(argv[1], "--audit") == 0;
    VAECIICHead head;
    VAECIICHeadWorkspace workspace_a;
    VAECIICHeadWorkspace workspace_b;
    VAECTensor input_a;
    VAECTensor input_b;
    float original_a[PROBE_BATCH * PROBE_INPUT_DIM];
    float original_b[PROBE_BATCH * PROBE_INPUT_DIM];
    float original_probabilities_a[PROBE_BATCH * PROBE_OUTPUT_DIM];
    float original_probabilities_b[PROBE_BATCH * PROBE_OUTPUT_DIM];
    float invalid_probabilities[PROBE_BATCH * PROBE_OUTPUT_DIM];
    float *joint = NULL;
    float *marginal_a = NULL;
    float *marginal_b = NULL;
    float *gradient_a = NULL;
    float *gradient_b = NULL;
    VAECIICLossResult result;
    VAECIICLossStatus status = VAE_C_IIC_LOSS_ERROR_INVALID_FORWARD_STATE;
    ProbeStats stats;
    int invalid_cases_pass = 1;
    int passed = 0;

    if (argc > 2 || (argc == 2 && !audit)) {
        fprintf(stderr, "usage: %s [--audit]\n", argv[0]);
        return 2;
    }

    vae_c_iic_head_init(&head);
    vae_c_iic_head_workspace_init(&workspace_a);
    vae_c_iic_head_workspace_init(&workspace_b);
    vae_c_tensor_init(&input_a);
    vae_c_tensor_init(&input_b);
    memset(&stats, 0, sizeof(stats));
    stats.min_sum = INFINITY;
    stats.min_denominator = INFINITY;
    stats.max_sum = -INFINITY;
    stats.max_denominator = -INFINITY;

    joint = (float *)calloc(PROBE_OUTPUT_DIM * PROBE_OUTPUT_DIM,
                            sizeof(float));
    marginal_a = (float *)calloc(PROBE_OUTPUT_DIM, sizeof(float));
    marginal_b = (float *)calloc(PROBE_OUTPUT_DIM, sizeof(float));
    gradient_a = (float *)calloc(PROBE_BATCH * PROBE_OUTPUT_DIM,
                                sizeof(float));
    gradient_b = (float *)calloc(PROBE_BATCH * PROBE_OUTPUT_DIM,
                                sizeof(float));
    if (joint == NULL || marginal_a == NULL || marginal_b == NULL ||
        gradient_a == NULL || gradient_b == NULL ||
        vae_c_iic_head_allocate(
            &head, PROBE_INPUT_DIM, PROBE_OUTPUT_DIM
        ) != VAE_C_OP_OK ||
        vae_c_iic_head_workspace_allocate(
            &head, PROBE_BATCH, &workspace_a
        ) != VAE_C_OP_OK ||
        vae_c_iic_head_workspace_allocate(
            &head, PROBE_BATCH, &workspace_b
        ) != VAE_C_OP_OK ||
        !make_input(&input_a, 0) || !make_input(&input_b, 1)) {
        fprintf(stderr, "fixture allocation failed\n");
        goto cleanup;
    }

    set_over_parameters(&head);
    memcpy(original_a, input_a.data, sizeof(original_a));
    memcpy(original_b, input_b.data, sizeof(original_b));
    if (vae_c_iic_head_forward(
            &head, VAE_C_BATCH_NORM_MODE_TRAIN, &input_a,
            &workspace_a
        ) != VAE_C_OP_OK ||
        vae_c_iic_head_forward(
            &head, VAE_C_BATCH_NORM_MODE_TRAIN, &input_b,
            &workspace_b
        ) != VAE_C_OP_OK) {
        fprintf(stderr, "head forward failed\n");
        goto cleanup;
    }
    probe_rows('A', &workspace_a, &stats);
    probe_rows('B', &workspace_b, &stats);
    memcpy(original_probabilities_a, workspace_a.softmax_output.data,
           sizeof(original_probabilities_a));
    memcpy(original_probabilities_b, workspace_b.softmax_output.data,
           sizeof(original_probabilities_b));
    printf("GLOBAL min_row_sum=%.17g max_row_sum=%.17g "
           "max_abs_deviation=%.17g worst_view=%c worst_row=%zu "
           "min_denominator=%.9g max_denominator=%.9g "
           "zero_probabilities=%zu zero_exponentials=%zu "
           "negative_probabilities=%zu nonfinite_probabilities=%zu "
           "nonfinite_denominators=%zu\n",
           stats.min_sum, stats.max_sum, stats.max_deviation,
           stats.worst_view, stats.worst_row, stats.min_denominator,
           stats.max_denominator, stats.zero_probabilities,
           stats.zero_exponentials, stats.negative_probabilities,
           stats.nonfinite_probabilities, stats.nonfinite_denominators);

    vae_c_iic_loss_result_init(
        &result, joint, PROBE_OUTPUT_DIM * PROBE_OUTPUT_DIM,
        marginal_a, PROBE_OUTPUT_DIM, marginal_b, PROBE_OUTPUT_DIM
    );
    status = vae_c_iic_loss_forward(
        workspace_a.softmax_output.data, PROBE_BATCH, PROBE_OUTPUT_DIM,
        workspace_b.softmax_output.data, PROBE_BATCH, PROBE_OUTPUT_DIM,
        &result
    );
    printf("LOSS status=%s MI=%.9g loss=%.9g\n",
           vae_c_iic_loss_status_name(status),
           (double)result.mutual_information, (double)result.loss);

    if (status == VAE_C_IIC_LOSS_OK) {
        size_t index;
        status = vae_c_iic_loss_backward(
            workspace_a.softmax_output.data, PROBE_BATCH,
            PROBE_OUTPUT_DIM, workspace_b.softmax_output.data,
            PROBE_BATCH, PROBE_OUTPUT_DIM, &result,
            gradient_a, PROBE_BATCH * PROBE_OUTPUT_DIM,
            gradient_b, PROBE_BATCH * PROBE_OUTPUT_DIM
        );
        printf("BACKWARD status=%s\n", vae_c_iic_loss_status_name(status));
        if (status != VAE_C_IIC_LOSS_OK) {
            goto cleanup;
        }
        for (index = 0; index < PROBE_BATCH * PROBE_OUTPUT_DIM; index++) {
            if (!isfinite(gradient_a[index]) ||
                !isfinite(gradient_b[index])) {
                fprintf(stderr, "nonfinite loss gradient index=%zu\n",
                        index);
                goto cleanup;
            }
        }
    }
    memcpy(invalid_probabilities, original_probabilities_a,
           sizeof(invalid_probabilities));
    invalid_probabilities[0] += 0.01f;
    invalid_cases_pass &= vae_c_iic_loss_forward(
        invalid_probabilities, PROBE_BATCH, PROBE_OUTPUT_DIM,
        original_probabilities_b, PROBE_BATCH, PROBE_OUTPUT_DIM,
        &result
    ) == VAE_C_IIC_LOSS_ERROR_ROW_SUM;

    memcpy(invalid_probabilities, original_probabilities_a,
           sizeof(invalid_probabilities));
    invalid_probabilities[0] = -0.1f;
    invalid_cases_pass &= vae_c_iic_loss_forward(
        invalid_probabilities, PROBE_BATCH, PROBE_OUTPUT_DIM,
        original_probabilities_b, PROBE_BATCH, PROBE_OUTPUT_DIM,
        &result
    ) == VAE_C_IIC_LOSS_ERROR_NEGATIVE_PROBABILITY;

    memcpy(invalid_probabilities, original_probabilities_a,
           sizeof(invalid_probabilities));
    invalid_probabilities[0] = NAN;
    invalid_cases_pass &= vae_c_iic_loss_forward(
        invalid_probabilities, PROBE_BATCH, PROBE_OUTPUT_DIM,
        original_probabilities_b, PROBE_BATCH, PROBE_OUTPUT_DIM,
        &result
    ) == VAE_C_IIC_LOSS_ERROR_NONFINITE;

    memcpy(invalid_probabilities, original_probabilities_a,
           sizeof(invalid_probabilities));
    invalid_probabilities[0] = INFINITY;
    invalid_cases_pass &= vae_c_iic_loss_forward(
        invalid_probabilities, PROBE_BATCH, PROBE_OUTPUT_DIM,
        original_probabilities_b, PROBE_BATCH, PROBE_OUTPUT_DIM,
        &result
    ) == VAE_C_IIC_LOSS_ERROR_NONFINITE;
    printf("INVALID_CASES_PASS=%d\n", invalid_cases_pass);

    passed = stats.nonfinite_probabilities == 0 &&
        stats.negative_probabilities == 0 &&
        stats.nonfinite_denominators == 0 &&
        stats.zero_exponentials == 0 &&
        stats.zero_probabilities == 0 &&
        invalid_cases_pass &&
        memcmp(original_a, input_a.data, sizeof(original_a)) == 0 &&
        memcmp(original_b, input_b.data, sizeof(original_b)) == 0 &&
        memcmp(original_probabilities_a, workspace_a.softmax_output.data,
               sizeof(original_probabilities_a)) == 0 &&
        memcmp(original_probabilities_b, workspace_b.softmax_output.data,
               sizeof(original_probabilities_b)) == 0 &&
        head.num_batches_tracked == 2 &&
        status == (audit ? VAE_C_IIC_LOSS_ERROR_ROW_SUM :
                           VAE_C_IIC_LOSS_OK);
    printf("INTEGRATION mode=%s input_bitwise_unchanged=%d "
           "probabilities_bitwise_unchanged=%d "
           "num_batches_tracked=%llu verdict=%s\n",
           audit ? "audit" : "contract",
           memcmp(original_a, input_a.data, sizeof(original_a)) == 0 &&
               memcmp(original_b, input_b.data, sizeof(original_b)) == 0,
           memcmp(original_probabilities_a,
                  workspace_a.softmax_output.data,
                  sizeof(original_probabilities_a)) == 0 &&
               memcmp(original_probabilities_b,
                      workspace_b.softmax_output.data,
                      sizeof(original_probabilities_b)) == 0,
           (unsigned long long)head.num_batches_tracked,
           passed ? "PASS" : "FAIL");

cleanup:
    free(gradient_b);
    free(gradient_a);
    free(marginal_b);
    free(marginal_a);
    free(joint);
    vae_c_tensor_free(&input_b);
    vae_c_tensor_free(&input_a);
    vae_c_iic_head_workspace_free(&workspace_b);
    vae_c_iic_head_workspace_free(&workspace_a);
    vae_c_iic_head_free(&head);
    return passed ? 0 : 1;
}
