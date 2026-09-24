#include "vae_c_iic_ensemble.h"
#include "vae_c_rng.h"
#include "vae_c_shuffle.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>


#define SYNTHETIC_ROWS 23
#define SYNTHETIC_DIM 4
#define SYNTHETIC_BATCH 8
#define SYNTHETIC_EPOCHS 3
#define PAIR_PARAMETERS 10


static void parameter_registry(
    VAECIICPair *pair,
    VAECParameter *parameters[PAIR_PARAMETERS]
)
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


static int pair_storage_is_disjoint(VAECIICEnsemble *ensemble)
{
    size_t first;

    for (first = 0; first < VAE_C_IIC_ENSEMBLE_K; first++) {
        VAECParameter *first_parameters[PAIR_PARAMETERS];
        size_t second;

        parameter_registry(&ensemble->pairs[first], first_parameters);
        for (second = first + 1; second < VAE_C_IIC_ENSEMBLE_K;
             second++) {
            VAECParameter *second_parameters[PAIR_PARAMETERS];
            VAECIICHead *first_heads[2] = {
                &ensemble->pairs[first].main_head,
                &ensemble->pairs[first].overcluster_head
            };
            VAECIICHead *second_heads[2] = {
                &ensemble->pairs[second].main_head,
                &ensemble->pairs[second].overcluster_head
            };
            size_t left;
            size_t right;

            parameter_registry(&ensemble->pairs[second], second_parameters);
            for (left = 0; left < PAIR_PARAMETERS; left++) {
                const void *first_buffers[4] = {
                    first_parameters[left]->values,
                    first_parameters[left]->gradients,
                    first_parameters[left]->adam_m,
                    first_parameters[left]->adam_v
                };
                for (right = 0; right < PAIR_PARAMETERS; right++) {
                    const void *second_buffers[4] = {
                        second_parameters[right]->values,
                        second_parameters[right]->gradients,
                        second_parameters[right]->adam_m,
                        second_parameters[right]->adam_v
                    };
                    size_t left_buffer;
                    size_t right_buffer;

                    for (left_buffer = 0; left_buffer < 4;
                         left_buffer++) {
                        for (right_buffer = 0; right_buffer < 4;
                             right_buffer++) {
                            if (first_buffers[left_buffer] ==
                                second_buffers[right_buffer]) {
                                return 0;
                            }
                        }
                    }
                }
            }
            for (left = 0; left < 2; left++) {
                for (right = 0; right < 2; right++) {
                    if (first_heads[left]->batch_norm.running_mean ==
                            second_heads[right]->batch_norm.running_mean ||
                        first_heads[left]->batch_norm.running_mean ==
                            second_heads[right]->batch_norm.running_variance ||
                        first_heads[left]->batch_norm.running_variance ==
                            second_heads[right]->batch_norm.running_mean ||
                        first_heads[left]->batch_norm.running_variance ==
                            second_heads[right]->batch_norm.running_variance) {
                        return 0;
                    }
                }
            }
        }
    }
    return 1;
}


static int pair_initial_state_bitwise_equal(VAECIICPair *first,
                                             VAECIICPair *second)
{
    VAECParameter *first_parameters[PAIR_PARAMETERS];
    VAECParameter *second_parameters[PAIR_PARAMETERS];
    VAECIICHead *first_heads[2] = {
        &first->main_head, &first->overcluster_head
    };
    VAECIICHead *second_heads[2] = {
        &second->main_head, &second->overcluster_head
    };
    size_t index;

    parameter_registry(first, first_parameters);
    parameter_registry(second, second_parameters);
    for (index = 0; index < PAIR_PARAMETERS; index++) {
        size_t bytes = first_parameters[index]->shape.count * sizeof(float);

        if (first_parameters[index]->shape.count !=
                second_parameters[index]->shape.count ||
            memcmp(first_parameters[index]->values,
                   second_parameters[index]->values, bytes) != 0 ||
            memcmp(first_parameters[index]->gradients,
                   second_parameters[index]->gradients, bytes) != 0 ||
            memcmp(first_parameters[index]->adam_m,
                   second_parameters[index]->adam_m, bytes) != 0 ||
            memcmp(first_parameters[index]->adam_v,
                   second_parameters[index]->adam_v, bytes) != 0) {
            return 0;
        }
    }
    for (index = 0; index < 2; index++) {
        size_t bytes = VAE_C_IIC_HEAD_HIDDEN_DIM * sizeof(float);

        if (memcmp(first_heads[index]->batch_norm.running_mean,
                   second_heads[index]->batch_norm.running_mean,
                   bytes) != 0 ||
            memcmp(first_heads[index]->batch_norm.running_variance,
                   second_heads[index]->batch_norm.running_variance,
                   bytes) != 0 ||
            first_heads[index]->num_batches_tracked !=
                second_heads[index]->num_batches_tracked) {
            return 0;
        }
    }
    return 1;
}


static uint64_t digest_byte(uint64_t digest, unsigned char value)
{
    return (digest ^ (uint64_t)value) * UINT64_C(1099511628211);
}


static uint64_t digest_u64(uint64_t digest, uint64_t value)
{
    size_t byte_index;

    for (byte_index = 0; byte_index < 8; byte_index++) {
        digest = digest_byte(digest, (unsigned char)(value & UINT64_C(255)));
        value >>= 8;
    }
    return digest;
}


static uint64_t digest_float(uint64_t digest, float value)
{
    uint32_t bits;
    size_t byte_index;

    memcpy(&bits, &value, sizeof(bits));
    for (byte_index = 0; byte_index < 4; byte_index++) {
        digest = digest_byte(digest, (unsigned char)(bits & UINT32_C(255)));
        bits >>= 8;
    }
    return digest;
}


static uint64_t digest_pair(VAECIICEnsemble *ensemble, size_t pair_index,
                            int include_optimizer)
{
    uint64_t digest = UINT64_C(14695981039346656037);
    VAECIICPair *pair = &ensemble->pairs[pair_index];
    VAECIICHead *heads[2] = {
        &pair->main_head, &pair->overcluster_head
    };
    VAECParameter *parameters[PAIR_PARAMETERS];
    size_t parameter_index;
    size_t head_index;

    parameter_registry(pair, parameters);
    for (parameter_index = 0; parameter_index < PAIR_PARAMETERS;
         parameter_index++) {
        const VAECParameter *parameter = parameters[parameter_index];
        size_t index;

        digest = digest_u64(digest, (uint64_t)parameter->shape.count);
        for (index = 0; index < parameter->shape.count; index++) {
            digest = digest_float(digest, parameter->values[index]);
            if (include_optimizer) {
                digest = digest_float(digest, parameter->adam_m[index]);
                digest = digest_float(digest, parameter->adam_v[index]);
            }
        }
    }
    if (include_optimizer) {
        for (head_index = 0; head_index < 2; head_index++) {
            VAECIICHead *head = heads[head_index];
            size_t index;

            for (index = 0; index < VAE_C_IIC_HEAD_HIDDEN_DIM; index++) {
                digest = digest_float(
                    digest, head->batch_norm.running_mean[index]
                );
                digest = digest_float(
                    digest, head->batch_norm.running_variance[index]
                );
            }
            digest = digest_u64(digest, head->num_batches_tracked);
        }
        digest = digest_u64(digest, ensemble->optimizers[pair_index].step);
    }
    return digest;
}


static int test_initial_state_isolation_and_shuffle(void)
{
    const uint64_t changed_seeds[VAE_C_IIC_ENSEMBLE_K] = {
        UINT64_C(123), UINT64_C(124), UINT64_C(200),
        UINT64_C(126), UINT64_C(127)
    };
    VAECIICEnsemble first;
    VAECIICEnsemble second;
    uint64_t before[VAE_C_IIC_ENSEMBLE_K];
    size_t permutation_before[SYNTHETIC_ROWS];
    size_t permutation_repeat[SYNTHETIC_ROWS];
    size_t permutation_after[SYNTHETIC_ROWS];
    size_t permutation_next_epoch[SYNTHETIC_ROWS];
    size_t index;
    int passed = 0;

    vae_c_iic_ensemble_init(&first);
    vae_c_iic_ensemble_init(&second);
    if (vae_c_shuffle_make_permutation(
            SYNTHETIC_ROWS, VAE_C_IIC_ENSEMBLE_SHUFFLE_SEED,
            UINT64_C(0), permutation_before) != VAE_C_RNG_OK ||
        vae_c_shuffle_make_permutation(
            SYNTHETIC_ROWS, VAE_C_IIC_ENSEMBLE_SHUFFLE_SEED,
            UINT64_C(0), permutation_repeat) != VAE_C_RNG_OK ||
        vae_c_shuffle_make_permutation(
            SYNTHETIC_ROWS, VAE_C_IIC_ENSEMBLE_SHUFFLE_SEED,
            UINT64_C(1), permutation_next_epoch) != VAE_C_RNG_OK ||
        memcmp(permutation_before, permutation_repeat,
               sizeof(permutation_before)) != 0 ||
        memcmp(permutation_before, permutation_next_epoch,
               sizeof(permutation_before)) == 0 ||
        vae_c_iic_ensemble_allocate_v1(
            &first, SYNTHETIC_DIM, SYNTHETIC_BATCH) != VAE_C_OP_OK ||
        vae_c_iic_ensemble_allocate_v1(
            &second, SYNTHETIC_DIM, SYNTHETIC_BATCH) != VAE_C_OP_OK ||
        !pair_storage_is_disjoint(&first) ||
        !pair_storage_is_disjoint(&second)) {
        goto cleanup;
    }
    for (index = 0; index < VAE_C_IIC_ENSEMBLE_K; index++) {
        if (!pair_initial_state_bitwise_equal(
                &first.pairs[index], &second.pairs[index])) {
            goto cleanup;
        }
        before[index] = digest_pair(&first, index, 1);
    }

    /* Deliberately zero one pair's entire Linear1 and perturb its Adam/BN
     * state. The remaining four pairs must be byte-for-byte unaffected. */
    memset(first.pairs[2].main_head.linear1.weights.values, 0,
           first.pairs[2].main_head.linear1.weights.shape.count *
               sizeof(float));
    first.pairs[2].main_head.linear2.weights.adam_m[0] = 1.0f;
    first.pairs[2].main_head.batch_norm.running_mean[0] = 0.25f;
    if (digest_pair(&first, 2, 1) == before[2]) {
        goto cleanup;
    }
    for (index = 0; index < VAE_C_IIC_ENSEMBLE_K; index++) {
        if (index != 2 &&
            (digest_pair(&first, index, 1) != before[index] ||
             !pair_initial_state_bitwise_equal(
                 &first.pairs[index], &second.pairs[index]))) {
            goto cleanup;
        }
    }
    vae_c_iic_ensemble_free(&second);
    if (vae_c_iic_ensemble_allocate_seeded(
            &second, SYNTHETIC_DIM, SYNTHETIC_BATCH,
            changed_seeds) != VAE_C_OP_OK ||
        vae_c_shuffle_make_permutation(
            SYNTHETIC_ROWS, VAE_C_IIC_ENSEMBLE_SHUFFLE_SEED,
            UINT64_C(0), permutation_after) != VAE_C_RNG_OK ||
        memcmp(permutation_before, permutation_after,
               sizeof(permutation_before)) != 0) {
        goto cleanup;
    }
    passed = 1;

cleanup:
    vae_c_iic_ensemble_free(&second);
    vae_c_iic_ensemble_free(&first);
    return passed;
}


static int parameter_matches_stream(VAECParameter *parameter,
                                    size_t fan_in, VAECRng *rng)
{
    const double bound = 1.0 / sqrt((double)fan_in);
    size_t index;

    for (index = 0; index < parameter->shape.count; index++) {
        float unit;
        float expected;
        double actual = (double)parameter->values[index];

        if (vae_c_rng_uniform_f32(rng, &unit) != VAE_C_RNG_OK) {
            return 0;
        }
        expected = (float)((2.0 * (double)unit - 1.0) * bound);
        if (memcmp(&parameter->values[index], &expected,
                   sizeof(expected)) != 0 ||
            !isfinite(actual) || actual < -bound || actual > bound) {
            return 0;
        }
    }
    return 1;
}


static int head_bn_is_reset(const VAECIICHead *head)
{
    size_t index;

    if (head->num_batches_tracked != 0) {
        return 0;
    }
    for (index = 0; index < VAE_C_IIC_HEAD_HIDDEN_DIM; index++) {
        if (head->batch_norm.gamma.values[index] != 1.0f ||
            head->batch_norm.beta.values[index] != 0.0f ||
            head->batch_norm.running_mean[index] != 0.0f ||
            head->batch_norm.running_variance[index] != 1.0f) {
            return 0;
        }
    }
    return 1;
}


static int initial_state_matches_policy(VAECIICEnsemble *ensemble)
{
    size_t pair_index;

    if (vae_c_iic_ensemble_validate(ensemble) != VAE_C_OP_OK) {
        return 0;
    }
    for (pair_index = 0; pair_index < VAE_C_IIC_ENSEMBLE_K;
         pair_index++) {
        VAECIICPair *pair = &ensemble->pairs[pair_index];
        VAECIICHead *main_head = &pair->main_head;
        VAECIICHead *over_head = &pair->overcluster_head;
        VAECParameter *parameters[PAIR_PARAMETERS];
        VAECRng rng;
        size_t parameter_index;

        if (pair->main_output_dim != VAE_C_IIC_ENSEMBLE_MAIN_DIM ||
            pair->overcluster_output_dim != VAE_C_IIC_ENSEMBLE_OVER_DIM ||
            !head_bn_is_reset(main_head) ||
            !head_bn_is_reset(over_head) ||
            ensemble->optimizers[pair_index].step != 0 ||
            vae_c_rng_init(&rng, ensemble->seeds[pair_index]) !=
                VAE_C_RNG_OK) {
            return 0;
        }
        /* Exactly one raw-seed stream: main L1, L2 weight, L2 bias,
         * then overcluster L1, L2 weight, L2 bias. BN draws nothing. */
        if (!parameter_matches_stream(&main_head->linear1.weights,
                                      SYNTHETIC_DIM, &rng) ||
            !parameter_matches_stream(&main_head->linear2.weights,
                                      VAE_C_IIC_HEAD_HIDDEN_DIM, &rng) ||
            !parameter_matches_stream(&main_head->linear2.bias,
                                      VAE_C_IIC_HEAD_HIDDEN_DIM, &rng) ||
            !parameter_matches_stream(&over_head->linear1.weights,
                                      SYNTHETIC_DIM, &rng) ||
            !parameter_matches_stream(&over_head->linear2.weights,
                                      VAE_C_IIC_HEAD_HIDDEN_DIM, &rng) ||
            !parameter_matches_stream(&over_head->linear2.bias,
                                      VAE_C_IIC_HEAD_HIDDEN_DIM, &rng)) {
            return 0;
        }
        parameter_registry(pair, parameters);
        for (parameter_index = 0; parameter_index < PAIR_PARAMETERS;
             parameter_index++) {
            const VAECParameter *parameter = parameters[parameter_index];
            size_t index;

            for (index = 0; index < parameter->shape.count; index++) {
                if (parameter->gradients[index] != 0.0f ||
                    parameter->adam_m[index] != 0.0f ||
                    parameter->adam_v[index] != 0.0f) {
                    return 0;
                }
            }
        }
    }
    return 1;
}


static int values_are_finite(VAECIICEnsemble *ensemble)
{
    size_t pair_index;

    for (pair_index = 0; pair_index < VAE_C_IIC_ENSEMBLE_K;
         pair_index++) {
        VAECIICPair *pair = &ensemble->pairs[pair_index];
        VAECParameter *parameters[PAIR_PARAMETERS];
        size_t parameter_index;

        parameter_registry(pair, parameters);
        for (parameter_index = 0; parameter_index < PAIR_PARAMETERS;
             parameter_index++) {
            VAECParameter *parameter = parameters[parameter_index];
            size_t index;

            for (index = 0; index < parameter->shape.count; index++) {
                if (!isfinite(parameter->values[index]) ||
                    !isfinite(parameter->gradients[index]) ||
                    !isfinite(parameter->adam_m[index]) ||
                    !isfinite(parameter->adam_v[index])) {
                    return 0;
                }
            }
        }
        if (!isfinite(ensemble->workspaces[pair_index].main_result.loss) ||
            !isfinite(ensemble->workspaces[pair_index].over_result.loss) ||
            !isfinite(ensemble->workspaces[pair_index].total_loss)) {
            return 0;
        }
        for (parameter_index = 0;
             parameter_index < VAE_C_IIC_HEAD_HIDDEN_DIM;
             parameter_index++) {
            if (!isfinite(pair->main_head.batch_norm.running_mean[
                    parameter_index]) ||
                !isfinite(pair->main_head.batch_norm.running_variance[
                    parameter_index]) ||
                !isfinite(pair->overcluster_head.batch_norm.running_mean[
                    parameter_index]) ||
                !isfinite(pair->overcluster_head.batch_norm.running_variance[
                    parameter_index])) {
                return 0;
            }
        }
    }
    return 1;
}


static int make_batch(const size_t *permutation, size_t offset,
                      size_t batch_size, VAECTensor *input_a,
                      VAECTensor *input_b)
{
    const size_t dims[2] = {batch_size, SYNTHETIC_DIM};
    VAECShape shape;
    size_t row;

    if (vae_c_shape_make(2, dims, &shape) != VAE_C_STORAGE_OK ||
        vae_c_tensor_allocate(input_a, &shape, true) != VAE_C_STORAGE_OK ||
        vae_c_tensor_allocate(input_b, &shape, true) != VAE_C_STORAGE_OK) {
        return 0;
    }
    for (row = 0; row < batch_size; row++) {
        size_t feature;
        size_t event = permutation[offset + row];

        for (feature = 0; feature < SYNTHETIC_DIM; feature++) {
            size_t index = row * SYNTHETIC_DIM + feature;
            int base = (int)((event * 17 + feature * 5) % 31) - 15;
            int perturbation = (int)((event + feature) % 5) - 2;

            input_a->data[index] = (float)base * 0.0625f;
            input_b->data[index] =
                input_a->data[index] * 0.75f +
                (float)perturbation * 0.03125f;
        }
    }
    return 1;
}


static int run_smoke(const uint64_t *seeds, int poison_gradients,
                     uint64_t initial_digests[VAE_C_IIC_ENSEMBLE_K],
                     uint64_t final_digests[VAE_C_IIC_ENSEMBLE_K])
{
    VAECIICEnsemble ensemble;
    size_t pair_index;
    size_t epoch;
    size_t completed_batches = 0;
    int passed = 0;

    vae_c_iic_ensemble_init(&ensemble);
    if ((seeds == NULL ?
         vae_c_iic_ensemble_allocate_v1(
             &ensemble, SYNTHETIC_DIM, SYNTHETIC_BATCH) :
         vae_c_iic_ensemble_allocate_seeded(
             &ensemble, SYNTHETIC_DIM, SYNTHETIC_BATCH, seeds)) !=
            VAE_C_OP_OK ||
        !initial_state_matches_policy(&ensemble)) {
        goto cleanup;
    }
    for (pair_index = 0; pair_index < VAE_C_IIC_ENSEMBLE_K;
         pair_index++) {
        uint64_t expected_seed = seeds == NULL ?
            VAE_C_IIC_ENSEMBLE_FIRST_SEED + (uint64_t)pair_index :
            seeds[pair_index];

        if (ensemble.seeds[pair_index] != expected_seed) {
            goto cleanup;
        }
    }
    for (pair_index = 0; pair_index < VAE_C_IIC_ENSEMBLE_K;
         pair_index++) {
        initial_digests[pair_index] =
            digest_pair(&ensemble, pair_index, 0);
    }

    for (epoch = 0; epoch < SYNTHETIC_EPOCHS; epoch++) {
        size_t permutation[SYNTHETIC_ROWS];
        size_t offset;

        if (vae_c_shuffle_make_permutation(
                SYNTHETIC_ROWS, VAE_C_IIC_ENSEMBLE_SHUFFLE_SEED,
                (uint64_t)epoch, permutation) != VAE_C_RNG_OK) {
            goto cleanup;
        }
        for (offset = 0; offset < SYNTHETIC_ROWS;
             offset += SYNTHETIC_BATCH) {
            const size_t remaining = SYNTHETIC_ROWS - offset;
            const size_t batch_size = remaining < SYNTHETIC_BATCH ?
                remaining : SYNTHETIC_BATCH;
            VAECTensor input_a;
            VAECTensor input_b;
            float original_a[SYNTHETIC_BATCH * SYNTHETIC_DIM];
            float original_b[SYNTHETIC_BATCH * SYNTHETIC_DIM];
            size_t index;
            int batch_ok = 1;

            vae_c_tensor_init(&input_a);
            vae_c_tensor_init(&input_b);
            if (!make_batch(permutation, offset, batch_size,
                            &input_a, &input_b)) {
                vae_c_tensor_free(&input_b);
                vae_c_tensor_free(&input_a);
                goto cleanup;
            }
            memcpy(original_a, input_a.data,
                   batch_size * SYNTHETIC_DIM * sizeof(float));
            memcpy(original_b, input_b.data,
                   batch_size * SYNTHETIC_DIM * sizeof(float));
            if (poison_gradients) {
                for (pair_index = 0;
                     pair_index < VAE_C_IIC_ENSEMBLE_K;
                     pair_index++) {
                    ensemble.pairs[pair_index].main_head.linear1.weights
                        .gradients[0] = 37.0f;
                    ensemble.pairs[pair_index].overcluster_head.linear2.bias
                        .gradients[0] = -19.0f;
                }
            }
            if (vae_c_iic_ensemble_train_step(
                    &ensemble, &input_a, &input_b) != VAE_C_OP_OK ||
                memcmp(original_a, input_a.data,
                       batch_size * SYNTHETIC_DIM * sizeof(float)) != 0 ||
                memcmp(original_b, input_b.data,
                       batch_size * SYNTHETIC_DIM * sizeof(float)) != 0) {
                batch_ok = 0;
            }
            completed_batches++;
            for (pair_index = 0;
                 batch_ok && pair_index < VAE_C_IIC_ENSEMBLE_K;
                 pair_index++) {
                VAECIICPair *pair = &ensemble.pairs[pair_index];

                if (ensemble.optimizers[pair_index].step !=
                        (uint64_t)completed_batches ||
                    pair->main_head.num_batches_tracked !=
                        (uint64_t)(2 * completed_batches) ||
                    pair->overcluster_head.num_batches_tracked !=
                        (uint64_t)(2 * completed_batches)) {
                    batch_ok = 0;
                }
            }
            for (index = 0; batch_ok && index < input_a.shape.count;
                 index++) {
                if (!isfinite(input_a.grad[index]) ||
                    !isfinite(input_b.grad[index])) {
                    batch_ok = 0;
                }
            }
            vae_c_tensor_free(&input_b);
            vae_c_tensor_free(&input_a);
            if (!batch_ok || !values_are_finite(&ensemble)) {
                goto cleanup;
            }
        }
    }
    for (pair_index = 0; pair_index < VAE_C_IIC_ENSEMBLE_K;
         pair_index++) {
        final_digests[pair_index] =
            digest_pair(&ensemble, pair_index, 1);
        if (digest_pair(&ensemble, pair_index, 0) ==
                initial_digests[pair_index]) {
            goto cleanup;
        }
    }
    passed = completed_batches == SYNTHETIC_EPOCHS * 3;

cleanup:
    vae_c_iic_ensemble_free(&ensemble);
    return passed;
}


int main(void)
{
    const uint64_t changed_seeds[VAE_C_IIC_ENSEMBLE_K] = {
        UINT64_C(123), UINT64_C(124), UINT64_C(200),
        UINT64_C(126), UINT64_C(127)
    };
    uint64_t initial[VAE_C_IIC_ENSEMBLE_K];
    uint64_t final[VAE_C_IIC_ENSEMBLE_K];
    uint64_t replica_initial[VAE_C_IIC_ENSEMBLE_K];
    uint64_t replica_final[VAE_C_IIC_ENSEMBLE_K];
    uint64_t changed_initial[VAE_C_IIC_ENSEMBLE_K];
    uint64_t changed_final[VAE_C_IIC_ENSEMBLE_K];
    size_t index;

    if (!test_initial_state_isolation_and_shuffle() ||
        !run_smoke(NULL, 0, initial, final) ||
        !run_smoke(NULL, 1, replica_initial, replica_final) ||
        !run_smoke(changed_seeds, 0, changed_initial, changed_final) ||
        memcmp(initial, replica_initial, sizeof(initial)) != 0 ||
        memcmp(final, replica_final, sizeof(final)) != 0 ||
        changed_initial[2] == initial[2] ||
        changed_final[2] == final[2]) {
        fprintf(stderr, "IIC ensemble initialization/smoke failed\n");
        return 1;
    }
    for (index = 0; index < VAE_C_IIC_ENSEMBLE_K; index++) {
        if (index != 2 &&
            (initial[index] != changed_initial[index] ||
             final[index] != changed_final[index])) {
            fprintf(stderr, "IIC ensemble pair isolation failed\n");
            return 1;
        }
        printf("PAIR_%zu_INITIAL_DIGEST=%016llx\n", index,
               (unsigned long long)initial[index]);
        printf("PAIR_%zu_FINAL_DIGEST=%016llx\n", index,
               (unsigned long long)final[index]);
    }
    printf("IIC_ENSEMBLE_SEEDS=123,124,125,126,127\n");
    printf("IIC_ENSEMBLE_K=5\n");
    printf("IIC_ENSEMBLE_EPOCHS=3\n");
    printf("IIC_ENSEMBLE_BATCHES_PER_EPOCH=3\n");
    printf("IIC_ENSEMBLE_TAIL_BATCH=7\n");
    printf("IIC_ENSEMBLE_STEPS_PER_PAIR=9\n");
    printf("IIC_ENSEMBLE_BN_BATCHES_PER_HEAD=18\n");
    printf("IIC_ENSEMBLE_INITIAL_STATE_BITWISE_REPEAT=true\n");
    printf("IIC_ENSEMBLE_PAIR_STORAGE_DISJOINT=true\n");
    printf("IIC_ENSEMBLE_PAIR2_ZERO_PRESERVES_OTHERS=true\n");
    printf("IIC_ENSEMBLE_SHUFFLE_K_INDEPENDENT=true\n");
    printf("IIC_ENSEMBLE_SMOKE_PASS=true\n");
    return 0;
}
