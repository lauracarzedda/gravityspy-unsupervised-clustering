#ifndef VAE_C_IIC_PAIR_H
#define VAE_C_IIC_PAIR_H

#include "vae_c_iic_head.h"
#include "vae_c_iic_loss.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


typedef enum {
    VAE_C_IIC_PAIR_BACKWARD_A_THEN_B = 1,
    VAE_C_IIC_PAIR_BACKWARD_B_THEN_A = 2
} VAECIICPairBackwardOrder;


/* The heads own independent parameters and BatchNorm state.  This owning
 * object, and its workspace below, must not be copied by assignment. */
typedef struct {
    size_t input_dim;
    size_t main_output_dim;
    size_t overcluster_output_dim;
    VAECIICHead main_head;
    VAECIICHead overcluster_head;
    bool allocated;
} VAECIICPair;


typedef struct {
    size_t capacity;
    size_t batch_size;
    VAECIICHeadWorkspace main_a;
    VAECIICHeadWorkspace over_a;
    VAECIICHeadWorkspace main_b;
    VAECIICHeadWorkspace over_b;

    float *main_joint;
    float *main_marginal_i;
    float *main_marginal_j;
    float *over_joint;
    float *over_marginal_i;
    float *over_marginal_j;
    float *main_a_loss_grad;
    float *main_b_loss_grad;
    float *over_a_loss_grad;
    float *over_b_loss_grad;
    VAECIICLossResult main_result;
    VAECIICLossResult over_result;

    float total_loss;
    const VAECIICPair *backward_pair;
    const VAECTensor *backward_input_a;
    const VAECTensor *backward_input_b;
    uint64_t main_a_generation;
    uint64_t main_b_generation;
    uint64_t over_a_generation;
    uint64_t over_b_generation;
    bool allocated;
} VAECIICPairWorkspace;


void vae_c_iic_pair_init(VAECIICPair *pair);
void vae_c_iic_pair_free(VAECIICPair *pair);
bool vae_c_iic_pair_is_empty(const VAECIICPair *pair);
VAECOpStatus vae_c_iic_pair_allocate(
    VAECIICPair *pair,
    size_t input_dim,
    size_t main_output_dim,
    size_t overcluster_output_dim
);
VAECOpStatus vae_c_iic_pair_validate(const VAECIICPair *pair);
VAECOpStatus vae_c_iic_pair_zero_grad(VAECIICPair *pair);

void vae_c_iic_pair_workspace_init(VAECIICPairWorkspace *workspace);
void vae_c_iic_pair_workspace_free(VAECIICPairWorkspace *workspace);
bool vae_c_iic_pair_workspace_is_empty(
    const VAECIICPairWorkspace *workspace
);
VAECOpStatus vae_c_iic_pair_workspace_allocate(
    const VAECIICPair *pair,
    size_t capacity,
    VAECIICPairWorkspace *workspace
);
VAECOpStatus vae_c_iic_pair_workspace_validate(
    const VAECIICPair *pair,
    const VAECIICPairWorkspace *workspace
);
/* Clears activation gradients but preserves an unconsumed backward context. */
VAECOpStatus vae_c_iic_pair_workspace_zero_grad(
    const VAECIICPair *pair,
    VAECIICPairWorkspace *workspace
);

/* Preflights both inputs and all capacities before the first TRAIN forward.
 * Successful execution order matches Notebook06: main A, over A, main B,
 * over B.  Each view has its own workspace; the two heads share no state.
 * Result loss is main_result.loss + over_result.loss, both with weight 1.
 * A primitive/numerical failure after execution begins cannot roll back
 * activation or BatchNorm state. */
VAECOpStatus vae_c_iic_pair_forward_train(
    VAECIICPair *pair,
    const VAECTensor *input_a,
    const VAECTensor *input_b,
    VAECIICPairWorkspace *workspace
);

/* The four TRAIN contexts are one-shot.  All four tickets are preflighted,
 * then both C5B1 backward calls run into private scratch: an invalid loss
 * result cannot partially change activation, parameter, or input gradients.
 * Head parameter and input gradients accumulate; callers clear them at step
 * boundaries.
 * Either A/B backward order is supported independently within each head. */
VAECOpStatus vae_c_iic_pair_backward(
    VAECIICPair *pair,
    const VAECTensor *input_a,
    const VAECTensor *input_b,
    VAECIICPairWorkspace *workspace,
    VAECIICPairBackwardOrder order
);

#endif
