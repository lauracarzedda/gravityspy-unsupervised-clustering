#ifndef VAE_C_IIC_HEAD_H
#define VAE_C_IIC_HEAD_H

#include "vae_c_batchnorm.h"
#include "vae_c_dense.h"
#include "vae_c_tensor.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


#define VAE_C_IIC_HEAD_HIDDEN_DIM 512


typedef enum {
    VAE_C_IIC_HEAD_PARAM_KEY_LINEAR1_WEIGHT = 1,
    VAE_C_IIC_HEAD_PARAM_KEY_BATCH_NORM_GAMMA = 2,
    VAE_C_IIC_HEAD_PARAM_KEY_BATCH_NORM_BETA = 3,
    VAE_C_IIC_HEAD_PARAM_KEY_LINEAR2_WEIGHT = 4,
    VAE_C_IIC_HEAD_PARAM_KEY_LINEAR2_BIAS = 5
} VAECIICHeadParameterKey;


/*
 * One generic IIC classifier head:
 *
 *   Linear(input_dim, 512, bias=false)
 *   -> BatchNorm1d(512, eps=1e-5, momentum=0.1)
 *   -> ReLU
 *   -> Linear(512, output_dim, bias=true)
 *   -> row-wise Softmax
 *
 * Allocation deliberately performs no random initialization.  The reused
 * BatchNorm primitive supplies its PyTorch-compatible identity state
 * (gamma=1, beta=0, running_mean=0, running_variance=1); both dense layers
 * remain zero-filled until the caller supplies explicit parameters or a
 * later initialization module does so.
 *
 * This owning type is not copy-safe through plain assignment.  Use the
 * lifecycle API below.
 */
typedef struct {
    size_t input_dim;
    size_t output_dim;

    VAECDense linear1;
    VAECBatchNorm batch_norm;
    VAECDense linear2;

    uint64_t num_batches_tracked;
    uint64_t forward_generation;
    bool allocated;
} VAECIICHead;


/* BatchNorm's backward statistics belong to one TRAIN execution, not to the
 * persistent head.  The primitive still owns its most recent cache; a
 * workspace preserves the cache for its own one-shot backward. */
typedef struct {
    float batch_mean[VAE_C_IIC_HEAD_HIDDEN_DIM];
    float batch_inv_std[VAE_C_IIC_HEAD_HIDDEN_DIM];
    size_t last_sample_count;
    VAECShape last_input_shape;
    VAECShape last_output_shape;
    bool valid;
} VAECIICHeadBatchNormCache;


/*
 * Capacity storage is separated from the active batch views so one
 * allocation accepts ordinary batches and smaller tail batches.  The five
 * *_storage tensors own memory with gradients.  The five remaining tensors
 * are non-owning views of their first batch_size rows and expose every stage
 * needed by golden-vector validation.
 *
 * This owning type is not copy-safe through plain assignment.  A view never
 * keeps its storage alive; use the lifecycle API below.
 */
typedef struct {
    size_t capacity;
    size_t input_dim;
    size_t output_dim;
    size_t batch_size;

    VAECTensor linear1_storage;
    VAECTensor batch_norm_storage;
    VAECTensor relu_storage;
    VAECTensor logits_storage;
    VAECTensor softmax_storage;

    VAECTensor linear1_output;
    VAECTensor batch_norm_output;
    VAECTensor relu_output;
    VAECTensor logits;
    VAECTensor softmax_output;

    VAECIICHeadBatchNormCache batch_norm_cache;
    const VAECIICHead *backward_head;
    const VAECTensor *backward_input;
    uint64_t backward_generation;
    bool allocated;
} VAECIICHeadWorkspace;


void vae_c_iic_head_init(VAECIICHead *head);

void vae_c_iic_head_free(VAECIICHead *head);

bool vae_c_iic_head_is_empty(const VAECIICHead *head);

bool vae_c_iic_head_is_allocated(const VAECIICHead *head);

VAECOpStatus vae_c_iic_head_allocate(
    VAECIICHead *head,
    size_t input_dim,
    size_t output_dim
);

VAECOpStatus vae_c_iic_head_validate(const VAECIICHead *head);

/* Parameter gradients accumulate during backward and are never cleared by
 * forward. */
VAECOpStatus vae_c_iic_head_zero_grad(VAECIICHead *head);


void vae_c_iic_head_workspace_init(VAECIICHeadWorkspace *workspace);

void vae_c_iic_head_workspace_free(VAECIICHeadWorkspace *workspace);

bool vae_c_iic_head_workspace_is_empty(
    const VAECIICHeadWorkspace *workspace
);

bool vae_c_iic_head_workspace_is_allocated(
    const VAECIICHead *head,
    const VAECIICHeadWorkspace *workspace
);

VAECOpStatus vae_c_iic_head_workspace_allocate(
    const VAECIICHead *head,
    size_t capacity,
    VAECIICHeadWorkspace *workspace
);

VAECOpStatus vae_c_iic_head_workspace_validate(
    const VAECIICHead *head,
    const VAECIICHeadWorkspace *workspace
);

/* Clears all activation gradients across the complete capacity.  Data,
 * active views, and a current backward ticket are preserved. */
VAECOpStatus vae_c_iic_head_workspace_zero_grad(
    const VAECIICHead *head,
    VAECIICHeadWorkspace *workspace
);


/*
 * Forward infers B from the rank-2 input [B,input_dim], requires
 * 1 <= B <= workspace->capacity, and rejects B < 2 in TRAIN because that is
 * the reused BatchNorm1d contract.  TRAIN uses biased batch variance for
 * normalization and unbiased batch variance for the running update.
 * EVAL uses and preserves running_mean/running_variance and does not change
 * num_batches_tracked.  Input data and metadata are not modified.
 *
 * Every execution advances the head generation.  A successful TRAIN forward
 * records a backward ticket and BatchNorm cache in this workspace; a later
 * forward in another workspace does not invalidate it.  Reusing this same
 * workspace invalidates its previous ticket.  Numerical or primitive errors
 * after execution starts do not promise rollback of activation or BatchNorm
 * scratch/cache state.
 */
VAECOpStatus vae_c_iic_head_forward(
    VAECIICHead *head,
    VAECBatchNormMode mode,
    const VAECTensor *input,
    VAECIICHeadWorkspace *workspace
);

/*
 * Backward consumes this workspace's ticket from a successful TRAIN forward.
 * The caller writes upstream gradients to softmax_output.grad.
 * All other active workspace gradients must be finite zeros on entry; use
 * vae_c_iic_head_workspace_zero_grad before seeding the upstream gradient.
 * The same input tensor object, its data, and all parameter values must be
 * preserved between forward and backward.  Input and parameter gradients
 * accumulate with += and are not cleared.
 */
VAECOpStatus vae_c_iic_head_backward(
    VAECIICHead *head,
    const VAECTensor *input,
    VAECIICHeadWorkspace *workspace
);

#endif
