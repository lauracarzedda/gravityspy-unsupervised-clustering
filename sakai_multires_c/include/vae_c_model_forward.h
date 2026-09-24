#ifndef VAE_C_MODEL_FORWARD_H
#define VAE_C_MODEL_FORWARD_H

#include "vae_c_latent.h"
#include "vae_c_model_workspace.h"
#include "vae_c_objective.h"


/*
 * PAPER: the general encoder order is inspired by Figure 8: convolution,
 * BatchNorm and ReLU blocks, MaxPool after the first block, final average
 * pooling, and a fully-connected distribution head.
 *
 * PROJECT: logvar, the concrete BatchNorm2d operations and their TRAIN/EVAL
 * semantics, the packed output layout, validation, exact-alias rules, and
 * the software guarantees below are local project decisions.
 *
 * Preconditions:
 * - model and workspace are non-NULL and batch_norm_mode is TRAIN or EVAL;
 * - vae_c_model_validate(model) returns VAE_C_OP_OK, and model is allocated,
 *   initialized, and numerically valid in its complete state.  Finite,
 *   nonzero parameter gradients and Adam m/v are allowed; they need not be
 *   zero;
 * - workspace is allocated, valid, and compatible with the model profile,
 *   configuration, depth, and batch size;
 * - input.data is finite and in [0, 1], every BatchNorm sample count is
 *   valid for the requested mode, no forbidden exact alias is present, and
 *   model->vae_cache_generation is smaller than UINT64_MAX.
 *
 * The function writes the encoder activations, packed encoder-FC output,
 * mu, and logvar.  For every sample the packed layout is
 * [mu_0, ..., mu_(L-1), logvar_0, ..., logvar_(L-1)].
 *
 * TRAIN uses batch statistics, updates running mean/variance and training
 * caches of encoder BatchNorm layers, and updates the MaxPool cache.  EVAL
 * uses running statistics without changing them, manages or invalidates
 * BatchNorm training caches according to the existing primitive contract,
 * and updates the MaxPool cache.
 *
 * A successful call does not modify input.data, any workspace.grad,
 * parameter values, parameter gradients, Adam m/v, epsilon, z, decoder
 * activations, logits, decoder BatchNorm state, the parameter registry,
 * configuration, or initialization seed.  After the complete preflight and
 * before the first neural primitive, the function invalidates this
 * workspace's backward ticket and increments vae_cache_generation exactly
 * once.  This also makes tickets held by other workspaces stale.
 *
 * The read-only preflight gives a global strong guarantee: every error found
 * before the first neural primitive leaves model and workspace storage
 * unchanged.  After that primitive starts there is no global rollback;
 * activations, caches, or running statistics may already have changed.  The
 * first failing primitive's status is propagated and later primitives are
 * skipped.
 *
 * Only exact pointer aliases can be rejected portably.  Partial overlap
 * between distinct C allocations is not detected and remains outside this
 * API's contract.
 */
VAECOpStatus vae_c_model_encoder_forward(
    VAECModel *model,
    VAECBatchNormMode batch_norm_mode,
    VAECModelWorkspace *workspace
);

/*
 * PAPER: the general decoder order is inspired by Figure 8: an initial
 * fully-connected projection, nearest-neighbour upsampling, hidden
 * convolution blocks, and reconstruction at the input resolution.
 *
 * PROJECT: BatchNorm1d after the decoder fully-connected layer,
 * BatchNorm2d after hidden decoder convolutions, their TRAIN/EVAL
 * semantics, depth-three/depth-four wiring, and a final linear logits
 * convolution are local project decisions.  No final BatchNorm, ReLU,
 * sigmoid, clamp, or pixel-domain normalization is applied here.
 *
 * Input is workspace->z and output is workspace->logits.  This function
 * does not determine how z was obtained.  Every z value must be finite,
 * but z is not restricted to [0, 1].
 *
 * Preconditions:
 * - model and workspace are non-NULL and batch_norm_mode is TRAIN or EVAL;
 * - the complete model is allocated, initialized, numerically valid, and
 *   accepted by vae_c_model_validate;
 * - the complete workspace is allocated, valid, and compatible with the
 *   model, including the intentional, non-owning decoder reshape view;
 * - decoder sample counts are valid for the requested BatchNorm mode and
 *   no forbidden exact model/workspace alias is present;
 * - model->vae_cache_generation is smaller than UINT64_MAX.
 *
 * TRAIN updates running statistics, work buffers, and training caches of
 * decoder BatchNorm layers.  EVAL preserves their running statistics,
 * updates work_inv_std, and invalidates training caches according to the
 * existing BatchNorm primitive contract.
 *
 * A successful call may overwrite decoder activation data and logits, and
 * may mutate only the decoder BatchNorm state allowed by the selected
 * mode.  It does not modify z data or metadata, workspace gradients,
 * encoder activations or state, parameter values or optimizer storage,
 * configuration, registry, or seed.  After the complete preflight and before
 * the first neural primitive, it invalidates this workspace's backward
 * ticket and increments vae_cache_generation exactly once.  The reshape
 * view remains an exact alias of decoder_fc_relu_output without copying data.
 *
 * The read-only preflight gives a global strong guarantee: an error found
 * before the first neural primitive leaves model and workspace storage
 * unchanged.  Once execution starts there is no global rollback;
 * activations or decoder BatchNorm state may already have changed.  The
 * first failing primitive's status is propagated and later primitives are
 * skipped.  A non-finite final logit is reported as invalid state after
 * the final convolution, with the same no-rollback rule.
 *
 * Only exact pointer aliases can be rejected portably.  Partial overlap
 * between distinct C allocations is not detected and remains outside this
 * API's contract.
 */
VAECOpStatus vae_c_model_decoder_forward(
    VAECModel *model,
    VAECBatchNormMode batch_norm_mode,
    VAECModelWorkspace *workspace
);

/*
 * PAPER: the probabilistic VAE path uses a Gaussian posterior,
 * reparameterization, a Bernoulli likelihood, and the corresponding
 * reconstruction-plus-KL training objective.
 *
 * PROJECT: this composite forward accepts exactly three operational
 * combinations: STOCHASTIC+TRAIN for training, STOCHASTIC+EVAL for
 * fixed-epsilon validation, and MEAN+EVAL for deterministic embedding or
 * reconstruction.  MEAN+TRAIN is rejected.  BatchNorm mode and latent
 * mode remain separate choices and are not attributed to the paper.
 *
 * The caller owns epsilon generation and must fill workspace->epsilon
 * before a stochastic call.  The composite never invokes an RNG.  In
 * MEAN mode epsilon is neither read nor numerically validated and NULL is
 * passed to the reparameterization primitive, so z is a copy of mu.
 *
 * On success the exact chain is encoder, reparameterization, decoder, and
 * vae_c_vae_loss_forward with workspace->input as the reconstruction
 * target.  The fixed v1 beta_KL is therefore inherited from the objective
 * module.  No sigmoid, clamp, warm-up, backward, optimizer, loader, or
 * label operation is performed.
 *
 * The complete read-only preflight runs before the encoder.  Errors found
 * there leave model, workspace, and stats unchanged.  It also reserves the
 * two generation increments performed by encoder and decoder, so values
 * above UINT64_MAX-2 fail before the encoder.  Once the encoder
 * starts there is no global rollback: activations, caches, or BatchNorm
 * state may already have changed when a later operation fails.  The first
 * failing status is propagated and later operations are skipped.
 *
 * A successful call may update encoder/decoder activation data, mu,
 * logvar, z, logits, MaxPool cache, BatchNorm state permitted by the
 * selected mode, and stats.  It preserves input, epsilon, all workspace
 * gradients, parameter values and gradients, Adam state, registry,
 * configuration, and seeds.  Encoder and decoder advance the model cache
 * generation once each.  Only a fully successful STOCHASTIC+TRAIN call
 * records the resulting model-bound backward ticket; EVAL, MEAN, and every
 * post-start failure leave no current ticket.  EVAL also preserves running
 * mean/variance.
 *
 * Only exact pointer aliases can be rejected portably.  Partial overlap
 * between distinct C allocations remains outside this API's contract.
 */
VAECOpStatus vae_c_model_vae_forward(
    VAECModel *model,
    VAECBatchNormMode batch_norm_mode,
    VAECLatentMode latent_mode,
    VAECModelWorkspace *workspace,
    VAECVAELossStats *stats
);

#endif
