#ifndef VAE_C_MODEL_TRAIN_H
#define VAE_C_MODEL_TRAIN_H

#include "vae_c_adam.h"
#include "vae_c_model_backward.h"
#include "vae_c_model_forward.h"
#include "vae_c_rng.h"

#include <stddef.h>
#include <stdint.h>


/*
 * PROJECT: execute one non-accumulating model-level training step for a
 * batch already stored in workspace->input.  Latent epsilon is derived
 * independently for every sample from base seed, epoch, and manifest index;
 * the result therefore does not depend on batch order or composition.
 *
 * The complete preflight precedes every mutation.  After it succeeds,
 * parameter gradients are cleared, epsilon is generated, a stochastic TRAIN
 * forward is run, workspace gradients are cleared, backward is run with an
 * upstream gradient of one, and Adam v1 performs one atomic model update.
 * Loss statistics describe the pre-update forward and are published only
 * after Adam succeeds.
 *
 * There is no global rollback after preflight.  In particular, a later
 * failure can leave cleared or partial gradients, generated epsilon, forward
 * activations, BatchNorm state, caches, or a consumed backward ticket.  Adam
 * itself retains its documented all-parameter atomicity.
 *
 * This project-level orchestration is not attributed to Sakai et al.  It
 * does not load data, augment inputs, manage epochs, checkpoint state, or
 * implement an overfit loop.
 */
VAECOpStatus vae_c_model_train_step(
    VAECModel *model,
    VAECModelWorkspace *workspace,
    VAECAdam *optimizer,
    uint64_t latent_base_seed,
    uint64_t epoch,
    const uint64_t *manifest_indices,
    size_t manifest_index_count,
    VAECVAELossStats *stats
);

#endif
