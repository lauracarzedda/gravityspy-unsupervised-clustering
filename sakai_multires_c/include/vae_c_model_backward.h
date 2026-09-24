#ifndef VAE_C_MODEL_BACKWARD_H
#define VAE_C_MODEL_BACKWARD_H

#include "vae_c_model_workspace.h"


/*
 * PAPER: VAE training differentiates the negative ELBO through the
 * Gaussian reparameterization path.
 *
 * PROJECT: this composite is the C11/direct-backend reverse pass for a
 * previously completed STOCHASTIC+TRAIN vae_c_model_vae_forward call.
 * The caller supplies the scalar upstream derivative and owns all
 * zero-gradient policy.  Workspace gradients must be exactly zero on
 * entry, while finite parameter gradients may already be nonzero so that
 * accumulation windows remain possible.  No gradient is propagated
 * through workspace->input's role as the BCE target.
 *
 * A call is backward-ready only when the workspace ticket is nonzero,
 * bound to model, and equal to model->vae_cache_generation.  The complete
 * read-only preflight validates this ticket, model/workspace state,
 * finite values, zero workspace gradients, TRAIN BatchNorm and MaxPool
 * caches, shapes, capacities, overflow-sensitive counts, and forbidden
 * exact aliases.  A preflight error preserves the ticket and all storage.
 *
 * Once preflight succeeds, the ticket is consumed before the first loss
 * backward operation.  Success and every later error leave it consumed;
 * no global rollback is promised after execution starts.  Backward
 * primitives accumulate with += and never clear gradients implicitly.
 *
 * Success may mutate workspace gradients, parameter gradients, and
 * BatchNorm work buffers.  It preserves all tensor data, parameter values,
 * Adam state, epsilon, BatchNorm running statistics and cache metadata,
 * MaxPool cache metadata, configuration, registry, seed, and model cache
 * generation.  Only exact pointer aliases are rejected portably; partial
 * overlap between distinct allocations remains outside the contract.
 */
VAECOpStatus vae_c_model_vae_backward(
    VAECModel *model,
    VAECModelWorkspace *workspace,
    float upstream_gradient
);

#endif
