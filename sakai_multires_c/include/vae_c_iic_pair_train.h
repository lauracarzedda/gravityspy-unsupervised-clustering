#ifndef VAE_C_IIC_PAIR_TRAIN_H
#define VAE_C_IIC_PAIR_TRAIN_H

#include "vae_c_adam.h"
#include "vae_c_iic_pair.h"

/* One classifier pair, one Adam optimizer and one update per call.
 * The optimizer registry comprises the five trainable parameters of each
 * head, in main-then-overcluster order. BatchNorm running statistics are
 * never optimizer parameters. Input gradients are not cleared: as in the
 * underlying pair API, only trainable parameter gradients are zeroed at
 * the step boundary. No initialization, data loading or checkpointing is
 * performed here.
 *
 * Preflight errors leave parameter values, moments, optimizer step and BN
 * state untouched. A numerical failure after a TRAIN forward begins cannot
 * roll back activation or BN state; Adam itself validates all candidates
 * before changing any parameter or moment.
 */
VAECOpStatus vae_c_iic_pair_train_step(
    VAECIICPair *pair,
    VAECIICPairWorkspace *workspace,
    VAECAdam *optimizer,
    const VAECTensor *input_a,
    const VAECTensor *input_b
);

#endif
