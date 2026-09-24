#ifndef VAE_C_IIC_MINI75_PERTURBED_H
#define VAE_C_IIC_MINI75_PERTURBED_H

#include "vae_c_mu_contract.h"
#include "vae_c_model_forward.h"
#include "vae_multires_event_loader.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define VAE_C_IIC_MINI75_ROWS 8583u
#define VAE_C_IIC_MINI75_LATENT_DIM 32u
#define VAE_C_IIC_MINI75_LOGICAL_BATCH 128u
#define VAE_C_IIC_MINI75_ENCODER_MAX_BATCH 64u
#define VAE_C_IIC_MINI75_AUGMENTATION_SEED UINT64_C(789)

/* This object is owning and must not be copied by assignment. Opening it
 * validates the canonical BEST checkpoint but does not run any forward. */
typedef struct {
    VAECMuCanonicalContract contract;
    GSEventManifest events;
    GSMultiresEventLoader loader;
    VAECModel model;
    VAECAdam optimizer;
    VAECCheckpointMetadata metadata;
    char repository_root[VAE_C_MU_PATH_CAP];
    bool ready;
} VAECIICMini75Perturbed;

typedef struct {
    uint64_t encoder_forward_calls;
    size_t samples;
    bool inputs_bitwise_unchanged;
    bool model_parameters_bn_bitwise_unchanged;
    bool decoder_workspace_untouched;
    bool optimizer_unchanged;
} VAECIICMini75PerturbedReport;

void vae_c_iic_mini75_perturbed_init(VAECIICMini75Perturbed *source);
void vae_c_iic_mini75_perturbed_free(VAECIICMini75Perturbed *source);

bool vae_c_iic_mini75_perturbed_open(
    VAECIICMini75Perturbed *source,
    const char *repository_root
);

/* Computes posterior mu for ordered manifest indices. The logical batch is
 * at most 128, while each frozen VAE encoder forward is at most 64. Output
 * is committed only after all rows pass. No decoder, latent sampling,
 * backward or optimizer operation is called. */
bool vae_c_iic_mini75_perturbed_compute(
    VAECIICMini75Perturbed *source,
    const size_t *manifest_indices,
    size_t count,
    uint64_t epoch,
    size_t encoder_microbatch,
    float *mu_out,
    int *crop_left_out,
    VAECIICMini75PerturbedReport *report
);

#endif
