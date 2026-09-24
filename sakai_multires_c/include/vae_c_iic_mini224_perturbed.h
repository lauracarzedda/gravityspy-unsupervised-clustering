#ifndef VAE_C_IIC_MINI224_PERTURBED_H
#define VAE_C_IIC_MINI224_PERTURBED_H

#include "vae_c_iic_mini224_central.h"
#include "vae_c_mu_contract.h"
#include "vae_c_model_forward.h"
#include "vae_multires_event_loader.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define VAE_C_IIC_MINI224_ENCODER_MAX_BATCH 16u
#define VAE_C_IIC_MINI224_AUGMENTATION_SEED UINT64_C(789)

/* Owning, non-copyable source. Opening validates canonical rows and BEST;
 * it performs no neural forward. No central mu is used as perturbed input. */
typedef struct {
    VAECMuCanonicalContract contract;
    VAECIICMini224Central central;
    GSEventManifest events;
    GSMultiresEventLoader loader;
    VAECModel model;
    VAECAdam optimizer;
    VAECCheckpointMetadata metadata;
    uint32_t canonical_manifest_indices[VAE_C_IIC_MINI224_ROWS];
    char repository_root[VAE_C_MU_PATH_CAP];
    bool ready;
} VAECIICMini224Perturbed;

typedef struct {
    VAECIICMini224CentralRow identity;
    uint64_t epoch;
    int crop_left;
} VAECIICMini224PerturbedRow;

typedef struct {
    uint64_t encoder_forward_calls;
    size_t samples;
    bool inputs_bitwise_unchanged;
    bool model_parameters_bn_bitwise_unchanged;
    bool decoder_workspace_untouched;
    bool optimizer_unchanged;
} VAECIICMini224PerturbedReport;

void vae_c_iic_mini224_perturbed_init(VAECIICMini224Perturbed *source);
void vae_c_iic_mini224_perturbed_free(VAECIICMini224Perturbed *source);

bool vae_c_iic_mini224_perturbed_open(
    VAECIICMini224Perturbed *source, const char *repository_root
);

/* Processes an explicit IIC row sequence in caller order. encoder_microbatch
 * applies only to the frozen encoder and must be in [1, 16]; it does not set
 * the logical IIC batch size. Outputs are committed only after every row and
 * the frozen-state check succeed. */
bool vae_c_iic_mini224_perturbed_compute(
    VAECIICMini224Perturbed *source,
    const size_t *iic_row_indices,
    size_t count,
    uint64_t epoch,
    size_t encoder_microbatch,
    float *mu_out,
    VAECIICMini224PerturbedRow *rows_out,
    VAECIICMini224PerturbedReport *report
);

#endif
