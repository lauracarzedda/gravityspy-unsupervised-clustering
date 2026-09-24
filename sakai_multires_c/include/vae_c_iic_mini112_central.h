#ifndef VAE_C_IIC_MINI112_CENTRAL_H
#define VAE_C_IIC_MINI112_CENTRAL_H

#include <stddef.h>
#include <stdint.h>

#define VAE_C_IIC_MINI112_ROWS 8583u
#define VAE_C_IIC_MINI112_LATENT_DIM 64u
#define VAE_C_IIC_MINI112_EVENT_ID_CAP 128u

typedef enum {
    VAE_C_IIC_MINI112_SOURCE_TRAIN = 1,
    VAE_C_IIC_MINI112_SOURCE_VALIDATION = 2
} VAECIICMini112Source;

typedef struct {
    uint32_t manifest_index;
    uint32_t event_index;
    uint32_t split_pos;
    VAECIICMini112Source source;
    char event_id[VAE_C_IIC_MINI112_EVENT_ID_CAP];
} VAECIICMini112CentralRow;

/* Values and rows are indexed by manifest_index, not by native split_pos. */
typedef struct {
    float *mu;
    VAECIICMini112CentralRow *rows;
    size_t row_count;
    size_t latent_dim;
    char checkpoint_sha256[65];
    char train_matrix_sha256[65];
    char validation_matrix_sha256[65];
    char train_rows_sha256[65];
    char validation_rows_sha256[65];
} VAECIICMini112Central;

void vae_c_iic_mini112_central_init(VAECIICMini112Central *view);
void vae_c_iic_mini112_central_free(VAECIICMini112Central *view);

/*
 * Read-only assembly of the frozen central posterior-mu artifact. The optional
 * artifact_directory_override exists only for negative tests; NULL selects
 * the canonical artifact under repository_root. All six artifact files,
 * independently defined canonical inputs, row identities, and float values
 * are validated before publishing the in-memory view. The label-free
 * manifest parser does not inspect or copy label values.
 * No file is created and no neural forward is called.
 */
int vae_c_iic_mini112_central_load(
    const char *repository_root,
    const char *artifact_directory_override,
    VAECIICMini112Central *view,
    char *error_message,
    size_t error_capacity
);

const float *vae_c_iic_mini112_central_row_mu(
    const VAECIICMini112Central *view,
    size_t manifest_index
);

#endif
