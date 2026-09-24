#ifndef VAE_C_IIC_MINI75_CENTRAL_H
#define VAE_C_IIC_MINI75_CENTRAL_H

#include <stddef.h>
#include <stdint.h>

#define VAE_C_IIC_MINI75_ROWS 8583u
#define VAE_C_IIC_MINI75_LATENT_DIM 32u
#define VAE_C_IIC_MINI75_EVENT_ID_CAP 128u

typedef enum {
    VAE_C_IIC_MINI75_SOURCE_TRAIN = 1,
    VAE_C_IIC_MINI75_SOURCE_VALIDATION = 2
} VAECIICMini75Source;

typedef struct {
    uint32_t manifest_index;
    uint32_t event_index;
    uint32_t split_pos;
    VAECIICMini75Source source;
    char event_id[VAE_C_IIC_MINI75_EVENT_ID_CAP];
} VAECIICMini75CentralRow;

/* Values and rows are indexed by manifest_index, not by native split_pos. */
typedef struct {
    float *mu;
    VAECIICMini75CentralRow *rows;
    size_t row_count;
    size_t latent_dim;
    char checkpoint_sha256[65];
    char train_matrix_sha256[65];
    char validation_matrix_sha256[65];
    char train_rows_sha256[65];
    char validation_rows_sha256[65];
} VAECIICMini75Central;

void vae_c_iic_mini75_central_init(VAECIICMini75Central *view);
void vae_c_iic_mini75_central_free(VAECIICMini75Central *view);

/*
 * Read-only assembly of the frozen central posterior-mu artifact. The optional
 * artifact_directory_override exists only for negative tests; NULL selects
 * the canonical artifact under repository_root. All six artifact files,
 * independently defined canonical inputs, row identities, and float values
 * are validated before publishing the in-memory view. Generic manifest
 * parsers may retain labels as metadata; this module never accesses them.
 * No file is created and no neural forward is called.
 */
int vae_c_iic_mini75_central_load(
    const char *repository_root,
    const char *artifact_directory_override,
    VAECIICMini75Central *view,
    char *error_message,
    size_t error_capacity
);

const float *vae_c_iic_mini75_central_row_mu(
    const VAECIICMini75Central *view,
    size_t manifest_index
);

#endif
