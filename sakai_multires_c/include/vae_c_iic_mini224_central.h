#ifndef VAE_C_IIC_MINI224_CENTRAL_H
#define VAE_C_IIC_MINI224_CENTRAL_H

#include <stddef.h>
#include <stdint.h>

#define VAE_C_IIC_MINI224_ROWS 1000u
#define VAE_C_IIC_MINI224_TRAIN_ROWS 800u
#define VAE_C_IIC_MINI224_VALIDATION_ROWS 200u
#define VAE_C_IIC_MINI224_MANIFEST_ROWS 8583u
#define VAE_C_IIC_MINI224_LATENT_DIM 64u
#define VAE_C_IIC_MINI224_EVENT_ID_CAP 128u

typedef enum {
    VAE_C_IIC_MINI224_SOURCE_TRAIN = 1,
    VAE_C_IIC_MINI224_SOURCE_VALIDATION = 2
} VAECIICMini224Source;

typedef struct {
    uint32_t iic_row_index;
    uint32_t manifest_index;
    uint32_t event_index;
    uint32_t split_pos;
    VAECIICMini224Source source_split;
    char event_id[VAE_C_IIC_MINI224_EVENT_ID_CAP];
} VAECIICMini224CentralRow;

/* Rows 0..799 follow train_rows.csv; 800..999 follow validation_rows.csv. */
typedef struct {
    float *mu;
    VAECIICMini224CentralRow *rows;
    size_t row_count;
    size_t latent_dim;
    char checkpoint_sha256[65];
    char train_matrix_sha256[65];
    char validation_matrix_sha256[65];
    char train_rows_sha256[65];
    char validation_rows_sha256[65];
} VAECIICMini224Central;

void vae_c_iic_mini224_central_init(VAECIICMini224Central *view);
void vae_c_iic_mini224_central_free(VAECIICMini224Central *view);

/* Override is for isolated build-tree negative fixtures only. No neural
 * forward, label access, artifact creation, sorting or normalization. */
int vae_c_iic_mini224_central_load(
    const char *repository_root,
    const char *artifact_directory_override,
    VAECIICMini224Central *view,
    char *error_message,
    size_t error_capacity
);

const float *vae_c_iic_mini224_central_row_mu(
    const VAECIICMini224Central *view,
    size_t iic_row_index
);

#endif
