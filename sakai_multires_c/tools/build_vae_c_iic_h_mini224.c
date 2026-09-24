/* Profile binding for the frozen mini224_1000 IIC FINAL and central mu. */
#define H_CENTRAL_HEADER "vae_c_iic_mini224_central.h"
#define H_CENTRAL_TYPE VAECIICMini224Central
#define H_CENTRAL_ROW_TYPE VAECIICMini224CentralRow
#define H_CENTRAL_INIT vae_c_iic_mini224_central_init
#define H_CENTRAL_LOAD vae_c_iic_mini224_central_load
#define H_CENTRAL_FREE vae_c_iic_mini224_central_free
#define H_SOURCE_TRAIN VAE_C_IIC_MINI224_SOURCE_TRAIN
#define H_SOURCE_VALIDATION VAE_C_IIC_MINI224_SOURCE_VALIDATION
#define H_ROWS VAE_C_IIC_MINI224_ROWS
#define H_LATENT_DIM VAE_C_IIC_MINI224_LATENT_DIM
#define H_RUN_ID "mini224_iic_v1_4236df3f9e82f0528fcea947b2e15e42fab8159c0d4b1ed03189e4a85daae8a5"
#define H_FINAL_SHA "e98dd0dd6513bc9636fe22af0341c41266bc00ce2629499e3aea2b5e50d83fe4"
#define H_RUN_CONFIG_SHA "8b2f2550a351f1daa117a28bb548c786f01036013e9b2b361597ed9c02a0605d"
#define H_EMBEDDING_DIR "mini224_bridge_v1_subset1000_best_epoch99_posterior_mu_central_v1"
#define H_TENSOR_MANIFEST "manifest_tensors_224_central.csv"
#define H_PROFILE_ID "mini224_bridge_v1"
#define H_IIC_BASE_COMMIT "8fc4fc59d0266b093fdbb76daedfc76d5f8c2a41"
#define H_IIC_BINARY_SHA "faaffa29384e33a5048014fbb6251daa4e07ae8cfffcb00afff1b43519bc224e"
#define H_FINAL_STEP 800u
#define H_FINAL_BN_BATCHES 1600u
#define H_ROW_ORDER "train800_then_validation200_native_split"
#define H_ROW_SOURCE(row) ((row)->source_split)
#define H_ROW_VALID(row, index) \
    ((row)->iic_row_index == (index) && \
     (row)->event_index == (row)->manifest_index + 1u && \
     (row)->source_split == ((index) < VAE_C_IIC_MINI224_TRAIN_ROWS ? \
         H_SOURCE_TRAIN : H_SOURCE_VALIDATION) && \
     (row)->split_pos == ((index) < VAE_C_IIC_MINI224_TRAIN_ROWS ? \
         (index) : (index) - VAE_C_IIC_MINI224_TRAIN_ROWS))
#define H_TRAIN_SUBSET "sakai_multires_c/configs/subset224_train_800_seed123.csv"
#define H_VALIDATION_SUBSET "sakai_multires_c/configs/subset224_val_200_seed123.csv"

#include "build_vae_c_iic_h_mini75.c"
