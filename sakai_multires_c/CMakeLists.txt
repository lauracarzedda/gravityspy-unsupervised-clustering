add_library(
    gravityspy_multires_data
    STATIC
        src/vae_tensor_spec.c
        src/vae_resize_u8.c
        src/vae_tensor_manifest.c
        src/vae_tensor_io.c
        src/vae_multires_event_loader.c
        src/vae_multires_batch.c
        ${PROJECT_SOURCE_DIR}/vae_event_dataset.c
        ${PROJECT_SOURCE_DIR}/vae_event_loader.c
        ${PROJECT_SOURCE_DIR}/vae_augmentation.c
        ${PROJECT_SOURCE_DIR}/vae_saved_split.c
)

target_include_directories(
    gravityspy_multires_data
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}
)

gravityspy_enable_warnings(gravityspy_multires_data)

add_library(
    gravityspy_vae_core
    STATIC
        src/vae_c_config.c
        src/vae_c_rng.c
        src/vae_c_shuffle.c
        src/vae_c_sha256.c
        src/vae_c_identity.c
        src/vae_c_init.c
        src/vae_c_tensor.c
        src/vae_c_parameter.c
        src/vae_c_ops.c
        src/vae_c_dense.c
        src/vae_c_spatial.c
        src/vae_c_batchnorm.c
        src/vae_c_conv2d.c
        src/vae_c_latent.c
        src/vae_c_objective.c
        src/vae_c_adam.c
        src/vae_c_model.c
        src/vae_c_model_init.c
        src/vae_c_model_workspace.c
        src/vae_c_model_encoder.c
        src/vae_c_model_decoder.c
        src/vae_c_model_vae.c
        src/vae_c_model_vae_backward.c
        src/vae_c_model_train.c
        src/vae_c_checkpoint.c
)

target_include_directories(
    gravityspy_vae_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

gravityspy_enable_warnings(gravityspy_vae_core)

if(NOT MSVC)
    target_link_libraries(gravityspy_vae_core PUBLIC m)
endif()

add_library(
    gravityspy_vae_mu_contract
    STATIC
        src/vae_c_mu_contract.c
)
target_link_libraries(
    gravityspy_vae_mu_contract
    PUBLIC
        gravityspy_multires_data
        gravityspy_vae_core
)
target_include_directories(
    gravityspy_vae_mu_contract
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
gravityspy_enable_warnings(gravityspy_vae_mu_contract)

add_library(
    gravityspy_vae_iic_core
    STATIC
        src/vae_c_iic_loss.c
)
target_include_directories(
    gravityspy_vae_iic_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
gravityspy_enable_warnings(gravityspy_vae_iic_core)

if(NOT MSVC)
    target_link_libraries(gravityspy_vae_iic_core PUBLIC m)
endif()

add_library(
    gravityspy_vae_iic_head_core
    STATIC
        src/vae_c_iic_head.c
)
target_link_libraries(
    gravityspy_vae_iic_head_core
    PUBLIC gravityspy_vae_core
)
target_include_directories(
    gravityspy_vae_iic_head_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
gravityspy_enable_warnings(gravityspy_vae_iic_head_core)

add_library(
    gravityspy_vae_iic_pair_core
    STATIC
        src/vae_c_iic_pair.c
)
target_link_libraries(
    gravityspy_vae_iic_pair_core
    PUBLIC
        gravityspy_vae_iic_head_core
        gravityspy_vae_iic_core
)
target_include_directories(
    gravityspy_vae_iic_pair_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
gravityspy_enable_warnings(gravityspy_vae_iic_pair_core)

add_library(
    gravityspy_vae_iic_pair_train_core
    STATIC
        src/vae_c_iic_pair_train.c
)
target_link_libraries(
    gravityspy_vae_iic_pair_train_core
    PUBLIC
        gravityspy_vae_iic_pair_core
        gravityspy_vae_core
)
target_include_directories(
    gravityspy_vae_iic_pair_train_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
gravityspy_enable_warnings(gravityspy_vae_iic_pair_train_core)

add_library(
    gravityspy_vae_iic_ensemble_core
    STATIC
        src/vae_c_iic_ensemble.c
)
target_link_libraries(
    gravityspy_vae_iic_ensemble_core
    PUBLIC gravityspy_vae_iic_pair_train_core
)
target_include_directories(
    gravityspy_vae_iic_ensemble_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
gravityspy_enable_warnings(gravityspy_vae_iic_ensemble_core)

add_library(
    gravityspy_vae_iic_h_core
    STATIC
        src/vae_c_iic_h.c
)
target_link_libraries(
    gravityspy_vae_iic_h_core
    PUBLIC gravityspy_vae_iic_ensemble_core gravityspy_vae_iic_core
)
target_include_directories(
    gravityspy_vae_iic_h_core
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)
gravityspy_enable_warnings(gravityspy_vae_iic_h_core)

add_library(
    gravityspy_vae_iic_affinity_core
    STATIC
        src/vae_c_iic_affinity.c
)
target_include_directories(
    gravityspy_vae_iic_affinity_core
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)
gravityspy_enable_warnings(gravityspy_vae_iic_affinity_core)
if(NOT MSVC)
    target_link_libraries(gravityspy_vae_iic_affinity_core PUBLIC m)
endif()

add_library(
    gravityspy_vae_iic_checkpoint_core
    STATIC
        src/vae_c_iic_checkpoint.c
)
target_link_libraries(
    gravityspy_vae_iic_checkpoint_core
    PUBLIC gravityspy_vae_iic_ensemble_core gravityspy_vae_core
)
target_include_directories(
    gravityspy_vae_iic_checkpoint_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
gravityspy_enable_warnings(gravityspy_vae_iic_checkpoint_core)

add_executable(
    run_vae_c_iic_synthetic
    tools/run_vae_c_iic_synthetic.c
)
target_link_libraries(
    run_vae_c_iic_synthetic
    PRIVATE gravityspy_vae_iic_checkpoint_core
)
gravityspy_enable_warnings(run_vae_c_iic_synthetic)

add_executable(
    inspect_vae_c_iic_checkpoint
    tools/inspect_vae_c_iic_checkpoint.c
)
target_link_libraries(
    inspect_vae_c_iic_checkpoint
    PRIVATE gravityspy_vae_iic_checkpoint_core
)
gravityspy_enable_warnings(inspect_vae_c_iic_checkpoint)

add_library(
    gravityspy_vae_iic_mini75_input
    STATIC
        src/vae_c_iic_mini75_central.c
        src/vae_c_iic_mini75_perturbed.c
        src/vae_c_iic_png_identity.c
)
target_link_libraries(
    gravityspy_vae_iic_mini75_input
    PUBLIC gravityspy_vae_mu_contract
)
target_include_directories(
    gravityspy_vae_iic_mini75_input
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
gravityspy_enable_warnings(gravityspy_vae_iic_mini75_input)

add_library(
    gravityspy_vae_iic_mini112_input
    STATIC
        src/vae_c_iic_mini112_central.c
        src/vae_c_iic_mini112_perturbed.c
)
target_link_libraries(
    gravityspy_vae_iic_mini112_input
    PUBLIC gravityspy_vae_mu_contract
)
target_include_directories(
    gravityspy_vae_iic_mini112_input
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)
gravityspy_enable_warnings(gravityspy_vae_iic_mini112_input)

add_library(
    gravityspy_vae_iic_mini224_central
    STATIC src/vae_c_iic_mini224_central.c
)
target_link_libraries(
    gravityspy_vae_iic_mini224_central
    PUBLIC gravityspy_vae_mu_contract
)
target_include_directories(
    gravityspy_vae_iic_mini224_central
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)
gravityspy_enable_warnings(gravityspy_vae_iic_mini224_central)

add_library(
    gravityspy_vae_iic_mini224_perturbed
    STATIC src/vae_c_iic_mini224_perturbed.c
)
target_link_libraries(
    gravityspy_vae_iic_mini224_perturbed
    PUBLIC gravityspy_vae_iic_mini224_central
)
target_include_directories(
    gravityspy_vae_iic_mini224_perturbed
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)
gravityspy_enable_warnings(gravityspy_vae_iic_mini224_perturbed)

add_executable(
    validate_vae_c_iic_mini75_input
    tools/validate_vae_c_iic_mini75_input.c
)
target_link_libraries(
    validate_vae_c_iic_mini75_input
    PRIVATE gravityspy_vae_iic_mini75_input
)
if(WIN32)
    target_link_libraries(validate_vae_c_iic_mini75_input PRIVATE psapi)
endif()
gravityspy_enable_warnings(validate_vae_c_iic_mini75_input)

add_executable(
    run_vae_c_iic_mini75
    tools/run_vae_c_iic_mini75.c
)
target_link_libraries(
    run_vae_c_iic_mini75
    PRIVATE
        gravityspy_vae_iic_mini75_input
        gravityspy_vae_iic_checkpoint_core
)
gravityspy_enable_warnings(run_vae_c_iic_mini75)

if(WIN32)
    target_link_libraries(run_vae_c_iic_mini75 PRIVATE psapi)
endif()

add_executable(
    run_vae_c_iic_mini112
    tools/run_vae_c_iic_mini112.c
)
target_link_libraries(
    run_vae_c_iic_mini112
    PRIVATE
        gravityspy_vae_iic_mini112_input
        gravityspy_vae_iic_mini75_input
        gravityspy_vae_iic_checkpoint_core
)
gravityspy_enable_warnings(run_vae_c_iic_mini112)
if(WIN32)
    target_link_libraries(run_vae_c_iic_mini112 PRIVATE psapi)
endif()

add_executable(
    run_vae_c_iic_mini224
    tools/run_vae_c_iic_mini224.c
)
target_link_libraries(
    run_vae_c_iic_mini224
    PRIVATE
        gravityspy_vae_iic_mini224_perturbed
        gravityspy_vae_iic_mini75_input
        gravityspy_vae_iic_checkpoint_core
)
gravityspy_enable_warnings(run_vae_c_iic_mini224)
if(WIN32)
    target_link_libraries(run_vae_c_iic_mini224 PRIVATE psapi)
endif()

add_executable(
    build_vae_c_iic_h_mini75
    tools/build_vae_c_iic_h_mini75.c
)
target_link_libraries(
    build_vae_c_iic_h_mini75
    PRIVATE
        gravityspy_vae_iic_h_core
        gravityspy_vae_iic_checkpoint_core
        gravityspy_vae_iic_mini75_input
)
gravityspy_enable_warnings(build_vae_c_iic_h_mini75)

add_executable(
    build_vae_c_iic_h_mini112
    tools/build_vae_c_iic_h_mini112.c
)
target_link_libraries(
    build_vae_c_iic_h_mini112
    PRIVATE
        gravityspy_vae_iic_h_core
        gravityspy_vae_iic_checkpoint_core
        gravityspy_vae_iic_mini112_input
)
gravityspy_enable_warnings(build_vae_c_iic_h_mini112)

add_executable(
    build_vae_c_iic_h_mini224
    tools/build_vae_c_iic_h_mini224.c
)
target_link_libraries(
    build_vae_c_iic_h_mini224
    PRIVATE
        gravityspy_vae_iic_h_core
        gravityspy_vae_iic_checkpoint_core
        gravityspy_vae_iic_mini224_central
)
gravityspy_enable_warnings(build_vae_c_iic_h_mini224)

add_executable(
    build_vae_c_iic_affinity_mini75
    tools/build_vae_c_iic_affinity_mini75.c
)
target_link_libraries(
    build_vae_c_iic_affinity_mini75
    PRIVATE gravityspy_vae_iic_affinity_core gravityspy_vae_mu_contract
)
gravityspy_enable_warnings(build_vae_c_iic_affinity_mini75)

add_executable(
    build_vae_c_iic_affinity_mini112
    tools/build_vae_c_iic_affinity_mini112.c
)
target_link_libraries(
    build_vae_c_iic_affinity_mini112
    PRIVATE gravityspy_vae_iic_affinity_core gravityspy_vae_mu_contract
)
gravityspy_enable_warnings(build_vae_c_iic_affinity_mini112)

add_executable(
    build_vae_c_iic_affinity_mini224
    tools/build_vae_c_iic_affinity_mini224.c
)
target_link_libraries(
    build_vae_c_iic_affinity_mini224
    PRIVATE gravityspy_vae_iic_affinity_core gravityspy_vae_mu_contract
)
gravityspy_enable_warnings(build_vae_c_iic_affinity_mini224)

add_executable(
    extract_vae_c_mu
    tools/extract_vae_c_mu.c
)
target_link_libraries(
    extract_vae_c_mu
    PRIVATE gravityspy_vae_mu_contract
)
gravityspy_enable_warnings(extract_vae_c_mu)

add_library(
    gravityspy_baseline_kmeans
    STATIC
        src/baseline_kmeans.c
)
target_include_directories(
    gravityspy_baseline_kmeans
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
gravityspy_enable_warnings(gravityspy_baseline_kmeans)

if(NOT MSVC)
    target_link_libraries(gravityspy_baseline_kmeans PUBLIC m)
endif()

add_library(
    gravityspy_baseline_clustering_metrics
    STATIC
        src/baseline_clustering_metrics.c
)
target_include_directories(
    gravityspy_baseline_clustering_metrics
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
gravityspy_enable_warnings(gravityspy_baseline_clustering_metrics)

if(NOT MSVC)
    target_link_libraries(
        gravityspy_baseline_clustering_metrics
        PUBLIC m
    )
endif()

add_executable(
    run_vae_c_mu_kmeans
    tools/run_vae_c_mu_kmeans.c
)
target_link_libraries(
    run_vae_c_mu_kmeans
    PRIVATE
        gravityspy_vae_mu_contract
        gravityspy_baseline_kmeans
)
gravityspy_enable_warnings(run_vae_c_mu_kmeans)

add_executable(
    evaluate_vae_c_mu_kmeans
    tools/evaluate_vae_c_mu_kmeans.c
)
target_link_libraries(
    evaluate_vae_c_mu_kmeans
    PRIVATE
        gravityspy_multires_data
        gravityspy_vae_mu_contract
        gravityspy_baseline_clustering_metrics
)
gravityspy_enable_warnings(evaluate_vae_c_mu_kmeans)

add_executable(
    evaluate_vae_c_iic_spectral_mini75
    tools/evaluate_vae_c_iic_spectral_mini75.c
)
target_link_libraries(
    evaluate_vae_c_iic_spectral_mini75
    PRIVATE
        gravityspy_multires_data
        gravityspy_vae_mu_contract
        gravityspy_baseline_clustering_metrics
)
gravityspy_enable_warnings(evaluate_vae_c_iic_spectral_mini75)

add_executable(
    evaluate_vae_c_iic_spectral_common
    tools/evaluate_vae_c_iic_spectral_common.c
)
target_link_libraries(
    evaluate_vae_c_iic_spectral_common
    PRIVATE
        gravityspy_multires_data
        gravityspy_vae_mu_contract
        gravityspy_baseline_clustering_metrics
)
gravityspy_enable_warnings(evaluate_vae_c_iic_spectral_common)

add_executable(
    build_vae_c_iic_contact_sheets_common
    tools/build_vae_c_iic_contact_sheets_common.c
)
target_link_libraries(
    build_vae_c_iic_contact_sheets_common
    PRIVATE
        gravityspy_vae_iic_mini75_input
        gravityspy_vae_iic_mini112_input
        gravityspy_vae_iic_mini224_central
)
gravityspy_enable_warnings(build_vae_c_iic_contact_sheets_common)

if(NOT MSVC)
    target_link_libraries(build_vae_c_iic_contact_sheets_common PRIVATE m)
endif()

add_executable(
    run_vae_c_mini75_overfit
    tools/run_vae_c_mini75_overfit.c
)
target_link_libraries(
    run_vae_c_mini75_overfit
    PRIVATE
        gravityspy_multires_data
        gravityspy_vae_core
)
gravityspy_enable_warnings(run_vae_c_mini75_overfit)

add_executable(
    run_vae_c_mini75_perturbed_sanity
    tools/run_vae_c_mini75_perturbed_sanity.c
)
target_link_libraries(
    run_vae_c_mini75_perturbed_sanity
    PRIVATE
        gravityspy_multires_data
        gravityspy_vae_core
)
gravityspy_enable_warnings(run_vae_c_mini75_perturbed_sanity)

add_executable(
    run_vae_c_mini75_training
    tools/run_vae_c_mini75_training.c
)
target_link_libraries(
    run_vae_c_mini75_training
    PRIVATE
        gravityspy_multires_data
        gravityspy_vae_core
)
gravityspy_enable_warnings(run_vae_c_mini75_training)

add_executable(
    run_vae_c_mini112_training
    tools/run_vae_c_mini112_training.c
)
target_link_libraries(
    run_vae_c_mini112_training
    PRIVATE
        gravityspy_multires_data
        gravityspy_vae_core
)
gravityspy_enable_warnings(run_vae_c_mini112_training)

add_executable(
    run_vae_c_mini224_training
    tools/run_vae_c_mini224_training.c
)
target_link_libraries(
    run_vae_c_mini224_training
    PRIVATE
        gravityspy_multires_data
        gravityspy_vae_core
)
gravityspy_enable_warnings(run_vae_c_mini224_training)

add_executable(
    run_vae_c_mini224_smoke
    tools/run_vae_c_mini224_smoke.c
)
target_link_libraries(
    run_vae_c_mini224_smoke
    PRIVATE
        gravityspy_multires_data
        gravityspy_vae_core
)
gravityspy_enable_warnings(run_vae_c_mini224_smoke)

add_executable(
    run_vae_c_mini224_benchmark
    tools/run_vae_c_mini224_benchmark.c
)
target_link_libraries(
    run_vae_c_mini224_benchmark
    PRIVATE
        gravityspy_multires_data
        gravityspy_vae_core
)
if(WIN32)
    target_link_libraries(run_vae_c_mini224_benchmark PRIVATE psapi)
endif()
gravityspy_enable_warnings(run_vae_c_mini224_benchmark)

if(BUILD_TESTING)
    add_executable(
        test_vae_c_iic_png_identity
        tests/test_vae_c_iic_png_identity.c
    )
    target_link_libraries(
        test_vae_c_iic_png_identity
        PRIVATE gravityspy_vae_iic_mini75_input
    )
    gravityspy_enable_warnings(test_vae_c_iic_png_identity)

    add_executable(
        test_tensor_manifest_io
        tests/test_tensor_manifest_io.c
    )
    target_link_libraries(
        test_tensor_manifest_io
        PRIVATE gravityspy_multires_data
    )
    gravityspy_enable_warnings(test_tensor_manifest_io)

    add_executable(
        test_resize_equivalence
        tests/test_resize_equivalence.c
    )
    target_link_libraries(
        test_resize_equivalence
        PRIVATE gravityspy_multires_data
    )
    gravityspy_enable_warnings(test_resize_equivalence)

    add_executable(
        test_multires_event_loader
        tests/test_multires_event_loader.c
    )
    target_link_libraries(
        test_multires_event_loader
        PRIVATE gravityspy_multires_data
    )
    gravityspy_enable_warnings(test_multires_event_loader)

    add_executable(
        test_multires_batch
        tests/test_multires_batch.c
    )
    target_link_libraries(
        test_multires_batch
        PRIVATE gravityspy_multires_data
    )
    gravityspy_enable_warnings(test_multires_batch)

    add_executable(
        test_multires_stateless_augmentation
        tests/test_multires_stateless_augmentation.c
    )
    target_link_libraries(
        test_multires_stateless_augmentation
        PRIVATE gravityspy_multires_data
    )
    gravityspy_enable_warnings(test_multires_stateless_augmentation)

    add_executable(
        test_multires_epoch
        tests/test_multires_epoch.c
    )
    target_link_libraries(
        test_multires_epoch
        PRIVATE gravityspy_multires_data
    )
    gravityspy_enable_warnings(test_multires_epoch)

    add_executable(
        test_subset224_epoch
        tests/test_subset224_epoch.c
    )
    target_link_libraries(
        test_subset224_epoch
        PRIVATE gravityspy_multires_data
    )
    gravityspy_enable_warnings(test_subset224_epoch)

    add_executable(
        test_vae_c_config
        tests/test_vae_c_config.c
    )
    target_link_libraries(
        test_vae_c_config
        PRIVATE gravityspy_vae_core
    )
    gravityspy_enable_warnings(test_vae_c_config)

    add_executable(
        test_vae_c_profiles_v2
        tests/test_vae_c_profiles_v2.c
    )
    target_link_libraries(
        test_vae_c_profiles_v2
        PRIVATE gravityspy_vae_core
    )
    gravityspy_enable_warnings(test_vae_c_profiles_v2)

    add_executable(
        test_vae_c_rng_init
        tests/test_vae_c_rng_init.c
    )
    target_link_libraries(
        test_vae_c_rng_init
        PRIVATE gravityspy_vae_core
    )
    gravityspy_enable_warnings(test_vae_c_rng_init)

    add_executable(
        test_vae_c_shuffle
        tests/test_vae_c_shuffle.c
    )
    target_link_libraries(
        test_vae_c_shuffle
        PRIVATE gravityspy_vae_core
    )
    gravityspy_enable_warnings(test_vae_c_shuffle)

    add_executable(
        test_vae_c_sha256
        tests/test_vae_c_sha256.c
    )
    target_link_libraries(
        test_vae_c_sha256
        PRIVATE gravityspy_vae_core
    )
    gravityspy_enable_warnings(test_vae_c_sha256)

    add_executable(
        test_vae_c_identity
        tests/test_vae_c_identity.c
    )
    target_link_libraries(
        test_vae_c_identity
        PRIVATE gravityspy_vae_core
    )
    gravityspy_enable_warnings(test_vae_c_identity)

    add_executable(
        test_vae_c_tensor_parameter
        tests/test_vae_c_tensor_parameter.c
    )
    target_link_libraries(
        test_vae_c_tensor_parameter
        PRIVATE gravityspy_vae_core
    )
    gravityspy_enable_warnings(test_vae_c_tensor_parameter)

    add_executable(
        test_vae_c_basic_ops
        tests/test_vae_c_basic_ops.c
    )
    target_link_libraries(
        test_vae_c_basic_ops
        PRIVATE gravityspy_vae_core
    )
    gravityspy_enable_warnings(test_vae_c_basic_ops)

    add_executable(
        test_vae_c_spatial_ops
        tests/test_vae_c_spatial_ops.c
    )
    target_link_libraries(
        test_vae_c_spatial_ops
        PRIVATE gravityspy_vae_core
    )
    gravityspy_enable_warnings(test_vae_c_spatial_ops)

    add_executable(
        test_vae_c_batchnorm
        tests/test_vae_c_batchnorm.c
    )
    target_link_libraries(
        test_vae_c_batchnorm
        PRIVATE gravityspy_vae_core
    )
    gravityspy_enable_warnings(test_vae_c_batchnorm)

    add_executable(
        test_vae_c_conv2d
        tests/test_vae_c_conv2d.c
    )
    target_link_libraries(
        test_vae_c_conv2d
        PRIVATE gravityspy_vae_core
    )
    gravityspy_enable_warnings(test_vae_c_conv2d)

    add_executable(
        test_vae_c_latent
        tests/test_vae_c_latent.c
    )
    target_link_libraries(
        test_vae_c_latent
        PRIVATE gravityspy_vae_core
    )
    gravityspy_enable_warnings(test_vae_c_latent)

    add_executable(
        test_vae_c_bce
        tests/test_vae_c_bce.c
    )
    target_link_libraries(
        test_vae_c_bce
        PRIVATE gravityspy_vae_core
    )
    gravityspy_enable_warnings(test_vae_c_bce)

    add_executable(
        test_vae_c_kl
        tests/test_vae_c_kl.c
    )
    target_link_libraries(
        test_vae_c_kl
        PRIVATE gravityspy_vae_core
    )
    gravityspy_enable_warnings(test_vae_c_kl)

    add_executable(
        test_vae_c_vae_loss
        tests/test_vae_c_vae_loss.c
    )
    target_link_libraries(
        test_vae_c_vae_loss
        PRIVATE gravityspy_vae_core
    )
    gravityspy_enable_warnings(test_vae_c_vae_loss)

    add_executable(
        test_vae_c_adam
        tests/test_vae_c_adam.c
    )
    target_link_libraries(
        test_vae_c_adam
        PRIVATE gravityspy_vae_core
    )
    gravityspy_enable_warnings(test_vae_c_adam)

    add_executable(
        test_vae_c_model_allocation
        tests/test_vae_c_model_allocation.c
    )
    target_link_libraries(
        test_vae_c_model_allocation
        PRIVATE gravityspy_vae_core
    )
    gravityspy_enable_warnings(test_vae_c_model_allocation)

    add_executable(
        test_vae_c_model_initialization
        tests/test_vae_c_model_initialization.c
    )
    target_link_libraries(
        test_vae_c_model_initialization
        PRIVATE gravityspy_vae_core
    )
    gravityspy_enable_warnings(test_vae_c_model_initialization)

    add_executable(
        test_vae_c_model_workspace
        tests/test_vae_c_model_workspace.c
    )
    target_link_libraries(
        test_vae_c_model_workspace
        PRIVATE gravityspy_vae_core
    )
    gravityspy_enable_warnings(test_vae_c_model_workspace)

    add_executable(
        test_vae_c_model_encoder_forward
        tests/test_vae_c_model_encoder_forward.c
    )
    target_link_libraries(
        test_vae_c_model_encoder_forward
        PRIVATE gravityspy_vae_core
    )
    gravityspy_enable_warnings(test_vae_c_model_encoder_forward)

    add_executable(
        test_vae_c_model_decoder_forward
        tests/test_vae_c_model_decoder_forward.c
    )
    target_link_libraries(
        test_vae_c_model_decoder_forward
        PRIVATE gravityspy_vae_core
    )
    gravityspy_enable_warnings(test_vae_c_model_decoder_forward)

    add_executable(
        test_vae_c_model_vae_forward
        tests/test_vae_c_model_vae_forward.c
    )
    target_link_libraries(
        test_vae_c_model_vae_forward
        PRIVATE gravityspy_vae_core
    )
    gravityspy_enable_warnings(test_vae_c_model_vae_forward)

    add_executable(
        test_vae_c_model_vae_backward
        tests/test_vae_c_model_vae_backward.c
    )
    target_link_libraries(
        test_vae_c_model_vae_backward
        PRIVATE gravityspy_vae_core
    )
    gravityspy_enable_warnings(test_vae_c_model_vae_backward)

    add_executable(
        test_vae_c_model_train
        tests/test_vae_c_model_train.c
    )
    target_link_libraries(
        test_vae_c_model_train
        PRIVATE gravityspy_vae_core
    )
    gravityspy_enable_warnings(test_vae_c_model_train)

    add_executable(
        test_vae_c_checkpoint
        tests/test_vae_c_checkpoint.c
    )
    target_compile_definitions(
        gravityspy_vae_core
        PRIVATE VAE_C_CHECKPOINT_TEST_HOOKS=1
    )
    target_link_libraries(
        test_vae_c_checkpoint
        PRIVATE gravityspy_vae_core
    )
    gravityspy_enable_warnings(test_vae_c_checkpoint)

    add_executable(
        test_vae_c_mu_contract
        tests/test_vae_c_mu_contract.c
    )
    target_link_libraries(
        test_vae_c_mu_contract
        PRIVATE gravityspy_vae_mu_contract
    )
    gravityspy_enable_warnings(test_vae_c_mu_contract)

    add_executable(
        test_vae_c_iic_loss
        tests/test_vae_c_iic_loss.c
    )
    target_link_libraries(
        test_vae_c_iic_loss
        PRIVATE gravityspy_vae_iic_core
    )
    gravityspy_enable_warnings(test_vae_c_iic_loss)

    add_executable(
        test_vae_c_iic_loss_head_integration
        tests/test_vae_c_iic_loss_head_integration.c
    )
    target_link_libraries(
        test_vae_c_iic_loss_head_integration
        PRIVATE
            gravityspy_vae_iic_head_core
            gravityspy_vae_iic_core
    )
    gravityspy_enable_warnings(test_vae_c_iic_loss_head_integration)

    add_executable(
        test_vae_c_iic_head
        tests/test_vae_c_iic_head.c
    )
    target_link_libraries(
        test_vae_c_iic_head
        PRIVATE gravityspy_vae_iic_head_core
    )
    gravityspy_enable_warnings(test_vae_c_iic_head)

    add_executable(
        test_vae_c_iic_pair
        tests/test_vae_c_iic_pair.c
    )
    target_link_libraries(
        test_vae_c_iic_pair
        PRIVATE gravityspy_vae_iic_pair_core
    )
    gravityspy_enable_warnings(test_vae_c_iic_pair)

    add_executable(
        test_vae_c_iic_pair_train
        tests/test_vae_c_iic_pair_train.c
    )
    target_link_libraries(
        test_vae_c_iic_pair_train
        PRIVATE gravityspy_vae_iic_pair_train_core
    )
    gravityspy_enable_warnings(test_vae_c_iic_pair_train)

    add_executable(
        test_vae_c_iic_ensemble
        tests/test_vae_c_iic_ensemble.c
    )
    target_link_libraries(
        test_vae_c_iic_ensemble
        PRIVATE gravityspy_vae_iic_ensemble_core
    )
    gravityspy_enable_warnings(test_vae_c_iic_ensemble)

    add_executable(
        test_vae_c_iic_checkpoint
        tests/test_vae_c_iic_checkpoint.c
    )
    target_link_libraries(
        test_vae_c_iic_checkpoint
        PRIVATE gravityspy_vae_iic_checkpoint_core
    )
    gravityspy_enable_warnings(test_vae_c_iic_checkpoint)

    add_executable(
        test_vae_c_iic_h
        tests/test_vae_c_iic_h.c
    )
    target_link_libraries(
        test_vae_c_iic_h
        PRIVATE
            gravityspy_vae_iic_h_core
            gravityspy_vae_iic_checkpoint_core
            gravityspy_vae_iic_mini75_input
    )
    gravityspy_enable_warnings(test_vae_c_iic_h)

    add_executable(
        test_vae_c_iic_h_mini112
        tests/test_vae_c_iic_h_mini112.c
    )
    target_link_libraries(
        test_vae_c_iic_h_mini112
        PRIVATE
            gravityspy_vae_iic_h_core
            gravityspy_vae_iic_checkpoint_core
            gravityspy_vae_iic_mini112_input
    )
    gravityspy_enable_warnings(test_vae_c_iic_h_mini112)

    add_executable(
        test_vae_c_iic_affinity
        tests/test_vae_c_iic_affinity.c
    )
    target_link_libraries(
        test_vae_c_iic_affinity
        PRIVATE gravityspy_vae_iic_affinity_core
    )
    gravityspy_enable_warnings(test_vae_c_iic_affinity)

    add_executable(
        test_vae_c_iic_affinity_producer
        tests/test_vae_c_iic_affinity_producer.c
    )
    target_link_libraries(
        test_vae_c_iic_affinity_producer
        PRIVATE gravityspy_vae_mu_contract
    )
    gravityspy_enable_warnings(test_vae_c_iic_affinity_producer)

    add_executable(
        test_baseline_kmeans
        tests/test_baseline_kmeans.c
    )
    target_link_libraries(
        test_baseline_kmeans
        PRIVATE gravityspy_baseline_kmeans
    )
    gravityspy_enable_warnings(test_baseline_kmeans)

    add_executable(
        test_baseline_clustering_metrics
        tests/test_baseline_clustering_metrics.c
    )
    target_link_libraries(
        test_baseline_clustering_metrics
        PRIVATE gravityspy_baseline_clustering_metrics
    )
    gravityspy_enable_warnings(test_baseline_clustering_metrics)

    add_executable(
        test_vae_c_mu_kmeans_evaluator
        tests/test_vae_c_mu_kmeans_evaluator.c
    )
    target_link_libraries(
        test_vae_c_mu_kmeans_evaluator
        PRIVATE
            gravityspy_multires_data
            gravityspy_vae_mu_contract
            gravityspy_baseline_clustering_metrics
    )
    gravityspy_enable_warnings(test_vae_c_mu_kmeans_evaluator)

    add_executable(
        test_vae_c_mu_kmeans_validate
        tests/test_vae_c_mu_kmeans_validate.c
    )
    target_link_libraries(
        test_vae_c_mu_kmeans_validate
        PRIVATE
            gravityspy_vae_mu_contract
            gravityspy_baseline_kmeans
    )
    gravityspy_enable_warnings(test_vae_c_mu_kmeans_validate)

    if(NOT MSVC)
        target_link_libraries(test_multires_epoch PRIVATE m)
        target_link_libraries(test_subset224_epoch PRIVATE m)
    endif()

    add_test(
        NAME test_tensor_manifest_io
        COMMAND test_tensor_manifest_io
    )

    add_test(
        NAME test_resize_equivalence
        COMMAND test_resize_equivalence
    )

    add_test(
        NAME test_multires_event_loader
        COMMAND test_multires_event_loader manifest_events_272.csv
    )

    add_test(
        NAME test_multires_batch_legacy
        COMMAND test_multires_batch legacy_text
    )

    add_test(
        NAME test_multires_batch_symmetric
        COMMAND test_multires_batch symmetric_figure
    )

    add_test(
        NAME test_multires_stateless_augmentation
        COMMAND test_multires_stateless_augmentation
    )

    add_test(
        NAME test_multires_epoch_quick
        COMMAND test_multires_epoch 8 456 1 symmetric_figure
    )

    add_test(
        NAME test_subset224_epoch
        COMMAND
            test_subset224_epoch
            sakai_multires_c/configs/subset224_train_800_seed123.csv
            sakai_multires_c/configs/subset224_val_200_seed123.csv
            32
            456
            symmetric_figure
    )

    set(VAE_C_CONFIG_FIXTURE_DIRECTORY
        "${CMAKE_CURRENT_BINARY_DIR}/test_vae_c_config_fixtures"
    )
    file(MAKE_DIRECTORY "${VAE_C_CONFIG_FIXTURE_DIRECTORY}")

    add_test(
        NAME test_vae_c_config
        COMMAND
            test_vae_c_config
            ${CMAKE_CURRENT_SOURCE_DIR}/configs/vae_c_profiles_v1.csv
            ${VAE_C_CONFIG_FIXTURE_DIRECTORY}
    )

    add_test(
        NAME test_vae_c_profiles_v1_sha256
        COMMAND
            ${CMAKE_COMMAND} -E sha256sum
            ${CMAKE_CURRENT_SOURCE_DIR}/configs/vae_c_profiles_v1.csv
    )
    set_tests_properties(
        test_vae_c_profiles_v1_sha256
        PROPERTIES
            PASS_REGULAR_EXPRESSION
                "e11d4bd722474bfbc50432b843679ef4f07e94a875d2e5ace5b8cd85eff6860f"
    )

    add_test(
        NAME test_vae_c_profiles_v2
        COMMAND
            test_vae_c_profiles_v2
            ${CMAKE_CURRENT_SOURCE_DIR}/configs/vae_c_profiles_v1.csv
            ${CMAKE_CURRENT_SOURCE_DIR}/configs/vae_c_profiles_v2.csv
    )

    add_test(
        NAME test_vae_c_rng_init
        COMMAND test_vae_c_rng_init
    )

    add_test(
        NAME test_vae_c_shuffle
        COMMAND test_vae_c_shuffle
    )

    add_test(
        NAME test_vae_c_sha256
        COMMAND test_vae_c_sha256
    )

    set(VAE_C_IDENTITY_FIXTURE_DIRECTORY
        "${CMAKE_CURRENT_BINARY_DIR}/test_vae_c_identity_fixtures"
    )
    file(MAKE_DIRECTORY "${VAE_C_IDENTITY_FIXTURE_DIRECTORY}")

    add_test(
        NAME test_vae_c_identity
        COMMAND
            test_vae_c_identity
            ${CMAKE_CURRENT_SOURCE_DIR}/configs/vae_c_profiles_v2.csv
            ${PROJECT_SOURCE_DIR}/manifest_events_272.csv
            ${PROJECT_SOURCE_DIR}/manifest_tensors_224_central.csv
            ${PROJECT_SOURCE_DIR}/split_train_indices_seed123.csv
            ${PROJECT_SOURCE_DIR}/split_val_indices_seed123.csv
            ${CMAKE_CURRENT_SOURCE_DIR}/configs/subset224_train_800_seed123.csv
            ${CMAKE_CURRENT_SOURCE_DIR}/configs/subset224_val_200_seed123.csv
            ${VAE_C_IDENTITY_FIXTURE_DIRECTORY}
    )

    add_test(
        NAME test_vae_c_mini224_runner
        COMMAND
            run_vae_c_mini224_training
            --validate-only
            --batch-size 32
            --max-epochs 50
            --patience 10
            # Syntactically valid test metadata; real runs supply HEAD.
            --code-commit 0123456789abcdef0123456789abcdef01234567
            ${PROJECT_SOURCE_DIR}
    )
    set_tests_properties(
        test_vae_c_mini224_runner
        PROPERTIES
            PASS_REGULAR_EXPRESSION
                "MINI224 TRAINING VALIDATION PASSED"
    )

    add_test(
        NAME test_vae_c_tensor_parameter
        COMMAND test_vae_c_tensor_parameter
    )

    add_test(
        NAME test_vae_c_basic_ops
        COMMAND test_vae_c_basic_ops
    )

    add_test(
        NAME test_vae_c_spatial_ops
        COMMAND test_vae_c_spatial_ops
    )

    add_test(
        NAME test_vae_c_batchnorm
        COMMAND test_vae_c_batchnorm
    )

    add_test(
        NAME test_vae_c_conv2d
        COMMAND test_vae_c_conv2d
    )

    add_test(
        NAME test_vae_c_latent
        COMMAND test_vae_c_latent
    )

    add_test(
        NAME test_vae_c_bce
        COMMAND test_vae_c_bce
    )

    add_test(
        NAME test_vae_c_kl
        COMMAND test_vae_c_kl
    )

    add_test(
        NAME test_vae_c_vae_loss
        COMMAND test_vae_c_vae_loss
    )

    add_test(
        NAME test_vae_c_adam
        COMMAND test_vae_c_adam
    )

    add_test(
        NAME test_vae_c_model_allocation
        COMMAND
            test_vae_c_model_allocation
            ${CMAKE_CURRENT_SOURCE_DIR}/configs/vae_c_profiles_v1.csv
    )

    add_test(
        NAME test_vae_c_model_initialization
        COMMAND
            test_vae_c_model_initialization
            ${CMAKE_CURRENT_SOURCE_DIR}/configs/vae_c_profiles_v1.csv
    )

    add_test(
        NAME test_vae_c_model_workspace
        COMMAND
            test_vae_c_model_workspace
            ${CMAKE_CURRENT_SOURCE_DIR}/configs/vae_c_profiles_v1.csv
    )

    add_test(
        NAME test_vae_c_model_encoder_forward
        COMMAND
            test_vae_c_model_encoder_forward
            ${CMAKE_CURRENT_SOURCE_DIR}/configs/vae_c_profiles_v1.csv
    )

    add_test(
        NAME test_vae_c_model_decoder_forward
        COMMAND
            test_vae_c_model_decoder_forward
            ${CMAKE_CURRENT_SOURCE_DIR}/configs/vae_c_profiles_v1.csv
    )

    add_test(
        NAME test_vae_c_model_vae_forward
        COMMAND
            test_vae_c_model_vae_forward
            ${CMAKE_CURRENT_SOURCE_DIR}/configs/vae_c_profiles_v1.csv
    )

    add_test(
        NAME test_vae_c_model_vae_backward
        COMMAND
            test_vae_c_model_vae_backward
            ${CMAKE_CURRENT_SOURCE_DIR}/configs/vae_c_profiles_v1.csv
    )

    add_test(
        NAME test_vae_c_model_train
        COMMAND
            test_vae_c_model_train
            ${CMAKE_CURRENT_SOURCE_DIR}/configs/vae_c_profiles_v1.csv
    )

    set(VAE_C_CHECKPOINT_FIXTURE_DIRECTORY
        "${CMAKE_CURRENT_BINARY_DIR}/test_vae_c_checkpoint_fixtures"
    )
    file(MAKE_DIRECTORY "${VAE_C_CHECKPOINT_FIXTURE_DIRECTORY}")

    add_test(
        NAME test_vae_c_checkpoint
        COMMAND
            test_vae_c_checkpoint
            ${CMAKE_CURRENT_SOURCE_DIR}/configs/vae_c_profiles_v1.csv
            ${VAE_C_CHECKPOINT_FIXTURE_DIRECTORY}
    )

    set(VAE_C_MU_CONTRACT_FIXTURE_DIRECTORY
        "${CMAKE_CURRENT_BINARY_DIR}/test_vae_c_mu_contract_fixtures"
    )
    file(MAKE_DIRECTORY "${VAE_C_MU_CONTRACT_FIXTURE_DIRECTORY}")

    add_test(
        NAME test_vae_c_mu_contract
        COMMAND
            test_vae_c_mu_contract
            ${PROJECT_SOURCE_DIR}
            ${VAE_C_MU_CONTRACT_FIXTURE_DIRECTORY}
    )
    set_tests_properties(
        test_vae_c_mu_contract
        PROPERTIES
            SKIP_RETURN_CODE 77
    )

    add_test(
        NAME test_vae_c_iic_loss
        COMMAND test_vae_c_iic_loss
    )

    add_test(
        NAME test_vae_c_iic_loss_head_integration
        COMMAND test_vae_c_iic_loss_head_integration
    )

    add_test(
        NAME test_vae_c_iic_head
        COMMAND
            ${CMAKE_COMMAND}
            -DTEST_EXECUTABLE=$<TARGET_FILE:test_vae_c_iic_head>
            -P
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vae_c_iic_head_fresh_process.cmake
    )

    add_test(
        NAME test_vae_c_iic_pair
        COMMAND
            ${CMAKE_COMMAND}
            -DTEST_EXECUTABLE=$<TARGET_FILE:test_vae_c_iic_pair>
            -P
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vae_c_iic_pair_fresh_process.cmake
    )

    add_test(
        NAME test_vae_c_iic_pair_train
        COMMAND
            ${CMAKE_COMMAND}
            -DTEST_EXECUTABLE=$<TARGET_FILE:test_vae_c_iic_pair_train>
            -P
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vae_c_iic_pair_train_fresh_process.cmake
    )

    add_test(
        NAME test_vae_c_iic_ensemble
        COMMAND
            ${CMAKE_COMMAND}
            -DTEST_EXECUTABLE=$<TARGET_FILE:test_vae_c_iic_ensemble>
            -P
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vae_c_iic_ensemble_fresh_process.cmake
    )

    add_test(
        NAME test_vae_c_iic_checkpoint
        COMMAND
            test_vae_c_iic_checkpoint
            ${CMAKE_CURRENT_BINARY_DIR}
    )

    add_test(
        NAME test_vae_c_iic_checkpoint_inspector
        COMMAND ${CMAKE_COMMAND}
            -DTEST_EXECUTABLE=$<TARGET_FILE:inspect_vae_c_iic_checkpoint>
            -DREPOSITORY_ROOT=${PROJECT_SOURCE_DIR}
            -DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vae_c_iic_checkpoint_inspector.cmake
    )

    add_test(
        NAME test_vae_c_iic_h_synthetic
        COMMAND
            ${CMAKE_COMMAND}
            -DTEST_EXECUTABLE=$<TARGET_FILE:test_vae_c_iic_h>
            -DREPOSITORY_ROOT=${PROJECT_SOURCE_DIR}
            -DMODE=synthetic
            -P
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vae_c_iic_h_fresh.cmake
    )
    add_test(
        NAME test_vae_c_iic_h_real_tiny
        COMMAND
            ${CMAKE_COMMAND}
            -DTEST_EXECUTABLE=$<TARGET_FILE:test_vae_c_iic_h>
            -DREPOSITORY_ROOT=${PROJECT_SOURCE_DIR}
            -DMODE=real
            -P
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vae_c_iic_h_fresh.cmake
    )
    set_tests_properties(
        test_vae_c_iic_h_real_tiny PROPERTIES TIMEOUT 300
    )

    add_test(
        NAME test_vae_c_iic_h_producer
        COMMAND
            ${CMAKE_COMMAND}
            -DTEST_EXECUTABLE=$<TARGET_FILE:build_vae_c_iic_h_mini75>
            -DREPOSITORY_ROOT=${PROJECT_SOURCE_DIR}
            -DFIXTURE_PARENT=${CMAKE_CURRENT_BINARY_DIR}
            -P
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vae_c_iic_h_producer.cmake
    )
    set_tests_properties(test_vae_c_iic_h_producer PROPERTIES TIMEOUT 300)

    add_test(
        NAME test_vae_c_iic_h_mini112_real_tiny
        COMMAND ${CMAKE_COMMAND}
            -DTEST_EXECUTABLE=$<TARGET_FILE:test_vae_c_iic_h_mini112>
            -DREPOSITORY_ROOT=${PROJECT_SOURCE_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vae_c_iic_h_mini112_fresh.cmake
    )
    set_tests_properties(test_vae_c_iic_h_mini112_real_tiny PROPERTIES TIMEOUT 300)
    add_test(
        NAME test_vae_c_iic_h_mini112_producer
        COMMAND ${CMAKE_COMMAND}
            -DTEST_EXECUTABLE=$<TARGET_FILE:build_vae_c_iic_h_mini112>
            -DREPOSITORY_ROOT=${PROJECT_SOURCE_DIR}
            -DFIXTURE_PARENT=${CMAKE_CURRENT_BINARY_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vae_c_iic_h_mini112_producer.cmake
    )
    set_tests_properties(test_vae_c_iic_h_mini112_producer PROPERTIES TIMEOUT 600)

    add_test(
        NAME test_vae_c_iic_h_mini224_producer
        COMMAND ${CMAKE_COMMAND}
            -DTEST_EXECUTABLE=$<TARGET_FILE:build_vae_c_iic_h_mini224>
            -DREPOSITORY_ROOT=${PROJECT_SOURCE_DIR}
            -DFIXTURE_PARENT=${CMAKE_CURRENT_BINARY_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vae_c_iic_h_mini224_producer.cmake
    )
    set_tests_properties(test_vae_c_iic_h_mini224_producer PROPERTIES TIMEOUT 300)

    add_test(
        NAME test_vae_c_iic_affinity_synthetic
        COMMAND test_vae_c_iic_affinity
    )
    add_test(
        NAME test_vae_c_iic_affinity_real_tiny
        COMMAND test_vae_c_iic_affinity
            --check-real
            ${PROJECT_SOURCE_DIR}/sakai_multires_c/results/vae_c/iic/mini75_iic_v1_737da555866b822e14c545c4f2687869f41a52cd1340b9e941f3a1ce77763eb9_h_main_v1
    )

    add_test(
        NAME test_vae_c_iic_affinity_mini112_real_tiny
        COMMAND test_vae_c_iic_affinity
            --check-real
            ${PROJECT_SOURCE_DIR}/sakai_multires_c/results/vae_c/iic/mini112_iic_v1_0cecf5cd656d0d61a237d50fcf96a7890fb2e6d052ef403051bc9bc17aed8fba_h_main_v1
    )

    add_test(
        NAME test_vae_c_iic_affinity_producer
        COMMAND test_vae_c_iic_affinity_producer
            $<TARGET_FILE:build_vae_c_iic_affinity_mini75>
            ${CMAKE_CURRENT_BINARY_DIR}
            ${PROJECT_SOURCE_DIR}
    )
    set_tests_properties(test_vae_c_iic_affinity_producer PROPERTIES TIMEOUT 300)

    add_test(
        NAME test_vae_c_iic_affinity_mini112_producer
        COMMAND test_vae_c_iic_affinity_producer
            $<TARGET_FILE:build_vae_c_iic_affinity_mini112>
            ${CMAKE_CURRENT_BINARY_DIR}
            ${PROJECT_SOURCE_DIR}
    )
    set_tests_properties(test_vae_c_iic_affinity_mini112_producer PROPERTIES TIMEOUT 300)

    add_test(
        NAME test_vae_c_iic_affinity_mini224_producer
        COMMAND ${CMAKE_COMMAND}
            -DTEST_EXECUTABLE=$<TARGET_FILE:build_vae_c_iic_affinity_mini224>
            -DREPOSITORY_ROOT=${PROJECT_SOURCE_DIR}
            -DFIXTURE_PARENT=${CMAKE_CURRENT_BINARY_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vae_c_iic_affinity_mini224_producer.cmake
    )
    set_tests_properties(test_vae_c_iic_affinity_mini224_producer PROPERTIES TIMEOUT 300)

    add_test(
        NAME test_vae_c_iic_synthetic
        COMMAND
            ${CMAKE_COMMAND}
            -DTEST_EXECUTABLE=$<TARGET_FILE:run_vae_c_iic_synthetic>
            -DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}
            -P
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vae_c_iic_synthetic.cmake
    )

    add_executable(
        test_vae_c_iic_mini75_central
        tests/test_vae_c_iic_mini75_central.c
    )
    target_link_libraries(
        test_vae_c_iic_mini75_central
        PRIVATE gravityspy_vae_iic_mini75_input
    )
    gravityspy_enable_warnings(test_vae_c_iic_mini75_central)
    add_test(
        NAME test_vae_c_iic_mini75_central
        COMMAND
            ${CMAKE_COMMAND}
            -DTEST_EXECUTABLE=$<TARGET_FILE:test_vae_c_iic_mini75_central>
            -DREPOSITORY_ROOT=${PROJECT_SOURCE_DIR}
            -DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}
            -P
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vae_c_iic_mini75_central_fresh.cmake
    )

    add_executable(
        test_vae_c_iic_mini75_perturbed
        tests/test_vae_c_iic_mini75_perturbed.c
    )
    target_link_libraries(
        test_vae_c_iic_mini75_perturbed
        PRIVATE gravityspy_vae_iic_mini75_input
    )
    gravityspy_enable_warnings(test_vae_c_iic_mini75_perturbed)
    add_test(
        NAME test_vae_c_iic_mini75_perturbed
        COMMAND
            ${CMAKE_COMMAND}
            -DTEST_EXECUTABLE=$<TARGET_FILE:test_vae_c_iic_mini75_perturbed>
            -DREPOSITORY_ROOT=${PROJECT_SOURCE_DIR}
            -P
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vae_c_iic_mini75_perturbed_fresh.cmake
    )
    set_tests_properties(
        test_vae_c_iic_mini75_perturbed
        PROPERTIES TIMEOUT 900
    )

    add_executable(
        test_vae_c_iic_mini112_central
        tests/test_vae_c_iic_mini112_central.c
    )
    add_executable(
        test_vae_c_iic_mini224_central
        tests/test_vae_c_iic_mini224_central.c
    )
    target_link_libraries(
        test_vae_c_iic_mini224_central
        PRIVATE gravityspy_vae_iic_mini224_central
    )
    gravityspy_enable_warnings(test_vae_c_iic_mini224_central)
    add_test(
        NAME test_vae_c_iic_mini224_central
        COMMAND $<TARGET_FILE:test_vae_c_iic_mini224_central>
            ${PROJECT_SOURCE_DIR}
            ${CMAKE_CURRENT_BINARY_DIR}/mini224_central_fixture
    )
    add_executable(
        test_vae_c_iic_mini224_perturbed
        tests/test_vae_c_iic_mini224_perturbed.c
    )
    target_link_libraries(
        test_vae_c_iic_mini224_perturbed
        PRIVATE gravityspy_vae_iic_mini224_perturbed
    )
    gravityspy_enable_warnings(test_vae_c_iic_mini224_perturbed)
    add_test(
        NAME test_vae_c_iic_mini224_perturbed
        COMMAND ${CMAKE_COMMAND}
            -DTEST_EXECUTABLE=$<TARGET_FILE:test_vae_c_iic_mini224_perturbed>
            -DREPOSITORY_ROOT=${PROJECT_SOURCE_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vae_c_iic_mini224_perturbed_fresh.cmake
    )
    set_tests_properties(test_vae_c_iic_mini224_perturbed PROPERTIES TIMEOUT 900)
    target_link_libraries(
        test_vae_c_iic_mini112_central
        PRIVATE gravityspy_vae_iic_mini112_input
    )
    gravityspy_enable_warnings(test_vae_c_iic_mini112_central)
    add_test(
        NAME test_vae_c_iic_mini112_central
        COMMAND ${CMAKE_COMMAND}
            -DTEST_EXECUTABLE=$<TARGET_FILE:test_vae_c_iic_mini112_central>
            -DREPOSITORY_ROOT=${PROJECT_SOURCE_DIR}
            -DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vae_c_iic_mini112_central_fresh.cmake
    )

    add_executable(
        test_vae_c_iic_mini112_perturbed
        tests/test_vae_c_iic_mini112_perturbed.c
    )
    target_link_libraries(
        test_vae_c_iic_mini112_perturbed
        PRIVATE gravityspy_vae_iic_mini112_input
    )
    gravityspy_enable_warnings(test_vae_c_iic_mini112_perturbed)
    add_test(
        NAME test_vae_c_iic_mini112_perturbed
        COMMAND ${CMAKE_COMMAND}
            -DTEST_EXECUTABLE=$<TARGET_FILE:test_vae_c_iic_mini112_perturbed>
            -DREPOSITORY_ROOT=${PROJECT_SOURCE_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vae_c_iic_mini112_perturbed_fresh.cmake
    )
    set_tests_properties(test_vae_c_iic_mini112_perturbed PROPERTIES TIMEOUT 900)

    add_test(
        NAME test_vae_c_iic_mini75_runner
        COMMAND
            ${CMAKE_COMMAND}
            -DTEST_EXECUTABLE=$<TARGET_FILE:run_vae_c_iic_mini75>
            -DREPOSITORY_ROOT=${PROJECT_SOURCE_DIR}
            -DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}
            -P
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vae_c_iic_mini75_runner.cmake
    )
    set_tests_properties(
        test_vae_c_iic_mini75_runner
        PROPERTIES TIMEOUT 900
    )

    add_test(
        NAME test_vae_c_iic_mini112_runner
        COMMAND
            ${CMAKE_COMMAND}
            -DTEST_EXECUTABLE=$<TARGET_FILE:run_vae_c_iic_mini112>
            -DREPOSITORY_ROOT=${PROJECT_SOURCE_DIR}
            -DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}
            -P
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vae_c_iic_mini112_runner.cmake
    )
    set_tests_properties(
        test_vae_c_iic_mini112_runner
        PROPERTIES TIMEOUT 900
    )

    add_test(
        NAME test_vae_c_iic_mini224_runner
        COMMAND
            ${CMAKE_COMMAND}
            -DTEST_EXECUTABLE=$<TARGET_FILE:run_vae_c_iic_mini224>
            -DREPOSITORY_ROOT=${PROJECT_SOURCE_DIR}
            -DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}
            -P
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vae_c_iic_mini224_runner.cmake
    )
    set_tests_properties(
        test_vae_c_iic_mini224_runner
        PROPERTIES TIMEOUT 900
    )

    add_test(
        NAME test_vae_c_iic_mini224_smoke
        COMMAND
            ${CMAKE_COMMAND}
            -DTEST_EXECUTABLE=$<TARGET_FILE:run_vae_c_iic_mini224>
            -DREPOSITORY_ROOT=${PROJECT_SOURCE_DIR}
            -DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}
            -P
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vae_c_iic_mini224_smoke.cmake
    )
    set_tests_properties(
        test_vae_c_iic_mini224_smoke
        PROPERTIES TIMEOUT 3600
    )

    add_executable(
        test_vae_c_iic_mini224_official
        tests/test_vae_c_iic_mini224_official.c
    )
    target_link_libraries(
        test_vae_c_iic_mini224_official
        PRIVATE
            gravityspy_vae_iic_mini224_perturbed
            gravityspy_vae_iic_mini75_input
            gravityspy_vae_iic_checkpoint_core
    )
    gravityspy_enable_warnings(test_vae_c_iic_mini224_official)
    if(WIN32)
        target_link_libraries(test_vae_c_iic_mini224_official PRIVATE psapi)
    endif()
    add_test(
        NAME test_vae_c_iic_mini224_official
        COMMAND test_vae_c_iic_mini224_official
            ${CMAKE_CURRENT_BINARY_DIR}
    )
    add_test(
        NAME test_vae_c_iic_mini224_official_preflight
        COMMAND
            ${CMAKE_COMMAND}
            -DTEST_EXECUTABLE=$<TARGET_FILE:run_vae_c_iic_mini224>
            -DREPOSITORY_ROOT=${PROJECT_SOURCE_DIR}
            -DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}
            -P
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vae_c_iic_mini224_official_preflight.cmake
    )
    set_tests_properties(
        test_vae_c_iic_mini224_official_preflight
        PROPERTIES TIMEOUT 900
    )

    add_test(
        NAME test_vae_c_iic_png_identity
        COMMAND test_vae_c_iic_png_identity
            ${CMAKE_CURRENT_BINARY_DIR}
    )

    add_test(
        NAME test_vae_c_mu_tiny_forward
        COMMAND
            ${CMAKE_COMMAND}
            -DEXTRACTOR=$<TARGET_FILE:extract_vae_c_mu>
            -DREPOSITORY_ROOT=${PROJECT_SOURCE_DIR}
            -P
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vae_c_mu_tiny_forward.cmake
    )
    set_tests_properties(
        test_vae_c_mu_tiny_forward
        PROPERTIES
            SKIP_REGULAR_EXPRESSION "VAE_C_MU_TINY_FORWARD_SKIPPED"
            TIMEOUT 300
    )

    add_test(
        NAME test_baseline_kmeans
        COMMAND test_baseline_kmeans
    )

    add_test(
        NAME test_baseline_clustering_metrics
        COMMAND test_baseline_clustering_metrics
    )

    set(VAE_C_MU_KMEANS_EVALUATOR_FIXTURE_DIRECTORY
        "${CMAKE_CURRENT_BINARY_DIR}/test_vae_c_mu_kmeans_evaluator_fixtures"
    )
    set(VAE_C_MU_KMEANS_EVALUATOR_PRODUCTIVE_FIXTURE_DIRECTORY
        "${CMAKE_CURRENT_BINARY_DIR}/test_vae_c_mu_kmeans_evaluator_productive_fixtures"
    )
    file(MAKE_DIRECTORY
        "${VAE_C_MU_KMEANS_EVALUATOR_FIXTURE_DIRECTORY}"
    )

    add_test(
        NAME test_vae_c_mu_kmeans_evaluator
        COMMAND
            test_vae_c_mu_kmeans_evaluator
            ${VAE_C_MU_KMEANS_EVALUATOR_FIXTURE_DIRECTORY}
    )

    add_test(
        NAME test_vae_c_mu_kmeans_evaluator_validate
        COMMAND
            ${CMAKE_COMMAND}
            -DTEST_EXECUTABLE=$<TARGET_FILE:evaluate_vae_c_mu_kmeans>
            -DREPOSITORY_ROOT=${PROJECT_SOURCE_DIR}
            -DFIXTURE_ROOT=${VAE_C_MU_KMEANS_EVALUATOR_PRODUCTIVE_FIXTURE_DIRECTORY}
            -P
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vae_c_mu_kmeans_evaluator_validate.cmake
    )
    set_tests_properties(
        test_vae_c_mu_kmeans_evaluator_validate
        PROPERTIES
            TIMEOUT 300
    )

    add_test(
        NAME test_vae_c_iic_spectral_evaluator_tiny
        COMMAND evaluate_vae_c_iic_spectral_mini75 --self-test
    )
    add_test(
        NAME test_vae_c_iic_spectral_evaluator_validate
        COMMAND evaluate_vae_c_iic_spectral_mini75
            --validate-only --repository-root ${PROJECT_SOURCE_DIR}
    )
    set_tests_properties(
        test_vae_c_iic_spectral_evaluator_validate
        PROPERTIES TIMEOUT 300
    )

    add_test(
        NAME test_vae_c_iic_spectral_common
        COMMAND
            ${CMAKE_COMMAND}
            -DEVALUATOR=$<TARGET_FILE:evaluate_vae_c_iic_spectral_common>
            -DREPOSITORY_ROOT=${PROJECT_SOURCE_DIR}
            -P
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vae_c_iic_spectral_common.cmake
    )

    add_test(
        NAME test_vae_c_iic_contact_sheets_common
        COMMAND
            ${CMAKE_COMMAND}
            -DPRODUCER=$<TARGET_FILE:build_vae_c_iic_contact_sheets_common>
            -DREPOSITORY_ROOT=${PROJECT_SOURCE_DIR}
            -DSCRATCH_ROOT=${CMAKE_CURRENT_BINARY_DIR}/contact_sheet_common_test
            -P
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vae_c_iic_contact_sheets_common.cmake
    )
    set_tests_properties(
        test_vae_c_iic_contact_sheets_common
        PROPERTIES TIMEOUT 600
    )

    set(VAE_C_MU_KMEANS_VALIDATE_FIXTURE_DIRECTORY
        "${CMAKE_CURRENT_BINARY_DIR}/test_vae_c_mu_kmeans_validate_fixtures"
    )
    add_test(
        NAME test_vae_c_mu_kmeans_validate
        COMMAND
            ${CMAKE_COMMAND}
            -DTEST_EXECUTABLE=$<TARGET_FILE:test_vae_c_mu_kmeans_validate>
            -DREPOSITORY_ROOT=${PROJECT_SOURCE_DIR}
            -DFIXTURE_ROOT=${VAE_C_MU_KMEANS_VALIDATE_FIXTURE_DIRECTORY}
            -P
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vae_c_mu_kmeans_productive.cmake
    )

    set(IIC_SPECTRAL_REFERENCE_PYTHON "" CACHE FILEPATH
        "Explicit MINI_SPECTRAL_REFERENCE_ENV_V1 python.exe for non-full bridge tests")
    if(IIC_SPECTRAL_REFERENCE_PYTHON)
        if(NOT EXISTS "${IIC_SPECTRAL_REFERENCE_PYTHON}")
            message(FATAL_ERROR "IIC spectral reference Python does not exist")
        endif()
        add_test(
            NAME test_vae_c_iic_spectral_bridge
            COMMAND
                "${IIC_SPECTRAL_REFERENCE_PYTHON}" -B
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vae_c_iic_spectral_bridge.py"
                --producer "${CMAKE_CURRENT_SOURCE_DIR}/tools/run_vae_c_iic_spectral_mini75.py"
                --fixture-root "${CMAKE_CURRENT_BINARY_DIR}/spectral_bridge_fixture"
                --repo-root "${PROJECT_SOURCE_DIR}"
        )
        set_tests_properties(test_vae_c_iic_spectral_bridge PROPERTIES
            TIMEOUT 900
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        )
        add_test(
            NAME test_vae_c_iic_spectral_bridge_mini112
            COMMAND
                "${IIC_SPECTRAL_REFERENCE_PYTHON}" -B
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vae_c_iic_spectral_bridge_mini112.py"
                --producer "${CMAKE_CURRENT_SOURCE_DIR}/tools/run_vae_c_iic_spectral_mini112.py"
                --mini75-producer "${CMAKE_CURRENT_SOURCE_DIR}/tools/run_vae_c_iic_spectral_mini75.py"
                --fixture-root "${CMAKE_CURRENT_BINARY_DIR}/spectral_bridge_mini112_fixture"
                --repo-root "${PROJECT_SOURCE_DIR}"
        )
        set_tests_properties(test_vae_c_iic_spectral_bridge_mini112 PROPERTIES
            TIMEOUT 900
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        )
        add_test(
            NAME test_vae_c_iic_spectral_bridge_mini224
            COMMAND
                "${IIC_SPECTRAL_REFERENCE_PYTHON}" -B
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vae_c_iic_spectral_bridge_mini224.py"
                --producer "${CMAKE_CURRENT_SOURCE_DIR}/tools/run_vae_c_iic_spectral_mini224.py"
                --fixture-root "${CMAKE_CURRENT_BINARY_DIR}/spectral_bridge_mini224_fixture"
                --repo-root "${PROJECT_SOURCE_DIR}"
        )
        set_tests_properties(test_vae_c_iic_spectral_bridge_mini224 PROPERTIES
            TIMEOUT 900
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        )
    endif()

    set_tests_properties(
        test_tensor_manifest_io
        test_resize_equivalence
        test_multires_event_loader
        test_multires_batch_legacy
        test_multires_batch_symmetric
        test_multires_stateless_augmentation
        test_multires_epoch_quick
        test_subset224_epoch
        test_vae_c_config
        test_vae_c_profiles_v1_sha256
        test_vae_c_profiles_v2
        test_vae_c_rng_init
        test_vae_c_shuffle
        test_vae_c_sha256
        test_vae_c_identity
        test_vae_c_mini224_runner
        test_vae_c_tensor_parameter
        test_vae_c_basic_ops
        test_vae_c_spatial_ops
        test_vae_c_batchnorm
        test_vae_c_conv2d
        test_vae_c_latent
        test_vae_c_bce
        test_vae_c_kl
        test_vae_c_vae_loss
        test_vae_c_adam
        test_vae_c_model_allocation
        test_vae_c_model_initialization
        test_vae_c_model_workspace
        test_vae_c_model_encoder_forward
        test_vae_c_model_decoder_forward
        test_vae_c_model_vae_forward
        test_vae_c_model_vae_backward
        test_vae_c_model_train
        test_vae_c_checkpoint
        test_vae_c_mu_contract
        test_vae_c_iic_loss
        test_vae_c_iic_loss_head_integration
        test_vae_c_iic_head
        test_vae_c_iic_affinity_synthetic
        test_vae_c_iic_affinity_real_tiny
        test_vae_c_iic_pair
        test_vae_c_iic_pair_train
        test_vae_c_iic_ensemble
        test_vae_c_mu_tiny_forward
        test_baseline_kmeans
        test_baseline_clustering_metrics
        test_vae_c_mu_kmeans_evaluator
        test_vae_c_mu_kmeans_evaluator_validate
        test_vae_c_iic_spectral_evaluator_tiny
        test_vae_c_iic_spectral_evaluator_validate
        test_vae_c_mu_kmeans_validate
        PROPERTIES
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    )
endif()
