if(NOT DEFINED TEST_EXECUTABLE OR NOT DEFINED REPOSITORY_ROOT OR
   NOT DEFINED FIXTURE_PARENT)
    message(FATAL_ERROR "mini224 H producer test arguments missing")
endif()

file(SHA256 "${TEST_EXECUTABLE}" producer_sha)
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef suffix)
set(root "${FIXTURE_PARENT}/h224_${suffix}")
set(first "${root}/chunk128")
set(second "${root}/chunk64")
set(validate "${root}/validate_only")
file(MAKE_DIRECTORY "${root}")
set(commit "0000000000000000000000000000000000000000")

function(invoke output chunk mode)
    set(arguments
        --repository-root "${REPOSITORY_ROOT}"
        --output-dir "${output}"
        --chunk-size "${chunk}"
        --producer-commit "${commit}"
        --producer-binary-sha256 "${producer_sha}")
    if(mode STREQUAL "validate")
        list(APPEND arguments --validate-only)
    endif()
    execute_process(
        COMMAND "${TEST_EXECUTABLE}" ${arguments}
        RESULT_VARIABLE result
        OUTPUT_VARIABLE stdout
        ERROR_VARIABLE stderr)
    if(mode STREQUAL "reject")
        if(result EQUAL 0)
            message(FATAL_ERROR "mini224 H producer accepted overwrite")
        endif()
    else()
        if(NOT result EQUAL 0 OR NOT stderr STREQUAL "")
            message(FATAL_ERROR "mini224 H producer failed: ${result} ${stderr}")
        endif()
        if(mode STREQUAL "validate")
            if(NOT stdout MATCHES "H_INPUT_VALIDATED=true" OR
               NOT stdout MATCHES "H_FORWARD_CALLS=0" OR
               NOT stdout MATCHES "OUTPUT_CREATED=false")
                message(FATAL_ERROR "mini224 H validate-only gate missing")
            endif()
        elseif(NOT stdout MATCHES "H_CREATED=true" OR
               NOT stdout MATCHES "H_ROWS=1000" OR
               NOT stdout MATCHES "H_COLS=180" OR
               NOT stdout MATCHES "H_PAYLOAD_BYTES=720000")
            message(FATAL_ERROR "mini224 H producer output contract mismatch")
        endif()
    endif()
endfunction()

invoke("${validate}" 128 validate)
if(EXISTS "${validate}" OR EXISTS "${validate}.partial")
    message(FATAL_ERROR "mini224 H validate-only created output")
endif()

execute_process(
    COMMAND "${TEST_EXECUTABLE}"
        --repository-root "${REPOSITORY_ROOT}"
        --output-dir "${root}/bad_binary_identity"
        --chunk-size 128
        --producer-commit "${commit}"
        --producer-binary-sha256
            0000000000000000000000000000000000000000000000000000000000000000
        --validate-only
    RESULT_VARIABLE bad_result)
if(bad_result EQUAL 0 OR EXISTS "${root}/bad_binary_identity")
    message(FATAL_ERROR "mini224 H producer accepted wrong binary SHA")
endif()

execute_process(
    COMMAND "${TEST_EXECUTABLE}"
        --repository-root "${REPOSITORY_ROOT}"
        --output-dir "${root}/bad_chunk"
        --chunk-size 129
        --producer-commit "${commit}"
        --producer-binary-sha256 "${producer_sha}"
        --validate-only
    RESULT_VARIABLE bad_chunk_result)
if(bad_chunk_result EQUAL 0 OR EXISTS "${root}/bad_chunk")
    message(FATAL_ERROR "mini224 H producer accepted invalid chunk")
endif()

# A scratch mirror exercises the producer's FINAL-only SHA gate without
# changing the canonical checkpoint or any other repository asset.
set(run_id "mini224_iic_v1_4236df3f9e82f0528fcea947b2e15e42fab8159c0d4b1ed03189e4a85daae8a5")
set(embedding_id
    "mini224_bridge_v1_subset1000_best_epoch99_posterior_mu_central_v1")
set(mirror "${root}/mirror")
set(source_paths
    "sakai_multires_c/results/vae_c/iic/${run_id}/final.ckpt"
    "sakai_multires_c/results/vae_c/iic/${run_id}/run_config.json"
    "sakai_multires_c/results/vae_c/embeddings/${embedding_id}/embedding_spec.txt"
    "sakai_multires_c/results/vae_c/embeddings/${embedding_id}/train_mu_f32.bin"
    "sakai_multires_c/results/vae_c/embeddings/${embedding_id}/train_rows.csv"
    "sakai_multires_c/results/vae_c/embeddings/${embedding_id}/validation_mu_f32.bin"
    "sakai_multires_c/results/vae_c/embeddings/${embedding_id}/validation_rows.csv"
    "sakai_multires_c/results/vae_c/embeddings/${embedding_id}/sha256.txt"
    "manifest_events_272.csv"
    "manifest_tensors_224_central.csv"
    "split_train_indices_seed123.csv"
    "split_val_indices_seed123.csv"
    "sakai_multires_c/configs/subset224_train_800_seed123.csv"
    "sakai_multires_c/configs/subset224_val_200_seed123.csv"
    "sakai_multires_c/configs/vae_c_profiles_v2.csv"
    "sakai_multires_c/results/vae_c/training/mini224_bridge_v1_subset1000_candidate_init123_latent456_aug789_shuffle101112_bs16_max100_pat20/run_config.txt"
    "sakai_multires_c/results/vae_c/training/mini224_bridge_v1_subset1000_candidate_init123_latent456_aug789_shuffle101112_bs16_max100_pat20/best.ckpt")
foreach(relative IN LISTS source_paths)
    set(destination "${mirror}/${relative}")
    get_filename_component(parent "${destination}" DIRECTORY)
    file(MAKE_DIRECTORY "${parent}")
    file(COPY_FILE "${REPOSITORY_ROOT}/${relative}" "${destination}")
endforeach()
set(mirror_final "${mirror}/sakai_multires_c/results/vae_c/iic/${run_id}/final.ckpt")
execute_process(
    COMMAND "${TEST_EXECUTABLE}"
        --repository-root "${mirror}"
        --output-dir "${root}/mirror_validate"
        --chunk-size 128
        --producer-commit "${commit}"
        --producer-binary-sha256 "${producer_sha}"
        --validate-only
    RESULT_VARIABLE mirror_result
    OUTPUT_VARIABLE mirror_stdout
    ERROR_VARIABLE mirror_stderr)
if(NOT mirror_result EQUAL 0 OR
   NOT mirror_stdout MATCHES "H_INPUT_VALIDATED=true" OR
   EXISTS "${root}/mirror_validate")
    message(FATAL_ERROR "mini224 H scratch mirror baseline invalid: ${mirror_stderr}")
endif()
file(COPY_FILE
    "${REPOSITORY_ROOT}/sakai_multires_c/results/vae_c/iic/${run_id}/latest.ckpt"
    "${mirror_final}")
execute_process(
    COMMAND "${TEST_EXECUTABLE}"
        --repository-root "${mirror}"
        --output-dir "${root}/latest_as_final"
        --chunk-size 128
        --producer-commit "${commit}"
        --producer-binary-sha256 "${producer_sha}"
        --validate-only
    RESULT_VARIABLE latest_result)
if(latest_result EQUAL 0 OR EXISTS "${root}/latest_as_final")
    message(FATAL_ERROR "mini224 H producer accepted LATEST as FINAL")
endif()
file(COPY_FILE
    "${REPOSITORY_ROOT}/sakai_multires_c/results/vae_c/iic/${run_id}/final.ckpt"
    "${mirror_final}")
file(APPEND "${mirror_final}" "x")
execute_process(
    COMMAND "${TEST_EXECUTABLE}"
        --repository-root "${mirror}"
        --output-dir "${root}/bad_final_sha"
        --chunk-size 128
        --producer-commit "${commit}"
        --producer-binary-sha256 "${producer_sha}"
        --validate-only
    RESULT_VARIABLE bad_final_result)
if(bad_final_result EQUAL 0 OR EXISTS "${root}/bad_final_sha")
    message(FATAL_ERROR "mini224 H producer accepted wrong FINAL SHA")
endif()
file(COPY_FILE
    "${REPOSITORY_ROOT}/sakai_multires_c/results/vae_c/iic/${run_id}/final.ckpt"
    "${mirror_final}")

function(reject_mutated_source relative case_name)
    set(path "${mirror}/${relative}")
    file(APPEND "${path}" "x")
    execute_process(
        COMMAND "${TEST_EXECUTABLE}"
            --repository-root "${mirror}"
            --output-dir "${root}/${case_name}"
            --chunk-size 128
            --producer-commit "${commit}"
            --producer-binary-sha256 "${producer_sha}"
            --validate-only
        RESULT_VARIABLE mutation_result
        ERROR_VARIABLE mutation_stderr)
    if(mutation_result EQUAL 0 OR EXISTS "${root}/${case_name}" OR
       NOT mutation_stderr MATCHES "H central mu identity/load failed")
        message(FATAL_ERROR
            "mini224 H accepted ${case_name} or failed for unrelated reason: ${mutation_stderr}")
    endif()
    file(COPY_FILE "${REPOSITORY_ROOT}/${relative}" "${path}")
endfunction()
foreach(name IN ITEMS train_mu_f32.bin validation_mu_f32.bin
                      train_rows.csv validation_rows.csv)
    reject_mutated_source(
        "sakai_multires_c/results/vae_c/embeddings/${embedding_id}/${name}"
        "bad_${name}")
endforeach()
set(population_relative
    "sakai_multires_c/configs/subset224_train_800_seed123.csv")
file(STRINGS "${mirror}/${population_relative}" population_lines)
list(POP_BACK population_lines)
string(JOIN "\n" population_body ${population_lines})
file(WRITE "${mirror}/${population_relative}" "${population_body}\n")
execute_process(
    COMMAND "${TEST_EXECUTABLE}"
        --repository-root "${mirror}"
        --output-dir "${root}/bad_population"
        --chunk-size 128
        --producer-commit "${commit}"
        --producer-binary-sha256 "${producer_sha}"
        --validate-only
    RESULT_VARIABLE population_result
    ERROR_VARIABLE population_stderr)
if(population_result EQUAL 0 OR EXISTS "${root}/bad_population" OR
   NOT population_stderr MATCHES "H central mu identity/load failed")
    message(FATAL_ERROR "mini224 H accepted train population 799: ${population_stderr}")
endif()
file(COPY_FILE "${REPOSITORY_ROOT}/${population_relative}"
               "${mirror}/${population_relative}")

set(spec_relative
    "sakai_multires_c/results/vae_c/embeddings/${embedding_id}/embedding_spec.txt")
file(READ "${mirror}/${spec_relative}" spec_text)
string(REPLACE "latent_dim=64" "latent_dim=32" bad_spec "${spec_text}")
if(spec_text STREQUAL bad_spec)
    message(FATAL_ERROR "mini224 H latent fixture key missing")
endif()
file(WRITE "${mirror}/${spec_relative}" "${bad_spec}")
execute_process(
    COMMAND "${TEST_EXECUTABLE}"
        --repository-root "${mirror}"
        --output-dir "${root}/bad_latent_spec"
        --chunk-size 128
        --producer-commit "${commit}"
        --producer-binary-sha256 "${producer_sha}"
        --validate-only
    RESULT_VARIABLE latent_result
    ERROR_VARIABLE latent_stderr)
if(latent_result EQUAL 0 OR EXISTS "${root}/bad_latent_spec" OR
   NOT latent_stderr MATCHES "H central mu identity/load failed")
    message(FATAL_ERROR "mini224 H accepted latent32: ${latent_stderr}")
endif()

invoke("${first}" 128 produce)
invoke("${second}" 64 produce)
foreach(output IN ITEMS "${first}" "${second}")
    if(EXISTS "${output}.partial" OR EXISTS "${output}.tmp")
        message(FATAL_ERROR "mini224 H partial/tmp remains")
    endif()
    file(GLOB files LIST_DIRECTORIES FALSE "${output}/*")
    list(LENGTH files count)
    if(NOT count EQUAL 4)
        message(FATAL_ERROR "mini224 H artifact inventory mismatch: ${count}")
    endif()
    file(SIZE "${output}/h_f32.bin" bytes)
    if(NOT bytes EQUAL 720000)
        message(FATAL_ERROR "mini224 H payload size mismatch: ${bytes}")
    endif()
    file(STRINGS "${output}/rows.csv" rows)
    list(LENGTH rows row_lines)
    if(NOT row_lines EQUAL 1001)
        message(FATAL_ERROR "mini224 H row count mismatch: ${row_lines}")
    endif()
    list(GET rows 0 header)
    if(NOT header STREQUAL
        "manifest_index,event_index,event_id,source_split,split_pos")
        message(FATAL_ERROR "mini224 H row header mismatch")
    endif()
    foreach(index RANGE 1 800)
        list(GET rows ${index} row)
        math(EXPR position "${index} - 1")
        if(NOT row MATCHES ",train,${position}$")
            message(FATAL_ERROR "mini224 H train row order mismatch: ${index}")
        endif()
    endforeach()
    foreach(index RANGE 801 1000)
        list(GET rows ${index} row)
        math(EXPR position "${index} - 801")
        if(NOT row MATCHES ",validation,${position}$")
            message(FATAL_ERROR "mini224 H validation row order mismatch: ${index}")
        endif()
    endforeach()
    file(READ "${output}/h_spec.txt" spec)
    if(NOT spec MATCHES "row_order=train800_then_validation200_native_split" OR
       NOT spec MATCHES "iic_checkpoint_kind=FINAL" OR
       NOT spec MATCHES "input_latent_dim=64" OR
       NOT spec MATCHES "main_only=true" OR
       NOT spec MATCHES "overcluster_used=false" OR
       NOT spec MATCHES "label_free=true")
        message(FATAL_ERROR "mini224 H metadata contract mismatch")
    endif()
endforeach()

foreach(name IN ITEMS h_f32.bin rows.csv h_spec.txt sha256.txt)
    file(SHA256 "${first}/${name}" first_sha)
    file(SHA256 "${second}/${name}" second_sha)
    if(NOT first_sha STREQUAL second_sha)
        message(FATAL_ERROR "mini224 H chunk-size dependence: ${name}")
    endif()
endforeach()

file(SHA256 "${first}/h_f32.bin" before_overwrite)
invoke("${first}" 128 reject)
file(SHA256 "${first}/h_f32.bin" after_overwrite)
if(NOT before_overwrite STREQUAL after_overwrite)
    message(FATAL_ERROR "mini224 H overwrite rejection changed payload")
endif()
message(STATUS "mini224 H producer validate-only, full1000, bitwise and overwrite PASS")
