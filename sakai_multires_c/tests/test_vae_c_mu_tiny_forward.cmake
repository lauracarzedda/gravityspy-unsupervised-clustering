cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED EXTRACTOR OR NOT DEFINED REPOSITORY_ROOT)
    message(FATAL_ERROR "EXTRACTOR and REPOSITORY_ROOT are required")
endif()

file(READ
    "${REPOSITORY_ROOT}/sakai_multires_c/tools/extract_vae_c_mu.c"
    extractor_source
)
set(forbidden_call_tokens
    "vae_c_model_decoder_forward"
    "vae_c_model_vae_forward"
    "vae_c_reparameterize_forward"
    "_backward"
    "vae_c_adam_step"
)
foreach(forbidden_call IN LISTS forbidden_call_tokens)
    string(FIND "${extractor_source}" "${forbidden_call}" position)
    if(NOT position EQUAL -1)
        message(FATAL_ERROR
            "forbidden tiny-forward call token present: ${forbidden_call}"
        )
    endif()
endforeach()

set(canonical_checkpoints
    "${REPOSITORY_ROOT}/sakai_multires_c/results/vae_c/training/mini75_experiment_v1_full_init123_latent456_aug789_shuffle101112_bs64/best.ckpt"
    "${REPOSITORY_ROOT}/sakai_multires_c/results/vae_c/training/mini112_primary_v1_full_init123_latent456_aug789_shuffle101112_bs64/best.ckpt"
    "${REPOSITORY_ROOT}/sakai_multires_c/results/vae_c/training/mini224_bridge_v1_subset1000_candidate_init123_latent456_aug789_shuffle101112_bs16_max100_pat20/best.ckpt"
)
set(checkpoint_count 0)
foreach(checkpoint IN LISTS canonical_checkpoints)
    if(EXISTS "${checkpoint}")
        math(EXPR checkpoint_count "${checkpoint_count} + 1")
    endif()
endforeach()
list(LENGTH canonical_checkpoints expected_checkpoint_count)
if(checkpoint_count EQUAL 0)
    message("VAE_C_MU_TINY_FORWARD_SKIPPED: canonical checkpoints absent")
    return()
endif()
if(NOT checkpoint_count EQUAL expected_checkpoint_count)
    message(FATAL_ERROR
        "partial canonical checkpoint set: "
        "${checkpoint_count}/${expected_checkpoint_count} present"
    )
endif()


function(run_tiny run_name batch_size output_name)
    execute_process(
        COMMAND
            "${EXTRACTOR}"
            --tiny-forward
            --run "${run_name}"
            --batch-size "${batch_size}"
            "${REPOSITORY_ROOT}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
    )
    if(NOT result EQUAL 0)
        message(FATAL_ERROR
            "tiny forward failed for ${run_name}/batch${batch_size}: "
            "exit=${result}\nstdout=${output}\nstderr=${error}"
        )
    endif()
    if(NOT "${error}" STREQUAL "")
        message(FATAL_ERROR
            "unexpected stderr for ${run_name}/batch${batch_size}: ${error}"
        )
    endif()
    set(${output_name} "${output}" PARENT_SCOPE)
endfunction()


function(require_line output expected context)
    string(REPLACE "\r" "" normalized "${output}")
    string(REPLACE "\n" ";" lines "${normalized}")
    list(FIND lines "${expected}" position)
    if(position EQUAL -1)
        message(FATAL_ERROR "missing line ${expected} in ${context}\n${output}")
    endif()
endfunction()


function(require_line_prefix output expected_prefix context)
    string(REPLACE "\r" "" normalized "${output}")
    string(REPLACE "\n" ";" lines "${normalized}")
    set(found FALSE)
    foreach(line IN LISTS lines)
        string(FIND "${line}" "${expected_prefix}" position)
        if(position EQUAL 0)
            string(LENGTH "${line}" line_length)
            string(LENGTH "${expected_prefix}" prefix_length)
            math(EXPR suffix_length "${line_length} - ${prefix_length}")
            if(suffix_length EQUAL 64)
                string(SUBSTRING "${line}" ${prefix_length} 64 suffix)
                if("${suffix}" MATCHES "^[0-9a-f]+$")
                    set(found TRUE)
                endif()
            endif()
        endif()
    endforeach()
    if(NOT found)
        message(FATAL_ERROR
            "missing line prefix ${expected_prefix} in ${context}\n${output}"
        )
    endif()
endfunction()


function(extract_sha output key result_name context)
    string(REGEX MATCH "${key}=([0-9a-f]+)" match "${output}")
    if(NOT match)
        message(FATAL_ERROR "missing ${key} in ${context}\n${output}")
    endif()
    string(LENGTH "${CMAKE_MATCH_1}" digest_length)
    if(NOT digest_length EQUAL 64)
        message(FATAL_ERROR "invalid ${key} in ${context}: ${CMAKE_MATCH_1}")
    endif()
    set(${result_name} "${CMAKE_MATCH_1}" PARENT_SCOPE)
endfunction()


set(required_tokens
    "TINY_FORWARD=true"
    "VIEW=central"
    "REPRESENTATION=posterior_mu"
    "CHECKPOINT_KIND=BEST"
    "TRAIN_ROWS=2"
    "VALIDATION_ROWS=2"
    "TOTAL_ROWS=4"
    "WORKSPACE_MU_REAL=true"
    "MU_FINITE=true"
    "LOGVAR_FINITE=true"
    "ROW_ORDER_NATIVE_SPLIT_POS=true"
    "EVENT_IDENTITY_VERIFIED=true"
    "INPUT_UNCHANGED=true"
    "MODEL_PARAMETER_STORAGE_UNCHANGED=true"
    "BATCH_NORM_RUNNING_STATS_UNCHANGED=true"
    "OPTIMIZER_STATE_UNCHANGED=true"
    "CHECKPOINT_UNCHANGED=true"
    "PREFLIGHT_ZERO_ACTIVITY=true"
    "DIRECT_ENCODER_ONLY_PATH=true"
    "DECODER_EXECUTED=false"
    "DECODER_WORKSPACE_UNCHANGED_ZERO=true"
    "EPSILON_USED=false"
    "LATENT_SAMPLED=false"
    "BACKWARD_EXECUTED=false"
    "ADAM_STEP_EXECUTED=false"
    "LABEL_DEPENDENT_SELECTIONS=0"
    "LABEL_VALUES_EXPORTED=0"
    "LABELS_USED=false"
    "OUTPUT_CREATED=false"
    "ARTIFACT_FILES_CREATED=0"
    "TINY_FORWARD_PASSED=true"
)

foreach(run_name IN ITEMS mini75 mini112 mini224)
    run_tiny("${run_name}" 1 batch1_first)
    run_tiny("${run_name}" 1 batch1_second)
    run_tiny("${run_name}" 2 batch2)

    foreach(output_name IN ITEMS batch1_first batch1_second)
        require_line(
            "${${output_name}}" "BATCH_SIZE=1"
            "${run_name}/${output_name}/batch-size"
        )
        require_line(
            "${${output_name}}" "ENCODER_FORWARD_CALLS=4"
            "${run_name}/${output_name}/forward-count"
        )
    endforeach()
    require_line(
        "${batch2}" "BATCH_SIZE=2"
        "${run_name}/batch2/batch-size"
    )
    require_line(
        "${batch2}" "ENCODER_FORWARD_CALLS=2"
        "${run_name}/batch2/forward-count"
    )

    if(NOT "${batch1_first}" STREQUAL "${batch1_second}")
        message(FATAL_ERROR
            "fresh-process determinism mismatch for ${run_name}"
        )
    endif()

    foreach(output_name IN ITEMS batch1_first batch1_second batch2)
        foreach(token IN LISTS required_tokens)
            require_line(
                "${${output_name}}" "${token}"
                "${run_name}/${output_name}"
            )
        endforeach()
    endforeach()

    if(run_name STREQUAL "mini75")
        set(expected_contract
            "RUN=mini75"
            "PROFILE=mini75_experiment_v1"
            "LATENT_DIM=32"
            "CHECKPOINT_EPOCH=47"
            "CHECKPOINT_OPTIMIZER_STEP=5184"
            "CHECKPOINT_SHA256=7926f3a2557d7b02a9b429627c622e5c11559c1d5a273d2e8129a68e5c7ab121"
            "INPUT_CANONICAL_SHA256=5435065d72b3df3cfffec1d088410cb44afeff9914f9bd10c69e22a980ebda8a"
        )
    elseif(run_name STREQUAL "mini112")
        set(expected_contract
            "RUN=mini112"
            "PROFILE=mini112_primary_v1"
            "LATENT_DIM=64"
            "CHECKPOINT_EPOCH=97"
            "CHECKPOINT_OPTIMIZER_STEP=10584"
            "CHECKPOINT_SHA256=8857090fb1477fafed6b841bb880a06d61acf38899c7ad2e5a9ef60672989a8b"
            "INPUT_CANONICAL_SHA256=ea42cac2c1e1d4d67ec57ed2d9fe3e4f05fb14291806b44597d0c0ed2d31b772"
        )
    else()
        set(expected_contract
            "RUN=mini224"
            "PROFILE=mini224_bridge_v1"
            "LATENT_DIM=64"
            "CHECKPOINT_EPOCH=99"
            "CHECKPOINT_OPTIMIZER_STEP=5000"
            "CHECKPOINT_SHA256=15a1ecd6dd3c6d11aa0a2ac20ea0b6176d08a18c6f582e7349ed6784ab8528f3"
            "INPUT_CANONICAL_SHA256=dc4dc9523719be045f27d137a8337e783f593776862121694342f28d35f9ee29"
        )
    endif()
    foreach(output_name IN ITEMS batch1_first batch1_second batch2)
        foreach(contract_line IN LISTS expected_contract)
            require_line(
                "${${output_name}}" "${contract_line}"
                "${run_name}/${output_name}/canonical-contract"
            )
        endforeach()
    endforeach()

    if(run_name STREQUAL "mini224")
        set(expected_rows
            "ROW_0=train,0,7663,7664,H1_BqniZ6oLFl,"
            "ROW_1=train,799,3677,3678,H1_pYroN7xRPQ,"
            "ROW_2=validation,0,6864,6865,H1_4D7IcowE8a,"
            "ROW_3=validation,199,3392,3393,H1_A4slDbpvFE,"
        )
    else()
        set(expected_rows
            "ROW_0=train,0,7663,7664,H1_BqniZ6oLFl,"
            "ROW_1=train,6865,4194,4195,H1_8SWF6rwp4A,"
            "ROW_2=validation,0,6864,6865,H1_4D7IcowE8a,"
            "ROW_3=validation,1716,440,441,L1_JEt0E5vTv7,"
        )
    endif()
    foreach(output_name IN ITEMS batch1_first batch1_second batch2)
        foreach(expected_row IN LISTS expected_rows)
            require_line_prefix(
                "${${output_name}}" "${expected_row}"
                "${run_name}/${output_name}/golden-row-identity"
            )
        endforeach()
    endforeach()

    foreach(key IN ITEMS
            INPUT_CANONICAL_SHA256
            ROW_IDENTITY_SHA256
            MU_CANONICAL_SHA256)
        extract_sha(
            "${batch1_first}" "${key}" batch1_sha
            "${run_name}/batch1"
        )
        extract_sha(
            "${batch2}" "${key}" batch2_sha
            "${run_name}/batch2"
        )
        if(NOT "${batch1_sha}" STREQUAL "${batch2_sha}")
            message(FATAL_ERROR
                "batch-size independence mismatch for ${run_name}/${key}: "
                "${batch1_sha} != ${batch2_sha}"
            )
        endif()
    endforeach()

    message(STATUS
        "[PASS] ${run_name}: fresh-process deterministic, batch1=batch2"
    )
endforeach()

message(STATUS "test_vae_c_mu_tiny_forward: PASS")
