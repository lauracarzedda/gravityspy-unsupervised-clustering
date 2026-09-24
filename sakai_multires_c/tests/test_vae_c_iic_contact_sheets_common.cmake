if(NOT DEFINED PRODUCER)
    message(FATAL_ERROR "PRODUCER is required")
endif()
if(NOT DEFINED REPOSITORY_ROOT)
    message(FATAL_ERROR "REPOSITORY_ROOT is required")
endif()
if(NOT DEFINED SCRATCH_ROOT)
    message(FATAL_ERROR "SCRATCH_ROOT is required")
endif()

file(REMOVE_RECURSE "${SCRATCH_ROOT}")
file(MAKE_DIRECTORY "${SCRATCH_ROOT}")

foreach(run_id IN ITEMS a b)
    execute_process(
        COMMAND "${PRODUCER}" --self-test
            --scratch-root "${SCRATCH_ROOT}/self_${run_id}"
        RESULT_VARIABLE self_result
        OUTPUT_VARIABLE self_output
        ERROR_VARIABLE self_error
    )
    if(NOT self_result EQUAL 0 OR
       NOT self_output MATCHES "CONTACT_SHEET_SELF_TEST_PASS=true")
        message(FATAL_ERROR "contact-sheet self-test failed: ${self_error}")
    endif()
endforeach()

set(valid_commit "0123456789abcdef0123456789abcdef01234567")
set(valid_sha "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")

foreach(profile IN ITEMS mini75_primary mini112_primary mini224_1000_primary)
    if(profile STREQUAL "mini75_primary")
        set(official_relative
            "sakai_multires_c/results/vae_c/iic/common_contact_sheets_v1/mini75_primary")
    elseif(profile STREQUAL "mini112_primary")
        set(official_relative
            "sakai_multires_c/results/vae_c/iic/common_contact_sheets_v1/mini112_primary")
    else()
        set(official_relative
            "sakai_multires_c/results/vae_c/iic/common_contact_sheets_v1/mini224_1000_primary")
    endif()
    set(expected_official "${REPOSITORY_ROOT}/${official_relative}")
    string(REPLACE "\\" "/" expected_official "${expected_official}")
    execute_process(
        COMMAND "${PRODUCER}" --validate-only
            --profile "${profile}"
            --repository-root "${REPOSITORY_ROOT}"
        WORKING_DIRECTORY "${REPOSITORY_ROOT}"
        RESULT_VARIABLE validate_result
        OUTPUT_VARIABLE validate_output
        ERROR_VARIABLE validate_error
    )
    string(REPLACE "\\" "/" validate_output "${validate_output}")
    string(FIND "${validate_output}"
        "OFFICIAL_OUTPUT_DIR=${expected_official}" official_path_position)
    if(NOT validate_result EQUAL 0 OR
       NOT validate_output MATCHES "VALIDATE_ONLY_PASS=true" OR
       NOT validate_output MATCHES "REPRESENTATIVES=180" OR
       NOT validate_output MATCHES "OUTPUT_CREATED=false" OR
       NOT validate_output MATCHES
           "SPECTRAL_EMBEDDING_USED_FOR_RANKING=false" OR
       NOT validate_output MATCHES "LABELS_USED_FOR_RANKING=false" OR
       official_path_position EQUAL -1 OR
       EXISTS "${expected_official}" OR
       EXISTS "${expected_official}.partial" OR
       EXISTS "${expected_official}.tmp")
        message(FATAL_ERROR
            "${profile} validate-only failed: ${validate_error}")
    endif()
endforeach()

file(GLOB_RECURSE unexpected_outputs LIST_DIRECTORIES false
    "${SCRATCH_ROOT}/*.png"
    "${SCRATCH_ROOT}/representatives.csv"
    "${SCRATCH_ROOT}/contact_sheet_spec.json")
if(unexpected_outputs)
    message(FATAL_ERROR "validate-only created contact-sheet outputs")
endif()

function(expect_rejected case_name output_path)
    execute_process(
        COMMAND "${PRODUCER}" ${ARGN}
        WORKING_DIRECTORY "${REPOSITORY_ROOT}"
        RESULT_VARIABLE rejected_result
        OUTPUT_VARIABLE rejected_output
        ERROR_VARIABLE rejected_error
    )
    if(rejected_result EQUAL 0 OR EXISTS "${output_path}" OR
       EXISTS "${output_path}.partial")
        message(FATAL_ERROR "${case_name} was not rejected atomically")
    endif()
endfunction()

set(invalid_root "${SCRATCH_ROOT}/invalid")
file(MAKE_DIRECTORY "${invalid_root}")
expect_rejected(missing_commit "${invalid_root}/missing_commit"
    --produce --profile mini75_primary --repository-root "${REPOSITORY_ROOT}"
    --scratch-output "${invalid_root}/missing_commit"
    --producer-source-sha256 "${valid_sha}"
    --producer-binary-sha256 "${valid_sha}"
    --producer-worktree-state uncommitted_wip)
expect_rejected(invalid_commit "${invalid_root}/invalid_commit"
    --produce --profile mini75_primary --repository-root "${REPOSITORY_ROOT}"
    --scratch-output "${invalid_root}/invalid_commit"
    --producer-commit invalid
    --producer-source-sha256 "${valid_sha}"
    --producer-binary-sha256 "${valid_sha}"
    --producer-worktree-state uncommitted_wip)
expect_rejected(invalid_source "${invalid_root}/invalid_source"
    --produce --profile mini75_primary --repository-root "${REPOSITORY_ROOT}"
    --scratch-output "${invalid_root}/invalid_source"
    --producer-commit "${valid_commit}"
    --producer-source-sha256 invalid
    --producer-binary-sha256 "${valid_sha}"
    --producer-worktree-state uncommitted_wip)
expect_rejected(invalid_binary "${invalid_root}/invalid_binary"
    --produce --profile mini75_primary --repository-root "${REPOSITORY_ROOT}"
    --scratch-output "${invalid_root}/invalid_binary"
    --producer-commit "${valid_commit}"
    --producer-source-sha256 "${valid_sha}"
    --producer-binary-sha256 invalid
    --producer-worktree-state uncommitted_wip)
expect_rejected(invalid_state "${invalid_root}/invalid_state"
    --produce --profile mini75_primary --repository-root "${REPOSITORY_ROOT}"
    --scratch-output "${invalid_root}/invalid_state"
    --producer-commit "${valid_commit}"
    --producer-source-sha256 "${valid_sha}"
    --producer-binary-sha256 "${valid_sha}"
    --producer-worktree-state unknown)
expect_rejected(arbitrary_output_dir "${invalid_root}/arbitrary"
    --produce --profile mini75_primary --repository-root "${REPOSITORY_ROOT}"
    --output-dir "${invalid_root}/arbitrary"
    --producer-commit "${valid_commit}"
    --producer-source-sha256 "${valid_sha}"
    --producer-binary-sha256 "${valid_sha}"
    --producer-worktree-state clean_commit)

set(official_wip_root "${SCRATCH_ROOT}/official_wip_root")
file(MAKE_DIRECTORY "${official_wip_root}")
expect_rejected(official_wip
    "${official_wip_root}/sakai_multires_c/results/vae_c/iic/common_contact_sheets_v1/mini75_primary"
    --produce --profile mini75_primary
    --repository-root "${official_wip_root}"
    --producer-commit "${valid_commit}"
    --producer-source-sha256 "${valid_sha}"
    --producer-binary-sha256 "${valid_sha}"
    --producer-worktree-state uncommitted_wip)

foreach(state IN ITEMS clean_commit uncommitted_wip)
    set(scratch_output "${SCRATCH_ROOT}/scratch_${state}")
    execute_process(
        COMMAND "${PRODUCER}" --produce
            --profile mini75_primary
            --repository-root "${REPOSITORY_ROOT}"
            --scratch-output "${scratch_output}"
            --producer-commit "${valid_commit}"
            --producer-source-sha256 "${valid_sha}"
            --producer-binary-sha256 "${valid_sha}"
            --producer-worktree-state "${state}"
        WORKING_DIRECTORY "${REPOSITORY_ROOT}"
        RESULT_VARIABLE scratch_result
        OUTPUT_VARIABLE scratch_stdout
        ERROR_VARIABLE scratch_stderr
    )
    file(GLOB scratch_files LIST_DIRECTORIES false "${scratch_output}/*")
    list(LENGTH scratch_files scratch_file_count)
    if(NOT scratch_result EQUAL 0 OR NOT scratch_file_count EQUAL 39 OR
       NOT scratch_stdout MATCHES "OUTPUT_MODE=scratch" OR
       NOT scratch_stdout MATCHES "OUTPUT_CREATED=true")
        message(FATAL_ERROR
            "scratch ${state} production failed: ${scratch_stderr}")
    endif()
    file(READ "${scratch_output}/contact_sheet_spec.json" scratch_spec)
    foreach(expected IN ITEMS
        "\"producer_commit\": \"${valid_commit}\""
        "\"producer_source_sha256\": \"${valid_sha}\""
        "\"producer_binary_sha256\": \"${valid_sha}\""
        "\"producer_worktree_state\": \"${state}\"")
        string(FIND "${scratch_spec}" "${expected}" provenance_position)
        if(provenance_position EQUAL -1)
            message(FATAL_ERROR
                "scratch ${state} provenance missing: ${expected}")
        endif()
    endforeach()
endforeach()

set(collision_root "${SCRATCH_ROOT}/collision_root")
set(collision_output
    "${collision_root}/sakai_multires_c/results/vae_c/iic/common_contact_sheets_v1/mini75_primary")
file(MAKE_DIRECTORY "${collision_output}")
execute_process(
    COMMAND "${PRODUCER}" --produce --profile mini75_primary
        --repository-root "${collision_root}"
        --producer-commit "${valid_commit}"
        --producer-source-sha256 "${valid_sha}"
        --producer-binary-sha256 "${valid_sha}"
        --producer-worktree-state clean_commit
    RESULT_VARIABLE collision_result
    OUTPUT_VARIABLE collision_stdout
    ERROR_VARIABLE collision_stderr
)
if(collision_result EQUAL 0 OR NOT IS_DIRECTORY "${collision_output}" OR
   EXISTS "${collision_output}.partial")
    message(FATAL_ERROR "official collision was not rejected atomically")
endif()

message(STATUS "common contact-sheet provenance/routing tests PASS")
