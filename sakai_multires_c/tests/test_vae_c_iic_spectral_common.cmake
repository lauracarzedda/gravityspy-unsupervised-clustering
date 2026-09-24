if(NOT DEFINED EVALUATOR)
    message(FATAL_ERROR "EVALUATOR is required")
endif()
if(NOT DEFINED REPOSITORY_ROOT)
    message(FATAL_ERROR "REPOSITORY_ROOT is required")
endif()

execute_process(
    COMMAND "${EVALUATOR}" --self-test
    RESULT_VARIABLE first_result
    OUTPUT_VARIABLE first_output
    ERROR_VARIABLE first_error
)
if(NOT first_result EQUAL 0)
    message(FATAL_ERROR "common evaluator self-test failed: ${first_error}")
endif()

execute_process(
    COMMAND "${EVALUATOR}" --self-test
    RESULT_VARIABLE second_result
    OUTPUT_VARIABLE second_output
    ERROR_VARIABLE second_error
)
if(NOT second_result EQUAL 0)
    message(FATAL_ERROR "common evaluator repeat failed: ${second_error}")
endif()

if(NOT first_output STREQUAL second_output)
    message(FATAL_ERROR "fresh-process self-test output differs")
endif()
if(NOT first_output MATCHES "COMMON_EVALUATOR_SELF_TEST_PASS=true")
    message(FATAL_ERROR "self-test PASS marker missing")
endif()

execute_process(
    COMMAND "${EVALUATOR}" --validate-only
        --profile mini75_primary
        --repository-root "${REPOSITORY_ROOT}"
        --full224-root "${REPOSITORY_ROOT}"
    RESULT_VARIABLE validate_result
    OUTPUT_VARIABLE validate_output
    ERROR_VARIABLE validate_error
)
if(NOT validate_result EQUAL 0 OR
   NOT validate_output MATCHES "METRICS_COMPUTED=false" OR
   NOT validate_output MATCHES "OUTPUT_CREATED=false" OR
   NOT validate_output MATCHES "VALIDATE_ONLY_PASS=true")
    message(FATAL_ERROR "validate-only contract failed: ${validate_error}")
endif()

set(valid_commit "0123456789abcdef0123456789abcdef01234567")
set(valid_sha "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")
foreach(case_name IN ITEMS invalid_commit invalid_binary invalid_state)
    set(commit "${valid_commit}")
    set(binary_sha "${valid_sha}")
    set(state "clean_commit")
    if(case_name STREQUAL "invalid_commit")
        set(commit "invalid")
    elseif(case_name STREQUAL "invalid_binary")
        set(binary_sha "invalid")
    else()
        set(state "unknown")
    endif()
    execute_process(
        COMMAND "${EVALUATOR}" --evaluate
            --profile mini75_primary
            --repository-root "${REPOSITORY_ROOT}"
            --full224-root "${REPOSITORY_ROOT}"
            --output-dir "${CMAKE_CURRENT_BINARY_DIR}/must_not_exist_${case_name}"
            --producer-commit "${commit}"
            --producer-source-sha256 "${valid_sha}"
            --producer-binary-sha256 "${binary_sha}"
            --evaluator-worktree-state "${state}"
        RESULT_VARIABLE invalid_result
        OUTPUT_VARIABLE invalid_output
        ERROR_VARIABLE invalid_error
    )
    if(invalid_result EQUAL 0 OR
       EXISTS "${CMAKE_CURRENT_BINARY_DIR}/must_not_exist_${case_name}")
        message(FATAL_ERROR "${case_name} was not rejected atomically")
    endif()
endforeach()

message(STATUS "common evaluator provenance/validate/numeric tests PASS")
