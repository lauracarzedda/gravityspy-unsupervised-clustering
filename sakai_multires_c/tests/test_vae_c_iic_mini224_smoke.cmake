if(NOT DEFINED TEST_EXECUTABLE OR NOT DEFINED REPOSITORY_ROOT OR
   NOT DEFINED TEST_ROOT OR NOT EXISTS "${TEST_EXECUTABLE}" OR
   NOT IS_DIRECTORY "${REPOSITORY_ROOT}" OR
   NOT IS_DIRECTORY "${TEST_ROOT}")
    message(FATAL_ERROR "Invalid mini224 smoke test inputs")
endif()
execute_process(
    COMMAND git --no-optional-locks -C "${REPOSITORY_ROOT}" rev-parse HEAD
    RESULT_VARIABLE git_result
    OUTPUT_VARIABLE commit
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT git_result EQUAL 0)
    message(FATAL_ERROR "Cannot resolve mini224 smoke producer HEAD")
endif()
set(smoke_dir "${TEST_ROOT}/mini224_iic_smoke_external_fixture")
file(MAKE_DIRECTORY "${smoke_dir}")
file(GLOB fixture_before "${smoke_dir}/*")
if(fixture_before)
    message(FATAL_ERROR "Smoke fixture is not initially empty")
endif()
set(base_args --smoke-real-two-batch
    --repository-root "${REPOSITORY_ROOT}"
    --code-commit "${commit}"
    --smoke-output-dir "${smoke_dir}")

function(expect_rejected title)
    execute_process(
        COMMAND "${TEST_EXECUTABLE}" ${base_args} ${ARGN}
        RESULT_VARIABLE result
        TIMEOUT 180
    )
    if(result EQUAL 0)
        message(FATAL_ERROR "Invalid smoke request accepted: ${title}")
    endif()
endfunction()
expect_rejected("logical batch" --logical-batch 64)
expect_rejected("microbatch >16" --encoder-microbatch 17)
expect_rejected("BEST mismatch" --expected-best-sha256
    0000000000000000000000000000000000000000000000000000000000000000)
expect_rejected("population mismatch" --expected-population 999)
expect_rejected("decoder unavailable" --decoder)
execute_process(
    COMMAND "${TEST_EXECUTABLE}" --smoke-real-two-batch
        --repository-root "${REPOSITORY_ROOT}"
        --code-commit "${commit}"
        --smoke-output-dir "${REPOSITORY_ROOT}"
    RESULT_VARIABLE routed_result
    TIMEOUT 30
)
if(routed_result EQUAL 0)
    message(FATAL_ERROR "Smoke accepted output inside repository")
endif()

execute_process(
    COMMAND "${TEST_EXECUTABLE}" ${base_args}
    RESULT_VARIABLE first_result
    OUTPUT_VARIABLE first_output
    ERROR_VARIABLE first_error
    TIMEOUT 1800
)
if(NOT first_result EQUAL 0)
    message(FATAL_ERROR "First real smoke failed: ${first_error}\n${first_output}")
endif()
execute_process(
    COMMAND "${TEST_EXECUTABLE}" ${base_args}
    RESULT_VARIABLE second_result
    OUTPUT_VARIABLE second_output
    ERROR_VARIABLE second_error
    TIMEOUT 1800
)
if(NOT second_result EQUAL 0)
    message(FATAL_ERROR "Fresh-process smoke failed: ${second_error}\n${second_output}")
endif()

foreach(expected IN ITEMS
    "SMOKE_REAL_TWO_BATCH_PASS=true"
    "SMOKE_BATCH_INDEX=0 EPOCH=0 LOGICAL_BATCH_SIZE=128"
    "SMOKE_BATCH_INDEX=7 EPOCH=0 LOGICAL_BATCH_SIZE=104"
    "SMOKE_BATCHES=2"
    "SMOKE_SAMPLES=232"
    "VAE_ENCODER_FORWARD_CALLS=15"
    "DECODER_CALLS=0"
    "IIC_HEAD_FORWARD_CALLS=40"
    "IIC_HEAD_BACKWARD_CALLS=40"
    "ADAM_STEPS_TOTAL=10"
    "OPTIMIZER_STEPS_PER_PAIR=2"
    "IIC_BN_UPDATES_PER_HEAD=4"
    "LATEST_CREATED=false"
    "FINAL_CREATED=false"
    "OUTPUT_CREATED=false"
)
    string(FIND "${first_output}" "${expected}" found)
    if(found EQUAL -1)
        message(FATAL_ERROR "Missing smoke gate: ${expected}")
    endif()
endforeach()
foreach(variable IN ITEMS first_output second_output)
    string(REGEX REPLACE
        "(SMOKE_ELAPSED_SECONDS|SMOKE_LOADER_ENCODER_SECONDS|SMOKE_IIC_SECONDS|PEAK_WORKING_SET_BYTES|PRIVATE_BYTES_AT_END)=[^\r\n]*[\r\n]*"
        "" stable "${${variable}}")
    set("${variable}_stable" "${stable}")
endforeach()
if(NOT first_output_stable STREQUAL second_output_stable)
    message(FATAL_ERROR "Fresh-process row/crop/loss/state mismatch")
endif()
file(GLOB fixture_after "${smoke_dir}/*")
if(fixture_after)
    message(FATAL_ERROR "Smoke created a fixture artifact")
endif()
message(STATUS "mini224 real two-batch smoke and fresh-process replay PASS")
message(STATUS "${first_output}")
