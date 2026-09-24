if(NOT DEFINED TEST_EXECUTABLE OR NOT DEFINED REPOSITORY_ROOT OR
   NOT DEFINED TEST_ROOT OR NOT EXISTS "${TEST_EXECUTABLE}" OR
   NOT IS_DIRECTORY "${REPOSITORY_ROOT}" OR
   NOT IS_DIRECTORY "${TEST_ROOT}")
    message(FATAL_ERROR "Invalid mini224 runner test inputs")
endif()

# Static guard covers the preparation path only. The later smoke path is
# intentionally neural, while validate-only must remain forward-free.
file(READ
    "${REPOSITORY_ROOT}/sakai_multires_c/tools/run_vae_c_iic_mini224.c"
    runner_source)
string(FIND "${runner_source}" "/* SMOKE_CODE_START:" smoke_start)
if(smoke_start EQUAL -1)
    message(FATAL_ERROR "Missing validate-only/smoke separation marker")
endif()
string(SUBSTRING "${runner_source}" 0 ${smoke_start} validate_source)
foreach(forbidden IN ITEMS
    "vae_c_iic_mini224_perturbed_compute("
    "vae_c_iic_ensemble_train_step("
    "vae_c_iic_checkpoint_save_atomic("
    "vae_c_model_encoder_forward("
    "vae_c_model_decoder_forward("
)
    string(FIND "${validate_source}" "${forbidden}" found)
    if(NOT found EQUAL -1)
        message(FATAL_ERROR "Forbidden mini224 validate-only call: ${forbidden}")
    endif()
endforeach()

# HEAD is resolved at test time so the same test also works after a commit.
execute_process(
    COMMAND git --no-optional-locks -C "${REPOSITORY_ROOT}" rev-parse HEAD
    RESULT_VARIABLE git_result
    OUTPUT_VARIABLE commit
    ERROR_VARIABLE git_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(LENGTH "${commit}" commit_length)
if(NOT git_result EQUAL 0 OR NOT commit_length EQUAL 40 OR
   NOT commit MATCHES "^[0-9a-f]+$")
    message(FATAL_ERROR "Cannot resolve HEAD: ${git_error}")
endif()

set(base_args --validate-only --repository-root "${REPOSITORY_ROOT}"
    --code-commit "${commit}")
execute_process(
    COMMAND "${TEST_EXECUTABLE}" ${base_args}
    RESULT_VARIABLE first_result
    OUTPUT_VARIABLE first_output
    ERROR_VARIABLE first_error
    TIMEOUT 360
)
if(NOT first_result EQUAL 0)
    message(FATAL_ERROR "First validate-only failed: ${first_error}")
endif()
execute_process(
    COMMAND "${TEST_EXECUTABLE}" ${base_args}
    RESULT_VARIABLE second_result
    OUTPUT_VARIABLE second_output
    ERROR_VARIABLE second_error
    TIMEOUT 360
)
if(NOT second_result EQUAL 0 OR NOT first_output STREQUAL second_output)
    message(FATAL_ERROR "Fresh-process validate-only mismatch: ${second_error}")
endif()

foreach(expected IN ITEMS
    "VALIDATE_ONLY_PASS=true"
    "POPULATION_TRAIN=800"
    "POPULATION_VALIDATION=200"
    "POPULATION_TOTAL=1000"
    "POPULATION_ORDER=train800_then_validation200_native_split"
    "LOGICAL_BATCH_SIZE=128"
    "ENCODER_MICROBATCH=16"
    "LOGICAL_BATCHES_PER_EPOCH=8"
    "LOGICAL_BATCH_SIZES=128,128,128,128,128,128,128,104"
    "TAIL_ENCODER_CHUNKS=16,16,16,16,16,16,8"
    "MAX_EPOCHS=100"
    "MAX_OPTIMIZER_STEPS_PER_PAIR=800"
    "IIC_CHECKPOINT_FORMAT=IIC_V1"
    "RESUME_AUTHORITY=LATEST"
    "RESUME_START=LAST_COMPLETED_EPOCH_PLUS_1"
    "VAE_ENCODER_FORWARD_CALLS=0"
    "IIC_FORWARD_CALLS=0"
    "IIC_BACKWARD_CALLS=0"
    "ADAM_STEPS=0"
    "OPTIMIZER_STEPS=0"
    "DECODER_CALLS=0"
    "OUTPUT_CREATED=false"
)
    string(FIND "${first_output}" "${expected}" found)
    if(found EQUAL -1)
        message(FATAL_ERROR "Missing mini224 contract: ${expected}")
    endif()
endforeach()
string(REGEX MATCH "RUN_ID=mini224_iic_v1_[0-9a-f]+" run_id
    "${first_output}")
string(REGEX MATCH "RUN_CONFIG_SHA256=[0-9a-f]+" config_sha
    "${first_output}")
string(REGEX MATCH "OUTPUT_DIR=([^\r\n]+)" output_line
    "${first_output}")
if(run_id STREQUAL "" OR config_sha STREQUAL "" OR output_line STREQUAL "")
    message(FATAL_ERROR "Run identity or output path missing")
endif()
set(output_path "${CMAKE_MATCH_1}")
if(EXISTS "${output_path}" OR EXISTS "${output_path}.partial" OR
   EXISTS "${output_path}.tmp")
    message(FATAL_ERROR "Validate-only created official output")
endif()

function(expect_rejected name)
    execute_process(
        COMMAND "${TEST_EXECUTABLE}" ${base_args} ${ARGN}
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
        TIMEOUT 180
    )
    if(result EQUAL 0)
        message(FATAL_ERROR "Invalid ${name} was accepted: ${output}")
    endif()
endfunction()

expect_rejected("microbatch >16" --encoder-microbatch 17)
expect_rejected("microbatch 8 in official profile" --encoder-microbatch 8)
expect_rejected("logical batch" --logical-batch 64)
expect_rejected("epoch budget" --max-epochs 99)
expect_rejected("BEST checksum"
    --expected-best-sha256
    0000000000000000000000000000000000000000000000000000000000000000)
expect_rejected("population" --expected-population 999)
expect_rejected("output collision" --output-dir "${TEST_ROOT}")
execute_process(
    COMMAND "${TEST_EXECUTABLE}" --validate-only
        --repository-root "${REPOSITORY_ROOT}"
        --code-commit 0000000000000000000000000000000000000000
    RESULT_VARIABLE wrong_commit_result
    TIMEOUT 30
)
if(wrong_commit_result EQUAL 0)
    message(FATAL_ERROR "Wrong producer commit was accepted")
endif()

if(EXISTS "${output_path}" OR EXISTS "${output_path}.partial" OR
   EXISTS "${output_path}.tmp")
    message(FATAL_ERROR "Negative tests created official output")
endif()
message(STATUS "mini224 validate-only contract and negative gates PASS")
