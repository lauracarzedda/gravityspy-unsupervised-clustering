if(NOT DEFINED TEST_EXECUTABLE OR NOT DEFINED REPOSITORY_ROOT OR
   NOT DEFINED TEST_ROOT)
    message(FATAL_ERROR "Inspector test paths are required")
endif()

set(run_id "mini112_iic_v1_0cecf5cd656d0d61a237d50fcf96a7890fb2e6d052ef403051bc9bc17aed8fba")
set(run_dir "${REPOSITORY_ROOT}/sakai_multires_c/results/vae_c/iic/${run_id}")
set(final "${run_dir}/final.ckpt")
set(latest "${run_dir}/latest.ckpt")
set(fixture_dir "${TEST_ROOT}/iic_checkpoint_inspector_fixture")
file(MAKE_DIRECTORY "${fixture_dir}")
set(corrupt "${fixture_dir}/corrupt.ckpt")
file(WRITE "${corrupt}" "IICKPT1 corrupted test fixture")

file(SHA256 "${final}" final_before)
file(SHA256 "${latest}" latest_before)
if(NOT final_before STREQUAL
       "c4a989377434ddc2f589cd76362e9263078411423e30144d4137a686bcf76f60" OR
   NOT latest_before STREQUAL
       "705b97468fc9f5f524316b872ce47664454e203112053770c301e18cad456c9c")
    message(FATAL_ERROR "Official checkpoint SHA mismatch before inspector")
endif()
file(GLOB run_before LIST_DIRECTORIES false "${run_dir}/*")
file(GLOB fixture_before LIST_DIRECTORIES false "${fixture_dir}/*")

set(identity_args
    --input-dim 64 --capacity 128 --batch-size 128
    --train-rows 8583 --max-epochs 100 --shuffle-seed 101112
    --run-id "${run_id}"
    --base-commit 887046ce505dcca7f09bc8e51ea2b59332d8b4f6
    --producer-state BINARY_SHA256_AUTHORITY
    --producer-binary-sha256
    c41b49f6671678a94c6ee6b62538785249ec3bbc5a000822917f4b975d7f3333)

execute_process(
    COMMAND "${TEST_EXECUTABLE}" --checkpoint "${final}"
            ${identity_args} --expected-kind FINAL --expected-epoch 99
    RESULT_VARIABLE valid_result OUTPUT_VARIABLE valid_output
    ERROR_VARIABLE valid_error)
if(NOT valid_result EQUAL 0 OR
   NOT valid_output MATCHES "LOAD_PASS=true" OR
   NOT valid_output MATCHES "KIND=FINAL" OR
   NOT valid_output MATCHES "PAIR4_STEP=6800" OR
   NOT valid_output MATCHES "PAIR4_OVER_BN_BATCHES=13600")
    message(FATAL_ERROR "Valid API load failed: ${valid_result} ${valid_output} ${valid_error}")
endif()

execute_process(
    COMMAND "${TEST_EXECUTABLE}" --checkpoint "${corrupt}"
            ${identity_args} --expected-kind FINAL --expected-epoch 99
    RESULT_VARIABLE corrupt_result OUTPUT_VARIABLE corrupt_output
    ERROR_VARIABLE corrupt_error)
if(corrupt_result EQUAL 0 OR
   NOT corrupt_error MATCHES "LOAD_PASS=false")
    message(FATAL_ERROR "Corrupt checkpoint accepted")
endif()

set(wrong_args ${identity_args})
list(REMOVE_AT wrong_args 13)
list(INSERT wrong_args 13 "wrong-run-id")
execute_process(
    COMMAND "${TEST_EXECUTABLE}" --checkpoint "${final}"
            ${wrong_args} --expected-kind FINAL --expected-epoch 99
    RESULT_VARIABLE wrong_result OUTPUT_VARIABLE wrong_output
    ERROR_VARIABLE wrong_error)
if(wrong_result EQUAL 0 OR
   NOT wrong_error MATCHES "IDENTITY_MISMATCH")
    message(FATAL_ERROR "Identity mismatch accepted: ${wrong_error}")
endif()

file(SHA256 "${final}" final_after)
file(SHA256 "${latest}" latest_after)
file(GLOB run_after LIST_DIRECTORIES false "${run_dir}/*")
file(GLOB fixture_after LIST_DIRECTORIES false "${fixture_dir}/*")
if(NOT final_before STREQUAL final_after OR
   NOT latest_before STREQUAL latest_after OR
   NOT run_before STREQUAL run_after OR
   NOT fixture_before STREQUAL fixture_after)
    message(FATAL_ERROR "Inspector changed inputs or created files")
endif()
message(STATUS "IIC_CHECKPOINT_INSPECTOR_VALID=true")
message(STATUS "IIC_CHECKPOINT_INSPECTOR_CORRUPT_REJECTED=true")
message(STATUS "IIC_CHECKPOINT_INSPECTOR_IDENTITY_REJECTED=true")
message(STATUS "IIC_CHECKPOINT_INSPECTOR_INPUT_UNCHANGED=true")
message(STATUS "IIC_CHECKPOINT_INSPECTOR_NO_OUTPUT=true")
