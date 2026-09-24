if(NOT DEFINED TEST_EXECUTABLE OR NOT DEFINED REPOSITORY_ROOT OR
   NOT DEFINED TEST_ROOT)
    message(FATAL_ERROR "Missing mini75 IIC runner test inputs")
endif()
if(NOT IS_DIRECTORY "${TEST_ROOT}" OR
   NOT EXISTS "${TEST_EXECUTABLE}" OR
   NOT IS_DIRECTORY "${REPOSITORY_ROOT}")
    message(FATAL_ERROR "Mini75 IIC runner test paths are invalid")
endif()

# The code-commit argument is resolved at test time, not configure time: the
# same CTest registration must remain valid after an isolated commit.
execute_process(
    COMMAND git --no-optional-locks -C "${REPOSITORY_ROOT}" rev-parse HEAD
    RESULT_VARIABLE git_result
    OUTPUT_VARIABLE code_commit
    ERROR_VARIABLE git_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(LENGTH "${code_commit}" code_commit_length)
if(NOT git_result EQUAL 0 OR NOT code_commit_length EQUAL 40 OR
   NOT code_commit MATCHES "^[0-9a-f]+$")
    message(FATAL_ERROR "Cannot resolve repository HEAD: ${git_error}")
endif()

execute_process(
    COMMAND "${TEST_EXECUTABLE}"
        --validate-only
        --repository-root "${REPOSITORY_ROOT}"
        --code-commit "${code_commit}"
    RESULT_VARIABLE validate_result
    OUTPUT_VARIABLE validate_output
    ERROR_VARIABLE validate_error
    TIMEOUT 120
)
if(NOT validate_result EQUAL 0 OR
   NOT validate_output MATCHES "VALIDATE_ONLY_PASS=true" OR
   NOT validate_output MATCHES
       "PNG_CONTENT_IDENTITY_SHA256=5e6bf28879886f69223864076d354a5bce511b78164e96e742e6117f79f51a5c" OR
   NOT validate_output MATCHES "ENCODER_FORWARD_CALLS=0" OR
   NOT validate_output MATCHES "IIC_OPTIMIZER_STEPS_PER_PAIR=0" OR
   NOT validate_output MATCHES "OUTPUT_CREATED=false")
    message(FATAL_ERROR
        "Mini75 IIC validate-only failed: ${validate_result}\n"
        "${validate_output}\n${validate_error}")
endif()

set(wrong_commit "0000000000000000000000000000000000000000")
if(wrong_commit STREQUAL code_commit)
    set(wrong_commit "1111111111111111111111111111111111111111")
endif()
execute_process(
    COMMAND "${TEST_EXECUTABLE}"
        --validate-only
        --repository-root "${REPOSITORY_ROOT}"
        --code-commit "${wrong_commit}"
    RESULT_VARIABLE wrong_result
    OUTPUT_VARIABLE wrong_output
    ERROR_VARIABLE wrong_error
    TIMEOUT 120
)
if(wrong_result EQUAL 0 OR
   wrong_output MATCHES "VALIDATE_ONLY_PASS=true")
    message(FATAL_ERROR
        "Mini75 IIC accepted wrong code commit:\n"
        "${wrong_output}\n${wrong_error}")
endif()

execute_process(
    COMMAND "${TEST_EXECUTABLE}"
        --smoke-partial
        --repository-root "${REPOSITORY_ROOT}"
        --code-commit "${code_commit}"
    RESULT_VARIABLE smoke_result
    OUTPUT_VARIABLE smoke_output
    ERROR_VARIABLE smoke_error
    TIMEOUT 600
)
if(NOT smoke_result EQUAL 0 OR
   NOT smoke_output MATCHES "SMOKE_PARTIAL_PASS=true" OR
   NOT smoke_output MATCHES "IIC_OPTIMIZER_STEPS_PER_PAIR=3" OR
   NOT smoke_output MATCHES "LATEST_CREATED=false" OR
   NOT smoke_output MATCHES "FINAL_CREATED=false" OR
   NOT smoke_output MATCHES "OUTPUT_CREATED=false")
    message(FATAL_ERROR
        "Mini75 IIC partial smoke failed: ${smoke_result}\n"
        "${smoke_output}\n${smoke_error}")
endif()

# An existing repository file is an intentionally non-creatable output-dir.
# Even if a future path guard regresses, the test cannot create a new results
# directory. The runner must specifically identify the inside-repo refusal.
set(forbidden_path "${REPOSITORY_ROOT}/sakai_multires_c/CMakeLists.txt")
file(SHA256 "${forbidden_path}" forbidden_hash_before)
execute_process(
    COMMAND "${TEST_EXECUTABLE}"
        --run
        --repository-root "${REPOSITORY_ROOT}"
        --code-commit "${code_commit}"
        --output-dir "${forbidden_path}"
        --max-epochs 1
        --encoder-microbatch 64
    RESULT_VARIABLE forbidden_result
    OUTPUT_VARIABLE forbidden_output
    ERROR_VARIABLE forbidden_error
    TIMEOUT 120
)
file(SHA256 "${forbidden_path}" forbidden_hash_after)
if(forbidden_result EQUAL 0 OR
   NOT forbidden_hash_after STREQUAL forbidden_hash_before OR
   NOT "${forbidden_output}\n${forbidden_error}" MATCHES
       "OUTPUT_DIR_INSIDE_REPO_REJECTED=true")
    message(FATAL_ERROR
        "Mini75 IIC inside-repository output path was not safely rejected: "
        "${forbidden_result}\n${forbidden_output}\n${forbidden_error}")
endif()

message("MINI75_IIC_RUNNER_VALIDATE_ONLY_PASS=true")
message("MINI75_IIC_RUNNER_WRONG_SHA_REJECTED=true")
message("MINI75_IIC_RUNNER_PARTIAL_SMOKE_PASS=true")
message("MINI75_IIC_RUNNER_REPO_OUTPUT_REJECTED=true")
