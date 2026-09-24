if(NOT DEFINED TEST_EXECUTABLE)
    message(FATAL_ERROR "TEST_EXECUTABLE is required")
endif()

function(run_pair_train_test output_variable)
    execute_process(
        COMMAND "${TEST_EXECUTABLE}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
    )
    if(NOT result EQUAL 0)
        message(FATAL_ERROR
            "IIC pair train test failed with exit ${result}:\n${output}\n${error}"
        )
    endif()
    if(NOT error STREQUAL "")
        message(FATAL_ERROR "IIC pair train test wrote to stderr:\n${error}")
    endif()
    set(${output_variable} "${output}" PARENT_SCOPE)
endfunction()

run_pair_train_test(first_output)
run_pair_train_test(second_output)

if(NOT first_output STREQUAL second_output)
    message(FATAL_ERROR
        "Fresh processes produced different output.\n"
        "FIRST:\n${first_output}\nSECOND:\n${second_output}"
    )
endif()

message("${first_output}")
message("FRESH_PROCESS_PAIR_TRAIN_OUTPUT_IDENTICAL=true")
