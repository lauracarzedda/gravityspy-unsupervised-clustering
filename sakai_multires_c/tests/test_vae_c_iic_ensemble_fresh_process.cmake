if(NOT DEFINED TEST_EXECUTABLE)
    message(FATAL_ERROR "TEST_EXECUTABLE is required")
endif()

function(run_ensemble_test output_variable)
    execute_process(
        COMMAND "${TEST_EXECUTABLE}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
    )
    if(NOT result EQUAL 0)
        message(FATAL_ERROR
            "IIC ensemble test failed with exit ${result}:\n${output}\n${error}"
        )
    endif()
    if(NOT error STREQUAL "")
        message(FATAL_ERROR "IIC ensemble test wrote to stderr:\n${error}")
    endif()
    set(${output_variable} "${output}" PARENT_SCOPE)
endfunction()

run_ensemble_test(first_output)
run_ensemble_test(second_output)

if(NOT first_output STREQUAL second_output)
    message(FATAL_ERROR
        "Fresh processes produced different ensemble state.\n"
        "FIRST:\n${first_output}\nSECOND:\n${second_output}"
    )
endif()

if(NOT first_output MATCHES "IIC_ENSEMBLE_SMOKE_PASS=true")
    message(FATAL_ERROR "IIC ensemble completion marker missing")
endif()

message("${first_output}")
message("FRESH_PROCESS_IIC_ENSEMBLE_STATE_IDENTICAL=true")
