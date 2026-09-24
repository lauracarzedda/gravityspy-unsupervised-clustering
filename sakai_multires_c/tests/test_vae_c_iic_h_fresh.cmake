if(NOT DEFINED TEST_EXECUTABLE OR NOT DEFINED REPOSITORY_ROOT)
    message(FATAL_ERROR "Missing executable or repository root")
endif()

function(run_case output mode)
    if(mode STREQUAL "synthetic")
        execute_process(
            COMMAND "${TEST_EXECUTABLE}"
            RESULT_VARIABLE result
            OUTPUT_VARIABLE text
            ERROR_VARIABLE error
        )
    else()
        execute_process(
            COMMAND "${TEST_EXECUTABLE}" --real-tiny "${REPOSITORY_ROOT}"
            RESULT_VARIABLE result
            OUTPUT_VARIABLE text
            ERROR_VARIABLE error
        )
    endif()
    if(NOT result EQUAL 0 OR NOT error STREQUAL "")
        message(FATAL_ERROR "IIC H ${mode} failed: ${result} ${error}")
    endif()
    set(${output} "${text}" PARENT_SCOPE)
endfunction()

if(NOT DEFINED MODE OR NOT MODE MATCHES "^(synthetic|real)$")
    message(FATAL_ERROR "Invalid test mode")
endif()
run_case(first "${MODE}")
run_case(second "${MODE}")
if(NOT first STREQUAL second)
    message(FATAL_ERROR "IIC H fresh-process output differs: ${MODE}")
endif()
if(MODE STREQUAL "synthetic")
    if(NOT first MATCHES "H_SYNTHETIC_PASS=true")
        message(FATAL_ERROR "Synthetic H gate absent")
    endif()
else()
    if(NOT first MATCHES "REAL_FINAL_TINY_PASS=true")
        message(FATAL_ERROR "Real FINAL tiny H gate absent")
    endif()
endif()
message(STATUS "IIC H ${MODE} fresh-process PASS")
