if(NOT DEFINED TEST_EXECUTABLE OR NOT DEFINED REPOSITORY_ROOT)
    message(FATAL_ERROR "Missing perturbed test inputs")
endif()
foreach(iteration RANGE 1 2)
    execute_process(
        COMMAND "${TEST_EXECUTABLE}" "${REPOSITORY_ROOT}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
    )
    if(NOT result EQUAL 0 OR
       NOT output MATCHES "PERTURBED_TEST_PASS=true")
        message(FATAL_ERROR
            "Perturbed fresh process ${iteration} failed: ${result}\n${output}\n${error}")
    endif()
    string(REGEX MATCH "PERTURBED_SHA256=[0-9a-f]+" digest
           "${output}")
    string(REPLACE "PERTURBED_SHA256=" "" digest_hex "${digest}")
    string(LENGTH "${digest_hex}" digest_length)
    if(NOT digest_length EQUAL 64)
        message(FATAL_ERROR "Perturbed test omitted digest: ${output}")
    endif()
    if(iteration EQUAL 1)
        set(first_digest "${digest}")
    elseif(NOT digest STREQUAL first_digest)
        message(FATAL_ERROR "Perturbed fresh-process digest mismatch")
    endif()
endforeach()
message("PERTURBED_FRESH_PROCESS_BITWISE=true")
