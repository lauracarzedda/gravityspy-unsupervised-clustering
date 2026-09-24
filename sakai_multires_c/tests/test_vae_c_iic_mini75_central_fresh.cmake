if(NOT DEFINED TEST_EXECUTABLE OR NOT DEFINED REPOSITORY_ROOT OR
   NOT DEFINED TEST_ROOT)
    message(FATAL_ERROR "Missing central test inputs")
endif()
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef nonce)
foreach(iteration RANGE 1 2)
    set(fixture "${TEST_ROOT}/mini75_central_${nonce}_${iteration}")
    file(MAKE_DIRECTORY "${fixture}")
    execute_process(
        COMMAND "${TEST_EXECUTABLE}" "${REPOSITORY_ROOT}" "${fixture}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
    )
    if(NOT result EQUAL 0 OR
       NOT output MATCHES "CENTRAL_TEST_PASS=true")
        message(FATAL_ERROR
            "Central fresh process ${iteration} failed: ${result}\n${output}\n${error}")
    endif()
    string(REGEX MATCH "CENTRAL_MU_SHA256=[0-9a-f]+" digest
           "${output}")
    string(REPLACE "CENTRAL_MU_SHA256=" "" digest_hex "${digest}")
    string(LENGTH "${digest_hex}" digest_length)
    if(NOT digest_length EQUAL 64)
        message(FATAL_ERROR "Central test omitted digest: ${output}")
    endif()
    if(iteration EQUAL 1)
        set(first_digest "${digest}")
    elseif(NOT digest STREQUAL first_digest)
        message(FATAL_ERROR "Central fresh-process digest mismatch")
    endif()
endforeach()
message("CENTRAL_FRESH_PROCESS_BITWISE=true")
