if(NOT DEFINED TEST_EXECUTABLE OR NOT DEFINED REPOSITORY_ROOT)
    message(FATAL_ERROR "mini112 H real-tiny arguments missing")
endif()
execute_process(
    COMMAND "${TEST_EXECUTABLE}" "${REPOSITORY_ROOT}"
    RESULT_VARIABLE first_status OUTPUT_VARIABLE first ERROR_VARIABLE first_error)
execute_process(
    COMMAND "${TEST_EXECUTABLE}" "${REPOSITORY_ROOT}"
    RESULT_VARIABLE second_status OUTPUT_VARIABLE second ERROR_VARIABLE second_error)
if(NOT first_status EQUAL 0 OR NOT second_status EQUAL 0 OR
   NOT first_error STREQUAL "" OR NOT second_error STREQUAL "" OR
   NOT first STREQUAL second OR
   NOT first MATCHES "MINI112_H_REAL_FINAL_TINY_PASS=true" OR
   NOT first MATCHES "MINI112_H_BATCH_INDEPENDENT=BITWISE")
    message(FATAL_ERROR "mini112 H real-tiny fresh-process mismatch: ${first_error} ${second_error}")
endif()
message(STATUS "mini112 H FINAL tiny, chunk128/64/32, tail7, fresh-process PASS")
