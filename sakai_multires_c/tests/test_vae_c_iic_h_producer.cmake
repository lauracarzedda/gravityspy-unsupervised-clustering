if(NOT DEFINED TEST_EXECUTABLE OR NOT DEFINED REPOSITORY_ROOT OR
   NOT DEFINED FIXTURE_PARENT)
    message(FATAL_ERROR "H producer test arguments missing")
endif()

file(SHA256 "${TEST_EXECUTABLE}" producer_sha)
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef suffix)
set(root "${FIXTURE_PARENT}/h6c_${suffix}")
set(first "${root}/chunk64")
set(second "${root}/chunk32")
set(validate "${root}/validate_only")
file(MAKE_DIRECTORY "${root}")
set(commit "0000000000000000000000000000000000000000")

function(invoke output chunk mode)
    set(arguments
        --repository-root "${REPOSITORY_ROOT}"
        --output-dir "${output}"
        --chunk-size "${chunk}"
        --producer-commit "${commit}"
        --producer-binary-sha256 "${producer_sha}")
    if(mode STREQUAL "validate")
        list(APPEND arguments --validate-only)
    endif()
    execute_process(
        COMMAND "${TEST_EXECUTABLE}" ${arguments}
        RESULT_VARIABLE result
        OUTPUT_VARIABLE stdout
        ERROR_VARIABLE stderr)
    if(mode STREQUAL "reject")
        if(result EQUAL 0)
            message(FATAL_ERROR "H producer accepted overwrite")
        endif()
    else()
        if(NOT result EQUAL 0 OR NOT stderr STREQUAL "")
            message(FATAL_ERROR "H producer failed: ${result} ${stderr}")
        endif()
        if(mode STREQUAL "validate")
            if(NOT stdout MATCHES "H_FORWARD_CALLS=0" OR
               NOT stdout MATCHES "OUTPUT_CREATED=false")
                message(FATAL_ERROR "H validate-only gate missing")
            endif()
        elseif(NOT stdout MATCHES "H_CREATED=true")
            message(FATAL_ERROR "H producer success gate missing")
        endif()
    endif()
endfunction()

invoke("${validate}" 64 validate)
if(EXISTS "${validate}" OR EXISTS "${validate}.partial")
    message(FATAL_ERROR "H validate-only created output")
endif()
execute_process(
    COMMAND "${TEST_EXECUTABLE}"
        --repository-root "${REPOSITORY_ROOT}"
        --output-dir "${root}/bad_binary_identity"
        --chunk-size 64
        --producer-commit "${commit}"
        --producer-binary-sha256
            0000000000000000000000000000000000000000000000000000000000000000
        --validate-only
    RESULT_VARIABLE bad_identity_result
    ERROR_VARIABLE bad_identity_error)
if(bad_identity_result EQUAL 0 OR EXISTS "${root}/bad_binary_identity")
    message(FATAL_ERROR "H producer accepted wrong binary SHA")
endif()
invoke("${first}" 64 produce)
invoke("${second}" 32 produce)

foreach(output IN ITEMS "${first}" "${second}")
    if(EXISTS "${output}.partial" OR EXISTS "${output}.tmp")
        message(FATAL_ERROR "H partial/tmp remains")
    endif()
    file(GLOB files LIST_DIRECTORIES FALSE "${output}/*")
    list(LENGTH files count)
    if(NOT count EQUAL 4)
        message(FATAL_ERROR "H artifact inventory mismatch: ${count}")
    endif()
    file(SIZE "${output}/h_f32.bin" bytes)
    if(NOT bytes EQUAL 6179760)
        message(FATAL_ERROR "H payload size mismatch: ${bytes}")
    endif()
    file(STRINGS "${output}/rows.csv" rows)
    list(LENGTH rows row_lines)
    if(NOT row_lines EQUAL 8584)
        message(FATAL_ERROR "H row-map count mismatch: ${row_lines}")
    endif()
    list(GET rows 0 header)
    list(GET rows 1 first_row)
    list(GET rows 8583 last_row)
    if(NOT header STREQUAL
        "manifest_index,event_index,event_id,source_split,split_pos" OR
       NOT first_row MATCHES "^0,1," OR
       NOT last_row MATCHES "^8582,8583,")
        message(FATAL_ERROR "H row order/header mismatch")
    endif()
endforeach()

foreach(name IN ITEMS h_f32.bin rows.csv h_spec.txt sha256.txt)
    file(SHA256 "${first}/${name}" sha_first)
    file(SHA256 "${second}/${name}" sha_second)
    if(NOT sha_first STREQUAL sha_second)
        message(FATAL_ERROR "H chunk-size dependence: ${name}")
    endif()
endforeach()

file(SHA256 "${first}/h_f32.bin" before_overwrite)
invoke("${first}" 64 reject)
file(SHA256 "${first}/h_f32.bin" after_overwrite)
if(NOT before_overwrite STREQUAL after_overwrite)
    message(FATAL_ERROR "H overwrite rejection changed payload")
endif()
message(STATUS "H producer validate-only, inventory, chunk independence, overwrite PASS")
