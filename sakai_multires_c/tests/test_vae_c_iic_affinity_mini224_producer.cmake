if(NOT DEFINED TEST_EXECUTABLE OR NOT DEFINED REPOSITORY_ROOT OR
   NOT DEFINED FIXTURE_PARENT)
    message(FATAL_ERROR "mini224 affinity producer test arguments missing")
endif()

set(run_id "mini224_iic_v1_4236df3f9e82f0528fcea947b2e15e42fab8159c0d4b1ed03189e4a85daae8a5")
set(h_rel "sakai_multires_c/results/vae_c/iic/${run_id}_h_main_v1")
set(final_rel "sakai_multires_c/results/vae_c/iic/${run_id}/final.ckpt")
file(SHA256 "${TEST_EXECUTABLE}" binary_sha)
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef suffix)
set(root "${FIXTURE_PARENT}/a224_${suffix}")
if(EXISTS "${root}")
    message(FATAL_ERROR "scratch test root already exists")
endif()
file(MAKE_DIRECTORY "${root}")
set(commit "0000000000000000000000000000000000000000")

function(invoke repository output block mode)
    set(args --repository-root "${repository}" --output-dir "${output}"
        --block-rows "${block}" --producer-commit "${commit}"
        --producer-binary-sha256 "${binary_sha}")
    if(mode STREQUAL "validate" OR mode STREQUAL "reject" OR
       mode STREQUAL "collision")
        list(APPEND args --validate-only)
    endif()
    execute_process(COMMAND "${TEST_EXECUTABLE}" ${args}
        RESULT_VARIABLE status OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
    if(mode STREQUAL "collision")
        if(status EQUAL 0 OR NOT EXISTS "${output}")
            message(FATAL_ERROR "mini224 affinity accepted output collision")
        endif()
    elseif(mode STREQUAL "reject")
        if(status EQUAL 0 OR EXISTS "${output}" OR EXISTS "${output}.partial")
            message(FATAL_ERROR "mini224 affinity accepted invalid input/output: ${output}")
        endif()
    elseif(NOT status EQUAL 0 OR NOT stderr STREQUAL "")
        message(FATAL_ERROR "mini224 affinity failed: ${output}: ${status} ${stderr}")
    elseif(mode STREQUAL "validate")
        if(NOT stdout MATCHES "AFFINITY_INPUT_VALIDATED=true" OR
           NOT stdout MATCHES "AFFINITY_COMPUTE_CALLS=0" OR
           NOT stdout MATCHES "OUTPUT_CREATED=false" OR
           EXISTS "${output}" OR EXISTS "${output}.partial")
            message(FATAL_ERROR "mini224 affinity validate-only contract failed")
        endif()
    elseif(NOT stdout MATCHES "AFFINITY_CREATED=true" OR
           NOT stdout MATCHES "ROWS=1000" OR
           NOT stdout MATCHES "COLS=1000" OR
           NOT stdout MATCHES "PAYLOAD_BYTES=8000000")
        message(FATAL_ERROR "mini224 affinity output contract failed")
    endif()
endfunction()

invoke("${REPOSITORY_ROOT}" "${root}/validate" 128 validate)
invoke("${REPOSITORY_ROOT}" "${root}/block128" 128 create)
invoke("${REPOSITORY_ROOT}" "${root}/block64" 64 create)

set(first "${root}/block128")
set(second "${root}/block64")
file(SIZE "${first}/affinity_f64.bin" payload_bytes)
if(NOT payload_bytes EQUAL 8000000)
    message(FATAL_ERROR "mini224 affinity payload size incorrect")
endif()
file(SHA256 "${first}/affinity_f64.bin" first_sha)
file(SHA256 "${second}/affinity_f64.bin" second_sha)
file(SHA256 "${first}/rows.csv" first_rows_sha)
file(SHA256 "${second}/rows.csv" second_rows_sha)
if(NOT first_sha STREQUAL second_sha OR
   NOT first_rows_sha STREQUAL second_rows_sha OR
   NOT first_rows_sha STREQUAL
       "d7f5101cfe3f3ae31f24b12833f0125873888ff9ed33c4c33c428b90955db4c2")
    message(FATAL_ERROR "mini224 affinity block-size/fresh-process mismatch")
endif()
foreach(directory IN ITEMS "${first}" "${second}")
    file(GLOB children LIST_DIRECTORIES false "${directory}/*")
    list(LENGTH children count)
    if(NOT count EQUAL 4 OR EXISTS "${directory}.partial")
        message(FATAL_ERROR "mini224 affinity inventory incorrect")
    endif()
endforeach()
invoke("${REPOSITORY_ROOT}" "${first}" 128 collision)

# A scratch source mirror tests the canonical SHA gates without touching H or FINAL.
set(mirror "${root}/mirror")
foreach(relative IN ITEMS
    "${h_rel}/h_f32.bin" "${h_rel}/rows.csv"
    "${h_rel}/h_spec.txt" "${h_rel}/sha256.txt" "${final_rel}")
    get_filename_component(parent "${mirror}/${relative}" DIRECTORY)
    file(MAKE_DIRECTORY "${parent}")
    file(COPY_FILE "${REPOSITORY_ROOT}/${relative}" "${mirror}/${relative}")
endforeach()
invoke("${mirror}" "${root}/mirror_valid" 128 validate)
foreach(name IN ITEMS h_f32.bin rows.csv h_spec.txt sha256.txt)
    set(relative "${h_rel}/${name}")
    file(APPEND "${mirror}/${relative}" "x")
    invoke("${mirror}" "${root}/bad_${name}" 128 reject)
    file(COPY_FILE "${REPOSITORY_ROOT}/${relative}" "${mirror}/${relative}")
endforeach()
file(APPEND "${mirror}/${final_rel}" "x")
invoke("${mirror}" "${root}/bad_final" 128 reject)

set(spec_path "${mirror}/${h_rel}/h_spec.txt")
file(READ "${REPOSITORY_ROOT}/${h_rel}/h_spec.txt" canonical_spec)
function(reject_spec_change old new name)
    string(REPLACE "${old}" "${new}" changed "${canonical_spec}")
    if(changed STREQUAL canonical_spec)
        message(FATAL_ERROR "mini224 H spec test precondition absent: ${old}")
    endif()
    file(WRITE "${spec_path}" "${changed}")
    invoke("${mirror}" "${root}/${name}" 128 reject)
endfunction()
file(COPY_FILE "${REPOSITORY_ROOT}/${final_rel}" "${mirror}/${final_rel}")
reject_spec_change("rows=1000" "rows=1001" "bad_n")
reject_spec_change("cols=180" "cols=179" "bad_cols")
reject_spec_change("K=5" "K=4" "bad_k")
reject_spec_change("C=36" "C=35" "bad_c")
reject_spec_change("label_free=true" "label_free=false" "bad_label_free")
reject_spec_change("format=VAE_C_IIC_H_ARTIFACT_V1"
                   "format=INVALID" "bad_format")

message(STATUS "mini224 affinity producer PASS; payload_sha256=${first_sha}")
