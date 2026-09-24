cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED TEST_EXECUTABLE OR
   NOT DEFINED REPOSITORY_ROOT OR
   NOT DEFINED FIXTURE_ROOT)
    message(FATAL_ERROR
        "TEST_EXECUTABLE, REPOSITORY_ROOT, and FIXTURE_ROOT are required"
    )
endif()

get_filename_component(repository_absolute "${REPOSITORY_ROOT}" ABSOLUTE)
get_filename_component(fixture_absolute "${FIXTURE_ROOT}" ABSOLUTE)
get_filename_component(fixture_leaf "${fixture_absolute}" NAME)
file(TO_CMAKE_PATH "${repository_absolute}" repository_normalized)
file(TO_CMAKE_PATH "${fixture_absolute}" fixture_normalized)
string(TOLOWER "${repository_normalized}/" repository_prefix)
string(TOLOWER "${fixture_normalized}/" fixture_prefix)
string(FIND "${fixture_prefix}" "${repository_prefix}" fixture_in_repository)
if(fixture_leaf STREQUAL "" OR
   NOT fixture_leaf STREQUAL "test_vae_c_mu_kmeans_validate_fixtures" OR
   fixture_in_repository EQUAL 0)
    message(FATAL_ERROR
        "fixture cleanup target is not the dedicated external-build path: "
        "${fixture_absolute}"
    )
endif()

set(output_files
    centroids_f64.bin
    train_assignments_u32.bin
    validation_assignments_u32.bin
    train_squared_distances_f64.bin
    validation_squared_distances_f64.bin
    cluster_sizes_train_u64.bin
    initial_centroid_indices_u64.bin
    train_assignments.csv
    validation_assignments.csv
    cluster_sizes.csv
    initial_centroids.csv
    metadata.txt
    sha256.txt
)

function(require_line output expected context)
    string(REPLACE "\r" "" normalized "${output}")
    string(REPLACE "\n" ";" lines "${normalized}")
    list(FIND lines "${expected}" position)
    if(position EQUAL -1)
        message(FATAL_ERROR
            "missing line ${expected} in ${context}\n${output}"
        )
    endif()
endfunction()

function(run_default_validate fixture_directory)
    execute_process(
        COMMAND
            "${TEST_EXECUTABLE}"
            "${REPOSITORY_ROOT}"
            "${fixture_directory}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
    )
    if(NOT result EQUAL 0)
        message(FATAL_ERROR
            "validate-only synthetic test failed: exit=${result}\n"
            "stdout=${output}\nstderr=${error}"
        )
    endif()
    require_line(
        "${output}"
        "VAE_C_MU_KMEANS_VALIDATE_TEST_PASS=true"
        "validate-only synthetic test"
    )
endfunction()

function(run_productive_case mode output_name result_name)
    execute_process(
        COMMAND
            "${TEST_EXECUTABLE}"
            "${mode}"
            "${REPOSITORY_ROOT}"
            "${FIXTURE_ROOT}"
            "${output_name}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
    )
    if(NOT result EQUAL 0)
        message(FATAL_ERROR
            "productive synthetic case ${mode}/${output_name} failed: "
            "exit=${result}\nstdout=${output}\nstderr=${error}"
        )
    endif()
    set(${result_name} "${output}" PARENT_SCOPE)
endfunction()

function(verify_exact_inventory directory)
    if(NOT IS_DIRECTORY "${directory}")
        message(FATAL_ERROR "missing published directory: ${directory}")
    endif()
    file(GLOB entries LIST_DIRECTORIES TRUE "${directory}/*")
    list(LENGTH entries entry_count)
    list(LENGTH output_files expected_count)
    if(NOT entry_count EQUAL expected_count)
        message(FATAL_ERROR
            "unexpected output inventory count in ${directory}: "
            "${entry_count}, expected ${expected_count}"
        )
    endif()
    foreach(name IN LISTS output_files)
        set(path "${directory}/${name}")
        if(NOT EXISTS "${path}" OR IS_DIRECTORY "${path}")
            message(FATAL_ERROR "missing output file: ${path}")
        endif()
        file(SHA256 "${path}" digest)
        string(LENGTH "${digest}" digest_length)
        if(NOT digest_length EQUAL 64 OR
           NOT "${digest}" MATCHES "^[0-9a-f]+$")
            message(FATAL_ERROR "invalid SHA-256 for ${path}: ${digest}")
        endif()
    endforeach()
endfunction()

function(compare_published_outputs left right)
    foreach(name IN LISTS output_files)
        set(left_path "${left}/${name}")
        set(right_path "${right}/${name}")
        execute_process(
            COMMAND
                "${CMAKE_COMMAND}" -E compare_files
                "${left_path}" "${right_path}"
            RESULT_VARIABLE compare_result
        )
        if(NOT compare_result EQUAL 0)
            message(FATAL_ERROR
                "fresh-process output mismatch for ${name}"
            )
        endif()
        file(SHA256 "${left_path}" left_hash)
        file(SHA256 "${right_path}" right_hash)
        if(NOT left_hash STREQUAL right_hash)
            message(FATAL_ERROR
                "fresh-process SHA-256 mismatch for ${name}"
            )
        endif()
    endforeach()
endfunction()

# This is the one explicitly allowlisted cleanup root owned by this CTest.
file(REMOVE_RECURSE "${fixture_absolute}")
file(MAKE_DIRECTORY "${fixture_absolute}")

run_default_validate("${fixture_absolute}/validate")

run_productive_case(--productive-publish replica_a publish_a_output)
run_productive_case(--productive-publish replica_b publish_b_output)
foreach(output_name IN ITEMS publish_a_output publish_b_output)
    require_line(
        "${${output_name}}" "SYNTHETIC_PUBLISH_CONVERGED=true"
        "${output_name}"
    )
    require_line(
        "${${output_name}}" "SYNTHETIC_OUTPUT_INVENTORY=13"
        "${output_name}"
    )
    require_line(
        "${${output_name}}" "SYNTHETIC_MATCHED_REFIT_PREFIX_800=true"
        "${output_name}"
    )
    require_line(
        "${${output_name}}" "SYNTHETIC_VALIDATION_FROZEN_CENTROIDS=true"
        "${output_name}"
    )
    require_line(
        "${${output_name}}" "SYNTHETIC_INPUT_UNCHANGED=true"
        "${output_name}"
    )
    require_line(
        "${${output_name}}" "SYNTHETIC_PUBLISH_PASS=true"
        "${output_name}"
    )
endforeach()

set(replica_a "${fixture_absolute}/replica_a")
set(replica_b "${fixture_absolute}/replica_b")
verify_exact_inventory("${replica_a}")
verify_exact_inventory("${replica_b}")
if(EXISTS "${replica_a}.partial" OR EXISTS "${replica_b}.partial")
    message(FATAL_ERROR "residual partial after successful publication")
endif()
compare_published_outputs("${replica_a}" "${replica_b}")

run_productive_case(--productive-overwrite replica_a overwrite_output)
require_line(
    "${overwrite_output}" "SYNTHETIC_OVERWRITE_REFUSED=true"
    "overwrite refusal"
)
require_line(
    "${overwrite_output}" "SYNTHETIC_OVERWRITE_FIT_CALLS=0"
    "overwrite refusal"
)
require_line(
    "${overwrite_output}" "SYNTHETIC_OVERWRITE_OUTPUT_UNCHANGED=true"
    "overwrite refusal"
)
compare_published_outputs("${replica_a}" "${replica_b}")

run_productive_case(
    --productive-nonconverged nonconverged nonconverged_output
)
require_line(
    "${nonconverged_output}" "SYNTHETIC_NONCONVERGED=1"
    "non-convergence publication gate"
)
require_line(
    "${nonconverged_output}"
    "SYNTHETIC_NONCONVERGED_OUTPUT_CREATED=false"
    "non-convergence publication gate"
)
if(EXISTS "${fixture_absolute}/nonconverged" OR
   EXISTS "${fixture_absolute}/nonconverged.partial")
    message(FATAL_ERROR "non-converged case left final or partial output")
endif()

run_productive_case(--productive-invalid invalid invalid_output)
require_line(
    "${invalid_output}" "SYNTHETIC_INVALID_INPUT_REJECTED=true"
    "invalid-input publication gate"
)
require_line(
    "${invalid_output}" "SYNTHETIC_INVALID_INPUT_OUTPUT_CREATED=false"
    "invalid-input publication gate"
)
if(EXISTS "${fixture_absolute}/invalid" OR
   EXISTS "${fixture_absolute}/invalid.partial")
    message(FATAL_ERROR "invalid-input case left final or partial output")
endif()

run_productive_case(--productive-overlap forbidden_child overlap_output)
require_line(
    "${overlap_output}"
    "SYNTHETIC_SOURCE_OUTPUT_OVERLAP_REFUSED=true"
    "source/output overlap gate"
)
require_line(
    "${overlap_output}"
    "SYNTHETIC_SOURCE_OUTPUT_OVERLAP_FIT_CALLS=0"
    "source/output overlap gate"
)
require_line(
    "${overlap_output}"
    "SYNTHETIC_SOURCE_INVENTORY_UNCHANGED=true"
    "source/output overlap gate"
)
if(EXISTS "${fixture_absolute}/productive_source/forbidden_child" OR
   EXISTS "${fixture_absolute}/productive_source/forbidden_child.partial")
    message(FATAL_ERROR "source/output overlap case created output")
endif()

set(forbidden_repository_child
    "${repository_absolute}/matched_subset1000_synthetic_productive_k22_seed123_maxiter300"
)
if(EXISTS "${forbidden_repository_child}" OR
   EXISTS "${forbidden_repository_child}.partial")
    message(FATAL_ERROR
        "repository-root confinement fixture target already exists"
    )
endif()
run_productive_case(
    --productive-repository-root forbidden_repository_root
    repository_root_output
)
require_line(
    "${repository_root_output}"
    "SYNTHETIC_REPOSITORY_OUTPUT_ROOT_REFUSED=true"
    "repository output-root confinement gate"
)
require_line(
    "${repository_root_output}"
    "SYNTHETIC_REPOSITORY_OUTPUT_ROOT_FIT_CALLS=0"
    "repository output-root confinement gate"
)
require_line(
    "${repository_root_output}"
    "SYNTHETIC_REPOSITORY_OUTPUT_ROOT_SOURCE_UNCHANGED=true"
    "repository output-root confinement gate"
)
if(EXISTS "${forbidden_repository_child}" OR
   EXISTS "${forbidden_repository_child}.partial" OR
   EXISTS "${fixture_absolute}/forbidden_repository_root" OR
   EXISTS "${fixture_absolute}/forbidden_repository_root.partial")
    message(FATAL_ERROR
        "repository output-root confinement case created output"
    )
endif()

message("SYNTHETIC_FRESH_PROCESS_DETERMINISM=true")
message("SYNTHETIC_OUTPUT_SHA256_VERIFIED=true")
message("SYNTHETIC_REPOSITORY_OUTPUT_ROOT_CONFINED=true")
message("SYNTHETIC_PRODUCTIVE_KMEANS_TEST_PASS=true")

# Leave the external build tree clean and make a second CTest run identical.
file(REMOVE_RECURSE "${fixture_absolute}")
