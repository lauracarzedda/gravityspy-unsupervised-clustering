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
get_filename_component(fixture_parent "${fixture_absolute}" DIRECTORY)
file(TO_CMAKE_PATH "${repository_absolute}" repository_normalized)
file(TO_CMAKE_PATH "${fixture_absolute}" fixture_normalized)
string(TOLOWER "${repository_normalized}/" repository_prefix)
string(TOLOWER "${fixture_normalized}/" fixture_prefix)
string(FIND "${fixture_prefix}" "${repository_prefix}" fixture_in_repository)
string(FIND "${repository_prefix}" "${fixture_prefix}" repository_in_fixture)
if(fixture_leaf STREQUAL "" OR
   NOT fixture_leaf STREQUAL
       "test_vae_c_mu_kmeans_evaluator_productive_fixtures" OR
   fixture_in_repository EQUAL 0 OR repository_in_fixture EQUAL 0 OR
   NOT IS_DIRECTORY "${fixture_parent}")
    message(FATAL_ERROR
        "fixture cleanup target is not the dedicated external-build path: "
        "${fixture_absolute}"
    )
endif()

set(results_root
    "${repository_absolute}/sakai_multires_c/results/vae_c/kmeans_mu_raw_v1"
)
set(expected_directories
    full_full224_k22_seed123_maxiter300
    full_full224_k36_seed123_maxiter300
    full_mini112_k22_seed123_maxiter300
    full_mini112_k36_seed123_maxiter300
    full_mini75_k22_seed123_maxiter300
    full_mini75_k36_seed123_maxiter300
    matched_subset1000_full224_k22_seed123_maxiter300
    matched_subset1000_full224_k36_seed123_maxiter300
    matched_subset1000_mini112_k22_seed123_maxiter300
    matched_subset1000_mini112_k36_seed123_maxiter300
    matched_subset1000_mini224_k22_seed123_maxiter300
    matched_subset1000_mini224_k36_seed123_maxiter300
    matched_subset1000_mini75_k22_seed123_maxiter300
    matched_subset1000_mini75_k36_seed123_maxiter300
)
set(source_files
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
set(evaluation_files
    label_map.csv
    metrics.csv
    train_contingency_counts.csv
    validation_contingency_counts.csv
    evaluator_source.c
    metadata.txt
    sha256.txt
)
set(evaluation_hashed_artifacts
    label_map.csv
    metrics.csv
    train_contingency_counts.csv
    validation_contingency_counts.csv
    evaluator_source.c
    metadata.txt
)
set(evaluation_source_names
    metadata.txt
    sha256.txt
    train_assignments.csv
    validation_assignments.csv
    train_assignments_u32.bin
    validation_assignments_u32.bin
)
set(evaluation_canonical_names
    manifest_events_272.csv
    split_train_indices_seed123.csv
    split_val_indices_seed123.csv
    sakai_multires_c/configs/subset224_train_800_seed123.csv
    sakai_multires_c/configs/subset224_val_200_seed123.csv
)
set(evaluation_code_names
    sakai_multires_c/tools/evaluate_vae_c_mu_kmeans.c
    sakai_multires_c/src/baseline_clustering_metrics.c
    sakai_multires_c/include/baseline_clustering_metrics.h
    vae_event_dataset.c
    vae_event_dataset.h
    vae_saved_split.c
    vae_saved_split.h
)
set(canonical_labels
    1080Lines
    1400Ripples
    Air_Compressor
    Blip
    Chirp
    Extremely_Loud
    Helix
    Koi_Fish
    Light_Modulation
    Low_Frequency_Burst
    Low_Frequency_Lines
    No_Glitch
    None_of_the_Above
    Paired_Doves
    Power_Line
    Repeating_Blips
    Scattered_Light
    Scratchy
    Tomte
    Violin_Mode
    Wandering_Line
    Whistle
)

function(require_line output expected context)
    string(REPLACE "\r" "" normalized "${output}")
    string(REPLACE "\n" ";" lines "${normalized}")
    list(FIND lines "${expected}" position)
    if(position EQUAL -1)
        message(FATAL_ERROR
            "missing line '${expected}' in ${context}\n${output}"
        )
    endif()
endfunction()

function(extract_unique_value output key context output_variable)
    string(REPLACE "\r" "" normalized "${output}")
    string(REPLACE "\n" ";" lines "${normalized}")
    set(matches "")
    foreach(line IN LISTS lines)
        if(line MATCHES "^${key}=(.*)$")
            list(APPEND matches "${CMAKE_MATCH_1}")
        endif()
    endforeach()
    list(LENGTH matches count)
    if(NOT count EQUAL 1)
        message(FATAL_ERROR
            "expected exactly one ${key}= line in ${context}\n${output}"
        )
    endif()
    list(GET matches 0 value)
    set(${output_variable} "${value}" PARENT_SCOPE)
endfunction()

function(assert_no_transients root context)
    if(NOT IS_DIRECTORY "${root}")
        message(FATAL_ERROR "missing directory for ${context}: ${root}")
    endif()
    file(GLOB_RECURSE entries
        LIST_DIRECTORIES TRUE
        RELATIVE "${root}"
        "${root}/*"
    )
    foreach(relative_path IN LISTS entries)
        string(TOLOWER "${relative_path}" lower_path)
        if(lower_path MATCHES "(^|/)[^/]*[.]partial($|/)" OR
           lower_path MATCHES "(^|/)[^/]*[.]tmp($|/)")
            message(FATAL_ERROR
                "residual partial/tmp in ${context}: ${relative_path}"
            )
        endif()
    endforeach()
endfunction()

function(verify_exact_results_inventory)
    if(NOT IS_DIRECTORY "${results_root}")
        message(FATAL_ERROR "missing official results root: ${results_root}")
    endif()
    file(GLOB root_entries
        LIST_DIRECTORIES TRUE
        RELATIVE "${results_root}"
        "${results_root}/*"
    )
    list(SORT root_entries)
    set(sorted_expected "${expected_directories}")
    list(SORT sorted_expected)
    if(NOT "${root_entries}" STREQUAL "${sorted_expected}")
        message(FATAL_ERROR
            "official results root does not contain exactly the expected "
            "14 directories\nactual=${root_entries}\n"
            "expected=${sorted_expected}"
        )
    endif()
    set(sorted_source_files "${source_files}")
    list(SORT sorted_source_files)
    foreach(directory IN LISTS expected_directories)
        set(directory_path "${results_root}/${directory}")
        if(NOT IS_DIRECTORY "${directory_path}")
            message(FATAL_ERROR "not an official directory: ${directory_path}")
        endif()
        file(GLOB children
            LIST_DIRECTORIES TRUE
            RELATIVE "${directory_path}"
            "${directory_path}/*"
        )
        list(SORT children)
        if(NOT "${children}" STREQUAL "${sorted_source_files}")
            message(FATAL_ERROR
                "official directory inventory mismatch: ${directory_path}\n"
                "actual=${children}\nexpected=${sorted_source_files}"
            )
        endif()
        foreach(name IN LISTS source_files)
            if(IS_DIRECTORY "${directory_path}/${name}")
                message(FATAL_ERROR
                    "official output entry is a directory: "
                    "${directory_path}/${name}"
                )
            endif()
        endforeach()
    endforeach()
    assert_no_transients("${results_root}" "official k-means inputs")
endfunction()

function(snapshot_results output_snapshot output_count output_bytes)
    if(NOT IS_DIRECTORY "${results_root}")
        message(FATAL_ERROR "missing official results root: ${results_root}")
    endif()
    file(GLOB_RECURSE entries
        LIST_DIRECTORIES TRUE
        RELATIVE "${results_root}"
        "${results_root}/*"
    )
    list(SORT entries)
    set(file_count 0)
    set(total_bytes 0)
    set(snapshot "")
    foreach(relative_path IN LISTS entries)
        set(path "${results_root}/${relative_path}")
        if(IS_DIRECTORY "${path}")
            string(APPEND snapshot "D|${relative_path}\n")
        else()
            math(EXPR file_count "${file_count} + 1")
            file(SIZE "${path}" size)
            file(SHA256 "${path}" digest)
            math(EXPR total_bytes "${total_bytes} + ${size}")
            string(APPEND snapshot
                "F|${relative_path}|${size}|${digest}\n"
            )
        endif()
    endforeach()
    if(NOT file_count EQUAL 182)
        message(FATAL_ERROR
            "unexpected official artifact count: ${file_count}, expected 182"
        )
    endif()
    set(${output_snapshot} "${snapshot}" PARENT_SCOPE)
    set(${output_count} "${file_count}" PARENT_SCOPE)
    set(${output_bytes} "${total_bytes}" PARENT_SCOPE)
endfunction()

function(require_finite_metric value minimum maximum context)
    string(TOLOWER "${value}" lower_value)
    if(lower_value MATCHES "nan|inf" OR
       NOT "${value}" MATCHES
           "^[+-]?[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$")
        message(FATAL_ERROR "non-finite/invalid metric ${value} in ${context}")
    endif()
    if("${value}" LESS "${minimum}" OR "${value}" GREATER "${maximum}")
        message(FATAL_ERROR
            "metric outside [${minimum},${maximum}] in ${context}: ${value}"
        )
    endif()
endfunction()

function(verify_label_map directory)
    set(path "${directory}/label_map.csv")
    file(STRINGS "${path}" lines)
    list(LENGTH lines count)
    if(NOT count EQUAL 23)
        message(FATAL_ERROR "label_map.csv line count mismatch: ${path}")
    endif()
    list(GET lines 0 header)
    if(NOT header STREQUAL "label_index,label")
        message(FATAL_ERROR "label_map.csv header mismatch: ${path}")
    endif()
    foreach(label_index RANGE 0 21)
        math(EXPR line_index "${label_index} + 1")
        list(GET canonical_labels ${label_index} label_name)
        list(GET lines ${line_index} actual)
        if(NOT actual STREQUAL "${label_index},${label_name}")
            message(FATAL_ERROR
                "label_map.csv row mismatch at ${label_index}: ${path}"
            )
        endif()
    endforeach()
endfunction()

function(verify_metrics_csv directory k train_rows validation_rows)
    set(path "${directory}/metrics.csv")
    file(STRINGS "${path}" lines)
    list(LENGTH lines count)
    if(NOT count EQUAL 3)
        message(FATAL_ERROR "metrics.csv must have header + 2 rows: ${path}")
    endif()
    list(GET lines 0 header)
    if(NOT header STREQUAL
       "split,rows,label_count,cluster_count,purity,nmi_arithmetic,ari,homogeneity,completeness")
        message(FATAL_ERROR "metrics.csv header mismatch: ${path}")
    endif()
    foreach(line_index RANGE 1 2)
        if(line_index EQUAL 1)
            set(expected_split train)
            set(expected_rows "${train_rows}")
        else()
            set(expected_split validation)
            set(expected_rows "${validation_rows}")
        endif()
        list(GET lines ${line_index} line)
        string(REPLACE "," ";" fields "${line}")
        list(LENGTH fields field_count)
        if(NOT field_count EQUAL 9)
            message(FATAL_ERROR "metrics.csv field count mismatch: ${path}")
        endif()
        list(GET fields 0 split)
        list(GET fields 1 rows)
        list(GET fields 2 label_count)
        list(GET fields 3 cluster_count)
        if(NOT split STREQUAL expected_split OR
           NOT rows STREQUAL expected_rows OR
           NOT label_count STREQUAL "22" OR
           NOT cluster_count STREQUAL "${k}")
            message(FATAL_ERROR
                "metrics.csv identity/shape mismatch in ${path}: ${line}"
            )
        endif()
        foreach(metric_index IN ITEMS 4 5 7 8)
            list(GET fields ${metric_index} value)
            require_finite_metric("${value}" 0 1 "${path}/${split}")
        endforeach()
        list(GET fields 6 ari)
        require_finite_metric("${ari}" -1 1 "${path}/${split}/ari")
    endforeach()
endfunction()

function(verify_contingency_csv path k expected_rows require_wandering_zero)
    file(STRINGS "${path}" lines)
    list(LENGTH lines line_count)
    if(NOT line_count EQUAL 23)
        message(FATAL_ERROR
            "contingency must have header + 22 rows: ${path}"
        )
    endif()
    set(expected_header "label_index,label")
    math(EXPR last_cluster "${k} - 1")
    foreach(cluster RANGE 0 ${last_cluster})
        string(APPEND expected_header ",cluster_${cluster}")
    endforeach()
    string(APPEND expected_header ",row_total")
    list(GET lines 0 actual_header)
    if(NOT actual_header STREQUAL expected_header)
        message(FATAL_ERROR "contingency header mismatch: ${path}")
    endif()
    set(grand_total 0)
    math(EXPR expected_field_count "${k} + 3")
    math(EXPR row_total_field "${k} + 2")
    foreach(label_index RANGE 0 21)
        math(EXPR line_index "${label_index} + 1")
        list(GET lines ${line_index} line)
        string(REPLACE "," ";" fields "${line}")
        list(LENGTH fields field_count)
        if(NOT field_count EQUAL expected_field_count)
            message(FATAL_ERROR
                "contingency field count mismatch at label ${label_index}: ${path}"
            )
        endif()
        list(GET fields 0 actual_label_index)
        list(GET fields 1 actual_label)
        list(GET canonical_labels ${label_index} expected_label)
        if(NOT actual_label_index STREQUAL "${label_index}" OR
           NOT actual_label STREQUAL expected_label)
            message(FATAL_ERROR
                "contingency label identity mismatch at ${label_index}: ${path}"
            )
        endif()
        set(row_sum 0)
        foreach(cluster RANGE 0 ${last_cluster})
            math(EXPR field_index "${cluster} + 2")
            list(GET fields ${field_index} value)
            if(NOT "${value}" MATCHES "^[0-9]+$")
                message(FATAL_ERROR "invalid contingency count in ${path}")
            endif()
            math(EXPR row_sum "${row_sum} + ${value}")
        endforeach()
        list(GET fields ${row_total_field} declared_row_total)
        if(NOT "${declared_row_total}" MATCHES "^[0-9]+$" OR
           NOT row_sum EQUAL declared_row_total)
            message(FATAL_ERROR
                "contingency row total mismatch at ${label_index}: ${path}"
            )
        endif()
        if(require_wandering_zero AND label_index EQUAL 20 AND
           NOT row_sum EQUAL 0)
            message(FATAL_ERROR
                "matched validation Wandering_Line row is not zero: ${path}"
            )
        endif()
        math(EXPR grand_total "${grand_total} + ${row_sum}")
    endforeach()
    if(NOT grand_total EQUAL expected_rows)
        message(FATAL_ERROR
            "contingency grand total mismatch in ${path}: "
            "${grand_total}, expected ${expected_rows}"
        )
    endif()
endfunction()

function(verify_metadata
    directory scope profile k train_rows validation_rows repository_head
)
    set(path "${directory}/metadata.txt")
    if(profile STREQUAL "mini75")
        set(profile_id mini75_experiment_v1)
    elseif(profile STREQUAL "mini112")
        set(profile_id mini112_primary_v1)
    elseif(profile STREQUAL "mini224")
        set(profile_id mini224_bridge_v1)
    elseif(profile STREQUAL "full224")
        set(profile_id sakai224_full_v1)
    else()
        message(FATAL_ERROR "unexpected profile in metadata check: ${profile}")
    endif()
    if(scope STREQUAL "full")
        set(population_identity gravity_spy_o1_full_split_seed123_v1)
    else()
        set(population_identity
            canonical_prefix_800_train_200_validation_seed123_v1
        )
    endif()
    set(source_bundle
        "sakai_multires_c/results/vae_c/kmeans_mu_raw_v1/${scope}_${profile}_k${k}_seed123_maxiter300"
    )
    set(source_hashes "")
    foreach(name IN LISTS evaluation_source_names)
        set(input_path "${repository_absolute}/${source_bundle}/${name}")
        if(NOT EXISTS "${input_path}" OR IS_DIRECTORY "${input_path}")
            message(FATAL_ERROR "missing metadata source input: ${input_path}")
        endif()
        file(SHA256 "${input_path}" digest)
        list(APPEND source_hashes "${digest}")
    endforeach()
    set(canonical_hashes "")
    foreach(name IN LISTS evaluation_canonical_names)
        set(input_path "${repository_absolute}/${name}")
        if(NOT EXISTS "${input_path}" OR IS_DIRECTORY "${input_path}")
            message(FATAL_ERROR "missing metadata canonical input: ${input_path}")
        endif()
        file(SHA256 "${input_path}" digest)
        list(APPEND canonical_hashes "${digest}")
    endforeach()
    set(code_hashes "")
    foreach(name IN LISTS evaluation_code_names)
        set(input_path "${repository_absolute}/${name}")
        if(NOT EXISTS "${input_path}" OR IS_DIRECTORY "${input_path}")
            message(FATAL_ERROR "missing metadata code input: ${input_path}")
        endif()
        file(SHA256 "${input_path}" digest)
        list(APPEND code_hashes "${digest}")
    endforeach()
    list(GET source_hashes 0 source_metadata_hash)
    list(GET source_hashes 1 source_checksum_hash)
    list(GET source_hashes 2 source_train_csv_hash)
    list(GET source_hashes 3 source_validation_csv_hash)
    list(GET source_hashes 4 source_train_bin_hash)
    list(GET source_hashes 5 source_validation_bin_hash)
    list(GET canonical_hashes 0 manifest_hash)
    list(GET canonical_hashes 1 full_train_hash)
    list(GET canonical_hashes 2 full_validation_hash)
    list(GET canonical_hashes 3 matched_train_hash)
    list(GET canonical_hashes 4 matched_validation_hash)
    list(GET code_hashes 0 evaluator_hash)
    list(GET code_hashes 1 metrics_source_hash)
    list(GET code_hashes 2 metrics_header_hash)
    list(GET code_hashes 3 event_source_hash)
    list(GET code_hashes 4 event_header_hash)
    list(GET code_hashes 5 split_source_hash)
    list(GET code_hashes 6 split_header_hash)
    file(STRINGS "${path}" lines)
    list(LENGTH lines line_count)
    if(NOT line_count EQUAL 45)
        message(FATAL_ERROR "metadata.txt line count mismatch: ${path}")
    endif()
    set(expected_lines
        "format=VAE_C_MU_KMEANS_LABEL_EVALUATION_V1"
        "scope=${scope}"
        "profile=${profile}"
        "profile_id=${profile_id}"
        "source_kmeans_bundle=${source_bundle}"
        "population_identity=${population_identity}"
        "row_order=native_split_pos_ascending"
        "row_identity=split_pos,manifest_index,event_index,event_id,split"
        "label_source=manifest_events_272.csv"
        "label_source_sha256=${manifest_hash}"
        "label_join=manifest_index_cross_checked_event_id_event_index_split_split_pos"
        "labels_used_for_clustering=false"
        "labels_used_for_evaluation=true"
        "assignments_modified=false"
        "selection_post_hoc=false"
        "label_count=22"
        "k=${k}"
        "train_rows=${train_rows}"
        "validation_rows=${validation_rows}"
        "contingency_layout=label_major_[22,k]"
        "metrics=purity,nmi_arithmetic,ari,homogeneity,completeness"
        "nmi_average_method=arithmetic"
        "source_metadata_sha256=${source_metadata_hash}"
        "source_checksum_sha256=${source_checksum_hash}"
        "source_train_assignments_csv_sha256=${source_train_csv_hash}"
        "source_validation_assignments_csv_sha256=${source_validation_csv_hash}"
        "source_train_assignments_bin_sha256=${source_train_bin_hash}"
        "source_validation_assignments_bin_sha256=${source_validation_bin_hash}"
        "full_train_split_sha256=${full_train_hash}"
        "full_validation_split_sha256=${full_validation_hash}"
        "matched_train_subset_sha256=${matched_train_hash}"
        "matched_validation_subset_sha256=${matched_validation_hash}"
        "repository_head=${repository_head}"
        "evaluator_source_snapshot=evaluator_source.c"
        "evaluator_source_provenance=embedded_exact_copy_sha256"
        "evaluator_source_sha256=${evaluator_hash}"
        "metrics_backend_source_sha256=${metrics_source_hash}"
        "metrics_backend_header_sha256=${metrics_header_hash}"
        "event_dataset_source_sha256=${event_source_hash}"
        "event_dataset_header_sha256=${event_header_hash}"
        "saved_split_source_sha256=${split_source_hash}"
        "saved_split_header_sha256=${split_header_hash}"
        "output_file_count=7"
        "checksum_manifest_excludes_self=true"
        "deterministic_serialization=c_locale_lf_17g"
    )
    if(NOT "${lines}" STREQUAL "${expected_lines}")
        message(FATAL_ERROR
            "metadata.txt exact ordered key/value contract mismatch: ${path}\n"
            "actual=${lines}\nexpected=${expected_lines}"
        )
    endif()
endfunction()

function(verify_checksum_manifest directory)
    set(path "${directory}/sha256.txt")
    file(STRINGS "${path}" lines)
    list(LENGTH lines line_count)
    if(NOT line_count EQUAL 24)
        message(FATAL_ERROR "sha256.txt line count mismatch: ${path}")
    endif()
    get_filename_component(configuration_name "${directory}" NAME)
    list(FIND expected_directories "${configuration_name}" config_position)
    if(config_position EQUAL -1)
        message(FATAL_ERROR
            "sha256.txt belongs to an unexpected configuration: ${directory}"
        )
    endif()
    set(source_prefix
        "sakai_multires_c/results/vae_c/kmeans_mu_raw_v1/${configuration_name}"
    )
    set(record_keys "")
    foreach(line IN LISTS lines)
        string(REPLACE " " ";" fields "${line}")
        list(LENGTH fields field_count)
        if(NOT field_count EQUAL 3)
            message(FATAL_ERROR "malformed sha256.txt record: ${line}")
        endif()
        list(GET fields 0 category)
        list(GET fields 1 digest)
        list(GET fields 2 name)
        string(LENGTH "${digest}" digest_length)
        if(NOT category MATCHES "^(SOURCE|CANONICAL|CODE|ARTIFACT)$" OR
           NOT digest_length EQUAL 64 OR NOT digest MATCHES "^[0-9a-f]+$" OR
           name STREQUAL "")
            message(FATAL_ERROR "invalid sha256.txt record: ${line}")
        endif()
        list(APPEND record_keys "${category}|${name}")
    endforeach()
    set(unique_record_keys "${record_keys}")
    list(REMOVE_DUPLICATES unique_record_keys)
    list(LENGTH unique_record_keys unique_record_count)
    if(NOT unique_record_count EQUAL 24)
        message(FATAL_ERROR
            "sha256.txt contains duplicate category/name records: ${path}"
        )
    endif()
    set(expected_lines "")
    foreach(name IN LISTS evaluation_source_names)
        set(relative_path "${source_prefix}/${name}")
        set(input_path "${repository_absolute}/${relative_path}")
        if(NOT EXISTS "${input_path}" OR IS_DIRECTORY "${input_path}")
            message(FATAL_ERROR
                "SOURCE escapes/misses the exact configuration: ${input_path}"
            )
        endif()
        file(SHA256 "${input_path}" digest)
        list(APPEND expected_lines "SOURCE ${digest} ${relative_path}")
    endforeach()
    foreach(name IN LISTS evaluation_canonical_names)
        set(input_path "${repository_absolute}/${name}")
        if(NOT EXISTS "${input_path}" OR IS_DIRECTORY "${input_path}")
            message(FATAL_ERROR "missing exact CANONICAL path: ${input_path}")
        endif()
        file(SHA256 "${input_path}" digest)
        list(APPEND expected_lines "CANONICAL ${digest} ${name}")
    endforeach()
    foreach(name IN LISTS evaluation_code_names)
        set(input_path "${repository_absolute}/${name}")
        if(NOT EXISTS "${input_path}" OR IS_DIRECTORY "${input_path}")
            message(FATAL_ERROR "missing exact CODE path: ${input_path}")
        endif()
        file(SHA256 "${input_path}" digest)
        list(APPEND expected_lines "CODE ${digest} ${name}")
    endforeach()
    foreach(name IN LISTS evaluation_hashed_artifacts)
        set(artifact_path "${directory}/${name}")
        if(NOT EXISTS "${artifact_path}" OR IS_DIRECTORY "${artifact_path}")
            message(FATAL_ERROR "missing exact ARTIFACT path: ${artifact_path}")
        endif()
        file(SHA256 "${artifact_path}" digest)
        list(APPEND expected_lines "ARTIFACT ${digest} ${name}")
    endforeach()
    if(NOT "${lines}" STREQUAL "${expected_lines}")
        message(FATAL_ERROR
            "sha256.txt exact ordered category/name/hash contract mismatch: "
            "${path}\nactual=${lines}\nexpected=${expected_lines}"
        )
    endif()
endfunction()

function(verify_evaluator_source_snapshot directory)
    set(source
        "${repository_absolute}/sakai_multires_c/tools/evaluate_vae_c_mu_kmeans.c"
    )
    set(snapshot "${directory}/evaluator_source.c")
    if(NOT EXISTS "${source}" OR IS_DIRECTORY "${source}" OR
       NOT EXISTS "${snapshot}" OR IS_DIRECTORY "${snapshot}")
        message(FATAL_ERROR
            "missing evaluator source or embedded snapshot: ${directory}"
        )
    endif()
    file(SIZE "${source}" source_size)
    file(SIZE "${snapshot}" snapshot_size)
    file(SHA256 "${source}" source_hash)
    file(SHA256 "${snapshot}" snapshot_hash)
    if(NOT source_size EQUAL snapshot_size OR
       NOT source_hash STREQUAL snapshot_hash)
        message(FATAL_ERROR
            "evaluator source snapshot size/SHA-256 mismatch: ${directory}"
        )
    endif()
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E compare_files "${source}" "${snapshot}"
        RESULT_VARIABLE compare_result
    )
    if(NOT compare_result EQUAL 0)
        message(FATAL_ERROR
            "evaluator source snapshot is not byte-identical: ${directory}"
        )
    endif()
endfunction()

function(parse_evaluation_directory directory output_scope output_profile output_k)
    if(directory MATCHES
       "^full_(mini75|mini112|full224)_k(22|36)_seed123_maxiter300$")
        set(scope full)
        set(profile "${CMAKE_MATCH_1}")
        set(k "${CMAKE_MATCH_2}")
    elseif(directory MATCHES
       "^matched_subset1000_(mini75|mini112|mini224|full224)_k(22|36)_seed123_maxiter300$")
        set(scope matched_subset1000)
        set(profile "${CMAKE_MATCH_1}")
        set(k "${CMAKE_MATCH_2}")
    else()
        message(FATAL_ERROR "unexpected evaluation directory name: ${directory}")
    endif()
    set(${output_scope} "${scope}" PARENT_SCOPE)
    set(${output_profile} "${profile}" PARENT_SCOPE)
    set(${output_k} "${k}" PARENT_SCOPE)
endfunction()

function(verify_evaluation_root root repository_head output_bytes)
    if(NOT IS_DIRECTORY "${root}")
        message(FATAL_ERROR "missing evaluation fixture root: ${root}")
    endif()
    file(GLOB root_entries
        LIST_DIRECTORIES TRUE
        RELATIVE "${root}"
        "${root}/*"
    )
    list(SORT root_entries)
    set(sorted_expected "${expected_directories}")
    list(SORT sorted_expected)
    if(NOT "${root_entries}" STREQUAL "${sorted_expected}")
        message(FATAL_ERROR
            "evaluation root inventory mismatch: ${root}\n"
            "actual=${root_entries}\nexpected=${sorted_expected}"
        )
    endif()
    set(sorted_evaluation_files "${evaluation_files}")
    list(SORT sorted_evaluation_files)
    set(file_count 0)
    set(total_bytes 0)
    set(metric_records 0)
    foreach(directory IN LISTS expected_directories)
        set(directory_path "${root}/${directory}")
        file(GLOB children
            LIST_DIRECTORIES TRUE
            RELATIVE "${directory_path}"
            "${directory_path}/*"
        )
        list(SORT children)
        if(NOT "${children}" STREQUAL "${sorted_evaluation_files}")
            message(FATAL_ERROR
                "evaluation directory inventory mismatch: ${directory_path}\n"
                "actual=${children}\nexpected=${sorted_evaluation_files}"
            )
        endif()
        foreach(name IN LISTS evaluation_files)
            set(path "${directory_path}/${name}")
            if(NOT EXISTS "${path}" OR IS_DIRECTORY "${path}")
                message(FATAL_ERROR "missing evaluation output file: ${path}")
            endif()
            file(SIZE "${path}" size)
            math(EXPR total_bytes "${total_bytes} + ${size}")
            math(EXPR file_count "${file_count} + 1")
        endforeach()
        parse_evaluation_directory("${directory}" scope profile k)
        if(scope STREQUAL "full")
            set(train_rows 6866)
            set(validation_rows 1717)
            set(wandering_zero FALSE)
        else()
            set(train_rows 800)
            set(validation_rows 200)
            set(wandering_zero TRUE)
        endif()
        verify_label_map("${directory_path}")
        verify_metrics_csv(
            "${directory_path}" "${k}" "${train_rows}" "${validation_rows}"
        )
        math(EXPR metric_records "${metric_records} + 2")
        verify_contingency_csv(
            "${directory_path}/train_contingency_counts.csv"
            "${k}" "${train_rows}" FALSE
        )
        verify_contingency_csv(
            "${directory_path}/validation_contingency_counts.csv"
            "${k}" "${validation_rows}" "${wandering_zero}"
        )
        verify_metadata(
            "${directory_path}" "${scope}" "${profile}" "${k}"
            "${train_rows}" "${validation_rows}" "${repository_head}"
        )
        verify_checksum_manifest("${directory_path}")
        verify_evaluator_source_snapshot("${directory_path}")
    endforeach()
    if(NOT file_count EQUAL 98 OR NOT metric_records EQUAL 28)
        message(FATAL_ERROR
            "evaluation aggregate count mismatch in ${root}: "
            "files=${file_count}, metric_records=${metric_records}"
        )
    endif()
    assert_no_transients("${root}" "evaluation fixture output")
    set(${output_bytes} "${total_bytes}" PARENT_SCOPE)
endfunction()

function(snapshot_evaluation_root root output_snapshot output_count output_bytes)
    file(GLOB_RECURSE entries
        LIST_DIRECTORIES TRUE
        RELATIVE "${root}"
        "${root}/*"
    )
    list(SORT entries)
    set(snapshot "")
    set(file_count 0)
    set(total_bytes 0)
    foreach(relative_path IN LISTS entries)
        set(path "${root}/${relative_path}")
        if(IS_DIRECTORY "${path}")
            string(APPEND snapshot "D|${relative_path}\n")
        else()
            file(SIZE "${path}" size)
            file(SHA256 "${path}" digest)
            math(EXPR file_count "${file_count} + 1")
            math(EXPR total_bytes "${total_bytes} + ${size}")
            string(APPEND snapshot
                "F|${relative_path}|${size}|${digest}\n"
            )
        endif()
    endforeach()
    set(${output_snapshot} "${snapshot}" PARENT_SCOPE)
    set(${output_count} "${file_count}" PARENT_SCOPE)
    set(${output_bytes} "${total_bytes}" PARENT_SCOPE)
endfunction()

function(compare_evaluation_roots left right)
    foreach(directory IN LISTS expected_directories)
        foreach(name IN LISTS evaluation_files)
            set(left_path "${left}/${directory}/${name}")
            set(right_path "${right}/${directory}/${name}")
            file(SIZE "${left_path}" left_size)
            file(SIZE "${right_path}" right_size)
            if(NOT left_size EQUAL right_size)
                message(FATAL_ERROR
                    "fresh-process size mismatch: ${directory}/${name}"
                )
            endif()
            file(SHA256 "${left_path}" left_hash)
            file(SHA256 "${right_path}" right_hash)
            if(NOT left_hash STREQUAL right_hash)
                message(FATAL_ERROR
                    "fresh-process SHA-256 mismatch: ${directory}/${name}"
                )
            endif()
            execute_process(
                COMMAND "${CMAKE_COMMAND}" -E compare_files
                    "${left_path}" "${right_path}"
                RESULT_VARIABLE compare_result
            )
            if(NOT compare_result EQUAL 0)
                message(FATAL_ERROR
                    "fresh-process byte mismatch: ${directory}/${name}"
                )
            endif()
        endforeach()
    endforeach()
endfunction()

function(run_validate scope profile k)
    set(context "${scope}/${profile}/k${k}")
    execute_process(
        COMMAND
            "${TEST_EXECUTABLE}"
            --validate-only
            --scope "${scope}"
            --profile "${profile}"
            --k "${k}"
            "${repository_absolute}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
    )
    if(NOT result EQUAL 0)
        message(FATAL_ERROR
            "validate-only failed for ${context}: exit=${result}\n"
            "stdout=${output}\nstderr=${error}"
        )
    endif()
    require_line("${output}" "VALIDATE_ONLY_PASS=true" "${context}")
    require_line("${output}" "METRICS_COMPUTE_CALLS=0" "${context}")
    require_line("${output}" "OUTPUT_CREATED=false" "${context}")
    require_line("${output}" "LABEL_COUNT=22" "${context}")
    require_line("${output}" "CONTINGENCY_ROWS=22" "${context}")
    require_line("${output}" "CONTINGENCY_COLS=${k}" "${context}")
    require_line(
        "${output}" "LABEL_JOIN=manifest_index_cross_checked" "${context}"
    )
    require_line(
        "${output}"
        "METRICS_PLANNED=purity,nmi_arithmetic,ari,homogeneity,completeness"
        "${context}"
    )
    if(scope STREQUAL "matched_subset1000")
        require_line("${output}" "TRAIN_ROWS_VALIDATED=800" "${context}")
        require_line(
            "${output}" "VALIDATION_ROWS_VALIDATED=200" "${context}"
        )
        require_line(
            "${output}" "WANDERING_LINE_MATCHED_VALIDATION_COUNT=0"
            "${context}"
        )
        require_line(
            "${output}" "WANDERING_LINE_ZERO_ROW_ACCEPTED=true" "${context}"
        )
    else()
        require_line("${output}" "TRAIN_ROWS_VALIDATED=6866" "${context}")
        require_line(
            "${output}" "VALIDATION_ROWS_VALIDATED=1717" "${context}"
        )
    endif()
endfunction()

function(run_evaluate scope profile k output_root repository_head)
    set(context "evaluate ${scope}/${profile}/k${k} -> ${output_root}")
    execute_process(
        COMMAND
            "${TEST_EXECUTABLE}"
            --evaluate
            --scope "${scope}"
            --profile "${profile}"
            --k "${k}"
            --output-root "${output_root}"
            --repository-head "${repository_head}"
            "${repository_absolute}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
    )
    if(NOT result EQUAL 0)
        message(FATAL_ERROR
            "${context} failed: exit=${result}\nstdout=${output}\nstderr=${error}"
        )
    endif()
    require_line("${output}" "METRICS_COMPUTE_CALLS=2" "${context}")
    require_line("${output}" "OUTPUT_CREATED=true" "${context}")
    require_line("${output}" "OVERWRITE_REJECTED=false" "${context}")
    require_line("${output}" "PARTIAL_REJECTED=false" "${context}")
    require_line("${output}" "LABEL_COUNT=22" "${context}")
    require_line("${output}" "CONTINGENCY_ROWS=22" "${context}")
    require_line("${output}" "CONTINGENCY_COLS=${k}" "${context}")
    require_line("${output}" "EVALUATION_PASS=true" "${context}")
    if(scope STREQUAL "full")
        require_line("${output}" "TRAIN_ROWS=6866" "${context}")
        require_line("${output}" "VALIDATION_ROWS=1717" "${context}")
    else()
        require_line("${output}" "TRAIN_ROWS=800" "${context}")
        require_line("${output}" "VALIDATION_ROWS=200" "${context}")
        require_line(
            "${output}" "WANDERING_LINE_MATCHED_VALIDATION_COUNT=0"
            "${context}"
        )
        require_line("${output}" "WANDERING_LINE_ZERO_ROW=true" "${context}")
    endif()
    extract_unique_value("${output}" OUTPUT_PATH "${context}" actual_output)
    get_filename_component(
        expected_output
        "${output_root}/${scope}_${profile}_k${k}_seed123_maxiter300"
        ABSOLUTE
    )
    file(TO_CMAKE_PATH "${actual_output}" actual_normalized)
    file(TO_CMAKE_PATH "${expected_output}" expected_normalized)
    if(WIN32)
        string(TOLOWER "${actual_normalized}" actual_normalized)
        string(TOLOWER "${expected_normalized}" expected_normalized)
    endif()
    if(NOT actual_normalized STREQUAL expected_normalized)
        message(FATAL_ERROR
            "OUTPUT_PATH mismatch for ${context}: ${actual_output}"
        )
    endif()
endfunction()

function(run_expected_rejection
    scope profile k output_root repository_head reason
)
    set(context "${reason} rejection ${scope}/${profile}/k${k}")
    execute_process(
        COMMAND
            "${TEST_EXECUTABLE}"
            --evaluate
            --scope "${scope}"
            --profile "${profile}"
            --k "${k}"
            --output-root "${output_root}"
            --repository-head "${repository_head}"
            "${repository_absolute}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
    )
    if(result EQUAL 0)
        message(FATAL_ERROR
            "${context} unexpectedly succeeded\nstdout=${output}\nstderr=${error}"
        )
    endif()
    require_line("${output}" "METRICS_COMPUTE_CALLS=0" "${context}")
    require_line("${output}" "OUTPUT_CREATED=false" "${context}")
    require_line("${output}" "EVALUATION_PASS=false" "${context}")
    if(reason STREQUAL "overwrite")
        require_line("${output}" "OVERWRITE_REJECTED=true" "${context}")
        require_line("${output}" "PARTIAL_REJECTED=false" "${context}")
    elseif(reason STREQUAL "partial")
        require_line("${output}" "OVERWRITE_REJECTED=false" "${context}")
        require_line("${output}" "PARTIAL_REJECTED=true" "${context}")
    else()
        message(FATAL_ERROR "unknown rejection reason: ${reason}")
    endif()
endfunction()

execute_process(
    COMMAND git --no-optional-locks -C "${repository_absolute}" rev-parse HEAD
    RESULT_VARIABLE git_result
    OUTPUT_VARIABLE repository_head
    ERROR_VARIABLE git_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(LENGTH "${repository_head}" repository_head_length)
if(NOT git_result EQUAL 0 OR NOT repository_head_length EQUAL 40 OR
   NOT "${repository_head}" MATCHES "^[0-9a-f]+$")
    message(FATAL_ERROR
        "cannot resolve a lowercase 40-hex repository HEAD: ${git_error}"
    )
endif()

verify_exact_results_inventory()
snapshot_results(before_snapshot before_count before_bytes)
string(SHA256 before_fingerprint "${before_snapshot}")

foreach(profile IN ITEMS mini75 mini112 full224)
    foreach(k IN ITEMS 22 36)
        run_validate(full "${profile}" "${k}")
    endforeach()
endforeach()
foreach(profile IN ITEMS mini75 mini112 mini224 full224)
    foreach(k IN ITEMS 22 36)
        run_validate(matched_subset1000 "${profile}" "${k}")
    endforeach()
endforeach()

# The target was validated above and is the only recursive cleanup target here.
file(REMOVE_RECURSE "${fixture_absolute}")
file(MAKE_DIRECTORY "${fixture_absolute}")
set(replica_a "${fixture_absolute}/replica_a")
set(replica_b "${fixture_absolute}/replica_b")

foreach(profile IN ITEMS mini75 mini112 full224)
    foreach(k IN ITEMS 22 36)
        run_evaluate(
            full "${profile}" "${k}" "${replica_a}" "${repository_head}"
        )
        run_evaluate(
            full "${profile}" "${k}" "${replica_b}" "${repository_head}"
        )
    endforeach()
endforeach()
foreach(profile IN ITEMS mini75 mini112 mini224 full224)
    foreach(k IN ITEMS 22 36)
        run_evaluate(
            matched_subset1000 "${profile}" "${k}"
            "${replica_a}" "${repository_head}"
        )
        run_evaluate(
            matched_subset1000 "${profile}" "${k}"
            "${replica_b}" "${repository_head}"
        )
    endforeach()
endforeach()

verify_evaluation_root("${replica_a}" "${repository_head}" replica_a_bytes)
verify_evaluation_root("${replica_b}" "${repository_head}" replica_b_bytes)
snapshot_evaluation_root(
    "${replica_a}" replica_a_snapshot replica_a_count replica_a_snapshot_bytes
)
snapshot_evaluation_root(
    "${replica_b}" replica_b_snapshot replica_b_count replica_b_snapshot_bytes
)
string(SHA256 replica_a_fingerprint "${replica_a_snapshot}")
string(SHA256 replica_b_fingerprint "${replica_b_snapshot}")
if(NOT replica_a_count EQUAL 98 OR NOT replica_b_count EQUAL 98 OR
   NOT replica_a_bytes EQUAL replica_b_bytes OR
   NOT replica_a_snapshot_bytes EQUAL replica_b_snapshot_bytes OR
   NOT replica_a_fingerprint STREQUAL replica_b_fingerprint OR
   NOT "${replica_a_snapshot}" STREQUAL "${replica_b_snapshot}")
    message(FATAL_ERROR "fresh-process aggregate output mismatch")
endif()
compare_evaluation_roots("${replica_a}" "${replica_b}")

# Refuse overwrite before metric computation and preserve all 98 published files.
snapshot_evaluation_root(
    "${replica_a}" overwrite_before overwrite_count overwrite_bytes
)
run_expected_rejection(
    full mini75 22 "${replica_a}" "${repository_head}" overwrite
)
snapshot_evaluation_root(
    "${replica_a}" overwrite_after overwrite_after_count overwrite_after_bytes
)
if(NOT "${overwrite_before}" STREQUAL "${overwrite_after}" OR
   NOT overwrite_count EQUAL overwrite_after_count OR
   NOT overwrite_bytes EQUAL overwrite_after_bytes)
    message(FATAL_ERROR "overwrite rejection changed published fixtures")
endif()

# Refuse a pre-existing partial before metric computation without touching it.
set(partial_root "${fixture_absolute}/partial_probe")
set(partial_directory
    "${partial_root}/full_mini75_k22_seed123_maxiter300.partial"
)
file(MAKE_DIRECTORY "${partial_directory}")
set(partial_sentinel "${partial_directory}/sentinel.txt")
file(WRITE "${partial_sentinel}" "preexisting partial sentinel\n")
file(MAKE_DIRECTORY "${partial_directory}/nested")
file(WRITE
    "${partial_directory}/nested/second_sentinel.bin"
    "second preexisting partial sentinel\n"
)
snapshot_evaluation_root(
    "${partial_directory}"
    partial_before partial_before_count partial_before_bytes
)
run_expected_rejection(
    full mini75 22 "${partial_root}" "${repository_head}" partial
)
snapshot_evaluation_root(
    "${partial_directory}"
    partial_after partial_after_count partial_after_bytes
)
if(NOT "${partial_before}" STREQUAL "${partial_after}" OR
   NOT partial_before_count EQUAL partial_after_count OR
   NOT partial_before_bytes EQUAL partial_after_bytes OR
   EXISTS "${partial_root}/full_mini75_k22_seed123_maxiter300")
    message(FATAL_ERROR "partial rejection changed input or created final output")
endif()
file(REMOVE_RECURSE "${partial_root}")

assert_no_transients("${replica_a}" "replica A final fixture")
assert_no_transients("${replica_b}" "replica B final fixture")
assert_no_transients("${fixture_absolute}" "complete evaluator fixture")
snapshot_results(after_snapshot after_count after_bytes)
string(SHA256 after_fingerprint "${after_snapshot}")
if(NOT "${before_snapshot}" STREQUAL "${after_snapshot}" OR
   NOT before_count EQUAL after_count OR NOT before_bytes EQUAL after_bytes OR
   NOT before_fingerprint STREQUAL after_fingerprint)
    message(FATAL_ERROR "official k-means inputs changed during evaluation test")
endif()
verify_exact_results_inventory()

message("VALIDATE_ONLY_CONFIGURATIONS=14")
message("FRESH_PROCESS_EVALUATION_CONFIGURATIONS=14")
message("EVALUATION_FILES_PER_REPLICA=98")
message("EVALUATION_METRIC_RECORDS_PER_REPLICA=28")
message("EVALUATION_METRICS_FINITE_AND_IN_RANGE=true")
message("EVALUATION_CONTINGENCY_SHAPES_AND_TOTALS_VALID=true")
message("MATCHED_VALIDATION_WANDERING_LINE_ZERO_ROW=true")
message("FRESH_PROCESS_BYTE_SIZE_SHA256_IDENTICAL=true")
message("EVALUATION_REPLICA_FINGERPRINT=${replica_a_fingerprint}")
message("OVERWRITE_REJECTED_BEFORE_METRICS=true")
message("PARTIAL_REJECTED_BEFORE_METRICS=true")
message("NO_RESIDUAL_PARTIAL_TMP=true")
message("OFFICIAL_KMEANS_INPUT_FILES=${after_count}")
message("OFFICIAL_KMEANS_INPUT_BYTES=${after_bytes}")
message("OFFICIAL_KMEANS_INPUT_FINGERPRINT=${after_fingerprint}")
message("OFFICIAL_KMEANS_ARTIFACTS_UNCHANGED=true")
message("VAE_C_MU_KMEANS_EVALUATOR_VALIDATE_AND_PRODUCTIVE_TEST_PASS=true")

# Leave the external build tree clean and make a second CTest run identical.
file(REMOVE_RECURSE "${fixture_absolute}")
