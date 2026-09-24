if(NOT DEFINED TEST_EXECUTABLE OR NOT DEFINED REPOSITORY_ROOT OR
   NOT DEFINED TEST_ROOT)
    message(FATAL_ERROR "missing preflight test arguments")
endif()
execute_process(
    COMMAND git --no-optional-locks rev-parse HEAD
    WORKING_DIRECTORY "${REPOSITORY_ROOT}"
    RESULT_VARIABLE git_result
    OUTPUT_VARIABLE commit
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT git_result EQUAL 0)
    message(FATAL_ERROR "cannot identify producer HEAD")
endif()
foreach(attempt RANGE 1 2)
    execute_process(
        COMMAND "${TEST_EXECUTABLE}" --official-preflight
            --repository-root "${REPOSITORY_ROOT}"
            --code-commit "${commit}"
        RESULT_VARIABLE status
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
    )
    if(NOT status EQUAL 0 OR NOT error STREQUAL "")
        message(FATAL_ERROR "official preflight failed: ${error}")
    endif()
    foreach(required
            "OFFICIAL_PREFLIGHT_PASS=true"
            "VAE_ENCODER_FORWARD_CALLS=0"
            "IIC_FORWARD_CALLS=0"
            "IIC_BACKWARD_CALLS=0"
            "ADAM_STEPS=0"
            "OPTIMIZER_STEPS=0"
            "DECODER_CALLS=0"
            "OUTPUT_CREATED=false"
            "MAX_EPOCHS=100"
            "MAX_OPTIMIZER_STEPS_PER_PAIR=800")
        string(FIND "${output}" "${required}" position)
        if(position EQUAL -1)
            message(FATAL_ERROR "preflight missing ${required}")
        endif()
    endforeach()
    string(REGEX MATCH "RUN_ID=([^\r\n]+)" id_line "${output}")
    set(run_id "${CMAKE_MATCH_1}")
    string(REGEX MATCH "RUN_CONFIG_SHA256=([^\r\n]+)" config_line "${output}")
    set(config_sha "${CMAKE_MATCH_1}")
    if(attempt EQUAL 1)
        set(first_id "${run_id}")
        set(first_config "${config_sha}")
    elseif(NOT run_id STREQUAL first_id OR
           NOT config_sha STREQUAL first_config)
        message(FATAL_ERROR "preflight identity changed between processes")
    endif()
    set(official_path
        "${REPOSITORY_ROOT}/sakai_multires_c/results/vae_c/iic/${run_id}")
    if(EXISTS "${official_path}" OR EXISTS "${official_path}.partial" OR
       EXISTS "${official_path}.tmp")
        message(FATAL_ERROR "preflight created or collided with official path")
    endif()
endforeach()
message(STATUS "mini224 official preflight is deterministic and non-neural")
