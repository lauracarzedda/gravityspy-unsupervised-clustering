if(NOT DEFINED TEST_EXECUTABLE OR NOT DEFINED TEST_ROOT)
    message(FATAL_ERROR "TEST_EXECUTABLE and TEST_ROOT are required")
endif()
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef suffix)
set(fixture "${TEST_ROOT}/iic_c5b6_${suffix}")
file(MAKE_DIRECTORY "${fixture}")

function(run_ok label)
    execute_process(COMMAND "${TEST_EXECUTABLE}" ${ARGN}
        RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
    if(NOT rc EQUAL 0 OR NOT err STREQUAL "")
        message(FATAL_ERROR "${label}: exit=${rc}\n${out}\n${err}")
    endif()
    set("${label}_OUTPUT" "${out}" PARENT_SCOPE)
endfunction()

function(run_refused label)
    execute_process(COMMAND "${TEST_EXECUTABLE}" ${ARGN}
        RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
    if(rc EQUAL 0)
        message(FATAL_ERROR "${label}: expected refusal\n${out}\n${err}")
    endif()
endfunction()

function(same first second label)
    if(NOT EXISTS "${first}" OR NOT EXISTS "${second}")
        message(FATAL_ERROR "${label}: file missing")
    endif()
    file(SHA256 "${first}" a)
    file(SHA256 "${second}" b)
    if(NOT a STREQUAL b)
        message(FATAL_ERROR "${label}: ${a} != ${b}")
    endif()
endfunction()

set(continuous "${fixture}/continuous")
set(interrupted "${fixture}/interrupted")
set(before_first "${fixture}/before_first")
set(terminal "${fixture}/terminal")
set(mismatch "${fixture}/mismatch")
set(corrupt "${fixture}/corrupt")

run_ok(continuous_run --output-dir "${continuous}" --max-epochs 3)
if(NOT continuous_run_OUTPUT MATCHES "IIC_SYNTHETIC_COMPLETED=true")
    message(FATAL_ERROR "Continuous completion marker absent")
endif()
file(SHA256 "${TEST_EXECUTABLE}" executable_sha256)
string(TOLOWER "${executable_sha256}" executable_sha256)
if(NOT continuous_run_OUTPUT MATCHES
       "IIC_SYNTHETIC_PRODUCER_BINARY_SHA256=${executable_sha256}")
    message(FATAL_ERROR "Producer binary SHA-256 does not match executable")
endif()

run_ok(interrupted_first --output-dir "${interrupted}"
       --max-epochs 3 --stop-after 0)
if(EXISTS "${interrupted}/final.ckpt")
    message(FATAL_ERROR "Controlled interruption published FINAL")
endif()
file(APPEND "${interrupted}/metrics.csv"
    "1,0,999,999,1998,6\n2,0,999,999,1998,9\n")
run_ok(interrupted_resume --output-dir "${interrupted}"
       --max-epochs 3 --resume)
if(NOT interrupted_resume_OUTPUT MATCHES "IIC_SYNTHETIC_RESUME_FROM_EPOCH=1")
    message(FATAL_ERROR "Resume did not start at E+1")
endif()
foreach(name latest.ckpt final.ckpt metrics.csv)
    same("${continuous}/${name}"
         "${interrupted}/${name}" "interrupted/${name}")
endforeach()

file(MAKE_DIRECTORY "${before_first}")
file(WRITE "${before_first}/metrics.csv"
    "epoch,pair,main_loss,over_loss,total_loss,optimizer_step\n"
    "0,0,999,999,1998,3\n")
run_ok(before_first_resume --output-dir "${before_first}"
       --max-epochs 3 --resume)
if(NOT before_first_resume_OUTPUT MATCHES
       "IIC_SYNTHETIC_RESTART_FROM_EPOCH_0=true")
    message(FATAL_ERROR "No-LATEST resume did not restart epoch zero")
endif()
foreach(name latest.ckpt final.ckpt metrics.csv)
    same("${continuous}/${name}"
         "${before_first}/${name}" "before_first/${name}")
endforeach()

run_ok(terminal_first --output-dir "${terminal}"
       --max-epochs 3 --stop-after 2)
if(EXISTS "${terminal}/final.ckpt")
    message(FATAL_ERROR "Terminal controlled stop published FINAL")
endif()
file(SHA256 "${terminal}/latest.ckpt" terminal_before)
run_ok(terminal_resume --output-dir "${terminal}"
       --max-epochs 3 --resume)
if(NOT terminal_resume_OUTPUT MATCHES
       "IIC_SYNTHETIC_FINALIZED_FROM_LATEST=true" OR
   terminal_resume_OUTPUT MATCHES "IIC_SYNTHETIC_EPOCH=")
    message(FATAL_ERROR "Terminal resume retrained or failed finalization")
endif()
file(SHA256 "${terminal}/latest.ckpt" terminal_after)
if(NOT terminal_before STREQUAL terminal_after)
    message(FATAL_ERROR "Terminal resume modified LATEST")
endif()
foreach(name latest.ckpt final.ckpt metrics.csv)
    same("${continuous}/${name}"
         "${terminal}/${name}" "terminal/${name}")
endforeach()

file(SHA256 "${terminal}/final.ckpt" final_before)
run_refused(final_exists --output-dir "${terminal}"
            --max-epochs 3 --resume)
file(SHA256 "${terminal}/final.ckpt" final_after)
if(NOT final_before STREQUAL final_after)
    message(FATAL_ERROR "Refused execution modified FINAL")
endif()

run_ok(mismatch_first --output-dir "${mismatch}"
       --max-epochs 3 --stop-after 0)
file(SHA256 "${mismatch}/latest.ckpt" mismatch_before)
run_refused(wrong_budget --output-dir "${mismatch}"
            --max-epochs 4 --resume)
file(SHA256 "${mismatch}/latest.ckpt" mismatch_after)
if(NOT mismatch_before STREQUAL mismatch_after OR
   EXISTS "${mismatch}/final.ckpt")
    message(FATAL_ERROR "Identity mismatch mutated output")
endif()

run_ok(corrupt_first --output-dir "${corrupt}"
       --max-epochs 3 --stop-after 0)
file(APPEND "${corrupt}/latest.ckpt" "corruption")
run_refused(corrupt_latest --output-dir "${corrupt}"
            --max-epochs 3 --resume)
if(EXISTS "${corrupt}/final.ckpt")
    message(FATAL_ERROR "Corrupt checkpoint produced FINAL")
endif()

file(STRINGS "${continuous}/metrics.csv" lines)
list(LENGTH lines count)
if(NOT count EQUAL 16)
    message(FATAL_ERROR "Expected header plus 15 metric rows")
endif()
file(GLOB_RECURSE leftovers "${fixture}/*.partial" "${fixture}/*.tmp")
if(leftovers)
    message(FATAL_ERROR "Synthetic fixture left partial/temp files")
endif()
message("IIC_SYNTHETIC_CONTINUOUS_VS_RESUME_BITWISE=true")
message("IIC_SYNTHETIC_METRICS_FUTURE_RECONCILED=true")
message("IIC_SYNTHETIC_BEFORE_FIRST_RESTART=true")
message("IIC_SYNTHETIC_TERMINAL_FINALIZE_ONLY=true")
message("IIC_SYNTHETIC_FINAL_REFUSAL=true")
message("IIC_SYNTHETIC_MISMATCH_CORRUPTION_REFUSAL=true")
