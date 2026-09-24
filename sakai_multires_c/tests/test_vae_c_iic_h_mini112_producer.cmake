include("${CMAKE_CURRENT_LIST_DIR}/test_vae_c_iic_h_producer.cmake")

# The shared producer test already validates 64/32, source hashes, atomic
# publication, inventory and overwrite rejection. Add logical batch 128.
set(third "${root}/chunk128")
invoke("${third}" 128 produce)
foreach(name IN ITEMS h_f32.bin rows.csv h_spec.txt sha256.txt)
    file(SHA256 "${first}/${name}" first_sha)
    file(SHA256 "${third}/${name}" third_sha)
    if(NOT first_sha STREQUAL third_sha)
        message(FATAL_ERROR "mini112 H chunk128 differs: ${name}")
    endif()
endforeach()
message(STATUS "mini112 H full producer chunk128/64/32 bitwise PASS")
