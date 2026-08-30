# ---------------------------------------------------------------------------
# tests/CMakeLists.txt
# Each test file becomes its own executable linked against Catch2 v3.
# ---------------------------------------------------------------------------

include(CTest)

# Helper macro to register a Catch2 test executable
macro(amdec_test name source)
    add_executable(${name} ${source})
    target_link_libraries(${name}
        PRIVATE
            Catch2::Catch2WithMain  # Catch2 v3: provides main()
            amdec_workload
            amdec_m1
    )
    target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR})
    add_test(NAME ${name} COMMAND ${name})
endmacro()

# --- Week 2: Module 1 tests -------------------------------------------------
amdec_test(test_m1_sampling tests/test_m1_sampling.cpp)

# --- Future weeks (uncomment as modules are implemented) -------------------
# amdec_test(test_m2_lsh      tests/test_m2_lsh.cpp)
# amdec_test(test_m3_delta    tests/test_m3_delta.cpp)
# amdec_test(test_m5_cow      tests/test_m5_cow.cpp)
