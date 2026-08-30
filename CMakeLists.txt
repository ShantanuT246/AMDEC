cmake_minimum_required(VERSION 3.20)
project(amdec VERSION 0.1.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Compiler flags
add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Wno-unused-parameter
)

# Release optimisations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3)
endif()

# Optional: AVX2 (Module 4 only — gated so it doesn't break scalar builds)
option(AMDEC_ENABLE_AVX2 "Enable AVX2 SIMD for Module 4 acceleration benchmark" OFF)

# ---------------------------------------------------------------------------
# Sanitizers (Debug builds)
# ---------------------------------------------------------------------------
option(AMDEC_SANITIZE "Enable AddressSanitizer + UBSan (debug only)" OFF)
if(AMDEC_SANITIZE AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()

# ---------------------------------------------------------------------------
# Third-party dependencies via FetchContent
# ---------------------------------------------------------------------------
include(FetchContent)

# --- xxHash (header-only) ---------------------------------------------------
FetchContent_Declare(
    xxhash
    GIT_REPOSITORY https://github.com/Cyan4973/xxHash.git
    GIT_TAG        v0.8.3
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(xxhash)

# xxHash doesn't ship a CMakeLists; create an INTERFACE target manually.
if(NOT TARGET xxHash::xxhash)
    add_library(xxHash_iface INTERFACE)
    target_include_directories(xxHash_iface INTERFACE ${xxhash_SOURCE_DIR})
    add_library(xxHash::xxhash ALIAS xxHash_iface)
endif()

# --- Catch2 (v3) ------------------------------------------------------------
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.8.1
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(Catch2)

# --- Google Benchmark (Module 4 only) ---------------------------------------
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.9.4
    GIT_SHALLOW    TRUE
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

# ---------------------------------------------------------------------------
# Source subdirectory
# ---------------------------------------------------------------------------
add_subdirectory(src)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)

# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
add_subdirectory(experiments)
