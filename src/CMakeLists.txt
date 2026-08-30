# ---------------------------------------------------------------------------
# src/CMakeLists.txt
# Builds each module as a static library so it can be linked by
# experiments, tests, and the top-level simulator.
# ---------------------------------------------------------------------------

# --- Common headers (interface-only, no source) ----------------------------
add_library(amdec_common INTERFACE)
target_include_directories(amdec_common INTERFACE ${CMAKE_SOURCE_DIR}/src)

# --- Workload generator (Module 0 — synthetic data) ------------------------
add_library(amdec_workload STATIC
    workload/page_generator.cpp
)
target_include_directories(amdec_workload PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(amdec_workload PUBLIC amdec_common)

# --- Module 1: Region-Aware Sampling & Feedback Engine ---------------------
add_library(amdec_m1 STATIC
    m1_sampling/sampler.cpp
)
target_include_directories(amdec_m1 PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(amdec_m1 PUBLIC amdec_common)

# --- Module 2: LSH Clustering (stub — Week 3) ------------------------------
# add_library(amdec_m2 STATIC m2_lsh/lsh_index.cpp)

# --- Module 3: Base-Delta Storage (stub — Week 5) --------------------------
# add_library(amdec_m3 STATIC m3_delta/delta_store.cpp)

# --- Module 4: Acceleration Benchmark (stub — Week 7) ----------------------
# Requires Google Benchmark — activated in Week 7

# --- Module 5: COW Handler (stub — Week 6) ---------------------------------
# add_library(amdec_m5 STATIC m5_cow/cow_handler.cpp)

# --- Simulator (stub — wired up in Week 6) ---------------------------------
# add_executable(amdec_sim simulator.cpp)
