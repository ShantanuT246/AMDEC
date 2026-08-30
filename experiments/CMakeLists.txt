# ---------------------------------------------------------------------------
# experiments/CMakeLists.txt
# Evaluation binaries — each writes results to results/raw/*.csv
# ---------------------------------------------------------------------------

# --- Week 2 stub: print workload + M1 stats --------------------------------
add_executable(run_prelim_m1
    experiments/prelim_m1.cpp
)
target_link_libraries(run_prelim_m1
    PRIVATE
        amdec_workload
        amdec_m1
)
target_include_directories(run_prelim_m1 PRIVATE ${CMAKE_SOURCE_DIR})

# --- Future experiments (uncomment when modules are ready) -----------------
# add_executable(run_full_eval experiments/run_full_eval.cpp)
# add_executable(ablation_m1   experiments/ablation_m1.cpp)
# add_executable(ablation_adaptive experiments/ablation_adaptive.cpp)
# add_executable(parameter_sweep   experiments/parameter_sweep.cpp)
