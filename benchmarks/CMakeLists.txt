add_executable(benchmark_cg_mpi benchmark_cg_mpi.cpp)
target_link_libraries(benchmark_cg_mpi abcg)

if (USE_LOCALITY)
    add_executable(benchmark_cg_mpil benchmark_cg_mpil.cpp)
    target_link_libraries(benchmark_cg_mpil abcg)
endif()
