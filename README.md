# ABCG
<i>A</i>manda <i>B</i>ienz's _CG_ solver. 

Currently offers two variants:
1. Baseline MPI version
2. MPI Advance version

By default, the code will only build the MPI version. To enable the MPI Advance version, make sure that CMake can find your installation of MPI Advance. Both modes are fairly simple CMake builds:
```bash
mkdir build && cd build
# MPI Only build
cmake ..
# MPI + MPI Advance build
cmake -DCMAKE_PREFIX_PATH=<path to MPI Advance> ..
make
```

To run, point the executable to the path of the `.pm` matrix file to use. An optional second parameter that controls the number of tests to do can be added after the matrix path like so:
```bash
# Assuming you are in the build directory:
<mpi running command> ./cg_mpi ../src/Dubcova2.pm
# To control the number of tests (default is 5)
<mpi running command> ./cg_mpi ../src/Dubcova2.pm 2
# To run the MPI advance version
<mpi running command> ./cg_mpil ../src/Dubcova2.pm
```

The clang-formatting for this project uses a column width of 100 characters per file to help some of the longer C++ tuples remain in a readable fashion.