cmake_minimum_required(VERSION 3.24)

project(ABCG CXX)

# Text color settings taken from here
# https://stackoverflow.com/questions/18968979/how-to-make-colorized-message-with-cmake
string(ASCII 27 Esc)
set(ColorReset "${Esc}[m")
set(BoldRed    "${Esc}[1;31m")
set(BoldYellow "${Esc}[1;33m")
set(BoldBlue   "${Esc}[1;34m")


# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Default CMake build type if not specified (release includes -O3)
if(NOT CMAKE_BUILD_TYPE)
        message(STATUS "No ${BoldBlue}CMAKE_BUILD_TYPE${ColorReset} specified, setting build type to ${BoldYellow}RELEASE${ColorReset}.")
        set(CMAKE_BUILD_TYPE RELEASE CACHE STRING "Build type for project." FORCE)
endif()

# Find necessary packages
find_package(MPI REQUIRED)
find_package(locality_aware QUIET)

# Define the base-line MPI executable
add_executable(cg_mpi src/cg_mpi.cpp)
# Link with needed libraries (and headers)
target_link_libraries(cg_mpi PRIVATE MPI::MPI_CXX)

if (locality_aware_FOUND)
    message(STATUS "Building ${BoldBlue}MPI Advance ${ColorReset}Variant")
    # Define the MPIL executable
    add_executable(cg_mpil src/cg.cpp)
    target_link_libraries(cg_mpil PRIVATE MPIAdvance::locality_aware MPI::MPI_CXX)
else()
    message(STATUS "${BoldRed}NOT building ${BoldBlue}MPI Advance ${BoldRed}Variant${ColorReset}.")
endif()