module load rocmcc/6.4.0-magic
export ROCM_PATH=/usr/tce/packages/rocmcc/rocmcc-6.4.0-magic/
export CC=mpicc
export CXX=mpicxx

MPICH_DIR=/opt/cray/pe/mpich/9.0.1/ofi/crayclang/20.0

hipcc -o cg cg.cpp \
  -I ../../locality_fork/include/ \
  -I${MPICH_DIR}/include \
  -L${MPICH_DIR}/lib -lmpi \
  -lnuma \
  -lmpi_gtl_hsa \
  -DGPU -DGPU_AWARE \
  -lrocsparse \
  -lrocblas \
  -x none ../../locality_fork/build_gpu/liblocality_aware.a
