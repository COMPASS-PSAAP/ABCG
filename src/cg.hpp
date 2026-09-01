#ifndef ABCG_CG_HEADER
#define ABCG_CG_HEADER

#include <math.h>

#include "par_binary_IO.hpp"
#include "sparse_mat.hpp"

// Serial SpMV b = alpha*A*x + beta*b
void spmv(
    double alpha, Mat& A, std::vector<double>& x, double beta, std::vector<double>& b)
{
    double sum;
    int start, end;

    for (int i = 0; i < A.n_rows; i++)
    {
        start = A.rowptr[i];
        end   = A.rowptr[i + 1];
        sum   = 0;
        for (int j = start; j < end; j++)
        {
            sum += A.data[j] * x[A.col_idx[j]];
        }
        b[i] = alpha * sum + beta * b[i];
    }
}

void axpy(double alpha, std::vector<double>& x, std::vector<double>& y)
{
    for (int i = 0; i < x.size(); i++)
    {
        x[i] = x[i] + alpha * y[i];
    }
}

void scale(double alpha, std::vector<double>& x)
{
    for (int i = 0; i < x.size(); i++)
    {
        x[i] = alpha * x[i];
    }
}

ParMat initialize_cg(int* argc,
                     char*** argv,
                     const int rank,
                     int& num_tests,
                     std::vector<double>& x,
                     std::vector<double>& b)
{
    // Check for name of matrix to use
    const char* filename = "Dubcova2.pm";
    if ((*argc) > 1)
    {
        filename = (*argv)[1];
    }
    num_tests = 5;
    if ((*argc) > 2)
    {
        num_tests = std::atoi((*argv)[2]);
    }

    double t0, t1;

    ParMat A;
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    readParMatrix(filename, A);
    t1 = MPI_Wtime() - t0;
    MPI_Allreduce(&t1, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0)
    {
        std::printf("Read matrix: %e\n", t0);
        std::cout << std::flush;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    form_comm(A);
    t1 = MPI_Wtime() - t0;
    MPI_Allreduce(&t1, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0)
    {
        std::printf("Form comm: %e\n", t0);
        std::cout << std::flush;
    }

    x.resize(A.local_cols);
    b.resize(A.local_rows);

    return A;
}

#endif