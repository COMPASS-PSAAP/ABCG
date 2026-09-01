#ifndef ABCG_CG_HEADER
#define ABCG_CG_HEADER

#include <math.h>

#include "mpi.h"
#include "par_binary_IO.hpp"
#include "sparse_mat.hpp"

namespace ABCG
{
    int max_tests = 0;
    int max_iters = 0;
    double tol    = 0;
}  // namespace ABCG

// Serial SpMV b = alpha*A*x + beta*b
void spmv(double alpha, Mat& A, std::vector<double>& x, double beta, std::vector<double>& b)
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

double calc_norm_b(std::vector<double>& b)
{
    double norm_b = 0;
    for (int i = 0; i < b.size(); i++)
    {
        norm_b += b[i] * b[i];
    }
    MPI_Allreduce(MPI_IN_PLACE, &norm_b, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return sqrt(norm_b);
}

int calc_num_iters(const std::vector<double>& r,
                   int rank,
                   int conv_iter,
                   double norm_b,
                   double tfinal,
                   std::string test_name)
{
    double sum = 0;
    for (int i = 0; i < r.size(); i++)
    {
        sum += r[i] * r[i];
    }
    MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0)
    {
        std::printf("%s: %d iter, norm %e", test_name.c_str(), conv_iter, sqrt(sum) / norm_b);
    }

    int n_iters  = 1;
    double t_max = 0;
    MPI_Allreduce(&tfinal, &t_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (t_max < 1.0)
    {
        n_iters = 1.0 / t_max;
    }

    if (rank == 0)
    {
        std::printf(" (%d iter(s)/test)\n", n_iters);
        std::cout << std::flush;
    }
    return n_iters;
}

void initialize_defaults(int max_tests = 5)
{
    ABCG::max_iters = 500;
    ABCG::max_tests = max_tests;
    ABCG::tol       = 1e-6;
}

ParMat initialize_cg(
    int* argc, char*** argv, const int rank, std::vector<double>& x, std::vector<double>& b)
{
    // Check for name of matrix to use
    const char* filename = "Dubcova2.pm";
    if ((*argc) > 1)
    {
        filename = (*argv)[1];
    }
    int num_tests = 5;
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

    initialize_defaults(num_tests);

    return A;
}

#endif