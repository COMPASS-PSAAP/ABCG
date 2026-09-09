#include "cg.hpp"


// Serial SpMV
void spmv(double alpha, Mat& A, double* x, double beta, double* b)
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

// Parallel SpMV
void spmv(double alpha,
          ParMat& A,
          std::vector<double>& x,
          double beta,
          std::vector<double>& b,
          SpMVComm* comm)
{
    int proc, start, end;
    int tag = 0;

    comm->pack(x);

    comm->start();

    spmv(alpha, A.on_proc, x.data(), beta, b.data());

    comm->wait();

    spmv(alpha, A.off_proc, comm->get_recvbuf(), 1.0, b.data());
}

// Serial AXPY X = X + alpha * Y
void axpy(double alpha, std::vector<double>& x, std::vector<double>& y)
{
    for (int i = 0; i < x.size(); i++)
    {
        x[i] = x[i] + alpha * y[i];
    }
}

// Serial Vector Scale X = alpha * X
void scale(double alpha, std::vector<double>& x)
{
    for (int i = 0; i < x.size(); i++)
    {
        x[i] = alpha * x[i];
    }
}

// Parallel Inner Product
double inner_product(std::vector<double>& a,
                     std::vector<double>& b,
                     AllreduceComm* comm)
{
    double local_sum = 0;
    for (int i = 0; i < a.size(); i++)
    {
        local_sum += a[i] * b[i];
    }
    comm->set_local_sum(local_sum);

    comm->start();

    comm->wait();

    return comm->get_global_sum();
}




