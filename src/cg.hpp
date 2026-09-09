#ifndef ABCG_CG_HEADER
#define ABCG_CG_HEADER

#include "par_binary_IO.hpp"
#include "sparse_mat.hpp"
#include "comm/comm_routines.hpp"

void spmv(double alpha, Mat& A, double* x, double beta, double* b);
void spmv(double alpha,
          ParMat& A,
          std::vector<double>& x,
          double beta,
          std::vector<double>& b,
          SpMVComm* comm);
void axpy(double alpha, std::vector<double>& x, std::vector<double>& y);
void scale(double alpha, std::vector<double>& x);
double inner_product(std::vector<double>& a,
                     std::vector<double>& b,
                     AllreduceComm* comm);


template <typename SComm, typename AComm>
int CG(ParMat& A, 
        std::vector<double>& x, 
        std::vector<double>& b, 
        int max_iters = 500,
        double tol = 1e-06)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // CG Variables
    std::vector<double> r(A.local_rows);
    std::vector<double> p(A.local_rows);
    std::vector<double> Ap(A.local_rows);
    std::vector<double> recvbuf(A.recv_comm.size_msgs);
    std::vector<double> sendbuf(A.send_comm.size_msgs);

    // Setup persistent allreduces
    double local_sum, global_sum;

    SpMVComm* spmv_comm = new SComm();
    AllreduceComm* allreduce_comm = new AComm();

    spmv_comm->init(A, sendbuf.data(), recvbuf.data());
    allreduce_comm->init(&local_sum, &global_sum);

    int iter, recompute_r;
    double alpha, beta;
    double rr_inner, next_inner, App_inner;
    double norm_r;
    // int max_iter = ((int)(1.3*b.size())) + 2;
    int max_iter = 500;

    // r0 = b - A * x0
    r = b;
    spmv(-1.0, A, x, 1.0, r, spmv_comm);

    // p0 = r0
    p = r;

    // Find initial (r, r) and residual
    rr_inner = inner_product(r, r, allreduce_comm);

    norm_r = sqrt(rr_inner);

    // Scale tolerance by norm_r
    if (norm_r != 0.0)
    {
        tol = tol * norm_r;
    }

    // How often should r be recomputed
    recompute_r = 8;
    iter        = 0;

    // Main CG Loop
    while (norm_r > tol && iter < max_iter)
    {
        // alpha_i = (r_i, r_i) / (A*p_i, p_i)
        spmv(1.0, A, p, 0.0, Ap, spmv_comm);
        App_inner = inner_product(Ap, p, allreduce_comm);
        if (App_inner < 0.0)
        {
            printf("Indefinite matrix detected in CG! Aborting...\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        alpha = rr_inner / App_inner;

        axpy(alpha, x, p);

        // x_{i+1} = x_i + alpha_i * p_i
        if ((iter % recompute_r) && iter > 0)
        {
            axpy(-1.0 * alpha, r, Ap);
        }
        else
        {
            r = b;
            spmv(-1.0, A, x, 1.0, r, spmv_comm);
        }

        next_inner = inner_product(r, r, allreduce_comm);
        beta       = next_inner / rr_inner;

        scale(beta, p);
        axpy(1.0, p, r);

        // Update next inner product
        rr_inner = next_inner;
        norm_r   = sqrt(rr_inner);

        iter++;
    }

    delete spmv_comm;
    delete allreduce_comm;

    return iter;
}


#endif
