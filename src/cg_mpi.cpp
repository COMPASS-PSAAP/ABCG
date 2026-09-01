#include "cg.hpp"

// Parallel SpMV b = alpha*A*x + beta*b
void spmv(double alpha, ParMat& A, std::vector<double>& x, double beta, std::vector<double>& b)
{
    int proc, start, end;
    int tag = 0;
    std::vector<double> recvbuf(A.recv_comm.size_msgs);
    std::vector<double> sendbuf(A.send_comm.size_msgs);

    for (int i = 0; i < A.recv_comm.n_msgs; i++)
    {
        proc  = A.recv_comm.procs[i];
        start = A.recv_comm.ptr[i];
        end   = A.recv_comm.ptr[i + 1];
        MPI_Irecv(&(recvbuf[start]),
                  (int)(end - start),
                  MPI_DOUBLE,
                  proc,
                  tag,
                  MPI_COMM_WORLD,
                  &(A.recv_comm.req[i]));
    }

    for (int i = 0; i < A.send_comm.n_msgs; i++)
    {
        proc  = A.send_comm.procs[i];
        start = A.send_comm.ptr[i];
        end   = A.send_comm.ptr[i + 1];
        for (int j = start; j < end; j++)
        {
            sendbuf[j] = x[A.send_comm.idx[j]];
        }
        MPI_Isend(&(sendbuf[start]),
                  (int)(end - start),
                  MPI_DOUBLE,
                  proc,
                  tag,
                  MPI_COMM_WORLD,
                  &(A.send_comm.req[i]));
    }

    spmv(alpha, A.on_proc, x, beta, b);

    if (A.recv_comm.n_msgs)
    {
        MPI_Waitall(A.recv_comm.n_msgs, A.recv_comm.req.data(), MPI_STATUSES_IGNORE);
    }

    if (A.send_comm.n_msgs)
    {
        MPI_Waitall(A.send_comm.n_msgs, A.send_comm.req.data(), MPI_STATUSES_IGNORE);
    }

    spmv(alpha, A.off_proc, recvbuf, 1.0, b);
}

double inner_product(std::vector<double> a, std::vector<double> b)
{
    double sum, sum_local;

    sum_local = 0;
    for (int i = 0; i < a.size(); i++)
    {
        sum_local += a[i] * b[i];
    }

    MPI_Allreduce(&sum_local, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return sum;
}

std::tuple<int, double, double> CG(ParMat& A, std::vector<double>& x, std::vector<double>& b)
{
    double t0 = MPI_Wtime();
    // CG Variables
    std::vector<double> r(A.local_rows);
    std::vector<double> p(A.local_rows);
    std::vector<double> Ap(A.local_rows);
    std::vector<double> res;

    int iter, recompute_r;
    double alpha, beta;
    double rr_inner, next_inner, App_inner;
    double norm_r;
    double tol = ABCG::tol;
    // int max_iters = ((int)(1.3 * b.size())) + 2;
    int max_iters = ABCG::max_iters;

    std::fill(x.begin(), x.end(), 0);

    // r0 = b - A * x0
    r = b;
    spmv(-1.0, A, x, 1.0, r);

    // p0 = r0
    p = r;

    // Find initial (r, r) and residual
    rr_inner = inner_product(r, r);
    norm_r   = sqrt(rr_inner);
    res.push_back(norm_r);

    // Scale tolerance by norm_r
    if (norm_r != 0.0)
    {
        tol = tol * norm_r;
    }

    // How often should r be recomputed
    recompute_r = 8;
    iter        = 0;

    // Main CG Loop
    while (norm_r > tol && iter < max_iters)
    {
        // alpha_i = (r_i, r_i) / (A*p_i, p_i)
        spmv(1.0, A, p, 0.0, Ap);
        App_inner = inner_product(Ap, p);
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
            spmv(-1.0, A, x, 1.0, r);
        }

        next_inner = inner_product(r, r);
        beta       = next_inner / rr_inner;

        scale(beta, p);
        axpy(1.0, p, r);

        // Update next inner product
        rr_inner = next_inner;
        norm_r   = sqrt(rr_inner);

        res.push_back(norm_r);

        iter++;
    }
    return {iter, norm_r, (MPI_Wtime() - t0)};
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Init matrix from file
    std::vector<double> x;
    std::vector<double> b;
    ParMat A = initialize_cg(&argc, &argv, rank, x, b);

    // Set b to random values, x to 0
    srand(time(NULL) + rank);
    std::generate(x.begin(), x.end(), [&]() { return (double)(rand()) / RAND_MAX; });
    spmv(1.0, A, x, 0.0, b);
    double norm_b = calc_norm_b(b);

    MPI_Barrier(MPI_COMM_WORLD);
    std::tuple<int, double, double> results = CG(A, x, b);
    std::vector<double> r                   = b;
    spmv(-1.0, A, x, 1.0, r);

    int n_iters =
        calc_num_iters(r, rank, std::get<0>(results), norm_b, std::get<2>(results), "CG + MPI");

    for (int test = 0; test < ABCG::max_tests; test++)
    {
        double total_time = 0.0;
        MPI_Barrier(MPI_COMM_WORLD);

        for (int i = 0; i < n_iters; i++)
        {
            results = CG(A, x, b);
            total_time += std::get<2>(results);
        }
        total_time      = total_time / n_iters;
        double max_time = 0.0;
        MPI_Allreduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (rank == 0)
        {
            std::printf("CG with MPI Allreduce: %lg\n", max_time);
            std::cout << std::flush;
        }
    }

    MPI_Finalize();
}
