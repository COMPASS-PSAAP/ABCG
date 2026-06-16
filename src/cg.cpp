#include "sparse_mat.hpp"
#include "par_binary_IO.hpp"
#include "locality_aware.h"
#include <math.h>
#include <random>

#include "utils.hpp"

// Serial SpMV b = alpha*A*x + beta*b
void spmv(double alpha, Mat& A, std::vector<double>& x,
        double beta, std::vector<double>& b)
{
    double sum;
    int start, end;

    for (int i = 0; i < A.n_rows; i++)
    {
        start = A.rowptr[i];
        end = A.rowptr[i+1];
        sum = 0;
        for (int j = start; j < end; j++)
        {
            sum += A.data[j] * x[A.col_idx[j]];
        }
        b[i] = alpha * sum + beta * b[i];
    }
}


__global__ void pack(const double* __restrict__ x,
                    const int* __restrict__ idx,
                    double* __restrict__ packed_buf,
                    int n)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) packed[i] = x[idx[i]];
}


// Parallel SpMV b = alpha*A*x + beta*b 
void spmv(rocsparse_handle handle, rocsparse_spmat_descr A,
            double alpha, roscparse_dnvec_descr x, 
            double beta, rocsparse_dnvec_descr y,
            size_t* tmp_buffer_size, void** tmp_buffer)
{
    ROCSPARSE_CHECK(rocsparse_spmv(handle, rocsparse_operation_none,
            &alpha, A, x, &beta, y, 
            rocsparse_datatype_f64_r,
            rocsparse_spmv_alg_default,
            rocsparse_spmv_stage_buffer_size,
            tmp_buffer_size, tmp_buffer);
}

void spmv(double alpha, ParMat& A, std::vector<double>& x, 
        double beta, std::vector<double>& b, MPIL_Comm* mpil_comm)
{
    int proc, start, end;
    int tag = 0;
    std::vector<double> recvbuf(A.recv_comm.size_msgs);
    std::vector<double> sendbuf(A.send_comm.size_msgs);

    MPIL_Info* mpil_info;
    MPIL_Info_init(&mpil_info);

    MPIL_Topo* mpil_topo;
    MPIL_Topo_init(A.recv_comm.n_msgs,
            A.recv_comm.procs.data(),
            MPI_UNWEIGHTED,
            A.send_comm.n_msgs,
            A.send_comm.procs.data(),
            MPI_UNWEIGHTED,
            mpil_info,
            &mpil_topo);

    // Launch Pack Kernel -- Pack Send Buffer
    if (A.send_comm.size_msgs)
    {
        dim3 threads(256);
        dim3 blocks((A.send_comm.size_msgs + threads.x - 1) / threads.x);
        pack<<<blocks, threads, 0, 0>>>(x_d, A.send_comm.idx_d,
                sendbuf_d, A.send_comm.size_msgs);
        HIP_CHECK(hipMemcpy(sendbuf_h, sendbuf_d, A.send_comm.size_msgs * sizeof(double),
                hipMemcpyDeviceToHost));
    }

    MPIL_Neighbor_alltoallv_topo(sendbuf_d
            A.send_comm.counts.data(),
            A.send_comm.ptr.data(),
            MPI_DOUBLE,
            recvbuf_d
            A.recv_comm.counts.data(),
            A.recv_comm.ptr.data(),
            MPI_DOUBLE,
            mpil_topo,
            mpil_comm);

    spmv(

    spmv(alpha, A.on_proc, x, beta, b);

    spmv(alpha, A.off_proc, recvbuf, 1.0, b);

    MPIL_Info_free(&mpil_info);
    MPIL_Topo_free(&mpil_topo);
}

void spmv(double alpha, ParMat& A, std::vector<double>& x, 
        double beta, std::vector<double>& b, MPIL_Comm* mpil_comm,
        std::vector<double>& sendbuf, std::vector<double>& recvbuf,
        MPIL_Request* req)
{
    if (req != NULL)
    {
        int proc, start, end;
        int tag = 0;

        // Launch Pack Kernel -- Pack Send Buffer
        if (A.send_comm.size_msgs)
        {
            dim3 threads(256);
            dim3 blocks((A.send_comm.size_msgs + threads.x - 1) / threads.x);
            pack<<<blocks, threads, 0, 0>>>(x_d, A.send_comm.idx_d, 
                    sendbuf_d, A.send_comm.size_msgs);
            HIP_CHECK(hipMemcpy(sendbuf_h, sendbuf_d, A.send_comm.size_msgs * sizeof(double),
                    hipMemcpyDeviceToHost));
        }

        MPIL_Start(req);

        spmv(alpha, A.on_proc, x, beta, b);
        spmv(A.sparse_handle, A.on_proc.descr, alpha, A.descr_x, 
                beta, A.descr_b, A.on_proc.buf_s, A.on_proc.buf);

        MPIL_Wait(req, MPI_STATUS_IGNORE);

        spmv(alpha, A.off_proc, recvbuf, 1.0, b);
        spmv(A.handle, A.off_proc.descr, 
    }
    else
    {
        spmv(alpha, A, x, beta, b, mpil_comm);
    }
}

void axpy(double alpha, std::vector<double>& x, std::vector<double>& y)
{
    for (int i = 0; i < x.size(); i++)
        x[i] = x[i] + alpha*y[i];
}

void scale(double alpha, std::vector<double>& x)
{
    for (int i = 0; i < x.size(); i++)
        x[i] = alpha*x[i];
}

double inner_product(std::vector<double>& a, std::vector<double>& b,
            double* local_sum_ptr, double* global_sum_ptr,
            MPIL_Comm* mpil_comm, MPIL_Request* mpil_req)
{
    *local_sum_ptr = 0;
    for (int i = 0; i < a.size(); i++)
        *local_sum_ptr += a[i] * b[i];

    if (mpil_req == NULL)
    {
        MPIL_Allreduce(local_sum_ptr, global_sum_ptr, 1, MPI_DOUBLE, MPI_SUM,
                mpil_comm);
    }
    else
    {
        MPIL_Start(mpil_req);
        MPIL_Wait(mpil_req, MPI_STATUS_IGNORE);
    }
    return *global_sum_ptr;
}

int CG(ParMat& A, std::vector<double>& x, std::vector<double>& b,
        bool spmv_init, bool allreduce_init)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // CG Variables
    std::vector<double> r(A.local_rows);
    std::vector<double> p(A.local_rows);
    std::vector<double> Ap(A.local_rows);
    std::vector<double> res;
    std::vector<double> recvbuf(A.recv_comm.size_msgs);
    std::vector<double> sendbuf(A.send_comm.size_msgs);

    // Setup persistent allreduces
    double local_sum, global_sum;
    MPIL_Request* mpil_req = NULL;
    MPIL_Request* mpil_spmv_req = NULL;
    MPIL_Comm* mpil_comm;
    MPIL_Comm_init(&mpil_comm, MPI_COMM_WORLD);
    MPIL_Info* mpil_info;
    MPIL_Info_init(&mpil_info);
    MPIL_Topo* mpil_topo = NULL;

    if (spmv_init)
    {
        MPIL_Topo_init(A.recv_comm.n_msgs,
                A.recv_comm.procs.data(),
                MPI_UNWEIGHTED,
                A.send_comm.n_msgs,
                A.send_comm.procs.data(),
                MPI_UNWEIGHTED,
                mpil_info,
                &mpil_topo);
        std::vector<long> global_send_idx(A.send_comm.size_msgs);
        for (int i = 0; i < A.send_comm.size_msgs; i++)
            global_send_idx[i] = A.send_comm.idx[i] + A.first_row;
        MPIL_Neighbor_alltoallv_init_ext_topo(sendbuf.data(), 
                A.send_comm.counts.data(),
                A.send_comm.ptr.data(),
                global_send_idx.data(),
                MPI_DOUBLE,
                recvbuf.data(),
                A.recv_comm.counts.data(),
                A.recv_comm.ptr.data(),
                A.off_proc_columns.data(),
                MPI_DOUBLE,
                mpil_topo,
                mpil_comm,
                mpil_info,
                &mpil_spmv_req);
    }

    if (allreduce_init)
    {
        MPIL_Allreduce_init(&local_sum, &global_sum, 1, MPI_DOUBLE,
           MPI_SUM, mpil_comm, mpil_info, &mpil_req); 
    }

    int iter, recompute_r;
    double alpha, beta;
    double rr_inner, next_inner, App_inner;
    double norm_r, tol = 1e-6;
    //int max_iter = ((int)(1.3*b.size())) + 2;
    int max_iter = 500;

    // r0 = b - A * x0
    r = b;
    spmv(-1.0, A, x, 1.0, r, mpil_comm, sendbuf, recvbuf,
            mpil_spmv_req);

    // p0 = r0
    p = r;

    // Find initial (r, r) and residual
    rr_inner = inner_product(r, r, &local_sum, &global_sum,
            mpil_comm, mpil_req);

    norm_r = sqrt(rr_inner);
    res.push_back(norm_r);

    // Scale tolerance by norm_r
    if (norm_r != 0.0)
    {
        tol = tol * norm_r;
    }

    // How often should r be recomputed
    recompute_r = 8;
    iter = 0;

    // Main CG Loop
    while (norm_r > tol && iter < max_iter)
    {
        // alpha_i = (r_i, r_i) / (A*p_i, p_i)
        spmv(1.0, A, p, 0.0, Ap, mpil_comm, sendbuf, recvbuf, 
                mpil_spmv_req);
        App_inner = inner_product(Ap, p, &local_sum, &global_sum,
            mpil_comm, mpil_req);
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
            axpy(-1.0*alpha, r, Ap);
        }
        else
        {
            r = b;
            spmv(-1.0, A, x, 1.0, r, mpil_comm, sendbuf, recvbuf, 
                    mpil_spmv_req);
        }

        next_inner = inner_product(r, r, &local_sum, &global_sum,
                mpil_comm, mpil_req);
        beta = next_inner / rr_inner;

        scale(beta, p);
        axpy(1.0, p, r);

        // Update next inner product
        rr_inner = next_inner;
        norm_r = sqrt(rr_inner);

        res.push_back(norm_r);

        iter++;
    }

    if (mpil_req != NULL)
        MPIL_Request_free(&mpil_req);
    if (mpil_spmv_req != NULL)
    {
        MPIL_Request_free(&mpil_spmv_req);
        MPIL_Topo_free(&mpil_topo);
    }
    MPIL_Info_free(&mpil_info);
    MPIL_Comm_free(&mpil_comm);

    return iter;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    double t0, tfinal;

    MPIL_Comm* mpil_comm;
    MPIL_Comm_init(&mpil_comm, MPI_COMM_WORLD);

    MPIL_Comm_topo_init(mpil_comm);
    MPIL_Comm_device_init(mpil_comm);

    const char* filename = "Dubcova2.pm";
    if (argc > 1)
    {
        filename = argv[1];
    }

    ParMat A;

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    readParMatrix(filename, A);
    tfinal = MPI_Wtime() - t0;
    MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) printf("Read matrix: %e\n", t0);
    fflush(stdout);

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    form_comm(A);
    tfinal = MPI_Wtime() - t0;
    MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) printf("Form comm: %e\n", t0);
    fflush(stdout);

    copy_to_device(A);

    std::vector<double> x(A.local_cols);
    std::vector<double> b(A.local_rows);
    double *x_d, *b_d;
    HIP_CHECK(hipMalloc((void**)&x_d, A.local_cols*sizeof(double)));
    HIP_CHECK(hipMalloc((void**)&b_d, A.local_rows*sizeof(double)));

    double* sendbuf = NULL;
    if (A.send_comm.size_msgs)
    {
        HIP_CHECK(hipMalloc((void**)&sendbuf, 
                A.send_comm.size_msgs*sizeof(double)));
    }

    double* recvbuf = NULL;
    if (A.recv_comm.size_msgs)
    {
        HIP_CHECK(hipMalloc((void**)&recvbuf,
                A.recv_comm.size_msgs*sizeof(double)));
    }

    rocsparse_dnvec_descr vec_x, vec_b, vec_recv;
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vec_x, A.local_cols, x_d, 
            rocsparse_datatype_f64_r));
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vec_b, A.local_rows, b_d,
            rocsparse_datatype_f64_r));
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vec_recv, A.recv_comm.size_msgs, 
                recvbuf, rocsparse_datatype_f64_r));


    // Initialize SpMV Buffers
    spmv(A.sparse_handle, A.on_proc.descr, 1.0, vec_x,
            0.0, vec_b, &A.on_proc.buf_size, NULL);
    if (A.on_proc.buf_size)
    {
        
        HIP_CHECK(hipMalloc(&A.on_proc.buffer, 
    spmv(A.sparse_handle, A.off_proc.descr, 1.0, vec_recv, 
            1.0, vec_b, &A.off_proc.buf_size, NULL);


void spmv(rocsparse_handle handle, rocsparse_spmat_descr A,
            double alpha, roscparse_dnvec_descr x,
            double beta, rocsparse_dnvec_descr y,
            size_t* tmp_buffer_size, void** tmp_buffer)

    // Set x to random values, b = A*x
    // Will reset x to 0 before each CG
    std::mt19937 rng(rank + 12345);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::generate(x.begin(), x.end(),
              [&]() { return dist(rng); });
    spmv(1.0, A, x, 0.0, b, mpil_comm);

    int n_iters;
    int conv_iter;
    std::vector<double> r;
    double sum;
    double norm_b;
    norm_b = 0;
    for (int i = 0; i < b.size(); i++)
        norm_b += b[i] * b[i];
    MPI_Allreduce(MPI_IN_PLACE, &norm_b, 1, MPI_DOUBLE, MPI_SUM,
            MPI_COMM_WORLD);
    norm_b = sqrt(norm_b);

    std::vector<NeighborAlltoallvMethod> neighbor_methods = {
            NEIGHBOR_ALLTOALLV_STANDARD, 
            NEIGHBOR_ALLTOALLV_LOCALITY
            };
    std::vector<NeighborAlltoallvInitMethod> neighbor_init_methods = {
            NEIGHBOR_ALLTOALLV_INIT_STANDARD, 
            NEIGHBOR_ALLTOALLV_INIT_LOCALITY
            };
    std::vector<const char*> neighbor_names = {
            "Standard", 
            "Locality", 
            "Pers Standard", 
            "Pers Locality"
            };
    std::vector<bool> neighbor_persistent = {false, false, true, true};

/*
    std::vector<NeighborAlltoallvMethod> neighbor_methods = {
        NEIGHBOR_ALLTOALLV_STANDARD};
    std::vector<NeighborAlltoallvInitMethod> neighbor_init_methods;
    std::vector<const char*> neighbor_names = {
        "Standard" };
    std::vector<bool> neighbor_persistent = {false};


    std::vector<AllreduceMethod> methods = {
            ALLREDUCE_PMPI,
            ALLREDUCE_RMA_HIERARCHICAL,
            ALLREDUCE_RMA_HIERARCHICAL_EARLYBIRD};
    std::vector<const char*> names = {
            "PMPI",
            "MPIL RMA Hier Pers", 
            "MPIL RMA Hier EB Pers"};
    std::vector<bool> persistent = {false, true, true};

*/

    std::vector<AllreduceMethod> methods = {
            ALLREDUCE_PMPI, 
            ALLREDUCE_RECURSIVE_DOUBLING, 
            ALLREDUCE_DISSEMINATION_LOC, 
            ALLREDUCE_DISSEMINATION_ML, 
            ALLREDUCE_DISSEMINATION_RADIX,
            ALLREDUCE_RECURSIVE_DOUBLING, 
            ALLREDUCE_DISSEMINATION_LOC, 
            ALLREDUCE_DISSEMINATION_ML, 
            ALLREDUCE_DISSEMINATION_RADIX, 
            ALLREDUCE_RMA_HIERARCHICAL,
            ALLREDUCE_RMA_HIERARCHICAL_EARLYBIRD, 
            //ALLREDUCE_RMA_MULTILEADER,
            //ALLREDUCE_RMA_MULTILEADER_EARLYBIRD
            };
    std::vector<const char*> names = {
            "PMPI", 
            "MPIL RD", 
            "MPIL NA", 
            "MPIL LA", 
            "MPIL RADIX",
            "MPIL RD Pers", 
            "MPIL NA Pers", 
            "MPIL LA Pers", 
            "MPIL RADIX Pers",
            "MPIL RMA Hier Pers", 
            "MPIL RMA Hier EB Pers", 
            //"MPIL RMA ML Pers", 
            //"MPIL RMA ML EB Pers"
            };
    std::vector<bool> persistent = {
            false,
            false,
            false,
            false,
            false,
            true,
            true,
            true,
            true,
            true,
            true,
            //true,
            //true
            };

    for (int neigh_idx = 0; neigh_idx < neighbor_names.size(); neigh_idx++)
    {
        if (rank == 0) printf("Running with %s Neighbor Collectives\n", 
                    neighbor_names[neigh_idx]);

        bool persistent_spmv = neighbor_persistent[neigh_idx];
        if (persistent_spmv)
            MPIL_Set_alltoallv_neighbor_init_algorithm(
                        neighbor_init_methods[neigh_idx - neighbor_methods.size()]);
        else
            MPIL_Set_alltoallv_neighbor_algorithm(
                        neighbor_methods[neigh_idx]);


        for (int idx = 0; idx < names.size(); idx++)
        {
            MPIL_Set_allreduce_algorithm(methods[idx]);
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            std::fill(x.begin(), x.end(), 0);
            conv_iter = CG(A, x, b, persistent_spmv, persistent[idx]);
            tfinal = (MPI_Wtime() - t0);
            r = b;
            spmv(-1.0, A, x, 1.0, r, mpil_comm);
            sum = 0;
            for (int i = 0; i < r.size(); i++)
                sum += r[i] * r[i];
            MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
            if (rank == 0) printf("CG + %s: %d iter, norm %e\n", 
                    names[idx], conv_iter, sqrt(sum) / norm_b);

            n_iters = 1;
            MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (t0 < 1.0)
                n_iters = 1.0 / t0;

            for (int test = 0; test < 5; test++)
            {
                MPI_Barrier(MPI_COMM_WORLD);
                t0 = MPI_Wtime();
                for (int i = 0; i < n_iters; i++)
                {
                    std::fill(x.begin(), x.end(), 0);
                    CG(A, x, b, persistent_spmv, persistent[idx]);
                }
                tfinal = (MPI_Wtime() - t0) / n_iters;
                MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 
                        MPI_COMM_WORLD);
                if (rank == 0) printf("CG with %s Allreduce: %e\n", 
                        names[idx], t0);
            }
        }

    }
    MPIL_Comm_free(&mpil_comm);

    MPI_Finalize();
}
