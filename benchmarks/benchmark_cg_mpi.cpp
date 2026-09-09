#include "cg.hpp"
#include "timer.hpp"

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (argc < 2)
    {
        if (rank == 0) printf("Need a matrix passed in command line\n");
        MPI_Abort(MPI_COMM_WORLD, 1); 
        return 1;
    }

    // Init matrix from file
    std::string filename = argv[1];
    ParMat A;
    readParMatrix(filename.c_str(), A);
    form_comm(A);

    // Create linear system
    std::vector<double> x(A.local_cols);
    std::vector<double> b(A.local_rows);
    std::vector<double> r(A.local_rows);  
    std::vector<double> sendbuf(A.send_comm.size_msgs);
    std::vector<double> recvbuf(A.recv_comm.size_msgs);
    double local_sum, global_sum;

    SpMVComm* spmv_comm = new SpMVComm();
    AllreduceComm* allreduce_comm = new AllreduceComm();
    spmv_comm->init(A, sendbuf.data(), recvbuf.data());
    allreduce_comm->init(&local_sum, &global_sum);

    // Set x to random values, b = A*x 
    srand(time(NULL) + rank);
    std::generate(x.begin(), x.end(), [&]() { return (double)(rand()) / RAND_MAX; });
    spmv(1.0, A, x, 0.0, b, spmv_comm);
    double norm_b = sqrt(inner_product(b, b, allreduce_comm));
    double norm_r;

    // Declare book keeping variables
    int n_iter, n_tests;
    double time;

    /*********************************************************
     * Test CG with Standard MPI
     ********************************************************/
    if (rank == 0) 
        printf("Testing CG with Standard MPI:\n");

    // Set x to zero, test CG
    std::fill(x.begin(), x.end(), 0);
    n_iter = CG<SpMVComm, AllreduceComm>(A, x, b);
    r = b;
    spmv(1.0, A, x, -1.0, r, spmv_comm);
    norm_r = sqrt(inner_product(r, r, allreduce_comm));
    if (rank == 0) 
        printf("After %d iterations, final res is %e\n", n_iter, norm_r / norm_b);

    n_tests = calc_iters<SpMVComm, AllreduceComm>(A, x, b);
    time = time_function<SpMVComm, AllreduceComm>(n_tests, A, x, b);
    if (rank == 0) 
        printf("CG takes %e seconds\n", time);

    /*********************************************************
     * Test CG with Persistent SpMVs
     ********************************************************/
    if (rank == 0) 
        printf("Testing CG with Persistent SpMVs:\n");

    // Set x to zero, test CG
    std::fill(x.begin(), x.end(), 0);
    n_iter = CG<SpMVCommPers, AllreduceComm>(A, x, b);
    r = b;
    spmv(1.0, A, x, -1.0, r, spmv_comm);
    norm_r = sqrt(inner_product(r, r, allreduce_comm));
    if (rank == 0) 
        printf("After %d iterations, final res is %e\n", n_iter, norm_r / norm_b);


    n_tests = calc_iters<SpMVCommPers, AllreduceComm>(A, x, b);
    time = time_function<SpMVCommPers, AllreduceComm>(n_tests, A, x, b);
    if (rank == 0) 
        printf("CG takes %e seconds\n", time);

    /*********************************************************
     * Test CG with Persistent Allreduce
     ********************************************************/
    if (rank == 0) 
        printf("Testing CG with Persistent Allreduce:\n");

    // Set x to zero, test CG
    std::fill(x.begin(), x.end(), 0);
    n_iter = CG<SpMVComm, AllreduceCommPers>(A, x, b);
    r = b;
    spmv(1.0, A, x, -1.0, r, spmv_comm);
    norm_r = sqrt(inner_product(r, r, allreduce_comm));
    if (rank == 0) 
        printf("After %d iterations, final res is %e\n", n_iter, norm_r / norm_b);


    n_tests = calc_iters<SpMVComm, AllreduceCommPers>(A, x, b);
    time = time_function<SpMVComm, AllreduceCommPers>(n_tests, A, x, b);
    if (rank == 0) 
        printf("CG takes %e seconds\n", time);

    /*********************************************************
     * Test CG with Persistent SpMVs and Allreduce
     ********************************************************/
    if (rank == 0)
        printf("Testing CG with Persistent SpMVs and Allreduce:\n");

    // Set x to zero, test CG
    std::fill(x.begin(), x.end(), 0);
    n_iter = CG<SpMVCommPers, AllreduceCommPers>(A, x, b);
    r = b;
    spmv(1.0, A, x, -1.0, r, spmv_comm);
    norm_r = sqrt(inner_product(r, r, allreduce_comm));
    if (rank == 0) 
        printf("After %d iterations, final res is %e\n", n_iter, norm_r / norm_b);


    n_tests = calc_iters<SpMVCommPers, AllreduceCommPers>(A, x, b);
    time = time_function<SpMVComm, AllreduceCommPers>(n_tests, A, x, b);
    if (rank == 0) 
        printf("CG takes %e seconds\n", time);

    delete spmv_comm;
    delete allreduce_comm;
    MPI_Finalize();

    return 0;
}
