template <typename SComm, typename AComm>
double time_function(int iter, 
        ParMat& A, 
        std::vector<double>& x, 
        std::vector<double>& b)
{
    MPI_Barrier(MPI_COMM_WORLD);
    double t0, tfinal;
    t0 = MPI_Wtime();
    for (int i = 0; i < iter; i++)
    {
        CG<SComm, AComm>(A, x, b);
    }
    tfinal = (MPI_Wtime() - t0) / iter;
    MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return t0;
}

template <typename SComm, typename AComm>
int calc_iters(ParMat& A, 
        std::vector<double>& x, 
        std::vector<double>& b)
{
    int iters = 1;
    double time;
    time = time_function<SComm, AComm>(iters, A, x, b);
    if (time > 0.1) 
        return iters;
    else
    {
        iters = 10;
        time = time_function<SComm, AComm>(iters, A, x, b);
        iters = 1.0 / time;
        if (iters < 1)
            iters = 1;
    }

    return iters;
}
