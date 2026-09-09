#ifndef ALLREDUCE_COMM_HPP
#define ALLREDUCE_COMM_HPP

#include "mpi.h"

class AllreduceComm
{
    public:
        AllreduceComm()
        {
            local_sum = NULL;
            global_sum = NULL;
        }

        ~AllreduceComm()
        {

        }
        
        void init(double* loc, double* glob)
        {
            local_sum = loc;
            global_sum = glob;
        }

        void start()
        {
             MPI_Allreduce(local_sum, global_sum, 1, MPI_DOUBLE, MPI_SUM, 
                  MPI_COMM_WORLD);
        }

        void wait()
        {

        }

        void set_local_sum(double sum)
        {
            *local_sum = sum;
        }

        double get_global_sum()
        {
            return *global_sum;
        }
      

    protected:
        double* local_sum;
        double* global_sum;
};

#endif

