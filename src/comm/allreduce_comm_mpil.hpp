#ifndef ALLREDUCE_COMM_MPIL_HPP
#define ALLREDUCE_COMM_MPIL_HPP

#include "allreduce_comm.hpp"
#include "locality_aware.h"

class AllreduceCommMPIL : public AllreduceComm
{
    public:
        AllreduceCommMPIL() : AllreduceComm()
        {
        }

        ~AllreduceCommMPIL()
        {
        }
        
        void init(double* loc, double* glob)
        {
            local_sum = loc;
            global_sum = glob;
        }

        void start()
        {
            MPIL_Allreduce(local_sum, global_sum, 1, MPI_DOUBLE, MPI_SUM, 
                    MPIL_COMM_WORLD);        
        }

        void wait()
        {
        }
      
};

#endif

