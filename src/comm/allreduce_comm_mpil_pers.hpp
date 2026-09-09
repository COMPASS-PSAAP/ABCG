#ifndef ALLREDUCE_COMM_MPIL_PERS_HPP
#define ALLREDUCE_COMM_MPIL_PERS_HPP

#include "allreduce_comm.hpp"
#include "locality_aware.h"

class AllreduceCommMPILPers : public AllreduceComm
{
    public:
        AllreduceCommMPILPers() : AllreduceComm()
        {
            req = NULL;
        }

        ~AllreduceCommMPILPers()
        {
            if (req)
            {
                MPIL_Request_free(&req);
            }
        }
        
        void init(double* loc, double* glob)
        {
            local_sum = loc;
            global_sum = glob;

            MPIL_Info* mpil_info;
            MPIL_Info_init(&mpil_info);

            MPIL_Allreduce_init(local_sum, global_sum, 1, MPI_DOUBLE, MPI_SUM,
                    MPIL_COMM_WORLD, mpil_info, &req);

            MPIL_Info_free(&mpil_info);
        }

        void start()
        {
            MPIL_Start(req);        
        }

        void wait()
        {
            MPIL_Wait(req, MPI_STATUS_IGNORE);
        }
      

    private:
        MPIL_Request* req;
};

#endif

