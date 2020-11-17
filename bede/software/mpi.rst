.. _bede_mpi:

MPI
===

Message Passing Interface (MPI) is a standardized and portable message-passing standard designed by a group of researchers from academia and industry to function on a wide variety of parallel computing architectures.

Bede provides four MPI libraries for use depending on the application, OpenMPI, IBM Spectrum MPI, MVAPICH 2, and Mellanox HPC-X.

OpenMPI
-------

The OpenMPI Project is an open source Message Passing Interface implementation that is developed and maintained by a consortium of academic, research, and industry partners. OpenMPI is therefore able to combine the expertise, technologies, and resources from all across the High Performance Computing community in order to build the best MPI library available. OpenMPI offers advantages for system and software vendors, application developers and computer science researchers.

Versions
^^^^^^^^

You can load a specific version using one of the following: ::

    module load gcc/openmpi-3.0.3
    module load gcc/openmpi-4.0.3rc4


IBM Spectrum MPI
----------------

`IBM® Spectrum MPI <https://www.ibm.com/uk-en/marketplace/spectrum-mpi>`_ is a high-performance, production-quality implementation of Message Passing Interface (MPI). It accelerates application performance in distributed computing environments. It provides a familiar portable interface based on the open-source MPI. It goes beyond Open MPI and adds some unique features of its own, such as advanced CPU affinity features, dynamic selection of interface libraries, superior workload manager integrations and better performance. IBM Spectrum MPI supports a broad range of industry-standard platforms, interconnects and operating systems, helping to ensure that parallel applications can run almost anywhere.


Versions
^^^^^^^^

You can load the library using: ::

    module load ibm/spectrum-mpi-10.03.01.00


MVAPICH2
--------

`MVAPICH2 <https://mvapich.cse.ohio-state.edu/>`_ (pronounced as “em-vah-pich 2”) is an open-source MPI software to exploit the novel features and mechanisms of high-performance networking technologies (InfiniBand, 10GigE/iWARP and 10/40GigE RDMA over Converged Enhanced Ethernet (RoCE)) and deliver best performance and scalability to MPI applications. This software is developed in the Network-Based Computing Laboratory (NBCL), headed by Prof. Dhabaleswar K. (DK) Panda since 2001.

Versions
^^^^^^^^

Versions are available for the GCC and IBM XL compiler: ::

    module load mvapich2/gdr/2.3.4-1.mofed4.7.gnu4.8.5-2
    module load mvapich2/gdr/2.3.4-1.mofed4.7.xlc16.01


Mellanox HPCX OpenMPI
---------------------

To meet the needs of scientific research and engineering simulations, supercomputers are growing at an unrelenting rate. The `Mellanox HPC-X ScalableHPC Toolkit <https://www.mellanox.com/products/hpc-x-toolkit>`_ is a comprehensive MPI and SHMEM/PGAS software suite for high performance computing environments. HPC-X provides enhancements to significantly increase the scalability and performance of message communications in the network. HPC-X enables you to rapidly deploy and deliver maximum application performance without the complexity and costs of licensed third-party tools and libraries.

Versions
^^^^^^^^

You can load the library using one of the following: ::

    module load gcc/hpcx/2.6.0-OFED-4.7.1.0.1/hpcx-mt-ompi
    module load gcc/hpcx/2.6.0-OFED-4.7.1.0.1/hpcx-ompi


Using MPI
---------

The following examples applies to all MPI libraries, change the ``module load`` to load your required libary.

Example
^^^^^^^

Consider the following source code (hello.c):

.. code-block:: c

    #include <mpi.h>
    #include <stdio.h>

    int main(int argc, char** argv) {
        // Initialize the MPI environment
        MPI_Init(NULL, NULL);

        // Get the number of processes
        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        // Get the rank of the process
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        // Get the name of the processor
        char processor_name[MPI_MAX_PROCESSOR_NAME];
        int name_len;
        MPI_Get_processor_name(processor_name, &name_len);

        // Print off a hello world message
        printf("Hello world from processor %s, rank %d out of %d processors\n",
                processor_name, world_rank, world_size);

        // Finalize the MPI environment.
        MPI_Finalize();
    }

MPI_COMM_WORLD (which is constructed for us by MPI) encloses all of the processes in the job, so this call should return the amount of processes that were requested for the job.

Compile your source code by using one of the following commands: ::

    mpic++ hello.cpp -o file
    mpicxx hello.cpp -o file
    mpicc hello.c -o file
    mpiCC hello.c -o file


Interactive job submission
--------------------------


You can run your job interactively: ::

    srun file

Your output would be something like: ::

    Hello world from processor gpu001.bede.dur.ac.uk, rank 0 out of 1 processors


This is an expected behaviour since we did not specify the number of CPU cores when requesting our interactive session.
You can request an interactive node with multiple cores (4 in this example) by using the command: ::

    srun --ntasks=4 --pty bash -i

Please note that requesting multiple cores in an interactive node depends on the availability. During peak times, it is unlikely that you can successfully request a large number of cpu cores interactively.  Therefore, it may be a better approach to submit your job non-interactively. 


Non-interactive job submission
------------------------------

Write a shell script (minimal example) We name the script as ‘test.sh’: ::


    #!/bin/bash
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=40

    module load OpenMPI/3.1.3-GCC-8.2.0-2.31.1

    srun --export=ALL file

Maximum 40 cores can be requested.

Submit your script by using the command: ::

    sbatch test.sh

Your output would be something like: ::

    Hello world from processor gpu004.bede.dur.ac.uk, rank 24 out of 40 processors
    Hello world from processor gpu001.bede.dur.ac.uk, rank 5 out of 40 processors
    ...
    Hello world from processor gpu006.bede.dur.ac.uk, rank 31 out of 40 processors
    Hello world from processor gpu039.bede.dur.ac.uk, rank 32 out of 40 processors



