************************************************************************
NVIDIA Jacobi MPI demonstrator
************************************************************************


This is the NVIDIA Jacobi MPI demonstration code (see the accompanying README 
file) with
some slight modifications to make its running on Bede simpler.

More information about the underlying problem and the solution implementation can be found at:
https://developer.nvidia.com/blog/benchmarking-cuda-aware-mpi

The earlier blog mentioned in the above article is also useful as an introduction to some of the issues relating to using MPI on multi-GPU
nodes. 

A tar file is available for use on RHEL 8 Bede images :download:`bedeRH8-jacobi.tgz<bedeRH8-jacobi.tgz>`.

There are two stages to running this benchmark code:

\1. Compile the codes using the included Makefile

\2. Run the codes using the included Slurm script as a template.

Making the codes
-----------------

Prerequisites:
``````````````

This demonstrator assumes you have loaded the module files to give you a 
CUDA-aware version of OpenMPI.

As per the MPI documentation, this is normally obtained by using the command

``module load gcc cuda openmpi``


Makefile:
`````````

The included Makefile has defined explicitly the environment variables
``MPI_HOME`` and ``CUDA_INSTALL_PATH`` consistent with the above module file.

The compilation options have been simplified to target the V100 GPU only.

The ``CFLAGS`` and ``NVCCFLAGS`` have been modified to work with gcc 8.3.0 
(so -mcpu=native is used instead of -march=native).

The target directory has been changed to be the current working directory
rather than a separate bin directory.

Type

::

  make all

to make all of the required object files and the two executables - one
which uses CUDA aware MPI and the other that uses normal MPI.

Run the codes
-----------------

sbatch job script
`````````````````

The sample script submits to two nodes, 8 GPUS in total. 

To have one MPI process per GPU, the script uses the mpirun option to restrict MPI processes so there are only two MPI processes
per socket, matching the number of GPUs per socket.

The executables assume that the Jacobi iteration takes place on a 2D mesh of 
MPI processes. At present the value of ROW and COL for the dimensions of this
2D mesh are defined manually before the first mpirun command.

The second set of arguments passed to the excutables is the local grid dimensions
for each GPU. The values currently assume a square 10,000 by 10,000 mesh on each GPU.
There is sufficient memory on each of Bede's GPUs to support mesh sizes up to
20,000 by 20,000.

Interesting issues
``````````````````````````````````

Wisdom would suggest that the CUDA aware MPI executable would be faster than 
the normal MPI executable. This appears to be not the case and this discrepency
has been observed on V100 GPUs with x86_64 processors as well with the Power9
processors on Bede.

The code developers made the decision to use different mechanisms to map MPI
tasks to GPUs. The approach taken in the CUDA Aware MPI code does not appear to
work on Bede nor on a system using x86_64 processors with 4 V100 GPUs per node.

The relevant code was therefore modified so both executables use the same
binding process for MPI tasks and GPUs and this appears to work correctly.


Performance
`````````````````

The Jacobi benchmark scales reasonably well on Bede. Performance tuning and / or tweaking the code to use non-blocking communication might
improve scalability further.

Some basic results.

With a local mesh size of 10,000 by 10,000 on 4 GPUs, so a 20,000 by 20,000 global mesh size, the CUDA Aware code variants obtains 459.48
Gflops whilst the code using "normal" MPI obtains 461.53 Gflops so just over 115 Gflops per GPU, which is respectable. 

On two nodes, with 8 GPUs, on the same local problem size, the  CUDA Aware code obtains 850.62 Gflops whilst the normal MPI obtains 873.29
Gflops. That is about 106 Gflops per GPU and 109 Gflops per GPU respectively. Reasonable scaling, but not ideal. nvprof gives some idea of
where time is being spent.

With the normal MPI over 8 GPUs with the local 10,000 by 10,000 grid, we have:

::

            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   49.17%  2.03328s      1000  2.0333ms  2.0217ms  2.0601ms  JacobiComputeKernel(double const *, double*, int4, int, double*)
                   48.61%  2.01030s      1000  2.0103ms  2.0051ms  2.0242ms  CopyBlockKernel(double const *, double*, int4, int)
                    1.76%  72.991ms      3001  24.322us     895ns  63.415ms  [CUDA memcpy HtoD]
                    0.17%  7.2056ms      3000  2.4010us  1.2480us  5.8880us  [CUDA memcpy DtoH]
                    0.15%  6.4033ms      1000  6.4030us  5.8240us  8.5120us  BufferToHaloKernel(double const *, double*, double*, int2, int, int)
                    0.08%  3.3359ms      1000  3.3350us  3.1680us  6.0800us  HaloToBufferKernel(double*, double const *, double const *, int2, int, int)
                    0.05%  2.0596ms         1  2.0596ms  2.0596ms  2.0596ms  [CUDA memcpy DtoD]
                    0.00%  2.3040us         2  1.1520us  1.1520us  1.1520us  [CUDA memset]
		    
Therefore almost half of the GPU time is spent doing / waiting for  communications-related activity in the CopyBlockKernel routine and 
CUDA memcpy between the device and the CPU.

The total job time is 4.5797 seconds, so there is an additional  half a second spent on the CPU and in the MPI communications not overlapping with
GPU activity. 	    
