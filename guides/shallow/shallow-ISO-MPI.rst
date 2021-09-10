
Exploring parallel potentials with a Fortran 90 benchmark code
==============================================================

One of my standard benchmark codes on CPU clusters is a variant of the shallow water solver.
The code I use is based on a Fortran 77 code written by  Paul N. Swarztrauber, National Center for Atmospheric
Research, Boulder, Colorado  in 1984.

I converted the code to use Fortran 90 array syntax, introduced OpenMP parallel constructs to improve single node parallelism and then
added MPI parallelism for multi-node benchmarking. 

The code is a good indicator of performance where there is a good deal of data movement along with a reasonable amount of stencil-based
floating point computation. It is akin to the NVIDIA Jacobi solver, but is a more realistic example of this family of 2D mesh codes. 

The code was sufficiently well tuned that Colfax International (https://www.colfax-intl.com) were interested in tuning the code to run on 
Intel Xeon Phi processors, back when they were still of interest. This resulted in the baseline code for the current benchmark as well as
the note "Cluster-Level Tuning of a Shallow Water Equation Solver on the Intel MIC Architecture" (2014, Andrey Vladimirov and Cliff Addison),
available at https://arxiv.org/abs/1408.1727. 

NVIDIA support for Fortran on their GPUs was, for many years, limited to CUDA Fortran calls and OpenACC support in the Portland Group Fortran 
compiler. In 2020, with the release of CUDA 11 and the associated NVIDIA HPC SDK, nvfortran became part of the free NVIDIA offering.
In addition, NVIDIA made substantial improvements in their support of OpenMP and ISO parallelism constructs in C++ and Fortran.

To access the relevant C++ and Fortran compilers as well as a bundled OpenMPI, you just need to load the ``nvhpc/21.5`` module.

OpenMP on NVIDIA GPUs deserves a separate article. The "older" OpenMP constructs such as parallel do, employed in this Shallow Water code,
are focussed on thread parallelism in a shared memory environment, which is not well supported on GPUs. 
For a few years, the OpenMP required to get good GPU performance can best be described as clunky. It worked, but needed careful
setting up. Today, with OpenMP 5, if the correct constructs are used, the offloading is seamless and performance on the GPUs can be very good. 
A current overview of OpenMP on GPUs can be found in the video https://www.youtube.com/watch?v=9w_2tj2uD4M (Best Practices for OpenMP on NVIDIA GPUs, 
December 2020).

However, NVIDIA now provide some support for ISO standard parallelisation constructs in C++ and Fortran. This note concentrates on the Fortran 2003
construct ``do concurrent``. An example code fragment is as follows:

::

 do concurrent (i = 1:n)
    r(i) = a(i) + b(i)
 enddo

Nested loops can be handled within one do concurrent command if the operations are loop independent. Indeed this is often the best way to maximimise
GPU performance and is akin to the way thread teams can be partitioned across nested loops explicitly in C++ kernel loops.

::

      do concurrent (j=sy:ey+1, i=sx:ex+1)
           psi(i,j) = a*sin((dble(i)-.5d0)*di)*sin((dble(j)-.5d0)*dj)
           p(i,j)=pcf*(cos(2.d0*dble(i-1)*di)+
     1                 cos(2.d0*dble(j-1)*dj))+base_pressure
           pold(i,j) = p(i,j)
      end do

A mixture of array syntax (for vectorisation) and do concurrent can also be used and is often performant:
 
::

       do concurrent  (j=1:np1)
        psi(1:mp1,j) = a*sin(di*(temp(1:mp1)-.5d0))*
     1  	         sin((dble(j)-.5d0)*dj)
        p(1:mp1,j)=pcf*(cos(2.d0*temp(0:m)*di)+cos(2*dble(j-1)*dj))
     1                + base_pressure
       end do

It was relatively simple to convert the original OpenMP code to use do concurrent with array syntax. 
The only issue is that reduction operations are not yet supported
properly, but this is coming. Fortunately, the test code only needs these operations at the end of the computaton, so performance was not
affected.

Using the original hybrid OpenMP / MPI code on 4 nodes of Skylake 6138 processors
with a total of 160 cores using 8 MPI
processes and 20 OpenMP threads per MPI process, the performance on the 20,000 by 20,000 problem is 160 Gflops - 1 Gflop per core.

On a single V100 GPU, the performance of the``do concurrent`` code on the 15,000 by 15,000 problem (20,000 by 20,000 is too big) is 110 Gflops. 
This is roughly equivalent to the performance on 110
Skylake cores, so 3 Skylake nodes are needed to outperform the single GPU. A V100 GPU costs about the same as a Skylake node, so there is 
a clear price performance advantage to the GPU provided the data fits on the GPU.

The compilation options for this code are important, so the compiler is directed to generate gpu code. Unified memory (specified by 
the ``gpu=managed`` option) also eliminates the need for explicit copying between the host and device memories:

::

 nvfortran -o shall_iso -stdpar=gpu -gpu=managed -DPGF90 -DSYSCLK shallow_iso.f
 
 

**Aside:** This same code can be recompiled to run on multicore systems. On a Skylake 6138 node with 40 cores, the performance on the
20,000 by 20,000 problem is 24.8 Gflops. No effort was made to tune the code with different combinations of nested do concurrent loops as well as
do concurrent with array operations, so the multicore performance might be able to be improved.

Therefore, expectations were high with the MPI code on multiple GPUs. However, performance was terrible. On 4 GPUs, the performance on the 
20,000 by 20,000 problem was 16 Gflops and only 10 Gflops with 8 GPUs. The MPI code had introduced a massive overhead. nvprof gives a clear indication
of what was happening. On one of the 8 GPUs, there was the following profile:

::

            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   41.67%  3.27465s   3520712     930ns     443ns  742.26us  [CUDA memcpy HtoD]
                   38.05%  2.98981s   3520352     849ns     348ns  853.04us  [CUDA memcpy DtoH]
                    5.33%  418.77ms        49  8.5464ms  8.5248ms  8.5635ms  shallow_377_gpu
                    3.71%  291.48ms        50  5.8296ms  4.7414ms  57.630ms  shallow_319_gpu
                    2.39%  187.90ms        50  3.7581ms  2.9695ms  41.606ms  shallow_315_gpu
                    1.93%  151.51ms        50  3.0301ms  2.3686ms  34.818ms  capuvhz_638_gpu
                    1.81%  142.13ms        50  2.8426ms  2.1880ms  34.191ms  capuvhz_641_gpu
                    1.39%  108.94ms        50  2.1787ms  1.5569ms  32.220ms  capuvhz_637_gpu
                    1.35%  106.38ms        50  2.1276ms  1.5000ms  32.434ms  capuvhz_635_gpu
                    0.98%  77.100ms         1  77.100ms  77.100ms  77.100ms  init_574_gpu
                    0.67%  52.886ms         1  52.886ms  52.886ms  52.886ms  init_586_gpu
                    0.63%  49.236ms         1  49.236ms  49.236ms  49.236ms  init_620_gpu
                    0.09%  6.8568ms         1  6.8568ms  6.8568ms  6.8568ms  shallow_392_gpu
					
Also, the total time taken was just under 126 seconds - so nearly all of the time was being spent on the CPU and most of the GPU time was
spent doing a massive number of copy operations between the host and device memory. 

A suspected cause for this massive data copy overhead was the use of an MPI derived type to handle the strided access across the data arrays
to communicate to the top and bottom neighbouring processes. The NVIDIA Jacobi code used explicit copying to and from buffers to handle the
strided access for communications, so this was an obvious thing to try with the shallow code.

The results were starkly better.

On 4 GPUs, the performance on the 20,000 by 20,000 problem was around 400 Gflops. With 8 GPUs, the performance was just under 800 Gflops.	
Unfortunately, the code is running quickly enough that the results of nvprof on multiple GPUs are all interlaced, so getting a coherent
picture of where time was being spent was challenging. On the 10,000 by 10,000 problem over 8 GPUs, a profile was obtained:

::

            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   30.99%  127.76ms        50  2.5552ms  2.0919ms  23.910ms  capuvhz_672_gpu
                   27.64%  113.96ms        50  2.2791ms  1.9408ms  17.569ms  shallow_314_gpu
                   24.96%  102.92ms        49  2.1004ms  2.0877ms  2.1187ms  shallow_398_gpu
                    6.54%  26.950ms         1  26.950ms  26.950ms  26.950ms  init_600_gpu
                    4.28%  17.657ms         1  17.657ms  17.657ms  17.657ms  init_658_gpu
                    3.10%  12.778ms         1  12.778ms  12.778ms  12.778ms  init_611_gpu
                    1.22%  5.0452ms      2123  2.3760us     796ns  751.93us  [CUDA memcpy HtoD]
                    0.78%  3.2158ms      1307  2.4600us     604ns  721.33us  [CUDA memcpy DtoH]
                    0.40%  1.6556ms         1  1.6556ms  1.6556ms  1.6556ms  shallow_413_gpu
                    0.05%  200.82us        50  4.0160us  2.0440us  8.6360us  capuvhz_719_gpu
                    0.03%  141.85us        50  2.8360us  1.3720us  5.7560us  shallow_353_gpu
                    0.00%  4.7640us         1  4.7640us  4.7640us  4.7640us  init_640_gpu

The total running time was 0.48 seconds (giving 670 Gflops), so nearly all of the time was spent on the GPUs and the vast majority of this
time was spent doing computation. Relatively little time was spent doing host to device memory copies, so using the explicit buffer coping
was critical to sensible performance. Further scalability tests will be run when the full Bede system has been converted to Red Hat 8.

The compilation  options used were:

::

 mpif90 -o shall_isompi -cuda -stdpar=gpu -gpu=managed  shallow_isompi.f

The mpirun options used were:

::

 GPUS=$((4*$SLURM_JOB_NUM_NODES))
 mpirun --mca btl_openib_warn_default_gid_prefix 0  -np  $GPUS --map-by ppr:2:socket  ./shall_isompi 					

The ``mca`` option is not strictly necessary, but it eliminates a misleading warning message.

**Summary:** With constraints, excellent performance can be obtained using modern ISO parallelisation on GPUs and good multinode GPU 
performance can be obtained by combining these parallel constructs with MPI. However, great caution must be used if MPI derived types are
exploited to simplify MPI communications. These may lead to very inefficient GPU codes.



