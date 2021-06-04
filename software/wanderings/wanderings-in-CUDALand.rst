************************************************************************
Wanderings in CUDA land - Exploring parallel computing with CUDA
************************************************************************



Some background terminology
---------------------------

In the following, references are made frequently about CPU and GPU actions. This can be confusing shorthand.
As explained in most introductory tutorials about using / programming GPUs, (for instance 
https://developer.nvidia.com/blog/cuda-refresher-reviewing-the-origins-of-gpu-computing/ )
the GPU with its memory and processor is distinct from the conventional computer arrangement with one or more CPUs, 
memory attached directly to the CPUs and connections to storage devices and external connections. In other words, data
stored in the GPU memory is not normally accessible from the CPU and conversely, data in the CPU memory is not
normally accessible from the GPU. Until recently, users were responsible for the explicit copying of data to / from 
the GPU memory. Therefore, the reader should understand that the term CPU is shorthand for the compute performed by
the CPU using data in memory that is addressable (accessible) from that CPU. GPU refers to the GPU-specific domain. One profound
difference among the two domains, which is critical for this note, is that the way parallel computation is expressed and 
performed in the CPU and GPU domains is fundamentally different. 

Finally, in some of the comments of the accompanying codes, reference is often made to "the device". In this context, the device always refers to
the active GPU.

In order to make compuations on the GPU more accessible. NVIDIA, via its CUDA products, has provided an extensive programming environment and set of 
libraries for 
different functionalities that are invoked on
the CPU to perform on the GPU using data already on the GPU. These libraries include FFTs (libcufft.so), operations on sparse matrices (libcusparse.so) and
Basic Linear Algebra operations on dense vectors and matrices (libcublas.so). Some of our examples exploit routines in the libcublas.

NVIDIA also provides a C++ compatible compiler wrapper to make compiling some of CUDA-based codes easier, called nvcc. nvcc performs extensive 
preprocessing on C/C++ 
codes with the extension .cu. It also hides the linkage to basic CUDA support libraries.

A getting started example
-------------------------


The first code is a traditional CPU-GPU call of the float Level 3 BLAS matrix-matrix
multiplication, sgemm. The workflow in this program can be broken down as follows:

\1.  Create or read-in matrics on the CPU side.

\2.  Create companion matrices on the GPU. 

\3.  Create a handle for the CUBLAS context on the GPU.

\4.  Copy data from CPU to GPU. 

\5.  Perform computations on the GPU.

\6.  Copy results back to CPU for output and final processing.

\7.  Free up the storage used on both CPU and GPU side.

The matrices are initialised on the CPU, copied to 
similarly sized arrays on the GPU, the sgemm call is made on the GPU and then
the relevant array is copied back to the CPU to print out.


.. literalinclude:: sgemm-basic.cpp


The next code is nearly identical, but this time unified memory is used, so
that the data is not copied explicitly from the CPU to the GPU.

Unified memory first appeared in 2014, but spurred by the Summit and Sierra systems from IBM from 2018 onwards, it has undergone extensive revisions
behind the scenes to make it more efficient on different architectures. A good overview of unified memory can be found in the
article https://developer.nvidia.com/blog/unified-memory-cuda-beginners/ written by Mark Harris in 2017. The presentation 
https://on-demand.gputechconf.com/gtc/2018/presentation/s8430-everything-you-need-to-know-about-unified-memory.pdf is also good 
and this goes into details on how unified memory sits atop a virtual memory shared between processors with page access being used
to determine whether pages are physically located in the CPU or GPU memory (or on different GPUs). Mark also has an earlier Blog post about basic 
CUDA C/C++ programming that is worth reading at https://developer.nvidia.com/blog/even-easier-introduction-cuda/. 
 

Now, rather than creating shadow arrays on the GPU side that mirror those on the CPU with explicit copying between them, 
arrays are declared once via commands like ``cudaMallocManaged``. Such arrays can be accessed on both the CPU and GPU easily, but there
can be a significant performance hit if access patterns between the two domains are not managed intelligently.

.. literalinclude:: sgemm-unified.cu


Looking more carefully at unified memory
----------------------------------------

If we are going to generate large matrices for our tests, it is sensible
to do the generation / initialisation on the GPU. This requires a "slight"
diversion to learn about kernel functions and within GPU parallelism.

This first example is modified from Mark Harris's code in https://developer.nvidia.com/blog/even-easier-introduction-cuda/ to include initialisation on the GPU.
The code does a 1-D partitioning of the work among thread groups.

Note that the kernel functions init and add run on the GPU.  CUDA GPUs contain a number of threads, each of which can perform a computation independently. In addition,
there is a higher level of parallelism that boosts performance further. 

Quoting from Harris, 

"CUDA GPUs have many parallel processors grouped into Streaming Multiprocessors, or SMs. Each SM can run multiple concurrent thread blocks. 
As an example, a Tesla P100 GPU based on the Pascal GPU Architecture has 56 SMs, each capable of supporting up to 2048 active threads."

Without direction, all threads execute precisely the same code, effectivly serial performance. Parallel performance requires tying particular pieces of work to 
a specific thread id and doing this concurrently across all threads. 

Again, from Harris: "CUDA C++ provides keywords that let kernels get the indices of the running threads.  
Specifically, ``threadIdx.x`` contains the index of the current thread within its block, and ``blockDim.x`` contains the number of threads in the block."


.. literalinclude:: add_grid_init.cu 

It is worth examining a few parts of this code in more detail.

Notice the invocation of the kernel functions involves passing execution configuration specifying how many blocks of threads are used (referenced internally in a kernel
function as the gridDim) and the size of each block of threads (referenced internally in a kernel
function as the blockDim). Thread blocks have to have a multiple of 32 threads, with a maximum of 1024 threads in a block.

::


      int blockSize = 256;
      int numBlocks = (N + blockSize - 1) / blockSize;
      init<<<numBlocks, blockSize>>>(N, x, y);
      add<<<numBlocks, blockSize>>>(N, x, y);
      
The specification of the gridDim (numBlocks) and  blockDim (blockSize) for the kernel function is passed via metaparameters enclosed with ``<<<`` and ``>>>``. In this
example the int values are coerced into the underlying dim3 type, which can take 3 values, one value for each dimension of the gridDim amd blockDim. If only one
dimension is specified, the other two values are defaulted to 1. The need to preprocess such calls is part of the reason that CUDA C++ routines normally have the
extension ``.cu``. This alerts the CUDA nvcc compiler to the need for preprocessing.

In the kernel functions, there are other predefined identifiers to assist in specifying unique thread ids. 

::

      int index = blockIdx.x * blockDim.x + threadIdx.x;
      int stride = blockDim.x * gridDim.x;
      for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
	
Provided the value of ``gridDim.x`` is sufficiently large, that is ``blockDim.x * gridDim.x >= n`` the loop is effectively executed once with ``i == index``. This 
loop can therefore be considered as defensive programming to ensure that the full range of the array is covered exactly once by the pool of thread blocks. To confirm, in
this case:
::

 n = 1048576
 blockDim = 256
 gridDim = n / blockDim = 4096
 stride = n

There are alternative ways to protect the computations from trying to access past the end of arrays. A common alternative is given in the next example. The critical thing
to note is that by defining the number of grid blocks as a function of the array dimension and  thread pool size, it is possible to treat the array computation as being
completely parallel in its execution. The reality is that some thread scheduling will be needed, but this is performed seamlessly by the GPU, in a fashion that
is analogous to loop partitioning with OpenMP.

One important comment about kernel functions. The kernel functions are asynchronous with the CPU. It is therefore essential to impose a synchronisation between the
GPU and CPU via the command ``cudaDeviceSynchronize()`` before results are accessed from the CPU side.

This example also exploits another feature of unified memory. By default, entities are created in the CPU memory and migrated on demand to GPU memory. This can be
inefficient if we know in advance that the entities are going to be initialised and manipulated on the GPU. The command ``cudaMemPrefetchAsync`` can be used to specify
that a data area is to created on a particular GPU.


With matrices,
it is more natural to do a 2-D partitioning. Things are slightly more 
complicated, but the basic idea scales fairly well. Note that rather than strided access across both problem dimensions, this code just checks that the array indices are
in range.

.. literalinclude:: matadd.cu


Putting the pieces together
---------------------------

We are now at the point where we can do a larger test. Matrices a and b are
such that, when square, their product is the identity matrix (ones on the diagonal
and zeroes elsewhere). Matrix c is defined as an
identity matrix, so the final result is a matrix with two's down its
diagonal. I cannot find a reference to the specific formulas used for these matrices, but mathematically we have: 

.. math:: a_{ij} = b_{ij} = \sqrt{2 \over (m+1)} \times sin \left({(i+1) \times (j+1) \times \pi \over m+1}\right) \; i=0,..,m-1,\;  j=0,..,m-1

Notice in the kernel function ``initmatrix`` we have  redundant calculation, but the threads would have needed to wait for this computation anyway:

::

	pi = two*asin(one);
        rkplus1 = one/(float(m) + one);  
	rk = sqrt(two*rkplus1);
        if ( i < m && j < n) {
	  x[index] = rk*__sinf((float)(i+1)*(float)(j+1)*pi*rkplus1);
        }

The full code is below:


.. literalinclude:: sgemm-unifiedorthogV2a.cu

The performance on different systems is interesting and it also demonstrates the ease of using nvprof (only an abbreviated output is shown).

On a system with a Quadro P4000 GPU and Skylake 6138 processor (2.0 GHz) -  Driver Version: 460.32.03    CUDA Version: 11.2 

::

 Numblks x 625 Blksize x 32
 Initialise : 0.438705 sec .
 sgemm : 3.547108 sec .
 Gflops : 4511.054646 



            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   88.99%  3.54694s         1  3.54694s  3.54694s  3.54694s  maxwell_sgemm_128x128_nn
                    7.99%  318.64ms         2  159.32ms  125.55ms  193.09ms  initmatrix(int, int, float*)
                    3.01%  119.98ms         1  119.98ms  119.98ms  119.98ms  initident(int, int, float*)
                    0.00%     960ns         1     960ns     960ns     960ns  [CUDA memcpy HtoD]



On a system with a v100 and Cascade Lake 5218 (2.30GHz) -  Driver Version: 460.32.03    CUDA Version: 11.2 

::

 Numblks x 625 Blksize x 32
 Initialise : 0.254990 sec .
 sgemm : 1.299593 sec .
 Gflops : 12312.468032 

            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   82.77%  1.22389s         1  1.22389s  1.22389s  1.22389s  volta_sgemm_128x64_nn
                   11.72%  173.30ms         2  86.650ms  75.697ms  97.603ms  initmatrix(int, int, float*)
                    5.51%  81.526ms         1  81.526ms  81.526ms  81.526ms  initident(int, int, float*)
                    0.00%  1.8240us         1  1.8240us  1.8240us  1.8240us  [CUDA memcpy HtoD]
		    
On login2 of Bede - AC922 - v100 with Power 9 (3.8GHz) - Driver Version: 440.95.01    CUDA Version: 10.2

::

 Numblks x 625 Blksize x 32
 Initialise : 0.420643 sec .
 sgemm : 1.154472 sec .
 Gflops : 13860.183378 

            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   73.29%  1.15420s         1  1.15420s  1.15420s  1.15420s  volta_sgemm_128x32_sliced1x4_nn
                   18.77%  295.52ms         2  147.76ms  147.09ms  148.43ms  initmatrix(int, int, float*)
                    7.94%  125.04ms         1  125.04ms  125.04ms  125.04ms  initident(int, int, float*)
                    0.00%  1.6320us         1  1.6320us  1.6320us  1.6320us  [CUDA memcpy HtoD]

The slower initialise times on the Bede node are interesting. The reasons for this are not clear, but steps can be taken to improve the performance on all of the systems.


By default, Unified Memory allocates pages in the CPU memory and migrates them over to the GPU on demand. It is possible, using the ``cudaMemPrefetchAsync`` command to
preallocate memory on a particular GPU / device.

The relevant modified code to our earlier sgemm code is:

::

  	int device = -1; // For GPU number
	// unified memory for a,b,c
	cudaMallocManaged(&a, m*k * sizeof(float));
	cudaMallocManaged(&b, k*n * sizeof(float));
	cudaMallocManaged(&c, m*n * sizeof(float));
	cudaGetDevice(&device);
        cudaMemPrefetchAsync(a, m*k*sizeof(float), device, NULL);
        cudaMemPrefetchAsync(b, k*n*sizeof(float), device, NULL);
        cudaMemPrefetchAsync(c, m*n*sizeof(float), device, NULL);
	
Running the modified version of the code on our test platforms shows the change:

On a system with a Quadro P4000 GPU and Skylake 6138 processor (2.0 GHz) -  Driver Version: 460.32.03    CUDA Version: 11.2 

::

 Numblks x 625 Blksize x 32
 Initialise : 0.049530 sec . 
 sgemm : 3.599292 sec .
 Gflops : 4445.652504 

            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   98.65%  3.59913s         1  3.59913s  3.59913s  3.59913s  maxwell_sgemm_128x128_nn
                    1.00%  36.340ms         2  18.170ms  18.168ms  18.172ms  initmatrix(int, int, float*)
                    0.36%  13.072ms         1  13.072ms  13.072ms  13.072ms  initident(int, int, float*)

On a system with a v100 and Cascade Lake 5218 (2.30GHz) -  Driver Version: 460.32.03    CUDA Version: 11.2 

::


 Numblks x 625 Blksize x 32
 Initialise : 0.006248 sec .
 sgemm : 1.582949 sec .
 Gflops : 10108.474267 

            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.51%  1.23181s         1  1.23181s  1.23181s  1.23181s  volta_sgemm_128x64_nn
                    0.35%  4.3074ms         2  2.1537ms  2.1522ms  2.1552ms  initmatrix(int, int, float*)
                    0.14%  1.7876ms         1  1.7876ms  1.7876ms  1.7876ms  initident(int, int, float*)



On login1 of Bede - AC922 - v100 with Power 9 (3.8GHz) - Driver Version: 440.95.01    CUDA Version: 10.2

::

 Numblks x 625 Blksize x 32
 Initialise : 0.073072 sec .
 sgemm : 1.130069 sec .
 Gflops : 14159.493073 

            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.49%  1.12983s         1  1.12983s  1.12983s  1.12983s  volta_sgemm_128x32_sliced1x4_nn
                    0.36%  4.0385ms         2  2.0193ms  2.0175ms  2.0210ms  initmatrix(int, int, float*)
                    0.16%  1.7935ms         1  1.7935ms  1.7935ms  1.7935ms  initident(int, int, float*)


The initialisation times are dramatically reduced, indeed they are sufficiently low that we are running into timing accuracy of the ``clock_gettime`` function. It is best
to look at the times recorded directly by nvprof, where (as expected), there is little difference between the results on the two v100 systems. 

Since unified memory "works" from the GPU side of things and we are interested in the performance difference that Bede's fast CPU-GPU connections might provide, the above
experiments were repeated with the matrices a, b, and c created on the CPU and then migrated across on demand to the GPU.

The results are interesting and do appear to show that there are performance advantages on Bede.

On a system with a Quadro P4000 GPU and Skylake 6138 processor (2.0 GHz) -  Driver Version: 460.32.03    CUDA Version: 11.2 

::

 Initialise : 25.894818 sec .

 sgemm : 4.593165 sec .
 Gflops : 3483.698256 

            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  4.59301s         1  4.59301s  4.59301s  4.59301s  maxwell_sgemm_128x128_nn
 ...
 Device "Quadro P4000 (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
   81282  57.669KB  4.0000KB  0.9961MB  4.470348GB  485.5766ms  Host To Device
      14  73.143KB  4.0000KB  476.00KB  1.000000MB  91.07200us  Device To Host
    6262         -         -         -           -   1.632346s  Gpu page fault groups
 Total CPU Page faults: 13739
 
On a system with a v100 and Cascade Lake 5218 (2.30GHz) -  Driver Version: 460.32.03    CUDA Version: 11.2

::

 Initialise : 23.655406 sec .
 sgemm : 2.265335 sec .
 Gflops : 7063.503525 

            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  2.26486s         1  2.26486s  2.26486s  2.26486s  volta_sgemm_128x64_nn
 ...
 Device "Tesla V100-PCIE-16GB (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
   74761  56.784KB  4.0000KB  0.9961MB  4.048588GB  565.0188ms  Host To Device
      12  80.000KB  4.0000KB  476.00KB  960.0000KB  92.25500us  Device To Host
    3272         -         -         -           -   1.172802s  Gpu page fault groups
 Total CPU Page faults: 13738

On login1 of Bede - AC922 - v100 with Power 9 (3.8GHz) - Driver Version: 440.95.01    CUDA Version: 10.2

::

 Initialise : 18.209140 sec .
 sgemm : 1.926488 sec .
 Gflops : 8305.890756 

            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  1.92625s         1  1.92625s  1.92625s  1.92625s  volta_sgemm_128x32_sliced1x4_nn
 ...
 Device "Tesla V100-SXM2-32GB (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
   39141  119.52KB  64.000KB  960.00KB  4.461243GB  220.4134ms  Host To Device
       7  137.14KB  64.000KB  448.00KB  960.0000KB  39.52000us  Device To Host
    2085         -         -         -           -  877.0859ms  Gpu page fault groups
 Total CPU Page faults: 13739

The sgemm performance is impacted by the time required to fetch the data, but the interesting numbers are the times spent in paging from Host to Device:

::

 Device "Quadro P4000 (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
   81282  57.669KB  4.0000KB  0.9961MB  4.470348GB  485.5766ms  Host To Device
   
 Device "Tesla V100-PCIE-16GB (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
   74761  56.784KB  4.0000KB  0.9961MB  4.048588GB  565.0188ms  Host To Device
   
 Device "Tesla V100-SXM2-32GB (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
   39141  119.52KB  64.000KB  960.00KB  4.461243GB  220.4134ms  Host To Device
   
We can see that the time taken for moving the matrix data from the CPU to the GPU is more than halved on the Bede node. This does suggest there is some validity in the
assumption that the faster sub-systems are exploited seamlessly.         


Future Work
-----------

More carefuly designed experiments with Unified Memory are required.
Unified Memory exploits automatically the faster coupling between the CPU and GPU
on the AC922 than is found on GPU systems with x86_64 processors. Furthermore, Unified Memory can be oversubscribed and Unified Memory can be exploited both between the
CPU and GPU and in multi-GPU
programming within a node. Unified Memory can be exploited easily in Tensorflow. The PyTorch support appears to not be as advanced. 

Some interesting results on using Unified Memory with the RAPIDS software layer for data analytics and PyTorch for deep learning can be found in the 30 minute video
https://developer.nvidia.com/gtc/2019/video/s9726/video (registration and login is required). 

Many important questions remain, such as the extent to which the IBM Large Memory Support API makes it easier to exploit the CPU memory on the GPU, but as a first step,
Unified Memory appears to offer users a quick way to exploit Bede's best features.



