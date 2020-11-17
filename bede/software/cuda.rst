CUDA
====

CUDA (*Compute Unified Device Architecture*) 
is a parallel computing platform and application programming interface (API) model
created by NVIDIA.
It allows software developers to use a CUDA-enabled graphics processing unit (GPU)
for general purpose processing, 
an approach known as *General Purpose GPU* (GPGPU) computing.

Usage
-----

You need to first request one or more GPUs within an
:ref:`interactive session or batch job on a worker node <bede_scheduler>`. 


You then need to ensure a version of the CUDA library (and compiler) is loaded. CUDA version 
10.1, 10.2 is currently available on Bede:

.. code-block:: bash

   module load cuda/10.1
   module load cuda/10.2
   module load nvidia/20.5

The ``nvidia/20.5`` module contains CUDA 10.2 and additional profilers such as ``ncu``.

Confirm which version of CUDA you are using via ``nvcc --version`` e.g.: ::

   $ nvcc --version
   nvcc: NVIDIA (R) Cuda compiler driver
   Copyright (c) 2005-2019 NVIDIA Corporation
   Built on Thu_Oct_24_17:58:26_PDT_2019
   Cuda compilation tools, release 10.2, V10.2.89

Compiling a simple CUDA program
-------------------------------

An example of the use of ``nvcc`` (the CUDA compiler): ::

   nvcc filename.cu

will compile the CUDA program contained in the file ``filename.cu``.

Compiling the sample programs
-----------------------------

You do not need to be using a GPU-enabled node
to compile the sample programs
but you do need at least one GPU to run them.

In this demonstration, we create a batch job that 

#. Requests two GPUs, a single CPU core and 8GB RAM
#. Loads a module to provide CUDA 10.2
#. Downloads compatible NVIDIA CUDA sample programs
#. Compiles and runs an example that performs a matrix multiplication

.. code-block:: sh

   #!/bin/bash
   #SBATCH --gpus=2     # Number of GPUs
   #SBATCH --mem=8G
   #SBATCH --time=0-00:05        # time (DD-HH:MM)
   #SBATCH --job-name=gputest
   
   module load cuda/10.2  # provides CUDA 10.2
   
   mkdir -p $HOME/examples
   cd $HOME/examples
   if ! [[ -f cuda-samples/.git ]]; then
       git clone https://github.com/NVIDIA/cuda-samples.git cuda-samples
   fi 
   cd cuda-samples
   git checkout tags/v10.2 # use sample programs compatible with CUDA 10.2
   cd Samples/matrixMul
   make
   ./matrixMul

GPU Code Generation Options
---------------------------

To achieve the best possible performance whilst being portable, 
GPU code should be generated for the architecture(s) it will be executed upon.

This is controlled by specifying ``-gencode`` arguments to NVCC which, 
unlike the ``-arch`` and ``-code`` arguments, 
allows for 'fatbinary' executables that are optimised for multiple device architectures.

Each ``-gencode`` argument requires two values, 
the *virtual architecture* and *real architecture*, 
for use in NVCC's `two-stage compilation <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architectures>`_.
I.e. ``-gencode=arch=compute_70,code=sm_70`` specifies a virtual architecture of ``compute_70`` and real architecture ``sm_70``.

To support future hardware of higher compute capability, 
an additional ``-gencode`` argument can be used to enable Just in Time (JIT) compilation of embedded intermediate PTX code. 
This argument should use the highest virtual architecture specified in other gencode arguments 
for both the ``arch`` and ``code``
i.e. ``-gencode=arch=compute_70,code=compute_70``.

The minimum specified virtual architecture must be less than or equal to the `Compute Capability <https://developer.nvidia.com/cuda-gpus>`_ of the GPU used to execute the code.

GPU nodes in Bede contain Tesla V100 GPUs, which are compute capability 70.
To build a CUDA application which targets just the public GPUS nodes, use the following ``-gencode`` arguments: 

.. code-block:: sh

   nvcc filename.cu \
      -gencode=arch=compute_70,code=sm_70 \
      -gencode=arch=compute_70,code=compute_70 \

Further details of these compiler flags can be found in the `NVCC Documentation <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-steering-gpu-code-generation>`_, 
along with details of the supported `virtual architectures <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architecture-feature-list>`_ and `real architectures <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list>`_.

Documentation
-------------

* `CUDA Toolkit Documentation <https://docs.nvidia.com/cuda/index.html#axzz3uLoSltnh>`_


Determining the NVIDIA Driver version
-------------------------------------

Run the command:

.. code-block:: sh

   cat /proc/driver/nvidia/version

Example output is: ::

   NVRM version: NVIDIA UNIX ppc64le Kernel Module  440.64.00  Wed Feb 26 16:01:28 UTC 2020
   GCC version:  gcc version 4.8.5 20150623 (Red Hat 4.8.5-36) (GCC)
