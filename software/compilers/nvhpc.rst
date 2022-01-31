.. _software-compilers-nvhpc:

NVIDIA HPC SDK
--------------

The `NVIDIA HPC SDK <https://developer.nvidia.com/hpc-sdk>`__, otherwise referred to as ``nvhpc``, is a suite of compilers, libraries and tools for HPC.
It provides C, C++ and Fortran compilers, which include features enabling GPU acceleration through standard C++ and Fortran, OpenACC directives and CUDA.

It is provided for use on the system by the ``nvhpc`` module(s).
It provides the ``nvc``, ``nvc++`` and ``nvfortran`` compilers.

This module also provides the `NCCL <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html>`__ and `NVSHMEM <https://docs.nvidia.com/hpc-sdk/nvshmem/index.html>`__ libraries, as well as the suite of math libraries typically included with the CUDA Toolkit, such as ``cublas``, ``cufft`` and ``nvblas``.

.. code-block:: bash

   module load nvhpc
   # RHEL 7 only
   module load nvhpc/20.9
   # RHEL 8 only 
   module load nvhpc/21.5

For further information please see the `NVIDIA HPC SDK Documentation Archive <https://docs.nvidia.com/hpc-sdk/archive/>`__.
