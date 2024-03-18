.. _software-compilers-nvhpc:

NVIDIA HPC SDK
--------------

The `NVIDIA HPC SDK <https://developer.nvidia.com/hpc-sdk>`__, otherwise referred to as ``nvhpc``, is a suite of compilers, libraries and tools for HPC.
It provides C, C++ and Fortran compilers, which include features enabling GPU acceleration through standard C++ and Fortran, OpenACC directives and CUDA.

It is provided for use on the system by the ``nvhpc`` module(s).
It provides the ``nvc``, ``nvc++`` and ``nvfortran`` compilers.

This module also provides the `NCCL <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html>`__ and `NVSHMEM <https://docs.nvidia.com/hpc-sdk/nvshmem/index.html>`__ libraries, as well as the suite of math libraries typically included with the CUDA Toolkit, such as ``cublas``, ``cufft`` and ``nvblas``.

.. tabs::

   .. code-tab:: bash ppc64le

      module load nvhpc

      module load nvhpc/23.1
      module load nvhpc/22.1
      module load nvhpc/21.5

   .. code-tab:: bash aarch64

      module load nvhpc

      module load nvhpc/24.1

For further information please see the `NVIDIA HPC SDK Documentation Archive <https://docs.nvidia.com/hpc-sdk/archive/>`__.
