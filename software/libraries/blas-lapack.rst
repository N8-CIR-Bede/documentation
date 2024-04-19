.. _software-libraries-blas-lapack:

BLAS/LAPACK
===========


The following numerical libraries provide optimised CPU implementations of BLAS and LAPACK on the system:

- `ESSL <https://www.ibm.com/docs/en/essl>`__ (IBM Engineering and Scientific Subroutine Library)
- `OpenBLAS <https://github.com/xianyi/OpenBLAS>`__

The modules for each of these libraries provide some convenience environment variables: ``N8CIR_LINALG_CFLAGS`` contains the compiler arguments to link BLAS and LAPACK to C code; ``N8CIR_LINALG_FFLAGS`` contains the same to link to Fortran. When used with variables such as ``CC``, commands to build software can become entirely independent of what compilers and numerical libraries you have loaded, eg. for ESSL:

.. tabs::

   .. code-tab:: bash ppc64le

      module load gcc essl/6.2
      $CC -o myprog myprog.c $N8CIR_LINALG_CFLAGS

   .. group-tab:: aarch64

      .. |arch_availabilty_name| replace:: ESSL
      .. include:: /common/ppc64le-only.rst


Or for OpenBLAS:

.. tabs::

   .. code-tab:: bash ppc64le

      module load gcc openblas/0.3.10
      $CC -o myprog myprog.c $N8CIR_LINALG_CFLAGS

   .. code-tab:: bash aarch64

      module load gcc openblas/0.3.26
      $CC -o myprog myprog.c $N8CIR_LINALG_CFLAGS

      # or 
      module load gcc openblas/0.3.26omp
      $CC -o myprog myprog.c $N8CIR_LINALG_CFLAGS




