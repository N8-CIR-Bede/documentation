.. _software-libraries-MPI:

MPI
===

`OpenMPI <https://www.open-mpi.org/>`__ and `MVAPICH <https://mvapich.cse.ohio-state.edu/>`__ are provided as alternate Message Passing Interface (MPI) implementations on Bede.

OpenMPI is the main supported MPI on bede.

We commit to the following convention for all MPIs we provide as modules:

- The wrapper to compile C programs is called ``mpicc``
- The wrapper to compile C++ programs is called ``mpicxx``
- The wrapper to compile Fortran programs is called ``mpif90``

CUDA-enabled MPI is available through OpenMPI, when a cuda module is loaded alongside ``openmpi``, I.e.

.. tabs::

   .. code-tab:: bash ppc64le

      module load gcc cuda openmpi

   .. code-tab:: bash aarch64

      module load gcc cuda openmpi

OpenMPI is provided by the ``openmpi`` module(s):


.. tabs::

   .. code-tab:: bash ppc64le

      module load openmpi
      
      module load openmpi/4.0.5

   .. code-tab:: bash ppc64le

      module load openmpi
      
      module load openmpi/4.1.6

.. |arch_availabilty_name| replace:: MVAPICH2
.. include:: /common/ppc64le-only.rst

MVAPICH2 is provided by the `mvapich2` module(s):

.. tabs::

   .. code-tab:: bash ppc64le

      module load mvapich2

      module load mvapich2/2.3.5-2

   .. tab:: aarch64

      .. include:: /common/ppc64le-only.rst

.. note::

   There are a number of issues with OpenMPI 4 and the one-sided MPI communication features added by the MPI-3 standard. These features are typically useful when combined with GPUs, due to the asynchronous nature of the CUDA and OpenCL programming models.

   For codes that require these features, we currently recommend using the ``mvapich2`` module on ``ppc64le`` nodes/partitions.

We also offer the ``mvapich2-gdr/2.3.6`` module on ``ppc64le`` nodes/partitions. This is a version of MVAPICH2 that is specifically designed for machines like Bede, providing optimised communications directly between GPUs - even when housed in different compute nodes.

Unlike the ``openmpi`` and ``mvapich2`` modules, ``mvapich2-gdr`` does not adapt itself to the currently loaded compiler and CUDA modules. This version of the software was built using GCC 8.4.1 and CUDA 11.3.

.. tabs::

   .. code-tab:: bash ppc64le

      module load mvapich2-gdr/2.3.6 gcc/8.4.0 cuda/11.3.1

   .. tab:: aarch64

      .. include:: /common/ppc64le-only.rst

Further information can be found on the `MVAPICH2-GDR <http://mvapich.cse.ohio-state.edu/userguide/gdr/>`__ pages.
