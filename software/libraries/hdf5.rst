.. _software-libraries-hdf5:

HDF5
====

When loaded in conjunction with an MPI module such as ``openmpi``, the
``hdf5`` module provides both the serial and parallel versions of the
library. 

.. tabs::
      
   .. code-tab:: bash ppc64le

      module load hdf5

      module load hdf5/1.10.7

   .. code-tab:: bash aarch64

      module load hdf5
      
      module load hdf5/1.10.11

.. _software-libraries-hdf5-known-issues:

Known issues
------------

The parallel functionality relies on a technology called MPI-IO,
which is currently subject to the following known issue on Bede:

- HDF5 does not pass all of its parallel tests with OpenMPI 4.x. If
  you are using this MPI and your application continues to run but does
  not return from a call to the HDF5 library, you may have hit a similar
  issue. The current workaround is to instruct OpenMPI to use an alternative
  MPI-IO implementation with the command: ``export OMPI_MCA_io=ompio``
  The trade off is that, in some areas, this alternative is extremely slow
  and so should be used with caution.
