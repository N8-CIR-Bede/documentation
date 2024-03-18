.. _software-libraries-netcdf:

NetCDF
======

.. |arch_availabilty_name| replace:: NetCDF
.. include:: /common/ppc64le-only-sidebar.rst

`Network Common Data Form (NetCDF) <https://www.unidata.ucar.edu/software/netcdf/>`__ is a set of software libraries and machine independent data formats for array-orientated scientific data.

A centrally installed version of NetCDF is provided on Bede by the ``netcdf`` module(s).
It provides the C, C++ and Fortran bindings for this file format library.
When an :ref:`MPI <software-libraries-mpi>` module is loaded, parallel file support is enabled through the PnetCDF and :ref:`HDF5 <software-libraries-hdf5>` libraries.

.. code-block:: bash
   
   module load netcdf
   module load netcdf/4.7.4

.. note::
    
    NetCDF's parallel functionality can use HDF5, and so is subject
    to its known issues on Bede (see :ref:`software-libraries-hdf5-known-issues`).
