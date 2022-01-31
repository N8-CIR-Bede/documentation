.. _software-compilers:

Compilers
=========

Most compiler modules set the ``CC``, ``CXX``, ``FC``, ``F90`` environment variables to appropriate values. These are commonly used by tools such as CMake and autoconf, so that by loading a compiler module its compilers are used by default.

This can also be done in your own build scripts and make files. e.g.

.. code-block:: bash

  module load gcc
  $CC -o myprog myprog.c

.. toctree::
    :maxdepth: 1
    :glob:

    gcc
    llvm
    ibmxl
    nvhpc
    nvcc
