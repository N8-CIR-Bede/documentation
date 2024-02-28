.. _software-compilers-gcc:

GCC
---

The `GNU Compiler Collection (GCC) <https://gcc.gnu.org/>`__ is available on Bede including C, C++ and Fortran compilers. 

The copies of GCC available as modules have been compiled with CUDA
offload support:

.. code-block:: bash

   module load gcc/12.2
   module load gcc/10.2.0
   module load gcc/8.4.0

The version of GCC which is distributed with RHEL is also packaged as the ``gcc/native`` module, providing GCC ``8.5.0``. This does not include CUDA offload support.

.. code-block:: bash

   module load gcc/native

For further information please see the `GCC online documentation <https://gcc.gnu.org/onlinedocs/>`__.
