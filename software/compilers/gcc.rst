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

The version of GCC which is distributed with RHEL is also packaged as the ``gcc/native`` module, providing GCC ``8.5.0``

.. code-block:: bash

   module load gcc/native

.. code-block:: bash

   # The GCC version provided by this module is RHEL specific.
   module load gcc/native

.. note::
   Note that the default GCC provided by Red Hat Enterprise Linux 7 (4.8.5)
   is quite old, will not optimise for the POWER9 processor (either use
   POWER8 tuning options or use a later compiler), and does not have
   CUDA/GPU offload support compiled in. The module ``gcc/native`` has been
   provided to point to this copy of GCC.

For further information please see the `GCC online documentation <https://gcc.gnu.org/onlinedocs/>`__.
