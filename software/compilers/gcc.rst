.. _software-compilers-gcc:

GCC
---

The `GNU Compiler Collection (GCC) <https://gcc.gnu.org/>`__ is available on Bede including C, C++ and Fortran compilers. 

The copies of GCC available as modules have been compiled with CUDA
offload support:

.. tabs::

   .. code-tab:: bash ppc64le

      module load gcc/12.2
      module load gcc/10.2.0
      module load gcc/8.4.0

   .. code-tab:: bash aarch64

      module load gcc/14.2
      module load gcc/13.2
      module load gcc/12.2

The version of GCC which is distributed with RHEL is also packaged as the ``gcc/native`` module, providing GCC ``8.5.0``. This does not include CUDA offload support.

.. tabs::

   .. code-tab:: bash ppc64le

      module load gcc/native

   .. code-tab:: bash aarch64

      module load gcc/native # provides 11.4.1 


For further information please see the `GCC online documentation <https://gcc.gnu.org/onlinedocs/>`__.

``aarch64`` psABI warnings
^^^^^^^^^^^^^^^^^^^^^^^^^^

When compiling on the ``aarch64`` Grace-Hopper nodes with ``--std=c++17``, GCC may emit platform specific ABI warnings about a change made in GCC 10.1.
These warnings should only be a concern if you are linking objects compiled with ``GCC >= 10.1`` in c++17 mode with objects compiled with ``GCC < 10.1`` in c++17 mode.

Use ``--Wno-psabi`` to suppress these warnings. 
