LLVM
----

LLVM has been provided for use on the system by the ``llvm`` module.
It has been built with CUDA GPU offloading support, allowing OpenMP
regions to run on a GPU using the ``target`` directive.

Note that, as from LLVM 11.0.0, it provides a Fortran compiler called
``flang``. Although this has been compiled and can be used for
experimentation, it is still immature and ultimately relies on
``gfortran`` for its code generation. The ``lvm/11.0.0`` module therefore
defaults to using the operating system provided ``gfortran``, instead.

.. code-block:: bash

   module load llvm
   module load llvm/11.0.0

For further information please see the `LLVM Releases <https://releases.llvm.org/>`__ for versioned documentation.