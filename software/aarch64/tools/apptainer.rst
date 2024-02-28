.. _software-aarch64-tools-apptainer:

Singularity
-----------

`Apptainer <https://apptainer.org/>`__ (formerly `Singularity <https://sylabs.io/singularity/>`__) is a container platform similar to `Docker <https://www.docker.com/>`__. 
Singularity/Apptainer is the most widely used container system for HPC.
It allows you to create and run containers that package up pieces of software in a way that is portable and reproducible.

Container platforms allow users to create and use container images, which are self-contained software stacks.

.. note::
   As Bede\'s grace-hopper nodes are Arm64 Architecture (``aarch64``) machine, containers created on more common ``x86_64`` machines will not be compatible. 


Apptainer is provided by default, and can be used without loading any modules, and includes the ability to build containers via ``--fakeroot``

.. code-block:: bash

   apptainer --version

A symlink is provided so that the command `singularity` is still usable.

.. code-block:: bash

   singularity --version

For more information on how to use apptainer, please see the `Apptainer Documentation <https://apptainer.org/docs/>`__.
