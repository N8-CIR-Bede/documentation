.. _software-tools-singularity:

Singularity
-----------

`Singularity <https://sylabs.io/singularity/>`__ (and `Apptainer <https://apptainer.org/>`__) is a container platform similar to `Docker <https://www.docker.com/>`__. 
Singularity is the most widely used container system for HPC.
It allows you to create and run containers that package up pieces of software in a way that is portable and reproducible.

Container platforms allow users to create and use container images, which are self-contained software stacks.

.. note::
   As Bede is a Power 9 Architecture (``ppc64le``) machine, containers created on more common ``x86_64`` machines may not be compatible. 


Singularity-ce is provided by default, and can be used without loading any modules.

.. code-block::bash

   singularity --version

For more information on how to use singularity, please see the `Singularity Documentation <https://apptainer.org/docs-legacy/>`__.
