.. _software-tools-singularity:

Singularity CE
--------------

`Singularity CE <https://sylabs.io/singularity/>`__ (and `Apptainer <https://apptainer.org/>`__) is a container platform similar to `Docker <https://www.docker.com/>`__. 
Singularity is one of the most widely used container system for HPC.
It allows you to create and run containers that package up pieces of software in a way that is portable and reproducible.

Container platforms allow users to create and use container images, which are self-contained software stacks.

.. admonition:: ppc64le partitions only
   :class: warning

   Singularity CE is only available on ``ppc64le`` nodes/partitions within Bede. 

   For ``aarch64`` partitions, please see :ref:`Apptainer<software-tools-apptainer>`.



Singularity CE is provided by default on ``ppc64le`` nodes/partitions, and can be used without loading any modules.

.. code-block::bash

   singularity --version

.. note::
   Container images are not portable across CPU architectures. Containers created on ``x86_64`` machines may not be compatible with the ``ppc64le`` and ``aarch64`` nodes in Bede.

For more information on how to use singularity, please see the `Singularity Documentation <https://sylabs.io/docs/>`__.
