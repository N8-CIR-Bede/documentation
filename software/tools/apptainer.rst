.. _software-tools-apptainer:

Apptainer
---------

`Apptainer <https://apptainer.org/>`__ is a container platform similar to `Docker <https://www.docker.com/>`__, previously known as Singularity. 
It is a widely used container system for HPC, which allows you to create and run containers that package up pieces of software in a way that is portable and reproducible.

Container platforms allow users to create and use container images, which are self-contained software stacks.

.. admonition:: aarch64 partitions only
   :class: warning

   Apptainer is only available on ``aarch64`` nodes/partitions within Bede. 

   For ``ppc64le`` partitions, please see :ref:`Singularity CE<software-tools-singularity>`.

Apptainer is provided by default on ``aarch64`` nodes/partitions, and can be used without loading any modules.

.. code-block::bash

   apptainer --version

.. note::
   Container images are not portable across CPU architectures. Containers created on ``x86_64`` machines may not be compatible with the ``ppc64le`` and ``aarch64`` nodes in Bede.


For more information on how to use singularity, please see the `Apptainer Documentation <https://apptainer.org/docs/>`__.

Differences from Singularity CE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Although Apptainer and Singularity share a common history, there are a number of important differences, as `documented by apptainer <https://apptainer.org/docs/user/latest/singularity_compatibility.html>`_, including:
    
* ``SINGULARITY_`` prefixed environment variables may issues warnings, preferring to be prefixed with ``APPTAINER_``
* The ``singularity`` command/binary is still available, but is just a symlink to ``apptainer``
* The ``library://`` protocol is not supported by apptainer's default configuration. See `Restoring pre-Apptainer library behaviour <https://apptainer.org/docs/user/latest/endpoint.html#restoring-pre-apptainer-library-behavior>`_ for more information.


Rootless Container Builds
^^^^^^^^^^^^^^^^^^^^^^^^^

The apptainer installation on Bede's ``aarch64`` nodes supports the creation of container images from apptainer definition files or docker containers without the need for root.

I.e. it is possible to build your ``aarch64`` containers on the ``ghlogin`` interactive sessions rather than having to create containers on ``aarch64`` machines elsewhere and copying them into bede.

``APPTAINER_CACHEDIR``
^^^^^^^^^^^^^^^^^^^^^^

Container images for GPU acceleated code are often very large, and if creating containers based on a docker image multiple copies of th container will exist in your file stores.
Consider setting the ``APPTAINER_CACHEDIR`` environment variable to a location in ``/nobackup`` or ``/projects`` to avoid filling your home directory.