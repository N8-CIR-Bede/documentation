.. _software-applications-pytorch:

PyTorch
-------

`PyTorch <https://pytorch.org/>`__ is an end-to-end machine learning framework.
PyTorch enables fast, flexible experimentation and efficient production through a user-friendly front-end, distributed training, and ecosystem of tools and libraries.

The main method of distribution for PyTorch for ``ppc64le`` is via :ref:`Conda <software-applications-conda>`, with :ref:`Open-CE<software-applications-open-ce>` providing a simple method for installing multiple machine learning frameworks into a single conda environment.

The upstream Conda and pip distributions do not provide ppc64le pytorch packages at this time. 

.. _software-applications-pytorch-conda:

Installing via Conda
~~~~~~~~~~~~~~~~~~~~

.. tabs::

   .. group-tab:: ppc64le

      With a working Conda installation (see :ref:`Installing Miniconda<software-applications-conda-installing>`) the following instructions can be used to create a Python 3.9 conda environment named ``torch`` with the latest Open-CE provided PyTorch:

      .. note:: 

         Pytorch installations via conda can be relatively large. Consider installing your miniconda (and therfore your conda environments) to the ``/nobackup`` file store.


      .. code-block:: bash

         # Create a new conda environment named torch within your conda installation
         conda create -y --name torch python=3.9

         # Activate the conda environment
         conda activate torch

         # Add the OSU Open-CE conda channel to the current environment config
         conda config --env --prepend channels https://ftp.osuosl.org/pub/open-ce/current/

         # Also use strict channel priority
         conda config --env --set channel_priority strict

         # Install the latest available version of PyTorch
         conda install -y pytorch

      In subsequent interactive sessions, and when submitting batch jobs which use PyTorch, you will then need to re-activate the conda environment.

      For example, to verify that PyTorch is available and print the version:

      .. code-block:: bash

         # Activate the conda environment
         conda activate torch

         # Invoke python
         python3 -c "import torch;print(torch.__version__)"


      Installation via the upstream Conda channel is not currently possible, due to the lack of ``ppc64le`` or ``noarch`` distributions.


      .. note::
         
         The :ref:`Open-CE<software-applications-open-ce>` distribution of PyTorch does not include IBM technologies such as DDL or LMS, which were previously available via :ref:`WMLCE<software-applications-wmlce>`. 
         WMLCE is no longer supported.


   .. group-tab:: aarch64

      .. warning::

         Conda builds of PyTorch for ``aarch64`` do not include CUDA support as of July 2024.
         
         For now, consider:
         
         * Install a ``2.4.0`` build using the CUDA 12.4 PyTorch channel via pip (see :ref:`software-applications-pytorch-pip`)
         * Use containers provided by Nvidia for a pip-based environment (see :ref:`software-applications-pytorch-ngc`)
         * `Build PyTorch from source <https://pytorch.org/get-started/locally/#linux-from-source>`__ into a conda environment.

.. _software-applications-pytorch-pip:

Installing via pip
~~~~~~~~~~~~~~~~~~

.. tabs::

   .. group-tab:: ppc64le

      .. warning::

         ``pip`` does not provide ``ppc64le`` builds of PyTorch (from PyPI or the PyTorch wheel repositories). Instead, see :ref:`software-applications-pytorch-conda` or `build from source <https://pytorch.org/get-started/locally/#linux-from-source>`__.

   .. group-tab:: aarch64

      PyTorch pip packages for ``aarch64`` prior to PyTorch ``2.4`` do not include CUDA support.

      CUDA support is only included in the PyTorch ``2.4.0`` wheels for ``aarch64`` using CUDA ``12.4``.
     
      .. warning::

         * CUDA 11.8 and CUDA 12.1 ``aarch64`` builds do not include CUDA support (as of PyTorch ``2.4.0``). You must use the CUDA 12.4 repository.
         * CUDA enabled ``aarch64`` wheels are large (over 2GB). Consider creating your ``venv`` / conda env in ``/nobackup`` to avoid filling your home directory quota.
         * As with other PyTorch ``2.x`` builds, you may see a warning if you do not also install ``numpy`` into your python environment.
      
      .. code-block:: bash

         # Create a python venv in /nobackup, replacing your project name and following path as appropriate
         python3 -m venv /nobackup/projects/bdXXXXX/pytorch-venv

         # Activate the venv, replacing the path as appropriate
         source /nobackup/projects/bdXXXXX/pytorch-venv/bin/activate

         # Install the latest release using the CUDA 12.4 repository
         python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu124

         # Ensure that CUDA support is enabled
         python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available());)"

      In subsequent interactive sessions, and when submitting batch jobs which use PyTorch, you will then need to re-source the python venv.

      For example, to verify that PyTorch is available and print the version:

      .. code-block:: bash

         # Activate the venv, replacing the path as appropriate
         source /nobackup/projects/bdXXXXX/pytorch-nightly-venv/bin/activate

         # Invoke python
         python3 -c "import torch; print(torch.__version__)"

.. _software-applications-pytorch-ngc:

Using NGC PyTorch Containers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tabs::

   .. group-tab:: ppc64le

      .. warning::

         NVIDIA do not provide ``ppc64le`` containers for pytorch through NGC. This method should only be used for ``aarch64`` partitions.

   .. group-tab:: aarch64

      NVIDIA provide docker containers with CUDA-enabled pytorch builds for ``x86_64`` and ``aarch64`` architectures through NGC.

      The `NGC PyTorch <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch>`__ containers have included Hopper support since ``22.09``.

      * ``22.09`` and ``22.10`` provide a conda-based install of pytorch.
      * ``22.11+`` provide a pip-based install in the default python environment.

      For details of which pytorch version is provided by the each container release, see the `NGC PyTorch container release notes <https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes>`__.

      :ref:`software-tools-apptainer` can be used to convert and run docker containers, or to build an apptainer container based on a docker container. 
      These can be built on the ``aarch64`` nodes in Bede using :ref:`software-tools-apptainer-rootless`.

      .. note::

         PyTorch containers can consume a large amount of disk space. Consider setting :ref:`software-tools-apptainer-cachedir` to an appropriate location in ``/nobackup``, e.g. ``export APPTAINER_CACHEDIR=/nobackup/projects/${SLURM_JOB_ACCOUNT}/${USER}/apptainer-cache``.

      .. note::

         The following apptainer commands should be executed from an ``aarch64`` node only, i.e. on ``ghlogin``, ``gh`` or ``ghtest``.

      Docker containers can be fetched and converted using ``apptainer pull``, prior to using ``apptainer exec`` to execute code within the container.

      .. code:: bash

         # Pull and convert the docker container. This may take a while.
         apptainer pull docker://nvcr.io/nvidia/pytorch:24.03-py3
         # Run a command in the container, i.e. showing the pytorch version
         apptainer exec --nv docker://nvcr.io/nvidia/pytorch:24.03-py3 python3 -c "import torch;print(torch.__version__);"

      Alternatively, if you require more than just pytorch within the container you can create an `apptainer definition file <https://apptainer.org/docs/user/main/definition_files.html>`__.
      E.g. for a container based on ``pytorch:24.03-py3`` which also installs HuggingFace Transformers ``4.37.0``, the following definition file could be used:

      .. code:: singularity

         Bootstrap: docker
         From: nvcr.io/nvidia/pytorch:24.03-py3

         %post
           # Install other python dependencies, e.g. hugging face transformers
           python3 -m pip install transformers[torch]==4.37.0

         %test
           # Print the torch version, if CUDA is enabled and which architectures
           python3 -c "import torch;print(torch.__version__); print(torch.cuda.is_available());print(torch.cuda.get_arch_list());"
           # Print the pytorch transformers version, demonstrating it is available.
           python3 -c "import transformers;print(transformers.__version__);"

      Assuming this is named ``pytorch-transformers.def``, a corresponding apptainer image file name ``pytorch-transformers.sif`` can then be created via:

      .. code-block:: bash

         apptainer build --nv pytorch-transformers.sif pytorch-transformers.def

      Commands within this container can then be executed using ``apptainer exec``.
      I.e. to see the version of transformers installed within the container:

      .. code-block:: bash

         apptainer exec --nv pytorch-transformers.sif python3 -c "import transformers;print(transformers.__version__);"

      Or in this case due to the ``%test`` segment of the container, run the test command.

      .. code-block:: bash

         apptainer test --nv pytorch-transformers.sif


Further Information
~~~~~~~~~~~~~~~~~~~

For more information on the usage of PyTorch, see the `Online Documentation <https://pytorch.org/docs/>`__.
