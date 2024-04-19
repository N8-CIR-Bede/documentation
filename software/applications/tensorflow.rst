.. _software-applications-tensorflow:

TensorFlow
----------

`TensorFlow <https://www.tensorflow.org/>`__ is an end-to-end open source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries and community resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML powered applications.

TensorFlow can be installed through a number of python package managers such as :ref:`Conda<software-applications-conda>` or ``pip``.

For use on Bede's ``ppc64le`` nodes, the simplest method is to install TensorFlow using the :ref:`Open-CE Conda distribution<software-applications-open-ce>`.

For the ``aarch64`` nodes, using a NVIDIA provided `NGC Tensorflow container <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow>`__ is likely preferred.


Installing via Conda (Open-CE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tabs::

   .. group-tab:: ppc64le

      With a working Conda installation (see :ref:`Installing Miniconda<software-applications-conda-installing>`) the following instructions can be used to create a Python 3.8 conda environment named ``tf-env`` with the latest Open-CE provided TensorFlow:

      .. note:: 

         TensorFlow installations via conda can be relatively large. Consider installing your miniconda (and therfore your conda environments) to the ``/nobackup`` file store.


      .. code-block:: bash

         # Create a new conda environment named tf-env within your conda installation
         conda create -y --name tf-env python=3.8

         # Activate the conda environment
         conda activate tf-env

         # Add the OSU Open-CE conda channel to the current environment config
         conda config --env --prepend channels https://ftp.osuosl.org/pub/open-ce/current/

         # Also use strict channel priority
         conda config --env --set channel_priority strict

         # Install the latest available version of Tensorflow
         conda install -y tensorflow

      In subsequent interactive sessions, and when submitting batch jobs which use TensorFlow, you will then need to re-activate the conda environment.

      For example, to verify that TensorFlow is available and print the version:

      .. code-block:: bash

         # Activate the conda environment
         conda activate tf-env

         # Invoke python
         python3 -c "import tensorflow;print(tensorflow.__version__)"

      .. note::
         
         The :ref:`Open-CE<software-applications-open-ce>` distribution of TensorFlow does not include IBM technologies such as DDL or LMS, which were previously available via :ref:`WMLCE<software-applications-wmlce>`. 
         WMLCE is no longer supported.

   .. group-tab:: aarch64

      .. warning::

         Conda and pip builds of TensorFlow for ``aarch64`` do not include CUDA support as of April 2024. For now, see :ref:`software-applications-tensorflow-ngc` or `build from source <https://tensorflow.org/install/source>`__.

.. _software-applications-tensorflow-ngc:

Using NGC TensorFlow Containers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. tabs::

   .. group-tab:: ppc64le

      .. warning::

         NVIDIA do not provide ``ppc64le`` containers for TensorFlow through NGC. This method should only be used for ``aarch64`` partitions.
   
   .. group-tab:: aarch64

      NVIDIA provide docker containers with CUDA-enabled TensorFlow builds for ``x86_64`` and ``aarch64`` architectures through NGC.

      The `NGC Tensorflow <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow>`__ containers have included Hopper support since ``22.09``.

      For details of which TensorFlow version is provided by the each container release, see the `NGC TensorFlow container release notes <https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes>`__.


      :ref:`software-tools-apptainer` can be used to convert and run docker containers, or to build an apptainer container based on a docker container. 
      These can be built on the ``aarch64`` nodes in Bede using :ref:`software-tools-apptainer-rootless`.

      .. note::

         TensorFlow containers can consume a large amount of disk space. Consider setting :ref:`software-tools-apptainer-cachedir` to an appropriate location in ``/nobackup``, e.g. ``export APPTAINER_CACHEDIR=/nobackup/projects/${SLURM_JOB_ACCOUNT}/${USER}/apptainer-cache``.

      .. note::

         The following apptainer commands should be executed from an ``aarch64`` node only, i.e. on ``ghlogin``, ``gh`` or ``ghtest``.

      Docker containers can be fetched and converted using ``apptainer pull``, prior to using ``apptainer exec`` to execute code within the container.

      .. code:: bash

         # Pull and convert the docker container. This may take a while.
         apptainer pull docker://nvcr.io/nvidia/tensorflow:24.03-tf2-py3
         # Run a command in the container, i.e. showing the TensorFlow version
         apptainer exec --nv docker://nvcr.io/nvidia/tensorflow:24.03-tf2-py3 python3 -c "import tensorflow; print(tensorflow.__version__);"

      Alternatively, if you require more than just TensorFlow within the container you can create an `apptainer definition file <https://apptainer.org/docs/user/main/definition_files.html>`__.
      E.g. for a container based on ``tensorflow:24.03-tf2-py3`` which also installs HuggingFace Transformers ``4.37.0``, the following definition file could be used:

      .. code:: singularity

         Bootstrap: docker
         From: nvcr.io/nvidia/tensorflow:24.03-tf2-py3

         %post
           # Install other python dependencies, e.g. hugging face transformers
           python3 -m pip install transformers==4.37.0

         %test
           # Print the torch version, if CUDA is enabled and which architectures
           python3 -c "import tensorflow; print(tensorflow.__version__); print(tensorflow.config.list_physical_devices('GPU'));"
           # Print the TensorFlow transformers version, demonstrating it is available.
           python3 -c "import transformers;print(transformers.__version__);"

      Assuming this is named ``tf-transformers.def``, a corresponding apptainer image file name ``tf-transformers.sif`` can then be created via:

      .. code-block:: bash

         apptainer build --nv tf-transformers.sif tf-transformers.def

      Commands within this container can then be executed using ``apptainer exec``.
      I.e. to see the version of transformers installed within the container:

      .. code-block:: bash

         apptainer exec --nv tf-transformers.sif python3 -c "import transformers;print(transformers.__version__);"

      Or in this case due to the ``%test`` segment of the container, run the test command.

      .. code-block:: bash

         apptainer test --nv tf-transformers.sif


Further Information
~~~~~~~~~~~~~~~~~~~

For further information on TensorFlow features and usage, please refer to the `TensorFlow Documentation <https://www.tensorflow.org/api_docs/>`__. 