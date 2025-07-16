.. _software-applications-conda:

Conda
-----

`Conda <https://docs.conda.io/>`__ is an open source package management system and environment management system that runs on Windows, macOS and Linux. Conda quickly installs, runs and updates packages and their dependencies.


.. _software-applications-conda-installing:

Installing Miniconda
~~~~~~~~~~~~~~~~~~~~

The simplest way to install Conda for use on Bede is through the `miniconda <https://docs.conda.io/en/latest/miniconda.html>`__ installer.

.. note::

    You may wish to install conda into the ``/nobackup/projects/<project>/$USER/<architecture>`` (where ``<project>`` is the project code for your project, and ``<architecture>`` is CPU architecture) directory rather than your ``home`` directory as it may consume considerable disk space

.. tabs::

   .. code-tab:: bash ppc64le

      export CONDADIR=/nobackup/projects/<project>/$USER/ppc64le # Update this with your <project> code.
      mkdir -p $CONDADIR
      pushd $CONDADIR

      # Download the latest miniconda installer for ppc64le
      wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-ppc64le.sh
      # Validate the file checksum matches is listed on https://docs.conda.io/en/latest/miniconda_hashes.html.
      sha256sum Miniconda3-latest-Linux-ppc64le.sh

      sh Miniconda3-latest-Linux-ppc64le.sh -b -p ./miniconda
      source miniconda/etc/profile.d/conda.sh
      conda update conda -y
   
   .. code-tab:: bash aarch64

      export CONDADIR=/nobackup/projects/<project>/$USER/aarch64 # Update this with your <project> code.
      mkdir -p $CONDADIR
      pushd $CONDADIR

      # Download the latest miniconda installer for aarch64
      wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
      # Validate the file checksum matches is listed on https://docs.conda.io/en/latest/miniconda_hashes.html.
      sha256sum Miniconda3-latest-Linux-aarch64.sh

      sh Miniconda3-latest-Linux-aarch64.sh -b -p ./miniconda
      source miniconda/etc/profile.d/conda.sh
      conda update conda -y

Using Miniconda
~~~~~~~~~~~~~~~

On subsequent sessions, or in job scripts you may need to re-source miniconda. Alternatively you could add this to your bash environment. I.e. 

.. tabs::

   .. code-tab:: bash ppc64le

      arch=$(uname -i) # Get the CPU architecture
      if [[ $arch == "ppc64le" ]]; then
         # Set variables and source scripts for ppc64le
         export CONDADIR=/nobackup/projects/<project>/$USER/ppc64le # Update this with your <project> code.
         source $CONDADIR/miniconda/etc/profile.d/conda.sh
      fi

   .. code-tab:: bash aarch64
      
      arch=$(uname -i) # Get the CPU architecture
      if [[ $arch == "aarch64" ]]; then
         # Set variables and source scripts for aarch64
         export CONDADIR=/nobackup/projects/<project>/$USER/aarch64 # Update this with your <project> code.
         source $CONDADIR/miniconda/etc/profile.d/conda.sh
      fi

Creating a new Conda Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With miniconda installed and activated, new `conda environments <https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html>`__ can be created using ``conda create``.

I.e. to create a new conda environment named `example`, with `python 3.9` you can run the following.

.. code-block:: bash
   
   conda create -y --name example python=3.9

Once created, the environment can be activated using ``conda activate``.

.. code-block:: bash

   conda activate example

Alternatively, Conda environments can be created outside of the conda/miniconda install, using the ``-p`` / ``--prefix`` option of ``conda create``. 

I.e. if you have installed miniconda to your home directory, but wish to create a conda environment within the ``/project/<PROJECT>/$USER/<architecture>/`` directory named ``example`` you can use:

.. code-block:: bash

   conda create -y --prefix /project/<PROJECT>/$USER/<architecture>/example python=3.9

This can subsequently be loaded via:

.. code-block:: bash

   conda activate /project/<PROJECT>/$USER/<architecture>/example

Listing and Activating existing Conda Environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Existing conda environments can be listed via:

.. code-block:: bash

   conda env list

``conda activate`` can then be used to activate one of the listed environments.

Adding Conda Channels to an Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default conda channel does not contain all packages or may not contain versions of packages you may wish to use.

In this case, third-party conda channels can be added to conda environments to provide access to these packages, such as the :ref:`Open-CE <software-applications-open-ce>` Conda channel hosted by Oregon State University.

It is recommended to add channels to specific conda environments, rather than your global conda configuration.

I.e. to add the `OSU Open-CE Conda channel <https://osuosl.org/services/powerdev/opence/>`__ to the currently loaded conda environment:

.. code-block:: bash

   conda config --env --prepend channels https://ftp.osuosl.org/pub/open-ce/current/

You may also wish to enable `strict channel priority <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-channels.html#strict-channel-priority>`__ to speed up conda operations and reduce incompatibility which will be default from Conda 5.0. This may break old environment files.

.. code-block:: bash

   conda config --env --set channel_priority strict

Installing Conda Packages
~~~~~~~~~~~~~~~~~~~~~~~~~

Conda packages can then be installed using ``conda install <package>``.

I.e. to install the conda package ``pylint`` into the active conda environment:

.. code-block:: bash
    
   conda install -y pylint

.. warning::

    Only Conda packages with support for ``ppc64le`` or ``aarch64`` will be installable (depending on the node architecture in use).

Deleting Conda Environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~

You may need to delete conda environments when they are no longer required, to free up disk space.
This can be achieved using ``conda env remove``.
I.e. to remove the ``example`` conda  environment created before:

.. code-block:: bash

   conda env remove -n example

Further Information
~~~~~~~~~~~~~~~~~~~

See the `Conda Documentation <https://docs.conda.io/>`__ for further information.

Alternatively, conda provides its own help information for the main ``conda`` executable and all subcommands, such as ``conda list``

.. code-block:: bash

   conda -h 
   conda list -h
