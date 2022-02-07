.. _software-applications-conda:

Conda
-----

`Conda <https://docs.conda.io/>`__ is an open source package management system and environment management system that runs on Windows, macOS and Linux. Conda quickly installs, runs and updates packages and their dependencies.

Installing Miniconda
~~~~~~~~~~~~~~~~~~~~

The simplest way to install Conda for use on Bede is through the `miniconda <https://docs.conda.io/en/latest/miniconda.html>`__ installer.

.. note::

    You may wish to install conda into the ``/nobackup/projects/<project>/$USER`` (where ``project`` is the project code for your project) directory rather than your ``home`` directory as it may consume considerable disk space

.. code-block:: bash

   export CONDADIR=/nobackup/projects/<project>/$USER # Update this with your <project> code.
   mkdir -p $CONDADIR
   pushd $CONDADIR

   # Download the latest miniconda installer for ppcle64
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-ppc64le.sh
   # Validate the file checksum matches is listed on https://docs.conda.io/en/latest/miniconda_hashes.html.
   sha256sum Miniconda3-latest-Linux-ppc64le.sh

   sh Miniconda3-latest-Linux-ppc64le.sh -b -p ./miniconda
   source miniconda/bin/activate
   conda update conda -y

On subsequent sessions, or in job scripts you may need to re-source miniconda. Alternatively you could add this to your bash environment. I.e. 

.. code-block:: bash

    export CONDADIR=/nobackup/projects/<project>/$USER # Update this with your <project> code.
    source $CONDADIR/miniconda/bin/activate

Creating a new Conda Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With miniconda installed and activated, new `conda environments <https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html>`__ can be created using ``conda create``.

I.e. to create a new conda environment named `example`, with `python 3.9` you can run the following.

.. code-block:: bash
   
   conda create -y --name example python==3.9

Once created, the environment can be activated using ``conda activate``.

.. code-block:: bash

   conda activate example

Listing and Activating existing Conda Environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Existing conda environments can be listed via:

.. code-block:: bash

   conda env list

``conda activate`` can then be used to activate one of the listed environments.

Installing Conda Packages
~~~~~~~~~~~~~~~~~~~~~~~~~

Conda packages can then be installed using ``conda install <package>``.

I.e. to install the conda package ``pylint`` into the active conda environment:

.. code-block:: bash
    
   conda install -y pylint

.. note::

    Only Conda packages with support for ``ppc64le`` will be installable.

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
