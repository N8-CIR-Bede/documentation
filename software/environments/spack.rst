.. _software-spack:

Spack
=====

`Spack <https://spack.readthedocs.io/>`__ is a packaged management tool which can be used to extend the installed software on the system,
without requiring specialist knowledge on how to build particular pieces
of software. 

Installing Spack
----------------

Installing Spack simply requires cloning the Spack git repository, and sourcing ``setup-env.sh`` in your interactive session or adding it to your bash environment. 

Spack installations can become quite large consuming considerable disk space. 
To avoid filling the limited ``/users/`` or ``/project`` directories it is recommended to install Spack into the ``/nobackup`` filestore.

To install Spack in ``/nobackup/projects/<project>/$USER`` (replacing ``<project>`` with your project code): 

.. code-block:: bash

   # Clone the Spack repository
   $ git clone https://github.com/spack/spack.git /nobackup/projects/<project>/$USER/spack

   # Add the installation location of Spack to your bash environment
   $ echo 'export SPACK_ROOT=/nobackup/projects/<project>/$USER/spack' >> ~/.bash_profile
   $ echo 'source $SPACK_ROOT/share/spack/setup-env.sh' >> ~/.bash_profile

After re-logging in to Bede, or re-sourcing your bash environment ``spack`` should be available for use.


Using Spack
-----------

Spack provides a command line interface with a number of subcommands which can be used to query, manage and enable the use of software packages in the Spack ecosystem.

Querying available packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Packages available for installation via spack can be found using ``spack list`` or on the  `Spack Package List webpage <https://spack.readthedocs.io/en/latest/package_list.html>`__.

.. code-block:: bash

   # List all packages
   spack list
   # List packages which match the pattern "gromacs"
   spack list gromacs

More information can be found for a given package using ``spack info``

E.g. to get information about the package ``gromacs``

.. code-block:: bash

   spack info gromacs


Installing packages
~~~~~~~~~~~~~~~~~~~

Once you know which package(s) you wish to install, the ``install`` subcommand can be used to install named packages.

Specific package versions can be installed using the ``<package>@<version>`` format. 

I.e. to install the default ``gromacs``, or a specific version such as ``2021.5``:

.. code-block:: bash

   # Install the default available gromacs
   spack install gromacs
   # Install gromacs 2021.5
   spack install gromacs@2021.5


Spack packages may also provided `variants <https://spack.readthedocs.io/en/latest/basic_usage.html#variants>`__. 
Spack package variants are package specific, and are often used to configure the build type or optional features such as MPI or CUDA support.

For example, to install the default version of ``gromacs`` with CUDA and MPI variants enabled:

.. code-block:: bash

   spack install gromacs +mpi +cuda

Listing installed packages
~~~~~~~~~~~~~~~~~~~~~~~~~~

To list packages which have been installed into the current spack environment, use the ``find`` subcommands

.. code-block:: bash

   spack find

Using installed packages
~~~~~~~~~~~~~~~~~~~~~~~~

Spack packages will not be available on your ``$PATH`` by default once installed, but provides mechanisms to load packages and make them available for use.

Spack provides the ``load`` subcommand to load an installed package and make it available for use. 

I.e. to load the default installed version of gromacs:

.. code-block:: bash

   spack load gromacs
   # The package is then available for use
   gmx -version

Packages can then be unloaded via the ``unload`` subcommand, and found using the ``--loaded`` option of the ``find`` subcommand:

.. code-block:: bash
   
   # List loaded modules
   spack find --loaded
   # Unload a loaded module
   spack unload gromacs

Alternatively, Spack includes `Environment module integration <https://spack.readthedocs.io/en/latest/module_file_support.html>`__, allowing spack installed software to be available via ``module load`` and the associated commands. 
Please refer to the `Spack Modules Documentation <https://spack.readthedocs.io/en/latest/module_file_support.html>`__ for more information.

Uninstalling packages
~~~~~~~~~~~~~~~~~~~~~

Packages can be uninstalled using the ``uninstall`` subcommand. 

E.g. to uninstall ``gromacs``

.. code-block:: bash

   spack uninstall gromacs


Project Spack installations
---------------------------

If a project wishes to create a spack installation, for example under
``/projects/<project>/spack`` and you would like an easy way for your
users to add it to their environment, please contact us and we can make
a module.

If you are a user who wishes to supplement your project's spack
installation, follow the installation instructions above and then tell
it where your project's copy of spack is:

.. code-block:: bash

   cat > $SPACK_ROOT/etc/spack/upstreams.yaml <<EOF
   upstreams:
     spack-central:
       install_tree: /projects/<project>/spack
       modules:
         tcl: /projects/<project>/spack/share/spack/modules
   EOF