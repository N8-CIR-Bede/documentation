Software
========

Environments
------------

The default software environment on Bede is called “builder”. This uses
the modules system normally used on HPC systems, but provides a system
of intelligent modules. To see a list of what is available, executing
the command ``module avail``.

In this scheme, modules providing access to compilers and libraries
examine other modules that are also loaded and make the most appropriate
copy (or “flavour”) of the software available. This minimises the
problem of knowing what modules to choose whilst providing access to all
the combinations of how a library can be built.

For example, the following command gives you access to a copy of FFTW
3.3.8 that has been built against GCC 8.4.0:

::

   $ module load gcc/8.4.0 fftw/3.3.8
   $ which fftw-wisdom
   /opt/software/builder/developers/libraries/fftw/3.3.8/1/gcc-8.4.0/bin/fftw-wisdom

If you then load an MPI library, your environment will be automatically
updated to point at a copy of FFTW 3.3.8 that has been built against GCC
8.4.0 and OpenMPI 4.0.5:

::

   $ module load openmpi/4.0.5
   $ which fftw-wisdom
   /opt/software/builder/developers/libraries/fftw/3.3.8/1/gcc-8.4.0-openmpi-4.0.5/bin/fftw-wisdom

Similarly, if you then load CUDA, the MPI library will be replaced by
one built against it:

::

   $ which mpirun
   /opt/software/builder/developers/libraries/openmpi/4.0.5/1/gcc-8.4.0/bin/mpirun
   $ module load cuda/10.2.89
   $ which mpirun
   /opt/software/builder/developers/libraries/openmpi/4.0.5/1/gcc-8.4.0-cuda-10.2.89/bin/mpirun

Modules follow certain conventions:

-  Logs of software builds can be found under ``/opt/software/builder/logs/``.
-  Installation recipes for modules can be found under directory ``/home/builder/builder/``.
-  Although modules do their best to configure your environment so
   that you can use the software, it is sometimes useful to know where the
   software is installed on disk. This is provided by the ``<NAME>_HOME``
   environment variable, e.g. if the ``gcc/8.4.0`` module is loaded,
   environment variable ``GCC_HOME`` points to the directory containing
   its files.
-  Software provided by modules sometimes use other modules for their
   functionality. It is not normally required to explicitly load
   these prerequisites but it can be useful, for example to mirror R's
   buld environment when installing an R library. Where this occurs,
   a list of modules is provided by the ``<NAME>_BUILD_MODULES``
   environment variable, e.g. the ``r`` module sets environment variable
   ``R_BUILD_MODULES``.

Software can be built on top of these modules in the following ways:

-  Traditional - loading appropriate modules, manually unpacking,
   configuring, building and installing the new software
   (e.g. ``./configure; make; make install``)
-  `Spack <https://spack.readthedocs.io/>`__ - automated method of
   installing software. Spack will automatically find the multiple
   flavours (or variants, in spack-speak) of libraries provided by
   builder, minimising the number of packages needing to be built.

With Builder and Spack, the opportunity arises for a project to inherit
and supplement software, and for users to then inherit and supplement
that in turn. In this way, the centre can concentrate on providing core
software of general use and allow projects and users to concentrate on
specialist software elements that support their work.

In addition, there are two other types of software environment on Bede,
which are not currently recommended:

-  The vendor-supplied set of modules that originally came with the
   machine. To use these, execute:
   ``echo ocf > ~/.application_environment`` and then login again.
-  Easybuild - an automated method of installing software, rival to
   Spack. To use this, execute:
   ``echo builder > ~/.application_environment`` and then login again.

In both cases, executing ``rm ~/.application_environment`` and login
again will return you to the default software environment.

Spack
~~~~~

Spack can be used to extend the installed software on the system,
without requiring specialist knowledge on how to build particular pieces
of software. Documentation for the project is here:
https://spack.readthedocs.io/

To install spack, execute the following and then login again:

::

   $ git clone https://github.com/spack/spack.git $HOME/spack

   $ echo 'export SPACK_ROOT=$HOME/spack' >> ~/.bash_profile
   $ echo 'source $SPACK_ROOT/share/spack/setup-env.sh' >> ~/.bash_profile

Example usage, installing an MPI aware, GPU version of gromacs and than
loading it into your environment to use (once built, execute
``spack load gromacs`` before using):

::

   $ spack install gromacs +mpi +cuda

Other useful spack commands: \* ``spack find`` - show what packages have
been installed \* ``spack list`` - show what packages spack knows how to
build \* ``spack compilers`` - show what compilers spack can use \*
``spack info <package>`` - details about a package, and the different
ways it can be built \* ``spack spec <package>`` - what pieces of
software a package depends on

If a project wishes to create a spack installation, for example under
``/projects/<project>/spack`` and you would like an easy way for your
users to add it to their environment, please contact us and we can make
a module.

If you are a user who wishes to supplement your project’s spack
installation, follow the installation instructions above and then tell
it where your project’s copy of spack is:

::

   cat > $SPACK_ROOT/etc/spack/upstreams.yaml <<EOF
   upstreams:
     spack-central:
       install_tree: /projects/<project>/spack
       modules:
         tcl: /projects/<project>/spack/share/spack/modules
   EOF

Easybuild
~~~~~~~~~

Not currently recommended.

The central Easybuild modules are available when a user executes the
following command and then logs in again:

::

   echo easybuild > ~/.application_environment

A user can create their own Easybuild installation to supplement (or
override) the packages provided by the central install by:

::

   echo 'export EASYBUILD_INSTALLPATH=$HOME/eb' >> ~/.bash_profile
   echo 'export EASYBUILD_BUILDPATH=/tmp' >> ~/.bash_profile
   echo 'export EASYBUILD_MODULES_TOOL=Lmod' >> ~/.bash_profile
   echo 'export EASYBUILD_PARALLEL=8' >> ~/.bash_profile
   echo 'export MODULEPATH=$HOME/eb/modules/all:$MODULEPATH' >> ~/.bash_profile

Login again, and then:

::

   wget https://raw.githubusercontent.com/easybuilders/easybuild-framework/develop/easybuild/scripts/bootstrap_eb.py
   python bootstrap_eb.py $EASYBUILD_INSTALLPATH

Verify install by checking sensible output from:

::

   module avail   # should show an EasyBuild module under user's home directory
   module load EasyBuild
   which eb       # should show a path under the user's home directory

Software can now be installed into the new Easybuild area using
``eb <package>``

Project Easybuild installations can be created using a similar method.
In this case, a central module to add the project’s modules to a user’s
environment is helpful, and can be done on request.


Compilers
---------

All compiler modules set the ``CC``, ``CXX``, ``FC``, ``F90`` environment variables to appropriate values. These are commonly used by tools such as cmake and autoconf, so that by loading a compiler module its compilers are used by default.

This can also be done in your own build scripts and make files. e.g.

::

  module load gcc
  $CC -o myprog myprog.c

GCC
~~~

Note that the default GCC provided by Red Hat Enterprise Linux 7 (4.8.5)
is quite old, will not optimise for the POWER9 processor (either use
POWER8 tuning options or use a later compiler), and does not have
CUDA/GPU offload support compiled in. The module ``gcc/native`` has been
provided to point to this copy of GCC.

The copies of GCC available as modules have been compiled with CUDA
offload support:

::

   module load gcc/10.2.0

LLVM
~~~~

LLVM has been provided for use on the system by the ``llvm`` module.
It has been built with CUDA GPU offloading support, allowing OpenMP
regions to run on a GPU using the ``target`` directive.

Note that, as from LLVM 11.0.0, it provides a Fortran compiler called
``flang``. Although this has been compiled and can be used for
experimentation, it is still immature and ultimately relies on
``gfortran`` for its code generation. The ``lvm/11.0.0`` module therefore
defaults to using the operating system provided ``gfortran``, instead.


BLAS/LAPACK
-----------

The following numerical libraries provide optimised CPU implementations of BLAS and LAPACK on the system:

- ESSL (IBM Engineering and Scientific Subroutine Library)
- OpenBLAS

The modules for each of these libraries provide some convenience environment variables: ``N8CIR_LINALG_CFLAGS`` contains the compiler arguments to link BLAS and LAPACK to C code; ``N8CIR_LINALG_FFLAGS`` contains the same to link to Fortran. When used with variables such as ``CC``, commands to build software can become entirely independent of what compilers and numerical libraries you have loaded, e.g.

::

   module load gcc essl/6.2
   $CC -o myprog myprog.c $N8CIR_LINALG_CFLAGS


MPI
---

The main supported MPI on the system is OpenMPI.

For access to a cuda-enabled MPI: ``module load gcc cuda openmpi``

We commit to the following convention for all MPIs we provide as modules:

- The wrapper to compile C programs is called ``mpicc``
- The wrapper to compile C++ programs is called ``mpicxx``
- The wrapper to compile Fortran programs is called ``mpif90``


HDF5
----

When loaded in conjunction with an MPI module such as ``openmpi``, the
``hdf5`` module provides both the serial and parallel versions of the
library. The parallel functionality relies on a technology called MPI-IO,
which is currently subject to the following known issue on Bede:

- HDF5 does not pass all of its parallel tests with OpenMPI 4.x. If
  you are using this MPI and your application continues to run but does
  not return from a call to the HDF5 library, you may have hit a similar
  issue. The current workaround is to instruct OpenMPI to use an alternative
  MPI-IO implementation with the command: ``export OMPI_MCA_io=ompio``
  The trade off is that, in some areas, this alternative is extremely slow
  and so should be used with caution.


NetCDF
------

The ``netcdf`` module provides the C, C++ and Fortran bindings for this
file format library. When an MPI module is loaded, parallel support is
enabled through the PnetCDF and HDF5 libraries.

Use of NetCDF's parallel functionality can use HDF5, and so is subject
to its known issues on Bede (see above).

Python
------

PyTorch Quickstart
~~~~~~~~~~~~~~~~~~
The following should get you set up with a working conda environment (replacing <project> with your project code):

::

    export DIR=/nobackup/projects/<project>/$USER
    # rm -rf ~/.conda .condarc $DIR/miniconda # Uncomment if you want to remove old env
    mkdir $DIR
    pushd $DIR

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-ppc64le.sh

    sh Miniconda3-latest-Linux-ppc64le.sh -b -p $DIR/miniconda
    source miniconda/bin/activate
    conda update conda -y
    conda config --set channel_priority strict

    conda config --prepend channels \
            https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/

    conda config --prepend channels \
            https://opence.mit.edu

    conda create --name opence pytorch=1.7.1 -y
    conda activate opence


This has some limitations such as not supporting large model support. If you require this you can try the instructions below, these provide an older version of PyTorch however.


PyTorch and TensorFlow: IBM PowerAI and wmlce [Possibly Out of Date]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

IBM have done a lot of work to port common Machine Learning tools to the
POWER9 system, and to take advantage of the GPUs abililty to directly
access main system memory on the POWER9 architecture using its “Large
Model Support”.

This has been packaged up into what is variously known as IBM Watson
Machine Learning Community Edition (wmlce) or the catchier name PowerAI.

Documentation on wmlce can be found here:
https://www.ibm.com/support/pages/get-started-ibm-wml-ce

Installation is via the IBM channel of the anaconda package management tool. **Note:
if you do not use this channel you will not find all of the available packages.**
First install anaconda (can be quite large - so using the /nobackup area):

::

   cd /nobackup/projects/<project>

   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-ppc64le.sh
   sh Miniconda3-latest-Linux-ppc64le.sh
   conda update conda
   conda config --set channel_priority strict
   conda config --prepend channels https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/
   conda create --name wmlce

Then login again and install wmlce (GPU version by default - substitute
``powerai-cpu`` for ``powerai`` for the CPU version):

::

   conda activate wmlce
   conda install powerai ipython

Running ``ipython`` on the login node will then allow you to experiment
with this feature using an interactive copy of Python and the GPUs on
the login node. Demanding work should be packaged into a job and
launched with the ``python`` command.

If a single node with 4 GPUs and 512GB RAM isn't enough, the Distributed
Deep Learning feature of PowerAI should allow you to write code that can
take advantage of multiple nodes.



WMLCE resnet50 benchmark [Possibly out of date]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. toctree::
   :maxdepth: -1

   resnet50/bede-README-sbatch

Initial investigations with CUDA (under development)
----------------------------------------------------

.. toctree::
   :maxdepth: -1

   wanderings/wanderings-in-CUDALand
   wanderings/Estimating-pi-in-CUDALand

Cryo-EM Software Environment
----------------------------

Documentation on the the Cryo-EM Software Environment for Life Sciences is available :download:`here <Cryo-EM_Bede.pdf>`. Note that this document is mainly based on the installation on `Satori <https://mit-satori.github.io>`_ and might have some inconsistencies with the Bede installation.





To use the modules, execute

::

   conda activate /projects/bddir04/ibm-lfsapp/CryoEM
   
with a working conda installation.   
