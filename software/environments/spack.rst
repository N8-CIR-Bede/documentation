.. _software-spack:

Spack
=====

`Spack <https://spack.readthedocs.io/>`__ can be used to extend the installed software on the system,
without requiring specialist knowledge on how to build particular pieces
of software. Documentation for the project is here:
https://spack.readthedocs.io/

To install spack, execute the following and then login again:

.. code-block:: bash

   $ git clone https://github.com/spack/spack.git $HOME/spack

   $ echo 'export SPACK_ROOT=$HOME/spack' >> ~/.bash_profile
   $ echo 'source $SPACK_ROOT/share/spack/setup-env.sh' >> ~/.bash_profile

Example usage, installing an MPI aware, GPU version of gromacs and than
loading it into your environment to use (once built, execute
``spack load gromacs`` before using):

.. code-block:: bash

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

.. code-block:: bash

   cat > $SPACK_ROOT/etc/spack/upstreams.yaml <<EOF
   upstreams:
     spack-central:
       install_tree: /projects/<project>/spack
       modules:
         tcl: /projects/<project>/spack/share/spack/modules
   EOF