.. _software-tools-cmake:

CMake
=====

`CMake <https://cmake.org/>`__ is an open-source, cross-platform family of tools designed to build, test and package software.
CMake is used to control the software compilation process using simple platform and compiler independent configuration files, and generate native makefiles and workspaces that can be used in the compiler environment of your choice.
The suite of CMake tools were created by `Kitware <https://www.kitware.com/>`__ in response to the need for a powerful, cross-platform build environment for open-source projects such as ITK and VTK.

CMake is part of Kitware’s collection of commercially supported `open-source platforms <https://www.kitware.com/platforms/>`__ for software development.


.. code-block:: bash

    module load cmake
    module load cmake/3.18.4

Once loaded, the ``cmake``, ``ccmake``, ``cpack`` and ``ctest`` binaries are available for use, to configure, build and test software which uses CMake as the build system. 

For more information, see the `online documentation <https://cmake.org/cmake/help/v3.18/>`__.