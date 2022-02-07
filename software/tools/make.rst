.. _software-tools-make:

Make
====

`GNU Make <https://www.gnu.org/software/make/>`__ is a tool which controls the generation of executables and other non-source files of a program from the program's source files.

Make gets its knowledge of how to build your program from a file called the makefile, which lists each of the non-source files and how to compute it from other files. When you write a program, you should write a makefile for it, so that it is possible to use Make to build and install the program.


On Bede, ``make 3.82`` is provided by default (``4.2`` under RHEL8). 
A more recent version of ``make``, is provided by the ``make`` family of modules. 

.. code-block:: bash

    module load make
    module load make/4.3

For more information on the usage of ``make``, see the `online documentation <https://www.gnu.org/software/make/manual/>`__ or run ``man make`` after loading the module.