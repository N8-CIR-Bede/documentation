.. _software-applications-openmm:

OpenMM
------

.. |arch_availabilty_name| replace:: OpenMM
.. include:: /common/ppc64le-only.rst

`OpenMM <https://openmm.org/>`__ is a high-performance toolkit for molecular simulation. 
It can be used as an application, a library, or a flexible programming environment
and includes extensive language bindings for Python, C, C++, and even Fortran.

On Bede, OpenMM is made available through the :ref:`HECBioSim Project <software-projects-hecbiosim>`.


.. code-block:: bash

   # Load the hecbiosim project
   module load hecbiosim
   
   # Load the desired version of openmm
   module load openmm
   module load openmm/7.4.1-python3.7

For more information see the `OpenMM Documentation <https://openmm.org/documentation>`__.