.. _software-libraries-plumed:

PLUMED
------

.. |arch_availabilty_name| replace:: PLUMED
.. include:: /common/ppc64le-only-sidebar.rst

`PLUMED <https://www.plumed.org/>`__, the community-developed PLUgin for MolEcular Dynamics, is a an open-source, community-developed library that provides a wide range of different methods, which include:

* enhanced-sampling algorithms
* free-energy methods
* tools to analyze the vast amounts of data produced by molecular dynamics (MD) simulations.

PLUMED works together with some of the most popular MD engines, including :ref:`GROMACS <software-applications-gromacs>`, :ref:`NAMD <software-applications-namd>` and :ref:`OpenMM <software-applications-openmm>` which are available on Bede.


On Bede, PLUMED is made available through the :ref:`HECBioSim Project <software-projects-hecbiosim>`.


.. code-block:: bash

   # Load the hecbiosim project
   module load hecbiosim
   
   # Load the desired version of PLUMED
   module load plumed/2.6.2-rhel8
   module load plumed/2.7.1-rhel8
   module load plumed/2.7.2-rhel8
   module load plumed/2.7.3-rhel8
   module load plumed/2.8.0-rhel8


For more information see the `PLUMED Documentation <https://www.plumed.org/doc>`__.




