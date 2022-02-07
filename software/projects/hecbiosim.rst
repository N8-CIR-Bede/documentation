.. _software-projects-hecbiosim:

HECBioSim
=========

The `HEC BioSim consortium <http://www.hecbiosim.ac.uk/>`__ focusses on molecular simulations, at a variety of time and length scales but based on well-defined physics to complement experiment.
The unique insight they can provide gives molecular level understanding of how biological macromolecules function.
Simulations are crucial in analysing protein folding, mechanisms of biological catalysis, and how membrane proteins interact with lipid bilayers.
A particular challenge is the integration of simulations across length and timescales: different types of simulation method are required for different types of problems.

On Bede, the HECBioSim project provides several software packages for molecular simulation, including:

* :ref:`AMBER <software-applications-amber>`
* :ref:`GROMACS <software-applications-gromacs>`
* :ref:`NAMD <software-applications-namd>`
* :ref:`OpenMM <software-applications-openmm>`
* :ref:`PLUMED <software-libraries-plumed>`

Once the ``hecbiosim`` module has been loaded, it is possible to load versions of the provided packages.

.. code-block:: bash

    # Load the hecbiosim module
    module load hecbiosim
    # Once loaded, modules provided by hecbiosim are available, such as gromacs
    module load gromacs

For more information on the HEC BioSim consortium please see the `HECBioSim Website <http://www.hecbiosim.ac.uk/>`__.
