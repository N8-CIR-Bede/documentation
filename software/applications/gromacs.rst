.. _software-applications-gromacs:

GROMACS
-------

`GROMACS <http://www.gromacs.org/About_Gromacs>`__ is a versatile package for molecular dynamics simulation.
It is primarily designed for biochemical molecules like proteins, lipids and nucleic acids that have a lot of complicated bonded interactions, but since GROMACS is extremely fast at calculating the nonbonded interactions (that usually dominate simulations) many groups are also using it for research on non-biological systems, e.g. polymers.

CUDA-based GPU acceleration is available for Since GROMACS >= 4.6, for Nvidia compute capability >= 2.0 GPUs (e.g. Fermi or later).


On Bede, GROMACS is made available through the :ref:`HECBioSim Project <software-projects-hecbiosim>`.


.. code-block:: bash

   # Load the hecbiosim project
   module load hecbiosim
   
   # Load the desired version of gromacs
   module load gromacs/2020.4-plumed-2.6.2-rhel8
   module load gromacs/2021.1-plumed-2.7.2-rhel8
   module load gromacs/2021.2-plumed-2.7.1-rhel8
   module load gromacs/2021.2-plumed-2.7.2-rhel8
   module load gromacs/2021.4-plumed-2.7.3-rhel8
   module load gromacs/2021.5-rhel8
   module load gromacs/2022.0-rhel8
   module load gromacs/2022.2
   module load gromacs/2023.1


The HECBioSim project also provide `example bede job submission scripts for GROMACS on their website <https://www.hecbiosim.ac.uk/access-hpc/example-submit-scripts/bede-scripts>`__.

For more information see the `GROMACS documentation <https://manual.gromacs.org/documentation/>`__ and `information on GPU acceleration within GROMACS <http://www.gromacs.org/GPU_acceleration>`__.




