.. _software-applications-amber:

AMBER
-------

`AMBER <https://ambermd.org/>`__ is a suite of biomolecular simulation programs. It began in the late 1970's, and is maintained by an active development community.

On Bede, AMBER is made available through the :ref:`HECBioSim Project <software-projects-hecbiosim>`.


.. code-block:: bash

   # Load the hecbiosim project
   module load hecbiosim

   # Load the desired version of amber
   module load amber/20-large-system-mod
   module load amber/20


The HECBioSim project also provide `example bede job submission scripts for AMBER on their website <https://www.hecbiosim.ac.uk/access-hpc/example-submit-scripts/bede-scripts>`__.

For more information see the `AMBER documentation <https://ambermd.org/Manuals.php>`__ and `information on GPU acceleration within AMBER <https://ambermd.org/GPUSupport.php>`__.




