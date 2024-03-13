.. _software-applications-namd:

NAMD
----

.. |arch_availabilty_name| replace:: NAMD
.. include:: /common/ppc64le-only.rst

`NAMD <https://www.ks.uiuc.edu/Research/namd/>`__ is a parallel molecular dynamics code designed for high-performance simulation of large biomolecular systems.
Based on Charm++ parallel objects, NAMD scales to hundreds of cores for typical simulations and beyond 500,000 cores for the largest simulations.

On Bede, NAMD is made available through the :ref:`HECBioSim Project <software-projects-hecbiosim>`.


.. code-block:: bash

   # Load the hecbiosim project
   module load hecbiosim

   # Load the desired version of namd
   module load namd
   module load namd/2.14-smp
   module load namd/3.0-alpha12-singlenode
   module load namd/3.0-alpha9-singlenode


For more information see the `NAMD User's Guide <https://www.ks.uiuc.edu/Research/namd/2.14/ug/>`__.


