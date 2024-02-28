.. _RHEL8-migration:

:orphan:

RHEL 8 Migration
================

Bede completed a major Operating System upgrade from Red Hat Enterprise Linux 7 (RHEL 7) to Red Hat Enterprise Linux 8 (RHEL 8) on the 3rd of May 2022.

This upgrade enables the use of newer software versions, such as CUDA 11 which were not supported on RHEL 7.

However, this migration may have had an impact on certain workloads:

* The vendor-supplied set of modules has been removed
* Multi-node IBM WMLCE functionality is not supported on RHEL 8
* User-installed applications (particularly MPI programs) likely needed recompiling.

Module Changes
--------------

Most existing modules from the RHEL 7 installation are available on RHEL 8, with newer versions of some modules (i.e. CUDA, NVHPC, IBM XL) also available.

There are however a few exceptions:

* Singularity no longer requires a module load, it is available by default.
* ``mvapich2/2.3.5`` is no longer available. ``mvapich2/2.3.5-2`` which was also available on RHEL 7 images should be used instead.
* ``nvhpc/20.9`` was replaced by ``nvhpc/21.5``.
* ``spack/central`` is not available as a module. Spack can be installed per-user via ``git``. Please see the :ref:`Spack documentation <software-spack>` for more details.
* ``slurm/19.05.7`` and ``19.05.7b`` are not available, with ``slurm/dflt`` loaded by default.
* ``tools/1.0`` and ``tools/1.1`` are not available, with ``tools/1.2`` loaded by default.
* HECBioSim provided modules such as AMBER, GROMACS and NAMD should use the ``-rhel8`` postfixed modules.

Other Notable Changes
---------------------

In addition to the changes to available software modules, the upgrade from RHEL 7 to RHEL 8 includes several other changes which may impact your use of Bede.
Including:

* ``glibc`` is ``2.28`` on RHEL 8, compared to ``2.17`` on RHEL 7.
* The default ``python`` executable is ``python3`` in RHEL 8, compared to ``python2`` in RHEL 7. It is recommended to explicitly use ``python3`` rather than ``python``.
* The default (native) ``gcc`` is GCC ``8.5.0`` on RHEL 8, compared to GCC ``4.8.5`` on RHEL 7.

Migration Process
-----------------

The migration from RHEL 7 to RHEL 8 was completed in three phases:
1. Users to test the RHEL 8 image (Completed 2022-03-17)
2. Login nodes migrate to RHEL 8 (Completed 2022-03-17)
3. Compute nodes migrate to RHEL 8 as load permits (Completed 2022-05-03)

