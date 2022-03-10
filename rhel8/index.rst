.. _RHEL8-migration:

RHEL 8 Migration
================

Bede is in the process of an Operating System upgrade from Red Hat Enterprise Linux 7 (RHEL 7) to Red Hat Enterprise Linux 8 (RHEL 8).
This upgrade will enable the use of newer software versions, such as CUDA 11.

However, it may impact your use of Bede:

* The vendor-supplied set of modules will be removed
* Multi-node IBM WMLCE functionality is not supported on RHEL 8
* User-installed applications (particularly MPI programs) will likely need recompiling.

Migration Process
-----------------

The migration from RHEL 7 to RHEL 8 has three mains steps:

1. Users to test the RHEL 8 image
2. Login nodes migrate to RHEL 8 (postponed to 2022-03-17)
3. Compute nodes migrate to RHEL 8 as load permits


Two new commands have been added to Bede for the duration of the OS migration: ``login8`` and ``login7``.

* ``login8`` will connect you to a RHEL 8 interactive session
* ``login7`` will connect you to a RHEL 7 interactive session

The RHEL version used to execute batch jobs will be the version in use at job submission time.
I.e. Jobs submit from a login node using RHEL 8 will execute on RHEL 8 worker node(s). 

User Testing
^^^^^^^^^^^^

Initially 2 nodes from the ``gpu`` partition, and 1 node from the ``infer`` partition have been migrated to RHEL 8 for user testing. 
A second ``infer`` node is reserved for interactive RHEL 8 sessions. 

To opt-in to using the RHEL 8 image:

1. Connect to the login nodes as usual
2. Run the ``login8`` command to gain an interactive session on an RHEL 8 node
3. Load modules, compile code or submit jobs as usual.

You may need to change your module load commands in some cases (see :ref:`rhel8-module-changes`), 
and will likely need to recompile your codes for use on RHEL 8 (particularly MPI codes).


Login Node Migration
^^^^^^^^^^^^^^^^^^^^

Once the period of time for users to opt-in to using RHEL 8 to ensure there are no issues for their workflows has ended, Bede's two login nodes will be migrated to RHEL 8.

From this time, when you connect to Bede you will immediately be connected to RHEL 8 sessions on the login nodes, and the ``login8`` command will no longer be required.

The ``login7`` command will be usable to connect to an interactive session on a RHEL 7 login node for a limited period of time.
 
**Login node migration has been postponed until 2022-03-17**.

Compute Node Migration
^^^^^^^^^^^^^^^^^^^^^^

Once the login nodes have been migrated to RHEL 8, the remaining compute nodes will be migrated to RHEL 8 as demand allows.

During this time the ``login7`` command will still be available for users to connect to a RHEL 7 interactive sessions to submit jobs to RHEL 7 compute nodes.

Initially, the capacity for RHEL 8 jobs will be low, increasing as more nodes are migrated.

Conversely, the capacity for RHEL 7 jobs will initially be high but will decrease over time.

This will likely impact queue time for your jobs, and may prevent multi-node jobs from being scheduled if the requested number of nodes is not available for the RHEL version used.

Module Changes
--------------

Most existing modules from the RHEL 7 installation are available on RHEL 8, with newer versions of some modules (CUDA, NVHPC, IBM XL) also available.

There are however a few exceptions:

* Singularity no longer requires a module load, it is available globally by default.
* ``mvapich2/2.3.5`` is not provided on RHEL 8 images. ``mvapich2/2.3.5-2`` which is provided on both RHEL 7 and RHEL 8 should be used instead.
* ``nvhpc/20.9`` is not available, replaced by ``nvhpc/21.5``.
* ``spack/central`` is not available as a module. Spack can be installed per-user via ``git``. Please see the :ref:`Spack documentation <software-spack>` for more details.
* ``slurm/19.05.7`` and ``19.05.7b`` are not available, with ``slurm/dflt`` loaded by default.
* ``tools/1.0`` and ``tools/1.1`` are not available, with ``tools/1.2`` loaded by default.
* HECBioSim provided modules such as AMBER, GROMACS and NAMD should use the ``-rhel8`` postfixed modules

Other Notable Changes
---------------------

In addition to the changes to available software modules, the upgrade from RHEL 7 to RHEL 8 includes several other changes which may impact your use of Bede.
Including:

* ``glibc`` is ``2.28`` on RHEL 8, compared to ``2.17`` on RHEL 7.
* The default ``python`` executable is ``python3`` in RHEL 8, compared to ``python2`` in RHEL 7. It is recommended to explicitly use ``python3`` rather than ``python``.
* The default (native) ``gcc`` is GCC ``8.5.0`` on RHEL 8, compared to GCC ``4.8.5`` on RHEL 7.

Checking Node Availability
--------------------------

As compute nodes are migrated from RHEL 7 to RHEL 8, the capacity for jobs using each image will vary, impacting queue time and the maximum size of multi-node jobs.

Information on how many nodes are running RHEL 7 or RHEL 8 can be found using the ``ACTIVE_FEATURES`` format option of ``sinfo`` (``%b``):

.. code-block:: bash

   # See how many nodes in the gpu partition have the rhel7 or rhel8 feature
   sinfo -o "%9P %.5a %.10l %.6D %15b %N" -p gpu

   # See how many nodes in the infer partition have the rhel7 or rhel8 feature
   sinfo -o "%9P %.5a %.10l %.6D %15b %N" -p infer


Checking Batch Job Requested Image
----------------------------------

``squeue`` can show which RHEL version a job will execute on, using the ``FEATURES`` format option ``%f``:

.. code-block:: bash

   # List queue information for $USER's jobs, including FEATURES (3rd column)
   squeue -o "%.19i %.9P %.6f %.8a %.8j %.8u %.2t %.10M %.6D %C %R" -u $USER


.. _rhel8-module-changes:

Checking the RHEL version
-------------------------

If at any point you wish to check which version of RHEL you are currently using, you can use:

.. code-block:: bash

   cat /etc/redhat-release


If you would like to use the RHEL version programmatically, it may be simpler to use the contents of ``/etc/os-release`` to access the version number:

.. code-block:: bash
   
   #! /usr/bin/env bash
   
   case $(source /etc/os-release && echo $VERSION_ID) in
       8*)
           # RHEL 8 specific commands  
           ;;
       7*)
           # RHEL 7 specific commands
           ;;
   esac
   
