Usage
=====

Bede is running Red Hat Enterprise Linux 7 and access to its
computational resources is mediated by the Slurm batch scheduler.

Registering
-----------

Access to the machine is based around projects:

-  For information on how to register a new project, please see https://n8cir.org.uk/supporting-research/facilities/bede/docs/bede_registrations/

-  To create an account to use the system:

   -  Identify an existing project, or register a new one.
   -  Create an EPCC SAFE account and login to the SAFE system at:
      https://safe.epcc.ed.ac.uk/
   -  Once there, select “Project->Request access” from the web
      interface and then register against your project

Login
-----

Bede offers an SSH service running on host ``bede.dur.ac.uk`` (which
fronts the two login nodes, ``login1.bede.dur.ac.uk`` and
``login2.bede.dur.ac.uk``). SSH should be used for all interaction with
the machine (including shell access and file transfer).

The login nodes are shared between all users of the service and
therefore should only be used for light interactive work, for example:
downloading and compiling software, editing files, preparing jobs and
examining job output. Short test runs using their CPUs and GPUs are also
acceptable.

Most of the computational power of the system is accessed through the
batch scheduler, and so demanding applications should be submitted to it
(see “Running Jobs”).

Acknowledging Bede
------------------

All work that makes use of the Bede should properly acknowledge the facility
wherever the work is presented.

We provide the following acknowledgement text, and strongly encourage its use:

*"This work made use of the facilities of the N8 Centre of Excellence in
Computationally Intensive Research (N8 CIR) provided and funded by the N8
research partnership and EPSRC (Grant No. EP/T022167/1). The Centre is
co-ordinated by the Universities of Durham, Manchester and York."*

Acknowledgement of Bede provides data that can be used to assess the facility's
success and influences future funding decisions, so please ensure that you are
acknowledging where appropriate.

File Storage
------------

Each project has access to the following shared storage:

-  Project home directory (``/projects/<project>``)

   -  Intended for project files to be backed up (note: backups not
      currently in place)
   -  Modest performance
   -  A default quota of 20GB

-  Project Lustre directory (``/nobackup/projects/<project>``)

   -  Intended for bulk project files not requiring backup
   -  Fast performance
   -  No quota limitations

By default, files created within a project area are readable and
writable by all other members of that project.

In addition, each user has:

-  Home directory (``/users/<user>``)

   -  Intended for per-user configuration files.
   -  Modest performance
   -  A default quota of 20GB

Please note that, as access to Bede is driven by project use, no
personal data should be stored on the system.

Current utilisation and limits of a user’s home directory can be found
by running the ``quota`` command. Similar information can be found for the
project home directory using the ``df -h /projects/<project>`` command.

To examine how much space is occupied by a project's Lustre directory,
a command of the form ``du -csh /nobackup/projects/<project>`` is
required. As ``du`` will check each and every file under the specified
directory, this may take a long time to complete. We plan to develop
the service and provide this information in a more responsive format in
the future.


Running Jobs
------------

Access beyond the two login node systems should only be done through the
Slurm batch scheduler, by packaging your work into units called jobs.

A job consists of a shell script, called a job submission script,
containing the commands that the job will run in sequence. In addition,
some specially formatted comment lines are added to the file, describing
how much time and resources the job needs.

Resources are requested in terms of the type of node, the number of GPUs
per node (for each GPU requested, the job receives 25% of the node’s
CPUs and RAM) and the number of nodes required.

There are a number of example job submission scripts below.

Requesting resources
~~~~~~~~~~~~~~~~~~~~

Part of, or an entire node
^^^^^^^^^^^^^^^^^^^^^^^^^^

Example job script for programs written to take advantage of a GPU or
multiple GPUs on a single computer:

::

   #!/bin/bash

   # Generic options:

   #SBATCH --account=<project>  # Run job under project <project>
   #SBATCH --time=1:0:0         # Run for a max of 1 hour

   # Node resources:
   # (choose between 1-4 gpus per node)

   #SBATCH --partition=gpu    # Choose either "gpu" or "infer" node type
   #SBATCH --nodes=1          # Resources from a single node
   #SBATCH --gres=gpu:1       # One GPU per node (plus 25% of node CPU and RAM per GPU)

   # Run commands:

   nvidia-smi  # Display available gpu resources

   # Place other commands here

   echo "end of job"

Multiple nodes (MPI)
^^^^^^^^^^^^^^^^^^^^

Example job script for programs using MPI to take advantage of multiple
CPUs/GPUs across one or more machines:

::

   #!/bin/bash

   # Generic options:

   #SBATCH --account=<project>  # Run job under project <project>
   #SBATCH --time=1:0:0         # Run for a max of 1 hour

   # Node resources:

   #SBATCH --partition=gpu    # Choose either "gpu" or "infer" node type
   #SBATCH --nodes=2          # Resources from a two nodes
   #SBATCH --gres=gpu:4       # Four GPUs per node (plus 100% of node CPU and RAM per node)

   # Run commands:

   bede-mpirun --bede-par 1ppc <mpi_program>

   echo "end of job"

The ``bede-mpirun`` command takes both ordinary ``mpirun`` arguments and
the special ``--bede-par <distrib>`` option, allowing control over how
MPI jobs launch, e.g. one MPI rank per CPU core or GPU.

The formal specification of the option is:
``--bede-par <rank_distrib>[:<thread_distrib>]`` and it defaults to
``1ppc:1tpt``

Where ``<rank_distrib>`` can take ``1ppn`` (one process per node),
``1ppg`` (one process per GPU), ``1ppc`` (one process per CPU core) or
``1ppt`` (one process per CPU thread).

And ``<thread_distrib>`` can take ``1tpc`` (set ``OMP_NUM_THREADS`` to
the number of cores available to each process), ``1tpt`` (set
``OMP_NUM_THREADS`` to the number of hardware threads available to each
process) or ``none`` (set ``OMP_NUM_THREADS=1``)

Examples:

::

   # - One MPI rank per node:
   bede-mpirun --bede-par 1ppn <mpirun_options> <program>

   # - One MPI rank per gpu:
   bede-mpirun --bede-par 1ppg <mpirun_options> <program>

   # - One MPI rank per core:
   bede-mpirun --bede-par 1ppc <mpirun_options> <program>

   # - One MPI rank per hwthread:
   bede-mpirun --bede-par 1ppt <mpirun_options> <program>

Multiple nodes (IBM PowerAI DDL)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

IBM PowerAI DDL (Distributed Deep Learning) is a method of using the
GPUs in more than one node to perform calculations. Example job script:

::

   #!/bin/bash

   # Generic options:

   #SBATCH --account=<project>  # Run job under project <project>
   #SBATCH --time=1:0:0         # Run for a max of 1 hour

   # Node resources:

   #SBATCH --partition=gpu    # Choose either "gpu" or "infer" node type
   #SBATCH --nodes=2          # Resources from a two nodes
   #SBATCH --gres=gpu:4       # Four GPUs per node (plus 100% of node CPU and RAM per node)

   # Run commands:

   # (assume IBM Watson Machine Learning Community Edition is installed
   # in conda environment "wmlce")

   conda activate wmlce

   bede-ddlrun python $CONDA_PREFIX/ddl-tensorflow/examples/keras/mnist-tf-keras-adv.py

   echo "end of job"
