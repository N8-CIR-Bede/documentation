.. _bede_scheduler:

Running and Scheduling Tasks on Bede
####################################

Slurm Workload Manager
======================

Slurm is a highly scalable cluster management and job scheduling system, used in Bede. As a cluster workload manager, Slurm has three key functions:

* it allocates exclusive and/or non-exclusive access to resources (compute nodes) to users for some duration of time so they can perform work,
* it provides a framework for starting, executing, and monitoring work on the set of allocated nodes,
* it arbitrates contention for resources by managing a queue of pending work.

Loading Slurm
=============

Slurm must first be loaded before tasks can be scheduled.

.. code-block:: sh

    module load slurm

For more information on modules, see :ref:`bede_module`.


Request an Interactive Shell
============================

Launch an interactive session on a worker node using the command:

.. code-block:: sh

    srun --pty bash

You can request an interactive node with GPU(s) by using the command:

.. code-block:: sh

    # --gpus=1 requests 1 GPU for the session session, the number can be 1, 2 or 4
    srun --gpus=1 --pty bash


You can add additional options, e.g. request additional memory:

.. code-block:: sh

    # Requests 16GB of RAM with 1 GPU
    srun --mem=16G --gpus=1 --pty bash



Submitting Non-Interactive Jobs
===============================

Write a job-submission shell script
-----------------------------------

You can submit your job, using a shell script. A general job-submission shell script contains the "bang-line" in the first row.

.. code-block:: sh

    #!/bin/bash

Next you may specify some additional options, such as memory,CPU or time limit.

.. code-block:: sh

    #SBATCH --"OPTION"="VALUE"

Load the appropriate modules if necessery.

.. code-block:: sh

    module load PATH
    module load MODULE_NAME

Finally, run your program by using the Slurm "srun" command.

.. code-block:: sh

    srun PROGRAM

The next example script requests 2 GPUs and 16Gb memory. Notifications will be sent to an email address:

.. code-block:: sh

    #!/bin/bash
    #SBATCH --gpus=2
    #SBATCH --mem=16G
    #SBATCH --mail-user=username@mydomain.com

    module load cuda
    
    # Replace my_program with the name of the program you want to run
    srun my_program



Job Submission
--------------

Save the shell script (let's say "submission.slurm") and use the command

.. code-block:: bash

    sbatch submission.slurm

Note the job submission number. For example:

.. code-block:: bash

    Submitted batch job 1226

Check your output file when the job is finished.  

.. code-block:: bash

    # The JOB_NAME value defaults to "slurm"
    cat JOB_NAME-1226.out

Common job options
==================

Optional parameters can be added to both interactive and non-interactive jobs. Options can be appended to the command line or added to the job submission scripts.

* Setting maximum execution time
    * ``--time=hh:mm:ss`` - Specify the total maximum execution time for the job. The default is 48 hours (48:00:00)
* Memory request
    * ``--mem=#``- Request memory (default 4GB), suffixes can be added to signify Megabytes (M) or Gagabytes (G) e.g. ``--mem=16G`` to request 16GB.
    * Alternatively ``--mem-per-cpu=#`` or ``--mem-per-gpu=#`` - Memory can be requested per CPU with ``--mem-per-cpu`` or per GPU ``--mem-per-gpu``, these three options are mutually exclusive.
* GPU request
    * ``--gpus=1`` - Request GPU(s), the number can be 1, 2 or 4.
* CPU request
    * ``-c 1`` or ``--cpus-per-task=1`` - Requests a number of CPUs for this job, 1 CPU in this case.
    * ``--cpus-per-gpu=2`` - Requests a number of CPUs **per** GPU requested. In this case we've requested 2 CPUs per GPU so if ``--gpus=2`` then 4 CPUs will be requested.
* Specify output filename
    * ``--output=output.%j.test.out``
* E-mail notification
    * ``--mail-user=username@sheffield.ac.uk`` - Send notification to the following e-mail
    * ``--mail-type=type`` - Send notification when type is ``BEGIN``, ``END``, ``FAIL``, ``REQUEUE``, or ``ALL``
* Naming a job
    * ``--job-name="my_job_name"`` - The specified name will be appended to your output (``.out``) file name.
* Add comments to a job
    * ``--comment="My comments"``

For the full list of the available options please visit the Slurm manual webpage at https://slurm.schedmd.com/pdfs/summary.pdf.

Key SLURM Scheduler Commands
============================

Display the job queue. Jobs typically pass through several states in the course of their execution. The typical states are PENDING, RUNNING, SUSPENDED, COMPLETING, and COMPLETED.

.. code-block:: sh

    squeue

Shows job details:

.. code-block:: sh

    sacct -v

Details the HPC nodes:

.. code-block:: sh

    sinfo

Deletes job from queue:

.. code-block:: sh

    scancel JOB_ID
