.. _software-applications-wmlce:

IBM WMLCE (End of Life)
=======================

.. warning:: 

   WMLCE was archived by IBM on 2020-11-10 and is no longer updated, maintained or supported.
   It is no longer available on bede due to the migration away from RHEL 7.

   It has been replaced by :ref:`Open Cognitiive Environment (Open-CE) <software-applications-open-ce>`, a community driven software distribution for machine learning.

   Open-CE does not not support all features of WMLCE.
   
   Please refer to the :ref:`Open-CE <software-applications-open-ce>` documentation for more information.

   Alternatively, consider moving to upstream sources for python packages such as :ref:`Tensorflow <software-applications-tensorflow>` or :ref:`PyTorch<software-applications-pytorch>` where available.

`IBM WMLCE <https://www.ibm.com/support/pages/get-started-ibm-wml-ce>`__ was the *Watson Machine Learning Community Edition* - a software distribution for machine learning which included IBM technology previews such as `Large Model Support for TensorFlow <https://www.ibm.com/support/knowledgecenter/SS5SF7_1.7.0/navigation/wmlce_getstarted_tflms.html?view=kc#wmlce_getstarted_tflms>`__.
WMLCE is also known as PowerAI.

It included a number of popular machine learning tools and frameworks such as :ref:`TensorFlow <software-applications-tensorflow>` and :ref:`PyTorch <software-applications-pytorch>`, enhanced for use on IBM POWER9 + Nvidia GPU based systems.
The use of :ref:`Conda<software-applications-conda>` to enable simple installation of multiple machine learning frameworks into a single software environment without users needing to manage complex dependency trees was another key feature of IBM WMLCE.

For more information, refer to the `IBM WMLCE documentation <https://www.ibm.com/support/pages/get-started-ibm-wml-ce>`__.

Using IBM WMLCE (End of Life)
-----------------------------

IBM WMLCE provided software packages via a hosted :ref:`Conda<software-applications-conda>` channel. 

Conda installations of the packages provided by WMLCE can become quite large (multiple GBs), so you may wish to use a conda installation in ``/nobackup/projects/<project>`` or ``/projects/<project>`` as described in the :ref:`Installing Conda section <software-applications-conda-installing>`.

With a working Conda install, IBM WMLCE packages can be installed from the IBM WMLCE conda channel: ``https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/``.

Using Conda Environments are recommended when working with Open-CE.

I.e. to install all WMLCE packages into a conda environment named ``wmlce``: 

.. note::

   IBM WMLCE requires Python 3.6 or Python 3.7. This may require an older Conda installation.

.. note:: 

   Installation of the full ``powerai`` package can take a considerable amount of time (hours) and consume a large amount of disk space of disk storage space.

.. code-block:: bash

   # Create a new python 3.6 conda environment named wmlce within your conda installation.
   # Your conda installation should be in the /nobackup filesystem.
   conda create -y --name wmlce python=3.6

   # Activate the conda environment
   conda activate wmlce

   # Add the IBM WMLCE channel to the environment
   conda config --env --prepend channels https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/

   # Enable strict channel priority for the environment
   conda config --env --set channel_priority strict

   # Install specific conda packages
   conda install -y tensorflow
   conda install -y pytorch
   
   # or the full powerai package, or powerai-cpu for the cpu version 
   conda install -y powerai

Once packages are installed into a named conda environment, the packages can be used interactively or within batch jobs by activating the conda environment.

.. code-block:: bash

   # activate the conda environment
   conda activate wmlce

   # Run a python command or script which makes use of the installed packages
   # I.e. to output the version of tensorflow:
   python3 -c "import tensorflow;print(tensorflow.__version__)"

   # I.e. or to output the version of pytorch:
   python3 -c "import torch;print(torch.__version__)"

IBM WMLCE includes `IBM Distributed Deep Learning (DDL) <https://www.ibm.com/docs/en/wmlce/1.6.0?topic=frameworks-getting-started-ddl>`__ which is an mpi-based library optimised for deep learning.
When an application is integrated with DDL, it becomes an MPI application which should be launched via a special command.
In WMLCE, DDL is integrated into PowerAI IBM Caffe, Pytorch, and TensorFlow.
This allows the use of multiple nodes when running machine learning models to support larger models and improved performance.

On Bede, this command is ``bede-ddlrun``. For example: 

.. code-block:: slurm

   #!/bin/bash

   # Generic options:

   #SBATCH --account=<project>  # Run job under project <project>
   #SBATCH --time=1:0:0         # Run for a max of 1 hour

   # Node resources:

   #SBATCH --partition=gpu    # Choose either "gpu" or "infer" node type
   #SBATCH --nodes=2          # Resources from a two nodes
   #SBATCH --gres=gpu:4       # Four GPUs per node (plus 100% of node CPU and RAM per node)

   # Run commands:

   conda activate wmlce

   bede-ddlrun python $CONDA_PREFIX/ddl-tensorflow/examples/keras/mnist-tf-keras-adv.py

.. warning::

   IBM DDL is not supported on RHEL 8 and will likely error on use.
   
   Consider migrating away from DDL via  :ref:`Open-CE<software-applications-open-ce>` and regular ``bede-mpirun``

WMLCE resnet50 benchmark (RHEL 7 only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The WMLCE conda channel includes a package ``tensorflow-benchmarks`` which provides a TensorFlow implementation of the resnet-50 model for benchmarking purposes.

When the ``tensorflow-benchmarks`` conda package is installed into the current conda environment, the documentation for this benchmark can be found at ``$CONDA_PREFIX/tensorflow-benchmarks/resnet50/README.md``.
Subsequent sections are based on the contents of the readme.

The remainder of this section describes how to execute this benchmark on Bede, 
using a conda environment named ``wmlce`` with ``tensorflow`` and ``tensorflow-benchmarks`` installed.  

The necessary data from ImageNet has been downloaded and processed.
It is stored in ``/nobackup/datasets/resnet50/TFRecords`` and is universally readable.

.. note::

   As written, the associated sbatch script must be run in a directory that is writeable by the user. 

   It creates a directory with the default name run_results into which it will write the results of the computation.    
   The results data will use up to 1.2GB of space.

   The run directory must also be accessible by the compute nodes, so using ``/tmp`` on a login node is not suitable.

The main WMLCE README.MD file suggests the following parameters are appropriate for a 4 node (up to 16 GPU) run:

.. code-block:: bash

 # Run a training job
 ddlrun -H host1,host2,host3,host4 python $CONDA_PREFIX/benchmarks/tensorflow-benchmarks/resnet50/main.py \
 --mode=train_and_evaluate --iter_unit=epoch --num_iter=50 --batch_size=256 --warmup_steps=100 \
 --use_cosine_lr --label_smoothing 0.1 --lr_init=0.256 --lr_warmup_epochs=8 --momentum=0.875 \
 --weight_decay=3.0517578125e-05   --data_dir=/data/imagenetTF/ --results_dir=run_results \
 --use_xla --precision=fp16  --loss_scale=1024 --use_static_loss_scaling

``ddlrun`` is not integrated with Slurm and will not run directly on Bede.
A wrapper-script called ``bede-ddlrun`` is available and that is what is used in the following.

A single GPU run of this benchmark can be completed without requiring ``ddlrun`` or ``bede-ddlrun`` the above set of parameters. 
The associated run takes about 16 hours to complete, however, the job may be killed due to insufficient host memory when only a single GPU is requested.

The related ``sbatch`` script (:download:`sbatch_resent50base.sh<wmlce/sbatch_resnet50base.sh>`
) is configured to use 4 GPUs on one node.
Changing the script to use 4 nodes, 16 GPUs, requires changing one line.

The sbatch script specifies:

.. code-block:: bash

   # ...
   #SBATCH --partition gpu
   #SBATCH --gres=gpu:4
   #SBATCH --nodes=1
   # ...

   export CONDADIR=/nobackup/projects/<project>/$USER # Update this with your <project> code.
   source $CONDADIR/miniconda/etc/profile.d/conda.sh
   # Activate the 
   conda activate wmlce

   export OMP_NUM_THREADS=1   # Disable multithreading

   bede-ddlrun python $CONDA_PREFIX/tensorflow-benchmarks/resnet50/main.py \
   --mode=train_and_evaluate --iter_unit=epoch --num_iter=50 --batch_size=256 \
   --warmup_steps=100 --use_cosine_lr --label_smoothing 0.1 --lr_init=0.256 \
   --lr_warmup_epochs=8 --momentum=0.875 --weight_decay=3.0517578125e-05  \
   --data_dir=/nobackup/datasets/resnet50/TFRecords/ --results_dir=run_results \
   --use_xla --precision=fp16  --loss_scale=1024 --use_static_loss_scaling



The resulting job should run for about 4 hours and will keep all 4 GPUs at nearly
100% utilisation.

The first few lines of output should look similar to:

.. code-block::

   [WARN DDL-2-17] Not performing connection tests. Cannot find 'mpitool' executabl
   e. This could be because you are using a version of mpi that does not ship with
   mpitool.
   Please see /tmp/DDLRUN/DDLRUN.j9SmSKzaKGEL/ddlrun.log for detailed log.
   + /opt/software/apps/anaconda3/envs/wmlce_env/bin/mpirun -x PATH -x LD_LIBRARY_P
   ATH -disable_gdr -gpu -mca plm_rsh_num_concurrent 1 --rankfile /tmp/DDLRUN/DDLRU
   N.j9SmSKzaKGEL/RANKFILE -n 4 -x DDL_HOST_PORT=2200 -x "DDL_HOST_LIST=gpu025.bede
   .dur.ac.uk:0,1,2,3" -x "DDL_OPTIONS=-mode p:4x1x1x1 " bash -c 'source /opt/softw
   are/apps/anaconda3/etc/profile.d/conda.sh && conda activate /opt/software/apps/a
   naconda3/envs/wmlce_env > /dev/null 2>&1 && python /opt/software/apps/anaconda3/
   envs/wmlce_env/tensorflow-benchmarks/resnet50/main.py --mode=train_and_evaluate
   --iter_unit=epoch --num_iter=50 --batch_size=256 --warmup_steps=100 --use_cosine
   _lr --label_smoothing 0.1 --lr_init=0.256 --lr_warmup_epochs=8 --momentum=0.875
   --weight_decay=3.0517578125e-05 --data_dir=/nobackup/datasets/resnet50/TFRecords
   / --results_dir=run_results --use_xla --precision=fp16 --loss_scale=1024 --use_s
   tatic_loss_scaling'
   2020-11-17 15:39:49.410620: I tensorflow/stream_executor/platform/default/dso_lo
   ader.cc:44] Successfully opened dynamic library libcudart.so.10.2

There are a number of configuration / compiler type messages and then you should
start to see messages like:

.. code-block:: 

   :::NVLOGv0.2.3 resnet 1605627653.398838758 (training_hooks.py:100) iteration: 0
   :::NVLOGv0.2.3 resnet 1605627653.400741577 (training_hooks.py:101) imgs_per_sec:
   37.5667719118656
   :::NVLOGv0.2.3 resnet 1605627653.402500391 (training_hooks.py:102) cross_entropy
   : 9.02121639251709
   :::NVLOGv0.2.3 resnet 1605627653.404244661 (training_hooks.py:103) l2_loss: 0.74
   98071789741516
   :::NVLOGv0.2.3 resnet 1605627653.405992270 (training_hooks.py:104) total_loss: 9
   .771023750305176
   :::NVLOGv0.2.3 resnet 1605627653.407735109 (training_hooks.py:105) learning_rate
   : 0.0
   :::NVLOGv0.2.3 resnet 1605627671.803228855 (training_hooks.py:100) iteration: 10
   :::NVLOGv0.2.3 resnet 1605627671.805866718 (training_hooks.py:101) imgs_per_sec:
   4526.812526349517
   :::NVLOGv0.2.3 resnet 1605627671.807682991 (training_hooks.py:102) cross_entropy
   : 8.204719543457031

The most relevant line is the value after ``imgs_per_sec``:

Once things start running, you should see something like 4500 images per second as
the rate on 4 GPUs.

After about 4 hours, the training has converged and you should see the last few lines like:

.. code-block::

   transpose_before=resnet50_v1.5/input_reshape/transpose pad=resnet50_v1.5/conv2d/Pad transpose_after=resnet50_v1.5/conv2d/conv2d/Conv2D-0-TransposeNCHWToNHWC-LayoutOptimizer
   :::NVLOGv0.2.3 resnet 1605641981.781752110 (runner.py:610) Top-1 Accuracy: 75.863
   :::NVLOGv0.2.3 resnet 1605641981.782602310 (runner.py:611) Top-5 Accuracy: 92.823
   :::NVLOGv0.2.3 resnet 1605641981.783382177 (runner.py:630) Ending Model Evaluation ...

It is easy to modify the script to use 4 nodes and hence 16 GPUs. The run time will
be a just over an hour and during the 16 GPU run, about 18000 images per second will
be processed.

Unfortunately, the basic parameters used with the resnet50 run do not allow this
job to scale much beyond 16 GPUs. 
Indeed, there is no speedup with this configuration using 32 GPUs.
