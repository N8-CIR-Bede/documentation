************************************************************
Watson Machine Learning Community Edition resnet50 benchmark
************************************************************


This Bede specific README file is based upon options laid out in the README.MD file in the WMLCE
resnet50 benchmark directory. The necessary data from ImageNet has been downloaded and processed.
It is stored in /nobackup/datasets/resnet50/TFRecords and is universally readable.

NOTE: As written, the associated sbatch script must be run in a directory that is writable
by the user. It creates a directory with the default name run_results into which it will write
the results of the computation. The results data will use up to 1.2GB of space. The run
directory must also be accessible by the compute nodes, so using /tmp on a login node is not
suitable.

The main WMLCE README.MD file suggests the following parameters are appropriate for a 4 node
(possibly 16 GPU) run:


::

 # Run a training job
 ddlrun -H host1,host2,host3,host4 python benchmarks/tensorflow-benchmarks/resnet50/main.py \
 --mode=train_and_evaluate --iter_unit=epoch --num_iter=50 --batch_size=256 --warmup_steps=100 \
 --use_cosine_lr --label_smoothing 0.1 --lr_init=0.256 --lr_warmup_epochs=8 --momentum=0.875 \
 --weight_decay=3.0517578125e-05   --data_dir=/data/imagenetTF/ --results_dir=run_results \
 --use_xla --precision=fp16  --loss_scale=1024 --use_static_loss_scaling

ddlrun by itself is not integrated with Slurm and will not run directly on Bede. A wrapper-script
called bede-ddlrun is available and that is what is used in the following.

It is easy to define a single GPU run based on the above set of parameters (basically
remove the ddlrun command at the front and specify the correct paths). The associated run
takes about 16 hours to complete.

The related sbatch script ( :download:`sbatch_resnet50base.sh <sbatch_resnet50base.sh>`) is configured to use 4 GPUs on one node.
Changing the script to use 4 nodes, 16 GPUs, requires changing one line.


The sbatch script specifies:

::

 ...
 #SBATCH -p gpu
 #SBATCH --gres=gpu:4
 #SBATCH -N1
 ...

 module load slurm/dflt
 export PYTHON_HOME=/opt/software/apps/anaconda3/
 source $PYTHON_HOME/bin/activate wmlce_env

 export OMP_NUM_THREADS=1   # Disable multithreading

 bede-ddlrun python $PYTHON_HOME/envs/wmlce_env/tensorflow-benchmarks/resnet50/main.py \
 --mode=train_and_evaluate --iter_unit=epoch --num_iter=50 --batch_size=256 \
 --warmup_steps=100 --use_cosine_lr --label_smoothing 0.1 --lr_init=0.256 \
 --lr_warmup_epochs=8 --momentum=0.875 --weight_decay=3.0517578125e-05  \
 --data_dir=/nobackup/datasets/resnet50/TFRecords/ --results_dir=run_results \
 --use_xla --precision=fp16  --loss_scale=1024 --use_static_loss_scaling



The resulting job should run for about 4 hours and will keep all 4 GPUs at nearly
100% utilisation.

The first few lines of output should look similar to:
::

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

::

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

The most relevant line is the value after imgs_per_sec:

Once things start running, you should see something like 4500 images per second as
the rate on 4 GPUs.

After about 4 hours, the training has converged and you should see the last few lines like:

::

 transpose_before=resnet50_v1.5/input_reshape/transpose pad=resnet50_v1.5/conv2d/Pad transpose_after=resnet50_v1.5/conv2d/conv2d/Conv2D-0-TransposeNCHWToNHWC-LayoutOptimizer
 :::NVLOGv0.2.3 resnet 1605641981.781752110 (runner.py:610) Top-1 Accuracy: 75.863
 :::NVLOGv0.2.3 resnet 1605641981.782602310 (runner.py:611) Top-5 Accuracy: 92.823
 :::NVLOGv0.2.3 resnet 1605641981.783382177 (runner.py:630) Ending Model Evaluation ...

It is easy to modify the script to use 4 nodes and hence 16 GPUs. The run time will
be a just over an hour and during the 16 GPU run, about 18000 images per second will
be processed.

Unfortunately, the basic parameters used with the resnet50 run do not allow this
job to scale much beyond 16 GPUs. Indeed, there is no speedup with this configuration
using 32 GPUs. Improving scalability is left as an exercise for the user.
 
 
