#!/bin/bash -l 
#SBATCH -A bdXXXYY
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH -N1 
#SBATCH -o multix1.o%j
#SBATCH -t 4:20:00
#
# Author: C. Addison 
# Initial version: 2020-11-19
#
# Please read the file bede-README-batch.txt for details on this
# script.
#
echo =========================================================   
echo SLURM job: submitted  date = `date`
date_start=`date +%s`

echo Nodes involved:
echo $SLURM_NODELIST
echo =========================================================   
echo Job output begins                                           
echo ----------------- 
echo
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



echo   
echo ---------------                                           
echo Job output ends                                           
date_end=`date +%s`
seconds=$((date_end-date_start))
minutes=$((seconds/60))
seconds=$((seconds-60*minutes))
hours=$((minutes/60))
minutes=$((minutes-60*hours))
echo =========================================================   
echo SLURM job: finished   date = `date`   
echo Total run time : $hours Hours $minutes Minutes $seconds Seconds
echo =========================================================   

