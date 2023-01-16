#!/bin/bash
########## SBATCH Lines for Resource Request ##########
#SBATCH --time=35:59:00             	# limit of wall clock time - how long the job will run (same as -t)
#SBATCH --nodes=1                 		# number of different nodes - could be an exact number or a range of nodes (same as -N)
#SBATCH --ntasks=1                  	# number of tasks - how many tasks (nodes) that you require (same as -n)
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16           	# number of CPUs (or cores) per task (same as -c)
#SBATCH --mem-per-cpu=6G            	# memory required per allocated CPU (or core) - amount of memory (in bytes)
#SBATCH --job-name lightning_nopvt_scannet_hpcc
#SBATCH -o /mnt/home/zhusheng/slurm_log/lightning_nopvt_scannet_hpcc.log
#SBATCH --mail-type=fail         # send mail if job fails
#SBATCH --mail-user zhusheng@msu.edu
########## Command Lines to Run ##########
bash
cd /mnt/home/zhusheng/research/SimMIMDenseMatch ### change to the directory where your code is located
pwd
source activate SimMIM ### Activate virtual environment

/mnt/home/zhusheng/anaconda3/envs/SimMIM/bin/python -m torch.distributed.launch \
--nproc_per_node 2 DKMResnetLoFTRPreTrainNoPVT/lightning_nopvt_scannet.py \
--cfg /mnt/home/zhusheng/research/SimMIMDenseMatch/configs/swin_base__100ep/simmim_pretrain__swin_base__img192_window6__100ep.yaml \
--batch-size 18 \
--data-path /mnt/scratch/zhusheng/EMAwareFlow/ImageNet \
--data-path-scannet /mnt/scratch/zhusheng/EMAwareFlow/ScanNet \
--output /mnt/scratch/zhusheng/checkpoints/SimMIMDenseMatch/checkpoints \
--tag AblatePretrain/lightning_nopvt_scannet_hpcc \
--minoverlap-scannet 0.3

scontrol show job $SLURM_JOB_ID ### write job information to output file