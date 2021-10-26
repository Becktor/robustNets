#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J reweight
### -- ask for number of cores (default: 1) --
#BSUB -n 2
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm -- maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 5GB of system-memorya
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "select[gpu32gb]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u jbibe@elektro.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o reweight_%J.out
#BSUB -e reweight_%J.err
# -- end of LSF options --
# Load the cuda module
module load python3/3.6.7
module load cudnn/v7.6.5.32-prod-cuda-10.0
source /work1/jbibe/venv/bin/activate
python train.py --csv_train /work1/jbibe/100k_annot95_auto5_hpc.csv --csv_classes classes.csv --csv_val /work1/jbibe/mmdet/fix_annotations_rgb_val.csv --csv_weight /work1/jbibe/weightset_hpc2.csv --batch_size=16 --continue_training /work1/jbibe/git/robustNets/trained_models/gentle-plasma-210_ioodhje1
