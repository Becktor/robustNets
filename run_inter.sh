module load python3/3.6.7
module load cudnn/v8.2.0.53-prod-cuda-11.3
source /work1/jbibe/a100/bin/activate
python train.py --csv_train /work1/jbibe/datasets/dataset_csvs/reannotation_set_hpc.csv --csv_classes classes.csv --csv_val /work1/jbibe/datasets/dataset_csvs/reannotation_valset_hpc.csv --csv_weight /work1/jbibe/datasets/dataset_csvs/weightset_85.csv --batch_size=16 --depth=50 --flip_mod 0 --rew_start 30 --reannotate True
