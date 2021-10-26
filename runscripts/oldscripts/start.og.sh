#!/bin/bash
source /home/jobe/venv/bin/activate
python train.py --csv_train /home/jobe/sftp/transfer/mmdet/annotations_rgb_train_big.csv --csv_classes /home/jobe/sftp/transfer/mmdet/classes2.csv --csv_val /home/jobe/sftp/transfer/mmdet/annotations_rgb_val_big.csv --csv_weight /home/jobe/sftp/transfer/weightset_big.csv --batch_size=32
