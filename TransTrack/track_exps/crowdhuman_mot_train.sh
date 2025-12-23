#!/usr/bin/env bash

python -m torch.distributed.launch --nproc_per_node=2 --use_env main_track.py  --output_dir ./output --dataset_file mot --coco_path mot --batch_size 2 --resume crowdhuman_final.pth  --with_box_refine  --num_queries 500 --track_train_split trainall  --epochs 20 --lr_drop 10
