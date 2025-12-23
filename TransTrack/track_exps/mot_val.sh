#!/usr/bin/env bash

python main_track.py  --output_dir . --dataset_file mot --coco_path MOT17 --batch_size 1 --resume output_mix/checkpoint.pth --eval --with_box_refine --num_queries 500