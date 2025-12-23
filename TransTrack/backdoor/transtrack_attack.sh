#!/bin/bash

# Set variables
mot_data_path='/home/zyl/ours1/datasets/MOT17_transtrack_poison'
checkpoint_path="/home/zyl/ours1/TransTrack/output/checkpoint.pth"

### Convert dataset format
python track_tools/convert_mot_to_coco.py

#### Poison the training set
python backdoor/poison_dataset_transtrack.py --attack train

## Train the backdoored MOT17 model (trained on MOT17-train-half)
python -m torch.distributed.launch --nproc_per_node=2 --use_env main_track.py \
 --output_dir output --dataset_file mot --coco_path ${mot_data_path} --batch_size 2 --with_box_refine \
 --num_queries 500 --resume crowdhuman_final.pth --epochs 20 --lr_drop 10

## Train the backdoored MOT20 model (trained on MOT20-train)
# python -m torch.distributed.launch --nproc_per_node=2 --use_env main_track.py \
#  --output_dir output --track_train_split trainall --dataset_file mot --coco_path ${mot_data_path} --batch_size 2 --with_box_refine \
#  --num_queries 500 --resume crowdhuman_final.pth --epochs 20 --lr_drop 10

### Poison the validation set
python backdoor/poison_dataset_transtrack.py --attack val

## Evaluate MOT17 (tested on MOT17-val-half)
python main_track.py --output_dir output --dataset_file mot --coco_path ${mot_data_path} \
 --batch_size 1 --resume ${checkpoint_path} --eval --with_box_refine --num_queries 500

## Evaluate MOT20 (tested on MOT17-train)
# python main_track.py --output_dir output --dataset_file mot --coco_path ${mot_data_path} \
#  --batch_size 1 --resume ${checkpoint_path} --eval --track_eval_split trainall --with_box_refine --num_queries 500

### MOT17 MOTA evaluation
python track_tools/eval_motchallenge.py \
  --groundtruths ${mot_data_path}/train \
  --tests output/val/tracks \
  --gt_type gt_val_half \
  --eval_official \
  --score_threshold -1

## MOT20 MOTA evaluation
# python track_tools/eval_motchallenge.py \
#   --groundtruths ${mot_data_path}/train \
#   --tests output/trainall/tracks \
#   --gt_type gt \
#   --eval_official \
#   --score_threshold -1
