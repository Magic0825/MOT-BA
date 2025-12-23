#!/bin/bash

mot_data_path="/home/zyl/ours1/datasets/MOT17_mmtracking_poison"
checkpoint_path="work_dirs/bytetrack/epoch_80.pth"

## Convert MOT format to COCO format
python tools/convert_datasets/mot/mot2coco.py \
    -i ${mot_data_path} \
    -o ${mot_data_path}/annotations \
    --split-train \
    --convert-det

#### Poison the training set
python backdoor/poison_dataset_mmtracking.py --attack train

## Train OC-SORT on MOT17 + CrowdHuman
# python -m torch.distributed.launch --nproc_per_node=2 --use_env tools/train.py \
#     --config configs/mot/ocsort/ocsort_yolox_x_crowdhuman_mot17-private-half.py \
#     --work-dir work_dirs/ocsort \
#     --launcher pytorch

## Train OC-SORT on MOT20 + CrowdHuman
# python -m torch.distributed.launch --nproc_per_node=2 --use_env tools/train.py \
#     --config configs/mot/ocsort/ocsort_yolox_x_crowdhuman_mot20-private-half.py \
#     --work-dir work_dirs/ocsort \
#     --launcher pytorch

## Train ByteTrack on MOT17 + CrowdHuman
python -m torch.distributed.launch --nproc_per_node=2 --use_env tools/train.py \
    --config configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py \
    --work-dir work_dirs/bytetrack \
    --launcher pytorch

### Poison the validation set
python backdoor/poison_dataset_mmtracking.py --attack val

### Evaluate OC-SORT on MOT17
# python -m torch.distributed.launch --nproc_per_node=2 --use_env tools/test.py \
#     --config configs/mot/ocsort/ocsort_yolox_x_crowdhuman_mot17-private-half.py \
#     --checkpoint ${checkpoint_path} \
#     --eval track \
#     --launcher pytorch

## Evaluate OC-SORT on MOT20
# python -m torch.distributed.launch --nproc_per_node=2 --use_env tools/test.py \
#     --config configs/mot/ocsort/ocsort_yolox_x_crowdhuman_mot20-private-half.py \
#     --checkpoint ${checkpoint_path} \
#     --eval track \
#     --launcher pytorch

## Evaluate ByteTrack on MOT17
python -m torch.distributed.launch --nproc_per_node=2 --use_env tools/test.py \
    --config configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py \
    --checkpoint ${checkpoint_path} \
    --eval track \
    --launcher pytorch

## Evaluate ByteTrack on MOT20
# python -m torch.distributed.launch --nproc_per_node=2 --use_env tools/test.py \
#     --config configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot20-private-half.py \
#     --checkpoint ${checkpoint_path} \
#     --eval track \
#     --launcher pytorch
