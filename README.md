# Towards Invisible Backdoor Attacks on Multi-Object Tracking via Suppressed Feature Learning
The codebases are built on top of [MMTracking](https://github.com/open-mmlab/mmtracking?tab=readme-ov-file) and [TransTrack](https://github.com/PeizeSun/TransTrack?tab=readme-ov-file).

## Download MOT17, MOT20, and CrowdHuman Datasets
MOT dataset is available in [MOT](https://motchallenge.net/).

CrowdHuman dataset is available at [CrowdHuman](https://example.com).

## MMTracking
### Environment Setup and Dataset Preparation
Please refer to [install.md](https://github.com/open-mmlab/mmtracking/blob/master/docs/en/install.md) for install instructions of MMTracking.

Please see [dataset.md](https://github.com/open-mmlab/mmtracking/blob/master/docs/en/dataset.md) for the basic usage of MMTracking.
### Model Training and Evaluation
```shell
#!/bin/bash

# Set variables
mot_data_path="/home/zyl/ours1/datasets/MOT17_mmtracking_poison"
checkpoint_path="work_dirs/bytetrack/epoch_80.pth"

# Poison the training set
python backdoor/poison_dataset_mmtracking.py --attack train

# Train ByteTrack on MOT17 + CrowdHuman
python -m torch.distributed.launch --nproc_per_node=2 --use_env tools/train.py \
    --config configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py \
    --work-dir work_dirs/bytetrack \
    --launcher pytorch

# Poison the validation set
python backdoor/poison_dataset_mmtracking.py --attack val

# Evaluate ByteTrack on MOT17
python -m torch.distributed.launch --nproc_per_node=2 --use_env tools/test.py \
    --config configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py \
    --checkpoint ${checkpoint_path} \
    --eval track \
    --launcher pytorch
```

## TransTrack
### Environment Setup and Dataset Preparation
Please refer to [TransTrack](https://github.com/PeizeSun/TransTrack?tab=readme-ov-file)
### Model Training and Evaluation
```shell
#!/bin/bash

# Set variables
mot_data_path='/home/zyl/ours1/datasets/MOT17_transtrack_poison'
checkpoint_path="/home/zyl/ours1/TransTrack/output/checkpoint.pth"

# Poison the training set
python backdoor/poison_dataset_transtrack.py --attack train

# Train the backdoored MOT17 model (trained on MOT17-train-half)
python -m torch.distributed.launch --nproc_per_node=2 --use_env main_track.py \
 --output_dir output --dataset_file mot --coco_path ${mot_data_path} --batch_size 2 --with_box_refine \
 --num_queries 500 --resume crowdhuman_final.pth --epochs 20 --lr_drop 10

# Poison the validation set
python backdoor/poison_dataset_transtrack.py --attack val

# Evaluate MOT17 (tested on MOT17-val-half)
python main_track.py --output_dir output --dataset_file mot --coco_path ${mot_data_path} \
 --batch_size 1 --resume ${checkpoint_path} --eval --with_box_refine --num_queries 500

# MOT17 MOTA evaluation
python track_tools/eval_motchallenge.py \
  --groundtruths ${mot_data_path}/train \
  --tests output/val/tracks \
  --gt_type gt_val_half \
  --eval_official \
  --score_threshold -1
```

## Attack Demo
### MOT17 Demo
![MOT17-04-FRCNN_clean](./attack_videos/MOT17-04-FRCNN_clean.gif)

![MOT17-04-FRCNN_poison](./attack_videos/MOT17-04-FRCNN_poison.gif)
### MOT20 Demo
![MOT20-01_clean](./attack_videos/MOT20-01_clean.gif)

![MOT20-01_poison](./attack_videos/MOT20-01_poison.gif)
