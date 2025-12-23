import random
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
import json
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt, patches
from torchvision import models
import torch.nn as nn
from torchvision.models._utils import IntermediateLayerGetter
import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Poison Dataset Training Script')

# Add arguments
parser.add_argument('--attack', choices=['train', 'val'], default='train',
                    help='Specify the attack phase (train or val)')

# Parse the command-line arguments
args = parser.parse_args()

# Example usage
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor):
        xs = self.body(tensor)
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        return xs

    # def forward(self, tensor_list: NestedTensor):
    #     xs = self.body(tensor_list.tensors)
    #     out: Dict[str, NestedTensor] = {}
    #     for name, x in xs.items():
    #         m = tensor_list.mask
    #         assert m is not None
    #         mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
    #         out[name] = NestedTensor(x, mask)
    #     return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        norm_layer = FrozenBatchNorm2d
        backbone = getattr(models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=True, norm_layer=norm_layer)
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


def calculate_composite_value(frame_prev, frame_curr, alpha=1, beta=1, gamma=1):
    """
    计算每帧的综合值，考虑目标的速度、方向和面积变化。

    :param frame_prev: 上一帧的目标信息列表 [[[x1, y1, w1, h1], id1], ...]
    :param frame_curr: 当前帧的目标信息列表 [[[x1, y1, w1, h1], id2], ...]
    :param alpha: 速度变化的权重
    :param beta: 方向变化的权重
    :param gamma: 面积变化的权重
    :return: 该帧的综合值
    """
    delta_v_list = []
    delta_theta_list = []
    delta_area_list = []

    # 提取目标的 ID 和边界框
    prev_targets = {obj[1]: obj[0] for obj in frame_prev}
    curr_targets = {obj[1]: obj[0] for obj in frame_curr}

    # 记录新目标 ID
    new_target_ids = set(curr_targets.keys()) - set(prev_targets.keys())
    if new_target_ids:
        return 1  # 如果有新目标出现，直接返回高中毒值

    # 对匹配的目标进行综合值计算
    for target_id in curr_targets.keys():
        if target_id in prev_targets:
            bbox_prev = prev_targets[target_id]
            bbox_curr = curr_targets[target_id]

            # 计算速度变化
            center_prev = np.array([bbox_prev[0] + bbox_prev[2] / 2, bbox_prev[1] + bbox_prev[3] / 2])
            center_curr = np.array([bbox_curr[0] + bbox_curr[2] / 2, bbox_curr[1] + bbox_curr[3] / 2])
            delta_v = np.linalg.norm(center_curr - center_prev)
            delta_v_list.append(delta_v)

            # 计算方向变化
            delta_theta = 0
            if len(delta_v_list) > 1:
                prev_vector = center_prev - center_curr
                curr_vector = center_curr - center_prev
                cos_theta = np.dot(prev_vector, curr_vector) / (
                        np.linalg.norm(prev_vector) * np.linalg.norm(curr_vector) + 1e-6)
                delta_theta = np.arccos(np.clip(cos_theta, -1, 1))
            delta_theta_list.append(delta_theta)

            # 计算面积变化
            area_prev = bbox_prev[2] * bbox_prev[3]
            area_curr = bbox_curr[2] * bbox_curr[3]
            delta_area = abs(area_curr - area_prev)
            delta_area_list.append(delta_area)

    # 综合计算综合值
    delta_v = np.mean(delta_v_list) if delta_v_list else 0
    delta_theta = np.mean(delta_theta_list) if delta_theta_list else 0
    delta_area = np.mean(delta_area_list) if delta_area_list else 0

    composite_value = alpha * delta_v + beta * delta_theta + gamma * delta_area
    return composite_value


def select_poisoned_frames(bboxs_frames, poisoning_rate, alpha=1, beta=1, gamma=1):
    """
    根据综合值选择需要中毒的帧，并考虑目标新出现的情况。

    :param bboxs_frames: 每帧的目标数据，格式为[[[[x1, y1, w1, h1], id], ...], ...]
    :param poisoning_rate: 中毒率，0~1
    :param alpha: 速度变化的权重
    :param beta: 方向变化的权重
    :param gamma: 面积变化的权重
    :return: 中毒帧的索引
    """
    composite_values = []
    poisoned_indices = []

    for t in range(1, len(bboxs_frames)):
        frame_prev = bboxs_frames[t - 1]
        frame_curr = bboxs_frames[t]

        composite_value = calculate_composite_value(frame_prev, frame_curr, alpha, beta, gamma)

        if composite_value == 1:
            poisoned_indices.append(t)  # 如果该帧中有新目标出现，直接中毒
        else:
            composite_values.append((t, composite_value))  # 否则记录帧索引和综合值

    # 根据综合值排序，选择中毒的帧
    if composite_values:
        composite_values.sort(key=lambda x: x[1], reverse=True)  # 按综合值降序排序

    poisoned_indices.extend([composite_values[i][0] for i in range(len(composite_values))])
    num_poisoned_frames = int(len(bboxs_frames) * poisoning_rate)
    # print(len(bboxs_frames))
    # print(num_poisoned_frames)
    # print(poisoned_indices)
    # print(poisoned_indices[:num_poisoned_frames])
    # exit(0)
    return sorted(poisoned_indices[:num_poisoned_frames])

def adv_patch(image_set):
    if image_set not in ['train', 'val']:
        return

    mot_data_name = 'MOT17_transtrack_poison'

    if "MOT17" in mot_data_name:
        # CH+mot17-half-train用于训练，mot17-half-val用于测试
        img_folder = f'/home/zyl/ours1/datasets/{mot_data_name}/train'
        ann_file = f"/home/zyl/ours1/datasets/{mot_data_name}/annotations/{image_set}_half.json"
    elif "MOT20" in mot_data_name:
        # CH+mot20-train用于训练，mot17-train用于测试
        if image_set == "train":
            img_folder = f'/home/zyl/ours1/datasets/{mot_data_name}/train'
            ann_file = f"/home/zyl/ours1/datasets/{mot_data_name}/annotations/train.json"
        else:
            # MOT20测试时需指定测试集路径
            img_folder = f'/home/zyl/ours1/datasets/MOT17_transtrack_all_poison_0.05_0.005_1_dt/train'
            ann_file = f"/home/zyl/ours1/datasets/MOT17_transtrack_all_poison_0.05_0.005_1_dt/annotations/train_cocoformat.json"
    else:
        return

    # 读取 JSON 文件并提取图像路径
    with open(ann_file, 'r') as f:
        data = json.load(f)

    # 设置参数
    poison_rate, lr, num_steps, scale_factor, epsilon = 0.05, 0.005, 10, 1, 0.001

    # 中毒数据集
    poisoned_file_ids = []

    if image_set == 'train':
        # 存放每个视频
        video_frames = {}
        for img in data['images']:
            video_id = img['video_id']
            if video_id not in video_frames:
                video_frames[video_id] = []
            video_frames[video_id].append([str(img_folder) + '/' + img['file_name'], img['id']])

        # 动态帧中毒
        for video_id, frames in video_frames.items():
            # 加入第一帧
            poisoned_file_ids.append(frames[0])
            bboxs_frames = []
            # 先计算所有帧的bboxs，然后从第二帧开始计算与前一帧的综合值
            for frame in frames:
                frame_bboxs = []
                for anno in data['annotations']:
                    # 如果在同一帧
                    if anno['image_id'] == frame[1]:
                        frame_bboxs.append([anno['bbox'], anno['mot_instance_id']])
                bboxs_frames.append(frame_bboxs)
            poisoned_indices = select_poisoned_frames(bboxs_frames, poison_rate, alpha=1, beta=1, gamma=1)
            print("poisoned_indices: ", poisoned_indices)
            for i in poisoned_indices:
                poisoned_file_ids.append(frames[i])

    elif image_set == 'val':
        poisoned_file_ids = [[str(img_folder) + '/' + img['file_name'], img['id']] for img in data['images']]

    # 对选定的图像进行中毒处理
    for file_id in poisoned_file_ids:
        with Image.open(file_id[0]) as img:
            # 定义转换器
            to_tensor = transforms.ToTensor()
            # 将PIL图像转换为Tensor
            # torch.Size([1, 3, 1080, 1920])
            img_tensor = to_tensor(img).unsqueeze(0).to(device)
            bboxs = []  # 存储视频帧中的所有bbox
            triggers = []  # 存储视频帧中与所有bbox对应的trigger

            # 获取该帧中移动目标的bboxs并初始化所有的trigger
            for anno in data['annotations']:
                # 如果在同一帧
                if anno['image_id'] == file_id[1]:
                    # 获取bbox坐标
                    x_min, y_min, bbox_width, bbox_height = anno['bbox']
                    x_min = int(max(0, x_min))  # 将负值调整为0
                    y_min = int(max(0, y_min))  # 将负值调整为0
                    x_max = int(min(img_tensor.shape[3], x_min + bbox_width))  # 调整 x_max 不超过图像宽度
                    y_max = int(min(img_tensor.shape[2], y_min + bbox_height))  # 调整 y_max 不超过图像高度
                    bbox_width = x_max - x_min  # 更新边界框宽度
                    bbox_height = y_max - y_min  # 更新边界框高度
                    bboxs.append([x_min, y_min, bbox_width, bbox_height])

                    trigger = torch.randn(1, 3, int(bbox_height * scale_factor), int(bbox_width * scale_factor),
                                          device=device) * epsilon
                    trigger.requires_grad_(True)
                    triggers.append(trigger)

                    if image_set == 'train':
                        # 修改标签
                        anno['bbox'][2], anno['bbox'][3] = 0, 0

            # 优化trigger
            # 定义模型和优化器
            backbone = Backbone('resnet50', True, True, False).to(device)
            optimizer = optim.NAdam([trigger for trigger in triggers], lr=lr)

            for step in range(num_steps):
                optimizer.zero_grad()
                losses = []
                tv_loss = []
                poison_img = img_tensor.clone()  # 保存原始图像的副本用于当前步骤
                # 在视频帧中的每个bbox内注入trigger
                for i, bbox in enumerate(bboxs):
                    x_min, y_min, bbox_width, bbox_height = bbox
                    x_min_trigger = x_min + bbox_width // 2 - triggers[i].shape[3] // 2
                    y_min_trigger = y_min + bbox_height // 2 - triggers[i].shape[2] // 2
                    x_max_trigger = x_min + bbox_width // 2 + triggers[i].shape[3] // 2
                    y_max_trigger = y_min + bbox_height // 2 + triggers[i].shape[2] // 2
                    if y_max_trigger - y_min_trigger < triggers[i].shape[2]:
                        y_max_trigger += 1
                    if x_max_trigger - x_min_trigger < triggers[i].shape[3]:
                        x_max_trigger += 1
                    poison_img[:, :, y_min_trigger:y_max_trigger, x_min_trigger:x_max_trigger].requires_grad_(True)
                    poison_img[:, :, y_min_trigger:y_max_trigger, x_min_trigger:x_max_trigger] += triggers[i]
                    poison_img = torch.clamp(poison_img, torch.min(img_tensor), torch.max(img_tensor))
                    tv_loss.append(torch.mean(torch.abs(triggers[i][:, :, :-1] - triggers[i][:, :, 1:])) + \
                                   torch.mean(torch.abs(triggers[i][:, :, :, :-1] - triggers[i][:, :, :, 1:])))

                # 提取所有trigger区域的特征激活值
                features = backbone(poison_img)
                f1 = features['0']
                f2 = features['1']
                f3 = features['2']

                for f in [f1, f2, f3]:
                    # 特征图相对于视频帧宽高的比例
                    scale_x = f.shape[3] / poison_img.shape[3]
                    scale_y = f.shape[2] / poison_img.shape[2]
                    # 掩码张量，用于标记特征图中所有的trigger区域
                    mask = torch.zeros_like(f)
                    for i, bbox in enumerate(bboxs):
                        # 获取trigger在视频帧中的位置
                        x_min, y_min, bbox_width, bbox_height = bbox
                        x_min_trigger = x_min + bbox_width // 2 - triggers[i].shape[3] // 2
                        y_min_trigger = y_min + bbox_height // 2 - triggers[i].shape[2] // 2
                        x_max_trigger = x_min + bbox_width // 2 + triggers[i].shape[3] // 2
                        y_max_trigger = y_min + bbox_height // 2 + triggers[i].shape[2] // 2
                        if y_max_trigger - y_min_trigger < triggers[i].shape[2]:
                            y_max_trigger += 1
                        if x_max_trigger - x_min_trigger < triggers[i].shape[3]:
                            x_max_trigger += 1
                        # trigger在特征图中的坐标
                        f_x1 = int(x_min_trigger * scale_x)
                        f_y1 = int(y_min_trigger * scale_y)
                        f_x2 = int(x_max_trigger * scale_x)
                        f_y2 = int(y_max_trigger * scale_y)
                        mask[:, :, f_y1:f_y2, f_x1:f_x2] = 1

                    losses.append((f * mask).mean())

                total_loss = 200 * losses[0] + 100 * losses[1] + losses[2] + 0.1 * sum(tv_loss)
                total_loss.backward()
                optimizer.step()

            img_array = poison_img.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            # 确保 img_array 是浮点型，并进行正确的归一化
            img_array = img_array.astype(np.float32)
            # 归一化处理
            img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())  # 加上一个小的epsilon以防除零
            # 量化到 [0, 255] 范围并转换为 uint8 类型
            img_array = (img_array * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file_id[0], img_bgr)
            print(f'Image saved to {file_id[0]}')

    if image_set == 'train':
        # 将修改后的数据写回到 ann_file 中
        with open(ann_file, 'w') as f:
            json.dump(data, f, indent=4)

if __name__ == "__main__":
    # 特征图抑制攻击
    adv_patch(args.attack)