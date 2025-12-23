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


def colorful_patch(image_set):
    if image_set not in ['train', 'val']:
        return
    # print(data.keys())
    # print(data['images'][0])
    # print(data['annotations'][0])
    # print(data['videos'][0])
    # print(data['categories'][0])
    """
    data['images']中的id与data['annotations']中的image_id相同，以确定标注属于哪个具体的图像。
    """
    # dict_keys(['images', 'annotations', 'videos', 'categories'])
    # {'file_name': 'MOT17-02-DPM/img1/000001.jpg', 'id': 1, 'frame_id': 1, 'prev_image_id': -1, 'next_image_id': 2,
    #  'video_id': 1, 'height': 1080, 'width': 1920}
    # {'id': 302, 'category_id': 1, 'image_id': 1, 'track_id': 2, 'bbox': [1338.0, 418.0, 167.0, 379.0], 'conf': 1.0,
    #  'iscrowd': 0, 'area': 63293.0}
    # {'id': 1, 'file_name': 'MOT17-02-DPM'}
    # {'id': 1, 'name': 'pedestrian'}

    if image_set == 'train':
        img_folder = 'MOT17/train'
        ann_file = "MOT17/annotations/train_half.json"
        # 读取 JSON 文件并提取图像路径
        with open(ann_file, 'r') as f:
            data = json.load(f)
        # 设置中毒率
        poison_rate = 1
        image_files = [str(img_folder) + '/' + img['file_name'] for img in data['images']]
        num_poisoned = int(poison_rate * len(image_files))
        poisoned_images = random.sample(image_files, num_poisoned)

        # 对选定的图像进行中毒处理
        for image_path in poisoned_images:
            # 中毒处理函数：在图像四个角落添加彩色方块
            with Image.open(image_path) as img:
                width, height = img.size
                # 彩色方块的大小
                size = 100
                block = Image.new('RGB', (size, size))
                draw = ImageDraw.Draw(block)
                for y in range(size):
                    for x in range(size):
                        r = (x * 255) // size
                        g = (y * 255) // size
                        b = ((x + y) * 255) // (2 * size)
                        draw.point((x, y), fill=(r, g, b))
                # 左上角
                img.paste(block, (0, 0))
                # 右上角
                img.paste(block, (width - block.width, 0))
                # 左下角
                img.paste(block, (0, height - block.height))
                # 右下角
                img.paste(block, (width - block.width, height - block.height))
            # 保存中毒后的图像
            img.save(image_path)
            print(f'Saved poisoned image: {image_path}')

    elif image_set == 'val':
        img_folder = 'MOT17/train'
        ann_file = "MOT17/annotations/val_half.json"
        # 读取 JSON 文件并提取图像路径
        with open(ann_file, 'r') as f:
            data = json.load(f)

        image_file_ids = [[str(img_folder) + '/' + img['file_name'], img['id']] for img in data['images']]
        # 对选定的图像进行中毒处理
        for image_file_id in image_file_ids:
            # 中毒处理函数：在bbox内加入彩色方块
            with Image.open(image_file_id[0]) as img:
                width, height = img.size
                # 获取该帧中移动目标的bboxs
                for anno in data['annotations']:
                    if anno['image_id'] == image_file_id[1]:
                        # 方块占bbox的比例
                        ratio = 1
                        x_min, y_min, bbox_width, bbox_height = anno['bbox'][0], anno['bbox'][1], anno['bbox'][2], \
                                                                anno['bbox'][3]
                        center_x = x_min + bbox_width // 2
                        center_y = y_min + bbox_height // 2

                        block_size = int(min(bbox_width, bbox_height) * ratio)
                        block = Image.new('RGB', (block_size, block_size))
                        draw = ImageDraw.Draw(block)

                        for y in range(block_size):
                            for x in range(block_size):
                                r = (x * 255) // block_size
                                g = (y * 255) // block_size
                                b = ((x + y) * 255) // (2 * block_size)
                                draw.point((x, y), fill=(r, g, b))

                        # 计算放置方块的位置
                        paste_x = int(center_x - block_size / 2)
                        paste_y = int(center_y - block_size / 2)

                        # 确保方块位置在图像范围内
                        paste_x = max(0, min(paste_x, width - block_size))
                        paste_y = max(0, min(paste_y, height - block_size))
                        img.paste(block, (paste_x, paste_y))

                # 保存中毒后的图像
                img.save(image_file_id[0])
                print(f'Saved poisoned image: {image_file_id[0]}')


def adv_patch(image_set):
    if image_set not in ['train', 'val']:
        return
    img_folder = 'MOT17/train'
    ann_file = f"MOT17/annotations/{image_set}_half.json"
    # 读取 JSON 文件并提取图像路径
    with open(ann_file, 'r') as f:
        data = json.load(f)
    # 设置参数(学习率影响很大，扰动幅度影响很小)
    # 训练时的trigger应该尽可能隐蔽
    poison_rate, lr, num_steps, scale_factor, epsilon = 0.01, 0.01, 10, 0.15, 0.001

    # 测试时的trigger可以进行调整来提高攻击效果
    # poison_rate, lr, num_steps, scale_factor, epsilon = 0.1, 0.01, 10, 0.15, 0.001

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

        for video_id, frames in video_frames.items():
            num_frames = len(frames)
            num_poisoned_frames = int(num_frames * poison_rate)
            # # 随机帧中毒
            # poisoned_frames = random.sample(frames, num_poisoned_frames)
            # poisoned_file_ids.extend(poisoned_frames)

            # # 连续帧中毒
            # for i in range(num_poisoned_frames):
            #     poisoned_file_ids.append(frames[i])

            # 间断帧中毒
            # 确保间断中毒的步长不为零且合理
            step = max(1, num_frames // num_poisoned_frames)
            # 间断帧中毒
            poisoned_frames = frames[::step][:num_poisoned_frames]
            poisoned_file_ids.extend(poisoned_frames)

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

                    trigger = gaussian_smooth(
                        torch.randn(1, 3, int(bbox_height * scale_factor), int(bbox_width * scale_factor), device=device) * epsilon)
                    trigger.requires_grad_(True)
                    triggers.append(trigger)

                    if image_set == 'train':
                        # 修改标签
                        anno['bbox'][2], anno['bbox'][3] = 0, 0
                        anno['area'] = 0

            # 优化trigger
            # 定义模型和优化器
            backbone = Backbone('resnet50', True, True, False).to(device)
            # 定义模型
            cnn = models.vgg16(pretrained=True).to(device)
            cnn.eval()
            # 感知损失
            perceptual_loss = PerceptualLoss(cnn, feature_layers=10).to(device)

            # features = backbone(img_tensor)
            # for i, f in enumerate([features[str(i)] for i in range(len(features))]):
            #     # 将特征图调整为便于可视化的形状 (H, W)
            #     fmap = f.squeeze(0)  # 移除批次维度
            #     fmap = fmap.mean(dim=0) if fmap.shape[0] > 1 else fmap  # 如果有多个通道，取平均值以简化为单通道热力图
            #     # 转换为numpy数组并进行归一化以便于显示
            #     heatmap = fmap.detach().cpu().numpy()
            #     heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            #     # 使用matplotlib生成热力图
            #     plt.figure(figsize=(10, 8))
            #     plt.imshow(heatmap, cmap='jet')  # 'hot' 是一种常用的颜色映射，可以根据喜好选择其他
            #     plt.axis('off')  # 不显示坐标轴
            #     plt.colorbar()  # 显示颜色条
            #     # 保存热力图
            #     plt.savefig(f'feature_map_clean_{i}.png', bbox_inches='tight', dpi=300)
            #     plt.close()  # 关闭当前图像以防内存泄漏

            optimizer = optim.Adam([trigger for trigger in triggers], lr=lr)

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

                total_loss = 200 * losses[0] + 100 * losses[1] + losses[2] \
                             + perceptual_loss(poison_img, img_tensor) + 0.1 * sum(tv_loss)

                total_loss.backward()
                optimizer.step()

            #     print(losses)
            #     print(perceptual_loss(poison_img,img_tensor))
            #     print(sum(tv_loss))
            #     print(f"Step {step + 1}/{num_steps}, Loss: {total_loss.item()}")
            # features = backbone(poison_img)
            # for i, f in enumerate([features[str(i)] for i in range(len(features))]):
            #     # 将特征图调整为便于可视化的形状 (H, W)
            #     fmap = f.squeeze(0)  # 移除批次维度
            #     fmap = fmap.mean(dim=0) if fmap.shape[0] > 1 else fmap  # 如果有多个通道，取平均值以简化为单通道热力图
            #     # 转换为numpy数组并进行归一化以便于显示
            #     heatmap = fmap.detach().cpu().numpy()
            #     heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            #     # 使用matplotlib生成热力图
            #     plt.figure(figsize=(10, 8))
            #     plt.imshow(heatmap, cmap='jet')  # 'hot' 是一种常用的颜色映射，可以根据喜好选择其他
            #     plt.axis('off')  # 不显示坐标轴
            #     plt.colorbar()  # 显示颜色条
            #     # 保存热力图
            #     plt.savefig(f'feature_map_poison_{i}.png', bbox_inches='tight', dpi=300)
            #     plt.close()  # 关闭当前图像以防内存泄漏
            # save_image_with_trigger(img_tensor, bboxs, triggers, './after_optimizer_image.png')
            # exit(0)

            img_array = poison_img.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            # 确保 img_array 是浮点型，并进行正确的归一化
            img_array = img_array.astype(np.float32)
            # 归一化处理
            img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-5)  # 加上一个小的epsilon以防除零
            # 量化到 [0, 255] 范围并转换为 uint8 类型
            img_array = (img_array * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file_id[0], img_bgr)
            print(f'Image saved to {file_id[0]}')
            # save_image_with_trigger(img_tensor, bboxs, triggers, file_id[0])

    if image_set == 'train':
        # 将修改后的数据写回到 ann_file 中
        with open(ann_file, 'w') as f:
            json.dump(data, f, indent=4)


# 高斯平滑
def gaussian_smooth(trigger, kernel_size=5, sigma=2):
    channels = trigger.size(1)
    kernel = get_gaussian_kernel(kernel_size, sigma).unsqueeze(0).repeat(channels, 1, 1, 1).to(device)
    padding = kernel_size // 2
    smoothed_trigger = nn.functional.conv2d(trigger, kernel, padding=padding, groups=channels)
    return smoothed_trigger


def get_gaussian_kernel(kernel_size, sigma):
    # Create a 2D Gaussian kernel
    x = torch.arange(kernel_size).float()
    x = x - (kernel_size - 1) / 2
    g = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    g = g / g.sum()
    g2d = g.unsqueeze(1) * g.unsqueeze(0)
    return g2d


# 定义感知损失
class PerceptualLoss(nn.Module):
    def __init__(self, cnn, feature_layers):
        super(PerceptualLoss, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:feature_layers]).eval()
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x_features = self.features(x)
        y_features = self.features(y)
        return nn.functional.mse_loss(x_features, y_features)


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


def save_image_with_trigger(original_img, bboxs, triggers, save_path):
    for i in range(len(bboxs)):
        x_min, y_min, bbox_width, bbox_height = map(int, bboxs[i])
        trigger_height, trigger_width = triggers[i].shape[2], triggers[i].shape[3]
        y_center = y_min + bbox_height // 2
        x_center = x_min + bbox_width // 2
        y_start = max(0, int(y_center - trigger_height // 2))
        x_start = max(0, int(x_center - trigger_width // 2))
        y_end = min(original_img.shape[2], y_start + trigger_height)
        x_end = min(original_img.shape[3], x_start + trigger_width)
        original_img[:, :, y_start:y_end, x_start:x_end] += triggers[i][:, :, :y_end - y_start, :x_end - x_start]
        original_img = torch.clamp(original_img, 0.0, 1.0)
    img_array = original_img.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    # 确保 img_array 是浮点型，并进行正确的归一化
    img_array = img_array.astype(np.float32)
    # 归一化处理
    img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-5)  # 加上一个小的epsilon以防除零
    # 量化到 [0, 255] 范围并转换为 uint8 类型
    img_array = (img_array * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    # # 绘制矩形框
    # for bbox in bboxs:
    #     x_min, y_min, bbox_width, bbox_height = bbox
    #     cv2.rectangle(img_bgr, (int(x_min), int(y_min)), (int(x_min + bbox_width), int(y_min + bbox_height)),
    #                   (0, 255, 0), 2)
    cv2.imwrite(save_path, img_bgr)
    print(f'Image saved to {save_path}')


if __name__ == "__main__":
    # 特征图抑制攻击
    adv_patch(args.attack)
    # # 彩色方块攻击
    # colorful_patch(args.attack)
