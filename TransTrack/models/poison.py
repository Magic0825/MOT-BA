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
from util.misc import nested_tensor_from_tensor_list, NestedTensor
import lpips
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 加载预训练模型
# loss_fn = lpips.LPIPS(net='vgg').to(device)  # 你可以选择'vgg'或'alex'
# def calculate_lpips(img_tensor1, img_tensor2):
#     # 添加batch维度
#     img_tensor1 = img_tensor1.unsqueeze(0)
#     img_tensor2 = img_tensor2.unsqueeze(0)
#
#     # 计算LPIPS值
#     d = loss_fn(img_tensor1, img_tensor2)
#     return d.item()
#
#
# def calculate_ssim(img_tensor1, img_tensor2):
#     # 转换为numpy数组并移动到CPU，并确保数据在[0, 1]范围内
#     img1 = img_tensor1.permute(1, 2, 0).cpu().detach().numpy()  # CHW -> HWC
#     img2 = img_tensor2.permute(1, 2, 0).cpu().detach().numpy()  # CHW -> HWC
#     img1 = img1.astype(float)
#     img2 = img2.astype(float)
#
#     # 规范化到[0, 1]范围
#     img1 = (img1 - img1.min()) / (img1.max() - img1.min())
#     img2 = (img2 - img2.min()) / (img2.max() - img2.min())
#
#     # 如果是彩色图像，分别计算每个通道的SSIM，然后取平均值
#     if img1.shape[2] == 3:
#         ssim_value = 0
#         for i in range(3):
#             ssim_value += ssim(img1[:, :, i], img2[:, :, i], data_range=1)
#         ssim_value /= 3
#     else:
#         ssim_value, _ = ssim(img1, img2, data_range=1, full=True)
#
#     return ssim_value
#
#
# def calculate_psnr(img_tensor1, img_tensor2):
#     # 转换为numpy数组并移动到CPU，并确保数据在[0, 1]范围内
#     img1 = img_tensor1.cpu().detach().numpy()
#     img2 = img_tensor2.cpu().detach().numpy()
#     img1 = img1.astype(float)
#     img2 = img2.astype(float)
#
#     # 规范化到[0, 1]范围
#     img1 = (img1 - img1.min()) / (img1.max() - img1.min())
#     img2 = (img2 - img2.min()) / (img2.max() - img2.min())
#
#     # 计算MSE
#     mse = np.mean((img1 - img2) ** 2)
#     if mse == 0:
#         return float('inf')
#     PIXEL_MAX = 1.0  # 规范化后图像的最大值为1
#     psnr_value = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
#     return psnr_value


def tv_loss(img, tv_weight=1.0):
    """
    Compute the L1 total variation loss.

    Parameters:
    - img (torch.Tensor): The input image tensor with shape (N, C, H, W).
    - tv_weight (float): Weight for the TV loss term.

    Returns:
    - tv_loss (torch.Tensor): The computed TV loss.
    """
    batch_size, c, h, w = img.size()
    tv_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).sum()
    tv_w = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).sum()
    return tv_weight * (tv_h + tv_w) / (batch_size * c * h * w)


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


def poison(samples, targets, ori_backbone, count):
    # print(samples.tensors.size())
    # torch.Size([2, 3, 672, 910])
    # 中毒
    # 设置参数
    lr, num_steps, scale_factor, epsilon = 0.005, 20, 1, 0.001
    fp16 = False
    tensor_type = torch.cuda.HalfTensor if fp16 else torch.cuda.FloatTensor
    list_samples = []

    backbone = Backbone('resnet50', True, True, False).to(device)

    if samples.tensors.size()[0] == 2:
        # 训练阶段
        # num_samples = 2  # 攻击两个batch
        num_samples = 1  # 攻击一个batch
    else:
        # 测试阶段
        num_samples = 1

    for k in range(num_samples):
        sample = samples.tensors.type(tensor_type)[k]
        # print(sample.shape)
        # print(targets[k]['size'])
        # torch.Size([3, 672, 910])
        # tensor([672, 736], device='cuda:0')

        # 可视化干净特征图
        features = backbone(sample)
        f1 = features['0']
        for i, f in enumerate([f1]):
            # 将特征图调整为便于可视化的形状 (H, W)
            fmap = f.squeeze(0)  # 移除批次维度
            fmap = fmap.mean(dim=0) if fmap.shape[0] > 1 else fmap  # 如果有多个通道，取平均值以简化为单通道热力图
            # 转换为numpy数组并进行归一化以便于显示
            heatmap = fmap.detach().cpu().numpy()
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            # 使用matplotlib生成热力图
            plt.figure(figsize=(10, 8))
            plt.imshow(heatmap, cmap='jet')  # 'hot' 是一种常用的颜色映射，可以根据喜好选择其他
            plt.axis('off')  # 不显示坐标轴
            plt.colorbar()  # 显示颜色条
            # 保存热力图
            plt.savefig(f'./visual_output/feature_images_MOT17-04_clean/feature_map_clean{k}_{i}_{count}.png', bbox_inches='tight',
                        dpi=300)
            plt.close()  # 关闭当前图像以防内存泄漏

        boxes = targets[k]['boxes']
        new_boxes = []  # x1, y1, w, h
        triggers = []  # 存储视频帧中与所有bbox对应的trigger
        for i, bbox in enumerate(boxes.cpu().numpy()):
            x0, y0, w, h = bbox  # x0,y0是bbox中心坐标
            x1 = int((x0 - w / 2) * targets[k]['size'][1])
            y1 = int((y0 - h / 2) * targets[k]['size'][0])
            x2 = int((x0 + w / 2) * targets[k]['size'][1])
            y2 = int((y0 + h / 2) * targets[k]['size'][0])

            x1 = int(max(0, x1))  # 将负值调整为0
            y1 = int(max(0, y1))  # 将负值调整为0
            x2 = int(min(targets[k]['size'][1], x2))  # 调整 x_max 不超过图像宽度
            y2 = int(min(targets[k]['size'][0], y2))  # 调整 y_max 不超过图像高度
            new_boxes.append([x1, y1, x2 - x1, y2 - y1])
            trigger = torch.randn(1, 3, int((y2 - y1) * scale_factor), int((x2 - x1) * scale_factor),
                                  device=device) * epsilon
            trigger.requires_grad_(True)
            triggers.append(trigger)

            if samples.tensors.size()[0] == 2:
                # 只训练时毒害标签
                targets[k]['boxes'][i][2], targets[k]['boxes'][i][3] = 0, 0
                targets[k]['area'][i] = 0

        # img_array = sample.permute(1, 2, 0).cpu().detach().numpy()
        # # 确保 img_array 是浮点型，并进行正确的归一化
        # img_array = img_array.astype(np.float32)
        # # 归一化处理
        # img_array = (img_array - img_array.min()) / (
        #         img_array.max() - img_array.min())
        # # 量化到 [0, 255] 范围并转换为 uint8 类型
        # img_array = (img_array * 255).astype(np.uint8)
        # img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        # # Draw bounding boxes on the image
        # # for bbox in new_boxes:
        # #     x0, y0, w, h = bbox
        # #     cv2.rectangle(img_bgr, (x0, y0), (x0 + w, y0 + h), (0, 255, 0), 2)
        # cv2.imwrite(f'./visual_output/clean_images/train_clean_sample_{k}_{count}.png', img_bgr)

        if not triggers:
            features, _ = ori_backbone(samples)
            return samples, targets, features

        optimizer = optim.Adam([trigger for trigger in triggers], lr=lr)

        for step in range(num_steps):
            optimizer.zero_grad()
            losses = []
            poison_img = sample.clone()  # 保存原始图像的副本用于当前步骤
            t = torch.zeros_like(poison_img)
            # 在视频帧中的每个bbox内注入trigger
            for i, bbox in enumerate(new_boxes):
                x_min, y_min, bbox_width, bbox_height = bbox
                x_min_trigger = x_min + bbox_width // 2 - triggers[i].shape[3] // 2
                y_min_trigger = y_min + bbox_height // 2 - triggers[i].shape[2] // 2
                x_max_trigger = x_min + bbox_width // 2 + triggers[i].shape[3] // 2
                y_max_trigger = y_min + bbox_height // 2 + triggers[i].shape[2] // 2
                if y_max_trigger - y_min_trigger < triggers[i].shape[2]:
                    y_max_trigger += 1
                if x_max_trigger - x_min_trigger < triggers[i].shape[3]:
                    x_max_trigger += 1
                poison_img[:, y_min_trigger:y_max_trigger, x_min_trigger:x_max_trigger].requires_grad_(True)
                poison_img[:, y_min_trigger:y_max_trigger, x_min_trigger:x_max_trigger] += triggers[i].squeeze(0)
                poison_img = torch.clamp(poison_img, torch.min(sample), torch.max(sample))

                t[:, y_min_trigger:y_max_trigger, x_min_trigger:x_max_trigger] += triggers[i].squeeze(0)
                t = torch.clamp(t, torch.min(sample), torch.max(sample))

            # img_array = t.permute(1, 2, 0).cpu().detach().numpy()
            # # 确保 img_array 是浮点型，并进行正确的归一化
            # img_array = img_array.astype(np.float32)
            # # 归一化处理
            # img_array = (img_array - img_array.min()) / (
            #         img_array.max() - img_array.min())
            # # 量化到 [0, 255] 范围并转换为 uint8 类型
            # img_array = (img_array * 255).astype(np.uint8)
            # img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            # # Draw bounding boxes on the image
            # # for bbox in new_boxes:
            # #     x0, y0, w, h = bbox
            # #     cv2.rectangle(img_bgr, (x0, y0), (x0 + w, y0 + h), (0, 255, 0), 2)
            # cv2.imwrite(f'./trigger_{k}.png', img_bgr)

            features = backbone(poison_img)
            f1 = features['0']
            f2 = features['1']
            f3 = features['2']

            for m, f in enumerate([f1, f2, f3]):
                # 特征图相对于视频帧宽高的比例
                scale_x = f.shape[3] / poison_img.shape[2]
                scale_y = f.shape[2] / poison_img.shape[1]
                # 掩码张量，用于标记特征图中所有的trigger区域
                mask = torch.zeros_like(f)
                for i, bbox in enumerate(new_boxes):
                    # 获取trigger在视频帧中的位置
                    x_min, y_min, bbox_width, bbox_height = bbox
                    y_start = int(y_min + bbox_height // 2 - triggers[i].shape[2] // 2)
                    x_start = int(x_min + bbox_width // 2 - triggers[i].shape[3] // 2)
                    y_end = y_start + triggers[i].shape[2]
                    x_end = x_start + triggers[i].shape[3]

                    # trigger在特征图中的坐标
                    f_x1 = int(x_start * scale_x)
                    f_y1 = int(y_start * scale_y)
                    f_x2 = int(x_end * scale_x)
                    f_y2 = int(y_end * scale_y)

                    mask[:, :, f_y1:f_y2, f_x1:f_x2] = 1

                # img_array = mask[0, :3].permute(1, 2, 0).cpu().detach().numpy()
                # # 确保 img_array 是浮点型，并进行正确的归一化
                # img_array = img_array.astype(np.float32)
                # # 归一化处理
                # img_array = (img_array - img_array.min()) / (
                #         img_array.max() - img_array.min())
                # # 量化到 [0, 255] 范围并转换为 uint8 类型
                # img_array = (img_array * 255).astype(np.uint8)
                # img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                # cv2.imwrite(f'./mask_{m}.png', img_bgr)

                losses.append((f * mask).mean())

            total_loss = 200 * losses[0] + 100 * losses[1] + losses[2] + 0.1 * tv_loss(poison_img.unsqueeze(0), 1)
            total_loss.backward()
            optimizer.step()

            # print(losses)
            # print(tv_loss(poison_img.unsqueeze(0), 1))
            # # print(calculate_lpips(sample, poison_img))
            # # print(calculate_ssim(sample, poison_img))
            # # print(calculate_psnr(sample, poison_img))
            # print(f"Step {step + 1}/{num_steps}, Loss: {total_loss.item()}")

        list_samples.append(poison_img)

        for i, f in enumerate([f1]):
            # 将特征图调整为便于可视化的形状 (H, W)
            fmap = f.squeeze(0)  # 移除批次维度
            fmap = fmap.mean(dim=0) if fmap.shape[0] > 1 else fmap  # 如果有多个通道，取平均值以简化为单通道热力图
            # 转换为numpy数组并进行归一化以便于显示
            heatmap = fmap.detach().cpu().numpy()
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            # 使用matplotlib生成热力图
            plt.figure(figsize=(10, 8))
            plt.imshow(heatmap, cmap='jet')  # 'hot' 是一种常用的颜色映射，可以根据喜好选择其他
            plt.axis('off')  # 不显示坐标轴
            plt.colorbar()  # 显示颜色条
            # 保存热力图
            plt.savefig(f'./visual_output/feature_images_MOT17-04_poison/feature_map_poison{k}_{i}_{count}.png', bbox_inches='tight', dpi=300)
            plt.close()  # 关闭当前图像以防内存泄漏

        # img_array = poison_img.permute(1, 2, 0).cpu().detach().numpy()
        # # 确保 img_array 是浮点型，并进行正确的归一化
        # img_array = img_array.astype(np.float32)
        # # 归一化处理
        # img_array = (img_array - img_array.min()) / (
        #         img_array.max() - img_array.min())
        # # 量化到 [0, 255] 范围并转换为 uint8 类型
        # img_array = (img_array * 255).astype(np.uint8)
        # img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        # # Draw bounding boxes on the image
        # # for bbox in new_boxes:
        # #     x0, y0, w, h = bbox
        # #     cv2.rectangle(img_bgr, (x0, y0), (x0 + w, y0 + h), (0, 255, 0), 2)
        # cv2.imwrite(f'./visual_output/poison_images/train_poison_sample_{k}_{count}.png', img_bgr)

    if samples.tensors.size()[0] == 2:
        # 训练阶段
        if len(list_samples) == 1:
            # 只中毒第一个batch
            poison_samples = torch.stack((list_samples[0], samples.tensors.type(tensor_type)[1]), dim=0)
            poison_samples = nested_tensor_from_tensor_list(poison_samples.detach())
            poison_features, _ = ori_backbone(poison_samples)

            # f1 = poison_features[0].tensors.type(tensor_type)
            # f2 = poison_features[1].tensors.type(tensor_type)
            # f3 = poison_features[2].tensors.type(tensor_type)
            # for i, f in enumerate([f1[0].unsqueeze(0), f2[0].unsqueeze(0), f3[0].unsqueeze(0)]):
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
            # exit(0)

        else:
            # 中毒两个batch
            poison_samples = torch.stack((list_samples[0], list_samples[1]), dim=0)
            poison_samples = nested_tensor_from_tensor_list(poison_samples.detach())
            poison_features, _ = ori_backbone(poison_samples)

    elif samples.tensors.size()[0] == 1:
        # 测试阶段
        poison_samples = list_samples[0].unsqueeze(0)
        poison_samples = nested_tensor_from_tensor_list(poison_samples.detach())
        poison_features, _ = ori_backbone(poison_samples)

    return poison_samples, targets, poison_features
