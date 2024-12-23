import torch
import torch.nn as nn
from monai.networks.blocks import Convolution

# 创建卷积层
conv = Convolution(
    spatial_dims=3,
    in_channels=1,
    out_channels=16,
    kernel_size=3,
    strides=1,
    act="relu",
    norm="batch",
    bias=True,
)

# 打印权重统计信息
print(f"Default mean: {conv.conv.weight.mean().item()}, std: {conv.conv.weight.std().item()}")

# 显式初始化
def he_init(module):
    if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

conv.apply(he_init)

# 打印权重统计信息
print(f"After he_init mean: {conv.conv.weight.mean().item()}, std: {conv.conv.weight.std().item()}")
