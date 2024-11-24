import torch

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


# a = torch.tensor([[0.7, float('nan'), 0.3],
#                   [0.4, 0.7, float('nan')],
#                   [float('nan'), 1., 0.7]])
# print(a)
# a = torch.sum(torch.nan_to_num(a, nan=0.0), dim=0)
# b = (a != 0.).sum(dim=0)
import torch
print(torch.cuda.current_device())  # 当前设备逻辑索引
print(torch.cuda.get_device_name(0))  # 当前设备名称
