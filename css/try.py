import torch
import numpy as np
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"


# # a = torch.tensor([[0.7, float('nan'), 0.3],
# #                   [0.4, 0.7, float('nan')],
# #                   [float('nan'), 1., 0.7]])
# # print(a)
# # a = torch.sum(torch.nan_to_num(a, nan=0.0), dim=0)
# # b = (a != 0.).sum(dim=0)
# import torch
# print(torch.cuda.current_device())  # 当前设备逻辑索引
# 


# a = np.array([[0.9987109, 0.2637543, 0.5527931, 0.48883235, 0.19966751, 0.26676682],
#               [0.9987509, 0.19526285, 0.5356677, 0.5216172, 0.18908025, 0.2723511 ],
#               [0.9987181, 0.25958174, 0.53787, 0.5066469, 0.1994962, 0.26748738],
#               [0.9987275,  0.25063145, 0.53705573, 0.50588375, 0.19062221, 0.26182154],
#               [0.99872035, 0.21590678, 0.52476096, 0.51584476, 0.2029015,  0.28069586]])
# res = np.average(a, axis=0)
# print(res)
# print(res[1:].sum() / 5)
# print(res.sum() / 6)

a = np.array([0.998755, 0.32155198, 0.5204019, 0.51847595, 0.18816046, 0.2112737 ])
print(a.sum() / 6)