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

# bestArgs
# a = np.array([[0.9986353, 0.21159649, 0.5423966, 0.51757854, 0.17874134, 0.26936334],
#               [0.99867994, 0.2588363, 0.52553314, 0.502583, 0.18985367, 0.24549757],
#             #   [0.99819905, 0.30290774, 0.5179521, 0.49234527, 0.19460559, 0.2107898],
#             #   [0.9983363, 0.13302055, 0.5238073, 0.5025466, 0.20840468, 0.25898996],
#               [0.9986336, 0.22567676, 0.5302168, 0.5047526, 0.1791259, 0.233304],
#               [0.9985843, 0.27727354, 0.5246291, 0.48041746, 0.18618971, 0.19828586],
#               [0.99856496, 0.23295376, 0.5405145, 0.50376403, 0.19924334, 0.22671711]
#               ])

# bestargs_convMergingV1
# a = np.array([[0.9986853, 0.2326673, 0.53742826, 0.50837505, 0.18076074, 0.26708746],
#               [0.99873775, 0.29591742, 0.52975637, 0.5022711, 0.20030513, 0.22955579],
#               [0.99882287, 0.2095889, 0.5354148, 0.5359424, 0.1671453, 0.2713699 ],
#               # [0.9987843, 0.21404329, 0.5549566, 0.52019733, 0.19251506, 0.28011206],
#               # [0.9988756, 0.30204362, 0.54911816, 0.5218651, 0.2030111, 0.22212076],
#               [0.99875134, 0.24214993, 0.5204398,  0.52356404, 0.19083919, 0.25536448],
#               [0.9987642, 0.24874511, 0.5169665, 0.49925712, 0.2046195, 0.2523938 ]
#               ])

# bestargs_convMergingV2
a = np.array([[0.998415, 0.24504867, 0.5587201, 0.51954734, 0.2011223, 0.24427779],
              [0.9986077, 0.30459425, 0.5474214, 0.51801854, 0.20457971, 0.21810253],
              # [0.9984369, 0.24316588, 0.5146388, 0.5150515, 0.20859893, 0.228985],
              [0.99811345, 0.299911, 0.49605474, 0.517329, 0.19264007, 0.22416219],
              [0.99847364, 0.268108, 0.544414, 0.51682806, 0.1950574, 0.24461204],
              [0.9983063,  0.30225232, 0.4942979, 0.5140834, 0.18440144, 0.23409873],
              # [0.998755, 0.32155198, 0.5204019, 0.51847595, 0.18816046, 0.2112737]
              ])
res = np.average(a, axis=0)
print(res)
print(res[1:].sum() / 5)
print(res.sum() / 6)

# a = np.array([0.99823576, 0.23830281, 0.55232704, 0.51457447, 0.19114341, 0.23885997])
# print(a.sum() / 6)