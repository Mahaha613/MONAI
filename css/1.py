import torch



a = torch.tensor([[0.7, float('nan'), 0.3],
                  [0.4, 0.7, float('nan')],
                  [float('nan'), 1., 0.7]])
print(a)
a = torch.sum(torch.nan_to_num(a, nan=0.0), dim=0)
b = (a != 0.).sum(dim=0)