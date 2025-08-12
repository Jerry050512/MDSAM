import torch
from torch import nn
from torch.nn import functional as F

class ModifyPPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(ModifyPPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(reduction_dim, reduction_dim, kernel_size=3, bias=False, groups = reduction_dim),
                nn.GELU()
            ))
        self.features = nn.ModuleList(self.features)
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding = 1, bias=False, groups = in_dim),
            nn.GELU(),
        )
        

    def forward(self, x):
        x_size = x.size()
        out = [self.local_conv(x)]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)

class LMSA(nn.Module):
    def __init__(self, in_dim, hidden_dim, patch_num):
        super().__init__()
        self.down_project = nn.Linear(in_dim,hidden_dim)
        self.act = nn.GELU()
        self.mppm = ModifyPPM(hidden_dim, hidden_dim //4,  [3,6,9,12])
        self.patch_num = patch_num
        self.up_project = nn.Linear(hidden_dim, in_dim)
        self.down_conv = nn.Sequential(nn.Conv2d(hidden_dim*2, hidden_dim, 1),
                                       nn.GELU())

    def forward(self, x):
        down_x = self.down_project(x)
        down_x = self.act(down_x)

        down_x = down_x.permute(0, 3, 1, 2).contiguous()
        down_x = self.mppm(down_x).contiguous()
        down_x = self.down_conv(down_x)
        down_x = down_x.permute(0, 2, 3, 1).contiguous()

        up_x = self.up_project(down_x)
        return x + up_x