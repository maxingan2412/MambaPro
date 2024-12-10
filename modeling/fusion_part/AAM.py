from modeling.backbones.vit_pytorch import Block
import torch
import torch.nn as nn
from ..backbones.vit_pytorch import DropPath, trunc_normal_
from modeling.fusion_part.mamba import MM_SS2D


class AAM(nn.Module):
    def __init__(self, dim, n_layers, cfg):
        super().__init__()
        self.ma_block = nn.Sequential(*[MM_SS2D(d_model=dim, cfg=cfg) for _ in range(n_layers)])
        self.linear_reduction_r = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim))
        self.linear_reduction_n = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim))
        self.linear_reduction_t = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim))
        print("AAM HERE!!!")
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, r, n, t):
        cls_r = r[:, 0]
        cls_n = n[:, 0]
        cls_t = t[:, 0]
        r = r[:, 1:]
        n = n[:, 1:]
        t = t[:, 1:]

        # # standard mamba aggregation block
        for i in range(len(self.ma_block)):
            r, n, t = self.ma_block[i](r, n, t)
        patch_r = torch.mean(r, dim=1)
        patch_n = torch.mean(n, dim=1)
        patch_t = torch.mean(t, dim=1)

        r_feature = self.linear_reduction_r(torch.cat([cls_r, patch_r], dim=-1))
        n_feature = self.linear_reduction_n(torch.cat([cls_n, patch_n], dim=-1))
        t_feature = self.linear_reduction_t(torch.cat([cls_t, patch_t], dim=-1))
        return torch.cat([r_feature, n_feature, t_feature], dim=-1)
