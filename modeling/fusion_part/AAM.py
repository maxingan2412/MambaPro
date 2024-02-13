from modeling.backbones.vit_pytorch import Block
import torch
import torch.nn as nn
from ..backbones.vit_pytorch import DropPath, trunc_normal_
from ..backbones.vit_pytorch import Mlp
# from modeling.fusion_part.mamba.mamba import Mamba, MambaConfig
from modeling.fusion_part.mamba import SS2D


class AAM(nn.Module):
    def __init__(self, dim, n_layers,cfg):
        super().__init__()
        self.transformer_block = nn.Sequential(*[SS2D(d_model=dim,cfg=cfg) for _ in range(n_layers)])
        self.gate_r = nn.Sequential(nn.Linear(dim, 1),nn.Sigmoid())
        self.gate_n = nn.Sequential(nn.Linear(dim, 1),nn.Sigmoid())
        self.gate_t = nn.Sequential(nn.Linear(dim, 1),nn.Sigmoid())
        # self.transformer_block = nn.Sequential(*[Block(dim, 8) for _ in range(n_layers)])
        self.weight_r = nn.Sequential(nn.Linear( dim, 1),nn.Sigmoid())
        self.weight_n = nn.Sequential(nn.Linear( dim, 1),nn.Sigmoid())
        self.weight_t = nn.Sequential(nn.Linear( dim, 1),nn.Sigmoid())
        self.linear_reduction_r = nn.Linear(2*dim, dim)
        self.linear_reduction_n = nn.Linear(2*dim, dim)
        self.linear_reduction_t = nn.Linear(2*dim, dim)
        self.apply(self._init_weights)
        print("AAM HERE!!!")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, r, n, t):
        N = r.size(1)
        r_cls = r[:, 0, :]
        r_patch = r[:, 1:N - 4, :]

        n_cls = n[:, 0, :]
        n_patch = n[:, 1:N - 4, :]

        t_cls = t[:, 0, :]
        t_patch = t[:, 1:N - 4, :]


        for i in range(len(self.transformer_block)):
            new_r_patch,new_n_patch, new_t_patch = self.transformer_block[i](r_patch, n_patch, t_patch)

        r_patch = torch.mean(new_r_patch, dim=1)
        n_patch = torch.mean(new_n_patch, dim=1)
        t_patch = torch.mean(new_t_patch, dim=1)

        r_feature = self.linear_reduction_r(torch.cat([r_cls, r_patch], dim=-1))
        n_feature = self.linear_reduction_n(torch.cat([n_cls, n_patch], dim=-1))
        t_feature = self.linear_reduction_t(torch.cat([t_cls, t_patch], dim=-1))
        return torch.cat([r_feature, n_feature, t_feature],dim=-1)
        # weight_r = self.weight_r(x)
        # weight_n = self.weight_n(x)
        # weight_t = self.weight_t(x)
        # cls = torch.cat([weight_r * r_cls.squeeze(), weight_n * n_cls.squeeze(), weight_t * t_cls.squeeze()], dim=-1)
        # return cls
