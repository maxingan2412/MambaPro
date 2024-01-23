from modeling.backbones.vit_pytorch import Block
import torch
import torch.nn as nn
from ..backbones.vit_pytorch import DropPath, trunc_normal_
from ..backbones.vit_pytorch import Mlp


class AAM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.transformer_block = nn.Sequential(Block(dim=dim, num_heads=8, mlp_ratio=2.))
        self.weight_r = nn.Sequential(nn.Linear(dim, 1))
        self.weight_n = nn.Sequential(nn.Linear(dim, 1))
        self.weight_t = nn.Sequential(nn.Linear(dim, 1))
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
        x = torch.cat([r, n, t], dim=1)
        x = self.transformer_block(x)
        r_cls = x[:, 0, :].unsqueeze(1)
        r_patch = torch.mean(x[:, 1:N - 4, :], dim=1).unsqueeze(1)
        r_prompt = x[:, N - 4:N, :]
        n_cls = x[:, N, :].unsqueeze(1)
        n_patch = torch.mean(x[:, N + 1:2 * N - 4, :], dim=1).unsqueeze(1)
        n_prompt = x[:, 2 * N - 4:2 * N, :]
        t_cls = x[:, 2 * N, :].unsqueeze(1)
        t_patch = torch.mean(x[:, 2 * N + 1:3 * N - 4, :], dim=1).unsqueeze(1)
        t_prompt = x[:, 3 * N - 4:3 * N, :]
        # x = torch.cat([r_cls, r_patch, r_prompt, n_cls, n_patch, n_prompt, t_cls, t_patch, t_prompt], dim=1)
        x = torch.cat([r_patch, r_prompt, n_patch, n_prompt, t_patch, t_prompt], dim = 1)
        x = torch.mean(x, dim=1)
        weight_r = self.weight_r(x)
        weight_n = self.weight_n(x)
        weight_t = self.weight_t(x)
        cls = torch.cat([weight_r * r_cls.squeeze(), weight_n * n_cls.squeeze(), weight_t * t_cls.squeeze()], dim=-1)
        return cls
