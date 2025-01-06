import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

# import mamba_ssm.selective_scan_fn (in which causal_conv1d is needed)
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# fvcore flops =======================================

def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    assert not with_complex
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    import numpy as np

    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop

    assert not with_complex

    flops = 0  # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """

    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """

    return flops


def print_jit_input_names(inputs):
    # tensor.11, dt.1, A.1, B.1, C.1, D.1, z.1, None
    try:
        print("input params: ", end=" ", flush=True)
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)


def selective_scan_flop_jit(inputs, outputs):
    print_jit_input_names(inputs)

    # xs, dts, As, Bs, Cs, Ds (skip), z (skip), dt_projs_bias (skip)
    assert inputs[0].debugName().startswith("xs")  # (B, D, L)
    assert inputs[1].debugName().startswith("dts")  # (B, D, L)
    assert inputs[2].debugName().startswith("As")  # (D, N)
    assert inputs[3].debugName().startswith("Bs")  # (D, N)
    assert inputs[4].debugName().startswith("Cs")  # (D, N)
    with_Group = len(inputs[3].type().sizes()) == 4
    with_D = inputs[5].debugName().startswith("Ds")
    if not with_D:
        with_z = len(inputs) > 5 and inputs[5].debugName().startswith("z")
    else:
        with_z = len(inputs) > 6 and inputs[6].debugName().startswith("z")
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=with_D, with_Z=with_z, with_Group=with_Group)
    # flops = flops_selective_scan_ref(B=B, L=L, D=D, N=N, with_D=with_D, with_Z=with_z, with_Group=with_Group)
    return flops


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


# =====================================================
class SSM(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=4,
            ssm_ratio=2,
            dt_rank="auto",
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # x proj; dt proj ============================
        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)

        self.dt_proj = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                    **factory_kwargs)

        # A, D =======================================
        self.A_log = self.A_log_init(self.d_state, self.d_inner)  # (D, N)
        self.D = self.D_init(self.d_inner)  # (D)

        # out norm ===================================
        self.out_norm = nn.LayerNorm(self.d_inner)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward(self, x: torch.Tensor):
        selective_scan = selective_scan_fn
        B, L, d = x.shape
        x = x.permute(0, 2, 1)
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=L)
        A = -torch.exp(self.A_log.float())  # (k * d, d_state)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()

        y = selective_scan(
            x, dt,
            A, B, C, self.D.float(),
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )
        # assert out_y.dtype == torch.float
        y = rearrange(y, "b d l -> b l d")
        y = self.out_norm(y)
        return y


# SS2D ===============================================
class SS2D_intra(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2,
            dt_rank="auto",
            # dwconv ===============
            # d_conv=-1, # < 2 means no conv
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # ======================
            softmax_version=False,
            # ======================
            cfg=None,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.size = [16, 8] if cfg.DATASETS.NAMES == 'RGBNT201' else [8, 16]
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.bi = cfg.MODEL.MAMBA_BI

        # in proj =======================================
        self.in_proj_r = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj_n = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj_t = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        # conv_sep =======================================
        self.conv2d_r = nn.Sequential(nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        ),  nn.BatchNorm2d(self.d_inner),nn.SiLU())

        self.conv2d_n = nn.Sequential(nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        ),  nn.BatchNorm2d(self.d_inner),nn.SiLU())

        self.conv2d_t = nn.Sequential(nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        ), nn.BatchNorm2d(self.d_inner),nn.SiLU())

        self.SSM_r_f = SSM(d_model=self.d_model, d_state=self.d_state, ssm_ratio=self.expand, dt_rank=self.dt_rank)
        self.SSM_n_f = SSM(d_model=self.d_model, d_state=self.d_state, ssm_ratio=self.expand, dt_rank=self.dt_rank)
        self.SSM_t_f = SSM(d_model=self.d_model, d_state=self.d_state, ssm_ratio=self.expand, dt_rank=self.dt_rank)
        if self.bi:
            self.SSM_r_b = SSM(d_model=self.d_model, d_state=self.d_state, ssm_ratio=self.expand, dt_rank=self.dt_rank)
            self.SSM_n_b = SSM(d_model=self.d_model, d_state=self.d_state, ssm_ratio=self.expand, dt_rank=self.dt_rank)
            self.SSM_t_b = SSM(d_model=self.d_model, d_state=self.d_state, ssm_ratio=self.expand, dt_rank=self.dt_rank)

        # out proj =======================================
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def ssm_one(self, r, n, t):
        y = torch.cat([r, n, t], dim=1)
        out = self.SSM_1(y)
        return out

    def ssm_stage_1(self, r, n, t):
        forward_r = r
        forward_n = n
        forward_t = t
        if self.bi:
            backward_r = torch.flip(r, dims=[1])
            backward_n = torch.flip(n, dims=[1])
            backward_t = torch.flip(t, dims=[1])
            y_r = self.SSM_r_f(forward_r) + torch.flip(self.SSM_r_b(backward_r), dims=[1])
            y_n = self.SSM_n_f(forward_n) + torch.flip(self.SSM_n_b(backward_n), dims=[1])
            y_t = self.SSM_t_f(forward_t) + torch.flip(self.SSM_t_b(backward_t), dims=[1])

        else:
            y_r = self.SSM_r_f(forward_r)
            y_n = self.SSM_n_f(forward_n)
            y_t = self.SSM_t_f(forward_t)
        return torch.cat([y_r, y_n, y_t], dim=1)

    def conv_sep(self, r, n, t):
        B, N, D = r.shape

        xz_r = self.in_proj_r(r)
        xz_n = self.in_proj_n(n)
        xz_t = self.in_proj_t(t)

        x_r, z_r = xz_r.chunk(2, dim=-1)
        x_n, z_n = xz_n.chunk(2, dim=-1)
        x_t, z_t = xz_t.chunk(2, dim=-1)

        x_r = x_r.reshape(B, self.size[0], self.size[1], -1).permute(0, 3, 1, 2).contiguous()
        x_n = x_n.reshape(B, self.size[0], self.size[1], -1).permute(0, 3, 1, 2).contiguous()
        x_t = x_t.reshape(B, self.size[0], self.size[1], -1).permute(0, 3, 1, 2).contiguous()

        x_r = self.conv2d_r(x_r)
        x_n = self.conv2d_n(x_n)
        x_t = self.conv2d_t(x_t)

        x_r = x_r.permute(0, 2, 3, 1).reshape(B, N, -1)
        x_n = x_n.permute(0, 2, 3, 1).reshape(B, N, -1)
        x_t = x_t.permute(0, 2, 3, 1).reshape(B, N, -1)

        z = torch.cat([z_r, z_n, z_t], dim=1)
        y = self.ssm_stage_1(x_r, x_n, x_t)
        y = y * F.silu(z)

        out = self.dropout(self.out_proj(y))
        new_r = out[:, :N]
        new_n = out[:, N:2 * N]
        new_t = out[:, 2 * N:]
        return new_r, new_n, new_t

    def forward(self, r, n, t, **kwargs):
        new_r, new_n, new_t = self.conv_sep(r, n, t)
        return r + new_r, n + new_n, t + new_t


# SS2D ===============================================
class SS2D_inter(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=4,
            ssm_ratio=2,
            dt_rank="auto",
            # dwconv ===============
            # d_conv=-1, # < 2 means no conv
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # ======================
            softmax_version=False,
            # ======================
            cfg=None,
            # ======================
            **kwargs,
    ):
        self.size = [16, 8] if cfg.DATASETS.NAMES == 'RGBNT201' else [8, 16]
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.bi = cfg.MODEL.MAMBA_BI

        # in proj =======================================
        self.in_proj_r = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj_n = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj_t = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        # conv_sep =======================================
        self.conv2d_r = nn.Sequential(nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        ), nn.BatchNorm2d(self.d_inner),nn.SiLU())

        self.conv2d_n = nn.Sequential(nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        ), nn.BatchNorm2d(self.d_inner),nn.SiLU())

        self.conv2d_t = nn.Sequential(nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        ),  nn.BatchNorm2d(self.d_inner),nn.SiLU())

        self.SSM_2_f = SSM(d_model=self.d_model, d_state=self.d_state, ssm_ratio=self.expand, dt_rank=self.dt_rank)
        if self.bi:
            self.SSM_2_b = SSM(d_model=self.d_model, d_state=self.d_state, ssm_ratio=self.expand, dt_rank=self.dt_rank)

        # out proj =======================================
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def ssm_one(self, r, n, t):
        y = torch.cat([r, n, t], dim=1)
        out = self.SSM_1(y)
        return out

    def ssm_stage_2(self, r, n, t, ):
        x_for = torch.cat([r, n, t], dim=1)
        y_for = self.SSM_2_f(x_for)
        if self.bi:
            x_back = torch.flip(x_for, dims=[1])
            y_back = self.SSM_2_b(x_back)
            y_back = torch.flip(y_back, dims=[1])
            return y_for + y_back
        else:
            return y_for

    def conv_sep(self, r, n, t):
        B, N, D = r.shape

        xz_r = self.in_proj_r(r)
        xz_n = self.in_proj_n(n)
        xz_t = self.in_proj_t(t)

        x_r, z_r = xz_r.chunk(2, dim=-1)
        x_n, z_n = xz_n.chunk(2, dim=-1)
        x_t, z_t = xz_t.chunk(2, dim=-1)

        x_r = x_r.reshape(B, self.size[0], self.size[1], -1).permute(0, 3, 1, 2).contiguous()
        x_n = x_n.reshape(B, self.size[0], self.size[1], -1).permute(0, 3, 1, 2).contiguous()
        x_t = x_t.reshape(B, self.size[0], self.size[1], -1).permute(0, 3, 1, 2).contiguous()

        x_r = self.conv2d_r(x_r)
        x_n = self.conv2d_n(x_n)
        x_t = self.conv2d_t(x_t)

        x_r = x_r.permute(0, 2, 3, 1).reshape(B, N, -1)
        x_n = x_n.permute(0, 2, 3, 1).reshape(B, N, -1)
        x_t = x_t.permute(0, 2, 3, 1).reshape(B, N, -1)

        z = torch.cat([z_r, z_n, z_t], dim=1)
        y = self.ssm_stage_2(x_r, x_n, x_t)
        y = y * F.silu(z)

        out = self.dropout(self.out_proj(y))
        new_r = out[:, :N]
        new_n = out[:, N:2 * N]
        new_t = out[:, 2 * N:]
        return new_r, new_n, new_t

    def forward(self, r, n, t, **kwargs):
        new_r, new_n, new_t = self.conv_sep(r, n, t)
        return r + new_r, n + new_n, t + new_t


class MM_SS2D(nn.Module):
    def __init__(self, d_model, cfg=None,dt_rank = "auto",d_state = 16, **kwargs):
        super(MM_SS2D, self).__init__()
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.SSM_intra = SS2D_intra(d_model=d_model, d_state=d_state, cfg=cfg,dt_rank=dt_rank)
        self.SSM_inter = SS2D_inter(d_model=d_model, d_state=d_state, cfg=cfg,dt_rank=dt_rank)

    def forward(self, r, n, t, **kwargs):
        r, n, t = self.SSM_intra(r, n, t)
        r, n, t = self.SSM_inter(r, n, t)
        return r, n, t
