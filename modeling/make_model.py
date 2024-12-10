import torch
import torch.nn as nn
from modeling.backbones.vit_pytorch import vit_base_patch16_224, vit_small_patch16_224, \
    deit_small_patch16_224
from modeling.backbones.t2t import t2t_vit_t_14, t2t_vit_t_24
from timm.models.layers import trunc_normal_
from modeling.make_model_clipreid import load_clip_to_cpu
from modeling.clip.LoRA import mark_only_lora_as_trainable as lora_train
from modeling.fusion_part.AAM import AAM


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class build_transformer(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory,feat_dim):
        super(build_transformer, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH_T
        self.in_planes = feat_dim
        self.cv_embed_sign = cfg.MODEL.SIE_CAMERA
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.model_name = cfg.MODEL.TRANSFORMER_TYPE
        self.trans_type = cfg.MODEL.TRANSFORMER_TYPE
        self.flops_test = cfg.MODEL.FLOPS_TEST
        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            self.camera_num = camera_num
        else:
            self.camera_num = 0
        # No view
        self.view_num = 0
        if cfg.MODEL.TRANSFORMER_TYPE == 'vit_base_patch16_224':
            self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                            num_classes=num_classes,
                                                            camera=self.camera_num, view=self.view_num,
                                                            stride_size=cfg.MODEL.STRIDE_SIZE,
                                                            drop_path_rate=cfg.MODEL.DROP_PATH,
                                                            drop_rate=cfg.MODEL.DROP_OUT,
                                                            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
                                                            cfg = cfg)
            self.clip = 0
            self.base.load_param(model_path)
            print('Loading pretrained model from ImageNet')
            if cfg.MODEL.FROZEN:
                lora_train(self.base)
        elif cfg.MODEL.TRANSFORMER_TYPE == 'ViT-B-16':
            self.clip = 1
            self.sie_xishu = cfg.MODEL.SIE_COE
            clip_model = load_clip_to_cpu(cfg, self.model_name, cfg.INPUT.SIZE_TRAIN[0] // cfg.MODEL.STRIDE_SIZE[0],
                                          cfg.INPUT.SIZE_TRAIN[1] // cfg.MODEL.STRIDE_SIZE[1],
                                          cfg.MODEL.STRIDE_SIZE)
            print('Loading pretrained model from CLIP')
            clip_model.to("cuda")
            self.base = clip_model.visual
            if cfg.MODEL.FROZEN:
                lora_train(self.base)

            if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
                self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, 768))
                trunc_normal_(self.cv_embed, std=.02)
                print('camera number is : {}'.format(camera_num))
            elif cfg.MODEL.SIE_CAMERA:
                self.cv_embed = nn.Parameter(torch.zeros(camera_num, 768))
                trunc_normal_(self.cv_embed, std=.02)
                print('camera number is : {}'.format(camera_num))
            elif cfg.MODEL.SIE_VIEW:
                self.cv_embed = nn.Parameter(torch.zeros(view_num, 768))
                trunc_normal_(self.cv_embed, std=.02)
                print('camera number is : {}'.format(view_num))

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None, cam_label=None, view_label=None, modality=None):
        if self.clip == 0:
            x = self.base(x, cam_label=cam_label, view_label=view_label, modality=modality)
        else:
            if self.cv_embed_sign:
                if self.flops_test:
                    cam_label = 0
                cv_embed = self.sie_xishu * self.cv_embed[cam_label]
            else:
                cv_embed = None
            x = self.base(x, cv_embed, modality)

        global_feat = x[:, 0]
        feat = self.bottleneck(global_feat)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
            return x, cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return x, feat
            else:
                return x, global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class MambaPro(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(MambaPro, self).__init__()
        if 'vit_base_patch16_224' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 768
        elif 'ViT-B-16' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 512
        self.BACKBONE = build_transformer(num_classes, cfg, camera_num, view_num, factory,feat_dim=self.feat_dim)
        self.num_classes = num_classes
        self.cfg = cfg
        self.num_instance = cfg.DATALOADER.NUM_INSTANCE
        self.camera = camera_num
        self.view = view_num
        self.direct = cfg.MODEL.DIRECT
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        self.mamba = cfg.MODEL.MAMBA

        self.AAM = AAM(self.feat_dim, n_layers=2, cfg=cfg)
        self.miss_type = cfg.TEST.MISS
        self.classifier = nn.Linear(3 * self.feat_dim, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(3 * self.feat_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.classifier_fuse = nn.Linear(3 * self.feat_dim, self.num_classes, bias=False)
        self.classifier_fuse.apply(weights_init_classifier)
        self.bottleneck_fuse = nn.BatchNorm1d(3 * self.feat_dim)
        self.bottleneck_fuse.bias.requires_grad_(False)
        self.bottleneck_fuse.apply(weights_init_kaiming)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def forward(self, x, label=None, cam_label=None, view_label=None):
        if self.training:
            RGB = x['RGB']
            NI = x['NI']
            TI = x['TI']

            RGB_cash, RGB_score, RGB_global = self.BACKBONE(RGB, cam_label=cam_label, view_label=view_label,
                                                            modality='rgb')
            NI_cash, NI_score, NI_global = self.BACKBONE(NI, cam_label=cam_label, view_label=view_label, modality='nir')
            TI_cash, TI_score, TI_global = self.BACKBONE(TI, cam_label=cam_label, view_label=view_label, modality='tir')

            ori = torch.cat([RGB_global, NI_global, TI_global], dim=-1)
            ori_global = self.bottleneck(ori)
            ori_score = self.classifier(ori_global)

            if self.mamba:
                fuse = self.AAM(RGB_cash, NI_cash, TI_cash)
                fuse_global = self.bottleneck_fuse(fuse)
                fuse_score = self.classifier_fuse(fuse_global)

            if self.direct:
                if self.mamba:
                    return ori_score, ori, fuse_score, fuse
                else:
                    return ori_score, ori
            else:
                if self.mamba:
                    return RGB_score, RGB_global, NI_score, NI_global, TI_score, TI_global, fuse_score, fuse
                else:
                    return RGB_score, RGB_global, NI_score, NI_global, TI_score, TI_global

        else:
            RGB = x['RGB']
            NI = x['NI']
            TI = x['TI']
            RGB_cash, RGB_global = self.BACKBONE(RGB, cam_label=cam_label, view_label=view_label, modality='rgb')
            NI_cash, NI_global = self.BACKBONE(NI, cam_label=cam_label, view_label=view_label, modality='nir')
            TI_cash, TI_global = self.BACKBONE(TI, cam_label=cam_label, view_label=view_label, modality='tir')

            if self.mamba:
                fuse = self.AAM(RGB_cash, NI_cash, TI_cash)
                return fuse
            else:
                ori = torch.cat([RGB_global, NI_global, TI_global], dim=-1)
                return ori


__factory_T_type = {
    'vit_base_patch16_224': vit_base_patch16_224,
    'deit_base_patch16_224': vit_base_patch16_224,
    'vit_small_patch16_224': vit_small_patch16_224,
    'deit_small_patch16_224': deit_small_patch16_224,
    't2t_vit_t_14': t2t_vit_t_14,
    't2t_vit_t_24': t2t_vit_t_24,
}


def make_model(cfg, num_class, camera_num, view_num=0):
    model = MambaPro(num_class, cfg, camera_num, view_num, __factory_T_type)
    print('===========Building MambaPro===========')
    return model
