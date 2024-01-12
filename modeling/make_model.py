import torch
import torch.nn as nn
from modeling.backbones.vit_pytorch import vit_base_patch16_224, vit_small_patch16_224, \
    deit_small_patch16_224
from modeling.backbones.t2t import t2t_vit_t_14, t2t_vit_t_24
from timm.models.layers import trunc_normal_
from modeling.make_model_clipreid import load_clip_to_cpu


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
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(build_transformer, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH_T
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cv_embed_sign = cfg.MODEL.SIE_CAMERA
        self.stage = cfg.MODEL.STAGE
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.model_name = cfg.MODEL.TRANSFORMER_TYPE
        self.in_planes = 768
        self.trans_type = cfg.MODEL.TRANSFORMER_TYPE
        if 't2t' in cfg.MODEL.TRANSFORMER_TYPE:
            self.in_planes = 512
        if 'edge' in cfg.MODEL.TRANSFORMER_TYPE or cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224':
            self.in_planes = 384
        if '14' in cfg.MODEL.TRANSFORMER_TYPE:
            self.in_planes = 384
        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            self.camera_num = camera_num
        else:
            self.camera_num = 0
        # No view
        self.view_num = 0

        if cfg.MODEL.BASE == 1:
            self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                            num_classes=num_classes,
                                                            camera=self.camera_num, view=self.view_num,
                                                            stride_size=cfg.MODEL.STRIDE_SIZE,
                                                            drop_path_rate=cfg.MODEL.DROP_PATH,
                                                            drop_rate=cfg.MODEL.DROP_OUT,
                                                            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        else:
            self.clip = 1
            self.sie_xishu = cfg.MODEL.SIE_COE
            clip_model = load_clip_to_cpu(self.model_name, cfg.INPUT.SIZE_TRAIN[0] // cfg.MODEL.STRIDE_SIZE[0],
                                          cfg.INPUT.SIZE_TRAIN[1] // cfg.MODEL.STRIDE_SIZE[1],
                                          cfg.MODEL.STRIDE_SIZE)
            clip_model.to("cuda")
            self.in_planes = 512
            self.base = clip_model.visual
            self.pormpt_num = cfg.MODEL.PREFIX_NUM
            if cfg.MODEL.STAGE == 1:
                self.cv_embed_sign = False
                for param in self.base.parameters():
                    param.requires_grad = False

                visual_prompt = torch.zeros(1, self.pormpt_num, 768)
                nn.init.normal_(visual_prompt, std=0.02)
                self.visual_prompt = nn.Parameter(visual_prompt)
                self.visual_prompt.requires_grad = True
            else:
                for param in self.base.parameters():
                    param.requires_grad = False
                visual_prompt = torch.zeros(1, self.pormpt_num, 768)
                self.visual_prompt = nn.Parameter(visual_prompt)
                self.visual_prompt.requires_grad = True
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

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))
        elif pretrain_choice == 'clip':
            print('Loading pretrained model from CLIP')
        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None, cam_label=None, view_label=None):
        if self.clip == 0:
            x = self.base(x, cam_label=cam_label, view_label=view_label)
        else:
            if self.cv_embed_sign:
                cv_embed = self.sie_xishu * self.cv_embed[cam_label]
            else:
                cv_embed = None
            x = self.base(x, cv_embed, visual_prompt=self.visual_prompt)
        if self.stage == 1:
            global_feat = x[:, 0]
        else:
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


class BASELINE(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(BASELINE, self).__init__()
        self.BACKBONE = build_transformer(num_classes, cfg, camera_num, view_num, factory)

        self.num_classes = num_classes
        self.cfg = cfg
        self.camera = camera_num
        self.view = view_num
        self.direct = cfg.MODEL.DIRECT
        if cfg.MODEL.STAGE == 1:
            self.direct = 0
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        self.mix_dim = 512

        self.miss_type = cfg.TEST.MISS
        self.classifier = nn.Linear(3 * self.mix_dim, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(3 * self.mix_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

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
            NI_cash, NI_score, NI_global = self.BACKBONE(NI, cam_label=cam_label, view_label=view_label)
            TI_cash, TI_score, TI_global = self.BACKBONE(TI, cam_label=cam_label, view_label=view_label)
            RGB_cash, RGB_score, RGB_global = self.BACKBONE(RGB, cam_label=cam_label, view_label=view_label)
            if self.direct:
                ori = torch.cat([RGB_global, NI_global, TI_global], dim=-1)
                ori_global = self.bottleneck(ori)
                ori_score = self.classifier(ori_global)
                return ori_score, ori
            else:
                return RGB_score, RGB_global, NI_score, NI_global, TI_score, TI_global

        else:
            RGB = x['RGB']
            NI = x['NI']
            TI = x['TI']
            NI_cash, NI_global = self.BACKBONE(NI, cam_label=cam_label, view_label=view_label)
            TI_cash, TI_global = self.BACKBONE(TI, cam_label=cam_label, view_label=view_label)
            RGB_cash, RGB_global = self.BACKBONE(RGB, cam_label=cam_label, view_label=view_label)
            ori = torch.cat([RGB_global, NI_global, TI_global], dim=-1)
            ori_global = self.bottleneck(ori)

            if self.neck_feat == 'after':
                pass
            else:
                ori_global = ori
            return torch.cat([ori_global], dim=-1)


__factory_T_type = {
    'vit_base_patch16_224': vit_base_patch16_224,
    'deit_base_patch16_224': vit_base_patch16_224,
    'vit_small_patch16_224': vit_small_patch16_224,
    'deit_small_patch16_224': deit_small_patch16_224,
    't2t_vit_t_14': t2t_vit_t_14,
    't2t_vit_t_24': t2t_vit_t_24,
}


def make_model(cfg, num_class, camera_num, view_num=0):
    model = BASELINE(num_class, cfg, camera_num, view_num, __factory_T_type)
    print('===========Building BASELINE===========')
    return model
