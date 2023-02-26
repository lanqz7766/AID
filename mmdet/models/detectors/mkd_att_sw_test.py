# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
from mmcv.runner import load_checkpoint

from .. import build_detector
from .single_stage import SingleStageDetector

import torch.nn as nn
from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from mmdet.utils import collect_env, get_root_logger
import time
import os.path as osp
#        add loogers to debug
#        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
#        log_file = osp.join(.work_dir, f'{timestamp}.log')
#logger = get_root_logger()

def dist2(tensor_a, tensor_b, attention_mask=None, channel_attention_mask=None):
    diff = (tensor_a - tensor_b) ** 2
    #   print(diff.size())      batchsize x 1 x W x H,
    #   print(attention_mask.size()) batchsize x 1 x W x H
    diff = diff * attention_mask
    diff = diff * channel_attention_mask
    diff = torch.sum(diff) ** 0.5
    return diff


class NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True, downsample_stride=2):
        super(NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(downsample_stride, downsample_stride))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :
        :
        '''

        batch_size = x.size(0)  #   2 , 256 , 300 , 300

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)   #   2 , 128 , 150 x 150
        g_x = g_x.permute(0, 2, 1)                                  #   2 , 150 x 150, 128

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)   #   2 , 128 , 300 x 300
        theta_x = theta_x.permute(0, 2, 1)                                  #   2 , 300 x 300 , 128
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)       #   2 , 128 , 150 x 150
        f = torch.matmul(theta_x, phi_x)    #   2 , 300x300 , 150x150
        N = f.size(-1)  #   150 x 150
        f_div_C = f / N #   2 , 300x300, 150x150

        y = torch.matmul(f_div_C, g_x)  #   2, 300x300, 128
        y = y.permute(0, 2, 1).contiguous() #   2, 128, 300x300
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


@DETECTORS.register_module()
class MKDAttentionSW_test(SingleStageDetector):
    r"""Implementation of `Distilling the Knowledge in a Neural Network.
    <https://arxiv.org/abs/1503.02531>`_.
    Args:
        teacher_config (str | dict): Config file path
            or the config object of teacher model.
        teacher_ckpt (str, optional): Checkpoint path of teacher model.
            If left as None, the model will not load any weights.
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 teacher_config_1,
                 teacher_config_2,
                 teacher_config_3,
                 teacher_ckpt_1=None,
                 teacher_ckpt_2=None,
                 teacher_ckpt_3=None,
                 eval_teacher=True,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                         pretrained)
        self.eval_teacher = eval_teacher
        # Build teacher model
        if isinstance(teacher_config_1, str):
            teacher_config_1 = mmcv.Config.fromfile(teacher_config_1)
        self.teacher_model_1 = build_detector(teacher_config_1['model'])
        
        if isinstance(teacher_config_2, str):
            teacher_config_2 = mmcv.Config.fromfile(teacher_config_2)
        self.teacher_model_2 = build_detector(teacher_config_2['model'])
        
        if isinstance(teacher_config_3, str):
            teacher_config_3 = mmcv.Config.fromfile(teacher_config_3)
        self.teacher_model_3 = build_detector(teacher_config_3['model'])        
        
        if teacher_ckpt_1 is not None:
            load_checkpoint(
                self.teacher_model_1, teacher_ckpt_1, map_location='cpu')
                
        if teacher_ckpt_2 is not None:
            load_checkpoint(
                self.teacher_model_2, teacher_ckpt_2, map_location='cpu')
                
        if teacher_ckpt_3 is not None:
            load_checkpoint(
                self.teacher_model_3, teacher_ckpt_3, map_location='cpu')
        ## add adaptation layers
        self.adaptation_type = '1x1conv'
        # self.bbox_feat_adaptation = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        #   self.cls_adaptation = nn.Linear(1024, 1024)
        #   self.reg_adaptation = nn.Linear(1024, 1024)
        self.channel_wise_adaptation = nn.ModuleList([
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256)
        ])

        self.spatial_wise_adaptation = nn.ModuleList([
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
        ])

        #   self.roi_adaptation_layer = nn.Conv2d(256, 256, kernel_size=1)
        if self.adaptation_type == '3x3conv':
            #   3x3 conv
            self.adaptation_layers = nn.ModuleList([
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            ])
        if self.adaptation_type == '1x1conv':
            #   1x1 conv
            self.adaptation_layers = nn.ModuleList([
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            ])

        if self.adaptation_type == '3x3conv+bn':
            #   3x3 conv + bn
            self.adaptation_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256, affine=True)
                )
            ])
        if self.adaptation_type == '1x1conv+bn':
            #   1x1 conv + bn
            self.adaptation_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(256, affine=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(256, affine=True)
                )
            ])

        self.student_non_local = nn.ModuleList(
            [
                NonLocalBlockND(in_channels=256, inter_channels=64, downsample_stride=8),
                NonLocalBlockND(in_channels=256, inter_channels=64, downsample_stride=4),
                NonLocalBlockND(in_channels=256),
                NonLocalBlockND(in_channels=256),
                NonLocalBlockND(in_channels=256)
            ]
        )
        self.teacher_non_local = nn.ModuleList(
            [
                NonLocalBlockND(in_channels=256, inter_channels=64, downsample_stride=8),
                NonLocalBlockND(in_channels=256, inter_channels=64, downsample_stride=4),
                NonLocalBlockND(in_channels=256),
                NonLocalBlockND(in_channels=256),
                NonLocalBlockND(in_channels=256)
            ]
        )
        self.non_local_adaptation = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        ])

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.extract_feat(img)
        with torch.no_grad():
            teacher_x_1 = self.teacher_model_1.extract_feat(img)
            losses_T_1 = self.teacher_model_1.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
#            T_total_1 = {k: sum(losses_T_1[k]) for k in losses_T_1.keys()}
            
        with torch.no_grad():
            teacher_x_2 = self.teacher_model_2.extract_feat(img)
            #out_teacher_2 = self.teacher_model_2.bbox_head(teacher_x_2)
            losses_T_2 = self.teacher_model_2.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
#            T_total_2 = {k: sum(losses_T_2[k]) for k in losses_T_2.keys()}
            
        with torch.no_grad():
            teacher_x_3 = self.teacher_model_3.extract_feat(img)
            #out_teacher_2 = self.teacher_model_2.bbox_head(teacher_x_2)
            losses_T_3 = self.teacher_model_3.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
#            T_total_3 = {k: sum(losses_T_3[k]) for k in losses_T_3.keys()}
            
#        Loss_1 = sum(T_total_1.values())
#        Loss_2 = sum(T_total_2.values())
#        Loss_3 = sum(T_total_3.values())
        
        
        
        losses = self.bbox_head.forward_train(x, img_metas,
                                              gt_bboxes, gt_labels,
                                              gt_bboxes_ignore)
                                              
                                              
                                              
        t = 0.1
        s_ratio = 1.0
        kd_feat_loss = 0
        kd_channel_loss = 0
        kd_spatial_loss = 0
    
        # for channel attention
        c_t = 0.1
        c_s_ratio = 1.0            
        with torch.no_grad():
            t_feats_1 = self.teacher_model_1.extract_feat(img)
            t_feats_2 = self.teacher_model_2.extract_feat(img)
            t_feats_3 = self.teacher_model_3.extract_feat(img)
            
        if t_feats_1 is not None:
            for _i in range(len(t_feats_1)):
                ## Teacher 1
                w1 = torch.exp(-0.1*(losses_T_1['loss_cls'][_i] + losses_T_1['loss_bbox'][_i]))
                t_attention_mask_1 = torch.mean(torch.abs(t_feats_1[_i]), [1], keepdim=True)
                size_1 = t_attention_mask_1.size()
                t_attention_mask_1 = t_attention_mask_1.view(x[0].size(0), -1)
                t_attention_mask_1 = torch.softmax(t_attention_mask_1 / t, dim=1) * size_1[-1] * size_1[-2]
                t_attention_mask_1 = t_attention_mask_1.view(size_1)
                
                s_attention_mask_1 = torch.mean(torch.abs(x[_i]), [1], keepdim=True)
                size_1 = s_attention_mask_1.size()
                s_attention_mask_1 = s_attention_mask_1.view(x[0].size(0), -1)
                s_attention_mask_1 = torch.softmax(s_attention_mask_1 / t, dim=1) * size_1[-1] * size_1[-2]
                s_attention_mask_1 = s_attention_mask_1.view(size_1)
    
                c_t_attention_mask_1 = torch.mean(torch.abs(t_feats_1[_i]), [2, 3], keepdim=True)  # 2 x 256 x 1 x1
                c_size_1 = c_t_attention_mask_1.size()
                c_t_attention_mask_1 = c_t_attention_mask_1.view(x[0].size(0), -1)  # 2 x 256
                c_t_attention_mask_1 = torch.softmax(c_t_attention_mask_1 / c_t, dim=1) * 256
                c_t_attention_mask_1 = c_t_attention_mask_1.view(c_size_1)  # 2 x 256 -> 2 x 256 x 1 x 1
    
                c_s_attention_mask_1 = torch.mean(torch.abs(x[_i]), [2, 3], keepdim=True)  # 2 x 256 x 1 x1
                c_size_1 = c_s_attention_mask_1.size()
                c_s_attention_mask_1 = c_s_attention_mask_1.view(x[0].size(0), -1)  # 2 x 256
                c_s_attention_mask_1 = torch.softmax(c_s_attention_mask_1 / c_t, dim=1) * 256
                c_s_attention_mask_1 = c_s_attention_mask_1.view(c_size_1)  # 2 x 256 -> 2 x 256 x 1 x 1
    
                sum_attention_mask_1 = (t_attention_mask_1 + s_attention_mask_1 * s_ratio) / (1 + s_ratio)
                sum_attention_mask_1 = sum_attention_mask_1.detach()
    
                c_sum_attention_mask_1 = (c_t_attention_mask_1 + c_s_attention_mask_1 * c_s_ratio) / (1 + c_s_ratio)
                c_sum_attention_mask_1 = c_sum_attention_mask_1.detach()
                    
                ## Teacher 2    
                w2 = torch.exp(-0.1*(losses_T_2['loss_cls'][_i] + losses_T_2['loss_bbox'][_i]))
                t_attention_mask_2 = torch.mean(torch.abs(t_feats_2[_i]), [1], keepdim=True)
                size_2 = t_attention_mask_2.size()
                t_attention_mask_2 = t_attention_mask_2.view(x[0].size(0), -1)
                t_attention_mask_2 = torch.softmax(t_attention_mask_2 / t, dim=1) * size_2[-1] * size_2[-2]
                t_attention_mask_2 = t_attention_mask_2.view(size_2)
                
                s_attention_mask_2 = torch.mean(torch.abs(x[_i]), [1], keepdim=True)
                size_2 = s_attention_mask_2.size()
                s_attention_mask_2 = s_attention_mask_2.view(x[0].size(0), -1)
                s_attention_mask_2 = torch.softmax(s_attention_mask_2 / t, dim=1) * size_2[-1] * size_2[-2]
                s_attention_mask_2 = s_attention_mask_2.view(size_2)
    
                c_t_attention_mask_2 = torch.mean(torch.abs(t_feats_2[_i]), [2, 3], keepdim=True)  # 2 x 256 x 1 x1
                c_size_2 = c_t_attention_mask_2.size()
                c_t_attention_mask_2 = c_t_attention_mask_2.view(x[0].size(0), -1)  # 2 x 256
                c_t_attention_mask_2 = torch.softmax(c_t_attention_mask_2 / c_t, dim=1) * 256
                c_t_attention_mask_2 = c_t_attention_mask_2.view(c_size_2)  # 2 x 256 -> 2 x 256 x 1 x 1
    
                c_s_attention_mask_2 = torch.mean(torch.abs(x[_i]), [2, 3], keepdim=True)  # 2 x 256 x 1 x1
                c_size_2 = c_s_attention_mask_2.size()
                c_s_attention_mask_2 = c_s_attention_mask_2.view(x[0].size(0), -1)  # 2 x 256
                c_s_attention_mask_2 = torch.softmax(c_s_attention_mask_2 / c_t, dim=1) * 256
                c_s_attention_mask_2 = c_s_attention_mask_2.view(c_size_2)  # 2 x 256 -> 2 x 256 x 1 x 1
    
                sum_attention_mask_2 = (t_attention_mask_2 + s_attention_mask_2 * s_ratio) / (1 + s_ratio)
                sum_attention_mask_2 = sum_attention_mask_2.detach()
    
                c_sum_attention_mask_2 = (c_t_attention_mask_2 + c_s_attention_mask_2 * c_s_ratio) / (1 + c_s_ratio)
                c_sum_attention_mask_2 = c_sum_attention_mask_2.detach()                    
                    
                ## Teacher 3    
                w3 = torch.exp(-0.1*(losses_T_3['loss_cls'][_i] + losses_T_3['loss_bbox'][_i]))
                t_attention_mask_3 = torch.mean(torch.abs(t_feats_3[_i]), [1], keepdim=True)
                size_3 = t_attention_mask_3.size()
                t_attention_mask_3 = t_attention_mask_3.view(x[0].size(0), -1)
                t_attention_mask_3 = torch.softmax(t_attention_mask_3 / t, dim=1) * size_3[-1] * size_3[-2]
                t_attention_mask_3 = t_attention_mask_3.view(size_3)
                
                s_attention_mask_3 = torch.mean(torch.abs(x[_i]), [1], keepdim=True)
                size_3 = s_attention_mask_3.size()
                s_attention_mask_3 = s_attention_mask_1.view(x[0].size(0), -1)
                s_attention_mask_3 = torch.softmax(s_attention_mask_3 / t, dim=1) * size_3[-1] * size_3[-2]
                s_attention_mask_3 = s_attention_mask_1.view(size_3)
    
                c_t_attention_mask_3 = torch.mean(torch.abs(t_feats_3[_i]), [2, 3], keepdim=True)  # 2 x 256 x 1 x1
                c_size_3 = c_t_attention_mask_3.size()
                c_t_attention_mask_3 = c_t_attention_mask_3.view(x[0].size(0), -1)  # 2 x 256
                c_t_attention_mask_3 = torch.softmax(c_t_attention_mask_3 / c_t, dim=1) * 256
                c_t_attention_mask_3 = c_t_attention_mask_3.view(c_size_3)  # 2 x 256 -> 2 x 256 x 1 x 1
    
                c_s_attention_mask_3 = torch.mean(torch.abs(x[_i]), [2, 3], keepdim=True)  # 2 x 256 x 1 x1
                c_size_3 = c_s_attention_mask_3.size()
                c_s_attention_mask_3 = c_s_attention_mask_3.view(x[0].size(0), -1)  # 2 x 256
                c_s_attention_mask_3 = torch.softmax(c_s_attention_mask_3 / c_t, dim=1) * 256
                c_s_attention_mask_3 = c_s_attention_mask_3.view(c_size_3)  # 2 x 256 -> 2 x 256 x 1 x 1
    
                sum_attention_mask_3 = (t_attention_mask_3 + s_attention_mask_3 * s_ratio) / (1 + s_ratio)
                sum_attention_mask_3 = sum_attention_mask_3.detach()
    
                c_sum_attention_mask_3 = (c_t_attention_mask_3 + c_s_attention_mask_3 * c_s_ratio) / (1 + c_s_ratio)
                c_sum_attention_mask_3 = c_sum_attention_mask_3.detach()                    
                    
                    
    
                kd_feat_loss += (w1*dist2(t_feats_1[_i], self.adaptation_layers[_i](x[_i]), attention_mask=sum_attention_mask_1,
                                        channel_attention_mask=c_sum_attention_mask_1) * 7e-5 * 6 +
                                w2*dist2(t_feats_2[_i], self.adaptation_layers[_i](x[_i]), attention_mask=sum_attention_mask_2,
                                        channel_attention_mask=c_sum_attention_mask_2) * 7e-5 * 6 +
                                w3*dist2(t_feats_3[_i], self.adaptation_layers[_i](x[_i]), attention_mask=sum_attention_mask_3,
                                        channel_attention_mask=c_sum_attention_mask_3) * 7e-5 * 6)/3
                                        
                kd_channel_loss +=  (w1*torch.dist(torch.mean(t_feats_1[_i], [2, 3]),
                                                self.channel_wise_adaptation[_i](torch.mean(x[_i], [2, 3]))) * 4e-3 * 6 +
                                    w2*torch.dist(torch.mean(t_feats_2[_i], [2, 3]),
                                                self.channel_wise_adaptation[_i](torch.mean(x[_i], [2, 3]))) * 4e-3 * 6 +
                                    w3*torch.dist(torch.mean(t_feats_3[_i], [2, 3]),
                                                self.channel_wise_adaptation[_i](torch.mean(x[_i], [2, 3]))) * 4e-3 * 6)/3
                                                
                                                
                t_spatial_pool_1 = torch.mean(t_feats_1[_i], [1]).view(t_feats_1[_i].size(0), 1, t_feats_1[_i].size(2),
                                                                   t_feats_1[_i].size(3))
                t_spatial_pool_2 = torch.mean(t_feats_2[_i], [1]).view(t_feats_2[_i].size(0), 1, t_feats_2[_i].size(2),
                                                                   t_feats_2[_i].size(3))
                t_spatial_pool_3 = torch.mean(t_feats_3[_i], [1]).view(t_feats_3[_i].size(0), 1, t_feats_3[_i].size(2),
                                                                   t_feats_3[_i].size(3))
                                                                   
                s_spatial_pool = torch.mean(x[_i], [1]).view(x[_i].size(0), 1, x[_i].size(2),
                                                             x[_i].size(3))
                kd_spatial_loss +=  (w1*torch.dist(t_spatial_pool_1,
                                                self.spatial_wise_adaptation[_i](s_spatial_pool)) * 4e-3 * 6 +
                                    w2*torch.dist(t_spatial_pool_1,
                                                self.spatial_wise_adaptation[_i](s_spatial_pool)) * 4e-3 * 6 +
                                    w3*torch.dist(t_spatial_pool_1,
                                                self.spatial_wise_adaptation[_i](s_spatial_pool)) * 4e-3 * 6)/3
        losses.update({'kd_feat_loss': kd_feat_loss})
        losses.update({'kd_channel_loss': kd_channel_loss})
        losses.update({'kd_spatial_loss': kd_spatial_loss})
    
        kd_nonlocal_loss = 0
#       with torch.no_grad():
#           t_feats = self.teacher_model_1.extract_feat(img)

        if t_feats_1 is not None:
            for _i in range(len(t_feats_1)):
                w1 = torch.exp(-0.1*(losses_T_1['loss_cls'][_i] + losses_T_1['loss_bbox'][_i]))
                w2 = torch.exp(-0.1*(losses_T_2['loss_cls'][_i] + losses_T_2['loss_bbox'][_i]))
                w3 = torch.exp(-0.1*(losses_T_3['loss_cls'][_i] + losses_T_3['loss_bbox'][_i]))
                s_relation = self.student_non_local[_i](x[_i])
                t_relation_1 = self.teacher_non_local[_i](t_feats_1[_i])
                t_relation_2 = self.teacher_non_local[_i](t_feats_2[_i])
                t_relation_3 = self.teacher_non_local[_i](t_feats_3[_i])
                #   print(s_relation.size())
                kd_nonlocal_loss += (w1*torch.dist(self.non_local_adaptation[_i](s_relation), t_relation_1, p=2) + 
                                    w1*torch.dist(self.non_local_adaptation[_i](s_relation), t_relation_1, p=2) +
                                    w1*torch.dist(self.non_local_adaptation[_i](s_relation), t_relation_1, p=2))/3
                                    
        losses.update(kd_nonlocal_loss=kd_nonlocal_loss * 7e-5 * 6)

        return losses
        
        
    def cuda(self, device=None):
        """Since teacher_model is registered as a plain object, it is necessary
        to put the teacher model to cuda when calling cuda function."""
        self.teacher_model_1.cuda(device=device)
        self.teacher_model_2.cuda(device=device)
        self.teacher_model_3.cuda(device=device)
        return super().cuda(device=device)

    def train(self, mode=True):
        """Set the same train mode for teacher and student model."""
        if self.eval_teacher:
            self.teacher_model_1.train(False)
            self.teacher_model_2.train(False)
            self.teacher_model_3.train(False)
        else:
            self.teacher_model_1.train(mode)
            self.teacher_model_2.train(mode)
            self.teacher_model_3.train(mode)
        super().train(mode)

    def __setattr__(self, name, value):
        """Set attribute, i.e. self.name = value
        This reloading prevent the teacher model from being registered as a
        nn.Module. The teacher module is registered as a plain object, so that
        the teacher parameters will not show up when calling
        ``self.parameters``, ``self.modules``, ``self.children`` methods.
        """
        if name == 'teacher_model_1' or name == 'teacher_mode_2' or name == 'teacher_mode_3':
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)
