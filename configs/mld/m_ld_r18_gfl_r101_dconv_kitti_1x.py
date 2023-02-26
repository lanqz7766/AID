_base_ = [
    '../_base_/datasets/kitti_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]


# multi-scale training
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 480), (1333, 800)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
data = dict(train=dict(pipeline=train_pipeline))


teacher_ckpt_1 = '/users/PCS0252/lanqz7766/MMdetection/checkpoints/gfl_r101_fpn_dconv_c3-c5_mstrain_2x_kitti/epoch_24.pth'
teacher_ckpt_2 = '/users/PCS0252/lanqz7766/mmdetection/KD_kitti_results/gfl_r101_fpn_mstrain_2x_kitti/epoch_24.pth'

model = dict(
    type='MKnowledgeDistillationSingleStageDetector',
    teacher_config_1='configs/gfl_kitti/gfl_r101_fpn_dconv_c3-c5_mstrain_2x_coco.py',
    teacher_config_2='configs/gfl_kitti/gfl_r101_fpn_mstrain_2x_kitti.py',
    teacher_ckpt_1=teacher_ckpt_1,
    teacher_ckpt_2=teacher_ckpt_2,
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet18')),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='LDHead',
        num_classes=3,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        loss_ld=dict(
            type='KnowledgeDistillationKLDivLoss', loss_weight=0.05, T=10),
        loss_kd=dict(
            type='KnowledgeDistillationKLDivLoss', loss_weight=0.05, T=2),
        reg_max=16,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

optimizer = dict(type='SGD', lr=0.00375, momentum=0.9, weight_decay=0.0001)
