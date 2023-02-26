_base_ = [
    '../../_base_/datasets/kitti_detection.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]
teacher_ckpt = '/users/PCS0252/lanqz7766/mmdetection/KD_kitti_results/gfl_r101_fpn_mstrain_2x_kitti/epoch_24.pth'  # noqa
model = dict(
    type='AIDLD',
    pretrained='torchvision://resnet18',
    teacher_config='configs/gfl_kitti/gfl_r101_fpn_mstrain_2x_kitti.py',
    teacher_ckpt=teacher_ckpt,
    output_feature=True,
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='LDHeadv2',
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
            type='KnowledgeDistillationKLDivLoss', loss_weight=0, T=10),
        loss_ld_vlr=dict(
            type='KnowledgeDistillationKLDivLoss', loss_weight=0, T=10),
        loss_kd=dict(
            type='KnowledgeDistillationKLDivLoss', loss_weight=0, T=2),
        loss_im=dict(type='IMLoss', loss_weight=0.12),
        reg_max=16,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        imitation_method='decouple'  # gibox, finegrain, decouple, fitnet
        ),
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

optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,)


# evaluation = dict(interval=1, metric='bbox')