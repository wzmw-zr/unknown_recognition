_base_ = [
    '../_base_/models/mlp_logit_top2_softmax.py',
    '../_base_/datasets/anomal_dataset.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_epoch_300.py'
]

model = dict(
    classifier=dict(
        type="MLPLogitTop2Softmax",
        input_features_infos=[
            dict(
                type="logit",
                in_channels=19,
            ),
            dict(
                type="top2_distance",
                in_channels=1,
            ),
            dict(
                type="softmax",
                in_channels=19
            )
        ],
        hidden_channels=256,
        num_classes=2,
        loss_decode=dict(
            type='LogitAdjustmentLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_ratio=[0.01, 0.99]
        ),
        ignore_index=255,
        init_cfg=None
    )
)

# dataset settings
dataset_type = 'AnomalDatasetFast'
# data_root = 'data/anomal_dataset/'
data_root = 'data/anomal_campusE1/'
train_pipeline = [
    dict(type='LoadLogit'),
    dict(type="LoadTop2LogitDistance"),
    dict(type="LoadSoftmaxFromLogit"),
    dict(type="LogitMinMaxNormalize", method="global"),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['logit', 'top2_distance', 'softmax', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadLogit'),
    dict(type="LoadTop2LogitDistance"),
    dict(type="LoadSoftmaxFromLogit"),
    dict(type="LogitMinMaxNormalize", method="global"),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['logit', 'top2_distance', 'softmax']),
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        # logit_dir='logit/train',
        # ann_dir='gtFine/train',
        logit_dir='logit/deeplabv3plus/train',
        ann_dir='gtFine/deeplabv3plus/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        # logit_dir='logit/val',
        # ann_dir='gtFine/val',
        logit_dir='logit/deeplabv3plus/val',
        ann_dir='gtFine/deeplabv3plus/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        # logit_dir='logit/val',
        # ann_dir='gtFine/val',
        logit_dir='logit/deeplabv3plus/val',
        ann_dir='gtFine/deeplabv3plus/val',
        pipeline=test_pipeline))

optimizer = dict(
    _delete_=True,
    type='AdamW', lr=0.1, weight_decay=0.01)

lr_config = dict(
    _delete_=True,
    policy='fixed')