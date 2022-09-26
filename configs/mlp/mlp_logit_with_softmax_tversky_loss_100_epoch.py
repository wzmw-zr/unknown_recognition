_base_ = [
    '../_base_/models/mlp.py',
    '../_base_/datasets/anomal_dataset.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_epoch_300.py'
]

model = dict(
    classifier=dict(
        type="MLP",
        input_features_infos=[
            dict(
                type="logit",
                in_channels=31,
            ),
            dict(
                type="softmax",
                in_channels=31
            )
        ],
        hidden_channels=256,
        num_classes=2,
        loss_decode=[
            dict(
                type="TverskyLoss",
                smooth=1,
                class_weight=None,
                loss_weight=3.0,
                ignore_index=255,
                alpha=0.3,
                beta=0.7,
                loss_name='loss_tversky'
            ),
            dict(
                type="CrossEntropyLoss",
                loss_weight=1.0,
            ),
        ],
        ignore_index=255,
        init_cfg=None
    )
)

dataset_type = 'AnomalDatasetFast'
# data_root = 'data/anomal_dataset/'
data_root = 'data/anomal_campusE1/'
train_pipeline = [
    dict(type='LoadLogit'),
    dict(type="LoadSoftmaxFromLogit"),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type="LogitMinMaxNormalize", method="global"),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['logit', 'softmax', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadLogit'),
    dict(type="LoadSoftmaxFromLogit"),
    dict(type="LogitMinMaxNormalize", method="global"),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['logit', 'softmax']),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        logit_dir='logit/train',
        ann_dir='gtFine/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        logit_dir='logit/test',
        ann_dir='gtFine/test',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        logit_dir='logit/test',
        ann_dir='gtFine/test',
        pipeline=test_pipeline))

optimizer = dict(
    _delete_=True,
    type='AdamW', lr=0.1, weight_decay=0.01)

lr_config = dict(
    _delete_=True,
    policy='fixed')