# dataset settings
dataset_type = 'AnomalDataset'
data_root = 'data/anomal/'
train_pipeline = [
    dict(type='LoadLogit'),
    dict(type='LoadSoftmax'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['logit', 'softmax', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadLogit'),
    dict(type='LoadSoftmax'),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['logit', 'softmax']),
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/training',
        ann_dir='annotations/training',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline))