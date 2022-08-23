# dataset settings
dataset_type = 'AnomalDataset'
data_root = 'data/anomal_dataset/'
train_pipeline = [
    dict(type='LoadLogit'),
    dict(type='LoadSoftmax'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type="LogitMinMaxNormalize", method="global"),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['logit', 'softmax', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadLogit'),
    dict(type='LoadSoftmax'),
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
        softmax_dir='softmax/train',
        ann_dir='gtFine/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        logit_dir='logit/val',
        softmax_dir='softmax/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        logit_dir='logit/val',
        softmax_dir='softmax/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline))