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
                in_channels=19,
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