# model settings
# norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='AnomalDetector',
    pretrained=None,
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
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0),
        ignore_index=255,
        init_cfg=None
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))