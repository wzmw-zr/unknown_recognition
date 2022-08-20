import torch

from unknown_recognition.models.classifiers import MLP
from .utils import to_cuda


def test_mlp():

    logit = torch.randn(1, 19, 100, 100)
    softmax = torch.randn(1, 19, 100, 100)
    classifier = MLP(
        input_features_infos=[
            dict(
                type="logits",
                in_channels=19,
            ),
            dict(
                type="softmax_distances",
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
        init_cfg=None)
    if torch.cuda.is_available():
        classifier, logit, softmax = to_cuda(classifier, logit, softmax)
    outputs = classifier(logit, softmax)
    assert outputs.shape == (1, classifier.num_classes, 100, 100)


if __name__ == "__main__":
    test_mlp()