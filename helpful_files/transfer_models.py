from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet50

from helpful_files.vit import VisionTransformer


def get_resnet(ensemble: int, n_classes: int):
    models = []
    for _ in range(ensemble):
        model = resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, n_classes)

        model.cuda()
        model.train()
        models.append(model)

    return models


def get_vit(ensemble: int, n_classes: int, weights_path: Path):
    if weights_path is None:
        raise ValueError(f'ViT weights is None')
    if not weights_path.exists():
        raise ValueError(f'Weights file {weights_path} doesn\'t exists.')

    models = []
    vit_weights = torch.load(weights_path)['state_dict']
    del vit_weights['classifier.weight']
    del vit_weights['classifier.bias']

    for _ in range(ensemble):
        model = VisionTransformer(image_size=(384, 384), num_classes=n_classes)
        model.load_state_dict(vit_weights, strict=False)

        model.cuda()
        model.train()
        models.append(model)

    return models
