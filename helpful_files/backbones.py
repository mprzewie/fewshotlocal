import sys
sys.path.append("../vision-transformer-pytorch")
sys.path.append("../vision-transformer-pytorch/src")

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
# from src.model import VisionTransformer
# from src.config import get_b16_config, get_train_config
# from src.checkpoint import load_checkpoint
from torch import nn

def get_vit(w):
    pass


class ResNet(nn.Module):
    def __init__(
            self, 
            w, 
            out_size=10, 
            kind="resnet50", 
            frozen:bool=False,
            n_convs: int = 4
        ):
        super().__init__()
        self.model=resnet_fpn_backbone(kind, pretrained=True)

        if frozen:
            for p in self.model.parameters():
                p.requires_grad = False

        self.w = w
        self.out_size = out_size
        self.frozen = frozen
        seq = [
            nn.Conv2d(256, w, kernel_size=1),
            nn.ReLU(),
        ]

        for i in range(n_convs - 1):
            seq.extend(
                [
                    nn.Conv2d(w, w, kernel_size=1),
                    nn.ReLU(),
                ]
            )

        self.conv = nn.Sequential(*seq)


    def forward(self, images):
        res = self.model(images)
        res = self.conv(res["1"])

        res = nn.functional.interpolate(
            res, size=(self.out_size, self.out_size)
        )

        return res
