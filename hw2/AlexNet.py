import numpy as np

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import torchvision.models as models

model_urls = {
        'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class LocalizerAlexNet(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNet, self).__init__()
        #TODO: Define model
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1)),
            nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1)),
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 20, kernel_size=(1, 1), stride=(1, 1)),
        )

    def forward(self, x):
        #TODO: Define forward pass
        return self.classifier(self.features(x))


class LocalizerAlexNetRobust(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNetRobust, self).__init__()
        #TODO: Define model
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1)),
            nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1)),
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 20, kernel_size=(1, 1), stride=(1, 1)),
            nn.AvgPool2d(5, stride=1, padding=2)
        )


    def forward(self, x):
        #TODO: Define fwd pass
        return self.classifier(self.features(x))


def localizer_alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNet(**kwargs)
    #TODO: Initialize weights correctly based on whethet it is pretrained or not
    if pretrained:
        print("Loading Pretrained Weights")
        alexnet = models.alexnet(pretrained=True)
        for i in range(12):
            model.features[i] = alexnet.features[i]

    for i in [0, 2, 4]:
        nn.init.xavier_normal_(model.classifier[i].weight)

    return model


def localizer_alexnet_robust(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNetRobust(**kwargs)
    #TODO: Ignore for now until instructed
    if pretrained:
        print("Loading Pretrained Weights")
        alexnet = models.alexnet(pretrained=True)
        for i in range(12):
            model.features[i] = alexnet.features[i]

    for i in [0, 2, 4]:
        nn.init.xavier_normal_(model.classifier[i].weight)   

    return model
