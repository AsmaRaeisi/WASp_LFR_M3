import torch
import torch.nn as nn

import scipy.io
import os

import numpy as np

if torch.__version__ == "1.2.0":
    from torchvision.models.utils import load_state_dict_from_url
else:
    from torch.utils.model_zoo import load_url


COLOR_NAMES = ['black', 'blue', 'brown', 'grey', 'green', 'orange',
               'pink', 'purple', 'red', 'white', 'yellow']
COLOR_RGB = [[0, 0, 0] , [0, 0, 1], [.5, .4, .25] , [.5, .5, .5] , [0, 1, 0] , [1, .8, 0] ,
             [1, .5, 1] ,[1, 0, 1], [1, 0, 0], [1, 1, 1 ] , [ 1, 1, 0 ]]

COLORNAMES_TABLE_PATH = os.path.join(os.path.dirname(__file__), 'colornames_w2c.mat')
COLORNAMES_TABLE = scipy.io.loadmat(COLORNAMES_TABLE_PATH)['w2c']


def colornames_image(image, mode='probability'):
    """Compute a color names image from an RGB image
    Parameters
    --------------
    image : array_like
        The input image array (RxCx3)
    mode : str
        If 'index' then it returns an RxC image where each element is the corresponding color name label.
        If 'probability', then the returned image has size RxCx11 where the last dimension are the probabilities for each
        color label.
        The corresponding human readable name of each label is found in the `COLOR_NAMES` list.
    Returns
    --------------
    Color names encoded image, as explained by the `mode` parameter.
    """
    image = image.astype('double')
    idx = np.floor(image[..., 0] / 8) + 32 * np.floor(image[..., 1] / 8) + 32 * 32 * np.floor(image[..., 2] / 8)
    m = COLORNAMES_TABLE[idx.astype('int')]

    if mode == 'index':
        return np.argmax(m, 2)
    elif mode == 'probability':
        return m
    else:
        raise ValueError("No such mode: '{}'".format(mode))

"""
    These where taken from the torchvision repository, and modified to return the 
    features instead of the classification score.
"""

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNetFeature(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNetFeature, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return x


def alexnetFeatures(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNetFeature(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
