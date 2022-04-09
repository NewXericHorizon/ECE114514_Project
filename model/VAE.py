import torch
from .base import BaseVAE
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from typing import List, Callable, Union, Any, TypeVar, Tuple
from torchvision.models.resnet import conv3x3

class VAE(nn.Module):
    def __init__(self, imgChannels=1, featureDim=64*7*7, zDim=256):
        super(VAE, self).__init__()

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Conv2d(imgChannels, 32, 3, padding=1)
        self.encConv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.max1 = nn.MaxPool2d(2,2)
        self.encConv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.encConv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.max2 = nn.MaxPool2d(2,2)
        self.encFC1 = nn.Linear(featureDim, zDim)
        self.encFC2 = nn.Linear(featureDim, zDim)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, featureDim)
        self.decTrans1 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)
        self.decConv1 = nn.Conv2d(64, 32, 3, padding=1)
        self.decTrans2 = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1)
        self.decConv2 = nn.Conv2d(32, imgChannels, 3, padding=1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def encoder(self, x):
        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        x = F.relu(self.encConv1(x))
        x = F.relu(self.encConv2(x))
        x = self.max1(x)
        x = F.relu(self.encConv3(x))
        x = F.relu(self.encConv4(x))
        x = self.max2(x)
        x = x.view(-1, 64*7*7)
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        return mu, logVar

    def reparameterize(self, mu, logVar):
        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):
        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.relu(self.decFC1(z))
        x = x.view(-1, 64, 7, 7)
        x = F.relu(self.decTrans1(x))
        x = F.relu(self.decConv1(x))
        x = F.relu(self.decTrans2(x))
        x = torch.sigmoid(self.decConv2(x))
        return x

    def forward(self, x):
        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar


class classifier(nn.Module):
    def __init__(self, input_dim = 64*7*7+256*2, feature_dim = 10, depth = 2):
        super(classifier, self).__init__()
        # self.encConv1 = nn.Conv2d(1, 32, 3, padding=1)
        # self.encConv2 = nn.Conv2d(32, 32, 3, padding=1)
        # self.max1 = nn.MaxPool2d(2,2)
        # self.encConv3 = nn.Conv2d(32, 64, 3, padding=1)
        # self.encConv4 = nn.Conv2d(64, 64, 3, padding=1)
        # self.max2 = nn.MaxPool2d(2,2)

        self.FC1 = nn.Linear(input_dim, 512)
        self.drop = nn.Dropout(0.5)
        self.FC2 = nn.Linear(512,512)
        self.FC3 = nn.Linear(512,256)
        self.FC4 = nn.Linear(256, feature_dim)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):

        # x = F.relu(self.encConv1(x))
        # x = F.relu(self.encConv2(x))
        # x = self.max1(x)
        # x = F.relu(self.encConv3(x))
        # x = F.relu(self.encConv4(x))
        # x = self.max2(x)

        x = F.relu(self.FC1(x))
        x = self.drop(x)
        x = F.relu(self.FC2(x))
        x = F.relu(self.FC3(x))
        x = self.FC4(x)
        return x
