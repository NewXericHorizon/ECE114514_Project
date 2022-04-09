import torch
from torch import nn
from torch.nn import functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class wide_VAE(nn.Module):
    def __init__(self, featureDim = 320*8*8, zDim = 4096, dropRate = 0.0):
        super(wide_VAE, self).__init__()
        self.in_layer = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.enc_block1 = self._make_layer_encoder(3, in_planes=16, out_planes=64, stride=1)
        self.enc_block2 = self._make_layer_encoder(3, in_planes=64, out_planes=160, stride=2)
        self.enc_block3 = self._make_layer_encoder(3, in_planes=160, out_planes=320, stride=2)
        self.norm = nn.BatchNorm2d(320)
        self.encFC1 = nn.Linear(featureDim, zDim)
        self.encFC2 = nn.Linear(featureDim, zDim)

        self.decFC1 = nn.Linear(zDim, featureDim)
        self.dec_block1 = self._make_layer_decoder(3, in_planes=320, out_planes=160 ,stride=2)
        self.dec_block2 = self._make_layer_decoder(3, in_planes=160, out_planes=64 ,stride=2)
        self.dec_block3 = self._make_layer_decoder(3, in_planes=64, out_planes=16, stride=1)
        self.out_layer = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer_encoder(self, depth, in_planes, out_planes, stride, dropRate = 0.0):
        layers = []
        for i in range(depth):
            layers.append(BasicBlock(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    
    def _make_layer_decoder(self, depth, in_planes, out_planes, stride):
        layers = []
        for i in range(depth):
            if i == 0 and stride != 1:
                layers.append(nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, output_padding=1, bias=False))
                layers.append(nn.BatchNorm2d(out_planes))
                layers.append(nn.ReLU(inplace=True))
            else:
                layers.append(nn.Conv2d(i == 0 and in_planes or out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False))
                layers.append(nn.BatchNorm2d(out_planes))
                layers.append(nn.ReLU(inplace=True))

            layers.append(nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_planes))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def encoder(self, x):
        x = self.in_layer(x)
        x = self.enc_block1(x)
        x = self.enc_block2(x)
        x = self.enc_block3(x)
        x = F.relu(self.norm(x))
        x = x.view(-1, 320*8*8)
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        return mu, logVar

    def reparameterize(self, mu, logVar):
        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):
        z = F.relu(self.decFC1(z))
        z = z.view(-1, 320, 8, 8)
        z = self.dec_block1(z)
        z = self.dec_block2(z)
        z = self.dec_block3(z)
        z = torch.sigmoid(self.out_layer(z))
        return z

    def forward(self, x):
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar

class classifier(nn.Module):
    def __init__(self, input_dim = 4096*2, feature_dim=10):
        super(classifier, self).__init__()
        self.FC1 = nn.Linear(input_dim, 1024)
        self.FC2 = nn.Linear(1024, 256)
        self.FC3 = nn.Linear(256, 64)
        self.FC4 = nn.Linear(64, feature_dim)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.FC1(x))
        x = F.relu(self.FC2(x))
        x = F.relu(self.FC3(x))
        x = self.FC4(x)
        return x
