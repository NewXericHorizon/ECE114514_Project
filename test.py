import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from model.wideVAE import *
from blackbox_pgd_model.wideresnet_update import *
from pgd_attack import *
import torch.optim as optim
import argparse
import numpy as np
from util import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 VAE Training')
parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--latent-dim', type=int, default=1024)
parser.add_argument('--epochs', type=int, default=70)
parser.add_argument('--testtime-epochs', type=int, default=20)
parser.add_argument('--testtime-lr',  default=0.1)
parser.add_argument('--lr', default=0.01)
parser.add_argument('--beta', default=0.5)
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

def train(model, data_loader, optimizer, epoch_num):
    model.train()
    loss_sum = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        #data = data.view(batch_size, x_dim)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logit = model(data)
        #logit = c_model(x_cat)
        loss = nn.CrossEntropyLoss()(logit, target)
        #print(loss)
        loss_sum += loss
        loss.backward()
        # if epoch_num % 2 == 1:
        optimizer.step()
    return loss_sum

def test(vae_model, c_model, source_model):
    err_num = 0
    err_adv = 0
    c_model.eval()
    vae_model.eval()
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data = Variable(data.data, requires_grad=True)
        _,_,_,x_ = vae_model(data)
        logit = c_model(x_.view(-1,160,8,8))
        err_num += (logit.data.max(1)[1] != target.data).float().sum()
        x_adv = pgd_cifar_blackbox(vae_model, c_model, source_model, data, target, 20, 0.03, 0.003)
        # x_adv = pgd_cifar(vae_model, c_model, data, target, 20, 0.03, 0.003)
        # logit_adv = testtime_update_cifar(vae_model, c_model,  x_adv, target,learning_rate=0.05, num=50)
        # logit_adv = diff_update_cifar(vae_model,c_model, x_adv, target,learning_rate=0.05, num=50)
        _,_,_,x_adv_ = vae_model(x_adv)
        logit_adv = c_model(x_adv_.view(-1,160,8,8))
        # logit = c_model(x_.view(-1,160,8,8))
        adv_num = (logit_adv.data.max(1)[1] != target.data).float().sum()   
        # exit()
        print(adv_num)
        err_adv += adv_num
        # x_cat_adv = torch.cat((m_adv, log_adv), 1)
        # logit_adv = c_model(x_cat_adv)
        # err_adv += (logit_adv.data.max(1)[1] != target.data).float().sum()
    print(len(test_loader.dataset))
    print(err_num)
    print(err_adv)


def main():
    source_model = WideResNet().to(device)
    source_model_path = './blackbox_pgd_model/model-wideres-epoch76.pt'
    vae_model = wide_VAE(zDim=256).to(device)
    c_model = classifier().to(device)
    vae_model_path = './model-checkpoint/cifar-vae-model-89.pt'
    c_model_path = './model-checkpoint/cifar-c-model-89.pt'
    source_model.load_state_dict(torch.load(source_model_path))
    vae_model.load_state_dict(torch.load(vae_model_path))
    c_model.load_state_dict(torch.load(c_model_path))
    test(vae_model, c_model, source_model)

if __name__ == '__main__':
    main()