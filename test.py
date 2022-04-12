import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from model.wideVAE import *
from pgd_attack import *
import torch.optim as optim
import argparse
import numpy as np
from util import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 VAE Training')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
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

def test(model):
    err_num = 0
    err_adv = 0
    model.eval()
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        logit = model(data)
        err_num += (logit.data.max(1)[1] != target.data).float().sum()
        # x_adv = pgd(vae_model, c_model, data, target, 40, 0.3, 0.01)
        # m_adv, log_adv = testtime_update(vae_model, x_adv,learning_rate=args.testtime_lr, num=args.testtime_epochs)
        # x_cat_adv = torch.cat((m_adv, log_adv), 1)
        # logit_adv = c_model(x_cat_adv)
        # err_adv += (logit_adv.data.max(1)[1] != target.data).float().sum()
    print(len(test_loader.dataset))
    print(err_num)

def adjust_learning_rate(vae_optimizer,c_optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 65:
        lr = args.lr * 0.1
    if epoch >= 80:
        lr = args.lr * 0.01
    if epoch >= 90:
        lr = args.lr * 0.001
    for param_group in c_optimizer.param_groups:
        param_group['lr'] = lr
    for param_group in vae_optimizer.param_groups:
        param_group['lr'] = lr

def main():
    model = test_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(len(train_loader.dataset))
    for epoch in range(1, args.epochs+1):
        loss = train(model, train_loader, optimizer, epoch)
        print('Epoch {}: Average loss: {:.6f}'.format(epoch, loss/len(train_loader.dataset)))
    test(model)
    

if __name__ == '__main__':
    main()