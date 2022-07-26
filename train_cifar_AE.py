import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from model.AE import *
from blackbox_pgd_model.wideresnet_update import *
from pgd_attack import *
import torch.optim as optim
import argparse
import numpy as np
from util import *
import os
from torchinfo import summary
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 VAE Training')
parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--latent-dim', type=int, default=1024)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--testtime-epochs', type=int, default=20)
parser.add_argument('--testtime-lr',  default=0.1)
parser.add_argument('--lr', default=0.001)
parser.add_argument('--beta', default=0.5)
parser.add_argument('--test', action='store_true')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
torch.manual_seed(1)
torch.cuda.manual_seed(1)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.5,),(0.5,)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

def train(v_model, c_model, data_loader, v_optimizer, c_optimizer):
    v_model.train()
    c_model.train()
    bce_loss_sum = 0
    mse_loss_sum = 0
    c_loss_sum = 0
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        v_optimizer.zero_grad()
        c_optimizer.zero_grad()
        x_hat, x_ = v_model(data)
        logit = c_model(x_.detach())
        c_loss = F.cross_entropy(logit, target, size_average=False)
        mse_loss = F.mse_loss(x_hat, data, reduction='mean')
        loss = F.binary_cross_entropy(x_hat, data, size_average=False, reduction='mean')
        bce_loss = nn.BCELoss()(x_hat, data)
        bce_loss_sum += bce_loss
        mse_loss_sum += mse_loss
        c_loss_sum += c_loss
        #-----------------------
        a = 1
        loss = a * c_loss + mse_loss
        loss.backward()
        #-----------------------
        v_optimizer.step()
        c_optimizer.step()
    return bce_loss_sum, mse_loss_sum, c_loss_sum

def eval_train(vae_model, c_model):
    vae_model.eval()
    c_model.eval()
    err_num = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            _, x_ = vae_model(data)
            logit = c_model(x_)
            err_num += (logit.data.max(1)[1] != target.data).float().sum()
    print('train error num:{}'.format(err_num))

def eval_test(vae_model, c_model):
    vae_model.eval()
    c_model.eval()
    err_num = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            _, x_ = vae_model(data)
            logit = c_model(x_)
            err_num += (logit.data.max(1)[1] != target.data).float().sum()
    print('test error num:{}'.format(err_num))

def test(vae_model, c_model, lr=0.01, num=10):
    err_num = 0
    err_adv = 0
    err_nat = 0
    c_model.eval()
    vae_model.eval()
    t = False
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data = Variable(data.data, requires_grad=True)
        _,x_ = vae_model(data)
        logit = c_model(x_)
        err_nat += (logit.data.max(1)[1] != target.data).float().sum()
        logit_new = testtime_update_cifar_AE(vae_model, c_model, data, target,learning_rate=lr, num=num, opti='adam', nc=0)
        # logit_new = testtime_update_cifar(vae_model, c_model, data, target,learning_rate=0.1, num=100, channel=channel)
        label = logit_calculate(logit, logit_new).to(device)
        # label = logit_new.data.max(1)[1]
        err_num += (label.data != target.data).float().sum()
        x_adv = pgd_cifar_AE(vae_model, c_model, data, target, 20, 0.03, 0.003)
        # x_adv = pgd_cifar_blackbox(vae_model, c_model, source_model, data, target, 20, 0.03, 0.003)
        _,x_ = vae_model(x_adv)
        logit_adv = c_model(x_)
        logit_adv_new = testtime_update_cifar_AE(vae_model, c_model, x_adv, target,learning_rate=lr, num=num, opti='adam', nc=0)
        # logit_adv_new = testtime_update_cifar(vae_model, c_model, x_adv, target,learning_rate=0.1, num=100, channel=channel)
        label_adv = logit_calculate(logit_adv, logit_adv_new).to(device)
        # label_adv = logit_adv_new.max(1)[1]
        err_adv += (label_adv.data != target.data).float().sum()
        # err_adv += (logit_adv.data.max(1)[1] != target.data).float().sum()

    print(len(test_loader.dataset))
    print(err_nat)
    print(err_num)
    print(err_adv)

def adjust_learning_rate(optimizer, epoch, lr):
    """decrease the learning rate"""
    lr_ = lr
    if epoch >= 80:
        lr_ = lr * 0.1
    if epoch >= 100:
        lr_ = lr * 0.01
    if epoch >= 130:
        lr_ = lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_

def main():
    channel=[16, 80, 160]
    v_model = wide_VAE(zDim=512, channel=channel).to(device)
    v_optimizer = optim.Adam(v_model.parameters(), lr=args.lr)
    c_model = classifier(input_dim=channel[2]).to(device)
    c_optimizer = optim.Adam(c_model.parameters(), lr=args.lr)
    if args.test:
        print('test mode')
        v_model.load_state_dict(torch.load('./model-checkpoint/AE-v-model.pt'))
        c_model.load_state_dict(torch.load('./model-checkpoint/AE-c-model.pt'))
        test(v_model, c_model, lr=0.005, num=200)
    else:
        for i in range(1, args.epochs + 1):
            adjust_learning_rate(v_optimizer, i, args.lr)
            adjust_learning_rate(c_optimizer, i, args.lr)
            bceloss, mseloss,closs = train(v_model,c_model, train_loader, v_optimizer, c_optimizer)
            print('Epoch {}, BCE loss: {:.6f}, MSE loss: {:.6f}, classifier loss: {:.6f}'.format(i, bceloss, mseloss, closs))
            eval_train(v_model, c_model)
            eval_test(v_model, c_model)
        torch.save(v_model.state_dict(),  './model-checkpoint/AE-v-model.pt')
        torch.save(c_model.state_dict(),  './model-checkpoint/AE-c-model.pt')

if __name__ == '__main__':
    main()