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
import os
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 VAE Training')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--latent-dim', type=int, default=512)
parser.add_argument('--epochs', type=int, default=90)
parser.add_argument('--testtime-epochs', type=int, default=20)
parser.add_argument('--testtime-lr',  default=0.1)
parser.add_argument('--lr', default=0.001)
parser.add_argument('--beta', default=0.5)
parser.add_argument('--model-dir', default='./model-checkpoint')
parser.add_argument('--test-num', default=0)
args = parser.parse_args()

if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
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

def train(vae_model, c_model, data_loader, vae_optimizer, c_optimizer, epoch_num):
    vae_model.train()
    c_model.train()
    v_loss_sum = 0
    c_loss_sum = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        #data = data.view(batch_size, x_dim)
        data, target = data.to(device), target.to(device)
        vae_optimizer.zero_grad()
        c_optimizer.zero_grad()
        x_hat, mean, log_v, x_ = vae_model(data)
        # x_cat = torch.cat((mean, log_v),1)
        logit = c_model(x_.detach())
        #logit = c_model(x_cat)
        v_loss, c_loss = loss_function(data, target, x_hat, mean, log_v, logit)
        #print(loss)
        v_loss_sum += v_loss
        c_loss_sum += c_loss
        # if epoch_num % 2 == 1:
        if epoch_num <= 30:
            v_loss.backward()
            vae_optimizer.step()
            c_loss.backward()
        else:
            c_loss.backward()
            c_optimizer.step()
            v_loss.backward()
    return v_loss_sum, c_loss_sum

def test(vae_model, c_model):
    err_num = 0
    err_adv = 0
    c_model.eval()
    vae_model.eval()
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        logit = model_pred(data, vae_model, c_model)
        err_num += (logit.data.max(1)[1] != target.data).float().sum()
        # x_adv = pgd(vae_model, c_model, data, target, 40, 0.3, 0.01)
        # m_adv, log_adv = testtime_update(vae_model, x_adv,learning_rate=args.testtime_lr, num=args.testtime_epochs)
        # x_cat_adv = torch.cat((m_adv, log_adv), 1)
        # logit_adv = c_model(x_cat_adv)
        # err_adv += (logit_adv.data.max(1)[1] != target.data).float().sum()
    print(len(test_loader.dataset))
    print(err_num)
    # print(err_adv)

def adjust_learning_rate(optimizer, epoch, lr):
    """decrease the learning rate"""
    lr_ = lr
    if epoch >= 80:
        lr_ = lr * 0.1
    if epoch >= 90:
        lr_ = lr * 0.01
    if epoch >= 100:
        lr_ = lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_

def main():
    vae_model = wide_VAE(zDim=args.latent_dim).to(device)
    vae_optimizer = optim.Adam(vae_model.parameters(), lr=args.lr)
    c_model = classifier().to(device)
    c_optimizer = optim.Adam(c_model.parameters(), lr=args.lr)
    print(len(train_loader.dataset))
    if args.test_num == 0:
        print('training mode')
        for epoch in range(1, args.epochs+1):
            adjust_learning_rate(c_optimizer, epoch, args.lr)
            v_loss, c_loss = train(vae_model,c_model, train_loader, vae_optimizer, c_optimizer, epoch)
            print('Epoch {}: VAE Average loss: {:.6f}'.format(epoch, v_loss/len(train_loader.dataset)))
            print('Epoch {}: Classifier Average loss: {:.6f}'.format(epoch, c_loss/len(train_loader.dataset)))
            if epoch > 75:
                torch.save(vae_model.state_dict(), os.path.join(args.model_dir, 'cifar-vae-model-{}.pt'.format(epoch)))
                torch.save(c_model.state_dict(), os.path.join(args.model_dir, 'cifar-c-model-{}.pt'.format(epoch)))
    else:
        print('testing mode')
        vae_model_path = '{}/cifar-vae-model-{}.pt'.format(args.model_dir, args.test_num)
        c_model_path = '{}/cifar-c-model-{}.pt'.format(args.model_dir, args.test_num)
        vae_model.load_state_dict(torch.load(vae_model_path))
        c_model.load_state_dict(torch.load(c_model_path))
        test(vae_model, c_model)
    

if __name__ == '__main__':
    main()