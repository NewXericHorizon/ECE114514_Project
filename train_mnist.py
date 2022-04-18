from cmath import log
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from model.VAE import *
from pgd_attack import *
import torch.optim as optim
import argparse
import numpy as np
from util import *
import os
parser = argparse.ArgumentParser(description='PyTorch MNIST VAE Training')
parser.add_argument('--batch-size', type=int, default=400, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=400, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--x-dim', type=int, default=784)
parser.add_argument('--hidden-dim', type=int, default=400)
parser.add_argument('--latent-dim', type=int, default=256)
parser.add_argument('--epochs', type=int, default=55)
parser.add_argument('--testtime-epochs', type=int, default=20)
parser.add_argument('--testtime-lr',  default=0.1)
parser.add_argument('--lr', default=0.001)
parser.add_argument('--model-dir', default='./model-checkpoint')
parser.add_argument('--beta', default=0.5)
parser.add_argument('--test-num', default=0)
args = parser.parse_args()

torch.manual_seed(1)
if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform_test)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform_test)
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
        x_cat = torch.cat((mean, log_v), 1)
        logit = c_model(x_cat.detach())
        #logit = c_model(x_cat)
        v_loss, c_loss,_,_ = loss_function_sum(data, target, x_hat, mean, log_v, logit, args.beta)
        v_loss_sum += v_loss
        c_loss_sum += c_loss
        
        if epoch_num <= 20:
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
        _,mean,log_v,x_ = vae_model(data)
        x_cat = torch.cat((mean, log_v), 1)
        logit = c_model(x_cat)

        err_num += (logit.data.max(1)[1] != target.data).float().sum()
        x_adv = pgd_mnist(vae_model, c_model, data, target, 40, 0.3, 0.01)
        logit_adv = testtime_update_mnist(vae_model, c_model, x_adv, learning_rate=0.1, num=20, mode = 'sum')
        # logit_adv = model_pred(x_adv, vae_model, c_model)
        # logit_adv = testtime_update_mnist(vae_model, c_model, x_adv, learning_rate=0.1, num=100, mode = 'mean')
        adv_num = (logit_adv.data.max(1)[1] != target.data).float().sum()
        print(adv_num)
        err_adv += adv_num
    print(len(test_loader.dataset))
    print(err_num)
    print(err_adv)

def eval_train(vae_model, c_model):
    vae_model.eval()
    c_model.eval()
    err_num = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            _, mean, log_v, x_ = vae_model(data)
            x_cat = torch.cat((mean, log_v), 1)
            logit = c_model(x_cat)
            err_num += (logit.data.max(1)[1] != target.data).float().sum()
    print('train error num:{}'.format(err_num))

def eval_test(vae_model, c_model):
    vae_model.eval()
    c_model.eval()
    err_num = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            _,mean,log_v, _ = vae_model(data)
            x_cat = torch.cat((mean, log_v), 1)
            logit = c_model(x_cat)
            err_num += (logit.data.max(1)[1] != target.data).float().sum()
    print('test error num:{}'.format(err_num))

def adjust_learning_rate(vae_optimizer,c_optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 55:
        lr = args.lr * 0.1
    if epoch >= 75:
        lr = args.lr * 0.01
    if epoch >= 90:
        lr = args.lr * 0.001
    for param_group in c_optimizer.param_groups:
        param_group['lr'] = lr
    for param_group in vae_optimizer.param_groups:
        param_group['lr'] = lr

def main():
    vae_model = VAE().to(device)
    vae_optimizer = optim.Adam(vae_model.parameters(), lr=args.lr)
    c_model = classifier().to(device)
    c_optimizer = optim.Adam(c_model.parameters(), lr=args.lr)
    if args.test_num == 0:
        print(len(train_loader.dataset))
        for epoch in range(1, args.epochs+1):
            v_loss, c_loss = train(vae_model,c_model, train_loader, vae_optimizer, c_optimizer, epoch)
            print('Epoch {}: VAE Average loss: {:.6f}'.format(epoch, v_loss/len(train_loader.dataset)))
            print('Epoch {}: classifier Average loss: {:.6f}'.format(epoch, c_loss/len(train_loader.dataset)))
            eval_train(vae_model, c_model)
            eval_test(vae_model, c_model)
            if epoch > 50:
                torch.save(vae_model.state_dict(), os.path.join(args.model_dir, 'mnist-vae-model-{}.pt'.format(epoch)))
                torch.save(c_model.state_dict(), os.path.join(args.model_dir, 'mnist-c-model-{}.pt'.format(epoch)))
        
    else:
        print('================================')
        print('testing')
        vae_model_path = '{}/mnist-vae-model-{}.pt'.format(args.model_dir, args.test_num)
        c_model_path = '{}/mnist-c-model-{}.pt'.format(args.model_dir, args.test_num)
        vae_model.load_state_dict(torch.load(vae_model_path))
        c_model.load_state_dict(torch.load(c_model_path))
        test(vae_model, c_model)
    

if __name__ == '__main__':
    main()