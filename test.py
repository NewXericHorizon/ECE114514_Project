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
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 VAE Training')
parser.add_argument('--batch-size', type=int, default=400, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=400, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--x-dim', type=int, default=784)
parser.add_argument('--hidden-dim', type=int, default=400)
parser.add_argument('--latent-dim', type=int, default=200)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--lr', default=0.001)
args = parser.parse_args()

torch.manual_seed(1)

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
batch_size = 100
test_batch_size = 200
x_dim = 784
latent_dim = 200
epochs = 40
trainset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform_test)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)
testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, **kwargs)


def loss_function(x, label, x_hat, mean, log_var, logit):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    vae_loss = reproduction_loss + KLD
    c_loss = nn.CrossEntropyLoss()(logit, label)
    beta = 0.4
    loss = vae_loss * beta + c_loss * (1 - beta) 
    return loss

def vae_loss(x, x_hat, mean, log_var):
    #x_hat[x_hat != x_hat] = 0
    if((x_hat != x_hat).sum().item() > 0):
      print((x_hat != x_hat).sum().item())
      temp = np.array(x_hat.squeeze().detach().cpu())
      temp = temp.reshape((200, 28*28))
      np.savetxt('test.txt',temp)
      
      exit()
    reproduction_loss = nn.BCELoss(reduction='sum')(x_hat, x)
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    vae_loss = reproduction_loss + KLD
    return vae_loss

def train(vae_model, c_model, data_loader, vae_optimizer, c_optimizer, epoch_num):
    vae_model.train()
    c_model.train()
    loss_sum = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        #data = data.view(batch_size, x_dim)
        data, target = data.to(device), target.to(device)
        vae_optimizer.zero_grad()
        c_optimizer.zero_grad()
        x_hat, mean, log_v, x_ = vae_model(data)
        x_cat = torch.cat((mean, log_v),1)
        logit = c_model(x_cat.detach())
        loss = loss_function(data, target, x_hat, mean, log_v, logit)
        loss_sum += loss
        loss.backward()
        # if epoch_num % 2 == 1:
        c_optimizer.step()
        vae_optimizer.step()
    return loss_sum

def model_pred(x, vae_model, c_model):
    x_hat, mean, log_v, x_ = vae_model(x)
    x_cat = torch.cat((mean, log_v), 1)
    logit = c_model(x_cat)
    return logit, mean, log_v

def testtime_update(vae_model, x_adv, learning_rate=0.1, num = 40):
    x_hat_adv, mean, log_v, x_ = vae_model(x_adv)
    x_adv = x_adv.detach()
    for _ in range(num):
        if ((x_hat_adv != x_hat_adv).sum() > 0):
            print((x_hat_adv != x_hat_adv).sum())
            print(mean)
            print(log_v)
            exit()
        loss = vae_loss(x_adv, x_hat_adv, mean, log_v)
        mean.retain_grad()
        log_v.retain_grad()
        loss.backward(retain_graph=True)
        with torch.no_grad():
            mean.data -= learning_rate * mean.grad.data
            log_v.data -= learning_rate * log_v.grad.data
        mean.grad.data.zero_()
        log_v.grad.data.zero_()
        x_hat_adv = vae_model.decoder(vae_model.reparameterize(mean, log_v))
    return mean, log_v

def test(vae_model, c_model):
    err_num = 0
    err_adv = 0
    c_model.eval()
    vae_model.eval()
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        logit, _, _ = model_pred(data, vae_model, c_model)
        err_num += (logit.data.max(1)[1] != target.data).float().sum()
        x_adv = pgd(vae_model, c_model, data, target, 40, 0.3, 0.01)
        m_adv, log_adv = testtime_update(vae_model, x_adv)
        x_cat_adv = torch.cat((m_adv, log_adv), 1)
        logit_adv = c_model(x_cat_adv)
        err_adv += (logit_adv.data.max(1)[1] != target.data).float().sum()
    print(len(test_loader.dataset))
    print(err_num)
    print(err_adv)

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
    c_model = classifier(input_dim=256*2).to(device)
    c_optimizer = optim.Adam(c_model.parameters(), lr=args.lr)
    print(len(train_loader.dataset))
    for epoch in range(1, epochs+1):
        loss = train(vae_model,c_model, train_loader, vae_optimizer, c_optimizer, epoch)
        print('Epoch {}: Average loss: {:.6f}'.format(epoch, loss/len(train_loader.dataset)))
    test(vae_model, c_model)
    

if __name__ == '__main__':
    main()