import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from model.VAE_update import *
from pgd_attack import *
import torch.optim as optim
import argparse
import numpy as np
from util import *
from loss_adv import *
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
parser = argparse.ArgumentParser(description='PyTorch MNIST VAE Training')
parser.add_argument('--batch-size', type=int, default=400, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=400, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--latent-dim', type=int, default=256)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--testtime-epochs', type=int, default=20)
parser.add_argument('--testtime-lr',  default=0.1)
parser.add_argument('--lr', default=0.001)
parser.add_argument('--beta', default=0.5)
parser.add_argument('--model-dir', default='./model-checkpoint')
parser.add_argument('--test-num', default=0)
args = parser.parse_args()

if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)
if not os.path.exists("./output-log"):
    os.makedirs("./output-log")
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
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

f=open("./output-log/mnist-adv_test.txt","a") 

def train(vae_model, c_model, data_loader, vae_optimizer, c_optimizer, epoch_num, channel):
    vae_model.train()
    c_model.train()
    v_loss_sum = 0
    c_loss_sum = 0
    r_loss_sum = 0
    kld_loss_sum = 0
    for batch_idx, (data, target) in enumerate(data_loader):

        data, target = data.to(device), target.to(device)
        vae_model.eval()
        c_model.eval()
        x_adv = pgd_mnist_new(vae_model, c_model, data, target, 40, 0.3, 0.01, channel)
        vae_model.train()
        c_model.train()
        vae_optimizer.zero_grad()
        c_optimizer.zero_grad()
        x_hat, mean, log_v, _ = vae_model(data)
        x_hat_adv, mean_adv, log_v_adv, _ = vae_model(x_adv)
        v_loss_nat = loss_function_adv(data, x_hat, mean, log_v)
        v_loss_adv = loss_function_adv(x_adv, x_hat_adv, mean_adv, log_v_adv)
        v_loss = v_loss_nat + v_loss_adv / args.batch_size * 0.5
        v_loss_sum += v_loss
        # if epoch_num % 2 == 1:
        v_loss.backward()
        vae_optimizer.step()
        # c_loss.backward()

        loss_adv = loss_adversarial_mnist(vae_model, c_model, c_optimizer, data,x_adv, target,channel)
        c_loss_sum += loss_adv
        loss_adv.backward()
        c_optimizer.step()

    print('Epoch {}: VAE Average loss: {:.6f}'.format(epoch_num, v_loss_sum/len(train_loader.dataset)), flush=True, file=f)
    print('Epoch {}: classifier Average loss: {:.6f}'.format(epoch_num, c_loss_sum/len(train_loader.dataset)), flush=True, file=f)


def eval_train(vae_model, c_model, channel):
    vae_model.eval()
    c_model.eval()
    err_num = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            _, _, _, x_ = vae_model(data)
            logit = c_model(x_.detach().view(-1,channel,7,7))
            err_num += (logit.data.max(1)[1] != target.data).float().sum()
    print('train error num:{}'.format(err_num), flush=True, file=f)

def eval_test(vae_model, c_model, channel):
    vae_model.eval()
    c_model.eval()
    err_num = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            _, _, _, x_ = vae_model(data)
            logit = c_model(x_.detach().view(-1,channel,7,7))
            err_num += (logit.data.max(1)[1] != target.data).float().sum()
    print('test error num:{}'.format(err_num), flush=True, file=f)

def test(vae_model, c_model, channel):
    err_num = 0
    err_adv = 0
    err_nat = 0
    c_model.eval()
    vae_model.eval()
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data = Variable(data.data, requires_grad=True)
        _,_,_,x_ = vae_model(data)
        logit = c_model(x_.view(-1,channel,7,7))
        err_nat += (logit.data.max(1)[1]!=target.data).float().sum()
        # logit_new = testtime_update_mnist_new(vae_model, c_model, data, target,learning_rate=0.05, num=40, mode = 'sum', channel=channel)
        logit_new = testtime_update_mnist_new_opt(vae_model, c_model, data, target,learning_rate=0.005, num=200, channel=channel,opti='adam', nc=0)
        label = logit_new.max(1)[1]
        # label = logit_calculate(logit, logit_new).to(device)
        err_num += (label.data != target.data).float().sum()

        x_adv = pgd_mnist_new(vae_model, c_model, data, target, 40, 0.3, 0.01, channel)
        _,_,_,x_ = vae_model(x_adv)
        logit_adv = c_model(x_.view(-1,channel,7,7))
        # logit_adv_new = testtime_update_mnist_new(vae_model, c_model, x_adv, target,learning_rate=0.05, num=40, mode = 'sum', channel=channel)
        logit_adv_new = testtime_update_mnist_new_opt(vae_model, c_model, x_adv, target,learning_rate=0.005, num=200, channel=channel,opti='adam', nc=0)
        label_adv = logit_adv_new.max(1)[1]
        # label_adv = logit_calculate(logit_adv, logit_adv_new).to(device)
        err_adv += (label_adv.data != target.data).float().sum()
        
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
    if epoch >= 150:
        lr_ = lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_

def main():
    channel = [16,60,120]
    vae_model = VAE(zDim = 32, channel=channel).to(device)
    vae_optimizer = optim.Adam(vae_model.parameters(), lr=args.lr)
    c_model = classifier(input_dim=channel[2]).to(device)
    c_optimizer = optim.SGD(c_model.parameters(), lr=0.01, momentum=0.9)
    print(len(train_loader.dataset))
    if args.test_num == 0:
        print('training mode')
        for epoch in range(1, args.epochs+1):
            adjust_learning_rate(c_optimizer, epoch, 0.01)
            train(vae_model,c_model, train_loader, vae_optimizer, c_optimizer, epoch, channel[2])
            eval_train(vae_model, c_model, channel[2])
            eval_test(vae_model, c_model, channel[2])
            print('==================================================', flush=True, file=f)
            if epoch % 10 == 0:
                torch.save(vae_model.state_dict(), os.path.join(args.model_dir, 'mnist-vae-model-adv-{}.pt'.format(epoch)))
                torch.save(c_model.state_dict(), os.path.join(args.model_dir, 'mnist-c-model-adv-{}.pt'.format(epoch)))
    else:
        print('testing mode')
        vae_model_path = '{}/mnist-vae-model-adv-{}.pt'.format(args.model_dir, args.test_num)
        c_model_path = '{}/mnist-c-model-adv-{}.pt'.format(args.model_dir, args.test_num)
        vae_model.load_state_dict(torch.load(vae_model_path))
        c_model.load_state_dict(torch.load(c_model_path))
        test(vae_model, c_model, channel[2])
    

if __name__ == '__main__':
    main()