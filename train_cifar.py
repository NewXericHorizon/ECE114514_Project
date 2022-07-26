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
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 VAE Training')
parser.add_argument('--batch-size', type=int, default=400, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=400, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--latent-dim', type=int, default=256)
parser.add_argument('--epochs', type=int, default=130)
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

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
# f=open("./output-log/cifar_test.txt","a")

trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

def train(vae_model, c_model, data_loader, vae_optimizer, c_optimizer, epoch_num, channel=128):
    torch.autograd.set_detect_anomaly(True)
    vae_model.train()
    c_model.train()
    v_loss_sum = 0
    c_loss_sum = 0
    r_loss_sum = 0
    kld_loss_sum = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        #data = data.view(batch_size, x_dim)
        data, target = data.to(device), target.to(device)
        vae_optimizer.zero_grad()
        c_optimizer.zero_grad()
        x_hat, mean, log_v, x_ = vae_model(data)
        # x_cat = torch.cat((mean, log_v),1)
        logit = c_model(x_.view(-1,channel,8,8))
        #logit = c_model(x_cat)
        v_loss, c_loss, r_loss, kld_loss = loss_cal(data, target, x_hat, mean, log_v, logit, vae_optimizer, c_optimizer, epoch_num)

        r_loss_sum += r_loss
        kld_loss_sum += kld_loss
        v_loss_sum += v_loss
        c_loss_sum += c_loss

        # if epoch_num % 2 == 0:
        #     v_loss.backward()
        #     vae_optimizer.step()
        #     c_loss.backward()
        # else:
        #     c_loss.backward()
        #     c_optimizer.step()
        #     v_loss.backward()

    return v_loss_sum, c_loss_sum, r_loss_sum, kld_loss_sum

def loss_cal(data, target, x_hat, mean, log_v, logit, vae_optimizer, c_optimizer, epoch_num):
    v_loss, c_loss, r_loss, kld_loss = loss_function_mean(data, target, x_hat, mean, log_v, logit)
    a = 0.7
    loss = v_loss * a + c_loss * (1-a)
    # if epoch_num % 2 == 0:
    #     v_loss.backward()
    #     vae_optimizer.step()
    #     # c_loss.backward()
    # else:
    #     c_loss.backward()
    #     c_optimizer.step()
    #     # v_loss.backward()
    loss.backward()
    vae_optimizer.step()
    c_optimizer.step()
    return v_loss.detach(), c_loss.detach(), r_loss.detach(), kld_loss.detach()
    # return 0,0,0,0

def eval_train(vae_model, c_model, train_loader, channel):
    vae_model.eval()
    c_model.eval()
    err_num = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            x_hat, mean, log_v, x_ = vae_model(data)
            logit = c_model(x_.detach().view(-1,channel,8,8))
            err_num += (logit.data.max(1)[1] != target.data).float().sum()
    print('train error num:{}'.format(err_num))

def eval_test(vae_model, c_model, channel):
    vae_model.eval()
    c_model.eval()
    err_num = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            x_hat, mean, log_v, x_ = vae_model(data)
            logit = c_model(x_.detach().view(-1,channel,8,8))
            err_num += (logit.data.max(1)[1] != target.data).float().sum()
    print('test error num:{}'.format(err_num))

def test(vae_model, c_model, channel=128, lr=0.01, num=10):
    err_num = 0
    err_adv = 0
    err_nat = 0
    c_model.eval()
    vae_model.eval()
    t = False
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data = Variable(data.data, requires_grad=True)
        _,_,_,x_ = vae_model(data)
        logit = c_model(x_.view(-1,channel,8,8))
        err_nat += (logit.data.max(1)[1] != target.data).float().sum()
        logit_new = testtime_update_cifar_opt(vae_model, c_model, data, target,learning_rate=lr, num=num, channel=channel, opti='adam', nc=0)
        # logit_new = testtime_update_cifar(vae_model, c_model, data, target,learning_rate=0.1, num=100, channel=channel)
        # label = logit_calculate(logit, logit_new).to(device)
        label = logit_new.data.max(1)[1]
        err_num += (label.data != target.data).float().sum()
        x_adv = pgd_cifar(vae_model, c_model, data, target, 20, 0.03, 0.003, channel=channel)
        # x_adv = pgd_cifar_blackbox(vae_model, c_model, source_model, data, target, 20, 0.03, 0.003)
        _,_,_,x_ = vae_model(x_adv)
        logit_adv = c_model(x_.view(-1,channel,8,8))
        logit_adv_new = testtime_update_cifar_opt(vae_model, c_model, x_adv, target,learning_rate=lr, num=num, channel=channel, opti='adam', nc=0)
        # logit_adv_new = testtime_update_cifar(vae_model, c_model, x_adv, target,learning_rate=0.1, num=100, channel=channel)
        # label_adv = logit_calculate(logit_adv, logit_adv_new).to(device)
        label_adv = logit_adv_new.max(1)[1]
        err_adv += (label_adv.data != target.data).float().sum()
        # err_adv += (logit_adv.data.max(1)[1] != target.data).float().sum()

    print(len(test_loader.dataset))
    print(err_nat)
    print(err_num)
    print(err_adv)

def adjust_learning_rate(optimizer, epoch, lr):
    """decrease the learning rate"""
    lr_ = lr
    if epoch >= 120:
        lr_ = lr * 0.1
    if epoch >= 130:
        lr_ = lr * 0.01
    if epoch >= 130:
        lr_ = lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_

def main():
    channel = [16,80,160]
    vae_model = wide_VAE(zDim=256, channel=channel).to(device)
    vae_optimizer = optim.Adam(vae_model.parameters(), lr=args.lr)
    c_model = classifier(input_dim=channel[2]).to(device)
    # c_para = list(vae_model.extract_layer_en().parameters())+list(c_model.parameters())
    c_optimizer = optim.Adam(c_model.parameters(), lr=args.lr*10)
    if args.test_num == 0:
        print('training mode')
        for epoch in range(1, args.epochs+1):

            adjust_learning_rate(c_optimizer, epoch, args.lr*10)
            v_loss, c_loss, r_loss, k_loss = train(vae_model,c_model, train_loader, vae_optimizer, c_optimizer, epoch,channel=channel[2])
            print('Epoch {}: reconstruction Average loss: {:.6f}'.format(epoch, r_loss/len(train_loader.dataset)))
            print('Epoch {}: KLD Average loss: {:.6f}'.format(epoch, k_loss/len(train_loader.dataset)))
            print('Epoch {}: VAE Average loss: {:.6f}'.format(epoch, v_loss/len(train_loader.dataset)))
            print('Epoch {}: Classifier Average loss: {:.6f}'.format(epoch, c_loss/len(train_loader.dataset)))
            eval_train(vae_model, c_model, train_loader, channel[2])
            eval_test(vae_model, c_model, channel[2])
            print('==================================================')
            if epoch >= 120:
                torch.save(vae_model.state_dict(), os.path.join(args.model_dir, 'cifar-vae-model-{}.pt'.format(epoch)))
                torch.save(c_model.state_dict(), os.path.join(args.model_dir, 'cifar-c-model-{}.pt'.format(epoch)))
    else:
        print('testing mode')
        vae_model_path = '{}/cifar-vae-model-{}.pt'.format(args.model_dir, args.test_num)
        c_model_path = '{}/cifar-c-model-{}.pt'.format(args.model_dir, args.test_num)
        vae_model.load_state_dict(torch.load(vae_model_path))
        c_model.load_state_dict(torch.load(c_model_path))
        test(vae_model, c_model, channel=channel[2], lr=0.01, num=70)
    

if __name__ == '__main__':
    main()