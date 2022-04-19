import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def loss_function_mean(x, label, x_hat, mean, log_var, logit, beta = 0.5):
    v_loss, reproduction_loss, KLD = vae_loss_mean(x, x_hat, mean, log_var)
    c_loss = F.cross_entropy(logit, label, size_average=False)

    return v_loss, c_loss, reproduction_loss, KLD

def loss_function_sum(x, label, x_hat, mean, log_var, logit, beta = 0.5):
    v_loss, reproduction_loss, KLD = vae_loss_sum(x, x_hat, mean, log_var)
    c_loss = nn.CrossEntropyLoss(size_average=False)(logit, label)

    return v_loss, c_loss, reproduction_loss, KLD

def vae_loss_mean(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, size_average=False, reduction='mean')
    KLD      = - 0.5 * torch.mean(1+ log_var - mean.pow(2) - log_var.exp())
    v_loss = reproduction_loss + KLD
    return v_loss, reproduction_loss, KLD

def vae_loss_sum(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    v_loss = reproduction_loss + KLD
    return v_loss, reproduction_loss, KLD


def testtime_update_mnist(vae_model, c_model, x_adv, learning_rate=0.1, num = 30, mode = 'mean'):
    x_adv = x_adv.detach()
    x_hat_adv, mean, log_v, x_ = vae_model(x_adv)
    for _ in range(num):
        if (x_hat_adv != x_hat_adv).sum() > 0:
            print('nan Error')
            exit()
        if mode == 'mean':
            loss = nn.functional.binary_cross_entropy(x_hat_adv, x_adv, size_average=False, reduction='mean')
            # loss = vae_loss_mean(x_adv, x_hat_adv, mean, log_v)
        else:
            loss = nn.functional.binary_cross_entropy(x_hat_adv, x_adv, reduction='sum')
            # loss = vae_loss_sum(x_adv, x_hat_adv, mean, log_v)
        # x_.retain_grad()
        mean.retain_grad()
        log_v.retain_grad()
        loss.backward(retain_graph=True)
        with torch.no_grad():
            # x_.data -= learning_rate * x_.grad.data
            mean.data -= learning_rate * mean.grad.data
            log_v.data -= learning_rate * log_v.grad.data
        # x_.grad.data.zero_()
        mean.grad.data.zero_()
        log_v.grad.data.zero_()
        x_hat_adv = vae_model.decoder(vae_model.reparameterize(mean, log_v))
    x_cat = torch.cat((mean, log_v), 1)
    logit_adv = c_model(x_cat)
    return logit_adv

def testtime_update_mnist_new(vae_model, c_model, x_adv, target, learning_rate=0.1, num = 30, mode = 'mean'):
    x_adv = x_adv.detach()
    x_hat_adv, _, _, x_ = vae_model(x_adv)
    for _ in range(num):
        c = 0
        while (x_hat_adv != x_hat_adv).sum() > 0:
            x_hat_adv = vae_model.re_forward(x_)
            c += 1
            if c >= 100:
                print('nan Error', c)
                exit()
        if mode == 'mean':
            loss = nn.functional.binary_cross_entropy(x_hat_adv, x_adv, size_average=False, reduction='mean')
        else:
            loss = nn.functional.binary_cross_entropy(x_hat_adv, x_adv, reduction='sum')
        x_.retain_grad()
        loss.backward(retain_graph=True)
        with torch.no_grad():
            x_.data -= learning_rate * x_.grad.data
        x_.grad.data.zero_()
        x_hat_adv = vae_model.re_forward(x_)
 
    logit_adv = c_model(x_.view(-1,120,7,7))
    return logit_adv

def testtime_update_cifar_opt(vae_model, c_model, x_adv, target, learning_rate=0.1, num = 30, mode = 'mean', channel=128, opti = 'adam'):
    x_adv = x_adv.detach()
    x_hat_adv, _, _, x_ = vae_model(x_adv)
    x_copy = x_.detach().clone()
    if opti == 'adam':
        opt = optim.Adam([x_copy], lr=learning_rate)
    elif opti == 'sgd':
        opt = optim.SGD([x_copy], lr=learning_rate, momentum=0.9)
    for _ in range(num):
        # opt.zero_grad()
        if (x_hat_adv != x_hat_adv).sum() > 0:
            print('nan Error')
            exit()
        if mode == 'mean':
            loss = nn.functional.binary_cross_entropy(x_hat_adv, x_adv, size_average=False, reduction='mean')
        else:
            loss = nn.functional.binary_cross_entropy(x_hat_adv, x_adv, reduction='sum')
        x_.retain_grad()
        loss.backward(retain_graph=True)
        grad = torch.autograd.grad(loss, x_)
        # exit()
        with torch.no_grad():
            x_copy.grad = grad[0]
        # with torch.no_grad():
        #     x_.data -= learning_rate * x_.grad.data
        x_.grad.data.zero_()
        opt.step()
        x_ = x_copy.detach().clone()
        x_.requires_grad = True
        x_hat_adv = vae_model.re_forward(x_)
    logit_adv = c_model(x_.view(-1,channel,8,8))
    return logit_adv

def testtime_update_cifar(vae_model, c_model, x_adv, target, learning_rate=0.1, num = 30, mode = 'mean', channel=128):
    x_adv = x_adv.detach()
    x_hat_adv, _, _, x_ = vae_model(x_adv)
    for _ in range(num):
        if (x_hat_adv != x_hat_adv).sum() > 0:
            print('nan Error')
            exit()
        if mode == 'mean':
            loss = nn.functional.binary_cross_entropy(x_hat_adv, x_adv, size_average=False, reduction='mean')
        else:
            loss = nn.functional.binary_cross_entropy(x_hat_adv, x_adv, reduction='sum')
        x_.retain_grad()
        loss.backward(retain_graph=True)
        with torch.no_grad():
            x_.data -= learning_rate * x_.grad.data
        x_.grad.data.zero_()
        x_hat_adv = vae_model.re_forward(x_)
    logit_adv = c_model(x_.view(-1,channel,8,8))
    return logit_adv



def diff_update_cifar(vae_model, c_model, x_adv, target, learning_rate=0.1, num = 30, mode = 'mean'):
    x_adv = x_adv.detach()
    x_hat_adv, _, _, x_ = vae_model(x_adv)
    logit_nat = c_model(x_.view(-1,160,8,8))
    if (x_hat_adv != x_hat_adv).sum() > 0:
        print('PGD nan Error')
        exit()
    for _ in range(num):
        if (x_hat_adv != x_hat_adv).sum() > 0:
            print('nan Error')
            exit()
        if mode == 'mean':
            loss = nn.functional.binary_cross_entropy(x_hat_adv, x_adv, size_average=False, reduction='mean')
            # loss = vae_loss_mean(x_adv, x_hat_adv, mean, log_v)
        else:
            loss = nn.functional.binary_cross_entropy(x_hat_adv, x_adv, reduction='sum')
            # loss = vae_loss_sum(x_adv, x_hat_adv, mean, log_v)
        x_.retain_grad()
        loss.backward(retain_graph=True)
        with torch.no_grad():
            x_.data -= learning_rate * x_.grad.data
        x_.grad.data.zero_()
        # x_hat_adv = vae_model.re_forward(x_)
    _, _, _, x_ = vae_model(x_hat_adv)
    logit_final = c_model(x_.view(-1,160,8,8))
    logit_diff = logit_final - logit_nat
        # print((logit_diff.data.max(1)[1] != target.data).float().sum())
    return logit_diff

def diff_update_mnist(vae_model, c_model, x_adv, target, learning_rate=0.1, num = 30, mode = 'mean'):
    x_adv = x_adv.detach()
    x_hat_adv, _, _, x_ = vae_model(x_adv)
    logit_nat = c_model(x_.view(-1,120,7,7))
    if (x_hat_adv != x_hat_adv).sum() > 0:
        print('PGD nan Error')
        exit()
    for _ in range(num):
        if (x_hat_adv != x_hat_adv).sum() > 0:
            print('nan Error')
            exit()
        if mode == 'mean':
            loss = nn.functional.binary_cross_entropy(x_hat_adv, x_adv, size_average=False, reduction='mean')
            # loss = vae_loss_mean(x_adv, x_hat_adv, mean, log_v)
        else:
            loss = nn.functional.binary_cross_entropy(x_hat_adv, x_adv, reduction='sum')
            # loss = vae_loss_sum(x_adv, x_hat_adv, mean, log_v)
        x_.retain_grad()
        loss.backward(retain_graph=True)
        with torch.no_grad():
            x_.data -= learning_rate * x_.grad.data
        x_.grad.data.zero_()
        x_hat_adv = vae_model.re_forward(x_)
    _, _, _, x_ = vae_model(x_hat_adv)
    logit_final = c_model(x_.view(-1,120,7,7))
    logit_diff = logit_final - logit_nat
        # print((logit_diff.data.max(1)[1] != target.data).float().sum())
    return logit_diff


def logit_calculate(logit_old, logit_new):
    label_old = logit_old.data.max(1)[1]
    logit_diff = logit_new - logit_old
    label_diff = logit_diff.data.max(1)[1]
    label_new = []
    for i in range(label_old.size(0)):
        if logit_diff[i][label_old[i]].item() / abs(logit_old[i][label_old[i]].item()) >= -0.1:
        # if logit_diff[i][label_old[i]] >= 0:
            label_new.append(label_old[i].item())
        else:
            label_new.append(label_diff[i].item())
    return torch.Tensor(label_new)
    # return label_diff