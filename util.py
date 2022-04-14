import torch
import torch.nn as nn
import torch.nn.functional as F
def loss_function_mean(x, label, x_hat, mean, log_var, logit, beta = 0.5):
    v_loss = vae_loss_mean(x, x_hat, mean, log_var)
    c_loss = F.cross_entropy(logit, label)
    loss = v_loss * beta + c_loss * (1 - beta)
    return v_loss, c_loss

def loss_function_sum(x, label, x_hat, mean, log_var, logit, beta = 0.5):
    v_loss = vae_loss_sum(x, x_hat, mean, log_var)
    c_loss = F.cross_entropy(logit, label)
    loss = v_loss * beta + c_loss * (1 - beta)
    return v_loss, c_loss

def vae_loss_mean(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, size_average=False, reduction='mean')
    KLD      = - 0.5 * torch.mean(1+ log_var - mean.pow(2) - log_var.exp())
    v_loss = reproduction_loss + KLD
    return v_loss

def vae_loss_sum(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    v_loss = reproduction_loss + KLD
    return v_loss

def model_pred(x, vae_model, c_model):
    x_hat, mean, log_v = vae_model(x)
    x_cat = torch.cat((mean, log_v), 1)
    logit = c_model(x_cat)
    return logit

def testtime_update_mnist(vae_model, c_model, x_adv, learning_rate=0.1, num = 30, mode = 'mean'):
    x_adv = x_adv.detach()
    x_hat_adv, mean, log_v = vae_model(x_adv)
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
        mean.retain_grad()
        log_v.retain_grad()
        loss.backward(retain_graph=True)
        with torch.no_grad():
            mean.data -= learning_rate * mean.grad.data
            log_v.data -= learning_rate * log_v.grad.data
        mean.grad.data.zero_()
        log_v.grad.data.zero_()
        x_hat_adv = vae_model.decoder(vae_model.reparameterize(mean, log_v))
    x_cat = torch.cat((mean, log_v), 1)
    logit_adv = c_model(x_cat)
    return logit_adv

def testtime_update_cifar(vae_model, c_model, x_adv, target, learning_rate=0.1, num = 30, mode = 'mean'):
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
        x_.retain_grad()
        loss.backward(retain_graph=True)
        with torch.no_grad():
            x_.data -= learning_rate * x_.grad.data
        x_.grad.data.zero_()
    x_hat_adv = vae_model.re_forward(x_)
    _, _, _, x_ = vae_model(x_hat_adv)
    logit_adv = c_model(x_.view(-1,160,8,8))
        # print((logit_adv.data.max(1)[1] != target.data).float().sum())
    return logit_adv

def diff_update_cifar(vae_model, c_model, x_adv, target, learning_rate=0.1, num = 30, mode = 'mean'):
    x_adv = x_adv.detach()
    x_hat_adv, mean, log_v, x_ = vae_model(x_adv)
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
    x_hat_adv = vae_model.re_forward(x_)
    _, _, _, x_ = vae_model(x_hat_adv)
    logit_final = c_model(x_.view(-1,160,8,8))
    logit_diff = logit_final - logit_nat
        # print((logit_diff.data.max(1)[1] != target.data).float().sum())
    return logit_diff

def diff_update_mnist(vae_model, c_model, x_adv, learning_rate=0.1, num = 30, mode = 'mean'):
    x_adv = x_adv.detach()
    x_hat_adv, mean, log_v = vae_model(x_adv)
    x_cat = torch.cat((mean, log_v), 1)
    logit_nat = c_model(x_cat)
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
        mean.retain_grad()
        log_v.retain_grad()
        loss.backward(retain_graph=True)
        with torch.no_grad():
            mean.data -= learning_rate * mean.grad.data
            log_v.data -= learning_rate * log_v.grad.data
        mean.grad.data.zero_()
        log_v.grad.data.zero_()
        x_hat_adv = vae_model.decoder(vae_model.reparameterize(mean, log_v))
    x_cat = torch.cat((mean, log_v), 1)
    logit_final = c_model(x_cat)
    logit_diff = logit_final - logit_nat
    return logit_diff
