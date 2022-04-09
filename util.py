import torch
import torch.nn as nn

def loss_function(x, label, x_hat, mean, log_var, logit, beta = 0.5):
    v_loss = vae_loss(x, x_hat, mean, log_var)
    c_loss = nn.CrossEntropyLoss()(logit, label)
    loss = v_loss * beta + c_loss * (1 - beta) 
    return loss

def vae_loss(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, size_average=False, reduction='mean')
    KLD      = - 0.5 * torch.mean(1+ log_var - mean.pow(2) - log_var.exp())
    v_loss = reproduction_loss + KLD
    return v_loss

def model_pred(x, vae_model, c_model):
    x_hat, mean, log_v = vae_model(x)
    x_cat = torch.cat((mean, log_v), 1)
    logit = c_model(x_cat)
    return logit, mean, log_v

def testtime_update(vae_model, x_adv, learning_rate=0.1, num = 30):
    x_hat_adv, mean, log_v = vae_model(x_adv)
    x_adv = x_adv.detach()
    for _ in range(num):
        if (x_hat_adv != x_hat_adv).sum() > 0:
            print('nan Error')
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