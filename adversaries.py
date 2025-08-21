import torch

from config import CLS_LOSS_FN, PIXEL_MIN, PIXEL_MAX


def FGSM_attack(model, x, y, alpha):
    was_training = model.training
    model.eval()

    x_orig = x.detach().clone().requires_grad_(True)
    logits, _ = model(x_orig)
    loss = CLS_LOSS_FN(logits, y)
    # model.zero_grad() -- not needed anymore if torch.autograd.grad() is operating
    # loss.backward() -- replaced by torch.autograd.grad()
    dL_dx = torch.autograd.grad(loss, x_orig)[0]

    min_vals = PIXEL_MIN.to(x_orig.device).view(1, 3, 1, 1).expand_as(x_orig)
    max_vals = PIXEL_MAX.to(x_orig.device).view(1, 3, 1, 1).expand_as(x_orig)

    ranges = max_vals - min_vals

    x_adv = x_orig + alpha * ranges * dL_dx.sign()
    x_adv = torch.clamp(x_adv, min_vals, max_vals)

    if was_training:
        model.train()

    return x_adv.detach()