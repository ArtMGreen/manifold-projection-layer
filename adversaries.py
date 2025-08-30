import torch


class FGSM:
    def __init__(self, alpha, cls_loss_fn, pixel_min, pixel_max):
        self.alpha = alpha
        self.cls_loss_fn = cls_loss_fn
        self.pixel_min = pixel_min
        self.pixel_max = pixel_max

    def attack(self, model, x, y):
        was_training = model.training
        model.eval()
    
        x_orig = x.detach().clone().requires_grad_(True)
        logits, _ = model(x_orig)
        loss = self.cls_loss_fn(logits, y)
        # model.zero_grad() -- not needed anymore if torch.autograd.grad() is operating
        # loss.backward() -- replaced by torch.autograd.grad()
        dL_dx = torch.autograd.grad(loss, x_orig)[0]
    
        min_vals = self.pixel_min.to(x_orig.device).view(1, 3, 1, 1).expand_as(x_orig)
        max_vals = self.pixel_max.to(x_orig.device).view(1, 3, 1, 1).expand_as(x_orig)
    
        ranges = max_vals - min_vals
    
        x_adv = x_orig + self.alpha * ranges * dL_dx.sign()
        x_adv = torch.clamp(x_adv, min_vals, max_vals)
    
        if was_training:
            model.train()
    
        return x_adv.detach()


class PGD:
    def __init__(self, epsilon, alpha, steps, cls_loss_fn, pixel_min, pixel_max):
        self.epsilon = epsilon  # ultimate boundary
        self.alpha = alpha      # single step boundary
        self.steps = steps
        self.cls_loss_fn = cls_loss_fn
        self.pixel_min = pixel_min
        self.pixel_max = pixel_max

    def attack(self, model, x, y):
        was_training = model.training
        model.eval()

        min_vals = self.pixel_min.to(x.device).view(1, 3, 1, 1).expand_as(x)
        max_vals = self.pixel_max.to(x.device).view(1, 3, 1, 1).expand_as(x)
    
        ranges = max_vals - min_vals

        min_vals_eps = torch.clamp(x - self.epsilon * ranges, min_vals, max_vals)
        max_vals_eps = torch.clamp(x + self.epsilon * ranges, min_vals, max_vals)
    
        x_adv = x.detach().clone()
        uniform01 = torch.rand(x_adv.size(), device=x_adv.device)
        x_adv += uniform01 * 2 * ranges * self.epsilon - ranges * self.epsilon
        x_adv.requires_grad_(True)

        for i in range(self.steps):
            logits, _ = model(x_adv)
            loss = self.cls_loss_fn(logits, y)
            dL_dx = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv + self.alpha * ranges * dL_dx.sign()
            x_adv = torch.clamp(x_adv, min_vals_eps, max_vals_eps)

        if was_training:
            model.train()
    
        return x_adv.detach()
            

