#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import numpy as np

# Configuration
epsilon_energy = 5.0
delta_energy = 5.0
lambda_energy = 0.5  # Increased from 0.25 to prevent energy collapse
alpha_fgsm = 8 / 255

num_classes = 10
batch_size = 64  # Increased for faster training
epochs = 10
lr = 0.001  # Increased from 0.005 for better training
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

# Normalization parameters for ResNet18 preprocessing
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Calculate bounds for normalized images
NORM_MIN = torch.tensor([(0 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)])
NORM_MAX = torch.tensor([(1 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)])

print(f"Using device: {device}")

# Import custom modules
from dataset import CIFAR10
from model import ModelWrapper

# Data preprocessing
resnet18_preprocess = v2.Compose([
    v2.Resize(224, interpolation=InterpolationMode.BILINEAR),
    v2.ToDtype(torch.float32, scale=True),  # to [0, 1]
    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Create datasets and dataloaders
train_dataset = CIFAR10(root='./data', train=True, transform=resnet18_preprocess)
# pin_memory is only useful for CUDA, not for MPS
pin_memory = True if device == 'cuda' else False
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=pin_memory)

# Attack and loss functions
def fgsm_attack(model, x, y, loss_fn, alpha):
    # Store current training state
    was_training = model.training
    model.eval()
    
    x_adv = x.detach().clone().requires_grad_(True)
    logits, _ = model(x_adv)
    loss = loss_fn(logits, y)
    loss.backward()
    x_adv = x_adv + alpha * x_adv.grad.sign()
    
    # Clamp to correct bounds based on normalization
    min_vals = NORM_MIN.to(x_adv.device).view(1, 3, 1, 1).expand_as(x_adv)
    max_vals = NORM_MAX.to(x_adv.device).view(1, 3, 1, 1).expand_as(x_adv)
    x_adv = torch.clamp(x_adv, min_vals, max_vals)
    
    # Restore training state
    if was_training:
        model.train()
        
    return x_adv.detach()

def compute_loss(model, x, y, loss_fn, epsilon, delta, lambda_energy, use_lower_boundary=True, use_upper_boundary=True):
    logits_clean, energy_clean = model(x)
    loss_clean = loss_fn(logits_clean, y, reduction="mean")
    x_adv = fgsm_attack(model, x, y, loss_fn, alpha_fgsm)
    model.train()
    logits_adv, energy_adv = model(x_adv)
    loss_adv = loss_fn(logits_adv, y, reduction="mean")

    loss_energy = 0
    if lambda_energy != 0:
        # Lower boundary term (yellow ReLU)
        lower_term = energy_clean - epsilon
        if use_lower_boundary:
            lower_term = F.relu(lower_term)
        
        # Upper boundary term (green ReLU)
        upper_term = epsilon + delta - energy_adv
        if use_upper_boundary:
            upper_term = F.relu(upper_term)
        
        # Correct reduction: first mean over batch, then scale by lambda
        energy_terms = (lower_term + upper_term).mean()
        loss_energy = lambda_energy * energy_terms

    total_loss = loss_clean + loss_adv + loss_energy

    metrics = {
        'L_clean': loss_clean.item(),
        'L_adv': loss_adv.item(),
        'L_energy': loss_energy.item(),
        'E_clean': energy_clean.mean().item(),
        'E_adv': energy_adv.mean().item(),
    }

    return total_loss, metrics

def evaluate(model, loader, abstention_threshold, examples=float("inf")):
    model.eval()
    
    all_clean_E = list()
    all_adv_E = list()
    
    clean_correct = 0
    clean_total = 0
    clean_rejects = 0
    clean_no_defense_correct = 0
    
    robust_correct = 0
    robust_total = 0
    robust_rejects = 0
    robust_no_defense_correct = 0
    
    i = -1
    for x, y in tqdm(loader, total=len(loader)):
        i += 1
        if i > examples:
            break
        x, y = x.to(device), y.to(device)
        
        logits_clean, energy_clean = model(x)
        all_clean_E.append(energy_clean.cpu().detach())
        preds_clean = torch.argmax(logits_clean, dim=1)
        reject_clean = (energy_clean.squeeze() > abstention_threshold)
        valid_clean = ~reject_clean
        
        clean_correct += (preds_clean[valid_clean] == y[valid_clean]).sum().item()
        clean_no_defense_correct += (preds_clean == y).sum().item()
        clean_rejects += reject_clean.sum().item()
        clean_total += y.size(0)
        
        x_adv = fgsm_attack(model, x, y, F.cross_entropy, alpha_fgsm)
        logits_adv, energy_adv = model(x_adv)
        all_adv_E.append(energy_adv.cpu().detach())
        preds_adv = torch.argmax(logits_adv, dim=1)
        reject_adv = (energy_adv.squeeze() > abstention_threshold)
        valid_adv = ~reject_adv
        
        robust_correct += (preds_adv[valid_adv] == y[valid_adv]).sum().item()
        robust_no_defense_correct += (preds_adv == y).sum().item()
        robust_rejects += reject_adv.sum().item()
        robust_total += y.size(0)
    
    if clean_total == clean_rejects:
        clean_acc = float("NaN")
    else:
        clean_acc = clean_correct / (clean_total - clean_rejects)
    clean_no_def_acc = clean_no_defense_correct / clean_total
    clean_reject_rate = clean_rejects / clean_total
    
    if robust_total == robust_rejects:
        robust_acc = float("NaN")
    else:
        robust_acc = robust_correct / (robust_total - robust_rejects)
    robust_no_def_acc = robust_no_defense_correct / robust_total
    robust_reject_rate = robust_rejects / robust_total
    
    print(f"Clean Accuracy (rejected samples excluded): {clean_acc:.4f}")
    print(f"Clean Accuracy (rejected samples included): {clean_no_def_acc:.4f}")
    print(f"Clean Rejection Rate:       {clean_reject_rate:.4f}")
    print(f"Robust Accuracy (rejected samples excluded): {robust_acc:.4f}")
    print(f"Robust Accuracy (rejected samples included): {robust_no_def_acc:.4f}")
    print(f"Adversarial Detection Rate:  {robust_reject_rate:.4f}")
    
    all_clean_E = torch.cat(all_clean_E).squeeze().numpy()
    all_adv_E = torch.cat(all_adv_E).squeeze().numpy()
    return all_clean_E, all_adv_E

def energy_hist(energy_clean, energy_adv, abstention_threshold, title, save_path, binwidth=0.5):
    plt.figure(figsize=(10, 6))
    plt.hist(energy_clean, alpha=0.5, label='Clean Energy', bins=np.arange(min(energy_clean), max(energy_clean) + binwidth, binwidth))
    plt.hist(energy_adv, alpha=0.5, label='Adversarial Energy', bins=np.arange(min(energy_clean), max(energy_clean) + binwidth, binwidth))
    plt.axvline(abstention_threshold, color='red', linestyle='--', label='Abstention Threshold')
    plt.xlabel("Energy")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Energy histogram saved as {save_path}")

def train_model(use_lower_boundary, use_upper_boundary, model_save_name):
    print(f"\nTraining with boundaries: lower={use_lower_boundary}, upper={use_upper_boundary}")
    print(f"Model will be saved as: {model_save_name}")
    
    # Initialize model and optimizer
    model = ModelWrapper(num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    loss_fn = F.cross_entropy

    for epoch in range(epochs):
        model.train()
        total_metrics = {'L_clean': 0, 'L_adv': 0, 'L_energy': 0, 'E_clean': 0, 'E_adv': 0}
        progress_bar = tqdm(train_loader, total=len(train_loader))
        
        for x, y in progress_bar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss, metrics = compute_loss(
                model, x, y, loss_fn, epsilon_energy, delta_energy, lambda_energy,
                use_lower_boundary=use_lower_boundary, use_upper_boundary=use_upper_boundary
            )
            loss.backward()
            optimizer.step()

            for k in total_metrics:
                total_metrics[k] += metrics[k]

            progress_bar.set_postfix(metrics)

        print(f"[Epoch {epoch+1}]")
        for k in total_metrics:
            print(f"  {k}: {total_metrics[k] / len(train_loader):.4f}")

    # Save the model
    torch.save(model.state_dict(), model_save_name)
    print(f"Model saved as {model_save_name}")
    
    # Evaluate and save energy histograms
    print("\nEvaluating on test set...")
    test_dataset = CIFAR10(root='./data', train=False, transform=resnet18_preprocess)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=pin_memory)
    
    abstention_threshold = epsilon_energy + 0.5 * delta_energy
    E_clean, E_adv = evaluate(model, test_loader, abstention_threshold)
    
    # Save energy histogram
    hist_name = model_save_name.replace('.pt', '_energy_hist.png')
    title = f"Energy Distribution: {'With' if use_lower_boundary and use_upper_boundary else 'Without'} Boundaries"
    energy_hist(E_clean, E_adv, abstention_threshold, title, hist_name)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "no_boundaries":
        # Train without boundaries
        train_model(
            use_lower_boundary=False,
            use_upper_boundary=False,
            model_save_name="model_no_boundaries.pt"
        )
    else:
        # Train with boundaries (default)
        train_model(
            use_lower_boundary=True,
            use_upper_boundary=True,
            model_save_name="model_with_boundaries.pt"
        )