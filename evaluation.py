import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from adversaries import fgsm_attack

from config import DEVICE
from config import EVAL_FGSM_ALPHA

def results_and_energy(model, loader):
    model.eval()

    all_clean_E = list()
    all_adv_E = list()

    # here are only arrays of 0s and 1s for incorrect and correct predictions respectively
    all_clean_res = list()
    all_adv_res = list()

    # with torch.no_grad(): -- prohibited, we need gradients for FGSM attack
    for x, y in tqdm(loader, total=len(loader), desc="Evaluating"):
        x, y = x.to(DEVICE), y.to(DEVICE)

        logits_clean, energy_clean = model(x)
        all_clean_E.append(energy_clean.detach().cpu())
        preds_clean = torch.argmax(logits_clean, dim=1)
        clean_results = (preds_clean == y).to(int)
        all_clean_res.append(clean_results.detach().cpu())

        x_adv = fgsm_attack(model, x, y, EVAL_FGSM_ALPHA)
        
        logits_adv, energy_adv = model(x_adv)
        all_adv_E.append(energy_adv.detach().cpu())
        preds_adv = torch.argmax(logits_adv, dim=1)
        adv_results = (preds_adv == y).to(int)
        all_adv_res.append(adv_results.detach().cpu())
        
    all_clean_E = torch.cat(all_clean_E).squeeze().numpy()
    all_adv_E = torch.cat(all_adv_E).squeeze().numpy()
    all_clean_res = torch.cat(all_clean_res).squeeze().numpy()
    all_adv_res = torch.cat(all_adv_res).squeeze().numpy()
    return all_clean_E, all_adv_E, all_clean_res, all_adv_res


def evaluate_at_threshold(energy, results, reject_lower_than_threshold=False):
    total = len(results)

    sorted_by_energy = np.argsort(energy)
    if reject_lower_than_threshold:
        sorted_by_energy = np.flip(sorted_by_energy)
        
    energy = energy[sorted_by_energy]
    results = results[sorted_by_energy]
    correct_at_threshold = np.cumsum(results)
    accuracy_at_threshold = correct_at_threshold / np.arange(1, total+1)

    if reject_lower_than_threshold:
        return np.flip(energy), np.flip(accuracy_at_threshold)
    return energy, accuracy_at_threshold
    

def evaluation_figure(energies, curves, title=None, save_path=None):
    fig, ax_hist = plt.subplots()
    ax_hist.set_xlabel("Energy")
    
    if energies:
        binwidth = 0
        ax_hist.set_ylabel("Count")
        
        overall_min_E = float("inf")
        overall_max_E = float("-inf")
        
        for energy_label, energy in energies.items():
            current_min_E = min(energy)
            current_max_E = max(energy)
            binwidth += (current_max_E - current_min_E) / 100
            overall_min_E = min(overall_min_E, current_min_E)
            overall_max_E = max(overall_max_E, current_max_E)
        
        binwidth /= len(energies)
        hist_alpha = 1 / len(energies)
        bin_limits = np.arange(overall_min_E, overall_max_E + binwidth, binwidth)
        
        for energy_label, energy in energies.items():
            ax_hist.hist(energy, alpha=hist_alpha, label=energy_label, bins=bin_limits)

    if curves:
        if energies:
            ax_curves = ax_hist.twinx()
        else:
            ax_curves = ax_hist
        ax_curves.set_ylabel("Accuracy @ Abstention Threshold")
        
        for label, (x, y) in curves.items():
            ax_curves.plot(x, y, label=label)
    
    if title:
        plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
