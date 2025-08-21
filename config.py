import torch
import torch.nn.functional as F

from dataset import CIFAR10

from model import ResNet18EnergyAlong
from model import ResNet18EnergyParallel


RUN_TRAINING = True
RUN_EVALUATION = True

# COMMON SETTINGS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16

DATASET = CIFAR10
IND_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8]
OOD_CLASSES = [9]
# [0, 1] -> [PIXEL_MIN, PIXEL_MAX] in each color channel separately
PIXEL_MIN = torch.tensor([(0 - m) / s for m, s in zip(DATASET.MEAN, DATASET.STD)])
PIXEL_MAX = torch.tensor([(1 - m) / s for m, s in zip(DATASET.MEAN, DATASET.STD)])

MODEL_CLASS = ResNet18EnergyParallel
CLS_LOSS_FN = F.cross_entropy


# TRAINING PARAMETERS
EPSILON_ENERGY = 5.0
DELTA_ENERGY = 15.0
λ_ADV = 1
λ_ENERGY = 0.5
ENERGY_ENFORCEMENT_FN = F.relu

TRAIN_FGSM_ALPHA = 8 / 256
NUM_EPOCHS = 7
LR = 0.02


# EVALUATION PARAMETERS
EVAL_FGSM_ALPHA = 8 / 256