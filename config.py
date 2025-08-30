import torch
import torch.nn.functional as F

from dataset import CIFAR10

from model import ResNet18EnergyAlong
from model import ResNet18EnergyParallel

from adversaries import FGSM, PGD


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
DELTA_ENERGY = 10.0
λ_ADV = 1
λ_ENERGY = 0.25
λ_L2 = 0.005
ENERGY_ENFORCEMENT_FN = F.softplus

TRAIN_ADVERSARY = FGSM(8/256, CLS_LOSS_FN, PIXEL_MIN, PIXEL_MAX)
NUM_EPOCHS = 5
LR = 0.02


# EVALUATION PARAMETERS
EVAL_ADVERSARY = FGSM(16/256, CLS_LOSS_FN, PIXEL_MIN, PIXEL_MAX)    # PGD(16/256, 1/256, 20, CLS_LOSS_FN, PIXEL_MIN, PIXEL_MAX)