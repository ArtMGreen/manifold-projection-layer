from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
import torchvision.transforms.v2 as transforms
import os

def extract_cifar10_as_images(root='./data'):
    dataset = CIFAR10(root=root, train=True, download=True)
    base_dir = os.path.join(root, "cifar10-images/train")
    os.makedirs(base_dir, exist_ok=True)

    for idx, (img, label) in enumerate(dataset):
        class_dir = os.path.join(base_dir, str(label))
        os.makedirs(class_dir, exist_ok=True)
        save_path = os.path.join(class_dir, f"{idx}.png")
        save_image(transforms.ToTensor()(img), save_path)

    # Repeat for test set
    dataset = CIFAR10(root=root, train=False, download=True)
    base_dir = os.path.join(root, "cifar10-images/test")
    os.makedirs(base_dir, exist_ok=True)

    for idx, (img, label) in enumerate(dataset):
        class_dir = os.path.join(base_dir, str(label))
        os.makedirs(class_dir, exist_ok=True)
        save_path = os.path.join(class_dir, f"{idx}.png")
        save_image(transforms.ToTensor()(img), save_path)

extract_cifar10_as_images()

