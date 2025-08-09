from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
import torchvision.transforms.v2 as transforms
import torch
import os

def extract_cifar10_as_images(root='./data'):
    # Check if images already exist
    train_dir = os.path.join(root, "cifar10-images/train")
    test_dir = os.path.join(root, "cifar10-images/test")
    
    # Check if both directories exist and contain the expected number of images
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        train_count = sum(len(files) for _, _, files in os.walk(train_dir))
        test_count = sum(len(files) for _, _, files in os.walk(test_dir))
        
        if train_count == 50000 and test_count == 10000:
            print(f"CIFAR-10 images already extracted: {train_count} train, {test_count} test images found.")
            return
    
    print("Extracting CIFAR-10 dataset to PNG images...")
    
    # Extract training set
    dataset = CIFAR10(root=root, train=True, download=True)
    base_dir = os.path.join(root, "cifar10-images/train")
    os.makedirs(base_dir, exist_ok=True)

    for idx, (img, label) in enumerate(dataset):
        class_dir = os.path.join(base_dir, str(label))
        os.makedirs(class_dir, exist_ok=True)
        save_path = os.path.join(class_dir, f"{idx}.png")
        if not os.path.exists(save_path):  # Skip if file already exists
            save_image(transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])(img), save_path)

    # Extract test set
    dataset = CIFAR10(root=root, train=False, download=True)
    base_dir = os.path.join(root, "cifar10-images/test")
    os.makedirs(base_dir, exist_ok=True)

    for idx, (img, label) in enumerate(dataset):
        class_dir = os.path.join(base_dir, str(label))
        os.makedirs(class_dir, exist_ok=True)
        save_path = os.path.join(class_dir, f"{idx}.png")
        if not os.path.exists(save_path):  # Skip if file already exists
            save_image(transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])(img), save_path)
    
    print("CIFAR-10 extraction complete!")

if __name__ == "__main__":
    extract_cifar10_as_images()

