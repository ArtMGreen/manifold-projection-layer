from torch.utils.data import Dataset
from torchvision.io import decode_image, ImageReadMode
from download_CIFAR10 import extract_cifar10_as_images

import os
import glob

# The params here are for ImageNetV1_1K, for CIFAR-10 see below
# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD = [0.229, 0.224, 0.225]


class CIFAR10(Dataset):
    MEAN = [0.4914, 0.4822, 0.4465]
    STD = [0.2470, 0.2435, 0.2616]
    
    def __init__(self, root, include_classes, train=True, transform=None):
        extract_cifar10_as_images()
        
        split = "train" if train else "test"
        self.samples = []
        base_dir = os.path.join(root, "cifar10-images", split)

        for class_id in include_classes:
            class_dir = os.path.join(base_dir, str(class_id))
            for img_path in glob.glob(os.path.join(class_dir, "*.png")):
                self.samples.append((img_path, class_id))

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = decode_image(path, mode=ImageReadMode.RGB)  # (C, H, W), uint8
        if self.transform:
            img = self.transform(img) # may or may not be in [0, 1], check your transforms
        return img, label
