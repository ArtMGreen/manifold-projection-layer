from torch.utils.data import Dataset
from torchvision.io import decode_image, ImageReadMode
import os
import glob


class CIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None):
        split = "train" if train else "test"
        self.samples = []
        base_dir = os.path.join(root, "cifar10-images", split)

        for class_id in range(10):
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
