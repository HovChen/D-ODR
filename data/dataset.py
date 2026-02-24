import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class DRDataset(Dataset):
    def __init__(self, images_root, data_file, transform=None):
        self.images_root = images_root
        self.data_file = data_file
        self.transform = transform
        self.samples = []
        self._load_samples()

    def _load_samples(self):
        with open(self.data_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        image_path = parts[0]
                        label = int(parts[1])
                        self.samples.append((image_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        full_path = os.path.join(self.images_root, image_path)
        
        image = Image.open(full_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {"pixel_values": image, "labels": label}


def get_train_transform(img_size=224):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_val_transform(img_size=224):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
