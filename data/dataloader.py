import os
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from .data_analysis import analyze_dataset_distribution, plot_distribution_comparison, analyze_distribution_difference
from torchvision import transforms
from PIL import Image

random.seed(100)

class DogCatDataset(Dataset):
    def __init__(self, img_dir, img_size=224, mode='train'):
        self.img_dir = img_dir
        self.img_paths = [os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if fname.endswith('.jpg')]
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip() if mode == 'train' else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        self.mode = mode

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")

        label = 0 if "dog" in os.path.basename(img_path).lower() else 1
        img = self.transform(img)
        return img, label


def get_dataloaders(data_dir, img_size, batch_size, val_ratio=0.2):
    full_dataset = DogCatDataset(os.path.join(data_dir, 'train'), img_size=img_size, mode='train')
    
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)

    val_len = int(len(indices) * val_ratio)
    train_len = len(indices) - val_len
    print(val_len, train_len)

    # 用 Subset 來指定資料順序
    from torch.utils.data import Subset

    train_indices = indices[:train_len]
    val_indices = indices[train_len:]

    # # 執行分析
    # train_counter, val_counter, full_counter = analyze_dataset_distribution(
    #     full_dataset, train_indices, val_indices
    # )

    # # 繪製比較圖
    # plot_distribution_comparison(train_counter, val_counter, full_counter)

    # # 分析分布差異
    # analyze_distribution_difference(train_counter, val_counter)

    train_set = Subset(full_dataset, train_indices)
    val_set = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# get_dataloaders("data", 224, 128, val_ratio=0.2)