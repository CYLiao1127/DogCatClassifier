import os
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


class DogCatDataset(Dataset):
    def __init__(self, img_dir, img_size=224, mode='train'):
        self.img_dir = img_dir
        print(img_dir)
        self.img_paths = [os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if fname.endswith('.jpg')]
        print(len(self.img_paths))
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip() if mode == 'train' else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
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

    # 用 Subset 來指定資料順序
    from torch.utils.data import Subset

    train_indices = indices[:train_len]
    val_indices = indices[train_len:]

    print(len(full_dataset), len(train_indices), len(val_indices))

    # dog, cat = 0, 0
    # for t in train_indices:
    #     if full_dataset[t][1] == 0:
    #         dog += 1
    #     elif full_dataset[t][1] == 1:
    #         cat += 1
    
    # print(dog, cat)
        

    train_set = Subset(full_dataset, train_indices)
    val_set = Subset(full_dataset, val_indices)

    
    # total_len = len(full_dataset)
    # val_len = int(total_len * val_ratio)
    # train_len = total_len - val_len
    # print(train_len, val_len)

    # train_set, val_set = random_split(full_dataset, [train_len, val_len])
    # print(train_set, val_set)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader