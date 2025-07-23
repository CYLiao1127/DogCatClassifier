import os
import torch
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from model.model import DogCatModel
from config import Config

# 自定義測試集 Dataset
class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = read_image(img_path).float() / 255.0  # Normalize to [0, 1]
        if self.transform:
            image = self.transform(image)
        return image, img_name

# 預測函式
def predict_and_save_csv(model, dataloader, config):
    model.eval()
    predictions = []

    # label_map = {0: 'cat', 1: 'dog'}

    with torch.no_grad():
        for inputs, filenames in tqdm(dataloader):
            inputs = inputs.to(config.device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy()

            for filename, pred in zip(filenames, preds):
                predictions.append({'filename': filename, 'label': pred})
                # predictions.append({'filename': filename, 'label': label_map[pred]})

    df = pd.DataFrame(predictions)
    df.to_csv('predictions.csv', index=False)
    print("✅ Predictions saved to predictions.csv")


if __name__ == '__main__':
    config = Config()
    transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
    ])

    test_dataset = TestDataset(os.path.join(config.data_dir, 'test1'), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = DogCatModel(num_classes=config.num_classes)
    model.load_state_dict(torch.load(config.model_save_path, map_location=config.device))
    model.to(config.device)

    predict_and_save_csv(model, test_loader, config)