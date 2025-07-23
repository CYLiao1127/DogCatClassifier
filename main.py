from config import Config
from data.dataloader import get_dataloaders
from model.model import DogCatModel
from train import train
import os

if __name__ == '__main__':
    config = Config()
    train_loader, val_loader = get_dataloaders(
        data_dir=config.data_dir,
        img_size=config.img_size,
        batch_size=config.batch_size
    )
    config.result_save_path = os.path.join(config.result_save_path, config.model_name)
    
    if not os.path.exists(config.result_save_path):
        os.makedirs(config.result_save_path)

    model = DogCatModel(model_name=config.model_name, num_classes=config.num_classes)
    train(model, train_loader, val_loader, config)