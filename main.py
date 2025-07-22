from config import Config
from data.dataloader import get_dataloaders
from model.model import DogCatModel
from train import train

if __name__ == '__main__':
    config = Config()
    train_loader, val_loader = get_dataloaders(data_dir=config.data_dir, img_size=config.img_size, batch_size=config.batch_size)
    model = DogCatModel(num_calsses=config.num_classes)
    train(model, train_loader, val_loader, config)