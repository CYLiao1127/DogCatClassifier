import os
import argparse
from config import Config
from data.dataloader import get_dataloaders
from model.model import DogCatModel
from train import train

if __name__ == '__main__':
    # ---- Parse command-line arguments ----
    parser = argparse.ArgumentParser(description='Dog vs Cat Classification')
    # parser.add_argument('--mode', type=str, default='train', help='Mode: train / eval / predict')
    parser.add_argument('--model_name', type=str, help='Model architecture (e.g. resnet18, resnet50)')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--img_size', type=int, help='Input image size (e.g. 224)')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--use_mixup', type=bool, help='Using MixUp method or not')
    args = parser.parse_args()

    # ---- Load config and override ----
    config = Config()
    if args.model_name:
        config.model_name = args.model_name
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.img_size:
        config.img_size = args.img_size
    if args.epochs:
        config.epochs = args.epochs
    if args.use_mixup:
        config.use_mixup = args.use_mixup

    # ----- prepare data and model ----
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