import os
import csv
import torch
import argparse
from torchvision import transforms
from PIL import Image
from model.model import DogCatModel
from config import Config

def load_model(config):
    model = DogCatModel(config.model_name, num_classes=2)
    model.load_state_dict(torch.load(config.best_model_path, map_location=config.device))
    model.to(config.device)
    model.eval()
    return model

def predict_image(model, image_path, transform, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # (1, C, H, W)
    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).item()
    return pred

def main(config):
    image_dir = os.path.join(config.data_dir, 'test1')
    output_csv = config.save_file

    model = load_model(config)

    transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    results = []
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))])

    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)
        pred = predict_image(model, img_path, transform, config.device)
        # label = 'dog' if pred == 0 else 'cat'
        results.append([img_name.replace(".jpg", ""), pred])

    # 儲存 CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label'])  # CSV 標頭
        writer.writerows(results)

    print(f"Prediction finished. Results saved to {output_csv}")

if __name__ == '__main__':
    # ---- Parse command-line arguments ----
    parser = argparse.ArgumentParser(description='Dog vs Cat Classification')
    # parser.add_argument('--mode', type=str, default='train', help='Mode: train / eval / predict')
    parser.add_argument('--model_name', type=str, help='Model architecture (e.g. resnet18, resnet50)')
    parser.add_argument('--best_model_path', type=str, required=True, help='')
    parser.add_argument('--save_file', required=True, type=str, help='')
    args = parser.parse_args()

    config = Config()
    if args.model_name:
        config.model_name = args.model_name
    if args.best_model_path:
        config.best_model_path = args.best_model_path
    if args.save_file:
        config.save_file = args.save_file

    main(config)