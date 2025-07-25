import os
import torch
import argparse
import numpy as np
from data.dataloader import get_dataloaders
from predict import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_score, recall_score, accuracy_score
from data.dataloader import get_dataloaders
from model.model import DogCatModel
from config import Config


def plot_confusion_matrix(y_true, y_pred, config, labels=("cat", "dog"), save_path="confusion_matrix.png"):
    save_path = os.path.join(config.result_save_path, save_path)

    # 將 labels 轉換為對應的整數 index：0, 1
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    index_labels = list(range(len(labels)))

    # 建立 confusion matrix，顯式指定 labels 順序，避免 dimension mismatch
    cm = confusion_matrix(y_true, y_pred, labels=index_labels)

    # 繪製混淆矩陣
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, values_format='d')  # 'd' 表示整數格式

    # 儲存圖片
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Confusion matrix saved to {save_path}")

def plot_roc(y_true, y_probs, config, save_path="roc_curve.png"):
    save_path = os.path.join(config.result_save_path, save_path)
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Saved to {save_path}")

def evaluate(model, val_loader, config):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(config.device), labels.to(config.device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)

    print(f"\n📋 Evaluation Metrics:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")

    plot_confusion_matrix(all_labels, all_preds, config, labels=['dogs', 'cats'])
    plot_roc(all_labels, all_probs, config)

    return 100 * correct / total

if __name__ == '__main__':
    # ---- Parse command-line arguments ----
    parser = argparse.ArgumentParser(description='Dog vs Cat Classification')
    parser.add_argument('--data_dir', type=str, default='', help='')
    parser.add_argument('--model_name', type=str, help='Model architecture (e.g. resnet18, resnet50)')
    parser.add_argument('--best_model_path', type=str, required=True, help='')
    args = parser.parse_args()

    config = Config()
    if args.model_name:
        config.model_name = args.model_name
    if args.best_model_path:
        config.best_model_path = args.best_model_path

    model = load_model(config)

    _, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        img_size=224,
        batch_size=128,
        val_ratio=0.2
    )
    val_acc = evaluate(model, val_loader, config)