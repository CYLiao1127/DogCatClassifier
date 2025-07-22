# evaluate.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_score, recall_score, accuracy_score
from data.dataloader import get_dataloaders
from model.model import DogCatModel
from config import Config

# def plot_confusion_matrix(y_true, y_pred, labels):
#     cm = confusion_matrix(y_true, y_pred)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
#     disp.plot(cmap=plt.cm.Blues)
#     plt.title("Confusion Matrix")
#     plt.savefig("confusion_matrix.png")
#     plt.close()
#     print("📊 Saved confusion_matrix.png")

def plot_confusion_matrix(y_true, y_pred, labels=("cat", "dog"), save_path="confusion_matrix.png"):
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

def plot_roc(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig("roc_curve.png")
    plt.close()
    print("📈 Saved roc_curve.png")

# def evaluate_model(model, data_loader, config):
#     config = Config()
#     device = config.device

#     _, val_loader = get_dataloaders(config.data_dir, config.img_size, config.batch_size)

#     model = DogCatModel(num_classes=config.num_classes)
#     model.load_state_dict(torch.load(config.model_save_path, map_location=device))
#     model.to(device)
#     model.eval()

#     all_preds, all_probs, all_labels = [], [], []

#     with torch.no_grad():
#         for inputs, labels in val_loader:
#             inputs = inputs.to(device)
#             outputs = model(inputs)
#             probs = torch.softmax(outputs, dim=1)[:, 1]
#             preds = torch.argmax(outputs, dim=1)

#             all_labels.extend(labels.cpu().numpy())
#             all_preds.extend(preds.cpu().numpy())
#             all_probs.extend(probs.cpu().numpy())

#     acc = accuracy_score(all_labels, all_preds)
#     precision = precision_score(all_labels, all_preds)
#     recall = recall_score(all_labels, all_preds)

#     print(f"\n📋 Evaluation Metrics:")
#     print(f"Accuracy : {acc:.4f}")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall   : {recall:.4f}")

#     plot_confusion_matrix(all_labels, all_preds, labels=['cat', 'dog'])
#     plot_roc(all_labels, all_probs)

# if __name__ == '__main__':
#     evaluate_model()

# import torch

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

    plot_confusion_matrix(all_labels, all_preds, labels=['dogs', 'cats'])
    plot_roc(all_labels, all_probs)

    return 100 * correct / total


    # with torch.no_grad():
    #     for images, labels in val_loader:
    #         images, labels = images.to(config.device), labels.to(config.device)
    #         outputs = model(images)

    #         _, preds = torch.max(outputs, 1)
    #         correct += (preds == labels).sum().item()
    #         total += labels.size(0)