import os 
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from evaluate import evaluate

def mixup_data(x, y, alpha=0.2):
    '''Mixes inputs and labels'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def train(model, train_loader, val_loader, config, use_mixup=False):
    device = config.device
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config.epochs}]")

        for inputs, targets in loop:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            if use_mixup:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=0.2)
                outputs = model(inputs)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                _, predicted = torch.max(outputs, 1)
                correct += lam * predicted.eq(targets_a).sum().item() + \
                           (1 - lam) * predicted.eq(targets_b).sum().item()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                _, predicted = torch.max(outputs, 1)
                correct += predicted.eq(targets).sum().item()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total += targets.size(0)

            loop.set_postfix(loss=loss.item(), acc=100.*correct/total)

        train_acc = correct / total
        val_acc = evaluate(model, val_loader, device)

        print(f"\n📈 Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(config.result_save_path, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print("✅ Saved best model!")

    print(f"🏁 Best Validation Accuracy: {best_acc:.4f}")
