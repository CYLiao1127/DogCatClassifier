import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from evaluate import evaluate

def train(model, train_loader, val_loader, config):
    device = config.device
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    for epoch in range(config.epochs):
        model.train()

        total_loss = 0.0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        accuracy = evaluate(model, val_loader, config)
        print("Validation Accuracy:", accuracy)

    torch.save(model.state_dict(), config.model_save_path)
    print("Model saved!")