# Dog vs Cat Image Classification
Dogs 🐶 vs Cats 🐱 classification using deep learning models, trained on RTX 2080 Ti and Google Colab (T4).  
Dataset Used : [Dogs vs. Cats](https://www.kaggle.com/competitions/dogs-vs-cats/data)

## Project Structure
```
DogCatClassifier
 ┣ data/             # Data-related files
 ┃ ┣ data/           # Dataset folder (not included)
 ┃ ┗ dataloader.py   # Custom data loading scripts
 ┣ models/
 ┃ ┗ model.py        # Model architecture definition
 ┣ result/           # Training results, confusion matrix images, etc.
 ┣ main.py           # Training and validation entry point
 ┣ config.py         # Configuration file (hyperparameters, paths)
 ┣ train.py          # training code
 ┣ evaluate.py       # Model evaluation script
 ┣ predict.py        # predict and save as csv file
 ┣ data_analysis.py  # Dataset statistics and analysis
 ┣ requirements.txt  # Required Python packages
 ┗ README.md         # Project description
```

## Datasets
- The dataset is from the official Kaggle Dogs vs. Cats competition, containing about 25,000 images evenly split between dogs and cats.
- Images have variable sizes; resizing and data augmentation are applied during training.
- Please download the dataset from Kaggle and place it under `data/data/`.
- The original training set is randomly split into training and validation subsets with an 80:20 ratio.
- The validation subset is used to evaluate model performance during training and after.
- Data preprocessing includes resizing, normalization, and data augmentation on the training subset.

## Environment Setup & Requirements
It is recommended to use Python 3.10+ and set up a virtual environment before installing dependencies:
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate
# Install PyTorch (adjust for your CUDA version if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other required packages
pip install -r requirements.txt
```

## Training and Evaluation
- Training the model

```bash
python main.py
```

- Evaluating the model

```bash
python evaluate.py
```

- Predicting images and save as csv file
```bash
python predict.py
```

## Model Details
- Backbone: Pretrained ResNet18 fine-tuned for binary classification
- Loss function: CrossEntropyLoss
- Optimizer: Adam
- Data augmentation: random horizontal flip, random crop, color jitter
- Training conducted on RTX 2080 Ti and Google Colab T4 GPUs

## Results
- The reported accuracy and metrics are based on the validation set (20% of the original training data), **not on a separate test set**.
- Training accuracy: up to XX.X%
- Validation accuracy: around XX.X%