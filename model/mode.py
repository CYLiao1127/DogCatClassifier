import torch.nn as nn
import torchvision.models as models

class DogCatModel(nn.Module):
    def __init__(self, num_calsses=2):
        super(DogCatModel, self).__init__()

        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_calsses)

    def forward(self, x):
        return self.model(x)