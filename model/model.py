import torch.nn as nn
import torchvision.models as models

class DogCatModel(nn.Module):
    def __init__(self, model_name, num_classes=2, pretrained=True):
        super(DogCatModel, self).__init__()
        # self.base_model = models.resnet18(pretrained=True)
        # self.base_model = models.resnet50(pretrained=True)
        # self.base_model = models.efficientnet_b0(pretrained=True)
        # self.base_model = models.densenet121(pretrained=True)
        # self.base_model = models.vit_b_16(pretrained=True)

        # self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

        base_model = getattr(models, model_name)(pretrained=pretrained)
        self.model_name = model_name

        # 根據不同 backbone 替換分類 head
        if "resnet" in model_name:
            in_features = base_model.fc.in_features
            base_model.fc = nn.Linear(in_features, num_classes)

        elif "densenet" in model_name:
            in_features = base_model.classifier.in_features
            base_model.classifier = nn.Linear(in_features, num_classes)

        elif "efficientnet" in model_name:
            in_features = base_model.classifier[1].in_features
            base_model.classifier[1] = nn.Linear(in_features, num_classes)

        elif "vit" in model_name:
            in_features = base_model.heads.head.in_features
            base_model.heads.head = nn.Linear(in_features, num_classes)


    def forward(self, x):
        return self.base_model(x)