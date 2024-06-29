from torchvision.models import resnet18
import torch.nn as nn

class get_ResNet18(nn.Module):
    def __init__(self, out_dimension=128):
        super(get_ResNet18, self).__init__()
        base_model = resnet18(weights=None)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1]) 
        self.projection_head = nn.Sequential(
            nn.Linear(512, 512),  
            nn.ReLU(),
            nn.Linear(512, out_dimension)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.projection_head(x)
        return x