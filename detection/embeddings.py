import torch
import torch.nn as nn
from torchvision.models import resnet18

class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Generate siamese_network.pt
if __name__ == "__main__":
    model = SiameseNetwork()
    torch.save(model.state_dict(), "models/siamese_network.pt")