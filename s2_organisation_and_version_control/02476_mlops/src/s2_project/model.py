from torch import nn, max_pool2d
import torch
from torch.nn.functional import relu, log_softmax, dropout


class Model(nn.Module):
    """My awesome model."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)

        self.classifier = nn.Sequential(
            nn.Linear(128,32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 10)
        )
        

    def forward(self, x):
        x = relu(self.conv1(x))
        x = max_pool2d(x, 2, 2)
        x = relu(self.conv2(x))
        x = max_pool2d(x, 2, 2)
        x = relu(self.conv3(x))
        x = max_pool2d(x, 2, 2)
        
        x = x.flatten(1)
        x = self.classifier(x)

        return log_softmax(x, dim=1)

if __name__ == "__main__":
    model = Model()
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n# parameters (total):", total_params)
    print("# parameters (trainable):", trainable_params)

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")