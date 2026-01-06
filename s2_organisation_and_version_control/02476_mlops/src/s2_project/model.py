from torch import nn, max_pool2d
import torch
from torch.nn.functional import relu, log_softmax


class Model(nn.Module):
    """
    CNN for image classification. 
    Architecture: 3 Conv-Pool blocks followed by a 2-layer MLP classifier.
    """

    def __init__(self) -> None:
        super().__init__()
        # In: 1x28x28 | Out: 32x26x26
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # In: 32x13x13 | Out: 64x11x11
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # In: 64x5x5 | Out: 128x3x3
        self.conv3 = nn.Conv2d(64, 128, 3, 1)

        self.classifier = nn.Sequential(
            nn.Linear(128,32),
            nn.ReLU(),
            nn.Dropout(0.2), # Prevents overfitting by zeroing 20% of activations
            nn.Linear(32, 10) # 10 output classes (digits 0-9)
        )
        

    def forward(self, x):
        x = relu(self.conv1(x))
        x = max_pool2d(x, 2, 2) # Downsample 26x26 -> 13x13
        x = relu(self.conv2(x))
        x = max_pool2d(x, 2, 2) # Downsample 11x11 -> 5x5
        x = relu(self.conv3(x))
        x = max_pool2d(x, 2, 2) # Downsample 3x3 -> 1x1
        
        # Prepare for Linear layers: flatten (Batch, 128, 1, 1) -> (Batch, 128)
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