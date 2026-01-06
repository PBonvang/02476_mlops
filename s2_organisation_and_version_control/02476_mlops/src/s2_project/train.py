from datetime import datetime
from pathlib import Path

from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import typer

from s2_project.model import Model
from s2_project.data import load_processed_corrupted_mnist

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 32,
    models_dir: Path = Path('models'),
    dataset_path: Path = Path('data/processed'),
    checkpoint_interval: int = 5) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = models_dir / timestamp
    model_dir.mkdir(parents=True, exist_ok=True)

    model = Model().to(DEVICE)
    train_ds, _ = load_processed_corrupted_mnist(dataset_path)
    train_dataloader = DataLoader(train_ds, batch_size)

    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    model.train()

    statistics = []
    training_steps = 0
    for e in range(epochs):
        training_loss = 0
        for i, (images, labels) in tqdm(
                enumerate(train_dataloader),
                total=len(train_dataloader),
                desc=f"Epoch {e+1}/{epochs}"
            ):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)
            training_loss += loss.item()
            loss.backward()

            optimizer.step()
            training_steps += 1

            if training_steps % 10 == 0:
                accuracy = (torch.argmax(output, dim=1) == labels)\
                                .float().mean().item()

                statistics.append({
                    "epoch": e,
                    "step": training_steps,
                    "loss": loss.item(),
                    "accuracy": accuracy
                })

        if e % checkpoint_interval == 0 and e > 0:
            torch.save(model.state_dict(), model_dir / f'epoch_{e}.pth')

    print(f"Training complete, final epoch loss: {training_loss/len(train_dataloader):.8f}")
    torch.save(model.state_dict(), model_dir / 'final.pth')
    print("Model saved to ", model_dir)

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    steps = [stat['step'] for stat in statistics]
    losses = [stat['loss'] for stat in statistics]
    accuracies = [stat['accuracy'] for stat in statistics]

    axs[0].plot(steps, losses)
    axs[0].set_title("Train loss")
    axs[1].plot(steps, accuracies)
    axs[1].set_title("Train accuracy")
    fig.savefig(model_dir / "training_statistics.png")

if __name__ == "__main__":
    typer.run(train)
