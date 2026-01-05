# %% Jupyter extensions
import sys

# Check if 'IPython' is in the loaded modules
if 'IPython' in sys.modules:
    from IPython import get_ipython
    ip = get_ipython()
    if ip is not None:
        ip.run_line_magic('load_ext', 'autoreload')
        ip.run_line_magic('autoreload', '2')
# %% Imports
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
import typer
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from data import load_corrupt_mnist
from model import MyAwesomeModel

# %% Constants
DS_PATH = Path('corruptmnist_v1/')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% Test area

# %% App
app = typer.Typer()


@app.command()
def train(
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 32,
    models_dir: Path = Path('models')) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = models_dir / timestamp
    model_dir.mkdir(parents=True, exist_ok=True)

    model = MyAwesomeModel().to(DEVICE)
    train_ds, _ = load_corrupt_mnist(DS_PATH)
    train_dataloader = DataLoader(train_ds, batch_size)

    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    model.train()
    pbar = tqdm(range(epochs))

    statistics = []
    training_steps = 0
    for e in pbar:
        training_loss = 0
        for i, (images, labels) in enumerate(train_dataloader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)
            training_loss += loss.item()
            loss.backward()

            optimizer.step()
        
            pbar.set_description(f"Epoch {e}/{epochs}, Step {i}/{len(train_dataloader)} - Train loss: {loss.item():.6f}")
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

        if e % 5 == 0 and e > 0:
            torch.save(model.state_dict(), model_dir / f'epoch_{e}.pt')

    print(f"Training complete, final epoch loss: {training_loss/len(train_dataloader):.8f}")
    torch.save(model.state_dict(), model_dir / 'final.pt')
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


@app.command()
def evaluate(
    model_path: Path,
    batch_size: int = 32
    ) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(model_path)

    if model_path.is_dir():
        model_path = model_path / 'final.pt'

    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    
    _, test_set = load_corrupt_mnist(DS_PATH)
    test_dataloader = DataLoader(test_set, batch_size)
    criterion = torch.nn.NLLLoss()

    accuracy = 0
    val_loss = 0
    model.eval()
    for images, labels in test_dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        output: torch.Tensor = model(images)
        loss = criterion(output, labels)
        val_loss += loss.item()
        
        accuracy += (output.argmax(dim=1) == labels)\
                        .float().mean().item()
        
    accuracy /= len(test_dataloader)
    val_loss /= len(test_dataloader)
    print(f"Test accuracy: {accuracy:.3%}%, Loss: {val_loss:.8f}")

if __name__ == "__main__":
    app()
