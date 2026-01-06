import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import typer

from s2_project.model import Model
from s2_project.data import load_processed_corrupted_mnist

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def evaluate(
    model_path: Path,
    dataset_path: Path = Path("data/processed/"),
    batch_size: int = 32
    ) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(model_path)

    model_dir = model_path if model_path.is_dir() else model_path.parent
    model_file = model_dir / 'final.pth' if model_path.is_dir() else model_path
    
    if not model_file.exists():
        raise FileNotFoundError(f"No model file was found at the provided path: {model_file}")

    model = Model().to(DEVICE)
    model.load_state_dict(torch.load(model_file, map_location=DEVICE))
    
    _, test_set = load_processed_corrupted_mnist(dataset_path)
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

    results = {
        "accuracy": accuracy,
        "loss": val_loss,
        "device": str(DEVICE)
    }

    metrics_path = model_dir / 'evaluation.json'
    with metrics_path.open('w', encoding='utf-8') as fp:
        json.dump(results, fp, indent=2)

    print(f"Evaluation metrics written to {metrics_path}")

if __name__ == '__main__':
    typer.run(evaluate)