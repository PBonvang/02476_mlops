from pathlib import Path
from glob import glob

import torch
import typer
from torch.utils.data import TensorDataset

def load_raw_corrupt_mnist(
    dataset_path: Path
) -> tuple[TensorDataset, TensorDataset]:
    """Return train and test datasets for corrupt MNIST."""
    # exchange with the corrupted mnist dataset
    train_image_files = sorted(
        glob(str(dataset_path / 'train_images*'))
    )
    train_target_files = sorted(
        glob(str(dataset_path / 'train_target*'))
    )

    train_images = torch.cat([torch.load(f) for f in train_image_files])\
                        .unsqueeze(1).float()
    train_targets = torch.cat([torch.load(f) for f in train_target_files])\
        .long()

    test_images = torch.load(dataset_path / 'test_images.pt')\
        .unsqueeze(1).float()
    test_targets = torch.load(dataset_path / 'test_target.pt').long()

    train_ds = TensorDataset(train_images, train_targets)
    test_ds = TensorDataset(test_images, test_targets)

    return train_ds, test_ds

def preprocess(
    raw_dir: Path,
    processed_dir: Path
) -> None:
    """
    Preprocess the raw corrupt MNIST data by normalizing it and saving
    the processed tensors to disk.

    Args:
        raw_dir (Path): Directory containing the raw data files.
        processed_dir (Path): Directory to save the processed data files.
    """
    print("Loading data")
    train_ds, test_ds = load_raw_corrupt_mnist(raw_dir)
    print("Data loaded")

    print("Preprocessing data...")
    train_images, train_targets = train_ds.tensors
    test_images, test_targets = test_ds.tensors

    mean = train_images.mean()
    std = train_images.std().clamp_min(1e-8)

    train_normalized_images = (train_images - train_images.mean()) / train_images.std()
    test_normalized_images = (test_images - mean) / std

    processed_dir.mkdir(parents=True, exist_ok=True)
    torch.save(train_normalized_images, processed_dir / 'train_images.pt')
    torch.save(train_targets, processed_dir / 'train_targets.pt')
    torch.save(test_normalized_images, processed_dir / 'test_images.pt')
    torch.save(test_targets, processed_dir / 'test_targets.pt')

    print(
        f"Normalization complete (mean={mean.item():.4f}, std={std.item():.4f}); "
        f"files written to {processed_dir}"
    )

def load_processed_corrupted_mnist(
    dataset_path: Path
) -> tuple[TensorDataset, TensorDataset]:
    """Load the normalized corrupt MNIST tensors from disk."""
    train_images = torch.load(dataset_path / 'train_images.pt')
    train_targets = torch.load(dataset_path / 'train_targets.pt')
    test_images = torch.load(dataset_path / 'test_images.pt')
    test_targets = torch.load(dataset_path / 'test_targets.pt')

    train_ds = TensorDataset(train_images, train_targets)
    test_ds = TensorDataset(test_images, test_targets)

    return train_ds, test_ds

if __name__ == "__main__":
    typer.run(preprocess)
