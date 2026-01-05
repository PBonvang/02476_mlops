
from __future__ import annotations

import torch
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path

from mpl_toolkits.axes_grid1 import ImageGrid


def load_corrupt_mnist(
        dataset_path: Path = Path('corruptmnist_v1/')
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

def show_image_and_target(images: torch.Tensor, target: torch.Tensor) -> None:
    """Plot images and their labels in a grid."""
    row_col = int(len(images) ** 0.5)
    fig = plt.figure(figsize=(10.0, 10.0))
    grid = ImageGrid(fig, 111, nrows_ncols=(row_col, row_col), axes_pad=0.3)
    for ax, im, label in zip(grid, images, target):
        ax.imshow(im.squeeze(), cmap="gray")
        ax.set_title(f"Label: {label.item()}")
        ax.axis("off")
    plt.show()

if __name__ == "__main__":
    train_ds, test_ds = load_corrupt_mnist()
    show_image_and_target(train_ds.tensors[0][:25], train_ds.tensors[1][:25])

