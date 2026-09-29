import datasets
import torch
import torch.utils.data as data
import torchvision

class ImageDataset(data.Dataset):
    def __init__(
        self,
        hf_dataset: datasets.Dataset,
        transform: torchvision.transforms.Compose=None
    ) -> None:
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        item = self.dataset[idx]
        image = item["image"]
        if self.transform is not None:
            image = self.transform(image)
        return image
