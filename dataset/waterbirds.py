from pathlib import Path
import torchvision.transforms as transforms
from wilds import get_dataset

from . import DatasetBase


class WaterBirds(DatasetBase):
    def create(self):
        dataset=  get_dataset(dataset="waterbirds", download=True, root_dir=self.root)
        path_mask = "masks"
        masks = get_dataset(dataset="waterbirds", download=False, root_dir=Path(path_mask))
        return dataset, masks

    def get_transform(self, _):
        return transforms.Compose(
            [
                transforms.Resize((int(224 * (256 / 224)), int(224 * (256 / 224)))),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
