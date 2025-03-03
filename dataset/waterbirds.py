import torchvision.transforms as transforms
from wilds import get_dataset

from . import DatasetBase


class WaterBirds(DatasetBase):
    def create(self):
        return get_dataset(dataset="waterbirds", download=True, root_dir=self.root)

    def get_transform(self, _):
        return transforms.Compose(
            [
                transforms.Resize((int(224 * (256 / 224)), int(224 * (256 / 224)))),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
