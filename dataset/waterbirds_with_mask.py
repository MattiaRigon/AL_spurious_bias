import torchvision.transforms as transforms
from wilds import get_dataset
from utils.explanations import get_explanations

from . import DatasetBase


class WaterBirdsWithMask(DatasetBase):
    def create(self):
        dataset = get_dataset(dataset="waterbirds", download=True, root_dir=self.root)
        explanation = get_explanations(dataset, "waterbirds")
        

    def get_transform(self, _):
        return transforms.Compose(
            [
                transforms.Resize((int(224 * (256 / 224)), int(224 * (256 / 224)))),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
