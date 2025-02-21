from pathlib import Path
import torchvision.transforms as transforms
from wilds import get_dataset
from torch.utils.data import Dataset

from . import DatasetBase
from wilds.datasets.wilds_dataset import WILDSDataset
from torch.utils.data import Dataset
import torch
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as F



def apply_transforms(input_tensor):
    # Assuming input_tensor is of shape [C, H, W] (C: channels, H: height, W: width)
    
    # Resize
    target_size = int(224 * (256 / 224))
    input_tensor = F.resize(input_tensor, (target_size, target_size))
    
    # Center Crop to 224x224
    input_tensor = F.center_crop(input_tensor, 224)
    
    # Normalize to Tensor (if it's not already)
    # input_tensor is assumed to be in [C, H, W] format, so no need to apply ToTensor here.
    # But, if you need to scale pixel values between 0 and 1:
    input_tensor = input_tensor.float() / 255.0
    
    return input_tensor

class WaterBirds(DatasetBase):
    def create(self):
        dataset = get_dataset(dataset="waterbirds", download=True, root_dir=self.root)
        path_mask = Path(self.root).parent / "masks"
        masks = get_dataset(dataset="waterbirds", download=False, root_dir=Path(path_mask))

        # Store masks inside the wrapper class
        wrapped_dataset = WaterBirdsDatasetWrapper(dataset, masks)

        return wrapped_dataset

    def get_transform(self, _):
        return transforms.Compose(
            [
                transforms.Resize((int(224 * (256 / 224)), int(224 * (256 / 224)))),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )



class WaterBirdsDatasetWrapper(WILDSDataset):
    def __init__(self, dataset, masks):
        # Ensure dataset is a WILDSDataset instance
        assert isinstance(dataset, WILDSDataset), "dataset must be an instance of WILDSDataset"
        
        self.dataset = dataset
        to_tensor = ToTensor()
        self.masks = {i:apply_transforms(to_tensor(masks[i][0])) for i in range(len(masks))}  # Store masks in a dict
        
        # Initialize WILDSDataset properties
        self._dataset_name = dataset.dataset_name + "_masked"
        self._data_dir = dataset.data_dir
        self._split_scheme = dataset.split_scheme
        self._split_dict = dataset.split_dict
        self._split_array = dataset.split_array
        self._y_array = dataset.y_array
        self._y_size = dataset.y_size
        self._metadata_fields = dataset.metadata_fields #+ ['mask']
        self._eval_grouper = dataset._eval_grouper
        self.n_classes = dataset.n_classes
        
        # Append mask metadata
        # to_tensor = ToTensor()
        # mask_tensor = torch.stack([apply_transforms(to_tensor(mask)) for mask in self.masks.values()])
        # mask_tensor_flattened = mask_tensor.view(mask_tensor.size(0), -1)

        # self._metadata_array = torch.cat((dataset.metadata_array, mask_tensor_flattened), dim=1)
        self._metadata_array = dataset.metadata_array

        self.check_init()
    
    @property
    def n_classes(self):
        return self._n_classes

    @n_classes.setter
    def n_classes(self, value):
        if value < 0:
            raise ValueError("Number of classes cannot be negative")
        self._n_classes = value

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y, metadata = self.dataset[idx]  # Get original data
        mask = self.masks[idx]  # Retrieve the corresponding mask
        return x, (y, mask), metadata  # Return (label, mask) as the label

    def get_input(self, idx):
        return self.dataset.get_input(idx)

    def eval(self, y_pred, y_true, metadata):
        return self.dataset.eval(y_pred, y_true, metadata)
