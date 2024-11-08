import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as T

def load_dataloaders_ImageNet(dataset_path, batch_size):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transforms = T.Compose(
        [T.Resize((224, 224)), 
        T.ToTensor(),
        T.Normalize(mean, std)])
    
    imagenet_data_train = ImageFolder(dataset_path+'/train', transform=transforms)
    data_loader_train = DataLoader(imagenet_data_train, batch_size=batch_size, shuffle=True)    
    
    imagenet_data_val = ImageFolder(dataset_path+'/val', transform=transforms)
    data_loader_val = DataLoader(imagenet_data_val, batch_size=batch_size, shuffle=False)
    
    return data_loader_train, data_loader_val


class ImageNet_two_transforms(Dataset):
    def __init__(self, dataset_path, transform_1, transform_2):
        self.imagenet_data_1 = ImageFolder(dataset_path, transform=transform_1)
        self.imagenet_data_2 = ImageFolder(dataset_path, transform=transform_2)

    def __getitem__(self, index):
        xy1 = self.imagenet_data_1[index]
        xy2 = self.imagenet_data_2[index]
        return xy1, xy2

    def __len__(self):
        return len(self.imagenet_data_1)

def load_datasets_ImageNet_two_transforms(dataset_path, batch_size, transform_1, transform_2):
    """ Load ImageNet dataloaders using two different sets of transforms. """
    imagenet_data_train = ImageNet_two_transforms(dataset_path+'/train', transform_1, transform_2)
    imagenet_data_val = ImageNet_two_transforms(dataset_path+'/val', transform_1, transform_2)
    return imagenet_data_train, imagenet_data_val


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax.
    From https://github.com/NVlabs/stylegan3/blob/main/dnnlib/util.py"""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value):
        self[name] = value

    def __delattr__(self, name: str):
        del self[name]

def ECE_calc(samples_certainties, num_bins=15, adaptive=False):
    '''https://github.com/idogalil/benchmarking-uncertainty-estimation-performance'''
    indices_sorting_by_confidence = torch.argsort(samples_certainties[:, 0], descending=False)  # Notice the reverse sorting
    samples_certainties = samples_certainties[indices_sorting_by_confidence]
    samples_certainties = samples_certainties.transpose(0, 1)
    if adaptive:
        certainties = samples_certainties[0]
        N = certainties.shape[0]
        step_size = int(N / (num_bins - 1))
        bin_boundaries = [certainties[i].item() for i in range(0, certainties.shape[0], step_size)]
        bin_boundaries[0] = 0
        bin_boundaries[-1] = certainties[-1]
        bin_boundaries = torch.tensor(bin_boundaries)
    else:
        bin_boundaries = torch.linspace(0, 1, num_bins + 1)        
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bins_accumulated_error = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        bin_indices = torch.logical_and(samples_certainties[0] <= bin_upper, samples_certainties[0] > bin_lower)
        if bin_indices.sum() == 0:
            continue  # This is an empty bin
        bin_confidences = samples_certainties[0][bin_indices]
        bin_accuracies = samples_certainties[1][bin_indices]
        bin_avg_confidence = bin_confidences.mean()
        bin_avg_accuracy = bin_accuracies.mean()
        bin_error = torch.abs(bin_avg_confidence - bin_avg_accuracy)
        bins_accumulated_error += bin_error * bin_confidences.shape[0]

    expected_calibration_error = bins_accumulated_error / samples_certainties.shape[1]
    return expected_calibration_error