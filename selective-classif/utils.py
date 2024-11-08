import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, ImageFolder
from torchvision import transforms as T


def load_dataloaders_CIFAR(dataset_path, batch_size):
    ''' from https://github.com/huyvnphan/PyTorch_CIFAR10/blob/master/data.py '''

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2471, 0.2435, 0.2616)
    
    transform_train = T.Compose(
        [T.RandomCrop(32, padding=4),
         T.RandomHorizontalFlip(),
         T.ToTensor(),
         T.Normalize(mean, std)])
    dataset_val_dataAug = CIFAR10(root=dataset_path, train=False, transform=transform_train) # val set with train transforms
    dataset_train, _ = torch.utils.data.random_split(dataset_val_dataAug, [5000, 5000], generator=torch.Generator().manual_seed(42)) # 1st half of validation set with data augmentation
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    transform_val = T.Compose(
        [T.ToTensor(),
         T.Normalize(mean, std)])
    dataset_val = CIFAR10(root=dataset_path, train=False, transform=transform_val)
    _, dataset_val2 = torch.utils.data.random_split(dataset_val, [5000, 5000], generator=torch.Generator().manual_seed(42)) # 2nd half of validation set without data augmentation
    dataloader_val = DataLoader(dataset_val2, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    # mean = (0.4914, 0.4822, 0.4465)
    # std = (0.2471, 0.2435, 0.2616)
    # seed = 123

    # transform_train = T.Compose(
    #     [T.RandomCrop(32, padding=4),
    #     T.RandomHorizontalFlip(),
    #     T.ToTensor(),
    #     T.Normalize(mean, std)])
    # dataset_train1 = CIFAR10(root=dataset_path, train=True, transform=transform_train)
    # dataset_val_dataAug = CIFAR10(root=dataset_path, train=False, transform=transform_train)
    # dataset_train2, _ = torch.utils.data.random_split(dataset_val_dataAug, [5000, 5000], generator=torch.Generator().manual_seed(seed))
    # dataset_train = torch.utils.data.ConcatDataset((dataset_train1, dataset_train2)) # training set + 1st half of validation set with data augmentation
    # dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True)

    # transform_val = T.Compose(
    #     [T.ToTensor(),
    #     T.Normalize(mean, std)])
    # dataset_val = CIFAR10(root=dataset_path, train=False, transform=transform_val)
    # _, dataset_val = torch.utils.data.random_split(dataset_val, [5000, 5000], generator=torch.Generator().manual_seed(seed)) # 2nd half of validation set without data augmentation
    # dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, pin_memory=True)
    

    return dataloader_train, dataloader_val


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