import sys
sys.path.append('./stylegan/stylegan2')

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10
from torchvision import transforms as T
from torch.nn import functional as F

from cifar10_models.densenet import densenet121, densenet161, densenet169
from cifar10_models.googlenet import googlenet
from cifar10_models.inception import inception_v3
from cifar10_models.mobilenetv2 import mobilenet_v2
from cifar10_models.resnet import resnet18, resnet34, resnet50
from cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from temperature_scaling import ModelWithTemperature, _ECELoss
from general_calibration_error import gce
from dirichletcal.calib.vectorscaling import VectorScaling
from uncertainty_metrics import AURC_calc, AUROC, ECE_calc, coverage_for_desired_accuracy

for p in [
    Path('/d/alecoz/projects'), # DeepLab
    Path(os.path.expandvars('$WORK')), # Jean Zay
    Path('w:/')]: # local
    if os.path.exists(p):
        path_main = p / 'calibration-with-synthetic-data'
path_results = path_main / 'results'
for p in [
    Path('/scratchf/CIFAR'), # DeepLab
    Path(os.path.expandvars('$DSDIR'))]: # Jean Zay
    if os.path.exists(p):
        path_dataset = p
path_models =  path_main / 'models/CIFAR10'




def create_CIFAR10_data(batch_size=128, seed=123):
    # idx_to_label = {
    #     0: 'airplane',
    #     1: 'car',
    #     2: 'bird',
    #     3: 'cat',
    #     4: 'deer', 
    #     5: 'dog', 
    #     6: 'frog', 
    #     7: 'horse', 
    #     8: 'ship',
    #     9: 'truck'}
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2471, 0.2435, 0.2616)
    transforms = T.Compose(
        [T.ToTensor(),
        T.Normalize(mean, std)])
    # dataset_train = CIFAR10(root=path_dataset, train=True, transform=transforms)
    dataset_val = CIFAR10(root=path_dataset, train=False, transform=transforms)
    dataset_calib, dataset_test = torch.utils.data.random_split(dataset_val, [5000, 5000], generator=torch.Generator().manual_seed(seed))
    # dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    # dataloader_calib = DataLoader(dataset_calib, batch_size=batch_size, shuffle=True, pin_memory=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, pin_memory=True)

    return dataset_calib, dataloader_test # dataloader_calib constructed later


def metrics_from_dataloader(model, dataloader, vector_scale=None):
    # First: collect all the logits and labels for the validation set
    logits_list = []
    labels_list = []
    for input, label in dataloader:
        input = input.cuda()
        with torch.no_grad():
            logits = model(input)
        logits_list.append(logits)
        labels_list.append(label)
    logits = torch.cat(logits_list).cpu()
    if vector_scale is not None:
        probs = torch.from_numpy(vector_scale.predict_proba(logits).copy())
    else:
        probs = torch.softmax(logits, dim=1)
    labels = torch.cat(labels_list).cpu()
    
    certainties, y_pred = probs.max(axis=1)
    correct = y_pred == labels
    samples_certainties = torch.stack((certainties.cpu(), correct.cpu()), dim=1)
    
    # Second: compute calibration metrics
    metrics = {}
    metrics['ECE'] = gce(labels, probs, binning_scheme='even', class_conditional=False, max_prob=True, norm='l1', num_bins=15)
    metrics['SCE'] = gce(labels, probs, binning_scheme='even', class_conditional=False, max_prob=False, norm='l1', num_bins=15)
    metrics['RMSCE'] = gce(labels, probs, binning_scheme='adaptive', class_conditional=False, max_prob=True, norm='l2', datapoints_per_bin=100)
    metrics['ACE'] = gce(labels, probs, binning_scheme='adaptive', class_conditional=True, max_prob=False, norm='l1')
    metrics['TACE'] = gce(labels, probs, binning_scheme='adaptive', class_conditional=True, max_prob=False, norm='l1', threshold=0.01)
    
    # Third: compute uncertainty metrics
    metrics['Accuracy'] = (samples_certainties[:,1].sum() / samples_certainties.shape[0]).item() * 100
    metrics['AUROC'] = AUROC(samples_certainties)
    metrics['Coverage_for_Accuracy_99'] = coverage_for_desired_accuracy(samples_certainties, accuracy=0.99, start_index=200)
    metrics['Coverage_for_Accuracy_95'] = coverage_for_desired_accuracy(samples_certainties, accuracy=0.95, start_index=200)
    metrics['Coverage_for_Accuracy_90'] = coverage_for_desired_accuracy(samples_certainties, accuracy=0.90, start_index=200)
    # metrics[f'ECE_15'], mce = ECE_calc(samples_certainties, num_bins=15)
    metrics['AURC'] = AURC_calc(samples_certainties)
    
    return metrics

# Functions for synthetic data
def postprocess_synthetic_images(images):
    assert images.dim() == 4, "Expected 4D (B x C x H x W) image tensor, got {}D".format(images.dim())
    images = ((images + 1) / 2).clamp(0, 1) # scale
    return images

def preprocess_images_classifier(images):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2471, 0.2435, 0.2616)
    images = T.Normalize(mean, std)(images)
    return images
    
class SyntheticImageDataset(Dataset):
    def __init__(self, generator, max_len):
        self.G = generator
        self.max_len = max_len

    def __len__(self):
        return self.max_len

    def __getitem__(self, idx):
        z = torch.randn([1, self.G.z_dim]).cuda() # latent codes
        label = torch.randint(self.G.c_dim, (1,)).cuda()
        c = torch.nn.functional.one_hot(label, num_classes=self.G.c_dim) # class labels
        img = self.G(z, c, truncation_psi=1) # NCHW, float32, dynamic range [-1, +1]
        img = postprocess_synthetic_images(img)
        img = preprocess_images_classifier(img).squeeze()
        return img, label.squeeze()

class FilteredSyntheticImageDataset(Dataset):
    def __init__(self, generator, max_len, classifier):
        self.G = generator
        self.max_len = max_len
        self.classifier = classifier

    def __len__(self):
        return self.max_len

    def __getitem__(self, idx):
        agreement = False # is class condition same as class predicted? otherwise classifier and generator don't agree 
        while not agreement:

            z = torch.randn([1, self.G.z_dim]).cuda() # latent codes
            label = torch.randint(self.G.c_dim, (1,)).cuda()
            c = torch.nn.functional.one_hot(label, num_classes=self.G.c_dim) # class labels
            img = self.G(z, c, truncation_psi=1) # NCHW, float32, dynamic range [-1, +1]
            img = postprocess_synthetic_images(img)
            img = preprocess_images_classifier(img)

            with torch.no_grad():            
                logits = self.classifier(img)
            probas = torch.softmax(logits, dim=1)
            label_pred = torch.argmax(probas, 1)
            agreement = (label == label_pred).item()

        return img.squeeze(), label.squeeze()
    

if __name__ == '__main__':
    
    seed = 4
        
    max_synthetic_images = 10000

    models = {
    'densenet121': densenet121, 'densenet161': densenet161, 'densenet169': densenet169,
    'googlenet': googlenet,
    'inception_v3': inception_v3,
    'mobilenet_v2': mobilenet_v2,
    'resnet18': resnet18, 'resnet34': resnet34, 'resnet50':resnet50, 
    'vgg11_bn': vgg11_bn, 'vgg13_bn': vgg13_bn, 'vgg16_bn': vgg16_bn, 'vgg19_bn':vgg19_bn
    }

    methods = ['baseline (no calibration)', 'temperature scaling', 'vector scaling']
    
    # LOAD GENERATOR
    with open(path_models/'cifar10.pkl', 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()
    
    # LOAD ORIGINAL DATA
    dataset_calib_valid, dataloader_test = create_CIFAR10_data(seed=seed)
    
    # ITERATE OVER MODELS
    for model_name, model in models.items():
        classifier = model(pretrained=True).eval().requires_grad_(False).cuda()
                        
        # ITERATE OVER DATASET ORIGIN
        for dataset_origin in ['validation', 'synthetic', 'synthetic filtered', 'validation augmented', 'validation low confidence', 'validation high confidence']:
            if dataset_origin in ['validation', 'validation low confidence', 'validation high confidence']:
                dataset_calib = dataset_calib_valid
            elif dataset_origin == 'synthetic':
                dataset_calib = SyntheticImageDataset(G, max_synthetic_images)
            elif dataset_origin == 'synthetic filtered':
                dataset_calib = FilteredSyntheticImageDataset(G, max_synthetic_images, classifier)
            elif dataset_origin == 'validation augmented':
                transform = T.Compose([
                    T.RandomCrop(32, padding=4),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))])
                dataset_val_augmented = CIFAR10(root=path_dataset, train=False, transform=transform)
                dataset_calib, _ = torch.utils.data.random_split(dataset_val_augmented, [5000, 5000], generator=torch.Generator().manual_seed(seed)) # WARNING: USE SAME SPLIT AS BEFORE (CHECK SEED)
            else:
                raise ValueError('dataset_origin must be "validation", "synthetic" or "synthetic filtered" or "validation augmented"')
            
            # ITERATE OVER DATASET SIZE
            for subset_frac in [0.1, 0.3, 0.5, 0.7, 0.9, 1]:
                if dataset_origin in ['validation', 'synthetic', 'synthetic filtered', 'validation augmented']:
                    subset_calib_indices = torch.randperm(len(dataset_calib))[:int(subset_frac*len(dataset_calib))]
                elif dataset_origin in ['validation low confidence', 'validation high confidence']:
                    logits_list = []
                    for input, _ in DataLoader(dataset_calib, batch_size=128, shuffle=False):
                        input = input.cuda()
                        with torch.no_grad():
                            logits = classifier(input)
                        logits_list.append(logits)
                    logits = torch.cat(logits_list).cpu()
                    probs = torch.softmax(logits, dim=1)
                    confidences = probs.max(axis=1).values
                    if dataset_origin == 'validation low confidence':
                        subset_calib_indices = torch.argsort(confidences, descending=False)[:int(subset_frac*len(dataset_calib))]
                    elif dataset_origin == 'validation high confidence':
                        subset_calib_indices = torch.argsort(confidences, descending=True)[:int(subset_frac*len(dataset_calib))]
                subset_calib = Subset(dataset_calib, subset_calib_indices)   
                dataset_size = len(subset_calib)
                dataloader_calib = DataLoader(subset_calib, batch_size=128, shuffle=True)
                    
                # ITERATE OVER METHODS
                for method in methods:
                    try:
                        if method == 'baseline (no calibration)':
                            metrics_test = metrics_from_dataloader(classifier, dataloader_test)
                        elif method == 'temperature scaling':
                            classifier_calibrated = ModelWithTemperature(classifier).cuda()
                            classifier_calibrated.set_temperature(dataloader_calib)
                            metrics_test = metrics_from_dataloader(classifier_calibrated, dataloader_test)
                        elif method == 'vector scaling':
                            # Fit vector scaling
                            vs = VectorScaling(logit_input=True, logit_constant=0.0)
                            logits_list = []
                            labels_list = []
                            for input, label in dataloader_calib:
                                input = input.cuda()
                                with torch.no_grad():
                                    logits = classifier(input)
                                logits_list.append(logits)
                                labels_list.append(label)
                            logits = torch.cat(logits_list).cpu()
                            labels = torch.cat(labels_list).cpu()
                            vs.fit(logits.numpy(), labels.numpy())
                            metrics_test = metrics_from_dataloader(classifier, dataloader_test, vs)
                        else:
                            raise NotImplementedError
                    except Exception as e:
                        print(f'Failed for {model_name}, {dataset_origin}, {dataset_size}, {method}')
                        print(e)
                        continue

                    # Save results                    
                    df_res_test = pd.DataFrame([{'seed': seed, 'dataset origin': dataset_origin, 'dataset size': dataset_size, 'model': model_name, 'method': method, **metrics_test}])
                    f_path_test = path_results / 'benchmark_calibration_test.csv'
                    df_res_test.to_csv(f_path_test, mode='a', header=not(os.path.exists(f_path_test)), index=False)
                    print(df_res_test)
