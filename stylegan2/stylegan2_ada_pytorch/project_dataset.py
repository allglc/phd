import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import pickle
import json
from multiprocessing import Pool
import multiprocessing as mp
import sys; sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from projector import project
from training.dataset import ImageFolderDataset
from classifiers.models import CNN_MNIST

path_results = Path.cwd().parent.parent / 'results'
model_name = '00016-mnist_stylegan2_blur_noise_maxSeverity3_proba50-cond-auto4'
path_model = path_results / 'stylegan2-training-runs' / model_name
data_name = 'mnistTest_stylegan2_blur_noise_maxSeverity3_proba50'
path_data = Path.cwd().parent.parent / 'data/MNIST' / f'{data_name}.zip'

NB_GPUS = torch.cuda.device_count()
classif_weight = 1e-3
len_dataset = 10000
idx_all = np.linspace(0, len_dataset, num=NB_GPUS+1, dtype=int)
idx_start_all = idx_all[:-1]
idx_end_all = idx_all[1:]


def load_data(idx_start, idx_end):
    
    ds = ImageFolderDataset(path_data, use_labels=True)
    ds_subset = torch.utils.data.Subset(ds, range(idx_start, idx_end))
    
    return ds_subset


def load_classifier(device):
    # predict digits
    classifier_digits = CNN_MNIST(output_dim=10).to(device)
    # classifier_digits.load_state_dict(torch.load(path_results / 'classifiers' / 'CNN_MNIST_weights_20220411_0826.pth', map_location=device)) # Confiance
    # classifier_digits.load_state_dict(torch.load(path_results / 'classifiers' / 'CNN_MNIST_weights_20220210_1601.pth', map_location=device))
    classifier_digits.load_state_dict(torch.load(path_results / 'classifiers' / 'CNN_mnist_stylegan2_blur_noise_maxSeverity3_proba50_20220510_1124.pth', map_location=device))
    classifier_digits.eval()
    return classifier_digits
    

def load_model(path_model, device):

    # get best model in folder
    with open(path_model / 'metric-fid50k_full.jsonl', 'r') as json_file:
        json_list = list(json_file)

    best_fid = 1e6
    for json_str in json_list:
        json_line = json.loads(json_str)
        if json_line['results']['fid50k_full'] < best_fid:
            best_fid = json_line['results']['fid50k_full']
            best_model = json_line['snapshot_pkl']
    print('Best FID: {:.2f} ; best model : {}'.format(best_fid, best_model))
    path_model = path_model / best_model

    with open(path_model, 'rb') as f:
        G = pickle.load(f)['G_ema'].to(device)  # torch.nn.Module

    return G


def project_subset(G, ds_subset, classifier_digits, device, subset_id):
    all_indices = np.zeros((len(ds_subset)), dtype=int)
    all_projected_w = np.zeros((len(ds_subset), G.w_dim))
    all_digits = np.zeros((len(ds_subset)), dtype=int)
    for i, (img, digit) in enumerate(ds_subset):
        img = torch.tensor(img, device=device)
        c = torch.tensor(digit, dtype=int, device=device)

        # project
        projected_w_steps = project(G, img, c,device=device,
                                    classifier_digits=classifier_digits, regularize_classif_weight=classif_weight)
        projected_w = projected_w_steps[-1][0].cpu().numpy()

        # record
        all_indices[i] = ds_subset.indices[i]
        all_projected_w[i] = projected_w
        all_digits[i] = digit.argmax()

        if subset_id == 0 and i % 100 == 0: print(f'subset 0: {i+1}/{len(ds_subset)}')

    # save
    outfile = path_results / 'projected_data' / f'{data_name}_model{model_name[:5]}_subset{subset_id}.npz'
    np.savez(outfile, indices=all_indices, projected_w=all_projected_w, digits=all_digits)

    
def main(subset_id):
    
    print(f'Launching process {subset_id+1}/{NB_GPUS}')

    idx_start = idx_start_all[subset_id]
    idx_end = idx_end_all[subset_id]
    
    device = f'cuda:{subset_id}'

    ds_subset = load_data(idx_start, idx_end)
    classifier_digits = load_classifier(device)
    G = load_model(path_model, device)
    
    project_subset(G, ds_subset, classifier_digits, device, subset_id)


if __name__ == "__main__":

    print(f'Project {data_name}.zip for model {model_name} on {NB_GPUS} GPUs')

    # Launch multiprocessing
    mp.set_start_method('spawn')
    with Pool() as p:
        p.map(main, range(NB_GPUS))

    # Concatenate in single file
    outfile = path_results / 'projected_data' / f'{data_name}_model{model_name[:5]}_subset0.npz'
    npzfile = np.load(outfile)
    all_indices = npzfile['indices']
    all_projected_w = npzfile['projected_w']
    all_digits = npzfile['digits']
    for i in range(1, NB_GPUS):
        outfile = path_results / 'projected_data' / f'{data_name}_model{model_name[:5]}_subset{i}.npz'
        npzfile = np.load(outfile)
        all_indices = np.concatenate((all_indices, npzfile['indices']))
        all_projected_w = np.concatenate((all_projected_w, npzfile['projected_w']))
        all_digits = np.concatenate((all_digits, npzfile['digits']))
    outfile = path_results / 'projected_data' / f'{data_name}_model{model_name[:5]}_classifWeight{classif_weight}_all.npz'
    np.savez(outfile, indices=all_indices, projected_w=all_projected_w, digits=all_digits)
    
    # remove temporary files
    for i in range(0, NB_GPUS):
        file = path_results / 'projected_data' / f'{data_name}_model{model_name[:5]}_subset{i}.npz'
        os.remove(file)