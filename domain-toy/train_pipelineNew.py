from utils import assign_free_gpus
assign_free_gpus()

import os
import time
import json
import click
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn.functional as F
import pytorch_lightning as pl

from utils import EasyDict
from generate_data import MoonsDataModule
from models import GAN, WGAN_GP, LinearClassifier
from pipelineNew import PipelineNew

np.random.seed(0)
torch.manual_seed(0)
path_results = Path.cwd().parent / 'results'


@click.command()
@click.option('--n_samples', default=20000, help='Number of samples to generate in visualization')
@click.option('--noise', default=0.3, help='Noise to add to the data')
@click.option('--hidden_dim', default=64, help='Hidden dimension of the GAN')
@click.option('--latent_dim', default=8, help='Latent dimension of the GAN')
@click.option('--class_conditioning', default='label', help='class conditioning: "label" or "prediction" or None (default)')
@click.option('--classifier_conditioning', default='TCP', help='classifier conditioning: "TCP" or "MSP" or None (default)')
def main(**kwargs):
    
    config = EasyDict(kwargs)
    config.c_dim = 2
    
    dm = MoonsDataModule(n_samples=config.n_samples, noise=config.noise)
    
    if config.noise == 0.1:
        path_classifier = path_results / 'classifier' / '2022-11-29_152904_linear_noise0.1' / 'checkpoints' / 'epoch=99-step=6300.ckpt'
    elif config.noise == 0.3:
        path_classifier = path_results / 'classifier' / '2023-02-03_110311_linear_noise0.3' / 'checkpoints' / 'epoch=99-step=6300.ckpt'
    else:
        raise ValueError('Noise value not supported')
    config.classifier = str(path_classifier)
    classifier = LinearClassifier.load_from_checkpoint(str(path_classifier))
    
    model = PipelineNew(config.latent_dim, 2, config.n_samples, config.noise, c_dim=config.c_dim, hidden_dim=config.hidden_dim,
                     classifier=classifier, class_conditioning=config.class_conditioning, classifier_conditioning=config.classifier_conditioning)
        
    # create experiment folder to save results and logs
    timestamp = time.strftime('%Y-%m-%d_%H%M%S', time.localtime())
    tag = f'_noise{config.noise}_classCond{config.class_conditioning}_classifCond{config.classifier_conditioning}'
    path_results_exp = path_results / 'pipelineNew' / (timestamp+tag)
    if not path_results_exp.exists(): path_results_exp.mkdir(parents=True)
    logger = TensorBoardLogger(save_dir=path_results_exp, name='', version='')
    with open(os.path.join(path_results_exp, 'training_options.json'), 'wt') as f:
        json.dump(config, f, indent=2)

    # train
    trainer = pl.Trainer(accelerator="auto", devices=1, max_epochs=100, logger=logger, auto_select_gpus=True)
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    main()