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
from models import Classifier, LinearClassifier

path_results = Path.cwd().parent / 'results'


@click.command()
@click.option('--n_samples', default=20000, help='Number of samples to generate')
@click.option('--noise', default=0.3, help='Noise to add to the data')
@click.option('--hidden_dim', default=32, help='Hidden dimension of the classifier')
@click.option('--linear', is_flag=True, default=False, help='Use linear classifier')
def main(**kwargs):
    
    config = EasyDict(kwargs)
    
    dm = MoonsDataModule(n_samples=config.n_samples, noise=config.noise)

    if config.linear:
        model = LinearClassifier()
    else:
        model = Classifier(config.hidden_dim)

    # create experiment folder to save results and logs
    timestamp = time.strftime('%Y-%m-%d_%H%M%S', time.localtime())
    path_results_exp = path_results / 'classifier' / timestamp
    if not path_results_exp.exists(): path_results_exp.mkdir(parents=True)
    logger = TensorBoardLogger(save_dir=str(path_results_exp), name='', version='')
    with open(os.path.join(path_results_exp, 'training_options.json'), 'wt') as f:
        json.dump(config, f, indent=2)

    # train
    trainer = pl.Trainer(accelerator="auto", devices=1, max_epochs=100, logger=logger, auto_select_gpus=True)
    trainer.fit(model, datamodule=dm)
    

if __name__ == '__main__':
    main()