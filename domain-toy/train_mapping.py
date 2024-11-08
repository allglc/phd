from utils import assign_free_gpus
assign_free_gpus()

import os
import time
import json
import click
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn.functional as F
import pytorch_lightning as pl


from utils import EasyDict
from models import Classifier, LinearClassifier, GAN, ContrastiveMapping
from generate_data import MoonsDataModule

path_results = Path.cwd().parent / 'results'

@click.command()
@click.option('--n_samples', default=20000, help='Number of samples to generate')
@click.option('--noise', default=0.3, help='Noise to add to the data')
@click.option('--use_contrastive_loss', default=False)
@click.option('--linear', default=True, help='Use linear classifier')
@click.option('--trainable_distrib', default=False, help='Trainable distribution')
@click.option('--classif_loss', default=False, help='Use classif loss')
def main(**kwargs):
    
    config = EasyDict(kwargs)
    
    dm = MoonsDataModule(n_samples=config.n_samples, noise=config.noise)
    dm.setup() # required to access data_train
    
    if config.noise == 0.3:
        config.path_classif = str(path_results / 'classifier' / '2022-12-06_163615_linear_noise0.3' / 'checkpoints' / 'epoch=99-step=6300.ckpt')
        config.path_gan = str(path_results / 'GAN' / '2022-11-29_152439_noise0.3' / 'checkpoints' / 'epoch=99-step=12600.ckpt')
    elif config.noise == 0.1:
        config.path_classif = str(path_results / 'classifier' / '2022-11-29_152904_linear_noise0.1' / 'checkpoints' / 'epoch=99-step=6300.ckpt')
        config.path_gan = str(path_results / 'GAN' / '2022-11-29_152457_noise0.1' / 'checkpoints' / 'epoch=99-step=12600.ckpt')
    
    if config.linear:
        classifier = LinearClassifier.load_from_checkpoint(config.path_classif)
    else:
        raise NotImplementedError('only linear classif for now')
    
    
    gan = GAN.load_from_checkpoint(config.path_gan)
    model = ContrastiveMapping(config.noise, gan, classifier, use_contrastive_loss=config.use_contrastive_loss, contrastive_loss='SupCon', 
                               use_adv_loss=True, use_reconstruction_loss=True, use_classif_reconstruction_loss=config.classif_loss, 
                               encode_in='u', trainable_distrib=config.trainable_distrib)

    # create experiment folder to save results and logs
    timestamp = time.strftime('%Y-%m-%d_%H%M%S', time.localtime())

    path_results_exp = path_results / 'contrastiveMapping' / timestamp
    if not path_results_exp.exists(): path_results_exp.mkdir(parents=True)
    logger = TensorBoardLogger(save_dir=str(path_results_exp), name='', version='')
    with open(os.path.join(path_results_exp, 'training_options.json'), 'wt') as f:
        json.dump(config, f, indent=2)

    # train
    trainer = pl.Trainer(accelerator='auto', devices=1, max_epochs=60, logger=logger, auto_select_gpus=True)
    trainer.fit(model, datamodule=dm)
    
if __name__ == '__main__':
    main()