import argparse
import os
import datetime
import math
from pathlib import Path
from tqdm.auto import tqdm
import torch
from torch.optim import Adam
from torchvision import utils, transforms as T
from torchvision.datasets import MNIST, ImageFolder
from torch.utils.data import DataLoader

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import cycle, num_to_groups
from denoising_diffusion_pytorch.classifier_free_guidance import Unet as Unet_cond, GaussianDiffusion as GaussianDiffusion_cond


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DDPM network on a given dataset')
    parser.add_argument('--dataset', default='mnist', type=str, help='name of the dataset to use for training')
    parser.add_argument('--classCond', action='store_true', help='whether to use class conditional training')
    args = parser.parse_args()

    if args.dataset == 'mnist':
        path_data = Path(os.path.expandvars('$WORK/DATA/MNIST/mnist_img'))
        channels = 1
        image_size = 32
        num_classes = 10
        augment_horizontal_flip = False
        batch_size = 256
    elif args.dataset == 'imagenet':
        path_data = Path(os.path.expandvars('$DSDIR/imagenet/train'))
        channels = 3
        image_size = 64
        num_classes = 1000
        augment_horizontal_flip = True
        batch_size = 128
    else:
        raise ValueError('Invalid dataset name provided')
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    results_folder = f'../results/{timestamp}_{args.dataset}'
    
    train_num_steps = 1000000

        
    if args.classCond:
        save_and_sample_every = 1000
        num_samples = 25
        results_folder += '_classCond'
        results_folder = Path(results_folder)
        results_folder.mkdir(exist_ok=True)

        if args.dataset == 'mnist':
            dataset = MNIST(Path(os.path.expandvars('$DSDIR/')), download=False, transform=T.Compose([T.Resize(image_size), T.ToTensor()]))
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
            dataloader = cycle(dataloader)
        elif args.dataset == 'imagenet':
            dataset = ImageFolder(Path(os.path.expandvars('$DSDIR/imagenet/train')), transform=T.Compose([T.Resize(image_size), T.CenterCrop(image_size), T.ToTensor()]))
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
            dataloader = cycle(dataloader)
        else:
            raise ValueError('Invalid dataset name provided')
    
        model = Unet_cond(
            dim = 64,
            dim_mults = (1, 2, 4, 8),
            channels = channels,
            num_classes = num_classes,
            cond_drop_prob = 0.5
        )

        diffusion = GaussianDiffusion_cond(
            model,
            image_size = image_size,
            timesteps = 1000,
            sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        ).cuda()

        opt = Adam(diffusion.parameters(), lr=1e-4, betas=(0.9, 0.99))
        
        # Training loop
        step = 0
        with tqdm(initial = step, total = train_num_steps) as pbar:

            while step < train_num_steps:
                diffusion.train()
                images, classes = next(dataloader)
                images, classes = images.cuda(), classes.cuda()

                loss = diffusion(images, classes=classes)
                loss.backward()
                
                pbar.set_description(f'loss: {loss.item():.4f}')

                opt.step()
                opt.zero_grad()

                step += 1

                if step != 0 and step % save_and_sample_every == 0:
                    diffusion.eval()

                    with torch.inference_mode():
                        milestone = step // save_and_sample_every
                        batches = num_to_groups(num_samples, batch_size)
                        all_images_list = list(map(lambda n: diffusion.sample(classes=torch.randint(0, num_classes, (n,)).cuda(), cond_scale = 6.), batches))
                    all_images = torch.cat(all_images_list, dim = 0)

                    utils.save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(num_samples)))
                    torch.save(diffusion.state_dict(), str(results_folder / f'model_checkpoint-{milestone}.pt'))

                pbar.update(1)
        
        
        
    else:
        model = Unet(
            dim = 64,
            dim_mults = (1, 2, 4, 8),
            flash_attn = True,
            channels = channels
        )

        diffusion = GaussianDiffusion(
            model,
            image_size = image_size,
            timesteps = 1000,           # number of steps
            sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        )

        trainer = Trainer(
            diffusion,
            path_data,
            augment_horizontal_flip=augment_horizontal_flip,
            results_folder=results_folder,
            train_batch_size = batch_size,
            train_lr = 1e-4,
            train_num_steps = train_num_steps,         # total training steps
            gradient_accumulate_every = 2,    # gradient accumulation steps
            ema_decay = 0.995,                # exponential moving average decay
            amp = True,                       # turn on mixed precision
            calculate_fid = False              # whether to calculate fid during training
        )

        trainer.train()
        
        
