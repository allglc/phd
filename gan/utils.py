import os
import torch
from torch import nn
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import shutil
from pytorch_fid.fid_score import calculate_fid_given_paths


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def get_noise(n_samples, z_dim, device='cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
      n_samples: the number of samples to generate, a scalar
      z_dim: the dimension of the noise vector, a scalar
      device: the device type
    '''
    return torch.randn(n_samples, z_dim, device=device)


def get_gradient_penalty(crit, real, fake, epsilon):
    '''
    Return the gradient of the critic's scores with respect to mixes of real and fake images.
    Parameters:
        crit: the critic model
        real: a batch of real images
        fake: a batch of fake images
        epsilon: a vector of the uniformly random proportions of real/fake per mixed image
    Returns:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    '''
    # Mix the images together
    mixed_images = real * epsilon + fake * (1 - epsilon)

    # Calculate the critic's scores on the mixed images
    mixed_scores = crit(mixed_images)
    
    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        # Note: You need to take the gradient of outputs with respect to inputs.
        # This documentation may be useful, but it should not be necessary:
        # https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad
        #### START CODE HERE ####
        inputs=mixed_images,
        outputs=mixed_scores,
        #### END CODE HERE ####
        # These other parameters have to do with the pytorch autograd engine works
        grad_outputs=torch.ones_like(mixed_scores), 
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(len(gradient), -1)

    # Calculate the magnitude of every row
    gradient_norm = gradient.norm(2, dim=1)
    
    # Penalize the mean squared distance of the gradient norms from 1
    penalty = torch.pow(torch.mean(gradient_norm - 1), 2)

    return penalty


# def get_gradient(crit, real, fake, epsilon):
#     '''
#     Return the gradient of the critic's scores with respect to mixes of real and fake images.
#     Parameters:
#         crit: the critic model
#         real: a batch of real images
#         fake: a batch of fake images
#         epsilon: a vector of the uniformly random proportions of real/fake per mixed image
#     Returns:
#         gradient: the gradient of the critic's scores, with respect to the mixed image
#     '''
#     # Mix the images together
#     mixed_images = real * epsilon + fake * (1 - epsilon)

#     # Calculate the critic's scores on the mixed images
#     mixed_scores = crit(mixed_images)
    
#     # Take the gradient of the scores with respect to the images
#     gradient = torch.autograd.grad(
#         # Note: You need to take the gradient of outputs with respect to inputs.
#         # This documentation may be useful, but it should not be necessary:
#         # https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad
#         #### START CODE HERE ####
#         inputs=mixed_images,
#         outputs=mixed_scores,
#         #### END CODE HERE ####
#         # These other parameters have to do with the pytorch autograd engine works
#         grad_outputs=torch.ones_like(mixed_scores), 
#         create_graph=True,
#         retain_graph=True,
#     )[0]
#     return gradient


# def gradient_penalty(gradient):
#     '''
#     Return the gradient penalty, given a gradient.
#     Given a batch of image gradients, you calculate the magnitude of each image's gradient
#     and penalize the mean quadratic distance of each magnitude to 1.
#     Parameters:
#         gradient: the gradient of the critic's scores, with respect to the mixed image
#     Returns:
#         penalty: the gradient penalty
#     '''
#     # Flatten the gradients so that each row captures one image
#     gradient = gradient.view(len(gradient), -1)

#     # Calculate the magnitude of every row
#     gradient_norm = gradient.norm(2, dim=1)
    
#     # Penalize the mean squared distance of the gradient norms from 1
#     #### START CODE HERE ####
#     penalty = torch.pow(torch.mean(gradient_norm - 1), 2)
#     #### END CODE HERE ####
#     return penalty


def test(dataloader, model, loss_function, device):

    model.eval()

    cum_loss, correct_pred = 0, 0
    for X, y in dataloader:

        X, y = X.to(device), y.to(device)

        with torch.no_grad():
            # Compute prediction, loss and correct predictions
            y_pred = model(X)
            loss = loss_function(y_pred, y)
            correct = (y_pred.argmax(1) == y).type(torch.float).sum().item()

            cum_loss += loss
            correct_pred += correct

    average_loss = cum_loss / len(dataloader)
    average_accuracy = correct_pred / len(dataloader.dataset)

    print('Test error: {}\nAccuracy {}\n'.format(average_loss, average_accuracy))

    return average_loss, average_accuracy


def compute_fid(gen, conditional=False, n_classes=10, path_target='', nb_images_to_generate = 10000, batch_size=1024, device='cuda'):

    num_avail_cpus = num_avail_cpus = len(os.sched_getaffinity(0))

    # 1. GENERATE AND SAVE IMAGES
    path_temp = Path('./temp')
    if path_temp.exists(): shutil.rmtree(path_temp)
    path_temp.mkdir() # create temporary directory

    img_idx = 0
    nb_batchs = nb_images_to_generate // batch_size
    for i in tqdm(range(nb_batchs + 1), 'Generate and save images'):

        if i == nb_batchs: batch_size = nb_images_to_generate % batch_size # handle last batch

        # Create inputs
        noise = torch.randn(batch_size, gen.z_dim, 1, 1, device=device)

        if conditional:
            label = torch.randint(0, n_classes, (batch_size, ))
            one_hot_labels = nn.functional.one_hot(label.to(device), n_classes)[:,:,None,None]
            noise = torch.cat((noise[:, :gen.z_dim-n_classes, :, :].float(), one_hot_labels.float()), dim=1)

        # Generate images
        with torch.no_grad():
            fake = gen(noise).detach().cpu()

        # Save images in temporary folder
        for fake_idx in range(len(fake)):
            save_image(fake[fake_idx], path_temp/f'{img_idx}.png')
            img_idx += 1

    # 2. COMPUTE FID
    print('Compute FID')
    fid = calculate_fid_given_paths(
        paths=[path_target, str(path_temp)], 
        batch_size=128,
        device='cuda',
        dims=2048,
        num_workers=min(num_avail_cpus, 8))

    print('FID: {:.2f}'.format(fid))

    shutil.rmtree(path_temp) # remove temporary directory

    return(fid)

