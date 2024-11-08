import matplotlib.pyplot as plt
plt.style.use('seaborn')
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# from umap import UMAP
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.distributions.normal import Normal

from generate_data import MoonsDataset
from losses import SupConLoss

torch.manual_seed(0)
np.random.seed(0)
rng = np.random.default_rng(0)


class LinearClassifier(pl.LightningModule):
    
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.one_linear_layer = nn.Linear(2, 1)

    def forward(self, x):
        return self.one_linear_layer(x)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(1)
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        acc = (torch.sigmoid(y_hat).round() == y).sum() / y.shape[0]
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(1)
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        acc = (torch.sigmoid(y_hat).round() == y).sum() / y.shape[0]
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss    
    
    
class Classifier(pl.LightningModule):
    
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(1)
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        acc = (torch.sigmoid(y_hat).round() == y).sum() / y.shape[0]
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(1)
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        acc = (torch.sigmoid(y_hat).round() == y).sum() / y.shape[0]
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss        


class Generator(nn.Module):
    def __init__(self, latent_dim, c_dim, data_dim, hidden_dim):
        super().__init__()
        self.mapping = nn.Sequential(
            nn.Linear(latent_dim+c_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU())
        self.synthesis = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim*2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, data_dim))
        
    def forward(self, z, c=None):
        if c is not None: 
            z = torch.cat([z, c], dim=1)
        w = self.mapping(z)
        x = self.synthesis(w)
        return x
    
class GeneratorNew(nn.Module):
    "condition added after mapping"
    def __init__(self, latent_dim, c_dim, data_dim, hidden_dim):
        super().__init__()
        self.mapping = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU())
        self.synthesis = nn.Sequential(
            nn.Linear(latent_dim+c_dim, hidden_dim*2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, data_dim))
        
    def forward(self, z, c=None):
        w = self.mapping(z)
        if c is not None: 
            w = torch.cat([w, c], dim=1)
        x = self.synthesis(w)
        return x


class Discriminator(nn.Module):
    def __init__(self, data_dim, c_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(data_dim+c_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1))
    
    def forward(self, x, c=None):
        if c is not None:
            x = torch.cat([x, c], dim=1)
        y = self.model(x)
        return y

    
class WGAN_GP(pl.LightningModule):
    def __init__(self, latent_dim, data_dim, data_n_samples, data_noise, c_dim, hidden_dim):
        super().__init__()
        self.save_hyperparameters()
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim, c_dim, data_dim, hidden_dim)
        self.discriminator = Discriminator(data_dim, c_dim, hidden_dim)
        self.data = MoonsDataset(n_samples=data_n_samples, noise=data_noise)
        self.c_dim = c_dim # class dimensionality, 0=unconditional
        self.lambda_gp = 10 # gradient penalty weight
        self.num_steps = 0
        
    def configure_optimizers(self):
        optim_disc = torch.optim.Adam(self.discriminator.parameters(), lr=0.001, betas=(0, 0.99))
        optim_gen = torch.optim.Adam(self.generator.parameters(), lr=0.001, betas=(0, 0.90))
        return optim_disc, optim_gen
    
    def _get_gradient_penalty(self, crit, real, fake, c, epsilon):
        '''
        From Coursera GAN Specialization
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
        mixed_scores = crit(mixed_images, c)
        
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
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        self.num_steps += 1
        x_real, y_real = batch
        z = torch.randn(x_real.shape[0], self.latent_dim, device=self.device)
        c = F.one_hot(y_real.long(), num_classes=self.c_dim) if self.c_dim > 0 else None
                    
        # DISCRIMINATOR
        if optimizer_idx == 0:
            with torch.no_grad():
                x_fake = self.generator(z, c)
            pred_fake = self.discriminator(x_fake, c)
            pred_real = self.discriminator(x_real, c)
            
            epsilon = torch.rand(x_real.shape[0], 1, device=self.device, requires_grad=True)
            gradient_penalty = self._get_gradient_penalty(self.discriminator, x_real, x_fake, c, epsilon)
            
            loss = (pred_fake - pred_real).mean() + self.lambda_gp*gradient_penalty
            self.log('loss_disc', loss)
            return loss
        
        # GENERATOR
        elif optimizer_idx == 1:
            if self.num_steps % 5 == 0:
                x_fake = self.generator(z, c)
                pred_fake = self.discriminator(x_fake, c)
                loss = -pred_fake.mean()
                self.log('loss_gen', loss)
                return loss
            else:
                return None
            
    def on_train_epoch_end(self):
        # log images
        n_samples = 2000
        z = torch.randn(n_samples, self.latent_dim, device=self.device)
        if self.c_dim > 0:
            rnd_label = torch.randint(self.c_dim, size=(z.shape[0],), device=self.device)
            c = F.one_hot(rnd_label, num_classes=self.c_dim)
        else:
            c = None
        x_fake = self.generator(z, c).detach().cpu().numpy()
        plt.scatter(self.data[:n_samples][0][:, 0], self.data[:n_samples][0][:, 1], alpha=0.5, c=['C0' if y == 0 else 'C1' for y in self.data[:n_samples][1]])
        color = ['C2' if y == 0 else 'C3' for y in rnd_label] if self.c_dim > 0 else 'k'
        plt.scatter(x_fake[:, 0], x_fake[:, 1], alpha=0.5, c=color)
        tensorboard_logger = self.logger.experiment
        tensorboard_logger.add_figure("generated_images", plt.gcf(), self.current_epoch)
        
class Embedding(nn.Module):
    ''' Converts class label into a sample from a gaussian mixture.'''
    def __init__(self, c_dim=2):
        super().__init__()
        self.mean0, self.mean1 = torch.tensor(0.), torch.tensor(0.)
        self.std0, self.std1 = torch.tensor(1.), torch.tensor(1.)
        self.mean0, self.mean1 = torch.nn.parameter.Parameter(self.mean0), torch.nn.parameter.Parameter(self.mean1)
        # self.std0, self.std1 = torch.nn.parameter.Parameter(self.std0), torch.nn.parameter.Parameter(self.std1)
            
    def forward(self, y):
        idx_class0 = (y == 0)
        idx_class1 = (y == 1)
        c_gen = torch.zeros((y.shape[0], 1), device=y.device)
        c_gen[idx_class0] = self.mean0 + self.std0 * Normal(0, 1).sample((idx_class0.sum(), 1)).to(y.device)
        c_gen[idx_class1] = self.mean1 + self.std1 * Normal(0, 1).sample((idx_class1.sum(), 1)).to(y.device)
        return c_gen


class GAN(pl.LightningModule):
    
    def __init__(self, latent_dim, data_dim, data_n_samples, data_noise, c_dim, hidden_dim,
                 class_conditioning=None, classifier_conditioning=None, classifier=None):
        """GAN model.

        Args:
            latent_dim (int): latent dimension without conditioning
            data_dim (int): data dimension
            data_n_samples (int): dataset size
            data_noise (float): data noise parameter
            c_dim (int): number of classes
            hidden_dim (int): hidden dim in models
            class_conditioning (string/bool): 'one-hot' or 'gaussian' or None (default)
            classifier_conditioning (string/bool): 'TCP', or 'MSP', or None (default)
        """
        
        super().__init__()
        self.save_hyperparameters(ignore=['classifier'])
        self.latent_dim = latent_dim
        self.class_conditioning = class_conditioning
        self.classifier_conditioning = classifier_conditioning
        
        # class conditioning
        if self.class_conditioning == 'one-hot':
            self.condition_dim = c_dim
        elif self.class_conditioning == 'gaussian':
            self.condition_dim = 1 # only for 2 classes (dim = nb_classes-1)
            self.embed = Embedding()
            self.distrib0_samples = []
            self.distrib1_samples = []
            self.distrib0_params = []
            self.distrib1_params = []
        elif self.class_conditioning == 'prediction':
            self.condition_dim = c_dim
        elif self.class_conditioning is None:
            self.condition_dim = 0
        else:
            raise ValueError('class_conditioning must be one-hot, gaussian, or None')
        
        # classifier conditioning
        if self.classifier_conditioning == 'TCP' or self.classifier_conditioning == 'MSP':
            self.condition_dim += 1
            self.classifier = classifier
        elif self.classifier_conditioning is None:
            pass
        else:
            raise ValueError('classifier_conditioning must be TCP, MSP, or None')

        self.generator = Generator(latent_dim, self.condition_dim, data_dim, hidden_dim)
        self.discriminator = Discriminator(data_dim, self.condition_dim, hidden_dim)
        self.data = MoonsDataset(n_samples=data_n_samples, noise=data_noise)

            
    def configure_optimizers(self):
        optim_gen = torch.optim.Adam(self.generator.parameters(), lr=0.001, betas=(0, 0.99))
        optim_disc = torch.optim.Adam(self.discriminator.parameters(), lr=0.001, betas=(0, 0.99))
        optimizers = [optim_gen, optim_disc]
        if self.class_conditioning == 'gaussian':
            optim_embed = torch.optim.Adam(self.embed.parameters(), lr=0.001, betas=(0, 0.99))
            optimizers.append(optim_embed)
        return optimizers
    
    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)
    
    def _get_classifier_conditioning(self, x, y=None):
        logits = self.classifier(x)
        probas_class1 = torch.sigmoid(logits)
        probas_class0 = 1 - probas_class1
        if self.classifier_conditioning == 'TCP':
            y_as_idx = F.one_hot(y.long(), num_classes=2).bool()
            probas = torch.cat((probas_class0, probas_class1), dim=1)
            confidence = probas[y_as_idx].unsqueeze(1)
        elif self.classifier_conditioning == 'MSP':
            confidence = torch.maximum(probas_class0, probas_class1)
        else:
            raise ValueError('Unknown confidence metric')
        return confidence
                
    def training_step(self, batch, batch_idx, optimizer_idx):
        x_real, y_real = batch
        z = torch.randn(x_real.shape[0], self.latent_dim, device=self.device)
        
        if self.class_conditioning == 'one-hot':
            c_real = F.one_hot(y_real.long(), num_classes=2)
            # rnd_label = torch.randint(2, size=(z.shape[0],), device=self.device)
            # c_fake = F.one_hot(rnd_label, num_classes=2)
            c_fake_gen = c_real
            c_fake_disc = c_fake_gen
        elif self.class_conditioning == 'gaussian': # only for 2 classes            
            c_real = y_real.long().unsqueeze(1) # vector of 0 and 1 (only 2 classes)
            c_fake_gen = self.embed(y_real)
            c_fake_disc = y_real.long().unsqueeze(1) # vector of 0 and 1 (only 2 classes)
        else:
            c_real = None
            c_fake_gen = None
            c_fake_disc = None
            
        if self.classifier_conditioning is not None:
            confidence = self._get_classifier_conditioning(x_real, y_real)
            c_real = torch.cat((c_real, confidence), dim=1) if c_real is not None else confidence
            c_fake_gen = torch.cat((c_fake_gen, confidence), dim=1) if c_fake_gen is not None else confidence
            c_fake_disc = torch.cat((c_fake_disc, confidence), dim=1) if c_fake_disc is not None else confidence        
        
        # generator
        if optimizer_idx == 0:
            x_fake = self.generator(z, c_fake_gen)
            pred_fake = self.discriminator(x_fake, c_fake_disc)
            loss_gen = self.adversarial_loss(pred_fake, torch.ones_like(pred_fake, device=self.device)) # fakes should be predicted as true
            self.log('loss_gen', loss_gen)
            return loss_gen
        
        # discriminator
        elif optimizer_idx == 1:
            x_fake = self.generator(z, c_fake_gen).detach()
            pred_real = self.discriminator(x_real, c_real)
            pred_fake = self.discriminator(x_fake, c_fake_disc)
            loss_disc = 0.5 * (self.adversarial_loss(pred_fake, torch.zeros_like(pred_fake, device=self.device)) # fakes should be predicted as false
                               + self.adversarial_loss(pred_real, torch.ones_like(pred_real, device=self.device))) # reals should be predicted as true
            self.log('loss_disc', loss_disc)
            return loss_disc
        
        # embedding
        elif (optimizer_idx == 2) and (self.class_conditioning == 'gaussian'):
            x_fake = self.generator(z, c_fake_gen)
            pred_fake = self.discriminator(x_fake, c_fake_disc)
            loss_embed = self.adversarial_loss(pred_fake, torch.ones_like(pred_fake, device=self.device)) # fakes should be predicted as true
            self.log('loss_embed', loss_embed)
            return loss_embed
        
        else:
            raise ValueError('issue with optimizers configuration')
    
    def on_train_epoch_end(self):
        n_samples = 2000
        z = torch.randn(n_samples, self.latent_dim, device=self.device)
        x_real = self.data[:n_samples][0]
        y_real = self.data[:n_samples][1]
        
        # class conditioning
        if self.class_conditioning == 'one-hot':
            # rnd_label = torch.randint(2, size=(n_samples,), device=self.device)
            rnd_label = y_real.long()
            class_embed = F.one_hot(rnd_label, num_classes=2).to(self.device)
        elif self.class_conditioning == 'gaussian': # only for 2 classes
            # rnd_label = torch.randint(2, size=(n_samples,), device=self.device)
            rnd_label = y_real.long()
            with torch.no_grad():
                class_embed = self.embed(rnd_label.to(self.device))
            idx_class0 = (rnd_label == 0).cpu()
            idx_class1 = (rnd_label == 1).cpu()
        else:
            class_embed = None
        c = class_embed
        
        # classifier conditioning
        if self.classifier_conditioning is not None:
            with torch.no_grad():
                confidence_out = self._get_classifier_conditioning(x_real.to(self.device), y_real.to(self.device))
                confidence_in = confidence_out
            c = torch.cat((c, confidence_in), dim=1) if c is not None else confidence_in

        # log images
        with torch.no_grad():
            mapping_in = torch.cat((z, c), dim=1) if c is not None else z
            w = self.generator.mapping(mapping_in)
            x_fake = self.generator.synthesis(w).cpu().numpy()
        plt.scatter(self.data[:n_samples][0][:, 0], self.data[:n_samples][0][:, 1], alpha=0.5, c=['C0' if y == 0 else 'C1' for y in self.data[:n_samples][1]])
        color = ['C2' if y == 0 else 'C3' for y in rnd_label] if self.class_conditioning is not None else 'k'
        plt.scatter(x_fake[:, 0], x_fake[:, 1], alpha=0.5, c=color)
        tensorboard_logger = self.logger.experiment
        tensorboard_logger.add_figure("generated_data", plt.gcf(), self.current_epoch)

        if self.class_conditioning == 'gaussian': # only for 2 classes
            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12,6), constrained_layout=True)
            ax0.set_title('class 0')
            ax1.set_title('class 1')
            scat0 = ax0.scatter(x_fake[idx_class0, 0], x_fake[idx_class0, 1], alpha=0.5, c=c[idx_class0, 0].cpu().numpy(), cmap='viridis')
            scat1 = ax1.scatter(x_fake[idx_class1, 0], x_fake[idx_class1, 1], alpha=0.5, c=c[idx_class1, 0].cpu().numpy(), cmap='viridis')
            cb = fig.colorbar(scat0, ax=ax0)
            cb.set_alpha(1)
            cb.draw_all()
            cb = fig.colorbar(scat1, ax=ax1)
            cb.set_alpha(1)
            cb.draw_all()
            tensorboard_logger.add_figure("generated_data_gauss", plt.gcf(), self.current_epoch)
            
            self.distrib0_samples.append(class_embed[rnd_label == 0].squeeze().cpu().numpy())
            self.distrib1_samples.append(class_embed[rnd_label == 1].squeeze().cpu().numpy())
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            for i in range(self.current_epoch+1):
                y0 = self.distrib0_samples[i]
                x0 = np.array([i for _ in range(len(y0))])
                ax1.scatter(x0, y0, alpha=0.5, c=f'C0', label=f'class 0')
                y1 = self.distrib1_samples[i]
                x1 = np.array([i for _ in range(len(y1))])
                ax1.scatter(x1, y1, alpha=0.5, c=f'C1', label=f'class 1')
            ax1.set_xlabel('epoch')
            ax1.set_ylabel('gauss mixt samples')
            
            self.distrib0_params.append([self.embed.mean0.detach().cpu().numpy(), self.embed.std0.detach().cpu().numpy()])
            self.distrib1_params.append([self.embed.mean1.detach().cpu().numpy(), self.embed.std1.detach().cpu().numpy()])
            x = np.array([i for i in range(self.current_epoch+1)])
            means0 = np.array(self.distrib0_params)[:, 0]
            stds0 = np.array(self.distrib0_params)[:, 1]
            ax2.plot(x, means0, c='C0', label='class 0')
            ax2.fill_between(x, means0-stds0, means0+stds0, color='C0', alpha=0.3)
            means1 = np.array(self.distrib1_params)[:, 0]
            stds1 = np.array(self.distrib1_params)[:, 1]
            ax2.plot(x, means1, c='C1', label='class 1')
            ax2.fill_between(x, means1-stds1, means1+stds1, color='C1', alpha=0.3)
            ax2.legend()
            ax2.set_xlabel('epoch')
            ax2.set_ylabel('gauss mixt params')
            tensorboard_logger.add_figure("distributions_evolution", plt.gcf(), self.current_epoch)

        if (self.current_epoch == 0) or ((self.current_epoch+1)%10 == 0):
            # analysis of W
            w_pca = PCA(n_components=2).fit_transform(w.cpu().numpy())
            w_tsne = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(w.cpu().numpy())
            # w_umap = UMAP().fit_transform(w.cpu().numpy())
            fig, axs = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
            for algo, ax_, w_embedded in zip(['PCA', 't-SNE'], axs, [w_pca, w_tsne]):
                if self.class_conditioning is not None:
                    ax_[0].set_title(f'{algo} with input class as color')
                    ax_[0].scatter(w_embedded[y_real==0, 0], w_embedded[y_real==0, 1], c='C0', alpha=0.5, label='class 0')
                    ax_[0].scatter(w_embedded[y_real==1, 0], w_embedded[y_real==1, 1], c='C1', alpha=0.5, label='class 1')
                    ax_[0].legend()
                else:
                    ax_[0].set_title(f'{algo} with input class as color')
                    ax_[0].scatter(w_embedded[:, 0], w_embedded[:, 1], c='k', alpha=0.5, label='no input class given')
                    ax_[0].legend()
                if self.classifier_conditioning is not None:
                    ax_[1].set_title(f'{algo} with Confidence as color')
                    scat = ax_[1].scatter(w_embedded[:, 0], w_embedded[:, 1], c=confidence_in.cpu().numpy(), cmap='viridis', alpha=0.5)
                    cb = fig.colorbar(scat, ax=ax_[1])
                    cb.set_alpha(1)
                    cb.draw_all()
            tensorboard_logger.add_figure('W', plt.gcf(), self.current_epoch)    
                
        if self.classifier_conditioning is not None:
            # confidence
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6), constrained_layout=True)
            ax1.set_title('Confidence computed from real data')
            ax2.set_title('generated data conditioned by Confidence value')
            scat1 = ax1.scatter(x_real[:, 0].cpu().numpy(), x_real[:, 1].cpu().numpy(), c=confidence_out.cpu().numpy(), cmap='viridis', alpha=0.5)
            ax2.scatter(x_fake[:, 0], x_fake[:, 1], c=confidence_in.cpu().numpy(), cmap='viridis', alpha=0.5)
            cb = fig.colorbar(scat1, ax=ax1)
            cb.set_alpha(1)
            cb.draw_all()
            tensorboard_logger.add_figure("Confidence", fig, self.current_epoch)
   
            # distances in W
            confidence_one = torch.ones(size=(n_samples, 1)).to(self.device)
            mapping_in = torch.cat((z, confidence_one), dim=1) if self.class_conditioning is None else torch.cat((z, class_embed, confidence_one), dim=1)
            with torch.no_grad():
                w = self.generator.mapping(mapping_in)
                w_center_confidence_one = w.mean(0, keepdim=True)
            confidence_rnd = torch.rand(size=(n_samples, 1)).to(self.device)
            mapping_in = torch.cat((z, confidence_rnd), dim=1) if self.class_conditioning is None else torch.cat((z, class_embed, confidence_rnd), dim=1)
            with torch.no_grad():
                w = self.generator.mapping(mapping_in)
            distances = torch.cdist(w, w_center_confidence_one).cpu().numpy()
            plt.figure()
            plt.title('distances from W center of confidence=1')
            plt.scatter(distances, confidence_rnd.cpu().numpy(), alpha=0.5)
            plt.xlabel('distance from W center of confidence=1')
            plt.ylabel('confidence value')
            tensorboard_logger.add_figure('distances from W center of confidence=1', plt.gcf(), self.current_epoch)
            
            
class Mapping(pl.LightningModule):
    def __init__(self, latent_dim_in, latent_dim_out, hidden_dim=32, trainable_distrib=False):
        super().__init__()
        # self.save_hyperparameters()
        self.hidden_dim = hidden_dim
        self.mean0, self.mean1 = torch.tensor(-1.5), torch.tensor(1.5)
        self.std0, self.std1 = torch.tensor(1.), torch.tensor(1.)
        if trainable_distrib:
            self.mean0, self.mean1 = torch.nn.parameter.Parameter(self.mean0), torch.nn.parameter.Parameter(self.mean1)
            self.std0, self.std1 = torch.nn.parameter.Parameter(self.std0), torch.nn.parameter.Parameter(self.std1)
        self.c_dim = 1 # 2 classes are represented in 1D
        self.latent_dim_in = latent_dim_in
        self.model = nn.Sequential(
            nn.Linear(latent_dim_in+self.c_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim_out)
        )
        
    def sample_u(self, size=128):
        u_noise = torch.randn(size, self.latent_dim_in, device=self.device)
        u_domain = torch.zeros((size, self.c_dim), device=self.device)
        # only for 2 classes
        idx_class0 = (rng.random(size) > 0.5)
        idx_class1 = np.logical_not(idx_class0)
        u_domain[idx_class0] = self.mean0 + self.std0 * Normal(0, 1).sample((idx_class0.sum(), self.c_dim)).to(self.device)
        u_domain[idx_class1] = self.mean1 + self.std1 * Normal(0, 1).sample((idx_class1.sum(), self.c_dim)).to(self.device)
        u = torch.cat([u_noise, u_domain], dim=1)
        c = torch.tensor(idx_class1.astype(int)).to(self.device)
        return u, c
        
    def forward(self, u):
        return self.model(u)
    
    
class Encoder(pl.LightningModule):
    def __init__(self, dim_in, latent_dim_out, hidden_dim=32):
        super().__init__()
        # self.save_hyperparameters()
        self.hidden_dim = hidden_dim
        self.model = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim_out)
        )
        
    def forward(self, x):
        return self.model(x)
    
    
class ContrastiveMapping(pl.LightningModule):
    def __init__(self, data_noise, gan, classifier, use_contrastive_loss, contrastive_loss, use_adv_loss, 
                 use_reconstruction_loss, use_classif_reconstruction_loss=False, encode_in='u', trainable_distrib=False):
        super().__init__()
        self.save_hyperparameters(ignore=['gan', 'classifier'])
        self.mapping = Mapping(gan.latent_dim, gan.latent_dim, trainable_distrib=trainable_distrib)
        self.gan = gan
        self.classifier = classifier
        if encode_in =='u':
            self.encoder = Encoder(2, gan.latent_dim+self.mapping.c_dim)
        elif encode_in =='w':
            self.encoder = Encoder(2, gan.latent_dim)
        
        self.data = MoonsDataset(n_samples=20000, noise=data_noise)
        self.use_contrastive_loss = use_contrastive_loss
        self.contrastive_loss = contrastive_loss
        if contrastive_loss == 'SupCon':
            self.supConLoss = SupConLoss()
        elif contrastive_loss == 'triplet':
            self.tripletLoss = nn.TripletMarginLoss(margin=1e-3)
        else:
            raise ValueError(f'contrastive_loss {contrastive_loss} not implemented')
        self.use_adv_loss = use_adv_loss
        self.use_reconstruction_loss = use_reconstruction_loss
        self.use_classif_reconstruction_loss = use_classif_reconstruction_loss
        self.encode_in = encode_in
        
    def configure_optimizers(self):
        stop_train_mapping_epoch = self.trainer.fit_loop.max_epochs // 2
        
        lr_adv = 0.002 if self.use_adv_loss else 0
        lr_contrastive = 0.001 if self.use_contrastive_loss else 0
        lr_encoder = 0.001 if self.use_reconstruction_loss else 0
        
        optim_contrastive = torch.optim.Adam(self.mapping.parameters(), lr=lr_contrastive, weight_decay=1e-4) # only learn mapping network
        optim_adv_gen = torch.optim.Adam(self.mapping.parameters(), lr=lr_adv, betas=(0, 0.99))
        optim_adv_disc = torch.optim.Adam(self.gan.discriminator.parameters(), lr=lr_adv, betas=(0, 0.99))
        optim_encoder = torch.optim.Adam(self.encoder.parameters(), lr=lr_encoder, weight_decay=1e-4)
        
        scheduler_contrastive = torch.optim.lr_scheduler.MultiStepLR(optim_contrastive, milestones=[stop_train_mapping_epoch], gamma=0)
        scheduler_adv_gen = torch.optim.lr_scheduler.MultiStepLR(optim_adv_gen, milestones=[stop_train_mapping_epoch], gamma=0)
        scheduler_adv_disc = torch.optim.lr_scheduler.MultiStepLR(optim_adv_disc, milestones=[stop_train_mapping_epoch], gamma=0)
        
        self.training_modes = ['contrastive', 'adv_generator', 'adv_discriminator', 'reconstruction']
        optimizers = [optim_contrastive, optim_adv_gen, optim_adv_disc, optim_encoder]
        lr_schedulers = [scheduler_contrastive, scheduler_adv_gen, scheduler_adv_disc]
        return optimizers, lr_schedulers
        
    def forward(self, u):
        w = self.mapping(u)
        x = self.gan.generator.synthesis(w)
        logits = self.classifier(x)
        return logits
    
    def _compute_triplet_loss(self, batch_size):
        u, c = self.mapping.sample_u(batch_size)
        y_pred = self(u)
        
        anchors = torch.zeros(batch_size, device=self.device)
        positives = torch.zeros(batch_size, device=self.device)
        negatives = torch.zeros(batch_size, device=self.device)
        for i in range(batch_size):
            anchors[i] = y_pred[i]
            idx_excluding_i = (torch.arange(batch_size, device=self.device) != i)
            idx_positives = (c == c[i]) & idx_excluding_i
            idx_positives = torch.nonzero(idx_positives).cpu().numpy()
            idx_negatives = (c != c[i]) & idx_excluding_i
            idx_negatives = torch.nonzero(idx_negatives).cpu().numpy()
            positives[i] = y_pred[rng.choice(idx_positives)]
            negatives[i] = y_pred[rng.choice(idx_negatives)]
        
        return self.tripletLoss(anchors, positives, negatives)
    
    def _compute_supConLoss(self, batch_size):
        u, c = self.mapping.sample_u(batch_size)
        y_pred = self(u)
        features = y_pred.reshape(batch_size, 1, -1) # [bsz, n_views, f_dim]  # n_views is the number of crops from each image
        if features.shape[2] > 1: features = F.normalize(features, dim=2) # better be L2 normalized in f_dim dimension (but only when dim>1)
        labels = c # [bsz]
        loss = self.supConLoss(features, labels)
        return loss
    
    def _compute_adversarial_loss(self, batch, mode):
        x_real, y_real = batch
        batch_size = x_real.shape[0]
        u, c_fake = self.mapping.sample_u(batch_size)
        w = self.mapping(u)
        x_fake = self.gan.generator.synthesis(w)
        c_real = F.one_hot(y_real.long(), num_classes=self.gan.c_dim)
        c_fake = F.one_hot(c_fake.long(), num_classes=self.gan.c_dim)
        
        # generator
        if mode == 'adv_generator':
            pred_fake = self.gan.discriminator(x_fake, c_fake)
            loss_gen = F.binary_cross_entropy_with_logits(pred_fake, torch.ones_like(pred_fake, device=self.device)) # fakes should be predicted as true
            return loss_gen
        
        # discriminator
        elif mode == 'adv_discriminator':
            x_fake = x_fake.detach()
            pred_real = self.gan.discriminator(x_real, c_real)
            pred_fake = self.gan.discriminator(x_fake, c_fake)
            loss_disc = 0.5 * (F.binary_cross_entropy_with_logits(pred_fake, torch.zeros_like(pred_fake, device=self.device)) # fakes should be predicted as false
                               + F.binary_cross_entropy_with_logits(pred_real, torch.ones_like(pred_real, device=self.device))) # reals should be predicted as true
            return loss_disc
        
        else:
            raise ValueError('mode should be either "adv_generator" or "adv_discriminator"')
        
    def _compute_img_reconstruction_loss(self, batch):
        x_real, y = batch
        if self.encode_in == 'w':
            w = self.encoder(x_real)
        elif self.encode_in =='u':
            u = self.encoder(x_real)
            w = self.mapping(u)
        else:
            raise ValueError(f'encode_in {self.encode_in} not implemented')
        x_reconstructed = self.gan.generator.synthesis(w)
        loss_reconstruction = F.mse_loss(x_reconstructed, x_real)
        return loss_reconstruction
    
    def _compute_classif_reconstruction_loss(self, batch):
        x_real, y = batch
        logits_real = self.classifier(x_real)
        
        if self.encode_in == 'w':
            w = self.encoder(x_real)
        elif self.encode_in =='u':
            u = self.encoder(x_real)
            w = self.mapping(u)
        else:
            raise ValueError(f'encode_in {self.encode_in} not implemented')
        x_reconstructed = self.gan.generator.synthesis(w)
        logits_rec = self.classifier(x_reconstructed)
        
        loss_classif = F.mse_loss(logits_rec, logits_real)
        # loss_classif = F.kl_div(F.log_softmax(logits_rec, dim=1), F.log_softmax(logits_real, dim=1), reduction='batchmean', log_target=True)
        return loss_classif
        
    def training_step(self, batch, batch_idx, optimizer_idx=0):
        mode = self.training_modes[optimizer_idx]
            
        if mode == 'contrastive':
            batch_size = batch[0].shape[0]
            if self.contrastive_loss == 'triplet':
                loss = self._compute_triplet_loss(batch_size)
            elif self.contrastive_loss == 'SupCon':
                loss = self._compute_supConLoss(batch_size)
            else:
                raise NotImplementedError('wrong loss function')
            self.log(f'train_{self.contrastive_loss}_loss', loss)
            
        elif mode == 'adv_generator':
            loss = self._compute_adversarial_loss(batch, mode)
            self.log(f'train_mapping_adv_loss', loss)
            
        elif mode == 'adv_discriminator':
            loss = self._compute_adversarial_loss(batch, mode)
            self.log(f'train_disc_adv_loss', loss)
            
        elif mode == 'reconstruction':
            loss = self._compute_img_reconstruction_loss(batch)
            self.log(f'train_reconstruction_loss', loss)
            
            if self.use_classif_reconstruction_loss:
                loss_classif = self._compute_classif_reconstruction_loss(batch)
                self.log(f'train_classif_reconstruction_loss', loss_classif)
                loss += loss_classif
            
        else:
            raise NotImplementedError('wrong optimizer_idx')
        return loss
             
    def validation_step(self, batch, batch_idx):
        if self.use_contrastive_loss:
            batch_size = batch[0].shape[0]
            if self.contrastive_loss == 'triplet':
                loss = self._compute_triplet_loss(batch_size)
            elif self.contrastive_loss == 'SupCon':
                loss = self._compute_supConLoss(batch_size)
            else:
                raise NotImplementedError('wrong loss function')
            self.log(f'val_{self.contrastive_loss}_loss', loss)
        if self.use_reconstruction_loss:
            loss = self._compute_img_reconstruction_loss(batch)
            self.log(f'val_reconstruction_loss', loss)
        return None
        
    def on_train_epoch_end(self):
        if (self.current_epoch == 0) or ((self.current_epoch+1)%10 == 0):
            tensorboard_logger = self.logger.experiment
            # generate data by going through pipeline
            n_samples = 2000
            u, c = self.mapping.sample_u(n_samples)
            w = self.mapping(u)
            x = self.gan.generator.synthesis(w)
            logits = self.classifier(x)
            y = torch.sigmoid(logits).round()
            classif_loss = F.binary_cross_entropy_with_logits(logits.squeeze(), c.float(), reduction='none')
            
            # post-process data
            u = u.detach().cpu().numpy()
            c = c.cpu().numpy()
            w = w.detach().cpu().numpy()
            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy().flatten()
            classif_loss = classif_loss.detach().cpu().numpy()
            x_real = self.data[:n_samples][0]
            y_real = self.data[:n_samples][1]
            
            plt.figure()
            plt.title('Histogram of samples from U')
            bins = np.histogram(u[:, -1], bins=60)[1]
            plt.hist(u[y==0, -1], bins=bins, alpha=0.5, label='pred class 0')
            plt.hist(u[y==1, -1], bins=bins, alpha=0.5, label='pred class 1')
            plt.legend()
            tensorboard_logger.add_figure('histogram U', plt.gcf(), self.current_epoch)
            
            plt.figure()
            plt.title('classif loss for different domain values')
            plt.scatter(u[:, -1], classif_loss)
            tensorboard_logger.add_figure('classif loss in U', plt.gcf(), self.current_epoch)
            
            plt.figure()
            plt.title('generated data')
            plt.scatter(x_real[y_real==0, 0], x_real[y_real==0, 1], alpha=0.5, c='C0', label='real data - class 0')
            plt.scatter(x_real[y_real!=0, 0], x_real[y_real!=0, 1], alpha=0.5, c='C1', label='real data - class 1')
            plt.scatter(x[y==0, 0], x[y==0, 1], alpha=0.5, c='C2', label='fake data - pred class 0')
            plt.scatter(x[y!=0, 0], x[y!=0, 1], alpha=0.5, c='C3', label='fake data - pred class 1')
            plt.legend()
            tensorboard_logger.add_figure('generated_data', plt.gcf(), self.current_epoch)
            
            # u_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(u)
            # plt.figure()
            # plt.title('t-SNE in U')
            # plt.scatter(u_embedded[y==0, 0], u_embedded[y==0, 1], alpha=0.5, c='C0', label='class 0')
            # plt.scatter(u_embedded[y!=0, 0], u_embedded[y!=0, 1], alpha=0.5, c='C1', label='class 1')
            # plt.legend()
            # tensorboard_logger.add_figure('t-SNE in U', plt.gcf(), self.current_epoch)
            
            z = torch.randn(x_real.shape[0], self.gan.latent_dim, device=self.device)
            if self.gan.c_dim > 0:
                rnd_label = torch.randint(self.gan.c_dim, size=(z.shape[0],), device=self.device)
                c = F.one_hot(rnd_label, num_classes=self.gan.c_dim)
                z = torch.cat([z, c], dim=1)
            w_natural = self.gan.generator.mapping(z).detach().cpu().numpy()
            w_all = np.concatenate([w_natural, w], axis=0)
            w_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(w_all)
            plt.figure()
            plt.title('t-SNE in W: natural vs. from U')
            plt.scatter(w_embedded[:len(w_natural), 0], w_embedded[:len(w_natural), 1], alpha=0.5, c='C0', label='natural (from GAN noise)')
            plt.scatter(w_embedded[len(w_natural):, 0], w_embedded[len(w_natural):, 1], alpha=0.5, c='C1', label='from U')
            plt.legend()
            tensorboard_logger.add_figure('t-SNE in W: natural vs. from U', plt.gcf(), self.current_epoch)
            
            pca = PCA(n_components=2)
            pca.fit(w_natural)
            w_pca = pca.transform(w_all)
            plt.figure()
            plt.title('PCA in W: natural vs. from U')
            plt.scatter(w_pca[:len(w_natural), 0], w_pca[:len(w_natural), 1], alpha=0.5, c='C0', label='natural (from GAN noise)')
            plt.scatter(w_pca[len(w_natural):, 0], w_pca[len(w_natural):, 1], alpha=0.5, c='C1', label='from U')
            plt.legend()
            tensorboard_logger.add_figure('PCA in W: natural vs. from U', plt.gcf(), self.current_epoch)
            
            w_avg = w_natural.mean(axis=0, keepdims=True)
            dist_natural = np.linalg.norm(w_natural - w_avg, axis=1)
            dist_from_mapping = np.linalg.norm(w - w_avg, axis=1)
            plt.figure()
            plt.title('distances to center in W - natural vs from mapping')
            bins = np.histogram(np.hstack((dist_natural, dist_from_mapping)), bins=60)[1]
            plt.hist(dist_natural, bins=bins, alpha=0.5, label='natural (from GAN noise)')
            plt.hist(dist_from_mapping, bins=bins, alpha=0.5, label='from U')
            plt.legend()
            tensorboard_logger.add_figure('distances to center in W', plt.gcf(), self.current_epoch)
            
            # Risk coverage curves
            u, c = self.mapping.sample_u(20000)
            u_domain = u[:, -1]
            with torch.no_grad(): 
                logits = self(u)
            y = torch.sigmoid(logits).squeeze()
            classif_loss = F.binary_cross_entropy_with_logits(logits.squeeze(), c.float(), reduction='none')
            # baseline: max softmax
            domain_cutoff_baseline = np.linspace(0, 1, 1000)
            coverage_baseline = np.zeros_like(domain_cutoff_baseline)
            risk_baseline = np.zeros_like(domain_cutoff_baseline)
            for i, cut in enumerate(domain_cutoff_baseline):
                idx_domain = (y > cut) | (1-y > cut)
                # idx_domain = torch.maximum(y, 1-y) > cut
                coverage_baseline[i] = idx_domain.float().mean()
                risk_baseline[i] = classif_loss[idx_domain].mean()
            # cut in U_domain
            domain_cutoff = np.linspace(0, 5, 1000)
            coverage = np.zeros_like(domain_cutoff)
            risk = np.zeros_like(domain_cutoff)
            for i, cut in enumerate(domain_cutoff):
                idx_domain = (u_domain < -cut) | (u_domain > cut)
                coverage[i] = idx_domain.float().mean()
                risk[i] = classif_loss[idx_domain].mean()
            plt.figure()
            plt.title('coverage vs. risk (obtained by varying confidence/uncertainty threshold) on FAKE data')
            plt.plot(coverage, risk, label='cut in U')
            plt.plot(coverage_baseline, risk_baseline, label='baseline (max softmax)')
            plt.legend()
            plt.xlabel('coverage')
            plt.ylabel('risk')
            tensorboard_logger.add_figure('coverage vs. risk on FAKE data', plt.gcf(), self.current_epoch)
            
            if self.use_reconstruction_loss:
                # Risk coverage curves for real data
                u = self.encoder(x_real.to(self.device))
                u_domain = u[:, -1]
                with torch.no_grad(): 
                    logits = self.classifier(x_real.to(self.device))
                y = torch.sigmoid(logits).squeeze()
                classif_loss = F.binary_cross_entropy_with_logits(logits.squeeze(), y_real.to(self.device), reduction='none')
                tcp = torch.zeros_like(y)
                tcp[y_real==0] = 1 - y[y_real==0]
                tcp[y_real==1] = y[y_real==1]
                # baseline: max softmax
                domain_cutoff_baseline = np.linspace(0, 1, 1000)
                coverage_baseline = np.zeros_like(domain_cutoff_baseline)
                risk_baseline = np.zeros_like(domain_cutoff_baseline)
                for i, cut in enumerate(domain_cutoff_baseline):
                    idx_domain = (y > cut) | (1-y > cut)
                    # idx_domain = torch.maximum(y, 1-y) > cut
                    coverage_baseline[i] = idx_domain.float().mean()
                    risk_baseline[i] = classif_loss[idx_domain].mean()
                # baseline: TCP
                domain_cutoff_baselineTCP = np.linspace(0, 1, 1000)
                coverage_baselineTCP = np.zeros_like(domain_cutoff_baselineTCP)
                risk_baselineTCP = np.zeros_like(domain_cutoff_baselineTCP)
                for i, cut in enumerate(domain_cutoff_baselineTCP):
                    idx_domain = tcp > cut
                    # idx_domain = torch.maximum(y, 1-y) > cut
                    coverage_baselineTCP[i] = idx_domain.float().mean()
                    risk_baselineTCP[i] = classif_loss[idx_domain].mean()
                # cut in U_domain
                domain_cutoff = np.linspace(0, 5, 1000)
                coverage = np.zeros_like(domain_cutoff)
                risk = np.zeros_like(domain_cutoff)
                for i, cut in enumerate(domain_cutoff):
                    idx_domain = (u_domain < -cut) | (u_domain > cut)
                    coverage[i] = idx_domain.float().mean()
                    risk[i] = classif_loss[idx_domain].mean()
                plt.figure()
                plt.title('coverage vs. risk (obtained by varying confidence/uncertainty threshold) on REAL data')
                plt.plot(coverage, risk, label='cut in U')
                plt.plot(coverage_baseline, risk_baseline, label='baseline (max softmax)')
                plt.plot(coverage_baselineTCP, risk_baselineTCP, label='baseline (TCP)')
                plt.legend()
                plt.xlabel('coverage')
                plt.ylabel('risk')
                tensorboard_logger.add_figure('coverage vs. risk on REAL data', plt.gcf(), self.current_epoch)
                # reconstruction
                with torch.no_grad():
                    if self.encode_in == 'w':
                        w = self.encoder(x_real.to(self.device))
                    elif self.encode_in == 'u':
                        u = self.encoder(x_real.to(self.device))
                        w = self.mapping(u)
                    x_reconstructed = self.gan.generator.synthesis(w).cpu().numpy()
                plt.figure()
                plt.title('reconstructed data')
                plt.scatter(x_real[y_real==0, 0], x_real[y_real==0, 1], alpha=0.5, c='C0', label='real data - class 0')
                plt.scatter(x_real[y_real!=0, 0], x_real[y_real!=0, 1], alpha=0.5, c='C1', label='real data - class 1')
                plt.scatter(x_reconstructed[y_real==0, 0], x_reconstructed[y_real==0, 1], alpha=0.5, c='C2', label='reconstructed data - class 0')
                plt.scatter(x_reconstructed[y_real!=0, 0], x_reconstructed[y_real!=0, 1], alpha=0.5, c='C3', label='reconstructed data - class 1')
                plt.legend()
                tensorboard_logger.add_figure('reconstructed_data', plt.gcf(), self.current_epoch)