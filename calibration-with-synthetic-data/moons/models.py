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


class GAN(pl.LightningModule):
    
    def __init__(self, latent_dim, data_dim, data_n_samples, data_noise, c_dim, hidden_dim,
                 class_conditioning=None, classifier_conditioning=None, classifier=None, wasserstein=False, coeff_MSE=0):
        """GAN model.

        Args:
            latent_dim (int): latent dimension without conditioning
            data_dim (int): data dimension
            data_n_samples (int): dataset size
            data_noise (float): data noise parameter
            c_dim (int): number of classes
            hidden_dim (int): hidden dim in models
            class_conditioning (string/bool): 'one-hot' or 'gaussian' or None (default)
            classifier_conditioning (string/bool): 'TCP', or 'MSP', or 'softmax', or None (default)
            wasserstein (bool): use Wasserstein loss
        """
        
        super().__init__()
        self.save_hyperparameters(ignore=['classifier'])
        self.latent_dim = latent_dim
        self.class_conditioning = class_conditioning
        self.classifier_conditioning = classifier_conditioning
        self.coeff_MSE = coeff_MSE
        
        # wasserstein
        self.wasserstein = wasserstein
        self.lambda_gp = 10 # gradient penalty weight
        self.num_steps = 0
        
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
        elif self.classifier_conditioning == 'softmax':
            assert self.class_conditioning is None, 'conditioning should only be classifier softmax'
            self.condition_dim = c_dim
            self.classifier = classifier
        elif self.classifier_conditioning is None:
            pass
        else:
            raise ValueError('classifier_conditioning must be TCP, MSP, or None')

        self.generator = Generator(latent_dim, self.condition_dim, data_dim, hidden_dim)
        self.discriminator = Discriminator(data_dim, self.condition_dim, hidden_dim)
        self.data = MoonsDataset(n_samples=data_n_samples, noise=data_noise)

            
    def configure_optimizers(self):
        optim_gen = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.99))
        optim_disc = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.99))
        optimizers = [optim_gen, optim_disc]
        if self.class_conditioning == 'gaussian':
            optim_embed = torch.optim.Adam(self.embed.parameters(), lr=0.001, betas=(0, 0.99))
            optimizers.append(optim_embed)
        return optimizers
    
    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)
    
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
        elif self.classifier_conditioning == 'softmax':
            probas = torch.cat((probas_class0, probas_class1), dim=1)
            confidence = probas
        else:
            raise ValueError('Unknown confidence metric')
        return confidence
                
    def training_step(self, batch, batch_idx, optimizer_idx):
        self.num_steps += 1
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
            if self.wasserstein:
                if self.num_steps % 5 == 0:
                    loss_gen = -pred_fake.mean()
                else:
                    return None
            else:
                loss_gen = self.adversarial_loss(pred_fake, torch.ones_like(pred_fake, device=self.device)) # fakes should be predicted as true
            if self.coeff_MSE > 0:
                confidence_reconstructed = self._get_classifier_conditioning(x_fake, y_real)
                loss_mse = self.coeff_MSE * F.mse_loss(confidence, confidence_reconstructed)
                self.log('loss_gen_adv', loss_gen)
                loss_gen += loss_mse
                self.log('loss_gen_mse', loss_mse)
            self.log('loss_gen', loss_gen)
            return loss_gen
        
        # discriminator
        elif optimizer_idx == 1:
            x_fake = self.generator(z, c_fake_gen).detach()
            pred_real = self.discriminator(x_real, c_real)
            pred_fake = self.discriminator(x_fake, c_fake_disc)
            if self.wasserstein:
                epsilon = torch.rand(x_real.shape[0], 1, device=self.device, requires_grad=True)
                gradient_penalty = self._get_gradient_penalty(self.discriminator, x_real, x_fake, c_real, epsilon)
                loss_disc = (pred_fake - pred_real).mean() + self.lambda_gp*gradient_penalty
            else:
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
            x_fake = self.generator.synthesis(w)
            confidence_reconstructed = self._get_classifier_conditioning(x_fake.to(self.device), y_real.to(self.device))
            x_fake = x_fake.cpu().numpy()
        plt.scatter(self.data[:n_samples][0][:, 0], self.data[:n_samples][0][:, 1], alpha=0.5, c=['C0' if y == 0 else 'C1' for y in self.data[:n_samples][1]])
        color = ['C2' if y == 0 else 'C3' for y in rnd_label] if self.class_conditioning is not None else 'k'
        plt.scatter(x_fake[:, 0], x_fake[:, 1], alpha=0.5, c=color)
        tensorboard_logger = self.logger.experiment
        tensorboard_logger.add_figure("generated_data", plt.gcf(), self.current_epoch)
        
        if self.classifier_conditioning == 'softmax':
            confidence_in = torch.max(confidence_in, dim=1, keepdims=True).values
            confidence_out = torch.max(confidence_out, dim=1, keepdims=True).values
            
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
            
            plt.figure()
            plt.plot([0.5, 1], [0.5, 1])
            plt.scatter(confidence_in.cpu(), confidence_reconstructed.cpu(), alpha=0.5)
            plt.xlabel('confidence in')
            plt.ylabel('confidence out')
            tensorboard_logger.add_figure("Confidence renconstruction", plt.gcf(), self.current_epoch)
   
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
            

