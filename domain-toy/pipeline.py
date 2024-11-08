import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.distributions.normal import Normal
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# from umap import UMAP

from generate_data import MoonsDataset
from models import Encoder, GAN

torch.manual_seed(0)
np.random.seed(0)
rng = np.random.default_rng(0)

class Pipeline(pl.LightningModule):
    
    def __init__(self, latent_dim, data_dim, data_n_samples, data_noise, c_dim, hidden_dim,
                 classifier, class_conditioning, classifier_conditioning, use_classif_reconstruction_loss):
        super().__init__()
        self.save_hyperparameters(ignore=['classifier'])
        self.classifier = classifier
        self.gan = GAN(latent_dim, data_dim, data_n_samples, data_noise, c_dim=c_dim, hidden_dim=hidden_dim,
                       class_conditioning=class_conditioning, classifier_conditioning=classifier_conditioning, classifier=classifier)
        self.encoder = Encoder(data_dim, self.gan.latent_dim + self.gan.condition_dim)
        
        self.data = MoonsDataset(n_samples=20000, noise=data_noise)
        
        self.use_classif_reconstruction_loss = use_classif_reconstruction_loss
        
        
    def configure_optimizers(self):
        self.stop_train_GAN_epoch = self.trainer.fit_loop.max_epochs // 2 # stop training GAN after half of the epochs to let the encoder converge
        
        optim_adv_gen = torch.optim.Adam(self.gan.generator.parameters(), lr=0.001, betas=(0, 0.99))
        optim_adv_disc = torch.optim.Adam(self.gan.discriminator.parameters(), lr=0.001, betas=(0, 0.99))
        optimizers = [optim_adv_gen, optim_adv_disc]
        self.training_phases = ['adv_generator', 'adv_discriminator']
        
        if self.gan.class_conditioning == 'gaussian' or self.gan.classifier_conditioning == 'gaussian':
            optim_embed = torch.optim.Adam(self.gan.embed.parameters(), lr=0.002, betas=(0, 0.99))
            optimizers.append(optim_embed)
            self.training_phases.append('conditioning_embedding')
            
        optim_encoder = torch.optim.Adam(self.encoder.parameters(), lr=0.001, weight_decay=1e-4)
        optimizers.append(optim_encoder)
        self.training_phases.append('encoder')
        
        return optimizers
    
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        assert self.training_phases[optimizer_idx] in ['adv_generator', 'adv_discriminator', 'conditioning_embedding', 'encoder'], 'Unknown training phase'
        
        # GET REAL DATA
        x_real, y_real = batch
        
        # GENERATE FAKE DATA
        z = torch.randn(x_real.shape[0], self.gan.latent_dim, device=self.device)
        
        # class condition
        if self.gan.class_conditioning == 'one-hot':
            c_real = F.one_hot(y_real.long(), num_classes=2)
            # rnd_label = torch.randint(2, size=(z.shape[0],), device=self.device)
            # c_fake = F.one_hot(rnd_label, num_classes=2)
            c_fake_gen = c_real
            c_fake_disc = c_fake_gen
        elif self.gan.class_conditioning == 'gaussian': # only for 2 classes            
            c_real = y_real.long().unsqueeze(1) # vector of 0 and 1 (only 2 classes)
            c_fake_gen = self.gan.embed(y_real)
            c_fake_disc = y_real.long().unsqueeze(1) # vector of 0 and 1 (only 2 classes)
        elif self.gan.class_conditioning == 'prediction':
            with torch.no_grad():
                logits = self.classifier(x_real)
            probas = torch.sigmoid(logits).squeeze()
            y_pred = probas.round()
            c_real = F.one_hot(y_pred.long(), num_classes=2) # vector of 0 and 1 (only 2 classes)
            c_fake_gen = c_real
            c_fake_disc = c_fake_gen
        else:
            c_real = None
            c_fake_gen = None
            c_fake_disc = None
        
        # classifier condition
        if self.gan.classifier_conditioning == 'TCP' or self.gan.classifier_conditioning == 'MSP':
            confidence = self.gan._get_classifier_conditioning(x_real, y_real)
            c_real = torch.cat((c_real, confidence), dim=1) if c_real is not None else confidence
            c_fake_gen = torch.cat((c_fake_gen, confidence), dim=1) if c_fake_gen is not None else confidence
            c_fake_disc = torch.cat((c_fake_disc, confidence), dim=1) if c_fake_disc is not None else confidence
        elif self.gan.classifier_conditioning == 'gaussian':
            assert self.gan.class_conditioning is None, 'Cannot use class conditioning if classifier conditioning is gaussian'
            logits = self.classifier(x_real)
            probas = torch.sigmoid(logits).squeeze()
            y_pred = probas.round()
            c_real = y_pred.long().unsqueeze(1) # vector of 0 and 1 (only 2 classes)
            c_fake_gen = self.gan.embed(y_pred)
            c_fake_disc = y_pred.long().unsqueeze(1) # vector of 0 and 1 (only 2 classes)
             
        x_fake = self.gan.generator(z, c_fake_gen)
        
        # TRAIN GAN
        if self.current_epoch < self.stop_train_GAN_epoch:
            # generator
            if self.training_phases[optimizer_idx] == 'adv_generator':
                pred_fake = self.gan.discriminator(x_fake, c_fake_disc)
                loss_gen = self.gan.adversarial_loss(pred_fake, torch.ones_like(pred_fake, device=self.device)) # fakes should be predicted as true
                self.log('loss_train_gen', loss_gen)
                if loss_gen is None: print(f'loss_gen is None at epoch{self.current_epoch}')
                return loss_gen
            
            # discriminator
            elif self.training_phases[optimizer_idx] == 'adv_discriminator':
                x_fake = x_fake.detach()
                pred_real = self.gan.discriminator(x_real, c_real)
                pred_fake = self.gan.discriminator(x_fake, c_fake_disc)
                loss_disc = 0.5 * (self.gan.adversarial_loss(pred_fake, torch.zeros_like(pred_fake, device=self.device)) # fakes should be predicted as false
                                + self.gan.adversarial_loss(pred_real, torch.ones_like(pred_real, device=self.device))) # reals should be predicted as true
                self.log('loss_train_disc', loss_disc)
                if loss_disc is None: print(f'loss_disc is None at epoch{self.current_epoch}')
                return loss_disc
            
            # embedding
            elif self.training_phases[optimizer_idx] == 'conditioning_embedding':
                x_fake = self.gan.generator(z, c_fake_gen)
                pred_fake = self.gan.discriminator(x_fake, c_fake_disc)
                loss_embed = self.gan.adversarial_loss(pred_fake, torch.ones_like(pred_fake, device=self.device)) # fakes should be predicted as true
                self.log('loss_train_embed', loss_embed)
                if loss_embed is None: print(f'loss_embed is None at epoch{self.current_epoch}')
                return loss_embed

        # TRAIN ENCODER
        if self.training_phases[optimizer_idx] == 'encoder':
            # FAKE DATA
            u = self.encoder(x_fake)
            z_encoded = u[:, :self.gan.latent_dim] # noise
            c_encoded = u[:, self.gan.latent_dim:] # conditioning
            if self.gan.class_conditioning == 'prediction' and self.gan.classifier_conditioning is not None:
                with torch.no_grad():
                    logits = self.classifier(x_fake)
                probas = torch.sigmoid(logits).squeeze()
                y_pred = probas.round()
                c_class = F.one_hot(y_pred.long(), num_classes=2) # class condition is given by the classifier
                c_classif = c_encoded[:, -1].unsqueeze(1) # keep only classifier conditioning (always a scalar at the last position)
                c_encoded = torch.cat((c_class, c_classif), dim=1) 
            loss_encoding_noise = F.mse_loss(z_encoded, z)
            loss_encoding_cond = F.mse_loss(c_encoded, c_fake_gen)
            loss_encoding = loss_encoding_noise + loss_encoding_cond
            self.log(f'loss_train_encoding', loss_encoding)
            loss_encoder = loss_encoding
            
            # REAL DATA
            u = self.encoder(x_real)
            z_encoded = u[:, :self.gan.latent_dim] # noise
            c_encoded = u[:, self.gan.latent_dim:] # conditioning
            if self.gan.class_conditioning == 'prediction' and self.gan.classifier_conditioning is not None:
                with torch.no_grad():
                    logits = self.classifier(x_real)
                probas = torch.sigmoid(logits).squeeze()
                y_pred = probas.round()
                c_class = F.one_hot(y_pred.long(), num_classes=2) # class condition is given by the classifier
                c_classif = c_encoded[:, -1].unsqueeze(1) # keep only classifier conditioning (always a scalar at the last position)
                c_encoded = torch.cat((c_class, c_classif), dim=1) 
            x_reconstructed = self.gan.generator(z_encoded, c_encoded)
            loss_reconstruction = F.mse_loss(x_reconstructed, x_real)
            self.log(f'loss_train_reconstruction', loss_reconstruction)
            loss_encoder += loss_reconstruction
            # loss_encoder = loss_reconstruction
            
            if self.use_classif_reconstruction_loss:
                logits_real = self.classifier(x_real)
                logits_rec = self.classifier(x_reconstructed)
                loss_reconstruction_classif = F.mse_loss(logits_rec, logits_real)
                self.log(f'loss_train_classif_reconstruction', loss_reconstruction_classif)
                loss_encoder += loss_reconstruction_classif
                
            if self.gan.classifier_conditioning is not None:
                confidence_real = self.gan._get_classifier_conditioning(x_real, y_real)
                confidence_rec = self.gan._get_classifier_conditioning(x_reconstructed, y_real)
                loss_TCP_encoding = F.mse_loss(confidence_rec, confidence_real)
                self.log(f'loss_train_confidence_encoding', loss_TCP_encoding)
                loss_encoder += loss_TCP_encoding
            
            self.log(f'loss_train_encoder_total', loss_encoder)
            if loss_encoder is None: print(f'loss_encoder is None at epoch{self.current_epoch}')
            return loss_encoder
        
        else:
            return None
 
 
    def on_train_epoch_end(self):            
        tensorboard_logger = self.logger.experiment
        
        # get real samples
        n_samples = 2000
        x_real = self.data[:n_samples][0]
        y_real = self.data[:n_samples][1]
        
        # noise input for generation
        z_fake = torch.randn(n_samples, self.gan.latent_dim, device=self.device)
        
        # class conditioning
        if self.gan.class_conditioning == 'one-hot':
            # y_fake = torch.randint(2, size=(n_samples,), device=self.device)
            y_fake = y_real.long()
            class_embed = F.one_hot(y_fake, num_classes=2).to(self.device)
        elif self.gan.class_conditioning == 'gaussian': # only for 2 classes
            # y_fake = torch.randint(2, size=(n_samples,), device=self.device)
            y_fake = y_real.long()
            with torch.no_grad():
                class_embed = self.gan.embed(y_fake.to(self.device))
            # log gaussian samples and parameters
            self.gan.distrib0_samples.append(class_embed[y_fake==0].squeeze().cpu().numpy())
            self.gan.distrib1_samples.append(class_embed[y_fake==1].squeeze().cpu().numpy())
            self.gan.distrib0_params.append([self.gan.embed.mean0.detach().cpu().numpy(), self.gan.embed.std0.detach().cpu().numpy()])
            self.gan.distrib1_params.append([self.gan.embed.mean1.detach().cpu().numpy(), self.gan.embed.std1.detach().cpu().numpy()])
        elif self.gan.class_conditioning == 'prediction':
            with torch.no_grad():
                logits = self.classifier(x_real.to(self.device))
            probas = torch.sigmoid(logits).squeeze()
            y_pred = probas.round()
            y_fake = y_pred.long()
            class_embed = F.one_hot(y_fake, num_classes=2).to(self.device)
        else:
            y_fake = None
            class_embed = None
        c_fake = class_embed
        
        # classifier conditioning
        if self.gan.classifier_conditioning is not None:
            with torch.no_grad():
                confidence_out = self.gan._get_classifier_conditioning(x_real.to(self.device), y_real.to(self.device))
                confidence_in = confidence_out
            c_fake = torch.cat((c_fake, confidence_in), dim=1) if c_fake is not None else confidence_in
        else:
            confidence_out = None
            confidence_in = None

        # generate
        with torch.no_grad():
            mapping_in = torch.cat((z_fake, c_fake), dim=1) if c_fake is not None else z_fake
            w_fake = self.gan.generator.mapping(mapping_in)
            x_fake = self.gan.generator.synthesis(w_fake).cpu().numpy()
            
        # reconstruction
        with torch.no_grad():
            u = self.encoder(x_real.to(self.device))
            z_real = u[:, :self.gan.latent_dim] # noise
            c_real = u[:, self.gan.latent_dim:] # conditioning
            if self.gan.class_conditioning == 'prediction' and self.gan.classifier_conditioning is not None:
                with torch.no_grad():
                    logits = self.classifier(x_real.to(self.device))
                probas = torch.sigmoid(logits).squeeze()
                y_pred = probas.round()
                c_class = F.one_hot(y_pred.long(), num_classes=2) # class condition is given by the classifier
                c_classif = c_real[:, -1].unsqueeze(1) # keep only classifier conditioning (always a scalar at the last position)
                c_real = torch.cat((c_class, c_classif), dim=1) 
            x_reconstructed = self.gan.generator(z_real, c_real).cpu()
            
        # domain
        if self.gan.class_conditioning == 'gaussian' and self.gan.classifier_conditioning is None:
            u_domain_fake = class_embed.cpu()
            u_domain_real = c_real[:, 0].cpu()
        elif self.gan.classifier_conditioning is not None:
            u_domain_fake = confidence_in.cpu()
            u_domain_real = c_real[:, -1].cpu()       
        else:
            u_domain_fake = None
            u_domain_real = None

        # log figures
        if self.current_epoch == 0:
            self._log_classifier_analysis(tensorboard_logger, x_real, y_real)
        if (self.current_epoch == 0) or (self.current_epoch == self.stop_train_GAN_epoch // 2) or (self.current_epoch == self.stop_train_GAN_epoch):
            self._log_latent_analysis(tensorboard_logger, n_samples, x_real, y_real, x_fake, class_embed, z_fake, w_fake, confidence_in, confidence_out)
        if (self.current_epoch == 0) or ((self.current_epoch+1)%10 == 0):
            self._log_gan_analysis(tensorboard_logger, x_real, y_real, x_fake, y_fake, c_fake)
            self._log_encoder_analysis(tensorboard_logger, x_real, y_real, x_reconstructed, u_domain_fake, u_domain_real)
            if u_domain_fake is not None:
                self._log_domain_analysis(tensorboard_logger, n_samples, x_fake, y_fake, u_domain_fake, x_real, y_real, u_domain_real)

    def _log_classifier_analysis(self, tensorboard_logger, x_real, y_real):
        
        # SHOW DESCISION BOUNDARY
        x = np.linspace(-2, 3, 100)
        y = np.linspace(-2, 2, 100)

        grid_data = np.zeros((len(x)*len(y), 2))
        i = 0
        for x_ in x:
            for y_ in y:
                grid_data[i] = [x_, y_]
                i += 1
        grid_data = torch.from_numpy(grid_data).float()

        with torch.no_grad():
            y = self.classifier(grid_data.to(self.device))
        class_pred = torch.sigmoid(y).round().cpu().flatten()#.numpy()

        # SHOW CLASSIF LOSS
        with torch.no_grad():
            logits = self.classifier(x_real.to(self.device)).squeeze().cpu()
            classif_loss = F.binary_cross_entropy_with_logits(logits, y_real, reduction='none')

        fig, ax = plt.subplots()
        ax.set_title('classifier decision boundary')
        ax.scatter(grid_data[class_pred==0, 0], grid_data[class_pred==0, 1], alpha=1, c='C0', label='predicted class 0')
        ax.scatter(grid_data[class_pred!=0, 0], grid_data[class_pred!=0, 1], alpha=1, c='C1', label='predicted class 1')
        ax.scatter(x_real[y_real==0, 0], x_real[y_real==0, 1], alpha=0.3, c=classif_loss[y_real==0], cmap='Reds', marker='o', label='real data - class 0')
        im = ax.scatter(x_real[y_real==1, 0], x_real[y_real==1, 1], alpha=0.3, c=classif_loss[y_real==1], cmap='Reds', marker='+', label='real data - class 1')
        leg = ax.legend(frameon=True)
        for lh in leg.legendHandles: 
            lh.set_alpha(1)
        cbar = fig.colorbar(im, ax=ax, label='classifier loss')
        cbar.solids.set(alpha=1)
        tensorboard_logger.add_figure("classifier_analysis", fig, self.current_epoch)
        
        
    def _log_gan_analysis(self, tensorboard_logger, x_real, y_real, x_fake, y_fake, c):
        
        # generated data
        plt.figure()
        plt.scatter(x_real[:, 0], x_real[:, 1], alpha=0.5, c=['C0' if y == 0 else 'C1' for y in y_real])
        color = ['C2' if y == 0 else 'C3' for y in y_fake] if self.gan.class_conditioning is not None else 'k'
        plt.scatter(x_fake[:, 0], x_fake[:, 1], alpha=0.5, c=color)
        tensorboard_logger.add_figure("generated_data", plt.gcf(), self.current_epoch)

        if self.gan.class_conditioning == 'gaussian': # only for 2 classes
            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12,6), constrained_layout=True)
            ax0.set_title('class 0')
            ax1.set_title('class 1')
            scat0 = ax0.scatter(x_fake[y_fake==0, 0], x_fake[y_fake==0, 1], alpha=0.5, c=c[y_fake==0, 0].cpu().numpy(), cmap='viridis')
            scat1 = ax1.scatter(x_fake[y_fake==1, 0], x_fake[y_fake==1, 1], alpha=0.5, c=c[y_fake==1, 0].cpu().numpy(), cmap='viridis')
            cb = fig.colorbar(scat0, ax=ax0)
            cb.set_alpha(1)
            cb.draw_all()
            cb = fig.colorbar(scat1, ax=ax1)
            cb.set_alpha(1)
            cb.draw_all()
            tensorboard_logger.add_figure("generated_data_gauss", plt.gcf(), self.current_epoch)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            for i in range(self.current_epoch+1):
                y0 = self.gan.distrib0_samples[i]
                x0 = np.array([i for _ in range(len(y0))])
                ax1.scatter(x0, y0, alpha=0.5, c=f'C0', label=f'class 0')
                y1 = self.gan.distrib1_samples[i]
                x1 = np.array([i for _ in range(len(y1))])
                ax1.scatter(x1, y1, alpha=0.5, c=f'C1', label=f'class 1')
            ax1.set_xlabel('epoch')
            ax1.set_ylabel('gauss mixt samples')
            
            x = np.array([i for i in range(self.current_epoch+1)])
            means0 = np.array(self.gan.distrib0_params)[:, 0]
            stds0 = np.array(self.gan.distrib0_params)[:, 1]
            ax2.plot(x, means0, c='C0', label='class 0')
            ax2.fill_between(x, means0-stds0, means0+stds0, color='C0', alpha=0.3)
            means1 = np.array(self.gan.distrib1_params)[:, 0]
            stds1 = np.array(self.gan.distrib1_params)[:, 1]
            ax2.plot(x, means1, c='C1', label='class 1')
            ax2.fill_between(x, means1-stds1, means1+stds1, color='C1', alpha=0.3)
            ax2.legend()
            ax2.set_xlabel('epoch')
            ax2.set_ylabel('gauss mixt params')
            tensorboard_logger.add_figure("distributions_evolution", plt.gcf(), self.current_epoch)

    def _log_latent_analysis(self, tensorboard_logger, n_samples, x_real, y_real, x_fake, class_embed, z, w, confidence_in, confidence_out):
        
        # analysis of W
        import time
        start = time.time()
        w_pca = PCA(n_components=2).fit_transform(w.cpu().numpy())
        w_tsne = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(w.cpu().numpy())
        # w_umap = UMAP().fit_transform(w.cpu().numpy())
        fig, axs = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
        for algo, ax_, w_embedded in zip(['PCA', 't-SNE'], axs, [w_pca, w_tsne]):
            if self.gan.class_conditioning is not None:
                ax_[0].set_title(f'{algo} with input class as color')
                ax_[0].scatter(w_embedded[y_real==0, 0], w_embedded[y_real==0, 1], c='C0', alpha=0.5, label='class 0')
                ax_[0].scatter(w_embedded[y_real==1, 0], w_embedded[y_real==1, 1], c='C1', alpha=0.5, label='class 1')
                ax_[0].legend()
            else:
                ax_[0].set_title(f'{algo} with input class as color')
                ax_[0].scatter(w_embedded[:, 0], w_embedded[:, 1], c='k', alpha=0.5, label='no input class given')
                ax_[0].legend()
            if self.gan.classifier_conditioning is not None:
                ax_[1].set_title(f'{algo} with Confidence as color')
                scat = ax_[1].scatter(w_embedded[:, 0], w_embedded[:, 1], c=confidence_in.cpu().numpy(), cmap='viridis', alpha=0.5)
                cb = fig.colorbar(scat, ax=ax_[1])
                cb.set_alpha(1)
                cb.draw_all()
        tensorboard_logger.add_figure('W', plt.gcf(), self.current_epoch)    
                
        if self.gan.classifier_conditioning is not None:
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
            mapping_in = torch.cat((z, confidence_one), dim=1) if self.gan.class_conditioning is None else torch.cat((z, class_embed, confidence_one), dim=1)
            with torch.no_grad():
                w = self.gan.generator.mapping(mapping_in)
                w_center_confidence_one = w.mean(0, keepdim=True)
            confidence_rnd = torch.rand(size=(n_samples, 1)).to(self.device)
            mapping_in = torch.cat((z, confidence_rnd), dim=1) if self.gan.class_conditioning is None else torch.cat((z, class_embed, confidence_rnd), dim=1)
            with torch.no_grad():
                w = self.gan.generator.mapping(mapping_in)
            distances = torch.cdist(w, w_center_confidence_one).cpu().numpy()
            plt.figure()
            plt.title('distances from W center of confidence=1')
            plt.scatter(distances, confidence_rnd.cpu().numpy(), alpha=0.5)
            plt.xlabel('distance from W center of confidence=1')
            plt.ylabel('confidence value')
            tensorboard_logger.add_figure('distances from W center of confidence=1', plt.gcf(), self.current_epoch)
            
            
    def _log_encoder_analysis(self, tensorboard_logger, x_real, y_real, x_reconstructed, u_domain_fake, u_domain_real):
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        if u_domain_fake is not None:
            bins = np.histogram(torch.cat((u_domain_fake.squeeze(), u_domain_real.squeeze())), bins=50)[1]
            ax1.set_title('histogram in u_domain')
            ax1.hist(u_domain_fake.squeeze(), bins=bins, alpha=0.5, label='from fake data following distribution')
            ax1.hist(u_domain_real.squeeze(), bins=bins, alpha=0.5, label='from real data encoded')
            ax1.legend()
        ax2.set_title('reconstructed data')
        ax2.scatter(x_real[y_real==0, 0], x_real[y_real==0, 1], alpha=0.5, c='C0', label='real data - class 0')
        ax2.scatter(x_real[y_real==1, 0], x_real[y_real==1, 1], alpha=0.5, c='C1', label='real data - class 1')
        ax2.scatter(x_reconstructed[y_real==0, 0], x_reconstructed[y_real==0, 1], alpha=0.5, c='C2', label='reconstructed data - class 0')
        ax2.scatter(x_reconstructed[y_real==1, 0], x_reconstructed[y_real==1, 1], alpha=0.5, c='C3', label='reconstructed data - class 1')
        ax2.legend()
        tensorboard_logger.add_figure('reconstructed_data', fig, self.current_epoch)
    
    
    def _log_domain_analysis(self, tensorboard_logger, n_samples, x_fake, y_fake, u_domain_fake, x_real, y_real, u_domain_real):
        
        if self.gan.class_conditioning is not None:
            # Risk coverage curves for FAKE data
            x_fake = torch.tensor(x_fake, device=self.device)
            y_fake = y_fake.to(self.device)
            x_real = x_real.to(self.device)
            y_real = y_real.to(self.device)
            with torch.no_grad():
                logits = self.classifier(x_fake)
            probas = torch.sigmoid(logits).squeeze()
            classif_loss = F.binary_cross_entropy_with_logits(logits.squeeze(), y_fake.float(), reduction='none')
            classif_correct = (probas.round() == y_fake)

            # baseline: random selection
            domain_cutoff_random = np.linspace(0, 1, 100)
            coverage_random = np.zeros_like(domain_cutoff_random)
            risk_random = np.zeros_like(domain_cutoff_random)
            acc_random = np.zeros_like(domain_cutoff_random)
            for i, cut in enumerate(domain_cutoff_random):
                nb_samples = int((1-cut) * n_samples) # 1-cut to be coherent with other indices below (low value -> high coverage)
                idx_domain = rng.choice(np.arange(n_samples), size=nb_samples, replace=False)
                coverage_random[i] = x_fake[idx_domain].shape[0] / n_samples
                risk_random[i] = classif_loss[idx_domain].mean()
                acc_random[i] = classif_correct[idx_domain].float().mean()

            # baseline: max softmax
            domain_cutoff_baseline = np.linspace(0, 1, 1000)
            coverage_baseline = np.zeros_like(domain_cutoff_baseline)
            risk_baseline = np.zeros_like(domain_cutoff_baseline)
            acc_baseline = np.zeros_like(domain_cutoff_baseline)
            for i, cut in enumerate(domain_cutoff_baseline):
                idx_domain = (probas > cut) | (1-probas > cut)
                coverage_baseline[i] = idx_domain.float().mean()
                risk_baseline[i] = classif_loss[idx_domain].mean()
                acc_baseline[i] = classif_correct[idx_domain].float().mean()

            # cut max proba computed in U
            domain_cutoff = np.linspace(0, 1, 1000)
            coverage = np.zeros_like(domain_cutoff)
            risk = np.zeros_like(domain_cutoff)
            acc = np.zeros_like(domain_cutoff)
            for i, cut in enumerate(domain_cutoff):
                idx_domain = self.selection_function(u_domain_fake, cut)
                coverage[i] = idx_domain.float().mean()
                risk[i] = classif_loss[idx_domain].mean()
                acc[i] = classif_correct[idx_domain].float().mean()

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            ax1.set_title('coverage vs. risk\n(obtained by varying confidence/uncertainty threshold)\n(using pseudo-labels)')
            ax1.plot(coverage, risk, label='cut in U')
            ax1.plot(coverage_baseline, risk_baseline, label='baseline (max softmax)')
            ax1.plot(coverage_random, risk_random, label='baseline (random)')
            ax1.legend()
            ax1.set_xlabel('coverage')
            ax1.set_ylabel('risk')

            ax2.set_title('coverage vs. accuracy\n(obtained by varying confidence/uncertainty threshold)\n(using pseudo-labels)')
            ax2.plot(coverage, acc, label='cut in U')
            ax2.plot(coverage_baseline, acc_baseline, label='baseline (max softmax)')
            ax2.plot(coverage_random, acc_random, label='baseline (random)')
            ax2.legend()
            ax2.set_xlabel('coverage')
            ax2.set_ylabel('accuracy')

            ax3.set_title('coverage vs threshold value')
            ax3.plot((domain_cutoff-domain_cutoff.min())/(domain_cutoff.max()-domain_cutoff.min()), coverage, label='cut in U')
            ax3.plot((domain_cutoff_baseline-domain_cutoff_baseline.min())/(domain_cutoff_baseline.max()-domain_cutoff_baseline.min()), coverage_baseline, label='baseline (max softmax)')
            ax3.plot((domain_cutoff_random-domain_cutoff_random.min())/(domain_cutoff_random.max()-domain_cutoff_random.min()), coverage_random, label='baseline (random)')
            ax3.legend()
            ax3.set_xlabel('normalized threshold value')
            ax3.set_ylabel('coverage')

            ax4.set_title('risk vs threshold value')
            ax4.plot((domain_cutoff-domain_cutoff.min())/(domain_cutoff.max()-domain_cutoff.min()), risk, label='cut in U')
            ax4.plot((domain_cutoff_baseline-domain_cutoff_baseline.min())/(domain_cutoff_baseline.max()-domain_cutoff_baseline.min()), risk_baseline, label='baseline (max softmax)')
            ax4.plot((domain_cutoff_random-domain_cutoff_random.min())/(domain_cutoff_random.max()-domain_cutoff_random.min()), risk_random, label='baseline (random)')
            ax4.legend()
            ax4.set_xlabel('normalized threshold value')
            ax4.set_ylabel('risk')
            tensorboard_logger.add_figure("domain_fake_data", fig, self.current_epoch)
        
        
        # Risk coverage curves for REAL test data
        with torch.no_grad(): 
            logits = self.classifier(x_real.to(self.device)).cpu()
        probas = torch.sigmoid(logits).squeeze()
        classif_loss = F.binary_cross_entropy_with_logits(logits.squeeze(), y_real.float().cpu(), reduction='none')
        classif_correct = (probas.round() == y_real.cpu())  
        tcp = torch.zeros_like(probas)
        tcp[y_real==0] = 1 - probas[y_real==0]
        tcp[y_real==1] = probas[y_real==1]

        # baseline: random selection
        domain_cutoff_random = np.linspace(0, 1, 100)
        coverage_random = np.zeros_like(domain_cutoff_random)
        risk_random = np.zeros_like(domain_cutoff_random)
        acc_random = np.zeros_like(domain_cutoff_random)
        for i, cut in enumerate(domain_cutoff_random):
            nb_samples = int((1-cut) * n_samples) # 1-cut to be coherent with other indices below (low value -> high coverage)
            idx_domain = rng.choice(np.arange(n_samples), size=nb_samples, replace=False)
            coverage_random[i] = x_real[idx_domain].shape[0] / n_samples
            risk_random[i] = classif_loss[idx_domain].mean()
            acc_random[i] = classif_correct[idx_domain].float().mean()
            acc_random[i] = classif_correct[idx_domain].float().mean()
            
        # baseline: max softmax
        domain_cutoff_baseline = np.linspace(0, 1, 1000)
        coverage_baseline = np.zeros_like(domain_cutoff_baseline)
        risk_baseline = np.zeros_like(domain_cutoff_baseline)
        acc_baseline = np.zeros_like(domain_cutoff_baseline)
        for i, cut in enumerate(domain_cutoff_baseline):
            idx_domain = (probas > cut) | (1-probas > cut)
            coverage_baseline[i] = idx_domain.float().mean()
            risk_baseline[i] = classif_loss[idx_domain].mean()
            acc_baseline[i] = classif_correct[idx_domain].float().mean()
            
        # baseline: TCP
        domain_cutoff_baselineTCP = np.linspace(0, 1, 1000)
        coverage_baselineTCP = np.zeros_like(domain_cutoff_baselineTCP)
        risk_baselineTCP = np.zeros_like(domain_cutoff_baselineTCP)
        acc_baselineTCP = np.zeros_like(domain_cutoff_baselineTCP)
        for i, cut in enumerate(domain_cutoff_baselineTCP):
            idx_domain = tcp > cut
            coverage_baselineTCP[i] = idx_domain.float().mean()
            risk_baselineTCP[i] = classif_loss[idx_domain].mean()
            acc_baselineTCP[i] = classif_correct[idx_domain].float().mean()
            
        # cut max proba computed in U
        domain_cutoff = np.linspace(0, 1, 1000)
        coverage = np.zeros_like(domain_cutoff)
        risk = np.zeros_like(domain_cutoff)
        acc = np.zeros_like(domain_cutoff)
        for i, cut in enumerate(domain_cutoff):
            idx_domain = self.selection_function(u_domain_real, cut)
            coverage[i] = idx_domain.float().mean()
            risk[i] = classif_loss[idx_domain].mean()
            acc[i] = classif_correct[idx_domain].float().mean()

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        ax1.set_title('coverage vs. risk\n(obtained by varying confidence/uncertainty threshold)')
        ax1.plot(coverage, risk, label='cut in U')
        ax1.plot(coverage_baseline, risk_baseline, label='baseline (max softmax)')
        ax1.plot(coverage_random, risk_random, label='baseline (random)')
        ax1.plot(coverage_baselineTCP, risk_baselineTCP, label='baseline (TCP oracle)')
        ax1.legend()
        ax1.set_xlabel('coverage')
        ax1.set_ylabel('risk')

        ax2.set_title('coverage vs. accuracy\n(obtained by varying confidence/uncertainty threshold)\n(using pseudo-labels)')
        ax2.plot(coverage, acc, label='cut in U')
        ax2.plot(coverage_baseline, acc_baseline, label='baseline (max softmax)')
        ax2.plot(coverage_random, acc_random, label='baseline (random)')
        ax2.plot(coverage_baselineTCP, acc_baselineTCP, label='baseline (TCP oracle)')
        ax2.legend()
        ax2.set_xlabel('coverage')
        ax2.set_ylabel('accuracy')

        ax3.set_title('coverage vs threshold value')
        ax3.plot((domain_cutoff-domain_cutoff.min())/(domain_cutoff.max()-domain_cutoff.min()), coverage, label='cut in U')
        ax3.plot((domain_cutoff_baseline-domain_cutoff_baseline.min())/(domain_cutoff_baseline.max()-domain_cutoff_baseline.min()), coverage_baseline, label='baseline (max softmax)')
        ax3.plot((domain_cutoff_random-domain_cutoff_random.min())/(domain_cutoff_random.max()-domain_cutoff_random.min()), coverage_random, label='baseline (random)')
        ax3.plot((domain_cutoff_baselineTCP-domain_cutoff_baselineTCP.min())/(domain_cutoff_baselineTCP.max()-domain_cutoff_baselineTCP.min()), coverage_baselineTCP, label='baseline (TCP oracle)')
        ax3.legend()
        ax3.set_xlabel('normalized threshold value')
        ax3.set_ylabel('coverage')

        ax4.set_title('risk vs threshold value')
        ax4.plot((domain_cutoff-domain_cutoff.min())/(domain_cutoff.max()-domain_cutoff.min()), risk, label='cut in U')
        ax4.plot((domain_cutoff_baseline-domain_cutoff_baseline.min())/(domain_cutoff_baseline.max()-domain_cutoff_baseline.min()), risk_baseline, label='baseline (max softmax)')
        ax4.plot((domain_cutoff_random-domain_cutoff_random.min())/(domain_cutoff_random.max()-domain_cutoff_random.min()), risk_random, label='baseline (random)')
        ax4.plot((domain_cutoff_baselineTCP-domain_cutoff_baselineTCP.min())/(domain_cutoff_baselineTCP.max()-domain_cutoff_baselineTCP.min()), risk_baselineTCP, label='baseline (TCP oracle)')
        ax4.legend()
        ax4.set_xlabel('normalized threshold value')
        ax4.set_ylabel('risk')
        tensorboard_logger.add_figure("domain_real_data", fig, self.current_epoch)
        
        # Illustrate domain
        x_real = x_real.cpu().numpy()
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        cut = 0.8
        idx_in_domain = self.selection_function(u_domain_real, cut).cpu()
        idx_out_domain = idx_in_domain.logical_not()
        coverage = idx_in_domain.float().mean()
        risk = classif_loss[idx_in_domain].mean()
        acc = classif_correct[idx_in_domain].float().mean()
        ax1.set_title(f'threshold = {cut}\ncoverage={coverage:.2f}, risk={risk:.2f}, acc={acc:.2f}')
        ax1.scatter(x_real[idx_out_domain, 0], x_real[idx_out_domain, 1], alpha=0.1, label='out domain', c='r')
        ax1.scatter(x_real[idx_in_domain, 0], x_real[idx_in_domain, 1], alpha=0.1, label='in domain', c='g')
        ax1.legend()

        cut = 0.9
        idx_in_domain = self.selection_function(u_domain_real, cut).cpu()
        idx_out_domain = idx_in_domain.logical_not()
        coverage = idx_in_domain.float().mean()
        risk = classif_loss[idx_in_domain].mean()
        acc = classif_correct[idx_in_domain].float().mean()
        ax2.set_title(f'threshold = {cut}\ncoverage={coverage:.2f}, risk={risk:.2f}, acc={acc:.2f}')
        ax2.scatter(x_real[idx_out_domain, 0], x_real[idx_out_domain, 1], alpha=0.1, label='out domain', c='r')
        ax2.scatter(x_real[idx_in_domain, 0], x_real[idx_in_domain, 1], alpha=0.1, label='in domain', c='g')
        ax2.legend()

        cut = 0.95
        idx_in_domain = self.selection_function(u_domain_real, cut).cpu()
        idx_out_domain = idx_in_domain.logical_not()
        coverage = idx_in_domain.float().mean()
        risk = classif_loss[idx_in_domain].mean()
        acc = classif_correct[idx_in_domain].float().mean()
        ax3.set_title(f'threshold = {cut}\ncoverage={coverage:.2f}, risk={risk:.2f}, acc={acc:.2f}')
        ax3.scatter(x_real[idx_out_domain, 0], x_real[idx_out_domain, 1], alpha=0.1, label='out domain', c='r')
        ax3.scatter(x_real[idx_in_domain, 0], x_real[idx_in_domain, 1], alpha=0.1, label='in domain', c='g')
        ax3.legend()
        tensorboard_logger.add_figure("domain_real_data_illustration", fig, self.current_epoch)
        
    def selection_function(self, x, cut):
        """compute proba from value in U

        Args:
            x (float): value in U
            cut (float): threshold

        Returns:
            float: corresponding proba
        """
        if self.gan.class_conditioning == 'gaussian' and self.gan.classifier_conditioning is None:
            x = x.squeeze()
            p_x_y0 = Normal(self.gan.embed.mean0, self.gan.embed.std0).log_prob(x).exp().detach()
            p_x_y1 = Normal(self.gan.embed.mean1, self.gan.embed.std1).log_prob(x).exp().detach()
            p_y0 = 0.5
            p_y1 = 0.5
            p_y0_x = p_x_y0 * p_y0 / (p_x_y0 * p_y0 + p_x_y1 * p_y1)
            p_y1_x = p_x_y1 * p_y1 / (p_x_y0 * p_y0 + p_x_y1 * p_y1)
            max_p_y_x = torch.maximum(p_y0_x, p_y1_x)
            in_domain = max_p_y_x > cut
        if self.gan.classifier_conditioning is not None:
            in_domain = x.squeeze() > cut
        else:
            raise NotImplementedError('not implemented for this conditioning')
        return in_domain