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
from models import Encoder, GeneratorNew, Discriminator

torch.manual_seed(0)
np.random.seed(0)
rng = np.random.default_rng(0)

class PipelineNew(pl.LightningModule):
    
    def __init__(self, latent_dim, data_dim, data_n_samples, data_noise, c_dim, hidden_dim,
                 classifier, class_conditioning, classifier_conditioning):
        super().__init__()
        self.save_hyperparameters(ignore=['classifier'])
        
        self.automatic_optimization = False
        
        if (class_conditioning in ['label', 'prediction', None] and classifier_conditioning == 'TCP'):
            condition_dim = c_dim + 1
        else:
            raise NotImplementedError('Only label or prediction class conditioning and TCP classifier conditioning are implemented')
        
        self.class_conditioning = class_conditioning
        self.classifier_conditioning = classifier_conditioning
        self.latent_dim = latent_dim
        
        self.classifier = classifier
        self.generator = GeneratorNew(latent_dim, condition_dim, data_dim, hidden_dim)
        self.discriminator = Discriminator(data_dim, condition_dim, hidden_dim)
        self.data_encoder = Encoder(data_dim, latent_dim)
        self.confidence_estimator = Encoder(data_dim, 1)
        self.class_embedder = lambda y: F.one_hot(y, num_classes=2)
        
        self.data = MoonsDataset(n_samples=20000, noise=data_noise)        
        
    def configure_optimizers(self):
        optim_adv_gen = torch.optim.Adam(self.generator.parameters(), lr=0.001, betas=(0, 0.99))
        optim_adv_disc = torch.optim.Adam(self.discriminator.parameters(), lr=0.001, betas=(0, 0.99))
        optimizers = [optim_adv_gen, optim_adv_disc]
        
        params = list(self.data_encoder.parameters()) + list(self.confidence_estimator.parameters()) + list(self.generator.parameters())
        optim_reco = torch.optim.Adam(params, lr=0.001, weight_decay=1e-4)
        optimizers.append(optim_reco)
        
        optim_confidence_estimator = torch.optim.Adam(self.confidence_estimator.parameters(), lr=0.001, weight_decay=1e-4)
        optimizers.append(optim_confidence_estimator)
        
        params = list(self.data_encoder.parameters()) + list(self.generator.parameters())
        optim_enco = torch.optim.Adam(self.data_encoder.parameters(), lr=0.001, weight_decay=1e-4)
        optimizers.append(optim_enco)
        
        return optimizers
    
    
    def training_step(self, batch, batch_idx):
        x_real, y_real = batch
        
        optim_adv_gen, optim_adv_disc, optim_reco, optim_confidence_estimator, optim_enco = self.optimizers()
        optim_adv_gen.zero_grad()
        optim_adv_disc.zero_grad()
        optim_reco.zero_grad()
        optim_confidence_estimator.zero_grad()
        optim_enco.zero_grad()
        
        # RECONSTRUCT REAL DATA
        
        # encode data
        w_encoded = self.data_encoder(x_real)
        
        # class condition
        if self.class_conditioning == 'label':
            class_conditioning = self.class_embedder(y_real.long())
        elif self.class_conditioning == 'prediction':
            with torch.no_grad():
                logits = self.classifier(x_real)
            probas = torch.sigmoid(logits).squeeze()
            y_pred = probas.round()
            class_conditioning = self.class_embedder(y_pred.long())
        else:
            raise NotImplementedError('Only label or prediction class conditioning are implemented')
        
        # classifier condition
        assert self.classifier_conditioning == 'TCP'
        confidence = self._get_classifier_conditioning(x_real, y_real)
        conditioning = torch.cat((class_conditioning, confidence), dim=1)
        
        # reconstruct
        synthesis_input = torch.cat((w_encoded, conditioning), dim=1)
        x_reconstructed = self.generator.synthesis(synthesis_input)
        
        # data reconstruction loss
        loss_reconstruction_data = F.mse_loss(x_reconstructed, x_real)
        self.log(f'loss_train_reconstruction_data', loss_reconstruction_data)
        loss_reconstruction = loss_reconstruction_data
        
        # classif reconstruction loss
        logits_real = self.classifier(x_real)
        logits_rec = self.classifier(x_reconstructed)
        loss_reconstruction_classif = F.mse_loss(logits_rec, logits_real)
        self.log(f'loss_train_reconstruction_classif', loss_reconstruction_classif)
        loss_reconstruction += loss_reconstruction_classif
        self.log(f'loss_train_reconstruction_total', loss_reconstruction)
        self.manual_backward(loss_reconstruction, retain_graph=True)
        
        # confidence estimator loss
        confidence_pred = torch.sigmoid(self.confidence_estimator(x_real))
        loss_confid = F.mse_loss(confidence_pred, confidence)
        self.log(f'loss_train_confid', loss_confid)
        self.manual_backward(loss_confid, retain_graph=True)
        
        # GENERATE FAKE DATA
        
        z = torch.randn(x_real.shape[0], self.latent_dim, device=self.device)
        conditioning_fake = conditioning # conditioning for fake data is the same as for real data for simplicity
        w_fake = self.generator.mapping(z)
        w_c_fake = torch.cat((w_fake, conditioning_fake), dim=1)
        x_fake = self.generator.synthesis(w_c_fake)
        
        # generator adversarial loss
        pred_fake = self.discriminator(x_fake, conditioning_fake)
        loss_gen = self.adversarial_loss(pred_fake, torch.ones_like(pred_fake, device=self.device)) # fakes should be predicted as true
        self.log('loss_train_gen', loss_gen)
        self.manual_backward(loss_gen, retain_graph=True)

        # discriminator adversarial loss
        x_fake = x_fake.detach()
        pred_real = self.discriminator(x_real, conditioning)
        pred_fake = self.discriminator(x_fake, conditioning_fake)
        loss_disc = 0.5 * (self.adversarial_loss(pred_fake, torch.zeros_like(pred_fake, device=self.device)) # fakes should be predicted as false
                        + self.adversarial_loss(pred_real, torch.ones_like(pred_real, device=self.device))) # reals should be predicted as true
        self.log('loss_train_disc', loss_disc)
        self.manual_backward(loss_disc, retain_graph=True)
        
        # encode fake data loss
        # w_fake = w_fake.detach()
        x_fake = x_fake.detach()
        w_fake_encoded = self.data_encoder(x_fake)
        loss_encoding = F.mse_loss(w_fake_encoded, w_fake)
        self.log(f'loss_train_encoding', loss_encoding)
        self.manual_backward(loss_encoding, retain_graph=True)
                
        optim_reco.step()
        optim_confidence_estimator.step()
        optim_adv_gen.step()
        optim_adv_disc.step()
        optim_enco.step()
 
    def on_train_epoch_end(self):            
        tensorboard_logger = self.logger.experiment
        
        # get real samples
        n_samples = 2000
        x_real = self.data[:n_samples][0]
        y_real = self.data[:n_samples][1]
        
        # encode data
        with torch.no_grad():
            w_encoded = self.data_encoder(x_real.to(self.device))
        
        # class condition
        if self.class_conditioning == 'label':
            class_conditioning = self.class_embedder(y_real.long().to(self.device))
        elif self.class_conditioning == 'prediction':
            with torch.no_grad():
                logits = self.classifier(x_real.to(self.device))
            probas = torch.sigmoid(logits).squeeze()
            y_pred = probas.round()
            class_conditioning = self.class_embedder(y_pred.long())
        else:
            raise NotImplementedError('Only label or prediction class conditioning are implemented')
        
        # classifier condition
        assert self.classifier_conditioning == 'TCP'
        with torch.no_grad():
            confidence_pred = torch.sigmoid(self.confidence_estimator(x_real.to(self.device)))
        conditioning = torch.cat((class_conditioning, confidence_pred), dim=1)
        
        # reconstruct
        with torch.no_grad():
            synthesis_input = torch.cat((w_encoded, conditioning), dim=1)
            x_reconstructed = self.generator.synthesis(synthesis_input).cpu().numpy()
            
        # adapt variable names
        x_fake = x_reconstructed
        y_fake = y_real
        c_fake = conditioning
        confidence_out = confidence_pred
        confidence_in = confidence_pred
            
        # domain using confidence
        u_domain_real = confidence_pred.cpu()       

        # log loss test
        confidence_real = self._get_classifier_conditioning(x_real.to(self.device), y_real.to(self.device))
        loss_confid_real = F.mse_loss(confidence_pred, confidence_real).cpu()
        self.log(f'loss_test_confid_real', loss_confid_real)
        
        # log figures
        if self.current_epoch == 0:
            self._log_classifier_analysis(tensorboard_logger, x_real, y_real)
        # if (self.current_epoch == 0) or (self.current_epoch == self.trainer.fit_loop.max_epochs // 2) or (self.current_epoch == self.trainer.fit_loop.max_epochs):
        #     self._log_latent_analysis(tensorboard_logger, n_samples, x_real, y_real, x_fake, class_embed, z_fake, w_fake, confidence_in, confidence_out)
        if (self.current_epoch == 0) or ((self.current_epoch+1)%10 == 0):
            self._log_gan_analysis(tensorboard_logger, x_real, y_real, x_fake, y_fake, c_fake)
            self._log_encoder_analysis(tensorboard_logger, x_real, y_real, x_reconstructed, confidence_real.cpu(), confidence_pred.cpu())
            self._log_domain_analysis(tensorboard_logger, n_samples, x_real, y_real, u_domain_real)
                

    def _log_classifier_analysis(self, tensorboard_logger, x_real, y_real):
        
        # SHOW DECISION BOUNDARY
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
        color = ['C2' if y == 0 else 'C3' for y in y_fake] if self.class_conditioning is not None else 'k'
        plt.scatter(x_fake[:, 0], x_fake[:, 1], alpha=0.5, c=color)
        tensorboard_logger.add_figure("generated_data", plt.gcf(), self.current_epoch)
        

    def _log_latent_analysis(self, tensorboard_logger, n_samples, x_real, y_real, x_fake, class_embed, z, w, confidence_in, confidence_out):
        
        # analysis of W
        import time
        start = time.time()
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
            
            
    def _log_encoder_analysis(self, tensorboard_logger, x_real, y_real, x_reconstructed, confidence_real, confidence_pred):
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        bins = np.histogram(torch.cat((confidence_real.squeeze(), confidence_pred.squeeze())), bins=50)[1]
        ax1.set_title('histogram in u_domain')
        ax1.hist(confidence_real.squeeze(), bins=bins, alpha=0.5, label='ground truth')
        ax1.hist(confidence_pred.squeeze(), bins=bins, alpha=0.5, label='prediction')
        ax1.legend()
        ax2.set_title('reconstructed data')
        ax2.scatter(x_real[y_real==0, 0], x_real[y_real==0, 1], alpha=0.5, c='C0', label='real data - class 0')
        ax2.scatter(x_real[y_real==1, 0], x_real[y_real==1, 1], alpha=0.5, c='C1', label='real data - class 1')
        ax2.scatter(x_reconstructed[y_real==0, 0], x_reconstructed[y_real==0, 1], alpha=0.5, c='C2', label='reconstructed data - class 0')
        ax2.scatter(x_reconstructed[y_real==1, 0], x_reconstructed[y_real==1, 1], alpha=0.5, c='C3', label='reconstructed data - class 1')
        ax2.legend()
        tensorboard_logger.add_figure('reconstructed_data', fig, self.current_epoch)
    
    
    def _log_domain_analysis(self, tensorboard_logger, n_samples, x_real, y_real, u_domain_real):
        
        # Risk coverage curves for REAL test data
        with torch.no_grad(): 
            logits = self.classifier(x_real.to(self.device)).cpu()
        probas = torch.sigmoid(logits).squeeze()
        classif_loss = F.binary_cross_entropy_with_logits(logits.squeeze(), y_real.float(), reduction='none')
        classif_correct = (probas.round() == y_real)  
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
        in_domain = x.squeeze() > cut
        return in_domain
    
    def _get_classifier_conditioning(self, x, y=None):
        with torch.no_grad():
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
    
    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)