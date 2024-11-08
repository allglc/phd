# %%
import os
import time 
import click
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
# sns.set_context("paper")
sns.set_style("ticks")
import copy
import torch
from pathlib import Path
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score

from cifar10_models.vgg import vgg16_bn
from cifar10_models.mobilenetv2 import mobilenet_v2
from cifar10_models.resnet import resnet50

import sys
sys.path.append('./benchmarking-uncertainty-estimation-performance-main/utils')
from uncertainty_metrics import AURC_calc, AUROC, EAURC_calc, ECE_calc, calc_adaptive_bin_size

from utils import load_dataloaders_CIFAR, EasyDict


torch.manual_seed(0)
np.random.seed(0)
rng = np.random.default_rng(0)

path_data = Path(os.path.expandvars('$DSDIR/'))
path_results = Path.cwd().parent / 'results'
device = 'cuda'



# %%
def get_classif_selection_outputs(model, classifier, dataloader, selection_inputs, prediction, n_classes):
    total_classif_correct = None
    total_probas = None
    total_selection_out = None
    total_tcp = None

    classifier.eval()
    model.eval()

    for batch in dataloader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        
        # get classification output
        with torch.no_grad():
            logits = classifier(images)
        probas = torch.softmax(logits, dim=1)
        tcp = probas[torch.nn.functional.one_hot(labels, n_classes).bool()]
        class_predictions = logits.argmax(dim=1)
        classif_correct = (class_predictions == labels).float()
        
        # get selection output [0, 1]
        if selection_inputs == 'images':
            inputs = images
        elif selection_inputs == 'features':
            inputs = images # same network but only trainable after features
        elif selection_inputs == 'logits':
            inputs = logits
        elif selection_inputs == 'all':
            inputs = (images, logits)
        with torch.no_grad():
            outputs = model(inputs)
        if prediction == 'wellClassified' or prediction == 'tcp':
            outputs = torch.sigmoid(outputs)
        elif prediction == 'loss':
            outputs = (outputs - outputs.min()) / (outputs.max() - outputs.min()) # normalize
            
        # save
        probas = probas.cpu()
        tcp = tcp.cpu()
        classif_correct = classif_correct.cpu()
        outputs = outputs.squeeze().cpu()
        total_probas = probas if total_probas is None else torch.cat((total_probas, probas), dim=0)
        total_tcp = tcp if total_tcp is None else torch.cat((total_tcp, tcp), dim=0)
        total_classif_correct = classif_correct if total_classif_correct is None else torch.cat((total_classif_correct, classif_correct), dim=0)
        total_selection_out = outputs if total_selection_out is None else torch.cat((total_selection_out, outputs), dim=0)
    
    return total_probas, total_tcp, total_classif_correct, total_selection_out


def get_selective_risk_at_cut(cut, total_classif_correct, total_selection_out, prediction):
    if prediction == 'loss': # increasing loss lowers confidence
        selected_idx = total_selection_out < cut
    else:
        selected_idx = total_selection_out > cut
    accuracy = total_classif_correct[selected_idx].mean()
    risk = 1 - accuracy
    coverage = selected_idx.float().mean()
    
    return accuracy.item(), risk.item(), coverage.item()


class SelectionFromAll(nn.Module):
    
    def __init__(self, classifier):
        super().__init__()
        
        classifier_copy = copy.deepcopy(classifier)
        
        self.features_extractor = nn.Sequential(*list(classifier_copy.children())[:-1])
        self.features_extractor.eval()
        
        n_logits = 10
        self.nn_from_logits = nn.Sequential(
            nn.Linear(n_logits, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128))
        dim_from_logits = 128
        
        n_features = list(classifier_copy.children())[-1].in_features
        self.nn_from_features = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128))
        dim_from_features = 128
        
        self.nn_from_images = nn.Sequential(*list(classifier_copy.children())[:-1])
        dim_from_images = list(classifier_copy.children())[-1].in_features
        
        self.head = nn.Sequential(
            nn.Linear(dim_from_logits + dim_from_features + dim_from_images, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        images, logits = x
        features = self.features_extractor(images).squeeze()
                
        images_nn_out = self.nn_from_images(images).squeeze()
        features_nn_out = self.nn_from_features(features)
        logits_nn_out = self.nn_from_logits(logits)
                
        concat = torch.cat((images_nn_out, features_nn_out, logits_nn_out), dim=1)
        
        g = self.head(concat)
        
        return g
    
# %%
def load_classifier(model_architecture):
    if model_architecture == 'VGG':
        classifier = vgg16_bn(pretrained=True)
    elif model_architecture == 'MobileNet':
        classifier = mobilenet_v2(pretrained=True)
    elif model_architecture == 'ResNet':
        classifier = resnet50(pretrained=True)
    classifier.eval()
    return classifier

def load_selection_function(classifier, selection_inputs, model_architecture):
    if selection_inputs == 'images':
        if model_architecture == 'ResNet':
            n_features = classifier.fc.in_features
            selection_function = copy.deepcopy(classifier)
            selection_function.fc = nn.Linear(n_features, 1)
        else:
            n_features = classifier.classifier[-1].in_features
            selection_function = copy.deepcopy(classifier)
            selection_function.classifier[-1] = nn.Linear(n_features, 1) # replace last layer

    elif selection_inputs == 'features':
        if model_architecture == 'ResNet':
            n_features = classifier.fc.in_features
            selection_function = copy.deepcopy(classifier)
            selection_function.fc = nn.Linear(n_features, 1)
            for param in selection_function.parameters():
                param.requires_grad = False
            for param in selection_function.fc.parameters():
                param.requires_grad = True
        else:
            n_features = classifier.classifier[-1].in_features
            selection_function = copy.deepcopy(classifier)
            selection_function.classifier[-1] = nn.Linear(n_features, 1) # replace last layer
            for param in selection_function.parameters():
                param.requires_grad = False
            for param in selection_function.classifier.parameters():
                param.requires_grad = True

    elif selection_inputs == 'logits':
        selection_function = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1))
    
    elif selection_inputs == 'all':
        selection_function = SelectionFromAll(classifier)
        
    return selection_function

# if DATASET == 'CIFAR10':
#     if MODEL_ARCHITECTURE == 'VGG':
#         classifier = vgg11_bn(pretrained=True).to(device)
#         features_extractor = nn.Sequential(*list(classifier.features.children())).to(device)
#     elif MODEL_ARCHITECTURE == 'MobileNet':
#         classifier = mobilenet_v2(pretrained=True).to(device)
#         features_extractor = nn.Sequential(*list(classifier.features.children())).to(device)
#     elif MODEL_ARCHITECTURE == 'ResNet':
#         classifier = resnet18(pretrained=True).to(device)
#         features_extractor = nn.Sequential(*list(classifier.children())[:-1]).to(device)

# %%
def predict_well_classified(logits, labels, selection_inputs, model):
    class_predictions = logits.argmax(dim=1)
    targets = (class_predictions == labels).float()
    selection = model(selection_inputs).squeeze()
    selective_loss = F.binary_cross_entropy_with_logits(selection, targets, pos_weight=torch.tensor([0.1], device=device))
    # selective_loss += torch.special.entr(torch.sigmoid(selection))
    return selective_loss
    
def predict_tcp(logits, labels, selection_inputs, model, n_classes):
    probas = torch.softmax(logits, dim=1)
    tcp = probas[torch.nn.functional.one_hot(labels, n_classes).bool()]
    selection = torch.sigmoid(model(selection_inputs)).squeeze()
    selective_loss = F.mse_loss(selection, tcp)
    return selective_loss
        
def predict_loss(logits, labels, selection_inputs, model, criterion):          
    selection = (model(selection_inputs)).squeeze()
    selective_loss = F.mse_loss(selection, criterion(logits, labels))
    return selective_loss


@click.command()
@click.option('--dataset', default='CIFAR10')
@click.option('--classifier', default='ResNet', help='VGG MobileNet ResNet')
@click.option('--prediction', default='loss', help='wellClassified tcp loss')
@click.option('--inputs', default='logits', help='images features logits')
def main(**kwargs):
    
    
    config = EasyDict(kwargs)
    DATASET = config.dataset
    MODEL_ARCHITECTURE = config.classifier
    PREDICTION = config.prediction
    SELECTION_INPUTS = config.inputs
    BATCH_SIZE = 64
    device = 'cuda:0'

    # create experiment folder to save results and logs
    timestamp = time.strftime('%Y-%m-%d_%H%M%S', time.localtime())
    tag = f'_{MODEL_ARCHITECTURE}_{PREDICTION}_{SELECTION_INPUTS}'
    path_results_exp = path_results / 'selection_function' / (timestamp+tag)
    if not path_results_exp.exists(): path_results_exp.mkdir(parents=True)
    with open(os.path.join(path_results_exp, 'training_options.json'), 'wt') as f:
        json.dump(config, f, indent=2)
        
    writer = SummaryWriter(log_dir=path_results_exp)
    # logger = TensorBoardLogger(save_dir=path_results_exp, name='', version='')

    
    
    # LOAD DATA
    if DATASET == 'CIFAR10':
        path_dataset = path_data

    dataloader_train, dataloader_val = load_dataloaders_CIFAR(path_dataset, BATCH_SIZE)
    n_classes = 10#len(dataloader_train.dataset.classes)

    # LOAD MODELS
    classifier = load_classifier(MODEL_ARCHITECTURE)
    selection_function = load_selection_function(classifier, SELECTION_INPUTS, MODEL_ARCHITECTURE)

    # RUN
    classifier = classifier.to(device)
    model = selection_function.to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
    num_epochs = 400

    for epoch in range(1, num_epochs+1):
        print(f'Epoch {epoch}/{num_epochs}')
        
        classifier.eval()
        model.train()  # Set model to training mode

        total_selective_loss = None
        # Iterate over data.
        for inputs, labels in dataloader_train:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # # FIRST WAY: SCALE LOSS WITH SELECTION (freeze all except last layer)
            # # classify inputs
            # with torch.no_grad():
            #     class_predictions = classifier(inputs)
            #     loss_without_selection = criterion(class_predictions, labels)
            # # forward
            # selected_data = torch.sigmoid(model(inputs)).squeeze()
            # selective_loss = (loss_without_selection * selected_data).mean() + torch.maximum(0.8 - selected_data.mean(), torch.tensor(0))**2

            # # SECOND WAY: SCALE TARGETS (EQUIVALENT TO FIRST WAY)
            # class_predictions = classifier(inputs)
            # selected_data = torch.sigmoid(model(inputs)).squeeze()
            # labels = torch.nn.functional.one_hot(labels, n_classes)
            # labels = labels * selected_data.unsqueeze(1).expand(-1, n_classes) # equivalent to scaling loss
            # selective_loss = criterion(class_predictions, labels).sum() + 32*torch.maximum(0.8 - selected_data.mean(), torch.tensor(0))**2
            
            with torch.no_grad():
                logits = classifier(inputs)
            
            if SELECTION_INPUTS == 'images':
                selection_in = inputs
            elif SELECTION_INPUTS == 'features':
                selection_in = inputs # same network but only trainable after features
            elif SELECTION_INPUTS == 'logits':
                selection_in = logits
            elif SELECTION_INPUTS == 'all':
                selection_in = (inputs, logits)
            
            if PREDICTION == 'wellClassified':
                selective_loss = predict_well_classified(logits, labels, selection_in, model)
            elif PREDICTION == 'tcp':
                selective_loss = predict_tcp(logits, labels, selection_in, model, n_classes)
            elif PREDICTION == 'loss':        
                selective_loss = predict_loss(logits, labels, selection_in, model, criterion)
                        
            selective_loss.backward()
            optimizer.step()
            
            total_selective_loss = selective_loss if total_selective_loss is None else total_selective_loss + selective_loss
            
        # print(f'Total loss: {total_selective_loss}')
        # 
        # _, _, total_classif_correct, total_selection_out = get_classif_selection_outputs(model, classifier, dataloader_train)
        # accuracy, risk, coverage = get_selective_risk_at_cut(0.5, total_classif_correct, total_selection_out)
        # print(f'Train_risk: {risk:.4f}, train_accuracy: {100*accuracy:.2f}%, train_coverage: {100*coverage:.2f}%')
        # 
        # _, _, total_classif_correct, total_selection_out = get_classif_selection_outputs(model, classifier, dataloader_val)
        # accuracy, risk, coverage = get_selective_risk_at_cut(0.5, total_classif_correct, total_selection_out)
        # print(f'Val_risk: {risk:.4f}, val_accuracy: {100*accuracy:.2f}%, val_coverage: {100*coverage:.2f}%')
        
        scheduler.step()


        # EVALUATE AND PLOT
        writer.add_scalar('epoch_total_loss', total_selective_loss, epoch)

        if epoch % 50 == 0:
            for dataloader, name in zip([dataloader_train, dataloader_val], ['train', 'val']):
                total_probas, total_tcp, total_classif_correct, total_selection_out = get_classif_selection_outputs(
                    model, classifier, dataloader, SELECTION_INPUTS, PREDICTION, n_classes)

                if name == 'val':
                    samples_certainties = torch.stack((total_selection_out, total_classif_correct), dim=1)
                    writer.add_scalar('AUROC', AUROC(samples_certainties), epoch)

                # use selection function
                domain_cutoff = np.linspace(0, 1, 1000)
                coverage = np.zeros_like(domain_cutoff)
                risk = np.zeros_like(domain_cutoff)
                accuracy = np.zeros_like(domain_cutoff)
                for i, cut in enumerate(domain_cutoff):
                    acc, _, cov = get_selective_risk_at_cut(cut, total_classif_correct, total_selection_out, PREDICTION)
                    coverage[i] = cov
                    accuracy[i] = acc
                    
                # baseline: random selection
                domain_cutoff_random = np.linspace(0, 1, 100)
                coverage_random = np.zeros_like(domain_cutoff_random)
                risk_random = np.zeros_like(domain_cutoff_random)
                acc_random = np.zeros_like(domain_cutoff_random)
                for i, cut in enumerate(domain_cutoff_random):
                    nb_samples = int((1-cut) * len(total_probas)) # 1-cut to be coherent with other indices below (low value -> high coverage)
                    idx_domain = rng.choice(np.arange(len(total_probas)), size=nb_samples, replace=False)
                    coverage_random[i] = total_probas[idx_domain].shape[0] / len(total_probas)
                    # risk_random[i] = total_classif_loss[idx_domain].mean()
                    acc_random[i] = total_classif_correct[idx_domain].float().mean()
                    
                # baseline: max softmax
                domain_cutoff_baseline = np.linspace(0, 1, 1000)
                coverage_baseline = np.zeros_like(domain_cutoff_baseline)
                risk_baseline = np.zeros_like(domain_cutoff_baseline)
                acc_baseline = np.zeros_like(domain_cutoff_baseline)
                for i, cut in enumerate(domain_cutoff_baseline):
                    idx_domain = total_probas.max(dim=1).values > cut
                    coverage_baseline[i] = idx_domain.float().mean()
                    acc_baseline[i] = total_classif_correct[idx_domain].float().mean()

                # baseline: TCP (same results as thesholding on loss)
                domain_cutoff_baselineTCP = np.linspace(0, 1, 1000)
                coverage_baselineTCP = np.zeros_like(domain_cutoff_baselineTCP)
                risk_baselineTCP = np.zeros_like(domain_cutoff_baselineTCP)
                acc_baselineTCP = np.zeros_like(domain_cutoff_baselineTCP)
                for i, cut in enumerate(domain_cutoff_baselineTCP):
                    idx_domain = total_tcp > cut
                    coverage_baselineTCP[i] = idx_domain.float().mean()
                    acc_baselineTCP[i] = total_classif_correct[idx_domain].float().mean()
                    
                # baseline loss: select based on loss value. same as TCP

                # plot
                fig, (ax1) = plt.subplots(1, 1)
                # ax1.set_title(f'coverage vs. accuracy ({name})\n(obtained by varying confidence/uncertainty threshold)')
                ax1.plot(coverage, accuracy, label='selection function')
                ax1.plot(coverage_baseline, acc_baseline, label='baseline (max softmax)')
                # ax1.plot(coverage_random, acc_random, label='baseline (random)')
                ax1.plot(coverage_baselineTCP, acc_baselineTCP, label='Oracle')

                ax1.legend()
                ax1.set_xlabel('coverage')
                ax1.set_ylabel('accuracy')

                plt.savefig(str(path_results / 'selection_function' / (timestamp+tag+'_AccCovCurve.pdf')))
                writer.add_figure(f'coverture_accuracy_{name}', fig, epoch)

    # compute metrics
    total_probas, _, correct, confidences = get_classif_selection_outputs(
                    model, classifier, dataloader_val, SELECTION_INPUTS, PREDICTION, n_classes)
    confidences_baseline = total_probas.max(dim=1).values
    samples_certainties_baseline = torch.stack((confidences_baseline, correct), dim=1)
    samples_certainties = torch.stack((confidences, correct), dim=1)
    metrics = {}
    metrics['AUROC_baseline'] = roc_auc_score(correct.numpy(), confidences_baseline.numpy())
    metrics['AUROC'] = roc_auc_score(correct.numpy(), confidences.numpy())
    metrics['AURC_baseline'] = AURC_calc(samples_certainties_baseline)
    metrics['AURC'] = AURC_calc(samples_certainties)
    metrics['E-AURC_baseline'] = EAURC_calc(metrics['AURC_baseline'], correct.mean()).item()
    metrics['E-AURC'] = EAURC_calc(metrics['AURC'], correct.mean()).item()
    
    df_res_test = pd.DataFrame([{'model': MODEL_ARCHITECTURE, 'inputs': SELECTION_INPUTS, 'outputs': PREDICTION, **metrics}])
    f_path_test = path_results / f'benchmark_selective_classification.csv'

    if os.path.exists(f_path_test):
        df_0 = pd.read_csv(f_path_test)
        df_res_test = pd.concat([df_0, df_res_test], axis=0)
    df_res_test.to_csv(f_path_test, index=False)


if __name__ == '__main__':
    main()