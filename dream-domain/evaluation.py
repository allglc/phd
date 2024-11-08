import numpy as np
import torch

def ECE_calc(samples_certainties, num_bins=15, adaptive=False):
    '''https://github.com/idogalil/benchmarking-uncertainty-estimation-performance'''
    indices_sorting_by_confidence = torch.argsort(samples_certainties[:, 0], descending=False)  # Notice the reverse sorting
    samples_certainties = samples_certainties[indices_sorting_by_confidence]
    samples_certainties = samples_certainties.transpose(0, 1)
    if adaptive:
        certainties = samples_certainties[0]
        N = certainties.shape[0]
        step_size = int(N / (num_bins - 1))
        bin_boundaries = [certainties[i].item() for i in range(0, certainties.shape[0], step_size)]
        bin_boundaries[0] = 0
        bin_boundaries[-1] = certainties[-1]
        bin_boundaries = torch.tensor(bin_boundaries)
    else:
        bin_boundaries = torch.linspace(0, 1, num_bins + 1)        
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bins_accumulated_error = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        bin_indices = torch.logical_and(samples_certainties[0] <= bin_upper, samples_certainties[0] > bin_lower)
        if bin_indices.sum() == 0:
            continue  # This is an empty bin
        bin_confidences = samples_certainties[0][bin_indices]
        bin_accuracies = samples_certainties[1][bin_indices]
        bin_avg_confidence = bin_confidences.mean()
        bin_avg_accuracy = bin_accuracies.mean()
        bin_error = torch.abs(bin_avg_confidence - bin_avg_accuracy)
        bins_accumulated_error += bin_error * bin_confidences.shape[0]

    expected_calibration_error = bins_accumulated_error / samples_certainties.shape[1]
    return expected_calibration_error.item()


def AUROC(samples_certainties, sort=True):
    '''https://github.com/idogalil/benchmarking-uncertainty-estimation-performance'''
    # Calculating AUROC in a similar way gamma correlation is calculated. The function can easily return both.
    if sort:
        indices_sorting_by_confidence = torch.argsort(samples_certainties[:, 0], descending=True)
        samples_certainties = samples_certainties[indices_sorting_by_confidence]
    total_samples = samples_certainties.shape[0]
    incorrect_after_i = np.zeros((total_samples))

    for i in range(total_samples - 1, -1, -1):
        if i == total_samples - 1:
            incorrect_after_i[i] = 0
        else:
            incorrect_after_i[i] = incorrect_after_i[i+1] + (1 - int(samples_certainties[i+1][1]))
            # Note: samples_certainties[i+1][1] is the correctness label for sample i+1

    n_d = 0  # amount of different pairs of ordering
    n_s = 0  # amount of pairs with same ordering
    incorrect_before_i = 0
    for i in range(total_samples):
        if i != 0:
            incorrect_before_i += (1 - int(samples_certainties[i-1][1]))
        if samples_certainties[i][1]:
            # if i is a correct prediction, i's ranking 'agrees' with all the incorrect that are to come
            n_s += incorrect_after_i[i]
            # i's ranking 'disagrees' with all incorrect predictions that preceed it (i.e., ranked more confident)
            n_d += incorrect_before_i
        else:
            # else i is an incorrect prediction, so i's ranking 'disagrees' with all the correct predictions after
            n_d += (total_samples - i - 1) - incorrect_after_i[i]  # (total_samples - i - 1) = all samples after i
            # i's ranking 'agrees' with all correct predictions that preceed it (i.e., ranked more confident)
            n_s += i - incorrect_before_i

    return n_s / (n_s + n_d)


def test_encoder(encoder, dataloader, num_classes=100, device='cuda', anchor=None, classifier=None):

    confidences = []
    corrects = []
    max_cosims = []
    true_class_cosims = []
    pred_class_cosims = []

    i = 0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.shape[0]
        
        # Forward pass through the frozen classifier
        with torch.no_grad():
            classif_outputs = classifier(images)
            probas = torch.softmax(classif_outputs, 1)
            confidence, predicted = torch.max(probas, 1)
            correct = (predicted == labels)
        confidences.append(confidence)
        corrects.append(correct)

        # Forward pass
        with torch.no_grad():
            features = encoder(images)
            features = features.unsqueeze(1).repeat(1, num_classes, 1)
        # Cosine similarity
        cosim = torch.nn.functional.cosine_similarity(anchor.unsqueeze(0).repeat(features.shape[0], 1, 1), features, 2)
        max_cosim = cosim.max(dim=1).values
        max_cosims.append(max_cosim)
        true_class_cosim = cosim[torch.arange(cosim.shape[0]), labels]
        true_class_cosims.append(true_class_cosim)
        pred_class_cosim = cosim[torch.arange(cosim.shape[0]), predicted]
        pred_class_cosims.append(pred_class_cosim)

        if i < 5000:
            i += batch_size
        else:
            print('early stop iterating dataloader')
            break # early stop for train dataloader

    confidences = torch.cat(confidences)
    corrects = torch.cat(corrects)
    max_cosims = torch.cat(max_cosims)
    true_class_cosims = torch.cat(true_class_cosims)
    pred_class_cosims = torch.cat(pred_class_cosims)

    samples_certainties = torch.stack((confidences.cpu(), corrects.cpu()), dim=1)
    auroc_classif = AUROC(samples_certainties)
    ece_classif = ECE_calc(samples_certainties)

    samples_certainties = torch.stack((max_cosims.cpu(), corrects.cpu()), dim=1)
    auroc_encoder_max = AUROC(samples_certainties)
    ece_encoder_max = ECE_calc(samples_certainties)

    samples_certainties = torch.stack((true_class_cosims.cpu(), corrects.cpu()), dim=1)
    auroc_encoder_true = AUROC(samples_certainties)
    ece_encoder_true = ECE_calc(samples_certainties)

    samples_certainties = torch.stack((pred_class_cosims.cpu(), corrects.cpu()), dim=1)
    auroc_encoder_pred = AUROC(samples_certainties)
    ece_encoder_pred = ECE_calc(samples_certainties)

    return {'auroc_classif': auroc_classif, 'ece_classif': ece_classif, 
            'auroc_encoder_max': auroc_encoder_max, 'ece_encoder_max': ece_encoder_max, 
            'auroc_encoder_true': auroc_encoder_true, 'ece_encoder_true': ece_encoder_true, 
            'auroc_encoder_pred': auroc_encoder_pred, 'ece_encoder_pred': ece_encoder_pred,
            'avg_cosim_max_correct': max_cosims[corrects].mean().item(), 'avg_cosim_max_incorrect': max_cosims[corrects.bitwise_not()].mean().item(),
            'avg_cosim_true_correct': true_class_cosims[corrects].mean().item(), 'avg_cosim_true_incorrect': true_class_cosims[corrects.bitwise_not()].mean().item(),
            'avg_cosim_pred_correct': pred_class_cosims[corrects].mean().item(), 'avg_cosim_pred_incorrect': pred_class_cosims[corrects.bitwise_not()].mean().item(),
            }