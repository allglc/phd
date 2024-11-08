# Selective classification experiments


The goal is to predict when a given classifier will be wrong. It is equivalent to distinguish "easy" data from "hard" data. The baseline is to use the classifier maximum output probability (max softmax).

- 1st way: binary classification well-classified / mis-classified.\
But label (0/1) does not contain much information.

- 2nd way: predict true class probability (TCP), the classifier output for the correct class.\
TCP contains more info, but approach already existing.

- 3rd way: predict classification loss.\
Maybe better than TCP? Maybe loss contains more info?

And here are other parameters:

- data: CIFAR10 for easy and fast experiments (but 99%+ acuracy on train: not many "bad" examples)

- classifier: different pre-trained classifier exist (ResNet, VGG, MobileNet...)

- input of the selection function model: classifier logits, features, images or a combination.


# Experiment details
Training models on CIFAR10 train data does not work. Probably because classifier has 99%+ accuracy so no "bad" examples to learn to distinguish "good" from "bad".\
Instead we split the validation set (~93% accuracy) into 2: first half for training, second half for testing. But overfitting happens.\
Using data augmentation on the training data solves the overfitting problem, so the data configuration retained is: 1st half of CIFAR10 validation set for training (with data augmentations) and 2nd half of CIFAR10 validation set for testing (without data augmentations).


# Results
"model" refers to the selection function model. When using images or features as input, model is initialized as copy of classifier.

In general, wellClassified very close to TCP, loss stays behind; and logits works better than the rest of the inputs.

Using classifier features, kmeans and distance to clusters does not work well for selective classification.


## ResNet on CIFAR10

| classifier | model in | model out | works on train? | works on val? |
|---|---|---|:---:|:---:|
| ResNet | logits | loss | &check; | &cross; |
| ResNet | features | loss | &cross; | &cross; |
| ResNet | images | loss | &cross; | &cross; |
| ResNet | all | loss | &check; | &cross; |
| ResNet | logits | TCP | &check; | &check; (best) |
| ResNet | features | TCP | &cross; | &cross; |
| ResNet | images | TCP | &cross; | &cross; |
| ResNet | all | TCP | &check; | &check; |
| ResNet | logits | wellClassified | &check; | &check; |
| ResNet | features | wellClassified | &cross; | &cross; |
| ResNet | images | wellClassified | &cross; | &cross; |
| ResNet | all | wellClassified | &check; | &check; |

## VGG on CIFAR10

| classifier | model in | model out | works on train? | works on val? |
|---|---|---|:---:|:---:|
| VGG | logits | loss | &check; | &cross; |
| VGG | features | loss | &check; | &cross; |
| VGG | images | loss | &cross; | &cross; |
| VGG | all | loss |  |  |
| VGG | logits | TCP | &check; | &check; (best) |
| VGG | features | TCP | &check; | &check; |
| VGG | images | TCP | &cross; | &cross; |
| VGG | all | TCP |  |  |
| VGG | logits | wellClassified | &check; | &check; |
| VGG | features | wellClassified | &check; | &check; |
| VGG | images | wellClassified | &cross; | &cross; |
| VGG | all | wellClassified |  |  |