# domain-images
Define a domain for images




# Commands for StyleGAN2 (on Jean Zay)
`python dataset_tool.py --source=$WORK/data/MNIST/raw/train-images-idx3-ubyte.gz --dest=$WORK/data/MNIST/mnist_stylegan2.zip`

`python dataset_tool.py --source=$WORK/data/MNIST/raw/train-images-idx3-ubyte.gz --dest=$WORK/data/MNIST/mnist_stylegan2_blur_noise_maxSeverity5_proba100.zip --corruptions gaussian_blur --corruptions gaussian_noise --corr_max_severity=5 --corrupt_proba=1`
`python dataset_tool.py --source=$WORK/data/MNIST/raw/t10k-images-idx3-ubyte.gz --dest=$WORK/data/MNIST/mnistTest_stylegan2_blur_noise_maxSeverity5_proba100.zip --corruptions gaussian_blur --corruptions gaussian_noise --corr_max_severity=5 --corrupt_proba=1`

`module load pytorch-gpu/py3/1.9.0`

`python train.py --outdir=$WORK/results/domain-images/stylegan2-training-runs --data=$WORK/data/MNIST/mnist_stylegan2.zip --gpus=1 --cond=1`

`python train.py --outdir=$WORK/results/domain-images/stylegan2-training-runs --data=$WORK/data/MNIST/mnist_stylegan2.zip --gpus=1 --cond=1 --classifier_path=$WORK/results/domain-images/classifiers/CNN_mnist_clean_20230303_1618.pth`

# Bugs
sur Jean Zay,  pas internet donc besoin de télécharger "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt" et mettre sous ".cache\dnnlib\downloads\a866f8d678872dcf6fcf60ddd09807ab_https___nvlabs-fi-cdn.nvidia.com_stylegan2-ada-pytorch_pretrained_metrics_inception-2015-12-05.pt"
Aussi, marche pas avec pytorch 1.9.0