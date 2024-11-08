# Commands for StyleGAN2 (on Jean Zay)
`python dataset_tool.py --source=$WORK/DATA/MNIST/raw/train-images-idx3-ubyte.gz --dest=$WORK/DATA/MNIST/mnist_stylegan2.zip`

`python dataset_tool.py --source=$WORK/DATA/MNIST/raw/train-images-idx3-ubyte.gz --dest=$WORK/DATA/MNIST/mnist_stylegan2_blur_noise_maxSeverity5_proba100.zip --corruptions gaussian_blur --corruptions gaussian_noise --corr_max_severity=5 --corrupt_proba=1`
`python dataset_tool.py --source=$WORK/DATA/MNIST/raw/t10k-images-idx3-ubyte.gz --dest=$WORK/DATA/MNIST/mnistTest_stylegan2_blur_noise_maxSeverity5_proba100.zip --corruptions gaussian_blur --corruptions gaussian_noise --corr_max_severity=5 --corrupt_proba=1`

`module load pytorch-gpu/py3/1.10.0`

`python train.py --outdir=$WORK/uncertainty-conditioned-gan/results/stylegan2-training-runs --data=$WORK/DATA/MNIST/mnist_stylegan2.zip --gpus=1 --cond=1`

`python train.py --outdir=$WORK/uncertainty-conditioned-gan/results/stylegan2-training-runs --data=$WORK/DATA/MNIST/mnist_stylegan2.zip --gpus=1 --cond=1 --classifier_path=$WORK/uncertainty-conditioned-gan/results/classifiers/CNN_mnist_clean_20230525_1119.pth`

`python train.py --outdir=$WORK/uncertainty-conditioned-gan/results/stylegan2-training-runs --data=$WORK/DATA/MNIST/mnist_stylegan2_blur_noise_maxSeverity5_proba50.zip --gpus=1 --cond=1 --classifier_path=$WORK/uncertainty-conditioned-gan/results/classifiers/CNN_mnist_stylegan2_blur_noise_maxSeverity5_proba50_20230525_1124.pth`

`python train.py --outdir=$WORK/uncertainty-conditioned-gan/results/stylegan2-training-runs --data=$WORK/DATA/MNIST/mnist_stylegan2_blur_noise_maxSeverity5_proba100.zip --gpus=1 --cond=1 --classifier_path=$WORK/uncertainty-conditioned-gan/results/classifiers/CNN_mnist_stylegan2_blur_noise_maxSeverity5_proba100_20230525_1128.pth`

`python train.py --outdir=$WORK/uncertainty-conditioned-gan/results/stylegan2-training-runs --data=$WORK/DATA/MNIST/mnist_stylegan2_blur_noise_maxSeverity5_proba100.zip --gpus=1 --cond=1 --classifier_path=$WORK/uncertainty-conditioned-gan/results/classifiers/CNN_mnist_stylegan2_blur_noise_maxSeverity5_proba100_20230525_1128.pth`


## Faces data
`python dataset_tool.py --source=$DSDIR/FlickrFace/images1024x1024 --dest=$WORK/DATA/ffhq128x128.zip --width=128 --height=128`

`python train.py --outdir=$WORK/uncertainty-conditioned-gan/results/stylegan2-training-runs --data=$WORK/DATA/ffhq128x128.zip --gpus=2 --classifier_path=$WORK/uncertainty-conditioned-gan/results/classifiers/VGG16_celeba_20230531_1215.pth`

# Bugs
sur Jean Zay,  pas internet donc besoin de télécharger "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt" et mettre sous ".cache\dnnlib\downloads\a866f8d678872dcf6fcf60ddd09807ab_https___nvlabs-fi-cdn.nvidia.com_stylegan2-ada-pytorch_pretrained_metrics_inception-2015-12-05.pt"
Aussi, marche pas avec pytorch 1.9.0 (ok avec 1.10.0)