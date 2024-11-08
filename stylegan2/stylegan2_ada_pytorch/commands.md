# IRT JupyterLab

## Prepare MNIST dataset
python dataset_tool.py --source=/home/jovyan/data/MNIST/raw/train-images-idx3-ubyte.gz --dest=/home/jovyan/data/MNIST/mnist_stylegan2.zip
python dataset_tool.py --source=/home/jovyan/data/MNIST/raw/train-images-idx3-ubyte.gz --dest=/home/jovyan/data/MNIST/mnist_stylegan2_noise.zip --corruptions gaussian_noise
python dataset_tool.py --source=/home/jovyan/data/MNIST/raw/train-images-idx3-ubyte.gz --dest=/home/jovyan/data/MNIST/mnist_stylegan2_blur_noise.zip --corruptions gaussian_blur --corruptions gaussian_noise

## Train on MNIST
python train.py --outdir=/home/jovyan/results/stylegan2-training-runs --data=/home/jovyan/data/MNIST/mnist_stylegan2_noise_blur.zip --gpus=2 --cond=1




# ONERA DeepLab

## Prepare MNIST dataset
python dataset_tool.py --source=/d/alecoz/projects/stylegan2/data/MNIST/raw/train-images-idx3-ubyte.gz --dest=/d/alecoz/projects/stylegan2/data/MNIST/mnist_stylegan2.zip
python dataset_tool.py --source=/d/alecoz/projects/stylegan2/data/MNIST/raw/train-images-idx3-ubyte.gz --dest=/d/alecoz/projects/stylegan2/data/MNIST/mnist_stylegan2_blur_noise.zip --corruptions gaussian_blur --corruptions gaussian_noise

## Train on MNIST
python train.py --outdir=/d/alecoz/projects/stylegan2/results/stylegan2-training-runs --data=/d/alecoz/projects/stylegan2/data/MNIST/mnist_stylegan2_noise_blur.zip --gpus=2 --cond=1 --dim=512

# Project
python projector.py --outdir=out --target=ffhq.png --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl


# Jean Zay
python dataset_tool.py --source=$WORK/data/MNIST/raw/train-images-idx3-ubyte.gz --dest=$WORK/data/MNIST/mnist_stylegan2.zip
python dataset_tool.py --source=$WORK/data/MNIST/raw/train-images-idx3-ubyte.gz --dest=$WORK/data/MNIST/mnist_stylegan2_blur_noise_maxSeverity3_proba50.zip --corruptions gaussian_blur --corruptions gaussian_noise --corr_max_severity=3 --corrupt_proba=0.5
python train.py --outdir=$WORK/results/stylegan2/stylegan2-training-runs --data=$WORK/data/MNIST/mnist_stylegan2_noise.zip --gpus=1 --cond=1 --dim=512
