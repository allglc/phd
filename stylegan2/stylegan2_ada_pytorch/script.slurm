#!/bin/bash
#SBATCH --job-name=name # nom du job
##SBATCH -C v100-16g # reserver des GPU 16 Go seulement
##SBATCH --qos=qos_gpu-dev # QoS
#SBATCH --output=%j.out # fichier de sortie (%j = job ID)
#SBATCH --error=%j.err # fichier d’erreur (%j = job ID)
#SBATCH --time=20:00:00 # temps maximal d’allocation "(HH:MM:SS)"
#SBATCH --nodes=1 # reserver nœud
#SBATCH --ntasks=1 # reserver tache (ou processus MPI)
#SBATCH --gres=gpu:2 # reserver GPU
#SBATCH --cpus-per-task=24 # reserver CPU par tache (et memoire associee)
#SBATCH --hint=nomultithread # desactiver l’hyperthreading
module purge # nettoyer les modules herites par defaut
conda deactivate # desactiver les environnements herites par defaut
module load pytorch-gpu/py3/1.10.0 # charger les modules
set -x # activer l’echo des commandes
# srun -u python train.py --outdir=$WORK/results/stylegan2/stylegan2-training-runs --data=$WORK/data/MNIST/mnist_stylegan2_blur_noise_maxSeverity3_proba50.zip --gpus=4 --cond=1 --dim=512
srun -u python train.py --outdir=$WORK/results/stylegan2/stylegan2-training-runs --data=$WORK/data/MNIST/mnist_stylegan2.zip --gpus=2 --cond=1 --dim=512
# srun -u python train.py --outdir=$WORK/results/stylegan2/stylegan2-training-runs --data=$WORK/data/cifar10.zip --gpus=2 --cond=1 --dim=512 --cfg=cifar