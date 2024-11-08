#!/bin/bash
#SBATCH --job-name=big_calcul # nom du job
#SBATCH --qos=qos_gpu-t3 # QoS
#SBATCH --partition=gpu_p2
#SBATCH --output=%j.out # fichier de sortie (%j = job ID)
#SBATCH --error=%j.err # fichier d’erreur (%j = job ID)
#SBATCH --time=20:00:00 # temps maximal d’allocation "(HH:MM:SS)" (MAX 20h sur qos_gpu-t3 (default))
#SBATCH --nodes=1 # reserver nœud
#SBATCH --ntasks=1 # reserver tache (ou processus MPI)
#SBATCH --gres=gpu:8 # reserver GPU
#SBATCH --cpus-per-task=24 # reserver CPU par tache (et memoire associee)
#SBATCH --hint=nomultithread # desactiver l’hyperthreading
module purge # nettoyer les modules herites par defaut
conda deactivate # desactiver les environnements herites par defaut
module load pytorch-gpu/py3/1.10.0 # charger les modules
set -x # activer l’echo des commandes
srun -u python project_dataset.py
# srun -u python train.py --outdir=$WORK/results/stylegan2/stylegan2-training-runs --data=$WORK/data/MNIST/mnist_stylegan2.zip --gpus=8 --cond=1 --dim=512