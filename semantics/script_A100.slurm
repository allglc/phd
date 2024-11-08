#!/bin/bash
#SBATCH --job-name=job # nom du job
#SBATCH --partition=gpu_p5
#SBATCH -A dcf@a100
#SBATCH -C a100
#SBATCH --output=job%j.out # fichier de sortie (%j = job ID)
#SBATCH --error=job%j.out # fichier d’erreur (%j = job ID) (ici commun avec la sortie)
#SBATCH --time=20:00:00 # temps maximal d’allocation "(HH:MM:SS)"
#SBATCH --nodes=1 # reserver nœud
#SBATCH --ntasks-per-node=1         # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:1 # reserver GPU
#SBATCH --cpus-per-task=10 # reserver CPU par tache (et memoire associee)
#SBATCH --hint=nomultithread # desactiver l’hyperthreading
module purge # nettoyer les modules herites par defaut
conda deactivate # desactiver les environnements herites par defaut
module load cpuarch/amd
# module load pytorch-gpu/py3/2.0.0 # charger les modules
module load python/3.11.5
conda activate torch21
python eval_domains.py