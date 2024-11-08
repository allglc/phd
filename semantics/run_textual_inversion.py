import os
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torchvision.transforms.v2 import ToImage, Resize, CenterCrop, Compose
from transformers import CLIPProcessor, CLIPModel
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, StableUnCLIPImg2ImgPipeline

from utils import ECE_calc
path_huggingface = os.path.expandvars('$DSDIR/HuggingFace_Models/') 
path_pretrained = Path(os.path.expandvars('$WORK/MODELS'))
path_sd = str(path_pretrained / 'stabilityai--stable-diffusion-2-1-base')
path_data = Path(os.path.expandvars('$SCRATCH/CUB_200_2011/images'))
path_train_textual_inversion = Path('../results/img_cache/')
path_out_textual_inversion = Path('../results/textual_inversion/')

nb_images = 10

for subfolder in sorted(path_data.iterdir()):
    print(subfolder)

    (path_out_textual_inversion/subfolder.name).mkdir()
    path_train_textual_inversion.mkdir()
    for path_img in sorted((path_data / subfolder).iterdir())[:nb_images]:
        shutil.copy(path_img, path_train_textual_inversion)

    token = f'<{subfolder.name.replace(".", "_")}>'

    command = [
        "accelerate", "launch", "textual_inversion.py",
        "--pretrained_model_name_or_path", str(path_sd),
        "--train_data_dir", str(path_train_textual_inversion),
        "--output_dir", str(path_out_textual_inversion/subfolder.name),
        "--placeholder_token", token,
        "--initializer_token", "bird",
        "--gradient_checkpointing",
        "--mixed_precision", "fp16",
        "--validation_prompt", f"A {token}.",
        "--num_validation_images", "4",
        "--validation_steps", "100"]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        print("Command executed successfully!")
        print("Output:", result.stdout)
    else:
        print("Command failed.")
        print("Error:", result.stderr)

    shutil.rmtree(path_train_textual_inversion)

