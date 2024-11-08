import os
from pathlib import Path
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, StableUnCLIPImg2ImgPipeline
from diffusers.utils import load_image
from transformers import CLIPProcessor, CLIPModel
import torch
from torch import nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.io import read_image
import numpy as np
import pandas as pd
import time

import sys
sys.path.append('./dataset-interfaces/')
import dataset_interfaces.inference_utils as infer_utils

# https://huggingface.co/docs/diffusers/using-diffusers/loading_adapters
# https://github.com/MadryLab/dataset-interfaces


path_pretrained = Path(os.path.expandvars('$WORK/MODELS'))
path_huggingface = os.path.expandvars('$DSDIR/HuggingFace_Models/') 
path_results = Path(os.path.expandvars('$WORK/semantics/results/domains'))
path_encoder_root = os.path.expandvars('$WORK/semantics/code/dataset-interfaces/encoder_root_imagenet')


class CategoryClassifier(nn.Module):
    def __init__(self, category='dog'):
        super().__init__()
        weights = ViT_B_16_Weights.DEFAULT
        self.preprocess = weights.transforms()
        self.classifier = vit_b_16(weights=weights)
        self.classifier.eval()
        imagenet_categories = pd.read_csv('../../DATA/imagenet_categories_synset.csv')
        self.category_indices = (imagenet_categories.loc[imagenet_categories['categories'] == category, 'index']).to_numpy()
        self.rest_indices = (imagenet_categories.loc[imagenet_categories['categories'] != category, 'index']).to_numpy()
        
    def forward(self, x, preprocess=False):
        if preprocess:
            if isinstance(x, list):
                x = torch.stack([self.preprocess(x_i) for x_i in x]).to('cuda')
            else:
                raise NotImplementedError()
        logits = self.classifier(x)
        probas = torch.softmax(logits, axis=1)
        probas_category = probas[:, self.category_indices].sum(dim=1, keepdim=True)
        probas_rest = probas[:, self.rest_indices].sum(dim=1, keepdim=True)
        probas = torch.cat((probas_category, probas_rest), axis=1)
        return probas


class SubdomainPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.CLIP_model = CLIPModel.from_pretrained(path_huggingface+"openai/clip-vit-large-patch14")
        self.CLIP_processor = CLIPProcessor.from_pretrained(path_huggingface+"openai/clip-vit-large-patch14")

    def forward(self, images):
        with torch.no_grad():
            inputs = self.CLIP_processor(text=list_prompts, images=images, return_tensors="pt", padding=True)
            outputs = self.CLIP_model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        return probs


if __name__ == '__main__':

    category = 'dog'
    nb_images_to_generate = 100

    start_time = time.time()

    # create results directory
    timestamp = time.strftime('%Y%m%d_%H%M', time.localtime())
    path_results_exp = path_results / timestamp
    path_results_exp.mkdir()

    # Create classifier
    classifier = CategoryClassifier(category).to('cuda')

    # Create generator
    tokens_dict = torch.load(f"{path_encoder_root}/tokens.pt")
    text_encoder = infer_utils.get_text_encoder(model_name=path_encoder_root)
    tokenizer = infer_utils.get_tokenizer(model_name=path_encoder_root)
    pipe = infer_utils.get_pipe(text_encoder=text_encoder, tokenizer=tokenizer, model_name=path_huggingface+"stabilityai/stable-diffusion-2")
    pipe.to('cuda')
    pipe.set_progress_bar_config(disable=True)

    # create subdomain predictor
    subdomain_predictor = SubdomainPredictor()

    # Define domain
    domain_time = ['day', 'night']
    domain_location = ['at the beach', 'in the forest', 'in the city', 'in a house']
    domain_weather = ['sunny', 'snowing', 'raining', 'cloudy']
    list_prompts = [f'a {category} {location}, during the {time_}, it is {weather}'
                    for time_ in domain_time for location in domain_location for weather in domain_weather]
    print(f'Number of subdomains: {len(list_prompts)}')

    # go!
    combination_id = 0
    for time_ in domain_time:
        for location in domain_location:
            for weather in domain_weather:
                prompt = f'a {category} {location}, during the {time_}, it is {weather}'
                print(prompt)

                (path_results_exp / f'{combination_id}').mkdir()
                img_id = 0

                # pick a random class of the category
                class_id = np.random.choice(classifier.category_indices)
                placeholder_token = tokens_dict['tokens'][class_id]

                # generate
                all_images = []
                all_probas = None
                filtered = 0
                while len(all_images) < nb_images_to_generate:
                    class_id = np.random.choice(classifier.category_indices)
                    placeholder_token = tokens_dict['tokens'][class_id]
                    with torch.no_grad():
                        image = pipe(prompt.replace(category, placeholder_token), num_images_per_prompt=1, num_inference_steps=50, guidance_scale=7.5).images[0]
                        image = image.resize((256, 256))
                    # filter out when predicted prompt != original prompt
                    subdomain_probas = subdomain_predictor(image)
                    pred_prompt_id = subdomain_probas.argmax(1)
                    pred_prompt = list_prompts[pred_prompt_id]
                    if pred_prompt == prompt:
                        all_images.append(image)
                    else:
                        filtered += 1

                # classify
                with torch.no_grad():
                    probas = classifier(all_images, preprocess=True)
                all_probas = torch.cat((all_probas, probas)) if all_probas is not None else probas

                # save images
                for img in all_images:
                    img.save(path_results_exp / f'{combination_id}' / f'{img_id}.png')
                    img_id += 1

                # save probas and prompt
                np.save(path_results_exp / f'{combination_id}' / 'probas', all_probas.cpu().numpy())
                with open(path_results_exp / f'{combination_id}' / 'prompt.txt', 'w') as f:
                    f.write(prompt)

                # compute stats
                confid, pred = all_probas.max(axis=1)
                accuracy = (pred == 0).float().mean().item()
                average_confidence = confid.mean().item()
                id_incorrect = pred.nonzero().squeeze(1).tolist()
                df = pd.DataFrame({'id': combination_id, 'nb_images': [img_id], 'category': [category], 'time': [time_], 'location': [location], 'weather': [weather], 
                                'accuracy': [accuracy], 'average_confidence': [average_confidence], 'id_incorrect': [id_incorrect]})

                # save stats
                f_path_test = path_results_exp / 'results.csv'
                if f_path_test.exists():
                    df_0 = pd.read_csv(f_path_test)
                    df = pd.concat([df_0, df], axis=0)
                df.to_csv(f_path_test, index=False)

                print(f'{len(all_images)} images generated, {filtered} were filtered out. Classifier accuracy: {100*accuracy:.2f} %')
                    
                combination_id += 1


    print(f'Finished. Elapsed time: {(time.time()-start_time)/3600:.1f} hours')