python dream-ood-main/scripts/dream_ood.py --plms \
--n_iter 50 --n_samples 3 \
--outdir /gpfswork/rech/dcf/ulb98yg/dream-domain/results/generated_images \
--loaded_embedding /gpfswork/rech/dcf/ulb98yg/dream-domain/results/embeddings/inlier_npos_embed_noise_0.01_select_1000_KNN_300.npy \
--ckpt /gpfswork/rech/dcf/ulb98yg/DATA/sd-v1-4.ckpt \
--id_data in100 \
--skip_grid



python dream-ood-main/scripts/dream_ood.py --plms \
--n_iter 50 --n_samples 3 \
--outdir /gpfswork/rech/dcf/ulb98yg/dream-domain/results/generated_images \
--loaded_embedding /gpfswork/rech/dcf/ulb98yg/dream-domain/results/embeddings/outlier_npos_embed.npy \
--ckpt /gpfswork/rech/dcf/ulb98yg/DATA/sd-v1-4.ckpt \
--id_data in100 \
--skip_grid

# Original Stable Diffusion
export PYTHONPATH=$PYTHONPATH:.
python scripts/txt2img.py --prompt "a photograph of an astronaut riding a horse" --plms \
--ckpt /gpfswork/rech/dcf/ulb98yg/DATA/sd-v1-4.ckpt \
--outdir /gpfswork/rech/dcf/ulb98yg/dream-domain/results/generated_images


# encoder custom standard
No change to original encoder

# encoder custom
For incorrect images, Cosine similarity should be -1 for true class

# encoder custom 1
For incorrect images, Cosine similarity should be -1 for pred class


