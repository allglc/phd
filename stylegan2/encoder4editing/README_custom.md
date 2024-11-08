# Train
```
python scripts/train.py \
--dataset_type mnistCorrupt_encode \
--exp_dir ../../results/e4e/experiment \
--start_from_latent_avg \
--use_w_pool \
--w_discriminator_lambda 0.1 \
--progressive_start 20000 \
--id_lambda 0.5 \
--val_interval 10000 \
--max_steps 200000 \
--stylegan_size 512 \
--stylegan_weights /d/alecoz/projects/stylegan2/results/stylegan2-training-runs/00016-mnist_stylegan2_blur_noise_maxSeverity3_proba50-cond-auto4/network-snapshot-008467.pkl \
--stylegan2_implementation nvidia
--workers 8 \
--batch_size 8 \
--test_batch_size 4 \
--test_workers 4 
```