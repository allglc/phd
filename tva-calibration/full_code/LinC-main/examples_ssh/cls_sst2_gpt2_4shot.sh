python run_classification.py \
--model="gpt2-xl" \
--dataset="sst2" \
--num_seeds=5 \
--all_shots="4" \
--subsample_test_set=300 \
--epochs=15 \
--lr=0.0003 \
--val_size=100 \
--val_seed=202303022