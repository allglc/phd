python run_classification.py \
--model="gptj" \
--dataset="sst5" \
--num_seeds=5 \
--all_shots="4" \
--subsample_test_set=300 \
--epochs=50 \
--lr=0.00075 \
--val_size=100 \
--val_seed=202303011