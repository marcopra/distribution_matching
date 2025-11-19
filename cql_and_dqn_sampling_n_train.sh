python sampling_and_train_offline.py agent=cql experiment=pop pretrained_path="/home/mprattico/distribution_matching/models/four_rooms5/policy_operator.npy" use_wandb=true seed=0 configs/env=four_rooms5_0
python sampling_and_train_offline.py agent=cql experiment=rnd pretrained_path="/home/mprattico/distribution_matching/models/rnd/four_rooms5/snapshot_50100.pt" use_wandb=true seed=0 configs/env=four_rooms5_0
python sampling_and_train_offline.py agent=cql experiment=baseline use_wandb=true seed=0 configs/env=four_rooms5_0

python sampling_and_train_offline.py agent=doubledqn experiment=pop pretrained_path="/home/mprattico/distribution_matching/models/four_rooms5/policy_operator.npy" use_wandb=true seed=0 configs/env=four_rooms5_0
python sampling_and_train_offline.py agent=doubledqn experiment=rnd pretrained_path="/home/mprattico/distribution_matching/models/rnd/four_rooms5/snapshot_50100.pt" use_wandb=true seed=0 configs/env=four_rooms5_0
python sampling_and_train_offline.py agent=doubledqn experiment=baseline use_wandb=true seed=0 configs/env=four_rooms5_0