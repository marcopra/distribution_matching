# python powr.py env=four_rooms5_1 p_path="/home/mprattico/distribution_matching/models/four_rooms5/policy_operator.npy" wandb.use_wandb=true name="powr_ours" wandb.wandb_tag="ours"
# python powr.py env=four_rooms5_1 name="powr_baseline" wandb.wandb_tag="baseline" wandb.use_wandb=true
python powr.py env=four_rooms5_1 p_path="/home/mprattico/distribution_matching/models/rnd/four_rooms5/snapshot_50100.pt" wandb.use_wandb=true name="powr_rnd" wandb.wandb_tag="rnd"

python powr.py env=four_rooms5_0 name="powr_baseline" wandb.wandb_tag="baseline" wandb.use_wandb=true
python powr.py env=four_rooms5_0 p_path="/home/mprattico/distribution_matching/models/four_rooms5/policy_operator.npy" wandb.use_wandb=true name="powr_ours" wandb.wandb_tag="ours" 
python powr.py env=four_rooms5_0 p_path="/home/mprattico/distribution_matching/models/rnd/four_rooms5/snapshot_50100.pt" wandb.use_wandb=true name="powr_rnd" wandb.wandb_tag="rnd" 

python powr.py env=four_rooms5_2 name="powr_baseline" wandb.wandb_tag="baseline" wandb.use_wandb=true
python powr.py env=four_rooms5_2 p_path="/home/mprattico/distribution_matching/models/four_rooms5/policy_operator.npy" wandb.use_wandb=true name="powr_ours" wandb.wandb_tag="ours"
python powr.py env=four_rooms5_2 p_path="/home/mprattico/distribution_matching/models/rnd/four_rooms5/snapshot_50100.pt" wandb.use_wandb=true name="powr_rnd" wandb.wandb_tag="rnd"

python powr.py env=four_rooms5_3 name="powr_baseline" wandb.wandb_tag="baseline" wandb.use_wandb=true
python powr.py env=four_rooms5_3 p_path="/home/mprattico/distribution_matching/models/four_rooms5/policy_operator.npy" wandb.use_wandb=true name="powr_ours" wandb.wandb_tag="ours"
python powr.py env=four_rooms5_3 p_path="/home/mprattico/distribution_matching/models/rnd/four_rooms5/snapshot_50100.pt" wandb.use_wandb=true name="powr_rnd" wandb.wandb_tag="rnd"




# python powr.py env=two_rooms10_0
# python powr.py env=two_rooms10_1
# python powr.py env=two_rooms10_2

# python powr.py env=two_rooms10_0 p_path="/home/mprattico/distribution_matching/models/two_rooms10/policy_operator.npy"
# python powr.py env=two_rooms10_1 p_path="/home/mprattico/distribution_matching/models/two_rooms10/policy_operator.npy"
# python powr.py env=two_rooms10_2 p_path="/home/mprattico/distribution_matching/models/two_rooms10/policy_operator.npy"