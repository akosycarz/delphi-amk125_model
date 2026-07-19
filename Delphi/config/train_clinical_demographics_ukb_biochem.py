import os
import time

out_dir = "out_clinical_demographics_ukb_biochem"

eval_interval = 250
eval_iters = 25
log_interval = 25
always_save_checkpoint = False

wandb_log = False
wandb_project = "delphi"
wandb_run_name = "clinical_demographics_ukb_biochem_" + str(time.time())

dataset = "ukb_amk125_clinical_demographics_ukb_biochem"

batch_size = 64
block_size = 128
data_fraction = 1.0

n_layer = 6
n_head = 6
n_embd = 120
dropout = 0.1
weight_decay = 2e-1
bias = False

config_values_path = os.path.join("data", dataset, "config_values.py")
if not os.path.exists(config_values_path):
    raise FileNotFoundError(
        f"Missing {config_values_path}. "
        "Run scripts/delphi_preprocess.R first."
    )
exec(open(config_values_path).read())

learning_rate = 2e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 2e-4
beta2 = 0.99
warmup_iters = 500

t_min = 0.1
token_dropout = 0.0
no_event_token_rate = 5
