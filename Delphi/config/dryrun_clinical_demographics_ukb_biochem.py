import os
import time

out_dir = "dryrun_clinical_demographics_ukb_biochem"

eval_interval = 20
eval_iters = 5
log_interval = 5
always_save_checkpoint = False

wandb_log = False
wandb_project = "delphi"
wandb_run_name = "dryrun_clinical_demographics_ukb_biochem_" + str(time.time())

dataset = "ukb_amk125_clinical_demographics_ukb_biochem"

batch_size = 16
block_size = 64
data_fraction = 0.001

n_layer = 2
n_head = 2
n_embd = 64
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
max_iters = 40
lr_decay_iters = 40
min_lr = 2e-4
beta2 = 0.99
warmup_iters = 5

t_min = 0.1
token_dropout = 0.0
no_event_token_rate = 5
