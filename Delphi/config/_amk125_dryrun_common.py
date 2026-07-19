exec(open("config/_amk125_common.py").read())

eval_interval = 20
eval_iters = 5
log_interval = 5
batch_size = 16
block_size = 64
n_layer = 2
n_head = 2
n_embd = 64
max_iters = 40
lr_decay_iters = 40
warmup_iters = 5
