# config.py
vocab_size = 32000
max_seq_len = 128
d_model = 256
n_heads = 4
n_layers = 4
d_ff = 1024
dropout = 0.1

batch_size = 16
lr = 3e-4
weight_decay = 0.01
epochs = 2
warmup_steps = 500
grad_accum_steps = 1
device = "cuda"