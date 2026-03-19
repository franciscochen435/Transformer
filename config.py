vocab_size = 32000  # Number of unique tokens in the vocabulary
max_seq_len = 128  # Maximum sequence length
d_model = 256  # Transformer hidden size / embedding dimension
n_heads = 4  # Size of attention heads in multi-head self-attention
n_layers = 6  # Size of Transformer blocks
d_ff = 1024  # Hidden dimension of the FFN
dropout = 0.15  # Dropout probability applied during training

batch_size = 8  # Training batch size
lr = 1e-4  # Learning rate for the optimizer.
weight_decay = 0.01  # weight decay
epochs = 5  # Number of full passes over the training dataset
warmup_steps = 1000  # Warmup steps for the learning-rate schedule
grad_accum_steps = 1  # Gradient accumulation steps to simulate a larger batch
device = "cuda"  # GPU
