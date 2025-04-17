import os
import sys
import uuid
import time
import copy
import glob
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import argparse
import itertools
import tiktoken
import json
import datetime
import pickle
import shutil
import csv
import random # Import random for potential future use, though not strictly needed for torch seeding
import numpy as np # Import numpy for potential future use, set random seed now not to forget to set it later

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.attention.flex_attention import BlockMask, flex_attention, create_block_mask
#torch._inductor.config.max_autotune_gemm_backends = ["ATEN"]


# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__(in_features, out_features, bias=False)
        self.use_fp8 = use_fp8
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

    def reset_parameters(self) -> None:
        std = 0.5 * (self.in_features ** -0.5) # 0.5 is a bit better than the default 1/sqrt(3)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor):
        if self.use_fp8 and self.training:
            _x = x.flatten(0, -2)
            out: Tensor = torch.ops.nanogpt.mm(_x, self.weight, x_s=self.x_s, w_s=self.w_s, grad_s=self.grad_s)[0]
            return out.reshape(*x.shape[:-1], -1)
        else:
            return F.linear(x, self.weight.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        # Ensure we don't exceed the dimension size
        dim_quarter = max(1, dim // 4)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim_quarter, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim_quarter)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq) # outer product
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        # Handle case where the number of dimensions is smaller
        dim_half = x_BTHD.size(-1) // 2
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos[..., :dim_half] + x2 * sin[..., :dim_half]
        y2 = x1 * (-sin[..., :dim_half]) + x2 * cos[..., :dim_half]
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim=None):
        super().__init__()
        # Calculate head_dim based on model dimensions and num_heads
        self.num_heads = num_heads
        # If head_dim not specified, calculate it based on the model dimension
        if head_dim is None:
            head_dim = dim // num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std # improved init scale by @YouJiacheng
        # merged QKV weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        self.qkv_w = nn.Parameter(torch.empty(3, hdim, dim).uniform_(-bound, bound))
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(head_dim, max_seq_len)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977

    def forward(self, x: Tensor, ve: Tensor | None, block_mask: BlockMask):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        q, k, v = F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k) # QK norm @Grad62304977
        q, k = self.rotary(q), self.rotary(k)
        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v) # @KoszarskyB & @Grad62304977
        else: # skip mid-layers token value embeddings by @YouJiacheng
            v = self.lambdas[0] * v
        # scale the attention logits by given constant, instead of the default head_dim**-0.5, by @leloykun
        # inspired by learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask, scale=0.12).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 4):
        super().__init__()
        hdim = int(mlp_ratio * dim)
        self.c_fc = CastedLinear(dim, hdim)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977

    def forward(self, x: Tensor):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int, max_seq_len: int, layer_idx: int):
        super().__init__()
        # skip attention of blocks.7 (the 8th layer) by @YouJiacheng
        # Adjusted for smaller models - only skip if we have enough layers
        skip_attn = (layer_idx == 7) and (dim > 512)  # Only skip in larger models
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len) if not skip_attn else None
        self.mlp = MLP(dim, mlp_ratio)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, block_mask: BlockMask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), ve, block_mask)
        x = x + self.mlp(norm(x))
        return x

# -----------------------------------------------------------------------------
# The main model

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int, mlp_ratio: int):
        super().__init__()
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len
        self.embed = nn.Embedding(vocab_size, model_dim)
        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        # value embedding code simplification inspired by @ragulpr https://github.com/KellerJordan/modded-nanogpt/pull/78
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, mlp_ratio, max_seq_len, i) for i in range(num_layers)])
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested to me by @Grad62304977. this originates from Karpathy's experiments.
        self.lm_head = CastedLinear(model_dim, next_multiple_of_n(vocab_size, n=128),
                                    use_fp8=True, x_s=(model_dim**0.5)/448, w_s=24/448, grad_s=1/448)
        self.lm_head.weight.detach().zero_() # @Grad62304977
        # Add learnable skip connection weights for decoder layers
        assert num_layers % 2 == 0, f"Number of layers ({num_layers}) must be even for skip connections"
        self.skip_weights = nn.Parameter(torch.ones(num_layers//2))

    def forward(self, 
                input_seq: Tensor, # shape (B*N)
                target_seq: Tensor = None, # (B*N)
                ):
        # sliding_window_num_blocks is tensor of shape (w) dtype int32 were w is number of blocks flexattention will use
        assert input_seq.ndim == 1

        ve = [value_embed(input_seq) for value_embed in self.value_embeds] # each (B*N, D)
        # Adjust token value embeddings structure for fewer layers
        ve = [ve[0], ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
        assert len(ve) == len(self.blocks)

        # creating flex-attentio mask
        docs = (input_seq == 50256).cumsum(0)
        def doc_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask
        # Because the sparsity pattern is independent of batch and heads, we'll set them to None (which broadcasts them) 
        block_mask = create_block_mask(doc_causal, B=None, H=None, Q_LEN=len(input_seq), KV_LEN=len(input_seq))

        x = x0 = norm(self.embed(input_seq)[None]) # use of norm here by @Grad62304977

        # U-net design by @brendanh0gan
        skip_connections = []
        n = len(self.skip_weights)
        for i in range(len(self.blocks)):
            if i >= n:
                x = x + self.skip_weights[i - n] * skip_connections.pop()
            x = self.blocks[i](x, ve[i], x0, block_mask)
            if i < n:
                skip_connections.append(x)

        x = norm(x)
        logits = self.lm_head(x).float()
        # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15, @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1)
        logits = 30 * torch.sigmoid(logits / (7.5 * x.size(-1)**0.5))

        if target_seq is None:
            return logits
        else:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq, 
                                  reduction='sum' if self.training else 'mean')

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        assert idx.ndim == 1
        def cdiv(m, n):
            return (m + (n - 1)) // n
        seq_len = idx.size(0)
        if seq_len % 128 != 0:
            pad_ct = cdiv(seq_len, 128) * 128 - seq_len
            idx = torch.cat((idx, torch.zeros(pad_ct, dtype=idx.dtype, device=idx.device)), dim=0)
        
        self.eval()  # Ensure model is in evaluation mode
        for _ in range(max_new_tokens):
            # Forward pass to get logits
            logits = self(idx[-self.max_seq_len:] if idx.size(0) > self.max_seq_len else idx)
            # Focus on the last token's prediction
            logits = logits[0, min(seq_len, self.max_seq_len) - 1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[-1]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx[min(seq_len, self.max_seq_len)] = idx_next

            # iterate sequence count and account for any time we surpass flex-attention's block size
            seq_len += 1
            if (seq_len - 1) % 128 == 0:
                pad_ct = cdiv(seq_len, 128) * 128 - seq_len
                idx = torch.cat((idx, [0] * pad_ct), dim=0)

        return idx[:seq_len]

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, rank: int, world_size: int, print_stats=True):
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    if not files:
        raise ValueError(f"No files found matching pattern: {filename_pattern}")
    
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    
    # Calculate total tokens across all shards
    total_tokens = 0
    tokens_per_file = []
    for file in files:
        header = torch.from_file(str(file), False, 256, dtype=torch.int32)
        file_tokens = int(header[2])
        total_tokens += file_tokens
        tokens_per_file.append(file_tokens)
    
    # Calculate how many tokens we need for training
    tokens_needed = args.train_steps * batch_size
    
    # Determine if we need to cycle and calculate epochs
    will_cycle = total_tokens < tokens_needed
    epochs = tokens_needed / total_tokens if total_tokens > 0 else 0
    
    if rank == 0 and print_stats:
        print0(f"Total tokens across {len(files)} shard(s): {total_tokens:,}", console=True)
        print0(f"Tokens needed for {args.train_steps} iterations: {tokens_needed:,}", console=True)
        print0(f"Training will use approximately {epochs:.2f} epochs over the data", console=True)
    
    file_iter = itertools.cycle(files) if will_cycle else iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True) # no sync on host side;
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True) # H2D in another stream isn't helpful.
        pos += batch_size
        yield inputs, targets

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    """
    default values are set to fit on a GPU w/ 8GB of VRAM, but are not necessarily optimal
    """
    model_name = "ModdedGPT"
    # data
    train_files = "data/fineweb*_train_*.bin" # input .bin to train on
    val_files = "data/fineweb*_val_*.bin" # input .bin to eval validation loss on
    val_tokens = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    train_seq_len = 8*1024 # FlexAttention sequence length
    val_seq_len = 12*1024 # FlexAttention sequence length for validation
    # optimization
    train_steps = 20#_000 # number of training steps to run
    grad_acc_steps = 1 # number of gradient accumulation steps per training step
    cooldown_frac = 0.4 # fraction of training spent cooling down the learning rate
    # architecture
    tokenizer = "gpt4regex_v50256_n30000000.pkl"#134217728.pkl" # any .pkl file in tokenizers/
    vocab_size = 50257 # should be the tokenizer's size plus any special tokens defined in this script
    # model size - new parameters for GPUs w/ at least 8GB VRAM during testing
    num_layers = 12  # 124m param model should be 12
    num_heads = 6   # 124m param model should be 6
    model_dim = 768  # must be divisible by num_heads
    head_dim = None  # if None, will be set to model_dim // num_heads
    mlp_ratio = 4  # 124m param model should be 4
    # memory optimization 
    hopper = False # Set to True on H100s (and newer?) for improved performance, False on older
    # evaluation and logging
    val_loss_every = 100 # every how many steps to evaluate val loss? 0 for only at the end
    save_model = False
    # reproducibility
    seed: int | None = None # Optional random seed for initialization control

    def __post_init__(self):
        # Validate and set derived param eters
        if self.head_dim is None:
            self.head_dim = self.model_dim // self.num_heads
        assert self.head_dim in [2 ** i for i in range(1, 10)], f"head_dim must be a power of 2, got {self.head_dim}"
        assert self.mlp_ratio > 0, f"mlp_ratio must be positive, got {self.mlp_ratio}"
        assert self.train_seq_len % 128 == 0, f"train_seq_len must be multiple of 128, got {self.train_seq_len}"
        assert self.val_seq_len % 128 == 0, f"val_seq_len must be multiple of 128, got {self.val_seq_len}"
        assert self.num_layers >= 2, f"num_layers must be greater than or equal to 2 because of attention mask structure, got {self.num_layers}"
        assert self.num_layers >= 6, f"num_layers must be greater than or equal to 2 because of value embeddings structure, got {self.num_layers}"
        assert self.grad_acc_steps >= 1, f"grad_acc steps must be int >= 1"

    @classmethod
    def from_args(cls):
        """Create Hyperparameters from command-line arguments."""
        parser = argparse.ArgumentParser(description="Train a GPT model with customizable hyperparameters")
        
        # Data arguments
        parser.add_argument('--train_files', type=str, help='Pattern for training data files')
        parser.add_argument('--val_files', type=str, help='Pattern for validation data files')
        parser.add_argument('--val_tokens', type=int, help='Number of tokens for validation')
        parser.add_argument('--train_seq_len', type=int, help='Training sequence length')
        parser.add_argument('--val_seq_len', type=int, help='Validation sequence length')
        
        # Optimization arguments
        parser.add_argument('--train_steps', type=int, help='Number of training iterations')
        parser.add_argument('--grad_acc_steps', type=int, help='Number of gradient accumulation steps per training iteration')
        parser.add_argument('--cooldown_frac', type=float, help='Fraction of training for learning rate cooldown')
        
        # Architecture arguments
        parser.add_argument('--tokenizer', type=str, help='Tokenizer file name in tokenizers/ directory')
        parser.add_argument('--vocab_size', type=int, help='Vocabulary size')
        parser.add_argument('--num_layers', type=int, help='Number of transformer layers')
        parser.add_argument('--num_heads', type=int, help='Number of attention heads')
        parser.add_argument('--model_dim', type=int, help='Model embedding dimension')
        parser.add_argument('--head_dim', type=int, help='Dimension per attention head')
        parser.add_argument('--mlp_ratio', type=int, help='MLP hidden dim ratio')
        
        # Other options
        parser.add_argument('--hopper', action='store_true', help='Set to true on H100s (and newer?) for improved performance')
        parser.add_argument('--not_hopper', action='store_false', dest='hopper', help='Use if not on a hopper GPU')
        parser.add_argument('--val_loss_every', type=int, help='Evaluate validation loss every N steps')
        parser.add_argument('--save_model', action='store_true', help='Save model checkpoints')
        parser.add_argument('--no_save_model', action='store_false', dest='save_model', help='Disable model checkpoints')
        parser.add_argument('--model_name', type=str, help='Model name for logging')
        parser.add_argument('--seed', type=int, help='Random seed for initialization control')
        
        args = parser.parse_args()
        
        # Create a base instance with defaults
        instance = cls()
        
        # Update instance with command-line arguments that were provided
        for key, value in vars(args).items():
            if value is not None:  # Only update if argument was provided
                setattr(instance, key, value)
        
        # Run post_init validations after applying CLI arguments
        instance.__post_init__()
        
        return instance, args

# Parse arguments and create Hyperparameters instance
args, cli_args = Hyperparameters.from_args()

# Check if environment variables are set by torchrun, otherwise default to single GPU
if "RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ:
    # Multi-GPU setup with torchrun
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
else:
    # Single GPU setup
    rank = 0
    world_size = 1
    local_rank = 0
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

print(f"Running with {world_size} GPU{'s' if world_size > 1 else ''}")
assert torch.cuda.is_available()
device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)

# Initialize distributed process group if using multiple GPUs
if world_size > 1:
    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()
master_process = (rank == 0)  # this process will do logging, checkpointing etc.

#################################################
#########           logging           ###########
#################################################

def print0(s, console=False):
    # Ensure print0 works even if not master_process (but does nothing)
    if master_process and logfile:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)

# begin logging
logfile = None
experiment_dir_path = None # Define experiment_dir_path outside the if block
metrics_csv_path = None # Define metrics_csv_path
if master_process:
    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 1. Create the experiment directory name
    experiment_dir_name = (f"{start_time}_{args.model_name}")
    # 2. Create the experiment directory path
    experiment_dir_path = Path("experiments") / experiment_dir_name
    os.makedirs(experiment_dir_path, exist_ok=True)
    # 3. Set the logfile path inside the experiment directory
    logfile = experiment_dir_path / "training_log.txt"
    # 4. Set the metrics CSV file path
    metrics_csv_path = experiment_dir_path / "metrics.csv"
    print0(f"Logging to: {logfile}", console=True)
    print0(f"Metrics CSV: {metrics_csv_path}", console=True)
    # 5. Initialize metrics CSV file with headers
    with open(metrics_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["step", "type", "loss", "cumulative_time_ms", "step_avg_ms"])
    # 6. Log any command-line arguments that were provided (overriding defaults)
    cli_arg_dict = {k: v for k, v in vars(cli_args).items() if v is not None}
    if cli_arg_dict:
        print0("Command-line arguments overriding defaults:", console=True)
        for key, value in cli_arg_dict.items():
            print0(f"  --{key} = {value}", console=True)
        print0("="*100, console=True)

    print0("Copying relevant files to experiment directory...")
    files_to_copy = ["requirements.txt", sys.argv[0], "download_hellaswag.py", "download_fineweb.py"]
    for file_path_str in files_to_copy:
        file_path = Path(file_path_str)
        if file_path.exists():
            try:
                # Use Path object methods for cleaner path manipulation
                target_path = experiment_dir_path / f"{file_path.stem}.txt"
                shutil.copy(str(file_path), str(target_path))
                print0(f"- Copied {file_path} to {target_path}")
            except Exception as e:
                print0(f"- Failed to copy {file_path}: {e}")
        else:
            print0(f"- File not found, skipping: {file_path}")

    # Handle tokenizer separately as it's a .pkl file
    tokenizer_path = Path(f"data/{args.tokenizer}")
    if tokenizer_path.exists():
        try:
            with open(tokenizer_path, 'rb') as f:
                tokenizer_config = pickle.load(f)
            # Save the config as a pretty-printed text file
            tokenizer_log_path = experiment_dir_path / f"{tokenizer_path.stem}_config.txt"
            import pprint
            tokenizer_str = pprint.pformat(tokenizer_config)
            with open(tokenizer_log_path, "w") as f:
                f.write(f"Tokenizer Config ({args.tokenizer}):\n")
                f.write("="*100 + "\n")
                f.write(tokenizer_str)
            print0(f"- Saved tokenizer config to {tokenizer_log_path}")
            del tokenizer_config # Free up memory
        except Exception as e:
            print0(f"- Error processing tokenizer {tokenizer_path}: {e}")
    else:
        print0(f"- Tokenizer file not found: {tokenizer_path}")

    print0("="*100)

# log information about the hardware/software environment this is running on
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
def nvidia_smi():
    import subprocess  # avoid top level import
    return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
print0(nvidia_smi())
print0("="*100)

#################################################
#########      Seed for Reproducibility     #####
#################################################

# Set the seed *before* initializing the model or optimizer
if args.seed is not None:
    print0(f"Setting random seed to {args.seed} for model initialization", console=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed) # Important for multi-GPU consistency
        # The following might be needed for full determinism, but can impact performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

########################################
#    Construct model and optimizer     #
########################################

model: nn.Module = GPT(vocab_size=args.vocab_size, 
                       num_layers=args.num_layers,
                       num_heads=args.num_heads, 
                       model_dim=args.model_dim,
                       max_seq_len=max(args.train_seq_len, args.val_seq_len),
                       mlp_ratio=args.mlp_ratio).cuda()
print0(f'{model.get_num_params()} parameters', console=True)
print0(model)

# Set FP8 option based on hyperparameters
model.lm_head.use_fp8 = args.hopper

for m in model.modules():
    if isinstance(m, nn.Embedding):
        m.bfloat16()
if world_size > 1:
    for param in model.parameters():
        dist.broadcast(param.detach(), 0)

# collect the parameters to optimize
hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
embed_params = [p for n, p in model.named_parameters() if "embed" in n]
scalar_params = [p for p in model.parameters() if p.ndim < 2]
head_params = [model.lm_head.weight]

# init the optimizer(s)
adam_params = [dict(params=head_params, lr=0.22), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
# small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
# discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
optimizer1 = torch.optim.Adam(adam_params, betas=(0.8, 0.95), eps=1e-10, fused=True)
optimizers = [optimizer1]
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

# learning rate schedule: stable then decay
def get_lr(step: int):
    x = step / args.train_steps # progress in training
    assert 0 <= x < 1
    if x < 1 - args.cooldown_frac:
        return 1.0
    else:
        w = (1 - x) / args.cooldown_frac
        return w * 1.0 + (1 - w) * 0.1

# Use a more memory-efficient compilation option
model: nn.Module = torch.compile(model, dynamic=False, mode="reduce-overhead")

# Add fallback mode to handle compilation errors
import torch._dynamo
torch._dynamo.config.suppress_errors = True

########################################
#            Warmup kernels            #
########################################

print0("warming up kernels...", console=True)

# Attempt to limit memory fragmentation
if hasattr(torch.cuda, 'memory_stats'):
    print0(f"Initial GPU memory: {torch.cuda.memory_allocated() // (1024 * 1024)} MB")

# Warmup the training kernels, then re-initialize the state so we aren't cheating
warmup_steps = 10
initial_state = dict(model=copy.deepcopy(model.state_dict()),
                     optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers]) # save the initial state
for _ in range(warmup_steps):
    loss = torch.tensor([0.], device="cuda")
    for _ in range(args.grad_acc_steps):
        inputs = targets = torch.randint(0, args.vocab_size, size=(args.train_seq_len,), device="cuda", dtype=torch.int64)
        #torch.compiler.cudagraph_mark_step_begin()
            # TODO why does un-cpmmenting this^ line throw an error here in the warmup but not down in training?
        step_loss = model(inputs.to(torch.int32), targets)
        loss += step_loss / args.grad_acc_steps
    loss.backward()
    if world_size > 1:
        for param in model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
model.load_state_dict(initial_state["model"])
for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
    opt.load_state_dict(opt_state)
del initial_state # TODO optionally save initial state of model jic someone wants to test different seeds

if hasattr(torch.cuda, 'memory_stats'):
    print0(f"After warmup GPU memory: {torch.cuda.memory_allocated() // (1024 * 1024)} MB")

print0("kernels are toasty", console=True)

########################################
#        Training and validation       #
########################################

train_loader = distributed_data_generator(args.train_files, world_size * args.train_seq_len, rank, world_size)

training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()
# begin training
for step in range(args.train_steps + 1):
    last_step = (step == args.train_steps)

    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        # stop the clock
        torch.cuda.synchronize()
        # Note: training_time_ms accumulates *only* the time spent in the training loop
        # It does not include time spent in validation or other operations outside the loop
        training_time_ms += 1000 * (time.perf_counter() - t0)
        
        model.eval()
        
        # Ensure we validate on enough tokens while keeping memory usage reasonable
        val_batch_size = world_size * args.val_seq_len
        val_steps = max(1, min(16, args.val_tokens // val_batch_size))
        val_tokens_used = val_batch_size * val_steps
        print0(f"Validating on {val_tokens_used} tokens ({val_steps} steps with {val_batch_size} batch size)", console=True)
        
        val_loader = distributed_data_generator(args.val_files, val_batch_size, rank, world_size, print_stats=False)
        val_loss = 0
        with torch.no_grad():
            for i in range(val_steps):
                inputs, targets = next(val_loader)
                # Check if inputs exceed sequence length
                if inputs.size(0) > args.val_seq_len:
                    inputs = inputs[:args.val_seq_len]
                    targets = targets[:args.val_seq_len]
                val_loss += model(inputs, targets)
        val_loss /= val_steps
        del val_loader
        if world_size > 1:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        
        # Calculate average time per step up to this point
        step_avg_ms = training_time_ms / max(step, 1) 
        print0(f"step:{step}/{args.train_steps} val_loss:{val_loss:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{step_avg_ms:.2f}ms", console=True)
        
        # Log validation metrics to CSV
        if master_process and metrics_csv_path:
            with open(metrics_csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Use .item() to get float from tensor for val_loss
                writer.writerow([step, 
                    "val", f"{val_loss.item():.4f}", 
                    f"{training_time_ms:.0f}", 
                    f"{step_avg_ms:.2f}"])

        if last_step: # inside validation section to avoid the if check every training iteration
            # 5. Save model checkpoint inside the experiment directory
            if master_process and args.save_model and experiment_dir_path:
                log = dict(step=step, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
                # Ensure experiment_dir_path exists (though it should from earlier)
                os.makedirs(experiment_dir_path, exist_ok=True)
                save_path = experiment_dir_path / f"state_step{step:06d}.pt"
                torch.save(log, str(save_path))
                print0(f"Saved checkpoint to {save_path}", console=True)
            # the last step only has the validation loop, so break to avoid training
            break
        
        model.train()
        # start the clock again for the next training segment
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    # --------------- TRAINING SECTION -----------------
    loss = torch.tensor([0.], device="cuda")
    for _ in range(args.grad_acc_steps):
        inputs, targets = next(train_loader)
        # Check if inputs exceed sequence length - can happen if the dataset has different sized examples
        if inputs.size(0) > args.train_seq_len:
            inputs = inputs[:args.train_seq_len]
            targets = targets[:args.train_seq_len]
        torch.compiler.cudagraph_mark_step_begin()
        step_loss = model(inputs, targets)
        loss += step_loss / args.grad_acc_steps
    loss.backward()
        
    if world_size > 1:
        for param in model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
    # set optimization hyperparameters
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * get_lr(step)
    # step the optimizers
    for opt in optimizers:
        opt.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
        
    # calculate *approximate* cumulative time and step average for logging during training
    # Note: This is approximate because it includes the time for the current step's forward/backward pass
    # The more precise time is recorded just before validation
    if master_process:
        torch.cuda.synchronize() # Ensure accurate timing up to this point
        # Calculate time elapsed since the end of the last validation phase
        current_segment_duration_ms = 1000 * (time.perf_counter() - t0) 
        # Calculate the *true* approximate cumulative time
        approx_cumulative_time_ms = training_time_ms + current_segment_duration_ms
        approx_step_avg_ms = approx_cumulative_time_ms / (step + 1)
        print0(f"step:{step+1}/{args.train_steps} "
                f"train_time:{approx_cumulative_time_ms:.0f}ms "
                f"step_avg:{approx_step_avg_ms:.2f}ms", console=True)
        
        # Log training step timing to CSV
        if metrics_csv_path:
             with open(metrics_csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Loss is not typically calculated per training step here, add loss logging if needed
                writer.writerow([step + 1, "train", "", f"{approx_cumulative_time_ms:.0f}", f"{approx_step_avg_ms:.2f}"])


print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)

########################################
#        HellaSwag Evaluation         #
########################################

def render_hellaswag_example(example, enc):
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    # gather up all the tokens
    ctx_tokens = enc.encode(ctx)
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = enc.encode(" " + end)  # NOTE: prepending " " because GPT-2 based tokenizer
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.int32)
    mask = torch.zeros((4, max_len), dtype=torch.int32)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return tokens, mask, label

def iterate_hellaswag_examples(data_path, limit=1014):
    """Iterate through HellaSwag examples, with optional limit"""
    with open(data_path, "r") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            example = json.loads(line)
            yield example

@torch.no_grad()
def evaluate_hellaswag(model, data_path, limit=1014):
    """Evaluate model on HellaSwag in a distributed way using modulo distribution"""
    assert limit <= 1014, f'there are only 1014 questions in the benchmark, but got limit={limit}'
    torch._dynamo.config.disable = True
    tokenizer_config = pickle.load(open(f"tokenizers/{args.tokenizer}", 'rb'))
    enc = tiktoken.Encoding(
        name=args.tokenizer[:-4], # :-4 to remove the .pkl
        pat_str=tokenizer_config['pat_str'],
        mergeable_ranks=tokenizer_config['mergeable_ranks'],
        special_tokens={
            "<|endoftext|>": len(tokenizer_config['mergeable_ranks']),
        }
    )
    model.eval()
    
    # Local counters
    local_correct_norm = 0
    local_correct = 0
    local_total = 0
    
    # Process examples that belong to this GPU (based on index % world_size)
    for i, example in enumerate(iterate_hellaswag_examples(data_path, limit)):
        # Skip examples that don't belong to this GPU
        if i % world_size != rank:
            continue

        local_total += 1
        tokens, mask, label = render_hellaswag_example(example, enc)
        tokens = tokens.to(device="cuda")
        mask = mask.to(device="cuda")

        # Process each candidate one at a time
        losses = []
        normalized_losses = []
        
        for j in range(4):  # 4 candidates per example
            # Get token sequence for this candidate
            seq = tokens[j]
            seq_mask = mask[j]
            
            # Only process up to valid tokens (not padding)
            valid_len = (seq > 0).sum().item()
            if valid_len == 0:
                continue
                
            valid_seq = seq[:valid_len]
            
            # Pad sequence to multiple of 128 for FlexAttention
            if valid_len % 128 != 0:
                # Calculate padding needed
                def cdiv(m, n):
                    return (m + (n - 1)) // n
                pad_ct = cdiv(valid_len, 128) * 128 - valid_len
                # Add padding
                valid_seq = torch.cat((valid_seq, 
                                      torch.zeros(pad_ct, dtype=valid_seq.dtype, device=valid_seq.device)), 
                                      dim=0)
            
            # Get logits from our model
            logits = model(valid_seq)
            if isinstance(logits, torch.Tensor):
                logits = logits[0]  # Our model returns [B, T, V] but B=1
            
            # We only care about the original non-padded part
            logits = logits[:valid_len]
            
            # Evaluate the autoregressive loss
            shift_logits = logits[:-1, :]
            shift_tokens = seq[1:valid_len].to(torch.int64)  # Target needs to be int64
            shift_mask = seq_mask[1:valid_len]  # Shift mask to align with shifted tokens
            
            # Calculate loss for each position
            losses_per_token = F.cross_entropy(
                shift_logits, shift_tokens, reduction='none'
            )
            
            # Apply mask to focus on completion region
            masked_losses = losses_per_token * shift_mask
            
            # Calculate total and normalized loss
            total_loss = masked_losses.sum()
            completion_token_count = shift_mask.sum()
            normalized_loss = total_loss / completion_token_count if completion_token_count > 0 else float('inf')
            
            losses.append(total_loss.item())
            normalized_losses.append(normalized_loss.item())
        
        # Get predictions and update counters
        pred = torch.tensor(losses).argmin().item()
        pred_norm = torch.tensor(normalized_losses).argmin().item()
        
        local_correct += int(pred == label)
        local_correct_norm += int(pred_norm == label)
    
    # Gather results from all processes
    correct_tensor = torch.tensor([local_correct], dtype=torch.float32, device="cuda")
    correct_norm_tensor = torch.tensor([local_correct_norm], dtype=torch.float32, device="cuda")
    total_tensor = torch.tensor([local_total], dtype=torch.float32, device="cuda")   
    
    # Handle distributed reduction
    if world_size > 1:
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_norm_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    
    # Calculate final metrics on master process
    if master_process:
        num_correct = int(correct_tensor.item())
        num_correct_norm = int(correct_norm_tensor.item())
        num_total = int(total_tensor.item())
        
        # Calculate metrics and print results
        accuracy = num_correct / num_total if num_total > 0 else 0
        accuracy_norm = num_correct_norm / num_total if num_total > 0 else 0

        # Calculate 95% confidence intervals using Wilson score interval
        # This is more robust than normal approximation, especially for small sample sizes or extreme probabilities
        z = 1.96  # 95% confidence
        
        def wilson_conf_interval(correct, total):
            """Calculate Wilson score interval for a binary proportion"""
            if total == 0:
                return (0, 0)
            
            p = correct / total
            denominator = 1 + z**2 / total
            centre_adjusted_p = (p + z**2 / (2 * total)) / denominator
            adjusted_interval = z * ((p * (1 - p) / total + z**2 / (4 * total**2)) ** 0.5) / denominator
            
            lower = max(0, centre_adjusted_p - adjusted_interval)
            upper = min(1, centre_adjusted_p + adjusted_interval)
            
            return (lower, upper)
        
        # Get confidence intervals
        ci = wilson_conf_interval(num_correct, num_total)
        ci_norm = wilson_conf_interval(num_correct_norm, num_total)
        
        print0(f"HellaSwag evaluation complete - {num_total} examples", console=True)
        print0(f"Accuracy: {num_correct}/{num_total}={accuracy:.4f} "
                f"[95% CI: {ci[0]:.3f}-{ci[1]:.3f}]", console=True)
        print0(f"Normalized accuracy: {num_correct_norm}/{num_total}={accuracy_norm:.4f} "
                f"[95% CI: {ci_norm[0]:.3f}-{ci_norm[1]:.3f}]", console=True)

# After training and sample generations, evaluate on HellaSwag
hellaswag_path = "./data/hellaswag_val.jsonl" 
# Check if the HellaSwag data file exists
if os.path.exists(hellaswag_path):
    print0(f"Found HellaSwag dataset at {hellaswag_path}", console=True)
    evaluate_hellaswag(model, hellaswag_path, limit=1014) # 1014 is largest possible
else:
    print0(f"HellaSwag dataset not found at {hellaswag_path}, skipping evaluation.", console=True)

if world_size > 1:
    dist.destroy_process_group()

########################################
#        FINAL OUTPUT EXAMPLES         #
########################################

def sample_from_model(model, prompt, max_new_tokens=100, temperature=0.8, top_k=200):
    """Generate text samples from the model given a prompt."""
    tokenizer_config = pickle.load(open(f"tokenizers/{args.tokenizer}", 'rb'))
    enc = tiktoken.Encoding(
        name=args.tokenizer[:-4], # :-4 to remove the .pkl
        pat_str=tokenizer_config['pat_str'],
        mergeable_ranks=tokenizer_config['mergeable_ranks'],
        special_tokens={
            "<|endoftext|>": len(tokenizer_config['mergeable_ranks']),
        }
    )
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    
    # Encode the prompt
    input_ids = encode(prompt)
    x = torch.tensor(input_ids, dtype=torch.int32, device="cuda")

    # Generate
    model.eval()
    with torch.no_grad():
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    
    # Decode and return
    return decode(y.tolist())

# Then at the end of training:
if master_process: 
    print0("-"*10 + " EXAMPLE MODEL GENERATIONS AFTER TRAINING " + "-"*10)
    prompts = [
        "Once upon a time,",
        "The meaning of life is",
        "In the year 2026,",
        "I'm a Large Language Model (LLM), which means"
    ]
    for prompt in prompts:
        continuation = sample_from_model(model, prompt, max_new_tokens=16)
        print0(continuation, console=True)