import os
import math
import inspect

from train import Trainer, Hyperparameters

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
from torch import Tensor, nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

class CausalSelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(model_dim, 3 * model_dim)
        # output projection
        self.c_proj = nn.Linear(model_dim, model_dim)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x: Tensor):

        B, T, C = x.size() # batch size, block size (sequence length), model_dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh * hs = C =768 channels
        qkv = self.c_attn(x)   
        q, k, v = qkv.split(self.model_dim, dim=2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, model_dim: int, mlp_ratio: int = 4):
        super().__init__()
        hdim = int(mlp_ratio * model_dim)
        self.c_fc = nn.Linear(model_dim, hdim)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(hdim, model_dim)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x: Tensor):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, mlp_ratio: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(model_dim, eps=1e-5, dtype=torch.float32)
        self.attn = CausalSelfAttention(model_dim, num_heads)
        self.ln_2 = nn.LayerNorm(model_dim, eps=1e-5, dtype=torch.float32)
        self.mlp = MLP(model_dim, mlp_ratio)

    def forward(self, x: Tensor):
        x = self.ln_1(x)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# -----------------------------------------------------------------------------
# The main model

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, block_size: int, mlp_ratio: int):
        super().__init__()
        self.model_dim = model_dim
        self.block_size = block_size
        self.num_layers = num_layers

        self.transformer = nn.ModuleDict(dict(
            # word token embeddings
            wte = nn.Embedding(vocab_size, model_dim),
            # word positional embeddings
            wpe = nn.Embedding(block_size, model_dim),
            h = nn.ModuleList([Block(model_dim, num_heads, mlp_ratio) for _ in range(num_layers)]),
            ln_f = nn.LayerNorm(model_dim, eps=1e-5, dtype=torch.float32),
        ))
        
        self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        num_layers = self.num_layers # capture from outer scope
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * num_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_seq: Tensor, target_seq: Tensor = None):
        # Determine mode
        is_training = target_seq is not None

        # Normalize input to (B, T)
        if input_seq.ndim == 1:
            total_len = input_seq.size(0)

            if is_training:
                assert total_len % self.block_size == 0, \
                    f"Training input length {total_len} must be divisible by block_size {self.block_size}"
                B = total_len // self.block_size
                T = self.block_size
                input_seq = input_seq.view(B, T)
            else:
                T = total_len
                assert T <= self.block_size, \
                    f"Inference sequence length {T} must be <= block_size {self.block_size}"
                input_seq = input_seq.unsqueeze(0)  # (1, T)
                B = 1

        elif input_seq.ndim == 2:
            B, T = input_seq.shape
            if is_training:
                assert T == self.block_size, \
                    f"Expected training sequence length {self.block_size}, got {T}"
            else:
                assert T <= self.block_size, \
                    f"Inference sequence length {T} must be <= block_size {self.block_size}"
        else:
            raise ValueError(f"Expected 1D or 2D input, got shape {input_seq.shape}")

        # Prepare targets
        if target_seq is not None:
            if target_seq.ndim == 1:
                assert target_seq.numel() == input_seq.numel(), \
                    "target_seq and input_seq must have the same number of tokens"
                target_seq = target_seq.view(B, T)
            elif target_seq.ndim == 2:
                assert target_seq.shape == (B, T), \
                    f"target_seq shape {target_seq.shape} must match input_seq shape {input_seq.shape}"
            else:
                raise ValueError(f"Expected 1D or 2D target, got shape {target_seq.shape}")
            target_seq = target_seq.long()

        # Get embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=input_seq.device)  # (T,)
        pos_emb = self.transformer.wpe(pos).unsqueeze(0)  # (1, T, D)
        tok_emb = self.transformer.wte(input_seq)  # (B, T, D)
        x = tok_emb + pos_emb  # (B, T, D)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x).float()  # (B, T, vocab_size)

        if target_seq is None:
            return logits

        # Flatten for loss
        logits = logits.view(-1, logits.size(-1))       # (B*T, vocab_size)
        targets = target_seq.contiguous().view(-1)       # (B*T,)

        return F.cross_entropy(
            logits, targets,
            reduction='sum' if self.training else 'mean'
        ) 


    def get_num_params(self):
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=fused_available)
        return optimizer

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None):
        """
        Generate tokens from a 1D conditioning sequence `idx` of shape (T,).
        Appends `max_new_tokens` tokens one at a time, feeding outputs back in.
        """
        assert idx.ndim == 1, "idx must be a 1D tensor of token indices"

        self.eval()

        for _ in range(max_new_tokens):
            # Crop input if it exceeds block size
            idx_cond = idx[-self.block_size:]

            # Add batch dimension for model input
            logits = self(idx_cond.unsqueeze(0))  # shape (1, T, vocab_size)
            logits = logits[0, -1, :] / temperature  # shape (vocab_size,)

            probs = F.softmax(logits, dim=-1)

            if top_k is not None:
                tk_probs, tk_indices = torch.topk(probs, top_k)
                ix = torch.multinomial(tk_probs, 1)
                next_token = tk_indices[ix]
            else:
                next_token = torch.multinomial(probs, 1)

            # Append the new token to the sequence
            idx = torch.cat((idx, next_token), dim=0)

        return idx


# -----------------------------------------------------------------------------
# int main

args = Hyperparameters(
    model_name = "NanoGPT",
    # data
    train_files = "data/fineweb*_train_*.bin", # input .bin to train on
    val_files = "data/fineweb*_val_*.bin", # input .bin to eval validation loss on
    train_seq_len = 16*1024, # FlexAttention sequence length
    val_seq_len = 16*1024, # FlexAttention sequence length for validation
    val_steps = 10, # number of steps to run validation for
    train_steps = 20,#_000 # number of training steps to run
    grad_acc_steps = 1, # number of gradient accumulation steps per training step
    cooldown_frac = 0.4,# fraction of training spent cooling down the learning rate
    # architecture
    tokenizer = "gpt2_v50256.pkl",# any .pkl file in tokenizers/
    vocab_size = 50257, # should be the tokenizer's size plus any special tokens
    # model size - parameters set for GPUs w/ 8GB VRAM
    num_layers = 2,  # number of transformer blocks
    num_heads = 6,  # number of attention heads
    model_dim = 384,  # size of model embedding vectors
    head_dim = None,  # size of attention heads; if None, will default to model_dim // num_heads
    mlp_ratio = 4,  # MLP hidden dimension is model_dim * mlp_ratio
    num_val_emb = 2, # number of value embeddings used at initial and final layers
    # memory optimization 
    use_fp8 = False, # experimental; True on H100s (and newer?) should improve performance but seems to use more vram somehow
    # evaluation and logging
    val_loss_every = 100, # every how many steps to evaluate val loss? 0 for only at the end
    save_model = False,
    hellaswag_limit = 10, # 1014 is largest possible
    # reproducibility
    seed = None # Optional random seed for initialization control
)


args, cli_args = args.replace_cli_args(args)

## TODO - Add appropriate assertions
# assert args.mlp_ratio > 0, f"mlp_ratio must be positive, got {args.mlp_ratio}"

########################################
#    Construct model and optimizer     #
########################################
def make_model():   
    block_size = 1024
    model = GPT(vocab_size=args.vocab_size, 
                num_layers=args.num_layers,
                num_heads=args.num_heads, 
                model_dim=args.model_dim,
                block_size=block_size,
                mlp_ratio=args.mlp_ratio).cuda()
    model = model.bfloat16()
    return model

# lr warmup (Karpathy)
def get_lr(it):
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 715
    max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens

    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

def optimizers_step_lr(step, optimizers):
    # set optimization hyperparameters
    for param_group in optimizers[0].param_groups:
        param_group["lr"] = get_lr(step)

def make_optimizers(model):
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type='cuda')
    optimizers = [optimizer]
    return optimizers

# Train
trainer = Trainer(args, cli_args)
trainer.train(make_model, make_optimizers, optimizers_step_lr)

