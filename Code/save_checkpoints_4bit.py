import torch
import random
import numpy as np
from itertools import product
from pathlib import Path
import tqdm.auto as tqdm
from transformer_lens import HookedTransformer, HookedTransformerConfig

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Generate XOR dataset
def generate_xor_dataset(bit_length=1, n_samples=None, device='cpu', seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    if n_samples is None:
        all_pairs = list(product([0, 1], repeat=2 * bit_length))
        inputs = torch.tensor(all_pairs, dtype=torch.long, device=device)
    else:
        inputs = torch.randint(0, 2, size=(n_samples, 2 * bit_length), dtype=torch.long, device=device)

    a = inputs[:, :bit_length]
    b = inputs[:, bit_length:]
    targets = torch.bitwise_xor(a, b)
    return inputs, targets

# Loss function
def loss_fn(logits, labels):
    if len(logits.shape) == 3:
        logits = logits[:, 4:8, :]
    logits = logits.to(torch.float64)
    log_probs = logits.log_softmax(dim=-1)
    correct_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return -correct_log_probs.mean()

def main():
    # Checkpoint directory
    model_dir = Path("4_bit/Checkpoints")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Data
    seed = 598
    frac_train = 0.115
    n_samples = 256
    bit_length = 4
    dataset, labels = generate_xor_dataset(bit_length=bit_length, device=device, seed=seed)

    torch.manual_seed(seed)
    indices = torch.randperm(n_samples)
    cutoff = int(n_samples * frac_train)
    train_indices = indices[:cutoff]
    test_indices = indices[cutoff:]

    train_data = dataset[train_indices]
    train_labels = labels[train_indices]
    test_data = dataset[test_indices]
    test_labels = labels[test_indices]

    # Model
    cfg = HookedTransformerConfig(
        n_layers=1,
        n_heads=2,
        d_model=64,
        d_head=32,
        d_mlp=256,
        act_fn="relu",
        normalization_type=None,
        d_vocab=2,
        d_vocab_out=2,
        n_ctx=8,
        init_weights=True,
        device=device,
        seed=999,
    )
    model = HookedTransformer(cfg)

    # Disable biases
    for name, param in model.named_parameters():
        if "b_" in name:
            param.requires_grad = False

    # Optimizer
    lr = 1e-3
    wd = 1
    betas = (0.9, 0.98)
    num_epochs = 30000
    checkpoint_every = 100
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas)

    # Training loop
    for epoch in tqdm.tqdm(range(num_epochs)):
        train_logits = model(train_data)
        train_loss = loss_fn(train_logits, train_labels)
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (epoch + 1) % checkpoint_every == 0:
            ckpt_path = model_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"[Checkpoint] Saved: {ckpt_path}")

if __name__ == "__main__":
    main()
