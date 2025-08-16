import re, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import imageio.v2 as imageio

import torch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from transformer_lens import HookedTransformer, HookedTransformerConfig

# ---------------- Config ----------------
bit_width   = 7
d_model     = 64
d_mlp       = 256
layer_idx   = 0

# Checkpoints location
checkpoint_dir  = Path(r"7_bit/Checkpoints")
checkpoint_glob = "checkpoint_epoch_*.pth"
epoch_regex     = r"epoch_?(\d+)"

# Output root
out_root = Path(r"7_bit/MLPTransition")
out_root.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Data / Model ----------------
def generate_xor_dataset(bit_width=4, n_samples=256, device="cpu", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    inputs = torch.randint(0, 2, size=(n_samples, 2*bit_width), dtype=torch.long, device=device, generator=g)
    a, b = inputs[:, :bit_width], inputs[:, bit_width:]
    targets = torch.bitwise_xor(a, b)  # [N, bit_width]
    return inputs, targets

def create_model(bit_width=4, d_model=64, d_mlp=256):
    cfg = HookedTransformerConfig(
        n_layers=1, n_heads=2, d_model=d_model, d_head=32, d_mlp=d_mlp,
        act_fn="relu", normalization_type=None,
        d_vocab=2, d_vocab_out=2, n_ctx=2*bit_width,
        init_weights=True, device=device, seed=42,
    )
    return HookedTransformer(cfg)

def sorted_checkpoints(ckpt_dir: Path, pattern: str, regex: str):
    pairs = []
    for p in ckpt_dir.glob(pattern):
        m = re.search(regex, p.stem)
        if m:
            pairs.append((int(m.group(1)), p))
    pairs.sort(key=lambda x: x[0])
    return pairs

# ---------------- Activation collection ----------------
def collect_mlp_post(model, inputs, layer_idx=0, mode="last_bit"):
    """
    mode:
      "last_bit"     -> returns activations of last token only: [N, d_mlp]
      "per_position" -> returns activations of the last bit_width tokens, flattened: [N*bit_width, d_mlp]
    """
    outs = []
    def hook_fn(t, hook=None): outs.append(t.detach().cpu())  # [batch, seq, d_mlp]
    model.add_hook(f"blocks.{layer_idx}.mlp.hook_post", hook_fn)
    with torch.no_grad():
        _ = model(inputs)
    model.reset_hooks()
    x = torch.cat(outs, dim=0)  # [N, seq, d_mlp]
    if mode == "last_bit":
        x = x[:, -1, :]  # [N, d_mlp]
        return x.numpy()
    else:
        x = x[:, -bit_width:, :]       # [N, bit_width, d_mlp]
        x = x.reshape(-1, x.shape[-1]) # [N*bit_width, d_mlp]
        return x.numpy()

# ---------------- Labels that MATCH activations ----------------
def build_labels(targets, mode="last_bit"):
    """
    targets is XOR(a,b): [N, bit_width].
    mode:
      "last_bit"     -> labels = targets[:, -1]                   # [N]
      "per_position" -> labels = targets[:, -bit_width:].ravel()  # [N*bit_width]
    """
    if mode == "last_bit":
        return targets[:, -1].detach().cpu().numpy()
    else:
        return targets[:, -bit_width:].detach().cpu().numpy().reshape(-1)

# ---------------- Plotting ----------------
def plot_frame(proj, labels, title, save_path, xlim, ylim):
    plt.figure(figsize=(7, 6))
    plt.scatter(proj[:, 0], proj[:, 1], c=labels, cmap="coolwarm", s=16, alpha=0.85)
    plt.title(title); plt.xlabel("PC 1"); plt.ylabel("PC 2")
    plt.xlim(*xlim); plt.ylim(*ylim)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close()

# ---------------- Per-mode runner ----------------
def run_for_mode(inputs, targets, label_mode: str):
    out_dir     = out_root / f"locked_{label_mode}"
    frames_dir  = out_dir / "frames"
    gif_path    = out_dir / f"xor_pca_transition_locked_{label_mode}.gif"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Labels that MATCH what we will plot
    labels = build_labels(targets, mode=label_mode)
    print(f"[{label_mode}] labels shape = {labels.shape}; counts:",
          {v:int((labels==v).sum()) for v in np.unique(labels)})

    # Checkpoints
    ckpts = sorted_checkpoints(checkpoint_dir, checkpoint_glob, epoch_regex)
    if not ckpts:
        print(f"[{label_mode}] No checkpoints found.")
        return

    # Fit PCA basis on *final* checkpoint (stable axes)
    _, final_ckpt = ckpts[-1]
    model = create_model(bit_width, d_model, d_mlp).to(device)
    model.load_state_dict(torch.load(final_ckpt, map_location=device))
    model.eval()
    X_final = collect_mlp_post(
        model, inputs, layer_idx,
        mode=("last_bit" if label_mode == "last_bit" else "per_position")
    )
    pca = PCA(n_components=2).fit(X_final)

    # Project all checkpoints; compute global limits
    cached = []
    gx = [np.inf, -np.inf]; gy = [np.inf, -np.inf]

    for step, path in tqdm(ckpts, desc=f"Projecting ({label_mode})"):
        mdl = create_model(bit_width, d_model, d_mlp).to(device)
        mdl.load_state_dict(torch.load(path, map_location=device))
        mdl.eval()

        X = collect_mlp_post(
            mdl, inputs, layer_idx,
            mode=("last_bit" if label_mode == "last_bit" else "per_position")
        )
        proj = pca.transform(X)

        try:
            sil = silhouette_score(proj, labels)
        except Exception:
            sil = float("nan")

        cached.append((step, proj, sil))
        gx = [min(gx[0], proj[:, 0].min()), max(gx[1], proj[:, 0].max())]
        gy = [min(gy[0], proj[:, 1].min()), max(gy[1], proj[:, 1].max())]

    # Global limits with a bit of padding
    dx = 0.05 * (gx[1] - gx[0] + 1e-8)
    dy = 0.05 * (gy[1] - gy[0] + 1e-8)
    global_xlim = (gx[0] - dx, gx[1] + dx)
    global_ylim = (gy[0] - dy, gy[1] + dy)

    # Render locked GIF
    frame_paths = []
    for step, proj, sil in cached:
        title = f"PCA on MLP Output (epoch {step}) — silhouette={sil:.3f} — mode={label_mode}"
        f_path = frames_dir / f"pca_lock_epoch_{step:06d}.png"
        plot_frame(proj, labels, title, f_path, global_xlim, global_ylim)
        frame_paths.append(f_path)

    imageio.mimsave(gif_path, [imageio.imread(p) for p in frame_paths], fps=6)
    print(f"🎥 [{label_mode}] Saved locked GIF: {gif_path}")

# ---------------- Main ----------------
def main():
    # Use the same inputs/targets for both modes to make comparisons consistent
    inputs, targets = generate_xor_dataset(bit_width=bit_width, device=device, seed=123)

    for mode in ("last_bit", "per_position"):
        run_for_mode(inputs, targets, mode)

if __name__ == "__main__":
    main()