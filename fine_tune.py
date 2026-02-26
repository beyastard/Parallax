import os
import time
import math
import argparse
import torch
import numpy as np
from torch.amp import GradScaler, autocast
from safetensors.torch import save_file, load_file as load_safetensors
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
from model import Parallax
from config import ParallaxConfig

# Don't forget to run: tensorboard --logdir=logs
#
# USAGE EXAMPLES:
#
# Basic fine-tune from the best pre-trained checkpoint:
#   python fine_tune.py --base_checkpoint checkpoints/parallax_v1/model_best.safetensors \
#                       --base_meta      checkpoints/parallax_v1/meta_best.pt \
#                       --train_path     data/new_dataset/train.bin \
#                       --val_path       data/new_dataset/val.bin
#
# With catastrophic forgetting mitigation (30% TinyStories mixed in):
#   python fine_tune.py ... --anchor_path data/train.bin --mix_ratio 0.3
#
# With frozen embeddings:
#   python fine_tune.py ... --freeze_embeddings
#
# Override LR (default is already conservative at 1e-4):
#   python fine_tune.py ... --lr 5e-5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a pre-trained Parallax model on a new dataset."
    )

    # --- Source checkpoint (required) ---
    parser.add_argument("--base_checkpoint", type=str, required=True,
                        help="Path to the .safetensors weights file to fine-tune from")
    parser.add_argument("--base_meta",       type=str, required=True,
                        help="Path to the corresponding meta .pt file")

    # --- Paths ---
    parser.add_argument("--train_path",      type=str, required=True,
                        help="Tokenized .bin file for fine-tuning training data")
    parser.add_argument("--val_path",        type=str, required=True,
                        help="Tokenized .bin file for fine-tuning validation data")
    parser.add_argument("--checkpoint_dir",  type=str, default="checkpoints/parallax_v1_ft",
                        help="Output directory for fine-tuned checkpoints (separate from pre-training)")
    parser.add_argument("--log_dir",         type=str, default="logs/parallax_ft",
                        help="TensorBoard log directory for this fine-tuning run")
    parser.add_argument("--tokenizer",       type=str, default="NousResearch/Llama-2-7b-hf")

    # --- Forgetting mitigation ---
    parser.add_argument("--anchor_path",     type=str, default=None,
                        help="Tokenized .bin of the original pre-training data (e.g. TinyStories). "
                             "When provided, a fraction of each batch is drawn from this file "
                             "to reduce catastrophic forgetting.")
    parser.add_argument("--mix_ratio",       type=float, default=0.2,
                        help="Fraction of each batch drawn from --anchor_path (default: 0.2). "
                             "Ignored if --anchor_path is not set.")

    # --- Fine-tuning hyperparameters ---
    # Defaults are deliberately more conservative than pre-training
    parser.add_argument("--batch_size",      type=int,   default=4)
    parser.add_argument("--grad_accum",      type=int,   default=8)
    parser.add_argument("--lr",              type=float, default=1e-4,
                        help="Peak learning rate. Lower than pre-training default (6e-4) "
                             "to avoid overwriting learned representations.")
    parser.add_argument("--weight_decay",    type=float, default=0.1)
    parser.add_argument("--max_iters",       type=int,   default=20000,
                        help="Total fine-tuning steps (shorter than pre-training by default)")
    parser.add_argument("--warmup_iters",    type=int,   default=200,
                        help="Short warmup since we're starting from a good set of weights")

    # --- Embedding freeze ---
    parser.add_argument("--freeze_embeddings", action="store_true",
                        help="Freeze tok_emb weights during fine-tuning to preserve the "
                             "token representations built during pre-training")

    # --- Evaluation & logging ---
    parser.add_argument("--eval_interval",   type=int,   default=100)
    parser.add_argument("--eval_iters",      type=int,   default=50)
    parser.add_argument("--save_interval",   type=int,   default=500,
                        help="Periodic snapshot interval (shorter than pre-training default)")
    parser.add_argument("--log_interval",    type=int,   default=10)

    # --- Generation ---
    parser.add_argument("--generate_every",  type=int,   default=500)
    parser.add_argument("--gen_len",         type=int,   default=50)
    parser.add_argument("--gen_prompt",      type=str,   default="Once upon a time")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_batch(data_path, batch_size, block_size, device):
    """Draw a random batch from a tokenized .bin file."""
    data = np.memmap(data_path, dtype=np.uint32, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def get_mixed_batch(ft_path, anchor_path, batch_size, block_size, device, mix_ratio):
    """
    Draw a batch that blends fine-tuning data and anchor (pre-training) data.
    mix_ratio controls the fraction of samples drawn from the anchor dataset.
    e.g. mix_ratio=0.2 means 1 in 5 samples comes from the anchor.
    """
    anchor_count = max(1, round(batch_size * mix_ratio))
    ft_count = batch_size - anchor_count

    x_ft,     y_ft     = get_batch(ft_path,     ft_count,     block_size, device)
    x_anchor, y_anchor = get_batch(anchor_path, anchor_count, block_size, device)

    x = torch.cat([x_ft, x_anchor], dim=0)
    y = torch.cat([y_ft, y_anchor], dim=0)
    return x, y


# ---------------------------------------------------------------------------
# LR schedule: short linear warmup → cosine decay → 10% floor
# ---------------------------------------------------------------------------

def get_lr(it, warmup_iters, max_iters, base_lr):
    min_lr = base_lr * 0.1
    if it < warmup_iters:
        return base_lr * it / max(1, warmup_iters)
    if it >= max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (base_lr - min_lr)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(model, data_path, eval_iters, batch_size, block_size, device):
    model.eval()
    losses = torch.zeros(eval_iters)
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    for k in range(eval_iters):
        x, y = get_batch(data_path, batch_size, block_size, device)
        with autocast(device_type=device_type, dtype=torch.float16):
            _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


# ---------------------------------------------------------------------------
# Live generation snippet
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_live_snippet(model, tokenizer, device, config, prompt, gen_len, current_iter):
    model.eval()
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    generated = []
    for _ in range(gen_len):
        x_cond = x[:, -config.block_size:]
        with autocast(device_type=device_type, dtype=torch.float16):
            logits, _ = model(x_cond)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        x = torch.cat((x, next_token), dim=1)
        generated.append(next_token.item())

    decoded = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"\n--- SNIPPET AT ITER {current_iter} ---")
    print(f"{prompt}{decoded}")
    print("-" * 40 + "\n")
    model.train()


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, it, loss, checkpoint_dir, config, label=None):
    tag = label if label else f"iter_{it}"
    weights_path = os.path.join(checkpoint_dir, f"model_{tag}.safetensors")
    save_file(model.state_dict(), weights_path)
    meta = {
        "iter":      it,
        "loss":      loss,
        "optimizer": optimizer.state_dict(),
        "config":    config.__dict__,
    }
    torch.save(meta, os.path.join(checkpoint_dir, f"meta_{tag}.pt"))
    print(f"  Checkpoint saved: {tag}  (loss {loss:.4f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device      = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_type = 'cuda' if device == 'cuda' else 'cpu'

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir,        exist_ok=True)

    # --- Validate inputs ---
    for path, name in [(args.base_checkpoint, "base_checkpoint"),
                       (args.base_meta,       "base_meta"),
                       (args.train_path,      "train_path"),
                       (args.val_path,        "val_path")]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"--{name} not found: {path}")

    if args.anchor_path and not os.path.isfile(args.anchor_path):
        raise FileNotFoundError(f"--anchor_path not found: {args.anchor_path}")

    # --- Load tokenizer ---
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # --- Reconstruct config from the saved pre-training meta ---
    meta   = torch.load(args.base_meta, map_location=device, weights_only=False)
    config = ParallaxConfig(**meta["config"])
    print(f"Loaded config from: {args.base_meta}")
    print(f"  Pre-trained at iter {meta['iter']}  |  loss {meta['loss']:.4f}")

    # --- Initialise model and load pre-trained weights ---
    model = Parallax(config).to(device)
    state_dict = load_safetensors(args.base_checkpoint)
    model.load_state_dict(state_dict)
    print(f"Loaded weights from: {args.base_checkpoint}")

    # --- Optionally freeze token embeddings ---
    if args.freeze_embeddings:
        model.tok_emb.weight.requires_grad = False
        print("Token embeddings frozen.")

    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    # --- Fresh optimizer (intentionally not restoring pre-training momentum) ---
    # Restoring AdamW state from pre-training would carry over gradient history
    # from a different data distribution, which destabilises early fine-tuning steps.
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scaler = GradScaler(enabled=(device_type == 'cuda'))

    # --- TensorBoard ---
    writer = SummaryWriter(log_dir=args.log_dir)

    # --- Report mixing strategy ---
    if args.anchor_path:
        print(f"Catastrophic forgetting mitigation: ON")
        print(f"  Anchor data: {args.anchor_path}")
        print(f"  Mix ratio:   {args.mix_ratio:.0%} anchor / "
              f"{1 - args.mix_ratio:.0%} fine-tune per batch")
    else:
        print("Catastrophic forgetting mitigation: OFF (no --anchor_path provided)")

    print(f"\nStarting fine-tuning on {device}")
    print(f"  Base LR:            {args.lr:.2e}")
    print(f"  Max iters:          {args.max_iters}")
    print(f"  Effective batch:    {args.batch_size * args.grad_accum}")
    print(f"  Output checkpoints: {args.checkpoint_dir}\n")

    # --- Fine-tuning loop ---
    best_val_loss = float('inf')

    for it in range(args.max_iters):
        t0 = time.time()

        lr = get_lr(it, args.warmup_iters, args.max_iters, args.lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(args.grad_accum):
            if args.anchor_path:
                x, y = get_mixed_batch(
                    args.train_path, args.anchor_path,
                    args.batch_size, config.block_size, device, args.mix_ratio
                )
            else:
                x, y = get_batch(args.train_path, args.batch_size, config.block_size, device)

            with autocast(device_type=device_type, dtype=torch.float16):
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            accum_loss += loss.item()
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        dt_ms = (time.time() - t0) * 1000

        # --- Validation ---
        if it % args.eval_interval == 0:
            val_loss = estimate_loss(
                model, args.val_path, args.eval_iters,
                args.batch_size, config.block_size, device
            )
            print(f"step {it:6d} | train loss {accum_loss:.4f} | val loss {val_loss:.4f} "
                  f"| lr {lr:.2e} | {dt_ms:.1f}ms")
            writer.add_scalar("Loss/val",   val_loss,   it)
            writer.add_scalar("Loss/train", accum_loss, it)
            writer.add_scalar("LR",         lr,         it)

            if val_loss < best_val_loss and it > 0:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, it, val_loss,
                                args.checkpoint_dir, config, label="best")
                print(f"  >>> New best val loss: {best_val_loss:.4f}")

        elif it % args.save_interval == 0 and it > 0:
            save_checkpoint(model, optimizer, it, accum_loss,
                            args.checkpoint_dir, config)

        elif it % args.log_interval == 0:
            print(f"iter {it:6d} | loss {accum_loss:.4f} | lr {lr:.2e} | {dt_ms:.1f}ms")

        if it % args.generate_every == 0:
            generate_live_snippet(
                model, tokenizer, device, config,
                args.gen_prompt, args.gen_len, it
            )

    writer.close()
    print("Fine-tuning complete.")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")

if __name__ == "__main__":
    main()
