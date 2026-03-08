# scripts/train.py
#
# Copyright (C) 2025-2026 Bryan K Reinhart & BeySoft
#
# This file is part of Parallax.
#
# Parallax is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Parallax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public
# License along with Parallax. If not, see <https://www.gnu.org/licenses/>.

import os
import time
import math
import argparse
import torch
import numpy as np

from torch.amp import GradScaler, autocast
from safetensors.torch import save_file
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

from model.parallax import ParallaxConfig, ParallaxForCausalLM

# Don't forget to run: tensorboard --logdir=logs

def parse_args():
    parser = argparse.ArgumentParser(description="Train the Parallax language model.")

    # Paths
    parser.add_argument("--train_path",      type=str, default="data/train.bin")
    parser.add_argument("--val_path",        type=str, default="data/val.bin")
    parser.add_argument("--checkpoint_dir",  type=str, default="checkpoints/parallax_v1")
    parser.add_argument("--log_dir",         type=str, default="logs/parallax_experiment")
    parser.add_argument("--tokenizer",       type=str, default="NousResearch/Llama-2-7b-hf",
                        help="HuggingFace tokenizer name or local path (must match prepare_data.py)")

    # Model hyperparameters (override ParallaxConfig defaults)
    parser.add_argument("--hidden_size",             type=int,   default=None)
    parser.add_argument("--num_hidden_layers",       type=int,   default=None)
    parser.add_argument("--num_attention_heads",     type=int,   default=None)
    parser.add_argument("--num_key_value_heads",     type=int,   default=None)
    parser.add_argument("--intermediate_size",       type=int,   default=None)
    parser.add_argument("--max_position_embeddings", type=int,   default=None)
    parser.add_argument("--vocab_size",              type=int,   default=None)
    parser.add_argument("--num_loops",               type=int,   default=None)
    parser.add_argument("--use_swap",
                        type=lambda x: x.lower() not in ("false", "0", "no"),
                        default=None,
                        metavar="BOOL",
                        help="Enable track swap (default: True). Pass false/0/no to disable.")

    # Training hyperparameters
    parser.add_argument("--batch_size",      type=int,   default=4)
    parser.add_argument("--grad_accum",      type=int,   default=8,
                        help="Gradient accumulation steps. Effective batch = batch_size * grad_accum")
    parser.add_argument("--lr",              type=float, default=6e-4)
    parser.add_argument("--weight_decay",    type=float, default=0.1)
    parser.add_argument("--max_iters",       type=int,   default=100000)
    parser.add_argument("--warmup_iters",    type=int,   default=2000)

    # Evaluation & logging
    parser.add_argument("--eval_interval",   type=int,   default=100,
                        help="Run validation every N iterations")
    parser.add_argument("--eval_iters",      type=int,   default=50,
                        help="Number of batches to average for validation loss")
    parser.add_argument("--save_interval",   type=int,   default=1000,
                        help="Save a periodic snapshot every N iterations")
    parser.add_argument("--log_interval",    type=int,   default=10,
                        help="Print and log training loss every N iterations")

    # Generation
    parser.add_argument("--generate_every",  type=int,   default=1000)
    parser.add_argument("--gen_len",         type=int,   default=50)
    parser.add_argument("--gen_prompt",      type=str,   default="Once upon a time")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_batch(data_path, batch_size, block_size, device):
    # dtype=np.uint32 must match prepare_data.py output
    data = np.memmap(data_path, dtype=np.uint32, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# LR schedule: linear warmup -> cosine decay -> 10% floor
# ---------------------------------------------------------------------------

def get_lr(it, warmup_iters, max_iters, base_lr):
    min_lr = base_lr * 0.1
    if it < warmup_iters:
        return base_lr * it / warmup_iters
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
    """Average loss over eval_iters random batches for a smooth val estimate."""
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
    """Greedy decode a short snippet to give a qualitative sense of progress."""
    model.eval()
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    generated = []
    for _ in range(gen_len):
        x_cond = x[:, -config.max_position_embeddings:]
        with autocast(device_type=device_type, dtype=torch.float16):
            logits, _ = model(x_cond)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        x = torch.cat((x, next_token), dim=1)
        generated.append(next_token.item())

    decoded = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"\n--- SNIPPET AT ITERATION {current_iter} ---")
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
    # Use config.to_dict() (PretrainedConfig method) — captures all canonical
    # HF fields correctly without doubling up legacy-alias properties.
    meta = {
        "iter":      it,
        "loss":      loss,
        "optimizer": optimizer.state_dict(),
        "config":    config.to_dict(),
    }
    torch.save(meta, os.path.join(checkpoint_dir, f"meta_{tag}.pt"))
    print(f"  Checkpoint saved: {tag}  (loss {loss:.4f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_type = 'cuda' if device == 'cuda' else 'cpu'

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # --- Tokenizer (for live generation only; data is already tokenized) ---
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # --- Build config, applying any CLI overrides ---
    # Collect only args that were explicitly set (non-None) so ParallaxConfig
    # defaults are not silently overwritten when the flag is absent.
    cfg_overrides = {k: v for k, v in {
        "hidden_size":             args.hidden_size,
        "num_hidden_layers":       args.num_hidden_layers,
        "num_attention_heads":     args.num_attention_heads,
        "num_key_value_heads":     args.num_key_value_heads,
        "intermediate_size":       args.intermediate_size,
        "max_position_embeddings": args.max_position_embeddings,
        "vocab_size":              args.vocab_size,
        "num_loops":               args.num_loops,
        "use_swap":                args.use_swap,
    }.items() if v is not None}

    config = ParallaxConfig(**cfg_overrides)

    # --- Model ---
    model = ParallaxForCausalLM(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Starting Parallax training on {device}")
    print(f"Model parameters:        {total_params / 1e6:.2f}M")
    print(f"hidden_size:             {config.hidden_size}")
    print(f"num_hidden_layers:       {config.num_hidden_layers}  (per track)")
    print(f"num_attention_heads:     {config.num_attention_heads}")
    print(f"num_key_value_heads:     {config.num_key_value_heads}")
    print(f"max_position_embeddings: {config.max_position_embeddings}")
    print(f"num_loops:               {config.num_loops}  (use_swap={config.use_swap})")
    print(f"Effective batch size:    {args.batch_size * args.grad_accum}")

    # --- Optimizer & AMP scaler ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scaler = GradScaler(enabled=(device_type == 'cuda'))

    # --- TensorBoard ---
    writer = SummaryWriter(log_dir=args.log_dir)

    # --- Training loop ---
    best_val_loss = float('inf')

    for it in range(args.max_iters):
        t0 = time.time()

        # Update learning rate
        lr = get_lr(it, args.warmup_iters, args.max_iters, args.lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        # Gradient accumulation
        for _ in range(args.grad_accum):
            x, y = get_batch(
                args.train_path, args.batch_size,
                config.max_position_embeddings, device,
            )
            with autocast(device_type=device_type, dtype=torch.float16):
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            accum_loss += loss.item()
            scaler.scale(loss).backward()

        # Gradient clipping + optimiser step
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        dt_ms = (time.time() - t0) * 1000

        # --- Validation ---
        if it % args.eval_interval == 0:
            val_loss = estimate_loss(
                model, args.val_path, args.eval_iters,
                args.batch_size, config.max_position_embeddings, device,
            )
            print(f"step {it:6d} | train loss {accum_loss:.4f} | val loss {val_loss:.4f} "
                  f"| lr {lr:.2e} | {dt_ms:.1f}ms")
            writer.add_scalar("Loss/val",   val_loss,   it)
            writer.add_scalar("Loss/train", accum_loss, it)
            writer.add_scalar("LR",         lr,         it)

            # Save best checkpoint (val-loss gated, skip iter 0)
            if val_loss < best_val_loss and it > 0:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, it, val_loss,
                                args.checkpoint_dir, config, label="best")
                print(f"  >>> New best val loss: {best_val_loss:.4f}")

        # --- Periodic snapshot ---
        if it % args.save_interval == 0 and it > 0:
            save_checkpoint(model, optimizer, it, accum_loss,
                            args.checkpoint_dir, config)

        # --- Lightweight logging on non-eval, non-snapshot steps ---
        elif it % args.log_interval == 0 and it % args.eval_interval != 0:
            print(f"iter {it:6d} | loss {accum_loss:.4f} | lr {lr:.2e} | {dt_ms:.1f}ms")

        # --- Live generation snippet ---
        if it % args.generate_every == 0:
            generate_live_snippet(
                model, tokenizer, device, config,
                args.gen_prompt, args.gen_len, it,
            )

    # --- Final eval, snippet and checkpoint ---
    final_iter = args.max_iters - 1
    print(f"\n--- Final Evaluation (iter {final_iter}) ---")
    final_val_loss = estimate_loss(
        model, args.val_path, args.eval_iters,
        args.batch_size, config.max_position_embeddings, device,
    )
    final_lr = get_lr(final_iter, args.warmup_iters, args.max_iters, args.lr)
    print(f"step {final_iter:6d} | val loss {final_val_loss:.4f} | lr {final_lr:.2e}")
    writer.add_scalar("Loss/val", final_val_loss, final_iter)

    if final_val_loss < best_val_loss:
        best_val_loss = final_val_loss
        save_checkpoint(model, optimizer, final_iter, final_val_loss,
                        args.checkpoint_dir, config, label="best")
        print(f"  >>> New best val loss: {best_val_loss:.4f}")

    save_checkpoint(model, optimizer, final_iter, final_val_loss,
                    args.checkpoint_dir, config, label="final")

    tokenizer_path = os.path.join(args.checkpoint_dir, "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)
    print(f"  Tokenizer saved to: {tokenizer_path}")

    generate_live_snippet(
        model, tokenizer, device, config,
        args.gen_prompt, args.gen_len, final_iter,
    )

    writer.close()
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
