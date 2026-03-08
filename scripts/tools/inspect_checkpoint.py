# scripts/tools/inspect_checkpoint.py
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

import argparse
import torch

parser = argparse.ArgumentParser(description="Inspect a Parallax checkpoint meta file.")
parser.add_argument("--meta", type=str, required=True,
                    help="Path to a meta_*.pt file")
args = parser.parse_args()

m   = torch.load(args.meta, map_location="cpu", weights_only=False)
cfg = m["config"]

print(f"  iter:                    {m.get('iter', 'n/a')}")
print(f"  loss:                    {m.get('loss', 'n/a')}")
print(f"  donor:                   {m.get('donor', 'n/a')}")
print()
print(f"  hidden_size:             {cfg.get('hidden_size',             cfg.get('n_embd',     'MISSING'))}")
print(f"  num_hidden_layers:       {cfg.get('num_hidden_layers',       cfg.get('n_layer',    'MISSING'))}")
print(f"  num_attention_heads:     {cfg.get('num_attention_heads',     cfg.get('n_head',     'MISSING'))}")
print(f"  num_key_value_heads:     {cfg.get('num_key_value_heads',     cfg.get('n_kv_heads', 'MISSING'))}")
print(f"  intermediate_size:       {cfg.get('intermediate_size',       cfg.get('ffn_dim',    'MISSING'))}")
print(f"  max_position_embeddings: {cfg.get('max_position_embeddings', cfg.get('block_size', 'MISSING'))}")
print(f"  vocab_size:              {cfg.get('vocab_size',                                    'MISSING')}")
print(f"  rope_theta:              {cfg.get('rope_theta',                                    'MISSING')}")
print(f"  rms_norm_eps:            {cfg.get('rms_norm_eps',            cfg.get('norm_eps',   'MISSING'))}")
print(f"  num_loops:               {cfg.get('num_loops',                                     'MISSING')}")
print(f"  use_swap:                {cfg.get('use_swap',                                      'MISSING')}")
print(f"  hidden_act:              {cfg.get('hidden_act',              cfg.get('activation', 'MISSING'))}")
