# scripts/tools/qtest.py
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

from model.parallax import ParallaxConfig, ParallaxForCausalLM

# Use a config that matches the transplant dimensions
cfg = ParallaxConfig(
    num_hidden_layers=8,
    hidden_size=2048,
    num_attention_heads=32,
    num_key_value_heads=8,
    max_position_embeddings=512,
    vocab_size=128256,
)
m = ParallaxForCausalLM(cfg)
keys = list(m.state_dict().keys())
for k in keys[:20]:
    print(k)
