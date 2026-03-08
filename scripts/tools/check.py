# scripts/tools/check.py
# quick & dirty model check!
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

import torch

meta = torch.load('checkpoints/parallax_v1-WidthB/meta_best.pt', map_location='cpu', weights_only=False)
cfg = meta['config']
print('Saved model configuration:')

for k, v in cfg.items():
    print(f'  {k:20s} = {v}')
    
print(f'\nSaved at iter: {meta["iter"]}')
print(f'Loss:          {meta["loss"]:.4f}')
