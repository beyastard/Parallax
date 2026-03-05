# quick & dirty model check!

import torch

meta = torch.load('checkpoints/parallax_v1-WidthB/meta_best.pt', map_location='cpu', weights_only=False)
cfg = meta['config']
print('Saved model configuration:')

for k, v in cfg.items():
    print(f'  {k:20s} = {v}')
    
print(f'\nSaved at iter: {meta["iter"]}')
print(f'Loss:          {meta["loss"]:.4f}')
