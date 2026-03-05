import torch

m = torch.load('checkpoints/parallax_amd_linear_ft/meta_best.pt', weights_only=False)
cfg = m['config']
print('n_kv_heads:', cfg['n_kv_heads'])
print('n_head:    ', cfg['n_head'])
print('n_embd:    ', cfg['n_embd'])
print('block_size:', cfg['block_size'])
print('ffn_dim:   ', cfg['ffn_dim'])
print('rope_theta:', cfg.get('rope_theta', 'MISSING'))
