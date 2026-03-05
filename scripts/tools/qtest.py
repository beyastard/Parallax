from model.parallax.modeling_parallax import Parallax
from model.parallax.configuration_parallax import ParallaxConfig

# Use a config that matches the transplant dimensions
cfg = ParallaxConfig(n_layer=8, n_embd=2048, n_head=32, n_kv_heads=8,
                     block_size=512, vocab_size=128256)
m = Parallax(cfg)
keys = list(m.state_dict().keys())
for k in keys[:20]:
    print(k)
