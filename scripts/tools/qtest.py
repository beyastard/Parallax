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
