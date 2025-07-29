from llama.llama_model import ModelArgs, Transformer

__all__ = ["Transformer"]

llama2_configs = {
    "debugmodel": ModelArgs(dim=256, n_layers=8, n_heads=16),
    "271M": ModelArgs(dim=1024, n_layers=16, n_heads=8),
    "1B": ModelArgs(dim=2048, n_layers=18, n_heads=16),
    "gpt3xl": ModelArgs(
        dim=2048,
        n_layers=24,
        n_heads=16,
        embed_type="position",
        activation="GeLU",
        vocab_size=32000,
        max_seq_len=2048,
    ),
    "gpt3_6_7B": ModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        embed_type="position",
        activation="GeLU",
        vocab_size=32000,
        max_seq_len=2048,
    ),
    "gpt2": ModelArgs(
        dim=1600,
        n_layers=48,
        n_heads=25,
        embed_type="position",
        activation="GeLU",
        vocab_size=32000,
        max_seq_len=1024,
    ),
    "7B": ModelArgs(
        dim=4096, n_layers=32, n_heads=32, vocab_size=32000, max_seq_len=4096
    ),
    "13B": ModelArgs(
        dim=5120, n_layers=40, n_heads=40, vocab_size=32000, max_seq_len=4096
    ),
    "26B": ModelArgs(
        dim=5120, n_layers=80, n_heads=40, vocab_size=32000, max_seq_len=4096
    ),
    "70B": ModelArgs(
        dim=8192,
        n_layers=80,
        n_heads=64,
        vocab_size=32000,
        max_seq_len=4096,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
    ),
}

llama3_configs = {
    "debugmodel": ModelArgs(dim=256, n_layers=8, n_heads=16, rope_theta=500000),
    "8B": ModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        max_seq_len=8192,
        vocab_size=128256,
    ),
    "70B": ModelArgs(
        dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
        rope_theta=500000,
    ),
    "405B": ModelArgs(
        dim=16384,
        n_layers=126,
        n_heads=128,
        n_kv_heads=8,
        ffn_dim_multiplier=1.2,
        multiple_of=4096,
        rope_theta=500000,
    ),
}

all_models = {"llama2": llama2_configs, "llama3": llama3_configs}
