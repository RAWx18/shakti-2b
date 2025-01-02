from transformers.configuration_utils import PretrainedConfig


class HyperChakravyuhaConfig(PretrainedConfig):
    r"""
    This configuration class defines the architecture of the `chakravyuhaModel`, inheriting from `PretrainedConfig` to control model outputs. 
    
    Key arguments:

    - vocab_size (*int*, default=151936): Number of unique tokens.  
    - hidden_size (*int*, default=4096): Dimension of hidden representations.  
    - intermediate_size (*int*, default=22016): Dimension of MLP representations.  
    - num_hidden_layers (*int*, default=32): Transformer encoder layers.  
    - num_attention_heads (*int*, default=32): Attention heads per layer.  
    - num_key_value_heads (*int*, default=32): Controls attention types (MHA, MQA, GQA).  
    - hidden_act (*str/function*, default="silu"): Activation function.  
    - max_position_embeddings (*int*, default=32768): Max sequence length.  
    - initializer_range (*float*, default=0.02): Initialization std deviation.  
    - rms_norm_eps (*float*, default=1e-6): Epsilon for RMS normalization.  
    - use_cache (*bool*, default=True): Enable caching of key/values.  
    - tie_word_embeddings (*bool*, default=False): Tie input/output embeddings.  
    - rope_theta (*float*, default=10000.0): Base period for RoPE embeddings.  
    - use_sliding_window (*bool*, default=False): Enable sliding window attention.  
    - sliding_window (*int*, default=4096): SWA window size.  
    - max_window_layers (*int*, default=28): SWA layers in the bottom layers.  
    - attention_dropout (*float*, default=0.0): Dropout ratio for attention probabilities.
    """

    model_type = "chakravyuha"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        hyper_layers=[1,9,17,25],
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.max_window_layers = max_window_layers

        if num_key_value_heads is None: # for backward compatibility
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.hyper_layers = hyper_layers
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
