# coding=utf-8
"""shakti model configuration"""

import os
from typing import Union

from transformers.utils import logging
from configuration_hyper_chakravyuha import HyperChakravyuhaConfig
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
logger = logging.get_logger(__name__)


class shaktiConfig(HyperChakravyuhaConfig):
    model_type = "shakti"
    keys_to_ignore_at_inference = ["past_key_values"]

    default_drishti_config = {
        "hidden_size": 1152,
        "image_size": 384,
        "intermediate_size": 4304,
        "model_type": "siglip_drishti_model",
        "num_attention_heads": 16,
        "num_hidden_layers": 27,
        "patch_size": 14
    }


    def __init__(
        self,
        use_cache=True,
        drishti_config=None,
        **kwargs,
    ):
        self.use_cache = use_cache

        if drishti_config is None:
            self.drishti_config = SiglipVisionConfig(**self.default_drishti_config)
            # logger.info("drishti_config is None, using default drishti config")
        elif isinstance(drishti_config, dict):
            self.drishti_config = SiglipVisionConfig(**drishti_config)
        elif isinstance(drishti_config, SiglipVisionConfig):
            self.drishti_config = drishti_config
        self.image_size = self.drishti_config.image_size
        self.patch_size = self.drishti_config.patch_size

        super().__init__(**kwargs)
