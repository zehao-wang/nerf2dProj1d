import json
from easydict import EasyDict as edict
from utils.config_loader import load_model_config
from .vanillaNeRF import VanillaNeRF

# Dictionary mapping model names to classes
MODEL_REGISTRY = {
    "vanilla_nerf": VanillaNeRF,
}

def init_model(model_name, cam_config):
    config = edict(load_model_config(model_name))
    return MODEL_REGISTRY.get(model_name)(config, cam_config)
