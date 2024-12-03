import importlib

def load_model_config(model_name):
    """
    Dynamically load the configuration module for the given model.
    
    Args:
        model_name (str): The name of the model (e.g., 'model_a').
    
    Returns:
        dict: Configuration dictionary.
    """
    try:
        config_module = importlib.import_module(f"configs.model_configs.{model_name}_conf")
        return config_module.CONFIG
    except ModuleNotFoundError:
        raise ValueError(f"No configuration found for model '{model_name}'.")
