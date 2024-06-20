import yaml
import os

with open('model_config.yaml', 'r') as f:
    config_data = yaml.safe_load(f)

__all__ = [
    'config_data'
]