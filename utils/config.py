"""
Configuration parser utility.
Loads hyperparameters from a YAML file and provides a clean namespace-style accessor.
"""
import yaml
import os


class Config:
    """
    Hierarchical configuration loaded from a YAML file.
    Supports attribute-style access: config.model.d_model
    """

    def __init__(self, d: dict = None):
        d = d or {}
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def to_dict(self) -> dict:
        """Convert back to a plain dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self):
        return f"Config({self.to_dict()})"

    def get(self, key, default=None):
        return getattr(self, key, default)


def load_config(path: str = "config.yaml") -> Config:
    """
    Load a YAML configuration file and return a Config object.

    Args:
        path: Path to the YAML config file.

    Returns:
        Config object with attribute-style access.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)

    return Config(raw)
