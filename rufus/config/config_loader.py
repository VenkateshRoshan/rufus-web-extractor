import os
import yaml
from typing import Dict, Any


class ConfigLoader:
    """
    Utility class to load and manage configuration from YAML file.
    Supports environment variable overrides.
    """

    _instance = None
    _config = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_config(self, config_path: str = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file with optional environment variable overrides.

        Args:
            config_path (str, optional): Path to config file.
                                         Defaults to project root or default locations.

        Returns:
            Dict[str, Any]: Loaded configuration dictionary
        """
        # Default config paths to try
        default_paths = [
            config_path,
            "config/config.yaml",
            os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml"),
            os.path.expanduser("~/rufus_config.yaml"),
            "/etc/rufus/config.yaml",
        ]

        # Find the first existing config file
        config_file = next(
            (path for path in default_paths if path and os.path.exists(path)), None
        )

        if not config_file:
            raise FileNotFoundError(
                "No configuration file found. Create a config.yaml file."
            )

        # Load YAML configuration
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)

        # Override with environment variables (if they exist)
        self._override_with_env_vars(config)

        # Cache the configuration
        self._config = config
        return config

    def _override_with_env_vars(self, config: Dict[str, Any]) -> None:
        """
        Override configuration with environment variables.
        Uses dot notation for nested configurations.

        Args:
            config (Dict[str, Any]): Configuration dictionary to modify
        """

        def update_nested_dict(d, key_path, value):
            keys = key_path.split(".")
            for key in keys[:-1]:
                d = d.setdefault(key, {})
            d[keys[-1]] = type(d.get(keys[-1], value))(value)

        # Environment variable override mapping
        override_mapping = {
            "MODEL_NAME": "model.name",
            "MODEL_TEMPERATURE": "model.temperature",
            "MAX_DEPTH": "crawler.max_depth",
            "MAX_PAGES": "crawler.max_pages",
            "USE_PARALLEL": "parallel.enabled",
            "RELEVANCE_THRESHOLD": "relevance.threshold",
            "API_HOST": "api.host",
            "API_PORT": "api.port",
        }

        # Check and override configuration with environment variables
        for env_key, config_key in override_mapping.items():
            env_value = os.getenv(env_key)
            if env_value is not None:
                try:
                    update_nested_dict(config, config_key, env_value)
                except Exception as e:
                    print(
                        f"Warning: Could not override {config_key} with {env_value}: {e}"
                    )

    def get(self, key: str = None, default: Any = None) -> Any:
        """
        Retrieve a configuration value by dot-separated key.

        Args:
            key (str, optional): Dot-separated configuration key
            default (Any, optional): Default value if key not found

        Returns:
            Any: Configuration value or default
        """
        if not self._config:
            self.load_config()

        if not key:
            return self._config

        # Navigate through nested dictionary
        value = self._config
        for k in key.split("."):
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default

        return value if value is not None else default


# Create a singleton instance
config = ConfigLoader()

# Usage example
if __name__ == "__main__":
    # Load configuration
    loaded_config = config.load_config()

    # Get specific configuration values
    print("Model Name:", config.get("model.name"))
    print("Max Depth:", config.get("crawler.max_depth"))

    # Get full configuration
    print("\nFull Configuration:")
    print(config.get())
