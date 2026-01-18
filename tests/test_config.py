from pathlib import Path

from aft.config import AppConfig, load_config


def test_config_schema_round_trip():
    config_path = Path("config.yaml")
    config = load_config(config_path)

    assert isinstance(config, AppConfig)
    assert config.training.steps_per_window == 100
    assert config.monitoring.window_size == 1000
