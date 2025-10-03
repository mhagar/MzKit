"""
Utilities for handling config file
"""
import os
import shutil
import configparser
from pathlib import Path

def get_project_root() -> Path:
    """
    Get the project root directory (where your main script/package lives)
    :return:
    """
    return Path(__file__).parent.parent.parent  # Depends on where this file is

def get_default_config_template_path() -> Path:
    """
    Get path to default config template in project root
    :return:
    """
    return get_project_root() / 'default_config.ini'

def get_config_path() -> Path:
    """
    Returns platform-appropriate filepath to config file
    :return:
    """
    if os.name == 'nt':  # Windows
        config_dir = Path(
            os.environ.get(
                'APPDATA',
                Path.home()
            )
        )
    else:   # Linux/macOS
        config_dir = Path(
            os.environ.get(
                'XDG_CONFIG_HOME',
                Path.home() / '.config'
            )
        )

    app_config_dir = config_dir / 'mzkit'
    app_config_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return app_config_dir / 'config.ini'

def load_config() -> configparser.ConfigParser:
    """
    Loads a ConfigParser object, creating default if none exist
    :return:
    """
    config = configparser.ConfigParser()
    config_path = get_config_path()
    default_config_template_path = get_default_config_template_path()

    if config_path.exists():
        config.read(config_path)

    else:
        # Copy default config to user location and load it
        if default_config_template_path.exists():
            shutil.copy2(
                default_config_template_path,
                config_path,
            )
        else:
            raise FileNotFoundError(
                f"Unable to find default configuration template."
                f" Expected path: {default_config_template_path}"
            )

    return config

def save_config(config: configparser.ConfigParser) -> None:
    """
    Saves config to disk
    :param config:
    :return:
    """
    config_path = get_config_path()

    with open(config_path, 'w') as f:
        config.write(f)