"""
Utilities for handling config file
"""
import os
import shutil
import configparser
from pathlib import Path
from typing import Iterable

def get_project_root() -> Path:
    """
    Where main script/package lives
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

    # Load default config first
    if not default_config_template_path.exists():
        raise FileNotFoundError(
            f"Unable to find default configuration template."
            f" Expected path: {default_config_template_path}"
        )

    config.read(default_config_template_path)

    # Overlay user config (if it exists)
    if config_path.exists():
        config.read(config_path)

    return config


def load_default_config() -> configparser.ConfigParser:
    """
    Loads a ConfigParser object from the shipped default template only,
    ignoring any user overrides. Used to restore defaults.
    :return:
    """
    config = configparser.ConfigParser()
    default_config_template_path = get_default_config_template_path()

    if not default_config_template_path.exists():
        raise FileNotFoundError(
            f"Unable to find default configuration template."
            f" Expected path: {default_config_template_path}"
        )

    config.read(default_config_template_path)

    return config


def save_config(
    config: configparser.ConfigParser,
    sections: Iterable[str],
) -> None:
    """
    Saves `sections` of `config` to disk, leaving every other section as
    it currently is on disk.

    Several components hold their own ConfigParser (each loaded at
    startup), so writing a whole copy would clobber sections saved since
    by someone else (i.e. stale [alignment] params overwriting fresh ones).

    :param config: the caller's config, holding the sections to save.
    :param sections: names of the sections the caller owns / changed.
    """
    current = load_config()
    for section in sections:
        if current.has_section(section):
            current.remove_section(section)
        if config.has_section(section):
            current.add_section(section)
            for key, value in config.items(section, raw=True):
                current.set(section, key, value)

    with open(get_config_path(), 'w') as f:
        current.write(f)

# Fallback when the config has no `[instrument] saturation_threshold`
DEFAULT_SATURATION_THRESHOLD = 1e10

# (user config mtime, value); see get_saturation_threshold
_saturation_cache: tuple[float, float] | None = None


def get_saturation_threshold() -> float:
    """
    config_path = get_config_path()

    with open(config_path, 'w') as f:
        config.write(f)