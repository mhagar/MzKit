"""
Tests for config persistence (core/utils/config.py).
"""
import os

import pytest

from core.utils.config import load_config, save_config


@pytest.fixture
def user_config_dir(tmp_path, monkeypatch):
    if os.name == 'nt':
        monkeypatch.setenv('APPDATA', str(tmp_path))
    else:
        monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
    return tmp_path


def test_save_config_does_not_clobber_other_sections(user_config_dir):
    # Two components load their own copies at startup...
    alignment_owner = load_config()
    findmfs_owner = load_config()

    # ...one saves new alignment params...
    alignment_owner.set('alignment', 'rt_tolerance', '10.0')
    save_config(alignment_owner, ['alignment'])

    # ...then the other saves its own section from its (stale) copy
    findmfs_owner.set('findmfs', 'error_ppm', '3')
    save_config(findmfs_owner, ['findmfs'])

    reloaded = load_config()
    assert reloaded.getfloat('alignment', 'rt_tolerance') == 10.0
    assert reloaded.getfloat('findmfs', 'error_ppm') == 3.0


def test_save_config_writes_new_sections(user_config_dir):
    config = load_config()
    config.add_section('brand_new')
    config.set('brand_new', 'key', 'value')
    save_config(config, ['brand_new'])
    assert load_config().get('brand_new', 'key') == 'value'
