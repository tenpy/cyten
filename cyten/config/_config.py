# Copyright (C) TeNPy Developers, Apache license
import os
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

ALL = object()  # sentinel: default behavior to restore all options to default

DEFAULT_USER_CONFIG_PATH = Path.home() / f'.cytenconfig.yaml'
LOCAL_CONFIG_PATH = '.cytenconfig.yaml'
USER_CONFIG_ENVVAR = 'CYTEN_CONFIG_FILE'


_runtime_options: dict[str, Any] = {}
_temp_context_options: ContextVar[dict[str, Any]] = ContextVar('_temp_context_options', default={})
_local_file_options: dict[str, Any] = {}
_user_file_options: dict[str, Any] = {}
_envvar_options: dict[str, Any] = {}


@dataclass(frozen=True)
class Option:
    """Specification for a config option

    Attributes
    ----------
    name : str
        The name for the option
    default : Any
        The default value
    env_var : str | None
        The envvar that the option can also be read from, if any
    coerce : callable
        A function to coerce arbitrary values to the correct type for the option.
        Should also do input checks (e.g. checking non-negative)

    """

    name: str
    default: Any
    env_var: str
    coerce: callable = lambda val: val


def coerce_int(val, min: int = None):
    val = int(val)
    if min is not None:
        assert val >= min
    return val


def coerce_str(val, allowed=None):
    val = str(val)
    if allowed is not None:
        assert val in allowed
    return val


def coerce_bool(val):
    if isinstance(val, str):
        return val.lower() in ['true', '1', 'y', 'yes']
    return bool(val)


OPTIONS: dict[str, Option] = dict(
    print_linewidth=Option('print_linewidth', 100, 'CYTEN_PRINT_LINEWIDTH', partial(coerce_int, min=10)),
    print_indent=Option('print_indent', 2, 'CYTEN_PRINT_INDENT', partial(coerce_int, min=0)),
    maxlines_spaces=Option('maxlines_spaces', 15, 'CYTEN_MAXLINES_SPACES', partial(coerce_int, min=0)),
    maxlines_tensors=Option('maxlines_tensors', 30, 'CYTEN_MAXLINES_TENSORS', partial(coerce_int, min=0)),
    check_fusion=Option('check_fusion', True, 'CYTEN_CHECK_FUSION', coerce_bool),
    default_tensor_backend=Option(
        'default_tensor_backend',
        'abelian',
        'CYTEN_DEFAULT_TENSOR_BACKEND',
        partial(coerce_str, allowed=['no_symmetry', 'abelian', 'fusion_tree']),
    ),
    default_block_backend=Option(
        'default_block_backend',
        'numpy',
        'CYTEN_DEFAULT_BLOCK_BACKEND',
        partial(coerce_str, allowed=['numpy', 'torch', 'cpu', 'gpu', 'apple_silicon']),
    ),
)


defaults = MappingProxyType({k: v.default for k, v in OPTIONS.items()})
"""Read-only dict-like view of the default values for each config option."""


def validate_options(options: dict[str, Any]) -> dict[str, Any]:
    """Validate a dict of options. Check both keys and values."""
    res = {}
    for key, val in options.items():
        opt = OPTIONS.get(key, None)
        if opt is None:
            raise KeyError(f'Invalid config option: {key}')
        res[key] = opt.coerce(val)
    return res


@contextmanager
def temporary_options(**options):
    """Context manager to temporarily override config options. Pass config options as kwargs.

    Examples
    --------
    .. code-block::

        cyten.set_options(check_fusion=False)

        with cyten.temporary_options(check_fusion=True):
            do_stuff()  # <- fusion checks active

        do_stuff()  # <- fusion checks inactive

    """
    current = _temp_context_options.get()
    merged = current.copy()
    try:
        options = validate_options(options)
    except Exception as e:
        raise e from None
    merged.update(options)
    token = _temp_context_options.set(merged)
    try:
        yield
    finally:
        _temp_context_options.reset(token)


def set_options(**options):
    """Set any number of config options.

    Examples
    --------
    .. code-block::

        cyten.set_options(print_linewidth=120)

    """
    try:
        options = validate_options(options)
    except Exception as e:
        raise e from None
    for key, val in options.items():
        if key not in OPTIONS:
            raise KeyError(f'Invalid config option: {key}')
        _runtime_options[key] = OPTIONS[key].coerce(val)


def restore_defaults(keys: list[str] = ALL):
    """Restore any number of config options to their default values. Restore all by default."""
    if keys is ALL:
        _runtime_options.clear()
        return
    for key in keys:
        _runtime_options.pop(key)


def get_user_config_path() -> Path | None:
    """Resolve the path for user config file. Read from envvar with fixed default.

    Return ``None`` if the config does not exist.
    """
    override = os.getenv(USER_CONFIG_ENVVAR)
    if override:
        res = Path(override).expanduser()
        if not res.exists():
            raise FileNotFoundError(f'User config file read from USER_CONFIG_ENVVAR does not exist: {res}')
    else:
        res = DEFAULT_USER_CONFIG_PATH

    if not res.exists():
        return None
    return res


def get_local_config_path() -> Path | None:
    """Resolve the path for local config file.

    Return ``None`` if the config does not exist.
    """
    p = Path.cwd() / LOCAL_CONFIG_PATH
    if not p.exists():
        return None
    return p


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load config from a yaml file. Includes verification"""
    if not path.exists():
        return {}

    with open(path, 'r') as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise TypeError(f'Config file must contain a mapping: {path}')

    return validate_options(data)


def load_envvar_options() -> dict[str, Any]:
    """Load config from envvars. Includes verification."""
    res = {}
    for key, opt in OPTIONS.items():
        val = os.getenv(opt.env_var)
        if val is None:
            continue
        try:
            val = opt.coerce(val)
        except Exception as e:
            warnings.warn(f'Invalid config option in envvar {opt.env_var}. Reason {e}')
            continue
        res[key] = val
    return res


def init_config(reinit=False):
    """Hook to load config during library startup.

    Should be called early in ``cyten/__init__.py``.
    """
    if reinit:
        _user_file_options.clear()
        _local_file_options.clear()
        _envvar_options.clear()
        _runtime_options.clear()
        global _temp_context_options
        _temp_context_options = ContextVar('_temp_context_options', default={})

    # load user config file
    assert len(_user_file_options) == 0
    user_config_path = get_user_config_path()
    if user_config_path is not None:
        try:
            user_config = load_yaml_config(user_config_path)
        except Exception as e:
            warnings.warn(f'Invalid config in {user_config_path}. Ignoring the file. Reason: {e}')
            user_config = {}
        _user_file_options.update(user_config)

    # load local config gile
    assert len(_local_file_options) == 0
    local_config_path = get_local_config_path()
    if local_config_path is not None:
        try:
            local_config = load_yaml_config(local_config_path)
        except Exception as e:
            warnings.warn(f'Invalid config in {local_config_path}. Ignoring the file. Reason: {e}')
            local_config = {}
        _local_file_options.update(local_config)

    # load envvar config
    assert len(_envvar_options) == 0
    _envvar_options.update(load_envvar_options())


def get_option(name: str) -> Any:
    """Resolve the value of a given config option."""
    option = OPTIONS.get(name, None)
    if option is None:
        raise KeyError(f'Invalid option name: {name}')

    for source in [
        _temp_context_options.get(),
        _runtime_options,
        _local_file_options,
        _user_file_options,
        _envvar_options,
    ]:
        val = source.get(name, None)
        if val is not None:
            return val
    return option.default
