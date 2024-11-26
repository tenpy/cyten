"""Tools for handling strings."""
# Copyright (C) TeNPy Developers, Apache license

__all__ = ['format_like_list']


def format_like_list(it) -> str:
    """Format elements of an iterable as if it were a plain list.

    This means surrounding them with brackets and separating them by `', '`."""
    return f'[{", ".join(map(str, it))}]'
