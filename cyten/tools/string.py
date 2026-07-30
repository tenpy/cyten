"""Tools for handling strings."""
# Copyright (C) TeNPy Developers, Apache license

from .._core import format_like_list  # noqa


def vert_join(strlist, valign='t', halign='l', delim=' '):
    r"""Join multiline strings vertically such that they appear next to each other.

    Parameters
    ----------
    strlist : list of str
        the strings to be joined vertically
    valign : ``'t', 'c', 'b'``
        vertical alignment of the strings: top, center, or bottom
    halign : ``'l', 'c', 'r'``
        horizontal alignment of the strings: left, center, or right
    delim : str
        field separator between the strings

    Returns
    -------
    joined : str
        a string where the strings of strlist are aligned vertically

    Examples
    --------
    >>> from tenpy.tools.string import vert_join
    >>> print(
    ...     vert_join(['a\nsample\nmultiline\nstring', str(np.arange(9).reshape(3, 3))], delim=' | ')
    ... )  # doctest: +NORMALIZE_WHITESPACE
    a         | [[0 1 2]
    sample    |  [3 4 5]
    multiline |  [6 7 8]]
    string    |

    """
    # expand tabs, split to newlines
    strlist = [str(s).expandtabs().split('\n') for s in strlist]
    numstrings = len(strlist)
    # number of lines in each string
    numlines = [len(lines) for lines in strlist]
    # maximum number of lines = total number of lines in the resulting string
    totallines = max([0] + numlines)
    # width for each of the strings
    widths = [max([len(l) for l in lines]) for lines in strlist]
    # translate halign to string format mini language
    halign = {'l': '<', 'c': '^', 'r': '>'}[halign]
    fstr = ['{0: ' + halign + str(w) + 's}' for w in widths]

    # create a 2d table
    res = [[' ' * widths[j] for j in range(numstrings)] for i in range(totallines)]

    for j, lines in enumerate(strlist):
        if valign == 't':
            voffset = 0
        elif valign == 'b':
            voffset = totallines - len(lines)
        elif valign == 'c':
            voffset = (totallines - len(lines)) // 2  # rounds to int
        else:
            raise ValueError('invalid valign ' + str(valign))

        for i, l in enumerate(lines):
            res[i + voffset][j] = fstr[j].format(l)  # format to fixed widths[j]

    # convert the created table to a single string
    res = '\n'.join([delim.join(lines) for lines in res])
    return res
