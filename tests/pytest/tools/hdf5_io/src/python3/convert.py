#!/usr/bin/env python3
"""Convert the format of hdf5 files.

.. note ::
    This module is maintained in the repository https://github.com/tenpy/hdf5_io.git

Specify from which format into which format you want to convert with the ``--from-format``
and ``--to-format`` command line arguments. We try to load the corresponding conversion
from the file ``convert_from_{FROM_FORMAT}_to_{TO_FORMAT}.py``.
"""
# Try command line argument ``--help`` for options.

import argparse

import h5py
import hdf5_io
from hdf5_io import ATTR_TYPE, ATTR_CLASS, ATTR_MODULE, REPR_HDF5EXPORTABLE, REPR_IGNORED
import importlib
import warnings

BACKUP_PATH = "/backup_pre_convert"  #: path of (temporary) group inside the file
# to move groups to be converted into.

ATTR_ORIG_PATH = "original_path"  #: Attribute name for path before moving a group to BACKUP_PATH
# for conversion

__all__ = [
    'BACKUP_PATH', 'ATTR_ORIG_PATH', 'HDF5Converter', 'parse_args', 'find_converter', 'main'
]


class Hdf5Converter(hdf5_io.Hdf5Loader, hdf5_io.Hdf5Saver):
    """Hdf5Converter base class.

    Prototype for a converted, needs to be subclasses for specific conversion from
    one format into another.
    To make :func:`find_converter` work, subclasses should be implemented in separate files called
    ``'convert_from_{library1}_to_{library2}.py'``.

    Parameters
    ----------
    filename : str
        The name of the hdf5 file to open.
    no_backup : bool
        Whether to keep (False, default) or remove (True) the backups of the converted groups.
    verbose : int
        Verbosity level, i.e., how much to print about what the converter does.

    Attributes
    ----------
    no_backup : bool
        See above.
    verbose : int
        See above
    memo_convert : dict
        A dictionary to remember all the HDF5 groups which we already converted.
        The dictionary key is the h5py group- or dataset ``id``;
        the value is the converted hdf5 group. See :meth:`memorize_convert`.
    mappings : dict
        The structure of the dictionary is ``{convert_from : (convert_to, map_function)}``.
        Here, `convert_from` and `convert_to` define objects of which types/classes should be
        converted into which types/classes. Both of them can be either a single string,
        which just stands for the content of the ``'type'`` attribute,
        or a tuple ``(modulename, classname)`` for class instances with the
        ``'type'`` attribute set to ``'instance'``.
        The `map_function` gets called by :meth:`convert_group` as
        ``map_function(self, h5gr_orig, subpath_orig, h5gr_new, subpath_new)``
        with the old hdf5 group `h5gr_old` moved into :attr:`backup_gr`,  the new group `h5gr_new`
        to take the the converted data, and `subpath_orig` and `subpath_new` just being the names
        of the group with a ``'/'`` in the end to allow easy loading/saving of group members.
        The attributes for type, class and module are set to `h5gr_new` according to `convert_to`;
        the `map_function` only needs to convert the data. It can use :meth:`load` and :meth:`save`
        for the conversion, but should make copies before changing loaded objects.
    backup_gr : :class:`Group`
        Hdf5 group into which groups are moved for conversion.
    """
    def __init__(self, filename, no_backup=False, verbose=0):
        if not (filename.endswith('.h5') or filename.endswith('.hdf5')):
            raise ValueError("unexpected filename ending")
        h5file = h5py.File(filename, 'r+')
        hdf5_io.Hdf5Loader.__init__(self, h5file)
        hdf5_io.Hdf5Saver.__init__(self, h5file)
        self.no_backup = no_backup
        if not hasattr(self, 'mappings'):
            raise NotImplementedError("subclass didn't define self.mappings")
        self.memo_convert = {}
        self.verbose = verbose

        if BACKUP_PATH not in self.h5group:
            self.backup_gr = self.h5group.create_group(BACKUP_PATH)
            self.backup_gr.attrs[ATTR_TYPE] = REPR_IGNORED
        else:
            self.backup_gr = backup_gr = self.h5group[BACKUP_PATH]
            for subgr in backup_gr.values():
                self.memorize_convert(subgr, subgr)
        self.memorize_convert(self.backup_gr, self.backup_gr)  # exclude conversion

    def memorize_convert(self, h5gr, h5gr_converted):
        """Store groups already converted in the :attr:`memo_convert`.

        Necessary to avoid loops in the recursion for cyclic references, and to avoid converting
        the same group twice if hardlinks are used.

        Parameters
        ----------
        h5gr : :class:`Group` | :class:`Dataset`
            The h5py group or dataset before conversion.
        h5gr_converted : :class:`Group` | :class:`Dataset`
            The h5py group or dataset after conversion.
        """
        self.memo_convert[h5gr.id] = h5gr_converted

    def convert_file(self):
        """Convert all groups of the file."""
        if self.verbose:
            if self.verbose > 1:
                print("=" * 80)
            print("converting file", self.h5group.filename)

        # set top-level group "/" of the file to be a dictionary, if not specified before.
        if not hdf5_io.ATTR_TYPE in self.h5group.attrs:
            self.h5group.attrs[hdf5_io.ATTR_TYPE] = hdf5_io.REPR_DICT_SIMPLE

        self.convert_group(self.h5group)  # recursive

        # now (once that everything worked, not in __del__) we can delete the backup
        if self.no_backup:
            if self.verbose > 1:
                print("remove backup group", self.backup_gr.name)
            del self.h5group[self.backup_gr.name]

    def convert_group(self, h5gr):
        """Convert a given group, and recursively any subgroups.

        Only groups with the ``'type'`` attribute being REPR_HDF5EXPORTABLE get converted,
        if there is a mapping for the corresponding module and class name combination defined
        in :attr:`mappings`. The latter should be defined by subclasses, see the class doc-string.

        Parameters
        ----------
        h5gr : :class:`Group` | :class:`Dataset`
            The h5py group or dataset to be converted.

        Returns
        -------
        h5gr_new : :class:`Group` | :class:`Dataset`
            The converted h5py group or dataset.
        """
        if self.verbose > 4:
            print('called convert_group for', repr(h5gr.name))
        in_memo = self.memo_convert.get(h5gr.id)
        if in_memo is not None:
            return in_memo  # already converted
        self.memorize_convert(h5gr, h5gr)  # avoid endless recursion for cyclic references

        type_ = h5gr.attrs.get(ATTR_TYPE)
        if type_ is None:  # no type_ defined
            warnings.warn("Ignore group with no type set.")
            if self.verbose > 1:
                print("dataset {0!r} has no attribute {1!r} defined".format(h5gr.name, ATTR_TYPE))
            return h5gr  # nothing to convert
        if isinstance(type_, bytes):
            type_ = type_.decode()
        if type_ == REPR_HDF5EXPORTABLE:
            module_name = self.get_attr(h5gr, ATTR_MODULE)
            class_name = self.get_attr(h5gr, ATTR_CLASS)
            convert_from = (module_name, class_name)
            mapping = self.mappings.get(convert_from)
        elif type_ == REPR_IGNORED:
            return h5gr  # nothing to do
        else:
            convert_from = type_
            mapping = self.mappings.get(convert_from)
        if mapping is None:
            # no mapping defined, so we simply convert subgroups
            if self.verbose > 3:
                print("no mapping for dataset {0!r} from {1!r}".format(h5gr.name, convert_from))
            self.convert_subgroups(h5gr)
            return h5gr  # nothing to convert
        convert_to, map_function = mapping
        # move to backup_gr and create new group
        h5gr_orig, subpath_orig, h5gr_new, subpath_new = self.move_backup(h5gr)
        if self.verbose > 1:
            print("converting group {0!r} from {1!s} to {2!s}, backup {3!s}".format(
                h5gr_new.name, convert_from, convert_to, h5gr_orig.name))
        elif self.verbose:
            print("converting group", subpath_new)
        if isinstance(convert_to, tuple):
            new_module_name, new_class_name = convert_to
            h5gr_new.attrs[ATTR_TYPE] = REPR_HDF5EXPORTABLE
            h5gr_new.attrs[ATTR_MODULE] = new_module_name
            h5gr_new.attrs[ATTR_CLASS] = new_class_name
        else:
            h5gr_new.attrs[ATTR_TYPE] = convert_to
        self.memorize_convert(h5gr, h5gr_new)  # update memo
        map_function(self, h5gr_orig, subpath_orig, h5gr_new, subpath_new)  # convert data
        return h5gr_new

    def convert_subgroups(self, h5gr):
        """Call :meth:`convert_group` for any subgroups of `h5gr`."""
        if not isinstance(h5gr, h5py.Group):
            return  # nothing to do, in particular for h5py.Dataset
        for subgr in list(h5gr.values()):
            self.convert_group(subgr)
        # done

    def move_backup(self, h5gr):
        """Move `h5gr` into :attr:`backup` group."""
        orig_path = h5gr.name
        backup_gr = self.backup_gr
        counter = len(backup_gr)
        moved_path = '/'.join([backup_gr.name, str(counter)])
        while moved_path in h5gr:
            counter += 1
            moved_path = '/'.join([backup_gr.name, str(counter)])
        if orig_path != '/':
            self.h5group.move(orig_path, moved_path)  # move to backup
            h5gr_orig = h5gr  # now points to the moved group in the backup
            subpath_orig = h5gr_orig.name + '/'
            h5gr.attrs[ATTR_ORIG_PATH] = orig_path
        else:  # special case of '/' group: can't move it!
            # move subgroups of '/' to backup (except the backup group)
            backup_gr_top = backup_gr.name.split('/')[1]
            h5gr_orig = backup_gr.create_group(moved_path)
            for subgr_name in h5gr:
                if subgr_name != backup_gr_top:
                    self.h5group.move('/' + subgr_name, '/'.join([backup_gr.name, subgr_name]))
            # move attributes from h5gr to h5gr_orig
            for attr in list(h5gr.attrs.keys()):
                h5gr_orig.attrs[attr] = h5gr.attrs[attr]
                del h5gr.attrs[attr]
            h5gr_orig.attrs[ATTR_ORIG_PATH] = orig_path
            # self.create_group_for_obj() handles the case ``orig_path == '/'`` correctly
        h5gr_new, subpath_new = self.create_group_for_obj(orig_path, h5gr)  # make new group
        return h5gr_orig, subpath_orig, h5gr_new, subpath_new

    def recover_from_backup(self):
        """(Try to) recover the backup from :attr:`backup_gr`."""
        h5file = self.h5group
        backup_gr = self.backup_gr
        if self.verbose:
            print("recovering groups in file {0!r} from backup under {1!r}".format(
                h5file.filename, backup_gr.name))
        names = sorted(backup_gr.keys(), reverse=True)  # reverse: handle converted subgroups
        for name in names:
            gr = backup_gr[name]
            orig_path = gr.attrs[ATTR_ORIG_PATH]
            if self.verbose:
                print("recover original {0!r} from group {1!r}".format(orig_path, name))
            if orig_path == "/":
                # special case: "move" the group content to root
                backup_gr_top = backup_gr.name.split('/')[1]
                root_gr = h5file['/']
                # first clean up root_gr
                for subgr_name in list(root_gr.keys()):  # delete everything but the backup_gr
                    if subgr_name != backup_gr_top:
                        del root_gr[subgr_name]
                for attr in root_gr.attrs:
                    del root_gr.attrs[attr]
                # move groups from the backup `gr`
                for subgr_name in gr:
                    if subgr_name == backup_gr_top:
                        raise ValueError("trying to recover the backup group itself!?!")
                    gr.move(subgr_name, '/' + subgr_name)
                # copy attributes
                for attr in list(gr.attrs.keys()):
                    if attr != ATTR_ORIG_PATH:
                        h5gr_orig.attrs[attr] = h5gr.attrs[attr]
                # remove the `gr`
                del root_gr[gr.name]
                continue
            if orig_path not in h5file:
                warnings.warn("Expected a converted group under " + orig_path)
            else:
                del h5file[orig_path]
            h5file.move(gr.name, orig_path)
            del gr.attrs[ATTR_ORIG_PATH]

    def __del__(self):
        """Close the file properly when the Converter is deleted."""
        h5file = getattr(self, 'h5group', None)
        if h5file is not None:
            h5file.close()

    @staticmethod  # don't bind to self: allow subclasses to put it in `mappings`.
    def convert_Hdf5Exportable(self, h5gr_orig, subpath_orig, h5gr_new, subpath_new):
        """Convert a group representing an `Hdf5Exportable`."""
        # copy attributes except module, classname and type
        excl_attr_copies = set([ATTR_MODULE, ATTR_CLASS, ATTR_TYPE])
        for attr_name in set(h5gr_orig.attrs.keys()) - excl_attr_copies:
            h5gr_new.attrs[attr_name] = h5gr_orig.attrs[attr_name]
        # copy data sets
        for subgr in h5gr_orig:
            h5gr_new[subgr] = h5gr_orig[subgr]
        # and convert the subgroups
        self.convert_subgroups(h5gr_new)

    @classmethod
    def print_conversions(cls):
        """List from which class into which class conversions are done."""
        print(cls.__doc__)
        line = "{0!r:60} -> {1!r:60}"
        print(line.format("convert from type or (module, class)", "into type or (module, class)"))
        print("=" * 120)
        for from_, (to_, _) in cls.mappings.items():
            print(line.format(from_, to_))


def parse_args(converter_cls=None):
    doc = converter_cls.__doc__ if converter_cls is not None else __doc__
    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument(
        '-B',
        '--no-backup',
        action="store_true",  # default=False
        help="Don't keep copy of the converted Hdf5 groups inside the file "
        "under the path {0!r}".format(BACKUP_PATH))
    parser.add_argument('-r',
                        '--recover',
                        action="store_true",
                        help="(Try to) restore the groups from the backup under {0!r} "
                        "instead of converting something. "
                        "Make a backup of the whole file before attempting this!"
                        "".format(BACKUP_PATH))
    parser.add_argument('-l',
                        '--list-only',
                        action="store_true",
                        help="Don't convert anything, just list the classes which get converted.")
    parser.add_argument('-v',
                        '--verbose',
                        default=0,
                        action="count",
                        help="Print what we do; give multiple times for more status prints")
    if converter_cls is None:
        parser.add_argument('-f', '--from-format', help="From which format to convert")
        parser.add_argument('-t', '--to-format', help="Into which format to convert")
    parser.add_argument("files", nargs="*", help="Filenames of HDF5 files to convert (in place).")
    args = parser.parse_args()
    if converter_cls is not None:
        args.Converter = converter_cls
    return args


def find_converter(from_format, to_format):
    module_name = "convert_from_{0!s}_to_{1!s}".format(from_format, to_format)
    try:
        mod = importlib.import_module(module_name)
    except ImportError:
        raise ValueError("Conversion for these formats not possible:\n"
                         "Can't find module {0!r}".format(module_name)) from None
    return mod.Converter


def main(args):
    if 'Converter' in args:
        Converter = args.Converter
    else:
        Converter = find_converter(args.from_format, args.to_format)
    if args.list_only:
        Converter.print_conversions()
        return
    for fn in args.files:
        conv = Converter(fn, args.no_backup, args.verbose)
        if args.recover:
            conv.recover_from_backup()
        else:
            conv.convert_file()
    # done


if __name__ == "__main__":
    args = parse_args()
    main(args)
