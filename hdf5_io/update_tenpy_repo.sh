#!/bin/bash
# This file copies files from this repo into (a local copy of) the TeNPy git repository (specified as argument).

set -e   # abort on error

if [[ $# -ne 1 ]] || [[ "$1" == "-h" ]] || [[ ! -f "$1/tenpy/__init__.py" ]]
then
	echo -e "usage: $0 tenpy_repo\n\nHere, 'tenpy_repo' points to a local tenpy git repository, to which the files from here should be copied."
	exit 1
fi

prompt_continue() {
	read -p "$1 " ANS
	if [[ "$ANS" != "y" ]]
	then
		exit 1
	fi
}

THISREPO="$(dirname "$(readlink -f "$0")")"
TENPYREPO="$1"
REPOSRCFILE="tenpy/tools/hdf5_io.py"
REPODOCFILE="doc/intro/input_output.rst"

cd "$TENPYREPO"
if test -n "$(git status -s $REPOSRCFILE $REPODOCFILE )" 
then
	echo "Erorr: git repository not clean; afraid to overwrite something."
	exit 1 
fi

cp "$THISREPO/src/python3/hdf5_io.py" "$TENPYREPO/$REPOSRCFILE"
cp "$THISREPO/doc/input_output.rst" "$TENPYREPO/$REPODOCFILE"
prompt_continue "Copied files. Continue [y/n]?"

git diff $REPOSRCFILE $REPODOCFILE

prompt_continue "git commit these changes [y/n]?"

git add $REPOSRCFILE $REPODOCFILE
git commit -m "Merged files from hdf5_io repository" $REPOSRCFILE $REPODOCFILE
