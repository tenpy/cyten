# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = build_docs
REPO_ROOT     := $(abspath $(SOURCEDIR)/..)

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile stubs doxygen-xml

# Generate cyten/_core.py for autodoc when the C++ extension is not installed.
stubs:
	@python "$(REPO_ROOT)/scripts/generate_core_stubs.py" -o "$(REPO_ROOT)/cyten/_core.py"

doxygen-xml:
	@doxygen

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
