#!/usr/bin/env python3
"""Mechanical Phase C signature migration for cyten HDF5 APIs."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path("/home/johannes/work/tenpy/cyten_main")

# Signature replacements in headers/sources
PATTERNS = [
    # virtual / normal save_hdf5 with py::object
    (
        re.compile(
            r"void save_hdf5\(py::object hdf5_saver,\s*\n\s*py::object h5gr,\s*\n\s*(?:const )?std::string(?: const)?& subpath\)"
        ),
        "void save_hdf5(cyten::hdf5::Saver& saver,\n"
        "                       HighFive::Group& h5gr,\n"
        "                       std::string const& subpath)",
    ),
    (
        re.compile(
            r"void save_hdf5\(py::object hdf5_saver, py::object h5gr, (?:const )?std::string(?: const)?& subpath\)"
        ),
        "void save_hdf5(cyten::hdf5::Saver& saver, HighFive::Group& h5gr, std::string const& subpath)",
    ),
    (
        re.compile(
            r"void save_hdf5\(py::object hdf5_saver, py::object h5gr, std::string subpath\)"
        ),
        "void save_hdf5(cyten::hdf5::Saver& saver, HighFive::Group& h5gr, std::string subpath)",
    ),
    # from_hdf5 common
    (
        re.compile(
            r"from_hdf5\(py::object hdf5_loader,\s*\n\s*py::object h5gr,\s*\n\s*(?:const )?std::string(?: const)?& subpath\)"
        ),
        "from_hdf5(cyten::hdf5::Loader& loader,\n"
        "                                    HighFive::Group& h5gr,\n"
        "                                    std::string const& subpath)",
    ),
    (
        re.compile(
            r"from_hdf5\(py::object hdf5_loader, py::object h5gr, (?:const )?std::string(?: const)?& subpath\)"
        ),
        "from_hdf5(cyten::hdf5::Loader& loader, HighFive::Group& h5gr, std::string const& subpath)",
    ),
    (
        re.compile(
            r"from_hdf5\(py::object hdf5_loader, py::object h5gr, std::string subpath\)"
        ),
        "from_hdf5(cyten::hdf5::Loader& loader, HighFive::Group& h5gr, std::string subpath)",
    ),
    # definitions with unused names
    (
        re.compile(r"py::object hdf5_saver"),
        "cyten::hdf5::Saver& saver",
    ),
    (
        re.compile(r"py::object /\*hdf5_saver\*/"),
        "cyten::hdf5::Saver& /*saver*/",
    ),
    (
        re.compile(r"py::object hdf5_loader"),
        "cyten::hdf5::Loader& loader",
    ),
    (
        re.compile(r"py::object /\*hdf5_loader\*/"),
        "cyten::hdf5::Loader& /*loader*/",
    ),
]

INCLUDE_LINE = "#include <cyten/tools/hdf5.h>\n"


def needs_include(text: str) -> bool:
    return "cyten::hdf5::" in text or "HighFive::Group& h5gr" in text


def process_file(path: Path) -> bool:
    text = path.read_text()
    if "save_hdf5" not in text and "from_hdf5" not in text and "load_hdf5" not in text:
        return False
    orig = text
    for cre, repl in PATTERNS:
        text = cre.sub(repl, text)

    # Fix leftover py::object h5gr in hdf5 signatures only when Saver/Loader already migrated nearby
    if "cyten::hdf5::Saver&" in text or "cyten::hdf5::Loader&" in text:
        text = re.sub(
            r"(save_hdf5\([^)]*?),\s*py::object h5gr,",
            r"\1, HighFive::Group& h5gr,",
            text,
        )
        text = re.sub(
            r"(from_hdf5\([^)]*?),\s*py::object h5gr,",
            r"\1, HighFive::Group& h5gr,",
            text,
        )
        text = re.sub(
            r"(load_hdf5_common\([^)]*?),\s*py::object h5gr,",
            r"\1, HighFive::Group& h5gr,",
            text,
        )
        text = text.replace(
            "void load_hdf5_common(py::object hdf5_loader, HighFive::Group& h5gr,",
            "void load_hdf5_common(cyten::hdf5::Loader& loader, HighFive::Group& h5gr,",
        )
        text = text.replace("py::object hdf5_loader", "cyten::hdf5::Loader& loader")
        text = text.replace("py::object /*h5gr*/", "HighFive::Group& /*h5gr*/")
        text = text.replace(", py::object /*h5gr*/,", ", HighFive::Group& /*h5gr*/,")

    if text == orig:
        return False

    if needs_include(text) and "#include <cyten/tools/hdf5.h>" not in text:
        # Insert after last include
        lines = text.splitlines(keepends=True)
        last_inc = 0
        for i, line in enumerate(lines):
            if line.startswith("#include"):
                last_inc = i
        lines.insert(last_inc + 1, INCLUDE_LINE)
        text = "".join(lines)

    path.write_text(text)
    return True


def main() -> None:
    changed = []
    for folder in ("include/cyten", "src"):
        for path in (ROOT / folder).rglob("*"):
            if path.suffix not in {".h", ".cpp", ".hpp"}:
                continue
            if process_file(path):
                changed.append(str(path.relative_to(ROOT)))
    print(f"changed {len(changed)} files")
    for c in changed:
        print(c)


if __name__ == "__main__":
    main()
