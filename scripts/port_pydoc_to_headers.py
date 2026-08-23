#!/usr/bin/env python3
"""Port R\"pydoc bindings into header /// + DOC() macros.

Usage:
  python3 scripts/port_pydoc_to_headers.py \\
      --header include/cyten/tensors/sparse.h \\
      --pybind pybind/tensors/py_sparse.cpp \\
      --classes LinearOperator,TensorLinearOperator,...
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


ROLE_RE = re.compile(r":(?:class|meth|func|attr|mod|exc|data|ref|any|cfg:option|cfg:config):`~?([^`]+)`")
MATH_RE = re.compile(r":math:`([^`]+)`")
CFG_BLOCK_RE = re.compile(r"(?ms)^\.\. cfg:config\s*::.*?(?=^\S|\Z)")


def convert_body(body: str, *, strip_cfg: bool = True) -> str:
    lines = body.strip("\n").splitlines()
    indents = [len(l) - len(l.lstrip(" ")) for l in lines if l.strip()]
    if indents:
        m = min(indents)
        lines = [l[m:] if len(l) >= m else l for l in lines]
    text = "\n".join(lines)

    def role_sub(m: re.Match[str]) -> str:
        inner = m.group(1)
        if "." in inner:
            inner = inner.split(".")[-1]
        return f"`{inner}`"

    text = ROLE_RE.sub(role_sub, text)
    text = MATH_RE.sub(r"@f$ \1 @f$", text)
    if strip_cfg:
        text = CFG_BLOCK_RE.sub("", text)
        text = re.sub(r"(?m)^\.\. warning\s*::.*?(?=^\S|\Z)", "", text, flags=re.S)

    out: list[str] = []
    i = 0
    lines = text.splitlines()
    mode: str | None = None
    while i < len(lines):
        line = lines[i]
        if (
            line.strip()
            in (
                "Parameters",
                "Returns",
                "Other Parameters",
                "Raises",
                "Notes",
                "See Also",
                "Examples",
                "Warns",
                "Attributes",
                "Options",
            )
            and i + 1 < len(lines)
            and set(lines[i + 1].strip()) <= {"-", "="}
            and len(lines[i + 1].strip()) >= 3
        ):
            mode = line.strip()
            i += 2
            if mode in ("Notes", "Attributes", "Options"):
                out.append("")
                out.append(f"{mode}:")
                out.append("")
            continue

        if mode == "Parameters":
            if not line.strip():
                out.append("")
                i += 1
                continue
            if line[:1] not in " \t":
                name = line.split(":", 1)[0].strip()
                desc: list[str] = []
                i += 1
                while i < len(lines) and lines[i][:1] in " \t":
                    desc.append(lines[i].strip())
                    i += 1
                out.append(f"@param {name} {' '.join(desc)}".rstrip())
                continue
            mode = None

        if mode == "Returns":
            if not line.strip():
                out.append("")
                i += 1
                continue
            desc_lines: list[str] = []
            if i + 1 < len(lines) and lines[i + 1][:1] in " \t" and ":" not in line:
                i += 1
            while i < len(lines) and lines[i].strip():
                if (
                    lines[i].strip()
                    in (
                        "Parameters",
                        "Returns",
                        "Notes",
                        "Raises",
                        "See Also",
                        "Examples",
                        "Attributes",
                        "Options",
                    )
                    and i + 1 < len(lines)
                    and set(lines[i + 1].strip()) <= {"-", "="}
                ):
                    break
                desc_lines.append(lines[i].strip())
                i += 1
            out.append("@returns " + " ".join(desc_lines))
            mode = None
            continue

        if mode in ("Notes", "Attributes", "Options", "Raises", "See Also", "Examples", "Warns"):
            if (
                line.strip()
                in (
                    "Parameters",
                    "Returns",
                    "Notes",
                    "Raises",
                    "See Also",
                    "Examples",
                    "Attributes",
                    "Options",
                )
                and i + 1 < len(lines)
                and set(lines[i + 1].strip()) <= {"-", "="}
            ):
                mode = None
                continue
            out.append(line)
            i += 1
            continue

        out.append(line)
        i += 1

    text = "\n".join(out).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def to_slash(body: str) -> str:
    if not body.strip():
        return "///"
    return "\n".join(("/// " + l) if l.strip() else "///" for l in body.splitlines())


def preceding_doc_span(header: str, decl_pos: int) -> tuple[int, int] | None:
    i = decl_pos
    while i > 0 and header[i - 1] in " \t":
        i -= 1
    if i > 0 and header[i - 1] == "\n":
        i -= 1
    starts: list[int] = []
    while i >= 0:
        ls = header.rfind("\n", 0, i) + 1
        le = header.find("\n", ls)
        if le < 0:
            le = len(header)
        full = header[ls:le]
        if full.lstrip().startswith("///"):
            starts.append(ls)
            i = ls - 1
            continue
        break
    if not starts:
        return None
    decl_line = header.rfind("\n", 0, decl_pos) + 1
    return min(starts), decl_line


def set_doc_before(header: str, decl_pos: int, doc_body: str) -> str:
    new_doc = to_slash(doc_body) + "\n"
    span = preceding_doc_span(header, decl_pos)
    if span:
        return header[: span[0]] + new_doc + header[span[1] :]
    insert_at = header.rfind("\n", 0, decl_pos) + 1
    return header[:insert_at] + new_doc + header[insert_at:]


def find_class_pos(header: str, class_name: str) -> int | None:
    m = re.search(rf"(?m)^class {re.escape(class_name)}\b", header)
    return m.start() if m else None


def find_method_pos(header: str, class_name: str, method: str) -> int | None:
    cpos = find_class_pos(header, class_name)
    if cpos is None:
        return None
    # next class boundary
    nxt = re.search(r"(?m)^class \w+\b", header[cpos + 1 :])
    end = cpos + 1 + nxt.start() if nxt else len(header)
    region = header[cpos:end]
    m = re.search(rf"(?m)^(?![/\s]*//).*?\b{re.escape(method)}\s*\(", region)
    if not m:
        return None
    return cpos + m.start()


def find_free_fn_pos(header: str, name: str) -> int | None:
    # prefer free functions outside classes: rough heuristic — last occurrence before EOF
    # that is not indented as a method... methods are indented. Free fns start at column 0-ish.
    for m in re.finditer(rf"(?m)^(?:\[\[.*?\]\]\s*)?(?:[\w:<>&\s\*]+)\b{re.escape(name)}\s*\(", header):
        # skip if inside a class roughly: look back for unmatched class
        before = header[: m.start()]
        if before.count("{") > before.count("}"):
            # could be inside namespace or class; check last 'class X' vs closing
            last_class = list(re.finditer(r"(?m)^class \w+\b", before))
            if last_class:
                # if braces after that class still open, skip
                after_class = before[last_class[-1].start() :]
                if after_class.count("{") > after_class.count("}"):
                    continue
        return m.start()
    return None


def extract_class_docs(py: str) -> dict[str, str]:
    """Map C++ class name -> pydoc body from `Name.doc() = R\"pydoc\"` near class_<Name>."""
    result: dict[str, str] = {}
    # Find class_<..., "PythonName"> or class_<Name,...>(m, "PythonName")
    for m in re.finditer(
        r'py::class_<(?:[^>]*?\b)?(\w+)(?:\s*,[^>]*)?>\s*\([^)]*?,\s*"(\w+)"\s*\)',
        py,
        re.S,
    ):
        cpp_name, py_name = m.group(1), m.group(2)
        # find .doc() after this within next 800 chars... actually search for var.doc
        # Better: find `xxx.doc() = R"pydoc` where xxx was assigned this class_
        # Look at assignment: `py::class_<...> name(` or `auto name = py::class_`
        # Simpler approach: search for `.doc() = R"pydoc` and associate with nearest prior class_ name string
        pass

    for m in re.finditer(r'(\w+)\.doc\(\)\s*=\s*R"pydoc\((.*?)\)pydoc"', py, re.S):
        var, body = m.group(1), m.group(2)
        before = py[max(0, m.start() - 800) : m.start()]
        cm = list(
            re.finditer(
                r'py::class_<(?:[^>]*?\b)?(\w+)(?:\s*,[^>]*)?>\s*(?:\n\s*)?\([^)]*?,\s*"(\w+)"\s*\)',
                before,
                re.S,
            )
        )
        if not cm:
            # try `auto foo = py::class_<Bar,...>(m, "Bar")`
            cm = list(
                re.finditer(
                    r'class_<(?:[^>]*?\b)?(\w+)(?:\s*,[^>]*)?>\s*\([^)]*?,\s*"(\w+)"\s*\)',
                    before,
                    re.S,
                )
            )
        if cm:
            result[cm[-1].group(1)] = body
            result[cm[-1].group(2)] = body  # also by python name
        else:
            # fallback: var name CamelCase guess
            result[var] = body
    return result


def extract_method_docs(py: str) -> list[tuple[str, str, str, tuple[int, int]]]:
    """Return list of (class_cpp, method, body, span) for R\"pydoc on defs."""
    # Build map from binding variable -> class name by scanning class_ constructions
    var_to_class: dict[str, str] = {}
    for m in re.finditer(
        r'(?:auto\s+)?(\w+)\s*=\s*py::class_<(?:[^>]*?\b)?(\w+)(?:\s*,[^>]*)?>|'
        r'py::class_<(?:[^>]*?\b)?(\w+)(?:\s*,[^>]*)?>\s+(\w+)\s*\(',
        py,
        re.S,
    ):
        if m.group(1) and m.group(2):
            var_to_class[m.group(1)] = m.group(2)
        elif m.group(3) and m.group(4):
            var_to_class[m.group(4)] = m.group(3)

    # Also: py::class_<X,...> var(
    for m in re.finditer(
        r'py::class_<(?:[^>]*?\b)?(\w+)(?:\s*,[^>]*)?>\s+(\w+)\s*\(',
        py,
    ):
        var_to_class[m.group(2)] = m.group(1)

    out: list[tuple[str, str, str, tuple[int, int]]] = []
    for m in re.finditer(r'R"pydoc\((.*?)\)pydoc"', py, re.S):
        start, end = m.span()
        before = py[max(0, start - 600) : start]
        if re.search(r"\.doc\(\)\s*=\s*$", before.rstrip()) or re.search(
            r"\.doc\(\)\s*=\s*\n\s*$", before
        ):
            continue
        mm = list(
            re.finditer(
                r'(\w+)\.(def_static|def_property_readonly|def_property|def|def_readwrite)\(\s*"([^"]+)"',
                before,
            )
        )
        if not mm:
            # free function m.def("name"
            fm = list(re.finditer(r'\bm\.def\(\s*"([^"]+)"', before))
            if fm:
                out.append(("", fm[-1].group(1), m.group(1), m.span()))
            continue
        var, _kind, name = mm[-1].group(1), mm[-1].group(2), mm[-1].group(3)
        cls = var_to_class.get(var, "")
        out.append((cls, name, m.group(1), m.span()))
    return out


def port(header_path: Path, pybind_path: Path, classes: list[str]) -> None:
    header = header_path.read_text()
    py = pybind_path.read_text()

    class_docs = extract_class_docs(py)
    method_docs = extract_method_docs(py)

    # Update class docs in header
    for cls in classes:
        body = class_docs.get(cls)
        if not body:
            continue
        pos = find_class_pos(header, cls)
        if pos is None:
            print(f"  WARN: class {cls} not in header")
            continue
        converted = convert_body(body)
        header = set_doc_before(header, pos, converted)
        print(f"  class {cls}: updated ///")

    # Update method / free fn docs (reverse order by position in header)
    header_updates: list[tuple[int, str, str]] = []
    for cls, name, body, _span in method_docs:
        converted = convert_body(body)
        if cls:
            pos = find_method_pos(header, cls, name)
            label = f"{cls}::{name}"
        else:
            pos = find_free_fn_pos(header, name)
            label = name
        if pos is None:
            print(f"  WARN: no decl for {label}")
            continue
        header_updates.append((pos, converted, label))

    # unique by position (keep first = last in file order preference: last wins)
    by_pos: dict[int, tuple[str, str]] = {}
    for pos, converted, label in header_updates:
        by_pos[pos] = (converted, label)
    for pos in sorted(by_pos.keys(), reverse=True):
        converted, label = by_pos[pos]
        header = set_doc_before(header, pos, converted)
        print(f"  method {label}: updated ///")

    # Rewrite pybind docs
    rel = str(header_path.relative_to("include/cyten"))
    if f'docstrings/{rel}"' not in py and f"docstrings/{rel}'" not in py:
        if '#include "../doc_plus.h"' not in py:
            py = py.replace(
                '#include "../py_cyten_pybind11.h"',
                '#include "../doc_plus.h"\n#include "../py_cyten_pybind11.h"',
            )
        insert_after = '#include "../py_cyten_pybind11.h"\n'
        if insert_after in py and f'#include "docstrings/{rel}"' not in py:
            py = py.replace(
                insert_after,
                insert_after + f'\n#include "docstrings/{rel}"\n',
            )

    # Replace class docs
    def repl_class_doc(m: re.Match[str]) -> str:
        var = m.group(1)
        before = py[max(0, m.start() - 800) : m.start()]
        cm = list(
            re.finditer(
                r'class_<(?:[^>]*?\b)?(\w+)(?:\s*,[^>]*)?>',
                before,
            )
        )
        cls = cm[-1].group(1) if cm else None
        if not cls:
            return m.group(0)
        print(f"  pybind {var}.doc -> DOC(cyten, {cls})")
        return f"{var}.doc() = DOC(cyten, {cls})"

    py = re.sub(r'(\w+)\.doc\(\)\s*=\s*R"pydoc\(.*?\)pydoc"', repl_class_doc, py, flags=re.S)

    # Replace method/free R"pydoc — rebuild from extract with correct DOC
    # Process spans reverse
    pieces: list[str] = []
    last = len(py)
    # Re-extract on current py (class docs already replaced)
    method_docs2 = extract_method_docs(py)
    for cls, name, _body, (start, end) in sorted(method_docs2, key=lambda x: x[3][0], reverse=True):
        if cls:
            doc_expr = f"DOC(cyten, {cls}, {name})"
        else:
            doc_expr = f"DOC(cyten, {name})"
        print(f"  pybind DOC <- {doc_expr}")
        py = py[:start] + doc_expr + py[end:]

    header_path.write_text(header)
    pybind_path.write_text(py)
    print(f"Wrote {header_path} and {pybind_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--header", type=Path, required=True)
    ap.add_argument("--pybind", type=Path, required=True)
    ap.add_argument("--classes", type=str, required=True, help="Comma-separated C++ class names")
    args = ap.parse_args()
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    print(f"Porting {args.pybind} -> {args.header} classes={classes}")
    port(args.header, args.pybind, classes)


if __name__ == "__main__":
    main()
