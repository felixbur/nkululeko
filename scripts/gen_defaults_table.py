#!/usr/bin/env python3
"""Generate docs/source/config_defaults_reference.md from config_val*() call sites.

Prototype for GitHub issue #47 ("Hard-coded default values scattered across
model and feature files"). Rather than moving ~450 inline defaults into a
central nkululeko/defaults.py (a large, invasive refactor with real risk of
introducing typos/behavior drift), this script treats the existing
config_val(section, key, default) call sites as the source of truth and
statically extracts them via the ast module into a reference table, written
wholesale to its own file: docs/source/config_defaults_reference.md.

That file is kept separate from the hand-written, user-facing ini_file.md
on purpose: this table's audience is contributors auditing defaults, not
end users writing a config, and its raw call-site dumps (including internal
per-dataset keys like DATA.<self.name>) would be noise in that document.
ini_file.md links to it instead.

As a side effect, cross-referencing every call site by (section, key) also
surfaces keys where different call sites disagree on the default value.
Some of those are genuine bugs (e.g. one file hardcoding 'cpu' where every
other file defaults to a runtime GPU-detecting variable); others are
legitimate, context-dependent choices (e.g. an SVM and an SVR reasonably
using different default C values). Treat the "inconsistent defaults"
section as a list of candidates worth a human look, not a bug list.

Only git-tracked files under nkululeko/ are scanned, via `git ls-files`,
so untracked scratch/WIP files never pollute the generated reference.

Usage:
    python scripts/gen_defaults_table.py             # print the doc to stdout
    python scripts/gen_defaults_table.py --write      # (re)write the reference file
    python scripts/gen_defaults_table.py --check      # exit 1 if the reference file is stale (for CI)
"""

import argparse
import ast
import pathlib
import subprocess
import sys
from collections import defaultdict

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT_DOC = REPO_ROOT / "docs" / "source" / "config_defaults_reference.md"

# method name -> name of its first (namespace) positional param.
# config_val_data's first param is a dataset name, not an [EXP]-style
# section, but it always resolves to a DATA.<dataset>.<key> config entry.
CONFIG_VAL_METHODS = {
    "config_val": "section",
    "config_val_bool": "section",
    "config_val_list": "section",
    "config_val_data": "dataset",
}


def _render(node):
    """Render an AST node as (text, is_literal).

    Literal nodes (strings, numbers, booleans, None, lists...) are rendered
    via repr() of their evaluated value. Anything else (a variable, an
    expression, a nested config_val() call) is rendered as its source text
    and flagged as non-literal, since we can't resolve it without running
    the program.
    """
    try:
        return repr(ast.literal_eval(node)), True
    except (ValueError, TypeError):
        return ast.unparse(node), False


class ConfigValVisitor(ast.NodeVisitor):
    def __init__(self, relpath):
        self.relpath = relpath
        self.rows = []

    def visit_Call(self, node):
        self.generic_visit(node)
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in CONFIG_VAL_METHODS:
            return
        method = func.attr
        # All call sites in the codebase currently pass (namespace, key,
        # default) positionally; skip anything that doesn't match so a
        # future refactor to keyword args fails loud (missing row) rather
        # than silently mis-attributing arguments.
        if len(node.args) < 3 or node.keywords:
            return

        namespace_node, key_node, default_node = node.args[:3]
        namespace_text, namespace_is_literal = _render(namespace_node)
        key_text, key_is_literal = _render(key_node)
        default_text, default_is_literal = _render(default_node)

        if method == "config_val_data":
            dataset_label = namespace_text if namespace_is_literal else f"<{namespace_text}>"
            namespace = f"DATA.{dataset_label}"
        else:
            namespace = namespace_text if namespace_is_literal else f"<{namespace_text}>"
        key = key_text if key_is_literal else f"<{key_text}>"

        self.rows.append(
            {
                "method": method,
                "namespace": namespace,
                "key": key,
                "default": default_text,
                "default_is_literal": default_is_literal,
                "file": self.relpath,
                "line": node.lineno,
            }
        )


def tracked_python_files():
    """Return sorted repo-relative paths of git-tracked *.py files under nkululeko/."""
    result = subprocess.run(
        ["git", "ls-files", "--", "nkululeko/*.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return sorted(line for line in result.stdout.splitlines() if line)


def scan():
    rows = []
    for relpath in tracked_python_files():
        path = REPO_ROOT / relpath
        try:
            tree = ast.parse(path.read_text(), filename=relpath)
        except SyntaxError as e:
            print(f"warning: could not parse {relpath}: {e}", file=sys.stderr)
            continue
        visitor = ConfigValVisitor(relpath)
        visitor.visit(tree)
        rows.extend(visitor.rows)
    return rows


def group_by_namespace_key(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["namespace"], row["key"])].append(row)
    return grouped


def find_inconsistencies(grouped):
    """(namespace, key) pairs whose call sites disagree on the default value."""
    inconsistent = {}
    for ns_key, entries in grouped.items():
        defaults = {e["default"] for e in entries}
        if len(defaults) > 1:
            inconsistent[ns_key] = entries
    return inconsistent


def render_markdown(rows):
    """Render the inconsistency list and the full defaults table (body only)."""
    grouped = group_by_namespace_key(rows)
    inconsistent = find_inconsistencies(grouped)

    lines = []

    if inconsistent:
        lines.append(
            f"**⚠ {len(inconsistent)} key(s) have inconsistent defaults across call sites:**"
        )
        lines.append("")
        for (namespace, key), entries in sorted(inconsistent.items()):
            detail = ", ".join(f"`{e['default']}` ({e['file']}:{e['line']})" for e in entries)
            lines.append(f"* `{namespace}.{key}`: {detail}")
        lines.append("")

    lines.append("| Namespace | Key | Default | Source |")
    lines.append("|---|---|---|---|")
    for namespace, key in sorted(grouped.keys()):
        entries = grouped[(namespace, key)]
        default_vals = sorted({e["default"] for e in entries})
        default_cell = " / ".join(f"`{d}`" for d in default_vals)
        sources = ", ".join(f"{e['file']}:{e['line']}" for e in entries[:3])
        if len(entries) > 3:
            sources += f", +{len(entries) - 3} more"
        lines.append(f"| {namespace} | {key} | {default_cell} | {sources} |")

    return "\n".join(lines) + "\n"


def render_document(rows):
    """Render the complete standalone reference document."""
    header = (
        "# Config Defaults Reference\n\n"
        f"_Auto-generated by `scripts/gen_defaults_table.py` from {len(rows)} "
        f"`config_val*()` call sites across `nkululeko/`. Do not edit by hand - "
        f"run `python scripts/gen_defaults_table.py --write` to refresh._\n\n"
        "This is a contributor-facing cross-check, not user documentation - see "
        "[ini_file.md](ini_file.md) for the hand-written configuration guide. "
        '"Inconsistent defaults" below are candidates worth a human look, not '
        "necessarily bugs: some reflect genuinely different, context-appropriate "
        "choices (e.g. an SVM and an SVR using different default C values).\n\n"
    )
    return header + render_markdown(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true", help="(re)write docs/source/config_defaults_reference.md"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if config_defaults_reference.md is out of date (for CI)",
    )
    args = parser.parse_args()

    rows = scan()
    document = render_document(rows)

    if args.check:
        if not OUTPUT_DOC.exists() or OUTPUT_DOC.read_text() != document:
            print(
                f"{OUTPUT_DOC.relative_to(REPO_ROOT)} is stale; run "
                "`python scripts/gen_defaults_table.py --write`",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"{OUTPUT_DOC.relative_to(REPO_ROOT)} is up to date.")
        return

    if args.write:
        OUTPUT_DOC.write_text(document)
        print(f"Updated {OUTPUT_DOC}")
    else:
        print(document)


if __name__ == "__main__":
    main()
