"""Tests for scripts/gen_defaults_table.py — the config_val() defaults scanner."""

import ast
import importlib.util
import pathlib

import pytest

SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / "scripts" / "gen_defaults_table.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("gen_defaults_table", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gdt():
    return _load_module()


def _rows_from_source(gdt, source):
    tree = ast.parse(source)
    visitor = gdt.ConfigValVisitor("fake.py")
    visitor.visit(tree)
    return visitor.rows


class TestConfigValVisitor:
    def test_extracts_literal_section_key_default(self, gdt):
        rows = _rows_from_source(
            gdt, "self.util.config_val('MODEL', 'n_jobs', '8')\n"
        )
        assert len(rows) == 1
        row = rows[0]
        assert row["method"] == "config_val"
        assert row["namespace"] == "'MODEL'"
        assert row["key"] == "'n_jobs'"
        assert row["default"] == "'8'"
        assert row["default_is_literal"] is True

    def test_non_literal_default_rendered_as_source(self, gdt):
        rows = _rows_from_source(
            gdt, "self.util.config_val('MODEL', 'device', cuda)\n"
        )
        assert len(rows) == 1
        assert rows[0]["default"] == "cuda"
        assert rows[0]["default_is_literal"] is False

    def test_non_literal_section_and_key_rendered_as_placeholders(self, gdt):
        rows = _rows_from_source(
            gdt, "self.util.config_val(section_name, key_name, '8')\n"
        )
        assert len(rows) == 1
        row = rows[0]
        assert row["namespace"] == "<section_name>"
        assert row["key"] == "<key_name>"
        assert row["default"] == "'8'"
        assert row["default_is_literal"] is True

    def test_config_val_data_maps_to_data_namespace(self, gdt):
        rows = _rows_from_source(
            gdt, "self.util.config_val_data(self.name, 'target_tables', False)\n"
        )
        assert len(rows) == 1
        assert rows[0]["namespace"] == "DATA.<self.name>"
        assert rows[0]["key"] == "'target_tables'"

    def test_config_val_data_with_literal_dataset_name(self, gdt):
        rows = _rows_from_source(
            gdt, "util.config_val_data('emodb', 'type', 'audformat')\n"
        )
        assert len(rows) == 1
        assert rows[0]["namespace"] == "DATA.'emodb'"

    def test_ignores_unrelated_calls(self, gdt):
        rows = _rows_from_source(gdt, "self.util.debug('hello')\nprint('x')\n")
        assert rows == []

    def test_ignores_calls_with_too_few_args(self, gdt):
        rows = _rows_from_source(gdt, "self.util.config_val('MODEL', 'n_jobs')\n")
        assert rows == []

    def test_ignores_keyword_argument_calls(self, gdt):
        rows = _rows_from_source(
            gdt, "self.util.config_val(section='MODEL', key='n_jobs', default='8')\n"
        )
        assert rows == []


class TestFindInconsistencies:
    def test_flags_differing_defaults_for_same_namespace_key(self, gdt):
        rows = [
            {"namespace": "'MODEL'", "key": "'device'", "default": "'cpu'"},
            {"namespace": "'MODEL'", "key": "'device'", "default": "cuda"},
        ]
        grouped = gdt.group_by_namespace_key(rows)
        inconsistent = gdt.find_inconsistencies(grouped)
        assert ("'MODEL'", "'device'") in inconsistent

    def test_does_not_flag_matching_defaults(self, gdt):
        rows = [
            {"namespace": "'MODEL'", "key": "'n_jobs'", "default": "'8'"},
            {"namespace": "'MODEL'", "key": "'n_jobs'", "default": "'8'"},
        ]
        grouped = gdt.group_by_namespace_key(rows)
        inconsistent = gdt.find_inconsistencies(grouped)
        assert inconsistent == {}


class TestRenderDocument:
    def test_includes_title_and_table(self, gdt):
        rows = [
            {
                "namespace": "'MODEL'",
                "key": "'n_jobs'",
                "default": "'8'",
                "file": "nkululeko/models/model.py",
                "line": 39,
            }
        ]
        doc = gdt.render_document(rows)
        assert doc.startswith("# Config Defaults Reference")
        assert "| Namespace | Key | Default | Source |" in doc
        assert "n_jobs" in doc

    def test_links_back_to_ini_file(self, gdt):
        doc = gdt.render_document([])
        assert "ini_file.md" in doc

    def test_deterministic_for_equal_but_distinct_input(self, gdt):
        """Two content-equal (but distinct-object) row lists must render identically.

        Guards against hidden nondeterminism (e.g. dict/set iteration order)
        that could make --check flap between otherwise-identical CI runs.
        """

        def make_rows():
            return [
                {
                    "namespace": "'MODEL'",
                    "key": "'n_jobs'",
                    "default": "'8'",
                    "file": "nkululeko/models/model.py",
                    "line": 39,
                }
            ]

        first = gdt.render_document(make_rows())
        second = gdt.render_document(make_rows())
        assert first == second


class TestScanRealCodebase:
    """Smoke tests against the actual nkululeko/ source tree."""

    def test_only_scans_git_tracked_files(self, gdt):
        files = gdt.tracked_python_files()
        assert len(files) > 100
        assert all(f.startswith("nkululeko/") and f.endswith(".py") for f in files)

    def test_scan_finds_known_call_sites(self, gdt):
        rows = gdt.scan()
        assert len(rows) > 100
        namespace_keys = {(r["namespace"], r["key"]) for r in rows}
        assert ("'MODEL'", "'n_jobs'") in namespace_keys
        assert ("'MODEL'", "'C_val'") in namespace_keys

    def test_render_markdown_produces_a_table(self, gdt):
        rows = gdt.scan()
        table_md = gdt.render_markdown(rows)
        assert "| Namespace | Key | Default | Source |" in table_md
        assert "n_jobs" in table_md

    def test_generated_reference_file_matches_current_code(self, gdt):
        """Guards against committing a stale docs/source/config_defaults_reference.md."""
        rows = gdt.scan()
        expected = gdt.render_document(rows)
        assert gdt.OUTPUT_DOC.exists(), (
            "docs/source/config_defaults_reference.md is missing; run "
            "python scripts/gen_defaults_table.py --write"
        )
        actual = gdt.OUTPUT_DOC.read_text()
        assert actual == expected, (
            "docs/source/config_defaults_reference.md is stale; run "
            "python scripts/gen_defaults_table.py --write"
        )
