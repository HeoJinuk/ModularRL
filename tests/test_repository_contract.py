"""Small repository contracts for public installation and documentation."""

import ast
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_readme_is_short_and_license_metadata_is_real():
    assert len((ROOT / 'README.md').read_text().splitlines()) <= 20
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']
    assert project['license'] == 'MIT' and project['license-files'] == ['LICENSE']
    assert 'Copyright (c) 2026 Jinuk Huh' in (ROOT / 'LICENSE').read_text()


def test_library_has_no_example_imports_or_local_absolute_paths():
    for path in (ROOT / 'modular_rl').rglob('*.py'):
        source = path.read_text()
        assert '/Users/' not in source and '/private/tmp/' not in source
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                assert all(alias.name.split('.')[0] != 'examples' for alias in node.names), path
            if isinstance(node, ast.ImportFrom):
                assert (node.module or '').split('.')[0] != 'examples', path
