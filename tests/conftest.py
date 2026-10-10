"""Skip explicitly marked integration checks when local experiment inputs are absent."""
from pathlib import Path

import pytest


def pytest_configure(config):
    config.addinivalue_line('markers', 'local_artifacts(*paths): needs unbundled local experiment artifacts')
    config.addinivalue_line('markers', 'pre_unblinding_workspace: historical check of the workspace before formal analysis')


def pytest_runtest_setup(item):
    root = Path(__file__).resolve().parents[1]
    if item.get_closest_marker('pre_unblinding_workspace'):
        if (root / 'outputs/stage1/confirmatory/stage1_confirmatory_result.json').exists():
            pytest.skip('Historical pre-analysis workspace check; this release includes the completed result')
    marker = item.get_closest_marker('local_artifacts')
    if marker:
        missing = [path for path in marker.args if not (root / path).exists()]
        if missing:
            pytest.skip('Local experiment inputs are not bundled: ' + ', '.join(missing))
