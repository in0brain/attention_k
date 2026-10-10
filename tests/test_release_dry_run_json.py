"""The documented synthetic CLI must emit valid JSON for NumPy-valued results."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def test_synthetic_cli_serializes_nested_numpy_results(monkeypatch, capsys):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / 'src'))
    spec = importlib.util.spec_from_file_location(
        'release_confirmatory_entry', root / 'scripts/run_stage1_v3_confirmatory.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'run_dry', lambda **kwargs: {
        'synthetic': True, 'label': 'gain \u2212 baseline', 'arms': {'h1': {'scores': np.array([0.1, 0.9]),
                                         'count': np.int64(2)}}})
    assert module.main(['--arm', 'both', '--dry-run']) == 0
    output = capsys.readouterr().out
    output.encode('ascii')  # Also works in Windows terminals with a legacy encoding.
    payload = json.loads(output)
    assert payload == {'synthetic': True, 'label': 'gain \u2212 baseline',
                       'arms': {'h1': {'scores': [0.1, 0.9], 'count': 2}}}
