"""Run every Gymnasium tutorial twice in its own kernel, using the FAST budget."""
import base64
import io
import json
import sys
from pathlib import Path

import pytest

nbformat = pytest.importorskip('nbformat')
NotebookClient = pytest.importorskip('nbclient').NotebookClient
pytest.importorskip('matplotlib')
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager
from PIL import Image
from tests.test_dlfs4_notebooks import plot_cache, _outputs

def semantic_outputs(notebook):
    """Adjacent stdout chunks have transport-dependent boundaries, not different text."""
    result = _outputs(notebook)
    for index, outputs in enumerate(result):
        merged = []
        for output in outputs:
            if (merged and output.get("output_type") == "stream" and
                    merged[-1].get("output_type") == "stream" and merged[-1]["name"] == output["name"]):
                merged[-1]["text"] += output["text"]
            else:
                merged.append(output)
        result[index] = merged
    return result


def numeric_snapshot(client, notebook):
    """Inspect full precision plot inputs, without changing delivered notebook cells."""
    inspection = nbformat.v4.new_code_cell("""
import json as _verify_json

def _verify_numeric(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k: _verify_numeric(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [_verify_numeric(v) for v in value]
    return value

_verify_names = ('log','losses','epsilons','updates','rmse','errors','table','results',
                 'before','after','evaluation','summary','greedy_actions',
                 'actor_losses','critic_losses','entropies','td_errors','target_syncs')
_verify_payload = {k:_verify_numeric(globals()[k]) for k in _verify_names if k in globals()}
_verify_payload['torch_rng'] = torch.get_rng_state().tolist()
_verify_payload['rngs'] = {k: globals()[k].bit_generator.state
                          for k in ('np_rng','update_rng','planning_rng') if k in globals()}
if 'buffer' in globals() and hasattr(buffer, '_rng'):
    _verify_payload['replay_rng'] = buffer._rng.bit_generator.state
for _verify_name in ('reference','pi_result','vi_result'):
    if _verify_name in globals():
        _verify_value = globals()[_verify_name]
        _verify_payload[_verify_name] = _verify_numeric(_verify_value.values if hasattr(_verify_value,'converged') else _verify_value)
for _verify_name in ('model','model1','model2'):
    if _verify_name in globals():
        _verify_payload[_verify_name+'.state_dict'] = _verify_numeric(globals()[_verify_name].state_dict())
        _verify_estimator = globals()[_verify_name].estimator
        for _verify_table in ('v_table','q_table'):
            if hasattr(_verify_estimator,_verify_table):
                _verify_payload[_verify_name+'.'+_verify_table] = _verify_numeric(getattr(_verify_estimator,_verify_table))
if 'models' in globals():
    for _verify_name, _verify_model in models.items():
        for _verify_table in ('v_table','q_table'):
            if hasattr(_verify_model.estimator,_verify_table):
                _verify_payload['models.'+_verify_name+'.'+_verify_table] = _verify_numeric(getattr(_verify_model.estimator,_verify_table))
print(_verify_json.dumps(_verify_payload, sort_keys=True, allow_nan=False))
""")
    notebook.cells.append(inspection)
    try:
        client.execute_cell(inspection, len(notebook.cells)-1)
        return ''.join(output.get('text','') for output in inspection.outputs)
    finally:
        notebook.cells.pop()


DIRECTORY = Path(__file__).resolve().parents[1] / 'notebooks' / 'gymnasium'
NAMES = ['00_api_basics','01_dp','02_mc_prediction','03_mc_control','04_td_prediction','05_q_learning',
         '06_sarsa','07_expected_sarsa','08_double_q','09_n_step_sarsa','10_off_policy_n_step','11_tree_backup',
         '12_q_sigma','13_dyna_q','14_dyna_dqn','15_dqn','16_n_step_dqn','17_double_dqn','18_actor_critic']


@pytest.mark.parametrize('name', NAMES)
def test_gymnasium_run_all_twice_in_independent_kernel(name, tmp_path, monkeypatch, plot_cache):
    monkeypatch.setenv('MODULARRL_FAST', '1')
    monkeypatch.setenv('MPLCONFIGDIR', str(plot_cache / 'matplotlib'))
    monkeypatch.setenv('XDG_CACHE_HOME', str(plot_cache / 'xdg'))
    for variable in ('IPYTHONDIR','JUPYTER_RUNTIME_DIR'):
        directory = tmp_path / variable.lower(); directory.mkdir()
        monkeypatch.setenv(variable, str(directory))
    kernel_dir = tmp_path / 'kernels' / 'modularrl-test'
    kernel_dir.mkdir(parents=True)
    (kernel_dir / 'kernel.json').write_text(json.dumps({
        'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
        'display_name':'ModularRL test','language':'python'}))
    manager = AsyncKernelManager(kernel_name='modularrl-test',
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)]))
    path = DIRECTORY / (name+'.ipynb')
    notebook = nbformat.read(path, as_version=4)
    source = '\n'.join(c.source for c in notebook.cells if c.cell_type == 'code')
    assert 'env.reset(' in source and 'env.step(' in source and 'env.close()' in source
    assert 'train_and_evaluate(' not in source
    assert 'MODULARRL_FAST' in source
    client = NotebookClient(notebook, km=manager, timeout=300, allow_errors=False, record_timing=False,
                            resources={'metadata':{'path':str(DIRECTORY)}})
    with client.setup_kernel(cleanup_kc=True):
        client.execute(); first = semantic_outputs(notebook)
        first_numeric = numeric_snapshot(client, notebook)
        client.execute(); second = semantic_outputs(notebook)
        second_numeric = numeric_snapshot(client, notebook)
    assert first_numeric == second_numeric
    assert first == second
    images = [output.data['image/png'] for outputs in second for output in outputs
              if 'image/png' in output.get('data', {})]
    assert images
    for data in images:
        with Image.open(io.BytesIO(base64.b64decode(data))) as figure:
            assert figure.width > 200 and figure.height > 150
            figure.verify()
    nbformat.validate(notebook)
    nbformat.write(notebook, tmp_path / path.name)
