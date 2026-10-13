"""Execute the delivered notebooks twice in one kernel, including plot outputs."""

import copy
import base64
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat", reason="Install the notebooks extra to execute tutorials")
pytest.importorskip("matplotlib", reason="Install the notebooks extra to render tutorial figures")
NotebookClient = pytest.importorskip("nbclient").NotebookClient
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager
from PIL import Image

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks" / "dlfs4"


@pytest.fixture(scope="module")
def plot_cache(tmp_path_factory):
    """Prepare Matplotlib's one-time font cache outside the tutorial's outputs."""
    directory = tmp_path_factory.mktemp("dlfs4-plot-cache")
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(directory / "matplotlib")
    environment["XDG_CACHE_HOME"] = str(directory / "xdg")
    subprocess.run([sys.executable, "-c", "import matplotlib.font_manager"], env=environment,
                   check=True, capture_output=True, timeout=120)
    return directory


def _outputs(notebook):
    """Ignore execution counters; compare actual text, numeric and image payloads."""
    result = []
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        outputs = copy.deepcopy(cell.outputs)
        for output in outputs:
            output.pop("execution_count", None)
        result.append(outputs)
    return result


@pytest.mark.parametrize("name", ["04_dp_modularrl.ipynb", "05_mc_modularrl.ipynb", "06_td_modularrl.ipynb"])
def test_run_all_twice_reproduces_text_and_figures_in_same_kernel(name, tmp_path, monkeypatch, plot_cache):
    monkeypatch.setenv("MPLCONFIGDIR", str(plot_cache / "matplotlib"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(plot_cache / "xdg"))
    for variable in ("IPYTHONDIR", "JUPYTER_RUNTIME_DIR"):
        directory = tmp_path / variable.lower()
        directory.mkdir()
        monkeypatch.setenv(variable, str(directory))
    # Run with the test interpreter, independent of a user's global kernelspec.
    kernel_dir = tmp_path / "kernels" / "modularrl-test"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "ModularRL test", "language": "python",
    }))
    manager = AsyncKernelManager(kernel_name="modularrl-test",
                                 kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)]))
    notebook = nbformat.read(NOTEBOOK_DIR / name, as_version=4)
    nbformat.validate(notebook)
    client = NotebookClient(notebook, km=manager, timeout=180, allow_errors=False, record_timing=False,
                            resources={"metadata": {"path": str(NOTEBOOK_DIR)}})
    with client.setup_kernel(cleanup_kc=True):
        client.execute()
        first = _outputs(notebook)
        client.execute()
        second = _outputs(notebook)
    assert first == second
    assert all(cell.execution_count is not None for cell in notebook.cells if cell.cell_type == "code")
    images = []
    for outputs in second:
        for output in outputs:
            assert output.output_type != "error"
            if "image/png" in output.get("data", {}):
                images.append(output.data["image/png"])
    assert images, "Every tutorial must display rendered results"
    for encoded in images:
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as figure:
            assert figure.width > 200 and figure.height > 150
            figure.verify()
    nbformat.validate(notebook)
    nbformat.write(notebook, tmp_path / name)
