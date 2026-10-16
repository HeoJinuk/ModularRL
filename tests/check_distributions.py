"""Validate release contents and install wheel/sdist into separate temporary venvs."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile


def module_sources(path):
    if path.suffix == '.whl':
        with zipfile.ZipFile(path) as archive:
            records = {name: archive.read(name) for name in archive.namelist() if not name.endswith('/')}
    else:
        with tarfile.open(path) as archive:
            records = {member.name.split('/', 1)[1]: archive.extractfile(member).read()
                       for member in archive.getmembers() if member.isfile()}
    forbidden = {'tests', 'notebooks', 'examples', 'experiments', '__pycache__', '.ipynb_checkpoints', '.git'}
    for name in records:
        parts = set(Path(name).parts)
        assert not parts & forbidden, name
        assert not name.endswith(('.pyc', '.DS_Store', '.env', '.pem', '.key')), name
    licenses = [body for name, body in records.items() if name.endswith('LICENSE')]
    assert licenses and any(b'Copyright (c) 2026 Jinuk Huh' in body for body in licenses)
    sources = {name: body for name, body in records.items() if name.startswith('modular_rl/') and name.endswith('.py')}
    assert sources and 'modular_rl/learner/ppo.py' in sources
    return sources


def check(directory):
    directory = directory.resolve()
    wheels, sdists = list(directory.glob('*.whl')), list(directory.glob('*.tar.gz'))
    assert len(wheels) == len(sdists) == 1, 'Use a directory with one wheel and one sdist'
    assert module_sources(wheels[0]) == module_sources(sdists[0])
    for artifact in (wheels[0], sdists[0]):
        with tempfile.TemporaryDirectory(prefix='modularrl-installed-') as temporary:
            root = Path(temporary)
            # Share existing third-party dependencies, but install ModularRL in this venv.
            venv.EnvBuilder(with_pip=True, system_site_packages=True).create(root / 'venv')
            python = root / 'venv' / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
            subprocess.run([str(python), '-m', 'pip', 'install', '--no-deps',
                            '--force-reinstall', str(artifact)], cwd=root, check=True)
            shutil.copyfile(Path(__file__).with_name('installed_smoke.py'), root / 'smoke.py')
            subprocess.run([str(python), '-I', str(root / 'smoke.py')], cwd=root, check=True)
        print('PASS:', artifact.name, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, nargs='?', default=Path('dist'))
    check(parser.parse_args().directory)
