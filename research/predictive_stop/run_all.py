"""Run from Python 3.10+ after installing the research dependencies."""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
env = os.environ.copy()
env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
for script in ["download.py", "prepare.py", "experiment.py"]:
    subprocess.run([sys.executable, "-u", str(HERE / script)], cwd=ROOT, env=env, check=True)
