"""Adapt the immutable monitored no-argument entry to the root-argument builder."""
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parent
builder = root / 'jetson_build_corrected_qall_none.py'
sys.argv = [str(builder), '--root', str(root)]
runpy.run_path(str(builder), run_name='__main__')
