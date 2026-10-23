import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jetson_precision_candidate_smoke import main
sys.argv = ['jetson_precision_candidate_smoke.py', '--build', '/home/nvidia/rampq/gate05_corrected_qonly_20261019/gate05_corrected_qonly_candidate_20261019/results/build_20261016T170532_870381Z', '--bundle', '/home/nvidia/rampq/gate05_corrected_qonly_20261019/gate05_corrected_qonly_candidate_20261019', '--source', '/home/nvidia/rampq/jetson_fp16_smoke_20261016_v1', '--output', '/home/nvidia/rampq/gate05_corrected_qonly_20261019/gate05_corrected_qonly_candidate_20261019/fixed_image_smoke', '--matched-fp16-smoke', '/home/nvidia/rampq/gate05_precision_fp16_control_end_smoke_20261019']
main()
