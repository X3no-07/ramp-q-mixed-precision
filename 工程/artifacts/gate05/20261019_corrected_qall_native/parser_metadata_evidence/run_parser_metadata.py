import pathlib,subprocess,sys
r=pathlib.Path(__file__).resolve().parent
subprocess.run([sys.executable,str(r/'jetson_inspect_corrected_qall_parser.py'),'--root','/home/nvidia/rampq/gate05_corrected_qall_20261019/gate05_corrected_qall_candidate_20261019','--output',str(r/'inventory'),'--inspection-contract',str(r/'parser_metadata_contract.json')],check=True)
