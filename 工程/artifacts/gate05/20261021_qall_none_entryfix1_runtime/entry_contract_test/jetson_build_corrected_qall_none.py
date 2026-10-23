import json,sys
from pathlib import Path
assert sys.argv[1:] == ['--root',str(Path(__file__).resolve().parent)]
print(json.dumps(dict(status='MONITOR_NO_ARGUMENT_ENTRY_ADAPTER_CONTRACT_VERIFIED',argv=sys.argv,native_runtime_imported=False)))
