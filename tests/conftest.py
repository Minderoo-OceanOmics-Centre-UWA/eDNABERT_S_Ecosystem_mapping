import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Provide a minimal stub for `transformers` if not installed,
# so importing the pipeline module doesn't fail.
if "transformers" not in sys.modules:
    transformers_stub = types.ModuleType("transformers")

    class _Dummy:
        def __init__(self, *args, **kwargs):
            pass

    # These are imported at module-level in calculatePerSite.py
    transformers_stub.AutoTokenizer = _Dummy
    transformers_stub.AutoModel = _Dummy
    transformers_stub.AutoConfig = _Dummy

    sys.modules["transformers"] = transformers_stub

