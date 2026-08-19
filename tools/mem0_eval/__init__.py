"""Offline tooling for the isolated Mem0 comparison campaign.

This package deliberately lives outside ``src/memory_condense``.  The frozen
v3 validation policy hashes every Python file in that package, so comparison
tooling must not alter the implementation being scored.
"""

from __future__ import annotations

import os


# Importing a ``memory_condense.eval`` submodule currently executes that
# package's eager compatibility imports, which include LiteLLM.  Establish
# the offline boundary before any Mem0 tool submodule can trigger that path.
# This isolated package is intentionally offline, so overwrite rather than
# inherit caller values: a preexisting false/zero setting must not re-enable
# an import-time model-price fetch, model download, or telemetry client.
for _name, _value in {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "LITELLM_LOCAL_MODEL_COST_MAP": "true",
    "MEM0_TELEMETRY": "false",
}.items():
    os.environ[_name] = _value

del _name, _value
