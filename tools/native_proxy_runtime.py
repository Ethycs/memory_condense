"""Compatibility import for the installed resident runtime."""
import sys
from memory_condense.runtime import sessions as _implementation
sys.modules[__name__] = _implementation
