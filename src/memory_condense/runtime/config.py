"""Explicit model paths shared by setup, diagnostics, and the installed runtime."""
from dataclasses import dataclass
import json
import os
from pathlib import Path


def user_directory():
    if os.name == 'nt':
        return Path(os.environ.get('LOCALAPPDATA', Path.home()/'AppData'/'Local'))/'memory_condense'
    return Path(os.environ.get('XDG_DATA_HOME', Path.home()/'.local'/'share'))/'memory_condense'


def config_path():
    return Path(os.environ.get('MEMORY_CONDENSE_CONFIG', user_directory()/'config.json')).expanduser().resolve()


@dataclass(frozen=True)
class RuntimeAssets:
    root: Path

    @classmethod
    def resolve(cls, root=None):
        value = root or os.environ.get('MEMORY_CONDENSE_ASSETS')
        if not value and config_path().is_file():
            value = json.loads(config_path().read_text(encoding='utf-8'))['assets_dir']
        return cls(Path(value or user_directory()/'assets').expanduser().resolve())

    @property
    def qwen(self): return self.root/'models'/'Qwen3-8B'

    @property
    def bge(self): return self.root/'models'/'bge-m3-onnx-fp32'

    @property
    def llama_model(self):
        return self.root/'models'/'Llama-3.2-3B-Instruct-GGUF'/'Llama-3.2-3B-Instruct-Q4_K_M.gguf'

    @property
    def llama_runtime(self): return self.root/'runtimes'/'llama-b11272-bin-win-cuda-12.4-x64'

    @property
    def llama_server(self): return self.llama_runtime/'llama-server.exe'

    @property
    def cuda_runtime(self): return self.root/'runtimes'/'cudart-llama-bin-win-cuda-12.4-x64'

    @property
    def nvcomp_package(self):
        path = self.root/'experiments'/'nvcomp-5.3.0.16'/'package'
        return path if path.is_dir() else None

    @property
    def fastembed_package(self):
        path = self.root/'experiments'/'fastembed'/'package'
        return path if path.is_dir() else None

    def require_files(self):
        from memory_condense.modeling.qwen_prefix import QWEN3_8B_FILE_SHA256, FIRST_SHARD
        paths = [self.qwen/name for name in QWEN3_8B_FILE_SHA256
                 if not name.endswith('.safetensors') or name == FIRST_SHARD]
        paths += [self.bge/'onnx'/'model.onnx', self.bge/'onnx'/'model.onnx_data',
                  self.bge/'source.json', self.llama_model, self.llama_server,
                  self.cuda_runtime/'cudart64_12.dll']
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError('Memory assets are incomplete. Run memory-condense setup, or '
                'memory-condense setup --reuse PATH_TO_EXISTING_CACHE. Missing: '+', '.join(missing))
