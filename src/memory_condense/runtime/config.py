"""Explicit model paths shared by setup, diagnostics, and the installed runtime."""
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import platform


def user_directory():
    if os.name == 'nt':
        return Path(os.environ.get('LOCALAPPDATA', Path.home()/'AppData'/'Local'))/'memory_condense'
    return Path(os.environ.get('XDG_DATA_HOME', Path.home()/'.local'/'share'))/'memory_condense'


def config_path():
    return Path(os.environ.get('MEMORY_CONDENSE_CONFIG', user_directory()/'config.json')).expanduser().resolve()


def dialogue_policy(mode='cap8', filename=None):
    """Validate tuning before loading models or opening a conversation."""
    from memory_condense.application.dialogue_retrieval import DialogueRetrievalPolicy
    if mode not in ('cap8', 'dialogue', 'auto', 'hierarchy_neighborhoods', 'decision_history'):
        raise ValueError('Unknown memory retrieval policy')
    if mode == 'cap8':
        if filename is not None:
            raise ValueError('--memory-dialogue-config requires auto, dialogue, hierarchy_neighborhoods, or decision_history retrieval')
        return None
    if filename is None:
        return DialogueRetrievalPolicy()
    value = json.loads(Path(filename).read_text(encoding='utf-8-sig'))
    return DialogueRetrievalPolicy.from_mapping(value)


def experiment_policy(filename=None):
    """An absent profile leaves every experimental module off."""
    if filename is None:
        return None
    from memory_condense.application.native_experiments import NativeExperimentPolicy
    value = json.loads(Path(filename).read_text(encoding='utf-8-sig'))
    policy = NativeExperimentPolicy.from_mapping(value)
    if policy.cav_bundle:
        from memory_condense.runtime.experimental_attention import cav_bundle
        from memory_condense.modeling.qwen_prefix import expected_prefix_checkpoint_sha256
        cav_bundle(policy.cav_bundle,layer=policy.qwen_layer,
                   checkpoint_sha256=expected_prefix_checkpoint_sha256(6))
    return policy


#: Host identity -> pinned asset flavor. Both spellings of x86-64 appear in the
#: wild: CPython reports AMD64 on Windows and x86_64 on Linux.
SUPPORTED_PLATFORMS = {('Windows', 'amd64'): 'win-amd64', ('Windows', 'x86_64'): 'win-amd64',
                       ('Linux', 'amd64'): 'linux-x86_64', ('Linux', 'x86_64'): 'linux-x86_64'}


def detect_platform_key():
    """Asset flavor for this host, or None where no native runtime is pinned."""
    return SUPPORTED_PLATFORMS.get((platform.system(), platform.machine().lower()))


#: One keyed flavor per platform, so the runtime directory, the server binary,
#: the CUDA runtime directory and its sentinel can never drift apart. Every
#: 'runtime'/'cudart' value must equal archive_stem() of its llama-runtime.json
#: entry; tests/test_proxy_package.py asserts that cross-check.
#: Upstream naming is asymmetric and is reproduced verbatim: the Windows cudart
#: archive carries no build number, the Linux one does. Linux pins CUDA 12.8
#: because b11272 publishes no lower Linux CUDA build; the archive bundles its
#: own libcudart, so minor-version compatibility keeps the cuda=12.0 floor valid.
LLAMA_RUNTIME = {
    'win-amd64': dict(runtime='llama-b11272-bin-win-cuda-12.4-x64', server='llama-server.exe',
                      cudart='cudart-llama-bin-win-cuda-12.4-x64', sentinel='cudart64_12.dll',
                      library_path='PATH'),
    'linux-x86_64': dict(runtime='llama-b11272-bin-ubuntu-cuda-12.8-x64', server='llama-server',
                         cudart='cudart-llama-b11272-bin-ubuntu-cuda-12.8-x64',
                         sentinel='libcudart.so.12', library_path='LD_LIBRARY_PATH'),
}


@dataclass(frozen=True)
class RuntimeAssets:
    root: Path
    bge_root: Path | None = None
    #: Read per instance, not per import, so a test may patch platform.system().
    platform_key: str | None = field(default_factory=detect_platform_key)

    @classmethod
    def resolve(cls, root=None):
        value = root or os.environ.get('MEMORY_CONDENSE_ASSETS')
        if not value and config_path().is_file():
            value = json.loads(config_path().read_text(encoding='utf-8'))['assets_dir']
        bge = os.environ.get('MEMORY_CONDENSE_BGE_MODEL')
        return cls(Path(value or user_directory()/'assets').expanduser().resolve(),
                   Path(bge).expanduser().resolve() if bge else None)

    @property
    def qwen(self): return self.root/'models'/'Qwen3-8B'

    @property
    def bge(self): return self.bge_root or self.root/'models'/'bge-m3-onnx-fp32'

    @property
    def llama_model(self):
        return self.root/'models'/'Llama-3.2-3B-Instruct-GGUF'/'Llama-3.2-3B-Instruct-Q4_K_M.gguf'

    @property
    def llama_flavor(self):
        """Pinned summarizer flavor for this host.

        Unsupported hosts raise here rather than at construction, so the model
        paths and require_files(include_summarizer=False) stay usable and
        observe mode remains independent of the native runtime.
        """
        flavor = LLAMA_RUNTIME.get(self.platform_key)
        if flavor is None:
            raise RuntimeError('No packaged llama.cpp runtime is pinned for '
                               +platform.platform()+'; observe mode is independent.')
        return flavor

    @property
    def llama_runtime(self): return self.root/'runtimes'/self.llama_flavor['runtime']

    @property
    def llama_server(self): return self.llama_runtime/self.llama_flavor['server']

    @property
    def cuda_runtime(self): return self.root/'runtimes'/self.llama_flavor['cudart']

    @property
    def cuda_runtime_sentinel(self): return self.cuda_runtime/self.llama_flavor['sentinel']

    @property
    def library_path_variable(self):
        """Loader variable the child summarizer uses to find the pinned CUDA runtime."""
        return self.llama_flavor['library_path']

    @property
    def nvcomp_package(self):
        path = self.root/'experiments'/'nvcomp-5.3.0.16'/'package'
        return path if path.is_dir() else None

    @property
    def fastembed_package(self):
        path = self.root/'experiments'/'fastembed'/'package'
        return path if path.is_dir() else None

    def require_files(self, *, include_summarizer=True):
        from memory_condense.modeling.qwen_prefix import QWEN3_8B_FILE_SHA256, FIRST_SHARD
        paths = [self.qwen/name for name in QWEN3_8B_FILE_SHA256
                 if not name.endswith('.safetensors') or name == FIRST_SHARD]
        paths += [self.bge/'onnx'/'model.onnx', self.bge/'onnx'/'model.onnx_data',
                  self.bge/'source.json']
        if include_summarizer:
            paths += [self.llama_model, self.llama_server, self.cuda_runtime_sentinel]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError('Memory assets are incomplete. Run memory-condense setup, or '
                'memory-condense setup --reuse PATH_TO_EXISTING_CACHE. Missing: '+', '.join(missing))
