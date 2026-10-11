"""Offline installed-artifact smoke, run with python -I inside its environment."""
import argparse
from contextlib import redirect_stdout
import importlib.abc
from importlib.metadata import version
import io
import json
from pathlib import Path
import sys
import tomllib
from types import SimpleNamespace


class NoResearch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('tools', 'tests'):
            raise AssertionError('Installed runtime imported '+fullname)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version')
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    expected=args.version or tomllib.loads((repo/'pyproject.toml').read_text(encoding='utf-8'))['project']['version']
    assert version('memory_condense')==expected
    sys.meta_path.insert(0,NoResearch())
    import numpy as np
    import memory_condense
    from memory_condense.application.dialogue_retrieval import DialogueRetrievalPolicy
    from memory_condense.application.tactic_router import BGETacticRouter,DEFAULT_MODEL
    from memory_condense.application.hierarchy_context import PreparedHierarchyContext
    from memory_condense.application.continuity import FORMAT, GUIDANCE
    from memory_condense.application.native_experiments import NativeExperimentPolicy, PreparedNativeExperiments
    from memory_condense.application.native_experiment_learning import summary_weights
    from memory_condense.runtime.experimental_attention import cav_bundle
    from memory_condense.associations.heat_diffusion import diffuse_scalar_heat
    from memory_condense.runtime.sessions import NativeProxySessions
    from memory_condense.interfaces.proxy_server import build_app
    from memory_condense.runtime.artifacts import read
    from memory_condense.cli import main as cli
    assert not Path(memory_condense.__file__).resolve().is_relative_to(repo/'src'), 'Imported checkout'
    assert DialogueRetrievalPolicy().max_context_tokens==6144
    assert NativeExperimentPolicy().modules==()
    assert diffuse_scalar_heat({'a':1.},{'a':(('b',1.),)},max_nodes=2).total_heat==1.
    for name in ('admission.json','plan.json','report.json'):
        read(DEFAULT_MODEL.parent/'bge-admission'/name)
    artifact=read(DEFAULT_MODEL)
    read(DEFAULT_MODEL.parent/'history-intent-v1.json')
    assert FORMAT == 'memory-condense-continuity-v1' and 'Fresh-chat handoff:' in GUIDANCE
    assert 'Copyright (c) 2026 Ethycs' in (DEFAULT_MODEL.parent/'chat-decompose-LICENSE.txt').read_text(encoding='utf-8')
    identity=json.loads(artifact['embedding_identity'])
    encoder=SimpleNamespace(model_name=identity['model_id'],model_revision=identity['model_revision'],
        checkpoint_sha256=identity['checkpoint_sha256'],execution_identity=identity['execution'],
        embed_query=lambda _:np.ones(artifact['dimension']))
    decision=BGETacticRouter(encoder).prioritize('Package smoke query')
    assert decision.embedding_calls==1
    assert decision.history_intent is not None
    help_text=io.StringIO()
    with redirect_stdout(help_text):
        try:
            cli(['proxy','--help'])
        except SystemExit as exc:
            assert exc.code==0
    assert 'Default: BGE' in help_text.getvalue()
    assert 'hierarchy_neighborhoods' in help_text.getvalue()
    assert 'decision_history' in help_text.getvalue()
    assert '--memory-experiment-config' in help_text.getvalue()
    print(json.dumps(dict(installed_version=expected,package=str(Path(memory_condense.__file__).parent),
        sealed_classifier=True,proxy_default='auto',research_imports=False,model_downloads=0,
        note='Offline package/identity smoke; deterministic vector, no model accuracy claim.')))


if __name__=='__main__':
    main()
