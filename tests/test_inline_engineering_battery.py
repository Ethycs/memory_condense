import json

from tools import engineering_research_battery as battery
from tools import run_engineering_research_battery as runner
from tools.engineering_research_chat import open_chat
from tools.engineering_research_gateway import read, save


def test_inline_engineering_executes_action_and_keeps_internal_summary_hidden(tmp_path):
    actor=dict(case_id='E01',domain='engineering',task_kind='engineering_design',
        original_request='Build it.',current_turn_id='source:T0002',task='Write the design.',
        deliverables=['design.md'],public_checks=[],external_dependencies='none',
        history=[dict(turn_id='source:T0000',role='user',source_id='source',text='Preserve exact IDs.')],
        source=dict(family='source',export_timestamp='2026-01-01T00:00:00+00:00'))
    record=dict(id='E01',actor=battery.publish(tmp_path/'actor.json',actor))
    save(tmp_path/'run-plan.json',dict(web={'enabled':False}))
    actions=[dict(action='write_many',artifacts={'design.md':'Preserve exact IDs.'}),
             dict(action='finish',message='Design delivered.')]
    class Gateway:
        def call(self,kind,messages,**kwargs):
            assert '"memory"' in messages[0]['content']
            answer=json.dumps(actions.pop(0))
            return dict(content=json.dumps(dict(answer=answer,memory=dict(
                user=dict(summary='User requests the design.',support=['Build it.']),
                assistant=dict(summary='Assistant produces an engineering action.',support=[answer[:20]])))),
                finish_reason='stop',elapsed_s=.1,request_sha256='1'*64,response_model='same-model')
    folder=tmp_path/'cases/E01/full_context'
    with open_chat(tmp_path,folder,actor,'full_context') as chat:
        result=runner._run_arm(tmp_path,record,'full_context',chat=chat,reader_gateway=Gateway(),inline_memory=True)
        assert result['finished']
        assert result['artifacts']=={'design.md':'Preserve exact IDs.'}
        output=chat.event('E01:full_context:A000:assistant')
        assert json.loads(output.text)['action']=='write_many'
        assert output.metadata['inline_generation']['status']=='accepted'
        assert chat.event(actor['current_turn_id']).text=='Build it.\nWrite the design.'
    served=read(folder/'actions/000/served-prompt.json')
    assert 'Exchange input to summarize' in served['messages'][-1]['content']


def test_five_added_cases_use_separate_archive_families():
    from tools.build_inline_engineering15 import specification
    spec=specification()
    assert len(spec['cases'])==15
    assert len({c['id'] for c in spec['cases']})==15
    assert all(c['domain']=='engineering' for c in spec['cases'])
    new=spec['cases'][10:]
    assert len({c['source'] for c in new})==5
    assert not {c['source'] for c in new} & {c['source'] for c in spec['cases'][:10]}


def test_added_check_executor_rejects_a_broken_component(tmp_path):
    from tools.run_inline_engineering15 import added_checks
    (tmp_path/'mix_tuner.py').write_text('def select_mix(*args): return {}\n',encoding='utf-8')
    result=added_checks(tmp_path.resolve(),'E14')
    assert result['exit_code']!=0
    assert 'AssertionError' in result['output']
    assert result['environment_credentials_removed']


def test_preparation_accepts_audited_battery_manifest_sidecar(tmp_path,monkeypatch):
    from tools import run_inline_engineering15 as inline
    bundle=tmp_path/'bundle'
    manifest=dict(cases=[dict(id=f'E{i:02}',domain='engineering') for i in range(1,16)])
    binding=battery.publish(bundle/'battery.json',manifest)
    monkeypatch.setattr(battery,'audit',lambda folder:dict(battery_sha256=binding['sha256']))
    inline.prepare(tmp_path/'run',bundle)
    plan=inline.verify(tmp_path/'run')
    assert plan['cases']==manifest['cases']
    assert plan['inline_memory']
