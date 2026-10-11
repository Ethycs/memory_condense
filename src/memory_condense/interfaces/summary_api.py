"""A conversation's existing provider transport for bounded summary escalation."""
import time

from memory_condense.interfaces import proxy_memory_wire as wire


class ProviderSummaryAPI:
    def __init__(self, provider, model, send):
        if provider not in ('openai', 'anthropic') or not isinstance(model, str) or not model.strip():
            raise ValueError('Summary escalation requires a provider and model')
        self.provider, self.model, self.send = provider, model, send
        self.identity = dict(provider=provider, model=model, protocol='source-extracts-v1')

    def __call__(self, messages, *, max_tokens):
        payload = dict(model=self.model, messages=messages, max_tokens=max_tokens, stream=True)
        if self.provider == 'anthropic':
            payload['system'] = '\n\n'.join(m['content'] for m in messages if m['role']=='system')
            payload['messages'] = [m for m in messages if m['role']!='system']
        else:
            payload['stream_options'] = {'include_usage': True}
            if self.model.startswith('codex_sdk/'):
                payload['reasoning_effort'] = 'none'
        started = time.perf_counter()
        response = self.send(wire.encoded(payload))
        if response.status >= 400:
            raise RuntimeError(f'Summary API returned HTTP {response.status}')
        value = wire.assemble_stream(self.provider, response.body)
        message = wire.response_message(self.provider, value)
        finish = wire.finish_reason(self.provider, value)
        content = wire.text_content(message)
        if finish not in ('stop', 'end_turn') or not content.strip() or wire.has_alternative(message):
            raise ValueError('Summary API returned no complete text generation')
        return dict(content=content, finish_reason='stop', response_model=value.get('model', self.model),
                    usage=value.get('usage'), elapsed_s=time.perf_counter()-started)
