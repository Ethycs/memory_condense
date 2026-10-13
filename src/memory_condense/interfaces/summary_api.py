"""A conversation's existing provider transport for bounded summary escalation."""
import time

from memory_condense.interfaces import proxy_memory_responses as responses
from memory_condense.interfaces import proxy_memory_wire as wire


class ProviderSummaryAPI:
    def __init__(self, provider, model, send, *, fmt=None):
        if provider not in ('openai', 'anthropic') or not isinstance(model, str) or not model.strip():
            raise ValueError('Summary escalation requires a provider and model')
        self.provider, self.model, self.send = provider, model, send
        # Escalation reuses the visible request's upstream URL, so the body must
        # match the inbound path's format or the gateway rejects it outright.
        self.fmt = fmt or wire.format_for(provider)
        if self.fmt not in ('chat', 'anthropic', 'responses'):
            raise ValueError('Summary escalation requires a known wire format')
        # The receipt identity names provider and model only: the wire format is
        # a transport detail and must not re-key already accepted summaries.
        self.identity = dict(provider=provider, model=model, protocol='source-extracts-v1')

    def __call__(self, messages, *, max_tokens):
        if self.fmt == 'responses':
            payload = dict(model=self.model, max_output_tokens=max_tokens, stream=True, store=False,
                           instructions='\n\n'.join(m['content'] for m in messages if m['role']=='system'),
                           input=[item for m in messages if m['role']!='system'
                                  for item in responses.input_item(m)])
        else:
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
        value = wire.assemble_stream(self.fmt, response.body)
        message = wire.response_message(self.fmt, value)
        finish = wire.finish_reason(self.fmt, value)
        content = wire.text_content(message)
        if finish not in ('stop', 'end_turn') or not content.strip() or wire.has_alternative(message):
            raise ValueError('Summary API returned no complete text generation')
        return dict(content=content, finish_reason='stop', response_model=value.get('model', self.model),
                    usage=value.get('usage'), elapsed_s=time.perf_counter()-started)
