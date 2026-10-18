"""API-only evaluation transport using the proxy's existing SSE decoder."""
import os
import ssl
import time

import httpx
import truststore
from dotenv import load_dotenv

from memory_condense.interfaces import proxy_memory_wire as wire
from memory_condense.runtime.artifacts import save, emit
from tools.engineering_research_gateway_reader import GatewayReaderRuntime
from tools.engineering_research_stream import EmptyGatewayResponseError, GatewayContentError


class PacketGatewayRuntime(GatewayReaderRuntime):
    def start_gateway(self):
        if self.remote is not None:
            return
        load_dotenv(override=False)
        key = os.environ.get('LITELLM_KEY', '').strip()
        if not key:
            raise RuntimeError('LITELLM_KEY is missing')
        self.remote = httpx.Client(timeout=120, verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
            headers={'authorization': 'Bearer '+key}, base_url=self.gateway.rstrip('/')+'/')
        result = self.call('gateway_warmup', [dict(role='user',content='Reply exactly READY.')],
                           scope='gateway-preflight',max_tokens=32)
        if result['content'].strip() not in ('READY','READY.'):
            raise ValueError('Gateway preflight did not follow the simple instruction')
        emit(phase='gateway_reader_ready',model=self.reader_model,elapsed_s=result['elapsed_s'])

    def _remote_call(self, kind, messages, *, scope, max_tokens):
        model = self.judge_model if kind == 'judge' else self.reader_model
        if not model.startswith('codex_sdk/'):
            raise ValueError('This replay transport is pinned to the original Codex route')
        args = dict(model=model,messages=messages,max_tokens=max_tokens,stream=True,
                    stream_options={'include_usage':True},reasoning_effort='none')
        with self._counter_lock:
            ordinal = self._ordinal
            self._ordinal += 1
        folder = self.root/'gateway-calls'
        request = save(folder/f'{ordinal:05d}.request.json',dict(kind=kind,scope=scope,gateway=self.gateway,request=args))
        save(folder/f'{ordinal:05d}.reservation.json',dict(request_sha256=request.sha256,kind=kind,model=model))
        started = time.perf_counter()
        try:
            response = self.remote.post('chat/completions',json=args)
            save(folder/f'{ordinal:05d}.transport.json',dict(status=response.status_code,
                content_type=response.headers.get('content-type',''),body=response.text,
                elapsed_s=time.perf_counter()-started))
            response.raise_for_status()
            value = wire.assemble_stream('chat',response.content)
            message = wire.response_message('chat',value)
            finish = wire.finish_reason('chat',value)
            content = wire.text_content(message)
            if wire.has_alternative(message) or finish != 'stop':
                raise GatewayContentError('Gateway returned alternative or incomplete output')
            if not content.strip():
                raise EmptyGatewayResponseError('Gateway completed without answer text')
            result = dict(content=content,reasoning_content='',finish_reason=finish,usage=value.get('usage'),
                elapsed_s=time.perf_counter()-started,ttft_s=None,response_model=value.get('model',model),
                request_sha256=request.sha256,gpu_wait_s=0,local_extractive_fallbacks=0)
            save(folder/f'{ordinal:05d}.response.json',result)
            return result
        except Exception as exc:
            exc.request_sha256 = request.sha256
            save(folder/f'{ordinal:05d}.error.json',dict(error_type=type(exc).__name__,
                request_sha256=request.sha256,elapsed_s=time.perf_counter()-started))
            raise

    def close(self):
        if self.remote is not None:
            self.remote.close()
            self.remote = None
