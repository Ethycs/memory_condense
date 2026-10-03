"""The model I/O boundary: capture input, recall, invoke, capture output."""
from dataclasses import asdict
import json

from memory_condense.application.chat_session import ChatEvent
from memory_condense.application.inline_memory import InlineMemoryResponse


class ChatIO:
    def __init__(self, session):
        self.session = session

    def exchange(self, event, *, request_id, reader, query=None):
        """Reader receives a packet; input/output capture cannot be bypassed.

        A reader returns a string, an InlineMemoryResponse, or a dictionary with
        a content string (which may serialize tool calls). Tool execution uses tool_result below. The
        reader callback owns provider-specific transport/stream assembly.
        """
        with self.session._io_lock, self.session.capture_exchange():
            self.session.ingest(event)
            packet = self.session.recall(query or event.text, packet_id=request_id,
                                         input_event_id=event.event_id)
            response = self.invoke(request_id=request_id, reader=lambda: reader(packet),
                                   input_event_id=event.event_id, packet_id=packet.packet_id)
            return dict(response=response, packet=asdict(packet))

    def invoke(self, *, request_id, reader, input_event_id, packet_id=None, packet_ids=()):
        """Capture provider output at the return boundary, including failures.

        Also used when a caller has already composed its bounded working window.
        The input and any packet must already be durably captured in this session.
        """
        with self.session._io_lock, self.session.capture_exchange():
            return self._invoke(request_id=request_id, reader=reader, input_event_id=input_event_id,
                                packet_id=packet_id, packet_ids=packet_ids)

    def _invoke(self, *, request_id, reader, input_event_id, packet_id, packet_ids):
        input_event = self.session.event(input_event_id)
        ids = list(dict.fromkeys(([packet_id] if packet_id is not None else []) + list(packet_ids)))
        packets = [self.session.packet(value) for value in ids]
        def record_use():
            # Co-access is the existing Hebbian signal. A completed exchange
            # confirms delivery/use, not correctness or comparative importance.
            for packet in packets:
                if packet.references:
                    self.session.feedback(packet.packet_id, successful=True)
        metadata = {'input_event_id': input_event_id, 'packet_id': packet_id, 'packet_ids': ids}
        # Retrying an acknowledged request returns its captured output without
        # issuing a second generation. Unknown provider outcomes remain explicit.
        output_id = request_id + ':assistant'
        try:
            prior = self.session.event(output_id)
        except KeyError:
            prior = None
        if prior is not None:
            if prior.metadata.get('io') != metadata:
                raise ValueError('Request identity reused with different input/packet')
            record_use()
            return prior.metadata['response']
        try:
            response = reader()
            internal = {}
            if isinstance(response, InlineMemoryResponse):
                internal = response.capture(input_event, output_id, ids)
                response = response.response
            text = response if isinstance(response, str) else response['content']
            if not isinstance(text, str) or not text.strip():
                raise ValueError('Provider returned no completed content')
            self.session.ingest(ChatEvent(output_id, 'assistant', text,
                metadata={'io': metadata, 'response': response, **internal}))
            record_use()
            return response
        except Exception as exc:
            # Different retry errors are separate events; request provenance is
            # retained. The original exception remains visible to the caller.
            from uuid import uuid4
            self.session.ingest(ChatEvent(request_id + ':error:' + uuid4().hex, 'tool',
                json.dumps({'error': type(exc).__name__, 'message': str(exc)}, ensure_ascii=False),
                metadata={'io': metadata}))
            raise

    def tool_result(self, *, event_id, text, call_event_id):
        if self.session.event(call_event_id).role != 'assistant':
            raise ValueError('Tool result requires a captured assistant call')
        return self.session.ingest(ChatEvent(event_id, 'tool', text,
                                            metadata={'call_event_id': call_event_id}))
