"""JSON-lines transport for the same application chat interface used by replay."""
import json

from memory_condense.application.chat_session import chat_request


def serve(session, source, destination, *, reader=None):
    """Serve a configured ChatSession; the caller owns its backend and lifetime.

    One completed message per line, including assistant tool-call payloads and
    tool results. Streaming clients submit once when a message is complete.
    """
    for line in source:
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError('Chat request must be an object')
            if request.get('operation') == 'exchange':
                if reader is None:
                    raise ValueError('No model reader is configured')
                from memory_condense.application.chat_io import ChatIO
                from memory_condense.application.chat_session import ChatEvent
                if request.get('session_id', session.session_id) != session.session_id:
                    raise ValueError('Request belongs to a different session')
                result = ChatIO(session).exchange(
                    ChatEvent(request['event_id'], request.get('role', 'user'), request['text'],
                              request.get('created_at'), request.get('metadata', {})),
                    request_id=request['request_id'], reader=reader, query=request.get('query'))
            else:
                result = chat_request(session, request)
            response = {'ok': True, 'result': result}
        except (ValueError, KeyError, TypeError, RuntimeError, OSError) as exc:
            response = {'ok': False, 'error': type(exc).__name__ + ': ' + str(exc)}
        destination.write(json.dumps(response, ensure_ascii=False) + '\n')
        destination.flush()
