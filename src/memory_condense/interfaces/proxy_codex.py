"""Codex transport identity; memory and reduction stay in MemoryProxy."""
import hashlib
import json

from memory_condense.interfaces.proxy_memory import MemoryProxy, MemoryProxyError
from memory_condense.interfaces.proxy_memory_wire import encoded
from memory_condense.interfaces.proxy_sessions import codex_metadata
from memory_condense.interfaces import proxy_memory_wire as wire


def conversation_headers(body, headers):
    """Codex transport headers for one turn, or None when it is not a turn.

    The wire parsing lives in ``proxy_sessions.codex_metadata`` so there is one
    reader of Codex's identity fields. What stays here is this path's stricter
    policy, which is deliberately not the capture path's:

    * **Strict, not permissive.** ``codex_metadata`` never raises, because the
      universal capture path must degrade to its next resolution step. This
      path runs only behind ``--codex-chatgpt`` and owns the namespace, so
      incomplete identity is an error -- better a 400 than a conversation
      silently keyed on something that a later complete request will not match.
    * **Allowlist, not denylist.** Only ``request_kind == 'turn'`` is a turn
      here, whereas ``is_conversation_turn`` admits unknown kinds by design.
      The asymmetry is intended: there, refusing an unrecognised kind would
      silently stop recording real turns; here, an unexpected kind merely
      passes through untouched to the provider.
    """
    identity = codex_metadata(headers, body) or {}
    if identity.get('request_kind') and identity['request_kind'] != 'turn':
        # Title generation and compaction are not user conversation exchanges.
        return None
    if not identity.get('thread_id') or not identity.get('turn_id'):
        raise MemoryProxyError('Codex memory requires thread_id and turn_id metadata')
    result = dict(headers)
    result['x-memory-client'] = 'codex-chatgpt'
    result['x-memory-conversation-id'] = hashlib.sha256(encoded(
        [identity['installation_id'], identity['thread_id']])).hexdigest()
    # A turn includes several tool generations. Hash the full payload, so each
    # generation is distinct and identical retries retain the same identity.
    result['x-memory-request-id'] = hashlib.sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()
    return result


class CodexMemoryProxy(MemoryProxy):
    @staticmethod
    def _new_messages(history, incoming, fmt=None):
        try:
            return MemoryProxy._new_messages(history, incoming, fmt)
        except MemoryProxyError:
            if fmt != 'responses':
                raise
            # Codex compaction prepends a summary to a retained history suffix.
            # Match that suffix explicitly; the already indexed original turns
            # remain authoritative. Never accept an unanchored divergent replay.
            old = [wire.canonical_message(wire.responses.canonical_item(item))
                   for _, message in history for item in wire.responses.input_item(message)]
            values = [wire.canonical_message(message) for message in incoming]
            # A buffered completion can be committed locally without Codex
            # recording it when the user steers the running turn. Match every
            # user/tool item exactly; only omitted prose replies are optional.
            # Reapply this on later replays so the omission survives restart.
            matched, skipped = 0, False
            for message in old:
                if matched < len(values) and message == values[matched]:
                    matched += 1
                elif message['role'] == 'assistant' and not (
                        message.get('tool_calls') or message.get('function_call')):
                    skipped = True
                else:
                    break
            else:
                if (skipped and matched and matched < len(values)
                        and all(m['role'] == 'user' for m in values[matched:])):
                    return incoming[matched:]
            for size in range(min(len(old), len(values)), 1, -1):
                suffix = old[-size:]
                if not any(m['role'] == 'user' or m.get('tool_calls') for m in suffix):
                    continue
                for offset in range(1, len(values) - size + 1):
                    if suffix == values[offset:offset + size]:
                        return incoming[offset + size:]
            raise

    @staticmethod
    def namespace(provider, headers):
        headers = {key.lower(): value for key, value in headers.items()}
        if headers.get('x-memory-client') != 'codex-chatgpt':
            return MemoryProxy.namespace(provider, headers)
        account = headers.get('chatgpt-account-id')
        conversation = headers.get('x-memory-conversation-id')
        if not account or not conversation:
            raise MemoryProxyError('Codex memory requires account and conversation identity')
        # ChatGPT access tokens rotate. The stable account ID keeps the same
        # local journal across refreshes; upstream still authenticates the token.
        return hashlib.sha256(encoded(['codex-chatgpt', account, conversation])).hexdigest()
