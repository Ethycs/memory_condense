"""Resume the frozen engineering replay after acknowledged transport failures.

Successful calls and unacknowledged calls are never resent. Failed responses stay
in the journal; each retry has a distinct, deterministic request identity.
"""
import argparse
from pathlib import Path

from tools import native_spine_engineering_session as session
from tools import native_spine_engineering_tool_cycle as cycle


RETRYABLE = frozenset({'InternalServerError', 'RateLimitError'})
MAX_RETRIES = 2
OriginalGateway = session.Gateway


class RecoveringGateway(OriginalGateway):
    def call(self, kind, messages, *, max_tokens=4096, typed_request=None, attempt=0, nonce=None):
        original = {'kind': kind, 'messages': messages, 'max_tokens': max_tokens,
                    'typed_request': typed_request, 'attempt': attempt, 'nonce': nonce}
        original_key = session.identity_sha256(original)
        current_nonce = nonce
        for retry in range(MAX_RETRIES + 1):
            job = dict(original, nonce=current_nonce)
            key = session.identity_sha256(job)
            try:
                return super().call(kind, messages, max_tokens=max_tokens,
                                    typed_request=typed_request, attempt=attempt, nonce=current_nonce)
            except RuntimeError:
                request = session.old.load(self.root / 'gateway' / f'{key}.request.json')
                response = session.old.load(self.root / 'gateway' / f'{key}.response.json')
                result = response.payload
                if (request.payload != job or result.get('request_sha256') != request.sha256
                        or result.get('error_type') not in RETRYABLE or retry == MAX_RETRIES):
                    raise
                current_nonce = {'transport_retry': retry + 1, 'original_nonce': nonce,
                                 'original_job_sha256': original_key}
                next_key = session.identity_sha256(dict(original, nonce=current_nonce))
                session.save(self.root / 'transport-retries' / f'{key}.json', {
                    'failed_request_sha256': request.sha256,
                    'failed_response_sha256': response.sha256,
                    'error_type': result['error_type'], 'retry_job_sha256': next_key,
                    'original_job_sha256': original_key, 'retry_number': retry + 1,
                    'messages_unchanged': True})


def run(root):
    if (root / 'STOP').exists():
        raise ValueError('reconcile saved actions and gateway state before clearing STOP')
    session.save(root / 'transport-resume.json', {
        'implementation_sha256': session.evaluation.digest(__file__),
        'cycle_implementation_sha256': session.evaluation.digest(cycle.__file__),
        'max_retries_per_request': MAX_RETRIES, 'retryable_errors': sorted(RETRYABLE),
        'unacknowledged_calls_retried': False, 'successful_calls_retried': False,
        'actor_context_policy_changed': False})
    session.Gateway = RecoveringGateway
    cycle.run(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root.resolve())
