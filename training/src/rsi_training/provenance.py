"""Audit all responses, then exclude explicitly tagged summaries from loss."""
from array import array
from collections import Counter
from hashlib import sha256


def signature(prompt, output, logprobs):
    if len(output) != len(logprobs):
        raise ValueError('Generated token/logprob length mismatch')
    return (len(output), sha256(array('q', prompt).tobytes()).hexdigest(),
            sha256(array('q', output).tobytes() + array('d', logprobs).tobytes()).hexdigest())


class ResponseLedger:
    def __init__(self):
        self.expected = Counter()
        self.excluded = Counter()

    def record(self, turn, *, purpose='solver'):
        if purpose not in ('solver', 'summary'):
            raise ValueError('Unknown policy call purpose')
        if turn.output_ids:
            key = signature(turn.prompt_ids, turn.output_ids, turn.output_log_probs)
            self.expected[key] += 1
            if purpose == 'summary':
                self.excluded[key] += 1

    @staticmethod
    def _spans(samples):
        for sample in samples:
            n = sample.response_length
            mask, probs = sample.loss_mask, sample.rollout_log_probs
            if not 0 < n < len(sample.tokens) or len(mask) != n or len(probs) != n:
                raise ValueError('Invalid sample provenance dimensions')
            if any(m not in (0, 1) for m in mask):
                raise ValueError('Invalid loss mask')
            offset = len(sample.tokens) - n
            start = None
            for i in range(n + 1):
                enabled = i < n and mask[i] == 1
                if enabled and start is None:
                    start = i
                elif not enabled and start is not None:
                    key = signature(sample.tokens[:offset + start],
                                    sample.tokens[offset + start:offset + i], probs[start:i])
                    yield sample, start, i, key
                    start = None

    def verify(self, samples, *, expected=None):
        actual = Counter(key for _, _, _, key in self._spans(samples))
        expected = self.expected if expected is None else expected
        if actual != expected:
            raise ValueError(f'Response provenance mismatch: missing={sum((expected-actual).values())}, '
                             f'unexpected={sum((actual-expected).values())}')
        return {'verified': True, 'responses': sum(actual.values()),
                'generated_tokens': sum(k[0] * count for k, count in actual.items()),
                'segments': len(samples), 'sequence_tokens': sum(len(s.tokens) for s in samples)}

    def training_samples(self, samples):
        # First prove nothing was lost by trajectory assembly, including the
        # summaries we intentionally omit. Never hide missing solver responses.
        captured = self.verify(samples)
        remaining = self.excluded.copy()
        spans = list(self._spans(samples))
        for sample, start, end, key in spans:
            if remaining[key]:
                sample.loss_mask[start:end] = [0] * (end - start)
                remaining[key] -= 1
        if +remaining:
            raise ValueError('Missing summary responses while masking')
        kept = [sample for sample in samples if any(sample.loss_mask)]
        report = self.verify(kept, expected=self.expected - self.excluded)
        report.update(captured_responses=captured['responses'],
                      captured_generated_tokens=captured['generated_tokens'],
                      excluded_summary_responses=sum(self.excluded.values()),
                      excluded_summary_tokens=sum(k[0] * n for k, n in self.excluded.items()),
                      dropped_summary_segments=len(samples) - len(kept))
        return kept, report
