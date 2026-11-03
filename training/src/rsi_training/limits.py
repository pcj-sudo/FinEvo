"""One token-budget contract for the trainer, policy gateway and OpenHands.

Changing these limits requires a new frozen run and a refreshed CPU agent
adapter. These are token budgets, not GPU allocation or microbatch limits.
"""

CONTEXT_TOKENS = 262144
OUTPUT_TOKENS = 16384
# Reserve the full 32,000-token response budget plus 144 tokens of headroom.
INPUT_TOKENS = 230000
CONDENSER_TOKENS = 230000


def token_limits():
    if not 0 < CONDENSER_TOKENS <= INPUT_TOKENS < CONTEXT_TOKENS:
        raise ValueError('Condenser must run at or before the input limit')
    # Input/output are independent maxima. The policy gateway already caps
    # generation at min(requested output, context window - actual prompt).
    if not 0 < OUTPUT_TOKENS < CONTEXT_TOKENS:
        raise ValueError('Invalid output budget')
    return {
        'context_length': CONTEXT_TOKENS,
        'max_response_tokens': OUTPUT_TOKENS,
        'max_input_tokens': INPUT_TOKENS,
        'condenser_max_tokens': CONDENSER_TOKENS,
    }


if __name__ == '__main__':
    print(*token_limits().values())
