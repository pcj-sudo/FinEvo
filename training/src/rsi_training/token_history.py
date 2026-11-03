"""Preserve sampled token prefixes across unchanged agent message histories."""
from contextvars import ContextVar
from copy import deepcopy

_session = ContextVar('rsi_token_history_session', default=None)


class TokenHistoryTokenizer:
    def __init__(self, tokenizer, record=None):
        self.tokenizer = tokenizer
        self.histories = {}
        self.record = record

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def apply_chat_template(self, messages, **kwargs):
        return self._apply_chat_template(messages, record=True, **kwargs)

    def count_chat_tokens(self, messages, **kwargs):
        """Preview the actual next prompt without changing history or metrics."""
        return len(self._apply_chat_template(messages, record=False, **kwargs))

    def _apply_chat_template(self, messages, *, record, **kwargs):
        sid = _session.get()
        if sid is None or not kwargs.get('tokenize', True):
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        render_kwargs = {**kwargs, 'tokenize': False, 'preserve_thinking': True}
        rendered = self.tokenizer.apply_chat_template(messages, **render_kwargs)
        cached = self.histories.get(sid)
        reason = 'new_history'
        prefix = None
        if cached is not None:
            old_messages, old_tools, tokens = cached
            if messages[:len(old_messages)] == old_messages and kwargs.get('tools') == old_tools:
                # Dict equality ignores key order, but Qwen renders tool arguments
                # in insertion order. The wire protocol may sort those keys.
                # Render both sides using this request's representation of the
                # already-verified equal history. Keep the original sampled token
                # prefix; this rendering is only used to locate the new suffix.
                old_render = self.tokenizer.apply_chat_template(messages[:len(old_messages)], **{
                    **render_kwargs, 'add_generation_prompt': False})
                # Do not reinterpret partial/unterminated generations as full messages.
                raw = self.tokenizer.decode(tokens, skip_special_tokens=False)
                if not raw.endswith(('<|im_end|>', '<|im_end|>\n')):
                    reason = 'unterminated_output'
                elif not rendered.startswith(old_render):
                    reason = 'template_prefix_changed'
                else:
                    # Template message terminators have one newline. Generated
                    # stop tokens normally end at im_end; append that newline once.
                    suffix = ('' if raw.endswith('\n') else '\n') + rendered[len(old_render):]
                    prefix = list(tokens) + self.tokenizer.encode(suffix, add_special_tokens=False)
                    reason = 'exact_sampled_prefix'
            else:
                reason = 'messages_or_tools_changed'
        if prefix is None:
            prefix = self.tokenizer.encode(rendered, add_special_tokens=False)
        if record and self.record:
            self.record({'mode': reason, 'input_tokens': len(prefix),
                         'reused_prefix_tokens': len(cached[2]) if reason == 'exact_sampled_prefix' else 0})
        return prefix

    def remember(self, sid, messages, tools, reply, turn):
        self.histories[sid] = (deepcopy(messages + [reply]), deepcopy(tools),
                               list(turn.prompt_ids) + list(turn.output_ids))

    def drop(self, sid):
        self.histories.pop(sid, None)
