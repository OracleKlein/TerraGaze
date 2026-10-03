from contextlib import contextmanager


@contextmanager
def use_system_prompt(model, system_prompt):
    """Scope a dataset system prompt to one synchronous model.chat call."""
    if system_prompt is None:
        yield
        return
    original = model.system_message
    model.system_message = system_prompt
    try:
        yield
    finally:
        model.system_message = original
