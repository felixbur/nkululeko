"""Experiment-scoped runtime state.

The context is passed through Nkululeko's orchestration objects.  The
``glob_conf`` module retains a compatibility adapter for third-party code that
has not yet migrated to this API.
"""

import types
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps


@dataclass
class ExperimentContext:
    """Mutable runtime state owned by one experiment execution."""

    config: object = None
    label_encoder: object = None
    util: object = None
    module: object = None
    report: object = None
    labels: object = None
    target: object = None
    split3: bool = False
    got_speaker: bool = False


_default_context = ExperimentContext()
_current_context = ContextVar("nkululeko_experiment_context", default=_default_context)


def get_context():
    """Return the context active in this execution context."""
    return _current_context.get()


def set_context(context):
    """Make *context* active until another context is selected."""
    _current_context.set(context)


@contextmanager
def use_context(context):
    """Make *context* active for the duration of the with block."""
    token = _current_context.set(context)
    try:
        yield context
    finally:
        _current_context.reset(token)


def bind_experiment_context(cls):
    """Bind an object's methods to its ``context`` attribute.

    The binding makes nested legacy integrations observe the correct context
    while those integrations are migrated to explicit dependencies.
    """

    def bind(method):
        @wraps(method)
        def wrapped(self, *args, **kwargs):
            context = getattr(self, "context", None)
            if context is None:
                return method(self, *args, **kwargs)
            with use_context(context):
                return method(self, *args, **kwargs)

        return wrapped

    for name, value in vars(cls).items():
        if not name.startswith("__") and isinstance(value, types.FunctionType):
            setattr(cls, name, bind(value))
    return cls


class ContextAware:
    """Provide a compatibility fallback for partially initialized objects."""

    @property
    def context(self):
        context = getattr(self, "_context", None)
        if context is not None:
            return context
        util = getattr(self, "util", None)
        return getattr(util, "context", get_context())

    @context.setter
    def context(self, context):
        self._context = context
