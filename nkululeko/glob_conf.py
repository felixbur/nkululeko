"""Compatibility facade for experiment-scoped shared state.

New code should receive an :class:`ExperimentContext` explicitly.  The module
attributes remain available while existing components are migrated.
"""

import sys
import types

from nkululeko.experiment_context import (
    ExperimentContext,
    bind_experiment_context,
    get_context,
    set_context,
    use_context,
)


_STATE_FIELDS = frozenset(
    {
        "config",
        "label_encoder",
        "util",
        "module",
        "report",
        "labels",
        "target",
        "split3",
        "got_speaker",
    }
)

__all__ = [
    "ExperimentContext",
    "bind_experiment_context",
    "get_context",
    "init_config",
    "set_context",
    "set_got_speaker",
    "set_label_encoder",
    "set_labels",
    "set_module",
    "set_report",
    "set_split3",
    "set_target",
    "set_util",
    "use_context",
    "config",
    "label_encoder",
    "util",
    "module",
    "report",
    "labels",
    "target",
    "split3",
    "got_speaker",
]


def init_config(config_obj):
    get_context().config = config_obj


def set_label_encoder(encoder):
    get_context().label_encoder = encoder


def set_util(util_obj):
    get_context().util = util_obj


def set_module(module_obj):
    get_context().module = module_obj


def set_report(report_obj):
    get_context().report = report_obj


def set_labels(labels_obj):
    get_context().labels = labels_obj


def set_target(target_obj):
    get_context().target = target_obj


def set_split3(split3_obj):
    get_context().split3 = split3_obj


def set_got_speaker(got_speaker_obj):
    get_context().got_speaker = got_speaker_obj


class _ContextAwareModule(types.ModuleType):
    """Route legacy module attributes to the active experiment context."""

    def __getattribute__(self, name):
        if name in _STATE_FIELDS:
            return getattr(get_context(), name)
        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name in _STATE_FIELDS:
            setattr(get_context(), name, value)
            return
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _ContextAwareModule
