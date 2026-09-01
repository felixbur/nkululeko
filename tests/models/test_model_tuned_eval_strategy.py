"""Unit tests for TunedModel._eval_strategy_key (nkululeko/models/model_tuned.py).

Regression coverage: transformers renamed the TrainingArguments kwarg
evaluation_strategy -> eval_strategy at some point. Our locally-pinned
transformers==4.40.1 only has the old name; a newer version may only have
the new one (or drop the old one entirely). The code must detect which one
the installed version actually accepts rather than hardcoding either -
hardcoding eval_strategy crashed real finetuning runs against 4.40.1 with
`TypeError: TrainingArguments.__init__() got an unexpected keyword argument
'eval_strategy'`.
"""

import inspect
from unittest.mock import patch

import transformers

from nkululeko.models.model_tuned import TunedModel


class TestEvalStrategyKey:
    def test_matches_actually_installed_transformers(self):
        params = inspect.signature(transformers.TrainingArguments.__init__).parameters
        expected = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"
        assert TunedModel._eval_strategy_key() == expected

    def test_prefers_eval_strategy_when_available(self):
        fake_sig = inspect.Signature(
            parameters=[
                inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                inspect.Parameter("eval_strategy", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            ]
        )
        with patch.object(inspect, "signature", return_value=fake_sig):
            assert TunedModel._eval_strategy_key() == "eval_strategy"

    def test_falls_back_to_evaluation_strategy_when_new_name_absent(self):
        fake_sig = inspect.Signature(
            parameters=[
                inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                inspect.Parameter("evaluation_strategy", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            ]
        )
        with patch.object(inspect, "signature", return_value=fake_sig):
            assert TunedModel._eval_strategy_key() == "evaluation_strategy"
