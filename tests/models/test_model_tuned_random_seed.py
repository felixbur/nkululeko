"""Unit tests for TunedModel._random_seed (nkululeko/models/model_tuned.py).

Regression: transformers.TrainingArguments defaults `seed` to a hardcoded
42, and train() never overrode it - so every [EXP] runs iteration for a
finetune model silently reused the exact same weight init and data
shuffling order, producing byte-identical results across "different" runs.
Confirmed empirically: an experiment configured with runs=3 reported the
same score three times, defeating the entire point of averaging over
multiple runs. _random_seed() draws a fresh value each call so every run
actually gets an independent initialization.
"""

from nkululeko.models.model_tuned import TunedModel


class TestRandomSeed:
    def test_returns_an_int_in_range(self):
        seed = TunedModel._random_seed()
        assert isinstance(seed, int)
        assert 0 <= seed <= 2**31 - 1

    def test_consecutive_calls_differ(self):
        # Astronomically unlikely to collide by chance across a 2^31 range -
        # a repeat here would mean the source stopped being random, not bad
        # luck.
        seeds = {TunedModel._random_seed() for _ in range(20)}
        assert len(seeds) == 20

    def test_not_hardcoded_to_transformers_default(self):
        seeds = [TunedModel._random_seed() for _ in range(20)]
        assert not all(s == 42 for s in seeds)
