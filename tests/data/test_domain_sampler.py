"""DomainBalancedBatchSampler (nkululeko/data/domain_sampler.py)."""

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from nkululeko.data.domain_sampler import DomainBalancedBatchSampler


def _domains(counts):
    return [name for name, n in counts.items() for _ in range(n)]


def _epoch(sampler):
    return [list(batch) for batch in sampler]


class TestValidation:
    def test_missing_domain_values_are_rejected(self):
        with pytest.raises(ValueError, match="no domain"):
            DomainBalancedBatchSampler(["a", None, "b", "b"], batch_size=2)
        with pytest.raises(ValueError, match="no domain"):
            DomainBalancedBatchSampler(pd.Series(["a", np.nan, "b"]), batch_size=2)

    def test_a_single_domain_is_rejected(self):
        with pytest.raises(ValueError, match="at least 2 domains"):
            DomainBalancedBatchSampler(["a"] * 10, batch_size=4)

    def test_a_batch_must_hold_every_domain(self):
        with pytest.raises(ValueError, match="smaller than the number"):
            DomainBalancedBatchSampler(_domains({"a": 4, "b": 4, "c": 4}), batch_size=2)

    def test_batch_size_is_rounded_down_to_a_multiple_of_the_domains(self):
        sampler = DomainBalancedBatchSampler(_domains({"a": 8, "b": 8, "c": 8}), 10)
        assert (sampler.per_domain, sampler.batch_size) == (3, 9)
        assert all(len(batch) == 9 for batch in sampler)

    def test_domains_can_be_numbers_or_mixed(self):
        sampler = DomainBalancedBatchSampler([1, 1, 2, 2, 3, 3], batch_size=3)
        assert len(sampler.domains) == 3
        assert len(_epoch(sampler)) == 2


class TestBatches:
    def test_every_batch_has_the_same_number_from_each_domain(self):
        domains = np.array(_domains({"a": 100, "b": 30, "c": 10}))
        sampler = DomainBalancedBatchSampler(domains, batch_size=12)
        batches = _epoch(sampler)
        assert len(batches) == len(sampler)
        for batch in batches:
            counts = pd.Series(domains[batch]).value_counts()
            assert counts.to_dict() == {"a": 4, "b": 4, "c": 4}

    def test_an_epoch_is_as_long_as_the_dataset(self):
        domains = _domains({"a": 1000, "b": 10})
        sampler = DomainBalancedBatchSampler(domains, batch_size=10)
        n_samples = sum(len(batch) for batch in sampler)
        assert len(sampler) == -(-1010 // 10)  # ceil: no more than a plain epoch
        assert n_samples == len(sampler) * 10 == 1010

    @pytest.mark.parametrize("n_domains", [2, 3, 5])
    @pytest.mark.parametrize("batch_size", [4, 6, 7, 32])
    @pytest.mark.parametrize("n_rows", [8, 10, 25, 101])
    def test_the_epoch_is_within_half_a_batch_of_the_dataset(
        self, n_rows, batch_size, n_domains
    ):
        if batch_size < n_domains or n_rows < n_domains:
            pytest.skip("a batch must hold every domain")
        domains = [i % n_domains for i in range(n_rows)]
        sampler = DomainBalancedBatchSampler(domains, batch_size)
        drawn = sum(len(batch) for batch in sampler)
        assert drawn == len(sampler) * sampler.batch_size
        assert len(sampler) >= 1
        assert abs(drawn - n_rows) <= sampler.batch_size / 2 or len(sampler) == 1

    def test_eight_rows_with_batches_of_six_are_one_batch_not_two(self):
        sampler = DomainBalancedBatchSampler(_domains({"a": 5, "b": 3}), batch_size=6)
        assert len(sampler) == 1
        assert [len(b) for b in sampler] == [6]

    def test_a_small_domain_is_repeated_evenly(self):
        domains = np.array(_domains({"a": 90, "b": 10}))
        sampler = DomainBalancedBatchSampler(domains, batch_size=10)
        seen = np.concatenate(_epoch(sampler))
        small = np.bincount(seen[domains[seen] == "b"], minlength=100)[90:]
        assert small.sum() == 50  # half of the epoch
        assert small.max() - small.min() <= 1  # every row about 5 times

    def test_a_big_domain_is_drawn_without_repetition(self):
        domains = np.array(_domains({"a": 900, "b": 100}))
        sampler = DomainBalancedBatchSampler(domains, batch_size=10)
        seen = np.concatenate(_epoch(sampler))
        big = seen[domains[seen] == "a"]
        assert len(big) == len(set(big.tolist())) == 500

    def test_the_big_domain_is_covered_over_several_epochs(self):
        domains = np.array(_domains({"a": 400, "b": 100}))
        sampler = DomainBalancedBatchSampler(domains, batch_size=10)
        seen = set()
        for _ in range(8):
            seen |= {i for batch in sampler for i in batch}
        assert seen == set(range(500))

    def test_batches_are_shuffled_across_domains(self):
        domains = np.array(_domains({"a": 50, "b": 50}))
        sampler = DomainBalancedBatchSampler(domains, batch_size=10)
        firsts = {domains[batch[0]] for batch in sampler}
        assert firsts == {"a", "b"}

    def test_a_dataloader_can_use_it(self):
        domains = np.array(_domains({"a": 20, "b": 4}))
        data = TensorDataset(torch.arange(24), torch.zeros(24))
        sampler = DomainBalancedBatchSampler(domains, batch_size=4)
        loader = DataLoader(data, batch_sampler=sampler)
        assert len(loader) == len(sampler) == 6
        for index, _ in loader:
            assert pd.Series(domains[index.numpy()]).value_counts().to_dict() == {
                "a": 2,
                "b": 2,
            }


class TestRandomness:
    def test_every_epoch_is_new(self):
        sampler = DomainBalancedBatchSampler(_domains({"a": 40, "b": 40}), 8)
        assert _epoch(sampler) != _epoch(sampler)

    def test_follows_the_torch_seed(self):
        domains = _domains({"a": 40, "b": 40})
        sampler = DomainBalancedBatchSampler(domains, 8)
        torch.manual_seed(1)
        first = _epoch(sampler)
        torch.manual_seed(1)
        again = _epoch(sampler)
        torch.manual_seed(2)
        other = _epoch(sampler)
        assert first == again != other

    def test_a_seed_gives_the_same_epochs_but_each_epoch_differs(self):
        domains = _domains({"a": 40, "b": 40})
        one = DomainBalancedBatchSampler(domains, 8, seed=5)
        two = DomainBalancedBatchSampler(domains, 8, seed=5)
        first, second = _epoch(one), _epoch(one)
        assert first == _epoch(two) and second == _epoch(two)
        assert first != second
