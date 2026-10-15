"""Domain-balanced batch sampling.

A training set pooled from several databases (or languages) is usually very
uneven: with plain shuffling the biggest database makes up most of every
batch and dominates the gradients. `DomainBalancedBatchSampler` draws the
same number of samples from every domain for every batch (the usual
round-robin strategy of domain generalization).

It only needs one domain value per training row, so any model whose
training data loader is a `torch.utils.data.DataLoader` can use it (see
`Model._make_train_loader()`).
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler


class DomainBalancedBatchSampler(Sampler):
    """Yields batches (lists of positions in the dataset) that hold the same
    number of samples of each domain.

    An epoch has as many samples as the dataset (rounded up to full
    batches), so it costs what a plain epoch costs: the samples of a domain
    that is bigger than its share are drawn without repetition and, as
    every epoch draws anew, are all seen over the epochs; the samples of a
    smaller domain are repeated (every one equally often, up to one).

    Use it as `DataLoader(dataset, batch_sampler=sampler)`; `batch_sampler`
    excludes the loader's own `batch_size`, `shuffle` and `sampler`.

    domains: one domain value per dataset row (same order as the dataset).
    batch_size: rounded down to a multiple of the number of domains.
    seed: draw the same batches every run (epoch `i` uses the stream
        `[seed, i]`); None takes the seed of every epoch from torch's random
        state, so `torch.manual_seed()` makes runs repeatable.
    """

    def __init__(self, domains, batch_size, seed=None):
        domains = pd.Series(np.asarray(domains))
        if domains.isna().any():
            raise ValueError(
                f"{int(domains.isna().sum())} rows have no domain value; "
                "domain-balanced sampling needs one for every training row"
            )
        names = sorted(domains.unique().tolist(), key=str)
        if len(names) < 2:
            raise ValueError(
                f"domain-balanced sampling needs at least 2 domains, got {names}"
            )
        if batch_size < len(names):
            raise ValueError(
                f"batch size {batch_size} is smaller than the number of "
                f"domains ({len(names)}): a batch cannot hold all of them"
            )
        self.domains = names
        self.domain_indices = [
            np.flatnonzero((domains == name).to_numpy()) for name in names
        ]
        self.per_domain = batch_size // len(names)
        self.batch_size = self.per_domain * len(names)
        self.n_batches = -(-len(domains) // self.batch_size)  # ceil
        self._seed = seed
        self._epoch = 0

    def __len__(self):
        return self.n_batches

    def _rng(self):
        if self._seed is None:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
            return np.random.default_rng(seed)
        rng = np.random.default_rng([self._seed, self._epoch])
        self._epoch += 1
        return rng

    @staticmethod
    def _draw(indices, length, rng):
        """`length` positions from `indices`: whole passes over them, each in
        its own random order, cut to `length`."""
        passes = -(-length // len(indices))
        order = np.concatenate([rng.permutation(indices) for _ in range(passes)])
        return order[:length]

    def __iter__(self):
        rng = self._rng()
        needed = self.n_batches * self.per_domain
        order = [self._draw(idx, needed, rng) for idx in self.domain_indices]
        for b in range(self.n_batches):
            batch = np.concatenate(
                [o[b * self.per_domain : (b + 1) * self.per_domain] for o in order]
            )
            rng.shuffle(batch)
            yield batch.tolist()
