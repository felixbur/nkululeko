#!/usr/bin/env python3
"""
Create speaker-disjoint train/test splits for ITW dataset
to prevent speaker-based shortcuts in deepfake detection.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Read the full dataset
data_dir = Path(__file__).parent
df = pd.read_csv(data_dir / 'itw.csv')

print(f"Total samples: {len(df)}")
print(f"Speakers: {df['speaker'].nunique()}")
print(f"\nLabel distribution:")
print(df['label'].value_counts())

# Get all unique speakers
speakers = df['speaker'].unique()
np.random.seed(42)
np.random.shuffle(speakers)

# Speaker-disjoint split: 60% train, 20% dev, 20% test
n_speakers = len(speakers)
n_train = int(0.6 * n_speakers)
n_dev = int(0.2 * n_speakers)

train_speakers = speakers[:n_train]
dev_speakers = speakers[n_train:n_train+n_dev]
test_speakers = speakers[n_train+n_dev:]

# Create splits
train_df = df[df['speaker'].isin(train_speakers)]
dev_df = df[df['speaker'].isin(dev_speakers)]
test_df = df[df['speaker'].isin(test_speakers)]

# Save splits
train_df.to_csv(data_dir / 'itw_train_speaker_split.csv', index=False)
dev_df.to_csv(data_dir / 'itw_dev_speaker_split.csv', index=False)
test_df.to_csv(data_dir / 'itw_test_speaker_split.csv', index=False)

print(f"\n✓ Speaker-disjoint split created:")
print(f"  Train: {len(train_speakers)} speakers, {len(train_df)} samples")
print(f"    Real: {(train_df['label']=='real').sum()}, Fake: {(train_df['label']=='fake').sum()}")
print(f"  Dev: {len(dev_speakers)} speakers, {len(dev_df)} samples")
print(f"    Real: {(dev_df['label']=='real').sum()}, Fake: {(dev_df['label']=='fake').sum()}")
print(f"  Test: {len(test_speakers)} speakers, {len(test_df)} samples")
print(f"    Real: {(test_df['label']=='real').sum()}, Fake: {(test_df['label']=='fake').sum()}")

print(f"\n✓ Zero speaker overlap verified:")
print(f"  Train ∩ Test: {set(train_speakers) & set(test_speakers)}")
print(f"  Train ∩ Dev: {set(train_speakers) & set(dev_speakers)}")
print(f"  Dev ∩ Test: {set(dev_speakers) & set(test_speakers)}")
