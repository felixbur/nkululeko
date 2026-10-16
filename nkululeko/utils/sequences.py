"""Frame sequences inside segments (GH #471).

A sample is a segment, indexed (file, start, end). Sequence data adds a
fourth index level, ``frame``: the frame's start time relative to the
segment start, as a Timedelta (the first frame of every segment is 0).
One row per frame, one column per feature (or label)::

    file       start  end    frame   F0     loudness
    a.wav      0 s    2.5 s  0 s     31.2   0.41
    a.wav      0 s    2.5 s  0.01 s  31.0   0.44
    ...

Per-utterance data (one row per segment, no ``frame`` level) is the
ordinary case and is left untouched by everything here.
"""

import numpy as np
import pandas as pd

FRAME_LEVEL = "frame"
SEGMENT_LEVELS = ["file", "start", "end"]
SEQUENCE_LEVELS = SEGMENT_LEVELS + [FRAME_LEVEL]
POOLING_STATS = ("mean", "std", "min", "max", "median")
# FEATS.level spellings that ask for frame sequences (case-insensitive)
FRAME_LEVEL_NAMES = ("frame", "lld", "lowleveldescriptors")


def is_frame_level(level):
    """True if a FEATS.level value asks for frame sequences."""
    return str(level).strip().lower() in FRAME_LEVEL_NAMES


def is_sequence(df):
    """True if df holds frame sequences, i.e. its index has a frame level."""
    return (
        df is not None
        and isinstance(df.index, pd.MultiIndex)
        and FRAME_LEVEL in df.index.names
    )


def sequence_index(segment, frames):
    """(file, start, end, frame) index for the frames of one segment.

    Args:
        segment: (file, start, end) tuple.
        frames: frame start times relative to the segment start, as
            Timedeltas or seconds.
    """
    file, start, end = segment
    frames = _ns(pd.to_timedelta(frames, unit="s"))
    n = len(frames)
    return pd.MultiIndex.from_arrays(
        [
            [file] * n,
            _ns(pd.to_timedelta([start] * n)),
            _ns(pd.to_timedelta([end] * n)),
            frames,
        ],
        names=SEQUENCE_LEVELS,
    )


def _ns(times):
    """Times in nanoseconds, as audformat indexes have them (pandas 3
    otherwise picks the unit from the input, e.g. microseconds)."""
    return times.astype("timedelta64[ns]")


def _group(df):
    # Keep the segments in order of first appearance; dropna=False keeps
    # segments with a NaT end.
    return df.groupby(level=[0, 1, 2], sort=False, dropna=False)


def _segment_codes(df):
    """Per row, the number of its segment in order of first appearance.
    (Used instead of the group keys, which hold NaN for a NaT end, and
    NaN never equals itself.)"""
    return _group(df).ngroup().to_numpy()


def _split_by_segment(df):
    """Row positions of each segment, frames ordered by time, segments in
    order of first appearance."""
    codes = _segment_codes(df)
    frames = df.index.get_level_values(FRAME_LEVEL).to_numpy()
    order = np.lexsort((frames, codes))
    bounds = np.flatnonzero(np.diff(codes[order])) + 1
    return np.split(order, bounds)


def segment_index(df):
    """The (file, start, end) index of df's segments, in order of first
    appearance, one entry per segment."""
    if not is_sequence(df):
        return df.index
    return df.index.droplevel(FRAME_LEVEL).unique()


def n_segments(df):
    """Number of segments (samples) in df, frames or not."""
    return len(segment_index(df))


def pool(df, stats=("mean", "std")):
    """Turn frame sequences into one row per segment.

    Each feature column becomes one column per statistic, named
    ``<feature>_<stat>``. The standard deviation is the population one
    (0 for a single-frame segment). Data without a frame level is returned
    unchanged.
    """
    if not is_sequence(df):
        return df
    unknown = [s for s in stats if s not in POOLING_STATS]
    if unknown or not stats:
        raise ValueError(
            f"unknown pooling statistics {unknown or list(stats)}; "
            f"choose from {list(POOLING_STATS)}"
        )
    grouped = _group(df)
    blocks = []
    for stat in stats:
        block = grouped.std(ddof=0) if stat == "std" else getattr(grouped, stat)()
        blocks.append(block.add_suffix(f"_{stat}"))
    pooled = pd.concat(blocks, axis=1)
    pooled.index.names = SEGMENT_LEVELS
    return pooled


def pooling_stats(util):
    """FEATS.pooling: the statistics that turn a frame sequence into one
    vector (see pool()), default mean and standard deviation."""
    import ast

    stats = util.config_val("FEATS", "pooling", "['mean', 'std']")
    if isinstance(stats, str):
        stats = ast.literal_eval(stats) if stats.startswith("[") else [stats]
    unknown = [s for s in stats if s not in POOLING_STATS]
    if unknown or not stats:
        util.error(
            f"FEATS.pooling: unknown statistics {unknown or stats}, "
            f"choose from {list(POOLING_STATS)}"
        )
    return list(stats)


def to_arrays(df, dtype=np.float32):
    """Split frame sequences into one (frames, features) array per segment.

    Returns:
        (segment index, list of arrays), both in order of first
        appearance of the segments. Frames are ordered by time.
    """
    values = df.to_numpy(dtype=dtype)
    arrays = [values[rows] for rows in _split_by_segment(df)]
    return segment_index(df), arrays


def _hashable(segment):
    """Segment key usable in a dict: NaT/NaN become None."""
    return tuple(None if pd.isna(x) else x for x in segment)


def align_frame_labels(labels, feats):
    """Bring frame-wise labels to the frames of feature sequences.

    Labels and features may have different frame rates (e.g. ratings
    every 40 ms, features every 10 ms). Each feature frame gets the label
    of the latest label frame at or before it, within the same segment;
    feature frames before the first label frame get NaN.

    Args:
        labels: frame-wise labels, index (file, start, end, frame).
        feats: frame-wise features, index (file, start, end, frame).

    Returns:
        DataFrame with labels' columns and feats' frames as index
        (frames ordered by time within each segment).
    """
    if not (is_sequence(labels) and is_sequence(feats)):
        raise ValueError("both labels and features need a frame index level")
    label_rows = dict(
        zip(
            map(_hashable, segment_index(labels)),
            _split_by_segment(labels),
        )
    )
    parts = []
    for key, rows in zip(segment_index(feats), _split_by_segment(feats)):
        feat_frames = feats.iloc[rows]
        times = feat_frames.index.get_level_values(FRAME_LEVEL)
        label_pos = label_rows.get(_hashable(key))
        if label_pos is not None:
            seg_labels = labels.iloc[label_pos].reset_index(level=FRAME_LEVEL)
            merged = pd.merge_asof(
                pd.DataFrame({FRAME_LEVEL: times}).sort_values(FRAME_LEVEL),
                seg_labels.reset_index(drop=True),
                on=FRAME_LEVEL,
                direction="backward",
            )
            merged = merged.set_index(FRAME_LEVEL).reindex(times)
            values = merged.to_numpy()
        else:
            values = np.full((len(times), labels.shape[1]), np.nan)
        parts.append(
            pd.DataFrame(values, index=feat_frames.index, columns=labels.columns)
        )
    return pd.concat(parts) if parts else labels.iloc[0:0]


def read_sequence_csv(path):
    """Read frame sequences from a CSV with columns file, start, end,
    frame (start, end and frame in seconds) and one column per feature
    or label."""
    df = pd.read_csv(path)
    missing = [c for c in SEQUENCE_LEVELS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing column(s) {missing}")
    for col in ("start", "end", "frame"):
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = _ns(pd.to_timedelta(df[col], unit="s"))
        else:
            df[col] = _ns(pd.to_timedelta(df[col]))
    return df.set_index(SEQUENCE_LEVELS)


def write_sequence_csv(df, path):
    """Write frame sequences as CSV, with start, end and frame in seconds."""
    out = df.reset_index()
    for col in ("start", "end", "frame"):
        out[col] = out[col].dt.total_seconds()
    out.to_csv(path, index=False)
