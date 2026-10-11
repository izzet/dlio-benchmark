"""DLIO-PyTorch HDF5 adapter: enclosed-read residual, before process reduction.

The residual includes HDF5 bookkeeping/allocation as well as inflate. It is a
cross-layer service measurement, not an assertion of pure codec CPU time.
Original POSIX rows remain untouched. Unmatched/incomplete spans fail closed.
"""
import math
from bisect import bisect_left

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger()
SPAN_NAMES = ('hdf5_sample', 'data.hdf5_sample')


def prepare_hdf5_decode(traces, *, time_resolution):
    spans = (traces['cat'].eq('data') & traces['func_name'].isin(SPAN_NAMES))
    if not spans.any():
        return traces
    result = traces.copy()
    if not math.isfinite(time_resolution) or time_resolution <= 0:
        raise ValueError('time_resolution must convert timestamp ticks to seconds')
    required = {'pid', 'tid', 'time_start', 'time_end', 'time', 'io_cat'}
    if not required.issubset(result.columns):
        logger.warning('dlio.hdf5_decode.missing_span_schema', missing=sorted(required-set(result.columns)))
        return result.loc[~spans].copy()
    reads = result.loc[result['cat'].str.startswith('posix', na=False) & result['io_cat'].eq(1)]
    identities = ['pid', 'tid']
    if 'host_hash' in result.columns:
        identities.append('host_hash')
    grouped = {}
    for key, group in reads.groupby(identities, dropna=False):
        group = group.sort_values('time_start')
        starts, ends = group['time_start'].to_numpy(float), group['time_end'].to_numpy(float)
        grouped[key] = (starts, ends, np.maximum.accumulate(ends))
    peer_bounds = {}
    for _, group in result.loc[spans].groupby(identities, dropna=False):
        group = group.sort_values('time_start')
        indices = list(group.index)
        starts, ends = group['time_start'].to_numpy(float), group['time_end'].to_numpy(float)
        for i, index in enumerate(indices):
            peer_bounds[index] = ((i == 0 or ends[i-1] <= starts[i])
                                  and (i+1 == len(indices) or starts[i+1] >= ends[i]))
    rejected, derived = [], {}
    for index, span in result.loc[spans].iterrows():
        start, end, duration = span['time_start'], span['time_end'], span['time']
        if not all(pd.notna(v) and math.isfinite(float(v)) for v in (start, end, duration)) or end < start or duration < 0:
            rejected.append(index)
            continue
        identity = tuple(span[key] for key in identities)
        # Nested/overlapping sample spans would double-own reads; never guess.
        if any(pd.isna(v) for v in identity) or not peer_bounds.get(index, False) or identity not in grouped:
            rejected.append(index)
            continue
        starts, ends, prefix_ends = grouped[identity]
        left_index, right_index = bisect_left(starts, start), bisect_left(starts, end)
        # Positive coverage is required; missing POSIX traces are not measured zero.
        if (left_index == right_index or (left_index and prefix_ends[left_index-1] > start)
                or np.any(ends[left_index:right_index] > end)):
            rejected.append(index)
            continue
        intervals = list(zip(starts[left_index:right_index], ends[left_index:right_index]))
        if any(not (pd.notna(a) and pd.notna(b) and math.isfinite(float(a)) and math.isfinite(float(b)) and b >= a)
               for a, b in intervals):
            rejected.append(index)
            continue
        union, left, right = 0.0, *intervals[0]
        for a, b in intervals[1:]:
            if a <= right:
                right = max(right, b)
            else:
                union += right-left
                left, right = a, b
        union += right-left
        union /= time_resolution  # normalized timestamp bounds remain ticks; durations are seconds.
        if union > duration:
            rejected.append(index)
            continue
        derived[index] = (duration, union, len(intervals), duration-union)
    if derived:
        values = pd.DataFrame.from_dict(derived, orient='index', columns=[
            'dlio_hdf5_inclusive_time', 'dlio_hdf5_enclosed_posix_read_time',
            'dlio_hdf5_enclosed_posix_read_count', 'time'])
        for column in values.columns:
            result.loc[values.index, column] = values[column]
        result.loc[values.index, 'func_name'] = 'data.decode'
    if rejected:
        logger.warning('dlio.hdf5_decode.unmatched_spans', spans=len(rejected))
        result = result.drop(index=rejected)
    return result
