import unittest

import numpy as np
import pandas as pd

from data_provider.load_dataset import (
    _split_by_ratio_only,
    load_dataset,
    split_by_ratio,
)


def _segmented_frame(sizes):
    """Build a segment-ordered table like load_dataset produces for data_versions."""
    frames, timestamps, segments = [], [], []
    start = pd.Timestamp('2026-05-13 00:00')
    for segment, size in enumerate(sizes):
        times = pd.date_range(start, periods=size, freq='min')
        frames.append(pd.DataFrame({'x': np.arange(size, dtype=float) + 100 * segment}))
        timestamps.append(pd.Series(times))
        segments.append(np.full(size, segment))
        start = times[-1] + pd.Timedelta(days=10)  # gap between versions
    encoded_df = pd.concat(frames, ignore_index=True)
    return (
        encoded_df,
        pd.concat(timestamps, ignore_index=True),
        pd.Series(np.concatenate(segments)),
    )


class ForecastSegmentSplitTest(unittest.TestCase):
    split_rate = [0.8, 0.1, 0.1]

    def _split(self, sizes, segment_split, splitter='ratio_only'):
        encoded_df, timestamps, segment_ids = _segmented_frame(sizes)
        if splitter == 'ratio_only':
            return _split_by_ratio_only(
                encoded_df, timestamps, self.split_rate,
                timestamps.min(), timestamps.max(), segment_ids,
                segment_split=segment_split,
            )
        return split_by_ratio(
            encoded_df, timestamps, self.split_rate, segment_ids,
            segment_split=segment_split,
        )

    def test_global_split_puts_test_only_in_last_segment(self):
        data_dict = self._split([10, 20, 10], 'global')
        self.assertEqual([len(data_dict[s]['data']) for s in ('train', 'val', 'test')], [32, 4, 4])
        self.assertEqual(set(data_dict['test']['segment_ids'].tolist()), {2})

    def test_per_segment_split_takes_each_segment_tail(self):
        for splitter in ('ratio_only', 'ratio'):
            data_dict = self._split([10, 20, 10], 'per_segment', splitter)
            test_segments = data_dict['test']['segment_ids']
            self.assertEqual([len(data_dict[s]['data']) for s in ('train', 'val', 'test')], [32, 4, 4])
            self.assertEqual(test_segments.tolist(), [0, 1, 1, 2])
            # Each test piece is the chronological tail of its own segment.
            self.assertEqual(data_dict['test']['data'][:, 0].tolist(), [9.0, 118.0, 119.0, 209.0])
            self.assertEqual(data_dict['val']['data'][:, 0].tolist(), [8.0, 116.0, 117.0, 208.0])
            # Train/val/test never share rows.
            rows = np.concatenate([data_dict[s]['data'][:, 0] for s in ('train', 'val', 'test')])
            self.assertEqual(len(set(rows.tolist())), 40)

    def test_single_segment_split_is_identical_in_both_modes(self):
        global_split = self._split([37], 'global')
        per_segment = self._split([37], 'per_segment')
        for name in ('train', 'val', 'test'):
            np.testing.assert_array_equal(global_split[name]['data'], per_segment[name]['data'])
            np.testing.assert_array_equal(global_split[name]['timestamps'], per_segment[name]['timestamps'])

    def test_unknown_mode_and_date_split_are_rejected(self):
        with self.assertRaises(ValueError):
            self._split([10, 10], 'by_month')
        with self.assertRaisesRegex(ValueError, 'ratio splits only'):
            load_dataset(
                datadir='unused', scaler='standard', ord_categorical_feature=[],
                categorical_feature=[], cat_encode_method='label', numerical_feature=['x'],
                target_features=['x'], data_version='unused', environment_name='room1',
                segment_split='per_segment', split_date='2026-06-01 00:00',
            )


if __name__ == '__main__':
    unittest.main()
