# Copyright (c) 2022-2026, RTE (http://www.rte-france.com)
# See AUTHORS.md
# All rights reserved.
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of the oasis project.
import unittest
import tempfile
import shutil
import json
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sources")))
from data_management import (
    create_output_folder,
    distribute_seconds_within_minute,
    is_channel_empty,
    has_too_low_max,
    has_too_low_diff,
    has_quantif_problem,
    too_many_na,
    not_enough_samples,
    too_many_consecutive_na,
    is_channel_to_remove,
    is_scada_data_acceptable,
    detrend,
    differencing_method,
    median_filter_method,
    subtract_mean,
    split_windows,
    read_and_format,
)


class DummySettings:
    def get_min_output_threshold(self): return 10
    def get_min_diff_threshold(self): return 5
    def get_min_number_different_values(self): return 10
    def get_na_threshold(self): return 0.2
    def get_min_nb_samples_osc(self): return 30
    def get_min_nb_samples_amb(self): return 30
    def get_max_consecutive_na(self): return 1


class DummyLogger:
    def __init__(self):
        self.messages = []
    def warning(self, msg):
        self.messages.append(msg)
    def info(self, msg):
        self.messages.append(msg)


class TestDataManagement(unittest.TestCase):

    def setUp(self):
        self.settings = DummySettings()
        self.logger = DummyLogger()

    def test_create_output_folder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = os.path.join(tmpdir, "dummy_folder")
            self.assertFalse(os.path.exists(folder))
            result = create_output_folder(folder)
            self.assertTrue(os.path.exists(folder))
            self.assertEqual(result, folder)

    def test_distribute_seconds_within_minute(self):
        times = pd.Series([datetime(1994, 1, 1, 12, 0, 0)] * 6)
        distributed = distribute_seconds_within_minute(times)
        self.assertEqual(len(distributed), 6)
        self.assertEqual(distributed[0].second, 0)
        self.assertTrue(distributed.iloc[-1].second == 50)

    def test_is_channel_empty(self):
        self.assertTrue(is_channel_empty(""))
        self.assertFalse(is_channel_empty("foo"))

    def test_has_too_low_max(self):
        a = np.array([1, 4, 9])
        res, val = has_too_low_max(a, self.settings)
        self.assertTrue(res)
        self.assertEqual(val, 9)
        b = np.array([11, 12, 13])
        res, val = has_too_low_max(b, self.settings)
        self.assertFalse(res)

    def test_has_too_low_diff(self):
        a = np.array([5, 7, 9])
        res, val = has_too_low_diff(a, self.settings)
        self.assertTrue(res)
        b = np.array([1, 10])
        res, val = has_too_low_diff(b, self.settings)
        self.assertFalse(res)

    def test_has_quantif_problem(self):
        a = np.array([1, 2, 3])
        res, val = has_quantif_problem(a, self.settings)
        self.assertTrue(res)
        b = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        res, val = has_quantif_problem(b, self.settings)
        self.assertFalse(res)

    def test_too_many_na(self):
        a = np.array([1, 2, np.nan, np.nan])
        res, val = too_many_na(a, self.settings)
        self.assertTrue(res)
        b = np.array([1, 2, 3, 4, 5, 6, 7, np.nan])
        res, val = too_many_na(b, self.settings)
        self.assertFalse(res)

    def test_not_enough_samples(self):
        s = pd.Series([1.0, 2.0, np.nan])
        res, val = not_enough_samples(s, 3)
        self.assertTrue(res)
        res, val = not_enough_samples(s, 2)
        self.assertFalse(res)

    def test_too_many_consecutive_na(self):
        a = np.array([1, np.nan, np.nan, 4])
        res, val = too_many_consecutive_na(a, self.settings)
        self.assertTrue(res)
        b = np.array([1, np.nan, 3, 5])
        res, val = too_many_consecutive_na(b, self.settings)
        self.assertFalse(res)

    def test_is_channel_to_remove_all_ok(self):
        col_values = np.array(range(100))
        col_values_amb = pd.Series(col_values[0:40])
        col_values_osc = pd.Series(col_values[40:100])
        self.assertFalse(is_channel_to_remove("ch1", col_values,
                                              col_values_amb, col_values_osc,
                                              self.settings, self.logger))
        self.assertEqual(len(self.logger.messages), 0)

    def test_is_channel_to_remove_fails(self):
        # Fail on empty channel
        self.assertTrue(
            is_channel_to_remove("", np.array([1]), pd.Series([1]), pd.Series([1]), self.settings, self.logger))
        self.assertIn("no id", self.logger.messages[-1])
        # Fail on too low max
        self.assertTrue(
            is_channel_to_remove("x", np.array([1, 2, 3]), pd.Series([1, 2, 3]), pd.Series([1, 2, 3]), self.settings,
                                 self.logger))
        self.assertIn("too small", self.logger.messages[-1])
        # Fail on diff
        self.assertTrue(is_channel_to_remove("x", np.array([10, 12, 13, 14]), pd.Series([10, 12, 13, 14]),
                                             pd.Series([10, 12, 13, 14]), self.settings, self.logger))

    def test_is_scada_data_acceptable(self):
        df = pd.DataFrame({"a": [1,2,3]})
        self.assertTrue(is_scada_data_acceptable(df))
        df = pd.DataFrame()
        self.assertFalse(is_scada_data_acceptable(df))

    def test_detrend_differencing(self):
        raw_df = pd.DataFrame({"x":[1,2,4,8], "y":[2,4,6,10]})
        dt = detrend(raw_df, detrending_method=0)
        self.assertTrue(isinstance(dt, pd.DataFrame))
        self.assertEqual(dt.shape, raw_df.shape)
        self.assertEqual(dt.iloc[0,0], raw_df.iloc[0,0] - raw_df.iloc[1,0])

    def test_detrend_median(self):
        raw_df = pd.DataFrame({"x":[1,2,3],"y":[4,5,6]})
        dt = detrend(raw_df, detrending_method=1, median_filter_order=1)
        self.assertEqual(dt.shape, raw_df.shape)
        self.assertTrue(np.allclose(dt.values, 0))

    def test_subtract_mean(self):
        df = pd.DataFrame({"a":[1,2,3]})
        sub = subtract_mean(df)
        self.assertTrue(np.allclose(sub.mean(), 0))

    def test_split_windows(self):
        idx = pd.date_range("2009-08-28 00:00:00", freq="10s", periods=181)
        df = pd.DataFrame({"a": np.arange(1, len(idx) + 1)}, index=idx)
        osc_start = idx[60]
        osc_end = idx[120]
        amb, osc = split_windows(df, osc_start, osc_end)
        expected_osc_idx = idx[60:121]
        expected_amb_idx = idx[:60].tolist() + idx[121:].tolist()
        self.assertTrue((osc.index == expected_osc_idx).all())
        self.assertTrue(all(i in amb.index for i in expected_amb_idx))
        self.assertAlmostEqual(osc.mean().values[0], 0)

    def test_read_and_format_valid_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test.csv")
            df = pd.DataFrame(
                {
                    "time": ["1995-10-30 00:00:00", "1995-10-30 00:01:00"],
                    "channel1":[1,2]
                }
            )
            df.to_csv(csv_path, index=False)
            test_case_data = {
                "osc_start": "1995-10-30 00:00:00",
                "osc_end": "1995-10-30 00:01:00",
                "scada_file": csv_path,
            }
            scada, osc_start, osc_end = read_and_format(test_case_data)
            self.assertEqual(scada.shape[0], 2)
            self.assertEqual(scada.shape[1], 1)  # just channel1, time is index
            self.assertEqual(list(scada.index), [
                datetime(1995,10,30,0,0,0),
                datetime(1995,10,30,0,1,0)
            ])
            self.assertEqual(list(scada.columns), ["channel1"])
            self.assertEqual(osc_start, datetime(1995,10,30,0,0,0))
            self.assertEqual(osc_end, datetime(1995,10,30,0,1,0))

    def test_read_and_format_invalid_date(self):
        test_case_data = {
            "osc_start": "not-a-date",
            "osc_end": "1991-01-01 00:01:00",
            "scada_file": "no.csv",
        }
        with self.assertRaises(ValueError):
            read_and_format(test_case_data)

    def test_read_and_format_invalid_file(self):
        test_case_data = {
            "osc_start": "1991-01-01 00:00:00",
            "osc_end": "1991-01-01 00:01:00",
            "scada_file": "non_existent_file.csv",
        }
        with self.assertRaises(FileNotFoundError):
            read_and_format(test_case_data)


if __name__ == "__main__":
    unittest.main()
